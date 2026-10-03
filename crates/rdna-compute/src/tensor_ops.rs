// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP tensor operation wrappers.
//!
//! These functions enforce the tensor layout and F32/BF16 boundaries at the
//! shared HIP launch site; architecture-specific tuning remains an explicit
//! selector rather than an admission requirement.

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::{DType, Gpu, GpuTensor};

pub(crate) const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const HYPER_READ_UP_WMMA_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_wmma.gfx1151.hip");
const HYPER_READ_UP_WMMA_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_wmma.gfx1201.hip");
const GATED_DELTA_CHUNK_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_chunk_wmma.gfx1151.hip");
pub(crate) const GATED_DELTA_CHUNK_Q8_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_chunk_q8_wmma.gfx1151.hip");

const INDEXED_ATTENTION_DENSE_WMMA_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_dense_wmma.gfx1151.hip");
const INDEXED_ATTENTION_GATHERED_WMMA_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_gathered_wmma.gfx1151.hip");
const INDEXED_ATTENTION_GATHERED_WMMA_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_gathered_wmma.gfx1201.hip");
const INDEXED_ATTENTION_SELECT_EXACT_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_select_exact.hip");
/// `HIPFIRE_QWEN4_QSA_WMMA_GATHER` (on unless `0`) routes QSA prefill
/// attention chunks that the full-window dense route does not take through
/// the gathered F16 WMMA kernels: gfx1151 on the F32 state from
/// QSA_ATTENTION_HG12_MIN_ROWS rows, gfx1201 on the fp8 state from
/// QWEN4_F16_WMMA_MIN_TOKENS rows.  Not bit-exact against the hg4
/// kernel; admitted because its error against an f64 reference is no worse
/// than BF16 storage of Q/K/V/P.  `0` keeps every launch of the incumbent
/// route.  Read once.
static QWEN4_QSA_WMMA_GATHER: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_WMMA_GATHER", true)
});
/// `HIPFIRE_QWEN4_QSA_PM` (on unless `0`) runs the gathered route's producer
/// and attention from the certified builder module (`kernels::QSA_GATHER_PM_*`,
/// same ABI, grid, LDS and output bytes as the hipcc kernels) instead of the
/// JIT source, on every gathered-route arch (gfx1151, gfx1201).  `0` keeps the
/// hipcc kernels.  Read once.
static QWEN4_QSA_PM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_PM", true)
});
/// `HIPFIRE_QWEN4_QSA_SELECT_EXACT=1` runs the batched QSA selector on the
/// `_exact` kernels (tile sort + fixed-order merge instead of the all-pairs
/// ranks; selected indices and mirror byte-identical) for complete <= 2048
/// pooled blocks, outside recording and capture.  Unset or `0` keeps the
/// incumbent selector.  Read once.
static QWEN4_QSA_SELECT_EXACT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SELECT_EXACT", false)
});
/// Pooled blocks the exact selector handles; larger contexts keep the
/// incumbent selector.
const QSA_SELECT_EXACT_MAX_BLOCKS: usize = 2048;
/// Budget blocks the exact selector's 512-entry merge bound covers.
const QSA_SELECT_EXACT_MAX_BUDGET_BLOCKS: usize = 512;
/// Dynamic LDS floor of the exact selector kernels
/// (`indexed_attention_select_exact.hip`), which keep no static LDS:
/// 6144 B even in global-score mode, otherwise 4 B per live block.
const QSA_SELECT_EXACT_DYNAMIC_LDS_FLOOR_BYTES: usize = 6 * 1024;
const QSA_SELECT_PARALLEL_THREADS: u32 = 256;
// gfx1151's 64-KiB dynamic LDS budget; other devices use the serial path.
// Oversized rows also use serial kernels without changing the contract.
const QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;
/// Static LDS of `indexed_attention_select_f32_batched` beside its dynamic
/// score row: 256 radix bins and the 512-entry chosen-block list
/// (`QSA_SELECT_LIST_CAPACITY` in tensor_ops.hip) plus scalars, rounded up.
/// The LDS route needs `shape_blocks * 4 + this` within the LDS limit, i.e. at
/// most 15360 pooled blocks (61440 tokens at compress 4); larger arenas keep
/// their score rows in global memory.
pub const QSA_SELECT_BATCHED_STATIC_LDS_BYTES: usize = 4 * 1024;
/// Rows per launch of the batched selector when its scores live in global
/// memory (arenas past the LDS row); longer batches launch in groups, so the
/// score scratch is `QSA_SELECT_GLOBAL_ROWS * padded blocks` floats.
const QSA_SELECT_GLOBAL_ROWS: usize = 256;

#[cfg(test)]
thread_local! {
    /// Test-only: route the batched selector to its serial oracle kernel.
    static SELECT_FORCE_SERIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn select_forced_serial() -> bool {
    SELECT_FORCE_SERIAL.with(|flag| flag.get())
}
#[cfg(not(test))]
#[inline(always)]
fn select_forced_serial() -> bool {
    false
}
const QSA_ATTENTION_PARALLEL_THREADS: u32 = 256;
const QSA_ATTENTION_LDS_BYTES_PER_ROW: usize = 8; // F32 score + i32 token.
const QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeError {
    WrongDtype,
    WrongShape,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongDtype => write!(f, "tensor operation expects F32 tensors"),
            Self::WrongShape => write!(f, "tensor operation tensor shape mismatch"),
        }
    }
}

impl std::error::Error for ComputeError {}

pub(crate) fn ensure_f32(tensor: &GpuTensor) -> HipResult<()> {
    if tensor.dtype != DType::F32 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    Ok(())
}

const KERNEL_BLOCK: usize = 256;
const MAX_KERNEL_EXTENT: usize = i32::MAX as usize;

fn checked_extent(value: usize, what: &'static str) -> HipResult<usize> {
    if value <= MAX_KERNEL_EXTENT {
        Ok(value)
    } else {
        Err(HipError::new(0, &format!("{what} exceeds i32")))
    }
}

fn checked_product(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    checked_extent(value, what)
}

fn checked_product3(
    first: usize,
    second: usize,
    third: usize,
    what: &'static str,
) -> HipResult<usize> {
    checked_product(first, second, what).and_then(|value| checked_product(value, third, what))
}

fn checked_add(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_add(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    checked_extent(value, what)
}

fn checked_i32(value: usize, what: &'static str) -> HipResult<i32> {
    checked_extent(value, what).map(|value| value as i32)
}

fn checked_u32(value: usize, what: &'static str) -> HipResult<u32> {
    u32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds u32")))
}

pub(crate) fn blocks(elements: usize) -> HipResult<u32> {
    let elements = checked_extent(elements, "tensor operation flattened extent")?;
    checked_u32(
        elements.div_ceil(KERNEL_BLOCK),
        "tensor operation block grid",
    )
}

/// Storage format of a GDN recurrent state (`kernels/src/tensor_ops.hip`,
/// "Q8 GDN recurrent state"); the launchers read it from the state tensor's
/// dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnStateFormat {
    /// F32 `[value_head][key][value]`: the exact reference.
    F32,
    /// Qwen3.5's Q8 DeltaNet state as one Raw byte slot: per value head 128
    /// rows (value channel) of 128 i8 key-channel codes, then one F32 scale
    /// per row. 128x128 heads only.
    Q8,
}

impl GdnStateFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8 => "q8",
        }
    }

    pub const fn supports(self, key_dim: usize, value_dim: usize) -> bool {
        match self {
            Self::F32 => key_dim > 0 && value_dim > 0,
            Self::Q8 => key_dim == 128 && value_dim == 128,
        }
    }

    pub const fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Q8 => DType::Raw,
        }
    }

    /// `dtype` units (F32 elements or bytes) of one state of `value_heads`
    /// heads of `key_dim x value_dim`.
    pub const fn state_units(self, value_heads: usize, key_dim: usize, value_dim: usize) -> usize {
        match self {
            Self::F32 => value_heads * key_dim * value_dim,
            Self::Q8 => value_heads * value_dim * (key_dim + 4),
        }
    }

    pub const fn state_bytes(self, value_heads: usize, key_dim: usize, value_dim: usize) -> usize {
        match self {
            Self::F32 => self.state_units(value_heads, key_dim, value_dim) * 4,
            Self::Q8 => self.state_units(value_heads, key_dim, value_dim),
        }
    }

    /// The format of a state tensor (F32, or a Raw Q8 slot).
    pub fn of(state: &GpuTensor) -> HipResult<Self> {
        match state.dtype {
            DType::F32 => Ok(Self::F32),
            DType::Raw => Ok(Self::Q8),
            _ => Err(HipError::new(0, &ComputeError::WrongDtype.to_string())),
        }
    }
}

pub struct GatedDeltaStep<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    /// F32, or a Q8 slot ([`GdnStateFormat`]).
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// The token's position: seeds a Q8 state's requantization (declared to
    /// the recorder, so a replayed decode reseeds per position).
    pub position: usize,
}

/// Head-pair arrival counters of the fused GDN step's optional rotation.
const GDN_PAIR_COUNTERS: usize = 256;

/// Allocate and zero the fused GDN step's head-pair arrival counters if they
/// do not exist yet. Each launch leaves them zero, so the zeroing memset runs
/// once. A retained-body boundary calls this before opening its capture
/// window: the tape replays dispatches only, and a memset inside the window
/// makes the capture effect-incomplete.
pub fn ensure_gdn_pair_counters(gpu: &mut Gpu) -> HipResult<()> {
    if gpu.scratch.gdn_pair_counters.is_none() {
        let counters = gpu.hip.malloc(GDN_PAIR_COUNTERS * 4)?;
        gpu.hip.memset(&counters, 0, GDN_PAIR_COUNTERS * 4)?;
        gpu.scratch.gdn_pair_counters = Some(counters);
    }
    Ok(())
}

pub fn gated_delta_step(gpu: &mut Gpu, p: &GatedDeltaStep<'_>) -> HipResult<()> {
    gated_delta_step_launch(gpu, p, None, None)
}

/// [`gated_delta_step`] followed by [`gated_delta_gate`] of its output
/// (`g.recurrent_output` must be `p.output`). On the gfx11+ 128x128 route one
/// launch keeps the recurrent output in LDS (bitwise the two launches; the
/// recurrent output is then not stored); elsewhere the two launches run.
/// With `rotate_into`, that launch also writes `mq_rotate_x(g.output)` there
/// (head pairs form the 256-wide groups) and the next matching
/// `Gpu::rotate_x_mq` skips (`ScratchState::prerotated`); returns whether it
/// did.
pub fn gated_delta_step_gated(
    gpu: &mut Gpu,
    p: &GatedDeltaStep<'_>,
    g: &GatedDeltaGate<'_>,
    rotate_into: Option<&GpuTensor>,
) -> HipResult<bool> {
    let fused = gpu.arch_caps.has_gfx11_plus_simt() && p.key_dim == 128 && p.value_dim == 128;
    if !fused || g.recurrent_output.buf.as_ptr() != p.output.buf.as_ptr() {
        gated_delta_step(gpu, p)?;
        gated_delta_gate(gpu, g)?;
        return Ok(false);
    }
    validate_gated_delta_gate(g)?;
    let width = g.output.numel();
    let rotate_into = rotate_into
        .filter(|r| p.value_heads.is_multiple_of(2) && r.dtype == DType::F32 && r.numel() >= width);
    gated_delta_step_launch(gpu, p, Some(g), rotate_into)?;
    if let Some(rotated) = rotate_into {
        gpu.scratch.prerotated = Some((
            g.output.buf.as_ptr() as usize,
            rotated.buf.as_ptr() as usize,
            width,
        ));
    }
    Ok(rotate_into.is_some())
}

fn gated_delta_step_launch(
    gpu: &mut Gpu,
    p: &GatedDeltaStep<'_>,
    gated: Option<&GatedDeltaGate<'_>>,
    rotate_into: Option<&GpuTensor>,
) -> HipResult<()> {
    let format = GdnStateFormat::of(p.state)?;
    for tensor in [p.q, p.k, p.v, p.gate, p.beta, p.output] {
        ensure_f32(tensor)?;
    }
    if p.key_heads == 0
        || p.value_heads == 0
        || p.value_heads % p.key_heads != 0
        || !format.supports(p.key_dim, p.value_dim)
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let key_elements = checked_product(p.key_heads, p.key_dim, "GDN key extent")?;
    let value_elements = checked_product(p.value_heads, p.value_dim, "GDN value extent")?;
    checked_product(value_elements, p.key_dim, "GDN state extent")?;
    if p.q.numel() != key_elements
        || p.k.numel() != key_elements
        || p.v.numel() != value_elements
        || p.gate.numel() != p.value_heads
        || p.beta.numel() != p.value_heads
        || p.state.numel() != format.state_units(p.value_heads, p.key_dim, p.value_dim)
        || p.output.numel() != value_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let key_heads = checked_i32(p.key_heads, "GDN key heads")?;
    let value_heads = checked_i32(p.value_heads, "GDN value heads")?;
    let key_dim = checked_i32(p.key_dim, "GDN key width")?;
    let value_dim = checked_i32(p.value_dim, "GDN value width")?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN value-head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    let q8 = format == GdnStateFormat::Q8;
    // The Q8 kernels are the 128x128 bodies on the dequantized state.
    let shared_norm128 = q8
        || gpu.arch_caps.has_gfx11_plus_simt() && p.key_dim == 128 && p.value_dim == 128;
    let kernel = match (gated, shared_norm128, q8) {
        (Some(_), _, true) => "gated_delta_step_gate_norm128_q8",
        (None, _, true) => "gated_delta_step_norm128_q8",
        (Some(_), true, false) => "gated_delta_step_gate_norm128_gfx1151",
        (None, true, false) => "gated_delta_step_shared_norm128_gfx1151",
        _ => "gated_delta_step_f32",
    };
    let block_x = if shared_norm128 { 128 } else { 256 };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.q.buf.as_ptr());
    args.push_ptr(p.k.buf.as_ptr());
    args.push_ptr(p.v.buf.as_ptr());
    args.push_ptr(p.gate.buf.as_ptr());
    args.push_ptr(p.beta.buf.as_ptr());
    args.push_ptr(p.state.buf.as_ptr());
    match gated {
        Some(g) => {
            args.push_ptr(g.z.buf.as_ptr());
            args.push_ptr(g.norm.buf.as_ptr());
            args.push_ptr(g.output.buf.as_ptr());
        }
        None => args.push_ptr(p.output.buf.as_ptr()),
    }
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_i32(key_dim);
    args.push_i32(value_dim);
    args.push_f32((p.key_dim as f32).sqrt().recip());
    if gated.is_some() && shared_norm128 {
        let null = std::ptr::null_mut();
        let (rotated, signs1, signs2, counters) = match rotate_into {
            Some(rotated) => {
                gpu.ensure_mq_signs()?;
                ensure_gdn_pair_counters(gpu)?;
                if p.value_heads / 2 > GDN_PAIR_COUNTERS {
                    return Err(HipError::new(0, "GDN rotate: too many head pairs"));
                }
                (
                    rotated.buf.as_ptr(),
                    gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr(),
                    gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr(),
                    gpu.scratch.gdn_pair_counters.as_ref().unwrap().as_ptr(),
                )
            }
            None => (null, null, null, null),
        };
        for ptr in [rotated, signs1, signs2, counters] {
            args.push_ptr(ptr);
        }
    }
    let mut frame_binding = None;
    if q8 {
        args.push_u32(checked_u32(p.position, "GDN Q8 frame")?);
        frame_binding = Some([crate::replay::ReplayKernargBinding::PositionPlusU32 {
            offset: args.len() - 4,
            addend: 0,
        }]);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [value_heads_grid, value_dim_grid, 1],
        [block_x, 1, 1],
        0,
        args.as_mut_slice(),
        match &frame_binding {
            Some(bindings) => crate::dispatch::ReplayLaunchBindings {
                grid: None,
                kernargs: bindings,
            },
            None => crate::dispatch::ReplayLaunchBindings::NONE,
        },
    )
}
/// Persistent row-batched GDN recurrence for the exact 128x128 geometry.
/// The kernel partitions each value channel's state column across two
/// 64-value halves and serializes cross-half dot chains in shared memory.
/// The older DeltaNet kernels use a different state orientation and omit the
/// source BF16 Q/K normalization, so they cannot replace this exact-state path.
pub struct GatedDeltaStepBatched<'a> {
    pub projection: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    /// F32, or a Q8 slot ([`GdnStateFormat`]).
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    /// Optional `[rows]` ring of states in `state`'s format: the recurrent
    /// state after the last row lands in slot `rows - 1` instead of updating
    /// `state` (speculative verify; `state` keeps the pre-call value).
    pub row_states: Option<&'a GpuTensor>,
    pub rows: usize,
    pub qkv_width: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// The first row's position; a Q8 state's requantization is seeded by
    /// the last row's.
    pub position: usize,
}

pub fn gated_delta_step_batched(gpu: &mut Gpu, p: &GatedDeltaStepBatched<'_>) -> HipResult<()> {
    let format = GdnStateFormat::of(p.state)?;
    for tensor in [p.projection, p.gate, p.beta, p.output] {
        ensure_f32(tensor)?;
    }
    if p.rows == 0
        || p.key_heads == 0
        || p.value_heads == 0
        || p.value_heads % p.key_heads != 0
        || p.key_dim != 128
        || p.value_dim != 128
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let qk = checked_product(p.key_heads, p.key_dim, "GDN batched qk extent")?;
    let value = checked_product(p.value_heads, p.value_dim, "GDN batched value extent")?;
    let expected_qkv = checked_add(
        checked_product(2, qk, "GDN batched qkv")?,
        value,
        "GDN batched qkv",
    )?;
    let rows_qkv = checked_product(p.rows, expected_qkv, "GDN batched projection extent")?;
    let rows_value = checked_product(p.rows, value, "GDN batched output extent")?;
    let rows_heads = checked_product(p.rows, p.value_heads, "GDN batched parameter extent")?;
    let state_units = format.state_units(p.value_heads, p.key_dim, p.value_dim);
    if p.qkv_width != expected_qkv
        || p.projection.numel() != rows_qkv
        || p.gate.numel() != rows_heads
        || p.beta.numel() != rows_heads
        || p.state.numel() != state_units
        || p.output.numel() != rows_value
        || p.row_states.is_some_and(|states| {
            states.dtype != p.state.dtype || states.numel() < p.rows * state_units
        })
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "GDN batched rows")?;
    let qkv_width = checked_i32(p.qkv_width, "GDN batched qkv width")?;
    let key_heads = checked_i32(p.key_heads, "GDN batched key heads")?;
    let value_heads = checked_i32(p.value_heads, "GDN batched value heads")?;
    let key_dim = checked_i32(p.key_dim, "GDN batched key width")?;
    let value_dim = checked_i32(p.value_dim, "GDN batched value width")?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN batched value-head grid")?;
    let kernel = match (p.row_states.is_some(), format) {
        (true, GdnStateFormat::F32) => "gated_delta_step_halves_state128_persistent256_capture_f32",
        (false, GdnStateFormat::F32) => "gated_delta_step_halves_state128_persistent256_f32",
        (true, GdnStateFormat::Q8) => "gated_delta_step_halves_state128_persistent256_capture_q8",
        (false, GdnStateFormat::Q8) => "gated_delta_step_halves_state128_persistent256_q8",
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.projection, p.gate, p.beta, p.state, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    if let Some(states) = p.row_states {
        args.push_ptr(states.buf.as_ptr() as *const _);
    }
    args.push_i32(rows);
    args.push_i32(qkv_width);
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_i32(key_dim);
    args.push_i32(value_dim);
    args.push_f32((p.key_dim as f32).sqrt().recip());
    if format == GdnStateFormat::Q8 {
        args.push_u32(checked_u32(p.position + p.rows - 1, "GDN Q8 frame")?);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [value_heads_grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Few-row verify rollback of every GDN layer in one launch (kernel
/// `gated_delta_rollback_layers_{f32,q8}`): `table` is `layers` pairs of
/// device pointers (captured recurrence input, ring of `format` states);
/// each layer re-runs its first `keep` of `rows` captured rows from ring
/// slot `from` and leaves the last kept row's state in slot
/// `to + keep - 1`. 128-wide heads.
pub struct GatedDeltaRollbackLayers<'a> {
    pub table: &'a GpuTensor,
    pub discard: &'a GpuTensor,
    pub format: GdnStateFormat,
    pub layers: usize,
    pub rows: usize,
    pub keep: usize,
    pub from: usize,
    pub to: usize,
    pub qkv_width: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    /// Position of the last kept row: seeds a Q8 state's requantization.
    pub position: usize,
}

pub fn gated_delta_rollback_layers(
    gpu: &mut Gpu,
    p: &GatedDeltaRollbackLayers<'_>,
) -> HipResult<()> {
    let value = checked_product(p.value_heads, 128, "GDN rollback value width")?;
    if p.keep == 0
        || p.keep > p.rows
        || p.key_heads == 0
        || !p.value_heads.is_multiple_of(p.key_heads)
        || p.qkv_width != 2 * 128 * p.key_heads + value
        || p.table.buf.size() < 16 * p.layers
        || p.discard.numel() < checked_product(p.keep, value, "GDN rollback output")?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let kernel = match p.format {
        GdnStateFormat::F32 => "gated_delta_rollback_layers_f32",
        GdnStateFormat::Q8 => "gated_delta_rollback_layers_q8",
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.table.buf.as_ptr());
    args.push_ptr(p.discard.buf.as_ptr());
    for (value, label) in [
        (p.rows, "GDN rollback rows"),
        (p.keep, "GDN rollback kept rows"),
        (p.from, "GDN rollback source slot"),
        (p.to, "GDN rollback target slot"),
        (p.qkv_width, "GDN rollback qkv width"),
        (p.key_heads, "GDN rollback key heads"),
        (p.value_heads, "GDN rollback value heads"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_f32(128f32.sqrt().recip());
    if p.format == GdnStateFormat::Q8 {
        args.push_u32(checked_u32(p.position, "GDN Q8 frame")?);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [
            checked_u32(p.value_heads, "GDN rollback value-head grid")?,
            checked_u32(p.layers, "GDN rollback layer grid")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// Whether [`gated_delta_step_gate_wmma`] applies: the Qwen4 F16 prefill
/// route (gfx11 WMMA, >= QWEN4_F16_WMMA_MIN_TOKENS rows, no recorder or capture,
/// not opted out) with 128-wide heads.  Decided before the convolution, which
/// then stores its output as packed BF16 for that kernel.
pub fn gated_delta_chunk_route(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.arch_caps.has_wmma_w32()
        && p.rows >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.key_dim == 128
        && p.value_dim == 128
        && p.key_heads > 0
        && p.value_heads % p.key_heads == 0
}

/// Whether H6's inline-Q8 chunk kernel (`gated_delta_chunk_gate_q8_wmma`) can
/// run this step, flag aside: exact gfx1151 on the chunked route, a Q8 slot
/// whose base is 16-byte aligned (the kernel loads codes as `int4`).
pub fn gated_delta_q8_inline_geometry(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.arch_caps.is_gfx1151()
        && gpu.arch_caps.has_wmma_w32()
        && gated_delta_chunk_route(gpu, p)
        && matches!(GdnStateFormat::of(p.state), Ok(GdnStateFormat::Q8))
        && (p.state.buf.as_ptr() as usize) % 16 == 0
}

/// [`gated_delta_q8_inline_geometry`] under `HIPFIRE_QWEN4_GDN_Q8_INLINE=1`
/// (default off): the selector [`gated_delta_step_gate_wmma`] uses.
pub fn gated_delta_q8_inline_route(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.flags.qwen4_gdn_q8_inline_enabled() && gated_delta_q8_inline_geometry(gpu, p)
}

/// Whether H5's convolution that also writes the normalized Q/K
/// (`gated_delta_conv_qknorm_bf16_f32_batched_k4`) can run this chunk, flag
/// aside: exact gfx1151 wave32 on the chunked route (so never a recorder, a
/// graph capture or the persistent gfx1201/gfx1100-persistent route), 16 key
/// heads of 128 in q|k|v channel order, `channels % 256 == 0`, K=4 with a
/// 3-row ring, and a BF16 convolution output.
pub fn gated_delta_conv_qknorm_geometry(
    gpu: &Gpu,
    step: &GatedDeltaStepBatched<'_>,
    conv: &GatedDeltaConvBatched<'_>,
) -> bool {
    gpu.arch_caps.is_gfx1151()
        && gpu.arch_caps.has_wmma_w32()
        && gated_delta_chunk_route(gpu, step)
        && step.key_heads == 16
        && step.key_dim == 128
        && step.qkv_width == conv.channels
        && conv.rows == step.rows
        && conv.channels >= 4096
        && conv.channels % 256 == 0
        && conv.history_rows == 3
        && conv.kernel_size == 4
        && conv.output.dtype == DType::BF16
}

/// [`gated_delta_conv_qknorm_geometry`] under
/// `HIPFIRE_QWEN4_GDN_CONV_QKNORM=1` (default off).  When it holds, the
/// caller launches [`gated_delta_conv_params_qknorm_batched`] in place of
/// [`gated_delta_conv_params_batched`] and then
/// [`gated_delta_step_gate_wmma_qknormed`] in place of
/// [`gated_delta_step_gate_wmma`].
pub fn gated_delta_conv_qknorm_route(
    gpu: &Gpu,
    step: &GatedDeltaStepBatched<'_>,
    conv: &GatedDeltaConvBatched<'_>,
) -> bool {
    gpu.flags.qwen4_gdn_conv_qknorm_enabled() && gated_delta_conv_qknorm_geometry(gpu, step, conv)
}

/// Opt-in arms of the chunked GDN launch.
#[derive(Clone, Copy, Default)]
struct GatedDeltaChunkArms {
    /// The convolution already wrote the normalized Q/K into the scratch.
    qk_prenormed: bool,
    /// Q8 state decoded/requantized inside the recurrence kernel.
    q8_inline: bool,
}

/// [`gated_delta_step_batched`] followed by [`gated_delta_gate_batched`] as
/// one chunked (16-row WY form) F16 WMMA kernel where
/// [`gated_delta_chunk_route`] holds; `p.projection` is the convolution
/// output as packed BF16.  `gate.output` receives the gated rows and
/// `p.output` is not written.  The recurrence's products round to F16, so
/// the route is KLD-gated; the gate is that kernel's expression.  A Q8 state
/// runs the inline-Q8 kernel where [`gated_delta_q8_inline_route`] holds.
pub fn gated_delta_step_gate_wmma(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
) -> HipResult<()> {
    let arms = GatedDeltaChunkArms {
        qk_prenormed: false,
        q8_inline: gated_delta_q8_inline_route(gpu, p),
    };
    gated_delta_step_gate_wmma_arms(gpu, p, gate, arms)
}

/// [`gated_delta_step_gate_wmma`] after
/// [`gated_delta_conv_params_qknorm_batched`] over the same rows: the
/// normalized Q/K are already in the F16 activation scratch, so the
/// Q/K-norm launch is skipped.  Nothing may use that scratch in between.
pub fn gated_delta_step_gate_wmma_qknormed(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
) -> HipResult<()> {
    let arms = GatedDeltaChunkArms {
        qk_prenormed: true,
        q8_inline: gated_delta_q8_inline_route(gpu, p),
    };
    gated_delta_step_gate_wmma_arms(gpu, p, gate, arms)
}

fn gated_delta_step_gate_wmma_arms(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
    arms: GatedDeltaChunkArms,
) -> HipResult<()> {
    if !gated_delta_chunk_route(gpu, p)
        || p.projection.dtype != DType::BF16
        || (arms.q8_inline && !gated_delta_q8_inline_geometry(gpu, p))
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let format = GdnStateFormat::of(p.state)?;
    for tensor in [p.gate, p.beta, gate.z] {
        ensure_f32(tensor)?;
    }
    // The gated output is BF16-rounded: stored as BF16 bits or F32.
    let output_bf16 = gate.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(gate.output)?;
    }
    let qk = checked_product(p.key_heads, p.key_dim, "GDN chunk qk extent")?;
    let value = checked_product(p.value_heads, p.value_dim, "GDN chunk value extent")?;
    if p.qkv_width != 2 * qk + value
        || p.projection.numel() < p.rows * p.qkv_width
        || p.gate.numel() < p.rows * p.value_heads
        || p.beta.numel() < p.rows * p.value_heads
        || p.state.numel() != format.state_units(p.value_heads, p.key_dim, p.value_dim)
        || gate.rows != p.rows
        || gate.value_heads != p.value_heads
        || gate.value_dim != p.value_dim
        || gate.norm.dtype != DType::BF16
        || gate.norm.numel() != p.value_dim
        || gate.z.numel() < p.rows * value
        || gate.output.numel() < p.rows * value
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "GDN chunk rows")?;
    let qkv_width = checked_i32(p.qkv_width, "GDN chunk qkv width")?;
    let key_heads = checked_i32(p.key_heads, "GDN chunk key heads")?;
    let value_heads = checked_i32(p.value_heads, "GDN chunk value heads")?;
    let qn = gpu.qwen4_f16_x_scratch(2 * p.rows * qk)?;
    let qp = qn.buf.as_ptr();
    let kp = unsafe { (qp as *mut u8).add(p.rows * qk * 2) } as *mut std::ffi::c_void;
    if !arms.qk_prenormed {
        gpu.ensure_kernel_public(
            "tensor_ops",
            TENSOR_OPS_SRC,
            "gated_delta_qk_norm_bf16_batched",
        )?;
    }
    if arms.q8_inline {
        gpu.ensure_kernel_public(
            "gated_delta_chunk_q8_wmma",
            GATED_DELTA_CHUNK_Q8_WMMA_SRC,
            "gated_delta_chunk_gate_q8_wmma",
        )?;
    } else {
        gpu.ensure_kernel_public(
            "gated_delta_chunk_wmma",
            GATED_DELTA_CHUNK_WMMA_SRC,
            "gated_delta_chunk_gate_wmma",
        )?;
    }
    // With `qk_prenormed` the convolution already stored the normalized Q/K
    // in this same scratch (`gated_delta_conv_params_qknorm_batched`).
    if !arms.qk_prenormed {
        let mut args = KernargBlob::new();
        args.push_ptr(p.projection.buf.as_ptr());
        args.push_ptr(qp);
        args.push_ptr(kp);
        args.push_i32(rows);
        args.push_i32(qkv_width);
        args.push_i32(key_heads);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            "gated_delta_qk_norm_bf16_batched",
            [checked_u32(p.rows, "GDN chunk row grid")?, 1, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
    }
    // A Q8 state runs the chunk on its F32 dequantization (the state scratch)
    // and is requantized from it after the chunk, seeded by the last row.
    // The inline arm (`gated_delta_chunk_gate_q8_wmma`) does both inside the
    // kernel on the slot itself.  Both seed the requantization with the last
    // row's absolute position, `position + rows - 1`.  The route excludes a
    // recorder and graph capture (`gated_delta_chunk_route`), so the frame is
    // never a replayed kernarg and needs no mutable-frame binding.
    let frame = match format {
        GdnStateFormat::F32 => 0,
        GdnStateFormat::Q8 => checked_u32(p.position + p.rows - 1, "GDN Q8 frame")?,
    };
    let state = match format {
        GdnStateFormat::F32 => p.state.buf.as_ptr(),
        GdnStateFormat::Q8 if arms.q8_inline => p.state.buf.as_ptr(),
        GdnStateFormat::Q8 => {
            let scratch = gpu.gdn_state_f32_scratch(
                GdnStateFormat::F32.state_bytes(p.value_heads, p.key_dim, p.value_dim),
            )?;
            gdn_state_convert(gpu, "gdn_state_q8_to_f32", p.state.buf.as_ptr(), scratch, p.value_heads, None)?;
            scratch
        }
    };
    let mut args = KernargBlob::new();
    args.push_ptr(p.projection.buf.as_ptr());
    args.push_ptr(qp);
    args.push_ptr(kp);
    args.push_ptr(p.gate.buf.as_ptr());
    args.push_ptr(p.beta.buf.as_ptr());
    args.push_ptr(state);
    for tensor in [gate.z, gate.norm, gate.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(qkv_width);
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_f32((p.key_dim as f32).sqrt().recip());
    args.push_i32(i32::from(output_bf16));
    if arms.q8_inline {
        args.push_i32(1);
        args.push_u32(frame);
    }
    args.pad_to(16);
    let chunk_kernel = if arms.q8_inline {
        "gated_delta_chunk_gate_q8_wmma"
    } else {
        "gated_delta_chunk_gate_wmma"
    };
    gpu.launch_blob_recorded(
        chunk_kernel,
        [checked_u32(p.value_heads, "GDN chunk head grid")?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if format == GdnStateFormat::Q8 && !arms.q8_inline {
        gdn_state_convert(gpu, "gdn_state_f32_to_q8", state, p.state.buf.as_ptr(), p.value_heads, Some(frame))?;
    }
    Ok(())
}

/// One of the Q8 GDN state conversions (`gdn_state_q8_to_f32`, or
/// `gdn_state_f32_to_q8` with its requantization `frame`): grid
/// `value_heads * 128` value rows, block 128.
fn gdn_state_convert(
    gpu: &mut Gpu,
    kernel: &str,
    source: *mut std::ffi::c_void,
    destination: *mut std::ffi::c_void,
    value_heads: usize,
    frame: Option<u32>,
) -> HipResult<()> {
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(source);
    args.push_ptr(destination);
    args.push_i32(checked_i32(value_heads, "GDN state heads")?);
    if let Some(frame) = frame {
        args.push_u32(frame);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [checked_u32(value_heads * 128, "GDN state row grid")?, 1, 1],
        [128, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// BF16-stored HC streams (the Qwen4 F16 prefill route) widened to F32 for
/// the PLE block, `out[i] = state[i]`; `state` is F32-typed with its first
/// `n` halves holding the BF16 bits.
pub fn hc_state_bf16_to_f32(
    gpu: &mut Gpu,
    state: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    hc_state_bf16_elementwise(gpu, "hc_state_bf16_to_f32", state, out, n)
}

/// `state[i] = bf16(state[i] + addend[i])` on BF16-stored HC streams; every
/// HC reader rounds the F32 path's unrounded sum to BF16 first.
pub fn hc_state_bf16_add_f32(
    gpu: &mut Gpu,
    state: &GpuTensor,
    addend: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    hc_state_bf16_elementwise(gpu, "hc_state_bf16_add_f32", state, addend, n)
}

fn hc_state_bf16_elementwise(
    gpu: &mut Gpu,
    kernel: &str,
    state: &GpuTensor,
    other: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    ensure_f32(state)?;
    ensure_f32(other)?;
    if n == 0 || state.numel() * 2 < n || other.numel() < n {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(state.buf.as_ptr());
    args.push_ptr(other.buf.as_ptr());
    args.push_i32(checked_i32(n, "HC state elements")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [blocks(n)?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Device-side F32 -> BF16 storage -> F32 conversion at a source activation
/// boundary.  `scratch` is caller-owned and is allocated with the forward
/// arena; no host transfer or per-token allocation occurs.
pub struct Bf16Roundtrip<'a> {
    pub input: &'a GpuTensor,
    pub scratch: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub elements: usize,
}

pub fn bf16_roundtrip_f32(gpu: &mut Gpu, p: &Bf16Roundtrip<'_>) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.output)?;
    let elements = checked_extent(p.elements, "BF16 roundtrip extent")?;
    if elements == 0
        || p.scratch.dtype != DType::BF16
        || p.input.numel() < elements
        || p.scratch.numel() < elements
        || p.output.numel() < elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "BF16 roundtrip extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_roundtrip_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.scratch.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(elements_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "bf16_roundtrip_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// HC-specific in-place fusion of three source BF16 boundaries, F32 scaling,
/// and the existing SiLU expression over a contiguous low-rank buffer.
pub struct HcActivationFused<'a> {
    pub values: &'a GpuTensor,
    pub scale: f32,
    /// Receives the (BF16-exact) results as packed BF16 instead of `values`.
    pub bf16_out: Option<&'a GpuTensor>,
}

pub fn hc_activation_fused_f32(gpu: &mut Gpu, p: &HcActivationFused<'_>) -> HipResult<()> {
    ensure_f32(p.values)?;
    let elements = checked_extent(p.values.numel(), "HC activation extent")?;
    if elements == 0
        || p.bf16_out
            .is_some_and(|out| out.dtype != DType::BF16 || out.numel() < elements)
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "HC activation extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hc_activation_fused_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_i32(elements_i);
    args.push_f32(p.scale);
    args.push_ptr(
        p.bf16_out
            .map_or(std::ptr::null_mut(), |out| out.buf.as_ptr()),
    );
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hc_activation_fused_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Source-exact BF16 product and residual add for the shared expert.
pub struct Bf16ScaledAdd<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub elements: usize,
}

pub fn bf16_scaled_add(gpu: &mut Gpu, p: &Bf16ScaledAdd<'_>) -> HipResult<()> {
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    let elements = checked_extent(p.elements, "BF16 scaled-add extent")?;
    if elements == 0
        || p.residual.numel() < elements
        || p.value.numel() < elements
        || p.scalar.numel() == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "BF16 scaled-add extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_scaled_add_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(elements_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "bf16_scaled_add_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// Source-exact BF16 product/residual add for a row-batched shared expert.
pub struct Bf16ScaledAddBatched<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub rows: usize,
    pub elements: usize,
}

pub fn bf16_scaled_add_batched(gpu: &mut Gpu, p: &Bf16ScaledAddBatched<'_>) -> HipResult<()> {
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    let flat_elements = checked_product(p.rows, p.elements, "BF16 batched scaled-add extent")?;
    if p.rows == 0
        || p.elements == 0
        || p.residual.numel() < flat_elements
        || p.value.numel() < flat_elements
        || p.scalar.numel() < p.rows
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "BF16 batched scaled-add rows")?;
    let elements = checked_i32(p.elements, "BF16 batched scaled-add width")?;
    let grid = blocks(flat_elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_scaled_add_batched_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(rows);
    args.push_i32(elements);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "bf16_scaled_add_batched_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct HyperRead<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub low: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    pub rank: usize,
}

pub fn hyper_read(gpu: &mut Gpu, p: &HyperRead<'_>) -> HipResult<()> {
    for tensor in [p.input, p.low, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 || p.rank == 0 || p.norm_weight.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let input_elements = checked_product(p.branches, p.hidden, "HC read input extent")?;
    let up_elements = checked_product(input_elements, p.rank, "HC read projection extent")?;
    let branches = checked_i32(p.branches, "HC read branch count")?;
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let rank = checked_i32(p.rank, "HC read rank")?;
    if p.input.numel() != input_elements
        || p.norm_weight.numel() != input_elements
        || p.low.numel() != p.rank
        || p.up.numel() != up_elements
        || p.normalized.numel() != input_elements
        || p.mixed.numel() != p.hidden
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.low, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rank);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_read_f32",
        [1, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// HC read where the up projection has already been evaluated to one F32
/// gate logit per branch/hidden column.
pub struct HyperReadProjected<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn hyper_read_projected(gpu: &mut Gpu, p: &HyperReadProjected<'_>) -> HipResult<()> {
    for tensor in [p.input, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.norm_weight.dtype != DType::BF16 || p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "projected HC width")?;
    let input_elements = checked_extent(p.input.numel(), "projected HC input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let mixed_elements = checked_product(rows, p.hidden, "projected HC mixed extent")?;
    let branches = checked_i32(p.branches, "projected HC branch count")?;
    let hidden = checked_i32(p.hidden, "projected HC hidden width")?;
    let rows_i = checked_i32(rows, "projected HC row count")?;
    let grid_x = blocks(wide)?;
    let grid_y = checked_u32(rows, "projected HC row grid")?;
    if p.norm_weight.numel() != wide
        || p.up.numel() != input_elements
        || p.normalized.numel() != input_elements
        || p.mixed.numel() != mixed_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_projected_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_read_projected_f32",
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct HyperWrite<'a> {
    pub input: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    /// `input` / `output` hold BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
}

pub fn hyper_write(gpu: &mut Gpu, p: &HyperWrite<'_>) -> HipResult<()> {
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "HC write width")?;
    let input_elements = checked_extent(p.input.numel(), "HC write input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let mixed_elements = checked_product(rows, p.hidden, "HC write mixed extent")?;
    let gate_elements = checked_product(rows, p.branches, "HC write gate extent")?;
    let branches = checked_i32(p.branches, "HC write branch count")?;
    let hidden = checked_i32(p.hidden, "HC write hidden width")?;
    let rows_i = checked_i32(rows, "HC write row count")?;
    let grid_x = blocks(wide)?;
    let grid_y = checked_u32(rows, "HC write row grid")?;
    if p.normalized.numel() != input_elements
        || p.mixed.numel() != mixed_elements
        || p.gates.numel() != gate_elements
        || p.output.numel() != input_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if p.state_bf16 {
        // Eight columns (four BF16 pairs) per thread.
        if p.hidden % 8 != 0 {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_write_bf16x8")?;
        let mut args = KernargBlob::new();
        for tensor in [p.input, p.mixed, p.gates, p.output] {
            args.push_ptr(tensor.buf.as_ptr());
        }
        args.push_i32(branches);
        args.push_i32(hidden);
        args.push_i32(rows_i);
        args.pad_to(16);
        return gpu.launch_blob_recorded(
            "hyper_write_bf16x8",
            [blocks(wide / 8)?, grid_y, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        );
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_write_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_write_f32",
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// The next hyper write's gate inputs: its norm weight and BF16 `[4, 4 *
/// hidden]` inject projection, and the 16-float quarter-dot output.
pub struct HyperNextGates<'a> {
    pub norm_weight: &'a GpuTensor,
    pub inject: &'a GpuTensor,
    pub quarters: &'a GpuTensor,
}

/// [`hyper_write`] of up to eight F32 token rows (`hidden == 2560`, four
/// branches; rows = `input` elements / width) followed by [`hyper_norm`] of
/// the written streams with `norm_weight` into `normalized`, one launch;
/// bitwise the two launches. `quarters_in` (16 per row, from a previous
/// call's `next`) replaces `p.gates` as the gate source; `next` also computes
/// the next hyper write's gate quarters (its norm and k4 gate GEMV of these
/// streams, bitwise); `clear` is zero-filled (a `zero_f32`).
pub fn hyper_write_norm(
    gpu: &mut Gpu,
    p: &HyperWrite<'_>,
    norm_weight: &GpuTensor,
    normalized: &GpuTensor,
    quarters_in: Option<&GpuTensor>,
    next: Option<&HyperNextGates<'_>>,
    clear: Option<&GpuTensor>,
) -> HipResult<()> {
    for tensor in [p.input, p.mixed, p.gates, p.output, normalized] {
        ensure_f32(tensor)?;
    }
    let wide = checked_product(p.branches, p.hidden, "HC write width")?;
    // Token rows: one block per (branch, row).
    let rows = p.input.numel() / wide.max(1);
    let quarters_ok = |q: &GpuTensor| q.dtype == DType::F32 && q.numel() == 16 * rows;
    if p.state_bf16
        || p.branches != 4
        || p.hidden != 2560
        || rows == 0
        || rows > 8
        || norm_weight.dtype != DType::BF16
        || p.input.numel() != rows * wide
        || p.output.numel() != rows * wide
        || normalized.numel() != rows * wide
        || norm_weight.numel() != wide
        || p.mixed.numel() != rows * p.hidden
        || p.gates.numel() != rows * p.branches
        || quarters_in.is_some_and(|q| !quarters_ok(q))
        || clear.is_some_and(|c| c.dtype != DType::F32 || c.numel() > i32::MAX as usize)
        || next.is_some_and(|n| {
            !quarters_ok(n.quarters)
                || n.norm_weight.dtype != DType::BF16
                || n.norm_weight.numel() != wide
                || n.inject.dtype != DType::BF16
                || n.inject.numel() != 4 * wide
        })
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel(
        "qwen4_gemv_bf16_xf32",
        crate::kernels::QWEN4_GEMV_BF16_XF32_SRC,
        "hyper_write_norm_f32",
    )?;
    let null = std::ptr::null_mut();
    let mut args = KernargBlob::new();
    for ptr in [
        p.input.buf.as_ptr(),
        p.mixed.buf.as_ptr(),
        p.gates.buf.as_ptr(),
        quarters_in.map_or(null, |q| q.buf.as_ptr()),
        p.output.buf.as_ptr(),
        norm_weight.buf.as_ptr(),
        normalized.buf.as_ptr(),
        next.map_or(null, |n| n.norm_weight.buf.as_ptr()),
        next.map_or(null, |n| n.inject.buf.as_ptr()),
        next.map_or(null, |n| n.quarters.buf.as_ptr()),
        clear.map_or(null, |c| c.buf.as_ptr()),
    ] {
        args.push_ptr(ptr);
    }
    args.push_i32(clear.map_or(0, |c| c.numel() as i32));
    args.push_i32(4);
    args.push_i32(2560);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_write_norm_f32",
        [4, rows as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct HyperNorm<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    /// `input` holds BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
}

pub fn hyper_norm(gpu: &mut Gpu, p: &HyperNorm<'_>) -> HipResult<()> {
    hyper_norm_impl(gpu, p, std::ptr::null_mut(), false, 0)
}

/// [`hyper_norm`] that writes the normalized rows as F16 into
/// `normalized_f16` at row pitch `ld16` elements (`>= branches * hidden`),
/// the F16 WMMA projections' input, and with `bf16_copy` also stores
/// `normalized` as BF16 bits (the values are BF16-rounded) in the first half
/// of its buffer: read it with `HyperReadUpFused::normalized_bf16`.  Without
/// it `normalized` is untouched.
pub fn hyper_norm_f16(
    gpu: &mut Gpu,
    p: &HyperNorm<'_>,
    normalized_f16: &GpuTensor,
    ld16: usize,
    bf16_copy: bool,
) -> HipResult<()> {
    let wide = checked_product(p.branches, p.hidden, "HC norm width")?;
    let rows = p.normalized.numel() / wide.max(1);
    if normalized_f16.dtype != DType::F16 || ld16 < wide || normalized_f16.numel() < rows * ld16 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let ld16 = checked_i32(ld16, "HC norm F16 pitch")?;
    hyper_norm_impl(gpu, p, normalized_f16.buf.as_ptr(), bf16_copy, ld16)
}

fn hyper_norm_impl(
    gpu: &mut Gpu,
    p: &HyperNorm<'_>,
    normalized_f16: *mut std::ffi::c_void,
    bf16_copy: bool,
    ld16: i32,
) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.normalized)?;
    if p.norm_weight.dtype != DType::BF16 || p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "HC norm width")?;
    let input_elements = checked_extent(p.input.numel(), "HC norm input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let branches = checked_i32(p.branches, "HC norm branch count")?;
    let hidden = checked_i32(p.hidden, "HC norm hidden width")?;
    let rows_i = checked_i32(rows, "HC norm row count")?;
    let branch_grid = checked_u32(p.branches, "HC norm branch grid")?;
    let row_grid = checked_u32(rows, "HC norm row grid")?;
    if p.norm_weight.numel() != wide || p.normalized.numel() != input_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_norm_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(if normalized_f16.is_null() || bf16_copy {
        p.normalized.buf.as_ptr()
    } else {
        std::ptr::null_mut()
    });
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.push_ptr(normalized_f16);
    args.push_i32(i32::from(p.state_bf16));
    args.push_i32(ld16);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_norm_f32",
        [branch_grid, row_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// `hyper_norm` of each row followed by its BF16 `[branches, branches *
/// hidden]` gate projection (the multi-row BF16 GEMM), fused per row without
/// writing `normalized`; bitwise identical to the two-launch sequence.
pub struct HyperNormGate<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub gate_weight: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub rows: usize,
    pub branches: usize,
    pub hidden: usize,
    /// `input` holds BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
    /// Also write the rows normalized with this (read) norm weight as F16 at
    /// row pitch `.2`: [`hyper_norm_f16`]'s output for a hyper read of the
    /// same streams, bitwise.
    pub read_f16: Option<(&'a GpuTensor, &'a GpuTensor, usize)>,
}

impl HyperNormGate<'_> {
    /// Geometry the fused kernel supports: four branches, `hidden` a multiple
    /// of 256 up to 2560.
    pub fn supports(branches: usize, hidden: usize) -> bool {
        branches == 4 && hidden % 256 == 0 && (256..=2560).contains(&hidden)
    }
}

pub fn hyper_norm_gate(gpu: &mut Gpu, p: &HyperNormGate<'_>) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.gates)?;
    let wide = checked_product(p.branches, p.hidden, "HC norm-gate width")?;
    if p.norm_weight.dtype != DType::BF16
        || p.gate_weight.dtype != DType::BF16
        || p.rows == 0
        || !HyperNormGate::supports(p.branches, p.hidden)
        || p.norm_weight.numel() != wide
        || p.gate_weight.numel() < p.branches * wide
        || p.input.numel() < p.rows * wide
        || p.gates.numel() < p.rows * p.branches
        || p.read_f16.is_some_and(|(norm, out, ld)| {
            norm.dtype != DType::BF16
                || norm.numel() != wide
                || out.dtype != DType::F16
                || ld < wide
                || out.numel() < p.rows * ld
        })
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let hidden = checked_i32(p.hidden, "HC norm-gate hidden width")?;
    let row_grid = checked_u32(p.rows, "HC norm-gate row grid")?;
    let lds_bytes = checked_u32((wide / 2 + 4 * 256) * 4, "HC norm-gate LDS")?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_norm_gate_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(p.gate_weight.buf.as_ptr());
    args.push_ptr(p.gates.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(i32::from(p.state_bf16));
    let (read_norm, read_out, read_ld) = match p.read_f16 {
        Some((norm, out, ld)) => (norm.buf.as_ptr(), out.buf.as_ptr(), ld),
        None => (std::ptr::null_mut(), std::ptr::null_mut(), 0),
    };
    args.push_ptr(read_norm);
    args.push_ptr(read_out);
    args.push_i32(checked_i32(read_ld, "HC norm-gate F16 pitch")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_norm_gate_f32",
        [row_grid, 1, 1],
        [256, 1, 1],
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// gfx1151 HC read tail: the BF16 `[4 * hidden, low_rank]` up projection of
/// `low` fused with `hyper_read_projected` (four branches), bitwise
/// identical to the multi-row BF16 GEMM followed by that kernel; the
/// projected logits are never written.
pub struct HyperReadUpFused<'a> {
    pub up_weight: &'a GpuTensor,
    pub low: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub rows: usize,
    pub hidden: usize,
    pub low_rank: usize,
    /// `normalized` holds BF16 bits ([`hyper_norm_f16`]), not F32.
    pub normalized_bf16: bool,
}

pub fn hyper_read_up_fused(gpu: &mut Gpu, p: &HyperReadUpFused<'_>) -> HipResult<()> {
    ensure_f32(p.low)?;
    ensure_f32(p.normalized)?;
    ensure_f32(p.mixed)?;
    let wide = checked_product(4, p.hidden, "HC read width")?;
    if p.up_weight.dtype != DType::BF16
        || p.rows == 0
        || p.hidden == 0
        || p.hidden % 8 != 0
        || p.low_rank % 8 != 0
        || !(257..=512).contains(&p.low_rank)
        || p.up_weight.numel() < wide * p.low_rank
        || p.low.numel() < p.rows * p.low_rank
        || p.normalized.numel() < p.rows * wide
        || p.mixed.numel() < p.rows * p.hidden
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let low_rank = checked_i32(p.low_rank, "HC read low rank")?;
    let rows = checked_i32(p.rows, "HC read rows")?;
    let column_grid = checked_u32(p.hidden / 8, "HC read column grid")?;
    let row_grid = checked_u32(p.rows.div_ceil(128), "HC read row grid")?;
    let lds_bytes = checked_u32(32 * (p.low_rank / 2 + 1) * 4, "HC read LDS")?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_up_fused_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.up_weight.buf.as_ptr());
    args.push_ptr(p.low.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.mixed.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(low_rank);
    args.push_i32(rows);
    args.push_i32(i32::from(p.normalized_bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_read_up_fused_f32",
        [column_grid, row_grid, 1],
        [256, 1, 1],
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// [`hyper_read_up_fused`] on gfx11 BF16 WMMA, or on gfx1201's gfx12 BF16
/// WMMA when [`Gpu::qwen4_f16_wmma_gfx1201`] admits it; `low` is packed BF16
/// ([`HcActivationFused::bf16_out`]) and `normalized` is
/// [`hyper_norm_f16`]'s F16 copy at row pitch `normalized_ld`
/// (`normalized_bf16` is ignored). Not bit-exact: the logits accumulate the
/// same exact BF16 products in WMMA's F32 order, so a gate occasionally
/// rounds one BF16 step apart; the epilogue is unchanged.
pub fn hyper_read_up_wmma(
    gpu: &mut Gpu,
    p: &HyperReadUpFused<'_>,
    normalized_ld: usize,
) -> HipResult<()> {
    let tiled = gpu.flags.qwen4_hc_up_tile_enabled();
    hyper_read_up_wmma_tiled(gpu, p, normalized_ld, tiled)
}

/// [`hyper_read_up_wmma`] on its baseline entries (`tiled == false`) or the
/// retiled operand-swapped ones (`tiled == true`, bytewise the baseline's).
fn hyper_read_up_wmma_tiled(
    gpu: &mut Gpu,
    p: &HyperReadUpFused<'_>,
    normalized_ld: usize,
    tiled: bool,
) -> HipResult<()> {
    ensure_f32(p.mixed)?;
    let wide = checked_product(4, p.hidden, "HC read width")?;
    let (module, source, entry) = if gpu.arch_caps.has_wmma_w32() {
        ("hyper_read_up_wmma", HYPER_READ_UP_WMMA_SRC, "hyper_read_up_wmma_bf16")
    } else if gpu.qwen4_f16_wmma_gfx1201() {
        (
            "hyper_read_up_wmma_gfx1201",
            HYPER_READ_UP_WMMA_GFX1201_SRC,
            "hyper_read_up_wmma_bf16_gfx1201",
        )
    } else {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    };
    if p.low.dtype != DType::BF16
        || p.normalized.dtype != DType::F16
        || p.up_weight.dtype != DType::BF16
        || p.rows == 0
        || p.hidden % 16 != 0
        || p.low_rank % 16 != 0
        || p.low_rank > 504
        || normalized_ld < wide
        || p.up_weight.numel() < wide * p.low_rank
        || p.low.numel() < p.rows * p.low_rank
        || p.normalized.numel() < p.rows * normalized_ld
        || p.mixed.numel() < p.rows * p.hidden
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let low_rank = checked_i32(p.low_rank, "HC read low rank")?;
    let rows = checked_i32(p.rows, "HC read rows")?;
    let column_grid = checked_u32(p.hidden / 16, "HC read column grid")?;
    // H3 (`HIPFIRE_QWEN4_HC_UP_TILE`, exact gfx1151 / gfx1201): the retiled
    // operand-swapped entries.  gfx1151 keeps the baseline launch geometry;
    // gfx1201's takes 128 rows per block, static LDS only, `low_rank % 64 == 0`.
    let tile_1201 = tiled && entry == "hyper_read_up_wmma_bf16_gfx1201" && p.low_rank % 64 == 0;
    let entry = match (tiled, entry) {
        (true, "hyper_read_up_wmma_bf16") => "hyper_read_up_wmma_bf16_swap",
        (true, "hyper_read_up_wmma_bf16_gfx1201") if tile_1201 => {
            "hyper_read_up_wmma_bf16_gfx1201_t128"
        }
        (_, entry) => entry,
    };
    let row_grid = checked_u32(
        p.rows.div_ceil(if tile_1201 { 128 } else { 512 }),
        "HC read row grid",
    )?;
    let lds_bytes = if tile_1201 {
        0
    } else {
        checked_u32(64 * (p.low_rank + 8) * 2, "HC read LDS")?
    };
    gpu.ensure_kernel_public(module, source, entry)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.up_weight.buf.as_ptr());
    args.push_ptr(p.low.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.mixed.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(low_rank);
    args.push_i32(rows);
    args.push_i32(checked_i32(normalized_ld, "HC read normalized pitch")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        entry,
        [column_grid, row_grid, 1],
        [256, 1, 1],
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct GatedDeltaConv<'a> {
    pub input: &'a GpuTensor,
    pub kernel: &'a GpuTensor,
    pub history: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub next_history: &'a GpuTensor,
    pub channels: usize,
    pub history_rows: usize,
    pub kernel_size: usize,
    pub cursor: usize,
    /// Index of this row inside the chunk whose start position the tape replays
    /// at. The caller derives `cursor = (start_position + row_index) %
    /// history_rows`, so the declared replay binding re-derives exactly that.
    pub row_index: usize,
}

pub fn gated_delta_conv(gpu: &mut Gpu, p: &GatedDeltaConv<'_>) -> HipResult<()> {
    gated_delta_conv_launch(gpu, p, None)
}

/// [`gated_delta_conv`] and [`gated_delta_params`] of one decode row in one
/// launch (`heads <= 256`); both outputs are bitwise the two launches.
pub fn gated_delta_conv_params(
    gpu: &mut Gpu,
    p: &GatedDeltaConv<'_>,
    params: &GatedDeltaParams<'_>,
    heads: usize,
) -> HipResult<()> {
    gated_delta_conv_launch(gpu, p, Some((params, heads)))
}

fn gated_delta_conv_launch(
    gpu: &mut Gpu,
    p: &GatedDeltaConv<'_>,
    params: Option<(&GatedDeltaParams<'_>, usize)>,
) -> HipResult<()> {
    for tensor in [p.input, p.history, p.output, p.next_history] {
        ensure_f32(tensor)?;
    }
    if p.kernel.dtype != DType::BF16 || p.channels == 0 || p.kernel_size == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let expected_history_rows = p.kernel_size - 1;
    let kernel_elements =
        checked_product(p.channels, p.kernel_size, "GDN convolution kernel extent")?;
    let history_elements = checked_product(
        p.channels,
        expected_history_rows,
        "GDN convolution history extent",
    )?;
    if p.history_rows != expected_history_rows
        || p.cursor >= expected_history_rows.max(1)
        || p.input.numel() != p.channels
        || p.kernel.numel() != kernel_elements
        || p.history.numel() != history_elements
        || p.output.numel() != p.channels
        || p.next_history.numel() != history_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let channels = checked_i32(p.channels, "GDN convolution channels")?;
    let history_rows = checked_i32(expected_history_rows, "GDN convolution history rows")?;
    let kernel_size = checked_i32(p.kernel_size, "GDN convolution kernel width")?;
    let cursor = checked_i32(p.cursor, "GDN convolution cursor")?;
    let conv_blocks = blocks(p.channels)?;
    let (kernel_name, grid) = match params {
        None => ("gated_delta_conv_bf16_f32", conv_blocks),
        Some(_) => ("gated_delta_conv_params_bf16_f32", conv_blocks + 1),
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(cursor);
    // The offset the scalar actually landed at, not a hand-counted layout
    // constant: a changed argument list cannot silently move the binding.
    let cursor_offset = args.len() - 4;
    if let Some((params, heads)) = params {
        let heads_i = validate_gated_delta_params(params, heads)?;
        if heads > 256 {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        for tensor in [
            params.a,
            params.b,
            params.a_log,
            params.dt_bias,
            params.gate,
            params.beta,
        ] {
            args.push_ptr(tensor.buf.as_ptr());
        }
        args.push_i32(heads_i);
        args.push_i32(checked_i32(conv_blocks as usize, "GDN convolution blocks")?);
    }
    args.pad_to(16);
    // The cursor is `(start_position + row_index) % history_rows` by
    // construction, so it is a declared dynamic field rather than an unexplained
    // kernarg difference: replay re-derives it instead of replaying the
    // capture-position ring slot.
    //
    // Kernel width 1 (`history_rows == 0`) has no ring to index: the cursor is
    // the constant 0, and a modulo binding would be meaningless (and rejected),
    // so nothing is declared. The placeholder below is never handed to the
    // recorder in that case.
    let cursor_binding = [crate::replay::ReplayKernargBinding::PositionModU32 {
        offset: cursor_offset,
        addend: u32::try_from(p.row_index)
            .map_err(|_| HipError::new(0, "GDN convolution row index exceeds u32"))?,
        modulus: u32::try_from(p.history_rows.max(1))
            .map_err(|_| HipError::new(0, "GDN convolution history rows exceed u32"))?,
    }];
    let declared: &[crate::replay::ReplayKernargBinding] = if p.history_rows > 0 {
        &cursor_binding
    } else {
        &[]
    };
    gpu.launch_blob_recorded(
        kernel_name,
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: declared,
        },
    )
}
/// Ordered K=4 causal convolution over row-major `[rows, channels]` input.
/// Each channel keeps its BF16-rounded history ring across the whole batch.
/// A BF16-typed `output` receives the (BF16-rounded) values as packed BF16.
pub struct GatedDeltaConvBatched<'a> {
    pub input: &'a GpuTensor,
    pub kernel: &'a GpuTensor,
    pub history: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub next_history: &'a GpuTensor,
    pub rows: usize,
    pub channels: usize,
    pub history_rows: usize,
    pub kernel_size: usize,
    pub start_cursor: usize,
}

pub fn gated_delta_conv_batched(gpu: &mut Gpu, p: &GatedDeltaConvBatched<'_>) -> HipResult<()> {
    gated_delta_conv_batched_impl(gpu, p, None, None)
}

/// [`gated_delta_conv_batched`] and [`gated_delta_params_batched`] in one
/// launch (the parameter blocks follow the convolution grid); bitwise both.
pub fn gated_delta_conv_params_batched(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
) -> HipResult<()> {
    validate_gated_delta_params_batched(params)?;
    gated_delta_conv_batched_impl(gpu, conv, Some(params), None)
}

/// [`gated_delta_conv_params_batched`] that also stores the Q/K the chunked
/// recurrence consumes already normalized
/// (`gated_delta_conv_qknorm_bf16_f32_batched_k4`): the convolution values
/// and parameters are bitwise the plain launch's, and the normalized Q/K are
/// bitwise what `gated_delta_qk_norm_bf16_batched` stores, written into the
/// shared F16 activation scratch ([`Gpu::qwen4_f16_x_scratch`]) where
/// [`gated_delta_step_gate_wmma_qknormed`] reads them.  `step` is the
/// recurrence this convolution feeds; it must satisfy
/// [`gated_delta_conv_qknorm_geometry`].  Without `keep_qk_conv` the Q/K
/// columns of `conv.output` are NOT written (the chunk kernel reads only V
/// from it): a caller that consumes them (debug / capture) passes `true`,
/// or stays on [`gated_delta_conv_params_batched`] and
/// [`gated_delta_step_gate_wmma`].
pub fn gated_delta_conv_params_qknorm_batched(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    step: &GatedDeltaStepBatched<'_>,
    keep_qk_conv: bool,
) -> HipResult<()> {
    if !gated_delta_conv_qknorm_geometry(gpu, step, conv) {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    validate_gated_delta_params_batched(params)?;
    let qk = checked_product(step.key_heads, step.key_dim, "GDN chunk qk extent")?;
    let scratch = gpu.qwen4_f16_x_scratch(2 * step.rows * qk)?;
    let qn = scratch.buf.as_ptr();
    let kn = unsafe { (qn as *mut u8).add(step.rows * qk * 2) } as *mut std::ffi::c_void;
    // The kernel stores qn / kn as 8-byte vectors.
    if (qn as usize) % 8 != 0 || (kn as usize) % 8 != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gated_delta_conv_batched_impl(
        gpu,
        conv,
        Some(params),
        Some(ConvQkNorm {
            qn,
            kn,
            keep_qk_conv,
        }),
    )
}

/// H5's extra outputs: the normalized Q/K (`[rows, 16, 128]` BF16 each).
#[derive(Clone, Copy)]
struct ConvQkNorm {
    qn: *mut std::ffi::c_void,
    kn: *mut std::ffi::c_void,
    /// Also store the unnormalized Q/K convolution into the output.
    keep_qk_conv: bool,
}

fn gated_delta_conv_batched_impl(
    gpu: &mut Gpu,
    p: &GatedDeltaConvBatched<'_>,
    params: Option<&GatedDeltaParamsBatched<'_>>,
    qk_norm: Option<ConvQkNorm>,
) -> HipResult<()> {
    for tensor in [p.history, p.next_history] {
        ensure_f32(tensor)?;
    }
    // Input and output are each F32 or BF16 bits (BF16-rounded values).
    let output_bf16 = p.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(p.output)?;
    }
    let input_bf16 = p.input.dtype == DType::BF16;
    if !input_bf16 {
        ensure_f32(p.input)?;
    }
    if p.kernel.dtype != DType::BF16
        || p.rows == 0
        || p.channels == 0
        || p.history_rows != 3
        || p.kernel_size != 4
        || p.start_cursor >= p.history_rows
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let input_elements = checked_product(p.rows, p.channels, "GDN batched convolution input")?;
    let history_elements = checked_product(
        p.channels,
        p.history_rows,
        "GDN batched convolution history",
    )?;
    let kernel_elements =
        checked_product(p.channels, p.kernel_size, "GDN batched convolution kernel")?;
    if p.input.numel() != input_elements
        || p.output.numel() != input_elements
        || p.history.numel() != history_elements
        || p.next_history.numel() != history_elements
        || p.kernel.numel() != kernel_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "GDN batched convolution rows")?;
    let channels = checked_i32(p.channels, "GDN batched convolution channels")?;
    let history_rows = checked_i32(p.history_rows, "GDN batched convolution history rows")?;
    let kernel_size = checked_i32(p.kernel_size, "GDN batched convolution kernel width")?;
    let start_cursor = checked_i32(p.start_cursor, "GDN batched convolution cursor")?;
    // BF16 in and out: four channels per thread, eight rows per block row
    // (H5's Q/K-norm kernel keeps the scalar launch geometry).
    let x4 = qk_norm.is_none() && input_bf16 && output_bf16 && p.channels % 4 == 0;
    let (grid, rows_per_block) = if x4 {
        (
            checked_u32(p.channels.div_ceil(1024), "GDN batched convolution grid")?,
            8,
        )
    } else {
        (blocks(p.channels)?, 16)
    };
    let row_grid = checked_u32(
        p.rows.div_ceil(rows_per_block),
        "GDN batched convolution row grid",
    )?;
    let kernel = if x4 {
        "gated_delta_conv_bf16x4_batched_k4"
    } else if qk_norm.is_some() {
        "gated_delta_conv_qknorm_bf16_f32_batched_k4"
    } else {
        "gated_delta_conv_bf16_f32_batched_k4"
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(channels);
    if !x4 {
        args.push_i32(history_rows);
        args.push_i32(kernel_size);
    }
    args.push_i32(start_cursor);
    let start_cursor_offset = args.len() - 4;
    if !x4 {
        // H5's `output_bf16` is a flag word: bit 0 = packed BF16 output, bit 1 =
        // also store the unnormalized Q/K convolution (debug / capture).
        args.push_i32(
            i32::from(output_bf16) | qk_norm.map_or(0, |n| 2 * i32::from(n.keep_qk_conv)),
        );
        args.push_i32(i32::from(input_bf16));
    }
    let (param_elements, param_heads) = match params {
        Some(q) => {
            for tensor in [q.a, q.b, q.a_log, q.dt_bias, q.gate, q.beta] {
                args.push_ptr(tensor.buf.as_ptr());
            }
            (q.rows * q.heads, q.heads)
        }
        None => {
            for _ in 0..6 {
                args.push_ptr(std::ptr::null());
            }
            (0, 1)
        }
    };
    args.push_i32(checked_i32(param_elements, "GDN batched parameter extent")?);
    args.push_i32(checked_i32(param_heads, "GDN batched parameter heads")?);
    if let Some(norm) = qk_norm {
        args.push_ptr(norm.qn);
        args.push_ptr(norm.kn);
    }
    args.pad_to(16);
    // The x4 kernel spreads the parameter blocks over every block row.
    let param_span = if x4 { 256 * row_grid as usize } else { 256 };
    let param_grid = checked_u32(
        param_elements.div_ceil(param_span),
        "GDN batched parameter grid",
    )?;
    // `start_cursor` is `start_position % history_rows` for the chunk (the
    // kernel advances the ring per row from there), so the declared binding
    // re-derives it at the replay position.
    let start_cursor_binding = [crate::replay::ReplayKernargBinding::PositionModU32 {
        offset: start_cursor_offset,
        addend: 0,
        modulus: u32::try_from(p.history_rows)
            .map_err(|_| HipError::new(0, "GDN batched convolution history rows exceed u32"))?,
    }];
    gpu.launch_blob_recorded(
        kernel,
        [grid + param_grid, row_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &start_cursor_binding,
        },
    )
}

pub struct GatedDeltaParams<'a> {
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
}

pub fn gated_delta_params(gpu: &mut Gpu, p: &GatedDeltaParams<'_>, heads: usize) -> HipResult<()> {
    let heads_i = validate_gated_delta_params(p, heads)?;
    let grid = blocks(heads)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_params_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_bf16_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

fn validate_gated_delta_params(p: &GatedDeltaParams<'_>, heads: usize) -> HipResult<i32> {
    for tensor in [p.a, p.b, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if heads == 0
        || p.a_log.dtype != DType::BF16
        || p.dt_bias.dtype != DType::BF16
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    checked_i32(heads, "GDN parameter head count")
}
/// Row-batched parameter expansion for the exact Qwen4 prefill route.
pub struct GatedDeltaParamsBatched<'a> {
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub rows: usize,
    pub heads: usize,
}

fn validate_gated_delta_params_batched(p: &GatedDeltaParamsBatched<'_>) -> HipResult<usize> {
    for tensor in [p.a, p.b, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if p.rows == 0 || p.heads == 0 || p.a_log.dtype != DType::BF16 || p.dt_bias.dtype != DType::BF16
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements = checked_product(p.rows, p.heads, "GDN batched parameter extent")?;
    if p.a.numel() != elements
        || p.b.numel() != elements
        || p.gate.numel() != elements
        || p.beta.numel() != elements
        || p.a_log.numel() != p.heads
        || p.dt_bias.numel() != p.heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok(elements)
}

pub fn gated_delta_params_batched(gpu: &mut Gpu, p: &GatedDeltaParamsBatched<'_>) -> HipResult<()> {
    let elements = validate_gated_delta_params_batched(p)?;
    let rows = checked_i32(p.rows, "GDN batched parameter rows")?;
    let heads = checked_i32(p.heads, "GDN batched parameter heads")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "gated_delta_params_bf16_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(heads);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_bf16_f32_batched",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn gated_delta_params_f32(
    gpu: &mut Gpu,
    p: &GatedDeltaParams<'_>,
    heads: usize,
) -> HipResult<()> {
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if heads == 0
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let heads_i = checked_i32(heads, "GDN parameter head count")?;
    let grid = blocks(heads)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_params_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct GatedDeltaGate<'a> {
    pub recurrent_output: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub value_heads: usize,
    pub value_dim: usize,
}

pub fn gated_delta_gate(gpu: &mut Gpu, p: &GatedDeltaGate<'_>) -> HipResult<()> {
    let (value_heads, value_dim) = validate_gated_delta_gate(p)?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN gate head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_gate_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_gate_bf16_f32",
        [value_heads_grid, value_dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

fn validate_gated_delta_gate(p: &GatedDeltaGate<'_>) -> HipResult<(i32, i32)> {
    for tensor in [p.recurrent_output, p.z, p.output] {
        ensure_f32(tensor)?;
    }
    if p.norm.dtype != DType::BF16 || p.value_heads == 0 || p.value_dim == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements = checked_product(p.value_heads, p.value_dim, "GDN gate extent")?;
    if p.recurrent_output.numel() != elements
        || p.z.numel() != elements
        || p.norm.numel() != p.value_dim
        || p.output.numel() != elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok((
        checked_i32(p.value_heads, "GDN gate head count")?,
        checked_i32(p.value_dim, "GDN gate width")?,
    ))
}
/// Row-batched gated RMSNorm with the exact BF16 recurrent boundary folded
/// into the kernel.
pub struct GatedDeltaGateBatched<'a> {
    pub recurrent_output: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub value_heads: usize,
    pub value_dim: usize,
}

fn validate_gated_delta_gate_batched(p: &GatedDeltaGateBatched<'_>) -> HipResult<usize> {
    for tensor in [p.recurrent_output, p.z, p.output] {
        ensure_f32(tensor)?;
    }
    if p.norm.dtype != DType::BF16 || p.rows == 0 || p.value_heads == 0 || p.value_dim != 128 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let row_elements = checked_product(p.value_heads, p.value_dim, "GDN batched gate row extent")?;
    let elements = checked_product(p.rows, row_elements, "GDN batched gate extent")?;
    if p.recurrent_output.numel() != elements
        || p.z.numel() != elements
        || p.norm.numel() != p.value_dim
        || p.output.numel() != elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok(elements)
}

pub fn gated_delta_gate_batched(gpu: &mut Gpu, p: &GatedDeltaGateBatched<'_>) -> HipResult<()> {
    let elements = validate_gated_delta_gate_batched(p)?;
    let rows = checked_i32(p.rows, "GDN batched gate rows")?;
    let value_heads = checked_i32(p.value_heads, "GDN batched gate heads")?;
    let value_dim = checked_i32(p.value_dim, "GDN batched gate width")?;
    let value_heads_grid = checked_u32(elements / p.value_dim, "GDN batched gate grid")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "gated_delta_gate_bf16_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_gate_bf16_f32_batched",
        [value_heads_grid, 1, 1],
        [32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// [`gated_delta_gate_batched`] (even `value_heads`) that also writes
/// `mq_rotate_x(p.output)` into `rotated` (head pairs form the 256-wide
/// groups); the next matching `Gpu::rotate_x_mq_batched` then skips
/// (`ScratchState::prerotated`).
pub fn gated_delta_gate_batched_rotate(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    rotated: &GpuTensor,
) -> HipResult<()> {
    const KERNEL: &str = "gated_delta_gate_rotate_bf16_f32_batched";
    let elements = validate_gated_delta_gate_batched(p)?;
    ensure_f32(rotated)?;
    if !p.value_heads.is_multiple_of(2) || rotated.numel() < elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, KERNEL)?;
    gpu.ensure_mq_signs()?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output, rotated] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
    args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
    args.pad_to(16);
    gpu.launch_blob_recorded(
        KERNEL,
        [
            checked_u32(elements / 256, "GDN batched gate pair grid")?,
            1,
            1,
        ],
        [64, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    gpu.scratch.prerotated = Some((
        p.output.buf.as_ptr() as usize,
        rotated.buf.as_ptr() as usize,
        elements,
    ));
    Ok(())
}
pub struct IndexedAttentionNormRope<'a> {
    pub values: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub heads: usize,
    pub head_dim: usize,
    /// Distance, in elements, between consecutive query/key heads.
    pub head_stride: usize,
    pub position: usize,
    pub rotary_dim: usize,
}

pub fn indexed_attention_norm_rope(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRope<'_>,
) -> HipResult<()> {
    ensure_f32(p.values)?;
    if p.norm.dtype != DType::BF16
        || p.heads == 0
        || p.head_dim == 0
        || p.head_dim > 256
        || p.head_stride < p.head_dim
        || p.rotary_dim == 0
        || p.rotary_dim % 2 != 0
        || p.rotary_dim > p.head_dim
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let span = checked_product(p.heads - 1, p.head_stride, "indexed attention head span")
        .and_then(|base| checked_add(base, p.head_dim, "indexed attention head span"))?;
    let heads = checked_i32(p.heads, "indexed attention head count")?;
    let head_dim = checked_i32(p.head_dim, "indexed attention head width")?;
    let head_stride = checked_i32(p.head_stride, "indexed attention head stride")?;
    let position = checked_i32(p.position, "indexed attention position")?;
    let rotary_dim = checked_i32(p.rotary_dim, "indexed attention rotary width")?;
    let head_grid = checked_u32(p.heads, "indexed attention head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.values.numel() < span || p.norm.numel() != p.head_dim {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_norm_rope_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_ptr(p.norm.buf.as_ptr());
    args.push_i32(heads);
    args.push_i32(head_dim);
    args.push_i32(head_stride);
    args.push_i32(position);
    args.push_i32(rotary_dim);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_norm_rope_f32",
        [head_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct IndexedAttentionNormRopeBatch<'a> {
    pub values: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub rows: usize,
    pub row_stride: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub head_stride: usize,
    pub position_start: usize,
    pub rotary_dim: usize,
}

/// Storage format of a QSA layer's state: the full K/V caches and the
/// indexer's raw and pooled keys (`kernels/src/tensor_ops.hip`, "QSA state
/// formats"). The index keys are BF16 values in every format, so the fp8
/// format's BF16 index arenas pool and select exactly what F32 arenas do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QsaKvFormat {
    /// F32 K/V and index keys: the exact reference.
    F32,
    /// E4M3 K/V with one f16 scale per head and token (the Qwen3.5 native fp8
    /// row; gfx12 only), BF16 index keys.
    Fp8,
}

impl QsaKvFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Fp8 => "fp8",
        }
    }

    /// Whether the kernels implement this format for a head geometry.
    pub const fn supports(self, kv_heads: usize, head_dim: usize) -> bool {
        match self {
            Self::F32 => head_dim > 0,
            // Whole-head scales over one 256-thread block, and 4-byte aligned
            // code rows (`kv_heads * 258` bytes).
            Self::Fp8 => head_dim == 256 && kv_heads % 2 == 0,
        }
    }

    /// Dtype of the K/V tensors: F32 elements, or fp8 byte rows as Raw.
    pub const fn kv_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Fp8 => DType::Raw,
        }
    }

    /// `kv_dtype` units (elements or bytes) per token row of K or of V.
    pub const fn kv_row_units(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::F32 => kv_heads * head_dim,
            Self::Fp8 => kv_heads * (head_dim + 2),
        }
    }

    /// Bytes per token row of K or of V.
    pub const fn kv_row_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::F32 => kv_heads * head_dim * 4,
            Self::Fp8 => self.kv_row_units(kv_heads, head_dim),
        }
    }

    /// Dtype of the raw and pooled index-key arenas.
    pub const fn index_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Fp8 => DType::BF16,
        }
    }

    /// This format's entry of a `[f32, fp8]` kernel-name table.
    fn kernel(self, names: [&'static str; 2]) -> &'static str {
        names[self as usize]
    }
}

/// Refuse a format the device or head geometry has no kernels for, and K/V
/// tensors that are not in its dtype.
fn check_qsa_format(
    gpu: &Gpu,
    format: QsaKvFormat,
    kv_heads: usize,
    head_dim: usize,
    tensors: [&GpuTensor; 2],
) -> HipResult<()> {
    if !format.supports(kv_heads, head_dim) {
        return Err(HipError::new(
            0,
            &format!(
                "QSA {} K/V needs another head geometry ({kv_heads} KV heads x {head_dim})",
                format.name()
            ),
        ));
    }
    if format == QsaKvFormat::Fp8 && !(gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201()) {
        return Err(HipError::new(0, "QSA fp8 K/V needs a gfx12 device"));
    }
    if tensors.iter().any(|tensor| tensor.dtype != format.kv_dtype()) {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    Ok(())
}

/// QSA prologue of `rows` consecutive rows (decode: one): the index-query,
/// query and key norm+RoPE, the key/value cache append and the index key's
/// BF16 round trip + raw-key copy, in one launch (bitwise the six launches it
/// replaces). Row buffers are row-major at their natural row widths. The
/// cache rows are `format`'s, the raw index keys its `index_dtype`.
pub struct IndexedAttentionDecodePrologue<'a> {
    /// `[index q (index_heads * index_dim) | index k (index_kv_width)]`.
    pub index_row: &'a GpuTensor,
    /// `[heads, 2 * head_dim]` query + gate.
    pub qgate: &'a GpuTensor,
    pub keys: &'a GpuTensor,
    pub values: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub index_q_norm: &'a GpuTensor,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_kv_width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub position: usize,
    pub rows: usize,
    pub format: QsaKvFormat,
}

pub fn indexed_attention_decode_prologue(
    gpu: &mut Gpu,
    p: &IndexedAttentionDecodePrologue<'_>,
) -> HipResult<()> {
    for tensor in [p.index_row, p.qgate, p.keys, p.values] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(gpu, p.format, p.kv_heads, p.head_dim, [p.full_keys, p.full_values])?;
    if p.raw_index_keys.dtype != p.format.index_dtype() {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let row_units = p.format.kv_row_units(p.kv_heads, p.head_dim);
    let kv_width = checked_product(p.kv_heads, p.head_dim, "QSA prologue KV width")?;
    let end = checked_add(p.position, p.rows, "QSA prologue position")?;
    let index_width =
        checked_product(p.index_heads, p.index_dim, "QSA prologue index")? + p.index_kv_width;
    let bad = [p.index_q_norm, p.q_norm, p.k_norm]
        .iter()
        .any(|n| n.dtype != DType::BF16)
        || p.rows == 0
        || p.index_heads == 0
        || p.heads == 0
        || p.kv_heads == 0
        || p.index_dim == 0
        || p.index_dim > 256
        || p.head_dim == 0
        || p.head_dim > 256
        || p.index_q_norm.numel() != p.index_dim
        || p.q_norm.numel() != p.head_dim
        || p.k_norm.numel() != p.head_dim
        || p.index_row.numel() < checked_product(p.rows, index_width, "QSA prologue index")?
        || p.qgate.numel()
            < checked_product3(p.rows, 2 * p.heads, p.head_dim, "QSA prologue query")?
        || p.keys.numel() < checked_product(p.rows, kv_width, "QSA prologue keys")?
        || p.values.numel() < checked_product(p.rows, kv_width, "QSA prologue values")?
        || p.full_keys.numel() < checked_product(end, row_units, "QSA prologue cache")?
        || p.full_values.numel() < checked_product(end, row_units, "QSA prologue cache")?
        || p.raw_index_keys.numel()
            < checked_product(end, p.index_kv_width, "QSA prologue raw keys")?;
    if bad {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let blocks_x = checked_u32(
        p.index_heads + p.heads + p.kv_heads + 1,
        "QSA prologue block count",
    )?;
    let kernel = p.format.kernel([
        "indexed_attention_decode_prologue_f32",
        "indexed_attention_decode_prologue_fp8",
    ]);
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [
        p.index_row,
        p.qgate,
        p.keys,
        p.values,
        p.full_keys,
        p.full_values,
        p.raw_index_keys,
        p.index_q_norm,
        p.q_norm,
        p.k_norm,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for (value, label) in [
        (p.index_heads, "QSA prologue index heads"),
        (p.index_dim, "QSA prologue index width"),
        (p.index_kv_width, "QSA prologue index KV width"),
        (p.heads, "QSA prologue heads"),
        (p.kv_heads, "QSA prologue KV heads"),
        (p.head_dim, "QSA prologue head width"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_i32(checked_i32(p.position, "QSA prologue position")?);
    // Declared dynamic field: replay re-derives the position (angles and
    // both cache rows follow it).
    let position_offset = args.len() - 4;
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    gpu.launch_blob_recorded(
        kernel,
        [blocks_x, checked_u32(p.rows, "QSA prologue rows")?, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )
}

pub fn indexed_attention_norm_rope_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRopeBatch<'_>,
) -> HipResult<()> {
    ensure_f32(p.values)?;
    if p.norm.dtype != DType::BF16
        || p.rows == 0
        || p.row_stride == 0
        || p.heads == 0
        || p.head_dim == 0
        || p.head_dim > 256
        || p.head_stride < p.head_dim
        || p.rotary_dim == 0
        || p.rotary_dim % 2 != 0
        || p.rotary_dim > p.head_dim
        || p.position_start
            .checked_add(p.rows)
            .map_or(true, |end| end > i32::MAX as usize)
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_span = checked_product(
        p.heads - 1,
        p.head_stride,
        "indexed attention batch head span",
    )
    .and_then(|base| checked_add(base, p.head_dim, "indexed attention batch head span"))?;
    if p.row_stride < head_span
        || p.norm.numel() != p.head_dim
        || p.values.numel()
            < checked_product(p.rows - 1, p.row_stride, "indexed attention batch rows")?
                .checked_add(head_span)
                .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "indexed attention batch rows")?;
    let row_stride = checked_i32(p.row_stride, "indexed attention batch row stride")?;
    let heads = checked_i32(p.heads, "indexed attention batch head count")?;
    let head_dim = checked_i32(p.head_dim, "indexed attention batch head width")?;
    let head_stride = checked_i32(p.head_stride, "indexed attention batch head stride")?;
    let position_start = checked_i32(p.position_start, "indexed attention batch position")?;
    let rotary_dim = checked_i32(p.rotary_dim, "indexed attention batch rotary width")?;
    let head_grid = checked_u32(p.heads, "indexed attention batch head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    let row_grid = checked_u32(p.rows, "indexed attention batch row grid")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_norm_rope_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.values, p.norm] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [rows, row_stride, heads, head_dim, head_stride] {
        args.push_i32(value);
    }
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it
    // from its own position instead of replaying the capture-position angle
    // base; the kernel adds the row index itself.
    let position_offset = args.len() - 4;
    args.push_i32(rotary_dim);
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    gpu.launch_blob_recorded(
        "indexed_attention_norm_rope_f32_batched",
        [head_grid, dim_grid, row_grid],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )
}

pub struct IndexedAttentionCacheAppend<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub position: usize,
    pub kv_width: usize,
}

pub fn indexed_attention_cache_append(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppend<'_>,
) -> HipResult<()> {
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        ensure_f32(tensor)?;
    }
    if p.kv_width == 0
        || p.key.numel() != p.kv_width
        || p.value.numel() != p.kv_width
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let end = checked_product(p.position, p.kv_width, "indexed attention cache offset")
        .and_then(|base| checked_add(base, p.kv_width, "indexed attention cache offset"))?;
    if end > p.full_keys.numel() {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let position = checked_i32(p.position, "indexed attention cache position")?;
    let kv_width = checked_i32(p.kv_width, "indexed attention cache width")?;
    let grid = blocks(p.kv_width)?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_cache_append_f32",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(position);
    args.push_i32(kv_width);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_cache_append_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Append `rows` F32 K/V rows at `position_start` to caches in `format`.
pub struct IndexedAttentionCacheAppendBatch<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub rows: usize,
    pub position_start: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub format: QsaKvFormat,
}

pub fn indexed_attention_cache_append_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppendBatch<'_>,
) -> HipResult<()> {
    for tensor in [p.key, p.value] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(gpu, p.format, p.kv_heads, p.head_dim, [p.full_keys, p.full_values])?;
    let kv_width = checked_product(p.kv_heads, p.head_dim, "indexed attention batch width")?;
    if p.rows == 0
        || kv_width == 0
        || p.key.numel() < checked_product(p.rows, kv_width, "indexed attention batch key")?
        || p.value.numel() < checked_product(p.rows, kv_width, "indexed attention batch value")?
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let end = checked_product(
        end_position,
        p.format.kv_row_units(p.kv_heads, p.head_dim),
        "indexed attention batch cache",
    )?;
    if end > p.full_keys.numel() || end_position > i32::MAX as usize {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "indexed attention batch rows")?;
    let position_start = checked_i32(p.position_start, "indexed attention batch position")?;
    let row_grid = checked_u32(p.rows, "indexed attention batch row grid")?;
    let kernel = p.format.kernel([
        "indexed_attention_cache_append_f32_batched",
        "indexed_attention_cache_append_fp8_batched",
    ]);
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it,
    // so the cache row this launch writes follows the replay position.
    let position_offset = args.len() - 4;
    // F32 copies channels across a flat grid; the quantized formats reduce
    // their scales over one block per KV head.
    let grid = if p.format == QsaKvFormat::F32 {
        args.push_i32(checked_i32(kv_width, "indexed attention batch width")?);
        [blocks(kv_width)?, row_grid, 1]
    } else {
        args.push_i32(checked_i32(p.kv_heads, "indexed attention batch KV heads")?);
        args.push_i32(checked_i32(p.head_dim, "indexed attention batch head width")?);
        [checked_u32(p.kv_heads, "indexed attention batch head grid")?, row_grid, 1]
    };
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    gpu.launch_blob_recorded(
        kernel,
        grid,
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )
}

/// Prefill index-key tail into a BF16 raw-key arena
/// (`indexed_attention_index_key_append_bf16_batched`): rounds each row's
/// index key in `index_rows` to BF16 in place and appends it at
/// `position_start + row`, exactly as the decode prologue does per row.
pub struct IndexedAttentionIndexKeyAppendBatch<'a> {
    /// `rows` rows of `[index q (index_q_width) | index k (index_kv_width)]`.
    pub index_rows: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub rows: usize,
    pub index_q_width: usize,
    pub index_kv_width: usize,
    pub position_start: usize,
}

pub fn indexed_attention_index_key_append_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionIndexKeyAppendBatch<'_>,
) -> HipResult<()> {
    ensure_f32(p.index_rows)?;
    if p.raw_index_keys.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let index_width = checked_add(p.index_q_width, p.index_kv_width, "QSA index-key width")?;
    let end = checked_add(p.position_start, p.rows, "QSA index-key position")?;
    if p.rows == 0
        || p.index_kv_width == 0
        || p.index_rows.numel() < checked_product(p.rows, index_width, "QSA index-key rows")?
        || p.raw_index_keys.numel() < checked_product(end, p.index_kv_width, "QSA raw keys")?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    const NAME: &str = "indexed_attention_index_key_append_bf16_batched";
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, NAME)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.index_rows.buf.as_ptr());
    args.push_ptr(p.raw_index_keys.buf.as_ptr());
    for (value, label) in [
        (p.rows, "QSA index-key rows"),
        (p.index_q_width, "QSA index-key query width"),
        (index_width, "QSA index-key row width"),
        (p.index_kv_width, "QSA index-key width"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_i32(checked_i32(p.position_start, "QSA index-key position")?);
    // Declared dynamic field: the chunk's start position (the raw-key row).
    let position_offset = args.len() - 4;
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    gpu.launch_blob_recorded(
        NAME,
        [
            blocks(p.index_kv_width)?,
            checked_u32(p.rows, "QSA index-key row grid")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )
}
pub struct IndexedAttentionSelect<'a> {
    pub query: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub block_count: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub visible: usize,
    pub capacity: usize,
}

pub fn indexed_attention_select(gpu: &mut Gpu, p: &IndexedAttentionSelect<'_>) -> HipResult<()> {
    for tensor in [p.query, p.pooled] {
        ensure_f32(tensor)?;
    }
    if p.selected.dtype != DType::Raw
        || p.index_heads == 0
        || p.index_dim == 0
        || p.compress == 0
        || p.capacity == 0
        || p.visible == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_elements = checked_product(p.index_heads, p.index_dim, "QSA select query extent")?;
    let pooled_elements = checked_product(p.block_count, p.index_dim, "QSA select pooled extent")?;
    checked_product(p.block_count, p.compress, "QSA select token extent")?;
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let block_count = checked_i32(p.block_count, "QSA select block count")?;
    let index_heads = checked_i32(p.index_heads, "QSA select index-head count")?;

    let index_dim = checked_i32(p.index_dim, "QSA select index width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA select budget")?;
    let compress = checked_i32(p.compress, "QSA select compression")?;
    let visible = checked_i32(p.visible, "QSA select visible count")?;
    let capacity = checked_i32(p.capacity, "QSA select capacity")?;
    if p.query.numel() != query_elements
        || p.pooled.numel() < pooled_elements
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    // The rank-parallel batched kernel emits the serial selection sort's
    // bytes; one row at `visible - 1` is this selection whenever the block
    // count follows the visible prefix (the single-row kernel is O(chosen x
    // blocks) in one thread).
    if p.block_count == p.visible / p.compress {
        return indexed_attention_select_batch(
            gpu,
            &IndexedAttentionSelectBatch {
                query: p.query,
                pooled: p.pooled,
                selected: p.selected,
                rows: 1,
                query_row_stride: query_elements,
                block_count: p.block_count,
                index_heads: p.index_heads,
                index_dim: p.index_dim,
                budget_blocks: p.budget_blocks,
                compress: p.compress,
                position_start: p.visible - 1,
                capacity: p.capacity,
                shape_blocks: p.block_count,
            },
        );
    }
    let (kernel_name, block, shared_mem) =
        match p.block_count.checked_mul(std::mem::size_of::<f32>()) {
            Some(bytes)
                if gpu.arch_caps.has_gfx11_plus_simt()
                    && bytes <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (
                    "indexed_attention_select_f32",
                    [QSA_SELECT_PARALLEL_THREADS, 1, 1],
                    bytes as u32,
                )
            }
            _ => ("indexed_attention_select_f32_serial", [1, 1, 1], 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.query, p.pooled, p.selected] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [
        block_count,
        index_heads,
        index_dim,
        budget_blocks,
        compress,
        visible,
        capacity,
    ] {
        args.push_i32(value);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel_name,
        [1, 1, 1],
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct IndexedAttentionSelectBatch<'a> {
    pub query: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub rows: usize,
    pub query_row_stride: usize,
    pub block_count: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub position_start: usize,
    pub capacity: usize,
    /// Declared shape bound for the LDS reservation and the kernel symbol. A
    /// retained tape needs both to be position-independent, so callers pass the
    /// pooling capacity (`>= block_count`); a caller without a capacity passes
    /// the active count. Must be `>= block_count`.
    pub shape_blocks: usize,
}

pub fn indexed_attention_select_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
) -> HipResult<()> {
    indexed_attention_select_batch_impl(gpu, p, None).map(|_| ())
}

/// [`indexed_attention_select_batch`] that also writes the final row's
/// selection into `mirror` (`capacity` i32) on the parallel route; returns
/// whether it did (the caller copies otherwise).
pub fn indexed_attention_select_batch_mirrored(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: &GpuTensor,
) -> HipResult<bool> {
    indexed_attention_select_batch_impl(gpu, p, Some(mirror))
}

fn indexed_attention_select_batch_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
) -> HipResult<bool> {
    ensure_f32(p.query)?;
    // Pooled keys are F32 or BF16 arenas (`QsaKvFormat::index_dtype`).
    let (parallel_kernel, serial_kernel) = match p.pooled.dtype {
        DType::F32 => (
            "indexed_attention_select_f32_batched",
            "indexed_attention_select_f32_batched_serial",
        ),
        DType::BF16 => (
            "indexed_attention_select_bf16_batched",
            "indexed_attention_select_bf16_batched_serial",
        ),
        _ => return Err(HipError::new(0, &ComputeError::WrongDtype.to_string())),
    };
    if p.selected.dtype != DType::Raw
        || p.rows == 0
        || p.query_row_stride == 0
        || p.block_count > 0 && p.compress == 0
        || p.index_heads == 0
        || p.index_dim == 0
        || p.compress == 0
        || p.capacity == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_elements = checked_product(p.index_heads, p.index_dim, "QSA batch select query")?;
    if p.query_row_stride < query_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_last = checked_product(
        p.rows - 1,
        p.query_row_stride,
        "QSA batch select query rows",
    )?
    .checked_add(query_elements)
    .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let pooled_elements = checked_product(p.block_count, p.index_dim, "QSA batch select pooled")?;
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA batch select rows")?,
        std::mem::size_of::<i32>(),
        "QSA batch select selected",
    )?;
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if end_position > i32::MAX as usize
        || p.query.numel() < query_last
        || p.pooled.numel() < pooled_elements
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "QSA batch select rows")?;
    let query_row_stride = checked_i32(p.query_row_stride, "QSA batch select query stride")?;
    let block_count = checked_i32(p.block_count, "QSA batch select blocks")?;
    let index_heads = checked_i32(p.index_heads, "QSA batch select heads")?;
    let index_dim = checked_i32(p.index_dim, "QSA batch select dim")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA batch select budget")?;
    let compress = checked_i32(p.compress, "QSA batch select compress")?;
    let position_start = checked_i32(p.position_start, "QSA batch select position")?;
    let capacity = checked_i32(p.capacity, "QSA batch select capacity")?;
    let row_grid = checked_u32(p.rows, "QSA batch select row grid")?;
    // The active count is declared to the recorder as `(position_start + rows) /
    // compress`, so replay re-derives it. Verify the caller used that formula:
    // a mismatch would make replay select against a different block count.
    let expected_blocks = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, "QSA batch select position overflow"))?
        / p.compress;
    if expected_blocks != p.block_count {
        return Err(HipError::new(
            0,
            &format!(
                "QSA batch select block count {} is not (position_start {} + rows {}) / compress {} = {expected_blocks}",
                p.block_count, p.position_start, p.rows, p.compress
            ),
        ));
    }
    // Reserved LDS bytes and kernel symbol come from the *bound*, not from the
    // active count, so a pinned bound makes both position-independent. The
    // kernel indexes its LDS by the active clamp (`row_block_count`), so a
    // larger reservation is never read.
    if p.shape_blocks < p.block_count {
        return Err(HipError::new(
            0,
            &format!(
                "QSA batch select shape bound {} is below the active block count {}",
                p.shape_blocks, p.block_count
            ),
        ));
    }
    let shape_blocks = p.shape_blocks;
    let lds_bytes = shape_blocks
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipError::new(0, "QSA batch select shape overflow"))?;
    let parallel = gpu.arch_caps.has_gfx11_plus_simt()
        && shape_blocks > 0
        && !select_forced_serial();
    // Live F32 launches with the pinned index geometry score 8 or 16 rows per
    // pooled key read into the shared F16 X scratch, then select from those
    // scores (any block count: the score rows live in global memory).
    if parallel
        && p.pooled.dtype == DType::F32
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.index_heads == 4
        && p.index_dim.is_multiple_of(4)
        && p.index_dim <= 128
        && p.block_count > 0
        && p.budget_blocks <= QSA_SELECT_FROM_SCORES_MAX_BUDGET
    {
        return indexed_attention_select_rows8(gpu, p, mirror);
    }
    // Past the LDS row the scores go to global rows (one per workgroup, at
    // most QSA_SELECT_GLOBAL_ROWS per launch); the selection is the same.
    let global = parallel
        && lds_bytes + QSA_SELECT_BATCHED_STATIC_LDS_BYTES > QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES;
    if global && p.rows > QSA_SELECT_GLOBAL_ROWS {
        let query_elements = p.query.numel();
        let selected_bytes = p.selected.numel();
        let row_bytes = p.capacity * std::mem::size_of::<i32>();
        let mut persisted = false;
        let mut first = 0;
        while first < p.rows {
            let rows = (p.rows - first).min(QSA_SELECT_GLOBAL_ROWS);
            let last = first + rows == p.rows;
            let query_offset = first * p.query_row_stride;
            let query = p.query.sub_offset(query_offset, query_elements - query_offset);
            let selected = p.selected.sub_offset(first * row_bytes, selected_bytes - first * row_bytes);
            let group = IndexedAttentionSelectBatch {
                query: &query,
                pooled: p.pooled,
                selected: &selected,
                rows,
                query_row_stride: p.query_row_stride,
                block_count: (p.position_start + first + rows) / p.compress,
                index_heads: p.index_heads,
                index_dim: p.index_dim,
                budget_blocks: p.budget_blocks,
                compress: p.compress,
                position_start: p.position_start + first,
                capacity: p.capacity,
                shape_blocks: p.shape_blocks,
            };
            persisted = indexed_attention_select_batch_impl(
                gpu,
                &group,
                if last { mirror } else { None },
            )?;
            first += rows;
        }
        return Ok(persisted);
    }
    let scores_stride = shape_blocks.div_ceil(4) * 4;
    let scores_global = if global {
        let bytes = QSA_SELECT_GLOBAL_ROWS
            .checked_mul(scores_stride)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HipError::new(0, "QSA select score rows overflow"))?;
        gpu.qsa_select_scores(bytes)?
    } else {
        std::ptr::null_mut()
    };
    let (kernel_name, block, shared_mem) = if global {
        (parallel_kernel, [QSA_SELECT_PARALLEL_THREADS, 1, 1], 0u32)
    } else if parallel && lds_bytes <= u32::MAX as usize {
        (
            parallel_kernel,
            [QSA_SELECT_PARALLEL_THREADS, 1, 1],
            lds_bytes as u32,
        )
    } else {
        (serial_kernel, [1, 1, 1], 0)
    };
    // `HIPFIRE_QWEN4_QSA_SELECT_EXACT`: the exact selector replaces the
    // incumbent parallel kernel (same ABI, grid and block) for live
    // complete <= 2048 blocks and <= 512 budget blocks, never under a
    // recorder or capture; every other launch is the incumbent's.  The
    // exact kernel keeps no static LDS and needs max(6144, live-score-row)
    // bytes of dynamic LDS: the floor applies even in global-score mode,
    // otherwise 4 B per live block.
    let exact_dynamic_need = QSA_SELECT_EXACT_DYNAMIC_LDS_FLOOR_BYTES.max(if global {
        0
    } else {
        p.block_count.saturating_mul(std::mem::size_of::<f32>())
    });
    let launch_exact = *QWEN4_QSA_SELECT_EXACT
        && kernel_name == parallel_kernel
        && p.block_count <= QSA_SELECT_EXACT_MAX_BLOCKS
        && p.budget_blocks <= QSA_SELECT_EXACT_MAX_BUDGET_BLOCKS
        && exact_dynamic_need <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode;
    let (launch_kernel, shared_mem) = if launch_exact {
        let exact = match p.pooled.dtype {
            DType::F32 => "indexed_attention_select_f32_batched_exact",
            _ => "indexed_attention_select_bf16_batched_exact",
        };
        gpu.ensure_kernel_public(
            "indexed_attention_select_exact",
            INDEXED_ATTENTION_SELECT_EXACT_SRC,
            exact,
        )?;
        (exact, exact_dynamic_need as u32)
    } else {
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
        (kernel_name, shared_mem)
    };
    let mut args = KernargBlob::new();
    for tensor in [p.query, p.pooled, p.selected] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [rows, query_row_stride] {
        args.push_i32(value);
    }
    args.push_i32(block_count);
    let block_count_offset = args.len() - 4;
    for value in [index_heads, index_dim, budget_blocks, compress] {
        args.push_i32(value);
    }
    args.push_i32(position_start);
    let position_offset = args.len() - 4;
    args.push_i32(capacity);
    let mirror = mirror.filter(|m| {
        kernel_name == parallel_kernel
            && m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>()
    });
    if kernel_name == parallel_kernel {
        args.push_ptr(mirror.map_or(std::ptr::null_mut(), |m| m.buf.as_ptr()));
        args.push_ptr(scores_global);
        args.push_i32(checked_i32(scores_stride, "QSA select score stride")?);
    }
    args.pad_to(16);
    // Both declared fields make the selection follow the replay position instead
    // of the capture position.
    let addend =
        u32::try_from(p.rows).map_err(|_| HipError::new(0, "QSA batch select rows exceed u32"))?;
    let divisor = u32::try_from(p.compress)
        .map_err(|_| HipError::new(0, "QSA batch select compression exceeds u32"))?;
    let bindings = [
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: block_count_offset,
            addend,
            divisor,
        },
        crate::replay::ReplayKernargBinding::PositionPlusU32 {
            offset: position_offset,
            addend: 0,
        },
    ];
    gpu.launch_blob_recorded(
        launch_kernel,
        [row_grid, 1, 1],
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &bindings,
        },
    )?;
    Ok(mirror.is_some())
}

/// Score-scratch budget of [`indexed_attention_select_rows8`]: rows are
/// selected in groups whose scores fit it.
const QSA_SELECT_SCORE_SCRATCH_BYTES: usize = 64 << 20;
/// `indexed_attention_select_from_scores` holds the chosen blocks in 512 LDS
/// entries; larger budgets take the batched kernel.
const QSA_SELECT_FROM_SCORES_MAX_BUDGET: usize = 512;

/// [`indexed_attention_select_batch_impl`]'s live route: row groups of
/// `indexed_attention_select_scores_rows{8,16}_f32` scores, each followed by
/// `indexed_attention_select_from_scores` reading them. Selection bytes are
/// unchanged.
fn indexed_attention_select_rows8(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
) -> HipResult<bool> {
    let stride = p.block_count;
    // Prefill scores sixteen rows per pooled-key read; decode and few-row
    // verify keep eight (the sixteen-row kernel costs them more than it saves).
    let (score_kernel, score_rows) = if p.rows > 8 {
        ("indexed_attention_select_scores_rows16_f32", 16)
    } else {
        ("indexed_attention_select_scores_rows8_f32", 8)
    };
    let group = (QSA_SELECT_SCORE_SCRATCH_BYTES / (stride * 4) / 16 * 16)
        .max(16)
        .min(p.rows);
    // Growth goes through the accessor that invalidates captured state first.
    let scores = gpu.qwen4_f16_x_scratch(group * stride * 2)?.buf.as_ptr();
    for kernel in [score_kernel, "indexed_attention_select_from_scores"] {
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    }
    let mirror =
        mirror.filter(|m| m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>());
    let block_count = checked_i32(p.block_count, "QSA batch select blocks")?;
    let block_tiles = checked_u32(p.block_count.div_ceil(256), "QSA select block tiles")?;
    let mut g0 = 0usize;
    while g0 < p.rows {
        let n = group.min(p.rows - g0);
        let query = unsafe {
            (p.query.buf.as_ptr() as *mut f32).add(g0 * p.query_row_stride) as *mut std::ffi::c_void
        };
        let selected = unsafe {
            (p.selected.buf.as_ptr() as *mut i32).add(g0 * p.capacity) as *mut std::ffi::c_void
        };
        let position_start = checked_i32(p.position_start + g0, "QSA batch select position")?;
        let rows = checked_i32(n, "QSA batch select rows")?;
        let query_row_stride = checked_i32(p.query_row_stride, "QSA batch select query stride")?;
        let mut args = KernargBlob::new();
        args.push_ptr(query);
        args.push_ptr(p.pooled.buf.as_ptr());
        args.push_ptr(scores);
        args.push_i32(rows);
        args.push_i32(query_row_stride);
        args.push_i32(block_count);
        args.push_i32(checked_i32(p.index_dim, "QSA batch select dim")?);
        args.push_i32(checked_i32(p.compress, "QSA batch select compress")?);
        args.push_i32(position_start);
        args.push_i32(block_count);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            score_kernel,
            [
                block_tiles,
                checked_u32(n.div_ceil(score_rows), "QSA select row groups")?,
                1,
            ],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::default(),
        )?;
        // The persistent selection is the final row's: the last group's.
        let last = g0 + n == p.rows;
        let mirror_ptr = mirror
            .filter(|_| last)
            .map_or(std::ptr::null_mut(), |m| m.buf.as_ptr());
        let mut args = KernargBlob::new();
        args.push_ptr(scores);
        args.push_i32(block_count);
        args.push_ptr(selected);
        args.push_i32(rows);
        args.push_i32(block_count);
        args.push_i32(checked_i32(p.budget_blocks, "QSA batch select budget")?);
        args.push_i32(checked_i32(p.compress, "QSA batch select compress")?);
        args.push_i32(position_start);
        args.push_i32(checked_i32(p.capacity, "QSA batch select capacity")?);
        args.push_ptr(mirror_ptr);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            "indexed_attention_select_from_scores",
            [rows as u32, 1, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::default(),
        )?;
        g0 += n;
    }
    Ok(mirror.is_some())
}
/// Device-side stable reuse of a prior MTP QSA selection row.
///
/// `selected` is a byte-addressed [`DType::Raw`] allocation containing i32
/// indices. `selected_len_out` is a persistent four-byte Raw scalar written by
/// the kernel; callers can read only that scalar when the logical span changes,
/// never the selected row itself.
pub struct IndexedAttentionReuseSelection<'a> {
    pub selected: &'a GpuTensor,
    pub selected_len: usize,
    pub position: usize,
    pub capacity: usize,
    pub selected_len_out: &'a GpuTensor,
}

pub fn indexed_attention_reuse_selection(
    gpu: &mut Gpu,
    p: &IndexedAttentionReuseSelection<'_>,
) -> HipResult<()> {
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if p.selected.dtype != DType::Raw
        || p.selected_len > p.capacity
        || p.capacity == 0
        || p.selected.numel() < selected_bytes
        || p.selected_len_out.dtype != DType::Raw
        || p.selected_len_out.numel() < std::mem::size_of::<i32>()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let selected_len = checked_i32(p.selected_len, "QSA reuse selected length")?;
    let position = checked_i32(p.position, "QSA reuse position")?;
    let capacity = checked_i32(p.capacity, "QSA reuse capacity")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_reuse_selection",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.selected_len_out.buf.as_ptr());
    args.push_i32(selected_len);
    args.push_i32(position);
    args.push_i32(capacity);
    args.pad_to(16);
    // The prior entries snapshot (the kernel compacts in place).
    let snapshot_bytes = p.selected_len.min(p.capacity) * std::mem::size_of::<i32>();
    if snapshot_bytes > QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES {
        return Err(HipError::new(
            0,
            "QSA reuse selection exceeds its LDS snapshot",
        ));
    }
    gpu.launch_blob_recorded(
        "indexed_attention_reuse_selection",
        [1, 1, 1],
        [256, 1, 1],
        snapshot_bytes as u32,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Formula inputs for a QSA launch whose active count is a quotient of the
/// decode position (`(position_start + rows) / compress`).
///
/// Declaring them lets the retained recorder re-derive the count for a replay
/// position instead of replaying the capture-position count. The wrapper
/// verifies the caller's count *equals* the declared formula, so the
/// declaration cannot drift from the launch.
#[derive(Clone, Copy, Debug)]
pub struct QsaPositionBinding {
    pub position_start: usize,
    pub rows: usize,
}

pub struct IndexedAttentionPoolRope<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    /// Learned index-key RMSNorm. `None` is reserved for synthetic parity
    /// buffers, which intentionally retain the plain-RMS behavior.
    pub norm: Option<&'a GpuTensor>,
    pub block_count: usize,
    pub compress: usize,
    pub index_dim: usize,
    /// Declared source of `block_count`, or `None` for a synthetic caller whose
    /// count is not position-derived (which then declares nothing).
    pub position: Option<QsaPositionBinding>,
    /// Declared `grid.x` for this launch. A retained tape needs the grid to be
    /// position-independent, so callers pass the capacity the kernel masks
    /// against (`>= block_count`); a caller without a capacity passes the active
    /// count. Must be `>= block_count`.
    pub grid_bound: usize,
}

pub fn indexed_attention_pool_rope(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
) -> HipResult<()> {
    indexed_attention_pool_rope_impl(gpu, p, false)
}

/// [`indexed_attention_pool_rope`] that pools only the blocks the launch's
/// rows complete: those below `position_start / compress` were pooled by an
/// earlier launch from the same raw keys and are left as they are (a declared
/// position is required).
pub fn indexed_attention_pool_rope_incremental(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
) -> HipResult<()> {
    indexed_attention_pool_rope_impl(gpu, p, true)
}

fn indexed_attention_pool_rope_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
    incremental: bool,
) -> HipResult<()> {
    if incremental && p.position.is_none() {
        return Err(HipError::new(
            0,
            "incremental QSA pooling needs a declared position",
        ));
    }
    // Raw and pooled keys share one arena dtype (`QsaKvFormat::index_dtype`).
    let kernel = match (p.raw_keys.dtype, p.pooled.dtype) {
        (DType::F32, DType::F32) => "indexed_attention_pool_rope_f32",
        (DType::BF16, DType::BF16) => "indexed_attention_pool_rope_bf16",
        _ => return Err(HipError::new(0, &ComputeError::WrongDtype.to_string())),
    };
    if p.block_count == 0 || p.compress == 0 || p.index_dim == 0 || p.index_dim % 2 != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if let Some(norm) = p.norm {
        if norm.dtype != DType::BF16 || norm.numel() != p.index_dim {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
    }
    let raw_elements = checked_product3(
        p.block_count,
        p.compress,
        p.index_dim,
        "QSA pool/RoPE raw extent",
    )?;
    let pooled_elements =
        checked_product(p.block_count, p.index_dim, "QSA pool/RoPE pooled extent")?;
    let block_count = checked_i32(p.block_count, "QSA pool/RoPE block count")?;
    let compress = checked_i32(p.compress, "QSA pool/RoPE compression")?;
    let index_dim = checked_i32(p.index_dim, "QSA pool/RoPE index width")?;
    // A declared position source must reproduce the count this launch performs;
    // otherwise replay would re-derive a different amount of pooling.
    if let Some(position) = p.position {
        let expected = position
            .position_start
            .checked_add(position.rows)
            .ok_or_else(|| HipError::new(0, "QSA pool/RoPE position overflow"))?
            / p.compress;
        if expected != p.block_count {
            return Err(HipError::new(
                0,
                &format!(
                    "QSA pool/RoPE block count {} is not (position_start {} + rows {}) / compress {} = {expected}",
                    p.block_count, position.position_start, position.rows, p.compress
                ),
            ));
        }
    }
    // The kernel masks inactive blocks before its first read, so a bound above
    // the active count only skips workgroups; a bound below it would leave work
    // undone and is refused.
    if p.grid_bound < p.block_count {
        return Err(HipError::new(
            0,
            &format!(
                "QSA pool/RoPE grid bound {} is below the active block count {}",
                p.grid_bound, p.block_count
            ),
        ));
    }
    let block_grid = checked_u32(p.grid_bound, "QSA pool/RoPE block grid")?;
    let dim_grid = blocks(p.index_dim)?;
    if p.raw_keys.numel() < raw_elements || p.pooled.numel() < pooled_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_ptr(
        p.norm
            .map(|norm| norm.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut()),
    );
    args.push_i32(block_count);
    // Declared dynamic field: replay re-derives `(position + rows) / compress`
    // rather than replaying the capture-position pooling count.
    let block_count_offset = args.len() - 4;
    args.push_i32(compress);
    args.push_i32(index_dim);
    let first_block = match (incremental, p.position) {
        (true, Some(position)) => position.position_start / p.compress,
        _ => 0,
    };
    args.push_i32(checked_i32(first_block, "QSA pool/RoPE first block")?);
    // Declared dynamic too when incremental: `position_start / compress`.
    let first_block_offset = args.len() - 4;
    args.pad_to(16);
    let addend = u32::try_from(p.position.map_or(0, |position| position.rows))
        .map_err(|_| HipError::new(0, "QSA pool/RoPE rows exceed u32"))?;
    let divisor = u32::try_from(p.compress)
        .map_err(|_| HipError::new(0, "QSA pool/RoPE compression exceeds u32"))?;
    let bindings = [
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: block_count_offset,
            addend,
            divisor,
        },
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: first_block_offset,
            addend: 0,
            divisor,
        },
    ];
    let kernargs: &[crate::replay::ReplayKernargBinding] = match (p.position, incremental) {
        (None, _) => &[],
        (Some(_), false) => &bindings[..1],
        (Some(_), true) => &bindings[..],
    };
    gpu.launch_blob_recorded(
        kernel,
        [block_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs,
        },
    )
}

pub struct IndexedAttentionAttention<'a> {
    pub q_with_gate: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub selected_len: usize,
    pub full_capacity: usize,
    /// K/V cache format.
    pub format: QsaKvFormat,
}

/// One device-to-device copy for [`copy_regions`].
pub struct CopyRegion<'a> {
    pub dst: &'a hip_bridge::DeviceBuffer,
    pub dst_offset: usize,
    pub src: &'a hip_bridge::DeviceBuffer,
    pub src_offset: usize,
    pub bytes: usize,
}

/// Regions per `copy_regions_u32` launch (its by-value kernarg table).
const COPY_REGIONS_MAX: usize = 64;

/// Many small independent device copies in one launch per 64 regions instead
/// of one blit dispatch (and host API call) each. Regions of one call run
/// concurrently, so none may write another's source or destination. A region
/// that is not 4-byte aligned takes a plain copy.
pub fn copy_regions(gpu: &mut Gpu, regions: &[CopyRegion<'_>]) -> HipResult<()> {
    let mut table = [(0u64, 0u64, 0u32); COPY_REGIONS_MAX];
    let mut count = 0;
    for region in regions {
        if region.dst_offset + region.bytes > region.dst.size()
            || region.src_offset + region.bytes > region.src.size()
        {
            return Err(HipError::new(0, "copy_regions: region exceeds its buffer"));
        }
        let dst = region.dst.as_ptr() as u64 + region.dst_offset as u64;
        let src = region.src.as_ptr() as u64 + region.src_offset as u64;
        let words = region.bytes / 4;
        if !(dst | src | region.bytes as u64).is_multiple_of(4) || words > u32::MAX as usize {
            gpu.memcpy_dtod_at_auto(
                region.dst,
                region.dst_offset,
                region.src,
                region.src_offset,
                region.bytes,
            )?;
            continue;
        }
        if words == 0 {
            continue;
        }
        table[count] = (dst, src, words as u32);
        count += 1;
        if count == COPY_REGIONS_MAX {
            launch_copy_regions(gpu, &table[..count])?;
            count = 0;
        }
    }
    if count > 0 {
        launch_copy_regions(gpu, &table[..count])?;
    }
    Ok(())
}

fn launch_copy_regions(gpu: &mut Gpu, table: &[(u64, u64, u32)]) -> HipResult<()> {
    const NAME: &str = "copy_regions_u32";
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, NAME)?;
    let mut args = KernargBlob::new();
    for index in 0..COPY_REGIONS_MAX {
        args.push_u64(table.get(index).map_or(0, |entry| entry.0));
    }
    for index in 0..COPY_REGIONS_MAX {
        args.push_u64(table.get(index).map_or(0, |entry| entry.1));
    }
    for index in 0..COPY_REGIONS_MAX {
        args.push_u32(table.get(index).map_or(0, |entry| entry.2));
    }
    args.push_i32(table.len() as i32);
    args.pad_to(16);
    let max_words = table.iter().map(|entry| entry.2).max().unwrap_or(0);
    gpu.launch_blob_recorded(
        NAME,
        [max_words.div_ceil(256).clamp(1, 64), table.len() as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn indexed_attention_attention(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttention<'_>,
) -> HipResult<()> {
    for tensor in [p.q_with_gate, p.output] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(gpu, p.format, p.n_kv_heads, p.head_dim, [p.full_keys, p.full_values])?;
    if p.selected.dtype != DType::Raw
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.head_dim == 0
        || p.n_heads % p.n_kv_heads != 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA attention head extent")?;
    let q_elements = checked_product(2, head_elements, "QSA attention query extent")?;
    let output_elements = head_elements;
    let full_elements = checked_product(
        p.full_capacity,
        p.format.kv_row_units(p.n_kv_heads, p.head_dim),
        "QSA attention cache extent",
    )?;
    let selected_bytes = p
        .selected_len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let n_heads = checked_i32(p.n_heads, "QSA attention head count")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA attention KV-head count")?;
    let head_dim = checked_i32(p.head_dim, "QSA attention head width")?;
    let selected_len = checked_i32(p.selected_len, "QSA attention selected length")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA attention capacity")?;
    let head_grid = checked_u32(p.n_heads, "QSA attention head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.q_with_gate.numel() != q_elements
        || p.output.numel() != output_elements
        || p.full_keys.numel() < full_elements
        || p.full_values.numel() < p.full_keys.numel()
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let [parallel, serial] = [
        p.format.kernel([
            "indexed_attention_attention_f32",
            "indexed_attention_attention_fp8",
        ]),
        p.format.kernel([
            "indexed_attention_attention_f32_serial",
            "indexed_attention_attention_fp8_serial",
        ]),
    ];
    let (kernel_name, shared_mem) =
        match p.selected_len.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if gpu.arch_caps.has_gfx11_plus_simt()
                    // The kernel adds 32 bytes of static LDS (per-wave maxes).
                    && bytes + 32 <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (parallel, bytes as u32)
            }
            _ => (serial, 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [
        p.q_with_gate,
        p.full_keys,
        p.full_values,
        p.selected,
        p.output,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [n_heads, n_kv_heads, head_dim, selected_len, full_capacity] {
        args.push_i32(value);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel_name,
        [head_grid, dim_grid, 1],
        [QSA_ATTENTION_PARALLEL_THREADS, 1, 1],
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct IndexedAttentionAttentionBatch<'a> {
    pub q_with_gate: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub position_start: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub capacity: usize,
    pub full_capacity: usize,
    /// K/V cache format.
    pub format: QsaKvFormat,
    /// Declared shape bound for the LDS reservation and the kernel symbol. A
    /// retained tape needs both to be position-independent, so callers pass the
    /// selected-row capacity (`>= max_selected`); a caller without a capacity
    /// passes the position-derived length. Must be `>= max_selected`.
    ///
    /// `grid.x` is derived from `n_heads` only (all heads, or head groups of
    /// four for the grouped kernel), so it stays position-independent.
    pub shape_selected: usize,
}

/// Query heads per workgroup in `indexed_attention_attention_f32_batched_hg4`.
const QSA_ATTENTION_HG4_HEADS: usize = 4;
/// Query heads per KV head of `indexed_attention_attention_f32_batched_hg12`
/// (one workgroup per KV group; its scores live in registers, QSA_T = 9
/// tiles of 256 selected rows).
const QSA_ATTENTION_HG12_HEADS: usize = 12;
const QSA_ATTENTION_HG12_MAX_SELECTED: usize = 9 * 256;
/// Few-row verify launches only `rows * n_kv_heads` hg12 workgroups; hg4's
/// three per KV group fill the GPU better there (4-row MTP verify at 32k
/// context: decode 51.4 -> 50.8 tok/s with hg12).
const QSA_ATTENTION_HG12_MIN_ROWS: usize = 16;

/// Below this many rows the grouped kernel launches too few workgroups
/// (`rows * n_heads / 4`) to fill the GPU and the per-head kernel is faster
/// (decode: 0.48 vs 0.73 ms per call at 1131 context on gfx1151). From two
/// rows (speculative verify) sharing each K/V row across four heads wins
/// (4 rows, 232 context: 55 -> 27 us per layer).
const QSA_ATTENTION_HG4_MIN_ROWS: usize = 2;

/// Dynamic LDS for the grouped kernel, or `None` when the shape is not the one
/// it is specialised for (gfx11+ ISA, head_dim 256, four heads per KV group,
/// prefill-sized row counts).
fn qsa_attention_hg4_lds_bytes(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    shape_selected: usize,
) -> Option<u32> {
    let group = p.n_heads / p.n_kv_heads;
    if !gpu.arch_caps.has_gfx11_plus_simt()
        || p.head_dim != 256
        || group % QSA_ATTENTION_HG4_HEADS != 0
        || p.rows < QSA_ATTENTION_HG4_MIN_ROWS
    {
        return None;
    }
    // weights[sel][4] + tokens[sel] + partial maxes[4][256] + maxes[4].
    let bytes = shape_selected
        .checked_mul(4 * QSA_ATTENTION_HG4_HEADS + 4)?
        .checked_add(4 * (QSA_ATTENTION_HG4_HEADS * 256 + QSA_ATTENTION_HG4_HEADS))?;
    (bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES).then_some(bytes as u32)
}

pub fn indexed_attention_attention_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
) -> HipResult<()> {
    indexed_attention_attention_batch_impl(gpu, p, QsaAttentionRoutes::F16)
}

/// Kernels [`indexed_attention_attention_batch_impl`] may pick from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum QsaAttentionRoutes {
    /// The per-head kernels (the exact reference).
    Exact,
    /// Plus the bitwise-equal grouped hg12/hg4 kernels (the tests' reference
    /// for those kernels; production always admits F16).
    #[cfg_attr(not(test), allow(dead_code))]
    Grouped,
    /// Plus the dense and sparse F16 WMMA routes (KLD-gated).
    F16,
}

/// Developer-only exact reference for the QSA oracle: the same
/// `IndexedAttentionAttentionBatch` through the per-head kernels
/// ([`QsaAttentionRoutes::Exact`]), bypassing the grouped hg4/hg12 and the
/// dense/sparse/gathered WMMA routes. Not a production route; Unit0 harness
/// only. Gated behind `lab` (or test) so production builds expose no new API.
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn indexed_attention_attention_batch_exact(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
) -> HipResult<()> {
    indexed_attention_attention_batch_impl(gpu, p, QsaAttentionRoutes::Exact)
}

/// The gathered route's launches for `p` on the hipcc kernels or, with `pm`,
/// the builder module (`HIPFIRE_QWEN4_QSA_PM`), whatever the route flags say.
/// Byte-equality and timing harness only (`examples/qsa_pm_check.rs`).
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn indexed_attention_gathered_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    pm: bool,
) -> HipResult<()> {
    let end_position = p.position_start + p.rows;
    let max_selected = end_position
        .min(p.capacity)
        .min(p.budget_blocks * p.compress + p.compress - 1);
    if pm && !qsa_gathered_pm_fits(p) {
        return Err(HipError::new(0, "QSA PM gathered route does not cover this shape"));
    }
    qsa_gathered_wmma_launch(gpu, p, end_position, max_selected, pm)
}
fn indexed_attention_attention_batch_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    routes: QsaAttentionRoutes,
) -> HipResult<()> {
    let allow_fast = routes != QsaAttentionRoutes::Exact;
    let allow_f16 = routes == QsaAttentionRoutes::F16;
    for tensor in [p.q_with_gate, p.output] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(gpu, p.format, p.n_kv_heads, p.head_dim, [p.full_keys, p.full_values])?;
    if p.selected.dtype != DType::Raw
        || p.rows == 0
        || p.position_start.checked_add(p.rows).is_none()
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.head_dim == 0
        || p.n_heads % p.n_kv_heads != 0
        || p.compress == 0
        || p.capacity == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA batch attention heads")?;
    let q_elements = checked_product(2, head_elements, "QSA batch attention query")?;
    let kv_elements = checked_product(
        p.full_capacity,
        p.format.kv_row_units(p.n_kv_heads, p.head_dim),
        "QSA batch attention cache",
    )?;
    let q_rows = checked_product(p.rows, q_elements, "QSA batch attention query rows")?;
    let output_rows = checked_product(p.rows, head_elements, "QSA batch attention output rows")?;
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA batch attention selected rows")?,
        std::mem::size_of::<i32>(),
        "QSA batch attention selected",
    )?;
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if end_position > i32::MAX as usize
        || end_position > p.full_capacity
        || p.q_with_gate.numel() < q_rows
        || p.output.numel() < output_rows
        || p.full_keys.numel() < kv_elements
        || p.full_values.numel() < p.full_keys.numel()
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let selected_bound = checked_product(
        p.budget_blocks,
        p.compress,
        "QSA batch attention selected bound",
    )?
    .checked_add(p.compress - 1)
    .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let max_selected = end_position.min(p.capacity).min(selected_bound);
    if allow_f16 && qsa_dense_wmma_applies(gpu, p, end_position) {
        return qsa_dense_wmma(gpu, p, end_position);
    }
    // Shape bound, not active length: the LDS reservation and symbol are what
    // must be position-independent. The kernel derives its own active
    // `selected_len` from the scalars, and the reservation is never read past
    // that length.
    if p.shape_selected < max_selected {
        return Err(HipError::new(
            0,
            &format!(
                "QSA batch attention shape bound {} is below the active selected length {max_selected}",
                p.shape_selected
            ),
        ));
    }
    // Gathered first: on the F32 state it covers every row count the sparse
    // route takes, at equal KLD and no slower; sparse remains the route on
    // other gfx11 parts and with HIPFIRE_QWEN4_QSA_WMMA_GATHER=0.
    if allow_f16 && qsa_gathered_wmma_applies(gpu, p) {
        return qsa_gathered_wmma(gpu, p, end_position, max_selected);
    }
    if allow_f16 && qsa_sparse_wmma_applies(gpu, p) {
        return qsa_sparse_wmma(gpu, p, end_position);
    }
    // Live (unrecorded) launches reserve LDS for this chunk's longest row
    // only; a recorded launch keeps the position-independent shape bound.
    let shape_selected = if gpu.replay.is_recording() || gpu.graphs.capture_mode {
        p.shape_selected
    } else {
        max_selected
    };
    let rows = checked_i32(p.rows, "QSA batch attention rows")?;
    let position_start = checked_i32(p.position_start, "QSA batch attention position")?;
    let n_heads = checked_i32(p.n_heads, "QSA batch attention heads")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA batch attention KV heads")?;
    let head_dim = checked_i32(p.head_dim, "QSA batch attention head width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA batch attention budget")?;
    let compress = checked_i32(p.compress, "QSA batch attention compress")?;
    let capacity = checked_i32(p.capacity, "QSA batch attention capacity")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA batch attention cache capacity")?;
    let row_grid = checked_u32(p.rows, "QSA batch attention row grid")?;
    let hg12 = allow_fast
        && p.format == QsaKvFormat::F32
        && gpu.arch_caps.has_gfx11_plus_simt()
        && p.head_dim == 256
        && p.n_heads == p.n_kv_heads * QSA_ATTENTION_HG12_HEADS
        && p.rows >= QSA_ATTENTION_HG12_MIN_ROWS
        && shape_selected <= QSA_ATTENTION_HG12_MAX_SELECTED;
    let hg4_bytes = (allow_fast && !hg12)
        .then(|| qsa_attention_hg4_lds_bytes(gpu, p, shape_selected))
        .flatten();
    let (kernel_name, grid, shared_mem) = if hg12 {
        (
            "indexed_attention_attention_f32_batched_hg12",
            [
                checked_u32(p.n_kv_heads, "QSA batch attention KV head grid")?,
                1,
                row_grid,
            ],
            0,
        )
    } else if let Some(bytes) = hg4_bytes {
        (
            p.format.kernel([
                "indexed_attention_attention_f32_batched_hg4",
                "indexed_attention_attention_fp8_batched_hg4",
            ]),
            [
                checked_u32(
                    p.n_heads / QSA_ATTENTION_HG4_HEADS,
                    "QSA batch attention head grid",
                )?,
                1,
                row_grid,
            ],
            bytes,
        )
    } else {
        let head_grid = checked_u32(p.n_heads, "QSA batch attention head grid")?;
        let dim_grid = blocks(p.head_dim)?;
        match shape_selected.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if gpu.arch_caps.has_gfx11_plus_simt()
                    && shape_selected > 0
                    // The kernel adds 32 bytes of static LDS (per-wave maxes).
                    && bytes + 32 <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (
                    p.format.kernel([
                        "indexed_attention_attention_f32_batched",
                        "indexed_attention_attention_fp8_batched",
                    ]),
                    [head_grid, dim_grid, row_grid],
                    bytes as u32,
                )
            }
            _ => (
                p.format.kernel([
                    "indexed_attention_attention_f32_batched_serial",
                    "indexed_attention_attention_fp8_batched_serial",
                ]),
                [head_grid, dim_grid, row_grid],
                0,
            ),
        }
    };
    let block = [QSA_ATTENTION_PARALLEL_THREADS, 1, 1];
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [
        p.q_with_gate,
        p.full_keys,
        p.full_values,
        p.selected,
        p.output,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it,
    // so the attention window follows the replay position.
    let position_offset = args.len() - 4;
    for value in [
        n_heads,
        n_kv_heads,
        head_dim,
        budget_blocks,
        compress,
        capacity,
        full_capacity,
    ] {
        args.push_i32(value);
    }
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    gpu.launch_blob_recorded(
        kernel_name,
        grid,
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )
}

/// Whether the dense F16 WMMA attention applies: the Qwen4 F16 route (gfx11 WMMA,
/// >= QWEN4_F16_WMMA_MIN_TOKENS rows, no recorder or capture, not opted out),
/// head_dim 256 in four-head KV groups, and every row's selection is its
/// whole causal window: the budget covers every visible block and the
/// capacity every visible token (indexed_attention_select then emits all of
/// them). Only F32 caches take this route (fp8 is gfx12-only).
fn qsa_dense_wmma_applies(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
) -> bool {
    gpu.arch_caps.has_wmma_w32()
        && p.rows >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.head_dim == 256
        && p.format == QsaKvFormat::F32
        && (p.n_heads / p.n_kv_heads) % 4 == 0
        && end_position / p.compress <= p.budget_blocks
        && end_position <= p.capacity
}

/// Whether the sparse F16 WMMA attention applies: the Qwen4 F16 route with
/// prefill-sized row counts, F32 caches, head_dim 256 in twelve-head KV groups
/// and compress-4 blocks (rows the dense route does not cover).
fn qsa_sparse_wmma_applies(gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>) -> bool {
    gpu.arch_caps.has_wmma_w32()
        && p.format == QsaKvFormat::F32
        && p.rows >= QSA_ATTENTION_HG12_MIN_ROWS
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.head_dim == 256
        && p.n_heads == p.n_kv_heads * QSA_ATTENTION_HG12_HEADS
        && p.compress == 4
}

/// Per-row selected-key attention in F16 WMMA
/// (kernels/src/indexed_attention_dense_wmma.gfx1151.hip,
/// `indexed_attention_sparse_wmma_f16`).  The F16 K and block-transposed V
/// copies of cache rows `[0, end_position)` live in the shared FP16 X scratch.
fn qsa_sparse_wmma(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
) -> HipResult<()> {
    let width = checked_product(p.n_kv_heads, 256, "QSA sparse KV width")?;
    let blocks = end_position.div_ceil(4);
    let k_elements = checked_product(end_position, width, "QSA sparse K")?;
    let v_elements = checked_product(blocks * 4, width, "QSA sparse V")?;
    let scratch = gpu.qwen4_f16_x_scratch(k_elements + v_elements)?;
    let k16 = scratch.buf.as_ptr();
    let v4t = unsafe { (k16 as *mut u8).add(k_elements * 2) } as *mut std::ffi::c_void;
    for kernel in [
        "indexed_attention_kv_f16_sparse",
        "indexed_attention_sparse_wmma_f16",
    ] {
        gpu.ensure_kernel_public(
            "indexed_attention_dense_wmma",
            INDEXED_ATTENTION_DENSE_WMMA_SRC,
            kernel,
        )?;
    }
    let kv_heads = checked_i32(p.n_kv_heads, "QSA sparse KV heads")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.full_keys.buf.as_ptr());
    args.push_ptr(p.full_values.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(v4t);
    args.push_i32(checked_i32(end_position, "QSA sparse tokens")?);
    args.push_i32(kv_heads);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_kv_f16_sparse",
        [
            checked_u32(blocks, "QSA sparse block grid")?,
            p.n_kv_heads as u32,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.q_with_gate.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(v4t);
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(checked_i32(p.rows, "QSA sparse rows")?);
    args.push_i32(checked_i32(p.position_start, "QSA sparse position")?);
    args.push_i32(checked_i32(p.n_heads, "QSA sparse heads")?);
    args.push_i32(kv_heads);
    args.push_i32(checked_i32(p.budget_blocks, "QSA sparse budget")?);
    args.push_i32(checked_i32(p.capacity, "QSA sparse capacity")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_sparse_wmma_f16",
        [
            checked_u32(p.rows.div_ceil(2), "QSA sparse row grid")?,
            p.n_kv_heads as u32,
            1,
        ],
        [128, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Causal GQA flash attention over cache rows `[0, end_position)` in F16
/// WMMA (kernels/src/indexed_attention_dense_wmma.gfx1151.hip).  The F16 K
/// and V^T copies live in the shared FP16 X scratch.
fn qsa_dense_wmma(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
) -> HipResult<()> {
    let width = checked_product(p.n_kv_heads, 256, "QSA dense KV width")?;
    let tpad = end_position.div_ceil(32) * 32;
    let k_elements = checked_product(end_position, width, "QSA dense K")?;
    let vt_elements = checked_product(width, tpad, "QSA dense V")?;
    let scratch = gpu.qwen4_f16_x_scratch(k_elements + vt_elements)?;
    let k16 = scratch.buf.as_ptr();
    let vt16 = unsafe { (k16 as *mut u8).add(k_elements * 2) } as *mut std::ffi::c_void;
    let tokens = checked_i32(end_position, "QSA dense tokens")?;
    let kv_heads = checked_i32(p.n_kv_heads, "QSA dense KV heads")?;
    let tpad_i = checked_i32(tpad, "QSA dense padded tokens")?;
    for kernel in [
        "indexed_attention_kv_f16",
        "indexed_attention_dense_wmma_f16",
    ] {
        gpu.ensure_kernel_public(
            "indexed_attention_dense_wmma",
            INDEXED_ATTENTION_DENSE_WMMA_SRC,
            kernel,
        )?;
    }
    let mut args = KernargBlob::new();
    args.push_ptr(p.full_keys.buf.as_ptr());
    args.push_ptr(p.full_values.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vt16);
    args.push_i32(tokens);
    args.push_i32(kv_heads);
    args.push_i32(tpad_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_kv_f16",
        [
            checked_u32(tpad, "QSA dense token grid")?,
            p.n_kv_heads as u32,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.q_with_gate.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vt16);
    args.push_i32(tpad_i);
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(checked_i32(p.rows, "QSA dense rows")?);
    args.push_i32(checked_i32(p.position_start, "QSA dense position")?);
    args.push_i32(checked_i32(p.n_heads, "QSA dense heads")?);
    args.push_i32(kv_heads);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_dense_wmma_f16",
        [
            checked_u32(p.rows.div_ceil(16), "QSA dense row grid")?,
            checked_u32(p.n_heads / 4, "QSA dense head grid")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Static LDS of the gathered kernels: F16 Q (16 heads x 256), two parity
/// sets of F16 probabilities (8 waves x 16 x 16) and of the per-wave max/sum.
const QSA_GATHERED_STATIC_LDS_BYTES: usize = 16 * 256 * 2 + 2 * 8 * 256 * 2 + 2 * 2 * 8 * 16 * 4;

/// Whether this process runs QSA prefill attention in `format` on the
/// gathered F16 WMMA route: `HIPFIRE_QWEN4_QSA_WMMA_GATHER` not `0` (read
/// first, so with it `0` nothing else is consulted), the Qwen4 F16 route not opted
/// out, and an arch with a kernel for the state format (gfx1151: F32,
/// gfx1201: fp8).  Loaders use it to reserve and account the route's scratch.
pub fn qsa_gathered_wmma_enabled(gpu: &Gpu, format: QsaKvFormat) -> bool {
    *QWEN4_QSA_WMMA_GATHER
        && *crate::gemm::QWEN4_F16_WMMA
        && match format {
            QsaKvFormat::F32 => gpu.arch_caps.is_gfx1151(),
            QsaKvFormat::Fp8 => gpu.arch_caps.is_gfx1201(),
        }
}

/// Bytes of the gathered route's F16 scratch for a QSA cache of `tokens`
/// rows with `n_kv_heads` heads of 256 channels: F16 K plus block-transposed
/// V (whole four-token blocks), 2 bytes per KV channel each.
pub fn qsa_gathered_wmma_scratch_bytes(n_kv_heads: usize, tokens: usize) -> Option<usize> {
    let width = n_kv_heads.checked_mul(256)?;
    tokens
        .checked_add(tokens.div_ceil(4).checked_mul(4)?)?
        .checked_mul(width)?
        .checked_mul(2)
}

/// Reserve the gathered route's scratch for a QSA cache of `tokens` rows
/// when the route is enabled for `format`; returns the bytes reserved (0 when
/// it is not enabled, and nothing is allocated).  Call at load, before any
/// graph capture or Redline record: a forward only needs the rows below its
/// end position, which never exceed the cache capacity, so the scratch then
/// never grows under a captured graph or tape.
pub fn reserve_qsa_gathered_wmma_scratch(
    gpu: &mut Gpu,
    format: QsaKvFormat,
    n_kv_heads: usize,
    tokens: usize,
) -> HipResult<usize> {
    if !qsa_gathered_wmma_enabled(gpu, format) {
        return Ok(0);
    }
    let bytes = qsa_gathered_wmma_scratch_bytes(n_kv_heads, tokens)
        .ok_or_else(|| HipError::new(0, "QSA gathered scratch size overflows"))?;
    gpu.qsa_gather_scratch(bytes)?;
    Ok(bytes)
}

/// Whether the gathered F16 WMMA attention applies (checked after the dense
/// route): the route is enabled for the state format
/// ([`qsa_gathered_wmma_enabled`]), enough rows, no recorder or capture,
/// head_dim 256 with at most 16 query heads per KV head, and a token list
/// that fits the LDS budget.  With `HIPFIRE_QWEN4_QSA_WMMA_GATHER=0` every
/// launch is the incumbent's.
fn qsa_gathered_wmma_applies(gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>) -> bool {
    let min_rows = match p.format {
        // gfx1151: also the sparse route's 16..511-row chunks (prompt-cache
        // suffixes, short tails). Against the exact kernels, gathered KLD
        // matched sparse (paired over 10 samples at 4K-16K context) and each
        // chunk ran 0-2% faster.
        QsaKvFormat::F32 => QSA_ATTENTION_HG12_MIN_ROWS,
        // gfx1201 has no sparse route; its small chunks keep the exact kernels.
        QsaKvFormat::Fp8 => crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS,
    };
    qsa_gathered_wmma_enabled(gpu, p.format)
        && p.rows >= min_rows
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.head_dim == 256
        && p.n_heads / p.n_kv_heads <= 16
        && p.capacity.div_ceil(128) * 128 * 4 + QSA_GATHERED_STATIC_LDS_BYTES
            <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
}

/// QSA attention over each row's selection in F16 WMMA
/// (kernels/src/indexed_attention_gathered_wmma.gfx{1151,1201}.hip).  The
/// cache rows `[0, end_position)` are first written as F16 K and
/// block-transposed V into the route's own scratch
/// ([`reserve_qsa_gathered_wmma_scratch`]; 2 KiB per token at 2 x 256 KV
/// channels).
fn qsa_gathered_wmma(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
    max_selected: usize,
) -> HipResult<()> {
    let pm = *QWEN4_QSA_PM && qsa_gathered_pm_fits(p);
    qsa_gathered_wmma_launch(gpu, p, end_position, max_selected, pm)
}

/// Whether the builder module covers `p`: its raw buffer offsets are 32-bit,
/// so the F16 K/V scratch over the whole cache capacity and the query row
/// stay below its out-of-range offset.
fn qsa_gathered_pm_fits(p: &IndexedAttentionAttentionBatch<'_>) -> bool {
    const PM_OOB_OFFSET: usize = 0x7fff_ff00;
    let scratch = p.full_capacity.div_ceil(4).checked_mul(4).and_then(|t| t.checked_mul(p.n_kv_heads * 512));
    scratch.is_some_and(|bytes| bytes + 4096 <= PM_OOB_OFFSET)
        && p.n_heads.checked_mul(2048).is_some_and(|bytes| bytes + 4096 <= PM_OOB_OFFSET)
        // `v_mad_u32_u24` token and stride operands.
        && p.full_capacity < 1 << 24
        && p.n_kv_heads * 2048 < 1 << 24
}

/// The gathered route's producer and attention launches: the hipcc kernels,
/// or with `pm` the certified builder module (`kernels::QSA_GATHER_PM_*`).
fn qsa_gathered_wmma_launch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
    max_selected: usize,
    pm: bool,
) -> HipResult<()> {
    let (module, src, convert, attend) = match (p.format, pm) {
        (QsaKvFormat::F32, false) => (
            "indexed_attention_gathered_wmma",
            INDEXED_ATTENTION_GATHERED_WMMA_SRC,
            "indexed_attention_kv_f16vb",
            "indexed_attention_gathered_wmma_f16",
        ),
        (QsaKvFormat::Fp8, false) => (
            "indexed_attention_gathered_wmma_gfx1201",
            INDEXED_ATTENTION_GATHERED_WMMA_GFX1201_SRC,
            "indexed_attention_kv_f16vb_fp8_gfx1201",
            "indexed_attention_gathered_wmma_f16_gfx1201",
        ),
        (QsaKvFormat::F32, true) => (
            "qsa_gather_pm_gfx1151",
            "",
            "indexed_attention_kv_f16vb_pm_gfx1151",
            "indexed_attention_gathered_wmma_f16_pm_gfx1151",
        ),
        (QsaKvFormat::Fp8, true) => (
            "qsa_gather_pm_gfx1201",
            "",
            "indexed_attention_kv_f16vb_fp8_pm_gfx1201",
            "indexed_attention_gathered_wmma_f16_pm_gfx1201",
        ),
    };
    let width = checked_product(p.n_kv_heads, 256, "QSA gathered KV width")?;
    let k_elements = checked_product(end_position, width, "QSA gathered K")?;
    let bytes = qsa_gathered_wmma_scratch_bytes(p.n_kv_heads, end_position)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let k16 = gpu.qsa_gather_scratch(bytes)?;
    // V follows K: whole four-token blocks, the last one padded.
    let vb16 = unsafe { (k16 as *mut u8).add(k_elements * 2) } as *mut std::ffi::c_void;
    let tokens = checked_i32(end_position, "QSA gathered tokens")?;
    let kv_heads = checked_i32(p.n_kv_heads, "QSA gathered KV heads")?;
    for kernel in [convert, attend] {
        if pm {
            let image = match p.format {
                QsaKvFormat::F32 => crate::kernels::QSA_GATHER_PM_GFX1151,
                QsaKvFormat::Fp8 => crate::kernels::QSA_GATHER_PM_GFX1201,
            };
            gpu.ensure_embedded_kernel(module, image, kernel)?;
        } else {
            gpu.ensure_kernel_public(module, src, kernel)?;
        }
    }
    let mut args = KernargBlob::new();
    args.push_ptr(p.full_keys.buf.as_ptr());
    args.push_ptr(p.full_values.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vb16);
    args.push_i32(tokens);
    args.push_i32(kv_heads);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        convert,
        [checked_u32(end_position, "QSA gathered token grid")?, p.n_kv_heads as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.q_with_gate.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vb16);
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    for (value, what) in [
        (p.rows, "QSA gathered rows"),
        (p.position_start, "QSA gathered position"),
        (p.n_heads, "QSA gathered heads"),
        (p.n_kv_heads, "QSA gathered KV heads"),
        (p.budget_blocks, "QSA gathered budget"),
        (p.compress, "QSA gathered compress"),
        (p.capacity, "QSA gathered capacity"),
        (p.full_capacity, "QSA gathered cache capacity"),
    ] {
        args.push_i32(checked_i32(value, what)?);
    }
    args.pad_to(16);
    // The validated token list of the longest selection, in 128-entry tiles.
    let lds = checked_u32(max_selected.div_ceil(128) * 128 * 4, "QSA gathered LDS")?;
    gpu.launch_blob_recorded(
        attend,
        [
            checked_u32(p.rows, "QSA gathered row grid")?,
            checked_u32(p.n_kv_heads, "QSA gathered head grid")?,
            1,
        ],
        [256, 1, 1],
        lds,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct IndexedAttentionPool<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub block_count: usize,
    pub head_dim: usize,
}

pub fn indexed_attention_pool(gpu: &mut Gpu, p: &IndexedAttentionPool<'_>) -> HipResult<()> {
    ensure_f32(p.raw_keys)?;
    ensure_f32(p.pooled)?;
    if p.block_count == 0 || p.head_dim == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let raw_elements = checked_product3(p.block_count, 4, p.head_dim, "QSA pool raw extent")?;
    let pooled_elements = checked_product(p.block_count, p.head_dim, "QSA pool output extent")?;
    let block_count = checked_i32(p.block_count, "QSA pool block count")?;
    let head_dim = checked_i32(p.head_dim, "QSA pool head width")?;
    let block_grid = checked_u32(p.block_count, "QSA pool block grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.raw_keys.numel() < raw_elements || p.pooled.numel() < pooled_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "indexed_attention_pool_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_i32(block_count);
    args.push_i32(head_dim);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "indexed_attention_pool_f32",
        [block_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// In-place scalar scaling for learned HC branch projections.
pub struct ScaleF32<'a> {
    pub values: &'a GpuTensor,
    pub scale: f32,
}

pub fn scale_f32(gpu: &mut Gpu, p: &ScaleF32<'_>) -> HipResult<()> {
    ensure_f32(p.values)?;
    let elements = checked_extent(p.values.numel(), "scale extent")?;
    if elements == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "scale extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "scale_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_i32(elements_i);
    args.push_f32(p.scale);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "scale_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Device-side greedy top-1 over contiguous F32 logits rows.
pub struct ArgmaxF32<'a> {
    pub logits: &'a GpuTensor,
    pub indices: &'a GpuTensor,
    pub rows: usize,
    pub vocab: usize,
}

pub fn argmax_f32(gpu: &mut Gpu, p: &ArgmaxF32<'_>) -> HipResult<()> {
    ensure_f32(p.logits)?;
    if p.rows == 0 || p.vocab == 0 || p.indices.dtype != DType::Raw {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let logits_elements = checked_product(p.rows, p.vocab, "argmax logits extent")?;
    let indices_bytes = p
        .rows
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let rows = checked_i32(p.rows, "argmax row count")?;
    let vocab = checked_i32(p.vocab, "argmax vocabulary width")?;
    let row_grid = checked_u32(p.rows, "argmax row grid")?;
    if p.logits.numel() != logits_elements || p.indices.numel() < indices_bytes {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "argmax_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.logits.buf.as_ptr());
    args.push_ptr(p.indices.buf.as_ptr());
    args.push_i32(rows);
    args.push_i32(vocab);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "argmax_f32",
        [row_grid, 1, 1],
        [1024, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// [`argmax_f32`] of one logits row, read back: `llama::argmax(logits)`
/// without downloading the row. Uses a lazily allocated 4-byte scratch.
pub fn argmax_f32_host(gpu: &mut Gpu, logits: &GpuTensor) -> HipResult<u32> {
    if gpu.scratch.argmax_host.is_none() {
        gpu.scratch.argmax_host = Some(gpu.alloc_tensor(&[1], DType::F32)?);
    }
    let result = gpu.scratch.argmax_host.as_ref().unwrap().sub_offset(0, 1);
    let mut indices = result.sub_offset(0, 1);
    indices.dtype = DType::Raw;
    indices.shape = vec![4];
    argmax_f32(
        gpu,
        &ArgmaxF32 {
            logits,
            indices: &indices,
            rows: 1,
            vocab: logits.numel(),
        },
    )?;
    Ok(gpu.download_f32(&result)?[0].to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok()
    }

    /// The batched gated RMSNorm (with its folded BF16 recurrent rounding)
    /// must equal round-trip + per-row gate bit for bit, for both the gated
    /// output and the in-place rounded recurrent buffer.
    #[test]
    fn gdn_gate_batch_is_bit_identical_to_per_row_gate() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (rows, heads, dim) = (5usize, 6usize, 128usize);
        let width = heads * dim;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let recurrent = wave(1, rows * width, 3.0);
        let z = wave(2, rows * width, 4.0);
        let norm_bits: Vec<u8> = wave(3, dim, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16 + 0x3f00).to_le_bytes())
            .collect();
        let mut norm = gpu
            .upload_raw(&norm_bits, &[norm_bits.len()])
            .expect("norm");
        norm.dtype = DType::BF16;
        norm.shape = vec![dim];
        let z_gpu = gpu.upload_f32(&z, &[z.len()]).expect("z");

        let rec_a = gpu.upload_f32(&recurrent, &[recurrent.len()]).expect("rec");
        let out_a = gpu.zeros(&[rows * width], DType::F32).expect("out");
        gated_delta_gate_batched(
            &mut gpu,
            &GatedDeltaGateBatched {
                recurrent_output: &rec_a,
                z: &z_gpu,
                norm: &norm,
                output: &out_a,
                rows,
                value_heads: heads,
                value_dim: dim,
            },
        )
        .expect("batched gate");

        let rec_b = gpu.upload_f32(&recurrent, &[recurrent.len()]).expect("rec");
        let out_b = gpu.zeros(&[rows * width], DType::F32).expect("out");
        gpu.bf16_round_trip_f32(&rec_b).expect("round trip");
        for row in 0..rows {
            gated_delta_gate(
                &mut gpu,
                &GatedDeltaGate {
                    recurrent_output: &rec_b.sub_offset(row * width, width),
                    z: &z_gpu.sub_offset(row * width, width),
                    norm: &norm,
                    output: &out_b.sub_offset(row * width, width),
                    value_heads: heads,
                    value_dim: dim,
                },
            )
            .expect("per-row gate");
        }
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &out_a), bits(&gpu, &out_b));
        assert!(a.iter().any(|v| *v != 0), "gate output is all zero");
        assert_eq!(a, b, "batched gate output differs");
        assert_eq!(
            bits(&gpu, &rec_a),
            bits(&gpu, &rec_b),
            "rounded recurrent differs"
        );
        for tensor in [norm, z_gpu, rec_a, out_a, rec_b, out_b] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The row-parallel batched convolution must equal the per-row ring kernel
    /// run row by row (in-place history), for outputs and the final history,
    /// across chunk boundaries, short batches and every start cursor.
    #[test]
    fn gdn_conv_batch_is_bit_identical_to_per_row_conv() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let channels = 300usize;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let kernel_bits: Vec<u8> = wave(1, channels * 4, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut kernel = gpu
            .upload_raw(&kernel_bits, &[kernel_bits.len()])
            .expect("kernel");
        kernel.dtype = DType::BF16;
        kernel.shape = vec![channels * 4];
        let history = wave(2, channels * 3, 2.0);
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for (rows, start_cursor) in [(1usize, 0usize), (2, 2), (3, 1), (17, 0), (37, 2), (40, 1)] {
            let input = gpu
                .upload_f32(&wave(3 + rows, rows * channels, 3.0), &[rows * channels])
                .expect("input");
            let hist_a = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let out_a = gpu.zeros(&[rows * channels], DType::F32).expect("out");
            gated_delta_conv_batched(
                &mut gpu,
                &GatedDeltaConvBatched {
                    input: &input,
                    kernel: &kernel,
                    history: &hist_a,
                    output: &out_a,
                    next_history: &hist_a,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor,
                },
            )
            .expect("batched conv");
            let hist_b = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let out_b = gpu.zeros(&[rows * channels], DType::F32).expect("out");
            for row in 0..rows {
                gated_delta_conv(
                    &mut gpu,
                    &GatedDeltaConv {
                        input: &input.sub_offset(row * channels, channels),
                        kernel: &kernel,
                        history: &hist_b,
                        output: &out_b.sub_offset(row * channels, channels),
                        next_history: &hist_b,
                        channels,
                        history_rows: 3,
                        kernel_size: 4,
                        cursor: (start_cursor + row) % 3,
                        row_index: row,
                    },
                )
                .expect("per-row conv");
            }
            assert_eq!(
                bits(&gpu, &out_a),
                bits(&gpu, &out_b),
                "rows {rows}: output"
            );
            assert_eq!(
                bits(&gpu, &hist_a),
                bits(&gpu, &hist_b),
                "rows {rows}: history"
            );
            // BF16 in and out with the gate parameters fused (the four-channel
            // kernel): the per-row outputs' BF16 bits, and the standalone
            // parameter kernel's gate and beta.
            let rne = |v: f32| -> [u8; 2] {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            };
            let in_bits: Vec<u8> = wave(3 + rows, rows * channels, 3.0)
                .into_iter()
                .flat_map(rne)
                .collect();
            let mut input_bf16 = gpu
                .upload_raw(&in_bits, &[in_bits.len()])
                .expect("bf16 input");
            input_bf16.dtype = DType::BF16;
            input_bf16.shape = vec![rows * channels];
            let mut out_c = gpu
                .zeros(&[rows * channels / 2], DType::F32)
                .expect("bf16 out");
            out_c.dtype = DType::BF16;
            out_c.shape = vec![rows * channels];
            let hist_c = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let heads = 5usize;
            let pa = gpu
                .upload_f32(&wave(7, rows * heads, 4.0), &[rows * heads])
                .expect("a");
            let pb = gpu
                .upload_f32(&wave(8, rows * heads, 4.0), &[rows * heads])
                .expect("b");
            let head_bits: Vec<u8> = wave(9, 2 * heads, 1.0).into_iter().flat_map(rne).collect();
            let mut a_log = gpu
                .upload_raw(&head_bits[..2 * heads], &[2 * heads])
                .expect("a_log");
            a_log.dtype = DType::BF16;
            a_log.shape = vec![heads];
            let mut dt_bias = gpu
                .upload_raw(&head_bits[2 * heads..], &[2 * heads])
                .expect("dt");
            dt_bias.dtype = DType::BF16;
            dt_bias.shape = vec![heads];
            let [gate_c, beta_c, gate_d, beta_d] =
                [(); 4].map(|()| gpu.zeros(&[rows * heads], DType::F32).expect("params"));
            let params = |gate, beta| GatedDeltaParamsBatched {
                a: &pa,
                b: &pb,
                a_log: &a_log,
                dt_bias: &dt_bias,
                gate,
                beta,
                rows,
                heads,
            };
            gated_delta_conv_params_batched(
                &mut gpu,
                &GatedDeltaConvBatched {
                    input: &input_bf16,
                    kernel: &kernel,
                    history: &hist_c,
                    output: &out_c,
                    next_history: &hist_c,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor,
                },
                &params(&gate_c, &beta_c),
            )
            .expect("bf16 conv");
            gated_delta_params_batched(&mut gpu, &params(&gate_d, &beta_d)).expect("params");
            out_c.dtype = DType::F32;
            out_c.shape = vec![rows * channels / 2];
            let got: Vec<u32> = bits(&gpu, &out_c)
                .iter()
                .flat_map(|w| [w & 0xFFFF, w >> 16])
                .collect();
            let want: Vec<u32> = bits(&gpu, &out_b).iter().map(|w| w >> 16).collect();
            assert_eq!(got, want, "rows {rows}: bf16 output");
            assert_eq!(
                bits(&gpu, &hist_c),
                bits(&gpu, &hist_b),
                "rows {rows}: bf16 history"
            );
            assert_eq!(
                bits(&gpu, &gate_c),
                bits(&gpu, &gate_d),
                "rows {rows}: gate"
            );
            assert_eq!(
                bits(&gpu, &beta_c),
                bits(&gpu, &beta_d),
                "rows {rows}: beta"
            );
            for tensor in [
                input, hist_a, out_a, hist_b, out_b, input_bf16, out_c, hist_c, pa, pb, a_log,
                dt_bias, gate_c, beta_c, gate_d, beta_d,
            ] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        gpu.free_tensor(kernel).expect("free");
    }

    /// The fused HC norm + BF16 gate projection must equal hyper_norm followed
    /// by the multi-row BF16 GEMM bit for bit, at the production width, and
    /// its F16 read output must equal hyper_norm_f16 of the same streams.
    #[test]
    fn hyper_norm_gate_is_bit_identical_to_norm_then_gemm() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (rows, branches, hidden) = (37usize, 4usize, 2560usize);
        let wide = branches * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bf16 = |gpu: &mut Gpu, values: &[f32]| -> GpuTensor {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut tensor = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16");
            tensor.dtype = DType::BF16;
            tensor.shape = vec![values.len()];
            tensor
        };
        let input = gpu
            .upload_f32(&wave(1, rows * wide, 3.0), &[rows * wide])
            .expect("input");
        let norm = bf16(&mut gpu, &wave(2, wide, 0.5));
        let gate_weight = bf16(&mut gpu, &wave(3, branches * wide, 0.05));
        let fused = gpu.zeros(&[rows * branches], DType::F32).expect("fused");
        let read_norm = bf16(&mut gpu, &wave(4, wide, 0.5));
        let ld = wide + 64;
        let f16_fused = gpu.zeros(&[rows * ld], DType::F16).expect("f16 fused");
        hyper_norm_gate(
            &mut gpu,
            &HyperNormGate {
                input: &input,
                norm_weight: &norm,
                gate_weight: &gate_weight,
                gates: &fused,
                rows,
                branches,
                hidden,
                state_bf16: false,
                read_f16: Some((&read_norm, &f16_fused, ld)),
            },
        )
        .expect("fused");
        let f16_ref = gpu.zeros(&[rows * ld], DType::F16).expect("f16 ref");
        let read_normalized = gpu
            .zeros(&[rows * wide], DType::F32)
            .expect("read normalized");
        hyper_norm_f16(
            &mut gpu,
            &HyperNorm {
                input: &input,
                norm_weight: &read_norm,
                normalized: &read_normalized,
                branches,
                hidden,
                state_bf16: false,
            },
            &f16_ref,
            ld,
            false,
        )
        .expect("norm f16");
        let (x, y) = (
            gpu.download_f16_bits(&f16_fused).expect("download"),
            gpu.download_f16_bits(&f16_ref).expect("download"),
        );
        assert!(y.iter().any(|v| *v != 0), "read F16 is all zero");
        assert_eq!(x, y, "fused read F16 differs");
        // The same streams as BF16 bits (the F16 prefill route) take the
        // vectorized normalization pass: same gates, same F16 output.
        let input_values = gpu.download_f32(&input).expect("download");
        let mut stream_bytes: Vec<u8> = input_values
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        stream_bytes.resize(rows * wide * 4, 0);
        let mut bf16_streams = gpu
            .upload_raw(&stream_bytes, &[stream_bytes.len()])
            .expect("bf16 streams");
        bf16_streams.dtype = DType::F32;
        bf16_streams.shape = vec![rows * wide];
        let gates_bf16 = gpu
            .zeros(&[rows * branches], DType::F32)
            .expect("gates bf16");
        let f16_bf16 = gpu.zeros(&[rows * ld], DType::F16).expect("f16 bf16");
        hyper_norm_gate(
            &mut gpu,
            &HyperNormGate {
                input: &bf16_streams,
                norm_weight: &norm,
                gate_weight: &gate_weight,
                gates: &gates_bf16,
                rows,
                branches,
                hidden,
                state_bf16: true,
                read_f16: Some((&read_norm, &f16_bf16, ld)),
            },
        )
        .expect("fused bf16");
        assert_eq!(
            gpu.download_f16_bits(&f16_bf16).expect("download"),
            y,
            "BF16-stream read F16 differs"
        );
        assert_eq!(
            gpu.download_f32(&gates_bf16)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            gpu.download_f32(&fused)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "BF16-stream gates differ"
        );
        for tensor in [bf16_streams, gates_bf16, f16_bf16] {
            gpu.free_tensor(tensor).expect("free");
        }
        let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
        hyper_norm(
            &mut gpu,
            &HyperNorm {
                input: &input,
                norm_weight: &norm,
                normalized: &normalized,
                branches,
                hidden,
                state_bf16: false,
            },
        )
        .expect("norm");
        let reference = gpu
            .zeros(&[rows * branches], DType::F32)
            .expect("reference");
        gpu.gemm_bf16_xf32_multirow(&gate_weight, &normalized, &reference, branches, wide, rows)
            .expect("gemm");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &fused), bits(&gpu, &reference));
        assert!(b.iter().any(|v| *v != 0), "gates are all zero");
        assert_eq!(a, b, "fused norm-gate differs");
        for tensor in [input, norm, gate_weight, fused, normalized, reference] {
            gpu.free_tensor(tensor).expect("free");
        }
        for tensor in [read_norm, f16_fused, f16_ref, read_normalized] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// HC streams stored as BF16 bits (the Qwen4 F16 prefill route) must give
    /// every stream reader exactly the F32 stream's result: the norm, the
    /// norm-gate and the write all round the stream to BF16 on load, and the
    /// write's BF16 output is its F32 output's value.
    #[test]
    fn hc_bf16_streams_match_f32_streams() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (rows, branches, hidden) = (21usize, 4usize, 2560usize);
        let wide = branches * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bf16 = |gpu: &mut Gpu, values: &[f32]| -> GpuTensor {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut tensor = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16");
            tensor.dtype = DType::BF16;
            tensor.shape = vec![values.len()];
            tensor
        };
        // Arbitrary F32 stream values (not BF16-exact), and their RNE BF16
        // bits in the first half of an F32-typed buffer.
        let state = wave(1, rows * wide, 3.0);
        let mut state_bytes: Vec<u8> = state
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        state_bytes.resize(rows * wide * 4, 0);
        let f32_state = gpu.upload_f32(&state, &[state.len()]).expect("state");
        let mut bf16_state = gpu
            .upload_raw(&state_bytes, &[state_bytes.len()])
            .expect("bf16 state");
        bf16_state.dtype = DType::F32;
        bf16_state.shape = vec![rows * wide];
        let norm = bf16(&mut gpu, &wave(2, wide, 0.5));
        let gate_weight = bf16(&mut gpu, &wave(3, branches * wide, 0.05));
        let mixed = gpu
            .upload_f32(&wave(4, rows * hidden, 2.0), &[rows * hidden])
            .expect("mixed");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let mut results = Vec::new();
        for (input, state_bf16) in [(&f32_state, false), (&bf16_state, true)] {
            let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
            hyper_norm(
                &mut gpu,
                &HyperNorm {
                    input,
                    norm_weight: &norm,
                    normalized: &normalized,
                    branches,
                    hidden,
                    state_bf16,
                },
            )
            .expect("norm");
            let gates = gpu.zeros(&[rows * branches], DType::F32).expect("gates");
            hyper_norm_gate(
                &mut gpu,
                &HyperNormGate {
                    input,
                    norm_weight: &norm,
                    gate_weight: &gate_weight,
                    gates: &gates,
                    rows,
                    branches,
                    hidden,
                    state_bf16,
                    read_f16: None,
                },
            )
            .expect("norm-gate");
            hyper_write(
                &mut gpu,
                &HyperWrite {
                    input,
                    normalized: &normalized,
                    mixed: &mixed,
                    gates: &gates,
                    output: input,
                    branches,
                    hidden,
                    state_bf16,
                },
            )
            .expect("write");
            let mut written = bits(&gpu, input);
            if state_bf16 {
                // Widen the BF16 halves to the F32 values they encode.
                written = written
                    .iter()
                    .flat_map(|w| [w << 16, w & 0xFFFF_0000])
                    .take(rows * wide)
                    .collect();
            }
            results.push((bits(&gpu, &normalized), bits(&gpu, &gates), written));
            gpu.free_tensor(normalized).expect("free");
            gpu.free_tensor(gates).expect("free");
        }
        assert!(results[0].1.iter().any(|v| *v != 0), "gates are all zero");
        assert_eq!(results[0].0, results[1].0, "normalized rows differ");
        assert_eq!(results[0].1, results[1].1, "gates differ");
        assert_eq!(results[0].2, results[1].2, "written streams differ");
        for tensor in [f32_state, bf16_state, norm, gate_weight, mixed] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The fused HC read tail (BF16 up projection + branch mix) must equal
    /// the multi-row BF16 GEMM followed by hyper_read_projected bit for bit.
    #[test]
    fn hyper_read_up_fused_is_bit_identical_to_gemm_then_read() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        let (rows, hidden, low_rank) = (131usize, 2560usize, 320usize);
        let wide = 4 * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bytes: Vec<u8> = wave(1, wide * low_rank, 0.2)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut up_weight = gpu.upload_raw(&bytes, &[bytes.len()]).expect("up weight");
        up_weight.dtype = DType::BF16;
        up_weight.shape = vec![wide * low_rank];
        // BF16 values, as hc_activation leaves them (the WMMA read needs it).
        let low_values: Vec<f32> = wave(2, rows * low_rank, 1.5)
            .iter()
            .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
            .collect();
        let low = gpu
            .upload_f32(&low_values, &[rows * low_rank])
            .expect("low");
        let normalized = gpu
            .upload_f32(&wave(3, rows * wide, 2.0), &[rows * wide])
            .expect("normalized");
        let fused = gpu.zeros(&[rows * hidden], DType::F32).expect("fused");
        hyper_read_up_fused(
            &mut gpu,
            &HyperReadUpFused {
                up_weight: &up_weight,
                low: &low,
                normalized: &normalized,
                mixed: &fused,
                rows,
                hidden,
                low_rank,
                normalized_bf16: false,
            },
        )
        .expect("fused");
        // The same rows as BF16 bits (RNE) in the first half of an F32
        // buffer, as hyper_norm_f16 stores them.
        let mut bf16_bytes: Vec<u8> = wave(3, rows * wide, 2.0)
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        bf16_bytes.resize(rows * wide * 4, 0);
        let mut normalized_bf16 = gpu
            .upload_raw(&bf16_bytes, &[bf16_bytes.len()])
            .expect("normalized bf16");
        normalized_bf16.dtype = DType::F32;
        normalized_bf16.shape = vec![rows * wide];
        let fused_bf16 = gpu.zeros(&[rows * hidden], DType::F32).expect("fused bf16");
        hyper_read_up_fused(
            &mut gpu,
            &HyperReadUpFused {
                up_weight: &up_weight,
                low: &low,
                normalized: &normalized_bf16,
                mixed: &fused_bf16,
                rows,
                hidden,
                low_rank,
                normalized_bf16: true,
            },
        )
        .expect("fused bf16");
        let up = gpu.zeros(&[rows * wide], DType::F32).expect("up");
        gpu.gemm_bf16_xf32_multirow(&up_weight, &low, &up, wide, low_rank, rows)
            .expect("gemm");
        let reference = gpu.zeros(&[rows * hidden], DType::F32).expect("reference");
        let norm_weight = gpu.zeros(&[wide], DType::BF16).expect("norm weight");
        hyper_read_projected(
            &mut gpu,
            &HyperReadProjected {
                input: &normalized,
                norm_weight: &norm_weight,
                up: &up,
                normalized: &normalized,
                mixed: &reference,
                branches: 4,
                hidden,
            },
        )
        .expect("read");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &fused), bits(&gpu, &reference));
        assert!(b.iter().any(|v| *v != 0), "mix is all zero");
        assert_eq!(a, b, "fused HC read differs");
        assert_eq!(
            bits(&gpu, &fused_bf16),
            b,
            "BF16-normalized HC read differs"
        );
        let wmma = gpu.zeros(&[rows * hidden], DType::F32).expect("wmma");
        if gpu.arch_caps.has_wmma_w32() || gpu.qwen4_f16_wmma_gfx1201() {
            // hyper_norm_f16's F16 copy: the BF16-rounded values, exact in F16
            // (all nonzero |v| here are normal F16s).
            let f16_bytes: Vec<u8> = wave(3, rows * wide, 2.0)
                .iter()
                .flat_map(|v| {
                    let u = v.to_bits();
                    let b = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000;
                    let sign = ((b >> 16) & 0x8000) as u16;
                    let bits = if b & 0x7FFF_FFFF == 0 {
                        sign
                    } else {
                        let exp = ((b >> 23) & 0xFF) as u16 + 15 - 127;
                        sign | (exp << 10) | ((b >> 13) & 0x3FF) as u16
                    };
                    bits.to_le_bytes()
                })
                .collect();
            let mut normalized_f16 = gpu
                .upload_raw(&f16_bytes, &[f16_bytes.len()])
                .expect("normalized f16");
            normalized_f16.dtype = DType::F16;
            normalized_f16.shape = vec![rows * wide];
            // `low` as packed BF16 bits, as hc_activation's bf16_out holds it.
            let low_bytes: Vec<u8> = low_values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut low_bf16 = gpu
                .upload_raw(&low_bytes, &[low_bytes.len()])
                .expect("low bf16");
            low_bf16.dtype = DType::BF16;
            low_bf16.shape = vec![rows * low_rank];
            hyper_read_up_wmma(
                &mut gpu,
                &HyperReadUpFused {
                    up_weight: &up_weight,
                    low: &low_bf16,
                    normalized: &normalized_f16,
                    mixed: &wmma,
                    rows,
                    hidden,
                    low_rank,
                    normalized_bf16: true,
                },
                wide,
            )
            .expect("wmma");
            // WMMA's F32 summation order may round a gate one BF16 step apart
            // from the chain's; anything more is a layout or indexing error.
            let reference = gpu.download_f32(&fused).expect("download");
            let got = gpu.download_f32(&wmma).expect("download");
            let far = reference
                .iter()
                .zip(&got)
                .filter(|(r, w)| (*r - *w).abs() > r.abs().max(w.abs()) / 64.0 + 1e-6)
                .count();
            let differ = reference.iter().zip(&got).filter(|(r, w)| r != w).count();
            assert_eq!(far, 0, "WMMA HC read beyond one BF16 step");
            assert!(
                differ * 100 < reference.len(),
                "WMMA HC read: {differ} differ"
            );
            // H3: the retiled operand-swapped entries are bytewise the
            // baseline entries' (also at the ragged row count 131).
            let tiled_out = gpu.zeros(&[rows * hidden], DType::F32).expect("tiled");
            hyper_read_up_wmma_tiled(
                &mut gpu,
                &HyperReadUpFused {
                    up_weight: &up_weight,
                    low: &low_bf16,
                    normalized: &normalized_f16,
                    mixed: &tiled_out,
                    rows,
                    hidden,
                    low_rank,
                    normalized_bf16: true,
                },
                wide,
                true,
            )
            .expect("tiled wmma");
            assert_eq!(
                bits(&gpu, &tiled_out),
                bits(&gpu, &wmma),
                "retiled HC read differs from the baseline entry"
            );
            gpu.free_tensor(tiled_out).expect("free");
            gpu.free_tensor(normalized_f16).expect("free");
            gpu.free_tensor(low_bf16).expect("free");
        }
        for tensor in [
            up_weight,
            low,
            normalized,
            normalized_bf16,
            fused_bf16,
            fused,
            wmma,
            up,
            reference,
            norm_weight,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The persistent row-batched GDN recurrence must equal the per-row step
    /// kernel bit for bit (outputs and final state), including across the
    /// kernel's 256-row prologue block boundary.
    #[test]
    fn gdn_persistent_batch_is_bit_identical_to_per_row_steps() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 300usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta: Vec<f32> = wave(3, rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let state0 = wave(4, value * dim, 0.2);
        let proj_gpu = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");

        let batched_state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
        let batched_out = gpu.zeros(&[rows * value], DType::F32).expect("output");
        gated_delta_step_batched(
            &mut gpu,
            &GatedDeltaStepBatched {
                projection: &proj_gpu,
                gate: &gate_gpu,
                beta: &beta_gpu,
                state: &batched_state,
                output: &batched_out,
                row_states: None,
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            },
        )
        .expect("batched GDN");

        let row_state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
        let row_out = gpu.zeros(&[rows * value], DType::F32).expect("output");
        for row in 0..rows {
            gated_delta_step(
                &mut gpu,
                &GatedDeltaStep {
                    q: &proj_gpu.sub_offset(row * qkv, qk),
                    k: &proj_gpu.sub_offset(row * qkv + qk, qk),
                    v: &proj_gpu.sub_offset(row * qkv + 2 * qk, value),
                    gate: &gate_gpu.sub_offset(row * value_heads, value_heads),
                    beta: &beta_gpu.sub_offset(row * value_heads, value_heads),
                    state: &row_state,
                    output: &row_out.sub_offset(row * value, value),
                    key_heads,
                    value_heads,
                    key_dim: dim,
                    value_dim: dim,
                    position: row,
                },
            )
            .expect("per-row GDN");
        }
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &batched_out), bits(&gpu, &row_out));
        assert!(a.iter().any(|v| *v != 0), "batched output is all zero");
        let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(differing, 0, "GDN outputs differ in {differing} cells");
        assert_eq!(
            bits(&gpu, &batched_state),
            bits(&gpu, &row_state),
            "GDN final state differs"
        );
        for tensor in [
            proj_gpu,
            gate_gpu,
            beta_gpu,
            batched_state,
            batched_out,
            row_state,
            row_out,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The chunked F16 WMMA GDN recurrence + gate (Qwen4 F16 prefill route)
    /// must track the exact persistent kernel followed by the gate kernel, in
    /// gated output and final state, across its 16-row chunks and a partial
    /// last chunk.
    #[test]
    fn gdn_chunk_gate_arm_matches_persistent_kernel_and_gate() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_wmma_w32() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx11 WMMA");
            return;
        }
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta: Vec<f32> = wave(3, rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let state0 = wave(4, value * dim, 0.2);
        let z = wave(5, rows * value, 2.0);
        let norm_bytes: Vec<u8> = wave(6, dim, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let proj_gpu = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let z_gpu = gpu.upload_f32(&z, &[z.len()]).expect("z");
        let mut norm_gpu = gpu
            .upload_raw(&norm_bytes, &[norm_bytes.len()])
            .expect("norm");
        norm_gpu.dtype = DType::BF16;
        norm_gpu.shape = vec![dim];
        // The chunked arm reads the convolution output as packed BF16: the
        // same values the exact kernel rounds the F32 projection to.
        let proj_bytes: Vec<u8> = projection
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        let mut proj_bf16 = gpu
            .upload_raw(&proj_bytes, &[proj_bytes.len()])
            .expect("projection bf16");
        proj_bf16.dtype = DType::BF16;
        proj_bf16.shape = vec![projection.len()];
        let run = |gpu: &mut Gpu, fused: bool| {
            let state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
            let recurrent = gpu.zeros(&[rows * value], DType::F32).expect("recurrent");
            let out = gpu.zeros(&[rows * value], DType::F32).expect("output");
            let step = GatedDeltaStepBatched {
                projection: if fused { &proj_bf16 } else { &proj_gpu },
                gate: &gate_gpu,
                beta: &beta_gpu,
                state: &state,
                output: &recurrent,
                row_states: None,
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            };
            let gated = GatedDeltaGateBatched {
                recurrent_output: &recurrent,
                z: &z_gpu,
                norm: &norm_gpu,
                output: &out,
                rows,
                value_heads,
                value_dim: dim,
            };
            if fused {
                assert!(
                    gated_delta_chunk_route(gpu, &step),
                    "chunked route did not apply"
                );
                gated_delta_step_gate_wmma(gpu, &step, &gated).expect("fused GDN");
            } else {
                gated_delta_step_batched(gpu, &step).expect("batched GDN");
                gated_delta_gate_batched(gpu, &gated).expect("gate");
            }
            let values = (
                gpu.download_f32(&out).expect("download"),
                gpu.download_f32(&state).expect("download"),
            );
            for tensor in [out, recurrent, state] {
                gpu.free_tensor(tensor).expect("free");
            }
            values
        };
        let (ref_out, ref_state) = run(&mut gpu, false);
        let (col_out, col_state) = run(&mut gpu, true);
        let rel = |a: &[f32], b: &[f32]| {
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                err += (*x as f64 - *y as f64).powi(2);
                norm += (*x as f64).powi(2);
            }
            assert!(norm > 0.0, "reference is all zero");
            (err / norm).sqrt()
        };
        // F16 operands flip an occasional BF16 rounding of the gated output
        // (~2e-3 relative) and leave the state ~3e-4 off; a wrong column, row,
        // head, chunk boundary or gate lands near 1.
        let (out_rel, state_rel) = (rel(&ref_out, &col_out), rel(&ref_state, &col_state));
        assert!(out_rel < 1e-2, "GDN chunk gate output rel L2 {out_rel:.3e}");
        assert!(state_rel < 2e-3, "GDN chunk state rel L2 {state_rel:.3e}");
        for tensor in [proj_gpu, proj_bf16, gate_gpu, beta_gpu, z_gpu, norm_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// Q8 GDN state (Qwen3.5's DeltaNet Q8 format): every route tracks its
    /// F32 twin (decode steps, the persistent batch, the chunked prefill),
    /// the verify capture writes exactly the plain batch's final state into
    /// its ring slot, and a rollback re-run of the kept rows writes exactly
    /// the state a plain batch over those rows does. A transposed row, a
    /// wrong scale or a slot offset lands near 1.
    #[test]
    fn gdn_q8_state_tracks_f32_on_every_route() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        // `rows` decode steps; the batch routes run every ragged row count
        // around the persistent kernel's 16-row unroll and 256-row blocks.
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 24usize);
        let batch_rows = [2usize, 15, 16, 17, 255, 256, 257];
        let max_rows = 257usize;
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let format = GdnStateFormat::Q8;
        let slot_bytes = format.state_units(value_heads, dim, dim);
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, max_rows * qkv, 1.5);
        let gate: Vec<f32> =
            wave(2, max_rows * value_heads, 0.5).iter().map(|g| g - 0.6).collect();
        let beta: Vec<f32> =
            wave(3, max_rows * value_heads, 0.45).iter().map(|b| b + 0.5).collect();
        let state0 = wave(4, value * dim, 0.2);
        let proj_gpu = gpu.upload_f32(&projection, &[projection.len()]).expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let slot0 = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        gdn_state_convert(&mut gpu, "gdn_state_f32_to_q8", state0_gpu.buf.as_ptr(), slot0.buf.as_ptr(), value_heads, Some(7))
            .expect("quantize");
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        let fresh_slot = |gpu: &mut Gpu| -> GpuTensor {
            let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
            gpu.copy_d2d(&slot0, &slot, slot_bytes).expect("copy slot");
            slot
        };
        let dequant = |gpu: &mut Gpu, slot: &GpuTensor| -> Vec<f32> {
            let f = gpu.zeros(&[value * dim], DType::F32).expect("f32");
            gdn_state_convert(gpu, "gdn_state_q8_to_f32", slot.buf.as_ptr(), f.buf.as_ptr(), value_heads, None)
                .expect("dequantize");
            let values = gpu.download_f32(&f).expect("download");
            gpu.free_tensor(f).expect("free");
            values
        };
        let rel = |a: &[f32], b: &[f32]| {
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                err += (*x as f64 - *y as f64).powi(2);
                norm += (*x as f64).powi(2);
            }
            assert!(norm > 0.0, "reference is all zero");
            (err / norm).sqrt()
        };
        let q0 = dequant(&mut gpu, &slot0);
        let round_trip = rel(&state0, &q0);
        assert!(round_trip < 1e-2, "Q8 state round trip rel L2 {round_trip:.3e}");

        // Decode: one step per row, F32 and Q8 states from the same start.
        let decode = |gpu: &mut Gpu, state: &GpuTensor| -> Vec<f32> {
            let out = gpu.zeros(&[rows * value], DType::F32).expect("output");
            for row in 0..rows {
                gated_delta_step(
                    gpu,
                    &GatedDeltaStep {
                        q: &proj_gpu.sub_offset(row * qkv, qk),
                        k: &proj_gpu.sub_offset(row * qkv + qk, qk),
                        v: &proj_gpu.sub_offset(row * qkv + 2 * qk, value),
                        gate: &gate_gpu.sub_offset(row * value_heads, value_heads),
                        beta: &beta_gpu.sub_offset(row * value_heads, value_heads),
                        state,
                        output: &out.sub_offset(row * value, value),
                        key_heads,
                        value_heads,
                        key_dim: dim,
                        value_dim: dim,
                        position: row,
                    },
                )
                .expect("GDN step");
            }
            let values = gpu.download_f32(&out).expect("download");
            gpu.free_tensor(out).expect("free");
            values
        };
        let f32_state = gpu.upload_f32(&q0, &[q0.len()]).expect("state");
        let f32_out = decode(&mut gpu, &f32_state);
        let f32_final = gpu.download_f32(&f32_state).expect("download");
        let q8_state = fresh_slot(&mut gpu);
        let q8_out = decode(&mut gpu, &q8_state);
        let q8_final = dequant(&mut gpu, &q8_state);
        let (out_rel, state_rel) = (rel(&f32_out, &q8_out), rel(&f32_final, &q8_final));
        eprintln!("Q8 decode: output rel {out_rel:.3e}, state rel {state_rel:.3e}");
        assert!(out_rel < 3e-2 && state_rel < 3e-2, "Q8 decode drifted: {out_rel:.3e} {state_rel:.3e}");

        // Persistent batch, plain and captured, and the F32 twin.
        let batch = |gpu: &mut Gpu, state: &GpuTensor, ring: Option<&GpuTensor>, n: usize| -> Vec<f32> {
            let out = gpu.zeros(&[n * value], DType::F32).expect("output");
            gated_delta_step_batched(
                gpu,
                &GatedDeltaStepBatched {
                    projection: &proj_gpu.sub_offset(0, n * qkv),
                    gate: &gate_gpu.sub_offset(0, n * value_heads),
                    beta: &beta_gpu.sub_offset(0, n * value_heads),
                    state,
                    output: &out,
                    row_states: ring,
                    rows: n,
                    qkv_width: qkv,
                    key_heads,
                    value_heads,
                    key_dim: dim,
                    value_dim: dim,
                    position: 0,
                },
            )
            .expect("batched GDN");
            let values = gpu.download_f32(&out).expect("download");
            gpu.free_tensor(out).expect("free");
            values
        };
        // The persistent route requantizes after every row with that row's
        // position, so it is the decode route bit for bit: an MTP verify of
        // these rows must produce decode's outputs and leave decode's state.
        let rows_state = fresh_slot(&mut gpu);
        let rows_out = batch(&mut gpu, &rows_state, None, rows);
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&rows_out), bits(&q8_out), "Q8 persistent outputs differ from decode");
        assert_eq!(
            bytes(&gpu, &rows_state),
            bytes(&gpu, &q8_state),
            "Q8 persistent state differs from decode"
        );
        gpu.free_tensor(rows_state).expect("free");
        // Digest of every Q8 state this test writes: equal across processes
        // (the multi-process stress compares it).
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        let mut fold = |data: &[u8]| {
            for byte in data {
                digest = (digest ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        fold(&bytes(&gpu, &q8_state));
        for n in batch_rows {
            let keep = (n / 2).max(1);
            let f32_batch_state = gpu.upload_f32(&q0, &[q0.len()]).expect("state");
            let f32_batch_out = batch(&mut gpu, &f32_batch_state, None, n);
            let f32_batch_final = gpu.download_f32(&f32_batch_state).expect("download");
            let q8_batch_state = fresh_slot(&mut gpu);
            let q8_batch_out = batch(&mut gpu, &q8_batch_state, None, n);
            let q8_batch_final = dequant(&mut gpu, &q8_batch_state);
            let (out_rel, state_rel) =
                (rel(&f32_batch_out, &q8_batch_out), rel(&f32_batch_final, &q8_batch_final));
            eprintln!("Q8 batch {n} rows: output rel {out_rel:.3e}, state rel {state_rel:.3e}");
            assert!(
                out_rel < 1e-2 && state_rel < 1e-2,
                "Q8 batch {n} rows drifted: {out_rel:.3e} {state_rel:.3e}"
            );
            fold(&bytes(&gpu, &q8_batch_state));

            let captured_state = fresh_slot(&mut gpu);
            let ring = gpu.zeros(&[2 * n * slot_bytes], DType::Raw).expect("ring");
            let captured_out = batch(&mut gpu, &captured_state, Some(&ring), n);
            assert_eq!(captured_out, q8_batch_out, "{n} rows: captured verify output differs");
            assert_eq!(
                bytes(&gpu, &captured_state),
                bytes(&gpu, &slot0),
                "{n} rows: capture wrote the live state"
            );
            assert_eq!(
                bytes(&gpu, &ring.sub_offset((n - 1) * slot_bytes, slot_bytes)),
                bytes(&gpu, &q8_batch_state),
                "{n} rows: captured slot is not the plain batch's final state"
            );

            // Rollback: `keep` rows re-run from ring slot 0 into slot `n`. Its
            // input is the rows' projection, then gate, then beta.
            let recurrence: Vec<f32> = projection[..n * qkv]
                .iter()
                .chain(&gate[..n * value_heads])
                .chain(&beta[..n * value_heads])
                .copied()
                .collect();
            let recurrence_gpu =
                gpu.upload_f32(&recurrence, &[recurrence.len()]).expect("recurrence");
            gpu.copy_d2d(&slot0, &ring, slot_bytes).expect("seed ring");
            let pointers: Vec<u8> = [recurrence_gpu.buf.as_ptr(), ring.buf.as_ptr()]
                .iter()
                .flat_map(|p| (*p as u64).to_ne_bytes())
                .collect();
            let table = gpu.upload_raw(&pointers, &[pointers.len()]).expect("table");
            let discard = gpu.zeros(&[n * value], DType::F32).expect("discard");
            gated_delta_rollback_layers(
                &mut gpu,
                &GatedDeltaRollbackLayers {
                    table: &table,
                    discard: &discard,
                    format,
                    layers: 1,
                    rows: n,
                    keep,
                    from: 0,
                    to: n,
                    qkv_width: qkv,
                    key_heads,
                    value_heads,
                    position: keep - 1,
                },
            )
            .expect("rollback");
            let kept_state = fresh_slot(&mut gpu);
            batch(&mut gpu, &kept_state, None, keep);
            assert_eq!(
                bytes(&gpu, &ring.sub_offset((n + keep - 1) * slot_bytes, slot_bytes)),
                bytes(&gpu, &kept_state),
                "{n} rows: rollback slot is not the kept rows' state"
            );
            fold(&bytes(&gpu, &kept_state));
            for tensor in [
                f32_batch_state, q8_batch_state, captured_state, ring, recurrence_gpu, table,
                discard, kept_state,
            ] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        eprintln!("GDN_Q8_DIGEST {digest:016x}");
        for tensor in [proj_gpu, gate_gpu, beta_gpu, state0_gpu, slot0, f32_state, q8_state] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The chunked F16 WMMA prefill route on a Q8 state (dequantize, chunk,
    /// requantize) tracks the same route on the F32 state.
    #[test]
    fn gdn_q8_state_chunk_route_tracks_f32() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_wmma_w32() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx11 WMMA");
            return;
        }
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, rows * value_heads, 0.5).iter().map(|g| g - 0.6).collect();
        let beta: Vec<f32> = wave(3, rows * value_heads, 0.45).iter().map(|b| b + 0.5).collect();
        let state0 = wave(4, value * dim, 0.2);
        let z = wave(5, rows * value, 2.0);
        let norm_bytes: Vec<u8> = wave(6, dim, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let proj_bytes: Vec<u8> = projection
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        let mut proj_bf16 = gpu.upload_raw(&proj_bytes, &[proj_bytes.len()]).expect("projection");
        proj_bf16.dtype = DType::BF16;
        proj_bf16.shape = vec![projection.len()];
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let z_gpu = gpu.upload_f32(&z, &[z.len()]).expect("z");
        let mut norm_gpu = gpu.upload_raw(&norm_bytes, &[norm_bytes.len()]).expect("norm");
        norm_gpu.dtype = DType::BF16;
        norm_gpu.shape = vec![dim];
        let slot_bytes = GdnStateFormat::Q8.state_units(value_heads, dim, dim);
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let seed_slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        gdn_state_convert(&mut gpu, "gdn_state_f32_to_q8", state0_gpu.buf.as_ptr(), seed_slot.buf.as_ptr(), value_heads, Some(3))
            .expect("quantize");
        let run = |gpu: &mut Gpu, state: &GpuTensor, n: usize| -> Vec<f32> {
            let recurrent = gpu.zeros(&[n * value], DType::F32).expect("recurrent");
            let out = gpu.zeros(&[n * value], DType::F32).expect("output");
            let step = GatedDeltaStepBatched {
                projection: &proj_bf16.sub_offset(0, n * qkv),
                gate: &gate_gpu.sub_offset(0, n * value_heads),
                beta: &beta_gpu.sub_offset(0, n * value_heads),
                state,
                output: &recurrent,
                row_states: None,
                rows: n,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            };
            let gated = GatedDeltaGateBatched {
                recurrent_output: &recurrent,
                z: &z_gpu.sub_offset(0, n * value),
                norm: &norm_gpu,
                output: &out,
                rows: n,
                value_heads,
                value_dim: dim,
            };
            gated_delta_step_gate_wmma(gpu, &step, &gated).expect("chunked GDN");
            let values = gpu.download_f32(&out).expect("download");
            gpu.free_tensor(recurrent).expect("free");
            gpu.free_tensor(out).expect("free");
            values
        };
        let rel = |a: &[f32], b: &[f32]| {
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                err += (*x as f64 - *y as f64).powi(2);
                norm += (*x as f64).powi(2);
            }
            assert!(norm > 0.0, "reference is all zero");
            (err / norm).sqrt()
        };
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        // The route's minimum and the 16-row chunk boundary +-1.
        for n in [512usize, 513, 527, 528, 529, 530] {
            let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
            gpu.copy_d2d(&seed_slot, &slot, slot_bytes).expect("copy slot");
            let f32_state = gpu.zeros(&[value * dim], DType::F32).expect("state");
            gdn_state_convert(&mut gpu, "gdn_state_q8_to_f32", slot.buf.as_ptr(), f32_state.buf.as_ptr(), value_heads, None)
                .expect("dequantize");
            let f32_out = run(&mut gpu, &f32_state, n);
            let q8_out = run(&mut gpu, &slot, n);
            let f32_final = gpu.download_f32(&f32_state).expect("download");
            let q8_final = gpu.zeros(&[value * dim], DType::F32).expect("state");
            gdn_state_convert(&mut gpu, "gdn_state_q8_to_f32", slot.buf.as_ptr(), q8_final.buf.as_ptr(), value_heads, None)
                .expect("dequantize");
            let q8_final_values = gpu.download_f32(&q8_final).expect("download");
            // The same inputs and dequantized start: the chunk output is
            // equal; the final state differs by one requantization.
            assert_eq!(f32_out, q8_out, "{n} rows: Q8 chunk output differs from its F32 twin");
            let state_rel = rel(&f32_final, &q8_final_values);
            eprintln!("Q8 chunk {n} rows: state rel {state_rel:.3e}");
            assert!(state_rel < 1e-2, "{n} rows: Q8 chunk state rel L2 {state_rel:.3e}");
            let mut slot_bytes_host = vec![0u8; slot_bytes];
            gpu.hip.memcpy_dtoh(&mut slot_bytes_host, &slot.buf).expect("download");
            for byte in slot_bytes_host {
                digest = (digest ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
            for tensor in [slot, f32_state, q8_final] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        eprintln!("GDN_Q8_CHUNK_DIGEST {digest:016x}");
        for tensor in [proj_bf16, gate_gpu, beta_gpu, z_gpu, norm_gpu, state0_gpu, seed_slot] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// Round-to-nearest-even BF16 bits of `values`, little-endian.
    fn bf16_le_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()
    }

    fn test_wave(seed: usize, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                (h as f32 - 5003.0) / 5003.0 * scale
            })
            .collect()
    }

    /// H5 hookup: the opt-in convolution that also stores the normalized Q/K
    /// plus the Q/K-norm-skipping chunk launch is bytewise the plain
    /// convolution + `gated_delta_qk_norm_bf16_batched` + chunk launch pair:
    /// gated output, final state, convolution ring, gate/beta and the V
    /// columns of the convolution output (all columns with `keep_qk_conv`).
    /// Ragged last row tile, a nonzero ring cursor and the in-place ring the
    /// pipeline uses.  The flag is off by default, and the geometry guards
    /// refuse every other head layout.
    #[test]
    fn gdn_conv_qknorm_hookup_is_bytewise_the_two_launch_pair() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx1151 and the F16 WMMA route");
            return;
        }
        let (key_heads, value_heads, dim, rows) = (16usize, 16usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let projection = gpu
            .upload_f32(&test_wave(1, rows * qkv, 1.5), &[rows * qkv])
            .expect("projection");
        let mut conv_kernel = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(7, qkv * 4, 0.6)), &[qkv * 4 * 2])
            .expect("conv kernel");
        conv_kernel.dtype = DType::BF16;
        conv_kernel.shape = vec![qkv * 4];
        let history0 = test_wave(8, 3 * qkv, 1.0);
        let a = gpu
            .upload_f32(&test_wave(9, rows * value_heads, 2.0), &[rows * value_heads])
            .expect("a");
        let b = gpu
            .upload_f32(&test_wave(10, rows * value_heads, 2.0), &[rows * value_heads])
            .expect("b");
        let mut a_log = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(11, value_heads, 1.0)), &[value_heads * 2])
            .expect("a_log");
        a_log.dtype = DType::BF16;
        a_log.shape = vec![value_heads];
        let mut dt_bias = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(12, value_heads, 1.0)), &[value_heads * 2])
            .expect("dt_bias");
        dt_bias.dtype = DType::BF16;
        dt_bias.shape = vec![value_heads];
        let state0 = test_wave(4, value_heads * dim * dim, 0.2);
        let z = gpu
            .upload_f32(&test_wave(5, rows * value, 2.0), &[rows * value])
            .expect("z");
        let mut norm = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(6, dim, 1.0)), &[dim * 2])
            .expect("norm");
        norm.dtype = DType::BF16;
        norm.shape = vec![dim];
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        // (gated output, state, ring, gate, beta, convolution output)
        let run = |gpu: &mut Gpu, fused: bool, keep_qk_conv: bool| -> [Vec<u8>; 6] {
            let history = gpu.upload_f32(&history0, &[history0.len()]).expect("history");
            let conv_out = gpu.zeros(&[rows * qkv], DType::BF16).expect("conv out");
            let gate = gpu.zeros(&[rows * value_heads], DType::F32).expect("gate");
            let beta = gpu.zeros(&[rows * value_heads], DType::F32).expect("beta");
            let state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
            let recurrent = gpu.zeros(&[rows * value], DType::F32).expect("recurrent");
            let out = gpu.zeros(&[rows * value], DType::F32).expect("output");
            let conv = GatedDeltaConvBatched {
                input: &projection,
                kernel: &conv_kernel,
                history: &history,
                output: &conv_out,
                next_history: &history,
                rows,
                channels: qkv,
                history_rows: 3,
                kernel_size: 4,
                start_cursor: 1,
            };
            let params = GatedDeltaParamsBatched {
                a: &a,
                b: &b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                gate: &gate,
                beta: &beta,
                rows,
                heads: value_heads,
            };
            let step = GatedDeltaStepBatched {
                projection: &conv_out,
                gate: &gate,
                beta: &beta,
                state: &state,
                output: &recurrent,
                row_states: None,
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            };
            let gated = GatedDeltaGateBatched {
                recurrent_output: &recurrent,
                z: &z,
                norm: &norm,
                output: &out,
                rows,
                value_heads,
                value_dim: dim,
            };
            assert!(gated_delta_chunk_route(gpu, &step), "chunked route did not apply");
            assert!(gated_delta_conv_qknorm_geometry(gpu, &step, &conv));
            // Default off: the flag alone selects the new kernels.
            assert_eq!(
                gated_delta_conv_qknorm_route(gpu, &step, &conv),
                gpu.flags.qwen4_gdn_conv_qknorm_enabled()
            );
            if fused {
                gated_delta_conv_params_qknorm_batched(gpu, &conv, &params, &step, keep_qk_conv)
                    .expect("conv + qk norm");
                gated_delta_step_gate_wmma_qknormed(gpu, &step, &gated).expect("chunk");
            } else {
                gated_delta_conv_params_batched(gpu, &conv, &params).expect("conv");
                gated_delta_step_gate_wmma(gpu, &step, &gated).expect("chunk");
            }
            let values = [
                bytes(gpu, &out),
                bytes(gpu, &state),
                bytes(gpu, &history),
                bytes(gpu, &gate),
                bytes(gpu, &beta),
                bytes(gpu, &conv_out),
            ];
            for tensor in [history, conv_out, gate, beta, state, recurrent, out] {
                gpu.free_tensor(tensor).expect("free");
            }
            values
        };
        let plain = run(&mut gpu, false, false);
        let fused = run(&mut gpu, true, false);
        let kept = run(&mut gpu, true, true);
        for (i, label) in ["gated output", "state", "conv ring", "gate", "beta"].iter().enumerate() {
            assert!(plain[i] == fused[i], "{label} differs (qk columns dropped)");
            assert!(plain[i] == kept[i], "{label} differs (qk columns kept)");
        }
        assert!(plain[5] == kept[5], "convolution output differs with keep_qk_conv");
        let (row_bytes, v_from) = (qkv * 2, 2 * qk * 2);
        for r in 0..rows {
            assert!(
                plain[5][r * row_bytes + v_from..(r + 1) * row_bytes]
                    == fused[5][r * row_bytes + v_from..(r + 1) * row_bytes],
                "row {r}: V convolution columns differ"
            );
        }
        // The geometry guards: only 16 key heads of 128 in q|k|v order.
        let conv_out = gpu.zeros(&[rows * qkv], DType::BF16).expect("conv out");
        let history = gpu.upload_f32(&history0, &[history0.len()]).expect("history");
        let state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
        let gate = gpu.zeros(&[rows * value_heads], DType::F32).expect("gate");
        let recurrent = gpu.zeros(&[rows * value], DType::F32).expect("recurrent");
        let step = GatedDeltaStepBatched {
            projection: &conv_out,
            gate: &gate,
            beta: &gate,
            state: &state,
            output: &recurrent,
            row_states: None,
            rows,
            qkv_width: qkv,
            key_heads,
            value_heads,
            key_dim: dim,
            value_dim: dim,
            position: 0,
        };
        let conv = GatedDeltaConvBatched {
            input: &projection,
            kernel: &conv_kernel,
            history: &history,
            output: &conv_out,
            next_history: &history,
            rows,
            channels: qkv,
            history_rows: 3,
            kernel_size: 4,
            start_cursor: 0,
        };
        assert!(gated_delta_conv_qknorm_geometry(&gpu, &step, &conv));
        let eight = GatedDeltaStepBatched { key_heads: 8, ..step };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &eight, &conv));
        let short = GatedDeltaStepBatched { rows: 511, ..eight };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &short, &conv));
        let f32_out = gpu.zeros(&[rows * qkv], DType::F32).expect("f32 conv out");
        let f32_conv = GatedDeltaConvBatched { output: &f32_out, ..conv };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &step, &f32_conv));
        let ring4 = GatedDeltaConvBatched { history_rows: 4, ..f32_conv };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &step, &ring4));
        for tensor in [
            projection, conv_kernel, a, b, a_log, dt_bias, z, norm, conv_out, history, state, gate,
            recurrent, f32_out,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// H6 hookup: the inline-Q8 chunk module is bytewise the
    /// dequantize + chunk + requantize arm it replaces (gated output and the
    /// whole Q8 slot), over the route's minimum rows and the 16-row chunk
    /// boundary +-1, at a nonzero start position (the requantization seed).
    #[test]
    fn gdn_q8_inline_hookup_is_bytewise_the_conversion_arm() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx1151 and the F16 WMMA route");
            return;
        }
        let (key_heads, value_heads, dim, max_rows) = (2usize, 6usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let mut proj_bf16 = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(1, max_rows * qkv, 1.5)), &[max_rows * qkv * 2])
            .expect("projection");
        proj_bf16.dtype = DType::BF16;
        proj_bf16.shape = vec![max_rows * qkv];
        let gate_values: Vec<f32> =
            test_wave(2, max_rows * value_heads, 0.5).iter().map(|g| g - 0.6).collect();
        let beta_values: Vec<f32> =
            test_wave(3, max_rows * value_heads, 0.45).iter().map(|b| b + 0.5).collect();
        let gate_gpu = gpu.upload_f32(&gate_values, &[gate_values.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta_values, &[beta_values.len()]).expect("beta");
        let z_gpu = gpu
            .upload_f32(&test_wave(5, max_rows * value, 2.0), &[max_rows * value])
            .expect("z");
        let mut norm_gpu = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(6, dim, 1.0)), &[dim * 2])
            .expect("norm");
        norm_gpu.dtype = DType::BF16;
        norm_gpu.shape = vec![dim];
        let state0 = test_wave(4, value_heads * dim * dim, 0.2);
        let slot_bytes = GdnStateFormat::Q8.state_units(value_heads, dim, dim);
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let seed_slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        gdn_state_convert(&mut gpu, "gdn_state_f32_to_q8", state0_gpu.buf.as_ptr(), seed_slot.buf.as_ptr(), value_heads, Some(3))
            .expect("quantize");
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        for (n, position) in [(512usize, 0usize), (513, 7), (527, 4096), (528, 1), (529, 99), (530, 262_143)] {
            let mut results = Vec::new();
            for inline in [false, true] {
                let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
                gpu.copy_d2d(&seed_slot, &slot, slot_bytes).expect("copy slot");
                let recurrent = gpu.zeros(&[n * value], DType::F32).expect("recurrent");
                let out = gpu.zeros(&[n * value], DType::F32).expect("output");
                let step = GatedDeltaStepBatched {
                    projection: &proj_bf16.sub_offset(0, n * qkv),
                    gate: &gate_gpu.sub_offset(0, n * value_heads),
                    beta: &beta_gpu.sub_offset(0, n * value_heads),
                    state: &slot,
                    output: &recurrent,
                    row_states: None,
                    rows: n,
                    qkv_width: qkv,
                    key_heads,
                    value_heads,
                    key_dim: dim,
                    value_dim: dim,
                    position,
                };
                let gated = GatedDeltaGateBatched {
                    recurrent_output: &recurrent,
                    z: &z_gpu.sub_offset(0, n * value),
                    norm: &norm_gpu,
                    output: &out,
                    rows: n,
                    value_heads,
                    value_dim: dim,
                };
                assert!(gated_delta_q8_inline_geometry(&gpu, &step));
                // Default off: the flag alone selects the new module.
                assert_eq!(
                    gated_delta_q8_inline_route(&gpu, &step),
                    gpu.flags.qwen4_gdn_q8_inline_enabled()
                );
                let arms = GatedDeltaChunkArms { qk_prenormed: false, q8_inline: inline };
                gated_delta_step_gate_wmma_arms(&mut gpu, &step, &gated, arms).expect("chunked GDN");
                results.push((bytes(&gpu, &out), bytes(&gpu, &slot)));
                for tensor in [slot, recurrent, out] {
                    gpu.free_tensor(tensor).expect("free");
                }
            }
            assert!(results[0].0 == results[1].0, "{n} rows @ {position}: gated output differs");
            assert!(results[0].1 == results[1].1, "{n} rows @ {position}: Q8 slot differs");
        }
        // An F32 state or a misaligned slot never takes the inline arm.
        let f32_state = gpu.zeros(&[value_heads * dim * dim], DType::F32).expect("f32 state");
        let step = GatedDeltaStepBatched {
            projection: &proj_bf16,
            gate: &gate_gpu,
            beta: &beta_gpu,
            state: &f32_state,
            output: &f32_state,
            row_states: None,
            rows: 512,
            qkv_width: qkv,
            key_heads,
            value_heads,
            key_dim: dim,
            value_dim: dim,
            position: 0,
        };
        assert!(!gated_delta_q8_inline_geometry(&gpu, &step));
        for tensor in [proj_bf16, gate_gpu, beta_gpu, z_gpu, norm_gpu, state0_gpu, seed_slot, f32_state] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    fn null_tensor(shape: &[usize], dtype: DType) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.shape = shape.to_vec();
        tensor.dtype = dtype;
        tensor
    }

    #[test]
    fn recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let input = [1.0f32, -2.0, 3.5, 0.25];
        let values = gpu
            .upload_f32(&input, &[input.len()])
            .expect("scale upload");

        // Ordinary path: no recorder armed, so the same launch leaves no tape.
        scale_f32(
            &mut gpu,
            &ScaleF32 {
                values: &values,
                scale: 2.0,
            },
        )
        .expect("scale");
        assert_eq!(gpu.replay.recorded_launches().len(), 0);

        // Open a recording window: the launch must now enter the tape with the
        // exact bytes it launched, resolve its owning artifact, AND still run.
        gpu.replay =
            crate::replay::ReplayController::new_armed(crate::replay::ReplayBackendRequest::Auto);
        gpu.replay.begin_capture().expect("open recording window");
        scale_f32(
            &mut gpu,
            &ScaleF32 {
                values: &values,
                scale: 3.0,
            },
        )
        .expect("scale");
        let recorded = gpu.replay.recorded_launches();
        assert_eq!(recorded.len(), 1, "one launch, one tape entry");
        assert_eq!(recorded[0].kernel, "scale_f32");
        assert!(
            recorded[0].artifact.is_some(),
            "the owning artifact must resolve, or preparation cannot lower the tape"
        );
        assert_eq!(recorded[0].grid, [1, 1, 1]);
        let kernarg = &recorded[0].kernarg;
        assert_eq!(kernarg.len() % 16, 0, "blob is tail-padded");
        assert_eq!(
            u64::from_ne_bytes(kernarg[0..8].try_into().unwrap()),
            values.buf.as_ptr() as u64
        );
        assert_eq!(
            i32::from_ne_bytes(kernarg[8..12].try_into().unwrap()),
            input.len() as i32
        );
        assert_eq!(f32::from_ne_bytes(kernarg[12..16].try_into().unwrap()), 3.0);

        // The recorded launch executed: (input * 2.0) * 3.0, all exact in f32.
        let actual = gpu.download_f32(&values).expect("scale download");
        assert_eq!(actual, vec![6.0, -12.0, 21.0, 1.5]);
    }

    #[test]
    fn checked_extents_reject_signed_flattening_overflow() {
        let max = i32::MAX as usize;
        assert_eq!(checked_product(max, 1, "boundary").unwrap(), max);
        assert!(checked_product(max, 2, "boundary").is_err());
        assert!(checked_add(max, 1, "boundary").is_err());
        assert!(blocks(max).is_ok());
        assert!(blocks(max + 1).is_err());
    }

    #[test]
    fn host_argmax_matches_sampler_semantics() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut long = vec![0.25f32; 248_320];
        long[200_001] = 7.0;
        long[5] = 7.0;
        long[9] = f32::INFINITY;
        let cases: [(Vec<f32>, u32); 4] = [
            // Ties resolve to the first index.
            (vec![1.0, 5.0, 3.0, 5.0], 1),
            // Non-finite values never win.
            (
                vec![f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY, 2.0],
                2,
            ),
            // No finite value: index 0.
            (vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 0),
            (long, 5),
        ];
        for (values, expected) in cases {
            let logits = gpu.upload_f32(&values, &[values.len()]).unwrap();
            assert_eq!(argmax_f32_host(&mut gpu, &logits).unwrap(), expected);
            gpu.free_tensor(logits).unwrap();
        }
    }

    #[test]
    fn raw_index_outputs_require_i32_byte_capacity() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };

        let logits = null_tensor(&[1], DType::F32);
        let indices = null_tensor(&[1], DType::Raw);
        let error = argmax_f32(
            &mut gpu,
            &ArgmaxF32 {
                logits: &logits,
                indices: &indices,
                rows: 1,
                vocab: 1,
            },
        )
        .expect_err("argmax must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let query = null_tensor(&[1], DType::F32);
        let pooled = null_tensor(&[1], DType::F32);
        let selected = null_tensor(&[1], DType::Raw);
        let error = indexed_attention_select(
            &mut gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: &pooled,
                selected: &selected,
                block_count: 1,
                index_heads: 1,
                index_dim: 1,
                budget_blocks: 1,
                compress: 1,
                visible: 1,
                capacity: 1,
            },
        )
        .expect_err("QSA selection must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let q_with_gate = null_tensor(&[2], DType::F32);
        let full_keys = null_tensor(&[1], DType::F32);
        let full_values = null_tensor(&[1], DType::F32);
        let output = null_tensor(&[1], DType::F32);
        let error = indexed_attention_attention(
            &mut gpu,
            &IndexedAttentionAttention {
                q_with_gate: &q_with_gate,
                full_keys: &full_keys,
                full_values: &full_values,
                selected: &selected,
                output: &output,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 1,
                selected_len: 1,
                full_capacity: 1,
                format: QsaKvFormat::F32,
            },
        )
        .expect_err("QSA attention must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn gdn_rejects_shape_before_kernel_launch() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let q = null_tensor(&[1], DType::F32);
        let k = null_tensor(&[2], DType::F32);
        let v = null_tensor(&[12], DType::F32);
        let gate = null_tensor(&[3], DType::F32);
        let beta = null_tensor(&[3], DType::F32);
        let state = null_tensor(&[24], DType::F32);
        let output = null_tensor(&[12], DType::F32);
        let params = GatedDeltaStep {
            q: &q,
            k: &k,
            v: &v,
            gate: &gate,
            beta: &beta,
            state: &state,
            output: &output,
            key_heads: 1,
            value_heads: 3,
            key_dim: 2,
            value_dim: 4,
            position: 0,
        };

        let error = gated_delta_step(&mut gpu, &params).expect_err("invalid shape must fail");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn qsa_reuse_selection_rejects_byte_capacity_mismatch() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let selected = null_tensor(&[3], DType::Raw);
        let selected_len_out = null_tensor(&[4], DType::Raw);
        let params = IndexedAttentionReuseSelection {
            selected: &selected,
            selected_len: 1,
            position: 0,
            capacity: 1,
            selected_len_out: &selected_len_out,
        };

        let error = indexed_attention_reuse_selection(&mut gpu, &params)
            .expect_err("byte capacity must be checked");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    /// Every position-derived QSA field must reach the recorder as a declaration
    /// pointing at the right kernarg slot: nothing else re-derives these values
    /// for a replay position. The assertion decodes the recorded bytes at each
    /// declared offset, so a wrong offset fails here rather than at replay.
    #[test]
    fn qsa_position_fields_are_declared_to_the_recorder() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        gpu.replay =
            crate::replay::ReplayController::new_armed(crate::replay::ReplayBackendRequest::Auto);
        gpu.replay.begin_capture().expect("open recording window");

        let compress = 4usize;
        let blocks = 2usize;
        let index_dim = 8usize;
        let index_heads = 2usize;
        let rows = 1usize;
        let position_start = blocks * compress - rows;
        let capacity = 8usize;
        let budget_blocks = 2usize;
        let cells = blocks * compress * index_dim;
        let raw: Vec<f32> = (0..cells).map(|i| (i % 17) as f32 * 0.5).collect();
        let raw_gpu = gpu.upload_f32(&raw, &[cells]).expect("raw upload");
        let pooled_gpu = gpu
            .zeros(&[blocks * index_dim], DType::F32)
            .expect("pooled allocation");
        indexed_attention_pool_rope(
            &mut gpu,
            &IndexedAttentionPoolRope {
                raw_keys: &raw_gpu,
                pooled: &pooled_gpu,
                norm: None,
                block_count: blocks,
                compress,
                index_dim,
                position: Some(QsaPositionBinding {
                    position_start,
                    rows,
                }),
                grid_bound: blocks,
            },
        )
        .expect("QSA pool/RoPE");

        let query_elements = index_heads * index_dim;
        let query: Vec<f32> = (0..query_elements)
            .map(|i| (i % 13) as f32 * 0.25)
            .collect();
        let query_gpu = gpu
            .upload_f32(&query, &[query_elements])
            .expect("query upload");
        let selected_gpu = gpu
            .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        indexed_attention_select_batch(
            &mut gpu,
            &IndexedAttentionSelectBatch {
                query: &query_gpu,
                pooled: &pooled_gpu,
                selected: &selected_gpu,
                rows,
                query_row_stride: query_elements,
                block_count: blocks,
                index_heads,
                index_dim,
                budget_blocks,
                compress,
                position_start,
                capacity,
                shape_blocks: blocks,
            },
        )
        .expect("QSA select");

        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let full_capacity = 8usize;
        let q_with_gate_gpu = gpu
            .zeros(&[rows * n_heads * 2 * head_dim], DType::F32)
            .expect("q allocation");
        let full_keys_gpu = gpu
            .zeros(&[full_capacity * n_kv_heads * head_dim], DType::F32)
            .expect("keys allocation");
        let full_values_gpu = gpu
            .zeros(&[full_capacity * n_kv_heads * head_dim], DType::F32)
            .expect("values allocation");
        let output_gpu = gpu
            .zeros(&[rows * n_heads * head_dim], DType::F32)
            .expect("attention output allocation");
        indexed_attention_attention_batch(
            &mut gpu,
            &IndexedAttentionAttentionBatch {
                q_with_gate: &q_with_gate_gpu,
                full_keys: &full_keys_gpu,
                full_values: &full_values_gpu,
                selected: &selected_gpu,
                output: &output_gpu,
                rows,
                position_start,
                n_heads,
                n_kv_heads,
                head_dim,
                budget_blocks,
                compress,
                capacity,
                full_capacity,
                format: QsaKvFormat::F32,
                shape_selected: capacity,
            },
        )
        .expect("QSA attention");

        // The index-key write: a row offset that moves with the position.
        let index_kv_width = 128usize;
        let dco = position_start * index_kv_width;
        let copy_src = gpu
            .zeros(&[index_kv_width], DType::F32)
            .expect("copy src allocation");
        let copy_dst = gpu
            .zeros(&[dco + index_kv_width], DType::F32)
            .expect("copy dst allocation");
        gpu.copy_rows_strided_f32(
            &copy_src,
            &copy_dst,
            rows,
            index_kv_width,
            index_kv_width,
            index_kv_width,
            dco,
            Some(index_kv_width),
        )
        .expect("index-key write");

        let launches = gpu.replay.recorded_launches();
        assert_eq!(launches.len(), 4, "the recorded QSA sequence changed");

        // The declared slot must hold the value this very launch used.
        let check = |launch: &crate::replay::RecordedHipLaunch,
                     binding: crate::replay::ReplayKernargBinding,
                     expected: u32,
                     label: &str| {
            assert!(
                launch.declared_kernarg_bindings().contains(&binding),
                "{label}: missing declaration {binding:?} in {:?}",
                launch.declared_kernarg_bindings()
            );
            let offset = binding.offset();
            let recorded = u32::from_ne_bytes(
                launch.kernarg[offset..offset + 4]
                    .try_into()
                    .expect("binding offset"),
            );
            assert_eq!(
                recorded, expected,
                "{label}: declaration points at the wrong slot"
            );
        };

        let pool_blocks = crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: 24,
            addend: rows as u32,
            divisor: compress as u32,
        };
        let pool_binding = launches[0]
            .declared_kernarg_bindings()
            .iter()
            .find(|binding| {
                matches!(
                    binding,
                    crate::replay::ReplayKernargBinding::PositionDivU32 { .. }
                )
            })
            .copied()
            .expect("pool declares its block count");
        assert_eq!(pool_binding, pool_blocks, "pool block-count offset drifted");
        check(
            &launches[0],
            pool_binding,
            blocks as u32,
            "pool block count",
        );

        let select_div = launches[1]
            .declared_kernarg_bindings()
            .iter()
            .find(|binding| {
                matches!(
                    binding,
                    crate::replay::ReplayKernargBinding::PositionDivU32 { .. }
                )
            })
            .copied()
            .expect("select declares its block count");
        let select_pos = launches[1]
            .declared_kernarg_bindings()
            .iter()
            .find(|binding| {
                matches!(
                    binding,
                    crate::replay::ReplayKernargBinding::PositionPlusU32 { .. }
                )
            })
            .copied()
            .expect("select declares its start position");
        check(
            &launches[1],
            select_div,
            blocks as u32,
            "select block count",
        );
        check(
            &launches[1],
            select_pos,
            position_start as u32,
            "select position",
        );

        let attention_pos = launches[2]
            .declared_kernarg_bindings()
            .first()
            .copied()
            .expect("attention declares its start position");
        check(
            &launches[2],
            attention_pos,
            position_start as u32,
            "attention position",
        );

        let copy_mul = launches[3]
            .declared_kernarg_bindings()
            .first()
            .copied()
            .expect("index-key write declares its row offset");
        assert_eq!(
            copy_mul,
            crate::replay::ReplayKernargBinding::PositionMulU32 {
                offset: copy_mul.offset(),
                factor: index_kv_width as u32,
            },
            "index-key write must declare `position * index_kv_width`"
        );
        check(&launches[3], copy_mul, dco as u32, "index-key row offset");

        gpu.free_tensor(raw_gpu).expect("free raw");
        gpu.free_tensor(pooled_gpu).expect("free pooled");
        gpu.free_tensor(query_gpu).expect("free query");
        gpu.free_tensor(selected_gpu).expect("free selected");
        gpu.free_tensor(q_with_gate_gpu).expect("free q");
        gpu.free_tensor(full_keys_gpu).expect("free keys");
        gpu.free_tensor(full_values_gpu).expect("free values");
        gpu.free_tensor(output_gpu).expect("free attention output");
        gpu.free_tensor(copy_src).expect("free copy src");
        gpu.free_tensor(copy_dst).expect("free copy dst");
    }

    /// The retained-replay shape pin (a fixed grid or LDS bound larger than the
    /// active length) must not change a single output byte: the kernels mask, so
    /// a bigger reservation is never read. This is the premise the G3 shape
    /// decision rests on, tested directly for all three QSA launches, including
    /// the `block_count == 0` case where the wrapper's symbol choice differs.
    #[test]
    fn pinned_qsa_shapes_are_bit_identical_to_derived_shapes() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };

        // ── pool: an oversized masked grid ──
        let index_dim = 8usize;
        let compress = 4usize;
        let blocks = 2usize;
        let cells = blocks * compress * index_dim;
        let raw: Vec<f32> = (0..cells)
            .map(|i| ((i * 37 % 251) as f32 - 125.0) / 37.0)
            .collect();
        let raw_gpu = gpu.upload_f32(&raw, &[cells]).expect("raw upload");
        let pool_run = |gpu: &mut Gpu, bound: usize| {
            let pooled = gpu
                .zeros(&[blocks * index_dim], DType::F32)
                .expect("pooled allocation");
            indexed_attention_pool_rope(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &raw_gpu,
                    pooled: &pooled,
                    norm: None,
                    block_count: blocks,
                    compress,
                    index_dim,
                    position: None,
                    grid_bound: bound,
                },
            )
            .expect("QSA pool/RoPE");
            let values = gpu.download_f32(&pooled).expect("pooled download");
            gpu.free_tensor(pooled).expect("free pooled");
            values
        };
        let pooled_derived = pool_run(&mut gpu, blocks);
        let pooled_pinned = pool_run(&mut gpu, blocks + 6);
        assert_eq!(
            pooled_pinned, pooled_derived,
            "a masked pool grid above the active block count changed the result"
        );

        // ── select: a larger LDS reservation, at an active count and at zero ──
        let index_heads = 2usize;
        let query_elements = index_heads * index_dim;
        let rows = 1usize;
        let capacity = 8usize;
        let budget_blocks = 2usize;
        let query: Vec<f32> = (0..rows * query_elements)
            .map(|i| ((i * 53 % 199) as f32 - 100.0) / 199.0)
            .collect();
        let query_gpu = gpu
            .upload_f32(&query, &[rows * query_elements])
            .expect("query upload");
        let pooled_gpu = gpu
            .zeros(&[blocks * index_dim], DType::F32)
            .expect("pooled allocation");
        // `position_start` must satisfy the declared formula
        // `(position_start + rows) / compress == active`, so the active count and
        // the position are always consistent (the wrapper refuses a mismatch).
        let select_run = |gpu: &mut Gpu, active: usize, bound: usize, position_start: usize| {
            let selected = gpu
                .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
                .expect("selected allocation");
            indexed_attention_select_batch(
                gpu,
                &IndexedAttentionSelectBatch {
                    query: &query_gpu,
                    pooled: &pooled_gpu,
                    selected: &selected,
                    rows,
                    query_row_stride: query_elements,
                    block_count: active,
                    index_heads,
                    index_dim,
                    budget_blocks,
                    compress,
                    position_start,
                    capacity,
                    shape_blocks: bound,
                },
            )
            .expect("QSA select");
            let mut bytes = vec![0u8; rows * capacity * std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &selected.buf)
                .expect("selected download");
            gpu.free_tensor(selected).expect("free selected");
            bytes
        };
        let active_position = blocks * compress - rows;
        assert_eq!(
            select_run(&mut gpu, blocks, capacity, active_position),
            select_run(&mut gpu, blocks, blocks, active_position),
            "a larger select LDS reservation changed the selection"
        );
        // Zero complete blocks: position 0 with one row. This is the shape the
        // production lowering hits on the first `compress` tokens, so the
        // batched symbol at zero must equal the serial symbol it replaces.
        assert_eq!(
            select_run(&mut gpu, 0, capacity, 0),
            select_run(&mut gpu, 0, 0, 0),
            "the batched select symbol at an active count of zero diverged from the serial symbol"
        );

        // ── attention: a larger LDS reservation ──
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let full_capacity = 8usize;
        let position_start = 2usize;
        let q_with_gate: Vec<f32> = (0..rows * n_heads * 2 * head_dim)
            .map(|i| ((i * 29 % 173) as f32 - 86.0) / 173.0)
            .collect();
        let kv: Vec<f32> = (0..full_capacity * n_kv_heads * head_dim)
            .map(|i| ((i * 41 % 211) as f32 - 105.0) / 211.0)
            .collect();
        let selected_tokens: Vec<i32> = vec![0, 1, 2, -1, -1, -1, -1, -1];
        let q_with_gate_gpu = gpu
            .upload_f32(&q_with_gate, &[q_with_gate.len()])
            .expect("q upload");
        let full_keys_gpu = gpu.upload_f32(&kv, &[kv.len()]).expect("keys upload");
        let full_values_gpu = gpu.upload_f32(&kv, &[kv.len()]).expect("values upload");
        let selected_gpu = gpu
            .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let selected_bytes = selected_tokens
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        gpu.hip
            .memcpy_htod(&selected_gpu.buf, &selected_bytes)
            .expect("selected upload");
        let attention_run = |gpu: &mut Gpu, bound: usize| {
            let output = gpu
                .zeros(&[rows * n_heads * head_dim], DType::F32)
                .expect("attention output allocation");
            indexed_attention_attention_batch(
                gpu,
                &IndexedAttentionAttentionBatch {
                    q_with_gate: &q_with_gate_gpu,
                    full_keys: &full_keys_gpu,
                    full_values: &full_values_gpu,
                    selected: &selected_gpu,
                    output: &output,
                    rows,
                    position_start,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    budget_blocks,
                    compress,
                    capacity,
                    full_capacity,
                    format: QsaKvFormat::F32,
                    shape_selected: bound,
                },
            )
            .expect("QSA attention");
            let values = gpu.download_f32(&output).expect("attention download");
            gpu.free_tensor(output).expect("free attention output");
            values
        };
        let derived_selected = (position_start + rows).min(capacity);
        assert_eq!(
            attention_run(&mut gpu, capacity),
            attention_run(&mut gpu, derived_selected),
            "a larger attention LDS reservation changed the output"
        );

        gpu.free_tensor(raw_gpu).expect("free raw");
        gpu.free_tensor(query_gpu).expect("free query");
        gpu.free_tensor(pooled_gpu).expect("free pooled");
        gpu.free_tensor(q_with_gate_gpu).expect("free q");
        gpu.free_tensor(full_keys_gpu).expect("free keys");
        gpu.free_tensor(full_values_gpu).expect("free values");
        gpu.free_tensor(selected_gpu).expect("free selected");
    }

    /// The grouped QSA attention kernels must equal the per-head batched
    /// kernel bit for bit (24 heads, head_dim 256; 2 KV heads is the
    /// production shape): permuted selections, invalid slots, a partial key
    /// tile and rows with and without a tail all included.
    #[test]
    fn qsa_attention_grouped_is_bit_identical_to_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        // hg12 (2 KV heads) over three and nine 256-row tiles, hg4 (6 KV heads).
        for (n_kv_heads, position_start, full_capacity, budget_blocks) in [
            (2usize, 610usize, 640usize, 150usize),
            (2, 2100, 2200, 512),
            (6, 610, 640, 150),
        ] {
            let (n_heads, head_dim, compress) = (24usize, 256usize, 4usize);
            let rows = 20usize;
            let capacity = budget_blocks * compress + compress - 1;
            let lcg = |seed: usize, n: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2003) as f32 - 1001.0)
                            / 997.0
                    })
                    .collect()
            };
            let q = lcg(1, rows * n_heads * 2 * head_dim);
            let keys = lcg(7, full_capacity * n_kv_heads * head_dim);
            let values = lcg(13, full_capacity * n_kv_heads * head_dim);
            let mut selected = vec![-1i32; rows * capacity];
            for row in 0..rows {
                let visible = position_start + row + 1;
                let blocks = visible / compress;
                let chosen = budget_blocks.min(blocks);
                // Descending-stride block choice, then the tail, as the selector emits.
                for slot in 0..chosen {
                    let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                    for r in 0..compress {
                        selected[row * capacity + slot * compress + r] =
                            block * compress as i32 + r as i32;
                    }
                }
                let mut offset = chosen * compress;
                for token in blocks * compress..visible {
                    selected[row * capacity + offset] = token as i32;
                    offset += 1;
                }
                // An invalid slot inside the active length must be skipped.
                selected[row * capacity + 3] = -1;
            }
            let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
            let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
            let values_gpu = gpu
                .upload_f32(&values, &[values.len()])
                .expect("values upload");
            let selected_gpu = gpu
                .zeros(&[selected.len() * std::mem::size_of::<i32>()], DType::Raw)
                .expect("selected allocation");
            let bytes = selected
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected_gpu.buf, &bytes)
                .expect("selected upload");
            let run = |gpu: &mut Gpu, routes: QsaAttentionRoutes| {
                let output = gpu
                    .zeros(&[rows * n_heads * head_dim], DType::F32)
                    .expect("output allocation");
                indexed_attention_attention_batch_impl(
                    gpu,
                    &IndexedAttentionAttentionBatch {
                        q_with_gate: &q_gpu,
                        full_keys: &keys_gpu,
                        full_values: &values_gpu,
                        selected: &selected_gpu,
                        output: &output,
                        rows,
                        position_start,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        budget_blocks,
                        compress,
                        capacity,
                        full_capacity,
                        format: QsaKvFormat::F32,
                        shape_selected: capacity,
                    },
                    routes,
                )
                .expect("QSA attention");
                let values = gpu.download_f32(&output).expect("output download");
                gpu.free_tensor(output).expect("free output");
                values
            };
            let reference = run(&mut gpu, QsaAttentionRoutes::Exact);
            let grouped = run(&mut gpu, QsaAttentionRoutes::Grouped);
            assert!(
                reference.iter().any(|v| *v != 0.0),
                "reference output is all zero"
            );
            let differing = reference
                .iter()
                .zip(&grouped)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "grouped QSA attention differs in {differing} cells"
            );
            for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
    }

    /// The dense (every row selects its whole causal window) and sparse
    /// (per-row block selection past the budget) F16 WMMA routes must match the
    /// exact per-head kernel to F16-rounding accuracy at the production head
    /// shape: chunk offsets, a partial row tile, permuted blocks with the
    /// causal tail, and an invalid slot inside a sparse row.
    #[test]
    fn qsa_f16_wmma_routes_match_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_wmma_w32() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx11 WMMA");
            return;
        }
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        // (rows, position_start, full_capacity, budget_blocks, capacity, sparse)
        for (rows, position_start, full_capacity, budget_blocks, capacity, sparse) in [
            (521usize, 100usize, 640usize, 2048usize, 640usize, false),
            (37, 2100, 2200, 512, 512 * 4 + 3, true),
        ] {
            let lcg = |seed: usize, n: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2003) as f32 - 1001.0)
                            / 997.0
                    })
                    .collect()
            };
            let q = lcg(1, rows * n_heads * 2 * head_dim);
            let keys = lcg(7, full_capacity * n_kv_heads * head_dim);
            let values = lcg(13, full_capacity * n_kv_heads * head_dim);
            // Dense: the whole window, as indexed_attention_select emits it when
            // the budget covers every block.  Sparse: permuted whole blocks, then
            // the tail, with one invalid slot.
            let mut selected = vec![-1i32; rows * capacity];
            for row in 0..rows {
                let visible = position_start + row + 1;
                if !sparse {
                    for token in 0..visible {
                        selected[row * capacity + token] = token as i32;
                    }
                    continue;
                }
                let blocks = visible / compress;
                let chosen = budget_blocks.min(blocks);
                for slot in 0..chosen {
                    let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                    for r in 0..compress {
                        selected[row * capacity + slot * compress + r] =
                            block * compress as i32 + r as i32;
                    }
                }
                let mut offset = chosen * compress;
                for token in blocks * compress..visible {
                    selected[row * capacity + offset] = token as i32;
                    offset += 1;
                }
                selected[row * capacity + 5] = -1;
            }
            let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
            let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
            let values_gpu = gpu
                .upload_f32(&values, &[values.len()])
                .expect("values upload");
            let selected_gpu = gpu
                .zeros(&[selected.len() * std::mem::size_of::<i32>()], DType::Raw)
                .expect("selected allocation");
            let bytes = selected
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected_gpu.buf, &bytes)
                .expect("selected upload");
            let run = |gpu: &mut Gpu, routes: QsaAttentionRoutes| {
                let output = gpu
                    .zeros(&[rows * n_heads * head_dim], DType::F32)
                    .expect("output allocation");
                indexed_attention_attention_batch_impl(
                    gpu,
                    &IndexedAttentionAttentionBatch {
                        q_with_gate: &q_gpu,
                        full_keys: &keys_gpu,
                        full_values: &values_gpu,
                        selected: &selected_gpu,
                        output: &output,
                        rows,
                        position_start,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        budget_blocks,
                        compress,
                        capacity,
                        full_capacity,
                        format: QsaKvFormat::F32,
                        shape_selected: capacity,
                    },
                    routes,
                )
                .expect("QSA attention");
                let values = gpu.download_f32(&output).expect("output download");
                gpu.free_tensor(output).expect("free output");
                values
            };
            let reference = run(&mut gpu, QsaAttentionRoutes::Exact);
            let fast = run(&mut gpu, QsaAttentionRoutes::F16);
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (r, d) in reference.iter().zip(&fast) {
                err += (*r as f64 - *d as f64).powi(2);
                norm += (*r as f64).powi(2);
            }
            let rel = (err / norm).sqrt();
            assert!(norm > 0.0, "reference output is all zero");
            // F16 operands and probabilities: ~1e-3 relative; a wrong row, head,
            // key range or dim mapping lands near 1.
            assert!(rel < 5e-3, "QSA WMMA (sparse={sparse}) rel L2 {rel:.3e}");
            // It is the WMMA route (F16 rounding), not the exact kernel.
            assert!(rel > 0.0, "QSA WMMA route (sparse={sparse}) did not run");
            for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
    }

    /// The gathered F16 WMMA route must match the exact per-head kernel of
    /// the same state format to F16-rounding accuracy: rows before and past
    /// the budget, causal tails, a `-1` slot per row, and rows whose blocks
    /// are emitted in descending token order (the per-token V path).  The
    /// query is scaled so the softmax is peaked: with the flat LCG query a
    /// mis-paired key and value barely moves the output.
    #[test]
    fn qsa_gathered_wmma_matches_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let format = if gpu.arch_caps.is_gfx1151() {
            QsaKvFormat::F32
        } else if gpu.arch_caps.is_gfx1201() {
            QsaKvFormat::Fp8
        } else {
            eprintln!("skip: the gathered route is gfx1151 / gfx1201 only");
            return;
        };
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        // Visible 1901..2420 against a 512-block budget: 475..605 blocks.
        let (rows, position_start, full_capacity) = (520usize, 1900usize, 2432usize);
        let budget_blocks = 512usize;
        let capacity = budget_blocks * compress + compress - 1;
        let mut q = qsa_lcg(1, rows * n_heads * 2 * head_dim);
        for head_row in q.chunks_mut(2 * head_dim) {
            head_row[..head_dim].iter_mut().for_each(|v| *v *= 8.0);
        }
        let keys = qsa_lcg(7, full_capacity * n_kv_heads * head_dim);
        let values = qsa_lcg(13, full_capacity * n_kv_heads * head_dim);
        let mut selected = vec![-1i32; rows * capacity];
        for row in 0..rows {
            let visible = position_start + row + 1;
            let blocks = visible / compress;
            let chosen = budget_blocks.min(blocks);
            for slot in 0..chosen {
                let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                for r in 0..compress {
                    // Odd rows list each block's tokens in descending order.
                    let token = if row % 2 == 1 { compress - 1 - r } else { r };
                    selected[row * capacity + slot * compress + r] =
                        block * compress as i32 + token as i32;
                }
            }
            let mut offset = chosen * compress;
            for token in blocks * compress..visible {
                selected[row * capacity + offset] = token as i32;
                offset += 1;
            }
            selected[row * capacity + 3] = -1;
        }
        let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
        let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
        let values_gpu = gpu.upload_f32(&values, &[values.len()]).expect("values upload");
        let (k, v) = if format == QsaKvFormat::F32 {
            (keys_gpu, values_gpu)
        } else {
            let units = format.kv_row_units(n_kv_heads, head_dim);
            let k = gpu.zeros(&[full_capacity * units], format.kv_dtype()).expect("K");
            let v = gpu.zeros(&[full_capacity * units], format.kv_dtype()).expect("V");
            indexed_attention_cache_append_batch(
                &mut gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &keys_gpu,
                    value: &values_gpu,
                    full_keys: &k,
                    full_values: &v,
                    rows: full_capacity,
                    position_start: 0,
                    kv_heads: n_kv_heads,
                    head_dim,
                    format,
                },
            )
            .expect("QSA append");
            gpu.free_tensor(keys_gpu).expect("free");
            gpu.free_tensor(values_gpu).expect("free");
            (k, v)
        };
        let selected_gpu = gpu
            .zeros(&[selected.len() * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let bytes = selected.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();
        gpu.hip.memcpy_htod(&selected_gpu.buf, &bytes).expect("selected upload");
        let output = gpu
            .zeros(&[rows * n_heads * head_dim], DType::F32)
            .expect("output allocation");
        let p = IndexedAttentionAttentionBatch {
            q_with_gate: &q_gpu,
            full_keys: &k,
            full_values: &v,
            selected: &selected_gpu,
            output: &output,
            rows,
            position_start,
            n_heads,
            n_kv_heads,
            head_dim,
            budget_blocks,
            compress,
            capacity,
            full_capacity,
            format,
            shape_selected: capacity,
        };
        indexed_attention_attention_batch_impl(&mut gpu, &p, QsaAttentionRoutes::Exact)
            .expect("per-head QSA");
        let reference = gpu.download_f32(&output).expect("reference download");
        let end_position = position_start + rows;
        qsa_gathered_wmma(&mut gpu, &p, end_position, capacity).expect("gathered QSA");
        let gathered = gpu.download_f32(&output).expect("gathered download");
        let rel = rel_l2(&reference, &gathered);
        eprintln!("gathered QSA WMMA ({}) rel L2 {rel:.3e}", format.name());
        // F16 operands and probabilities: ~1e-4 here; a wrong entry, head,
        // dim or key/value pairing lands near 1e-1 or above.
        assert!(rel < 5e-3, "gathered QSA WMMA ({}) rel L2 {rel:.3e}", format.name());
        assert!(rel > 0.0, "gathered QSA WMMA route did not run");
        for tensor in [q_gpu, k, v, selected_gpu, output] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    struct SelectCase {
        compress: usize,
        index_heads: usize,
        index_dim: usize,
        rows: usize,
        block_count: usize,
        budget_blocks: usize,
        capacity: usize,
        position_start: usize,
    }

    fn run_select_case(
        gpu: &mut Gpu,
        case: &SelectCase,
        pooled: &[f32],
        query: &[f32],
        shape_blocks: usize,
    ) -> Vec<i32> {
        let pooled_gpu = gpu
            .upload_f32(pooled, &[pooled.len().max(1)])
            .expect("pooled upload");
        let query_gpu = gpu
            .upload_f32(query, &[query.len().max(1)])
            .expect("query upload");
        let selected = gpu
            .zeros(&[case.rows * case.capacity * 4], DType::Raw)
            .expect("selected allocation");
        indexed_attention_select_batch(
            gpu,
            &IndexedAttentionSelectBatch {
                query: &query_gpu,
                pooled: &pooled_gpu,
                selected: &selected,
                rows: case.rows,
                query_row_stride: case.index_heads * case.index_dim,
                block_count: case.block_count,
                index_heads: case.index_heads,
                index_dim: case.index_dim,
                budget_blocks: case.budget_blocks,
                compress: case.compress,
                position_start: case.position_start,
                capacity: case.capacity,
                shape_blocks,
            },
        )
        .expect("QSA select");
        let mut bytes = vec![0u8; case.rows * case.capacity * 4];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &selected.buf)
            .expect("selected download");
        gpu.free_tensor(selected).expect("free selected");
        gpu.free_tensor(pooled_gpu).expect("free pooled");
        gpu.free_tensor(query_gpu).expect("free query");
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 chunk")))
            .collect()
    }

    /// Arena bound whose score row exceeds the LDS limit: the global route.
    const GLOBAL_BOUND: usize = QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES / 4 + 1;

    /// The serial selection-sort oracle kernel for `case`.
    fn run_select_serial(gpu: &mut Gpu, case: &SelectCase, pooled: &[f32], query: &[f32]) -> Vec<i32> {
        SELECT_FORCE_SERIAL.with(|flag| flag.set(true));
        let selected = run_select_case(gpu, case, pooled, query, case.block_count);
        SELECT_FORCE_SERIAL.with(|flag| flag.set(false));
        selected
    }

    /// Scores that increase with the block index must select blocks in
    /// descending order, and equal scores must break ties toward the lower
    /// block index — the order the serial scan's strict `>` comparison
    /// produced. This pins the semantics rather than mutual agreement.
    #[test]
    fn batched_select_orders_blocks_by_score_then_index() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let compress = 4usize;
        let index_dim = 4usize;
        // Block b contributes b to the dot product, so score(b) = b / 2.
        let pooled: Vec<f32> = (0..4 * index_dim)
            .map(|i| {
                if i % index_dim == 0 {
                    (i / index_dim) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let query = vec![1.0f32; index_dim];
        let case = SelectCase {
            compress,
            index_heads: 1,
            index_dim,
            rows: 1,
            block_count: 4,
            budget_blocks: 8,
            capacity: 18,
            position_start: 4 * compress,
        };
        // Slot 16 is the tail: block 4 covers tokens 16..19 but `visible` is 17,
        // so the partial block still contributes its one visible token.
        let selected = run_select_case(&mut gpu, &case, &pooled, &query, case.block_count);
        assert_eq!(
            selected,
            vec![12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 16, -1],
            "descending-score selection changed"
        );

        // All-zero pooled keys tie every block at score 0.
        let tied = vec![0.0f32; 4 * index_dim];
        let selected = run_select_case(&mut gpu, &case, &tied, &query, case.block_count);
        assert_eq!(
            selected,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1],
            "tie-break by block index changed"
        );
    }

    /// The parallel rank path and the serial selection-sort fallback are two
    /// symbols for the same logical selection, chosen only by the LDS
    /// reservation; they must emit identical `selected` bytes. A
    /// `shape_blocks` above the dynamic-LDS limit selects the serial symbol.
    #[test]
    fn batched_select_ranking_matches_the_serial_selection_sort() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 65_536.0) - 8.0
        };

        for &compress in &[2usize, 4, 8] {
            for &index_dim in &[8usize, 128] {
                for &index_heads in &[1usize, 4] {
                    // 37 rows: three sixteen-row scoring groups, the last
                    // partial, each row with its own visible block count.
                    for &rows in &[1usize, 5, 37] {
                        // `(position_start + rows) / compress` must equal the
                        // wrapper's declared block count.
                        if rows < 37 && rows > compress - 1 {
                            continue;
                        }
                        // Rows of at least two budgets take the threshold
                        // path; integer keys force ties at the threshold.
                        for &block_count in &[0usize, 1, 3, 17, 72, 128, 500, 1100, 2100] {
                            for &budget_blocks in &[1usize, 4, 64, 512] {
                                // The serial reference costs budget^2 * blocks:
                                // past 500 blocks only the pinned geometry
                                // runs the 512-block budget.
                                let pinned = compress == 4 && index_heads == 4 && index_dim == 128;
                                if block_count > 500
                                    && budget_blocks > 64
                                    && !(pinned && block_count == 1100)
                                {
                                    continue;
                                }
                                // Many rows: the pinned geometry, smaller budgets.
                                if rows == 37
                                    && (!pinned
                                        || budget_blocks > 64
                                        || block_count * compress + compress - 1 < rows)
                                {
                                    continue;
                                }
                                for ties in [false, true] {
                                    let case = SelectCase {
                                        compress,
                                        index_heads,
                                        index_dim,
                                        rows,
                                        block_count,
                                        budget_blocks,
                                        capacity: budget_blocks * compress + compress - 1,
                                        position_start: if rows < compress {
                                            block_count * compress
                                        } else {
                                            block_count * compress + compress - 1 - rows
                                        },
                                    };
                                    let pooled: Vec<f32> = (0..block_count * index_dim + index_dim)
                                        .map(|_| if ties { next().round() } else { next() })
                                        .collect();
                                    let query: Vec<f32> = (0..rows * index_heads * index_dim)
                                        .map(|_| if ties { next().round() } else { next() })
                                        .collect();
                                    let parallel = run_select_case(
                                        &mut gpu,
                                        &case,
                                        &pooled,
                                        &query,
                                        block_count,
                                    );
                                    let serial = run_select_serial(&mut gpu, &case, &pooled, &query);
                                    let global = run_select_case(
                                        &mut gpu,
                                        &case,
                                        &pooled,
                                        &query,
                                        GLOBAL_BOUND,
                                    );
                                    assert_eq!(
                                        global, serial,
                                        "global-score selection diverged from the serial selection sort: \
                                         compress={compress} heads={index_heads} dim={index_dim} \
                                         rows={rows} blocks={block_count} budget={budget_blocks} ties={ties}"
                                    );
                                    assert_eq!(
                                    parallel, serial,
                                    "parallel ranking diverged from the serial selection sort: \
                                     compress={compress} heads={index_heads} dim={index_dim} \
                                     rows={rows} blocks={block_count} budget={budget_blocks} ties={ties}"
                                );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Production QSA geometry past its budget (512 of 4096 blocks, 4 heads x
    /// 128, compress 4): the parallel radix-select route must emit the serial
    /// selection sort's bytes, including when the budget boundary falls inside
    /// a run of equal scores (ReLU zeros), where the lower block index wins.
    #[test]
    fn past_budget_select_matches_serial_on_production_geometry() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (compress, index_heads, index_dim, rows, block_count, budget_blocks) =
            (4usize, 4usize, 128usize, 3usize, 4096usize, 512usize);
        let mut state = 0x2545_f491u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0
        };
        // Positive queries: a block whose pooled row is all negative scores 0.
        let query: Vec<f32> = (0..rows * index_heads * index_dim).map(|_| next()).collect();
        // positive_every: 1 = every block scores > 0 (threshold above zero);
        // 10 = ~410 positive blocks, so the boundary is inside the zero run;
        // 0 = every block ties at zero.
        for positive_every in [1usize, 10, 0] {
            let pooled: Vec<f32> = (0..block_count * index_dim)
                .map(|i| {
                    let block = i / index_dim;
                    let value = next() - 0.25;
                    if positive_every != 0 && block % positive_every == 0 {
                        value
                    } else {
                        -value.abs() - 0.01
                    }
                })
                .collect();
            let case = SelectCase {
                compress,
                index_heads,
                index_dim,
                rows,
                block_count,
                budget_blocks,
                capacity: budget_blocks * compress + compress - 1,
                position_start: block_count * compress,
            };
            let parallel = run_select_case(&mut gpu, &case, &pooled, &query, block_count);
            let serial = run_select_serial(&mut gpu, &case, &pooled, &query);
            let global = run_select_case(&mut gpu, &case, &pooled, &query, GLOBAL_BOUND);
            assert_eq!(
                parallel, serial,
                "past-budget radix select diverged (positive_every={positive_every})"
            );
            assert_eq!(
                global, serial,
                "past-budget global-score select diverged (positive_every={positive_every})"
            );
            if positive_every == 0 {
                // All tied: the lowest 512 block indices, in index order.
                let expected: Vec<i32> = (0..(budget_blocks * compress) as i32).collect();
                assert_eq!(&parallel[..budget_blocks * compress], &expected[..]);
            }
        }
    }

    /// A prefill-sized batch (more rows than one global-score launch holds)
    /// runs in row groups; each row's selection must equal the single-launch
    /// LDS route's, including rows whose visible prefix ends mid-block.
    #[test]
    fn global_score_row_groups_match_the_lds_route() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (compress, index_heads, index_dim, rows, block_count, budget_blocks) =
            (4usize, 4usize, 128usize, 2 * QSA_SELECT_GLOBAL_ROWS + 37, 4096usize, 512usize);
        let mut state = 0x7f4a_7c15u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 8_388_608.0 - 1.0
        };
        let query: Vec<f32> = (0..rows * index_heads * index_dim).map(|_| next()).collect();
        let pooled: Vec<f32> = (0..block_count * index_dim).map(|_| next()).collect();
        let case = SelectCase {
            compress,
            index_heads,
            index_dim,
            rows,
            block_count,
            budget_blocks,
            capacity: budget_blocks * compress + compress - 1,
            // (position_start + rows) / compress == block_count.
            position_start: block_count * compress - rows + 2,
        };
        let lds = run_select_case(&mut gpu, &case, &pooled, &query, block_count);
        let global = run_select_case(&mut gpu, &case, &pooled, &query, GLOBAL_BOUND);
        assert_eq!(global, lds, "grouped global-score selection diverged from the LDS route");
    }

    #[test]
    fn qsa_reuse_selection_preserves_order_and_boundaries() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let capacity = 4usize;
        let mut selected = gpu
            .zeros(&[capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let selected_len_out = gpu
            .zeros(&[std::mem::size_of::<i32>()], DType::Raw)
            .expect("length allocation");
        let cases: [([i32; 4], usize, usize, [i32; 4], i32); 4] = [
            ([-1, -1, -1, -1], 0usize, 7usize, [7, -1, -1, -1], 1),
            ([0, 3, 1, 99], 4, 2, [0, 1, 2, -1], 3),
            ([0, 2, 1, -1], 3, 2, [0, 2, 1, -1], 3),
            ([0, 1, 2, 3], 4, 4, [0, 1, 2, 3], 4),
        ];
        for (input, selected_len, position, expected, expected_len) in cases {
            let input_bytes = input
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected.buf, &input_bytes)
                .expect("upload selected row");
            gpu.hip
                .memset(&selected_len_out.buf, 0, selected_len_out.byte_size())
                .expect("clear selected length");
            indexed_attention_reuse_selection(
                &mut gpu,
                &IndexedAttentionReuseSelection {
                    selected: &selected,
                    selected_len,
                    position,
                    capacity,
                    selected_len_out: &selected_len_out,
                },
            )
            .expect("reuse selection");

            let mut output_bytes = vec![0u8; capacity * std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut output_bytes, &selected.buf)
                .expect("download selected row");
            let output = output_bytes
                .chunks_exact(std::mem::size_of::<i32>())
                .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
                .collect::<Vec<_>>();
            assert_eq!(output, expected);
            let mut length_bytes = [0u8; std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut length_bytes, &selected_len_out.buf)
                .expect("download selected length");
            assert_eq!(i32::from_ne_bytes(length_bytes), expected_len);
        }
        gpu.free_tensor(selected).expect("free selected");
        gpu.free_tensor(selected_len_out)
            .expect("free selected length");
    }
    #[test]
    fn qsa_norm_rope_matches_production_shape_and_preserves_gates() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const HEAD_STRIDE: usize = 2 * HEAD_DIM;
        const ROTARY_DIM: usize = 64;
        const POSITION: usize = 11;
        let mut input = vec![0.0f32; HEADS * HEAD_STRIDE];
        for head in 0..HEADS {
            for channel in 0..HEAD_DIM {
                input[head * HEAD_STRIDE + channel] =
                    (((head * HEAD_DIM + channel) * 37 % 251) as f32 - 125.0) / 37.0;
                input[head * HEAD_STRIDE + HEAD_DIM + channel] =
                    1000.0 + (head * HEAD_DIM + channel) as f32;
            }
        }
        let values = gpu
            .upload_f32(&input, &[input.len()])
            .expect("query/gate upload");
        let norm = gpu
            .zeros(&[HEAD_DIM], DType::BF16)
            .expect("zero norm allocation");
        indexed_attention_norm_rope(
            &mut gpu,
            &IndexedAttentionNormRope {
                values: &values,
                norm: &norm,
                heads: HEADS,
                head_dim: HEAD_DIM,
                head_stride: HEAD_STRIDE,
                position: POSITION,
                rotary_dim: ROTARY_DIM,
            },
        )
        .expect("QSA norm/RoPE");
        let actual = gpu.download_f32(&values).expect("query/gate download");
        let mut expected = input.clone();
        for head in 0..HEADS {
            let start = head * HEAD_STRIDE;
            let inv = (input[start..start + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / HEAD_DIM as f32
                + 1.0e-6)
                .sqrt()
                .recip();
            for channel in 0..HEAD_DIM {
                expected[start + channel] = input[start + channel] * inv;
            }
            for channel in 0..ROTARY_DIM / 2 {
                let angle = POSITION as f32
                    / 10_000_000.0f32.powf(2.0 * channel as f32 / ROTARY_DIM as f32);
                let (sine, cosine) = angle.sin_cos();
                let first = input[start + channel] * inv;
                let second = input[start + ROTARY_DIM / 2 + channel] * inv;
                expected[start + channel] = first * cosine - second * sine;
                expected[start + ROTARY_DIM / 2 + channel] = first * sine + second * cosine;
            }
        }
        for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1.0e-4,
                "QSA norm/RoPE mismatch at {index}: got {got}, expected {want}"
            );
        }
        gpu.free_tensor(values).expect("free query/gate");
        gpu.free_tensor(norm).expect("free norm");
    }

    /// Formats this device has QSA kernels for, beyond F32 (fp8: gfx12 only).
    fn quantized_qsa_formats(gpu: &Gpu) -> Vec<QsaKvFormat> {
        if gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201() {
            vec![QsaKvFormat::Fp8]
        } else {
            eprintln!("skip: fp8 QSA K/V needs a gfx12 GPU");
            Vec::new()
        }
    }

    fn qsa_lcg(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2003) as f32 - 1001.0) / 997.0
            })
            .collect()
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let mantissa = (bits & 0x3FF) as f32;
        sign * if exponent == 0 {
            mantissa / 1024.0 * 2f32.powi(-14)
        } else {
            (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
        }
    }

    /// Round-to-nearest-even BF16 bits of a finite value.
    fn bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn e4m3_to_f32(code: u8) -> f32 {
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = ((code >> 3) & 0xF) as i32;
        let mantissa = (code & 7) as f32;
        sign * if exponent == 0 {
            mantissa / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
        }
    }

    /// Host dequantization of `tokens` quantized K or V rows.
    fn dequantize_qsa_rows(
        format: QsaKvFormat,
        bytes: &[u8],
        tokens: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let row = format.kv_row_bytes(kv_heads, head_dim);
        let half = |at: usize| f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
        let mut values = Vec::with_capacity(tokens * kv_heads * head_dim);
        for token in 0..tokens {
            let base = token * row;
            for element in 0..kv_heads * head_dim {
                values.push(match format {
                    QsaKvFormat::Fp8 => {
                        let scale = half(base + kv_heads * head_dim + element / head_dim * 2);
                        e4m3_to_f32(bytes[base + element]) * scale
                    }
                    QsaKvFormat::F32 => unreachable!("F32 rows are not quantized"),
                });
            }
        }
        values
    }

    fn rel_l2(reference: &[f32], actual: &[f32]) -> f64 {
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for (r, a) in reference.iter().zip(actual) {
            err += (*r as f64 - *a as f64).powi(2);
            norm += (*r as f64).powi(2);
        }
        (err / norm).sqrt()
    }

    fn download_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u8> {
        let mut bytes = vec![0u8; tensor.byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &tensor.buf)
            .expect("download");
        bytes
    }

    /// The decode prologue and the prefill appends must write the same
    /// quantized K/V rows and BF16 raw index keys for the same rows (decode
    /// after prefill reads what token-by-token decode would have written),
    /// and those rows must dequantize to the keys and values they encode.
    #[test]
    fn qsa_quantized_prologue_and_prefill_appends_write_identical_rows() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (heads, kv_heads, head_dim) = (4usize, 2usize, 256usize);
        let (index_heads, index_dim, index_kv_width) = (4usize, 128usize, 128usize);
        let (rows, position, capacity) = (3usize, 5usize, 16usize);
        let index_width = index_heads * index_dim + index_kv_width;
        let kv_width = kv_heads * head_dim;
        for format in quantized_qsa_formats(&gpu) {
            let row_units = format.kv_row_units(kv_heads, head_dim);
            let index_row = qsa_lcg(3, rows * index_width);
            let index_gpu = gpu.upload_f32(&index_row, &[index_row.len()]).expect("index");
            let qgate = qsa_lcg(5, rows * 2 * heads * head_dim);
            let qgate_gpu = gpu.upload_f32(&qgate, &[qgate.len()]).expect("qgate");
            let keys = qsa_lcg(7, rows * kv_width);
            let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys");
            let values = qsa_lcg(11, rows * kv_width);
            let values_gpu = gpu.upload_f32(&values, &[values.len()]).expect("values");
            let index_norm = gpu.zeros(&[index_dim], DType::BF16).expect("index norm");
            let head_norm = gpu.zeros(&[head_dim], DType::BF16).expect("head norm");
            let cache = |gpu: &mut Gpu| {
                gpu.zeros(&[capacity * row_units], format.kv_dtype())
                    .expect("cache")
            };
            let (decode_k, decode_v, prefill_k, prefill_v) =
                (cache(&mut gpu), cache(&mut gpu), cache(&mut gpu), cache(&mut gpu));
            let decode_raw = gpu
                .zeros(&[capacity * index_kv_width], DType::BF16)
                .expect("raw");
            let prefill_raw = gpu
                .zeros(&[capacity * index_kv_width], DType::BF16)
                .expect("raw");
            indexed_attention_decode_prologue(
                &mut gpu,
                &IndexedAttentionDecodePrologue {
                    index_row: &index_gpu,
                    qgate: &qgate_gpu,
                    keys: &keys_gpu,
                    values: &values_gpu,
                    full_keys: &decode_k,
                    full_values: &decode_v,
                    raw_index_keys: &decode_raw,
                    index_q_norm: &index_norm,
                    q_norm: &head_norm,
                    k_norm: &head_norm,
                    index_heads,
                    index_dim,
                    index_kv_width,
                    heads,
                    kv_heads,
                    head_dim,
                    position,
                    rows,
                    format,
                },
            )
            .expect("QSA prologue");
            // The prologue leaves the normed keys and rounded index keys in
            // its scratch: the prefill appends start from those.
            indexed_attention_cache_append_batch(
                &mut gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &keys_gpu,
                    value: &values_gpu,
                    full_keys: &prefill_k,
                    full_values: &prefill_v,
                    rows,
                    position_start: position,
                    kv_heads,
                    head_dim,
                    format,
                },
            )
            .expect("QSA append");
            indexed_attention_index_key_append_batch(
                &mut gpu,
                &IndexedAttentionIndexKeyAppendBatch {
                    index_rows: &index_gpu,
                    raw_index_keys: &prefill_raw,
                    rows,
                    index_q_width: index_heads * index_dim,
                    index_kv_width,
                    position_start: position,
                },
            )
            .expect("QSA index-key append");
            let name = format.name();
            for (decode, prefill, what) in [
                (&decode_k, &prefill_k, "keys"),
                (&decode_v, &prefill_v, "values"),
                (&decode_raw, &prefill_raw, "raw index keys"),
            ] {
                assert_eq!(
                    download_bytes(&gpu, decode),
                    download_bytes(&gpu, prefill),
                    "{name} prologue and prefill {what} differ"
                );
            }
            let row_bytes = format.kv_row_bytes(kv_heads, head_dim);
            let tail = |tensor: &GpuTensor| {
                download_bytes(&gpu, tensor)[position * row_bytes..].to_vec()
            };
            let normed = gpu.download_f32(&keys_gpu).expect("normed keys");
            let (k_rows, v_rows) = (tail(&decode_k), tail(&decode_v));
            let dequant_k = dequantize_qsa_rows(format, &k_rows, rows, kv_heads, head_dim);
            let dequant_v = dequantize_qsa_rows(format, &v_rows, rows, kv_heads, head_dim);
            // E4M3 keeps 3 mantissa bits. A wrong head or scale lands near 1.
            let tolerance = 5e-2;
            for (reference, actual, what) in
                [(&normed, &dequant_k, "keys"), (&values, &dequant_v, "values")]
            {
                let rel = rel_l2(reference, actual);
                assert!(rel < tolerance, "{name} {what} rel L2 {rel:.3e}");
            }
            for tensor in [
                index_gpu, qgate_gpu, keys_gpu, values_gpu, index_norm, head_norm, decode_k,
                decode_v, prefill_k, prefill_v, decode_raw, prefill_raw,
            ] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
    }

    /// Every quantized attention kernel must agree bit for bit with its
    /// per-head twin (grouped hg4 batch, single-row) and stay near the F32
    /// reference at the production head shape.
    #[test]
    fn qsa_quantized_attention_kernels_agree() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        let (rows, position_start, full_capacity) = (20usize, 610usize, 640usize);
        let budget_blocks = 150usize;
        let capacity = budget_blocks * compress + compress - 1;
        let kv_width = n_kv_heads * head_dim;
        let q = qsa_lcg(1, rows * n_heads * 2 * head_dim);
        let keys = qsa_lcg(7, full_capacity * kv_width);
        let values = qsa_lcg(13, full_capacity * kv_width);
        let mut selected = vec![-1i32; rows * capacity];
        for row in 0..rows {
            let visible = position_start + row + 1;
            let blocks = visible / compress;
            let chosen = budget_blocks.min(blocks);
            for slot in 0..chosen {
                let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                for r in 0..compress {
                    selected[row * capacity + slot * compress + r] =
                        block * compress as i32 + r as i32;
                }
            }
            let mut offset = chosen * compress;
            for token in blocks * compress..visible {
                selected[row * capacity + offset] = token as i32;
                offset += 1;
            }
            selected[row * capacity + 3] = -1;
        }
        let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
        let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
        let values_gpu = gpu.upload_f32(&values, &[values.len()]).expect("values upload");
        let selected_gpu = gpu
            .zeros(&[selected.len() * 4], DType::Raw)
            .expect("selected allocation");
        let selected_bytes: Vec<u8> = selected.iter().flat_map(|v| v.to_ne_bytes()).collect();
        gpu.hip
            .memcpy_htod(&selected_gpu.buf, &selected_bytes)
            .expect("selected upload");
        let run = |gpu: &mut Gpu,
                   format: QsaKvFormat,
                   k: &GpuTensor,
                   v: &GpuTensor,
                   routes: QsaAttentionRoutes| {
            let output = gpu
                .zeros(&[rows * n_heads * head_dim], DType::F32)
                .expect("output");
            indexed_attention_attention_batch_impl(
                gpu,
                &IndexedAttentionAttentionBatch {
                    q_with_gate: &q_gpu,
                    full_keys: k,
                    full_values: v,
                    selected: &selected_gpu,
                    output: &output,
                    rows,
                    position_start,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    budget_blocks,
                    compress,
                    capacity,
                    full_capacity,
                    format,
                    shape_selected: capacity,
                },
                routes,
            )
            .expect("QSA attention");
            let result = gpu.download_f32(&output).expect("output download");
            gpu.free_tensor(output).expect("free output");
            result
        };
        let reference = run(
            &mut gpu,
            QsaKvFormat::F32,
            &keys_gpu,
            &values_gpu,
            QsaAttentionRoutes::Exact,
        );
        for format in quantized_qsa_formats(&gpu) {
            let name = format.name();
            let row_units = format.kv_row_units(n_kv_heads, head_dim);
            let k = gpu
                .zeros(&[full_capacity * row_units], format.kv_dtype())
                .expect("K");
            let v = gpu
                .zeros(&[full_capacity * row_units], format.kv_dtype())
                .expect("V");
            indexed_attention_cache_append_batch(
                &mut gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &keys_gpu,
                    value: &values_gpu,
                    full_keys: &k,
                    full_values: &v,
                    rows: full_capacity,
                    position_start: 0,
                    kv_heads: n_kv_heads,
                    head_dim,
                    format,
                },
            )
            .expect("QSA append");
            let per_head = run(&mut gpu, format, &k, &v, QsaAttentionRoutes::Exact);
            let grouped = run(&mut gpu, format, &k, &v, QsaAttentionRoutes::Grouped);
            let differing = per_head
                .iter()
                .zip(&grouped)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(differing, 0, "{name} grouped attention differs in {differing} cells");
            let rel = rel_l2(&reference, &per_head);
            let tolerance = 5e-2;
            assert!(rel > 0.0 && rel < tolerance, "{name} attention rel L2 {rel:.3e}");
            // The single-row entry (explicit selection length) is the batched
            // per-head body for that row.
            let row = rows - 1;
            let visible = position_start + row + 1;
            let blocks = visible / compress;
            let selected_len = budget_blocks.min(blocks) * compress + visible - blocks * compress;
            let row_q = q_gpu.sub_offset(row * n_heads * 2 * head_dim, n_heads * 2 * head_dim);
            let row_selected = selected_gpu.sub_offset(row * capacity * 4, capacity * 4);
            let output = gpu.zeros(&[n_heads * head_dim], DType::F32).expect("output");
            indexed_attention_attention(
                &mut gpu,
                &IndexedAttentionAttention {
                    q_with_gate: &row_q,
                    full_keys: &k,
                    full_values: &v,
                    selected: &row_selected,
                    output: &output,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    selected_len,
                    full_capacity: visible,
                    format,
                },
            )
            .expect("QSA single-row attention");
            let single = gpu.download_f32(&output).expect("single download");
            let batched_row = &per_head[row * n_heads * head_dim..];
            assert!(
                single.iter().zip(batched_row).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name} single-row attention differs from the batched row"
            );
            for tensor in [output, k, v] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// BF16 raw/pooled index arenas hold the same BF16 values as F32 arenas,
    /// so pooling and both selection kernels must produce identical pooled
    /// keys and selections.
    #[test]
    fn qsa_bf16_index_arenas_pool_and_select_like_f32() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (compress, index_heads, index_dim) = (4usize, 4usize, 128usize);
        let (rows, position_start, budget_blocks) = (3usize, 1021usize, 64usize);
        let block_count = (position_start + rows) / compress;
        let capacity = budget_blocks * compress + compress - 1;
        // Raw keys are BF16-rounded before they reach the arena.
        let raw: Vec<f32> = qsa_lcg(17, block_count * compress * index_dim)
            .into_iter()
            .map(|v| f32::from_bits((bf16_bits(v) as u32) << 16))
            .collect();
        let raw_bits: Vec<u8> = raw
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        let raw_f32 = gpu.upload_f32(&raw, &[raw.len()]).expect("raw f32");
        let raw_bf16 = gpu.zeros(&[raw.len()], DType::BF16).expect("raw bf16");
        gpu.hip
            .memcpy_htod(&raw_bf16.buf, &raw_bits)
            .expect("raw bf16 upload");
        let norm_bits: Vec<u8> = qsa_lcg(19, index_dim)
            .iter()
            .flat_map(|v| bf16_bits(v * 0.25).to_le_bytes())
            .collect();
        let norm = gpu.zeros(&[index_dim], DType::BF16).expect("norm");
        gpu.hip.memcpy_htod(&norm.buf, &norm_bits).expect("norm upload");
        let pooled_f32 = gpu
            .zeros(&[block_count * index_dim], DType::F32)
            .expect("pooled f32");
        let pooled_bf16 = gpu
            .zeros(&[block_count * index_dim], DType::BF16)
            .expect("pooled bf16");
        for (raw_keys, pooled) in [(&raw_f32, &pooled_f32), (&raw_bf16, &pooled_bf16)] {
            indexed_attention_pool_rope(
                &mut gpu,
                &IndexedAttentionPoolRope {
                    raw_keys,
                    pooled,
                    norm: Some(&norm),
                    block_count,
                    compress,
                    index_dim,
                    position: None,
                    grid_bound: block_count,
                },
            )
            .expect("QSA pool");
        }
        let pooled_reference = gpu.download_f32(&pooled_f32).expect("pooled f32");
        let pooled_widened: Vec<f32> = download_bytes(&gpu, &pooled_bf16)
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect();
        assert!(
            pooled_reference
                .iter()
                .zip(&pooled_widened)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "BF16 pooled keys differ from the F32 arena"
        );
        let query = qsa_lcg(23, rows * index_heads * index_dim);
        let query_gpu = gpu.upload_f32(&query, &[query.len()]).expect("query");
        // The parallel kernel, then the serial one (a shape bound past its LDS).
        for shape_blocks in [block_count, QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES / 4 + 1] {
            let mut selections = Vec::new();
            for pooled in [&pooled_f32, &pooled_bf16] {
                let selected = gpu.zeros(&[rows * capacity * 4], DType::Raw).expect("selected");
                indexed_attention_select_batch(
                    &mut gpu,
                    &IndexedAttentionSelectBatch {
                        query: &query_gpu,
                        pooled,
                        selected: &selected,
                        rows,
                        query_row_stride: index_heads * index_dim,
                        block_count,
                        index_heads,
                        index_dim,
                        budget_blocks,
                        compress,
                        position_start,
                        capacity,
                        shape_blocks,
                    },
                )
                .expect("QSA select");
                selections.push(download_bytes(&gpu, &selected));
                gpu.free_tensor(selected).expect("free selected");
            }
            assert_eq!(
                selections[0], selections[1],
                "BF16 selection differs (shape bound {shape_blocks})"
            );
        }
        for tensor in [raw_f32, raw_bf16, norm, pooled_f32, pooled_bf16, query_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// H4 shared-down fold: the BF16 shared-down GEMM that applies the scaled
    /// add and the paired HC write in its epilogue must leave the HC streams
    /// (and, when asked, the routed rows) bytewise as the F16 WMMA GEMM,
    /// `bf16_scaled_add_batched` and `hyper_write` do, F32 and BF16-bit
    /// streams, ragged row counts.
    #[test]
    fn shared_down_hcsd_is_bytewise_the_gemm_scaled_add_hyper_write_chain() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let (m, k) = (2560usize, 640usize);
        let wide = 4 * m;
        let mut weight = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(1, m * k, 0.3)), &[m * k * 2])
            .expect("weight");
        weight.dtype = DType::BF16;
        weight.shape = vec![m * k];
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for rows in [512usize, 530, 1100] {
            assert!(gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd_applies(&weight, m, k, rows));
            let x = gpu
                .upload_f32(&test_wave(2, rows * k, 2.0), &[rows * k])
                .expect("x");
            let selector = gpu
                .upload_f32(&test_wave(3, rows, 1.0), &[rows])
                .expect("selector");
            let gates = gpu
                .upload_f32(&test_wave(4, rows * 4, 1.5), &[rows * 4])
                .expect("gates");
            let routed_values = test_wave(5, rows * m, 3.0);
            let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
            for state_bf16 in [false, true] {
                let initial = test_wave(6, rows * wide, 3.0);
                let raw: Vec<u8> = if state_bf16 {
                    let mut raw = bf16_le_bytes(&initial);
                    raw.resize(rows * wide * 4, 0);
                    raw
                } else {
                    initial.iter().flat_map(|v| v.to_le_bytes()).collect()
                };
                let mut streams = |gpu: &mut Gpu| {
                    let mut t = gpu.upload_raw(&raw, &[rows * wide * 4]).expect("streams");
                    t.dtype = DType::F32;
                    t.shape = vec![rows * wide];
                    t
                };
                let reference_streams = streams(&mut gpu);
                let reference_routed = gpu
                    .upload_f32(&routed_values, &[rows * m])
                    .expect("routed");
                let projected = gpu.zeros(&[rows * m], DType::F32).expect("projected");
                assert!(gpu
                    .gemm_bf16_xf32_f16_wmma_qwen4(&[(&weight, &projected, m)], &x, k, rows)
                    .expect("gemm"));
                bf16_scaled_add_batched(
                    &mut gpu,
                    &Bf16ScaledAddBatched {
                        residual: &reference_routed,
                        value: &projected,
                        scalar: &selector,
                        rows,
                        elements: m,
                    },
                )
                .expect("scaled add");
                hyper_write(
                    &mut gpu,
                    &HyperWrite {
                        input: &reference_streams,
                        normalized: &normalized,
                        mixed: &reference_routed,
                        gates: &gates,
                        output: &reference_streams,
                        branches: 4,
                        hidden: m,
                        state_bf16,
                    },
                )
                .expect("write");
                for write_routed in [false, true] {
                    let fused_streams = streams(&mut gpu);
                    let fused_routed = gpu
                        .upload_f32(&routed_values, &[rows * m])
                        .expect("routed");
                    gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
                        &weight,
                        &x,
                        m,
                        k,
                        rows,
                        &fused_routed,
                        &selector,
                        &fused_streams,
                        &gates,
                        state_bf16,
                        write_routed,
                    )
                    .expect("hcsd");
                    let (want, got) = (bits(&gpu, &reference_streams), bits(&gpu, &fused_streams));
                    assert_ne!(want.len(), 0);
                    let differing = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing, 0,
                        "{differing} stream words differ: rows={rows} state_bf16={state_bf16} \
                         write_routed={write_routed}"
                    );
                    if write_routed {
                        assert_eq!(
                            bits(&gpu, &reference_routed),
                            bits(&gpu, &fused_routed),
                            "routed rows differ: rows={rows}"
                        );
                    } else {
                        assert_eq!(
                            routed_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            bits(&gpu, &fused_routed),
                            "routed rows were rewritten without hc_write_routed"
                        );
                    }
                }
            }
        }
    }

    /// HC row fold (`HIPFIRE_QWEN4_HC_ROW_FOLD`): the BF16-store shared-down GEMM
    /// plus `hc_row_fold_norm_gate` must leave the HC streams, the next read's F16
    /// normalized row and its paired write's gate logits bytewise as the
    /// zero-initialized combine, the `hcsd` fold + HC write GEMM and
    /// `hyper_norm_gate` with its F16 read output do, with dead (-1) routes, tied experts and
    /// ragged token counts.
    #[test]
    fn hc_row_fold_is_bytewise_combine_hcsd_norm_gate_chain() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let (m, k) = (2560usize, 640usize);
        let wide = 4 * m;
        let ld16 = gpu.f16_row_pitch(wide);
        let bf16_tensor = |gpu: &mut Gpu, values: &[f32]| -> GpuTensor {
            let mut t = gpu
                .upload_raw(&bf16_le_bytes(values), &[values.len() * 2])
                .expect("bf16 upload");
            t.dtype = DType::BF16;
            t.shape = vec![values.len()];
            t
        };
        let weight = bf16_tensor(&mut gpu, &test_wave(1, m * k, 0.3));
        let norm_weight = bf16_tensor(&mut gpu, &test_wave(7, wide, 0.3));
        let gate_weight = bf16_tensor(&mut gpu, &test_wave(8, 4 * wide, 0.05));
        let ints = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for rows in [512usize, 530, 1100] {
            let grouped_rows = rows * 10 + 16;
            let grouped = gpu
                .upload_raw(&bf16_le_bytes(&test_wave(9, grouped_rows * m, 2.0)), &[grouped_rows * m * 2])
                .expect("grouped");
            let inverse: Vec<i32> = (0..rows * 10)
                .map(|i| if i % 13 == 5 { -1 } else { ((i * 7919) % grouped_rows) as i32 })
                .collect();
            let experts: Vec<i32> = (0..rows * 10).map(|i| ((i * 37) % 64) as i32).collect();
            let inverse = gpu.upload_raw(&ints(&inverse), &[rows * 40]).expect("inverse");
            let experts = gpu.upload_raw(&ints(&experts), &[rows * 40]).expect("experts");
            let route_weights: Vec<f32> =
                test_wave(10, rows * 10, 1.0).iter().map(|v| v.abs() / 3.0).collect();
            let route_weights = gpu.upload_f32(&route_weights, &[rows * 10]).expect("weights");
            let order = gpu.zeros(&[rows * 20], DType::F32).expect("order");
            let x = gpu.upload_f32(&test_wave(2, rows * k, 2.0), &[rows * k]).expect("x");
            let selector = gpu.upload_f32(&test_wave(3, rows, 1.0), &[rows]).expect("selector");
            let gates = gpu.upload_f32(&test_wave(4, rows * 4, 1.5), &[rows * 4]).expect("gates");
            let mut raw = bf16_le_bytes(&test_wave(6, rows * wide, 3.0));
            raw.resize(rows * wide * 4, 0);
            let streams = |gpu: &mut Gpu| {
                let mut t = gpu.upload_raw(&raw, &[rows * wide * 4]).expect("streams");
                t.dtype = DType::F32;
                t.shape = vec![rows * wide];
                t
            };
            // Reference: zero-initialized combine, hcsd, hyper_norm_gate.
            let routed = gpu.zeros(&[rows * m], DType::F32).expect("routed");
            gpu.moe_down_combine_grouped_top10_bf16in(
                &grouped, &inverse, &experts, &route_weights, &routed, &order, m, grouped_rows, rows, true,
            )
            .expect("combine");
            let ref_streams = streams(&mut gpu);
            gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
                &weight, &x, m, k, rows, &routed, &selector, &ref_streams, &gates, true, false,
            )
            .expect("hcsd");
            let ref_gates = gpu.zeros(&[rows * 4], DType::F32).expect("ref gates");
            let ref_row = gpu.zeros(&[rows * ld16], DType::F16).expect("ref row");
            hyper_norm_gate(
                &mut gpu,
                &HyperNormGate {
                    input: &ref_streams,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    gates: &ref_gates,
                    rows,
                    branches: 4,
                    hidden: m,
                    state_bf16: true,
                    read_f16: Some((&norm_weight, &ref_row, ld16)),
                },
            )
            .expect("norm gate");
            // Row fold.
            let new_streams = streams(&mut gpu);
            let shared = gpu.zeros(&[rows * m], DType::F32).expect("shared");
            let new_gates = gpu.zeros(&[rows * 4], DType::F32).expect("new gates");
            let new_row = gpu.zeros(&[rows * ld16], DType::F16).expect("new row");
            gpu.moe_combine_order_top10(&inverse, &experts, &route_weights, &order, grouped_rows, rows)
                .expect("order");
            gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(&weight, &x, m, k, rows, &shared)
                .expect("bf16st");
            gpu.hc_row_fold_norm_gate(&crate::hc_row_fold::HcRowFold {
                grouped_down: &grouped,
                order: &order,
                shared_bf16: &shared,
                selector: &selector,
                streams: &new_streams,
                gates: &gates,
                norm_weight: &norm_weight,
                gate_weight: &gate_weight,
                next_gates: &new_gates,
                normalized_f16: &new_row,
                ld16,
                rows,
                hidden: m,
            })
            .expect("row fold");
            let words = |gpu: &Gpu, t: &GpuTensor| gpu.download_raw_bytes(t).expect("download");
            for (name, want, got) in [
                ("streams", words(&gpu, &ref_streams), words(&gpu, &new_streams)),
                ("gate logits", words(&gpu, &ref_gates), words(&gpu, &new_gates)),
                ("normalized f16 row", words(&gpu, &ref_row), words(&gpu, &new_row)),
            ] {
                assert_ne!(want.len(), 0);
                assert!(want == got, "{name} differ: rows={rows}");
            }
        }
    }
}
