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

pub(crate) const TENSOR_OPS_SRC: &str = include_str!("../../../kernels/src/tensor_ops.hip");
const HYPER_READ_UP_WMMA_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_wmma.gfx1151.hip");
const GATED_DELTA_CHUNK_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_chunk_wmma.gfx1151.hip");
const INDEXED_ATTENTION_DENSE_WMMA_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_dense_wmma.gfx1151.hip");
const QSA_SELECT_PARALLEL_THREADS: u32 = 256;
// gfx1151's 64-KiB dynamic LDS budget; other devices use the serial path.
// Oversized rows also use serial kernels without changing the contract.
const QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;
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

pub struct GatedDeltaStep<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub fn gated_delta_step(gpu: &mut Gpu, p: &GatedDeltaStep<'_>) -> HipResult<()> {
    for tensor in [p.q, p.k, p.v, p.gate, p.beta, p.state, p.output] {
        ensure_f32(tensor)?;
    }
    if p.key_heads == 0
        || p.value_heads == 0
        || p.value_heads % p.key_heads != 0
        || p.key_dim == 0
        || p.value_dim == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let key_elements = checked_product(p.key_heads, p.key_dim, "GDN key extent")?;
    let value_elements = checked_product(p.value_heads, p.value_dim, "GDN value extent")?;
    let state_elements = checked_product(value_elements, p.key_dim, "GDN state extent")?;
    if p.q.numel() != key_elements
        || p.k.numel() != key_elements
        || p.v.numel() != value_elements
        || p.gate.numel() != p.value_heads
        || p.beta.numel() != p.value_heads
        || p.state.numel() != state_elements
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
    let shared_norm128 =
        gpu.arch_caps.qwen4_tuned_routes() && p.key_dim == 128 && p.value_dim == 128;
    let kernel = if shared_norm128 {
        "gated_delta_step_shared_norm128_gfx1151"
    } else {
        "gated_delta_step_f32"
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
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_i32(key_dim);
    args.push_i32(value_dim);
    args.push_f32((p.key_dim as f32).sqrt().recip());
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [value_heads_grid, value_dim_grid, 1],
        [block_x, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
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
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub qkv_width: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub fn gated_delta_step_batched(gpu: &mut Gpu, p: &GatedDeltaStepBatched<'_>) -> HipResult<()> {
    for tensor in [p.projection, p.gate, p.beta, p.state, p.output] {
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
    let state_elements = checked_product(value, p.key_dim, "GDN batched state extent")?;
    if p.qkv_width != expected_qkv
        || p.projection.numel() != rows_qkv
        || p.gate.numel() != rows_heads
        || p.beta.numel() != rows_heads
        || p.state.numel() != state_elements
        || p.output.numel() != rows_value
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
    let kernel = "gated_delta_step_halves_state128_persistent256_f32";
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.projection, p.gate, p.beta, p.state, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(qkv_width);
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_i32(key_dim);
    args.push_i32(value_dim);
    args.push_f32((p.key_dim as f32).sqrt().recip());
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
/// Whether [`gated_delta_step_gate_wmma`] applies: the Qwen4 F16 prefill
/// route (Qwen4-tuned arch, >= QWEN4_F16_WMMA_MIN_TOKENS rows, no recorder or capture,
/// not opted out) with 128-wide heads.  Decided before the convolution, which
/// then stores its output as packed BF16 for that kernel.
pub fn gated_delta_chunk_route(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.arch_caps.qwen4_tuned_routes()
        && p.rows >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.key_dim == 128
        && p.value_dim == 128
        && p.key_heads > 0
        && p.value_heads % p.key_heads == 0
}

/// [`gated_delta_step_batched`] followed by [`gated_delta_gate_batched`] as
/// one chunked (16-row WY form) F16 WMMA kernel where
/// [`gated_delta_chunk_route`] holds; `p.projection` is the convolution
/// output as packed BF16.  `gate.output` receives the gated rows and
/// `p.output` is not written.  The recurrence's products round to F16, so
/// the route is KLD-gated; the gate is that kernel's expression.
pub fn gated_delta_step_gate_wmma(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
) -> HipResult<()> {
    if !gated_delta_chunk_route(gpu, p) || p.projection.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    for tensor in [p.gate, p.beta, p.state, gate.z] {
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
        || p.state.numel() != value * p.key_dim
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
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "gated_delta_qk_norm_bf16_batched",
    )?;
    gpu.ensure_kernel_public(
        "gated_delta_chunk_wmma",
        GATED_DELTA_CHUNK_WMMA_SRC,
        "gated_delta_chunk_gate_wmma",
    )?;
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
    let mut args = KernargBlob::new();
    args.push_ptr(p.projection.buf.as_ptr());
    args.push_ptr(qp);
    args.push_ptr(kp);
    for tensor in [p.gate, p.beta, p.state, gate.z, gate.norm, gate.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(qkv_width);
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_f32((p.key_dim as f32).sqrt().recip());
    args.push_i32(i32::from(output_bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_chunk_gate_wmma",
        [checked_u32(p.value_heads, "GDN chunk head grid")?, 1, 1],
        [256, 1, 1],
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
        // Two columns (one BF16 pair) per thread.
        if p.hidden % 2 != 0 {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_write_bf16x2")?;
        let mut args = KernargBlob::new();
        for tensor in [p.input, p.mixed, p.gates, p.output] {
            args.push_ptr(tensor.buf.as_ptr());
        }
        args.push_i32(branches);
        args.push_i32(hidden);
        args.push_i32(rows_i);
        args.pad_to(16);
        return gpu.launch_blob_recorded(
            "hyper_write_bf16x2",
            [blocks(wide / 2)?, grid_y, 1],
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
    hyper_norm_impl(gpu, p, std::ptr::null_mut(), false)
}

/// [`hyper_norm`] that writes the normalized rows as F16 into
/// `normalized_f16` (same element count), the F16 WMMA projections' input,
/// and with `bf16_copy` also stores `normalized` as BF16 bits (the values are
/// BF16-rounded) in the first half of its buffer: read it with
/// `HyperReadUpFused::normalized_bf16`.  Without it `normalized` is untouched.
pub fn hyper_norm_f16(
    gpu: &mut Gpu,
    p: &HyperNorm<'_>,
    normalized_f16: &GpuTensor,
    bf16_copy: bool,
) -> HipResult<()> {
    if normalized_f16.dtype != DType::F16 || normalized_f16.numel() != p.normalized.numel() {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    hyper_norm_impl(gpu, p, normalized_f16.buf.as_ptr(), bf16_copy)
}

fn hyper_norm_impl(
    gpu: &mut Gpu,
    p: &HyperNorm<'_>,
    normalized_f16: *mut std::ffi::c_void,
    bf16_copy: bool,
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
/// [`hyper_read_up_fused`] on gfx11 BF16 WMMA; `low` is packed BF16
/// ([`HcActivationFused::bf16_out`]) and `normalized` is
/// [`hyper_norm_f16`]'s F16 copy (`normalized_bf16` is ignored). Not
/// bit-exact: the logits accumulate the same exact BF16 products in WMMA's F32
/// order, so a gate occasionally rounds one BF16 step apart; the epilogue is
/// unchanged.
pub fn hyper_read_up_wmma(gpu: &mut Gpu, p: &HyperReadUpFused<'_>) -> HipResult<()> {
    ensure_f32(p.mixed)?;
    let wide = checked_product(4, p.hidden, "HC read width")?;
    if !gpu.arch_caps.qwen4_tuned_routes()
        || p.low.dtype != DType::BF16
        || p.normalized.dtype != DType::F16
        || p.up_weight.dtype != DType::BF16
        || p.rows == 0
        || p.hidden % 16 != 0
        || p.low_rank % 16 != 0
        || p.low_rank > 504
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
    let column_grid = checked_u32(p.hidden / 16, "HC read column grid")?;
    let row_grid = checked_u32(p.rows.div_ceil(512), "HC read row grid")?;
    let lds_bytes = checked_u32(64 * (p.low_rank + 8) * 2, "HC read LDS")?;
    gpu.ensure_kernel_public(
        "hyper_read_up_wmma",
        HYPER_READ_UP_WMMA_SRC,
        "hyper_read_up_wmma_bf16",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.up_weight.buf.as_ptr());
    args.push_ptr(p.low.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.mixed.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(low_rank);
    args.push_i32(rows);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_read_up_wmma_bf16",
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
    let grid = blocks(p.channels)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_conv_bf16_f32")?;
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
        "gated_delta_conv_bf16_f32",
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
    let grid = blocks(p.channels)?;
    let row_grid = checked_u32(p.rows.div_ceil(16), "GDN batched convolution row grid")?;
    let kernel = "gated_delta_conv_bf16_f32_batched_k4";
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(start_cursor);
    let start_cursor_offset = args.len() - 4;
    args.push_i32(i32::from(output_bf16));
    args.push_i32(i32::from(input_bf16));
    args.pad_to(16);
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
        [grid, row_grid, 1],
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
    let heads_i = checked_i32(heads, "GDN parameter head count")?;
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

pub fn gated_delta_params_batched(gpu: &mut Gpu, p: &GatedDeltaParamsBatched<'_>) -> HipResult<()> {
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
    let value_heads = checked_i32(p.value_heads, "GDN gate head count")?;
    let value_dim = checked_i32(p.value_dim, "GDN gate width")?;
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

pub fn gated_delta_gate_batched(gpu: &mut Gpu, p: &GatedDeltaGateBatched<'_>) -> HipResult<()> {
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

pub struct IndexedAttentionCacheAppendBatch<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub rows: usize,
    pub position_start: usize,
    pub kv_width: usize,
}

pub fn indexed_attention_cache_append_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppendBatch<'_>,
) -> HipResult<()> {
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        ensure_f32(tensor)?;
    }
    if p.rows == 0
        || p.kv_width == 0
        || p.key.numel() < checked_product(p.rows, p.kv_width, "indexed attention batch key")?
        || p.value.numel() < checked_product(p.rows, p.kv_width, "indexed attention batch value")?
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let end = checked_product(end_position, p.kv_width, "indexed attention batch cache")?;
    if end > p.full_keys.numel() || end_position > i32::MAX as usize {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "indexed attention batch rows")?;
    let position_start = checked_i32(p.position_start, "indexed attention batch position")?;
    let kv_width = checked_i32(p.kv_width, "indexed attention batch width")?;
    let grid = blocks(p.kv_width)?;
    let row_grid = checked_u32(p.rows, "indexed attention batch row grid")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_cache_append_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it,
    // so the cache row this launch writes follows the replay position.
    let position_offset = args.len() - 4;
    args.push_i32(kv_width);
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    gpu.launch_blob_recorded(
        "indexed_attention_cache_append_f32_batched",
        [grid, row_grid, 1],
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
    let (kernel_name, block, shared_mem) =
        match p.block_count.checked_mul(std::mem::size_of::<f32>()) {
            Some(bytes)
                if gpu.arch_caps.qwen4_tuned_routes()
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
    for tensor in [p.query, p.pooled] {
        ensure_f32(tensor)?;
    }
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
    let (kernel_name, block, shared_mem) =
        match shape_blocks.checked_mul(std::mem::size_of::<f32>()) {
            Some(bytes)
                if gpu.arch_caps.qwen4_tuned_routes()
                    && shape_blocks > 0
                    && bytes <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (
                    "indexed_attention_select_f32_batched",
                    [QSA_SELECT_PARALLEL_THREADS, 1, 1],
                    bytes as u32,
                )
            }
            _ => ("indexed_attention_select_f32_batched_serial", [1, 1, 1], 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
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
        kernel_name,
        [row_grid, 1, 1],
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &bindings,
        },
    )
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
    gpu.launch_blob_recorded(
        "indexed_attention_reuse_selection",
        [1, 1, 1],
        [1, 1, 1],
        0,
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
    for tensor in [p.raw_keys, p.pooled] {
        ensure_f32(tensor)?;
    }
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
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_pool_rope_f32",
    )?;
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
    args.pad_to(16);
    let block_count_binding = match p.position {
        None => None,
        Some(position) => {
            let addend = u32::try_from(position.rows)
                .map_err(|_| HipError::new(0, "QSA pool/RoPE rows exceed u32"))?;
            let divisor = u32::try_from(p.compress)
                .map_err(|_| HipError::new(0, "QSA pool/RoPE compression exceeds u32"))?;
            Some([crate::replay::ReplayKernargBinding::PositionDivU32 {
                offset: block_count_offset,
                addend,
                divisor,
            }])
        }
    };
    let kernargs: &[crate::replay::ReplayKernargBinding] = match block_count_binding.as_ref() {
        None => &[],
        Some(bindings) => &bindings[..],
    };
    gpu.launch_blob_recorded(
        "indexed_attention_pool_rope_f32",
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
}

pub fn indexed_attention_attention(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttention<'_>,
) -> HipResult<()> {
    for tensor in [p.q_with_gate, p.full_keys, p.full_values, p.output] {
        ensure_f32(tensor)?;
    }
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
    let full_elements = checked_product3(
        p.full_capacity,
        p.n_kv_heads,
        p.head_dim,
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
    let (kernel_name, shared_mem) =
        match p.selected_len.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if gpu.arch_caps.qwen4_tuned_routes()
                    && bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                ("indexed_attention_attention_f32", bytes as u32)
            }
            _ => ("indexed_attention_attention_f32_serial", 0),
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

/// Below this many rows the grouped kernel launches too few workgroups
/// (`rows * n_heads / 4`) to fill the GPU and the per-head kernel is faster
/// (decode: 0.48 vs 0.73 ms per call at 1131 context on gfx1151).
const QSA_ATTENTION_HG4_MIN_ROWS: usize = 16;

/// Dynamic LDS for the grouped kernel, or `None` when the shape is not the one
/// it is specialised for (Qwen4-tuned arch, head_dim 256, four heads per KV group,
/// prefill-sized row counts).
fn qsa_attention_hg4_lds_bytes(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    shape_selected: usize,
) -> Option<u32> {
    let group = p.n_heads / p.n_kv_heads;
    if !gpu.arch_caps.qwen4_tuned_routes()
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
    indexed_attention_attention_batch_impl(gpu, p, true)
}

/// `allow_fast` admits the grouped hg4 kernel and the dense F16 WMMA route;
/// without it the per-head kernels run (the exact reference).
fn indexed_attention_attention_batch_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    allow_fast: bool,
) -> HipResult<()> {
    for tensor in [p.q_with_gate, p.full_keys, p.full_values, p.output] {
        ensure_f32(tensor)?;
    }
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
    let kv_elements = checked_product3(
        p.n_kv_heads,
        p.head_dim,
        p.full_capacity,
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
    if allow_fast && qsa_dense_wmma_applies(gpu, p, end_position) {
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
    let hg4_bytes = allow_fast
        .then(|| qsa_attention_hg4_lds_bytes(gpu, p, shape_selected))
        .flatten();
    let (kernel_name, grid, shared_mem) = if let Some(bytes) = hg4_bytes {
        (
            "indexed_attention_attention_f32_batched_hg4",
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
                if gpu.arch_caps.qwen4_tuned_routes()
                    && shape_selected > 0
                    && bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (
                    "indexed_attention_attention_f32_batched",
                    [head_grid, dim_grid, row_grid],
                    bytes as u32,
                )
            }
            _ => (
                "indexed_attention_attention_f32_batched_serial",
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

/// Whether the dense F16 WMMA attention applies: the Qwen4 F16 route (Qwen4-tuned arch,
/// >= QWEN4_F16_WMMA_MIN_TOKENS rows, no recorder or capture, not opted out),
/// head_dim 256 in four-head KV groups, and every row's selection is its
/// whole causal window: the budget covers every visible block and the
/// capacity every visible token (indexed_attention_select then emits all of
/// them).
fn qsa_dense_wmma_applies(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
) -> bool {
    gpu.arch_caps.qwen4_tuned_routes()
        && p.rows >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.head_dim == 256
        && (p.n_heads / p.n_kv_heads) % 4 == 0
        && end_position / p.compress <= p.budget_blocks
        && end_position <= p.capacity
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
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
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
            for tensor in [input, hist_a, out_a, hist_b, out_b] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        gpu.free_tensor(kernel).expect("free");
    }

    /// The fused HC norm + BF16 gate projection must equal hyper_norm followed
    /// by the multi-row BF16 GEMM bit for bit, at the production width.
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
            },
        )
        .expect("fused");
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
        if !gpu.arch_caps.qwen4_tuned_routes() {
            eprintln!("skip: Qwen4-tuned routes are off on this GPU");
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
        if gpu.arch_caps.qwen4_tuned_routes() {
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
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
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
        if !gpu.arch_caps.qwen4_tuned_routes() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: Qwen4-tuned routes are off on this GPU");
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
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
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

    /// The grouped QSA attention kernel must equal the per-head batched kernel
    /// bit for bit at the production shape (24 heads, 2 KV heads, head_dim
    /// 256): permuted selections, invalid slots, a partial key tile and
    /// rows with and without a tail all included.
    #[test]
    fn qsa_attention_hg4_is_bit_identical_to_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.qwen4_tuned_routes() {
            eprintln!("skip: Qwen4-tuned routes are off on this GPU");
            return;
        }
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        // 600+ visible tokens with a 150-block budget: selections longer than
        // one 256-row score pass, plus the causal tail.
        let (rows, position_start, full_capacity) = (20usize, 610usize, 640usize);
        let budget_blocks = 150usize;
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
        let run = |gpu: &mut Gpu, allow_fast: bool| {
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
                    shape_selected: capacity,
                },
                allow_fast,
            )
            .expect("QSA attention");
            let values = gpu.download_f32(&output).expect("output download");
            gpu.free_tensor(output).expect("free output");
            values
        };
        let reference = run(&mut gpu, false);
        let grouped = run(&mut gpu, true);
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

    /// The dense F16 WMMA route (every row selects its whole causal window)
    /// must match the exact per-head kernel to F16-rounding accuracy at the
    /// production head shape, with a chunk offset and a partial row tile.
    #[test]
    fn qsa_dense_wmma_matches_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.qwen4_tuned_routes() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: Qwen4-tuned routes are off on this GPU");
            return;
        }
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        let (rows, position_start, full_capacity) = (521usize, 100usize, 640usize);
        let (budget_blocks, capacity) = (2048usize, 640usize);
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
        // The whole window, blocks then tail, as indexed_attention_select
        // emits it when the budget covers every block.
        let mut selected = vec![-1i32; rows * capacity];
        for row in 0..rows {
            for token in 0..position_start + row + 1 {
                selected[row * capacity + token] = token as i32;
            }
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
        let run = |gpu: &mut Gpu, allow_fast: bool| {
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
                    shape_selected: capacity,
                },
                allow_fast,
            )
            .expect("QSA attention");
            let values = gpu.download_f32(&output).expect("output download");
            gpu.free_tensor(output).expect("free output");
            values
        };
        let reference = run(&mut gpu, false);
        let dense = run(&mut gpu, true);
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for (r, d) in reference.iter().zip(&dense) {
            err += (*r as f64 - *d as f64).powi(2);
            norm += (*r as f64).powi(2);
        }
        let rel = (err / norm).sqrt();
        assert!(norm > 0.0, "reference output is all zero");
        // F16 operands and probabilities: ~1e-3 relative; a wrong row, head,
        // key range or dim mapping lands near 1.
        assert!(rel < 5e-3, "dense QSA WMMA rel L2 {rel:.3e}");
        // It is the WMMA route (F16 rounding), not the exact kernel.
        assert!(rel > 0.0, "dense QSA WMMA route did not run");
        for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
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
        const SERIAL_BOUND: usize = QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES / 4 + 1;

        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 65_536.0) - 8.0
        };

        for &compress in &[2usize, 4, 8] {
            for &index_dim in &[8usize, 128] {
                for &index_heads in &[1usize, 4] {
                    for &rows in &[1usize, 5] {
                        // `(position_start + rows) / compress` must equal the
                        // wrapper's declared block count.
                        if rows > compress - 1 {
                            continue;
                        }
                        for &block_count in &[0usize, 1, 3, 17, 72, 128, 500] {
                            for &budget_blocks in &[1usize, 4, 64, 512] {
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
                                let pooled: Vec<f32> = (0..block_count * index_dim + index_dim)
                                    .map(|_| next())
                                    .collect();
                                let query: Vec<f32> = (0..rows * index_heads * index_dim)
                                    .map(|_| next())
                                    .collect();
                                let parallel =
                                    run_select_case(&mut gpu, &case, &pooled, &query, block_count);
                                let serial =
                                    run_select_case(&mut gpu, &case, &pooled, &query, SERIAL_BOUND);
                                assert_eq!(
                                    parallel, serial,
                                    "parallel ranking diverged from the serial selection sort: \
                                     compress={compress} heads={index_heads} dim={index_dim} \
                                     rows={rows} blocks={block_count} budget={budget_blocks}"
                                );
                            }
                        }
                    }
                }
            }
        }
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
}
