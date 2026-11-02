// SPDX-License-Identifier: Apache-2.0
//! Direct BF16 row-major trunk consumers. Incumbent launchers remain independent.
use super::*;
use sha2::{Digest, Sha256};
pub(crate) const ARENA_SRC: &str = concat!(
    "#define NPU_TRUNK_ARENA 1\n#define gated_delta_conv_bf16_f32_batched_k4 gated_delta_conv_batched_arena\n#define gated_delta_conv_qknorm_bf16_f32_batched_k4 gated_delta_conv_qknorm_batched_arena\n",
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip"),
    include_str!("../../../kernels/src/npu_trunk_consumers.hip")
);
pub(crate) const CHUNK_ARENA_SRC: &str = concat!("#define NPU_TRUNK_ARENA 1\n#define gated_delta_chunk_gate_wmma gated_delta_chunk_gate_wmma_arena\n",
    include_str!("../../../kernels/src/gated_delta_chunk_wmma.gfx1151.hip"));
pub(crate) const CHUNK_Q8_ARENA_SRC: &str = concat!("#define NPU_TRUNK_ARENA 1\n#define gated_delta_chunk_gate_q8_wmma gated_delta_chunk_gate_q8_wmma_arena\n",
    include_str!("../../../kernels/src/gated_delta_chunk_q8_wmma.gfx1151.hip"));

/// Immutable projection view in a fused BF16 arena; all offsets count elements.
#[derive(Clone, Copy)]
pub struct Bf16Arena<'a> {
    pub tensor: &'a GpuTensor,
    pub row_pitch_elems: usize,
    pub col0_elems: usize,
}
impl Bf16Arena<'_> {
    fn validate(self, rows: usize, cols: usize) -> HipResult<()> {
        if self.tensor.dtype != DType::BF16 {
            return Err(HipError::new(0, "arena input must be BF16"));
        }
        let span = checked_add(self.col0_elems, cols, "arena columns")?;
        if rows == 0 || cols == 0 || span > self.row_pitch_elems {
            return Err(HipError::new(0, "invalid arena row/column range"));
        }
        let end = checked_add(
            checked_product(rows - 1, self.row_pitch_elems, "arena rows")?,
            span,
            "arena extent",
        )?;
        checked_i32(end, "arena element offset")?;
        if end > self.tensor.numel()
            || checked_product(end, 2, "arena bytes")? > self.tensor.buf.size()
        {
            return Err(HipError::new(0, "undersized BF16 arena"));
        }
        Ok(())
    }
    fn args(self, args: &mut KernargBlob) -> HipResult<()> {
        args.push_i32(checked_i32(self.row_pitch_elems, "arena pitch")?);
        args.push_i32(checked_i32(self.col0_elems, "arena column")?);
        Ok(())
    }
}
fn arena_launch(
    gpu: &mut Gpu,
    name: &str,
    grid: [u32; 3],
    threads: u32,
    args: &mut KernargBlob,
    bindings: crate::dispatch::ReplayLaunchBindings<'_>,
) -> HipResult<()> {
    gpu.ensure_kernel_public("npu_trunk_consumers", ARENA_SRC, name)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        name,
        grid,
        [threads, 1, 1],
        0,
        args.as_mut_slice(),
        bindings,
    )
}

pub fn gated_delta_conv_batched_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaConvBatched<'_>,
    src: Bf16Arena<'_>,
) -> HipResult<()> {
    conv_arena(gpu, p, None, None, Some(src))
}
/// Fused-convolution API with direct arena parameters; no materialization.
pub fn gated_delta_conv_params_batched_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    src: Bf16Arena<'_>,
    a: Bf16Arena<'_>,
    b: Bf16Arena<'_>,
) -> HipResult<()> {
    conv_arena(gpu, p, Some((params, Some(a), Some(b))), None, Some(src))
}
pub fn gated_delta_conv_params_qknorm_batched_arena(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    step: &GatedDeltaStepBatched<'_>,
    keep_qk_conv: bool,
    src: Bf16Arena<'_>,
    a: Bf16Arena<'_>,
    b: Bf16Arena<'_>,
) -> HipResult<()> {
    if !gated_delta_conv_qknorm_geometry(gpu, step, conv) {
        return Err(HipError::new(0, "invalid arena qknorm geometry"));
    }
    let qk = checked_product(step.key_heads, step.key_dim, "arena qknorm")?;
    let scratch = gpu.qwen4_f16_x_scratch(2 * step.rows * qk)?;
    let qn = scratch.buf.as_ptr();
    let kn = unsafe { (qn as *mut u8).add(step.rows * qk * 2) } as *mut std::ffi::c_void;
    conv_arena(
        gpu,
        conv,
        Some((params, Some(a), Some(b))),
        Some(ConvQkNorm {
            qn,
            kn,
            keep_qk_conv,
        }),
        Some(src),
    )
}

/// Each absent source retains its native projection dtype and layout.
pub fn gated_delta_conv_params_batched_mixed(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    src: Option<Bf16Arena<'_>>,
    a: Option<Bf16Arena<'_>>,
    b: Option<Bf16Arena<'_>>,
) -> HipResult<()> {
    conv_arena(gpu, conv, Some((params, a, b)), None, src)
}
pub fn gated_delta_conv_params_qknorm_batched_mixed(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    step: &GatedDeltaStepBatched<'_>,
    keep_qk_conv: bool,
    src: Option<Bf16Arena<'_>>,
    a: Option<Bf16Arena<'_>>,
    b: Option<Bf16Arena<'_>>,
) -> HipResult<()> {
    if !gated_delta_conv_qknorm_geometry(gpu, step, conv) {
        return Err(HipError::new(0, "invalid arena qknorm geometry"));
    }
    let qk = checked_product(step.key_heads, step.key_dim, "arena qknorm")?;
    let scratch = gpu.qwen4_f16_x_scratch(2 * step.rows * qk)?;
    let qn = scratch.buf.as_ptr();
    let kn = unsafe { (qn as *mut u8).add(step.rows * qk * 2) } as *mut std::ffi::c_void;
    conv_arena(
        gpu,
        conv,
        Some((params, a, b)),
        Some(ConvQkNorm {
            qn,
            kn,
            keep_qk_conv,
        }),
        src,
    )
}

fn mixed_source<'a>(
    src: Option<Bf16Arena<'a>>,
    native: &'a GpuTensor,
    rows: usize,
    cols: usize,
) -> HipResult<&'a GpuTensor> {
    if let Some(src) = src {
        src.validate(rows, cols)?;
        Ok(src.tensor)
    } else {
        if native.dtype != DType::F32 && native.dtype != DType::BF16 {
            return Err(HipError::new(0, "mixed source needs native F32/BF16"));
        }
        if native.numel() < rows * cols {
            return Err(HipError::new(0, "undersized native source"));
        }
        Ok(native)
    }
}
fn mixed_args(src: Option<Bf16Arena<'_>>, args: &mut KernargBlob) -> HipResult<()> {
    if let Some(src) = src {
        src.args(args)
    } else {
        args.push_i32(0);
        args.push_i32(0);
        Ok(())
    }
}

fn conv_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaConvBatched<'_>,
    params: Option<(
        &GatedDeltaParamsBatched<'_>,
        Option<Bf16Arena<'_>>,
        Option<Bf16Arena<'_>>,
    )>,
    qk_norm: Option<ConvQkNorm>,
    src: Option<Bf16Arena<'_>>,
) -> HipResult<()> {
    if let Some((q, a, b)) = params {
        mixed_source(a, q.a, q.rows, q.heads)?;
        mixed_source(b, q.b, q.rows, q.heads)?;
        if a.is_none() {
            ensure_f32(q.a)?;
        }
        if b.is_none() {
            ensure_f32(q.b)?;
        }
        for t in [q.gate, q.beta] {
            ensure_f32(t)?;
            if t.numel() != q.rows * q.heads {
                return Err(HipError::new(0, "parameter output shape"));
            }
        }
        for t in [q.a_log, q.dt_bias] {
            if t.dtype != DType::BF16 || t.numel() != q.heads {
                return Err(HipError::new(0, "parameter weight shape"));
            }
        }
    }
    for tensor in [p.history, p.next_history] {
        ensure_f32(tensor)?;
    }
    // Input and output are each F32 or BF16 bits (BF16-rounded values).
    let output_bf16 = p.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(p.output)?;
    }
    let input = mixed_source(src, p.input, p.rows, p.channels)?;
    let input_bf16 = src.is_some() || p.input.dtype == DType::BF16;
    if qk_norm.is_some()
        && src.is_some_and(|s| {
            s.row_pitch_elems % 4 != 0
                || s.col0_elems % 4 != 0
                || s.tensor.buf.as_ptr() as usize % 8 != 0
        })
    {
        return Err(HipError::new(
            0,
            "arena qknorm input must be 8-byte aligned",
        ));
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
    if p.output.numel() != input_elements
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
    let kernel = if qk_norm.is_some() {
        "gated_delta_conv_qknorm_batched_arena"
    } else {
        "gated_delta_conv_batched_arena"
    };
    gpu.ensure_kernel_public("npu_trunk_consumers", ARENA_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(start_cursor);
    let start_cursor_offset = args.len() - 4;
    // H5's `output_bf16` is a flag word: bit 0 = packed BF16 output, bit 1 =
    // also store the unnormalized Q/K convolution (debug / capture).
    args.push_i32(i32::from(output_bf16) | qk_norm.map_or(0, |n| 2 * i32::from(n.keep_qk_conv)));
    args.push_i32(i32::from(input_bf16));
    let (param_elements, param_heads) = match params {
        Some((q, a, b)) => {
            for tensor in [
                a.map_or(q.a, |s| s.tensor),
                b.map_or(q.b, |s| s.tensor),
                q.a_log,
                q.dt_bias,
                q.gate,
                q.beta,
            ] {
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
    mixed_args(src, &mut args)?;
    if let Some((_, a, b)) = params {
        mixed_args(a, &mut args)?;
        mixed_args(b, &mut args)?;
    } else {
        for _ in 0..4 {
            args.push_i32(0);
        }
    }
    args.pad_to(16);
    let param_grid = checked_u32(param_elements.div_ceil(256), "GDN batched parameter grid")?;
    // Compact 1-D grid: `conv_blocks * row_grid` convolution tiles (block `i`
    // is channel block `i % conv_blocks`, row block `i / conv_blocks`), then the
    // parameter blocks.  The kernel re-derives the split from rows / channels,
    // so the grid depends only on those and the heads, fixed at capture.
    let conv_tiles = checked_product(
        grid as usize,
        row_grid as usize,
        "GDN batched convolution tile grid",
    )?;
    let total_grid = checked_u32(
        checked_add(
            conv_tiles,
            param_grid as usize,
            "GDN batched convolution launch grid",
        )?,
        "GDN batched convolution launch grid",
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
        [total_grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &start_cursor_binding,
        },
    )
}

fn chunk_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
    arms: GatedDeltaChunkArms,
    src: Bf16Arena<'_>,
) -> HipResult<()> {
    if !gated_delta_chunk_route(gpu, p)
        || p.projection.dtype != DType::BF16
        || (arms.q8_inline && !gated_delta_q8_inline_geometry(gpu, p))
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let format = p.format;
    for tensor in [p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    // The gated output is BF16-rounded: stored as BF16 bits or F32.
    let output_bf16 = gate.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(gate.output)?;
    }
    let qk = checked_product(p.key_heads, p.key_dim, "GDN chunk qk extent")?;
    let value = checked_product(p.value_heads, p.value_dim, "GDN chunk value extent")?;
    src.validate(p.rows, value)?;
    if !gpu.arch_caps.is_gfx1151() {
        return Err(HipError::new(0, "arena chunk requires gfx1151"));
    }
    if p.qkv_width != 2 * qk + value
        || p.projection.numel() < p.rows * p.qkv_width
        || p.gate.numel() < p.rows * p.value_heads
        || p.beta.numel() < p.rows * p.value_heads
        || validate_gdn_state(
            format,
            p.state,
            p.residual,
            None,
            p.rows,
            p.value_heads,
            p.key_dim,
            p.value_dim,
        )
        .is_err()
        || gate.rows != p.rows
        || gate.value_heads != p.value_heads
        || gate.value_dim != p.value_dim
        || gate.norm.dtype != DType::BF16
        || gate.norm.numel() != p.value_dim
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
    // The kernel, module and source follow the state format and the inline/bracket arm; the
    // plain Q8 inline module is its own registry entry over its own arena source.
    let chunk_kernel = gdn_chunk_kernel_name(format, arms.q8_inline, true);
    let chunk_module = gdn_chunk_module_name(format, arms.q8_inline, true);
    gpu.ensure_kernel_public(
        chunk_module,
        gdn_chunk_source(format, arms.q8_inline, true),
        chunk_kernel,
    )?;
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
    // A Q8/Q8Ef state not on the inline arm runs the chunk on its F32 dequantization (the
    // state scratch) and is requantized from it after the chunk: Q8 seeded by the last row,
    // Q8Ef (no frame, nearest-even) rebuilding `code * scale + residual` once on entry and
    // writing fresh codes/scales/residual on exit.  Only the plain Q8 inline arm
    // (`gated_delta_chunk_gate_q8_wmma`) does both inside the kernel on the slot itself.  Q8
    // seeds with the last row's absolute position, `position + rows - 1`.  The route excludes
    // a recorder and graph capture (`gated_delta_chunk_route`), so the frame is never a
    // replayed kernarg and needs no mutable-frame binding.
    let frame = match format {
        GdnStateFormat::Q8 => checked_u32(p.position + p.rows - 1, "GDN Q8 frame")?,
        GdnStateFormat::F32 | GdnStateFormat::Q8Ef => 0,
    };
    let slot = p.state.buf.as_ptr();
    let residual = p.residual.map(|residual| residual.buf.as_ptr());
    // A Q8Ef inline request never reaches here: `arms.q8_inline &&
    // !gated_delta_q8_inline_geometry` above already refused it (Q8Ef has no
    // inline kernel), so a Q8Ef chunk always takes the scratch arm.
    let (chunk_state, bracket) = match (format, arms.q8_inline) {
        (GdnStateFormat::F32, _) => (GdnChunkState::F32 { state: slot }, None),
        (GdnStateFormat::Q8, true) => (GdnChunkState::Q8Inline { state: slot, frame }, None),
        (GdnStateFormat::Q8, false) | (GdnStateFormat::Q8Ef, _) => {
            let scratch = gpu.gdn_state_f32_scratch(GdnStateFormat::F32.state_bytes(
                p.value_heads,
                p.key_dim,
                p.value_dim,
            ))?;
            gdn_state_to_f32(gpu, format, slot, residual, scratch, p.value_heads)?;
            (GdnChunkState::F32 { state: scratch }, Some(scratch))
        }
    };
    let mut args = gdn_chunk_kernargs(&GdnChunkKernargs {
        projection: p.projection.buf.as_ptr(),
        qn: qp,
        kn: kp,
        gate: p.gate.buf.as_ptr(),
        beta: p.beta.buf.as_ptr(),
        state: chunk_state,
        z: src.tensor.buf.as_ptr(),
        norm: gate.norm.buf.as_ptr(),
        output: gate.output.buf.as_ptr(),
        rows,
        qkv_width,
        key_heads,
        value_heads,
        query_scale: (p.key_dim as f32).sqrt().recip(),
        output_bf16: i32::from(output_bf16),
    });
    src.args(&mut args)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        chunk_kernel,
        [checked_u32(p.value_heads, "GDN chunk head grid")?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some(scratch) = bracket {
        gdn_state_from_f32(gpu, format, scratch, slot, residual, p.value_heads, frame)?;
    }
    Ok(())
}

pub fn gated_delta_step_gate_wmma_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
    z: Bf16Arena<'_>,
) -> HipResult<()> {
    chunk_arena(
        gpu,
        p,
        gate,
        GatedDeltaChunkArms {
            qk_prenormed: false,
            q8_inline: gated_delta_q8_inline_route(gpu, p),
        },
        z,
    )
}
pub fn gated_delta_step_gate_wmma_qknormed_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
    z: Bf16Arena<'_>,
) -> HipResult<()> {
    chunk_arena(
        gpu,
        p,
        gate,
        GatedDeltaChunkArms {
            qk_prenormed: true,
            q8_inline: gated_delta_q8_inline_route(gpu, p),
        },
        z,
    )
}
/// Explicit inline-Q8 twin, independent of the environment's inline selector.
pub fn gated_delta_step_gate_q8_wmma_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
    z: Bf16Arena<'_>,
) -> HipResult<()> {
    chunk_arena(
        gpu,
        p,
        gate,
        GatedDeltaChunkArms {
            qk_prenormed: false,
            q8_inline: true,
        },
        z,
    )
}

pub fn indexed_attention_norm_rope_batch_arena(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRopeBatch<'_>,
    src: Bf16Arena<'_>,
) -> HipResult<()> {
    ensure_f32(p.values)?;
    src.validate(p.rows, p.row_stride)?;
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
        "npu_trunk_consumers",
        ARENA_SRC,
        "indexed_attention_norm_rope_batch_arena",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(src.tensor.buf.as_ptr());
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
    src.args(&mut args)?;
    // After the declared position field, as in the incumbent launcher.
    p.yarn.push(&mut args);
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents (`indexed_attention_norm_rope_batch_arena`): grid `[heads, 1,
    // rows]` exact; each head copies its whole `head_stride` (the last head the
    // rest of `row_stride`) from the BF16 arena and writes the F32 row, so the
    // kernel touches the FULL `row_stride` of every row (wider than the incumbent
    // launcher's `head_span`): BF16 source `(rows - 1) * src_pitch + row_stride`
    // elements from `src_col0`, F32 `values` `rows * row_stride`, `head_dim`
    // norm BF16.
    let extents = || {
        (
            [
                (
                    "qsa_norm_rope_src",
                    src.tensor,
                    src.col0_elems * 2,
                    ((p.rows - 1) * src.row_pitch_elems + p.row_stride) * 2,
                ),
                (
                    "qsa_norm_rope_values",
                    p.values,
                    0usize,
                    p.rows * p.row_stride * 4,
                ),
                ("qsa_norm_rope_norm", p.norm, 0, p.head_dim * 2),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position: p.position_start + p.rows,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_arena_extent_check(
            gpu,
            &sites,
            depth,
            "indexed_attention_norm_rope_batch_arena",
        );
    }
    gpu.launch_blob_recorded(
        "indexed_attention_norm_rope_batch_arena",
        [head_grid, dim_grid, row_grid],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_arena_extent_post(
            gpu,
            &sites,
            depth,
            "indexed_attention_norm_rope_batch_arena",
        );
    }
    Ok(())
}

fn validate_arena_params(
    p: &GatedDeltaParamsBatched<'_>,
    a: Bf16Arena<'_>,
    b: Bf16Arena<'_>,
) -> HipResult<usize> {
    a.validate(p.rows, p.heads)?;
    b.validate(p.rows, p.heads)?;
    let elements = checked_product(p.rows, p.heads, "arena parameters")?;
    for t in [p.gate, p.beta] {
        ensure_f32(t)?;
        if t.numel() != elements {
            return Err(HipError::new(0, "parameter output shape"));
        }
    }
    for t in [p.a_log, p.dt_bias] {
        if t.dtype != DType::BF16 || t.numel() != p.heads {
            return Err(HipError::new(0, "parameter weight shape"));
        }
    }
    Ok(elements)
}

pub fn gated_delta_params_batched_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaParamsBatched<'_>,
    a: Bf16Arena<'_>,
    b: Bf16Arena<'_>,
) -> HipResult<()> {
    let elements = validate_arena_params(p, a, b)?;
    let rows = checked_i32(p.rows, "GDN batched parameter rows")?;
    let heads = checked_i32(p.heads, "GDN batched parameter heads")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public(
        "npu_trunk_consumers",
        ARENA_SRC,
        "gated_delta_params_batched_arena",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [a.tensor, b.tensor, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(heads);
    a.args(&mut args)?;
    b.args(&mut args)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_batched_arena",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn gated_delta_gate_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaGate<'_>,
    z: Bf16Arena<'_>,
) -> HipResult<()> {
    z.validate(
        1,
        checked_product(p.value_heads, p.value_dim, "arena gate width")?,
    )?;
    let local = GatedDeltaGate {
        recurrent_output: p.recurrent_output,
        z: p.recurrent_output,
        norm: p.norm,
        output: p.output,
        value_heads: p.value_heads,
        value_dim: p.value_dim,
    };
    let (value_heads, value_dim) = validate_gated_delta_gate(&local)?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN gate head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    gpu.ensure_kernel_public("npu_trunk_consumers", ARENA_SRC, "gated_delta_gate_arena")?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, z.tensor, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    z.args(&mut args)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_gate_arena",
        [value_heads_grid, value_dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn gated_delta_gate_batched_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    z: Bf16Arena<'_>,
) -> HipResult<()> {
    z.validate(
        p.rows,
        checked_product(p.value_heads, p.value_dim, "arena gate width")?,
    )?;
    let local = GatedDeltaGateBatched {
        recurrent_output: p.recurrent_output,
        z: p.recurrent_output,
        norm: p.norm,
        output: p.output,
        rows: p.rows,
        value_heads: p.value_heads,
        value_dim: p.value_dim,
    };
    let elements = validate_gated_delta_gate_batched(&local)?;
    let rows = checked_i32(p.rows, "GDN batched gate rows")?;
    let value_heads = checked_i32(p.value_heads, "GDN batched gate heads")?;
    let value_dim = checked_i32(p.value_dim, "GDN batched gate width")?;
    let value_heads_grid = checked_u32(elements / p.value_dim, "GDN batched gate grid")?;
    gpu.ensure_kernel_public(
        "npu_trunk_consumers",
        ARENA_SRC,
        "gated_delta_gate_batched_arena",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, z.tensor, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    z.args(&mut args)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_gate_batched_arena",
        [value_heads_grid, 1, 1],
        [32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Direct copy primitive used by the cache consumers and the lab's incumbent
/// materialization oracle. F32 widening is exactly `bits << 16`.
pub fn npu_trunk_copy_projection_arena(
    gpu: &mut Gpu,
    src: Bf16Arena<'_>,
    dst: &GpuTensor,
    rows: usize,
    cols: usize,
    dst_pitch: usize,
    dst_col0: usize,
    position_start: usize,
) -> HipResult<()> {
    src.validate(rows, cols)?;
    arena_copy(
        gpu,
        src.tensor,
        src.row_pitch_elems,
        src.col0_elems,
        dst,
        rows,
        cols,
        dst_pitch,
        dst_col0,
        position_start,
        false,
    )
}

/// One operand of an arena QSA append launch checked under
/// `qwen4_qsa_extent_check`: arena label, tensor, byte offset, accessed bytes.
type QsaArenaExtent<'a> = (&'static str, &'a GpuTensor, usize, usize);

/// Pre-launch extent check of every operand (callers gate the call and the
/// computation of `extents` on `gpu.flags.qwen4_qsa_extent_check`).
fn qsa_arena_extent_check(
    gpu: &Gpu,
    extents: &[QsaArenaExtent<'_>],
    depth: crate::dispatch::QsaExtentDepth,
    site: &'static str,
) {
    for &(arena, tensor, offset, len) in extents {
        gpu.qsa_extent_check(arena, tensor, offset, len, depth, site);
    }
}

/// Post-launch helper for every operand, called immediately after the launch.
fn qsa_arena_extent_post(
    gpu: &Gpu,
    extents: &[QsaArenaExtent<'_>],
    depth: crate::dispatch::QsaExtentDepth,
    site: &'static str,
) {
    for &(arena, tensor, offset, len) in extents {
        gpu.qsa_extent_post_launch(arena, tensor, offset, len, depth, site);
    }
}

fn arena_copy(
    gpu: &mut Gpu,
    src: &GpuTensor,
    src_pitch: usize,
    src_col0: usize,
    dst: &GpuTensor,
    rows: usize,
    cols: usize,
    dst_pitch: usize,
    dst_col0: usize,
    position_start: usize,
    bind_position: bool,
) -> HipResult<()> {
    if !gpu.arch_caps.is_gfx1151() || rows == 0 || cols == 0 {
        return Err(HipError::new(
            0,
            "arena PM copy requires gfx1151 and nonempty rows",
        ));
    }
    let symbol = match (src.dtype, dst.dtype) {
        (DType::BF16, DType::F32) => "npu_arena_copy_bf16_f32_pm",
        (DType::BF16, DType::BF16) => "npu_arena_copy_bf16_bf16_pm",
        (DType::F32, DType::F32) => "npu_arena_copy_f32_f32_pm",
        _ => return Err(HipError::new(0, "unsupported arena PM copy dtype")),
    };
    let src_span = checked_add(src_col0, cols, "arena copy source span")?;
    let dst_span = checked_add(dst_col0, cols, "arena copy destination span")?;
    let src_end = checked_add(
        checked_product(rows - 1, src_pitch, "arena copy source")?,
        src_span,
        "arena copy source",
    )?;
    let last_row = checked_add(position_start, rows - 1, "arena copy position")?;
    let dst_end = checked_add(
        checked_product(last_row, dst_pitch, "arena copy destination")?,
        dst_span,
        "arena copy destination",
    )?;
    if src_span > src_pitch
        || dst_span > dst_pitch
        || src_end > src.numel()
        || dst_end > dst.numel()
        || checked_product(src_end, src.dtype.size(), "arena copy source bytes")? > src.buf.size()
        || checked_product(dst_end, dst.dtype.size(), "arena copy destination bytes")?
            > dst.buf.size()
    {
        return Err(HipError::new(0, "arena copy exceeds row or allocation"));
    }
    checked_u32(
        checked_product(src_end, src.dtype.size(), "arena PM source offset")?,
        "arena PM source offset",
    )?;
    checked_u32(
        checked_product(dst_end, dst.dtype.size(), "arena PM destination offset")?,
        "arena PM destination offset",
    )?;
    static VERIFIED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        format!(
            "{:x}",
            Sha256::digest(crate::kernels::NPU_TRUNK_ARENA_PM_GFX1151)
        ) == crate::kernels::NPU_TRUNK_ARENA_PM_GFX1151_SHA256
    });
    if !*VERIFIED {
        return Err(HipError::new(
            0,
            "arena PM image differs from certified pin",
        ));
    }
    gpu.bind_thread()?;
    gpu.ensure_embedded_kernel(
        "npu_trunk_arena_pm_gfx1151",
        crate::kernels::NPU_TRUNK_ARENA_PM_GFX1151,
        symbol,
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(src.buf.as_ptr());
    args.push_ptr(dst.buf.as_ptr());
    for n in [
        src_pitch,
        src_col0,
        dst_pitch,
        dst_col0,
        cols,
        position_start,
    ] {
        args.push_i32(checked_i32(n, "arena copy argument")?);
    }
    let position_offset = args.len() - 4;
    args.pad_to(16);
    let binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents (`npu_arena_copy_*_pm`): thread `t < cols` of row `r < rows` copies
    // `src[r * src_pitch + src_col0 + t]` to `dst[(position_start + r) * dst_pitch
    // + dst_col0 + t]` (the `cols.div_ceil(32)` grid round-up is masked by
    // `cols > t`). This shared helper writes the QSA full K/V caches (F32 path),
    // the raw index keys and the index-row tail, so every call is checked.
    let extents = || {
        let (src_elem, dst_elem) = (src.dtype.size(), dst.dtype.size());
        (
            [
                (
                    "qsa_arena_copy_src",
                    src,
                    src_col0 * src_elem,
                    ((rows - 1) * src_pitch + cols) * src_elem,
                ),
                (
                    "qsa_arena_copy_dst",
                    dst,
                    (position_start * dst_pitch + dst_col0) * dst_elem,
                    ((rows - 1) * dst_pitch + cols) * dst_elem,
                ),
            ],
            crate::dispatch::QsaExtentDepth {
                rows,
                end_position: position_start + rows,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_arena_extent_check(gpu, &sites, depth, "arena_copy");
    }
    gpu.launch_blob_recorded(
        symbol,
        [
            checked_u32(cols.div_ceil(32), "arena copy grid")?,
            checked_u32(rows, "arena copy rows")?,
            1,
        ],
        [32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: if bind_position { &binding } else { &[] },
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_arena_extent_post(gpu, &sites, depth, "arena_copy");
    }
    Ok(())
}

/// `key = None` reads normalized F32 K from `p.key`; `Some` reads direct BF16.
/// V always reads directly from its BF16 arena view.
pub fn indexed_attention_cache_append_batch_arena(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppendBatch<'_>,
    key: Option<Bf16Arena<'_>>,
    value: Bf16Arena<'_>,
) -> HipResult<()> {
    let width = checked_product(p.kv_heads, p.head_dim, "arena KV width")?;
    value.validate(p.rows, width)?;
    let (key_tensor, key_pitch, key_col0, key_bf16) = match key {
        Some(k) => {
            k.validate(p.rows, width)?;
            (k.tensor, k.row_pitch_elems, k.col0_elems, true)
        }
        None => {
            ensure_f32(p.key)?;
            if p.key.numel() < checked_product(p.rows, width, "arena normalized K")? {
                return Err(HipError::new(0, "undersized normalized K"));
            }
            (p.key, width, 0, false)
        }
    };
    check_qsa_format(
        gpu,
        p.format,
        p.kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    let end = checked_add(p.position_start, p.rows, "arena KV position")?;
    let units = checked_product(
        end,
        p.format.kv_row_units(p.kv_heads, p.head_dim),
        "arena KV cache",
    )?;
    if p.full_keys.numel() != p.full_values.numel() || p.full_keys.numel() < units {
        return Err(HipError::new(0, "undersized arena KV cache"));
    }
    checked_i32(end, "arena KV position")?;
    if p.format == QsaKvFormat::F32 {
        arena_copy(
            gpu,
            key_tensor,
            key_pitch,
            key_col0,
            p.full_keys,
            p.rows,
            width,
            width,
            0,
            p.position_start,
            true,
        )?;
        return arena_copy(
            gpu,
            value.tensor,
            value.row_pitch_elems,
            value.col0_elems,
            p.full_values,
            p.rows,
            width,
            width,
            0,
            p.position_start,
            true,
        );
    }
    let kernel = match p.format {
        QsaKvFormat::Fp8 => "indexed_attention_cache_append_fp8_batch_arena",
        QsaKvFormat::Q8 => "indexed_attention_cache_append_q8_batch_arena",
        _ => {
            return Err(HipError::new(
                0,
                "QSA format has no BF16 projection-arena append kernel",
            ))
        }
    };
    let mut args = KernargBlob::new();
    for t in [key_tensor, value.tensor, p.full_keys, p.full_values] {
        args.push_ptr(t.buf.as_ptr());
    }
    args.push_i32(checked_i32(p.rows, "arena KV rows")?);
    args.push_i32(checked_i32(p.position_start, "arena KV position")?);
    let position_offset = args.len() - 4;
    for n in [
        p.kv_heads,
        p.head_dim,
        key_pitch,
        key_col0,
        value.row_pitch_elems,
        value.col0_elems,
    ] {
        args.push_i32(checked_i32(n, "arena KV argument")?);
    }
    args.push_i32(i32::from(key_bf16));
    let binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents (`qsa_cache_append_batched_arena`): guard `head < kv_heads && row <
    // rows`, reads (`c < head_dim`) key `row * key_pitch + key_col0 + head *
    // head_dim + c` (BF16 arena or F32 normalized K) and BF16 value `row *
    // value_pitch + value_col0 + ...`, i.e. `(rows - 1) * pitch + width`
    // elements from the column origin; each cache receives `rows` whole format
    // rows (`kv_row_bytes`: fp8 `kv_heads * (head_dim + 2)`, q8 `kv_heads *
    // head_dim / 32 * 34`) at `position_start * kv_row_bytes`
    // (`qsa_kv_store` stores nothing for `c >= head_dim`).
    let extents = || {
        let row_bytes = p.format.kv_row_bytes(p.kv_heads, p.head_dim);
        let key_elem = key_tensor.dtype.size();
        (
            [
                (
                    "qsa_keys",
                    key_tensor,
                    key_col0 * key_elem,
                    ((p.rows - 1) * key_pitch + width) * key_elem,
                ),
                (
                    "qsa_values",
                    value.tensor,
                    value.col0_elems * 2,
                    ((p.rows - 1) * value.row_pitch_elems + width) * 2,
                ),
                (
                    "qsa_full_keys",
                    p.full_keys,
                    p.position_start * row_bytes,
                    p.rows * row_bytes,
                ),
                (
                    "qsa_full_values",
                    p.full_values,
                    p.position_start * row_bytes,
                    p.rows * row_bytes,
                ),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position: end,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_arena_extent_check(
            gpu,
            &sites,
            depth,
            "indexed_attention_cache_append_batch_arena",
        );
    }
    arena_launch(
        gpu,
        kernel,
        [
            checked_u32(p.kv_heads, "arena KV heads")?,
            checked_u32(p.rows, "arena KV rows")?,
            1,
        ],
        256,
        &mut args,
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_arena_extent_post(
            gpu,
            &sites,
            depth,
            "indexed_attention_cache_append_batch_arena",
        );
    }
    Ok(())
}

/// Direct BF16 index-key append, also writing the incumbent's F32 tail in
/// `p.index_rows`. Its normalized query columns remain unchanged. Raw-key
/// storage can be BF16 or F32; downstream select reads the normalized F32 query.
pub fn indexed_attention_index_key_append_batch_arena(
    gpu: &mut Gpu,
    p: &IndexedAttentionIndexKeyAppendBatch<'_>,
    src: Bf16Arena<'_>,
) -> HipResult<()> {
    let width = checked_add(p.index_q_width, p.index_kv_width, "arena index width")?;
    src.validate(p.rows, width)?;
    ensure_f32(p.index_rows)?;
    let tail = Bf16Arena {
        col0_elems: checked_add(src.col0_elems, p.index_q_width, "arena index tail")?,
        ..src
    };
    npu_trunk_copy_projection_arena(
        gpu,
        tail,
        p.index_rows,
        p.rows,
        p.index_kv_width,
        width,
        p.index_q_width,
        0,
    )?;
    arena_copy(
        gpu,
        tail.tensor,
        tail.row_pitch_elems,
        tail.col0_elems,
        p.raw_index_keys,
        p.rows,
        p.index_kv_width,
        p.index_kv_width,
        0,
        p.position_start,
        true,
    )
}

/// Non-chunk gate plus the incumbent output-projection FWHT rotation.
pub fn gated_delta_gate_batched_rotate_arena(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    rotated: &GpuTensor,
    z: Bf16Arena<'_>,
) -> HipResult<()> {
    z.validate(
        p.rows,
        checked_product(p.value_heads, p.value_dim, "arena gate width")?,
    )?;
    let local = GatedDeltaGateBatched {
        recurrent_output: p.recurrent_output,
        z: p.recurrent_output,
        norm: p.norm,
        output: p.output,
        rows: p.rows,
        value_heads: p.value_heads,
        value_dim: p.value_dim,
    };
    let elements = validate_gated_delta_gate_batched(&local)?;
    ensure_f32(rotated)?;
    if !p.value_heads.is_multiple_of(2) || rotated.numel() < elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    const KERNEL: &str = "gated_delta_gate_batched_rotate_arena";
    gpu.ensure_kernel_public("npu_trunk_consumers", ARENA_SRC, KERNEL)?;
    gpu.ensure_mq_signs()?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, z.tensor, p.norm, p.output, rotated] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
    args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
    z.args(&mut args)?;
    args.push_i32(checked_i32(p.value_heads, "arena gate heads")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        KERNEL,
        [checked_u32(elements / 256, "arena gate pair grid")?, 1, 1],
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

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata(n: usize, bytes: usize, dtype: DType) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut(), bytes) },
            shape: vec![n],
            dtype,
        }
    }
    #[test]
    fn arena_extent_checks_pitch_columns_ragged_rows_and_allocation() {
        for (rows, pitch, col, cols) in [(8192, 16640, 16384, 48), (4647, 14080, 13312, 640)] {
            let n = (rows - 1) * pitch + col + cols;
            let t = metadata(n, n * 2, DType::BF16);
            let a = Bf16Arena {
                tensor: &t,
                row_pitch_elems: pitch,
                col0_elems: col,
            };
            assert!(a.validate(rows, cols).is_ok());
            assert!(a.validate(rows + 1, cols).is_err());
            assert!(a.validate(rows, pitch - col + 1).is_err());
            assert!(a.validate(0, cols).is_err());
            let short = metadata(n, n * 2 - 1, DType::BF16);
            assert!(Bf16Arena {
                tensor: &short,
                ..a
            }
            .validate(rows, cols)
            .is_err());
            let wrong = metadata(n, n * 4, DType::F32);
            assert!(Bf16Arena {
                tensor: &wrong,
                ..a
            }
            .validate(rows, cols)
            .is_err());
        }
    }
    #[test]
    fn arena_image_is_pinned_and_incumbent_sources_are_not_armed() {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(crate::kernels::NPU_TRUNK_ARENA_PM_GFX1151)
            ),
            crate::kernels::NPU_TRUNK_ARENA_PM_GFX1151_SHA256
        );
        assert!(!TENSOR_OPS_SRC.contains("#define NPU_TRUNK_ARENA"));
        assert!(!GATED_DELTA_CHUNK_WMMA_SRC.contains("#define NPU_TRUNK_ARENA"));
        assert!(!GATED_DELTA_CHUNK_Q8_WMMA_SRC.contains("#define NPU_TRUNK_ARENA"));
        assert!(ARENA_SRC.starts_with("#define NPU_TRUNK_ARENA 1"));
    }
    /// The Q8 arena source renames only the plain Q8 slot entry and carries no EF text, so the
    /// plain Q8 kernel's code object is its own.
    #[test]
    fn arena_q8_source_aliases_plain_entry_only() {
        assert!(CHUNK_Q8_ARENA_SRC.starts_with("#define NPU_TRUNK_ARENA 1"));
        assert!(CHUNK_Q8_ARENA_SRC.contains(
            "#define gated_delta_chunk_gate_q8_wmma gated_delta_chunk_gate_q8_wmma_arena\n"
        ));
        assert!(GATED_DELTA_CHUNK_Q8_WMMA_SRC.contains("void gated_delta_chunk_gate_q8_wmma("));
        assert_eq!(
            CHUNK_Q8_ARENA_SRC
                .matches("#define gated_delta_chunk_gate_")
                .count(),
            1
        );
        assert!(!CHUNK_Q8_ARENA_SRC.contains("q8ef"));
        assert!(!GATED_DELTA_CHUNK_Q8_WMMA_SRC.contains("q8ef"));
    }
    /// The arena QSA norm/RoPE twin equals the incumbent batched launch bit for
    /// bit (every column, including the untouched gate / index-key tails),
    /// unscaled and under YaRN (factor 4 over 262,144) past the native window.
    #[test]
    fn qsa_norm_rope_arena_matches_incumbent_with_and_without_yarn() {
        let Ok(mut gpu) = crate::gpu_test::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        use crate::rope::{Rope, RopeScaling, RopeYarn, Yarn};
        let yarn = Rope {
            theta: 1e7,
            scaling: RopeScaling::Yarn(Yarn::new(4.0, 262_144, None, None, None, None)),
        }
        .yarn(64);
        // Finite BF16 values (truncated f32 bit patterns) in a pitched arena.
        let bf16 = |x: f32| (x.to_bits() >> 16) as u16;
        let (rows, pitch, col) = (5usize, 1792usize, 96usize);
        let bits: Vec<u16> = (0..rows * pitch)
            .map(|i| bf16(((i * 37) % 101) as f32 / 25.0 - 2.0))
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut arena = gpu.upload_raw(&bytes, &[bits.len()]).expect("arena upload");
        arena.dtype = DType::BF16;
        // (heads, head_dim, head_stride, width): query+gate rows and index rows.
        for (heads, head_dim, head_stride, width) in [(2, 256, 512, 1024), (4, 128, 128, 640)] {
            let norm_bits: Vec<u8> = (0..head_dim)
                .flat_map(|c| bf16((c % 7) as f32 * 0.125 - 0.25).to_le_bytes())
                .collect();
            let mut norm = gpu
                .upload_raw(&norm_bits, &[head_dim])
                .expect("norm upload");
            norm.dtype = DType::BF16;
            let input: Vec<f32> = (0..rows * width)
                .map(|i| f32::from_bits((bits[(i / width) * pitch + col + i % width] as u32) << 16))
                .collect();
            for (position_start, yarn) in [(3, RopeYarn::OFF), (300_003, yarn)] {
                let incumbent = gpu
                    .upload_f32(&input, &[input.len()])
                    .expect("incumbent upload");
                let direct = gpu
                    .upload_f32(&vec![f32::NAN; input.len()], &[input.len()])
                    .expect("direct upload");
                let p = |values| IndexedAttentionNormRopeBatch {
                    values,
                    norm: &norm,
                    rows,
                    row_stride: width,
                    heads,
                    head_dim,
                    head_stride,
                    position_start,
                    rotary_dim: 64,
                    yarn,
                };
                indexed_attention_norm_rope_batch(&mut gpu, &p(&incumbent)).expect("incumbent");
                let src = Bf16Arena {
                    tensor: &arena,
                    row_pitch_elems: pitch,
                    col0_elems: col,
                };
                indexed_attention_norm_rope_batch_arena(&mut gpu, &p(&direct), src).expect("arena");
                let want = gpu.download_f32(&incumbent).expect("incumbent download");
                let got = gpu.download_f32(&direct).expect("direct download");
                for (index, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{heads}x{head_dim} {yarn:?} at {index}: {g} vs {w}"
                    );
                }
                gpu.free_tensor(incumbent).expect("free incumbent");
                gpu.free_tensor(direct).expect("free direct");
            }
            gpu.free_tensor(norm).expect("free norm");
        }
        gpu.free_tensor(arena).expect("free arena");
    }
}
