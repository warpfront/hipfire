// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-batched PLE tail launcher (`HIPFIRE_QWEN4_LANE_PLE_BATCH`, exact
//! gfx1201).
//!
//! One launch replaces the per-lane `copy_f32_buffer` / `grouped_gate_bf16` /
//! `grouped_norm_bf16` / `grouped_depthwise_conv_silu_add_bf16` / `add_f32`
//! chain of every lane of an AR lane step (one row per lane, F32 HC streams),
//! byte for byte (`kernels/src/ple_lanes.hip` header). Each lane's own
//! convolution state rides in a by-value kernarg lane table (no device table,
//! no upload), so the launch is retained under capture like any other
//! `launch_maybe_blob` launch. Self-contained through `include_str!`: the
//! shipping `grouped_ops.hip` source (and every walked route plan digest of it)
//! is untouched.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source of [`Gpu::ple_lanes_tail_f32`].
pub const PLE_LANES_SRC: &str = include_str!("../../../kernels/src/ple_lanes.hip");
/// Module key and device symbol.
pub const PLE_LANES_SYMBOL: &str = "ple_lanes_tail_f32";
/// Lanes per launch (`PLE_LANES_MAX` in the kernel).
pub const PLE_LANES_MAX: usize = 8;
const BLOCK: u32 = 256;

/// `#[repr(C)]` mirror of `struct PleLaneTable` (the by-value kernel argument).
#[repr(C)]
struct PleLaneTableArgs {
    state: [u64; PLE_LANES_MAX],
}

/// Operands of [`Gpu::ple_lanes_tail_f32`]. `lanes = states.len()` rows of
/// `groups * group_size` F32 channels (value: `group_size` per row), one row
/// per lane in lane order.
pub struct PleLanesTail<'a> {
    pub key: &'a GpuTensor,
    pub streams: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub key_norm: &'a GpuTensor,
    pub query_norm: &'a GpuTensor,
    pub conv_norm: &'a GpuTensor,
    pub conv_weight: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    /// Lane `i`'s convolution state `[history_rows, channels]`.
    pub states: &'a [&'a GpuTensor],
    pub groups: usize,
    pub group_size: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub epsilon: f32,
}

fn bad(what: &str) -> HipError {
    HipError::new(0, what)
}

fn require(tensor: &GpuTensor, dtype: DType, elements: usize, what: &str) -> HipResult<()> {
    if tensor.dtype != dtype {
        return Err(bad(&format!(
            "{what} must have dtype {dtype:?}, got {:?}",
            tensor.dtype
        )));
    }
    if tensor.numel() != elements {
        return Err(bad(&format!(
            "{what} must have {elements} elements, got {}",
            tensor.numel()
        )));
    }
    Ok(())
}

impl Gpu {
    /// The per-lane PLE gate / norm / conv / stream-add chain of
    /// `p.states.len()` (1..=[`PLE_LANES_MAX`]) lanes in ONE launch; grid
    /// `[groups, lanes]` x 256. `streams` is updated in place, `normed` and
    /// every state are written exactly as the incumbent chain writes them.
    pub fn ple_lanes_tail_f32(&mut self, p: &PleLanesTail<'_>) -> HipResult<()> {
        let lanes = p.states.len();
        if !(1..=PLE_LANES_MAX).contains(&lanes) {
            return Err(bad(&format!(
                "ple_lanes_tail_f32: {lanes} lanes outside 1..={PLE_LANES_MAX}"
            )));
        }
        if p.groups == 0 || p.group_size == 0 || p.kernel_size == 0 || p.dilation == 0 {
            return Err(bad("ple_lanes_tail_f32: invalid geometry"));
        }
        let channels = p
            .groups
            .checked_mul(p.group_size)
            .ok_or_else(|| bad("ple_lanes_tail_f32: channels overflow"))?;
        let elements = channels
            .checked_mul(lanes)
            .filter(|v| *v <= i32::MAX as usize)
            .ok_or_else(|| bad("ple_lanes_tail_f32: lane rows exceed i32"))?;
        let history_rows = (p.kernel_size - 1)
            .checked_mul(p.dilation)
            .ok_or_else(|| bad("ple_lanes_tail_f32: history rows overflow"))?;
        let history_elements = history_rows
            .checked_mul(channels)
            .filter(|v| *v <= i32::MAX as usize)
            .ok_or_else(|| bad("ple_lanes_tail_f32: history exceeds i32"))?;
        let lds_bytes = p.group_size * 8;
        if lds_bytes > 64 * 1024 {
            return Err(bad("ple_lanes_tail_f32: gate row exceeds LDS"));
        }
        require(p.key, DType::F32, elements, "ple lanes key")?;
        require(p.streams, DType::F32, elements, "ple lanes streams")?;
        require(p.value, DType::F32, lanes * p.group_size, "ple lanes value")?;
        require(p.normed, DType::F32, elements, "ple lanes normed")?;
        require(p.key_norm, DType::BF16, channels, "ple lanes key norm")?;
        require(p.query_norm, DType::BF16, channels, "ple lanes query norm")?;
        require(
            p.conv_norm,
            DType::BF16,
            channels,
            "ple lanes convolution norm",
        )?;
        require(
            p.conv_weight,
            DType::BF16,
            channels * p.kernel_size,
            "ple lanes convolution weight",
        )?;
        let mut args = PleLaneTableArgs {
            state: [0; PLE_LANES_MAX],
        };
        for (i, state) in p.states.iter().enumerate() {
            require(state, DType::F32, history_elements, "ple lanes state")?;
            args.state[i] = state.buf.as_ptr() as u64;
        }
        self.bind_thread()?;
        self.ensure_kernel(PLE_LANES_SYMBOL, PLE_LANES_SRC, PLE_LANES_SYMBOL)?;
        let key_ptr = p.key.buf.as_ptr();
        let streams_ptr = p.streams.buf.as_ptr();
        let value_ptr = p.value.buf.as_ptr();
        let key_norm_ptr = p.key_norm.buf.as_ptr();
        let query_norm_ptr = p.query_norm.buf.as_ptr();
        let conv_norm_ptr = p.conv_norm.buf.as_ptr();
        let conv_weight_ptr = p.conv_weight.buf.as_ptr();
        let normed_ptr = p.normed.buf.as_ptr();
        let groups_i = p.groups as i32;
        let hidden_i = p.group_size as i32;
        let kernel_i = p.kernel_size as i32;
        let dilation_i = p.dilation as i32;
        let epsilon = p.epsilon;
        let mut params: Vec<*mut c_void> = vec![
            &key_ptr as *const _ as *mut c_void,
            &streams_ptr as *const _ as *mut c_void,
            &value_ptr as *const _ as *mut c_void,
            &key_norm_ptr as *const _ as *mut c_void,
            &query_norm_ptr as *const _ as *mut c_void,
            &conv_norm_ptr as *const _ as *mut c_void,
            &conv_weight_ptr as *const _ as *mut c_void,
            &normed_ptr as *const _ as *mut c_void,
            &mut args as *mut _ as *mut c_void,
            &groups_i as *const _ as *mut c_void,
            &hidden_i as *const _ as *mut c_void,
            &kernel_i as *const _ as *mut c_void,
            &dilation_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let blob_args = &args;
        self.launch_maybe_blob(
            PLE_LANES_SYMBOL,
            [p.groups as u32, lanes as u32, 1],
            [BLOCK, 1, 1],
            lds_bytes as u32,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(key_ptr);
                b.push_ptr(streams_ptr);
                b.push_ptr(value_ptr);
                b.push_ptr(key_norm_ptr);
                b.push_ptr(query_norm_ptr);
                b.push_ptr(conv_norm_ptr);
                b.push_ptr(conv_weight_ptr);
                b.push_ptr(normed_ptr);
                // The 64 B table is 8-aligned: it starts at offset 64.
                for v in &blob_args.state {
                    b.push_u64(*v);
                }
                b.push_i32(groups_i);
                b.push_i32(hidden_i);
                b.push_i32(kernel_i);
                b.push_i32(dilation_i);
                b.push_f32(epsilon);
                b
            },
        )
    }
}
