// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parameterized grouped GPU lowering.
//!
//! The CPU owner supplies completed contiguous BF16 rows.  These launchers only
//! perform bounded device-side conversion and equation-matching projection,
//! gating, normalization, and depthwise convolution.  Source I/O and row
//! lookup stay outside the GPU segment, so the launch path remains valid for
//! ordinary HIP and for graph capture after kernels have been warmed.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::dispatch::{DType, Gpu, GpuTensor};

/// Embedded source for grouped tensor kernels.
pub const GROUPED_KERNEL_SRC: &str = include_str!("../../../kernels/src/grouped_ops.hip");
const GROUPED_MODULE: &str = "grouped_ops";
const BLOCK: u32 = 256;

impl Gpu {
    /// Widen contiguous little-endian BF16 rows into F32 on device.
    ///
    /// `staged` is flattened `[tokens, rows_per_token, row_width]` BF16 and
    /// `output` is the same flattened logical shape in F32.  No host row I/O occurs here.
    pub fn grouped_gather_convert_bf16(
        &mut self,
        staged: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        rows_per_token: usize,
        row_width: usize,
    ) -> HipResult<()> {
        let rows = checked_product(tokens, rows_per_token, "grouped token rows")?;
        let elements = checked_product(rows, row_width, "grouped row width")?;
        require_dtype(staged, DType::BF16, "grouped BF16 staging")?;
        require_dtype(output, DType::F32, "grouped F32 embedding output")?;
        require_numel(staged, elements, "grouped BF16 staging")?;
        require_numel(output, elements, "grouped F32 embedding output")?;
        if elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped token count")?;
        let rows_i = checked_i32(rows_per_token, "grouped rows per token")?;
        let width_i = checked_i32(row_width, "grouped row width")?;
        self.bind_thread()?;
        self.ensure_kernel(
            GROUPED_MODULE,
            GROUPED_KERNEL_SRC,
            "grouped_gather_convert_bf16",
        )?;
        let staged_ptr = staged.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params = [
            &staged_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &width_i as *const _ as *mut c_void,
        ];
        let grid = checked_grid(elements, BLOCK, "grouped gather grid")?;
        self.launch_maybe_blob(
            "grouped_gather_convert_bf16",
            [grid, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(staged_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(rows_i);
                blob.push_i32(width_i);
                blob
            },
        )
    }

    /// Row-major F32 projection `output = input · weight^T`.
    pub fn grouped_linear_f32(
        &mut self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> HipResult<()> {
        let input_elements = checked_product(tokens, in_dim, "grouped linear input")?;
        let weight_elements = checked_product(out_dim, in_dim, "grouped linear weight")?;
        let output_elements = checked_product(tokens, out_dim, "grouped linear output")?;
        require_dtype(input, DType::F32, "grouped linear input")?;
        require_dtype(weight, DType::F32, "grouped linear weight")?;
        require_dtype(output, DType::F32, "grouped linear output")?;
        require_numel(input, input_elements, "grouped linear input")?;
        require_numel(weight, weight_elements, "grouped linear weight")?;
        require_numel(output, output_elements, "grouped linear output")?;
        if output_elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped linear token count")?;
        let in_dim_i = checked_i32(in_dim, "grouped linear input width")?;
        let out_dim_i = checked_i32(out_dim, "grouped linear output width")?;
        self.bind_thread()?;
        self.ensure_kernel(GROUPED_MODULE, GROUPED_KERNEL_SRC, "grouped_linear_f32")?;
        let input_ptr = input.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params = [
            &input_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &in_dim_i as *const _ as *mut c_void,
            &out_dim_i as *const _ as *mut c_void,
        ];
        let grid_x = checked_grid(out_dim, BLOCK, "grouped linear output grid")?;
        let grid_y = checked_u32(tokens, "grouped linear token grid")?;
        self.launch_maybe_blob(
            "grouped_linear_f32",
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(in_dim_i);
                blob.push_i32(out_dim_i);
                blob
            },
        )
    }

    /// Project/gate the shared value across all grouped residual branches.
    ///
    /// `key` and `query` are `[tokens, hc_count * hidden_size]`, `value` is
    /// `[tokens, hidden_size]`, and each normalization weight is
    /// `[hc_count * hidden_size]`.  The kernel applies grouped RMS with
    /// `(1 + weight)`, signed-square-root dot gating, and sigmoid activation.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_gate_f32(
        &mut self,
        key: &GpuTensor,
        query: &GpuTensor,
        value: &GpuTensor,
        key_norm: &GpuTensor,
        query_norm: &GpuTensor,
        gated: &GpuTensor,
        tokens: usize,
        hc_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(hc_count, hidden_size, "grouped gated channels")?;
        require_dtype(key, DType::F32, "grouped key projection")?;
        require_dtype(query, DType::F32, "grouped query projection")?;
        require_dtype(value, DType::F32, "grouped value projection")?;
        require_dtype(key_norm, DType::F32, "grouped key norm")?;
        require_dtype(query_norm, DType::F32, "grouped query norm")?;
        require_dtype(gated, DType::F32, "grouped gated output")?;
        require_numel(
            key,
            checked_product(tokens, channels, "grouped key elements")?,
            "grouped key projection",
        )?;
        require_numel(
            query,
            checked_product(tokens, channels, "grouped query elements")?,
            "grouped query projection",
        )?;
        require_numel(
            value,
            checked_product(tokens, hidden_size, "grouped value elements")?,
            "grouped value projection",
        )?;
        require_numel(key_norm, channels, "grouped key norm")?;
        require_numel(query_norm, channels, "grouped query norm")?;
        require_numel(
            gated,
            checked_product(tokens, channels, "grouped gated elements")?,
            "grouped gated output",
        )?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        // The kernel stages rows in GROUPED_GATE_CHUNK (64) index chunks.
        if hidden_size % 64 != 0 {
            return Err(HipError::new(
                0,
                "grouped gate hidden width must be a multiple of 64",
            ));
        }
        let tokens_i = checked_i32(tokens, "grouped gate token count")?;
        let hc_i = checked_i32(hc_count, "grouped branch count")?;
        let hidden_i = checked_i32(hidden_size, "grouped hidden width")?;
        self.bind_thread()?;
        self.ensure_kernel(GROUPED_MODULE, GROUPED_KERNEL_SRC, "grouped_gate_f32")?;
        let key_ptr = key.buf.as_ptr();
        let query_ptr = query.buf.as_ptr();
        let value_ptr = value.buf.as_ptr();
        let key_norm_ptr = key_norm.buf.as_ptr();
        let query_norm_ptr = query_norm.buf.as_ptr();
        let gated_ptr = gated.buf.as_ptr();
        let mut params = [
            &key_ptr as *const _ as *mut c_void,
            &query_ptr as *const _ as *mut c_void,
            &value_ptr as *const _ as *mut c_void,
            &key_norm_ptr as *const _ as *mut c_void,
            &query_norm_ptr as *const _ as *mut c_void,
            &gated_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &hc_i as *const _ as *mut c_void,
            &hidden_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(hc_count, "grouped gate branch grid")?;
        let grid_y = checked_u32(tokens, "grouped gate token grid")?;
        self.launch_maybe_blob(
            "grouped_gate_f32",
            [grid_x, grid_y, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(key_ptr);
                blob.push_ptr(query_ptr);
                blob.push_ptr(value_ptr);
                blob.push_ptr(key_norm_ptr);
                blob.push_ptr(query_norm_ptr);
                blob.push_ptr(gated_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(hc_i);
                blob.push_i32(hidden_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }
    /// Variant of [`Self::grouped_gate_f32`] for resident BF16 norm
    /// weights.  The caller keeps these norms native BF16; widening happens in
    /// the kernel and never through a host-side conversion.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_gate_bf16(
        &mut self,
        key: &GpuTensor,
        query: &GpuTensor,
        value: &GpuTensor,
        key_norm: &GpuTensor,
        query_norm: &GpuTensor,
        gated: &GpuTensor,
        tokens: usize,
        hc_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(hc_count, hidden_size, "grouped gated channels")?;
        require_dtype(key, DType::F32, "grouped key projection")?;
        require_dtype(query, DType::F32, "grouped query projection")?;
        require_dtype(value, DType::F32, "grouped value projection")?;
        require_dtype(key_norm, DType::BF16, "grouped BF16 key norm")?;
        require_dtype(query_norm, DType::BF16, "grouped BF16 query norm")?;
        require_dtype(gated, DType::F32, "grouped gated output")?;
        require_numel(
            key,
            checked_product(tokens, channels, "grouped key elements")?,
            "grouped key projection",
        )?;
        require_numel(
            query,
            checked_product(tokens, channels, "grouped query elements")?,
            "grouped query projection",
        )?;
        require_numel(
            value,
            checked_product(tokens, hidden_size, "grouped value elements")?,
            "grouped value projection",
        )?;
        require_numel(key_norm, channels, "grouped BF16 key norm")?;
        require_numel(query_norm, channels, "grouped BF16 query norm")?;
        require_numel(
            gated,
            checked_product(tokens, channels, "grouped gated elements")?,
            "grouped gated output",
        )?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped gate token count")?;
        let hc_i = checked_i32(hc_count, "grouped branch count")?;
        let hidden_i = checked_i32(hidden_size, "grouped hidden width")?;
        self.bind_thread()?;
        self.ensure_kernel(GROUPED_MODULE, GROUPED_KERNEL_SRC, "grouped_gate_bf16")?;
        let key_ptr = key.buf.as_ptr();
        let query_ptr = query.buf.as_ptr();
        let value_ptr = value.buf.as_ptr();
        let key_norm_ptr = key_norm.buf.as_ptr();
        let query_norm_ptr = query_norm.buf.as_ptr();
        let gated_ptr = gated.buf.as_ptr();
        let mut params = [
            &key_ptr as *const _ as *mut c_void,
            &query_ptr as *const _ as *mut c_void,
            &value_ptr as *const _ as *mut c_void,
            &key_norm_ptr as *const _ as *mut c_void,
            &query_norm_ptr as *const _ as *mut c_void,
            &gated_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &hc_i as *const _ as *mut c_void,
            &hidden_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(hc_count, "grouped gate branch grid")?;
        let grid_y = checked_u32(tokens, "grouped gate token grid")?;
        // Key and query rows staged in LDS.
        let lds_bytes = checked_u32(hidden_size * 8, "grouped gate LDS")?;
        if lds_bytes > 64 * 1024 {
            return Err(HipError::new(0, "grouped gate row exceeds LDS"));
        }
        self.launch_maybe_blob(
            "grouped_gate_bf16",
            [grid_x, grid_y, 1],
            [256, 1, 1],
            lds_bytes,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(key_ptr);
                blob.push_ptr(query_ptr);
                blob.push_ptr(value_ptr);
                blob.push_ptr(key_norm_ptr);
                blob.push_ptr(query_norm_ptr);
                blob.push_ptr(gated_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(hc_i);
                blob.push_i32(hidden_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }

    /// Apply grouped `(1 + weight)` RMS normalization to F32 rows.
    pub fn grouped_norm_f32(
        &mut self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        groups: usize,
        group_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(groups, group_size, "grouped normalization channels")?;
        let elements = checked_product(tokens, channels, "grouped normalization elements")?;
        require_dtype(input, DType::F32, "grouped normalization input")?;
        require_dtype(weight, DType::F32, "grouped normalization weight")?;
        require_dtype(output, DType::F32, "grouped normalization output")?;
        require_numel(input, elements, "grouped normalization input")?;
        require_numel(weight, channels, "grouped normalization weight")?;
        require_numel(output, elements, "grouped normalization output")?;
        if elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped normalization token count")?;
        let groups_i = checked_i32(groups, "grouped normalization group count")?;
        let group_size_i = checked_i32(group_size, "grouped normalization group width")?;
        self.bind_thread()?;
        self.ensure_kernel(GROUPED_MODULE, GROUPED_KERNEL_SRC, "grouped_norm_f32")?;
        let input_ptr = input.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params = [
            &input_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &groups_i as *const _ as *mut c_void,
            &group_size_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(groups, "grouped normalization group grid")?;
        let grid_y = checked_u32(tokens, "grouped normalization token grid")?;
        self.launch_maybe_blob(
            "grouped_norm_f32",
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(groups_i);
                blob.push_i32(group_size_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }
    /// Grouped RMS normalization against a resident BF16 norm vector.
    pub fn grouped_norm_bf16(
        &mut self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        groups: usize,
        group_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(groups, group_size, "grouped normalization channels")?;
        let elements = checked_product(tokens, channels, "grouped normalization elements")?;
        require_dtype(input, DType::F32, "grouped normalization input")?;
        require_dtype(weight, DType::BF16, "grouped BF16 normalization weight")?;
        require_dtype(output, DType::F32, "grouped normalization output")?;
        require_numel(input, elements, "grouped normalization input")?;
        require_numel(weight, channels, "grouped BF16 normalization weight")?;
        require_numel(output, elements, "grouped normalization output")?;
        if elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped normalization token count")?;
        let groups_i = checked_i32(groups, "grouped normalization group count")?;
        let group_size_i = checked_i32(group_size, "grouped normalization group width")?;
        self.bind_thread()?;
        self.ensure_kernel(GROUPED_MODULE, GROUPED_KERNEL_SRC, "grouped_norm_bf16")?;
        let input_ptr = input.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params = [
            &input_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &groups_i as *const _ as *mut c_void,
            &group_size_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(groups, "grouped normalization group grid")?;
        let grid_y = checked_u32(tokens, "grouped normalization token grid")?;
        self.launch_maybe_blob(
            "grouped_norm_bf16",
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(groups_i);
                blob.push_i32(group_size_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }

    /// Depthwise causal convolution with SiLU and gated-value residual add.
    ///
    /// `state` is `[history_rows, channels]`, where
    /// `history_rows=(kernel_size-1)*dilation`; the kernel updates it in place
    /// after producing all token outputs.  This state is separate from GDN.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_depthwise_conv_silu_add_f32(
        &mut self,
        gated: &GpuTensor,
        normed: &GpuTensor,
        conv_weight: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> HipResult<()> {
        let elements = checked_product(tokens, channels, "grouped convolution elements")?;
        let history_rows = kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dilation))
            .ok_or_else(|| HipError::new(0, "grouped convolution history size overflow"))?;
        if tokens != 0 && channels != 0 {
            checked_add(tokens, history_rows, "grouped convolution source rows")?;
        }
        let history_elements =
            checked_product(history_rows, channels, "grouped convolution history")?;
        let kernel_elements =
            checked_product(channels, kernel_size, "grouped convolution weights")?;
        require_dtype(gated, DType::F32, "grouped gated values")?;
        require_dtype(normed, DType::F32, "grouped normalized values")?;
        require_dtype(conv_weight, DType::F32, "grouped convolution weights")?;
        require_dtype(state, DType::F32, "grouped convolution state")?;
        require_dtype(output, DType::F32, "grouped convolution output")?;
        require_numel(gated, elements, "grouped gated values")?;
        require_numel(normed, elements, "grouped normalized values")?;
        require_numel(conv_weight, kernel_elements, "grouped convolution weights")?;
        require_numel(state, history_elements, "grouped convolution state")?;
        require_numel(output, elements, "grouped convolution output")?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped convolution token count")?;
        let channels_i = checked_i32(channels, "grouped convolution channel count")?;
        let kernel_i = checked_i32(kernel_size, "grouped convolution kernel size")?;
        let dilation_i = checked_i32(dilation, "grouped convolution dilation")?;
        self.bind_thread()?;
        self.ensure_kernel(
            GROUPED_MODULE,
            GROUPED_KERNEL_SRC,
            "grouped_depthwise_conv_silu_add_f32",
        )?;
        let gated_ptr = gated.buf.as_ptr();
        let normed_ptr = normed.buf.as_ptr();
        let weight_ptr = conv_weight.buf.as_ptr();
        let state_ptr = state.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params = [
            &gated_ptr as *const _ as *mut c_void,
            &normed_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &state_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &channels_i as *const _ as *mut c_void,
            &kernel_i as *const _ as *mut c_void,
            &dilation_i as *const _ as *mut c_void,
        ];
        let grid = checked_grid(channels, BLOCK, "grouped convolution channel grid")?;
        self.launch_maybe_blob(
            "grouped_depthwise_conv_silu_add_f32",
            [grid, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gated_ptr);
                blob.push_ptr(normed_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(state_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(channels_i);
                blob.push_i32(kernel_i);
                blob.push_i32(dilation_i);
                blob
            },
        )
    }
    /// BF16 convolution-weight variant used by the assembled grouped layer.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_depthwise_conv_silu_add_bf16(
        &mut self,
        gated: &GpuTensor,
        normed: &GpuTensor,
        conv_weight: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> HipResult<()> {
        let elements = checked_product(tokens, channels, "grouped convolution elements")?;
        let history_rows = kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dilation))
            .ok_or_else(|| HipError::new(0, "grouped convolution history size overflow"))?;
        if tokens != 0 && channels != 0 {
            checked_add(tokens, history_rows, "grouped convolution source rows")?;
        }
        let history_elements =
            checked_product(history_rows, channels, "grouped convolution history")?;
        let kernel_elements =
            checked_product(channels, kernel_size, "grouped convolution weights")?;
        require_dtype(gated, DType::F32, "grouped gated values")?;
        require_dtype(normed, DType::F32, "grouped normalized values")?;
        require_dtype(conv_weight, DType::BF16, "grouped BF16 convolution weights")?;
        require_dtype(state, DType::F32, "grouped convolution state")?;
        require_dtype(output, DType::F32, "grouped convolution output")?;
        require_numel(gated, elements, "grouped gated values")?;
        require_numel(normed, elements, "grouped normalized values")?;
        require_numel(
            conv_weight,
            kernel_elements,
            "grouped BF16 convolution weights",
        )?;
        require_numel(state, history_elements, "grouped convolution state")?;
        require_numel(output, elements, "grouped convolution output")?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "grouped convolution token count")?;
        let channels_i = checked_i32(channels, "grouped convolution channel count")?;
        let kernel_i = checked_i32(kernel_size, "grouped convolution kernel size")?;
        let dilation_i = checked_i32(dilation, "grouped convolution dilation")?;
        // Tokens run in parallel chunks; a chunk of at least `history_rows`
        // keeps every `state` read in chunk 0.
        let chunk = history_rows.max(32);
        let chunk_i = checked_i32(chunk, "grouped convolution token chunk")?;
        let chunk_grid = checked_u32(tokens.div_ceil(chunk), "grouped convolution chunk grid")?;
        self.bind_thread()?;
        self.ensure_kernel(
            GROUPED_MODULE,
            GROUPED_KERNEL_SRC,
            "grouped_depthwise_conv_silu_add_bf16",
        )?;
        let gated_ptr = gated.buf.as_ptr();
        let normed_ptr = normed.buf.as_ptr();
        let weight_ptr = conv_weight.buf.as_ptr();
        let state_ptr = state.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params = [
            &gated_ptr as *const _ as *mut c_void,
            &normed_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &state_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &channels_i as *const _ as *mut c_void,
            &kernel_i as *const _ as *mut c_void,
            &dilation_i as *const _ as *mut c_void,
            &chunk_i as *const _ as *mut c_void,
        ];
        let grid = checked_grid(channels, BLOCK, "grouped convolution channel grid")?;
        self.launch_maybe_blob(
            "grouped_depthwise_conv_silu_add_bf16",
            [grid, chunk_grid, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gated_ptr);
                blob.push_ptr(normed_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(state_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(channels_i);
                blob.push_i32(kernel_i);
                blob.push_i32(dilation_i);
                blob.push_i32(chunk_i);
                blob
            },
        )
    }
}

fn require_dtype(tensor: &GpuTensor, expected: DType, what: &'static str) -> HipResult<()> {
    if tensor.dtype == expected {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            &format!(
                "{what} must have dtype {expected:?}, got {:?}",
                tensor.dtype
            ),
        ))
    }
}

fn require_numel(tensor: &GpuTensor, expected: usize, what: &'static str) -> HipResult<()> {
    if tensor.numel() == expected {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            &format!(
                "{what} must have {expected} elements, got {}",
                tensor.numel()
            ),
        ))
    }
}

const MAX_KERNEL_EXTENT: usize = i32::MAX as usize;

fn checked_product(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    if value <= MAX_KERNEL_EXTENT {
        Ok(value)
    } else {
        Err(HipError::new(0, &format!("{what} exceeds i32")))
    }
}

fn checked_add(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_add(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    if value <= MAX_KERNEL_EXTENT {
        Ok(value)
    } else {
        Err(HipError::new(0, &format!("{what} exceeds i32")))
    }
}

fn checked_i32(value: usize, what: &'static str) -> HipResult<i32> {
    i32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds i32")))
}

fn checked_u32(value: usize, what: &'static str) -> HipResult<u32> {
    u32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds u32")))
}

fn checked_grid(elements: usize, block: u32, what: &'static str) -> HipResult<u32> {
    let elements = checked_product(elements, 1, what)?;
    let grid = elements.div_ceil(block as usize);
    checked_u32(grid, what)
}

/// Operation-neutral grouped gate contract used by dispatch layer lowering.
pub struct GroupedGate<'a> {
    pub key: &'a GpuTensor,
    pub query: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub norm_key: &'a GpuTensor,
    pub norm_query: &'a GpuTensor,
    pub gated: &'a GpuTensor,
    pub rows: usize,
    pub groups: usize,
    pub group_size: usize,
    pub epsilon: f32,
}

pub fn grouped_gate_bf16(gpu: &mut Gpu, p: &GroupedGate<'_>) -> HipResult<()> {
    gpu.grouped_gate_bf16(
        p.key,
        p.query,
        p.value,
        p.norm_key,
        p.norm_query,
        p.gated,
        p.rows,
        p.groups,
        p.group_size,
        p.epsilon,
    )
}

/// Operation-neutral grouped normalization contract.
pub struct GroupedNorm<'a> {
    pub input: &'a GpuTensor,
    pub weight: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub tokens: usize,
    pub groups: usize,
    pub group_size: usize,
    pub epsilon: f32,
}

pub fn grouped_norm_bf16(gpu: &mut Gpu, p: &GroupedNorm<'_>) -> HipResult<()> {
    gpu.grouped_norm_bf16(
        p.input,
        p.weight,
        p.output,
        p.tokens,
        p.groups,
        p.group_size,
        p.epsilon,
    )
}

/// Operation-neutral grouped depthwise causal convolution contract.
pub struct GroupedDepthwiseConv<'a> {
    pub gated: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    pub conv_weight: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub tokens: usize,
    pub channels: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

pub fn grouped_depthwise_conv_silu_add_bf16(
    gpu: &mut Gpu,
    p: &GroupedDepthwiseConv<'_>,
) -> HipResult<()> {
    gpu.grouped_depthwise_conv_silu_add_bf16(
        p.gated,
        p.normed,
        p.conv_weight,
        p.state,
        p.output,
        p.tokens,
        p.channels,
        p.kernel_size,
        p.dilation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok()
    }

    fn null_tensor(shape: &[usize], dtype: DType) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.shape = shape.to_vec();
        tensor.dtype = dtype;
        tensor
    }

    #[test]
    fn checked_grouped_extents_reject_signed_overflow() {
        let max = i32::MAX as usize;
        assert_eq!(checked_product(max, 1, "boundary").unwrap(), max);
        assert!(checked_product(max, 2, "boundary").is_err());
        assert!(checked_add(max, 1, "boundary").is_err());
        assert!(checked_grid(max, BLOCK, "boundary").is_ok());
        assert!(checked_grid(max + 1, BLOCK, "boundary").is_err());
    }

    #[test]
    fn grouped_launchers_reject_flattened_extent_overflow() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let max = i32::MAX as usize;

        let input = null_tensor(&[0], DType::F32);
        let weight = null_tensor(&[0], DType::F32);
        let output = null_tensor(&[0], DType::F32);
        let error = gpu
            .grouped_linear_f32(&input, &weight, &output, 2, max, 1)
            .expect_err("linear flattened input extent must be bounded");
        assert!(error.to_string().contains("overflow") || error.to_string().contains("i32"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let staged = null_tensor(&[0], DType::BF16);
        let output = null_tensor(&[0], DType::F32);
        let error = gpu
            .grouped_gather_convert_bf16(&staged, &output, 2, 1, max)
            .expect_err("gather flattened extent must be bounded");
        assert!(error.to_string().contains("overflow") || error.to_string().contains("i32"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let key = null_tensor(&[0], DType::F32);
        let query = null_tensor(&[0], DType::F32);
        let value = null_tensor(&[0], DType::F32);
        let norm_key = null_tensor(&[0], DType::F32);
        let norm_query = null_tensor(&[0], DType::F32);
        let gated = null_tensor(&[0], DType::F32);
        let error = gpu
            .grouped_gate_f32(
                &key,
                &query,
                &value,
                &norm_key,
                &norm_query,
                &gated,
                2,
                max,
                2,
                1.0e-6,
            )
            .expect_err("grouped gate flattened extent must be bounded");
        assert!(error.to_string().contains("overflow") || error.to_string().contains("i32"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let input = null_tensor(&[0], DType::F32);
        let weight = null_tensor(&[0], DType::F32);
        let output = null_tensor(&[0], DType::F32);
        let error = gpu
            .grouped_norm_f32(&input, &weight, &output, 2, max, 2, 1.0e-6)
            .expect_err("grouped norm flattened extent must be bounded");
        assert!(error.to_string().contains("overflow") || error.to_string().contains("i32"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let gated = null_tensor(&[0], DType::F32);
        let normed = null_tensor(&[0], DType::F32);
        let conv_weight = null_tensor(&[0], DType::F32);
        let state = null_tensor(&[0], DType::F32);
        let output = null_tensor(&[0], DType::F32);
        let error = gpu
            .grouped_depthwise_conv_silu_add_f32(
                &gated,
                &normed,
                &conv_weight,
                &state,
                &output,
                2,
                max,
                2,
                1,
            )
            .expect_err("grouped convolution flattened extent must be bounded");
        assert!(error.to_string().contains("overflow") || error.to_string().contains("i32"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn grouped_gate_bf16_is_bit_identical_to_serial_f32_gate() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        const TOKENS: usize = 7;
        const BRANCHES: usize = 4;
        const HIDDEN: usize = 2560;
        const CHANNELS: usize = BRANCHES * HIDDEN;
        let wave = |n: usize, a: usize, b: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * a % b) as f32 - b as f32 / 2.0) / b as f32)
                .collect()
        };
        let key = gpu
            .upload_f32(&wave(TOKENS * CHANNELS, 37, 251), &[TOKENS * CHANNELS])
            .unwrap();
        let query = gpu
            .upload_f32(&wave(TOKENS * CHANNELS, 53, 241), &[TOKENS * CHANNELS])
            .unwrap();
        let value = gpu
            .upload_f32(&wave(TOKENS * HIDDEN, 29, 233), &[TOKENS * HIDDEN])
            .unwrap();
        // BF16 norms and their exact F32 widening for the serial reference.
        let norm_bits = |a: usize| -> Vec<u16> {
            wave(CHANNELS, a, 199)
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect()
        };
        let (key_bits, query_bits) = (norm_bits(17), norm_bits(23));
        let widen = |bits: &[u16]| -> Vec<f32> {
            bits.iter()
                .map(|b| f32::from_bits((*b as u32) << 16))
                .collect()
        };
        let upload_bf16 = |gpu: &mut Gpu, bits: &[u16]| {
            let tensor = gpu.zeros(&[CHANNELS], DType::BF16).unwrap();
            let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&tensor.buf, &bytes).unwrap();
            tensor
        };
        let key_norm = upload_bf16(&mut gpu, &key_bits);
        let query_norm = upload_bf16(&mut gpu, &query_bits);
        let key_norm_f32 = gpu.upload_f32(&widen(&key_bits), &[CHANNELS]).unwrap();
        let query_norm_f32 = gpu.upload_f32(&widen(&query_bits), &[CHANNELS]).unwrap();
        let got = gpu.zeros(&[TOKENS * CHANNELS], DType::F32).unwrap();
        let want = gpu.zeros(&[TOKENS * CHANNELS], DType::F32).unwrap();
        gpu.grouped_gate_bf16(
            &key,
            &query,
            &value,
            &key_norm,
            &query_norm,
            &got,
            TOKENS,
            BRANCHES,
            HIDDEN,
            1.0e-6,
        )
        .unwrap();
        gpu.grouped_gate_f32(
            &key,
            &query,
            &value,
            &key_norm_f32,
            &query_norm_f32,
            &want,
            TOKENS,
            BRANCHES,
            HIDDEN,
            1.0e-6,
        )
        .unwrap();
        let got = gpu.download_f32(&got).unwrap();
        let want = gpu.download_f32(&want).unwrap();
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "index {i}");
        }
    }

    /// The BF16 convolution runs tokens in parallel chunks; outputs and the
    /// rewritten history must still be bitwise the serial F32 kernel's, for a
    /// prompt shorter than the history and one spanning several chunks.
    #[test]
    fn grouped_depthwise_bf16_chunks_match_serial_f32() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        const CHANNELS: usize = 300;
        const KERNEL: usize = 4;
        const DILATION: usize = 2;
        const HISTORY: usize = (KERNEL - 1) * DILATION;
        let wave = |n: usize, a: usize, b: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * a % b) as f32 - b as f32 / 2.0) / b as f32)
                .collect()
        };
        let weight_bits: Vec<u16> = wave(CHANNELS * KERNEL, 19, 197)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let weight_f32: Vec<f32> = weight_bits
            .iter()
            .map(|b| f32::from_bits((*b as u32) << 16))
            .collect();
        let weight = gpu.zeros(&[CHANNELS * KERNEL], DType::BF16).unwrap();
        let bytes: Vec<u8> = weight_bits.iter().flat_map(|b| b.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&weight.buf, &bytes).unwrap();
        let weight_wide = gpu.upload_f32(&weight_f32, &[CHANNELS * KERNEL]).unwrap();
        for tokens in [3usize, 70] {
            let gated = gpu
                .upload_f32(&wave(tokens * CHANNELS, 37, 251), &[tokens * CHANNELS])
                .unwrap();
            let normed = gpu
                .upload_f32(&wave(tokens * CHANNELS, 53, 241), &[tokens * CHANNELS])
                .unwrap();
            let history = wave(HISTORY * CHANNELS, 29, 233);
            let state_got = gpu.upload_f32(&history, &[HISTORY * CHANNELS]).unwrap();
            let state_want = gpu.upload_f32(&history, &[HISTORY * CHANNELS]).unwrap();
            let got = gpu.zeros(&[tokens * CHANNELS], DType::F32).unwrap();
            let want = gpu.zeros(&[tokens * CHANNELS], DType::F32).unwrap();
            gpu.grouped_depthwise_conv_silu_add_bf16(
                &gated, &normed, &weight, &state_got, &got, tokens, CHANNELS, KERNEL, DILATION,
            )
            .unwrap();
            gpu.grouped_depthwise_conv_silu_add_f32(
                &gated,
                &normed,
                &weight_wide,
                &state_want,
                &want,
                tokens,
                CHANNELS,
                KERNEL,
                DILATION,
            )
            .unwrap();
            for (what, a, b) in [("output", &got, &want), ("state", &state_got, &state_want)] {
                let a = gpu.download_f32(a).unwrap();
                let b = gpu.download_f32(b).unwrap();
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "{tokens} tokens {what} index {i}");
                }
            }
        }
    }
}
