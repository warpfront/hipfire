// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU dispatch for the fused FLUX MMDiT elementwise kernels
//! (`kernels/src/layernorm_modulate_f32.hip`,
//! `kernels/src/qk_rmsnorm_rope_flux.hip`).
//!
//! The MMDiT forward's per-stream elementwise work arrives as a chain of
//! single-purpose launches over the same activation — LayerNorm, then
//! modulate, then a cast; RMSNorm, then RoPE, then a cast — each of which
//! reads and rewrites the whole tensor for one pass of arithmetic. These two
//! launchers collapse those chains into one launch each, and let the kernel
//! emit the dtype the next kernel wants so the `cast_f32_to_f16` disappears
//! with them.
//!
//! Both live here rather than in `norm.rs` because they are FLUX-shaped, not
//! general norm ops: the modulation affine, the axial position split, and the
//! text/image row asymmetry are all specific to the MMDiT block.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::HipResult;

/// Head dim ceiling of `qk_rmsnorm_rope_flux`, set by the kernel's
/// `MAX_PAIRS_PER_LANE` register buffer (32 lanes x 4 pairs x 2 values).
/// Real FLUX uses 128.
const QK_MAX_HEAD_DIM: usize = 256;

/// Waves per workgroup in `qk_rmsnorm_rope_flux` — one wave handles one
/// (row, head) unit, so this is also units per workgroup.
const QK_WAVES_PER_BLOCK: usize = 8;

/// QK-RMSNorm epsilon, matching the `rmsnorm_batched` call it replaces.
const QK_EPS: f32 = 1e-6;

/// Waves (= output rows) per workgroup in `gemv_f16_bias_xf32`. MUST match
/// the kernel's `GEMV_MB_WAVES` (pinned by `gemv_bias_waves_matches_kernel`).
const GEMV_BIAS_WAVES: usize = 4;

/// Is `gemv_f16_bias_xf32`'s weight pointer legal for the K it will be given?
///
/// The kernel takes its 8-wide (16 B) `gemv_half8` load path whenever
/// `K % 8 == 0`, which needs the weight base 16 B-aligned — every row is then
/// aligned too, since the row stride is `K * 2` bytes. A ragged K takes the
/// scalar path, where `_Float16`'s natural 2 B alignment is all that is
/// required, so alignment is irrelevant there.
///
/// Whole pool tensors are ≥256 B-aligned, so no caller can trip this today;
/// a `sub_offset` weight VIEW at an odd element offset could. The launcher
/// rejects that loudly rather than silently dropping to the ~4x slower scalar
/// loop, because an invisible perf cliff is the worse failure mode.
fn gemv_bias_weight_aligned(weight_addr: usize, k: usize) -> bool {
    !k.is_multiple_of(8) || weight_addr.is_multiple_of(16)
}

impl Gpu {
    /// Weightless LayerNorm fused with the FLUX adaLN-Zero modulation affine:
    ///
    /// ```text
    /// out[r,i] = (x[r,i] - mean(x[r])) * rsqrt(var(x[r]) + eps) * (1 + scale[i]) + shift[i]
    /// ```
    ///
    /// `x` is `[n_rows, d]` F32 and `shift`/`scale` are `[d]` F32 row vectors
    /// broadcast over every row, exactly as [`Gpu::modulate_f32`] takes them.
    /// `out` is `[n_rows, d]` and picks the kernel by its dtype: F32 or F16
    /// (round-to-nearest-even, the same conversion [`Gpu::cast_f32_to_f16`]
    /// applies), so a caller feeding an F16 GEMM needs no separate cast.
    ///
    /// The F32 output is BIT-IDENTICAL to [`Gpu::layernorm_batched`] with
    /// gamma = 1 / beta = 0 followed by [`Gpu::modulate_f32`], and the F16
    /// output is bit-identical to that chain plus the cast. The block size is
    /// derived the same way `layernorm_batched` derives it because the
    /// reduction tree depends on it; see the kernel source for the rest.
    ///
    /// `out` MUST NOT alias `x` (the kernel marks both `__restrict__`), and
    /// neither may alias `shift`/`scale`. The FLUX caller always has a
    /// distinct destination anyway — that is the point of the fused cast.
    pub fn layernorm_modulate(
        &mut self,
        x: &GpuTensor,
        shift: &GpuTensor,
        scale: &GpuTensor,
        out: &GpuTensor,
        n_rows: usize,
        d: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if n_rows == 0 || d == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "layernorm_modulate: dims must be > 0",
            ));
        }
        for (name, t) in [("x", x), ("shift", shift), ("scale", scale)] {
            if t.dtype != DType::F32 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "layernorm_modulate: {name} dtype must be F32 (got {:?})",
                        t.dtype
                    ),
                ));
            }
        }
        let kernel = match out.dtype {
            DType::F32 => "layernorm_modulate_f32",
            DType::F16 => "layernorm_modulate_f16",
            other => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("layernorm_modulate: out dtype must be F32 or F16 (got {other:?})"),
                ));
            }
        };
        let f32_sz = DType::F32.size();
        let elems = n_rows.checked_mul(d).unwrap();
        for (name, t, need) in [
            ("x", x, elems * f32_sz),
            ("out", out, elems * out.dtype.size()),
            ("shift", shift, d * f32_sz),
            ("scale", scale, d * f32_sz),
        ] {
            if t.buf.size() < need {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "layernorm_modulate: {name} buffer too small (have {} need {need})",
                        t.buf.size()
                    ),
                ));
            }
        }

        self.ensure_kernel(
            "layernorm_modulate",
            crate::kernels::LAYERNORM_MODULATE_F32_SRC,
            kernel,
        )?;

        let x_ptr = x.buf.as_ptr();
        let shift_ptr = shift.buf.as_ptr();
        let scale_ptr = scale.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let d_i = d as i32;
        let eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &shift_ptr as *const _ as *mut c_void,
            &scale_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &d_i as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        // Same derivation as `Gpu::layernorm_batched`: the LDS reduction tree
        // is shaped by the block size, so changing it would change the
        // rounding and break bit-identity.
        let block = (256u32.min(d as u32)).next_power_of_two();
        let shared_mem = block * 4;

        let bytes = elems * f32_sz + elems * out.dtype.size();
        let timer = crate::profile::begin_timer(&self.hip, "layernorm_modulate", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [n_rows as u32, 1, 1],
            [block, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(shift_ptr);
                blob.push_ptr(scale_ptr);
                blob.push_ptr(out_ptr);
                blob.push_i32(d_i);
                blob.push_f32(eps_v);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// FLUX Q/K head prep: per-`(row, head)` RMSNorm over `head_dim` with the
    /// shared `[head_dim]` `scale`, fused with the 2D axial RoPE.
    ///
    /// `x` and `out` are `[n_txt + n_img, heads * head_dim]` row-major, each
    /// F32 or F16 independently (all four combinations exist), and `scale` is
    /// `[head_dim]` F32. The first `n_txt` rows are text and get the norm
    /// only; the remaining `n_img` rows are image tokens at position
    /// `(0, t / grid_w, t % grid_w, 0)` — unless `ids` supplies an F32
    /// `[n_img, 4]` table of per-image-row positions — and get norm +
    /// rotation, matching [`Gpu::rope_2d_flux_f32`] called with `row_offset =
    /// n_txt`. `axes_dim` must be even and sum to `head_dim` ([16, 56, 56, 0]
    /// for real FLUX.1 dev/schnell). Epsilon is fixed at 1e-6, the value the
    /// `rmsnorm_batched` call it replaces uses.
    ///
    /// Not bit-identical to that pair — the RMS reduction is a per-wave
    /// butterfly where `rmsnorm_f32` is a 128-thread LDS tree — but the
    /// rotation is, and measured f32 agreement is well inside 1e-6 relative.
    /// `out` MUST NOT alias `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_rmsnorm_rope_flux(
        &mut self,
        x: &GpuTensor,
        scale: &GpuTensor,
        out: &GpuTensor,
        n_txt: usize,
        n_img: usize,
        heads: usize,
        head_dim: usize,
        grid_w: usize,
        axes_dim: [usize; 4],
        theta: f64,
        ids: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if heads == 0 || head_dim == 0 || grid_w == 0 || n_txt + n_img == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "qk_rmsnorm_rope_flux: dims must be > 0",
            ));
        }
        if head_dim % 2 != 0 || head_dim > QK_MAX_HEAD_DIM {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "qk_rmsnorm_rope_flux: head_dim must be even and <= {QK_MAX_HEAD_DIM} (got {head_dim})"
                ),
            ));
        }
        if scale.dtype != DType::F32 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "qk_rmsnorm_rope_flux: scale dtype must be F32 (got {:?})",
                    scale.dtype
                ),
            ));
        }
        let sum_ax: usize = axes_dim.iter().sum();
        if sum_ax != head_dim {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "qk_rmsnorm_rope_flux: axes_dim {axes_dim:?} must sum to head_dim {head_dim}"
                ),
            ));
        }
        for a in axes_dim {
            if a % 2 != 0 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("qk_rmsnorm_rope_flux: axes_dim must be even (got {a})"),
                ));
            }
        }
        if let Some(t) = ids {
            if t.dtype != DType::F32 || t.numel() < n_img * 4 {
                return Err(hip_bridge::HipError::new(
                    0,
                    "qk_rmsnorm_rope_flux: ids must be F32 [n_img, 4]",
                ));
            }
        }
        let kernel = match (x.dtype, out.dtype) {
            (DType::F32, DType::F32) => "qk_rmsnorm_rope_flux_f32_f32",
            (DType::F32, DType::F16) => "qk_rmsnorm_rope_flux_f32_f16",
            (DType::F16, DType::F32) => "qk_rmsnorm_rope_flux_f16_f32",
            (DType::F16, DType::F16) => "qk_rmsnorm_rope_flux_f16_f16",
            (xd, od) => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "qk_rmsnorm_rope_flux: x/out dtypes must each be F32 or F16 (got {xd:?}/{od:?})"
                    ),
                ));
            }
        };
        let n_all = n_txt + n_img;
        let elems = n_all
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(head_dim))
            .unwrap();
        for (name, t, need) in [
            ("x", x, elems * x.dtype.size()),
            ("out", out, elems * out.dtype.size()),
            ("scale", scale, head_dim * DType::F32.size()),
        ] {
            if t.buf.size() < need {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "qk_rmsnorm_rope_flux: {name} buffer too small (have {} need {need})",
                        t.buf.size()
                    ),
                ));
            }
        }

        self.ensure_kernel(
            "qk_rmsnorm_rope_flux",
            crate::kernels::QK_RMSNORM_ROPE_FLUX_SRC,
            kernel,
        )?;

        let x_ptr = x.buf.as_ptr();
        let ids_ptr = ids.map_or(std::ptr::null_mut(), |t| t.buf.as_ptr());
        let scale_ptr = scale.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_txt_i = n_txt as i32;
        let n_img_i = n_img as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let grid_w_i = grid_w as i32;
        let ax0_i = axes_dim[0] as i32;
        let ax1_i = axes_dim[1] as i32;
        let ax2_i = axes_dim[2] as i32;
        let ax3_i = axes_dim[3] as i32;
        let eps_v = QK_EPS;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &ids_ptr as *const _ as *mut c_void,
            &scale_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_txt_i as *const _ as *mut c_void,
            &n_img_i as *const _ as *mut c_void,
            &heads_i as *const _ as *mut c_void,
            &hd_i as *const _ as *mut c_void,
            &grid_w_i as *const _ as *mut c_void,
            &ax0_i as *const _ as *mut c_void,
            &ax1_i as *const _ as *mut c_void,
            &ax2_i as *const _ as *mut c_void,
            &ax3_i as *const _ as *mut c_void,
            &theta as *const _ as *mut c_void,
            &eps_v as *const _ as *mut c_void,
        ];

        // One wave per (row, head) unit, QK_WAVES_PER_BLOCK waves per block.
        let units = n_all * heads;
        let grid = units.div_ceil(QK_WAVES_PER_BLOCK) as u32;
        let block = (QK_WAVES_PER_BLOCK * 32) as u32;

        let bytes = elems * x.dtype.size() + elems * out.dtype.size();
        let timer = crate::profile::begin_timer(&self.hip, "qk_rmsnorm_rope_flux", kernel, bytes);
        let result =
            self.launch_maybe_blob(kernel, [grid, 1, 1], [block, 1, 1], 0, &mut params, || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(ids_ptr);
                blob.push_ptr(scale_ptr);
                blob.push_ptr(out_ptr);
                blob.push_i32(n_txt_i);
                blob.push_i32(n_img_i);
                blob.push_i32(heads_i);
                blob.push_i32(hd_i);
                blob.push_i32(grid_w_i);
                blob.push_i32(ax0_i);
                blob.push_i32(ax1_i);
                blob.push_i32(ax2_i);
                blob.push_i32(ax3_i);
                blob.push_u64(theta.to_bits()); // double is 64-bit, no push_f64
                blob.push_f32(eps_v);
                blob
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Row-major F16-weight × F32-input GEMV with the F32 bias fused into the
    /// store: `y[m] = bias[m] + Σ_k weight[m, k] · x[k]`.
    ///
    /// `weight` is `[m, k]` F16 (the layout every FLUX `.weight` is uploaded
    /// in), `x` is `[k]` F32, `bias` is `[m]` F32 or `None`, and `y` is `[m]`
    /// F32. `y` may be a `sub_offset` view, which is how the MMDiT forward
    /// writes all 57 blocks' modulation vectors into one buffer.
    ///
    /// This is the batch-1 replacement for routing a modulation linear
    /// through `gemm_f16_x_f16_wmma_lds_auto`: the 128-row macro-tile streams
    /// the same weight bytes but computes 127 padding rows, and it casts the
    /// activation to F16 first. The GEMV keeps the activation in F32, so it is
    /// NOT bit-identical to the GEMM — the K reduction order differs too (the
    /// GEMV sums a lane-strided partial then a shuffle tree; the WMMA tile
    /// sums 16-wide K chunks in hardware order).
    pub fn gemv_f16_bias_xf32(
        &mut self,
        weight: &GpuTensor,
        x: &GpuTensor,
        bias: Option<&GpuTensor>,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if m == 0 || k == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "gemv_f16_bias_xf32: dims must be > 0",
            ));
        }
        if weight.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemv_f16_bias_xf32: weight must be F16 (got {:?})",
                    weight.dtype
                ),
            ));
        }
        for (name, t) in [("x", x), ("y", y)]
            .into_iter()
            .chain(bias.map(|b| ("bias", b)))
        {
            if t.dtype != DType::F32 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("gemv_f16_bias_xf32: {name} must be F32 (got {:?})", t.dtype),
                ));
            }
        }
        for (name, have, need) in [
            ("weight", weight.buf.size(), m * k * DType::F16.size()),
            ("x", x.buf.size(), k * DType::F32.size()),
            ("y", y.buf.size(), m * DType::F32.size()),
        ]
        .into_iter()
        .chain(bias.map(|b| ("bias", b.buf.size(), m * DType::F32.size())))
        {
            if have < need {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "gemv_f16_bias_xf32: {name} buffer too small (have {have} need {need})"
                    ),
                ));
            }
        }

        // The 8-wide weight load path is chosen by the kernel from K alone, so
        // the base alignment it assumes has to be checked here — see
        // [`gemv_bias_weight_aligned`]. `x` is read one f32 at a time, so it
        // needs no alignment beyond its own dtype.
        if !gemv_bias_weight_aligned(weight.buf.as_ptr() as usize, k) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemv_f16_bias_xf32: weight base {:p} is not 16-byte aligned, \
                     required for the 8-wide load path taken at k={k} (k % 8 == 0). \
                     Pass a whole tensor, not a sub_offset view at an odd element offset.",
                    weight.buf.as_ptr()
                ),
            ));
        }

        self.ensure_kernel(
            "gemv_f16_bias_xf32",
            crate::kernels::GEMV_F16_BIAS_XF32_SRC,
            "gemv_f16_bias_xf32",
        )?;

        let w_ptr = weight.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let b_ptr = bias.map_or(std::ptr::null_mut(), |b| b.buf.as_ptr());
        let y_ptr = y.buf.as_ptr();
        let m_i = m as i32;
        let k_i = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];

        let grid = m.div_ceil(GEMV_BIAS_WAVES) as u32;
        let block = (GEMV_BIAS_WAVES * 32) as u32;
        let bytes = m * k * DType::F16.size();
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv_f16_bias_xf32",
            "gemv_f16_bias_xf32",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "gemv_f16_bias_xf32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(w_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(b_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(m_i);
                blob.push_i32(k_i);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel picks its 8-wide load path from K, so the alignment rule is
    /// conditional on K — a ragged K is legal at any alignment, and only a
    /// K that is a multiple of 8 constrains the base pointer.
    #[test]
    fn gemv_bias_weight_alignment_rule_is_conditional_on_k() {
        // k % 8 == 0: the vector path, so the base must be 16 B-aligned.
        assert!(gemv_bias_weight_aligned(0x1000, 3072));
        assert!(gemv_bias_weight_aligned(0x1010, 3072));
        assert!(!gemv_bias_weight_aligned(0x1002, 3072));
        assert!(!gemv_bias_weight_aligned(0x1008, 3072));
        // A sub_offset view one f16 element into a 256 B-aligned tensor is
        // exactly the case the launcher has to reject.
        assert!(!gemv_bias_weight_aligned(0x1000 + 2, 3072));
        // ...eight elements in is 16 B on again, and legal.
        assert!(gemv_bias_weight_aligned(0x1000 + 16, 3072));

        // k % 8 != 0: the scalar path, alignment is irrelevant.
        assert!(gemv_bias_weight_aligned(0x1002, 3070));
        assert!(gemv_bias_weight_aligned(0x1001, 17));

        // Whole pool tensors are >= 256 B aligned, which is why no caller
        // trips the check today.
        assert!(gemv_bias_weight_aligned(0x2_0000, 3072));
    }

    /// `GEMV_BIAS_WAVES` sets the launch geometry from Rust; the kernel's
    /// `GEMV_MB_WAVES` sets which row each wave owns. If they drift, every
    /// launch silently computes the wrong rows.
    #[test]
    fn gemv_bias_waves_matches_kernel() {
        assert!(
            crate::kernels::GEMV_F16_BIAS_XF32_SRC
                .contains(&format!("#define GEMV_MB_WAVES {GEMV_BIAS_WAVES}")),
            "kernel GEMV_MB_WAVES must equal GEMV_BIAS_WAVES ({GEMV_BIAS_WAVES})"
        );
    }

    /// FLUX.2 Klein adds a 4th RoPE axis and an optional per-image-row id
    /// table (plan `2026-09-04-flux2-klein` Task 1). All three RoPE kernels
    /// must carry both: the `ax3` axis-length parameter and the nullable
    /// `ids` position table.
    #[test]
    fn rope_kernels_take_four_axes_and_an_id_table() {
        for (name, src) in [
            ("rope_2d_flux_f32", crate::kernels::ROPE_2D_FLUX_F32_SRC),
            (
                "rope_2d_flux_f32_fast",
                crate::kernels::ROPE_2D_FLUX_F32_FAST_SRC,
            ),
            (
                "qk_rmsnorm_rope_flux",
                crate::kernels::QK_RMSNORM_ROPE_FLUX_SRC,
            ),
        ] {
            assert!(src.contains("int ax3"), "{name}: missing ax3 parameter");
            assert!(
                src.contains("const float* __restrict__ ids"),
                "{name}: missing ids table"
            );
        }
    }
}
