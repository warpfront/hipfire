// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Slot-boundary MoE ops that run from certified builder (PeaceMaker) modules.
//!
//! Today: the grouped BF16-input down combine of the symmetric IU4 expert route
//! ([`Gpu::moe_down_combine_grouped_bf16in_pm`]). The grouped down GEMM of that
//! route writes BF16 rows in grouped (expert-sorted) order; this kernel gathers
//! each token's `k_top` rows through the inverse permutation, widens BF16 to F32,
//! accumulates `w * v` in ascending top-k slot order with one fused multiply-add
//! per slot, and adds the sum to the F32 residual stream in place. No atomics.
//!
//! Kernel semantics, per `(t = workgroup_id.x, m = workgroup_id.y * 256 + tid)`:
//! return if `m >= dim` or `t >= n_tokens`; `acc = +0.0f32`; for `k` in
//! `0..k_top` ascending: `slot = inverse_perm[t * k_top + k]`, skip when
//! `slot < 0` or `slot >= grouped_rows` (unsigned compare);
//! `w = topk_weights[t * k_top + k]`;
//! `v = f32::from_bits((y[slot * dim + m] as u32) << 16)`;
//! `acc = fma(w, v, acc)`. Then
//! `x_residual[t * dim + m] = RN(x_residual[t * dim + m] + acc)`.
//!
//! v2 ([`Gpu::moe_down_combine_grouped_bf16in_v2_pm`], separate bundle
//! `kernels/moe_slot_ops_v2_pm_gfx1201.hxaco`, symbol
//! `moe_down_combine_grouped_bf16in_v2_pm_gfx1201`) has the same operands and
//! per-column arithmetic, but each of the 256 lanes owns eight contiguous
//! columns (one 16-byte BF16 group per slot), so the slot index and weight are
//! uniform per wave and reused across the eight columns. Grid
//! `[n_tokens, dim / 2048, 1]`; `dim` a positive multiple of 2048. It is
//! selected only by `HIPFIRE_MOE_COMBINE_V2` (`PrefillRoute::MoeCombineV2`).

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use crate::moe::require_span;
use hip_bridge::HipResult;

/// Module key and device symbol of the grouped BF16-input down combine twin
/// (`peacemaker-isa emit --kernel moe_down_combine_grouped_bf16in_pm --arch gfx1201`).
pub const MOE_DOWN_COMBINE_BF16IN_PM_GFX1201_SYMBOL: &str =
    "moe_down_combine_grouped_bf16in_pm_gfx1201";
/// Threads per workgroup (v1: one output column per thread; v2: eight
/// contiguous columns per thread).
pub const MOE_DOWN_COMBINE_BF16IN_BLOCK: u32 = 256;
/// Largest `k_top` the kernel's fixed-order accumulation is certified for.
pub const MOE_DOWN_COMBINE_BF16IN_MAX_K_TOP: usize = 8;
/// Module key and device symbol of the grouped BF16-input down combine v2
/// twin (bundle `kernels/moe_slot_ops_v2_pm_gfx1201.hxaco`).
pub const MOE_DOWN_COMBINE_BF16IN_V2_PM_GFX1201_SYMBOL: &str =
    "moe_down_combine_grouped_bf16in_v2_pm_gfx1201";
/// Output columns of one v2 workgroup: 256 lanes x eight contiguous columns.
pub const MOE_DOWN_COMBINE_V2_COLS_PER_BLOCK: usize = 2048;
/// HIP's grid y extent limit.
const GRID_Y_MAX: usize = 65_535;

/// Embedded image and device symbol (also the module key) of the slot-ops
/// module for `arch`; `None` where no builder object exists.
fn moe_slot_ops_kernel(arch: &str) -> Option<(&'static [u8], &'static str)> {
    match arch {
        "gfx1201" => Some((
            kernels::MOE_SLOT_OPS_PM_GFX1201,
            MOE_DOWN_COMBINE_BF16IN_PM_GFX1201_SYMBOL,
        )),
        _ => None,
    }
}

/// Embedded image and device symbol (also the module key) of the v2 combine
/// bundle for `arch`; `None` where no builder object exists. The anchor of
/// `PrefillRoute::MoeCombineV2`'s raw inventory.
pub(crate) fn moe_slot_ops_v2_kernel(arch: &str) -> Option<(&'static [u8], &'static str)> {
    match arch {
        "gfx1201" => Some((
            kernels::MOE_SLOT_OPS_V2_PM_GFX1201,
            MOE_DOWN_COMBINE_BF16IN_V2_PM_GFX1201_SYMBOL,
        )),
        _ => None,
    }
}

/// Byte spans and grid of one validated combine launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MoeDownCombineGeometry {
    /// `grouped_rows * dim * 2`: BF16 grouped rows.
    pub y_bytes: usize,
    /// `n_tokens * k_top * 4`: the inverse permutation and the top-k weights.
    pub slot_bytes: usize,
    /// `n_tokens * dim * 4`: the F32 residual stream.
    pub residual_bytes: usize,
    /// Workgroups along y: `ceil(dim / 256)`.
    pub grid_y: u32,
}

/// Extent checks shared by both combine twins (pure). Every extent must fit
/// the kernarg `u32`s, every byte span must stay below 2^32 (the kernels
/// address with 32-bit byte offsets) and `k_top` is `1..=8`. Returns
/// `(y_bytes, slot_bytes, residual_bytes)`.
fn validate_moe_down_combine_extents(
    dim: usize,
    k_top: usize,
    n_tokens: usize,
    grouped_rows: usize,
) -> Result<(usize, usize, usize), String> {
    if !(1..=MOE_DOWN_COMBINE_BF16IN_MAX_K_TOP).contains(&k_top) {
        return Err(format!("k_top {k_top} is outside 1..=8"));
    }
    if dim == 0 || n_tokens == 0 {
        return Err("need dim > 0 and n_tokens > 0".to_string());
    }
    if u32::try_from(dim).is_err()
        || u32::try_from(grouped_rows).is_err()
        || i32::try_from(n_tokens).is_err()
    {
        return Err(format!(
            "an extent exceeds 32 bits (dim {dim}, grouped_rows {grouped_rows}, n_tokens {n_tokens})"
        ));
    }
    let span = |parts: [usize; 3], what: &str| -> Result<usize, String> {
        parts
            .iter()
            .try_fold(1usize, |acc, &p| acc.checked_mul(p))
            .filter(|&bytes| bytes <= u32::MAX as usize)
            .ok_or_else(|| format!("{what} byte span exceeds 32 bits"))
    };
    let y_bytes = span([grouped_rows, dim, 2], "y_grouped_bf16")?;
    let slot_bytes = span([n_tokens, k_top, 4], "inverse_perm/topk_weights")?;
    let residual_bytes = span([n_tokens, dim, 4], "x_residual")?;
    Ok((y_bytes, slot_bytes, residual_bytes))
}

/// Host-side domain check of the v1 combine launch (pure): the shared extent
/// checks, and the grid must fit.
pub(crate) fn validate_moe_down_combine(
    dim: usize,
    k_top: usize,
    n_tokens: usize,
    grouped_rows: usize,
) -> Result<MoeDownCombineGeometry, String> {
    let (y_bytes, slot_bytes, residual_bytes) =
        validate_moe_down_combine_extents(dim, k_top, n_tokens, grouped_rows)?;
    let grid_y = dim.div_ceil(MOE_DOWN_COMBINE_BF16IN_BLOCK as usize);
    if grid_y > GRID_Y_MAX {
        return Err(format!("grid y {grid_y} exceeds {GRID_Y_MAX}"));
    }
    Ok(MoeDownCombineGeometry {
        y_bytes,
        slot_bytes,
        residual_bytes,
        grid_y: grid_y as u32,
    })
}

/// Host-side domain check of the v2 combine launch (pure): the shared extent
/// checks, `dim` a positive multiple of [`MOE_DOWN_COMBINE_V2_COLS_PER_BLOCK`]
/// (256 lanes x eight contiguous columns), grid y `dim / 2048` must fit.
pub(crate) fn validate_moe_down_combine_v2(
    dim: usize,
    k_top: usize,
    n_tokens: usize,
    grouped_rows: usize,
) -> Result<MoeDownCombineGeometry, String> {
    let (y_bytes, slot_bytes, residual_bytes) =
        validate_moe_down_combine_extents(dim, k_top, n_tokens, grouped_rows)?;
    if dim % MOE_DOWN_COMBINE_V2_COLS_PER_BLOCK != 0 {
        return Err(format!(
            "dim {dim} is not a multiple of {MOE_DOWN_COMBINE_V2_COLS_PER_BLOCK}"
        ));
    }
    let grid_y = dim / MOE_DOWN_COMBINE_V2_COLS_PER_BLOCK;
    if grid_y > GRID_Y_MAX {
        return Err(format!("grid y {grid_y} exceeds {GRID_Y_MAX}"));
    }
    Ok(MoeDownCombineGeometry {
        y_bytes,
        slot_bytes,
        residual_bytes,
        grid_y: grid_y as u32,
    })
}

/// Whether the v2 combine accepts this launch shape (pure; no device, no
/// kernel image): `dim` a positive multiple of 2048, `k_top` in `1..=8`, and
/// the v1 extent limits. The dispatch selection consults it before choosing v2.
pub fn moe_combine_v2_shape_supported(
    dim: usize,
    k_top: usize,
    n_tokens: usize,
    grouped_rows: usize,
) -> bool {
    validate_moe_down_combine_v2(dim, k_top, n_tokens, grouped_rows).is_ok()
}

impl Gpu {
    /// This device has the embedded builder object of the slot ops
    /// (`kernels::MOE_SLOT_OPS_PM_GFX1201` on gfx1201; every other arch: no).
    pub fn moe_slot_ops_available(&self) -> bool {
        moe_slot_ops_kernel(&self.arch).is_some()
    }

    /// Grouped BF16-input down combine, certified builder kernel
    /// ([`MOE_DOWN_COMBINE_BF16IN_PM_GFX1201_SYMBOL`]), stream-ordered on the
    /// active stream.
    ///
    /// `y_grouped_bf16`: BF16 bits, `[grouped_rows, dim]`; `inverse_perm`: i32
    /// `[n_tokens * k_top]`, flat slot to grouped row (`< 0` = unrouted);
    /// `topk_weights`: F32 `[n_tokens * k_top]`; `x_residual`: F32
    /// `[n_tokens, dim]`, updated in place (`+=`). Kernarg (48 B):
    /// `y@0, inverse_perm@8, topk_weights@16, x_residual@24` (u64 each), then
    /// u32 `dim@32, k_top@36, grouped_rows@40, n_tokens@44`. Grid
    /// `[n_tokens, ceil(dim / 256), 1]`, block `[256, 1, 1]`, no LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_combine_grouped_bf16in_pm(
        &mut self,
        y_grouped_bf16: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_weights: &GpuTensor,
        x_residual: &GpuTensor,
        dim: usize,
        k_top: usize,
        n_tokens: usize,
        grouped_rows: usize,
    ) -> HipResult<()> {
        let kernel = moe_slot_ops_kernel(&self.arch);
        self.moe_down_combine_launch(
            "moe_down_combine_grouped_bf16in_pm",
            kernel,
            validate_moe_down_combine,
            [y_grouped_bf16, inverse_perm, topk_weights, x_residual],
            [dim, k_top, n_tokens, grouped_rows],
        )
    }

    /// This device has the embedded v2 builder object of the combine
    /// (`kernels::MOE_SLOT_OPS_V2_PM_GFX1201` on gfx1201; every other arch: no).
    pub fn moe_combine_v2_available(&self) -> bool {
        moe_slot_ops_v2_kernel(&self.arch).is_some()
    }

    /// Grouped BF16-input down combine v2, certified builder kernel
    /// ([`MOE_DOWN_COMBINE_BF16IN_V2_PM_GFX1201_SYMBOL`], bundle
    /// `kernels::MOE_SLOT_OPS_V2_PM_GFX1201`), stream-ordered on the active
    /// stream. Same operands, kernarg layout and arithmetic as
    /// [`Gpu::moe_down_combine_grouped_bf16in_pm`] (each output `+0`, ascending
    /// `k`, unsigned skip of `slot >= grouped_rows`, `fma(w, bf16, acc)`, final
    /// `residual + acc`); each lane owns eight contiguous columns. `dim` must be
    /// a positive multiple of 2048 and `k_top` in `1..=8`
    /// ([`moe_combine_v2_shape_supported`]). Grid
    /// `[n_tokens, dim / 2048, 1]`, block `[256, 1, 1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_combine_grouped_bf16in_v2_pm(
        &mut self,
        y_grouped_bf16: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_weights: &GpuTensor,
        x_residual: &GpuTensor,
        dim: usize,
        k_top: usize,
        n_tokens: usize,
        grouped_rows: usize,
    ) -> HipResult<()> {
        let kernel = moe_slot_ops_v2_kernel(&self.arch);
        self.moe_down_combine_launch(
            "moe_down_combine_grouped_bf16in_v2_pm",
            kernel,
            validate_moe_down_combine_v2,
            [y_grouped_bf16, inverse_perm, topk_weights, x_residual],
            [dim, k_top, n_tokens, grouped_rows],
        )
    }

    /// Shared launch of the v1 and v2 combine twins: they differ only in the
    /// embedded object, the host domain check and the grid.
    fn moe_down_combine_launch(
        &mut self,
        label: &str,
        kernel: Option<(&'static [u8], &'static str)>,
        validate: fn(usize, usize, usize, usize) -> Result<MoeDownCombineGeometry, String>,
        tensors: [&GpuTensor; 4],
        extents: [usize; 4],
    ) -> HipResult<()> {
        let [y_grouped_bf16, inverse_perm, topk_weights, x_residual] = tensors;
        let [dim, k_top, n_tokens, grouped_rows] = extents;
        self.bind_thread()?;
        let refuse = |what: &str| Err(hip_bridge::HipError::new(0, &format!("{label}: {what}")));
        let Some((image, symbol)) = kernel else {
            return refuse(&format!(
                "the slot-ops combine is a gfx1201 builder module (device is {}); no hipcc fallback",
                self.arch
            ));
        };
        let geometry = match validate(dim, k_top, n_tokens, grouped_rows) {
            Ok(geometry) => geometry,
            Err(why) => return refuse(&why),
        };
        require_span(y_grouped_bf16, &[grouped_rows, dim, 2], "y_grouped_bf16")?;
        require_span(inverse_perm, &[n_tokens, k_top, 4], "inverse_perm")?;
        require_span(topk_weights, &[n_tokens, k_top, 4], "topk_weights")?;
        require_span(x_residual, &[n_tokens, dim, 4], "x_residual")?;
        self.ensure_embedded_kernel(symbol, image, symbol)?;
        let y = y_grouped_bf16.buf.as_ptr();
        let perm = inverse_perm.buf.as_ptr();
        let weights = topk_weights.buf.as_ptr();
        let residual = x_residual.buf.as_ptr();
        let (dim_u32, k_top_u32, rows_u32, tokens_u32) = (
            dim as u32,
            k_top as u32,
            grouped_rows as u32,
            n_tokens as u32,
        );
        let mut params = [
            &y as *const _ as *mut c_void,
            &perm as *const _ as *mut c_void,
            &weights as *const _ as *mut c_void,
            &residual as *const _ as *mut c_void,
            &dim_u32 as *const _ as *mut c_void,
            &k_top_u32 as *const _ as *mut c_void,
            &rows_u32 as *const _ as *mut c_void,
            &tokens_u32 as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            symbol,
            geometry.y_bytes + geometry.slot_bytes * 2 + geometry.residual_bytes * 2,
        );
        let result = self.launch_maybe_blob(
            symbol,
            [tokens_u32, geometry.grid_y, 1],
            [MOE_DOWN_COMBINE_BF16IN_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(y);
                b.push_ptr(perm);
                b.push_ptr(weights);
                b.push_ptr(residual);
                b.push_u32(dim_u32);
                b.push_u32(k_top_u32);
                b.push_u32(rows_u32);
                b.push_u32(tokens_u32);
                b
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

    const A3B_DIM: usize = 2048;

    #[test]
    fn kernel_selection_is_gfx1201_only() {
        let (_, symbol) = moe_slot_ops_kernel("gfx1201").unwrap();
        assert_eq!(symbol, MOE_DOWN_COMBINE_BF16IN_PM_GFX1201_SYMBOL);
        assert!(symbol.ends_with("gfx1201"));
        for arch in [
            "gfx1200", "gfx1100", "gfx1101", "gfx1151", "gfx942", "gfx1030", "gfx11", "",
        ] {
            assert!(moe_slot_ops_kernel(arch).is_none(), "{arch:?}");
        }
    }

    #[test]
    fn embedded_image_exports_the_launcher_symbol() {
        let (image, symbol) = moe_slot_ops_kernel("gfx1201").unwrap();
        assert!(
            image.windows(4).any(|w| w == b"\x7fELF"),
            "no device ELF in the {symbol} bundle"
        );
        assert!(
            image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
            "{symbol} missing from its image"
        );
    }

    #[test]
    fn a3b_prefill_chunk_geometry() {
        // 8192-token chunk, E=256, top_k=8, H=2048: 65_536 grouped rows (plus pad slack).
        let g = validate_moe_down_combine(A3B_DIM, 8, 8192, 8192 * 8 + 256 * 15).unwrap();
        assert_eq!(g.grid_y, 8);
        assert_eq!(g.slot_bytes, 8192 * 8 * 4);
        assert_eq!(g.residual_bytes, 8192 * A3B_DIM * 4);
        assert_eq!(g.y_bytes, (8192 * 8 + 256 * 15) * A3B_DIM * 2);
    }

    #[test]
    fn grid_y_is_ceil_of_dim_over_256() {
        for (dim, grid) in [
            (1usize, 1u32),
            (255, 1),
            (256, 1),
            (257, 2),
            (2048, 8),
            (2049, 9),
        ] {
            assert_eq!(
                validate_moe_down_combine(dim, 8, 1, 1).unwrap().grid_y,
                grid
            );
        }
    }

    #[test]
    fn k_top_domain() {
        assert!(validate_moe_down_combine(256, 0, 1, 1).is_err());
        assert!(validate_moe_down_combine(256, 9, 1, 1).is_err());
        for k in 1..=8 {
            assert!(validate_moe_down_combine(256, k, 1, 1).is_ok(), "{k}");
        }
    }

    #[test]
    fn zero_extents_are_refused_but_empty_grouped_rows_are_not() {
        assert!(validate_moe_down_combine(0, 8, 1, 1).is_err());
        assert!(validate_moe_down_combine(256, 8, 0, 1).is_err());
        // Every slot unrouted: the combine adds nothing, y is never read.
        assert_eq!(validate_moe_down_combine(256, 8, 1, 0).unwrap().y_bytes, 0);
    }

    #[test]
    fn byte_offsets_stay_below_2_pow_32() {
        let max = u32::MAX as usize;
        // y: rows * dim * 2.
        assert!(validate_moe_down_combine(2048, 8, 1, max / (2048 * 2)).is_ok());
        assert!(validate_moe_down_combine(2048, 8, 1, max / (2048 * 2) + 1).is_err());
        // residual: tokens * dim * 4.
        assert!(validate_moe_down_combine(2048, 1, max / (2048 * 4), 1).is_ok());
        assert!(validate_moe_down_combine(2048, 1, max / (2048 * 4) + 1, 1).is_err());
        // slots: tokens * k_top * 4.
        assert!(validate_moe_down_combine(1, 8, max / (8 * 4), 1).is_ok());
        assert!(validate_moe_down_combine(1, 8, max / (8 * 4) + 1, 1).is_err());
    }

    #[test]
    fn extents_beyond_32_bits_and_overflow_are_refused() {
        assert!(validate_moe_down_combine(u32::MAX as usize + 1, 8, 1, 1).is_err());
        assert!(validate_moe_down_combine(256, 8, 1, u32::MAX as usize + 1).is_err());
        assert!(validate_moe_down_combine(256, 8, i32::MAX as usize + 1, 1).is_err());
        assert!(validate_moe_down_combine(usize::MAX, 8, 1, usize::MAX).is_err());
    }

    #[test]
    fn grid_y_limit() {
        // 65_535 * 256 columns is the largest dim whose grid fits; its BF16 row is < 2^32 bytes.
        assert!(validate_moe_down_combine(65_535 * 256, 8, 1, 1).is_ok());
        assert!(validate_moe_down_combine(65_535 * 256 + 1, 8, 1, 1).is_err());
    }

    #[test]
    fn v2_kernel_selection_is_gfx1201_only_and_separate_from_v1() {
        let (v2_image, v2_symbol) = moe_slot_ops_v2_kernel("gfx1201").unwrap();
        let (v1_image, v1_symbol) = moe_slot_ops_kernel("gfx1201").unwrap();
        assert_eq!(v2_symbol, MOE_DOWN_COMBINE_BF16IN_V2_PM_GFX1201_SYMBOL);
        assert_eq!(v2_symbol, "moe_down_combine_grouped_bf16in_v2_pm_gfx1201");
        assert_ne!(v2_symbol, v1_symbol);
        assert_ne!(v2_image, v1_image, "v2 has its own bundle");
        for arch in [
            "gfx1200", "gfx1100", "gfx1101", "gfx1151", "gfx942", "gfx1030", "gfx11", "",
        ] {
            assert!(moe_slot_ops_v2_kernel(arch).is_none(), "{arch:?}");
        }
    }

    #[test]
    fn v2_embedded_image_exports_the_launcher_symbol() {
        let (image, symbol) = moe_slot_ops_v2_kernel("gfx1201").unwrap();
        assert!(
            image.windows(4).any(|w| w == b"\x7fELF"),
            "no device ELF in the {symbol} bundle"
        );
        assert!(
            image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
            "{symbol} missing from its image"
        );
    }

    #[test]
    fn v2_geometry_is_dim_over_2048_columns_per_block() {
        for (dim, grid) in [
            (2048usize, 1u32),
            (4096, 2),
            (6144, 3),
            (2048 * 65_535, 65_535),
        ] {
            let g = validate_moe_down_combine_v2(dim, 8, 1, 1).unwrap();
            assert_eq!(g.grid_y, grid, "dim {dim}");
            assert_eq!(g.y_bytes, dim * 2);
            assert_eq!(g.slot_bytes, 8 * 4);
            assert_eq!(g.residual_bytes, dim * 4);
        }
        let g = validate_moe_down_combine_v2(A3B_DIM, 8, 8192, 8192 * 8 + 256 * 15).unwrap();
        assert_eq!(g.grid_y, 1);
        assert_eq!(g.slot_bytes, 8192 * 8 * 4);
        assert_eq!(g.residual_bytes, 8192 * A3B_DIM * 4);
        assert_eq!(g.y_bytes, (8192 * 8 + 256 * 15) * A3B_DIM * 2);
    }

    #[test]
    fn v2_dim_must_be_a_positive_multiple_of_2048() {
        for dim in [0usize, 1, 256, 1024, 2047, 2049, 4095, 4097] {
            assert!(validate_moe_down_combine_v2(dim, 8, 1, 1).is_err(), "{dim}");
            assert!(!moe_combine_v2_shape_supported(dim, 8, 1, 1), "{dim}");
        }
        for dim in [2048usize, 4096, 2048 * 3] {
            assert!(validate_moe_down_combine_v2(dim, 8, 1, 1).is_ok(), "{dim}");
            assert!(moe_combine_v2_shape_supported(dim, 8, 1, 1), "{dim}");
        }
        // v1 still accepts the dims v2 refuses.
        assert!(validate_moe_down_combine(256, 8, 1, 1).is_ok());
        assert!(validate_moe_down_combine(2049, 8, 1, 1).is_ok());
    }

    #[test]
    fn v2_k_top_domain_and_shared_extent_limits_match_v1() {
        assert!(validate_moe_down_combine_v2(2048, 0, 1, 1).is_err());
        assert!(validate_moe_down_combine_v2(2048, 9, 1, 1).is_err());
        for k in 1..=8 {
            assert!(validate_moe_down_combine_v2(2048, k, 1, 1).is_ok(), "{k}");
        }
        assert!(validate_moe_down_combine_v2(2048, 8, 0, 1).is_err());
        assert_eq!(
            validate_moe_down_combine_v2(2048, 8, 1, 0).unwrap().y_bytes,
            0
        );
        let max = u32::MAX as usize;
        assert!(validate_moe_down_combine_v2(2048, 8, 1, max / (2048 * 2)).is_ok());
        assert!(validate_moe_down_combine_v2(2048, 8, 1, max / (2048 * 2) + 1).is_err());
        assert!(validate_moe_down_combine_v2(2048, 1, max / (2048 * 4), 1).is_ok());
        assert!(validate_moe_down_combine_v2(2048, 1, max / (2048 * 4) + 1, 1).is_err());
        assert!(validate_moe_down_combine_v2(2048, 8, i32::MAX as usize + 1, 1).is_err());
        assert!(validate_moe_down_combine_v2(2048 * 65_536, 8, 1, 1).is_err());
    }
}
