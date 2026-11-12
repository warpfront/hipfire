// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission of the batched QSA decode core on exact gfx942
//! (kernels `qwen4_qsa_lanes_*_gfx942`, launcher
//! [`crate::cdna::gfx942::Gfx942Device::qwen4_qsa_lanes_core`]).
//!
//! A child module of `tensor_ops` like `qsa_core_lanes` (the gfx1201 twin), so the admission
//! shares ONE definition of "which kernels would the per-lane call take" with the per-lane
//! path: the singleton launchers' private validators and route predicates. Every shape,
//! route and flag refusal happens in [`qsa_core_lanes_gfx942`] before anything launches:
//! `Ok(None)` means the set does not take this route (nothing launched; run the lanes one by
//! one), so a caller that must act exactly once before the launch does so only when a plan
//! comes back.
//!
//! What a gfx942 lane's Core is, per launch (`execute_indexed_attention_hc_impl`, eager):
//! the one-launch decode prologue (`indexed_attention_decode_prologue_{f32,fp8_fnuz}`), the
//! incremental pool (`indexed_attention_pool_rope_f32`), the live score / select-from-scores
//! pair of `indexed_attention_select_rows8` with the persistent-selection mirror (hipcc arm:
//! no PM images and no filter trio exist on gfx942), and the one-row per-head parallel
//! attention (`indexed_attention_attention_{f32,fp8_fnuz}_batched`, policy `PerHead`: no
//! split-K, hg4, hg12 or WMMA route is admitted for a decode call on gfx942).

use super::qsa_core_lanes::{disjoint, pool_refusal, ptr, PoolExpect};
use super::{
    check_qsa_format, checked_i32, qsa_attention_selected_at, qsa_decode_attention_policy_for_row,
    qsa_decode_per_head_parallel, qsa_decode_rows_admitted, qsa_decode_splitk_eager,
    qsa_select_batch_validate, select_forced_serial, QsaCoreLane, QsaDecodeAttentionPolicy,
    QsaKvFormat, QSA_SELECT_FROM_SCORES_MAX_BUDGET, QWEN4_QSA_SELECT_EXACT,
};
use crate::cdna::gfx942::{
    Qwen4QsaLaneDesc, Qwen4QsaLanesPlan, Qwen4QsaLanesShape, QWEN4_QSA_LANES_MAX,
    QWEN4_QSA_LANE_ROWS,
};
use crate::{DType, Gpu};
use hip_bridge::HipResult;

/// Most bytes of F32 score scratch (every lane's score rows at the pooled-capacity stride) a
/// plan may ask the shared F16-X scratch for; a larger set runs per lane.
const SCORE_SCRATCH_MAX_BYTES: usize = 512 << 20;

/// Everything about the device and launch mode the admission weighs before it looks at a
/// single lane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QsaLanesGfx942Facts {
    /// The device is an exact gfx942.
    pub enabled: bool,
    pub recording: bool,
    pub capture: bool,
    /// Developer extent check (`qwen4_qsa_extent_check`): wraps every singleton launch with
    /// its own checks, which the lane launches skip.
    pub extent_check: bool,
    /// The opt-in exact selector (`HIPFIRE_QWEN4_QSA_SELECT_EXACT`) replaces the live
    /// select route the lanes twin (the per-lane path would take the fused select).
    pub select_exact: bool,
}

/// Why `lanes` lanes (at least `min_lanes`) cannot take the lane route on this device /
/// launch mode, or `None`.
pub(crate) fn qsa_lanes_gfx942_gpu_refusal(
    f: &QsaLanesGfx942Facts,
    lanes: usize,
    min_lanes: usize,
) -> Option<&'static str> {
    if lanes < min_lanes.max(1) {
        Some("fewer lanes than the route needs")
    } else if lanes > QWEN4_QSA_LANES_MAX {
        Some("more lanes than the lane kernels hold")
    } else if !f.enabled {
        Some("the arch is not exact gfx942")
    } else if f.recording {
        Some("a replay tape is recording")
    } else if f.capture {
        Some("hipGraph capture is active")
    } else if f.extent_check {
        Some("the QSA extent check is on")
    } else if f.select_exact {
        Some("the opt-in exact selector replaces the live select route")
    } else {
        None
    }
}

/// Admission of the per-lane QSA Core launches of 2..=[`QWEN4_QSA_LANES_MAX`] lanes as five
/// launches (see [`Qwen4QsaLanesPlan`] and `Gfx942Device::qwen4_qsa_lanes_core`).
///
/// Refuses (`Ok(None)`: nothing launched, run the lanes one by one) unless the arch is exact gfx942, no replay tape is recording and no
/// hipGraph capture is open, and every lane is an F32-state Full decode / verify
/// window of 1..=8 rows that completes at least one pooled block and whose per-lane launches
/// would be: the one-launch prologue, the incremental pool, the live rows-8 score /
/// select-from-scores pair with the persistent mirror, and the one-row per-head parallel
/// attention for every row. Lanes share the weights and shape, own their state and their
/// scratch rows, and every launch argument agrees with its siblings'.
pub fn qsa_core_lanes_gfx942(
    gpu: &Gpu,
    lanes: &[QsaCoreLane<'_>],
) -> HipResult<Option<Qwen4QsaLanesPlan>> {
    // E4M3-FNUZ K/V takes the per-lane route: its kernels convert through opcodes the PeaceMaker
    // CDNA twin sealer does not model. (`qsa_core_lanes_gfx942_min`, the byte oracle's entry,
    // still admits them.)
    if lanes
        .iter()
        .any(|lane| !qsa_lanes_gfx942_format_admitted(lane.prologue.format))
    {
        return Ok(None);
    }
    qsa_core_lanes_gfx942_min(gpu, lanes, 2)
}

/// Whether the interpreter's lane route (`qsa_core_lanes_gfx942`) takes lanes with K/V state
/// `format`: F32 only. E4M3-FNUZ lanes run the per-lane launches.
pub(crate) fn qsa_lanes_gfx942_format_admitted(format: QsaKvFormat) -> bool {
    format == QsaKvFormat::F32
}

/// [`qsa_core_lanes_gfx942`] admitting `min_lanes..=`[`QWEN4_QSA_LANES_MAX`] lanes. The
/// interpreter never groups fewer than two (`min_lanes = 2`); the byte oracle runs the
/// kernels with one lane too.
#[doc(hidden)]
pub fn qsa_core_lanes_gfx942_min(
    gpu: &Gpu,
    lanes: &[QsaCoreLane<'_>],
    min_lanes: usize,
) -> HipResult<Option<Qwen4QsaLanesPlan>> {
    let facts = QsaLanesGfx942Facts {
        enabled: gpu.arch_caps.is_gfx942(),
        recording: gpu.replay.is_recording(),
        capture: gpu.graphs.capture_mode,
        extent_check: gpu.flags.qwen4_qsa_extent_check,
        select_exact: *QWEN4_QSA_SELECT_EXACT,
    };
    if qsa_lanes_gfx942_gpu_refusal(&facts, lanes.len(), min_lanes).is_some() {
        return Ok(None);
    }
    // Admission reads metadata only: a failing validator is the per-lane path's error to
    // raise, so it declines instead.
    Ok(admit(gpu, lanes).unwrap_or(None))
}

const NPTR: usize = 11;

fn admit(gpu: &Gpu, lanes: &[QsaCoreLane<'_>]) -> HipResult<Option<Qwen4QsaLanesPlan>> {
    let mut descs: Vec<Qwen4QsaLaneDesc> = Vec::with_capacity(lanes.len());
    let (mut positions, mut rows_all) = (Vec::new(), Vec::new());
    let mut shape: Option<Qwen4QsaLanesShape> = None;
    let mut pool_cfg: Option<(u64, crate::rope::RopeYarn)> = None;
    let (mut max_rows, mut pool_grid, mut max_blocks, mut attention_lds, mut pooled_capacity) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    // Per scratch / state family: the lane's [start, end) byte range.
    let mut families: Vec<Vec<(usize, usize)>> = vec![Vec::new(); NPTR];
    for lane in lanes {
        let (p, s, a) = (&lane.prologue, &lane.select, &lane.attention);
        let rows = p.rows;
        let position = p.position;
        let compress = a.compress;
        let format = p.format;
        let kv_width = p.kv_heads * p.head_dim;
        let index_q_width = p.index_heads * p.index_dim;
        let index_width = index_q_width + p.index_kv_width;
        let q_width = p.heads * p.head_dim;
        let blocks_l = position
            .saturating_add(rows)
            .checked_div(compress)
            .unwrap_or(0);
        // -- The per-lane launches' own admission (same predicates). ------------------------
        if !matches!(format, QsaKvFormat::F32 | QsaKvFormat::Fp8Fnuz)
            || a.format != format
            || !gpu.arch_caps.is_gfx942()
            || !(1..=QWEN4_QSA_LANE_ROWS).contains(&rows)
            || s.rows != rows
            || a.rows != rows
            || s.position_start != position
            || a.position_start != position
            || compress == 0
            || s.compress != compress
            || s.block_count != blocks_l
            // The live score / select-from-scores route needs a live block; a lane that
            // completes none takes the fused select (not twinned here).
            || blocks_l == 0
            || position
                .checked_add(rows)
                .is_none_or(|e| e > i32::MAX as usize)
            // The score kernel is the rows8 tile (`route_rows > 8` takes rows16).
            || gpu.route_rows(rows) > QWEN4_QSA_LANE_ROWS
            || gpu.route_rows(rows) > gpu.few_row_limit()
            || !qsa_decode_rows_admitted(gpu, rows)
            || !qsa_decode_splitk_eager(gpu)
            || p.head_dim != 256
            || p.index_heads != 4
            || p.index_dim > 128
            || p.index_dim % 4 != 0
            || p.index_kv_width != p.index_dim
            || s.pooled_bf16.is_some()
            || p.raw_index_keys.dtype != format.index_dtype()
            || s.pooled.dtype != format.index_dtype()
            || format.index_dtype() != DType::F32
        {
            return Ok(None);
        }
        // Prologue validity (`indexed_attention_decode_prologue`).
        check_qsa_format(
            gpu,
            format,
            p.kv_heads,
            p.head_dim,
            [p.full_keys, p.full_values],
        )?;
        let row_units = format.kv_row_units(p.kv_heads, p.head_dim);
        let row_bytes = format.kv_row_bytes(p.kv_heads, p.head_dim);
        let end = position + rows;
        let bad = [p.index_q_norm, p.q_norm, p.k_norm]
            .iter()
            .any(|n| n.dtype != DType::BF16)
            || p.index_heads == 0
            || p.heads == 0
            || p.kv_heads == 0
            || p.index_dim == 0
            || p.index_q_norm.numel() != p.index_dim
            || p.q_norm.numel() != p.head_dim
            || p.k_norm.numel() != p.head_dim
            || p.index_row.numel() < rows * index_width
            || p.qgate.numel() < rows * 2 * q_width
            || p.keys.numel() < rows * kv_width
            || p.values.numel() < rows * kv_width
            || p.full_keys.numel() < end * row_units
            || p.full_values.numel() < end * row_units
            || p.raw_index_keys.numel() < end * p.index_kv_width;
        if bad {
            return Ok(None);
        }
        for t in [p.index_row, p.qgate, p.keys, p.values] {
            if t.dtype != DType::F32 {
                return Ok(None);
            }
        }
        // Select: the live score / select-from-scores route with the persistent mirror
        // (`indexed_attention_select_batch_impl`: parallel arch, no exact selector, F32 pooled
        // keys, four index heads, <= 128 wide, a live block, a budget of at most 512 blocks).
        qsa_select_batch_validate(s)?;
        let parallel = gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
            && s.shape_blocks > 0
            && !select_forced_serial();
        if !parallel
            || s.budget_blocks > QSA_SELECT_FROM_SCORES_MAX_BUDGET
            || s.index_heads != p.index_heads
            || s.index_dim != p.index_dim
            || s.query_row_stride != index_width
            || s.capacity != a.capacity
            || s.budget_blocks != a.budget_blocks
            || s.query.buf.as_ptr() != p.index_row.buf.as_ptr()
            || s.pooled.buf.as_ptr() == p.raw_index_keys.buf.as_ptr()
            || lane.persistent_selection.dtype != DType::Raw
            || lane.persistent_selection.numel() < s.capacity * std::mem::size_of::<i32>()
        {
            return Ok(None);
        }
        // Attention: the per-head parallel kernel for every row.
        for t in [a.q_with_gate, a.output] {
            if t.dtype != DType::F32 {
                return Ok(None);
            }
        }
        check_qsa_format(
            gpu,
            a.format,
            a.n_kv_heads,
            a.head_dim,
            [a.full_keys, a.full_values],
        )?;
        let selected_bound = a.budget_blocks * compress + (compress - 1);
        let max_selected = end.min(a.capacity).min(selected_bound);
        if a.selected.dtype != DType::Raw
            || a.n_heads != p.heads
            || a.n_kv_heads != p.kv_heads
            || a.head_dim != p.head_dim
            || a.n_heads == 0
            || a.n_heads % a.n_kv_heads != 0
            || a.capacity == 0
            || end > a.full_capacity
            || a.q_with_gate.numel() < rows * 2 * q_width
            || a.output.numel() < rows * q_width
            || a.full_keys.numel() < a.full_capacity * row_units
            || a.full_values.numel() < a.full_keys.numel()
            || a.selected.numel() < rows * a.capacity * std::mem::size_of::<i32>()
            || a.shape_selected < max_selected
            || a.q_with_gate.buf.as_ptr() != p.qgate.buf.as_ptr()
            || a.full_keys.buf.as_ptr() != p.full_keys.buf.as_ptr()
            || a.full_values.buf.as_ptr() != p.full_values.buf.as_ptr()
            || a.selected.buf.as_ptr() != s.selected.buf.as_ptr()
        {
            return Ok(None);
        }
        for row in 0..rows {
            if qsa_decode_attention_policy_for_row(gpu, a, row) != QsaDecodeAttentionPolicy::PerHead
                || !qsa_decode_per_head_parallel(gpu, a, row)
            {
                return Ok(None);
            }
        }
        // Pool.
        let expect = PoolExpect {
            position,
            rows,
            compress,
            blocks: blocks_l,
            index_kv_width: p.index_kv_width,
            raw_keys: p.raw_index_keys.buf.as_ptr(),
            pooled: s.pooled.buf.as_ptr(),
            index_dtype: DType::F32,
        };
        if pool_refusal(lane.pool.as_ref(), &expect).is_some() {
            return Ok(None);
        }
        // -- Lane-to-lane agreement. -----------------------------------------------------
        let lane_shape = Qwen4QsaLanesShape {
            format,
            index_heads: p.index_heads,
            index_dim: p.index_dim,
            index_kv_width: p.index_kv_width,
            heads: p.heads,
            kv_heads: p.kv_heads,
            head_dim: p.head_dim,
            budget_blocks: a.budget_blocks,
            compress,
            capacity: a.capacity,
            full_capacity: a.full_capacity,
            pooled_capacity: 0,
            query_row_stride: s.query_row_stride,
            index_q_norm: ptr(p.index_q_norm),
            q_norm: ptr(p.q_norm),
            k_norm: ptr(p.k_norm),
            pool_norm: 0,
            yarn: p.yarn,
            pool_yarn: p.yarn,
            max_rows: 0,
            pool_grid: 0,
            max_blocks: 0,
            attention_lds: 0,
        };
        match &shape {
            None => shape = Some(lane_shape),
            Some(c) if same_launch_args(c, &lane_shape) => {}
            Some(_) => return Ok(None),
        }
        let Some(pool) = &lane.pool else {
            return Ok(None);
        };
        let cfg = (pool.norm.map_or(0, ptr), pool.yarn);
        match pool_cfg {
            None => pool_cfg = Some(cfg),
            Some(c) if c == cfg => {}
            Some(_) => return Ok(None),
        }
        pool_grid = pool_grid.max(rows.div_ceil(compress).max(1).min(pool.grid_bound));
        max_rows = max_rows.max(rows);
        max_blocks = max_blocks.max(blocks_l);
        pooled_capacity = pooled_capacity.max(s.shape_blocks);
        attention_lds = attention_lds.max(qsa_attention_selected_at(a, rows - 1) * 8);
        // Table slot: state, scratch (each lane's first row), scalars.
        let desc = Qwen4QsaLaneDesc {
            full_keys: ptr(p.full_keys),
            full_values: ptr(p.full_values),
            raw_idx: ptr(p.raw_index_keys),
            pooled: ptr(s.pooled),
            sel_persist: ptr(lane.persistent_selection),
            index_row: ptr(p.index_row),
            qgate: ptr(p.qgate),
            keys: ptr(p.keys),
            values: ptr(p.values),
            selected: ptr(s.selected),
            output: ptr(a.output),
        };
        // Extents the lane touches in each family (bytes, half-open).
        let slot = [
            (desc.full_keys, end * row_bytes),
            (desc.full_values, end * row_bytes),
            (desc.raw_idx, end * p.index_kv_width * 4),
            (desc.pooled, blocks_l * p.index_kv_width * 4),
            (desc.sel_persist, s.capacity * std::mem::size_of::<i32>()),
            (desc.index_row, rows * index_width * 4),
            (desc.qgate, rows * 2 * q_width * 4),
            (desc.keys, rows * kv_width * 4),
            (desc.values, rows * kv_width * 4),
            (
                desc.selected,
                rows * s.capacity * std::mem::size_of::<i32>(),
            ),
            (desc.output, rows * q_width * 4),
        ];
        for (family, (base, len)) in families.iter_mut().zip(slot) {
            family.push((base as usize, base as usize + len));
        }
        descs.push(desc);
        positions.push(checked_i32(position, "QSA lane position")?);
        rows_all.push(checked_i32(rows, "QSA lane rows")?);
    }
    // Lanes never share a state tensor or a scratch row range (ptrs of the same family are
    // distinct and their touched extents disjoint).
    if !families.into_iter().all(disjoint) {
        return Ok(None);
    }
    let Some(mut shape) = shape else {
        return Ok(None);
    };
    let score_bytes = descs
        .len()
        .checked_mul(max_rows)
        .and_then(|n| n.checked_mul(pooled_capacity))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()));
    if score_bytes.is_none_or(|b| b > SCORE_SCRATCH_MAX_BYTES) {
        return Ok(None);
    }
    shape.pooled_capacity = pooled_capacity;
    // Every lane's pool (there is one per lane: `blocks_l > 0`) shares norm and yarn.
    let (pool_norm, pool_yarn) = pool_cfg.expect("a lane's pool was admitted");
    shape.pool_norm = pool_norm;
    shape.pool_yarn = pool_yarn;
    shape.max_rows = max_rows;
    shape.pool_grid = pool_grid;
    shape.max_blocks = max_blocks;
    shape.attention_lds = attention_lds;
    Ok(Some(Qwen4QsaLanesPlan {
        shape,
        lanes: descs,
        positions,
        rows: rows_all,
    }))
}

/// Whether two lanes agree on every launch argument the kernels share.
fn same_launch_args(a: &Qwen4QsaLanesShape, b: &Qwen4QsaLanesShape) -> bool {
    a.format == b.format
        && a.index_heads == b.index_heads
        && a.index_dim == b.index_dim
        && a.index_kv_width == b.index_kv_width
        && a.heads == b.heads
        && a.kv_heads == b.kv_heads
        && a.head_dim == b.head_dim
        && a.budget_blocks == b.budget_blocks
        && a.compress == b.compress
        && a.capacity == b.capacity
        && a.full_capacity == b.full_capacity
        && a.query_row_stride == b.query_row_stride
        && a.index_q_norm == b.index_q_norm
        && a.q_norm == b.q_norm
        && a.k_norm == b.k_norm
        && a.yarn == b.yarn
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> QsaLanesGfx942Facts {
        QsaLanesGfx942Facts {
            enabled: true,
            recording: false,
            capture: false,
            extent_check: false,
            select_exact: false,
        }
    }

    #[test]
    fn interpreter_route_declines_fnuz_kv() {
        assert!(qsa_lanes_gfx942_format_admitted(QsaKvFormat::F32));
        assert!(!qsa_lanes_gfx942_format_admitted(QsaKvFormat::Fp8Fnuz));
    }

    #[test]
    fn gpu_refusals() {
        assert_eq!(qsa_lanes_gfx942_gpu_refusal(&facts(), 2, 2), None);
        assert_eq!(
            qsa_lanes_gfx942_gpu_refusal(&facts(), QWEN4_QSA_LANES_MAX, 2),
            None
        );
        assert_eq!(qsa_lanes_gfx942_gpu_refusal(&facts(), 1, 1), None);
        assert!(qsa_lanes_gfx942_gpu_refusal(&facts(), 1, 2).is_some());
        assert!(qsa_lanes_gfx942_gpu_refusal(&facts(), 0, 2).is_some());
        assert!(qsa_lanes_gfx942_gpu_refusal(&facts(), QWEN4_QSA_LANES_MAX + 1, 2).is_some());
        let mut f = facts();
        f.enabled = false;
        assert!(qsa_lanes_gfx942_gpu_refusal(&f, 4, 2).is_some());
        for field in ["recording", "capture", "extent_check", "select_exact"] {
            let mut f = facts();
            match field {
                "recording" => f.recording = true,
                "capture" => f.capture = true,
                "extent_check" => f.extent_check = true,
                _ => f.select_exact = true,
            }
            assert!(qsa_lanes_gfx942_gpu_refusal(&f, 4, 2).is_some(), "{field}");
        }
    }

    /// The lane route is only admitted where the per-lane attention is the per-head kernel:
    /// gfx942 F32 and FNUZ-fp8 attention is, for every row count and policy input.
    #[test]
    fn gfx942_attention_policy_is_per_head() {
        let flags = std::sync::Arc::new(crate::feature_flags::FeatureFlags::for_test("gfx942"));
        let caps = crate::arch_caps::ArchCaps::new("gfx942", flags.clone());
        for format in [QsaKvFormat::F32, QsaKvFormat::Fp8Fnuz] {
            for eager in [true, false] {
                assert_eq!(
                    super::super::qsa_decode_attention_policy_with(
                        &flags, &caps, format, 256, 24, 2, 2051, eager,
                    ),
                    QsaDecodeAttentionPolicy::PerHead,
                    "{format:?} eager={eager}"
                );
            }
        }
    }

    /// The live select route twinned here is the rows8 score + select-from-scores pair only
    /// where nothing selects another image: on gfx942 the builder pair never fits.
    #[test]
    fn gfx942_select_pm_never_fits() {
        assert!(!super::super::qsa_select_pm_shape_fits(
            false,
            DType::F32,
            1,
            4,
            128,
            4096,
            640,
            2051,
        ));
    }
}
