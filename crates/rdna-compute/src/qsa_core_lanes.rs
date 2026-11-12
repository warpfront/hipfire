// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-lane QSA decode core (fn-lanes, exact gfx1201, fp8 K/V): the four
//! launches of one lane's QSA Core phase (decode prologue, incremental pool,
//! fused select with the persistent-selection mirror, per-head attention) for
//! 2..=[`QSA_CORE_LANES_MAX`] lanes as ONE launch per kernel
//! (`kernels/src/qsa_lanes.gfx1201.hip`, concatenated onto the incumbent
//! `tensor_ops.hip` translation unit like `gdn_lanes.hip`). Each lane's blocks
//! run the incumbent device templates on the lane's own state pointers, its
//! own position and its own 1..=8 rows, so every output and state byte equals
//! the per-lane launches'.
//!
//! This is a child module of `tensor_ops`: it reuses the singleton launchers'
//! private validators and route predicates, so the admission below shares ONE
//! definition of "which kernel would the per-lane call take" with the
//! per-lane path. Every shape, route and flag refusal happens in
//! [`qsa_core_lanes`] before anything launches: `Ok(None)` means the set does
//! not take this route (nothing launched; run the lanes one by one), so a
//! caller that must act exactly once before the launch (the interpreter's
//! per-step hooks) does so only when a plan comes back.

use super::{
    blocks, check_qsa_format, checked_i32, checked_u32, qsa_attention_selected_at,
    qsa_bf16_split_route, qsa_decode_attention_policy_for_row, qsa_decode_per_head_parallel,
    qsa_decode_rows_admitted, qsa_decode_splitk_eager, qsa_select_batch_validate,
    qsa_select_pm_fits, select_forced_serial, IndexedAttentionAttentionBatch,
    IndexedAttentionDecodePrologue, IndexedAttentionPoolRope, IndexedAttentionSelectBatch,
    QsaDecodeAttentionPolicy, QsaKvFormat, QsaPositionBinding, QSA_SELECT_BATCHED_STATIC_LDS_BYTES,
    QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES, QSA_SELECT_PARALLEL_THREADS, QWEN4_QSA_SELECT_EXACT,
};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};

/// `tensor_ops.hip` plus the multi-lane QSA core kernels: the lane kernels
/// inline the same device templates under the same flags, and the
/// `tensor_ops` module's source stays untouched.
pub(crate) const QSA_LANES_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip"),
    include_str!("../../../kernels/src/qsa_lanes.gfx1201.hip")
);

/// Module key of [`QSA_LANES_SRC`].
pub const QSA_LANES_MODULE: &str = "qsa_lanes";
/// The four lane kernel symbols, in launch order.
pub const QSA_LANES_PROLOGUE_SYMBOL: &str = "indexed_attention_decode_prologue_fp8_lanes";
pub const QSA_LANES_POOL_SYMBOL: &str = "indexed_attention_pool_rope_bf16_lanes";
pub const QSA_LANES_SELECT_SYMBOL: &str = "indexed_attention_select_bf16_lanes";
pub const QSA_LANES_ATTENTION_SYMBOL: &str = "indexed_attention_attention_fp8_lanes";

/// [`QSA_LANES_SRC`] plus the fused lane kernels (`qsa_lanes_fuse.gfx1201.hip`): its own
/// module so the unfused module's code object does not change.
pub(crate) const QSA_LANES_FUSE_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip"),
    include_str!("../../../kernels/src/qsa_lanes.gfx1201.hip"),
    include_str!("../../../kernels/src/qsa_lanes_fuse.gfx1201.hip")
);
/// Module key of [`QSA_LANES_FUSE_SRC`].
pub const QSA_LANES_FUSE_MODULE: &str = "qsa_lanes_fuse";
/// Prologue + pool + select as one launch (one row per lane).
pub const QSA_LANES_FUSED_CORE_SYMBOL: &str = "indexed_attention_core_fused_fp8_lanes";
/// Attention + the out-projection's `mq_rotate_x` as one launch.
pub const QSA_LANES_FUSED_ATTENTION_SYMBOL: &str = "indexed_attention_attention_rot_fp8_lanes";

/// `HIPFIRE_QWEN4_QSA_LANES_FUSE` (developer, default off): the admitted lane set takes the
/// two fused launches ([`QsaCoreLanesPlan::launch_fused`]) instead of the four
/// [`QsaCoreLanesPlan::launch`] launches plus the out-projection's `mq_rotate_x`. Read once.
static QWEN4_QSA_LANES_FUSE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_LANES_FUSE", true)
});

/// Whether `HIPFIRE_QWEN4_QSA_LANES_FUSE` is on.
pub fn qsa_lanes_fuse_enabled() -> bool {
    *QWEN4_QSA_LANES_FUSE
}

/// Most dynamic + static LDS of a fused launch the admission allows (the fused core
/// kernel carries the prologue's and the pool's static LDS next to the select's).
const QSA_FUSE_LDS_MARGIN_BYTES: usize = 16 * 1024;

/// The out-projection the fused attention rotates for: the op that follows the Core
/// group and would run `rotate_x_mq_batched(x, x_rot, k, rows)` on the lanes' outputs.
pub struct QsaOutRotate<'a> {
    /// The out-projection's input (`rows` bank rows of `heads * head_dim` F32).
    pub x: &'a GpuTensor,
    /// Its rotation scratch (F32, at least `rows * heads * head_dim` elements).
    pub x_rot: &'a GpuTensor,
    /// Rows of the out-projection (the bank).
    pub rows: usize,
}

/// Most lanes one [`qsa_core_lanes`] call covers (`QSA_LANES_MAX`).
pub const QSA_CORE_LANES_MAX: usize = 8;
/// Most rows one lane covers (the verify window of the lane steps,
/// `LANE_MAX_ROWS`; the singleton few-row limit).
pub const QSA_CORE_LANE_ROWS: usize = 8;

/// One AR lane's (or verify window's) QSA Core step: the arguments of its
/// per-lane [`super::indexed_attention_decode_prologue`],
/// [`super::indexed_attention_pool_rope_incremental`],
/// [`super::indexed_attention_select_batch_mirrored`] and
/// [`super::indexed_attention_attention_batch_with_arena`] calls, exactly as
/// the interpreter builds them.
pub struct QsaCoreLane<'a> {
    pub prologue: IndexedAttentionDecodePrologue<'a>,
    /// `Some` exactly when the lane's `(position + rows) / compress` is
    /// non-zero (the per-lane path launches the pool then and only then).
    pub pool: Option<IndexedAttentionPoolRope<'a>>,
    pub select: IndexedAttentionSelectBatch<'a>,
    /// The lane's persistent selection (`selected_indices`), the mirror
    /// destination of the select kernel's last row.
    pub persistent_selection: &'a GpuTensor,
    pub attention: IndexedAttentionAttentionBatch<'a>,
}

/// A validated, admitted set of lanes ready for [`QsaCoreLanesPlan::launch`];
/// produced only by [`qsa_core_lanes`]. Plain data: the device pointers are
/// the ones the admitted lanes borrowed, so launch the plan before releasing
/// them.
pub struct QsaCoreLanesPlan {
    shape: Shape,
    table: LaneTable,
}

/// Launch arguments every admitted lane agrees on.
#[derive(Clone, Debug, PartialEq)]
struct Common {
    index_heads: usize,
    index_dim: usize,
    index_kv_width: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    budget_blocks: usize,
    compress: usize,
    capacity: usize,
    full_capacity: usize,
    query_row_stride: usize,
    index_q_norm: u64,
    q_norm: u64,
    k_norm: u64,
    yarn: crate::rope::RopeYarn,
}

/// Shape of an admitted set: the shared arguments plus the per-lane extents
/// that only widen the launch grids.
#[derive(Clone, Debug)]
struct Shape {
    lanes: usize,
    common: Common,
    max_rows: usize,
    /// Pool norm pointer and yarn (shared by every lane that pools).
    pool: Option<(u64, crate::rope::RopeYarn)>,
    /// `grid.x` of the pool launch; 0 when no lane pools.
    pool_grid: usize,
    /// Dynamic LDS of the select launch (scores of the widest lane's arena).
    select_lds: usize,
    /// Dynamic LDS of the attention launch (widest selected list).
    attention_lds: usize,
}

const NPTR: usize = 11;
const NINT: usize = 4;
#[cfg(test)]
const LANE_PTR_FIELDS: [&str; NPTR] = [
    "full_keys",
    "full_values",
    "raw_idx",
    "pooled",
    "sel_persist",
    "index_row",
    "qgate",
    "keys",
    "values",
    "selected",
    "output",
];

/// `#[repr(C)]`-order mirror of `struct QsaLanes` (`qsa_lanes.gfx1201.hip`):
/// eleven pointer arrays, then `pos`, `rows`, `blocks`, `first_block`; unused
/// slots are zero (832 bytes, 8-aligned).
#[derive(Clone, Debug, Default, PartialEq)]
struct LaneTable {
    ptrs: [[u64; QSA_CORE_LANES_MAX]; NPTR],
    ints: [[i32; QSA_CORE_LANES_MAX]; NINT],
}

impl LaneTable {
    /// Append the table as the kernels' by-value `QsaLanes` argument.
    fn push(&self, args: &mut KernargBlob) {
        for field in &self.ptrs {
            for &p in field {
                args.push_u64(p);
            }
        }
        for field in &self.ints {
            for &v in field {
                args.push_i32(v);
            }
        }
    }
}

/// Everything about the device and launch mode the admission weighs before it
/// looks at a single lane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QsaLanesGpuFacts {
    /// `HIPFIRE_QWEN4_QSA_LANES` on an exact gfx1201.
    pub enabled: bool,
    pub recording: bool,
    pub capture: bool,
    /// Developer extent check (`qwen4_qsa_extent_check`): wraps every
    /// singleton launch with its own checks, which the lane launches skip.
    pub extent_check: bool,
    /// The opt-in exact selector (`HIPFIRE_QWEN4_QSA_SELECT_EXACT`) replaces
    /// the fused select kernel the lanes twin.
    pub select_exact: bool,
}

/// Why `lanes` lanes (at least `min_lanes`) cannot take the lane route on this
/// device / launch mode, or `None`.
pub(crate) fn qsa_lanes_gpu_refusal(
    f: &QsaLanesGpuFacts,
    lanes: usize,
    min_lanes: usize,
) -> Option<&'static str> {
    if lanes < min_lanes.max(1) {
        Some("fewer lanes than the route needs")
    } else if lanes > QSA_CORE_LANES_MAX {
        Some("more lanes than the lane table holds")
    } else if !f.enabled {
        Some("HIPFIRE_QWEN4_QSA_LANES is off or the arch is not exact gfx1201")
    } else if f.recording {
        Some("a replay tape is recording")
    } else if f.capture {
        Some("hipGraph capture is active")
    } else if f.extent_check {
        Some("the QSA extent check is on")
    } else if f.select_exact {
        Some("the opt-in exact selector replaces the fused select kernel")
    } else {
        None
    }
}

/// What a lane's pool launch must agree with: the lane's prologue / select
/// scalars and key arenas.
pub(super) struct PoolExpect {
    pub(super) position: usize,
    pub(super) rows: usize,
    pub(super) compress: usize,
    pub(super) blocks: usize,
    pub(super) index_kv_width: usize,
    pub(super) raw_keys: *mut std::ffi::c_void,
    pub(super) pooled: *mut std::ffi::c_void,
    /// Dtype of the raw and pooled index-key arenas (`QsaKvFormat::index_dtype`).
    pub(super) index_dtype: DType,
}

/// Why a lane's pool launch (as the interpreter issues it) is not the one the
/// lane kernels twin, or `None`.
pub(super) fn pool_refusal(
    pool: Option<&IndexedAttentionPoolRope<'_>>,
    expect: &PoolExpect,
) -> Option<&'static str> {
    let PoolExpect {
        position,
        rows,
        compress,
        blocks,
        index_kv_width,
        raw_keys,
        pooled,
        index_dtype,
    } = *expect;
    let Some(p) = pool else {
        return (blocks != 0).then_some("no pool launch for a lane that completes blocks");
    };
    let binding = p.position;
    let Some(QsaPositionBinding {
        position_start,
        rows: binding_rows,
    }) = binding
    else {
        return Some("pool launch without a declared position");
    };
    if p.block_count != blocks
        || position_start != position
        || binding_rows != rows
        || p.compress != compress
        || p.index_dim != index_kv_width
        || p.raw_keys.buf.as_ptr() != raw_keys
        || p.pooled.buf.as_ptr() != pooled
        || p.raw_keys.dtype != index_dtype
        || p.pooled.dtype != index_dtype
        || p.grid_bound < p.block_count
        || p.block_count == 0
        || p.index_dim == 0
        || p.index_dim % 2 != 0
        || p.norm
            .is_none_or(|n| n.dtype != DType::BF16 || n.numel() != p.index_dim)
    {
        return Some("pool launch differs from the lane's prologue/select scalars");
    }
    let raw_elements = p
        .block_count
        .checked_mul(p.compress)
        .and_then(|n| n.checked_mul(p.index_dim));
    let pooled_elements = p.block_count.checked_mul(p.index_dim);
    match (raw_elements, pooled_elements) {
        (Some(raw), Some(pooled)) if p.raw_keys.numel() >= raw && p.pooled.numel() >= pooled => {
            None
        }
        _ => Some("pool extents exceed the key arenas"),
    }
}

/// Half-open byte ranges must be pairwise disjoint.
pub(super) fn disjoint(mut ranges: Vec<(usize, usize)>) -> bool {
    ranges.sort_unstable();
    ranges.windows(2).all(|w| w[0].1 <= w[1].0)
}

pub(super) fn ptr(t: &GpuTensor) -> u64 {
    t.buf.as_ptr() as usize as u64
}

/// Admission of the per-lane QSA Core launches of 2..=[`QSA_CORE_LANES_MAX`]
/// lanes as four launches (see [`QsaCoreLanesPlan::launch`]).
///
/// Refuses (`Ok(None)`: nothing launched, run the lanes one by one) unless the
/// flag is on (`HIPFIRE_QWEN4_QSA_LANES`, exact gfx1201), no replay tape is
/// recording and no hipGraph capture is open, and every lane is an fp8 Full
/// decode / verify window of 1..=8 rows whose per-lane launches would be:
/// the one-launch prologue, the incremental pool, the FUSED parallel select
/// with LDS-resident scores and the persistent mirror (not the PM / split /
/// global-score routes), and the one-row per-head parallel attention for every
/// row. Lanes share the weights and shape, own their state and their scratch
/// rows, and every launch argument agrees with its siblings'.
pub fn qsa_core_lanes(gpu: &Gpu, lanes: &[QsaCoreLane<'_>]) -> HipResult<Option<QsaCoreLanesPlan>> {
    qsa_core_lanes_min(gpu, lanes, 2)
}

/// [`qsa_core_lanes`] admitting `min_lanes..=`[`QSA_CORE_LANES_MAX`] lanes. The
/// interpreter never groups fewer than two (`min_lanes = 2`); the byte oracle
/// runs the kernels with one lane too.
#[doc(hidden)]
pub fn qsa_core_lanes_min(
    gpu: &Gpu,
    lanes: &[QsaCoreLane<'_>],
    min_lanes: usize,
) -> HipResult<Option<QsaCoreLanesPlan>> {
    let facts = QsaLanesGpuFacts {
        enabled: gpu.flags.qwen4_qsa_lanes_enabled() && gpu.arch_caps.is_gfx1201(),
        recording: gpu.replay.is_recording(),
        capture: gpu.graphs.capture_mode,
        extent_check: gpu.flags.qwen4_qsa_extent_check,
        select_exact: *QWEN4_QSA_SELECT_EXACT,
    };
    if qsa_lanes_gpu_refusal(&facts, lanes.len(), min_lanes).is_some() {
        return Ok(None);
    }
    // Admission reads metadata only: a failing validator is the per-lane
    // path's error to raise, so it declines instead.
    Ok(admit(gpu, lanes).unwrap_or(None))
}

fn admit(gpu: &Gpu, lanes: &[QsaCoreLane<'_>]) -> HipResult<Option<QsaCoreLanesPlan>> {
    let mut table = LaneTable::default();
    let mut common: Option<Common> = None;
    let mut pool_cfg: Option<(u64, crate::rope::RopeYarn)> = None;
    let (mut max_rows, mut pool_grid, mut select_lds, mut attention_lds) = (0, 0, 0, 0);
    // Per scratch / state family: the lane's [start, end) byte range.
    let mut families: Vec<Vec<(usize, usize)>> = vec![Vec::new(); NPTR];
    for (i, lane) in lanes.iter().enumerate() {
        let (p, s, a) = (&lane.prologue, &lane.select, &lane.attention);
        let rows = p.rows;
        let position = p.position;
        let compress = a.compress;
        let kv_width = p.kv_heads * p.head_dim;
        let index_q_width = p.index_heads * p.index_dim;
        let index_width = index_q_width + p.index_kv_width;
        let q_width = p.heads * p.head_dim;
        let blocks_l = position
            .saturating_add(rows)
            .checked_div(compress)
            .unwrap_or(0);
        // -- The per-lane launches' own admission (same predicates). ------------
        if p.format != QsaKvFormat::Fp8
            || a.format != QsaKvFormat::Fp8
            || !gpu.arch_caps.is_gfx1201()
            || !(1..=QSA_CORE_LANE_ROWS).contains(&rows)
            || s.rows != rows
            || a.rows != rows
            || s.position_start != position
            || a.position_start != position
            || compress == 0
            || s.compress != compress
            || s.block_count != blocks_l
            || position
                .checked_add(rows)
                .is_none_or(|e| e > i32::MAX as usize)
            || gpu.route_rows(rows) > gpu.few_row_limit()
            || !qsa_decode_rows_admitted(gpu, rows)
            || !qsa_decode_splitk_eager(gpu)
            || p.head_dim != 256
            || p.index_dim > 256
            || p.index_kv_width != p.index_dim
            || s.pooled_bf16.is_some()
            || p.raw_index_keys.dtype != DType::BF16
            || s.pooled.dtype != DType::BF16
        {
            return Ok(None);
        }
        // Prologue validity (`indexed_attention_decode_prologue`).
        check_qsa_format(
            gpu,
            p.format,
            p.kv_heads,
            p.head_dim,
            [p.full_keys, p.full_values],
        )?;
        let row_units = p.format.kv_row_units(p.kv_heads, p.head_dim);
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
        // Select: the fused parallel route with LDS-resident scores.
        qsa_select_batch_validate(s)?;
        let lds = s.shape_blocks.checked_mul(4);
        let parallel = gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
            && s.shape_blocks > 0
            && !select_forced_serial();
        if !parallel
            || lds.is_none_or(|b| {
                b + QSA_SELECT_BATCHED_STATIC_LDS_BYTES > QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
            })
            || qsa_select_pm_fits(gpu, s)
            || qsa_bf16_split_route(gpu, s).is_some()
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
            index_dtype: DType::BF16,
        };
        if pool_refusal(lane.pool.as_ref(), &expect).is_some() {
            return Ok(None);
        }
        // -- Lane-to-lane agreement. ---------------------------------------------
        let lane_common = Common {
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
            query_row_stride: s.query_row_stride,
            index_q_norm: ptr(p.index_q_norm),
            q_norm: ptr(p.q_norm),
            k_norm: ptr(p.k_norm),
            yarn: p.yarn,
        };
        match &common {
            None => common = Some(lane_common),
            Some(c) if *c == lane_common => {}
            Some(_) => return Ok(None),
        }
        if let Some(pool) = &lane.pool {
            let cfg = (pool.norm.map_or(0, ptr), pool.yarn);
            match pool_cfg {
                None => pool_cfg = Some(cfg),
                Some(c) if c == cfg => {}
                Some(_) => return Ok(None),
            }
            pool_grid = pool_grid.max(rows.div_ceil(compress).max(1).min(pool.grid_bound));
        }
        max_rows = max_rows.max(rows);
        select_lds = select_lds.max(lds.unwrap_or(0));
        attention_lds = attention_lds.max(qsa_attention_selected_at(a, rows - 1) * 8);
        // Table slot `i`: state, scratch (each lane's first row), scalars.
        let (pos, rows_i, blocks_i, first_block) = (
            checked_i32(position, "QSA lane position")?,
            checked_i32(rows, "QSA lane rows")?,
            checked_i32(blocks_l, "QSA lane blocks")?,
            checked_i32(position / compress, "QSA lane first block")?,
        );
        let slot = [
            ptr(p.full_keys),
            ptr(p.full_values),
            ptr(p.raw_index_keys),
            ptr(s.pooled),
            ptr(lane.persistent_selection),
            ptr(p.index_row),
            ptr(p.qgate),
            ptr(p.keys),
            ptr(p.values),
            ptr(s.selected),
            ptr(a.output),
        ];
        // Extents the lane touches in each family (bytes, half-open).
        let extent = [
            end * row_units,
            end * row_units,
            end * p.index_kv_width * 2,
            blocks_l.max(1) * p.index_kv_width * 2,
            s.capacity * std::mem::size_of::<i32>(),
            rows * index_width * 4,
            rows * 2 * q_width * 4,
            rows * kv_width * 4,
            rows * kv_width * 4,
            rows * s.capacity * std::mem::size_of::<i32>(),
            rows * q_width * 4,
        ];
        for (f, (&base, &len)) in slot.iter().zip(&extent).enumerate() {
            table.ptrs[f][i] = base;
            families[f].push((base as usize, base as usize + len));
        }
        table.ints[0][i] = pos;
        table.ints[1][i] = rows_i;
        table.ints[2][i] = blocks_i;
        table.ints[3][i] = first_block;
    }
    // Lanes never share a state tensor or a scratch row range (ptrs of the same
    // family are distinct and their touched extents disjoint).
    if !families.into_iter().all(disjoint) {
        return Ok(None);
    }
    let Some(common) = common else {
        return Ok(None);
    };
    let shape = Shape {
        lanes: lanes.len(),
        common,
        max_rows,
        pool: pool_cfg,
        pool_grid,
        select_lds,
        attention_lds,
    };
    Ok(Some(QsaCoreLanesPlan { shape, table }))
}

impl QsaCoreLanesPlan {
    /// Kernel launches [`Self::launch`] issues: prologue, select, attention,
    /// plus the pool when any lane pools.
    pub fn launch_count(&self) -> usize {
        3 + usize::from(self.shape.pool.is_some())
    }

    /// The admitted lanes as four launches (three without a pooling lane):
    /// each lane's blocks run the incumbent prologue / pool / select /
    /// attention bodies on its own pointers, position and rows, so every
    /// output and state byte equals the per-lane calls'.
    pub fn launch(self, gpu: &mut Gpu) -> HipResult<()> {
        let Self { shape, table } = self;
        let c = &shape.common;
        let n = checked_u32(shape.lanes, "QSA lane count")?;
        let max_rows = checked_u32(shape.max_rows, "QSA lane rows")?;
        let ints = |args: &mut KernargBlob, values: &[usize]| -> HipResult<()> {
            for &v in values {
                args.push_i32(checked_i32(v, "QSA lane argument")?);
            }
            Ok(())
        };

        gpu.ensure_kernel_public(QSA_LANES_MODULE, QSA_LANES_SRC, QSA_LANES_PROLOGUE_SYMBOL)?;
        let mut args = KernargBlob::new();
        for norm in [c.index_q_norm, c.q_norm, c.k_norm] {
            args.push_u64(norm);
        }
        ints(
            &mut args,
            &[
                c.index_heads,
                c.index_dim,
                c.index_kv_width,
                c.heads,
                c.kv_heads,
                c.head_dim,
            ],
        )?;
        c.yarn.push(&mut args);
        table.push(&mut args);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            QSA_LANES_PROLOGUE_SYMBOL,
            [
                checked_u32(
                    c.index_heads + c.heads + c.kv_heads + 1,
                    "QSA lane prologue blocks",
                )?,
                max_rows,
                n,
            ],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        if let Some((pool_norm, pool_yarn)) = shape.pool {
            gpu.ensure_kernel_public(QSA_LANES_MODULE, QSA_LANES_SRC, QSA_LANES_POOL_SYMBOL)?;
            let mut args = KernargBlob::new();
            args.push_u64(pool_norm);
            ints(&mut args, &[c.compress, c.index_kv_width])?;
            pool_yarn.push(&mut args);
            table.push(&mut args);
            args.pad_to(16);
            gpu.launch_blob_recorded(
                QSA_LANES_POOL_SYMBOL,
                [
                    checked_u32(shape.pool_grid, "QSA lane pool grid")?,
                    blocks(c.index_kv_width)?,
                    n,
                ],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
                crate::dispatch::ReplayLaunchBindings::NONE,
            )?;
        }

        gpu.ensure_kernel_public(QSA_LANES_MODULE, QSA_LANES_SRC, QSA_LANES_SELECT_SYMBOL)?;
        let mut args = KernargBlob::new();
        ints(
            &mut args,
            &[
                c.query_row_stride,
                c.index_heads,
                c.index_dim,
                c.budget_blocks,
                c.compress,
                c.capacity,
            ],
        )?;
        table.push(&mut args);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            QSA_LANES_SELECT_SYMBOL,
            [max_rows, 1, n],
            [QSA_SELECT_PARALLEL_THREADS, 1, 1],
            u32::try_from(shape.select_lds)
                .map_err(|_| HipError::new(0, "QSA lane select LDS exceeds u32"))?,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        gpu.ensure_kernel_public(QSA_LANES_MODULE, QSA_LANES_SRC, QSA_LANES_ATTENTION_SYMBOL)?;
        let mut args = KernargBlob::new();
        ints(
            &mut args,
            &[
                shape.max_rows,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.budget_blocks,
                c.compress,
                c.capacity,
                c.full_capacity,
            ],
        )?;
        table.push(&mut args);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            QSA_LANES_ATTENTION_SYMBOL,
            [
                checked_u32(c.heads, "QSA lane attention heads")?,
                1,
                n * max_rows,
            ],
            [256, 1, 1],
            u32::try_from(shape.attention_lds)
                .map_err(|_| HipError::new(0, "QSA lane attention LDS exceeds u32"))?,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
        Ok(())
    }

    /// Whether [`Self::launch_fused`] can run this set: exactly one row per lane (the
    /// fused core folds each lane's index path into one workgroup) and an LDS budget that
    /// leaves room for the fused kernel's static LDS. `false` keeps [`Self::launch`].
    pub fn fusable(&self) -> bool {
        self.shape.max_rows == 1
            && self.shape.select_lds + QSA_FUSE_LDS_MARGIN_BYTES
                <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
    }

    /// Kernel launches [`Self::launch_fused`] issues (the rotation is inside the second).
    pub fn fused_launch_count(&self) -> usize {
        2
    }

    /// Where each lane's rotated row lands for `rotate`, or `None` when the lanes do not
    /// tile the out-projection's rows exactly (then no rotation is fused). Per lane:
    /// `x_rot + row * k` with `row` the lane's row in `rotate.x`.
    fn rotate_targets(&self, rotate: &QsaOutRotate<'_>) -> Option<[u64; QSA_CORE_LANES_MAX]> {
        let c = &self.shape.common;
        let k = c.heads.checked_mul(c.head_dim)?;
        let k_bytes = k.checked_mul(std::mem::size_of::<f32>())?;
        let lanes = self.shape.lanes;
        if k == 0
            || rotate.rows != lanes
            || rotate.x.dtype != DType::F32
            || rotate.x_rot.dtype != DType::F32
            || rotate.x.numel() < rotate.rows.checked_mul(k)?
            || rotate.x_rot.numel() < rotate.rows.checked_mul(k)?
        {
            return None;
        }
        let (x, x_rot) = (ptr(rotate.x), ptr(rotate.x_rot));
        if x == x_rot {
            return None;
        }
        let mut targets = [0u64; QSA_CORE_LANES_MAX];
        let mut seen = [false; QSA_CORE_LANES_MAX];
        for lane in 0..lanes {
            let out = self.table.ptrs[NPTR - 1][lane];
            let offset = out.checked_sub(x)? as usize;
            if offset % k_bytes != 0 {
                return None;
            }
            let row = offset / k_bytes;
            if row >= lanes || seen[row] {
                return None;
            }
            seen[row] = true;
            targets[lane] = x_rot + (row * k_bytes) as u64;
        }
        Some(targets)
    }

    /// The admitted lanes as two launches (see `qsa_lanes_fuse.gfx1201.hip`): prologue +
    /// pool + select, then attention + `mq_rotate_x`. Only for a [`Self::fusable`] set. With
    /// `rotate` (the out-projection that follows) and lanes that tile its rows, the second
    /// launch also writes the rotated rows into `rotate.x_rot` and records
    /// `gpu.scratch.prerotated` exactly as the standalone rotation would, so the
    /// out-projection's `rotate_x_mq_batched(x, x_rot, k, rows)` is skipped. Returns whether
    /// the rotation was fused (the caller drops the memo after the out-projection).
    pub fn launch_fused(self, gpu: &mut Gpu, rotate: Option<&QsaOutRotate<'_>>) -> HipResult<bool> {
        if !self.fusable() {
            return Err(HipError::new(0, "QSA lane set is not fusable"));
        }
        let targets = rotate.and_then(|r| self.rotate_targets(r));
        let Self { shape, table } = self;
        let c = &shape.common;
        let n = checked_u32(shape.lanes, "QSA lane count")?;
        let ints = |args: &mut KernargBlob, values: &[usize]| -> HipResult<()> {
            for &v in values {
                args.push_i32(checked_i32(v, "QSA lane argument")?);
            }
            Ok(())
        };

        gpu.ensure_kernel_public(
            QSA_LANES_FUSE_MODULE,
            QSA_LANES_FUSE_SRC,
            QSA_LANES_FUSED_CORE_SYMBOL,
        )?;
        let (pool_norm, pool_yarn) = shape.pool.unwrap_or((0, c.yarn));
        let mut args = KernargBlob::new();
        for norm in [c.index_q_norm, c.q_norm, c.k_norm, pool_norm] {
            args.push_u64(norm);
        }
        ints(
            &mut args,
            &[
                c.index_heads,
                c.index_dim,
                c.index_kv_width,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.query_row_stride,
                c.budget_blocks,
                c.compress,
                c.capacity,
            ],
        )?;
        c.yarn.push(&mut args);
        pool_yarn.push(&mut args);
        table.push(&mut args);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            QSA_LANES_FUSED_CORE_SYMBOL,
            [
                checked_u32(c.heads + c.kv_heads + 2, "QSA fused core blocks")?,
                1,
                n,
            ],
            [256, 1, 1],
            u32::try_from(shape.select_lds)
                .map_err(|_| HipError::new(0, "QSA lane select LDS exceeds u32"))?,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        gpu.ensure_kernel_public(
            QSA_LANES_FUSE_MODULE,
            QSA_LANES_FUSE_SRC,
            QSA_LANES_FUSED_ATTENTION_SYMBOL,
        )?;
        gpu.ensure_mq_signs()?;
        let signs1 = gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr() as usize as u64;
        let signs2 = gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr() as usize as u64;
        let mut args = KernargBlob::new();
        ints(
            &mut args,
            &[
                shape.max_rows,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.budget_blocks,
                c.compress,
                c.capacity,
                c.full_capacity,
            ],
        )?;
        args.push_u64(signs1);
        args.push_u64(signs2);
        table.push(&mut args);
        for lane in 0..QSA_CORE_LANES_MAX {
            args.push_u64(targets.map_or(0, |t| t[lane]));
        }
        args.pad_to(16);
        gpu.launch_blob_recorded(
            QSA_LANES_FUSED_ATTENTION_SYMBOL,
            [checked_u32(c.heads, "QSA lane attention heads")?, 1, n],
            [256, 1, 1],
            u32::try_from(shape.attention_lds)
                .map_err(|_| HipError::new(0, "QSA lane attention LDS exceeds u32"))?,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
        if let (Some(rotate), Some(_)) = (rotate, targets) {
            let k = c.heads * c.head_dim;
            let (x, x_rot) = (rotate.x.buf.as_ptr(), rotate.x_rot.buf.as_ptr());
            gpu.invalidate_x_caches_for(x_rot);
            gpu.scratch.prerotated = Some((x as usize, x_rot as usize, k * rotate.rows));
            return Ok(true);
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> QsaLanesGpuFacts {
        QsaLanesGpuFacts {
            enabled: true,
            recording: false,
            capture: false,
            extent_check: false,
            select_exact: false,
        }
    }

    #[test]
    fn flag_is_off_by_default_and_exact_gfx1201_and_gfx942_only() {
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx942"] {
            let flags = crate::feature_flags::FeatureFlags::for_test(arch);
            assert!(!flags.qwen4_qsa_lanes_enabled(), "{arch}: default off");
        }
        for arch in ["gfx1201", "gfx942"] {
            let mut flags = crate::feature_flags::FeatureFlags::for_test(arch);
            flags.qwen4_qsa_lanes = true;
            assert!(flags.qwen4_qsa_lanes_enabled(), "{arch}");
        }
        for arch in ["gfx1100", "gfx1151"] {
            let mut flags = crate::feature_flags::FeatureFlags::for_test(arch);
            flags.qwen4_qsa_lanes = true;
            assert!(
                !flags.qwen4_qsa_lanes_enabled(),
                "{arch}: gfx1201 and gfx942 only"
            );
        }
    }

    #[test]
    fn gpu_refusals() {
        assert_eq!(qsa_lanes_gpu_refusal(&facts(), 2, 2), None);
        assert_eq!(qsa_lanes_gpu_refusal(&facts(), QSA_CORE_LANES_MAX, 2), None);
        assert!(qsa_lanes_gpu_refusal(&facts(), 1, 2).is_some());
        assert!(qsa_lanes_gpu_refusal(&facts(), 0, 2).is_some());
        assert!(qsa_lanes_gpu_refusal(&facts(), QSA_CORE_LANES_MAX + 1, 2).is_some());
        let mut f = facts();
        f.enabled = false;
        assert!(qsa_lanes_gpu_refusal(&f, 4, 2).is_some());
        for field in ["recording", "capture", "extent_check", "select_exact"] {
            let mut f = facts();
            match field {
                "recording" => f.recording = true,
                "capture" => f.capture = true,
                "extent_check" => f.extent_check = true,
                _ => f.select_exact = true,
            }
            assert!(qsa_lanes_gpu_refusal(&f, 4, 2).is_some(), "{field}");
        }
    }

    /// The lane route is only admitted where the per-lane attention is the
    /// per-head kernel: gfx1201 fp8 is, for every row count and policy input.
    #[test]
    fn gfx1201_fp8_attention_policy_is_per_head() {
        let flags = std::sync::Arc::new(crate::feature_flags::FeatureFlags::for_test("gfx1201"));
        let caps = crate::arch_caps::ArchCaps::new("gfx1201", flags.clone());
        for eager in [true, false] {
            assert_eq!(
                super::super::qsa_decode_attention_policy_with(
                    &flags,
                    &caps,
                    QsaKvFormat::Fp8,
                    256,
                    24,
                    2,
                    2051,
                    eager,
                ),
                QsaDecodeAttentionPolicy::PerHead
            );
        }
    }

    #[test]
    fn table_layout_matches_the_kernel_struct() {
        let mut table = LaneTable::default();
        for f in 0..NPTR {
            for l in 0..QSA_CORE_LANES_MAX {
                table.ptrs[f][l] = ((f as u64) << 8) | l as u64;
            }
        }
        for f in 0..NINT {
            for l in 0..QSA_CORE_LANES_MAX {
                table.ints[f][l] = (f * 16 + l) as i32;
            }
        }
        let mut args = KernargBlob::new();
        // Anything before the table leaves it 8-aligned (the struct's alignment).
        args.push_i32(7);
        table.push(&mut args);
        let bytes = args.as_bytes();
        assert_eq!(bytes.len(), 8 + 832);
        let u64_at = |o: usize| u64::from_ne_bytes(bytes[o..o + 8].try_into().unwrap());
        let i32_at = |o: usize| i32::from_ne_bytes(bytes[o..o + 4].try_into().unwrap());
        for (f, name) in LANE_PTR_FIELDS.iter().enumerate() {
            for l in 0..QSA_CORE_LANES_MAX {
                assert_eq!(u64_at(8 + (f * 8 + l) * 8), table.ptrs[f][l], "{name}[{l}]");
            }
        }
        for f in 0..NINT {
            for l in 0..QSA_CORE_LANES_MAX {
                assert_eq!(i32_at(8 + 704 + (f * 8 + l) * 4), table.ints[f][l]);
            }
        }
    }

    #[test]
    fn disjoint_ranges() {
        assert!(disjoint(vec![(0, 8), (8, 16), (32, 40)]));
        assert!(disjoint(vec![(16, 24), (0, 16)]));
        assert!(!disjoint(vec![(0, 9), (8, 16)]));
        assert!(!disjoint(vec![(4, 8), (4, 8)]));
    }

    #[test]
    fn pool_must_match_the_lane_scalars() {
        let expect = |position, rows, blocks| PoolExpect {
            position,
            rows,
            compress: 4,
            blocks,
            index_kv_width: 128,
            raw_keys: std::ptr::null_mut(),
            pooled: std::ptr::null_mut(),
            index_dtype: DType::BF16,
        };
        // A lane that completes a block must launch the pool; one that
        // completes none launches none.
        assert!(pool_refusal(None, &expect(10, 1, 2)).is_some());
        assert_eq!(pool_refusal(None, &expect(0, 1, 0)), None);
    }
}
