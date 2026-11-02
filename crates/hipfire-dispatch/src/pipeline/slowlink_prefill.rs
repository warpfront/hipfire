// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 slow-link layer-major prefill: the dispatch half of one MoE layer over a whole prompt
//! window.
//!
//! Over a narrow host link the incumbent's chunk-major prefill sweeps every host expert once
//! per chunk.  S1 instead runs, per layer, the mixer of every bounded tile first (the
//! architecture's business), then this module's window-wide routed tail, so each host expert
//! crosses the link once per window:
//!
//! 1. [`prepare_routes`], per tile: the incumbent's own prepare stages (normalize, input
//!    basis, router and shared projections, shared activation, route) through
//!    `execute_sealed_prepare`, then a D2D append of the tile's prepared activation, route
//!    ids/weights and shared activation into the window's [`PromptRouteStorage`].
//! 2. [`finish_route_plan`], once: the window-wide A4 quantization of the saved activations,
//!    the stable `(expert, slot)` pad16 grouping of every K2 slot, and the one count barrier.
//! 3. [`LayerRoutePlan::schedule`] + [`execute_panel`], per bounded panel of grouped rows
//!    (a panel never crosses an expert binding; a single expert is split across panels): the
//!    incumbent's K2 gate/up, down-sidecar and down GEMMs over the panel's tiles, then the
//!    ordered combine of the panel into the window accumulator (`slowlink_combine_panel`).
//!    A short-tail tile (`route_rows <= few_row_limit`) is the incumbent's K1 lane-union
//!    cohort ([`qt44_qt53_prefill::slowlink_k1_panel`](super::qt44_qt53_prefill)), run per
//!    binding against the same weight lease, never merged into the K2 grouping.
//! 4. [`finish_combine`]: the canonical terminal products, once, then the K1 cohorts' own
//!    canonical combine.
//! 5. [`finish_shared`], per tile: the saved shared activation restored and the shared down
//!    folded into the tile's rows of the window accumulator.
//!
//! # Routes and tile kinds
//!
//! A window belongs to one [`WindowRoute`], fixed by the layer's declared route, and every
//! tile to one [`TileKind`] (the arm the incumbent's own call for those rows takes):
//!
//! * `SlabIu4` (sealed expert-slab): K2 tiles run the grouped IU4 kernels over the window's A4
//!   sidecars; `route_rows <= few_row_limit` tiles are the K1 lane-union GEMV cohort.
//! * `Aggregate` (stacked-expert QT44 gate/up + QT53 down, no slab): K2 tiles
//!   ([`TileKind::AggregateWmma`], `route_rows >= 512`) run the F16 WMMA grouped arm over the
//!   window's rotated F16 activations with a panel-local unscatter + rotation and the BF16-row
//!   WMMA down; tiles below 512 rows ([`TileKind::AggregateF32Tail`]) run the incumbent's own F32
//!   routed tail per expert binding over their route ids masked to it (accumulating in place,
//!   bindings ascending); a one-row tile is the decode GEMV cohort.  All three share one lease
//!   and one sweep; a window need contain no WMMA tile.
//! * `AggregateSymIu4` (the same stacked-expert layout, resolved onto the symmetric IU4 route,
//!   default OFF): K2 tiles ([`TileKind::AggregateIu4`], `route_rows >= 512`) run the grouped
//!   NT1 IU4 kernels over the window's A4 sidecars of the saved F32 activations (the slab
//!   route's K2 arm, over the aggregate expert pointer tables of the binding), with the same
//!   F32-tail and one-row decode cohorts as `Aggregate`; the window holds no F16 activations.
//!
//! The shared combine is the same for the BF16-row K2 kinds: `slowlink_combine_panel`.
//! The arithmetic of every kernel is the incumbent's: this module adds no GEMM algorithm.
//! Nothing here records a stage-free event: retiring a panel's weights is the transport
//! owner's job.  All CPU planning (layout, scheduling, ownership) is pure and tested without
//! a device.

use super::layer_ops::hip;
use super::sealed_moe::{checked_grouped_m_total_bound, execute_sealed_prepare, SealedMoeCall};
use crate::families::moe::{MoePrefillParams, MOE_GROUPED_BLOCK_M};
use crate::types::DispatchError;
use rdna_compute::scratch::Int4MmqPrepared;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::marker::PhantomData;
use std::time::Instant;

/// Route ranks per token of the sealed Qwen4 route.
pub const K_TOP: usize = 10;
const ALIGN: usize = 256;
const F32_BYTES: usize = 4;
const I32_BYTES: usize = 4;
const BF16_BYTES: usize = 2;
const F16_BYTES: usize = 2;
/// Bytes of one `block_i4_128` record of the IU4 A4 sidecars.
const A4_RECORD_BYTES: usize = 72;
/// Width of the A4 sidecar block (`k / 128` records per row).
const A4_BLOCK_K: usize = 128;

/// The three Gpu grow-only scratch needs `[a4_gate_up, a4_down, group_scatter]` of a window of
/// `rows` tokens: the IU4 A4 sidecars (`0, 0` on the F16 aggregate route, which quantizes nothing)
/// and the grouped-scatter layout. One expression for the route layout's census and the
/// selftest's ledger.
fn window_scratch_sizes(
    route: WindowRoute,
    hidden: usize,
    mi: usize,
    n_exp: usize,
    rows: usize,
) -> [usize; 3] {
    let (a4_gate_up, a4_down) = if !route.quantizes_window() {
        (0, 0)
    } else {
        (
            (hidden / A4_BLOCK_K) * rows * A4_RECORD_BYTES,
            (mi / A4_BLOCK_K) * rows * K_TOP * A4_RECORD_BYTES,
        )
    };
    let group_scatter = rdna_compute::scratch::qwen4_moe_group_layout(rows * K_TOP, n_exp).bytes;
    [a4_gate_up, a4_down, group_scatter]
}

fn bail(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("slowlink layer-major prefill: {message}"))
}

fn mul(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_mul(b).ok_or_else(|| bail("extent overflows"))
}

fn add(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_add(b).ok_or_else(|| bail("extent overflows"))
}

fn align_up(value: usize) -> Result<usize, DispatchError> {
    Ok(add(value, ALIGN - 1)? / ALIGN * ALIGN)
}

// ───────────────────────────── limits ─────────────────────────────

/// Hard bounds of one panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelLimits {
    /// Experts one transport slot (binding) holds.
    pub experts_per_slot: usize,
    /// Grouped rows one panel may span (a multiple of the 16-row tile).
    pub max_grouped_rows: usize,
}

impl PanelLimits {
    /// The frozen S1 bounds: 32-expert slots, 2048 grouped rows.
    pub const S1: PanelLimits = PanelLimits {
        experts_per_slot: 32,
        max_grouped_rows: 2048,
    };

    pub fn validate(&self) -> Result<(), DispatchError> {
        if self.experts_per_slot == 0
            || self.max_grouped_rows == 0
            || self.max_grouped_rows % MOE_GROUPED_BLOCK_M != 0
        {
            return Err(bail(format!(
                "panel limits need experts_per_slot > 0 and a tile-aligned max_grouped_rows, got {self:?}"
            )));
        }
        Ok(())
    }

    /// Bytes of the two BF16 panel scratches a panel needs: SwiGLU rows `[rows, mi]` and down
    /// rows `[rows, down_m]`.
    pub fn scratch_bytes(&self, mi: usize, down_m: usize) -> Result<usize, DispatchError> {
        self.validate()?;
        mul(
            self.max_grouped_rows,
            add(mi, down_m)?
                .checked_mul(BF16_BYTES)
                .ok_or_else(|| bail("extent overflows"))?,
        )
    }
}

// ───────────────────────────── tile scopes ─────────────────────────────

/// The architecture's per-tile scope switch.  A tile's numerics can depend on ambient `Gpu`
/// state set per call (`route_rows`, the prefill-invariant, append-F16-MoE and trunk-IU4
/// scopes, the BF16-stream arm).  The K2 grouped panels of a window run under the ambient
/// scope of the FIRST tile (the frozen K2 arm: every K2 tile of a window shares it); the
/// steps that belong to one tile alone are bracketed with that tile's own scope:
///
/// * the K1 cohort of a short-tail tile (its input restore, id mask and K1 launches), per
///   binding, in [`execute_panel`];
/// * the tile's K1 canonical combine, in [`finish_combine`];
/// * the tile's saved-activation restore and shared down, in [`finish_shared`].
///
/// `enter` installs the scope of the tile starting at window row `row0`, `leave` restores the
/// ambient one; `leave` runs even when the bracketed step failed.
pub trait TileScopeHook {
    fn enter(&mut self, gpu: &mut Gpu, row0: usize) -> Result<(), DispatchError>;
    fn leave(&mut self, gpu: &mut Gpu, row0: usize) -> Result<(), DispatchError>;
}

/// The identity hook: every tile runs under the ambient scope (diagnostics and tests).
pub struct NoScope;

impl TileScopeHook for NoScope {
    fn enter(&mut self, _gpu: &mut Gpu, _row0: usize) -> Result<(), DispatchError> {
        Ok(())
    }
    fn leave(&mut self, _gpu: &mut Gpu, _row0: usize) -> Result<(), DispatchError> {
        Ok(())
    }
}

fn scoped<T>(
    gpu: &mut Gpu,
    hook: &mut dyn TileScopeHook,
    row0: usize,
    body: impl FnOnce(&mut Gpu) -> Result<T, DispatchError>,
) -> Result<T, DispatchError> {
    hook.enter(gpu, row0)?;
    let result = body(gpu);
    let left = hook.leave(gpu, row0);
    let value = result?;
    left?;
    Ok(value)
}

// ───────────────────────────── layout ─────────────────────────────

/// The routed lowering family of a window's layer, fixed by the layer's declared route (never
/// by a dtype or a model name).  Every tile of a window belongs to it; the tile's own numerical
/// class within it is its [`TileKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowRoute {
    /// Sealed expert-slab route: K2 tiles run the grouped IU4 kernels over the window's A4
    /// sidecars, short tails the K1 lane-union GEMV cohort.
    SlabIu4,
    /// Aggregate stacked-expert QT44/QT53 route (no slab): K2 tiles (>= the WMMA row
    /// threshold) run the F16 WMMA grouped arm over the window's rotated F16 activations,
    /// tiles below it the incumbent's F32 grouped/indexed arm as a per-binding cohort, and a
    /// one-row tile the decode GEMV cohort.
    Aggregate,
    /// The aggregate layout resolved onto the symmetric IU4 route (every routed trunk layer's
    /// headers symmetric, opt-in, descriptor present): K2 tiles (>= 512 route rows) run the
    /// grouped IU4 kernels over the window's A4 sidecars of the saved F32 activations; tiles
    /// below it and one-row tiles are the same F32-tail / decode cohorts as [`Self::Aggregate`].
    AggregateSymIu4,
}

impl WindowRoute {
    /// The receipt label (`moe_iu4.window_route`).
    pub fn label(self) -> &'static str {
        match self {
            Self::SlabIu4 => "slab-iu4",
            Self::Aggregate => "aggregate",
            Self::AggregateSymIu4 => "aggregate-sym-iu4",
        }
    }

    /// Stacked-expert (non-slab) layout: F32-tail and decode cohorts run per binding over the
    /// aggregate pointer tables.
    pub fn is_aggregate_layout(self) -> bool {
        matches!(self, Self::Aggregate | Self::AggregateSymIu4)
    }

    /// The window's K2 arm runs the grouped IU4 kernels over window-wide A4 sidecars.
    pub fn quantizes_window(self) -> bool {
        matches!(self, Self::SlabIu4 | Self::AggregateSymIu4)
    }
}

/// The numerical class of one tile of a window: which of the incumbent's lowerings its rows run.
/// A window may mix kinds of one [`WindowRoute`]; the K2 kinds ([`TileKind::SlabK2`],
/// [`TileKind::AggregateWmma`], [`TileKind::AggregateIu4`]) run window-wide grouped panels,
/// every other kind is a cohort run per expert binding against the same lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileKind {
    /// Slab K2 (grouped IU4), `route_rows > few_row_limit`.
    SlabK2,
    /// Slab K1 short tail (lane-union GEMV), `route_rows <= few_row_limit`.
    SlabK1Tail,
    /// Aggregate F16 WMMA grouped arm (`route_rows >= QWEN4_F16_WMMA_MIN_TOKENS`).
    AggregateWmma,
    /// Aggregate symmetric IU4 grouped arm (`route_rows >= QWEN4_MOE_SYM_IU4_MIN_ROWS`), the
    /// K2 tile of an [`WindowRoute::AggregateSymIu4`] window.
    AggregateIu4,
    /// Aggregate F32 grouped (or few-row indexed-down) arm below the WMMA threshold, 2.. rows.
    AggregateF32Tail,
    /// The incumbent's one-row DECODE call.
    Decode,
}

impl TileKind {
    /// Whether the tile's rows join the window-wide grouped K2 panels.
    pub fn is_k2(self) -> bool {
        matches!(
            self,
            Self::SlabK2 | Self::AggregateWmma | Self::AggregateIu4
        )
    }

    /// Whether a tile of this kind may run inside a `window` of the given route.  The F32 tail
    /// belongs to both aggregate routes and a decode tile to any (its own lowering family is
    /// checked against the window by [`PreparedChunkCall::new`]); the F16 WMMA K2 arm never runs
    /// inside an IU4 window and the IU4 K2 arm never inside an F16 one.
    pub fn compatible_with(self, window: WindowRoute) -> bool {
        match self {
            Self::SlabK2 | Self::SlabK1Tail => window == WindowRoute::SlabIu4,
            Self::AggregateWmma => window == WindowRoute::Aggregate,
            Self::AggregateIu4 => window == WindowRoute::AggregateSymIu4,
            Self::AggregateF32Tail => window.is_aggregate_layout(),
            Self::Decode => true,
        }
    }

    /// The window route the kind belongs to.
    pub fn route(self) -> Option<WindowRoute> {
        match self {
            Self::SlabK2 | Self::SlabK1Tail => Some(WindowRoute::SlabIu4),
            Self::AggregateWmma | Self::AggregateF32Tail => Some(WindowRoute::Aggregate),
            Self::AggregateIu4 => Some(WindowRoute::AggregateSymIu4),
            Self::Decode => None,
        }
    }
}

/// The numerical/cohort class a tile takes in the window's route state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileClass {
    /// Runs the K2 grouped kernels: its route ids join the window's stable grouping.
    K2,
    /// A K1 short-tail cohort tile (a slab short tail or a one-row decode call).
    K1,
    /// An aggregate F32-tail cohort tile (accumulates in place per binding).
    Tail,
}

/// The class a tile of `kind` takes in a window whose SP flag is `sp`
/// ([`PromptRouteStorage::set_sp`]).
///
/// Without SP every kind keeps its own class.  Under SP only the aggregate F16 route's tiles
/// exist: an [`TileKind::AggregateF32Tail`] tile is K2-ized (it rotates its activations to F16
/// and joins the K2 grouping, because a range-masked cohort is unsafe over sparse streamed
/// groups) and every other kind is refused (a one-row decode tile, the slab route and the IU4
/// routes never belong to an SP window).
pub fn sp_tile_class(kind: TileKind, sp: bool) -> Result<TileClass, DispatchError> {
    if !sp {
        return Ok(match kind {
            TileKind::SlabK2 | TileKind::AggregateWmma | TileKind::AggregateIu4 => TileClass::K2,
            TileKind::SlabK1Tail | TileKind::Decode => TileClass::K1,
            TileKind::AggregateF32Tail => TileClass::Tail,
        });
    }
    match kind {
        TileKind::AggregateWmma | TileKind::AggregateF32Tail => Ok(TileClass::K2),
        other => Err(bail(format!(
            "a {other:?} tile does not run inside an SP window"
        ))),
    }
}

/// Grouped rows the aggregate route's panel-local slot map holds: the S1 panel bound.
pub const AGGREGATE_PANEL_ROWS: usize = PanelLimits::S1.max_grouped_rows;

/// Rows from which an aggregate tile takes the F16 WMMA grouped arm (the incumbent's own
/// [`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`]); tiles below it are
/// [`TileKind::AggregateF32Tail`].
pub const AGGREGATE_MIN_TILE_ROWS: usize = rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS;

/// Window geometry the route storage is sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptRouteDims {
    /// Rows of the window (the prompt rows this layer sweep covers).
    pub rows: usize,
    /// Hidden width (`gate_up_k`, the routed down's `m`).
    pub hidden: usize,
    pub n_exp: usize,
    /// Expert intermediate width (`down_k`).
    pub mi: usize,
    /// Intermediate width of the shared expert, 0 when the layer has none.
    pub shared_intermediate: usize,
    /// Upper bound of the window rows lowered to the K1 cohort (slab short tails and one-row
    /// decode tiles), 0 when none.
    pub k1_rows: usize,
    /// Upper bound of the window rows of aggregate F32-tail tiles, 0 when none (and always 0
    /// on the slab route).
    pub tail_rows: usize,
    /// The routed lowering of the window's layer.
    pub route: WindowRoute,
}

/// Byte offsets of every field of the route storage (one region, 256-aligned fields).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptRouteLayout {
    dims: PromptRouteDims,
    grouped_rows: usize,
    x_norm: usize,
    route_ids: usize,
    topk_ids: usize,
    topk_weights: usize,
    sorted: usize,
    tiles: usize,
    inverse: usize,
    counts: usize,
    offsets: usize,
    accumulator: usize,
    shared_hidden: usize,
    shared_scalar: usize,
    k1_y0: usize,
    k1_ids: usize,
    k1_shared: usize,
    tail_ids: usize,
    x_f16: usize,
    local_sorted: usize,
    k1_tmp: usize,
    cohort_counts: usize,
    /// SP only (zero otherwise): byte offsets of the hybrid BF16 row table
    /// `[grouped_rows, hidden]` and of the hybrid inverse `[rows * 10]` I32.  Both are the LAST
    /// fields of the region, so every other offset equals the non-SP layout's.
    sp_rows: usize,
    sp_inverse: usize,
    /// Bytes of the hybrid row table (`grouped_rows * hidden * 2`); 0 when SP is not carved.
    sp_rows_bytes: usize,
    total: usize,
}

impl PromptRouteLayout {
    pub fn new(dims: &PromptRouteDims) -> Result<Self, DispatchError> {
        Self::build(dims, false)
    }

    /// The window layout of an SP window: [`Self::new`]'s fields at the same offsets, then the
    /// SP hybrid row table (BF16 `[H_max, hidden]`, `H_max` = the grouped-row bound of
    /// `10 * rows`) and the hybrid inverse (I32 `[rows * 10]`), carved last in the same
    /// region.  SP K2-izes the aggregate F32-tail tiles and has no cohort, so the dims must be
    /// the aggregate F16 route with `k1_rows == 0` and `tail_rows == 0`.
    pub fn with_sp_rows(dims: &PromptRouteDims) -> Result<Self, DispatchError> {
        if dims.route != WindowRoute::Aggregate || dims.k1_rows != 0 || dims.tail_rows != 0 {
            return Err(bail(format!(
                "an SP window layout is the aggregate F16 route with no K1/tail cohort: {dims:?}"
            )));
        }
        Self::build(dims, true)
    }

    fn build(dims: &PromptRouteDims, sp: bool) -> Result<Self, DispatchError> {
        let d = *dims;
        let cohort_rows = d.k1_rows.checked_add(d.tail_rows);
        if d.rows == 0
            || d.hidden == 0
            || d.n_exp == 0
            || d.mi == 0
            || cohort_rows.is_none_or(|n| n > d.rows)
            || (d.tail_rows > 0 && !d.route.is_aggregate_layout())
        {
            return Err(bail(format!("degenerate route storage dimensions {d:?}")));
        }
        if d.hidden % 8 != 0 || i32::try_from(d.rows).is_err() || d.n_exp > 1024 {
            return Err(bail(format!("unsupported route storage dimensions {d:?}")));
        }
        // The count barrier reads the K2 padded counts (i32, one word an expert) and, beside
        // them, the cohort's slot count of every expert (one word an expert, no packing), so
        // neither is bounded by the other.
        if i32::try_from(d.rows * K_TOP + (MOE_GROUPED_BLOCK_M - 1)).is_err() {
            return Err(bail(format!(
                "the window's padded slot count exceeds an i32 word (rows {})",
                d.rows
            )));
        }
        let slots = mul(d.rows, K_TOP)?;
        let grouped_rows = checked_grouped_m_total_bound(slots, d.n_exp, MOE_GROUPED_BLOCK_M)?;
        let mut next = 0usize;
        let mut field = |bytes: usize| -> Result<usize, DispatchError> {
            let at = next;
            next = add(at, align_up(bytes)?)?;
            Ok(at)
        };
        let x_norm = field(mul(mul(d.rows, d.hidden)?, F32_BYTES)?)?;
        let route_ids = field(mul(slots, I32_BYTES)?)?;
        let topk_ids = field(mul(slots, I32_BYTES)?)?;
        let topk_weights = field(mul(slots, F32_BYTES)?)?;
        let sorted = field(mul(grouped_rows, I32_BYTES)?)?;
        let tiles = field(mul(grouped_rows / MOE_GROUPED_BLOCK_M, I32_BYTES)?)?;
        let inverse = field(mul(slots, I32_BYTES)?)?;
        let counts = field(mul(d.n_exp, I32_BYTES)?)?;
        let offsets = field(mul(add(d.n_exp, 1)?, I32_BYTES)?)?;
        let accumulator = field(mul(mul(d.rows, d.hidden)?, F32_BYTES)?)?;
        let shared_hidden = field(mul(mul(d.rows, d.shared_intermediate)?, F32_BYTES)?)?;
        let shared_scalar = field(if d.shared_intermediate == 0 {
            0
        } else {
            mul(d.rows, F32_BYTES)?
        })?;
        let k1_y0 = field(mul(mul(mul(d.k1_rows, K_TOP)?, d.hidden)?, F32_BYTES)?)?;
        let k1_ids = field(mul(mul(d.k1_rows, K_TOP)?, I32_BYTES)?)?;
        let k1_shared = field(mul(
            mul(mul(d.k1_rows, 2)?, d.shared_intermediate)?,
            F32_BYTES,
        )?)?;
        // The aggregate route's own fields (empty for the slab route, whose region is
        // unchanged): the F32-tail tiles' masked-id scratch, the window's rotated F16
        // activations and the panel-local slot map (only when some row takes the WMMA arm), and
        // the decode cohort's per-binding scratch rows (one decode tile's `[10, hidden]`).
        let aggregate = d.route.is_aggregate_layout();
        let wmma = d.route == WindowRoute::Aggregate && d.rows > d.k1_rows + d.tail_rows;
        let tail_ids = field(mul(mul(d.tail_rows, K_TOP)?, I32_BYTES)?)?;
        let x_f16 = field(if wmma {
            mul(mul(d.rows, d.hidden)?, F16_BYTES)?
        } else {
            0
        })?;
        let local_sorted = field(if wmma {
            mul(AGGREGATE_PANEL_ROWS, I32_BYTES)?
        } else {
            0
        })?;
        let k1_tmp = field(if aggregate && d.k1_rows > 0 {
            mul(mul(K_TOP, d.hidden)?, F32_BYTES)?
        } else {
            0
        })?;
        // The cohort's per-expert slot counts (read back with the K2 counts at the one barrier).
        let cohort_counts = field(if d.k1_rows + d.tail_rows > 0 {
            mul(d.n_exp, I32_BYTES)?
        } else {
            0
        })?;
        // SP's own fields, last so that every field above keeps the non-SP offset (zero bytes,
        // hence no growth, when `sp` is off).
        let sp_rows_bytes = if sp {
            mul(mul(grouped_rows, d.hidden)?, BF16_BYTES)?
        } else {
            0
        };
        let sp_rows = field(sp_rows_bytes)?;
        let sp_inverse = field(if sp { mul(slots, I32_BYTES)? } else { 0 })?;
        Ok(Self {
            dims: d,
            grouped_rows,
            x_norm,
            route_ids,
            topk_ids,
            topk_weights,
            sorted,
            tiles,
            inverse,
            counts,
            offsets,
            accumulator,
            shared_hidden,
            shared_scalar,
            k1_y0,
            k1_ids,
            k1_shared,
            tail_ids,
            x_f16,
            local_sorted,
            k1_tmp,
            cohort_counts,
            sp_rows,
            sp_inverse,
            sp_rows_bytes,
            total: next,
        })
    }

    pub fn dims(&self) -> &PromptRouteDims {
        &self.dims
    }

    /// Bytes of the route storage region: [`PromptRouteStorage::alloc`] allocates exactly this.
    /// The A4 sidecars ([`Self::a4_sidecar_bytes`]) are Gpu scratch and NOT part of it.
    pub fn total_bytes(&self) -> usize {
        self.total
    }

    /// Bytes of the two IU4 A4 sidecars at window rows (`int4_mmq_x_scratch` of the gate/up
    /// activation and `qwen4_moe_down_i4_scratch` of the down activation): what the producer
    /// kernels reserve when the window's rows quantize.  Zero on the F16 aggregate route, which
    /// quantizes nothing: its window activations are the region's `x_f16` field
    /// ([`Self::x_f16_bytes`]) and its panel-local F16 down input fits the F16 X scratch the
    /// tile's own rotation already reserves.  The symmetric aggregate route charges both
    /// sidecars (gate `20 * rows * 72`, down `5 * rows * 10 * 72` at the S1 geometry) and no
    /// `x_f16`.
    pub fn a4_sidecar_bytes(&self) -> usize {
        let [gate_up, down, _] = self.scratch_allocation_sizes();
        gate_up + down
    }

    /// The window's routed lowering.
    pub fn route(&self) -> WindowRoute {
        self.dims.route
    }

    /// Aggregate route: the window's rotated F16 activations `[rows, hidden]` (inside the
    /// region, counted by [`Self::total_bytes`]); 0 on the slab route and for an aggregate
    /// window with no WMMA-arm rows (`rows == k1_rows + tail_rows`).
    pub fn x_f16_bytes(&self) -> usize {
        let d = &self.dims;
        if d.route == WindowRoute::Aggregate && d.rows > d.k1_rows + d.tail_rows {
            d.rows * d.hidden * F16_BYTES
        } else {
            0
        }
    }

    /// Bytes of the aggregate F32-tail tiles' masked-id scratch (`tail_rows * 10` I32).
    pub fn tail_ids_bytes(&self) -> usize {
        self.dims.tail_rows * K_TOP * I32_BYTES
    }

    /// Bytes of the cohort's per-expert slot count words (one I32 an expert, read back with the
    /// K2 padded counts at the one barrier); 0 for a window with no cohort rows.
    pub fn cohort_counts_bytes(&self) -> usize {
        if self.dims.k1_rows + self.dims.tail_rows > 0 {
            self.dims.n_exp * I32_BYTES
        } else {
            0
        }
    }

    /// Bytes of the grouped-scatter Gpu scratch at FULL window size: exactly what
    /// `qwen4_moe_group_layout(rows * 10, n_exp)` reserves (rank, per-chunk partial counts and
    /// chunk prefixes).  Gpu scratch, grow-only, NOT part of [`Self::total_bytes`]; the
    /// 1536-row tile's own scatter scratch may already be reserved (this is the full size, not
    /// the net growth).
    pub fn group_scatter_bytes(&self) -> usize {
        self.scratch_allocation_sizes()[2]
    }

    /// The three Gpu grow-only scratch buffers a full window reserves, in order
    /// `[a4_gate_up, a4_down, group_scatter]`: each ONE `Gpu` scratch slot grown to exactly this
    /// many bytes (`grow_scratch_buffer` mallocs exactly `needed`, no rounding or doubling).
    /// The A4 pair is `0, 0` on the aggregate route. Their sum is
    /// `charged_bytes() - total_bytes()`; [`Self::a4_sidecar_bytes`] is the first two and
    /// [`Self::group_scatter_bytes`] the third.
    pub fn scratch_allocation_sizes(&self) -> [usize; 3] {
        let d = &self.dims;
        window_scratch_sizes(d.route, d.hidden, d.mi, d.n_exp, d.rows)
    }

    /// Everything the window charges: the route region, the A4 sidecars and the grouped-scatter
    /// scratch.
    pub fn charged_bytes(&self) -> usize {
        self.total_bytes() + self.a4_sidecar_bytes() + self.group_scatter_bytes()
    }

    /// Saved prepared input, F32 `[rows, hidden]`.
    pub fn x_norm_bytes(&self) -> usize {
        self.dims.rows * self.dims.hidden * F32_BYTES
    }
    /// I32 `[rows, 10]` of the route ids and F32 `[rows, 10]` of the weights.
    pub fn topk_ids_bytes(&self) -> usize {
        self.dims.rows * K_TOP * I32_BYTES
    }
    pub fn topk_weights_bytes(&self) -> usize {
        self.dims.rows * K_TOP * F32_BYTES
    }
    /// The stable-grouping maps: sorted slots, tile ids, inverse, counts, offsets.
    pub fn route_map_bytes(&self) -> usize {
        self.grouped_rows * I32_BYTES
            + (self.grouped_rows / MOE_GROUPED_BLOCK_M) * I32_BYTES
            + self.dims.rows * K_TOP * I32_BYTES
            + self.dims.n_exp * I32_BYTES
            + (self.dims.n_exp + 1) * I32_BYTES
    }
    /// Widened BF16 combine accumulator, F32 `[rows, hidden]`.
    pub fn accumulator_bytes(&self) -> usize {
        self.dims.rows * self.dims.hidden * F32_BYTES
    }
    /// Saved shared activation (`N * I * 4`) and selector scalar (`N * 4`).
    pub fn shared_hidden_bytes(&self) -> usize {
        self.dims.rows * self.dims.shared_intermediate * F32_BYTES
    }
    pub fn shared_scalar_bytes(&self) -> usize {
        if self.dims.shared_intermediate == 0 {
            0
        } else {
            self.dims.rows * F32_BYTES
        }
    }
    /// Unweighted F32 down rows of the K1 cohort, its masked-id scratch and, per K1 row, the
    /// saved pre-activation shared gate/up (`2 * I` F32: a one-row decode tile's shared
    /// activation is fused into its routed activation and recomputed from them).
    pub fn k1_bytes(&self) -> usize {
        self.dims.k1_rows
            * (K_TOP * (self.dims.hidden * F32_BYTES + I32_BYTES)
                + 2 * self.dims.shared_intermediate * F32_BYTES)
    }
    /// Upper bound of the padded grouped rows of the window (`G`).
    pub fn grouped_rows_bound(&self) -> usize {
        self.grouped_rows
    }

    /// Bytes of the SP hybrid row table, BF16 `[grouped_rows_bound, hidden]`; 0 for a layout
    /// without SP fields ([`Self::new`]).
    pub fn sp_rows_bytes(&self) -> usize {
        self.sp_rows_bytes
    }

    /// Bytes of the SP hybrid inverse, I32 `[rows * 10]`; 0 for a layout without SP fields.
    pub fn sp_inverse_bytes(&self) -> usize {
        if self.has_sp() {
            self.dims.rows * K_TOP * I32_BYTES
        } else {
            0
        }
    }

    /// Whether the layout carries the SP fields ([`Self::with_sp_rows`]).
    pub fn has_sp(&self) -> bool {
        self.sp_rows_bytes != 0
    }
}

// ───────────────────────────── prepared tile call ─────────────────────────────

/// One tile of the window: the incumbent's sealed grouped-prefill call for those rows,
/// exactly as the incumbent builds it (rows, `route_rows`, recipe, shared bindings, scratch),
/// or, for a one-row tile whose `route_rows` is 1 (the incumbent seals a DECODE call there),
/// that decode call.  S1 runs its prepare half and its shared down, never its routed tail.
pub struct PreparedChunkCall<'a> {
    call: &'a SealedMoeCall<'a>,
    rows: usize,
    route_rows: usize,
    row0: usize,
    position: usize,
    k2: bool,
    kind: TileKind,
    route: WindowRoute,
    decode: bool,
    use_path2: bool,
    hidden: usize,
    n_exp: usize,
    mi: usize,
    down_m: usize,
    shared_intermediate: usize,
}

/// Pre-mutation admission of a one-row sealed DECODE call as a K1 tile of the window: the
/// plain sealed grouped decode (GPU top-k with the fused router/shared gate-side, the K1
/// lane-union GEMV, the canonical top-10 combine, then the shared down after it), no EP, CPU
/// split/splice, ninepath, deferred combine, fused down-last combine or routed-out partial,
/// over either a uniform MQ4 expert-slab layer or the aggregate stacked-expert QT44/QT53 route
/// (no per-expert tags; its indexed GEMVs write a row for every slot, so its cohort merges
/// per binding).  Returns the window route the tile belongs to.
fn decode_admission(
    gpu: &Gpu,
    params: &crate::families::moe::MoeParams<'_>,
) -> Result<WindowRoute, DispatchError> {
    use crate::families::moe::MoeRouteCapability;
    use rdna_compute::slab_moe::SlabBuckets;
    let refuse = |why: &str| {
        Err(bail(format!(
            "{why}; the one-row decode tile stays on the incumbent route"
        )))
    };
    let Some(policy) = params.route_policy.as_ref() else {
        return refuse("the call declares no grouped route");
    };
    let slab = policy.capability.is_slab();
    if !slab
        && !matches!(
            policy.capability,
            MoeRouteCapability::Qt44Qt53Grouped | MoeRouteCapability::Qt44Qt53GroupedSymmetric
        )
    {
        return refuse(
            "the call declares neither the sealed expert-slab nor the aggregate QT44/QT53 route",
        );
    }
    if slab && !super::slab_route::arch_supported(&gpu.arch) {
        return refuse("the sealed expert-slab route is gfx1100/gfx1151/gfx1201-only");
    }
    if params.batch_size != 1
        || params.k != K_TOP
        || params.hidden % 8 != 0
        || params.routed_down_m != params.hidden
        || params.routed_down_k != params.mi
        || (gpu.arch == "gfx1100" && params.routed_down_k != 640)
        || params.routed_down_m % 64 != 0
        || params.routed_down_k % 128 != 0
    {
        return refuse("the one-row geometry is not the sealed grouped geometry");
    }
    if !params.recipe.bf16_round_trip() || !params.recipe.shared_after_combine() {
        return refuse("the recipe is not the BF16-row combine with the shared down after it");
    }
    if params.shared.is_none() || params.skip_shared || params.routed_out.is_some() {
        return refuse("the call carries no plain shared expert");
    }
    if !slab {
        if params.expert_dtype_tags.is_some()
            || !crate::families::moe::grouped_route_geometry_supported(
                policy,
                params.n_exp,
                params.k,
                params.hidden,
                params.mi,
                params.routed_gate_up_k,
                params.routed_down_m,
                params.routed_down_k,
                params.dtypes.routed_gate_up,
                params.dtypes.routed_down,
            )
        {
            return refuse(
                "the aggregate one-row geometry or dtypes are not the sealed grouped route",
            );
        }
        return Ok(WindowRoute::Aggregate);
    }
    let layer = super::slab_route::layer(
        &gpu.arch,
        params.route_policy.as_ref(),
        &params.dtypes,
        params.expert_dtype_tags,
    )?;
    match layer.buckets {
        SlabBuckets::Uniform(bucket) if bucket != 0 => Ok(WindowRoute::SlabIu4),
        _ => refuse("the layer is not uniform MQ4"),
    }
}

/// Pre-mutation admission of a tile of `kind` and own lowering `family` into a `window`: the
/// kind must be one the window's route runs ([`TileKind::compatible_with`]) and the tile's
/// lowering family must be the window's (a one-row decode call or an F32 tail of the stacked
/// layout belongs to both aggregate routes; nothing else crosses families).
fn admit_into_window(
    kind: TileKind,
    family: WindowRoute,
    window: WindowRoute,
) -> Result<(), DispatchError> {
    if !kind.compatible_with(window) {
        return Err(bail(format!(
            "a {kind:?} tile does not run inside a {window:?} window"
        )));
    }
    let family_ok = match window {
        WindowRoute::SlabIu4 => family == WindowRoute::SlabIu4,
        WindowRoute::Aggregate => family == WindowRoute::Aggregate,
        WindowRoute::AggregateSymIu4 => matches!(
            family,
            WindowRoute::Aggregate | WindowRoute::AggregateSymIu4
        ),
    };
    if !family_ok {
        return Err(bail(format!(
            "the tile's routed lowering {family:?} differs from the window's {window:?}"
        )));
    }
    Ok(())
}

impl<'a> PreparedChunkCall<'a> {
    /// Pre-mutation admission.  `Err` means the window must stay on the incumbent route; it
    /// is returned before anything is launched or written.  `gpu.route_rows` must already be
    /// set for this tile's rows.
    pub fn new(
        gpu: &Gpu,
        call: &'a SealedMoeCall<'a>,
        row0: usize,
        position: usize,
        window: WindowRoute,
    ) -> Result<Self, DispatchError> {
        if let Some(params) = call.decode_params() {
            // `seal_moe_for_layer` seals a decode call when `route_rows == 1`.
            let family = decode_admission(gpu, params)?;
            admit_into_window(TileKind::Decode, family, window)?;
            let selection = match call.kernel_selection() {
                super::moe_program::MoeKernelSelection::Decode(selection) => selection,
                super::moe_program::MoeKernelSelection::Prefill(_) => {
                    return Err(bail("the call carries a prefill selection"))
                }
            };
            if selection.cpu_split
                || selection.cpu_deferred_experts
                || selection.down_last_combine
                || selection.ninepath_d3
                || selection.ninepath_d4
                || !selection.resolution.use_gpu_topk
                || !selection
                    .route
                    .is_some_and(crate::families::moe::MoeRouteCapability::is_sealed_grouped)
                || params.defer_routed_combine
                || params.ep_row_gather.is_some()
                || params.ep_mode != crate::families::moe::MoeEpMode::None
            {
                return Err(bail(
                    "the one-row decode call is not a plain sealed grouped decode",
                ));
            }
            let shared_intermediate = params.shared.as_ref().map_or(0, |s| s.intermediate);
            return Ok(Self {
                call,
                rows: 1,
                route_rows: gpu.route_rows(1),
                row0,
                position,
                k2: false,
                kind: TileKind::Decode,
                route: window,
                decode: true,
                use_path2: false,
                hidden: params.hidden,
                n_exp: params.n_exp,
                mi: params.mi,
                down_m: params.routed_down_m,
                shared_intermediate,
            });
        }
        let params = call
            .prefill_params()
            .ok_or_else(|| bail("the call carries no operands"))?;
        let selection = match call.kernel_selection() {
            super::moe_program::MoeKernelSelection::Prefill(selection) => selection,
            super::moe_program::MoeKernelSelection::Decode(_) => {
                return Err(bail("the call carries a decode selection"))
            }
        };
        let use_path2 = selection.uses_grouped_rows();
        let tile = super::qt44_qt53_prefill::slowlink_admission(gpu, params, use_path2)?;
        let (k2, kind, family) = match tile {
            super::qt44_qt53_prefill::SlowlinkTile::SlabK2 => {
                (true, TileKind::SlabK2, WindowRoute::SlabIu4)
            }
            super::qt44_qt53_prefill::SlowlinkTile::SlabK1 => {
                (false, TileKind::SlabK1Tail, WindowRoute::SlabIu4)
            }
            super::qt44_qt53_prefill::SlowlinkTile::AggregateWmma => {
                (true, TileKind::AggregateWmma, WindowRoute::Aggregate)
            }
            super::qt44_qt53_prefill::SlowlinkTile::AggregateIu4 => {
                (true, TileKind::AggregateIu4, WindowRoute::AggregateSymIu4)
            }
            super::qt44_qt53_prefill::SlowlinkTile::AggregateF32Tail => {
                (false, TileKind::AggregateF32Tail, WindowRoute::Aggregate)
            }
        };
        admit_into_window(kind, family, window)?;
        if selection.cpu_split || selection.cpu_deferred_experts {
            return Err(bail(
                "a CPU-split or CPU-deferred call is not a layer-major tile",
            ));
        }
        let shared_intermediate = params.prelude.shared.as_ref().map_or(0, |s| s.intermediate);
        Ok(Self {
            call,
            rows: params.batch_size,
            route_rows: gpu.route_rows(params.batch_size),
            row0,
            position,
            k2,
            kind,
            route: window,
            decode: false,
            use_path2,
            hidden: params.gate_up_k,
            n_exp: params.n_exp,
            mi: params.mi,
            down_m: params.down_m,
            shared_intermediate,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn route_rows(&self) -> usize {
        self.route_rows
    }
    pub fn row0(&self) -> usize {
        self.row0
    }
    pub fn position(&self) -> usize {
        self.position
    }
    /// Whether the tile runs the K2 grouped kernels (else the K1 short-tail cohort).
    pub fn is_k2(&self) -> bool {
        self.k2
    }
    /// Whether the tile is the incumbent's one-row DECODE call (a K1 tile whose prepare,
    /// routed K1 and shared down are the decode stages).
    pub fn is_decode(&self) -> bool {
        self.decode
    }
    /// The routed lowering of the window this tile belongs to: [`PromptRouteDims::route`] of
    /// the window's storage must equal it for every tile.
    pub fn window_route(&self) -> WindowRoute {
        self.route
    }

    /// The tile's numerical class (the arm the incumbent's own call for these rows takes).
    pub fn tile_kind(&self) -> TileKind {
        self.kind
    }

    /// The prefill operands of a prefill tile.
    fn params(&self) -> &MoePrefillParams<'_> {
        self.call
            .prefill_params()
            .expect("a prefill tile (admitted at construction)")
    }

    /// The decode operands of a one-row decode tile.
    fn decode_params(&self) -> &crate::families::moe::MoeParams<'_> {
        self.call
            .decode_params()
            .expect("a decode tile (admitted at construction)")
    }

    /// The numerical/shape signature of a tile: hidden width, expert count, expert
    /// intermediate width, routed down width, shared intermediate width (0 = none) and
    /// whether the call sealed the grouped (path-2) lowering.  The FIRST FIVE must be equal
    /// across every tile of a window.  The sixth is the K2 arm: it must be equal across the
    /// K2 tiles only, so a K1 short-tail tile (a prefill K1 call or a one-row decode call,
    /// each with its own original path) may differ from it.  Tiles may also differ in rows,
    /// `route_rows` and position.  Any other difference is refused by [`prepare_routes`]
    /// before it launches anything; the selector compares every distinct tile with
    /// [`window_arms_compatible`] before the first mutation to bypass instead.
    pub fn arm_signature(&self) -> [usize; 6] {
        [
            self.hidden,
            self.n_exp,
            self.mi,
            self.down_m,
            self.shared_intermediate,
            usize::from(self.use_path2),
        ]
    }

    /// This tile as the pre-mutation window compatibility check sees it.
    pub fn window_arm(&self) -> WindowArm {
        WindowArm {
            signature: self.arm_signature(),
            k2: self.k2,
            kind: self.kind,
        }
    }
}

/// One tile of a window for [`window_arms_compatible`]: its [`PreparedChunkCall::arm_signature`],
/// whether it runs the K2 grouped kernels and its numerical class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowArm {
    pub signature: [usize; 6],
    pub k2: bool,
    pub kind: TileKind,
}

/// Whether the tiles of a `window` (in order) are compatible: every tile's kind runs inside the
/// window's route ([`TileKind::compatible_with`]: the F16 WMMA arm never in an IU4 window, the
/// IU4 arm never in an F16 one), every tile has the first tile's geometry (the first five
/// signature values) and every K2 tile the first K2 tile's arm (the sixth).  K1 tiles, prefill
/// tails and one-row decode calls, keep their own original path.  This is exactly what
/// [`prepare_routes`] enforces tile by tile; the selector runs it before the first mutation to
/// bypass instead.
pub fn window_arms_compatible(window: WindowRoute, tiles: &[WindowArm]) -> bool {
    let Some(first) = tiles.first() else {
        return true;
    };
    let k2_arm = tiles.iter().find(|t| t.k2).map(|t| t.signature[5]);
    tiles.iter().all(|t| {
        t.kind.compatible_with(window)
            && t.signature[..5] == first.signature[..5]
            && (!t.k2 || Some(t.signature[5]) == k2_arm)
    })
}

// ───────────────────────────── storage ─────────────────────────────

#[derive(Clone, Copy, Debug)]
struct TileRec {
    row0: usize,
    rows: usize,
    k1: bool,
    /// A one-row decode tile (its K1 stack is the decode stages).
    decode: bool,
    /// First K1 row of this tile in the K1 y0 / masked-id scratch.
    k1_row0: usize,
    /// An aggregate F32-tail tile (`k1` is set too: it is a per-binding cohort tile).  Its
    /// `k1_row0` indexes the tail masked-id scratch, not the K1 y0 rows.
    tail: bool,
}

/// The window's per-layer route state, one region reused by every layer.
pub struct PromptRouteStorage {
    region: GpuTensor,
    owned: bool,
    layout: PromptRouteLayout,
    combined: GpuTensor,
    filled: usize,
    k1_used: usize,
    tail_used: usize,
    /// The first tile's signature and K2-ness: the geometry reference for every tile and
    /// the K2 arm reference for the K2 tiles.
    first_arm: Option<([usize; 6], bool)>,
    /// The K2 arm (`use_path2`) of the first K2 tile.
    k2_arm: Option<usize>,
    tiles: Vec<TileRec>,
    window_xq: Option<Int4MmqPrepared>,
    /// The window's stable grouping has run (set by [`finish_route_plan`], per layer).
    planned: bool,
    /// The window runs SP ([`Self::set_sp`]): `prepare_routes` K2-izes F32-tail tiles and
    /// refuses decode tiles.  Window-level: [`Self::reset_for_layer`] keeps it.
    sp: bool,
}

impl PromptRouteStorage {
    /// Allocate one region of exactly `layout.total_bytes()`.
    pub fn alloc(gpu: &mut Gpu, layout: &PromptRouteLayout) -> Result<Self, DispatchError> {
        let region = hip(gpu.alloc_tensor(&[layout.total_bytes()], DType::Raw))?;
        Self::over(region, true, layout)
    }

    /// Carve the storage from a region the caller owns (at least `layout.total_bytes()`
    /// bytes, 256-aligned base); the storage never frees it.
    pub fn from_region(
        region: &GpuTensor,
        layout: &PromptRouteLayout,
    ) -> Result<Self, DispatchError> {
        if region.buf.size() < layout.total_bytes() || region.buf.as_ptr() as usize % ALIGN != 0 {
            return Err(bail("the route region is too small or unaligned"));
        }
        Self::over(region.shallow_clone(), false, layout)
    }

    fn over(
        region: GpuTensor,
        owned: bool,
        layout: &PromptRouteLayout,
    ) -> Result<Self, DispatchError> {
        let combined = GpuTensor {
            buf: region
                .buf
                .byte_view(layout.accumulator, layout.accumulator_bytes()),
            shape: vec![layout.dims.rows * layout.dims.hidden],
            dtype: DType::F32,
        };
        Ok(Self {
            region,
            owned,
            layout: layout.clone(),
            combined,
            filled: 0,
            k1_used: 0,
            tail_used: 0,
            first_arm: None,
            k2_arm: None,
            tiles: Vec::new(),
            window_xq: None,
            planned: false,
            sp: false,
        })
    }

    /// Free the region (when owned).  The caller has drained every stream that reads it.
    pub fn release(self, gpu: &mut Gpu) {
        if self.owned {
            let _ = gpu.free_tensor(self.region);
        }
    }

    /// CPU-only: start a new layer (nothing filled, no plan).
    pub fn reset_for_layer(&mut self) {
        self.filled = 0;
        self.k1_used = 0;
        self.tail_used = 0;
        self.first_arm = None;
        self.k2_arm = None;
        self.tiles.clear();
        self.window_xq = None;
        self.planned = false;
    }

    pub fn rows(&self) -> usize {
        self.layout.dims.rows
    }
    pub fn filled_rows(&self) -> usize {
        self.filled
    }
    pub fn layout(&self) -> &PromptRouteLayout {
        &self.layout
    }
    /// The layer's routed+shared output, F32 `[rows, hidden]` (the tiles' `moe_output`).
    pub fn combined(&self) -> &GpuTensor {
        &self.combined
    }

    /// Arm (or disarm) the SP branch of [`prepare_routes`] for this window.  Arming requires
    /// the SP layout ([`PromptRouteLayout::with_sp_rows`]); arming a layout without the SP
    /// fields is a caller bug and panics.
    pub fn set_sp(&mut self, on: bool) {
        assert!(
            !on || self.layout.has_sp(),
            "set_sp(true) needs a PromptRouteLayout built with with_sp_rows"
        );
        self.sp = on;
    }

    /// Whether the window runs SP.
    pub fn sp(&self) -> bool {
        self.sp
    }

    /// The SP hybrid row table, BF16 `[grouped_rows_bound * hidden]` (a view into the route
    /// region; do not free).  Panics on a layout without the SP fields.
    pub fn sp_rows(&self) -> GpuTensor {
        assert!(self.layout.has_sp(), "the layout carries no SP fields");
        self.view(self.layout.sp_rows, self.layout.sp_rows_bytes, DType::BF16)
    }

    /// The SP hybrid inverse, I32 `[rows * 10]` bytes (a view into the route region).  Panics
    /// on a layout without the SP fields.
    pub fn sp_inverse(&self) -> GpuTensor {
        assert!(self.layout.has_sp(), "the layout carries no SP fields");
        self.view(
            self.layout.sp_inverse,
            self.layout.sp_inverse_bytes(),
            DType::Raw,
        )
    }

    /// The grouped slot map the stable scatter writes, raw I32 `[grouped_rows_bound]` (valid
    /// after the scatter; the SP barrier hook downloads it).
    pub fn sorted(&self) -> GpuTensor {
        self.view(
            self.layout.sorted,
            self.layout.grouped_rows * I32_BYTES,
            DType::Raw,
        )
    }

    /// The window's rotated F16 activations of rows `[row0, row0 + rows)`, F16
    /// `[rows * hidden]` (aggregate F16 route; written by `prepare_routes`).
    pub fn x_f16_rows(&self, row0: usize, rows: usize) -> Result<GpuTensor, DispatchError> {
        let d = self.layout.dims;
        if self.layout.x_f16_bytes() == 0 || row0.checked_add(rows).is_none_or(|e| e > d.rows) {
            return Err(bail("the rows are outside the window's F16 activations"));
        }
        Ok(self.view(
            self.layout.x_f16 + row0 * d.hidden * F16_BYTES,
            rows * d.hidden * F16_BYTES,
            DType::F16,
        ))
    }

    fn view(&self, offset: usize, bytes: usize, dtype: DType) -> GpuTensor {
        let shape = if dtype == DType::Raw {
            vec![bytes]
        } else {
            vec![bytes / dtype.size()]
        };
        GpuTensor {
            buf: self.region.buf.byte_view(offset, bytes),
            shape,
            dtype,
        }
    }

    fn ids_view(&self, offset: usize, row0: usize, rows: usize) -> GpuTensor {
        self.view(
            offset + row0 * K_TOP * I32_BYTES,
            rows * K_TOP * I32_BYTES,
            DType::Raw,
        )
    }
    fn f32_rows(&self, offset: usize, row0: usize, rows: usize, width: usize) -> GpuTensor {
        self.view(
            offset + row0 * width * F32_BYTES,
            rows * width * F32_BYTES,
            DType::F32,
        )
    }
}

// ───────────────────────────── prepare ─────────────────────────────

/// The decode selection of a one-row decode tile.
fn decode_selection<'c>(
    call: &'c PreparedChunkCall<'_>,
) -> &'c super::moe_program::MoeDecodeSelection {
    match call.call.kernel_selection() {
        super::moe_program::MoeKernelSelection::Decode(selection) => selection,
        super::moe_program::MoeKernelSelection::Prefill(_) => {
            unreachable!("a decode tile was admitted with a decode selection")
        }
    }
}

/// The activation tensor the decode K1 gate/up reads: the rotated local copy when the
/// resolution rotates, else the normalized input.
fn decode_activation_input<'p>(
    params: &'p crate::families::moe::MoeParams<'_>,
    selection: &super::moe_program::MoeDecodeSelection,
) -> &'p GpuTensor {
    if selection.resolution.needs_x_rot_local {
        params.x_rot_local
    } else {
        params.x_norm
    }
}

/// Run the tile's prepare stages and append its saved state to `dst` (tiles ascend and are
/// contiguous; every tile of a window shares the geometry, and every K2 tile its K2 arm).
pub fn prepare_routes(
    gpu: &mut Gpu,
    call: &PreparedChunkCall<'_>,
    dst: &mut PromptRouteStorage,
) -> Result<(), DispatchError> {
    let layout = dst.layout.clone();
    let d = layout.dims;
    if call.row0 != dst.filled {
        return Err(bail(format!(
            "tiles must be contiguous: expected row {}, got {}",
            dst.filled, call.row0
        )));
    }
    if call.rows == 0 || call.row0 + call.rows > d.rows {
        return Err(bail("the tile exceeds the window"));
    }
    if call.route != d.route {
        return Err(bail(format!(
            "the tile's routed lowering {:?} differs from the window's {:?}",
            call.route, d.route
        )));
    }
    if call.hidden != d.hidden
        || call.n_exp != d.n_exp
        || call.mi != d.mi
        || call.down_m != d.hidden
        || call.shared_intermediate != d.shared_intermediate
    {
        return Err(bail(
            "the tile's geometry differs from the window's route storage",
        ));
    }
    let sig = call.arm_signature();
    if let Some((first, _)) = dst.first_arm {
        if first[..5] != sig[..5] {
            return Err(bail("a tile's geometry differs from the window's"));
        }
    }
    if let (true, Some(arm)) = (call.k2, dst.k2_arm) {
        if arm != sig[5] {
            return Err(bail(
                "a K2 tile's numerical arm differs from the window's K2 arm",
            ));
        }
    }
    // Under SP an F32-tail tile is K2-ized (rotated F16 + route ids, no cohort row); a decode
    // tile is refused here, before anything runs.
    let class = sp_tile_class(call.kind, dst.sp)?;
    let k1 = class != TileClass::K2;
    let tail = class == TileClass::Tail;
    let k2_tail = dst.sp && call.kind == TileKind::AggregateF32Tail;
    if tail {
        if dst.tail_used + call.rows > d.tail_rows {
            return Err(bail("the F32 tail cohort exceeds its reserved rows"));
        }
    } else if k1 && dst.k1_used + call.rows > d.k1_rows {
        return Err(bail("the K1 short-tail cohort exceeds its reserved rows"));
    }
    let slots = call.rows * K_TOP;
    // The incumbent's own prepare stages, for this tile's own call.
    execute_sealed_prepare(gpu, call.call)?;
    if call.kind == TileKind::AggregateF32Tail {
        // The activation the tile's fused unscatter would fold in: produced here, once.
        super::qt44_qt53_prefill::slowlink_tail_shared_activation(
            gpu,
            call.params(),
            call.use_path2,
        )?;
    }
    // Append the tile's state to the window.
    let copy = |gpu: &Gpu, dst_off: usize, src: &GpuTensor, bytes: usize| {
        hip(gpu.memcpy_dtod_at_ordered_async(&dst.region.buf, dst_off, &src.buf, 0, bytes))
    };
    let route_ids = dst.ids_view(layout.route_ids, call.row0, call.rows);
    if call.decode {
        let params = *call.decode_params();
        let selection = decode_selection(call);
        let shared = params
            .shared
            .as_ref()
            .ok_or_else(|| bail("the decode tile carries no shared expert"))?;
        let input = decode_activation_input(&params, selection);
        copy(
            gpu,
            layout.x_norm + call.row0 * d.hidden * F32_BYTES,
            input,
            d.hidden * F32_BYTES,
        )?;
        copy(
            gpu,
            layout.topk_ids + call.row0 * K_TOP * I32_BYTES,
            params.topk_indices,
            slots * I32_BYTES,
        )?;
        copy(
            gpu,
            layout.topk_weights + call.row0 * K_TOP * F32_BYTES,
            params.topk_weights,
            slots * F32_BYTES,
        )?;
        // The pre-activation shared gate/up and selector the decode route stage left: the
        // shared activation is fused into the routed activation, so it is recomputed from
        // them when the shared down runs.
        let shared_row = layout.k1_shared + dst.k1_used * 2 * d.shared_intermediate * F32_BYTES;
        copy(
            gpu,
            shared_row,
            shared.gate_out,
            d.shared_intermediate * F32_BYTES,
        )?;
        copy(
            gpu,
            shared_row + d.shared_intermediate * F32_BYTES,
            shared.up_out,
            d.shared_intermediate * F32_BYTES,
        )?;
        copy(
            gpu,
            layout.shared_scalar + call.row0 * F32_BYTES,
            shared.scalar,
            F32_BYTES,
        )?;
        hip(gpu.slowlink_fill_bytes(&route_ids, 0xFF, slots * I32_BYTES))?;
    } else {
        let params = *call.params();
        if call.kind == TileKind::AggregateWmma || k2_tail {
            // The aggregate K2 arm reads the rotated F16 activations, the bytes the incumbent's
            // own gate/up rotation produces (`rotate_x_mq_batched_f16` is the F32 rotation
            // rounded to F16 row by row, whatever the tile's row count): rotate the tile once
            // into the shared F16 X scratch and append it to the window's `x_f16` at once, as
            // the scratch is valid only until the next F16 conversion.  No F32 copy is saved.
            let x16 = hip(gpu.rotate_x_mq_batched_f16(params.x_norm_batch, d.hidden, call.rows))?;
            copy(
                gpu,
                layout.x_f16 + call.row0 * d.hidden * F16_BYTES,
                &x16,
                call.rows * d.hidden * F16_BYTES,
            )?;
        }
        if call.kind != TileKind::AggregateWmma {
            // (A K2-ized SP tail keeps this copy too: only its F16 rows feed the K2 arm.)
            copy(
                gpu,
                layout.x_norm + call.row0 * d.hidden * F32_BYTES,
                params.x_norm_batch,
                call.rows * d.hidden * F32_BYTES,
            )?;
        }
        copy(
            gpu,
            layout.topk_ids + call.row0 * K_TOP * I32_BYTES,
            params.topk_indices,
            slots * I32_BYTES,
        )?;
        copy(
            gpu,
            layout.topk_weights + call.row0 * K_TOP * F32_BYTES,
            params.topk_weights,
            slots * F32_BYTES,
        )?;
        if k1 {
            // The cohort stays out of the K2 grouping: unrouted ids for the scatter/combine.  An
            // F32-tail tile accumulates in place per binding with the incumbent's own combine
            // (no trailing products), so its rows also carry the marker that skips the layer's
            // terminal products (`slowlink_combine_pad`).
            let fill = if tail { 0x80 } else { 0xFF };
            hip(gpu.slowlink_fill_bytes(&route_ids, fill, slots * I32_BYTES))?;
        } else {
            copy(
                gpu,
                layout.route_ids + call.row0 * K_TOP * I32_BYTES,
                params.topk_indices,
                slots * I32_BYTES,
            )?;
        }
        if let Some(shared) = params.prelude.shared.as_ref() {
            copy(
                gpu,
                layout.shared_hidden + call.row0 * d.shared_intermediate * F32_BYTES,
                shared.rotated,
                call.rows * d.shared_intermediate * F32_BYTES,
            )?;
            copy(
                gpu,
                layout.shared_scalar + call.row0 * F32_BYTES,
                shared.scalar,
                call.rows * F32_BYTES,
            )?;
        }
    }
    dst.tiles.push(TileRec {
        row0: call.row0,
        rows: call.rows,
        k1,
        decode: call.decode,
        k1_row0: if tail { dst.tail_used } else { dst.k1_used },
        tail,
    });
    if tail {
        dst.tail_used += call.rows;
    } else if k1 {
        dst.k1_used += call.rows;
    }
    if dst.first_arm.is_none() {
        dst.first_arm = Some((sig, call.k2));
    }
    if call.k2 && dst.k2_arm.is_none() {
        dst.k2_arm = Some(sig[5]);
    }
    dst.filled += call.rows;
    Ok(())
}

// ───────────────────────────── route plan ─────────────────────────────

/// What the one count barrier cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BarrierCost {
    pub sync_ms: f64,
    pub bytes_downloaded: usize,
}

/// One bounded panel of grouped rows of one expert binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelRange {
    /// Position in the schedule (the order panels must execute).
    pub index: usize,
    /// Index into the `binding_ranges` the schedule was built from.
    pub binding: usize,
    pub bind_lo: usize,
    pub bind_hi: usize,
    /// First grouped row and row count (a multiple of 16; 0 for a K1-only/pad panel).
    pub row_base: usize,
    pub rows: usize,
    /// Experts holding rows in this panel: `[expert_lo, expert_hi)`.
    pub expert_lo: usize,
    pub expert_hi: usize,
    /// Experts whose invalid-row slots this panel owns; partitions `[0, n_exp)` in order.
    pub own_lo: usize,
    pub own_hi: usize,
    /// Run the K1 cohort against this panel's binding (first panel of a binding that holds a
    /// K1 expert).
    pub k1: bool,
}

/// The layer's grouping: exact padded counts, row starts and the single barrier's cost.
#[derive(Clone, Debug)]
pub struct LayerRoutePlan {
    pub counts_padded: Vec<u32>,
    /// Grouped-row start of every expert, `[n_exp + 1]`.
    pub offsets: Vec<usize>,
    /// Padded grouped rows of the window.
    pub total_rows: usize,
    /// Experts the K1 cohort routes to (ascending, unique).
    pub k1_experts: Vec<usize>,
    pub barrier: BarrierCost,
}

impl LayerRoutePlan {
    /// The plan of device-produced padded counts (each a multiple of 16).
    pub fn from_counts(
        counts: &[i32],
        k1_experts: Vec<usize>,
        barrier: BarrierCost,
    ) -> Result<Self, DispatchError> {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0usize);
        let mut counts_padded = Vec::with_capacity(counts.len());
        for &count in counts {
            if count < 0 || count as usize % MOE_GROUPED_BLOCK_M != 0 {
                return Err(bail(format!(
                    "the stable grouping produced a count of {count}"
                )));
            }
            counts_padded.push(count as u32);
            let last = *offsets.last().unwrap();
            offsets.push(add(last, count as usize)?);
        }
        if let Some(&e) = k1_experts.iter().find(|&&e| e >= counts.len()) {
            return Err(bail(format!("K1 expert {e} is out of range")));
        }
        let total_rows = *offsets.last().unwrap();
        Ok(Self {
            counts_padded,
            offsets,
            total_rows,
            k1_experts,
            barrier,
        })
    }

    pub fn n_exp(&self) -> usize {
        self.counts_padded.len()
    }

    /// Experts with at least one routed K2 slot.
    pub fn experts_with_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.counts_padded
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(e, _)| e)
    }

    fn expert_of_row(&self, row: usize) -> usize {
        (self.offsets.partition_point(|&s| s <= row) - 1).min(self.n_exp() - 1)
    }

    /// Bounded panels over `binding_ranges` (ascending, disjoint `[lo, hi)` runs of global
    /// expert ids, one per transport binding).  Rows never cross a binding; a single expert
    /// is split across panels when it exceeds `max_grouped_rows`; a binding without rows or
    /// K1 experts emits nothing; a binding the K1 cohort needs but no K2 row does emits one
    /// zero-row panel.  Always at least one panel.
    pub fn schedule(
        &self,
        binding_ranges: &[(usize, usize)],
        limits: PanelLimits,
    ) -> Result<Vec<PanelRange>, DispatchError> {
        limits.validate()?;
        let n_exp = self.n_exp();
        if binding_ranges.is_empty() {
            return Err(bail("no expert bindings to schedule"));
        }
        let mut prev_hi = 0usize;
        for (i, &(lo, hi)) in binding_ranges.iter().enumerate() {
            if lo >= hi || hi > n_exp || (i > 0 && lo < prev_hi) {
                return Err(bail(format!(
                    "binding ranges must be ascending, disjoint and inside [0, {n_exp}): {binding_ranges:?}"
                )));
            }
            prev_hi = hi;
        }
        // Every routed expert must lie in some binding, or its rows would be dropped.
        for e in self
            .experts_with_rows()
            .chain(self.k1_experts.iter().copied())
        {
            if !binding_ranges.iter().any(|&(lo, hi)| e >= lo && e < hi) {
                return Err(bail(format!("routed expert {e} is in no binding")));
            }
        }
        let mut panels: Vec<PanelRange> = Vec::new();
        for (binding, &(lo, hi)) in binding_ranges.iter().enumerate() {
            let (start, end) = (self.offsets[lo], self.offsets[hi]);
            let k1_needed = self.k1_experts.iter().any(|&e| e >= lo && e < hi);
            let mut first = true;
            let mut base = start;
            while base < end {
                let rows = (end - base).min(limits.max_grouped_rows);
                panels.push(PanelRange {
                    index: panels.len(),
                    binding,
                    bind_lo: lo,
                    bind_hi: hi,
                    row_base: base,
                    rows,
                    expert_lo: self.expert_of_row(base),
                    expert_hi: self.expert_of_row(base + rows - 1) + 1,
                    own_lo: 0,
                    own_hi: 0,
                    k1: first && k1_needed,
                });
                first = false;
                base += rows;
            }
            if first && k1_needed {
                panels.push(PanelRange {
                    index: panels.len(),
                    binding,
                    bind_lo: lo,
                    bind_hi: hi,
                    row_base: end,
                    rows: 0,
                    expert_lo: lo,
                    expert_hi: lo,
                    own_lo: 0,
                    own_hi: 0,
                    k1: true,
                });
            }
        }
        if panels.is_empty() {
            let (lo, hi) = binding_ranges[0];
            panels.push(PanelRange {
                index: 0,
                binding: 0,
                bind_lo: lo,
                bind_hi: hi,
                row_base: 0,
                rows: 0,
                expert_lo: lo,
                expert_hi: lo,
                own_lo: 0,
                own_hi: 0,
                k1: false,
            });
        }
        // Ownership of invalid-row slots: partition [0, n_exp) in panel order, each panel
        // closing at its last row's expert, the last panel at n_exp.
        let count = panels.len();
        let mut prev = 0usize;
        for (i, panel) in panels.iter_mut().enumerate() {
            let hi = if i + 1 == count {
                n_exp
            } else if panel.rows > 0 {
                (self.expert_of_row(panel.row_base + panel.rows - 1) + 1).max(prev)
            } else {
                prev
            };
            panel.own_lo = prev;
            panel.own_hi = hi;
            prev = hi;
        }
        Ok(panels)
    }

    /// The panel's view of the window's route state.
    pub fn view<'a>(
        &'a self,
        routes: &'a PromptRouteStorage,
        panel: &'a PanelRange,
    ) -> RoutedPanelView<'a> {
        RoutedPanelView { routes, panel }
    }
}

/// The window's single stable grouping and count barrier.
pub fn finish_route_plan(
    gpu: &mut Gpu,
    routes: &mut PromptRouteStorage,
    limits: PanelLimits,
) -> Result<LayerRoutePlan, DispatchError> {
    finish_route_plan_with(gpu, routes, limits, &mut |_, _| Ok(()))
}

/// [`finish_route_plan`] with a hook that runs right after the stable scatter has been
/// enqueued and before the count barrier (the SP hook enqueues the `sorted` download on the
/// compute stream here, so the barrier's own synchronous download proves it complete).
pub fn finish_route_plan_with(
    gpu: &mut Gpu,
    routes: &mut PromptRouteStorage,
    _limits: PanelLimits,
    after_scatter: &mut dyn FnMut(&mut Gpu, &PromptRouteStorage) -> Result<(), DispatchError>,
) -> Result<LayerRoutePlan, DispatchError> {
    let layout = routes.layout.clone();
    let d = layout.dims;
    if routes.filled != d.rows {
        return Err(bail(format!(
            "the window has {} of {} rows prepared",
            routes.filled, d.rows
        )));
    }
    let slots = d.rows * K_TOP;
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    let sorted = routes.view(layout.sorted, layout.grouped_rows * I32_BYTES, DType::Raw);
    let tiles = routes.view(
        layout.tiles,
        (layout.grouped_rows / MOE_GROUPED_BLOCK_M) * I32_BYTES,
        DType::Raw,
    );
    let inverse = routes.ids_view(layout.inverse, 0, d.rows);
    let counts = routes.view(layout.counts, d.n_exp * I32_BYTES, DType::Raw);
    let offsets = routes.view(layout.offsets, (d.n_exp + 1) * I32_BYTES, DType::Raw);
    if d.route.quantizes_window() {
        // The window's activations quantize once per layer (row-wise, so independent of the
        // tile that produced a row): the saved F32 rows through the descriptor's gate producer
        // (C2).  The aggregate F16 route instead rotated its F16 activations tile by tile in
        // `prepare_routes`.
        let x_norm = routes.f32_rows(layout.x_norm, 0, d.rows, d.hidden);
        routes.window_xq = Some(hip(gpu.qwen4_moe_rotate256_i4(&x_norm, d.hidden, d.rows))?);
    }
    let scatter = if d.route == WindowRoute::SlabIu4 {
        Gpu::moe_scatter_stable_top10_slab
    } else {
        // The incumbent's own stable grouping of the aggregate routes (the arch gate admits
        // gfx1100, gfx1151 and gfx1201 here, not the slab entry).
        Gpu::moe_scatter_stable_top10
    };
    hip(scatter(
        gpu,
        &route_ids,
        &counts,
        &offsets,
        &sorted,
        &tiles,
        &inverse,
        slots,
        d.n_exp,
        layout.grouped_rows,
    ))?;
    after_scatter(gpu, &*routes)?;
    // The cohort (K1 / F32-tail / decode tiles): count its slots per expert on the device into
    // the layout's `cohort_counts` words (zeroed here), so the ONE barrier below reads both
    // count arrays and no token-route id ever reaches the host.
    let cohort = (layout.dims.k1_rows + layout.dims.tail_rows > 0)
        .then(|| routes.view(layout.cohort_counts, d.n_exp * I32_BYTES, DType::Raw));
    if let Some(cohort_counts) = cohort.as_ref() {
        hip(gpu.slowlink_fill_bytes(cohort_counts, 0, d.n_exp * I32_BYTES))?;
        for tile in routes.tiles.iter().filter(|t| t.k1) {
            let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
            hip(gpu.slowlink_fold_k1_counts(&ids, cohort_counts, tile.rows * K_TOP, d.n_exp))?;
        }
    }
    // The one count barrier: stream-ordered D2H of both count arrays on the compute stream,
    // one stream wait.
    let started = Instant::now();
    let (counts_host, cohort_host) = match cohort.as_ref() {
        Some(cohort_counts) => {
            hip(gpu.slowlink_sync_download_i32_pair(&counts, cohort_counts, d.n_exp))?
        }
        None => (
            hip(gpu.slowlink_sync_download_i32(&counts, d.n_exp))?,
            vec![0; d.n_exp],
        ),
    };
    let barrier = BarrierCost {
        sync_ms: started.elapsed().as_secs_f64() * 1e3,
        bytes_downloaded: d.n_exp * I32_BYTES * if cohort.is_some() { 2 } else { 1 },
    };
    let cohort_experts = cohort_experts(&counts_host, &cohort_host)?;
    let plan = LayerRoutePlan::from_counts(&counts_host, cohort_experts, barrier)?;
    if plan.total_rows > layout.grouped_rows {
        return Err(bail("the stable grouping exceeds its row bound"));
    }
    routes.planned = true;
    Ok(plan)
}

/// Validate the K2 padded counts and return the experts the cohort routes to (non-zero
/// cohort slot count), ascending.
fn cohort_experts(counts: &[i32], cohort: &[i32]) -> Result<Vec<usize>, DispatchError> {
    if counts.len() != cohort.len() {
        return Err(bail("the two count arrays differ in length"));
    }
    let mut experts = Vec::new();
    for (e, (&k2, &k1)) in counts.iter().zip(cohort).enumerate() {
        if k2 < 0 || k1 < 0 {
            return Err(bail(format!("a count of expert {e} is negative")));
        }
        if k1 > 0 {
            experts.push(e);
        }
    }
    Ok(experts)
}

// ───────────────────────────── weights and panels ─────────────────────────────

/// Codec identity of a panel's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelCodec {
    Mq4,
}

/// Where a binding's weights live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingSource {
    Resident,
    Staged { slot: u32 },
}

/// Immutable binding of resident or staged expert weights: `u64` pointer tables indexed by
/// GLOBAL expert id (`n_exp` entries each; only `[expert_lo, expert_hi)` are read).
#[derive(Clone, Copy)]
pub struct ExpertPanelBinding<'a> {
    pub layer: usize,
    pub epoch: u64,
    pub expert_lo: usize,
    pub expert_hi: usize,
    pub gate_up_table: &'a GpuTensor,
    pub down_table: &'a GpuTensor,
    pub codec: PanelCodec,
    pub source: BindingSource,
}

impl<'a> ExpertPanelBinding<'a> {
    /// A binding over the ascending, contiguous run of global expert ids `experts`.
    pub fn from_run(
        layer: usize,
        epoch: u64,
        experts: &[usize],
        gate_up_table: &'a GpuTensor,
        down_table: &'a GpuTensor,
        source: BindingSource,
    ) -> Result<Self, DispatchError> {
        let (&lo, &last) = match (experts.first(), experts.last()) {
            (Some(lo), Some(last)) => (lo, last),
            _ => return Err(bail("an empty expert binding")),
        };
        if experts.windows(2).any(|w| w[1] != w[0] + 1) {
            return Err(bail(
                "a binding must be an ascending contiguous run of expert ids",
            ));
        }
        Ok(Self {
            layer,
            epoch,
            expert_lo: lo,
            expert_hi: last + 1,
            gate_up_table,
            down_table,
            codec: PanelCodec::Mq4,
            source,
        })
    }

    /// Every expert of the layer resident behind one table pair.
    pub fn resident(
        layer: usize,
        n_exp: usize,
        gate_up_table: &'a GpuTensor,
        down_table: &'a GpuTensor,
    ) -> Self {
        Self {
            layer,
            epoch: 0,
            expert_lo: 0,
            expert_hi: n_exp,
            gate_up_table,
            down_table,
            codec: PanelCodec::Mq4,
            source: BindingSource::Resident,
        }
    }
}

/// A panel and the window state it reads.
#[derive(Clone, Copy)]
pub struct RoutedPanelView<'a> {
    routes: &'a PromptRouteStorage,
    panel: &'a PanelRange,
}

impl RoutedPanelView<'_> {
    pub fn panel(&self) -> &PanelRange {
        self.panel
    }
}

/// A one-row decode tile registered with the scratch: the decode operands and selection its
/// K1 stack and shared down run with.
struct DecodeTile<'a> {
    row0: usize,
    params: crate::families::moe::MoeParams<'a>,
    resolution: crate::families::moe::MoeResolution,
    route: Option<crate::families::moe::MoeRouteCapability>,
    /// The activation tensor the decode K1 gate/up reads (restored before every launch).
    input: &'a GpuTensor,
}

/// An aggregate F32-tail tile registered with the scratch: the tile's own call operands (its
/// row count, flags and scratch decide every kernel its routed tail runs).
struct TailTile<'a> {
    row0: usize,
    params: MoePrefillParams<'a>,
    use_path2: bool,
}

/// The BF16 panel scratches and the tile calls the K1 cohort borrows its scratch from.
pub struct PanelScratch<'a> {
    params: MoePrefillParams<'a>,
    decode: Vec<DecodeTile<'a>>,
    tails: Vec<TailTile<'a>>,
    limits: PanelLimits,
}

impl<'a> PanelScratch<'a> {
    /// Borrow the y scratches of `call` (the model's global grouped scratch: any PREFILL
    /// tile's call of the layer names the same tensors).
    pub fn new(
        call: &'a PreparedChunkCall<'a>,
        limits: PanelLimits,
    ) -> Result<Self, DispatchError> {
        if call.decode {
            return Err(bail(
                "the panel scratch is borrowed from a prefill tile's call, not a decode call",
            ));
        }
        let params = *call.params();
        let mi_bytes = mul(mul(limits.max_grouped_rows, params.mi)?, BF16_BYTES)?;
        let down_bytes = mul(mul(limits.max_grouped_rows, params.down_m)?, BF16_BYTES)?;
        if params.y_gate_up_grouped.buf.size() < mi_bytes
            || params.y_down_grouped.buf.size() < down_bytes
        {
            return Err(bail(format!(
                "panel scratch holds {} + {} bytes, a {}-row panel needs {mi_bytes} + {down_bytes}",
                params.y_gate_up_grouped.buf.size(),
                params.y_down_grouped.buf.size(),
                limits.max_grouped_rows
            )));
        }
        Ok(Self {
            params,
            decode: Vec::new(),
            tails: Vec::new(),
            limits,
        })
    }

    /// Register a one-row decode tile's call: its K1 stack runs with these operands against
    /// every binding.  Call once per decode tile of the layer, before the first panel.
    pub fn add_decode_tile(
        &mut self,
        call: &'a PreparedChunkCall<'a>,
    ) -> Result<(), DispatchError> {
        if !call.decode {
            return Err(bail("only a one-row decode tile registers decode operands"));
        }
        if self.decode.iter().any(|t| t.row0 == call.row0) {
            return Err(bail("the decode tile is already registered"));
        }
        let params = *call.decode_params();
        let selection = decode_selection(call);
        let input = decode_activation_input(call.decode_params(), selection);
        self.decode.push(DecodeTile {
            row0: call.row0,
            params,
            resolution: selection.resolution,
            route: selection.route,
            input,
        });
        Ok(())
    }

    /// Register an aggregate F32-tail tile's call: its routed tail runs with its own operands
    /// against every binding.  Call once per [`TileKind::AggregateF32Tail`] tile of the layer,
    /// before the first panel.
    pub fn add_tail_tile(&mut self, call: &'a PreparedChunkCall<'a>) -> Result<(), DispatchError> {
        if call.kind != TileKind::AggregateF32Tail {
            return Err(bail(
                "only an aggregate F32-tail tile registers tail operands",
            ));
        }
        if self.tails.iter().any(|t| t.row0 == call.row0) {
            return Err(bail("the F32 tail tile is already registered"));
        }
        self.tails.push(TailTile {
            row0: call.row0,
            params: *call.params(),
            use_path2: call.use_path2,
        });
        Ok(())
    }

    /// Bytes of both panel scratches at the hard bound.
    pub fn bytes(&self) -> usize {
        self.limits
            .scratch_bytes(self.params.mi, self.params.down_m)
            .unwrap_or(usize::MAX)
    }
}

/// The ordered combine's progress over one layer.
pub struct OrderedCombineCursor<'a> {
    rows: usize,
    total_rows: usize,
    expected_panels: usize,
    panels_done: usize,
    next_row_base: usize,
    combined: bool,
    _routes: PhantomData<&'a PromptRouteStorage>,
}

impl<'a> OrderedCombineCursor<'a> {
    /// Zero the window accumulator (+0.0, exactly the incumbent's cleared target) and expect
    /// `panels` in order.
    pub fn start(
        gpu: &mut Gpu,
        routes: &'a PromptRouteStorage,
        panels: &[PanelRange],
    ) -> Result<Self, DispatchError> {
        if panels.is_empty() || panels.iter().enumerate().any(|(i, p)| p.index != i) {
            return Err(bail("the schedule's panels are not a 0-based sequence"));
        }
        let d = routes.layout.dims;
        hip(gpu.slowlink_fill_bytes(&routes.combined, 0, routes.layout.accumulator_bytes()))?;
        let total_rows = panels.iter().map(|p| p.rows).sum();
        Ok(Self {
            rows: d.rows,
            total_rows,
            expected_panels: panels.len(),
            panels_done: 0,
            next_row_base: panels.iter().find(|p| p.rows > 0).map_or(0, |p| p.row_base),
            combined: false,
            _routes: PhantomData,
        })
    }

    pub fn panels_done(&self) -> usize {
        self.panels_done
    }

    pub fn is_complete(&self) -> bool {
        self.combined
    }
}

/// The K2 grouped GEMMs of one panel's rows (none for a zero-row panel), the down GEMM writing
/// its BF16 rows to `y_down` (row 0 = the panel's first row).  [`execute_panel`] passes the
/// model's grouped down scratch; the SP GPU arm passes a view of the hybrid row table.
fn panel_gemms(
    gpu: &mut Gpu,
    weights: ExpertPanelBinding<'_>,
    routes: &PromptRouteStorage,
    range: &PanelRange,
    params: &MoePrefillParams<'_>,
    y_down: &GpuTensor,
) -> Result<(), DispatchError> {
    let layout = &routes.layout;
    let d = layout.dims;
    let slots = d.rows * K_TOP;
    if range.rows > 0 {
        let tile0 = range.row_base / MOE_GROUPED_BLOCK_M;
        let tiles = routes.view(
            layout.tiles + tile0 * I32_BYTES,
            (range.rows / MOE_GROUPED_BLOCK_M) * I32_BYTES,
            DType::Raw,
        );
        let sorted = routes.view(
            layout.sorted + range.row_base * I32_BYTES,
            range.rows * I32_BYTES,
            DType::Raw,
        );
        match d.route {
            WindowRoute::SlabIu4 | WindowRoute::AggregateSymIu4 => {
                // The grouped IU4 K2 arm over the window's A4 sidecars.  The sorted/tile views
                // are this panel's, their entries GLOBAL window slot ids: the gate/up GEMM reads
                // the saved row `slot / 10` of the `d.rows`-row sidecar, the down producer
                // quantizes the panel's BF16 SwiGLU rows into their global slot records of a
                // `slots`-slot sidecar and the down GEMM reads them at `slot / 1`, however the
                // panel's own Y rows are numbered.  The aggregate route runs it over the
                // aggregate expert pointer tables of the binding (same raw MQ4 leases).
                let xq = routes
                    .window_xq
                    .as_ref()
                    .ok_or_else(|| bail("the window's route plan has not run"))?;
                hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                    weights.gate_up_table,
                    &tiles,
                    &sorted,
                    xq,
                    params.y_gate_up_grouped,
                    2 * d.mi,
                    d.hidden,
                    K_TOP,
                    range.rows,
                    d.rows,
                    false,
                ))?;
                let xdown = hip(gpu.qwen4_moe_rotate128_i4(
                    params.y_gate_up_grouped,
                    &sorted,
                    d.mi,
                    range.rows,
                    slots,
                ))?;
                hip(gpu.gemm_qwen4_moe_down_iu4_sym(
                    weights.down_table,
                    &tiles,
                    &sorted,
                    &xdown,
                    y_down,
                    d.hidden,
                    d.mi,
                    1,
                    range.rows,
                    slots,
                    false,
                ))?;
            }
            WindowRoute::Aggregate => {
                if !routes.planned || range.rows > AGGREGATE_PANEL_ROWS {
                    return Err(bail(
                        "the window's route plan has not run or the panel exceeds the aggregate \
                         panel bound",
                    ));
                }
                // The incumbent's F16 WMMA grouped route over this panel's tiles: SwiGLU in the
                // gate/up epilogue (BF16 rows), the down's own unscatter + 128-wide rotation to
                // F16, the BF16-row WMMA down.  The window's rotated activations are the region's
                // `x_f16` (indexed by the window slot / 10); the down's F16 input is panel-local
                // through a compact slot map, so the F16 X scratch holds only the panel's rows.
                let x_f16 = routes.view(layout.x_f16, routes.layout.x_f16_bytes(), DType::F16);
                hip(gpu.gemm_mq4g256v2_moe_grouped_top10_silu_bf16out(
                    weights.gate_up_table,
                    &tiles,
                    &sorted,
                    &x_f16,
                    params.y_gate_up_grouped,
                    2 * d.mi,
                    d.hidden,
                    K_TOP,
                    range.rows,
                    d.rows,
                ))?;
                let local = routes.view(layout.local_sorted, range.rows * I32_BYTES, DType::Raw);
                hip(gpu.slowlink_remap_sorted(&sorted, &local, range.rows))?;
                let x_down = hip(gpu.moe_unscatter_rotate128_f16(
                    params.y_gate_up_grouped,
                    &local,
                    d.mi,
                    range.rows,
                    range.rows,
                ))?;
                hip(gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
                    weights.down_table,
                    &tiles,
                    &local,
                    &x_down,
                    y_down,
                    d.hidden,
                    d.mi,
                    range.rows,
                    true,
                ))?;
            }
        }
    }
    Ok(())
}

/// One SP GPU-arm panel: [`execute_panel`]'s K2 GEMMs over the panel's grouped rows, the down
/// rows written to `y_rows` (BF16 `[panel.rows, hidden]`, typically a view of the hybrid row
/// table at the panel's compact placement).  No combine and no K1 cohort: the layer's
/// [`sp_finish_combine`] applies every contribution once.
pub fn execute_panel_rows(
    gpu: &mut Gpu,
    weights: ExpertPanelBinding<'_>,
    panel: RoutedPanelView<'_>,
    scratch: &mut PanelScratch<'_>,
    y_rows: &GpuTensor,
) -> Result<(), DispatchError> {
    let (routes, range) = (panel.routes, panel.panel);
    let layout = &routes.layout;
    if !routes.sp {
        return Err(bail("execute_panel_rows runs inside an SP window only"));
    }
    if range.k1 {
        return Err(bail("an SP panel carries no K1 cohort"));
    }
    if range.rows > scratch.limits.max_grouped_rows {
        return Err(bail("a panel exceeds the hard row bound"));
    }
    if range.rows % MOE_GROUPED_BLOCK_M != 0
        || range.row_base % MOE_GROUPED_BLOCK_M != 0
        || range
            .row_base
            .checked_add(range.rows)
            .is_none_or(|end| end > layout.grouped_rows)
    {
        return Err(bail(
            "a panel's rows must be 16-aligned and inside the grouped rows",
        ));
    }
    if range.rows > 0
        && !(weights.expert_lo <= range.expert_lo && range.expert_hi <= weights.expert_hi)
    {
        return Err(bail("the weight binding does not hold the panel's experts"));
    }
    let need = mul(mul(range.rows, layout.dims.hidden)?, BF16_BYTES)?;
    if y_rows.buf.size() < need {
        return Err(bail(format!(
            "the panel's row view holds {} bytes, {} rows need {need}",
            y_rows.buf.size(),
            range.rows
        )));
    }
    let params = scratch.params;
    panel_gemms(gpu, weights, routes, range, &params, y_rows)
}

/// One panel: the K2 GEMMs over the panel's grouped rows and its ordered combine, then the K1
/// cohort against the panel's binding when the schedule says so.
pub fn execute_panel(
    gpu: &mut Gpu,
    weights: ExpertPanelBinding<'_>,
    panel: RoutedPanelView<'_>,
    scratch: &mut PanelScratch<'_>,
    combine: &mut OrderedCombineCursor<'_>,
    hook: &mut dyn TileScopeHook,
) -> Result<(), DispatchError> {
    let (routes, range) = (panel.routes, panel.panel);
    let layout = &routes.layout;
    let d = layout.dims;
    if combine.combined
        || range.index != combine.panels_done
        || range.index >= combine.expected_panels
    {
        return Err(bail("panels must execute once each in schedule order"));
    }
    if range.rows > scratch.limits.max_grouped_rows {
        return Err(bail("a panel exceeds the hard row bound"));
    }
    if range.rows > 0 && range.row_base != combine.next_row_base {
        return Err(bail("panel rows are not contiguous"));
    }
    let covers = |lo: usize, hi: usize| weights.expert_lo <= lo && hi <= weights.expert_hi;
    if (range.rows > 0 && !covers(range.expert_lo, range.expert_hi))
        || (range.k1 && !covers(range.bind_lo, range.bind_hi))
    {
        return Err(bail("the weight binding does not hold the panel's experts"));
    }
    let params = scratch.params;
    panel_gemms(gpu, weights, routes, range, &params, params.y_down_grouped)?;
    // The ordered combine; a zero-row panel still applies its ownership range.
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    let weights_t = routes.f32_rows(layout.topk_weights, 0, d.rows, K_TOP);
    let inverse = routes.ids_view(layout.inverse, 0, d.rows);
    hip(gpu.slowlink_combine_panel(
        params.y_down_grouped,
        &inverse,
        &route_ids,
        &weights_t,
        &routes.combined,
        d.hidden,
        d.rows,
        d.n_exp,
        combine.total_rows,
        range.row_base,
        range.rows,
        range.own_lo,
        range.own_hi,
    ))?;
    if range.k1 {
        run_k1_cohort(
            gpu,
            &params,
            &scratch.decode,
            &scratch.tails,
            routes,
            &weights,
            range,
            hook,
        )?;
    }
    combine.panels_done += 1;
    combine.next_row_base = range.row_base + range.rows;
    Ok(())
}

/// The cohort tiles of every kind against one binding: the incumbent's K1 arithmetic of a slab
/// short tail or a one-row decode tile, or the incumbent's own F32 routed tail of an aggregate
/// tile below the WMMA threshold, each over the tile's route ids masked to the binding.
#[allow(clippy::too_many_arguments)]
fn run_k1_cohort(
    gpu: &mut Gpu,
    params: &MoePrefillParams<'_>,
    decode_tiles: &[DecodeTile<'_>],
    tail_tiles: &[TailTile<'_>],
    routes: &PromptRouteStorage,
    weights: &ExpertPanelBinding<'_>,
    range: &PanelRange,
    hook: &mut dyn TileScopeHook,
) -> Result<(), DispatchError> {
    let layout = &routes.layout;
    let d = layout.dims;
    for tile in routes.tiles.iter().filter(|t| t.k1) {
        if tile.tail {
            // The aggregate F32-tail tile: restore its prepared input and route weights into
            // the model's scratch the tile's own call names, mask its ids to the binding, and
            // run the incumbent's routed tail stages for these rows.  Its combine accumulates
            // in place into the tile's rows of the window accumulator (zero at the layer
            // start); bindings run in ascending expert order, which is the canonical one.
            let tt = tail_tiles
                .iter()
                .find(|t| t.row0 == tile.row0)
                .ok_or_else(|| bail("an F32 tail tile was not registered with the scratch"))?;
            let slots = tile.rows * K_TOP;
            let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
            let masked = routes.view(
                layout.tail_ids + tile.k1_row0 * K_TOP * I32_BYTES,
                slots * I32_BYTES,
                DType::Raw,
            );
            let target = routes
                .combined
                .sub_offset(tile.row0 * d.hidden, tile.rows * d.hidden);
            scoped(gpu, hook, tile.row0, |gpu| {
                hip(gpu.memcpy_dtod_at_ordered_async(
                    &tt.params.x_norm_batch.buf,
                    0,
                    &routes.region.buf,
                    layout.x_norm + tile.row0 * d.hidden * F32_BYTES,
                    tile.rows * d.hidden * F32_BYTES,
                ))?;
                hip(gpu.memcpy_dtod_at_ordered_async(
                    &tt.params.topk_weights.buf,
                    0,
                    &routes.region.buf,
                    layout.topk_weights + tile.row0 * K_TOP * F32_BYTES,
                    slots * F32_BYTES,
                ))?;
                hip(gpu.slowlink_mask_ids(&ids, &masked, slots, range.bind_lo, range.bind_hi))?;
                super::qt44_qt53_prefill::slowlink_f32_tail_panel(
                    gpu,
                    &tt.params,
                    tt.use_path2,
                    &masked,
                    weights.gate_up_table,
                    weights.down_table,
                    &target,
                )
            })?;
            continue;
        }
        let slots = tile.rows * K_TOP;
        let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
        let masked = routes.view(
            layout.k1_ids + tile.k1_row0 * K_TOP * I32_BYTES,
            slots * I32_BYTES,
            DType::Raw,
        );
        let y0 = routes.f32_rows(layout.k1_y0, tile.k1_row0 * K_TOP, slots, d.hidden);
        if tile.decode {
            // The one-row decode tile: the incumbent's decode K1 stack (gate/up GEMV,
            // activation with the down rotation, down GEMV) over the ids masked to this
            // binding and the binding's tables, writing its slots' unweighted rows into the
            // tile's persistent `[10, hidden]` rows.  The shared expert is not part of it
            // (`skip_shared`): its fused activation runs once, in `finish_shared`.
            let dt = decode_tiles
                .iter()
                .find(|t| t.row0 == tile.row0)
                .ok_or_else(|| bail("a one-row decode tile was not registered with the scratch"))?;
            scoped(gpu, hook, tile.row0, |gpu| {
                hip(gpu.memcpy_dtod_at_ordered_async(
                    &dt.input.buf,
                    0,
                    &routes.region.buf,
                    layout.x_norm + tile.row0 * d.hidden * F32_BYTES,
                    d.hidden * F32_BYTES,
                ))?;
                hip(gpu.slowlink_mask_ids(&ids, &masked, slots, range.bind_lo, range.bind_hi))?;
                let mut view = dt.params.with_topk_indices(&masked);
                view.expert_gate_up_ptrs = weights.gate_up_table;
                view.expert_down_ptrs = weights.down_table;
                view.skip_shared = true;
                // The slab K1 kernels leave the rows of unheld (-1) slots untouched, so they
                // write the persistent rows directly.  The aggregate indexed GEMVs write EVERY
                // slot (zero for an id outside the binding): each binding's launch goes to the
                // tile's scratch rows and only its own slots' rows are merged.
                let aggregate = d.route.is_aggregate_layout();
                let tmp = aggregate
                    .then(|| routes.view(layout.k1_tmp, slots * d.hidden * F32_BYTES, DType::F32));
                view.down_expanded = tmp.as_ref().unwrap_or(&y0);
                super::decode_gate_up_stage(gpu, &view, dt.resolution, dt.input, false, dt.route)?;
                super::decode_activation_stage(gpu, &view, dt.resolution, dt.route, None)?;
                super::decode_down_stage(
                    gpu,
                    &view,
                    dt.resolution,
                    view.x_residual,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    dt.route,
                )?;
                if let Some(tmp) = tmp.as_ref() {
                    hip(gpu.slowlink_merge_rows(
                        tmp,
                        &y0,
                        &ids,
                        slots,
                        d.hidden,
                        range.bind_lo,
                        range.bind_hi,
                    ))?;
                }
                Ok(())
            })?;
            continue;
        }
        let active = slots * d.mi * F32_BYTES;
        if params.gate_batch.buf.size() < active
            || params.up_batch.buf.size() < active
            || params.rot_batch.buf.size() < active
            || params.x_norm_batch.buf.size() < tile.rows * d.hidden * F32_BYTES
        {
            return Err(bail("the model's scratch cannot hold a K1 short-tail tile"));
        }
        let mut view = *params;
        view.batch_size = tile.rows;
        // Under the tile's own scope (its route rows, few-row limit and arm flags): restore
        // its prepared input into the model's scratch, mask its ids to the binding, run K1.
        scoped(gpu, hook, tile.row0, |gpu| {
            hip(gpu.memcpy_dtod_at_ordered_async(
                &params.x_norm_batch.buf,
                0,
                &routes.region.buf,
                layout.x_norm + tile.row0 * d.hidden * F32_BYTES,
                tile.rows * d.hidden * F32_BYTES,
            ))?;
            hip(gpu.slowlink_mask_ids(&ids, &masked, slots, range.bind_lo, range.bind_hi))?;
            super::qt44_qt53_prefill::slowlink_k1_panel(
                gpu,
                &view,
                &masked,
                weights.gate_up_table,
                weights.down_table,
                &y0,
            )
        })?;
    }
    Ok(())
}

// ───────────────────────────── SP combine ─────────────────────────────

/// The SP layer's one canonical combine: zero the accumulator (+0.0, as
/// [`OrderedCombineCursor::start`]), ONE `slowlink_combine_panel` over the hybrid row table
/// (`hybrid_rows` rows from row 0, ownership `[0, n_exp)`, the hybrid inverse) and the terminal
/// products.  The kernel applies every valid slot of a token in ascending expert id (ties by
/// rank) whichever arm produced its row, so the result is the canonical combine over the rows
/// `sp_inverse` names.  The caller has ordered the GEMM panels and the CPU-rows upload before
/// this launch.
pub fn sp_finish_combine(
    gpu: &mut Gpu,
    routes: &PromptRouteStorage,
    hybrid_rows: usize,
) -> Result<(), DispatchError> {
    let layout = &routes.layout;
    let d = layout.dims;
    if !routes.sp || !routes.planned {
        return Err(bail(
            "sp_finish_combine needs an SP window past its barrier",
        ));
    }
    if routes.filled != d.rows {
        return Err(bail("the window's rows are not all prepared"));
    }
    if hybrid_rows > layout.grouped_rows {
        return Err(bail(format!(
            "{hybrid_rows} hybrid rows exceed the grouped-row bound {}",
            layout.grouped_rows
        )));
    }
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    let weights = routes.f32_rows(layout.topk_weights, 0, d.rows, K_TOP);
    let inverse = routes.sp_inverse();
    let hybrid = routes.view(
        layout.sp_rows,
        hybrid_rows * d.hidden * BF16_BYTES,
        DType::BF16,
    );
    hip(gpu.slowlink_fill_bytes(&routes.combined, 0, layout.accumulator_bytes()))?;
    hip(gpu.slowlink_combine_panel(
        &hybrid,
        &inverse,
        &route_ids,
        &weights,
        &routes.combined,
        d.hidden,
        d.rows,
        d.n_exp,
        hybrid_rows,
        0,
        hybrid_rows,
        0,
        d.n_exp,
    ))?;
    hip(gpu.slowlink_combine_pad(&route_ids, &routes.combined, d.hidden, d.rows, d.n_exp))
}

// ───────────────────────────── finish ─────────────────────────────

/// The canonical terminal products (once), then the K1 cohort's own canonical combine.
pub fn finish_combine(
    gpu: &mut Gpu,
    routes: &PromptRouteStorage,
    combine: &mut OrderedCombineCursor<'_>,
    hook: &mut dyn TileScopeHook,
) -> Result<(), DispatchError> {
    if combine.combined || combine.panels_done != combine.expected_panels {
        return Err(bail(
            "the ordered combine is incomplete or already finished",
        ));
    }
    let layout = &routes.layout;
    let d = layout.dims;
    if combine.rows != d.rows {
        return Err(bail("the cursor belongs to another window"));
    }
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    // K1 rows carry no routed id here, so the pad leaves their (+0) accumulator unchanged.
    hip(gpu.slowlink_combine_pad(&route_ids, &routes.combined, d.hidden, d.rows, d.n_exp))?;
    // An F32-tail tile already combined in place, binding by binding, in `run_k1_cohort`.
    for tile in routes.tiles.iter().filter(|t| t.k1 && !t.tail) {
        let slots = tile.rows * K_TOP;
        let y0 = routes.f32_rows(layout.k1_y0, tile.k1_row0 * K_TOP, slots, d.hidden);
        let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
        let weights = routes.f32_rows(layout.topk_weights, tile.row0, tile.rows, K_TOP);
        let target = routes
            .combined
            .sub_offset(tile.row0 * d.hidden, tile.rows * d.hidden);
        // The tile's own scope: its canonical K1 combine under its own route rows.
        scoped(gpu, hook, tile.row0, |gpu| {
            hip(gpu.moe_down_combine_top10_batched(
                &y0, &ids, &weights, &target, d.hidden, tile.rows, d.n_exp,
            ))
        })?;
    }
    combine.combined = true;
    Ok(())
}

/// The tile's shared down: restore its saved shared activation and selector, then fold the
/// shared expert into the tile's rows of the window accumulator.
pub fn finish_shared(
    gpu: &mut Gpu,
    routes: &PromptRouteStorage,
    call: &PreparedChunkCall<'_>,
    hook: &mut dyn TileScopeHook,
) -> Result<(), DispatchError> {
    let layout = &routes.layout;
    let d = layout.dims;
    let rec = routes
        .tiles
        .iter()
        .find(|t| t.row0 == call.row0 && t.rows == call.rows)
        .ok_or_else(|| bail("the tile was not prepared in this window"))?;
    let target = routes
        .combined
        .sub_offset(rec.row0 * d.hidden, rec.rows * d.hidden);
    if call.decode {
        // The one-row decode tile: its shared activation rides in the routed activation
        // launch (and applies the selector sigmoid in place), so restore the pre-activation
        // shared gate/up and selector and run that fused launch, then the shared down.  The
        // routed half of the launch reads stale scratch and its output is dead: the routed
        // K1 stack finished before.
        let params = *call.decode_params();
        let selection = decode_selection(call);
        let shared = params
            .shared
            .as_ref()
            .ok_or_else(|| bail("the decode tile carries no shared expert"))?;
        let smi = d.shared_intermediate;
        let saved = layout.k1_shared + rec.k1_row0 * 2 * smi * F32_BYTES;
        return scoped(gpu, hook, rec.row0, |gpu| {
            hip(gpu.memcpy_dtod_at_ordered_async(
                &shared.gate_out.buf,
                0,
                &routes.region.buf,
                saved,
                smi * F32_BYTES,
            ))?;
            hip(gpu.memcpy_dtod_at_ordered_async(
                &shared.up_out.buf,
                0,
                &routes.region.buf,
                saved + smi * F32_BYTES,
                smi * F32_BYTES,
            ))?;
            hip(gpu.memcpy_dtod_at_ordered_async(
                &shared.scalar.buf,
                0,
                &routes.region.buf,
                layout.shared_scalar + rec.row0 * F32_BYTES,
                F32_BYTES,
            ))?;
            let gate = super::slice_moe_f32_view(shared.gate_out, 0, smi);
            let up = super::slice_moe_f32_view(shared.up_out, 0, smi);
            super::decode_activation_stage(
                gpu,
                &params,
                selection.resolution,
                selection.route,
                Some((&gate, &up)),
            )?;
            super::decode_shared_down_stage(
                call.call.dispatch_ctx(),
                gpu,
                &params,
                &gate,
                &up,
                &target,
                selection.router_shared_fuse,
                selection.route,
            )
        });
    }
    let params = call.params();
    scoped(gpu, hook, rec.row0, |gpu| {
        if let Some(shared) = params.prelude.shared.as_ref() {
            hip(gpu.memcpy_dtod_at_ordered_async(
                &shared.rotated.buf,
                0,
                &routes.region.buf,
                layout.shared_hidden + rec.row0 * d.shared_intermediate * F32_BYTES,
                rec.rows * d.shared_intermediate * F32_BYTES,
            ))?;
            hip(gpu.memcpy_dtod_at_ordered_async(
                &shared.scalar.buf,
                0,
                &routes.region.buf,
                layout.shared_scalar + rec.row0 * F32_BYTES,
                rec.rows * F32_BYTES,
            ))?;
        }
        super::qt44_qt53_prefill::slowlink_shared_down(gpu, params, call.use_path2, &target)
    })
}

// ───────────────────────────── real-device primitive self-tests ─────────────────────────────

/// D-owned diagnostic entrypoints for the hold's primitive stage.  Each runs the PRODUCTION
/// kernels/wrappers and [`execute_panel`] over a synthetic window on a real device and checks
/// the result against the CPU canonical model and the monolithic GPU combine.  The caller's
/// forward supplies a real [`PreparedChunkCall`] (its sealed tile call names the model's
/// scratch and geometry) and a real [`ExpertPanelBinding`] (resident or a staged lease), so
/// no private call is needed outside this crate.  Synthetic routes and values; the weights
/// are the model's.  Nothing here runs on the CPU in place of a device check.
pub mod selftest {
    use super::*;
    use rdna_compute::slab_moe::slowlink_selftest as model;

    const GUARD: usize = 4096;
    const GUARD_BYTE: u8 = 0xA5;

    /// One named check of a primitive.
    #[derive(Clone, Debug)]
    pub struct PrimitiveCheck {
        pub name: String,
        pub ok: bool,
        pub detail: String,
    }

    /// The outcome of one primitive; `ok` is the conjunction of its checks.
    #[derive(Clone, Debug)]
    pub struct PrimitiveReport {
        pub name: &'static str,
        pub ok: bool,
        pub checks: Vec<PrimitiveCheck>,
    }

    fn json_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                c if (c as u32) < 0x20 => out.push(' '),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    impl PrimitiveReport {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                ok: true,
                checks: Vec::new(),
            }
        }

        fn check(&mut self, name: &str, ok: bool, detail: impl Into<String>) {
            self.ok &= ok;
            self.checks.push(PrimitiveCheck {
                name: name.to_owned(),
                ok,
                detail: detail.into(),
            });
        }

        pub fn to_json(&self) -> String {
            let checks: Vec<String> = self
                .checks
                .iter()
                .map(|c| {
                    format!(
                        "{{\"name\":{},\"ok\":{},\"detail\":{}}}",
                        json_str(&c.name),
                        c.ok,
                        json_str(&c.detail)
                    )
                })
                .collect();
            format!(
                "{{\"primitive\":{},\"ok\":{},\"checks\":[{}]}}",
                json_str(self.name),
                self.ok,
                checks.join(",")
            )
        }
    }

    fn sync(gpu: &mut Gpu) -> Result<(), DispatchError> {
        let gpu: &Gpu = gpu;
        match gpu.active_stream.as_ref() {
            Some(stream) => hip(gpu.hip.stream_synchronize(stream)),
            None => hip(gpu.hip.device_synchronize()),
        }
    }

    /// The device operations [`diag_scope`] needs: tests substitute a fake that models the
    /// pool, so the lifecycle is exercised on the CPU.
    pub trait DiagDevice {
        /// Wait for every stream the diagnostic used; `Err` means quiescence is NOT proven.
        fn quiesce(&mut self) -> Result<(), DispatchError>;
        /// Return the buffers parked in the allocator's pool to the runtime (`hipFree`).
        fn release_parked(&mut self);
    }

    impl DiagDevice for Gpu {
        fn quiesce(&mut self) -> Result<(), DispatchError> {
            sync(self)
        }

        fn release_parked(&mut self) {
            // Only after `quiesce` succeeded: `free_tensor` parks buffers (a `DeviceBuffer`
            // has no `Drop`), so this is what returns a diagnostic's allocations to HIP.
            self.drain_pool();
        }
    }

    /// Run one diagnostic and leave nothing of it parked: the body's result is kept, then the
    /// stream is waited on and ONLY if that succeeded the parked buffers are returned.  A
    /// failed wait returns an error and frees nothing (work may still read the buffers).
    pub fn diag_scope<D: DiagDevice, T>(
        dev: &mut D,
        body: impl FnOnce(&mut D) -> Result<T, DispatchError>,
    ) -> Result<T, DispatchError> {
        let result = body(dev);
        match dev.quiesce() {
            Ok(()) => {
                dev.release_parked();
                result
            }
            Err(wait) => Err(match result {
                Ok(_) => wait,
                Err(error) => bail(format!(
                    "{error}; the stream wait after it also failed ({wait}), nothing was freed"
                )),
            }),
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn i32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn diffs(a: &[f32], b: &[f32]) -> usize {
        a.iter()
            .zip(b)
            .filter(|(x, y)| !model::same(**x, **y))
            .count()
    }

    /// The shipped canonical combine kernels (`moe_combine_order_top10` +
    /// `moe_down_combine_grouped_top10_bf16in`) over an assembled grouped-row buffer and a
    /// zero accumulator.
    #[allow(clippy::too_many_arguments)]
    fn monolithic(
        gpu: &mut Gpu,
        tokens: usize,
        hidden: usize,
        n_exp: usize,
        grouped_rows: usize,
        down: &[u16],
        inverse: &[i32],
        ids: &[i32],
        weights: &[f32],
    ) -> Result<Vec<f32>, DispatchError> {
        let down_bytes: Vec<u8> = down.iter().flat_map(|v| v.to_le_bytes()).collect();
        let down_t = hip(gpu.upload_raw(&down_bytes, &[down_bytes.len()]))?;
        let inv_t = hip(gpu.upload_raw(&i32_bytes(inverse), &[inverse.len() * 4]))?;
        let ids_t = hip(gpu.upload_raw(&i32_bytes(ids), &[ids.len() * 4]))?;
        let w_t = hip(gpu.upload_f32(weights, &[weights.len()]))?;
        let res_t = hip(gpu.upload_f32(&vec![0.0; tokens * hidden], &[tokens * hidden]))?;
        let order_t = hip(gpu.alloc_tensor(&[tokens * K_TOP * 8], DType::Raw))?;
        let result = (|| {
            hip(gpu.moe_down_combine_grouped_top10_bf16in(
                &down_t,
                &inv_t,
                &ids_t,
                &w_t,
                &res_t,
                &order_t,
                hidden,
                grouped_rows,
                tokens,
                false,
                n_exp,
            ))?;
            sync(gpu)?;
            hip(gpu.download_f32(&res_t))
        })();
        for t in [down_t, inv_t, ids_t, w_t, res_t, order_t] {
            let _ = gpu.free_tensor(t);
        }
        result
    }

    struct SynthTile {
        k1: bool,
        ids: Vec<i32>,
        weights: Vec<f32>,
        x: Vec<f32>,
    }

    impl SynthTile {
        fn rows(&self) -> usize {
            self.ids.len() / K_TOP
        }
    }

    fn synth_tile(
        rng: &mut model::Rng,
        rows: usize,
        hidden: usize,
        k1: bool,
        mut id_of: impl FnMut(usize, usize) -> i32,
    ) -> SynthTile {
        SynthTile {
            k1,
            ids: (0..rows * K_TOP)
                .map(|s| id_of(s / K_TOP, s % K_TOP))
                .collect(),
            weights: (0..rows * K_TOP).map(|_| rng.unit()).collect(),
            x: (0..rows * hidden).map(|_| rng.unit()).collect(),
        }
    }

    struct Window {
        routes: PromptRouteStorage,
        plan: LayerRoutePlan,
        panels: Vec<PanelRange>,
        acc: Vec<f32>,
        /// The K2 panels' BF16 down rows assembled by grouped row.
        down: Vec<u16>,
        inverse: Vec<i32>,
        canary_checks: usize,
        canary_violations: usize,
    }

    struct WindowOut {
        plan: LayerRoutePlan,
        panels: Vec<PanelRange>,
        acc: Vec<f32>,
        down: Vec<u16>,
        inverse: Vec<i32>,
        canary_checks: usize,
        canary_violations: usize,
    }

    fn guard_view(t: &GpuTensor, used: usize) -> Option<GpuTensor> {
        let size = t.buf.size();
        (used < size).then(|| {
            let len = GUARD.min(size - used);
            GpuTensor {
                buf: t.buf.byte_view(used, len),
                shape: vec![len],
                dtype: DType::Raw,
            }
        })
    }

    /// Seed `tiles` into `routes`, then run the production pipeline: route plan, schedule,
    /// panels (with guard canaries around both panel scratches when `canaries`), combine.
    fn run_window_inner(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        routes: &mut PromptRouteStorage,
        bindings: &[ExpertPanelBinding<'_>],
        limits: PanelLimits,
        tiles: &[SynthTile],
        canaries: bool,
        decode: Option<&PreparedChunkCall<'_>>,
    ) -> Result<WindowOut, DispatchError> {
        let layout = routes.layout.clone();
        let d = layout.dims;
        sync(gpu)?;
        let (mut row0, mut k1_used) = (0usize, 0usize);
        for tile in tiles {
            let rows = tile.rows();
            if tile.x.len() != rows * d.hidden
                || tile.weights.len() != rows * K_TOP
                || row0 + rows > d.rows
            {
                return Err(bail("a synthetic tile does not fit the window"));
            }
            let mut put =
                |view: GpuTensor, bytes: &[u8]| hip(gpu.hip.memcpy_htod(&view.buf, bytes));
            put(
                routes.f32_rows(layout.x_norm, row0, rows, d.hidden),
                &f32_bytes(&tile.x),
            )?;
            put(
                routes.ids_view(layout.topk_ids, row0, rows),
                &i32_bytes(&tile.ids),
            )?;
            put(
                routes.f32_rows(layout.topk_weights, row0, rows, K_TOP),
                &f32_bytes(&tile.weights),
            )?;
            let route_ids = if tile.k1 {
                vec![-1i32; rows * K_TOP]
            } else {
                tile.ids.clone()
            };
            put(
                routes.ids_view(layout.route_ids, row0, rows),
                &i32_bytes(&route_ids),
            )?;
            routes.tiles.push(TileRec {
                row0,
                rows,
                k1: tile.k1,
                decode: false,
                k1_row0: k1_used,
                tail: false,
            });
            if tile.k1 {
                k1_used += rows;
            }
            row0 += rows;
        }
        routes.filled = row0;
        if d.route.is_aggregate_layout() && tiles.iter().any(|t| t.k1) {
            return Err(bail("the aggregate routes have no prefill K1 tile"));
        }
        if d.route == WindowRoute::Aggregate {
            // The aggregate F16 K2 arm reads the window's rotated F16 activations, which
            // `prepare_routes` produces tile by tile; the synthetic tiles' F32 activations are
            // rotated the same way (all K2: the aggregate route has no prefill K1 tile).
            // In pieces of the aggregate route's minimum tile (512 rows, whose F16 X scratch
            // every real tile already reserves), so the diagnostic grows no Gpu scratch.
            for lo in (0..row0).step_by(AGGREGATE_MIN_TILE_ROWS) {
                let n = AGGREGATE_MIN_TILE_ROWS.min(row0 - lo);
                let x = routes.f32_rows(layout.x_norm, lo, n, d.hidden);
                let x16 = hip(gpu.rotate_x_mq_batched_f16(&x, d.hidden, n))?;
                hip(gpu.memcpy_dtod_at_ordered_async(
                    &routes.region.buf,
                    layout.x_f16 + lo * d.hidden * F16_BYTES,
                    &x16.buf,
                    0,
                    n * d.hidden * F16_BYTES,
                ))?;
            }
        }
        routes.k1_used = k1_used;
        routes.first_arm = Some((call.arm_signature(), call.k2));
        routes.k2_arm = call.k2.then(|| call.arm_signature()[5]);
        // A real decode tile (the incumbent's one-row decode call) prepares through the
        // production `prepare_routes` after the synthetic tiles.
        if let Some(decode_call) = decode {
            prepare_routes(gpu, decode_call, routes)?;
            row0 += decode_call.rows;
        }
        if row0 != d.rows {
            return Err(bail("the tiles do not fill the window"));
        }
        let plan = finish_route_plan(gpu, routes, limits)?;
        let ranges: Vec<(usize, usize)> = bindings
            .iter()
            .map(|b| (b.expert_lo, b.expert_hi))
            .collect();
        let panels = plan.schedule(&ranges, limits)?;
        let mut scratch = PanelScratch::new(call, limits)?;
        if let Some(decode_call) = decode {
            if decode_call.decode {
                scratch.add_decode_tile(decode_call)?;
            } else {
                scratch.add_tail_tile(decode_call)?;
            }
        }
        let (y_gu, y_dn) = (
            scratch.params.y_gate_up_grouped,
            scratch.params.y_down_grouped,
        );
        let mut down = vec![0u16; plan.total_rows * d.hidden];
        let (mut canary_checks, mut canary_violations) = (0usize, 0usize);
        {
            let shared: &PromptRouteStorage = routes;
            let mut cursor = OrderedCombineCursor::start(gpu, shared, &panels)?;
            for panel in &panels {
                let (gu_used, dn_used) = (
                    panel.rows * d.mi * BF16_BYTES,
                    panel.rows * d.hidden * BF16_BYTES,
                );
                let guards: Vec<GpuTensor> = if canaries && panel.rows > 0 {
                    [guard_view(y_gu, gu_used), guard_view(y_dn, dn_used)]
                        .into_iter()
                        .flatten()
                        .collect()
                } else {
                    Vec::new()
                };
                for g in &guards {
                    hip(gpu.slowlink_fill_bytes(g, GUARD_BYTE, g.buf.size()))?;
                }
                execute_panel(
                    gpu,
                    bindings[panel.binding],
                    plan.view(shared, panel),
                    &mut scratch,
                    &mut cursor,
                    &mut NoScope,
                )?;
                if panel.rows > 0 {
                    sync(gpu)?;
                    for g in &guards {
                        let bytes = hip(gpu.download_raw_bytes(g))?;
                        canary_checks += bytes.len();
                        canary_violations += bytes.iter().filter(|&&b| b != GUARD_BYTE).count();
                    }
                    let rows_view = GpuTensor {
                        buf: y_dn.buf.byte_view(0, dn_used),
                        shape: vec![dn_used],
                        dtype: DType::Raw,
                    };
                    let bytes = hip(gpu.download_raw_bytes(&rows_view))?;
                    for (i, pair) in bytes.chunks_exact(2).enumerate() {
                        down[panel.row_base * d.hidden + i] =
                            u16::from_le_bytes([pair[0], pair[1]]);
                    }
                }
            }
            finish_combine(gpu, shared, &mut cursor, &mut NoScope)?;
        }
        sync(gpu)?;
        let acc = hip(gpu.download_f32(routes.combined()))?;
        let inverse_view = routes.ids_view(layout.inverse, 0, d.rows);
        let inverse = hip(gpu.slowlink_sync_download_i32(&inverse_view, d.rows * K_TOP))?;
        Ok(WindowOut {
            plan,
            panels,
            acc,
            down,
            inverse,
            canary_checks,
            canary_violations,
        })
    }

    fn run_window(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        bindings: &[ExpertPanelBinding<'_>],
        limits: PanelLimits,
        tiles: &[SynthTile],
        canaries: bool,
        decode: Option<&PreparedChunkCall<'_>>,
    ) -> Result<Window, DispatchError> {
        // The real cohort call (a one-row decode tile or an aggregate F32-tail tile) prepares
        // through the production `prepare_routes`; it saves its shared gate/up and selector
        // (decode) or its activated shared rows (tail).
        let real_rows = decode.map_or(0, |d| d.rows);
        let real_tail = decode.is_some_and(|d| d.kind == TileKind::AggregateF32Tail);
        let rows: usize = tiles.iter().map(SynthTile::rows).sum::<usize>() + real_rows;
        let k1_rows: usize = tiles
            .iter()
            .filter(|t| t.k1)
            .map(SynthTile::rows)
            .sum::<usize>()
            + if real_tail { 0 } else { real_rows };
        let layout = PromptRouteLayout::new(&PromptRouteDims {
            rows,
            hidden: call.hidden,
            n_exp: call.n_exp,
            mi: call.mi,
            shared_intermediate: decode.map_or(0, |d| d.shared_intermediate),
            k1_rows,
            tail_rows: if real_tail { real_rows } else { 0 },
            route: call.route,
        })?;
        let mut routes = PromptRouteStorage::alloc(gpu, &layout)?;
        match run_window_inner(
            gpu,
            call,
            &mut routes,
            bindings,
            limits,
            tiles,
            canaries,
            decode,
        ) {
            Ok(out) => Ok(Window {
                routes,
                plan: out.plan,
                panels: out.panels,
                acc: out.acc,
                down: out.down,
                inverse: out.inverse,
                canary_checks: out.canary_checks,
                canary_violations: out.canary_violations,
            }),
            Err(error) => {
                let _ = sync(gpu);
                routes.release(gpu);
                Err(error)
            }
        }
    }

    /// The CPU canonical combine over the window's actual K2 grouped down rows.
    fn cpu_canonical(
        window: &Window,
        call: &PreparedChunkCall<'_>,
        ids: &[i32],
        weights: &[f32],
    ) -> Vec<f32> {
        let tokens = ids.len() / K_TOP;
        let case = model::Case {
            tokens,
            h: call.hidden,
            n_exp: call.n_exp,
            g: window.plan.total_rows,
            start: window.plan.offsets.clone(),
            down: window.down.clone(),
            inv: window.inverse.clone(),
            topk: ids.to_vec(),
            w: weights.to_vec(),
            init: vec![0.0; tokens * call.hidden],
            class: vec![0; tokens],
        };
        model::canonical(&case)
    }

    // ── the diagnostics' own memory ledger (pure; no device) ──

    /// Which diagnostic a byte ledger describes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DiagCase {
        CanonicalCombine,
        WithinExpertCanary,
        K1TailCohort,
        ZeroRowPanel,
        DecodeTailCohort,
    }

    impl DiagCase {
        /// The primitive name of the case (the gate's and the receipt's name).
        pub fn name(self) -> &'static str {
            match self {
                DiagCase::CanonicalCombine => "canonical_combine_exact",
                DiagCase::WithinExpertCanary => "within_expert_panel_canary",
                DiagCase::K1TailCohort => "k1_tail_cohort",
                DiagCase::ZeroRowPanel => "zero_row_panel",
                DiagCase::DecodeTailCohort => "decode_tail_cohort",
            }
        }

        /// The case of a primitive name, if it is a dispatch diagnostic.
        pub fn from_name(name: &str) -> Option<DiagCase> {
            DEFAULT_SUITE.into_iter().find(|c| c.name() == name)
        }
    }

    /// The dispatch half (D4) of the gate's seven named primitives, in the gate's order.
    pub const GATE_PRIMITIVES: [DiagCase; 4] = [
        DiagCase::WithinExpertCanary,
        DiagCase::CanonicalCombine,
        DiagCase::K1TailCohort,
        DiagCase::ZeroRowPanel,
    ];

    /// Every dispatch diagnostic the hold runs: the four gate primitives plus the one-row
    /// decode tail beside a K2 tile.  Callers take the case list from here, never copy it.
    pub const DEFAULT_SUITE: [DiagCase; 5] = [
        DiagCase::WithinExpertCanary,
        DiagCase::CanonicalCombine,
        DiagCase::K1TailCohort,
        DiagCase::ZeroRowPanel,
        DiagCase::DecodeTailCohort,
    ];

    /// Geometry of the call and binding the diagnostics run over.
    #[derive(Clone, Copy, Debug)]
    pub struct DiagGeometry {
        pub hidden: usize,
        pub n_exp: usize,
        /// Expert intermediate width (`mi`, the call's `down_k`).
        pub mi: usize,
        /// Experts of the supplied binding (`expert_hi - expert_lo`).
        pub binding_experts: usize,
        /// K1 tile rows (`3.min(few_row_limit)`; the diagnostics use at most 3).  On the
        /// aggregate route: the rows of the real row-0 F32-tail call `k1_tail_cohort` runs.
        pub k1_rows: usize,
        /// Intermediate width of the shared expert of the real decode call
        /// (`DecodeTailCohort`; on the aggregate route also `K1TailCohort`'s real tail call;
        /// the other diagnostics seed no shared expert).
        pub shared_intermediate: usize,
        /// The routed lowering of the call the diagnostics run over
        /// ([`PreparedChunkCall::window_route`]): it decides the window's region layout and
        /// which Gpu scratch slots a window grows.
        pub route: WindowRoute,
    }

    /// Unique-owner peak bytes of one diagnostic EXTRA to the model's own tensors: the panel
    /// y scratches and the K1 scratches belong to the call and are charged by A.  Device
    /// parts are what the diagnostic (or the Gpu scratch it grows) allocates; host parts are
    /// its `Vec`s and transient conversion/readback buffers.
    #[derive(Clone, Debug, Default)]
    pub struct DiagBytes {
        pub device_peak: usize,
        /// Case-local device bytes: `device_peak` without the grow-only Gpu scratch parts.
        pub case_local_device: usize,
        /// The grow-only Gpu scratch slots the case touches (its largest window per slot).
        pub scratch: ScratchSlots,
        pub host_peak: usize,
        pub device_parts: Vec<(&'static str, usize)>,
        pub host_parts: Vec<(&'static str, usize)>,
    }

    /// The Gpu's three grow-only scratch slots the windows touch: the gate/up A4 sidecar
    /// (`int4_mmq_x_scratch`), the down A4 sidecar (`qwen4_moe_down_i4_scratch`) and the
    /// grouped-scatter scratch.  Each slot grows to the largest need (free old, malloc new),
    /// so peak growth holds old + new.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ScratchSlots {
        pub a4_gate_up: usize,
        pub a4_down: usize,
        pub group_scatter: usize,
    }

    impl ScratchSlots {
        pub fn total(&self) -> usize {
            self.a4_gate_up + self.a4_down + self.group_scatter
        }

        fn fields(&self) -> [usize; 3] {
            [self.a4_gate_up, self.a4_down, self.group_scatter]
        }

        fn from_fields(f: [usize; 3]) -> Self {
            Self {
                a4_gate_up: f[0],
                a4_down: f[1],
                group_scatter: f[2],
            }
        }

        /// Per-slot maximum.
        pub fn max_with(&self, other: &ScratchSlots) -> ScratchSlots {
            let (a, b) = (self.fields(), other.fields());
            Self::from_fields([a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])])
        }

        /// The largest OLD slot size a growth step can free while allocating the new one:
        /// per slot, the largest need strictly below that slot's final size across `needs`;
        /// growth steps of different slots do not overlap, so the worst single one counts.
        fn second_largest_transient(&self, needs: &[ScratchSlots]) -> usize {
            let finals = self.fields();
            (0..3)
                .map(|i| {
                    needs
                        .iter()
                        .map(|n| n.fields()[i])
                        .filter(|&v| v < finals[i])
                        .max()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0)
        }
    }

    /// Suite-level device and host bytes of the diagnostics (see [`suite_diagnostic_bytes`]).
    #[derive(Clone, Debug, Default)]
    pub struct SuiteBytes {
        pub cases: Vec<(DiagCase, DiagBytes)>,
        /// Largest case-local device total (everything except the grow-only Gpu scratch): the
        /// buffers of one case, all freed and returned to HIP at its end.
        pub case_local_device_peak: usize,
        /// The grow-only slots at their per-slot maximum over the suite.
        pub persistent: ScratchSlots,
        /// The old + new overlap while a slot grows (the old size is bounded by the next
        /// smaller case's need).
        pub growth_transient: usize,
        /// Host (`Vec` and transient conversion) peak of the largest case.
        pub host_peak: usize,
    }

    impl SuiteBytes {
        /// Whole-suite device bytes when nothing of the Gpu scratch was reserved before.
        pub fn device_total(&self) -> usize {
            self.case_local_device_peak + self.persistent.total() + self.growth_transient
        }

        /// Device bytes left to budget once the admitted census already reserved `admitted`
        /// (the S1 full-window A4 and scatter slots, or whatever the product holds): a slot
        /// the diagnostics need no more than `admitted` never grows (no extra, no transient);
        /// a larger one costs its excess, and its growth transient is bounded by
        /// `max(admitted, the previous need)`.
        pub fn extra_after_admitted(&self, admitted: &ScratchSlots) -> usize {
            let (need, have) = (self.persistent.fields(), admitted.fields());
            let mut excess = 0usize;
            let mut transient = 0usize;
            for i in 0..3 {
                if need[i] > have[i] {
                    excess += need[i] - have[i];
                    transient = transient.max(have[i].max(self.growth_transient));
                }
            }
            self.case_local_device_peak + excess + transient
        }
    }

    fn sum_parts(parts: &[(&'static str, usize)]) -> usize {
        parts.iter().map(|p| p.1).sum()
    }

    /// Bytes of one synthetic window run by `run_window` (route region, grouped-scatter and A4
    /// scratch on the device; seeded tiles, assembled rows, accumulator and inverse map on the
    /// host) with `rows` tokens of which `k1_rows` are the K1 cohort and `tail_rows` the
    /// aggregate F32-tail cohort, and at most `grouped` padded K2 rows.
    #[allow(clippy::too_many_arguments)]
    fn window_bytes(
        g: &DiagGeometry,
        limits: PanelLimits,
        rows: usize,
        k1_rows: usize,
        tail_rows: usize,
        shared: usize,
        grouped: usize,
        dev: &mut Vec<(&'static str, usize)>,
        host: &mut Vec<(&'static str, usize)>,
    ) -> Result<(), DispatchError> {
        let layout = PromptRouteLayout::new(&PromptRouteDims {
            rows,
            hidden: g.hidden,
            n_exp: g.n_exp,
            mi: g.mi,
            shared_intermediate: shared,
            k1_rows,
            tail_rows,
            route: g.route,
        })?;
        let slots = rows * K_TOP;
        dev.push(("route_region", layout.total_bytes()));
        dev.push((
            "grouped_scatter_scratch (Gpu, grow-only)",
            rdna_compute::scratch::qwen4_moe_group_layout(slots, g.n_exp).bytes,
        ));
        dev.push(("a4_sidecars (Gpu, grow-only)", layout.a4_sidecar_bytes()));
        // Host: the seeded tiles (x, ids, weights) stay alive for the whole run; `f32_bytes`
        // adds a transient copy of x while it uploads.
        host.push(("tiles_x_f32", rows * g.hidden * F32_BYTES));
        host.push(("tiles_ids_and_weights", 2 * slots * I32_BYTES));
        host.push(("x_upload_transient", rows * g.hidden * F32_BYTES));
        host.push(("assembled_down_bf16", grouped * g.hidden * BF16_BYTES));
        host.push(("accumulator_f32", rows * g.hidden * F32_BYTES));
        host.push(("inverse_map", slots * I32_BYTES));
        host.push((
            "panel_readback_transient",
            limits.max_grouped_rows.min(grouped.max(1)) * g.hidden * BF16_BYTES,
        ));
        Ok(())
    }

    /// Host bytes of [`cpu_canonical`] (its `Case` clones the assembled rows, map, ids,
    /// weights and a zero init) plus its output.
    fn cpu_canonical_bytes(g: &DiagGeometry, rows: usize, grouped: usize) -> usize {
        let slots = rows * K_TOP;
        grouped * g.hidden * BF16_BYTES
            + (g.n_exp + 1) * std::mem::size_of::<usize>()
            + 3 * slots * I32_BYTES
            + rows * g.hidden * F32_BYTES
            + rows * std::mem::size_of::<usize>()
            + rows * g.hidden * F32_BYTES
    }

    /// Device and host bytes of `monolithic`: the down rows, inverse, ids, weights, zero
    /// accumulator and rank-order scratch uploaded at once; host conversion buffers and the
    /// readback.
    fn monolithic_bytes(g: &DiagGeometry, rows: usize, grouped: usize) -> (usize, usize) {
        let slots = rows * K_TOP;
        let device = grouped * g.hidden * BF16_BYTES
            + 3 * slots * I32_BYTES
            + rows * g.hidden * F32_BYTES
            + slots * 8;
        let host = grouped * g.hidden * BF16_BYTES
            + 2 * slots * I32_BYTES
            + 2 * rows * g.hidden * F32_BYTES;
        (device, host)
    }

    /// Authoritative unique-owner peak of one diagnostic, from the source above (device
    /// bytes the diagnostic or the Gpu scratch it grows allocates; host `Vec`s and transient
    /// buffers).  Cases that run windows in sequence take the maximum, not the sum.
    pub fn diagnostic_bytes(
        case: DiagCase,
        g: &DiagGeometry,
        limits: PanelLimits,
    ) -> Result<DiagBytes, DispatchError> {
        limits.validate()?;
        if g.binding_experts == 0 || g.hidden == 0 || g.n_exp == 0 {
            return Err(bail("empty diagnostic geometry"));
        }
        let active = g.binding_experts.min(g.n_exp);
        let upper = |slots: usize| {
            (slots + MOE_GROUPED_BLOCK_M.saturating_sub(1) * active.min(slots)) / 16 * 16
        };
        let mut out = DiagBytes::default();
        match case {
            DiagCase::CanonicalCombine => {
                let c = model::make_case(0, 6);
                let (t, h, rows) = (c.tokens, c.h, c.g);
                out.device_parts = vec![
                    ("down_rows_bf16", rows * h * BF16_BYTES),
                    ("inverse_ids_weights", 3 * t * K_TOP * I32_BYTES),
                    ("residual_and_accumulator_f32", 2 * t * h * F32_BYTES),
                    ("order_scratch", t * K_TOP * 8),
                ];
                out.host_parts = vec![
                    (
                        "case_rows_map_ids_weights",
                        rows * h * BF16_BYTES + 3 * t * K_TOP * I32_BYTES,
                    ),
                    ("case_init_and_cpu_canonical", 2 * t * h * F32_BYTES),
                    (
                        "upload_byte_copies",
                        rows * h * BF16_BYTES + 2 * t * K_TOP * I32_BYTES,
                    ),
                    (
                        "monolithic_reversed_and_panelled_results",
                        3 * t * h * F32_BYTES,
                    ),
                    ("init_bytes", t * h * F32_BYTES),
                ];
            }
            DiagCase::WithinExpertCanary => {
                let rows = limits.max_grouped_rows;
                let grouped = upper(rows * K_TOP);
                let mut dev = Vec::new();
                let mut host = Vec::new();
                window_bytes(g, limits, rows, 0, 0, 0, grouped, &mut dev, &mut host)?;
                let (mono_dev, mono_host) = monolithic_bytes(g, rows, grouped);
                // Device: the window stays alive while the monolithic reference runs.
                dev.push(("monolithic_reference", mono_dev));
                // Host: the call keeps ids/weights clones; `want` stays alive across the
                // monolithic step, whose transient buffers come after the CPU model's.
                host.push(("ids_weights_clones", 2 * rows * K_TOP * I32_BYTES));
                host.push(("cpu_canonical_want", rows * g.hidden * F32_BYTES));
                let cpu_phase = cpu_canonical_bytes(g, rows, grouped) - rows * g.hidden * F32_BYTES;
                host.push((
                    "max(cpu_model_case_clones, monolithic_host_transients)",
                    cpu_phase.max(mono_host),
                ));
                out.device_parts = dev;
                out.host_parts = host;
            }
            DiagCase::K1TailCohort if g.route.is_aggregate_layout() => {
                // The aggregate route's cohort primitive: one real F32-tail call (`k1_rows`
                // rows, shared expert of `shared_intermediate`) is the whole window; its
                // reference is the incumbent's unmasked routed tail into a zero target.
                let rows = g.k1_rows;
                let mut dev = Vec::new();
                let mut host = Vec::new();
                window_bytes(
                    g,
                    limits,
                    rows,
                    0,
                    rows,
                    g.shared_intermediate,
                    0,
                    &mut dev,
                    &mut host,
                )?;
                dev.push(("tail_reference_target_f32", rows * g.hidden * F32_BYTES));
                host.push(("tail_reference_readback", rows * g.hidden * F32_BYTES));
                host.push(("accumulator_readback", rows * g.hidden * F32_BYTES));
                out.device_parts = dev;
                out.host_parts = host;
            }
            DiagCase::K1TailCohort => {
                let k1 = g.k1_rows;
                let rows = 64 + k1;
                let grouped = upper(64 * K_TOP);
                let mut dev = Vec::new();
                let mut host = Vec::new();
                window_bytes(g, limits, rows, k1, 0, 0, grouped, &mut dev, &mut host)?;
                dev.push((
                    "k1_reference_y0_and_accumulator",
                    k1 * K_TOP * g.hidden * F32_BYTES + k1 * g.hidden * F32_BYTES,
                ));
                host.push((
                    "cpu_model_case_clones_and_output",
                    cpu_canonical_bytes(g, rows, grouped),
                ));
                host.push(("k1_reference_readback", k1 * g.hidden * F32_BYTES));
                out.device_parts = dev;
                out.host_parts = host;
            }
            DiagCase::ZeroRowPanel => {
                // Window A (nothing routes, 40 tokens) is released before window B (K1-only);
                // the larger of the two is the peak.
                let k1 = g.k1_rows;
                let (mut dev_a, mut host_a) = (Vec::new(), Vec::new());
                window_bytes(g, limits, 40, 0, 0, 0, 0, &mut dev_a, &mut host_a)?;
                let (mut dev_b, mut host_b) = (Vec::new(), Vec::new());
                // The aggregate route has no synthetic K1-only window (its cohort windows hold
                // a real tail/decode call, run by the K1/decode primitives).
                if g.route == WindowRoute::SlabIu4 {
                    window_bytes(g, limits, k1, k1, 0, 0, 0, &mut dev_b, &mut host_b)?;
                }
                dev_b.push((
                    "k1_reference_y0_and_accumulator",
                    k1 * K_TOP * g.hidden * F32_BYTES + k1 * g.hidden * F32_BYTES,
                ));
                host_b.push(("k1_reference_readback", k1 * g.hidden * F32_BYTES));
                if sum_parts(&dev_a) >= sum_parts(&dev_b) {
                    out.device_parts = dev_a;
                } else {
                    out.device_parts = dev_b;
                }
                out.host_parts = if sum_parts(&host_a) >= sum_parts(&host_b) {
                    host_a
                } else {
                    host_b
                };
            }
            DiagCase::DecodeTailCohort => {
                // A K2 anchor tile of 64 tokens plus the real one-row decode call, over a binding
                // that may be ALL experts (the grouped padding then spreads over every expert).
                let k1 = 1usize;
                let rows = DECODE_TAIL_ANCHOR_ROWS + k1;
                let grouped = upper(DECODE_TAIL_ANCHOR_ROWS * K_TOP);
                let mut dev = Vec::new();
                let mut host = Vec::new();
                window_bytes(
                    g,
                    limits,
                    rows,
                    k1,
                    0,
                    g.shared_intermediate,
                    grouped,
                    &mut dev,
                    &mut host,
                )?;
                dev.push(("decode_reference_target_f32", g.hidden * F32_BYTES));
                host.push((
                    "cpu_model_case_clones_and_output",
                    cpu_canonical_bytes(g, rows, grouped),
                ));
                host.push(("decode_reference_readback", g.hidden * F32_BYTES));
                host.push(("routed_ids_readback", K_TOP * I32_BYTES));
                out.device_parts = dev;
                out.host_parts = host;
            }
        }
        out.device_peak = sum_parts(&out.device_parts);
        out.host_peak = sum_parts(&out.host_parts);
        out.case_local_device = out
            .device_parts
            .iter()
            .filter(|p| !p.0.contains(GROW_ONLY))
            .map(|p| p.1)
            .sum();
        out.scratch = case_scratch_slots(case, g, limits)?;
        Ok(out)
    }

    /// Marker in a ledger part's name for the Gpu's grow-only scratch slots (persist across
    /// cases at their per-slot maximum instead of being freed with the case).
    const GROW_ONLY: &str = "grow-only";

    /// The three Gpu grow-only scratch needs of a window of `rows` tokens.
    fn window_scratch(g: &DiagGeometry, rows: usize) -> ScratchSlots {
        let [a4_gate_up, a4_down, group_scatter] =
            window_scratch_sizes(g.route, g.hidden, g.mi, g.n_exp, rows);
        ScratchSlots {
            a4_gate_up,
            a4_down,
            group_scatter,
        }
    }

    /// The grow-only Gpu scratch slots a case touches: per slot, the largest window the case
    /// runs (`CanonicalCombine` runs no window).
    fn case_scratch_slots(
        case: DiagCase,
        g: &DiagGeometry,
        limits: PanelLimits,
    ) -> Result<ScratchSlots, DispatchError> {
        Ok(match case {
            DiagCase::CanonicalCombine => ScratchSlots::default(),
            DiagCase::WithinExpertCanary => window_scratch(g, limits.max_grouped_rows),
            DiagCase::K1TailCohort if g.route.is_aggregate_layout() => window_scratch(g, g.k1_rows),
            DiagCase::K1TailCohort => window_scratch(g, 64 + g.k1_rows),
            DiagCase::ZeroRowPanel if g.route.is_aggregate_layout() => window_scratch(g, 40),
            DiagCase::ZeroRowPanel => {
                window_scratch(g, 40).max_with(&window_scratch(g, g.k1_rows.max(1)))
            }
            DiagCase::DecodeTailCohort => window_scratch(g, DECODE_TAIL_ANCHOR_ROWS + 1),
        })
    }

    /// The byte ledger of a whole diagnostic suite run in sequence on one `Gpu`, where every
    /// entry ends with [`diag_scope`] (sync, then return the parked pool buffers to HIP): the
    /// case-local allocations of different cases are never live together (peak = max), while
    /// the Gpu's grow-only scratch slots persist at their per-slot maximum and their growth
    /// briefly holds old + new.  `A` already charges the full-window slots; see
    /// [`SuiteBytes::extra_after_admitted`] for what is left to budget.
    pub fn suite_diagnostic_bytes(
        cases: &[DiagCase],
        g: &DiagGeometry,
        limits: PanelLimits,
    ) -> Result<SuiteBytes, DispatchError> {
        let mut suite = SuiteBytes::default();
        let mut needs: Vec<ScratchSlots> = Vec::new();
        for &case in cases {
            let b = diagnostic_bytes(case, g, limits)?;
            suite.case_local_device_peak = suite.case_local_device_peak.max(b.case_local_device);
            suite.host_peak = suite.host_peak.max(b.host_peak);
            suite.persistent = suite.persistent.max_with(&b.scratch);
            needs.push(b.scratch);
            suite.cases.push((case, b));
        }
        suite.growth_transient = suite.persistent.second_largest_transient(&needs);
        Ok(suite)
    }

    /// `canonical_combine_exact`: the production panel/pad combine kernels over the
    /// adversarial fixture and many partitions (within-expert splits included) equal the CPU
    /// canonical model and the monolithic GPU combine; reversed panel order differs.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn canonical_combine_exact(
        gpu: &mut Gpu,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        diag_scope(gpu, |gpu| canonical_combine_exact_body(gpu, seed))
    }

    fn canonical_combine_exact_body(
        gpu: &mut Gpu,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        let mut rep = PrimitiveReport::new("canonical_combine_exact");
        let r = hip(model::canonical_combine_exact(gpu, seed))?;
        rep.check(
            "panels_equal_cpu_canonical",
            r.mismatches_vs_cpu == 0,
            r.to_json(),
        );
        rep.check(
            "panels_equal_monolithic_gpu",
            r.mismatches_vs_monolithic_gpu == 0,
            format!(
                "{} mismatches over {} partitions",
                r.mismatches_vs_monolithic_gpu, r.partitions
            ),
        );
        rep.check(
            "reversed_order_negative_control_differs",
            r.reversed_order_diffs > 0,
            format!("{} diffs", r.reversed_order_diffs),
        );
        Ok(rep)
    }

    /// `within_expert_panel_canary`: one hot expert with more rows than a panel may hold is
    /// split across panels by the production schedule; guard canaries after every panel's
    /// scratch spans stay intact, every panel respects the row bound, and the combine equals
    /// the CPU canonical model and the monolithic GPU combine over the actual GEMM rows.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn within_expert_panel_canary(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        diag_scope(gpu, |gpu| {
            within_expert_panel_canary_body(gpu, call, binding, limits, seed)
        })
    }

    fn within_expert_panel_canary_body(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        let mut rep = PrimitiveReport::new("within_expert_panel_canary");
        let (lo, hi) = (binding.expert_lo, binding.expert_hi);
        if hi <= lo || hi > call.n_exp {
            return Err(bail("the binding names no experts"));
        }
        let width = hi - lo;
        let tokens = limits.max_grouped_rows;
        let mut rng = model::Rng(seed | 1);
        let tile = synth_tile(&mut rng, tokens, call.hidden, false, |tok, r| {
            if r < 3 {
                lo as i32
            } else {
                (lo + (tok * 7 + r * 3) % width) as i32
            }
        });
        let (ids, weights) = (tile.ids.clone(), tile.weights.clone());
        let window = run_window(gpu, call, &[binding], limits, &[tile], true, None)?;
        let hot_rows = window.plan.counts_padded[lo] as usize;
        rep.check(
            "hot_expert_exceeds_one_panel",
            hot_rows > limits.max_grouped_rows,
            format!(
                "hot expert {lo}: {hot_rows} rows, bound {}",
                limits.max_grouped_rows
            ),
        );
        let hot_panels = window
            .panels
            .iter()
            .filter(|p| p.rows > 0 && p.expert_lo <= lo && lo < p.expert_hi)
            .count();
        rep.check(
            "hot_expert_split_across_panels",
            hot_panels >= 2,
            format!("{hot_panels} panels hold expert {lo}"),
        );
        let max_rows = window.panels.iter().map(|p| p.rows).max().unwrap_or(0);
        rep.check(
            "every_panel_within_hard_bound",
            max_rows <= limits.max_grouped_rows,
            format!("max panel rows {max_rows}"),
        );
        rep.check(
            "panels_cover_every_grouped_row_once",
            window.panels.iter().map(|p| p.rows).sum::<usize>() == window.plan.total_rows,
            format!(
                "{} rows over {} panels",
                window.plan.total_rows,
                window.panels.len()
            ),
        );
        rep.check(
            "scratch_guard_canaries_intact",
            window.canary_checks > 0 && window.canary_violations == 0,
            format!(
                "{} guard bytes checked, {} overwritten",
                window.canary_checks, window.canary_violations
            ),
        );
        let want = cpu_canonical(&window, call, &ids, &weights);
        rep.check(
            "combine_equals_cpu_canonical",
            diffs(&window.acc, &want) == 0,
            format!("{} mismatching elements", diffs(&window.acc, &want)),
        );
        if window.plan.total_rows > 0 {
            let mono = monolithic(
                gpu,
                tokens,
                call.hidden,
                call.n_exp,
                window.plan.total_rows,
                &window.down,
                &window.inverse,
                &ids,
                &weights,
            )?;
            rep.check(
                "combine_equals_monolithic_gpu",
                diffs(&window.acc, &mono) == 0,
                format!("{} mismatching elements", diffs(&window.acc, &mono)),
            );
        }
        window.routes.release(gpu);
        Ok(rep)
    }

    /// The K1 short-tail scenario: a K1 tile (and, with `with_k2`, a K2 tile) over a binding
    /// split in two halves, so the cohort runs once per half with masked ids; the K1 rows
    /// must equal the incumbent's own unmasked K1 path over the whole binding.
    fn k1_scenario(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
        with_k2: bool,
        rep: &mut PrimitiveReport,
    ) -> Result<(), DispatchError> {
        let (lo, hi) = (binding.expert_lo, binding.expert_hi);
        if hi - lo < 2 || hi > call.n_exp {
            return Err(bail(
                "the K1 scenario needs a binding of at least two experts",
            ));
        }
        let mid = lo + (hi - lo) / 2;
        let halves = [
            ExpertPanelBinding {
                expert_hi: mid,
                ..binding
            },
            ExpertPanelBinding {
                expert_lo: mid,
                ..binding
            },
        ];
        let width = hi - lo;
        let k1_rows = 3usize.min(gpu.few_row_limit());
        let mut rng = model::Rng(seed | 1);
        let mut tiles = Vec::new();
        if with_k2 {
            tiles.push(synth_tile(&mut rng, 64, call.hidden, false, |tok, r| {
                (lo + (tok * 11 + r * 5) % width) as i32
            }));
        }
        let k1_tile = synth_tile(&mut rng, k1_rows, call.hidden, true, |tok, r| {
            match (tok, r) {
                (0, 0) => lo as i32,
                (0, 1) => (hi - 1) as i32,
                (t, 9) if t + 1 == k1_rows => -1,
                _ => (lo + (tok * 5 + r * 3) % width) as i32,
            }
        });
        let k1_ids = k1_tile.ids.clone();
        tiles.push(k1_tile);
        let k2_rows: usize = if with_k2 { tiles[0].rows() } else { 0 };
        let all_ids: Vec<i32> = tiles
            .iter()
            .flat_map(|t| {
                if t.k1 {
                    vec![-1; t.ids.len()]
                } else {
                    t.ids.clone()
                }
            })
            .collect();
        let all_weights: Vec<f32> = tiles.iter().flat_map(|t| t.weights.clone()).collect();
        let window = run_window(gpu, call, &halves, limits, &tiles, true, None)?;
        let mut expected_experts: Vec<usize> = k1_ids
            .iter()
            .filter(|&&id| id >= 0 && (id as usize) < call.n_exp)
            .map(|&id| id as usize)
            .collect();
        expected_experts.sort_unstable();
        expected_experts.dedup();
        rep.check(
            "k1_experts_downloaded_at_the_one_barrier",
            window.plan.k1_experts == expected_experts,
            format!("{:?} vs {:?}", window.plan.k1_experts, expected_experts),
        );
        let k1_panels = window.panels.iter().filter(|p| p.k1).count();
        rep.check(
            "k1_cohort_runs_once_per_binding",
            k1_panels == 2,
            format!("{k1_panels} flagged panels over 2 bindings"),
        );
        if !with_k2 {
            rep.check(
                "k1_only_window_has_only_zero_row_panels",
                window.plan.total_rows == 0
                    && window.panels.len() == 2
                    && window.panels.iter().all(|p| p.rows == 0 && p.k1),
                format!(
                    "{} rows, {} panels",
                    window.plan.total_rows,
                    window.panels.len()
                ),
            );
        } else {
            let want = cpu_canonical(&window, call, &all_ids, &all_weights);
            let n = k2_rows * call.hidden;
            rep.check(
                "k2_rows_equal_cpu_canonical_beside_the_cohort",
                diffs(&window.acc[..n], &want[..n]) == 0,
                format!(
                    "{} mismatching elements",
                    diffs(&window.acc[..n], &want[..n])
                ),
            );
            rep.check(
                "scratch_guard_canaries_intact",
                window.canary_violations == 0,
                format!(
                    "{} of {} guard bytes overwritten",
                    window.canary_violations, window.canary_checks
                ),
            );
        }
        // The incumbent's own K1 path over the whole binding, from the same saved input.
        let layout = window.routes.layout.clone();
        let d = layout.dims;
        let tile = *window
            .routes
            .tiles
            .iter()
            .find(|t| t.k1)
            .ok_or_else(|| bail("no K1 tile was recorded"))?;
        let params = *call.params();
        let mut view = params;
        view.batch_size = tile.rows;
        hip(gpu.memcpy_dtod_at_ordered_async(
            &params.x_norm_batch.buf,
            0,
            &window.routes.region.buf,
            layout.x_norm + tile.row0 * d.hidden * F32_BYTES,
            tile.rows * d.hidden * F32_BYTES,
        ))?;
        let ids = window
            .routes
            .ids_view(layout.topk_ids, tile.row0, tile.rows);
        let weights = window
            .routes
            .f32_rows(layout.topk_weights, tile.row0, tile.rows, K_TOP);
        let y0 = hip(gpu.alloc_tensor(&[tile.rows * K_TOP * d.hidden], DType::F32))?;
        let reference = hip(gpu.zeros(&[tile.rows * d.hidden], DType::F32))?;
        let run = (|| -> Result<Vec<f32>, DispatchError> {
            super::super::qt44_qt53_prefill::slowlink_k1_panel(
                gpu,
                &view,
                &ids,
                binding.gate_up_table,
                binding.down_table,
                &y0,
            )?;
            hip(gpu.moe_down_combine_top10_batched(
                &y0, &ids, &weights, &reference, d.hidden, tile.rows, d.n_exp,
            ))?;
            sync(gpu)?;
            hip(gpu.download_f32(&reference))
        })();
        let _ = gpu.free_tensor(y0);
        let _ = gpu.free_tensor(reference);
        let reference = run?;
        let got = &window.acc[tile.row0 * d.hidden..(tile.row0 + tile.rows) * d.hidden];
        rep.check(
            "k1_rows_equal_the_incumbent_k1_path",
            diffs(got, &reference) == 0,
            format!(
                "{} mismatching elements over {} rows",
                diffs(got, &reference),
                tile.rows
            ),
        );
        window.routes.release(gpu);
        Ok(())
    }

    /// The aggregate F32-tail scenario: the incumbent's REAL tail call (built by the caller as the
    /// model builds it, `row0 == 0`, its input state, scratch and weights in place) is the whole
    /// window, run through the production prepare, route plan, panels and ordered combine over
    /// a binding split in two halves, so its routed tail runs once per half on ids masked to
    /// it and accumulates in place; the rows must equal, bitwise, the incumbent's own routed
    /// tail (the same stage functions, unmasked ids, the whole binding, one launch chain) over
    /// the SAME restored input and route weights.  The binding MUST hold every expert the call's
    /// router selects (the resident all-expert binding).
    fn f32_tail_scenario(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        rep: &mut PrimitiveReport,
    ) -> Result<(), DispatchError> {
        let (lo, hi) = (binding.expert_lo, binding.expert_hi);
        if hi - lo < 2
            || hi > call.n_exp
            || call.row0 != 0
            || call.kind != TileKind::AggregateF32Tail
        {
            return Err(bail(
                "the F32 tail scenario needs a row-0 aggregate F32-tail call and a binding of at \
                 least two experts",
            ));
        }
        let mid = lo + (hi - lo) / 2;
        let halves = [
            ExpertPanelBinding {
                expert_hi: mid,
                ..binding
            },
            ExpertPanelBinding {
                expert_lo: mid,
                ..binding
            },
        ];
        let window = run_window(gpu, call, &halves, limits, &[], true, Some(call))?;
        rep.check(
            "cohort_only_window_has_only_zero_row_panels",
            window.plan.total_rows == 0 && window.panels.iter().all(|p| p.rows == 0),
            format!(
                "{} rows, {} panels",
                window.plan.total_rows,
                window.panels.len()
            ),
        );
        let layout = window.routes.layout.clone();
        let d = layout.dims;
        let params = *call.params();
        hip(gpu.memcpy_dtod_at_ordered_async(
            &params.x_norm_batch.buf,
            0,
            &window.routes.region.buf,
            layout.x_norm,
            call.rows * d.hidden * F32_BYTES,
        ))?;
        hip(gpu.memcpy_dtod_at_ordered_async(
            &params.topk_weights.buf,
            0,
            &window.routes.region.buf,
            layout.topk_weights,
            call.rows * K_TOP * F32_BYTES,
        ))?;
        let ids = window.routes.ids_view(layout.topk_ids, 0, call.rows);
        let reference = hip(gpu.zeros(&[call.rows * d.hidden], DType::F32))?;
        let run = (|| -> Result<Vec<f32>, DispatchError> {
            super::super::qt44_qt53_prefill::slowlink_f32_tail_panel(
                gpu,
                &params,
                call.use_path2,
                &ids,
                binding.gate_up_table,
                binding.down_table,
                &reference,
            )?;
            sync(gpu)?;
            hip(gpu.download_f32(&reference))
        })();
        let _ = gpu.free_tensor(reference);
        let reference = run?;
        let got = &window.acc[..call.rows * d.hidden];
        rep.check(
            "f32_tail_rows_equal_the_incumbent_routed_tail",
            diffs(got, &reference) == 0,
            format!(
                "{} mismatching elements over {} rows",
                diffs(got, &reference),
                call.rows
            ),
        );
        rep.check(
            "scratch_guard_canaries_intact",
            window.canary_violations == 0,
            format!(
                "{} of {} guard bytes overwritten",
                window.canary_violations, window.canary_checks
            ),
        );
        window.routes.release(gpu);
        Ok(())
    }

    /// `k1_tail_cohort`: the route's short-tail cohort.  Slab route: a K1 tile beside a K2 tile,
    /// cohort run per binding against masked ids, equals the incumbent K1 arithmetic; the K2
    /// rows are unaffected.  Aggregate route: `call` is a real row-0
    /// [`TileKind::AggregateF32Tail`] call and the cohort is its F32 routed tail
    /// ([`f32_tail_scenario`]).
    pub fn k1_tail_cohort(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        diag_scope(gpu, |gpu| {
            k1_tail_cohort_body(gpu, call, binding, limits, seed)
        })
    }

    fn k1_tail_cohort_body(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        let mut rep = PrimitiveReport::new("k1_tail_cohort");
        match call.route {
            WindowRoute::SlabIu4 => k1_scenario(gpu, call, binding, limits, seed, true, &mut rep)?,
            WindowRoute::Aggregate | WindowRoute::AggregateSymIu4 => {
                f32_tail_scenario(gpu, call, binding, limits, &mut rep)?
            }
        }
        Ok(rep)
    }

    /// Window rows the K2 anchor tile of [`decode_tail_cohort`] occupies: the caller builds
    /// the decode call's [`PreparedChunkCall`] with `row0` equal to this.
    /// Tools-only.
    pub const DECODE_TAIL_ANCHOR_ROWS: usize = 64;

    /// `decode_tail_cohort`: the incumbent's REAL one-row decode call (built by the caller as
    /// the model builds it, its input state, scratch and weights in place) beside a synthetic
    /// K2 tile, over a binding split in two halves so the decode K1 stack runs per half on ids
    /// masked to it.  Checks, bitwise: the decode row of the window equals the incumbent's own
    /// decode stages (gate/up, fused activation with the shared expert, down, canonical
    /// combine, shared down) run unmasked over the SAME restored ids/input/shared state
    /// against the whole binding; the K2 rows equal the CPU canonical combine; the K1 experts
    /// and per-binding cohort launches are what the real route needs.  The binding MUST hold
    /// every expert the decode call's router selects (the resident all-expert binding).
    /// `anchor` is any K2 prefill tile call of the layer (its scratch/geometry).
    /// Tools-only.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn decode_tail_cohort(
        gpu: &mut Gpu,
        anchor: &PreparedChunkCall<'_>,
        decode_call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        diag_scope(gpu, |gpu| {
            decode_tail_cohort_body(gpu, anchor, decode_call, binding, limits, seed)
        })
    }

    fn decode_tail_cohort_body(
        gpu: &mut Gpu,
        anchor: &PreparedChunkCall<'_>,
        decode_call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        let mut rep = PrimitiveReport::new("decode_tail_cohort");
        if anchor.decode || !anchor.k2 || !decode_call.decode {
            return Err(bail(
                "decode_tail_cohort needs a K2 prefill anchor call and a one-row decode call",
            ));
        }
        if decode_call.row0 != DECODE_TAIL_ANCHOR_ROWS
            || decode_call.hidden != anchor.hidden
            || decode_call.n_exp != anchor.n_exp
            || decode_call.mi != anchor.mi
        {
            return Err(bail(format!(
                "the decode call must sit at window row {DECODE_TAIL_ANCHOR_ROWS} with the anchor's geometry"
            )));
        }
        let (lo, hi) = (binding.expert_lo, binding.expert_hi);
        if hi - lo < 2 || hi > anchor.n_exp {
            return Err(bail(
                "the decode tail test needs a binding of at least two experts",
            ));
        }
        let mid = lo + (hi - lo) / 2;
        let halves = [
            ExpertPanelBinding {
                expert_hi: mid,
                ..binding
            },
            ExpertPanelBinding {
                expert_lo: mid,
                ..binding
            },
        ];
        let width = hi - lo;
        let mut rng = model::Rng(seed | 1);
        let k2_tile = synth_tile(
            &mut rng,
            DECODE_TAIL_ANCHOR_ROWS,
            anchor.hidden,
            false,
            |tok, r| (lo + (tok * 11 + r * 5) % width) as i32,
        );
        let (k2_ids, k2_weights) = (k2_tile.ids.clone(), k2_tile.weights.clone());
        let window = run_window(
            gpu,
            anchor,
            &halves,
            limits,
            &[k2_tile],
            true,
            Some(decode_call),
        )?;
        let layout = window.routes.layout.clone();
        let d = layout.dims;
        let rec = *window
            .routes
            .tiles
            .iter()
            .find(|t| t.decode)
            .ok_or_else(|| bail("the decode tile was not recorded"))?;
        // What the real router selected for the decode row, from the saved route.
        let ids_view = window.routes.ids_view(layout.topk_ids, rec.row0, 1);
        let routed = hip(gpu.slowlink_sync_download_i32(&ids_view, K_TOP))?;
        let mut expected: Vec<usize> = routed
            .iter()
            .filter(|&&id| id >= 0 && (id as usize) < d.n_exp)
            .map(|&id| id as usize)
            .collect();
        expected.sort_unstable();
        expected.dedup();
        rep.check(
            "decode_row_experts_read_back_at_the_one_barrier",
            window.plan.k1_experts == expected && !expected.is_empty(),
            format!("{:?} vs {:?}", window.plan.k1_experts, expected),
        );
        let touched = [(lo, mid), (mid, hi)]
            .iter()
            .filter(|&&(a, b)| expected.iter().any(|&e| e >= a && e < b))
            .count();
        let flagged = window.panels.iter().filter(|p| p.k1).count();
        rep.check(
            "decode_cohort_runs_once_per_touched_binding",
            flagged == touched && flagged > 0,
            format!("{flagged} flagged panels, {touched} touched halves"),
        );
        rep.check(
            "scratch_guard_canaries_intact",
            window.canary_violations == 0,
            format!(
                "{} of {} guard bytes overwritten",
                window.canary_violations, window.canary_checks
            ),
        );
        // The K2 anchor rows against the CPU canonical combine (the decode row carries no
        // K2 ids).
        let mut all_ids = k2_ids;
        all_ids.extend([-1i32; K_TOP]);
        let mut all_weights = k2_weights;
        all_weights.extend([0.0f32; K_TOP]);
        let want = cpu_canonical(&window, anchor, &all_ids, &all_weights);
        let n = DECODE_TAIL_ANCHOR_ROWS * d.hidden;
        rep.check(
            "k2_rows_equal_cpu_canonical_beside_the_decode_tile",
            diffs(&window.acc[..n], &want[..n]) == 0,
            format!(
                "{} mismatching elements",
                diffs(&window.acc[..n], &want[..n])
            ),
        );
        // The incumbent's own decode stages over the same ids/input/shared state.
        let params = *decode_call.decode_params();
        let selection = decode_selection(decode_call);
        let shared = params
            .shared
            .as_ref()
            .ok_or_else(|| bail("the decode call carries no shared expert"))?;
        let input = decode_activation_input(&params, selection);
        let smi = d.shared_intermediate;
        let saved = layout.k1_shared + rec.k1_row0 * 2 * smi * F32_BYTES;
        let region = &window.routes.region.buf;
        let restore = |gpu: &Gpu, dst: &hip_bridge::DeviceBuffer, src_off: usize, bytes: usize| {
            hip(gpu.memcpy_dtod_at_ordered_async(dst, 0, region, src_off, bytes))
        };
        restore(
            gpu,
            &input.buf,
            layout.x_norm + rec.row0 * d.hidden * F32_BYTES,
            d.hidden * F32_BYTES,
        )?;
        restore(
            gpu,
            &params.topk_indices.buf,
            layout.topk_ids + rec.row0 * K_TOP * I32_BYTES,
            K_TOP * I32_BYTES,
        )?;
        restore(
            gpu,
            &params.topk_weights.buf,
            layout.topk_weights + rec.row0 * K_TOP * F32_BYTES,
            K_TOP * F32_BYTES,
        )?;
        restore(gpu, &shared.gate_out.buf, saved, smi * F32_BYTES)?;
        restore(
            gpu,
            &shared.up_out.buf,
            saved + smi * F32_BYTES,
            smi * F32_BYTES,
        )?;
        restore(
            gpu,
            &shared.scalar.buf,
            layout.shared_scalar + rec.row0 * F32_BYTES,
            F32_BYTES,
        )?;
        let mut p = params;
        p.expert_gate_up_ptrs = binding.gate_up_table;
        p.expert_down_ptrs = binding.down_table;
        let gate = super::super::slice_moe_f32_view(shared.gate_out, 0, smi);
        let up = super::super::slice_moe_f32_view(shared.up_out, 0, smi);
        let target = hip(gpu.zeros(&[d.hidden], DType::F32))?;
        let reference = (|| -> Result<Vec<f32>, DispatchError> {
            super::super::decode_gate_up_stage(
                gpu,
                &p,
                selection.resolution,
                input,
                false,
                selection.route,
            )?;
            super::super::decode_activation_stage(
                gpu,
                &p,
                selection.resolution,
                selection.route,
                Some((&gate, &up)),
            )?;
            super::super::decode_down_stage(
                gpu,
                &p,
                selection.resolution,
                p.x_residual,
                false,
                false,
                false,
                false,
                false,
                false,
                selection.route,
            )?;
            super::super::decode_combine_stage(gpu, &p, &target, selection.route)?;
            super::super::decode_shared_down_stage(
                decode_call.call.dispatch_ctx(),
                gpu,
                &p,
                &gate,
                &up,
                &target,
                selection.router_shared_fuse,
                selection.route,
            )?;
            sync(gpu)?;
            hip(gpu.download_f32(&target))
        })();
        let _ = gpu.free_tensor(target);
        let reference = reference?;
        let got = &window.acc[rec.row0 * d.hidden..(rec.row0 + 1) * d.hidden];
        rep.check(
            "decode_row_equals_the_incumbent_decode_stages",
            diffs(got, &reference) == 0,
            format!(
                "{} mismatching elements of {}",
                diffs(got, &reference),
                d.hidden
            ),
        );
        window.routes.release(gpu);
        Ok(rep)
    }

    /// `zero_row_panel`: (a) a window where nothing routes yields exactly one zero-row
    /// combine-only panel owning every expert and an all +0 accumulator; (b) a K1-only
    /// window yields only zero-row K1 panels whose rows still equal the incumbent K1 path.
    /// Tools-only.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn zero_row_panel(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        diag_scope(gpu, |gpu| {
            zero_row_panel_body(gpu, call, binding, limits, seed)
        })
    }

    fn zero_row_panel_body(
        gpu: &mut Gpu,
        call: &PreparedChunkCall<'_>,
        binding: ExpertPanelBinding<'_>,
        limits: PanelLimits,
        seed: u64,
    ) -> Result<PrimitiveReport, DispatchError> {
        let mut rep = PrimitiveReport::new("zero_row_panel");
        let mut rng = model::Rng(seed | 1);
        let tile = synth_tile(&mut rng, 40, call.hidden, false, |_, _| -1);
        let window = run_window(gpu, call, &[binding], limits, &[tile], true, None)?;
        let only = window.panels.first().copied();
        rep.check(
            "empty_window_is_one_zero_row_panel_owning_every_expert",
            window.plan.total_rows == 0
                && window.panels.len() == 1
                && only.is_some_and(|p| {
                    p.rows == 0 && !p.k1 && p.own_lo == 0 && p.own_hi == call.n_exp
                }),
            format!(
                "{} rows, panels {:?}",
                window.plan.total_rows, window.panels
            ),
        );
        rep.check(
            "empty_window_accumulator_is_all_positive_zero",
            window.acc.iter().all(|v| v.to_bits() == 0),
            format!(
                "{} non-+0 elements",
                window.acc.iter().filter(|v| v.to_bits() != 0).count()
            ),
        );
        window.routes.release(gpu);
        if call.route == WindowRoute::SlabIu4 {
            k1_scenario(gpu, call, binding, limits, seed ^ 0x5a5a, false, &mut rep)?;
        }
        Ok(rep)
    }
}

#[cfg(test)]
mod slowlink_panel_tests {
    use super::*;
    use rdna_compute::slab_moe::slowlink_panel_owns;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    const N_EXP: usize = 512;

    fn dims(rows: usize) -> PromptRouteDims {
        PromptRouteDims {
            rows,
            hidden: 2560,
            n_exp: N_EXP,
            mi: 640,
            shared_intermediate: 640,
            k1_rows: rows.min(16),
            tail_rows: 0,
            route: WindowRoute::SlabIu4,
        }
    }

    /// CPU twin of `qwen4_moe_group_*`: stable `(expert, slot)` grouping, pad16 per expert.
    struct Grouping {
        counts_padded: Vec<i32>,
        start: Vec<usize>,
        /// `inverse[slot]`, -1 for an unrouted slot.
        inverse: Vec<i32>,
        sorted: Vec<i32>,
    }

    fn group(ids: &[i32]) -> Grouping {
        let mut count = vec![0usize; N_EXP];
        for &id in ids {
            if (0..N_EXP as i32).contains(&id) {
                count[id as usize] += 1;
            }
        }
        let counts_padded: Vec<i32> = count
            .iter()
            .map(|&c| (c.div_ceil(16) * 16) as i32)
            .collect();
        let mut start = vec![0usize; N_EXP + 1];
        for e in 0..N_EXP {
            start[e + 1] = start[e] + counts_padded[e] as usize;
        }
        let mut fill = vec![0usize; N_EXP];
        let mut inverse = vec![-1i32; ids.len()];
        let mut sorted = vec![-1i32; start[N_EXP]];
        for (slot, &id) in ids.iter().enumerate() {
            if (0..N_EXP as i32).contains(&id) {
                let e = id as usize;
                let row = start[e] + fill[e];
                fill[e] += 1;
                inverse[slot] = row as i32;
                sorted[row] = slot as i32;
            }
        }
        Grouping {
            counts_padded,
            start,
            inverse,
            sorted,
        }
    }

    fn plan(ids: &[i32], k1: Vec<usize>) -> (Grouping, LayerRoutePlan) {
        let g = group(ids);
        let p = LayerRoutePlan::from_counts(&g.counts_padded, k1, BarrierCost::default()).unwrap();
        (g, p)
    }

    fn runs(width: usize) -> Vec<(usize, usize)> {
        (0..N_EXP.div_ceil(width))
            .map(|i| (i * width, ((i + 1) * width).min(N_EXP)))
            .collect()
    }

    fn random_ids(rng: &mut Rng, tokens: usize, hot: usize) -> Vec<i32> {
        let mut ids = Vec::new();
        for _ in 0..tokens {
            for _ in 0..K_TOP {
                ids.push(match rng.below(20) {
                    0 => -1,
                    1 => N_EXP as i32 + rng.below(9) as i32,
                    2..=6 => (hot + rng.below(3)) as i32,
                    _ => rng.below(N_EXP) as i32,
                });
            }
        }
        ids
    }

    #[test]
    fn slowlink_panel_k1_shared_region_carries_the_decode_tile_preactivation_gate_up() {
        // 48 K1 rows * (gate + up) * I640 * 4 B = 245,760 B, charged inside total_bytes.
        let mut d = dims(16_384);
        d.k1_rows = 48;
        let with = PromptRouteLayout::new(&d).unwrap();
        d.k1_rows = 0;
        let without = PromptRouteLayout::new(&d).unwrap();
        assert_eq!(
            with.k1_bytes(),
            48 * (K_TOP * (2560 * 4 + 4) + 2 * 640 * 4),
            "the K1 account names y0, masked ids and the shared pre-activation rows"
        );
        assert_eq!(without.k1_bytes(), 0);
        assert!(with.total_bytes() >= without.total_bytes() + with.k1_bytes());
        // A layer with no shared expert saves no shared rows.
        let mut d = dims(1536);
        d.shared_intermediate = 0;
        d.k1_rows = 8;
        assert_eq!(
            PromptRouteLayout::new(&d).unwrap().k1_bytes(),
            8 * K_TOP * (2560 * 4 + 4)
        );
    }

    use super::selftest::{
        diag_scope, suite_diagnostic_bytes, DiagCase, DiagDevice, DiagGeometry, ScratchSlots,
        DEFAULT_SUITE, GATE_PRIMITIVES,
    };

    /// A fake device that models the allocator's parked pool: `free_tensor` parks (a
    /// `DeviceBuffer` has no `Drop`, so a dropped tensor would leak instead), only the drain
    /// returns bytes to the runtime.
    #[derive(Default)]
    struct FakeDev {
        log: Vec<&'static str>,
        parked: usize,
        returned: usize,
        sync_fails: bool,
    }

    impl DiagDevice for FakeDev {
        fn quiesce(&mut self) -> Result<(), DispatchError> {
            self.log.push("sync");
            if self.sync_fails {
                Err(bail("the stream wait failed"))
            } else {
                Ok(())
            }
        }

        fn release_parked(&mut self) {
            self.log.push("drain");
            self.returned += self.parked;
            self.parked = 0;
        }
    }

    fn park(dev: &mut FakeDev, bytes: usize) {
        dev.log.push("body");
        dev.parked += bytes;
    }

    #[test]
    fn slowlink_panel_diag_scope_syncs_then_drains_after_success_and_after_an_error() {
        let mut dev = FakeDev::default();
        let ok = diag_scope(&mut dev, |d| {
            park(d, 100);
            Ok(7)
        });
        assert_eq!(ok.unwrap(), 7);
        assert_eq!(dev.log, ["body", "sync", "drain"]);
        assert_eq!((dev.parked, dev.returned), (0, 100));
        // A failing body is still drained once the wait proved quiescence; its error is kept.
        let mut dev = FakeDev::default();
        let err = diag_scope(&mut dev, |d| -> Result<(), DispatchError> {
            park(d, 40);
            Err(bail("check failed"))
        })
        .unwrap_err();
        assert!(format!("{err:?}").contains("check failed"));
        assert_eq!(dev.log, ["body", "sync", "drain"]);
        assert_eq!((dev.parked, dev.returned), (0, 40));
    }

    #[test]
    fn slowlink_panel_diag_scope_frees_nothing_when_quiescence_is_not_proven() {
        let mut dev = FakeDev {
            sync_fails: true,
            ..FakeDev::default()
        };
        let err = diag_scope(&mut dev, |d| {
            park(d, 64);
            Ok(())
        })
        .unwrap_err();
        assert!(format!("{err:?}").contains("stream wait failed"));
        assert_eq!(dev.log, ["body", "sync"], "no drain after a failed wait");
        assert_eq!(
            (dev.parked, dev.returned),
            (64, 0),
            "work may still read the buffers"
        );
        // Both failures are reported; still nothing is freed.
        let mut dev = FakeDev {
            sync_fails: true,
            ..FakeDev::default()
        };
        let err = diag_scope(&mut dev, |d| -> Result<(), DispatchError> {
            park(d, 8);
            Err(bail("mismatch"))
        })
        .unwrap_err();
        let text = format!("{err:?}");
        assert!(text.contains("mismatch") && text.contains("stream wait"));
        assert_eq!((dev.parked, dev.returned), (8, 0));
    }

    #[test]
    fn slowlink_panel_diag_scope_retains_nothing_across_a_suite_of_cases() {
        // Different buffer classes per case park separately and are never reused (the
        // `upload_raw` case), so without the drain the parked bytes would SUM.
        let sizes = [128_696_320usize, 42_394_880, 7_372_800, 2_949_120, 241_200];
        let mut dev = FakeDev::default();
        for &s in &sizes {
            diag_scope(&mut dev, |d| {
                park(d, s);
                park(d, s / 2);
                Ok(())
            })
            .unwrap();
            assert_eq!(dev.parked, 0, "nothing parked after a case");
        }
        assert_eq!(dev.returned, sizes.iter().map(|s| s + s / 2).sum::<usize>());
        let without_drain: usize = sizes.iter().map(|s| s + s / 2).sum();
        assert!(
            without_drain > 2 * sizes[0],
            "the parked sum is what the drain removes"
        );
    }

    #[test]
    fn slowlink_panel_default_suite_is_the_gate_composition_plus_the_decode_tail() {
        let names: Vec<&str> = DEFAULT_SUITE.iter().map(|c| c.name()).collect();
        assert_eq!(
            names,
            [
                "within_expert_panel_canary",
                "canonical_combine_exact",
                "k1_tail_cohort",
                "zero_row_panel",
                "decode_tail_cohort",
            ]
        );
        // The gate's four dispatch primitives are the first four, in the gate's order.
        assert_eq!(GATE_PRIMITIVES, DEFAULT_SUITE[..4]);
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        for case in DEFAULT_SUITE {
            assert_eq!(DiagCase::from_name(case.name()), Some(case));
        }
        assert_eq!(DiagCase::from_name("payload_table_identity"), None);
    }

    fn real_identity() -> DiagGeometry {
        DiagGeometry {
            hidden: 2560,
            n_exp: 512,
            mi: 640,
            binding_experts: 32,
            k1_rows: 3,
            shared_intermediate: 640,
            route: WindowRoute::SlabIu4,
        }
    }

    #[test]
    fn slowlink_panel_suite_bytes_split_case_local_from_persistent_scratch() {
        let g = real_identity();
        let suite = suite_diagnostic_bytes(&DEFAULT_SUITE, &g, PanelLimits::S1).unwrap();
        assert_eq!(suite.cases.len(), DEFAULT_SUITE.len());
        // Case-local peak: the canary's route region + the monolithic reference.
        assert_eq!(suite.case_local_device_peak, 171_091_200);
        assert_eq!(
            suite.persistent,
            ScratchSlots {
                a4_gate_up: 2_949_120,
                a4_down: 7_372_800,
                group_scatter: 411_648,
            }
        );
        assert_eq!(suite.persistent.total(), 10_733_568);
        // The old buffer a growth step frees while allocating the new one: the next smaller
        // case's largest slot (the K1 tail's down sidecar).
        assert_eq!(suite.growth_transient, 241_200);
        assert_eq!(suite.device_total(), 182_065_968);
        assert_eq!(suite.host_peak, 351_518_720);
        // Each case's own ledger separates the same two kinds of bytes.
        for (case, b) in &suite.cases {
            assert_eq!(
                b.device_peak,
                b.case_local_device + b.scratch.total(),
                "{case:?}: the grow-only parts are exactly the scratch slots"
            );
        }
        // T's diagnostic buffers (page + 128 MiB) are an independent sequential case: the
        // dispatch extra stays under T's actual 217,821,184.
        assert!(suite.device_total() <= 217_821_184);
    }

    #[test]
    fn slowlink_panel_suite_extra_after_the_admitted_census_does_not_double_count() {
        let g = real_identity();
        let suite = suite_diagnostic_bytes(&DEFAULT_SUITE, &g, PanelLimits::S1).unwrap();
        // The S1 full-window census already reserves the 16384-row A4 and scatter slots: the
        // diagnostics never grow them, so only the case-local buffers are left.
        let window = PromptRouteLayout::new(&PromptRouteDims {
            rows: 16_384,
            hidden: 2560,
            n_exp: 512,
            mi: 640,
            shared_intermediate: 640,
            k1_rows: 48,
            tail_rows: 0,
            route: WindowRoute::SlabIu4,
        })
        .unwrap();
        let admitted = ScratchSlots {
            a4_gate_up: (2560 / A4_BLOCK_K) * 16_384 * A4_RECORD_BYTES,
            a4_down: (640 / A4_BLOCK_K) * 16_384 * K_TOP * A4_RECORD_BYTES,
            group_scatter: window.group_scatter_bytes(),
        };
        assert_eq!(
            admitted.a4_gate_up + admitted.a4_down,
            window.a4_sidecar_bytes()
        );
        assert_eq!(suite.extra_after_admitted(&admitted), 171_091_200);
        // Only the 1536-row tile's slots reserved: the excess plus the old-slot overlap.
        let tile = PromptRouteLayout::new(&PromptRouteDims {
            rows: 1536,
            hidden: 2560,
            n_exp: 512,
            mi: 640,
            shared_intermediate: 640,
            k1_rows: 48,
            tail_rows: 0,
            route: WindowRoute::SlabIu4,
        })
        .unwrap();
        let tile_slots = ScratchSlots {
            a4_gate_up: (2560 / A4_BLOCK_K) * 1536 * A4_RECORD_BYTES,
            a4_down: (640 / A4_BLOCK_K) * 1536 * K_TOP * A4_RECORD_BYTES,
            group_scatter: tile.group_scatter_bytes(),
        };
        assert_eq!(suite.extra_after_admitted(&tile_slots), 179_303_680);
        // Nothing reserved: the whole suite total.
        assert_eq!(
            suite.extra_after_admitted(&ScratchSlots::default()),
            suite.device_total()
        );
    }

    #[test]
    fn slowlink_panel_suite_bytes_of_one_case_equal_its_own_ledger() {
        let g = real_identity();
        let one =
            suite_diagnostic_bytes(&[DiagCase::CanonicalCombine], &g, PanelLimits::S1).unwrap();
        // The canonical-combine fixture runs no window: no Gpu scratch, no growth.
        assert_eq!(one.persistent, ScratchSlots::default());
        assert_eq!(one.growth_transient, 0);
        assert_eq!(one.device_total(), one.case_local_device_peak);
        assert!(
            suite_diagnostic_bytes(&[], &g, PanelLimits::S1)
                .unwrap()
                .device_total()
                == 0
        );
    }

    #[test]
    fn slowlink_panel_aggregate_suite_ledger_has_no_a4_scratch_and_runs_a_real_tail_window() {
        use super::selftest::diagnostic_bytes;
        let slab = real_identity();
        let agg = DiagGeometry {
            route: WindowRoute::Aggregate,
            ..slab
        };
        let s = suite_diagnostic_bytes(&DEFAULT_SUITE, &slab, PanelLimits::S1).unwrap();
        let a = suite_diagnostic_bytes(&DEFAULT_SUITE, &agg, PanelLimits::S1).unwrap();
        // The slab windows grow the two A4 sidecar slots; the aggregate ones never do.
        assert!(s.persistent.a4_gate_up > 0 && s.persistent.a4_down > 0);
        assert_eq!((a.persistent.a4_gate_up, a.persistent.a4_down), (0, 0));
        assert_eq!(a.persistent.group_scatter, s.persistent.group_scatter);
        // Its cohort primitive is one real F32-tail call (3 rows) as the whole window, with a
        // zero-target reference: the region carries the tail rows' masked-id scratch, no x_f16.
        let tail = diagnostic_bytes(DiagCase::K1TailCohort, &agg, PanelLimits::S1).unwrap();
        assert!(tail
            .device_parts
            .iter()
            .any(|p| p.0 == "tail_reference_target_f32" && p.1 == 3 * 2560 * 4));
        let region = PromptRouteLayout::new(&PromptRouteDims {
            rows: 3,
            hidden: 2560,
            n_exp: 512,
            mi: 640,
            shared_intermediate: 640,
            k1_rows: 0,
            tail_rows: 3,
            route: WindowRoute::Aggregate,
        })
        .unwrap();
        assert_eq!(region.x_f16_bytes(), 0);
        assert!(tail
            .device_parts
            .iter()
            .any(|p| p.0 == "route_region" && p.1 == region.total_bytes()));
        // The aggregate ZeroRowPanel has no synthetic K1-only window: only the 40-row window.
        let zero = diagnostic_bytes(DiagCase::ZeroRowPanel, &agg, PanelLimits::S1).unwrap();
        let region40 = PromptRouteLayout::new(&PromptRouteDims {
            rows: 40,
            hidden: 2560,
            n_exp: 512,
            mi: 640,
            shared_intermediate: 0,
            k1_rows: 0,
            tail_rows: 0,
            route: WindowRoute::Aggregate,
        })
        .unwrap();
        assert!(zero
            .device_parts
            .iter()
            .any(|p| p.0 == "route_region" && p.1 == region40.total_bytes()));
    }

    #[test]
    fn slowlink_panel_decode_tail_diag_ledger_is_pinned_and_within_the_canary_peak() {
        use super::selftest::{diagnostic_bytes, DiagCase, DiagGeometry};
        // The real MQ4 layer identity with the RESIDENT all-expert binding, a 64-token K2
        // anchor plus one decode row, shared I640.
        let g = DiagGeometry {
            hidden: 2560,
            n_exp: 512,
            mi: 640,
            binding_experts: 512,
            k1_rows: 1,
            shared_intermediate: 640,
            route: WindowRoute::SlabIu4,
        };
        let limits = PanelLimits::S1;
        let decode = diagnostic_bytes(DiagCase::DecodeTailCohort, &g, limits).unwrap();
        // Grouped padding spreads over every expert: 640 slots + 15 * 512 pad rows = 8,320.
        // The previous 2,012,120 plus the cohort's 512 count words (2,048 B): the window
        // region is 1,659,392 B with them.
        assert_eq!(decode.device_peak, 2_014_168, "{:?}", decode.device_parts);
        assert_eq!(decode.host_peak, 99_041_064, "{:?}", decode.host_parts);
        let canary = diagnostic_bytes(
            DiagCase::WithinExpertCanary,
            &DiagGeometry {
                binding_experts: 32,
                shared_intermediate: 0,
                ..g
            },
            limits,
        )
        .unwrap();
        assert!(decode.device_peak <= canary.device_peak);
        assert!(decode.host_peak <= canary.host_peak);
    }

    #[test]
    fn slowlink_panel_arm_signatures_enforce_geometry_and_the_k2_arm_only_between_k2_tiles() {
        // signature = [hidden, n_exp, mi, down_m, shared_I, use_path2].
        let arm = |signature: [usize; 6], k2: bool, kind: TileKind| WindowArm {
            signature,
            k2,
            kind,
        };
        let w = WindowRoute::SlabIu4;
        let k2 = arm([2560, 512, 640, 2560, 640, 1], true, TileKind::SlabK2);
        // A K1 prefill tail and a one-row decode tile, each on its own original path.
        let k1_tail = arm([2560, 512, 640, 2560, 640, 0], false, TileKind::SlabK1Tail);
        let decode = arm([2560, 512, 640, 2560, 640, 0], false, TileKind::Decode);
        assert!(window_arms_compatible(w, &[k2, k2, k1_tail]));
        assert!(window_arms_compatible(w, &[k2, k2, decode]));
        assert!(
            window_arms_compatible(w, &[k1_tail, k2, k2, decode]),
            "a window may open with K1"
        );
        assert!(
            window_arms_compatible(w, &[decode, k1_tail]),
            "a K1-only window"
        );
        // Two K2 tiles on different K2 arms never share a window, even when a K1 tile opens it.
        let other_arm = arm([2560, 512, 640, 2560, 640, 0], true, TileKind::SlabK2);
        assert!(!window_arms_compatible(w, &[k2, other_arm]));
        assert!(!window_arms_compatible(w, &[decode, k2, other_arm]));
        // Any geometry difference refuses, K1 or not.
        for field in 0..5 {
            let mut bad = k1_tail;
            bad.signature[field] += 1;
            assert!(!window_arms_compatible(w, &[k2, bad]), "field {field}");
            assert!(!window_arms_compatible(w, &[bad, k2]), "field {field}");
        }
    }

    #[test]
    fn slowlink_panel_window_arms_refuse_a_tile_kind_of_another_route() {
        let arm = |k2: bool, kind: TileKind| WindowArm {
            signature: [2560, 512, 640, 2560, 640, 1],
            k2,
            kind,
        };
        let wmma = arm(true, TileKind::AggregateWmma);
        let iu4 = arm(true, TileKind::AggregateIu4);
        let tail = arm(false, TileKind::AggregateF32Tail);
        let decode = arm(false, TileKind::Decode);
        let slab = arm(true, TileKind::SlabK2);
        let f16 = WindowRoute::Aggregate;
        let sym = WindowRoute::AggregateSymIu4;
        // The F32 tail and the decode tile stay valid inside both aggregate windows.
        assert!(window_arms_compatible(f16, &[wmma, wmma, tail, decode]));
        assert!(window_arms_compatible(sym, &[iu4, iu4, tail, decode]));
        assert!(window_arms_compatible(sym, &[tail, decode]));
        // The F16 WMMA arm never runs in an IU4 window and the IU4 arm never in an F16 one.
        assert!(!window_arms_compatible(sym, &[iu4, wmma, tail]));
        assert!(!window_arms_compatible(f16, &[wmma, iu4, tail]));
        assert!(!window_arms_compatible(sym, &[wmma]));
        assert!(!window_arms_compatible(f16, &[iu4]));
        // Neither aggregate window holds a slab tile, nor the slab window an aggregate one.
        assert!(!window_arms_compatible(sym, &[slab]));
        assert!(!window_arms_compatible(f16, &[slab]));
        assert!(!window_arms_compatible(WindowRoute::SlabIu4, &[iu4]));
        assert!(!window_arms_compatible(WindowRoute::SlabIu4, &[tail]));
    }

    #[test]
    fn slowlink_panel_charged_bytes_are_region_a4_and_full_window_scatter_scratch() {
        // The real MQ4 layer identity: H2560, mi 640, E512, shared I640, a 16384-row window
        // with up to 48 K1 rows.
        let mut d = dims(16_384);
        d.k1_rows = 48;
        let l = PromptRouteLayout::new(&d).unwrap();
        assert_eq!(l.total_bytes(), 386_072_832);
        assert_eq!(l.a4_sidecar_bytes(), 82_575_360);
        assert_eq!(l.group_scatter_bytes(), 3_278_848);
        assert_eq!(
            l.group_scatter_bytes(),
            rdna_compute::scratch::qwen4_moe_group_layout(163_840, 512).bytes
        );
        assert_eq!(l.charged_bytes(), 471_927_040);
        // The 1536-row tile's scratch is a strict subset of the full-window size.
        let tile = PromptRouteLayout::new(&dims(1536)).unwrap();
        assert_eq!(tile.group_scatter_bytes(), 309_248);
        assert!(tile.group_scatter_bytes() < l.group_scatter_bytes());
    }

    #[test]
    fn slowlink_panel_scratch_allocation_sizes_are_the_three_grow_only_slots() {
        for rows in [1usize, 16, 1536, 1537, 16_384] {
            let slab = PromptRouteLayout::new(&dims(rows)).unwrap();
            let aggregate = PromptRouteLayout::new(&aggregate_dims(rows, 0, 0)).unwrap();
            let [gate_up, down, scatter] = slab.scratch_allocation_sizes();
            assert_eq!(
                gate_up,
                (2560 / A4_BLOCK_K) * rows * A4_RECORD_BYTES,
                "{rows}"
            );
            assert_eq!(
                down,
                (640 / A4_BLOCK_K) * rows * K_TOP * A4_RECORD_BYTES,
                "{rows}"
            );
            assert_eq!(
                scatter,
                rdna_compute::scratch::qwen4_moe_group_layout(rows * K_TOP, 512).bytes
            );
            assert_eq!(slab.a4_sidecar_bytes(), gate_up + down);
            assert_eq!(slab.group_scatter_bytes(), scatter);
            assert_eq!(
                slab.scratch_allocation_sizes().iter().sum::<usize>(),
                slab.charged_bytes() - slab.total_bytes()
            );
            // The aggregate route quantizes nothing: only the scatter scratch.
            assert_eq!(aggregate.scratch_allocation_sizes(), [0, 0, scatter]);
            assert_eq!(
                aggregate.scratch_allocation_sizes().iter().sum::<usize>(),
                aggregate.charged_bytes() - aggregate.total_bytes()
            );
        }
    }

    #[test]
    fn slowlink_panel_layout_is_aligned_disjoint_and_census_exact() {
        for rows in [1usize, 3, 16, 1536, 1537, 16384] {
            let layout = PromptRouteLayout::new(&dims(rows)).unwrap();
            assert_eq!(layout.total_bytes() % ALIGN, 0);
            let fields = [
                layout.x_norm,
                layout.route_ids,
                layout.topk_ids,
                layout.topk_weights,
                layout.sorted,
                layout.tiles,
                layout.inverse,
                layout.counts,
                layout.offsets,
                layout.accumulator,
                layout.shared_hidden,
                layout.shared_scalar,
                layout.k1_y0,
                layout.k1_ids,
            ];
            assert!(fields.iter().all(|f| f % ALIGN == 0));
            assert!(fields.windows(2).all(|w| w[0] <= w[1]));
            // The named byte accounts sum (before alignment) to at most the region and the
            // region exceeds them by alignment slack only.
            let named = layout.x_norm_bytes()
                + layout.topk_ids_bytes() * 3
                + layout.route_map_bytes()
                + layout.accumulator_bytes()
                + layout.shared_hidden_bytes()
                + layout.shared_scalar_bytes()
                + layout.k1_bytes()
                + layout.cohort_counts_bytes();
            assert!(layout.total_bytes() >= named);
            assert!(layout.total_bytes() - named <= 16 * ALIGN, "rows {rows}");
        }
        let l = PromptRouteLayout::new(&dims(16384)).unwrap();
        assert_eq!(l.a4_sidecar_bytes(), 20 * 16384 * 72 + 5 * 16384 * 10 * 72);
        assert_eq!(l.a4_sidecar_bytes(), 82_575_360);
    }

    #[test]
    fn slowlink_panel_layout_refuses_degenerate_and_counts_the_shared_dims() {
        let mut d = dims(1536);
        d.rows = 0;
        assert!(PromptRouteLayout::new(&d).is_err());
        let mut d = dims(1536);
        d.k1_rows = 1537;
        assert!(PromptRouteLayout::new(&d).is_err());
        let with = PromptRouteLayout::new(&dims(1536)).unwrap();
        let mut d = dims(1536);
        d.shared_intermediate = 0;
        d.k1_rows = 0;
        let without = PromptRouteLayout::new(&d).unwrap();
        assert_eq!(without.shared_hidden_bytes(), 0);
        assert_eq!(without.shared_scalar_bytes(), 0);
        assert_eq!(with.shared_hidden_bytes(), 1536 * 640 * 4);
        assert_eq!(with.shared_scalar_bytes(), 1536 * 4);
        assert!(with.total_bytes() > without.total_bytes());
    }

    fn aggregate_dims(rows: usize, k1_rows: usize, tail_rows: usize) -> PromptRouteDims {
        PromptRouteDims {
            rows,
            hidden: 2560,
            n_exp: N_EXP,
            mi: 640,
            shared_intermediate: 640,
            k1_rows,
            tail_rows,
            route: WindowRoute::Aggregate,
        }
    }

    #[test]
    fn slowlink_panel_aggregate_layout_swaps_the_a4_sidecars_for_the_f16_window_activations() {
        // The real split artifact's identity: H2560, mi 640, E512, shared I640, 16384 rows.
        let mut slab = dims(16_384);
        slab.k1_rows = 48;
        let s = PromptRouteLayout::new(&slab).unwrap();
        let a = PromptRouteLayout::new(&aggregate_dims(16_384, 48, 0)).unwrap();
        assert_eq!(a.route(), WindowRoute::Aggregate);
        assert_eq!(a.a4_sidecar_bytes(), 0);
        assert_eq!(a.x_f16_bytes(), 16_384 * 2560 * 2);
        assert_eq!(a.x_f16_bytes(), 83_886_080);
        // Region: the slab's fields unchanged, then the F16 window rows, the 2048-row slot
        // map and one decode tile's `[10, hidden]` F32 scratch.
        assert_eq!(
            a.total_bytes() - s.total_bytes(),
            83_886_080 + 2048 * 4 + 10 * 2560 * 4
        );
        assert_eq!(a.total_bytes(), 470_069_504);
        assert_eq!(a.group_scatter_bytes(), s.group_scatter_bytes());
        assert_eq!(a.charged_bytes(), a.total_bytes() + a.group_scatter_bytes());
        assert_eq!(a.charged_bytes(), 473_348_352);
        // The slab route pays the A4 sidecars instead: the two routes' charges are within a
        // few MiB of each other.
        assert!(a.charged_bytes().abs_diff(s.charged_bytes()) < 2 << 20);
    }

    #[test]
    fn slowlink_panel_aggregate_window_without_wmma_rows_carries_no_f16_activations() {
        // Only F32-tail rows (and a decode row): nothing takes the WMMA arm.
        let only_tails = PromptRouteLayout::new(&aggregate_dims(512, 1, 511)).unwrap();
        assert_eq!(only_tails.x_f16_bytes(), 0);
        assert_eq!(only_tails.tail_ids_bytes(), 511 * 10 * 4);
        let decode_only = PromptRouteLayout::new(&aggregate_dims(1, 1, 0)).unwrap();
        assert_eq!(decode_only.x_f16_bytes(), 0);
        // One more row is a WMMA row: the window now carries its F16 activations.
        let mixed = PromptRouteLayout::new(&aggregate_dims(513, 1, 511)).unwrap();
        assert_eq!(mixed.x_f16_bytes(), 513 * 2560 * 2);
        assert!(mixed.total_bytes() > only_tails.total_bytes() + 513 * 2560 * 2 - 1);
        // The slab route has neither F16 rows nor tail scratch.
        let slab = PromptRouteLayout::new(&dims(513)).unwrap();
        assert_eq!(slab.x_f16_bytes(), 0);
        assert_eq!(slab.tail_ids_bytes(), 0);
    }
    fn sym_iu4_dims(rows: usize, k1_rows: usize, tail_rows: usize) -> PromptRouteDims {
        PromptRouteDims {
            route: WindowRoute::AggregateSymIu4,
            ..aggregate_dims(rows, k1_rows, tail_rows)
        }
    }

    #[test]
    fn slowlink_panel_aggregate_sym_iu4_layout_charges_both_a4_sidecars_and_no_f16_window() {
        // The real split artifact's identity at the full 16384-row window with its 48 decode
        // rows: the F16 aggregate window rows and 2048-row slot map are gone, both A4 sidecars
        // (gate 20 * N * 72, down 5 * N * 10 * 72) and the saved F32 activations (the region's
        // `x_norm`, every route's) are charged.
        let f16 = PromptRouteLayout::new(&aggregate_dims(16_384, 48, 0)).unwrap();
        let s = PromptRouteLayout::new(&sym_iu4_dims(16_384, 48, 0)).unwrap();
        assert_eq!(s.route(), WindowRoute::AggregateSymIu4);
        assert_eq!(s.x_f16_bytes(), 0);
        assert_eq!(
            s.a4_sidecar_bytes(),
            20 * 16_384 * 72 + 5 * 16_384 * 10 * 72
        );
        assert_eq!(s.a4_sidecar_bytes(), 82_575_360);
        assert_eq!(s.a4_sidecar_bytes() / (1 << 20), 78);
        assert_eq!(s.x_norm_bytes(), 16_384 * 2560 * 4);
        assert_eq!(s.local_sorted, s.x_f16, "no panel-local F16 slot map");
        assert_eq!(s.x_f16, s.tail_ids + align_up(s.tail_ids_bytes()).unwrap());
        // Region: the F16 aggregate region minus its F16 rows and slot map.
        assert_eq!(f16.total_bytes() - s.total_bytes(), 83_886_080 + 2048 * 4);
        assert_eq!(s.total_bytes(), 386_175_232);
        assert_eq!(s.group_scatter_bytes(), f16.group_scatter_bytes());
        assert_eq!(s.charged_bytes(), 472_029_440);
        assert_eq!(
            s.charged_bytes(),
            s.total_bytes() + 82_575_360 + s.group_scatter_bytes()
        );
        // Against the slab route: the same A4 sidecars and region but the one decode tile's
        // `[10, hidden]` F32 scratch.
        let mut slab = dims(16_384);
        slab.k1_rows = 48;
        let slab = PromptRouteLayout::new(&slab).unwrap();
        assert_eq!(s.a4_sidecar_bytes(), slab.a4_sidecar_bytes());
        assert_eq!(s.total_bytes() - slab.total_bytes(), 10 * 2560 * 4);
        // The F16 aggregate route is byte-for-byte what it was (flag-off equivalence): the
        // layout numbers pinned by the aggregate tests above are unchanged.
        assert_eq!(f16.a4_sidecar_bytes(), 0);
        assert_eq!(f16.total_bytes(), 470_069_504);
        assert_eq!(f16.charged_bytes(), 473_348_352);
    }

    #[test]
    fn slowlink_panel_aggregate_sym_iu4_layout_keeps_the_aggregate_tail_and_decode_cohorts() {
        for (rows, k1, tail) in [
            (1usize, 1usize, 0usize),
            (3, 0, 3),
            (512, 1, 511),
            (513, 1, 511),
            (1537, 1, 8),
            (16384, 48, 3066),
        ] {
            let f16 = PromptRouteLayout::new(&aggregate_dims(rows, k1, tail)).unwrap();
            let s = PromptRouteLayout::new(&sym_iu4_dims(rows, k1, tail)).unwrap();
            assert_eq!(s.total_bytes() % ALIGN, 0, "{rows}/{k1}/{tail}");
            // Cohort scratch is the aggregate route's, field for field.
            assert_eq!(s.tail_ids_bytes(), f16.tail_ids_bytes());
            assert_eq!(s.cohort_counts_bytes(), f16.cohort_counts_bytes());
            assert_eq!(s.k1_bytes(), f16.k1_bytes());
            assert_eq!(
                s.cohort_counts - s.k1_tmp,
                f16.cohort_counts - f16.k1_tmp,
                "the decode cohort's scratch rows: {rows}/{k1}/{tail}"
            );
            // No F16 window whatever the row mix, and a window with WMMA-class rows costs the
            // F16 region less its F16 rows and slot map.
            assert_eq!(s.x_f16_bytes(), 0);
            assert_eq!(s.local_sorted, s.x_f16);
            if rows > k1 + tail {
                assert_eq!(
                    f16.total_bytes() - s.total_bytes(),
                    align_up(rows * 2560 * 2).unwrap() + 2048 * 4,
                    "{rows}/{k1}/{tail}"
                );
            } else {
                assert_eq!(f16.total_bytes(), s.total_bytes());
            }
            let fields = [
                s.x_norm,
                s.route_ids,
                s.topk_ids,
                s.topk_weights,
                s.sorted,
                s.tiles,
                s.inverse,
                s.counts,
                s.offsets,
                s.accumulator,
                s.shared_hidden,
                s.shared_scalar,
                s.k1_y0,
                s.k1_ids,
                s.k1_shared,
                s.tail_ids,
                s.x_f16,
                s.local_sorted,
                s.k1_tmp,
                s.cohort_counts,
                s.total,
            ];
            assert!(fields.iter().all(|f| f % ALIGN == 0), "{rows}/{k1}/{tail}");
            assert!(
                fields.windows(2).all(|w| w[0] <= w[1]),
                "{rows}/{k1}/{tail}"
            );
        }
        // Tail rows exist on both aggregate routes only, and with the K1 rows fit the window.
        assert!(PromptRouteLayout::new(&sym_iu4_dims(8, 4, 5)).is_err());
        assert!(PromptRouteLayout::new(&sym_iu4_dims(8, 4, 4)).is_ok());
    }

    #[test]
    fn slowlink_panel_window_route_labels_and_families() {
        assert_eq!(WindowRoute::SlabIu4.label(), "slab-iu4");
        assert_eq!(WindowRoute::Aggregate.label(), "aggregate");
        assert_eq!(WindowRoute::AggregateSymIu4.label(), "aggregate-sym-iu4");
        for (route, aggregate, quantizes) in [
            (WindowRoute::SlabIu4, false, true),
            (WindowRoute::Aggregate, true, false),
            (WindowRoute::AggregateSymIu4, true, true),
        ] {
            assert_eq!(route.is_aggregate_layout(), aggregate, "{route:?}");
            assert_eq!(route.quantizes_window(), quantizes, "{route:?}");
        }
    }

    #[test]
    fn slowlink_panel_aggregate_sym_iu4_slots_stay_global_under_panel_local_offsets() {
        // 800 tokens, one heavy expert holding 7000 of the 8000 slots (3+ bounded panels).
        let tokens = 800usize;
        let slots = tokens * K_TOP;
        let heavy = 7usize;
        let ids: Vec<i32> = (0..slots)
            .map(|s| {
                if s % 8 != 0 {
                    heavy as i32
                } else {
                    (100 + s / 8 % 60) as i32
                }
            })
            .collect();
        let (g, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        let heavy_panels: Vec<_> = panels
            .iter()
            .filter(|r| r.rows > 0 && r.expert_lo <= heavy && heavy < r.expert_hi)
            .collect();
        assert!(heavy_panels.len() >= 3, "{} panels", heavy_panels.len());
        let mut beyond_one_panel = false;
        let mut seen = vec![false; slots];
        for r in panels.iter().filter(|r| r.rows > 0) {
            assert!(r.rows <= PanelLimits::S1.max_grouped_rows);
            for row in r.row_base..r.row_base + r.rows {
                let slot = g.sorted[row];
                if slot < 0 {
                    continue;
                }
                let slot = slot as usize;
                // The panel's own Y row is local; the sorted entry the A4 producer and the IU4
                // GEMMs index with is the GLOBAL slot: gate/up reads the saved row `slot / 10`
                // of the window, down the sidecar record `slot / 1` of the window's slots.
                let y_row = row - r.row_base;
                assert!(y_row < r.rows);
                assert!(slot < slots && slot / K_TOP < tokens);
                assert_eq!(
                    g.inverse[slot] as usize, row,
                    "inverse is a GLOBAL grouped row"
                );
                beyond_one_panel |= slot >= PanelLimits::S1.max_grouped_rows;
                assert!(!seen[slot], "slot {slot} routed twice");
                seen[slot] = true;
            }
        }
        assert!(
            beyond_one_panel,
            "global slots exceed any panel-local numbering"
        );
        assert_eq!(seen.iter().filter(|s| **s).count(), slots);
        // The sidecar extents the launches pass: gate/up `x_row_div 10, x_src_rows = rows`,
        // down `x_row_div 1, x_src_rows = rows * 10`, both inside the builder GEMMs' slot bound.
        let l = PromptRouteLayout::new(&sym_iu4_dims(16_384, 48, 0)).unwrap();
        assert!(K_TOP * l.dims().rows <= 1 << 22);
        assert!(l.dims().rows * K_TOP * I32_BYTES <= l.topk_ids_bytes());
    }

    #[test]
    fn slowlink_panel_aggregate_layout_fields_are_aligned_and_disjoint() {
        for (rows, k1, tail) in [
            (1usize, 1usize, 0usize),
            (3, 0, 3),
            (512, 1, 511),
            (1537, 1, 8),
            (16384, 1, 511),
        ] {
            let l = PromptRouteLayout::new(&aggregate_dims(rows, k1, tail)).unwrap();
            assert_eq!(l.total_bytes() % ALIGN, 0, "{rows}/{k1}/{tail}");
            let fields = [
                l.x_norm,
                l.route_ids,
                l.topk_ids,
                l.topk_weights,
                l.sorted,
                l.tiles,
                l.inverse,
                l.counts,
                l.offsets,
                l.accumulator,
                l.shared_hidden,
                l.shared_scalar,
                l.k1_y0,
                l.k1_ids,
                l.k1_shared,
                l.tail_ids,
                l.x_f16,
                l.local_sorted,
                l.k1_tmp,
                l.cohort_counts,
                l.total,
            ];
            assert!(fields.iter().all(|f| f % ALIGN == 0), "{rows}/{k1}/{tail}");
            assert!(
                fields.windows(2).all(|w| w[0] <= w[1]),
                "{rows}/{k1}/{tail}"
            );
            // The F16 rows end where the next field starts.
            assert_eq!(l.local_sorted - l.x_f16, align_up(l.x_f16_bytes()).unwrap());
        }
    }

    #[test]
    fn slowlink_panel_cohort_row_bounds_follow_the_route() {
        // Tail rows exist only on the aggregate route and, with the K1 rows, fit the window.
        let mut slab = dims(4096);
        slab.tail_rows = 1;
        assert!(PromptRouteLayout::new(&slab).is_err());
        assert!(PromptRouteLayout::new(&aggregate_dims(8, 4, 5)).is_err());
        assert!(PromptRouteLayout::new(&aggregate_dims(8, 4, 4)).is_ok());
        // Cohort slot counts are one word an expert beside the K2 counts (no packing bound).
        assert!(PromptRouteLayout::new(&aggregate_dims(4096, 1, 2047)).is_ok());
        // The slab K1 cohort has no packed-field cap either (its counts are their own words):
        // it may fill the whole window and only cannot exceed it.
        let mut k1 = dims(4096);
        k1.k1_rows = 205;
        assert!(PromptRouteLayout::new(&k1).is_ok());
        k1.k1_rows = 4096;
        assert!(PromptRouteLayout::new(&k1).is_ok());
        k1.k1_rows = 4097;
        assert!(PromptRouteLayout::new(&k1).is_err());
    }

    #[test]
    fn slowlink_panel_tile_kinds_name_their_numerical_class_and_window_compatibility() {
        assert!(
            TileKind::SlabK2.is_k2()
                && TileKind::AggregateWmma.is_k2()
                && TileKind::AggregateIu4.is_k2()
        );
        for cohort in [
            TileKind::SlabK1Tail,
            TileKind::AggregateF32Tail,
            TileKind::Decode,
        ] {
            assert!(!cohort.is_k2(), "{cohort:?}");
        }
        let routes = [
            WindowRoute::SlabIu4,
            WindowRoute::Aggregate,
            WindowRoute::AggregateSymIu4,
        ];
        // (kind, the windows it may run in): the compatibility matrix.
        let matrix: [(TileKind, [bool; 3]); 6] = [
            (TileKind::SlabK2, [true, false, false]),
            (TileKind::SlabK1Tail, [true, false, false]),
            (TileKind::AggregateWmma, [false, true, false]),
            (TileKind::AggregateIu4, [false, false, true]),
            (TileKind::AggregateF32Tail, [false, true, true]),
            (TileKind::Decode, [true, true, true]),
        ];
        for (kind, allowed) in matrix {
            for (route, ok) in routes.iter().zip(allowed) {
                assert_eq!(kind.compatible_with(*route), ok, "{kind:?} in {route:?}");
            }
        }
        assert_eq!(TileKind::SlabK2.route(), Some(WindowRoute::SlabIu4));
        assert_eq!(TileKind::SlabK1Tail.route(), Some(WindowRoute::SlabIu4));
        assert_eq!(
            TileKind::AggregateWmma.route(),
            Some(WindowRoute::Aggregate)
        );
        assert_eq!(
            TileKind::AggregateF32Tail.route(),
            Some(WindowRoute::Aggregate)
        );
        assert_eq!(
            TileKind::AggregateIu4.route(),
            Some(WindowRoute::AggregateSymIu4)
        );
        assert_eq!(TileKind::Decode.route(), None);
        assert_eq!(AGGREGATE_MIN_TILE_ROWS, 512);
        assert_eq!(
            AGGREGATE_MIN_TILE_ROWS,
            rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS,
            "the IU4 K2 tile starts at the F16 WMMA K2 tile's route rows"
        );
    }

    #[test]
    fn slowlink_panel_tile_admission_checks_kind_and_lowering_family_against_the_window() {
        // A tile of `kind` with its own lowering `family` in a `window`.
        let ok = |kind, family, window| admit_into_window(kind, family, window).is_ok();
        let (slab, f16, sym) = (
            WindowRoute::SlabIu4,
            WindowRoute::Aggregate,
            WindowRoute::AggregateSymIu4,
        );
        assert!(ok(TileKind::AggregateIu4, sym, sym));
        assert!(!ok(TileKind::AggregateIu4, sym, f16));
        assert!(!ok(TileKind::AggregateWmma, f16, sym));
        assert!(ok(TileKind::AggregateWmma, f16, f16));
        // The F32 tail (family aggregate) rides in both aggregate windows, never the slab one.
        assert!(ok(TileKind::AggregateF32Tail, f16, f16));
        assert!(ok(TileKind::AggregateF32Tail, f16, sym));
        assert!(!ok(TileKind::AggregateF32Tail, f16, slab));
        // A decode tile follows its own layer family: aggregate decode in either aggregate
        // window, slab decode only in the slab window.
        assert!(ok(TileKind::Decode, f16, f16));
        assert!(ok(TileKind::Decode, f16, sym));
        assert!(!ok(TileKind::Decode, f16, slab));
        assert!(ok(TileKind::Decode, slab, slab));
        assert!(!ok(TileKind::Decode, slab, sym));
        assert!(!ok(TileKind::Decode, slab, f16));
    }

    #[test]
    fn slowlink_panel_counts_are_pad16_and_offsets_monotone() {
        let mut rng = Rng(0x5eed);
        let ids = random_ids(&mut rng, 700, 40);
        let (g, p) = plan(&ids, Vec::new());
        assert!(p.counts_padded.iter().all(|c| c % 16 == 0));
        assert!(p.offsets.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(p.total_rows, g.start[N_EXP]);
        assert_eq!(p.offsets[N_EXP], p.total_rows);
        assert!(LayerRoutePlan::from_counts(&[3], Vec::new(), BarrierCost::default()).is_err());
        assert!(LayerRoutePlan::from_counts(&[-16], Vec::new(), BarrierCost::default()).is_err());
        assert!(LayerRoutePlan::from_counts(&[16], vec![1], BarrierCost::default()).is_err());
    }

    #[test]
    fn slowlink_panel_schedule_is_bounded_contiguous_and_within_bindings() {
        let mut rng = Rng(0xfeed_f00d);
        for trial in 0..24 {
            let tokens = [1usize, 7, 300, 1536][trial % 4];
            let hot = rng.below(N_EXP - 4);
            let ids = random_ids(&mut rng, tokens, hot);
            let (g, p) = plan(&ids, Vec::new());
            for width in [32usize, 64, 512] {
                for max_rows in [16usize, 48, 256, 2048] {
                    let limits = PanelLimits {
                        experts_per_slot: width,
                        max_grouped_rows: max_rows,
                    };
                    let bindings = runs(width);
                    let panels = p.schedule(&bindings, limits).unwrap();
                    assert!(!panels.is_empty());
                    let mut covered = vec![false; p.total_rows];
                    for (i, panel) in panels.iter().enumerate() {
                        assert_eq!(panel.index, i);
                        assert!(panel.rows <= limits.max_grouped_rows, "scratch hard bound");
                        assert_eq!(panel.rows % 16, 0);
                        assert_eq!(panel.row_base % 16, 0);
                        let (lo, hi) = bindings[panel.binding];
                        assert_eq!((panel.bind_lo, panel.bind_hi), (lo, hi));
                        if panel.rows > 0 {
                            assert!(
                                panel.row_base >= p.offsets[lo]
                                    && panel.row_base + panel.rows <= p.offsets[hi]
                            );
                            assert!(panel.expert_lo >= lo && panel.expert_hi <= hi);
                            assert!(panel.expert_lo < panel.expert_hi);
                            for r in panel.row_base..panel.row_base + panel.rows {
                                assert!(!covered[r], "row {r} scheduled twice");
                                covered[r] = true;
                                let e = p.expert_of_row(r);
                                assert!(e >= panel.expert_lo && e < panel.expert_hi);
                            }
                        }
                    }
                    assert!(covered.iter().all(|&c| c), "a grouped row was dropped");
                    // Panels ascend by row and binding.
                    assert!(panels.windows(2).all(|w| w[0].binding <= w[1].binding
                        && w[0].row_base + w[0].rows <= w[1].row_base));
                    // Ownership partitions [0, n_exp) in panel order.
                    let mut next = 0;
                    for panel in &panels {
                        assert_eq!(panel.own_lo, next);
                        assert!(panel.own_hi >= panel.own_lo);
                        next = panel.own_hi;
                    }
                    assert_eq!(next, N_EXP);
                    let _ = &g;
                }
            }
        }
    }

    #[test]
    fn slowlink_panel_single_expert_splits_across_panels() {
        // 5000 slots on one expert: no unbounded exception, 3 panels of <= 2048 rows.
        let ids = vec![7i32; 5000];
        let (_, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 3);
        assert_eq!(panels.iter().map(|x| x.rows).sum::<usize>(), 5008);
        assert!(panels
            .iter()
            .all(|x| x.rows <= 2048 && x.expert_lo == 7 && x.expert_hi == 8));
        // The split expert's invalid-row slots are owned by exactly its first panel.
        assert_eq!((panels[0].own_lo, panels[0].own_hi), (0, 8));
        assert_eq!((panels[1].own_lo, panels[1].own_hi), (8, 8));
        assert_eq!((panels[2].own_lo, panels[2].own_hi), (8, N_EXP));
    }

    #[test]
    fn slowlink_panel_sparse_duplicate_invalid_ids_and_empty_windows() {
        // Nothing routed at all: one zero-row combine-only panel owning everything.
        let ids = vec![-1i32; 40];
        let (_, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 1);
        let only = panels[0];
        assert_eq!(
            (only.rows, only.own_lo, only.own_hi, only.k1),
            (0, 0, N_EXP, false)
        );
        // Sparse: experts 3 and 500 only: two panels in two bindings, the middle ones skipped.
        let mut ids = vec![3i32; 10];
        ids.extend(vec![500i32; 10]);
        ids.extend([3, 3, 500, N_EXP as i32, -5, i32::MAX, i32::MIN, 3, 3, 3]);
        let (_, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 2);
        assert_eq!((panels[0].bind_lo, panels[1].bind_lo), (0, 480));
        // An expert outside every binding is refused rather than dropped.
        let err = p.schedule(&[(0, 100)], PanelLimits::S1).unwrap_err();
        assert!(format!("{err:?}").contains("in no binding"));
        // Malformed bindings and limits.
        assert!(p.schedule(&[], PanelLimits::S1).is_err());
        assert!(p.schedule(&[(10, 20), (15, 30)], PanelLimits::S1).is_err());
        assert!(p.schedule(&[(20, 10)], PanelLimits::S1).is_err());
        assert!(p.schedule(&[(0, N_EXP + 1)], PanelLimits::S1).is_err());
        let bad = PanelLimits {
            experts_per_slot: 32,
            max_grouped_rows: 100,
        };
        assert!(p.schedule(&runs(32), bad).is_err());
    }

    #[test]
    fn slowlink_panel_k1_cohort_binding_without_rows_gets_a_zero_row_panel() {
        // K2 rows only in binding 0; the K1 tail routes to experts 40 (binding 1) and 3.
        let ids = vec![3i32; 20];
        let (_, p) = plan(&ids, vec![3, 40]);
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 2);
        assert!(panels[0].k1 && panels[0].rows == 32);
        assert!(panels[1].k1 && panels[1].rows == 0 && panels[1].bind_lo == 32);
        // K1 runs once per binding even when the binding's rows split into several panels.
        let ids = vec![3i32; 5000];
        let (_, p) = plan(&ids, vec![3]);
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.iter().filter(|x| x.k1).count(), 1);
        assert!(panels[0].k1);
        // K1-only window: no K2 rows, the one zero-row panel per K1 binding.
        let (_, p) = plan(&vec![-1i32; 10], vec![5, 500]);
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 2);
        assert!(panels.iter().all(|x| x.rows == 0 && x.k1));
        assert_eq!(panels.last().unwrap().own_hi, N_EXP);
    }

    #[test]
    fn slowlink_panel_every_slot_is_owned_once_and_inverse_maps_into_its_panel() {
        let mut rng = Rng(0xabcd_ef01);
        for _ in 0..6 {
            let tokens = 100 + rng.below(300);
            let hot = rng.below(N_EXP - 4);
            let ids = random_ids(&mut rng, tokens, hot);
            let (g, p) = plan(&ids, Vec::new());
            for (width, max_rows) in [(32usize, 2048usize), (32, 64), (64, 16), (512, 512)] {
                let panels = p
                    .schedule(
                        &runs(width),
                        PanelLimits {
                            experts_per_slot: width,
                            max_grouped_rows: max_rows,
                        },
                    )
                    .unwrap();
                for (slot, &id) in ids.iter().enumerate() {
                    let valid = (0..N_EXP as i32).contains(&id);
                    let owners: Vec<usize> = panels
                        .iter()
                        .filter(|q| {
                            slowlink_panel_owns(
                                id,
                                g.inverse[slot],
                                N_EXP,
                                p.total_rows,
                                q.row_base,
                                q.rows,
                                q.own_lo,
                                q.own_hi,
                            )
                        })
                        .map(|q| q.index)
                        .collect();
                    assert_eq!(owners.len(), usize::from(valid), "slot {slot} id {id}");
                    if valid {
                        let q = &panels[owners[0]];
                        let row = g.inverse[slot] as usize;
                        assert!(row >= q.row_base && row < q.row_base + q.rows);
                        // Inverse mapping: the grouped row names its slot and its expert.
                        assert_eq!(g.sorted[row], slot as i32);
                        assert_eq!(p.expert_of_row(row), id as usize);
                        assert!(id as usize >= q.expert_lo && (id as usize) < q.expert_hi);
                    } else {
                        assert_eq!(g.inverse[slot], -1);
                    }
                }
                // Stability: rows of one expert ascend with the flat slot.
                for e in 0..N_EXP {
                    let rows = &g.sorted[g.start[e]..g.start[e + 1]];
                    let live: Vec<i32> = rows.iter().copied().filter(|&s| s >= 0).collect();
                    assert!(live.windows(2).all(|w| w[0] < w[1]));
                }
            }
        }
    }

    #[test]
    fn slowlink_panel_invalid_row_valid_id_slots_are_owned_by_one_panel() {
        // A valid id whose grouped row is invalid (the canonical kernel multiplies +0 by its
        // weight) belongs to exactly the panel whose ownership range holds its id.
        let ids = vec![3i32; 5000];
        let (_, p) = plan(&ids, vec![]);
        let mut ids2 = ids.clone();
        ids2.extend([200, 300, 511]);
        let (_, p2) = plan(&ids2, vec![]);
        for (plan, bad_id) in [(&p, 3i32), (&p2, 200), (&p2, 300), (&p2, 511), (&p2, 0)] {
            let panels = plan.schedule(&runs(32), PanelLimits::S1).unwrap();
            for bad_row in [-1i32, plan.total_rows as i32 + 7, i32::MAX] {
                let owners = panels
                    .iter()
                    .filter(|q| {
                        slowlink_panel_owns(
                            bad_id,
                            bad_row,
                            N_EXP,
                            plan.total_rows,
                            q.row_base,
                            q.rows,
                            q.own_lo,
                            q.own_hi,
                        )
                    })
                    .count();
                assert_eq!(owners, 1, "id {bad_id} row {bad_row}");
            }
        }
    }

    #[test]
    fn slowlink_panel_limits_bound_the_scratch() {
        let l = PanelLimits::S1;
        assert_eq!(l.scratch_bytes(640, 2560).unwrap(), 2048 * (640 + 2560) * 2);
        assert!(PanelLimits {
            experts_per_slot: 0,
            max_grouped_rows: 16
        }
        .validate()
        .is_err());
        assert!(PanelLimits {
            experts_per_slot: 32,
            max_grouped_rows: 0
        }
        .validate()
        .is_err());
        assert!(PanelLimits {
            experts_per_slot: 32,
            max_grouped_rows: 17
        }
        .validate()
        .is_err());
        assert!(PanelLimits {
            experts_per_slot: 32,
            max_grouped_rows: usize::MAX & !15
        }
        .scratch_bytes(640, 2560)
        .is_err());
    }

    #[test]
    fn slowlink_panel_count_arrays_carry_k2_padding_and_cohort_touch_separately() {
        // Two device arrays, one word an expert: the K2 padded counts and the cohort slot counts.
        let k2 = [0, 32, 48, 0, 16, (1 << 20) - 16];
        let cohort = [0, 0, 3, 1, 0, 480];
        let experts = cohort_experts(&k2, &cohort).unwrap();
        assert_eq!(experts, vec![2, 3, 5]);
        assert!(cohort_experts(&[-1], &[0]).is_err());
        assert!(cohort_experts(&[0], &[-1]).is_err());
        assert!(cohort_experts(&[0, 0], &[0]).is_err());
        // The counts are exactly what the plan consumes.
        let plan =
            LayerRoutePlan::from_counts(&k2[..5], experts[..2].to_vec(), BarrierCost::default())
                .unwrap();
        assert_eq!(plan.total_rows, 96);
        assert_eq!(plan.k1_experts, vec![2, 3]);
    }

    #[test]
    fn slowlink_panel_cohort_counts_are_not_bounded_by_the_k2_count_word() {
        // No packing: a worst-case aggregate window (every F32-tail segment of a window plus
        // the decode rows) and a large slab K1 cohort are admitted, and only a window whose
        // padded slot count cannot be an i32 word is refused before any mutation.
        assert!(PromptRouteLayout::new(&aggregate_dims(16_384, 48, 3066)).is_ok());
        let mut d = dims(16_384);
        d.k1_rows = 1000;
        assert!(PromptRouteLayout::new(&d).is_ok());
        assert!(PromptRouteLayout::new(&dims(214_748_364)).is_err());
        // Each cohort window reserves one count word per expert beside the K2 counts.
        let none = PromptRouteLayout::new(&{
            let mut d = dims(1536);
            d.k1_rows = 0;
            d
        })
        .unwrap();
        let some = PromptRouteLayout::new(&dims(1536)).unwrap();
        assert!(some.total_bytes() >= none.total_bytes() + 512 * 4);
    }

    // ───────────── SP (short-prompt prefill) primitives: layout, class, delegation ─────────────

    /// Every pre-SP field offset of a layout, in declaration order, plus the legacy total.
    fn legacy_offsets(l: &PromptRouteLayout) -> [usize; 20] {
        [
            l.grouped_rows,
            l.x_norm,
            l.route_ids,
            l.topk_ids,
            l.topk_weights,
            l.sorted,
            l.tiles,
            l.inverse,
            l.counts,
            l.offsets,
            l.accumulator,
            l.shared_hidden,
            l.shared_scalar,
            l.k1_y0,
            l.k1_ids,
            l.k1_shared,
            l.tail_ids,
            l.x_f16,
            l.local_sorted,
            l.k1_tmp,
        ]
    }

    #[test]
    fn slowlink_sp_layout_prefix_is_the_non_sp_layout_and_fields_are_aligned() {
        for rows in [5usize, 16, 768, 1024, 1536, 2048, 4096, 5376] {
            let d = aggregate_dims(rows, 0, 0);
            let base = PromptRouteLayout::new(&d).unwrap();
            let sp = PromptRouteLayout::with_sp_rows(&d).unwrap();
            // `new` carries no SP field and does not move: its SP offsets sit at the region end
            // with zero bytes, and the SP builder only appends.
            assert!(!base.has_sp());
            assert_eq!((base.sp_rows_bytes(), base.sp_inverse_bytes()), (0, 0));
            assert_eq!((base.sp_rows, base.sp_inverse), (base.total, base.total));
            assert_eq!(
                base,
                PromptRouteLayout::new(&d).unwrap(),
                "new is deterministic"
            );
            // Non-SP layout byte-equal to `new`: every legacy offset and the cohort-count field
            // of the SP layout is the non-SP one, and only the two appended fields grow the total.
            assert_eq!(legacy_offsets(&sp), legacy_offsets(&base), "rows {rows}");
            assert_eq!(sp.cohort_counts, base.cohort_counts);
            assert_eq!(sp.dims(), base.dims());
            assert!(sp.has_sp());
            // H_max is the grouped-row bound of 10 * rows slots: each slot adds at most one
            // 16-row pad run, and at most `n_exp` experts pad (`slots + 15 * min(slots, 512)`).
            let slots = 10 * rows;
            let h_max = (slots + 15 * slots.min(512)).div_ceil(16) * 16;
            assert_eq!(sp.grouped_rows_bound(), h_max, "rows {rows}");
            assert_eq!(sp.sp_rows_bytes(), h_max * 2560 * BF16_BYTES);
            assert_eq!(sp.sp_inverse_bytes(), rows * 10 * I32_BYTES);
            // The two SP fields are carved last, aligned and disjoint.
            assert_eq!(sp.sp_rows, base.total_bytes());
            assert_eq!(
                sp.sp_inverse,
                sp.sp_rows + align_up(sp.sp_rows_bytes()).unwrap()
            );
            assert_eq!(
                sp.total_bytes(),
                sp.sp_inverse + align_up(sp.sp_inverse_bytes()).unwrap()
            );
            assert_eq!(
                sp.total_bytes(),
                base.total_bytes()
                    + align_up(sp.sp_rows_bytes()).unwrap()
                    + align_up(sp.sp_inverse_bytes()).unwrap()
            );
            let fields = [
                sp.x_norm,
                sp.route_ids,
                sp.topk_ids,
                sp.topk_weights,
                sp.sorted,
                sp.tiles,
                sp.inverse,
                sp.counts,
                sp.offsets,
                sp.accumulator,
                sp.shared_hidden,
                sp.shared_scalar,
                sp.k1_y0,
                sp.k1_ids,
                sp.k1_shared,
                sp.tail_ids,
                sp.x_f16,
                sp.local_sorted,
                sp.k1_tmp,
                sp.cohort_counts,
                sp.sp_rows,
                sp.sp_inverse,
                sp.total,
            ];
            assert!(fields.iter().all(|f| f % ALIGN == 0), "rows {rows}");
            assert!(fields.windows(2).all(|w| w[0] <= w[1]), "rows {rows}");
            // The SP fields are strictly new bytes (the hybrid table cannot overlap `x_f16`).
            assert!(sp.sp_rows >= sp.local_sorted + AGGREGATE_PANEL_ROWS * I32_BYTES);
            // A window with SP fields and the F16 activations is the WMMA window.
            assert_eq!(sp.x_f16_bytes(), rows * 2560 * F16_BYTES);
        }
    }

    #[test]
    fn slowlink_sp_layout_refuses_cohort_rows_and_other_routes() {
        assert!(PromptRouteLayout::with_sp_rows(&aggregate_dims(1024, 0, 0)).is_ok());
        assert!(PromptRouteLayout::with_sp_rows(&aggregate_dims(1024, 1, 0)).is_err());
        assert!(PromptRouteLayout::with_sp_rows(&aggregate_dims(1024, 0, 1)).is_err());
        assert!(PromptRouteLayout::with_sp_rows(&sym_iu4_dims(1024, 0, 0)).is_err());
        assert!(PromptRouteLayout::with_sp_rows(&dims(1024)).is_err());
        let mut degenerate = aggregate_dims(1024, 0, 0);
        degenerate.rows = 0;
        assert!(PromptRouteLayout::with_sp_rows(&degenerate).is_err());
    }

    /// The product S1 profile's route-region budget: the aggregate route at its full window
    /// (16,896 rows, K1 48, F32-tail 3,066), which is what `census.route_storage` charges.
    fn product_route_region_bytes() -> usize {
        PromptRouteLayout::new(&aggregate_dims(16_896, 48, 3_066))
            .unwrap()
            .total_bytes()
    }

    /// Largest window rows whose SP layout fits `budget` bytes.
    fn max_sp_rows_within(budget: usize) -> usize {
        let fits = |rows: usize| {
            PromptRouteLayout::with_sp_rows(&aggregate_dims(rows, 0, 0))
                .is_ok_and(|l| l.total_bytes() <= budget)
        };
        assert!(fits(1));
        let (mut lo, mut hi) = (1usize, 16_896usize);
        assert!(
            !fits(hi),
            "the SP layout at the full window cannot fit its own budget"
        );
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    #[test]
    fn slowlink_sp_fit_limit_on_the_product_profile_is_pinned() {
        let budget = product_route_region_bytes();
        // The design's 462.3 MiB route region of the product profile.
        assert_eq!(budget, 484_716_032, "product route region bytes");
        let limit = max_sp_rows_within(budget);
        assert_eq!(
            limit, 5_593,
            "max SP window rows that fit the product route region"
        );
        // The pin is exact: one more row does not fit, the limit itself does.
        let total = |rows: usize| {
            PromptRouteLayout::with_sp_rows(&aggregate_dims(rows, 0, 0))
                .unwrap()
                .total_bytes()
        };
        assert!(total(limit) <= budget && total(limit + 1) > budget);
        // The design's cap (5,376 rows, §5/§6) fits with margin; the memory table's rows too.
        for rows in [1024usize, 4096, 5376] {
            assert!(total(rows) <= budget, "rows {rows}");
        }
        assert!(total(5376) < budget);
        // MiB of the SP window layout at 1K / 4K / 5,376 rows (design §6: 115 / 349 / 446).
        let mib = |rows: usize| total(rows) as f64 / (1024.0 * 1024.0);
        assert!((mib(1024) - 115.0).abs() < 1.0, "{}", mib(1024));
        assert!((mib(4096) - 349.0).abs() < 1.0, "{}", mib(4096));
        assert!((mib(5376) - 446.0).abs() < 1.0, "{}", mib(5376));
    }

    #[test]
    fn slowlink_sp_tile_class_k2_izes_the_f32_tail_only_under_sp() {
        use TileClass::{Tail, K1, K2};
        let kinds = [
            TileKind::SlabK2,
            TileKind::SlabK1Tail,
            TileKind::AggregateWmma,
            TileKind::AggregateIu4,
            TileKind::AggregateF32Tail,
            TileKind::Decode,
        ];
        // Without SP every kind keeps today's class (the K2 kinds are `TileKind::is_k2`).
        let plain = [K2, K1, K2, K2, Tail, K1];
        for (kind, want) in kinds.into_iter().zip(plain) {
            assert_eq!(sp_tile_class(kind, false).unwrap(), want, "{kind:?}");
            assert_eq!(want == K2, kind.is_k2(), "{kind:?}");
        }
        // Under SP: the F32 tail becomes K2, the WMMA tile stays K2, a decode tile is an error
        // (an SP window never holds one), and so is every kind outside the aggregate F16 route.
        assert_eq!(sp_tile_class(TileKind::AggregateF32Tail, true).unwrap(), K2);
        assert_eq!(sp_tile_class(TileKind::AggregateWmma, true).unwrap(), K2);
        for kind in [
            TileKind::Decode,
            TileKind::SlabK2,
            TileKind::SlabK1Tail,
            TileKind::AggregateIu4,
        ] {
            assert!(sp_tile_class(kind, true).is_err(), "{kind:?}");
        }
        // F32-tail is a cohort tile ONLY without SP.
        assert_eq!(
            sp_tile_class(TileKind::AggregateF32Tail, false).unwrap(),
            Tail
        );
        for kind in kinds {
            if let Ok(class) = sp_tile_class(kind, true) {
                assert_eq!(class, K2, "{kind:?}: SP tiles are all K2");
            }
        }
    }

    /// The source text of the item that starts at `head` and ends at its first column-0 `}`.
    fn item_source<'a>(src: &'a str, head: &str) -> &'a str {
        let start = src.find(head).unwrap_or_else(|| panic!("{head} not found"));
        let len = src[start..].find("\n}\n").expect("item end");
        &src[start..start + len + 2]
    }

    #[test]
    fn slowlink_finish_route_plan_delegates_to_the_hooked_body_with_a_noop() {
        let src = include_str!("slowlink_prefill.rs");
        let plain = item_source(src, "pub fn finish_route_plan(");
        assert!(
            plain.contains("finish_route_plan_with(gpu, routes, limits, &mut |_, _| Ok(()))"),
            "{plain}"
        );
        // It owns no barrier work of its own.
        for forbidden in ["scatter", "sync_download", "Instant", "planned"] {
            assert!(!plain.contains(forbidden), "{forbidden}");
        }
        // The hooked body keeps the original sequence; the hook runs once, after the stable
        // scatter and before the cohort counts and the one barrier.
        let with = item_source(src, "pub fn finish_route_plan_with(");
        let at = |needle: &str| with.find(needle).unwrap_or_else(|| panic!("{needle}"));
        let scatter = at("hip(scatter(");
        let hook = at("after_scatter(gpu, &*routes)?;");
        let cohort = at("let cohort = (layout.dims.k1_rows");
        let barrier = at("let started = Instant::now();");
        let planned = at("routes.planned = true;");
        assert!(scatter < hook && hook < cohort && cohort < barrier && barrier < planned);
        assert_eq!(with.matches("after_scatter(").count(), 1);
    }

    #[test]
    fn slowlink_execute_panel_passes_the_model_down_scratch_to_the_shared_gemm_block() {
        // `execute_panel` runs the extracted GEMM block over `params.y_down_grouped` (so it is the
        // pre-SP bytes) and the SP rows entry over the caller's hybrid row view.
        let src = include_str!("slowlink_prefill.rs");
        let panel = item_source(src, "pub fn execute_panel(");
        assert!(panel.contains(
            "panel_gemms(gpu, weights, routes, range, &params, params.y_down_grouped)?;"
        ));
        assert!(!panel.contains("gemm_mq4g128v2_moe_grouped_top10_xf16"));
        let rows = item_source(src, "pub fn execute_panel_rows(");
        assert!(rows.contains("panel_gemms(gpu, weights, routes, range, &params, y_rows)"));
        assert!(
            !rows.contains("combine"),
            "the SP panel entry never combines"
        );
        let gemms = item_source(src, "fn panel_gemms(");
        assert_eq!(
            gemms.matches("y_down,").count(),
            2,
            "both down GEMMs write y_down"
        );
        assert!(!gemms.contains("y_down_grouped"));
    }
}
