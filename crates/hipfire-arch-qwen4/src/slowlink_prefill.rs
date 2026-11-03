// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 slow-link layer-major prompt prefill.
//!
//! The incumbent prefills a long prompt chunk-major: every chunk walks all
//! layers, and on a narrow host link every chunk therefore re-streams every
//! host-resident routed expert. This route walks the same prompt layer-major:
//! a window of up to [`S1_WINDOW_ROWS`] rows crosses each layer once, in the
//! original 1,536-row mixer tiles, so each host layer's expert payload crosses
//! the link **once per window** while layer `L`'s tail and layer `L+1`'s mixer
//! compute under the copies of `L+1`'s weights.
//!
//! # Ownership
//!
//! * the typed per-layer operations are built by
//!   [`crate::gpu_forward::LayerPhaseEnv`], the same builder the incumbent
//!   loop uses, so no second arithmetic lowering exists;
//! * the expert page ring and the layer-ahead schedule are
//!   [`crate::gpu_forward::panel_stage`];
//! * route preparation, the single count barrier per layer, panel GEMMs, the
//!   canonical BF16 combine and the shared expert are
//!   `hipfire_dispatch::pipeline::slowlink_prefill`;
//! * admission, the selector truth table and every byte census are
//!   [`crate::slowlink_admission`];
//! * the entry snapshot, per-boundary snapshots, publication and rollback are
//!   the bundle's `prompt_txn_*` transaction.
//!
//! This module owns the schedule: the pure plan (tiles, windows, boundaries),
//! the request selector, the layer-major run and the receipt.
//!
//! # Only host layers are layer-major
//!
//! A device-resident layer reads its experts in place at full speed. Such a
//! layer runs exactly the incumbent per-tile steps (prefix, MoE, HyperWrite) on
//! its tile, so its arithmetic is the incumbent's. A host layer splits at its
//! route barrier: every tile's mixer prefix and route preparation first, then
//! one count barrier, the panels in ascending expert order, and every tile's
//! shared expert and HyperWrite.
//!
//! # Numerical manifest
//!
//! Tile `t` of a window is the incumbent chunk `t` of the same chronology: the
//! tile row counts, the routing-row signature (`Gpu::route_rows`), the HC
//! stream tag (BF16 or F32), the QSA/GDN positions and the PLE history are all
//! per tile. Host expert panels use the canonical ordered combine.
//! Cross-route qualification uses long-context route-relative KL at
//! beta's own benign-route scale, same-route determinism and coherent text;
//! it does not claim byte equality against the incumbent's row grouping.

#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
use crate::bundle::SlowlinkFailPoint;
use crate::bundle::{PromptTxn, Qwen4Bundle};
use crate::config::Qwen4Config;
use crate::gpu_forward::panel_stage::{
    ExpertGroupPlan, ExpertPanelRing, OverlapSummary, PanelBindingParts, PanelSchedule, PanelStats,
    PanelTicket, RingSchedule,
};
use crate::gpu_forward::{
    f32_view, invalid, layer_desc, layer_scratch, matrix_view, program_dims, router_columns,
    stage_ple_rows, view, LayerPhaseEnv, PleUploadMode, Qwen4GpuForward, Qwen4GpuForwardError,
};
use crate::ple::PleHistory;
use crate::ple_stage::ple_async_upload;
use crate::program::{
    execute_final_hyper, execute_lm_head, validate_final_hyper, validate_lm_head, Qwen4FfnBinding,
    Qwen4ProgramDims,
};
use crate::slowlink_admission::{
    qualify_request, ArenaField, CaptureKind, RequestBypass, RequestShape, SlowlinkMode,
    SlowlinkProfile, EXPERTS_PER_PAGE, S1_MAX_BOUNDARY_SNAPSHOTS,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::slowlink_prefill as dsp;
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, validate_steps, EmbeddingOp, Step,
};
use hipfire_runtime::external_rows::{RowFetch, RowTicket};
use rdna_compute::tensor_ops::{argmax_f32, ensure_qsa_gathered_wmma_workspace, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::ops::Range;
use std::time::Instant;

mod sp;
use crate::slowlink_sp::{SpBypass, SpDecline, SpMode, SpProfile};
use sp::{
    sp_floor_verdict, sp_load_decision, sp_receipt_json, sp_request_verdict, sp_window_routes,
    sp_window_verdicts, SpBind, SpBindPlan, SpRuntime, SpSummary, SpWindowEntry, SpWindowReport,
    TouchedRuntime,
};

type R<T> = Result<T, Qwen4GpuForwardError>;

/// Receipt schema string (`V-RECEIPT-API.md`).
pub(crate) const RECEIPT_SCHEMA: &str = "hipfire.qwen4.slowlink_receipt.v1";

// ── the pure plan ────────────────────────────────────────────────────

/// What one tile's launch signature depends on at runtime: the routing rows
/// every row-count routing decision of the tile sees and the HC stream tag.
/// Both are decided once per tile from [`Gpu::route_rows`] and
/// [`Gpu::qwen4_bf16_streams`] under the tile's scope, exactly as the
/// incumbent decides them per chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileRouting {
    pub(crate) route_rows: usize,
    pub(crate) bf16_streams: bool,
}

/// One mixer tile: the unit the incumbent would have run as one chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TilePlan {
    /// Index in the request's chronological tile list.
    pub(crate) index: usize,
    pub(crate) window: usize,
    /// First request row (0 = first prompt row after the entry position).
    pub(crate) prompt_row: usize,
    /// First row inside its window (the arena row base).
    pub(crate) window_row: usize,
    /// Offset (F32 elements) of the tile's HC-stream region in the window
    /// arena. Each tile's region is sized by its own tag (BF16 rows take half
    /// the bytes of F32 rows), so the arena holds exactly the tagged spans.
    pub(crate) stream_offset: usize,
    pub(crate) rows: usize,
    /// Absolute position of the tile's first row.
    pub(crate) position: usize,
    pub(crate) routing: TileRouting,
    /// `Some(i)` when the tile ends at the window's `i`-th strictly-inside
    /// session boundary (a snapshot cut the layer's state is copied at).
    pub(crate) ends_at_cut: Option<usize>,
    pub(crate) last_in_window: bool,
}

impl TilePlan {
    /// Absolute position one past the tile's last row.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn end_position(&self) -> usize {
        self.position + self.rows
    }
}

/// One window: the rows that share one expert sweep, one transaction and one
/// route/arena allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowPlan {
    pub(crate) index: usize,
    pub(crate) prompt_row: usize,
    pub(crate) rows: usize,
    /// Absolute position of the window's first row.
    pub(crate) position: usize,
    pub(crate) tiles: Range<usize>,
    /// Packed bytes of the window's tagged HC-stream regions.
    pub(crate) stream_bytes: u64,
    /// Session boundaries strictly inside the window (absolute, ascending).
    pub(crate) cuts: Vec<usize>,
    /// Window rows of the K1 cohort: slab short tails (route rows within the
    /// few-row limit) or, on the aggregate route, one-row decode tiles.
    pub(crate) k1_rows: usize,
    /// Window rows of aggregate F32-tail tiles (2..512 route rows); 0 on the
    /// slab route.
    pub(crate) tail_rows: usize,
}

impl WindowPlan {
    pub(crate) fn end_position(&self) -> usize {
        self.position + self.rows
    }
}

/// The resource class of one tile inside its window, read off the tile's
/// ORIGINAL routing rows (`Gpu::route_rows` under the tile's own scope). This
/// is resource accounting of the certified original routing, never a second
/// selector: the dispatch call's own [`dsp::TileKind`] stays authoritative and
/// the pre-mutation preflight refuses a tile whose sealed call disagrees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TileCohort {
    /// Window-wide grouped panels: slab K2, the aggregate F16 WMMA arm or the aggregate IU4 arm.
    K2,
    /// The K1 cohort: slab few-row tails, or the aggregate one-row decode.
    K1,
    /// An aggregate F32 grouped / indexed-down tail tile (2.. rows below the
    /// WMMA threshold), run per binding against the same lease.
    F32Tail,
}

/// The cohort of a tile of `route_rows` original routing rows on `route`.
pub(crate) fn tile_cohort(
    route: dsp::WindowRoute,
    route_rows: usize,
    few_row_limit: usize,
) -> TileCohort {
    match route {
        dsp::WindowRoute::SlabIu4 => {
            if route_rows <= few_row_limit {
                TileCohort::K1
            } else {
                TileCohort::K2
            }
        }
        dsp::WindowRoute::Aggregate | dsp::WindowRoute::AggregateSymIu4 => {
            if route_rows <= 1 {
                TileCohort::K1
            } else if route_rows < dsp::AGGREGATE_MIN_TILE_ROWS {
                TileCohort::F32Tail
            } else {
                TileCohort::K2
            }
        }
    }
}

/// The cohort the dispatch crate's own classification of a sealed call lands in.
pub(crate) fn kind_cohort(kind: dsp::TileKind) -> TileCohort {
    match kind {
        dsp::TileKind::SlabK2 | dsp::TileKind::AggregateWmma | dsp::TileKind::AggregateIu4 => {
            TileCohort::K2
        }
        dsp::TileKind::SlabK1Tail | dsp::TileKind::Decode => TileCohort::K1,
        dsp::TileKind::AggregateF32Tail => TileCohort::F32Tail,
    }
}

/// Whether a tile of `route_rows` original routing rows runs its OWN original
/// call in the panel sweep, registering its operands with the panel scratch:
/// the one-row decode call, and on the aggregate route each F32-tail call.
pub(crate) fn registers_operands(
    route: dsp::WindowRoute,
    route_rows: usize,
    few_row_limit: usize,
) -> bool {
    route_rows == 1 || tile_cohort(route, route_rows, few_row_limit) == TileCohort::F32Tail
}

/// The tiles of a window the panel sweep frames: the scratch anchor and every
/// other tile that registers its operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SweepTiles {
    /// The window tile the panel scratch is borrowed from; `None` when the
    /// window holds one-row decode tiles only (a stand-in anchors it).
    pub(crate) anchor: Option<usize>,
    /// Every other tile that registers operands, ascending.
    pub(crate) cohort: Vec<usize>,
}

impl SweepTiles {
    /// The window tiles that register operands: the anchor when it is itself
    /// a cohort tile (an F32 tail), then the rest. Each appears exactly once.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn registered(
        &self,
        tiles: &[TilePlan],
        route: dsp::WindowRoute,
        few_row_limit: usize,
    ) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .anchor
            .filter(|&ti| registers_operands(route, tiles[ti].routing.route_rows, few_row_limit))
            .into_iter()
            .collect();
        out.extend(self.cohort.iter().copied());
        out.sort_unstable();
        out
    }
}

/// Pick the anchor and the cohort tiles of a window. The anchor is a prefill
/// tile (on the aggregate route a WMMA tile when the window has one, else its
/// first multi-row tile, which may be an F32 tail); the slab route keeps its
/// first multi-row tile.
pub(crate) fn sweep_tiles(
    route: dsp::WindowRoute,
    tiles: &[TilePlan],
    few_row_limit: usize,
) -> SweepTiles {
    let preferred = if route.is_aggregate_layout() {
        tiles.iter().position(|tile| {
            tile_cohort(route, tile.routing.route_rows, few_row_limit) == TileCohort::K2
        })
    } else {
        None
    };
    let anchor = preferred.or_else(|| tiles.iter().position(|tile| tile.routing.route_rows != 1));
    let cohort = (0..tiles.len())
        .filter(|&ti| Some(ti) != anchor)
        .filter(|&ti| registers_operands(route, tiles[ti].routing.route_rows, few_row_limit))
        .collect();
    SweepTiles { anchor, cohort }
}

/// Everything the profile charges for a request, from the same layouts the
/// loader's census read: the receipt's `reserve` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptByteCensus {
    /// The single window HC-stream arena.
    pub(crate) hc_streams: u64,
    /// The dispatch crate's route storage (saved prepared input, top-k, route
    /// maps, widened accumulator, saved shared activation).
    pub(crate) route_storage: u64,
    pub(crate) a4_sidecars: u64,
    /// The window-wide stable-grouping scatter scratch (`qwen4_moe_group_layout`
    /// at the window's slots), charged beside the A4 sidecars.
    pub(crate) group_scatter: u64,
    /// Active ring pages / lookahead pages / pointer tables of the page pool.
    pub(crate) ring: u64,
    pub(crate) lookahead: u64,
    pub(crate) tables: u64,
    pub(crate) boundary_state: u64,
    /// Rows of the largest window and of the largest mixer tile.
    pub(crate) window_rows: usize,
    pub(crate) tile_rows: usize,
}

impl PromptByteCensus {
    /// Sum of every device term but the forward's own tile scratch: the
    /// physical allocation, equal to the profile's
    /// [`crate::slowlink_admission::SlowlinkCensus::owned_total`]. The admission's runtime-overhead and
    /// diagnostic budget terms are reservations, never allocations, and are
    /// not in this sum (`SlowlinkProfile::reserve_bytes` carries them).
    pub(crate) fn total(&self) -> Option<u64> {
        [
            self.hc_streams,
            self.route_storage,
            self.a4_sidecars,
            self.group_scatter,
            self.ring,
            self.lookahead,
            self.tables,
            self.boundary_state,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    }
}

/// A request S1 will run, fully decided before any mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen4PromptPlan {
    pub(crate) entry_pos: usize,
    pub(crate) rows: usize,
    pub(crate) tile_rows: usize,
    pub(crate) tiles: Vec<TilePlan>,
    pub(crate) windows: Vec<WindowPlan>,
    /// Session boundaries strictly inside the request (absolute, ascending).
    pub(crate) cuts: Vec<usize>,
    pub(crate) census: PromptByteCensus,
    /// Largest sum of K1 (few-row) tile rows any one window holds.
    pub(crate) k1_rows_max: usize,
    /// Largest sum of aggregate F32-tail tile rows any one window holds (0 on
    /// the slab route).
    pub(crate) tail_rows_max: usize,
}

impl Qwen4PromptPlan {
    /// Chronological exit row boundaries relative to the entry position: the
    /// end row of every tile (what the incumbent's chunk loop would exit at).
    pub(crate) fn launch_cuts(&self) -> Vec<usize> {
        self.tiles
            .iter()
            .map(|tile| tile.prompt_row + tile.rows)
            .collect()
    }

    pub(crate) fn window_tiles(&self, window: &WindowPlan) -> &[TilePlan] {
        &self.tiles[window.tiles.clone()]
    }

    /// Distinct `(rows, route_rows, bf16)` launch signatures in the plan, in
    /// first-use order: what the pre-mutation admission walks.
    pub(crate) fn signatures(&self) -> Vec<(usize, TileRouting)> {
        let mut out: Vec<(usize, TileRouting)> = Vec::new();
        for tile in &self.tiles {
            if !out
                .iter()
                .any(|(rows, routing)| *rows == tile.rows && *routing == tile.routing)
            {
                out.push((tile.rows, tile.routing));
            }
        }
        out
    }
}

/// Inputs of [`plan_prompt`]: all pure data. `routing` is the production
/// per-tile routing decision (a closure over the `Gpu` scope at runtime, a
/// model in CPU tests).
pub(crate) struct PromptPlanInputs<'a> {
    pub(crate) profile: &'a SlowlinkProfile,
    pub(crate) entry_pos: usize,
    pub(crate) rows: usize,
    /// Session boundaries strictly inside `(entry_pos, entry_pos + rows)`,
    /// ascending.
    pub(crate) cuts: &'a [usize],
    pub(crate) routing: &'a dyn Fn(usize) -> TileRouting,
    /// Largest routing row count of a few-row (K1, decode-exact) route.
    pub(crate) few_row_limit: usize,
}

/// Why a well-formed request stays on the incumbent route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlanBypass {
    /// A window's K1 tail rows exceed what the route storage was sized for.
    K1TailRows {
        window: usize,
        rows: usize,
        bound: usize,
    },
    /// A window's aggregate F32-tail rows exceed what the route storage was
    /// sized for.
    F32TailRows {
        window: usize,
        rows: usize,
        bound: usize,
    },
    /// A window's tagged HC-stream regions (plus the one-tile view slack) do
    /// not fit the admitted arena.
    StreamArena { window: usize, need: u64, have: u64 },
}

impl std::fmt::Display for PlanBypass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::K1TailRows { window, rows, bound } => write!(
                f,
                "window {window} holds {rows} few-row tail rows, above the {bound} the route storage admits"
            ),
            Self::F32TailRows { window, rows, bound } => write!(
                f,
                "window {window} holds {rows} F32 tail rows, above the {bound} the route storage admits"
            ),
            Self::StreamArena { window, need, have } => write!(
                f,
                "window {window} needs {need} bytes of tagged HC streams, the admitted arena holds {have}"
            ),
        }
    }
}

/// The plan of one request, or why it is the incumbent's. `Err` is a
/// malformed input (a bug), never a verdict.
///
/// The window COUNT is the greedy packing's at the admitted window
/// ([`SlowlinkProfile::window_rows`]), so it never exceeds `windows_for`; the
/// rows are then levelled across those windows. Every window pays one full
/// expert sweep, so a short tail window (17,000 rows = 22 tiles + 1) is a
/// 7.95-s wire-bound sweep for 768 rows; the same count of windows at the
/// smallest tile-multiple cap that still yields that many windows has no such
/// tail. A request whose greedy windows were closed by session cuts or the
/// boundary snapshot cap (more windows than its rows need) keeps the greedy
/// plan, whose window ends are the transaction's chronology; so does a cap
/// that needs more windows or bypasses. The admitted window is always the
/// last candidate.
pub(crate) fn plan_prompt_verdict(
    inputs: &PromptPlanInputs<'_>,
) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
    let profile = inputs.profile;
    let greedy = plan_prompt_at_window(inputs, profile.window_rows)?;
    let Ok(plan) = &greedy else {
        return Ok(greedy);
    };
    let (windows, tile) = (plan.windows.len(), profile.tile_rows);
    // Cut-closed windows are the transaction's, not a row packing to level.
    if windows < 2 || windows > profile.windows_for(plan.rows) {
        return Ok(greedy);
    }
    let mut cap = plan.rows.div_ceil(windows).div_ceil(tile) * tile;
    while cap < profile.window_rows {
        match plan_prompt_at_window(inputs, cap)? {
            Ok(levelled) if levelled.windows.len() <= windows => return Ok(Ok(levelled)),
            Ok(_) => cap += tile,
            Err(_) => break,
        }
    }
    Ok(greedy)
}

/// [`plan_prompt_verdict`] at one explicit window cap (a whole number of
/// tiles no larger than the admitted window).
fn plan_prompt_at_window(
    inputs: &PromptPlanInputs<'_>,
    window_rows: usize,
) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
    let profile = inputs.profile;
    let tile_rows = profile.tile_rows;
    let admitted = profile.route_dims();
    if inputs.rows == 0 {
        return Err(invalid("slow-link plan has no rows"));
    }
    if tile_rows == 0 || window_rows < tile_rows {
        return Err(invalid(format!(
            "slow-link plan tile {tile_rows} does not fit window {window_rows}"
        )));
    }
    let end_position = inputs
        .entry_pos
        .checked_add(inputs.rows)
        .ok_or_else(|| invalid("slow-link plan end position overflows"))?;
    let mut previous = inputs.entry_pos;
    for &cut in inputs.cuts {
        if cut <= previous || cut >= end_position {
            return Err(invalid(format!(
                "slow-link cut {cut} is not strictly inside ({previous}, {end_position}) in ascending order"
            )));
        }
        previous = cut;
    }

    // Chronology: the incumbent splits at every boundary, then tiles each
    // segment by the tile size. A tile therefore never straddles a cut.
    struct RawTile {
        position: usize,
        rows: usize,
        ends_at_cut: bool,
    }
    let mut raw: Vec<RawTile> = Vec::new();
    let mut position = inputs.entry_pos;
    for (index, &boundary) in inputs
        .cuts
        .iter()
        .chain(std::iter::once(&end_position))
        .enumerate()
    {
        let is_cut = index < inputs.cuts.len();
        while position < boundary {
            let rows = (boundary - position).min(tile_rows);
            raw.push(RawTile {
                position,
                rows,
                ends_at_cut: is_cut && position + rows == boundary,
            });
            position += rows;
        }
    }

    // Windows: contiguous tiles whose rows fit one window and whose interior
    // cuts fit the boundary snapshots. A window ends at a tile end, so every
    // tile keeps the signature the chunk-major run would give it.
    let mut windows: Vec<WindowPlan> = Vec::new();
    let mut tiles: Vec<TilePlan> = Vec::with_capacity(raw.len());
    let mut window_start_tile = 0usize;
    let mut window_rows_used = 0usize;
    let mut interior_cuts = 0usize;
    let mut prompt_row = 0usize;
    let mut window_prompt_row = 0usize;
    let mut window_position = inputs.entry_pos;
    let mut window_cuts: Vec<usize> = Vec::new();
    let mut k1_rows = 0usize;
    let mut k1_rows_max = 0usize;
    let mut tail_rows = 0usize;
    let mut tail_rows_max = 0usize;
    let close = |windows: &mut Vec<WindowPlan>,
                 tiles: &mut Vec<TilePlan>,
                 start_tile: usize,
                 rows: usize,
                 prompt_row: usize,
                 position: usize,
                 cuts: Vec<usize>,
                 k1_rows: usize,
                 tail_rows: usize| {
        let index = windows.len();
        for tile in tiles[start_tile..].iter_mut() {
            tile.window = index;
        }
        if let Some(last) = tiles[start_tile..].last_mut() {
            last.last_in_window = true;
            // The window END boundary is the transaction commit's, not a
            // strictly-inside cut.
            last.ends_at_cut = None;
        }
        windows.push(WindowPlan {
            index,
            prompt_row,
            rows,
            position,
            tiles: start_tile..tiles.len(),
            stream_bytes: 0,
            cuts,
            k1_rows,
            tail_rows,
        });
    };
    // The cohort capacities are the immutable admitted route dims: a window
    // whose actual K1 or F32-tail rows exceed them is the incumbent's.
    let cohort_bypass = |window: usize, k1: usize, tail: usize| -> Option<PlanBypass> {
        if k1 > admitted.k1_rows {
            Some(PlanBypass::K1TailRows {
                window,
                rows: k1,
                bound: admitted.k1_rows,
            })
        } else if tail > admitted.tail_rows {
            Some(PlanBypass::F32TailRows {
                window,
                rows: tail,
                bound: admitted.tail_rows,
            })
        } else {
            None
        }
    };
    for tile in &raw {
        let previous_ended_at_cut = tiles.last().is_some_and(|last| last.ends_at_cut.is_some());
        let would_rows = window_rows_used + tile.rows;
        let would_cuts = interior_cuts + usize::from(previous_ended_at_cut && window_rows_used > 0);
        if window_rows_used > 0
            && (would_rows > window_rows || would_cuts > S1_MAX_BOUNDARY_SNAPSHOTS)
        {
            let tail = tiles.last().map(|last| last.position + last.rows);
            debug_assert_eq!(tail, Some(window_position + window_rows_used));
            close(
                &mut windows,
                &mut tiles,
                window_start_tile,
                window_rows_used,
                window_prompt_row,
                window_position,
                std::mem::take(&mut window_cuts),
                k1_rows,
                tail_rows,
            );
            if let Some(bypass) = cohort_bypass(windows.len() - 1, k1_rows, tail_rows) {
                return Ok(Err(bypass));
            }
            k1_rows_max = k1_rows_max.max(k1_rows);
            tail_rows_max = tail_rows_max.max(tail_rows);
            k1_rows = 0;
            tail_rows = 0;
            window_start_tile = tiles.len();
            window_rows_used = 0;
            interior_cuts = 0;
            window_prompt_row = prompt_row;
            window_position = tile.position;
        }
        // The previous tile's cut is interior to this window only when this
        // tile joins it.
        if window_rows_used > 0 {
            if let Some(last) = tiles.last() {
                if last.ends_at_cut.is_some() {
                    window_cuts.push(last.position + last.rows);
                    interior_cuts += 1;
                }
            }
        }
        let routing = (inputs.routing)(tile.rows);
        if routing.route_rows == 0 {
            return Err(invalid("slow-link tile routes zero rows"));
        }
        let overflow = || invalid("slow-link window cohort rows overflow");
        match tile_cohort(
            profile.geometry.route,
            routing.route_rows,
            inputs.few_row_limit,
        ) {
            TileCohort::K2 => {}
            TileCohort::K1 => k1_rows = k1_rows.checked_add(tile.rows).ok_or_else(overflow)?,
            TileCohort::F32Tail => {
                tail_rows = tail_rows.checked_add(tile.rows).ok_or_else(overflow)?
            }
        }
        let ends_at_cut = tile.ends_at_cut.then_some(window_cuts.len());
        tiles.push(TilePlan {
            index: tiles.len(),
            window: 0,
            prompt_row,
            window_row: window_rows_used,
            stream_offset: 0,
            rows: tile.rows,
            position: tile.position,
            routing,
            ends_at_cut,
            last_in_window: false,
        });
        window_rows_used += tile.rows;
        prompt_row += tile.rows;
    }
    close(
        &mut windows,
        &mut tiles,
        window_start_tile,
        window_rows_used,
        window_prompt_row,
        window_position,
        std::mem::take(&mut window_cuts),
        k1_rows,
        tail_rows,
    );
    if let Some(bypass) = cohort_bypass(windows.len() - 1, k1_rows, tail_rows) {
        return Ok(Err(bypass));
    }
    k1_rows_max = k1_rows_max.max(k1_rows);
    tail_rows_max = tail_rows_max.max(tail_rows);

    // The window's HC-stream regions: each tile takes the bytes of its own tag
    // (BF16 = 2 bytes per element, F32 = 4), packed in tile order. The F32-typed
    // view every stream op validates reaches past a BF16 region by up to its own
    // size, so the arena also carries one tile of BF16 slack.
    let wide = profile.geometry.hc_count * profile.geometry.hidden;
    if wide % 8 != 0 {
        return Err(invalid(format!(
            "slow-link HC width {wide} is not a multiple of 8 elements"
        )));
    }
    let slack_bytes = (tile_rows * wide * 2) as u64;
    for window in windows.iter_mut() {
        let mut elements = 0usize;
        for tile in tiles[window.tiles.clone()].iter_mut() {
            tile.stream_offset = elements;
            let bytes_per_element = if tile.routing.bf16_streams { 2 } else { 4 };
            elements += tile.rows * wide * bytes_per_element / 4;
        }
        window.stream_bytes = elements as u64 * 4;
        let need = window.stream_bytes + slack_bytes;
        if need > profile.arena.total_bytes {
            return Ok(Err(PlanBypass::StreamArena {
                window: window.index,
                need,
                have: profile.arena.total_bytes,
            }));
        }
    }
    let census = census_for(profile, &windows)?;
    Ok(Ok(Qwen4PromptPlan {
        entry_pos: inputs.entry_pos,
        rows: inputs.rows,
        tile_rows,
        tiles,
        windows,
        cuts: inputs.cuts.to_vec(),
        census,
        k1_rows_max,
        tail_rows_max,
    }))
}

/// [`plan_prompt_verdict`] with the verdict folded to `Option`: `Ok(None)` is
/// the incumbent, chosen before any mutation or S1 allocation.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn plan_prompt(inputs: &PromptPlanInputs<'_>) -> R<Option<Qwen4PromptPlan>> {
    Ok(plan_prompt_verdict(inputs)?.ok())
}

/// The census of a plan: the profile's layouts, checked against the plan's
/// windows. The allocation reads the same layouts, so census == allocation.
fn census_for(profile: &SlowlinkProfile, windows: &[WindowPlan]) -> R<PromptByteCensus> {
    let widest = windows.iter().map(|window| window.rows).max().unwrap_or(0);
    if widest > profile.window_rows {
        return Err(invalid(format!(
            "slow-link window of {widest} rows exceeds the admitted {} rows",
            profile.window_rows
        )));
    }
    let spec = profile.pool_spec().map_err(invalid)?;
    let reserve = spec
        .reserve()
        .ok_or_else(|| invalid("slow-link page pool reserve overflows"))?;
    let page = reserve.page_bytes as u64;
    Ok(PromptByteCensus {
        hc_streams: profile.arena.total_bytes,
        route_storage: profile.census.route_storage,
        a4_sidecars: profile.census.a4_sidecars,
        group_scatter: profile.census.group_scatter,
        ring: page * spec.active_slots as u64,
        lookahead: page * spec.lookahead_pages as u64,
        tables: reserve.tables as u64,
        boundary_state: profile.census.boundary_state,
        window_rows: widest,
        tile_rows: profile.tile_rows,
    })
}

/// The byte census of a plan ([`Qwen4PromptPlan::census`]) re-derived and
/// checked against the plan.
pub(crate) fn prompt_bytes(plan: &Qwen4PromptPlan) -> R<PromptByteCensus> {
    if plan
        .windows
        .iter()
        .any(|window| window.rows > plan.census.window_rows)
    {
        return Err(invalid("slow-link plan window exceeds its census"));
    }
    Ok(plan.census)
}

/// Binding ranges of a host layer's expert groups: ascending, contiguous and
/// covering `[0, num_experts)`, one per group (a pruned expert between two
/// groups belongs to the earlier range; nothing routes to it). Index `i` is
/// the group at position `i` of the layer's schedule.
pub(crate) fn group_binding_ranges(
    groups: &[ExpertGroupPlan],
    num_experts: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(groups.len());
    let mut lo = 0usize;
    for (index, group) in groups.iter().enumerate() {
        let hi = if index + 1 == groups.len() {
            num_experts
        } else {
            group.experts.last().map_or(lo, |last| last + 1)
        };
        out.push((lo, hi));
        lo = hi;
    }
    out
}

// ── receipt ──────────────────────────────────────────────────────────

fn mode_label(mode: SlowlinkMode) -> &'static str {
    match mode {
        SlowlinkMode::Off => "env0",
        SlowlinkMode::Force => "env1",
        SlowlinkMode::Auto => "auto",
    }
}

/// Measured quantities of one executed window, summed into the receipt.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WindowReport {
    pub(crate) stats: PanelStats,
    pub(crate) overlap: OverlapSummary,
    pub(crate) useful_bytes: u64,
    pub(crate) sweep_bytes: u64,
    pub(crate) route_barrier_ms: f64,
    pub(crate) wall_ms: f64,
    pub(crate) nonexpert_ms: f64,
}

impl WindowReport {
    /// Submitted payload bytes the window's routed rows never read (the
    /// untouched experts inside acquired groups plus any group never acquired):
    /// `submitted - useful`. `None` when the useful count exceeds the
    /// submission, a measurement that cannot be true.
    pub(crate) fn unused_bytes(&self) -> Option<u64> {
        self.stats
            .submitted_payload_bytes
            .checked_sub(self.useful_bytes)
    }
}

/// Everything the layer-major receipt reports.
pub(crate) struct ReceiptInputs<'a> {
    pub(crate) mode: SlowlinkMode,
    pub(crate) plan: &'a Qwen4PromptPlan,
    pub(crate) resident_layers: usize,
    /// Ascending band layers whose routed experts are device-resident, derived
    /// from the actual per-layer mappings (never from the count).
    pub(crate) resident_layer_set: &'a [usize],
    pub(crate) host_layers: usize,
    pub(crate) tile_scratch_bytes: u64,
    pub(crate) reports: &'a [WindowReport],
    /// Device bytes in use (`total - free`) from ONE `hipMemGetInfo` sample
    /// taken after the request; not a continuous peak. `None` when the query
    /// failed. Reported under the frozen key `peak_vram_bytes`.
    pub(crate) peak_vram_bytes: Option<u64>,
    /// `"ok"` or `"aborted: ..."`.
    pub(crate) status: String,
    /// The C2 `moe_iu4` object ([`moe_iu4_receipt`]).
    pub(crate) moe_iu4: Value,
}

/// The layer-major receipt: measured fields only. A window that never ran
/// contributes zeros and the status says so; nothing is inferred.
pub(crate) fn layer_major_receipt(inputs: &ReceiptInputs<'_>) -> Value {
    let plan = inputs.plan;
    let sum = |f: &dyn Fn(&WindowReport) -> f64| inputs.reports.iter().map(f).sum::<f64>();
    let sum_u = |f: &dyn Fn(&WindowReport) -> u64| inputs.reports.iter().map(f).sum::<u64>();
    let census = &plan.census;
    let reserve_total = census
        .total()
        .and_then(|total| total.checked_add(inputs.tile_scratch_bytes))
        .unwrap_or(u64::MAX);
    let host_expert_bytes_total = inputs
        .reports
        .first()
        .map_or(0, |report| report.sweep_bytes);
    let device_copied = sum_u(&|r| r.stats.device_payload_bytes);
    let mut receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "route": "layer-major",
        "selected_by": mode_label(inputs.mode),
        "bypass_reason": Value::Null,
        "moe_iu4": inputs.moe_iu4.clone(),
        "status": inputs.status,
        "entry_pos": plan.entry_pos,
        "prompt_rows": plan.rows,
        "windows": plan.windows.iter().map(|w| json!({"start": w.prompt_row, "rows": w.rows})).collect::<Vec<_>>(),
        "launch_cuts": plan.launch_cuts(),
        "session_cuts": plan.cuts,
        "tile_rows": plan.tile_rows,
        "retained_scratch_rows": census.tile_rows,
        "sweeps": inputs.reports.len(),
        "resident_layers": inputs.resident_layers,
        "resident_layer_set": inputs.resident_layer_set,
        "host_layers": inputs.host_layers,
        "host_expert_bytes_total": host_expert_bytes_total,
        "submitted_bytes": sum_u(&|r| r.stats.submitted_payload_bytes),
        "useful_bytes": sum_u(&|r| r.useful_bytes),
        // Submitted payload no routed row read: the untouched experts inside
        // acquired groups and the groups never acquired. `null` (missing) when a
        // window's useful bytes exceed its submission, never an underflow.
        "unused_prefetched_bytes": inputs
            .reports
            .iter()
            .try_fold(0u64, |acc, r| acc.checked_add(r.unused_bytes()?)),
        "discarded_group_payload_bytes": sum_u(&|r| r.stats.discarded_payload_bytes),
        "reserve": {
            "tile": inputs.tile_scratch_bytes,
            "arena": census.hc_streams + census.route_storage + census.a4_sidecars + census.group_scatter,
            "ring": census.ring,
            "lookahead": census.lookahead,
            "tables": census.tables,
            "total": reserve_total,
        },
        "spans_ms": {
            "wall": sum(&|r| r.wall_ms),
            "copy": sum(&|r| r.overlap.copy_busy_ms),
            "compute": sum(&|r| r.overlap.compute_busy_ms),
            "route_barrier": sum(&|r| r.route_barrier_ms),
            "nonexpert": sum(&|r| r.nonexpert_ms),
        },
        "overlap": {
            "cross_layer_intersections": sum_u(&|r| r.overlap.cross_layer_intersections),
            "cross_layer_overlap_ms": sum(&|r| r.overlap.cross_layer_overlap_ms),
            "copy_idle_ms": sum(&|r| r.overlap.copy_idle_ms),
            "page_full_stalls": sum_u(&|r| r.stats.page_full_stalls),
            "copy_starved_stalls": sum_u(&|r| r.stats.copy_starved_stalls),
        },
        "timeline_overflow": sum_u(&|r| r.stats.timeline_overflow + r.stats.layer_mark_overflow),
        // Frozen key name. The value is a single post-request device-memory
        // sample (`vram_measurement` names the method); `null` if unavailable.
        "peak_vram_bytes": inputs.peak_vram_bytes,
        "vram_measurement": {
            "method": "hipMemGetInfo_post_request_sample",
            "continuous_peak": false,
        },
    });
    // Only a tier load (HIPFIRE_QWEN4_S1_TIER) copies any hot expert device-to-device: the
    // key is absent otherwise, so every other receipt is byte for byte as before. The link
    // bytes of a tier request are `submitted_bytes - device_copied_bytes`.
    if device_copied > 0 {
        receipt["device_copied_bytes"] = json!(device_copied);
    }
    receipt
}

/// The C2 `moe_iu4.tiles` counts of a request's tiles by dispatch class `[aggregate_iu4,
/// aggregate_wmma, aggregate_f32_tail, slab_k2, k1_or_decode]`: each mixer tile of the plan once
/// (not once per layer sweep), classified from its ORIGINAL routing rows ([`tile_cohort`]) on the
/// window `route`, exactly the cohort the dispatch call's own [`dsp::TileKind`] must land in
/// ([`kind_cohort`], checked before any mutation).
pub(crate) fn moe_iu4_tile_counts(
    route: dsp::WindowRoute,
    tiles: &[TilePlan],
    few_row_limit: usize,
) -> [u64; 5] {
    let mut counts = [0u64; 5];
    for tile in tiles {
        let slot = match tile_cohort(route, tile.routing.route_rows, few_row_limit) {
            TileCohort::K2 => match route {
                dsp::WindowRoute::AggregateSymIu4 => 0,
                dsp::WindowRoute::Aggregate => 1,
                dsp::WindowRoute::SlabIu4 => 3,
            },
            TileCohort::F32Tail => 2,
            TileCohort::K1 => 4,
        };
        counts[slot] += 1;
    }
    counts
}

/// What the C2 `moe_iu4` receipt object reports.
pub(crate) struct MoeIu4Facts {
    /// The window route the profile's census was charged for.
    pub(crate) route: dsp::WindowRoute,
    /// `Gpu::qwen4_moe_sym_iu4_requested` at receipt time (reporting only; the route was
    /// resolved at load).
    pub(crate) requested: bool,
    pub(crate) descriptor: Option<&'static rdna_compute::gemm::MoeSymIu4Kernel>,
    /// The load's source-header verdict ([`SlowlinkProfile::sym_iu4_source_symmetric`]).
    pub(crate) source_symmetric: Option<bool>,
    /// The final bound tables' `(verified, total)` routed MoE layers
    /// ([`Qwen4GpuForward::sym_iu4_coverage`]).
    pub(crate) final_verified: (usize, usize),
    pub(crate) a4_sidecar_bytes: u64,
    pub(crate) window_rows: usize,
    /// [`moe_iu4_tile_counts`].
    pub(crate) tiles: [u64; 5],
}

/// The descriptor object of the C2 receipt: the device descriptor's image, serving entries,
/// NT, block threads, producers and checker.
fn moe_iu4_descriptor(kernel: &rdna_compute::gemm::MoeSymIu4Kernel) -> Value {
    use rdna_compute::gemm::MoeSymIu4DownProducer;
    let down_producer = match kernel.down_producer {
        MoeSymIu4DownProducer::Hipcc { module } => module,
        MoeSymIu4DownProducer::Embedded { symbol, .. } => symbol,
    };
    json!({
        "arch": kernel.arch,
        "module": kernel.module,
        "gate_up": kernel.entries[0][0],
        "down": kernel.entries[0][1],
        "nt": kernel.tiles_per_cta[0],
        "threads": kernel.block_threads,
        "gate_producer": kernel.gate_producer.module,
        "down_producer": down_producer,
        "checker": (kernel.checker)().symbol,
        "c2": kernel.gate_producer.require_c2,
    })
}

/// The required top-level `moe_iu4` object of `hipfire.qwen4.slowlink_receipt.v1` (C2), for
/// every window route. `window_slot_stride` is the GLOBAL flat-slot extent every K2 panel
/// indexes the window's A4 sidecars with (`window_rows * 10`, down `x_row_div 1`; gate/up reads
/// the saved row `slot / 10`), reported only on the IU4 route (0 elsewhere); the panels' own Y
/// rows are local to their view.
pub(crate) fn moe_iu4_receipt(facts: &MoeIu4Facts) -> Value {
    let sym = facts.route == dsp::WindowRoute::AggregateSymIu4;
    let [aggregate_iu4, aggregate_wmma, aggregate_f32_tail, slab_k2, k1_or_decode] = facts.tiles;
    let stride: u64 = if sym {
        (facts.window_rows * dsp::K_TOP) as u64
    } else {
        0
    };
    json!({
        "window_route": facts.route.label(),
        "requested": facts.requested,
        "descriptor": facts.descriptor.map_or(Value::Null, moe_iu4_descriptor),
        "source_headers_symmetric": facts.source_symmetric,
        "final_headers_verified": [facts.final_verified.0 as u64, facts.final_verified.1 as u64],
        "a4_sidecar_bytes": facts.a4_sidecar_bytes,
        "window_slot_stride": stride,
        "tiles": {
            "aggregate_iu4": aggregate_iu4,
            "aggregate_wmma": aggregate_wmma,
            "aggregate_f32_tail": aggregate_f32_tail,
            "slab_k2": slab_k2,
            "k1_or_decode": k1_or_decode,
        },
    })
}

// ── request output ───────────────────────────────────────────────────

/// One mixer tile's final-layer hidden rows, handed to a [`PromptOutput::Capture`]
/// sink after the tile's last layer finished.
pub(crate) struct PromptTile<'a> {
    /// Absolute position of the tile's first row.
    pub(crate) start: usize,
    pub(crate) rows: usize,
    /// F32 `rows * hc_count * hidden` HC streams of the tile, in the bundle's
    /// own `spec_hidden` rows `[0, rows)`; valid during the call.
    pub(crate) wide: &'a GpuTensor,
    /// The last tile of the window being run.
    pub(crate) last: bool,
}

/// The sink of a bounded request-owned capture (the MTP target fill).
pub(crate) type CaptureSink<'a> =
    dyn FnMut(&mut Gpu, &mut Qwen4Bundle, PromptTile<'_>) -> Result<(), String> + 'a;

/// What the request returns. An all-row logits capture and any caller-owned
/// external capture stay on the incumbent route.
pub(crate) enum PromptOutput<'a> {
    /// State only.
    Silent,
    /// The final row's logits (and its greedy id when `top1` is given).
    Final {
        logits: &'a GpuTensor,
        top1: Option<&'a GpuTensor>,
    },
    /// Bounded per-tile wide-hidden capture plus the final row's logits/top-1
    /// (the speculative head seed).
    Capture {
        sink: &'a mut CaptureSink<'a>,
        head: Option<(&'a GpuTensor, &'a GpuTensor)>,
        /// The drafter's entry `pending` row (the target hidden of the last
        /// already-prefilled position), saved by the selector once the
        /// request is selected.
        pending: Option<&'a GpuTensor>,
    },
}

impl PromptOutput<'_> {
    fn capture_kind(&self) -> CaptureKind {
        match self {
            Self::Silent | Self::Final { .. } => CaptureKind::None,
            Self::Capture { .. } => CaptureKind::Bounded,
        }
    }

    fn head(&self) -> Option<(&GpuTensor, Option<&GpuTensor>)> {
        match self {
            Self::Silent => None,
            Self::Final { logits, top1 } => Some((*logits, *top1)),
            Self::Capture { head, .. } => {
                head.as_ref().map(|(logits, top1)| (*logits, Some(*top1)))
            }
        }
    }
}

// ── errors ───────────────────────────────────────────────────────────

/// The CPU-deferred flag an S1 tile hands the MoE seal: always `false`.
/// `offload_exec` (the D1 splice, an explicit D2 split) governs decode and
/// <=4-row verify only; every S1 row, 1..=4-row tails included, runs GPU
/// arithmetic.
const fn s1_tile_cpu_deferred(_layer_armed: bool) -> bool {
    false
}

fn bundle_err(what: &'static str) -> impl Fn(crate::bundle::BundleError) -> Qwen4GpuForwardError {
    move |error| invalid(format!("slow-link {what}: {error}"))
}

fn dsp_err(
    what: &'static str,
) -> impl Fn(hipfire_dispatch::types::DispatchError) -> Qwen4GpuForwardError {
    move |error| Qwen4GpuForwardError::Dispatch(format!("slow-link {what}: {error:?}"))
}

/// The route storage layout of one window: its ACTUAL rows and the K1 /
/// F32-tail rows its tiles hold (counted from the plan's original routing,
/// [`tile_cohort`]), checked against the immutable admitted route dims before
/// the layout is built. The runtime region is sized for the full admitted
/// window, so a layout that passes never exceeds it (the census bound is
/// re-checked here). A zero-row window is refused.
fn window_route_layout(
    profile: &SlowlinkProfile,
    rows: usize,
    k1_rows: usize,
    tail_rows: usize,
) -> R<dsp::PromptRouteLayout> {
    let admitted = profile.route_dims();
    if rows == 0 || rows > admitted.rows {
        return Err(invalid(format!(
            "slow-link window of {rows} rows is outside the admitted 1..={} rows",
            admitted.rows
        )));
    }
    if k1_rows > admitted.k1_rows || tail_rows > admitted.tail_rows {
        return Err(invalid(format!(
            "slow-link window holds {k1_rows} K1 and {tail_rows} F32 tail rows, the admitted \
             capacities are {} and {}",
            admitted.k1_rows, admitted.tail_rows
        )));
    }
    if k1_rows
        .checked_add(tail_rows)
        .is_none_or(|cohort| cohort > rows)
    {
        return Err(invalid(format!(
            "slow-link window of {rows} rows cannot hold {k1_rows} K1 + {tail_rows} F32 tail rows"
        )));
    }
    let dims = dsp::PromptRouteDims {
        rows,
        k1_rows,
        tail_rows,
        ..admitted
    };
    let layout = dsp::PromptRouteLayout::new(&dims).map_err(dsp_err("route layout"))?;
    if layout.total_bytes() as u64 > profile.census.route_storage {
        return Err(invalid(format!(
            "slow-link window layout is {} bytes, the admitted census holds {}",
            layout.total_bytes(),
            profile.census.route_storage
        )));
    }
    Ok(layout)
}

/// Every window's actual layout, validated against the admitted profile
/// BEFORE any mutation or allocation of the request.
fn validate_window_layouts(profile: &SlowlinkProfile, plan: &Qwen4PromptPlan) -> R<()> {
    for window in &plan.windows {
        window_route_layout(profile, window.rows, window.k1_rows, window.tail_rows)?;
    }
    Ok(())
}

// ── runtime ──────────────────────────────────────────────────────────

/// What a forward holds of S1: the admitted profile (set before attach, never
/// mutated by a request) and the device resources the profile's census
/// charged, bound once at the selected load ([`bind_at_load`]) before any
/// request.
#[derive(Default)]
pub(crate) struct SlowlinkSlot {
    pub(crate) profile: Option<SlowlinkProfile>,
    runtime: Option<SlowlinkRuntime>,
    /// Sticky: why S1 was declined at load (a mode or native-kernel capability
    /// this forward cannot run); set once by [`bind_at_load`], never by a request.
    unavailable: Option<String>,
    /// What the load decided about SP ([`bind_at_load`]); the default (knob
    /// off) on every other load.
    sp: SpLoad,
    /// A hybrid SP window runs the F16 `Aggregate` route on a symmetric-IU4 profile: every
    /// tile scope of the window suppresses the symmetric IU4 MoE arm
    /// ([`crate::slowlink_sp::f16_moe_scope`]). Set by the run at window entry, cleared at
    /// window exit and on any failure; an atomic so the shared-reference preflight can set it.
    sp_f16_moe: std::sync::atomic::AtomicBool,
}

impl SlowlinkSlot {
    /// Drain and free the ring, route region and HC-stream arena.
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> R<()> {
        match self.runtime {
            Some(runtime) => runtime.free_gpu(gpu),
            None => Ok(()),
        }
    }

    /// [`Self::sp_f16_moe`] field, read by every tile scope.
    pub(crate) fn sp_f16_moe(&self) -> bool {
        self.sp_f16_moe.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn set_sp_f16_moe(&self, on: bool) {
        self.sp_f16_moe
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// The hybrid SP is admitted on this load: every session snapshot scope carries
    /// the `sp-e1v1` identity token ([`crate::bundle::with_sp_identity`]). The
    /// touched-only mode computes S1's own bytes, so it never changes the identity.
    pub(crate) fn sp_armed(&self) -> bool {
        self.sp.admitted.is_some_and(|sp| sp.mode == SpMode::Hybrid)
    }

    /// `developer.qwen4_sp` was on at this load (the receipt carries an `sp`
    /// object); `false` on every other load.
    fn sp_knob(&self) -> bool {
        self.sp.knob
    }

    /// The SP profile a request may select with: admitted, bound, and not
    /// disabled by a failed window.
    fn sp_profile(&self) -> Option<SpProfile> {
        let admitted = self.sp.admitted?;
        let runtime = self.runtime.as_ref()?;
        match admitted.mode {
            SpMode::Hybrid => runtime.sp.as_ref()?.unavailable().is_none(),
            SpMode::TouchedOnly => runtime.touched.as_ref()?.unavailable().is_none(),
        }
        .then_some(admitted)
    }

    fn sp_summary(&self) -> SpSummary {
        SpSummary {
            knob: self.sp.knob,
            admitted: self.sp.admitted,
            decline: self.sp.decline.clone(),
            unavailable: self
                .runtime
                .as_ref()
                .and_then(SlowlinkRuntime::sp_unavailable)
                .map(str::to_owned),
        }
    }

    /// The receipt's `sp` object (`None` while the knob is off: the receipt is
    /// then unchanged).
    pub(crate) fn sp_receipt(&self, windows: &[SpWindowEntry]) -> Option<Value> {
        sp_receipt_json(&self.sp_summary(), windows)
    }
}

/// The load-time SP verdict: whether the knob was on, the admitted profile, or
/// the sticky named decline.
#[derive(Default)]
struct SpLoad {
    knob: bool,
    admitted: Option<SpProfile>,
    decline: Option<SpDecline>,
}

/// The S1 device resources of one forward, allocated once from the profile:
/// exactly what [`crate::slowlink_admission::SlowlinkCensus`] charged.
struct SlowlinkRuntime {
    ring: ExpertPanelRing,
    schedule: PanelSchedule,
    /// One region sized for the widest window; every window carves its own
    /// route storage out of it ([`dsp::PromptRouteStorage::from_region`]).
    route_region: GpuTensor,
    /// The window HC-stream arena (F32; a BF16 tile uses its first half).
    streams: GpuTensor,
    /// Trunk layer -> its routed experts live in host memory.
    host_layer: Vec<bool>,
    host_layers: usize,
    resident_layers: usize,
    tile_scratch_bytes: u64,
    /// The SP host owner ([`LoadOwner::SpHost`]); bound only when the hybrid SP
    /// was admitted at load.
    sp: Option<SpRuntime>,
    /// The touched-only SP state; present only when that mode was admitted at load.
    /// It owns no device or pinned memory.
    touched: Option<TouchedRuntime>,
}

impl SlowlinkRuntime {
    /// Why the armed SP mode was disabled, when a window failed.
    fn sp_unavailable(&self) -> Option<&str> {
        self.sp
            .as_ref()
            .and_then(SpRuntime::unavailable)
            .or_else(|| self.touched.as_ref().and_then(TouchedRuntime::unavailable))
    }

    /// Every executed SP window's report of the request, by window index.
    fn sp_reports(&self) -> Vec<(usize, SpWindowReport)> {
        match (&self.sp, &self.touched) {
            (Some(sp), _) => sp.reports().to_vec(),
            (None, Some(touched)) => touched.reports().to_vec(),
            (None, None) => Vec::new(),
        }
    }

    fn free_gpu(self, gpu: &mut Gpu) -> R<()> {
        let Self {
            ring,
            schedule,
            route_region,
            streams,
            sp,
            ..
        } = self;
        // SP's xfer stream reads the route region (the x16 rows) and writes it
        // (the hybrid rows): it drains first, and a failed drain quarantines
        // the route region and the HC arena along with the SP owners.
        let sp_drained = sp.map_or(Ok(()), |sp| sp.free_gpu(gpu));
        let tensors = if sp_drained.is_ok() {
            vec![route_region, streams]
        } else {
            std::mem::forget((route_region, streams));
            Vec::new()
        };
        let rest = free_runtime_parts(gpu, Some(ring), Some(schedule), tensors);
        sp_drained.and(rest)
    }
}

/// Release S1's device owners in the only safe order: the ring drains both
/// streams before anything it or the route region feeds is released, and a
/// failed drain QUARANTINES everything (`tensors` are deliberately not freed:
/// the device may still be reading them, so they stay allocated until process
/// exit). A successful release only returns the buffers to the device pool
/// (`Gpu::free_tensor` parks them; they go back to HIP on reuse pressure or a
/// pool drain), never straight to the driver.
fn free_runtime_parts(
    gpu: &mut Gpu,
    ring: Option<ExpertPanelRing>,
    schedule: Option<PanelSchedule>,
    tensors: Vec<GpuTensor>,
) -> R<()> {
    let mut first: Option<Qwen4GpuForwardError> = None;
    let drained = ring.map_or(Ok(()), |ring| ring.free_gpu(gpu));
    drop(schedule);
    match drained {
        Ok(()) => {
            for tensor in tensors {
                if let Err(error) = gpu.free_tensor(tensor) {
                    first.get_or_insert(error.into());
                }
            }
        }
        Err(error) => first = Some(error),
    }
    first.map_or(Ok(()), Err)
}

// ── load-time binding ────────────────────────────────────────────────

/// One physical S1 owner the selected load binds BEFORE any product request,
/// in allocation order. Every owned census term has exactly one owner here
/// (the CPU proof's `load_owner_charge` maps each term); the native-MTP entry `pending` backup is the one
/// charged byte range bound later, by `Qwen4Bundle::attach_mtp` (the head's
/// width is only known there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoadOwner {
    /// The expert page ring: payload pages (active + lookahead), their tables,
    /// timeline events and the copy stream, plus the layer schedule.
    PagePool,
    /// The one window-wide route storage region.
    RouteRegion,
    /// The window HC-stream arena.
    HcArena,
    /// The Gpu grow-only scratch grown to the full window: grouped-scatter
    /// (and on the slab route the two A4 sidecars).
    WindowScratch,
    /// The retained boundary stages (the target's session fixed parts).
    BoundaryStages,
    /// The SP host owner: pinned host buffers, the xfer stream, the four
    /// protocol events and the CPU scratch. Bound only when SP is admitted;
    /// charges no device bytes.
    SpHost,
}

/// The binding order. The big transport allocation goes first so an
/// over-committed load fails on it before small owners are built.
pub(crate) const LOAD_OWNERS: [LoadOwner; 5] = [
    LoadOwner::PagePool,
    LoadOwner::RouteRegion,
    LoadOwner::HcArena,
    LoadOwner::WindowScratch,
    LoadOwner::BoundaryStages,
];

/// The binding order of a load that admitted SP: [`LOAD_OWNERS`] plus the SP
/// host owner last, so an SP failure unwinds the S1 owners already bound.
pub(crate) const LOAD_OWNERS_SP: [LoadOwner; 6] = [
    LoadOwner::PagePool,
    LoadOwner::RouteRegion,
    LoadOwner::HcArena,
    LoadOwner::WindowScratch,
    LoadOwner::BoundaryStages,
    LoadOwner::SpHost,
];

/// Census bytes `owner` holds ([`crate::slowlink_admission::SlowlinkCensus::owned_terms`]).
/// `BoundaryStages` carries the whole `boundary_state` term (its native-MTP
/// `pending` share is bound at MTP attach).
#[cfg(test)]
pub(crate) fn load_owner_charge(profile: &SlowlinkProfile, owner: LoadOwner) -> u64 {
    let census = &profile.census;
    match owner {
        LoadOwner::PagePool => census.ring + census.lookahead,
        LoadOwner::RouteRegion => census.route_storage,
        LoadOwner::HcArena => census.arena,
        LoadOwner::WindowScratch => census.a4_sidecars + census.group_scatter,
        LoadOwner::BoundaryStages => census.boundary_state,
        // Pinned host bytes, charged by the SP admission's own budget.
        LoadOwner::SpHost => 0,
    }
}

/// What a load binds its owners through: the production binder allocates on
/// the device, the CPU model of the lifecycle tests records the same steps.
pub(crate) trait LoadBinder {
    /// Allocate one owner.
    fn bind(&mut self, owner: LoadOwner) -> Result<(), String>;
    /// Release what `bound` (in binding order, the failed owner excluded) and
    /// the failed owner's partial state hold, last first. Returns the
    /// DISPOSITION of that cleanup (what was released, what was quarantined),
    /// never an assumption that parking returned memory to the driver.
    fn unwind(&mut self, bound: &[LoadOwner]) -> String;
}

/// Bind every owner or none: the first failure unwinds the owners already
/// bound and names the owner, the reason and the cleanup disposition.
pub(crate) fn bind_load_owners(binder: &mut impl LoadBinder) -> Result<(), String> {
    bind_owner_list(binder, &LOAD_OWNERS)
}

/// [`bind_load_owners`] for a load that admitted SP: the SP host owner binds
/// last ([`LOAD_OWNERS_SP`]).
pub(crate) fn bind_load_owners_with_sp(binder: &mut impl LoadBinder) -> Result<(), String> {
    bind_owner_list(binder, &LOAD_OWNERS_SP)
}

fn bind_owner_list(binder: &mut impl LoadBinder, owners: &[LoadOwner]) -> Result<(), String> {
    let mut bound: Vec<LoadOwner> = Vec::with_capacity(owners.len());
    for &owner in owners {
        if let Err(reason) = binder.bind(owner) {
            let cleanup = binder.unwind(&bound);
            return Err(format!(
                "slow-link load could not bind the {owner:?} owner: {reason}; cleanup: {cleanup}"
            ));
        }
        bound.push(owner);
    }
    Ok(())
}

/// The arguments of the full-window Gpu scratch growth: what the census'
/// `group_scatter` and `a4_sidecars` terms are computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowScratchArgs {
    pub(crate) route: dsp::WindowRoute,
    pub(crate) rows: usize,
    pub(crate) experts: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
}

pub(crate) fn window_scratch_args(profile: &SlowlinkProfile) -> WindowScratchArgs {
    let geometry = &profile.geometry;
    WindowScratchArgs {
        route: geometry.route,
        rows: profile.window_rows,
        experts: geometry.num_experts,
        hidden: geometry.hidden,
        intermediate: geometry.moe_intermediate,
    }
}

/// Grow the Gpu scratch the window's one count barrier would otherwise grow on
/// the first long request, through the same reserve entries the forward's
/// tile-sized reservation uses: the slab route and the symmetric aggregate route
/// reserve the grouping scratch, the down sidecar and the gate/up sidecar at the
/// FULL window (`reserve_qwen4_moe_sym`, load-bound: no lazy first-request growth);
/// the F16 aggregate route (no A4 sidecars) only the grouping scratch.
fn reserve_window_scratch(gpu: &mut Gpu, args: WindowScratchArgs) -> Result<(), String> {
    match args.route {
        dsp::WindowRoute::SlabIu4 | dsp::WindowRoute::AggregateSymIu4 => gpu.reserve_qwen4_moe_sym(
            args.rows,
            dsp::K_TOP,
            args.experts,
            args.hidden,
            args.intermediate,
        ),
        dsp::WindowRoute::Aggregate => {
            gpu.reserve_qwen4_moe_group_scratch(args.rows, dsp::K_TOP, args.experts)
        }
    }
    .and_then(|()| {
        // The count barrier's host-mapped landing zone: the per-expert counts (`experts`
        // words) and the window's inverse route map (`rows * K_TOP` words) both go through it.
        gpu.reserve_slowlink_count_host(args.experts.max(args.rows * dsp::K_TOP))
    })
    .map_err(|error| {
        format!(
            "window scratch growth to {} rows failed: {error}",
            args.rows
        )
    })
}

/// The validated, still-unallocated shape of one forward's runtime: everything
/// the binder needs before its first device allocation.
struct RuntimePlan {
    spec: Option<crate::gpu_forward::panel_stage::PanelPoolSpec>,
    schedule: Option<PanelSchedule>,
    host_layer: Vec<bool>,
    host_layers: usize,
    resident_layers: usize,
    route_bytes: usize,
    stream_elements: usize,
}

/// Validate the admitted profile against this forward's layers and layouts.
/// Allocates nothing; an `Err` is a pre-materialization decline.
fn plan_runtime(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    profile: &SlowlinkProfile,
) -> Result<RuntimePlan, String> {
    let config = &bundle.config;
    let mut host_layer = vec![false; config.num_hidden_layers];
    let mut sources = Vec::new();
    let mut resident_layers = 0usize;
    for layer in forward.band_layers() {
        let Some(runtime) = forward.moe.get(layer).and_then(Option::as_ref) else {
            continue;
        };
        if runtime.s1_num_experts() != profile.geometry.num_experts {
            return Err(format!(
                "layer {layer} holds {} experts, the profile admitted {}",
                runtime.s1_num_experts(),
                profile.geometry.num_experts
            ));
        }
        if runtime.s1_host_mapped() {
            host_layer[layer] = true;
            sources.push(
                runtime
                    .s1_host_sources(&mut *gpu, bundle, layer)
                    .map_err(|error| error.to_string())?,
            );
        } else {
            resident_layers += 1;
        }
    }
    let host_layers = sources.len();
    if host_layers == 0 {
        return Err("no host-resident expert layer to stream".to_string());
    }
    let spec = profile.pool_spec()?;
    if config.num_hidden_layers > spec.timeline_layers {
        return Err(format!(
            "{} layers exceed the {} timeline layer marks the pool admitted",
            config.num_hidden_layers, spec.timeline_layers
        ));
    }
    if host_layers * profile.pages.pages_per_layer > spec.timeline_records {
        return Err(format!(
            "{host_layers} host layers exceed the {} timeline records the pool admitted",
            spec.timeline_records
        ));
    }
    let layout =
        dsp::PromptRouteLayout::new(&profile.route_dims()).map_err(|error| format!("{error:?}"))?;
    if layout.total_bytes() as u64 != profile.census.route_storage {
        return Err(format!(
            "route storage layout is {} bytes, the census charged {}",
            layout.total_bytes(),
            profile.census.route_storage
        ));
    }
    if layout.a4_sidecar_bytes() as u64 != profile.census.a4_sidecars
        || layout.group_scatter_bytes() as u64 != profile.census.group_scatter
    {
        return Err(format!(
            "{} route A4 sidecars/group scatter layout is {}/{} bytes, the census charged {}/{}",
            profile.geometry.route.label(),
            layout.a4_sidecar_bytes(),
            layout.group_scatter_bytes(),
            profile.census.a4_sidecars,
            profile.census.group_scatter
        ));
    }
    // Phase 2 of the symmetric IU4 route proof, repeated at the S1 attach: every routed trunk
    // layer's final bound tables verified (the forward's load already refused a shortfall).
    let (verified, total) = forward.sym_iu4_coverage();
    crate::slowlink_admission::check_sym_iu4_final_headers(
        profile.geometry.route,
        verified,
        total,
    )?;
    let stream_bytes = profile.arena.span(ArenaField::HcStreams).bytes;
    if profile.arena.total_bytes % 4 != 0 || stream_bytes > profile.arena.total_bytes {
        return Err("HC-stream arena bytes are not whole F32 elements".to_string());
    }
    if config.num_experts_per_tok != dsp::K_TOP {
        return Err(format!(
            "the model routes {} experts per token, the admitted window layout holds {}",
            config.num_experts_per_tok,
            dsp::K_TOP
        ));
    }
    let schedule =
        PanelSchedule::new(sources, EXPERTS_PER_PAGE).map_err(|error| error.to_string())?;
    Ok(RuntimePlan {
        spec: Some(spec),
        schedule: Some(schedule),
        host_layer,
        host_layers,
        resident_layers,
        route_bytes: layout.total_bytes(),
        stream_elements: (profile.arena.total_bytes / 4) as usize,
    })
}

/// The production binder: allocates every owner through the factories the
/// request path used to call lazily (the page ring, `alloc_tensor` of the
/// route region and arena, the Gpu scratch reserves, the bundle's boundary
/// stage allocator) and holds them in `SlowlinkRuntime` parts.
struct HipLoadBinder<'a> {
    forward: &'a Qwen4GpuForward,
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    profile: &'a SlowlinkProfile,
    plan: RuntimePlan,
    ring: Option<ExpertPanelRing>,
    route_region: Option<GpuTensor>,
    streams: Option<GpuTensor>,
    sp_plan: Option<SpBindPlan>,
    sp: Option<SpRuntime>,
}

impl LoadBinder for HipLoadBinder<'_> {
    fn bind(&mut self, owner: LoadOwner) -> Result<(), String> {
        match owner {
            LoadOwner::PagePool => {
                let spec = self
                    .plan
                    .spec
                    .take()
                    .ok_or_else(|| "the page pool spec was already consumed".to_string())?;
                self.ring = Some(
                    ExpertPanelRing::new(&mut *self.gpu, spec)
                        .map_err(|error| format!("expert page ring allocation failed: {error}"))?,
                );
            }
            LoadOwner::RouteRegion => {
                self.route_region = Some(
                    self.gpu
                        .alloc_tensor(&[self.plan.route_bytes], DType::Raw)
                        .map_err(|error| format!("route region allocation failed: {error}"))?,
                );
            }
            LoadOwner::HcArena => {
                self.streams = Some(
                    self.gpu
                        .alloc_tensor(&[self.plan.stream_elements], DType::F32)
                        .map_err(|error| format!("HC-stream arena allocation failed: {error}"))?,
                );
            }
            LoadOwner::WindowScratch => {
                reserve_window_scratch(&mut *self.gpu, window_scratch_args(self.profile))?;
            }
            LoadOwner::BoundaryStages => {
                self.bundle
                    .slowlink_bind_boundary_stages(&mut *self.gpu)
                    .map_err(|error| format!("boundary stage allocation failed: {error}"))?;
            }
            LoadOwner::SpHost => {
                let plan = self
                    .sp_plan
                    .ok_or_else(|| "the SP bind plan is missing".to_string())?;
                self.sp = Some(SpRuntime::bind(&mut *self.gpu, &plan)?);
            }
        }
        Ok(())
    }

    fn unwind(&mut self, bound: &[LoadOwner]) -> String {
        let mut notes: Vec<String> = Vec::new();
        let stages = self.bundle.slowlink_take_boundary_stage_tensors();
        if !stages.is_empty() {
            let count = stages.len();
            let mut failed = None;
            for tensor in stages {
                if let Err(error) = self.gpu.free_tensor(tensor) {
                    failed.get_or_insert(error);
                }
            }
            notes.push(match failed {
                None => {
                    format!("{count} boundary stage buffers returned to the device pool")
                }
                Some(error) => format!("boundary stage release failed: {error}"),
            });
        }
        if bound.contains(&LoadOwner::WindowScratch) {
            notes.push(
                "the window scratch is Gpu-owned grow-only scratch and stays until Gpu teardown"
                    .to_string(),
            );
        }
        let mut sp_quarantined = false;
        if let Some(sp) = self.sp.take() {
            notes.push(match sp.free_gpu(&mut *self.gpu) {
                Ok(()) => "the SP owners (pinned host buffers, xfer stream, events) were \
                           drained and released"
                    .to_string(),
                Err(error) => {
                    sp_quarantined = true;
                    format!("SP cleanup reported {error}; the SP owners are QUARANTINED")
                }
            });
        }
        let mut tensors: Vec<GpuTensor> = [self.streams.take(), self.route_region.take()]
            .into_iter()
            .flatten()
            .collect();
        if sp_quarantined {
            // The xfer stream may still read or write the route region: it is
            // quarantined with the SP owners, never returned to the pool.
            std::mem::forget(std::mem::take(&mut tensors));
            notes.push(
                "the route region and HC arena are QUARANTINED with them (leaked until process \
                 exit)"
                    .to_string(),
            );
        }
        let held = self.ring.is_some() || !tensors.is_empty();
        let result = free_runtime_parts(
            &mut *self.gpu,
            self.ring.take(),
            self.plan.schedule.take(),
            tensors,
        );
        if held {
            notes.push(match result {
                Ok(()) => "the expert ring, route region and HC arena were drained and returned \
                           to the device pool (parked until reuse or a pool drain, not yet \
                           returned to HIP)"
                    .to_string(),
                Err(error) => format!(
                    "ring/route/HC cleanup reported {error}; a failed ring drain QUARANTINES the \
                     ring, route region and HC arena (not freed while the device may still use \
                     them; leaked until process exit)"
                ),
            });
        }
        if notes.is_empty() {
            "nothing had been allocated".to_string()
        } else {
            notes.join("; ")
        }
    }
}

impl HipLoadBinder<'_> {
    fn into_runtime(self) -> Result<SlowlinkRuntime, String> {
        let Self {
            forward,
            plan,
            ring,
            route_region,
            streams,
            sp,
            sp_plan,
            ..
        } = self;
        if sp_plan.is_some() != sp.is_some() {
            return Err("the SP owner binding disagrees with its plan".to_string());
        }
        match (ring, plan.schedule, route_region, streams) {
            (Some(ring), Some(schedule), Some(route_region), Some(streams)) => {
                Ok(SlowlinkRuntime {
                    ring,
                    schedule,
                    route_region,
                    streams,
                    host_layer: plan.host_layer,
                    host_layers: plan.host_layers,
                    resident_layers: plan.resident_layers,
                    tile_scratch_bytes: forward.slowlink_tile_scratch_bytes(),
                    sp,
                    touched: None,
                })
            }
            _ => Err("a bound slow-link owner is missing".to_string()),
        }
    }
}

/// `few_row_limit` of a `tile_rows`-row tile under the tile's own scope.
fn few_row_limit_of(forward: &Qwen4GpuForward, gpu: &mut Gpu, tile_rows: usize) -> usize {
    let (scopes, _) = forward.enter_tile_scopes(gpu, tile_rows);
    let limit = gpu.few_row_limit();
    Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
    limit
}

/// Why this load must NOT materialize S1 (a mode the forward or bundle does
/// not own, or a native kernel/route the dispatch capability dry run refuses
/// for the admitted tile), asked BEFORE any device allocation.
fn load_decline(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    profile: &SlowlinkProfile,
) -> Option<String> {
    if let Err(why) = bundle.slowlink_prompt_eligible() {
        return Some(why.to_string());
    }
    if let Some(why) = forward.slowlink_forward_blocker(&bundle.config) {
        return Some(why.to_string());
    }
    for layer in forward.band_layers() {
        if let Some(runtime) = forward.moe.get(layer).and_then(Option::as_ref) {
            if runtime.s1_host_mapped() {
                if let Err(why) = runtime.s1_layer_check() {
                    return Some(format!("host layer {layer}: {why}"));
                }
            }
        }
    }
    let few_row_limit = few_row_limit_of(forward, gpu, profile.tile_rows);
    let geo = match RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts) {
        Ok(geo) => geo,
        Err(error) => return Some(error.to_string()),
    };
    let routing = tile_routing(forward, gpu, profile.tile_rows);
    preflight_signatures(
        forward,
        bundle,
        gpu,
        &geo,
        vec![(profile.tile_rows, routing)],
        bundle.state.position,
        profile,
    )
    .err()
}

/// The one line of the SP load decision (a SP knob is on): the armed mode and its row cap, or
/// the named sticky decline. SP is otherwise silent, so a declined knob used to look like an SP
/// that ran no layers. Pure: [`sp_log_decision`] prints it.
pub(crate) fn sp_decision_line(decision: Result<&SpProfile, &SpDecline>) -> String {
    fn route_name(route: dsp::WindowRoute) -> &'static str {
        match route {
            dsp::WindowRoute::Aggregate => "Aggregate F16",
            dsp::WindowRoute::AggregateSymIu4 => "SymIu4",
            dsp::WindowRoute::SlabIu4 => "SlabIu4",
        }
    }
    match decision {
        Ok(sp) if sp.window_route != sp.base_route => format!(
            "qwen4 SP: {} mode armed (SP windows use {}; others {})",
            sp.mode.label(),
            route_name(sp.window_route),
            route_name(sp.base_route),
        ),
        Ok(sp) => format!(
            "qwen4 SP: {} mode armed (single windows of {}..={} rows, no session cut)",
            sp.mode.label(),
            crate::slowlink_sp::SP_MIN_ROWS,
            sp.sp_max_rows,
        ),
        Err(decline) => format!("qwen4 SP: knob on but not armed: {decline}"),
    }
}

/// Print [`sp_decision_line`] to stderr like the other load lines (`qwen4 S1 plan:`, the S1
/// load line), NOT through `tracing`: the daemon installs a subscriber but the examples
/// (`qwen4_kld`, `qwen4_greedy_ids`) and tools do not, and the SP decision must be visible on
/// every path that runs `bind_at_load`.
pub(crate) fn sp_log_decision(decision: Result<&SpProfile, &SpDecline>) {
    eprintln!("  {}", sp_decision_line(decision));
}

/// The load-time dry run of the F16 `Aggregate` route on a symmetric-IU4 profile: the admitted
/// tile's sealed call must be admitted as an F16 WMMA tile of an `Aggregate` window under the F16
/// MoE scope (the scope the hybrid SP's windows run under). A profile that is already F16
/// has nothing to prove.
fn sp_f16_route_probe(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    profile: &SlowlinkProfile,
) -> Result<(), String> {
    if profile.geometry.route != dsp::WindowRoute::AggregateSymIu4 {
        return Ok(());
    }
    let few_row_limit = few_row_limit_of(forward, gpu, profile.tile_rows);
    let geo = RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts)
        .map_err(|error| error.to_string())?;
    let routing = tile_routing(forward, gpu, profile.tile_rows);
    preflight_signatures_on(
        forward,
        bundle,
        gpu,
        &geo,
        vec![(profile.tile_rows, routing)],
        bundle.state.position,
        profile,
        PreflightRoute {
            route: dsp::WindowRoute::Aggregate,
            f16_moe: true,
        },
    )
}

/// The SP decision line of a load whose S1 runtime itself was declined (SP rides on S1).
pub(crate) fn sp_s1_unavailable_line(reason: &str) -> String {
    format!(
        "qwen4 SP: knob on but not armed: the slow-link prefill (S1) is unavailable on this load: {reason}"
    )
}

/// The load line of a hybrid decline that touched-only replaced (default on gfx1100): the
/// reason hybrid did not arm and the mode that runs instead.
pub(crate) fn sp_hybrid_fallback_line(hybrid: &SpDecline, now: SpMode) -> String {
    format!(
        "qwen4 SP: hybrid mode declined, falling back to {}: {hybrid}",
        now.label()
    )
}

fn sp_log_s1_unavailable(reason: &str) {
    eprintln!("  {}", sp_s1_unavailable_line(reason));
}

/// Materialize the whole admitted S1 runtime ONCE, at the selected load, after
/// the profile is attached and the forward built, before ANY product request
/// (so no later first-use allocation can take the memory the placement
/// charged). `Ok(())` with `unavailable` set is a named pre-materialization
/// decline (a mode or native-kernel capability this load cannot run; nothing
/// was allocated); `Err` is a named load failure whose partial owners were
/// unwound with their disposition reported. No profile: nothing happens.
pub(crate) fn bind_at_load(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let hybrid_knob = crate::slowlink_sp::hybrid_knob(&gpu.arch);
    let sp_knob = hybrid_knob || crate::slowlink_sp::touched_knob(&gpu.arch);
    let Some(profile) = forward.slowlink_profile().copied() else {
        if sp_knob {
            sp_log_decision(Err(&SpDecline::NoS1Profile));
        }
        return Ok(());
    };
    if forward.slowlink.runtime.is_some() || forward.slowlink.unavailable.is_some() {
        return Ok(());
    }
    if sp_knob {
        forward.slowlink.sp.knob = true;
        forward.slowlink.sp.decline = Some(SpDecline::NoS1Profile);
    }
    if let Some(reason) = load_decline(forward, &*bundle, gpu, &profile) {
        if sp_knob {
            sp_log_s1_unavailable(&reason);
        }
        forward.slowlink.unavailable = Some(reason);
        return Ok(());
    }
    let plan = match plan_runtime(forward, &*bundle, gpu, &profile) {
        Ok(plan) => plan,
        Err(reason) => {
            if sp_knob {
                sp_log_s1_unavailable(&reason);
            }
            forward.slowlink.unavailable = Some(reason);
            return Ok(());
        }
    };
    let sp_bind = if sp_knob {
        let pages = plan.spec.as_ref().map_or(0, |spec| spec.pages());
        // The F16 `Aggregate` route dry run: only a hybrid SP on a symmetric-IU4 profile needs
        // it (its windows run the F16 route); every other load is ready by construction (a
        // slab profile's hybrid windows keep the slab route and never run the F16 arm).
        let f16_route_ready = if hybrid_knob && profile.geometry.route != dsp::WindowRoute::SlabIu4
        {
            match sp_f16_route_probe(&*forward, &*bundle, gpu, &profile) {
                Ok(()) => true,
                Err(why) => {
                    eprintln!(
                        "  qwen4 SP: the F16 Aggregate route does not admit the admitted tile on this \
                         load: {why}"
                    );
                    false
                }
            }
        } else {
            true
        };
        match sp_load_decision(&*forward, &*gpu, &profile, pages, f16_route_ready)? {
            Ok(bind) => {
                forward.slowlink.sp.decline = None;
                sp_log_decision(Ok(&bind.profile()));
                Some(bind)
            }
            Err(decline) => {
                sp_log_decision(Err(&decline));
                forward.slowlink.sp.decline = Some(decline);
                None
            }
        }
    } else {
        None
    };
    let sp_plan = match sp_bind {
        Some(SpBind::Hybrid(plan)) => Some(plan),
        _ => None,
    };
    let runtime = {
        let mut binder = HipLoadBinder {
            forward: &*forward,
            bundle: &mut *bundle,
            gpu: &mut *gpu,
            profile: &profile,
            plan,
            ring: None,
            route_region: None,
            streams: None,
            sp_plan,
            sp: None,
        };
        if sp_plan.is_some() {
            bind_load_owners_with_sp(&mut binder)?;
        } else {
            bind_load_owners(&mut binder)?;
        }
        binder.into_runtime()?
    };
    let mut runtime = runtime;
    if let Some(SpBind::TouchedOnly(plan)) = sp_bind {
        runtime.touched = Some(TouchedRuntime::new(&plan));
    }
    forward.slowlink.runtime = Some(runtime);
    forward.slowlink.sp.admitted = sp_bind.as_ref().map(SpBind::profile);
    Ok(())
}

// ── geometry of one request ──────────────────────────────────────────

/// Per-request constants every tile of the request reads.
struct RunGeo {
    config: Qwen4Config,
    dims: Qwen4ProgramDims,
    wide: usize,
    hidden: usize,
    few_row_limit: usize,
    ple_layer: Option<usize>,
    ple_heads: usize,
    ple_width: usize,
    band: Range<usize>,
    last_layer: usize,
    num_experts: usize,
}

impl RunGeo {
    fn new(
        forward: &Qwen4GpuForward,
        bundle: &Qwen4Bundle,
        few_row_limit: usize,
        num_experts: usize,
    ) -> R<Self> {
        let config = bundle.config.clone();
        let dims = program_dims(&config);
        let band = forward.band_layers();
        let ple_layer = if config.ple.is_some() {
            let index = config
                .ple_layer_ids
                .first()
                .and_then(|id| id.checked_sub(1))
                .ok_or_else(|| invalid("PLE layer id missing"))?;
            if index >= config.num_hidden_layers {
                return Err(invalid("PLE layer id is outside the trunk"));
            }
            Some(index)
        } else {
            None
        };
        let (ple_heads, ple_width) = config
            .ple
            .as_ref()
            .map_or((0, 0), |ple| (ple.head_count, ple.row_width));
        if band.is_empty() {
            return Err(invalid("slow-link request has no trunk layers"));
        }
        Ok(Self {
            wide: dims.wide(),
            hidden: config.hidden_size,
            few_row_limit,
            ple_layer,
            ple_heads,
            ple_width,
            last_layer: band.end - 1,
            band,
            dims,
            config,
            num_experts,
        })
    }
}

/// The routing decision of a `rows`-row tile under the tile's own scope: what
/// the incumbent decides for the chunk of the same rows.
fn tile_routing(forward: &Qwen4GpuForward, gpu: &mut Gpu, rows: usize) -> TileRouting {
    let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
    let routing = TileRouting {
        route_rows: gpu.route_rows(rows),
        bf16_streams: gpu.qwen4_bf16_streams(rows),
    };
    Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
    routing
}

// ── PLE ordered history ──────────────────────────────────────────────

/// The private PLE cursor of a request: the token history each tile of the
/// window starts from (entry history advanced by the earlier tiles' tokens,
/// never the bundle's published history, which only the window commit moves)
/// and the retained lookahead ticket of the next tile's rows.
struct PleCursor {
    retained: Option<RowTicket>,
    histories: Vec<PleHistory>,
}

impl PleCursor {
    /// The histories of `tiles` over `window_tokens`, from `entry`.
    fn histories_for(
        entry: PleHistory,
        tiles: &[TilePlan],
        window_tokens: &[u32],
    ) -> Vec<PleHistory> {
        let mut history = entry;
        let mut out = Vec::with_capacity(tiles.len());
        for tile in tiles {
            out.push(history);
            for &token in &window_tokens[tile.window_row..tile.window_row + tile.rows] {
                history.push(token);
            }
        }
        out
    }

    /// Fetch, stage, upload and widen the PLE rows of tile `ti`, then start
    /// the next tile's fetch so its worker overlaps this tile's compute.
    #[allow(clippy::too_many_arguments)]
    fn stage_tile(
        &mut self,
        owner: &crate::bundle::Qwen4BundlePle,
        host: &mut crate::ple_stage::PleHostStage,
        gpu: &mut Gpu,
        w: &WinCtx<'_>,
        ti: usize,
        staged: &GpuTensor,
        ple_rows: &GpuTensor,
        rows_f16: bool,
        heads: usize,
        width: usize,
    ) -> R<()> {
        let tile = &w.tiles[ti];
        let ids = self.histories[ti].row_ids(&owner.metadata, w.tile_tokens(ti));
        let mut fetch = RowFetch::begin_or_adopt(&owner.rows, self.retained.take(), ids)
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        let mode = if ple_async_upload(host.enabled(), false, true, gpu.graphs.capture_mode) {
            PleUploadMode::Async
        } else {
            PleUploadMode::Blocking
        };
        stage_ple_rows(
            gpu, host, &mut fetch, staged, ple_rows, rows_f16, tile.rows, heads, width, mode,
        )?;
        fetch.complete();
        if ti + 1 < w.tiles.len() {
            let next = w.tile_tokens(ti + 1);
            let fit = owner.rows.max_rows_per_prefetch() / heads.max(1);
            let ahead = &next[..next.len().min(fit)];
            self.retained = owner
                .rows
                .prefetch(
                    owner.rows.current_epoch(),
                    self.histories[ti + 1].row_ids(&owner.metadata, ahead),
                )
                .ok();
        }
        Ok(())
    }
}

// ── the group walk (layer-ahead transport driver) ────────────────────

/// What the layer walk needs of the transport: the production ring (HIP) and
/// the CPU model of the lifecycle tests implement it, so one walker is both.
pub(crate) trait GroupRing {
    type Ctx: ?Sized;
    /// Ticket of current-layer `group` (enqueued if still missing).
    fn require(&mut self, ctx: &Self::Ctx, layer: usize, group: u32) -> R<PanelTicket>;
    /// Acquire `ticket`, run `compute` while the lease is held, release the
    /// lease. The page is freed only by [`Self::retire`].
    fn with_lease(
        &mut self,
        ctx: &mut Self::Ctx,
        ticket: PanelTicket,
        compute: &mut dyn FnMut(&mut Self::Ctx, Option<&PanelBindingParts<'_>>) -> R<()>,
    ) -> R<()>;
    fn retire(&mut self, ctx: &Self::Ctx, ticket: PanelTicket) -> R<()>;
    /// Fill freed pages: the current layer's missing groups first, then the
    /// next layer's.
    fn pump(&mut self, ctx: &Self::Ctx) -> R<usize>;
}

/// The production ring and the window's schedule: S1's page-order sweep
/// ([`PanelSchedule`], the default) or SP's selective per-layer schedule.
struct HipRing<'a, S: RingSchedule = PanelSchedule> {
    ring: &'a mut ExpertPanelRing,
    schedule: &'a mut S,
}

impl<S: RingSchedule> GroupRing for HipRing<'_, S> {
    type Ctx = Gpu;

    fn require(&mut self, gpu: &Gpu, layer: usize, group: u32) -> R<PanelTicket> {
        self.ring.require(gpu, self.schedule, layer, group)
    }

    fn with_lease(
        &mut self,
        gpu: &mut Gpu,
        ticket: PanelTicket,
        compute: &mut dyn FnMut(&mut Gpu, Option<&PanelBindingParts<'_>>) -> R<()>,
    ) -> R<()> {
        let lease = self.ring.acquire(&*gpu, ticket)?;
        let parts = lease.parts();
        let result = compute(gpu, Some(&parts));
        result
    }

    fn retire(&mut self, gpu: &Gpu, ticket: PanelTicket) -> R<()> {
        self.ring.retire(gpu, ticket)
    }

    fn pump(&mut self, gpu: &Gpu) -> R<usize> {
        self.ring.pump(gpu, self.schedule)
    }
}

/// The Gpu scope of the tile whose rows a dispatch step handles: the dispatch
/// crate brackets every tile-specific launch (a K1 tail cohort, a tile's shared
/// expert) with `enter(row0)`/`leave(row0)` so a tail's routing rows, canonical
/// scope and HC tag never leak into the K2 panels' (tile 0's) frozen arm, and
/// the K2 arm never leaks into a tail.
struct ForwardScopeHook<'a> {
    forward: &'a Qwen4GpuForward,
    tiles: &'a [TilePlan],
    open: Vec<crate::gpu_forward::TileScopes>,
}

impl dsp::TileScopeHook for ForwardScopeHook<'_> {
    fn enter(
        &mut self,
        gpu: &mut Gpu,
        row0: usize,
    ) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let tile = self
            .tiles
            .iter()
            .find(|tile| tile.window_row == row0)
            .ok_or_else(|| {
                hipfire_dispatch::types::DispatchError::Hip(format!(
                    "slow-link scope hook: no tile starts at window row {row0}"
                ))
            })?;
        let (scopes, _) = self.forward.enter_tile_scopes(gpu, tile.rows);
        self.open.push(scopes);
        Ok(())
    }

    fn leave(
        &mut self,
        gpu: &mut Gpu,
        _row0: usize,
    ) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let scopes = self.open.pop().ok_or_else(|| {
            hipfire_dispatch::types::DispatchError::Hip(
                "slow-link scope hook: leave without a matching enter".to_string(),
            )
        })?;
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        Ok(())
    }
}

/// A group of the layer's schedule: whether any panel reads it. A group no
/// panel reads (no K2 row and no K1 expert) is retired unread, never refetched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupWork {
    pub(crate) group: u32,
    pub(crate) bound: bool,
}

/// What a layer walk did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WalkSummary {
    pub(crate) acquired: usize,
    pub(crate) discarded: usize,
}

/// Walk one host layer's groups in ascending order: `require` each, run
/// `compute(index, binding)` for those panels read, `retire` every one (the
/// last reader of a group is the last panel of that group, all cohorts
/// included), and `pump` after each retire so freed pages immediately refill,
/// the current layer first and then the next layer's weights.
pub(crate) fn walk_layer_groups<G: GroupRing>(
    ring: &mut G,
    ctx: &mut G::Ctx,
    layer: usize,
    work: &[GroupWork],
    compute: &mut dyn FnMut(&mut G::Ctx, usize, Option<&PanelBindingParts<'_>>) -> R<()>,
) -> R<WalkSummary> {
    let mut summary = WalkSummary::default();
    for (index, group) in work.iter().enumerate() {
        let ticket = ring.require(&*ctx, layer, group.group)?;
        if group.bound {
            ring.with_lease(ctx, ticket, &mut |ctx, parts| compute(ctx, index, parts))?;
            summary.acquired += 1;
        } else {
            summary.discarded += 1;
        }
        ring.retire(&*ctx, ticket)?;
        ring.pump(&*ctx)?;
    }
    Ok(summary)
}

/// The panels of each group: ranges into the schedule's panel list. The panels
/// of one binding must be contiguous and bindings ascending (the order the
/// ordered combine requires).
pub(crate) fn panels_by_group(
    bindings: impl Iterator<Item = usize>,
    groups: usize,
) -> R<Vec<Range<usize>>> {
    let mut out = vec![0..0; groups];
    let mut previous: Option<usize> = None;
    for (index, binding) in bindings.enumerate() {
        if binding >= groups {
            return Err(invalid(format!(
                "panel {index} names group {binding} of {groups}"
            )));
        }
        if previous.is_some_and(|prev| binding < prev) {
            return Err(invalid("panels are not in ascending binding order"));
        }
        previous = Some(binding);
        let range = &mut out[binding];
        if range.start >= range.end {
            *range = index..index + 1;
        } else if range.end != index {
            return Err(invalid("the panels of one binding are not contiguous"));
        } else {
            range.end = index + 1;
        }
    }
    Ok(out)
}

/// What one layer's panel walk reads besides the device, the ring and its schedule.
struct PanelSweep<'s, 'w> {
    geo: &'s RunGeo,
    profile: &'s SlowlinkProfile,
    w: &'s WinCtx<'w>,
    layer: usize,
    routes: &'s dsp::PromptRouteStorage,
    route_plan: &'s dsp::LayerRoutePlan,
    /// The layer's groups in ring order (one transport page each); group `i` binds
    /// `ranges[i]`.
    groups: &'s [ExpertGroupPlan],
    /// Ascending, disjoint binding ranges, one per group, covering every expert that
    /// holds a K2 row or a cohort slot.
    ranges: &'s [(usize, usize)],
    /// Lab fault injection: fail inside the first panel that reads a group.
    inject: bool,
}

/// One host layer's panels over `groups`, in ascending binding order: each group's
/// K2 GEMMs, its ordered combine and the cohort tiles (one-row decode, K1, F32 tail)
/// against its binding ([`dsp::execute_panel`]), then the terminal combine
/// ([`dsp::finish_combine`]). The S1 page-order sweep passes the layer's
/// [`PanelSchedule`] groups and [`group_binding_ranges`]; the touched-only SP mode
/// passes a [`SpSchedule`]'s groups and tight ranges over the experts the layer
/// touches. The arithmetic is the one the walk always ran: only which experts a
/// binding holds differs.
fn sweep_panel_groups<S: RingSchedule>(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    streams: &GpuTensor,
    ring_core: &mut ExpertPanelRing,
    schedule: &mut S,
    sweep: PanelSweep<'_, '_>,
) -> R<()> {
    let PanelSweep {
        geo,
        profile,
        w,
        layer,
        routes,
        route_plan,
        groups,
        ranges,
        inject,
    } = sweep;
    #[cfg(not(any(test, feature = "lab", feature = "reference-parity")))]
    let _ = inject;
    let panels = route_plan
        .schedule(ranges, dsp::PanelLimits::S1)
        .map_err(dsp_err("panel schedule"))?;
    let by_group = panels_by_group(panels.iter().map(|panel| panel.binding), groups.len())?;
    let work: Vec<GroupWork> = groups
        .iter()
        .enumerate()
        .map(|(index, group)| GroupWork {
            group: group.group,
            bound: by_group[index].start < by_group[index].end,
        })
        .collect();
    #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
    let first_bound = work.iter().position(|group| group.bound);
    // The panel scratch is anchored on a prefill tile of the window (on the
    // aggregate route a WMMA tile when the window has one, else its first
    // multi-row tile, which may itself be an F32-tail tile). Every cohort
    // tile that runs its own original call registers its own operands
    // exactly once: each one-row decode tile (the incumbent's decode call)
    // and, on the aggregate route, each F32-tail tile. A window of decode
    // tiles only has no prefill tile of its own: its anchor is a stand-in
    // prefill tile of the admitted tile rows, used only to borrow the
    // shared scratch views (it is never prepared and never registered, so
    // no row reads or writes through it).
    let SweepTiles {
        anchor: anchor_tile,
        cohort: cohort_tiles,
    } = sweep_tiles(profile.geometry.route, w.tiles, geo.few_row_limit);
    let anchor_real = anchor_tile.is_some();
    let stand_in;
    let anchor: &TilePlan = match anchor_tile {
        Some(ti) => &w.tiles[ti],
        None => {
            let rows = profile.tile_rows;
            stand_in = TilePlan {
                index: 0,
                window: w.window.index,
                prompt_row: 0,
                window_row: 0,
                stream_offset: 0,
                rows,
                position: w.tiles[0].position,
                routing: tile_routing(forward, &mut *gpu, rows),
                ends_at_cut: None,
                last_in_window: false,
            };
            &stand_in
        }
    };
    let mut frame_tiles: Vec<&TilePlan> = vec![anchor];
    frame_tiles.extend(cohort_tiles.iter().map(|&ti| &w.tiles[ti]));
    // The K2 panels run under the anchor tile's ambient scope; each decode
    // or K1 step enters its own tile's scope through the hook.
    let (scopes, _) = forward.enter_tile_scopes(&mut *gpu, anchor.rows);
    let result = with_tile_frames(
        forward,
        bundle,
        &mut *gpu,
        streams,
        geo,
        &frame_tiles,
        layer,
        |gpu, frames| {
            // Each call is sealed and admitted under its own tile's scope
            // (its routing rows), exactly as its route preparation was.
            let mut sealed = Vec::with_capacity(frames.len());
            for (tile, frame) in frame_tiles.iter().zip(frames) {
                let (scopes, _) = forward.enter_tile_scopes(gpu, tile.rows);
                let one = (|| -> R<_> {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a host layer's FFN is not MoE"));
                    };
                    Ok(frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?)
                })();
                Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                sealed.push(one?);
            }
            let mut calls = Vec::with_capacity(sealed.len());
            for (tile, sealed) in frame_tiles.iter().zip(&sealed) {
                let (scopes, _) = forward.enter_tile_scopes(gpu, tile.rows);
                let call = dsp::PreparedChunkCall::new(
                    &*gpu,
                    sealed,
                    tile.window_row,
                    tile.position,
                    profile.geometry.route,
                );
                Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                calls.push(call.map_err(dsp_err("admission"))?);
            }
            let mut scratch = dsp::PanelScratch::new(&calls[0], dsp::PanelLimits::S1)
                .map_err(dsp_err("panel scratch"))?;
            // Each call registers once by its own actual kind, the anchor
            // included when it is itself a tail tile; the stand-in never.
            for (index, call) in calls.iter().enumerate() {
                if index == 0 && !anchor_real {
                    continue;
                }
                match call.tile_kind() {
                    dsp::TileKind::Decode => scratch
                        .add_decode_tile(call)
                        .map_err(dsp_err("decode tile"))?,
                    dsp::TileKind::AggregateF32Tail => scratch
                        .add_tail_tile(call)
                        .map_err(dsp_err("F32 tail tile"))?,
                    _ if index == 0 => {}
                    kind => {
                        return Err(invalid(format!(
                            "slow-link cohort tile at window row {} sealed as {kind:?}, not a \
                             decode or F32-tail call",
                            call.row0()
                        )))
                    }
                }
            }
            let mut cursor = dsp::OrderedCombineCursor::start(gpu, routes, &panels)
                .map_err(dsp_err("combine start"))?;
            let mut hook = ForwardScopeHook {
                forward,
                tiles: w.tiles,
                open: Vec::new(),
            };
            let mut ring = HipRing {
                ring: &mut *ring_core,
                schedule: &mut *schedule,
            };
            walk_layer_groups(&mut ring, gpu, layer, &work, &mut |gpu, index, parts| {
                let parts =
                    parts.ok_or_else(|| invalid("slow-link panel lease carries no binding"))?;
                #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
                if inject && Some(index) == first_bound {
                    return Err(invalid("injected slow-link failure during a panel"));
                }
                let group = &groups[index];
                let (Some(&lo), Some(&last)) = (group.experts.first(), group.experts.last()) else {
                    return Err(invalid("an empty expert group"));
                };
                let binding = dsp::ExpertPanelBinding {
                    layer,
                    epoch: parts.epoch,
                    expert_lo: lo,
                    expert_hi: last + 1,
                    gate_up_table: parts.gate_up_table,
                    down_table: parts.down_table,
                    codec: dsp::PanelCodec::Mq4,
                    source: dsp::BindingSource::Staged {
                        slot: parts.ticket.slot as u32,
                    },
                };
                for panel_index in by_group[index].clone() {
                    let panel = &panels[panel_index];
                    dsp::execute_panel(
                        gpu,
                        binding,
                        route_plan.view(routes, panel),
                        &mut scratch,
                        &mut cursor,
                        &mut hook,
                    )
                    .map_err(dsp_err("panel"))?;
                }
                Ok(())
            })?;
            dsp::finish_combine(gpu, routes, &mut cursor, &mut hook).map_err(dsp_err("combine"))
        },
    );
    Qwen4GpuForward::leave_tile_scopes(&mut *gpu, scopes);
    result
}

// ── abort bookkeeping ────────────────────────────────────────────────

/// How far a window got: what its failure must undo.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    /// The window transaction is open (entry snapshot taken).
    pub(crate) txn_open: bool,
    /// The ring started a window and may have copies or readers in flight.
    pub(crate) ring_primed: bool,
    /// The window runs SP: the xfer stream may have copies in flight.
    pub(crate) sp_armed: bool,
}

/// The cleanup of a failed window, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AbortSteps {
    /// Drain both streams first: an in-flight copy or reader must finish before
    /// anything it reads or writes is restored or released.
    pub(crate) drain_ring: bool,
    /// Then join SP's xfer stream (quarantining the SP owners when that
    /// fails) and disable SP for the process.
    pub(crate) sync_xfer: bool,
    /// Then restore the entry state and drop the transaction.
    pub(crate) abort_txn: bool,
}

impl Progress {
    pub(crate) fn abort_steps(self) -> AbortSteps {
        AbortSteps {
            drain_ring: self.ring_primed,
            sync_xfer: self.sp_armed,
            abort_txn: self.txn_open,
        }
    }
}

// ── selection ────────────────────────────────────────────────────────

struct Selection {
    plan: Qwen4PromptPlan,
    profile: SlowlinkProfile,
    geo: RunGeo,
    /// One pre-mutation SP verdict per window; empty when SP is not armed.
    sp_verdicts: Vec<Result<(), SpBypass>>,
}

/// The full-hidden and final-row head of the request against the forward's
/// scratch, before any mutation (the incumbent's own preflight).
fn validate_head(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    output: &PromptOutput<'_>,
    geo: &RunGeo,
    last_rows: usize,
) -> Result<(), String> {
    let Some((logits, top1)) = output.head() else {
        return Ok(());
    };
    if logits.dtype != DType::F32 || logits.numel() != geo.config.vocab_size {
        return Err(format!(
            "head logits must be F32 with {} elements",
            geo.config.vocab_size
        ));
    }
    if let Some(top1) = top1 {
        if top1.dtype != DType::Raw || top1.buf.size() < std::mem::size_of::<i32>() {
            return Err("head top1 output must be Raw with one i32".to_string());
        }
    }
    let final_hyper =
        crate::gpu_forward::hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)
            .map_err(|error| error.to_string())?;
    let lm_head = crate::gpu_forward::dense_ref(&bundle.weights, &bundle.weights.root.lm_head)
        .map_err(|error| error.to_string())?;
    let router_logits = matrix_view(
        &forward.scratch.router_logits,
        last_rows,
        router_columns(geo.dims),
    )
    .map_err(|error| error.to_string())?;
    let scratch_desc = layer_scratch(&forward.scratch, &router_logits);
    validate_final_hyper(
        geo.dims,
        &final_hyper,
        &forward.scratch.streams,
        &scratch_desc,
        last_rows,
    )
    .map_err(|error| format!("validate final hyper: {error:?}"))?;
    let hidden = view(&forward.scratch.hc_mixed, 0, last_rows * geo.hidden);
    validate_lm_head(&lm_head, &hidden, logits, last_rows, 1)
        .map_err(|error| format!("validate LM head: {error:?}"))
}

/// The pre-mutation dry run of the dispatch admission: seal the first host
/// layer's MoE call for every distinct tile signature of the plan and ask the
/// dispatch crate to admit it. The real sealed call's route must also equal
/// the admitted profile's route: the census (route storage, tile-F32 and
/// scratch charges) was admitted for `profile.geometry.route` alone, so a call
/// sealed for another route is refused here, after the tile scope is restored
/// and before the S1 runtime gate or any other mutation, instead of running
/// against the other route's census.
fn preflight_tiles(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    geo: &RunGeo,
    plan: &Qwen4PromptPlan,
    profile: &SlowlinkProfile,
    window_routes: &[dsp::WindowRoute],
) -> Result<(), String> {
    let mut signatures = plan.signatures();
    // A window of one-row decode tiles only has no prefill tile to anchor the
    // panel scratch: the sweep uses a stand-in prefill tile of the admitted tile
    // rows, admitted here with the rest.
    if plan.windows.iter().any(|window| {
        plan.window_tiles(window)
            .iter()
            .all(|tile| tile.routing.route_rows == 1)
    }) {
        let rows = forward.scratch.max_chunk;
        let routing = tile_routing(forward, gpu, rows);
        if !signatures.iter().any(|(r, known)| {
            *r == rows
                && known.route_rows == routing.route_rows
                && known.bf16_streams == routing.bf16_streams
        }) {
            signatures.push((rows, routing));
        }
    }
    preflight_signatures(
        forward,
        bundle,
        gpu,
        geo,
        signatures,
        plan.entry_pos,
        profile,
    )?;
    // The windows that run another route than the profile's (the hybrid SP's F16
    // `Aggregate` windows on a symmetric-IU4 profile) are admitted for that route under the
    // F16 MoE scope, the scope the run itself sets for them.
    let mut override_signatures: Vec<(usize, TileRouting)> = Vec::new();
    for (window, &route) in plan.windows.iter().zip(window_routes) {
        if route == profile.geometry.route {
            continue;
        }
        for tile in plan.window_tiles(window) {
            let known = |(rows, routing): &(usize, TileRouting)| {
                *rows == tile.rows
                    && routing.route_rows == tile.routing.route_rows
                    && routing.bf16_streams == tile.routing.bf16_streams
            };
            if !override_signatures.iter().any(known) {
                override_signatures.push((tile.rows, tile.routing));
            }
        }
    }
    if override_signatures.is_empty() {
        return Ok(());
    }
    preflight_signatures_on(
        forward,
        bundle,
        gpu,
        geo,
        override_signatures,
        plan.entry_pos,
        profile,
        PreflightRoute {
            route: dsp::WindowRoute::Aggregate,
            f16_moe: true,
        },
    )
}

/// The window route a preflight admits its tiles for and whether the tile scope suppresses
/// the symmetric IU4 MoE arm (a hybrid SP window on a symmetric-IU4 profile).
#[derive(Clone, Copy)]
struct PreflightRoute {
    route: dsp::WindowRoute,
    f16_moe: bool,
}

/// [`preflight_tiles`] over explicit launch signatures at `entry_pos`: the
/// request path's plan signatures, or the load's own admitted-tile signature
/// (the capability question asked before the runtime is materialized).
fn preflight_signatures(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    geo: &RunGeo,
    signatures: Vec<(usize, TileRouting)>,
    entry_pos: usize,
    profile: &SlowlinkProfile,
) -> Result<(), String> {
    preflight_signatures_on(
        forward,
        bundle,
        gpu,
        geo,
        signatures,
        entry_pos,
        profile,
        PreflightRoute {
            route: profile.geometry.route,
            f16_moe: false,
        },
    )
}

/// [`preflight_signatures`] for an explicit window route and MoE scope.
#[allow(clippy::too_many_arguments)]
fn preflight_signatures_on(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    geo: &RunGeo,
    signatures: Vec<(usize, TileRouting)>,
    entry_pos: usize,
    profile: &SlowlinkProfile,
    on: PreflightRoute,
) -> Result<(), String> {
    let layer = geo
        .band
        .clone()
        .find(|&layer| {
            forward
                .moe
                .get(layer)
                .and_then(Option::as_ref)
                .is_some_and(|runtime| runtime.s1_host_mapped())
        })
        .ok_or_else(|| "no host-resident expert layer".to_string())?;
    let mut arms: Vec<dsp::WindowArm> = Vec::new();
    for (rows, routing) in signatures {
        let stub = TilePlan {
            index: 0,
            window: 0,
            prompt_row: 0,
            window_row: 0,
            stream_offset: 0,
            rows,
            position: entry_pos,
            routing,
            ends_at_cut: None,
            last_in_window: false,
        };
        forward.slowlink.set_sp_f16_moe(on.f16_moe);
        let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
        let result = with_tile_frame(
            forward,
            bundle,
            gpu,
            &forward.scratch.streams,
            geo,
            &stub,
            layer,
            |gpu, frame| {
                let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                    return Err(invalid("a host layer's FFN is not MoE"));
                };
                let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                dsp::PreparedChunkCall::new(&*gpu, &sealed, 0, stub.position, on.route)
                    .map(|call| {
                        (
                            call.arm_signature(),
                            call.is_k2(),
                            call.window_route(),
                            call.tile_kind(),
                        )
                    })
                    .map_err(dsp_err("admission"))
            },
        );
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        forward.slowlink.set_sp_f16_moe(false);
        let (arm, is_k2, route, kind) = result.map_err(|error| {
            format!(
                "tile of {rows} rows (route rows {}): {error}",
                routing.route_rows
            )
        })?;
        if route != on.route {
            return Err(format!(
                "tile of {rows} rows (route rows {}): the sealed call's window route is \
                 {route:?} but the window was admitted for {:?} (the profile's census is for \
                 {:?})",
                routing.route_rows, on.route, profile.geometry.route
            ));
        }
        // The plan counted this tile's rows toward its window's K1 / F32-tail
        // capacities from its original routing rows; the sealed call's own
        // class must be that cohort, else the capacities were not the call's.
        let expected = tile_cohort(route, routing.route_rows, geo.few_row_limit);
        if kind_cohort(kind) != expected {
            return Err(format!(
                "tile of {rows} rows (route rows {}): the sealed call is {kind:?} but its \
                 original routing accounts it as {expected:?}",
                routing.route_rows
            ));
        }
        arms.push(dsp::WindowArm {
            signature: arm,
            k2: is_k2,
            kind,
        });
    }
    // Every tile shares the layer geometry (the first five signature values);
    // the grouped-path flag is the K2 tiles' one frozen arm. A few-row tail (a
    // prefill K1 call or the one-row decode call) keeps its own original arm as
    // a separate cohort, so it is compared on geometry alone and never refused
    // for its arm. One shared rule with the dispatch crate's prepare.
    if !dsp::window_arms_compatible(on.route, &arms) {
        return Err(format!(
            "the plan's tiles seal incompatible dispatch arms (signature, K2): {arms:?}"
        ));
    }
    Ok(())
}

/// Decide the request. `Err` is the bypass reason: the incumbent runs, nothing
/// was mutated and no S1 resource was allocated for this request.
fn select(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    tokens: &[u32],
    output: &PromptOutput<'_>,
    mode: SlowlinkMode,
) -> Result<Selection, std::borrow::Cow<'static, str>> {
    if mode == SlowlinkMode::Off {
        return Err("HIPFIRE_QWEN4_SLOWLINK_PREFILL=0".into());
    }
    let Some(profile) = forward.slowlink_profile().copied() else {
        return Err("no admitted slow-link profile".into());
    };
    if let Some(reason) = forward.slowlink.unavailable.clone() {
        return Err(reason.into());
    }
    bundle
        .slowlink_prompt_eligible()
        .map_err(std::borrow::Cow::Borrowed)?;
    if let Some(why) = forward.slowlink_forward_blocker(&bundle.config) {
        return Err(std::borrow::Cow::Borrowed(why));
    }
    let rows = tokens.len();
    let entry = bundle.state.position;
    let end = entry
        .checked_add(rows)
        .ok_or("request position overflows")?;
    if end > bundle.state.max_seq_len {
        return Err(format!(
            "request end {end} exceeds the context capacity {}",
            bundle.state.max_seq_len
        )
        .into());
    }
    if tokens
        .iter()
        .any(|&token| token as usize >= bundle.config.vocab_size)
    {
        return Err("a token id is outside the embedding vocabulary".into());
    }
    // The forward's scratch must be EXACTLY the admitted tile: a larger one
    // (an incumbent 4096/8192-row chunk the caller asked for) would be retained
    // scratch the profile never charged, and S1 never reroutes after allocation.
    if forward.scratch.max_chunk != profile.tile_rows {
        return Err(format!(
            "forward scratch holds {} rows, the admitted S1 tile is {}",
            forward.scratch.max_chunk, profile.tile_rows
        )
        .into());
    }
    let cuts = bundle.prompt_txn_cuts(entry, end);
    let sp = forward.slowlink.sp_profile();
    let shape = RequestShape {
        rows,
        entry_pos: entry,
        session_cuts: 0,
        capture: output.capture_kind(),
    };
    sp_request_verdict(mode, &profile, sp.as_ref(), shape, cuts.len())
        .map_err(|bypass: RequestBypass| format!("{bypass:?}"))?;
    let few_row_limit = few_row_limit_of(forward, gpu, profile.tile_rows);
    let verdict = {
        let cell = std::cell::RefCell::new(&mut *gpu);
        let routing_forward: &Qwen4GpuForward = &*forward;
        let routing =
            |tile_rows: usize| tile_routing(routing_forward, &mut **cell.borrow_mut(), tile_rows);
        plan_prompt_verdict(&PromptPlanInputs {
            profile: &profile,
            entry_pos: entry,
            rows,
            cuts: &cuts,
            routing: &routing,
            few_row_limit,
        })
        .map_err(|error| error.to_string())?
    };
    let plan = verdict.map_err(|bypass| bypass.to_string())?;
    let sp_verdicts = sp_window_verdicts(sp.as_ref(), &plan, output.capture_kind());
    // Below S1's own multi-chunk floor the request is SP's or the incumbent's:
    // a window SP does not take never runs a stream-all sweep.
    sp_floor_verdict(profile.tile_rows, rows, &sp_verdicts)
        .map_err(|bypass| format!("{bypass:?}"))?;
    prompt_bytes(&plan).map_err(|error| error.to_string())?;
    validate_window_layouts(&profile, &plan).map_err(|error| error.to_string())?;
    // One line per request: every window is one full expert sweep, so this is
    // the sweep count the receipts could not previously show.
    eprintln!(
        "  qwen4 S1 plan: {rows} rows from position {entry}, {} windows (= expert sweeps), window rows {:?} of the admitted {}, {}-row tiles",
        plan.windows.len(),
        plan.windows.iter().map(|w| w.rows).collect::<Vec<_>>(),
        profile.window_rows,
        profile.tile_rows,
    );
    for window in &plan.windows {
        if bundle.prompt_txn_cuts(window.position, window.end_position()) != window.cuts {
            return Err(format!(
                "window {} cuts differ from the transaction's pending boundaries",
                window.index
            )
            .into());
        }
    }
    let num_experts = profile.geometry.num_experts;
    let geo = RunGeo::new(forward, bundle, few_row_limit, num_experts)
        .map_err(|error| error.to_string())?;
    let last_rows = plan.tiles.last().map_or(0, |tile| tile.rows);
    validate_head(forward, bundle, output, &geo, last_rows)?;
    for layer in geo.band.clone() {
        if let Some(runtime) = forward.moe.get(layer).and_then(Option::as_ref) {
            if runtime.s1_host_mapped() {
                runtime
                    .s1_layer_check()
                    .map_err(|why| format!("host layer {layer}: {why}"))?;
            }
        }
    }
    // The MTP entry `pending` backup is bound at MTP attach, never built here.
    if matches!(
        output,
        PromptOutput::Capture {
            pending: Some(_),
            ..
        }
    ) && !bundle.slowlink_pending_bound()
    {
        return Err("the MTP pending backup was not bound at load".into());
    }
    let window_routes = sp_window_routes(
        profile.geometry.route,
        sp.as_ref(),
        &sp_verdicts,
        plan.windows.len(),
    );
    preflight_tiles(forward, bundle, gpu, &geo, &plan, &profile, &window_routes)?;
    runtime_gate(
        forward.slowlink.runtime.is_some(),
        forward.slowlink.unavailable.as_deref(),
    )?;
    Ok(Selection {
        plan,
        profile,
        geo,
        sp_verdicts,
    })
}

/// The S1 runtime is bound ONCE at the selected load ([`bind_at_load`]); a
/// request only consumes it. A load-time decline stays the request's named
/// bypass; a runtime that was never bound is an error, never a request-time
/// allocation (a mid-request device allocation could fail after the request
/// mutated state or fall back to the incumbent with S1's placement already
/// charged).
fn runtime_gate(runtime_bound: bool, declined: Option<&str>) -> Result<(), String> {
    if let Some(reason) = declined {
        return Err(reason.to_string());
    }
    if !runtime_bound {
        return Err(
            "the slow-link runtime was not bound at load (requests never construct it)".to_string(),
        );
    }
    Ok(())
}

// ── tile frame ───────────────────────────────────────────────────────

/// Everything the typed operations of one `(tile, layer)` borrow, built under
/// the tile's scope and handed to a callback ([`with_tile_frame`]).
struct TileFrame<'f> {
    env: &'f LayerPhaseEnv<'f>,
    description: &'f crate::program::Qwen4LayerDescription<'f>,
    ctx: &'f DispatchCtx,
    scratch_desc: &'f crate::program::Qwen4LayerScratch<'f>,
    layer: &'f crate::weights::Qwen4LayerWeights,
}

/// Build the layer description, scratch view and [`LayerPhaseEnv`] of `layer`
/// on `tile` (the tile's rows of `streams` are the layer's HC streams) and run
/// `f` on them. The caller has entered the tile's Gpu scope.
#[allow(clippy::too_many_arguments)]
fn with_tile_frame<T>(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    streams: &GpuTensor,
    geo: &RunGeo,
    tile: &TilePlan,
    layer_index: usize,
    f: impl FnOnce(&mut Gpu, &TileFrame<'_>) -> R<T>,
) -> R<T> {
    with_tile_frames(
        forward,
        bundle,
        gpu,
        streams,
        geo,
        &[tile],
        layer_index,
        |gpu, frames| f(gpu, &frames[0]),
    )
}

/// The frames of several tiles of one layer at once, all alive together (the
/// panel scratch holds the operands of its one-row decode tiles for the whole
/// sweep). `frames[i]` belongs to `tiles[i]`; the caller has entered whichever
/// Gpu scope its own steps need.
#[allow(clippy::too_many_arguments)]
fn with_tile_frames<T>(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    streams: &GpuTensor,
    geo: &RunGeo,
    tiles: &[&TilePlan],
    layer_index: usize,
    f: impl FnOnce(&mut Gpu, &[TileFrame<'_>]) -> R<T>,
) -> R<T> {
    let layer = bundle
        .weights
        .layer(layer_index)
        .ok_or_else(|| invalid(format!("layer {layer_index} is not owned by this band")))?;
    // The D2 executor gate (`cpu_split_executor`) sees only the row count, so
    // the dry-run and execution frames bind the same selected-runtime routed
    // view as the forward loop: S1 never runs expert parallel, so no
    // balanced-live or EP-full runtime exists here and the static runtime is
    // the selection, through the shared selector. An actually CPU-split sealed
    // selection refuses at the dispatch gate, never S1-routed.
    let layer_moe = crate::gpu_forward::select_layer_moe(
        None::<&crate::gpu_forward::Qwen4MoeLayerRuntime>,
        None,
        false,
        false,
        forward.moe.get(layer_index).and_then(Option::as_ref),
    );
    // S1 per-call CPU-wide scope: never armed from the global lab override.
    // Ordinary prefill tiles stay on the default-4 gate; the frozen row-exact
    // width is an upper clamp only. Read once from this GPU device.
    let verify_rows = crate::gpu_forward::qwen4_row_exact_verify_rows(gpu);
    let tile_streams: Vec<GpuTensor> = tiles
        .iter()
        .map(|tile| f32_view(streams, tile.stream_offset, tile.rows * geo.wide))
        .collect();
    let router_logits = tiles
        .iter()
        .map(|tile| {
            matrix_view(
                &forward.scratch.router_logits,
                tile.rows,
                router_columns(geo.dims),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scratch_descs: Vec<_> = router_logits
        .iter()
        .map(|logits| layer_scratch(&forward.scratch, logits))
        .collect();
    let descriptions = tiles
        .iter()
        .map(|tile| {
            let mut description = layer_desc(
                &bundle.weights,
                layer,
                layer_moe,
                &geo.config,
                &forward.decode_q8,
                tile.routing.route_rows,
                geo.few_row_limit,
                None,
            )?;
            if let Qwen4FfnBinding::Moe(moe) = &mut description.ffn {
                // S1 runs no host arithmetic: `layer_desc` copied the layer's D1
                // flag, and a 1..=4-row tile (`rows == route_rows` on gfx1100)
                // would seal as a CPU-deferred call that dispatch refuses
                // mid-window. D2 is already splitless through
                // `routed_view(false, ..)`.
                moe.cpu_deferred_experts = s1_tile_cpu_deferred(moe.cpu_deferred_experts);
                moe.routed_experts = crate::gpu_forward::bind_routed_view(
                    layer_moe,
                    moe.routed_experts,
                    |runtime| {
                        runtime.routed_view(
                            false,
                            tile.rows,
                            tile.routing.route_rows,
                            false,
                            verify_rows,
                        )
                    },
                );
            }
            Ok::<_, Qwen4GpuForwardError>(description)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ctx = DispatchCtx::new(gpu);
    let envs: Vec<LayerPhaseEnv<'_>> = tiles
        .iter()
        .zip(&tile_streams)
        .map(|(tile, tile_streams)| LayerPhaseEnv {
            scratch: &forward.scratch,
            decode_q8: &forward.decode_q8,
            trunk_zba: &forward.trunk_zba,
            trunk_a4: &forward.trunk_a4,
            prefill_scope: forward.prefill_scope,
            bundle,
            dims: geo.dims,
            bf16_state: tile.routing.bf16_streams,
            route_n: tile.routing.route_rows,
            few_row_limit: geo.few_row_limit,
            rows: tile.rows,
            start_position: tile.position,
            streams: tile_streams,
            gates: &forward.scratch.hc_gates,
            ffn_partial: false,
        })
        .collect();
    let frames: Vec<TileFrame<'_>> = (0..tiles.len())
        .map(|index| TileFrame {
            env: &envs[index],
            description: &descriptions[index],
            ctx: &ctx,
            scratch_desc: &scratch_descs[index],
            layer,
        })
        .collect();
    f(gpu, &frames)
}

// ── the run ──────────────────────────────────────────────────────────

/// The window being run, as the layer loops see it.
#[derive(Clone, Copy)]
struct WinCtx<'w> {
    window: &'w WindowPlan,
    tiles: &'w [TilePlan],
    /// The window's tokens.
    tokens: &'w [u32],
    last_window: bool,
}

impl<'w> WinCtx<'w> {
    fn tile_tokens(&self, ti: usize) -> &'w [u32] {
        let tile = &self.tiles[ti];
        &self.tokens[tile.window_row..tile.window_row + tile.rows]
    }
}

struct Run<'a, 'o> {
    forward: &'a mut Qwen4GpuForward,
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    rt: &'a mut SlowlinkRuntime,
    geo: &'a RunGeo,
    profile: &'a SlowlinkProfile,
    plan: &'a Qwen4PromptPlan,
    tokens: &'a [u32],
    output: &'a mut PromptOutput<'o>,
    ple: PleCursor,
    /// One pre-mutation SP verdict per window of the plan (empty: SP not armed).
    sp_verdicts: &'a [Result<(), SpBypass>],
}

impl Run<'_, '_> {
    /// One window: transaction, layer-major walk, commit (or abort).
    fn window(&mut self, window: &WindowPlan, last_window: bool) -> R<WindowReport> {
        let (plan, tokens) = (self.plan, self.tokens);
        let window_tokens = &tokens[window.prompt_row..window.prompt_row + window.rows];
        let w = WinCtx {
            window,
            tiles: plan.window_tiles(window),
            tokens: window_tokens,
            last_window,
        };
        // Each window is its own transaction, restored on its own failure, so
        // the drafter's `pending` row is backed up at THIS window's entry (the
        // previous window's sink left it at the window's first pair), exactly
        // the state the window's abort restores target and head to. A declined
        // request never reaches here, so it never touches `pending`.
        if let PromptOutput::Capture {
            pending: Some(pending),
            ..
        } = &*self.output
        {
            self.bundle
                .slowlink_save_pending(&mut *self.gpu, pending)
                .map_err(bundle_err("save pending hidden"))?;
        }
        let mut txn = self
            .bundle
            .prompt_txn_begin(&mut *self.gpu, window_tokens)
            .map_err(bundle_err("begin"))?;
        let mut progress = Progress {
            txn_open: true,
            ring_primed: false,
            sp_armed: false,
        };
        match self.window_body(&w, &mut txn, &mut progress) {
            Ok(report) => self
                .bundle
                .prompt_txn_commit(&mut *self.gpu, txn, window_tokens)
                .map(|()| report)
                .map_err(bundle_err("commit")),
            Err(error) => Err(self.fail_window(txn, progress, error)),
        }
    }

    /// The failed window's cleanup, in order: drain the ring, then restore the
    /// entry state. The original error survives unless a cleanup also failed.
    fn fail_window(
        &mut self,
        txn: PromptTxn,
        progress: Progress,
        error: Qwen4GpuForwardError,
    ) -> Qwen4GpuForwardError {
        let steps = progress.abort_steps();
        let mut extra: Vec<String> = Vec::new();
        self.forward.slowlink.set_sp_f16_moe(false);
        if steps.drain_ring {
            if let Err(drain) = self.rt.ring.drain(&*self.gpu) {
                extra.push(format!("ring drain failed: {drain}"));
            }
        }
        if steps.sync_xfer {
            self.sp_abort(&error.to_string(), &mut extra);
        }
        self.touched_abort(&error.to_string());
        if steps.abort_txn {
            if let Err(abort) = self.bundle.prompt_txn_abort(&mut *self.gpu, txn) {
                extra.push(format!("transaction abort failed: {abort}"));
            }
        } else {
            drop(txn);
        }
        if extra.is_empty() {
            error
        } else {
            invalid(format!("{error}; {}", extra.join("; ")))
        }
    }

    fn window_body(
        &mut self,
        w: &WinCtx<'_>,
        txn: &mut PromptTxn,
        progress: &mut Progress,
    ) -> R<WindowReport> {
        let geo = self.geo;
        self.ple.histories =
            PleCursor::histories_for(self.bundle.state.ple_history, w.tiles, w.tokens);
        if w.tiles
            .iter()
            .any(|tile| tile.routing.route_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS)
        {
            if let Some(qsa) = self
                .bundle
                .state
                .qsa
                .first()
                .filter(|qsa| qsa.f16_arena.is_none())
            {
                ensure_qsa_gathered_wmma_workspace(
                    &mut *self.gpu,
                    qsa.format,
                    geo.config.num_key_value_heads,
                    w.window.end_position(),
                )?;
            }
        }
        #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
        if self
            .bundle
            .slowlink_take_injection(SlowlinkFailPoint::BeforeDma)
        {
            return Err(invalid(
                "injected slow-link failure before the first expert copy",
            ));
        }
        let wall = Instant::now();
        let sp_on = self.sp_window_on(w);
        let touched_on = self.touched_window_on(w);
        // A selective window (SP or touched-only) primes with zero enqueues.
        let sparse = sp_on || touched_on;
        let layout = self.window_layout(w, sp_on)?;
        let mut routes = dsp::PromptRouteStorage::from_region(&self.rt.route_region, &layout)
            .map_err(dsp_err("route storage"))?;
        let mut report = WindowReport::default();
        progress.ring_primed = true;
        // The band's leading device-resident layers that upload per-tile host data (the
        // band-start layer's token ids in `embed_tile`, the PLE layer's staged rows) would queue
        // those small uploads behind the prime's page copies on the shared copy engine, so the
        // prime waits until the last of them has enqueued its tiles; the remaining resident
        // layers still cover the prime's copies. A band that starts on a host layer (or has
        // none) primes first, as before. An SP or touched-only window primes with zero enqueues
        // and its resident layers never pump the sweep schedule.
        let first_host = geo.band.clone().find(|&layer| self.rt.host_layer[layer]);
        let prime_after = first_host
            .filter(|&host| !sparse && host > geo.band.start)
            .map(|host| match geo.ple_layer {
                Some(ple) if ple > geo.band.start && ple < host => ple,
                _ => geo.band.start,
            });
        if sp_on {
            let f16_moe = self.window_route(w) != self.profile.geometry.route;
            self.sp_begin_window(&mut routes, progress, f16_moe)?;
        } else if touched_on {
            self.touched_begin_window()?;
        } else if prime_after.is_none() {
            self.rt.ring.prime(&*self.gpu, &mut self.rt.schedule)?;
        }
        for layer in geo.band.clone() {
            self.rt.ring.mark_layer_begin(&*self.gpu, layer)?;
            if self.rt.host_layer[layer] {
                self.host_layer(w, layer, &mut routes, txn, &mut report)?;
            } else {
                let primed = !sparse && prime_after.is_none_or(|after| layer > after);
                self.resident_layer(w, layer, txn, primed)?;
            }
            self.rt.ring.mark_layer_end(&*self.gpu, layer)?;
            if prime_after == Some(layer) {
                self.rt.ring.prime(&*self.gpu, &mut self.rt.schedule)?;
            }
            #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
            if layer == geo.band.start
                && self
                    .bundle
                    .slowlink_take_injection(SlowlinkFailPoint::AfterLayer)
            {
                return Err(invalid("injected slow-link failure after the first layer"));
            }
        }
        self.rt.ring.drain(&*self.gpu)?;
        progress.ring_primed = false;
        let timeline = self.rt.ring.timeline(&*self.gpu)?;
        let stats = timeline.stats;
        if stats.timeline_overflow > 0 || stats.layer_mark_overflow > 0 {
            return Err(invalid(format!(
                "slow-link timeline overflowed ({} copy records, {} layer marks): the receipt would be incomplete",
                stats.timeline_overflow, stats.layer_mark_overflow
            )));
        }
        if sp_on {
            self.sp_end_window(&stats, w.window.index)?;
        } else if touched_on {
            self.touched_end_window(&stats, w.window.index)?;
        }
        let sweep = self.rt.schedule.sweep_bytes();
        if !sparse && stats.submitted_payload_bytes != sweep {
            return Err(invalid(format!(
                "slow-link window submitted {} payload bytes, one sweep is {sweep}",
                stats.submitted_payload_bytes
            )));
        }
        if report.useful_bytes > stats.submitted_payload_bytes {
            return Err(invalid(format!(
                "slow-link window read {} useful bytes of {} submitted",
                report.useful_bytes, stats.submitted_payload_bytes
            )));
        }
        report.stats = stats;
        report.overlap = timeline.overlap;
        report.sweep_bytes = sweep;
        report.wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let panel_busy: f64 = timeline
            .spans
            .iter()
            .filter_map(|span| span.compute.map(|(begin, end)| (end - begin).max(0.0)))
            .sum();
        report.nonexpert_ms = (timeline.overlap.compute_busy_ms - panel_busy).max(0.0);
        Ok(report)
    }

    /// A device-resident layer: the incumbent's per-tile steps.  `primed` is false only while
    /// the window's ring prime is deferred past this layer: no pump before the prime.
    fn resident_layer(
        &mut self,
        w: &WinCtx<'_>,
        layer: usize,
        txn: &mut PromptTxn,
        primed: bool,
    ) -> R<()> {
        let geo = self.geo;
        for (ti, tile) in w.tiles.iter().enumerate() {
            if layer == geo.band.start {
                self.embed_tile(w, ti)?;
            }
            self.tile_layer(w, ti, layer, None)?;
            self.cut_if(txn, tile, layer)?;
            if layer == geo.last_layer {
                self.tile_output(w, ti)?;
            }
            if primed {
                self.rt.ring.pump(&*self.gpu, &mut self.rt.schedule)?;
            }
        }
        Ok(())
    }

    /// A host layer: every tile's prefix and route preparation, the one count
    /// barrier, the panels in ascending expert order, every tile's tail.
    fn host_layer(
        &mut self,
        w: &WinCtx<'_>,
        layer: usize,
        routes: &mut dsp::PromptRouteStorage,
        txn: &mut PromptTxn,
        report: &mut WindowReport,
    ) -> R<()> {
        let geo = self.geo;
        if routes.sp() {
            return self.sp_host_layer(w, layer, routes, txn, report);
        }
        if self.touched_in_window() {
            return self.touched_host_layer(w, layer, routes, txn, report);
        }
        self.rt
            .ring
            .enter_layer(&*self.gpu, &mut self.rt.schedule, layer)?;
        routes.reset_for_layer();
        for (ti, tile) in w.tiles.iter().enumerate() {
            if layer == geo.band.start {
                self.embed_tile(w, ti)?;
            }
            self.tile_layer(w, ti, layer, Some(&mut *routes))?;
            self.cut_if(txn, tile, layer)?;
            self.rt.ring.pump(&*self.gpu, &mut self.rt.schedule)?;
        }
        let route_plan = dsp::finish_route_plan(&mut *self.gpu, routes, dsp::PanelLimits::S1)
            .map_err(dsp_err("route plan"))?;
        report.route_barrier_ms += route_plan.barrier.sync_ms;
        self.sweep_panels(w, layer, routes, &route_plan, report)?;
        for ti in 0..w.tiles.len() {
            self.tile_tail(w, ti, layer, routes)?;
            if layer == geo.last_layer {
                self.tile_output(w, ti)?;
            }
        }
        Ok(())
    }

    /// The layer's panels over the ring's groups.
    fn sweep_panels(
        &mut self,
        w: &WinCtx<'_>,
        layer: usize,
        routes: &dsp::PromptRouteStorage,
        route_plan: &dsp::LayerRoutePlan,
        report: &mut WindowReport,
    ) -> R<()> {
        let groups: Vec<ExpertGroupPlan> = self
            .rt
            .schedule
            .layer_groups(layer)
            .ok_or_else(|| invalid(format!("layer {layer} has no scheduled expert groups")))?
            .to_vec();
        let ranges = group_binding_ranges(&groups, self.geo.num_experts);
        self.credit_touched_bytes(layer, route_plan, report);
        #[cfg(any(test, feature = "lab", feature = "reference-parity"))]
        let inject = self
            .bundle
            .slowlink_take_injection(SlowlinkFailPoint::DuringPanel);
        #[cfg(not(any(test, feature = "lab", feature = "reference-parity")))]
        let inject = false;
        sweep_panel_groups(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            &mut self.rt.ring,
            &mut self.rt.schedule,
            PanelSweep {
                geo: self.geo,
                profile: self.profile,
                w,
                layer,
                routes,
                route_plan,
                groups: &groups,
                ranges: &ranges,
                inject,
            },
        )
    }

    /// Touched bytes of a layer: the experts any K2 row or the K1 cohort read,
    /// at the placement's own extents; never the prefetched remainder.
    pub(super) fn credit_touched_bytes(
        &self,
        layer: usize,
        route_plan: &dsp::LayerRoutePlan,
        report: &mut WindowReport,
    ) {
        let mut touched: Vec<usize> = route_plan
            .experts_with_rows()
            .chain(route_plan.k1_experts.iter().copied())
            .collect();
        touched.sort_unstable();
        touched.dedup();
        for expert in touched {
            report.useful_bytes += self
                .rt
                .schedule
                .expert_extent_bytes(layer, expert)
                .unwrap_or(0) as u64;
        }
    }

    /// Tile `ti` of `layer`: PLE rows, the mixer prefix, the QSA cursor, and
    /// either the rest of the layer (resident) or the route preparation (host).
    fn tile_layer(
        &mut self,
        w: &WinCtx<'_>,
        ti: usize,
        layer: usize,
        routes: Option<&mut dsp::PromptRouteStorage>,
    ) -> R<()> {
        let tile = &w.tiles[ti];
        let n = tile.rows;
        self.gpu
            .set_router_row_base(if n > 1 { tile.position } else { 0 });
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, n);
        let outcome = self.tile_layer_scoped(w, ti, layer, routes);
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        let qsa = outcome?;
        if let Some((full_len, raw_len, pooled_len, selected_len, position)) = qsa {
            // The layer's private cursor: only this layer's state moves with
            // the tile; the bundle's global position waits for the commit.
            let slot = self
                .bundle
                .state
                .qsa_local(layer)
                .ok_or_else(|| invalid("QSA state slot missing"))?;
            let state = self
                .bundle
                .state
                .qsa_mut(slot)
                .ok_or_else(|| invalid("QSA state slot disappeared"))?;
            state.full_len = full_len;
            state.raw_len = raw_len;
            state.pooled_len = pooled_len;
            state.selected_len = selected_len;
            state.position = position;
        }
        Ok(())
    }

    fn tile_layer_scoped(
        &mut self,
        w: &WinCtx<'_>,
        ti: usize,
        layer: usize,
        routes: Option<&mut dsp::PromptRouteStorage>,
    ) -> R<Option<(usize, usize, usize, usize, usize)>> {
        let geo = self.geo;
        let window_route = self.window_route(w);
        let tile = &w.tiles[ti];
        let n = tile.rows;
        if geo.ple_layer == Some(layer) {
            if let Some(owner) = self.bundle.ple.as_ref() {
                let ple_weights = self
                    .bundle
                    .weights
                    .layer(layer)
                    .and_then(|weights| weights.ple.as_ref())
                    .ok_or_else(|| invalid("PLE weights missing"))?;
                let desc = crate::gpu_forward::ple_desc(&self.bundle.weights, ple_weights)?;
                let rows_f16 = crate::gpu_forward::qwen4_ple_rows_f16(
                    &*self.gpu,
                    tile.routing.bf16_streams,
                    geo.dims.hc_count,
                    geo.hidden,
                    &desc,
                    n,
                );
                let len = n * geo.ple_heads * geo.ple_width;
                let staged = view(&self.forward.scratch.ple_staged, 0, len);
                let rows = view(&self.forward.scratch.ple_rows, 0, len);
                self.ple.stage_tile(
                    owner,
                    &mut self.forward.host_ple,
                    &mut *self.gpu,
                    w,
                    ti,
                    &staged,
                    &rows,
                    rows_f16,
                    geo.ple_heads,
                    geo.ple_width,
                )?;
            }
        }
        with_tile_frame(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            geo,
            tile,
            layer,
            |gpu, frame| {
                let env = frame.env;
                let mut steps: Vec<Step<'_>> = Vec::with_capacity(8);
                if geo.ple_layer == Some(layer) {
                    let ple_weights = frame
                        .layer
                        .ple
                        .as_ref()
                        .ok_or_else(|| invalid("PLE weights missing"))?;
                    let (step, _rows_f16) =
                        env.ple_step(gpu, ple_weights, geo.ple_heads, geo.ple_width)?;
                    steps.push(step);
                }
                env.push_mixer_prefix(&mut steps, layer, frame.layer, frame.description)?;
                if routes.is_none() {
                    match &frame.description.ffn {
                        Qwen4FfnBinding::Moe(moe) => {
                            env.moe_steps(&mut steps, frame.ctx, *moe, frame.scratch_desc)?
                        }
                        Qwen4FfnBinding::Dense(dense) => steps.push(env.dense_step(dense)),
                    }
                    steps.push(env.mlp_write_step(frame.description, &env.scratch.ffn_output));
                }
                validate_steps(&*gpu, &steps).map_err(dsp_err("preflight"))?;
                let mut qsa = None;
                for step in &steps {
                    if let Step::IndexedAttention(op) = step {
                        qsa = Some(op.next_lengths().map_err(dsp_err("derive QSA state"))?);
                    }
                }
                execute_validated_steps(gpu, frame.ctx, &steps)
                    .map_err(dsp_err("layer program"))?;
                if let Some(routes) = routes {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a host layer's FFN is not MoE"));
                    };
                    let sealed = env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                    let call = dsp::PreparedChunkCall::new(
                        &*gpu,
                        &sealed,
                        tile.window_row,
                        tile.position,
                        window_route,
                    )
                    .map_err(dsp_err("admission"))?;
                    dsp::prepare_routes(gpu, &call, routes)
                        .map_err(dsp_err("route preparation"))?;
                }
                Ok(qsa)
            },
        )
    }

    /// Tile `ti`'s shared expert and closing HC write of a host layer.
    fn tile_tail(
        &mut self,
        w: &WinCtx<'_>,
        ti: usize,
        layer: usize,
        routes: &dsp::PromptRouteStorage,
    ) -> R<()> {
        let geo = self.geo;
        let window_route = self.window_route(w);
        let tile = &w.tiles[ti];
        self.gpu
            .set_router_row_base(if tile.rows > 1 { tile.position } else { 0 });
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, tile.rows);
        let result = with_tile_frame(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            geo,
            tile,
            layer,
            |gpu, frame| {
                let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                    return Err(invalid("a host layer's FFN is not MoE"));
                };
                let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                let call = dsp::PreparedChunkCall::new(
                    &*gpu,
                    &sealed,
                    tile.window_row,
                    tile.position,
                    window_route,
                )
                .map_err(dsp_err("admission"))?;
                let mut hook = ForwardScopeHook {
                    forward: &*self.forward,
                    tiles: w.tiles,
                    open: Vec::new(),
                };
                dsp::finish_shared(gpu, routes, &call, &mut hook)
                    .map_err(dsp_err("shared expert"))?;
                let mixed = f32_view(
                    routes.combined(),
                    tile.window_row * geo.hidden,
                    tile.rows * geo.hidden,
                );
                let step = frame.env.mlp_write_step(frame.description, &mixed);
                let steps = [step];
                validate_steps(&*gpu, &steps).map_err(dsp_err("preflight"))?;
                execute_validated_steps(gpu, frame.ctx, &steps).map_err(dsp_err("hyper write"))
            },
        );
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        result
    }

    /// Tile `ti`'s embedding into its rows of the HC-stream arena.
    fn embed_tile(&mut self, w: &WinCtx<'_>, ti: usize) -> R<()> {
        let geo = self.geo;
        let tile = &w.tiles[ti];
        let n = tile.rows;
        for (index, token) in w.tile_tokens(ti).iter().copied().enumerate() {
            self.forward.host_token_bytes[index * 4..index * 4 + 4]
                .copy_from_slice(&(token as i32).to_ne_bytes());
        }
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, n);
        let result = (|| -> R<()> {
            let scratch = &self.forward.scratch;
            self.gpu.memcpy_htod_auto(
                &scratch.token_ids.buf,
                &self.forward.host_token_bytes[..n * std::mem::size_of::<i32>()],
            )?;
            let ids = view(&scratch.token_ids, 0, n * std::mem::size_of::<i32>());
            let embedding_rot = view(&scratch.embedding_rot, 0, n * geo.hidden);
            let embeddings = view(&scratch.embeddings, 0, n * geo.hidden);
            let table = self
                .bundle
                .weights
                .resident(&self.bundle.weights.root.embedding)?;
            execute_embedding(
                &mut *self.gpu,
                &EmbeddingOp {
                    table,
                    rotated: &embedding_rot,
                    token_ids: &ids,
                    output: &embeddings,
                    rows: n,
                    dim: geo.hidden,
                },
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("Qwen4 embedding: {error:?}"))
            })?;
            let tile_streams = f32_view(&self.rt.streams, tile.stream_offset, n * geo.wide);
            if tile.routing.bf16_streams {
                self.gpu.hc_streams_init_from_embed_batched_bf16(
                    &embeddings,
                    &tile_streams,
                    geo.hidden as i32,
                    geo.config.hc_count as i32,
                    n as i32,
                )?;
            } else {
                self.gpu.hc_streams_init_from_embed_batched(
                    &embeddings,
                    &tile_streams,
                    geo.hidden as i32,
                    geo.config.hc_count as i32,
                    n as i32,
                )?;
            }
            Ok(())
        })();
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        self.forward.slowlink_note_ids_rewritten();
        result
    }

    /// The layer `layer`'s state at the cut tile `tile` ends at.
    fn cut_if(&mut self, txn: &mut PromptTxn, tile: &TilePlan, layer: usize) -> R<()> {
        if let Some(cut) = tile.ends_at_cut {
            self.bundle
                .prompt_txn_layer_cut(&mut *self.gpu, txn, cut, layer)
                .map_err(bundle_err("layer cut"))?;
        }
        Ok(())
    }

    /// What tile `ti` returns once its last layer ran: the wide-hidden capture
    /// sink and, for the last tile of the request, the head.
    fn tile_output(&mut self, w: &WinCtx<'_>, ti: usize) -> R<()> {
        let geo = self.geo;
        let tile = &w.tiles[ti];
        let n = tile.rows;
        let capture = matches!(self.output, PromptOutput::Capture { .. });
        let wants_head = w.last_window && tile.last_in_window && self.output.head().is_some();
        if !capture && !wants_head {
            return Ok(());
        }
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, n);
        let scoped = (|| -> R<Option<GpuTensor>> {
            let tile_streams = f32_view(&self.rt.streams, tile.stream_offset, n * geo.wide);
            let bf16 = tile.routing.bf16_streams;
            let wide = if capture {
                let target = self
                    .bundle
                    .spec_hidden
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 spec hidden is not allocated"))?;
                if target.numel() < n * geo.wide {
                    return Err(invalid(
                        "the wide-hidden capture buffer is narrower than the tile",
                    ));
                }
                let destination = f32_view(target, 0, n * geo.wide);
                let source = view(&tile_streams, 0, n * geo.wide);
                if bf16 {
                    rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                        &mut *self.gpu,
                        &source,
                        &destination,
                        n * geo.wide,
                    )?;
                } else {
                    self.gpu
                        .copy_d2d(&source, &destination, source.byte_size())?;
                }
                Some(destination)
            } else {
                None
            };
            if wants_head {
                let (logits, top1) = self
                    .output
                    .head()
                    .ok_or_else(|| invalid("slow-link head disappeared"))?;
                let scratch = &self.forward.scratch;
                let final_hyper = crate::gpu_forward::hyper_read_desc(
                    &self.bundle.weights,
                    &self.bundle.weights.root.final_hyper,
                )?;
                let lm_head = crate::gpu_forward::dense_ref(
                    &self.bundle.weights,
                    &self.bundle.weights.root.lm_head,
                )?;
                let router_logits =
                    matrix_view(&scratch.router_logits, n, router_columns(geo.dims))?;
                let scratch_desc = layer_scratch(scratch, &router_logits);
                execute_final_hyper(
                    &mut *self.gpu,
                    geo.dims,
                    &final_hyper,
                    &tile_streams,
                    &scratch_desc,
                    n,
                    bf16,
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 final hyper: {error:?}"))
                })?;
                let final_hidden = view(&scratch.hc_mixed, 0, n * geo.hidden);
                let head_rotation = view(&scratch.rotation, 0, n * geo.hidden);
                execute_lm_head(
                    &mut *self.gpu,
                    &lm_head,
                    &final_hidden,
                    logits,
                    n,
                    1,
                    Some(&head_rotation),
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 LM head: {error:?}"))
                })?;
                if let Some(top1) = top1 {
                    argmax_f32(
                        &mut *self.gpu,
                        &ArgmaxF32 {
                            logits,
                            indices: top1,
                            rows: 1,
                            vocab: geo.config.vocab_size,
                        },
                    )?;
                }
            }
            Ok(wide)
        })();
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        let wide = scoped?;
        if let (Some(wide), PromptOutput::Capture { sink, .. }) = (wide, &mut *self.output) {
            sink(
                &mut *self.gpu,
                &mut *self.bundle,
                PromptTile {
                    start: tile.position,
                    rows: n,
                    wide: &wide,
                    last: tile.last_in_window,
                },
            )
            .map_err(|error| invalid(format!("slow-link capture sink: {error}")))?;
        }
        Ok(())
    }
}

/// The band's device-resident routed-expert layers, ascending: the layers
/// [`plan_runtime`] counted as resident (an expert runtime that is not
/// host-mapped), read from the actual per-layer mappings so any placement
/// shape (prefix, suffix or arbitrary set) reports exactly what ran.
fn resident_layer_set(forward: &Qwen4GpuForward) -> Vec<usize> {
    forward
        .band_layers()
        .filter(|&layer| {
            forward
                .moe
                .get(layer)
                .and_then(Option::as_ref)
                .is_some_and(|runtime| !runtime.s1_host_mapped())
        })
        .collect()
}

// ── entry ────────────────────────────────────────────────────────────

/// Run `tokens` (the request's rows after its entry position) layer-major when
/// the request qualifies. `Ok(false)` is the incumbent, decided BEFORE any
/// mutation or S1 allocation for this request; after the first window opened,
/// every outcome is `Ok(true)` or `Err` with the state restored (the incumbent
/// is never rerun on a mixed-position state).
pub(crate) fn try_run_prompt(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    tokens: &[u32],
    mut output: PromptOutput<'_>,
) -> R<bool> {
    let mode = SlowlinkMode::from_env();
    let selection = match select(forward, bundle, gpu, tokens, &output, mode) {
        Ok(selection) => selection,
        Err(reason) => {
            if forward.slowlink.sp_knob() && tracing::enabled!(tracing::Level::DEBUG) {
                tracing::debug!(
                    rows = tokens.len(),
                    reason = %reason,
                    "qwen4 SP request: not run layer-major, the incumbent runs"
                );
            }
            bundle.slowlink_note_bypass(reason);
            return Ok(false);
        }
    };
    let Selection {
        plan,
        profile,
        geo,
        sp_verdicts,
    } = selection;
    // (The head's entry `pending` hidden row is saved per window, right before
    // each window's transaction begins: see `Run::window`.)
    let ple_retained = forward.slowlink_begin(gpu)?;
    let mut runtime = forward
        .slowlink
        .runtime
        .take()
        .ok_or_else(|| invalid("slow-link runtime disappeared after selection"))?;
    if let Some(sp) = runtime.sp.as_mut() {
        sp.begin_request();
    }
    if let Some(touched) = runtime.touched.as_mut() {
        touched.begin_request();
    }
    let (resident_layers, host_layers, tile_scratch_bytes) = (
        runtime.resident_layers,
        runtime.host_layers,
        runtime.tile_scratch_bytes,
    );
    let mut reports: Vec<WindowReport> = Vec::with_capacity(plan.windows.len());
    let outcome = {
        let mut run = Run {
            forward: &mut *forward,
            bundle: &mut *bundle,
            gpu: &mut *gpu,
            rt: &mut runtime,
            geo: &geo,
            profile: &profile,
            plan: &plan,
            tokens,
            output: &mut output,
            ple: PleCursor {
                retained: ple_retained,
                histories: Vec::new(),
            },
            sp_verdicts: &sp_verdicts,
        };
        let mut outcome = Ok(());
        for window in &plan.windows {
            let last = window.index + 1 == plan.windows.len();
            match run.window(window, last) {
                Ok(report) => reports.push(report),
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        outcome
    };
    forward.slowlink.runtime = Some(runtime);
    // The route each window ran (the hybrid SP's windows run the F16 route on a
    // symmetric-IU4 profile): the receipt's per-window route and tile counts follow it.
    let window_routes = sp_window_routes(
        profile.geometry.route,
        forward.slowlink.sp.admitted.as_ref(),
        &sp_verdicts,
        plan.windows.len(),
    );
    // The `sp` receipt object exists only with the knob on: with it off nothing
    // is built, so the request allocates and reports exactly what it did.
    let sp_receipt = forward
        .slowlink
        .sp_knob()
        .then(|| {
            let sp_reports: Vec<(usize, SpWindowReport)> = forward
                .slowlink
                .runtime
                .as_ref()
                .map(SlowlinkRuntime::sp_reports)
                .unwrap_or_default();
            let sp_windows: Vec<SpWindowEntry> = plan
                .windows
                .iter()
                .map(|window| SpWindowEntry {
                    rows: window.rows,
                    route: window_routes[window.index],
                    verdict: sp_verdicts.get(window.index).copied().unwrap_or(Ok(())),
                    report: sp_reports
                        .iter()
                        .find(|(index, _)| *index == window.index)
                        .map(|(_, report)| *report)
                        .unwrap_or_default(),
                })
                .collect();
            if tracing::enabled!(tracing::Level::DEBUG) {
                for (window, entry) in plan.windows.iter().zip(&sp_windows) {
                    let engaged = entry.report.layers > 0;
                    tracing::debug!(
                        window = window.index,
                        rows = entry.rows,
                        route = entry.route.label(),
                        engaged,
                        layers = entry.report.layers,
                        streamed_experts = entry.report.streamed_experts,
                        bypass = entry
                            .verdict
                            .err()
                            .map(|bypass| bypass.to_string())
                            .unwrap_or_default(),
                        "qwen4 SP window: {}",
                        if engaged { "engaged" } else { "not engaged" },
                    );
                }
            }
            forward.slowlink.sp_receipt(&sp_windows)
        })
        .flatten();
    let resident_layer_set = resident_layer_set(forward);
    // ONE `hipMemGetInfo` (total - free) sample after the request, outside every
    // overlap loop: the device's used bytes at that moment, not a continuous
    // peak. `None` (a null in the receipt) when the query fails.
    let post_request_vram_used = gpu
        .device_mem_info()
        .ok()
        .map(|(free, total)| total.saturating_sub(free) as u64);
    let moe_iu4 = moe_iu4_receipt(&MoeIu4Facts {
        route: profile.geometry.route,
        requested: gpu.qwen4_moe_sym_iu4_requested(),
        // The kernel family the windows actually ran: only the IU4 window route has one.
        descriptor: gpu
            .qwen4_moe_sym_kernel()
            .filter(|_| profile.geometry.route == dsp::WindowRoute::AggregateSymIu4),
        source_symmetric: profile.sym_iu4_source_symmetric,
        final_verified: forward.sym_iu4_coverage(),
        a4_sidecar_bytes: profile.census.a4_sidecars,
        window_rows: profile.window_rows,
        tiles: plan.windows.iter().zip(&window_routes).fold(
            [0u64; 5],
            |mut total, (window, &route)| {
                let counts =
                    moe_iu4_tile_counts(route, plan.window_tiles(window), geo.few_row_limit);
                total.iter_mut().zip(counts).for_each(|(sum, n)| *sum += n);
                total
            },
        ),
    });
    let receipt = |status: String| {
        let mut value = layer_major_receipt(&ReceiptInputs {
            mode,
            plan: &plan,
            resident_layers,
            resident_layer_set: &resident_layer_set,
            host_layers,
            tile_scratch_bytes,
            reports: &reports,
            peak_vram_bytes: post_request_vram_used,
            status,
            moe_iu4: moe_iu4.clone(),
        });
        if let (Some(sp), Value::Object(map)) = (&sp_receipt, &mut value) {
            map.insert("sp".to_string(), sp.clone());
        }
        value
    };
    match outcome {
        Ok(()) => {
            bundle.slowlink_set_receipt(receipt("ok".to_string()));
            Ok(true)
        }
        Err(error) => {
            bundle.slowlink_set_receipt(receipt(format!("aborted: {error}")));
            Err(error)
        }
    }
}

// ── production-primitive self-test ───────────────────────────────────

/// Compute-stream delay (in repeats of the transport's delay kernel) that
/// keeps each page's reader queued while the page refills.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
const SELFTEST_DELAY_REPEATS: usize = 128;
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
const SELFTEST_SEED: u64 = 0x5eed_51ee_d5b0_0001;

/// `{"name","status":"pass","mismatches":0,"detail"}` of one dispatch
/// primitive that observed every check; a failed check is an `Err` carrying
/// the whole raw report (no mismatch count is invented).
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
fn primitive_case_json(name: &str, report: &dsp::selftest::PrimitiveReport) -> R<Value> {
    let raw = report.to_json();
    if !report.ok {
        return Err(invalid(format!("slow-link primitive {name} failed: {raw}")));
    }
    let detail: Value = serde_json::from_str(&raw).map_err(|error| {
        invalid(format!(
            "slow-link primitive {name}: unreadable report: {error}"
        ))
    })?;
    Ok(json!({
        "name": name,
        "status": "pass",
        "mismatches": 0u64,
        "detail": detail,
    }))
}

/// The transport primitives, measured by the ring's own diagnostic driven
/// through the same prime / enter / require / acquire / retire / pump path the
/// prompt run uses, on as many real host layers as two full pool sweeps need.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
fn transport_case(rt: &mut SlowlinkRuntime, gpu: &mut Gpu, name: &str) -> R<Value> {
    let pages = rt
        .ring
        .spec()
        .reserve()
        .ok_or_else(|| invalid("slow-link page pool reserve overflows"))?
        .pages;
    let (mut layers, mut groups) = (0usize, 0usize);
    for (layer, host) in rt.host_layer.iter().enumerate() {
        if !host {
            continue;
        }
        layers += 1;
        groups += rt.schedule.layer_groups(layer).map_or(0, <[_]>::len);
        if groups >= 2 * pages {
            break;
        }
    }
    if groups < 2 * pages {
        return Err(invalid(format!(
            "the transport check needs {} expert groups to reuse every page twice, the host layers hold {groups}",
            2 * pages
        )));
    }
    let report = crate::gpu_forward::panel_stage::primitive_selftest(
        gpu,
        &mut rt.ring,
        &mut rt.schedule,
        layers,
        SELFTEST_DELAY_REPEATS,
    )?;
    let pass = match name {
        "payload_table_identity" => {
            report.payload_bytes_compared > 0
                && report.table_entries_checked > 0
                && report.pad_bytes_checked > 0
        }
        "repeated_epoch_same_address" => {
            report.repeated_address_epochs > 0 && report.min_epochs_per_address >= 2
        }
        "delayed_reader_reuse" => {
            report.delayed_reader_reuses > 0 && report.stale_ticket_rejections > 0
        }
        other => return Err(invalid(format!("{other:?} is not a transport primitive"))),
    };
    let detail = json!({
        "layers": report.layers,
        "groups_checked": report.groups_checked,
        "payload_bytes_compared": report.payload_bytes_compared,
        "table_entries_checked": report.table_entries_checked,
        "pad_bytes_checked": report.pad_bytes_checked,
        "delayed_reader_reuses": report.delayed_reader_reuses,
        "stale_ticket_rejections": report.stale_ticket_rejections,
        "distinct_addresses": report.distinct_addresses,
        "repeated_address_epochs": report.repeated_address_epochs,
        "min_epochs_per_address": report.min_epochs_per_address,
        "max_epochs_per_address": report.max_epochs_per_address,
        "submitted_payload_bytes": report.submitted_payload_bytes,
    });
    // Only observed counters pass. A case whose required counters were not
    // observed is an error carrying the whole raw report: the check never
    // invents a mismatch count.
    if !pass {
        return Err(invalid(format!(
            "slow-link primitive {name} did not observe its required counters: {detail}"
        )));
    }
    Ok(json!({
        "name": name,
        "status": "pass",
        "mismatches": 0u64,
        "detail": detail,
    }))
}

/// A dispatch primitive on a real sealed tile call and a real staged page
/// binding of the first host layer.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
fn dispatch_case(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    rt: &mut SlowlinkRuntime,
    profile: &SlowlinkProfile,
    name: &str,
) -> R<Value> {
    let layer = rt
        .host_layer
        .iter()
        .position(|host| *host)
        .ok_or_else(|| invalid("no host layer to check against"))?;
    let few_row_limit = {
        let (scopes, _) = forward.enter_tile_scopes(gpu, profile.tile_rows);
        let limit = gpu.few_row_limit();
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        limit
    };
    let geo = RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts)?;
    let rows = profile.tile_rows;
    let routing = tile_routing(forward, gpu, rows);
    let stub = TilePlan {
        index: 0,
        window: 0,
        prompt_row: 0,
        window_row: 0,
        stream_offset: 0,
        rows,
        position: bundle.state.position,
        routing,
        ends_at_cut: None,
        last_in_window: true,
    };
    let group = rt
        .schedule
        .layer_groups(layer)
        .and_then(|groups| groups.first())
        .cloned()
        .ok_or_else(|| invalid("the first host layer has no scheduled group"))?;
    rt.ring.prime(&*gpu, &mut rt.schedule)?;
    rt.ring.enter_layer(&*gpu, &mut rt.schedule, layer)?;
    let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
    let result = with_tile_frame(
        forward,
        bundle,
        gpu,
        &rt.streams,
        &geo,
        &stub,
        layer,
        |gpu, frame| {
            let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                return Err(invalid("a host layer's FFN is not MoE"));
            };
            let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
            let call = dsp::PreparedChunkCall::new(
                &*gpu,
                &sealed,
                0,
                stub.position,
                profile.geometry.route,
            )
            .map_err(dsp_err("admission"))?;
            if call.window_route() != profile.geometry.route {
                return Err(invalid(format!(
                    "the {name} check sealed a {:?} call but the admitted profile is {:?}",
                    call.window_route(),
                    profile.geometry.route
                )));
            }
            // On an aggregate route these checks run on an actual K2 call (the admitted tile
            // rows): the F16 WMMA arm on `Aggregate`, the IU4 arm on `AggregateSymIu4`; an
            // F32-tail call never stands in.
            let k2_kind = match profile.geometry.route {
                dsp::WindowRoute::Aggregate => Some(dsp::TileKind::AggregateWmma),
                dsp::WindowRoute::AggregateSymIu4 => Some(dsp::TileKind::AggregateIu4),
                dsp::WindowRoute::SlabIu4 => None,
            };
            if let Some(expected) = k2_kind.filter(|expected| call.tile_kind() != *expected) {
                return Err(invalid(format!(
                    "the {name} check anchors on an actual {expected:?} call; the {rows}-row \
                     tile sealed as {:?}",
                    call.tile_kind()
                )));
            }
            let mut ring = HipRing {
                ring: &mut rt.ring,
                schedule: &mut rt.schedule,
            };
            let ticket = ring.require(&*gpu, layer, group.group)?;
            let mut out: Option<R<Value>> = None;
            ring.with_lease(gpu, ticket, &mut |gpu, parts| {
                let parts =
                    parts.ok_or_else(|| invalid("slow-link page lease carries no binding"))?;
                let (Some(&lo), Some(&last)) = (group.experts.first(), group.experts.last()) else {
                    return Err(invalid("an empty expert group"));
                };
                let binding = dsp::ExpertPanelBinding {
                    layer,
                    epoch: parts.epoch,
                    expert_lo: lo,
                    expert_hi: last + 1,
                    gate_up_table: parts.gate_up_table,
                    down_table: parts.down_table,
                    codec: dsp::PanelCodec::Mq4,
                    source: dsp::BindingSource::Staged {
                        slot: parts.ticket.slot as u32,
                    },
                };
                let limits = dsp::PanelLimits::S1;
                let report = match name {
                    "within_expert_panel_canary" => dsp::selftest::within_expert_panel_canary(
                        gpu,
                        &call,
                        binding,
                        limits,
                        SELFTEST_SEED,
                    ),
                    "k1_tail_cohort" => {
                        dsp::selftest::k1_tail_cohort(gpu, &call, binding, limits, SELFTEST_SEED)
                    }
                    "zero_row_panel" => {
                        dsp::selftest::zero_row_panel(gpu, &call, binding, limits, SELFTEST_SEED)
                    }
                    other => return Err(invalid(format!("{other:?} is not a dispatch primitive"))),
                }
                .map_err(dsp_err("primitive"))?;
                out = Some(primitive_case_json(name, &report));
                Ok(())
            })?;
            ring.retire(&*gpu, ticket)?;
            out.ok_or_else(|| invalid("slow-link primitive produced no report"))?
        },
    );
    Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
    result
}

/// The one-row decode tail beside a K2 tile, on a device-resident layer: the
/// dispatch crate's `decode_tail_cohort` over the incumbent's real K2 prefill
/// call (64 rows) and real decode call (1 row at window row 64), each sealed
/// the way the model seals it, against the layer's resident all-expert binding
/// (the router may select any expert, a staged 32-expert page cannot hold them).
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
fn decode_tail_case(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    rt: &SlowlinkRuntime,
    profile: &SlowlinkProfile,
) -> R<Value> {
    let few_row_limit = {
        let (scopes, _) = forward.enter_tile_scopes(gpu, profile.tile_rows);
        let limit = gpu.few_row_limit();
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        limit
    };
    let geo = RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts)?;
    let layer = resident_check_layer(forward, &geo, "decode tail")?;
    // The decode call sits at window row `decode_row0`, beside the dispatch
    // check's own synthetic K2 tile. The anchor is any actual K2 prefill call
    // of the layer: the slab route's 64-row K2 tile, the aggregate route's
    // WMMA tile of the admitted tile rows (a tile below the WMMA threshold is
    // an F32 tail and refused by the K2 check below).
    let decode_row0 = dsp::selftest::DECODE_TAIL_ANCHOR_ROWS;
    let anchor_rows = match profile.geometry.route {
        dsp::WindowRoute::SlabIu4 => decode_row0,
        dsp::WindowRoute::Aggregate | dsp::WindowRoute::AggregateSymIu4 => profile.tile_rows,
    };
    let position = bundle.state.position;
    let anchor_tile = TilePlan {
        index: 0,
        window: 0,
        prompt_row: 0,
        window_row: 0,
        stream_offset: 0,
        rows: anchor_rows,
        position,
        routing: tile_routing(forward, gpu, anchor_rows),
        ends_at_cut: None,
        last_in_window: false,
    };
    let decode_tile = TilePlan {
        index: 1,
        window: 0,
        prompt_row: decode_row0,
        window_row: decode_row0,
        stream_offset: anchor_rows * geo.wide,
        rows: 1,
        position: position + decode_row0,
        routing: tile_routing(forward, gpu, 1),
        ends_at_cut: None,
        last_in_window: true,
    };
    let tiles = [&anchor_tile, &decode_tile];
    with_tile_frames(
        forward,
        bundle,
        gpu,
        &rt.streams,
        &geo,
        &tiles,
        layer,
        |gpu, frames| {
            // Each call is sealed and admitted under its own tile's scope.
            let mut sealed = Vec::with_capacity(frames.len());
            for (tile, frame) in tiles.iter().zip(frames) {
                let (scopes, _) = forward.enter_tile_scopes(gpu, tile.rows);
                let one = (|| -> R<_> {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a resident layer's FFN is not MoE"));
                    };
                    Ok(frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?)
                })();
                Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                sealed.push(one?);
            }
            let mut calls = Vec::with_capacity(sealed.len());
            for (tile, sealed) in tiles.iter().zip(&sealed) {
                let (scopes, _) = forward.enter_tile_scopes(gpu, tile.rows);
                let call = dsp::PreparedChunkCall::new(
                    &*gpu,
                    sealed,
                    tile.window_row,
                    tile.position,
                    profile.geometry.route,
                );
                Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                calls.push(call.map_err(dsp_err("admission"))?);
            }
            if !calls[0].is_k2() || calls[0].is_decode() || !calls[1].is_decode() {
                return Err(invalid(format!(
                    "the decode tail check expects a K2 anchor and a decode tile; the seals gave k2={} decode={} and decode={}",
                    calls[0].is_k2(),
                    calls[0].is_decode(),
                    calls[1].is_decode()
                )));
            }
            let Qwen4FfnBinding::Moe(moe) = &frames[0].description.ffn else {
                return Err(invalid("a resident layer's FFN is not MoE"));
            };
            let binding = dsp::ExpertPanelBinding::resident(
                layer,
                geo.num_experts,
                moe.expert_gate_up_ptrs,
                moe.expert_down_ptrs,
            );
            let (scopes, _) = forward.enter_tile_scopes(gpu, anchor_rows);
            let report = dsp::selftest::decode_tail_cohort(
                gpu,
                &calls[0],
                &calls[1],
                binding,
                dsp::PanelLimits::S1,
                SELFTEST_SEED,
            );
            Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
            primitive_case_json("decode_tail_cohort", &report.map_err(dsp_err("primitive"))?)
        },
    )
}

/// The first device-resident qualified expert layer of the band, the layer a
/// resident all-expert binding check runs on (`what` names the check).
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
fn resident_check_layer(forward: &Qwen4GpuForward, geo: &RunGeo, what: &str) -> R<usize> {
    geo.band
        .clone()
        .find(|&layer| {
            forward
                .moe
                .get(layer)
                .and_then(Option::as_ref)
                .is_some_and(|runtime| {
                    // A tier layer's own pointer tables are an all-expert binding too (hot
                    // experts in VRAM, cold ones read through their mapped alias).
                    (!runtime.s1_host_mapped() || runtime.s1_is_tier())
                        && runtime.s1_layer_check().is_ok()
                })
        })
        .ok_or_else(|| {
            invalid(format!(
                "the {what} check needs a device-resident qualified expert layer; this placement has none"
            ))
        })
}

/// Rows of the two aggregate F32-tail calls the cohort check runs: 3 takes the
/// incumbent's few-row indexed-down arm, 9 (one above the few-row limit of 8)
/// its F32 grouped-down arm.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
const F32_TAIL_CASE_ROWS: [usize; 2] = [3, 9];

/// The aggregate route's F32-tail cohort: the dispatch crate's `k1_tail_cohort`
/// over the incumbent's REAL row-0 F32-tail calls of [`F32_TAIL_CASE_ROWS`]
/// rows, each sealed and admitted under its own tile's actual routing scope the
/// way the model seals it, on a device-resident qualified layer against its
/// resident all-expert binding (the router may select any expert; a staged
/// page cannot hold them all). Every call must be an F32-tail call whose
/// routing rows land in its named arm.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
fn f32_tail_case(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    rt: &SlowlinkRuntime,
    profile: &SlowlinkProfile,
) -> R<Value> {
    let few_row_limit = {
        let (scopes, _) = forward.enter_tile_scopes(gpu, profile.tile_rows);
        let limit = gpu.few_row_limit();
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        limit
    };
    let geo = RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts)?;
    let layer = resident_check_layer(forward, &geo, "F32 tail cohort")?;
    let mut details = serde_json::Map::new();
    for rows in F32_TAIL_CASE_ROWS {
        let routing = tile_routing(forward, gpu, rows);
        if tile_cohort(profile.geometry.route, routing.route_rows, few_row_limit)
            != TileCohort::F32Tail
            || (rows == F32_TAIL_CASE_ROWS[0]) != (routing.route_rows <= few_row_limit)
        {
            return Err(invalid(format!(
                "the {rows}-row F32 tail check routes {} rows against a few-row limit of \
                 {few_row_limit}: it does not reach its indexed-down / grouped-down arm",
                routing.route_rows
            )));
        }
        let tile = TilePlan {
            index: 0,
            window: 0,
            prompt_row: 0,
            window_row: 0,
            stream_offset: 0,
            rows,
            position: bundle.state.position,
            routing,
            ends_at_cut: None,
            last_in_window: true,
        };
        let case = with_tile_frame(
            forward,
            bundle,
            gpu,
            &rt.streams,
            &geo,
            &tile,
            layer,
            |gpu, frame| {
                let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
                let one = (|| -> R<Value> {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a resident layer's FFN is not MoE"));
                    };
                    let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                    let call = dsp::PreparedChunkCall::new(
                        &*gpu,
                        &sealed,
                        0,
                        tile.position,
                        profile.geometry.route,
                    )
                    .map_err(dsp_err("admission"))?;
                    if call.tile_kind() != dsp::TileKind::AggregateF32Tail {
                        return Err(invalid(format!(
                            "the {rows}-row F32 tail check sealed as {:?}, not an F32-tail call",
                            call.tile_kind()
                        )));
                    }
                    let binding = dsp::ExpertPanelBinding::resident(
                        layer,
                        geo.num_experts,
                        moe.expert_gate_up_ptrs,
                        moe.expert_down_ptrs,
                    );
                    let report = dsp::selftest::k1_tail_cohort(
                        gpu,
                        &call,
                        binding,
                        dsp::PanelLimits::S1,
                        SELFTEST_SEED,
                    )
                    .map_err(dsp_err("primitive"))?;
                    primitive_case_json("k1_tail_cohort", &report)
                })();
                Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                one
            },
        )?;
        details.insert(format!("rows_{rows}"), case["detail"].clone());
    }
    Ok(json!({
        "name": "k1_tail_cohort",
        "status": "pass",
        "mismatches": 0u64,
        "detail": Value::Object(details),
    }))
}

/// One named primitive check on the attached forward's real owners; the
/// ring is drained and idle afterwards whatever the outcome.
#[cfg(any(test, feature = "lab", feature = "reference-parity"))]
pub(crate) fn primitive_selftest(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    name: &str,
) -> R<Value> {
    let profile = forward
        .slowlink_profile()
        .copied()
        .ok_or_else(|| invalid("slow-link primitive check needs an admitted profile"))?;
    if let Some(why) = forward.slowlink_forward_blocker(&bundle.config) {
        return Err(invalid(format!(
            "slow-link primitive check cannot run: {why}"
        )));
    }
    runtime_gate(
        forward.slowlink.runtime.is_some(),
        forward.slowlink.unavailable.as_deref(),
    )
    .map_err(invalid)?;
    let mut rt = forward
        .slowlink
        .runtime
        .take()
        .ok_or_else(|| invalid("slow-link runtime disappeared"))?;
    let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
    let result = match name {
        "payload_table_identity" | "repeated_epoch_same_address" | "delayed_reader_reuse" => {
            transport_case(&mut rt, gpu, name)
        }
        "canonical_combine_exact" => dsp::selftest::canonical_combine_exact(gpu, SELFTEST_SEED)
            .map_err(dsp_err("canonical combine check"))
            .and_then(|report| primitive_case_json(name, &report)),
        "within_expert_panel_canary" | "zero_row_panel" => {
            dispatch_case(forward, &*bundle, gpu, &mut rt, &profile, name)
        }
        // The few-row cohort is checks that must ALL pass. Slab: the prefill K1
        // tail on a staged host-layer binding, and the incumbent's REAL one-row
        // decode call beside a K2 tile on a resident all-expert binding.
        // Aggregate: the REAL row-0 F32-tail calls of 3 rows (indexed-down) and
        // 9 rows (F32 grouped-down) on a resident all-expert binding, and the
        // same decode call beside an actual WMMA anchor.
        "k1_tail_cohort" => match profile.geometry.route {
            dsp::WindowRoute::SlabIu4 => {
                dispatch_case(forward, &*bundle, gpu, &mut rt, &profile, name).and_then(|prefill| {
                    decode_tail_case(forward, &*bundle, gpu, &rt, &profile).map(|decode| {
                        json!({
                            "name": name,
                            "status": "pass",
                            "mismatches": 0u64,
                            "detail": {
                                "prefill_k1": prefill["detail"],
                                "decode_tail": decode["detail"],
                            },
                        })
                    })
                })
            }
            dsp::WindowRoute::Aggregate | dsp::WindowRoute::AggregateSymIu4 => {
                f32_tail_case(forward, &*bundle, gpu, &rt, &profile).and_then(|f32_tail| {
                    decode_tail_case(forward, &*bundle, gpu, &rt, &profile).map(|decode| {
                        json!({
                            "name": name,
                            "status": "pass",
                            "mismatches": 0u64,
                            "detail": {
                                "f32_tail": f32_tail["detail"],
                                "decode_tail": decode["detail"],
                            },
                        })
                    })
                })
            }
        },
        other => Err(invalid(format!("unknown slow-link primitive {other:?}"))),
    };
    let drained = rt.ring.drain(&*gpu);
    forward.slowlink.runtime = Some(rt);
    forward.prefill_scope = prior_scope;
    let report = result?;
    drained?;
    Ok(report)
}

#[cfg(test)]
mod slowlink_prompt {
    use super::*;
    use crate::slowlink_admission::{
        admit, admit_at_tile, AdmissionInputs, AdmissionPurpose, ModelGeometry,
        S1_AGGREGATE_TAIL_ROWS, S1_K1_TAIL_ROWS, S1_PRIMITIVE_TILE_ROWS, S1_PRODUCT_TILE_ROWS,
        S1_RUNTIME_OVERHEAD_BYTES, S1_TILE_ROWS, S1_WINDOW_ROWS,
    };
    use rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS;

    const MIB: u64 = 1 << 20;
    const GATE_UP_LAYER: u64 = 512 * 1_740_800;
    const DOWN_LAYER: u64 = 512 * 870_400;

    /// The HISTORICAL counterfactual profile: the maximum legal 1536-row tile
    /// ([`S1_TILE_ROWS`]) on a Flash-Next MQ4 load with ample memory. The
    /// numeric class fixtures below (ten 1536-row tiles, 15,360-row windows,
    /// cuts at 1536/1537) declare THIS tile as their intent; it is admitted
    /// through [`admit_at_tile`] (arena and census recomputed at that tile),
    /// never by editing `profile.tile_rows` of another profile. The ACTUAL
    /// shipped tiles are [`product_profile_on`] and [`primitive_profile_on`].
    pub(super) fn profile() -> SlowlinkProfile {
        profile_with_budgets(0, 0)
    }

    /// [`profile`] with the Hold 1 diagnostic budget `diagnostic_bytes` and the
    /// runtime overhead `base_runtime_reserved_bytes` the base reserve already holds.
    fn profile_with_budgets(
        diagnostic_bytes: u64,
        base_runtime_reserved_bytes: u64,
    ) -> SlowlinkProfile {
        profile_on(
            dsp::WindowRoute::SlabIu4,
            diagnostic_bytes,
            base_runtime_reserved_bytes,
        )
    }

    /// The fixture's admission inputs on the artifact route `route`: the
    /// geometry names the route before the census is computed, so every
    /// route-dependent byte is that route's own layout.
    fn fixture_inputs(
        route: dsp::WindowRoute,
        purpose: AdmissionPurpose,
        diagnostic_bytes: u64,
        base_runtime_reserved_bytes: u64,
    ) -> AdmissionInputs {
        let geometry =
            ModelGeometry::from_config(&crate::config::compact_test_config(), route).unwrap();
        AdmissionInputs {
            geometry,
            memory: crate::slowlink_admission::AdmissionMemory {
                pre_init_free_bytes: None,
                free_bytes: 40 << 30,
            },
            non_expert: 5_364_000_000,
            layer_expert_bytes: GATE_UP_LAYER + DOWN_LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: 48,
            base_reserve: 5261 * MIB,
            baseline_layers: 10,
            fixed_state_bytes: 64 * MIB,
            pending_save_bytes: 4 * 2560 * 4,
            diagnostic_bytes,
            base_runtime_reserved_bytes,
            granule: crate::slowlink_admission::GranuleInputs::default(),
            checkpoint_stride_rows: 8192,
            window_rows: S1_WINDOW_ROWS,
            link_gbps: 7.09,
            purpose,
            sym_iu4_source_symmetric: None,
        }
    }

    /// The `moe_iu4` object of a receipt whose window ran the slab route with no request.
    pub(super) fn no_moe_iu4() -> Value {
        moe_iu4_receipt(&MoeIu4Facts {
            route: dsp::WindowRoute::SlabIu4,
            requested: false,
            descriptor: None,
            source_symmetric: None,
            final_verified: (0, 0),
            a4_sidecar_bytes: 0,
            window_rows: S1_WINDOW_ROWS,
            tiles: [0; 5],
        })
    }

    /// [`profile_with_budgets`] admitted FRESH on the artifact route `route` at
    /// the explicit 1536 counterfactual tile.
    pub(super) fn profile_on(
        route: dsp::WindowRoute,
        diagnostic_bytes: u64,
        base_runtime_reserved_bytes: u64,
    ) -> SlowlinkProfile {
        admit_at_tile(
            &fixture_inputs(
                route,
                AdmissionPurpose::Product,
                diagnostic_bytes,
                base_runtime_reserved_bytes,
            ),
            S1_TILE_ROWS,
        )
        .expect("a 40 GiB load admits S1 at the 1536 counterfactual tile")
    }

    /// The ACTUAL product profile (`AdmissionPurpose::Product`, the 768-row
    /// tile) through the one purpose-driven [`admit`].
    pub(super) fn product_profile_on(route: dsp::WindowRoute) -> SlowlinkProfile {
        let profile = admit(&fixture_inputs(route, AdmissionPurpose::Product, 0, 0))
            .expect("a 40 GiB load admits the product S1 profile");
        assert_eq!(profile.tile_rows, S1_PRODUCT_TILE_ROWS);
        profile
    }

    /// The product profile admitted with a window of `window_rows` rows (the load's own
    /// `S1_WINDOW_ROWS.min(max_seq)`: a small `max_seq` shrinks the window and its route region).
    pub(super) fn product_profile_window(
        route: dsp::WindowRoute,
        window_rows: usize,
    ) -> SlowlinkProfile {
        let mut inputs = fixture_inputs(route, AdmissionPurpose::Product, 0, 0);
        inputs.window_rows = window_rows;
        admit(&inputs).expect("a 40 GiB load admits the product S1 profile")
    }

    /// The ACTUAL primitive-purpose profile (the 512-row tile).
    pub(super) fn primitive_profile_on(route: dsp::WindowRoute) -> SlowlinkProfile {
        let profile = admit(&fixture_inputs(
            route,
            AdmissionPurpose::PrimitiveSelftest,
            0,
            0,
        ))
        .expect("a 40 GiB load admits the primitive S1 profile");
        assert_eq!(profile.tile_rows, S1_PRIMITIVE_TILE_ROWS);
        profile
    }

    /// The incumbent's decisions on a gfx1100-class tile: the routing rows are
    /// the launch rows and the HC streams are BF16 from the WMMA minimum up.
    fn gfx1100_routing(rows: usize) -> TileRouting {
        TileRouting {
            route_rows: rows,
            bf16_streams: rows >= QWEN4_F16_WMMA_MIN_TOKENS,
        }
    }

    /// The canonical-routing (batch-invariant) variant: every tile below the
    /// canonical minimum routes as the canonical chunk.
    fn canonical_routing(rows: usize) -> TileRouting {
        let route_rows = rows.max(QWEN4_F16_WMMA_MIN_TOKENS);
        TileRouting {
            route_rows,
            bf16_streams: route_rows >= QWEN4_F16_WMMA_MIN_TOKENS,
        }
    }

    /// [`plan_prompt_verdict`] of a request on a profile admitted FRESH on `route`
    /// at the explicit 1536 counterfactual tile.
    pub(super) fn plan_on(
        route: dsp::WindowRoute,
        entry: usize,
        rows: usize,
        cuts: &[usize],
        routing: &dyn Fn(usize) -> TileRouting,
        few_row_limit: usize,
    ) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
        plan_for_profile(
            &profile_on(route, 0, 0),
            entry,
            rows,
            cuts,
            routing,
            few_row_limit,
        )
    }

    /// [`plan_prompt_verdict`] of a request on an explicit `profile`.
    pub(super) fn plan_for_profile(
        profile: &SlowlinkProfile,
        entry: usize,
        rows: usize,
        cuts: &[usize],
        routing: &dyn Fn(usize) -> TileRouting,
        few_row_limit: usize,
    ) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
        plan_prompt_verdict(&PromptPlanInputs {
            profile,
            entry_pos: entry,
            rows,
            cuts,
            routing,
            few_row_limit,
        })
    }

    pub(super) fn plan_with(
        entry: usize,
        rows: usize,
        cuts: &[usize],
        routing: &dyn Fn(usize) -> TileRouting,
        few_row_limit: usize,
    ) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
        plan_on(
            dsp::WindowRoute::SlabIu4,
            entry,
            rows,
            cuts,
            routing,
            few_row_limit,
        )
    }

    /// A qualified request on the aggregate route under the gfx1100 routing.
    fn plan_aggregate(entry: usize, rows: usize, cuts: &[usize]) -> Qwen4PromptPlan {
        plan_on(
            dsp::WindowRoute::Aggregate,
            entry,
            rows,
            cuts,
            &gfx1100_routing,
            8,
        )
        .unwrap()
        .expect("a qualified aggregate request plans")
    }

    /// A resumed request: the session restored 3000 rows, 100 follow.
    fn plan_resumed() -> Qwen4PromptPlan {
        plan(3000, 100, &[])
    }

    pub(super) fn plan(entry: usize, rows: usize, cuts: &[usize]) -> Qwen4PromptPlan {
        plan_with(entry, rows, cuts, &gfx1100_routing, 8)
            .unwrap()
            .expect("a qualified request plans")
    }

    /// Every tile is contiguous with the previous one across windows.
    fn assert_chronology(plan: &Qwen4PromptPlan) {
        let mut position = plan.entry_pos;
        let mut row = 0;
        for tile in &plan.tiles {
            assert_eq!(
                tile.position, position,
                "tile {} starts at the previous end",
                tile.index
            );
            assert_eq!(tile.prompt_row, row);
            assert!(tile.rows > 0 && tile.rows <= S1_TILE_ROWS);
            position += tile.rows;
            row += tile.rows;
        }
        assert_eq!(row, plan.rows);
        let mut window_start = plan.entry_pos;
        for window in &plan.windows {
            assert_eq!(window.position, window_start);
            let tiles = plan.window_tiles(window);
            assert_eq!(tiles.iter().map(|t| t.rows).sum::<usize>(), window.rows);
            assert!(window.rows <= S1_WINDOW_ROWS);
            assert!(tiles.last().unwrap().last_in_window);
            assert!(tiles[..tiles.len() - 1].iter().all(|t| !t.last_in_window));
            let mut window_row = 0;
            for tile in tiles {
                assert_eq!(tile.window_row, window_row);
                assert_eq!(tile.window, window.index);
                window_row += tile.rows;
            }
            window_start += window.rows;
        }
        assert_eq!(window_start, plan.entry_pos + plan.rows);
    }

    #[test]
    fn slowlink_prompt_tiles_are_the_incumbent_chunk_chronology() {
        let plan = plan(0, 16384, &[]);
        assert_chronology(&plan);
        assert_eq!(plan.windows.len(), 1, "16384 rows are one admitted window");
        let rows: Vec<usize> = plan.tiles.iter().map(|t| t.rows).collect();
        assert_eq!(rows, [vec![1536; 10], vec![1024]].concat());
        assert_eq!(plan.launch_cuts().last(), Some(&16384));
        assert!(plan.tiles.iter().all(|t| t.ends_at_cut.is_none()));
        assert!(plan
            .tiles
            .iter()
            .all(|t| t.routing == gfx1100_routing(t.rows)));
        assert_eq!(plan.k1_rows_max, 0);
    }

    #[test]
    fn slowlink_prompt_resumed_entry_keeps_absolute_positions() {
        let plan = plan(3000, 5000, &[]);
        assert_chronology(&plan);
        assert_eq!(plan.tiles[0].position, 3000);
        assert_eq!(plan.tiles.last().unwrap().end_position(), 8000);
        assert_eq!(plan.windows[0].position, 3000);
    }

    #[test]
    fn slowlink_prompt_boundaries_split_tiles_like_the_incumbent() {
        // The boundary cuts the plan's tile list the way the incumbent's
        // `prefill_final_from` split its chunk loop: segments between cuts,
        // each tiled by the tile size, never a tile across a cut.
        for cuts in [
            vec![1],
            vec![8],
            vec![3],
            vec![4],
            vec![1536],
            vec![1537],
            vec![8192],
            vec![3, 4],
            vec![3, 1536, 1537],
        ] {
            let plan = plan(0, 16384, &cuts);
            assert_chronology(&plan);
            for &cut in &cuts {
                assert!(
                    plan.tiles.iter().any(|t| t.end_position() == cut),
                    "cut {cut} is a tile end"
                );
                assert!(
                    plan.tiles
                        .iter()
                        .all(|t| !(t.position < cut && cut < t.end_position())),
                    "no tile straddles cut {cut}"
                );
            }
            assert_eq!(
                plan.windows.len(),
                1,
                "{cuts:?} fit one window's boundary snapshots"
            );
            let window = &plan.windows[0];
            assert_eq!(window.cuts, cuts);
            for tile in plan.window_tiles(window) {
                if let Some(index) = tile.ends_at_cut {
                    assert_eq!(window.cuts[index], tile.end_position());
                }
            }
            let ended: Vec<usize> = plan
                .window_tiles(window)
                .iter()
                .filter_map(|t| t.ends_at_cut.map(|index| window.cuts[index]))
                .collect();
            assert_eq!(
                ended, cuts,
                "every cut is staged by exactly one tile end, in order"
            );
        }
        let at_3 = plan(0, 16384, &[3]);
        assert_eq!(at_3.tiles[0].rows, 3);
        assert_eq!(at_3.tiles[1].rows, 1536);
        assert_eq!(at_3.k1_rows_max, 3, "a 3-row head tile is a K1 tile");
        // The 1-row and 8-row tails the mandatory short-tail contract names: each
        // is its own few-row tile with the original routing rows (1 and 8), which
        // the dispatch admission must take as a K1 cohort, never refuse.
        for rows in [1usize, 8] {
            let tail = plan(0, 16384, &[rows]);
            assert_eq!(tail.tiles[0].rows, rows);
            assert_eq!(tail.tiles[0].routing.route_rows, rows);
            assert!(!tail.tiles[0].routing.bf16_streams);
            assert_eq!(tail.k1_rows_max, rows);
        }
    }

    #[test]
    fn slowlink_prompt_boundary_snapshots_force_budgeted_superwindows() {
        // Five pending boundaries: a window keeps at most four strictly inside
        // it, so the fifth ENDS the first window (the transaction's own rule).
        let cuts = [3, 4, 1536, 1537, 8192];
        let plan = plan(0, 16384, &cuts);
        assert_chronology(&plan);
        assert_eq!(plan.windows.len(), 2);
        assert_eq!(plan.windows[0].end_position(), 8192);
        assert_eq!(plan.windows[0].cuts, vec![3, 4, 1536, 1537]);
        assert_eq!(plan.windows[1].position, 8192);
        assert!(plan.windows[1].cuts.is_empty());
        assert!(
            plan.windows
                .iter()
                .all(|w| w.cuts.len() <= S1_MAX_BOUNDARY_SNAPSHOTS),
            "no window exceeds the admitted boundary snapshots"
        );
        assert!(plan.windows[0].tiles.end == plan.windows[1].tiles.start);
        // The window END boundary is the commit's: its tile carries no cut.
        assert!(plan.tiles[plan.windows[0].tiles.end - 1]
            .ends_at_cut
            .is_none());
    }

    #[test]
    fn slowlink_prompt_row_cap_makes_superwindows_and_sweep_count() {
        let plan = plan(0, 40000, &[]);
        assert_chronology(&plan);
        let windows: Vec<usize> = plan.windows.iter().map(|w| w.rows).collect();
        // Three windows (a window ends at a tile end, eleven 1536-row tiles
        // = 16896 rows fill one, so 40000 rows need three), levelled to the
        // smallest tile-multiple cap that still yields three: nine tiles.
        assert_eq!(windows, vec![13824, 13824, 12352]);
        assert_eq!(windows.iter().sum::<usize>(), 40000);
        assert!(windows.iter().all(|&rows| rows <= S1_WINDOW_ROWS));
        // One sweep per window: a long request can never claim one per prompt.
        let reports = vec![WindowReport::default(); plan.windows.len()];
        let receipt = layer_major_receipt(&ReceiptInputs {
            mode: SlowlinkMode::Force,
            plan: &plan,
            resident_layers: 10,
            resident_layer_set: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            host_layers: 38,
            tile_scratch_bytes: 1 << 30,
            reports: &reports,
            peak_vram_bytes: None,
            status: "ok".to_string(),
            moe_iu4: no_moe_iu4(),
        });
        assert_eq!(receipt["sweeps"], 3);
        assert_eq!(receipt["windows"].as_array().unwrap().len(), 3);
        assert!(plan.windows.len() >= profile().windows_for(plan.rows));
    }

    /// S1 binds no CPU mode on any tile. The override exists because a
    /// 1..=4-row tile on gfx1100 (`rows == route_rows`) is exactly the D1
    /// splice window; a 768-row tile is not.
    #[test]
    fn s1_tile_never_cpu_deferred() {
        assert!(!s1_tile_cpu_deferred(false) && !s1_tile_cpu_deferred(true));
        for rows in 1..=4 {
            assert!(hipfire_dispatch::cpu_split::cpu_splice_window(
                true, rows, rows
            ));
        }
        assert!(!hipfire_dispatch::cpu_split::cpu_splice_window(
            true, 768, 768
        ));

        // The plan shapes that used to hit the D1 window: a 1..=4-row tail
        // after one full 768-row tile (route_rows == rows on gfx1100).
        for route in BOTH_ROUTES {
            let profile = product_profile_on(route);
            for tail in 1..=4usize {
                let plan = plan_for_profile(&profile, 0, 768 + tail, &[], &gfx1100_routing, 8)
                    .unwrap()
                    .expect("plans");
                let last = plan.tiles.last().unwrap();
                assert_eq!(last.rows, tail);
                assert_eq!(last.routing.route_rows, tail);
                assert!(!s1_tile_cpu_deferred(true));
            }
        }
    }

    #[test]
    fn slowlink_prompt_mixed_signature_tails_keep_their_original_routing() {
        // Both routing models: a tail keeps the routing rows and HC tag of ITS
        // rows, never the prompt total or the 1536-row tile's.
        for routing in [
            &gfx1100_routing as &dyn Fn(usize) -> TileRouting,
            &canonical_routing,
        ] {
            let plan = plan_with(0, 5000, &[3, 1537], routing, 8)
                .unwrap()
                .expect("plans");
            assert_chronology(&plan);
            for tile in &plan.tiles {
                assert_eq!(
                    tile.routing,
                    routing(tile.rows),
                    "tile of {} rows",
                    tile.rows
                );
            }
            let signatures = plan.signatures();
            assert!(signatures.iter().any(|(rows, _)| *rows == 3));
            assert!(signatures.iter().any(|(rows, _)| *rows == S1_TILE_ROWS));
            assert_eq!(
                signatures.len(),
                {
                    let mut distinct: Vec<usize> = plan.tiles.iter().map(|t| t.rows).collect();
                    distinct.sort_unstable();
                    distinct.dedup();
                    distinct.len()
                },
                "one signature per distinct tile shape"
            );
        }
        let tail = plan(0, 1536 + 100, &[]);
        assert_eq!(
            tail.tiles.iter().map(|t| t.rows).collect::<Vec<_>>(),
            [1536, 100]
        );
        assert_ne!(tail.tiles[1].routing, tail.tiles[0].routing);
        assert!(!gfx1100_routing(3).bf16_streams && gfx1100_routing(1536).bf16_streams);
        let canonical = plan_with(0, 5000, &[3], &canonical_routing, 8)
            .unwrap()
            .unwrap();
        assert_eq!(
            canonical.tiles[0].routing.route_rows,
            QWEN4_F16_WMMA_MIN_TOKENS
        );
        assert_eq!(canonical.k1_rows_max, 0, "canonical rows are never K1");
    }

    #[test]
    fn slowlink_prompt_k1_tail_rows_above_the_route_storage_bound_bypass_before_mutation() {
        // Every tile counting as K1 (a limit above the tile) overflows the
        // few-row region the route storage was sized for.
        let verdict = plan_with(0, 16384, &[], &gfx1100_routing, S1_TILE_ROWS).unwrap();
        assert!(matches!(
            verdict,
            Err(PlanBypass::K1TailRows { window: 0, bound, .. }) if bound == S1_K1_TAIL_ROWS
        ));
        // With the incumbent's limit a window's K1 rows are the short tails
        // only and stay within the bound by construction (four interior cuts
        // plus the entry and final tails, eight rows each).
        let worst = plan(0, 30, &[8, 16, 24]);
        assert!(worst.k1_rows_max <= S1_K1_TAIL_ROWS);
    }

    #[test]
    fn slowlink_prompt_malformed_cuts_are_errors_not_verdicts() {
        for cuts in [vec![0], vec![5, 5], vec![9, 4], vec![100]] {
            assert!(
                plan_with(0, 100, &cuts, &gfx1100_routing, 8).is_err(),
                "{cuts:?}"
            );
        }
        assert!(plan_with(0, 0, &[], &gfx1100_routing, 8).is_err());
    }

    #[test]
    fn slowlink_prompt_ordered_ple_histories_follow_the_private_cursor() {
        let plan = plan(0, 1536 * 3 + 7, &[1536]);
        let tokens: Vec<u32> = (0..plan.rows as u32).map(|t| t * 3 + 1).collect();
        let entry = PleHistory::new(151_643);
        let histories = PleCursor::histories_for(entry, &plan.tiles, &tokens);
        assert_eq!(histories.len(), plan.tiles.len());
        assert_eq!(
            histories[0], entry,
            "the first tile starts at the entry history"
        );
        for (index, tile) in plan.tiles.iter().enumerate().skip(1) {
            // The history at a tile is the entry history advanced by every
            // earlier token, exactly what the sequential chunk loop commits.
            let mut expected = entry;
            for &token in &tokens[..tile.prompt_row] {
                expected.push(token);
            }
            assert_eq!(histories[index], expected, "tile {index}");
            assert_eq!(
                histories[index].previous(),
                [tokens[tile.prompt_row - 2], tokens[tile.prompt_row - 1]]
            );
        }
        // Resumed entry: the published history at the window start is the base.
        let resumed = PleHistory::from_previous(151_643, [11, 22]);
        let from_resume = PleCursor::histories_for(resumed, &plan.tiles, &tokens);
        assert_eq!(from_resume[0], resumed);
        assert_ne!(
            from_resume[0], histories[0],
            "the entry history is the base"
        );
        // The history is the last TWO tokens: once a tile has consumed two rows
        // the entry no longer matters, so a full-size tile's successors agree.
        assert_eq!(
            from_resume[1], histories[1],
            "past two rows of tokens the entry history is forgotten"
        );
        // A one-row head tile leaves one entry token in its successor's history:
        // that is where the entry must (and does) show, by the private cursor.
        let short = self::plan(0, 16384, &[1]);
        assert_eq!(short.tiles[0].rows, 1);
        let short_tokens: Vec<u32> = (0..short.rows as u32).map(|t| t * 7 + 2).collect();
        let from_entry = PleCursor::histories_for(entry, &short.tiles, &short_tokens);
        let from_resumed = PleCursor::histories_for(resumed, &short.tiles, &short_tokens);
        assert_ne!(from_entry[1], from_resumed[1]);
        assert_eq!(
            from_resumed[1].previous(),
            [22, short_tokens[0]],
            "one entry token plus the head tile's only token"
        );
        assert_eq!(
            from_entry[1].previous(),
            [entry.previous()[1], short_tokens[0]]
        );
    }

    #[test]
    fn slowlink_prompt_mtp_tiles_pair_the_previous_last_row_in_order() {
        // The head consumes tiles ascending and contiguous; token `p` pairs
        // with the target hidden of `p - 1`, which for a tile's first row is
        // the previous tile's LAST row (`pending`); position 0 uses its own.
        let plan = plan(0, 1536 * 2 + 40, &[1536, 1540]);
        assert_chronology(&plan);
        for pair in plan.tiles.windows(2) {
            assert_eq!(
                pair[1].position,
                pair[0].end_position(),
                "the previous tile's last row is position {} - 1",
                pair[1].position
            );
            assert!(pair[0].rows <= S1_TILE_ROWS, "bounded capture rows");
        }
        assert_eq!(
            plan.tiles[0].position, 0,
            "prompt position 0 pairs with itself"
        );
        // A resumed MTP fill starts mid-prompt: its first tile pairs with the
        // entry `pending` row saved before the window.
        let resumed_plan = plan_resumed();
        assert_eq!(resumed_plan.tiles[0].position, 3000);
        assert_eq!(
            plan.tiles.iter().filter(|t| t.last_in_window).count(),
            plan.windows.len()
        );
    }

    #[test]
    fn slowlink_prompt_hc_streams_are_exact_tagged_spans_per_tile() {
        let profile = profile();
        let wide = profile.geometry.hc_count * profile.geometry.hidden;
        // 1536-row BF16 tiles, then a 100-row F32 tail and a 3-row F32 head.
        let plan = plan(0, 1536 * 2 + 100, &[3]);
        let mut elements = 0usize;
        for tile in &plan.tiles {
            assert_eq!(
                tile.stream_offset, elements,
                "tiles pack back to back in tag size"
            );
            let bytes = if tile.routing.bf16_streams { 2 } else { 4 };
            elements += tile.rows * wide * bytes / 4;
        }
        let window = &plan.windows[0];
        assert_eq!(window.stream_bytes, elements as u64 * 4);
        let all_f32 = (plan.rows * wide * 4) as u64;
        assert!(
            window.stream_bytes < all_f32,
            "BF16 tiles take half the bytes of an all-F32 reservation"
        );
        // BF16 regions reach past their own bytes by one F32 view: the arena
        // keeps one tile of BF16 slack so the last view stays inside it.
        let slack = (profile.tile_rows * wide * 2) as u64;
        assert!(window.stream_bytes + slack <= profile.arena.total_bytes);
    }

    #[test]
    fn slowlink_prompt_arena_that_cannot_hold_the_tagged_spans_bypasses_before_mutation() {
        let mut small = profile();
        small.arena.total_bytes = 1 << 20;
        let verdict = plan_prompt_verdict(&PromptPlanInputs {
            profile: &small,
            entry_pos: 0,
            rows: 16384,
            cuts: &[],
            routing: &gfx1100_routing,
            few_row_limit: 8,
        })
        .unwrap();
        assert!(matches!(
            verdict,
            Err(PlanBypass::StreamArena { window: 0, .. })
        ));
    }

    #[test]
    fn slowlink_prompt_abort_phase_drains_before_restore() {
        // The cleanup of each named failure point.
        let before_dma = Progress {
            txn_open: true,
            ring_primed: false,
            sp_armed: false,
        };
        assert_eq!(
            before_dma.abort_steps(),
            AbortSteps {
                drain_ring: false,
                sync_xfer: false,
                abort_txn: true
            }
        );
        let during_panel = Progress {
            txn_open: true,
            ring_primed: true,
            sp_armed: false,
        };
        assert_eq!(
            during_panel.abort_steps(),
            AbortSteps {
                drain_ring: true,
                sync_xfer: false,
                abort_txn: true
            }
        );
        // After the window drained (commit stage) only the transaction remains
        // to restore; a commit failure already restored it itself.
        let before_commit = Progress {
            txn_open: true,
            ring_primed: false,
            sp_armed: false,
        };
        assert_eq!(
            before_commit.abort_steps(),
            AbortSteps {
                drain_ring: false,
                sync_xfer: false,
                abort_txn: true
            }
        );
        let consumed = Progress::default();
        assert_eq!(
            consumed.abort_steps(),
            AbortSteps {
                drain_ring: false,
                sync_xfer: false,
                abort_txn: false
            }
        );
        // An SP window also joins the xfer stream, after the ring drain and
        // before the transaction restore; a non-SP window never touches it.
        let sp_panel = Progress {
            txn_open: true,
            ring_primed: true,
            sp_armed: true,
        };
        assert_eq!(
            sp_panel.abort_steps(),
            AbortSteps {
                drain_ring: true,
                sync_xfer: true,
                abort_txn: true
            }
        );
    }

    #[test]
    fn slowlink_prompt_census_equals_the_allocation_layouts() {
        let profile = profile();
        let plan = plan(0, 16384, &[]);
        let census = prompt_bytes(&plan).unwrap();
        let layout = dsp::PromptRouteLayout::new(&profile.route_dims()).unwrap();
        assert_eq!(census.route_storage, layout.total_bytes() as u64);
        assert_eq!(census.route_storage, profile.census.route_storage);
        assert_eq!(census.hc_streams, profile.arena.total_bytes);
        let spec = profile.pool_spec().unwrap();
        let reserve = spec.reserve().unwrap();
        assert_eq!(census.ring + census.lookahead, reserve.payload as u64);
        assert_eq!(census.tables, reserve.tables as u64);
        assert_eq!(census.group_scatter, profile.census.group_scatter);
        assert_eq!(
            census.group_scatter,
            crate::slowlink_admission::group_scatter_bytes(&profile.geometry, profile.window_rows)
                .unwrap(),
            "the receipt's scatter charge is the layout's, one source"
        );
        let wide = profile.geometry.hc_count * profile.geometry.hidden;
        let slack = (profile.tile_rows * wide * 2) as u64;
        assert!(
            profile.arena.span(ArenaField::HcStreams).bytes >= plan.windows[0].stream_bytes + slack,
            "the admitted arena holds the window's tagged spans plus one tile of view slack"
        );
        assert!(census.total().is_some());
    }

    /// The window route layout the runtime builds for each window, for BOTH
    /// artifact route kinds, from the window's ACTUAL cohort rows, at every
    /// window row count of the matrix: no owned span exceeds the full
    /// admitted window's, the full admitted capacities are exactly the
    /// census, capacities above the admitted ones and an empty window are
    /// refused. CPU layout arithmetic only; no kernel or GPU arithmetic is
    /// exercised.
    #[test]
    fn slowlink_prompt_window_route_layout_is_valid_for_every_row_count_on_both_routes() {
        for route in [dsp::WindowRoute::SlabIu4, dsp::WindowRoute::Aggregate] {
            let profile = profile_on(route, 0, 0);
            assert_eq!(profile.geometry.route, route);
            let full = profile.census.route_storage;
            let admitted = profile.route_dims();
            for rows in [1usize, 3, 4, 47, 48, 49, 512, 1536, 16384] {
                let k1_rows = rows.min(S1_K1_TAIL_ROWS);
                let tail_rows = rows.min(S1_AGGREGATE_TAIL_ROWS);
                let mut shapes = vec![(0usize, 0usize), (k1_rows, 0)];
                if route == dsp::WindowRoute::Aggregate {
                    shapes.push((0, tail_rows));
                    if rows > 3 {
                        shapes.push((1, rows.min(S1_AGGREGATE_TAIL_ROWS) - 1));
                    }
                }
                for (k1, tail) in shapes {
                    let layout =
                        window_route_layout(&profile, rows, k1, tail).unwrap_or_else(|error| {
                            panic!("{route:?} rows {rows} k1 {k1} tail {tail}: {error:?}")
                        });
                    let dims = layout.dims();
                    assert_eq!(dims.rows, rows, "{route:?}");
                    assert_eq!(layout.route(), route, "{route:?} rows {rows}");
                    assert_eq!(
                        (dims.k1_rows, dims.tail_rows),
                        (k1, tail),
                        "{route:?} rows {rows}: the ACTUAL counts, never the admitted bounds"
                    );
                    assert!(layout.total_bytes() > 0, "{route:?} rows {rows}");
                    assert!(
                        layout.total_bytes() as u64 <= full,
                        "{route:?} rows {rows}: {} bytes exceed the admitted {full}",
                        layout.total_bytes()
                    );
                }
                if rows == S1_WINDOW_ROWS {
                    let layout =
                        window_route_layout(&profile, rows, admitted.k1_rows, admitted.tail_rows)
                            .unwrap();
                    assert_eq!(layout.total_bytes() as u64, full, "{route:?}");
                }
            }
            assert!(
                window_route_layout(&profile, 0, 0, 0).is_err(),
                "{route:?}: a zero-row window is refused"
            );
            assert!(
                window_route_layout(&profile, S1_WINDOW_ROWS + 1, 0, 0).is_err(),
                "{route:?}: a window wider than the admitted one is refused"
            );
            assert!(
                window_route_layout(&profile, 16384, S1_K1_TAIL_ROWS + 1, 0).is_err(),
                "{route:?}: K1 rows above the admitted capacity are refused"
            );
            assert!(
                window_route_layout(&profile, 3, 1, 3).is_err(),
                "{route:?}: cohorts cannot hold more rows than the window"
            );
        }
        let aggregate = profile_on(dsp::WindowRoute::Aggregate, 0, 0);
        assert!(
            window_route_layout(&aggregate, 16384, 0, S1_AGGREGATE_TAIL_ROWS + 1).is_err(),
            "F32 tail rows above the admitted capacity are refused"
        );
        assert!(
            window_route_layout(&profile_on(dsp::WindowRoute::SlabIu4, 0, 0), 16384, 0, 1).is_err(),
            "the slab route has no F32 tail rows"
        );
    }

    /// The census of each route kind is the layout of a profile admitted FRESH
    /// on that route (never the slab census with the route field flipped).
    #[test]
    fn slowlink_prompt_census_of_each_route_is_its_own_fresh_admission() {
        let slab = profile_on(dsp::WindowRoute::SlabIu4, 0, 0);
        let aggregate = profile_on(dsp::WindowRoute::Aggregate, 0, 0);
        for profile in [&slab, &aggregate] {
            let plan = plan_prompt_verdict(&PromptPlanInputs {
                profile,
                entry_pos: 0,
                rows: 16384,
                cuts: &[],
                routing: &gfx1100_routing,
                few_row_limit: 8,
            })
            .unwrap()
            .expect("a qualified request plans");
            let layout = dsp::PromptRouteLayout::new(&profile.route_dims()).unwrap();
            assert_eq!(layout.route(), profile.geometry.route);
            assert_eq!(profile.census.route_storage, layout.total_bytes() as u64);
            let census = prompt_bytes(&plan).unwrap();
            assert_eq!(census.route_storage, profile.census.route_storage);
            assert_eq!(
                census.total().unwrap(),
                profile.census.owned_total().unwrap()
            );
        }
        let wide = aggregate.route_dims();
        assert_eq!(wide.route, dsp::WindowRoute::Aggregate);
        assert_eq!(wide.k1_rows, S1_K1_TAIL_ROWS);
        assert_eq!(wide.tail_rows, S1_AGGREGATE_TAIL_ROWS);
        assert_eq!(slab.route_dims().tail_rows, 0);
        assert!(
            aggregate.census.route_storage > slab.census.route_storage,
            "the aggregate census carries its own F16 activations and scratch"
        );
    }

    /// The cohort class of a tile is read off its ORIGINAL routing rows on its
    /// own window route, and the dispatch call's kind lands in the same
    /// cohort; only the decode and F32-tail calls register their operands.
    #[test]
    fn slowlink_prompt_cohort_classes_follow_each_routes_original_routing_rows() {
        use dsp::WindowRoute::{Aggregate, SlabIu4};
        assert_eq!(dsp::AGGREGATE_MIN_TILE_ROWS, QWEN4_F16_WMMA_MIN_TOKENS);
        for (rows, want) in [
            (1, TileCohort::K1),
            (2, TileCohort::F32Tail),
            (3, TileCohort::F32Tail),
            (8, TileCohort::F32Tail),
            (9, TileCohort::F32Tail),
            (511, TileCohort::F32Tail),
            (512, TileCohort::K2),
            (1536, TileCohort::K2),
            (16384, TileCohort::K2),
        ] {
            assert_eq!(tile_cohort(Aggregate, rows, 8), want, "aggregate {rows}");
        }
        for (rows, want) in [
            (1, TileCohort::K1),
            (3, TileCohort::K1),
            (8, TileCohort::K1),
            (9, TileCohort::K2),
            (511, TileCohort::K2),
            (512, TileCohort::K2),
            (1536, TileCohort::K2),
        ] {
            assert_eq!(tile_cohort(SlabIu4, rows, 8), want, "slab {rows}");
        }
        for (kind, want) in [
            (dsp::TileKind::SlabK2, TileCohort::K2),
            (dsp::TileKind::AggregateWmma, TileCohort::K2),
            (dsp::TileKind::SlabK1Tail, TileCohort::K1),
            (dsp::TileKind::Decode, TileCohort::K1),
            (dsp::TileKind::AggregateF32Tail, TileCohort::F32Tail),
        ] {
            assert_eq!(kind_cohort(kind), want, "{kind:?}");
        }
        for rows in [1usize, 3, 511] {
            assert!(registers_operands(Aggregate, rows, 8), "aggregate {rows}");
        }
        for rows in [512usize, 1536] {
            assert!(!registers_operands(Aggregate, rows, 8), "aggregate {rows}");
        }
        assert!(registers_operands(SlabIu4, 1, 8));
        assert!(!registers_operands(SlabIu4, 3, 8));
        assert!(!registers_operands(SlabIu4, 1536, 8));
    }

    /// Every mixture counts the real kind rows of each tile: a window of a
    /// 3-row F32 tail, a decode row and a 512-row WMMA tile holds 1 K1 and 3
    /// tail rows on the aggregate route, 4 K1 rows and no tail on the slab
    /// route (its old few-row semantics); a canonical-promoted tile
    /// (`route_rows >= 512`) is never an F32 tail whatever its raw rows.
    #[test]
    fn slowlink_prompt_window_cohort_rows_count_the_real_kind_of_every_tile() {
        let cuts = [3, 4];
        let aggregate = plan_aggregate(0, 516, &cuts);
        assert_chronology(&aggregate);
        assert_eq!(
            aggregate.tiles.iter().map(|t| t.rows).collect::<Vec<_>>(),
            [3, 1, 512]
        );
        assert_eq!(aggregate.windows.len(), 1);
        let window = &aggregate.windows[0];
        assert_eq!((window.k1_rows, window.tail_rows), (1, 3));
        assert_eq!((aggregate.k1_rows_max, aggregate.tail_rows_max), (1, 3));
        let slab = plan(0, 516, &cuts);
        assert_eq!((slab.windows[0].k1_rows, slab.windows[0].tail_rows), (4, 0));
        assert_eq!((slab.k1_rows_max, slab.tail_rows_max), (4, 0));
        let canonical = plan_on(
            dsp::WindowRoute::Aggregate,
            0,
            5000,
            &[3],
            &canonical_routing,
            8,
        )
        .unwrap()
        .expect("plans");
        assert_eq!(canonical.tiles[0].rows, 3);
        assert_eq!(
            canonical.tiles[0].routing.route_rows,
            QWEN4_F16_WMMA_MIN_TOKENS
        );
        assert!(canonical
            .windows
            .iter()
            .all(|w| (w.k1_rows, w.tail_rows) == (0, 0)));
        assert_eq!((canonical.k1_rows_max, canonical.tail_rows_max), (0, 0));
        // The per-window sums are the sums of the tiles' own kinds.
        for (route, plan) in [
            (dsp::WindowRoute::Aggregate, &aggregate),
            (dsp::WindowRoute::SlabIu4, &slab),
            (dsp::WindowRoute::Aggregate, &canonical),
        ] {
            for window in &plan.windows {
                let (mut k1, mut tail) = (0, 0);
                for tile in plan.window_tiles(window) {
                    match tile_cohort(route, tile.routing.route_rows, 8) {
                        TileCohort::K1 => k1 += tile.rows,
                        TileCohort::F32Tail => tail += tile.rows,
                        TileCohort::K2 => {}
                    }
                }
                assert_eq!((window.k1_rows, window.tail_rows), (k1, tail));
            }
        }
    }

    /// Pure F32-only and decode-only windows are valid: the layout holds the
    /// window's actual rows (not the 3,066-row tail bound on a 3-row view, nor
    /// 3 K1 + 3 tail reserved for an actual 3-row F32 window), stays inside the
    /// fresh full-window census, and a zero-row window is invalid.
    #[test]
    fn slowlink_prompt_pure_f32_and_decode_windows_have_valid_actual_layouts() {
        let profile = profile_on(dsp::WindowRoute::Aggregate, 0, 0);
        let f32_only = plan_aggregate(0, 3, &[]);
        let window = &f32_only.windows[0];
        assert_eq!((window.rows, window.k1_rows, window.tail_rows), (3, 0, 3));
        let layout =
            window_route_layout(&profile, window.rows, window.k1_rows, window.tail_rows).unwrap();
        assert_eq!(layout.route(), dsp::WindowRoute::Aggregate);
        assert_eq!(
            (
                layout.dims().rows,
                layout.dims().k1_rows,
                layout.dims().tail_rows
            ),
            (3, 0, 3)
        );
        assert!(layout.total_bytes() as u64 <= profile.census.route_storage);
        validate_window_layouts(&profile, &f32_only).unwrap();
        let decode_only = plan_aggregate(0, 1, &[]);
        let window = &decode_only.windows[0];
        assert_eq!((window.rows, window.k1_rows, window.tail_rows), (1, 1, 0));
        let layout =
            window_route_layout(&profile, window.rows, window.k1_rows, window.tail_rows).unwrap();
        assert_eq!(
            (
                layout.dims().rows,
                layout.dims().k1_rows,
                layout.dims().tail_rows
            ),
            (1, 1, 0)
        );
        assert!(layout.total_bytes() as u64 <= profile.census.route_storage);
        validate_window_layouts(&profile, &decode_only).unwrap();
        // The decode-only window has no prefill tile: the sweep frames a
        // stand-in anchor and registers only the real decode tile.
        let sweep = sweep_tiles(
            dsp::WindowRoute::Aggregate,
            decode_only.window_tiles(&decode_only.windows[0]),
            8,
        );
        assert_eq!(sweep.anchor, None);
        assert_eq!(sweep.cohort, [0]);
        assert!(window_route_layout(&profile, 0, 0, 0).is_err());
    }

    /// A valid long window of 14,843 rows cut at 511, 2558, 4605 and 6652 is
    /// ONE aggregate window holding five 511-row F32 tails (2,555 rows, above
    /// the old 1,999 / 2,047 limits, inside the admitted 3,066) and eight
    /// 1,536-row WMMA tiles: no bypass, no retile. Every original exit, F32
    /// HC tag and operand registration occurs exactly once (the WMMA tile
    /// anchors, so no tail is registered twice) and no decode tile exists.
    #[test]
    fn slowlink_prompt_five_f32_tails_of_a_long_window_stay_one_aggregate_window() {
        let cuts = [511, 2558, 4605, 6652];
        let plan = plan_aggregate(0, 14843, &cuts);
        assert_chronology(&plan);
        assert_eq!(plan.windows.len(), 1);
        assert_eq!(
            plan.tiles.iter().map(|t| t.rows).collect::<Vec<_>>(),
            [
                vec![511],
                vec![1536, 511],
                vec![1536, 511],
                vec![1536, 511],
                vec![1536; 5],
                vec![511]
            ]
            .concat()
        );
        let window = &plan.windows[0];
        assert_eq!(window.cuts, cuts);
        assert_eq!((window.k1_rows, window.tail_rows), (0, 2555));
        assert!(window.tail_rows > 1999 && window.tail_rows <= S1_AGGREGATE_TAIL_ROWS);
        let tiles = plan.window_tiles(window);
        let tails: Vec<usize> = (0..tiles.len()).filter(|&i| tiles[i].rows == 511).collect();
        assert_eq!(tails.len(), 5);
        assert_eq!(tiles.iter().filter(|t| t.rows == 1536).count(), 8);
        // Original exits: every tile end once, every cut among them once.
        let exits = plan.launch_cuts();
        assert_eq!(exits.len(), plan.tiles.len());
        for cut in cuts {
            assert_eq!(exits.iter().filter(|&&exit| exit == cut).count(), 1);
        }
        assert_eq!(exits.last(), Some(&14843));
        // F32 HC tags on the tails, BF16 on the WMMA tiles.
        for tile in &plan.tiles {
            assert_eq!(tile.routing, gfx1100_routing(tile.rows));
            assert_eq!(tile.routing.bf16_streams, tile.rows == 1536);
        }
        // Operand registration: the first WMMA tile anchors, each tail is
        // registered once, nothing is a decode tile.
        let sweep = sweep_tiles(dsp::WindowRoute::Aggregate, tiles, 8);
        assert_eq!(sweep.anchor, Some(1));
        assert_eq!(sweep.cohort, tails);
        assert_eq!(
            sweep.registered(tiles, dsp::WindowRoute::Aggregate, 8),
            tails
        );
        assert!(tiles.iter().all(|tile| tile.routing.route_rows != 1));
        // The layout of the actual window and the independent cohort-count
        // array (no 2,047 / 4,095 limit): the full admitted worst case too.
        let profile = profile_on(dsp::WindowRoute::Aggregate, 0, 0);
        let layout = window_route_layout(&profile, window.rows, 0, 2555).unwrap();
        assert_eq!(layout.dims().tail_rows, 2555);
        assert!(window_route_layout(
            &profile,
            S1_WINDOW_ROWS,
            S1_K1_TAIL_ROWS,
            S1_AGGREGATE_TAIL_ROWS
        )
        .is_ok());
        validate_window_layouts(&profile, &plan).unwrap();
    }

    /// The mandatory 9,000-row boundary request cut at 3, 4, 1536, 1537 and
    /// 8192 keeps the incumbent's chronology: the fifth boundary ends the
    /// first (budgeted) window, whose 3- and 511-row F32 tails and two 1-row
    /// decode tiles are counted from the real kinds and register once each
    /// around the 1,532-row WMMA anchor; the second window is pure WMMA.
    #[test]
    fn slowlink_prompt_boundary_9000_keeps_f32_tails_and_decode_rows_in_their_windows() {
        let cuts = [3, 4, 1536, 1537, 8192];
        let plan = plan_aggregate(0, 9000, &cuts);
        assert_chronology(&plan);
        assert_eq!(
            plan.tiles.iter().map(|t| t.rows).collect::<Vec<_>>(),
            [vec![3, 1, 1532, 1], vec![1536; 4], vec![511, 808]].concat()
        );
        assert_eq!(plan.windows.len(), 2);
        assert!(plan
            .windows
            .iter()
            .all(|w| w.cuts.len() <= S1_MAX_BOUNDARY_SNAPSHOTS));
        assert_eq!(plan.windows[0].end_position(), 8192);
        assert_eq!(plan.windows[0].cuts, vec![3, 4, 1536, 1537]);
        let (first, second) = (&plan.windows[0], &plan.windows[1]);
        assert_eq!((first.k1_rows, first.tail_rows), (2, 3 + 511));
        assert_eq!((second.k1_rows, second.tail_rows), (0, 0));
        let exits = plan.launch_cuts();
        for cut in cuts {
            assert_eq!(exits.iter().filter(|&&exit| exit == cut).count(), 1);
        }
        let tiles = plan.window_tiles(first);
        let sweep = sweep_tiles(dsp::WindowRoute::Aggregate, tiles, 8);
        assert_eq!(sweep.anchor, Some(2), "the first WMMA tile anchors");
        assert_eq!(sweep.cohort, [0, 1, 3, 8]);
        assert_eq!(
            sweep.registered(tiles, dsp::WindowRoute::Aggregate, 8),
            [0, 1, 3, 8]
        );
        assert_eq!(
            tiles.iter().filter(|t| t.routing.route_rows == 1).count(),
            2,
            "decode tiles exist only for real one-row tiles"
        );
        let tail_tiles: Vec<usize> = tiles
            .iter()
            .enumerate()
            .filter(|(_, tile)| !tile.routing.bf16_streams && tile.routing.route_rows > 1)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(tail_tiles, [0, 8], "the F32 tails carry F32 HC tags");
        let tiles = plan.window_tiles(second);
        let sweep = sweep_tiles(dsp::WindowRoute::Aggregate, tiles, 8);
        assert_eq!((sweep.anchor, sweep.cohort.len()), (Some(0), 0));
        let profile = profile_on(dsp::WindowRoute::Aggregate, 0, 0);
        validate_window_layouts(&profile, &plan).unwrap();
    }

    /// A request whose last window holds only 3 rows (16384 rows, then the
    /// 3-row tail the fifth cut separates): the production per-window layout
    /// is valid for it on both route kinds from its ACTUAL cohort rows (K1 on
    /// the slab route, a pure F32 tail on the aggregate route, no WMMA work
    /// and the tail anchors and registers itself once), and the plan's tile
    /// manifest is the same before and after the layouts are built.
    #[test]
    fn slowlink_prompt_three_row_last_window_builds_a_valid_layout_on_both_routes() {
        let cuts = [1536, 3072, 4608, 6144, 16384];
        let slab_plan = plan(0, 16387, &cuts);
        let aggregate_plan = plan_aggregate(0, 16387, &cuts);
        for (route, plan, k1, tail) in [
            (dsp::WindowRoute::SlabIu4, &slab_plan, 3usize, 0usize),
            (dsp::WindowRoute::Aggregate, &aggregate_plan, 0, 3),
        ] {
            assert_chronology(plan);
            assert_eq!(plan.windows.len(), 2);
            assert_eq!(plan.windows[0].rows, 16384);
            assert_eq!(
                (plan.windows[0].k1_rows, plan.windows[0].tail_rows),
                (0, 0),
                "{route:?}: the first window is all K2 tiles"
            );
            let last = &plan.windows[1];
            assert_eq!(last.rows, 3);
            assert_eq!(last.position, 16384);
            assert_eq!(last.end_position(), 16387);
            assert_eq!(last.tiles.len(), 1);
            assert_eq!(plan.tiles[last.tiles.start].rows, 3);
            assert_eq!(
                (plan.k1_rows_max, plan.tail_rows_max),
                (k1, tail),
                "{route:?}"
            );
            assert_eq!((last.k1_rows, last.tail_rows), (k1, tail), "{route:?}");
            let manifest = plan.tiles.clone();
            let windows = plan.windows.clone();
            let profile = profile_on(route, 0, 0);
            let dims_before = profile.route_dims();
            for window in &plan.windows {
                let layout =
                    window_route_layout(&profile, window.rows, window.k1_rows, window.tail_rows)
                        .unwrap_or_else(|error| {
                            panic!("{route:?} window {}: {error:?}", window.index)
                        });
                assert_eq!(layout.dims().rows, window.rows);
                assert_eq!(
                    (layout.dims().k1_rows, layout.dims().tail_rows),
                    (window.k1_rows, window.tail_rows)
                );
                assert!(layout.total_bytes() as u64 <= profile.census.route_storage);
            }
            validate_window_layouts(&profile, plan).unwrap();
            let layout =
                window_route_layout(&profile, last.rows, last.k1_rows, last.tail_rows).unwrap();
            assert_eq!(
                (layout.dims().k1_rows, layout.dims().tail_rows),
                (k1, tail),
                "{route:?}"
            );
            assert_eq!(profile.route_dims(), dims_before, "{route:?}");
            assert_eq!(plan.tiles, manifest, "the tile manifest is untouched");
            assert_eq!(plan.windows, windows);
            let tiles = plan.window_tiles(last);
            let sweep = sweep_tiles(route, tiles, 8);
            assert_eq!(sweep.anchor, Some(0), "{route:?}: the 3-row tile anchors");
            assert!(sweep.cohort.is_empty(), "{route:?}: no extra sweep tile");
            assert_eq!(
                sweep.registered(tiles, route, 8),
                if route == dsp::WindowRoute::Aggregate {
                    vec![0]
                } else {
                    Vec::new()
                },
                "{route:?}: the aggregate tail registers once through the anchor"
            );
        }
    }

    /// A plan whose actual F32-tail or decode rows exceed the immutable
    /// admitted capacities is the incumbent's, decided before any mutation.
    #[test]
    fn slowlink_prompt_aggregate_cohort_rows_above_admitted_capacity_bypass_before_mutation() {
        let all_tails = |_: usize| TileRouting {
            route_rows: 100,
            bf16_streams: false,
        };
        let verdict = plan_on(dsp::WindowRoute::Aggregate, 0, 16384, &[], &all_tails, 8).unwrap();
        assert!(matches!(
            verdict,
            Err(PlanBypass::F32TailRows { window: 0, rows, bound })
                if bound == S1_AGGREGATE_TAIL_ROWS && rows > bound
        ));
        let all_decode = |_: usize| TileRouting {
            route_rows: 1,
            bf16_streams: false,
        };
        let verdict = plan_on(dsp::WindowRoute::Aggregate, 0, 16384, &[], &all_decode, 8).unwrap();
        assert!(matches!(
            verdict,
            Err(PlanBypass::K1TailRows { window: 0, rows, bound })
                if bound == S1_K1_TAIL_ROWS && rows > bound
        ));
        // The structural worst case of a window (five segments between four
        // interior cuts, each ending in a 511-row tail) plans and stays
        // inside the admitted bound.
        let worst = plan_aggregate(0, 5 * 511, &[511, 1022, 1533, 2044]);
        assert_eq!(worst.tail_rows_max, 5 * 511);
        assert!(worst.tail_rows_max <= S1_AGGREGATE_TAIL_ROWS);
        // A zero-row routing is malformed input, not a verdict.
        let zero = |_: usize| TileRouting {
            route_rows: 0,
            bf16_streams: false,
        };
        assert!(plan_on(dsp::WindowRoute::Aggregate, 0, 100, &[], &zero, 8).is_err());
    }

    /// The physical prompt census is the allocation layouts' total whatever
    /// budget the admission reserves beside them: the runtime-overhead and
    /// diagnostic terms are in the full reserve, never in the census, and a
    /// runtime overhead the base reserve already holds only shrinks the
    /// supplement, never an allocation.
    #[test]
    fn slowlink_prompt_census_stays_the_owned_total_beside_budget_terms() {
        const DIAGNOSTIC: u64 = 300 * MIB;
        let baseline_owned = profile().census.owned_total().unwrap();
        for base_runtime in [0, 600 * MIB, S1_RUNTIME_OVERHEAD_BYTES + 476 * MIB] {
            let budgeted = profile_with_budgets(DIAGNOSTIC, base_runtime);
            let runtime = S1_RUNTIME_OVERHEAD_BYTES.saturating_sub(base_runtime);
            assert_eq!(budgeted.census.runtime_overhead, runtime, "{base_runtime}");
            assert_eq!(budgeted.census.diagnostic, DIAGNOSTIC);
            let plan = plan_prompt_verdict(&PromptPlanInputs {
                profile: &budgeted,
                entry_pos: 0,
                rows: 16384,
                cuts: &[],
                routing: &gfx1100_routing,
                few_row_limit: 8,
            })
            .unwrap()
            .expect("a qualified request plans");
            let physical = prompt_bytes(&plan).unwrap().total().unwrap();
            let owned = budgeted.census.owned_total().unwrap();
            assert_eq!(physical, owned, "allocations are the owned layouts only");
            assert_eq!(
                physical, baseline_owned,
                "budget terms move no allocation ({base_runtime})"
            );
            assert_eq!(
                budgeted.reserve_bytes().unwrap(),
                owned + runtime + DIAGNOSTIC,
                "the full reserve carries both budget terms ({base_runtime})"
            );
            assert_eq!(
                budgeted.reserve_bytes().unwrap(),
                budgeted.census.total().unwrap()
            );
            assert_eq!(budgeted.census.owned_terms().len(), 7);
            assert_eq!(budgeted.census.terms().len(), 9);
        }
    }

    #[test]
    fn slowlink_prompt_receipt_carries_every_required_key() {
        let plan = plan(0, 16384, &[1536]);
        let report = WindowReport {
            sweep_bytes: 50_803_507_200,
            stats: PanelStats {
                submitted_payload_bytes: 50_803_507_200,
                ..PanelStats::default()
            },
            ..WindowReport::default()
        };
        let receipt = layer_major_receipt(&ReceiptInputs {
            mode: SlowlinkMode::Force,
            plan: &plan,
            resident_layers: 10,
            resident_layer_set: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            host_layers: 38,
            tile_scratch_bytes: 5 << 30,
            reports: &[report],
            peak_vram_bytes: Some(22 << 30),
            status: "ok".to_string(),
            moe_iu4: no_moe_iu4(),
        });
        for key in [
            "schema",
            "route",
            "selected_by",
            "bypass_reason",
            "moe_iu4",
            "entry_pos",
            "prompt_rows",
            "windows",
            "launch_cuts",
            "tile_rows",
            "retained_scratch_rows",
            "sweeps",
            "resident_layers",
            "resident_layer_set",
            "host_layers",
            "host_expert_bytes_total",
            "submitted_bytes",
            "useful_bytes",
            "unused_prefetched_bytes",
            "reserve",
            "spans_ms",
            "overlap",
            "peak_vram_bytes",
        ] {
            assert!(receipt.get(key).is_some(), "receipt lacks {key}");
        }
        // The VRAM figure is one post-request sample, and the receipt says so:
        // never a continuous peak.
        assert_eq!(
            receipt["vram_measurement"],
            json!({"method": "hipMemGetInfo_post_request_sample", "continuous_peak": false})
        );
        assert_eq!(receipt["peak_vram_bytes"], 22u64 << 30);
        assert_eq!(receipt["schema"], RECEIPT_SCHEMA);
        assert_eq!(receipt["route"], "layer-major");
        assert_eq!(receipt["selected_by"], "env1");
        assert_eq!(
            receipt["sweeps"],
            receipt["windows"].as_array().unwrap().len()
        );
        assert!(receipt["retained_scratch_rows"].as_u64().unwrap() <= 1536);
        assert!(receipt["tile_rows"].as_u64().unwrap() <= 1536);
        assert_eq!(receipt["submitted_bytes"], 50_803_507_200u64);
        assert_eq!(
            receipt["resident_layer_set"],
            json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        );
        // A non-prefix placement passes through verbatim, ascending.
        let interleaved = layer_major_receipt(&ReceiptInputs {
            mode: SlowlinkMode::Force,
            plan: &plan,
            resident_layers: 5,
            resident_layer_set: &[1, 3, 4, 20, 47],
            host_layers: 43,
            tile_scratch_bytes: 0,
            reports: &[],
            peak_vram_bytes: None,
            status: "ok".to_string(),
            moe_iu4: no_moe_iu4(),
        });
        assert_eq!(interleaved["resident_layer_set"], json!([1, 3, 4, 20, 47]));
        assert_eq!(interleaved["resident_layers"], 5);
        for key in ["tile", "arena", "ring", "lookahead", "tables", "total"] {
            assert!(receipt["reserve"].get(key).is_some(), "reserve lacks {key}");
        }
        for key in ["wall", "copy", "compute", "route_barrier", "nonexpert"] {
            assert!(
                receipt["spans_ms"].get(key).is_some(),
                "spans_ms lacks {key}"
            );
        }
        for key in [
            "cross_layer_intersections",
            "cross_layer_overlap_ms",
            "copy_idle_ms",
            "page_full_stalls",
            "copy_starved_stalls",
        ] {
            assert!(receipt["overlap"].get(key).is_some(), "overlap lacks {key}");
        }
        let total = receipt["reserve"]["total"].as_u64().unwrap();
        let parts: u64 = ["tile", "arena", "ring", "lookahead", "tables"]
            .iter()
            .map(|key| receipt["reserve"][key].as_u64().unwrap())
            .sum();
        assert!(total >= parts, "the charged total covers every named term");
    }

    #[test]
    fn slowlink_prompt_receipt_unused_bytes_include_untouched_experts_of_acquired_groups() {
        let plan = plan(0, 16384, &[1536]);
        let receipt_of = |reports: &[WindowReport]| {
            layer_major_receipt(&ReceiptInputs {
                mode: SlowlinkMode::Force,
                plan: &plan,
                resident_layers: 10,
                resident_layer_set: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                host_layers: 38,
                tile_scratch_bytes: 0,
                reports,
                peak_vram_bytes: None,
                status: "ok".to_string(),
                moe_iu4: no_moe_iu4(),
            })
        };
        // Every group was acquired (the transport discarded nothing), yet the
        // routed rows read only part of the experts inside them: the rest of
        // the submitted payload was moved for nothing.
        let partial = WindowReport {
            sweep_bytes: 1000,
            useful_bytes: 640,
            stats: PanelStats {
                submitted_payload_bytes: 1000,
                discarded_payload_bytes: 0,
                ..PanelStats::default()
            },
            ..WindowReport::default()
        };
        assert_eq!(partial.unused_bytes(), Some(360));
        let receipt = receipt_of(&[partial, partial]);
        assert_eq!(receipt["submitted_bytes"], 2000u64);
        assert_eq!(receipt["useful_bytes"], 1280u64);
        assert_eq!(
            receipt["unused_prefetched_bytes"], 720u64,
            "submitted minus useful, even with the discard counter at zero"
        );
        assert_eq!(receipt["discarded_group_payload_bytes"], 0u64);
        // Wholly unused groups are part of the same difference, never added twice.
        let with_discard = WindowReport {
            useful_bytes: 640,
            stats: PanelStats {
                submitted_payload_bytes: 1000,
                discarded_payload_bytes: 200,
                ..PanelStats::default()
            },
            ..WindowReport::default()
        };
        let receipt = receipt_of(&[with_discard]);
        assert_eq!(receipt["unused_prefetched_bytes"], 360u64);
        assert_eq!(receipt["discarded_group_payload_bytes"], 200u64);
        // Fully used: nothing unused. Useful above submitted is a broken
        // measurement: the receipt reports it as missing, never underflows.
        let full = WindowReport {
            useful_bytes: 1000,
            ..partial
        };
        assert_eq!(receipt_of(&[full])["unused_prefetched_bytes"], 0u64);
        let broken = WindowReport {
            useful_bytes: 1001,
            ..partial
        };
        assert_eq!(broken.unused_bytes(), None);
        assert!(receipt_of(&[broken])["unused_prefetched_bytes"].is_null());
    }

    // ── the ACTUAL shipped tiles: product 768, primitive-purpose 512 ──────
    //
    // The numeric class fixtures above declare the 1536 counterfactual
    // ([`profile`]); these prove the same planner structure on the profiles
    // `admit` really returns, with arena and census recomputed at the tile.

    const BOTH_ROUTES: [dsp::WindowRoute; 2] =
        [dsp::WindowRoute::SlabIu4, dsp::WindowRoute::Aggregate];

    /// The window layout of `window` validated against `profile`, its tagged
    /// stream spans inside the arena with one tile of BF16 view slack, and its
    /// census equal to the allocation layouts.
    fn assert_window_fits_profile(profile: &SlowlinkProfile, plan: &Qwen4PromptPlan) {
        validate_window_layouts(profile, plan).unwrap();
        let wide = profile.geometry.hc_count * profile.geometry.hidden;
        let slack = (profile.tile_rows * wide * 2) as u64;
        for window in &plan.windows {
            window_route_layout(profile, window.rows, window.k1_rows, window.tail_rows).unwrap();
            assert!(
                window.stream_bytes + slack <= profile.arena.total_bytes,
                "the window's tagged HC spans plus one tile of slack fit the admitted arena"
            );
        }
        let census = prompt_bytes(plan).unwrap();
        assert_eq!(census.route_storage, profile.census.route_storage);
        assert_eq!(census.hc_streams, profile.arena.total_bytes);
        assert!(plan.tiles.iter().all(|t| t.rows <= profile.tile_rows));
    }

    #[test]
    fn slowlink_prompt_actual_product_tile_768_runs_16384_native_wmma_with_a_final_256_f32_tile() {
        for route in BOTH_ROUTES {
            let profile = product_profile_on(route);
            assert_eq!(profile.tile_rows, S1_PRODUCT_TILE_ROWS);
            assert_eq!(profile.tile_rows, 768);
            assert_eq!(
                profile.window_rows, S1_WINDOW_ROWS,
                "the full 22-tile window"
            );
            let plan = plan_for_profile(&profile, 0, 16384, &[], &gfx1100_routing, 8)
                .unwrap()
                .expect("the product tile plans a full window");
            let rows: Vec<usize> = plan.tiles.iter().map(|t| t.rows).collect();
            assert_eq!(
                rows,
                [vec![768; 21], vec![256]].concat(),
                "{route:?}: 21 native 768-row tiles then the final 256-row tile"
            );
            assert_eq!(
                plan.windows.len(),
                1,
                "{route:?}: one full 16,384-row window"
            );
            assert_eq!(plan.windows[0].rows, 16384);
            // 768 rows is at or above the WMMA minimum: BF16 streams, K2.
            for tile in &plan.tiles[..21] {
                assert!(tile.routing.bf16_streams, "{route:?}");
                assert_eq!(
                    tile_cohort(route, tile.routing.route_rows, 8),
                    TileCohort::K2
                );
            }
            // 256 rows is below it: an F32-stream tile.
            let last = plan.tiles.last().unwrap();
            assert_eq!(last.rows, 256);
            assert!(
                !last.routing.bf16_streams,
                "{route:?}: the final tile is F32"
            );
            let window = &plan.windows[0];
            match route {
                dsp::WindowRoute::Aggregate | dsp::WindowRoute::AggregateSymIu4 => {
                    assert_eq!(
                        tile_cohort(route, 256, 8),
                        TileCohort::F32Tail,
                        "the final 256 rows are an aggregate F32 tail"
                    );
                    assert_eq!((window.k1_rows, window.tail_rows), (0, 256));
                    assert_eq!(plan.tail_rows_max, 256);
                    assert!(plan.tail_rows_max <= S1_AGGREGATE_TAIL_ROWS);
                }
                dsp::WindowRoute::SlabIu4 => {
                    assert_eq!(tile_cohort(route, 256, 8), TileCohort::K2);
                    assert_eq!((window.k1_rows, window.tail_rows), (0, 0));
                }
            }
            assert_eq!(plan.k1_rows_max, 0);
            assert_chronology(&plan);
            assert_window_fits_profile(&profile, &plan);
        }
    }

    #[test]
    fn slowlink_prompt_actual_primitive_tile_512_is_all_native_wmma_over_a_full_window() {
        for route in BOTH_ROUTES {
            let profile = primitive_profile_on(route);
            assert_eq!(profile.tile_rows, S1_PRIMITIVE_TILE_ROWS);
            assert_eq!(profile.tile_rows, 512);
            let plan = plan_for_profile(&profile, 0, 16384, &[], &gfx1100_routing, 8)
                .unwrap()
                .expect("the primitive tile plans a full window");
            assert_eq!(
                plan.tiles.iter().map(|t| t.rows).collect::<Vec<_>>(),
                vec![512; 32],
                "{route:?}"
            );
            assert!(plan.tiles.iter().all(|t| t.routing.bf16_streams));
            assert_eq!((plan.k1_rows_max, plan.tail_rows_max), (0, 0));
            assert_eq!(plan.windows.len(), 1);
            assert_chronology(&plan);
            assert_window_fits_profile(&profile, &plan);
        }
    }

    #[test]
    fn slowlink_prompt_actual_tiles_recompute_arena_and_census_not_just_the_tile() {
        for route in BOTH_ROUTES {
            let counterfactual = profile_on(route, 0, 0);
            let product = product_profile_on(route);
            let primitive = primitive_profile_on(route);
            assert_eq!(counterfactual.tile_rows, S1_TILE_ROWS);
            // The HC arena is sized by the tile: a smaller tile is a smaller
            // arena, so the explicit 1536 counterfactual is a distinct admission,
            // never the actual profile with a patched `tile_rows`.
            assert!(primitive.arena.total_bytes < product.arena.total_bytes);
            assert!(product.arena.total_bytes < counterfactual.arena.total_bytes);
            // Window-sized owners do not depend on the tile.
            for actual in [&product, &primitive] {
                assert_eq!(
                    actual.census.route_storage,
                    counterfactual.census.route_storage
                );
                assert_eq!(
                    actual.census.group_scatter,
                    counterfactual.census.group_scatter
                );
                assert_eq!(actual.census.a4_sidecars, counterfactual.census.a4_sidecars);
                assert_eq!(actual.window_rows, counterfactual.window_rows);
            }
        }
    }

    #[test]
    fn slowlink_prompt_actual_tiles_keep_the_mandatory_boundary_chronology() {
        // Cuts 3, 4, 1536, 1537 and 8192 with a 9000-row request: the same
        // chronology as the 1536 fixtures, on the actual tiles.
        let cuts = [3, 4, 1536, 1537, 8192];
        for route in BOTH_ROUTES {
            for profile in [product_profile_on(route), primitive_profile_on(route)] {
                let plan = plan_for_profile(&profile, 0, 9000, &cuts, &gfx1100_routing, 8)
                    .unwrap()
                    .expect("the mandatory boundary request plans");
                assert_chronology(&plan);
                assert_eq!(
                    plan.windows.len(),
                    2,
                    "{route:?}: the fifth cut ends window one"
                );
                assert_eq!(plan.windows[0].end_position(), 8192);
                assert_eq!(plan.windows[0].cuts, vec![3, 4, 1536, 1537]);
                assert_eq!(plan.windows[1].position, 8192);
                assert!(plan.windows[1].cuts.is_empty());
                for &cut in &cuts {
                    assert!(
                        plan.tiles.iter().any(|t| t.end_position() == cut),
                        "{route:?}/{}: cut {cut} is a tile end",
                        profile.tile_rows
                    );
                    assert!(plan
                        .tiles
                        .iter()
                        .all(|t| !(t.position < cut && cut < t.end_position())));
                }
                assert!(plan.tiles.iter().all(|t| t.rows <= profile.tile_rows));
                assert_window_fits_profile(&profile, &plan);
            }
        }
    }

    /// ACTUAL fixture admission inputs with exactly `free` actual free bytes.
    fn adaptive_inputs(
        route: dsp::WindowRoute,
        purpose: AdmissionPurpose,
        free: u64,
    ) -> AdmissionInputs {
        let mut inputs = fixture_inputs(route, purpose, 0, 0);
        inputs.memory = crate::slowlink_admission::AdmissionMemory {
            pre_init_free_bytes: None,
            free_bytes: free,
        };
        inputs
    }

    /// Bytes the shared quote needs to keep `layers` resident (the same
    /// function `admit`, the frontier and the projection use).
    fn adaptive_needed(route: dsp::WindowRoute, purpose: AdmissionPurpose, layers: usize) -> u64 {
        let inputs = fixture_inputs(route, purpose, 0, 0);
        let projection = crate::slowlink_admission::project_at_tile(&inputs, tile_of(purpose))
            .expect("the fixture projects at its purpose's tile");
        u64::try_from(projection.budget.needed_bytes_for(layers)).unwrap()
    }

    fn tile_of(purpose: AdmissionPurpose) -> usize {
        crate::slowlink_admission::purpose_tile_rows(purpose)
    }

    /// CPU proof of the adaptive admission over the zero, nine and ten
    /// resident schedules, for both purposes on both routes: the count is the
    /// LARGEST that fits (no invented floor, ten is just data), a base that
    /// does not fit is a capacity refusal distinct from the valid zero-resident
    /// fit, an all-resident fit keeps the no-host bypass, the snapshot
    /// allowance is exactly the surplus, and the prompt plan (tiles, windows,
    /// cuts, retained checkpoint cadence) is the SAME whatever the count.
    #[test]
    fn slowlink_prompt_adaptive_zero_nine_ten_resident_keep_one_plan_and_exact_boundaries() {
        use crate::slowlink_admission::SlowlinkDecline;
        let cuts = [3, 4, 1536, 1537, 8192, 9000];
        for route in BOTH_ROUTES {
            for purpose in [
                AdmissionPurpose::Product,
                AdmissionPurpose::PrimitiveSelftest,
            ] {
                let need = |layers| adaptive_needed(route, purpose, layers);
                let admit_free = |free| admit(&adaptive_inputs(route, purpose, free));
                let layer_cost = need(1) - need(0);
                // Base does not fit by one byte: refused with the exact
                // capacity decline, never admitted as zero resident.
                let refusal = admit_free(need(0) - 1).unwrap_err();
                assert!(refusal.is_capacity(), "{route:?}/{purpose:?}: {refusal:?}");
                let mut shared: Option<(usize, usize, Vec<WindowPlan>, Vec<usize>, Vec<usize>)> =
                    None;
                for layers in [0usize, 9, 10] {
                    let exact = admit_free(need(layers)).unwrap();
                    assert_eq!(exact.resident_layers, layers, "{route:?}/{purpose:?}");
                    assert_eq!(exact.tile_rows, tile_of(purpose));
                    assert_eq!(exact.purpose, purpose);
                    assert_eq!(exact.device_snapshot_budget_bytes, 0);
                    let spare = admit_free(need(layers) + 12_345).unwrap();
                    assert_eq!(spare.resident_layers, layers);
                    assert_eq!(spare.device_snapshot_budget_bytes, 12_345);
                    if layers > 0 {
                        let short = admit_free(need(layers) - 1).unwrap();
                        assert_eq!(short.resident_layers, layers - 1);
                        assert_eq!(short.device_snapshot_budget_bytes, layer_cost - 1);
                    }
                    let plan = plan_for_profile(&exact, 0, 40000, &cuts, &gfx1100_routing, 8)
                        .unwrap()
                        .expect("a qualified request plans");
                    assert_chronology(&plan);
                    let tiles: Vec<usize> = plan.tiles.iter().map(|tile| tile.rows).collect();
                    let summary = (
                        exact.checkpoint_stride_rows,
                        exact.window_rows,
                        plan.windows.clone(),
                        plan.launch_cuts(),
                        tiles,
                    );
                    assert_eq!(
                        summary.0, 8192,
                        "the retained snapshot cadence is unchanged"
                    );
                    match &shared {
                        None => shared = Some(summary),
                        Some(first) => assert_eq!(
                            first, &summary,
                            "{route:?}/{purpose:?}: {layers} resident changes the prompt plan"
                        ),
                    }
                }
                // Every routed layer fitting is the no-host bypass, not an S1 profile.
                assert!(matches!(
                    admit_free(need(48)),
                    Err(SlowlinkDecline::NoHostLayers { .. })
                ));
                assert_eq!(admit_free(need(47)).unwrap().resident_layers, 47);
            }
        }
    }
}

#[cfg(test)]
mod slowlink_overlap {
    use super::*;
    use crate::gpu_forward::panel_stage::{
        ExpertExtent, ExpertSource, LayerExpertSources, PanelBackend, PanelCodec, PanelPoolSpec,
        RingCore, TimelineEvent,
    };
    use std::sync::Arc;

    const EXPERTS: usize = 512;
    const GU: usize = 256;
    const DN: usize = 256;
    const PAGES: usize = 18;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ev {
        Begin { slot: usize, wait_free: bool },
        End { slot: usize },
        Acquire { slot: usize },
        Release { slot: usize },
        Drain,
    }

    /// What the model device saw, in issue order (one log per test thread).
    #[derive(Default)]
    struct Log {
        events: Vec<Ev>,
        payload_bytes: u64,
    }

    thread_local! {
        static LOG: std::cell::RefCell<Log> = std::cell::RefCell::new(Log::default());
    }

    fn record(f: impl FnOnce(&mut Log)) {
        LOG.with(|log| f(&mut log.borrow_mut()));
    }

    /// A CPU model of the device: records stream operations in issue order.
    struct Model {
        tables: Vec<(Vec<u64>, Vec<u64>)>,
    }

    impl Model {
        fn new(pages: usize) -> Self {
            LOG.with(|log| *log.borrow_mut() = Log::default());
            Self {
                tables: (0..pages)
                    .map(|_| (vec![0; EXPERTS], vec![0; EXPERTS]))
                    .collect(),
            }
        }
    }

    impl PanelBackend for Model {
        type Ctx = ();

        fn page_base(&self, slot: usize) -> usize {
            0x4000_0000 + slot * 0x20_0000
        }
        fn tables_mut(&mut self, slot: usize) -> (&mut [u64], &mut [u64]) {
            let (gate_up, down) = &mut self.tables[slot];
            (gate_up.as_mut_slice(), down.as_mut_slice())
        }
        fn await_copy_done(&mut self, _: &(), _: usize) -> R<()> {
            Ok(())
        }
        fn copy_begin(&mut self, _: &(), slot: usize, wait_free: bool, _: Option<usize>) -> R<()> {
            record(|log| log.events.push(Ev::Begin { slot, wait_free }));
            Ok(())
        }
        fn copy_payload(&mut self, _: &(), _: usize, _: usize, _: usize, len: usize) -> R<()> {
            record(|log| log.payload_bytes += len as u64);
            Ok(())
        }
        fn copy_payload_device(&mut self, _: &(), _: usize, _: usize, _: usize, _: usize) -> R<()> {
            Ok(())
        }
        fn copy_zero(&mut self, _: &(), _: usize, _: usize, _: usize) -> R<()> {
            Ok(())
        }
        fn copy_tables(&mut self, _: &(), _: usize) -> R<()> {
            Ok(())
        }
        fn copy_end(&mut self, _: &(), slot: usize, _: Option<usize>) -> R<()> {
            record(|log| log.events.push(Ev::End { slot }));
            Ok(())
        }
        fn compute_acquire(&mut self, _: &(), slot: usize, _: Option<usize>) -> R<bool> {
            record(|log| log.events.push(Ev::Acquire { slot }));
            Ok(false)
        }
        fn compute_release(&mut self, _: &(), slot: usize, _: Option<usize>) -> R<()> {
            record(|log| log.events.push(Ev::Release { slot }));
            Ok(())
        }
        fn mark_base(&mut self, _: &()) -> R<()> {
            Ok(())
        }
        fn mark_layer(&mut self, _: &(), _: usize, _: bool) -> R<()> {
            Ok(())
        }
        fn drain(&mut self, _: &()) -> R<()> {
            record(|log| log.events.push(Ev::Drain));
            Ok(())
        }
        fn elapsed_ms(&self, _: &(), _: TimelineEvent, _: TimelineEvent) -> R<f64> {
            Ok(0.0)
        }
    }

    /// The model's two transport views of the ring: the walker's `GroupRing`.
    struct CoreRing<'a> {
        core: &'a mut RingCore<Model>,
        schedule: &'a mut PanelSchedule,
    }

    impl GroupRing for CoreRing<'_> {
        type Ctx = ();

        fn require(&mut self, ctx: &(), layer: usize, group: u32) -> R<PanelTicket> {
            self.schedule.require(self.core, ctx, layer, group)
        }
        fn with_lease(
            &mut self,
            ctx: &mut (),
            ticket: PanelTicket,
            compute: &mut dyn FnMut(&mut (), Option<&PanelBindingParts<'_>>) -> R<()>,
        ) -> R<()> {
            self.core.acquire(&*ctx, ticket)?;
            compute(ctx, None)
        }
        fn retire(&mut self, ctx: &(), ticket: PanelTicket) -> R<()> {
            self.core.retire(ctx, ticket)
        }
        fn pump(&mut self, ctx: &()) -> R<usize> {
            self.schedule.pump(self.core, ctx)
        }
    }

    /// One pinned host arena all fixture layers read: expert `e`'s gate/up at
    /// `e * GU`, its down after the gate/up region.
    struct Arena {
        bytes: Vec<u8>,
    }

    impl Arena {
        fn new() -> Self {
            Self {
                bytes: (0..EXPERTS * (GU + DN) + 4096)
                    .map(|i| (i % 251) as u8)
                    .collect(),
            }
        }

        fn sources(&self, layer: usize) -> LayerExpertSources {
            let base = self.bytes.as_ptr() as usize;
            let experts = (0..EXPERTS)
                .map(|id| {
                    Some(ExpertSource {
                        codec: PanelCodec(0),
                        gate_up: ExpertExtent {
                            addr: base + id * GU,
                            len: GU,
                        },
                        down: ExpertExtent {
                            addr: base + EXPERTS * GU + id * DN,
                            len: DN,
                        },
                    })
                })
                .collect();
            // SAFETY: the extents lie inside `self.bytes`, which the test keeps
            // alive (and unmodified) for as long as the sources.
            unsafe { LayerExpertSources::new(layer, base, self.bytes.len(), experts, Arc::new(())) }
                .unwrap()
        }
    }

    fn spec() -> PanelPoolSpec {
        PanelPoolSpec::uniform(2, 16, 32, EXPERTS, GU, DN, 0, 0).unwrap()
    }

    fn fixture(host_layers: &[usize]) -> (Arena, RingCore<Model>, PanelSchedule) {
        let arena = Arena::new();
        let schedule =
            PanelSchedule::new(host_layers.iter().map(|&l| arena.sources(l)).collect(), 32)
                .unwrap();
        let core = RingCore::new(spec(), Model::new(PAGES)).unwrap();
        (arena, core, schedule)
    }

    fn work(
        schedule: &PanelSchedule,
        layer: usize,
        bound: impl Fn(usize) -> bool,
    ) -> Vec<GroupWork> {
        schedule
            .layer_groups(layer)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, group)| GroupWork {
                group: group.group,
                bound: bound(index),
            })
            .collect()
    }

    /// The window start: lookahead covers the first host layer while the
    /// resident layers compute.
    #[test]
    fn slowlink_overlap_first_host_layer_prefetches_before_any_compute() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        let issued = schedule.prime(&mut core, &()).unwrap();
        assert_eq!(issued, 16, "the whole first host layer fits the page store");
        assert!(core
            .model_events()
            .iter()
            .all(|e| !matches!(e, Ev::Acquire { .. })));
        for group in 0..16u32 {
            assert!(schedule.ticket(10, group).is_some());
        }
        assert!(
            schedule.ticket(11, 0).is_none(),
            "nothing of layer 11 until compute is on layer 10"
        );
    }

    #[test]
    fn slowlink_overlap_next_layer_pages_enqueue_before_the_counts_barrier() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        // Priming, before any compute, prefetches the FIRST host layer into the
        // sixteen lookahead pages: its mixer and counts are still ahead.
        schedule.prime(&mut core, &()).unwrap();
        assert_eq!(core.stats().enqueued, 16, "the first host layer, 16 groups");
        assert_eq!(core.stats().acquired, 0);
        assert!((0..16).all(|group| schedule.ticket(10, group).is_some()));
        assert!(schedule.ticket(11, 0).is_none());
        // Compute reaches layer 10: its mixer and counts are still ahead, yet
        // the two pages the current layer does not hold already carry the first
        // two groups of layer 11 (every current-layer group holds a page, so no
        // page is reserved for pending current work).
        schedule.enter_layer(&mut core, &(), 10).unwrap();
        assert_eq!(
            core.stats().enqueued,
            18,
            "16 current + 2 next-layer pages, no acquire yet"
        );
        assert_eq!(core.stats().acquired, 0);
        assert!(schedule.ticket(11, 0).is_some() && schedule.ticket(11, 1).is_some());
        assert!(schedule.ticket(11, 2).is_none(), "the page store is full");
        // The walk retires layer 10's groups; every freed page is refilled with
        // the NEXT layer's group right away, so layer 11's whole payload is
        // enqueued before layer 11 is entered (before its counts barrier).
        let work = work(&schedule, 10, |_| true);
        walk_layer_groups(
            &mut CoreRing {
                core: &mut core,
                schedule: &mut schedule,
            },
            &mut (),
            10,
            &work,
            &mut |_, _, _| Ok(()),
        )
        .unwrap();
        assert_eq!(
            core.stats().enqueued,
            32,
            "16 current-layer groups, then all 16 next-layer groups, no acquire of layer 11"
        );
        assert_eq!(core.stats().acquired, 16);
        assert!((0..16).all(|group| schedule.ticket(11, group).is_some()));
        assert!(
            schedule.ticket(12, 0).is_none(),
            "one next layer ahead at most"
        );
        // Every layer-11 transfer was enqueued after layer 10's groups and
        // before layer 11 began.
        let begins: Vec<usize> = core
            .model_events()
            .iter()
            .filter_map(|event| match event {
                Ev::End { slot } => Some(*slot),
                _ => None,
            })
            .collect();
        assert_eq!(begins.len(), 32);
    }

    #[test]
    fn slowlink_overlap_walk_retires_every_group_once_and_refills_in_order() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        schedule.prime(&mut core, &()).unwrap();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for layer in [10usize, 11, 12] {
            schedule.enter_layer(&mut core, &(), layer).unwrap();
            // Every third group is read by no panel: retired unread, never refetched.
            let work = work(&schedule, layer, |index| index % 3 != 0);
            let summary = walk_layer_groups(
                &mut CoreRing {
                    core: &mut core,
                    schedule: &mut schedule,
                },
                &mut (),
                layer,
                &work,
                &mut |_, index, parts| {
                    assert!(parts.is_none(), "the model has no device binding");
                    seen.push((layer, index));
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(summary.acquired + summary.discarded, 16);
            assert_eq!(summary.discarded, 6, "groups 0,3,6,9,12,15 are unread");
        }
        // Ascending (layer, group), each computed group exactly once.
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        assert_eq!(seen, sorted);
        assert_eq!(seen.len(), 3 * 10);
        let stats = core.stats();
        assert_eq!(
            stats.enqueued, 48,
            "one transfer per group per window, no refetch"
        );
        assert_eq!(stats.retired, 48, "every ticket of every layer retired");
        assert_eq!(stats.discarded, 18);
        assert_eq!(stats.acquired, 30);
        assert_eq!(stats.submitted_payload_bytes, schedule.sweep_bytes());
        assert_eq!(core.model_payload(), schedule.sweep_bytes());
    }

    #[test]
    fn slowlink_overlap_page_reuse_waits_for_the_last_reader() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        schedule.prime(&mut core, &()).unwrap();
        for layer in [10usize, 11, 12] {
            schedule.enter_layer(&mut core, &(), layer).unwrap();
            let work = work(&schedule, layer, |_| true);
            walk_layer_groups(
                &mut CoreRing {
                    core: &mut core,
                    schedule: &mut schedule,
                },
                &mut (),
                layer,
                &work,
                &mut |_, _, _| Ok(()),
            )
            .unwrap();
        }
        // Per page the event order is Begin, End, Acquire, Release and every
        // refill after a Release has the copy stream wait on its free event.
        let events = core.model_events();
        for slot in 0..PAGES {
            let mut released = false;
            for event in events.iter().filter(|e| match e {
                Ev::Begin { slot: s, .. }
                | Ev::End { slot: s }
                | Ev::Acquire { slot: s }
                | Ev::Release { slot: s } => *s == slot,
                Ev::Drain => false,
            }) {
                match event {
                    Ev::Begin { wait_free, .. } => {
                        assert_eq!(
                            *wait_free, released,
                            "slot {slot}: a reuse waits the free event"
                        );
                    }
                    Ev::Release { .. } => released = true,
                    _ => {}
                }
            }
        }
        assert!(
            events.iter().any(|e| matches!(
                e,
                Ev::Begin {
                    wait_free: true,
                    ..
                }
            )),
            "pages were reused"
        );
    }

    #[test]
    fn slowlink_overlap_abort_drain_leaves_the_ring_reusable_and_tickets_stale() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11]);
        schedule.prime(&mut core, &()).unwrap();
        schedule.enter_layer(&mut core, &(), 10).unwrap();
        let stale = schedule.ticket(10, 0).unwrap();
        // The window fails after two groups: the abort path drains first.
        let work = work(&schedule, 10, |_| true);
        let mut count = 0;
        let error = walk_layer_groups(
            &mut CoreRing {
                core: &mut core,
                schedule: &mut schedule,
            },
            &mut (),
            10,
            &work,
            &mut |_, _, _| {
                count += 1;
                if count == 3 {
                    Err(invalid("injected failure during a panel"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("injected"));
        core.drain(&()).unwrap();
        assert!(core.model_events().contains(&Ev::Drain));
        assert!(
            core.acquire(&(), stale).is_err(),
            "an old ticket cannot reach a page that may hold other bytes"
        );
        // The next window starts clean: first host layer first, ascending.
        schedule.prime(&mut core, &()).unwrap();
        assert!(schedule.ticket(10, 0).is_some());
        assert_eq!(
            core.stats().enqueued,
            16,
            "statistics restart with the window"
        );
    }

    #[test]
    fn slowlink_overlap_binding_ranges_cover_the_layer_in_group_order() {
        let (_arena, _core, schedule) = fixture(&[10]);
        let groups = schedule.layer_groups(10).unwrap();
        let ranges = group_binding_ranges(groups, EXPERTS);
        assert_eq!(ranges.len(), groups.len());
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, EXPERTS);
        for (index, pair) in ranges.windows(2).enumerate() {
            assert_eq!(
                pair[0].1,
                pair[1].0,
                "ranges {index}/{} are contiguous",
                index + 1
            );
        }
        for (range, group) in ranges.iter().zip(groups) {
            assert!(group.experts.iter().all(|&e| e >= range.0 && e < range.1));
        }
    }

    #[test]
    fn slowlink_overlap_panels_group_by_ascending_binding() {
        let by_group = panels_by_group([0usize, 0, 1, 3, 3, 3].into_iter(), 4).unwrap();
        assert_eq!(by_group, vec![0..2, 2..3, 0..0, 3..6]);
        assert!(
            panels_by_group([1usize, 0].into_iter(), 2).is_err(),
            "bindings must ascend"
        );
        assert!(panels_by_group([0usize, 1, 0].into_iter(), 2).is_err());
        assert!(
            panels_by_group([2usize].into_iter(), 2).is_err(),
            "a panel names a real group"
        );
    }

    #[test]
    fn slowlink_overlap_resident_layers_have_no_scheduled_groups() {
        let (_arena, _core, schedule) = fixture(&[10, 12]);
        assert!(schedule.layer_groups(11).is_none());
        assert!(schedule.layer_groups(3).is_none());
        assert_eq!(schedule.layer_groups(10).unwrap().len(), 16);
    }

    /// CPU proof of the adaptive resident counts. Admission keeps the LARGEST
    /// count that fits (ten is only preferred), so a window may stream all 48
    /// routed layers (zero resident), 39 (nine resident) or 38 (ten resident).
    /// The schedule, ring and walker never assume a resident prefix: the first
    /// host layer is whichever layer follows the resident ones, its whole
    /// payload is primed before any compute, every host layer is submitted
    /// exactly once per window, and the next window restarts the same way.
    #[test]
    fn slowlink_overlap_adaptive_resident_counts_schedule_every_host_layer_once() {
        const TRUNK: usize = 48;
        let (_one_arena, _one_core, one) = fixture(&[0]);
        let layer_bytes = one.sweep_bytes();
        assert!(layer_bytes > 0);
        for resident in [0usize, 9, 10] {
            let host: Vec<usize> = (resident..TRUNK).collect();
            let (_arena, mut core, mut schedule) = fixture(&host);
            assert_eq!(
                schedule.sweep_bytes(),
                host.len() as u64 * layer_bytes,
                "{resident} resident: one sweep is every host layer, nothing else"
            );
            for layer in 0..TRUNK {
                assert_eq!(
                    schedule.layer_groups(layer).is_some(),
                    layer >= resident,
                    "{resident} resident: layer {layer}"
                );
            }
            for window in 0..2u64 {
                let acquires_before = core
                    .model_events()
                    .iter()
                    .filter(|event| matches!(event, Ev::Acquire { .. }))
                    .count();
                assert_eq!(
                    schedule.prime(&mut core, &()).unwrap(),
                    16,
                    "{resident} resident: the whole first host layer fits the page store"
                );
                assert_eq!(
                    core.model_events()
                        .iter()
                        .filter(|event| matches!(event, Ev::Acquire { .. }))
                        .count(),
                    acquires_before,
                    "{resident} resident: priming reads no page"
                );
                assert!(
                    (0..16).all(|group| schedule.ticket(resident, group).is_some()),
                    "{resident} resident: layer {resident} is the first host layer"
                );
                assert!(schedule.ticket(resident + 1, 0).is_none());
                // Resident layers only pump: the full lookahead is already queued.
                for _ in 0..resident {
                    assert_eq!(schedule.pump(&mut core, &()).unwrap(), 0);
                }
                let mut seen: Vec<(usize, usize)> = Vec::new();
                for layer in resident..TRUNK {
                    schedule.enter_layer(&mut core, &(), layer).unwrap();
                    let work = work(&schedule, layer, |index| index % 3 != 0);
                    let summary = walk_layer_groups(
                        &mut CoreRing {
                            core: &mut core,
                            schedule: &mut schedule,
                        },
                        &mut (),
                        layer,
                        &work,
                        &mut |_, index, parts| {
                            assert!(parts.is_none(), "the model has no device binding");
                            seen.push((layer, index));
                            Ok(())
                        },
                    )
                    .unwrap();
                    assert_eq!(summary.acquired + summary.discarded, 16);
                }
                let mut sorted = seen.clone();
                sorted.sort_unstable();
                assert_eq!(seen, sorted, "{resident} resident: ascending layer, group");
                assert_eq!(seen.len(), host.len() * 10);
                let stats = core.stats();
                assert_eq!(stats.enqueued as usize, host.len() * 16, "{resident}");
                assert_eq!(stats.retired as usize, host.len() * 16, "{resident}");
                assert_eq!(stats.acquired as usize, host.len() * 10, "{resident}");
                assert_eq!(stats.discarded as usize, host.len() * 6, "{resident}");
                assert_eq!(stats.submitted_payload_bytes, schedule.sweep_bytes());
                assert_eq!(
                    core.model_payload(),
                    (window + 1) * schedule.sweep_bytes(),
                    "{resident} resident: no refetch, one sweep per window"
                );
                core.drain(&()).unwrap();
            }
            // Compute can never enter a layer the schedule does not hold: a
            // resident layer is not a host layer, whatever the count.
            assert!(schedule.enter_layer(&mut core, &(), TRUNK).is_err());
        }
    }

    trait ModelView {
        fn model_events(&self) -> Vec<Ev>;
        fn model_payload(&self) -> u64;
    }

    impl ModelView for RingCore<Model> {
        fn model_events(&self) -> Vec<Ev> {
            LOG.with(|log| log.borrow().events.clone())
        }
        fn model_payload(&self) -> u64 {
            LOG.with(|log| log.borrow().payload_bytes)
        }
    }
}

/// CPU proofs of the load-time owner binding: the same [`bind_load_owners`]
/// walk the production binder runs, against a model device that records the
/// real allocation steps, their charge and their cleanup disposition.
#[cfg(test)]
mod slowlink_eager_load {
    use super::slowlink_prompt::{primitive_profile_on, product_profile_on};
    use super::*;

    const ROUTES: [dsp::WindowRoute; 3] = [
        dsp::WindowRoute::SlabIu4,
        dsp::WindowRoute::Aggregate,
        dsp::WindowRoute::AggregateSymIu4,
    ];

    /// The two ACTUAL admitted profiles of `route`: product (768-row tile) and
    /// primitive-purpose (512-row tile).
    fn actual_profiles(route: dsp::WindowRoute) -> [SlowlinkProfile; 2] {
        [product_profile_on(route), primitive_profile_on(route)]
    }

    /// A device with a byte capacity: owners allocate their census charge,
    /// cleanup either returns them (in reverse) or quarantines them.
    struct ModelDevice {
        profile: SlowlinkProfile,
        capacity: u64,
        live: Vec<(LoadOwner, u64)>,
        quarantined: Vec<LoadOwner>,
        log: Vec<String>,
        fail_at: Option<LoadOwner>,
        drain_fails: bool,
        /// An S1 request mutation (transaction begin) happened.
        transaction_begun: bool,
    }

    impl ModelDevice {
        fn new(route: dsp::WindowRoute, capacity: u64) -> Self {
            Self {
                profile: product_profile_on(route),
                capacity,
                live: Vec::new(),
                quarantined: Vec::new(),
                log: Vec::new(),
                fail_at: None,
                drain_fails: false,
                transaction_begun: false,
            }
        }

        fn used(&self) -> u64 {
            self.live.iter().map(|(_, bytes)| bytes).sum()
        }

        /// The incumbent's own first-use allocation (a short prompt's scratch).
        fn incumbent_alloc(&mut self, bytes: u64) -> Result<(), String> {
            if self.used() + bytes > self.capacity {
                return Err("hipMalloc: out of memory (incumbent first use)".to_string());
            }
            self.log.push(format!("incumbent {bytes}"));
            Ok(())
        }
    }

    impl LoadBinder for ModelDevice {
        fn bind(&mut self, owner: LoadOwner) -> Result<(), String> {
            let bytes = load_owner_charge(&self.profile, owner);
            if self.fail_at == Some(owner) || self.used() + bytes > self.capacity {
                return Err(format!("hipMalloc: out of memory ({bytes} B)"));
            }
            self.live.push((owner, bytes));
            self.log.push(format!("bind {owner:?}"));
            Ok(())
        }

        fn unwind(&mut self, bound: &[LoadOwner]) -> String {
            assert_eq!(
                bound,
                self.live
                    .iter()
                    .map(|(owner, _)| *owner)
                    .collect::<Vec<_>>(),
                "the unwind covers exactly the owners that were bound"
            );
            if self.drain_fails {
                self.quarantined
                    .extend(self.live.drain(..).map(|(owner, _)| owner));
                return "drain failed: owners QUARANTINED, not released".to_string();
            }
            while let Some((owner, _)) = self.live.pop() {
                self.log.push(format!("release {owner:?}"));
            }
            "released to the device pool".to_string()
        }
    }

    /// What a request's select does about the runtime: only the gate.
    fn select_model(dev: &ModelDevice, bound: bool, declined: Option<&str>) -> Result<(), String> {
        let before = dev.log.len();
        let verdict = runtime_gate(bound, declined);
        assert_eq!(dev.log.len(), before, "select never allocates");
        verdict
    }

    #[test]
    fn slowlink_eager_load_owners_cover_every_census_term_once() {
        for (route, profile) in ROUTES
            .into_iter()
            .flat_map(|route| actual_profiles(route).map(move |profile| (route, profile)))
        {
            let owned: u64 = profile.census.owned_terms().iter().map(|(_, b)| b).sum();
            let bound: u64 = LOAD_OWNERS
                .iter()
                .map(|owner| load_owner_charge(&profile, *owner))
                .sum();
            assert_eq!(
                bound, owned,
                "{route:?}: every charged region has one owner"
            );
            let mut seen = LOAD_OWNERS.to_vec();
            seen.dedup();
            assert_eq!(seen.len(), LOAD_OWNERS.len());
            assert!(
                load_owner_charge(&profile, LoadOwner::RouteRegion) > 0
                    && load_owner_charge(&profile, LoadOwner::HcArena) > 0
                    && load_owner_charge(&profile, LoadOwner::BoundaryStages) > 0,
                "{route:?}: the owners the first-use path used to allocate are charged"
            );
        }
    }

    #[test]
    fn slowlink_eager_load_window_scratch_args_are_the_census_bytes() {
        for (route, profile) in ROUTES
            .into_iter()
            .flat_map(|route| actual_profiles(route).map(move |profile| (route, profile)))
        {
            let args = window_scratch_args(&profile);
            assert_eq!(args.rows, profile.window_rows);
            let group =
                rdna_compute::scratch::qwen4_moe_group_layout(args.rows * dsp::K_TOP, args.experts)
                    .bytes as u64;
            let sidecars = match args.route {
                dsp::WindowRoute::SlabIu4 | dsp::WindowRoute::AggregateSymIu4 => {
                    rdna_compute::scratch::qwen4_moe_down_i4_bytes(args.hidden, args.rows).unwrap()
                        as u64
                        + rdna_compute::scratch::qwen4_moe_down_i4_bytes(
                            args.intermediate,
                            args.rows * dsp::K_TOP,
                        )
                        .unwrap() as u64
                }
                dsp::WindowRoute::Aggregate => 0,
            };
            assert_eq!(group, profile.census.group_scatter, "{route:?}");
            assert_eq!(sidecars, profile.census.a4_sidecars, "{route:?}");
        }
    }

    #[test]
    fn slowlink_eager_load_owns_every_charged_region_before_the_first_short_incumbent() {
        for route in ROUTES {
            let mut dev = ModelDevice::new(route, 0);
            let owned: u64 = dev
                .profile
                .census
                .owned_terms()
                .iter()
                .map(|(_, b)| b)
                .sum();
            let incumbent = 3 << 30;
            dev.capacity = owned + incumbent;
            bind_load_owners(&mut dev).expect("the charged owners fit the charged capacity");
            assert_eq!(
                dev.live.iter().map(|(o, _)| *o).collect::<Vec<_>>(),
                LOAD_OWNERS,
                "{route:?}: every owner is live at load"
            );
            assert_eq!(dev.used(), owned);
            // The first short incumbent prefill (and any later first-use
            // owner) can only take what the placement left: it never lands
            // on S1's bytes, and the S1 owners are untouched when it
            // over-asks.
            dev.incumbent_alloc(incumbent).unwrap();
            assert!(dev.incumbent_alloc(incumbent + 1).is_err());
            assert_eq!(dev.used(), owned, "S1's owners survive the incumbent's OOM");
            let first_incumbent = dev
                .log
                .iter()
                .position(|entry| entry.starts_with("incumbent"))
                .unwrap();
            assert_eq!(
                dev.log[..first_incumbent].len(),
                LOAD_OWNERS.len(),
                "all {} binds precede the first incumbent allocation",
                LOAD_OWNERS.len()
            );
        }
    }

    #[test]
    fn slowlink_eager_load_select_consumes_the_bound_runtime_and_never_allocates() {
        let mut dev = ModelDevice::new(dsp::WindowRoute::SlabIu4, u64::MAX);
        bind_load_owners(&mut dev).unwrap();
        let live = dev.live.clone();
        let log = dev.log.clone();
        // Repeated short (bypassed) and long (S1) selects read only the gate.
        for _ in 0..3 {
            assert!(select_model(&dev, true, None).is_ok());
        }
        assert_eq!((dev.live.clone(), dev.log.clone()), (live, log));
        // A runtime that is not bound is an error, never an allocation.
        let unbound = select_model(&dev, false, None).unwrap_err();
        assert!(unbound.contains("not bound at load"), "{unbound}");
        // A load-time decline is the request's named bypass.
        assert_eq!(
            select_model(&dev, true, Some("lane scope")).unwrap_err(),
            "lane scope"
        );
        assert_eq!(
            select_model(&dev, false, Some("the NPU trunk")).unwrap_err(),
            "the NPU trunk"
        );
    }

    #[test]
    fn slowlink_eager_load_failure_unwinds_what_was_bound_and_begins_no_transaction() {
        for route in ROUTES {
            for (index, failing) in LOAD_OWNERS.into_iter().enumerate() {
                let mut dev = ModelDevice::new(route, u64::MAX);
                dev.fail_at = Some(failing);
                let error = bind_load_owners(&mut dev).unwrap_err();
                assert!(error.contains(&format!("{failing:?}")), "{error}");
                assert!(error.contains("out of memory"), "{error}");
                assert!(
                    error.contains("cleanup: released to the device pool"),
                    "{error}"
                );
                assert!(
                    dev.live.is_empty(),
                    "{route:?}/{failing:?}: nothing stays live"
                );
                assert!(dev.quarantined.is_empty());
                assert!(!dev.transaction_begun, "no S1 transaction began");
                let released: Vec<_> = dev
                    .log
                    .iter()
                    .filter(|entry| entry.starts_with("release"))
                    .cloned()
                    .collect();
                let expected: Vec<_> = LOAD_OWNERS[..index]
                    .iter()
                    .rev()
                    .map(|owner| format!("release {owner:?}"))
                    .collect();
                assert_eq!(released, expected, "last bound is released first");
            }
        }
    }

    #[test]
    fn slowlink_eager_load_over_capacity_fails_on_the_first_owner_that_does_not_fit() {
        let mut dev = ModelDevice::new(dsp::WindowRoute::SlabIu4, 0);
        let ring = load_owner_charge(&dev.profile, LoadOwner::PagePool);
        let route = load_owner_charge(&dev.profile, LoadOwner::RouteRegion);
        // Room for the ring and a route region one byte short: the route
        // region (the allocation that failed in j1778) is the named failure.
        dev.capacity = ring + route - 1;
        let error = bind_load_owners(&mut dev).unwrap_err();
        assert!(error.contains("RouteRegion"), "{error}");
        assert!(dev.live.is_empty());
        assert_eq!(dev.log, vec!["bind PagePool", "release PagePool"]);
    }

    #[test]
    fn slowlink_eager_load_failed_drain_quarantines_instead_of_releasing() {
        let mut dev = ModelDevice::new(dsp::WindowRoute::Aggregate, u64::MAX);
        dev.fail_at = Some(LoadOwner::BoundaryStages);
        dev.drain_fails = true;
        let error = bind_load_owners(&mut dev).unwrap_err();
        assert!(error.contains("QUARANTINED"), "{error}");
        assert!(!error.contains("released to the device pool"), "{error}");
        assert_eq!(dev.quarantined, LOAD_OWNERS[..4].to_vec());
        assert!(dev.log.iter().all(|entry| !entry.starts_with("release")));
    }

    #[test]
    fn slowlink_eager_load_sp_host_owner_binds_last_and_charges_no_device_bytes() {
        for route in ROUTES {
            let mut dev = ModelDevice::new(route, 0);
            let owned: u64 = dev
                .profile
                .census
                .owned_terms()
                .iter()
                .map(|(_, b)| b)
                .sum();
            assert_eq!(load_owner_charge(&dev.profile, LoadOwner::SpHost), 0);
            dev.capacity = owned;
            bind_load_owners_with_sp(&mut dev)
                .expect("SP adds no device byte: the S1 capacity still fits");
            assert_eq!(
                dev.live.iter().map(|(o, _)| *o).collect::<Vec<_>>(),
                LOAD_OWNERS_SP,
                "{route:?}: the SP owner binds last"
            );
            assert_eq!(dev.used(), owned, "{route:?}: resident bytes are unchanged");
            assert_eq!(LOAD_OWNERS_SP[..5], LOAD_OWNERS);
        }
    }

    #[test]
    fn slowlink_eager_load_without_sp_binds_exactly_the_s1_owners() {
        let mut dev = ModelDevice::new(dsp::WindowRoute::Aggregate, u64::MAX);
        bind_load_owners(&mut dev).unwrap();
        assert_eq!(
            dev.live.iter().map(|(o, _)| *o).collect::<Vec<_>>(),
            LOAD_OWNERS
        );
        assert!(dev.log.iter().all(|entry| !entry.contains("SpHost")));
    }

    #[test]
    fn slowlink_eager_load_sp_failure_unwinds_every_earlier_owner_in_reverse() {
        for route in ROUTES {
            for (index, failing) in LOAD_OWNERS_SP.into_iter().enumerate() {
                let mut dev = ModelDevice::new(route, u64::MAX);
                dev.fail_at = Some(failing);
                let error = bind_load_owners_with_sp(&mut dev).unwrap_err();
                assert!(error.contains(&format!("{failing:?}")), "{error}");
                assert!(
                    error.contains("cleanup: released to the device pool"),
                    "{error}"
                );
                assert!(dev.live.is_empty() && dev.quarantined.is_empty());
                let released: Vec<_> = dev
                    .log
                    .iter()
                    .filter(|entry| entry.starts_with("release"))
                    .cloned()
                    .collect();
                let expected: Vec<_> = LOAD_OWNERS_SP[..index]
                    .iter()
                    .rev()
                    .map(|owner| format!("release {owner:?}"))
                    .collect();
                assert_eq!(released, expected, "{route:?}/{failing:?}");
            }
        }
    }

    #[test]
    fn slowlink_eager_load_sp_failed_drain_quarantines_every_bound_owner() {
        let mut dev = ModelDevice::new(dsp::WindowRoute::Aggregate, u64::MAX);
        dev.fail_at = Some(LoadOwner::SpHost);
        dev.drain_fails = true;
        let error = bind_load_owners_with_sp(&mut dev).unwrap_err();
        assert!(error.contains("SpHost"), "{error}");
        assert!(error.contains("QUARANTINED"), "{error}");
        assert!(!error.contains("released to the device pool"), "{error}");
        assert_eq!(dev.quarantined, LOAD_OWNERS.to_vec());
        assert!(dev.log.iter().all(|entry| !entry.starts_with("release")));
    }

    #[cfg(test)]
    mod slowlink_moe_iu4 {
        use super::slowlink_prompt::{no_moe_iu4, plan_for_profile, product_profile_on};
        use super::*;
        use rdna_compute::expert_capabilities::QT_MQ4_XTS;

        const SYM: dsp::WindowRoute = dsp::WindowRoute::AggregateSymIu4;
        const F16: dsp::WindowRoute = dsp::WindowRoute::Aggregate;

        fn gfx1100_routing(rows: usize) -> TileRouting {
            TileRouting {
                route_rows: rows,
                bf16_streams: rows >= dsp::AGGREGATE_MIN_TILE_ROWS,
            }
        }

        fn gfx1100_descriptor() -> &'static rdna_compute::gemm::MoeSymIu4Kernel {
            let caps = rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                std::sync::Arc::new(rdna_compute::FeatureFlags::for_test("gfx1100")),
            );
            rdna_compute::gemm::moe_sym_iu4_kernel(&caps, QT_MQ4_XTS)
                .expect("a gfx1100 MQ4 descriptor")
        }

        #[test]
        fn slowlink_moe_iu4_cohorts_split_at_the_512_and_1536_route_row_boundaries() {
            let few_row_limit = 8;
            // One row is the decode cohort, 2..=511 the F32 tail, 512+ the K2 (IU4) tile; every
            // boundary of the 512/1536 floors, identical to the F16 aggregate route's.
            for (rows, expected) in [
                (1usize, TileCohort::K1),
                (2, TileCohort::F32Tail),
                (511, TileCohort::F32Tail),
                (512, TileCohort::K2),
                (513, TileCohort::K2),
                (1535, TileCohort::K2),
                (1536, TileCohort::K2),
                (1537, TileCohort::K2),
            ] {
                assert_eq!(
                    tile_cohort(SYM, rows, few_row_limit),
                    expected,
                    "{rows} rows"
                );
                assert_eq!(
                    tile_cohort(SYM, rows, few_row_limit),
                    tile_cohort(F16, rows, few_row_limit),
                    "{rows} rows: the IU4 window never re-tags the F32 tail or decode tile"
                );
            }
            assert_eq!(dsp::AGGREGATE_MIN_TILE_ROWS, 512);
            // The dispatch classes the cohorts must land in.
            assert_eq!(kind_cohort(dsp::TileKind::AggregateIu4), TileCohort::K2);
            assert_eq!(kind_cohort(dsp::TileKind::AggregateWmma), TileCohort::K2);
            assert_eq!(
                kind_cohort(dsp::TileKind::AggregateF32Tail),
                TileCohort::F32Tail
            );
            assert_eq!(kind_cohort(dsp::TileKind::Decode), TileCohort::K1);
        }

        #[test]
        fn slowlink_moe_iu4_sweep_anchors_and_registers_like_the_aggregate_route() {
            let tile = |index: usize, rows: usize| TilePlan {
                index,
                window: 0,
                prompt_row: 0,
                window_row: 0,
                stream_offset: 0,
                rows,
                position: 0,
                routing: gfx1100_routing(rows),
                ends_at_cut: None,
                last_in_window: false,
            };
            // An F32 tail opens the window: the anchor is still the first K2 (IU4) tile.
            let tiles = [tile(0, 300), tile(1, 768), tile(2, 1), tile(3, 256)];
            let f16 = sweep_tiles(F16, &tiles, 8);
            let sym = sweep_tiles(SYM, &tiles, 8);
            assert_eq!(sym, f16);
            assert_eq!(sym.anchor, Some(1));
            assert_eq!(sym.cohort, vec![0, 2, 3]);
        }

        #[test]
        fn slowlink_moe_iu4_16k_window_keeps_the_final_256_row_tail_f32() {
            let profile = product_profile_on(SYM);
            assert_eq!(profile.tile_rows, 768);
            let plan = plan_for_profile(&profile, 0, 16384, &[], &gfx1100_routing, 8)
                .unwrap()
                .expect("a 16K request plans on the IU4 route");
            assert_eq!(plan.windows.len(), 1);
            // 21 full 768-row IU4 tiles and the final 256-row tile, which never joins the IU4 K2.
            let counts = moe_iu4_tile_counts(SYM, &plan.tiles, 8);
            assert_eq!(counts, [21, 0, 1, 0, 0]);
            assert_eq!(plan.tiles.last().unwrap().rows, 256);
            assert_eq!(plan.windows[0].tail_rows, 256);
            assert_eq!(plan.windows[0].k1_rows, 0);
            // The F16 route classifies the same tiles with the WMMA arm in the IU4 tiles' place.
            let f16 = moe_iu4_tile_counts(F16, &plan.tiles, 8);
            assert_eq!(f16, [0, 21, 1, 0, 0]);
            let slab = moe_iu4_tile_counts(dsp::WindowRoute::SlabIu4, &plan.tiles, 8);
            assert_eq!(slab, [0, 0, 0, 22, 0]);
        }

        #[test]
        fn slowlink_moe_iu4_census_is_the_runtime_layout_and_charges_both_sidecars() {
            for route in [dsp::WindowRoute::SlabIu4, F16, SYM] {
                let profile = product_profile_on(route);
                let layout = dsp::PromptRouteLayout::new(&profile.route_dims()).unwrap();
                assert_eq!(layout.route(), route);
                assert_eq!(layout.total_bytes() as u64, profile.census.route_storage);
                assert_eq!(layout.a4_sidecar_bytes() as u64, profile.census.a4_sidecars);
                assert_eq!(
                    layout.group_scatter_bytes() as u64,
                    profile.census.group_scatter
                );
            }
            let sym = product_profile_on(SYM);
            let f16 = product_profile_on(F16);
            assert_eq!(sym.window_rows, crate::slowlink_admission::S1_WINDOW_ROWS);
            // Both A4 sidecars (gate 20 * N * 72, down 5 * N * 10 * 72): 78.75 MiB at the former 16,384-row window, 81.21 MiB at 16,896.
            assert_eq!(sym.census.a4_sidecars, 85_155_840);
            assert_eq!(
                sym.census.a4_sidecars,
                20 * 16896 * 72 + 5 * 16896 * 10 * 72
            );
            assert_eq!(f16.census.a4_sidecars, 0);
            // No window F16 input and no panel-local F16 slot map on the IU4 route.
            let layout = dsp::PromptRouteLayout::new(&sym.route_dims()).unwrap();
            assert_eq!(layout.x_f16_bytes(), 0);
            assert_eq!(
                f16.census.route_storage - sym.census.route_storage,
                86_507_520 + 2048 * 4
            );
            // The F32 tail cohort and decode rows keep the aggregate bounds on both routes.
            assert_eq!(sym.route_dims().tail_rows, f16.route_dims().tail_rows);
            assert_eq!(sym.route_dims().k1_rows, f16.route_dims().k1_rows);
            assert_eq!(sym.census.group_scatter, f16.census.group_scatter);
            // Every other term of the census is unchanged between the routes.
            assert_eq!(sym.census.ring, f16.census.ring);
            assert_eq!(sym.census.lookahead, f16.census.lookahead);
            assert_eq!(sym.census.arena, f16.census.arena);
            assert_eq!(sym.census.boundary_state, f16.census.boundary_state);
            assert_eq!(
                Some(sym.resident_layers),
                sym.placement.largest_fitting_layers()
            );
        }

        #[test]
        fn slowlink_moe_iu4_window_scratch_reserves_the_full_window_sidecars() {
            let profile = product_profile_on(SYM);
            let args = window_scratch_args(&profile);
            assert_eq!(args.route, SYM);
            assert_eq!(args.rows, crate::slowlink_admission::S1_WINDOW_ROWS);
            assert_eq!(args.hidden, 2560);
            assert_eq!(args.intermediate, 640);
            // The reserve entry's own byte formula (`reserve_qwen4_moe_sym`: grouping scratch, down
            // sidecar `intermediate / 128 * slots * 72`, gate sidecar) is the census' charge.
            let slots = args.rows * dsp::K_TOP;
            assert_eq!(
                (args.intermediate / 128 * slots * 72) as u64
                    + (args.hidden / 128 * args.rows * 72) as u64,
                profile.census.a4_sidecars
            );
            assert_eq!(
                rdna_compute::scratch::qwen4_moe_group_layout(slots, args.experts).bytes as u64,
                profile.census.group_scatter
            );
        }

        #[test]
        fn slowlink_moe_iu4_receipt_names_the_route_descriptor_tiles_and_verdicts() {
            let kernel = gfx1100_descriptor();
            let receipt = moe_iu4_receipt(&MoeIu4Facts {
                route: SYM,
                requested: true,
                descriptor: Some(kernel),
                source_symmetric: Some(true),
                final_verified: (48, 48),
                a4_sidecar_bytes: 82_575_360,
                window_rows: 16384,
                tiles: [21, 0, 1, 0, 0],
            });
            assert_eq!(receipt["window_route"], "aggregate-sym-iu4");
            assert_eq!(receipt["requested"], true);
            assert_eq!(receipt["source_headers_symmetric"], true);
            assert_eq!(receipt["final_headers_verified"], json!([48u64, 48u64]));
            assert_eq!(receipt["a4_sidecar_bytes"], 82_575_360u64);
            assert_eq!(receipt["window_slot_stride"], 163_840u64);
            let d = &receipt["descriptor"];
            assert_eq!(d["arch"], "gfx1100");
            assert_eq!(d["module"], "qwen4_moe_iu4_sym_pm_gfx1100");
            assert_eq!(d["gate_up"], "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100");
            assert_eq!(d["down"], "qwen4_moe_down_iu4_sym_pm_gfx1100");
            assert_eq!(d["nt"], 1);
            assert_eq!(d["threads"], json!([64, 128]));
            assert_eq!(d["gate_producer"], "mq_rotate_x_i4");
            assert_eq!(d["down_producer"], "mq_i4_a4_r128_k640_gfx1100");
            assert_eq!(d["checker"], "qwen4_moe_sym_check_gfx1100");
            assert_eq!(d["c2"], true);
            assert_eq!(
                receipt["tiles"],
                json!({
                    "aggregate_iu4": 21,
                    "aggregate_wmma": 0,
                    "aggregate_f32_tail": 1,
                    "slab_k2": 0,
                    "k1_or_decode": 0,
                })
            );
            // Every other route reports its own label, a null verdict where no scan ran and no
            // slot stride.
            let f16 = moe_iu4_receipt(&MoeIu4Facts {
                route: F16,
                requested: false,
                descriptor: None,
                source_symmetric: None,
                final_verified: (0, 48),
                a4_sidecar_bytes: 0,
                window_rows: 16384,
                tiles: [0, 21, 1, 0, 0],
            });
            assert_eq!(f16["window_route"], "aggregate");
            assert!(f16["descriptor"].is_null() && f16["source_headers_symmetric"].is_null());
            assert_eq!(f16["window_slot_stride"], 0u64);
            assert_eq!(
                no_moe_iu4()["window_route"],
                dsp::WindowRoute::SlabIu4.label()
            );
        }
    }
}
