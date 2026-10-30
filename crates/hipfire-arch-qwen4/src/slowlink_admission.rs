// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission, placement census and prompt-arena layout of the S1 slow-link
//! layer-major prefill.
//!
//! S1 streams every host-resident routed expert across a narrow PCIe link once
//! per prompt window, layer by layer, instead of once per prefill chunk. This
//! module is the pure policy half: it decides *before any allocation* whether
//! a load may take the S1 profile, sizes everything that profile costs from
//! one layout, and judges request shapes against the frozen profile. It never
//! touches a GPU, so every number here is exercised by CPU tests.
//!
//! `HIPFIRE_QWEN4_SLOWLINK_PREFILL` (`developer.qwen4_slowlink_prefill`, a
//! typed boolean): `0` incumbent (kill switch), `1` lab candidate, unset
//! automatic. The automatic selector admits S1 from 1,537 rows (one row past
//! the 1,536-row tile: the first genuinely multi-chunk request). The
//! candidate (forced or automatic) is selected only when all of these hold,
//! checked in this order, cheapest first:
//!
//! 1. a single-device, whole-model, auto-placed discrete load: no expert
//!    parallelism, bands, hot set, `whole-first`, explicit expert layer count,
//!    `HIPFIRE_QWEN4_EXPERT_STAGE=0` opt-out;
//! 2. a context long enough to hold a multi-chunk window;
//! 3. the measured host link ([`link_verdict`]): discrete and **below**
//!    16 GB/s. A fast link, unified memory and an unmeasured link never select
//!    S1, whatever the selector says;
//! 4. the joint placement ([`project_at_tile`], consumed by [`admit`]): the
//!    purpose's tile ([`purpose_tile_rows`]), the prompt arena, the expert
//!    page store, the route storage and A4 sidecars, the HC arena and the
//!    boundary state are all charged together against the ACTUAL
//!    post-initialization free device memory ([`AdmissionMemory`]). The
//!    placement keeps the largest resident layer count that fits
//!    ([`PlacementBudget::largest_fitting_layers`]);
//!    [`S1_PREFERRED_RESIDENT_LAYERS`] is a preferred count reported as
//!    information, never a floor. Only a base budget that cannot hold the
//!    non-expert weights plus the whole reserve with zero resident layers is
//!    refused, with its exact byte deficit ([`Deficit`]).
//!
//! Probe-free checks run first; only a load that passes them measures the host
//! link, and a load that fails any step is the incumbent byte for byte with a
//! zero S1 charge ([`crate::expert_residency::PlacementCharges::slowlink`] is `0`).
//! Sealed capability gates (panels, layout, codec, kernel, load decline and the
//! preflight dry run) refuse before materialization; only an actually admitted
//! sealed expert-slab or aggregate layout and its real native kernels run S1.

use crate::host_link::{HostLink, NARROW_LINK_GBPS};
use crate::Qwen4Config;
pub use hipfire_dispatch::pipeline::slowlink_prefill::WindowRoute;
use std::fmt;

/// Developer variable of the selector (`developer.qwen4_slowlink_prefill`).
pub const SLOWLINK_PREFILL_ENV: &str = "HIPFIRE_QWEN4_SLOWLINK_PREFILL";

/// LARGEST legal S1 mixer tile, in rows: the upper bound of
/// [`legal_tile_rungs`] and of the named short-request incumbent bypass
/// ([`S1_MIN_FORCED_REQUEST_ROWS`]). It is NOT the admitted tile: the load's
/// forward scratch is sized from, and the forward runs, the
/// [`SlowlinkProfile::tile_rows`] that [`admit`] chose from the load's
/// [`AdmissionPurpose`] ([`purpose_tile_rows`]). Never the 8192-row incumbent
/// chunk.
pub const S1_TILE_ROWS: usize = 1536;

/// The tile a [`AdmissionPurpose::Product`] load admits: the largest legal
/// rung that keeps the preferred [`S1_PREFERRED_RESIDENT_LAYERS`] resident
/// beside the whole charged reserve on the j1778 reference fixture (Main's
/// choice from the joint frontier). Every state, headroom, greedy and product
/// claim runs on it.
pub const S1_PRODUCT_TILE_ROWS: usize = 768;

/// The tile an [`AdmissionPurpose::PrimitiveSelftest`] load admits: the
/// largest legal rung that still funds the primitive self-test diagnostic
/// inside the reserve with the preferred [`S1_PREFERRED_RESIDENT_LAYERS`]
/// resident on the j1778 reference fixture (kernel coverage only; never a
/// product claim).
pub const S1_PRIMITIVE_TILE_ROWS: usize = 512;

/// Prompt rows one S1 window admits. A longer prompt runs several windows
/// (each its own expert sweep) and is receipted as such; it never claims one
/// sweep per prompt.
///
/// Whole product tiles: the planner closes a window before the tile that would
/// pass it, so a window of `S1_WINDOW_TILES` 768-row tiles carries every row.
/// 21 tiles (16,128 rows) left a 32,780-row prompt a 524-row third window that
/// still paid a full expert sweep; 22 tiles (16,896 rows) makes it two. The
/// window is also a whole number of the 1536-row and 512-row tiles (11 and 33).
pub const S1_WINDOW_TILES: usize = 22;
pub const S1_WINDOW_ROWS: usize = S1_WINDOW_TILES * S1_PRODUCT_TILE_ROWS;

/// Device memory an [`admit_grown`] profile leaves free beyond every charge,
/// the 512 MiB safety margin and its own
/// [`SlowlinkProfile::device_snapshot_budget_bytes`]: the quote it is judged on
/// is the actual free minus this, so the session cache's device snapshots
/// never reach it either. A grown load therefore never leaves less than this
/// (plus the margin) free after placement.
pub const S1_GROWN_FREE_HOLD_BYTES: u64 = 1 << 30;

/// Most whole tiles an [`admit_grown`] window ever holds (49,152 rows at the
/// 768-row product tile). The window's HC-stream elements stay at about
/// 5e8, a quarter of the 32-bit element index space the row-major kernels
/// address, however much free memory a larger card leaves; nothing above
/// this size has been run.
pub const S1_GROWN_WINDOW_MAX_TILES: usize = 64;

/// Fewest request rows S1 runs when forced: one row past a tile, the first
/// genuinely multi-chunk request (a request of at most one tile is a named
/// bypass). The automatic selector starts one row past the ADMITTED tile
/// ([`qualify_request`]): on a 768-row-tile load a 769..=1536-row request is
/// already two incumbent chunks, each re-streaming every host layer's experts.
pub const S1_MIN_FORCED_REQUEST_ROWS: usize = S1_TILE_ROWS + 1;

/// Preferred resident trunk layers of an S1 load beside its own charges.
/// INFORMATION ONLY: placement keeps the largest count that fits
/// ([`PlacementBudget::largest_fitting_layers`]), which may be fewer (down to
/// zero) or more; no kernel or correctness gate needs this many.
pub const S1_PREFERRED_RESIDENT_LAYERS: usize = 10;

/// Developer switch (`developer.qwen4_slowlink_resident_layers`, read through
/// the config snapshot) choosing WHICH trunk layers are resident on an S1
/// load; the COUNT stays [`admit`]'s decision. Unset or `front` is the
/// leading layers `0..N` (the incumbent placement); `spread` is the
/// evenly-spaced set of [`s1_resident_layer_set`]; otherwise a comma list of
/// exactly `N` distinct layer indices below the trunk layer count.
pub const S1_RESIDENT_LAYERS_ENV: &str = "HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS";

/// The `N` resident trunk layers of a `T`-layer model under
/// [`S1_RESIDENT_LAYERS_ENV`], sorted ascending. `spread` puts the `k`-th of
/// `N` at `floor(k*T/N) + floor(T/(2N)) + 1`, clamped to `T-1` (T=48, N=9 gives
/// 3, 8, 13, 19, 24, 29, 35, 40, 45); `N == T` is every layer. Any invalid
/// value, or `N > T`, is an `Err` naming the switch.
pub fn s1_resident_layer_set(
    num_hidden_layers: usize,
    resident: usize,
) -> Result<Vec<usize>, String> {
    let raw = hipfire_config::developer_var(S1_RESIDENT_LAYERS_ENV).ok();
    resolve_resident_layer_set(raw.as_deref(), num_hidden_layers, resident)
}

/// Pure body of [`s1_resident_layer_set`] for unit tests.
fn resolve_resident_layer_set(
    raw: Option<&str>,
    total: usize,
    resident: usize,
) -> Result<Vec<usize>, String> {
    let invalid = |why: String| format!("{S1_RESIDENT_LAYERS_ENV}: {why}");
    if resident > total {
        return Err(invalid(format!(
            "{resident} resident layers exceed the {total} trunk layers"
        )));
    }
    let Some(value) = raw.map(str::trim) else {
        return Ok((0..resident).collect());
    };
    if value.eq_ignore_ascii_case("front") {
        return Ok((0..resident).collect());
    }
    if value.eq_ignore_ascii_case("spread") {
        if resident == total {
            return Ok((0..total).collect());
        }
        let offset = if resident == 0 {
            0
        } else {
            total / (2 * resident) + 1
        };
        let set: Vec<usize> = (0..resident)
            .map(|k| (k * total / resident + offset).min(total - 1))
            .collect();
        if set.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(format!(
                "spread of {resident} layers over {total} is not strictly ascending: {set:?}"
            )));
        }
        return Ok(set);
    }
    let mut set = Vec::with_capacity(resident);
    for entry in value.split(',') {
        let entry = entry.trim();
        let layer: usize = entry.parse().map_err(|_| {
            invalid(format!(
                "entry {entry:?} of {value:?} is not a layer index (expected `front`, `spread` or a comma list of {resident} layer indices)"
            ))
        })?;
        if layer >= total {
            return Err(invalid(format!(
                "layer {layer} is not below the {total} trunk layers"
            )));
        }
        set.push(layer);
    }
    if set.len() != resident {
        return Err(invalid(format!(
            "{value:?} lists {} layers but the admitted resident count is {resident}",
            set.len()
        )));
    }
    set.sort_unstable();
    if let Some(pair) = set.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid(format!(
            "layer {} is listed twice in {value:?}",
            pair[0]
        )));
    }
    Ok(set)
}

/// Whether a [`S1_RESIDENT_LAYERS_ENV`] value is anything but the incumbent
/// leading-layers placement (unset or `front`, after trimming). An empty or
/// whitespace value is an override, and [`s1_resident_layer_set`] rejects it.
pub fn resident_layer_switch_is_override(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .is_some_and(|value| !value.eq_ignore_ascii_case("front"))
}

/// Routed experts one transport page holds.
pub const EXPERTS_PER_PAGE: usize = 32;

/// Dedicated double-buffered active pages (the ring).
pub const ACTIVE_PAGES: usize = 2;

/// Grouped rows one expert panel computes at most (`PanelLimits`).
pub const PANEL_MAX_GROUPED_ROWS: usize = 2048;

/// New boundary snapshots of the session fixed state one window may hold (the
/// entry snapshot is the incumbent's own). A request with more session cuts is a bypass.
pub const S1_MAX_BOUNDARY_SNAPSHOTS: usize = 4;

/// Generic occupancy budget charged on every admitted S1 profile before the
/// 512 MiB safety margin: the UNATTRIBUTED load/runtime device occupancy the
/// loaded model showed beyond every modelled census term (j1761: 847,209,812 B
/// incremental after the pre-load baseline, 887,055,700 B in total). It is a
/// conservative reservation, not a driver measurement and not an allocation:
/// nothing allocates it, and no arch, card or host name selects it.
pub const S1_RUNTIME_OVERHEAD_BYTES: u64 = 1 << 30;

/// Byte alignment of every arena span and page region.
pub const S1_ALIGN: u64 = 256;

fn align(bytes: u64) -> Result<u64, String> {
    bytes
        .checked_add(S1_ALIGN - 1)
        .map(|bytes| bytes / S1_ALIGN * S1_ALIGN)
        .ok_or_else(|| "slow-link prefill byte alignment overflows".to_string())
}

fn checked_mul(a: u64, b: u64, what: &str) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("slow-link prefill {what} overflows"))
}

fn checked_add(a: u64, b: u64, what: &str) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("slow-link prefill {what} overflows"))
}

/// The selector value: `0` is [`Self::Off`], `1` is [`Self::Force`], unset is
/// [`Self::Auto`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlowlinkMode {
    Off,
    Force,
    Auto,
}

impl SlowlinkMode {
    /// From the typed-boolean selector value (`None` when unset).
    pub fn from_typed(value: Option<bool>) -> Self {
        match value {
            Some(false) => Self::Off,
            Some(true) => Self::Force,
            None => Self::Auto,
        }
    }

    /// The process-snapshot selector (`developer.qwen4_slowlink_prefill`).
    pub fn from_env() -> Self {
        Self::from_typed(hipfire_config::developer_typed_bool(SLOWLINK_PREFILL_ENV))
    }
}

/// Whether the NPU trunk engine is selected for `arch` under `mode`: the same
/// resolution the forward uses ([`crate::npu_trunk::NpuTrunkMode::resolve`]).
/// S1 declines a selected NPU before any reserve, so a pair route never meets
/// the layer-major schedule.
pub fn npu_trunk_selected(arch: &str, mode: crate::npu_trunk::NpuTrunkMode) -> bool {
    !mode.resolve(arch).is_off()
}

/// What the loader knows about a load before it sizes any placement.
#[derive(Clone, Copy, Debug)]
pub struct LoadShape<'a> {
    pub arch: &'a str,
    pub unified_memory: bool,
    /// Expert-parallel mesh, or an agreed expert-parallel layer split.
    pub expert_parallel: bool,
    pub pipeline_band: bool,
    /// Recipe, layout and kernel admission is by sealed verification
    /// ([`window_route_of`], the panel/codec/kernel preflight and
    /// [`crate::slowlink_prefill::bind_at_load`]), never by recipe name.
    pub routed_experts: bool,
    pub expert_keep: bool,
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` set explicitly.
    pub explicit_vram_layers: bool,
    /// A ranked hot-set sidecar applies.
    pub hotset_sidecar: bool,
    pub whole_first: bool,
    /// `HIPFIRE_QWEN4_EXPERT_STAGE=0`: the staged-transport opt-out.
    pub stage_opt_out: bool,
    /// The NPU trunk engine is selected (`HIPFIRE_QWEN4_NPU_TRUNK` resolved for
    /// this arch is not `off`: `auto` resolves to off outside gfx1151, an
    /// explicit `on`/`emulate` selects it anywhere). Read from the config
    /// snapshot ([`npu_trunk_selected`]); no device is opened or probed.
    pub npu_trunk: bool,
    /// Raw `HIPFIRE_PREFILL_CHUNK_ROWS`, when set.
    pub chunk_rows_env: Option<&'a str>,
    pub max_seq: usize,
}

/// Exact byte shortfall of a base budget: the non-expert weights plus the
/// whole charged reserve do not fit the actual free memory even with zero
/// resident layers (every routed expert host-streamed). Counts below
/// [`S1_PREFERRED_RESIDENT_LAYERS`] are never a deficit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deficit {
    /// The actual memory the placement was sized from.
    pub memory: AdmissionMemory,
    pub non_expert: u64,
    /// Backed device bytes of ONE resident layer (payload plus its own
    /// granule slack): the cost [`Self::required_layers`] multiplies.
    pub resident_layer_bytes: u64,
    /// The complete charged reserve (base + margin + S1 census + granule).
    pub reserve: u64,
    /// Resident layers the deficit is quoted for: 0, the base budget.
    pub required_layers: usize,
    /// `non_expert + reserve + required_layers * resident_layer_bytes`
    /// beyond `memory.free_bytes`.
    pub deficit_bytes: u64,
}

/// Why a load or request does not take S1. Every variant is an incumbent
/// route chosen before any S1 allocation or state mutation.
#[derive(Clone, Debug, PartialEq)]
pub enum SlowlinkDecline {
    /// `HIPFIRE_QWEN4_SLOWLINK_PREFILL=0`.
    KillSwitch,
    UnifiedMemory,
    Unsupported(&'static str),
    /// `HIPFIRE_PREFILL_CHUNK_ROWS` asks for a chunk other than the admitted S1
    /// tile (`tile`: the purpose's tile, `None` before admission when the value
    /// is not a legal tile at all).
    ExplicitChunk {
        rows: usize,
        tile: Option<usize>,
    },
    ShortContext {
        max_seq: usize,
    },
    /// The window (`min(S1_WINDOW_ROWS, max_seq)` rows) is smaller than the route's worst-case
    /// cohort bound: the dispatch layout needs `k1_rows + tail_rows <= rows`
    /// ([`check_window_cohort`]). An incumbent decline, never a capacity refusal: the bound is
    /// the route's own constants, so no free memory makes the window legal.
    WindowTooSmall {
        rows: usize,
        k1_rows: usize,
        tail_rows: usize,
    },
    /// Measured discrete link at or above 16 GB/s.
    FastLink {
        gbps: f64,
    },
    /// The link probe failed or returned an unusable figure.
    UnmeasuredLink,
    /// Every routed expert already fits: there is no host layer to stream.
    NoHostLayers {
        resident_layers: usize,
    },
    Deficit(Deficit),
    Geometry(String),
}

impl SlowlinkDecline {
    /// A capacity refusal of an otherwise eligible load (the forced selector
    /// fails the load on it instead of silently running the incumbent).
    pub fn is_capacity(&self) -> bool {
        matches!(self, Self::Deficit(_) | Self::Geometry(_))
    }
}

impl fmt::Display for SlowlinkDecline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const MIB: u64 = 1 << 20;
        match self {
            Self::KillSwitch => write!(f, "{SLOWLINK_PREFILL_ENV}=0"),
            Self::UnifiedMemory => write!(f, "unified memory has no link to stream over"),
            Self::Unsupported(what) => write!(f, "unsupported mode: {what}"),
            Self::ExplicitChunk { rows, tile: None } => write!(
                f,
                "HIPFIRE_PREFILL_CHUNK_ROWS={rows} is not a legal S1 tile ({S1_MIN_TILE_ROWS}..={S1_TILE_ROWS} rows in 256-row steps)"
            ),
            Self::ExplicitChunk {
                rows,
                tile: Some(tile),
            } => write!(
                f,
                "HIPFIRE_PREFILL_CHUNK_ROWS={rows} conflicts with the admitted {tile}-row tile"
            ),
            Self::ShortContext { max_seq } => write!(
                f,
                "context {max_seq} is shorter than the {S1_MIN_FORCED_REQUEST_ROWS}-row minimum multi-chunk window"
            ),
            Self::WindowTooSmall {
                rows,
                k1_rows,
                tail_rows,
            } => write!(
                f,
                "window too small for its cohort rows: rows {rows} < k1 {k1_rows} + tail {tail_rows}"
            ),
            Self::FastLink { gbps } => write!(
                f,
                "host link {gbps:.1} GB/s is not narrow (below {NARROW_LINK_GBPS} GB/s)"
            ),
            Self::UnmeasuredLink => write!(f, "host link is unmeasured (never selects S1)"),
            Self::NoHostLayers { resident_layers } => write!(
                f,
                "{resident_layers} resident layers hold every routed expert: nothing streams"
            ),
            Self::Deficit(d) => write!(
                f,
                "the base placement does not fit even with {} resident layers: {} MiB actual free, \
                 {} MiB non-expert, {} MiB reserve, {} MiB per backed layer; short by {} bytes ({} MiB)",
                d.required_layers,
                d.memory.free_bytes / MIB,
                d.non_expert / MIB,
                d.reserve / MIB,
                d.resident_layer_bytes / MIB,
                d.deficit_bytes,
                d.deficit_bytes / MIB
            ),
            Self::Geometry(error) => write!(f, "{error}"),
        }
    }
}

/// The explicit `HIPFIRE_PREFILL_CHUNK_ROWS` value rounded down to 256-row
/// tiles, `None` when unset, unparsable or below one tile (the same reading
/// the chunk policy uses).
fn explicit_chunk_rows(env: Option<&str>) -> Option<usize> {
    env.and_then(|value| value.trim().parse::<usize>().ok())
        .map(|rows| rows / 256 * 256)
        .filter(|&rows| rows >= 256)
}

/// Probe-free load checks: everything that can refuse S1 without measuring
/// the host link. Order is part of the contract (cheapest and most absolute
/// first), so an automatic or killed selector never reaches the rest.
pub fn preflight(mode: SlowlinkMode, shape: &LoadShape<'_>) -> Result<(), SlowlinkDecline> {
    if mode == SlowlinkMode::Off {
        return Err(SlowlinkDecline::KillSwitch);
    }
    // No arch allowlist, recipe-name or raw offload-exec gate: only an actually
    // admitted sealed layout/panel/codec/kernel verdict admits, and the S1
    // runtime declines unsupported modes before materialization. The unified
    // memory check stays: there is no link to stream over.
    let _ = shape.arch;
    if shape.unified_memory {
        return Err(SlowlinkDecline::UnifiedMemory);
    }
    let unsupported = [
        (shape.expert_parallel, "expert parallelism"),
        (shape.pipeline_band, "pipeline parallelism"),
        (!shape.routed_experts, "no routed experts"),
        (shape.expert_keep, "expert-keep artifact"),
        (
            shape.explicit_vram_layers,
            "explicit expert VRAM layer count",
        ),
        (shape.hotset_sidecar, "ranked expert hot set"),
        (shape.whole_first, "whole-first placement policy"),
        (shape.stage_opt_out, "HIPFIRE_QWEN4_EXPERT_STAGE=0 opt-out"),
        (shape.npu_trunk, "NPU trunk selected"),
    ];
    if let Some((_, what)) = unsupported.into_iter().find(|(hit, _)| *hit) {
        return Err(SlowlinkDecline::Unsupported(what));
    }
    // An explicit chunk must at least be a legal S1 tile; it must then equal
    // the PURPOSE's tile, which is known only once admission has run
    // ([`select_profile`] checks that).
    if let Some(rows) = explicit_chunk_rows(shape.chunk_rows_env) {
        if !legal_tile_rungs().contains(&rows) {
            return Err(SlowlinkDecline::ExplicitChunk { rows, tile: None });
        }
    }
    if shape.max_seq < S1_MIN_FORCED_REQUEST_ROWS {
        return Err(SlowlinkDecline::ShortContext {
            max_seq: shape.max_seq,
        });
    }
    Ok(())
}

/// The measured link verdict: the link's GB/s when it is discrete and narrow,
/// else the decline. Fast links, unified memory and an unmeasured link never
/// select S1.
pub fn link_verdict(link: HostLink) -> Result<f64, SlowlinkDecline> {
    match link {
        HostLink::Discrete { gbps } if link.is_narrow_discrete() => Ok(gbps),
        HostLink::Discrete { gbps } => Err(SlowlinkDecline::FastLink { gbps }),
        HostLink::Unified => Err(SlowlinkDecline::UnifiedMemory),
        HostLink::Unmeasured => Err(SlowlinkDecline::UnmeasuredLink),
    }
}

/// The window route of a manifest, from the REAL artifact layout and never a
/// default: a slab layer arena in the manifest is the sealed expert-slab route
/// ([`WindowRoute::SlabIu4`], A4 sidecars); stacked per-layer expert tensors
/// are the aggregate route ([`WindowRoute::Aggregate`], window F16
/// activations). The dispatch side refuses any tile whose own sealed route
/// differs from this declaration, so a wrong declaration cannot run.
pub fn window_route_of(weights: &[hipfire_runtime::weight_manifest::WeightEntry]) -> WindowRoute {
    if weights
        .iter()
        .any(|entry| entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX))
    {
        WindowRoute::SlabIu4
    } else {
        WindowRoute::Aggregate
    }
}
/// The Qwen4 MoE programs' recipe keeps the BF16 storage boundaries the symmetric IU4 kernels
/// implement (`MoeRecipe::SoftmaxGatedShared { bf16_round_trip: true, .. }` in `program.rs` and
/// the forward's own params): the recipe fact of [`SymIu4RouteFacts`]. Every tile call still
/// re-checks its own recipe in the dispatch crate's admission.
pub const QWEN4_MOE_RECIPE_BF16_ROUND_TRIP: bool = true;

/// The terms of [`resolve_window_route`] besides the artifact layout. All four are decided
/// BEFORE census and allocation; none is read again at execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SymIu4RouteFacts {
    /// The device has a symmetric IU4 MoE kernel descriptor for MQ4 experts
    /// (`Gpu::qwen4_moe_sym_kernel`).
    pub descriptor: bool,
    /// `Gpu::qwen4_moe_sym_iu4_requested`: the typed flag requested the route (unset follows the
    /// arch descriptor's `default_requested`, true on gfx1100, gfx1151 and gfx1201, so the S1 window
    /// and the regular prefill agree; `HIPFIRE_QWEN4_MOE_SYM_IU4=0` kills), the C2 candidate set
    /// holds and a descriptor exists.
    pub requested: bool,
    /// The MoE recipe's BF16 round trip ([`QWEN4_MOE_RECIPE_BF16_ROUND_TRIP`]).
    pub recipe_bf16_round_trip: bool,
    /// Source-byte symmetry of EVERY routed trunk layer's QT44/QT53 headers
    /// (`artifact::routed_headers_symmetric`); `None` until the scan ran, which it does only when
    /// the three other terms and the layout hold ([`Self::scan_needed`]).
    pub source_symmetric: Option<bool>,
}

impl SymIu4RouteFacts {
    /// The device terms of `gpu` (no source scan).
    pub fn of_gpu(gpu: &rdna_compute::Gpu) -> Self {
        Self {
            descriptor: gpu.qwen4_moe_sym_kernel().is_some(),
            requested: gpu.qwen4_moe_sym_iu4_requested(),
            recipe_bf16_round_trip: QWEN4_MOE_RECIPE_BF16_ROUND_TRIP,
            source_symmetric: None,
        }
    }

    /// Whether the source header scan can change the verdict: the artifact layout is the
    /// stacked aggregate one and the other three terms hold. Otherwise the layout route is the
    /// answer and no header byte is read.
    pub fn scan_needed(&self, layout: WindowRoute) -> bool {
        layout == WindowRoute::Aggregate
            && self.descriptor
            && self.requested
            && self.recipe_bf16_round_trip
    }
}

/// The window route of a load, resolved from the pure layout classifier
/// ([`window_route_of`]) and the terms of [`SymIu4RouteFacts`]: [`WindowRoute::AggregateSymIu4`]
/// iff the layout is the stacked aggregate one AND the device has a descriptor AND the route was
/// requested AND every routed trunk layer's source headers are symmetric AND the recipe keeps the
/// BF16 boundaries; otherwise the layout route (the whole profile stays on the F16 aggregate
/// route when any layer fails). Pure.
pub fn resolve_window_route(layout: WindowRoute, facts: &SymIu4RouteFacts) -> WindowRoute {
    if facts.scan_needed(layout) && facts.source_symmetric == Some(true) {
        WindowRoute::AggregateSymIu4
    } else {
        layout
    }
}

/// A resolved window route and the source-header verdict behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedWindowRoute {
    pub route: WindowRoute,
    /// `Some` only when the scan ran ([`SymIu4RouteFacts::scan_needed`]).
    pub source_symmetric: Option<bool>,
}

/// [`resolve_window_route`] running `scan` (the artifact's routed-header scan) once, and only
/// when [`SymIu4RouteFacts::scan_needed`]. A scan error is the caller's decline, never a guess.
pub fn resolve_window_route_with_scan(
    layout: WindowRoute,
    mut facts: SymIu4RouteFacts,
    scan: impl FnOnce() -> Result<bool, String>,
) -> Result<ResolvedWindowRoute, String> {
    if facts.source_symmetric.is_none() && facts.scan_needed(layout) {
        facts.source_symmetric = Some(scan()?);
    }
    Ok(ResolvedWindowRoute {
        route: resolve_window_route(layout, &facts),
        source_symmetric: facts.source_symmetric,
    })
}

/// Phase 2 of the route proof: the FINAL bound pointer tables' header verdicts
/// (`Qwen4GpuForward`'s load-time `verify_symmetric` loop over every routed trunk layer's
/// device-visible tables, host-mapped S1 layers included) against the charged route. An
/// [`WindowRoute::AggregateSymIu4`] profile needs `verified == total > 0`; anything else is a
/// load error BEFORE serving (the census, placement and scratch were charged for this route, so
/// the load never silently switches to a differently budgeted one). Every other route passes.
pub fn check_sym_iu4_final_headers(
    route: WindowRoute,
    verified: usize,
    total: usize,
) -> Result<(), String> {
    if route != WindowRoute::AggregateSymIu4 || (total > 0 && verified == total) {
        return Ok(());
    }
    Err(format!(
        "qwen4: the slow-link symmetric IU4 window route ({}) was admitted from the artifact's \
         source headers, but the loaded final expert tables verified only {verified}/{total} \
         routed MoE layers symmetric; declining the load before serving (the charged census is \
         for this route and no differently budgeted route is substituted)",
        route.label()
    ))
}
/// Whether the manifest's sourced routed-expert dtypes are a source-priced
/// mixed MQ2-R2 / E8P pair: one gate/up or down projection declares an MQ2-I4
/// slab dtype while another declares E8P-I4 (either R256 or R128 rung). Such an
/// artifact never reaches the sealed panel/codec/kernel verdict: S1 declines
/// pre-mutation even when incumbent kernels exist for each dtype alone.
/// Aggregate (packed, non-slab) dtypes and uniform cold codecs stay untouched.
pub fn mixed_source_priced_mq2r2_e8p(
    weights: &[hipfire_runtime::weight_manifest::WeightEntry],
    bytes_of: &dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
) -> bool {
    use rdna_compute::DType;
    let mut saw_mq2 = false;
    let mut saw_e8p = false;
    for entry in weights {
        if entry.residency.is_external() {
            continue;
        }
        if bytes_of(entry).is_none() {
            continue;
        }
        let gate_up = entry.name.ends_with(".mlp.experts.gate_up_proj");
        let down = entry.name.ends_with(".mlp.experts.down_proj");
        if !(gate_up || down) {
            continue;
        }
        match entry.dtype {
            DType::MQ2I4G128R256 | DType::MQ2I4G128R128 => saw_mq2 = true,
            DType::E8PI4G128R256 | DType::E8PI4G128R128 => saw_e8p = true,
            _ => {}
        }
        if saw_mq2 && saw_e8p {
            return true;
        }
    }
    false
}

/// The model dimensions the S1 layouts are sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelGeometry {
    pub hidden: usize,
    pub hc_count: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    /// The routed lowering the window's K2 panels run, fixed by the artifact's
    /// declared layout ([`window_route_of`]). Every route-dependent byte (the
    /// A4 sidecars, the window F16 activations, the F32 tails) is read from
    /// the dispatch layout of THIS route.
    pub route: WindowRoute,
}

impl ModelGeometry {
    /// A MoE profile's dimensions on the window `route` of its artifact
    /// ([`window_route_of`]); a dense profile has nothing to stream.
    pub fn from_config(config: &Qwen4Config, route: WindowRoute) -> Result<Self, String> {
        match crate::gpu_forward::program_dims(config) {
            dims @ crate::program::Qwen4ProgramDims {
                ffn:
                    crate::program::Qwen4FfnDims::Moe {
                        num_experts,
                        experts_per_token,
                        intermediate,
                        shared_intermediate: shared,
                    },
                ..
            } => Ok(Self {
                hidden: dims.hidden,
                hc_count: dims.hc_count,
                num_experts,
                experts_per_token,
                moe_intermediate: intermediate,
                shared_intermediate: shared,
                route,
            }),
            _ => Err("slow-link prefill needs a routed-expert (MoE) profile".to_string()),
        }
    }

    fn wide(&self) -> Result<u64, String> {
        checked_mul(
            self.hc_count as u64,
            self.hidden as u64,
            "hyper-connection width",
        )
    }
}

/// Copy/compute interval records the receipt timeline keeps per window
/// (`PanelPoolSpec::timeline_records`): 48 trunk layers x 16 pages x 2
/// (current and lookahead tags), with room. Host memory and pooled HIP events
/// only; no device bytes.
pub const S1_TIMELINE_RECORDS: usize = 2048;

/// Whole-layer compute marks the receipt timeline keeps per window.
pub const S1_TIMELINE_LAYERS: usize = 64;

/// The transport page pool of the profile: the geometry that
/// `gpu_forward::panel_stage::PanelPoolSpec` allocates, built here so the
/// census and the pool are ONE reserve (`PanelPoolSpec::reserve` is the single
/// authority for page, table and staging bytes). The pool's page payload is
/// the per-page upper bound [`Self::payload_bytes`]: the pool is constructed from
/// exactly this number, and attach refuses a source set whose real
/// `required_payload_bytes` exceeds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelPageLayout {
    pub experts_per_page: usize,
    pub num_experts: usize,
    /// Pages of one full layer (the lookahead store).
    pub pages_per_layer: usize,
    pub active_pages: usize,
    pub gate_up_expert_bytes: u64,
    pub down_expert_bytes: u64,
    /// Upper bound of one page's packed payload before the tail pad:
    /// `experts_per_page * (gate_up + down)` plus the worst-case run slack
    /// (`2 * experts_per_page` unmerged extents, each at most
    /// `2 * 255` bytes of 256-byte alignment and source-address congruence).
    pub payload_bytes: u64,
    /// Bytes of one page (payload, tail pad, rounded to 256).
    pub page_bytes: u64,
    /// Rebased gate/up and down pointer tables of one page (device).
    pub table_bytes_per_page: u64,
    /// Pinned host staging of every page's tables.
    pub host_staging_bytes: u64,
    pub timeline_records: usize,
    pub timeline_layers: usize,
}

impl PanelPageLayout {
    /// From one layer's packed gate/up and down payload bytes (the largest
    /// trunk layer's). Every expert's span must be equal and a whole number
    /// of bytes: a non-uniform layer is refused, not rounded.
    pub fn new(
        num_experts: usize,
        gate_up_layer_bytes: u64,
        down_layer_bytes: u64,
    ) -> Result<Self, String> {
        if num_experts == 0 {
            return Err("slow-link prefill page layout has no experts".to_string());
        }
        let experts = num_experts as u64;
        if gate_up_layer_bytes == 0
            || down_layer_bytes == 0
            || gate_up_layer_bytes % experts != 0
            || down_layer_bytes % experts != 0
        {
            return Err(format!(
                "slow-link prefill needs uniform per-expert payloads: {gate_up_layer_bytes} gate/up \
                 and {down_layer_bytes} down bytes do not split over {num_experts} experts"
            ));
        }
        let gate_up_expert_bytes = gate_up_layer_bytes / experts;
        let down_expert_bytes = down_layer_bytes / experts;
        let to_usize = |bytes: u64| {
            usize::try_from(bytes)
                .map_err(|_| "slow-link prefill expert bytes overflow".to_string())
        };
        let spec = crate::gpu_forward::panel_stage::PanelPoolSpec::uniform(
            ACTIVE_PAGES,
            num_experts.div_ceil(EXPERTS_PER_PAGE),
            EXPERTS_PER_PAGE,
            num_experts,
            to_usize(gate_up_expert_bytes)?,
            to_usize(down_expert_bytes)?,
            S1_TIMELINE_RECORDS,
            S1_TIMELINE_LAYERS,
        )
        .map_err(|error| format!("slow-link prefill page pool: {error:?}"))?;
        let reserve = spec
            .reserve()
            .ok_or_else(|| "slow-link prefill page pool bytes overflow".to_string())?;
        Ok(Self {
            experts_per_page: EXPERTS_PER_PAGE,
            num_experts,
            pages_per_layer: spec.lookahead_pages,
            active_pages: ACTIVE_PAGES,
            gate_up_expert_bytes,
            down_expert_bytes,
            payload_bytes: spec.payload_bytes as u64,
            page_bytes: reserve.page_bytes as u64,
            table_bytes_per_page: (reserve.tables / reserve.pages) as u64,
            host_staging_bytes: reserve.staging as u64,
            timeline_records: S1_TIMELINE_RECORDS,
            timeline_layers: S1_TIMELINE_LAYERS,
        })
    }

    /// The pool the forward constructs (`PanelPool::new(gpu, layout.pool_spec()?)`):
    /// the same `PanelPoolSpec::uniform` the census read.
    pub(crate) fn pool_spec(
        &self,
    ) -> Result<crate::gpu_forward::panel_stage::PanelPoolSpec, String> {
        crate::gpu_forward::panel_stage::PanelPoolSpec::uniform(
            self.active_pages,
            self.pages_per_layer,
            self.experts_per_page,
            self.num_experts,
            usize::try_from(self.gate_up_expert_bytes)
                .map_err(|_| "slow-link prefill expert bytes overflow".to_string())?,
            usize::try_from(self.down_expert_bytes)
                .map_err(|_| "slow-link prefill expert bytes overflow".to_string())?,
            self.timeline_records,
            self.timeline_layers,
        )
        .map_err(|error| format!("slow-link prefill page pool: {error:?}"))
    }

    /// Active plus lookahead pages: the whole admitted working set.
    pub fn total_pages(&self) -> usize {
        self.active_pages + self.pages_per_layer
    }

    /// Device bytes of one page with its pointer tables.
    pub fn device_page_bytes(&self) -> Result<u64, String> {
        checked_add(
            self.page_bytes,
            self.table_bytes_per_page,
            "device page bytes",
        )
    }

    /// Device bytes of the double-buffered active ring.
    pub fn ring_bytes(&self) -> Result<u64, String> {
        checked_mul(
            self.active_pages as u64,
            self.device_page_bytes()?,
            "active ring bytes",
        )
    }

    /// Device bytes of the one-layer-ahead lookahead store.
    pub fn lookahead_bytes(&self) -> Result<u64, String> {
        checked_mul(
            self.pages_per_layer as u64,
            self.device_page_bytes()?,
            "lookahead store bytes",
        )
    }

    /// Ring plus lookahead: `PanelPoolReserve::device_total`.
    pub fn total_bytes(&self) -> Result<u64, String> {
        checked_add(
            self.ring_bytes()?,
            self.lookahead_bytes()?,
            "page store bytes",
        )
    }

    /// Pooled HIP events the timeline keeps (host handles, no device bytes).
    pub fn timeline_events(&self) -> usize {
        4 * self.timeline_records + 2 * self.timeline_layers
    }
}

/// The spans of the prompt arena, in layout order. The arena holds only what
/// the architecture (not the dispatch crate) owns across a layer: the
/// hyper-connection streams. The saved prepared input, top-k ids/weights,
/// route maps and widened accumulator are the dispatch crate's
/// `PromptRouteStorage` and are charged from its own layout
/// ([`route_storage_bytes`]), never duplicated here. Per-tile HC read/write
/// gates, normalised rows and the mixed row are tile-local scratch the 1,536-row
/// forward already owns (`HyperWriteOp` recomputes its gates from the saved
/// streams and the layer's combined output), so no gate is saved per window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaField {
    /// Four hyper-connection branch streams of every window row, each tile
    /// packed at its own tag width (BF16 from `QWEN4_F16_WMMA_MIN_TOKENS` rows
    /// up, F32 below): `(window + S1_F32_STREAM_ROWS + tile) * wide * 2` bytes,
    /// the last term slack for F32-typed validation views.
    HcStreams,
}

/// Every [`ArenaField`], in layout order.
pub const ARENA_FIELDS: [ArenaField; 1] = [ArenaField::HcStreams];

impl ArenaField {
    pub fn name(self) -> &'static str {
        match self {
            Self::HcStreams => "HC streams",
        }
    }
}

/// One arena span: [`S1_ALIGN`]-aligned offset, exact payload bytes (the
/// next span starts at the aligned end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSpan {
    pub field: ArenaField,
    pub offset: u64,
    pub bytes: u64,
}

impl ArenaSpan {
    /// One past the last payload byte.
    pub fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}

/// The architecture-owned prompt arena of one window: a single device
/// allocation of [`Self::total_bytes`], sliced by [`Self::span`]. The census
/// charges exactly this number and the allocator allocates exactly this number,
/// from this one function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptArenaLayout {
    pub rows: usize,
    pub tile_rows: usize,
    pub tiles: usize,
    pub spans: [ArenaSpan; ARENA_FIELDS.len()],
    pub total_bytes: u64,
}

/// Largest tile (in rows) whose streams stay F32: one row below the BF16 stream
/// predicate's threshold ([`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`]).
/// A change of that threshold changes the arena, never silently narrows it.
pub const S1_F32_TILE_ROWS_MAX: usize = rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS - 1;

/// Rows of one window whose tiles can be F32 under the BF16-stream predicate:
/// the window's final tile, one tile per [`S1_MAX_BOUNDARY_SNAPSHOTS`] session
/// cut segment boundary and the resumed entry head, each at most
/// [`S1_F32_TILE_ROWS_MAX`] rows: `(4 + 2) * 511 = 3066`. A tile of 512 or more
/// rows is BF16 on the production predicate; a plan that needs more F32 rows
/// bypasses before any mutation.
pub const S1_F32_STREAM_ROWS: usize = (S1_MAX_BOUNDARY_SNAPSHOTS + 2) * S1_F32_TILE_ROWS_MAX;

impl PromptArenaLayout {
    /// The layout of a window of `rows` prompt rows.
    pub fn new(geometry: &ModelGeometry, rows: usize, tile_rows: usize) -> Result<Self, String> {
        if rows == 0 || tile_rows == 0 {
            return Err("slow-link prefill arena has no rows".to_string());
        }
        // Each tile's streams are packed at their OWN tag width: BF16 (2 bytes)
        // for a tile of at least `QWEN4_F16_WMMA_MIN_TOKENS` rows on the
        // production predicate (`Gpu::qwen4_bf16_streams`), F32 below it. The
        // F32 tiles are the cut-bounded heads and tails of at most
        // `S1_F32_STREAM_ROWS` rows per window; each costs 2 bytes per element
        // more than BF16. One more tile of BF16 elements is slack: stream ops
        // validate an F32-typed view of `rows * wide` elements, which reaches
        // past a BF16 region by at most its own size.
        let bf16_bytes = 2u64;
        let wide = geometry.wide()?;
        let tagged_rows = (rows as u64)
            .checked_add(S1_F32_STREAM_ROWS as u64)
            .and_then(|total| total.checked_add(tile_rows as u64))
            .ok_or_else(|| "slow-link prefill arena rows overflow".to_string())?;
        let hc_streams = checked_mul(
            checked_mul(tagged_rows, wide, "arena row bytes")?,
            bf16_bytes,
            "arena bytes",
        )?;
        let sizes: [u64; ARENA_FIELDS.len()] = [hc_streams];
        let mut offset = 0u64;
        let mut spans = [ArenaSpan {
            field: ArenaField::HcStreams,
            offset: 0,
            bytes: 0,
        }; ARENA_FIELDS.len()];
        for (index, (field, bytes)) in ARENA_FIELDS.into_iter().zip(sizes).enumerate() {
            spans[index] = ArenaSpan {
                field,
                offset,
                bytes,
            };
            offset = align(checked_add(offset, bytes, "arena bytes")?)?;
        }
        Ok(Self {
            rows,
            tile_rows,
            tiles: rows.div_ceil(tile_rows),
            spans,
            total_bytes: offset,
        })
    }

    /// The span of `field`.
    pub fn span(&self, field: ArenaField) -> ArenaSpan {
        self.spans[ARENA_FIELDS
            .iter()
            .position(|candidate| *candidate == field)
            .expect("every field is in ARENA_FIELDS")]
    }
}

/// Window rows that can land in K1 short-tail tiles (at most the historic
/// `FEW_ROW_ROUTE_ROWS` rows each): the final tile of each of the
/// [`S1_MAX_BOUNDARY_SNAPSHOTS`] + 1 session-cut segments plus one short entry
/// tile of a resumed prompt, so at most 48 rows. A plan whose short tiles
/// exceed this is a pre-mutation bypass.
pub const S1_K1_TAIL_ROWS: usize =
    (S1_MAX_BOUNDARY_SNAPSHOTS + 2) * rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS;

/// Window rows of the aggregate route's F32-tail tiles (the 2..511-row
/// prefill tiles of cut heads and tails, which stay inside the same window
/// and lease): at most one per cut segment head or tail, i.e. the same
/// `(4 interior cuts + 2) * 511 = 3066` rows the HC-stream arena already
/// bounds ([`S1_F32_STREAM_ROWS`], one definition). The slab route has none.
/// The layout charges only 40 B per row for the tail ids; a plan whose
/// F32-tail rows exceed this is a pre-mutation bypass. The dispatch layout
/// must accept this worst case rather than the census narrowing to its bound.
pub const S1_AGGREGATE_TAIL_ROWS: usize = S1_F32_STREAM_ROWS;

/// The dispatch crate's route-storage dimensions of a window of `rows` rows.
pub fn route_dims(
    geometry: &ModelGeometry,
    rows: usize,
) -> hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteDims {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteDims {
        rows,
        hidden: geometry.hidden,
        n_exp: geometry.num_experts,
        mi: geometry.moe_intermediate,
        shared_intermediate: geometry.shared_intermediate,
        // The conservative 48 on both routes (decode rows only on the
        // aggregate route; at most 6 are possible, 48 stays below the
        // dispatch packing bound beside the 3,066 tail rows).
        k1_rows: S1_K1_TAIL_ROWS,
        tail_rows: match geometry.route {
            WindowRoute::SlabIu4 => 0,
            WindowRoute::Aggregate | WindowRoute::AggregateSymIu4 => S1_AGGREGATE_TAIL_ROWS,
        },
        route: geometry.route,
    }
}

/// Whether a window of `rows` rows can hold the cohort capacities [`route_dims`] charges
/// (`k1_rows + tail_rows <= rows`, the same bound `PromptRouteLayout::new` enforces), so a window
/// too small for them is a named incumbent decline before the census instead of a layout error
/// at load. Only the aggregate routes charge tail rows: the slab route's 48 K1 rows always fit a
/// window of at least [`S1_MIN_FORCED_REQUEST_ROWS`] rows.
pub fn check_window_cohort(geometry: &ModelGeometry, rows: usize) -> Result<(), SlowlinkDecline> {
    let dims = route_dims(geometry, rows);
    if dims
        .k1_rows
        .checked_add(dims.tail_rows)
        .is_none_or(|cohort| cohort > rows)
    {
        return Err(SlowlinkDecline::WindowTooSmall {
            rows,
            k1_rows: dims.k1_rows,
            tail_rows: dims.tail_rows,
        });
    }
    Ok(())
}

/// Device bytes of the dispatch crate's `PromptRouteStorage` for a window of
/// `rows` rows (saved prepared input, top-k ids and weights, route maps and the
/// widened accumulator): exactly
/// `PromptRouteLayout::total_bytes`, the one layout `PromptRouteStorage::alloc`
/// allocates from.
pub fn route_storage_bytes(geometry: &ModelGeometry, rows: usize) -> Result<u64, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        u64::try_from(layout.total_bytes())
            .map_err(|_| "slow-link prefill route storage overflows".to_string())
    })
}

/// Device bytes of the two window-wide A4 activation sidecars
/// (`int4_mmq_x_scratch`, `qwen4_moe_down_i4_scratch`) the single count barrier
/// quantizes into, from the dispatch crate's own layout.
pub fn a4_sidecar_bytes(geometry: &ModelGeometry, rows: usize) -> Result<u64, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        u64::try_from(layout.a4_sidecar_bytes())
            .map_err(|_| "slow-link prefill A4 sidecars overflow".to_string())
    })
}

/// Device bytes of the window-size grouped-scatter Gpu scratch
/// (`qwen4_moe_group_layout(rows * top-k, n_exp)`) the single count barrier
/// grows, from the dispatch crate's own layout. Excluded from
/// [`route_storage_bytes`] and [`a4_sidecar_bytes`].
pub fn group_scatter_bytes(geometry: &ModelGeometry, rows: usize) -> Result<u64, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        u64::try_from(layout.group_scatter_bytes())
            .map_err(|_| "slow-link prefill group scatter overflows".to_string())
    })
}

/// The Gpu grow-only scratch buffers a full window reserves, one entry per
/// allocation (the two A4 sidecar slots on the slab route, the grouped-scatter
/// slot on every route), from the dispatch crate's own layout. They sum to
/// [`a4_sidecar_bytes`] + [`group_scatter_bytes`].
pub fn window_scratch_sizes(geometry: &ModelGeometry, rows: usize) -> Result<Vec<u64>, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        layout
            .scratch_allocation_sizes()
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| {
                u64::try_from(bytes)
                    .map_err(|_| "slow-link prefill window scratch overflows".to_string())
            })
            .collect()
    })
}

/// What an admitted S1 profile is for. The product charges nothing for
/// diagnostics; the Hold 1 primitive self-tests charge their own peak INSIDE
/// the admitted reserve ([`diagnostic_budget_bytes`]), so they are never
/// allocated beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionPurpose {
    Product,
    PrimitiveSelftest,
}

/// The tile a load of `purpose` admits ([`S1_PRODUCT_TILE_ROWS`] /
/// [`S1_PRIMITIVE_TILE_ROWS`]). A later request never falls to another
/// purpose's tile: the load's profile is immutable.
pub fn purpose_tile_rows(purpose: AdmissionPurpose) -> usize {
    match purpose {
        AdmissionPurpose::Product => S1_PRODUCT_TILE_ROWS,
        AdmissionPurpose::PrimitiveSelftest => S1_PRIMITIVE_TILE_ROWS,
    }
}

/// Device bytes the primitive self-tests allocate beyond the tile call's own
/// tensors, from the authoritative helpers (no copied case list or constants):
/// the transport's [`PanelPoolSpec::selftest_reserve`] total and the dispatch
/// crate's whole-suite ledger (`selftest::suite_diagnostic_bytes` over
/// `selftest::DEFAULT_SUITE`: the largest case-local total plus the grow-only
/// scratch slots and their growth transient, `SuiteBytes::device_total`, the
/// conservative figure that deducts nothing the census already reserved) over
/// the staged binding the primitives really run: one transport slot, i.e.
/// `PanelLimits::S1.experts_per_slot.min(n_exp)` experts, never every expert.
/// The transport's primitive and the dispatch suite are separate sequential
/// cases that each release before the next begins (`hip.malloc`/`free`
/// immediately after sync; `diag_scope` drains the parked pool), so the budget
/// is the larger of the two, never the sum. 0 for [`AdmissionPurpose::Product`].
///
/// [`PanelPoolSpec::selftest_reserve`]: crate::gpu_forward::panel_stage::PanelPoolSpec::selftest_reserve
pub fn diagnostic_budget_bytes(
    geometry: &ModelGeometry,
    purpose: AdmissionPurpose,
    gate_up_layer_bytes: u64,
    down_layer_bytes: u64,
) -> Result<u64, String> {
    use hipfire_dispatch::pipeline::slowlink_prefill::selftest::{
        suite_diagnostic_bytes, DiagGeometry, DEFAULT_SUITE,
    };
    use hipfire_dispatch::pipeline::slowlink_prefill::PanelLimits;
    if purpose == AdmissionPurpose::Product {
        return Ok(0);
    }
    let limits = PanelLimits::S1;
    let pages = PanelPageLayout::new(geometry.num_experts, gate_up_layer_bytes, down_layer_bytes)?;
    let transport = pages
        .pool_spec()?
        .selftest_reserve()
        .ok_or_else(|| "slow-link prefill selftest reserve overflows".to_string())?;
    let transport = u64::try_from(transport.total_device_bytes)
        .map_err(|_| "slow-link prefill selftest reserve overflows".to_string())?;
    let diag_geometry = DiagGeometry {
        hidden: geometry.hidden,
        n_exp: geometry.num_experts,
        mi: geometry.moe_intermediate,
        binding_experts: limits.experts_per_slot.min(geometry.num_experts),
        // Slab: the few-row K1 arm (3 rows). Aggregate: the F32-tail calls the
        // k1_tail_cohort diagnostic really runs, the 3-row indexed-down arm and
        // the 9-row grouped-down arm (`FEW_ROW_ROUTE_ROWS + 1`, the first
        // grouped-down row count): the maximum, 9.
        k1_rows: match geometry.route {
            WindowRoute::SlabIu4 => 3.min(rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS),
            WindowRoute::Aggregate | WindowRoute::AggregateSymIu4 => {
                rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS + 1
            }
        },
        shared_intermediate: geometry.shared_intermediate,
        route: geometry.route,
    };
    let suite = suite_diagnostic_bytes(&DEFAULT_SUITE, &diag_geometry, limits)
        .map_err(|error| error.to_string())?;
    let dispatch = u64::try_from(suite.device_total())
        .map_err(|_| "slow-link prefill diagnostic bytes overflow".to_string())?;
    Ok(transport.max(dispatch))
}

/// Everything the S1 profile costs beyond the incumbent's 1,536-row reserve,
/// term by term. [`Self::total`] is the one number placement charges
/// ([`crate::expert_residency::PlacementCharges::slowlink`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlowlinkCensus {
    /// The double-buffered active expert pages.
    pub ring: u64,
    /// The one-layer-ahead lookahead pages.
    pub lookahead: u64,
    /// The dispatch crate's `PromptRouteStorage`
    /// ([`route_storage_bytes`]: saved prepared input, top-k, route maps,
    /// widened accumulator).
    pub route_storage: u64,
    /// The window-wide A4 activation sidecars ([`a4_sidecar_bytes`]).
    pub a4_sidecars: u64,
    /// The window-size grouped-scatter Gpu scratch of the single count barrier
    /// ([`group_scatter_bytes`], `qwen4_moe_group_layout` at window slots): the
    /// FULL window figure, charged on top of the 1536-row tile workspace the
    /// base reserve already holds (the grown buffer frees the old one, so the
    /// replacement peak holds both).
    pub group_scatter: u64,
    /// The architecture-owned HC-stream arena ([`PromptArenaLayout`]).
    pub arena: u64,
    /// [`S1_MAX_BOUNDARY_SNAPSHOTS`] boundary stages of the target's session
    /// fixed parts, charged once each, plus the native-MTP entry `pending`
    /// hidden copy. The entry snapshot is the incumbent's own snapshot arena,
    /// already in the base reserve, and is never charged again.
    pub boundary_state: u64,
    /// BUDGET, not an allocation: [`S1_RUNTIME_OVERHEAD_BYTES`], the generic
    /// unattributed load/runtime occupancy the S1 load observed beyond every
    /// modelled term. Charged before the 512 MiB safety margin.
    pub runtime_overhead: u64,
    /// BUDGET: the Hold 1 primitive self-tests' own peak
    /// ([`diagnostic_budget_bytes`]); 0 for the product purpose.
    pub diagnostic: u64,
}

impl SlowlinkCensus {
    /// Every term, named, in log order: the seven OWNED allocations, then the
    /// two BUDGET terms ([`Self::owned_terms`], [`Self::budget_terms`]).
    pub fn terms(&self) -> [(&'static str, u64); 9] {
        let owned = self.owned_terms();
        let budget = self.budget_terms();
        [
            owned[0], owned[1], owned[2], owned[3], owned[4], owned[5], owned[6], budget[0],
            budget[1],
        ]
    }

    /// The seven terms S1 itself allocates, each from the layout that is also
    /// its allocation.
    pub fn owned_terms(&self) -> [(&'static str, u64); 7] {
        [
            ("expert ring", self.ring),
            ("expert lookahead", self.lookahead),
            ("route storage", self.route_storage),
            ("A4 sidecars", self.a4_sidecars),
            ("group scatter", self.group_scatter),
            ("HC arena", self.arena),
            ("boundary state", self.boundary_state),
        ]
    }

    /// The two budget (reservation) terms: not allocations S1 makes, but
    /// occupancy the placement must leave free.
    pub fn budget_terms(&self) -> [(&'static str, u64); 2] {
        [
            ("runtime overhead budget", self.runtime_overhead),
            ("diagnostic budget", self.diagnostic),
        ]
    }

    fn sum(terms: &[(&'static str, u64)]) -> Result<u64, String> {
        terms.iter().try_fold(0u64, |total, (name, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| format!("slow-link prefill census overflows at '{name}'"))
        })
    }

    /// Sum of the seven owned terms, checked.
    pub fn owned_total(&self) -> Result<u64, String> {
        Self::sum(&self.owned_terms())
    }

    /// Sum of every term, checked: the one number placement charges.
    pub fn total(&self) -> Result<u64, String> {
        Self::sum(&self.terms())
    }

    /// `name MiB, ...` for the placement log.
    pub fn describe(&self) -> String {
        self.terms()
            .into_iter()
            .map(|(name, bytes)| format!("{name} {} MiB", bytes >> 20))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The allocation-granule model of an admission ([`GranuleCharge`]): the
/// device's physical allocation granule and the incumbent base reserve's
/// per-owner slack at it. [`Default`] is payload-only accounting (granule 0),
/// which every non-S1 caller keeps byte for byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GranuleInputs {
    /// The device's allocation granule, from its own query
    /// ([`crate::expert_residency::granule_slack`] documents the backing
    /// model); 0 models no rounding.
    pub granule_bytes: u64,
    /// [`crate::expert_residency::base_granule_slack`] of the base reserve at
    /// the purpose's tile.
    pub base: crate::expert_residency::GranuleSlack,
    /// Granule slack of ONE boundary stage's part allocations
    /// (`Qwen4State::session_fixed_part_sizes`, one `zeros` per part); the
    /// profile charges [`S1_MAX_BOUNDARY_SNAPSHOTS`] stages.
    pub boundary_stage_slack: u64,
}

/// Physical-granule slack an admitted profile carries, owner by owner: the
/// sum over each owner's real device allocations of the distance to the next
/// granule. These are OWNED bytes the payload census omits; they are charged
/// beside the unattributed runtime budget, never inside it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GranuleCharge {
    pub granule_bytes: u64,
    /// The base reserve's owners ([`crate::expert_residency::GranuleSlack`]).
    pub base: crate::expert_residency::GranuleSlack,
    /// The page pool: [`PanelPageLayout::total_pages`] payload allocations of
    /// [`PanelPageLayout::page_bytes`] (the pointer tables are sub-granule).
    pub expert_pages: u64,
    /// The route region, one allocation of [`SlowlinkCensus::route_storage`].
    pub route_region: u64,
    /// The HC-stream arena, one allocation of [`PromptArenaLayout::total_bytes`].
    pub hc_arena: u64,
    /// The window's Gpu grow-only scratch buffers (A4 sidecar slots on the
    /// slab route, the grouped-scatter slot), one allocation each
    /// ([`window_scratch_sizes`]).
    pub window_scratch: u64,
    /// [`S1_MAX_BOUNDARY_SNAPSHOTS`] boundary stages' part allocations.
    pub boundary_stages: u64,
    /// Granule slack of ONE resident expert layer (its gate/up and down
    /// tensors, one allocation each). NOT a fixed term of [`Self::total`]: it
    /// is part of every resident layer's cost
    /// ([`SlowlinkProfile::placement_layer_bytes`]), so it is charged for the
    /// layers actually admitted.
    pub layer: u64,
}

impl GranuleCharge {
    /// Every fixed term, named, in log order (the per-layer slack is not one).
    pub fn terms(&self) -> [(&'static str, u64); 12] {
        let base = self.base.terms();
        [
            base[0],
            base[1],
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
            ("expert pages", self.expert_pages),
            ("route region", self.route_region),
            ("HC arena", self.hc_arena),
            ("window scratch", self.window_scratch),
            ("boundary stages", self.boundary_stages),
        ]
    }

    /// Sum of every fixed term, checked.
    pub fn total(&self) -> Result<u64, String> {
        self.terms().iter().try_fold(0u64, |total, (name, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| format!("slow-link prefill granule slack overflows at '{name}'"))
        })
    }

    /// `name KiB, ...` for the placement log, then the per-layer slack.
    pub fn describe(&self) -> String {
        self.terms()
            .into_iter()
            .map(|(name, bytes)| format!("{name} {} KiB", bytes >> 10))
            .chain(std::iter::once(format!(
                "per resident layer {} KiB",
                self.layer >> 10
            )))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The device memory an admission is sized from. `free_bytes` is the FRESH
/// `hipMemGetInfo` value sampled AFTER the host-link/copy-path and every other
/// load initialization the admission depends on, never an outer gate's
/// pre-load figure: the runtime residency that initialization created is
/// charged ONCE, by this lower actual free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionMemory {
    /// Optional free-memory sample taken before that initialization began.
    /// Observation only; placement never reads it.
    pub pre_init_free_bytes: Option<u64>,
    /// Actual free bytes at admission (post-initialization).
    pub free_bytes: u64,
}

impl AdmissionMemory {
    /// Actual free memory with no earlier sample.
    pub const fn actual(free_bytes: u64) -> Self {
        Self {
            pre_init_free_bytes: None,
            free_bytes,
        }
    }

    /// Actual free memory with the earlier pre-initialization sample.
    pub const fn measured(pre_init_free_bytes: u64, free_bytes: u64) -> Self {
        Self {
            pre_init_free_bytes: Some(pre_init_free_bytes),
            free_bytes,
        }
    }

    /// The MEASURED interval loss `before - after` between the
    /// pre-initialization sample and [`Self::free_bytes`]; `None` when no
    /// earlier sample exists. It is an observed interval, not a universal
    /// constant and not a per-allocation attribution, and it is never
    /// subtracted again: placement already reads the lower actual free. If
    /// free rose after the sample, the actual `free_bytes` stays authoritative
    /// and this is `Some(0)` (see [`Self::free_gained_after_initialization_bytes`]);
    /// no positive loss is invented.
    pub fn observed_initialization_bytes(&self) -> Option<u64> {
        self.pre_init_free_bytes
            .map(|before| before.saturating_sub(self.free_bytes))
    }

    /// How far actual free EXCEEDS the earlier sample (`after - before`), when
    /// it does; `Some(0)` otherwise, `None` without a sample.
    pub fn free_gained_after_initialization_bytes(&self) -> Option<u64> {
        self.pre_init_free_bytes
            .map(|before| self.free_bytes.saturating_sub(before))
    }
}

/// The one budget equation of a joint placement, scalars only. Every
/// consumer (admission, frontier, projection, fixture emit, receipt witness)
/// reads it; nothing recomputes it.
///
/// `needed(n) = non_expert_bytes + n * resident_layer_bytes + reserve_bytes`
/// must be at most `memory.free_bytes` for `n` resident layers to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementBudget {
    pub memory: AdmissionMemory,
    /// Resident non-expert weight payload (backing slack stays in the reserve).
    pub non_expert_bytes: u64,
    /// Backed device bytes of one resident layer: payload plus its own
    /// granule slack ([`GranuleCharge::layer`]).
    pub resident_layer_bytes: u64,
    /// The ENTIRE joint reserve, once: base reserve, safety margin, S1 census
    /// and granule slack.
    pub reserve_bytes: u64,
    /// Trunk layers of the model: the most layers that can be resident.
    pub trunk_layers: usize,
}

impl PlacementBudget {
    /// Device bytes needed to keep `layers` resident layers.
    pub fn needed_bytes_for(&self, layers: usize) -> u128 {
        u128::from(self.non_expert_bytes)
            + layers as u128 * u128::from(self.resident_layer_bytes)
            + u128::from(self.reserve_bytes)
    }

    /// `free - needed` for `layers` resident layers (negative: does not fit).
    pub fn slack_bytes_at(&self, layers: usize) -> i128 {
        i128::from(self.memory.free_bytes)
            .saturating_sub(i128::try_from(self.needed_bytes_for(layers)).unwrap_or(i128::MAX))
    }

    /// Whether `layers` resident layers fit the actual free memory.
    pub fn fits(&self, layers: usize) -> bool {
        self.needed_bytes_for(layers) <= u128::from(self.memory.free_bytes)
    }

    /// The largest resident layer count (at most [`Self::trunk_layers`]) that
    /// fits. `Some(0)` is a valid placement (every routed layer host
    /// streamed); `None` means even the base budget (zero layers) does not
    /// fit. There is no layer floor.
    pub fn largest_fitting_layers(&self) -> Option<usize> {
        self.fits(0).then(|| {
            crate::expert_residency::auto_vram_layers(
                self.memory.free_bytes,
                self.non_expert_bytes,
                self.resident_layer_bytes,
                self.trunk_layers,
                self.reserve_bytes,
            )
        })
    }

    /// Bytes the base budget (zero resident layers) is short by; 0 when it fits.
    pub fn base_deficit_bytes(&self) -> u64 {
        self.deficit_for(0)
    }

    /// Bytes the preferred [`S1_PREFERRED_RESIDENT_LAYERS`] are short by; 0
    /// when they fit. Diagnostic only, never a rejection.
    pub fn preferred_deficit_bytes(&self) -> u64 {
        self.deficit_for(S1_PREFERRED_RESIDENT_LAYERS)
    }

    fn deficit_for(&self, layers: usize) -> u64 {
        let short = self
            .needed_bytes_for(layers)
            .saturating_sub(u128::from(self.memory.free_bytes));
        u64::try_from(short).unwrap_or(u64::MAX)
    }

    /// The quote for the log, at `layers` resident layers.
    pub fn describe(&self, layers: usize) -> String {
        let initialization = match self.memory.pre_init_free_bytes {
            Some(before) => format!(
                ", {} B measured free interval since the {before} B pre-initialization sample",
                self.memory.observed_initialization_bytes().unwrap_or(0)
            ),
            None => String::new(),
        };
        format!(
            "{} B actual free after initialization{initialization}; {} B needed, {} B slack at {layers} resident layers of {} B backed each; preferred {S1_PREFERRED_RESIDENT_LAYERS} short by {} B",
            self.memory.free_bytes,
            self.needed_bytes_for(layers),
            self.slack_bytes_at(layers),
            self.resident_layer_bytes,
            self.preferred_deficit_bytes(),
        )
    }
}

/// What [`admit`] needs from the loader's own placement inputs.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionInputs {
    pub geometry: ModelGeometry,
    /// The actual memory the placement is sized from ([`AdmissionMemory`]).
    pub memory: AdmissionMemory,
    pub non_expert: u64,
    /// The largest trunk layer's routed-expert bytes (`resident_split`).
    pub layer_expert_bytes: u64,
    /// Gate/up and down payload bytes of that layer, separately.
    pub gate_up_layer_bytes: u64,
    pub down_layer_bytes: u64,
    pub trunk_layers: usize,
    /// [`crate::expert_residency::AutoReserve::census_total`] at the PURPOSE's
    /// tile ([`purpose_tile_rows`]), F16 weight shadow cold bound included: the
    /// S1 base reserve (census alone; the S1 terms and the safety margin are
    /// charged on top).
    pub base_reserve: u64,
    /// Layers the incumbent placement at the requested chunk would keep with
    /// the safety margin and no dedicated G2 stage charge (log only; see
    /// [`SlowlinkProfile::baseline_layers`]).
    pub baseline_layers: usize,
    /// Bytes of ONE boundary stage: the target's session fixed parts (GDN
    /// recurrent and conv, QSA selections, PLE conv, hyper feedback; no context
    /// rows, no MTP head), unpadded:
    /// [`crate::state::Qwen4State::session_fixed_stage_bytes`].
    pub fixed_state_bytes: u64,
    /// Entry `pending` hidden copy of a native-MTP request (0 for AR).
    pub pending_save_bytes: u64,
    /// The Hold 1 primitive self-tests' own peak device bytes
    /// ([`diagnostic_budget_bytes`]); 0 for the product purpose.
    pub diagnostic_bytes: u64,
    /// The HIP runtime overhead the base reserve ([`Self::base_reserve`])
    /// already includes (`AutoReserve.census.runtime`); 0 where it does not
    /// model one. The S1 runtime budget only supplements what is missing:
    /// `S1_RUNTIME_OVERHEAD_BYTES.saturating_sub(base_runtime_reserved_bytes)`,
    /// so the combined runtime reservation is the exact max of the two, never
    /// their sum.
    pub base_runtime_reserved_bytes: u64,
    /// The device's allocation granule and the base reserve's slack at it
    /// ([`GranuleInputs`]); default is payload-only accounting.
    pub granule: GranuleInputs,
    /// The incumbent's requested prefill chunk resolved at load: the non-S1
    /// checkpoint cadence (8192 on the narrow target). NEVER used for
    /// placement or scratch; those are the purpose's tile.
    pub checkpoint_stride_rows: usize,
    pub window_rows: usize,
    pub link_gbps: f64,
    /// What the profile is for; [`purpose_tile_rows`] of it is the tile
    /// [`admit`] admits.
    pub purpose: AdmissionPurpose,
    /// The source-header verdict the window route was resolved from
    /// ([`SymIu4RouteFacts::source_symmetric`]); `None` when no scan ran.
    pub sym_iu4_source_symmetric: Option<bool>,
}

/// The immutable admitted S1 profile: the frozen placement the load took and
/// the layouts the forward allocates from. Handed to the bundle before the
/// forward attaches; never mutated by a request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlowlinkProfile {
    /// The tile the load admitted ([`purpose_tile_rows`] of [`Self::purpose`]):
    /// the forward's scratch rows and the only tile a request ever runs.
    pub tile_rows: usize,
    /// What the load is for ([`AdmissionPurpose`]); immutable. A profile never
    /// serves another purpose.
    pub purpose: AdmissionPurpose,
    pub window_rows: usize,
    pub geometry: ModelGeometry,
    pub pages: PanelPageLayout,
    pub arena: PromptArenaLayout,
    pub census: SlowlinkCensus,
    /// Layers the charged placement keeps resident: the largest count that
    /// fits ([`PlacementBudget::largest_fitting_layers`]) of the quote it was
    /// admitted on (the actual free memory, or for an [`admit_grown`] profile
    /// that actual free less [`S1_GROWN_FREE_HOLD_BYTES`]), possibly zero,
    /// possibly above [`S1_PREFERRED_RESIDENT_LAYERS`]; always below the trunk
    /// layer count (a fully resident load takes the incumbent).
    pub resident_layers: usize,
    /// Layers the incumbent placement at the requested prefill chunk
    /// (8192 rows on a narrow gfx1100 link) would keep with the safety margin
    /// but WITHOUT the dedicated G2 stage charge: a log-only no-stage baseline
    /// that can exceed the real incumbent's layer count. Never a measured or
    /// gating figure; the preferred count ([`S1_PREFERRED_RESIDENT_LAYERS`]) does not read it.
    pub baseline_layers: usize,
    /// The non-S1 checkpoint cadence (see [`AdmissionInputs::checkpoint_stride_rows`]),
    /// immutable for the life of the load. Never used for placement or scratch.
    pub checkpoint_stride_rows: usize,
    pub link_gbps: f64,
    /// The physical-granule slack charged beside the census
    /// ([`GranuleCharge`]); part of [`Self::reserve_bytes`].
    pub granule: GranuleCharge,
    /// The quote this profile was admitted on. `placement.slack_bytes_at(resident_layers)`
    /// is [`Self::device_snapshot_budget_bytes`].
    pub placement: PlacementBudget,
    /// The placement charges this profile was admitted under, exactly the
    /// [`PlacementProjection::charges`] its joint reserve was quoted with: the
    /// 512 MiB safety margin, NO dedicated long-prefill stage (an S1 load
    /// charges its own page ring instead and creates no donor or slab stage),
    /// and [`Self::reserve_bytes`] as the S1 census. The profile owns the
    /// settled placement's charges; a loader that places exactly
    /// [`Self::resident_layers`] records these, never a recomputation.
    pub charges: crate::expert_residency::PlacementCharges,
    /// Device bytes the placement leaves free beyond every owned and budget
    /// charge AND the 512 MiB safety margin, at the admitted resident layers:
    /// exactly `placement.slack_bytes_at(resident_layers)` (actual free less
    /// non-expert weights, the resident layers and the whole joint reserve,
    /// which already includes the safety margin). Discretionary capacity only:
    /// this much, never the margin.
    pub device_snapshot_budget_bytes: u64,
    /// The source-header verdict behind [`ModelGeometry::route`] (see
    /// [`AdmissionInputs::sym_iu4_source_symmetric`]).
    pub sym_iu4_source_symmetric: Option<bool>,
}

impl SlowlinkProfile {
    /// Device bytes one resident layer costs the placement: the loader's
    /// `layer_expert_bytes` plus the layer's own physical-granule slack
    /// ([`GranuleCharge::layer`]). The loader's resident-layer resolver MUST
    /// size layers with this, against [`Self::reserve_bytes`], so it keeps
    /// exactly [`Self::resident_layers`].
    pub fn placement_layer_bytes(&self, layer_expert_bytes: u64) -> Result<u64, String> {
        layer_expert_bytes
            .checked_add(self.granule.layer)
            .ok_or_else(|| "slow-link prefill resident layer bytes overflow".to_string())
    }

    /// The one number placement charges for this profile besides its layers.
    pub fn reserve_bytes(&self) -> Result<u64, String> {
        self.census.total().and_then(|census| {
            self.granule.total().and_then(|granule| {
                census
                    .checked_add(granule)
                    .ok_or_else(|| "slow-link prefill reserve overflows".to_string())
            })
        })
    }

    /// Windows a request of `rows` rows runs: the planner packs whole tiles,
    /// so a window holds `window_rows / tile_rows` of them (the last possibly
    /// short). Session cuts may close more windows; this is the cut-free count.
    pub fn windows_for(&self, rows: usize) -> usize {
        let tiles_per_window = (self.window_rows / self.tile_rows).max(1);
        rows.div_ceil(self.tile_rows).div_ceil(tiles_per_window)
    }

    /// The dispatch crate's route-storage dimensions at this window
    /// ([`route_dims`]): what the forward passes to
    /// `PromptRouteLayout::new` and `PromptRouteStorage::alloc`, the same call
    /// the census made.
    pub fn route_dims(&self) -> hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteDims {
        route_dims(&self.geometry, self.window_rows)
    }

    /// The expert page pool spec the forward constructs its ring from
    /// ([`PanelPageLayout::pool_spec`]), the same spec the census read.
    pub(crate) fn pool_spec(
        &self,
    ) -> Result<crate::gpu_forward::panel_stage::PanelPoolSpec, String> {
        self.pages.pool_spec()
    }

    /// Pinned host bytes the profile adds beside the host-mapped experts: the
    /// page pool's table staging (`PanelPoolReserve::staging`). Timeline events
    /// are pooled HIP handles ([`PanelPageLayout::timeline_events`]), no bytes.
    pub fn host_pinned_bytes(&self) -> u64 {
        self.pages.host_staging_bytes
    }

    /// Device bytes left free beyond every charge and the safety margin at the
    /// admitted resident layers on the REAL free memory: the quote's own slack
    /// plus whatever the quote held back ([`S1_GROWN_FREE_HOLD_BYTES`] for a
    /// grown profile, 0 otherwise).
    pub fn remaining_slack_bytes(&self, actual_free_bytes: u64) -> u64 {
        self.device_snapshot_budget_bytes
            .saturating_add(actual_free_bytes.saturating_sub(self.placement.memory.free_bytes))
    }

    /// [`Self::describe`] extended with the chosen window (rows and tiles), the
    /// resident layers and the remaining slack on `actual_free_bytes`, the free
    /// hold included. The S1 load line.
    pub fn describe_against(&self, actual_free_bytes: u64) -> String {
        format!(
            "{}; window {} rows = {} tiles, {} resident layers, {} B slack remaining of {} B actual free ({} B free hold included)",
            self.describe(),
            self.window_rows,
            self.window_rows / self.tile_rows,
            self.resident_layers,
            self.remaining_slack_bytes(actual_free_bytes),
            actual_free_bytes,
            actual_free_bytes.saturating_sub(self.placement.memory.free_bytes),
        )
    }

    /// The placement log line.
    pub fn describe(&self) -> String {
        // Only the IU4 route names itself: every other load logs byte for byte as before.
        let route = match self.geometry.route {
            WindowRoute::AggregateSymIu4 => {
                format!(", window route {}", WindowRoute::AggregateSymIu4.label())
            }
            WindowRoute::SlabIu4 | WindowRoute::Aggregate => String::new(),
        };
        format!(
            "slow-link prefill (S1): {} MiB charged [{}], {}-row tile, {}-row window, \
             {} resident layers (no-stage incumbent baseline {}), {}+{} expert pages of {} bytes ({} KiB pinned host table staging), link {:.1} GB/s, \
             allocation granule {} KiB: {} MiB owner slack [{}] charged beside the census, {} MiB device snapshot budget; placement quote: {}{}",
            self.census.total().unwrap_or(0) >> 20,
            self.census.describe(),
            self.tile_rows,
            self.window_rows,
            self.resident_layers,
            self.baseline_layers,
            self.pages.active_pages,
            self.pages.pages_per_layer,
            self.pages.page_bytes,
            self.host_pinned_bytes() >> 10,
            self.link_gbps,
            self.granule.granule_bytes >> 10,
            self.granule.total().unwrap_or(0) >> 20,
            self.granule.describe(),
            self.device_snapshot_budget_bytes >> 20,
            self.placement.describe(self.resident_layers),
            route,
        )
    }
}

/// The layouts and the term-by-term census of an S1 profile for `inputs` at
/// the purpose's tile ([`purpose_tile_rows`]), exactly what [`admit`] charges
/// and the forward allocates from. Pure; the hold ledger reads it too, so no
/// figure is computed twice.
pub fn census_of(
    inputs: &AdmissionInputs,
) -> Result<(PanelPageLayout, PromptArenaLayout, SlowlinkCensus), String> {
    census_at_tile(inputs, purpose_tile_rows(inputs.purpose))
}

/// [`census_of`] at an explicit tile: only the HC-stream arena slack depends
/// on it (every other term is the window's).
fn census_at_tile(
    inputs: &AdmissionInputs,
    tile_rows: usize,
) -> Result<(PanelPageLayout, PromptArenaLayout, SlowlinkCensus), String> {
    let window_rows = inputs.window_rows;
    let pages = PanelPageLayout::new(
        inputs.geometry.num_experts,
        inputs.gate_up_layer_bytes,
        inputs.down_layer_bytes,
    )?;
    let arena = PromptArenaLayout::new(&inputs.geometry, window_rows, tile_rows)?;
    let census = SlowlinkCensus {
        ring: pages.ring_bytes()?,
        lookahead: pages.lookahead_bytes()?,
        route_storage: route_storage_bytes(&inputs.geometry, window_rows)?,
        a4_sidecars: a4_sidecar_bytes(&inputs.geometry, window_rows)?,
        group_scatter: group_scatter_bytes(&inputs.geometry, window_rows)?,
        arena: arena.total_bytes,
        boundary_state: checked_mul(
            S1_MAX_BOUNDARY_SNAPSHOTS as u64,
            inputs.fixed_state_bytes,
            "boundary state",
        )
        .and_then(|stages| checked_add(stages, inputs.pending_save_bytes, "boundary state"))?,
        runtime_overhead: S1_RUNTIME_OVERHEAD_BYTES
            .saturating_sub(inputs.base_runtime_reserved_bytes),
        diagnostic: inputs.diagnostic_bytes,
    };
    Ok((pages, arena, census))
}

/// The joint placement of one tile: the layouts and charges [`admit_at_tile`]
/// (the runtime, at the purpose's tile) and [`tile_frontier`] (the CPU proof)
/// share, with the one [`PlacementBudget`] they are all judged on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacementProjection {
    pub tile_rows: usize,
    pub pages: PanelPageLayout,
    pub arena: PromptArenaLayout,
    pub census: SlowlinkCensus,
    pub charges: crate::expert_residency::PlacementCharges,
    pub granule: GranuleCharge,
    /// `non_expert + reserve + layers * layer`, the whole reserve
    /// (`base_reserve + safety margin + S1 census + granule slack`) once.
    pub budget: PlacementBudget,
    /// The largest resident count that fits ([`PlacementBudget::largest_fitting_layers`]);
    /// `None`: the base budget does not fit. Not yet judged for a host layer.
    pub largest_layers: Option<usize>,
}

impl PlacementProjection {
    /// Resident layers the projection keeps (0 when the base budget does not fit;
    /// [`Self::verdict`] distinguishes that from a valid zero).
    pub fn resident_layers(&self) -> usize {
        self.largest_layers.unwrap_or(0)
    }

    /// The runtime verdict: the resident layer count to admit, or why not.
    /// A base budget that does not fit is the exact [`Deficit`]; a placement
    /// holding every routed layer has nothing to stream
    /// ([`SlowlinkDecline::NoHostLayers`]); any other count, zero and
    /// counts below [`S1_PREFERRED_RESIDENT_LAYERS`] included, is admitted.
    pub fn verdict(&self) -> Result<usize, SlowlinkDecline> {
        let budget = &self.budget;
        let Some(layers) = self.largest_layers else {
            return Err(SlowlinkDecline::Deficit(Deficit {
                memory: budget.memory,
                non_expert: budget.non_expert_bytes,
                resident_layer_bytes: budget.resident_layer_bytes,
                reserve: budget.reserve_bytes,
                required_layers: 0,
                deficit_bytes: budget.base_deficit_bytes(),
            }));
        };
        if layers >= budget.trunk_layers {
            return Err(SlowlinkDecline::NoHostLayers {
                resident_layers: layers,
            });
        }
        Ok(layers)
    }
}

/// The pure joint placement of `inputs` at `tile_rows` (a legal rung,
/// [`legal_tile_rungs`]): layouts, census, granule slack, the whole reserve
/// and the [`PlacementBudget`]. The ONE function [`admit_at_tile`],
/// [`tile_frontier`] and the fixture emit read; nothing else computes a
/// layer count or a needed/slack byte figure. `inputs.base_reserve` and
/// `base_runtime_reserved_bytes` must describe the base reserve at
/// `tile_rows`. Allocates and queries nothing.
pub fn project_at_tile(
    inputs: &AdmissionInputs,
    tile_rows: usize,
) -> Result<PlacementProjection, SlowlinkDecline> {
    if !legal_tile_rungs().contains(&tile_rows) {
        return Err(SlowlinkDecline::Geometry(format!(
            "slow-link prefill tile {tile_rows} is not a legal rung"
        )));
    }
    check_window_cohort(&inputs.geometry, inputs.window_rows)?;
    let (pages, arena, census) =
        census_at_tile(inputs, tile_rows).map_err(SlowlinkDecline::Geometry)?;
    let granule =
        granule_charge(inputs, &pages, &arena, &census).map_err(SlowlinkDecline::Geometry)?;
    let charges = crate::expert_residency::PlacementCharges::new(inputs.layer_expert_bytes, false)
        .and_then(|charges| {
            census
                .total()
                .and_then(|bytes| {
                    granule.total().and_then(|slack| {
                        bytes
                            .checked_add(slack)
                            .ok_or_else(|| "slow-link prefill reserve overflows".to_string())
                    })
                })
                .map(|bytes| charges.with_slowlink(bytes))
        })
        .map_err(SlowlinkDecline::Geometry)?;
    let reserve_bytes = charges
        .reserve(inputs.base_reserve)
        .map_err(SlowlinkDecline::Geometry)?;
    let resident_layer_bytes = inputs
        .layer_expert_bytes
        .checked_add(granule.layer)
        .ok_or_else(|| {
            SlowlinkDecline::Geometry("slow-link prefill resident layer bytes overflow".into())
        })?;
    let budget = PlacementBudget {
        memory: inputs.memory,
        non_expert_bytes: inputs.non_expert,
        resident_layer_bytes,
        reserve_bytes,
        trunk_layers: inputs.trunk_layers,
    };
    Ok(PlacementProjection {
        tile_rows,
        pages,
        arena,
        census,
        charges,
        granule,
        largest_layers: budget.largest_fitting_layers(),
        budget,
    })
}

/// The physical-granule slack of the profile's own owners at `inputs`'
/// granule: the page pool, route region, HC arena and boundary stages (fixed
/// terms) and one resident layer's tensors (charged per admitted layer), each
/// from the same layout that allocates it, beside the base reserve's.
fn granule_charge(
    inputs: &AdmissionInputs,
    pages: &PanelPageLayout,
    arena: &PromptArenaLayout,
    census: &SlowlinkCensus,
) -> Result<GranuleCharge, String> {
    let granule = inputs.granule.granule_bytes;
    let slack = |bytes: u64| crate::expert_residency::granule_slack(bytes, granule);
    let layer = checked_add(
        slack(inputs.gate_up_layer_bytes),
        slack(inputs.down_layer_bytes),
        "resident layer granule slack",
    )?;
    Ok(GranuleCharge {
        granule_bytes: granule,
        base: inputs.granule.base,
        expert_pages: checked_mul(
            pages.total_pages() as u64,
            slack(pages.page_bytes),
            "page granule slack",
        )?,
        route_region: slack(census.route_storage),
        hc_arena: slack(arena.total_bytes),
        window_scratch: crate::expert_residency::granule_slack_total(
            window_scratch_sizes(&inputs.geometry, inputs.window_rows)?,
            granule,
        )?,
        boundary_stages: checked_mul(
            S1_MAX_BOUNDARY_SNAPSHOTS as u64,
            inputs.granule.boundary_stage_slack,
            "boundary stage granule slack",
        )?,
        layer,
    })
}

/// The joint S1 placement at the purpose's tile
/// ([`purpose_tile_rows`]`(inputs.purpose)`): charge the tile, HC arena, route
/// storage, A4 sidecars, page store and boundary state together, then keep
/// the largest resident layer count that fits the actual free memory
/// ([`project_at_tile`]). [`S1_PREFERRED_RESIDENT_LAYERS`] is information
/// only. Refuses, before anything is allocated, with the exact deficit only
/// when the base budget (zero resident layers) does not fit; a product load
/// never falls to the primitive tile (or any other) on a refusal.
/// `inputs.base_reserve` must be the census at that tile.
pub fn admit(inputs: &AdmissionInputs) -> Result<SlowlinkProfile, SlowlinkDecline> {
    admit_at_tile(inputs, purpose_tile_rows(inputs.purpose))
}

/// [`admit`] with the window sized at load from the actual free device bytes,
/// never leaving less than [`S1_GROWN_FREE_HOLD_BYTES`] of device memory free.
///
/// The base profile is the ordinary admission at `inputs.window_rows` (the
/// [`S1_WINDOW_ROWS`] floor) on the real free memory: its refusals are
/// returned unchanged. Growth is judged on a quote that holds
/// [`S1_GROWN_FREE_HOLD_BYTES`] back from the actual free bytes, so a grown
/// profile leaves that much free beyond every charge, the 512 MiB safety
/// margin and its own [`SlowlinkProfile::device_snapshot_budget_bytes`]. The
/// held-back quote is the profile's placement quote, so it is internally
/// consistent and the loader places exactly its resident count.
///
/// A context whose covering window (`max_seq` rounded UP to a tile) is at most
/// [`S1_GROWN_WINDOW_MAX_TILES`] runs every request in one window, so the
/// window takes priority over extra resident layers: the covering window is
/// tried first, then a binary search down to the largest whole-tile window
/// above the floor that admits on the held-back quote, each candidate with its
/// own resident-layer verdict. A longer context runs several windows anyway:
/// the floor admission's resident count is kept and the window grows (to at
/// most the cap) only with the memory left at that count. The KV charge is
/// that of the load's `max_seq` (`inputs.base_reserve`). Nothing is decided at
/// request time. If no window above the floor admits, the base profile is
/// returned unchanged (growth never refuses a load).
///
/// Every term is the same pure census the base used, so charge ==
/// allocation; [`S1_F32_STREAM_ROWS`], the boundary snapshot cap and the
/// checkpoint stride are window independent.
pub fn admit_grown(
    inputs: &AdmissionInputs,
    max_seq: usize,
) -> Result<SlowlinkProfile, SlowlinkDecline> {
    let base = admit(inputs)?;
    let tile = base.tile_rows;
    let ceiling_rows = max_seq.div_ceil(tile).saturating_mul(tile);
    let (floor_tiles, ceiling_tiles) = (
        base.window_rows / tile,
        (ceiling_rows / tile).min(S1_GROWN_WINDOW_MAX_TILES),
    );
    // A window that is not a whole number of tiles (a context shorter than the
    // floor) already holds every row the context can.
    if base.window_rows % tile != 0 || ceiling_tiles <= floor_tiles {
        return Ok(base);
    }
    let held = |tiles: usize| AdmissionInputs {
        window_rows: tiles * tile,
        memory: AdmissionMemory {
            pre_init_free_bytes: inputs
                .memory
                .pre_init_free_bytes
                .map(|free| free.saturating_sub(S1_GROWN_FREE_HOLD_BYTES)),
            free_bytes: inputs
                .memory
                .free_bytes
                .saturating_sub(S1_GROWN_FREE_HOLD_BYTES),
        },
        ..*inputs
    };
    // A context whose covering window is within the cap runs every request in
    // one window: the window outranks resident layers, each candidate with its
    // own layer verdict. A longer context runs several windows anyway: the
    // floor admission's resident count is kept and the window grows only with
    // the memory left at that count.
    let covering = ceiling_rows / tile;
    let one_window = covering <= S1_GROWN_WINDOW_MAX_TILES;
    let fits = |tiles: usize| {
        admit(&held(tiles))
            .ok()
            .filter(|profile| one_window || profile.resident_layers == base.resident_layers)
    };
    if one_window {
        // The max_seq-covering window first.
        if let Some(profile) = fits(ceiling_tiles) {
            return Ok(profile);
        }
    }
    // Slack only falls as the window grows, so "fits" is monotone in tiles.
    let top = if one_window {
        ceiling_tiles - 1
    } else {
        ceiling_tiles
    };
    let (mut lo, mut hi, mut best) = (floor_tiles, top, None);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        match fits(mid) {
            Some(profile) => {
                best = Some(profile);
                lo = mid;
            }
            None => hi = mid - 1,
        }
    }
    Ok(best.unwrap_or(base))
}

/// [`admit`] at an explicit legal tile ([`legal_tile_rungs`]) through the same
/// [`project_at_tile`] quote and verdict: the shared helper tests use to
/// build a real computed profile at any rung (the historical 1,536-row
/// counterfactual included). Production admission goes through [`admit`] only.
/// `inputs.base_reserve` and `base_runtime_reserved_bytes` must describe the
/// base reserve at `tile_rows`.
pub(crate) fn admit_at_tile(
    inputs: &AdmissionInputs,
    tile_rows: usize,
) -> Result<SlowlinkProfile, SlowlinkDecline> {
    let projection = project_at_tile(inputs, tile_rows)?;
    let resident_layers = projection.verdict()?;
    // Everything the placement keeps beside the joint reserve (safety margin
    // included): what is left of the actual free is discretionary capacity.
    let device_snapshot_budget_bytes =
        u64::try_from(projection.budget.slack_bytes_at(resident_layers)).map_err(|_| {
            SlowlinkDecline::Geometry(
                "slow-link prefill placement exceeds the free device memory it admitted".into(),
            )
        })?;
    Ok(SlowlinkProfile {
        tile_rows,
        window_rows: inputs.window_rows,
        geometry: inputs.geometry,
        pages: projection.pages,
        arena: projection.arena,
        census: projection.census,
        granule: projection.granule,
        placement: projection.budget,
        charges: projection.charges,
        device_snapshot_budget_bytes,
        resident_layers,
        baseline_layers: inputs.baseline_layers,
        checkpoint_stride_rows: inputs.checkpoint_stride_rows,
        link_gbps: inputs.link_gbps,
        sym_iu4_source_symmetric: inputs.sym_iu4_source_symmetric,
        purpose: inputs.purpose,
    })
}

/// Smallest legal S1 tile: [`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`].
/// Below it every full tile of a window would be an F32-stream, aggregate
/// F32-tail tile, so the window's F32 rows ([`S1_F32_STREAM_ROWS`] = 3,066,
/// also the dispatch layout's tail-id bound [`S1_AGGREGATE_TAIL_ROWS`]) would
/// be exceeded by the 16,384-row window (64 tiles of 256 rows); at or above
/// it every full tile is a WMMA (K2) cohort tile and only the cut-bounded
/// window head/tail tiles of fewer than 512 rows are F32.
pub const S1_MIN_TILE_ROWS: usize =
    hipfire_dispatch::pipeline::slowlink_prefill::AGGREGATE_MIN_TILE_ROWS;

/// Every legal S1 tile, largest first: the existing 256-row chunk granularity
/// ([`crate::gpu_forward::explicit_prefill_chunk_rows`]) from [`S1_TILE_ROWS`]
/// (the largest allowed) down to [`S1_MIN_TILE_ROWS`]; both purposes' tiles
/// ([`S1_PRODUCT_TILE_ROWS`], [`S1_PRIMITIVE_TILE_ROWS`]) are rungs.
/// No new tile size is introduced: each rung is a value the explicit chunk
/// policy already accepts unchanged.
pub fn legal_tile_rungs() -> Vec<usize> {
    (S1_MIN_TILE_ROWS..=S1_TILE_ROWS)
        .rev()
        .filter(|&rows| {
            let value = rows.to_string();
            crate::gpu_forward::explicit_prefill_chunk_rows(Some(value.as_str())) == Some(rows)
        })
        .collect()
}

/// One (tile, resident layers) point of the joint admission, every memory
/// figure the proof reads. Not a runtime profile: [`admit`] only returns the
/// purpose's tile ([`purpose_tile_rows`], [`Self::selected`]).
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierPoint {
    pub tile_rows: usize,
    pub tiles_per_window: usize,
    /// Rows of the window's final tile; below [`S1_MIN_TILE_ROWS`] it is the
    /// one F32 tail tile of the window (inside [`S1_F32_STREAM_ROWS`]).
    pub final_tile_rows: usize,
    /// `AutoReserve::placement_bytes(true)` at this tile: the census alone,
    /// INCLUDING [`Self::f16_shadow_bytes`] (never added again).
    pub base_reserve: u64,
    /// The F16 weight shadow cold bound inside [`Self::base_reserve`].
    pub f16_shadow_bytes: u64,
    /// The physical-granule slack charged inside [`Self::reserve`], owner by
    /// owner (all zero without a granule model).
    pub granule: GranuleCharge,
    /// The S1 terms at this tile, the 1 GiB runtime budget
    /// ([`SlowlinkCensus::runtime_overhead`]) and the diagnostic budget kept
    /// separately visible.
    pub census: SlowlinkCensus,
    /// The 512 MiB safety margin charged on top of everything else.
    pub safety_margin: u64,
    /// `base_reserve + census.total() + granule.total() + safety_margin`.
    pub reserve: u64,
    /// The quote this point was judged on ([`project_at_tile`]): actual
    /// memory, non-expert weights, backed layer cost and the whole reserve.
    pub budget: PlacementBudget,
    /// Payload bytes of the largest layer (the backed cost is
    /// `budget.resident_layer_bytes`).
    pub layer_expert_bytes: u64,
    /// Layers the charged placement keeps resident (0 when the base budget
    /// does not fit too; [`Self::verdict`] distinguishes the two).
    pub resident_layers: usize,
    /// The runtime verdict: `Ok` only when [`admit_at_tile`] would admit this
    /// tile (a placement with a host layer left; no resident-layer floor).
    /// `Ok` on a point that is not [`Self::selected`] is "would fit",
    /// never an admission: [`admit`] only ever takes the purpose's tile.
    pub verdict: Result<(), SlowlinkDecline>,
    /// This is the tile [`admit`] takes for the inputs' purpose
    /// ([`purpose_tile_rows`]); every other point is a would-fit report.
    pub selected: bool,
    /// Another feasible point has at least this tile AND at least these
    /// resident layers (strictly more of one).
    pub dominated: bool,
}

impl FrontierPoint {
    /// Whether [`admit`]'s rules admit this tile.
    pub fn admits(&self) -> bool {
        self.verdict.is_ok()
    }

    /// Device bytes the joint placement needs for `layers` resident layers
    /// ([`PlacementBudget::needed_bytes_for`]).
    pub fn needed_bytes_for(&self, layers: usize) -> u128 {
        self.budget.needed_bytes_for(layers)
    }

    /// `free - needed` for `layers` resident layers (negative: does not fit).
    pub fn slack_for(&self, layers: usize) -> i128 {
        self.budget.slack_bytes_at(layers)
    }
}

/// The feasible (tile, resident layers) frontier of the joint admission on
/// `inputs`: for every legal tile ([`legal_tile_rungs`]) the charged joint
/// placement and its runtime verdict. `reserve_at(tile)` is the loader's own
/// base reserve at that tile (`placement_inputs`'
/// [`crate::expert_residency::AutoReserve`], census
/// with the F16 shadow cold bound); its `placement_bytes(true)` and runtime
/// credit replace `inputs.base_reserve`/`base_runtime_reserved_bytes` per
/// tile. The window, lookahead and route F32 capacity are exactly the
/// runtime's, as is [`project_at_tile`] (the largest count that fits, no
/// resident-layer floor); nothing is lowered. Pure CPU.
pub fn tile_frontier(
    inputs: &AdmissionInputs,
    mut reserve_at: impl FnMut(
        usize,
    ) -> Result<
        (
            crate::expert_residency::AutoReserve,
            crate::expert_residency::GranuleSlack,
        ),
        String,
    >,
) -> Result<Vec<FrontierPoint>, String> {
    let mut points = Vec::new();
    for tile_rows in legal_tile_rungs() {
        let (reserve, base_slack) = reserve_at(tile_rows)?;
        let at = AdmissionInputs {
            base_reserve: reserve.placement_bytes(true),
            base_runtime_reserved_bytes: reserve.census.runtime,
            granule: GranuleInputs {
                base: base_slack,
                ..inputs.granule
            },
            ..*inputs
        };
        let joint = project_at_tile(&at, tile_rows).map_err(|decline| decline.to_string())?;
        let tiles_per_window = joint.arena.tiles;
        let final_tile_rows = at.window_rows - (tiles_per_window - 1) * tile_rows;
        points.push(FrontierPoint {
            tile_rows,
            tiles_per_window,
            final_tile_rows,
            base_reserve: at.base_reserve,
            f16_shadow_bytes: reserve.census.f16_shadows,
            census: joint.census,
            granule: joint.granule,
            safety_margin: joint.charges.margin,
            reserve: joint.budget.reserve_bytes,
            budget: joint.budget,
            layer_expert_bytes: at.layer_expert_bytes,
            resident_layers: joint.resident_layers(),
            verdict: joint.verdict().map(|_| ()),
            selected: tile_rows == purpose_tile_rows(inputs.purpose),
            dominated: false,
        });
    }
    let keys: Vec<(usize, usize, bool)> = points
        .iter()
        .map(|point| (point.tile_rows, point.resident_layers, point.admits()))
        .collect();
    for (index, point) in points.iter_mut().enumerate() {
        let (tile, layers, admits) = keys[index];
        point.dominated = keys.iter().enumerate().any(|(other, &(t, n, ok))| {
            other != index
                && t >= tile
                && n >= layers
                && (t > tile || n > layers)
                && (ok || !admits)
        });
    }
    Ok(points)
}

/// What a load knows once it has the manifest, the free device memory and the
/// two reserves, all of which the incumbent placement already computes. The
/// product loader and every example/gate build this the same way, so one
/// admission decides for all of them.
pub struct LoadFacts<'a> {
    pub config: &'a Qwen4Config,
    pub weights: &'a [hipfire_runtime::weight_manifest::WeightEntry],
    pub bytes_of: &'a dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    /// What the admitted profile is for ([`AdmissionPurpose`]).
    pub purpose: AdmissionPurpose,
    /// The actual device memory the placement is sized from: `free_bytes` is
    /// sampled AFTER every initialization the admission depends on
    /// ([`AdmissionMemory`]); `pre_init_free_bytes` is the optional earlier sample.
    pub memory: AdmissionMemory,
    /// [`crate::expert_residency::AutoReserve::census_total`] at the PURPOSE's
    /// tile ([`purpose_tile_rows`], `placement_bytes(true)`; F16 weight shadow
    /// cold bound included): the S1 profile's base
    /// reserve is the bottom-up census alone, never the historical floor.
    pub tile_reserve: u64,
    /// [`crate::expert_residency::AutoReserve::bytes`] (the incumbent's
    /// historical-floor-or-census figure) at the incumbent placement chunk
    /// (sizes the logged baseline layer count only).
    pub incumbent_reserve: u64,
    /// The HIP runtime overhead the base reserve already charges
    /// (`AutoReserve.census.runtime`, [`crate::expert_residency::runtime_overhead_bytes`];
    /// 0 where the census does not model it). A fact of the reserve, never an
    /// arch or card test: the S1 runtime budget is credited by it
    /// ([`SlowlinkCensus::runtime_overhead`]).
    pub base_runtime_reserved_bytes: u64,
    /// The device's allocation granule and the base reserve's per-owner slack
    /// at it ([`GranuleInputs`]).
    pub granule: GranuleInputs,
    pub state_format: crate::Qwen4StateFormat,
    pub native_mtp: bool,
    pub window_rows: usize,
    /// load preserves. Never a placement or scratch input.
    pub checkpoint_stride_rows: usize,
    /// The window route resolved BEFORE the census ([`resolve_window_route`] over
    /// [`window_route_of`]`(weights)`): what [`load_admission_inputs`] sizes every route-dependent
    /// byte from. Must be the layout route or [`WindowRoute::AggregateSymIu4`] of an aggregate
    /// layout; a hidden read of the typed flag later would defeat the charge.
    pub window_route: WindowRoute,
    /// The source-header verdict behind [`Self::window_route`]; `None` when no scan ran.
    pub sym_iu4_source_symmetric: Option<bool>,
}

/// [`AdmissionInputs`] from [`LoadFacts`] and the measured narrow `link_gbps`:
/// the non-expert and per-layer expert bytes from the manifest, the packed
/// gate/up and down payload split, the exact session fixed snapshot bytes and
/// the baseline layer count.
pub fn load_admission_inputs(
    facts: &LoadFacts<'_>,
    link_gbps: f64,
) -> Result<AdmissionInputs, SlowlinkDecline> {
    use crate::expert_residency as residency;
    let layout = window_route_of(facts.weights);
    let route = facts.window_route;
    if route != layout
        && !(layout == WindowRoute::Aggregate && route == WindowRoute::AggregateSymIu4)
    {
        return Err(SlowlinkDecline::Geometry(format!(
            "the resolved window route {route:?} is not the artifact layout {layout:?}'s"
        )));
    }
    if mixed_source_priced_mq2r2_e8p(facts.weights, facts.bytes_of) {
        return Err(SlowlinkDecline::Geometry(
            "source-priced mixed MQ2-R2/E8P routed experts are not an S1 layout: declined pre-mutation even with incumbent kernels".into(),
        ));
    }
    let geometry =
        ModelGeometry::from_config(facts.config, route).map_err(SlowlinkDecline::Geometry)?;
    let (non_expert, layer_expert_bytes) = residency::resident_split(facts.weights, facts.bytes_of)
        .map_err(SlowlinkDecline::Geometry)?;
    let baseline_layers = residency::PlacementCharges::new(layer_expert_bytes, false)
        .and_then(|charges| {
            residency::auto_placement_layers(
                facts.memory.free_bytes,
                non_expert,
                layer_expert_bytes,
                facts.config.num_hidden_layers,
                facts.incumbent_reserve,
                &charges,
            )
        })
        .unwrap_or(0);
    let (gate_up_layer_bytes, down_layer_bytes) =
        residency::routed_layer_payload_split(facts.weights, facts.bytes_of)
            .map_err(SlowlinkDecline::Geometry)?;
    // One boundary stage = the TARGET's session fixed parts only (the head's
    // rows are not staged), exactly what `Qwen4Bundle::prompt_txn_stage_bytes`
    // allocates ([`crate::state::Qwen4State::session_fixed_stage_bytes`], the
    // unpadded sum of the same fixed-part allocator). Each new boundary stage
    // is charged once.
    let fixed_state_bytes =
        crate::state::Qwen4State::session_fixed_stage_bytes(facts.config, facts.state_format)
            .ok_or_else(|| {
                SlowlinkDecline::Geometry("session fixed stage bytes overflow".into())
            })?;
    // A native-MTP request also saves the entry `pending` hidden (one F32 wide
    // row, `hc_count * hidden * 4`) for rollback: a NEW buffer beyond the
    // existing pending owner (`Qwen4Bundle::slowlink_save_pending`).
    let pending_save_bytes = if facts.native_mtp {
        checked_mul(
            checked_mul(
                facts.config.hc_count as u64,
                facts.config.hidden_size as u64,
                "pending hidden",
            )
            .map_err(SlowlinkDecline::Geometry)?,
            4,
            "pending hidden",
        )
        .map_err(SlowlinkDecline::Geometry)?
    } else {
        0
    };
    // The Hold 1 primitive self-tests' own peak (0 for the product), from the
    // authoritative transport and dispatch helpers.
    let diagnostic_bytes = diagnostic_budget_bytes(
        &geometry,
        facts.purpose,
        gate_up_layer_bytes,
        down_layer_bytes,
    )
    .map_err(SlowlinkDecline::Geometry)?;
    Ok(AdmissionInputs {
        geometry,
        memory: facts.memory,
        non_expert,
        layer_expert_bytes,
        gate_up_layer_bytes,
        down_layer_bytes,
        trunk_layers: facts.config.num_hidden_layers,
        base_reserve: facts.tile_reserve,
        baseline_layers,
        fixed_state_bytes,
        pending_save_bytes,
        diagnostic_bytes,
        base_runtime_reserved_bytes: facts.base_runtime_reserved_bytes,
        granule: facts.granule,
        checkpoint_stride_rows: facts.checkpoint_stride_rows,
        window_rows: facts.window_rows,
        link_gbps,
        purpose: facts.purpose,
        sym_iu4_source_symmetric: facts.sym_iu4_source_symmetric,
    })
}

/// Select the S1 profile for a load, or `Ok(None)` for the incumbent. Probe-free
/// checks run first; `link` (the host-link probe) runs only for a load that
/// passed them, and `inputs` (the memory query and census inputs) only for a
/// link that is narrow, so every other load does no S1 work and keeps its
/// placement byte for byte. A forced selector fails the load only on a capacity
/// refusal of an otherwise eligible load (exact byte deficit); every other
/// decline is the incumbent. Run it BEFORE the placement is sized: the profile
/// carries the charge ([`PlacementCharges::with_slowlink`]) and the resident
/// layers the placement must keep.
///
/// [`PlacementCharges::with_slowlink`]: crate::expert_residency::PlacementCharges::with_slowlink
pub fn select_profile(
    mode: SlowlinkMode,
    shape: &LoadShape<'_>,
    link: impl FnOnce() -> HostLink,
    inputs: impl FnOnce(f64) -> Result<AdmissionInputs, SlowlinkDecline>,
) -> Result<Option<SlowlinkProfile>, String> {
    let decline = match preflight(mode, shape) {
        Err(decline) => decline,
        Ok(()) => match link_verdict(link())
            .and_then(inputs)
            .and_then(|inputs| {
                admit_grown(&inputs, shape.max_seq)
                    .map(|profile| (profile, inputs.memory.free_bytes))
            })
            .and_then(
                |(profile, free)| match explicit_chunk_rows(shape.chunk_rows_env) {
                    Some(rows) if rows != profile.tile_rows => {
                        Err(SlowlinkDecline::ExplicitChunk {
                            rows,
                            tile: Some(profile.tile_rows),
                        })
                    }
                    _ => Ok((profile, free)),
                },
            ) {
            Ok((profile, free)) => {
                // A non-front switch value names WHICH layers are resident;
                // resolve it now so an invalid value fails the load before
                // placement. Unset or `front` is the incumbent line.
                let set = if resident_layer_switch_is_override(
                    hipfire_config::developer_var(S1_RESIDENT_LAYERS_ENV)
                        .ok()
                        .as_deref(),
                ) {
                    Some(s1_resident_layer_set(
                        profile.placement.trunk_layers,
                        profile.resident_layers,
                    )?)
                } else {
                    None
                };
                match set {
                    Some(set) => {
                        eprintln!(
                            "  qwen4 {} resident_layer_set={set:?}",
                            profile.describe_against(free)
                        )
                    }
                    None => eprintln!("  qwen4 {}", profile.describe_against(free)),
                }
                return Ok(Some(profile));
            }
            Err(decline) => decline,
        },
    };
    if mode == SlowlinkMode::Force && decline.is_capacity() {
        return Err(format!(
            "qwen4: {SLOWLINK_PREFILL_ENV}=1 cannot be admitted: {decline}"
        ));
    }
    eprintln!(
        "  qwen4 slow-link prefill (S1): not selected ({decline}); incumbent placement unchanged"
    );
    Ok(None)
}

/// [`select_profile`] with [`load_admission_inputs`]: the one call the product
/// loader and the examples/gates make. `facts` runs only for a narrow link (it
/// queries free memory and builds the reserves), `link` only after the
/// probe-free checks passed. Returns the profile (with its resident layer
/// count) before any placement is sized.
pub fn select_load_profile<'a>(
    mode: SlowlinkMode,
    shape: &LoadShape<'_>,
    link: impl FnOnce() -> HostLink,
    facts: impl FnOnce() -> Result<LoadFacts<'a>, String>,
) -> Result<Option<SlowlinkProfile>, String> {
    select_profile(mode, shape, link, |link_gbps| {
        let facts = facts().map_err(SlowlinkDecline::Geometry)?;
        load_admission_inputs(&facts, link_gbps)
    })
}

/// The structural facts of a load that only its caller knows; the rest of a
/// [`LoadShape`] is read from the process snapshot
/// ([`LoadShape::from_process`]). Pass what is true: a load that applies a hot
/// set or `whole-first` must say so, never `false` to get a profile.
#[derive(Clone, Copy, Debug)]
pub struct StructuralShape {
    pub expert_parallel: bool,
    pub pipeline_band: bool,
    pub routed_experts: bool,
    pub expert_keep: bool,
    pub hotset_sidecar: bool,
    pub whole_first: bool,
}

impl<'a> LoadShape<'a> {
    /// The shape of a load whose process-level controls are read from the
    /// config snapshot exactly as the product loader reads them: an explicit
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` and the `HIPFIRE_QWEN4_EXPERT_STAGE=0`
    /// opt-out. Recipe, layout and kernel admission is by sealed verification
    /// before materialization, never by recipe name or a raw offload-exec
    /// veto; an actually CPU-split sealed selection refuses at the dispatch
    /// gate. `chunk_rows_env` is the raw `HIPFIRE_PREFILL_CHUNK_ROWS` value,
    /// when set.
    pub fn from_process(
        arch: &'a str,
        unified_memory: bool,
        structural: StructuralShape,
        max_seq: usize,
        chunk_rows_env: Option<&'a str>,
    ) -> Result<Self, String> {
        let explicit = crate::expert_residency::expert_vram_layers_from_env()
            .map_err(|error| format!("qwen4: {error}"))?;
        let stage = hipfire_config::developer_var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV).ok();
        Ok(Self {
            arch,
            unified_memory,
            expert_parallel: structural.expert_parallel,
            pipeline_band: structural.pipeline_band,
            routed_experts: structural.routed_experts,
            expert_keep: structural.expert_keep,
            explicit_vram_layers: explicit.is_some(),
            hotset_sidecar: structural.hotset_sidecar,
            whole_first: structural.whole_first,
            stage_opt_out: stage.as_deref() == Some("0"),
            npu_trunk: npu_trunk_selected(
                arch,
                crate::npu_trunk::NpuTrunkMode::from_env()
                    .map_err(|error| format!("qwen4: {error}"))?,
            ),
            chunk_rows_env,
            max_seq,
        })
    }
}

/// A single-device load the S1 admission sizes: the same census inputs the
/// product loader hands the placement
/// ([`crate::expert_residency::placement_inputs`]).
pub struct SingleDeviceLoad<'a> {
    pub gpu: &'a rdna_compute::Gpu,
    pub config: &'a Qwen4Config,
    pub weights: &'a [hipfire_runtime::weight_manifest::WeightEntry],
    pub bytes_of: &'a dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    pub backend: crate::Qwen4KvBackend,
    pub max_seq: usize,
    pub vmm_granularity: usize,
    /// The native MTP head attaches (sizes the session fixed snapshot).
    pub native_mtp: bool,
    /// The head is still attached after the host-mapped placement (sizes the
    /// reserve): `native_mtp` and the host-spill policy kept it.
    pub mtp_kept: bool,
    pub max_k: usize,
    pub state_format: crate::Qwen4StateFormat,
    pub scratch_mode: crate::gpu_forward::Qwen4ForwardScratchMode,
    /// What the admitted profile is for: the product, or the Hold 1 primitive
    /// self-tests (which charge their own diagnostic reserve inside the
    /// admission, never beside it).
    pub purpose: AdmissionPurpose,
    /// The artifact's source-byte symmetry scan of EVERY routed trunk layer
    /// (`artifact::routed_headers_symmetric`), run at most once and only when
    /// [`SymIu4RouteFacts::scan_needed`] holds. A caller with no artifact bytes passes a scan
    /// that answers `Ok(false)`: the route then stays on the layout.
    pub routed_source_symmetric: &'a dyn Fn() -> Result<bool, String>,
}

/// The two-family weight arena plan of a single-device S1 load: the resident
/// non-expert weights ([`crate::expert_residency::resident_split`]'s entries)
/// and the lazy F16 shadow owners the placement census charges, one shared
/// canonical root plan with every root bounded by the manifest's largest
/// existing eligible allocation. Roles and the shadow predicate come from
/// the manifest and the device's own shadow capability
/// (`Gpu::qwen4_f16_shadow_capable`); no quant, card or host name gates it.
///
/// This is the ONE device allocation-granule query of an S1 load (an
/// unanswered query refuses the load rather than modelling no rounding);
/// the census and the allocator read the plan's `granule_bytes` afterwards.
/// Built only where S1 selection asks for facts (never on the fast, UMA,
/// unmeasured or non-S1 paths), then borrowed by
/// [`single_device_facts_with_arenas`] and passed unchanged to the weight
/// fulfillment.
pub fn single_device_arena_plan(
    load: &SingleDeviceLoad<'_>,
) -> Result<std::sync::Arc<rdna_compute::byte_arena::ModelWeightArenaPlan>, String> {
    let granule_bytes = load
        .gpu
        .vmm_recommended_granularity()
        .map_err(|error| format!("qwen4: device allocation granularity: {error}"))?;
    crate::expert_residency::model_weight_arena_plan(
        load.weights,
        load.bytes_of,
        load.config.num_hidden_layers,
        load.gpu.qwen4_f16_shadow_capable(),
        granule_bytes,
    )
    .map_err(|error| format!("qwen4: {error}"))
}

/// Free device bytes now (`hipMemGetInfo`): the one sample type an admission
/// reads ([`AdmissionMemory`]).
fn device_free_bytes(gpu: &rdna_compute::Gpu) -> Result<u64, String> {
    gpu.device_mem_info()
        .map(|(free, _)| free as u64)
        .map_err(|error| format!("qwen4: device memory query: {error}"))
}

/// Exact [`LoadFacts`] of a single-device discrete load from one fresh
/// [`single_device_arena_plan`]: the print-memory and CPU callers' entry. It
/// samples free memory BEFORE the plan's factory and any host-link/copy-path
/// initialization and hands that earlier sample to
/// [`single_device_facts_with_arenas`]. A load that allocates weights hands
/// its own retained plan (and its own earlier sample) there instead.
pub fn single_device_facts<'a>(load: &SingleDeviceLoad<'a>) -> Result<LoadFacts<'a>, String> {
    let pre_init_free_bytes = device_free_bytes(load.gpu)?;
    let arenas = single_device_arena_plan(load)?;
    single_device_facts_with_arenas(load, &arenas, Some(pre_init_free_bytes))
}

/// Exact [`LoadFacts`] of a single-device discrete load: the reserve at the S1
/// tile and at the incumbent placement chunk, both
/// from [`crate::expert_residency::placement_inputs`] (the product's own
/// bottom-up census; no second convention), and the ACTUAL free device memory
/// ([`AdmissionMemory`]). The weights and F16 shadows are
/// charged the root backing of the PROVIDED `arenas` minus their unchanged
/// logical payload; the plan is never rebuilt here and its granule is the
/// census granule.
///
/// `pre_init_free_bytes` is the OPTIONAL earlier sample the caller took
/// before the host-link/copy-path and plan initialization (`None`: unknown;
/// never assumed). The incumbent's cached host-link resolution runs first,
/// and the actual `free_bytes` is sampled LAST, after every initialization
/// this preparation forces, so the runtime residency they created is charged
/// once, by the lower actual free. No stale earlier figure reaches the facts.
pub fn single_device_facts_with_arenas<'a>(
    load: &SingleDeviceLoad<'a>,
    arenas: &rdna_compute::byte_arena::ModelWeightArenaPlan,
    pre_init_free_bytes: Option<u64>,
) -> Result<LoadFacts<'a>, String> {
    // Beta's single incumbent helper (probes the host link once, cached).
    let placement_chunk = crate::gpu_forward::qwen4_prefill_chunk_requested(
        load.gpu,
        load.max_seq,
        load.scratch_mode,
    );
    // The tile reserve once: its census total sizes the placement and its
    // runtime charge is the credit the S1 runtime budget supplements.
    let tile_reserve = single_device_reserve(load, purpose_tile_rows(load.purpose))?;
    // Resolve the optional route before the final post-initialization memory sample.
    let resolved = resolve_window_route_with_scan(
        window_route_of(load.weights),
        SymIu4RouteFacts::of_gpu(load.gpu),
        || (load.routed_source_symmetric)(),
    )?;
    // The device's own allocation granule (`crate::expert_residency::granule_slack`
    // documents the backing model), read once by the plan's factory.
    let granule_bytes = arenas.granule_bytes as u64;
    let base_slack = single_device_granule_slack(
        load,
        purpose_tile_rows(load.purpose),
        &tile_reserve,
        granule_bytes,
        Some(arenas),
    )?;
    // One boundary stage's own part allocations (`allocate_prompt_stages`
    // makes one `zeros` per session fixed part), from the state's own list.
    let boundary_stage_slack = crate::expert_residency::granule_slack_total(
        crate::state::Qwen4State::session_fixed_part_sizes(load.config, load.state_format)
            .ok_or_else(|| "boundary stage part sizes overflow".to_string())?,
        granule_bytes,
    )?;
    let incumbent_reserve = single_device_reserve(load, placement_chunk)?;
    // Sampled LAST: every reserve, plan and cached link resolution above has
    // run, so this is the actual post-initialization figure.
    let free_bytes = device_free_bytes(load.gpu)?;
    Ok(LoadFacts {
        purpose: load.purpose,
        config: load.config,
        weights: load.weights,
        bytes_of: load.bytes_of,
        memory: AdmissionMemory {
            pre_init_free_bytes,
            free_bytes,
        },
        // S1 sizes the purpose's tile from the bottom-up census alone
        // (`AutoReserve::placement_bytes(true)`, which carries the F16 weight
        // shadow cold bound); the incumbent/log baseline keeps `bytes()`, the
        // larger of the historical floor and the census.
        tile_reserve: tile_reserve.placement_bytes(true),
        base_runtime_reserved_bytes: tile_reserve.census.runtime,
        granule: GranuleInputs {
            granule_bytes,
            base: base_slack,
            boundary_stage_slack,
        },
        incumbent_reserve: incumbent_reserve.placement_bytes(false),
        state_format: load.state_format,
        native_mtp: load.native_mtp,
        window_rows: S1_WINDOW_ROWS.min(load.max_seq),
        // The incumbent's own checkpoint cadence: the requested chunk resolved
        // once, exactly as the non-S1 load sizes and chunks with it.
        checkpoint_stride_rows: placement_chunk,
        window_route: resolved.route,
        sym_iu4_source_symmetric: resolved.source_symmetric,
    })
}

/// The load's [`crate::expert_residency::AutoReserve`] at `chunk_rows`, from
/// [`crate::expert_residency::placement_inputs`].
fn single_device_reserve(
    load: &SingleDeviceLoad<'_>,
    chunk_rows: usize,
) -> Result<crate::expert_residency::AutoReserve, String> {
    crate::expert_residency::placement_inputs(&crate::expert_residency::PlacementInputsArgs {
        gpu: load.gpu,
        config: load.config,
        weights: load.weights,
        bytes_of: load.bytes_of,
        backend: load.backend,
        use_ranges: false,
        max_seq: load.max_seq,
        vmm_granularity: load.vmm_granularity,
        mtp_kept: load.mtp_kept,
        max_k: load.max_k,
        routed_experts: !load.config.ffn.is_dense(),
        state_format: load.state_format,
        scratch_mode: load.scratch_mode,
        spans: &crate::expert_residency::EpExpertSpans::single(),
        chunk_rows,
    })
    .map(|inputs| inputs.reserve)
}

/// The base reserve's per-owner granule slack at `chunk_rows`
/// ([`crate::expert_residency::base_granule_slack`]): the load's own manifest,
/// state and forward plan, the reserve [`single_device_reserve`] returned and,
/// when given, the provided weight arena plan (`None`: individual weight and
/// shadow allocations).
fn single_device_granule_slack(
    load: &SingleDeviceLoad<'_>,
    chunk_rows: usize,
    reserve: &crate::expert_residency::AutoReserve,
    granule_bytes: u64,
    weight_arenas: Option<&rdna_compute::byte_arena::ModelWeightArenaPlan>,
) -> Result<crate::expert_residency::GranuleSlack, String> {
    crate::expert_residency::base_granule_slack(&crate::expert_residency::GranuleSlackArgs {
        gpu: load.gpu,
        config: load.config,
        weights: load.weights,
        bytes_of: load.bytes_of,
        backend: load.backend,
        max_seq: load.max_seq,
        vmm_granularity: load.vmm_granularity,
        mtp_kept: load.mtp_kept,
        max_k: load.max_k,
        state_format: load.state_format,
        scratch_mode: load.scratch_mode,
        chunk_rows,
        reserve: *reserve,
        granule: granule_bytes,
        weight_arenas,
    })
}

/// The feasible (tile, resident layers) frontier ([`tile_frontier`]) of a live
/// single-device load at the measured `link_gbps`: the same facts and
/// per-tile reserves the loader's admission reads, from the same
/// [`project_at_tile`]. Emits memory only; the rate model is Main's, and the
/// runtime still admits only the purpose's tile ([`purpose_tile_rows`], the
/// point with [`FrontierPoint::selected`]) with the largest count that fits.
pub fn single_device_tile_frontier(
    load: &SingleDeviceLoad<'_>,
    link_gbps: f64,
) -> Result<Vec<FrontierPoint>, String> {
    let pre_init_free_bytes = device_free_bytes(load.gpu)?;
    let arenas = single_device_arena_plan(load)?;
    let facts = single_device_facts_with_arenas(load, &arenas, Some(pre_init_free_bytes))?;
    let inputs = load_admission_inputs(&facts, link_gbps).map_err(|decline| decline.to_string())?;
    let granule_bytes = inputs.granule.granule_bytes;
    tile_frontier(&inputs, |tile_rows| {
        let reserve = single_device_reserve(load, tile_rows)?;
        let slack =
            single_device_granule_slack(load, tile_rows, &reserve, granule_bytes, Some(&*arenas))?;
        Ok((reserve, slack))
    })
}

/// How a request's prefill output is captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureKind {
    /// Final row only (or silent): the product path.
    None,
    /// Bounded row capture that fits the admitted arena (MTP target capture).
    Bounded,
    /// Every row's logits in caller-owned storage: stays incumbent.
    AllRows,
    /// Externally captured or retained prefill: stays incumbent.
    External,
}

/// A request as the architecture sees it before any mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestShape {
    pub rows: usize,
    pub entry_pos: usize,
    /// Session snapshot cuts inside the request.
    pub session_cuts: usize,
    pub capture: CaptureKind,
}

/// A request S1 will run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestVerdict {
    pub rows: usize,
    /// Expert sweeps: 1 means one sweep for the whole prompt.
    pub windows: usize,
}

/// Why a request stays on the incumbent route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestBypass {
    KillSwitch,
    TooShort {
        rows: usize,
        minimum: usize,
    },
    AllRowCapture,
    ExternalCapture,
    TooManyBoundaries {
        cuts: usize,
        maximum: usize,
    },
    /// The profile's tile is not a valid chunk for this request.
    NoRows,
    /// SP is armed, the request is a single window of 5..=tile rows, and that
    /// window is not SP-qualified: the incumbent runs.
    Sp(crate::slowlink_sp::SpBypass),
}

/// Judge a request against the frozen profile, before any mutation or S1
/// allocation. `Err` is an incumbent request that uses only resources the
/// profile already admitted.
pub fn qualify_request(
    mode: SlowlinkMode,
    profile: &SlowlinkProfile,
    request: &RequestShape,
) -> Result<RequestVerdict, RequestBypass> {
    let minimum = match mode {
        SlowlinkMode::Off => return Err(RequestBypass::KillSwitch),
        SlowlinkMode::Auto => profile.tile_rows.saturating_add(1),
        SlowlinkMode::Force => S1_MIN_FORCED_REQUEST_ROWS,
    };
    if request.rows == 0 {
        return Err(RequestBypass::NoRows);
    }
    match request.capture {
        CaptureKind::None | CaptureKind::Bounded => {}
        CaptureKind::AllRows => return Err(RequestBypass::AllRowCapture),
        CaptureKind::External => return Err(RequestBypass::ExternalCapture),
    }
    if request.rows < minimum {
        return Err(RequestBypass::TooShort {
            rows: request.rows,
            minimum,
        });
    }
    if request.session_cuts > S1_MAX_BOUNDARY_SNAPSHOTS {
        return Err(RequestBypass::TooManyBoundaries {
            cuts: request.session_cuts,
            maximum: S1_MAX_BOUNDARY_SNAPSHOTS,
        });
    }
    Ok(RequestVerdict {
        rows: request.rows,
        windows: profile.windows_for(request.rows),
    })
}

/// [`qualify_request`] with SP armed (HOOK-DESIGN §5.1, per request).
///
/// A request of `SP_MIN_ROWS ..= profile.tile_rows` rows is one window of one
/// tile, below S1's own multi-chunk floor: S1 admits it **only if that window
/// is SP-qualified** ([`crate::slowlink_sp::qualify_sp_shape`]); otherwise the
/// incumbent runs, with the named reason. Every other request (off, shorter,
/// or longer than one tile) is exactly [`qualify_request`], with SP decided per
/// window by the runtime.
pub(crate) fn qualify_request_sp(
    mode: SlowlinkMode,
    profile: &SlowlinkProfile,
    sp: &crate::slowlink_sp::SpProfile,
    request: &RequestShape,
) -> Result<RequestVerdict, RequestBypass> {
    let single_tile = (crate::slowlink_sp::SP_MIN_ROWS..=profile.tile_rows).contains(&request.rows);
    if mode == SlowlinkMode::Off || !single_tile {
        return qualify_request(mode, profile, request);
    }
    match request.capture {
        CaptureKind::None | CaptureKind::Bounded => {}
        CaptureKind::AllRows => return Err(RequestBypass::AllRowCapture),
        CaptureKind::External => return Err(RequestBypass::ExternalCapture),
    }
    // One window of one tile of `rows` routing rows (rows >= 5: never a one-row
    // tile); K1 rows are the one-row decode tiles, none here.
    crate::slowlink_sp::qualify_sp_shape(
        sp,
        request.rows,
        request.session_cuts,
        None,
        0,
        request.capture,
    )
    .map_err(RequestBypass::Sp)?;
    Ok(RequestVerdict {
        rows: request.rows,
        windows: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_residency::{auto_placement_layers, PlacementCharges};
    /// The 16,384-row window the pinned j1778/j1786 reference ledgers were
    /// measured at. The product window is [`S1_WINDOW_ROWS`] (22 tiles); the
    /// reference fixtures keep their historical window so their slack pins stay
    /// real measurements, and the window change is pinned as a delta in
    /// `slowlink_admission_window_charge_delta_is_pinned`.
    const REFERENCE_WINDOW_ROWS: usize = 16_384;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    // Flash-Next MQ4 packed: 512 experts of 1,740,800 gate/up and 870,400 down bytes.
    const GATE_UP_LAYER: u64 = 512 * 1_740_800;
    const DOWN_LAYER: u64 = 512 * 870_400;
    const LAYER: u64 = GATE_UP_LAYER + DOWN_LAYER;

    fn shape() -> LoadShape<'static> {
        LoadShape {
            arch: "gfx1100",
            unified_memory: false,
            expert_parallel: false,
            pipeline_band: false,
            routed_experts: true,
            expert_keep: false,
            explicit_vram_layers: false,
            hotset_sidecar: false,
            whole_first: false,
            stage_opt_out: false,
            npu_trunk: false,
            chunk_rows_env: None,
            max_seq: 16640,
        }
    }

    fn geometry() -> ModelGeometry {
        ModelGeometry::from_config(&crate::config::compact_test_config(), WindowRoute::SlabIu4)
            .unwrap()
    }

    fn inputs(free: u64) -> AdmissionInputs {
        AdmissionInputs {
            geometry: geometry(),
            memory: AdmissionMemory::actual(free),
            non_expert: 5_364_000_000,
            layer_expert_bytes: LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: 48,
            base_reserve: 5261 * MIB,
            baseline_layers: 10,
            fixed_state_bytes: 64 * MIB,
            pending_save_bytes: 4 * 2560 * 4,
            diagnostic_bytes: 0,
            base_runtime_reserved_bytes: 0,
            granule: GranuleInputs::default(),
            checkpoint_stride_rows: 8192,
            window_rows: REFERENCE_WINDOW_ROWS,
            link_gbps: 7.09,
            purpose: AdmissionPurpose::Product,
            sym_iu4_source_symmetric: None,
        }
    }

    /// The non-S1 checkpoint cadence survives an attached S1 profile: the
    /// stride is the incumbent's requested chunk (8192 on the narrow target),
    /// never the mixer tile. The boundaries are the incumbent's own
    /// planner (`session_cache::stride_boundaries`, GLOBAL multiples of the
    /// stride in `(entry, end]`), fed the profile's stride: a 16K prompt has
    /// one interior boundary, resumed or not.
    #[test]
    fn slowlink_admission_checkpoint_cadence_is_the_incumbent_chunk_not_the_tile() {
        use hipfire_runtime::session_cache::stride_boundaries;
        let profile = admit(&inputs(free_for(&inputs(0), 10))).unwrap();
        assert_eq!(profile.tile_rows, S1_PRODUCT_TILE_ROWS);
        assert_eq!(profile.checkpoint_stride_rows, 8192);
        let stride = profile.checkpoint_stride_rows;
        let interior = |stride: usize, entry: usize, end: usize| -> Vec<usize> {
            stride_boundaries(stride, &[], None, entry, end)
                .into_iter()
                .filter(|&boundary| boundary < end)
                .collect()
        };
        // Fresh 16K: exactly one interior boundary.
        assert_eq!(interior(stride, 0, 16384), vec![8192]);
        // Resumed entries: global multiples, never entry + k * stride.
        assert_eq!(interior(stride, 3, 16384), vec![8192], "not 8195");
        assert_eq!(interior(stride, 1537, 16384), vec![8192]);
        assert_eq!(interior(stride, 8193, 16384), Vec::<usize>::new());
        // A short prompt or an exact stride has no interior boundary.
        assert!(interior(stride, 0, 8192).is_empty());
        assert!(interior(stride, 0, 100).is_empty());
        // The mixer tile would have placed an interior boundary every tile
        // (21 at 768 rows): it is never the checkpoint cadence.
        let tile_boundaries = (16384 - 1) / profile.tile_rows;
        assert_eq!(interior(profile.tile_rows, 0, 16384).len(), tile_boundaries);
        assert_eq!(tile_boundaries, 21, "768-row product tile");
        assert_ne!(
            interior(stride, 0, 16384),
            interior(profile.tile_rows, 0, 16384)
        );
    }

    fn census_total(inputs: &AdmissionInputs) -> u64 {
        let pages = PanelPageLayout::new(
            inputs.geometry.num_experts,
            inputs.gate_up_layer_bytes,
            inputs.down_layer_bytes,
        )
        .unwrap();
        let arena = PromptArenaLayout::new(
            &inputs.geometry,
            inputs.window_rows,
            purpose_tile_rows(inputs.purpose),
        )
        .unwrap();
        pages.total_bytes().unwrap()
            + route_storage_bytes(&inputs.geometry, inputs.window_rows).unwrap()
            + a4_sidecar_bytes(&inputs.geometry, inputs.window_rows).unwrap()
            + group_scatter_bytes(&inputs.geometry, inputs.window_rows).unwrap()
            + arena.total_bytes
            + S1_MAX_BOUNDARY_SNAPSHOTS as u64 * inputs.fixed_state_bytes
            + inputs.pending_save_bytes
            + S1_RUNTIME_OVERHEAD_BYTES.saturating_sub(inputs.base_runtime_reserved_bytes)
            + inputs.diagnostic_bytes
    }

    /// Free VRAM that keeps exactly `layers` layers beside the S1 charge.
    fn free_for(inputs: &AdmissionInputs, layers: u64) -> u64 {
        inputs.non_expert
            + inputs.base_reserve
            + EXPERT_MARGIN
            + census_total(inputs)
            + layers * LAYER
            + 1
    }
    const EXPERT_MARGIN: u64 = 512 << 20;

    #[test]
    fn slowlink_admission_selector_modes_follow_the_typed_value() {
        assert_eq!(SlowlinkMode::from_typed(Some(false)), SlowlinkMode::Off);
        assert_eq!(SlowlinkMode::from_typed(Some(true)), SlowlinkMode::Force);
        assert_eq!(SlowlinkMode::from_typed(None), SlowlinkMode::Auto);
        assert_eq!(SLOWLINK_PREFILL_ENV, "HIPFIRE_QWEN4_SLOWLINK_PREFILL");
        assert!(hipfire_config::is_typed_bool_developer_key(
            "developer.qwen4_slowlink_prefill"
        ));
    }

    #[test]
    fn slowlink_admission_unset_selects_automatically_with_kill_switch_only() {
        assert_eq!(preflight(SlowlinkMode::Auto, &shape()), Ok(()));
        assert_eq!(
            preflight(SlowlinkMode::Off, &shape()),
            Err(SlowlinkDecline::KillSwitch)
        );
        assert_eq!(preflight(SlowlinkMode::Force, &shape()), Ok(()));
    }

    #[test]
    fn slowlink_admission_arch_is_identity_not_an_allowlist_gate() {
        // The exact-arch name gate is removed: probe-free admission never
        // refuses by arch string. Every arch value passes preflight; only the
        // sealed panel/codec/kernel verdict before materialization admits.
        for arch in [
            "gfx906", "gfx908", "gfx90a", "gfx942", "gfx1010", "gfx1030", "gfx1031", "gfx1100",
            "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
            "gfx1100x", "",
        ] {
            let load = LoadShape { arch, ..shape() };
            assert_eq!(preflight(SlowlinkMode::Force, &load), Ok(()), "{arch}");
        }
        // Unified memory still declines: there is no link to stream over.
        let load = LoadShape {
            arch: "gfx1201",
            unified_memory: true,
            expert_parallel: true,
            ..shape()
        };
        assert_eq!(
            preflight(SlowlinkMode::Force, &load),
            Err(SlowlinkDecline::UnifiedMemory)
        );
    }

    #[test]
    fn slowlink_admission_link_truth_table() {
        let verdict = |gbps: f64| link_verdict(HostLink::Discrete { gbps });
        assert_eq!(verdict(7.09), Ok(7.09));
        assert_eq!(verdict(12.0), Ok(12.0));
        assert_eq!(verdict(15.99), Ok(15.99));
        assert_eq!(verdict(16.0), Err(SlowlinkDecline::FastLink { gbps: 16.0 }));
        assert_eq!(verdict(26.0), Err(SlowlinkDecline::FastLink { gbps: 26.0 }));
        assert_eq!(verdict(50.0), Err(SlowlinkDecline::FastLink { gbps: 50.0 }));
        assert_eq!(
            link_verdict(HostLink::Unified),
            Err(SlowlinkDecline::UnifiedMemory)
        );
        assert_eq!(
            link_verdict(HostLink::Unmeasured),
            Err(SlowlinkDecline::UnmeasuredLink)
        );
        // Unusable figures classify as unmeasured, never narrow.
        for bad in [
            Some(f64::NAN),
            Some(0.0),
            Some(-3.0),
            Some(f64::INFINITY),
            None,
        ] {
            assert_eq!(
                link_verdict(HostLink::classify(false, bad)),
                Err(SlowlinkDecline::UnmeasuredLink),
                "{bad:?}"
            );
        }
        assert_eq!(
            link_verdict(HostLink::classify(true, Some(7.0))),
            Err(SlowlinkDecline::UnifiedMemory)
        );
    }

    #[test]
    fn slowlink_admission_unsupported_modes_decline_before_any_probe() {
        // Recipe-name and raw offload-exec gates are removed: sealed
        // panel/codec/kernel verdicts refuse before materialization. The
        // remaining structural modes still decline probe-free.
        let cases: [(&str, LoadShape<'static>); 9] = [
            (
                "ep",
                LoadShape {
                    expert_parallel: true,
                    ..shape()
                },
            ),
            (
                "pp",
                LoadShape {
                    pipeline_band: true,
                    ..shape()
                },
            ),
            (
                "dense",
                LoadShape {
                    routed_experts: false,
                    ..shape()
                },
            ),
            (
                "keep",
                LoadShape {
                    expert_keep: true,
                    ..shape()
                },
            ),
            (
                "layers",
                LoadShape {
                    explicit_vram_layers: true,
                    ..shape()
                },
            ),
            (
                "hotset",
                LoadShape {
                    hotset_sidecar: true,
                    ..shape()
                },
            ),
            (
                "whole-first",
                LoadShape {
                    whole_first: true,
                    ..shape()
                },
            ),
            (
                "stage off",
                LoadShape {
                    stage_opt_out: true,
                    ..shape()
                },
            ),
            (
                "npu",
                LoadShape {
                    npu_trunk: true,
                    ..shape()
                },
            ),
        ];
        for (name, load) in cases {
            assert!(
                matches!(
                    preflight(SlowlinkMode::Force, &load),
                    Err(SlowlinkDecline::Unsupported(_))
                ),
                "{name}"
            );
        }
        assert_eq!(
            preflight(
                SlowlinkMode::Force,
                &LoadShape {
                    unified_memory: true,
                    ..shape()
                }
            ),
            Err(SlowlinkDecline::UnifiedMemory)
        );
        for max_seq in [0, 1, 1536] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &LoadShape { max_seq, ..shape() }),
                Err(SlowlinkDecline::ShortContext { max_seq })
            );
        }
        for max_seq in [1537, 3072] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &LoadShape { max_seq, ..shape() }),
                Ok(()),
                "{max_seq}"
            );
        }
    }

    #[test]
    fn slowlink_admission_selected_npu_truth_table() {
        use crate::npu_trunk::NpuTrunkMode::{Auto, Emulate, Off, On};
        // `auto` resolves to off outside gfx1151, so the default never selects
        // the NPU on the S1 arch; an explicit `on` or `emulate` does, anywhere.
        for arch in ["gfx1100", "gfx1201", "gfx906", "gfx1030", "gfx942"] {
            assert!(!npu_trunk_selected(arch, Off), "{arch}");
            assert!(!npu_trunk_selected(arch, Auto), "{arch}");
            assert!(npu_trunk_selected(arch, On), "{arch}");
            assert!(npu_trunk_selected(arch, Emulate), "{arch}");
        }
        assert!(npu_trunk_selected("gfx1151", Auto));
        assert!(!npu_trunk_selected("gfx1151", Off));
        // A selected NPU is an incumbent decline on an otherwise admitted load,
        // before any probe or reserve.
        let npu = |mode| LoadShape {
            npu_trunk: npu_trunk_selected("gfx1100", mode),
            ..shape()
        };
        assert_eq!(preflight(SlowlinkMode::Force, &npu(Auto)), Ok(()));
        for mode in [On, Emulate] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &npu(mode)),
                Err(SlowlinkDecline::Unsupported("NPU trunk selected"))
            );
        }
    }

    #[test]
    fn slowlink_admission_explicit_chunk_rows_must_equal_the_purpose_tile() {
        let with = |env| LoadShape {
            chunk_rows_env: Some(env),
            ..shape()
        };
        // Probe-free: any legal tile passes the preflight (the purpose's tile
        // is known only after admission, checked below).
        for legal in ["1536", " 1536 ", "1280", "1024", "768", "512"] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &with(legal)),
                Ok(()),
                "{legal}"
            );
        }
        // Rounded down to whole 256-row tiles like the chunk policy.
        assert_eq!(preflight(SlowlinkMode::Force, &with("1600")), Ok(()));
        // Not a legal S1 tile: above the maximum or below the WMMA threshold.
        for illegal in ["8192", "2048", "256"] {
            let rows = illegal.parse().unwrap();
            assert_eq!(
                preflight(SlowlinkMode::Force, &with(illegal)),
                Err(SlowlinkDecline::ExplicitChunk { rows, tile: None }),
                "{illegal}"
            );
        }
        // Unusable values are unset, as in the chunk policy.
        for env in ["", "abc", "100", "-1"] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &with(env)),
                Ok(()),
                "{env:?}"
            );
        }
        // After admission an explicit chunk must equal the PURPOSE's tile:
        // a product load admits 768, so 1536 is the incumbent (not a capacity
        // refusal) and 768 admits.
        let select = |env| {
            select_profile(
                SlowlinkMode::Force,
                &with(env),
                || HostLink::Discrete { gbps: 7.09 },
                |_| Ok(inputs(free_for(&inputs(0), 10))),
            )
        };
        assert_eq!(select("1536"), Ok(None));
        assert_eq!(select("512"), Ok(None));
        let profile = select("768").unwrap().expect("the purpose tile admits");
        assert_eq!(profile.tile_rows, S1_PRODUCT_TILE_ROWS);
        assert!(select("").unwrap().is_some());
    }
    #[test]
    fn slowlink_admission_source_priced_mixed_mq2r2_e8p_declines_pre_mutation() {
        use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
        use rdna_compute::DType;
        // Source-priced mixed MQ2-R2 / E8P routed experts (one gate/up MQ2-I4,
        // one down E8P-I4) decline pre-mutation even though incumbent kernels
        // exist for each dtype alone: the sealed panel/codec/kernel tables
        // have no mixed route, and the decline names the layout, not a host.
        let entry = |name: &str, dtype: DType| {
            let mut entry = WeightEntry::model(
                name.to_string(),
                vec![512, 1280, 2560],
                dtype,
                hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
            );
            entry.layer = Some(0);
            entry.residency = WeightResidency::Resident;
            entry
        };
        let weights = vec![
            entry(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                DType::MQ2I4G128R256,
            ),
            entry(
                "model.language_model.layers.0.mlp.experts.down_proj",
                DType::E8PI4G128R256,
            ),
        ];
        let bytes_of = |_: &WeightEntry| Some(1_740_800u64);
        assert!(mixed_source_priced_mq2r2_e8p(&weights, &bytes_of));
        // Uniform dtypes (MQ4 only, or E8P only) are not a mixed pair: the
        // sealed panel/codec/kernel verdict decides them, never this decline.
        let uniform = vec![
            entry(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                DType::MQ4G256V2,
            ),
            entry(
                "model.language_model.layers.0.mlp.experts.down_proj",
                DType::MQ4G256V2,
            ),
        ];
        assert!(!mixed_source_priced_mq2r2_e8p(&uniform, &bytes_of));
        let e8p_only = vec![
            entry(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                DType::E8PI4G128R256,
            ),
            entry(
                "model.language_model.layers.0.mlp.experts.down_proj",
                DType::E8PI4G128R256,
            ),
        ];
        assert!(!mixed_source_priced_mq2r2_e8p(&e8p_only, &bytes_of));
    }

    #[test]
    fn slowlink_admission_page_layout_is_the_pool_reserve() {
        let pages = PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER).unwrap();
        assert_eq!(pages.experts_per_page, 32);
        assert_eq!(pages.pages_per_layer, 16);
        assert_eq!(pages.total_pages(), 18);
        assert_eq!(pages.gate_up_expert_bytes, 1_740_800);
        assert_eq!(pages.down_expert_bytes, 870_400);
        // Payload bound: 32 experts' extents plus worst-case unmerged-run slack.
        let extents = 32 * 2_611_200u64;
        assert_eq!(pages.payload_bytes, extents + 64 * 2 * 255);
        let spec = pages.pool_spec().unwrap();
        assert_eq!(spec.active_slots, 2);
        assert_eq!(spec.lookahead_pages, 16);
        assert_eq!(spec.experts_per_slot, 32);
        assert_eq!(spec.num_experts, 512);
        assert_eq!(spec.payload_bytes as u64, pages.payload_bytes);
        let reserve = spec.reserve().unwrap();
        // ONE tail pad per page (rounded to 256) and two u64 pointer tables of
        // every expert per page, exactly the pool's own accounting.
        assert_eq!(
            pages.page_bytes,
            (pages.payload_bytes + 4096).div_ceil(256) * 256
        );
        assert_eq!(pages.table_bytes_per_page, 2 * 512 * 8);
        assert_eq!(reserve.page_bytes as u64, pages.page_bytes);
        assert_eq!(reserve.payload as u64, 18 * pages.page_bytes);
        assert_eq!(reserve.tables as u64, 18 * pages.table_bytes_per_page);
        assert_eq!(reserve.staging as u64, pages.host_staging_bytes);
        assert_eq!(pages.total_bytes().unwrap(), reserve.device_total as u64);
        assert_eq!(
            pages.ring_bytes().unwrap(),
            2 * pages.device_page_bytes().unwrap()
        );
        assert_eq!(
            pages.lookahead_bytes().unwrap(),
            16 * pages.device_page_bytes().unwrap()
        );
        // Plan: ~1.401 GiB of payload beside tables and pads.
        let payload = 18 * extents;
        assert!(
            payload > 1_500_000_000 && payload < 1_510_000_000,
            "{payload}"
        );
        assert_eq!(pages.timeline_events(), 4 * 2048 + 2 * 64);
    }

    #[test]
    fn slowlink_admission_page_layout_refuses_non_uniform_and_overflow() {
        assert!(PanelPageLayout::new(0, 1, 1).is_err());
        assert!(PanelPageLayout::new(512, 0, DOWN_LAYER).is_err());
        assert!(PanelPageLayout::new(512, GATE_UP_LAYER, 0).is_err());
        assert!(PanelPageLayout::new(512, GATE_UP_LAYER + 1, DOWN_LAYER).is_err());
        assert!(PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER + 7).is_err());
        assert!(PanelPageLayout::new(1, u64::MAX, 1).is_err());
        assert!(PanelPageLayout::new(1, 1, u64::MAX).is_err());
        // A layer that is not a multiple of 32 experts still has a short last page.
        let odd = PanelPageLayout::new(40, 40 * 256, 40 * 256).unwrap();
        assert_eq!(odd.pages_per_layer, 2);
    }

    #[test]
    fn slowlink_admission_arena_is_the_hc_streams_only_and_exact() {
        let geometry = geometry();
        let arena = PromptArenaLayout::new(&geometry, S1_WINDOW_ROWS, S1_TILE_ROWS).unwrap();
        assert_eq!(arena.tiles, 11);
        assert_eq!(arena.rows, S1_WINDOW_ROWS);
        let wide = (geometry.hc_count * geometry.hidden) as u64;
        let rows = S1_WINDOW_ROWS as u64;
        // Saved prepared input, route maps and accumulator belong to the
        // dispatch crate's route storage, never duplicated in the arena.
        assert_eq!(ARENA_FIELDS, [ArenaField::HcStreams]);
        // Predicate pins: a tile of >= 512 rows is BF16; at most six
        // sub-512-row (F32) tiles per window; the arena changes with either.
        assert_eq!(S1_F32_TILE_ROWS_MAX, 511);
        assert_eq!(rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS, 512);
        assert_eq!(S1_F32_STREAM_ROWS, 6 * 511);
        assert_eq!(S1_F32_STREAM_ROWS, 3066);
        let bytes = (rows + 3066 + S1_TILE_ROWS as u64) * wide * 2;
        assert_eq!(arena.span(ArenaField::HcStreams).bytes, bytes);
        assert_eq!(arena.span(ArenaField::HcStreams).offset, 0);
        assert_eq!(arena.total_bytes % S1_ALIGN, 0);
        assert!(arena.total_bytes >= bytes);
        // Far below the all-F32 window, which a BF16-off load must bypass.
        assert!(arena.total_bytes < rows * wide * 4);
        assert!(arena.total_bytes - arena.span(ArenaField::HcStreams).end() < S1_ALIGN);
        // One window row more or fewer changes the allocation and the census alike.
        let shorter = PromptArenaLayout::new(&geometry, S1_WINDOW_ROWS - 1, S1_TILE_ROWS).unwrap();
        assert!(shorter.total_bytes < arena.total_bytes);
        assert!(PromptArenaLayout::new(&geometry, 0, S1_TILE_ROWS).is_err());
        assert!(PromptArenaLayout::new(&geometry, 1, 0).is_err());
        assert!(PromptArenaLayout::new(&geometry, usize::MAX / 2, S1_TILE_ROWS).is_err());
    }

    // The real j1778 inputs: the HIP total and PRE-LOAD free the hold measured,
    // the real non-expert and per-layer expert bytes, the real artifact's F16
    // shadow owners (model-index metadata, 291 owners) and the 16,640-token
    // Q8/Q8 context of the gate (aggregate window 16,384, 18 pages).
    const J1778_TOTAL: u64 = 25_753_026_560;
    const J1778_PRELOAD_FREE: u64 = 25_713_180_672;
    const J1778_NON_EXPERT: u64 = 5_364_055_296;
    const J1778_SHADOWS: u64 = 1_298_923_520;
    const J1778_MAX_SEQ: usize = 16_640;
    const J1778_HISTORICAL_1536: u64 = 5_517_207_808;
    const J1778_MARGIN: u64 = 536_870_912;
    const J1778_PRIMITIVE_DIAGNOSTIC: u64 = 217_821_184;

    fn j1778_state_format() -> crate::Qwen4StateFormat {
        crate::Qwen4StateFormat {
            qsa: rdna_compute::tensor_ops::QsaKvFormat::Q8,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        }
    }

    /// The load's base reserve at a `tile`-row chunk from the real helpers
    /// (gfx1100, discrete Legacy KV, Full scratch, Q8/Q8, no MTP, no gather):
    /// the state, forward and workspace census of that tile PLUS the F16 weight
    /// shadow cold bound of the real artifact through the shared predicate.
    fn j1778_reserve_at(tile: usize) -> crate::expert_residency::AutoReserve {
        j1778_reserve_for(tile, J1778_MAX_SEQ)
    }

    /// [`j1778_reserve_at`] for a load of `max_seq` tokens (the KV charge).
    fn j1778_reserve_for(tile: usize, max_seq: usize) -> crate::expert_residency::AutoReserve {
        use crate::expert_residency as residency;
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        let config = crate::config::compact_test_config();
        let format = j1778_state_format();
        let context =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Legacy, max_seq, max_seq, 1);
        let reserve_context = crate::Qwen4ContextCommit::for_expert_reserve(
            crate::Qwen4KvBackend::Legacy,
            false,
            max_seq,
            tile,
            1,
        );
        let historical = residency::auto_vram_reserve(
            &config,
            &reserve_context,
            tile,
            Full,
            format.layout(),
            None,
            None,
        )
        .unwrap();
        let state = residency::state_census(&config, format, &context, tile, Full).unwrap();
        let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
            hidden_size: config.hidden_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hc_count: config.hc_count,
            hc_lowrank: config.hc_lowrank,
            gdn_format: None,
            routed_i4: true,
            include_short_tails: true,
        };
        let f16_shadows = residency::f16_shadow_charge(
            true,
            residency::f16_shadow_route_rows_with("gfx1100", tile, true, 48, false),
            &residency::j1778_bf16_manifest(),
            config.num_hidden_layers,
        )
        .unwrap();
        residency::AutoReserve::new(
            Some(historical),
            residency::ReserveCensus {
                qsa_context: state.qsa_context,
                state: state.state,
                forward: state.forward,
                f16_shadows,
                shared_i8_shadows: 0,
                workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(tile, &dims)
                    .unwrap(),
                qsa_selector: 0,
                mtp: 0,
                gather: 0,
                runtime: 0,
            },
        )
        .unwrap()
    }

    /// The real j1778 admission inputs for a route, purpose and free figure, at
    /// the PURPOSE's tile exactly as the loader hands them to [`admit`]: the
    /// base reserve is the census at [`purpose_tile_rows`] and the purpose is
    /// the load's own, never inferred from the diagnostic bytes.
    fn j1778_inputs(route: WindowRoute, purpose: AdmissionPurpose, free: u64) -> AdmissionInputs {
        j1778_inputs_at(route, purpose, free, purpose_tile_rows(purpose))
    }

    /// [`j1778_inputs`] with the base reserve at an explicit `tile` (the
    /// historical 1,536-row counterfactual, admitted through [`admit_at_tile`]).
    fn j1778_inputs_at(
        route: WindowRoute,
        purpose: AdmissionPurpose,
        free: u64,
        tile: usize,
    ) -> AdmissionInputs {
        let config = crate::config::compact_test_config();
        let geometry = ModelGeometry::from_config(&config, route).unwrap();
        let reserve = j1778_reserve_at(tile);
        AdmissionInputs {
            geometry,
            memory: AdmissionMemory::actual(free),
            non_expert: J1778_NON_EXPERT,
            layer_expert_bytes: LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: config.num_hidden_layers,
            base_reserve: reserve.placement_bytes(true),
            baseline_layers: 0,
            fixed_state_bytes: crate::state::Qwen4State::session_fixed_stage_bytes(
                &config,
                j1778_state_format(),
            )
            .unwrap(),
            pending_save_bytes: 0,
            diagnostic_bytes: diagnostic_budget_bytes(
                &geometry,
                purpose,
                GATE_UP_LAYER,
                DOWN_LAYER,
            )
            .unwrap(),
            base_runtime_reserved_bytes: reserve.census.runtime,
            granule: GranuleInputs::default(),
            checkpoint_stride_rows: 8192,
            window_rows: REFERENCE_WINDOW_ROWS.min(J1778_MAX_SEQ),
            link_gbps: 7.09,
            purpose,
            sym_iu4_source_symmetric: None,
        }
    }

    /// `free - needed` for `layers` resident layers of an admitted `profile`:
    /// needed is the non-expert weights, the layers, the base reserve at the
    /// profile's tile (F16 shadow cold bound inside), every S1 term (runtime
    /// budget and diagnostic included) and the 512 MiB safety margin on top.
    fn j1778_slack(inputs: &AdmissionInputs, profile: &SlowlinkProfile, layers: u64) -> i128 {
        let needed = i128::from(inputs.non_expert)
            + i128::from(layers) * i128::from(LAYER)
            + i128::from(inputs.base_reserve)
            + i128::from(profile.census.total().unwrap())
            + i128::from(J1778_MARGIN);
        i128::from(inputs.memory.free_bytes) - needed
    }

    /// The legal rungs' base reserves (census incl. the F16 shadow cold bound)
    /// and the N=10 slack of each purpose on each route, from the real
    /// fixture: the frontier the product and primitive tiles were chosen on
    /// (1536, 1280, 1024, 768, 512).
    const J1778_RUNG_BASE: [u64; 5] = [
        3_536_042_980,
        3_251_928_164,
        2_967_813_348,
        2_683_698_532,
        2_399_583_716,
    ];
    const J1778_AGG_PRODUCT_SLACK: [i128; 5] = [
        -711_513_124,
        -422_155_428,
        -132_797_732,
        156_559_964,
        445_917_660,
    ];
    const J1778_AGG_PRIMITIVE_SLACK: [i128; 5] = [
        -929_334_308,
        -639_976_612,
        -350_618_916,
        -61_261_220,
        228_096_476,
    ];
    const J1778_SLAB_PRODUCT_SLACK: [i128; 5] = [
        -709_968_932,
        -420_611_236,
        -131_253_540,
        158_104_156,
        447_461_852,
    ];
    const J1778_SLAB_PRIMITIVE_SLACK: [i128; 5] = [
        -927_790_116,
        -638_432_420,
        -349_074_724,
        -59_717_028,
        229_640_668,
    ];
    /// The N=10 slack of each rung on the symmetric IU4 aggregate route: the F16
    /// aggregate slack less the route's own census delta (both A4 sidecars charged,
    /// 82,575,360 B at the 16K window, minus the window F16 input and slot map the IU4
    /// layout drops, 83,886,080 + 2,048 * 4 B of region: net −1,318,912 B). Every other
    /// census term (ring, lookahead, group scatter, arena, boundary, runtime, diagnostic)
    /// is the aggregate route's byte for byte, so the delta is rung- and purpose-independent.
    /// Pinned by `slowlink_residency_sym_iu4_route_adapts_before_declining_on_a_base_miss`.
    const J1778_SYM_PRODUCT_SLACK: [i128; 5] = [
        -710_194_212,
        -420_836_516,
        -131_478_820,
        157_878_876,
        447_236_572,
    ];
    const J1778_SYM_PRIMITIVE_SLACK: [i128; 5] = [
        -928_015_396,
        -638_657_700,
        -349_300_004,
        -59_942_308,
        229_415_388,
    ];

    /// The HISTORICAL 1,536-row COUNTERFACTUAL on the real 24 GiB XTX fixture
    /// with the CORRECTED source census (admitted through [`admit_at_tile`],
    /// never by production [`admit`]): the base reserve at the 1,536-row tile is
    /// the 2,237,119,460 B source census (the 2,237,111,724 B first measured
    /// with the 32-lane slab union descriptor, plus the 7,736 B by which the
    /// 64-lane native row-tile descriptor, 8 tiles x 1,928 B = 15,424 B, exceeds
    /// that 7,688 B) PLUS the 1,298,923,520 B F16 weight
    /// shadow cold bound of the real artifact (the gfx1100 runtime predicate
    /// holds there), so the PREFERRED ten layers no longer fit beside the S1
    /// terms, the 1 GiB runtime budget and the 512 MiB safety margin: the
    /// admission keeps the largest count that fits (nine) and reports the
    /// exact preferred deficit as information. The
    /// incumbent's reserve policy (`max(historical floor, census)`) is
    /// unchanged.
    #[test]
    fn slowlink_residency_real_fixture_1536_counterfactual_admits_nine_with_the_preferred_deficit()
    {
        let reserve = j1778_reserve_at(S1_TILE_ROWS);
        assert_eq!(reserve.census.f16_shadows, J1778_SHADOWS);
        assert_eq!(reserve.census_total, 2_237_119_460 + J1778_SHADOWS);
        assert_eq!(reserve.placement_bytes(true), 3_536_042_980);
        // Incumbent policy untouched: `max(historical floor, census)` for every
        // load that is not an admitted S1 profile; S1 reads the census alone.
        let historical = reserve
            .historical
            .expect("a discrete gfx1100 keeps its floor");
        assert_eq!(reserve.bytes(), historical.max(reserve.census_total));
        assert_eq!(reserve.placement_bytes(false), reserve.bytes());
        assert!(
            historical > reserve.census_total,
            "the stale floor still wins"
        );
        assert_eq!(reserve.winner(), "historical floor");

        let inputs = j1778_inputs_at(
            WindowRoute::SlabIu4,
            AdmissionPurpose::Product,
            J1778_PRELOAD_FREE,
            S1_TILE_ROWS,
        );
        assert_eq!(GATE_UP_LAYER + DOWN_LAYER, 1_336_934_400);
        let (_, _, extras) = census_at_tile(&inputs, S1_TILE_ROWS).unwrap();
        assert_eq!(extras.owned_total().unwrap(), 2_543_094_592);
        assert_eq!(extras.runtime_overhead, S1_RUNTIME_OVERHEAD_BYTES);
        assert_eq!(
            crate::expert_residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
            J1778_MARGIN
        );
        // Census base + S1 owned + 1 GiB runtime + 512 MiB safety, on top.
        let charged = 3_536_042_980 + 2_543_094_592 + S1_RUNTIME_OVERHEAD_BYTES + J1778_MARGIN;
        assert_eq!(charged, 7_689_750_308);
        let nine = admit_at_tile(&inputs, S1_TILE_ROWS)
            .expect("nine layers fit: the preferred ten are information, not a gate");
        assert_eq!(nine.resident_layers, 9);
        assert_eq!(nine.placement.reserve_bytes, charged);
        assert_eq!(
            nine.placement.preferred_deficit_bytes(),
            10 * LAYER + J1778_NON_EXPERT + charged - J1778_PRELOAD_FREE
        );
        assert_eq!(nine.placement.preferred_deficit_bytes(), 709_968_932);
        assert!(nine.placement.fits(9) && !nine.placement.fits(10));
        // The retired source omission, as a counterfactual: without the shadow
        // cold bound ten layers fit, so the shadows alone cost the tenth layer.
        let omitted = AdmissionInputs {
            base_reserve: 2_237_119_460,
            ..inputs
        };
        assert_eq!(
            admit_at_tile(&omitted, S1_TILE_ROWS)
                .unwrap()
                .resident_layers,
            10
        );
        // The stale historical floor as the base is a deeper (seven layer) fit.
        let floor = AdmissionInputs {
            base_reserve: J1778_HISTORICAL_1536,
            ..inputs
        };
        assert_eq!(
            admit_at_tile(&floor, S1_TILE_ROWS).unwrap().resident_layers,
            7
        );
        // The old conservative stress input (24 GiB less a fixed 1.5 GiB
        // driver allowance) keeps fewer layers still.
        let conservative = AdmissionInputs {
            memory: AdmissionMemory::actual(24 * GIB - 3 * GIB / 2),
            ..inputs
        };
        assert_eq!(
            admit_at_tile(&conservative, S1_TILE_ROWS)
                .unwrap()
                .resident_layers,
            8
        );
        // A tile outside the legal rungs is never admitted.
        assert!(matches!(
            admit_at_tile(&inputs, 256),
            Err(SlowlinkDecline::Geometry(_))
        ));
    }

    /// The primitive self-test budget is the authoritative transport/dispatch
    /// figure on both routes: the dispatch suite over the staged 32-expert
    /// binding (182,065,968 B on the slab route) is below the transport's
    /// 217,821,184 B reserve, which is the larger; the product charges none.
    #[test]
    fn slowlink_residency_diagnostic_budget_is_the_transport_reserve_on_both_routes() {
        use hipfire_dispatch::pipeline::slowlink_prefill::selftest::{
            suite_diagnostic_bytes, DiagGeometry, DEFAULT_SUITE,
        };
        use hipfire_dispatch::pipeline::slowlink_prefill::PanelLimits;
        let config = crate::config::compact_test_config();
        let slab = suite_diagnostic_bytes(
            &DEFAULT_SUITE,
            &DiagGeometry {
                hidden: 2560,
                n_exp: 512,
                mi: 640,
                binding_experts: 32,
                k1_rows: 3,
                shared_intermediate: 640,
                route: WindowRoute::SlabIu4,
            },
            PanelLimits::S1,
        )
        .unwrap();
        assert_eq!(slab.device_total(), 182_065_968);
        for route in [WindowRoute::SlabIu4, WindowRoute::Aggregate] {
            let geometry = ModelGeometry::from_config(&config, route).unwrap();
            let budget = |purpose| {
                diagnostic_budget_bytes(&geometry, purpose, GATE_UP_LAYER, DOWN_LAYER).unwrap()
            };
            assert_eq!(
                budget(AdmissionPurpose::PrimitiveSelftest),
                J1778_PRIMITIVE_DIAGNOSTIC,
                "{route:?}: the transport reserve is the larger"
            );
            assert_eq!(budget(AdmissionPurpose::Product), 0, "{route:?}");
        }
    }

    /// The 1,536-row COUNTERFACTUAL on both routes and purposes: every purpose
    /// keeps the largest count that fits (nine) at the historical maximum
    /// tile and reports the exact preferred deficit (the aggregate N=10 slack
    /// is its negation) as information. The 512 MiB safety
    /// margin and the 1 GiB runtime budget stay visible beside the corrected
    /// census, and a tenth layer's shortfall is exactly the preferred deficit
    /// (never a shrunk floor). Admitted through [`admit_at_tile`]; production
    /// never admits it.
    #[test]
    fn slowlink_residency_j1778_1536_counterfactual_keeps_nine_with_exact_preferred_deficits() {
        use hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout;
        for route in [WindowRoute::SlabIu4, WindowRoute::Aggregate] {
            let mut previous = None;
            for (label, purpose, diagnostic) in [
                ("product", AdmissionPurpose::Product, 0u64),
                (
                    "primitive selftest",
                    AdmissionPurpose::PrimitiveSelftest,
                    J1778_PRIMITIVE_DIAGNOSTIC,
                ),
            ] {
                let inputs = j1778_inputs_at(route, purpose, J1778_PRELOAD_FREE, S1_TILE_ROWS);
                assert_eq!(inputs.diagnostic_bytes, diagnostic, "{label}");
                let (pages, _, census) = census_at_tile(&inputs, S1_TILE_ROWS).unwrap();
                assert_eq!(pages.total_pages(), 18, "{label}");
                assert_eq!(
                    census.runtime_overhead, S1_RUNTIME_OVERHEAD_BYTES,
                    "{label}"
                );
                assert_eq!(census.diagnostic, diagnostic, "{label}");
                let charged = inputs.base_reserve
                    + census.total().unwrap()
                    + crate::expert_residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES;
                let profile = admit_at_tile(&inputs, S1_TILE_ROWS)
                    .unwrap_or_else(|decline| panic!("{route:?} {label}: {decline}"));
                assert_eq!(profile.resident_layers, 9, "{route:?} {label}");
                assert_eq!(
                    profile.placement.reserve_bytes, charged,
                    "{route:?} {label}"
                );
                let tenth = 10 * LAYER + J1778_NON_EXPERT + charged;
                let deficit = profile.placement.preferred_deficit_bytes();
                assert_eq!(deficit, tenth - J1778_PRELOAD_FREE, "{route:?} {label}");
                // Nine layers fit beside the whole charged reserve
                // (safety margin included, on top); ten do not.
                let nine = 9 * LAYER + J1778_NON_EXPERT + charged;
                assert!(nine <= J1778_PRELOAD_FREE, "{route:?} {label}");
                assert!(profile.placement.fits(9) && !profile.placement.fits(10));
                if let Some(previous) = previous {
                    assert!(deficit > previous, "the diagnostic is funded");
                }
                previous = Some(deficit);
                if route == WindowRoute::Aggregate {
                    assert_eq!(census.a4_sidecars, 0, "{label}");
                    let layout =
                        PromptRouteLayout::new(&route_dims(&inputs.geometry, inputs.window_rows))
                            .unwrap();
                    assert_eq!(layout.route(), WindowRoute::Aggregate);
                    assert_eq!(census.route_storage, layout.total_bytes() as u64);
                    assert_eq!(
                        census.route_storage + census.a4_sidecars + census.group_scatter,
                        layout.charged_bytes() as u64
                    );
                }
            }
        }
    }

    /// The purpose-derived tile: product 768, primitive 512, both legal rungs
    /// below the 1,536 maximum, each a value the existing explicit chunk policy
    /// accepts.
    #[test]
    fn slowlink_admission_purpose_tiles_are_legal_rungs_below_the_maximum() {
        assert_eq!(purpose_tile_rows(AdmissionPurpose::Product), 768);
        assert_eq!(purpose_tile_rows(AdmissionPurpose::PrimitiveSelftest), 512);
        assert_eq!(S1_PRODUCT_TILE_ROWS, 768);
        assert_eq!(S1_PRIMITIVE_TILE_ROWS, 512);
        assert!(S1_PRODUCT_TILE_ROWS < S1_TILE_ROWS);
        let rungs = legal_tile_rungs();
        assert!(rungs.contains(&S1_PRODUCT_TILE_ROWS));
        assert!(rungs.contains(&S1_PRIMITIVE_TILE_ROWS));
        assert_eq!(rungs.first(), Some(&S1_TILE_ROWS));
        // The short-request incumbent bypass keeps its 1,536-row bound (the
        // maximum tile), not the purpose's tile.
        assert_eq!(S1_MIN_FORCED_REQUEST_ROWS, S1_TILE_ROWS + 1);
    }

    /// The ACTUAL chosen profiles on the real j1778 fixture, both routes:
    /// product 768 and primitive 512 each admit exactly ten resident layers
    /// with the 16K window and 18 pages, 1 GiB runtime and 512 MiB safety on
    /// top, with the frontier's exact N=10 slack; one byte less free admits
    /// nine (a one byte preferred deficit), and an eleventh layer never fits.
    #[test]
    fn slowlink_residency_j1778_chosen_profiles_admit_ten_layers_with_exact_slack() {
        let cases = [
            (
                WindowRoute::Aggregate,
                AdmissionPurpose::Product,
                S1_PRODUCT_TILE_ROWS,
                J1778_AGG_PRODUCT_SLACK[3],
            ),
            (
                WindowRoute::Aggregate,
                AdmissionPurpose::PrimitiveSelftest,
                S1_PRIMITIVE_TILE_ROWS,
                J1778_AGG_PRIMITIVE_SLACK[4],
            ),
            (
                WindowRoute::SlabIu4,
                AdmissionPurpose::Product,
                S1_PRODUCT_TILE_ROWS,
                J1778_SLAB_PRODUCT_SLACK[3],
            ),
            (
                WindowRoute::SlabIu4,
                AdmissionPurpose::PrimitiveSelftest,
                S1_PRIMITIVE_TILE_ROWS,
                J1778_SLAB_PRIMITIVE_SLACK[4],
            ),
        ];
        // The two actual real-artifact (aggregate) slacks Main chose on.
        assert_eq!(J1778_AGG_PRODUCT_SLACK[3], 156_559_964);
        assert_eq!(J1778_AGG_PRIMITIVE_SLACK[4], 228_096_476);
        for (route, purpose, tile, slack) in cases {
            let label = format!("{route:?} {purpose:?}");
            let inputs = j1778_inputs(route, purpose, J1778_PRELOAD_FREE);
            assert_eq!(inputs.purpose, purpose, "{label}");
            let profile = admit(&inputs).unwrap_or_else(|decline| panic!("{label}: {decline}"));
            assert_eq!(profile.purpose, purpose, "{label}");
            assert_eq!(profile.tile_rows, tile, "{label}");
            assert_eq!(profile.tile_rows, purpose_tile_rows(purpose), "{label}");
            assert_eq!(profile.window_rows, REFERENCE_WINDOW_ROWS, "{label}");
            assert_eq!(profile.resident_layers, 10, "{label}");
            assert_eq!(profile.pages.total_pages(), 18, "{label}");
            assert_eq!(profile.arena.tile_rows, tile, "{label}");
            assert_eq!(profile.census.runtime_overhead, S1_RUNTIME_OVERHEAD_BYTES);
            assert_eq!(
                profile.census.diagnostic,
                if purpose == AdmissionPurpose::Product {
                    0
                } else {
                    J1778_PRIMITIVE_DIAGNOSTIC
                },
                "{label}"
            );
            assert_eq!(j1778_slack(&inputs, &profile, 10), slack, "{label}");
            assert!(j1778_slack(&inputs, &profile, 11) < 0, "{label}");
            // The safety margin sits on top: the slack is what remains AFTER it.
            assert!(slack > 0, "{label}");
            // Exact boundary: the slack in bytes less actual free still admits
            // ten layers; one byte fewer admits nine and reports the one byte
            // preferred deficit (information, not a refusal).
            let slack_bytes = u64::try_from(slack).unwrap();
            let edge = AdmissionInputs {
                memory: AdmissionMemory::actual(J1778_PRELOAD_FREE - slack_bytes),
                ..inputs
            };
            let at_edge = admit(&edge).unwrap();
            assert_eq!(at_edge.resident_layers, 10, "{label}");
            assert_eq!(at_edge.device_snapshot_budget_bytes, 0, "{label}");
            assert_eq!(at_edge.placement.preferred_deficit_bytes(), 0, "{label}");
            let below = AdmissionInputs {
                memory: AdmissionMemory::actual(edge.memory.free_bytes - 1),
                ..edge
            };
            let nine = admit(&below).unwrap_or_else(|decline| panic!("{label}: {decline}"));
            assert_eq!(nine.resident_layers, 9, "{label}");
            assert_eq!(nine.placement.preferred_deficit_bytes(), 1, "{label}");
        }
    }

    /// A product load never silently falls to the finer primitive tile (and a
    /// primitive load never takes the product tile): with less free memory
    /// than 768 needs for ten layers but 512 would keep ten, `admit` keeps
    /// 768 with nine layers and the exact preferred deficit while the
    /// frontier only REPORTS 512 as would-fit; and the primitive purpose at
    /// 768 keeps nine with its exact preferred deficit.
    #[test]
    fn slowlink_residency_j1778_purposes_never_fall_to_another_purposes_tile() {
        let free = J1778_PRELOAD_FREE - 200_000_000;
        let product = j1778_inputs(WindowRoute::Aggregate, AdmissionPurpose::Product, free);
        let kept = admit(&product).expect("nine layers fit: the preferred ten are information");
        assert_eq!(kept.tile_rows, S1_PRODUCT_TILE_ROWS, "never the finer tile");
        assert_eq!(kept.resident_layers, 9);
        assert_eq!(
            kept.placement.preferred_deficit_bytes(),
            200_000_000 - 156_559_964
        );
        let points = tile_frontier(&product, |tile| {
            Ok((
                j1778_reserve_at(tile),
                crate::expert_residency::GranuleSlack::default(),
            ))
        })
        .unwrap();
        let selected: Vec<_> = points.iter().filter(|point| point.selected).collect();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].tile_rows, S1_PRODUCT_TILE_ROWS);
        assert!(selected[0].admits());
        assert_eq!(selected[0].resident_layers, 9);
        let finer = points
            .iter()
            .find(|point| point.tile_rows == S1_PRIMITIVE_TILE_ROWS)
            .unwrap();
        assert!(finer.admits(), "512 would fit: report only");
        assert_eq!(
            finer.resident_layers, 10,
            "512 would keep the preferred ten"
        );
        assert!(!finer.selected);

        // The primitive purpose at the product's 768 tile: the diagnostic
        // leaves nine layers, so only 512 keeps ten.
        let primitive = j1778_inputs_at(
            WindowRoute::Aggregate,
            AdmissionPurpose::PrimitiveSelftest,
            J1778_PRELOAD_FREE,
            S1_PRODUCT_TILE_ROWS,
        );
        let kept = admit_at_tile(&primitive, 768).expect("nine layers fit at 768");
        assert_eq!(kept.resident_layers, 9);
        assert_eq!(kept.placement.preferred_deficit_bytes(), 61_261_220);
    }

    fn frontier_json(points: &[FrontierPoint]) -> serde_json::Value {
        let signed = |value: i128| i64::try_from(value).expect("a byte figure fits i64");
        serde_json::Value::Array(
            points
                .iter()
                .map(|point| {
                    let census = &point.census;
                    serde_json::json!({
                        "tile_rows": point.tile_rows,
                        "tiles_per_window": point.tiles_per_window,
                        "final_tile_rows": point.final_tile_rows,
                        "resident_layers": point.resident_layers,
                        "admits": point.admits(),
                        "preferred_fits": point.budget.fits(S1_PREFERRED_RESIDENT_LAYERS),
                        "selected_by_purpose": point.selected,
                        "decline": point.verdict.as_ref().err().map(|decline| decline.to_string()),
                        "dominated": point.dominated,
                        "pre_init_free_bytes": point.budget.memory.pre_init_free_bytes,
                        "free_bytes": point.budget.memory.free_bytes,
                        "non_expert_bytes": point.budget.non_expert_bytes,
                        "resident_layer_bytes": point.budget.resident_layer_bytes,
                        "layer_expert_bytes": point.layer_expert_bytes,
                        "base_reserve_bytes": point.base_reserve,
                        "f16_shadow_cold_bound_bytes_in_base": point.f16_shadow_bytes,
                        "s1_owned_terms": census
                            .owned_terms()
                            .iter()
                            .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                            .collect::<Vec<_>>(),
                        "s1_owned_total_bytes": census.owned_total().unwrap(),
                        "runtime_budget_bytes": census.runtime_overhead,
                        "diagnostic_budget_bytes": census.diagnostic,
                        "safety_margin_bytes": point.safety_margin,
                        "reserve_bytes": point.reserve,
                        "needed_bytes_at_resident": signed(
                            point.needed_bytes_for(point.resident_layers) as i128
                        ),
                        "slack_bytes_at_resident": signed(point.slack_for(point.resident_layers)),
                        "needed_bytes_at_ten": signed(point.needed_bytes_for(10) as i128),
                        "slack_bytes_at_ten": signed(point.slack_for(10)),
                        "slack_bytes_by_resident_layers": (0..=S1_PREFERRED_RESIDENT_LAYERS)
                            .map(|layers| signed(point.slack_for(layers)))
                            .collect::<Vec<_>>(),
                    })
                })
                .collect(),
        )
    }

    /// The feasible (resident, tile) frontier of the REAL j1778 fixture on the
    /// aggregate route (the real artifact) and the slab route, for the product
    /// and the primitive-purpose profiles: every legal tile rung from the
    /// existing 256-row chunk granularity, each with exact physical capacity,
    /// cold F16 shadow bound, S1 owned terms, the runtime budget, the
    /// diagnostic and the safety margin separately. The purpose's tile is
    /// `selected`; every other point is a would-fit report only. Emits one JSON
    /// line (`--nocapture`); memory only, no rate model, nothing lowered.
    #[test]
    fn slowlink_residency_j1778_feasible_tile_frontier_for_both_purposes() {
        // Legal rungs: the existing chunk granularity, never below the WMMA
        // threshold. A 256-row tile would make the 16,384-row window 64 F32
        // tail tiles, far past the 3,066-row tail bound.
        assert_eq!(legal_tile_rungs(), vec![1536, 1280, 1024, 768, 512]);
        assert_eq!(
            S1_MIN_TILE_ROWS,
            rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        );
        assert!(S1_WINDOW_ROWS / 256 * 255 > S1_AGGREGATE_TAIL_ROWS);
        assert!(legal_tile_rungs()
            .iter()
            .all(|&tile| tile > S1_F32_TILE_ROWS_MAX));

        let mut routes = Vec::new();
        for route in [WindowRoute::Aggregate, WindowRoute::SlabIu4] {
            let mut purposes = Vec::new();
            let mut product_layers = Vec::new();
            for (label, purpose, diagnostic) in [
                ("product", AdmissionPurpose::Product, 0u64),
                (
                    "primitive_selftest",
                    AdmissionPurpose::PrimitiveSelftest,
                    J1778_PRIMITIVE_DIAGNOSTIC,
                ),
            ] {
                let inputs = j1778_inputs(route, purpose, J1778_PRELOAD_FREE);
                let points = tile_frontier(&inputs, |tile| {
                    Ok((
                        j1778_reserve_at(tile),
                        crate::expert_residency::GranuleSlack::default(),
                    ))
                })
                .unwrap();
                assert_eq!(points.len(), 5);
                // Exactly the purpose's tile is selected and its verdict is
                // exactly the runtime's.
                let chosen = points.iter().filter(|point| point.selected).count();
                assert_eq!(chosen, 1, "{route:?} {label}");
                let selected = points.iter().find(|point| point.selected).unwrap();
                assert_eq!(selected.tile_rows, purpose_tile_rows(purpose));
                assert_eq!(selected.verdict, admit(&inputs).map(|_| ()));
                assert!(selected.admits(), "{route:?} {label}");
                let (slack, max_layers) = match (route, purpose) {
                    (WindowRoute::AggregateSymIu4, AdmissionPurpose::Product) => {
                        (J1778_SYM_PRODUCT_SLACK, [9, 9, 9, 10, 10])
                    }
                    (WindowRoute::AggregateSymIu4, _) => {
                        (J1778_SYM_PRIMITIVE_SLACK, [9, 9, 9, 9, 10])
                    }
                    (WindowRoute::Aggregate, AdmissionPurpose::Product) => {
                        (J1778_AGG_PRODUCT_SLACK, [9, 9, 9, 10, 10])
                    }
                    (WindowRoute::Aggregate, _) => (J1778_AGG_PRIMITIVE_SLACK, [9, 9, 9, 9, 10]),
                    (_, AdmissionPurpose::Product) => (J1778_SLAB_PRODUCT_SLACK, [9, 9, 9, 10, 10]),
                    (_, _) => (J1778_SLAB_PRIMITIVE_SLACK, [9, 9, 9, 9, 10]),
                };
                let mut last_layers = 0;
                for (index, point) in points.iter().enumerate() {
                    let census = &point.census;
                    // Every rung charges the cold shadow bound, once, inside the base.
                    assert_eq!(point.f16_shadow_bytes, J1778_SHADOWS);
                    assert_eq!(census.runtime_overhead, S1_RUNTIME_OVERHEAD_BYTES);
                    assert_eq!(census.diagnostic, diagnostic);
                    assert_eq!(point.safety_margin, J1778_MARGIN);
                    assert_eq!(
                        point.reserve,
                        point.base_reserve + census.total().unwrap() + J1778_MARGIN
                    );
                    assert_eq!(point.budget.memory.free_bytes, J1778_PRELOAD_FREE);
                    assert!(point.budget.memory.free_bytes < J1778_TOTAL);
                    let at = format!("{route:?} {label} tile {}", point.tile_rows);
                    assert_eq!(point.base_reserve, J1778_RUNG_BASE[index], "{at}");
                    assert_eq!(point.slack_for(10), slack[index], "{at}");
                    assert_eq!(point.resident_layers, max_layers[index], "{at}");
                    // Smaller tiles never keep fewer layers.
                    assert!(point.resident_layers >= last_layers);
                    last_layers = point.resident_layers;
                    // The point fits its own layers and not one more.
                    assert!(point.slack_for(point.resident_layers) >= 0);
                    assert!(point.slack_for(point.resident_layers + 1) < 0);
                    assert_eq!(
                        point.admits(),
                        point.budget.fits(0) && point.resident_layers < point.budget.trunk_layers
                    );
                    // Full window cohorts: whole WMMA tiles, and at most the one
                    // window-final F32 tail tile inside the 3,066-row bound.
                    assert!(point.tile_rows >= S1_MIN_TILE_ROWS);
                    assert_eq!(
                        point.tiles_per_window,
                        REFERENCE_WINDOW_ROWS.div_ceil(point.tile_rows)
                    );
                    assert_eq!(
                        point.final_tile_rows + (point.tiles_per_window - 1) * point.tile_rows,
                        REFERENCE_WINDOW_ROWS
                    );
                    if point.final_tile_rows < S1_MIN_TILE_ROWS {
                        assert!(point.final_tile_rows <= S1_F32_TILE_ROWS_MAX);
                    }
                }
                let finals: Vec<usize> = points.iter().map(|point| point.final_tile_rows).collect();
                assert_eq!(finals, vec![1024, 1024, 1024, 256, 512]);
                if label == "product" {
                    product_layers = points.iter().map(|point| point.resident_layers).collect();
                } else {
                    // A funded diagnostic never keeps more layers than the product.
                    for (point, product) in points.iter().zip(&product_layers) {
                        assert!(point.resident_layers <= *product);
                    }
                }
                let feasible: Vec<usize> = points
                    .iter()
                    .filter(|point| point.admits() && !point.dominated)
                    .map(|point| point.tile_rows)
                    .collect();
                // Adaptive admission has no layer floor: every rung keeps the
                // largest count that fits. The undominated rungs are the
                // 1,536-row maximum (nine layers) and the purpose's own tile
                // (the preferred ten); Main's tile choice is unchanged.
                assert_eq!(
                    feasible,
                    vec![S1_TILE_ROWS, purpose_tile_rows(purpose)],
                    "{route:?} {label}"
                );
                purposes.push(serde_json::json!({
                    "purpose": format!("{purpose:?}"),
                    "purpose_tile_rows": purpose_tile_rows(purpose),
                    "diagnostic_budget_bytes": diagnostic,
                    "feasible_frontier_tiles": feasible,
                    "points": frontier_json(&points),
                }));
            }
            routes.push(serde_json::json!({
                "route": format!("{route:?}"),
                "purposes": purposes,
            }));
        }
        let report = serde_json::json!({
            "fixture": {
                "config": "compact_test_config",
                "arch": "gfx1100",
                "state_format": { "qsa": "q8", "gdn": "q8" },
                "max_seq": J1778_MAX_SEQ,
                "window_rows": S1_WINDOW_ROWS,
                "pages": 18,
                "preferred_resident_layers": S1_PREFERRED_RESIDENT_LAYERS,
                "hip_total_bytes": J1778_TOTAL,
                "hip_preload_free_bytes": J1778_PRELOAD_FREE,
                "hip_preload_used_bytes": J1778_TOTAL - J1778_PRELOAD_FREE,
                "non_expert_bytes": J1778_NON_EXPERT,
                "layer_expert_bytes": LAYER,
                "f16_shadow_cold_bound_bytes": J1778_SHADOWS,
                "f16_shadow_owners": 291,
                "runtime_budget_bytes": S1_RUNTIME_OVERHEAD_BYTES,
                "safety_margin_bytes": J1778_MARGIN,
                "legal_tile_rungs": legal_tile_rungs(),
                "product_tile_rows": S1_PRODUCT_TILE_ROWS,
                "primitive_tile_rows": S1_PRIMITIVE_TILE_ROWS,
                "max_tile_rows": S1_TILE_ROWS,
                "note": "CPU census of declared inputs, not a GPU measurement; memory only",
            },
            "routes": routes,
        });
        println!(
            "SLOWLINK_J1778_TILE_FRONTIER {}",
            serde_json::to_string(&report).unwrap()
        );
    }

    /// Emits the ACTUAL chosen profiles of the real j1778 fixture (purpose
    /// `Product` at 768 and `PrimitiveSelftest` at 512, both routes), each with
    /// its full base-census breakdown (forward, state, QSA context, workspace,
    /// F16 weight shadow), the S1 owned terms and arena, the diagnostic,
    /// runtime and safety budgets and the N=10 needed/free/slack, as one JSON
    /// line (`SLOWLINK_J1778_CHOSEN_PROFILES`, `--nocapture`). Every figure is
    /// read from the real getters ([`admit`], the census), not copied.
    #[test]
    fn slowlink_residency_j1778_emits_the_chosen_profiles_with_the_full_base_breakdown() {
        let mut profiles = Vec::new();
        for route in [WindowRoute::Aggregate, WindowRoute::SlabIu4] {
            for purpose in [
                AdmissionPurpose::Product,
                AdmissionPurpose::PrimitiveSelftest,
            ] {
                let inputs = j1778_inputs(route, purpose, J1778_PRELOAD_FREE);
                let profile = admit(&inputs).expect("the chosen profile admits");
                assert_eq!(profile.purpose, purpose);
                let tile = profile.tile_rows;
                let reserve = j1778_reserve_at(tile);
                let census = &profile.census;
                assert_eq!(reserve.placement_bytes(true), inputs.base_reserve);
                let workspace_total: u64 = reserve.census.workspace.iter().map(|t| t.1).sum();
                let needed_ten = i128::from(inputs.non_expert)
                    + 10 * i128::from(LAYER)
                    + i128::from(inputs.base_reserve)
                    + i128::from(census.total().unwrap())
                    + i128::from(J1778_MARGIN);
                let terms = |terms: &[(&str, u64)]| {
                    terms
                        .iter()
                        .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                        .collect::<Vec<_>>()
                };
                profiles.push(serde_json::json!({
                    "route": format!("{route:?}"),
                    "purpose": format!("{purpose:?}"),
                    "tile_rows": tile,
                    "window_rows": profile.window_rows,
                    "page_count": profile.pages.total_pages(),
                    "resident_layers": profile.resident_layers,
                    "base_census": {
                        "forward_bytes": reserve.census.forward,
                        "state_bytes": reserve.census.state,
                        "qsa_context_bytes": reserve.census.qsa_context,
                        "workspace_total_bytes": workspace_total,
                        "workspace_terms": terms(&reserve.census.workspace),
                        "f16_weight_shadow_bytes": reserve.census.f16_shadows,
                        "qsa_selector_bytes": reserve.census.qsa_selector,
                        "mtp_bytes": reserve.census.mtp,
                        "gather_bytes": reserve.census.gather,
                        "runtime_bytes": reserve.census.runtime,
                        "census_total_bytes": reserve.census_total,
                        "s1_placement_base_bytes": reserve.placement_bytes(true),
                        "incumbent_historical_floor_bytes": reserve.historical,
                    },
                    "s1": {
                        "owned_terms": terms(&census.owned_terms()),
                        "owned_total_bytes": census.owned_total().unwrap(),
                        "hc_arena_bytes": profile.arena.total_bytes,
                        "hc_arena_tile_rows": profile.arena.tile_rows,
                        "diagnostic_budget_bytes": census.diagnostic,
                        "runtime_budget_bytes": census.runtime_overhead,
                        "s1_reserve_bytes": census.total().unwrap(),
                    },
                    "safety_margin_bytes": J1778_MARGIN,
                    "projection": {
                        "hip_total_bytes": J1778_TOTAL,
                        "free_bytes": inputs.memory.free_bytes,
                        "non_expert_bytes": inputs.non_expert,
                        "resident_expert_bytes": 10 * LAYER,
                        // Physical owners only (no runtime/diagnostic budget, no
                        // safety): weights, then + the base census and S1 owned
                        // terms (core), then + this purpose's sequential diagnostic.
                        "device_weights_bytes": inputs.non_expert + 10 * LAYER,
                        "device_core_bytes": inputs.non_expert
                            + 10 * LAYER
                            + reserve.census_total
                            + census.owned_total().unwrap(),
                        "device_owned_bytes": inputs.non_expert
                            + 10 * LAYER
                            + reserve.census_total
                            + census.owned_total().unwrap()
                            + census.diagnostic,
                        "needed_at_ten_bytes": i64::try_from(needed_ten).unwrap(),
                        "slack_at_ten_bytes": i64::try_from(j1778_slack(&inputs, &profile, 10)).unwrap(),
                    },
                }));
            }
        }
        println!(
            "SLOWLINK_J1778_CHOSEN_PROFILES {}",
            serde_json::to_string(&serde_json::json!({
                "fixture": "j1778 real-artifact inputs (CPU census, not a GPU measurement)",
                "max_seq": J1778_MAX_SEQ,
                "profiles": profiles,
            }))
            .unwrap()
        );
    }

    /// The generic runtime budget supplements what the base reserve already
    /// charges and never doubles it: the combined runtime reservation (the
    /// base census's `runtime` plus the S1 `runtime_overhead` term) is exactly
    /// `max(base, S1_RUNTIME_OVERHEAD_BYTES)` for a base of 0, below, at and
    /// above 1 GiB, with the base reserve actually carrying the credited bytes
    /// (`AutoReserve.census.runtime`), so the same XTX input (base 0) still
    /// gets the full 1 GiB and a capable path never pays 1 GiB twice. No arch,
    /// card or host name is consulted.
    /// SYNTHETIC census terms (base 0 plus a runtime credit), not the real
    /// artifact; the real fixture's base lives in `j1778_reserve_at`.
    #[test]
    fn slowlink_residency_runtime_budget_supplements_the_base_runtime_without_double_counting() {
        use crate::expert_residency::{AutoReserve, ReserveCensus, WORKSPACE_TERM_COUNT};
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        // The measured capable-path HIP runtime charge.
        const CAPABLE_RUNTIME: u64 = 1_476_395_008;
        let non_runtime = 2_237_111_724u64;
        let config = crate::config::compact_test_config();
        for base_runtime in [
            0,
            512 * MIB,
            GIB - 1,
            GIB,
            GIB + 1,
            CAPABLE_RUNTIME,
            2 * GIB,
        ] {
            // The base reserve really carries the credited runtime bytes.
            let mut workspace = [("workspace", 0u64); WORKSPACE_TERM_COUNT];
            workspace[0].1 = non_runtime;
            let reserve = AutoReserve::new(
                None,
                ReserveCensus {
                    qsa_context: 0,
                    state: 0,
                    forward: 0,
                    f16_shadows: 0,
                    shared_i8_shadows: 0,
                    workspace,
                    qsa_selector: 0,
                    mtp: 0,
                    gather: 0,
                    runtime: base_runtime,
                },
            )
            .unwrap();
            assert_eq!(reserve.census_total, non_runtime + base_runtime);
            let inputs = AdmissionInputs {
                geometry: ModelGeometry::from_config(&config, WindowRoute::SlabIu4).unwrap(),
                memory: AdmissionMemory::actual(40 * GIB),
                non_expert: 5_364_055_296,
                layer_expert_bytes: GATE_UP_LAYER + DOWN_LAYER,
                gate_up_layer_bytes: GATE_UP_LAYER,
                down_layer_bytes: DOWN_LAYER,
                trunk_layers: config.num_hidden_layers,
                base_reserve: reserve.placement_bytes(true),
                baseline_layers: 0,
                fixed_state_bytes: 34_128_016,
                pending_save_bytes: 0,
                diagnostic_bytes: 0,
                base_runtime_reserved_bytes: reserve.census.runtime,
                granule: GranuleInputs::default(),
                checkpoint_stride_rows: 8192,
                window_rows: REFERENCE_WINDOW_ROWS,
                link_gbps: 7.09,
                purpose: AdmissionPurpose::Product,
                sym_iu4_source_symmetric: None,
            };
            let (_, _, census) = census_of(&inputs).unwrap();
            // The supplement is exactly what the base lacks.
            assert_eq!(census.runtime_overhead, GIB.saturating_sub(base_runtime));
            // Base census runtime + supplement == max(base, 1 GiB): no double count.
            assert_eq!(
                inputs.base_reserve - non_runtime + census.runtime_overhead,
                base_runtime.max(S1_RUNTIME_OVERHEAD_BYTES),
                "base runtime {base_runtime}"
            );
            // The census budget terms stay itemised.
            assert_eq!(census.budget_terms()[0].1, census.runtime_overhead);
        }
    }

    /// Every non-S1 load (the incumbent, fast links, unified memory, every
    /// opt-out: `slowlink == false`) resolves the byte-identical reserve it
    /// always did, whether the historical floor wins, the census wins, or the
    /// floor is retired; only an admitted S1 profile reads the census alone.
    /// SYNTHETIC census (zero terms, not the real artifact): the policy is
    /// arithmetic over the census it is given.
    #[test]
    fn slowlink_residency_non_s1_loads_keep_the_incumbent_reserve() {
        use crate::expert_residency::{AutoReserve, ReserveCensus, WORKSPACE_TERM_COUNT};
        let census = |bytes: u64| {
            let mut workspace = [("workspace", 0u64); WORKSPACE_TERM_COUNT];
            workspace[0].1 = bytes;
            ReserveCensus {
                qsa_context: 0,
                state: 0,
                forward: 0,
                f16_shadows: 0,
                shared_i8_shadows: 0,
                workspace,
                qsa_selector: 0,
                mtp: 0,
                gather: 0,
                runtime: 0,
            }
        };
        for (historical, census_bytes) in [
            (Some(5_517_207_808u64), 2_237_111_724u64), // floor wins
            (Some(1 << 30), 3 << 30),                   // census wins
            (None, 2_237_111_724),                      // floor retired (gfx1201)
        ] {
            let reserve = AutoReserve::new(historical, census(census_bytes)).unwrap();
            assert_eq!(reserve.placement_bytes(false), reserve.bytes());
            assert_eq!(reserve.placement_bytes(true), census_bytes);
        }
        // Fast, unified and unmeasured links, a killed selector and an NPU
        // never reach the S1 reserve: the facts (free-memory query and both
        // reserves) are not even built, so no budget is read.
        for link in [
            HostLink::Discrete { gbps: 16.0 },
            HostLink::Discrete { gbps: 26.0 },
            HostLink::Unified,
            HostLink::Unmeasured,
        ] {
            let outcome = select_load_profile(
                SlowlinkMode::Force,
                &shape(),
                || link,
                || -> Result<LoadFacts<'static>, String> {
                    unreachable!("facts built for {link:?}")
                },
            );
            assert_eq!(outcome, Ok(None), "{link:?}");
        }
        for (mode, load) in [
            (SlowlinkMode::Off, shape()),
            (
                SlowlinkMode::Force,
                LoadShape {
                    npu_trunk: true,
                    ..shape()
                },
            ),
        ] {
            let outcome = select_load_profile(
                mode,
                &load,
                || unreachable!("probed"),
                || -> Result<LoadFacts<'static>, String> { unreachable!("facts built") },
            );
            assert_eq!(outcome, Ok(None));
        }
    }

    /// Hold 1 residency ledger on the REAL compact Flash-Next config (the
    /// persisted HFQ geometry): Full scratch, gfx1100 (no compact plan), a
    /// 16,640-token context, the 1,536-row tile, Q8 GDN and Q8 QSA, no MTP.
    /// Calls the actual census helpers (no reimplemented formula) and prints
    /// one structured JSON ledger (`--nocapture`). The legacy context is the
    /// discrete card's charge; the VMM 2 MiB row is conditional on the backend
    /// the load actually observes.
    /// The `tile` section carries the corrected base census with the F16
    /// weight shadow cold bound beside it; the S1 terms stay the layout's.
    #[test]
    fn slowlink_residency_hold1_ledger_for_the_real_mq4_fixture() {
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        use rdna_compute::tensor_ops::QsaKvFormat;
        let config = crate::config::compact_test_config();
        let format = crate::Qwen4StateFormat {
            qsa: QsaKvFormat::Q8,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        };
        let (max_seq, tile) = (16_640usize, purpose_tile_rows(AdmissionPurpose::Product));

        // S1 layout: the known boundary stage and the known census total.
        let stage = crate::state::Qwen4State::session_fixed_stage_bytes(&config, format).unwrap();
        assert_eq!(stage, 34_128_016, "target-only boundary stage (no MTP)");
        let inputs = AdmissionInputs {
            geometry: ModelGeometry::from_config(&config, WindowRoute::SlabIu4).unwrap(),
            memory: AdmissionMemory::actual(0),
            non_expert: 0,
            layer_expert_bytes: GATE_UP_LAYER + DOWN_LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: config.num_hidden_layers,
            base_reserve: 0,
            baseline_layers: 0,
            fixed_state_bytes: stage,
            pending_save_bytes: 0,
            diagnostic_bytes: 0,
            base_runtime_reserved_bytes: 0,
            granule: GranuleInputs::default(),
            checkpoint_stride_rows: 8192,
            window_rows: REFERENCE_WINDOW_ROWS.min(max_seq),
            link_gbps: 7.09,
            purpose: AdmissionPurpose::Product,
            sym_iu4_source_symmetric: None,
        };
        let (pages, arena, census) = census_of(&inputs).unwrap();
        let s1_total = census.owned_total().unwrap();
        // Route storage includes D's `k1_shared` (245,760 B) and the cohort
        // count words (2,048 B): 386,072,832 B, charged with A4 and the
        // grouped-scatter scratch at 471,927,040 B.
        assert_eq!(
            s1_total,
            2_543_094_592 - (S1_TILE_ROWS - tile) as u64 * inputs.geometry.wide().unwrap() * 2,
            "S1 layout at the product tile, no MTP"
        );
        // The ledger's sums agree before anything is emitted: the owned terms
        // sum to the owned total, and owned + budgets is the placement reserve.
        let owned_sum: u64 = census.owned_terms().iter().map(|(_, bytes)| bytes).sum();
        let budget_sum: u64 = census.budget_terms().iter().map(|(_, bytes)| bytes).sum();
        assert_eq!(owned_sum, s1_total);
        assert_eq!(owned_sum + budget_sum, census.total().unwrap());
        assert_eq!(
            census.total().unwrap(),
            s1_total + census.runtime_overhead + census.diagnostic
        );
        // Native MTP adds exactly the entry pending copy (hc * hidden * 4 B).
        let mtp_pending = (config.hc_count * config.hidden_size * 4) as u64;
        assert_eq!(mtp_pending, 40_960);
        let (_, _, mtp_census) = census_of(&AdmissionInputs {
            pending_save_bytes: mtp_pending,
            ..inputs
        })
        .unwrap();
        assert_eq!(
            mtp_census.owned_total().unwrap(),
            2_543_135_552 - (S1_TILE_ROWS - tile) as u64 * inputs.geometry.wide().unwrap() * 2,
            "S1 layout, native MTP"
        );
        assert_eq!(census.boundary_state, 4 * 34_128_016);
        assert_eq!(pages.total_pages(), 18);

        // The incumbent's tile reserve terms from the real helpers.
        let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
            hidden_size: config.hidden_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hc_count: config.hc_count,
            hc_lowrank: config.hc_lowrank,
            gdn_format: None,
            routed_i4: true,
            include_short_tails: true,
        };
        let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(tile, &dims)
            .expect("workspace terms");
        let workspace_total: u64 = workspace.iter().map(|(_, bytes)| *bytes).sum();
        let forward = crate::gpu_forward::qwen4_forward_device_bytes(&config, tile, Full).unwrap();
        let forward_census = crate::expert_residency::forward_census_scope(
            &config,
            tile,
            Full,
            &crate::weights::Qwen4BandScope::single(&config),
        )
        .unwrap();
        assert_eq!(forward_census.total().unwrap(), forward);
        let safety = crate::expert_residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES;
        // The corrected base census at this tile: the measured terms PLUS the F16
        // weight shadow cold bound of the real artifact (the retired
        // zero-shadow figure is never emitted as real-input truth).
        let base = j1778_reserve_at(tile);
        assert_eq!(base.census.f16_shadows, J1778_SHADOWS);
        assert_eq!(
            base.census.workspace.iter().map(|term| term.1).sum::<u64>(),
            workspace_total
        );

        let mut contexts = Vec::new();
        for (label, backend, granularity, conditional) in [
            ("legacy", crate::Qwen4KvBackend::Legacy, 1usize, false),
            ("vmm_2mib", crate::Qwen4KvBackend::Vmm, 2usize << 20, true),
        ] {
            let census_context =
                crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
            let reserve_context = crate::Qwen4ContextCommit::for_expert_reserve(
                backend,
                false,
                max_seq,
                tile,
                granularity,
            );
            let state =
                crate::expert_residency::state_census(&config, format, &census_context, tile, Full)
                    .unwrap();
            let historical = crate::expert_residency::auto_vram_reserve(
                &config,
                &reserve_context,
                tile,
                Full,
                format.layout(),
                None,
                None,
            )
            .unwrap();
            contexts.push(serde_json::json!({
                "backend": label,
                "conditional_on_observed_backend": conditional,
                "qsa_context_bytes": state.qsa_context,
                "state_bytes": state.state,
                "forward_bytes": state.forward,
                "historical_reserve_bytes": historical,
            }));
        }

        let ledger = serde_json::json!({
            "fixture": {
                "config": "compact_test_config",
                "arch": "gfx1100",
                "scratch_mode": "Full",
                "max_seq": max_seq,
                "tile_rows": tile,
                "window_rows": inputs.window_rows,
                "state_format": { "qsa": "q8", "gdn": "q8" },
                "native_mtp": false,
            },
            "tile": {
                "forward_device_bytes": forward,
                "forward_census": {
                    "stage_shared": forward_census.stage_shared,
                    "band_local": forward_census.band_local,
                    "input_only": forward_census.input_only,
                    "output_only": forward_census.output_only,
                },
                "f16_shadow_cold_bound_bytes": base.census.f16_shadows,
                "base_census_bytes": base.census_total,
                "base_census_bytes_without_shadows": base.census_total - base.census.f16_shadows,
                "workspace_total_bytes": workspace_total,
                "workspace_terms": workspace
                    .iter()
                    .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                    .collect::<Vec<_>>(),
            },
            "contexts": contexts,
            "safety_margin_bytes": safety,
            "s1": {
                // Physical allocations: `total_bytes` is exactly the sum of
                // `terms` (the seven owned terms); the two budget terms are
                // reported beside them and `reserve_bytes` is all nine, the
                // number placement charges.
                "total_bytes": s1_total,
                "terms": census
                    .owned_terms()
                    .iter()
                    .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                    .collect::<Vec<_>>(),
                "budget_terms": census
                    .budget_terms()
                    .iter()
                    .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                    .collect::<Vec<_>>(),
                "reserve_bytes": census.total().unwrap(),
                "boundary_stage_bytes": stage,
                "boundary_stages": S1_MAX_BOUNDARY_SNAPSHOTS,
                "hc_arena_bytes": arena.total_bytes,
                "page_payload_bound_bytes": pages.payload_bytes,
                "page_bytes": pages.page_bytes,
                "page_count": pages.total_pages(),
                "pinned_host_staging_bytes": pages.host_staging_bytes,
            },
        });
        println!(
            "SLOWLINK_RESIDENCY_LEDGER {}",
            serde_json::to_string(&ledger).unwrap()
        );
    }

    /// The window route comes from the REAL artifact layout, never a default,
    /// and reaches every dispatch dimension the census reads: a slab layer
    /// arena in the manifest is the slab route, stacked per-layer expert
    /// tensors are the aggregate route, and the declared route is what the
    /// route dims, the census layout and the diagnostic geometry carry.
    #[test]
    fn slowlink_admission_window_route_follows_the_artifact_layout() {
        use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
        let entry = |name: &str| {
            WeightEntry::model(
                name,
                vec![4, 4, 256],
                rdna_compute::DType::MQ4G256V2,
                ShardPolicy::Replicate,
            )
        };
        let aggregate = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            entry("model.language_model.layers.0.mlp.experts.down_proj"),
        ];
        let slab = vec![entry(&format!(
            "model.language_model.layers.0{}",
            crate::expert_arena::ARENA_SUFFIX
        ))];
        assert_eq!(window_route_of(&aggregate), WindowRoute::Aggregate);
        assert_eq!(window_route_of(&slab), WindowRoute::SlabIu4);
        // An empty manifest has no arena: stacked tensors are the layout.
        assert_eq!(window_route_of(&[]), WindowRoute::Aggregate);

        let config = crate::config::compact_test_config();
        for route in [
            WindowRoute::SlabIu4,
            WindowRoute::Aggregate,
            WindowRoute::AggregateSymIu4,
        ] {
            let geometry = ModelGeometry::from_config(&config, route).unwrap();
            assert_eq!(geometry.route, route);
            assert_eq!(route_dims(&geometry, 16384).route, route);
            // The census reads the layout of THIS route: A4 sidecars exist only
            // on the slab route, and the aggregate route's window F16
            // activations are inside its region.
            let layout = hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(
                &route_dims(&geometry, 16384),
            )
            .unwrap();
            assert_eq!(layout.route(), route);
            assert_eq!(
                route_storage_bytes(&geometry, 16384).unwrap(),
                layout.total_bytes() as u64
            );
            assert_eq!(
                a4_sidecar_bytes(&geometry, 16384).unwrap(),
                layout.a4_sidecar_bytes() as u64
            );
            assert_eq!(
                group_scatter_bytes(&geometry, 16384).unwrap(),
                layout.group_scatter_bytes() as u64
            );
            // Both routes carry a cohort (K1 rows are admitted on each), so
            // each owns one I32 count word per expert inside its region.
            assert_eq!(layout.cohort_counts_bytes(), geometry.num_experts * 4);
            match route {
                WindowRoute::SlabIu4 => {
                    assert!(layout.a4_sidecar_bytes() > 0);
                }
                WindowRoute::Aggregate => {
                    assert_eq!(layout.a4_sidecar_bytes(), 0);
                    assert_eq!(layout.x_f16_bytes(), 16384 * 2560 * 2);
                    assert_eq!(layout.x_f16_bytes(), 83_886_080);
                    // The tail ids are inside total_bytes: the getter is the raw
                    // 40 B (10 top-k I32) per tail row at the full 3,066-row
                    // worst case (122,640 B); the 256 B alignment belongs to the
                    // region field that holds them, not to this getter.
                    assert_eq!(
                        layout.tail_ids_bytes(),
                        S1_AGGREGATE_TAIL_ROWS
                            * hipfire_dispatch::pipeline::slowlink_prefill::K_TOP
                            * 4
                    );
                    assert_eq!(layout.tail_ids_bytes(), 122_640);
                    assert_eq!(
                        layout.charged_bytes() as u64,
                        route_storage_bytes(&geometry, 16384).unwrap()
                            + a4_sidecar_bytes(&geometry, 16384).unwrap()
                            + group_scatter_bytes(&geometry, 16384).unwrap()
                    );
                }
                WindowRoute::AggregateSymIu4 => {
                    // Both A4 sidecars, no window F16 input, the aggregate cohort bounds.
                    assert_eq!(layout.a4_sidecar_bytes(), 82_575_360);
                    assert_eq!(layout.x_f16_bytes(), 0);
                    assert_eq!(layout.tail_ids_bytes(), 122_640);
                    assert_eq!(
                        layout.charged_bytes() as u64,
                        route_storage_bytes(&geometry, 16384).unwrap()
                            + a4_sidecar_bytes(&geometry, 16384).unwrap()
                            + group_scatter_bytes(&geometry, 16384).unwrap()
                    );
                }
            }
        }
    }

    /// The fixture every layout test sizes from is the real MQ4 model, not a
    /// smaller synthetic one: the H0 MQ4 receipt's layer identity is gate/up
    /// `[512, 1280, 2560]` and down `[512, 2560, 640]` (H2560, expert
    /// intermediate 640, NOT 512), 1,740,800 / 870,400 payload bytes per expert.
    #[test]
    fn slowlink_admission_fixture_is_the_h0_mq4_layer_identity() {
        let geometry = geometry();
        assert_eq!(geometry.hidden, 2560);
        assert_eq!(geometry.moe_intermediate, 640);
        assert_eq!(geometry.num_experts, 512);
        assert_eq!(geometry.experts_per_token, 10);
        assert_eq!(geometry.hc_count, 4);
        assert_eq!(geometry.shared_intermediate, 640);
        // gate/up is [n_exp, 2 * mi, hidden], down [n_exp, hidden, mi].
        assert_eq!(2 * geometry.moe_intermediate, 1280);
        assert_eq!(GATE_UP_LAYER, 891_289_600);
        assert_eq!(DOWN_LAYER, 445_644_800);
        let pages = PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER).unwrap();
        assert_eq!(pages.gate_up_expert_bytes, 1_740_800);
        assert_eq!(pages.down_expert_bytes, 870_400);
    }

    #[test]
    fn slowlink_admission_census_equals_the_allocated_layouts() {
        let inputs = inputs(free_for(&inputs(0), 10));
        let profile = admit(&inputs).unwrap();
        assert_eq!(profile.census.total().unwrap(), census_total(&inputs));
        assert_eq!(profile.census.arena, profile.arena.total_bytes);
        assert_eq!(
            profile.census.ring + profile.census.lookahead,
            profile.pages.total_bytes().unwrap()
        );
        // The pool's own reserve is the page store: one authority.
        let reserve = profile.pages.pool_spec().unwrap().reserve().unwrap();
        assert_eq!(
            profile.pages.total_bytes().unwrap(),
            reserve.device_total as u64
        );
        assert_eq!(profile.pages.host_staging_bytes, reserve.staging as u64);
        assert_eq!(reserve.pages, 18);
        assert_eq!(
            profile.census.route_storage,
            route_storage_bytes(&profile.geometry, profile.window_rows).unwrap()
        );
        assert_eq!(
            profile.census.a4_sidecars,
            a4_sidecar_bytes(&profile.geometry, profile.window_rows).unwrap()
        );
        assert!(profile.census.a4_sidecars > 0 && profile.census.route_storage > 0);
        assert_eq!(
            profile.census.group_scatter,
            group_scatter_bytes(&profile.geometry, profile.window_rows).unwrap()
        );
        assert!(profile.census.group_scatter > 0);
        assert_eq!(profile.reserve_bytes().unwrap(), census_total(&inputs));
        assert_eq!(profile.tile_rows, S1_PRODUCT_TILE_ROWS);
        assert_eq!(profile.purpose, AdmissionPurpose::Product);
        assert!(profile.tile_rows <= S1_TILE_ROWS);
        assert_eq!(profile.window_rows, REFERENCE_WINDOW_ROWS);
        // Seven owned terms then two budget terms, every one itemised in log order.
        let names: Vec<&str> = profile
            .census
            .terms()
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            [
                "expert ring",
                "expert lookahead",
                "route storage",
                "A4 sidecars",
                "group scatter",
                "HC arena",
                "boundary state",
                "runtime overhead budget",
                "diagnostic budget",
            ]
        );
        assert!(profile.describe().contains("10 resident layers"));
    }

    #[test]
    fn slowlink_admission_keeps_the_largest_fitting_layers_and_the_loader_agrees() {
        let probe = inputs(0);
        // Zero resident layers is valid, ten is only preferred, more is allowed.
        for layers in 0..=14u64 {
            let inputs = inputs(free_for(&probe, layers));
            let profile = admit(&inputs).unwrap();
            assert_eq!(profile.resident_layers as u64, layers);
            assert_eq!(
                profile.placement.largest_fitting_layers(),
                Some(layers as usize)
            );
            // The placement the loader runs is the same arithmetic.
            let charges = PlacementCharges::new(LAYER, false)
                .unwrap()
                .with_slowlink(profile.reserve_bytes().unwrap());
            assert_eq!(
                auto_placement_layers(
                    inputs.memory.free_bytes,
                    inputs.non_expert,
                    LAYER,
                    48,
                    inputs.base_reserve,
                    &charges
                )
                .unwrap() as u64,
                layers
            );
        }
    }

    /// The profile OWNS the settled placement's charges: the margin (512 MiB),
    /// no dedicated stage, and the S1 census as `slowlink`, the very charges its
    /// joint reserve was quoted with. A loader that places the profile's
    /// resident layers records `profile.charges` (never a recomputation, never
    /// zero), and base + charges is exactly the admitted reserve.
    #[test]
    fn slowlink_profile_owns_its_settled_placement_charges() {
        let probe = inputs(0);
        for layers in [0u64, 8, 10, 12] {
            let inputs = inputs(free_for(&probe, layers));
            let profile = admit(&inputs).unwrap();
            let charges = profile.charges;
            assert_eq!(
                charges.margin,
                crate::expert_residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES
            );
            assert_eq!(charges.stage, 0, "an S1 load charges no dedicated stage");
            assert_eq!(charges.slowlink, profile.reserve_bytes().unwrap());
            assert_ne!(charges.slowlink, 0);
            assert_eq!(
                charges.reserve(inputs.base_reserve).unwrap(),
                profile.placement.reserve_bytes,
                "base + owned charges is the reserve the layers were quoted against"
            );
            assert_eq!(
                charges,
                PlacementCharges::new(LAYER, false)
                    .unwrap()
                    .with_slowlink(profile.reserve_bytes().unwrap())
            );
        }
    }

    #[test]
    fn slowlink_admission_below_preferred_admits_and_only_a_base_shortfall_is_a_deficit() {
        let probe = inputs(0);
        let nine = inputs(free_for(&probe, 9));
        let profile = admit(&nine).expect("nine layers are admitted");
        assert_eq!(profile.resident_layers, 9);
        let deficit = profile.placement.preferred_deficit_bytes();
        assert!(deficit > 0 && deficit <= LAYER);
        // Adding exactly the preferred deficit reaches ten; one byte short
        // keeps nine.
        let fixed = inputs(nine.memory.free_bytes + deficit);
        assert_eq!(admit(&fixed).unwrap().resident_layers, 10);
        let short = inputs(nine.memory.free_bytes + deficit - 1);
        assert_eq!(admit(&short).unwrap().resident_layers, 9);
        // Not even the non-expert weights and the charged reserve fit: the
        // BASE deficit, quoted for zero layers.
        let none = inputs(GIB);
        let Err(SlowlinkDecline::Deficit(none)) = admit(&none) else {
            panic!("a nearly empty card is a base deficit");
        };
        assert_eq!(none.required_layers, 0);
        assert_eq!(none.memory.free_bytes, GIB);
        assert_eq!(none.resident_layer_bytes, LAYER);
        assert_eq!(
            u128::from(none.non_expert)
                + u128::from(none.reserve)
                + none.required_layers as u128 * u128::from(none.resident_layer_bytes)
                - u128::from(GIB),
            u128::from(none.deficit_bytes)
        );
        assert!(SlowlinkDecline::Deficit(none).is_capacity());
        let text = SlowlinkDecline::Deficit(none).to_string();
        assert!(text.contains(&none.deficit_bytes.to_string()), "{text}");
        assert!(text.contains("actual free"), "{text}");
    }

    #[test]
    fn slowlink_admission_fully_resident_has_nothing_to_stream() {
        let inputs = inputs(200 * GIB);
        assert!(matches!(
            admit(&inputs),
            Err(SlowlinkDecline::NoHostLayers {
                resident_layers: 48
            })
        ));
        assert!(!SlowlinkDecline::NoHostLayers {
            resident_layers: 48
        }
        .is_capacity());
    }

    #[test]
    fn slowlink_admission_overflow_refuses_before_mutation() {
        let probe = inputs(0);
        let mut huge = inputs(free_for(&probe, 12));
        huge.fixed_state_bytes = u64::MAX;
        assert!(matches!(admit(&huge), Err(SlowlinkDecline::Geometry(_))));
        let mut huge = inputs(free_for(&probe, 12));
        huge.base_reserve = u64::MAX;
        assert!(matches!(admit(&huge), Err(SlowlinkDecline::Geometry(_))));
        let mut huge = inputs(free_for(&probe, 12));
        huge.window_rows = usize::MAX / 4;
        assert!(matches!(admit(&huge), Err(SlowlinkDecline::Geometry(_))));
        let mut bad = inputs(free_for(&probe, 12));
        bad.gate_up_layer_bytes += 1;
        assert!(matches!(admit(&bad), Err(SlowlinkDecline::Geometry(_))));
        assert!(SlowlinkDecline::Geometry(String::new()).is_capacity());
    }

    #[test]
    fn slowlink_admission_fast_route_charges_are_zero() {
        let layer = LAYER;
        let incumbent = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(incumbent.slowlink, 0);
        assert_eq!(incumbent.reserve(123).unwrap(), 123 + incumbent.margin);
        assert!(!incumbent.describe().contains("slow-link"));
        let charged = incumbent.with_slowlink(4 * GIB);
        assert_eq!(
            charged.reserve(123).unwrap(),
            123 + incumbent.margin + 4 * GIB
        );
        assert!(charged.describe().contains("slow-link"));
        assert!(PlacementCharges::new(layer, true)
            .unwrap()
            .with_slowlink(u64::MAX)
            .reserve(1)
            .is_err());
    }

    fn product_window_inputs(free: u64) -> AdmissionInputs {
        AdmissionInputs {
            window_rows: S1_WINDOW_ROWS,
            ..inputs(free)
        }
    }

    /// The product profile: [`S1_WINDOW_ROWS`] window, ten layers resident.
    fn profile_at_ten() -> SlowlinkProfile {
        admit(&product_window_inputs(free_for(&inputs(0), 10))).unwrap()
    }

    #[test]
    fn slowlink_admission_request_truth_table() {
        let profile = profile_at_ten();
        let req = |rows| RequestShape {
            rows,
            entry_pos: 0,
            session_cuts: 0,
            capture: CaptureKind::None,
        };
        // Forced: from one row past a tile up (the first genuinely multi-chunk
        // request); a request of at most one tile is a named bypass.
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(1536)),
            Err(RequestBypass::TooShort {
                rows: 1536,
                minimum: 1537
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(1537)),
            Ok(RequestVerdict {
                rows: 1537,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(3072)),
            Ok(RequestVerdict {
                rows: 3072,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(S1_WINDOW_ROWS)),
            Ok(RequestVerdict {
                rows: S1_WINDOW_ROWS,
                windows: 1
            })
        );
        // A prompt past the window runs superwindows and says so.
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(S1_WINDOW_ROWS + 1)),
            Ok(RequestVerdict {
                rows: S1_WINDOW_ROWS + 1,
                windows: 2
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Off, &profile, &req(16384)),
            Err(RequestBypass::KillSwitch)
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Auto, &profile, &req(16384)),
            Ok(RequestVerdict {
                rows: 16384,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(0)),
            Err(RequestBypass::NoRows)
        );
        for (capture, bypass) in [
            (CaptureKind::AllRows, RequestBypass::AllRowCapture),
            (CaptureKind::External, RequestBypass::ExternalCapture),
        ] {
            assert_eq!(
                qualify_request(
                    SlowlinkMode::Force,
                    &profile,
                    &RequestShape {
                        capture,
                        ..req(16384)
                    }
                ),
                Err(bypass)
            );
        }
        for capture in [CaptureKind::None, CaptureKind::Bounded] {
            assert!(qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape {
                    capture,
                    entry_pos: 4096,
                    ..req(16384)
                }
            )
            .is_ok());
        }
        assert_eq!(
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape {
                    session_cuts: 4,
                    ..req(16384)
                }
            )
            .unwrap()
            .windows,
            1
        );
        assert_eq!(
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape {
                    session_cuts: 5,
                    ..req(16384)
                }
            ),
            Err(RequestBypass::TooManyBoundaries {
                cuts: 5,
                maximum: 4
            })
        );
        // At a 1536-row admitted tile automatic and forced share the 1537-row
        // floor (one row past the tile): at most one tile is a named bypass.
        assert_eq!(S1_MIN_FORCED_REQUEST_ROWS, 1537);
        let wide = SlowlinkProfile {
            tile_rows: 1536,
            ..profile
        };
        for mode in [SlowlinkMode::Auto, SlowlinkMode::Force] {
            assert_eq!(
                qualify_request(mode, &wide, &req(1536)),
                Err(RequestBypass::TooShort {
                    rows: 1536,
                    minimum: 1537
                }),
                "{mode:?}"
            );
            assert_eq!(
                qualify_request(mode, &wide, &req(1537)),
                Ok(RequestVerdict {
                    rows: 1537,
                    windows: 1
                }),
                "{mode:?}"
            );
        }
        // A 768-row admitted tile (this profile, and the slow-link XTX
        // Flash-Next load): the automatic floor follows the tile, so
        // 769..=1536 rows (two incumbent chunks) run layer-major; the forced
        // floor stays the legal-maximum one.
        assert_eq!(profile.tile_rows, 768);
        assert_eq!(
            qualify_request(SlowlinkMode::Auto, &profile, &req(768)),
            Err(RequestBypass::TooShort {
                rows: 768,
                minimum: 769
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Auto, &profile, &req(769)),
            Ok(RequestVerdict {
                rows: 769,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(1536)),
            Err(RequestBypass::TooShort {
                rows: 1536,
                minimum: 1537
            })
        );
        assert_eq!(profile.windows_for(1), 1);
        assert_eq!(profile.windows_for(32768), 2);
        // 22 x 768-row tiles per window: fills (cache on/off do not change the
        // cut-free count) 8,192 / 32,768 / 32,780 / 131,084 / 250,012.
        assert_eq!(S1_WINDOW_ROWS, 16_896);
        assert_eq!(profile.tile_rows, 768);
        for (rows, windows) in [
            (8_192usize, 1usize),
            (16_896, 1),
            (16_897, 2),
            (32_768, 2),
            (32_780, 2),
            (33_792, 2),
            (33_793, 3),
            (131_084, 8),
            (250_012, 15),
        ] {
            assert_eq!(profile.windows_for(rows), windows, "{rows} rows");
        }
    }

    /// The REAL planner's window and sweep counts at the 22-tile window: every
    /// window is one expert sweep, so windows == sweeps. Cache off has no
    /// session cuts; cache on cuts every 8192-row stride boundary.
    #[test]
    fn slowlink_admission_window_sweeps_at_the_tile_aligned_window() {
        use crate::slowlink_prefill::{plan_prompt_verdict, PromptPlanInputs, TileRouting};
        use hipfire_runtime::session_cache::stride_boundaries;
        let profile = profile_at_ten();
        assert_eq!(profile.tile_rows, S1_PRODUCT_TILE_ROWS);
        assert_eq!(profile.window_rows, S1_WINDOW_TILES * S1_PRODUCT_TILE_ROWS);
        let routing = |rows: usize| TileRouting {
            route_rows: rows,
            bf16_streams: rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS,
        };
        let stride = profile.checkpoint_stride_rows;
        for (rows, expect_off, expect_on) in [
            (8_192usize, 1usize, 1usize),
            (32_768, 2, 2),
            (32_780, 2, 2),
            (131_084, 8, 8),
            (250_012, 15, 16),
        ] {
            for cache in [false, true] {
                let cuts: Vec<usize> = if cache {
                    stride_boundaries(stride, &[], None, 0, rows)
                        .into_iter()
                        .filter(|&b| b < rows)
                        .collect()
                } else {
                    Vec::new()
                };
                let plan = plan_prompt_verdict(&PromptPlanInputs {
                    profile: &profile,
                    entry_pos: 0,
                    rows,
                    cuts: &cuts,
                    routing: &routing,
                    few_row_limit: 8,
                })
                .unwrap()
                .expect("a qualified request plans");
                let expect = if cache { expect_on } else { expect_off };
                assert_eq!(plan.windows.len(), expect, "{rows} rows, cache {cache}");
                assert_eq!(plan.windows.iter().map(|w| w.rows).sum::<usize>(), rows);
                assert!(plan.windows.iter().all(|w| w.rows <= S1_WINDOW_ROWS));
            }
        }
    }

    /// Placement slack at the 22-tile product window of the two measured
    /// 262,144-row XTX Flash-Next loads (F0 §2 less the +25,070,848-byte window
    /// delta): AR-loaded keeps 6 resident layers, MTP-loaded 5.
    const AR_SLACK_AT_22_TILES: u64 = 1_158_091_868 - 25_070_848;
    const MTP_SLACK_AT_22_TILES: u64 = 488_382_532 - 25_070_848;
    const FN_MAX_SEQ: usize = 262_144;

    /// Inputs whose free memory keeps `layers` resident with exactly `slack`
    /// bytes left at the 22-tile window.
    fn grown_inputs(layers: u64, slack: u64) -> AdmissionInputs {
        product_window_inputs(free_for(&product_window_inputs(0), layers) - 1 + slack)
    }

    fn plan_windows(profile: &SlowlinkProfile, rows: usize, cache: bool) -> Vec<usize> {
        use crate::slowlink_prefill::{plan_prompt_verdict, PromptPlanInputs, TileRouting};
        use hipfire_runtime::session_cache::stride_boundaries;
        let routing = |rows: usize| TileRouting {
            route_rows: rows,
            bf16_streams: rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS,
        };
        let cuts: Vec<usize> = if cache {
            stride_boundaries(profile.checkpoint_stride_rows, &[], None, 0, rows)
                .into_iter()
                .filter(|&b| b < rows)
                .collect()
        } else {
            Vec::new()
        };
        plan_prompt_verdict(&PromptPlanInputs {
            profile,
            entry_pos: 0,
            rows,
            cuts: &cuts,
            routing: &routing,
            few_row_limit: 8,
        })
        .unwrap()
        .expect("a qualified request plans")
        .windows
        .iter()
        .map(|w| w.rows)
        .collect()
    }

    /// The real-memory slack of a grown profile (its quote holds
    /// [`S1_GROWN_FREE_HOLD_BYTES`] back from `real`).
    fn real_slack(profile: &SlowlinkProfile, real: AdmissionMemory) -> u64 {
        let budget = PlacementBudget {
            memory: real,
            ..profile.placement
        };
        u64::try_from(budget.slack_bytes_at(profile.resident_layers)).unwrap()
    }

    /// The AR-loaded measured 262,144-row Flash-Next load's inputs with the
    /// KV charge of `max_seq`: `base_reserve` carries the load's KV (the
    /// production census [`single_device_reserve`] charges
    /// `for_expert_reserve(max_seq)` / `Qwen4ContextCommit::new(.., max_seq, ..)`),
    /// so the reserve moves by the real helper's difference between `max_seq` and
    /// the measured [`FN_MAX_SEQ`]; the free memory is the same card's.
    fn fn_inputs_at(max_seq: usize) -> AdmissionInputs {
        let inputs = grown_inputs(6, AR_SLACK_AT_22_TILES);
        let at = |max_seq| {
            i128::from(j1778_reserve_for(S1_PRODUCT_TILE_ROWS, max_seq).placement_bytes(true))
        };
        AdmissionInputs {
            base_reserve: u64::try_from(
                i128::from(inputs.base_reserve) + at(max_seq) - at(FN_MAX_SEQ),
            )
            .unwrap(),
            ..inputs
        }
    }

    /// The load line's numbers at `max_seq`: (window tiles, resident layers,
    /// real remaining slack bytes, free hold included).
    fn fn_load_at(max_seq: usize) -> (usize, usize, u64) {
        let inputs = fn_inputs_at(max_seq);
        let grown = admit_grown(&inputs, max_seq).unwrap();
        let slack = grown.remaining_slack_bytes(inputs.memory.free_bytes);
        println!(
            "max_seq {max_seq}: tiles {} layers {} slack {slack}",
            grown.window_rows / S1_PRODUCT_TILE_ROWS,
            grown.resident_layers
        );
        (
            grown.window_rows / S1_PRODUCT_TILE_ROWS,
            grown.resident_layers,
            slack,
        )
    }

    /// The window is sized at load for the load's `max_seq` (KV charged for it)
    /// and takes priority over extra resident layers; a 262,144-row load that
    /// cannot afford growth keeps the 22-tile window.
    #[test]
    fn slowlink_admission_grown_window_follows_max_seq_on_flash_next() {
        let tile = S1_PRODUCT_TILE_ROWS;
        let long = fn_load_at(262_144);
        let mid = fn_load_at(131_072);
        let short = fn_load_at(33_024);
        println!("(tiles, layers, slack): 262144 {long:?} 131072 {mid:?} 33024 {short:?}");
        // A multi-window context keeps the floor's 6 layers and grows only
        // with the memory left at that count above the hold: one tile here.
        assert_eq!((long.0, long.1), (S1_WINDOW_TILES + 1, 6));
        // 131,072 keeps its floor layers (8: less KV than 262,144); nothing
        // above the hold is left for a tile, so it is the floor admission and
        // its slack carries no hold.
        assert_eq!((mid.0, mid.1), (S1_WINDOW_TILES, 8));
        // A single-window context: window priority, 43 tiles, 8 layers.
        assert_eq!((short.0, short.1), (43, 8));
        assert!(long.2 >= S1_GROWN_FREE_HOLD_BYTES);
        assert!(short.0 >= 43, "32,780 rows in one window");
        // 43 is the covering tile count of a 32,780-row prompt: 42 tiles fall short.
        assert!(42 * tile < 32_780 && 43 * tile >= 32_780);
        assert!(short.2 >= S1_GROWN_FREE_HOLD_BYTES, "1 GiB held free");
    }

    /// Growth never refuses, never leaves less than the hold, and the window is
    /// the largest whole-tile one that admits (maximal on the ladder).
    #[test]
    fn slowlink_admission_grown_window_never_exceeds_the_budget() {
        let tile = S1_PRODUCT_TILE_ROWS;
        for layers in [2u64, 6, 9] {
            for slack_mib in [
                0u64, 1, 200, 256, 300, 442, 700, 1_080, 1_279, 1_500, 4_000, 40_000,
            ] {
                let inputs = grown_inputs(layers, slack_mib * MIB);
                let floor = admit(&inputs).unwrap();
                let grown = admit_grown(&inputs, FN_MAX_SEQ).unwrap();
                assert!(grown.window_rows >= S1_WINDOW_ROWS);
                assert_eq!(grown.window_rows % tile, 0);
                assert!(grown.window_rows <= S1_GROWN_WINDOW_MAX_TILES * tile);
                if grown.window_rows == S1_WINDOW_ROWS {
                    assert_eq!(grown, floor, "no room: the floor profile exactly");
                } else {
                    let real = real_slack(&grown, inputs.memory);
                    assert!(
                        real >= S1_GROWN_FREE_HOLD_BYTES,
                        "{layers} layers {slack_mib} MiB"
                    );
                    assert_eq!(grown.remaining_slack_bytes(inputs.memory.free_bytes), real);
                    let held = AdmissionInputs {
                        window_rows: grown.window_rows,
                        memory: AdmissionMemory::actual(
                            inputs.memory.free_bytes - S1_GROWN_FREE_HOLD_BYTES,
                        ),
                        ..inputs
                    };
                    assert_eq!(admit(&held).unwrap(), grown);
                    assert_eq!(grown.census.arena, grown.arena.total_bytes);
                    assert_eq!(
                        grown.census.route_storage,
                        route_storage_bytes(&grown.geometry, grown.window_rows).unwrap()
                    );
                }
                // Maximal on the held-back quote.
                let next = grown.window_rows + tile;
                if next <= S1_GROWN_WINDOW_MAX_TILES * tile {
                    let held = AdmissionInputs {
                        window_rows: next,
                        memory: AdmissionMemory::actual(
                            inputs.memory.free_bytes - S1_GROWN_FREE_HOLD_BYTES,
                        ),
                        ..inputs
                    };
                    // A multi-window context keeps the floor's layers: one more
                    // tile must not admit at that count.
                    assert!(
                        admit(&held).map_or(true, |p| p.resident_layers != floor.resident_layers),
                        "{layers} layers {slack_mib} MiB"
                    );
                }
            }
        }
        // A context shorter than the floor window, or one that fits it, is not grown.
        let inputs = grown_inputs(6, AR_SLACK_AT_22_TILES);
        for max_seq in [S1_MIN_FORCED_REQUEST_ROWS, 8_204, S1_WINDOW_ROWS] {
            let inputs = AdmissionInputs {
                window_rows: S1_WINDOW_ROWS.min(max_seq),
                ..inputs
            };
            assert_eq!(
                admit_grown(&inputs, max_seq).unwrap().window_rows,
                S1_WINDOW_ROWS.min(max_seq)
            );
        }
        // A refusal of the base admission stays the base refusal.
        let none = product_window_inputs(GIB);
        assert_eq!(
            admit_grown(&none, FN_MAX_SEQ).map(|_| ()),
            admit(&none).map(|_| ())
        );
    }

    /// Window and sweep counts (every window is one expert sweep) of the REAL
    /// planner on the grown profiles at the depth-sweep prompts, cache off and
    /// on (cuts at every 8,192-row stride boundary).
    #[test]
    fn slowlink_admission_grown_window_sweep_counts_at_the_depths() {
        let ar = admit_grown(&grown_inputs(6, AR_SLACK_AT_22_TILES), FN_MAX_SEQ).unwrap();
        let mtp = admit_grown(&grown_inputs(5, MTP_SLACK_AT_22_TILES), FN_MAX_SEQ).unwrap();
        let floor = admit(&grown_inputs(6, AR_SLACK_AT_22_TILES)).unwrap();
        assert_eq!(
            (
                ar.window_rows,
                ar.resident_layers,
                mtp.window_rows,
                mtp.resident_layers
            ),
            (17_664, 6, S1_WINDOW_ROWS, 5)
        );
        // 262,144 is a multi-window context: the floor's layers are kept and
        // the AR load's leftover above the hold buys one tile (23); the
        // MTP-loaded load has none and keeps the floor.
        for (rows, ar_sweeps, mtp_sweeps, floor_sweeps) in [
            (8_204usize, 1usize, 1usize, 1usize),
            (32_780, 2, 2, 2),
            (131_084, 8, 8, 8),
            (250_012, 15, 15, 15),
        ] {
            for cache in [false, true] {
                for (label, profile, expect) in [
                    ("AR", &ar, ar_sweeps),
                    ("MTP", &mtp, mtp_sweeps),
                    ("22-tile floor", &floor, floor_sweeps),
                ] {
                    let windows = plan_windows(profile, rows, cache);
                    assert_eq!(windows.iter().sum::<usize>(), rows);
                    assert!(windows.iter().all(|&w| w <= profile.window_rows));
                    // Cuts may split a window (cache on); never fewer than the cut-free count.
                    assert_eq!(profile.windows_for(rows), expect, "{label} {rows}");
                    if cache {
                        assert!(windows.len() >= expect, "{label} {rows}");
                    } else {
                        assert_eq!(windows.len(), expect, "{label} {rows}");
                    }
                }
            }
        }
    }

    /// Levelled windows: the same window count as the greedy packing, no
    /// wire-bound sliver tail, never above the admitted window.
    #[test]
    fn slowlink_admission_windows_are_levelled_across_the_sweeps() {
        let tile = S1_PRODUCT_TILE_ROWS;
        let floor = admit(&grown_inputs(6, AR_SLACK_AT_22_TILES)).unwrap();
        // 22 tiles + one row-tile: greedy was 16,896 + 768; levelled is 12 + 11 tiles.
        let rows = S1_WINDOW_ROWS + tile;
        assert_eq!(
            plan_windows(&floor, rows, false),
            vec![12 * tile, 11 * tile]
        );
        // 22 tiles + 1 row: 12 / 11 tiles, the last short by 767 rows.
        let rows = S1_WINDOW_ROWS + 1;
        assert_eq!(
            plan_windows(&floor, rows, false),
            vec![12 * tile, rows - 12 * tile]
        );
        // 32,780 at the floor: 22 + 21 tiles was already level.
        let windows = plan_windows(&floor, 32_780, false);
        assert_eq!(windows, vec![22 * tile, 32_780 - 22 * tile]);
        // Levelled counts equal the greedy counts everywhere and never grow a window.
        for rows in [
            769usize, 5_000, 16_897, 17_000, 25_000, 40_000, 131_084, 250_012,
        ] {
            for cache in [false, true] {
                let windows = plan_windows(&floor, rows, cache);
                assert!(windows.iter().all(|&w| w <= floor.window_rows));
                if !cache {
                    assert_eq!(windows.len(), floor.windows_for(rows), "{rows}");
                }
                // The widest window is the even share of the tiles, rounded up.
                if !cache && windows.len() > 1 {
                    let share = rows.div_ceil(tile).div_ceil(windows.len()) * tile;
                    assert_eq!(*windows.iter().max().unwrap(), share, "{rows}: {windows:?}");
                }
            }
        }
    }

    /// The window change's whole charge, same fixture, 16,384 -> 16,896 rows.
    #[test]
    fn slowlink_admission_window_charge_delta_is_pinned() {
        let free = free_for(&inputs(0), 10);
        let old = admit(&inputs(free)).unwrap();
        let new = admit(&product_window_inputs(free)).unwrap();
        let delta = new.reserve_bytes().unwrap() - old.reserve_bytes().unwrap();
        assert_eq!(new.census.arena - old.census.arena, 10_485_760);
        assert_eq!(
            new.census.route_storage - old.census.route_storage,
            11_902_208
        );
        assert_eq!(new.census.a4_sidecars - old.census.a4_sidecars, 2_580_480);
        assert_eq!(new.census.group_scatter - old.census.group_scatter, 102_400);
        assert_eq!(delta, 25_070_848, "23.9 MiB");
    }

    #[test]
    fn slowlink_admission_route_dims_are_the_census_dims() {
        let profile = profile_at_ten();
        let dims = profile.route_dims();
        assert_eq!(dims.rows, profile.window_rows);
        assert_eq!(dims.hidden, profile.geometry.hidden);
        assert_eq!(dims.n_exp, 512);
        assert_eq!(dims.mi, profile.geometry.moe_intermediate);
        assert_eq!(
            dims.shared_intermediate,
            profile.geometry.shared_intermediate
        );
        // At most (4 session-cut segments + 1 entry tile + the window tail) short tiles of 8 rows.
        assert_eq!(dims.k1_rows, 48);
        assert_eq!(S1_K1_TAIL_ROWS, 48);
        let layout =
            hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&dims).unwrap();
        assert_eq!(
            profile.census.route_storage,
            layout.total_bytes() as u64,
            "census and allocation read one layout"
        );
        assert_eq!(profile.census.a4_sidecars, layout.a4_sidecar_bytes() as u64);
        assert_eq!(
            profile.census.group_scatter,
            layout.group_scatter_bytes() as u64
        );
        // Full window grouped-scatter scratch at N=16896 slots=168960, E=512
        // (3,278,848 B at the former 16384-row window): the 1536-tile's
        // 309,248 B stays in the base reserve (replacement peak holds old +
        // new), so the full figure is charged on top.
        assert_eq!(profile.census.group_scatter, 3_381_248);
    }

    #[test]
    fn slowlink_admission_shared_helper_builds_nothing_for_a_load_that_cannot_select() {
        use std::cell::Cell;
        let facts_built = Cell::new(false);
        let facts = || -> Result<LoadFacts<'static>, String> {
            facts_built.set(true);
            Err("facts must not be built".to_string())
        };
        // Killed and structurally ineligible loads: no probe, no facts. Arch
        // is identity only: an arch-named load still probes (only sealed
        // verdicts refuse by capability).
        for (mode, load) in [
            (SlowlinkMode::Off, shape()),
            (
                SlowlinkMode::Force,
                LoadShape {
                    explicit_vram_layers: true,
                    ..shape()
                },
            ),
        ] {
            let outcome = select_load_profile(mode, &load, || unreachable!("probed"), facts);
            assert_eq!(outcome, Ok(None));
            assert!(!facts_built.get());
        }
        // A measured fast link probes once and still builds no facts.
        for link in [
            HostLink::Discrete { gbps: 16.0 },
            HostLink::Discrete { gbps: 26.0 },
            HostLink::Unified,
            HostLink::Unmeasured,
        ] {
            let probes = Cell::new(0);
            let outcome = select_load_profile(
                SlowlinkMode::Force,
                &shape(),
                || {
                    probes.set(probes.get() + 1);
                    link
                },
                facts,
            );
            assert_eq!(outcome, Ok(None), "{link:?}");
            assert_eq!(probes.get(), 1);
            assert!(!facts_built.get(), "{link:?}");
        }
        // A narrow link asks for the facts, and a facts failure is a capacity
        // refusal of an eligible forced load, never a silent incumbent.
        let error = select_load_profile(
            SlowlinkMode::Force,
            &shape(),
            || HostLink::Discrete { gbps: 7.09 },
            facts,
        )
        .unwrap_err();
        assert!(facts_built.get());
        assert!(error.contains("facts must not be built"), "{error}");
    }

    /// A granule-model copy of `inputs`: the measured 2 MiB granule (a test
    /// input, exactly what the load takes from the device's query) and a
    /// distinct nonzero base slack per owner.
    fn with_granule(inputs: AdmissionInputs) -> AdmissionInputs {
        AdmissionInputs {
            granule: GranuleInputs {
                granule_bytes: 2 * MIB,
                base: crate::expert_residency::GranuleSlack {
                    weights: 1_000_003,
                    f16_shadows: 2_000_005,
                    forward: 3_000_007,
                    state: 4_000_009,
                    workspace: 5_000_011,
                    native_mtp: 6_000_013,
                    qsa_workspace: 7_000_017,
                },
                boundary_stage_slack: 1_234,
            },
            ..inputs
        }
    }

    /// The per-owner slack is the distance to the next granule for an
    /// allocation of at least one granule and nothing below it; the measured
    /// j1786 allocations reproduce exactly.
    #[test]
    fn slowlink_granule_slack_is_the_distance_to_the_next_granule() {
        use crate::expert_residency::{granule_slack, granule_slack_total};
        let g = 2 * MIB;
        // Language head, resident down arena, gate/up arena, the cache's two
        // snapshot sizes: each sampler step of the run, byte for byte.
        assert_eq!(496_640_000 + granule_slack(496_640_000, g), 497_025_024);
        assert_eq!(445_644_800 + granule_slack(445_644_800, g), 446_693_376);
        assert_eq!(granule_slack(891_289_600, g), 0);
        assert_eq!(34_130_944 + granule_slack(34_130_944, g), 35_651_584);
        assert_eq!(34_183_168 + granule_slack(34_183_168, g), 35_651_584);
        // Below one granule, exactly one granule, and no granule model.
        assert_eq!(granule_slack(g - 1, g), 0);
        assert_eq!(granule_slack(g, g), 0);
        assert_eq!(granule_slack(g + 1, g), g - 1);
        assert_eq!(granule_slack(123_456_789, 0), 0);
        // The largest allocation never overflows; a total that would is refused.
        assert_eq!(granule_slack(u64::MAX, g), 1);
        let huge = 1u64 << 63;
        assert_eq!(granule_slack_total([huge + 1; 2], huge), Ok(2 * (huge - 1)));
        assert!(granule_slack_total([huge + 1; 3], huge)
            .unwrap_err()
            .contains("overflows"));
        assert_eq!(granule_slack_total([], g), Ok(0));
    }

    /// The shadow slack is the exact per-owner figure of the real 291-owner
    /// list the placement census charges, on the one shared predicate.
    #[test]
    fn slowlink_granule_shadow_slack_is_exact_on_the_real_owner_list() {
        use crate::expert_residency as residency;
        let manifest = residency::j1778_bf16_manifest();
        let payload = residency::f16_shadow_bytes(&manifest, 48).unwrap();
        assert_eq!(payload, J1778_SHADOWS);
        let slack = residency::f16_shadow_granule_slack(&manifest, 48, 2 * MIB).unwrap();
        assert_eq!(slack, 387_186_688);
        assert_eq!(payload + slack, 1_686_110_208);
        assert_eq!(residency::f16_shadow_granule_slack(&manifest, 48, 0), Ok(0));
        // A shorter trunk drops owners on both sides of the predicate alike.
        let short = residency::f16_shadow_bytes(&manifest, 24).unwrap();
        assert!(short < payload);
        assert!(residency::f16_shadow_granule_slack(&manifest, 24, 2 * MIB).unwrap() < slack);
    }

    /// Every S1 owner's slack is the slack of the allocation its own layout
    /// makes, the reserve is the census plus those, and a payload-only
    /// admission (the default, every non-S1 caller) is byte for byte the
    /// census alone.
    #[test]
    fn slowlink_granule_owners_equal_their_allocations() {
        use crate::expert_residency::granule_slack;
        let payload_only = inputs(free_for(&inputs(0), 10));
        let off = admit(&payload_only).unwrap();
        assert_eq!(off.granule, GranuleCharge::default());
        assert_eq!(off.granule.total(), Ok(0));
        assert_eq!(off.reserve_bytes().unwrap(), census_total(&payload_only));

        let on_inputs = with_granule(inputs(0));
        let (pages, arena, census) = census_of(&on_inputs).unwrap();
        let charge = granule_charge(&on_inputs, &pages, &arena, &census).unwrap();
        let g = 2 * MIB;
        let pool = pages.pool_spec().unwrap().reserve().unwrap();
        assert_eq!(pool.page_bytes as u64, pages.page_bytes);
        assert_eq!(
            charge.expert_pages,
            pages.total_pages() as u64 * granule_slack(pool.page_bytes as u64, g)
        );
        assert_eq!(charge.route_region, granule_slack(census.route_storage, g));
        assert_eq!(charge.hc_arena, granule_slack(arena.total_bytes, g));
        assert_eq!(
            charge.layer,
            granule_slack(GATE_UP_LAYER, g) + granule_slack(DOWN_LAYER, g)
        );
        assert_eq!(charge.boundary_stages, 4 * 1_234);
        // The slab route's three Gpu grow-only scratch buffers: two A4 slots
        // (20 x 16,384 x 72 B and 5 x 16,384 x 10 x 72 B) and the grouped
        // scatter slot (3,278,848 B, over one granule), each its own
        // allocation: 1,572,864 + 1,835,008 + 915,456 B of slack.
        let scratch = window_scratch_sizes(&on_inputs.geometry, on_inputs.window_rows).unwrap();
        assert_eq!(scratch, vec![23_592_960, 58_982_400, 3_278_848]);
        assert_eq!(
            scratch.iter().sum::<u64>(),
            a4_sidecar_bytes(&on_inputs.geometry, on_inputs.window_rows).unwrap()
                + group_scatter_bytes(&on_inputs.geometry, on_inputs.window_rows).unwrap()
        );
        assert_eq!(granule_slack(3_278_848, g), 915_456);
        assert_eq!(charge.window_scratch, 1_572_864 + 1_835_008 + 915_456);
        // The aggregate route has no A4 sidecars: only the scatter slot.
        let aggregate = ModelGeometry::from_config(
            &crate::config::compact_test_config(),
            WindowRoute::Aggregate,
        )
        .unwrap();
        assert_eq!(
            window_scratch_sizes(&aggregate, on_inputs.window_rows).unwrap(),
            vec![3_278_848]
        );
        // The Flash-Next layer: gate/up is a granule multiple, down is not.
        assert_eq!(granule_slack(GATE_UP_LAYER, g), 0);
        assert_eq!(granule_slack(DOWN_LAYER, g), 1_048_576);
        assert_eq!(charge.base, on_inputs.granule.base);
        let owners: u64 = charge.terms().iter().map(|(_, bytes)| bytes).sum();
        assert_eq!(charge.total(), Ok(owners));
        assert!(owners > 15_000_000);

        // Free memory that keeps exactly ten layers beside the payload census
        // AND the granule slack.
        let probe = with_granule(inputs(
            free_for(&inputs(0), 10) + charge.total().unwrap() + 10 * charge.layer,
        ));
        let on = admit(&probe).unwrap();
        assert_eq!(on.granule, charge);
        assert_eq!(
            on.reserve_bytes().unwrap(),
            census.total().unwrap() + charge.total().unwrap()
        );
        // The census itself, and so every receipt owner reading it, is unchanged.
        assert_eq!(on.census, off.census);
        assert_eq!(on.census.terms().len(), 9);
        assert_eq!(on.census.owned_terms().len(), 7);
        // The loader's own placement over the profile's reserve agrees.
        let layers = auto_placement_layers(
            probe.memory.free_bytes,
            probe.non_expert,
            on.placement_layer_bytes(probe.layer_expert_bytes).unwrap(),
            probe.trunk_layers,
            probe.base_reserve,
            &PlacementCharges::new(probe.layer_expert_bytes, false)
                .unwrap()
                .with_slowlink(on.reserve_bytes().unwrap()),
        )
        .unwrap();
        assert_eq!(layers, on.resident_layers);
    }

    /// Owner overflow refuses the load before any allocation, with a named
    /// error, exactly like every other census term.
    #[test]
    fn slowlink_granule_overflow_refuses_before_mutation() {
        let mut huge = with_granule(inputs(free_for(&inputs(0), 12)));
        huge.granule.base.weights = u64::MAX;
        match admit(&huge) {
            Err(SlowlinkDecline::Geometry(message)) => {
                assert!(message.contains("overflows"), "{message}");
            }
            other => panic!("an overflowing granule term must refuse, got {other:?}"),
        }
        // A reserve that no longer fits the free memory is a deficit, never a
        // wrapped budget.
        let mut tight = with_granule(inputs(0));
        tight.memory = AdmissionMemory::actual(1 << 30);
        assert!(matches!(admit(&tight), Err(SlowlinkDecline::Deficit(_))));
    }

    /// The device snapshot budget is exactly the free memory left after the
    /// non-expert weights, the resident layers and the whole joint reserve
    /// (safety margin and granule slack included): never a draw on the safety
    /// margin, zero at the exact edge, and one byte short is a deficit.
    #[test]
    fn slowlink_granule_budget_never_draws_the_safety_margin() {
        let probe = with_granule(inputs(0));
        let (pages, arena, census) = census_of(&probe).unwrap();
        let charge = granule_charge(&probe, &pages, &arena, &census).unwrap();
        let granule = charge.total().unwrap();
        let layer_cost = LAYER + charge.layer;
        let fixed = probe.non_expert
            + probe.base_reserve
            + EXPERT_MARGIN
            + census.total().unwrap()
            + granule;
        for layers in 0..=12u64 {
            for extra in [0, 1, 4096, MIB, 100 * MIB, LAYER - 1] {
                let free = fixed + layers * layer_cost + extra;
                let profile = admit(&AdmissionInputs {
                    memory: AdmissionMemory::actual(free),
                    ..probe
                })
                .unwrap();
                assert_eq!(profile.resident_layers as u64, layers, "{layers} {extra}");
                assert_eq!(profile.device_snapshot_budget_bytes, extra);
                // free = weights + layers + base + SAFETY + S1 reserve + budget.
                assert_eq!(
                    free,
                    probe.non_expert
                        + layers * layer_cost
                        + probe.base_reserve
                        + EXPERT_MARGIN
                        + profile.reserve_bytes().unwrap()
                        + profile.device_snapshot_budget_bytes
                );
            }
        }
        // The snapshot budget is the quote slack at the selected count.
        let edge = fixed + 10 * layer_cost;
        let at_edge = admit(&AdmissionInputs {
            memory: AdmissionMemory::actual(edge),
            ..probe
        })
        .unwrap();
        assert_eq!(at_edge.resident_layers, 10);
        assert_eq!(at_edge.device_snapshot_budget_bytes, 0);
        assert_eq!(at_edge.placement.slack_bytes_at(10), 0);
        // One byte short keeps nine layers, never a draw on the margin.
        let below = admit(&AdmissionInputs {
            memory: AdmissionMemory::actual(edge - 1),
            ..probe
        })
        .unwrap();
        assert_eq!(below.resident_layers, 9);
        assert_eq!(
            i128::from(below.device_snapshot_budget_bytes),
            below.placement.slack_bytes_at(9)
        );
        assert_eq!(below.device_snapshot_budget_bytes, layer_cost - 1);
        // The base itself: zero layers admitted at the exact edge, a deficit
        // one byte below.
        let base = admit(&AdmissionInputs {
            memory: AdmissionMemory::actual(fixed),
            ..probe
        })
        .unwrap();
        assert_eq!(base.resident_layers, 0);
        assert_eq!(base.device_snapshot_budget_bytes, 0);
        assert!(matches!(
            admit(&AdmissionInputs {
                memory: AdmissionMemory::actual(fixed - 1),
                ..probe
            }),
            Err(SlowlinkDecline::Deficit(_))
        ));
    }

    /// A load that cannot select S1 (fast link, UMA, explicit layers, other
    /// mode) never builds the granule facts or queries the device granule, and
    /// an incumbent charge set is byte for byte what it was.
    #[test]
    fn slowlink_granule_leaves_the_incumbent_and_fast_uma_loads_unchanged() {
        use std::cell::Cell;
        let built = Cell::new(false);
        let facts = || -> Result<LoadFacts<'static>, String> {
            built.set(true);
            Err("granule facts must not be built".to_string())
        };
        for load in [
            LoadShape {
                unified_memory: true,
                ..shape()
            },
            LoadShape {
                explicit_vram_layers: true,
                ..shape()
            },
        ] {
            let outcome =
                select_load_profile(SlowlinkMode::Force, &load, || unreachable!("probed"), facts);
            assert_eq!(outcome, Ok(None));
        }
        for link in [HostLink::Unified, HostLink::Discrete { gbps: 26.0 }] {
            assert_eq!(
                select_load_profile(SlowlinkMode::Force, &shape(), || link, facts),
                Ok(None)
            );
        }
        assert!(
            !built.get(),
            "no granule facts for a load that cannot select"
        );
        let incumbent = PlacementCharges::new(LAYER, false).unwrap();
        assert_eq!(incumbent.slowlink, 0);
        assert_eq!(incumbent.reserve(123).unwrap(), 123 + incumbent.margin);
        assert_eq!(GranuleInputs::default().granule_bytes, 0);
        assert_eq!(
            GranuleInputs::default().base,
            crate::expert_residency::GranuleSlack::default()
        );
    }

    /// The corrected, owner-derived census on the REAL artifact: the same
    /// base reserve and S1 layouts as the payload-only frontier plus each
    /// owner's allocation-granule slack (weights from the artifact's
    /// model-index, shadows from its 291-owner list, forward and state from
    /// the owners' own allocation lists). Every legal rung at ten resident
    /// layers is then short, each by exactly its payload-only slack less the
    /// granule total; the product tile is NOT changed here. The full
    /// per-tile frontier is emitted for the report.
    #[test]
    fn slowlink_residency_j1778_granule_census_frontier_on_the_real_fixture() {
        use crate::expert_residency as residency;
        // Granule slack of the 1,096 GPU-resident non-expert tensors of the
        // real artifact (536 of them at least one granule), from its
        // model-index; its 1,099 non-expert records also list 3 CPU-only PLE
        // metadata arrays (24 + 128 + 128 = 280 B, all sub-granule, outside
        // the 5,364,055,296 B GPU census), which add no slack.
        const J1778_WEIGHT_GRANULE_SLACK: u64 = 635_740_160;
        let granule = 2 * MIB;
        // The resident non-expert weights packed in ONE 256 B-aligned arena of
        // the artifact (its 1,096 GPU tensors; the worst internal
        // alignment is below the arena's granule tail).
        const J1778_WEIGHT_PACKED_ROOT: u64 = 5_364_514_816;
        // The owners' own allocation lists at `tile`: forward scratch (owning
        // slots + spec buffers), request state (context arenas as the Legacy
        // backend commits them + snapshot mirror) and the kernel workspace
        // terms, one entry per allocation.
        let lists = |tile: usize| {
            let config = crate::config::compact_test_config();
            let context = crate::Qwen4ContextCommit::new(
                crate::Qwen4KvBackend::Legacy,
                J1778_MAX_SEQ,
                J1778_MAX_SEQ,
                1,
            );
            let forward = crate::gpu_forward::qwen4_forward_allocation_sizes(
                &config,
                tile,
                crate::gpu_forward::Qwen4ForwardScratchMode::Full,
            )
            .unwrap();
            let state = crate::state::Qwen4State::device_allocation_sizes(
                &config,
                j1778_state_format(),
                &context,
            )
            .unwrap();
            let workspace: Vec<u64> = j1778_reserve_at(tile)
                .census
                .workspace
                .iter()
                .map(|term| term.1)
                .filter(|bytes| *bytes > 0)
                .collect();
            (forward, state, workspace)
        };
        let base_slack = |tile: usize| {
            let config = crate::config::compact_test_config();
            let reserve = j1778_reserve_at(tile);
            let (forward, state, workspace) = lists(tile);
            residency::GranuleSlack {
                weights: J1778_WEIGHT_GRANULE_SLACK,
                f16_shadows: residency::f16_shadow_granule_slack(
                    &residency::j1778_bf16_manifest(),
                    config.num_hidden_layers,
                    granule,
                )
                .unwrap(),
                forward: residency::granule_slack_total(forward, granule).unwrap(),
                state: residency::granule_slack_total(state, granule).unwrap(),
                workspace: residency::granule_slack_total(workspace, granule).unwrap(),
                // Hold 1 runs no native MTP head (the CPU claim for a kept
                // head is `base_granule_slack`'s `native_mtp` list).
                native_mtp: 0,
                // Legacy backend: the gather workspace is one allocation and
                // the selector scratch another; both are 0 on this fixture.
                qsa_workspace: residency::granule_slack_total(
                    [reserve.census.gather, reserve.census.qsa_selector],
                    granule,
                )
                .unwrap(),
            }
        };
        let boundary_stage_slack = residency::granule_slack_total(
            crate::state::Qwen4State::session_fixed_part_sizes(
                &crate::config::compact_test_config(),
                j1778_state_format(),
            )
            .unwrap(),
            granule,
        )
        .unwrap();
        let mut rows = Vec::new();
        for (route, purpose, payload_slack) in [
            (
                WindowRoute::Aggregate,
                AdmissionPurpose::Product,
                J1778_AGG_PRODUCT_SLACK,
            ),
            (
                WindowRoute::Aggregate,
                AdmissionPurpose::PrimitiveSelftest,
                J1778_AGG_PRIMITIVE_SLACK,
            ),
        ] {
            let payload_only = j1778_inputs(route, purpose, J1778_PRELOAD_FREE);
            let inputs = AdmissionInputs {
                granule: GranuleInputs {
                    granule_bytes: granule,
                    base: base_slack(purpose_tile_rows(purpose)),
                    boundary_stage_slack,
                },
                ..payload_only
            };
            let points = tile_frontier(&inputs, |tile| {
                Ok((j1778_reserve_at(tile), base_slack(tile)))
            })
            .unwrap();
            assert_eq!(points.len(), 5);
            for (index, point) in points.iter().enumerate() {
                let total = point.granule.total().unwrap();
                // Owned bytes the payload census omitted: more than the
                // weights and shadows alone.
                assert!(
                    total > J1778_WEIGHT_GRANULE_SLACK + 387_186_688,
                    "{} {total}",
                    point.tile_rows
                );
                assert_eq!(
                    point.slack_for(10),
                    payload_slack[index] - i128::from(total) - 10 * i128::from(point.granule.layer),
                    "{route:?} {purpose:?} tile {}",
                    point.tile_rows
                );
                assert!(
                    !point.budget.fits(S1_PREFERRED_RESIDENT_LAYERS),
                    "{route:?} {purpose:?} tile {} must not fit the preferred ten on individual allocations",
                    point.tile_rows
                );
                rows.push(serde_json::json!({
                    "route": format!("{route:?}"),
                    "purpose": format!("{purpose:?}"),
                    "tile_rows": point.tile_rows,
                    "granule_terms": point
                        .granule
                        .terms()
                        .iter()
                        .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                        .collect::<Vec<_>>(),
                    "granule_total_bytes": total,
                    "payload_only_slack_at_ten_bytes": payload_slack[index] as i64,
                    "slack_at_ten_bytes": i64::try_from(point.slack_for(10)).unwrap(),
                    "resident_layers": point.resident_layers,
                    "selected": point.selected,
                }));
            }
            // The load itself keeps the largest count that fits and reports the
            // exact preferred deficit.
            let tile = purpose_tile_rows(purpose);
            let inputs_layer_slack = census_of(&inputs)
                .map(|(pages, arena, census)| {
                    granule_charge(&inputs, &pages, &arena, &census)
                        .unwrap()
                        .layer
                })
                .unwrap();
            let reserve = inputs.base_reserve
                + census_of(&inputs).unwrap().2.total().unwrap()
                + {
                    let (pages, arena, census) = census_of(&inputs).unwrap();
                    granule_charge(&inputs, &pages, &arena, &census)
                        .unwrap()
                        .total()
                        .unwrap()
                }
                + J1778_MARGIN;
            let profile = admit(&inputs).unwrap_or_else(|decline| panic!("tile {tile}: {decline}"));
            assert_eq!(profile.placement.reserve_bytes, reserve, "tile {tile}");
            assert!(
                profile.resident_layers < S1_PREFERRED_RESIDENT_LAYERS,
                "tile {tile}"
            );
            assert_eq!(
                profile.placement.preferred_deficit_bytes(),
                10 * (LAYER + inputs_layer_slack) + J1778_NON_EXPERT + reserve - J1778_PRELOAD_FREE,
                "tile {tile}"
            );
        }
        // The exact CPU backing model at the product tile: each owner group's
        // payload, its physical size as individual allocations and as ONE
        // 256 B-aligned arena of that lifetime group (rounded once). The
        // weights and shadows rows use the artifact's own lists (constants
        // from the model-index, shadows computed); forward, state and kernel
        // workspace use the owners' allocation lists above.
        let product = purpose_tile_rows(AdmissionPurpose::Product);
        let (forward, state, workspace) = lists(product);
        let pack = |sizes: &[u64]| -> (u64, u64, u64) {
            let payload: u64 = sizes.iter().sum();
            let individual: u64 = sizes
                .iter()
                .map(|bytes| bytes + residency::granule_slack(*bytes, granule))
                .sum();
            let aligned: u64 = sizes.iter().map(|bytes| bytes.next_multiple_of(256)).sum();
            (payload, individual, aligned.next_multiple_of(granule))
        };
        let mut backing = Vec::new();
        for (name, (payload, individual, packed)) in [
            ("forward scratch + spec buffers", pack(&forward)),
            ("request state + snapshot mirror", pack(&state)),
            ("kernel workspace terms", pack(&workspace)),
            (
                "F16 shadows (291 owners, 256 B multiples)",
                (
                    J1778_SHADOWS,
                    J1778_SHADOWS + 387_186_688,
                    J1778_SHADOWS.next_multiple_of(granule),
                ),
            ),
            (
                "resident non-expert weights (model-index)",
                (
                    J1778_NON_EXPERT,
                    J1778_NON_EXPERT + J1778_WEIGHT_GRANULE_SLACK,
                    J1778_WEIGHT_PACKED_ROOT,
                ),
            ),
        ] {
            assert!(packed <= individual, "{name}: an arena never costs more");
            backing.push(serde_json::json!({
                "owner": name,
                "payload_bytes": payload,
                "individual_physical_bytes": individual,
                "packed_physical_bytes": packed,
                "recovered_vs_individual_bytes": individual - packed,
                "recovered_credit_vs_census_bytes": i128::from(individual) - i128::from(packed),
            }));
        }
        println!(
            "SLOWLINK_GRANULE_BACKING {}",
            serde_json::to_string(&backing).unwrap()
        );
        println!(
            "SLOWLINK_GRANULE_FRONTIER {}",
            serde_json::to_string(&rows).unwrap()
        );
    }

    // The real j1811 hold: the outer gate sampled 25,713,180,672 B free before
    // the load; the admission's own fresh sample after the load's host-link and
    // copy-path initialization read 25,560,088,576 B.
    const J1811_PRE_LOAD_FREE: u64 = 25_713_180_672;
    const J1811_POST_INIT_FREE: u64 = 25_560_088_576;
    const J1811_OBSERVED_INITIALIZATION: u64 = 153_092_096;

    /// One `SLOWLINK_ADMISSION_QUOTE_R5` record of `inputs` at `tile_rows`.
    /// Every figure is read from the SAME [`project_at_tile`] quote that
    /// [`admit_at_tile`] and [`tile_frontier`] judge on; nothing is
    /// recomputed by hand.
    fn quote_r5(
        observation: &str,
        inputs: &AdmissionInputs,
        tile_rows: usize,
    ) -> serde_json::Value {
        let signed = |value: i128| i64::try_from(value).expect("a byte figure fits i64");
        let unsigned = |value: u128| u64::try_from(value).expect("a byte figure fits u64");
        let terms = |terms: &[(&str, u64)]| {
            terms
                .iter()
                .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                .collect::<Vec<_>>()
        };
        let projection = project_at_tile(inputs, tile_rows).expect("the quote projects");
        let budget = projection.budget;
        let verdict = projection.verdict();
        let profile = admit_at_tile(inputs, tile_rows);
        if let Ok(profile) = &profile {
            assert_eq!(
                profile.placement, budget,
                "the profile carries the one quote"
            );
            assert_eq!(Ok(profile.resident_layers), verdict);
        }
        let memory = budget.memory;
        let selected = projection.largest_layers;
        let census = &projection.census;
        serde_json::json!({
            "schema": "SLOWLINK_ADMISSION_QUOTE_R5",
            "observation": observation,
            "purpose": format!("{:?}", inputs.purpose),
            "tile_rows": tile_rows,
            "memory_basis": "post_initialization_actual",
            "initialization_charged_via_free": true,
            "memory": {
                "pre_init_free_bytes": memory.pre_init_free_bytes,
                "free_bytes": memory.free_bytes,
                "observed_initialization_bytes": memory.observed_initialization_bytes(),
                "free_gained_after_initialization_bytes": memory.free_gained_after_initialization_bytes(),
            },
            "non_expert_bytes": budget.non_expert_bytes,
            "layer_payload_bytes": inputs.layer_expert_bytes,
            "layer_granule_slack_bytes": projection.granule.layer,
            "resident_layer_bytes": budget.resident_layer_bytes,
            "reserve_bytes": budget.reserve_bytes,
            "trunk_layers": budget.trunk_layers,
            "preferred_resident_layers": S1_PREFERRED_RESIDENT_LAYERS,
            "selected_layers": selected,
            "verdict": match &verdict {
                Ok(layers) => format!("admit {layers}"),
                Err(decline) => decline.to_string(),
            },
            "needed_bytes": selected.map(|layers| unsigned(budget.needed_bytes_for(layers))),
            "slack_bytes": selected.map(|layers| signed(budget.slack_bytes_at(layers))),
            "next_layer_needed_bytes": selected.map(|layers| unsigned(budget.needed_bytes_for(layers + 1))),
            "next_layer_slack_bytes": selected.map(|layers| signed(budget.slack_bytes_at(layers + 1))),
            "base_deficit_bytes": budget.base_deficit_bytes(),
            "preferred_needed_bytes": unsigned(budget.needed_bytes_for(S1_PREFERRED_RESIDENT_LAYERS)),
            "preferred_slack_bytes": signed(budget.slack_bytes_at(S1_PREFERRED_RESIDENT_LAYERS)),
            "preferred_deficit_bytes": budget.preferred_deficit_bytes(),
            // The pre-initialization-domain equation: the measured interval
            // loss plus the slack at the selected count. Initialization is
            // charged ONCE, by the lower actual free; this is not a second charge.
            "pre_init_domain_slack_at_selected_bytes": memory
                .pre_init_free_bytes
                .zip(selected)
                .map(|(before, layers)| signed(
                    i128::from(before) - budget.needed_bytes_for(layers) as i128
                )),
            "device_snapshot_budget_bytes": profile.as_ref().ok().map(|p| p.device_snapshot_budget_bytes),
            "owner_terms": {
                "base_reserve_bytes": inputs.base_reserve,
                "safety_margin_bytes": projection.charges.margin,
                "s1_owned_terms": terms(&census.owned_terms()),
                "s1_owned_total_bytes": census.owned_total().unwrap(),
                "runtime_budget_bytes": census.runtime_overhead,
                "diagnostic_budget_bytes": census.diagnostic,
                "s1_census_total_bytes": census.total().unwrap(),
                "granule_terms": terms(&projection.granule.terms()),
                "granule_total_bytes": projection.granule.total().unwrap(),
                "hc_arena_bytes": projection.arena.total_bytes,
                "page_count": projection.pages.total_pages(),
            },
        })
    }

    /// The pre-initialization sample is an observation: it exposes the
    /// measured interval loss (146 MiB on j1811, an interval of THAT run, not
    /// a constant) and never reaches the placement arithmetic. A free that rose
    /// after the sample leaves the actual figure authoritative.
    #[test]
    fn slowlink_admission_memory_is_the_measured_interval_never_a_tax() {
        let j1811 = AdmissionMemory::measured(J1811_PRE_LOAD_FREE, J1811_POST_INIT_FREE);
        assert_eq!(J1811_PRE_LOAD_FREE, J1778_PRELOAD_FREE);
        assert_eq!(
            J1811_PRE_LOAD_FREE - J1811_POST_INIT_FREE,
            J1811_OBSERVED_INITIALIZATION
        );
        assert_eq!(J1811_OBSERVED_INITIALIZATION, 146 * MIB);
        assert_eq!(
            j1811.observed_initialization_bytes(),
            Some(J1811_OBSERVED_INITIALIZATION)
        );
        assert_eq!(j1811.free_gained_after_initialization_bytes(), Some(0));
        let rose = AdmissionMemory::measured(100, 150);
        assert_eq!(rose.observed_initialization_bytes(), Some(0));
        assert_eq!(rose.free_gained_after_initialization_bytes(), Some(50));
        assert_eq!(
            AdmissionMemory::measured(7, 7).observed_initialization_bytes(),
            Some(0)
        );
        let unknown = AdmissionMemory::actual(150);
        assert_eq!(unknown.pre_init_free_bytes, None);
        assert_eq!(unknown.observed_initialization_bytes(), None);
        assert_eq!(unknown.free_gained_after_initialization_bytes(), None);

        // Placement reads the actual free only, whatever the earlier sample.
        let base = inputs(free_for(&inputs(0), 12));
        let expect = project_at_tile(&base, S1_PRODUCT_TILE_ROWS).unwrap();
        assert_eq!(expect.largest_layers, Some(12));
        for memory in [
            AdmissionMemory::measured(base.memory.free_bytes + 7 * GIB, base.memory.free_bytes),
            AdmissionMemory::measured(1, base.memory.free_bytes),
            AdmissionMemory::measured(base.memory.free_bytes, base.memory.free_bytes),
        ] {
            let projection =
                project_at_tile(&AdmissionInputs { memory, ..base }, S1_PRODUCT_TILE_ROWS).unwrap();
            assert_eq!(projection.largest_layers, expect.largest_layers);
            assert_eq!(
                projection.budget.slack_bytes_at(12),
                expect.budget.slack_bytes_at(12)
            );
            assert_eq!(
                projection.budget.needed_bytes_for(12),
                expect.budget.needed_bytes_for(12)
            );
        }
    }

    /// The whole joint reserve is charged once: base + safety margin + S1
    /// census + granule slack, with the runtime budget the exact
    /// `max(base runtime, 1 GiB)` and no other term (no device cache cap, no
    /// host snapshot charge): the census stays its nine terms and the backed
    /// layer cost is the payload plus its own slack.
    #[test]
    fn slowlink_admission_reserve_is_charged_once_with_the_runtime_max_and_no_other_term() {
        for base_runtime in [0, 512 * MIB, GIB - 1, GIB, GIB + 1, 2 * GIB] {
            for granule in [false, true] {
                let probe = AdmissionInputs {
                    base_runtime_reserved_bytes: base_runtime,
                    ..inputs(free_for(&inputs(0), 11))
                };
                let probe = if granule { with_granule(probe) } else { probe };
                let projection = project_at_tile(&probe, S1_PRODUCT_TILE_ROWS).unwrap();
                let census = &projection.census;
                assert_eq!(census.terms().len(), 9);
                assert_eq!(
                    census.runtime_overhead,
                    S1_RUNTIME_OVERHEAD_BYTES.saturating_sub(base_runtime)
                );
                assert_eq!(
                    base_runtime + census.runtime_overhead,
                    base_runtime.max(S1_RUNTIME_OVERHEAD_BYTES)
                );
                assert_eq!(projection.charges.margin, EXPERT_MARGIN);
                assert_eq!(
                    projection.budget.reserve_bytes,
                    probe.base_reserve
                        + EXPERT_MARGIN
                        + census.total().unwrap()
                        + projection.granule.total().unwrap()
                );
                assert_eq!(projection.budget.non_expert_bytes, probe.non_expert);
                assert_eq!(
                    projection.budget.resident_layer_bytes,
                    LAYER + projection.granule.layer
                );
                assert_eq!(granule, projection.granule.total().unwrap() > 0);
            }
        }
    }

    /// Exact and one-byte boundaries at the base, N0, N1, N9, N10, N11 and the
    /// all-resident bypass on the one quote. There is no layer floor: a base
    /// that fits keeps zero resident layers; only a base that does not fit is
    /// a deficit.
    #[test]
    fn slowlink_admission_boundaries_are_exact_at_the_base_n0_n9_n10_and_all_resident() {
        let probe = inputs(0);
        let budget = project_at_tile(&probe, S1_PRODUCT_TILE_ROWS)
            .unwrap()
            .budget;
        let need = |layers: usize| u64::try_from(budget.needed_bytes_for(layers)).unwrap();
        let admit_free = |free: u64| {
            admit(&AdmissionInputs {
                memory: AdmissionMemory::actual(free),
                ..probe
            })
        };
        assert_eq!(budget.resident_layer_bytes, LAYER);

        // N0: the base fits exactly: a valid admission with every routed layer
        // host-streamed.
        let zero = admit_free(need(0)).expect("a base that fits keeps zero resident layers");
        assert_eq!(zero.resident_layers, 0);
        assert_eq!(zero.device_snapshot_budget_bytes, 0);
        assert_eq!(zero.placement.largest_fitting_layers(), Some(0));
        assert_eq!(zero.placement.preferred_deficit_bytes(), 10 * LAYER);
        // One byte less: the BASE does not fit, an exact deficit, never N0.
        let Err(SlowlinkDecline::Deficit(base)) = admit_free(need(0) - 1) else {
            panic!("the base must not fit one byte short");
        };
        assert_eq!(base.deficit_bytes, 1);
        assert_eq!(base.required_layers, 0);
        assert_eq!(base.memory.free_bytes, need(0) - 1);
        assert_eq!(base.non_expert, probe.non_expert);
        assert_eq!(base.reserve, budget.reserve_bytes);
        assert_eq!(base.resident_layer_bytes, LAYER);
        assert!(SlowlinkDecline::Deficit(base).is_capacity());
        assert!(SlowlinkDecline::Deficit(base)
            .to_string()
            .contains("short by 1 bytes"));
        let Err(SlowlinkDecline::Deficit(empty)) = admit_free(0) else {
            panic!("no free memory is a base deficit");
        };
        assert_eq!(empty.deficit_bytes, need(0));

        for layers in [1usize, 9, 10, 11, 47] {
            let exact = admit_free(need(layers)).unwrap();
            assert_eq!(exact.resident_layers, layers);
            assert_eq!(exact.device_snapshot_budget_bytes, 0);
            assert_eq!(
                exact.placement.preferred_deficit_bytes(),
                need(S1_PREFERRED_RESIDENT_LAYERS).saturating_sub(need(layers))
            );
            let under = admit_free(need(layers) - 1).unwrap();
            assert_eq!(under.resident_layers, layers - 1);
            assert_eq!(under.placement.slack_bytes_at(layers), -1);
            // The next layer needs one full backed layer more.
            let almost = admit_free(need(layers + 1) - 1).unwrap();
            assert_eq!(almost.resident_layers, layers);
            assert_eq!(almost.device_snapshot_budget_bytes, LAYER - 1);
        }
        // Every routed layer fits: the incumbent bypass, never an S1 claim.
        assert_eq!(
            admit_free(need(48)),
            Err(SlowlinkDecline::NoHostLayers {
                resident_layers: 48
            })
        );
        assert_eq!(
            admit_free(need(48) + 50 * GIB),
            Err(SlowlinkDecline::NoHostLayers {
                resident_layers: 48
            })
        );
    }

    /// `project_at_tile`, `admit_at_tile`, `admit` and `tile_frontier` are one
    /// quote: for the same actual memory they agree byte for byte at every
    /// rung, both purposes and routes, whether the earlier sample is unknown,
    /// equal to the actual or the real j1811 measured interval.
    #[test]
    fn slowlink_admission_project_admit_and_frontier_are_one_quote() {
        for route in [WindowRoute::Aggregate, WindowRoute::SlabIu4] {
            for purpose in [
                AdmissionPurpose::Product,
                AdmissionPurpose::PrimitiveSelftest,
            ] {
                for memory in [
                    AdmissionMemory::actual(J1811_PRE_LOAD_FREE),
                    AdmissionMemory::measured(J1811_PRE_LOAD_FREE, J1811_PRE_LOAD_FREE),
                    AdmissionMemory::measured(J1811_PRE_LOAD_FREE, J1811_POST_INIT_FREE),
                ] {
                    let inputs = AdmissionInputs {
                        memory,
                        ..j1778_inputs(route, purpose, J1811_PRE_LOAD_FREE)
                    };
                    let points = tile_frontier(&inputs, |tile| {
                        Ok((
                            j1778_reserve_at(tile),
                            crate::expert_residency::GranuleSlack::default(),
                        ))
                    })
                    .unwrap();
                    assert_eq!(points.len(), legal_tile_rungs().len());
                    for point in &points {
                        let at = AdmissionInputs {
                            memory,
                            ..j1778_inputs_at(route, purpose, J1811_PRE_LOAD_FREE, point.tile_rows)
                        };
                        let label =
                            format!("{route:?} {purpose:?} {memory:?} tile {}", point.tile_rows);
                        let projection = project_at_tile(&at, point.tile_rows).unwrap();
                        assert_eq!(point.budget, projection.budget, "{label}");
                        assert_eq!(point.census, projection.census, "{label}");
                        assert_eq!(point.granule, projection.granule, "{label}");
                        assert_eq!(point.reserve, projection.budget.reserve_bytes, "{label}");
                        assert_eq!(
                            point.resident_layers,
                            projection.resident_layers(),
                            "{label}"
                        );
                        assert_eq!(point.verdict, projection.verdict().map(|_| ()), "{label}");
                        assert_eq!(
                            point.needed_bytes_for(10),
                            projection.budget.needed_bytes_for(10),
                            "{label}"
                        );
                        match admit_at_tile(&at, point.tile_rows) {
                            Ok(profile) => {
                                assert_eq!(
                                    Ok(profile.resident_layers),
                                    projection.verdict(),
                                    "{label}"
                                );
                                assert_eq!(profile.placement, projection.budget, "{label}");
                                assert_eq!(profile.census, projection.census, "{label}");
                                assert_eq!(
                                    i128::from(profile.device_snapshot_budget_bytes),
                                    projection.budget.slack_bytes_at(profile.resident_layers),
                                    "{label}"
                                );
                            }
                            Err(decline) => {
                                assert_eq!(Err(decline), projection.verdict(), "{label}")
                            }
                        }
                        if point.selected {
                            assert_eq!(point.verdict, admit(&inputs).map(|_| ()), "{label}");
                        }
                    }
                }
            }
        }
    }

    /// The bounded two-family weight arenas on the REAL Flash-Next artifact
    /// metadata (`rdna-compute`'s `jit-g0/header-flash-next.json`, header
    /// SHA-256 8703bb16…, the artifact of the j1778/j1786 holds) and the closed
    /// 291-owner shadow fixture. No product count is hard-coded where the
    /// metadata derives it; the independent reference model below is plain
    /// next-fit over the canonical bytewise-name order.
    mod slowlink_arena {
        use super::*;
        use crate::expert_residency as residency;
        use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
        use rdna_compute::byte_arena::{ModelWeightArenaPlan, NamedByteArenaPlan};
        use rdna_compute::DType;
        use std::collections::HashMap;

        const ALIAS_NAME: &str = "model.arena_fixture.tied_alias.weight";

        /// The real manifest's roles, with an independent expectation list.
        struct Fixture {
            weights: Vec<WeightEntry>,
            sizes: HashMap<String, u64>,
            /// Name and bytes of every resident non-expert tensor, read straight
            /// from the header (not through the code under test).
            resident: Vec<(String, u64)>,
            external: Vec<String>,
            experts: Vec<String>,
            cpu_metadata: Vec<(String, u64)>,
        }

        impl Fixture {
            fn bytes_of(&self) -> impl Fn(&WeightEntry) -> Option<u64> + '_ {
                |entry| self.sizes.get(&entry.name).copied()
            }

            fn largest(&self) -> u64 {
                self.resident.iter().map(|member| member.1).max().unwrap()
            }
        }

        fn dtype_of(tag: u64) -> DType {
            match tag {
                3 => DType::Q8_0,
                16 => DType::BF16,
                44 => DType::MQ4G256V2,
                47 => DType::MQ6G256V2,
                53 => DType::MQ4G128V2,
                other => panic!("header fixture quant tag {other} has no test mapping"),
            }
        }

        fn fixture() -> Fixture {
            let header: serde_json::Value = serde_json::from_str(include_str!(
                "../../rdna-compute/tests/fixtures/jit-g0/header-flash-next.json"
            ))
            .unwrap();
            assert_eq!(
                header["header_sha256"],
                "8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013"
            );
            let mut fixture = Fixture {
                weights: Vec::new(),
                sizes: HashMap::new(),
                resident: Vec::new(),
                external: Vec::new(),
                experts: Vec::new(),
                cpu_metadata: Vec::new(),
            };
            for tensor in header["tensors"].as_array().unwrap() {
                let name = tensor["name"].as_str().unwrap().to_owned();
                let tag = tensor["quant_tag"].as_u64().unwrap();
                let bytes = tensor["data_size"].as_u64().unwrap();
                let shape: Vec<usize> = tensor["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|dim| dim.as_u64().unwrap() as usize)
                    .collect();
                if tag == 54 {
                    // The exact I64 PLE records: CPU metadata, never a
                    // `WeightEntry` (see `MetadataTensor`).
                    fixture.cpu_metadata.push((name, bytes));
                    continue;
                }
                let mut entry = WeightEntry::model(
                    name.clone(),
                    shape.clone(),
                    dtype_of(tag),
                    ShardPolicy::Replicate,
                );
                entry.layer = name
                    .strip_prefix("model.language_model.layers.")
                    .and_then(|rest| rest.split_once('.'))
                    .and_then(|(layer, _)| layer.parse().ok());
                let expert = name.ends_with(".mlp.experts.gate_up_proj")
                    || name.ends_with(".mlp.experts.down_proj");
                if name.contains(".ngram_embedding.shard_") {
                    entry = entry.external_rows(shape[1] / 32 * 34, shape[0]);
                    fixture.external.push(name.clone());
                } else if expert {
                    fixture.experts.push(name.clone());
                } else {
                    fixture.resident.push((name.clone(), bytes));
                }
                fixture.sizes.insert(name, bytes);
                fixture.weights.push(entry);
            }
            // A tied alias owns no payload; give it a size above every real
            // tensor so any leak into the cap or the roots is visible.
            let alias_bytes = fixture.largest() * 2;
            fixture.weights.push(WeightEntry::model(
                ALIAS_NAME,
                vec![248_320, 2_560],
                DType::Q8_0,
                ShardPolicy::Tied {
                    source: "model.language_model.embed_tokens.weight".to_string(),
                },
            ));
            fixture.sizes.insert(ALIAS_NAME.to_string(), alias_bytes);
            fixture
        }

        fn plan_of(fixture: &Fixture, granule: usize) -> ModelWeightArenaPlan {
            let layers = crate::config::compact_test_config().num_hidden_layers;
            let plan = residency::model_weight_arena_plan(
                &fixture.weights,
                fixture.bytes_of(),
                layers,
                true,
                granule,
            )
            .unwrap();
            ModelWeightArenaPlan::clone(&plan)
        }

        /// Plain next-fit reference over the canonical bytewise-name order:
        /// 256 B member alignment, a member that would cross `cap` opens a new
        /// root, a root request is its end offset (at least 256).
        fn reference_roots(members: &[(String, u64)], cap: u64) -> Vec<u64> {
            let mut sorted: Vec<&(String, u64)> = members.iter().collect();
            sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
            let mut roots = Vec::new();
            let (mut end, mut count) = (0u64, 0usize);
            for (_, bytes) in sorted {
                if count > 0 && end.next_multiple_of(256) + bytes > cap {
                    roots.push(end.max(256));
                    end = 0;
                    count = 0;
                }
                end = end.next_multiple_of(256) + bytes;
                count += 1;
            }
            if count > 0 {
                roots.push(end.max(256));
            }
            roots
        }

        fn shadow_members(fixture: &Fixture) -> Vec<(String, u64)> {
            let layers = crate::config::compact_test_config().num_hidden_layers;
            residency::f16_shadow_arena_members(&fixture.weights, layers)
                .unwrap()
                .into_iter()
                .map(|(name, bytes)| (name, bytes as u64))
                .collect()
        }

        fn assert_family(plan: &NamedByteArenaPlan, expected: &[(String, u64)], cap: u64) {
            assert_eq!(plan.alignment, 256);
            assert_eq!(plan.max_root_bytes as u64, cap);
            assert_eq!(plan.members.len(), expected.len());
            assert!(plan
                .members
                .windows(2)
                .all(|pair| pair[0].name.as_bytes() < pair[1].name.as_bytes()));
            let sizes: HashMap<&str, u64> = expected
                .iter()
                .map(|(name, bytes)| (name.as_str(), *bytes))
                .collect();
            let mut previous: Option<(usize, usize)> = None;
            for span in &plan.members {
                assert_eq!(
                    span.bytes as u64,
                    sizes[span.name.as_str()],
                    "{}",
                    span.name
                );
                assert_eq!(span.offset % 256, 0, "{}", span.name);
                assert!(
                    span.offset + span.bytes <= plan.roots[span.root],
                    "{}",
                    span.name
                );
                if let Some((root, end)) = previous.filter(|(root, _)| *root == span.root) {
                    assert_eq!(root, span.root);
                    assert!(span.offset >= end, "{} overlaps its neighbour", span.name);
                }
                previous = Some((span.root, span.offset + span.bytes));
            }
            let requests: Vec<u64> = plan.roots.iter().map(|root| *root as u64).collect();
            assert_eq!(requests, reference_roots(expected, cap));
            assert!(requests.iter().all(|root| (256..=cap).contains(root)));
            assert_eq!(
                plan.logical_bytes().unwrap(),
                expected.iter().map(|member| member.1).sum::<u64>()
            );
        }

        /// The canonical 9 + 2 roots of the real artifact, each request the
        /// reference next-fit's, the cap the largest existing eligible
        /// allocation, and the closed shadow fixture's owners equal the
        /// header-derived shadow members.
        #[test]
        fn slowlink_arena_real_metadata_canonical_roots_and_requests() {
            let fixture = fixture();
            let granule = 2 * MIB as usize;
            let plan = plan_of(&fixture, granule);
            plan.validate().unwrap();
            assert_eq!(plan.granule_bytes, granule);

            let cap = fixture.largest();
            assert_eq!(
                cap, fixture.sizes["model.language_model.embed_tokens.weight"],
                "the token embedding is the largest existing eligible allocation"
            );
            assert!(
                fixture.sizes[ALIAS_NAME] > cap,
                "the alias must not set the cap"
            );
            assert_family(&plan.weights, &fixture.resident, cap);
            let shadows = shadow_members(&fixture);
            assert_family(&plan.f16_shadows, &shadows, cap);

            assert_eq!(plan.weights.roots.len(), 9);
            assert_eq!(plan.f16_shadows.roots.len(), 2);
            assert_eq!(plan.weights.logical_bytes().unwrap(), J1778_NON_EXPERT);
            assert_eq!(plan.f16_shadows.logical_bytes().unwrap(), J1778_SHADOWS);

            // The closed 291-owner fixture names exactly these shadows.
            let tsv: Vec<(String, usize)> = residency::f16_shadow_arena_members(
                &residency::j1778_bf16_manifest(),
                crate::config::compact_test_config().num_hidden_layers,
            )
            .unwrap();
            let planned: Vec<(String, usize)> = plan
                .f16_shadows
                .members
                .iter()
                .map(|span| (span.name.clone(), span.bytes))
                .collect();
            let mut tsv_sorted = tsv;
            tsv_sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
            assert_eq!(planned, tsv_sorted);
            assert_eq!(
                residency::f16_shadow_bytes(&fixture.weights, 48),
                Ok(J1778_SHADOWS)
            );
        }

        /// Per root, the census backing equals the backing of the exact
        /// allocator capacity; an oversized reused buffer or a wrong root
        /// count is refused instead of being treated as equal.
        #[test]
        fn slowlink_arena_per_root_census_backing_equals_actual_exact_capacity() {
            let fixture = fixture();
            let granule = 2 * MIB as usize;
            let plan = plan_of(&fixture, granule);
            for (family, roots_expected) in [(&plan.weights, 9usize), (&plan.f16_shadows, 2)] {
                assert_eq!(family.roots.len(), roots_expected);
                let census = family.backing_bytes(granule).unwrap();
                let by_hand: u64 = family
                    .roots
                    .iter()
                    .map(|root| (*root as u64).next_multiple_of(granule as u64))
                    .sum();
                assert_eq!(census, by_hand);
                // Exact capacity model: the allocator returns what was asked.
                let exact: Vec<usize> = family.roots.clone();
                assert_eq!(family.validate_capacities(&exact, granule), Ok(census));
                for (index, root) in family.roots.iter().enumerate() {
                    let mut actual = exact.clone();
                    actual[index] = *root + 1;
                    assert!(
                        family.validate_capacities(&actual, granule).is_err(),
                        "oversized root {index} must be rejected"
                    );
                    actual[index] = root - 1;
                    assert!(family.validate_capacities(&actual, granule).is_err());
                }
                assert!(family
                    .validate_capacities(&exact[..exact.len() - 1], granule)
                    .is_err());
                assert!(family.backing_bytes(0).is_err());
            }
            assert_eq!(plan.weights.backing_bytes(granule), Ok(5_375_000_576));
            assert_eq!(plan.f16_shadows.backing_bytes(granule), Ok(1_300_234_240));
        }

        fn base_slack_at(tile: usize, weights: u64, f16_shadows: u64) -> residency::GranuleSlack {
            let granule = 2 * MIB;
            let config = crate::config::compact_test_config();
            let reserve = j1778_reserve_at(tile);
            let context = crate::Qwen4ContextCommit::new(
                crate::Qwen4KvBackend::Legacy,
                J1778_MAX_SEQ,
                J1778_MAX_SEQ,
                1,
            );
            let forward = crate::gpu_forward::qwen4_forward_allocation_sizes(
                &config,
                tile,
                crate::gpu_forward::Qwen4ForwardScratchMode::Full,
            )
            .unwrap();
            let state = crate::state::Qwen4State::device_allocation_sizes(
                &config,
                j1778_state_format(),
                &context,
            )
            .unwrap();
            let workspace: Vec<u64> = reserve
                .census
                .workspace
                .iter()
                .map(|term| term.1)
                .filter(|bytes| *bytes > 0)
                .collect();
            residency::GranuleSlack {
                weights,
                f16_shadows,
                forward: residency::granule_slack_total(forward, granule).unwrap(),
                state: residency::granule_slack_total(state, granule).unwrap(),
                workspace: residency::granule_slack_total(workspace, granule).unwrap(),
                native_mtp: 0,
                qsa_workspace: residency::granule_slack_total(
                    [reserve.census.gather, reserve.census.qsa_selector],
                    granule,
                )
                .unwrap(),
            }
        }

        /// The real j1786 inputs (aggregate route) at every rung of a purpose
        /// with the given weight and shadow slack, as the loader's admission
        /// reads them.
        fn frontier(
            purpose: AdmissionPurpose,
            weights: u64,
            f16_shadows: u64,
        ) -> (AdmissionInputs, Vec<FrontierPoint>) {
            let granule = 2 * MIB;
            let boundary_stage_slack = residency::granule_slack_total(
                crate::state::Qwen4State::session_fixed_part_sizes(
                    &crate::config::compact_test_config(),
                    j1778_state_format(),
                )
                .unwrap(),
                granule,
            )
            .unwrap();
            let payload_only = j1778_inputs(WindowRoute::Aggregate, purpose, J1778_PRELOAD_FREE);
            let inputs = AdmissionInputs {
                granule: GranuleInputs {
                    granule_bytes: granule,
                    base: base_slack_at(purpose_tile_rows(purpose), weights, f16_shadows),
                    boundary_stage_slack,
                },
                ..payload_only
            };
            let points = tile_frontier(&inputs, |tile| {
                Ok((
                    j1778_reserve_at(tile),
                    base_slack_at(tile, weights, f16_shadows),
                ))
            })
            .unwrap();
            (inputs, points)
        }

        fn selected(points: &[FrontierPoint]) -> &FrontierPoint {
            let mut selected = points.iter().filter(|point| point.selected);
            let point = selected.next().expect("a selected tile");
            assert!(selected.next().is_none());
            point
        }

        /// With the canonical plan's root backing charged, the product 768-row
        /// tile keeps N=10 with exactly 48,509,588 B of slack and the
        /// primitive 512-row tile (217,821,184 B diagnostic) 113,608,212 B,
        /// on the SAME real inputs, the unchanged 1 GiB runtime budget and
        /// 512 MiB safety margin, tiles and N. The historical individual
        /// allocation charge is the named counterfactual: it declines by the
        /// exact deficit, and the two differ by the recovered bytes only.
        #[test]
        fn slowlink_arena_n10_product_768_and_primitive_512_on_real_j1786_inputs() {
            let fixture = fixture();
            let granule = 2 * MIB;
            let layers = crate::config::compact_test_config().num_hidden_layers;
            let plan = plan_of(&fixture, granule as usize);
            let (weights_arena, shadows_arena) = residency::weight_arena_granule_slack(
                &plan,
                &fixture.weights,
                fixture.bytes_of(),
                layers,
                true,
                granule,
            )
            .unwrap();
            assert_eq!((weights_arena, shadows_arena), (10_945_280, 1_310_720));
            // The historical counterfactual: one allocation per owner.
            let weights_individual =
                residency::weight_granule_slack(&fixture.weights, fixture.bytes_of(), granule)
                    .unwrap();
            let shadows_individual =
                residency::f16_shadow_granule_slack(&fixture.weights, layers, granule).unwrap();
            assert_eq!(weights_individual, 635_740_160);
            assert_eq!(shadows_individual, 387_186_688);
            let recovered = weights_individual + shadows_individual - weights_arena - shadows_arena;
            assert_eq!(recovered, 1_010_670_848);

            for (purpose, tile, slack, payload_only) in [
                (
                    AdmissionPurpose::Product,
                    768usize,
                    48_501_852i128,
                    J1778_AGG_PRODUCT_SLACK[3],
                ),
                (
                    AdmissionPurpose::PrimitiveSelftest,
                    512usize,
                    113_600_476i128,
                    J1778_AGG_PRIMITIVE_SLACK[4],
                ),
            ] {
                assert_eq!(purpose_tile_rows(purpose), tile);
                let (arena_inputs, arena_points) = frontier(purpose, weights_arena, shadows_arena);
                let (individual_inputs, individual_points) =
                    frontier(purpose, weights_individual, shadows_individual);
                let point = selected(&arena_points);
                assert_eq!(point.tile_rows, tile);
                assert_eq!(point.slack_for(10), slack, "{purpose:?}");
                assert!(point.admits(), "{purpose:?}");
                assert_eq!(point.census.runtime_overhead, S1_RUNTIME_OVERHEAD_BYTES);
                assert_eq!(point.safety_margin, EXPERT_MARGIN);
                if purpose == AdmissionPurpose::PrimitiveSelftest {
                    assert_eq!(arena_inputs.diagnostic_bytes, J1778_PRIMITIVE_DIAGNOSTIC);
                    assert_eq!(point.census.diagnostic, J1778_PRIMITIVE_DIAGNOSTIC);
                }
                let profile = admit(&arena_inputs).unwrap();
                assert_eq!(profile.tile_rows, tile);
                assert_eq!(profile.resident_layers, S1_PREFERRED_RESIDENT_LAYERS);

                // Same tiles and base reserves, only the two granule terms move.
                assert_eq!(arena_points.len(), individual_points.len());
                for (arena, individual) in arena_points.iter().zip(&individual_points) {
                    assert_eq!(arena.tile_rows, individual.tile_rows);
                    assert_eq!(arena.base_reserve, individual.base_reserve);
                    assert_eq!(arena.census, individual.census);
                    assert_eq!(arena.safety_margin, individual.safety_margin);
                    assert_eq!(
                        individual.slack_for(10) + i128::from(recovered),
                        arena.slack_for(10),
                        "tile {}",
                        arena.tile_rows
                    );
                }

                let counterfactual = selected(&individual_points);
                assert_eq!(
                    counterfactual.slack_for(10),
                    payload_only
                        - i128::from(counterfactual.granule.total().unwrap())
                        - 10 * i128::from(counterfactual.granule.layer),
                    "{purpose:?}: the historical individual frontier is unchanged"
                );
                assert_eq!(counterfactual.slack_for(10), slack - i128::from(recovered));
                assert!(
                    !counterfactual.budget.fits(S1_PREFERRED_RESIDENT_LAYERS),
                    "{purpose:?}"
                );
                let nine = admit(&individual_inputs)
                    .unwrap_or_else(|decline| panic!("{purpose:?}: {decline}"));
                assert_eq!(nine.resident_layers, 9, "{purpose:?}");
                assert_eq!(
                    i128::from(nine.placement.preferred_deficit_bytes()),
                    -counterfactual.slack_for(10),
                    "{purpose:?}"
                );
            }
        }

        /// R5 on the real j1811 sequence and the REAL header-derived owners:
        /// with the outer pre-load free (25,713,180,672 B, the reference) the
        /// product 768 and primitive 512 profiles keep N=10 with 48,509,588 B
        /// and 113,608,212 B of slack; with the admission's own post-
        /// initialization free (25,560,088,576 B, 153,092,096 B = 146 MiB lower
        /// as MEASURED, not assumed) the SAME inputs keep N=9 and the exact
        /// errors are -104,582,508 B and -39,483,884 B at N=10. Emits the fresh
        /// `SLOWLINK_ADMISSION_QUOTE_R5` line of both purposes and both memory
        /// observations from the one quote, with the host owner payload and pad
        /// the real manifest's placement query gives at the selected and
        /// preferred counts.
        #[test]
        fn slowlink_arena_j1811_post_initialization_memory_keeps_nine_and_the_reference_ten() {
            let fixture = fixture();
            let granule = 2 * MIB;
            let config = crate::config::compact_test_config();
            let layers = config.num_hidden_layers;
            let plan = plan_of(&fixture, granule as usize);
            let (weights_arena, shadows_arena) = residency::weight_arena_granule_slack(
                &plan,
                &fixture.weights,
                fixture.bytes_of(),
                layers,
                true,
                granule,
            )
            .unwrap();

            // The REAL owner query: the manifest after `place_routed_experts`
            // and the pinned bytes `host_mapped_bytes` charges (payload plus
            // the per-entry pad), for `resident_layers` resident trunk layers.
            assert_eq!(fixture.experts.len() % 2, 0);
            let expert_layers = fixture.experts.len() / 2;
            assert_eq!(
                expert_layers,
                layers + 1,
                "trunk plus the weight-only MTP layer"
            );
            let host_owner_bytes = |resident_layers: usize| -> (usize, u64, u64) {
                let mut weights = fixture.weights.clone();
                residency::place_routed_experts(&mut weights, resident_layers);
                let entries = weights
                    .iter()
                    .filter(|entry| {
                        entry.residency
                            == hipfire_runtime::weight_manifest::WeightResidency::HostMapped
                    })
                    .count();
                let total = residency::host_mapped_bytes(&weights, fixture.bytes_of()).unwrap();
                let pad = entries as u64 * residency::HOST_MAPPED_PAD_BYTES;
                let host_layers = expert_layers - resident_layers;
                assert_eq!(entries, 2 * host_layers, "{resident_layers} resident");
                assert_eq!(
                    total - pad,
                    host_layers as u64 * LAYER,
                    "{resident_layers} resident"
                );
                (entries, total - pad, pad)
            };
            let host_owners = |resident_layers: usize| {
                let (entries, payload, pad) = host_owner_bytes(resident_layers);
                serde_json::json!({
                    "resident_layers": resident_layers,
                    "host_layers": expert_layers - resident_layers,
                    "host_owner_entries": entries,
                    "host_payload_bytes": payload,
                    "host_owner_pad_bytes": pad,
                    "host_total_bytes": payload + pad,
                })
            };
            // N=10 pins 39 host layers (38 trunk plus the MTP layer); N=9 one more.
            assert_eq!(
                host_owner_bytes(10),
                (78, 52_140_441_600, 81_788_928),
                "the preferred count's real host owners"
            );
            let (_, nine_payload, nine_pad) = host_owner_bytes(9);
            assert_eq!(
                nine_payload + nine_pad - (52_140_441_600 + 81_788_928),
                1_339_031_552
            );
            assert_eq!(1_339_031_552, LAYER + 2 * residency::HOST_MAPPED_PAD_BYTES);

            for (purpose, tile, reference_slack, actual_slack) in [
                (
                    AdmissionPurpose::Product,
                    768usize,
                    48_501_852i128,
                    -104_590_244i128,
                ),
                (
                    AdmissionPurpose::PrimitiveSelftest,
                    512usize,
                    113_600_476i128,
                    -39_491_620i128,
                ),
            ] {
                assert_eq!(purpose_tile_rows(purpose), tile);
                let (base, _) = frontier(purpose, weights_arena, shadows_arena);
                let reference = AdmissionInputs {
                    memory: AdmissionMemory::measured(J1811_PRE_LOAD_FREE, J1811_PRE_LOAD_FREE),
                    ..base
                };
                let actual = AdmissionInputs {
                    memory: AdmissionMemory::measured(J1811_PRE_LOAD_FREE, J1811_POST_INIT_FREE),
                    ..base
                };
                let reference_quote = project_at_tile(&reference, tile).unwrap();
                let actual_quote = project_at_tile(&actual, tile).unwrap();
                // Only the memory differs: one reserve, one census, one layer cost.
                assert_eq!(reference_quote.census, actual_quote.census);
                assert_eq!(reference_quote.granule, actual_quote.granule);
                assert_eq!(
                    reference_quote.budget.reserve_bytes,
                    actual_quote.budget.reserve_bytes
                );
                assert_eq!(
                    reference_quote.budget.resident_layer_bytes,
                    actual_quote.budget.resident_layer_bytes
                );
                assert_eq!(
                    reference_quote
                        .budget
                        .memory
                        .observed_initialization_bytes(),
                    Some(0)
                );
                assert_eq!(
                    actual_quote.budget.memory.observed_initialization_bytes(),
                    Some(J1811_OBSERVED_INITIALIZATION)
                );

                // Reference (pre = post = the outer pre-load free): N=10.
                assert_eq!(reference_quote.largest_layers, Some(10), "{purpose:?}");
                assert_eq!(reference_quote.budget.slack_bytes_at(10), reference_slack);
                assert_eq!(admit(&reference).unwrap().resident_layers, 10);

                // Actual post-initialization free: the SAME inputs keep N=9 and
                // the N=10 error is exactly the measured interval below the
                // reference slack.
                assert_eq!(actual_quote.budget.slack_bytes_at(10), actual_slack);
                assert_eq!(
                    actual_slack,
                    reference_slack - i128::from(J1811_OBSERVED_INITIALIZATION)
                );
                assert_eq!(actual_quote.largest_layers, Some(9), "{purpose:?}");
                assert!(actual_quote.budget.fits(9) && !actual_quote.budget.fits(10));
                assert_eq!(
                    i128::from(actual_quote.budget.preferred_deficit_bytes()),
                    -actual_slack
                );
                let admitted = admit(&actual).unwrap();
                assert_eq!(admitted.tile_rows, tile);
                assert_eq!(admitted.resident_layers, 9);
                assert_eq!(admitted.placement, actual_quote.budget);
                assert_eq!(
                    i128::from(admitted.device_snapshot_budget_bytes),
                    actual_quote.budget.slack_bytes_at(9)
                );
                // Exact edge: the actual free that holds ten, and one byte less.
                let edge_free = J1811_POST_INIT_FREE + u64::try_from(-actual_slack).unwrap();
                for (free, expected) in [(edge_free, 10usize), (edge_free - 1, 9)] {
                    let at = AdmissionInputs {
                        memory: AdmissionMemory::measured(J1811_PRE_LOAD_FREE, free),
                        ..actual
                    };
                    assert_eq!(admit(&at).unwrap().resident_layers, expected, "{free}");
                }
                // The frontier reads the same quote at the actual memory.
                let points = tile_frontier(&actual, |tile| {
                    Ok((
                        j1778_reserve_at(tile),
                        base_slack_at(tile, weights_arena, shadows_arena),
                    ))
                })
                .unwrap();
                let point = selected(&points);
                assert_eq!(point.budget, actual_quote.budget);
                assert_eq!(point.resident_layers, 9);
                assert_eq!(point.verdict, admit(&actual).map(|_| ()));

                for (observation, inputs) in [
                    ("reference_pre_equals_post", &reference),
                    ("actual_post_initialization", &actual),
                ] {
                    let projection = project_at_tile(inputs, tile).unwrap();
                    let mut quote = quote_r5(observation, inputs, tile);
                    quote["host_owners"] = serde_json::json!({
                        "weight_layers_with_experts": expert_layers,
                        "selected": host_owners(projection.resident_layers()),
                        "preferred": host_owners(S1_PREFERRED_RESIDENT_LAYERS),
                    });
                    if let Some(init) = quote["memory"]["observed_initialization_bytes"].as_u64() {
                        // The pre-initialization-domain equation is the same
                        // slack plus the measured interval: one charge only.
                        assert_eq!(
                            quote["pre_init_domain_slack_at_selected_bytes"]
                                .as_i64()
                                .unwrap(),
                            quote["slack_bytes"].as_i64().unwrap() + i64::try_from(init).unwrap()
                        );
                    }
                    println!(
                        "SLOWLINK_ADMISSION_QUOTE_R5 {}",
                        serde_json::to_string(&quote).unwrap()
                    );
                }
            }
        }

        /// The factory's roles are `resident_split`'s non-expert entries: CPU
        /// metadata records (not manifest entries), tied aliases, external row
        /// tables and routed experts never become members; the resident
        /// payload equals `resident_split`'s.
        #[test]
        fn slowlink_arena_roles_exclude_cpu_metadata_aliases_externals_and_experts() {
            let fixture = fixture();
            let plan = plan_of(&fixture, 2 * MIB as usize);
            assert_eq!(fixture.cpu_metadata.len(), 3);
            assert_eq!(
                fixture
                    .cpu_metadata
                    .iter()
                    .map(|record| record.1)
                    .sum::<u64>(),
                280
            );
            assert!(!fixture.external.is_empty() && !fixture.experts.is_empty());
            for (name, _) in &fixture.cpu_metadata {
                assert!(plan.weights.member(name).is_none(), "{name}");
                assert!(plan.f16_shadows.member(name).is_none(), "{name}");
            }
            for name in fixture.external.iter().chain(&fixture.experts) {
                assert!(plan.weights.member(name).is_none(), "{name}");
                assert!(plan.f16_shadows.member(name).is_none(), "{name}");
            }
            assert!(plan.weights.member(ALIAS_NAME).is_none());
            assert!(plan.f16_shadows.member(ALIAS_NAME).is_none());
            // Every resident non-expert tensor is a weight member, and every
            // shadow owner is one of them.
            for (name, bytes) in &fixture.resident {
                assert_eq!(
                    plan.weights.member(name).unwrap().bytes as u64,
                    *bytes,
                    "{name}"
                );
            }
            assert_eq!(plan.weights.members.len(), fixture.resident.len());
            for span in &plan.f16_shadows.members {
                assert!(plan.weights.member(&span.name).is_some(), "{}", span.name);
            }
            assert_eq!(
                residency::resident_split(&fixture.weights, fixture.bytes_of())
                    .unwrap()
                    .0,
                plan.weights.logical_bytes().unwrap()
            );
            // A device that cannot shadow plans no shadow member at all.
            let layers = crate::config::compact_test_config().num_hidden_layers;
            let uncapable = residency::model_weight_arena_plan(
                &fixture.weights,
                fixture.bytes_of(),
                layers,
                false,
                2 * MIB as usize,
            )
            .unwrap();
            assert!(uncapable.f16_shadows.members.is_empty());
            assert!(uncapable.f16_shadows.roots.is_empty());
            assert_eq!(uncapable.weights, plan.weights);
        }

        /// The slack is the provided plan's root backing minus the unchanged
        /// logical payload, only for what the reserve charges, and refuses a
        /// plan that is not this manifest's or not at this granule.
        #[test]
        fn slowlink_arena_slack_is_root_backing_minus_the_logical_payload() {
            let fixture = fixture();
            let granule = 2 * MIB;
            let layers = crate::config::compact_test_config().num_hidden_layers;
            let plan = plan_of(&fixture, granule as usize);
            let slack = |plan: &ModelWeightArenaPlan, charge: bool, granule: u64| {
                residency::weight_arena_granule_slack(
                    plan,
                    &fixture.weights,
                    fixture.bytes_of(),
                    layers,
                    charge,
                    granule,
                )
            };
            assert_eq!(
                slack(&plan, true, granule),
                Ok((
                    plan.weights.backing_bytes(granule as usize).unwrap() - J1778_NON_EXPERT,
                    plan.f16_shadows.backing_bytes(granule as usize).unwrap() - J1778_SHADOWS,
                ))
            );
            // Shadows the reserve does not charge add nothing.
            assert_eq!(slack(&plan, false, granule), Ok((10_945_280, 0)));
            // A different granule than the plan's is refused, not re-rounded.
            assert!(slack(&plan, true, granule + 4096)
                .unwrap_err()
                .contains("granule"));
            // A plan of another manifest (one weight missing; shadows absent
            // while charged) never reproduces the census payload.
            let mut short = fixture.weights.clone();
            short.retain(|entry| entry.name != "model.language_model.embed_tokens.weight");
            let other = residency::model_weight_arena_plan(
                &short,
                |entry| fixture.sizes.get(&entry.name).copied(),
                layers,
                true,
                granule as usize,
            )
            .unwrap();
            assert!(slack(&other, true, granule)
                .unwrap_err()
                .contains("census charges"));
            let no_shadows = residency::model_weight_arena_plan(
                &fixture.weights,
                fixture.bytes_of(),
                layers,
                false,
                granule as usize,
            )
            .unwrap();
            assert!(slack(&no_shadows, true, granule)
                .unwrap_err()
                .contains("census charges"));
        }

        /// Overflow, the 256 B cap floor, the default (no weights) plan, the
        /// missing-size and zero-granule refusals, and the unchanged Copy
        /// semantics and non-S1 laziness of the admission types.
        #[test]
        fn slowlink_arena_overflow_cap_default_fast_uma_and_unmeasured_are_unchanged() {
            use std::cell::Cell;
            fn is_copy<T: Copy>() {}
            is_copy::<AdmissionInputs>();
            is_copy::<SlowlinkProfile>();

            let layers = crate::config::compact_test_config().num_hidden_layers;
            let entry = |name: &str| {
                WeightEntry::model(name, vec![4, 4], DType::BF16, ShardPolicy::Replicate)
            };
            let granule = 2 * MIB as usize;
            // Overflow: a payload whose rounded backing exceeds u64 refuses
            // before any allocation, with a named error.
            let huge = vec![entry("model.huge.weight")];
            let overflowed = residency::model_weight_arena_plan(
                &huge,
                |_| Some(u64::MAX),
                layers,
                true,
                granule,
            )
            .and_then(|plan| {
                residency::weight_arena_granule_slack(
                    &plan,
                    &huge,
                    |_| Some(u64::MAX),
                    layers,
                    true,
                    granule as u64,
                )
            });
            assert!(overflowed.is_err(), "{overflowed:?}");
            // A missing payload size, and a zero granule, refuse.
            assert!(
                residency::model_weight_arena_plan(&huge, |_| None, layers, true, granule)
                    .unwrap_err()
                    .contains("no payload size")
            );
            assert!(
                residency::model_weight_arena_plan(&huge, |_| Some(10), layers, true, 0)
                    .unwrap_err()
                    .contains("granule")
            );
            // The cap never falls below 256 B and every root asks for at least
            // that.
            let small: Vec<WeightEntry> = (0..4).map(|i| entry(&format!("w{i}"))).collect();
            let tiny =
                residency::model_weight_arena_plan(&small, |_| Some(10), layers, true, granule)
                    .unwrap();
            assert_eq!(tiny.weights.max_root_bytes, 256);
            assert_eq!(tiny.weights.roots, vec![256; 4]);
            // No weights, no roots and no backing.
            let empty =
                residency::model_weight_arena_plan(&[], |_| Some(10), layers, true, granule)
                    .unwrap();
            assert!(empty.weights.roots.is_empty() && empty.f16_shadows.roots.is_empty());
            assert_eq!(empty.weights.backing_bytes(granule), Ok(0));
            empty.validate().unwrap();

            // Fast, UMA, unmeasured, explicit-layer and non-S1 loads never ask
            // for facts, so they never build the plan or query the granule; an
            // incumbent charge set carries no arena term.
            let built = Cell::new(false);
            let facts = || -> Result<LoadFacts<'static>, String> {
                built.set(true);
                Err("the arena plan must not be built".to_string())
            };
            for load in [
                LoadShape {
                    unified_memory: true,
                    ..shape()
                },
                LoadShape {
                    explicit_vram_layers: true,
                    ..shape()
                },
            ] {
                assert_eq!(
                    select_load_profile(
                        SlowlinkMode::Force,
                        &load,
                        || unreachable!("probed"),
                        facts
                    ),
                    Ok(None)
                );
            }
            assert_eq!(
                select_load_profile(
                    SlowlinkMode::Off,
                    &shape(),
                    || unreachable!("probed"),
                    facts
                ),
                Ok(None)
            );
            for link in [
                HostLink::Unified,
                HostLink::Unmeasured,
                HostLink::Discrete { gbps: 16.0 },
                HostLink::Discrete { gbps: 26.0 },
            ] {
                assert_eq!(
                    select_load_profile(SlowlinkMode::Force, &shape(), || link, facts),
                    Ok(None),
                    "{link:?}"
                );
            }
            assert!(!built.get());
            let payload_only = admit(&inputs(free_for(&inputs(0), 10))).unwrap();
            assert_eq!(payload_only.granule, GranuleCharge::default());
            assert_eq!(GranuleInputs::default().granule_bytes, 0);
        }
    }

    #[test]
    fn s1_resident_layer_set_front_is_the_leading_layers() {
        for raw in [None, Some("front"), Some(" front "), Some("FRONT")] {
            assert_eq!(
                resolve_resident_layer_set(raw, 48, 9),
                Ok((0..9).collect::<Vec<_>>()),
                "{raw:?}"
            );
            assert_eq!(
                resolve_resident_layer_set(raw, 48, 0),
                Ok(Vec::new()),
                "{raw:?}"
            );
            assert_eq!(
                resolve_resident_layer_set(raw, 48, 48),
                Ok((0..48).collect::<Vec<_>>()),
                "{raw:?}"
            );
        }
        assert_eq!(
            S1_RESIDENT_LAYERS_ENV,
            "HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS"
        );
        for raw in [None, Some("front"), Some(" Front ")] {
            assert!(!resident_layer_switch_is_override(raw), "{raw:?}");
        }
        for raw in [
            Some("spread"),
            Some("1,2"),
            Some("bogus"),
            Some(""),
            Some("  "),
        ] {
            assert!(resident_layer_switch_is_override(raw), "{raw:?}");
        }
        // The environment entry point reads the same table; with the switch
        // unset in the test process it is the front set.
        if hipfire_config::developer_var(S1_RESIDENT_LAYERS_ENV).is_err() {
            assert_eq!(s1_resident_layer_set(48, 3), Ok(vec![0, 1, 2]));
        }
    }

    #[test]
    fn s1_resident_layer_set_spread_matches_the_pinned_formula() {
        let spread = |total, count| resolve_resident_layer_set(Some("spread"), total, count);
        assert_eq!(spread(48, 9), Ok(vec![3, 8, 13, 19, 24, 29, 35, 40, 45]));
        assert_eq!(
            resolve_resident_layer_set(Some(" Spread "), 48, 9),
            spread(48, 9)
        );
        assert_eq!(spread(48, 0), Ok(Vec::new()));
        assert_eq!(spread(48, 1), Ok(vec![25]));
        // Dense: every layer resident, never a clamp collision.
        assert_eq!(spread(48, 48), Ok((0..48).collect::<Vec<_>>()));
        assert_eq!(spread(1, 1), Ok(vec![0]));
        // Strictly ascending, distinct and in range for every legal (T, N).
        for total in 1..=64usize {
            for count in 0..=total {
                let set = spread(total, count).unwrap_or_else(|e| panic!("T{total} N{count}: {e}"));
                assert_eq!(set.len(), count, "T{total} N{count}");
                assert!(
                    set.windows(2).all(|w| w[0] < w[1]),
                    "T{total} N{count}: {set:?}"
                );
                assert!(
                    set.iter().all(|&layer| layer < total),
                    "T{total} N{count}: {set:?}"
                );
            }
        }
    }

    #[test]
    fn s1_resident_layer_set_explicit_list_is_sorted_and_exact() {
        let list = |raw: &str, total, count| resolve_resident_layer_set(Some(raw), total, count);
        assert_eq!(list("45,3,20", 48, 3), Ok(vec![3, 20, 45]));
        assert_eq!(list(" 7 , 0 ,47", 48, 3), Ok(vec![0, 7, 47]));
        assert_eq!(list("5", 48, 1), Ok(vec![5]));
        assert_eq!(list("2,1,0", 3, 3), Ok(vec![0, 1, 2]));
    }

    #[test]
    fn s1_resident_layer_set_rejects_invalid_values_naming_the_switch() {
        let named = |result: Result<Vec<usize>, String>, what: &str| {
            let error = result.expect_err(what);
            assert!(error.contains(S1_RESIDENT_LAYERS_ENV), "{what}: {error}");
        };
        // Wrong entry count (too few, too many, any list at N = 0).
        named(resolve_resident_layer_set(Some("1,2"), 48, 3), "short list");
        named(
            resolve_resident_layer_set(Some("1,2,3,4"), 48, 3),
            "long list",
        );
        named(resolve_resident_layer_set(Some("1"), 48, 0), "list at N=0");
        // Out of range, duplicates, malformed entries.
        named(
            resolve_resident_layer_set(Some("1,2,48"), 48, 3),
            "layer == T",
        );
        named(
            resolve_resident_layer_set(Some("1,2,99"), 48, 3),
            "layer > T",
        );
        named(
            resolve_resident_layer_set(Some("1,2,2"), 48, 3),
            "duplicate",
        );
        named(
            resolve_resident_layer_set(Some("1,,2"), 48, 3),
            "empty entry",
        );
        named(resolve_resident_layer_set(Some(""), 48, 3), "empty value");
        named(resolve_resident_layer_set(Some("  "), 48, 3), "blank value");
        named(
            resolve_resident_layer_set(Some(""), 48, 0),
            "empty value at N=0",
        );
        named(
            resolve_resident_layer_set(Some("1,2,"), 48, 3),
            "trailing comma",
        );
        named(
            resolve_resident_layer_set(Some("1,-2,3"), 48, 3),
            "negative",
        );
        named(
            resolve_resident_layer_set(Some("1,x,3"), 48, 3),
            "non-numeric",
        );
        named(
            resolve_resident_layer_set(Some("middle"), 48, 3),
            "unknown word",
        );
        // More resident layers than trunk layers is impossible for any form.
        named(
            resolve_resident_layer_set(Some("spread"), 4, 5),
            "spread N>T",
        );
        named(
            resolve_resident_layer_set(Some("0,1,2,3,4"), 4, 5),
            "list N>T",
        );
    }
    // ── symmetric IU4 window route (requested by default; HIPFIRE_QWEN4_MOE_SYM_IU4=0 kills) ──

    #[test]
    fn slowlink_admission_sym_iu4_route_resolution_truth_table() {
        for layout in [WindowRoute::SlabIu4, WindowRoute::Aggregate] {
            for descriptor in [false, true] {
                for requested in [false, true] {
                    for recipe in [false, true] {
                        for source in [None, Some(false), Some(true)] {
                            let facts = SymIu4RouteFacts {
                                descriptor,
                                requested,
                                recipe_bf16_round_trip: recipe,
                                source_symmetric: source,
                            };
                            let all = layout == WindowRoute::Aggregate
                                && descriptor
                                && requested
                                && recipe;
                            let expected = if all && source == Some(true) {
                                WindowRoute::AggregateSymIu4
                            } else {
                                layout
                            };
                            let label = format!("{layout:?} {facts:?}");
                            assert_eq!(resolve_window_route(layout, &facts), expected, "{label}");
                            // The scan can change the verdict exactly when the layout and the
                            // three device/recipe terms hold.
                            assert_eq!(facts.scan_needed(layout), all, "{label}");
                        }
                    }
                }
            }
        }
        // Flag off (not requested) is the incumbent layout route whatever else holds.
        let off = SymIu4RouteFacts {
            descriptor: true,
            requested: false,
            recipe_bf16_round_trip: true,
            source_symmetric: Some(true),
        };
        assert_eq!(
            resolve_window_route(WindowRoute::Aggregate, &off),
            WindowRoute::Aggregate
        );
    }

    #[test]
    fn slowlink_admission_sym_iu4_scan_runs_once_and_only_when_it_can_change_the_verdict() {
        use std::cell::Cell;
        let base = SymIu4RouteFacts {
            descriptor: true,
            requested: true,
            recipe_bf16_round_trip: true,
            source_symmetric: None,
        };
        let run = |layout, facts: SymIu4RouteFacts, answer: Result<bool, String>| {
            let calls = Cell::new(0usize);
            let out = resolve_window_route_with_scan(layout, facts, || {
                calls.set(calls.get() + 1);
                answer
            });
            (out, calls.get())
        };
        let resolved = |route, source_symmetric| {
            Ok(ResolvedWindowRoute {
                route,
                source_symmetric,
            })
        };
        let (sym, f16) = (WindowRoute::AggregateSymIu4, WindowRoute::Aggregate);
        assert_eq!(run(f16, base, Ok(true)), (resolved(sym, Some(true)), 1));
        assert_eq!(run(f16, base, Ok(false)), (resolved(f16, Some(false)), 1));
        // No scan when it cannot change the verdict.
        for facts in [
            SymIu4RouteFacts {
                requested: false,
                ..base
            },
            SymIu4RouteFacts {
                descriptor: false,
                ..base
            },
            SymIu4RouteFacts {
                recipe_bf16_round_trip: false,
                ..base
            },
        ] {
            assert_eq!(
                run(f16, facts, Ok(true)),
                (resolved(f16, None), 0),
                "{facts:?}"
            );
        }
        assert_eq!(
            run(WindowRoute::SlabIu4, base, Ok(true)),
            (resolved(WindowRoute::SlabIu4, None), 0)
        );
        // A verdict already known is never rescanned.
        let known = SymIu4RouteFacts {
            source_symmetric: Some(true),
            ..base
        };
        assert_eq!(run(f16, known, Ok(false)), (resolved(sym, Some(true)), 0));
        // A scan error is the caller's decline, never a guess.
        let (error, calls) = run(f16, base, Err("short payload".to_string()));
        assert_eq!(error, Err("short payload".to_string()));
        assert_eq!(calls, 1);
    }

    #[test]
    fn slowlink_admission_sym_iu4_late_final_header_mismatch_is_a_load_error() {
        let sym = WindowRoute::AggregateSymIu4;
        assert_eq!(check_sym_iu4_final_headers(sym, 48, 48), Ok(()));
        for (verified, total) in [(47usize, 48usize), (0, 48), (0, 0)] {
            let error = check_sym_iu4_final_headers(sym, verified, total).unwrap_err();
            assert!(error.contains(&format!("{verified}/{total}")), "{error}");
            assert!(error.contains("before serving"), "{error}");
            assert!(error.contains("aggregate-sym-iu4"), "{error}");
        }
        // Every other route has no IU4 coverage to meet (the F16 route keeps its asymmetric
        // layers on the F16 arm).
        for route in [WindowRoute::SlabIu4, WindowRoute::Aggregate] {
            assert_eq!(check_sym_iu4_final_headers(route, 0, 48), Ok(()));
        }
    }

    #[test]
    fn slowlink_residency_sym_iu4_route_adapts_before_declining_on_a_base_miss() {
        let sym_route = WindowRoute::AggregateSymIu4;
        for purpose in [
            AdmissionPurpose::Product,
            AdmissionPurpose::PrimitiveSelftest,
        ] {
            let f16_inputs = j1778_inputs(WindowRoute::Aggregate, purpose, J1778_PRELOAD_FREE);
            let inputs = j1778_inputs(sym_route, purpose, J1778_PRELOAD_FREE);
            let f16 = admit(&f16_inputs).unwrap();
            let profile = admit(&inputs).unwrap();
            let label = format!("{purpose:?}");
            assert_eq!(profile.geometry.route, sym_route, "{label}");
            assert_eq!(profile.window_rows, REFERENCE_WINDOW_ROWS, "{label}");
            assert_eq!(profile.resident_layers, 10, "{label}");
            // Both A4 sidecars (78.75 MiB at 16K) and no window F16 input or F16 slot map.
            assert_eq!(profile.census.a4_sidecars, 82_575_360, "{label}");
            assert_eq!(f16.census.a4_sidecars, 0, "{label}");
            assert_eq!(
                f16.census.route_storage - profile.census.route_storage,
                83_886_080 + 2048 * 4,
                "{label}"
            );
            // Every other owned term is the aggregate route's, byte for byte.
            assert_eq!(profile.census.ring, f16.census.ring, "{label}");
            assert_eq!(profile.census.lookahead, f16.census.lookahead, "{label}");
            assert_eq!(
                profile.census.group_scatter, f16.census.group_scatter,
                "{label}"
            );
            assert_eq!(profile.census.arena, f16.census.arena, "{label}");
            assert_eq!(
                profile.census.boundary_state, f16.census.boundary_state,
                "{label}"
            );
            // The census is the runtime layout's (what `plan_runtime` re-checks at attach).
            let layout = hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(
                &profile.route_dims(),
            )
            .unwrap();
            assert_eq!(
                layout.total_bytes() as u64,
                profile.census.route_storage,
                "{label}"
            );
            assert_eq!(
                layout.a4_sidecar_bytes() as u64,
                profile.census.a4_sidecars,
                "{label}"
            );
            // Exact budget boundary: the slack in bytes less free still admits ten layers;
            // one byte fewer keeps nine, and only a miss on the zero-layer base declines.
            let slack = j1778_slack(&inputs, &profile, 10);
            assert!(slack > 0, "{label}");
            let edge = AdmissionInputs {
                memory: AdmissionMemory::actual(J1778_PRELOAD_FREE - u64::try_from(slack).unwrap()),
                ..inputs
            };
            assert_eq!(admit(&edge).unwrap().resident_layers, 10, "{label}");
            let below = AdmissionInputs {
                memory: AdmissionMemory::actual(edge.memory.free_bytes - 1),
                ..edge
            };
            let nine = admit(&below).unwrap();
            assert_eq!(nine.resident_layers, 9, "{label}");
            assert_eq!(nine.placement.preferred_deficit_bytes(), 1, "{label}");
            let base_free = edge.memory.free_bytes - 10 * LAYER;
            let base = AdmissionInputs {
                memory: AdmissionMemory::actual(base_free),
                ..inputs
            };
            assert_eq!(admit(&base).unwrap().resident_layers, 0, "{label}");
            let miss = AdmissionInputs {
                memory: AdmissionMemory::actual(base_free - 1),
                ..base
            };
            assert!(
                matches!(admit(&miss), Err(SlowlinkDecline::Deficit(_))),
                "{label}"
            );
        }
    }

    /// A window smaller than the aggregate routes' worst-case cohort bound
    /// (`k1 48 + tail 3066 = 3114` rows) is a named incumbent decline before the census, not a
    /// layout error at load; every size that admits is untouched and builds a valid layout.
    #[test]
    fn slowlink_admission_window_smaller_than_its_cohort_rows_declines_by_name() {
        use hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout;
        let cohort = S1_K1_TAIL_ROWS + S1_AGGREGATE_TAIL_ROWS;
        assert_eq!(cohort, 3114);
        for route in [WindowRoute::Aggregate, WindowRoute::AggregateSymIu4] {
            for max_seq in [1536, 1537, 3113, 3114, S1_WINDOW_ROWS] {
                let load = LoadShape { max_seq, ..shape() };
                let pre = preflight(SlowlinkMode::Force, &load);
                if max_seq < S1_MIN_FORCED_REQUEST_ROWS {
                    assert_eq!(pre, Err(SlowlinkDecline::ShortContext { max_seq }));
                    continue;
                }
                assert_eq!(pre, Ok(()), "{route:?} {max_seq}");
                let inputs = AdmissionInputs {
                    window_rows: S1_WINDOW_ROWS.min(max_seq),
                    ..j1778_inputs(route, AdmissionPurpose::Product, J1778_PRELOAD_FREE)
                };
                let admitted = admit_grown(&inputs, max_seq);
                if max_seq < cohort {
                    let decline = SlowlinkDecline::WindowTooSmall {
                        rows: max_seq,
                        k1_rows: S1_K1_TAIL_ROWS,
                        tail_rows: S1_AGGREGATE_TAIL_ROWS,
                    };
                    assert_eq!(
                        admitted.map(|_| ()),
                        Err(decline.clone()),
                        "{route:?} {max_seq}"
                    );
                    assert!(!decline.is_capacity());
                    assert_eq!(
                        decline.to_string(),
                        format!(
                            "window too small for its cohort rows: rows {max_seq} < k1 48 + tail 3066"
                        )
                    );
                    // The forced selector keeps the incumbent instead of failing the load.
                    let selected = select_profile(
                        SlowlinkMode::Force,
                        &load,
                        || HostLink::Discrete { gbps: 7.09 },
                        |_| Ok(inputs.clone()),
                    );
                    assert_eq!(
                        selected.map(|p| p.is_some()),
                        Ok(false),
                        "{route:?} {max_seq}"
                    );
                } else {
                    let profile = admitted.unwrap_or_else(|d| panic!("{route:?} {max_seq}: {d}"));
                    assert_eq!(profile.window_rows, S1_WINDOW_ROWS.min(max_seq));
                    let layout = PromptRouteLayout::new(&profile.route_dims()).unwrap();
                    assert_eq!(layout.total_bytes() as u64, profile.census.route_storage);
                    assert_eq!(layout.dims().k1_rows, S1_K1_TAIL_ROWS);
                    assert_eq!(layout.dims().tail_rows, S1_AGGREGATE_TAIL_ROWS);
                }
            }
        }
        // The slab route charges no tail rows: every legal context admits.
        for max_seq in [1537, 3113, 3114, S1_WINDOW_ROWS] {
            let inputs = AdmissionInputs {
                window_rows: S1_WINDOW_ROWS.min(max_seq),
                ..j1778_inputs(
                    WindowRoute::SlabIu4,
                    AdmissionPurpose::Product,
                    J1778_PRELOAD_FREE,
                )
            };
            let profile = admit_grown(&inputs, max_seq).unwrap();
            let layout = PromptRouteLayout::new(&profile.route_dims()).unwrap();
            assert_eq!((layout.dims().k1_rows, layout.dims().tail_rows), (48, 0));
        }
    }

    #[test]
    fn slowlink_admission_sym_iu4_route_dims_keep_the_aggregate_cohort_bounds() {
        let config = crate::config::compact_test_config();
        let dims = |route| {
            route_dims(
                &ModelGeometry::from_config(&config, route).unwrap(),
                S1_WINDOW_ROWS,
            )
        };
        let (f16, sym) = (
            dims(WindowRoute::Aggregate),
            dims(WindowRoute::AggregateSymIu4),
        );
        assert_eq!(sym.tail_rows, S1_AGGREGATE_TAIL_ROWS);
        assert_eq!((sym.k1_rows, sym.tail_rows), (f16.k1_rows, f16.tail_rows));
        assert_eq!(sym.route, WindowRoute::AggregateSymIu4);
        // The pure classifier never answers the IU4 route: it is resolved, not declared.
        assert_eq!(window_route_of(&[]), WindowRoute::Aggregate);
    }

    #[test]
    fn slowlink_admission_sym_iu4_profile_log_names_only_the_sym_iu4_route() {
        let purpose = AdmissionPurpose::Product;
        let f16 = admit(&j1778_inputs(
            WindowRoute::Aggregate,
            purpose,
            J1778_PRELOAD_FREE,
        ))
        .unwrap()
        .describe();
        let sym = admit(&j1778_inputs(
            WindowRoute::AggregateSymIu4,
            purpose,
            J1778_PRELOAD_FREE,
        ))
        .unwrap()
        .describe();
        assert!(!f16.contains("window route"), "{f16}");
        assert!(sym.ends_with(", window route aggregate-sym-iu4"), "{sym}");
    }

    #[test]
    fn slowlink_admission_sym_iu4_flag_off_profile_is_the_f16_aggregate_profile() {
        // A route that is not requested (`HIPFIRE_QWEN4_MOE_SYM_IU4=0`, the kill switch; the
        // flag unset is requested on gfx1100, gfx1151 and gfx1201 because the S1 window reads the
        // same `Gpu::qwen4_moe_sym_iu4_requested` as the regular prefill) resolves the incumbent
        // layout route whatever else holds, so every byte, placement and selection of the F16
        // Aggregate route is unchanged.
        let off = SymIu4RouteFacts {
            descriptor: true,
            requested: false,
            recipe_bf16_round_trip: true,
            source_symmetric: Some(true),
        };
        let route = resolve_window_route(WindowRoute::Aggregate, &off);
        assert_eq!(route, WindowRoute::Aggregate);
        for purpose in [
            AdmissionPurpose::Product,
            AdmissionPurpose::PrimitiveSelftest,
        ] {
            let resolved = j1778_inputs(route, purpose, J1778_PRELOAD_FREE);
            let incumbent = j1778_inputs(WindowRoute::Aggregate, purpose, J1778_PRELOAD_FREE);
            assert_eq!(
                census_of(&resolved).unwrap().2,
                census_of(&incumbent).unwrap().2
            );
            assert_eq!(admit(&resolved).unwrap(), admit(&incumbent).unwrap());
            assert_eq!(census_of(&resolved).unwrap().2.a4_sidecars, 0);
        }
    }

    fn sp_facts_of(
        profile: &SlowlinkProfile,
        knob: bool,
    ) -> crate::slowlink_sp::SpLoadFacts<'static> {
        crate::slowlink_sp::SpLoadFacts {
            knob,
            touched_knob: false,
            force_all: false,
            pages: 18,
            f16_route_ready: true,
            window_rows: profile.window_rows,
            s1_route: Some(profile.geometry.route),
            arch: "gfx1100",
            n_exp: 512,
            cpu_gate_up_quant: true,
            cpu_down_quant: true,
            splice_workers: Some(16),
            region_fit_rows: 5_593,
            hidden: 2560,
            pinned_budget_bytes: 1 << 30,
            prefix_cache: false,
        }
    }

    /// SP adds no VRAM and moves no layer: on the real j1778 fixtures, both
    /// routes and purposes, the admitted profile (resident layers, every census
    /// term, the placement quote) is identical before and after SP is armed
    /// and admitted, and SP's own admission reads nothing the profile charges.
    #[test]
    fn slowlink_sp_armed_leaves_the_j1778_profiles_identical() {
        use crate::slowlink_sp::{admit_sp, SpDecline};
        for (route, purpose) in [
            (WindowRoute::Aggregate, AdmissionPurpose::Product),
            (WindowRoute::Aggregate, AdmissionPurpose::PrimitiveSelftest),
            (WindowRoute::SlabIu4, AdmissionPurpose::Product),
            (WindowRoute::SlabIu4, AdmissionPurpose::PrimitiveSelftest),
        ] {
            let label = format!("{route:?} {purpose:?}");
            let inputs = j1778_inputs(route, purpose, J1778_PRELOAD_FREE);
            let before = admit(&inputs).unwrap();
            assert_eq!(before.resident_layers, 10, "{label}");
            let total = before.census.total().unwrap();
            let census = census_of(&inputs).unwrap().2;

            let armed = admit_sp(&sp_facts_of(&before, true));
            match route {
                WindowRoute::Aggregate => {
                    let sp = armed.unwrap_or_else(|d| panic!("{label}: {d}"));
                    assert_eq!(sp.sp_max_rows, 5_376, "{label}");
                }
                _ => assert_eq!(
                    armed.unwrap_err(),
                    SpDecline::Route {
                        route,
                        mode: crate::slowlink_sp::SpMode::Hybrid
                    },
                    "{label}"
                ),
            }
            let unarmed = admit_sp(&sp_facts_of(&before, false));
            assert_eq!(unarmed.unwrap_err(), SpDecline::KnobOff, "{label}");

            // The S1 admission after SP armed: bit-for-bit the same profile.
            let after = admit(&inputs).unwrap();
            assert_eq!(after, before, "{label}");
            assert_eq!(after.resident_layers, before.resident_layers, "{label}");
            assert_eq!(after.census.total().unwrap(), total, "{label}");
            assert_eq!(census_of(&inputs).unwrap().2, census, "{label}");
        }
    }

    #[test]
    fn slowlink_sp_request_truth_table() {
        use crate::slowlink_sp::{admit_sp, SpBypass};
        let profile = profile_at_ten();
        let sp = admit_sp(&crate::slowlink_sp::SpLoadFacts {
            s1_route: Some(WindowRoute::Aggregate),
            ..sp_facts_of(&profile, true)
        })
        .unwrap();
        let tile = profile.tile_rows;
        assert_eq!(tile, S1_PRODUCT_TILE_ROWS);
        let req = |rows, cuts, capture| RequestShape {
            rows,
            entry_pos: 0,
            session_cuts: cuts,
            capture,
        };
        let sp_req =
            |mode, rows| qualify_request_sp(mode, &profile, &sp, &req(rows, 0, CaptureKind::None));
        for mode in [SlowlinkMode::Auto, SlowlinkMode::Force] {
            // 5..=tile rows: S1 admits one SP window below its own floor.
            for rows in [5, 100, 512, tile] {
                assert_eq!(
                    sp_req(mode, rows),
                    Ok(RequestVerdict { rows, windows: 1 }),
                    "{mode:?} {rows}"
                );
            }
            // Below SP's minimum, and above the tile, the request is exactly
            // `qualify_request`.
            for rows in [
                0,
                1,
                4,
                tile + 1,
                1536,
                1537,
                3072,
                S1_WINDOW_ROWS,
                S1_WINDOW_ROWS + 1,
            ] {
                assert_eq!(
                    sp_req(mode, rows),
                    qualify_request(mode, &profile, &req(rows, 0, CaptureKind::None)),
                    "{mode:?} {rows}"
                );
            }
        }
        // The kill switch wins everywhere.
        assert_eq!(
            sp_req(SlowlinkMode::Off, 512),
            Err(RequestBypass::KillSwitch)
        );
        // A bounded capture qualifies; all-rows / external stay incumbent.
        assert_eq!(
            qualify_request_sp(
                SlowlinkMode::Auto,
                &profile,
                &sp,
                &req(512, 0, CaptureKind::Bounded)
            ),
            Ok(RequestVerdict {
                rows: 512,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request_sp(
                SlowlinkMode::Auto,
                &profile,
                &sp,
                &req(512, 0, CaptureKind::AllRows)
            ),
            Err(RequestBypass::AllRowCapture)
        );
        assert_eq!(
            qualify_request_sp(
                SlowlinkMode::Auto,
                &profile,
                &sp,
                &req(512, 0, CaptureKind::External)
            ),
            Err(RequestBypass::ExternalCapture)
        );
        // An interior session cut makes the one window non-SP: the incumbent runs.
        assert_eq!(
            qualify_request_sp(
                SlowlinkMode::Auto,
                &profile,
                &sp,
                &req(512, 1, CaptureKind::None)
            ),
            Err(RequestBypass::Sp(SpBypass::InteriorCuts { cuts: 1 }))
        );
        // Requests above one tile keep today's cut rule.
        assert_eq!(
            qualify_request_sp(
                SlowlinkMode::Force,
                &profile,
                &sp,
                &req(3072, 5, CaptureKind::None)
            ),
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &req(3072, 5, CaptureKind::None)
            )
        );
    }
}
