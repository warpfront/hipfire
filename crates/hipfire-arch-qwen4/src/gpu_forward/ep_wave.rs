// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The EXACT expert-parallel prefill wave: one joint forward of several lanes' prompt tiles on an
//! EP rank, byte-identical per lane to the serial lane tiles it replaces (BEAT-PAITON S1).
//!
//! A sub-512-row lane tile runs the EP FULL-TABLE MoE: every rank computes every expert of the
//! layer (peer tables over P2P, host-mapped arenas), no gather, no tile exchange, so its cost is
//! dominated by streaming the whole layer's expert weights, paid once PER TILE. The wave reads
//! those weights once for all lanes' rows: ONE sealed MoE call per layer over the concatenated
//! rows, on the same full-table runtime (`moe_full[layer]`) the singleton chunk binds. The
//! row-local dense stages (embed, PLE rows, HC read / write, the GDN and QSA in / out
//! projections) run once over all lanes' rows too (`HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT`, default
//! on; `0` runs them per lane); the stateful cores always run per lane on the lane's own state.
//!
//! # Why the bytes are the lanes' own
//!
//! Every routing decision of the sealed chain keys on `Gpu::route_rows(tokens)` (512-row floors:
//! the F16-WMMA router/shared GEMMs, the INT8 shared expert, the fused front, the symmetric IU4
//! route, the F16-WMMA grouped gate/up and down), never on the grouped row count. The wave runs
//! its whole forward under `Gpu::qwen4_route_rows_cap = 511`: lanes are below 512 rows so every
//! per-lane op is untouched, and the wide MoE call sees `511` routing rows, i.e. it picks the
//! sub-512 kernels (BF16 multirow router and shared expert, `o4_r8_x4` gate/up, `O8R16` down,
//! F32 grouped chain) a lane's own singleton picks, while its launches carry the real rows. Those
//! kernels compute each (slot, output row) with a fixed accumulation order independent of the
//! other slots of its tile, of the padding and of the launch's grouped row count.
//!
//! # Admission
//!
//! [`ep_wave_refusal`] is a pure function of host facts every rank holds identically
//! ([`EpWaveFacts`]); every rank refuses or admits together before any launch, and a refused
//! wave leaves the caller to run serial lane tiles (today's route).

//! # The tile route
//!
//! With the lowered owner-routed floor (`HIPFIRE_EP_PREFILL_TILE_MIN_ROWS`, default 9 while the
//! A8 chain is active) a sub-512-row lane tile runs the owner-routed TILE route on the A8 chain:
//! each rank computes only the experts it owns, BF16 grouped rows cross the tile exchange once
//! per layer. A wave of such lanes ([`EpWaveRoute::Tile`]) runs the MoE once per layer over all
//! rows on that route: the compact runtime, the exchange of the singleton tile chunk, the same
//! routing-row cap and the lanes' own tile scopes ([`ep_wave_scope_rows`]). Per-slot
//! independence of the chain (exact int32 K128 epochs in a fixed order, per-row quantization
//! blocks, tiles owned per EXPERT, per-token combine) is documented in
//! `qwen4_moe_ep_a8.gfx12.hip`; the dense half is the one the full-table wave proved.
//!
use super::ep_reserve_scratch::EpBankRoute;
use super::*;
use crate::gpu_forward_lanes::{LaneStagePolicy, StageName, StageShare};
use rdna_compute::prefill_invariant::QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP;

/// Most rows one wave carries (the plan of record's 2048-row pass).
pub const QWEN4_EP_WAVE_ROW_CAP: usize = 2048;

/// Fewest rows of a lane the exact wave takes: one more than the widest few-row (wide-verify)
/// window, `WIDE_VERIFY_MAX_ROWS` (64 on gfx1201). A lane of at most 8 rows runs the few-row
/// routes and 9..=64 rows the g32 gate/up family and the few-row-window projections in its
/// singleton. The wave is NOT byte-identical to that serial tile for a 49..=64-row lane (j8439,
/// `burstprefill/runs/g60`: u60x8 0/8, m8 6/8 exact; first-token KL(singleton||wave) mean 0.018,
/// max 0.114, top-1 flip in 1/17 lanes, greedy diverges), and neither int8 MoE kill switch
/// fixes it, so those lanes keep their serial tile.
pub const EXACT_WAVE_MIN_LANE_ROWS: usize =
    rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS + 1;

/// Sub-switch of the TILE-route wave (default ON after its card identity receipt, j7460; the master
/// switch `HIPFIRE_EP_LANE_PREFILL_WAVE_EXACT=0` kills every route): a burst whose lanes all take
/// the owner-routed tile route runs as ONE tile-route wave. Exact `0` refuses it
/// (`ep wave refused:`): the burst runs as serial tile-route tiles.
pub const QWEN4_EP_WAVE_TILE_ENV: &str = "HIPFIRE_EP_LANE_PREFILL_WAVE_TILE";

/// [`QWEN4_EP_WAVE_TILE_ENV`] parsed: on unless exact `0`.
pub(crate) fn ep_wave_tile_enabled_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

static EP_WAVE_TILE_KNOB: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    ep_wave_tile_enabled_from(
        hipfire_config::developer_var(QWEN4_EP_WAVE_TILE_ENV)
            .ok()
            .as_deref(),
    )
});

/// Whether the tile-route wave is switched on in this process (every rank thread agrees).
pub(crate) fn ep_wave_tile_enabled() -> bool {
    *EP_WAVE_TILE_KNOB
}

/// Sub-switch of the trunk-shard PARTIAL-route wave (default ON after its card identity receipt,
/// j7464; the master switch `HIPFIRE_EP_LANE_PREFILL_WAVE_EXACT=0` kills every route): a burst of
/// 65..=511-row lanes on a trunk-shard load runs as ONE wave on the shard's partial chunk route.
/// Exact `0` refuses it (`ep wave refused:`): the burst runs as serial partial-chunk tiles.
pub const QWEN4_EP_WAVE_SHARD_ENV: &str = "HIPFIRE_EP_LANE_PREFILL_WAVE_SHARD";

/// [`QWEN4_EP_WAVE_SHARD_ENV`] parsed: on unless exact `0`.
pub(crate) fn ep_wave_shard_enabled_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

static EP_WAVE_SHARD_KNOB: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    ep_wave_shard_enabled_from(
        hipfire_config::developer_var(QWEN4_EP_WAVE_SHARD_ENV)
            .ok()
            .as_deref(),
    )
});

/// Whether the trunk-shard partial-route wave is switched on in this process (every rank
/// thread agrees).
pub(crate) fn ep_wave_shard_enabled() -> bool {
    *EP_WAVE_SHARD_KNOB
}

/// Knob (default unset: [`SMALL_LANE_MIN_ROWS`] with the small lanes on, else
/// [`EXACT_WAVE_MIN_LANE_ROWS`]): the fewest rows of a lane the exact wave takes, 9 at the
/// lowest (a lane of at most 8 rows runs the few-row routes and is never a wave lane). An
/// explicit value overrides [`QWEN4_EP_WAVE_SMALL_LANES_ENV`].
pub const QWEN4_EP_WAVE_MIN_LANE_ROWS_ENV: &str = "HIPFIRE_EP_LANE_PREFILL_WAVE_MIN_ROWS";

static EXACT_WAVE_MIN_LANE_ROWS_KNOB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    exact_wave_lane_floor_from(
        hipfire_config::developer_var(QWEN4_EP_WAVE_MIN_LANE_ROWS_ENV)
            .ok()
            .as_deref(),
        ep_wave_small_lanes_enabled(),
    )
});

/// The fewest rows of a wave lane of this process (every rank thread agrees).
pub(crate) fn exact_wave_min_lane_rows() -> usize {
    *EXACT_WAVE_MIN_LANE_ROWS_KNOB
}

/// Kill switch of the SMALL LANES of the exact wave (default ON): lanes of 9..=64 rows
/// ([`SMALL_LANE_MIN_ROWS`]..[`EXACT_WAVE_MIN_LANE_ROWS`]) join the wave. Their singleton tile
/// takes the 16..=64-row lane WMMA arm of the BF16 projections (`Gpu::lane_wmma_applies`, keyed on
/// the REAL rows, not bitwise the SIMT arm), which a joint call over the wave's rows would not;
/// the wave therefore runs those projections once per lane (`Gpu::qwen4_wave_lane_rows`) and, when
/// a small lane is in the wave, every other row-local stage per lane too
/// ([`ep_wave_joint_for_lanes`]). Exact `0` restores the floor of [`EXACT_WAVE_MIN_LANE_ROWS`]
/// (65): such lanes run serial singleton tiles. An explicit `HIPFIRE_EP_LANE_PREFILL_WAVE_MIN_ROWS`
/// overrides either.
pub const QWEN4_EP_WAVE_SMALL_LANES_ENV: &str = "HIPFIRE_EP_LANE_PREFILL_WAVE_SMALL_LANES";

/// [`QWEN4_EP_WAVE_SMALL_LANES_ENV`] parsed: on unless exact `0`.
pub(crate) fn ep_wave_small_lanes_enabled_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

static EP_WAVE_SMALL_LANES_KNOB: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    ep_wave_small_lanes_enabled_from(
        hipfire_config::developer_var(QWEN4_EP_WAVE_SMALL_LANES_ENV)
            .ok()
            .as_deref(),
    )
});

/// Whether small lanes join the exact wave in this process (every rank thread agrees).
pub(crate) fn ep_wave_small_lanes_enabled() -> bool {
    *EP_WAVE_SMALL_LANES_KNOB
}

/// Fewest rows of a lane of the small-lane wave: lanes of at most 8 rows run the few-row routes.
pub const SMALL_LANE_MIN_ROWS: usize = rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS + 1;

/// The fewest rows of a wave lane: the explicit `HIPFIRE_EP_LANE_PREFILL_WAVE_MIN_ROWS` when it
/// parses (an integer in `9..=EXACT_WAVE_MAX_LANE_ROWS`), else [`SMALL_LANE_MIN_ROWS`] with
/// `small_lanes` on, else [`EXACT_WAVE_MIN_LANE_ROWS`].
pub(crate) fn exact_wave_lane_floor_from(value: Option<&str>, small_lanes: bool) -> usize {
    let default = if small_lanes {
        SMALL_LANE_MIN_ROWS
    } else {
        EXACT_WAVE_MIN_LANE_ROWS
    };
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|rows| (SMALL_LANE_MIN_ROWS..=EXACT_WAVE_MAX_LANE_ROWS).contains(rows))
        .unwrap_or(default)
}

/// Whether a wave of `lane_rows` runs its row-local dense stages jointly
/// ([`ep_wave_joint_enabled`]): only when EVERY lane has at least `floor` rows, where `floor` is
/// `WIDE_VERIFY_MAX_ROWS + 1` (65, every lane wider than the few-row window) unless the LAB
/// prefill lane WMMA floor (`HIPFIRE_QWEN4_PREFILL_LWMMA_WAVE_FLOOR`,
/// `FeatureFlags::qwen4_prefill_lwmma_floor_knob`) lowered it. The joint HC / GDN / QSA launches
/// select their kernels on the real rows of the call (lane WMMA at 16..=64 rows, the HC8 decode
/// route, the 9..=48-row MQ6 tile), so a lane below `floor` reproduces its singleton bytes only
/// with its own per-lane launches; a lane at or above a lowered floor takes the same prefill
/// entries at the same pinned split-K in its own tile and in the joint call.
pub(crate) fn ep_wave_joint_for_lanes(
    joint_enabled: bool,
    lane_rows: &[usize],
    floor: usize,
) -> bool {
    joint_enabled && lane_rows.iter().all(|&rows| rows >= floor)
}

/// Most rows of a lane the exact wave takes: every lane stays below the 512-row floors (a
/// larger tile takes the owner-routed tile route alone and keeps its serial tile).
pub const EXACT_WAVE_MAX_LANE_ROWS: usize = QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP;

/// Kill switch of the wave's JOINT dense stages (default ON): the row-local dense stages of the
/// wave (embed, PLE rows, HC read / write, the GDN and QSA in / out projections) run once over
/// every lane's rows beside the once-over-all-rows MoE; exact `0` restores the incumbent policy
/// (every stage per lane, MoE once). Byte-identical on the burst oracle (G0a, 8 x 149 rows).
pub const QWEN4_EP_WAVE_JOINT_ENV: &str = "HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT";

/// [`QWEN4_EP_WAVE_JOINT_ENV`] parsed: on unless exact `0`.
pub(crate) fn ep_wave_joint_enabled_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

static EP_WAVE_JOINT_KNOB: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    ep_wave_joint_enabled_from(
        hipfire_config::developer_var(QWEN4_EP_WAVE_JOINT_ENV)
            .ok()
            .as_deref(),
    )
});

/// Whether the wave's dense stages run jointly in this process (every rank thread agrees).
pub(crate) fn ep_wave_joint_enabled() -> bool {
    *EP_WAVE_JOINT_KNOB
}

/// The wave's stage shares: `joint` (the default) runs every row-local stage once over all lanes'
/// rows (embed, PLE rows, HC read / write, GDN and QSA in / out projections, the MoE); the
/// stateful stages stay per lane. Not `joint` is the incumbent: every stage per lane (each
/// lane's launches are its singleton tile's by construction), the MoE once over all rows.
pub(crate) fn ep_wave_policy_for(joint: bool) -> LaneStagePolicy {
    let once = if joint {
        &[
            StageName::Embed,
            StageName::PleRows,
            StageName::HcRead,
            StageName::HcWrite,
            StageName::GdnInProj,
            StageName::GdnOutProj,
            StageName::QsaInProj,
            StageName::QsaOutProj,
            StageName::Moe,
        ][..]
    } else {
        &[StageName::Moe][..]
    };
    once.iter()
        .fold(LaneStagePolicy::all_per_lane(), |policy, stage| {
            policy
                .with(*stage, StageShare::OnceAllRows)
                .expect("a joint stage does not read request state")
        })
}

/// The lane-row table a wave hands `Gpu::qwen4_wave_lane_rows`: the lanes' rows when any lane
/// has fewer than `floor` rows (65 by default: within the few-row window, 64 rows; lower with the
/// LAB prefill lane WMMA floor), else empty (every projection of such a wave is the
/// singleton's whatever the call's total).
pub(crate) fn ep_wave_lane_table(lane_rows: &[usize], floor: usize) -> Vec<usize> {
    if lane_rows.iter().any(|&rows| rows < floor) {
        lane_rows.to_vec()
    } else {
        Vec::new()
    }
}

/// The route a wave reproduces: the one every lane's own singleton tile takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpWaveRoute {
    /// The EP FULL-TABLE MoE (every rank computes every expert; no exchange): sub-512 tiles on a
    /// load whose owner-routed floor is 512 (A8 killed, knob 512, no tile transport).
    FullTable,
    /// The owner-routed TILE route (each rank its owned experts, BF16 grouped rows exchanged per
    /// layer) on the A8 chain: sub-512 tiles under the lowered floor (default 9).
    Tile,
    /// The trunk shard's PARTIAL chunk route (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`): the compact table
    /// (non-owned experts the zero dummy, static `e % ranks` owners), no exchange, no gather; the
    /// rank's FFN is its partial, reduced by the all-reduce of each lane's FFN HC write. The
    /// lanes' own partial chunks are not admitted by the owner-routed tile route (floor 512:
    /// `HIPFIRE_EP_PREFILL_TILE_MIN_ROWS=512`, `HIPFIRE_QWEN4_MOE_EP_A8=0`, or the mask off), so
    /// no tile mask is bound.
    Partial,
    /// [`Self::Partial`] with the shard's partial tile mask (`HIPFIRE_QWEN4_EP_TRUNK_SHARD_PMASK`):
    /// every lane's own partial chunk is admitted by the owner-routed tile route
    /// (`ep_tile_active`: the lowered floor on the A8 chain) and masks the non-owned tiles, so
    /// the wave's one MoE call binds the same mask-only exchange (owned experts only, no peer
    /// traffic).
    PartialMask,
}

/// Host facts of one wave, identical on every rank. All of them are decided from replicated
/// state (lane rows, config, flags), never from device contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EpWaveFacts {
    /// Rows of every lane, in lane order.
    pub lane_rows: Vec<usize>,
    /// Per lane: the lane's own rows take the owner-routed tile route (see [`ep_wave_route`]).
    pub lane_tile: Vec<bool>,
    /// Per lane: the A8 chain is the lane's active owner-routed chain (`Gpu::qwen4_moe_ep_a8_requested`
    /// under the lane's own tile scopes).
    pub lane_a8: Vec<bool>,
    /// The EP full tables are built (`attach_ep` ran).
    pub moe_full_ready: bool,
    /// Trunk-shard partial FFN route active: every chunk of 9+ rows binds the rank's compact
    /// table and the all-reduce of its HC writes completes the FFN ([`EpWaveRoute::Partial`]).
    pub ffn_partial: bool,
    /// A non-EP decode-share bank swap or borrowed runtime view is configured.
    pub decode_share: bool,
    /// Some lane or the wave itself enters a non-default routing scope (invariant prefill
    /// profile, agent-turn append route): not the plain sub-512 routes.
    pub scoped_route: bool,
    /// Fewest rows of a lane the wave takes ([`exact_wave_min_lane_rows`]).
    pub min_lane_rows: usize,
    /// FullTable route: rows the selected scratch holds for the F32 grouped chain: the forward's
    /// `max_chunk` in Full scratch mode, `QWEN4_COMPACT_FALLBACK_ROWS` in compact mode; on an EP
    /// reserve load the resident bank's call limit.
    pub capacity_rows: usize,
    /// Tile route: rows its chain holds: the forward's `max_chunk` (the BF16 grouped chain has no
    /// compact-mode limit: a wide singleton chunk runs it in either scratch mode); on an EP
    /// reserve load the resident bank's call limit (a wider wave would take the borrowed bank,
    /// which no wave receipt covers).
    pub tile_capacity_rows: usize,
    /// An EP reserve load refuses the wave's full-table call (`EpEpoch::plan_call` with the
    /// full-table route): its reason.
    pub reserve_refusal: Option<String>,
    /// The tile-route wave is switched on ([`ep_wave_tile_enabled`]); the full-table wave needs
    /// no sub-switch (the master switch decides).
    pub tile_wave_enabled: bool,
    /// The trunk-shard partial-route wave is switched on ([`ep_wave_shard_enabled`]).
    pub shard_wave_enabled: bool,
    /// Per lane: the lane's own partial chunk runs the SYMMETRIC grouped chain
    /// (`Gpu::qwen4_moe_sym_iu4_applies` under the lane's tile scopes), else the F32 grouped
    /// chain. Only read on the Partial route.
    pub lane_sym: Vec<bool>,
    /// Trunk-shard partial tile mask in effect on this load (`HIPFIRE_QWEN4_EP_TRUNK_SHARD_PMASK`
    /// and the installed tile-route object): every lane's own partial chunk that the tile route
    /// admits masks the non-owned tiles, and the wave's one MoE call must mask them the same way.
    /// Only read on the Partial route.
    pub partial_mask: bool,
}

impl EpWaveFacts {
    fn total(&self) -> usize {
        self.lane_rows.iter().sum()
    }
}

/// Why a wave cannot run as one exact joint pass (`None`: admitted). The message starts with
/// `"ep wave refused:"` so a caller can fall back to serial tiles.
pub(crate) fn ep_wave_refusal(facts: &EpWaveFacts) -> Option<String> {
    let refuse = |why: String| Some(format!("ep wave refused: {why}"));
    if facts.lane_rows.len() < 2 {
        return refuse("a wave needs at least two lanes".to_string());
    }
    for (lane, &rows) in facts.lane_rows.iter().enumerate() {
        if !(facts.min_lane_rows..=EXACT_WAVE_MAX_LANE_ROWS).contains(&rows) {
            return refuse(format!(
                "lane {lane} has {rows} rows, outside {}..={EXACT_WAVE_MAX_LANE_ROWS} (its serial tile is not a sub-512-row tile of the route the wave reproduces)",
                facts.min_lane_rows
            ));
        }
    }
    let total = facts.total();
    if total > QWEN4_EP_WAVE_ROW_CAP {
        return refuse(format!(
            "{total} rows exceed the {QWEN4_EP_WAVE_ROW_CAP}-row wave cap"
        ));
    }
    if !facts.moe_full_ready {
        return refuse("the EP full tables are not built (attach_ep has not run)".to_string());
    }
    if facts.decode_share {
        return refuse("a decode-share bank swap is configured on this EP load".to_string());
    }
    if facts.scoped_route {
        return refuse("a routing scope (invariant / append route) is active".to_string());
    }
    match ep_wave_route(facts) {
        Err(why) => refuse(why),
        Ok(EpWaveRoute::FullTable) => {
            if let Some(why) = f32_chain_small_lane(facts) {
                return refuse(why);
            }
            if total > facts.capacity_rows {
                return refuse(format!(
                    "{total} rows exceed the {} rows the selected scratch holds for the F32 grouped chain",
                    facts.capacity_rows
                ));
            }
            if let Some(why) = facts.reserve_refusal.as_ref() {
                return refuse(format!(
                    "the EP reserve refuses the full-table call ({why})"
                ));
            }
            None
        }
        Ok(EpWaveRoute::Tile) => {
            if !facts.tile_wave_enabled {
                return refuse(format!(
                    "the tile-route wave is switched off ({QWEN4_EP_WAVE_TILE_ENV}=0)"
                ));
            }
            if let Some(lane) = facts.lane_a8.iter().position(|a8| !*a8) {
                return refuse(format!(
                    "lane {lane} ({} rows) is on the tile route without the A8 chain; only the A8 tile chain is proven per-slot independent",
                    facts.lane_rows[lane]
                ));
            }
            if total > facts.tile_capacity_rows {
                return refuse(format!(
                    "{total} rows exceed the {} rows the tile route's scratch / resident bank holds",
                    facts.tile_capacity_rows
                ));
            }
            None
        }
        Ok(EpWaveRoute::Partial | EpWaveRoute::PartialMask) => {
            if !facts.shard_wave_enabled {
                return refuse(format!(
                    "the trunk-shard partial-route wave is switched off ({QWEN4_EP_WAVE_SHARD_ENV}=0)"
                ));
            }
            // The singleton partial chunk runs the symmetric chain (floor / A8 scope) or the F32
            // grouped chain by the lane's rows against the floor in scope: one chain for all
            // lanes, else the wide call has no lane's route to reproduce.
            let sym = facts.lane_sym.iter().filter(|sym| **sym).count();
            if sym != 0 && sym != facts.lane_sym.len() {
                let first_sym = facts.lane_sym.iter().position(|s| *s).unwrap_or(0);
                let first_f32 = facts.lane_sym.iter().position(|s| !*s).unwrap_or(0);
                return refuse(format!(
                    "the lanes' own partial chunks run different chains: lane {first_sym} ({} rows) the symmetric chain, lane {first_f32} ({} rows) the F32 grouped chain",
                    facts.lane_rows[first_sym], facts.lane_rows[first_f32]
                ));
            }
            // The partial tile mask is per lane (its own tile-route admission): a wave reproduces
            // lanes that are all masked or all unmasked, never a mix.
            if facts.partial_mask {
                let masked = facts.lane_tile.iter().filter(|tile| **tile).count();
                if masked != 0 && masked != facts.lane_tile.len() {
                    return refuse(
                        "the lanes' own partial chunks differ in the tile mask (some lanes take the owner-routed tile admission, some do not)"
                            .to_string(),
                    );
                }
            }
            if sym != 0 {
                if let Some(lane) = facts.lane_a8.iter().position(|a8| !*a8) {
                    return refuse(format!(
                        "lane {lane} ({} rows) runs the symmetric chain on its partial chunk without the A8 chain; only the A8 chain is proven per-slot independent",
                        facts.lane_rows[lane]
                    ));
                }
            }
            if sym == 0 {
                if let Some(why) = f32_chain_small_lane(facts) {
                    return refuse(why);
                }
            }
            let capacity = if sym != 0 {
                facts.tile_capacity_rows
            } else {
                facts.capacity_rows
            };
            if total > capacity {
                return refuse(format!(
                    "{total} rows exceed the {capacity} rows the partial route's chain scratch / resident bank holds"
                ));
            }
            None
        }
    }
}

/// The small-lane wave (`HIPFIRE_EP_LANE_PREFILL_WAVE_SMALL_LANES`: lanes of 9..=64 rows) is
/// exact only on the symmetric A8 chain (the Tile wave, a Partial wave whose lanes run the
/// symmetric chain). On the F32 grouped chain (the FullTable wave, a Partial wave whose lanes are
/// below the owner-routed floor: floor 512 or `HIPFIRE_QWEN4_MOE_EP_A8=0`) a lane of at most
/// `WIDE_VERIFY_MAX_ROWS` rows runs the 9..=64-row MoE families in its own chunk (the routed-down
/// A8 grouped window, the g32 gate/up), which the wave's one call over every lane's rows does not
/// (j8439 u60x8 0/8; j8706 floor 512 u60x8 0/8, logits differ at row 0), so such lanes keep
/// their serial tile ([`EXACT_WAVE_MIN_LANE_ROWS`]).
fn f32_chain_small_lane(facts: &EpWaveFacts) -> Option<String> {
    let lane = facts
        .lane_rows
        .iter()
        .position(|&rows| rows < EXACT_WAVE_MIN_LANE_ROWS)?;
    Some(format!(
        "lane {lane} ({} rows) is below {EXACT_WAVE_MIN_LANE_ROWS} rows on the F32 grouped chain: its own chunk runs the 9..=64-row MoE families, only the A8 chain's small-lane wave is exact",
        facts.lane_rows[lane]
    ))
}

/// The route a wave takes, decided from its lanes' own routes: every lane's singleton tile on
/// the FULL-TABLE route -> [`EpWaveRoute::FullTable`], every lane's on the owner-routed TILE
/// route -> [`EpWaveRoute::Tile`]; lanes of both kinds (a floor between their rows) have no
/// single route to reproduce. Under the trunk shard: [`EpWaveRoute::PartialMask`] when every
/// lane's own partial chunk takes the partial tile mask, else [`EpWaveRoute::Partial`] (a mix is
/// refused by [`ep_wave_refusal`]).
pub(crate) fn ep_wave_route(facts: &EpWaveFacts) -> Result<EpWaveRoute, String> {
    // Under the trunk shard every chunk of 9+ rows is the PARTIAL chunk route
    // (`ep_trunk_partial_chunk`): it takes precedence over the tile route, whose exchange the
    // shard refuses (it would complete the routed FFN on both ranks). The lanes' own tile flags
    // then decide only the partial tile mask: `forward_chunk_scoped` masks a partial chunk
    // exactly when the mask is on and `ep_tile_active` admits it.
    if facts.ffn_partial {
        let masked = facts.partial_mask
            && !facts.lane_tile.is_empty()
            && facts.lane_tile.iter().all(|tile| *tile);
        return Ok(if masked {
            EpWaveRoute::PartialMask
        } else {
            EpWaveRoute::Partial
        });
    }
    let tiles = facts.lane_tile.iter().filter(|tile| **tile).count();
    if tiles == 0 {
        Ok(EpWaveRoute::FullTable)
    } else if tiles == facts.lane_tile.len() {
        Ok(EpWaveRoute::Tile)
    } else {
        let first_tile = facts.lane_tile.iter().position(|tile| *tile).unwrap_or(0);
        let first_full = facts.lane_tile.iter().position(|tile| !*tile).unwrap_or(0);
        Err(format!(
            "the lanes' own routes differ: lane {first_tile} ({} rows) takes the owner-routed tile route, lane {first_full} ({} rows) the full-table route",
            facts.lane_rows[first_tile], facts.lane_rows[first_full]
        ))
    }
}

/// The runtime a wave's MoE invocation of `layer` binds: exactly the singleton chunk's
/// selection for a sub-512-row tile of the wave's route (`select_layer_moe` with `full_chunk`):
/// [`EpWaveRoute::FullTable`] the layer's EP full runtime of the bank in use (never the balanced
/// live or the decode table), [`EpWaveRoute::Tile`], [`EpWaveRoute::Partial`] and
/// [`EpWaveRoute::PartialMask`] the compact (owned-experts, non-owned = zero dummy) runtime the
/// owner-routed tile exchange / the trunk shard's partial chunk reads.
pub(crate) fn ep_wave_layer_moe<'a>(
    route: EpWaveRoute,
    ep: Option<&'a Qwen4ForwardEp>,
    moe: &'a [Option<Qwen4MoeLayerRuntime>],
    moe_borrow: &'a [Vec<Option<Qwen4MoeLayerRuntime>>],
    borrow_runtimes_active: bool,
    borrow_view: usize,
    layer: usize,
) -> Option<&'a Qwen4MoeLayerRuntime> {
    let tile_moe = tile_moe_of(borrow_runtimes_active, borrow_view, moe, moe_borrow);
    select_layer_moe(
        None,
        ep.map(|ep| ep.moe_full.get(layer)),
        true,
        route != EpWaveRoute::FullTable,
        tile_moe.get(layer).and_then(Option::as_ref),
    )
}

/// The owner-routed tile exchange a Tile-route wave binds to every MoE invocation: the
/// installed transport and this rank's masked tile ids, exactly the singleton tile chunk's
/// `MoeTileExchangeParams`. `None` without an installed transport.
pub(crate) fn ep_wave_tile_exchange(
    ep: Option<&Qwen4ForwardEp>,
) -> Option<hipfire_dispatch::families::moe::MoeTileExchangeParams<'_>> {
    let route = ep?.tile.as_ref()?;
    Some(hipfire_dispatch::families::moe::MoeTileExchangeParams {
        exchange: route.hook.as_deref()?,
        tile_ids_owned: &route.tile_ids_owned,
    })
}

/// Whether the trunk-shard partial tile mask is in effect: the switch and the installed tile
/// route object (the same pair [`Qwen4GpuForward::forward_chunk_tile`] checks per chunk).
pub(crate) fn ep_wave_partial_mask_on(ep: Option<&Qwen4ForwardEp>) -> bool {
    super::trunk_shard_pmask_env_on()
        && crate::trunk_route::ffn_partial_active()
        && ep.is_some_and(|ep| ep.tile.as_ref().is_some_and(|route| route.hook.is_some()))
}

/// The mask-only exchange of the [`EpWaveRoute::PartialMask`] wave's MoE call (see
/// [`super::PartialMaskArc`]) and the owned-tile id buffer it masks with.
pub(crate) fn ep_wave_partial_mask(
    ep: Option<&Qwen4ForwardEp>,
) -> Option<hipfire_dispatch::families::moe::MoeTileExchangeParams<'_>> {
    let route = ep?.tile.as_ref()?;
    Some(hipfire_dispatch::families::moe::MoeTileExchangeParams {
        exchange: route.mask_hook.as_deref()?,
        tile_ids_owned: &route.tile_ids_owned,
    })
}

/// Rows of the routing scopes ([`Qwen4GpuForward::enter_tile_scopes`]) a wave runs under: the
/// widest LANE, never the wave total. The scopes a tile enters key on its rows (the A8 chain is
/// the default only below 512 rows, the sub-512 floor follows it), and the wave must run under
/// exactly the scopes every lane's own singleton tile enters; the total (up to 2048 rows) would
/// select the >= 512-row A4 chain and the 512 floor.
pub(crate) fn ep_wave_scope_rows(lane_rows: &[usize]) -> usize {
    lane_rows.iter().copied().max().unwrap_or(0)
}

/// The wave program's MoE invocation count: one per MoE layer of the band, never one per lane.
pub(crate) fn ep_wave_expected_moe_calls(moe_layers: usize) -> usize {
    moe_layers
}

/// Rows the Tile route's chain holds (see [`EpWaveFacts::tile_capacity_rows`]): the forward's
/// `max_chunk` whatever the scratch mode, on an EP reserve load at most the resident bank.
pub(crate) fn ep_wave_tile_rows(
    max_chunk: usize,
    reserve_resident_call_limit: Option<usize>,
) -> usize {
    reserve_resident_call_limit.map_or(max_chunk, |limit| max_chunk.min(limit))
}

/// Rows the F32 grouped chain's scratch holds (see [`EpWaveFacts::capacity_rows`]).
pub(crate) fn ep_wave_scratch_rows(
    mode: Qwen4ForwardScratchMode,
    max_chunk: usize,
    reserve_resident_call_limit: Option<usize>,
) -> usize {
    let by_mode = match mode {
        Qwen4ForwardScratchMode::Full => max_chunk,
        Qwen4ForwardScratchMode::FnSymGfx1201 => max_chunk.min(QWEN4_COMPACT_FALLBACK_ROWS),
    };
    match reserve_resident_call_limit {
        Some(limit) => by_mode.min(limit),
        None => by_mode,
    }
}

impl Qwen4GpuForward {
    /// The host facts of a wave of `lane_rows` on this forward. Touches no device state; the
    /// routing scopes are entered and left per probe.
    pub(crate) fn ep_wave_facts(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        lane_rows: &[usize],
    ) -> EpWaveFacts {
        let total: usize = lane_rows.iter().sum();
        let prior_prefill = std::mem::replace(&mut self.prefill_scope, true);
        // (tile route, any scope entered, A8 chain, symmetric chain) of `rows` rows under the
        // routing scopes a prefill tile of those rows enters.
        let probe = |this: &Self, gpu: &mut Gpu, rows: usize| -> (bool, bool, bool, bool) {
            let (pm_invariant, append) = Self::route_scopes(this.prefill_scope, gpu, rows);
            let (saved, _) = this.enter_tile_scopes(gpu, rows);
            let route_rows = gpu.route_rows(rows);
            let tile =
                ep_reserve_forward::ep_tile_active(this.ep.as_ref(), gpu, config, rows, route_rows);
            let a8 = gpu.qwen4_moe_ep_a8_requested();
            let sym = gpu.qwen4_moe_sym_iu4_applies(rows);
            Self::leave_tile_scopes(gpu, saved);
            (tile, pm_invariant || append, a8, sym)
        };
        let mut scoped_route = probe(self, gpu, total).1;
        let mut lane_tile = Vec::with_capacity(lane_rows.len());
        let mut lane_a8 = Vec::with_capacity(lane_rows.len());
        let mut lane_sym = Vec::with_capacity(lane_rows.len());
        for &rows in lane_rows {
            let (tile, scoped, a8, sym) = probe(self, gpu, rows);
            lane_tile.push(tile);
            lane_a8.push(a8);
            lane_sym.push(sym);
            scoped_route |= scoped;
        }
        self.prefill_scope = prior_prefill;
        let reserve = self.ep_reserve.as_ref();
        let resident_limit = reserve.map(|reserve| reserve.banks.resident_call_limit());
        EpWaveFacts {
            lane_rows: lane_rows.to_vec(),
            lane_tile,
            lane_a8,
            moe_full_ready: self
                .ep
                .as_ref()
                .is_some_and(|ep| ep.moe_full.len() == self.moe_count()),
            ffn_partial: crate::trunk_route::ffn_partial_active(),
            decode_share: reserve.is_none()
                && (self.share.is_some() || !self.moe_borrow.is_empty()),
            scoped_route,
            min_lane_rows: exact_wave_min_lane_rows(),
            capacity_rows: ep_wave_scratch_rows(
                self.scratch.mode,
                self.scratch.max_chunk,
                resident_limit,
            ),
            tile_capacity_rows: ep_wave_tile_rows(self.scratch.max_chunk, resident_limit),
            reserve_refusal: reserve.and_then(|reserve| {
                reserve
                    .epoch
                    .plan_call(
                        total,
                        reserve.banks.resident_call_limit(),
                        reserve.banks.prompt_rows(),
                        EpBankRoute::FullTable,
                    )
                    .err()
            }),
            tile_wave_enabled: ep_wave_tile_enabled(),
            shard_wave_enabled: ep_wave_shard_enabled(),
            lane_sym,
            partial_mask: ep_wave_partial_mask_on(self.ep.as_ref()),
        }
    }

    /// What every full-table forward of the singleton chunk does on an EP load before it
    /// launches, once per forward on every rank in lockstep: the installed exchange waits for the
    /// peer's last pull of this rank's rows (a full-table chunk also writes `y_down_grouped`),
    /// and a load with an expert stage drains it and crosses the host barrier so no rank reads a
    /// donor the stage may still overwrite.
    fn ep_wave_prelude(&mut self, gpu: &mut Gpu) -> Result<(), Qwen4GpuForwardError> {
        let hook = self
            .ep
            .as_ref()
            .and_then(|ep| ep.tile.as_ref())
            .and_then(|route| route.hook.clone());
        let Some(hook) = hook else {
            return Ok(());
        };
        hook.before_down(gpu)
            .map_err(|error| invalid(format!("EP tile exchange: {error}")))?;
        if let Some(stage) = self.expert_stage.as_ref() {
            stage.ep_quiesce(gpu)?;
            hook.host_barrier(
                hipfire_runtime::ep_tile_exchange::HOST_STAGE_QUIESCE,
                self.ep_reserve_epoch(),
            )
            .map_err(|error| invalid(format!("EP stage quiesce barrier: {error}")))?;
        }
        Ok(())
    }

    /// Run a wave's forward `run` over `n` total rows as a lane prefill tile of `route`: lane and
    /// prefill scope set, the whole-prompt entry bracketed (nest-safe: the caller's own bracket
    /// makes this one inner), the routing scopes of a prefill tile entered for `scope_rows` (the
    /// widest LANE, [`ep_wave_scope_rows`]), on the full-table route the singleton's prelude, the
    /// EP reserve bank switch (restore of a dirty epoch, resident bank) taken exactly as a
    /// singleton tile of that route takes it, the routing-row cap set for the call, and
    /// everything put back on every return path.
    pub(crate) fn ep_wave_run(
        &mut self,
        gpu: &mut Gpu,
        n: usize,
        scope_rows: usize,
        lane_rows: &[usize],
        route: EpWaveRoute,
        run: impl FnOnce(&mut Self, &mut Gpu) -> Result<(), Qwen4GpuForwardError>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.share_check()?;
        let prior_prefill = std::mem::replace(&mut self.prefill_scope, true);
        let prior_lane = std::mem::replace(&mut self.lane_scope, true);
        let prior_route = self.wave_route.replace(route);
        self.share_prompt_begin();
        self.ep_ordinary_note_call(n);
        let (scopes, _) = self.enter_tile_scopes(gpu, scope_rows);
        if route == EpWaveRoute::FullTable {
            // The wave's wide MoE call is a full-table call: the sub-512 floor of the owner-routed
            // tile route (`ep_prefill_tile_sym_floor`) never reaches it. An admitted full-table
            // wave has no lane on the tile route, but a floor between the lanes' rows and the cap
            // (knob 100, lanes of 60 rows) would still let the routing cap's 511 rows select the
            // symmetric arm that no lane's own singleton takes; `leave_tile_scopes` restores it.
            // The Tile route keeps the floor of the lanes' own scopes (every lane is at or above
            // it, so is the cap).
            gpu.qwen4_moe_sym_iu4_floor = rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS;
        }
        let prior_cap = std::mem::replace(
            &mut gpu.qwen4_route_rows_cap,
            QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP,
        );
        // The wave's lane-row table for the BF16 projections of its wide calls (see
        // [`QWEN4_EP_WAVE_SMALL_LANES_ENV`]): set only when a small lane is in the wave, cleared on
        // every return path below (the forward runs in the closure, no `?` between here and the
        // restore after it).
        let prior_lane_rows = std::mem::replace(
            &mut gpu.qwen4_wave_lane_rows,
            ep_wave_lane_table(lane_rows, gpu.qwen4_prefill_lwmma_floor),
        );
        let bank_route = match route {
            EpWaveRoute::FullTable => EpBankRoute::FullTable,
            EpWaveRoute::Tile | EpWaveRoute::Partial | EpWaveRoute::PartialMask => {
                EpBankRoute::TileExchange
            }
        };
        let result = (|| {
            if self.ep_reserve.is_some() {
                self.ep_reserve_call(gpu, n, bank_route, false, |this, gpu| {
                    if route == EpWaveRoute::FullTable {
                        this.ep_wave_prelude(gpu)?;
                    }
                    run(this, gpu)
                })
            } else {
                if route == EpWaveRoute::FullTable {
                    self.ep_wave_prelude(gpu)?;
                }
                run(self, gpu)
            }
        })();
        gpu.qwen4_route_rows_cap = prior_cap;
        gpu.qwen4_wave_lane_rows = prior_lane_rows;
        Self::leave_tile_scopes(gpu, scopes);
        let ended = self.share_prompt_end(gpu, result.is_ok());
        self.prefill_scope = prior_prefill;
        self.lane_scope = prior_lane;
        self.wave_route = prior_route;
        merge_share_result(result, ended)
    }

    /// Record the route a wave took as the lane tile note of its total rows (the
    /// `HIPFIRE_CB_PHASES` per-tile line reads it; it never feeds a decision).
    pub(crate) fn ep_wave_note(&mut self, rows: usize) {
        let route = match self.wave_route {
            Some(EpWaveRoute::Tile) => ep_reserve_forward::LaneTileRoute::Tile,
            Some(EpWaveRoute::PartialMask) => ep_reserve_forward::LaneTileRoute::PartialMask,
            Some(EpWaveRoute::Partial) => ep_reserve_forward::LaneTileRoute::Partial,
            Some(EpWaveRoute::FullTable) | None => ep_reserve_forward::LaneTileRoute::FullTable,
        };
        self.lane_tile_note = Some(ep_reserve_forward::LaneTileNote {
            rows,
            route,
            staged: false,
            stage_min_rows: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(lane_rows: &[usize]) -> EpWaveFacts {
        EpWaveFacts {
            lane_rows: lane_rows.to_vec(),
            lane_tile: vec![false; lane_rows.len()],
            lane_a8: vec![true; lane_rows.len()],
            moe_full_ready: true,
            ffn_partial: false,
            decode_share: false,
            scoped_route: false,
            min_lane_rows: EXACT_WAVE_MIN_LANE_ROWS,
            capacity_rows: 4096,
            tile_capacity_rows: 4096,
            reserve_refusal: None,
            tile_wave_enabled: true,
            shard_wave_enabled: true,
            lane_sym: vec![false; lane_rows.len()],
            partial_mask: false,
        }
    }

    fn refused(facts: &EpWaveFacts) -> String {
        ep_wave_refusal(facts).expect("the wave must be refused")
    }

    #[test]
    fn the_burst_cells_are_admitted_without_any_row_floor() {
        // The code C8 burst, the story C8 burst (480 rows: below every 512 floor), a pair.
        assert_eq!(ep_wave_refusal(&facts(&[149; 8])), None);
        assert_eq!(ep_wave_refusal(&facts(&[65; 8])), None);
        assert_eq!(ep_wave_refusal(&facts(&[65, 65])), None);
        // Lanes of at most 64 rows keep their serial tile (the wave is not exact for them).
        assert!(refused(&facts(&[60; 8])).contains("lane 0"));
        assert!(refused(&facts(&[49, 49])).contains("lane 0"));
        assert_eq!(
            ep_wave_refusal(&facts(&[65, 65, 149, 149, 149, 256, 256, 511])),
            None
        );
        // The wave cap: 2048 fits, 2049 does not.
        assert_eq!(ep_wave_refusal(&facts(&[511, 511, 511, 300, 215])), None);
        let mut over = facts(&[511, 511, 511, 300, 216]);
        over.capacity_rows = 8192;
        assert!(refused(&over).contains("wave cap"), "{}", refused(&over));
    }

    #[test]
    fn the_min_lane_rows_knob_only_lowers_to_the_few_row_boundary() {
        // The default is one more than the widest few-row window (65 on gfx1201), not a literal.
        assert_eq!(
            EXACT_WAVE_MIN_LANE_ROWS,
            rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS + 1
        );
        assert_eq!(
            exact_wave_lane_floor_from(None, false),
            EXACT_WAVE_MIN_LANE_ROWS
        );
        for default in [
            None,
            Some(""),
            Some("x"),
            Some("8"),
            Some("0"),
            Some("512"),
            Some("-1"),
        ] {
            assert_eq!(
                exact_wave_lane_floor_from(default, false),
                EXACT_WAVE_MIN_LANE_ROWS,
                "{default:?}"
            );
        }
        assert_eq!(exact_wave_lane_floor_from(Some("9"), false), 9);
        assert_eq!(exact_wave_lane_floor_from(Some(" 24 "), false), 24);
        assert_eq!(exact_wave_lane_floor_from(Some("511"), false), 511);
        // The process default (knob unset in the test environment) is the const. The knob
        // lowers the floor of the A8 tile chain's wave; the F32 grouped chain (FullTable) keeps
        // its 65-row lanes whatever it says.
        let mut f = facts(&[9, 24]);
        f.lane_tile = vec![true; 2];
        f.min_lane_rows = 9;
        assert_eq!(ep_wave_refusal(&f), None);
        f.min_lane_rows = EXACT_WAVE_MIN_LANE_ROWS;
        assert!(refused(&f).contains("lane 0"));
        f.min_lane_rows = 9;
        f.lane_tile = vec![false; 2];
        assert!(refused(&f).contains("lane 0 (9 rows) is below 65 rows on the F32 grouped chain"));
    }

    #[test]
    fn lane_rows_outside_the_exact_window_refuse_the_wave() {
        for rows in [0usize, 1, 8, 9, 48, 49, 64, 512, 513, 1024] {
            let f = facts(&[149, rows]);
            assert!(
                refused(&f).contains("lane 1"),
                "rows {rows}: {}",
                refused(&f)
            );
        }
        for rows in [65usize, 66, 149, 511] {
            assert_eq!(ep_wave_refusal(&facts(&[149, rows])), None, "rows {rows}");
        }
        assert_eq!(EXACT_WAVE_MIN_LANE_ROWS, 65);
        assert_eq!(EXACT_WAVE_MAX_LANE_ROWS, 511);
    }

    #[test]
    fn one_lane_or_none_is_not_a_wave() {
        assert!(refused(&facts(&[])).contains("at least two lanes"));
        assert!(refused(&facts(&[149])).contains("at least two lanes"));
    }

    #[test]
    fn the_small_lanes_switch_is_on_unless_exact_zero() {
        for on in [
            None,
            Some(""),
            Some("1"),
            Some("on"),
            Some("true"),
            Some("2"),
        ] {
            assert!(ep_wave_small_lanes_enabled_from(on), "{on:?}");
        }
        for off in [Some("0"), Some(" 0\n")] {
            assert!(!ep_wave_small_lanes_enabled_from(off), "{off:?}");
        }
        assert_eq!(
            QWEN4_EP_WAVE_SMALL_LANES_ENV,
            "HIPFIRE_EP_LANE_PREFILL_WAVE_SMALL_LANES"
        );
    }

    #[test]
    fn the_lane_floor_is_9_with_small_lanes_65_without_and_an_explicit_value_wins() {
        assert_eq!(exact_wave_lane_floor_from(None, true), SMALL_LANE_MIN_ROWS);
        assert_eq!(SMALL_LANE_MIN_ROWS, 9);
        assert_eq!(
            exact_wave_lane_floor_from(None, false),
            EXACT_WAVE_MIN_LANE_ROWS
        );
        for bad in [
            Some(""),
            Some("x"),
            Some("8"),
            Some("0"),
            Some("512"),
            Some("-1"),
        ] {
            assert_eq!(exact_wave_lane_floor_from(bad, true), 9, "{bad:?}");
            assert_eq!(exact_wave_lane_floor_from(bad, false), 65, "{bad:?}");
        }
        // An explicit MIN_ROWS overrides the switch either way.
        assert_eq!(exact_wave_lane_floor_from(Some("65"), true), 65);
        assert_eq!(exact_wave_lane_floor_from(Some("100"), true), 100);
        assert_eq!(exact_wave_lane_floor_from(Some("20"), false), 20);
    }

    #[test]
    fn joint_dense_stages_need_every_lane_wider_than_the_few_row_window() {
        let floor = rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS + 1;
        assert!(ep_wave_joint_for_lanes(true, &[65, 149, 511], floor));
        assert!(!ep_wave_joint_for_lanes(true, &[60, 149], floor));
        assert!(!ep_wave_joint_for_lanes(true, &[64, 64], floor));
        assert!(!ep_wave_joint_for_lanes(false, &[149, 149], floor));
        assert!(ep_wave_joint_for_lanes(true, &[], floor));
    }

    /// `HIPFIRE_QWEN4_PREFILL_LWMMA_WAVE_FLOOR` (49): lanes of 49..=64 rows join the joint stages;
    /// a lane below the floor still keeps every row-local stage per lane.
    #[test]
    fn a_lowered_prefill_floor_admits_49_to_64_row_lanes_to_the_joint_stages() {
        assert!(ep_wave_joint_for_lanes(true, &[60; 8], 49));
        assert!(ep_wave_joint_for_lanes(true, &[49, 64, 149], 49));
        assert!(!ep_wave_joint_for_lanes(true, &[48, 60], 49));
        assert!(!ep_wave_joint_for_lanes(false, &[60; 8], 49));
        assert!(ep_wave_lane_table(&[60; 8], 49).is_empty());
        assert_eq!(ep_wave_lane_table(&[40, 60, 149], 49), vec![40, 60, 149]);
    }

    #[test]
    fn the_lane_table_is_handed_over_only_with_a_lane_of_at_most_64_rows() {
        let floor = rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS + 1;
        assert_eq!(ep_wave_lane_table(&[60; 8], floor), vec![60; 8]);
        assert_eq!(
            ep_wave_lane_table(&[149, 64, 149], floor),
            vec![149, 64, 149]
        );
        assert!(ep_wave_lane_table(&[65, 149], floor).is_empty());
        assert!(ep_wave_lane_table(&[], floor).is_empty());
    }

    #[test]
    fn every_refusal_is_named_distinct_and_deterministic() {
        let mut cases: Vec<EpWaveFacts> = Vec::new();
        cases.push(facts(&[149]));
        cases.push(facts(&[149, 8]));
        cases.push(facts(&[511, 511, 511, 511, 511]));
        let mut f = facts(&[200, 200]);
        f.moe_full_ready = false;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.ffn_partial = true;
        f.shard_wave_enabled = false;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.ffn_partial = true;
        f.lane_sym = vec![true, false];
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.ffn_partial = true;
        f.lane_sym = vec![true, true];
        f.lane_a8 = vec![true, false];
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.ffn_partial = true;
        f.capacity_rows = 399;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.decode_share = true;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.scoped_route = true;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.lane_tile = vec![false, true];
        f.lane_a8 = vec![true, true];
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.lane_tile = vec![true, true];
        f.lane_a8 = vec![true, false];
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.lane_tile = vec![true, true];
        f.tile_capacity_rows = 399;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.lane_tile = vec![true, true];
        f.tile_wave_enabled = false;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.capacity_rows = 399;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.reserve_refusal = Some("a 400-row call would read Full peer tables".to_string());
        cases.push(f);
        let mut reasons = Vec::new();
        for f in &cases {
            let first = refused(f);
            assert!(first.starts_with("ep wave refused:"), "{first}");
            // The other rank sees the same facts and the same verdict, byte for byte.
            assert_eq!(refused(&f.clone()), first);
            reasons.push(first);
        }
        reasons.sort();
        reasons.dedup();
        assert_eq!(reasons.len(), cases.len(), "{reasons:#?}");
    }

    /// The sub-512 owner-routed floor (`HIPFIRE_EP_PREFILL_TILE_MIN_ROWS`, default 9 on the A8
    /// chain) decides each lane's own route: `rows >= max(floor, few-row limit + 1)` on a load with
    /// the tile hook (the pure admission `Gpu::qwen4_moe_sym_iu4_applies` evaluates with the
    /// floor in scope). Lanes all on the tile route are the Tile wave, none the FullTable wave,
    /// a mix has no route and the wave is refused.
    #[test]
    fn the_floor_decides_the_wave_route_and_a_mix_refuses() {
        use super::ep_reserve_forward::ep_prefill_tile_sym_floor;
        let few = rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS;
        let facts_under = |knob: Option<usize>, hook: bool, a8: bool, lanes: &[usize]| {
            let floor = ep_prefill_tile_sym_floor(knob, true, hook, a8);
            let mut f = facts(lanes);
            f.lane_tile = lanes
                .iter()
                .map(|&rows| hook && rows >= floor.max(few + 1))
                .collect();
            f.lane_a8 = vec![a8; lanes.len()];
            f
        };
        // Default (unset knob) on the A8 chain: every 65..=511-row lane is on the tile route and
        // the wave is the Tile wave.
        for lanes in [
            &[149usize; 8][..],
            &[65; 8],
            &[65, 511],
            &[65, 65, 149, 256],
        ] {
            let f = facts_under(None, true, true, lanes);
            assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::Tile), "{lanes:?}");
            assert_eq!(ep_wave_refusal(&f), None, "{lanes:?}");
        }
        // A8 killed (floor 512), the floor kill switch, no tile hook, not a prefill forward:
        // today's full-table lanes, the FullTable wave.
        for f in [
            facts_under(None, true, false, &[149; 8]),
            facts_under(Some(512), true, true, &[149; 8]),
            facts_under(None, false, true, &[149; 8]),
        ] {
            assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::FullTable));
            assert_eq!(ep_wave_refusal(&f), None);
        }
        // A lab floor above every lane keeps them all on the full table; a floor in the middle
        // of the burst splits it into two routes: refused, naming one lane of each route.
        let f = facts_under(Some(300), true, true, &[65, 149, 256]);
        assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::FullTable));
        assert_eq!(ep_wave_refusal(&f), None);
        let f = facts_under(Some(150), true, true, &[65, 149, 150]);
        let why = refused(&f);
        assert!(
            why.contains("differ")
                && why.contains("lane 2 (150 rows) takes the owner-routed tile route")
                && why.contains("lane 0 (65 rows) the full-table route"),
            "{why}"
        );
        // A lab floor of 100 on the A4 chain puts the lanes at or above it on the tile route
        // without A8: refused (only the A8 tile chain is proven per-slot independent).
        let f = facts_under(Some(100), true, false, &[149, 256]);
        assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::Tile));
        assert!(
            refused(&f).contains("without the A8 chain"),
            "{}",
            refused(&f)
        );
    }

    /// The tile-route wave is off until its card receipt: unset (and anything but exact `1`) it is
    /// refused by name and the burst runs serial tile-route tiles; the full-table wave is not
    /// gated by it.
    #[test]
    fn the_tile_wave_has_its_own_kill_switch_and_the_full_table_wave_does_not() {
        for on in [
            None,
            Some(""),
            Some("1"),
            Some(" 1 "),
            Some("true"),
            Some("2"),
        ] {
            assert!(ep_wave_tile_enabled_from(on), "{on:?}");
        }
        assert!(!ep_wave_tile_enabled_from(Some("0")));
        assert!(!ep_wave_tile_enabled_from(Some(" 0\n")));
        assert_eq!(QWEN4_EP_WAVE_TILE_ENV, "HIPFIRE_EP_LANE_PREFILL_WAVE_TILE");
        let mut tile = facts(&[149; 8]);
        tile.lane_tile = vec![true; 8];
        assert_eq!(ep_wave_refusal(&tile), None);
        tile.tile_wave_enabled = false;
        assert!(
            refused(&tile).contains(QWEN4_EP_WAVE_TILE_ENV),
            "{}",
            refused(&tile)
        );
        let mut full = facts(&[149; 8]);
        full.tile_wave_enabled = false;
        assert_eq!(ep_wave_refusal(&full), None);
    }

    /// Under the trunk shard every 9+-row chunk is the partial route whatever the tile flags say
    /// (a reserve-sharing shard load installs the tile transport only as a rendezvous carrier):
    /// the wave is the Partial wave, its chain is the lanes' own (symmetric A8 or F32 grouped),
    /// one chain for all lanes, with that chain's capacity, behind its own sub-switch.
    #[test]
    fn the_shard_wave_is_the_partial_route_and_follows_the_lanes_own_chain() {
        for on in [None, Some(""), Some("1"), Some("true"), Some("2")] {
            assert!(ep_wave_shard_enabled_from(on), "{on:?}");
        }
        assert!(!ep_wave_shard_enabled_from(Some("0")));
        assert_eq!(
            QWEN4_EP_WAVE_SHARD_ENV,
            "HIPFIRE_EP_LANE_PREFILL_WAVE_SHARD"
        );
        let mut f = facts(&[149; 8]);
        f.ffn_partial = true;
        // The carrier tile hook may flag the lanes as tile lanes: the route is still Partial.
        f.lane_tile = vec![true; 8];
        assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::Partial));
        // The partial tile mask follows the lanes' own partial chunks: every lane admitted by the
        // tile route (floor 9, A8) -> the PartialMask wave; with the floor at 512
        // (`HIPFIRE_EP_PREFILL_TILE_MIN_ROWS=512`, `HIPFIRE_QWEN4_MOE_EP_A8=0`) no lane is, and the
        // wave binds no mask (the lanes' F32 grouped chain has none: binding it was a dispatch
        // error); a mix is refused.
        f.partial_mask = true;
        assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::PartialMask));
        assert_eq!(ep_wave_refusal(&f), None);
        f.lane_tile = vec![false; 8];
        assert_eq!(ep_wave_route(&f), Ok(EpWaveRoute::Partial));
        assert_eq!(ep_wave_refusal(&f), None);
        f.lane_tile[4] = true;
        assert!(refused(&f).contains("tile mask"), "{}", refused(&f));
        // Small lanes (60 rows, `HIPFIRE_EP_LANE_PREFILL_WAVE_SMALL_LANES`) join the masked wave on
        // the symmetric A8 chain only; the unmasked Partial wave on the F32 grouped chain refuses
        // them (j8706: floor 512, u60x8 0/8 lane == singleton).
        let mut s = facts(&[60; 8]);
        s.ffn_partial = true;
        s.min_lane_rows = 9;
        s.partial_mask = true;
        s.lane_tile = vec![true; 8];
        s.lane_sym = vec![true; 8];
        assert_eq!(ep_wave_route(&s), Ok(EpWaveRoute::PartialMask));
        assert_eq!(ep_wave_refusal(&s), None);
        s.lane_tile = vec![false; 8];
        s.lane_sym = vec![false; 8];
        assert_eq!(ep_wave_route(&s), Ok(EpWaveRoute::Partial));
        assert!(refused(&s).contains("F32 grouped chain"), "{}", refused(&s));
        f.lane_tile = vec![true; 8];
        f.partial_mask = false;
        // F32 grouped chain (D1 load: floor 512): the compact-mode capacity binds.
        assert_eq!(ep_wave_refusal(&f), None);
        f.capacity_rows = 512;
        assert!(
            refused(&f).contains("partial route's chain scratch"),
            "{}",
            refused(&f)
        );
        // Symmetric A8 chain on the partial chunk (reserve-sharing shard load, floor 9): the
        // tile chain's capacity applies, A8 is required of every lane, a mix is refused.
        f.lane_sym = vec![true; 8];
        assert_eq!(ep_wave_refusal(&f), None);
        f.lane_a8[3] = false;
        assert!(refused(&f).contains("lane 3"), "{}", refused(&f));
        f.lane_a8 = vec![true; 8];
        f.lane_sym[5] = false;
        assert!(
            refused(&f).contains("different chains") && refused(&f).contains("lane 0"),
            "{}",
            refused(&f)
        );
        // On by default (kill switch =0); the master switch / planner rules still hold.
        f.lane_sym = vec![false; 8];
        f.capacity_rows = 4096;
        f.shard_wave_enabled = false;
        assert!(refused(&f).contains(QWEN4_EP_WAVE_SHARD_ENV));
        f.shard_wave_enabled = true;
        f.lane_rows[2] = 512;
        assert!(refused(&f).contains("lane 2"), "{}", refused(&f));
    }

    #[test]
    fn the_tile_wave_has_its_own_capacity_and_ignores_the_full_table_ones() {
        let mut f = facts(&[300, 300]);
        f.lane_tile = vec![true; 2];
        // The F32 grouped chain's capacity (compact scratch: 512 rows) and the full-table
        // reserve refusal are the FullTable wave's, not the Tile wave's.
        f.capacity_rows = 512;
        f.reserve_refusal = Some("full peer tables".to_string());
        assert_eq!(ep_wave_refusal(&f), None);
        f.tile_capacity_rows = 599;
        assert!(
            refused(&f).contains("tile route's scratch"),
            "{}",
            refused(&f)
        );
        f.tile_capacity_rows = 600;
        assert_eq!(ep_wave_refusal(&f), None);
        assert_eq!(ep_wave_tile_rows(4096, None), 4096);
        assert_eq!(ep_wave_tile_rows(4096, Some(512)), 512);
        assert_eq!(ep_wave_tile_rows(256, Some(512)), 256);
        // Trunk shard is the Partial route, not a refusal: it needs its own sub-switch.
        let mut g = facts(&[300, 300]);
        g.ffn_partial = true;
        assert_eq!(ep_wave_route(&g), Ok(EpWaveRoute::Partial));
        assert_eq!(ep_wave_refusal(&g), None);
        g.shard_wave_enabled = false;
        assert!(
            refused(&g).contains(QWEN4_EP_WAVE_SHARD_ENV),
            "{}",
            refused(&g)
        );
    }

    /// The Tile wave runs under the scopes of its lanes' own tiles: `enter_tile_scopes` is entered
    /// with the widest lane's rows, never the total (a total of 512+ rows would select the A4
    /// chain and the 512 floor, the chain no lane's singleton takes).
    #[test]
    fn the_wave_enters_the_tile_scopes_with_the_widest_lane_not_the_total() {
        assert_eq!(ep_wave_scope_rows(&[149; 8]), 149);
        assert_eq!(ep_wave_scope_rows(&[65, 65, 149, 256, 511]), 511);
        assert!(ep_wave_scope_rows(&[511; 4]) < rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS);
        assert_eq!(ep_wave_scope_rows(&[]), 0);
        let src = include_str!("ep_wave.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(body.contains("self.enter_tile_scopes(gpu, scope_rows)"));
        assert!(!body.contains("self.enter_tile_scopes(gpu, n)"));
        let lanes = include_str!("../gpu_forward_lanes.rs");
        assert!(lanes.contains("ep_wave_scope_rows(&lane_rows)"));
        assert!(lanes.contains("fwd.ep_wave_run(gpu, total, scope_rows, &lane_rows, route,"));
    }

    #[test]
    fn scratch_capacity_follows_the_mode_and_the_reserve_bank() {
        use Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
        assert_eq!(ep_wave_scratch_rows(Full, 4096, None), 4096);
        assert_eq!(ep_wave_scratch_rows(FnSymGfx1201, 4096, None), 512);
        assert_eq!(ep_wave_scratch_rows(FnSymGfx1201, 300, None), 300);
        // A reserve load runs the resident bank only.
        assert_eq!(ep_wave_scratch_rows(Full, 4096, Some(512)), 512);
        assert_eq!(ep_wave_scratch_rows(Full, 256, Some(512)), 256);
    }

    /// The reserve epoch a wave drives is the singleton full-table tile's: a clean epoch passes on
    /// the resident bank with no barrier, a dirty one restores globally first, and a call wider
    /// than the resident bank can never run the full-table route.
    #[test]
    fn the_reserve_epoch_serves_a_full_table_wave_like_a_full_table_tile() {
        use super::super::ep_reserve_forward::EpEpoch;
        use super::super::ep_reserve_scratch::EpBankKind;
        let (resident, prompt) = (512usize, 4096usize);
        let full = EpBankRoute::FullTable;
        let mut epoch = EpEpoch::default();
        let clean = epoch.plan_call(480, resident, prompt, full).unwrap();
        assert_eq!(
            (clean.bank, clean.restore_first, clean.acquire),
            (EpBankKind::Resident, false, false)
        );
        epoch.acquired();
        let dirty = epoch.plan_call(480, resident, prompt, full).unwrap();
        assert_eq!(
            (dirty.bank, dirty.restore_first, dirty.acquire),
            (EpBankKind::Resident, true, false)
        );
        let error = epoch.plan_call(1192, resident, prompt, full).unwrap_err();
        assert!(error.contains("Full peer tables"), "{error}");
    }

    #[test]
    fn the_wave_policy_is_joint_by_default_and_per_lane_with_the_moe_once_when_killed() {
        let joint = [
            StageName::Embed,
            StageName::PleRows,
            StageName::HcRead,
            StageName::HcWrite,
            StageName::GdnInProj,
            StageName::GdnOutProj,
            StageName::QsaInProj,
            StageName::QsaOutProj,
            StageName::Moe,
        ];
        for stage in StageName::ALL {
            // Incumbent (`=0`): only the MoE once.
            let want = if stage == StageName::Moe {
                StageShare::OnceAllRows
            } else {
                StageShare::PerLane
            };
            assert_eq!(
                ep_wave_policy_for(false).share(stage),
                want,
                "{}",
                stage.key()
            );
            // Joint (default): the dense row-local stages and the MoE once; the stateful cores,
            // PLE conv and the tail stages stay per lane.
            let want = if joint.contains(&stage) {
                StageShare::OnceAllRows
            } else {
                StageShare::PerLane
            };
            assert_eq!(
                ep_wave_policy_for(true).share(stage),
                want,
                "{}",
                stage.key()
            );
        }
        assert_eq!(ep_wave_policy_for(true).validate(), Ok(()));
        assert_eq!(ep_wave_policy_for(false).validate(), Ok(()));
        assert_eq!(ep_wave_expected_moe_calls(48), 48);
    }

    #[test]
    fn the_joint_wave_has_its_own_kill_switch() {
        for on in [
            None,
            Some(""),
            Some("1"),
            Some(" 1 "),
            Some("true"),
            Some("2"),
        ] {
            assert!(ep_wave_joint_enabled_from(on), "{on:?}");
        }
        assert!(!ep_wave_joint_enabled_from(Some("0")));
        assert!(!ep_wave_joint_enabled_from(Some(" 0\n")));
        assert_eq!(
            QWEN4_EP_WAVE_JOINT_ENV,
            "HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT"
        );
    }

    /// Pins that bind each wave route to its singleton's emission: the FullTable wave binds no
    /// gather and no tile exchange, picks the full runtime and runs the singleton's prelude
    /// once inside the reserve bracket with its floor pinned to 512; the Tile wave binds the
    /// compact runtime plus the installed exchange, the TileExchange bank route, no prelude, and
    /// keeps the floor of the lanes' own scopes; both set and restore the routing-row cap.
    #[test]
    fn each_wave_route_is_its_singletons_emission_by_construction() {
        let src = include_str!("ep_wave.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(!body.contains("row_gather"), "no gather is bound");
        assert!(body.contains("select_layer_moe("));
        // Full-table: the full runtime (`compact_chunk` false), the full-table bank route, the
        // singleton prelude exactly on that route, the floor pin.
        assert!(
            body.contains("route != EpWaveRoute::FullTable,\n"),
            "compact runtime on every route but the full-table one"
        );
        assert!(body.contains("EpWaveRoute::FullTable => EpBankRoute::FullTable"));
        assert_eq!(body.matches("ep_wave_prelude(gpu)?").count(), 2);
        assert_eq!(
            body.matches("if route == EpWaveRoute::FullTable {").count(),
            3
        );
        assert!(body.contains("before_down(gpu)"));
        assert!(body.contains("HOST_STAGE_QUIESCE"));
        assert!(
            body.contains(
                "gpu.qwen4_moe_sym_iu4_floor = rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS"
            ),
            "the full-table wave pins the symmetric-IU4 floor to 512"
        );
        // Tile / Partial: the tile bank route (the partial chunk's wide route is the same class,
        // `ep_wide_bank_route(false, true)`), and for Tile alone the exchange of the installed
        // transport.
        assert!(body.contains(
            "EpWaveRoute::Tile | EpWaveRoute::Partial | EpWaveRoute::PartialMask => {\n                EpBankRoute::TileExchange"
        ));
        assert!(body.contains("exchange: route.hook.as_deref()?"));
        assert!(body.contains("tile_ids_owned: &route.tile_ids_owned"));
        // The lane program binds the transport's exchange on the Tile route and, on the Partial
        // route, only the MASK-ONLY exchange of the shard's partial tile mask: the Partial wave
        // has no cross-rank traffic in its MoE call (its ARs are the per-lane HC writes).
        let lanes = include_str!("../gpu_forward_lanes.rs");
        let arm = &lanes[lanes
            .find("let route = wave_route.unwrap_or(EpWaveRoute::FullTable);")
            .unwrap()..];
        let arm = &arm[..arm.find("wave_moe_calls += 1;").unwrap()];
        assert_eq!(arm.matches("moe.tile_exchange =").count(), 2, "{arm}");
        assert!(arm.contains("if route == EpWaveRoute::Tile {"), "{arm}");
        assert!(
            arm.contains("if route == EpWaveRoute::PartialMask {")
                && arm.contains("ep_wave_partial_mask(fwd.ep.as_ref())")
                && !arm.contains("ep_wave_partial_mask_on"),
            "{arm}"
        );
        // Both routes run under the routing-row cap, set and restored once.
        assert_eq!(body.matches("gpu.qwen4_route_rows_cap").count(), 2);
        assert_eq!(body.matches("self.wave_route.replace(route)").count(), 1);
        assert_eq!(body.matches("self.wave_route = prior_route").count(), 1);
    }
}
