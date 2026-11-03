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
//! rows, on the same full-table runtime (`moe_full[layer]`) the singleton chunk binds, while
//! every other stage runs per lane on the lane's own rows and state.
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

use super::ep_reserve_scratch::EpBankRoute;
use super::*;
use crate::gpu_forward_lanes::{LaneStagePolicy, StageName, StageShare};
use rdna_compute::prefill_invariant::QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP;

/// Most rows one wave carries (the plan of record's 2048-row pass).
pub const QWEN4_EP_WAVE_ROW_CAP: usize = 2048;

/// Fewest rows of a lane the exact wave takes. A lane of at most 8 rows runs the few-row routes
/// and 9..=48 rows the g32 gate/up family in its singleton; g32 is documented bit-identical to
/// the `o4_r8_x4` arm the wave runs, but no wave receipt covers it, so those lanes keep their
/// serial tile until the card says otherwise.
pub const EXACT_WAVE_MIN_LANE_ROWS: usize = 49;

/// Lab knob (default unset: [`EXACT_WAVE_MIN_LANE_ROWS`], byte-identical): lowers the fewest
/// rows of a lane the exact wave takes, to 9 at the lowest (a lane of at most 8 rows runs the
/// few-row routes and is never a wave lane). A measurement switch for the g32 family: a wave
/// of 9..=48-row lanes runs the `o4_r8_x4` arm while each lane's serial tile runs g32.
pub const QWEN4_EP_WAVE_MIN_LANE_ROWS_ENV: &str = "HIPFIRE_EP_LANE_PREFILL_WAVE_MIN_ROWS";

/// [`QWEN4_EP_WAVE_MIN_LANE_ROWS_ENV`] parsed: an integer in `9..=EXACT_WAVE_MAX_LANE_ROWS`,
/// anything else (unset, empty, not a number, out of range) is the default.
pub(crate) fn exact_wave_min_lane_rows_from(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|rows| (9..=EXACT_WAVE_MAX_LANE_ROWS).contains(rows))
        .unwrap_or(EXACT_WAVE_MIN_LANE_ROWS)
}

static EXACT_WAVE_MIN_LANE_ROWS_KNOB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    exact_wave_min_lane_rows_from(
        hipfire_config::developer_var(QWEN4_EP_WAVE_MIN_LANE_ROWS_ENV)
            .ok()
            .as_deref(),
    )
});

/// The fewest rows of a wave lane of this process (every rank thread agrees).
pub(crate) fn exact_wave_min_lane_rows() -> usize {
    *EXACT_WAVE_MIN_LANE_ROWS_KNOB
}

/// Most rows of a lane the exact wave takes: every lane stays below the 512-row floors (a
/// larger tile takes the owner-routed tile route alone and keeps its serial tile).
pub const EXACT_WAVE_MAX_LANE_ROWS: usize = QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP;

/// The wave's stage shares: every stage per lane (each lane's launches are its singleton tile's
/// by construction), the MoE once over all rows.
pub(crate) fn ep_wave_policy() -> LaneStagePolicy {
    LaneStagePolicy::all_per_lane()
        .with(StageName::Moe, StageShare::OnceAllRows)
        .expect("the MoE is not a stateful stage")
}

/// Host facts of one wave, identical on every rank. All of them are decided from replicated
/// state (lane rows, config, flags), never from device contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EpWaveFacts {
    /// Rows of every lane, in lane order.
    pub lane_rows: Vec<usize>,
    /// Per lane: the lane's own rows take the owner-routed tile route (its singleton is then
    /// not the full-table tile the wave reproduces).
    pub lane_tile: Vec<bool>,
    /// The EP full tables are built (`attach_ep` ran).
    pub moe_full_ready: bool,
    /// Trunk-shard partial FFN route: its chunks bind the compact table, not the full one.
    pub ffn_partial: bool,
    /// A non-EP decode-share bank swap or borrowed runtime view is configured.
    pub decode_share: bool,
    /// Some lane or the wave itself enters a non-default routing scope (invariant prefill
    /// profile, agent-turn append route): not the plain sub-512 routes.
    pub scoped_route: bool,
    /// Fewest rows of a lane the wave takes ([`exact_wave_min_lane_rows`]).
    pub min_lane_rows: usize,
    /// Rows the selected scratch holds for the F32 grouped chain: the forward's `max_chunk` in
    /// Full scratch mode, `QWEN4_COMPACT_FALLBACK_ROWS` in compact mode; on an EP reserve load
    /// the resident bank's call limit.
    pub capacity_rows: usize,
    /// An EP reserve load refuses the wave's full-table call (`EpEpoch::plan_call` with the
    /// full-table route): its reason.
    pub reserve_refusal: Option<String>,
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
                "lane {lane} has {rows} rows, outside {}..={EXACT_WAVE_MAX_LANE_ROWS} (its serial tile is not the sub-512 full-table route the wave reproduces)",
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
    if facts.ffn_partial {
        return refuse(
            "the trunk-shard partial FFN route binds the compact table, not the full one"
                .to_string(),
        );
    }
    if facts.decode_share {
        return refuse("a decode-share bank swap is configured on this EP load".to_string());
    }
    if facts.scoped_route {
        return refuse("a routing scope (invariant / append route) is active".to_string());
    }
    if let Some(lane) = facts.lane_tile.iter().position(|tile| *tile) {
        return refuse(format!(
            "lane {lane} ({} rows) takes the owner-routed tile route on its own",
            facts.lane_rows[lane]
        ));
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

/// The runtime a wave's MoE invocation of `layer` binds: exactly the singleton chunk's
/// full-table selection (`select_layer_moe` with `full_chunk` and no tile route: the layer's EP
/// full runtime of the bank in use, never the balanced live or the decode table).
pub(crate) fn ep_wave_layer_moe<'a>(
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
        false,
        tile_moe.get(layer).and_then(Option::as_ref),
    )
}

/// The wave program's MoE invocation count: one per MoE layer of the band, never one per lane.
pub(crate) fn ep_wave_expected_moe_calls(moe_layers: usize) -> usize {
    moe_layers
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
        // (tile route, any scope entered) of `rows` rows under the routing scopes a prefill tile
        // of those rows enters.
        let probe = |this: &Self, gpu: &mut Gpu, rows: usize| -> (bool, bool) {
            let (pm_invariant, append) = Self::route_scopes(this.prefill_scope, gpu, rows);
            let (saved, _) = this.enter_tile_scopes(gpu, rows);
            let route_rows = gpu.route_rows(rows);
            let tile =
                ep_reserve_forward::ep_tile_active(this.ep.as_ref(), gpu, config, rows, route_rows);
            Self::leave_tile_scopes(gpu, saved);
            (tile, pm_invariant || append)
        };
        let mut scoped_route = probe(self, gpu, total).1;
        let mut lane_tile = Vec::with_capacity(lane_rows.len());
        for &rows in lane_rows {
            let (tile, scoped) = probe(self, gpu, rows);
            lane_tile.push(tile);
            scoped_route |= scoped;
        }
        self.prefill_scope = prior_prefill;
        let reserve = self.ep_reserve.as_ref();
        EpWaveFacts {
            lane_rows: lane_rows.to_vec(),
            lane_tile,
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
                reserve.map(|reserve| reserve.banks.resident_call_limit()),
            ),
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

    /// Run a wave's forward `run` over `n` total rows as a lane prefill tile of the full-table
    /// route: lane and prefill scope set, the whole-prompt entry bracketed (nest-safe: the
    /// caller's own bracket makes this one inner), the routing scopes of a prefill tile entered,
    /// the singleton's full-table prelude, the EP reserve bank switch (restore of a dirty epoch,
    /// resident bank) taken exactly as a singleton tile takes it, the routing-row cap set for
    /// the call, and everything put back on every return path.
    pub(crate) fn ep_wave_run(
        &mut self,
        gpu: &mut Gpu,
        n: usize,
        run: impl FnOnce(&mut Self, &mut Gpu) -> Result<(), Qwen4GpuForwardError>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.share_check()?;
        let prior_prefill = std::mem::replace(&mut self.prefill_scope, true);
        let prior_lane = std::mem::replace(&mut self.lane_scope, true);
        self.share_prompt_begin();
        self.ep_ordinary_note_call(n);
        let (scopes, _) = self.enter_tile_scopes(gpu, n);
        let prior_cap = std::mem::replace(
            &mut gpu.qwen4_route_rows_cap,
            QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP,
        );
        let result = (|| {
            if self.ep_reserve.is_some() {
                self.ep_reserve_call(gpu, n, EpBankRoute::FullTable, false, |this, gpu| {
                    this.ep_wave_prelude(gpu)?;
                    run(this, gpu)
                })
            } else {
                self.ep_wave_prelude(gpu)?;
                run(self, gpu)
            }
        })();
        gpu.qwen4_route_rows_cap = prior_cap;
        Self::leave_tile_scopes(gpu, scopes);
        let ended = self.share_prompt_end(gpu, result.is_ok());
        self.prefill_scope = prior_prefill;
        self.lane_scope = prior_lane;
        merge_share_result(result, ended)
    }

    /// Record the route a wave took as the lane tile note of its total rows (the
    /// `HIPFIRE_CB_PHASES` per-tile line reads it; it never feeds a decision).
    pub(crate) fn ep_wave_note(&mut self, rows: usize) {
        self.lane_tile_note = Some(ep_reserve_forward::LaneTileNote {
            rows,
            tile_exchange: false,
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
            moe_full_ready: true,
            ffn_partial: false,
            decode_share: false,
            scoped_route: false,
            min_lane_rows: EXACT_WAVE_MIN_LANE_ROWS,
            capacity_rows: 4096,
            reserve_refusal: None,
        }
    }

    fn refused(facts: &EpWaveFacts) -> String {
        ep_wave_refusal(facts).expect("the wave must be refused")
    }

    #[test]
    fn the_burst_cells_are_admitted_without_any_row_floor() {
        // The code C8 burst, the story C8 burst (480 rows: below every 512 floor), a pair.
        assert_eq!(ep_wave_refusal(&facts(&[149; 8])), None);
        assert_eq!(ep_wave_refusal(&facts(&[60; 8])), None);
        assert_eq!(ep_wave_refusal(&facts(&[49, 49])), None);
        assert_eq!(
            ep_wave_refusal(&facts(&[60, 60, 149, 149, 149, 256, 256, 511])),
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
                exact_wave_min_lane_rows_from(default),
                EXACT_WAVE_MIN_LANE_ROWS,
                "{default:?}"
            );
        }
        assert_eq!(exact_wave_min_lane_rows_from(Some("9")), 9);
        assert_eq!(exact_wave_min_lane_rows_from(Some(" 24 ")), 24);
        assert_eq!(exact_wave_min_lane_rows_from(Some("511")), 511);
        // The process default (knob unset in the test environment) is the const.
        let mut f = facts(&[9, 24]);
        f.min_lane_rows = 9;
        assert_eq!(ep_wave_refusal(&f), None);
        f.min_lane_rows = 49;
        assert!(refused(&f).contains("lane 0"));
    }

    #[test]
    fn lane_rows_outside_the_exact_window_refuse_the_wave() {
        for rows in [0usize, 1, 8, 9, 48, 512, 513, 1024] {
            let f = facts(&[149, rows]);
            assert!(
                refused(&f).contains("lane 1"),
                "rows {rows}: {}",
                refused(&f)
            );
        }
        for rows in [49usize, 50, 149, 511] {
            assert_eq!(ep_wave_refusal(&facts(&[149, rows])), None, "rows {rows}");
        }
        assert_eq!(EXACT_WAVE_MIN_LANE_ROWS, 49);
        assert_eq!(EXACT_WAVE_MAX_LANE_ROWS, 511);
    }

    #[test]
    fn one_lane_or_none_is_not_a_wave() {
        assert!(refused(&facts(&[])).contains("at least two lanes"));
        assert!(refused(&facts(&[149])).contains("at least two lanes"));
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
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.decode_share = true;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.scoped_route = true;
        cases.push(f);
        let mut f = facts(&[200, 200]);
        f.lane_tile = vec![false, true];
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
    fn the_wave_policy_is_per_lane_with_the_moe_once() {
        let policy = ep_wave_policy();
        for stage in StageName::ALL {
            let want = if stage == StageName::Moe {
                StageShare::OnceAllRows
            } else {
                StageShare::PerLane
            };
            assert_eq!(policy.share(stage), want, "{}", stage.key());
        }
        assert_eq!(policy.validate(), Ok(()));
        assert_eq!(ep_wave_expected_moe_calls(48), 48);
    }

    /// Pins that bind the wave to the singleton's full-table emission: the sealed call binds no
    /// gather and no tile exchange, picks the full runtime, runs the singleton's prelude once
    /// inside the reserve bracket, and sets and restores the routing-row cap.
    #[test]
    fn the_wave_is_the_full_table_emission_by_construction() {
        let src = include_str!("ep_wave.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(
            !body.contains("tile_exchange ="),
            "no tile exchange is bound"
        );
        assert!(!body.contains("row_gather"), "no gather is bound");
        assert!(body.contains("select_layer_moe("));
        assert!(body.contains("EpBankRoute::FullTable"));
        assert!(!body.contains("EpBankRoute::TileExchange"));
        assert_eq!(body.matches("ep_wave_prelude(gpu)?").count(), 2);
        assert_eq!(body.matches("gpu.qwen4_route_rows_cap").count(), 2);
        assert!(body.contains("before_down(gpu)"));
        assert!(body.contains("HOST_STAGE_QUIESCE"));
    }
}
