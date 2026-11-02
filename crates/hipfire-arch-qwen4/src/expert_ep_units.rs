// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-expert VRAM/host unit planner of an expert-parallel (EP) rank (D1.2).
//!
//! A *unit* is one routed expert of one trunk layer, `(layer, expert)`, held by
//! rank `expert % ranks` (the seal's static ownership). Today the EP placement
//! is whole-layer: every owned expert of trunk layers `0..vram_layers` is
//! device-resident, every other one pinned host RAM
//! ([`crate::expert_residency::place_routed_experts`]). This planner decides
//! residency per unit instead, from a global hot-set ranking
//! ([`crate::expert_hotset`], `.expert-hotset` v1: global ids
//! `layer * experts_per_layer + expert`, hottest first) and a per-rank byte
//! budget, and reproduces today's split exactly when no ranking is given.
//!
//! Pure CPU arithmetic: no file, environment, device or manifest access. Only
//! the trunk layers are planned; the MTP head's experts stay on today's
//! placement (always host).
//!
//! # Walk
//!
//! For each rank independently: walk the ranking from the hottest unit, skip
//! units another rank holds, keep a unit while its charge fits the rank's
//! remaining budget, and close the rank at the first unit that does not fit
//! (the ranked *prefix* that fits, as the single-card hot set does). A kept
//! unit charges [`EpUnitGeometry::unit_bytes`]; the first kept unit of a
//! `(rank, layer)` also charges the two arena tail pads of that layer's VRAM
//! arena ([`VRAM_ARENA_TAIL_PAD_BYTES`] each, one per projection block), never
//! a per-expert pad.
//!
//! # Host accounting
//!
//! A layer of a rank is [`EpLayerState::Resident`] (all owned experts kept,
//! today's resident path), [`EpLayerState::Host`] (none kept, today's
//! host-mapped path: two pinned allocations, two 1 MiB pads) or
//! [`EpLayerState::Mixed`] (one VRAM arena plus ONE pinned host arena per rank,
//! one 1 MiB pad: [`MIXED_HOST_ARENA_PAD_BYTES`]).

use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_residency::HOST_MAPPED_PAD_BYTES;
use crate::weights::Qwen4EpPlacement;
use rdna_compute::moe::QWEN4_EP_OWNER_PIN_FREE;

/// Tail pad of one VRAM arena block (gate/up or down) of a mixed layer.
pub const VRAM_ARENA_TAIL_PAD_BYTES: u64 = ARENA_TAIL_PAD_BYTES as u64;

/// Pinned-host pad of one mixed layer's single host arena (gate/up block then
/// down block in one `hipHostMalloc`).
pub const MIXED_HOST_ARENA_PAD_BYTES: u64 = HOST_MAPPED_PAD_BYTES;

/// Shape of the planned expert space and the byte size of one expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpUnitGeometry {
    /// Trunk MoE layers planned (ids `>= layers * experts_per_layer` are not units).
    pub layers: usize,
    pub experts_per_layer: usize,
    pub ranks: usize,
    /// Encoded bytes of one expert's gate/up projection.
    pub gate_up_bytes: u64,
    /// Encoded bytes of one expert's down projection.
    pub down_bytes: u64,
}

impl EpUnitGeometry {
    /// Gate/up plus down bytes of one expert.
    pub fn unit_bytes(&self) -> u64 {
        self.gate_up_bytes + self.down_bytes
    }

    /// Whether this geometry is the drop-zero planar Flash-Next expert
    /// (`MQ4G256V2P` 1,689,600 + `MQ4G128V2P` 844,800 bytes), as opposed to the shipped
    /// 1,740,800 + 870,400.
    pub fn is_mq4_planar(&self) -> bool {
        use hipfire_runtime::mq4_planar::{
            planar_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN, FLASH_NEXT_GATE_UP,
        };
        planar_bytes(
            Mq4PlanarKind::GateUp,
            FLASH_NEXT_GATE_UP.0,
            FLASH_NEXT_GATE_UP.1,
        )
        .ok()
            == Some(self.gate_up_bytes as usize)
            && planar_bytes(Mq4PlanarKind::Down, FLASH_NEXT_DOWN.0, FLASH_NEXT_DOWN.1).ok()
                == Some(self.down_bytes as usize)
    }

    /// Units (experts) of one layer that `rank` holds.
    pub fn owned_per_layer(&self, rank: usize) -> usize {
        Qwen4EpPlacement {
            rank,
            ranks: self.ranks,
        }
        .local_count(self.experts_per_layer)
    }

    /// Total unit ids (`layers * experts_per_layer`).
    pub fn total_ids(&self) -> usize {
        self.layers * self.experts_per_layer
    }

    fn check(&self) -> Result<(), String> {
        if self.ranks == 0 || self.layers == 0 || self.experts_per_layer < self.ranks {
            return Err(format!(
                "EP unit plan: {} layers x {} experts cannot cover {} ranks",
                self.layers, self.experts_per_layer, self.ranks
            ));
        }
        if self.unit_bytes() == 0 {
            return Err("EP unit plan: an expert has no bytes".to_string());
        }
        Ok(())
    }
}

/// A ranking read from an `.expert-hotset` v1 sidecar.
#[derive(Clone, Copy, Debug)]
pub struct EpRanking<'a> {
    /// Global unit ids `layer * experts_per_layer + expert`, hottest first; a
    /// permutation of `0..layers * experts_per_layer`.
    pub rank: &'a [u32],
    /// Pooled slot-read counts per global id (for the receipt's coverage).
    pub counts: Option<&'a [u64]>,
    /// SHA-256 of the sidecar bytes (for the receipt).
    pub sha256: Option<&'a str>,
}

/// What decides residency.
#[derive(Clone, Debug)]
pub enum EpUnitSource<'a> {
    /// No ranking: trunk layers `0..vram_layers` fully resident on every rank,
    /// every other layer fully host (today's per-layer split).
    WholeLayers { vram_layers: usize },
    /// Ranking walk under a per-rank byte budget (`rank_budget_bytes[r]` is the
    /// expert bytes rank `r` may keep in VRAM, including arena pads).
    Ranking {
        ranking: EpRanking<'a>,
        rank_budget_bytes: Vec<u64>,
    },
    /// [`Self::Ranking`] with `forced_layers` wholly resident on every rank first (see
    /// [`place_routed_experts_units_forced`]).
    RankingForced {
        ranking: EpRanking<'a>,
        rank_budget_bytes: Vec<u64>,
        forced_layers: Vec<usize>,
    },
}

/// Residency of one rank's owned experts of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpLayerState {
    /// Every owned expert device-resident (today's resident layer).
    Resident,
    /// No owned expert device-resident (today's host-mapped layer).
    Host,
    /// `resident` owned experts in the VRAM arena, `host` in the host arena.
    Mixed { resident: usize, host: usize },
}

/// Per-(rank, layer, expert) residency of one EP load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpUnitPlan {
    geometry: EpUnitGeometry,
    /// `[rank][layer][expert]`; true only for experts the rank owns.
    resident: Vec<bool>,
    ranking_sha256: Option<String>,
    /// Pooled counts mass of the units each rank keeps and of all units it owns.
    coverage: Vec<Option<(u64, u64)>>,
    rank_budget_bytes: Option<Vec<u64>>,
    /// The two whole-resident layers (even, odd parity) a borrowed unit stage lends as its
    /// parity buffers ([`place_routed_experts_units_borrowed_staged`]); `None` for every other
    /// plan (the dedicated stage, or no unit stage).
    stage_borrow: Option<[usize; 2]>,
    /// The ranking the plan walked (global ids `layer * experts_per_layer + expert`, hottest
    /// first); `None` for a whole-layers plan. The D2 tail seeds and orders itself by it.
    ranking_order: Option<Vec<u32>>,
}

impl EpUnitPlan {
    /// The walked ranking, hottest first (`None` without one).
    pub fn ranking_order(&self) -> Option<&[u32]> {
        self.ranking_order.as_deref()
    }

    fn index(&self, rank: usize, layer: usize, expert: usize) -> usize {
        let g = &self.geometry;
        (rank * g.layers + layer) * g.experts_per_layer + expert
    }

    pub fn geometry(&self) -> &EpUnitGeometry {
        &self.geometry
    }

    /// Whether `rank` keeps `(layer, expert)` in VRAM.
    pub fn is_resident(&self, rank: usize, layer: usize, expert: usize) -> bool {
        self.resident[self.index(rank, layer, expert)]
    }

    /// Whether `rank` holds (owns) `expert`.
    pub fn owns(&self, rank: usize, expert: usize) -> bool {
        expert % self.geometry.ranks == rank
    }

    /// One layer's resident bitmap of `rank` over all `experts_per_layer`
    /// experts (never true for an expert another rank owns).
    pub fn layer_resident(&self, rank: usize, layer: usize) -> &[bool] {
        let at = self.index(rank, layer, 0);
        &self.resident[at..at + self.geometry.experts_per_layer]
    }

    pub fn layer_state(&self, rank: usize, layer: usize) -> EpLayerState {
        let owned = self.geometry.owned_per_layer(rank);
        let resident = self
            .layer_resident(rank, layer)
            .iter()
            .filter(|&&kept| kept)
            .count();
        match resident {
            0 => EpLayerState::Host,
            n if n == owned => EpLayerState::Resident,
            n => EpLayerState::Mixed {
                resident: n,
                host: owned - n,
            },
        }
    }

    /// Layers every owned expert of which `rank` keeps in VRAM (today's resident layers), ascending.
    pub fn whole_resident_layers(&self, rank: usize) -> Vec<usize> {
        (0..self.geometry.layers)
            .filter(|&layer| self.layer_state(rank, layer) == EpLayerState::Resident)
            .collect()
    }

    /// Layers that are [`EpLayerState::Resident`] on EVERY rank, ascending (the layers a decoupled
    /// EP reserve can borrow as donors).
    pub fn whole_resident_layers_all_ranks(&self) -> Vec<usize> {
        (0..self.geometry.layers)
            .filter(|&layer| {
                (0..self.geometry.ranks)
                    .all(|rank| self.layer_state(rank, layer) == EpLayerState::Resident)
            })
            .collect()
    }

    /// Resident units of `layer` summed over ranks.
    pub fn layer_resident_units(&self, layer: usize) -> usize {
        (0..self.geometry.ranks)
            .map(|rank| {
                self.layer_resident(rank, layer)
                    .iter()
                    .filter(|&&kept| kept)
                    .count()
            })
            .sum()
    }

    /// Layers of `rank` in each state: `(resident, mixed, host)`.
    pub fn layer_counts(&self, rank: usize) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for layer in 0..self.geometry.layers {
            match self.layer_state(rank, layer) {
                EpLayerState::Resident => counts.0 += 1,
                EpLayerState::Mixed { .. } => counts.1 += 1,
                EpLayerState::Host => counts.2 += 1,
            }
        }
        counts
    }

    /// Units `rank` holds in the planned (trunk) layers.
    pub fn owned_units(&self, rank: usize) -> usize {
        self.geometry.owned_per_layer(rank) * self.geometry.layers
    }

    /// Units `rank` keeps in VRAM.
    pub fn resident_units(&self, rank: usize) -> usize {
        let layer = self.geometry.layers * self.geometry.experts_per_layer;
        self.resident[rank * layer..(rank + 1) * layer]
            .iter()
            .filter(|&&kept| kept)
            .count()
    }

    /// Units `rank` leaves in pinned host RAM.
    pub fn host_units(&self, rank: usize) -> usize {
        self.owned_units(rank) - self.resident_units(rank)
    }

    /// Expert payload bytes `rank` keeps in VRAM (no pads).
    pub fn resident_payload_bytes(&self, rank: usize) -> u64 {
        self.resident_units(rank) as u64 * self.geometry.unit_bytes()
    }

    /// Device bytes of `rank`'s routed experts: payload plus the two arena tail
    /// pads of every mixed layer.
    pub fn vram_bytes(&self, rank: usize) -> u64 {
        let (_, mixed, _) = self.layer_counts(rank);
        self.resident_payload_bytes(rank) + mixed as u64 * 2 * VRAM_ARENA_TAIL_PAD_BYTES
    }

    /// Pinned host bytes of `rank`'s trunk experts including pads: a host layer
    /// keeps today's two allocations (two 1 MiB pads), a mixed layer one host
    /// arena (one pad).
    pub fn host_pinned_bytes(&self, rank: usize) -> u64 {
        let (_, mixed, host) = self.layer_counts(rank);
        self.host_units(rank) as u64 * self.geometry.unit_bytes()
            + host as u64 * 2 * HOST_MAPPED_PAD_BYTES
            + mixed as u64 * MIXED_HOST_ARENA_PAD_BYTES
    }

    /// Layers with at least one resident unit on `rank` (distinct layers touched).
    pub fn layers_touched(&self, rank: usize) -> usize {
        let (resident, mixed, _) = self.layer_counts(rank);
        resident + mixed
    }

    /// `rank`'s resident units expressed in whole layers of its own experts.
    pub fn whole_layer_equivalent(&self, rank: usize) -> f64 {
        self.resident_units(rank) as f64 / self.geometry.owned_per_layer(rank) as f64
    }

    /// The whole-layer split this plan *is*: `Some(n)` when on every rank
    /// exactly the layers `0..n` are fully resident and every other layer fully
    /// host; `None` for any per-expert plan.
    pub fn whole_layer_split(&self) -> Option<usize> {
        let layers = self.geometry.layers;
        let mut split: Option<usize> = None;
        for rank in 0..self.geometry.ranks {
            let (resident, mixed, host) = self.layer_counts(rank);
            if mixed != 0 || resident + host != layers {
                return None;
            }
            if (0..resident).any(|l| self.layer_state(rank, l) != EpLayerState::Resident) {
                return None;
            }
            match split {
                Some(n) if n != resident => return None,
                _ => split = Some(resident),
            }
        }
        split
    }

    /// Whether the plan is exactly today's whole-layer split with `vram_layers`
    /// resident trunk layers on every rank.
    pub fn equals_whole_layers(&self, vram_layers: usize) -> bool {
        self.whole_layer_split() == Some(vram_layers.min(self.geometry.layers))
    }

    /// Fraction of the counts mass of `rank`'s owned units the resident set
    /// covers (`None` without counts).
    pub fn coverage(&self, rank: usize) -> Option<f64> {
        self.coverage[rank].map(|(kept, all)| {
            if all == 0 {
                0.0
            } else {
                kept as f64 / all as f64
            }
        })
    }

    /// The byte budget the ranking walk gave `rank` (net of any stage charge); `None` for a
    /// whole-layer plan.
    pub fn rank_budget(&self, rank: usize) -> Option<u64> {
        self.rank_budget_bytes
            .as_ref()
            .and_then(|budgets| budgets.get(rank).copied())
    }

    /// The one-line load receipt of `rank` (see module docs); stable format:
    /// `qwen4 EP expert tier rank r/n: resident U/O units (X GiB VRAM) | host H units (Y GiB pinned) |
    /// layers R resident, M mixed, H host | whole-layer equivalent W | ranking ...`.
    pub fn receipt_line(&self, rank: usize) -> String {
        const GIB: f64 = (1u64 << 30) as f64;
        let (resident, mixed, host) = self.layer_counts(rank);
        let ranking = match (&self.ranking_sha256, self.coverage(rank)) {
            (Some(sha), Some(coverage)) => {
                format!(
                    "ranking sha256 {} coverage {:.2}% budget {:.2} GiB",
                    &sha[..sha.len().min(16)],
                    coverage * 100.0,
                    self.rank_budget_bytes
                        .as_ref()
                        .map_or(0.0, |budget| budget[rank] as f64 / GIB)
                )
            }
            (Some(sha), None) => format!(
                "ranking sha256 {} budget {:.2} GiB",
                &sha[..sha.len().min(16)],
                self.rank_budget_bytes
                    .as_ref()
                    .map_or(0.0, |budget| budget[rank] as f64 / GIB)
            ),
            (None, _) => "ranking none (whole layers)".to_string(),
        };
        format!(
            "qwen4 EP expert tier rank {rank}/{}: resident {}/{} units ({:.2} GiB VRAM) | host {} units ({:.2} GiB pinned) | layers {resident} resident, {mixed} mixed, {host} host ({} touched) | whole-layer equivalent {:.2} | {ranking}",
            self.geometry.ranks,
            self.resident_units(rank),
            self.owned_units(rank),
            self.vram_bytes(rank) as f64 / GIB,
            self.host_units(rank),
            self.host_pinned_bytes(rank) as f64 / GIB,
            resident + mixed,
            self.whole_layer_equivalent(rank),
        )
    }

    /// [`Self::receipt_line`] of every rank.
    pub fn receipt_lines(&self) -> Vec<String> {
        (0..self.geometry.ranks)
            .map(|rank| self.receipt_line(rank))
            .collect()
    }
}

/// Plan the per-unit residency of every rank of an EP load.
///
/// [`EpUnitSource::WholeLayers`] reproduces
/// [`crate::expert_residency::place_routed_experts`]: trunk layers
/// `< vram_layers` fully resident, all others fully host, the same on every
/// rank. [`EpUnitSource::Ranking`] walks the ranking (see the module docs).
pub fn place_routed_experts_units(
    geometry: &EpUnitGeometry,
    source: &EpUnitSource<'_>,
) -> Result<EpUnitPlan, String> {
    geometry.check()?;
    let g = *geometry;
    let layer_span = g.experts_per_layer;
    let mut resident = vec![false; g.ranks * g.layers * layer_span];
    let at =
        |rank: usize, layer: usize, expert: usize| (rank * g.layers + layer) * layer_span + expert;
    match source {
        EpUnitSource::WholeLayers { vram_layers } => {
            for rank in 0..g.ranks {
                for layer in 0..g.layers.min(*vram_layers) {
                    for expert in (rank..g.experts_per_layer).step_by(g.ranks) {
                        resident[at(rank, layer, expert)] = true;
                    }
                }
            }
            Ok(EpUnitPlan {
                geometry: g,
                resident,
                ranking_sha256: None,
                coverage: vec![None; g.ranks],
                rank_budget_bytes: None,
                stage_borrow: None,
                ranking_order: None,
            })
        }
        EpUnitSource::Ranking {
            ranking,
            rank_budget_bytes,
        } => place_ranking(g, *ranking, rank_budget_bytes, &[]),
        EpUnitSource::RankingForced {
            ranking,
            rank_budget_bytes,
            forced_layers,
        } => place_ranking(g, *ranking, rank_budget_bytes, forced_layers),
    }
}

/// The `count` donor layers a decoupled EP reserve borrows, in PREFERENCE order: the layers the
/// unforced ranking plan already keeps the most units of (summed over ranks; the hottest layers need
/// the fewest extra units to become whole), ties to the lower layer index. `unforced` is the
/// ranking plan placed with no forced layer. Returns fewer than `count` layers only when the model
/// has fewer.
pub fn ep_unit_donor_layers(unforced: &EpUnitPlan, count: usize) -> Vec<usize> {
    ep_unit_donor_layers_excluding(unforced, count, &[])
}

/// [`ep_unit_donor_layers`] that never picks a layer of `excluded`: the borrowed unit stage's two
/// stage layers ([`EP_UNIT_STAGE_BORROW_LAYERS`]) are lent as stage buffers and must not also be
/// scratch donors. Returns fewer than `count` layers when fewer remain.
pub fn ep_unit_donor_layers_excluding(
    unforced: &EpUnitPlan,
    count: usize,
    excluded: &[usize],
) -> Vec<usize> {
    let mut layers: Vec<usize> = (0..unforced.geometry.layers)
        .filter(|layer| !excluded.contains(layer))
        .collect();
    layers.sort_by_key(|&layer| {
        (
            std::cmp::Reverse(unforced.layer_resident_units(layer)),
            layer,
        )
    });
    layers.truncate(count);
    layers
}

/// A ranking plan with `forced_layers` wholly resident on every rank first (they count against the
/// budget; the walk fills the rest). With no forced layer this is exactly
/// [`place_routed_experts_units`] over [`EpUnitSource::Ranking`].
pub fn place_routed_experts_units_forced(
    geometry: &EpUnitGeometry,
    ranking: EpRanking<'_>,
    rank_budget_bytes: &[u64],
    forced_layers: &[usize],
) -> Result<EpUnitPlan, String> {
    geometry.check()?;
    place_ranking(*geometry, ranking, rank_budget_bytes, forced_layers)
}

/// The ranking walk of [`place_routed_experts_units`]; `forced_layers` are placed first as wholly
/// resident layers (every owned expert on every rank, charged to the budget like any resident unit)
/// and the walk skips their units. Forced layers are the whole-layer donors of a decoupled EP
/// reserve (`ep_reserve`), which borrows the bytes of whole resident layers.
fn place_ranking(
    g: EpUnitGeometry,
    ranking: EpRanking<'_>,
    rank_budget_bytes: &[u64],
    forced_layers: &[usize],
) -> Result<EpUnitPlan, String> {
    let layer_span = g.experts_per_layer;
    let mut resident = vec![false; g.ranks * g.layers * layer_span];
    let at =
        |rank: usize, layer: usize, expert: usize| (rank * g.layers + layer) * layer_span + expert;
    if rank_budget_bytes.len() != g.ranks {
        return Err(format!(
            "EP unit plan: {} rank budgets for {} ranks",
            rank_budget_bytes.len(),
            g.ranks
        ));
    }
    let total = g.total_ids();
    if ranking.rank.len() != total {
        return Err(format!(
            "EP unit plan: the ranking has {} ids, the model has {} units ({} layers x {} experts)",
            ranking.rank.len(),
            total,
            g.layers,
            g.experts_per_layer
        ));
    }
    if let Some(counts) = ranking.counts {
        if counts.len() != total {
            return Err(format!(
                "EP unit plan: {} counts for {total} units",
                counts.len()
            ));
        }
    }
    let mut seen = vec![false; total];
    let mut remaining = rank_budget_bytes.to_vec();
    let mut open = vec![true; g.ranks];
    let mut touched = vec![false; g.ranks * g.layers];
    let unit = g.unit_bytes();
    let mut forced_seen = vec![false; g.layers];
    for &layer in forced_layers {
        if layer >= g.layers {
            return Err(format!(
                "EP unit plan: forced resident layer {layer} is outside the {} layers",
                g.layers
            ));
        }
        if std::mem::replace(&mut forced_seen[layer], true) {
            return Err(format!(
                "EP unit plan: forced resident layer {layer} appears twice"
            ));
        }
        for rank in 0..g.ranks {
            let charge = g.owned_per_layer(rank) as u64 * unit + 2 * VRAM_ARENA_TAIL_PAD_BYTES;
            if charge > remaining[rank] {
                return Err(format!(
                    "EP unit plan: forced resident layer {layer} needs {charge} bytes on rank {rank}, the budget has {}",
                    remaining[rank]
                ));
            }
            remaining[rank] -= charge;
            touched[rank * g.layers + layer] = true;
            for expert in (rank..g.experts_per_layer).step_by(g.ranks) {
                resident[at(rank, layer, expert)] = true;
            }
        }
    }
    for &id in ranking.rank {
        let id = id as usize;
        if id >= total {
            return Err(format!(
                "EP unit plan: ranking id {id} is outside the {total} units"
            ));
        }
        if std::mem::replace(&mut seen[id], true) {
            return Err(format!("EP unit plan: ranking id {id} appears twice"));
        }
        let (layer, expert) = (id / layer_span, id % layer_span);
        let rank = expert % g.ranks;
        if resident[at(rank, layer, expert)] || !open[rank] {
            continue;
        }
        let first = !touched[rank * g.layers + layer];
        let charge = unit
            + if first {
                2 * VRAM_ARENA_TAIL_PAD_BYTES
            } else {
                0
            };
        if charge > remaining[rank] {
            open[rank] = false;
            continue;
        }
        remaining[rank] -= charge;
        touched[rank * g.layers + layer] = true;
        resident[at(rank, layer, expert)] = true;
    }
    let coverage = (0..g.ranks)
        .map(|rank| {
            ranking.counts.map(|counts| {
                let (mut kept, mut all) = (0u64, 0u64);
                for layer in 0..g.layers {
                    for expert in (rank..g.experts_per_layer).step_by(g.ranks) {
                        let count = counts[layer * layer_span + expert];
                        all += count;
                        if resident[at(rank, layer, expert)] {
                            kept += count;
                        }
                    }
                }
                (kept, all)
            })
        })
        .collect();
    Ok(EpUnitPlan {
        geometry: g,
        resident,
        ranking_sha256: ranking.sha256.map(str::to_owned),
        coverage,
        rank_budget_bytes: Some(rank_budget_bytes.to_vec()),
        stage_borrow: None,
        ranking_order: Some(ranking.rank.to_vec()),
    })
}

/// Per-rank expert byte budget of a per-unit EP load: the bytes `auto`
/// ([`crate::expert_residency::auto_vram_layers`]) divides by the layer size,
/// before the division. `free_per_rank[r]` is rank `r`'s free VRAM,
/// `non_expert_bytes` the replicated trunk, `reserve_bytes` the whole reserve
/// every rank keeps free (the ordinary reserve with the placement charges, plus
/// the AR-lane store when a lane reservation applies). A rank that cannot hold
/// the trunk and the reserve gets `0` (every expert in host RAM).
pub fn ep_unit_rank_budgets(
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    reserve_bytes: u64,
) -> Vec<u64> {
    free_per_rank
        .iter()
        .map(|&free| {
            free.saturating_sub(non_expert_bytes)
                .saturating_sub(reserve_bytes)
        })
        .collect()
}

/// Lab knob (`HIPFIRE_EP_UNIT_BUDGET_DELTA_MIB`, signed, default 0): MiB added to (or, negative,
/// taken from) every rank's unit budget before the ranking walk, to sweep the resident-unit count
/// on one binary. Lab only: a positive delta can plan more experts than the free VRAM leaves.
pub const EP_UNIT_BUDGET_DELTA_ENV: &str = "HIPFIRE_EP_UNIT_BUDGET_DELTA_MIB";

/// Parse [`EP_UNIT_BUDGET_DELTA_ENV`]: unset or blank is `0`; anything but a signed integer is an
/// error (never silently ignored).
pub fn ep_unit_budget_delta_mib(raw: Option<&str>) -> Result<i64, String> {
    match raw.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(0),
        Some(value) => value.parse::<i64>().map_err(|_| {
            format!("{EP_UNIT_BUDGET_DELTA_ENV}={value:?} is not a signed integer (MiB)")
        }),
    }
}

/// `budgets` with `delta_mib` MiB added to (or taken from, saturating at 0) every rank.
pub fn ep_unit_apply_budget_delta(budgets: Vec<u64>, delta_mib: i64) -> Vec<u64> {
    let delta = delta_mib.unsigned_abs().saturating_mul(1 << 20);
    budgets
        .into_iter()
        .map(|budget| {
            if delta_mib >= 0 {
                budget.saturating_add(delta)
            } else {
                budget.saturating_sub(delta)
            }
        })
        .collect()
}

/// Suffix of a mixed layer's VRAM arena entry (`...layers.{L}.mlp.experts.unit_vram_arena`).
pub const UNIT_VRAM_ARENA_SUFFIX: &str = ".mlp.experts.unit_vram_arena";
/// Suffix of a mixed layer's pinned-host arena entry (`...layers.{L}.mlp.experts.unit_host_arena`).
pub const UNIT_HOST_ARENA_SUFFIX: &str = ".mlp.experts.unit_host_arena";

/// Which arena of a mixed layer holds an expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitTier {
    /// The compact device arena.
    Vram,
    /// The pinned host arena.
    Host,
}

/// Position of one owned expert inside its tier's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitSlot {
    pub tier: UnitTier,
    /// Rank among the owned experts of the same tier, ascending expert id.
    pub slot: usize,
}

/// Where every expert of one `(rank, layer)` lives under a per-expert plan: the
/// address arithmetic shared by the loader (arena bytes), the table builders
/// and the peer sources. Owned experts of each tier take consecutive slots in
/// ascending expert id; an arena is `[gate/up block | down block]`, the gate/up
/// block `n * gate_up_bytes` bytes, the down block `n * down_bytes` bytes (both
/// multiples of 256 for Flash-Next MQ4), so every expert pointer inherits the
/// arena base alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpUnitLayerMap {
    rank: usize,
    layer: usize,
    gate_up_bytes: usize,
    down_bytes: usize,
    /// Per global expert; `None` = another rank's expert.
    slots: Vec<Option<UnitSlot>>,
    n_vram: usize,
    n_host: usize,
}

impl EpUnitLayerMap {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The slot of global expert `expert`, `None` when this rank does not hold it.
    pub fn slot(&self, expert: usize) -> Option<UnitSlot> {
        self.slots.get(expert).copied().flatten()
    }

    /// Experts in `tier`'s arena.
    pub fn count(&self, tier: UnitTier) -> usize {
        match tier {
            UnitTier::Vram => self.n_vram,
            UnitTier::Host => self.n_host,
        }
    }

    /// Whether the layer has both a VRAM and a host arena (the only case that
    /// is arena-backed; otherwise today's whole-layer path applies).
    pub fn is_mixed(&self) -> bool {
        self.n_vram > 0 && self.n_host > 0
    }

    /// Per-expert gate/up bytes.
    pub fn gate_up_bytes(&self) -> usize {
        self.gate_up_bytes
    }

    /// Per-expert down bytes.
    pub fn down_bytes(&self) -> usize {
        self.down_bytes
    }

    /// Payload bytes of `tier`'s arena (gate/up block then down block, no pad).
    pub fn arena_payload_bytes(&self, tier: UnitTier) -> usize {
        self.count(tier) * (self.gate_up_bytes + self.down_bytes)
    }

    /// Bytes the loader hands the weight store for `tier`'s arena entry: the
    /// payload plus, for the VRAM arena, [`VRAM_ARENA_TAIL_PAD_BYTES`] zeroed
    /// tail pad per projection block (the host arena's 1 MiB pad is the pinned
    /// allocator's own).
    pub fn arena_entry_bytes(&self, tier: UnitTier) -> usize {
        let pad = match tier {
            UnitTier::Vram => 2 * VRAM_ARENA_TAIL_PAD_BYTES as usize,
            UnitTier::Host => 0,
        };
        self.arena_payload_bytes(tier) + pad
    }

    /// Offset of the gate/up of `slot` inside its arena.
    pub fn gate_up_offset(&self, slot: UnitSlot) -> usize {
        slot.slot * self.gate_up_bytes
    }

    /// Offset of the down of `slot` inside its arena.
    pub fn down_offset(&self, slot: UnitSlot) -> usize {
        self.count(slot.tier) * self.gate_up_bytes + slot.slot * self.down_bytes
    }

    /// `(tier, gate_up offset, down offset)` of global expert `expert`.
    pub fn expert_offsets(&self, expert: usize) -> Option<(UnitTier, usize, usize)> {
        let slot = self.slot(expert)?;
        Some((slot.tier, self.gate_up_offset(slot), self.down_offset(slot)))
    }

    /// Build `tier`'s arena entry bytes from the layer's full stacks
    /// (`gate_up_all` / `down_all`: all `experts_per_layer` experts, expert `e` at
    /// `e * per-expert bytes`): every owned expert of the tier copied to its
    /// slot, the VRAM tail pad zeroed. The loader's fulfilment payload and the
    /// CPU round-trip reference are this one function.
    pub fn gather_arena(
        &self,
        tier: UnitTier,
        gate_up_all: &[u8],
        down_all: &[u8],
    ) -> Result<Vec<u8>, String> {
        let experts = self.slots.len();
        if gate_up_all.len() != experts * self.gate_up_bytes
            || down_all.len() != experts * self.down_bytes
        {
            return Err(format!(
                "EP unit arena layer {}: source stacks hold {}+{} bytes, {experts} experts need {}+{}",
                self.layer,
                gate_up_all.len(),
                down_all.len(),
                experts * self.gate_up_bytes,
                experts * self.down_bytes
            ));
        }
        let mut arena = vec![0u8; self.arena_entry_bytes(tier)];
        for (expert, slot) in self.slots.iter().enumerate() {
            let Some(slot) = slot.filter(|slot| slot.tier == tier) else {
                continue;
            };
            let gate_up = self.gate_up_offset(slot);
            arena[gate_up..gate_up + self.gate_up_bytes].copy_from_slice(
                &gate_up_all[expert * self.gate_up_bytes..(expert + 1) * self.gate_up_bytes],
            );
            let down = self.down_offset(slot);
            arena[down..down + self.down_bytes].copy_from_slice(
                &down_all[expert * self.down_bytes..(expert + 1) * self.down_bytes],
            );
        }
        Ok(arena)
    }
}

impl EpUnitLayerMap {
    /// [`Self::gather_arena`] for a drop-zero planar plan (DESIGN F1): the layer's full SHIPPED
    /// QT44/QT53 stacks (`gate_up_all` / `down_all`, expert `e` at `e * shipped bytes`) are
    /// repacked planar (`hipfire_runtime::mq4_planar`, every header verified symmetric) into each
    /// owned expert's slot, so the arena holds `2,534,400` bytes per expert. `self`'s per-expert
    /// bytes must be the planar Flash-Next sizes ([`EpUnitGeometry::is_mq4_planar`]).
    pub fn gather_arena_mq4_planar(
        &self,
        tier: UnitTier,
        gate_up_all: &[u8],
        down_all: &[u8],
    ) -> Result<Vec<u8>, String> {
        use hipfire_runtime::mq4_planar::{
            repack_tensor, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN, FLASH_NEXT_GATE_UP,
        };
        let experts = self.slots.len();
        let (gu_rows, gu_k) = FLASH_NEXT_GATE_UP;
        let (dn_rows, dn_k) = FLASH_NEXT_DOWN;
        let gu_src =
            shipped_bytes(Mq4PlanarKind::GateUp, gu_rows, gu_k).map_err(|e| e.to_string())?;
        let dn_src =
            shipped_bytes(Mq4PlanarKind::Down, dn_rows, dn_k).map_err(|e| e.to_string())?;
        if gate_up_all.len() != experts * gu_src || down_all.len() != experts * dn_src {
            return Err(format!(
                "EP unit arena layer {}: shipped source stacks hold {}+{} bytes, {experts} experts need {}+{}",
                self.layer,
                gate_up_all.len(),
                down_all.len(),
                experts * gu_src,
                experts * dn_src
            ));
        }
        let mut arena = vec![0u8; self.arena_entry_bytes(tier)];
        for (expert, slot) in self.slots.iter().enumerate() {
            let Some(slot) = slot.filter(|slot| slot.tier == tier) else {
                continue;
            };
            let gate_up = self.gate_up_offset(slot);
            repack_tensor(
                Mq4PlanarKind::GateUp,
                gu_rows,
                gu_k,
                &gate_up_all[expert * gu_src..(expert + 1) * gu_src],
                &mut arena[gate_up..gate_up + self.gate_up_bytes],
            )
            .map_err(|error| format!("layer {} expert {expert} gate/up: {error}", self.layer))?;
            let down = self.down_offset(slot);
            repack_tensor(
                Mq4PlanarKind::Down,
                dn_rows,
                dn_k,
                &down_all[expert * dn_src..(expert + 1) * dn_src],
                &mut arena[down..down + self.down_bytes],
            )
            .map_err(|error| format!("layer {} expert {expert} down: {error}", self.layer))?;
        }
        Ok(arena)
    }
}

impl EpUnitPlan {
    /// The slot map of `rank`'s `layer`.
    pub fn layer_map(&self, rank: usize, layer: usize) -> EpUnitLayerMap {
        let g = &self.geometry;
        let resident = self.layer_resident(rank, layer);
        let (mut n_vram, mut n_host) = (0, 0);
        let slots = (0..g.experts_per_layer)
            .map(|expert| {
                if expert % g.ranks != rank {
                    return None;
                }
                Some(if resident[expert] {
                    n_vram += 1;
                    UnitSlot {
                        tier: UnitTier::Vram,
                        slot: n_vram - 1,
                    }
                } else {
                    n_host += 1;
                    UnitSlot {
                        tier: UnitTier::Host,
                        slot: n_host - 1,
                    }
                })
            })
            .collect();
        EpUnitLayerMap {
            rank,
            layer,
            gate_up_bytes: g.gate_up_bytes as usize,
            down_bytes: g.down_bytes as usize,
            slots,
            n_vram,
            n_host,
        }
    }

    /// Trunk layers of `rank` that are arena-backed ([`EpLayerState::Mixed`]).
    pub fn mixed_layers(&self, rank: usize) -> Vec<usize> {
        (0..self.geometry.layers)
            .filter(|&layer| matches!(self.layer_state(rank, layer), EpLayerState::Mixed { .. }))
            .collect()
    }
}

/// The kill switch of pinned-owner balanced ownership on per-expert unit layers (D1.9):
/// with a unit plan, a layer that is mixed on some rank, or host on some rank but not on
/// every rank, runs the balanced per-call owner assignment with the resident experts pinned
/// to their holder ([`ep_unit_pin_plane`]) instead of the static `e % ranks` owners. Only the
/// exact value `0` is off (D1 behaviour: static owners on those layers); unset or any other
/// value is on. It only matters where balanced ownership itself is admitted
/// (`HIPFIRE_EP_BALANCED_OWNERS`, exact gfx1201 at 2..=4 ranks).
pub const EP_UNIT_PINNED_OWNERS_ENV: &str = "HIPFIRE_EP_UNIT_PINNED_OWNERS";

/// Whether [`EP_UNIT_PINNED_OWNERS_ENV`] allows pinned owners, from its raw value: on unless
/// exactly `0`.
pub fn ep_unit_pinned_owners_on(raw: Option<&str>) -> bool {
    raw != Some("0")
}

/// How one trunk layer of a unit plan is owned across the mesh, decided from the plan alone
/// so every rank agrees ([`ep_unit_layer_balance`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpUnitLayerBalance {
    /// No rank holds a host expert of the layer: every expert is device-resident on its
    /// holder, the static `e % ranks` route (today's resident layer).
    StaticResident,
    /// Every rank holds the layer wholly in host RAM: today's balanced host layer, the
    /// UNPINNED owner-assign kernel.
    BalancedHost,
    /// Some rank holds a host expert of the layer and not every rank holds it wholly on host:
    /// the PINNED owner-assign kernel (resident experts pinned to their holder, host experts
    /// free). A rank whose own layer is wholly resident runs it too, with every pin its own.
    BalancedPinned,
}

/// The balanced-ownership decision of `layer`, from the plan only (see
/// [`EpUnitLayerBalance`]); the kill switch ([`EP_UNIT_PINNED_OWNERS_ENV`]) is the caller's.
pub fn ep_unit_layer_balance(plan: &EpUnitPlan, layer: usize) -> EpUnitLayerBalance {
    let ranks = plan.geometry().ranks;
    let (mut with_host, mut wholly_host) = (0usize, 0usize);
    for rank in 0..ranks {
        match plan.layer_state(rank, layer) {
            EpLayerState::Resident => {}
            EpLayerState::Mixed { .. } => with_host += 1,
            EpLayerState::Host => {
                with_host += 1;
                wholly_host += 1;
            }
        }
    }
    if with_host == 0 {
        EpUnitLayerBalance::StaticResident
    } else if wholly_host == ranks {
        EpUnitLayerBalance::BalancedHost
    } else {
        EpUnitLayerBalance::BalancedPinned
    }
}

/// The pin plane of `layer` for the pinned owner assignment (`experts_per_layer` bytes, the
/// shared plan: identical on every rank): expert `e`, held by `h = e % ranks`, is pinned to
/// `h` when `h` keeps it device-resident (it exists only in `h`'s VRAM arena), otherwise
/// [`rdna_compute::moe::QWEN4_EP_OWNER_PIN_FREE`] (a host expert is readable from every GPU
/// over the pinned host arena's device-visible address, so any rank may compute it).
pub fn ep_unit_pin_plane(plan: &EpUnitPlan, layer: usize) -> Vec<u8> {
    let g = plan.geometry();
    (0..g.experts_per_layer)
        .map(|expert| {
            let holder = expert % g.ranks;
            if plan.is_resident(holder, layer, expert) {
                // The owner-assign kernels cover at most 4 ranks; a pin byte is a rank index.
                holder as u8
            } else {
                QWEN4_EP_OWNER_PIN_FREE
            }
        })
        .collect()
}

/// The flag of the per-expert EP hot set (D1): a sidecar path, or `0` off.
/// Unset follows the shipped default ([`EpUnitsPolicy::Default`]): with
/// `hipfire_config::memory::EP_EXPERT_HOTSET_DEFAULT_ON` off (today) that is
/// off; with it on, an EP2 gfx1201 load discovers `<model>.ep.expert-hotset`.
pub const EP_EXPERT_HOTSET_ENV: &str = "HIPFIRE_QWEN4_EP_EXPERT_HOTSET";

/// What [`EP_EXPERT_HOTSET_ENV`] says, from its raw value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EpUnitsPolicy {
    /// Unset: the shipped default (discovery when the gate is on).
    Default,
    /// Blank or `0`: never plan per unit. The explicit off switch.
    Off,
    /// A sidecar path (trimmed); never discovered, always required to bind.
    Explicit(std::path::PathBuf),
}

/// Parse the raw value of [`EP_EXPERT_HOTSET_ENV`] (`None` = unset).
pub fn ep_units_policy(raw: Option<&str>) -> EpUnitsPolicy {
    match raw.map(str::trim) {
        None => EpUnitsPolicy::Default,
        Some("") | Some("0") => EpUnitsPolicy::Off,
        Some(path) => EpUnitsPolicy::Explicit(std::path::PathBuf::from(path)),
    }
}

/// The GPU architecture the shipped (discovered) EP hot set runs on.
pub const EP_UNITS_DISCOVERY_ARCH: &str = "gfx1201";
/// The expert-parallel width the shipped (discovered) EP hot set is built for.
pub const EP_UNITS_DISCOVERY_RANKS: usize = 2;

/// What the loader knows when [`EpUnitsPolicy::Default`] may discover a sidecar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpUnitsDiscovery<'a> {
    /// `hipfire_config::memory::EP_EXPERT_HOTSET_DEFAULT_ON`.
    pub gate: bool,
    /// GPU architecture name, e.g. `gfx1201`.
    pub gpu_arch: &'a str,
    /// The load's refusal facts (`ep`, `pp`, the conflicting levers).
    pub facts: EpUnitsFacts,
    /// `<model>.ep.expert-hotset`.
    pub sidecar: std::path::PathBuf,
    /// Whether that file exists.
    pub sidecar_exists: bool,
}

/// Where a load's per-expert plan comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EpUnitsSource {
    /// No per-expert plan; today's whole-layer placement.
    Off,
    /// [`EP_EXPERT_HOTSET_ENV`] named this file.
    Explicit(std::path::PathBuf),
    /// The shipped sidecar next to the model, found by the default policy.
    Discovered(std::path::PathBuf),
    /// A shipped sidecar exists but this load cannot use it (carries why); the load
    /// keeps whole-layer placement. Never a load failure.
    Skipped(String),
    /// The default policy would discover the shipped sidecar on this load (gate on, EP2
    /// [`EP_UNITS_DISCOVERY_ARCH`], no lever refuses), but `<model>.ep.expert-hotset` is not
    /// there: the load keeps whole-layer placement and tells the user how to fetch it
    /// ([`ep_hotset_missing_hint`]).
    Missing,
}

/// The pure source decision. Discovery needs the gate, a present sidecar and an
/// EP2 [`EP_UNITS_DISCOVERY_ARCH`] load that no whole-layer-only lever refuses
/// ([`ep_units_refusal`]); a single-device or pipeline load never notices the
/// file. An explicit path or `0` ignores the gate.
pub fn ep_units_source(policy: &EpUnitsPolicy, found: &EpUnitsDiscovery<'_>) -> EpUnitsSource {
    match policy {
        EpUnitsPolicy::Off => EpUnitsSource::Off,
        EpUnitsPolicy::Explicit(path) => EpUnitsSource::Explicit(path.clone()),
        EpUnitsPolicy::Default => {
            if !found.gate || found.facts.pp > 1 || found.facts.ep < 2 {
                return EpUnitsSource::Off;
            }
            if !found.sidecar_exists {
                return if found.facts.ep == EP_UNITS_DISCOVERY_RANKS
                    && found.gpu_arch == EP_UNITS_DISCOVERY_ARCH
                    && ep_units_refusal(&found.facts).is_ok()
                {
                    EpUnitsSource::Missing
                } else {
                    EpUnitsSource::Off
                };
            }
            if found.facts.ep != EP_UNITS_DISCOVERY_RANKS {
                return EpUnitsSource::Skipped(format!(
                    "topology.ep is {}; the shipped hot set is built for ep = {EP_UNITS_DISCOVERY_RANKS}",
                    found.facts.ep
                ));
            }
            if found.gpu_arch != EP_UNITS_DISCOVERY_ARCH {
                return EpUnitsSource::Skipped(format!(
                    "gpu is {}; the shipped hot set runs on {EP_UNITS_DISCOVERY_ARCH} only",
                    found.gpu_arch
                ));
            }
            match ep_units_refusal(&found.facts) {
                Ok(()) => EpUnitsSource::Discovered(found.sidecar.clone()),
                Err(why) => EpUnitsSource::Skipped(why),
            }
        }
    }
}

/// The registry tag `hipfire pull` takes to fetch the shipped EP hot set of the published
/// Flash-Next MQ4 XTS base (split base or the canonical monolithic it is a view of).
pub const EP_HOTSET_PULL_TAG: &str = "qwen3.8:flash-next";

/// The one-line hint of [`EpUnitsSource::Missing`] for a model file named `model_file_name`,
/// or `None` for every file no shipped sidecar is published for (only the Flash-Next MQ4 XTS
/// base and its canonical monolithic have one).
pub fn ep_hotset_missing_hint(model_file_name: &str) -> Option<String> {
    matches!(
        model_file_name,
        "qwen3.8-flash-next-base.mq4-xts" | "qwen3.8-flash-next.mq4-xts"
    )
    .then(|| {
        format!(
            "EP expert hot set missing: run `hipfire pull {EP_HOTSET_PULL_TAG}` to fetch it (placement falls back to whole-layer)"
        )
    })
}

/// `HIPFIRE_EP_UNIT_RESERVE` for a load that plans per unit, from its raw value: only an exact
/// `1` keeps the decoupled reserve (units budgeted against its census, whole-resident donor
/// layers); unset or anything else plans the units under the ordinary reserve, the shipped
/// default (the explicit-donor reserve costs prefill time, see `docs/env-vars.md`).
pub fn ep_unit_reserve_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// The tier a manifest entry name denotes when it is a mixed layer's arena
/// entry ([`UNIT_VRAM_ARENA_SUFFIX`] / [`UNIT_HOST_ARENA_SUFFIX`]), else `None`.
pub fn unit_arena_tier(name: &str) -> Option<UnitTier> {
    if name.ends_with(UNIT_VRAM_ARENA_SUFFIX) {
        Some(UnitTier::Vram)
    } else if name.ends_with(UNIT_HOST_ARENA_SUFFIX) {
        Some(UnitTier::Host)
    } else {
        None
    }
}

/// Whether `name` is a mixed layer's arena entry.
pub fn is_unit_arena_name(name: &str) -> bool {
    unit_arena_tier(name).is_some()
}

/// The names of a mixed layer's two aggregate stacks, from the arena entry
/// name: `(gate_up_proj, down_proj)` of the same layer prefix. `None` when
/// `arena_name` is not an arena entry.
pub fn unit_arena_source_names(arena_name: &str) -> Option<(String, String)> {
    let suffix = match unit_arena_tier(arena_name)? {
        UnitTier::Vram => UNIT_VRAM_ARENA_SUFFIX,
        UnitTier::Host => UNIT_HOST_ARENA_SUFFIX,
    };
    let prefix = arena_name.strip_suffix(suffix)?;
    Some((
        format!("{prefix}.mlp.experts.gate_up_proj"),
        format!("{prefix}.mlp.experts.down_proj"),
    ))
}

/// The bytes of the arena entry `arena_name` of `map`'s layer, gathered from
/// the layer's full stacks (the loader's fulfilment payload: the entry's tier
/// comes from its name, the bytes from [`EpUnitLayerMap::gather_arena`]).
pub fn unit_arena_entry_bytes(
    map: &EpUnitLayerMap,
    arena_name: &str,
    gate_up_all: &[u8],
    down_all: &[u8],
) -> Result<Vec<u8>, String> {
    let tier = unit_arena_tier(arena_name)
        .ok_or_else(|| format!("'{arena_name}' is not an EP unit arena entry"))?;
    map.gather_arena(tier, gate_up_all, down_all)
}

/// The per-expert geometry of an EP unit plan from the byte sizes of every
/// trunk layer's aggregate stacks (`gate_up_stack_bytes[layer]` /
/// `down_stack_bytes[layer]`, each stack holding `experts_per_layer` experts):
/// every layer must have the same sizes, each a whole number of experts whose
/// per-expert bytes are 256-aligned (so every arena pointer inherits the arena
/// base alignment).
pub fn ep_unit_geometry(
    layers: usize,
    experts_per_layer: usize,
    ranks: usize,
    gate_up_stack_bytes: &[u64],
    down_stack_bytes: &[u64],
) -> Result<EpUnitGeometry, String> {
    if gate_up_stack_bytes.len() != layers || down_stack_bytes.len() != layers {
        return Err(format!(
            "EP unit plan: {} gate/up and {} down stack sizes for {layers} trunk layers",
            gate_up_stack_bytes.len(),
            down_stack_bytes.len()
        ));
    }
    if experts_per_layer == 0 {
        return Err("EP unit plan: no routed experts per layer".to_string());
    }
    let per_expert = |what: &str, sizes: &[u64]| -> Result<u64, String> {
        let first = *sizes.first().ok_or("EP unit plan: no trunk layers")?;
        if let Some(layer) = sizes.iter().position(|&bytes| bytes != first) {
            return Err(format!(
                "EP unit plan: layer {layer} {what} stack holds {} bytes, layer 0 holds {first}: the layers are not uniform",
                sizes[layer]
            ));
        }
        let experts = experts_per_layer as u64;
        if first == 0 || first % experts != 0 {
            return Err(format!(
                "EP unit plan: the {what} stack of {first} bytes is not a whole number of {experts} experts"
            ));
        }
        let bytes = first / experts;
        if bytes % 256 != 0 {
            return Err(format!(
                "EP unit plan: {bytes} {what} bytes per expert is not 256-byte aligned"
            ));
        }
        Ok(bytes)
    };
    let geometry = EpUnitGeometry {
        layers,
        experts_per_layer,
        ranks,
        gate_up_bytes: per_expert("gate/up", gate_up_stack_bytes)?,
        down_bytes: per_expert("down", down_stack_bytes)?,
    };
    geometry.check()?;
    Ok(geometry)
}

/// The facts of one load [`ep_units_refusal`] judges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpUnitsFacts {
    /// `topology.ep` of the load.
    pub ep: usize,
    /// `topology.pp` of the load.
    pub pp: usize,
    /// `HIPFIRE_QWEN4_EXPERT_HOTSET` names an explicit sidecar.
    pub single_card_hotset: bool,
    /// `HIPFIRE_EP_CHUNK_AUTOFIT=1`.
    pub chunk_autofit: bool,
    /// `HIPFIRE_EP_LAB_RESIDENT_LAYERS` is set.
    pub lab_resident_cap: bool,
}

/// Why a load with [`EP_EXPERT_HOTSET_ENV`] set is refused, if it is: the
/// flag needs expert parallelism and composes with none of the single-device
/// or whole-layer-only levers. The EP long-prefill expert stage is NOT one of
/// them: a unit plan has its own stage ([`EpUnitStageKind::Units`]). Every
/// message names the flag.
pub fn ep_units_refusal(facts: &EpUnitsFacts) -> Result<(), String> {
    let refuse = |why: &str| Err(format!("{EP_EXPERT_HOTSET_ENV}: {why}"));
    if facts.pp > 1 {
        return refuse("pipeline parallelism (topology.pp > 1) is not supported; the per-expert hot set needs expert parallelism (topology.ep >= 2)");
    }
    if facts.ep < 2 {
        return refuse(
            "needs expert parallelism (topology.ep >= 2); unset the flag for a single-device load",
        );
    }
    if facts.single_card_hotset {
        return refuse("HIPFIRE_QWEN4_EXPERT_HOTSET names the single-device hot set, which expert parallelism never applies; unset one of the two");
    }
    if facts.chunk_autofit {
        return refuse("HIPFIRE_EP_CHUNK_AUTOFIT=1 re-plans the prefill chunk and the reserve under the per-expert budget; unset it");
    }
    if facts.lab_resident_cap {
        return refuse("HIPFIRE_EP_LAB_RESIDENT_LAYERS caps whole resident layers, which the per-expert plan does not use; unset it");
    }
    Ok(())
}

/// `HIPFIRE_EP_EXPERT_STAGE`: the EP long-prefill expert stage lever (default
/// on, exact `0` off).
pub const EP_EXPERT_STAGE_ENV: &str = "HIPFIRE_EP_EXPERT_STAGE";

/// Zeroed tail pad after the staged payload of each unit-stage buffer (the
/// kernels' tail over-read, as the tier stage's).
pub const EP_UNIT_STAGE_PAD_BYTES: u64 = ARENA_TAIL_PAD_BYTES as u64;

/// Which G2 stage an EP load builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpUnitStageKind {
    /// Today's donor stage: two resident layers lent as stage buffers. No
    /// unit plan (flag off), or a plan that IS a whole-layer split
    /// ([`EpUnitPlan::whole_layer_split`]).
    Donors,
    /// Two dedicated VRAM stage buffers; a host-bearing layer (mixed or
    /// wholly host) copies its host region into its parity's buffer.
    Units,
}

/// The stage kind `plan` (the load's unit plan, `None` without the flag) calls
/// for. The one chooser `Qwen4ExpertStage::new` and the loader's charge share.
pub fn ep_unit_stage_kind(plan: Option<&EpUnitPlan>) -> EpUnitStageKind {
    match plan {
        Some(plan) if plan.whole_layer_split().is_none() => EpUnitStageKind::Units,
        _ => EpUnitStageKind::Donors,
    }
}

impl EpUnitPlan {
    /// Bytes `rank`'s `layer` copies into a unit-stage buffer: the pinned host
    /// region the layer's host experts live in (a mixed layer's single host
    /// arena, a host layer's two aggregates), `0` for a resident layer.
    pub fn stage_host_region_bytes(&self, rank: usize, layer: usize) -> u64 {
        let host = match self.layer_state(rank, layer) {
            EpLayerState::Resident => 0,
            EpLayerState::Host => self.geometry.owned_per_layer(rank),
            EpLayerState::Mixed { host, .. } => host,
        };
        host as u64 * self.geometry.unit_bytes()
    }

    /// Whether `rank` holds host experts in any trunk layer.
    pub fn rank_has_host_layer(&self, rank: usize) -> bool {
        (0..self.geometry.layers).any(|layer| self.stage_host_region_bytes(rank, layer) > 0)
    }

    /// The largest [`Self::stage_host_region_bytes`] over every rank and layer.
    pub fn max_stage_host_region_bytes(&self) -> u64 {
        (0..self.geometry.ranks)
            .flat_map(|rank| {
                (0..self.geometry.layers)
                    .map(move |layer| self.stage_host_region_bytes(rank, layer))
            })
            .max()
            .unwrap_or(0)
    }

    /// The two whole-resident layers a borrowed unit stage lends as its parity buffers (even
    /// layer first), `None` for a plan with the dedicated stage or none.
    pub fn stage_borrow_layers(&self) -> Option<[usize; 2]> {
        self.stage_borrow
    }

    /// Bytes of the two tensors (compact gate/up, compact down) one lent stage layer of `rank`
    /// holds: its owned experts' aggregates, the two buffers of a parity.
    pub fn stage_borrow_buffer_bytes(&self, rank: usize) -> [u64; 2] {
        let owned = self.geometry.owned_per_layer(rank) as u64;
        [
            owned * self.geometry.gate_up_bytes,
            owned * self.geometry.down_bytes,
        ]
    }

    /// The borrowed stage's lent bytes per parity (both tensors) of the largest rank.
    fn stage_borrow_parity_bytes(&self) -> u64 {
        (0..self.geometry.ranks)
            .map(|rank| self.stage_borrow_buffer_bytes(rank).iter().sum::<u64>())
            .max()
            .unwrap_or(0)
    }

    /// The invariants of a borrowed stage: the lent layers are `[even, odd]` trunk layers,
    /// whole-resident on every rank (their own experts are what is lent and restored), and
    /// precede every layer of their parity that needs staging (a lent layer's own MoE must have
    /// read its tensors before the first staged copy overwrites them). `Ok` for a plan without a
    /// borrowed stage.
    pub fn check_stage_borrow(&self) -> Result<(), String> {
        let Some(lent) = self.stage_borrow else {
            return Ok(());
        };
        if lent[0] % 2 != 0 || lent[1] % 2 != 1 {
            return Err(format!(
                "EP unit stage: the lent layers {lent:?} are not an (even, odd) pair"
            ));
        }
        for rank in 0..self.geometry.ranks {
            for layer in lent {
                if layer >= self.geometry.layers
                    || self.layer_state(rank, layer) != EpLayerState::Resident
                {
                    return Err(format!(
                        "EP unit stage: the lent layer {layer} is not a whole-resident trunk layer on rank {rank}"
                    ));
                }
            }
            for layer in 0..self.geometry.layers {
                if self.stage_host_region_bytes(rank, layer) > 0 && layer <= lent[layer % 2] {
                    return Err(format!(
                        "EP unit stage: layer {layer} of rank {rank} needs staging but does not follow the lent layer {} of its parity",
                        lent[layer % 2]
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The switches that decide whether an EP rank builds a unit stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpUnitStageSwitches<'a> {
    pub arch: &'a str,
    /// `HIPFIRE_QWEN4_EXPERT_STAGE`.
    pub expert_stage: Option<&'a str>,
    /// [`EP_EXPERT_STAGE_ENV`].
    pub ep_stage: Option<&'a str>,
    /// `HIPFIRE_EP_OWNER_ROUTED_PREFILL`.
    pub owner_routed: Option<&'a str>,
}

/// Whether the switches admit the EP expert stage on this arch: the arch
/// admits it under EP (gfx1201, `stage_policy(arch).expert_parallel_stage`),
/// the stage switch is on (`HIPFIRE_QWEN4_EXPERT_STAGE` explicit `1`, or the
/// arch default; probe-free: gfx1201 never reads the host link, so this is the
/// same value `expert_stage_enabled` settles on), and neither
/// `HIPFIRE_EP_EXPERT_STAGE` nor `HIPFIRE_EP_OWNER_ROUTED_PREFILL` is `0`. The
/// loader's VRAM charge and the forward's build decision both read this. The
/// owner-routed route's own runtime admission (S1 symmetric headers) can still
/// decline the stage after the load; the bytes are then charged and unused.
pub fn ep_unit_stage_switches_on(switches: &EpUnitStageSwitches<'_>) -> bool {
    rdna_compute::expert_capabilities::stage_policy(switches.arch).expert_parallel_stage
        && crate::expert_residency::expert_stage_switch(
            switches.arch,
            switches.expert_stage,
            || crate::host_link::HostLink::Unmeasured,
        )
        && switches.ep_stage != Some("0")
        && switches.owner_routed != Some("0")
}

/// [`ep_unit_stage_switches_on`] with the switches read from the process
/// configuration, for `arch`.
pub fn ep_unit_stage_on(arch: &str) -> bool {
    let var = |name: &str| hipfire_config::developer_var(name).ok();
    let (expert_stage, ep_stage, owner_routed) = (
        var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV),
        var(EP_EXPERT_STAGE_ENV),
        var(crate::gpu_forward::QWEN4_EP_OWNER_ROUTED_PREFILL_ENV),
    );
    ep_unit_stage_switches_on(&EpUnitStageSwitches {
        arch,
        expert_stage: expert_stage.as_deref(),
        ep_stage: ep_stage.as_deref(),
        owner_routed: owner_routed.as_deref(),
    })
}

/// Whether `rank` builds a unit stage: the switches are on, the plan calls for
/// the unit stage and the rank holds a host expert. The forward builds exactly
/// when this holds; the loader charges when it holds on any rank.
pub fn ep_unit_stage_builds(stage_on: bool, plan: Option<&EpUnitPlan>, rank: usize) -> bool {
    stage_on
        && ep_unit_stage_kind(plan) == EpUnitStageKind::Units
        && plan.is_some_and(|plan| plan.rank_has_host_layer(rank))
}

/// VRAM one unit stage takes on every rank: capacity of the stage buffers and
/// the bytes charged for them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpUnitStageCharge {
    /// Payload capacity `C` of each of the two stage buffers (largest host
    /// region of any layer of any rank); `0` when no stage is built. A borrowed stage
    /// ([`Self::borrowed`]) reports the lent bytes per parity (both lent tensors of the largest
    /// rank), which are NOT charged.
    pub capacity: u64,
    /// `2 * (C + pad)` plus the stage pointer tables; `0` when `capacity` is. A borrowed stage
    /// charges only the pointer tables.
    pub bytes: u64,
    /// The whole-resident layers lent as the parity buffers
    /// ([`place_routed_experts_units_borrowed_staged`]); `None` for the dedicated stage.
    pub borrowed: Option<[usize; 2]>,
}

/// Device bytes of the stage pointer tables of every trunk layer (two tables
/// of `2 * experts` f32 per layer), an upper bound of what a rank allocates.
pub fn ep_unit_stage_table_bytes(geometry: &EpUnitGeometry) -> u64 {
    geometry.layers as u64 * 16 * geometry.experts_per_layer as u64
}

fn ep_unit_stage_charge_for(capacity: u64, tables: u64) -> EpUnitStageCharge {
    if capacity == 0 {
        return EpUnitStageCharge::default();
    }
    EpUnitStageCharge {
        capacity,
        bytes: capacity
            .saturating_add(EP_UNIT_STAGE_PAD_BYTES)
            .saturating_mul(2)
            .saturating_add(tables),
        borrowed: None,
    }
}

/// The stage VRAM of the FINAL `plan`: zero unless some rank builds a unit
/// stage ([`ep_unit_stage_builds`]), else `C` = the plan's largest host region. A plan with a
/// borrowed stage ([`EpUnitPlan::stage_borrow_layers`]) charges only the stage pointer tables.
pub fn ep_unit_plan_stage_charge(plan: &EpUnitPlan, stage_on: bool) -> EpUnitStageCharge {
    if let Some(lent) = plan.stage_borrow {
        return if stage_on {
            EpUnitStageCharge {
                capacity: plan.stage_borrow_parity_bytes(),
                bytes: ep_unit_stage_table_bytes(plan.geometry()),
                borrowed: Some(lent),
            }
        } else {
            EpUnitStageCharge::default()
        };
    }
    let builds =
        (0..plan.geometry().ranks).any(|rank| ep_unit_stage_builds(stage_on, Some(plan), rank));
    if !builds {
        return EpUnitStageCharge::default();
    }
    ep_unit_stage_charge_for(
        plan.max_stage_host_region_bytes(),
        ep_unit_stage_table_bytes(plan.geometry()),
    )
}

/// The load receipt's stage clause.
pub fn ep_unit_stage_receipt(stage: &EpUnitStageCharge) -> String {
    if stage.capacity == 0 {
        "stage none".to_string()
    } else if let Some(lent) = stage.borrowed {
        format!(
            "stage borrowed from layers [{}, {}] ({} MiB lent per parity, {} KiB pointer tables charged)",
            lent[0],
            lent[1],
            stage.capacity >> 20,
            stage.bytes >> 10
        )
    } else {
        format!(
            "stage {} MiB x 2 charged ({} MiB with tables)",
            stage.capacity >> 20,
            stage.bytes >> 20
        )
    }
}

/// Iteration cap of the stage-capacity fixed point.
pub const EP_UNIT_STAGE_FIXED_POINT_ITERATIONS: usize = 8;

/// A unit plan placed under its own stage charge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpUnitStagedPlan {
    pub plan: EpUnitPlan,
    pub stage: EpUnitStageCharge,
    /// Placements made.
    pub iterations: usize,
    /// The fixed point converged within the cap (`false`: `C` is the largest
    /// whole-layer region).
    pub converged: bool,
}

/// The fixed point, over any placement function of the per-rank charge (bytes
/// taken off every rank's budget): `C_0 = 0`, `plan_k = place(charge(C_k))`,
/// `C_{k+1}` = the largest host region of `plan_k` (zero when the plan does
/// not take the unit stage and `donor_floor` is zero; `donor_floor` is the whole-layer region a
/// borrowed donor layer needs in the stage), stop at the first `C_{k+1} <= C_k`; the plan and
/// `C_k` are returned. After [`EP_UNIT_STAGE_FIXED_POINT_ITERATIONS`]
/// placements without a fixed point `C` is the largest whole-layer region. A
/// plan that needs no unit stage returns zero charge (its placement was made
/// under a possibly larger charge, so it still fits).
fn solve_stage_capacity(
    geometry: &EpUnitGeometry,
    stage_on: bool,
    donor_floor: u64,
    mut place: impl FnMut(u64) -> Result<EpUnitPlan, String>,
) -> Result<EpUnitStagedPlan, String> {
    let tables = ep_unit_stage_table_bytes(geometry);
    let need = |plan: &EpUnitPlan| {
        let region = if ep_unit_stage_kind(Some(plan)) == EpUnitStageKind::Units {
            plan.max_stage_host_region_bytes()
        } else {
            0
        };
        region.max(donor_floor)
    };
    if !stage_on {
        return Ok(EpUnitStagedPlan {
            plan: place(0)?,
            stage: EpUnitStageCharge::default(),
            iterations: 1,
            converged: true,
        });
    }
    let mut capacity = 0u64;
    for iteration in 1..=EP_UNIT_STAGE_FIXED_POINT_ITERATIONS {
        let plan = place(ep_unit_stage_charge_for(capacity, tables).bytes)?;
        let region = need(&plan);
        if region <= capacity {
            let capacity = if region == 0 { 0 } else { capacity };
            return Ok(EpUnitStagedPlan {
                plan,
                stage: ep_unit_stage_charge_for(capacity, tables),
                iterations: iteration,
                converged: true,
            });
        }
        capacity = region;
    }
    let full = (0..geometry.ranks)
        .map(|rank| geometry.owned_per_layer(rank) as u64)
        .max()
        .unwrap_or(0)
        .saturating_mul(geometry.unit_bytes());
    let plan = place(ep_unit_stage_charge_for(full, tables).bytes)?;
    let region = need(&plan);
    if region > full {
        return Err(format!(
            "EP unit stage: a host region of {region} bytes exceeds the whole-layer region of {full} bytes"
        ));
    }
    Ok(EpUnitStagedPlan {
        plan,
        stage: ep_unit_stage_charge_for(if region == 0 { 0 } else { full }, tables),
        iterations: EP_UNIT_STAGE_FIXED_POINT_ITERATIONS + 1,
        converged: false,
    })
}

/// [`place_routed_experts_units`] over a ranking with the unit stage paid for:
/// the two stage buffers are fresh VRAM on every rank, so every rank's budget
/// is reduced by `2 * (C + pad)` plus the stage tables before the ranking walk,
/// `C` the largest host region of the returned plan (fixed point, see
/// [`solve_stage_capacity`]). With `stage_on` false nothing is charged and the
/// plan is exactly the plain ranking plan.
pub fn place_routed_experts_units_staged(
    geometry: &EpUnitGeometry,
    ranking: EpRanking<'_>,
    rank_budget_bytes: &[u64],
    stage_on: bool,
) -> Result<EpUnitStagedPlan, String> {
    place_routed_experts_units_forced_staged(geometry, ranking, rank_budget_bytes, &[], stage_on)
}

/// [`place_routed_experts_units_staged`] with `forced_layers` wholly resident on every rank (the
/// donor layers of a decoupled EP reserve). A borrowed donor layer is read through the unit stage,
/// so with `stage_on` and a non-empty `forced_layers` the stage capacity `C` is at least one whole
/// layer region.
pub fn place_routed_experts_units_forced_staged(
    geometry: &EpUnitGeometry,
    ranking: EpRanking<'_>,
    rank_budget_bytes: &[u64],
    forced_layers: &[usize],
    stage_on: bool,
) -> Result<EpUnitStagedPlan, String> {
    let donor_floor = if stage_on && !forced_layers.is_empty() {
        (0..geometry.ranks)
            .map(|rank| geometry.owned_per_layer(rank) as u64)
            .max()
            .unwrap_or(0)
            .saturating_mul(geometry.unit_bytes())
    } else {
        0
    };
    solve_stage_capacity(geometry, stage_on, donor_floor, |charge| {
        place_routed_experts_units_forced(
            geometry,
            ranking,
            &rank_budget_bytes
                .iter()
                .map(|budget| budget.saturating_sub(charge))
                .collect::<Vec<_>>(),
            forced_layers,
        )
    })
}

/// The two whole-resident layers a borrowed unit stage lends as its parity buffers (even, odd):
/// the first two trunk layers, which precede every layer that needs staging (a mixed or host
/// layer, a scratch donor), so each lent layer's own MoE has read its tensors before the first
/// staged copy overwrites them.
pub const EP_UNIT_STAGE_BORROW_LAYERS: [usize; 2] = [0, 1];

/// [`place_routed_experts_units_forced_staged`] with a BORROWED unit stage
/// (`HIPFIRE_EP_UNIT_STAGE_BORROW=1`): instead of two dedicated stage buffers the stage lends two
/// whole-resident layers, [`EP_UNIT_STAGE_BORROW_LAYERS`], as its parity buffers (their gate/up and
/// down tensors are overwritten by staged copies after their own MoE and restored from pinned host
/// copies). Those layers are forced whole-resident on every rank first and count against the budget
/// like any resident layer; the dedicated stage charge is zero and only the stage pointer tables are
/// charged. `scratch_donors` (the decoupled reserve's whole-layer donors) must not include a lent
/// layer ([`ep_unit_donor_layers_excluding`]). Refuses when the stage is off, the model has no layer
/// beyond the lent ones, a donor is a lent layer, or the plan is a whole-layer split (the donor
/// stage serves it).
pub fn place_routed_experts_units_borrowed_staged(
    geometry: &EpUnitGeometry,
    ranking: EpRanking<'_>,
    rank_budget_bytes: &[u64],
    scratch_donors: &[usize],
    stage_on: bool,
) -> Result<EpUnitStagedPlan, String> {
    let lent = EP_UNIT_STAGE_BORROW_LAYERS;
    if !stage_on {
        return Err("EP unit stage: a borrowed stage needs the unit stage on".to_string());
    }
    if geometry.layers <= lent[1] + 1 {
        return Err(format!(
            "EP unit stage: a borrowed stage needs a layer beyond the lent layers {lent:?}, the model has {} layers",
            geometry.layers
        ));
    }
    if let Some(layer) = scratch_donors.iter().find(|layer| lent.contains(layer)) {
        return Err(format!(
            "EP unit stage: the scratch donor layer {layer} is a lent stage layer {lent:?}"
        ));
    }
    let tables = ep_unit_stage_table_bytes(geometry);
    let forced: Vec<usize> = lent.iter().chain(scratch_donors).copied().collect();
    let budgets: Vec<u64> = rank_budget_bytes
        .iter()
        .map(|budget| budget.saturating_sub(tables))
        .collect();
    let mut plan = place_routed_experts_units_forced(geometry, ranking, &budgets, &forced)?;
    if ep_unit_stage_kind(Some(&plan)) != EpUnitStageKind::Units {
        return Err(
            "EP unit stage: the plan is a whole-layer split, which takes the donor stage"
                .to_string(),
        );
    }
    plan.stage_borrow = Some(lent);
    plan.check_stage_borrow()?;
    let stage = ep_unit_plan_stage_charge(&plan, true);
    Ok(EpUnitStagedPlan {
        plan,
        stage,
        iterations: 1,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_residency::{host_mapped_bytes, place_routed_experts};
    use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};
    use rdna_compute::DType;

    // Flash-Next mq4-xts: 1280 x 10 x 136 gate/up, 2560 x 5 x 68 down.
    const GATE_UP: u64 = 1280 * 10 * 136;
    const DOWN: u64 = 2560 * 5 * 68;

    fn geometry(ranks: usize) -> EpUnitGeometry {
        EpUnitGeometry {
            layers: 48,
            experts_per_layer: 512,
            ranks,
            gate_up_bytes: GATE_UP,
            down_bytes: DOWN,
        }
    }

    /// Deterministic pseudo-random permutation of `0..n` (Fisher-Yates, LCG).
    fn permutation(n: usize, mut state: u64) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ids.swap(i, (state >> 33) as usize % (i + 1));
        }
        ids
    }

    fn entry(name: String, layer: usize, experts: usize) -> WeightEntry {
        let mut entry = WeightEntry::model(
            name,
            vec![experts, 4, 256],
            DType::MQ4G256V2,
            ShardPolicy::Replicate,
        );
        entry.layer = Some(layer);
        entry
    }

    #[test]
    fn unit_bytes_are_the_flash_next_expert() {
        assert_eq!(geometry(2).unit_bytes(), 2_611_200);
        assert_eq!(GATE_UP % 256, 0);
        assert_eq!(DOWN % 256, 0);
    }

    /// The flag-off identity: with no ranking the unit plan is exactly today's
    /// per-layer split, per rank, per expert, and its host-mapped bytes equal
    /// `place_routed_experts` + `host_mapped_bytes` on a manifest.
    #[test]
    fn no_ranking_equals_todays_per_layer_split() {
        for ranks in [2usize, 3] {
            let g = geometry(ranks);
            for vram_layers in [0usize, 1, 17, 33, 47, 48, 60] {
                let plan =
                    place_routed_experts_units(&g, &EpUnitSource::WholeLayers { vram_layers })
                        .unwrap();
                assert!(plan.equals_whole_layers(vram_layers));
                assert_eq!(plan.whole_layer_split(), Some(vram_layers.min(48)));
                for rank in 0..ranks {
                    for layer in 0..48 {
                        let want = layer < vram_layers;
                        for expert in 0..512 {
                            assert_eq!(
                                plan.is_resident(rank, layer, expert),
                                want && expert % ranks == rank,
                                "rank {rank} layer {layer} expert {expert} vram_layers {vram_layers}"
                            );
                        }
                    }
                    assert_eq!(
                        plan.resident_units(rank),
                        vram_layers.min(48) * g.owned_per_layer(rank)
                    );
                    assert_eq!(plan.layer_counts(rank).1, 0);
                }
                // Manifest cross-check: residency per layer entry and pinned bytes.
                let mut weights = Vec::new();
                for layer in 0..48 {
                    for proj in ["gate_up_proj", "down_proj"] {
                        weights.push(entry(
                            format!("model.language_model.layers.{layer}.mlp.experts.{proj}"),
                            layer,
                            512,
                        ));
                    }
                }
                place_routed_experts(&mut weights, vram_layers);
                for rank in 0..ranks {
                    let bytes_of = |entry: &WeightEntry| {
                        let per_expert = if entry.name.ends_with("gate_up_proj") {
                            GATE_UP
                        } else {
                            DOWN
                        };
                        Some(per_expert * g.owned_per_layer(rank) as u64)
                    };
                    let manifest_host = host_mapped_bytes(&weights, bytes_of).unwrap();
                    assert_eq!(plan.host_pinned_bytes(rank), manifest_host, "rank {rank}");
                    for (layer, state) in (0..48).map(|l| (l, plan.layer_state(rank, l))) {
                        let host = weights
                            .iter()
                            .find(|e| e.layer == Some(layer))
                            .unwrap()
                            .residency
                            == WeightResidency::HostMapped;
                        assert_eq!(state == EpLayerState::Host, host, "layer {layer}");
                    }
                }
            }
        }
    }

    #[test]
    fn ranking_walk_keeps_owned_units_until_the_budget_is_spent() {
        let g = geometry(2);
        let rank = permutation(g.total_ids(), 7);
        // Budget for exactly 1,000 units on rank 0 and 500 on rank 1 plus pads.
        let budget = vec![
            1_000 * g.unit_bytes() + 48 * 2 * VRAM_ARENA_TAIL_PAD_BYTES,
            500 * g.unit_bytes() + 48 * 2 * VRAM_ARENA_TAIL_PAD_BYTES,
        ];
        let plan = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &rank,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: budget,
            },
        )
        .unwrap();
        assert_eq!(plan.resident_units(0), 1_000);
        assert_eq!(plan.resident_units(1), 500);
        // The kept set is the ranked prefix per rank: the hottest owned units.
        for r in 0..2 {
            let want: Vec<usize> = rank
                .iter()
                .map(|&id| id as usize)
                .filter(|id| (id % 512) % 2 == r)
                .take(plan.resident_units(r))
                .collect();
            let mut got = Vec::new();
            for layer in 0..48 {
                for expert in 0..512 {
                    if plan.is_resident(r, layer, expert) {
                        assert_eq!(expert % 2, r, "never an expert the rank does not own");
                        got.push(layer * 512 + expert);
                    }
                }
            }
            let mut want_sorted = want;
            want_sorted.sort_unstable();
            assert_eq!(got, want_sorted, "rank {r}");
        }
        assert!(plan.vram_bytes(0) <= 1_000 * g.unit_bytes() + 48 * 2 * VRAM_ARENA_TAIL_PAD_BYTES);
        assert_eq!(plan.owned_units(0), 48 * 256);
        assert_eq!(plan.host_units(0), 48 * 256 - 1_000);
    }

    /// A whole-layers ranking at the whole-layer budget is the whole-layer split
    /// (identical bytes): rank the units of layers 0..N hottest, give each rank
    /// N layers of bytes plus the arena pads.
    #[test]
    fn a_whole_layers_ranking_reproduces_the_whole_layer_split() {
        let g = geometry(2);
        let layers_kept = 33usize;
        let mut rank: Vec<u32> = (0..g.total_ids() as u32).collect(); // layer-major: layers 0.. first
        rank.sort_by_key(|&id| id as usize / 512);
        let per_rank = layers_kept as u64 * 256 * g.unit_bytes()
            + layers_kept as u64 * 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        let plan = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &rank,
                    counts: None,
                    sha256: Some("00".repeat(32).as_str()),
                },
                rank_budget_bytes: vec![per_rank; 2],
            },
        )
        .unwrap();
        let whole = place_routed_experts_units(
            &g,
            &EpUnitSource::WholeLayers {
                vram_layers: layers_kept,
            },
        )
        .unwrap();
        assert_eq!(plan.resident, whole.resident);
        assert_eq!(plan.resident_units(0), 8_448);
        assert_eq!(plan.whole_layer_split(), Some(33));
        assert_eq!(plan.host_pinned_bytes(0), whole.host_pinned_bytes(0));
    }

    #[test]
    fn coverage_is_the_counts_mass_of_the_kept_units() {
        let g = EpUnitGeometry {
            layers: 2,
            experts_per_layer: 4,
            ranks: 2,
            gate_up_bytes: 256,
            down_bytes: 256,
        };
        // Ids: layer 0 experts 0..4 = ids 0..4, layer 1 = ids 4..8. Counts favour id 6, 1, 0.
        let counts = vec![10, 30, 0, 5, 0, 0, 100, 5];
        let rank = vec![6, 1, 0, 3, 7, 2, 4, 5];
        let budget_per_rank = 2 * g.unit_bytes() + 4 * VRAM_ARENA_TAIL_PAD_BYTES;
        let plan = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &rank,
                    counts: Some(&counts),
                    sha256: Some(&"ab".repeat(32)),
                },
                rank_budget_bytes: vec![budget_per_rank; 2],
            },
        )
        .unwrap();
        // Rank 0 owns experts 0, 2: ranked 6 (l1 e2), 0 (l0 e0); then 2, 4 (no budget left).
        assert!(plan.is_resident(0, 1, 2) && plan.is_resident(0, 0, 0));
        assert_eq!(plan.resident_units(0), 2);
        // Rank 1 owns experts 1, 3: ranked 1 (l0 e1), 3 (l0 e3).
        assert!(plan.is_resident(1, 0, 1) && plan.is_resident(1, 0, 3));
        assert_eq!(plan.resident_units(1), 2);
        // Rank 0 mass: owned = ids 0,2,4,6 = 10+0+0+100 = 110; kept ids 6,0 = 110 -> 100%.
        assert!((plan.coverage(0).unwrap() - 1.0).abs() < 1e-12);
        // Rank 1 mass: owned = ids 1,3,5,7 = 30+5+0+5 = 40; kept 1,3 = 35.
        assert!((plan.coverage(1).unwrap() - 35.0 / 40.0).abs() < 1e-12);
        assert_eq!(plan.layer_state(1, 0), EpLayerState::Resident);
        assert_eq!(plan.layer_state(1, 1), EpLayerState::Host);
        assert_eq!(
            plan.layer_state(0, 0),
            EpLayerState::Mixed {
                resident: 1,
                host: 1
            }
        );
    }

    #[test]
    fn a_rank_closes_at_the_first_unit_that_does_not_fit() {
        let g = EpUnitGeometry {
            layers: 1,
            experts_per_layer: 4,
            ranks: 2,
            gate_up_bytes: 256,
            down_bytes: 256,
        };
        // Budget fits one unit plus the layer's pads, not two.
        let budget = g.unit_bytes() + 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        let rank = vec![0, 2, 1, 3];
        let plan = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &rank,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![budget, budget + g.unit_bytes()],
            },
        )
        .unwrap();
        assert_eq!(plan.resident_units(0), 1);
        assert_eq!(plan.resident_units(1), 2);
        assert!(plan.is_resident(0, 0, 0) && !plan.is_resident(0, 0, 2));
    }

    #[test]
    fn a_malformed_ranking_is_refused() {
        let g = EpUnitGeometry {
            layers: 1,
            experts_per_layer: 4,
            ranks: 2,
            gate_up_bytes: 256,
            down_bytes: 256,
        };
        let run = |rank: &[u32]| {
            place_routed_experts_units(
                &g,
                &EpUnitSource::Ranking {
                    ranking: EpRanking {
                        rank,
                        counts: None,
                        sha256: None,
                    },
                    rank_budget_bytes: vec![1 << 20; 2],
                },
            )
        };
        assert!(run(&[0, 1, 2]).unwrap_err().contains("3 ids"));
        assert!(run(&[0, 1, 2, 2]).unwrap_err().contains("twice"));
        assert!(run(&[0, 1, 2, 9]).unwrap_err().contains("outside"));
        assert!(run(&[0, 1, 2, 3]).is_ok());
        let wrong = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &[0, 1, 2, 3],
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![1 << 20],
            },
        );
        assert!(wrong.unwrap_err().contains("rank budgets"));
    }

    #[test]
    fn receipt_line_formats_both_sources() {
        let g = geometry(2);
        let whole =
            place_routed_experts_units(&g, &EpUnitSource::WholeLayers { vram_layers: 33 }).unwrap();
        let line = whole.receipt_line(0);
        assert!(line.starts_with("qwen4 EP expert tier rank 0/2: resident 8448/12288 units"));
        assert!(line.contains("layers 33 resident, 0 mixed, 15 host (33 touched)"));
        assert!(line.contains("whole-layer equivalent 33.00"));
        assert!(line.ends_with("ranking none (whole layers)"));

        let rank = permutation(g.total_ids(), 3);
        let counts: Vec<u64> = (0..g.total_ids() as u64).map(|i| 1 + i % 7).collect();
        let budget = 8_448 * g.unit_bytes() + 48 * 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        let sha = "ef".repeat(32);
        let plan = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &rank,
                    counts: Some(&counts),
                    sha256: Some(&sha),
                },
                rank_budget_bytes: vec![budget; 2],
            },
        )
        .unwrap();
        let line = plan.receipt_line(1);
        println!("{line}");
        assert!(
            line.contains("rank 1/2: resident 8448/12288 units"),
            "{line}"
        );
        assert!(
            line.contains("ranking sha256 efefefefefefefef coverage "),
            "{line}"
        );
        assert_eq!(plan.receipt_lines().len(), 2);
    }
}

#[cfg(test)]
mod planar_arena_tests {
    //! DESIGN F1 x D1: the unit arena of a drop-zero planar plan, CPU oracle. The shipped
    //! (qt44/qt53) stacks are repacked planar into each owned expert's slot; every slot re-read
    //! and dequantised planar must equal the shipped dequant of the same source expert bit for
    //! bit, and the plan charges 2,534,400 B per expert.
    use super::*;
    use hipfire_runtime::mq4_planar::{
        dequant_planar, dequant_shipped, planar_bytes, shipped_bytes, zero_bits_for_scale,
        Mq4PlanarKind, FLASH_NEXT_DOWN, FLASH_NEXT_GATE_UP,
    };

    const EXPERTS: usize = 6;

    fn symmetric_stack(kind: Mq4PlanarKind, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let groups = rows * (k / kind.shipped_group_cols());
        let header = kind.shipped_group_bytes() - 64 * kind.shipped_group_cols() / 128;
        let mut out = vec![0u8; groups * kind.shipped_group_bytes()];
        for group in out.chunks_exact_mut(kind.shipped_group_bytes()) {
            for half in 0..kind.shipped_group_cols() / 128 {
                let scale = (next() % 0x6c00) as u16;
                group[4 * half..4 * half + 2].copy_from_slice(&scale.to_le_bytes());
                group[4 * half + 2..4 * half + 4]
                    .copy_from_slice(&zero_bits_for_scale(scale).to_le_bytes());
            }
            for b in &mut group[header..] {
                *b = next() as u8;
            }
        }
        out
    }

    #[test]
    fn planar_geometry_is_2_534_400_bytes_per_expert() {
        let (gr, gk) = FLASH_NEXT_GATE_UP;
        let (dr, dk) = FLASH_NEXT_DOWN;
        let planar = EpUnitGeometry {
            layers: 1,
            experts_per_layer: EXPERTS,
            ranks: 2,
            gate_up_bytes: planar_bytes(Mq4PlanarKind::GateUp, gr, gk).unwrap() as u64,
            down_bytes: planar_bytes(Mq4PlanarKind::Down, dr, dk).unwrap() as u64,
        };
        assert!(planar.is_mq4_planar());
        assert_eq!(planar.unit_bytes(), 2_534_400);
        let shipped = EpUnitGeometry {
            gate_up_bytes: shipped_bytes(Mq4PlanarKind::GateUp, gr, gk).unwrap() as u64,
            down_bytes: shipped_bytes(Mq4PlanarKind::Down, dr, dk).unwrap() as u64,
            ..planar
        };
        assert!(!shipped.is_mq4_planar());
        assert_eq!(shipped.unit_bytes(), 2_611_200);
        // The unit budget buys more units by exactly the byte ratio.
        assert!(planar.unit_bytes() % 256 == 0 && shipped.unit_bytes() % 256 == 0);
    }

    #[test]
    fn planar_unit_arena_dequants_bit_equal_to_the_shipped_source() {
        let (gr, gk) = FLASH_NEXT_GATE_UP;
        let (dr, dk) = FLASH_NEXT_DOWN;
        let g = EpUnitGeometry {
            layers: 1,
            experts_per_layer: EXPERTS,
            ranks: 2,
            gate_up_bytes: planar_bytes(Mq4PlanarKind::GateUp, gr, gk).unwrap() as u64,
            down_bytes: planar_bytes(Mq4PlanarKind::Down, dr, dk).unwrap() as u64,
        };
        // Rank 0 owns experts 0, 2, 4; keep exactly one of them resident so the layer is
        // MIXED (one VRAM slot, two host slots).
        let ids: Vec<u32> = vec![0, 2, 4, 1, 3, 5];
        let budget = g.unit_bytes() + 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        let plan = place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &ids,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![budget; 2],
            },
        )
        .unwrap();
        let map = plan.layer_map(0, 0);
        assert!(map.is_mixed(), "the test needs a mixed layer");
        let gu_src = shipped_bytes(Mq4PlanarKind::GateUp, gr, gk).unwrap();
        let dn_src = shipped_bytes(Mq4PlanarKind::Down, dr, dk).unwrap();
        let mut gate_up_all = Vec::new();
        let mut down_all = Vec::new();
        for expert in 0..EXPERTS {
            gate_up_all.extend(symmetric_stack(
                Mq4PlanarKind::GateUp,
                gr,
                gk,
                11 + expert as u64,
            ));
            down_all.extend(symmetric_stack(
                Mq4PlanarKind::Down,
                dr,
                dk,
                97 + expert as u64,
            ));
        }
        for tier in [UnitTier::Vram, UnitTier::Host] {
            let arena = map
                .gather_arena_mq4_planar(tier, &gate_up_all, &down_all)
                .unwrap();
            assert_eq!(arena.len(), map.arena_entry_bytes(tier));
            for expert in (0..EXPERTS).filter(|e| e % 2 == 0) {
                let (slot_tier, gu_off, dn_off) = map.expert_offsets(expert).unwrap();
                if slot_tier != tier {
                    continue;
                }
                let gu = &arena[gu_off..gu_off + map.gate_up_bytes()];
                let dn = &arena[dn_off..dn_off + map.down_bytes()];
                let src_gu = &gate_up_all[expert * gu_src..(expert + 1) * gu_src];
                let src_dn = &down_all[expert * dn_src..(expert + 1) * dn_src];
                for r in (0..gr).step_by(61) {
                    for c in (0..gk).step_by(3) {
                        assert_eq!(
                            dequant_shipped(Mq4PlanarKind::GateUp, gk, src_gu, r, c).to_bits(),
                            dequant_planar(gr, gk, gu, r, c).to_bits(),
                            "{tier:?} expert {expert} gate/up r{r} c{c}"
                        );
                    }
                }
                for r in (0..dr).step_by(97) {
                    for c in (0..dk).step_by(3) {
                        assert_eq!(
                            dequant_shipped(Mq4PlanarKind::Down, dk, src_dn, r, c).to_bits(),
                            dequant_planar(dr, dk, dn, r, c).to_bits(),
                            "{tier:?} expert {expert} down r{r} c{c}"
                        );
                    }
                }
            }
        }
        // A source stack of the wrong (planar) size is refused.
        assert!(map
            .gather_arena_mq4_planar(UnitTier::Vram, &gate_up_all[1..], &down_all)
            .is_err());
    }
}

#[cfg(test)]
mod layer_map_tests {
    use super::*;

    const GU: usize = 512;
    const DN: usize = 256;

    fn source_stacks(experts: usize, layer: usize) -> (Vec<u8>, Vec<u8>) {
        let mut gate_up = Vec::with_capacity(experts * GU);
        let mut down = Vec::with_capacity(experts * DN);
        for expert in 0..experts {
            gate_up.extend((0..GU).map(|i| (layer * 31 + expert * 7 + i) as u8));
            down.extend((0..DN).map(|i| (layer * 17 + expert * 13 + i * 3) as u8));
        }
        (gate_up, down)
    }

    fn plan(ranks: usize, experts: usize, kept_per_rank: usize) -> EpUnitPlan {
        let g = EpUnitGeometry {
            layers: 3,
            experts_per_layer: experts,
            ranks,
            gate_up_bytes: GU as u64,
            down_bytes: DN as u64,
        };
        // Rank order: layer-major, but within a layer every third expert first so the
        // resident set is neither a prefix nor a stride.
        let mut ids: Vec<u32> = Vec::new();
        for layer in 0..3 {
            for pass in 0..3 {
                for expert in (0..experts).filter(|e| e % 3 == pass) {
                    ids.push((layer * experts + expert) as u32);
                }
            }
        }
        let budget = kept_per_rank as u64 * (GU + DN) as u64 + 3 * 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        place_routed_experts_units(
            &g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &ids,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![budget; ranks],
            },
        )
        .unwrap()
    }

    /// The arena round trip: every unit re-read through the slot map equals the
    /// source bytes, for both tiers, three layers, two and three ranks, uneven
    /// ownership included.
    #[test]
    fn arena_round_trip_re_reads_every_unit() {
        for (ranks, experts) in [(2usize, 32usize), (3, 32)] {
            let plan = plan(ranks, experts, 11);
            for rank in 0..ranks {
                for layer in 0..3 {
                    let map = plan.layer_map(rank, layer);
                    let (gate_up_all, down_all) = source_stacks(experts, layer);
                    let vram = map
                        .gather_arena(UnitTier::Vram, &gate_up_all, &down_all)
                        .unwrap();
                    let host = map
                        .gather_arena(UnitTier::Host, &gate_up_all, &down_all)
                        .unwrap();
                    assert_eq!(vram.len(), map.arena_entry_bytes(UnitTier::Vram));
                    assert_eq!(host.len(), map.arena_entry_bytes(UnitTier::Host));
                    let owned = (0..experts).filter(|e| e % ranks == rank).count();
                    assert_eq!(map.count(UnitTier::Vram) + map.count(UnitTier::Host), owned);
                    for expert in 0..experts {
                        let Some((tier, gate_up, down)) = map.expert_offsets(expert) else {
                            assert_ne!(expert % ranks, rank);
                            continue;
                        };
                        assert_eq!(expert % ranks, rank);
                        assert_eq!(
                            map.slot(expert).unwrap().tier == UnitTier::Vram,
                            plan.is_resident(rank, layer, expert)
                        );
                        let arena = if tier == UnitTier::Vram { &vram } else { &host };
                        assert_eq!(
                            &arena[gate_up..gate_up + GU],
                            &gate_up_all[expert * GU..(expert + 1) * GU],
                            "rank {rank} layer {layer} expert {expert} gate/up"
                        );
                        assert_eq!(
                            &arena[down..down + DN],
                            &down_all[expert * DN..(expert + 1) * DN],
                            "rank {rank} layer {layer} expert {expert} down"
                        );
                    }
                    // The VRAM tail pad is zeroed and the two blocks tile the payload.
                    let payload = map.arena_payload_bytes(UnitTier::Vram);
                    assert!(vram[payload..].iter().all(|&byte| byte == 0));
                    assert_eq!(
                        map.down_offset(UnitSlot {
                            tier: UnitTier::Vram,
                            slot: 0
                        }),
                        map.count(UnitTier::Vram) * GU
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_layers_need_both_tiers_and_slots_are_ascending() {
        let plan = plan(2, 32, 11);
        for rank in 0..2 {
            for layer in plan.mixed_layers(rank) {
                let map = plan.layer_map(rank, layer);
                assert!(map.is_mixed());
                let mut last = [None, None];
                for expert in 0..32 {
                    if let Some(slot) = map.slot(expert) {
                        let at = usize::from(slot.tier == UnitTier::Host);
                        assert_eq!(slot.slot, last[at].map_or(0, |previous| previous + 1));
                        last[at] = Some(slot.slot);
                    }
                }
            }
        }
        // A whole-layer plan has no mixed layers and one tier per layer map.
        let whole = place_routed_experts_units(
            &EpUnitGeometry {
                layers: 3,
                experts_per_layer: 32,
                ranks: 2,
                gate_up_bytes: GU as u64,
                down_bytes: DN as u64,
            },
            &EpUnitSource::WholeLayers { vram_layers: 1 },
        )
        .unwrap();
        assert!(whole.mixed_layers(0).is_empty());
        assert!(!whole.layer_map(0, 0).is_mixed());
        assert_eq!(whole.layer_map(0, 0).count(UnitTier::Host), 0);
        assert_eq!(whole.layer_map(0, 2).count(UnitTier::Vram), 0);
    }

    #[test]
    fn gather_refuses_wrong_stack_sizes() {
        let plan = plan(2, 32, 11);
        let map = plan.layer_map(0, 0);
        let (gate_up_all, down_all) = source_stacks(32, 0);
        assert!(map
            .gather_arena(UnitTier::Host, &gate_up_all[1..], &down_all)
            .is_err());
        assert!(map
            .gather_arena(UnitTier::Host, &gate_up_all, &down_all[..10])
            .is_err());
    }

    #[test]
    fn rank_budgets_are_the_bytes_auto_divides_by_the_layer_size() {
        use crate::expert_residency::auto_vram_layers;
        let layer = 637 << 20;
        for (free, non_expert, reserve) in [
            (31u64 << 30, 5u64 << 30, 7u64 << 30),
            (20 << 30, 5 << 30, 7 << 30),
            (3 << 30, 5 << 30, 7 << 30),
        ] {
            let budgets = ep_unit_rank_budgets(&[free, free / 2], non_expert, reserve);
            assert_eq!(budgets.len(), 2);
            for (rank, &f) in [free, free / 2].iter().enumerate() {
                assert_eq!(
                    (budgets[rank] / layer).min(48) as usize,
                    auto_vram_layers(f, non_expert, layer, 48, reserve)
                );
            }
        }
        assert_eq!(ep_unit_rank_budgets(&[1 << 30], 5 << 30, 1 << 30), vec![0]);
    }
}

#[cfg(test)]
mod loader_support_tests {
    use super::*;
    use crate::expert_residency::{
        ep_unit_host_bytes_ranks, host_mapped_bytes_ranks, place_routed_experts,
    };
    use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
    use rdna_compute::DType;

    // Flash-Next mq4-xts per-expert bytes.
    const GATE_UP: u64 = 1280 * 10 * 136;
    const DOWN: u64 = 2560 * 5 * 68;

    #[test]
    fn the_flag_names_an_explicit_path_and_off_values_are_off() {
        assert_eq!(EP_EXPERT_HOTSET_ENV, "HIPFIRE_QWEN4_EP_EXPERT_HOTSET");
        assert_eq!(ep_units_policy(None), EpUnitsPolicy::Default);
        for off in [Some(""), Some("   "), Some("0"), Some(" 0 ")] {
            assert_eq!(ep_units_policy(off), EpUnitsPolicy::Off, "{off:?}");
        }
        assert_eq!(
            ep_units_policy(Some(" /models/fn.mq4.expert-hotset ")),
            EpUnitsPolicy::Explicit(std::path::PathBuf::from("/models/fn.mq4.expert-hotset"))
        );
        assert_eq!(
            ep_units_policy(Some("rel/path")),
            EpUnitsPolicy::Explicit(std::path::PathBuf::from("rel/path"))
        );
    }

    fn discovery(
        gate: bool,
        arch: &'static str,
        ep: usize,
        pp: usize,
    ) -> EpUnitsDiscovery<'static> {
        EpUnitsDiscovery {
            gate,
            gpu_arch: arch,
            facts: EpUnitsFacts {
                ep,
                pp,
                ..Default::default()
            },
            sidecar: std::path::PathBuf::from("/m/base.mq4-xts.ep.expert-hotset"),
            sidecar_exists: true,
        }
    }

    #[test]
    fn the_gate_off_discovers_nothing_but_an_explicit_path_still_applies() {
        let off = discovery(false, "gfx1201", 2, 1);
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Default, &off),
            EpUnitsSource::Off
        );
        let path = std::path::PathBuf::from("/x/h.expert-hotset");
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Explicit(path.clone()), &off),
            EpUnitsSource::Explicit(path)
        );
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Off, &off),
            EpUnitsSource::Off
        );
    }

    #[test]
    fn the_gate_on_discovers_for_ep2_gfx1201_and_zero_is_the_off_switch() {
        let on = discovery(true, "gfx1201", 2, 1);
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Default, &on),
            EpUnitsSource::Discovered(on.sidecar.clone())
        );
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Off, &on),
            EpUnitsSource::Off
        );
        let mut absent = on.clone();
        absent.sidecar_exists = false;
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Default, &absent),
            EpUnitsSource::Missing
        );
        // The hint is for the load that would have discovered the file, nothing else: the other
        // arch, ep != 2, a pipeline, a lever that refuses the units, the gate off, an explicit
        // off or path stay silent.
        let silent = |found: &EpUnitsDiscovery<'_>| {
            assert_eq!(
                ep_units_source(&EpUnitsPolicy::Default, found),
                EpUnitsSource::Off
            )
        };
        for (gate, arch, ep, pp) in [
            (true, "gfx1100", 2, 1),
            (true, "gfx1201", 4, 1),
            (true, "gfx1201", 1, 1),
            (true, "gfx1201", 2, 2),
            (false, "gfx1201", 2, 1),
        ] {
            let mut found = discovery(gate, arch, ep, pp);
            found.sidecar_exists = false;
            silent(&found);
        }
        let mut lever = absent.clone();
        lever.facts.chunk_autofit = true;
        silent(&lever);
        assert_eq!(
            ep_units_source(&EpUnitsPolicy::Off, &absent),
            EpUnitsSource::Off
        );
    }

    #[test]
    fn the_missing_hotset_hint_names_the_pull_tag_for_the_published_flash_next_base_only() {
        let hint = ep_hotset_missing_hint("qwen3.8-flash-next-base.mq4-xts").unwrap();
        assert_eq!(
            hint,
            "EP expert hot set missing: run `hipfire pull qwen3.8:flash-next` to fetch it (placement falls back to whole-layer)"
        );
        assert!(!hint.contains('\n'));
        assert!(ep_hotset_missing_hint("qwen3.8-flash-next.mq4-xts").is_some());
        assert!(ep_hotset_missing_hint("qwen3.8-flash-next-base.mq3-xts").is_none());
        assert!(ep_hotset_missing_hint("qwen3.8-27b.mq4-xts").is_none());
    }

    #[test]
    fn discovery_is_silent_off_for_single_device_and_pipeline_and_skips_the_rest() {
        for (ep, pp) in [(1, 1), (0, 1), (2, 2)] {
            assert_eq!(
                ep_units_source(&EpUnitsPolicy::Default, &discovery(true, "gfx1201", ep, pp)),
                EpUnitsSource::Off,
                "ep {ep} pp {pp}"
            );
        }
        let skipped =
            |found: &EpUnitsDiscovery<'_>| match ep_units_source(&EpUnitsPolicy::Default, found) {
                EpUnitsSource::Skipped(why) => why,
                other => panic!("expected Skipped, got {other:?}"),
            };
        assert!(skipped(&discovery(true, "gfx1201", 4, 1)).contains("ep = 2"));
        assert!(skipped(&discovery(true, "gfx1100", 2, 1)).contains("gfx1201 only"));
        for lever in 0..3 {
            let mut found = discovery(true, "gfx1201", 2, 1);
            match lever {
                0 => found.facts.single_card_hotset = true,
                1 => found.facts.chunk_autofit = true,
                _ => found.facts.lab_resident_cap = true,
            }
            assert!(skipped(&found).starts_with(EP_EXPERT_HOTSET_ENV), "{lever}");
        }
    }

    #[test]
    fn the_unit_reserve_is_opt_in_only() {
        // Only an exact 1 turns the decoupled unit reserve on, under any gate: the shipped
        // default plans the units under the ordinary reserve.
        for (raw, want) in [
            (None, false),
            (Some("1"), true),
            (Some("0"), false),
            (Some("yes"), false),
        ] {
            assert_eq!(ep_unit_reserve_on(raw), want, "{raw:?}");
        }
    }

    #[test]
    fn arena_entry_names_carry_their_tier_and_source_stacks() {
        let prefix = "model.language_model.layers.17";
        let vram = format!("{prefix}{UNIT_VRAM_ARENA_SUFFIX}");
        let host = format!("{prefix}{UNIT_HOST_ARENA_SUFFIX}");
        assert_eq!(unit_arena_tier(&vram), Some(UnitTier::Vram));
        assert_eq!(unit_arena_tier(&host), Some(UnitTier::Host));
        assert!(is_unit_arena_name(&vram) && is_unit_arena_name(&host));
        for ordinary in [
            format!("{prefix}.mlp.experts.gate_up_proj"),
            format!("{prefix}.mlp.experts.down_proj"),
            format!("{prefix}.mlp.experts.slab_arena"),
            "model.embed_tokens.weight".to_string(),
        ] {
            assert_eq!(unit_arena_tier(&ordinary), None, "{ordinary}");
            assert_eq!(unit_arena_source_names(&ordinary), None);
        }
        for name in [&vram, &host] {
            assert_eq!(
                unit_arena_source_names(name),
                Some((
                    format!("{prefix}.mlp.experts.gate_up_proj"),
                    format!("{prefix}.mlp.experts.down_proj")
                ))
            );
        }
    }

    /// The payload helper is `gather_arena` selected by the entry name: both
    /// tiers of a two-rank, eight-expert mixed layer.
    #[test]
    fn arena_entry_bytes_is_gather_arena_picked_by_name() {
        const GU: usize = 256;
        const DN: usize = 256;
        let geometry = EpUnitGeometry {
            layers: 1,
            experts_per_layer: 8,
            ranks: 2,
            gate_up_bytes: GU as u64,
            down_bytes: DN as u64,
        };
        let ids: Vec<u32> = (0..8).collect();
        let plan = place_routed_experts_units(
            &geometry,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &ids,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![
                    2 * geometry.unit_bytes() + 2 * VRAM_ARENA_TAIL_PAD_BYTES;
                    2
                ],
            },
        )
        .unwrap();
        let gate_up: Vec<u8> = (0..8 * GU).map(|i| (i % 251) as u8).collect();
        let down: Vec<u8> = (0..8 * DN).map(|i| (i % 241) as u8).collect();
        for rank in 0..2 {
            let map = plan.layer_map(rank, 0);
            assert!(map.is_mixed());
            for (suffix, tier) in [
                (UNIT_VRAM_ARENA_SUFFIX, UnitTier::Vram),
                (UNIT_HOST_ARENA_SUFFIX, UnitTier::Host),
            ] {
                let name = format!("model.language_model.layers.0{suffix}");
                assert_eq!(
                    unit_arena_entry_bytes(&map, &name, &gate_up, &down).unwrap(),
                    map.gather_arena(tier, &gate_up, &down).unwrap()
                );
            }
            assert!(unit_arena_entry_bytes(
                &map,
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                &gate_up,
                &down
            )
            .is_err());
            assert!(unit_arena_entry_bytes(
                &map,
                &format!("model.language_model.layers.0{UNIT_VRAM_ARENA_SUFFIX}"),
                &gate_up[..GU],
                &down
            )
            .is_err());
        }
    }

    #[test]
    fn geometry_comes_from_uniform_aligned_stack_sizes() {
        let stacks = |gu: u64, dn: u64| (vec![gu; 48], vec![dn; 48]);
        let (gate_up, down) = stacks(512 * GATE_UP, 512 * DOWN);
        let geometry = ep_unit_geometry(48, 512, 2, &gate_up, &down).unwrap();
        assert_eq!(geometry.gate_up_bytes, GATE_UP);
        assert_eq!(geometry.down_bytes, DOWN);
        assert_eq!(
            (geometry.layers, geometry.experts_per_layer, geometry.ranks),
            (48, 512, 2)
        );
        // A layer of another size.
        let mut odd = gate_up.clone();
        odd[7] += 256 * 512;
        assert!(ep_unit_geometry(48, 512, 2, &odd, &down).is_err());
        // Not a whole number of experts.
        let (bad, _) = stacks(512 * GATE_UP + 1, 0);
        assert!(ep_unit_geometry(48, 512, 2, &bad, &down).is_err());
        // Unaligned per-expert bytes.
        let (unaligned, _) = stacks(512 * (GATE_UP + 8), 0);
        assert!(ep_unit_geometry(48, 512, 2, &unaligned, &down).is_err());
        // Wrong layer count, empty stacks, fewer experts than ranks.
        assert!(ep_unit_geometry(47, 512, 2, &gate_up, &down).is_err());
        let (empty, _) = stacks(0, 0);
        assert!(ep_unit_geometry(48, 512, 2, &empty, &down).is_err());
        assert!(ep_unit_geometry(48, 512, 600, &gate_up, &down).is_err());
    }

    #[test]
    fn ep_unit_stage_loader_refuses_only_the_remaining_incompatibilities() {
        let ok = EpUnitsFacts {
            ep: 2,
            pp: 1,
            ..EpUnitsFacts::default()
        };
        assert_eq!(ep_units_refusal(&ok), Ok(()));
        let refused = |facts: EpUnitsFacts, needle: &str| {
            let error = ep_units_refusal(&facts).unwrap_err();
            assert!(error.starts_with(EP_EXPERT_HOTSET_ENV), "{error}");
            assert!(error.contains(needle), "{error}");
        };
        refused(EpUnitsFacts { ep: 1, ..ok }, "needs expert parallelism");
        refused(EpUnitsFacts { ep: 0, ..ok }, "needs expert parallelism");
        refused(EpUnitsFacts { pp: 2, ..ok }, "pipeline parallelism");
        refused(
            EpUnitsFacts {
                single_card_hotset: true,
                ..ok
            },
            "HIPFIRE_QWEN4_EXPERT_HOTSET",
        );
        refused(
            EpUnitsFacts {
                chunk_autofit: true,
                ..ok
            },
            "HIPFIRE_EP_CHUNK_AUTOFIT",
        );
        refused(
            EpUnitsFacts {
                lab_resident_cap: true,
                ..ok
            },
            "HIPFIRE_EP_LAB_RESIDENT_LAYERS",
        );
        // The EP expert stage is no incompatibility: no refusal names its lever.
        for facts in [
            EpUnitsFacts { ep: 1, ..ok },
            EpUnitsFacts { pp: 2, ..ok },
            EpUnitsFacts {
                single_card_hotset: true,
                ..ok
            },
            EpUnitsFacts {
                chunk_autofit: true,
                ..ok
            },
            EpUnitsFacts {
                lab_resident_cap: true,
                ..ok
            },
        ] {
            let error = ep_units_refusal(&facts).unwrap_err();
            assert!(!error.contains(EP_EXPERT_STAGE_ENV), "{error}");
        }
    }

    /// A whole-layers plan's per-rank pinned census equals the manifest census
    /// of the whole-layer placement, MTP head and embedding included, for
    /// even and uneven splits.
    #[test]
    fn unit_host_census_equals_the_manifest_census_for_a_whole_layers_plan() {
        let entry = |name: String, layer: Option<usize>| {
            let mut entry = WeightEntry::model(
                name,
                vec![512, 4, 256],
                DType::MQ4G256V2,
                ShardPolicy::Replicate,
            );
            entry.layer = layer;
            entry
        };
        let bytes_of = |entry: &WeightEntry| {
            Some(if entry.name == crate::weights::QWEN4_EMBED {
                5_000_000
            } else if entry.name.ends_with("gate_up_proj") {
                512 * GATE_UP
            } else {
                512 * DOWN
            })
        };
        for ranks in [2usize, 3] {
            let geometry = EpUnitGeometry {
                layers: 48,
                experts_per_layer: 512,
                ranks,
                gate_up_bytes: GATE_UP,
                down_bytes: DOWN,
            };
            for vram_layers in [0usize, 5, 31, 48] {
                let mut weights = Vec::new();
                for layer in 0..48 {
                    for proj in ["gate_up_proj", "down_proj"] {
                        weights.push(entry(
                            format!("model.language_model.layers.{layer}.mlp.experts.{proj}"),
                            Some(layer),
                        ));
                    }
                }
                for proj in ["gate_up_proj", "down_proj"] {
                    weights.push(entry(format!("mtp.layers.0.mlp.experts.{proj}"), None));
                }
                let mut embed = entry(crate::weights::QWEN4_EMBED.to_string(), None);
                embed.logical_shape = vec![4, 4];
                weights.push(embed);
                place_routed_experts(&mut weights, vram_layers);
                crate::expert_residency::place_embedding_host(&mut weights);
                let plan = place_routed_experts_units(
                    &geometry,
                    &EpUnitSource::WholeLayers { vram_layers },
                )
                .unwrap();
                assert_eq!(
                    ep_unit_host_bytes_ranks(&weights, bytes_of, &plan).unwrap(),
                    host_mapped_bytes_ranks(&weights, bytes_of, ranks).unwrap(),
                    "ranks {ranks} vram_layers {vram_layers}"
                );
            }
        }
    }
}

#[cfg(test)]
mod ep_unit_stage_tests {
    use super::*;

    const GATE_UP: u64 = 1280 * 10 * 136;
    const DOWN: u64 = 2560 * 5 * 68;

    fn geometry(layers: usize, experts: usize, ranks: usize) -> EpUnitGeometry {
        EpUnitGeometry {
            layers,
            experts_per_layer: experts,
            ranks,
            gate_up_bytes: GATE_UP,
            down_bytes: DOWN,
        }
    }

    fn permutation(n: usize, mut state: u64) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ids.swap(i, (state >> 33) as usize % (i + 1));
        }
        ids
    }

    fn ranking_plan(g: &EpUnitGeometry, order: &[u32], budget: u64) -> EpUnitPlan {
        place_routed_experts_units(
            g,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: order,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![budget; g.ranks],
            },
        )
        .unwrap()
    }

    fn staged(g: &EpUnitGeometry, order: &[u32], budget: u64, on: bool) -> EpUnitStagedPlan {
        place_routed_experts_units_staged(
            g,
            EpRanking {
                rank: order,
                counts: None,
                sha256: None,
            },
            &vec![budget; g.ranks],
            on,
        )
        .unwrap()
    }

    #[test]
    fn ep_unit_stage_chooser_is_todays_path_for_no_plan_and_whole_layer_plans() {
        assert_eq!(ep_unit_stage_kind(None), EpUnitStageKind::Donors);
        let g = geometry(8, 32, 2);
        let whole =
            place_routed_experts_units(&g, &EpUnitSource::WholeLayers { vram_layers: 3 }).unwrap();
        assert_eq!(ep_unit_stage_kind(Some(&whole)), EpUnitStageKind::Donors);
        let none =
            place_routed_experts_units(&g, &EpUnitSource::WholeLayers { vram_layers: 0 }).unwrap();
        assert_eq!(ep_unit_stage_kind(Some(&none)), EpUnitStageKind::Donors);
        // A per-expert plan (mixed layers) takes the unit stage.
        let order = permutation(8 * 32, 7);
        let mixed = ranking_plan(&g, &order, 40 * g.unit_bytes());
        assert!(mixed.whole_layer_split().is_none());
        assert_eq!(ep_unit_stage_kind(Some(&mixed)), EpUnitStageKind::Units);
    }

    #[test]
    fn ep_unit_stage_switches_are_one_shared_predicate() {
        let on = EpUnitStageSwitches {
            arch: "gfx1201",
            ..EpUnitStageSwitches::default()
        };
        assert!(ep_unit_stage_switches_on(&on));
        assert!(ep_unit_stage_switches_on(&EpUnitStageSwitches {
            expert_stage: Some("1"),
            ..on
        }));
        for off in [
            EpUnitStageSwitches {
                expert_stage: Some("0"),
                ..on
            },
            EpUnitStageSwitches {
                ep_stage: Some("0"),
                ..on
            },
            EpUnitStageSwitches {
                owner_routed: Some("0"),
                ..on
            },
            // Only gfx1201 builds the EP stage, whatever the explicit switch.
            EpUnitStageSwitches {
                arch: "gfx1100",
                expert_stage: Some("1"),
                ..on
            },
            EpUnitStageSwitches {
                arch: "gfx1151",
                ..on
            },
        ] {
            assert!(!ep_unit_stage_switches_on(&off), "{off:?}");
        }
    }

    #[test]
    fn ep_unit_stage_charge_is_zero_unless_a_stage_is_built() {
        let g = geometry(8, 32, 2);
        let order = permutation(8 * 32, 11);
        let mixed = ranking_plan(&g, &order, 40 * g.unit_bytes());
        let charged = ep_unit_plan_stage_charge(&mixed, true);
        assert_eq!(charged.capacity, mixed.max_stage_host_region_bytes());
        assert!(charged.capacity > 0);
        assert_eq!(
            charged.bytes,
            2 * (charged.capacity + EP_UNIT_STAGE_PAD_BYTES) + ep_unit_stage_table_bytes(&g)
        );
        // Stage off: nothing.
        assert_eq!(
            ep_unit_plan_stage_charge(&mixed, false),
            EpUnitStageCharge::default()
        );
        // Whole-layer plan (donor stage) and a fully resident plan: nothing.
        let whole =
            place_routed_experts_units(&g, &EpUnitSource::WholeLayers { vram_layers: 3 }).unwrap();
        assert_eq!(
            ep_unit_plan_stage_charge(&whole, true),
            EpUnitStageCharge::default()
        );
        let all = ranking_plan(&g, &order, u64::MAX / 4);
        assert!(all.max_stage_host_region_bytes() == 0);
        assert_eq!(
            ep_unit_plan_stage_charge(&all, true),
            EpUnitStageCharge::default()
        );
        // Charge > 0 exactly when some rank builds.
        for plan in [&mixed, &whole, &all] {
            let builds = (0..g.ranks).any(|rank| ep_unit_stage_builds(true, Some(plan), rank));
            assert_eq!(builds, ep_unit_plan_stage_charge(plan, true).bytes > 0);
        }
        assert!(!ep_unit_stage_builds(true, None, 0));
        assert!(!ep_unit_stage_builds(false, Some(&mixed), 0));
    }

    #[test]
    fn ep_unit_stage_host_regions_follow_the_layer_state() {
        let g = geometry(8, 32, 2);
        let order = permutation(8 * 32, 3);
        let plan = ranking_plan(&g, &order, 30 * g.unit_bytes());
        let mut saw_mixed = false;
        for rank in 0..g.ranks {
            for layer in 0..g.layers {
                let bytes = plan.stage_host_region_bytes(rank, layer);
                match plan.layer_state(rank, layer) {
                    EpLayerState::Resident => assert_eq!(bytes, 0),
                    EpLayerState::Host => {
                        assert_eq!(bytes, g.owned_per_layer(rank) as u64 * g.unit_bytes());
                    }
                    EpLayerState::Mixed { host, .. } => {
                        saw_mixed = true;
                        assert_eq!(bytes, host as u64 * g.unit_bytes());
                        // The host arena the weights hold is exactly this region.
                        assert_eq!(
                            plan.layer_map(rank, layer)
                                .arena_entry_bytes(UnitTier::Host)
                                as u64,
                            bytes
                        );
                    }
                }
            }
            assert!(plan.rank_has_host_layer(rank));
        }
        assert!(saw_mixed);
        // A wholly host layer stages both aggregates: every owned expert.
        let host = ranking_plan(&g, &order, 0);
        for rank in 0..g.ranks {
            for layer in 0..g.layers {
                assert_eq!(host.layer_state(rank, layer), EpLayerState::Host);
                assert_eq!(
                    host.stage_host_region_bytes(rank, layer),
                    g.owned_per_layer(rank) as u64 * g.unit_bytes()
                );
            }
        }
    }

    #[test]
    fn ep_unit_stage_capacity_is_monotone_in_the_budget() {
        // The regions the fixed point reads shrink (never grow) as the budget grows.
        let g = geometry(12, 64, 2);
        let order = permutation(12 * 64, 5);
        let mut last = u64::MAX;
        for step in 0..40u64 {
            let plan = ranking_plan(&g, &order, step * 12 * g.unit_bytes());
            let region = plan.max_stage_host_region_bytes();
            assert!(region <= last, "step {step}: {region} > {last}");
            last = region;
        }
        assert_eq!(last, 0, "the largest budget keeps every unit");
    }

    #[test]
    fn ep_unit_stage_capacity_converges_and_covers_the_final_plan() {
        let g = geometry(48, 512, 2);
        let order = permutation(48 * 512, 9);
        let unit = g.unit_bytes();
        for budget_units in [1200u64, 4000, 9000, 14_000] {
            let budget = budget_units * unit;
            let result = staged(&g, &order, budget, true);
            assert!(result.converged, "budget {budget_units}");
            assert!(result.iterations <= EP_UNIT_STAGE_FIXED_POINT_ITERATIONS);
            let plan = &result.plan;
            // C covers every layer of every rank of the final plan.
            assert!(result.stage.capacity >= plan.max_stage_host_region_bytes());
            if plan.max_stage_host_region_bytes() > 0 {
                assert!(result.stage.capacity > 0 && result.stage.bytes > 0);
            }
            // The plan fits the budget net of the stage.
            for rank in 0..g.ranks {
                assert!(
                    plan.vram_bytes(rank) + result.stage.bytes <= budget,
                    "budget {budget_units} rank {rank}"
                );
            }
            // The final plan's own charge is never above what the walk paid.
            assert!(ep_unit_plan_stage_charge(plan, true).bytes <= result.stage.bytes);
            // The stage costs resident units against the plain plan.
            let plain = ranking_plan(&g, &order, budget);
            for rank in 0..g.ranks {
                assert!(plan.resident_units(rank) <= plain.resident_units(rank));
            }
        }
    }

    #[test]
    fn ep_unit_stage_off_is_the_plain_plan_with_nothing_charged() {
        let g = geometry(48, 512, 2);
        let order = permutation(48 * 512, 21);
        let budget = 5000 * g.unit_bytes();
        let result = staged(&g, &order, budget, false);
        assert_eq!(result.plan, ranking_plan(&g, &order, budget));
        assert_eq!(result.stage, EpUnitStageCharge::default());
        assert_eq!((result.iterations, result.converged), (1, true));
    }

    #[test]
    fn ep_unit_stage_capacity_without_a_fixed_point_is_the_whole_layer_region() {
        // A placement whose host region grows by one unit on every call never
        // settles within the cap.
        let g = geometry(1, 64, 1);
        let order: Vec<u32> = (0..64).collect();
        let unit = g.unit_bytes();
        let mut calls = 0u64;
        let result = solve_stage_capacity(&g, true, 0, |_charge| {
            calls += 1;
            // `calls` host units, the rest resident (budget in units + the pads).
            let keep = 64 - calls.min(63);
            Ok(ranking_plan(
                &g,
                &order,
                keep * unit + 2 * VRAM_ARENA_TAIL_PAD_BYTES,
            ))
        })
        .unwrap();
        assert!(!result.converged);
        assert_eq!(calls as usize, EP_UNIT_STAGE_FIXED_POINT_ITERATIONS + 1);
        assert_eq!(result.stage.capacity, 64 * unit);
        assert!(result.stage.capacity >= result.plan.max_stage_host_region_bytes());
    }

    #[test]
    fn ep_unit_stage_receipt_clause_names_the_charge() {
        assert_eq!(
            ep_unit_stage_receipt(&EpUnitStageCharge::default()),
            "stage none"
        );
        let charge = EpUnitStageCharge {
            capacity: 640 << 20,
            bytes: (1280 << 20) + 8192,
            borrowed: None,
        };
        let line = ep_unit_stage_receipt(&charge);
        assert!(line.starts_with("stage 640 MiB x 2 charged"), "{line}");
    }

    #[test]
    fn the_lab_budget_delta_parses_and_applies_per_rank() {
        assert_eq!(ep_unit_budget_delta_mib(None), Ok(0));
        assert_eq!(ep_unit_budget_delta_mib(Some("  ")), Ok(0));
        assert_eq!(ep_unit_budget_delta_mib(Some("1573")), Ok(1573));
        assert_eq!(ep_unit_budget_delta_mib(Some("-512")), Ok(-512));
        assert!(ep_unit_budget_delta_mib(Some("1.5")).is_err());
        assert!(ep_unit_budget_delta_mib(Some("lots")).is_err());
        assert_eq!(
            ep_unit_apply_budget_delta(vec![10 << 20, 0], 0),
            vec![10 << 20, 0]
        );
        assert_eq!(
            ep_unit_apply_budget_delta(vec![10 << 20, 0], 5),
            vec![15 << 20, 5 << 20]
        );
        assert_eq!(
            ep_unit_apply_budget_delta(vec![10 << 20, 0], -5),
            vec![5 << 20, 0]
        );
        assert_eq!(ep_unit_apply_budget_delta(vec![10 << 20], -50), vec![0]);
    }

    fn forced_plan(
        g: &EpUnitGeometry,
        order: &[u32],
        budget: u64,
        forced: &[usize],
    ) -> Result<EpUnitPlan, String> {
        place_routed_experts_units_forced(
            g,
            EpRanking {
                rank: order,
                counts: None,
                sha256: None,
            },
            &vec![budget; g.ranks],
            forced,
        )
    }

    #[test]
    fn forced_layers_are_whole_resident_and_counted_in_the_budget() {
        let g = geometry(48, 512, 2);
        let order = permutation(48 * 512, 31);
        let budget = 6000 * g.unit_bytes();
        let plain = ranking_plan(&g, &order, budget);
        assert_eq!(
            forced_plan(&g, &order, budget, &[]).unwrap(),
            plain,
            "no forced layer is the plain ranking plan"
        );
        let forced = [5usize, 17, 40];
        let plan = forced_plan(&g, &order, budget, &forced).unwrap();
        for rank in 0..2 {
            for &layer in &forced {
                assert_eq!(plan.layer_state(rank, layer), EpLayerState::Resident);
            }
            assert!(plan.vram_bytes(rank) <= budget);
        }
        assert_eq!(
            plan.whole_resident_layers_all_ranks()
                .iter()
                .filter(|l| forced.contains(l))
                .count(),
            3
        );
    }

    #[test]
    fn forced_layers_reject_bad_input_and_overspend() {
        let g = geometry(48, 512, 2);
        let order = permutation(48 * 512, 32);
        let budget = 6000 * g.unit_bytes();
        assert!(forced_plan(&g, &order, budget, &[48])
            .unwrap_err()
            .contains("outside"));
        assert!(forced_plan(&g, &order, budget, &[3, 3])
            .unwrap_err()
            .contains("twice"));
        let tiny = 100 * g.unit_bytes();
        assert!(forced_plan(&g, &order, tiny, &[3])
            .unwrap_err()
            .contains("needs"));
    }

    #[test]
    fn donor_layers_prefer_the_hottest_and_are_deterministic() {
        let g = geometry(48, 512, 2);
        // A ranking that puts layer 30 first, then 7, then the rest by layer.
        let mut order: Vec<u32> = Vec::new();
        for layer in [30usize, 7] {
            order.extend((0..512).map(|e| (layer * 512 + e) as u32));
        }
        for layer in (0..48).filter(|l| *l != 30 && *l != 7) {
            order.extend((0..512).map(|e| (layer * 512 + e) as u32));
        }
        let plan = ranking_plan(&g, &order, 400 * g.unit_bytes());
        assert_eq!(ep_unit_donor_layers(&plan, 3), vec![30, 7, 0]);
        assert_eq!(ep_unit_donor_layers(&plan, 2), vec![30, 7]);
        assert_eq!(ep_unit_donor_layers(&plan, 99).len(), 48);
    }

    #[test]
    fn forced_stage_capacity_is_at_least_one_whole_layer_region() {
        let g = geometry(48, 512, 2);
        let order = permutation(48 * 512, 33);
        let budget = 7000 * g.unit_bytes();
        let ranking = EpRanking {
            rank: &order,
            counts: None,
            sha256: None,
        };
        let whole = 256 * g.unit_bytes();
        let plain = place_routed_experts_units_staged(&g, ranking, &vec![budget; 2], true).unwrap();
        let forced = place_routed_experts_units_forced_staged(
            &g,
            ranking,
            &vec![budget; 2],
            &[9, 10, 11],
            true,
        )
        .unwrap();
        assert!(plain.stage.capacity <= whole);
        assert_eq!(forced.stage.capacity, whole);
        for rank in 0..2 {
            for layer in [9usize, 10, 11] {
                assert_eq!(forced.plan.layer_state(rank, layer), EpLayerState::Resident);
            }
        }
        // Stage off: nothing is charged, the forced plan is the plain forced plan.
        let off =
            place_routed_experts_units_forced_staged(&g, ranking, &vec![budget; 2], &[9], false)
                .unwrap();
        assert_eq!(off.stage, EpUnitStageCharge::default());
        assert_eq!(off.plan, forced_plan(&g, &order, budget, &[9]).unwrap());
    }
}

/// D1.9: the pin plane, the plan-derived balance decision and the device-free equivalence of
/// the pinned owner map with the plan.
#[cfg(test)]
mod ep_unit_pin_tests {
    use super::*;
    use rdna_compute::moe::{
        qwen4_ep_owner_map_pinned, qwen4_ep_owner_pins_check, QWEN4_EP_OWNER_EXPERTS,
    };

    const LAYERS: usize = 3;
    const UNIT: u64 = 512;
    const PAD: u64 = 2 * VRAM_ARENA_TAIL_PAD_BYTES;

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn geometry(ranks: usize, experts: usize) -> EpUnitGeometry {
        EpUnitGeometry {
            layers: LAYERS,
            experts_per_layer: experts,
            ranks,
            gate_up_bytes: UNIT / 2,
            down_bytes: UNIT / 2,
        }
    }

    /// A plan over a random ranking where rank `r` keeps about `kept[r]` units.
    fn ranked_plan(ranks: usize, experts: usize, kept: &[usize], seed: u64) -> EpUnitPlan {
        let total = LAYERS * experts;
        let mut state = seed;
        let mut ranking: Vec<u32> = (0..total as u32).collect();
        for i in (1..total).rev() {
            ranking.swap(i, lcg(&mut state) as usize % (i + 1));
        }
        let budgets: Vec<u64> = (0..ranks)
            .map(|rank| match kept[rank] {
                0 => 0,
                kept => kept as u64 * UNIT + LAYERS as u64 * PAD,
            })
            .collect();
        place_routed_experts_units(
            &geometry(ranks, experts),
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &ranking,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: budgets,
            },
        )
        .unwrap()
    }

    /// 3 layers x 8 experts over 2 ranks; `hot` is ranked first, `budgets` the per-rank bytes.
    fn small_plan(hot: &[u32], budgets: [u64; 2]) -> EpUnitPlan {
        let mut ranking = hot.to_vec();
        ranking.extend((0..24u32).filter(|id| !hot.contains(id)));
        place_routed_experts_units(
            &geometry(2, 8),
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &ranking,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: budgets.to_vec(),
            },
        )
        .unwrap()
    }

    #[test]
    fn the_pinned_owner_switch_is_on_unless_exactly_zero() {
        assert_eq!(EP_UNIT_PINNED_OWNERS_ENV, "HIPFIRE_EP_UNIT_PINNED_OWNERS");
        assert!(ep_unit_pinned_owners_on(None));
        for on in ["1", "", " 0", "0 ", "off", "true", "2"] {
            assert!(ep_unit_pinned_owners_on(Some(on)), "{on:?}");
        }
        assert!(!ep_unit_pinned_owners_on(Some("0")));
    }

    #[test]
    fn the_pin_plane_of_a_whole_layer_plan_is_all_holders_or_all_free() {
        for ranks in [2usize, 3, 4] {
            for vram_layers in [0usize, 1, 2, 3] {
                let plan = place_routed_experts_units(
                    &geometry(ranks, 512),
                    &EpUnitSource::WholeLayers { vram_layers },
                )
                .unwrap();
                for layer in 0..LAYERS {
                    let plane = ep_unit_pin_plane(&plan, layer);
                    assert_eq!(plane.len(), QWEN4_EP_OWNER_EXPERTS);
                    qwen4_ep_owner_pins_check(&plane, ranks).unwrap();
                    for (expert, &pin) in plane.iter().enumerate() {
                        if layer < vram_layers {
                            assert_eq!(usize::from(pin), expert % ranks);
                        } else {
                            assert_eq!(pin, QWEN4_EP_OWNER_PIN_FREE);
                        }
                    }
                    // A whole-layer plan has the same state on every rank.
                    assert_eq!(
                        ep_unit_layer_balance(&plan, layer),
                        if layer < vram_layers {
                            EpUnitLayerBalance::StaticResident
                        } else {
                            EpUnitLayerBalance::BalancedHost
                        },
                        "ranks {ranks} vram_layers {vram_layers} layer {layer}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_pin_plane_follows_the_plan_on_mixed_layers() {
        for ranks in [2usize, 3, 4] {
            let plan = ranked_plan(ranks, 512, &vec![200; ranks], 0x5EED_0000 + ranks as u64);
            for layer in 0..LAYERS {
                let plane = ep_unit_pin_plane(&plan, layer);
                assert_eq!(plane.len(), 512);
                qwen4_ep_owner_pins_check(&plane, ranks).unwrap();
                for (expert, &pin) in plane.iter().enumerate() {
                    let holder = expert % ranks;
                    if plan.is_resident(holder, layer, expert) {
                        assert_eq!(usize::from(pin), holder);
                    } else {
                        assert_eq!(pin, QWEN4_EP_OWNER_PIN_FREE);
                    }
                    // Only the holder ever keeps an expert resident.
                    for rank in (0..ranks).filter(|&rank| rank != holder) {
                        assert!(!plan.is_resident(rank, layer, expert));
                    }
                }
            }
        }
    }

    #[test]
    fn the_balance_of_a_layer_is_decided_from_the_plan_for_every_rank_state_combination() {
        use EpUnitLayerBalance::{BalancedHost, BalancedPinned, StaticResident};
        let pad = PAD;
        // Layer 1 holds ids 8..16: rank 0 holds the even experts, rank 1 the odd ones.
        let balance = |plan: &EpUnitPlan| {
            (0..LAYERS)
                .map(|layer| ep_unit_layer_balance(plan, layer))
                .collect::<Vec<_>>()
        };
        let states = |plan: &EpUnitPlan| {
            (0..2)
                .map(|rank| plan.layer_state(rank, 1))
                .collect::<Vec<_>>()
        };
        // Host + host: today's balanced host layer (every layer of an empty budget).
        let plan = small_plan(&[9, 11], [0, 0]);
        assert_eq!(states(&plan), vec![EpLayerState::Host; 2]);
        assert_eq!(balance(&plan), vec![BalancedHost; 3]);
        // Host + mixed.
        let plan = small_plan(&[9, 11], [0, pad + 2 * UNIT]);
        assert_eq!(plan.layer_state(0, 1), EpLayerState::Host);
        assert!(matches!(plan.layer_state(1, 1), EpLayerState::Mixed { .. }));
        assert_eq!(
            balance(&plan),
            vec![BalancedHost, BalancedPinned, BalancedHost]
        );
        // Host + resident.
        let plan = small_plan(&[9, 11, 13, 15], [0, pad + 4 * UNIT]);
        assert_eq!(
            states(&plan),
            vec![EpLayerState::Host, EpLayerState::Resident]
        );
        assert_eq!(
            balance(&plan),
            vec![BalancedHost, BalancedPinned, BalancedHost]
        );
        // Mixed + mixed.
        let plan = small_plan(&[8, 9, 10, 11], [pad + 2 * UNIT, pad + 2 * UNIT]);
        assert!(matches!(plan.layer_state(0, 1), EpLayerState::Mixed { .. }));
        assert!(matches!(plan.layer_state(1, 1), EpLayerState::Mixed { .. }));
        assert_eq!(balance(&plan)[1], BalancedPinned);
        // Mixed + resident.
        let plan = small_plan(&[9, 11, 13, 15, 8], [pad + 2 * UNIT, pad + 4 * UNIT]);
        assert!(matches!(plan.layer_state(0, 1), EpLayerState::Mixed { .. }));
        assert_eq!(plan.layer_state(1, 1), EpLayerState::Resident);
        assert_eq!(balance(&plan)[1], BalancedPinned);
        // Resident + resident: the static route, no host expert anywhere in the layer.
        let plan = small_plan(
            &[8, 9, 10, 11, 12, 13, 14, 15],
            [pad + 4 * UNIT, pad + 4 * UNIT],
        );
        assert_eq!(
            states(&plan),
            vec![EpLayerState::Resident, EpLayerState::Resident]
        );
        assert_eq!(balance(&plan)[1], StaticResident);
        // The pin plane of every balanced-pinned layer has a pin and a free expert; the other
        // two kinds are all-free (host) or all-pinned (resident).
        let plane = ep_unit_pin_plane(&plan, 1);
        assert!((0..8).all(|expert| usize::from(plane[expert]) == expert % 2));
        let plan = small_plan(&[9, 11], [0, pad + 2 * UNIT]);
        let plane = ep_unit_pin_plane(&plan, 1);
        assert_eq!(
            plane,
            vec![
                QWEN4_EP_OWNER_PIN_FREE,
                1,
                QWEN4_EP_OWNER_PIN_FREE,
                1,
                QWEN4_EP_OWNER_PIN_FREE,
                QWEN4_EP_OWNER_PIN_FREE,
                QWEN4_EP_OWNER_PIN_FREE,
                QWEN4_EP_OWNER_PIN_FREE
            ]
        );
    }

    #[test]
    fn a_balanced_layer_has_a_host_expert_on_some_rank_and_a_static_layer_has_none() {
        for (ranks, kept, seed) in [
            (2usize, [0usize, 0, 0, 0], 11u64),
            (2, [60, 90, 0, 0], 12),
            (3, [250, 120, 300, 0], 13),
            (4, [400, 400, 400, 400], 14),
            (4, [511, 511, 511, 511], 15),
        ] {
            let plan = ranked_plan(ranks, 512, &kept[..ranks], seed);
            for layer in 0..LAYERS {
                let any_host =
                    (0..ranks).any(|rank| plan.layer_state(rank, layer) != EpLayerState::Resident);
                let all_host =
                    (0..ranks).all(|rank| plan.layer_state(rank, layer) == EpLayerState::Host);
                let want = if !any_host {
                    EpUnitLayerBalance::StaticResident
                } else if all_host {
                    EpUnitLayerBalance::BalancedHost
                } else {
                    EpUnitLayerBalance::BalancedPinned
                };
                assert_eq!(ep_unit_layer_balance(&plan, layer), want);
                let plane = ep_unit_pin_plane(&plan, layer);
                let pinned = plane
                    .iter()
                    .filter(|&&pin| pin != QWEN4_EP_OWNER_PIN_FREE)
                    .count();
                match want {
                    EpUnitLayerBalance::StaticResident => assert_eq!(pinned, 512),
                    EpUnitLayerBalance::BalancedHost => assert_eq!(pinned, 0),
                    EpUnitLayerBalance::BalancedPinned => {
                        assert!(pinned < 512, "a host expert exists, so some expert is free")
                    }
                }
            }
        }
    }

    /// Device-free equivalence: the pinned owner map (the byte twin of the kernel) over a
    /// plan's pin plane gives every routed resident expert to its holder, deals the routed
    /// free experts across the ranks within one expert of each other, and so never asks a
    /// rank to read a peer's VRAM expert.
    #[test]
    fn the_pinned_owner_map_over_a_plan_keeps_residents_home_and_balances_the_free_experts() {
        let mut state = 0x9150_0000_0000_0D19u64;
        for ranks in [2usize, 3, 4] {
            for (variant, kept) in [
                [30usize, 60, 90, 120],
                [200, 150, 100, 50],
                [0, 300, 0, 300],
            ]
            .into_iter()
            .enumerate()
            {
                let plan = ranked_plan(
                    ranks,
                    512,
                    &kept[..ranks],
                    0xD19 + variant as u64 * 7 + ranks as u64,
                );
                for layer in 0..LAYERS {
                    let pins = ep_unit_pin_plane(&plan, layer);
                    qwen4_ep_owner_pins_check(&pins, ranks).unwrap();
                    for slots in [1usize, 10, 40, 80, 200] {
                        let ids: Vec<i32> = (0..slots)
                            .map(|_| match lcg(&mut state) % 9 {
                                0 => -1,
                                _ => (lcg(&mut state) % 512) as i32,
                            })
                            .collect();
                        let (owners, slot_owners) =
                            qwen4_ep_owner_map_pinned(&ids, 512, ranks, &pins);
                        let mut free_per_rank = vec![0usize; ranks];
                        for (expert, owner) in owners.iter().enumerate() {
                            let Some(owner) = *owner else { continue };
                            let holder = expert % ranks;
                            if plan.is_resident(holder, layer, expert) {
                                assert_eq!(owner, holder, "resident expert {expert} left home");
                            } else {
                                assert_eq!(pins[expert], QWEN4_EP_OWNER_PIN_FREE);
                                free_per_rank[owner] += 1;
                            }
                        }
                        let (lo, hi) = (
                            *free_per_rank.iter().min().unwrap(),
                            *free_per_rank.iter().max().unwrap(),
                        );
                        assert!(
                            hi - lo <= 1,
                            "ranks {ranks} layer {layer} slots {slots}: free experts {free_per_rank:?}"
                        );
                        for (slot, &id) in ids.iter().enumerate() {
                            let want = usize::try_from(id)
                                .ok()
                                .and_then(|expert| owners[expert])
                                .map_or(-1, |owner| owner as i32);
                            assert_eq!(slot_owners[slot], want);
                        }
                    }
                }
            }
        }
    }

    /// A wholly-host layer's plane is all free: the pinned map equals the unpinned one, which
    /// is why a `BalancedHost` layer keeps today's unpinned kernel.
    #[test]
    fn an_all_host_layer_plane_reproduces_the_unpinned_owner_map() {
        use rdna_compute::moe::qwen4_ep_owner_map;
        let plan = ranked_plan(3, 512, &[0, 0, 0], 77);
        let pins = ep_unit_pin_plane(&plan, 0);
        assert!(pins.iter().all(|&pin| pin == QWEN4_EP_OWNER_PIN_FREE));
        let mut state = 0xFEED_0D19u64;
        for slots in [1usize, 10, 80, 200] {
            let ids: Vec<i32> = (0..slots).map(|_| (lcg(&mut state) % 512) as i32).collect();
            assert_eq!(
                qwen4_ep_owner_map_pinned(&ids, 512, 3, &pins),
                qwen4_ep_owner_map(&ids, 512, 3)
            );
        }
    }
}
