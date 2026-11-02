// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SP (short-prompt prefill) per-layer plan — **pure host planning, no device, no wiring**.
//!
//! After S1's one count barrier a layer has a [`LayerRoutePlan`] (pad16 grouped row starts) and the
//! device's `sorted` map (grouped row → flat slot `token * 10 + rank`, `-1` for a pad row).
//! [`plan_sp_layer`] turns them into everything the SP hook's two arms and its single combine need
//! (HOOK-DESIGN §3.4–3.5):
//!
//! 1. [`exact_counts`]: the *exact* routed-token count of every expert (pad rows excluded), checked
//!    against the device's padded counts;
//! 2. the stream/CPU split ([`crate::sp_plan::plan_layer_split`] over those counts);
//! 3. `groups`: the streamed experts chunked ascending into transport groups (one ring page set);
//! 4. `gpu_panels`: for every group, runs of grouped rows that hold only that group's experts
//!    (a run never crosses a CPU expert's rows), chunked to the K2 panel row bound;
//! 5. `cpu_rows`: the real grouped rows of the CPU experts, in grouped order;
//! 6. `inverse_hybrid`: for every slot, the row of the *hybrid table* holding its expert output —
//!    GPU panel rows first (compact, prefix-summed by `gpu_base`), then the CPU rows.
//!
//! The one canonical combine then reads the hybrid table through `inverse_hybrid`, so its result
//! is a function of the route ids alone, never of which arm produced a row.

use crate::families::moe::MOE_GROUPED_BLOCK_M;
use crate::pipeline::slowlink_prefill::{LayerRoutePlan, PanelRange};
use crate::sp_plan::{plan_layer_split, SplitCost, SplitPlan, NUM_EXPERTS, ROUTE_TOPK};
use crate::types::DispatchError;

fn bail(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("sp layer plan: {message}"))
}

/// Structural bounds of an SP layer plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpLimits {
    /// Experts one transport group holds (S1: 32).
    pub experts_per_slot: usize,
    /// Most groups one layer may stream (the ring's page count bounds it; S1: 16).
    pub max_groups: usize,
    /// Grouped rows one panel may span (a multiple of the 16-row tile; S1: 2048).
    pub max_grouped_rows: usize,
    /// Most CPU slots (rows of the CPU staging buffer) one layer may have
    /// (`5 * SP_MAX_ROWS` = 26,880).
    pub cpu_slot_cap: usize,
}

impl SpLimits {
    /// The product bounds: 32-expert groups, ≤ 16 groups, 2048-row panels, 26,880 CPU slots.
    pub const S1_XTX: SpLimits = SpLimits {
        experts_per_slot: 32,
        max_groups: 16,
        max_grouped_rows: 2048,
        cpu_slot_cap: 5 * crate::sp_plan::SP_MAX_ROWS,
    };

    pub fn validate(&self) -> Result<(), DispatchError> {
        if self.experts_per_slot == 0
            || self.max_groups == 0
            || self.max_grouped_rows == 0
            || self.max_grouped_rows % MOE_GROUPED_BLOCK_M != 0
        {
            return Err(bail(format!(
                "limits need experts_per_slot > 0, max_groups > 0 and a tile-aligned max_grouped_rows, got {self:?}"
            )));
        }
        Ok(())
    }
}

/// The exact routed-token count of every expert, from the device's `sorted` map.
///
/// `exact[e] = #{ r in [offsets[e], offsets[e+1]) : sorted[r] >= 0 }`. Each expert's padded count
/// must equal `round_up16(exact[e])`; a mismatch is a readback error (the window fails).
pub fn exact_counts(sorted: &[i32], plan: &LayerRoutePlan) -> Result<Vec<u32>, DispatchError> {
    let n_exp = plan.n_exp();
    if plan.offsets.len() != n_exp + 1 || plan.offsets[n_exp] != plan.total_rows {
        return Err(bail("the route plan's offsets do not span its padded rows"));
    }
    if sorted.len() < plan.total_rows {
        return Err(bail(format!(
            "sorted has {} entries for {} grouped rows",
            sorted.len(),
            plan.total_rows
        )));
    }
    let mut exact = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let (lo, hi) = (plan.offsets[e], plan.offsets[e + 1]);
        if lo > hi {
            return Err(bail(format!("expert {e}'s offsets run backwards")));
        }
        let count = sorted[lo..hi].iter().filter(|&&s| s >= 0).count();
        let padded = count.div_ceil(MOE_GROUPED_BLOCK_M) * MOE_GROUPED_BLOCK_M;
        if padded != plan.counts_padded[e] as usize || padded != hi - lo {
            return Err(bail(format!(
                "readback mismatch at expert {e}: {count} valid slots round to {padded} rows, \
                 the device counted {} padded ({} offset rows)",
                plan.counts_padded[e],
                hi - lo
            )));
        }
        exact.push(count as u32);
    }
    Ok(exact)
}

/// One layer's SP plan (all indices are host-side; nothing here touches a device).
#[derive(Clone, Debug, PartialEq)]
pub struct SpLayerPlan {
    /// The stream/CPU placement the plan realizes.
    pub split: SplitPlan,
    /// Streamed experts, ascending, `<= experts_per_slot` each; their concatenation is
    /// `split.stream_ids`.
    pub groups: Vec<Vec<u16>>,
    /// K2 panels over streamed rows, in execution order. `binding` = group index,
    /// `bind_lo/hi` = the group's first expert and last expert + 1, `own` = `0..0`, `k1` false.
    /// A panel holds only streamed experts' grouped rows (idle experts have none); it never spans
    /// a CPU expert's rows.
    pub gpu_panels: Vec<PanelRange>,
    /// Hybrid row of each panel's first row (prefix sum of panel rows).
    pub gpu_base: Vec<usize>,
    /// Grouped rows (`sorted[r] >= 0`, CPU expert), grouped order. List index `j` is hybrid row
    /// `G_gpu + j`.
    pub cpu_rows: Vec<u32>,
    /// `[rows * 10]`: hybrid row of every slot, `-1` for an unrouted slot.
    pub inverse_hybrid: Vec<i32>,
    /// `H = G_gpu + S_cpu`.
    pub hybrid_rows: usize,
}

impl SpLayerPlan {
    /// Hybrid rows written by the GPU panels (`G_gpu`).
    pub fn gpu_rows(&self) -> usize {
        self.hybrid_rows - self.cpu_rows.len()
    }
}

/// Plan one SP layer.
///
/// * `plan` / `sorted`: the layer's grouping and the device's grouped-row → slot map (may be
///   longer than `plan.total_rows`: the bound);
/// * `rows`: tokens in the window (`rows * 10` slots);
/// * `cost` / `limits`: the placement model and the structural bounds.
///
/// Refuses (typed error, nothing partially built): `n_exp != 512`, bad limits, a `sorted` /
/// counts readback mismatch, an out-of-range or duplicated slot, more groups than
/// `limits.max_groups`, more CPU slots than `limits.cpu_slot_cap`.
pub fn plan_sp_layer(
    plan: &LayerRoutePlan,
    sorted: &[i32],
    rows: usize,
    cost: &SplitCost,
    limits: SpLimits,
) -> Result<SpLayerPlan, DispatchError> {
    let exact = validated_counts(plan, sorted, rows, limits)?;
    let split = plan_layer_split(&exact, cost);
    assemble(plan, sorted, rows, split, limits)
}

/// Everything [`plan_sp_layer`] checks before it plans: limits, geometry, the readback, and that
/// `sorted` is a partial injection into `[0, rows * 10)`.
fn validated_counts(
    plan: &LayerRoutePlan,
    sorted: &[i32],
    rows: usize,
    limits: SpLimits,
) -> Result<Vec<u32>, DispatchError> {
    limits.validate()?;
    if plan.n_exp() != NUM_EXPERTS {
        return Err(bail(format!(
            "the planner is calibrated for {NUM_EXPERTS} experts, the layer has {}",
            plan.n_exp()
        )));
    }
    let exact = exact_counts(sorted, plan)?;
    let slots = rows
        .checked_mul(ROUTE_TOPK)
        .ok_or_else(|| bail("slot count overflows"))?;
    let mut seen = vec![false; slots];
    for (r, &s) in sorted[..plan.total_rows].iter().enumerate() {
        if s < 0 {
            continue;
        }
        let slot = s as usize;
        if slot >= slots {
            return Err(bail(format!(
                "grouped row {r} names slot {slot}, outside the window's {slots} slots"
            )));
        }
        if std::mem::replace(&mut seen[slot], true) {
            return Err(bail(format!("slot {slot} is grouped twice")));
        }
    }
    Ok(exact)
}

fn expert_of_row(offsets: &[usize], row: usize) -> usize {
    (offsets.partition_point(|&s| s <= row) - 1).min(offsets.len() - 2)
}

/// Realize `split` over the grouping. `sorted` and `plan` must already be validated.
fn assemble(
    plan: &LayerRoutePlan,
    sorted: &[i32],
    rows: usize,
    split: SplitPlan,
    limits: SpLimits,
) -> Result<SpLayerPlan, DispatchError> {
    let offsets = &plan.offsets;
    let n_exp = plan.n_exp();
    let slots = rows * ROUTE_TOPK;

    // Transport groups: streamed experts ascending, chunked.
    let groups: Vec<Vec<u16>> = split
        .stream_ids
        .chunks(limits.experts_per_slot)
        .map(<[u16]>::to_vec)
        .collect();
    if groups.len() > limits.max_groups {
        return Err(bail(format!(
            "{} groups exceed the {} the ring can hold",
            groups.len(),
            limits.max_groups
        )));
    }

    // CPU slots (checked before anything is built from them).
    let cpu_slots = split.cpu_slots as usize;
    if cpu_slots > limits.cpu_slot_cap {
        return Err(bail(format!(
            "{cpu_slots} CPU slots exceed the staging cap of {}",
            limits.cpu_slot_cap
        )));
    }

    // GPU panels: per group, maximal runs of grouped rows holding only that group's experts.
    let mut gpu_panels: Vec<PanelRange> = Vec::new();
    let mut gpu_base: Vec<usize> = Vec::new();
    let mut gpu_total = 0usize;
    for (g, group) in groups.iter().enumerate() {
        let (bind_lo, bind_hi) = (group[0] as usize, group[group.len() - 1] as usize + 1);
        let mut i = 0usize;
        while i < group.len() {
            // A run continues while the next expert's rows start exactly where the previous
            // one's end: only idle (zero-row) experts lie between, so no CPU expert's rows
            // can sit inside a run.
            let first = group[i] as usize;
            let mut last = first;
            while i + 1 < group.len() && offsets[last + 1] == offsets[group[i + 1] as usize] {
                i += 1;
                last = group[i] as usize;
            }
            i += 1;
            let (run_lo, run_hi) = (offsets[first], offsets[last + 1]);
            let mut base = run_lo;
            while base < run_hi {
                let n = (run_hi - base).min(limits.max_grouped_rows);
                gpu_panels.push(PanelRange {
                    index: gpu_panels.len(),
                    binding: g,
                    bind_lo,
                    bind_hi,
                    row_base: base,
                    rows: n,
                    expert_lo: expert_of_row(offsets, base),
                    expert_hi: expert_of_row(offsets, base + n - 1) + 1,
                    own_lo: 0,
                    own_hi: 0,
                    k1: false,
                });
                gpu_base.push(gpu_total);
                gpu_total += n;
                base += n;
            }
        }
    }

    // CPU rows: the real rows of the CPU experts, grouped order.
    let mut cpu_rows: Vec<u32> = Vec::with_capacity(cpu_slots);
    for e in 0..n_exp {
        if split.cpu_mask[e] == 0 {
            continue;
        }
        for r in offsets[e]..offsets[e + 1] {
            if sorted[r] >= 0 {
                cpu_rows.push(r as u32);
            }
        }
    }
    if cpu_rows.len() != cpu_slots {
        return Err(bail(format!(
            "the split has {cpu_slots} CPU slots, the grouping {}",
            cpu_rows.len()
        )));
    }

    // Inverse over the hybrid table.
    let mut inverse_hybrid = vec![-1i32; slots];
    let mut assigned = 0usize;
    for (panel, &base) in gpu_panels.iter().zip(&gpu_base) {
        for r in panel.row_base..panel.row_base + panel.rows {
            if sorted[r] >= 0 {
                inverse_hybrid[sorted[r] as usize] = (base + r - panel.row_base) as i32;
                assigned += 1;
            }
        }
    }
    for (j, &r) in cpu_rows.iter().enumerate() {
        inverse_hybrid[sorted[r as usize] as usize] = (gpu_total + j) as i32;
        assigned += 1;
    }
    let valid = sorted[..plan.total_rows]
        .iter()
        .filter(|&&s| s >= 0)
        .count();
    if assigned != valid {
        return Err(bail(format!(
            "the plan places {assigned} of {valid} routed slots (an active expert is neither \
             streamed nor on the CPU)"
        )));
    }

    let hybrid_rows = gpu_total + cpu_rows.len();
    Ok(SpLayerPlan {
        split,
        groups,
        gpu_panels,
        gpu_base,
        cpu_rows,
        inverse_hybrid,
        hybrid_rows,
    })
}

// ───────────────────────────── TouchedOnly: selective S1 sweep ─────────────────────────────

/// The experts a layer's panel sweep reads: every expert holding at least one K2 grouped row
/// (`counts_padded > 0`, which is `exact > 0` because a padded count is the round-up of the
/// exact one) or at least one cohort slot (`k1_experts`: the K1 short-tail / one-row decode /
/// aggregate F32-tail tiles, whose ids never enter the K2 grouping). Ascending, unique.
///
/// Both sets come from the one count barrier ([`LayerRoutePlan::from_counts`]); no `sorted`
/// readback and no per-token id reaches the host.
pub fn touched_experts(plan: &LayerRoutePlan) -> Vec<u16> {
    let mut touched: Vec<u16> = plan
        .experts_with_rows()
        .chain(plan.k1_experts.iter().copied())
        .map(|e| e as u16)
        .collect();
    touched.sort_unstable();
    touched.dedup();
    touched
}

/// The transport groups of a TouchedOnly layer: [`touched_experts`] chunked ascending into
/// groups of at most `experts_per_slot` (S1: 32), one ring page each.
///
/// `pages` is the ring's page count: the whole layer's groups are enqueued at once (the host
/// never waits for a page), so more groups than pages is refused here, before any ring state is
/// touched. A layer that routes no slot at all (every id unrouted) has no touched expert; it
/// streams the single `fallback` expert (a held expert of the layer) so the panel schedule keeps
/// its one zero-row panel (`None`: no fallback, an error).
pub fn touched_groups(
    plan: &LayerRoutePlan,
    experts_per_slot: usize,
    pages: usize,
    fallback: Option<u16>,
) -> Result<Vec<Vec<u16>>, DispatchError> {
    if experts_per_slot == 0 {
        return Err(bail("touched groups need experts_per_slot > 0"));
    }
    if plan.n_exp() > usize::from(u16::MAX) + 1 {
        return Err(bail(format!(
            "{} experts do not fit a u16 expert id",
            plan.n_exp()
        )));
    }
    let mut touched = touched_experts(plan);
    if touched.is_empty() {
        touched
            .push(fallback.ok_or_else(|| {
                bail("the layer routes no slot and no fallback expert was named")
            })?);
    }
    let groups: Vec<Vec<u16>> = touched
        .chunks(experts_per_slot)
        .map(<[u16]>::to_vec)
        .collect();
    if groups.len() > pages {
        return Err(bail(format!(
            "{} touched groups need {} ring pages at once, the ring has {pages}",
            groups.len(),
            groups.len()
        )));
    }
    Ok(groups)
}

/// The panel binding ranges of touched groups: `(first expert, last expert + 1)` of each
/// group, ascending and disjoint. They are tight (they do not extend over the untouched
/// experts between two groups), which is what the weight binding of a panel holds: a K1 or
/// F32-tail cohort panel requires the binding to cover `[bind_lo, bind_hi)`. An expert between
/// two groups has no K2 row and no cohort slot (it is untouched), so no row is dropped.
pub fn touched_binding_ranges(groups: &[Vec<u16>]) -> Vec<(usize, usize)> {
    groups
        .iter()
        .filter_map(|group| {
            let (&lo, &last) = (group.first()?, group.last()?);
            Some((usize::from(lo), usize::from(last) + 1))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::slowlink_prefill::BarrierCost;
    use crate::sp_plan::{plan_threshold, SP_MAX_ROWS};

    const TOPK: usize = ROUTE_TOPK;

    // ---------------- test-only reference: stable grouping from per-token ids ----------------

    struct Grouped {
        plan: LayerRoutePlan,
        /// grouped row -> slot (-1 pad); length `total_rows + extra` (the layout's bound).
        sorted: Vec<i32>,
        /// exact counts (ground truth, independent of `exact_counts`).
        truth: Vec<u32>,
    }

    /// CPU twin of the device's stable `(expert, slot)` pad16 scatter: slots of one expert keep
    /// ascending slot order, pad rows (`-1`) at the tail of each expert block. Ids outside
    /// `[0, n_exp)` are unrouted. `extra` appends `-1` bound rows like the device's buffer.
    fn group_reference(ids: &[i32], n_exp: usize, extra: usize) -> Grouped {
        let mut truth = vec![0u32; n_exp];
        for &id in ids {
            if (0..n_exp as i32).contains(&id) {
                truth[id as usize] += 1;
            }
        }
        let padded: Vec<i32> = truth
            .iter()
            .map(|&c| (c.div_ceil(16) * 16) as i32)
            .collect();
        let mut start = vec![0usize; n_exp + 1];
        for e in 0..n_exp {
            start[e + 1] = start[e] + padded[e] as usize;
        }
        let mut fill = vec![0usize; n_exp];
        let mut sorted = vec![-1i32; start[n_exp] + extra];
        for (slot, &id) in ids.iter().enumerate() {
            if (0..n_exp as i32).contains(&id) {
                let e = id as usize;
                sorted[start[e] + fill[e]] = slot as i32;
                fill[e] += 1;
            }
        }
        let plan =
            LayerRoutePlan::from_counts(&padded, Vec::new(), BarrierCost::default()).unwrap();
        Grouped {
            plan,
            sorted,
            truth,
        }
    }

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

    /// `rows` tokens, 10 distinct experts each, skewed toward a hot subset; `junk` sprinkles
    /// unrouted ids (-1 / >= n_exp).
    fn random_ids(rng: &mut Rng, rows: usize, n_exp: usize, hot: usize, junk: bool) -> Vec<i32> {
        let mut ids = Vec::with_capacity(rows * TOPK);
        for _ in 0..rows {
            let mut row: Vec<i32> = Vec::new();
            while row.len() < TOPK {
                let e = if rng.below(3) == 0 {
                    rng.below(hot)
                } else {
                    rng.below(n_exp)
                } as i32;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            if junk && rng.below(7) == 0 {
                let i = rng.below(TOPK);
                row[i] = if rng.below(2) == 0 {
                    -1
                } else {
                    n_exp as i32 + rng.below(9) as i32
                };
            }
            ids.extend(row);
        }
        ids
    }

    // ---------------- the property checker ----------------

    fn expert_of(offsets: &[usize], row: usize) -> usize {
        offsets.partition_point(|&s| s <= row) - 1
    }

    /// Everything §3.5 and §8 promise of a plan, against ground truth derived from `ids`.
    fn check(g: &Grouped, ids: &[i32], p: &SpLayerPlan, limits: SpLimits) {
        let off = &g.plan.offsets;
        let n_exp = g.plan.n_exp();
        let rows = ids.len() / TOPK;
        let valid = |id: i32| (0..n_exp as i32).contains(&id);

        // split consistency with ground-truth counts
        assert_eq!(p.split.n_active, g.truth.iter().filter(|&&c| c > 0).count());
        let streamed = |e: usize| p.split.stream_ids.binary_search(&(e as u16)).is_ok();
        let on_cpu = |e: usize| p.split.cpu_mask[e] == 1;
        for e in 0..n_exp {
            if g.truth[e] == 0 {
                assert!(!streamed(e) && !on_cpu(e), "idle expert {e} placed");
            } else {
                assert_ne!(streamed(e), on_cpu(e), "expert {e} on exactly one side");
            }
        }
        assert_eq!(p.split.k + p.split.n_cpu, p.split.n_active);

        // groups: ascending, <= experts_per_slot, <= max_groups, concat == stream_ids
        assert!(p.groups.len() <= limits.max_groups);
        assert!(p
            .groups
            .iter()
            .all(|g| !g.is_empty() && g.len() <= limits.experts_per_slot));
        let flat: Vec<u16> = p.groups.iter().flatten().copied().collect();
        assert_eq!(flat, p.split.stream_ids);
        assert!(flat.windows(2).all(|w| w[0] < w[1]));

        // panels
        assert_eq!(p.gpu_panels.len(), p.gpu_base.len());
        let mut expect_base = 0usize;
        let mut covered = vec![false; g.plan.total_rows];
        for (i, (panel, &base)) in p.gpu_panels.iter().zip(&p.gpu_base).enumerate() {
            assert_eq!(panel.index, i);
            assert_eq!(base, expect_base, "gpu_base is the prefix sum");
            expect_base += panel.rows;
            let group = &p.groups[panel.binding];
            assert_eq!(panel.bind_lo, group[0] as usize);
            assert_eq!(panel.bind_hi, *group.last().unwrap() as usize + 1);
            assert_eq!((panel.own_lo, panel.own_hi, panel.k1), (0, 0, false));
            assert!(
                panel.rows > 0 && panel.rows % 16 == 0 && panel.rows <= limits.max_grouped_rows
            );
            assert_eq!(panel.row_base % 16, 0);
            assert_eq!(panel.expert_lo, expert_of(off, panel.row_base));
            assert_eq!(
                panel.expert_hi,
                expert_of(off, panel.row_base + panel.rows - 1) + 1
            );
            for r in panel.row_base..panel.row_base + panel.rows {
                // disjoint panels
                assert!(
                    !std::mem::replace(&mut covered[r], true),
                    "row {r} in two panels"
                );
                // never a CPU expert's (or an ungrouped expert's) row: it belongs to this group
                let e = expert_of(off, r);
                assert!(
                    group.binary_search(&(e as u16)).is_ok(),
                    "row {r} (expert {e}) outside group {}",
                    panel.binding
                );
                assert!(!on_cpu(e));
            }
        }
        let g_gpu = expect_base;
        // coverage: every streamed expert's full padded block is in exactly one panel
        for e in 0..n_exp {
            for r in off[e]..off[e + 1] {
                assert_eq!(covered[r], streamed(e), "row {r} of expert {e}");
            }
        }

        // cpu rows
        assert!(p.cpu_rows.windows(2).all(|w| w[0] < w[1]), "grouped order");
        let want: Vec<u32> = (0..g.plan.total_rows)
            .filter(|&r| g.sorted[r] >= 0 && on_cpu(expert_of(off, r)))
            .map(|r| r as u32)
            .collect();
        assert_eq!(p.cpu_rows, want);
        assert_eq!(p.cpu_rows.len() as u64, p.split.cpu_slots);
        assert!(p.cpu_rows.len() <= limits.cpu_slot_cap);
        assert_eq!(p.hybrid_rows, g_gpu + p.cpu_rows.len());
        assert_eq!(p.gpu_rows(), g_gpu);
        assert!(
            p.hybrid_rows <= g.plan.total_rows,
            "H within the grouped bound"
        );

        // inverse: defined exactly on routed slots, bijective onto the rows that hold them
        assert_eq!(p.inverse_hybrid.len(), rows * TOPK);
        let mut holder = vec![-1i32; p.hybrid_rows]; // hybrid row -> slot
        let mut n_valid = 0usize;
        for (slot, &h) in p.inverse_hybrid.iter().enumerate() {
            if !valid(ids[slot]) {
                assert_eq!(h, -1, "unrouted slot {slot}");
                continue;
            }
            n_valid += 1;
            assert!((0..p.hybrid_rows as i32).contains(&h), "slot {slot} -> {h}");
            assert_eq!(holder[h as usize], -1, "hybrid row {h} written twice");
            holder[h as usize] = slot as i32;
            // GPU rows [0, G) for streamed slots, CPU rows [G, H) for CPU slots (disjoint writers)
            let e = ids[slot] as usize;
            assert_eq!(streamed(e), (h as usize) < g_gpu, "slot {slot} expert {e}");
            assert_eq!(on_cpu(e), (h as usize) >= g_gpu);
        }
        assert_eq!(n_valid, g.truth.iter().map(|&c| c as usize).sum::<usize>());
        // the inverse of the inverse names the grouped row's own slot (no layout re-implemented)
        for (panel, &base) in p.gpu_panels.iter().zip(&p.gpu_base) {
            for r in panel.row_base..panel.row_base + panel.rows {
                let h = base + r - panel.row_base;
                assert_eq!(holder[h], g.sorted[r], "panel row {r}");
            }
        }
        for (j, &r) in p.cpu_rows.iter().enumerate() {
            assert_eq!(holder[g_gpu + j], g.sorted[r as usize]);
        }

        // extremes
        if p.split.k == p.split.n_active {
            assert!(
                p.cpu_rows.is_empty() && p.hybrid_rows == g_gpu,
                "k = U: no CPU rows"
            );
        }
        if p.split.k == 0 {
            assert!(
                p.groups.is_empty() && p.gpu_panels.is_empty() && g_gpu == 0,
                "k = 0: no panels"
            );
            assert_eq!(p.cpu_rows.len(), n_valid);
        }
    }

    fn costs(rows: usize) -> Vec<SplitCost> {
        let auto = SplitCost::sp_hook_xtx(rows as u32, 2560, 16);
        let mut all_stream = auto.clone();
        all_stream.stream_ms_per_expert = 0.0; // k = U
        let mut all_cpu = auto.clone();
        all_cpu.stream_ms_per_expert = 1e6; // k = 0
        let mut mid = auto.clone();
        mid.stream_ms_per_expert = 0.05; // many streamed experts: several groups
        vec![auto, all_stream, all_cpu, mid]
    }

    fn run(ids: &[i32], extra: usize, cost: &SplitCost, limits: SpLimits) -> SpLayerPlan {
        let g = group_reference(ids, NUM_EXPERTS, extra);
        let plan = plan_sp_layer(&g.plan, &g.sorted, ids.len() / TOPK, cost, limits).unwrap();
        check(&g, ids, &plan, limits);
        // determinism: same input, same plan
        assert_eq!(
            plan,
            plan_sp_layer(&g.plan, &g.sorted, ids.len() / TOPK, cost, limits).unwrap()
        );
        plan
    }

    // ---------------- tests ----------------

    #[test]
    fn sp_layer_exact_counts_match_truth_and_ignore_the_bound_tail() {
        let mut rng = Rng(0x5eed);
        let ids = random_ids(&mut rng, 300, NUM_EXPERTS, 24, true);
        let g = group_reference(&ids, NUM_EXPERTS, 7 * 16);
        assert_eq!(exact_counts(&g.sorted, &g.plan).unwrap(), g.truth);
        // all-idle window
        let g = group_reference(&[-1; 40], NUM_EXPERTS, 0);
        assert!(exact_counts(&g.sorted, &g.plan)
            .unwrap()
            .iter()
            .all(|&c| c == 0));
    }

    #[test]
    fn sp_layer_readback_mismatch_is_refused() {
        let mut rng = Rng(7);
        let ids = random_ids(&mut rng, 200, NUM_EXPERTS, 16, false);
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        // (a) a whole expert block turned into pad rows: exact 0 against a padded count > 0
        let e = (0..NUM_EXPERTS).find(|&e| g.truth[e] > 0).unwrap();
        let mut sorted = g.sorted.clone();
        // make the whole block of `e` invalid -> exact 0 but padded > 0
        for r in g.plan.offsets[e]..g.plan.offsets[e + 1] {
            sorted[r] = -1;
        }
        let err = exact_counts(&sorted, &g.plan).unwrap_err().to_string();
        assert!(err.contains("readback mismatch"), "{err}");
        assert!(plan_sp_layer(
            &g.plan,
            &sorted,
            200,
            &SplitCost::sp_hook_xtx(200, 2560, 16),
            SpLimits::S1_XTX
        )
        .is_err());
        // (b) counts disagree with sorted the other way: a count that rounds differently
        let mut bad = g.plan.clone();
        bad.counts_padded[e] += 16;
        let err = exact_counts(&g.sorted, &bad).unwrap_err().to_string();
        assert!(err.contains("readback mismatch"), "{err}");
        // (c) sorted shorter than the grouped rows
        let short = &g.sorted[..g.plan.total_rows - 1];
        assert!(exact_counts(short, &g.plan).is_err());
    }

    #[test]
    fn sp_layer_refuses_bad_slots() {
        let mut rng = Rng(11);
        let ids = random_ids(&mut rng, 64, NUM_EXPERTS, 16, false);
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let cost = SplitCost::sp_hook_xtx(64, 2560, 16);
        let first = g.sorted.iter().position(|&s| s >= 0).unwrap();
        // slot outside the window
        let mut s = g.sorted.clone();
        s[first] = (64 * TOPK) as i32;
        assert!(plan_sp_layer(&g.plan, &s, 64, &cost, SpLimits::S1_XTX)
            .unwrap_err()
            .to_string()
            .contains("outside"));
        // slot grouped twice
        let second = g
            .sorted
            .iter()
            .skip(first + 1)
            .position(|&s| s >= 0)
            .unwrap()
            + first
            + 1;
        let mut s = g.sorted.clone();
        s[second] = s[first];
        assert!(plan_sp_layer(&g.plan, &s, 64, &cost, SpLimits::S1_XTX)
            .unwrap_err()
            .to_string()
            .contains("twice"));
    }

    #[test]
    fn sp_layer_refuses_wrong_expert_count() {
        let ids: Vec<i32> = (0..30).map(|i| i % 256).collect();
        for n_exp in [256usize, 511, 513] {
            let g = group_reference(&ids, n_exp, 0);
            let err = plan_sp_layer(
                &g.plan,
                &g.sorted,
                3,
                &SplitCost::sp_hook_xtx(3, 2560, 16),
                SpLimits::S1_XTX,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("512 experts"), "{err}");
        }
    }

    #[test]
    fn sp_layer_refuses_bad_limits() {
        let ids: Vec<i32> = (0..10).collect();
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let cost = SplitCost::sp_hook_xtx(1, 2560, 16);
        for bad in [
            SpLimits {
                experts_per_slot: 0,
                ..SpLimits::S1_XTX
            },
            SpLimits {
                max_groups: 0,
                ..SpLimits::S1_XTX
            },
            SpLimits {
                max_grouped_rows: 0,
                ..SpLimits::S1_XTX
            },
            SpLimits {
                max_grouped_rows: 2040,
                ..SpLimits::S1_XTX
            },
        ] {
            assert!(
                plan_sp_layer(&g.plan, &g.sorted, 1, &cost, bad).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn sp_layer_refuses_more_groups_than_max() {
        let mut rng = Rng(3);
        let ids = random_ids(&mut rng, 512, NUM_EXPERTS, 64, false);
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let mut cost = SplitCost::sp_hook_xtx(512, 2560, 16);
        cost.stream_ms_per_expert = 0.0; // stream everything: U ~ 500 experts -> 16 groups
        let ok = plan_sp_layer(&g.plan, &g.sorted, 512, &cost, SpLimits::S1_XTX).unwrap();
        assert!(ok.groups.len() > 4 && ok.groups.len() <= 16);
        let tight = SpLimits {
            max_groups: ok.groups.len() - 1,
            ..SpLimits::S1_XTX
        };
        let err = plan_sp_layer(&g.plan, &g.sorted, 512, &cost, tight)
            .unwrap_err()
            .to_string();
        assert!(err.contains("groups exceed"), "{err}");
        // exactly at the bound is fine
        let at = SpLimits {
            max_groups: ok.groups.len(),
            ..SpLimits::S1_XTX
        };
        assert!(plan_sp_layer(&g.plan, &g.sorted, 512, &cost, at).is_ok());
    }

    #[test]
    fn sp_layer_refuses_cpu_slots_over_cap() {
        let mut rng = Rng(5);
        let ids = random_ids(&mut rng, 512, NUM_EXPERTS, 64, false);
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let mut cost = SplitCost::sp_hook_xtx(512, 2560, 16);
        cost.stream_ms_per_expert = 1e6; // CPU-all
        cost.cpu_slot_cap = None; // the planner itself would not respect any cap
        let all_cpu = plan_sp_layer(&g.plan, &g.sorted, 512, &cost, SpLimits::S1_XTX).unwrap();
        assert_eq!(all_cpu.cpu_rows.len(), 5120);
        let lim = SpLimits {
            cpu_slot_cap: 5119,
            ..SpLimits::S1_XTX
        };
        let err = plan_sp_layer(&g.plan, &g.sorted, 512, &cost, lim)
            .unwrap_err()
            .to_string();
        assert!(err.contains("CPU slots exceed"), "{err}");
        let lim = SpLimits {
            cpu_slot_cap: 5120,
            ..SpLimits::S1_XTX
        };
        assert!(plan_sp_layer(&g.plan, &g.sorted, 512, &cost, lim).is_ok());
        // and with the product cost the planner honors the cap itself, so the same cap is never hit
        let cost = SplitCost::sp_hook_xtx(512, 2560, 16);
        let p = plan_sp_layer(
            &g.plan,
            &g.sorted,
            512,
            &cost,
            SpLimits {
                cpu_slot_cap: 1000,
                ..SpLimits::S1_XTX
            },
        );
        // cost cap (26,880) > limit cap: refused iff the plan keeps > 1000 slots on the CPU
        match p {
            Ok(p) => assert!(p.cpu_rows.len() <= 1000),
            Err(e) => assert!(e.to_string().contains("CPU slots exceed")),
        }
    }

    #[test]
    fn sp_layer_random_windows_satisfy_every_property() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for (rows, hot, junk, extra) in [
            (5usize, 4usize, false, 0usize),
            (1, 4, false, 16),
            (64, 8, true, 0),
            (300, 16, true, 3 * 16),
            (1024, 24, false, 0),
            (1024, 200, true, 512),
            (2048, 64, false, 0),
        ] {
            let ids = random_ids(&mut rng, rows, NUM_EXPERTS, hot, junk);
            for cost in costs(rows) {
                run(&ids, extra, &cost, SpLimits::S1_XTX);
            }
        }
    }

    #[test]
    fn sp_layer_extremes() {
        let mut rng = Rng(99);
        let ids = random_ids(&mut rng, 700, NUM_EXPERTS, 30, false);
        let [auto, all_stream, all_cpu, _] = <[SplitCost; 4]>::try_from(costs(700)).unwrap();
        let u = run(&ids, 0, &all_stream, SpLimits::S1_XTX);
        assert_eq!(u.split.k, u.split.n_active);
        assert!(u.cpu_rows.is_empty() && u.groups.len() == u.split.n_active.div_ceil(32));
        let z = run(&ids, 0, &all_cpu, SpLimits::S1_XTX);
        assert_eq!(z.split.k, 0);
        assert!(z.gpu_panels.is_empty() && z.gpu_base.is_empty() && z.groups.is_empty());
        assert_eq!(z.hybrid_rows, z.cpu_rows.len());
        // a hybrid placement streams some, computes the rest
        let a = run(&ids, 0, &auto, SpLimits::S1_XTX);
        assert!(
            a.split.k > 0 && a.split.k < a.split.n_active,
            "auto split is a mix at this size"
        );
        // an all-idle layer plans to nothing
        let idle = run(&[-1; 50], 0, &auto, SpLimits::S1_XTX);
        assert_eq!((idle.hybrid_rows, idle.groups.len()), (0, 0));
        assert!(idle.inverse_hybrid.iter().all(|&h| h == -1));
    }

    #[test]
    fn sp_layer_panels_never_span_cpu_rows() {
        // Experts 0..40 with decreasing counts, plus a streamed expert 200 whose block comes
        // after CPU experts: a threshold split streams the heavy ones (ids 0, 2, 5, 9, 200) and
        // leaves the experts between them on the CPU.
        let mut ids: Vec<i32> = Vec::new();
        let heavy = [0i32, 2, 5, 9, 200];
        for t in 0..64 {
            let mut row = vec![-1i32; TOPK];
            for (i, &e) in heavy.iter().enumerate() {
                row[i] = e;
            }
            // light experts between the heavy ones: 1, 3, 4, 6, 7, 8 (one token each-ish)
            for (i, e) in [1, 3, 4, 6, 7, 8].iter().enumerate() {
                if (t + i) % 11 == 0 {
                    row[5 + i % 5] = *e;
                }
            }
            ids.extend(row);
        }
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let cost = SplitCost::sp_hook_xtx(64, 2560, 16);
        let split = plan_threshold(&g.truth, 64, &cost);
        assert_eq!(split.stream_ids, vec![0, 2, 5, 9, 200]);
        assert!(split.n_cpu > 0);
        let limits = SpLimits::S1_XTX;
        let p = assemble(&g.plan, &g.sorted, 64, split, limits).unwrap();
        check(&g, &ids, &p, limits);
        // Experts 9 and 200 are separated only by idle experts, so their blocks form ONE run;
        // the other heavy experts are fenced in by CPU experts' blocks.
        assert_eq!(p.gpu_panels.len(), 4);
        assert!(p.gpu_panels[..3]
            .iter()
            .all(|pn| pn.expert_hi == pn.expert_lo + 1));
        assert_eq!(
            (
                p.gpu_panels[3].expert_lo,
                p.gpu_panels[3].expert_hi,
                p.gpu_panels[3].rows
            ),
            (9, 201, 128)
        );
    }

    #[test]
    fn sp_layer_runs_join_across_idle_experts_and_chunk_at_the_row_bound() {
        // Experts 10, 20, 30 stream and are adjacent in grouped order (everything between is
        // idle); expert 10 holds 4,100 rows' worth of slots -> chunked 2048 + 2048 + 16.
        let mut ids: Vec<i32> = Vec::new();
        for t in 0..4100 {
            let mut row = vec![-1i32; TOPK];
            row[0] = 10;
            if t < 100 {
                row[1] = 20;
            }
            if t < 40 {
                row[2] = 30;
            }
            ids.extend(row);
        }
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let mut cost = SplitCost::sp_hook_xtx(4100, 2560, 16);
        cost.stream_ms_per_expert = 0.0;
        let p = run(&ids, 0, &cost, SpLimits::S1_XTX);
        assert_eq!(p.groups, vec![vec![10u16, 20, 30]]);
        let shape: Vec<(usize, usize)> = p
            .gpu_panels
            .iter()
            .map(|pn| (pn.row_base, pn.rows))
            .collect();
        let off = &g.plan.offsets;
        assert_eq!(off[10], 0);
        // 4100 -> pad 4112; expert 20: 112; expert 30: 48; contiguous run of 4272 rows
        assert_eq!(off[31], 4112 + 112 + 48);
        assert_eq!(shape, vec![(0, 2048), (2048, 2048), (4096, 176)]);
        assert_eq!(p.gpu_base, vec![0, 2048, 4096]);
        // a smaller bound re-chunks the same rows
        let small = SpLimits {
            max_grouped_rows: 64,
            ..SpLimits::S1_XTX
        };
        let q = run(&ids, 0, &cost, small);
        assert_eq!(q.gpu_panels.iter().map(|pn| pn.rows).sum::<usize>(), 4272);
        assert!(q.gpu_panels.iter().all(|pn| pn.rows <= 64));
    }

    #[test]
    fn sp_layer_groups_split_every_32_experts() {
        // 70 equally-hot experts, all streamed: 32 + 32 + 6, ascending, panel per group.
        let mut ids: Vec<i32> = Vec::new();
        for t in 0..7 {
            for j in 0..10 {
                ids.push(((t * 10 + j) % 70) as i32);
            }
        }
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let mut cost = SplitCost::sp_hook_xtx(7, 2560, 16);
        cost.stream_ms_per_expert = 0.0;
        let p = run(&ids, 0, &cost, SpLimits::S1_XTX);
        assert_eq!(p.split.n_active, 70);
        assert_eq!(
            p.groups.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![32, 32, 6]
        );
        assert_eq!(p.gpu_panels.len(), 3);
        assert_eq!(
            p.gpu_panels.iter().map(|pn| pn.binding).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(g.plan.total_rows, 70 * 16);
    }

    #[test]
    fn sp_layer_determinism_under_ties() {
        // Every active expert has the same count: placement ties break by expert id, so the
        // streamed set is the lowest ids and every call returns the identical plan.
        let mut ids: Vec<i32> = Vec::new();
        let active: Vec<i32> = (0..NUM_EXPERTS as i32).step_by(3).take(120).collect();
        for t in 0..48 {
            for j in 0..10 {
                ids.push(active[(t * 10 + j) % active.len()]);
            }
        }
        let g = group_reference(&ids, NUM_EXPERTS, 0);
        let c0 = g.truth[active[0] as usize];
        assert!(
            active.iter().all(|&e| g.truth[e as usize] == c0),
            "ties by construction"
        );
        let mut cost = SplitCost::sp_hook_xtx(48, 2560, 16);
        cost.stream_ms_per_expert = 0.1;
        let a = plan_sp_layer(&g.plan, &g.sorted, 48, &cost, SpLimits::S1_XTX).unwrap();
        let b = plan_sp_layer(&g.plan, &g.sorted, 48, &cost, SpLimits::S1_XTX).unwrap();
        assert_eq!(a, b);
        let expect: Vec<u16> = active.iter().take(a.split.k).map(|&e| e as u16).collect();
        assert_eq!(a.split.stream_ids, expect);
        check(&g, &ids, &a, SpLimits::S1_XTX);
    }

    // ---------------- the 24 real route files ----------------

    /// `screens/SP/ab/counts/*.route.bin`: raw little-endian `int16 [N, 10]`, the actual
    /// per-token expert ids of one layer's window (`extract_counts.py`), i.e. real per-token
    /// ids (not just counts), so the reference grouping above builds `sorted` from them.
    const ROUTE_DIR: &str = "/home/kaden/qcal/release-0.4.3/xtx-decode/screens/SP/ab/counts";

    fn load_route(path: &std::path::Path) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % (TOPK * 2), 0, "{path:?}");
        bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
            .collect()
    }

    #[test]
    fn sp_layer_real_route_files_satisfy_every_property() {
        let dir = ROUTE_DIR;
        let Ok(rd) = std::fs::read_dir(&dir) else {
            eprintln!("sp_layer: {dir} not present; real-route property test skipped");
            return;
        };
        let mut files: Vec<_> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".route.bin"))
            .collect();
        files.sort();
        assert_eq!(
            files.len(),
            24,
            "expected the 24 recorded (doc, layer, N) windows in {dir}"
        );
        let mut engaged = 0usize;
        for f in &files {
            let ids = load_route(f);
            let rows = ids.len() / TOPK;
            assert!(rows <= SP_MAX_ROWS);
            assert!(
                ids.iter().all(|&e| (0..NUM_EXPERTS as i32).contains(&e)),
                "{f:?}: real routes are all valid"
            );
            // the device's bound rows past the padded total
            for cost in costs(rows) {
                let p = run(&ids, 15 * 16, &cost, SpLimits::S1_XTX);
                if p.split.k > 0 && p.split.k < p.split.n_active {
                    engaged += 1;
                }
            }
            // the model's own choice at the real window is a mix of both arms
            let auto = run(
                &ids,
                0,
                &SplitCost::sp_hook_xtx(rows as u32, 2560, 16),
                SpLimits::S1_XTX,
            );
            assert!(
                auto.split.k > 0 && auto.split.n_cpu > 0,
                "{f:?}: auto split k={} U={}",
                auto.split.k,
                auto.split.n_active
            );
            assert!(auto.groups.len() <= 16);
            // 8 workers shifts the split, never the invariants
            run(
                &ids,
                0,
                &SplitCost::sp_hook_xtx(rows as u32, 2560, 8),
                SpLimits::S1_XTX,
            );
        }
        assert!(
            engaged >= 24,
            "every window exercised a hybrid split ({engaged})"
        );
    }

    // ---------------- TouchedOnly: touched groups and the S1 panel schedule over them ----------------

    use crate::pipeline::slowlink_prefill::PanelLimits;

    /// A layer grouping of `ids` whose cohort (K1 / F32-tail tiles) routes to `cohort`.
    fn with_cohort(ids: &[i32], n_exp: usize, cohort: &[usize]) -> LayerRoutePlan {
        let g = group_reference(ids, n_exp, 0);
        LayerRoutePlan::from_counts(
            &g.plan
                .counts_padded
                .iter()
                .map(|&c| c as i32)
                .collect::<Vec<_>>(),
            cohort.to_vec(),
            BarrierCost::default(),
        )
        .unwrap()
    }

    #[test]
    fn touched_experts_are_the_k2_and_cohort_experts_ascending() {
        // K2 rows on 3, 7, 7, 200; the cohort routes to 7 (also K2), 5 and 511 (cohort only).
        let mut ids = vec![-1i32; 40];
        ids[0] = 200;
        ids[1] = 3;
        ids[2] = 7;
        ids[15] = 7;
        let plan = with_cohort(&ids, 512, &[5, 7, 511]);
        assert_eq!(touched_experts(&plan), vec![3, 5, 7, 200, 511]);
        // No cohort: only the K2 experts. No rows at all: nothing.
        assert_eq!(
            touched_experts(&with_cohort(&ids, 512, &[])),
            vec![3, 7, 200]
        );
        assert!(touched_experts(&with_cohort(&[-1; 20], 512, &[])).is_empty());
    }

    #[test]
    fn touched_groups_chunk_ascending_cover_everything_and_respect_the_ring() {
        let mut rng = Rng(0x70c4_ed);
        for rows in [5usize, 100, 768, 1536, 2304] {
            for hot in [4usize, 64, 512] {
                let ids = random_ids(&mut rng, rows, 512, hot, true);
                let cohort: Vec<usize> = {
                    let mut c: Vec<usize> = (0..rng.below(40)).map(|_| rng.below(512)).collect();
                    c.sort_unstable();
                    c.dedup();
                    c
                };
                let plan = with_cohort(&ids, 512, &cohort);
                let groups = touched_groups(&plan, 32, 18, Some(0)).unwrap();
                let flat: Vec<usize> = groups.iter().flatten().map(|&e| usize::from(e)).collect();
                // Every expert with K2 rows or cohort slots, ascending, once.
                let mut want: Vec<usize> = plan
                    .counts_padded
                    .iter()
                    .enumerate()
                    .filter(|&(e, &c)| c > 0 || cohort.contains(&e))
                    .map(|(e, _)| e)
                    .collect();
                if want.is_empty() {
                    want.push(0);
                }
                assert_eq!(flat, want, "rows {rows} hot {hot}");
                assert!(groups.iter().all(|g| (1..=32).contains(&g.len())));
                assert!(
                    groups.len() <= 16,
                    "512 experts never need more than 16 groups"
                );
                // Every group but the last is full: the fewest pages.
                assert!(groups[..groups.len() - 1].iter().all(|g| g.len() == 32));
                // The ring bound: exactly at the pages passes, one fewer is refused.
                assert!(touched_groups(&plan, 32, groups.len(), Some(0)).is_ok());
                if groups.len() > 1 {
                    let error = touched_groups(&plan, 32, groups.len() - 1, Some(0)).unwrap_err();
                    assert!(error.to_string().contains("ring pages"), "{error}");
                }
            }
        }
        // No touched expert: the fallback streams alone; without one it is an error.
        let none = with_cohort(&[-1; 20], 512, &[]);
        assert_eq!(
            touched_groups(&none, 32, 18, Some(9)).unwrap(),
            vec![vec![9u16]]
        );
        assert!(touched_groups(&none, 32, 18, None).is_err());
        assert!(touched_groups(&none, 0, 18, Some(9)).is_err());
    }

    #[test]
    fn touched_panels_are_the_s1_schedule_over_tight_ascending_bindings() {
        let mut rng = Rng(0xA11_70c4);
        for rows in [5usize, 300, 768, 1024, 2304] {
            for hot in [4usize, 40, 512] {
                let ids = random_ids(&mut rng, rows, 512, hot, true);
                let cohort: Vec<usize> = {
                    let mut c: Vec<usize> =
                        (0..1 + rng.below(30)).map(|_| rng.below(512)).collect();
                    c.sort_unstable();
                    c.dedup();
                    c
                };
                let plan = with_cohort(&ids, 512, &cohort);
                let groups = touched_groups(&plan, 32, 18, Some(0)).unwrap();
                let ranges = touched_binding_ranges(&groups);
                assert_eq!(ranges.len(), groups.len());
                assert!(
                    ranges.windows(2).all(|w| w[0].1 <= w[1].0),
                    "disjoint ascending"
                );
                for (g, &(lo, hi)) in groups.iter().zip(&ranges) {
                    assert_eq!(
                        (lo, hi),
                        (usize::from(g[0]), usize::from(g[g.len() - 1]) + 1)
                    );
                }
                // `schedule` accepts the tight ranges (every routed expert lies in a binding).
                let panels = plan.schedule(&ranges, PanelLimits::S1).unwrap();
                // Rows: contiguous in ascending order and exactly the layer's padded rows.
                let mut next = None;
                let mut covered = 0usize;
                for panel in panels.iter().filter(|p| p.rows > 0) {
                    if let Some(expected) = next {
                        assert_eq!(
                            panel.row_base, expected,
                            "rows are contiguous across untouched gaps"
                        );
                    }
                    next = Some(panel.row_base + panel.rows);
                    covered += panel.rows;
                    // A panel's experts sit inside its (tight) binding, so the binding covers them.
                    assert!(panel.bind_lo <= panel.expert_lo && panel.expert_hi <= panel.bind_hi);
                    assert!(panel.rows <= PanelLimits::S1.max_grouped_rows);
                }
                assert_eq!(covered, plan.total_rows, "rows {rows} hot {hot}");
                // Ownership of invalid-row slots partitions [0, n_exp) in panel order.
                let mut own = 0usize;
                for panel in &panels {
                    assert_eq!(panel.own_lo, own);
                    own = panel.own_hi;
                }
                assert_eq!(own, 512);
                // The cohort runs once per binding that holds a cohort expert, on the binding's
                // first panel, and the binding covers all of its range.
                for (binding, &(lo, hi)) in ranges.iter().enumerate() {
                    let holds = plan.k1_experts.iter().any(|&e| (lo..hi).contains(&e));
                    let k1_panels = panels
                        .iter()
                        .filter(|p| p.binding == binding && p.k1)
                        .count();
                    assert_eq!(k1_panels, usize::from(holds), "binding {binding}");
                }
                // Every cohort expert is inside exactly one binding.
                for &e in &plan.k1_experts {
                    assert_eq!(
                        ranges
                            .iter()
                            .filter(|&&(lo, hi)| (lo..hi).contains(&e))
                            .count(),
                        1
                    );
                }
            }
        }
    }
}
