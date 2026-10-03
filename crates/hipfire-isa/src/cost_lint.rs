// SPDX-License-Identifier: Apache-2.0
//! Static schedule evidence for lifted code objects. Ratios describe an emitted
//! loop trip, not elapsed cycles; without routing data padding is unknown.
use peacemaker_ir::{inst::{Arch, Form}, passes::{cfg::Cfg, waits::replay}, reg::Kind, wait::Counter};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Route {
    /// Real rows per routed expert in this capture; absent for dense kernels.
    #[serde(default)] pub rows: Vec<u32>,
    /// Kernel's executed row tile, from its launch contract.
    #[serde(default)] pub tile_rows: u32,
    /// K128 epochs executed in one hot-loop trip, from the kernel tile rule.
    #[serde(default)] pub k128_epochs: Option<usize>,
    /// Exact instruction-layout starts of each K128 epoch, keyed by natural
    /// loop header block ID. Optional because ISA alone cannot identify the
    /// producer's K-step boundaries in arbitrary hand-written HIP kernels.
    #[serde(default)] pub epoch_starts: BTreeMap<usize, Vec<usize>>,
}
impl Route {
    pub fn padding_fraction(&self) -> Result<Option<f64>, String> {
        if self.rows.is_empty() { return Ok(None); }
        if self.tile_rows == 0 { return Err("tile_rows must be positive when rows are provided".into()); }
        let real: u64 = self.rows.iter().map(|&n| u64::from(n)).sum();
        let executed: u64 = self.rows.iter().map(|&n| u64::from(n).div_ceil(u64::from(self.tile_rows)) * u64::from(self.tile_rows)).sum();
        if executed == 0 { return Err("route has no executed rows".into()); }
        Ok(Some((executed - real) as f64 / executed as f64))
    }
}

#[derive(Debug, Serialize)]
pub struct LoopCost {
    pub blocks: Vec<usize>, pub wmma_per_trip: usize, pub k128_epochs: Option<usize>,
    pub wmma_per_k128: Option<f64>, pub independent_chains: usize,
    pub valu_per_wmma: f64, pub vmem_per_wmma: f64, pub vmem_bytes_per_wmma: f64,
    /// Minimum pending VMEM units immediately before an epoch's first WMMA.
    pub wait_depth_before_wmma: Option<u64>,
    /// Greatest epoch distance between a loop VMEM load and the first wait
    /// retiring it; requires declared epoch boundaries. Null if unobservable.
    pub prefetch_distance_epochs: Option<usize>,
    pub padding_fraction: Option<f64>, pub findings: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct KernelCost { pub symbol: String, pub loops: Vec<LoopCost> }
#[derive(Debug, Serialize)]
pub struct CostReport { pub arch: String, pub kernels: Vec<KernelCost> }

/// Lost VMEM overlap per issued WMMA in a paired-object comparison. This is
/// a *ranking* cost, not a calibrated latency or predicted speedup. It does
/// not use the number of deleted wait instructions.
pub fn drain_pressure(before_depth: u64, after_depth: u64, vmem_per_wmma: f64, wmma_per_trip: usize) -> f64 {
    if wmma_per_trip == 0 { return 0.0; }
    before_depth.saturating_sub(after_depth) as f64 * vmem_per_wmma / wmma_per_trip as f64
}

fn vmem_bytes(name: &str) -> u64 {
    if let Some(n) = name.rsplit('_').next().and_then(|s| s.strip_prefix('b')).and_then(|s| s.parse::<u64>().ok()) { return n / 8; }
    for (suffix, bytes) in [("_u8",1),("_i8",1),("_u16",2),("_i16",2),("_f16",2),("_b32",4),("_f32",4)] {
        if name.ends_with(suffix) { return bytes; }
    }
    0 // An unmodeled image/atomic byte width is not guessed.
}
fn vmem_pending(state: &peacemaker_ir::wait::WaitState, arch: Arch) -> u64 {
    let counter = if arch == Arch::Gfx1201 { Counter::Load } else { Counter::Vm };
    state.pending.iter().filter(|p| p.counters.contains(counter) && !p.satisfied.contains(counter))
        .map(|p| u64::from(p.units[counter as usize])).sum()
}
fn prefetch_distance(body: &peacemaker_ir::cfg::Body, waits: &peacemaker_ir::passes::waits::WaitReplay,
    positions: &[usize], starts: &[usize]) -> Result<Option<usize>, String> {
    if starts.is_empty() { return Ok(None); }
    if starts[0] != positions[0] || starts.windows(2).any(|w| w[0] >= w[1])
        || starts.iter().any(|start| positions.binary_search(start).is_err()) {
        return Err("epoch starts must begin at the loop's first instruction and increase within the loop".into());
    }
    let layout_positions: std::collections::HashMap<_, _> = body.layout.iter().enumerate().map(|(pos, &id)| (id, pos)).collect();
    let epoch = |pos: usize| starts.partition_point(|start| *start <= pos) - 1;
    let mut distance = Some(0usize);
    for event in &waits.events {
        if !matches!(event.class, peacemaker_ir::effects::MemClass::VmemLoad | peacemaker_ir::effects::MemClass::FlatLoad) {
            continue;
        }
        let Some(&issued) = layout_positions.get(&event.inst) else { continue };
        if positions.binary_search(&issued).is_err() { continue; }
        // A wait at an earlier layout position retires this event on the next
        // loop trip. Choose the first such wait in cyclic instruction order.
        let first = waits.facts.iter().filter(|fact| fact.satisfies.contains(&event.id))
            .filter_map(|fact| layout_positions.get(&fact.wait).copied())
            .filter(|pos| positions.binary_search(pos).is_ok())
            .min_by_key(|&pos| (pos < issued, pos));
        let Some(retired) = first else { return Ok(None) };
        let epochs = if retired >= issued { epoch(retired) - epoch(issued) }
            else { starts.len() + epoch(retired) - epoch(issued) };
        distance = distance.map(|d| d.max(epochs));
    }
    Ok(distance)
}


pub fn analyze(program: &peacemaker_ir::Program, routes: &BTreeMap<String, Route>) -> Result<CostReport, String> {
    let arch = program.target.arch;
    let mut kernels = Vec::new();
    for kernel in &program.kernels {
        let body = &kernel.body;
        let graph = Cfg::build(body).map_err(|e| format!("{}: {e}", kernel.symbol.0))?;
        let waits = replay(body, arch, kernel.wave).map_err(|e| format!("{}: {e}", kernel.symbol.0))?;
        let mut loops = Vec::new();
        for info in graph.loops() {
            // A natural loop may have multiple exits; no dynamic branch frequency
            // is inferred. Each instruction in its member blocks is counted once.
            let mut positions: Vec<usize> = info.members.iter().flat_map(|id| body.blocks.iter()
                .filter(move |block| block.id == *id)
                .flat_map(|block| block.range.0..block.range.1)).collect();
            positions.sort_unstable(); positions.dedup();
            let mut chains = BTreeSet::new();
            let (mut wmma, mut valu, mut vmem, mut bytes) = (0, 0, 0, 0);
            let mut depth = None::<u64>;
            for &pos in &positions {
                let id = body.layout[pos]; let inst = body.insts.get(id).ok_or("dangling instruction")?;
                let name = inst.op.name(arch).unwrap_or("");
                if name.starts_with("v_wmma_") || name.starts_with("v_swmmac_") {
                    wmma += 1;
                    if let Some(dst) = inst.effects.defs.iter().find(|r| r.kind == Kind::V) {
                        // Re-seeding a destination is not a second chain. The
                        // physical destination is the in-loop recurrence identity.
                        chains.insert((dst.base, dst.len));
                    }
                    if let Some(state) = waits.before.get(&id) {
                        let pending = vmem_pending(state, arch);
                        depth = Some(depth.map_or(pending, |d| d.min(pending)));
                    }
                } else if name.starts_with("v_") { valu += 1; }
                if matches!(inst.form, Form::Vmem(_)) {
                    vmem += 1;
                    bytes += vmem_bytes(name);
                }
            }
            if wmma == 0 { continue; }
            let mut findings = Vec::new();
            if chains.len() == 1 { findings.push("single_accumulator_chain".into()); }
            // A full drain matters here when a single-chain, VMEM-heavy loop
            // has no outstanding work before WMMA. Dense multi-chain kernels
            // and low-VMEM loops can legitimately drain without this finding.
            if depth == Some(0) && chains.len() == 1 && vmem * 2 >= wmma {
                findings.push("prefetch_drained_before_wmma".into());
            }
            let route = routes.get(&kernel.symbol.0);
            let padding = route.map(Route::padding_fraction).transpose()?.flatten();
            let epochs = route.and_then(|r| r.k128_epochs);
            if epochs == Some(0) { return Err(format!("{}: k128_epochs must be positive", kernel.symbol.0)); }
            let per_epoch = epochs.map(|n| wmma as f64 / n as f64);
            let prefetch = if let Some(starts) = route.and_then(|r| r.epoch_starts.get(&info.header.0)) {
                if Some(starts.len()) != epochs { return Err(format!("{}: epoch starts must match k128_epochs", kernel.symbol.0)); }
                prefetch_distance(body, &waits, &positions, starts)?
            } else { None };
            loops.push(LoopCost { blocks: info.members.iter().map(|id| id.0).collect(), wmma_per_trip: wmma,
                k128_epochs: epochs, wmma_per_k128: per_epoch, independent_chains: chains.len(),
                valu_per_wmma: valu as f64 / wmma as f64, vmem_per_wmma: vmem as f64 / wmma as f64,
                vmem_bytes_per_wmma: bytes as f64 / wmma as f64,
                wait_depth_before_wmma: depth, prefetch_distance_epochs: prefetch,
                padding_fraction: padding, findings });
        }
        kernels.push(KernelCost { symbol: kernel.symbol.0.clone(), loops });
    }
    Ok(CostReport { arch: format!("{arch:?}"), kernels })
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn padding_on_ragged_experts() {
        assert_eq!(Route { rows: vec![1, 16, 17], tile_rows: 16, ..Default::default() }.padding_fraction().unwrap(), Some(30.0/64.0));
        assert!(Route { rows: vec![1], tile_rows: 0, ..Default::default() }.padding_fraction().is_err());
    }

    #[test] fn first_retiring_wait_measures_prefetch_epoch_distance() {
        use peacemaker_ir::{cfg::{Body, InstId}, effects::{MemClass, OrderType}, passes::waits::WaitReplay, wait::{CounterSet, EventId, PendingEvent, WaitFact}};
        let body = Body { layout: (0..6).map(InstId).collect(), ..Body::default() };
        let event = PendingEvent { id: EventId(0), inst: InstId(1), counters: CounterSet(0),
            units: [0; 10], satisfied: CounterSet(0), younger: [0; 10],
            class: MemClass::VmemLoad, defs: Default::default(), src_locks: Default::default(),
            in_order_type: OrderType::Load };
        let mut waits = WaitReplay { events: vec![event], facts: vec![WaitFact { wait: InstId(4),
            satisfies: vec![EventId(0)] }], obligations: Vec::new(), before: Default::default() };
        let positions = [0, 1, 2, 3, 4, 5]; let starts = [0, 3];
        assert_eq!(prefetch_distance(&body, &waits, &positions, &starts).unwrap(), Some(1));
        waits.facts.push(WaitFact { wait: InstId(2), satisfies: vec![EventId(0)] });
        assert_eq!(prefetch_distance(&body, &waits, &positions, &starts).unwrap(), Some(0));
        assert!(prefetch_distance(&body, &waits, &positions, &[0, 7]).is_err());
        waits.facts.clear();
        assert_eq!(prefetch_distance(&body, &waits, &positions, &starts).unwrap(), None);
    }

    #[test] fn calibrated_regression_precision_and_decisive_pair_rho() {
        let data: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/cost_lint/wait_delete.json")).unwrap();
        let rows = data["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 26);
        let mut points = Vec::new();
        let mut neutral = Vec::new();
        for row in rows {
            let fields = row.as_array().unwrap();
            let ratios = fields[1].as_array().unwrap();
            let median = ratios[1].as_f64().unwrap();
            let before = fields[2].as_u64().unwrap();
            let after = fields[3].as_u64().unwrap();
            let wmma = fields[4].as_u64().unwrap() as usize;
            let vmem = fields[5].as_f64().unwrap();
            let measured_regression = median > 1.10; // the six observed +12–24% losses
            let flagged = before > after && after == 0 && vmem >= 0.5;
            assert_eq!(flagged, measured_regression, "{}", fields[0]);
            if !measured_regression { neutral.push(median - 1.0); }
            points.push((median, drain_pressure(before, after, vmem, wmma)));
        }
        assert_eq!(neutral.len(), 20);
        assert_eq!(points.iter().filter(|p| p.1 > 0.0).count(), 6);
        let mean = neutral.iter().sum::<f64>() / neutral.len() as f64;
        let sigma = (neutral.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / (neutral.len()-1) as f64).sqrt();
        let floor = 3.0 * sigma;
        let mut pairs = Vec::new();
        for (i, a) in points.iter().enumerate() {
            for b in points.iter().skip(i+1) {
                let measured = a.0-b.0;
                if measured.abs() > floor { pairs.push((measured, a.1-b.1)); }
            }
        }
        // Mid-ranks give static ties (same emitted loop for gate/up vs down)
        // their honest average position; do not invent a tie-breaking signal.
        fn ranks(values: &[f64]) -> Vec<f64> {
            let mut ordered = values.to_vec(); ordered.sort_by(f64::total_cmp);
            values.iter().map(|x| {
                let first = ordered.partition_point(|y| y < x);
                let last = ordered.partition_point(|y| y <= x);
                (first + last + 1) as f64 / 2.0
            }).collect()
        }
        let x = ranks(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let y = ranks(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
        let (mx,my) = (x.iter().sum::<f64>() / x.len() as f64, y.iter().sum::<f64>() / y.len() as f64);
        let (mut cross, mut xx, mut yy) = (0.0,0.0,0.0);
        for (a,b) in x.iter().zip(y.iter()) {
            cross += (a-mx)*(b-my); xx += (a-mx).powi(2); yy += (b-my).powi(2);
        }
        let rho = cross / (xx*yy).sqrt();
        assert_eq!(pairs.len(), 149);
        assert!(rho >= 0.8, "decisive-pair rank rho {rho:.6}, noise floor {floor:.6}");
    }

    /// External historical modules are not checked into the source tree; the
    /// gate runs whenever the pinned CPU-only calibration corpus is mounted.
    #[test] fn wait_delete_flags_six_and_not_the_other_twenty() {
        let root = std::path::Path::new("/home/kaden/qcal/release-0.4.1/pm-wait-delete");
        if !root.join("before/sym1151.co").exists() { return; }
        let mut total = 0;
        let mut newly_flagged = 0;
        for module in ["b1", "b1s", "v2b", "v2c", "sym1151", "sym1201"] {
            let lift = |phase: &str| {
                let object = std::fs::read(root.join(phase).join(format!("{module}.co"))).unwrap();
                let lifted = peacemaker_lift::lift_object(&object, peacemaker_lift::Options {
                    frontend: peacemaker_ir::inst::Frontend::Hipcc,
                }).unwrap();
                analyze(&lifted.program, &BTreeMap::new()).unwrap()
            };
            let before = lift("before"); let after = lift("after");
            for (a,b) in before.kernels.iter().zip(&after.kernels) {
                assert_eq!(a.symbol, b.symbol);
                let previously = a.loops.iter().any(|l| l.findings.iter().any(|f| f == "prefetch_drained_before_wmma"));
                let currently = b.loops.iter().any(|l| l.findings.iter().any(|f| f == "prefetch_drained_before_wmma"));
                let regressed = a.symbol.contains("_nt4") || a.symbol.contains("_nt8");
                assert_eq!(currently && !previously, regressed, "{}", a.symbol);
                newly_flagged += usize::from(currently && !previously);
                total += 1;
            }
        }
        assert_eq!((total, newly_flagged), (26, 6));
    }
}
