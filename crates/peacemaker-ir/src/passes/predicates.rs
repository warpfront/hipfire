//! Correlation of repeated scalar comparisons whose inputs have one dominating,
//! non-cyclic definition. Unknown or mutable predicates remain unconstrained.
use std::collections::HashMap;
use crate::{cfg::{Body, InstId}, effects::Control, inst::Arch, operand::Operand, reg::Kind};

pub(crate) fn successors(body: &Body) -> Vec<Vec<usize>> {
    let n = body.layout.len();
    body.layout.iter().enumerate().map(|(p, id)| {
        let inst = body.insts.get(*id).expect("live instruction");
        let target = inst.operands.iter().find_map(|o| match o {
            Operand::Label(b) => body.blocks.get(b.0).map(|b| b.range.0), _ => None,
        });
        let mut out = Vec::new();
        if !matches!(inst.effects.control, Control::Jump | Control::EndPgm | Control::Halt | Control::Trap) && p + 1 < n { out.push(p + 1); }
        if matches!(inst.effects.control, Control::Jump | Control::Branch { .. }) {
            if let Some(t) = target { if !out.contains(&t) { out.push(t); } }
        }
        out
    }).collect()
}


fn reachable(edges: &[Vec<usize>], starts: &[usize], avoid: Option<usize>) -> Vec<bool> {
    let mut seen = vec![false; edges.len()];
    let mut work = Vec::new();
    for &p in starts {
        if Some(p) != avoid && !seen[p] { seen[p] = true; work.push(p); }
    }
    while let Some(p) = work.pop() {
        for &q in &edges[p] {
            if Some(q) != avoid && !seen[q] { seen[q] = true; work.push(q); }
        }
    }
    seen
}

/// Each map fixes taken/not-taken for correlated branches. Enumerating both
/// outcomes retains every feasible execution, without guessing scalar values.
pub(crate) fn partitions(body: &Body, arch: Arch) -> Vec<HashMap<InstId, bool>> {
    if body.layout.is_empty() { return vec![HashMap::new()]; }
    let mut definitions: HashMap<u16, Option<usize>> = HashMap::new();
    for (p, id) in body.layout.iter().enumerate() {
        for reg in &body.insts.get(*id).unwrap().effects.defs {
            if reg.kind == Kind::S {
                for s in reg.base..reg.base + u16::from(reg.len) {
                    definitions.entry(s).and_modify(|d| *d = None).or_insert(Some(p));
                }
            }
        }
    }
    let mut groups: Vec<(InstId, Vec<(InstId, bool)>)> = Vec::new();
    for p in 1..body.layout.len() {
        let branch = body.insts.get(body.layout[p]).unwrap();
        let Some(name) = branch.op.name(arch) else { continue };
        if !matches!(name, "s_cbranch_scc0" | "s_cbranch_scc1") { continue; }
        if !branch.operands.iter().any(|o| matches!(o, Operand::Label(b) if body.blocks.get(b.0).is_some())) { continue; }
        // A branch-target leader may bypass the preceding comparison.
        if body.blocks.iter().any(|b| b.range.0 == p) { continue; }
        let cmp = body.insts.get(body.layout[p - 1]).unwrap();
        if !cmp.op.name(arch).is_some_and(|n| n.starts_with("s_cmp_")) { continue; }
        let mut has_register = false;
        let valid = cmp.operands.iter().all(|o| match o {
            Operand::Reg(r) if r.kind == Kind::S => {
                has_register = true;
                (r.base..r.base + u16::from(r.len)).all(|s| {
                    definitions.get(&s).is_some_and(|def| def.is_some())
                })
            }
            Operand::Inline(_) | Operand::Literal(_) => true,
            _ => false,
        });
        if !valid || !has_register { continue; }
        let group = match groups.iter().position(|(id, _)| {
            let prior = body.insts.get(*id).unwrap();
            prior.op == cmp.op && prior.operands == cmp.operands
        }) {
            Some(g) => g,
            None => { groups.push((body.layout[p - 1], Vec::new())); groups.len() - 1 }
        };
        groups[group].1.push((body.layout[p], name == "s_cbranch_scc1"));
    }
    groups.retain(|(_, branches)| branches.len() > 1);
    if groups.is_empty() { return vec![HashMap::new()]; }
    // Most kernels have no repeated immutable comparison. Do not build an
    // instruction graph, much less run dominance/SCC algorithms, for them.
    // Reachability is iterative so long straight-line kernels cannot exhaust
    // the test thread's (or a certifier caller's) stack.
    let edges = successors(body);
    let mut proofs: HashMap<usize, (bool, Vec<bool>)> = HashMap::new();
    groups.retain(|(_, branches)| branches.iter().all(|(id, _)| {
        let p = body.layout.iter().position(|candidate| candidate == id).expect("branch in layout");
        let cmp = body.insts.get(body.layout[p - 1]).unwrap();
        cmp.operands.iter().all(|operand| {
            let Operand::Reg(reg) = operand else { return true };
            (reg.base..reg.base + u16::from(reg.len)).all(|s| {
                let def = definitions[&s].expect("single definition");
                let (cyclic, bypass) = proofs.entry(def).or_insert_with(|| {
                    let cycle = reachable(&edges, &edges[def], None)[def];
                    (cycle, reachable(&edges, &[0], Some(def)))
                });
                !*cyclic && !bypass[p - 1]
            })
        })
    }));
    // A precision-only bound: unselected predicates keep both CFG edges.
    groups.truncate(8);
    (0..1usize << groups.len()).map(|mask| {
        let mut choices = HashMap::new();
        for (g, (_, branches)) in groups.iter().enumerate() {
            let scc = mask & (1 << g) != 0;
            for &(id, polarity) in branches { choices.insert(id, scc == polarity); }
        }
        choices
    }).collect()
}
