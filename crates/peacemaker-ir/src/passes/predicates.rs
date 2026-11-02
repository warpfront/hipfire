// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correlation of repeated scalar comparisons whose inputs have one dominating,
//! non-cyclic definition. Unknown or mutable predicates remain unconstrained.
//!
//! The shared eligibility scan feeds two consumers: the existing wait-partition
//! enumeration (first eight groups only) and all-group predicate discovery for
//! definedness refinement. A small reduced ordered BDD manager backs the
//! symbolic undefined-path computation.
use crate::{
    cfg::{Body, InstId},
    effects::Control,
    inst::Arch,
    operand::Operand,
    reg::Kind,
};
use std::collections::HashMap;

pub(crate) fn successors(body: &Body) -> Vec<Vec<usize>> {
    let n = body.layout.len();
    body.layout
        .iter()
        .enumerate()
        .map(|(p, id)| {
            let inst = body.insts.get(*id).expect("live instruction");
            let target = inst.operands.iter().find_map(|o| match o {
                Operand::Label(b) => body.blocks.get(b.0).map(|b| b.range.0),
                _ => None,
            });
            let mut out = Vec::new();
            if !matches!(
                inst.effects.control,
                Control::Jump | Control::EndPgm | Control::Halt | Control::Trap
            ) && p + 1 < n
            {
                out.push(p + 1);
            }
            if matches!(inst.effects.control, Control::Jump | Control::Branch { .. }) {
                if let Some(t) = target {
                    if !out.contains(&t) {
                        out.push(t);
                    }
                }
            }
            out
        })
        .collect()
}

fn reachable(edges: &[Vec<usize>], starts: &[usize], avoid: Option<usize>) -> Vec<bool> {
    let mut seen = vec![false; edges.len()];
    let mut work = Vec::new();
    for &p in starts {
        if Some(p) != avoid && !seen[p] {
            seen[p] = true;
            work.push(p);
        }
    }
    while let Some(p) = work.pop() {
        for &q in &edges[p] {
            if Some(q) != avoid && !seen[q] {
                seen[q] = true;
                work.push(q);
            }
        }
    }
    seen
}

/// One repeated immutable comparison and the branches correlated by it.
///
/// `branches` holds instruction LAYOUT positions (recorded during the scan, so
/// no position recovery pass is needed) plus SCC polarity: `true` when the
/// branch is taken for SCC1 (`s_cbranch_scc1`), `false` for SCC0. Groups are
/// kept in discovery order and the helper itself never truncates.
struct EligibleGroup {
    compare: InstId,
    // Layout position and SCC polarity for taking the branch.
    branches: Vec<(usize, bool)>,
}

/// Shared eligibility: identical comparison opcode/operands, repeated
/// branches, SCC0/SCC1 polarity, a scalar register input, one immutable
/// scalar definition, dominance (no comparison bypass), and no cyclic scalar
/// definition. Discovery and retention order are preserved. An empty body or
/// no repeated groups yields an empty vector, and the instruction
/// graph/proof state is only built once a candidate group exists.
fn eligible_groups(body: &Body, arch: Arch) -> Vec<EligibleGroup> {
    if body.layout.is_empty() {
        return Vec::new();
    }
    let mut definitions: HashMap<u16, Option<usize>> = HashMap::new();
    for (p, id) in body.layout.iter().enumerate() {
        for reg in &body.insts.get(*id).unwrap().effects.defs {
            if reg.kind == Kind::S {
                for s in reg.base..reg.base + u16::from(reg.len) {
                    definitions
                        .entry(s)
                        .and_modify(|d| *d = None)
                        .or_insert(Some(p));
                }
            }
        }
    }
    let mut groups: Vec<EligibleGroup> = Vec::new();
    for p in 1..body.layout.len() {
        let branch = body.insts.get(body.layout[p]).unwrap();
        let Some(name) = branch.op.name(arch) else {
            continue;
        };
        if !matches!(name, "s_cbranch_scc0" | "s_cbranch_scc1") {
            continue;
        }
        if !branch
            .operands
            .iter()
            .any(|o| matches!(o, Operand::Label(b) if body.blocks.get(b.0).is_some()))
        {
            continue;
        }
        // A branch-target leader may bypass the preceding comparison.
        if body.blocks.iter().any(|b| b.range.0 == p) {
            continue;
        }
        let cmp = body.insts.get(body.layout[p - 1]).unwrap();
        if !cmp.op.name(arch).is_some_and(|n| n.starts_with("s_cmp_")) {
            continue;
        }
        let mut has_register = false;
        let valid = cmp.operands.iter().all(|o| match o {
            Operand::Reg(r) if r.kind == Kind::S => {
                has_register = true;
                (r.base..r.base + u16::from(r.len))
                    .all(|s| definitions.get(&s).is_some_and(|def| def.is_some()))
            }
            Operand::Inline(_) | Operand::Literal(_) => true,
            _ => false,
        });
        if !valid || !has_register {
            continue;
        }
        let group = match groups.iter().position(|group| {
            let prior = body.insts.get(group.compare).unwrap();
            prior.op == cmp.op && prior.operands == cmp.operands
        }) {
            Some(g) => g,
            None => {
                groups.push(EligibleGroup {
                    compare: body.layout[p - 1],
                    branches: Vec::new(),
                });
                groups.len() - 1
            }
        };
        groups[group].branches.push((p, name == "s_cbranch_scc1"));
    }
    groups.retain(|group| group.branches.len() > 1);
    if groups.is_empty() {
        return groups;
    }
    // Most kernels have no repeated immutable comparison. Do not build an
    // instruction graph, much less run dominance/SCC algorithms, for them.
    // Reachability is iterative so long straight-line kernels cannot exhaust
    // the test thread's (or a certifier caller's) stack.
    let edges = successors(body);
    let mut proofs: HashMap<usize, (bool, Vec<bool>)> = HashMap::new();
    groups.retain(|group| {
        group.branches.iter().all(|&(p, _)| {
            let cmp = body.insts.get(body.layout[p - 1]).unwrap();
            cmp.operands.iter().all(|operand| {
                let Operand::Reg(reg) = operand else {
                    return true;
                };
                (reg.base..reg.base + u16::from(reg.len)).all(|s| {
                    let def = definitions[&s].expect("single definition");
                    let (cyclic, bypass) = proofs.entry(def).or_insert_with(|| {
                        let cycle = reachable(&edges, &edges[def], None)[def];
                        (cycle, reachable(&edges, &[0], Some(def)))
                    });
                    !*cyclic && !bypass[p - 1]
                })
            })
        })
    });
    groups
}

/// Each map fixes taken/not-taken for correlated branches. Enumerating both
/// outcomes retains every feasible execution, without guessing scalar values.
///
/// This retains exactly the existing first eight eligible groups and the old
/// mask enumeration: for each branch position, `scc == polarity` is mapped.
/// Zero groups yields one empty map. Distinct eligible comparisons stay
/// independent atoms; no arithmetic implications across thresholds are
/// assumed here either.
pub(crate) fn partitions(body: &Body, arch: Arch) -> Vec<HashMap<InstId, bool>> {
    let mut groups = eligible_groups(body, arch);
    if groups.is_empty() {
        return vec![HashMap::new()];
    }
    // A precision-only bound: unselected predicates keep both CFG edges.
    groups.truncate(8);
    (0..1usize << groups.len())
        .map(|mask| {
            let mut choices = HashMap::new();
            for (g, group) in groups.iter().enumerate() {
                let scc = mask & (1 << g) != 0;
                for &(p, polarity) in &group.branches {
                    choices.insert(body.layout[p], scc == polarity);
                }
            }
            choices
        })
        .collect()
}

/// One eligible conditional branch: which predicate atom guards it, which
/// outcome takes the target edge, and where that edge leads.
///
/// `atom` is the retained discovery order index (atom metadata lives in Flow,
/// not in BDD handles). `taken_when` is true for SCC1 and false for SCC0.
/// `target` is an instruction LAYOUT position: the selected label block's
/// `range.0`. It is not a BlockId, InstId, byte PC, or block-vector index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BranchGuard {
    pub atom: usize,
    pub taken_when: bool,
    pub target: usize,
}

/// All-group predicate guards for a body, indexed by instruction layout
/// position. Positions without an eligible branch yield `None`.
pub(crate) struct PredicateGuards {
    branches: Vec<Option<BranchGuard>>,
}

/// All-group discovery: every eligible group from [`eligible_groups`] becomes
/// one atom, without truncation. Distinct eligible comparisons are
/// independent atoms; no arithmetic implications between thresholds (such as
/// `s35 > 6` versus `s35 > 7`) are inferred.
pub(crate) fn discover(body: &Body, arch: Arch) -> PredicateGuards {
    let groups = eligible_groups(body, arch);
    let mut branches = vec![None; body.layout.len()];
    for (atom, group) in groups.iter().enumerate() {
        for &(position, taken_when) in &group.branches {
            let branch = body.insts.get(body.layout[position]).expect("live branch");
            let target = branch
                .operands
                .iter()
                .find_map(|o| match o {
                    Operand::Label(b) => body.blocks.get(b.0).map(|b| b.range.0),
                    _ => None,
                })
                .expect("eligible branch has a label target");
            branches[position] = Some(BranchGuard {
                atom,
                taken_when,
                target,
            });
        }
    }
    PredicateGuards { branches }
}

impl PredicateGuards {
    pub(crate) fn is_empty(&self) -> bool {
        self.branches.iter().all(|branch| branch.is_none())
    }

    /// Guard for the branch at instruction layout `position`, if eligible.
    pub(crate) fn branch(&self, position: usize) -> Option<BranchGuard> {
        self.branches.get(position).copied().flatten()
    }
}

/// Canonical handle for a Boolean function over discovered predicate atoms.
///
/// `FALSE` and `TRUE` are the two distinct terminal functions. Nonterminal
/// handles are only meaningful within the [`Bdd`] epoch that created them:
/// [`Bdd::reset`] invalidates every nonterminal handle while the terminals
/// stay stable. Never use a handle from a previous epoch after a reset.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct BddId(usize);

impl BddId {
    pub(crate) const FALSE: Self = Self(0);
    pub(crate) const TRUE: Self = Self(1);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct BddNode {
    atom: usize,
    low: BddId,
    high: BddId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum BddOp {
    And,
    Or,
}

#[derive(Clone, Copy)]
enum ApplyTask {
    Compute(BddId, BddId),
    Assemble(BddId, BddId, usize),
}

#[derive(Clone, Copy)]
enum NotTask {
    Visit(BddId),
    Build(BddId),
}

fn norm_pair(a: BddId, b: BddId) -> (BddId, BddId) {
    if a > b {
        (b, a)
    } else {
        (a, b)
    }
}

fn terminal(op: BddOp, a: BddId, b: BddId) -> Option<BddId> {
    match op {
        BddOp::And => {
            if a == BddId::FALSE || b == BddId::FALSE {
                Some(BddId::FALSE)
            } else if a == BddId::TRUE {
                Some(b)
            } else if b == BddId::TRUE {
                Some(a)
            } else if a == b {
                Some(a)
            } else {
                None
            }
        }
        BddOp::Or => {
            if a == BddId::TRUE || b == BddId::TRUE {
                Some(BddId::TRUE)
            } else if a == BddId::FALSE {
                Some(b)
            } else if b == BddId::FALSE {
                Some(a)
            } else if a == b {
                Some(a)
            } else {
                None
            }
        }
    }
}

fn complemented(cached: &HashMap<BddId, BddId>, id: BddId) -> Option<BddId> {
    if id == BddId::FALSE {
        Some(BddId::TRUE)
    } else if id == BddId::TRUE {
        Some(BddId::FALSE)
    } else {
        cached.get(&id).copied()
    }
}

/// Reduced ordered Boolean DAG over predicate atoms, canonicalized by
/// `(atom, low, high)` interning with `low == high` elimination. Variable
/// ordering follows atom indices consistently, and root equality means
/// Boolean-function equality within a live epoch. `and`, `or`, and `not` are
/// exact Boolean operations with terminal simplification and memoization of
/// completed results only. Both apply and complement traversals use explicit
/// iterative work stacks, never recursion proportional to CFG length or atom
/// count. There are no assignment masks, fixed-width atom bitsets, group
/// cutoffs, lossy node merging, or cap-triggered precision fallback. Ordinary
/// `Vec`/`HashMap` storage only; internal work buffers are reused rather
/// than reallocated per operation.
pub(crate) struct Bdd {
    nodes: Vec<BddNode>,
    unique: HashMap<BddNode, BddId>,
    apply_cache: HashMap<(BddOp, BddId, BddId), BddId>,
    not_cache: HashMap<BddId, BddId>,
    apply_work: Vec<ApplyTask>,
    not_work: Vec<NotTask>,
}

impl Bdd {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            unique: HashMap::new(),
            apply_cache: HashMap::new(),
            not_cache: HashMap::new(),
            apply_work: Vec::new(),
            not_work: Vec::new(),
        }
    }

    /// Clears node identity and every intern/apply/complement cache while
    /// retaining capacities. This invalidates all nonterminal handles from the
    /// previous epoch; the caller must rebuild every literal and edge root in
    /// the new epoch and hold no handle across resets. Terminal meanings are
    /// stable and no handle identity is promised across resets.
    pub(crate) fn reset(&mut self) {
        self.nodes.clear();
        self.unique.clear();
        self.apply_cache.clear();
        self.not_cache.clear();
        self.apply_work.clear();
        self.not_work.clear();
    }

    pub(crate) fn literal(&mut self, atom: usize, positive: bool) -> BddId {
        let (low, high) = if positive {
            (BddId::FALSE, BddId::TRUE)
        } else {
            (BddId::TRUE, BddId::FALSE)
        };
        self.intern(atom, low, high)
    }

    pub(crate) fn and(&mut self, a: BddId, b: BddId) -> BddId {
        self.apply(BddOp::And, a, b)
    }

    pub(crate) fn or(&mut self, a: BddId, b: BddId) -> BddId {
        self.apply(BddOp::Or, a, b)
    }

    pub(crate) fn not(&mut self, a: BddId) -> BddId {
        if a == BddId::FALSE {
            return BddId::TRUE;
        }
        if a == BddId::TRUE {
            return BddId::FALSE;
        }
        if let Some(&negated) = self.not_cache.get(&a) {
            return negated;
        }
        // Iterative post-order over the reachable DAG: parents build only
        // after their children complete, with no recursion by atom depth.
        let mut work = std::mem::take(&mut self.not_work);
        work.clear();
        work.push(NotTask::Visit(a));
        while let Some(task) = work.pop() {
            match task {
                NotTask::Visit(id) => {
                    if id == BddId::FALSE || id == BddId::TRUE {
                        continue;
                    }
                    if self.not_cache.contains_key(&id) {
                        continue;
                    }
                    let node = self.nodes[id.0 - 2];
                    work.push(NotTask::Build(id));
                    work.push(NotTask::Visit(node.low));
                    work.push(NotTask::Visit(node.high));
                }
                NotTask::Build(id) => {
                    if self.not_cache.contains_key(&id) {
                        continue;
                    }
                    let node = self.nodes[id.0 - 2];
                    let (Some(low), Some(high)) = (
                        complemented(&self.not_cache, node.low),
                        complemented(&self.not_cache, node.high),
                    ) else {
                        // A child is not ready yet; retry after it completes.
                        work.push(NotTask::Build(id));
                        work.push(NotTask::Visit(node.low));
                        work.push(NotTask::Visit(node.high));
                        continue;
                    };
                    let negated = self.intern(node.atom, low, high);
                    self.not_cache.insert(id, negated);
                }
            }
        }
        self.not_work = work;
        self.not_cache
            .get(&a)
            .copied()
            .expect("complement traversal resolves its root")
    }

    fn apply(&mut self, op: BddOp, a: BddId, b: BddId) -> BddId {
        let (x, y) = norm_pair(a, b);
        if let Some(root) = terminal(op, x, y) {
            return root;
        }
        if let Some(&root) = self.apply_cache.get(&(op, x, y)) {
            return root;
        }
        // Iterative Shannon expansion with commutative operand normalization
        // before caching: parents assemble only after both cofactor pairs
        // complete, with no recursion by atom count.
        let mut work = std::mem::take(&mut self.apply_work);
        work.clear();
        work.push(ApplyTask::Compute(x, y));
        while let Some(task) = work.pop() {
            match task {
                ApplyTask::Compute(a, b) => {
                    let (x, y) = norm_pair(a, b);
                    if let Some(root) = terminal(op, x, y) {
                        self.apply_cache.insert((op, x, y), root);
                        continue;
                    }
                    if self.apply_cache.contains_key(&(op, x, y)) {
                        continue;
                    }
                    let v = self.level(x).min(self.level(y));
                    work.push(ApplyTask::Assemble(x, y, v));
                    let (xlo, xhi) = self.cofactors(x, v);
                    let (ylo, yhi) = self.cofactors(y, v);
                    work.push(ApplyTask::Compute(xlo, ylo));
                    work.push(ApplyTask::Compute(xhi, yhi));
                }
                ApplyTask::Assemble(x, y, v) => {
                    if self.apply_cache.contains_key(&(op, x, y)) {
                        continue;
                    }
                    let (xlo, xhi) = self.cofactors(x, v);
                    let (ylo, yhi) = self.cofactors(y, v);
                    let (Some(lo), Some(hi)) = (
                        self.apply_cache
                            .get(&(op, norm_pair(xlo, ylo).0, norm_pair(xlo, ylo).1))
                            .copied(),
                        self.apply_cache
                            .get(&(op, norm_pair(xhi, yhi).0, norm_pair(xhi, yhi).1))
                            .copied(),
                    ) else {
                        // A cofactor pair is not ready yet; retry after it
                        // completes.
                        work.push(ApplyTask::Assemble(x, y, v));
                        work.push(ApplyTask::Compute(xlo, ylo));
                        work.push(ApplyTask::Compute(xhi, yhi));
                        continue;
                    };
                    let root = self.intern(v, lo, hi);
                    self.apply_cache.insert((op, x, y), root);
                }
            }
        }
        self.apply_work = work;
        self.apply_cache
            .get(&(op, x, y))
            .copied()
            .expect("apply traversal resolves its root")
    }

    fn intern(&mut self, atom: usize, low: BddId, high: BddId) -> BddId {
        if low == high {
            return low;
        }
        let node = BddNode { atom, low, high };
        if let Some(&id) = self.unique.get(&node) {
            return id;
        }
        let id = BddId(self.nodes.len() + 2);
        self.nodes.push(node);
        self.unique.insert(node, id);
        id
    }

    fn level(&self, id: BddId) -> usize {
        if id == BddId::FALSE || id == BddId::TRUE {
            usize::MAX
        } else {
            self.nodes[id.0 - 2].atom
        }
    }

    fn cofactors(&self, id: BddId, v: usize) -> (BddId, BddId) {
        if self.level(id) == v {
            let node = &self.nodes[id.0 - 2];
            (node.low, node.high)
        } else {
            (id, id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inst::{Form, FormFields, Inst, NamedField};
    use crate::operand::{ImmField, InlineConst, Modifiers};
    use crate::provenance::Provenance;
    use crate::reg::RegRef;
    use smallvec::SmallVec;
    use std::collections::HashSet;

    const ARCH: Arch = Arch::Gfx1201;

    fn row(name: &str) -> &'static crate::isa::OpRow {
        crate::isa::gfx12()
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("no row {name}"))
    }

    /// The don't-care fields a zero-filled encoding decodes to (mirrors the
    /// typed-IR test builders: the codec refuses an `Inst` whose fields its
    /// own words would not reproduce).
    fn canonical_fields(r: &crate::isa::OpRow) -> FormFields {
        let mut ignored: SmallVec<[NamedField; 4]> = SmallVec::new();
        let mut honored: SmallVec<[NamedField; 4]> = SmallVec::new();
        if r.form == Form::Vop3 && !r.grammar.contains("SRC2:") && !r.benign_src2.is_empty() {
            if r.fields.len() == 1 {
                return FormFields::Vop3b {
                    src2_unused: r.benign_src2[0],
                };
            }
            ignored.push(NamedField {
                name: "src2_unused",
                value: u32::from(r.benign_src2[0]),
            });
        }
        for rule in r.fields.iter().filter(|rule| rule.name != "src2_unused") {
            let item = NamedField {
                name: rule.name,
                value: rule.allowed.first().copied().unwrap_or(0),
            };
            if rule.class == crate::isa::FieldClass::Honored {
                honored.push(item)
            } else {
                ignored.push(item)
            }
        }
        if ignored.is_empty() && honored.is_empty() {
            FormFields::None
        } else {
            FormFields::Bits { ignored, honored }
        }
    }

    fn mkm(name: &str, operands: Vec<Operand>, mods: Modifiers) -> Inst {
        let r = row(name);
        let inst = Inst::from_parts(
            ARCH,
            r.op,
            r.form,
            canonical_fields(r),
            SmallVec::from_vec(operands),
            mods,
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        if !inst
            .operands
            .iter()
            .any(|op| matches!(op, Operand::Label(_)))
        {
            crate::codec::gfx12::encode(&inst).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        inst
    }

    fn mk(name: &str, operands: Vec<Operand>) -> Inst {
        mkm(name, operands, Modifiers::default())
    }

    fn sr(base: u16) -> Operand {
        Operand::Reg(RegRef {
            kind: Kind::S,
            base,
            len: 1,
        })
    }

    fn int(n: i8) -> Operand {
        Operand::Inline(InlineConst::Integer(n))
    }

    fn smov_def(dst: u16) -> Inst {
        mk("s_mov_b32", vec![sr(dst), sr(0)])
    }

    fn cmp_ge(scalar: u16, thresh: i8) -> Inst {
        mk("s_cmp_ge_u32", vec![sr(scalar), int(thresh)])
    }

    fn nop() -> Inst {
        mk("s_nop", vec![Operand::Imm(ImmField::Sopp(0))])
    }

    fn endpgm() -> Inst {
        mk("s_endpgm", vec![])
    }

    enum It {
        I(Inst),
        L(&'static str),
        B(&'static str, &'static str),
    }
    use It::{B, I, L};

    /// Assemble items (labels, branches by label) into a body with built
    /// blocks, mirroring the typed-IR test builders.
    fn body_of(items: Vec<It>) -> Body {
        let mut insts = Vec::new();
        let mut labels = HashMap::new();
        let mut fixups = Vec::new();
        for item in items {
            match item {
                I(inst) => insts.push(inst),
                L(name) => {
                    labels.insert(name, insts.len());
                }
                B(op, target) => {
                    fixups.push((insts.len(), target));
                    insts.push(mk(op, vec![Operand::Imm(ImmField::Sopp(0))]));
                }
            }
        }
        let mut pcs = vec![0usize];
        for inst in &insts {
            pcs.push(pcs.last().unwrap() + crate::passes::cfg::dwords_of(inst, ARCH));
        }
        for (at, target) in fixups {
            let off = pcs[labels[target]] as i64 - pcs[at + 1] as i64;
            insts[at].operands[0] = Operand::Imm(ImmField::Sopp(off as i16));
        }
        let mut body = Body::default();
        for inst in insts {
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        crate::passes::cfg::build_blocks(&mut body, ARCH).unwrap();
        body
    }

    /// Evaluate a BDD root under a complete atom assignment (test oracle
    /// only; production code never enumerates assignments).
    fn eval_bdd(bdd: &Bdd, id: BddId, values: &[bool]) -> bool {
        let mut cur = id;
        loop {
            if cur == BddId::FALSE {
                return false;
            }
            if cur == BddId::TRUE {
                return true;
            }
            let node = &bdd.nodes[cur.0 - 2];
            cur = if values[node.atom] {
                node.high
            } else {
                node.low
            };
        }
    }

    /// Body with `count` distinct repeated comparisons over one immutable
    /// scalar (thresholds `0..count`), each appearing twice. All branches
    /// target the final label; group 7 uses SCC0 branches, the rest SCC1.
    fn multi_group_body(count: usize) -> Body {
        let mut items = vec![I(smov_def(32))];
        for k in 0..count {
            let branch = if k == 7 {
                "s_cbranch_scc0"
            } else {
                "s_cbranch_scc1"
            };
            for _ in 0..2 {
                items.push(I(cmp_ge(32, k as i8)));
                items.push(B(branch, "done"));
            }
        }
        items.push(L("done"));
        items.push(I(endpgm()));
        body_of(items)
    }

    #[test]
    fn bdd_truth_tables_and_canonical_roots() {
        let mut bdd = Bdd::new();
        let x0 = bdd.literal(0, true);
        let nx0 = bdd.literal(0, false);
        assert_ne!(x0, nx0);
        assert_eq!(bdd.not(x0), nx0);
        assert_eq!(bdd.not(nx0), x0);
        let not_x0 = bdd.not(x0);
        assert_eq!(bdd.not(not_x0), x0);
        // Terminal and idempotent simplifications.
        assert_eq!(bdd.and(BddId::FALSE, x0), BddId::FALSE);
        assert_eq!(bdd.and(BddId::TRUE, x0), x0);
        assert_eq!(bdd.and(x0, x0), x0);
        assert_eq!(bdd.and(x0, nx0), BddId::FALSE);
        assert_eq!(bdd.or(BddId::TRUE, x0), BddId::TRUE);
        assert_eq!(bdd.or(BddId::FALSE, x0), x0);
        assert_eq!(bdd.or(x0, x0), x0);
        assert_eq!(bdd.or(x0, nx0), BddId::TRUE);
        // Commutative operand order yields canonical roots.
        let x1 = bdd.literal(1, true);
        let nx1 = bdd.literal(1, false);
        let and_fwd = bdd.and(x0, x1);
        let and_rev = bdd.and(x1, x0);
        assert_eq!(and_fwd, and_rev);
        let or_fwd = bdd.or(x0, x1);
        let or_rev = bdd.or(x1, x0);
        assert_eq!(or_fwd, or_rev);
        // Absorption collapses through interning: x0 | (x0 & x1) == x0.
        let absorb_inner = bdd.and(x0, x1);
        assert_eq!(bdd.or(x0, absorb_inner), x0);
        // De Morgan duality holds as root equality.
        let dm_and = bdd.and(x0, x1);
        let dm_or = bdd.or(nx0, nx1);
        assert_eq!(bdd.not(dm_and), dm_or);
        // Exhaustive truth tables over two atoms, plus a three-atom spot check.
        let x2 = bdd.literal(2, true);
        let and01 = bdd.and(x0, x1);
        let or01 = bdd.or(x0, x1);
        let and012 = bdd.and(and01, x2);
        for a in [false, true] {
            assert_eq!(eval_bdd(&bdd, x0, &[a, false, false]), a);
            assert_eq!(eval_bdd(&bdd, nx0, &[a, false, false]), !a);
            for b in [false, true] {
                let v = [a, b, false];
                assert_eq!(eval_bdd(&bdd, and01, &v), a && b);
                assert_eq!(eval_bdd(&bdd, or01, &v), a || b);
                for c in [false, true] {
                    assert_eq!(eval_bdd(&bdd, and012, &[a, b, c]), a && b && c);
                }
            }
        }
    }

    #[test]
    fn bdd_handles_more_than_64_atoms() {
        let mut bdd = Bdd::new();
        let mut conj = BddId::TRUE;
        for atom in 0..70usize {
            let lit = bdd.literal(atom, true);
            conj = bdd.and(conj, lit);
        }
        // No machine-word atom boundary: semantics hold past 64 atoms.
        let all_true = vec![true; 70];
        assert!(eval_bdd(&bdd, conj, &all_true));
        for flip in [0, 1, 63, 64, 65, 69] {
            let mut values = vec![true; 70];
            values[flip] = false;
            assert!(
                !eval_bdd(&bdd, conj, &values),
                "wide conjunction fails open at atom {flip}"
            );
        }
        // Complement of the wide conjunction: only all-true falsifies it.
        let disj = bdd.not(conj);
        assert!(!eval_bdd(&bdd, disj, &all_true));
        let mut values = vec![true; 70];
        values[17] = false;
        assert!(eval_bdd(&bdd, disj, &values));
    }

    #[test]
    fn bdd_reset_rebuilds_roots() {
        let mut bdd = Bdd::new();
        let x0 = bdd.literal(0, true);
        let x1_fresh = bdd.literal(1, true);
        let conj = bdd.and(x0, x1_fresh);
        assert!(eval_bdd(&bdd, conj, &[true, true]));
        assert!(!bdd.nodes.is_empty());
        bdd.reset();
        assert!(bdd.nodes.is_empty());
        assert!(bdd.unique.is_empty());
        assert!(bdd.apply_cache.is_empty());
        assert!(bdd.not_cache.is_empty());
        // Terminals stay meaningful; every nonterminal root is rebuilt fresh
        // in the new epoch and no prior handle is reused.
        assert_eq!(bdd.and(BddId::FALSE, BddId::TRUE), BddId::FALSE);
        assert_eq!(bdd.or(BddId::FALSE, BddId::TRUE), BddId::TRUE);
        let y0 = bdd.literal(0, true);
        let ny0 = bdd.literal(0, false);
        assert_eq!(bdd.not(y0), ny0);
        let y1 = bdd.literal(1, true);
        let rebuilt = bdd.and(y0, y1);
        assert!(eval_bdd(&bdd, rebuilt, &[true, true]));
        assert!(!eval_bdd(&bdd, rebuilt, &[true, false]));
        assert!(!eval_bdd(&bdd, rebuilt, &[false, true]));
    }

    #[test]
    fn discovery_preserves_eligibility() {
        // Immutable repeated comparison: both branches guarded by one atom,
        // with SCC1/SCC0 polarity and layout-position targets.
        let body = body_of(vec![
            I(smov_def(32)),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "skip"),
            I(nop()),
            L("skip"),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc0", "done"),
            I(nop()),
            L("done"),
            I(endpgm()),
        ]);
        let guards = discover(&body, ARCH);
        assert!(!guards.is_empty());
        let first = guards.branch(2).expect("first repeated branch is guarded");
        assert_eq!(first.atom, 0);
        assert!(first.taken_when);
        assert_eq!(first.target, 4);
        let second = guards.branch(5).expect("second repeated branch is guarded");
        assert_eq!(second.atom, first.atom);
        assert!(!second.taken_when);
        assert_eq!(second.target, 7);
        for p in [0, 1, 3, 4, 6, 7] {
            assert_eq!(guards.branch(p), None, "position {p} carries no guard");
        }
        assert_eq!(guards.branch(body.layout.len() + 16), None);

        // Mutable scalar (redefined between the branches): unconstrained.
        let body = body_of(vec![
            I(smov_def(32)),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "skip"),
            I(nop()),
            L("skip"),
            I(smov_def(32)),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "done"),
            L("done"),
            I(endpgm()),
        ]);
        assert!(discover(&body, ARCH).is_empty());

        // Cyclic scalar definition (defined inside its own loop): unconstrained.
        let body = body_of(vec![
            L("loop"),
            I(smov_def(32)),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "out"),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "out"),
            B("s_branch", "loop"),
            L("out"),
            I(endpgm()),
        ]);
        assert!(discover(&body, ARCH).is_empty());

        // Comparison bypass (entry reaches a comparison without passing the
        // single definition): unconstrained.
        let body = body_of(vec![
            B("s_branch", "mid"),
            I(smov_def(32)),
            L("mid"),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "out"),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "out"),
            L("out"),
            I(endpgm()),
        ]);
        assert!(discover(&body, ARCH).is_empty());
    }

    #[test]
    fn discovery_has_no_machine_word_group_limit() {
        // A single (non-repeated) comparison yields no guards; an empty body
        // behaves the same.
        let body = body_of(vec![
            I(smov_def(32)),
            I(cmp_ge(32, 2)),
            B("s_cbranch_scc1", "done"),
            L("done"),
            I(endpgm()),
        ]);
        assert!(discover(&body, ARCH).is_empty());
        assert!(discover(&Body::default(), ARCH).is_empty());

        // Nine eligible groups: discovery covers all nine with distinct atoms,
        // SCC polarity, and layout-position targets.
        let body = multi_group_body(9);
        let guards = discover(&body, ARCH);
        assert!(!guards.is_empty());
        let mut atoms = HashSet::new();
        for k in 0..9 {
            let first = guards
                .branch(2 + 4 * k)
                .expect("repeated branch is guarded");
            let second = guards
                .branch(4 + 4 * k)
                .expect("repeated branch is guarded");
            assert_eq!(second.atom, first.atom, "group {k} shares one atom");
            assert_eq!(first.target, 37);
            assert_eq!(first.taken_when, k != 7, "group {k} polarity");
            assert_eq!(second.taken_when, first.taken_when);
            assert!(atoms.insert(first.atom), "group {k} has a distinct atom");
        }
        assert_eq!(atoms.len(), 9);

        // Discovery has no group cap: 65 distinct eligible comparisons yield
        // 65 atoms with the same per-group shape.
        let body = multi_group_body(65);
        let guards = discover(&body, ARCH);
        let mut atoms = HashSet::new();
        for k in 0..65 {
            let first = guards.branch(2 + 4 * k).expect("guarded");
            let second = guards.branch(4 + 4 * k).expect("guarded");
            assert_eq!(second.atom, first.atom);
            assert_eq!(first.target, 4 * 65 + 1);
            assert_eq!(second.taken_when, first.taken_when);
            atoms.insert(first.atom);
        }
        assert_eq!(atoms.len(), 65);
    }
}
