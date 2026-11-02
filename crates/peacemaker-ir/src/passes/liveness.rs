// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C4: physical-register liveness and locally-free register facts.
//!
//! Standard backward may-dataflow over the CFG at dword granularity, per
//! core.md §2.5. The lattice locations are individual register dwords plus
//! the implicit state (`Exec`/`Vcc`/`Scc`/`M0`, each covering every dword the
//! wavefront uses) and a small set of kernel-input specials. Explicit
//! `Reg`/`Half` direction comes from `Effects.defs`/`uses` (the C3 contract);
//! explicit `Special` operands and `Hwreg` are treated as both use and def,
//! which over-approximates liveness in the sound direction for free-register
//! queries (fewer free registers, never a live register reported free).
//! `ImplicitSet` bit meanings are C1's pinned masks
//! (`ImplicitSet::{SCC,VCC,EXEC,M0,MODE}`).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::cfg::{BlockId, Body, InstId};
use crate::effects::ImplicitSet;
use crate::inst::Inst;
use crate::operand::{Operand, Special};
use crate::passes::cfg::containing_block;
use crate::reg::{Kind, RegRef, RegSet};

/// One dword-granular liveness location.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Loc {
    V(u16),
    S(u16),
    Ttmp(u16),
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    Scc,
    M0,
    /// Kernel-input specials with no architectural dword: 0 = flat scratch,
    /// 1..=4 = shared/private base/limit, 5 = pc, 6 = hwreg, 7 = mode.
    Other(u8),
}

fn reg_locs(reg: RegRef) -> Vec<Loc> {
    let mut out = Vec::with_capacity(reg.len as usize);
    for i in 0..reg.len {
        let index = reg.base + u16::from(i);
        out.push(match reg.kind {
            Kind::V => Loc::V(index),
            Kind::S => Loc::S(index),
            Kind::Ttmp => Loc::Ttmp(index),
        });
    }
    out
}

fn special_locs(special: Special) -> Vec<Loc> {
    match special {
        Special::Vcc => vec![Loc::VccLo, Loc::VccHi],
        Special::VccLo => vec![Loc::VccLo],
        Special::VccHi => vec![Loc::VccHi],
        Special::Exec => vec![Loc::ExecLo, Loc::ExecHi],
        Special::ExecLo => vec![Loc::ExecLo],
        Special::ExecHi => vec![Loc::ExecHi],
        Special::Scc => vec![Loc::Scc],
        Special::M0 => vec![Loc::M0],
        Special::Null => Vec::new(),
        Special::Ttmp(n) => vec![Loc::Ttmp(u16::from(n))],
        Special::FlatScratch => vec![Loc::Other(0)],
        Special::SrcSharedBase => vec![Loc::Other(1)],
        Special::SrcSharedLimit => vec![Loc::Other(2)],
        Special::SrcPrivateBase => vec![Loc::Other(3)],
        Special::SrcPrivateLimit => vec![Loc::Other(4)],
        Special::Pc => vec![Loc::Other(5)],
    }
}

fn implicit_locs(bits: u8) -> Vec<Loc> {
    let mut out = Vec::new();
    if bits & ImplicitSet::SCC != 0 {
        out.push(Loc::Scc);
    }
    if bits & ImplicitSet::VCC != 0 {
        out.extend([Loc::VccLo, Loc::VccHi]);
    }
    if bits & ImplicitSet::EXEC != 0 {
        out.extend([Loc::ExecLo, Loc::ExecHi]);
    }
    if bits & ImplicitSet::M0 != 0 {
        out.push(Loc::M0);
    }
    if bits & ImplicitSet::MODE != 0 {
        out.push(Loc::Other(7));
    }
    out
}

/// (uses, defs) of one instruction as dword sets.
fn inst_use_def(inst: &Inst) -> (BTreeSet<Loc>, BTreeSet<Loc>) {
    let mut uses = BTreeSet::new();
    let mut defs = BTreeSet::new();
    for reg in &inst.effects.uses {
        uses.extend(reg_locs(*reg));
    }
    for reg in &inst.effects.defs {
        defs.extend(reg_locs(*reg));
    }
    for operand in &inst.operands {
        match operand {
            Operand::Special(special) => {
                let locs = special_locs(*special);
                uses.extend(locs.iter().copied());
                defs.extend(locs.iter().copied());
            }
            Operand::Half(reg, _) => {
                // True16 halves address one dword; the whole dword is the
                // conservative unit, in both directions.
                let locs = reg_locs(*reg);
                uses.extend(locs.iter().copied());
                defs.extend(locs.iter().copied());
            }
            Operand::Hwreg(_) => {
                uses.insert(Loc::Other(6));
                defs.insert(Loc::Other(6));
            }
            _ => {}
        }
    }
    uses.extend(implicit_locs(inst.effects.implicit.reads));
    defs.extend(implicit_locs(inst.effects.implicit.writes));
    (uses, defs)
}

/// Liveness facts over built blocks: per-block live-in/out plus per-point
/// live-before/after sets for ranges and free-register queries.
#[derive(Clone, Debug, Default)]
pub struct Liveness {
    live_in: BTreeMap<BlockId, BTreeSet<Loc>>,
    live_out: BTreeMap<BlockId, BTreeSet<Loc>>,
    before: HashMap<InstId, BTreeSet<Loc>>,
    after: HashMap<InstId, BTreeSet<Loc>>,
}

/// One defining occurrence and the furthest point its value may be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRange {
    pub reg: RegRef,
    pub from: InstId,
    pub to: InstId,
}

/// A query point: live-before `before`, or the block live-out when `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LivePoint {
    pub block: BlockId,
    pub before: Option<InstId>,
}

impl Liveness {
    /// Backward may-analysis to a fixpoint. Total: needs built blocks, but
    /// any CFG shape (diamonds, loops, unreachable padding) converges.
    pub fn analyze(body: &Body) -> Self {
        let mut use_b: HashMap<BlockId, BTreeSet<Loc>> = HashMap::new();
        let mut def_b: HashMap<BlockId, BTreeSet<Loc>> = HashMap::new();
        let mut inst_facts: HashMap<InstId, (BTreeSet<Loc>, BTreeSet<Loc>)> = HashMap::new();
        for block in &body.blocks {
            let (start, end) = block.range;
            let mut def_accum = BTreeSet::new();
            let mut use_accum = BTreeSet::new();
            // Upward-exposed uses, forward: a use counts only when no earlier
            // instruction in the block defines the location.
            for &id in &body.layout[start..end] {
                let Some(inst) = body.insts.get(id) else {
                    continue;
                };
                let (uses, defs) = inst_use_def(inst);
                inst_facts.insert(id, (uses.clone(), defs.clone()));
                for loc in uses {
                    if !def_accum.contains(&loc) {
                        use_accum.insert(loc);
                    }
                }
                def_accum.extend(defs);
            }
            use_b.insert(block.id, use_accum);
            def_b.insert(block.id, def_accum);
        }

        let mut live_in: BTreeMap<BlockId, BTreeSet<Loc>> = body
            .blocks
            .iter()
            .map(|block| (block.id, BTreeSet::new()))
            .collect();
        let mut live_out: BTreeMap<BlockId, BTreeSet<Loc>> = body
            .blocks
            .iter()
            .map(|block| (block.id, BTreeSet::new()))
            .collect();
        let mut queue: VecDeque<BlockId> = body.blocks.iter().rev().map(|block| block.id).collect();
        let mut queued: BTreeSet<BlockId> = queue.iter().copied().collect();
        while let Some(id) = queue.pop_front() {
            queued.remove(&id);
            let block = &body.blocks[id.0];
            let mut out = BTreeSet::new();
            for &succ in &block.succs {
                if let Some(ins) = live_in.get(&succ) {
                    out.extend(ins.iter().copied());
                }
            }
            let mut inn = use_b.get(&id).cloned().unwrap_or_default();
            for loc in out.iter() {
                if !def_b
                    .get(&id)
                    .map(|defs| defs.contains(loc))
                    .unwrap_or(false)
                {
                    inn.insert(*loc);
                }
            }
            let changed_in = live_in.get(&id) != Some(&inn);
            let changed_out = live_out.get(&id) != Some(&out);
            if changed_in || changed_out {
                live_in.insert(id, inn);
                live_out.insert(id, out);
                for &pred in &block.preds {
                    if queued.insert(pred) {
                        queue.push_back(pred);
                    }
                }
            }
        }

        let mut before = HashMap::new();
        let mut after = HashMap::new();
        for block in &body.blocks {
            let (start, end) = block.range;
            let mut live = live_out.get(&block.id).cloned().unwrap_or_default();
            for &id in body.layout[start..end].iter().rev() {
                after.insert(id, live.clone());
                if let Some((uses, defs)) = inst_facts.get(&id) {
                    for loc in defs {
                        live.remove(loc);
                    }
                    live.extend(uses.iter().copied());
                }
                before.insert(id, live.clone());
            }
        }
        Self {
            live_in,
            live_out,
            before,
            after,
        }
    }

    /// Live-in set of a block (empty when the block is unknown).
    pub fn live_in(&self, block: BlockId) -> &BTreeSet<Loc> {
        self.live_in
            .get(&block)
            .as_ref()
            .map_or(const { &EMPTY }, |set| set)
    }

    /// Live-out set of a block.
    pub fn live_out(&self, block: BlockId) -> &BTreeSet<Loc> {
        self.live_out
            .get(&block)
            .as_ref()
            .map_or(const { &EMPTY }, |set| set)
    }

    /// Live-before set of an instruction, if analysed.
    pub fn live_before(&self, id: InstId) -> Option<&BTreeSet<Loc>> {
        self.before.get(&id)
    }

    /// Live-after set of an instruction, if analysed.
    pub fn live_after(&self, id: InstId) -> Option<&BTreeSet<Loc>> {
        self.after.get(&id)
    }

    /// One range per defining occurrence per dword: `from` is the defining
    /// instruction, `to` the furthest instruction (reachable from `from`)
    /// whose live-before set still holds the dword. Dead defs get `to == from`.
    pub fn ranges(&self, body: &Body) -> Vec<LiveRange> {
        // Reachable blocks per block, over the block graph.
        let mut reachable: HashMap<BlockId, BTreeSet<BlockId>> = HashMap::new();
        for block in &body.blocks {
            let mut seen = BTreeSet::from([block.id]);
            let mut queue = VecDeque::from([block.id]);
            while let Some(id) = queue.pop_front() {
                for &succ in &body.blocks[id.0].succs {
                    if seen.insert(succ) {
                        queue.push_back(succ);
                    }
                }
            }
            reachable.insert(block.id, seen);
        }
        let pos_of: HashMap<InstId, usize> = body
            .layout
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, pos))
            .collect();
        let mut out = Vec::new();
        for &from in &body.layout {
            let Some(after) = self.after.get(&from) else {
                continue;
            };
            let Some(inst) = body.insts.get(from) else {
                continue;
            };
            let (_, defs) = inst_use_def(inst);
            let from_block = containing_block(body, from);
            for loc in defs {
                let (kind, index) = match loc {
                    Loc::V(i) => (Kind::V, i),
                    Loc::S(i) => (Kind::S, i),
                    Loc::Ttmp(i) => (Kind::Ttmp, i),
                    _ => continue,
                };
                let mut to = from;
                let mut best = pos_of.get(&from).copied().unwrap_or(0);
                for (&id, set) in &self.before {
                    if !set.contains(&loc) {
                        continue;
                    }
                    let reachable_ok = match (from_block, containing_block(body, id)) {
                        (Some(fb), Some(tb)) => reachable
                            .get(&fb)
                            .map(|set| set.contains(&tb))
                            .unwrap_or(false),
                        _ => false,
                    };
                    if !reachable_ok {
                        continue;
                    }
                    let pos = pos_of.get(&id).copied().unwrap_or(0);
                    if pos >= best {
                        best = pos;
                        to = id;
                    }
                }
                // A def is always live at least across itself when something
                // downstream reads it; `after` covers the straight case where
                // no later live-before point exists (e.g. use in the same
                // instruction is impossible, so keep `to == from`).
                let _ = after;
                out.push(LiveRange {
                    reg: RegRef {
                        kind,
                        base: index,
                        len: 1,
                    },
                    from,
                    to,
                });
            }
        }
        out
    }

    /// Maximal free `V`/`S`/`Ttmp` ranges at a point: every architectural
    /// dword not live there, coalesced. Specials are never reported free
    /// (they are not allocatable).
    pub fn free_at(&self, _body: &Body, at: LivePoint) -> RegSet {
        let live = match at.before {
            Some(id) => self.before.get(&id),
            None => self.live_out.get(&at.block),
        };
        let mut free: Vec<RegRef> = Vec::new();
        // `RegRef.len` is a u8, so a fully free bank is emitted as 255-long
        // chunks plus a remainder.
        let push_range = |free: &mut Vec<RegRef>, kind: Kind, base: u16, end: u16| {
            let mut cur = base;
            while cur < end {
                let len = (end - cur).min(u16::from(u8::MAX)) as u8;
                free.push(RegRef {
                    kind,
                    base: cur,
                    len,
                });
                cur += u16::from(len);
            }
        };
        for (kind, limit) in [(Kind::V, 256u16), (Kind::S, 106u16), (Kind::Ttmp, 16u16)] {
            let mut start: Option<u16> = None;
            let is_live = |index: u16| {
                let loc = match kind {
                    Kind::V => Loc::V(index),
                    Kind::S => Loc::S(index),
                    Kind::Ttmp => Loc::Ttmp(index),
                };
                live.map(|set| set.contains(&loc)).unwrap_or(false)
            };
            for index in 0..limit {
                if is_live(index) {
                    if let Some(base) = start.take() {
                        push_range(&mut free, kind, base, index);
                    }
                } else if start.is_none() {
                    start = Some(index);
                }
            }
            if let Some(base) = start.take() {
                push_range(&mut free, kind, base, limit);
            }
        }
        RegSet(free)
    }
}

const EMPTY: BTreeSet<Loc> = BTreeSet::new();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Arena, Cond, Terminator};
    use crate::effects::Effects;
    use crate::inst::{Form, Inst};
    use crate::isa;
    use crate::operand::{ImmField, Operand};
    use crate::passes::cfg::{build_blocks, Cfg};
    use crate::provenance::Provenance;

    fn row_op(name: &str, form: Form) -> crate::inst::Opcode {
        isa::gfx12()
            .iter()
            .find(|row| row.name == name && row.form == form)
            .unwrap_or_else(|| panic!("missing table row {name} {form:?}"))
            .op
    }

    fn mk(name: &str, form: Form) -> Inst {
        Inst {
            op: row_op(name, form),
            form,
            fields: Default::default(),
            operands: Default::default(),
            mods: Default::default(),
            literal: None,
            effects: Effects::default(),
            prov: Provenance::default(),
        }
    }

    fn body_of(insts: Vec<Inst>) -> Body {
        let mut arena: Arena<Inst> = Arena::new();
        let mut layout = Vec::new();
        for inst in insts {
            layout.push(arena.insert(inst));
        }
        Body {
            insts: arena,
            blocks: Vec::new(),
            layout,
        }
    }

    fn branch(off: i16) -> Inst {
        let mut inst = mk("s_cbranch_scc1", Form::Sopp);
        inst.effects.control = crate::effects::Control::Branch { cond: Cond::Scc1 };
        inst.operands.push(Operand::Imm(ImmField::Sopp(off)));
        inst.effects.implicit.reads |= ImplicitSet::SCC;
        inst
    }

    fn jump(off: i16) -> Inst {
        let mut inst = mk("s_branch", Form::Sopp);
        inst.effects.control = crate::effects::Control::Jump;
        inst.operands.push(Operand::Imm(ImmField::Sopp(off)));
        inst
    }

    fn endpgm() -> Inst {
        let mut inst = mk("s_endpgm", Form::Sopp);
        inst.effects.control = crate::effects::Control::EndPgm;
        inst
    }

    fn vdef(base: u16) -> Inst {
        let mut inst = mk("v_mov_b32_e32", Form::Vop1);
        inst.effects.defs.push(RegRef {
            kind: Kind::V,
            base,
            len: 1,
        });
        inst
    }

    fn vuse(base: u16) -> Inst {
        let mut inst = mk("v_add_nc_u32_e32", Form::Vop2);
        inst.effects.uses.push(RegRef {
            kind: Kind::V,
            base,
            len: 1,
        });
        inst.effects.defs.push(RegRef {
            kind: Kind::V,
            base: 100,
            len: 1,
        });
        inst
    }

    fn live_set(liveness: &Liveness, block: BlockId) -> Vec<Loc> {
        let mut v: Vec<Loc> = liveness.live_in(block).iter().copied().collect();
        v.sort();
        v
    }

    #[test]
    fn def_reaches_use_across_diamond_join() {
        // True blocks: b0 def+branch, b1 plain, b2 plain (target), b3 join+use.
        let mut body = body_of(vec![
            vdef(5),
            branch(2),
            mk("v_mov_b32_e32", Form::Vop1),
            jump(1),
            mk("v_mov_b32_e32", Form::Vop1),
            vuse(5),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 4);
        let live = Liveness::analyze(&body);
        // v5 is live-in at the join and along both paths (may-analysis).
        assert!(live.live_in(BlockId(3)).contains(&Loc::V(5)));
        assert!(live.live_in(BlockId(1)).contains(&Loc::V(5)));
        assert!(live.live_in(BlockId(2)).contains(&Loc::V(5)));
        assert!(live.live_out(BlockId(0)).contains(&Loc::V(5)));
        assert!(!live.live_in(BlockId(0)).contains(&Loc::V(5)));
        // SCC is live-in at entry: the branch reads it.
        assert!(live.live_in(BlockId(0)).contains(&Loc::Scc));
        let cfg = Cfg::build(&body).unwrap();
        assert!(cfg.dominates(BlockId(0), BlockId(3)));
    }

    #[test]
    fn v200_diamond_stays_live_on_uninitialised_path() {
        // Spec §6.2 scenario at the liveness level: one path defines v200,
        // the other does not; the join's live-in still holds v200 (may), so a
        // later definite-assignment check (C6) must refuse unguarded uses.
        // b0 branch, b1 defines v200, b2 never defines it, b3 is the join.
        let mut def = mk("v_mov_b32_e32", Form::Vop1);
        def.effects.defs.push(RegRef {
            kind: Kind::V,
            base: 200,
            len: 1,
        });
        let mut body = body_of(vec![
            branch(2),
            def,
            jump(1),
            mk("v_mov_b32_e32", Form::Vop1),
            vuse(200),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 4);
        let live = Liveness::analyze(&body);
        assert!(live.live_in(BlockId(3)).contains(&Loc::V(200)));
        // May-merge: v200 is live-out of BOTH predecessors, including the one
        // that never defined it. Liveness cannot tell initialised from live,
        // which is exactly why C6 needs definite assignment on top.
        assert!(live.live_out(BlockId(1)).contains(&Loc::V(200)));
        assert!(live.live_out(BlockId(2)).contains(&Loc::V(200)));
    }

    #[test]
    fn loop_carried_value_stays_live() {
        // True blocks: b0 def+jump, b1 use, b2 branch, b3 body+jump-back.
        // Leaders {0,2,3,4,6}.
        let mut use5 = mk("v_add_nc_u32_e32", Form::Vop2);
        use5.effects.uses.push(RegRef {
            kind: Kind::V,
            base: 5,
            len: 1,
        });
        use5.effects.defs.push(RegRef {
            kind: Kind::V,
            base: 6,
            len: 1,
        });
        let mut body = body_of(vec![
            vdef(5),
            jump(1),
            use5,
            branch(2),
            mk("v_mov_b32_e32", Form::Vop1),
            jump(-4),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 5);
        let live = Liveness::analyze(&body);
        assert!(live.live_in(BlockId(1)).contains(&Loc::V(5)));
        assert!(live.live_in(BlockId(2)).contains(&Loc::V(5)));
        // The def's range spans the loop back to the back-edge jump.
        let ranges = live.ranges(&body);
        let r5 = ranges
            .iter()
            .find(|r| {
                r.reg
                    == RegRef {
                        kind: Kind::V,
                        base: 5,
                        len: 1,
                    }
            })
            .unwrap();
        assert_eq!(r5.from, body.layout[0]);
        assert_eq!(r5.to, body.layout[5]);
    }

    #[test]
    fn implicit_exec_write_is_a_def_and_scc_read_is_live() {
        // v_cmpx-like: writes EXEC, no explicit defs. Downstream predicated
        // use reads EXEC implicitly.
        let mut cmpx = mk("v_cmpx_gt_f32_e32", Form::Vopc);
        cmpx.effects.implicit.writes |= ImplicitSet::EXEC;
        let mut predicated = mk("v_mov_b32_e32", Form::Vop1);
        predicated.effects.implicit.reads |= ImplicitSet::EXEC;
        predicated.effects.defs.push(RegRef {
            kind: Kind::V,
            base: 7,
            len: 1,
        });
        let mut body = body_of(vec![cmpx, predicated, endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let live = Liveness::analyze(&body);
        assert!(!live.live_in(BlockId(0)).contains(&Loc::ExecLo));
        let after_cmpx = live.live_after(body.layout[0]).unwrap();
        assert!(after_cmpx.contains(&Loc::ExecLo) && after_cmpx.contains(&Loc::ExecHi));
    }

    #[test]
    fn free_at_reports_coalesced_dead_ranges() {
        // v0 live across the cursor, everything else free.
        let mut body = body_of(vec![
            vdef(0),
            mk("v_mov_b32_e32", Form::Vop1),
            vuse(0),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let live = Liveness::analyze(&body);
        let at = LivePoint {
            block: BlockId(0),
            before: Some(body.layout[1]),
        };
        let free = live.free_at(&body, at);
        let v_ranges: Vec<_> = free
            .0
            .iter()
            .filter(|r| r.kind == Kind::V)
            .copied()
            .collect();
        assert_eq!(
            v_ranges,
            vec![RegRef {
                kind: Kind::V,
                base: 1,
                len: 255
            }]
        );
        // Past the use, v0 is free again.
        let at_end = LivePoint {
            block: BlockId(0),
            before: Some(body.layout[3]),
        };
        let free_end = live.free_at(&body, at_end);
        assert!(free_end.0.iter().any(|r| r.kind == Kind::V && r.base == 0));
    }

    #[test]
    fn redefinition_kills_liveness() {
        // def v1; use v1 (kills the first def's range); def v1 again; endpgm.
        // v1 must not be live-in at entry.
        let mut body = body_of(vec![vdef(1), vuse(1), vdef(1), endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let live = Liveness::analyze(&body);
        assert!(!live.live_in(BlockId(0)).contains(&Loc::V(1)));
        // The first def's range ends at the use; the last def is dead.
        let ranges = live.ranges(&body);
        let first = ranges
            .iter()
            .find(|r| r.from == body.layout[0] && r.reg.base == 1)
            .unwrap();
        assert_eq!(first.to, body.layout[1]);
        let last = ranges
            .iter()
            .find(|r| r.from == body.layout[2] && r.reg.base == 1)
            .unwrap();
        assert_eq!(last.to, body.layout[2]);
        let _ = Terminator::FallThrough;
    }
}
