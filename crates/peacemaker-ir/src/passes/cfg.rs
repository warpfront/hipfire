// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C4: branch target leaders, CFG construction, dominators and loops.
//!
//! Input contract (shared with the C3 decoder): `build_blocks` takes a `Body`
//! whose `insts` arena and `layout` order are filled and whose `blocks` are
//! still empty. Branch instructions carry their target as a raw
//! `Operand::Imm(ImmField::Sopp)` dword offset from the next PC (signed,
//! decoded from the encoding, never from text); the pass validates the target,
//! splits leaders into blocks, and rewrites the offset to
//! `Operand::Label(BlockId)`. Which instructions branch is read from
//! `Effects.control` (`Branch`/`Jump`/`EndPgm`); this pass knows no opcode ids.
//!
//! PC model: dword offsets from the start of `layout`, using per-target form
//! widths and one more dword when a literal follows. The pinned ISA tables
//! carry sample words for both gfx11 and gfx12 (e.g. gfx1100
//! `buffer_load_b64` is two dwords; gfx1201 VMEM is three).

use std::collections::{BTreeSet, HashMap, VecDeque};

use petgraph::algo::dominators::{self, Dominators};
use petgraph::algo::tarjan_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use smallvec::SmallVec;
use thiserror::Error;

use crate::cfg::{Block, BlockId, Body, InstId, Terminator};
use crate::effects::Control;
use crate::inst::{Form, Inst};
use crate::operand::{ImmField, Operand};

/// Base dword width of one instruction word stream without a literal dword.
///
/// Grounded in the per-architecture ISA tables' encoding samples.
pub fn base_dwords(form: Form, arch: crate::inst::Arch) -> usize {
    match form {
        Form::Sop1 | Form::Sop2 | Form::Sopc | Form::Sopk | Form::Sopp => 1,
        Form::Vop1 | Form::Vop2 | Form::Vopc => 1,
        Form::Smem => 2,
        Form::Vop1Dpp | Form::Vop2Dpp | Form::Vop3 | Form::Vop3p | Form::Vopd | Form::Vinterp => 2,
        Form::Ds => 2,
        Form::Vmem(_) => {
            if arch == crate::inst::Arch::Gfx1201 {
                3
            } else {
                2
            }
        }
        // No M1 table rows yet; provisional until the forms are tabled.
        Form::Export => 2,
    }
}

/// Dword width of a decoded instruction, including any literal.
pub fn dwords_of(inst: &Inst, arch: crate::inst::Arch) -> usize {
    base_dwords(inst.form, arch) + usize::from(inst.literal.is_some())
}

/// Dword PCs per layout position plus per-instruction widths, as computed by
/// [`build_blocks`]. PCs are relative to the start of `layout` (branch
/// offsets are PC-relative, so no VA is needed). `entry_target_leaders` is
/// the sorted entry-plus-branch-target subset of leaders (fall-through-only
/// leaders excluded); on KT48 it must be hipcc's 57 labels plus the entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutInfo {
    pub pcs: Vec<u32>,
    pub dwords: Vec<usize>,
    pub entry_target_leaders: Vec<usize>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CfgError {
    #[error("layout is empty")]
    Empty,
    #[error("blocks are already built")]
    AlreadyBuilt,
    #[error("layout refers to a tombstoned or missing instruction")]
    DanglingInst { id: InstId },
    #[error("branch at layout index {index} has no SOPP offset operand")]
    MissingOffset { index: usize },
    #[error("branch at layout index {index} already carries a resolved label")]
    UnexpectedLabel { index: usize },
    #[error("branch at layout index {index} targets pc {target}, outside the kernel")]
    TargetOutsideKernel { index: usize, target: i64 },
    #[error("branch at layout index {index} targets pc {target}, mid-instruction")]
    TargetMidInstruction { index: usize, target: u32 },
    #[error("kernel does not end in s_endpgm (last block falls through)")]
    MissingEndPgm,
    #[error("body has no blocks; run build_blocks first")]
    BlocksNotBuilt,
}

/// Raw SOPP branch offset of a branch/jump instruction.
fn branch_offset(inst: &Inst, index: usize) -> Result<i16, CfgError> {
    let mut found: Option<i16> = None;
    for op in &inst.operands {
        match op {
            Operand::Imm(ImmField::Sopp(off)) => {
                if found.is_some() {
                    return Err(CfgError::MissingOffset { index });
                }
                found = Some(*off);
            }
            Operand::Label(_) => return Err(CfgError::UnexpectedLabel { index }),
            _ => {}
        }
    }
    found.ok_or(CfgError::MissingOffset { index })
}

/// Block containing `id`, by linear scan of block ranges.
pub fn containing_block(body: &Body, id: InstId) -> Option<BlockId> {
    body.blocks.iter().find_map(|block| {
        let (start, end) = block.range;
        let pos = body.layout[start..end].iter().position(|&x| x == id)?;
        let _ = pos;
        Some(block.id)
    })
}

/// Build `body.blocks` (plus preds/succs/terminators) from typed branch
/// targets and rewrite branch offsets to `Operand::Label`.
pub fn build_blocks(body: &mut Body, arch: crate::inst::Arch) -> Result<LayoutInfo, CfgError> {
    if !body.blocks.is_empty() {
        return Err(CfgError::AlreadyBuilt);
    }
    let n = body.layout.len();
    if n == 0 {
        return Err(CfgError::Empty);
    }
    let mut dwords = Vec::with_capacity(n);
    for id in &body.layout {
        let inst = body
            .insts
            .get(*id)
            .ok_or(CfgError::DanglingInst { id: *id })?;
        dwords.push(dwords_of(inst, arch));
    }
    let mut pcs = vec![0u32; n];
    let mut pc = 0u32;
    for i in 0..n {
        pcs[i] = pc;
        pc += dwords[i] as u32;
    }
    let total = pc;
    let mut pos_of_pc: HashMap<u32, usize> = HashMap::with_capacity(n);
    for (i, &pc) in pcs.iter().enumerate() {
        pos_of_pc.insert(pc, i);
    }

    enum End {
        Branch { target_pos: usize },
        Jump { target_pos: usize },
        Terminating,
    }
    let mut ends: Vec<Option<End>> = Vec::with_capacity(n);
    for _ in 0..n {
        ends.push(None);
    }
    for (i, id) in body.layout.iter().enumerate() {
        let inst = body
            .insts
            .get(*id)
            .ok_or(CfgError::DanglingInst { id: *id })?;
        match inst.effects.control {
            Control::Branch { .. } | Control::Jump => {
                let off = branch_offset(inst, i)?;
                let target = pcs[i] as i64 + dwords[i] as i64 + off as i64;
                if target < 0 || target >= total as i64 {
                    return Err(CfgError::TargetOutsideKernel { index: i, target });
                }
                let target = target as u32;
                let target_pos = pos_of_pc
                    .get(&target)
                    .copied()
                    .ok_or(CfgError::TargetMidInstruction { index: i, target })?;
                ends[i] = Some(if matches!(inst.effects.control, Control::Jump) {
                    End::Jump { target_pos }
                } else {
                    End::Branch { target_pos }
                });
            }
            Control::EndPgm | Control::Halt | Control::Trap => {
                ends[i] = Some(End::Terminating);
            }
            _ => {}
        }
    }

    // Leaders: true basic blocks. Every branch (conditional or not) ends its
    // block, so leaders are the entry, every branch target, and the
    // fall-through after every branch, jump, or terminator. (Halt/Trap
    // terminate the path like s_endpgm; see the Terminator mapping below.)
    // `entry_target_leaders` records the entry-plus-target subset separately:
    // on KT48 it must equal hipcc's 57 labels plus the unlabeled entry (the
    // T4 sub-assertion; C7 owns the full T4).
    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);
    let mut entry_target_leaders = BTreeSet::new();
    entry_target_leaders.insert(0usize);
    for (i, end) in ends.iter().enumerate() {
        match end {
            Some(End::Branch { target_pos } | End::Jump { target_pos }) => {
                leaders.insert(*target_pos);
                entry_target_leaders.insert(*target_pos);
                if i + 1 < n {
                    leaders.insert(i + 1);
                }
            }
            Some(End::Terminating) => {
                if i + 1 < n {
                    leaders.insert(i + 1);
                }
            }
            None => {}
        }
    }
    let leader_list: Vec<usize> = leaders.into_iter().collect();
    let pos_block: Vec<usize> = {
        let mut map = vec![0usize; n];
        for (block_idx, window) in leader_list.windows(2).enumerate() {
            for pos in window[0]..window[1] {
                map[pos] = block_idx;
            }
        }
        let last = leader_list.len() - 1;
        for pos in leader_list[last]..n {
            map[pos] = last;
        }
        map
    };
    let block_count = leader_list.len();

    // Stage terms and edges locally; commit to `body` only on success so a
    // MissingEndPgm never leaves a half-built CFG behind. Leaders guarantee
    // the only control instruction in a block is its final one, so the term
    // names it directly and each block has at most two successors.
    #[derive(Clone)]
    enum Term {
        Branch { taken: usize },
        Jump { taken: usize },
        EndPgm,
        FallThrough { next: Option<usize> },
    }
    let mut terms: Vec<Term> = vec![Term::FallThrough { next: None }; block_count];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); block_count];
    for block_idx in 0..block_count {
        let end = if block_idx + 1 < block_count {
            leader_list[block_idx + 1]
        } else {
            n
        };
        let next = if block_idx + 1 < block_count {
            Some(block_idx + 1)
        } else {
            None
        };
        match &ends[end - 1] {
            Some(End::Jump { target_pos }) => {
                let taken = pos_block[*target_pos];
                terms[block_idx] = Term::Jump { taken };
                succs[block_idx].push(taken);
            }
            Some(End::Terminating) => {
                terms[block_idx] = Term::EndPgm;
            }
            Some(End::Branch { target_pos }) => {
                let taken = pos_block[*target_pos];
                let Some(next) = next else {
                    return Err(CfgError::MissingEndPgm);
                };
                terms[block_idx] = Term::Branch { taken };
                succs[block_idx].push(taken);
                succs[block_idx].push(next);
            }
            None => {
                // Last-block fall-through is tentative: endpgm padding stays
                // Unreachable, while a reachable open tail is MissingEndPgm
                // (checked after reachability below).
                let Some(next) = next else {
                    terms[block_idx] = Term::FallThrough { next: None };
                    continue;
                };
                terms[block_idx] = Term::FallThrough { next: Some(next) };
                succs[block_idx].push(next);
            }
        }
        succs[block_idx].sort_unstable();
        succs[block_idx].dedup();
    }
    // Reachability from entry; unreachable blocks (endpgm padding) become
    // Unreachable with no successors.
    let mut reachable = vec![false; block_count];
    let mut queue = VecDeque::from([0usize]);
    reachable[0] = true;
    while let Some(b) = queue.pop_front() {
        for &s in &succs[b] {
            if !reachable[s] {
                reachable[s] = true;
                queue.push_back(s);
            }
        }
    }
    if matches!(terms[block_count - 1], Term::FallThrough { next: None })
        && reachable[block_count - 1]
    {
        return Err(CfgError::MissingEndPgm);
    }

    let mut blocks: Vec<Block> = Vec::with_capacity(block_count);
    for (block_idx, &start) in leader_list.iter().enumerate() {
        let end = if block_idx + 1 < block_count {
            leader_list[block_idx + 1]
        } else {
            n
        };
        let term = if !reachable[block_idx] {
            Terminator::Unreachable
        } else {
            match &terms[block_idx] {
                Term::Branch { taken } => {
                    let cond =
                        match body.insts.get(body.layout[end - 1]).and_then(|inst| {
                            match inst.effects.control {
                                Control::Branch { cond } => Some(cond),
                                _ => None,
                            }
                        }) {
                            Some(cond) => cond,
                            None => {
                                return Err(CfgError::MissingOffset { index: end - 1 });
                            }
                        };
                    Terminator::Branch {
                        cond,
                        taken: BlockId(*taken),
                        fallthrough: BlockId(block_idx + 1),
                    }
                }
                Term::Jump { taken } => Terminator::Jump(BlockId(*taken)),
                Term::EndPgm => Terminator::EndPgm,
                Term::FallThrough { .. } => Terminator::FallThrough,
            }
        };
        blocks.push(Block {
            id: BlockId(block_idx),
            range: (start, end),
            term,
            preds: SmallVec::new(),
            succs: SmallVec::new(),
        });
    }
    for (block_idx, succ_list) in succs.iter().enumerate() {
        if !reachable[block_idx] {
            continue;
        }
        for &s in succ_list {
            if reachable[s] {
                blocks[block_idx].succs.push(BlockId(s));
                blocks[s].preds.push(BlockId(block_idx));
            }
        }
    }

    // Rewrite branch offsets to labels. Unreachable blocks keep consistent
    // labels too; their edges are simply absent above.
    for (i, id) in body.layout.iter().enumerate() {
        let taken_block = match &ends[i] {
            Some(End::Branch { target_pos } | End::Jump { target_pos }) => {
                Some(BlockId(pos_block[*target_pos]))
            }
            _ => None,
        };
        if let Some(taken) = taken_block {
            let inst = body
                .insts
                .get_mut(*id)
                .ok_or(CfgError::DanglingInst { id: *id })?;
            for op in inst.operands.iter_mut() {
                if matches!(op, Operand::Imm(ImmField::Sopp(_))) {
                    *op = Operand::Label(taken);
                    break;
                }
            }
        }
    }
    body.blocks = blocks;
    Ok(LayoutInfo {
        pcs,
        dwords,
        entry_target_leaders: entry_target_leaders.into_iter().collect(),
    })
}

/// Layout positions that are branch targets, derived from rewritten
/// `Operand::Label` operands (call after [`build_blocks`]). Used by the
/// window passes to distinguish a branch-target crossing (reject) from a
/// mere fall-through split (straight-line, allowed).
pub fn branch_target_positions(body: &Body) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for id in &body.layout {
        let Some(inst) = body.insts.get(*id) else {
            continue;
        };
        for op in &inst.operands {
            if let Operand::Label(target) = op {
                out.insert(body.blocks[target.0].range.0);
            }
        }
    }
    out
}

/// CFG analysis over built blocks: dominators, post-dominators, loop nesting.
pub struct Cfg {
    graph: DiGraph<BlockId, ()>,
    node: HashMap<BlockId, NodeIndex>,
    rev: DiGraph<BlockId, ()>,
    rev_node: HashMap<BlockId, NodeIndex>,
    rev_exit: Option<NodeIndex>,
    /// Blocks in layout order.
    pub entry: BlockId,
    pub exits: Vec<BlockId>,
    dom: Dominators<NodeIndex>,
    postdom: Option<Dominators<NodeIndex>>,
    loops: Vec<LoopInfo>,
}

/// One natural loop: `header` dominates every member and is the target of the
/// back edge. `members` is sorted by `BlockId`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopInfo {
    pub header: BlockId,
    pub members: Vec<BlockId>,
}

impl Cfg {
    /// Analyse built blocks. Errors only when blocks are missing or no exit
    /// block exists for the post-dominator tree (endless kernels); dominators
    /// and loops are always computed.
    pub fn build(body: &Body) -> Result<Self, CfgError> {
        if body.blocks.is_empty() {
            return Err(CfgError::BlocksNotBuilt);
        }
        let mut graph = DiGraph::new();
        let mut node = HashMap::new();
        for block in &body.blocks {
            node.insert(block.id, graph.add_node(block.id));
        }
        for block in &body.blocks {
            let from = node[&block.id];
            for &succ in &block.succs {
                graph.add_edge(from, node[&succ], ());
            }
        }
        let entry = BlockId(0);
        let entry_node = node[&entry];
        let dom = dominators::simple_fast(&graph, entry_node);
        let exits: Vec<BlockId> = body
            .blocks
            .iter()
            .filter(|block| matches!(block.term, Terminator::EndPgm))
            .map(|block| block.id)
            .collect();

        // Post-dominators on the reversed graph with one synthetic exit fed
        // by every EndPgm block.
        let mut rev = DiGraph::new();
        let mut rev_node = HashMap::new();
        for block in &body.blocks {
            rev_node.insert(block.id, rev.add_node(block.id));
        }
        for block in &body.blocks {
            for &succ in &block.succs {
                rev.add_edge(rev_node[&succ], rev_node[&block.id], ());
            }
        }
        let (rev_exit, postdom) = if exits.is_empty() {
            (None, None)
        } else {
            let exit_node = rev.add_node(BlockId(usize::MAX));
            for &exit in &exits {
                rev.add_edge(exit_node, rev_node[&exit], ());
            }
            // simple_fast gives predecessors-of-start an empty set; reverse
            // the direction back by querying on the reversed graph directly.
            let synthetic = rev.add_node(BlockId(usize::MAX - 1));
            rev.add_edge(synthetic, exit_node, ());
            let pd = dominators::simple_fast(&rev, synthetic);
            let _ = exit_node;
            (Some(synthetic), Some(pd))
        };

        // Natural loops from SCCs: multi-block SCCs, or a single block with a
        // self edge. The header is the member with a predecessor outside the
        // SCC that dominates the rest; fall back to the smallest BlockId.
        let sccs = tarjan_scc(&graph);
        let mut loops = Vec::new();
        for scc in sccs {
            let members: Vec<BlockId> = {
                let mut ids: Vec<BlockId> = scc.iter().map(|&n| graph[n]).collect();
                ids.sort();
                ids
            };
            let is_loop = members.len() > 1
                || members
                    .iter()
                    .any(|id| graph.contains_edge(node[id], node[id]));
            if !is_loop {
                continue;
            }
            let in_set: BTreeSet<BlockId> = members.iter().copied().collect();
            let mut header = members[0];
            for &candidate in &members {
                let has_outside_pred = body
                    .blocks
                    .iter()
                    .find(|block| block.id == candidate)
                    .map(|block| block.preds.iter().any(|pred| !in_set.contains(pred)))
                    .unwrap_or(false);
                if has_outside_pred {
                    header = candidate;
                    break;
                }
            }
            loops.push(LoopInfo { header, members });
        }
        loops.sort_by_key(|info| info.header);

        Ok(Self {
            graph,
            node,
            rev,
            rev_node,
            rev_exit,
            entry,
            exits,
            dom,
            postdom,
            loops,
        })
    }

    fn node(&self, id: BlockId) -> Option<NodeIndex> {
        self.node.get(&id).copied()
    }

    /// True when `a` dominates `b` (every path from entry through `b` visits
    /// `a`; a block dominates itself).
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        match (self.node(a), self.node(b)) {
            (Some(na), Some(nb)) => self
                .dom
                .dominators(nb)
                .map(|mut it| it.any(|n| n == na))
                .unwrap_or(false),
            _ => false,
        }
    }

    /// True when `a` post-dominates `b`. False for every pair when the kernel
    /// has no exit block.
    pub fn post_dominates(&self, a: BlockId, b: BlockId) -> bool {
        let (Some(pd), Some(ra), Some(rb)) = (
            self.postdom.as_ref(),
            self.rev_node.get(&a).copied(),
            self.rev_node.get(&b).copied(),
        ) else {
            return false;
        };
        pd.dominators(rb)
            .map(|mut it| it.any(|n| n == ra))
            .unwrap_or(false)
    }

    /// Natural loops found by SCC; empty for straight-line and diamond CFGs.
    pub fn loops(&self) -> &[LoopInfo] {
        &self.loops
    }

    /// Number of loops containing `id`.
    pub fn loop_depth(&self, id: BlockId) -> usize {
        self.loops
            .iter()
            .filter(|info| info.members.contains(&id))
            .count()
    }

    /// True when the block is reachable from entry.
    pub fn is_reachable(&self, id: BlockId) -> bool {
        self.node(id).and_then(|n| self.dom.dominators(n)).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Arena, Cond};
    use crate::effects::{Effects, MemClass, MemEffect, OrderType, SrcRead};
    use crate::inst::{Arch, Opcode};
    use crate::isa;
    use crate::operand::Operand;
    use crate::provenance::Provenance;
    use crate::reg::{Kind, RegRef};
    use crate::wait::CounterSet;

    fn row(name: &str, form: Form) -> (Opcode, &'static str) {
        let row = isa::gfx12()
            .iter()
            .find(|row| row.name == name && row.form == form)
            .unwrap_or_else(|| panic!("missing table row {name} {form:?}"));
        (row.op, row.name)
    }

    fn inst(name: &str, form: Form, control: Control, operands: Vec<Operand>) -> Inst {
        let (op, _) = row(name, form);
        Inst {
            op,
            form,
            fields: Default::default(),
            operands: operands.into_iter().collect(),
            mods: Default::default(),
            literal: None,
            effects: Effects {
                control,
                ..Default::default()
            },
            prov: Provenance::default(),
        }
    }

    fn plain(name: &str, form: Form) -> Inst {
        inst(name, form, Control::None, Vec::new())
    }

    fn mem_inst(name: &str, form: Form, class: MemClass) -> Inst {
        let (op, _) = row(name, form);
        Inst {
            op,
            form,
            fields: Default::default(),
            operands: Default::default(),
            mods: Default::default(),
            literal: None,
            effects: Effects {
                mem: Some(MemEffect {
                    class,
                    counters: CounterSet(0),
                    in_order_type: OrderType::Load,
                    src_read: SrcRead::AtIssue,
                }),
                ..Default::default()
            },
            prov: Provenance::default(),
        }
    }

    fn reg_def(inst: &mut Inst, base: u16) {
        inst.effects.defs.push(RegRef {
            kind: Kind::V,
            base,
            len: 1,
        });
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

    fn sopp_branch(name: &str, cond: Option<Cond>, off: i16) -> Inst {
        let control = match cond {
            Some(cond) => Control::Branch { cond },
            None => Control::Jump,
        };
        inst(
            name,
            Form::Sopp,
            control,
            vec![Operand::Imm(ImmField::Sopp(off))],
        )
    }

    #[test]
    fn table_widths_match_form_bases() {
        // AMD MR-ISA sample dwords (gfx1100/gfx1151) and pinned llvm-mc
        // samples (gfx1201): base words, or base+1 with a literal. No gfx11
        // MIMG NSA row is tabled; an unseen variable-length MIMG fails decode.
        for arch in [
            crate::inst::Arch::Gfx1100,
            crate::inst::Arch::Gfx1151,
            crate::inst::Arch::Gfx1201,
        ] {
            for row in isa::table(arch) {
                let words = row.encoding.split_whitespace().count();
                let base = base_dwords(row.form, arch);
                assert!(
                    words == base || words == base + 1,
                    "{} {:?} {arch:?}: {words} words vs base {base}",
                    row.name,
                    row.form
                );
                if words == base + 1 {
                    assert!(
                        row.grammar.contains("literal@last"),
                        "{} {:?} {arch:?}: {words} words without a literal marker",
                        row.name,
                        row.form
                    );
                }
            }
        }
    }

    #[test]
    fn linear_kernel_is_one_endpgm_block() {
        let mut body = body_of(vec![
            plain("v_mov_b32_e32", Form::Vop1),
            plain("s_wait_dscnt", Form::Sopp),
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
        ]);
        let info = build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(info.pcs, vec![0, 1, 2]);
        assert_eq!(body.blocks.len(), 1);
        assert_eq!(body.blocks[0].term, Terminator::EndPgm);
        assert!(body.blocks[0].preds.is_empty() && body.blocks[0].succs.is_empty());
        let cfg = Cfg::build(&body).unwrap();
        assert!(cfg.exits == vec![BlockId(0)]);
        assert!(cfg.dominates(BlockId(0), BlockId(0)));
        assert!(cfg.loops().is_empty());
    }

    #[test]
    fn diamond_splits_four_true_blocks_with_typed_targets() {
        // True basic blocks: every branch ends its block, so the conditional
        // fall-through starts a new one. Leaders {0,2,4,5}.
        // mov@0 cbr@1(taken mul@4) add@2 jmp@3(to sub@5) mul@4 sub@5 endpgm@6
        let mut body = body_of(vec![
            plain("v_mov_b32_e32", Form::Vop1),
            sopp_branch("s_cbranch_scc1", Some(Cond::Scc1), 2),
            plain("v_add_nc_u32_e32", Form::Vop2),
            sopp_branch("s_branch", None, 1),
            plain("v_mul_u32_u24_e32", Form::Vop2),
            plain("v_sub_nc_u32_e32", Form::Vop2),
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
        ]);
        let info = build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        // Entry-plus-target leaders exclude the fall-through-only leader @2.
        assert_eq!(info.entry_target_leaders, vec![0, 4, 5]);
        assert_eq!(body.blocks.len(), 4);
        let terms: Vec<_> = body.blocks.iter().map(|b| &b.term).collect();
        assert!(matches!(
            terms[0],
            Terminator::Branch { cond: Cond::Scc1, taken, fallthrough }
            if *taken == BlockId(2) && *fallthrough == BlockId(1)
        ));
        assert!(matches!(terms[1], Terminator::Jump(t) if *t == BlockId(3)));
        assert!(matches!(terms[2], Terminator::FallThrough));
        assert!(matches!(terms[3], Terminator::EndPgm));
        // Offsets are rewritten to labels.
        for id in &body.layout {
            for op in &body.insts.get(*id).unwrap().operands {
                assert!(
                    !matches!(op, Operand::Imm(ImmField::Sopp(_))),
                    "offset survived"
                );
            }
        }
        assert_eq!(body.blocks[3].preds.len(), 2);
        let cfg = Cfg::build(&body).unwrap();
        assert!(cfg.dominates(BlockId(0), BlockId(3)));
        assert!(!cfg.dominates(BlockId(1), BlockId(3)));
        assert!(!cfg.dominates(BlockId(2), BlockId(3)));
        assert!(cfg.post_dominates(BlockId(3), BlockId(0)));
        assert!(cfg.post_dominates(BlockId(3), BlockId(1)));
        assert!(!cfg.post_dominates(BlockId(1), BlockId(0)));
        assert!(cfg.loops().is_empty());
        assert_eq!(cfg.loop_depth(BlockId(1)), 0);
    }

    #[test]
    fn loop_backedge_forms_one_loop_with_signed_offset() {
        // True blocks split at the conditional fall-through. Leaders {0,2,4,6}.
        // mov@0 jmp@1(->H@2) add@2 cbr@3(taken exit@6) mul@4 jmp@5(->H@2, off -4) endpgm@6
        let mut body = body_of(vec![
            plain("v_mov_b32_e32", Form::Vop1),
            sopp_branch("s_branch", None, 0),
            plain("v_add_nc_u32_e32", Form::Vop2),
            sopp_branch("s_cbranch_scc1", Some(Cond::Scc1), 2),
            plain("v_mul_u32_u24_e32", Form::Vop2),
            sopp_branch("s_branch", None, -4),
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 4);
        assert!(matches!(body.blocks[2].term, Terminator::Jump(t) if t == BlockId(1)));
        let cfg = Cfg::build(&body).unwrap();
        assert_eq!(cfg.loops().len(), 1);
        assert_eq!(cfg.loops()[0].header, BlockId(1));
        assert_eq!(cfg.loops()[0].members, vec![BlockId(1), BlockId(2)]);
        assert_eq!(cfg.loop_depth(BlockId(2)), 1);
        assert_eq!(cfg.loop_depth(BlockId(3)), 0);
        assert!(cfg.dominates(BlockId(1), BlockId(2)));
    }

    #[test]
    fn multi_block_loop_nests_header_and_body() {
        // Leaders {0,1,2,3,5}: the conditional fall-through splits the header
        // from the body entry, so the SCC spans three blocks.
        // jmp@0(->H@1) H:cbr@1(taken exit@5) jmp@2(->body@3) mul@3 jmp@4(->H@1) endpgm@5
        let mut body = body_of(vec![
            sopp_branch("s_branch", None, 0),
            sopp_branch("s_cbranch_scc1", Some(Cond::Scc1), 3),
            sopp_branch("s_branch", None, 0),
            plain("v_mul_u32_u24_e32", Form::Vop2),
            sopp_branch("s_branch", None, -4),
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 5);
        let cfg = Cfg::build(&body).unwrap();
        assert_eq!(cfg.loops().len(), 1);
        assert_eq!(cfg.loops()[0].header, BlockId(1));
        assert_eq!(
            cfg.loops()[0].members,
            vec![BlockId(1), BlockId(2), BlockId(3)]
        );
        assert!(cfg.dominates(BlockId(1), BlockId(3)));
        assert!(!cfg.dominates(BlockId(3), BlockId(1)));
    }

    #[test]
    fn rejects_branch_to_mid_instruction_and_past_end() {
        // vop3 is 2 dwords: pcs 0, 2, 3. Branch at 0 with off 0 targets pc 2
        // (ok); off 1 targets pc 3... build a mid-instruction target instead:
        // branch@0 (next pc 1) off 1 -> pc 2, but instruction at pc 1 is the
        // 2-dword vop3 spanning pcs 1..3? Layout pcs depend on widths; force a
        // mid target: [s_branch(off 1), vop3(2 dwords), endpgm]: pcs 0,1,3.
        // off 1 from next-pc 1 -> pc 2, which is inside the vop3.
        let (vop3op, _) = row("v_add_co_u32", Form::Vop3);
        let mut wide = plain("v_mov_b32_e32", Form::Vop1);
        wide.op = vop3op;
        wide.form = Form::Vop3;
        let mut body = body_of(vec![
            sopp_branch("s_branch", None, 1),
            wide,
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
        ]);
        assert_eq!(
            build_blocks(&mut body, crate::inst::Arch::Gfx1201),
            Err(CfgError::TargetMidInstruction {
                index: 0,
                target: 2
            })
        );

        let mut body = body_of(vec![
            sopp_branch("s_branch", None, 90),
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
        ]);
        assert_eq!(
            build_blocks(&mut body, crate::inst::Arch::Gfx1201),
            Err(CfgError::TargetOutsideKernel {
                index: 0,
                target: 91
            })
        );
    }

    #[test]
    fn rejects_missing_endpgm_and_marks_padding_unreachable() {
        let mut body = body_of(vec![
            plain("v_mov_b32_e32", Form::Vop1),
            plain("v_add_nc_u32_e32", Form::Vop2),
        ]);
        assert_eq!(
            build_blocks(&mut body, crate::inst::Arch::Gfx1201),
            Err(CfgError::MissingEndPgm)
        );

        // Padding after s_endpgm becomes Unreachable with no successors.
        let mut body = body_of(vec![
            inst("s_endpgm", Form::Sopp, Control::EndPgm, Vec::new()),
            plain("v_mov_b32_e32", Form::Vop1),
            plain("v_add_nc_u32_e32", Form::Vop2),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 2);
        assert_eq!(body.blocks[0].term, Terminator::EndPgm);
        assert_eq!(body.blocks[1].term, Terminator::Unreachable);
        assert!(body.blocks[1].succs.is_empty() && body.blocks[1].preds.is_empty());
        let cfg = Cfg::build(&body).unwrap();
        assert!(!cfg.is_reachable(BlockId(1)));
    }

    #[test]
    fn opcode_names_resolve_for_test_rows() {
        // Guards the test-only assumption that the rows used above exist.
        for (name, form) in [
            ("v_mov_b32_e32", Form::Vop1),
            ("v_add_nc_u32_e32", Form::Vop2),
            ("v_mul_u32_u24_e32", Form::Vop2),
            ("v_sub_nc_u32_e32", Form::Vop2),
            ("s_cbranch_scc1", Form::Sopp),
            ("s_branch", Form::Sopp),
            ("s_endpgm", Form::Sopp),
            ("s_wait_dscnt", Form::Sopp),
            ("v_add_co_u32", Form::Vop3),
        ] {
            let (op, _) = row(name, form);
            assert_eq!(op.name(Arch::Gfx1201), Some(name));
        }
        let _ = reg_def as fn(&mut Inst, u16);
        let _ = mem_inst as fn(&str, Form, MemClass) -> Inst;
    }
}
