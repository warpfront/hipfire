// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C4: clause windows and path-sensitive delay-ALU facts.
//!
//! Clause rule (core.md §2.8, §5.2): an `s_clause N` window is the N+1
//! following layout instructions. Every member must live in the opener's
//! block (a window crossing a leader is a lift rejection), be a memory
//! instruction of one class (widths may mix: `s_load_b256` + `s_load_b128`
//! share `SmemLoad`), and contain no wait and no `s_endpgm`.
//!
//! Delay rule (RDNA4 ISA §5.8 + §16.5, `rdna4-instruction-set-architecture.pdf`
//! pp56/269, confirmed against pinned `llvm-mc`/`llvm-dis` sweeps):
//! `INSTID0 = SIMM16[3:0]` names the hazard for the next instruction,
//! `INSTSKIP = SIMM16[6:4]` selects the instruction carrying the second
//! dependency (`SAME` = same next instruction, `NEXT` = the one after,
//! `SKIP_N` = N further), `INSTID1 = SIMM16[10:7]` names its hazard. Each
//! INSTID counts backwards on the executed path (branched-over instructions do
//! not count; EXEC-skipped VALU do): `VALU_DEP_n` counts non-TRANS VALUs,
//! `TRANS32_DEP_n` TRANS ones (LLVM `AMDGPUInsertDelayAlu`), from the
//! instruction it applies to: INSTID0 from the next instruction, INSTID1 from
//! the one INSTSKIP selects, so VALUs between the two consumers count for
//! INSTID1 (RDNA4 §16.5 p270 example: `SKIP_1` + `VALU_DEP_1` names the
//! skipped `v_sub_f32 v11`, which `v_mul_f32 v10, v13, v11` reads). Values:
//! 0 = NO_DEP, 1-4 = VALU_DEP_n, 5-7 = TRANS32_DEP_n, 8 = FMA_ACCUM_CYCLE_1
//! (reserved), 9-11 = SALU_CYCLE_1..3, 12-15 invalid. This pass assumes C3
//! stores those raw field values in `DelayAluHint { instid0, instskip,
//! instid1 }` (C3 confirmed the bit ranges).
//!
//! Resolution is exact along the unique reaching path: straight-line within
//! the block, continued across single-predecessor edges. A consumer past the
//! block end is `Ambiguous` when only plain leader splits (fall-through block
//! boundaries) lie between the hint and it, and rejects (`DelayCrossesLeader`)
//! when the skip counts over a branch: that would count across control
//! flow, which the window model cannot preserve. A producer past a
//! join, a loop revisit, or an invalid id yields `DelayStatus::Ambiguous`,
//! never a rejection: hints are performance-only and `NO_DEP` is always
//! sound, so downstream edits touching the reach must rewrite the hint to
//! `NO_DEP` (core.md §2.8). Per-path enumeration across joins is
//! intentionally not attempted: the two paths into a hint may disagree on
//! producer distance, and `Ambiguous` is the honest fact.

use thiserror::Error;

use crate::cfg::{BlockId, Body, InstId, Terminator};
use crate::effects::{Control, MemClass};
use crate::inst::Form;
use crate::operand::Modifiers;
use crate::passes::cfg::containing_block;

/// One validated `s_clause` window: opener, members in layout order, and
/// their shared memory class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClauseWindow {
    pub opener: InstId,
    pub block: BlockId,
    pub members: Vec<InstId>,
    pub class: MemClass,
}

/// Per-path delay fact. `Resolved` carries both dependency slots (each
/// `None` = NO_DEP) plus the two consumer positions (the instructions the
/// hint constrains); `consumers[1]` is what C6's window policy calls the
/// hint's "last target". `Ambiguous` means some path needs a cross-block
/// walk: touchers must rewrite to NO_DEP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayFact {
    pub hint: InstId,
    pub block: BlockId,
    pub status: DelayStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DelayStatus {
    Resolved {
        producers: [Option<InstId>; 2],
        consumers: [Option<InstId>; 2],
    },
    Ambiguous,
}

impl DelayFact {
    /// Constructor for the edit layer (C6): mark a hint ambiguous after a
    /// structural change without re-resolving.
    pub fn ambiguous(hint: InstId, block: BlockId) -> Self {
        Self {
            hint,
            block,
            status: DelayStatus::Ambiguous,
        }
    }

    pub fn is_ambiguous(&self) -> bool {
        matches!(self.status, DelayStatus::Ambiguous)
    }

    /// Dependency slots when resolved (`None` slot = NO_DEP); `None` when
    /// ambiguous.
    pub fn producers(&self) -> Option<[Option<InstId>; 2]> {
        match self.status {
            DelayStatus::Resolved { producers, .. } => Some(producers),
            DelayStatus::Ambiguous => None,
        }
    }

    /// Constrained consumer positions when resolved; `None` when ambiguous.
    pub fn consumers(&self) -> Option<[Option<InstId>; 2]> {
        match self.status {
            DelayStatus::Resolved { consumers, .. } => Some(consumers),
            DelayStatus::Ambiguous => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct WindowFacts {
    pub clauses: Vec<ClauseWindow>,
    pub delays: Vec<DelayFact>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WindowError {
    #[error("body has no blocks; run passes::cfg::build_blocks first")]
    BlocksNotBuilt,
    #[error("s_clause at layout index {index} opens a window of {len} that leaves its block")]
    CrossesLeader { index: usize, len: usize },
    #[error("s_clause at layout index {index} has no decoded window length")]
    ClauseLengthMissing { index: usize },
    #[error("s_clause at layout index {index}: member at layout index {member} is not a memory instruction")]
    NonMemoryMember { index: usize, member: usize },
    #[error("s_clause at layout index {index}: mixed memory classes at layout index {member}")]
    MixedClass { index: usize, member: usize },
    #[error(
        "s_clause at layout index {index}: wait inside clause window at layout index {member}"
    )]
    WaitInsideClause { index: usize, member: usize },
    #[error("s_endpgm inside s_clause window opened at layout index {index}")]
    EndPgmInsideClause { index: usize },
    #[error("s_delay_alu at layout index {index} has no decoded hint fields")]
    HintMissing { index: usize },
    #[error("s_delay_alu at layout index {index} has its consumer past its block")]
    DelayCrossesLeader { index: usize },
}

/// VALU-issuing forms. Comparisons issue on the vector ALU; SALU/DS/VMEM/SMEM/export
/// do not.
fn is_valu(inst: &crate::inst::Inst) -> bool {
    matches!(
        inst.form,
        Form::Vop1
            | Form::Vop2
            | Form::Vop3
            | Form::Vop3p
            | Form::Vopd
            | Form::Vopc
            | Form::Vinterp
    )
}

/// Transcendental VALU ops (LLVM `TRANS = 1`: exp, log, rcp, rcp_iflag, rsq, sqrt, sin,
/// cos in f16/f32/f64). `VALU_DEP_n` counts VALUs that are not TRANS and
/// `TRANS32_DEP_n` counts TRANS only (LLVM `AMDGPUInsertDelayAlu`: `VALUNum` advances
/// on VALU, `TRANSNum` on TRANS). KT48 pins it: `v_rcp_f32` sits between hipcc's
/// `VALU_DEP_3` hints and the producers their consumers read.
pub fn is_trans(inst: &crate::inst::Inst) -> bool {
    const TRANS: [&str; 8] = [
        "v_exp_f",
        "v_log_f",
        "v_rcp_f",
        "v_rcp_iflag_f",
        "v_rsq_f",
        "v_sqrt_f",
        "v_sin_f",
        "v_cos_f",
    ];
    is_valu(inst)
        && inst
            .op
            .name(crate::inst::Arch::Gfx1201)
            .is_some_and(|n| TRANS.iter().any(|t| n.starts_with(t)))
}

/// What `VALU_DEP_n` counts: issued VALUs other than TRANS.
pub fn counts_for_valu_dep(inst: &crate::inst::Inst) -> bool {
    is_valu(inst) && !is_trans(inst)
}

fn is_salu(inst: &crate::inst::Inst) -> bool {
    matches!(
        inst.form,
        Form::Sop1 | Form::Sop2 | Form::Sopc | Form::Sopk | Form::Sopp
    )
}

/// What an INSTID value names: `None` = NO_DEP, the n-th previous non-TRANS VALU, the
/// n-th previous TRANS, the nearest SALU (cycle penalty), or `Unknown` (values 12-15
/// are invalid per the ISA: ambiguous, never reject).
#[derive(Clone, Copy)]
enum IdNeed {
    None,
    Valu(usize),
    Trans(usize),
    Salu,
    Unknown,
}

fn id_need(value: u8) -> IdNeed {
    match value {
        0 => IdNeed::None,
        1..=4 => IdNeed::Valu(usize::from(value)),
        5..=7 => IdNeed::Trans(usize::from(value - 4)),
        // Reserved FMA accumulator penalty: treat as one VALU back.
        8 => IdNeed::Valu(1),
        9..=11 => IdNeed::Salu,
        _ => IdNeed::Unknown,
    }
}

/// Forward consumer distance for an INSTSKIP value: `SAME` = the next
/// instruction, `NEXT` = the one after, `SKIP_N` = N further still.
/// Values 6-7 are invalid per the ISA.
fn skip_forward(value: u8) -> Option<usize> {
    match value {
        0..=5 => Some(usize::from(value)),
        _ => None,
    }
}

fn clause_length(mods: &Modifiers) -> Option<usize> {
    mods.clause.map(usize::from)
}

/// Validate every clause window and resolve every delay hint. Requires built
/// blocks. Clause violations and delay consumers past the block end reject;
/// backward producer over-reach across joins goes ambiguous (never rejects).
pub fn check_windows(body: &Body) -> Result<WindowFacts, WindowError> {
    check_windows_with(body, false)
}

/// As [`check_windows`]; with `lenient_delay` an `s_delay_alu` whose skip counts across a block boundary that is not a
/// plain fall-through (hipcc legally emits a hint whose second consumer lies past a conditional branch or in the
/// fall-through block of one) resolves to `Ambiguous` instead of rejecting. The hint is performance-only (NO_DEP is
/// always sound; the interpreter ignores it), so ambiguity loses nothing; PM-authored code keeps the strict rule.
pub fn check_windows_with(body: &Body, lenient_delay: bool) -> Result<WindowFacts, WindowError> {
    if body.blocks.is_empty() {
        return Err(WindowError::BlocksNotBuilt);
    }
    let mut facts = WindowFacts::default();
    for (pos, &id) in body.layout.iter().enumerate() {
        let inst = body.insts.get(id).expect("layout references live insts");
        match inst.effects.control {
            Control::Clause => {
                let len = clause_length(&inst.mods)
                    .ok_or(WindowError::ClauseLengthMissing { index: pos })?;
                let block = containing_block(body, id).expect("inst is laid out in a block");
                let mut members = Vec::with_capacity(len + 1);
                let mut class: Option<MemClass> = None;
                for k in 1..=len + 1 {
                    let mpos = pos + k;
                    let Some(&member) = body.layout.get(mpos) else {
                        return Err(WindowError::CrossesLeader { index: pos, len });
                    };
                    if containing_block(body, member) != Some(block) {
                        return Err(WindowError::CrossesLeader { index: pos, len });
                    }
                    let minst = body
                        .insts
                        .get(member)
                        .expect("layout references live insts");
                    if matches!(minst.effects.control, Control::EndPgm) {
                        return Err(WindowError::EndPgmInsideClause { index: pos });
                    }
                    if matches!(minst.effects.control, Control::Wait) {
                        return Err(WindowError::WaitInsideClause {
                            index: pos,
                            member: mpos,
                        });
                    }
                    let Some(mem) = &minst.effects.mem else {
                        return Err(WindowError::NonMemoryMember {
                            index: pos,
                            member: mpos,
                        });
                    };
                    if let Some(first) = class {
                        if mem.class != first {
                            return Err(WindowError::MixedClass {
                                index: pos,
                                member: mpos,
                            });
                        }
                    } else {
                        class = Some(mem.class);
                    }
                    members.push(member);
                }
                facts.clauses.push(ClauseWindow {
                    opener: id,
                    block,
                    members,
                    class: class.expect("non-empty window has a class"),
                });
            }
            Control::Delay => {
                let hint = inst
                    .mods
                    .delay
                    .ok_or(WindowError::HintMissing { index: pos })?;
                let block = containing_block(body, id).expect("inst is laid out in a block");
                facts
                    .delays
                    .push(resolve_delay(body, id, block, pos, hint, lenient_delay)?);
            }
            _ => {}
        }
    }
    Ok(facts)
}

fn resolve_delay(
    body: &Body,
    hint: InstId,
    block: BlockId,
    pos: usize,
    raw: crate::operand::DelayAluHint,
    lenient: bool,
) -> Result<DelayFact, WindowError> {
    let ambiguous = DelayFact::ambiguous(hint, block);
    let (_, end) = body.blocks[block.0].range;
    debug_assert_eq!(body.layout.get(pos), Some(&hint));

    // Consumers (forward): consumer0 is the next layout instruction,
    // consumer1 is skip_forward further. SKIP counts layout instructions, so
    // the consumers are positional. A consumer past the block end is exact
    // only when every block boundary the skip crosses is a plain leader split
    // (`Terminator::FallThrough`: a branch target starts the next block, the
    // hint's block runs into it): the next dynamic instruction after the hint
    // is then the next layout instruction (hipcc emits such a hint right
    // before a loop header). The window is not resolved across the split
    // (the join has other predecessors, so its producers are `Ambiguous`).
    // A skip that counts over a branch executes differently per path: reject
    // like a clause window crossing a leader.
    let fwd = match skip_forward(raw.instskip) {
        Some(fwd) => fwd,
        None => return Ok(ambiguous),
    };
    if pos + 1 >= end || pos + 1 + fwd >= end {
        let last = pos + 1 + fwd;
        if body.layout.get(last).is_none() {
            return if lenient {
                Ok(ambiguous)
            } else {
                Err(WindowError::DelayCrossesLeader { index: pos })
            };
        }
        let mut b = block.0;
        while body.blocks[b].range.1 <= last {
            let fall = body.blocks[b].term == Terminator::FallThrough;
            let next = b + 1;
            if !fall
                || body
                    .blocks
                    .get(next)
                    .is_none_or(|n| n.range.0 != body.blocks[b].range.1)
            {
                return if lenient {
                    Ok(ambiguous)
                } else {
                    Err(WindowError::DelayCrossesLeader { index: pos })
                };
            }
            b = next;
        }
        return Ok(ambiguous);
    }
    let consumer0 = body.layout[pos + 1];
    let consumer1 = body.layout[pos + 1 + fwd];

    // Producers (backward): INSTID counts issued VALU on the executed path
    // (branched-over instructions do not count; EXEC-skipped VALU do), from
    // its own consumer. Walk back along single-predecessor chains, which carry
    // exactly one path; a join, a loop revisit, an invalid id, or an unmet
    // need all yield Ambiguous, never a rejection: hints are performance-only
    // and NO_DEP is always sound.
    let mut producers: [Option<InstId>; 2] = [None, None];
    for (slot, value) in [raw.instid0, raw.instid1].iter().enumerate() {
        let from = if slot == 0 { pos + 1 } else { pos + 1 + fwd };
        match id_need(*value) {
            IdNeed::None => {}
            IdNeed::Unknown => return Ok(ambiguous),
            need => match back_search(body, block, from, need) {
                BackResult::Found(inst) => producers[slot] = Some(inst),
                BackResult::Ambiguous => return Ok(ambiguous),
            },
        }
    }
    Ok(DelayFact {
        hint,
        block,
        status: DelayStatus::Resolved {
            producers,
            consumers: [Some(consumer0), Some(consumer1)],
        },
    })
}

enum BackResult {
    Found(InstId),
    Ambiguous,
}

/// Backward search for a producer along the unique reaching path, from layout
/// position `pos` (exclusive): within the block, then across single-predecessor
/// edges (exact: one path reaches the hint through such an edge). `Valu(n)` counts
/// the n-th previous non-TRANS VALU, `Trans(n)` the n-th previous TRANS, `Salu` finds
/// the nearest previous SALU-form instruction (M1 approximation of the SALU cycle
/// penalty; `s_delay_alu` itself is not one). Unmet needs and multi-predecessor joins
/// yield Ambiguous.
fn back_search(body: &Body, block: BlockId, pos: usize, need: IdNeed) -> BackResult {
    let (counts, need): (fn(&crate::inst::Inst) -> bool, usize) = match need {
        IdNeed::Valu(n) => (counts_for_valu_dep, n),
        IdNeed::Trans(n) => (is_trans, n),
        IdNeed::Salu => (
            |i: &crate::inst::Inst| is_salu(i) && i.effects.control != Control::Delay,
            1,
        ),
        IdNeed::None | IdNeed::Unknown => return BackResult::Ambiguous,
    };
    let mut seen = 0usize;
    let mut cur = block;
    let mut j = pos;
    let mut visited = vec![block];
    loop {
        let (start, _) = body.blocks[cur.0].range;
        while j > start {
            j -= 1;
            let candidate = body.layout[j];
            if counts(body.insts.get(candidate).expect("laid out")) {
                seen += 1;
                if seen == need {
                    return BackResult::Found(candidate);
                }
            }
        }
        let preds = &body.blocks[cur.0].preds;
        if preds.len() != 1 || visited.contains(&preds[0]) {
            return BackResult::Ambiguous;
        }
        cur = preds[0];
        visited.push(cur);
        j = body.blocks[cur.0].range.1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Arena, Cond};
    use crate::effects::{Effects, MemEffect, OrderType, SrcRead};
    use crate::inst::{Form, Inst};
    use crate::isa;
    use crate::operand::{DelayAluHint, ImmField, Operand};
    use crate::passes::cfg::build_blocks;
    use crate::provenance::Provenance;
    use crate::wait::CounterSet;

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

    fn mem(name: &str, form: Form, class: MemClass) -> Inst {
        let mut inst = mk(name, form);
        inst.effects.mem = Some(MemEffect {
            class,
            counters: CounterSet(0),
            in_order_type: OrderType::Load,
            src_read: SrcRead::AtIssue,
        });
        inst
    }

    fn clause_opener(len: u8) -> Inst {
        let mut inst = mk("s_clause", Form::Sopp);
        inst.effects.control = Control::Clause;
        inst.mods.clause = Some(len);
        inst
    }

    fn delay_hint(instid0: u8, instskip: u8, instid1: u8) -> Inst {
        let mut inst = mk("s_delay_alu", Form::Sopp);
        inst.effects.control = Control::Delay;
        inst.mods.delay = Some(DelayAluHint {
            instid0,
            instskip,
            instid1,
        });
        inst
    }

    fn valu() -> Inst {
        mk("v_add_nc_u32_e32", Form::Vop2)
    }

    fn branch_to(off: i16) -> Inst {
        let mut inst = mk("s_cbranch_scc1", Form::Sopp);
        inst.effects.control = Control::Branch { cond: Cond::Scc1 };
        inst.operands.push(Operand::Imm(ImmField::Sopp(off)));
        inst
    }

    fn endpgm() -> Inst {
        let mut inst = mk("s_endpgm", Form::Sopp);
        inst.effects.control = Control::EndPgm;
        inst
    }

    #[test]
    fn delay_past_block_start_without_unique_path_is_ambiguous() {
        // Hint needs 2 VALU back, only 1 precedes it on the unique entry path.
        let mut body = body_of(vec![valu(), delay_hint(2, 0, 0), valu(), endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let facts = check_windows(&body).unwrap();
        assert!(facts.delays[0].is_ambiguous());
        assert_eq!(facts.delays[0].producers(), None);
    }

    #[test]
    fn delay_producer_resolves_across_single_pred_edge() {
        // The 2nd VALU back lives in the single-predecessor entry block: one
        // reaching path, so the walk continues across the edge exactly.
        // valu@0 jump@1(->@2) valu@2 hint@3(need 2) valu@4 endpgm@5.
        let mut entry = mk("s_branch", Form::Sopp);
        entry.effects.control = Control::Jump;
        entry.operands.push(Operand::Imm(ImmField::Sopp(0)));
        let mut body = body_of(vec![
            valu(),
            entry,
            valu(),
            delay_hint(2, 0, 0),
            valu(),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks.len(), 2);
        let facts = check_windows(&body).unwrap();
        assert_eq!(
            facts.delays[0].producers(),
            Some([Some(body.layout[0]), None])
        );
    }

    #[test]
    fn delay_consumer_across_a_leader_split_is_ambiguous() {
        // A branch targets the instruction right after the hint, so the hint
        // is alone in its block and runs into the next one by fall-through:
        // the next dynamic instruction is the next layout instruction, so the
        // hint is legal, but the window is not resolved across the split.
        // Layout: hint@0 valu@1 valu@2 branch@3(taken valu@1, off -3) endpgm@4.
        let mut back = mk("s_branch", Form::Sopp);
        back.effects.control = Control::Jump;
        back.operands.push(Operand::Imm(ImmField::Sopp(-3)));
        let mut body = body_of(vec![delay_hint(0, 0, 0), valu(), valu(), back, endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks[0].range, (0, 1));
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.delays.len(), 1);
        assert!(facts.delays[0].is_ambiguous());
    }

    #[test]
    fn delay_skip_over_a_branch_rejects() {
        // The second consumer (SKIP_1: two layout instructions after the hint's
        // next) lies past a conditional branch: the skip would count over
        // control flow, which differs per path.
        // Layout: hint@0 valu@1 cbr@2(taken endpgm@4) valu@3 endpgm@4.
        let mut body = body_of(vec![
            delay_hint(1, 2, 1),
            valu(),
            branch_to(1),
            valu(),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(body.blocks[0].range, (0, 3));
        assert_eq!(
            check_windows(&body),
            Err(WindowError::DelayCrossesLeader { index: 0 })
        );
    }
    #[test]
    fn hipcc_delay_skip_over_a_branch_is_accepted_as_ambiguous() {
        // Same shape as `delay_skip_over_a_branch_rejects`: hipcc emits such hints (consumer in the fall-through
        // block of a conditional branch); the lenient check keeps them as ambiguous, performance-only facts.
        let mut body = body_of(vec![
            delay_hint(1, 2, 1),
            valu(),
            branch_to(1),
            valu(),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::DelayCrossesLeader { index: 0 })
        );
        let facts = check_windows_with(&body, true).unwrap();
        assert_eq!(facts.delays.len(), 1);
        assert!(facts.delays[0].is_ambiguous());
    }
    #[test]
    fn smem_clause_of_mixed_widths_validates() {
        // KT48's s_load_b256 + s_load_b128 clause: mixed widths, one class.
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b256", Form::Smem, MemClass::SmemLoad),
            mem("s_load_b128", Form::Smem, MemClass::SmemLoad),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.clauses.len(), 1);
        assert_eq!(facts.clauses[0].class, MemClass::SmemLoad);
        assert_eq!(facts.clauses[0].members.len(), 2);
    }

    #[test]
    fn clause_window_crossing_a_target_rejects() {
        // Branch targets the second clause member: leaders split the window.
        // Layout: clause(1)@0 mem@1 cbr@2(taken mem@1... forward: taken endpgm@4) mem@3 endpgm@4.
        // Simpler: clause@0, mem@1, branch@2 taken->endpgm@4, mem@3, endpgm@4
        // with the branch targeting mem@3? No: crossing needs a leader INSIDE
        // the window. Target mem@1 from a later branch instead:
        // clause@0 mem@1 endpgm... layout: [clause(1), mem, branch(taken mem@1), mem2, endpgm]:
        // branch@2 next=3, target mem@1: off = 1-3 = -2. Leaders: 0, target 1,
        // after-branch 3. Window members @1,@2: @2 is branch (non-mem)... that
        // rejects NonMemory, not CrossesLeader. Make the window longer:
        // clause(2)@0 mem@1 mem@2 branch@3(taken mem@2): SMEM is 2 dwords, so
        // pcs are 0,1,3,5,6 and off = 3-6 = -3. Leaders: 0, 2, 4.
        let mut body = body_of(vec![
            clause_opener(2),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            branch_to(-3),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::CrossesLeader { index: 0, len: 2 })
        );
    }

    #[test]
    fn clause_rejects_wait_nonmem_mixedclass_and_endpgm() {
        // Wait inside.
        let mut wait = mk("s_wait_dscnt", Form::Sopp);
        wait.effects.control = Control::Wait;
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            wait,
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::WaitInsideClause {
                index: 0,
                member: 2
            })
        );
        // Non-memory member.
        let mut body = body_of(vec![clause_opener(0), valu(), endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::NonMemoryMember {
                index: 0,
                member: 1
            })
        );
        // Mixed classes.
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            mem(
                "global_load_b32",
                Form::Vmem(crate::inst::VmemForm::Global),
                MemClass::VmemLoad,
            ),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::MixedClass {
                index: 0,
                member: 2
            })
        );
        // s_endpgm inside.
        let mut body = body_of(vec![
            clause_opener(1),
            mem("s_load_b32", Form::Smem, MemClass::SmemLoad),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        assert_eq!(
            check_windows(&body),
            Err(WindowError::EndPgmInsideClause { index: 0 })
        );
    }

    #[test]
    fn delay_resolves_producers_and_consumers_in_block() {
        // valu@0 valu@1 hint@2(VALU_DEP_1, SAME, VALU_DEP_1) consumer@3 endpgm@4.
        let mut body = body_of(vec![valu(), valu(), delay_hint(1, 0, 1), valu(), endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.delays.len(), 1);
        let delay = &facts.delays[0];
        assert_eq!(
            delay.producers(),
            Some([Some(body.layout[1]), Some(body.layout[1])])
        );
        assert_eq!(
            delay.consumers(),
            Some([Some(body.layout[3]), Some(body.layout[3])])
        );
    }

    #[test]
    fn delay_skip_selects_the_second_consumer() {
        // hint@1 SKIP_1: consumer1 is two ahead (p+1+2 with skip value 2).
        let mut body = body_of(vec![
            valu(),
            delay_hint(1, 2, 0),
            valu(),
            valu(),
            valu(),
            endpgm(),
        ]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let facts = check_windows(&body).unwrap();
        let delay = &facts.delays[0];
        assert_eq!(
            delay.consumers(),
            Some([Some(body.layout[2]), Some(body.layout[4])])
        );
        assert_eq!(delay.producers(), Some([Some(body.layout[0]), None]));
    }

    #[test]
    fn no_dep_hint_resolves_empty() {
        let mut body = body_of(vec![valu(), delay_hint(0, 0, 0), valu(), endpgm()]);
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let facts = check_windows(&body).unwrap();
        assert_eq!(facts.delays[0].producers(), Some([None, None]));
    }
}

#[cfg(test)]
mod instid1_tests {
    use super::*;
    use crate::cfg::Arena;
    use crate::inst::Inst;
    use crate::operand::DelayAluHint;
    use crate::passes::cfg::build_blocks;

    fn decoded(words: &[u32]) -> Inst {
        crate::codec::gfx12::decode(words).unwrap().0
    }

    /// RDNA4 ISA §16.5 p270, `S_DELAY_ALU` example: INSTID1 counts back from the
    /// instruction INSTSKIP selects, so the skipped VALU is `VALU_DEP_1` of the third.
    /// Encodings from pinned `llvm-mc -mcpu=gfx1201` (the `_e32` spellings of the example).
    #[test]
    fn instid1_counts_back_from_its_own_consumer() {
        let insts = vec![
            decoded(&[0x7e06_0300]), // v_mov_b32_e32 v3, v0
            decoded(&[0x303c_3e81]), // v_lshlrev_b32_e32 v30, 1, v31
            decoded(&[0x3030_3281]), // v_lshlrev_b32_e32 v24, 1, v25
            decoded(&[0xbf87_00a3]), // s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
            decoded(&[0x0600_0701]), // v_add_f32_e32 v0, v1, v3
            decoded(&[0x0816_1309]), // v_sub_f32_e32 v11, v9, v9
            decoded(&[0x1014_170d]), // v_mul_f32_e32 v10, v13, v11
            decoded(&[0xbfb0_0000]), // s_endpgm
        ];
        assert_eq!(
            insts[3].mods.delay,
            Some(DelayAluHint {
                instid0: 3,
                instskip: 2,
                instid1: 1
            })
        );
        let mut arena: Arena<Inst> = Arena::new();
        let layout: Vec<_> = insts.into_iter().map(|i| arena.insert(i)).collect();
        let mut body = Body {
            insts: arena,
            blocks: Vec::new(),
            layout,
        };
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).unwrap();
        let facts = check_windows(&body).unwrap();
        let l = &body.layout;
        assert_eq!(facts.delays[0].consumers(), Some([Some(l[4]), Some(l[6])]));
        assert_eq!(
            facts.delays[0].producers(),
            Some([Some(l[0]), Some(l[5])]),
            "v_mov v3 for v_add; v_sub v11 for v_mul"
        );
    }
}
