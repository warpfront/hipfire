// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CFG-aware wait verification of an emitted [`Program`].
//!
//! The builder's [`Ledger`] runs linearly across raw branches: at a label it
//! carries the ledger of the *preceding instruction* (after an unconditional
//! `s_branch`/`s_endpgm`, a dead fall-through), not of the label's real
//! predecessors, so a wait that exists on the linear path may be missing on a
//! branch path that reaches the same label, and a load that exists only on a
//! dead path would be a ghost if the ledgers were joined. This module
//! re-derives the pending operations from the emitted instruction stream alone
//! and checks every use, WAR and WAW with the builder's own
//! [`Ledger::required`] / [`Ledger::record`] on **every** control-flow path:
//!
//! * Path-sensitive: each path carries its own `Ledger`; nothing is joined, so
//!   an operation is pending exactly on the paths that issue it (no `maybe`
//!   ghost loads). States are deduplicated at labels by id-free pending shapes,
//!   omitting gfx942 store source locks that replay never reads. Strict gfx942
//!   replay also identifies continuation-equivalent VM load queues: if EVERY
//!   path reaches vmcnt(0), every intervening VM issue is an ordered load, and
//!   no incoming load destination is read OR written before that drain, the
//!   incoming VM queue is omitted from the key only. Stores/FLAT never qualify;
//!   LGKM/EXP and control-proof directions remain in the key. Actual ledgers,
//!   recording, waits and issue clipping are unchanged (`vm_drain_observers`).
//!   Backedges run until the label state set reaches a fixed point; operation
//!   ids that merely advance per iteration do not grow it.
//! * Waits are decoded from the instruction text actually in the program, never
//!   from the builder's `WaitProof` records: gfx12 `s_wait_loadcnt`/`storecnt`/
//!   `dscnt`/`kmcnt` and the combined `loadcnt_dscnt`/`storecnt_dscnt`,
//!   `s_wait_alu depctr_vm_vsrc(0)`; gfx11 and CDNA `s_waitcnt` fields, gfx11 raw
//!   immediates, `s_waitcnt_vscnt`/`_vmcnt`/`_lgkmcnt`/`_expcnt`/`_depctr`. A count
//!   equal to the counter's architectural maximum (the all-ones "no wait" field:
//!   gfx12 63, KMcnt 31, gfx11 63/EXPcnt 7, CDNA `Caps`) retires nothing; a count
//!   above it, reserved immediate bits, a bare `s_waitcnt`, a wait of the
//!   wrong architecture and any `s_wait*` form not listed below are errors. The
//!   only `s_wait*` forms accepted without effect are `s_wait_alu` (other than
//!   `depctr_vm_vsrc(0)`), `s_wait_idle`, `s_wait_event`, `s_wait_expcnt`,
//!   `s_wait_samplecnt` and `s_wait_bvhcnt`: the ledger records no operation on
//!   those counters.
//! * A nonzero wait retires by position only when every pending operation of
//!   its counter is in order, certain and of one order class
//!   (`Ledger::wait_replay`): with mixed classes (DS with SMEM on LGKMcnt, buffer
//!   with global on gfx11, out-of-order SMEM) nothing retires.
//! * `s_endpgm` ends a path (pending operations die with the wave, as in the
//!   builder, which resets its ledger there). Unreachable code is not checked.
//! * Foreign text (lifted hipcc bodies, `Builder::foreign_region`) carries no
//!   def/use/memory metadata. The spans passed as `foreign` (`region_spans`, or the
//!   whole program for a lifted object) are decoded line by
//!   line through the lift's instruction table (`peacemaker_lift::text::parse_line`,
//!   whose `Effects` give defs, uses and the memory class) and checked like any
//!   other instruction. A foreign line the table cannot decode, a memory class the
//!   ledger cannot track (LDS DMA; FLAT counts on both the VMEM counter and DScnt), or a `ttmp` operand of a memory
//!   operation is an error, not a skipped line (`ttmp` operands of other
//!   instructions, e.g. workgroup-id reads, are not tracked). Outside `foreign`, an instruction with empty metadata is taken
//!   at its word: it has no memory effect.
//! * CDNA (gfx942) specifics, both exercised by `cdna::lifted::certify`: `s_barrier` is not
//!   a memory wait, so it is a violation (`Lgkm` 0, `Reason::Barrier`) while any DS operation
//!   of the wave is still pending (LDS RAW for a pending write, WAR for a pending read);
//!   and a pending store's source registers are not held until its counter retires
//!   (`Ledger::required_replay`: no `Reason::War` on gfx942, the store still occupies its
//!   counter slot and still blocks a barrier; other architectures unchanged).
//! * Architectural issue bound (`Ledger::clip_to_issue_bound`, applied after every
//!   recorded memory operation; verifier only, the Builder's emission is unchanged): a
//!   counter field holds at most its maximum of outstanding operations (63 on
//!   LOADcnt/STOREcnt/DScnt and gfx11 VMcnt/VScnt/LGKMcnt, 31 on KMcnt, the `Caps` maxima
//!   on CDNA) and a wave stalls issue at that maximum, so the oldest operation has retired
//!   when the one past the maximum issues. A counter's queue is clipped to its youngest
//!   maximum **only when every operation on it is in order, certain and of one order
//!   class**; a mixed or out-of-order queue (atomics, SMEM, DS with SMEM, buffer with global
//!   on gfx11) is never clipped, since its retirement order is unknown. This is what bounds
//!   the fixpoint of a loop that issues ordered stores or loads without a wait, and it
//!   frees exactly the oldest operation's registers (the 64th store issued retires the
//!   first; the second stays locked).
//! * Label dedup after a wait-only prefix: a label's contiguous run of `s_wait*` lines
//!   (ended by the next label or any other instruction, so a branch target is never inside
//!   it) is applied to the path's ledger first, counting its steps, and the label's seen
//!   states are keyed on the ledger after it. Nothing uses a register or branches between
//!   the label and those waits, so this is the walk it would do anyway; distinct incoming
//!   states that a full drain collapses are one state. General rule, no label is special
//!   and the state bound is unchanged.
//! * No operation is ever dropped from the ledger except by an explicit wait or
//!   the architectural issue bound. More than [`COUNTER_CAP`] pending operations
//!   on one counter (unclipped mixed/out-of-order queues), more than
//!   [`MAX_STATES_PER_LABEL`] continuation classes at one label, more than
//!   [`MAX_STEPS`] visits, or unknown control flow fail closed.
//!
//! A violation names the instruction, the counter wait it lacks, and the branch
//! edges of the path that reaches it.
//!
//! [`verify_dispatcher`] runs the same walk for a pool whose hosted bodies are foreign
//! lifted text with findings of their own: it tracks every body memory operation and wait
//! (so the dispatcher after a body sees the state the body leaves) but does not report a
//! hazard whose pc lies inside a body span; every hazard outside the exact spans is
//! fatal and all are reported. It proves the dispatcher only, never a body.
//!
//! Static branch proofs (strict only, crate-private [`verify_with_branch_proofs`]): the
//! typed gfx942 certification passes one `ledger_control::EdgeChoice` per instruction. A
//! conditional `s_cbranch_scc0/scc1` proven (block-local scalar facts valid for every state
//! entering its basic block) to go one way follows only that edge, with the current ledger and
//! the same path-witness edge; its other edge is not walked. A second table of
//! `ledger_control::EntryProof`s refines the same decision for executions that entered the
//! branch's block over one specific edge (a join whose predecessor leaves a constant in an
//! SGPR, e.g. hipcc's `s_mov_b64 s[14:15], 0; s_branch J` / `J: s_and_b64 s[14:15], s[14:15],
//! exec; s_cselect_b32; s_cmp_lg_u32; s_cbranch_scc1`): the walk remembers the edge it
//! crossed into a label, applies the matching proof to the block's terminal branch only, and
//! keys the label's dedup state on the applied direction as well, so two arrivals with equal
//! pending memory but different feasible directions are never merged. This is a
//! control-feasibility filter, not a hazard rule: memory recording, waits, the issue clip, the
//! caps and the ledger half of the dedup key (`Ledger::replay_state_key` / `full_state_key`)
//! are untouched. Equal extended keys have identical successors. Edge-conditioned choices
//! can split an old label state, so no monotone state-count reduction is assumed; the same
//! unchanged state/step caps still fail closed. With no `EntryProof` the label key is the old
//! one. Target labels of an
//! impossible edge are still validated. [`verify`], [`verify_emitted`], [`verify_dispatcher`]
//! and [`verify_dispatcher_emitted`] pass no proofs (all `Both`).
//!
//! Region summary (dispatcher mode only): a body whose every exiting path provably leaves
//! every memory counter drained is not walked; the walk goes from its entry label to its
//! exit label with an empty ledger. The proof is a finite may-pending dataflow over the
//! body's exact CFG (`region_exit_is_drained`): every counter of the architecture is
//! potentially pending at the entry, a memory operation sets its counter's bit, a wait of
//! count zero clears it, joins are bitwise OR and loops reach a fixed point. Every branch
//! must stay inside the body or land on the exit label, and nothing may be undecodable or
//! indirect; otherwise (and for any counter bit left at the exit) there is no summary and
//! the body is walked as before, so the effect of an unproven body on the dispatcher is
//! unchanged. The summary says the emitted waits retire any incoming or body operation
//! on every exiting path; it says nothing about the body's uses, which only the strict
//! [`verify`] checks. It is general (all counters, no label names) and finite, so it
//! bounds the walk without raising [`MAX_STATES_PER_LABEL`].

use crate::ledger_control::{EdgeChoice, EntryProof};
use crate::{
    arch::Arch,
    insn::{Instruction, MemoryClass, Program},
    ledger::{Counter, Ledger, MemoryFamily, Reason, StateKey},
    plan::{Emitted, RegionProof},
    reg::{Kind, RegRef},
    sync,
};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// Most distinct in-flight states tracked at one label before the walk fails closed.
pub const MAX_STATES_PER_LABEL: usize = 512;
/// Most instruction visits of one verification before the walk fails closed.
pub const MAX_STEPS: u64 = 20_000_000;
/// Most pending operations of one counter on one path (hardware holds at most 63;
/// past this the walk reports unbounded in-flight growth instead of dropping any).
pub const COUNTER_CAP: usize = 256;
/// Branch edges named in a violation's path witness (the last ones).
const WITNESS_EDGES: usize = 16;

/// What one verification explored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Instruction visits.
    pub steps: u64,
    /// Distinct (label, in-flight state) pairs reached.
    pub label_states: usize,
    /// Distinct hazards (pc, counter, count, reason) found inside the body spans of
    /// [`verify_dispatcher`] and not reported there; always 0 for [`verify`].
    pub suppressed: usize,
    /// Regions [`verify_dispatcher`] did not walk because their exit is proven to leave every
    /// memory counter drained on every path (module docs); always 0 for [`verify`].
    pub summarized: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Effect {
    Wait(Counter, u8),
    /// `depctr_vm_vsrc(0)`: every issued VMEM store has read its sources.
    ReleaseStoreSources,
}

fn imm(text: &str) -> Option<u32> {
    let token = text.split_whitespace().next()?;
    match token.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => token.parse().ok(),
    }
}

/// A 16-bit SOPP immediate as the raw field: unsigned up to 0xffff, or the signed
/// spelling of the same bits (`-1017` is 0xfc07); anything outside both is `None`.
fn sopp16(text: &str) -> Option<u32> {
    let token = text.split_whitespace().next()?;
    match token.strip_prefix('-') {
        Some(magnitude) => {
            let value = -i64::from(imm(magnitude)?);
            let signed = i16::try_from(value).ok()?;
            Some(u32::from(signed as u16))
        }
        None => imm(token).filter(|&v| v <= 0xffff),
    }
}

/// The largest count of `counter` on `arch`: the all-ones field is the encoding
/// of "no wait".
fn capacity(arch: Arch, counter: Counter) -> Option<u8> {
    if let Some(cdna) = arch.cdna() {
        let caps = cdna.caps();
        return match counter {
            Counter::Vm => Some(caps.vmcnt_max),
            Counter::Lgkm => Some(caps.lgkmcnt_max),
            Counter::Exp => Some(caps.expcnt_max),
            _ => None,
        };
    }
    if arch.gfx12() {
        match counter {
            Counter::Load | Counter::Store | Counter::Ds => Some(63),
            Counter::Km => Some(31),
            _ => None,
        }
    } else {
        match counter {
            Counter::Vm | Counter::Vs | Counter::Lgkm => Some(63),
            Counter::Exp => Some(7),
            _ => None,
        }
    }
}

/// `s_wait*` forms with no ledger effect: the ledger records no operation on the
/// counters they wait for (or they only order ALU hazards).
fn inert_wait(name: &str) -> bool {
    matches!(
        name,
        "s_wait_alu"
            | "s_wait_idle"
            | "s_wait_event"
            | "s_wait_expcnt"
            | "s_wait_samplecnt"
            | "s_wait_bvhcnt"
    )
}

/// The ledger effects of a wait instruction (empty for any other instruction).
fn decode(text: &str, arch: Arch) -> Result<Vec<Effect>, String> {
    let (name, operands) = match text.split_once(char::is_whitespace) {
        Some((name, operands)) => (name, operands.trim()),
        None => (text, ""),
    };
    if !name.starts_with("s_wait") {
        return Ok(Vec::new());
    }
    let bad = |why: &str| format!("`{text}`: {why}");
    // One counter's wait value: the no-wait maximum retires nothing, more is an error.
    let wait = |counter: Counter, value: u32| -> Result<Option<Effect>, String> {
        let cap = capacity(arch, counter)
            .ok_or_else(|| bad(&format!("{counter:?} is not a {} counter", arch.name())))?;
        if value > u32::from(cap) {
            return Err(bad(&format!(
                "{counter:?} count {value} exceeds its maximum {cap}"
            )));
        }
        Ok((value < u32::from(cap)).then_some(Effect::Wait(counter, value as u8)))
    };
    let waits = |list: &[(Counter, u32)]| -> Result<Vec<Effect>, String> {
        let mut out = Vec::new();
        for &(counter, value) in list {
            out.extend(wait(counter, value)?);
        }
        Ok(out)
    };
    let one = |counter: Counter| -> Result<Vec<Effect>, String> {
        let value = imm(operands).ok_or_else(|| bad("invalid wait count"))?;
        waits(&[(counter, value)])
    };
    let releases = operands.contains("depctr_vm_vsrc(0)");
    if arch.gfx12() {
        return match name {
            "s_wait_loadcnt" => one(Counter::Load),
            "s_wait_storecnt" => one(Counter::Store),
            "s_wait_dscnt" => one(Counter::Ds),
            "s_wait_kmcnt" => one(Counter::Km),
            "s_wait_loadcnt_dscnt" | "s_wait_storecnt_dscnt" => {
                let encoded = imm(operands).ok_or_else(|| bad("invalid combined wait"))?;
                if encoded & !0x3f3f != 0 {
                    return Err(bad("out-of-range combined wait (reserved bits set)"));
                }
                let first = if name == "s_wait_loadcnt_dscnt" {
                    Counter::Load
                } else {
                    Counter::Store
                };
                waits(&[
                    (first, (encoded >> 8) & 0x3f),
                    (Counter::Ds, encoded & 0x3f),
                ])
            }
            "s_wait_alu" if releases => Ok(vec![Effect::ReleaseStoreSources]),
            _ if name == "s_waitcnt" || name.starts_with("s_waitcnt_") => {
                Err(bad("not a gfx12 wait"))
            }
            _ if inert_wait(name) => Ok(Vec::new()),
            _ => Err(bad("unknown wait form")),
        };
    }
    // The last comma-separated operand is the count (`s_waitcnt_vscnt null, 0x0`).
    let last = |counter: Counter| -> Result<Vec<Effect>, String> {
        let value = imm(operands.rsplit(',').next().unwrap_or(""))
            .ok_or_else(|| bad("invalid wait count"))?;
        waits(&[(counter, value)])
    };
    match name {
        "s_waitcnt" => {
            if operands.is_empty() {
                return Err(bad("s_waitcnt without counter fields"));
            }
            if operands
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_digit() || b == b'-')
            {
                // gfx11 raw immediate: VM[15:10] LGKM[9:4] EXP[2:0]; bit 3 is reserved. The
                // IR printer shows the 16-bit SOPP field signed (`s_waitcnt -1017` is 0xfc07).
                if arch.cdna().is_some() {
                    return Err(bad("raw s_waitcnt immediates are not decoded on CDNA"));
                }
                let encoded = sopp16(operands).ok_or_else(|| bad("invalid s_waitcnt immediate"))?;
                if encoded & 0x8 != 0 {
                    return Err(bad("s_waitcnt immediate sets reserved bits"));
                }
                waits(&[
                    (Counter::Vm, (encoded >> 10) & 0x3f),
                    (Counter::Lgkm, (encoded >> 4) & 0x3f),
                    (Counter::Exp, encoded & 0x7),
                ])
            } else {
                let mut list = Vec::new();
                for field in operands
                    .split(|c: char| c.is_whitespace() || c == '&' || c == ',')
                    .filter(|f| !f.is_empty())
                {
                    let (field_name, value) = field
                        .split_once('(')
                        .ok_or_else(|| bad("invalid s_waitcnt field"))?;
                    let value: u32 = value
                        .strip_suffix(')')
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| bad("invalid s_waitcnt field"))?;
                    let counter = match field_name {
                        "vmcnt" => Counter::Vm,
                        "lgkmcnt" => Counter::Lgkm,
                        "expcnt" => Counter::Exp,
                        _ => return Err(bad("unknown s_waitcnt field")),
                    };
                    list.push((counter, value));
                }
                waits(&list)
            }
        }
        "s_waitcnt_vscnt" => last(Counter::Vs),
        "s_waitcnt_vmcnt" => last(Counter::Vm),
        "s_waitcnt_lgkmcnt" => last(Counter::Lgkm),
        "s_waitcnt_expcnt" => last(Counter::Exp),
        "s_waitcnt_depctr" if releases => Ok(vec![Effect::ReleaseStoreSources]),
        "s_waitcnt_depctr" => Ok(Vec::new()),
        _ if name.starts_with("s_wait_") && !inert_wait(name) => Err(bad("not a pre-gfx12 wait")),
        _ if inert_wait(name) => Ok(Vec::new()),
        _ => Err(bad("unknown wait form")),
    }
}

/// An instruction with its ledger metadata decoded from its text through the lift's
/// instruction table (`Effects`: defs, uses, memory class).
fn enrich(text: &str, arch: Arch) -> Result<Instruction, String> {
    use peacemaker_ir::{effects::MemClass, inst::Arch as IrArch, reg::Kind as IrKind};
    let ir = match arch {
        Arch::Gfx1100 => IrArch::Gfx1100,
        Arch::Gfx1151 => IrArch::Gfx1151,
        Arch::Gfx1200 => IrArch::Gfx1200,
        Arch::Gfx1201 => IrArch::Gfx1201,
        Arch::Gfx942 => return Err("foreign text has no gfx942 front end".into()),
    };
    let inst = peacemaker_lift::text::parse_line(text, ir)
        .map_err(|e| format!("not decodable by the instruction table ({e})"))?;
    let memory = inst.effects.mem.is_some();
    // The ledger has no ttmp registers: a ttmp operand of an ALU instruction (a workgroup-id
    // read) cannot be a pending-memory hazard and is not tracked; one of a memory operation
    // fails closed.
    let reg = |r: peacemaker_ir::reg::RegRef| -> Result<Option<RegRef>, String> {
        let kind = match r.kind {
            IrKind::V => Kind::V,
            IrKind::S => Kind::S,
            IrKind::Ttmp if memory => return Err("a ttmp operand of a memory operation".into()),
            IrKind::Ttmp => return Ok(None),
        };
        Ok(Some(RegRef {
            kind,
            base: u8::try_from(r.base).map_err(|_| format!("register {r} out of range"))?,
            len: r.len,
        }))
    };
    let regs = |list: &[peacemaker_ir::reg::RegRef]| -> Result<Vec<RegRef>, String> {
        let mut out = Vec::new();
        for &r in list {
            out.extend(reg(r)?);
        }
        Ok(out)
    };
    let defs = regs(&inst.effects.defs)?;
    let uses = regs(&inst.effects.uses)?;
    let mut out = Instruction::new(text, defs, uses);
    if let Some(mem) = inst.effects.mem {
        out = out.memory(match mem.class {
            MemClass::VmemLoad => MemoryClass::VmemLoad,
            MemClass::VmemStore => MemoryClass::VmemStore,
            MemClass::VmemAtomic { returns } => MemoryClass::VmemAtomic { returns },
            MemClass::DsLoad => MemoryClass::DsLoad,
            MemClass::DsStore => MemoryClass::DsStore,
            // A DS atomic counts on the DS counter and returns in order with it: a
            // returning one defines its destination like a load, a returnless one
            // keeps its sources locked like a store (the ledger's own model).
            MemClass::DsAtomic { returns: true } => MemoryClass::DsLoad,
            MemClass::DsAtomic { returns: false } => MemoryClass::DsStore,
            // SMEM returns out of order on its own counter; a store defines nothing.
            MemClass::SmemLoad | MemClass::SmemStore => MemoryClass::SmemLoad,
            MemClass::Export => MemoryClass::Export,
            MemClass::FlatLoad => MemoryClass::Flat { load: true },
            MemClass::FlatStore => MemoryClass::Flat { load: false },
            MemClass::FlatAtomic { returns } => MemoryClass::Flat { load: returns },
            class @ MemClass::LdsDma => {
                return Err(format!(
                    "a {class:?} memory operation, which the wait ledger cannot track"
                ))
            }
        });
    }
    Ok(out)
}

fn is_label(text: &str) -> Option<&str> {
    text.strip_suffix(':')
        .filter(|l| !l.is_empty() && !l.contains(' '))
}

/// Control flow the walk cannot follow.
fn indirect(mnemonic: &str) -> bool {
    ["s_setpc", "s_swappc", "s_call", "s_rfe", "s_subvector_loop"]
        .iter()
        .any(|p| mnemonic.starts_with(p))
}

/// Instructions that need no decoded effects: labels, branches, the program end
/// and waits are handled from their text.
fn handled_by_text(text: &str) -> bool {
    let mnemonic = text.split_whitespace().next().unwrap_or("");
    is_label(text).is_some()
        || mnemonic.starts_with("s_wait")
        || mnemonic.starts_with("s_endpgm")
        || mnemonic == "s_branch"
        || mnemonic.starts_with("s_cbranch_")
        || indirect(mnemonic)
}

/// Branch edges taken on the way to a visit, as a parent-linked arena.
#[derive(Default)]
struct Witness {
    /// (parent edge, pc of the branch, taken)
    edges: Vec<(Option<usize>, usize, bool)>,
}
impl Witness {
    fn push(&mut self, parent: Option<usize>, pc: usize, taken: bool) -> Option<usize> {
        self.edges.push((parent, pc, taken));
        Some(self.edges.len() - 1)
    }
    fn describe(&self, mut at: Option<usize>, insns: &[Instruction]) -> String {
        let mut chain = Vec::new();
        while let Some(i) = at {
            let (parent, pc, taken) = self.edges[i];
            chain.push((pc, taken));
            at = parent;
        }
        let total = chain.len();
        chain.reverse();
        let mut out = String::from("entry");
        if total > WITNESS_EDGES {
            out.push_str(&format!(
                " -> ... ({} earlier edges)",
                total - WITNESS_EDGES
            ));
        }
        for (pc, taken) in chain.into_iter().skip(total.saturating_sub(WITNESS_EDGES)) {
            out.push_str(&format!(
                " -> pc {pc} `{}` {}",
                insns[pc].text.trim(),
                if taken { "taken" } else { "not taken" }
            ));
        }
        out
    }
}

/// The instruction ranges of the region bodies of a built program: its `RegionProof`s'
/// `entry_pc + 1 .. exit_pc` (foreign lifted bodies and typed regions alike).
pub fn region_spans(regions: &[RegionProof]) -> Vec<Range<usize>> {
    regions.iter().map(|r| r.entry_pc + 1..r.exit_pc).collect()
}

/// `verify` of a finished emission, its region bodies decoded through the lift's
/// instruction table. Strict: the bodies' findings are findings.
pub fn verify_emitted(emitted: &Emitted) -> Result<Stats, String> {
    verify(&emitted.program, &region_spans(&emitted.proof.regions))
}

/// [`verify_dispatcher`] of a finished emission: its region bodies are exactly the
/// `entry_pc + 1 .. exit_pc` spans of its `RegionProof`s (`region_spans`).
pub fn verify_dispatcher_emitted(emitted: &Emitted) -> Result<Stats, String> {
    verify_dispatcher(&emitted.program, &region_spans(&emitted.proof.regions))
}

/// How the walk treats a hazard (a use, WAR or WAW whose wait is missing on a path).
#[derive(Clone, Copy)]
enum Mode<'a> {
    /// Every hazard is a finding; the first stops the walk.
    Strict,
    /// Hazards at a pc inside `body` are not reported (only counted); every other hazard
    /// is collected and the walk goes on, so a later dispatcher hazard is found even when
    /// a body has findings.
    Dispatcher { body: &'a [Range<usize>] },
}

/// Check that every instruction of `program` has every wait it needs on every
/// control-flow path from the entry (module docs). `foreign` are the instruction
/// ranges of foreign text, decoded through the lift's instruction table. `Err`
/// names the first violation found, or the bound that stopped the walk.
pub fn verify(program: &Program, foreign: &[Range<usize>]) -> Result<Stats, String> {
    walk(program, foreign, Mode::Strict)
}

/// Prove the **dispatcher** of `program` and nothing of its bodies: `body` are the
/// instruction ranges of the hosted bodies (foreign lifted text, decoded as in [`verify`]).
/// The walk is the same as [`verify`]'s, so every body's memory operations and waits are
/// tracked and their pending state reaches the dispatcher code after the body exactly as
/// it would at run time; the only difference is that a hazard at a pc inside one of the
/// `body` ranges is not reported (it is counted in [`Stats::suppressed`]). Every other
/// hazard, anywhere outside those exact spans (including a region's entry and exit
/// labels and the drains around them), is fatal with no allowance, and all of them are
/// reported, not only the first. Decode errors, indirect control flow and the walk's
/// bounds are errors everywhere, bodies included.
///
/// `Ok` therefore says nothing about the bodies' own waits: use [`verify`] for those.
///
/// A region whose exit is proven to leave every counter drained (`Stats::summarized`) is not
/// walked, so its body hazards are not even counted in `Stats::suppressed`.
pub fn verify_dispatcher(program: &Program, body: &[Range<usize>]) -> Result<Stats, String> {
    walk(program, body, Mode::Dispatcher { body })
}

/// [`verify`] under the static edge decisions of a typed control analysis
/// (`ledger_control::EdgeChoice`, one per instruction of `program`, labels included, and the
/// edge-conditioned `ledger_control::EntryProof`s).
/// Crate-private and strict only: a `Taken`/`Fallthrough` entry on a conditional
/// `s_cbranch_scc0/scc1` follows that edge alone (target with the current ledger, or the next
/// instruction with the current ledger) and is sound only if the producer proved it for every
/// state entering the branch's basic block (`cdna::lifted::ledger_program`). An `EntryProof`
/// does the same for the terminal branch of a labelled block, only for walks that reached the
/// label over its named edge, and is sound only if the producer proved it for every execution
/// crossing that edge. Everything else is the same walk: the same memory, wait, key and cap
/// rules, the same ledger half of the label dedup key (`Ledger::replay_state_key`, which
/// depends on the instruction, its edge choice and the pending state only), the same target
/// validation of an impossible edge. `choices` of the wrong length, a proof on a label or any
/// non-eligible instruction, or an `EntryProof` whose edge, label or branch is not exactly
/// what it names (`check_branch_proofs`) is an error. All `Both` and no entries is exactly
/// [`verify`].
pub(crate) fn verify_with_branch_proofs(
    program: &Program,
    foreign: &[Range<usize>],
    choices: &[EdgeChoice],
    entries: &[EntryProof],
) -> Result<Stats, String> {
    walk_with(
        program,
        foreign,
        Mode::Strict,
        Ledger::replay_state_key,
        Some(choices),
        entries,
        true,
    )
}

/// The bit of `counter` in a may-pending set.
fn counter_bit(counter: Counter) -> u8 {
    match counter {
        Counter::Load => 1,
        Counter::Store => 2,
        Counter::Ds => 4,
        Counter::Km => 8,
        Counter::Vm => 16,
        Counter::Vs => 32,
        Counter::Lgkm => 64,
        Counter::Exp => 128,
    }
}

/// Every counter that a recorded operation can occupy.
const COUNTERS: [Counter; 8] = [
    Counter::Load,
    Counter::Store,
    Counter::Ds,
    Counter::Km,
    Counter::Vm,
    Counter::Vs,
    Counter::Lgkm,
    Counter::Exp,
];

/// Whether the region whose body is `span` (`entry_pc + 1 .. exit_pc`, the entry label at
/// `span.start - 1` and the exit label at `span.end`) leaves every memory counter at zero on
/// every path that reaches its exit label, whatever was in flight at its entry.
///
/// A finite forward may-pending dataflow over the body's exact CFG: at the entry every
/// counter is potentially pending; a memory operation sets its counter's bit (taken from the
/// ledger's own classification of the decoded instruction); a wait of count zero clears the
/// bit of its counter (a nonzero wait and `depctr_vm_vsrc` clear nothing: conservative);
/// joins are bitwise OR, loops reach a fixed point since a state only grows. Every branch
/// must land inside the body or on the exit label, and nothing may be undecodable or
/// indirect; any such failure, or a nonzero set at the exit label, is `false` (no summary,
/// the caller walks the body instead). It proves only that the emitted waits retire any
/// incoming and body operation on every exiting path; it says nothing about the body's uses.
fn region_exit_is_drained(
    insns: &[Instruction],
    decoded: &HashMap<usize, Instruction>,
    labels: &HashMap<&str, usize>,
    arch: Arch,
    span: &Range<usize>,
) -> bool {
    let (start, end) = (span.start, span.end);
    if start == 0
        || start > end
        || end >= insns.len()
        || is_label(insns[start - 1].text.trim()).is_none()
        || is_label(insns[end].text.trim()).is_none()
    {
        return false;
    }
    let mut may: Vec<Option<u8>> = vec![None; end - start + 1];
    // Every counter the architecture has may hold an incoming operation (`capacity` is `Some`
    // exactly for those); a bit of a counter it lacks could never be cleared by a wait.
    let entry_bits = COUNTERS
        .into_iter()
        .filter(|&counter| capacity(arch, counter).is_some())
        .fold(0u8, |bits, counter| bits | counter_bit(counter));
    may[0] = Some(entry_bits);
    let mut work = vec![start];
    while let Some(pc) = work.pop() {
        if pc == end {
            continue;
        }
        let Some(bits) = may[pc - start] else {
            continue;
        };
        let text = insns[pc].text.trim();
        let mut out = bits;
        let mut successors: Vec<usize> = Vec::new();
        if is_label(text).is_some() {
            successors.push(pc + 1);
        } else {
            let insn = decoded.get(&pc).unwrap_or(&insns[pc]);
            let mnemonic = insn.mnemonic();
            if mnemonic.starts_with("s_endpgm") {
                continue;
            }
            if let Some(target) = sync::branch_target(text) {
                let Some(&target_pc) = labels.get(target) else {
                    return false;
                };
                successors.push(target_pc);
                if mnemonic != "s_branch" {
                    successors.push(pc + 1);
                }
            } else {
                if indirect(mnemonic) {
                    return false;
                }
                if text.starts_with("s_wait") {
                    let Ok(effects) = decode(text, arch) else {
                        return false;
                    };
                    for effect in effects {
                        if let Effect::Wait(counter, 0) = effect {
                            out &= !counter_bit(counter);
                        }
                    }
                } else if insn.memory.is_some() {
                    let mut probe = Ledger::default();
                    probe.record(arch, insn);
                    for counter in COUNTERS {
                        if probe.has_pending(counter) {
                            out |= counter_bit(counter);
                        }
                    }
                }
                successors.push(pc + 1);
            }
        }
        for next in successors {
            if next < start || next > end {
                return false;
            }
            let slot = &mut may[next - start];
            let merged = slot.unwrap_or(0) | out;
            if *slot != Some(merged) {
                *slot = Some(merged);
                work.push(next);
            }
        }
    }
    may[end - start].is_none_or(|bits| bits == 0)
}

fn walk(program: &Program, foreign: &[Range<usize>], mode: Mode) -> Result<Stats, String> {
    walk_with(
        program,
        foreign,
        mode,
        Ledger::replay_state_key,
        None,
        &[],
        true,
    )
}

/// Why `choices` and `entries` cannot be the proof tables of `program`: `choices` must have
/// exactly one entry per instruction and only an eligible conditional branch
/// (`s_cbranch_scc0/scc1` naming a label) may carry a proof, so labels and every other
/// instruction are `Both`. Each [`EntryProof`] must name exactly what the walk will find:
/// a label at `label`; a conditional `s_cbranch_scc0/scc1` at `branch` that is `Both` in
/// `choices`, with no label, branch or `s_endpgm` strictly between them (one basic block);
/// an edge `from` -> `label` that really exists (the taken edge of the branch at `from`
/// naming that label, or a fall-through from the instruction before the label, which must
/// not end control flow); a `Taken`/`Fallthrough` direction; and each edge at most once.
fn check_branch_proofs(
    program: &Program,
    choices: &[EdgeChoice],
    entries: &[EntryProof],
) -> Result<(), String> {
    let insns = &program.instructions;
    if choices.len() != insns.len() {
        return Err(format!(
            "branch proof map has {} entries for {} instructions (one per instruction, labels \
             included)",
            choices.len(),
            insns.len()
        ));
    }
    let eligible = |pc: usize| {
        let text = insns[pc].text.trim();
        is_label(text).is_none()
            && matches!(insns[pc].mnemonic(), "s_cbranch_scc0" | "s_cbranch_scc1")
            && sync::branch_target(text).is_some()
    };
    for (pc, (choice, insn)) in choices.iter().zip(insns).enumerate() {
        if *choice != EdgeChoice::Both && !eligible(pc) {
            return Err(format!(
                "branch proof map: pc {pc} `{}` carries {choice:?} but is not a conditional \
                 s_cbranch_scc0/scc1 branch",
                insn.text.trim()
            ));
        }
    }
    let mut edges: HashSet<(usize, bool)> = HashSet::new();
    for entry in entries {
        let EntryProof {
            from,
            taken,
            label,
            branch,
            choice,
        } = *entry;
        let bad = |why: &str| Err(format!("branch entry proof {entry:?}: {why}"));
        if choice == EdgeChoice::Both {
            return bad("a proof must select an edge");
        }
        if from >= insns.len() || label >= insns.len() || branch >= insns.len() {
            return bad("a pc is outside the program");
        }
        let Some(name) = is_label(insns[label].text.trim()) else {
            return bad("`label` is not a label");
        };
        if !eligible(branch) || label >= branch {
            return bad("`branch` is not a conditional s_cbranch_scc0/scc1 after the label");
        }
        if choices[branch] != EdgeChoice::Both {
            return bad("the branch already carries a block-local proof");
        }
        for (pc, insn) in insns.iter().enumerate().take(branch).skip(label + 1) {
            let text = insn.text.trim();
            if is_label(text).is_some()
                || sync::branch_target(text).is_some()
                || insn.mnemonic().starts_with("s_endpgm")
                || indirect(insn.mnemonic())
            {
                return bad(&format!("pc {pc} `{text}` ends or splits the block"));
            }
        }
        let real_edge = if taken {
            sync::branch_target(insns[from].text.trim()) == Some(name)
        } else {
            let mnemonic = insns[from].mnemonic();
            from + 1 == label
                && mnemonic != "s_branch"
                && !mnemonic.starts_with("s_endpgm")
                && !indirect(mnemonic)
        };
        if !real_edge {
            return bad("`from` -> `label` is not a control-flow edge of the program");
        }
        if !edges.insert((from, taken)) {
            return bad("the edge already has a proof");
        }
    }
    Ok(())
}

/// Registers observed (read OR written, irrespective of EXEC) before a full VM drain.
/// RegRef's u8 base/length can describe up to 510 dwords; keep all of that domain,
/// with disjoint scalar, vector and accumulator banks.
#[derive(Clone, Copy, Default)]
struct VmObservers([u64; 24]);

impl VmObservers {
    fn bank(reg: RegRef) -> usize {
        match reg.kind {
            Kind::V => 0,
            Kind::S => 8,
            Kind::A => 16,
        }
    }

    fn insert(&mut self, reg: RegRef) {
        let bank = Self::bank(reg);
        for r in usize::from(reg.base)..usize::from(reg.base) + usize::from(reg.len) {
            self.0[bank + r / 64] |= 1 << (r % 64);
        }
    }

    fn observes(&self, reg: RegRef) -> bool {
        let bank = Self::bank(reg);
        (usize::from(reg.base)..usize::from(reg.base) + usize::from(reg.len))
            .any(|r| self.0[bank + r / 64] & (1 << (r % 64)) != 0)
    }

    fn union(&mut self, other: Self) {
        for (word, incoming) in self.0.iter_mut().zip(other.0) {
            *word |= incoming;
        }
    }
}

/// Continuation equivalence for incoming gfx942 VM *loads*. A successful entry
/// proves that EVERY syntactic path reaches vmcnt(0), before termination, unknown
/// control flow, or a non-load VM issue; it lists every pre-drain register read/write.
/// No branch predicate is assumed. Backward must-dataflow starts only at drains:
/// a cycle that can avoid the drain never acquires a proof. Each node settles once.
///
/// If all incoming VM loads define unobserved registers, their queue cannot affect
/// this continuation: new VM loads share Gfx9Vmem's order, and their younger counts
/// and issue-bound retirement are independent of the older incoming prefix.
/// Partial waits can only retire that prefix or the same suffix of new loads.
/// The full drain then makes both executions identical. Other counters are never
/// projected. This changes label keys only, not the ledger or any transition.
fn vm_drain_observers(
    program: &Program,
    decoded: &HashMap<usize, Instruction>,
    labels: &HashMap<&str, usize>,
) -> Vec<Option<VmObservers>> {
    let insns = &program.instructions;
    let n = insns.len();
    let mut successors = vec![[None; 2]; n];
    let mut predecessors = vec![Vec::new(); n];
    let mut local = vec![VmObservers::default(); n];
    let mut result = vec![None; n];
    let mut work = Vec::new();
    for (pc, emitted) in insns.iter().enumerate() {
        let insn = decoded.get(&pc).unwrap_or(emitted);
        let text = emitted.text.trim();
        let mnemonic = insn.mnemonic();
        if text.starts_with("s_wait") {
            let Ok(effects) = decode(text, program.arch) else {
                continue;
            };
            if effects.iter().any(|e| *e == Effect::Wait(Counter::Vm, 0)) {
                result[pc] = Some(VmObservers::default());
                work.push(pc);
                continue;
            }
        } else if mnemonic.starts_with("s_endpgm") || indirect(mnemonic) {
            continue;
        } else if is_label(text).is_none() && sync::branch_target(text).is_none() {
            if decode(text, program.arch).is_err() {
                continue;
            }
            if matches!(
                insn.memory,
                Some(
                    MemoryClass::VmemStore
                        | MemoryClass::VmemAtomic { .. }
                        | MemoryClass::Flat { .. }
                )
            ) {
                continue;
            }
            for reg in insn.defs.iter().chain(&insn.uses) {
                local[pc].insert(*reg);
            }
        }
        if let Some(target) = sync::branch_target(text) {
            let Some(&target) = labels.get(target) else {
                continue;
            };
            successors[pc][0] = Some(target);
            if mnemonic != "s_branch" {
                // A missing fallthrough must not become a one-successor proof.
                if pc + 1 == n {
                    successors[pc] = [None; 2];
                    continue;
                }
                successors[pc][1] = Some(pc + 1);
            }
        } else if pc + 1 < n {
            successors[pc][0] = Some(pc + 1);
        }
        for next in successors[pc].iter().flatten() {
            predecessors[*next].push(pc);
        }
    }
    while let Some(pc) = work.pop() {
        for &pred in &predecessors[pc] {
            if result[pred].is_some() {
                continue;
            }
            let mut observed = local[pred];
            let mut ready = true;
            for next in successors[pred].iter().flatten() {
                match result[*next] {
                    Some(incoming) => observed.union(incoming),
                    None => {
                        ready = false;
                        break;
                    }
                }
            }
            if ready {
                result[pred] = Some(observed);
                work.push(pred);
            }
        }
    }
    result
}

/// [`walk`] with the label-dedup key and the optional static branch proofs as parameters:
/// production passes `Ledger::replay_state_key` (and proofs only through
/// [`verify_with_branch_proofs`]); the tests pass the un-canonicalized `full_state_key` to
/// compare outcomes against the baseline keying, with the same proofs in both walks.
/// `entries` are meaningful only with `choices` (validated together, strict only).
fn walk_with(
    program: &Program,
    foreign: &[Range<usize>],
    mode: Mode,
    state_key: fn(&Ledger) -> StateKey,
    choices: Option<&[EdgeChoice]>,
    entries: &[EntryProof],
    abstract_vm: bool,
) -> Result<Stats, String> {
    if choices.is_some() || !entries.is_empty() {
        if !matches!(mode, Mode::Strict) {
            return Err("branch proofs are strict-only: the dispatcher walk takes none".into());
        }
        check_branch_proofs(
            program,
            choices.ok_or("branch entry proofs need the block-local proof map")?,
            entries,
        )?;
    }
    // (from pc, taken) -> the proof of the label the edge enters.
    let entry_proofs: HashMap<(usize, bool), EntryProof> =
        entries.iter().map(|e| ((e.from, e.taken), *e)).collect();
    let insns = &program.instructions;
    let arch = program.arch;
    let labels: HashMap<&str, usize> = insns
        .iter()
        .enumerate()
        .filter_map(|(pc, i)| is_label(i.text.trim()).map(|l| (l, pc)))
        .collect();
    let mut decoded: HashMap<usize, Instruction> = HashMap::new();
    for range in foreign {
        for pc in range.clone().take_while(|&pc| pc < insns.len()) {
            let text = insns[pc].text.trim();
            if !handled_by_text(text) {
                let insn =
                    enrich(text, arch).map_err(|e| format!("pc {pc} foreign `{text}`: {e}"))?;
                decoded.insert(pc, insn);
            }
        }
    }
    let vm_observers = if abstract_vm && arch == Arch::Gfx942 && matches!(mode, Mode::Strict) {
        vm_drain_observers(program, &decoded, &labels)
    } else {
        Vec::new()
    };
    // Entry-proof directions remain distinct even when incoming VM queues are equivalent.
    let mut seen: HashMap<usize, HashSet<(StateKey, Option<EdgeChoice>)>> = HashMap::new();
    let mut witness = Witness::default();
    let mut stats = Stats::default();
    // Dispatcher mode: the hazards reported (deduplicated per pc and need), their messages,
    // and the distinct hazards left unreported inside the body spans.
    let mut reported: HashSet<(usize, String)> = HashSet::new();
    let mut collected: Vec<String> = Vec::new();
    let mut suppressed: HashSet<(usize, String)> = HashSet::new();
    // Dispatcher mode: a region whose exit is proven drained on every path (see
    // `region_exit_is_drained`) is not walked; the walk goes from its entry label to its exit
    // label with an empty ledger. An unproven region is walked as before.
    let mut summaries: HashMap<usize, usize> = HashMap::new();
    if let Mode::Dispatcher { body } = mode {
        for span in body {
            if region_exit_is_drained(insns, &decoded, &labels, arch, span) {
                summaries.insert(span.start - 1, span.end);
            }
        }
    }
    stats.summarized = summaries.len();
    // (pc, ledger, witness edge, the branch pc whose taken edge this state follows).
    type Pending = (usize, Ledger, Option<usize>, Option<usize>);
    let mut stack: Vec<Pending> = vec![(0, Ledger::default(), None, None)];
    while let Some((mut pc, mut ledger, mut path, mut via)) = stack.pop() {
        // The label this walk last passed: the block a violation sits in.
        let mut block: &str = "entry";
        // The `EntryProof` of the edge this walk crossed into the current block, as
        // (its branch pc, direction); consumed by that branch.
        let mut refined: Option<(usize, EdgeChoice)> = None;
        while let Some(emitted) = insns.get(pc) {
            let insn = decoded.get(&pc).unwrap_or(emitted);
            stats.steps += 1;
            if stats.steps > MAX_STEPS {
                return Err(format!(
                    "wait verification exceeded {MAX_STEPS} instruction visits (path-sensitive \
                     state explosion); drain the memory counters before branches that fan out \
                     or loop back so fewer in-flight states reach each label"
                ));
            }
            let text = emitted.text.trim();
            if let Some(label) = is_label(text) {
                // The contiguous wait-only prefix right after the label (no use, def or
                // branch between the label and these waits, and the next label or any other
                // instruction ends it, so a branch target is never inside it) runs on every
                // path that reaches the label. Apply it first, counting its steps, and
                // dedup on the state after it: distinct in-flight states that the prefix
                // collapses (a full drain) are one state, and the walk continues after the
                // prefix with exactly the ledger it would have had passing through it.
                let mut end = pc + 1;
                while let Some(next) = insns.get(end) {
                    let next_text = next.text.trim();
                    if !next_text.starts_with("s_wait") || is_label(next_text).is_some() {
                        break;
                    }
                    let effects = decode(next_text, arch).map_err(|e| format!("pc {end} {e}"))?;
                    for effect in effects {
                        match effect {
                            Effect::Wait(counter, count) => ledger.wait_replay(counter, count),
                            Effect::ReleaseStoreSources => ledger.release_store_sources(),
                        }
                    }
                    stats.steps += 1;
                    end += 1;
                }
                // The edge just crossed: the taken edge of the branch at `via`, else the
                // fall-through from the previous instruction (the program entry has none).
                let edge = match via.take() {
                    Some(from) => Some((from, true)),
                    None => pc.checked_sub(1).map(|from| (from, false)),
                };
                let entry = edge.and_then(|e| entry_proofs.get(&e));
                let states = seen.entry(pc).or_default();
                let mut key = state_key(&ledger);
                if vm_observers
                    .get(end)
                    .and_then(Option::as_ref)
                    .is_some_and(|observed| {
                        ledger.vmem_loads_unobserved(|reg| observed.observes(reg))
                    })
                {
                    key.retain(|entry| entry.0 != Counter::Vm);
                }
                if !states.insert((key, entry.map(|e| e.choice))) {
                    break;
                }
                stats.label_states += 1;
                if states.len() > MAX_STATES_PER_LABEL {
                    return Err(format!(
                        "wait verification: more than {MAX_STATES_PER_LABEL} distinct in-flight \
                         memory states reach label `{label}` (path-sensitive state explosion); \
                         wait the counters before the branches that reach it"
                    ));
                }
                refined = entry.map(|e| (e.branch, e.choice));
                block = label;
                pc = match summaries.get(&pc) {
                    Some(&exit) => {
                        ledger = Ledger::default();
                        exit
                    }
                    None => end,
                };
                continue;
            }
            let mnemonic = insn.mnemonic();
            if mnemonic.starts_with("s_endpgm") {
                break;
            }
            if let Some(target) = sync::branch_target(text) {
                let target_pc = *labels.get(target).ok_or_else(|| {
                    format!("pc {pc} `{text}`: branch to label `{target}`, which is never placed")
                })?;
                if mnemonic == "s_branch" {
                    path = witness.push(path, pc, true);
                    via = Some(pc);
                    pc = target_pc;
                } else {
                    // The static proof (`verify_with_branch_proofs`) selects one edge only
                    // when it holds for every state entering the branch's block; the ledger
                    // is never edited, so every retained edge is the old transition.
                    let local = choices.map_or(EdgeChoice::Both, |c| c[pc]);
                    // The proof of the edge this block was entered over, if any, belongs to
                    // exactly this branch (`check_branch_proofs`); the block-local choice
                    // applies to every other arrival.
                    let entered = refined.take().filter(|(b, _)| *b == pc).map(|(_, c)| c);
                    match entered.unwrap_or(local) {
                        EdgeChoice::Both => {
                            let taken = witness.push(path, pc, true);
                            stack.push((target_pc, ledger.clone(), taken, Some(pc)));
                            path = witness.push(path, pc, false);
                            pc += 1;
                        }
                        EdgeChoice::Taken => {
                            path = witness.push(path, pc, true);
                            via = Some(pc);
                            pc = target_pc;
                        }
                        EdgeChoice::Fallthrough => {
                            path = witness.push(path, pc, false);
                            pc += 1;
                        }
                    }
                }
                continue;
            }
            if indirect(mnemonic) {
                return Err(format!(
                    "pc {pc} `{text}`: control flow the wait verifier cannot follow \
                     (indirect branch, call or return)"
                ));
            }
            let effects = decode(text, arch).map_err(|e| format!("pc {pc} {e}"))?;
            if text.starts_with("s_wait") {
                for effect in effects {
                    match effect {
                        Effect::Wait(counter, count) => ledger.wait_replay(counter, count),
                        Effect::ReleaseStoreSources => ledger.release_store_sources(),
                    }
                }
                pc += 1;
                continue;
            }
            // A required count at or above the counter's maximum is satisfied without a
            // wait: the counter cannot hold the operation plus that many younger ones, so
            // the operation has retired (`Ledger::retire_count` reports the exact count up
            // to 63 younger on RDNA). The pending operations stay in the ledger.
            let mut unmet: Vec<_> = ledger
                .required_replay(insn)
                .into_iter()
                .filter(|(counter, count, _)| {
                    capacity(arch, *counter).is_none_or(|cap| *count < cap)
                })
                .collect();
            // CDNA `s_barrier` is not a memory wait (CDNA3 ISA §4.3 p.19; `ledger` module docs):
            // it only synchronises the workgroup's waves. No DS operation of this wave may
            // still be pending across it: a pending write is an LDS RAW for the waves released
            // by the barrier, a pending read an LDS WAR against their next write. Which LDS
            // addresses an operation touches is unknown, so every pending DS operation counts
            // (the builder's `barrier` drains the same set). Global-memory visibility is not
            // modelled here.
            if arch.cdna().is_some()
                && mnemonic == "s_barrier"
                && ledger
                    .shape()
                    .iter()
                    .any(|entry| entry.4 == MemoryFamily::Ds)
            {
                unmet.push((Counter::Lgkm, 0, Reason::Barrier));
            }
            if !unmet.is_empty() {
                match mode {
                    Mode::Strict => {
                        let (counter, count, reason) = &unmet[0];
                        return Err(format!(
                            "pc {pc} `{text}` (after label `{block}`): needs {counter:?} wait {count} \
                             ({reason:?}) but a path reaches it without one: {}",
                            witness.describe(path, insns)
                        ));
                    }
                    Mode::Dispatcher { body } => {
                        let in_body = body.iter().any(|span| span.contains(&pc));
                        for (counter, count, reason) in &unmet {
                            let need = format!("{counter:?} wait {count} ({reason:?})");
                            if in_body {
                                suppressed.insert((pc, need));
                            } else if reported.insert((pc, need.clone())) {
                                collected.push(format!(
                                    "pc {pc} `{text}` (after label `{block}`): needs {need} but a \
                                     path reaches it without one: {}",
                                    witness.describe(path, insns)
                                ));
                            }
                        }
                    }
                }
            }
            ledger.record(arch, insn);
            if insn.memory.is_some() {
                // The architectural issue bound (module docs): the oldest operation of an
                // ordered same-class queue has retired when the one past the counter field's
                // maximum issues; mixed or out-of-order queues are never clipped.
                ledger.clip_to_issue_bound(|counter| capacity(arch, counter));
            }
            if insn.memory.is_some() {
                if let Some((counter, depth)) = ledger.deepest_counter() {
                    if depth > COUNTER_CAP {
                        return Err(format!(
                            "pc {pc} `{text}` (after label `{block}`): {depth} pending {counter:?} \
                             operations on one path (more than {COUNTER_CAP}; hardware holds at \
                             most 63): code or a loop issues memory operations without waiting, \
                             so in-flight state is unbounded; add a wait or drain on that path: {}",
                            witness.describe(path, insns)
                        ));
                    }
                }
            }
            pc += 1;
        }
    }
    if !collected.is_empty() {
        return Err(format!(
            "{} dispatcher hazard(s) outside the body spans:\n{}",
            collected.len(),
            collected.join("\n")
        ));
    }
    stats.suppressed = suppressed.len();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(base: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base,
            len: 1,
        }
    }
    fn s(base: u8) -> RegRef {
        RegRef {
            kind: Kind::S,
            base,
            len: 1,
        }
    }
    fn ins(text: &str) -> Instruction {
        Instruction::new(text, vec![], vec![])
    }
    fn load(dst: u8) -> Instruction {
        Instruction::new(
            format!("global_load_b32 v{dst}, v0, s[0:1]"),
            vec![v(dst)],
            vec![v(0)],
        )
        .memory(MemoryClass::VmemLoad)
    }
    fn store(src: u8) -> Instruction {
        Instruction::new(
            format!("global_store_b32 v0, v{src}, s[0:1]"),
            vec![],
            vec![v(0), v(src)],
        )
        .memory(MemoryClass::VmemStore)
    }
    fn mov(dst: u8, src: u8) -> Instruction {
        Instruction::new(
            format!("v_mov_b32 v{dst}, v{src}"),
            vec![v(dst)],
            vec![v(src)],
        )
    }
    fn program(arch: Arch, insns: Vec<Instruction>) -> Program {
        Program {
            arch,
            instructions: insns,
        }
    }
    fn gfx12(insns: Vec<Instruction>) -> Program {
        program(Arch::Gfx1201, insns)
    }
    /// Builder-emitted code: every instruction carries its own metadata.
    fn check(p: &Program) -> Result<Stats, String> {
        verify(p, &[])
    }

    #[test]
    fn waited_straight_line_and_loop_pass() {
        let p = gfx12(vec![
            load(1),
            ins("s_wait_loadcnt 0x0"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
        // A loop that waits before it uses, reached by a forward fall-through and a
        // back edge: one state at the head, ids advancing per iteration do not grow it.
        let p = gfx12(vec![
            ins("H:"),
            load(1),
            ins("s_wait_loadcnt 0x0"),
            mov(3, 1),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        let stats = check(&p).unwrap();
        assert_eq!(stats.label_states, 1, "{stats:?}");
    }

    /// A load on a dead fall-through path (after an unconditional branch, no label
    /// reaches it) is not pending at a label whose only predecessor is a branch:
    /// the builder's linear ledger would carry it there; the CFG walk does not.
    #[test]
    fn ghost_load_on_unreachable_path_is_not_pending() {
        let p = gfx12(vec![
            ins("s_cbranch_scc1 L"),
            ins("s_branch END"),
            load(1),
            ins("L:"),
            mov(3, 1),
            ins("END:"),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
        // Two arms issue different loads and wait each; neither arm's load is
        // pending at the join (no `maybe` entries, so no drain is demanded either).
        let p = gfx12(vec![
            ins("s_cbranch_scc1 B"),
            load(1),
            ins("s_wait_loadcnt 0x0"),
            ins("s_branch J"),
            ins("B:"),
            load(4),
            ins("s_wait_loadcnt 0x0"),
            ins("J:"),
            mov(3, 1),
            mov(6, 4),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
    }

    #[test]
    fn wait_missing_on_one_branch_path_is_rejected_with_its_witness() {
        let p = gfx12(vec![
            load(1),
            ins("s_cbranch_scc1 SKIP"),
            ins("s_wait_loadcnt 0x0"),
            ins("SKIP:"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        let e = check(&p).unwrap_err();
        assert!(e.contains("v_mov_b32 v3, v1"), "{e}");
        assert!(e.contains("Load"), "{e}");
        assert!(e.contains("SKIP"), "{e}");
        assert!(e.contains("s_cbranch_scc1 SKIP` taken"), "{e}");
    }

    #[test]
    fn back_edge_carries_the_pending_load_into_the_head() {
        // The first iteration enters the head with nothing in flight; the second
        // enters it with the load issued after the use.
        let p = gfx12(vec![
            ins("H:"),
            mov(3, 1),
            load(1),
            ins("s_cbranch_scc1 H"),
            ins("s_wait_loadcnt 0x0"),
            ins("s_endpgm"),
        ]);
        let e = check(&p).unwrap_err();
        assert!(e.contains("v_mov_b32 v3, v1"), "{e}");
        assert!(e.contains("s_cbranch_scc1 H` taken"), "{e}");
        // Waiting before the back edge fixes it.
        let p = gfx12(vec![
            ins("H:"),
            mov(3, 1),
            load(1),
            ins("s_wait_loadcnt 0x0"),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
    }

    #[test]
    fn war_on_a_store_source_needs_the_wait_on_every_path() {
        let p = gfx12(vec![
            store(1),
            ins("s_cbranch_scc1 L"),
            ins("s_wait_alu depctr_vm_vsrc(0)"),
            ins("L:"),
            mov(1, 5),
            ins("s_endpgm"),
        ]);
        assert!(check(&p).is_err());
        let p = gfx12(vec![
            store(1),
            ins("s_wait_alu depctr_vm_vsrc(0)"),
            ins("s_cbranch_scc1 L"),
            ins("L:"),
            mov(1, 5),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
    }

    #[test]
    fn waits_are_decoded_from_the_program_text() {
        // gfx12 combined wait retires both the load and the DS load.
        let ds = Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)])
            .memory(MemoryClass::DsLoad);
        let p = gfx12(vec![
            load(1),
            ds.clone(),
            ins("s_wait_loadcnt_dscnt 0x0"),
            mov(3, 1),
            mov(4, 2),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
        // Waiting the load to zero but leaving one DS op in flight keeps the (youngest)
        // DS load pending: its use is rejected.
        let p = gfx12(vec![
            load(1),
            ds,
            ins("s_wait_loadcnt_dscnt 0x0001"),
            mov(4, 2),
            ins("s_endpgm"),
        ]);
        assert!(check(&p).is_err());
        // gfx11 field waits and the store counter.
        let gfx11 = |w: &str| {
            program(
                Arch::Gfx1100,
                vec![load(1), ins(w), mov(3, 1), ins("s_endpgm")],
            )
        };
        check(&gfx11("s_waitcnt vmcnt(0) lgkmcnt(0)")).unwrap();
        assert!(check(&gfx11("s_waitcnt lgkmcnt(0)")).is_err());
        assert!(check(&gfx11("s_waitcnt_vscnt null, 0x0")).is_err());
        // Raw gfx11 immediate: VM[15:10]=0, LGKM[9:4]=0x3f (none), EXP[2:0]=7 (none).
        check(&gfx11("s_waitcnt 0x3f7")).unwrap();
    }

    /// The IR printer shows the SOPP field signed: `s_waitcnt -1017` is 0xfc07 (VM none, LGKM 0,
    /// EXP none), and a signed spelling outside 16 bits is an error.
    #[test]
    fn signed_raw_waitcnt_immediates_decode_as_their_sixteen_bits() {
        let ds = Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)])
            .memory(MemoryClass::DsLoad);
        let body = |w: &str| {
            program(
                Arch::Gfx1100,
                vec![ds.clone(), ins(w), mov(3, 2), ins("s_endpgm")],
            )
        };
        check(&body("s_waitcnt -1017")).unwrap();
        // 0xfc07 spelled unsigned is the same wait.
        check(&body("s_waitcnt 0xfc07")).unwrap();
        // -9 is 0xfff7: every field is the no-wait value, so the DS load stays pending; -1
        // (0xffff) sets the reserved bit 3.
        assert!(check(&body("s_waitcnt -9")).unwrap_err().contains("needs"));
        assert!(check(&body("s_waitcnt -1"))
            .unwrap_err()
            .contains("reserved"));
        let e = check(&body("s_waitcnt -40000")).unwrap_err();
        assert!(e.contains("invalid s_waitcnt immediate"), "{e}");
        let e = check(&body("s_waitcnt 0x10000")).unwrap_err();
        assert!(e.contains("invalid s_waitcnt immediate"), "{e}");
    }

    #[test]
    fn no_wait_sentinels_retire_nothing_and_malformed_waits_fail_closed() {
        let gfx11 = |w: &str| program(Arch::Gfx1100, vec![load(1), ins(w), mov(3, 1)]);
        // The all-ones field is the architectural "no wait": the load stays pending.
        assert!(check(&gfx11("s_waitcnt vmcnt(63)")).is_err());
        assert!(check(&gfx11("s_waitcnt 0xfff7")).is_err());
        // Out of range / reserved bits / no fields / unknown fields.
        for bad in [
            "s_waitcnt vmcnt(64)",
            "s_waitcnt expcnt(8)",
            "s_waitcnt 0x3fff",
            "s_waitcnt 0x10000",
            "s_waitcnt",
            "s_waitcnt foocnt(0)",
        ] {
            let e = check(&gfx11(bad)).unwrap_err();
            assert!(
                !e.contains("needs"),
                "{bad}: decode error expected, got {e}"
            );
        }
        let gfx12 = |w: &str| gfx12(vec![load(1), ins(w), mov(3, 1)]);
        assert!(check(&gfx12("s_wait_loadcnt 0x3f")).is_err()); // no wait
        let e = check(&gfx12("s_wait_loadcnt 0x40")).unwrap_err();
        assert!(e.contains("exceeds"), "{e}");
        assert!(check(&gfx12("s_wait_kmcnt 0x20"))
            .unwrap_err()
            .contains("exceeds"));
        assert!(check(&gfx12("s_wait_loadcnt_dscnt 0x4000"))
            .unwrap_err()
            .contains("reserved"));
        assert!(check(&gfx12("s_wait_frobnicate 0x0"))
            .unwrap_err()
            .contains("unknown wait"));
        assert!(check(&gfx12("s_waitcnt vmcnt(0)"))
            .unwrap_err()
            .contains("not a gfx12 wait"));
        // Documented inert forms are accepted and retire nothing.
        assert!(check(&gfx12("s_wait_alu 0x0"))
            .unwrap_err()
            .contains("needs"));
        // CDNA: maxima come from the arch caps (vmcnt 63, lgkmcnt 15).
        let cdna = |w: &str| program(Arch::Gfx942, vec![load(1), ins(w), mov(3, 1)]);
        check(&cdna("s_waitcnt vmcnt(0)")).unwrap();
        assert!(check(&cdna("s_waitcnt vmcnt(63)"))
            .unwrap_err()
            .contains("needs"));
        assert!(check(&cdna("s_waitcnt vmcnt(64)"))
            .unwrap_err()
            .contains("exceeds"));
        assert!(check(&cdna("s_waitcnt lgkmcnt(16)"))
            .unwrap_err()
            .contains("exceeds"));
    }

    /// A partial wait on a counter that mixes in-order DS with out-of-order SMEM
    /// retires nothing: completion order is unknown.
    #[test]
    fn mixed_smem_ds_partial_wait_does_not_clear_smem() {
        let smem = Instruction::new("s_load_b32 s4, s[0:1], 0x0", vec![s(4)], vec![])
            .memory(MemoryClass::SmemLoad);
        let ds = Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)])
            .memory(MemoryClass::DsLoad);
        let use_s = Instruction::new("s_mov_b32 s5, s4", vec![s(5)], vec![s(4)]);
        let body = |wait: &str| {
            program(
                Arch::Gfx1100,
                vec![
                    smem.clone(),
                    ds.clone(),
                    ins(wait),
                    use_s.clone(),
                    ins("s_endpgm"),
                ],
            )
        };
        // The builder's optimistic `Ledger::wait` would retire the older SMEM here.
        let e = check(&body("s_waitcnt lgkmcnt(1)")).unwrap_err();
        assert!(e.contains("s_mov_b32 s5, s4"), "{e}");
        check(&body("s_waitcnt lgkmcnt(0)")).unwrap();
    }

    /// gfx11: `buffer_load` and `global_load` share VMcnt in issue order, so a partial
    /// wait retires the older buffer loads behind younger global loads (the rows_x4
    /// weight/x pattern); SMEM with DS stays conservative (see the SMEM test).
    #[test]
    fn gfx11_buffer_and_global_loads_retire_in_order_on_vmcnt() {
        let buffer = |dst: u8| {
            Instruction::new(
                format!(
                    "buffer_load_b128 v[{dst}:{}], v0, s[4:7], s8 offen",
                    dst + 3
                ),
                vec![RegRef {
                    kind: Kind::V,
                    base: dst,
                    len: 4,
                }],
                vec![v(0)],
            )
            .memory(MemoryClass::VmemLoad)
        };
        let global = |dst: u8| {
            Instruction::new(
                format!("global_load_b128 v[{dst}:{}], v0, s[0:1]", dst + 3),
                vec![RegRef {
                    kind: Kind::V,
                    base: dst,
                    len: 4,
                }],
                vec![v(0)],
            )
            .memory(MemoryClass::VmemLoad)
        };
        let use_v = |reg: u8| mov(100, reg);
        let body = |wait: &str, reg: u8| {
            let mut insns: Vec<Instruction> = (0..4).map(|k| buffer(8 + 4 * k)).collect();
            insns.extend((0..4).map(|k| global(40 + 4 * k)));
            insns.extend([ins(wait), use_v(reg), ins("s_endpgm")]);
            program(Arch::Gfx1100, insns)
        };
        // vmcnt(4) retires the four older buffer loads; the global loads stay pending.
        check(&body("s_waitcnt vmcnt(4)", 8)).unwrap();
        check(&body("s_waitcnt vmcnt(4)", 20)).unwrap();
        let e = check(&body("s_waitcnt vmcnt(4)", 40)).unwrap_err();
        assert!(e.contains("needs Vm wait"), "{e}");
        // vmcnt(5) leaves the youngest buffer load pending.
        assert!(check(&body("s_waitcnt vmcnt(5)", 20)).is_err());
    }

    /// Exactly the counter maximum (63) of younger loads in flight: the oldest has
    /// retired by capacity, so its use needs no wait; one fewer younger load leaves it
    /// pending. The younger loads stay pending (a use of one of them is still checked).
    #[test]
    fn required_count_at_the_counter_maximum_is_satisfied_without_a_wait() {
        let body = |younger: u8, tail: Instruction| {
            let mut insns = vec![load(1)];
            insns.extend((0..younger).map(|k| load(10 + k)));
            insns.extend([tail, ins("s_endpgm")]);
            gfx12(insns)
        };
        check(&body(63, mov(3, 1))).unwrap();
        let e = check(&body(62, mov(3, 1))).unwrap_err();
        assert!(e.contains("needs Load wait 62"), "{e}");
        // The youngest of the 63 is still pending: using it still needs a wait.
        let e = check(&body(63, mov(3, 10 + 62))).unwrap_err();
        assert!(e.contains("needs Load wait 0"), "{e}");
    }

    /// A returnless atomic: STOREcnt, never assumed in order, its sources locked until it retires.
    fn atomic(src: u8) -> Instruction {
        Instruction::new(
            format!("global_atomic_add_u32 v0, v{src}, s[0:1]"),
            vec![],
            vec![v(0), v(src)],
        )
        .memory(MemoryClass::VmemAtomic { returns: false })
    }

    /// Below the architectural issue bound the oldest pending operation's hazards are still
    /// checked; at it the oldest has retired and only it (the stores here use distinct
    /// source registers v1..).
    #[test]
    fn issue_bound_retires_exactly_the_oldest_of_an_ordered_queue() {
        // The oldest store locks v1; 10 younger stores do not hide it.
        let mut insns = vec![store(1)];
        insns.extend((0..10).map(|_| store(2)));
        insns.extend([mov(1, 5), ins("s_endpgm")]);
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("War"), "{e}");
        let stores = |n: u8, tail: Instruction| {
            let mut insns: Vec<Instruction> = (1..=n).map(store).collect();
            insns.extend([tail, ins("s_endpgm")]);
            gfx12(insns)
        };
        // 63 outstanding stores is the counter maximum: the oldest is still locked.
        let e = check(&stores(63, mov(1, 5))).unwrap_err();
        assert!(e.contains("War"), "{e}");
        // The 64th issues only after the oldest retired: its source is free, the second's
        // is still locked.
        check(&stores(64, mov(1, 5))).unwrap();
        let e = check(&stores(64, mov(2, 5))).unwrap_err();
        assert!(e.contains("War") && e.contains("v2"), "{e}");
        // Loads retire the same way (LOADcnt, one order class).
        let loads = |n: u8, use_reg: u8| {
            let mut insns: Vec<Instruction> = (1..=n).map(load).collect();
            insns.extend([mov(100, use_reg), ins("s_endpgm")]);
            gfx12(insns)
        };
        check(&loads(64, 1)).unwrap();
        assert!(check(&loads(64, 2)).is_err());
    }

    /// A mixed or out-of-order queue is never clipped: one atomic among in-order stores, or
    /// out-of-order SMEM past the KMcnt maximum, keeps its oldest operation pending.
    #[test]
    fn issue_bound_never_clips_mixed_or_out_of_order_queues() {
        let mut insns = vec![atomic(1)];
        insns.extend((2..=64).map(store));
        insns.extend([mov(1, 5), ins("s_endpgm")]);
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("War"), "{e}");
        // 32 SMEM loads (KMcnt maximum 31): out of order, so the first stays pending.
        let smem = |dst: u8| {
            Instruction::new(
                format!("s_load_b32 s{dst}, s[0:1], 0x0"),
                vec![s(dst)],
                vec![],
            )
            .memory(MemoryClass::SmemLoad)
        };
        let mut insns: Vec<Instruction> = (4..36).map(smem).collect();
        insns.extend([
            Instruction::new("s_mov_b32 s100, s4", vec![s(100)], vec![s(4)]),
            ins("s_endpgm"),
        ]);
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("needs Km wait"), "{e}");
    }

    /// What the issue bound does not bound still fails closed: more than the cap pending on
    /// one counter is refused, never truncated, when its queue is unclipped (atomics).
    #[test]
    fn unclipped_pending_depth_fails_closed() {
        let mut insns = vec![atomic(1)];
        insns.extend((0..=COUNTER_CAP).map(|_| atomic(2)));
        insns.extend([mov(1, 5), ins("s_endpgm")]);
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("pending") && e.contains("unbounded"), "{e}");
        // A loop issuing an atomic per iteration without waiting grows without bound.
        let p = gfx12(vec![
            ins("H:"),
            atomic(1),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        let e = check(&p).unwrap_err();
        assert!(e.contains("unbounded"), "{e}");
        // An in-order store per iteration is bounded by the issue bound: the queue reaches the
        // 63-deep maximum and stays there (the store only reads its source: no hazard).
        let p = gfx12(vec![
            ins("H:"),
            store(1),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
        // ... and still passes once the iteration waits.
        let p = gfx12(vec![
            ins("H:"),
            store(1),
            ins("s_wait_storecnt 0x0"),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
    }

    #[test]
    fn fan_out_beyond_the_state_bound_fails_closed() {
        // Each diamond either issues a store (distinct source register) or not, so
        // the number of distinct pending sets at the last label doubles per diamond.
        let mut insns = Vec::new();
        for k in 0..12u8 {
            insns.push(ins(&format!("s_cbranch_scc1 J{k}")));
            insns.push(store(10 + k));
            insns.push(ins(&format!("J{k}:")));
        }
        insns.push(ins("s_endpgm"));
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("state explosion"), "{e}");
    }

    /// More than `MAX_STATES_PER_LABEL` distinct states reach the last label: it fails closed
    /// unless the label starts with a wait-only prefix that drains them (the walk dedups on
    /// the state after that prefix), and a label that merely follows the prefix does not
    /// count as part of it.
    #[test]
    fn label_starting_with_a_drain_collapses_incoming_states() {
        // Ten diamonds: the intermediate labels hold at most 512 states, the last 1024.
        let diamonds = |tail: &[&str]| {
            let mut insns = Vec::new();
            for k in 0..10u8 {
                insns.push(ins(&format!("s_cbranch_scc1 J{k}")));
                insns.push(store(10 + k));
                insns.push(ins(&format!("J{k}:")));
            }
            insns.extend(tail.iter().map(|t| ins(t)));
            insns.push(ins("s_endpgm"));
            gfx12(insns)
        };
        let e = check(&diamonds(&[])).unwrap_err();
        assert!(e.contains("state explosion") && e.contains("J9"), "{e}");
        check(&diamonds(&["s_wait_storecnt 0x0"])).unwrap();
        // A prefix that does not drain the stores does not collapse the states.
        let e = check(&diamonds(&["s_wait_loadcnt 0x0"])).unwrap_err();
        assert!(e.contains("state explosion"), "{e}");
        // The prefix ends at the next label: a branch to that label skips the drain.
        let p = gfx12(vec![
            store(1),
            ins("s_cbranch_scc1 B"),
            ins("A:"),
            ins("s_wait_storecnt 0x0"),
            ins("B:"),
            mov(1, 5),
            ins("s_endpgm"),
        ]);
        let e = check(&p).unwrap_err();
        assert!(e.contains("War"), "{e}");
        // ... while the same drain right after B (its actual prefix) covers every path.
        let p = gfx12(vec![
            store(1),
            ins("s_cbranch_scc1 B"),
            ins("A:"),
            ins("B:"),
            ins("s_wait_alu depctr_vm_vsrc(0)"),
            mov(1, 5),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
    }

    #[test]
    fn gfx942_store_sources_are_not_counter_held_but_the_store_still_counts() {
        // A younger write to a pending store's data register: a WAR wait on gfx12 (and gfx11),
        // none on gfx942 (`Ledger::required_replay`).
        let overwrite = |arch| program(arch, vec![store(1), mov(1, 5), ins("s_endpgm")]);
        let e = check(&overwrite(Arch::Gfx1201)).unwrap_err();
        assert!(e.contains("War"), "{e}");
        check(&overwrite(Arch::Gfx942)).unwrap();
        // The source-side relaxation does not touch results: a pending load's destination
        // still needs its wait (WAW) and its uses (RAW).
        let waw = program(Arch::Gfx942, vec![load(1), mov(1, 5), ins("s_endpgm")]);
        assert!(check(&waw).unwrap_err().contains("Waw"));
        let raw = program(Arch::Gfx942, vec![load(1), mov(3, 1), ins("s_endpgm")]);
        assert!(check(&raw).unwrap_err().contains("Use"));
        // A DS store keeps its data register free to overwrite but still blocks a barrier.
        let ds_store = Instruction::new("ds_write_b32 v2, v3", vec![], vec![v(2), v(3)])
            .memory(MemoryClass::DsStore);
        let body = |tail: Vec<Instruction>| {
            let mut insns = vec![ds_store.clone(), mov(3, 5)];
            insns.extend(tail);
            insns.push(ins("s_endpgm"));
            program(Arch::Gfx942, insns)
        };
        check(&body(vec![])).unwrap();
        let e = check(&body(vec![ins("s_barrier")])).unwrap_err();
        assert!(e.contains("Barrier"), "{e}");
        check(&body(vec![ins("s_waitcnt lgkmcnt(0)"), ins("s_barrier")])).unwrap();
    }

    #[test]
    fn gfx942_barrier_requires_pending_ds_loads_and_stores_to_have_retired() {
        let ds_load = Instruction::new("ds_read_b32 v2, v0", vec![v(2)], vec![v(0)])
            .memory(MemoryClass::DsLoad);
        let ds_store = Instruction::new("ds_write_b32 v0, v3", vec![], vec![v(0), v(3)])
            .memory(MemoryClass::DsStore);
        for op in [ds_load, ds_store] {
            let body = |tail: &[&str]| {
                let mut insns = vec![op.clone()];
                insns.extend(tail.iter().map(|t| ins(t)));
                insns.extend([ins("s_barrier"), ins("s_endpgm")]);
                program(Arch::Gfx942, insns)
            };
            let e = check(&body(&[])).unwrap_err();
            assert!(e.contains("Lgkm") && e.contains("Barrier"), "{e}");
            check(&body(&["s_waitcnt lgkmcnt(0)"])).unwrap();
        }
        // Not a CDNA rule elsewhere: a text `s_barrier` on gfx12 is unaffected.
        let ds_store = Instruction::new("ds_store_b32 v0, v3", vec![], vec![v(0), v(3)])
            .memory(MemoryClass::DsStore);
        check(&gfx12(vec![ds_store, ins("s_barrier"), ins("s_endpgm")])).unwrap();
    }

    fn cdna(insns: Vec<Instruction>) -> Program {
        program(Arch::Gfx942, insns)
    }
    fn ds_load(dst: u8) -> Instruction {
        Instruction::new(format!("ds_read_b32 v{dst}, v0"), vec![v(dst)], vec![v(0)])
            .memory(MemoryClass::DsLoad)
    }
    fn ds_store(src: u8) -> Instruction {
        Instruction::new(
            format!("ds_write_b32 v0, v{src}"),
            vec![],
            vec![v(0), v(src)],
        )
        .memory(MemoryClass::DsStore)
    }
    fn smem_load(dst: u8) -> Instruction {
        Instruction::new(
            format!("s_load_dword s{dst}, s[0:1], 0x0"),
            vec![s(dst)],
            vec![],
        )
        .memory(MemoryClass::SmemLoad)
    }
    /// The baseline walk: label dedup on the un-canonicalized key (`Ledger::full_state_key`).
    fn walk_full(p: &Program) -> Result<Stats, String> {
        walk_with(
            p,
            &[],
            Mode::Strict,
            Ledger::full_state_key,
            None,
            &[],
            false,
        )
    }

    /// gfx942 `replay_state_key` drops only the registers a pending store read, which the gfx942
    /// replay never observes (`Ledger::required_replay`): paths that differ only in which store
    /// occupies a slot are one state. Ten diamonds that each store a different register leave
    /// 2^10 distinct baseline states at the last label (over the bound) and 11 canonical ones;
    /// a scatter-shaped loop with two conditional stores grows the baseline sequence without
    /// bound until the 63-slot issue clip while the canonical state is just the count. Every
    /// other architecture keeps the registers in the key and still fails closed.
    #[test]
    fn gfx942_stores_that_differ_only_in_sources_share_one_state() {
        let diamonds = |arch| {
            let mut insns = Vec::new();
            for k in 0..10u8 {
                insns.push(ins(&format!("s_cbranch_scc1 J{k}")));
                insns.push(store(10 + k));
                insns.push(ins(&format!("J{k}:")));
            }
            insns.push(ins("s_endpgm"));
            program(arch, insns)
        };
        let p = diamonds(Arch::Gfx942);
        let baseline = walk_full(&p).unwrap_err();
        assert!(
            baseline.contains("state explosion") && baseline.contains("J9"),
            "{baseline}"
        );
        let stats = check(&p).unwrap();
        assert!(stats.label_states <= 100, "{stats:?}");
        let e = check(&diamonds(Arch::Gfx1201)).unwrap_err();
        assert!(e.contains("state explosion") && e.contains("J9"), "{e}");

        let scatter = cdna(vec![
            ins("H:"),
            ins("s_cbranch_scc1 A"),
            store(5),
            ins("A:"),
            ins("s_cbranch_scc1 B"),
            store(6),
            ins("B:"),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        let baseline = walk_full(&scatter).unwrap_err();
        assert!(baseline.contains("state explosion"), "{baseline}");
        let stats = check(&scatter).unwrap();
        assert!(stats.label_states <= 3 * 64 + 3, "{stats:?}");
    }

    /// What the canonical key keeps on gfx942: a pending operation's position among the
    /// counter's operations (partial waits retire by position), the registers a load
    /// defines, and the order class of the counter's queue (DS with SMEM never retires by
    /// position). Each program must agree with the baseline keying.
    #[test]
    fn gfx942_canonical_key_keeps_position_defs_and_order_class() {
        let agree = |p: &Program| -> Result<Stats, String> {
            let new = check(p);
            let old = walk_full(p);
            assert_eq!(new.is_ok(), old.is_ok(), "{new:?} vs {old:?}");
            new
        };
        // Position: [store, load] vs [load, store] merge at J; `vmcnt(1)` retires the store in
        // the first and the load in the second, so the first still needs a wait for v1.
        let positions = |x: Vec<Instruction>, y: Vec<Instruction>, wait: &str| {
            let mut insns = vec![ins("s_cbranch_scc1 Y")];
            insns.extend(x);
            insns.push(ins("s_branch J"));
            insns.push(ins("Y:"));
            insns.extend(y);
            insns.extend([ins("J:"), ins(wait), mov(3, 1), ins("s_endpgm")]);
            cdna(insns)
        };
        let e = agree(&positions(
            vec![store(5), load(1)],
            vec![load(1), store(5)],
            "s_waitcnt vmcnt(1)",
        ))
        .unwrap_err();
        assert!(e.contains("Use"), "{e}");
        agree(&positions(
            vec![store(5), load(1)],
            vec![load(1), store(5)],
            "s_waitcnt vmcnt(0)",
        ))
        .unwrap();
        agree(&positions(
            vec![load(1), store(5)],
            vec![load(1), store(6)],
            "s_waitcnt vmcnt(1)",
        ))
        .unwrap();

        // A partial wait retires only what is certain on each path: one path drained, the
        // other did not, so `vmcnt(1)` leaves v2 pending on the path that skipped the drain.
        let partial = |used: u8| {
            cdna(vec![
                load(1),
                load(2),
                ins("s_cbranch_scc1 J"),
                ins("s_waitcnt vmcnt(0)"),
                ins("J:"),
                ins("s_waitcnt vmcnt(1)"),
                mov(4, used),
                ins("s_endpgm"),
            ])
        };
        let e = agree(&partial(2)).unwrap_err();
        assert!(e.contains("Use"), "{e}");
        agree(&partial(1)).unwrap();

        // A retained missing wait: the branch around the drain reaches the use.
        let skipped = cdna(vec![
            load(1),
            ins("s_cbranch_scc1 J"),
            ins("s_waitcnt vmcnt(0)"),
            ins("J:"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        let e = agree(&skipped).unwrap_err();
        assert!(
            e.contains("Use") && e.contains("s_cbranch_scc1 J` taken"),
            "{e}"
        );

        // Defs: two paths load different registers; the merged use of v2 is a RAW on one.
        let defs = |second: u8| {
            cdna(vec![
                ins("s_cbranch_scc1 Y"),
                load(1),
                ins("s_branch J"),
                ins("Y:"),
                load(second),
                ins("J:"),
                mov(3, 2),
                ins("s_endpgm"),
            ])
        };
        let e = agree(&defs(2)).unwrap_err();
        assert!(e.contains("Use"), "{e}");
        agree(&defs(1)).unwrap();

        // Order class: an SMEM load on one path makes the LGKM queue mixed there, so
        // `lgkmcnt(1)` retires nothing on that path while it retires ds_read v1 on the other.
        let mixed = |first: Vec<Instruction>| {
            let mut insns = vec![ins("s_cbranch_scc1 Y")];
            insns.extend(first);
            insns.extend([ins("s_branch J"), ins("Y:"), ins("J:")]);
            insns.extend([
                ds_load(1),
                ds_load(2),
                ins("s_waitcnt lgkmcnt(1)"),
                mov(3, 1),
                ins("s_endpgm"),
            ]);
            cdna(insns)
        };
        let e = agree(&mixed(vec![smem_load(10)])).unwrap_err();
        assert!(e.contains("Use"), "{e}");
        agree(&mixed(vec![])).unwrap();
    }

    /// The source locks stay in the key where they are observed: on gfx12 a store's data
    /// register is a WAR on the path that stored it, on gfx942 it is not (and merging the two
    /// stores there is exact).
    #[test]
    fn store_source_locks_stay_in_the_key_outside_gfx942() {
        let two_stores = |arch| {
            program(
                arch,
                vec![
                    ins("s_cbranch_scc1 Y"),
                    store(5),
                    ins("s_branch J"),
                    ins("Y:"),
                    store(6),
                    ins("J:"),
                    mov(6, 9),
                    ins("s_endpgm"),
                ],
            )
        };
        let e = check(&two_stores(Arch::Gfx1201)).unwrap_err();
        assert!(e.contains("War"), "{e}");
        check(&two_stores(Arch::Gfx942)).unwrap();
        walk_full(&two_stores(Arch::Gfx942)).unwrap();
    }

    /// Fixed-seed LCG for the corpus below.
    struct Lcg(u64);
    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }
    fn vm_wait(cdna: bool, k: u64) -> String {
        if cdna {
            format!("s_waitcnt vmcnt({k})")
        } else {
            format!("s_wait_loadcnt {k:#x}")
        }
    }
    fn ds_wait(cdna: bool, k: u64) -> String {
        if cdna {
            format!("s_waitcnt lgkmcnt({k})")
        } else {
            format!("s_wait_dscnt {k:#x}")
        }
    }
    /// A small random CFG: loads, stores, DS and SMEM operations, partial waits, movs,
    /// barriers, forward conditional branches to labels placed later and an optional back edge.
    fn corpus_program(rng: &mut Lcg, arch: Arch) -> Program {
        let is_cdna = arch == Arch::Gfx942;
        let mut insns = Vec::new();
        let looped = rng.below(4) == 0;
        if looped {
            insns.push(ins("H:"));
        }
        let mut open: Vec<u64> = Vec::new();
        let mut labels = 0u64;
        for _ in 0..(6 + rng.below(10)) {
            match rng.below(14) {
                0 | 1 => insns.push(load(1 + rng.below(4) as u8)),
                2..=4 => insns.push(store(5 + rng.below(4) as u8)),
                5 => {
                    let dst = 1 + rng.below(8) as u8;
                    let src = 1 + rng.below(8) as u8;
                    insns.push(mov(dst, src));
                }
                6 => insns.push(ins(&vm_wait(is_cdna, rng.below(3)))),
                7 => insns.push(ds_load(1 + rng.below(4) as u8)),
                8 => insns.push(ds_store(5 + rng.below(4) as u8)),
                9 => insns.push(ins(&ds_wait(is_cdna, rng.below(3)))),
                10 => insns.push(smem_load(10 + rng.below(2) as u8)),
                11 => insns.push(ins("s_barrier")),
                12 => {
                    insns.push(ins(&format!("s_cbranch_scc1 L{labels}")));
                    open.push(labels);
                    labels += 1;
                }
                _ => {
                    if !open.is_empty() {
                        let at = rng.below(open.len() as u64) as usize;
                        insns.push(ins(&format!("L{}:", open.remove(at))));
                    }
                }
            }
        }
        for label in open {
            insns.push(ins(&format!("L{label}:")));
        }
        if looped && rng.below(2) == 0 {
            insns.push(ins("s_cbranch_scc1 H"));
        }
        insns.push(ins("s_endpgm"));
        program(arch, insns)
    }

    /// The canonical key against the baseline on a deterministic corpus of random small CFGs
    /// (gfx942 and gfx12): wherever the baseline walk finishes within the state bound the
    /// two keyings prove and refuse exactly the same programs, canonical states never exceed
    /// the baseline's, and on gfx12 the state sets are identical. Programs whose baseline
    /// exceeds the bound are the ones the canonical key may newly prove (exact equivalence).
    #[test]
    fn replay_state_key_matches_the_baseline_on_a_deterministic_corpus() {
        let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
        let (mut proven, mut refused, mut fewer) = (0u32, 0u32, 0u32);
        for i in 0..6000u32 {
            let arch = if i % 2 == 0 {
                Arch::Gfx942
            } else {
                Arch::Gfx1201
            };
            let p = corpus_program(&mut rng, arch);
            let new = check(&p);
            let old = walk_full(&p);
            if matches!(&old, Err(e) if e.contains("state explosion")) {
                continue;
            }
            let text: Vec<&str> = p.instructions.iter().map(|i| i.text.as_str()).collect();
            assert_eq!(
                new.is_ok(),
                old.is_ok(),
                "{arch:?}: {new:?} vs {old:?}\n{text:#?}"
            );
            match (&new, &old) {
                (Ok(n), Ok(o)) => {
                    proven += 1;
                    assert!(n.label_states <= o.label_states, "{text:#?}");
                    if arch == Arch::Gfx942 {
                        fewer += u32::from(n.label_states < o.label_states);
                    } else {
                        assert_eq!(n.label_states, o.label_states, "{text:#?}");
                    }
                }
                _ => refused += 1,
            }
        }
        assert!(
            proven > 0 && refused > 0 && fewer > 0,
            "{proven} {refused} {fewer}"
        );
    }

    /// An all-[`EdgeChoice::Both`] map but for the listed `(pc, choice)` overrides.
    fn proofs(p: &Program, at: &[(usize, EdgeChoice)]) -> Vec<EdgeChoice> {
        let mut choices = vec![EdgeChoice::Both; p.instructions.len()];
        for &(pc, choice) in at {
            choices[pc] = choice;
        }
        choices
    }
    /// Optional loads in distinct registers, all drained after the EXEC restore.
    /// No source or instruction rewriting is needed to collapse their 2^10 subsets.
    fn optional_vm_loads(prefix: Vec<Instruction>, before_wait: Vec<Instruction>) -> Program {
        let mut insns = prefix;
        for i in 0..10 {
            insns.push(ins(&format!("s_cbranch_scc1 J{i}")));
            insns.push(load(10 + i));
            insns.push(ins(&format!("J{i}:")));
        }
        insns.extend(before_wait);
        insns.push(ins("s_or_b64 exec, exec, s[40:41]"));
        insns.push(ins("s_waitcnt vmcnt(0)"));
        for i in 0..10 {
            insns.push(mov(30 + i, 10 + i));
        }
        insns.push(ins("s_endpgm"));
        cdna(insns)
    }

    #[test]
    fn unobserved_load_subsets_share_a_drain_continuation_class() {
        let p = optional_vm_loads(vec![], vec![]);
        let old = walk_full(&p).unwrap_err();
        assert!(old.contains("state explosion"), "{old}");
        let new = check(&p).unwrap();
        assert_eq!(new.label_states, 10, "{new:?}");
        // The scalar-control adapter uses the same abstraction with identical proof maps.
        let choices = vec![EdgeChoice::Both; p.instructions.len()];
        assert_eq!(
            verify_with_branch_proofs(&p, &[], &choices, &[]).unwrap(),
            new
        );
    }

    #[test]
    fn predrain_read_write_and_exec_masked_overwrite_are_not_erased() {
        for access in [mov(80, 1), mov(1, 80)] {
            let p = optional_vm_loads(vec![load(1)], vec![ins("s_mov_b64 exec, 1"), access]);
            let e = check(&p).unwrap_err();
            // The optional dead loads can be collapsed only if the ENTIRE incoming
            // queue is unobserved. Here a real hazard remains, not a waived use/WAW.
            assert!(
                e.contains("Use") || e.contains("Waw") || e.contains("state explosion"),
                "{e}"
            );
            assert!(walk_full(&p).is_err());
        }
        // Minimal versions name the actual hazard rather than the fanout bound.
        for access in [mov(80, 1), mov(1, 80)] {
            let p = cdna(vec![
                ins("s_cbranch_scc1 J"),
                load(1),
                ins("J:"),
                ins("s_mov_b64 exec, 1"),
                access,
                ins("s_waitcnt vmcnt(0)"),
                ins("s_endpgm"),
            ]);
            let e = check(&p).unwrap_err();
            assert!(e.contains("Use") || e.contains("Waw"), "{e}");
        }
    }

    #[test]
    fn every_continuation_must_reach_a_full_drain() {
        for suffix in [
            vec![
                ins("s_cbranch_scc1 USE"),
                ins("s_waitcnt vmcnt(0)"),
                ins("USE:"),
                mov(3, 1),
                ins("s_endpgm"),
            ],
            vec![ins("s_waitcnt vmcnt(1)"), mov(3, 1), ins("s_endpgm")],
            vec![
                ins("s_cbranch_scc1 LOOP"),
                ins("s_waitcnt vmcnt(0)"),
                ins("s_endpgm"),
                ins("LOOP:"),
                mov(3, 1),
                ins("s_branch LOOP"),
            ],
        ] {
            let mut insns = vec![ins("s_cbranch_scc1 J"), load(1), ins("J:")];
            insns.extend(suffix);
            let e = check(&cdna(insns)).unwrap_err();
            assert!(e.contains("Use"), "{e}");
        }
        // Both alternatives drain: a branch before the drain is not itself a refusal.
        let p = cdna(vec![
            load(1),
            ins("J:"),
            ins("s_cbranch_scc1 A"),
            ins("s_waitcnt vmcnt(0)"),
            ins("s_branch USE"),
            ins("A:"),
            ins("s_waitcnt vmcnt(0)"),
            ins("USE:"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        check(&p).unwrap();
    }

    #[test]
    fn pending_stores_and_flat_loads_never_enter_the_erased_class() {
        let flat = Instruction::new("flat_load_dword v1, v[2:3]", vec![v(1)], vec![v(2)])
            .memory(MemoryClass::Flat { load: true });
        for pending in [store(1), flat] {
            let p = optional_vm_loads(vec![pending], vec![]);
            let e = check(&p).unwrap_err();
            assert!(e.contains("state explosion"), "{e}");
        }
        // A future store/FLAT issue also prevents erasing an incoming load queue.
        for future in [
            store(80),
            Instruction::new("flat_load_dword v80, v[2:3]", vec![v(80)], vec![v(2)])
                .memory(MemoryClass::Flat { load: true }),
        ] {
            let p = optional_vm_loads(vec![], vec![future]);
            let e = check(&p).unwrap_err();
            assert!(e.contains("state explosion"), "{e}");
        }
    }

    #[test]
    fn other_counter_obligations_survive_vm_projection() {
        for pending in [ds_load(1), smem_load(1)] {
            let use_ = if matches!(pending.memory, Some(MemoryClass::SmemLoad)) {
                Instruction::new("s_mov_b32 s3, s1", vec![s(3)], vec![s(1)])
            } else {
                mov(3, 1)
            };
            let p = cdna(vec![
                ins("s_cbranch_scc1 J"),
                pending,
                ins("J:"),
                ins("s_waitcnt vmcnt(0)"),
                use_,
                ins("s_endpgm"),
            ]);
            let e = check(&p).unwrap_err();
            assert!(e.contains("Lgkm") && e.contains("Use"), "{e}");
        }
        let p = cdna(vec![
            ins("s_cbranch_scc1 J"),
            Instruction::new("exp", vec![], vec![v(1)]).memory(MemoryClass::Export),
            ins("J:"),
            ins("s_waitcnt vmcnt(0)"),
            mov(1, 2),
            ins("s_endpgm"),
        ]);
        let new = check(&p).unwrap();
        assert_eq!(new.label_states, 2, "{new:?}");
        assert_eq!(new, walk_full(&p).unwrap());
    }

    #[test]
    fn drain_proof_does_not_assume_loop_exit_or_merge_register_banks() {
        let p = cdna(vec![
            ins("H:"),
            ins("s_cbranch_scc1 H"),
            ins("s_waitcnt vmcnt(0)"),
            ins("s_endpgm"),
        ]);
        let labels = HashMap::from([("H", 0)]);
        assert!(vm_drain_observers(&p, &HashMap::new(), &labels)[0].is_none());
        let mut observed = VmObservers::default();
        observed.insert(RegRef {
            kind: Kind::V,
            base: 10,
            len: 4,
        });
        assert!(observed.observes(v(13)));
        assert!(!observed.observes(v(14)));
        assert!(!observed.observes(s(10)));
        assert!(!observed.observes(RegRef {
            kind: Kind::A,
            base: 10,
            len: 1
        }));
    }

    /// The baseline-keyed walk under the same proof map as `verify_with_branch_proofs`.
    fn walk_full_proved(p: &Program, choices: &[EdgeChoice]) -> Result<Stats, String> {
        walk_with(
            p,
            &[],
            Mode::Strict,
            Ledger::full_state_key,
            Some(choices),
            &[],
            false,
        )
    }

    #[test]
    fn malformed_proof_maps_are_rejected_before_the_walk() {
        let p = cdna(vec![
            load(1),
            ins("s_cbranch_scc1 T"),
            ins("s_branch T"),
            ins("s_cbranch_vccz T"),
            ins("T:"),
            ins("s_endpgm"),
        ]);
        let n = p.instructions.len();
        for len in [0, n - 1, n + 1] {
            let e =
                verify_with_branch_proofs(&p, &[], &vec![EdgeChoice::Both; len], &[]).unwrap_err();
            assert!(e.contains("branch proof map"), "{len}: {e}");
        }
        // pc 0 load, pc 2 unconditional, pc 3 ineligible conditional, pc 4 label, pc 5 endpgm.
        for pc in [0, 2, 3, 4, 5] {
            for choice in [EdgeChoice::Taken, EdgeChoice::Fallthrough] {
                let e = verify_with_branch_proofs(&p, &[], &proofs(&p, &[(pc, choice)]), &[])
                    .unwrap_err();
                assert!(e.contains("branch proof map"), "pc {pc} {choice:?}: {e}");
                assert!(e.contains(&format!("pc {pc}")), "pc {pc}: {e}");
            }
        }
        // An eligible conditional may carry a proof, and Both is accepted anywhere.
        verify_with_branch_proofs(&p, &[], &proofs(&p, &[(1, EdgeChoice::Taken)]), &[]).unwrap();
        verify_with_branch_proofs(&p, &[], &proofs(&p, &[(1, EdgeChoice::Fallthrough)]), &[])
            .unwrap();
        verify_with_branch_proofs(&p, &[], &proofs(&p, &[]), &[]).unwrap();
        // The test seam validates identically.
        let e = walk_full_proved(&p, &proofs(&p, &[(2, EdgeChoice::Taken)])).unwrap_err();
        assert!(e.contains("branch proof map"), "{e}");
    }

    #[test]
    fn proofs_are_strict_only() {
        let p = cdna(vec![ins("s_cbranch_scc1 T"), ins("T:"), ins("s_endpgm")]);
        let choices = proofs(&p, &[(0, EdgeChoice::Taken)]);
        let e = walk_with(
            &p,
            &[],
            Mode::Dispatcher { body: &[] },
            Ledger::replay_state_key,
            Some(&choices),
            &[],
            true,
        )
        .unwrap_err();
        assert!(e.contains("strict"), "{e}");
        // The dispatcher entry points take no proofs: all-Both, as before.
        assert_eq!(
            verify_dispatcher(&p, &[]).unwrap(),
            verify(&p, &[]).unwrap()
        );
    }

    #[test]
    fn taken_follows_only_the_target_and_fallthrough_only_the_next_instruction() {
        // The load is pending on both edges; only the fall-through path waits before the use.
        let wait_on_fallthrough = cdna(vec![
            load(1),
            ins("s_cbranch_scc1 T"),
            ins("s_waitcnt vmcnt(0)"),
            ins("T:"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        let p = &wait_on_fallthrough;
        let both = verify(p, &[]).unwrap_err();
        assert!(
            both.contains("pc 4") && both.contains("pc 1 `s_cbranch_scc1 T` taken"),
            "{both}"
        );
        let taken = verify_with_branch_proofs(p, &[], &proofs(p, &[(1, EdgeChoice::Taken)]), &[])
            .unwrap_err();
        assert!(
            taken.contains("pc 4") && taken.contains("pc 1 `s_cbranch_scc1 T` taken"),
            "{taken}"
        );
        verify_with_branch_proofs(p, &[], &proofs(p, &[(1, EdgeChoice::Fallthrough)]), &[])
            .unwrap();

        // The mirror image: only the taken path waits.
        let wait_on_taken = cdna(vec![
            load(1),
            ins("s_cbranch_scc1 T"),
            mov(3, 1),
            ins("s_endpgm"),
            ins("T:"),
            ins("s_waitcnt vmcnt(0)"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        let p = &wait_on_taken;
        assert!(verify(p, &[]).is_err());
        verify_with_branch_proofs(p, &[], &proofs(p, &[(1, EdgeChoice::Taken)]), &[]).unwrap();
        let fall =
            verify_with_branch_proofs(p, &[], &proofs(p, &[(1, EdgeChoice::Fallthrough)]), &[])
                .unwrap_err();
        assert!(
            fall.contains("pc 2") && fall.contains("pc 1 `s_cbranch_scc1 T` not taken"),
            "{fall}"
        );
    }

    #[test]
    fn an_impossible_edge_is_still_target_validated() {
        let p = cdna(vec![ins("s_cbranch_scc1 NOWHERE"), ins("s_endpgm")]);
        for choice in [EdgeChoice::Both, EdgeChoice::Taken, EdgeChoice::Fallthrough] {
            let e =
                verify_with_branch_proofs(&p, &[], &proofs(&p, &[(0, choice)]), &[]).unwrap_err();
            assert!(e.contains("never placed"), "{choice:?}: {e}");
        }
    }

    fn entry(
        from: usize,
        taken: bool,
        label: usize,
        branch: usize,
        choice: EdgeChoice,
    ) -> EntryProof {
        EntryProof {
            from,
            taken,
            label,
            branch,
            choice,
        }
    }
    /// The baseline-keyed walk under the same block-local and edge proofs.
    fn walk_full_entries(
        p: &Program,
        choices: &[EdgeChoice],
        entries: &[EntryProof],
    ) -> Result<Stats, String> {
        walk_with(
            p,
            &[],
            Mode::Strict,
            Ledger::full_state_key,
            Some(choices),
            entries,
            false,
        )
    }
    /// A join `J` entered over two edges (the `s_branch` at pc 2, the fall-through of pc 4) and
    /// ended by the conditional at pc 6; pc 7 overwrites `v1` with a VALU write, which is not
    /// ordered with VMEM (a WAW if a load of `v1` is pending; a VMEM reload would be ordered).
    /// pc 0 chooses the side; `branch_side` runs only before the `s_branch`, `fall_side` only
    /// before the fall-through. Ledger pcs: 0 branch, 1 branch side, 2 `s_branch J`, 3 `B:`,
    /// 4 fall side, 5 `J:`, 6 `s_cbranch_scc1 E`, 7 reload, 8 `E:`, 9 endpgm.
    fn join(branch_side: Instruction, fall_side: Instruction) -> Program {
        join_writing(branch_side, fall_side, valu_v1())
    }
    fn valu_v1() -> Instruction {
        Instruction::new("v_mov_b32 v1, 0", vec![v(1)], vec![])
    }
    fn join_writing(
        branch_side: Instruction,
        fall_side: Instruction,
        write: Instruction,
    ) -> Program {
        cdna(vec![
            ins("s_cbranch_scc1 B"),
            branch_side,
            ins("s_branch J"),
            ins("B:"),
            fall_side,
            ins("J:"),
            ins("s_cbranch_scc1 E"),
            write,
            ins("E:"),
            ins("s_endpgm"),
        ])
    }

    #[test]
    fn an_entry_proof_prunes_only_the_walk_that_entered_over_its_edge() {
        let nop = || ins("s_nop 0");
        let none: Vec<EdgeChoice> = vec![EdgeChoice::Both; 10];

        // Ordered VMEM load/load writeback is certified on its own, with no proof at all.
        let ordered = join_writing(load(1), nop(), load(1));
        verify_with_branch_proofs(&ordered, &[], &none, &[]).unwrap();

        // The pending load reaches J only over the `s_branch` edge: without the proof the
        // not-taken edge of pc 6 overwrites `v1` under it.
        let p = join(load(1), nop());
        let e = verify_with_branch_proofs(&p, &[], &none, &[]).unwrap_err();
        assert!(
            e.contains("pc 7") && e.contains("Waw") && e.contains("pc 2 `s_branch J` taken"),
            "{e}"
        );
        let over_branch = [entry(2, true, 5, 6, EdgeChoice::Taken)];
        verify_with_branch_proofs(&p, &[], &none, &over_branch).unwrap();
        assert!(walk_full_entries(&p, &none, &over_branch).is_ok());
        // The baseline key agrees: without the proof it refuses the same program.
        let reference = walk_full_entries(&p, &none, &[]).unwrap_err();
        assert!(reference.contains("pc 7"), "{reference}");

        // A proof for the other edge does not rescue the walk over this one.
        let wrong_edge = [entry(4, false, 5, 6, EdgeChoice::Taken)];
        let e = verify_with_branch_proofs(&p, &[], &none, &wrong_edge).unwrap_err();
        assert!(e.contains("pc 7"), "{e}");

        // The mirror image: the pending load reaches J only by falling through.
        let p = join(nop(), load(1));
        assert!(verify_with_branch_proofs(&p, &[], &none, &[]).is_err());
        let over_fallthrough = [entry(4, false, 5, 6, EdgeChoice::Taken)];
        verify_with_branch_proofs(&p, &[], &none, &over_fallthrough).unwrap();
        assert!(walk_full_entries(&p, &none, &over_fallthrough).is_ok());
        let e = verify_with_branch_proofs(&p, &[], &none, &over_branch).unwrap_err();
        assert!(e.contains("pc 7"), "{e}");

        // `Fallthrough` selects the not-taken edge for the proven arrival only.
        let p = join(nop(), nop());
        let to_end = [entry(2, true, 5, 6, EdgeChoice::Fallthrough)];
        verify_with_branch_proofs(&p, &[], &none, &to_end).unwrap();
    }

    #[test]
    fn arrivals_with_equal_memory_but_different_proofs_are_never_merged() {
        // Both arrivals at J carry the same pending shape (one load of `v1`), so the ledger
        // half of the key is equal; the proven arrival (pc 2 -> J, Taken) must not hide the
        // unproven one (fall-through), which reaches the overwrite at pc 7.
        let p = join(load(1), load(1));
        let none: Vec<EdgeChoice> = vec![EdgeChoice::Both; 10];
        let proof = [entry(2, true, 5, 6, EdgeChoice::Taken)];
        let e = verify_with_branch_proofs(&p, &[], &none, &proof).unwrap_err();
        assert!(
            e.contains("pc 7") && e.contains("pc 0 `s_cbranch_scc1 B` taken"),
            "{e}"
        );
        assert!(walk_full_entries(&p, &none, &proof).is_err());
        // With a proof on each edge both arrivals are decided and the program passes.
        let both = [
            entry(2, true, 5, 6, EdgeChoice::Taken),
            entry(4, false, 5, 6, EdgeChoice::Taken),
        ];
        let stats = verify_with_branch_proofs(&p, &[], &none, &both).unwrap();
        assert_eq!(stats, walk_full_entries(&p, &none, &both).unwrap());
        // No proofs: the label key is the old one (one state, both arrivals merged).
        let p = join(ins("s_nop 0"), ins("s_nop 0"));
        let plain = verify_with_branch_proofs(&p, &[], &none, &[]).unwrap();
        assert_eq!(plain, verify(&p, &[]).unwrap());
    }

    #[test]
    fn malformed_entry_proofs_are_rejected_before_the_walk() {
        let p = join(ins("s_nop 0"), ins("s_nop 0"));
        let none: Vec<EdgeChoice> = vec![EdgeChoice::Both; 10];
        let good = entry(2, true, 5, 6, EdgeChoice::Taken);
        verify_with_branch_proofs(&p, &[], &none, &[good]).unwrap();
        verify_with_branch_proofs(&p, &[], &none, &[entry(4, false, 5, 6, EdgeChoice::Taken)])
            .unwrap();

        let bad = |list: &[EntryProof], why: &str| {
            let e = verify_with_branch_proofs(&p, &[], &none, list).unwrap_err();
            assert!(
                e.contains("branch entry proof") && e.contains(why),
                "{why}: {e}"
            );
        };
        bad(
            &[entry(2, true, 5, 6, EdgeChoice::Both)],
            "must select an edge",
        );
        bad(
            &[entry(2, true, 99, 6, EdgeChoice::Taken)],
            "outside the program",
        );
        bad(
            &[entry(99, true, 5, 6, EdgeChoice::Taken)],
            "outside the program",
        );
        bad(&[entry(2, true, 4, 6, EdgeChoice::Taken)], "is not a label");
        bad(
            &[entry(2, true, 5, 7, EdgeChoice::Taken)],
            "is not a conditional",
        );
        bad(
            &[entry(2, true, 5, 2, EdgeChoice::Taken)],
            "is not a conditional",
        );
        bad(
            &[entry(0, true, 3, 6, EdgeChoice::Taken)],
            "ends or splits the block",
        );
        bad(
            &[entry(2, true, 5, 0, EdgeChoice::Taken)],
            "after the label",
        );
        // The named edge must exist: not a branch, a wrong target, an unconditional
        // fall-through, or a fall-through into a different pc.
        bad(
            &[entry(1, true, 5, 6, EdgeChoice::Taken)],
            "not a control-flow edge",
        );
        bad(
            &[entry(0, true, 5, 6, EdgeChoice::Taken)],
            "not a control-flow edge",
        );
        bad(
            &[entry(2, false, 5, 6, EdgeChoice::Taken)],
            "not a control-flow edge",
        );
        bad(
            &[entry(3, false, 5, 6, EdgeChoice::Taken)],
            "not a control-flow edge",
        );
        bad(&[good, good], "already has a proof");
        bad(
            &[good, entry(2, true, 5, 6, EdgeChoice::Fallthrough)],
            "already has a proof",
        );

        // A block-local proof on the same branch leaves nothing to refine.
        let local = proofs(&p, &[(6, EdgeChoice::Taken)]);
        let e = verify_with_branch_proofs(&p, &[], &local, &[good]).unwrap_err();
        assert!(e.contains("already carries a block-local proof"), "{e}");
        // Strict only, and only together with the block-local map.
        let e = walk_with(
            &p,
            &[],
            Mode::Dispatcher { body: &[] },
            Ledger::replay_state_key,
            Some(&none),
            &[good],
            true,
        )
        .unwrap_err();
        assert!(e.contains("strict"), "{e}");
        let e = walk_with(
            &p,
            &[],
            Mode::Strict,
            Ledger::replay_state_key,
            None,
            &[good],
            true,
        )
        .unwrap_err();
        assert!(e.contains("block-local proof map"), "{e}");
    }

    #[test]
    fn proofs_prune_loop_edges_without_changing_the_fixed_point() {
        // The previous iteration's load is still pending at the head's use: the back edge
        // is the hazard.
        let carried = cdna(vec![
            ins("H:"),
            mov(3, 1),
            load(1),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        let p = &carried;
        assert!(verify(p, &[]).unwrap_err().contains("pc 1"));
        let taken = verify_with_branch_proofs(p, &[], &proofs(p, &[(3, EdgeChoice::Taken)]), &[])
            .unwrap_err();
        assert!(taken.contains("pc 1"), "{taken}");
        verify_with_branch_proofs(p, &[], &proofs(p, &[(3, EdgeChoice::Fallthrough)]), &[])
            .unwrap();

        // A back edge that is always taken never reaches the code after it.
        let spin = cdna(vec![
            ins("H:"),
            load(1),
            ins("s_waitcnt vmcnt(0)"),
            ins("s_cbranch_scc1 H"),
            load(2),
            mov(3, 2),
            ins("s_endpgm"),
        ]);
        let p = &spin;
        assert!(verify(p, &[]).is_err());
        let stats =
            verify_with_branch_proofs(p, &[], &proofs(p, &[(3, EdgeChoice::Taken)]), &[]).unwrap();
        assert_eq!(stats.label_states, 1, "{stats:?}");
    }

    #[test]
    fn wait_prefix_dedup_is_unchanged_under_proofs() {
        // Two states reach A (load pending, nothing pending); the label's wait prefix
        // drains both into one.
        let p = cdna(vec![
            ins("s_cbranch_scc1 A"),
            load(1),
            ins("A:"),
            ins("s_waitcnt vmcnt(0)"),
            mov(3, 1),
            ins("s_endpgm"),
        ]);
        for choice in [EdgeChoice::Both, EdgeChoice::Taken, EdgeChoice::Fallthrough] {
            let choices = proofs(&p, &[(0, choice)]);
            let canonical = verify_with_branch_proofs(&p, &[], &choices, &[]).unwrap();
            let full = walk_full_proved(&p, &choices).unwrap();
            assert_eq!(canonical.label_states, 1, "{choice:?}: {canonical:?}");
            assert_eq!(canonical, full, "{choice:?}");
        }
    }

    #[test]
    fn all_both_proofs_are_the_unadapted_walk_on_the_corpus() {
        let mut rng = Lcg(0x2545_f491_4f6c_dd1d);
        for i in 0..1500u32 {
            let arch = if i % 2 == 0 {
                Arch::Gfx942
            } else {
                Arch::Gfx1201
            };
            let p = corpus_program(&mut rng, arch);
            let plain = verify(&p, &[]);
            let both = verify_with_branch_proofs(&p, &[], &proofs(&p, &[]), &[]);
            assert_eq!(plain, both, "{arch:?}");
        }
    }

    /// The canonical key against the baseline key under one **identical** random proof map
    /// (refined semantics are compared with themselves, never with the unrefined walk: pruned
    /// edges legitimately change verdicts). The edge choices here are arbitrary, not sound
    /// facts; they only exercise the seam.
    #[test]
    fn canonical_and_full_keys_agree_under_identical_proof_maps() {
        let mut rng = Lcg(0xd1b5_4a32_d192_ed03);
        let (mut proven, mut refused, mut pruned) = (0u32, 0u32, 0u32);
        for i in 0..3000u32 {
            let arch = if i % 2 == 0 {
                Arch::Gfx942
            } else {
                Arch::Gfx1201
            };
            let p = corpus_program(&mut rng, arch);
            let mut choices = proofs(&p, &[]);
            for (pc, insn) in p.instructions.iter().enumerate() {
                if insn.mnemonic() == "s_cbranch_scc1" {
                    choices[pc] = match rng.below(3) {
                        0 => EdgeChoice::Both,
                        1 => EdgeChoice::Taken,
                        _ => EdgeChoice::Fallthrough,
                    };
                    pruned += u32::from(choices[pc] != EdgeChoice::Both);
                }
            }
            let new = verify_with_branch_proofs(&p, &[], &choices, &[]);
            let old = walk_full_proved(&p, &choices);
            if matches!(&old, Err(e) if e.contains("state explosion")) {
                continue;
            }
            let text: Vec<&str> = p.instructions.iter().map(|i| i.text.as_str()).collect();
            assert_eq!(
                new.is_ok(),
                old.is_ok(),
                "{arch:?}: {new:?} vs {old:?}\n{text:#?}"
            );
            match (&new, &old) {
                (Ok(n), Ok(o)) => {
                    proven += 1;
                    assert!(n.label_states <= o.label_states, "{text:#?}");
                    if arch != Arch::Gfx942 {
                        assert_eq!(n.label_states, o.label_states, "{text:#?}");
                    }
                }
                _ => refused += 1,
            }
        }
        assert!(
            proven > 0 && refused > 0 && pruned > 0,
            "{proven} {refused} {pruned}"
        );
    }

    #[test]
    fn indirect_control_flow_fails_closed() {
        let e = check(&gfx12(vec![ins("s_setpc_b64 s[0:1]"), ins("s_endpgm")])).unwrap_err();
        assert!(e.contains("cannot follow"), "{e}");
    }

    /// Foreign text carries no metadata: its lines are decoded through the lift's
    /// instruction table, so a load without a wait before its use is found.
    #[test]
    fn foreign_text_is_decoded_and_checked() {
        let text = |lines: &[&str]| {
            let mut insns: Vec<Instruction> = lines.iter().map(|l| ins(l)).collect();
            insns.push(ins("s_endpgm"));
            gfx12(insns)
        };
        let missing = text(&["global_load_b32 v1, v0, s[0:1]", "v_mov_b32_e32 v3, v1"]);
        let e = verify(&missing, &[0..2]).unwrap_err();
        assert!(
            e.contains("v_mov_b32_e32 v3, v1") && e.contains("needs Load"),
            "{e}"
        );
        // The same text outside a foreign span has no metadata and is taken at its word.
        verify(&missing, &[]).unwrap();
        let waited = text(&[
            "global_load_b32 v1, v0, s[0:1]",
            "s_wait_loadcnt 0x0",
            "v_mov_b32_e32 v3, v1",
        ]);
        verify(&waited, &[0..3]).unwrap();
        // A branch path that skips the wait is found in foreign text too.
        let skipped = gfx12(vec![
            ins("global_load_b32 v1, v0, s[0:1]"),
            ins("s_cbranch_scc1 L"),
            ins("s_wait_loadcnt 0x0"),
            ins("L:"),
            ins("v_mov_b32_e32 v3, v1"),
            ins("s_endpgm"),
        ]);
        let e = verify(&skipped, &[0..5]).unwrap_err();
        assert!(e.contains("s_cbranch_scc1 L` taken"), "{e}");
        // A foreign line the table cannot decode fails closed.
        let e = verify(&text(&["v_not_an_instruction v1"]), &[0..1]).unwrap_err();
        assert!(e.contains("not decodable"), "{e}");
    }
}

#[cfg(test)]
mod dispatcher_tests {
    use super::*;

    fn v(base: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base,
            len: 1,
        }
    }
    fn ins(text: &str) -> Instruction {
        Instruction::new(text, vec![], vec![])
    }
    /// A dispatcher store: it carries its own metadata and locks v1 until its store wait.
    fn store_v1() -> Instruction {
        Instruction::new("global_store_b32 v0, v1, s[0:1]", vec![], vec![v(0), v(1)])
            .memory(MemoryClass::VmemStore)
    }
    /// A dispatcher instruction that overwrites v1 (a WAR on a pending store of v1).
    fn set_v1(imm: u32) -> Instruction {
        Instruction::new(format!("v_mov_b32_e32 v1, {imm}"), vec![v(1)], vec![])
    }
    /// A dispatcher instruction that defines `dst` and uses `src` (a RAW on a pending load of
    /// `src`).
    fn dispatcher_mov(dst: u8, src: u8) -> Instruction {
        Instruction::new(
            format!("v_mov_b32 v{dst}, v{src}"),
            vec![v(dst)],
            vec![v(src)],
        )
    }
    fn gfx12(insns: Vec<Instruction>) -> Program {
        Program {
            arch: Arch::Gfx1201,
            instructions: insns,
        }
    }

    /// A hazard inside the body span is not reported (counted); the same hazard one pc
    /// outside the span is fatal. Strict mode reports the body's.
    #[test]
    fn only_hazards_inside_the_exact_span_are_suppressed() {
        // pc 0 dispatcher store locking v1; pc 1 and pc 2 each write v1 with no wait.
        let p = gfx12(vec![store_v1(), set_v1(5), set_v1(6), ins("s_endpgm")]);
        assert!(verify(&p, &[1..3]).is_err(), "strict reports the body");
        let stats = verify_dispatcher(&p, &[1..3]).unwrap();
        assert_eq!(stats.suppressed, 2, "{stats:?}");
        // One pc too short: pc 2 is dispatcher code and its hazard is fatal, pc 1's is not.
        let e = verify_dispatcher(&p, &[1..2]).unwrap_err();
        assert!(e.contains("1 dispatcher hazard(s)"), "{e}");
        assert!(e.contains("pc 2 `v_mov_b32_e32 v1, 6`"), "{e}");
        assert!(!e.contains("pc 1 `"), "{e}");
        // One pc late: pc 1 is dispatcher code.
        let e = verify_dispatcher(&p, &[2..3]).unwrap_err();
        assert!(e.contains("pc 1 `v_mov_b32_e32 v1, 5`"), "{e}");
        assert!(!e.contains("pc 2 `"), "{e}");
        // No span at all: nothing is allowed.
        let e = verify_dispatcher(&p, &[]).unwrap_err();
        assert!(e.contains("2 dispatcher hazard(s)"), "{e}");
    }

    /// A body's memory operations are tracked: the load a body leaves pending is a hazard
    /// for dispatcher code that uses it, and a body finding does not hide a later
    /// dispatcher one (every dispatcher hazard is collected, not the first).
    #[test]
    fn body_effects_reach_the_dispatcher_and_body_findings_hide_nothing() {
        let p = gfx12(vec![
            // body span 0..2: a load and its unwaited use (a body finding)
            ins("global_load_b32 v1, v0, s[0:1]"),
            ins("v_mov_b32_e32 v3, v1"),
            // dispatcher: still pending from the body, two distinct uses
            dispatcher_mov(4, 1),
            dispatcher_mov(5, 1),
            ins("s_endpgm"),
        ]);
        let e = verify_dispatcher(&p, &[0..2]).unwrap_err();
        assert!(e.contains("2 dispatcher hazard(s)"), "{e}");
        assert!(e.contains("pc 2 `v_mov_b32 v4, v1`"), "{e}");
        assert!(e.contains("pc 3 `v_mov_b32 v5, v1`"), "{e}");
        assert!(!e.contains("v_mov_b32_e32 v3, v1"), "{e}");
        // The body waited for itself before exit: only its finding remains, suppressed.
        let p = gfx12(vec![
            ins("global_load_b32 v1, v0, s[0:1]"),
            ins("v_mov_b32_e32 v3, v1"),
            ins("s_wait_loadcnt 0x0"),
            dispatcher_mov(4, 1),
            ins("s_endpgm"),
        ]);
        let stats = verify_dispatcher(&p, &[0..3]).unwrap();
        assert_eq!(stats.suppressed, 1, "{stats:?}");
    }

    /// Only hazards are suppressed in a body: a line the lift table cannot decode and
    /// unfollowable control flow stay errors.
    #[test]
    fn body_decode_errors_stay_fatal() {
        let e = verify_dispatcher(
            &gfx12(vec![ins("v_not_an_instruction v1"), ins("s_endpgm")]),
            &[0..1],
        )
        .unwrap_err();
        assert!(e.contains("not decodable"), "{e}");
        let e = verify_dispatcher(
            &gfx12(vec![ins("s_setpc_b64 s[0:1]"), ins("s_endpgm")]),
            &[0..1],
        )
        .unwrap_err();
        assert!(e.contains("cannot follow"), "{e}");
    }

    /// With no hazard anywhere dispatcher mode agrees with strict mode and suppresses nothing.
    #[test]
    fn clean_program_is_identical_in_both_modes() {
        let p = gfx12(vec![
            ins("global_load_b32 v1, v0, s[0:1]"),
            ins("s_wait_loadcnt 0x0"),
            ins("v_mov_b32_e32 v3, v1"),
            ins("s_endpgm"),
        ]);
        let strict = verify(&p, &[0..3]).unwrap();
        let dispatcher = verify_dispatcher(&p, &[0..3]).unwrap();
        assert_eq!(strict, dispatcher);
        assert_eq!(dispatcher.suppressed, 0);
    }

    fn sreg(base: u8) -> RegRef {
        RegRef {
            kind: Kind::S,
            base,
            len: 1,
        }
    }
    /// A dispatcher load into `dst` (VMEM, LOADcnt).
    fn dispatcher_load(dst: u8) -> Instruction {
        Instruction::new(
            format!("global_load_b32 v{dst}, v0, s[0:1]"),
            vec![v(dst)],
            vec![v(0)],
        )
        .memory(MemoryClass::VmemLoad)
    }
    /// `before`, a region (entry label, `body`, exit label) and `after`; the span is the body.
    fn region(
        before: Vec<Instruction>,
        body: &[&str],
        after: Vec<Instruction>,
    ) -> (Program, Range<usize>) {
        let mut insns = before;
        insns.push(ins("R_entry:"));
        let start = insns.len();
        insns.extend(body.iter().map(|t| ins(t)));
        let end = insns.len();
        insns.push(ins("R_exit:"));
        insns.extend(after);
        insns.push(ins("s_endpgm"));
        (gfx12(insns), start..end)
    }
    const DRAIN: [&str; 4] = [
        "s_wait_loadcnt 0x0",
        "s_wait_storecnt 0x0",
        "s_wait_dscnt 0x0",
        "s_wait_kmcnt 0x0",
    ];

    /// A region whose every exiting path ends in a full drain is not walked: the dispatcher
    /// continues from its exit with nothing in flight, so what was pending at its entry is
    /// retired, whatever the loop inside did.
    #[test]
    fn draining_suffix_region_is_summarized_to_an_empty_exit() {
        let mut body = vec![
            "LOOP:",
            "global_load_b32 v1, v0, s[0:1]",
            "global_store_b32 v0, v2, s[0:1]",
            "s_cbranch_scc1 LOOP",
        ];
        body.extend(DRAIN);
        let (p, span) = region(vec![dispatcher_load(7)], &body, vec![dispatcher_mov(3, 7)]);
        let stats = verify_dispatcher(&p, &[span]).unwrap();
        assert_eq!(stats.summarized, 1, "{stats:?}");
    }

    /// No summary, and the dispatcher hazard is still caught by walking the body, when a
    /// branch bypasses the drain, when one counter's drain is missing on one branch, when a
    /// wait is nonzero, and when a branch leaves the body; and a region with no drain at all
    /// is never summarized.
    #[test]
    fn unproven_regions_are_walked_and_dispatcher_hazards_are_caught() {
        // A branch around the whole drain: the load pending at entry stays pending.
        let mut body = vec!["s_cbranch_scc1 SKIP"];
        body.extend(DRAIN);
        body.push("SKIP:");
        let (p, span) = region(vec![dispatcher_load(7)], &body, vec![dispatcher_mov(3, 7)]);
        let e = verify_dispatcher(&p, &[span]).unwrap_err();
        assert!(e.contains("v_mov_b32 v3, v7") && e.contains("Load"), "{e}");
        // One branch misses the KMcnt drain only: the SMEM load pending at entry is not retired.
        let smem = Instruction::new("s_load_b32 s4, s[0:1], 0x0", vec![sreg(4)], vec![])
            .memory(MemoryClass::SmemLoad);
        let use_s4 = Instruction::new("s_mov_b32 s5, s4", vec![sreg(5)], vec![sreg(4)]);
        let body = [
            "s_cbranch_scc1 P",
            "s_wait_kmcnt 0x0",
            "P:",
            "s_wait_loadcnt 0x0",
            "s_wait_storecnt 0x0",
            "s_wait_dscnt 0x0",
        ];
        let (p, span) = region(vec![smem.clone()], &body, vec![use_s4.clone()]);
        let e = verify_dispatcher(&p, &[span]).unwrap_err();
        assert!(
            e.contains("s_mov_b32 s5, s4") && e.contains("Km wait"),
            "{e}"
        );
        // The same body with the KMcnt drain on every path is summarized and passes.
        let body = [
            "s_cbranch_scc1 P",
            "P:",
            "s_wait_kmcnt 0x0",
            "s_wait_loadcnt 0x0",
            "s_wait_storecnt 0x0",
            "s_wait_dscnt 0x0",
        ];
        let (p, span) = region(vec![smem], &body, vec![use_s4]);
        assert_eq!(verify_dispatcher(&p, &[span]).unwrap().summarized, 1);
        // A nonzero wait clears nothing; a branch out of the body and a body with no waits
        // are not summarized either (they pass here only because nothing is used after).
        let (p, span) = region(
            vec![],
            &[
                "s_wait_loadcnt 0x1",
                "s_wait_storecnt 0x0",
                "s_wait_dscnt 0x0",
                "s_wait_kmcnt 0x0",
            ],
            vec![],
        );
        assert_eq!(verify_dispatcher(&p, &[span]).unwrap().summarized, 0);
        let (p, span) = region(
            vec![],
            &[
                "s_cbranch_scc1 OUT",
                "s_wait_loadcnt 0x0",
                "s_wait_storecnt 0x0",
                "s_wait_dscnt 0x0",
                "s_wait_kmcnt 0x0",
            ],
            vec![ins("OUT:")],
        );
        assert_eq!(verify_dispatcher(&p, &[span]).unwrap().summarized, 0);
        let (p, span) = region(vec![], &["global_load_b32 v1, v0, s[0:1]"], vec![]);
        assert_eq!(verify_dispatcher(&p, &[span]).unwrap().summarized, 0);
    }
}
