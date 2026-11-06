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
//!   ghost loads). Only states are deduplicated, by [`Ledger::state_key`]
//!   (pending shapes without operation ids), at labels, the only join points:
//!   a forward branch is explored as its own alternative; a backward edge is
//!   explored until a label's state set reaches a fixed point, and ids that
//!   merely advance per iteration do not grow it.
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
//!   ledger cannot track (flat, LDS DMA), or a `ttmp` operand of a memory
//!   operation is an error, not a skipped line (`ttmp` operands of other
//!   instructions, e.g. workgroup-id reads, are not tracked). Outside `foreign`, an instruction with empty metadata is taken
//!   at its word: it has no memory effect.
//! * No operation is ever dropped: more than [`COUNTER_CAP`] pending operations
//!   on one counter, more than [`MAX_STATES_PER_LABEL`] distinct states at one
//!   label, more than [`MAX_STEPS`] instruction visits, or control flow the walk
//!   cannot follow (indirect branches, calls) fail closed.
//!
//! A violation names the instruction, the counter wait it lacks, and the branch
//! edges of the path that reaches it.

use crate::{
    arch::Arch,
    insn::{Instruction, MemoryClass, Program},
    ledger::{Counter, Ledger, StateKey},
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
            class @ (MemClass::LdsDma
            | MemClass::FlatLoad
            | MemClass::FlatStore
            | MemClass::FlatAtomic { .. }) => {
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
/// instruction table.
pub fn verify_emitted(emitted: &Emitted) -> Result<Stats, String> {
    verify(&emitted.program, &region_spans(&emitted.proof.regions))
}

/// Check that every instruction of `program` has every wait it needs on every
/// control-flow path from the entry (module docs). `foreign` are the instruction
/// ranges of foreign text, decoded through the lift's instruction table. `Err`
/// names the first violation found, or the bound that stopped the walk.
pub fn verify(program: &Program, foreign: &[Range<usize>]) -> Result<Stats, String> {
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
    let mut seen: HashMap<usize, HashSet<StateKey>> = HashMap::new();
    let mut witness = Witness::default();
    let mut stats = Stats::default();
    let mut stack: Vec<(usize, Ledger, Option<usize>)> = vec![(0, Ledger::default(), None)];
    while let Some((mut pc, mut ledger, mut path)) = stack.pop() {
        // The label this walk last passed: the block a violation sits in.
        let mut block: &str = "entry";
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
                let states = seen.entry(pc).or_default();
                if !states.insert(ledger.state_key()) {
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
                block = label;
                pc += 1;
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
                    pc = target_pc;
                } else {
                    let taken = witness.push(path, pc, true);
                    stack.push((target_pc, ledger.clone(), taken));
                    path = witness.push(path, pc, false);
                    pc += 1;
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
            if let Some((counter, count, reason)) = ledger
                .required(insn)
                .into_iter()
                .find(|(counter, count, _)| capacity(arch, *counter).is_none_or(|cap| *count < cap))
            {
                return Err(format!(
                    "pc {pc} `{text}` (after label `{block}`): needs {counter:?} wait {count} \
                     ({reason:?}) but a path reaches it without one: {}",
                    witness.describe(path, insns)
                ));
            }
            ledger.record(arch, insn);
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

    /// Past the per-counter bound the walk fails closed; below it the oldest
    /// pending operation's hazards are still checked.
    #[test]
    fn pending_depth_is_bounded_without_dropping_the_oldest_operation() {
        // The oldest store locks v1; 10 younger stores do not hide it.
        let mut insns = vec![store(1)];
        insns.extend((0..10).map(|_| store(2)));
        insns.extend([mov(1, 5), ins("s_endpgm")]);
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("War"), "{e}");
        // More than the cap pending on one counter is refused, never truncated.
        let mut insns = vec![store(1)];
        insns.extend((0..=COUNTER_CAP).map(|_| store(2)));
        insns.extend([mov(1, 5), ins("s_endpgm")]);
        let e = check(&gfx12(insns)).unwrap_err();
        assert!(e.contains("pending") && e.contains("unbounded"), "{e}");
        // A loop issuing a store per iteration without waiting grows without bound.
        let p = gfx12(vec![
            ins("H:"),
            store(1),
            ins("s_cbranch_scc1 H"),
            ins("s_endpgm"),
        ]);
        let e = check(&p).unwrap_err();
        assert!(e.contains("unbounded"), "{e}");
        // ... and passes once the iteration waits.
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
