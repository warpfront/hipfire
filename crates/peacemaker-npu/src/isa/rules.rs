// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Silicon-proven AIE2P core control rules, enforced on every assembled core program (`Program::finish`).
//!
//! Each rule admits only forms that ran exact on Strix Halo silicon (NPU5, `17f0_11`, fw 1.1.2.65) and rejects the
//! forms that failed there or never ran. Evidence logs are on hipx under `~/qcal/npu-logs/` (copies of the older ones
//! in `logs/`); the narrative is in `report.md`. `check` decodes the program linearly (it must be one contiguous
//! bundle stream from pc 0) and returns the first violation.
//!
//! | rule | admitted (silicon-proven) | rejected | evidence |
//! |---|---|---|---|
//! | [`Rule::BranchTarget`] | immediate jump targets on a 16-byte boundary, or an unaligned target whose 16-byte line holds only whole NOP bundles before it | any other unaligned target; indirect jumps (target not checkable) | Fast pair core with unaligned targets (mid-line, after real instructions) hung / corrupted ~48% of C, the same program with 16-byte targets was exact: `probe1-aaa165d.log` (21:17 UTC 2026-10-02, knob `align` vs Fast), `probe2-1f3dd24.log`. FastSlowCtl (`m3-a29f1af.log`, `v8b-773e1e4-2101.log`) and the single-core `gemm_i8` (`batch-2320869.log`) were exact with unaligned targets preceded only by NOPs in their line. llvm-aie pads every jump target to `getMachineBlockAlignmentBytes() == 16` (`AIEMachineAlignment.cpp`, `aie2ps/AIE2PSInstrInfo.h:127`). |
//! | [`Rule::ZeroOverheadLoop`] | no hardware loop | any write of `ls`, `le` or `lc` (`MOVXM`, `ADD.NC`, ...) | `LockOnlyZol` stored 1 of 64 words and FastSlowCtl's ZOL body ran once (3072/8192 C words right) with both `movxm lc` and the llvm-aie `add.nc lc, r1, #imm` form: `m3-a29f1af.log`, `m3-5d95c79.log`. Loops are `JNZ` back edges (exact: every V8/V9 run since `8cccd8b`). |
//! | [`Rule::LockSpacing`] | lock requests at least [`LOCK_SPACING`] issue bundles apart in straight-line order (taken paths are covered by `DelaySlot`: no lock in the 5 delay bundles) | closer lock requests | Fast control (`ACQ`/`REL` + 3 NOPs, i.e. 4 bundles) is exact once targets are aligned: `probe2-1f3dd24.log`, `probe3-8a8d88e.log`, `probe5-02f7cbd.log`. Closer spacing never ran; `LCKREQ` holds its resource 4 cycles (`sched.rs` `ACQ_mLockId_imm`, RES_2). FastSlowCtl's 7-NOP lock gaps were conservative, not required (same logs). |
//! | [`Rule::DelaySlot`] | exactly [`JUMP_DELAY_SLOTS`] bundles after every control transfer, holding no control transfer, `DONE` or lock request, and no branch target inside them | branch/DONE/lock in a delay slot, a jump target inside another branch's delay slots, a transfer with fewer than 5 following bundles | Scalar ALU work in delay slots (`XOR r4` / `ADD r5` in the `JNZ r4` delay slots of `pair_release`, VLIW compute in the chunk/epilogue `JNZ` back-edge delay slots) is exact on silicon (`probe2-1f3dd24.log` onwards), so FastSlowCtl's NOP-only delay slots were conservative. Branches in delay slots are illegal in llvm-aie (`AIEBaseInstrInfo.cpp` "Cannot have branch in branch delay slot!"); locks and `DONE` in delay slots never ran on silicon. |
use super::{
    decode::{self, DecodedBundle, DecodedInst},
    sched::{self, JUMP_DELAY_SLOTS},
};
use std::fmt;

/// Minimum issue-bundle distance between two lock requests (`ACQ`/`REL`).
pub const LOCK_SPACING: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    BranchTarget,
    ZeroOverheadLoop,
    LockSpacing,
    DelaySlot,
    Decode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub rule: Rule,
    pub pc: u64,
    pub detail: String,
}
impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AIE2P rule {:?} violated at pc {:#x}: {} (see peacemaker-npu isa::rules)",
            self.rule, self.pc, self.detail
        )
    }
}
impl std::error::Error for Violation {}

fn violation(rule: Rule, pc: u64, detail: impl Into<String>) -> Violation {
    Violation {
        rule,
        pc,
        detail: detail.into(),
    }
}

fn is_transfer(inst: &DecodedInst) -> bool {
    sched::itinerary(inst.encoding.name).is_some_and(|i| i.delay_slots > 0)
}
fn is_lock(inst: &DecodedInst) -> bool {
    inst.encoding.name.starts_with("ACQ_") || inst.encoding.name.starts_with("REL_")
}
fn is_done(inst: &DecodedInst) -> bool {
    inst.encoding.name == "DONE"
}
fn is_nop_bundle(b: &DecodedBundle) -> bool {
    b.instructions
        .iter()
        .all(|i| i.encoding.mnemonic.starts_with("nop"))
}
/// Immediate (absolute) target of a control transfer, `None` for register-indirect ones.
fn target(inst: &DecodedInst) -> Option<u64> {
    inst.operands
        .iter()
        .find(|o| o.name == "i" && !o.output)
        .map(|o| o.value as u64)
}
fn writes_loop_register(inst: &DecodedInst) -> Option<&'static str> {
    inst.operands
        .iter()
        .enumerate()
        .filter(|(_, o)| o.output)
        .find_map(|(n, o)| {
            let op = inst
                .encoding
                .operands
                .iter()
                .find(|e| e.name == o.name)
                .or(inst.encoding.operands.get(n))?;
            decode::resolve_register(op, o.value as u64)
                .map(|r| r.name)
                .filter(|r| matches!(*r, "ls" | "le" | "lc"))
        })
}

/// Check every rule of the module table on a core program image (bundles from pc 0).
pub fn check(bytes: &[u8]) -> Result<(), Violation> {
    let mut bundles = Vec::new();
    let mut pc = 0;
    while pc < bytes.len() {
        let b = decode::decode(&bytes[pc..], pc as u64)
            .map_err(|e| violation(Rule::Decode, pc as u64, e.to_string()))?;
        pc += b.len;
        bundles.push(b);
    }
    let index = |pc: u64| bundles.binary_search_by_key(&pc, |b| b.pc).ok();
    let has = |i: usize, f: fn(&DecodedInst) -> bool| bundles[i].instructions.iter().any(f);
    // Control transfers: (bundle index, immediate target bundle index).
    let mut transfers = Vec::new();
    for (i, b) in bundles.iter().enumerate() {
        for inst in &b.instructions {
            if let Some(reg) = writes_loop_register(inst) {
                return Err(violation(
                    Rule::ZeroOverheadLoop,
                    b.pc,
                    format!(
                        "{} writes {reg}: hardware loops never looped on silicon",
                        inst.encoding.name
                    ),
                ));
            }
            if !is_transfer(inst) {
                continue;
            }
            let Some(t) = target(inst) else {
                return Err(violation(
                    Rule::BranchTarget,
                    b.pc,
                    format!(
                        "{} is register-indirect: target alignment cannot be checked",
                        inst.encoding.name
                    ),
                ));
            };
            let ti = index(t).ok_or_else(|| {
                violation(
                    Rule::BranchTarget,
                    b.pc,
                    format!("target {t:#x} is not a bundle start"),
                )
            })?;
            if t % 16 != 0 {
                let line = index(t & !15).filter(|&s| bundles[s..ti].iter().all(is_nop_bundle));
                if line.is_none() {
                    return Err(violation(Rule::BranchTarget, b.pc, format!(
                        "target {t:#x} is not 16-byte aligned and its line {:#x} holds non-NOP or partial bundles before it", t & !15)));
                }
            }
            transfers.push((i, ti));
        }
    }
    for &(i, _) in &transfers {
        let pc = bundles[i].pc;
        if i + JUMP_DELAY_SLOTS >= bundles.len() {
            return Err(violation(
                Rule::DelaySlot,
                pc,
                format!("control transfer needs {JUMP_DELAY_SLOTS} following bundles"),
            ));
        }
        for d in i + 1..=i + JUMP_DELAY_SLOTS {
            if has(d, is_transfer) || has(d, is_done) || has(d, is_lock) {
                return Err(violation(Rule::DelaySlot, bundles[d].pc, format!("delay slot of the transfer at {pc:#x} holds a control transfer, DONE or lock request")));
            }
            if let Some(&(src, _)) = transfers.iter().find(|&&(_, t)| t == d) {
                return Err(violation(
                    Rule::DelaySlot,
                    bundles[d].pc,
                    format!(
                        "jump target of {:#x} lies in the delay slots of {pc:#x}",
                        bundles[src].pc
                    ),
                ));
            }
        }
    }
    // Lock spacing in straight-line order. On a taken path the last 5 bundles before the target are delay slots,
    // which hold no lock request (DelaySlot), so a lock at the target is always >= 6 bundles after the previous one.
    let locks: Vec<usize> = (0..bundles.len()).filter(|&i| has(i, is_lock)).collect();
    for w in locks.windows(2) {
        if w[1] - w[0] < LOCK_SPACING {
            return Err(violation(
                Rule::LockSpacing,
                bundles[w[1]].pc,
                format!(
                    "lock request {} bundles after the one at {:#x}",
                    w[1] - w[0],
                    bundles[w[0]].pc
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{self, Program, Reg};

    fn rule(p: Program) -> Option<Rule> {
        p.try_finish().err().map(|v| v.rule)
    }
    fn nops(p: &mut Program, n: usize) -> &mut Program {
        p.nops(n)
    }

    #[test]
    fn aligned_and_nop_line_targets_pass() {
        let mut p = Program::new();
        nops(&mut p, 8); // 16 bytes
        p.push(isa::jnz(Reg::R(3), 0x10)); // target 0x10: aligned
        nops(&mut p, 5);
        assert_eq!(rule(p), None);
        let mut p = Program::new();
        nops(&mut p, 10); // target 0x14: line 0x10 holds two NOP bundles before it
        p.push(isa::jnz(Reg::R(3), 0x14));
        nops(&mut p, 5);
        assert_eq!(rule(p), None);
    }
    #[test]
    fn unaligned_target_after_real_instruction_fails() {
        let mut p = Program::new();
        p.push(isa::movxm(Reg::R(3), 1)); // 6 bytes: 0x0..0x5
        p.push(isa::jnz(Reg::R(3), 0x6)); // target 0x6: line 0x0 starts with a MOVXM
        nops(&mut p, 5);
        assert_eq!(rule(p), Some(Rule::BranchTarget));
    }
    #[test]
    fn zero_overhead_loop_register_write_fails() {
        let enc = isa::gen::ENCODINGS
            .iter()
            .find(|e| e.name == "ADD_NC_mv_add_ri")
            .unwrap();
        let lc = enc.operands[0]
            .registers
            .iter()
            .find(|r| r.name == "lc")
            .unwrap()
            .value;
        let (slot, bits) = (
            enc.slot,
            isa::bundle::encode_slot(enc, &[("dst", lc), ("s0", 1), ("imm", 63)], 0).unwrap(),
        );
        let mut p = Program::new();
        p.bytes
            .extend_from_slice(isa::bundle::pack_slots(&[(slot, bits)]).unwrap().as_slice());
        nops(&mut p, 1);
        assert_eq!(rule(p), Some(Rule::ZeroOverheadLoop));
    }
    #[test]
    fn close_lock_requests_fail() {
        let mut p = Program::new();
        p.push(isa::acq(48, Reg::R(0)));
        nops(&mut p, 2);
        p.push(isa::rel(49, Reg::R(1))); // 3 bundles apart
        assert_eq!(rule(p), Some(Rule::LockSpacing));
        let mut p = Program::new();
        p.push(isa::acq(48, Reg::R(0)));
        nops(&mut p, 3);
        p.push(isa::rel(49, Reg::R(1)));
        assert_eq!(rule(p), None);
    }
    #[test]
    fn delay_slot_contents() {
        // ALU work in delay slots is silicon-proven and allowed.
        let mut p = Program::new();
        p.push(isa::jnz(Reg::R(4), 0));
        p.push(isa::add_ri(Reg::R(5), Reg::R(5), -1));
        nops(&mut p, 4);
        assert_eq!(rule(p), None);
        // A lock request in a delay slot is rejected.
        let mut p = Program::new();
        p.push(isa::jnz(Reg::R(4), 0));
        p.push(isa::rel(49, Reg::R(1)));
        nops(&mut p, 4);
        assert_eq!(rule(p), Some(Rule::DelaySlot));
        // A branch in a delay slot is rejected.
        let mut p = Program::new();
        p.push(isa::jnz(Reg::R(4), 0));
        p.push(isa::j(0));
        nops(&mut p, 9);
        assert_eq!(rule(p), Some(Rule::DelaySlot));
        // Too few bundles after a transfer (raw image, no `finish` padding).
        let mut p = Program::new();
        p.push(isa::j(0));
        nops(&mut p, 2);
        assert_eq!(check(&p.bytes).map_err(|v| v.rule), Err(Rule::DelaySlot));
    }
}
