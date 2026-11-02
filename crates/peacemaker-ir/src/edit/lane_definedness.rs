// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Conservative lane coverage for partial VGPR writes. The existing half/dword
//! definite-assignment analysis remains authoritative for fully undefined reads.
//! Fully undefined reads retain their separate entry-definedness findings.

use super::{reg_dwords, Flow, EXEC_HI, EXEC_LO, NDW, S0};
use crate::cfg::Body;
use crate::inst::{Arch, Inst, Wave};
use crate::operand::{ImmField, InlineConst, Operand, Special};
use crate::reg::Kind;
use std::collections::HashMap;

const SCALARS: usize = NDW - S0;
// Each scalar bit is either definitely one, possibly one, or definitely zero.
// On merge only facts true on all incoming paths survive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mask {
    must: u32,
    may: u32,
}
impl Mask {
    const UNKNOWN: Self = Self {
        must: 0,
        may: u32::MAX,
    };
    fn exact(n: u32) -> Self {
        Self { must: n, may: n }
    }
    fn known(self) -> Option<u32> {
        (self.must == self.may).then_some(self.must)
    }
    fn and(self, rhs: Self) -> Self {
        Self {
            must: self.must & rhs.must,
            may: self.may & rhs.may,
        }
    }
    fn or(self, rhs: Self) -> Self {
        Self {
            must: self.must | rhs.must,
            may: self.may | rhs.may,
        }
    }
    fn not(self) -> Self {
        Self {
            must: !self.may,
            may: !self.must,
        }
    }
    fn join(self, rhs: Self) -> Self {
        Self {
            must: self.must & rhs.must,
            may: self.may | rhs.may,
        }
    }
}
type Constants = [Mask; SCALARS];

fn location(op: &Operand) -> Option<usize> {
    match op {
        Operand::Reg(r) if r.kind != Kind::V && r.len == 1 => reg_dwords(*r).next(),
        Operand::Special(Special::ExecLo) => Some(EXEC_LO),
        Operand::Special(Special::ExecHi) => Some(EXEC_HI),
        _ => None,
    }
}

fn pair_location(op: &Operand) -> Option<(usize, usize)> {
    match op {
        Operand::Reg(r) if r.kind != Kind::V && r.len == 2 => {
            let d = reg_dwords(*r).next()?;
            Some((d, d + 1))
        }
        Operand::Special(Special::Exec) => Some((EXEC_LO, EXEC_HI)),
        _ => None,
    }
}

fn pair_value(op: &Operand, state: &Constants) -> (Mask, Mask) {
    if let Some((lo, hi)) = pair_location(op) {
        return (state[lo - S0], state[hi - S0]);
    }
    // The sign/zero-extension of general 32-bit literals in a b64 source
    // is not inferred; zero and -1 have unambiguous high halves.
    match op {
        Operand::Inline(InlineConst::Integer(0)) => (Mask::exact(0), Mask::exact(0)),
        Operand::Inline(InlineConst::Integer(-1)) => (Mask::exact(u32::MAX), Mask::exact(u32::MAX)),
        _ => (Mask::UNKNOWN, Mask::UNKNOWN),
    }
}

fn value(op: &Operand, state: &Constants) -> Mask {
    match op {
        Operand::Inline(InlineConst::Integer(n)) => Mask::exact(*n as i32 as u32),
        Operand::Literal(n) | Operand::Imm(ImmField::Unsigned(n)) => Mask::exact(*n),
        Operand::Imm(ImmField::Sopk(n)) => Mask::exact(*n as i32 as u32),
        _ => location(op).map_or(Mask::UNKNOWN, |d| state[d - S0]),
    }
}

fn transfer(inst: &Inst, arch: Arch, old: &Constants, writes: &super::Bits) -> Constants {
    let mut next = *old;
    for d in S0..NDW {
        if writes.has(d * 2) || writes.has(d * 2 + 1) {
            next[d - S0] = Mask::UNKNOWN;
        }
    }
    let Some(name) = inst.op.name(arch) else {
        return next;
    };
    let ops = inst.operands.as_slice();
    if let Some((lo, hi)) = ops.first().and_then(pair_location) {
        let (src_lo, src_hi) = ops
            .get(1)
            .map_or((Mask::UNKNOWN, Mask::UNKNOWN), |op| pair_value(op, old));
        let old_exec = (old[EXEC_LO - S0], old[EXEC_HI - S0]);
        let value = match name {
            "s_mov_b64" => (src_lo, src_hi),
            "s_and_b64" => {
                let (a, b) = ops
                    .get(2)
                    .map_or((Mask::UNKNOWN, Mask::UNKNOWN), |op| pair_value(op, old));
                (src_lo.and(a), src_hi.and(b))
            }
            "s_or_b64" => {
                let (a, b) = ops
                    .get(2)
                    .map_or((Mask::UNKNOWN, Mask::UNKNOWN), |op| pair_value(op, old));
                (src_lo.or(a), src_hi.or(b))
            }
            "s_and_saveexec_b64" => {
                next[EXEC_LO - S0] = old_exec.0.and(src_lo);
                next[EXEC_HI - S0] = old_exec.1.and(src_hi);
                old_exec
            }
            "s_or_saveexec_b64" => {
                next[EXEC_LO - S0] = old_exec.0.or(src_lo);
                next[EXEC_HI - S0] = old_exec.1.or(src_hi);
                old_exec
            }
            _ => (Mask::UNKNOWN, Mask::UNKNOWN),
        };
        next[lo - S0] = value.0;
        next[hi - S0] = value.1;
        return next;
    }
    let Some(dst) = ops.first().and_then(location) else {
        return next;
    };
    let src = |n| ops.get(n).map_or(Mask::UNKNOWN, |op| value(op, old));
    let result = match name {
        "s_mov_b32" => src(1),
        "s_and_b32" => src(1).and(src(2)),
        "s_or_b32" => src(1).or(src(2)),
        "s_xor_b32" => src(1).and(src(2).not()).or(src(1).not().and(src(2))),
        "s_andn2_b32" => src(1).and(src(2).not()),
        "s_orn2_b32" => src(1).or(src(2).not()),
        "s_lshl_b32" => src(2).known().map_or(Mask::UNKNOWN, |b| Mask {
            must: src(1).must.wrapping_shl(b & 31),
            may: src(1).may.wrapping_shl(b & 31),
        }),
        "s_lshr_b32" => src(2).known().map_or(Mask::UNKNOWN, |b| Mask {
            must: src(1).must.wrapping_shr(b & 31),
            may: src(1).may.wrapping_shr(b & 31),
        }),
        "s_and_saveexec_b32" => {
            let exec = old[EXEC_LO - S0];
            next[EXEC_LO - S0] = exec.and(src(1));
            exec
        }
        "s_or_saveexec_b32" => {
            let exec = old[EXEC_LO - S0];
            next[EXEC_LO - S0] = exec.or(src(1));
            exec
        }
        _ => Mask::UNKNOWN,
    };
    next[dst - S0] = result;
    next
}

fn scalar_before(body: &Body, flow: &Flow, arch: Arch) -> Vec<Constants> {
    let mut before = vec![[Mask::UNKNOWN; SCALARS]; flow.n];
    let mut after = vec![[Mask::UNKNOWN; SCALARS]; flow.n];
    let mut reached = vec![false; flow.n];
    let mut work: Vec<_> = (0..flow.n).rev().collect();
    let mut queued = vec![true; flow.n];
    while let Some(p) = work.pop() {
        queued[p] = false;
        let mut incoming = [Mask::UNKNOWN; SCALARS];
        let mut have = p == 0;
        for &q in &flow.pred[p] {
            if !reached[q] {
                continue;
            }
            if !have {
                incoming = after[q];
                have = true;
            } else {
                for (a, b) in incoming.iter_mut().zip(after[q]) {
                    *a = a.join(b);
                }
            }
        }
        if !have {
            continue;
        }
        before[p] = incoming;
        let inst = body.insts.get(flow.ids[p]).expect("laid out");
        let out = transfer(inst, arch, &incoming, &flow.acc[p].writes);
        if out != after[p] || !reached[p] {
            reached[p] = true;
            after[p] = out;
            for &s in &flow.succ[p] {
                if !queued[s] {
                    queued[s] = true;
                    work.push(s);
                }
            }
        }
    }
    before
}

fn active(state: &Constants, wave: Wave, writing: bool) -> u64 {
    let select = |mask: Mask| if writing { mask.must } else { mask.may };
    let low = select(state[EXEC_LO - S0]);
    if wave == Wave::Wave32 {
        return u64::from(low);
    }
    u64::from(low) | (u64::from(select(state[EXEC_HI - S0])) << 32)
}

fn lane_index(inst: &Inst, state: &Constants) -> Option<u32> {
    inst.operands
        .last()
        .and_then(|op| value(op, state).known())
        .filter(|&lane| lane < 64)
}

fn destination(inst: &Inst) -> Option<usize> {
    match inst.operands.first()? {
        Operand::Reg(r) if r.kind == Kind::V && r.len == 1 => Some(usize::from(r.base)),
        _ => None,
    }
}

fn written_lanes(
    inst: &Inst,
    flow: &Flow,
    p: usize,
    reg: usize,
    constants: &Constants,
    wave: Wave,
    arch: Arch,
) -> u64 {
    if destination(inst) == Some(reg) && inst.op.name(arch) == Some("v_writelane_b32") {
        return lane_index(inst, constants).map_or(0, |lane| 1u64 << lane);
    }
    // DPP/SDWA selection can preserve destination lanes/halves even when the
    // operand table names a whole VGPR. Do not claim their full EXEC coverage.
    if inst.mods.dpp.is_some() || inst.mods.sdwa.is_some() {
        return 0;
    }
    let bits = &flow.acc[p].writes;
    if bits.has(reg * 2) && bits.has(reg * 2 + 1) {
        if super::is_vector_form(inst.form) {
            active(constants, wave, true)
        } else {
            u64::MAX
        }
    } else {
        0
    }
}

fn observes(
    inst: &Inst,
    flow: &Flow,
    p: usize,
    reg: usize,
    constants: &Constants,
    wave: Wave,
    arch: Arch,
) -> u64 {
    if !flow.acc[p].reads.has(reg * 2) && !flow.acc[p].reads.has(reg * 2 + 1) {
        return 0;
    }
    let name = inst.op.name(arch).unwrap_or("");
    if name == "v_writelane_b32" && destination(inst) == Some(reg) {
        // This table use is the unmodified lanes, not a consumed operand.
        // A source explicitly naming the same VGPR is still a real read.
        if !inst.operands.iter().skip(1).any(|op| matches!(op, Operand::Reg(r) if r.kind == Kind::V && usize::from(r.base) <= reg && reg < usize::from(r.base) + usize::from(r.len))) {
            return 0;
        }
    }
    if name == "v_readlane_b32" {
        return lane_index(inst, constants).map_or(u64::MAX, |lane| 1u64 << lane);
    }
    if super::is_vector_form(inst.form) {
        active(constants, wave, false)
    } else {
        u64::MAX
    }
}

/// Return the original partial-write sites whose preserved lanes cannot be
/// observed uninitialised on any path. The half/dword pass continues to own
/// fully undefined reads; this refinement only proves its open lane writes.
pub(super) fn prove(
    body: &Body,
    flow: &Flow,
    arch: Arch,
    wave: Wave,
    sites: &[usize],
) -> Vec<usize> {
    let constants = scalar_before(body, flow, arch);
    let mut by_reg: HashMap<usize, Vec<usize>> = HashMap::new();
    for &p in sites {
        if let Some(reg) = body.insts.get(flow.ids[p]).and_then(destination) {
            by_reg.entry(reg).or_default().push(p);
        }
    }
    let mut proved = Vec::new();
    for (reg, candidates) in by_reg {
        let mut before = vec![u64::MAX; flow.n];
        let mut after = vec![u64::MAX; flow.n];
        let mut work: Vec<_> = (0..flow.n).rev().collect();
        let mut queued = vec![true; flow.n];
        while let Some(p) = work.pop() {
            queued[p] = false;
            let mut incoming = if p == 0 {
                if reg == 0 {
                    u64::MAX
                } else {
                    0
                }
            } else {
                u64::MAX
            };
            for &q in &flow.pred[p] {
                incoming &= after[q];
            }
            before[p] = incoming;
            let inst = body.insts.get(flow.ids[p]).expect("laid out");
            let out = incoming | written_lanes(inst, flow, p, reg, &constants[p], wave, arch);
            if out != after[p] {
                after[p] = out;
                for &s in &flow.succ[p] {
                    if !queued[s] {
                        queued[s] = true;
                        work.push(s);
                    }
                }
            }
        }
        for p in candidates {
            let writer = body.insts.get(flow.ids[p]).expect("laid out");
            if observes(writer, flow, p, reg, &constants[p], wave, arch) != 0 {
                continue; // an actual source read in the partial write itself
            }
            let mut seen = vec![false; flow.n];
            let mut todo = flow.succ[p].to_vec();
            let mut valid = true;
            while let Some(q) = todo.pop() {
                if std::mem::replace(&mut seen[q], true) {
                    continue;
                }
                let inst = body.insts.get(flow.ids[q]).expect("laid out");
                let required = observes(inst, flow, q, reg, &constants[q], wave, arch);
                if required & !before[q] != 0 {
                    valid = false;
                    break;
                }
                todo.extend(flow.succ[q].iter().copied());
            }
            if valid {
                proved.push(p);
            }
        }
    }
    proved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inst::FormFields;
    use crate::operand::Modifiers;
    use crate::provenance::Provenance;
    use crate::reg::RegRef;
    use smallvec::smallvec;

    #[test]
    fn wave64_high_lanes_and_b64_exec_copy_are_not_lost() {
        let mut state = [Mask::UNKNOWN; SCALARS];
        state[EXEC_LO - S0] = Mask::exact(3);
        state[EXEC_HI - S0] = Mask::exact(4);
        assert_eq!(active(&state, Wave::Wave64, false), (1u64 << 34) | 3);
        state[EXEC_HI - S0] = Mask::UNKNOWN;
        assert_eq!(active(&state, Wave::Wave64, true), 3);
        assert_eq!(
            active(&state, Wave::Wave64, false),
            (u64::from(u32::MAX) << 32) | 3
        );

        let row = crate::isa::gfx12()
            .iter()
            .find(|r| r.name == "s_mov_b64")
            .unwrap();
        let src = RegRef {
            kind: Kind::S,
            base: 20,
            len: 2,
        };
        state[20] = Mask::exact(3);
        state[21] = Mask::exact(4);
        let inst = Inst::from_parts(
            Arch::Gfx1201,
            row.op,
            row.form,
            FormFields::None,
            smallvec![Operand::Special(Special::Exec), Operand::Reg(src)],
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap();
        let access = super::super::access(Arch::Gfx1201, Wave::Wave64, &inst).unwrap();
        let out = transfer(&inst, Arch::Gfx1201, &state, &access.writes);
        assert_eq!(active(&out, Wave::Wave64, true), (1u64 << 34) | 3);
    }
}
