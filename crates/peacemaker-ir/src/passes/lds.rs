//! C5: LDS segment and address facts for lifted kernels.
//!
//! Lifted code gets facts, never invented slots (core.md §2.7). A forward
//! CFG fixed point tracks unsigned 32-bit intervals through integer address
//! arithmetic, lane counts and bounded hardware-register extracts. Joins
//! include every predecessor; growing loop intervals widen to the full
//! domain, and unsigned loop guards can narrow them again. Unsupported,
//! memory-derived and overflowing arithmetic stays unknown. Partial EXEC
//! writes retain the old VGPR value. DS offsets are applied in byte units.
//! All lifted accesses attach to one whole segment.

use thiserror::Error;

use crate::cfg::{Body, InstId};
use crate::effects::MemClass;
use crate::inst::{Arch, Inst};
use crate::lds::{AddrFact, LdsAccess, LdsFacts, LdsKind, LdsSegment, SegId, SegmentOrigin};
use crate::operand::{ImmField, InlineConst, Operand};
use crate::reg::Kind;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LdsError {
    #[error("LDS facts need a gfx11 or gfx12 opcode table (got {0:?})")]
    UnsupportedArch(Arch),
    #[error("layout refers to a tombstoned or missing instruction")]
    DanglingInst { id: InstId },
}

/// M1 LDS analysis is facts-only: `max_end == None` (an unresolved address,
/// width or extent) records the launch-contract obligation, not a
/// separate object. Dynamic LDS sizes come from the dispatch packet, never
/// from the object, so no lifted kernel can prove them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LdsAnalysis {
    pub facts: LdsFacts,
}

fn access_kind(class: MemClass) -> Option<LdsKind> {
    match class {
        MemClass::DsLoad => Some(LdsKind::Load),
        MemClass::DsStore => Some(LdsKind::Store),
        MemClass::DsAtomic { .. } => Some(LdsKind::Atomic),
        _ => None,
    }
}

/// Data width in bytes from the opcode suffix (`_b32` → 4 …).
fn access_bytes(name: &str) -> u32 {
    if name.contains("_b128") {
        16
    } else if name.contains("_b96") {
        12
    } else if name.contains("_b64") || name.ends_with("_u64") || name.ends_with("_i64") {
        8
    } else if name.contains("_b32") || name.ends_with("_u32") || name.ends_with("_i32") || name.ends_with("_f32") {
        4
    } else if name.contains("_b16") {
        2
    } else if name.contains("_b8") {
        1
    } else {
        0
    }
}

/// DS offset operands carried by the access itself.
fn own_offsets(inst: &Inst, bytes: u32, name: &str) -> Value {
    let mut lo = u32::MAX;
    let mut hi = 0;
    for op in &inst.operands {
        let n = match op {
            Operand::Imm(ImmField::DsOffset(n)) => u32::from(*n),
            Operand::Imm(ImmField::DsOffset0(n) | ImmField::DsOffset1(n)) =>
                u32::from(*n) * bytes * if name.contains("st64") || name.contains("stride64") { 64 } else { 1 },
            _ => continue,
        };
        lo = lo.min(n);
        hi = hi.max(n);
    }
    Value { lo: if lo == u32::MAX { 0 } else { lo }, hi }
}

fn const_operand(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::Inline(InlineConst::Integer(n)) => Some(*n as u32),
        Operand::Literal(n) => Some(*n),
        _ => None,
    }
}

/// Inclusive unsigned interval. The full domain is the unknown value, not
/// evidence that a launch's LDS allocation is large enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Value { lo: u32, hi: u32 }
impl Value {
    const UNKNOWN: Self = Self { lo: 0, hi: u32::MAX };
    fn constant(n: u32) -> Self { Self { lo: n, hi: n } }
    fn join(self, rhs: Self) -> Self {
        Self { lo: self.lo.min(rhs.lo), hi: self.hi.max(rhs.hi) }
    }
    fn constant_value(self) -> Option<u32> { (self.lo == self.hi).then_some(self.lo) }
    fn add(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.constant_value(), rhs.constant_value()) {
            return Self::constant(a.wrapping_add(b));
        }
        match (self.lo.checked_add(rhs.lo), self.hi.checked_add(rhs.hi)) {
            (Some(lo), Some(hi)) => Self { lo, hi },
            _ => Self::UNKNOWN,
        }
    }
    fn mul(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.constant_value(), rhs.constant_value()) {
            return Self::constant(a.wrapping_mul(b));
        }
        match (self.lo.checked_mul(rhs.lo), self.hi.checked_mul(rhs.hi)) {
            (Some(lo), Some(hi)) => Self { lo, hi },
            _ => Self::UNKNOWN,
        }
    }
    fn fact(self) -> AddrFact {
        if self == Self::UNKNOWN { AddrFact::Unknown }
        else if self.lo == self.hi { AddrFact::Const(self.lo) }
        else { AddrFact::Bounded { lo: self.lo, hi: self.hi } }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Values {
    regs: Vec<Value>,
    full_exec: bool,
}
fn location(reg: crate::reg::RegRef) -> Option<usize> {
    match reg.kind {
        Kind::V => Some(usize::from(reg.base)),
        Kind::S => Some(512 + usize::from(reg.base)),
        _ => None,
    }
}
impl Values {
    fn operand(&self, op: &Operand) -> Value {
        if let Some(n) = const_operand(op) { return Value::constant(n); }
        match op {
            Operand::Reg(r) if r.len == 1 => location(*r)
                .and_then(|i| self.regs.get(i).copied()).unwrap_or(Value::UNKNOWN),
            _ => Value::UNKNOWN,
        }
    }
    fn merge(&mut self, rhs: &Self, widen: bool, thresholds: &[u32]) -> bool {
        let mut changed = false;
        for (old, next) in self.regs.iter_mut().zip(&rhs.regs) {
            let mut joined = old.join(*next);
            if widen {
                if joined.lo < old.lo { joined.lo = 0; }
                if joined.hi > old.hi {
                    joined.hi = thresholds.iter().copied().find(|n| *n >= joined.hi).unwrap_or(u32::MAX);
                }
            }
            changed |= *old != joined;
            *old = joined;
        }
        changed |= self.full_exec && !rhs.full_exec;
        self.full_exec &= rhs.full_exec;
        changed
    }
}

fn evaluate(name: &str, src: &[Value]) -> Value {
    let name = name.strip_suffix("_e32").or_else(|| name.strip_suffix("_e64")).unwrap_or(name);
    let a = src.first().copied().unwrap_or(Value::UNKNOWN);
    let b = src.get(1).copied().unwrap_or(Value::UNKNOWN);
    let c = src.get(2).copied().unwrap_or(Value::UNKNOWN);
    if name.starts_with("s_mov_b32") || name.starts_with("v_mov_b32")
        || name == "v_readfirstlane_b32" || name == "v_readlane_b32" {
        return a;
    }
    match name {
        "s_add_u32" | "s_add_co_u32" | "s_add_i32" | "v_add_u32" | "v_add_nc_u32" | "v_add_co_u32" => a.add(b),
        "s_sub_u32" | "s_sub_co_u32" | "s_sub_i32" | "v_sub_u32" | "v_sub_nc_u32" | "v_sub_co_u32" => {
            if let (Some(x), Some(y)) = (a.constant_value(), b.constant_value()) {
                Value::constant(x.wrapping_sub(y))
            } else if a.lo >= b.hi { Value { lo: a.lo - b.hi, hi: a.hi - b.lo } }
            else { Value::UNKNOWN }
        }
        "s_mul_i32" | "v_mul_lo_u32" | "v_mul_lo_i32" => a.mul(b),
        "v_mad_u32_u24" => {
            let truncate = |x: Value| match x.constant_value() {
                Some(n) => Value::constant(n & 0xffffff),
                None if x.hi <= 0xffffff => x,
                None => Value { lo: 0, hi: 0xffffff },
            };
            let a = truncate(a);
            let b = truncate(b);
            a.mul(b).add(c)
        }
        "v_add3_u32" => a.add(b).add(c),
        "s_and_b32" | "v_and_b32" => {
            if let (Some(x), Some(y)) = (a.constant_value(), b.constant_value()) {
                Value::constant(x & y)
            } else { Value { lo: 0, hi: a.hi.min(b.hi) } }
        }
        "s_or_b32" | "v_or_b32" | "s_xor_b32" | "v_xor_b32" => {
            let xor = name.contains("xor");
            if let (Some(x), Some(y)) = (a.constant_value(), b.constant_value()) {
                Value::constant(if xor { x ^ y } else { x | y })
            } else {
                let bits = 32 - (a.hi | b.hi).leading_zeros();
                Value { lo: 0, hi: u32::MAX.checked_shr(32 - bits).unwrap_or(0) }
            }
        }
        "s_lshl_b32" | "v_lshlrev_b32" | "s_lshr_b32" | "v_lshrrev_b32" => {
            let (value, shift) = if name.starts_with("v_") { (b, a) } else { (a, b) };
            let Some(n) = shift.constant_value().map(|n| n & 31) else { return Value::UNKNOWN };
            if name.contains("lshr") {
                Value { lo: value.lo >> n, hi: value.hi >> n }
            } else if let Some(x) = value.constant_value() {
                Value::constant(x.wrapping_shl(n))
            } else if value.hi <= u32::MAX >> n {
                Value { lo: value.lo << n, hi: value.hi << n }
            } else { Value::UNKNOWN }
        }
        "v_lshl_add_u32" => {
            let shift = b.constant_value().map(|n| n & 31);
            match shift {
                Some(n) if a.hi <= u32::MAX >> n =>
                    Value { lo: a.lo << n, hi: a.hi << n }.add(c),
                _ => Value::UNKNOWN,
            }
        }
        "v_add_lshl_u32" => {
            let sum = a.add(b);
            evaluate("s_lshl_b32", &[sum, c])
        }
        "v_lshl_or_b32" => {
            let shifted = evaluate("s_lshl_b32", &[a, b]);
            evaluate("s_or_b32", &[shifted, c])
        }
        // In wave64 the upper lanes count all 32 lower-mask bits.
        "v_mbcnt_lo_u32_b32" => b.add(Value { lo: 0, hi: 32 }),
        "v_mbcnt_hi_u32_b32" => b.add(Value { lo: 0, hi: 32 }),
        "s_min_u32" | "v_min_u32" => Value { lo: a.lo.min(b.lo), hi: a.hi.min(b.hi) },
        "s_max_u32" | "v_max_u32" => Value { lo: a.lo.max(b.lo), hi: a.hi.max(b.hi) },
        "s_cselect_b32" | "v_cndmask_b32" => a.join(b),
        "s_bfe_u32" => match b.constant_value() {
            Some(control) => {
                let shift = control & 31;
                let width = (control >> 16) & 127;
                let mask = u32::MAX.checked_shr(32u32.saturating_sub(width)).unwrap_or(0);
                if let Some(x) = a.constant_value() { Value::constant((x >> shift) & mask) }
                else { Value { lo: 0, hi: (a.hi >> shift).min(mask) } }
            }
            None => Value::UNKNOWN,
        },
        "v_bfe_u32" => match (b.constant_value(), c.constant_value()) {
            (Some(shift), Some(width)) => {
                let mask = u32::MAX.checked_shr(32 - (width & 31)).unwrap_or(0);
                if let Some(x) = a.constant_value() { Value::constant((x >> (shift & 31)) & mask) }
                else { Value { lo: 0, hi: (a.hi >> (shift & 31)).min(mask) } }
            }
            _ => Value::UNKNOWN,
        },
        _ => Value::UNKNOWN,
    }
}

fn transfer(st: &mut Values, inst: &Inst, arch: Arch) {
    use crate::effects::ImplicitSet;
    let Some(row) = crate::isa::lookup(arch, inst.op, inst.form) else { return };
    let mut src = smallvec::SmallVec::<[Value; 4]>::new();
    for ((field, _), op) in row.slots(crate::isa::atomic_returns(arch, &inst.mods.cpol)).zip(&inst.operands) {
        if row.uses.split(',').any(|f| f == field) {
            src.push(st.operand(op));
        }
    }
    let value = if inst.mods.abs != 0 || inst.mods.neg != 0 || inst.mods.sdwa.is_some()
        || inst.mods.op_sel != 0 || inst.effects.mem.is_some() {
        Value::UNKNOWN
    } else if row.name == "s_getreg_b32" {
        // A hardware-register extract includes wave IDs without assuming a
        // particular dispatch's number of waves.
        inst.operands.iter().find_map(|op| match op {
            Operand::Hwreg(r) if r.size > 0 && r.size <= 32 =>
                Some(Value { lo: 0, hi: u32::MAX >> (32 - r.size) }),
            _ => None,
        }).unwrap_or(Value::UNKNOWN)
    } else { evaluate(row.name, &src) };
    for reg in &inst.effects.defs {
        if let Some(base) = location(*reg) {
            for i in 0..usize::from(reg.len) {
                if let Some(old) = st.regs.get_mut(base + i) {
                    let next = if i == 0 && reg.len == 1 { value } else { Value::UNKNOWN };
                    let partial = inst.operands.iter().any(|op| matches!(op, Operand::Half(r, _) if r == reg));
                    *old = if reg.kind == Kind::V && (!st.full_exec || partial) { old.join(next) } else { next };
                }
            }
        }
    }
    if inst.effects.implicit.writes & ImplicitSet::EXEC != 0 { st.full_exec = false; }
}

/// Refine unsigned scalar loop guards only while their SCC and source
/// register still describe the comparison at the terminator.
fn refine(st: &mut Values, body: &Body, arch: Arch, range: (usize, usize), scc: bool) {
    use crate::effects::ImplicitSet;
    for pos in (range.0..range.1).rev() {
        let inst = body.insts.get(body.layout[pos]).expect("validated layout");
        if inst.effects.implicit.writes & ImplicitSet::SCC == 0 { continue; }
        let name = inst.op.name(arch).unwrap_or("");
        let Some(Operand::Reg(reg)) = inst.operands.first() else { return };
        if reg.kind != Kind::S || reg.len != 1 { return; }
        let Some(n) = inst.operands.get(1).and_then(const_operand) else { return };
        if body.layout[pos+1..range.1].iter().any(|id| body.insts.get(*id).unwrap().effects.defs.iter()
            .any(|r| r.kind == reg.kind && r.base <= reg.base && reg.base < r.base + u16::from(r.len))) { return; }
        let value = &mut st.regs[location(*reg).unwrap()];
        let (lo, hi) = match (name, scc) {
            ("s_cmp_lt_u32", true) | ("s_cmp_ge_u32", false) if n > 0 => (0, n - 1),
            ("s_cmp_lt_u32", false) | ("s_cmp_ge_u32", true) => (n, u32::MAX),
            ("s_cmp_le_u32", true) | ("s_cmp_gt_u32", false) => (0, n),
            ("s_cmp_le_u32", false) | ("s_cmp_gt_u32", true) if n < u32::MAX => (n + 1, u32::MAX),
            _ => return,
        };
        // An impossible edge may be retained, but must never create an
        // invalid interval.
        if value.lo.max(lo) <= value.hi.min(hi) {
            *value = Value { lo: value.lo.max(lo), hi: value.hi.min(hi) };
        }
        return;
    }
}

fn entries(body: &Body, arch: Arch) -> Vec<Option<Values>> {
    use crate::cfg::Cond;
    use crate::effects::Control;
    let count = body.blocks.len().max(1);
    let size = body.insts.iter().flat_map(|(_, i)| i.effects.defs.iter().chain(&i.effects.uses))
        .filter_map(|r| location(*r).map(|n| n + usize::from(r.len))).max().unwrap_or(0);
    let seed = Values { regs: vec![Value::UNKNOWN; size], full_exec: true };
    let mut input = vec![None; count];
    input[0] = Some(seed);
    if body.blocks.is_empty() { return input; }
    let mut queue = std::collections::VecDeque::from([0]);
    let mut queued = vec![false; count];
    queued[0] = true;
    let mut updates = vec![0; count];
    // Threshold widening preserves finite induction bounds from loop guards
    // without iterating once per trip. Every selected threshold is a
    // superset of the growing interval, even for unrelated constants.
    let mut thresholds = vec![u32::MAX];
    for (_, inst) in body.insts.iter() {
        for n in inst.operands.iter().filter_map(const_operand) {
            thresholds.push(n);
            if n > 0 { thresholds.push(n - 1); }
            if n < u32::MAX { thresholds.push(n + 1); }
        }
    }
    thresholds.sort_unstable();
    thresholds.dedup();
    let indices: std::collections::HashMap<_, _> = body.blocks.iter().enumerate()
        .map(|(i, b)| (b.id, i)).collect();
    let first_at: std::collections::HashMap<_, _> = body.blocks.iter().enumerate()
        .map(|(i, b)| (b.range.0, i)).collect();
    while let Some(b) = queue.pop_front() {
        queued[b] = false;
        let block = &body.blocks[b];
        let mut out = input[b].as_ref().unwrap().clone();
        let mut enqueue = |next: usize, edge: &Values| {
            let changed = match &mut input[next] {
                Some(old) => old.merge(edge, updates[next] >= 3, &thresholds),
                slot @ None => { *slot = Some(edge.clone()); true },
            };
            if changed {
                updates[next] += 1;
                if !queued[next] { queue.push_back(next); queued[next] = true; }
            }
        };
        let mut falls = true;
        for pos in block.range.0..block.range.1 {
            let inst = body.insts.get(body.layout[pos]).unwrap();
            transfer(&mut out, inst, arch);
            if matches!(inst.effects.control, Control::Jump | Control::Branch { .. }) {
                let target = inst.operands.iter().find_map(|op| match op {
                    Operand::Label(id) => indices.get(id).copied(),
                    _ => None,
                });
                let scc = match inst.effects.control {
                    Control::Branch { cond: Cond::Scc1 } => Some(true),
                    Control::Branch { cond: Cond::Scc0 } => Some(false),
                    _ => None,
                };
                if let Some(next) = target {
                    let mut edge = out.clone();
                    if let Some(scc) = scc { refine(&mut edge, body, arch, (block.range.0, pos + 1), scc); }
                    enqueue(next, &edge);
                }
                if matches!(inst.effects.control, Control::Jump) { falls = false; break; }
                if let Some(scc) = scc { refine(&mut out, body, arch, (block.range.0, pos + 1), !scc); }
            }
            if matches!(inst.effects.control, Control::EndPgm | Control::Trap | Control::Halt) {
                falls = false;
                break;
            }
        }
        if falls {
            if let Some(&next) = first_at.get(&block.range.1).filter(|&&next| next != b) {
                enqueue(next, &out);
            }
        }
    }
    input
}

/// LDS facts for one kernel body.
pub fn analyze(body: &Body, arch: Arch) -> Result<LdsAnalysis, LdsError> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(LdsError::UnsupportedArch(arch));
    }
    let whole = SegId(0);
    let mut facts = LdsFacts {
        segments: vec![LdsSegment { id: whole, range: None, origin: SegmentOrigin::Whole }],
        accesses: Vec::new(),
        max_end: Some(0),
    };
    let mut end = 0u32;
    let mut bounded = false;
    for id in &body.layout {
        if body.insts.get(*id).is_none() { return Err(LdsError::DanglingInst { id: *id }); }
    }
    let input = entries(body, arch);
    let ranges: Vec<_> = if body.blocks.is_empty() { vec![(0, body.layout.len())] }
        else { body.blocks.iter().map(|b| b.range).collect() };
    for (block, range) in ranges.iter().enumerate() {
        let Some(mut st) = input[block].clone() else { continue };
        for pos in range.0..range.1 {
            let id = body.layout[pos];
            let inst = body.insts.get(id).expect("validated layout");
            if let Some(kind) = inst.effects.mem.as_ref().and_then(|m| access_kind(m.class)) {
                let name = inst.op.name(arch).unwrap_or("");
                let bytes = access_bytes(name);
                let value = inst.effects.uses.first().and_then(|r| location(*r))
                    .and_then(|i| st.regs.get(i).copied()).unwrap_or(Value::UNKNOWN)
                    .add(own_offsets(inst, bytes, name));
                let addr = value.fact();
                match value.hi.checked_add(bytes) {
                    Some(n) if addr != AddrFact::Unknown && bytes != 0 => end = end.max(n),
                    _ => bounded = true,
                }
                facts.accesses.push((LdsAccess { inst: id, addr, bytes, kind }, whole));
            }
            transfer(&mut st, inst, arch);
            if matches!(inst.effects.control, crate::effects::Control::Jump | crate::effects::Control::EndPgm
                | crate::effects::Control::Trap | crate::effects::Control::Halt) { break; }
        }
    }
    facts.max_end = if bounded { None } else { Some(end) };
    Ok(LdsAnalysis { facts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    use crate::cfg::Body;
    use crate::inst::{Form, Inst, VmemForm};
    use crate::operand::Modifiers;
    use crate::provenance::Provenance;

    fn row(name: &str) -> (crate::inst::Opcode, Form) {
        let row = crate::isa::gfx12().iter().find(|row| row.name == name).expect(name);
        (row.op, row.form)
    }

    fn v(base: u16) -> Operand {
        Operand::Reg(crate::reg::RegRef { kind: Kind::V, base, len: 1 })
    }

    fn inst(name: &str, operands: Vec<Operand>) -> Inst {
        let (op, form) = row(name);
        Inst::from_parts(
            Arch::Gfx1201,
            op,
            form,
            crate::inst::FormFields::None,
            SmallVec::from_vec(operands),
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"))
    }

    fn body_of(insts: Vec<Inst>) -> Body {
        let mut body = Body::default();
        for inst in insts {
            let id = body.insts.insert(inst);
            body.layout.push(id);
        }
        body
    }

    #[test]
    fn constant_store_resolves_with_own_offset() {
        let mov = inst("v_mov_b32_e32", vec![v(10), Operand::Literal(0x1000)]);
        let store = inst(
            "ds_store_b32",
            vec![v(10), v(11), Operand::Imm(ImmField::DsOffset(64))],
        );
        let body = body_of(vec![mov, store]);
        let analysis = analyze(&body, Arch::Gfx1201).unwrap();
        assert_eq!(analysis.facts.accesses.len(), 1);
        assert_eq!(
            analysis.facts.accesses[0].0.addr,
            AddrFact::Const(0x1040),
            "{:?}",
            analysis.facts.accesses[0].0
        );
        assert_eq!(analysis.facts.max_end, Some(0x1044));
    }

    #[test]
    fn computed_address_is_unknown_with_no_max_end() {
        let store = inst("ds_store_b32", vec![v(10), v(11)]);
        let body = body_of(vec![store]);
        let analysis = analyze(&body, Arch::Gfx1201).unwrap();
        assert_eq!(analysis.facts.accesses[0].0.addr, AddrFact::Unknown);
        assert_eq!(analysis.facts.max_end, None);
    }

    fn s(base: u16) -> Operand {
        Operand::Reg(crate::reg::RegRef { kind: Kind::S, base, len: 1 })
    }
    fn literal(n: u32) -> Operand { Operand::Literal(n) }
    fn bounded_address(body: &Body) -> AddrFact {
        analyze(body, Arch::Gfx1201).unwrap().facts.accesses[0].0.addr
    }

    #[test]
    fn lane_wave_scalar_mul_and_mad_address_chain() {
        let body = body_of(vec![
            inst("s_getreg_b32", vec![s(4), Operand::Hwreg(crate::operand::HwReg { id: 4, offset: 0, size: 5 })]),
            inst("s_mul_i32", vec![s(4), s(4), literal(128)]),
            inst("v_mbcnt_lo_u32_b32", vec![v(2), literal(u32::MAX), literal(0)]),
            inst("v_mad_u32_u24", vec![v(3), v(2), literal(4), s(4)]),
            inst("ds_store_b32", vec![v(3), v(11), Operand::Imm(ImmField::DsOffset(64))]),
        ]);
        assert_eq!(bounded_address(&body), AddrFact::Bounded { lo: 64, hi: 4160 });
        assert_eq!(analyze(&body, Arch::Gfx1201).unwrap().facts.max_end, Some(4164));
    }

    #[test]
    fn loop_induction_reaches_a_sound_bounded_fixed_point() {
        use crate::cfg::{Block, BlockId, Cond, Terminator};
        let mut body = body_of(vec![
            inst("s_mov_b32", vec![s(4), literal(0)]),
            inst("v_mov_b32_e32", vec![v(2), s(4)]),
            inst("v_lshlrev_b32_e32", vec![v(2), literal(2), v(2)]),
            inst("ds_store_b32", vec![v(2), v(11)]),
            inst("s_add_co_u32", vec![s(4), s(4), literal(1)]),
            inst("s_cmp_lt_u32", vec![s(4), literal(16)]),
            inst("s_cbranch_scc1", vec![Operand::Label(BlockId(1))]),
        ]);
        body.blocks = vec![
            Block { id: BlockId(0), range: (0,1), term: Terminator::FallThrough, preds: smallvec::smallvec![], succs: smallvec::smallvec![BlockId(1)] },
            Block { id: BlockId(1), range: (1,7), term: Terminator::Branch { cond: Cond::Scc1, taken: BlockId(1), fallthrough: BlockId(2) }, preds: smallvec::smallvec![BlockId(0),BlockId(1)], succs: smallvec::smallvec![BlockId(1),BlockId(2)] },
            Block { id: BlockId(2), range: (7,7), term: Terminator::EndPgm, preds: smallvec::smallvec![BlockId(1)], succs: smallvec::smallvec![] },
        ];
        assert_eq!(bounded_address(&body), AddrFact::Bounded { lo: 0, hi: 60 });
    }

    #[test]
    fn mid_block_branch_uses_values_at_issue_not_at_block_end() {
        use crate::cfg::{Block, BlockId, Terminator};
        let mut body = body_of(vec![
            inst("v_mov_b32_e32", vec![v(2), literal(16)]),
            inst("s_cbranch_scc1", vec![Operand::Label(BlockId(1))]),
            inst("v_mov_b32_e32", vec![v(2), literal(128)]),
            inst("s_branch", vec![Operand::Label(BlockId(2))]),
            inst("ds_store_b32", vec![v(2), v(11)]),
            inst("s_branch", vec![Operand::Label(BlockId(2))]),
            inst("ds_store_b32", vec![v(2), v(11)]),
        ]);
        body.blocks = vec![
            Block { id: BlockId(0), range: (0,4), term: Terminator::Jump(BlockId(2)), preds: smallvec::smallvec![], succs: smallvec::smallvec![BlockId(1),BlockId(2)] },
            Block { id: BlockId(1), range: (4,6), term: Terminator::Jump(BlockId(2)), preds: smallvec::smallvec![BlockId(0)], succs: smallvec::smallvec![BlockId(2)] },
            Block { id: BlockId(2), range: (6,7), term: Terminator::EndPgm, preds: smallvec::smallvec![BlockId(0),BlockId(1)], succs: smallvec::smallvec![] },
        ];
        let facts = analyze(&body, Arch::Gfx1201).unwrap().facts;
        assert_eq!(facts.accesses[0].0.addr, AddrFact::Const(16));
        assert_eq!(facts.accesses[1].0.addr, AddrFact::Bounded { lo: 16, hi: 128 });
    }

    #[test]
    fn unknown_predecessor_and_memory_write_do_not_invent_bounds() {
        use crate::cfg::{Block, BlockId, Cond, Terminator};
        let mut body = body_of(vec![
            inst("s_cbranch_scc1", vec![Operand::Label(BlockId(2))]),
            inst("v_mov_b32_e32", vec![v(2), literal(16)]),
            inst("ds_store_b32", vec![v(2), v(11)]),
        ]);
        body.blocks = vec![
            Block { id: BlockId(0), range: (0,1), term: Terminator::Branch { cond: Cond::Scc1, taken: BlockId(2), fallthrough: BlockId(1) }, preds: smallvec::smallvec![], succs: smallvec::smallvec![BlockId(1),BlockId(2)] },
            Block { id: BlockId(1), range: (1,2), term: Terminator::FallThrough, preds: smallvec::smallvec![BlockId(0)], succs: smallvec::smallvec![BlockId(2)] },
            Block { id: BlockId(2), range: (2,3), term: Terminator::EndPgm, preds: smallvec::smallvec![BlockId(0),BlockId(1)], succs: smallvec::smallvec![] },
        ];
        assert_eq!(bounded_address(&body), AddrFact::Unknown);
        let body = body_of(vec![
            inst("v_mov_b32_e32", vec![v(2), literal(16)]),
            inst("ds_load_b32", vec![v(2), v(10)]),
            inst("ds_store_b32", vec![v(2), v(11)]),
        ]);
        let facts = analyze(&body, Arch::Gfx1201).unwrap().facts;
        assert_eq!(facts.accesses[1].0.addr, AddrFact::Unknown);
    }

    #[test]
    fn partial_exec_keeps_old_addresses_and_offset_pairs_use_byte_units() {
        let body = body_of(vec![
            inst("v_mov_b32_e32", vec![v(2), literal(1024)]),
            inst("s_and_saveexec_b32", vec![s(4), s(5)]),
            inst("v_mov_b32_e32", vec![v(2), literal(16)]),
            inst("ds_store_b32", vec![v(2), v(11)]),
        ]);
        assert_eq!(bounded_address(&body), AddrFact::Bounded { lo: 16, hi: 1024 });
        let body = body_of(vec![
            inst("v_mov_b32_e32", vec![v(2), literal(128)]),
            inst("ds_load_2addr_b64", vec![
                Operand::Reg(crate::reg::RegRef { kind: Kind::V, base: 4, len: 4 }), v(2),
                Operand::Imm(ImmField::DsOffset0(3)), Operand::Imm(ImmField::DsOffset1(1)),
            ]),
        ]);
        assert_eq!(bounded_address(&body), AddrFact::Bounded { lo: 136, hi: 152 });
        assert_eq!(analyze(&body, Arch::Gfx1201).unwrap().facts.max_end, Some(160));
    }

    #[test]
    fn overflow_and_u24_truncation_are_not_unsigned_linear_arithmetic() {
        assert_eq!(Value { lo: u32::MAX - 4, hi: u32::MAX }.add(Value::constant(8)), Value::UNKNOWN);
        assert_eq!(Value { lo: 1, hi: u32::MAX }.mul(Value::constant(4)), Value::UNKNOWN);
        assert_eq!(evaluate("v_mad_u32_u24", &[Value::constant(0x1000001),Value::constant(4),Value::constant(8)]), Value::constant(12));
    }

    #[test]
    fn vmem_instructions_are_not_lds_accesses() {
        let load = inst(
            "global_load_b32",
            vec![
                v(1),
                Operand::Reg(crate::reg::RegRef { kind: Kind::V, base: 2, len: 2 }),
                Operand::Vmem(crate::operand::VmemToken::Off),
            ],
        );
        let body = body_of(vec![load]);
        let analysis = analyze(&body, Arch::Gfx1201).unwrap();
        assert!(analysis.facts.accesses.is_empty());
        assert_eq!(analysis.facts.max_end, Some(0));
    }

    #[allow(dead_code)]
    fn form_shape(_form: VmemForm) {}
}

#[cfg(test)]
mod kt48_tests {
    use crate::cfg::Body;
    use crate::inst::Arch;
    use crate::lds::AddrFact;
    use crate::passes::cfg::build_blocks;

    fn kt48_body() -> Body {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        let words: Vec<u32> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = crate::codec::gfx12::decode(&words[index..]).expect("KT48 decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        build_blocks(&mut body, crate::inst::Arch::Gfx1201).expect("CFG");
        body
    }

    /// Some KT48 addresses remain unresolved even after CFG value tracking,
    /// so the launch-contract obligation remains open. Its dynamic LDS
    /// allocation comes from the dispatch packet, never the object.
    #[test]
    fn kt48_lds_facts() {
        let body = kt48_body();
        let analysis = super::analyze(&body, Arch::Gfx1201).unwrap();
        assert!(!analysis.facts.accesses.is_empty());
        assert!(
            analysis.facts.accesses.iter().any(|(access, _)| access.addr == AddrFact::Unknown),
            "expected computed LDS addresses"
        );
        assert_eq!(analysis.facts.max_end, None);
        assert_eq!(analysis.facts.segments.len(), 1);
    }
}
