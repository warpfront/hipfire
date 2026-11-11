// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Workgroup uniformity and barrier participation of a lifted hipcc body (the
//! geometry obligation of [`super::region`]).
//!
//! A region hosted by a multi-wave pool dispatcher must make every wave execute
//! the same barriers: a wave that leaves the body early would pair its next
//! dispatcher barrier with another wave's body barrier. This module proves it by
//! abstract interpretation over a product graph of (block, EXEC state):
//!
//! * **SGPRs, SCC, VCC**: workgroup-uniform (the same value in every wave) or
//!   per-wave. The kernarg segment pointer `s[0:1]`, the workgroup ids
//!   (`ttmp7`/`ttmp9`) and constants are uniform; a scalar result is uniform when
//!   its sources are; `s_load` through the kernarg pointer is uniform (the segment
//!   is read-only); any other uniform-address load is uniform under the source
//!   program's data-race freedom (the HIP memory model), which the verdict
//!   reports. Lane masks additionally track whether they are 0, all-ones, or
//!   per-wave one of the two (`Coh`).
//! * **VGPRs**: uniform, wave-uniform (equal across the lanes of each wave),
//!   `v0` = the flat thread id (`Tid`, the region's one workitem-id axis), or
//!   varying. `v_lshrrev_b32 v, k>=5, Tid` is wave-uniform; `v_readfirstlane` of a
//!   uniform VGPR is uniform; a compare of `Tid` against a constant whose
//!   threshold is a multiple of 32 is wave-coherent (every wave's mask is 0 or
//!   all-ones). Wave32 VALU lane-mask destinations write one SGPR, not the pair the
//!   effects tables over-approximate.
//! * **EXEC**: `Full`, `Empty`, `Coh` (each wave full or empty) or `Var` (lanes
//!   mixed in some wave). It is part of the node: `s_cbranch_execz`/`execnz`
//!   splits a `Coh` node into an `Empty` and a `Full` successor and prunes the
//!   infeasible edge of a `Full`/`Empty` node, so the structurizer's if/else over
//!   a wave-coherent predicate (each wave runs exactly one side) is not mistaken
//!   for a path that runs both or neither. `s_or_b32 exec, exec, s` restores
//!   `Full` when `s` is the saved full mask or the complement recorded by
//!   `s_and_not1_saveexec` / the loop break-mask idiom.
//! * **Sync dependence**: values defined between a per-wave branch and its
//!   immediate post-dominator become per-wave at the post-dominator; the analysis
//!   reruns until that taint is stable.
//!
//! The barrier check then runs on the product graph: every per-wave branch node
//! reaches its immediate post-dominator with the same barrier count on every
//! path, and no barrier lies on a cycle inside that region.

use std::collections::{BTreeMap, BTreeSet};

use peacemaker_ir::cfg::{Body, Cond, Terminator};
use peacemaker_ir::effects::{Control, ImplicitSet};
use peacemaker_ir::inst::{Arch, Inst};
use peacemaker_ir::operand::{InlineConst, Operand, Special};
use peacemaker_ir::reg::{Kind, RegRef};

const NS: usize = 128;
const NV: usize = 256;
const VCC: usize = 106;
const M0: usize = 125;
const EXEC: u16 = 126;
const LANES: i64 = 32;

/// The shape of a 32-bit lane mask within one wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mask {
    Zero,
    Full,
    /// 0 or all-ones in each wave
    Coh,
    Other,
    /// The mask (or its complement) a compare in this block produced: block-local, so
    /// `x | ~x` and `x & ~x` are exact (the structurizer's if/else); `coh` when it is 0
    /// or all-ones in each wave. Atoms do not cross block boundaries ([`St::flow`]).
    Atom {
        id: u32,
        neg: bool,
        coh: bool,
    },
}
impl Mask {
    fn constant(self) -> bool {
        matches!(self, Mask::Zero | Mask::Full)
    }
    /// Forget the atom's identity.
    fn kind(self) -> Self {
        match self {
            Mask::Atom { coh: true, .. } => Mask::Coh,
            Mask::Atom { coh: false, .. } => Mask::Other,
            m => m,
        }
    }
    fn join(self, o: Self) -> Self {
        if self == o {
            return self;
        }
        let (a, b) = (self.kind(), o.kind());
        if a == b {
            a
        } else if a != Mask::Other && b != Mask::Other {
            Mask::Coh
        } else {
            Mask::Other
        }
    }
    fn not(self) -> Self {
        match self {
            Mask::Zero => Mask::Full,
            Mask::Full => Mask::Zero,
            Mask::Atom { id, neg, coh } => Mask::Atom { id, neg: !neg, coh },
            m => m,
        }
    }
    /// `Some(same polarity)` when both are the same atom.
    fn same_atom(self, o: Self) -> Option<bool> {
        match (self, o) {
            (Mask::Atom { id: a, neg: n, .. }, Mask::Atom { id: b, neg: m, .. }) if a == b => {
                Some(n == m)
            }
            _ => None,
        }
    }
    fn and(self, o: Self) -> Self {
        match (self, o) {
            (Mask::Zero, _) | (_, Mask::Zero) => Mask::Zero,
            (Mask::Full, m) | (m, Mask::Full) => m,
            _ => match self.same_atom(o) {
                Some(true) => self,
                Some(false) => Mask::Zero,
                None => match (self.kind(), o.kind()) {
                    (Mask::Coh, Mask::Coh) => Mask::Coh,
                    _ => Mask::Other,
                },
            },
        }
    }
    fn or(self, o: Self) -> Self {
        self.not().and(o.not()).not()
    }
    fn xor(self, o: Self) -> Self {
        match (self, o) {
            (Mask::Zero, m) | (m, Mask::Zero) => m,
            (Mask::Full, m) | (m, Mask::Full) => m.not(),
            _ => match self.same_atom(o) {
                Some(true) => Mask::Zero,
                Some(false) => Mask::Full,
                None => match (self.kind(), o.kind()) {
                    (Mask::Coh, Mask::Coh) => Mask::Coh,
                    _ => Mask::Other,
                },
            },
        }
    }
}

/// A scalar value: workgroup-uniform or not, its lane-mask shape, whether it is the
/// kernarg segment pointer, and whether its uniformity rests on data-race freedom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SVal {
    uni: bool,
    mask: Mask,
    karg: bool,
    drf: bool,
}
impl SVal {
    const VAR: Self = Self {
        uni: false,
        mask: Mask::Other,
        karg: false,
        drf: false,
    };
    const UNI: Self = Self {
        uni: true,
        mask: Mask::Other,
        karg: false,
        drf: false,
    };
    fn mask(mask: Mask, uni: bool) -> Self {
        Self {
            uni: uni || mask.constant(),
            mask,
            karg: false,
            drf: false,
        }
    }
    fn join(self, o: Self) -> Self {
        let mask = self.mask.join(o.mask);
        Self {
            uni: (self.uni && o.uni) || mask.constant(),
            mask,
            karg: self.karg && o.karg,
            drf: self.drf || o.drf,
        }
    }
    /// The combination of sources: uniform when all are.
    fn of(srcs: &[SVal]) -> Self {
        Self {
            uni: srcs.iter().all(|s| s.uni),
            mask: Mask::Other,
            karg: false,
            drf: srcs.iter().any(|s| s.drf),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VK {
    Uni,
    Wave,
    Tid,
    Var,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VVal {
    k: VK,
    drf: bool,
}
impl VVal {
    const VAR: Self = Self {
        k: VK::Var,
        drf: false,
    };
    fn join(self, o: Self) -> Self {
        let k = if self.k == o.k {
            self.k
        } else if self.k == VK::Tid || o.k == VK::Tid {
            VK::Var
        } else {
            self.k.max(o.k)
        };
        Self {
            k,
            drf: self.drf || o.drf,
        }
    }
    /// The result class of an elementwise operation on `srcs`.
    fn of(srcs: &[VVal]) -> Self {
        let k = srcs
            .iter()
            .map(|v| if v.k == VK::Tid { VK::Var } else { v.k })
            .max()
            .unwrap_or(VK::Uni);
        Self {
            k,
            drf: srcs.iter().any(|v| v.drf),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Ex {
    Full,
    Empty,
    Coh,
    Var,
}

#[derive(Clone, Debug, PartialEq)]
struct St {
    s: Vec<SVal>,
    v: Vec<VVal>,
    scc: SVal,
    /// For `Ex::Coh`: every wave has the same EXEC (all full or all empty).
    exec_uni: bool,
    /// An SGPR `s` with `exec | s == all-ones` (the saved complement of EXEC).
    compl: Option<u16>,
    /// EXEC as a block-local atom, when a compare in this block produced it.
    exec_atom: Option<Mask>,
}
impl St {
    fn entry() -> Self {
        let mut s = vec![SVal::VAR; NS];
        for r in &mut s[0..2] {
            *r = SVal {
                karg: true,
                ..SVal::UNI
            };
        }
        let mut v = vec![VVal::VAR; NV];
        v[0] = VVal {
            k: VK::Tid,
            drf: false,
        };
        St {
            s,
            v,
            scc: SVal::VAR,
            exec_uni: true,
            compl: None,
            exec_atom: None,
        }
    }
    fn join(&self, o: &Self) -> Self {
        St {
            s: self.s.iter().zip(&o.s).map(|(a, b)| a.join(*b)).collect(),
            v: self.v.iter().zip(&o.v).map(|(a, b)| a.join(*b)).collect(),
            scc: self.scc.join(o.scc),
            exec_uni: self.exec_uni && o.exec_uni,
            compl: if self.compl == o.compl {
                self.compl
            } else {
                None
            },
            exec_atom: None,
        }
    }
    /// On an edge where EXEC is known (`Empty` after `s_cbranch_execz` taken, `Full` after a
    /// wave-coherent EXEC was found non-empty), the block-local atom EXEC holds, and every
    /// mask equal to it or its complement, is that constant.
    fn refine(&self, to: Ex) -> Self {
        let mut st = self.clone();
        let Some(a) = self.exec_atom else {
            return st;
        };
        let full = match to {
            Ex::Empty => false,
            Ex::Full => true,
            _ => return st,
        };
        let fix = |m: &mut SVal| {
            if let Some(same) = m.mask.same_atom(a) {
                m.mask = if same == full { Mask::Full } else { Mask::Zero };
                m.uni = true;
            }
        };
        for r in &mut st.s {
            fix(r);
        }
        st
    }
    /// The state a successor block starts from: block-local atoms lose their identity
    /// (a compare re-executed in a loop is a new value).
    fn flow(&self) -> Self {
        let mut st = self.clone();
        for r in &mut st.s {
            r.mask = r.mask.kind();
        }
        st.scc.mask = st.scc.mask.kind();
        st.exec_atom = None;
        st
    }
    fn sreg(&self, r: RegRef) -> SVal {
        (r.base..r.base + u16::from(r.len))
            .map(|i| self.s.get(usize::from(i)).copied().unwrap_or(SVal::VAR))
            .reduce(SVal::join)
            .unwrap_or(SVal::VAR)
    }
    fn vreg(&self, r: RegRef) -> VVal {
        (r.base..r.base + u16::from(r.len))
            .map(|i| self.v.get(usize::from(i)).copied().unwrap_or(VVal::VAR))
            .reduce(VVal::join)
            .unwrap_or(VVal::VAR)
    }
    fn exec_val(&self, ex: Ex) -> SVal {
        match ex {
            Ex::Full => SVal::mask(Mask::Full, true),
            Ex::Empty => SVal::mask(Mask::Zero, true),
            Ex::Coh | Ex::Var if self.exec_atom.is_some() => SVal::mask(
                self.exec_atom.expect("checked"),
                ex == Ex::Coh && self.exec_uni,
            ),
            Ex::Coh => SVal::mask(Mask::Coh, self.exec_uni),
            Ex::Var => SVal::VAR,
        }
    }
    fn set_exec(&mut self, ex: &mut Ex, v: SVal) {
        (*ex, self.exec_uni) = match v.mask.kind() {
            Mask::Full => (Ex::Full, true),
            Mask::Zero => (Ex::Empty, true),
            Mask::Coh => (Ex::Coh, v.uni),
            _ => (Ex::Var, false),
        };
        self.exec_atom = matches!(v.mask, Mask::Atom { .. }).then_some(v.mask);
        self.compl = None;
    }
}

/// Sync-dependence taint applied at a node's entry.
#[derive(Clone, Debug, Default, PartialEq)]
struct Taint {
    s: BTreeSet<u16>,
    v: BTreeSet<u16>,
    scc: bool,
}
impl Taint {
    fn apply(&self, st: &mut St) {
        for &i in &self.s {
            let r = &mut st.s[usize::from(i)];
            if !r.mask.constant() {
                r.uni = false;
                r.karg = false;
            }
        }
        for &i in &self.v {
            let r = &mut st.v[usize::from(i)];
            r.k = match r.k {
                VK::Uni | VK::Wave => VK::Wave,
                _ => VK::Var,
            };
        }
        if self.scc {
            st.scc.uni = st.scc.mask.constant();
        }
        st.exec_uni = false;
    }
    fn union(&mut self, o: &Taint) -> bool {
        let before = (self.s.len(), self.v.len(), self.scc);
        self.s.extend(o.s.iter().copied());
        self.v.extend(o.v.iter().copied());
        self.scc |= o.scc;
        before != (self.s.len(), self.v.len(), self.scc)
    }
}

fn sop(st: &St, ex: Ex, op: &Operand) -> SVal {
    match op {
        Operand::Reg(r) | Operand::Half(r, _) => match r.kind {
            Kind::S => st.sreg(*r),
            Kind::Ttmp => SVal::UNI,
            Kind::V => SVal::VAR,
        },
        Operand::Special(sp) => match sp {
            Special::Vcc | Special::VccLo => st.s[VCC],
            Special::VccHi => st.s[VCC + 1],
            Special::Exec | Special::ExecLo => st.exec_val(ex),
            Special::Scc => st.scc,
            Special::M0 => st.s[M0],
            Special::Null => SVal::mask(Mask::Zero, true),
            _ => SVal::UNI,
        },
        Operand::Inline(InlineConst::Integer(0)) | Operand::Literal(0) => {
            SVal::mask(Mask::Zero, true)
        }
        Operand::Inline(InlineConst::Integer(-1)) | Operand::Literal(u32::MAX) => {
            SVal::mask(Mask::Full, true)
        }
        _ => SVal::UNI,
    }
}

fn vop(st: &St, ex: Ex, op: &Operand) -> VVal {
    match op {
        Operand::Reg(r) | Operand::Half(r, _) if r.kind == Kind::V => st.vreg(*r),
        _ => {
            let s = sop(st, ex, op);
            VVal {
                k: if s.uni { VK::Uni } else { VK::Wave },
                drf: s.drf,
            }
        }
    }
}

fn konst(op: &Operand) -> Option<i64> {
    match op {
        Operand::Inline(InlineConst::Integer(n)) => Some(i64::from(*n)),
        Operand::Literal(u) => Some(i64::from(*u)),
        _ => None,
    }
}

/// Registers an instruction writes, by kind (SGPR numbers include VCC 106/107, M0 125
/// and EXEC 126/127).
fn written(st_ops: &[&Operand]) -> (BTreeSet<u16>, BTreeSet<u16>) {
    let (mut s, mut v) = (BTreeSet::new(), BTreeSet::new());
    for op in st_ops {
        match op {
            Operand::Reg(r) | Operand::Half(r, _) => {
                let set = match r.kind {
                    Kind::S => &mut s,
                    Kind::V => &mut v,
                    Kind::Ttmp => continue,
                };
                set.extend(r.base..r.base + u16::from(r.len));
            }
            Operand::Special(Special::Vcc) => {
                s.extend([106, 107]);
            }
            Operand::Special(Special::VccLo) => {
                s.insert(106);
            }
            Operand::Special(Special::VccHi) => {
                s.insert(107);
            }
            Operand::Special(Special::Exec) => {
                s.extend([126, 127]);
            }
            Operand::Special(Special::ExecLo) => {
                s.insert(126);
            }
            Operand::Special(Special::M0) => {
                s.insert(125);
            }
            _ => {}
        }
    }
    (s, v)
}

/// The threshold `t` of a compare of `Tid` against a constant (the predicate holds
/// exactly for `tid >= t` or exactly for `tid < t`), when it is one.
fn tid_threshold(name: &str, a: &Operand, b: &Operand, st: &St, ex: Ex) -> Option<i64> {
    let op = name
        .strip_prefix("v_cmpx_")
        .or_else(|| name.strip_prefix("v_cmp_"))?;
    let (rel, ty) = op.split_once('_')?;
    if !ty.starts_with("u32") && !ty.starts_with("i32") {
        return None;
    }
    let (tid_first, k) = match (vop(st, ex, a).k, vop(st, ex, b).k) {
        (VK::Tid, _) => (true, konst(b)?),
        (_, VK::Tid) => (false, konst(a)?),
        _ => return None,
    };
    if !(0..=i64::from(u32::MAX >> 1)).contains(&k) {
        return None;
    }
    // tid OP k  or  k OP tid
    Some(match (rel, tid_first) {
        ("lt", true) | ("ge", true) | ("gt", false) | ("le", false) => k,
        ("le", true) | ("gt", true) | ("ge", false) | ("lt", false) => k + 1,
        _ => return None,
    })
}

/// One wave32 instruction's effect on the abstract state.
fn step(inst: &Inst, id: u32, st: &mut St, ex: &mut Ex, block: u32) {
    let Some(row) = peacemaker_ir::isa::lookup_inst(Arch::Gfx1201, &inst).ok() else {
        havoc(inst, st, ex, &BTreeSet::new(), &BTreeSet::new());
        return;
    };
    let name = row.name;
    let defset: Vec<&str> = row.defs.split(',').collect();
    let slots: Vec<&str> = row.slots(true).map(|(n, _)| n).collect();
    let (mut dsts, mut srcs): (Vec<&Operand>, Vec<&Operand>) = (Vec::new(), Vec::new());
    for (i, op) in inst.operands.iter().enumerate() {
        match slots.get(i) {
            Some(s) if defset.contains(s) => dsts.push(op),
            Some(_) => srcs.push(op),
            None => {}
        }
    }
    let e = &inst.effects;
    let atomic = name.contains("atomic");
    let (mut cs, cv) = written(&dsts);
    if atomic || name.starts_with("s_getreg") || name.starts_with("s_sendmsg_rtn") {
        // per-wave by construction: RMW results, hardware registers, timers
        havoc(inst, st, ex, &BTreeSet::new(), &BTreeSet::new());
        return;
    }
    let ssrc: Vec<SVal> = srcs.iter().map(|o| sop(st, *ex, o)).collect();
    let mut implicit_srcs = Vec::new();
    if e.implicit.reads & ImplicitSet::SCC != 0 {
        implicit_srcs.push(st.scc);
    }
    if e.implicit.reads & ImplicitSet::VCC != 0 {
        implicit_srcs.push(st.s[VCC]);
    }
    let old_ex = *ex;
    let exec_now = st.exec_val(old_ex);
    if name.starts_with("s_") {
        let all: Vec<SVal> = ssrc.iter().chain(&implicit_srcs).copied().collect();
        let mut val = SVal::of(&all);
        let m = |i: usize| ssrc.get(i).map_or(Mask::Other, |s| s.mask);
        let compl_keep = st.compl;
        let mut new_compl = None;
        let mut exec_rule = false;
        if name.starts_with("s_load") || name.starts_with("s_buffer_load") {
            let base = ssrc.first().copied().unwrap_or(SVal::VAR);
            val = SVal {
                drf: val.drf || !base.karg,
                ..SVal::of(&all)
            };
        } else if name.contains("saveexec") {
            let a = ssrc.first().copied().unwrap_or(SVal::VAR);
            let new = match name.trim_end_matches("_b32") {
                "s_and_saveexec" => exec_now.mask.and(a.mask),
                "s_or_saveexec" => exec_now.mask.or(a.mask),
                "s_xor_saveexec" => exec_now.mask.xor(a.mask),
                "s_and_not1_saveexec" => a.mask.and(exec_now.mask.not()),
                "s_and_not0_saveexec" => a.mask.not().and(exec_now.mask),
                "s_or_not1_saveexec" => a.mask.or(exec_now.mask.not()),
                "s_or_not0_saveexec" => a.mask.not().or(exec_now.mask),
                _ => Mask::Other,
            };
            let uni = a.uni && exec_now.uni;
            for d in &dsts {
                write_s(st, d, exec_now);
            }
            st.set_exec(ex, SVal::mask(new, uni));
            if name == "s_and_not1_saveexec_b32" && a.mask == Mask::Full {
                // exec = ~old and the destination holds old: their union is all-ones
                if let Some(Operand::Reg(r)) = dsts.first() {
                    new_compl = Some(r.base);
                }
            }
            st.scc = SVal::mask(Mask::Other, SVal::mask(new, uni).uni);
            exec_rule = true;
            cs.extend([126, 127]);
        } else if let Some(mask) = match name {
            "s_mov_b32" | "s_mov_b64" => Some(m(0)),
            "s_not_b32" => Some(m(0).not()),
            "s_and_b32" => Some(m(0).and(m(1))),
            "s_or_b32" => Some(m(0).or(m(1))),
            "s_xor_b32" => Some(m(0).xor(m(1))),
            "s_and_not1_b32" => Some(m(0).and(m(1).not())),
            "s_or_not1_b32" => Some(m(0).or(m(1).not())),
            "s_nand_b32" => Some(m(0).and(m(1)).not()),
            "s_nor_b32" => Some(m(0).or(m(1)).not()),
            "s_xnor_b32" => Some(m(0).xor(m(1)).not()),
            "s_cselect_b32" => Some(m(0).join(m(1))),
            _ => None,
        } {
            val.mask = mask;
            val.uni = val.uni || mask.constant();
            if name.starts_with("s_mov") {
                val = ssrc.first().copied().unwrap_or(SVal::VAR);
            }
            let reg = |o: Option<&&Operand>| match o {
                Some(Operand::Reg(r)) if r.kind == Kind::S && r.len == 1 => Some(r.base),
                _ => None,
            };
            let dst_exec = matches!(
                dsts.first(),
                Some(Operand::Special(Special::ExecLo | Special::Exec))
            );
            let src_exec = |i: usize| {
                matches!(
                    srcs.get(i),
                    Some(Operand::Special(Special::ExecLo | Special::Exec))
                )
            };
            if dst_exec && name == "s_or_b32" {
                let other = if src_exec(0) {
                    reg(srcs.get(1))
                } else {
                    reg(srcs.first())
                };
                if other.is_some() && other == compl_keep {
                    val = SVal::mask(Mask::Full, true);
                }
            }
            if dst_exec && name == "s_and_not1_b32" && src_exec(0) {
                // exec & ~s: s keeps exec's complement if it was, or becomes it from a full exec
                if let Some(sx) = reg(srcs.get(1)) {
                    if compl_keep == Some(sx) || old_ex == Ex::Full {
                        new_compl = Some(sx);
                    }
                }
            }
            if !dst_exec {
                if let Some(d) = reg(dsts.first()) {
                    let ors_into_itself = name == "s_or_b32"
                        && (reg(srcs.first()) == Some(d) || reg(srcs.get(1)) == Some(d));
                    if compl_keep == Some(d) && ors_into_itself {
                        new_compl = Some(d);
                    } else if name == "s_mov_b32" && m(0) == Mask::Zero && old_ex == Ex::Full {
                        new_compl = Some(d);
                    } else if compl_keep != Some(d) {
                        new_compl = compl_keep;
                    }
                }
            }
        }
        if !exec_rule {
            for d in &dsts {
                write_s(st, d, val);
                if matches!(d, Operand::Special(Special::ExecLo | Special::Exec)) {
                    st.set_exec(ex, val);
                }
            }
            if e.implicit.writes & ImplicitSet::SCC != 0 {
                st.scc = SVal {
                    mask: Mask::Other,
                    karg: false,
                    ..SVal::of(&[SVal {
                        mask: Mask::Other,
                        ..val
                    }])
                };
            }
        }
        st.compl = new_compl;
        havoc(inst, st, ex, &cs, &cv);
        return;
    }
    // VALU and memory
    let vsrc: Vec<VVal> = srcs.iter().map(|o| vop(st, old_ex, o)).collect();
    let mem = name.starts_with("global_")
        || name.starts_with("buffer_")
        || name.starts_with("ds_")
        || name.starts_with("flat_")
        || name.starts_with("scratch_");
    let mut res = VVal::of(&vsrc);
    if mem {
        let load = name.contains("load") && !name.starts_with("ds_bpermute");
        res = if load && res.k <= VK::Wave {
            VVal {
                k: res.k,
                drf: true,
            }
        } else {
            VVal::VAR
        };
    } else if name == "v_lshrrev_b32_e32" || name == "v_lshrrev_b32_e64" {
        if let (Some(k), Some(VK::Tid)) = (
            srcs.first().and_then(|o| konst(o)),
            vsrc.get(1).map(|v| v.k),
        ) {
            if (5..32).contains(&k) {
                res = VVal {
                    k: VK::Wave,
                    drf: false,
                };
            }
        }
    } else if name.starts_with("v_mov_b32") && !name.contains("dpp") {
        res = vsrc.first().copied().unwrap_or(VVal::VAR);
    } else if name.contains("dpp") || name == "v_writelane_b32" {
        // a partial-lane write keeps the old lanes
        let olds: Vec<VVal> = cv.iter().map(|&i| st.v[usize::from(i)]).collect();
        res = VVal::of(&[res, VVal::of(&olds)]);
    }
    let mut cs = cs;
    if name == "v_readfirstlane_b32" || name == "v_readlane_b32" {
        let uni = vsrc.iter().all(|v| v.k == VK::Uni);
        for d in &dsts {
            write_s(
                st,
                d,
                SVal {
                    uni,
                    mask: Mask::Other,
                    karg: false,
                    drf: res.drf,
                },
            );
        }
        havoc(inst, st, ex, &cs, &cv);
        return;
    }
    if name.starts_with("v_cmp") {
        let base = match (srcs.first(), srcs.get(1)) {
            _ if vsrc.iter().all(|v| v.k == VK::Uni) => SVal::mask(Mask::Coh, true),
            _ if vsrc.iter().all(|v| v.k <= VK::Wave) => SVal::mask(Mask::Coh, false),
            (Some(a), Some(b)) => match tid_threshold(name, a, b, st, old_ex) {
                Some(t) if t <= 0 || t >= i64::from(block) => SVal::mask(Mask::Coh, true),
                Some(t) if t % LANES == 0 => SVal::mask(Mask::Coh, false),
                _ => SVal::VAR,
            },
            _ => SVal::VAR,
        };
        let base = if base.mask == Mask::Coh || base.mask == Mask::Other {
            SVal {
                mask: Mask::Atom {
                    id,
                    neg: false,
                    coh: base.mask == Mask::Coh,
                },
                ..base
            }
        } else {
            base
        };
        // inactive lanes write 0
        let mask = base.mask.and(exec_now.mask);
        let val = SVal {
            drf: res.drf,
            ..SVal::mask(mask, base.uni && exec_now.uni)
        };
        if name.starts_with("v_cmpx") {
            st.set_exec(ex, val);
            cs.extend([126, 127]);
        } else if dsts.is_empty() || e.implicit.writes & ImplicitSet::VCC != 0 {
            st.s[VCC] = val;
            cs.insert(106);
        }
        for d in dsts
            .iter()
            .filter(|d| !matches!(d, Operand::Reg(r) if r.kind == Kind::V))
        {
            write_mask_dst(st, d, val, &mut cs);
        }
        havoc(inst, st, ex, &cs, &cv);
        return;
    }
    // elementwise results; lane-mask/carry SGPR results
    let carry = SVal {
        drf: res.drf,
        ..SVal::mask(Mask::Other, res.k == VK::Uni && old_ex == Ex::Full)
    };
    for d in &dsts {
        match d {
            Operand::Reg(r) | Operand::Half(r, _) if r.kind == Kind::V => {
                for i in r.base..r.base + u16::from(r.len) {
                    write_v(st, old_ex, i, res);
                }
            }
            _ => write_mask_dst(st, d, carry, &mut cs),
        }
    }
    if e.implicit.writes & ImplicitSet::VCC != 0 {
        st.s[VCC] = carry;
        cs.insert(106);
    }
    havoc(inst, st, ex, &cs, &cv);
}

/// A wave32 VALU lane-mask destination: one SGPR (the tables' pair over-approximates).
fn write_mask_dst(st: &mut St, d: &Operand, val: SVal, cs: &mut BTreeSet<u16>) {
    match d {
        Operand::Reg(r) if r.kind == Kind::S => {
            st.s[usize::from(r.base)] = val;
            cs.extend(r.base..r.base + u16::from(r.len));
        }
        _ => write_s(st, d, val),
    }
}

fn write_s(st: &mut St, d: &Operand, val: SVal) {
    match d {
        Operand::Reg(r) if r.kind == Kind::S => {
            for i in r.base..r.base + u16::from(r.len) {
                if let Some(x) = st.s.get_mut(usize::from(i)) {
                    *x = val;
                }
                if st.compl == Some(i) {
                    st.compl = None;
                }
            }
        }
        Operand::Special(Special::Vcc) => {
            st.s[VCC] = val;
            st.s[VCC + 1] = val;
        }
        Operand::Special(Special::VccLo) => st.s[VCC] = val,
        Operand::Special(Special::VccHi) => st.s[VCC + 1] = val,
        Operand::Special(Special::M0) => st.s[M0] = val,
        _ => {}
    }
}

fn write_v(st: &mut St, ex: Ex, i: u16, val: VVal) {
    let Some(old) = st.v.get(usize::from(i)).copied() else {
        return;
    };
    let drf = val.drf || old.drf;
    st.v[usize::from(i)] = match ex {
        Ex::Full => val,
        Ex::Empty => old,
        // a wave either writes every lane or none
        Ex::Coh if val.k <= VK::Wave && old.k <= VK::Wave => VVal {
            k: if st.exec_uni && val.k == VK::Uni && old.k == VK::Uni {
                VK::Uni
            } else {
                VK::Wave
            },
            drf,
        },
        _ => VVal { k: VK::Var, drf },
    };
}

/// Every register the effects say the instruction defines and the rules above did not
/// write becomes per-wave (fail closed).
fn havoc(inst: &Inst, st: &mut St, ex: &mut Ex, cs: &BTreeSet<u16>, cv: &BTreeSet<u16>) {
    let e = &inst.effects;
    let valu = peacemaker_ir::isa::lookup_inst(Arch::Gfx1201, &inst)
        .ok()
        .is_some_and(|r| r.name.starts_with("v_"));
    for r in &e.defs {
        for i in r.base..r.base + u16::from(r.len) {
            match r.kind {
                Kind::S if !cs.contains(&i) => {
                    // wave32 lane masks: the pair's high half is not written
                    if valu && r.len == 2 && i == r.base + 1 && cs.contains(&r.base) {
                        continue;
                    }
                    if i == EXEC || i == EXEC + 1 {
                        *ex = Ex::Var;
                        st.exec_uni = false;
                        st.compl = None;
                    } else if let Some(x) = st.s.get_mut(usize::from(i)) {
                        *x = SVal::VAR;
                    }
                    if st.compl == Some(i) {
                        st.compl = None;
                    }
                }
                Kind::V if !cv.contains(&i) => {
                    if let Some(x) = st.v.get_mut(usize::from(i)) {
                        *x = VVal::VAR;
                    }
                }
                _ => {}
            }
        }
    }
    if e.implicit.writes & ImplicitSet::SCC != 0 && !valu && !cs.contains(&u16::MAX) {
        // SCC was set by the scalar rules (s_*): nothing to do; a VALU never writes it
    }
    if e.implicit.writes & ImplicitSet::VCC != 0 && !cs.contains(&106) {
        st.s[VCC] = SVal::VAR;
    }
    if e.implicit.writes & ImplicitSet::EXEC != 0 && !cs.contains(&126) {
        *ex = Ex::Var;
        st.exec_uni = false;
        st.compl = None;
    }
}

type Key = (usize, Ex);

struct Graph {
    nodes: Vec<Key>,
    succ: Vec<Vec<usize>>,
    /// per-wave branch nodes and why
    per_wave: BTreeMap<usize, String>,
    uniform: usize,
    drf: usize,
    /// each node's entry state (taint applied)
    entry: Vec<St>,
    /// each node's EXEC state at its block's end
    exit: Vec<Ex>,
}

impl Graph {
    /// The graph over (block, EXEC at the block's end): nodes of one block that leave in
    /// the same EXEC state merge (waves that entered differently have run the same
    /// instructions and leave alike), the edges and per-wave marks unite. It has every
    /// path of `self` and possibly more, so a barrier check on it is conservative; it lets
    /// an if/else whose sides restore EXEC reconverge where EXEC is restored.
    fn by_exit(&self) -> Graph {
        let keys: BTreeSet<Key> = (0..self.nodes.len())
            .map(|i| (self.nodes[i].0, self.exit[i]))
            .collect();
        let nodes: Vec<Key> = keys.into_iter().collect();
        let id: BTreeMap<Key, usize> = nodes.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let of = |i: usize| id[&(self.nodes[i].0, self.exit[i])];
        let mut succ: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nodes.len()];
        let mut per_wave = BTreeMap::new();
        for i in 0..self.nodes.len() {
            succ[of(i)].extend(self.succ[i].iter().map(|&j| of(j)));
            if let Some(w) = self.per_wave.get(&i) {
                per_wave.entry(of(i)).or_insert_with(|| w.clone());
            }
        }
        Graph {
            exit: nodes.iter().map(|k| k.1).collect(),
            nodes,
            succ: succ.into_iter().map(|s| s.into_iter().collect()).collect(),
            per_wave,
            uniform: self.uniform,
            drf: self.drf,
            entry: Vec::new(),
        }
    }
}

fn run_block(body: &Body, bi: usize, st: &mut St, ex: &mut Ex, block: u32) {
    let b = &body.blocks[bi];
    for (k, &id) in body.layout[b.range.0..b.range.1].iter().enumerate() {
        step(
            body.insts.get(id).expect("laid out"),
            (b.range.0 + k) as u32,
            st,
            ex,
            block,
        );
    }
}

/// The successor keys of block `bi` left in state (`st`, `ex`), and whether its
/// branch is per-wave (with the reason) or uniform-on-DRF.
fn successors(
    body: &Body,
    index: &BTreeMap<peacemaker_ir::cfg::BlockId, usize>,
    bi: usize,
    st: &St,
    ex: Ex,
) -> (Vec<Key>, Option<String>, bool) {
    let b = &body.blocks[bi];
    let next = |id| index[&id];
    match b.term {
        Terminator::EndPgm | Terminator::Unreachable => (vec![], None, false),
        Terminator::FallThrough | Terminator::Jump(_) => (
            b.succs.iter().map(|s| (next(*s), ex)).collect(),
            None,
            false,
        ),
        Terminator::Branch {
            cond,
            taken,
            fallthrough,
        } => {
            let (t, f) = (next(taken), next(fallthrough));
            match cond {
                Cond::Scc0 | Cond::Scc1 => (
                    vec![(t, ex), (f, ex)],
                    (!st.scc.uni).then(|| format!("{cond:?} on a per-wave SCC")),
                    st.scc.drf,
                ),
                Cond::Vccz | Cond::Vccnz => {
                    let v = st.s[VCC];
                    let zero_taken = cond == Cond::Vccz;
                    let succ = match v.mask {
                        Mask::Zero => vec![(if zero_taken { t } else { f }, ex)],
                        Mask::Full => vec![(if zero_taken { f } else { t }, ex)],
                        _ => vec![(t, ex), (f, ex)],
                    };
                    (
                        succ,
                        (!v.uni).then(|| format!("{cond:?} on a per-wave VCC")),
                        v.drf,
                    )
                }
                Cond::Execz | Cond::Execnz => {
                    // (successor when EXEC is empty, when it is not)
                    let (on_empty, on_some) = if cond == Cond::Execz { (t, f) } else { (f, t) };
                    match ex {
                        Ex::Full => (vec![(on_some, Ex::Full)], None, false),
                        Ex::Empty => (vec![(on_empty, Ex::Empty)], None, false),
                        Ex::Coh => (
                            vec![(on_empty, Ex::Empty), (on_some, Ex::Full)],
                            (!st.exec_uni).then(|| format!("{cond:?} on a wave-coherent EXEC")),
                            false,
                        ),
                        Ex::Var => (
                            vec![(on_empty, Ex::Empty), (on_some, Ex::Var)],
                            Some(format!("{cond:?} on a lane-varying EXEC")),
                            false,
                        ),
                    }
                }
            }
        }
    }
}

fn explore(
    body: &Body,
    index: &BTreeMap<peacemaker_ir::cfg::BlockId, usize>,
    taint: &BTreeMap<usize, Taint>,
    entry_ex: Ex,
    block: u32,
) -> Result<Graph, String> {
    let mut at: BTreeMap<Key, St> = BTreeMap::new();
    let start = (0usize, entry_ex);
    at.insert(start, St::entry());
    let mut work = vec![start];
    let mut rounds = 0usize;
    while let Some(key) = work.pop() {
        rounds += 1;
        if rounds > 200_000 {
            return Err("uniformity: the fixpoint did not converge".into());
        }
        let mut st = at[&key].clone();
        if let Some(t) = taint.get(&key.0) {
            t.apply(&mut st);
        }
        let mut ex = key.1;
        run_block(body, key.0, &mut st, &mut ex, block);
        let (succ, _, _) = successors(body, index, key.0, &st, ex);
        for s in succ {
            let st = st.refine(s.1).flow();
            let next = match at.get(&s) {
                None => st.clone(),
                Some(old) => old.join(&st),
            };
            if at.get(&s) != Some(&next) {
                at.insert(s, next);
                work.push(s);
            }
        }
    }
    let nodes: Vec<Key> = at.keys().copied().collect();
    let id: BTreeMap<Key, usize> = nodes.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let mut g = Graph {
        succ: vec![Vec::new(); nodes.len()],
        per_wave: BTreeMap::new(),
        uniform: 0,
        drf: 0,
        entry: Vec::new(),
        exit: Vec::new(),
        nodes,
    };
    for (i, key) in g.nodes.clone().into_iter().enumerate() {
        let mut st = at[&key].clone();
        if let Some(t) = taint.get(&key.0) {
            t.apply(&mut st);
        }
        g.entry.push(st.clone());
        let mut ex = key.1;
        run_block(body, key.0, &mut st, &mut ex, block);
        g.exit.push(ex);
        let (succ, why, drf) = successors(body, index, key.0, &st, ex);
        let branch = matches!(body.blocks[key.0].term, Terminator::Branch { .. });
        match why {
            Some(w) => {
                g.per_wave.insert(i, w);
            }
            None if branch => {
                g.uniform += 1;
                g.drf += usize::from(drf);
            }
            None => {}
        }
        g.succ[i] = succ.iter().map(|s| id[s]).collect();
    }
    Ok(g)
}

/// Immediate post-dominator of every node of the graph `succ` (an empty successor list
/// is an exit), `None` for the virtual exit; bitset post-dominator sets.
fn immediate_post_dominators(succ: &[Vec<usize>]) -> Vec<Option<usize>> {
    let n = succ.len();
    let words = (n + 1).div_ceil(64);
    let bit = |set: &[u64], i: usize| set[i / 64] >> (i % 64) & 1 == 1;
    let mut pdom = vec![vec![u64::MAX; words]; n + 1];
    pdom[n] = vec![0; words];
    pdom[n][n / 64] |= 1 << (n % 64);
    let succs = |i: usize| -> Vec<usize> {
        if succ[i].is_empty() {
            vec![n]
        } else {
            succ[i].clone()
        }
    };
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..n).rev() {
            let mut acc = vec![u64::MAX; words];
            for s in succs(i) {
                for (a, b) in acc.iter_mut().zip(&pdom[s]) {
                    *a &= b;
                }
            }
            acc[i / 64] |= 1 << (i % 64);
            if acc != pdom[i] {
                pdom[i] = acc;
                changed = true;
            }
        }
    }
    let count = |s: &[u64]| s.iter().map(|w| w.count_ones()).sum::<u32>();
    (0..n)
        .map(|i| {
            // the closest strict post-dominator has the largest post-dominator set
            (0..=n)
                .filter(|&c| c != i && bit(&pdom[i], c))
                .max_by_key(|&c| count(&pdom[c]))
                .filter(|&c| c != n)
        })
        .collect()
}

/// For each per-wave node: its immediate post-dominator (`None` = the exit) and the nodes
/// on the way there.
fn regions(
    g: &Graph,
    ipdom: &[Option<usize>],
) -> BTreeMap<usize, (Option<usize>, BTreeSet<usize>)> {
    let mut out = BTreeMap::new();
    for &b in g.per_wave.keys() {
        let join = ipdom[b];
        let mut inside = BTreeSet::new();
        let mut stack = g.succ[b].clone();
        while let Some(x) = stack.pop() {
            if Some(x) == join || !inside.insert(x) {
                continue;
            }
            stack.extend(g.succ[x].iter().copied());
        }
        out.insert(b, (join, inside));
    }
    out
}

fn barrier_weight(body: &Body, bi: usize) -> i64 {
    let b = &body.blocks[bi];
    body.layout[b.range.0..b.range.1]
        .iter()
        .filter(|&&id| {
            matches!(
                body.insts.get(id).expect("laid out").effects.control,
                Control::Barrier(_)
            )
        })
        .count() as i64
}

/// Registers the blocks of `nodes` define (the sync-dependence taint).
fn defined_in(body: &Body, g: &Graph, nodes: &BTreeSet<usize>) -> Taint {
    let mut t = Taint::default();
    let blocks: BTreeSet<usize> = nodes.iter().map(|&i| g.nodes[i].0).collect();
    for bi in blocks {
        let b = &body.blocks[bi];
        for &id in &body.layout[b.range.0..b.range.1] {
            let e = &body.insts.get(id).expect("laid out").effects;
            for r in &e.defs {
                let set = match r.kind {
                    Kind::S => &mut t.s,
                    Kind::V => &mut t.v,
                    Kind::Ttmp => continue,
                };
                set.extend((r.base..r.base + u16::from(r.len)).filter(|&i| usize::from(i) < NV));
            }
            t.scc |= e.implicit.writes & ImplicitSet::SCC != 0;
            if e.implicit.writes & ImplicitSet::VCC != 0 {
                t.s.insert(106);
            }
        }
    }
    t.s.retain(|&i| usize::from(i) < NS);
    t
}

/// The barrier-participation obligation of a `block`-thread (wave32) region body.
/// Returns (uniform branch nodes, per-wave branch nodes, verdict).
pub fn barrier_participation(body: &Body, block: u32) -> Result<(usize, usize, String), String> {
    let waves = block.div_ceil(32);
    let index: BTreeMap<_, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();
    let barriers = (0..body.blocks.len())
        .filter(|&b| barrier_weight(body, b) > 0)
        .count();
    // a partial last wave starts with some lanes off
    let entry_ex = if block % 32 == 0 { Ex::Full } else { Ex::Var };
    let mut taint: BTreeMap<usize, Taint> = BTreeMap::new();
    let mut graph = None;
    for _ in 0..64 {
        let explored = explore(body, &index, &taint, entry_ex, block)?;
        // waves reconverge where every path meets in the same block and EXEC state
        let g = explored.by_exit();
        let ipdom = immediate_post_dominators(&g.succ);
        let mut grew = false;
        for (join, inside) in regions(&g, &ipdom).values() {
            let Some(join) = join else { continue };
            let t = defined_in(body, &g, inside);
            grew |= taint.entry(g.nodes[*join].0).or_default().union(&t);
        }
        if !grew {
            graph = Some((explored, g, ipdom));
            break;
        }
    }
    let Some((explored, g, ipdom)) = graph else {
        return Err("uniformity: the sync-dependence taint did not converge".into());
    };
    let (uniform, divergent) = (explored.uniform, explored.per_wave.len());
    if waves <= 1 {
        return Ok((
            uniform,
            divergent,
            "single wave: every barrier involves only this wave".into(),
        ));
    }
    if barriers == 0 {
        return Ok((uniform, divergent, "no barrier".into()));
    }
    let n = g.nodes.len();
    let succs = |i: usize| -> Vec<usize> {
        if g.succ[i].is_empty() {
            vec![n]
        } else {
            g.succ[i].clone()
        }
    };
    let weight = |i: usize| barrier_weight(body, g.nodes[i].0);
    for (&b, (join, inside)) in &regions(&g, &ipdom) {
        let why = &g.per_wave[&b];
        let (bb, bx) = g.nodes[b];
        // a barrier on a cycle inside the region could run a per-wave number of times
        for &x in inside.iter().filter(|&&x| weight(x) > 0) {
            let mut seen = BTreeSet::new();
            let mut st: Vec<usize> = succs(x)
                .into_iter()
                .filter(|s| inside.contains(s))
                .collect();
            while let Some(y) = st.pop() {
                if y == x {
                    return Err(format!(
                        "barrier participation: block {} repeats a barrier on a cycle under the per-wave branch of block {bb} (EXEC {bx:?}; {why})",
                        g.nodes[x].0
                    ));
                }
                if seen.insert(y) {
                    st.extend(succs(y).into_iter().filter(|s| inside.contains(s)));
                }
            }
        }
        let (mut lo, mut hi): (BTreeMap<usize, i64>, BTreeMap<usize, i64>) = (
            inside.iter().map(|&x| (x, i64::MAX / 4)).collect(),
            inside.iter().map(|&x| (x, i64::MIN / 4)).collect(),
        );
        for _ in 0..=inside.len() {
            for &x in inside {
                let (mut l, mut h) = (i64::MAX / 4, i64::MIN / 4);
                for s in succs(x) {
                    let (sl, sh) = if inside.contains(&s) {
                        (lo[&s], hi[&s])
                    } else {
                        (0, 0)
                    };
                    l = l.min(sl);
                    h = h.max(sh);
                }
                lo.insert(x, weight(x) + l);
                hi.insert(x, weight(x) + h);
            }
        }
        let counts: BTreeSet<(i64, i64)> = succs(b)
            .into_iter()
            .map(|s| {
                if inside.contains(&s) {
                    (lo[&s], hi[&s])
                } else {
                    (0, 0)
                }
            })
            .collect();
        if counts.len() != 1 || counts.iter().any(|(l, h)| l != h) {
            return Err(format!(
                "barrier participation: the paths from the per-wave branch of block {bb} (EXEC {bx:?}; {why}) to its post-dominator {} pass different numbers of barriers {counts:?}; a {waves}-wave region would pair a barrier with another wave's dispatcher barrier",
                match join {
                    None => "(the exit)".to_string(),
                    Some(j) => format!("block {}", g.nodes[*j].0),
                }
            ));
        }
    }
    Ok((
        uniform,
        divergent,
        format!(
            "{waves} waves, {barriers} barrier blocks, {} (block, EXEC) states: every per-wave branch passes the same barrier count on all its paths to its post-dominator, with no barrier on a per-wave cycle; {} uniform branches rest on uniform-address loads (data-race-free source)",
            g.nodes.len(),
            g.drf
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::cfg::Arena;
    use peacemaker_ir::passes::cfg::build_blocks;

    /// Assemble `lines` (a `NAME:` line is a label; a branch may name one) into a body.
    fn body(lines: &[&str]) -> Body {
        let encode = |l: &str| {
            let inst =
                crate::text::parse_line(l, Arch::Gfx1201).unwrap_or_else(|e| panic!("{l}: {e}"));
            peacemaker_ir::codec::gfx12::encode(&inst).unwrap_or_else(|e| panic!("{l}: {e}"))
        };
        let is_branch = |l: &str| l.starts_with("s_branch") || l.starts_with("s_cbranch");
        // pass 1: word offsets (branches are one word)
        let (mut at, mut labels, mut placed) = (0usize, BTreeMap::new(), Vec::new());
        for l in lines {
            if let Some(name) = l.strip_suffix(':') {
                labels.insert(name.to_string(), at);
                continue;
            }
            let n = if is_branch(l) { 1 } else { encode(l).len() };
            placed.push((at, *l));
            at += n;
        }
        let mut words = Vec::new();
        for (at, l) in placed {
            let text = match l.split_once(' ') {
                Some((m, target)) if is_branch(l) && labels.contains_key(target) => {
                    format!("{m} {}", labels[target] as i64 - at as i64 - 1)
                }
                _ => l.to_string(),
            };
            words.extend(encode(&text));
        }
        let (mut insts, mut layout, mut i) = (Arena::new(), Vec::new(), 0);
        while i < words.len() {
            let (inst, n) = peacemaker_ir::codec::gfx12::decode(&words[i..]).unwrap();
            layout.push(insts.insert(inst));
            i += n;
        }
        let mut body = Body {
            insts,
            blocks: Vec::new(),
            layout,
        };
        build_blocks(&mut body, Arch::Gfx1201).unwrap();
        body
    }

    fn loop_on(bound: &[&'static str]) -> Vec<&'static str> {
        let mut l = bound.to_vec();
        l.extend([
            "L:",
            "s_barrier_signal -1",
            "s_barrier_wait -1",
            "s_add_co_i32 s2, s2, -1",
            "s_cmp_lg_u32 s2, 0",
            "s_cbranch_scc1 L",
            "s_endpgm",
        ]);
        l
    }

    /// `PM_UNI_OBJ=path PM_UNI_SYM=symbol PM_UNI_B=threads`: print the product graph's
    /// per-wave nodes and their regions for a real object (diagnostics, not a gate).
    #[test]
    #[ignore]
    fn dump_regions_of_an_object() {
        let (Ok(path), Ok(sym), Ok(b)) = (
            std::env::var("PM_UNI_OBJ"),
            std::env::var("PM_UNI_SYM"),
            std::env::var("PM_UNI_B"),
        ) else {
            return;
        };
        let block: u32 = b.parse().unwrap();
        let bytes = std::fs::read(path).unwrap();
        let lifted = crate::lift_object(
            &bytes,
            crate::Options {
                frontend: peacemaker_ir::inst::Frontend::Hipcc,
            },
        )
        .unwrap();
        let k = lifted
            .program
            .kernels
            .iter()
            .find(|k| k.symbol.0 == sym)
            .unwrap();
        let body = &k.body;
        let index: BTreeMap<_, usize> = body
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();
        let explored = explore(body, &index, &BTreeMap::new(), Ex::Full, block).unwrap();
        let m = explored.by_exit();
        for (b, (ipdom, inside)) in regions(&m, &immediate_post_dominators(&m.succ)) {
            let names: Vec<String> = inside
                .iter()
                .map(|&i| format!("{:?}", m.nodes[i]))
                .collect();
            println!(
                "merged per-wave {:?} ({}) ipdom {:?} inside {}",
                m.nodes[b],
                m.per_wave[&b],
                ipdom.map(|j| m.nodes[j]),
                names.join(" ")
            );
        }
        let g = explored;
        for (b, (ipdom, inside)) in regions(&g, &immediate_post_dominators(&g.succ)) {
            let names: Vec<String> = inside
                .iter()
                .map(|&i| format!("{:?}", g.nodes[i]))
                .collect();
            println!(
                "per-wave {:?} ({}) ipdom {:?} inside {}",
                g.nodes[b],
                g.per_wave[&b],
                ipdom,
                names.join(" ")
            );
        }
        let trace: Option<usize> = std::env::var("PM_UNI_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok());
        for (i, key) in g.nodes.iter().enumerate() {
            println!(
                "{key:?} -> {:?}",
                g.succ[i].iter().map(|&j| g.nodes[j]).collect::<Vec<_>>()
            );
            if Some(key.0) != trace {
                continue;
            }
            let (mut st, mut ex) = (g.entry[i].clone(), key.1);
            let b = &body.blocks[key.0];
            for &id in &body.layout[b.range.0..b.range.1] {
                let inst = body.insts.get(id).unwrap();
                step(inst, 0, &mut st, &mut ex, block);
                let uni: Vec<String> = inst
                    .operands
                    .iter()
                    .filter_map(|o| match o {
                        Operand::Reg(r) if r.kind == Kind::S => {
                            Some(format!("s{}:{}", r.base, st.sreg(*r).uni))
                        }
                        Operand::Reg(r) if r.kind == Kind::V => {
                            Some(format!("v{}:{:?}", r.base, st.vreg(*r).k))
                        }
                        _ => None,
                    })
                    .collect();
                println!(
                    "   {:<28} scc {} exec {ex:?} {}",
                    crate::rewrite::mnemonic(inst),
                    st.scc.uni,
                    uni.join(" ")
                );
            }
        }
    }

    #[test]
    fn a_barrier_loop_bounded_by_a_kernarg_is_uniform() {
        let b = body(&loop_on(&[
            "s_load_b32 s2, s[0:1], 0x0",
            "s_wait_kmcnt 0x0",
        ]));
        let (uniform, divergent, verdict) = barrier_participation(&b, 64).unwrap();
        assert_eq!((uniform, divergent), (1, 0), "{verdict}");
        assert!(verdict.contains("0 uniform branches rest on"), "{verdict}");
    }

    #[test]
    fn a_barrier_loop_bounded_by_the_thread_id_is_refused() {
        let b = body(&loop_on(&["v_readfirstlane_b32 s2, v0"]));
        let e = barrier_participation(&b, 64).unwrap_err();
        assert!(e.contains("on a cycle"), "{e}");
    }

    #[test]
    fn readfirstlane_of_a_uniform_vgpr_is_uniform() {
        let b = body(&loop_on(&[
            "s_load_b32 s3, s[0:1], 0x0",
            "s_wait_kmcnt 0x0",
            "v_mov_b32_e32 v1, s3",
            "v_readfirstlane_b32 s2, v1",
        ]));
        barrier_participation(&b, 64).unwrap();
    }

    #[test]
    fn readfirstlane_of_a_wave_index_is_per_wave() {
        let b = body(&loop_on(&[
            "v_lshrrev_b32_e32 v1, 5, v0",
            "v_readfirstlane_b32 s2, v1",
        ]));
        assert!(barrier_participation(&b, 64).is_err());
    }

    #[test]
    fn a_load_through_a_loaded_pointer_is_uniform_under_data_race_freedom() {
        let b = body(&loop_on(&[
            "s_load_b64 s[4:5], s[0:1], 0x0",
            "s_wait_kmcnt 0x0",
            "s_load_b32 s2, s[4:5], 0x0",
            "s_wait_kmcnt 0x0",
        ]));
        let (_, _, verdict) = barrier_participation(&b, 64).unwrap();
        assert!(verdict.contains("1 uniform branches rest on"), "{verdict}");
    }

    #[test]
    fn a_wave32_lane_mask_writes_one_sgpr() {
        // the effects tables give v_cmp_*_e64 an SGPR pair: s3 (the loop bound) survives
        let b = body(&loop_on(&[
            "s_load_b32 s2, s[0:1], 0x0",
            "s_wait_kmcnt 0x0",
            "v_cmp_gt_u32_e64 s1, 0x80, v0",
        ]));
        barrier_participation(&b, 64).unwrap();
    }

    fn split(threshold: &str, else_barrier: bool) -> Body {
        let cmp = format!("v_cmp_gt_u32_e64 s2, {threshold}, v0");
        let mut l = vec![
            cmp.as_str(),
            "s_and_saveexec_b32 s6, s2",
            "s_cbranch_execz ELSE",
            "s_barrier_signal -1",
            "s_barrier_wait -1",
            "ELSE:",
            "s_and_not1_saveexec_b32 s6, s6",
            "s_cbranch_execz END",
            "v_mov_b32_e32 v1, 0",
        ];
        if else_barrier {
            l.extend(["s_barrier_signal -1", "s_barrier_wait -1"]);
        }
        l.extend(["END:", "s_or_b32 exec_lo, exec_lo, s6", "s_endpgm"]);
        body(&l)
    }

    #[test]
    fn a_wave_coherent_if_else_with_a_barrier_on_each_side_is_accepted() {
        // tid < 32: wave 0 takes the then side, wave 1 the else side, one barrier each
        let (_, divergent, verdict) = barrier_participation(&split("32", true), 64).unwrap();
        assert!(divergent >= 1, "{verdict}");
    }

    #[test]
    fn a_lane_varying_if_else_with_barriers_is_refused() {
        // tid < 16 splits wave 0's lanes: it runs both sides
        let e = barrier_participation(&split("16", true), 64).unwrap_err();
        assert!(e.contains("different numbers of barriers"), "{e}");
    }

    #[test]
    fn a_wave_coherent_split_with_a_barrier_on_one_side_only_is_refused() {
        let e = barrier_participation(&split("32", false), 64).unwrap_err();
        assert!(e.contains("different numbers of barriers"), "{e}");
    }
}
