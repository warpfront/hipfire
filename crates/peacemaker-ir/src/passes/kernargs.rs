// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A1/A3/A4 (railgun design §1.5): kernarg pointer provenance, per-argument
//! access summaries, read-cache classes and scalar kernarg roles for one
//! lifted kernel, as [`KernargFacts`].
//!
//! The pass is a forward may-dataflow over the CFG. Every 32-bit location
//! (SGPR, TTMP, VCC, EXEC, M0, SCC, VGPR summarised over lanes, and VGPR lanes
//! written by `v_writelane_b32` at a constant lane) carries a [`Val`]:
//!
//! * `roots`/`src` — pointer provenance: the kernarg dwords the value may be
//!   a (derived) copy of, plus non-argument sources (kernarg segment base,
//!   PC-relative, AQL packet, apertures, scratch V#, and `OPAQUE` for anything
//!   loaded from memory).
//! * `taint`/`tf` — A4 influence: the kernarg dwords that may flow into the
//!   value through any arithmetic, and whether a work-item/work-group id or
//!   loaded data did.
//!
//! Transfer rules. Moves, `s_cselect_b64`/`v_cndmask` (union of both
//! candidates), lane save/restore and 32/64-bit adds, logic ops and unknown
//! instructions propagate provenance (union of sources, dword by dword where
//! widths match). Multiplies, shifts, bit-field extracts, conversions, float
//! arithmetic and compares produce integers: their provenance is dropped
//! (their taint is kept). Shift-add/multiply-add forms keep only the addend's
//! provenance. Carry outputs carry no provenance. VGPR writes are strong only
//! while EXEC is known to be a superset of the entry EXEC (tracked through
//! `s_*_saveexec`, `s_mov`/`s_or` restores); under a partial EXEC they join
//! with the old value, so divergent writes of a pointer register keep every
//! candidate. SALU writes and `v_writelane` are strong.
//!
//! Attribution. A global/flat/buffer/scalar access resolves through the high
//! dword of its 64-bit base (SADDR, VADDR pair, V# dwords 0–1, SBASE): its
//! argument roots are the candidates; argument roots of the low dword are
//! added for pointer-kind arguments. A base whose high dword has no
//! provenance, may be memory-derived, or mixes argument and non-argument
//! sources makes the kernel `Unknown`; so does a store to the kernarg
//! segment, a module global or the AQL packet, a call, and any memory
//! instruction family the pass does not model. Assumption (named in the
//! receipt): a pointer is never rebuilt by multiplication, shifting or
//! bit-field extraction of a value loaded from memory.
//!
//! Guarded analysis ([`analyze_guarded`], railgun design A amendment
//! 2026-10-25). The same pass with launch/argument [`Guard`]s: kernarg dwords
//! loaded from the segment become the guarded constants (a null-pointer guard
//! is a constant with no provenance, never an allocation at address 0) and a
//! scalar compare of constants fixes its SCC, so a conditional SCC branch keeps
//! only its live edge: the CFG the guards leave. Alongside `roots/src/taint`
//! every location carries an exact integer/address witness
//! (`kernarg_ix`): loaded `i32`s from a directly rooted argument are index
//! leaves, and moves, extensions, constant shifts, add/sub/mul and matched
//! 64-bit lo/hi/carry chains keep exact machine-width expressions; a join of
//! different expressions, an EXEC-partial rewrite, or any unmodelled
//! operation is "no witness". Under a guard an address that depends on loaded
//! data (a memory-derived base, or any `MEM`-tainted address: the legacy
//! lattice drops `OPAQUE` through shifts, multiplies and sign extensions)
//! resolves only through a complete witness `direct pointer argument + integer
//! displacement` with at least one qualifying index load, and is reported as
//! an [`IndexedArgEffect`] while still credited allocation-wide to its
//! base argument. Every other unresolved base keeps its refusal, and the
//! unguarded [`analyze`] is unchanged. A `Conditional` result is void
//! without bind witnesses for each guard and indexed effect.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use petgraph::graph::{DiGraph, NodeIndex};
use smallvec::SmallVec;

use crate::cfg::{Body, InstId};
use crate::effects::ImplicitSet;
use crate::inst::{
    Abi, Arch, EntryCell, EntryValue, Form, FormFields, Inst, Kernel, RawAbi, VmemForm, Wave,
};
use crate::isa::OpRow;
use crate::kernarg::{
    AccessClass, AccessMode, ArgAccess, ArgFacts, ArgRole, CacheBits, DwordFacts, EntryReg, Guard,
    IdxExpr, ImplicitUse, IndexElem, IndexLoad, IndexedArgEffect, KernargFacts, KernelStatus,
    MemPath, ReadCacheClass, RegionBound, SealedAccess, SealedBound, SealedEntryLoad, SealedIndex,
    SealedTableUse, SideReads, Sinks, Unresolved, HIDDEN_GRID_KINDS,
};
use crate::metadata::Kernarg;
use crate::operand::{ImmField, InlineConst, Operand, Special};
use crate::reg::Kind;

use super::kernarg_ix::{
    certify, certify_source, load_ext, table_entry, Interner, Ix, KargKind, Node, CARRY_SCALAR,
};

/// Kernarg dwords tracked individually (512 bytes). Kernels with a larger
/// segment fall back to the overflow flags: an overflowing pointer root is
/// `Unknown`, an overflowing taint is reported as `taint_overflow`.
const MAX_DW: u32 = 128;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
struct DwSet(u128);
impl DwSet {
    const ALL: DwSet = DwSet(u128::MAX);
    fn one(dw: u32) -> Self {
        DwSet(1u128 << dw)
    }
    fn is_empty(self) -> bool {
        self.0 == 0
    }
    fn union(self, other: Self) -> Self {
        DwSet(self.0 | other.0)
    }
    fn minus(self, other: Self) -> Self {
        DwSet(self.0 & !other.0)
    }
    fn contains(self, dw: u32) -> bool {
        dw < MAX_DW && self.0 >> dw & 1 != 0
    }
    fn iter(self) -> impl Iterator<Item = u32> {
        (0..MAX_DW).filter(move |&dw| self.0 >> dw & 1 != 0)
    }
}

// `Val::src`: non-argument provenance.
const KSEG: u8 = 1;
const APERTURE: u8 = 2;
const PCREL: u8 = 4;
const PACKET: u8 = 8;
const OPAQUE: u8 = 16;
const SCRATCH: u8 = 32;
const ROOT_OVF: u8 = 64;
// `Val::tf`: A4 influence flags.
const WI: u8 = 1;
const WG: u8 = 2;
const MEM: u8 = 4;
const TAINT_OVF: u8 = 8;
/// Influenced by a value loaded from a sealed-table entry (sealed analysis).
const TBL: u8 = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Val {
    roots: DwSet,
    src: u8,
    taint: DwSet,
    tf: u8,
    /// Sealed analysis: bit `k` set when the value may be a derived copy of
    /// the 64-bit entry loaded by qualified entry site `k` (provenance, like
    /// `roots`: dropped by `kill`).
    tbl: u8,
    konst: Option<u32>,
    /// Byte offset from the kernarg segment base, for a `KSEG`-rooted value.
    koff: Option<i64>,
    /// Exact integer/address witness (`kernarg_ix`), independent of the
    /// provenance lattice above: a loaded `i32` is still `OPAQUE` there.
    ix: Ix,
}

impl Val {
    fn konst(value: u32) -> Self {
        Val {
            konst: Some(value),
            ..Val::default()
        }
    }
    fn src(src: u8) -> Self {
        Val {
            src,
            ..Val::default()
        }
    }
    fn flags(tf: u8) -> Self {
        Val {
            tf,
            ..Val::default()
        }
    }
    fn karg(dw: u32) -> Self {
        if dw < MAX_DW {
            Val {
                roots: DwSet::one(dw),
                taint: DwSet::one(dw),
                ..Val::default()
            }
        } else {
            Val {
                src: ROOT_OVF,
                tf: TAINT_OVF,
                ..Val::default()
            }
        }
    }
    fn loaded() -> Self {
        Val {
            src: OPAQUE,
            tf: MEM,
            ..Val::default()
        }
    }
    /// The value no lane holds: identity of [`Val::join`] for the witness.
    fn bottom() -> Self {
        Val {
            ix: Ix::BOT,
            ..Val::default()
        }
    }
    /// A dword read from the kernarg segment at a dynamic offset: any argument.
    fn any_karg() -> Self {
        Val {
            src: OPAQUE,
            taint: DwSet::ALL,
            tf: TAINT_OVF,
            ..Val::default()
        }
    }
    fn join(self, other: Self) -> Self {
        Val {
            roots: self.roots.union(other.roots),
            src: self.src | other.src,
            taint: self.taint.union(other.taint),
            tf: self.tf | other.tf,
            konst: if self.konst == other.konst {
                self.konst
            } else {
                None
            },
            koff: if self.koff == other.koff {
                self.koff
            } else {
                None
            },
            ix: self.ix.join(other.ix),
            tbl: self.tbl | other.tbl,
        }
    }
    /// Arithmetic combination keeping provenance (add, logic, unknown ops).
    fn mix<'v>(vals: impl IntoIterator<Item = &'v Val>) -> Self {
        let mut out = Val::default();
        for v in vals {
            out.roots = out.roots.union(v.roots);
            out.src |= v.src;
            out.taint = out.taint.union(v.taint);
            out.tf |= v.tf;
            out.tbl |= v.tbl;
        }
        out
    }
    /// Arithmetic producing an integer: provenance dropped, influence kept.
    fn kill<'v>(vals: impl IntoIterator<Item = &'v Val>) -> Self {
        let mut out = Val::mix(vals);
        out.roots = DwSet::default();
        out.src = 0;
        out.tbl = 0;
        out
    }
}

const NS: usize = 108; // s0..s105 plus VCC_LO/VCC_HI at 106/107
const VCC_LO: usize = 106;
const VCC_HI: usize = 107;
const NV: usize = 256;

/// What is known about an SGPR (or EXEC) holding a lane mask, relative to the
/// stack of saved EXEC masks `S_0 ⊇ …` (`S_0` = entry EXEC; `S_k` = EXEC when
/// level `k` was opened). Claims hold per execution: joins keep a claim only
/// when both sides make it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mask {
    Unknown,
    /// All lanes on (`-1`): a superset of `S_0`.
    Full,
    /// The value is a subset of `S_level`; `exact` = it equals `S_level`.
    Within {
        level: u8,
        exact: bool,
    },
}
impl Mask {
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Mask::Full, x) | (x, Mask::Full) => x,
            (Mask::Within { level: a, .. }, Mask::Within { level: b, .. }) => Mask::Within {
                level: a.max(b),
                exact: false,
            },
            (Mask::Within { level, .. }, Mask::Unknown)
            | (Mask::Unknown, Mask::Within { level, .. }) => Mask::Within {
                level,
                exact: false,
            },
            _ => Mask::Unknown,
        }
    }
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Mask::Full, _) | (_, Mask::Full) => Mask::Full,
            (
                Mask::Within {
                    level: a,
                    exact: ea,
                },
                Mask::Within {
                    level: b,
                    exact: eb,
                },
            ) => Mask::Within {
                level: a.min(b),
                exact: (ea && a <= b) || (eb && b <= a),
            },
            _ => Mask::Unknown,
        }
    }
    fn xor(self, other: Self) -> Self {
        match (self, other) {
            (Mask::Within { level: a, .. }, Mask::Within { level: b, .. }) => Mask::Within {
                level: a.min(b),
                exact: false,
            },
            _ => Mask::Unknown,
        }
    }
    /// `self & !other`.
    fn and_not(self, _other: Self) -> Self {
        match self {
            Mask::Within { level, .. } => Mask::Within {
                level,
                exact: false,
            },
            _ => Mask::Unknown,
        }
    }
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (a, b) if a == b => a,
            (Mask::Within { level: a, .. }, Mask::Within { level: b, .. }) => Mask::Within {
                level: a.min(b),
                exact: false,
            },
            _ => Mask::Unknown,
        }
    }
    /// Levels above `depth - 1` were closed: a claim about them weakens to
    /// the deepest open level (`S_j ⊆ S_{depth-1}` for `j ≥ depth`).
    fn clamp(self, depth: usize) -> Self {
        match self {
            Mask::Within { level, .. } if usize::from(level) >= depth => Mask::Within {
                level: (depth - 1) as u8,
                exact: false,
            },
            other => other,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct State {
    s: Vec<Val>,
    t: Vec<Val>,
    v: Vec<Val>,
    /// VGPR lanes written by `v_writelane_b32` at a constant lane.
    lanes: BTreeMap<(u16, u8), Val>,
    /// Per EXEC level `k`, per VGPR: every value a lane of `S_k` that was
    /// disabled at level `k` (and not yet re-enabled) may hold. VGPR writes are
    /// strong for the enabled lanes; these are joined back when EXEC may
    /// re-enable the lanes. Level 0 always exists.
    levels: Vec<Vec<Val>>,
    /// EXEC equals `S_k`.
    exec_exact: Option<u8>,
    /// Mask facts per SGPR (VCC at 106/107).
    masks: Vec<Mask>,
    exec: [Val; 2],
    m0: Val,
    scc: Val,
    /// Sealed analysis: EXEC is known nonzero (a decided EXECZ/EXECNZ edge
    /// set it; any EXEC write but a superset `s_or_b32 exec_lo, exec_lo, x`
    /// clears it; joins keep it only when both sides know it).
    exec_nz: bool,
}

impl State {
    fn empty() -> Self {
        State {
            s: vec![Val::default(); NS],
            t: vec![Val::default(); 16],
            v: vec![Val::default(); NV],
            lanes: BTreeMap::new(),
            levels: vec![vec![Val::bottom(); NV]],
            exec_exact: Some(0),
            masks: vec![Mask::Unknown; NS],
            exec: [Val::default(); 2],
            m0: Val::default(),
            scc: Val::default(),
            exec_nz: false,
        }
    }
    fn top(&self) -> u8 {
        (self.levels.len() - 1) as u8
    }
    fn read_v(&self, r: u16) -> Val {
        let mut out = self.v[usize::from(r)];
        for (_, lane) in self.lanes.range((r, 0)..=(r, u8::MAX)) {
            out = out.join(*lane);
        }
        out
    }
    /// Values held by currently disabled lanes of VGPR `r` (cross-lane reads).
    fn disabled_v(&self, r: u16) -> Val {
        self.levels
            .iter()
            .fold(Val::bottom(), |acc, level| acc.join(level[usize::from(r)]))
    }
    /// Close every level at or above `depth` into level `depth - 1`.
    fn collapse(&mut self, depth: usize) {
        while self.levels.len() > depth {
            let closed = self.levels.pop().expect("level");
            let below = self.levels.last_mut().expect("level 0");
            for (a, b) in below.iter_mut().zip(closed) {
                *a = a.join(b);
            }
        }
        if self.exec_exact.is_some_and(|k| usize::from(k) >= depth) {
            self.exec_exact = None;
        }
        for m in &mut self.masks {
            *m = m.clamp(depth);
        }
    }
    fn join_from(&mut self, o: &State) {
        for (a, b) in self.s.iter_mut().zip(&o.s) {
            *a = a.join(*b);
        }
        for (a, b) in self.t.iter_mut().zip(&o.t) {
            *a = a.join(*b);
        }
        let mut lanes = BTreeMap::new();
        for (&(r, l), &val) in &self.lanes {
            let other = o.lanes.get(&(r, l)).copied().unwrap_or(o.v[usize::from(r)]);
            lanes.insert((r, l), val.join(other));
        }
        for (&(r, l), &val) in &o.lanes {
            lanes
                .entry((r, l))
                .or_insert_with(|| val.join(self.v[usize::from(r)]));
        }
        self.lanes = lanes;
        for (a, b) in self.v.iter_mut().zip(&o.v) {
            *a = a.join(*b);
        }
        let depth = self.levels.len().min(o.levels.len());
        self.collapse(depth);
        let mut other = o.clone();
        other.collapse(depth);
        for (mine, theirs) in self.levels.iter_mut().zip(&other.levels) {
            for (a, b) in mine.iter_mut().zip(theirs) {
                *a = a.join(*b);
            }
        }
        if self.exec_exact != other.exec_exact {
            self.exec_exact = None;
        }
        for (a, b) in self.masks.iter_mut().zip(&other.masks) {
            *a = a.join(*b);
        }
        self.exec = [self.exec[0].join(o.exec[0]), self.exec[1].join(o.exec[1])];
        self.m0 = self.m0.join(o.m0);
        self.scc = self.scc.join(o.scc);
        self.exec_nz &= o.exec_nz;
    }
    /// EXEC ⊆ its old value: lanes being disabled keep their current values.
    fn exec_narrow(&mut self) {
        let top = self.levels.len() - 1;
        for r in 0..NV {
            let cur = self.read_v(r as u16);
            self.levels[top][r] = self.levels[top][r].join(cur);
        }
        self.exec_exact = None;
    }
    /// EXEC takes a value described by `mask`: lanes disabled at the levels it
    /// may re-enable come back with every value they may hold.
    fn exec_set(&mut self, mask: Mask) {
        match mask {
            Mask::Full => {
                for level in &mut self.levels {
                    for (v, dead) in self.v.iter_mut().zip(level.iter_mut()) {
                        *v = v.join(*dead);
                        *dead = Val::bottom();
                    }
                }
                self.exec_exact = Some(0);
            }
            Mask::Within { level, exact } => {
                let b = usize::from(level).min(self.levels.len() - 1);
                let mut joined = vec![Val::bottom(); NV];
                for l in &self.levels[b..] {
                    for (j, x) in joined.iter_mut().zip(l) {
                        *j = j.join(*x);
                    }
                }
                for (v, j) in self.v.iter_mut().zip(&joined) {
                    *v = v.join(*j);
                }
                self.levels.truncate(b + 1);
                if exact {
                    self.levels[b] = vec![Val::bottom(); NV];
                    self.exec_exact = Some(b as u8);
                } else {
                    for r in 0..NV {
                        joined[r] = joined[r].join(self.read_v(r as u16));
                    }
                    self.levels[b] = joined;
                    self.exec_exact = None;
                }
                for m in &mut self.masks {
                    *m = m.clamp(b + 1);
                }
            }
            Mask::Unknown => self.exec_set(Mask::Within {
                level: 0,
                exact: false,
            }),
        }
    }
    /// A copy of the current EXEC: the level it is exactly equal to, opening
    /// a new level when EXEC is not a known saved mask.
    fn exec_copy_level(&mut self) -> u8 {
        let top = self.top();
        if self.exec_exact == Some(top) {
            return top;
        }
        self.levels.push(vec![Val::bottom(); NV]);
        let new = self.top();
        self.exec_exact = Some(new);
        new
    }
}

/// One decoded operand in its grammar field.
struct Field<'a> {
    name: &'static str,
    bits: u16,
    op: &'a Operand,
    def: bool,
    used: bool,
}
/// One opcode half (two for VOPD).
struct Part<'a> {
    name: &'static str,
    fields: SmallVec<[Field<'a>; 6]>,
}
/// A VALU lane-mask source (carry-in, select mask). The grammar types it
/// 64-bit; in wave32 it is one SGPR, and it never carries a value.
fn is_mask(part: &str, field: &str) -> bool {
    field == "SRC2"
        && (part.starts_with("v_add_co_ci")
            || part.starts_with("v_sub_co_ci")
            || part.starts_with("v_subrev_co_ci")
            || part.starts_with("v_cndmask"))
}
impl<'a> Part<'a> {
    /// Value sources: used, not also defined, not a lane mask.
    fn sources(&self) -> impl Iterator<Item = &Field<'a>> {
        self.fields
            .iter()
            .filter(move |f| f.used && !f.def && !is_mask(self.name, f.name))
    }
    fn field(&self, name: &str) -> Option<&Field<'a>> {
        self.fields.iter().find(|f| f.name == name)
    }
    fn first_use(&self) -> Option<&Field<'a>> {
        self.sources().next()
    }
}
struct Dec<'a> {
    id: InstId,
    /// Position in the kernel's instruction layout.
    pos: usize,
    inst: &'a Inst,
    parts: SmallVec<[Part<'a>; 2]>,
    extras: SmallVec<[&'a Operand; 4]>,
}

fn map_fields<'a>(
    row: &'static OpRow,
    form: Form,
    returns: bool,
    ops: &'a [Operand],
    extras: &mut SmallVec<[&'a Operand; 4]>,
) -> SmallVec<[Field<'a>; 6]> {
    let mut names = row.slots(returns);
    if form == Form::Vop3 && row.name.starts_with("v_cmpx_") {
        names.next();
    }
    let prefix = usize::from(
        form == Form::Vopc
            && row.name.starts_with("v_cmp_")
            && matches!(
                ops.first(),
                Some(Operand::Special(Special::Vcc | Special::VccLo))
            ),
    );
    let defs: SmallVec<[&str; 4]> = row.defs.split(',').collect();
    let uses: SmallVec<[&str; 6]> = row.uses.split(',').collect();
    let mut out = SmallVec::new();
    for op in ops.iter().skip(prefix) {
        match names.next() {
            Some((name, bits)) if !name.is_empty() => {
                out.push(Field {
                    name,
                    bits,
                    op,
                    def: defs.contains(&name),
                    used: uses.contains(&name),
                });
            }
            _ => extras.push(op),
        }
    }
    out
}

fn decode<'a>(arch: Arch, id: InstId, inst: &'a Inst) -> Option<Dec<'a>> {
    let mut extras = SmallVec::new();
    let mut parts = SmallVec::new();
    let returns = crate::isa::atomic_returns(arch, &inst.mods.cpol);
    if let FormFields::Vopd { y_op, x_operands } = inst.fields {
        let split = usize::from(x_operands);
        let x = crate::isa::lookup(arch, inst.op, Form::Vopd)?;
        let y = crate::isa::lookup(arch, y_op, Form::Vopd)?;
        parts.push(Part {
            name: x.name,
            fields: map_fields(x, Form::Vopd, returns, &inst.operands[..split], &mut extras),
        });
        parts.push(Part {
            name: y.name,
            fields: map_fields(y, Form::Vopd, returns, &inst.operands[split..], &mut extras),
        });
    } else {
        let row = crate::isa::lookup(arch, inst.op, inst.form)?;
        parts.push(Part {
            name: row.name,
            fields: map_fields(row, inst.form, returns, &inst.operands, &mut extras),
        });
    }
    Some(Dec {
        id,
        pos: 0,
        inst,
        parts,
        extras,
    })
}

fn inline_value(op: &Operand) -> Option<u32> {
    match op {
        Operand::Inline(InlineConst::Integer(n)) => Some(*n as i32 as u32),
        Operand::Inline(InlineConst::FloatBits(bits)) => Some(*bits),
        Operand::Literal(n) => Some(*n),
        Operand::Imm(ImmField::Sopk(n) | ImmField::Sopp(n)) => Some(i32::from(*n) as u32),
        Operand::Imm(ImmField::Unsigned(n)) => Some(*n),
        _ => None,
    }
}

/// Names whose result is an integer that is never a pointer.
fn is_kill(name: &str) -> bool {
    const PREFIX: &[&str] = &[
        "s_mul",
        "v_mul",
        "s_lshl_b",
        "s_lshr",
        "s_ashr",
        "v_lshlrev",
        "v_lshrrev",
        "v_ashrrev",
        "v_dual_lshlrev",
        "s_bfe",
        "v_bfe",
        "s_bcnt",
        "s_ff",
        "s_flbit",
        "s_cls",
        "v_ffbh",
        "v_ffbl",
        "v_cls",
        "v_clz",
        "v_ctz",
        "v_cvt",
        "s_cvt",
        "v_rcp",
        "v_rsq",
        "v_sqrt",
        "v_exp",
        "v_log",
        "v_sin",
        "v_cos",
        "v_frac",
        "v_ldexp",
        "v_div",
        "v_mbcnt",
        "v_bcnt",
        "s_bitcmp",
        "v_cmp",
        "s_cmp",
        "v_dot",
        "v_wmma",
        "v_swmmac",
        "v_sad",
        "v_msad",
        "v_qsad",
        "v_mqsad",
        "s_sext",
        "v_bfrev",
        "s_brev",
        "s_abs",
        "s_absdiff",
        "v_trig",
        "v_frexp",
        "v_class",
        "s_bitset",
        "s_quadmask",
        "s_wqm",
        "s_bitreplicate",
    ];
    const FLOAT: &[&str] = &["_f32", "_f16", "_f64", "_bf16", "_fp8", "_bf8"];
    PREFIX.iter().any(|p| name.starts_with(p)) || FLOAT.iter().any(|f| name.contains(f))
}

fn is_mov(name: &str) -> bool {
    matches!(
        name,
        "s_mov_b32"
            | "s_mov_b64"
            | "s_movk_i32"
            | "v_mov_b32_e32"
            | "v_mov_b32_e64"
            | "v_mov_b32_dpp"
            | "v_dual_mov_b32"
            | "v_readfirstlane_b32"
            | "v_mov_b64_e32"
            | "v_mov_b64_e64"
            | "s_mov_fed_b32"
            | "v_mov_b16_e32"
            | "v_mov_b16_e64"
    )
}
fn is_select(name: &str) -> bool {
    name.starts_with("s_cselect_")
        || name.starts_with("v_cndmask_b32")
        || name == "v_dual_cndmask_b32"
}
fn is_add(name: &str) -> bool {
    matches!(
        name,
        "s_add_u32"
            | "s_add_co_u32"
            | "s_add_i32"
            | "s_add_co_i32"
            | "v_add_co_u32"
            | "v_add_nc_u32_e32"
            | "v_add_nc_u32_e64"
            | "v_add_u32_e32"
            | "v_add_u32_e64"
            | "v_dual_add_nc_u32"
            | "s_add_nc_u64"
    )
}
/// Shift-add / multiply-add: provenance of the addend field only.
fn keep_field(name: &str) -> Option<&'static str> {
    if name.starts_with("s_lshl") && name.ends_with("_add_u32") {
        return Some("SSRC1");
    }
    match name {
        "v_lshl_add_u32" | "v_lshl_or_b32" | "v_mad_u32_u24" | "v_mad_i32_i24"
        | "v_mad_u32_u16" | "v_mad_i32_i16" | "v_mad_u16" | "v_mad_i16" | "v_mad_nc_u64_u32"
        | "v_mad_nc_i64_i32" => Some("SRC2"),
        _ => None,
    }
}
fn is_mad64(name: &str) -> bool {
    matches!(
        name,
        "v_mad_u64_u32"
            | "v_mad_i64_i32"
            | "v_mad_co_u64_u32"
            | "v_mad_co_i64_i32"
            | "v_lshl_add_u64"
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemOp {
    Load,
    Store,
    Atomic { commutative: bool },
    LdsDma,
    Other,
}
fn mem_op(name: &str) -> MemOp {
    if name.contains("atomic") {
        let noncommutative = name.contains("cmpswap")
            || name.contains("swap")
            || name.contains("cond_sub")
            || name.contains("csub");
        return MemOp::Atomic {
            commutative: !noncommutative,
        };
    }
    if name.contains("_lds_") || name.ends_with("_lds") {
        return MemOp::LdsDma;
    }
    if name.contains("_load") {
        return MemOp::Load;
    }
    if name.contains("_store") {
        return MemOp::Store;
    }
    MemOp::Other
}

/// Resolution of one access base.
enum Base {
    Args {
        hi: DwSet,
        lo: DwSet,
    },
    Side(u8),
    /// Derived only from the sealed-table entries named by the site mask.
    Table(u8),
    /// Either a sealed-table entry (site mask) or an argument pointer (the
    /// roots): both are plain candidates of one read.
    Both {
        mask: u8,
        hi: DwSet,
        lo: DwSet,
    },
    Unresolved(&'static str),
}
fn resolve_base(lo: &Val, hi: &Val) -> Base {
    if hi.src & ROOT_OVF != 0 || lo.src & ROOT_OVF != 0 {
        return Base::Unresolved("base-root-overflow");
    }
    if hi.tbl != 0 || lo.tbl != 0 {
        // A sealed-table pointer is a base only as a whole, with nothing else
        // mixed into either half.
        let clean = hi.tbl != 0 && hi.tbl == lo.tbl && hi.src == 0 && lo.src == 0;
        return if clean && hi.roots.is_empty() && lo.roots.is_empty() {
            Base::Table(hi.tbl)
        } else if clean && !hi.roots.is_empty() {
            Base::Both {
                mask: hi.tbl,
                hi: hi.roots,
                lo: lo.roots,
            }
        } else {
            Base::Unresolved("sealed-table-base-mixed")
        };
    }
    if hi.src & OPAQUE != 0 {
        return Base::Unresolved("base-memory-derived");
    }
    let side = hi.src & (KSEG | APERTURE | PCREL | PACKET | SCRATCH);
    let categories = u32::from(!hi.roots.is_empty()) + side.count_ones();
    if categories == 0 {
        return Base::Unresolved("base-no-provenance");
    }
    if categories > 1 {
        return Base::Unresolved("base-mixed");
    }
    if !hi.roots.is_empty() {
        Base::Args {
            hi: hi.roots,
            lo: lo.roots,
        }
    } else {
        Base::Side(side)
    }
}

/// Kernarg segment layout: declared arguments plus the dword → argument map.
struct Layout {
    args: Vec<Kernarg>,
    size: u32,
}
impl Layout {
    fn arg_of(&self, dw: u32) -> Option<usize> {
        let byte = dw * 4;
        self.args
            .iter()
            .position(|a| byte + 4 > a.offset && byte < a.offset + a.size.max(1))
    }
    fn is_pointer(&self, index: usize) -> bool {
        let a = &self.args[index];
        a.address_space.is_some()
            || a.value_kind == "global_buffer"
            || matches!(
                a.value_kind.as_str(),
                "hidden_printf_buffer"
                    | "hidden_hostcall_buffer"
                    | "hidden_default_queue"
                    | "hidden_completion_action"
                    | "hidden_multigrid_sync_arg"
                    | "hidden_heap_v1"
                    | "hidden_queue_ptr"
                    | "dynamic_shared_pointer"
            )
    }
}

/// A resolved argument target: a declared argument or a synthetic slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Target {
    Declared(usize),
    Synthetic(u32),
}

/// A qualifying index load (§A9.3 rule 1), recorded during the replay.
struct LoadInfo {
    pc: u32,
    src_arg_offset: u32,
    class: AccessClass,
    path: MemPath,
    cache: CacheBits,
    byte_offset: IdxExpr,
}

/// An access resolved through the address witness, before the index-load
/// records it names are attached (`finish`).
struct Pending {
    inst: InstId,
    pc: u32,
    base_arg_offset: u32,
    class: AccessClass,
    path: MemPath,
    cache: CacheBits,
    width_bytes: u8,
    disp: IdxExpr,
    loads: Vec<InstId>,
}

/// A sealed-table entry load seen by the collecting pass: its shape is right,
/// its guard is not yet proved.
struct SealedCand {
    inst: InstId,
    pos: usize,
    table_arg_offset: u32,
    /// The 32-bit index value (`Load(_)` or `K32(_)` node).
    index: Ix,
    index_inst: Option<InstId>,
    index_kernarg: Option<u32>,
}

/// A qualified entry load: its table, bound and guard compares.
struct SealedSite {
    /// Bit in `Val::tbl`.
    bit: u8,
    table_arg_offset: u32,
    bound: SealedBound,
    index: SealedIndex,
    guard_pcs: Vec<u32>,
}

#[derive(Default)]
struct SealedCtx {
    /// Entry loads of the right shape, collected by the recording pass.
    cands: RefCell<BTreeMap<InstId, SealedCand>>,
    /// `None` while collecting; then the loads whose index is proved below
    /// its bound on every path.
    qualified: Option<BTreeMap<InstId, SealedSite>>,
}

/// A qualified entry load as the recording pass executed it.
struct EntryRec {
    pc: u32,
    class: AccessClass,
    path: MemPath,
    cache: CacheBits,
    width: u8,
}

/// A read through a base derived from sealed-table entries.
struct TableAccessRec {
    inst: InstId,
    pc: u32,
    mask: u8,
    class: AccessClass,
    path: MemPath,
    cache: CacheBits,
    width: u8,
}

struct Rec {
    accesses: BTreeMap<Target, Vec<ArgAccess>>,
    unresolved: Vec<Unresolved>,
    side: SideReads,
    loaded: DwSet,
    loaded_ovf: bool,
    dynamic: bool,
    sinks: Vec<u8>,
    taint_overflow: bool,
    loads: BTreeMap<InstId, LoadInfo>,
    pending: Vec<Pending>,
    /// Indices (into the supplied guards) of kernarg guards whose constant
    /// was substituted on the replayed CFG.
    guards_used: BTreeSet<usize>,
    sealed_entries: BTreeMap<InstId, EntryRec>,
    sealed_accesses: Vec<TableAccessRec>,
}
impl Rec {
    fn new(loaded: DwSet) -> Self {
        Rec {
            accesses: BTreeMap::new(),
            unresolved: Vec::new(),
            side: SideReads::default(),
            loaded,
            loaded_ovf: false,
            dynamic: false,
            sinks: vec![0; MAX_DW as usize],
            taint_overflow: false,
            loads: BTreeMap::new(),
            pending: Vec::new(),
            guards_used: BTreeSet::new(),
            sealed_entries: BTreeMap::new(),
            sealed_accesses: Vec::new(),
        }
    }
    fn sink(&mut self, taint: DwSet, tf: u8, bit: u8) {
        for dw in taint.iter() {
            self.sinks[dw as usize] |= bit;
        }
        if tf & TAINT_OVF != 0 {
            self.taint_overflow = true;
        }
    }
    fn fail(&mut self, arch: Arch, dec: &Dec, rule: &'static str) {
        let text = dec
            .inst
            .text(arch)
            .unwrap_or_else(|_| format!("{:?}", dec.inst.op));
        self.unresolved.push(Unresolved {
            inst: Some(dec.id),
            rule,
            detail: text,
        });
    }
}

/// The guarded case: kernarg dwords replaced by constants, plus the supplied
/// guard list (sorted, deduplicated).
struct GuardCtx {
    supplied: Vec<Guard>,
    /// Kernarg byte offset (4-aligned) -> (substituted dword, guard index).
    consts: BTreeMap<u32, (u32, usize)>,
}

struct Analyzer<'k> {
    arch: Arch,
    wave: Wave,
    body: &'k Body,
    layout: Layout,
    entry: State,
    /// Blocks inside a CFG cycle (A4 loop-bound vs guard).
    cyclic: Vec<bool>,
    /// The same, per layout position.
    pos_cyclic: Vec<bool>,
    decoded: Vec<Option<Dec<'k>>>,
    /// Integer/address witness nodes (`kernarg_ix`).
    ix: RefCell<Interner>,
    /// `Some` for the guarded analysis: constants are substituted and
    /// branches whose SCC is then known are pruned.
    guard: Option<GuardCtx>,
    /// `Some` for the sealed-table analysis ([`analyze_sealed`]).
    sealed: Option<SealedCtx>,
    /// Compare positions that produced each flag atom (sealed analysis).
    atom_pc: RefCell<BTreeMap<Ix, BTreeSet<u32>>>,
    /// The access the null-guard replay watches and the table bits its base
    /// carried when it was replayed.
    probe: RefCell<(Option<InstId>, u8)>,
}

/// A1/A3/A4 facts for `kernel`. Never fails: an unsupported arch or ABI, or
/// an instruction the pass does not model, yields `KernelStatus::Unknown`
/// with the reason in `unresolved`.
pub fn analyze(kernel: &Kernel, arch: Arch) -> KernargFacts {
    analyze_impl(kernel, arch, None, false)
}

/// The guarded analysis of amendment 2026-10-25 §A9.1/§A9.3: the same pass
/// with the `guards` substituted into the lifted ISA (kernarg constants) and
/// recorded (grid/block), following the CFG that results (a branch whose SCC
/// is then constant keeps only its live edge). Every unresolved access on
/// that CFG needs a certificate; accesses whose base is `direct pointer
/// argument + integer displacement over loaded `i32` indices` are resolved
/// through the address witness and reported as [`IndexedArgEffect`]s.
///
/// Returns `Conditional` when everything resolves and the proof used an
/// indexed effect or a substituted guard, `Proven` when it needed neither
/// (guards cleared), `Unknown` otherwise. The unguarded [`analyze`] result is
/// unaffected and stays the diagnostic baseline.
pub fn analyze_guarded(kernel: &Kernel, arch: Arch, guards: &[Guard]) -> KernargFacts {
    analyze_impl(kernel, arch, Some(guards), false)
}

/// The sealed-table analysis (design: `SealedTable` certificate). The same
/// pass with extra proof obligations for kernels that read a 64-bit pointer
/// from `table[i]` and then only read through it (`base-memory-derived` in the
/// unguarded analysis):
///
/// 1. the entry load's address is exactly `table pointer argument +
///    (ext(i) << 3)` with `i` the 32-bit value one instruction loaded (or a
///    kernarg scalar), outside every CFG cycle;
/// 2. on **every** feasible path from the kernel entry to the entry load a
///    scalar unsigned compare established `i < n` (`n` a kernarg scalar or a
///    constant). Paths are enumerated and replayed with the pass's own
///    transfer function (exact constants, SCC, VCC; infeasible branch edges
///    are pruned), so a flag materialised by a diamond is followed, not
///    guessed;
/// 3. every access whose base derives from such an entry is a plain load
///    (`vmem_load`/`smem_load`), no entry-derived value is stored, used in a
///    write/LDS address or recombined with another base, and a loaded base is
///    never an address again.
///
/// Returns `Conditional` (with [`KernargFacts::sealed`]) when everything
/// resolves, else `Unknown` with the exact rule; unqualifiable entry loads add
/// `sealed-*` rules. The result is void without a matching `sealed_table`
/// runtime witness.
pub fn analyze_sealed(kernel: &Kernel, arch: Arch) -> KernargFacts {
    analyze_impl(kernel, arch, None, true)
}

fn analyze_impl(
    kernel: &Kernel,
    arch: Arch,
    guards: Option<&[Guard]>,
    sealed: bool,
) -> KernargFacts {
    let (args, size) = match &kernel.abi {
        Abi::Hsa {
            metadata,
            descriptor,
        } => {
            let size = if metadata.parsed.kernarg_segment_size != 0 {
                metadata.parsed.kernarg_segment_size
            } else {
                descriptor.kernarg_size
            };
            (metadata.parsed.args.clone(), size)
        }
        Abi::Raw { .. } => (Vec::new(), 0),
    };
    let layout = Layout { args, size };
    let mut facts = KernargFacts {
        kernarg_size: size,
        ..KernargFacts::default()
    };
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        facts.unresolved.push(Unresolved {
            inst: None,
            rule: "unsupported-arch",
            detail: format!("{arch:?}"),
        });
        return finish(facts, &layout, None, &[], Vec::new());
    }
    let body = &kernel.body;
    if body.blocks.is_empty() {
        facts.unresolved.push(Unresolved {
            inst: None,
            rule: "blocks-not-built",
            detail: String::new(),
        });
        return finish(facts, &layout, None, &[], Vec::new());
    }
    let mut decoded = Vec::with_capacity(body.layout.len());
    for (pos, &id) in body.layout.iter().enumerate() {
        decoded.push(
            body.insts
                .get(id)
                .and_then(|inst| decode(arch, id, inst))
                .map(|d| Dec { pos, ..d }),
        );
    }
    let (mut entry, entry_loaded) = entry_state(kernel, arch);
    let mut interner = Interner::default();
    if arch == Arch::Gfx1201 && matches!(kernel.abi, Abi::Hsa { .. }) {
        entry.t[9].ix = interner.entry(EntryReg::WorkgroupIdX);
        entry.t[7].ix = interner.entry(EntryReg::WorkgroupIdYZ);
        entry.v[0].ix = interner.entry(EntryReg::WorkitemIdPacked);
    }
    let mut rec = Rec::new(entry_loaded);
    let guard = guards.map(|g| {
        let mut supplied = g.to_vec();
        supplied.sort();
        supplied.dedup();
        let mut consts = BTreeMap::new();
        for (gi, guard) in supplied.iter().enumerate() {
            let ok = match *guard {
                Guard::KernargEq {
                    offset,
                    size,
                    value,
                } if offset % 4 == 0
                    && (size == 8 || (size == 4 && value <= u64::from(u32::MAX))) =>
                {
                    let words = if size == 8 { 2 } else { 1 };
                    (0..words).all(|i| {
                        let word = (value >> (32 * i)) as u32;
                        consts.insert(offset + 4 * i as u32, (word, gi)).is_none()
                    })
                }
                Guard::GridDim { axis, .. } | Guard::BlockDim { axis, .. } => axis < 3,
                _ => false,
            };
            if !ok {
                rec.unresolved.push(Unresolved {
                    inst: None,
                    rule: "guard-unsupported",
                    detail: format!("{guard:?}"),
                });
            }
        }
        GuardCtx { supplied, consts }
    });
    let cyclic = cyclic_blocks(body);
    let mut pos_cyclic = vec![false; body.layout.len()];
    for (b, block) in body.blocks.iter().enumerate() {
        for pos in block.range.0..block.range.1 {
            pos_cyclic[pos] = cyclic[b];
        }
    }
    let mut an = Analyzer {
        arch,
        wave: kernel.wave,
        body,
        layout,
        entry,
        cyclic,
        pos_cyclic,
        decoded,
        ix: RefCell::new(interner),
        guard,
        sealed: sealed.then(SealedCtx::default),
        atom_pc: RefCell::new(BTreeMap::new()),
        probe: RefCell::new((None, 0)),
    };
    let ins = an.fixpoint();
    an.record(&ins, &mut rec);
    let mut rejected: Vec<Unresolved> = Vec::new();
    let mut sealed_uses: Vec<SealedTableUse> = Vec::new();
    if let Some(ctx) = &an.sealed {
        let mut cands: Vec<SealedCand> = std::mem::take(&mut *ctx.cands.borrow_mut())
            .into_values()
            .collect();
        cands.sort_by_key(|c| c.pos);
        let (qualified, rej) = an.qualify(cands);
        rejected = rej;
        if !qualified.is_empty() {
            if let Some(ctx) = an.sealed.as_mut() {
                ctx.qualified = Some(qualified);
            }
            // Second run: the qualified entry loads now carry their table
            // provenance, so every access through them is attributed.
            let ins = an.fixpoint();
            rec = Rec::new(entry_loaded);
            an.record(&ins, &mut rec);
            sealed_uses = an.assemble_sealed(&rec);
        }
    }
    rejected.append(&mut rec.unresolved);
    facts.unresolved = rejected;
    let supplied = an.guard.take().map(|g| g.supplied).unwrap_or_default();
    let layout = std::mem::replace(
        &mut an.layout,
        Layout {
            args: Vec::new(),
            size: 0,
        },
    );
    finish(facts, &layout, Some(rec), &supplied, sealed_uses)
}

fn cyclic_blocks(body: &Body) -> Vec<bool> {
    let mut graph: DiGraph<(), ()> = DiGraph::new();
    let nodes: Vec<NodeIndex> = body.blocks.iter().map(|_| graph.add_node(())).collect();
    for block in &body.blocks {
        for succ in &block.succs {
            graph.add_edge(nodes[block.id.0], nodes[succ.0], ());
        }
    }
    let mut out = vec![false; body.blocks.len()];
    for scc in petgraph::algo::tarjan_scc(&graph) {
        let looped = scc.len() > 1 || graph.contains_edge(scc[0], scc[0]);
        if looped {
            for n in scc {
                out[n.index()] = true;
            }
        }
    }
    out
}

/// HSA user-SGPR roles in `KERNEL_CODE_PROPERTIES` bit order.
#[derive(Clone, Copy)]
enum HsaUserSgpr {
    PrivateSegmentBuffer,
    DispatchPtr,
    QueuePtr,
    KernargSegmentPtr,
    DispatchId,
    FlatScratchInit,
    PrivateSegmentSize,
}

/// Entry values of a raw kernel: exactly its declared facts. A workgroup id or packed local
/// id keeps its influence flag, an exact value its constant, an unconstrained defined value an
/// opaque non-argument source; nothing is a kernarg pointer.
fn raw_entry_state(raw: &RawAbi) -> (State, DwSet) {
    let mut st = State::empty();
    // Malformed facts were rejected by `Program::validate`; analysis of an unvalidated
    // program sees an empty entry state, which is the conservative reading.
    let Ok(cells) = raw.entry_cells() else {
        return (st, DwSet::default());
    };
    for (cell, fact) in cells {
        let val = match &fact.value {
            EntryValue::Defined => Val::src(OPAQUE),
            EntryValue::WorkgroupId(_) => Val::flags(WG),
            EntryValue::Exact(value) => Val::konst(*value),
            EntryValue::Bounded { .. } => Val::default(),
            EntryValue::LocalId(_) => Val::flags(WI),
        };
        match cell {
            EntryCell::Vgpr(n) => {
                if let Some(slot) = st.v.get_mut(usize::from(n)) {
                    *slot = val
                }
            }
            EntryCell::Sgpr(n) => {
                if let Some(slot) = st.s.get_mut(usize::from(n)) {
                    *slot = val
                }
            }
            EntryCell::Ttmp(n) => {
                if let Some(slot) = st.t.get_mut(usize::from(n)) {
                    *slot = val
                }
            }
            _ => {}
        }
    }
    (st, DwSet::default())
}

/// ABI entry values: user SGPRs by role, preloaded kernargs, workgroup ids
/// (gfx11 system SGPRs, gfx12 TTMP7/TTMP9), work-item ids in v0.
fn entry_state(kernel: &Kernel, arch: Arch) -> (State, DwSet) {
    let descriptor = match &kernel.abi {
        Abi::Hsa { descriptor, .. } => descriptor,
        Abi::Raw(raw) => return raw_entry_state(raw),
    };
    let mut st = State::empty();
    let mut loaded = DwSet::default();
    let mut roles: Vec<(HsaUserSgpr, usize)> = Vec::new();
    let props = descriptor.kernel_code_properties.0;
    let table = [
        (HsaUserSgpr::PrivateSegmentBuffer, 4),
        (HsaUserSgpr::DispatchPtr, 2),
        (HsaUserSgpr::QueuePtr, 2),
        (HsaUserSgpr::KernargSegmentPtr, 2),
        (HsaUserSgpr::DispatchId, 2),
        (HsaUserSgpr::FlatScratchInit, 2),
        (HsaUserSgpr::PrivateSegmentSize, 1),
    ];
    for (bit, (role, n)) in table.into_iter().enumerate() {
        if props >> bit & 1 != 0 {
            roles.push((role, n));
        }
    }
    let pre = u32::from(descriptor.kernarg_preload.0);
    let rsrc2 = descriptor.compute_pgm_rsrc2.0;
    let (user_count, preload) = ((rsrc2 >> 1 & 0x1f) as usize, (pre & 0x7f, pre >> 7));
    let mut next = 0usize;
    for (role, n) in &roles {
        let val = match role {
            HsaUserSgpr::PrivateSegmentBuffer => Val::src(SCRATCH),
            HsaUserSgpr::DispatchPtr | HsaUserSgpr::QueuePtr => Val::src(PACKET),
            HsaUserSgpr::KernargSegmentPtr => Val {
                koff: Some(0),
                ..Val::src(KSEG)
            },
            HsaUserSgpr::FlatScratchInit => Val::src(SCRATCH),
            _ => Val::default(),
        };
        for i in 0..*n {
            if next + i < NS {
                st.s[next + i] = val;
            }
        }
        next += n;
    }
    // Kernarg preload (KERNARG_PRELOAD_SPEC_LENGTH/OFFSET, in dwords).
    let (count, offset) = preload;
    for i in 0..count {
        let dw = offset + i;
        if next < NS {
            st.s[next] = Val::karg(dw);
        }
        if dw < MAX_DW {
            loaded = loaded.union(DwSet::one(dw));
        }
        next += 1;
    }
    if arch == Arch::Gfx1201 {
        st.t[9] = Val::flags(WG);
        st.t[7] = Val::flags(WG);
    } else {
        let mut sys = user_count.max(next);
        for bit in 7..=10 {
            if rsrc2 >> bit & 1 != 0 {
                if sys < NS {
                    st.s[sys] = Val::flags(WG);
                }
                sys += 1;
            }
        }
    }
    st.v[0] = Val::flags(WI);
    (st, loaded)
}

fn lane_of(st: &State, op: &Operand) -> Option<u8> {
    let value = match op {
        Operand::Reg(r) if r.kind == Kind::S => st.s.get(usize::from(r.base))?.konst?,
        other => inline_value(other)?,
    };
    u8::try_from(value).ok()
}

/// An `s_*_saveexec` operation: EXEC after the save, relative to the saved mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SaveOp {
    /// `s_and_saveexec` (`EXEC &= S0`) and `s_and_not0_saveexec` (`EXEC &= !S0`).
    Narrow,
    Or,
    Xor,
    /// `s_and_not1_saveexec`: `EXEC = S0 & !EXEC`.
    SrcAndNotExec,
    Other,
}

/// An instruction's effect on EXEC and on the mask facts of its SGPR results.
#[derive(Clone, Debug)]
enum ExecPlan {
    None,
    MaskResult {
        dst: SmallVec<[usize; 2]>,
        mask: Mask,
    },
    CopyExec {
        dst: SmallVec<[usize; 2]>,
    },
    Narrow,
    Set(Mask),
    SaveExec {
        dst: SmallVec<[usize; 2]>,
        op: SaveOp,
        src: Mask,
    },
}

impl<'k> Analyzer<'k> {
    fn fixpoint(&self) -> Vec<Option<State>> {
        let n = self.body.blocks.len();
        let mut ins: Vec<Option<State>> = vec![None; n];
        ins[0] = Some(self.entry.clone());
        let mut work: Vec<usize> = vec![0];
        let mut queued = vec![false; n];
        queued[0] = true;
        while let Some(b) = work.pop() {
            queued[b] = false;
            let Some(mut st) = ins[b].clone() else {
                continue;
            };
            let block = &self.body.blocks[b];
            for pos in block.range.0..block.range.1 {
                self.step(&mut st, pos, None, false);
            }
            for target in self.live_succs(block, &st) {
                let changed = match &mut ins[target] {
                    Some(existing) => {
                        let before = existing.clone();
                        existing.join_from(&st);
                        *existing != before
                    }
                    slot @ None => {
                        *slot = Some(st.clone());
                        true
                    }
                };
                if changed && !queued[target] {
                    queued[target] = true;
                    work.push(target);
                }
            }
        }
        ins
    }

    /// Which edge of a conditional branch `st` decides: `Some(true)` the taken
    /// edge only, `Some(false)` the fall-through only, `None` both. The
    /// guarded analysis decides SCC branches whose SCC is a known constant;
    /// the sealed analysis additionally decides wave32 VCC branches on a
    /// known `vcc_lo` (the hardware sets VCCZ from `vcc_lo` alone in wave32,
    /// which the compiler relies on). EXEC branches are never decided.
    fn branch_outcome(&self, cond: crate::cfg::Cond, st: &State) -> Option<bool> {
        use crate::cfg::Cond;
        if self.guard.is_none() && self.sealed.is_none() {
            return None;
        }
        match cond {
            Cond::Scc0 | Cond::Scc1 => {
                let scc = st.scc.konst?;
                Some((scc != 0) == (cond == Cond::Scc1))
            }
            Cond::Vccz | Cond::Vccnz if self.sealed.is_some() && self.wave == Wave::Wave32 => {
                let vcc = st.s[VCC_LO].konst?;
                Some((vcc == 0) == (cond == Cond::Vccz))
            }
            _ => None,
        }
    }

    /// Successors of `block` that can execute after `st`. In the guarded and
    /// sealed analyses a conditional branch decided by [`Self::branch_outcome`]
    /// keeps only its live edge; otherwise every edge.
    fn live_succs(&self, block: &crate::cfg::Block, st: &State) -> SmallVec<[usize; 2]> {
        use crate::cfg::Terminator;
        if let Terminator::Branch {
            cond,
            taken,
            fallthrough,
        } = &block.term
        {
            if let Some(take) = self.branch_outcome(*cond, st) {
                return smallvec::smallvec![if take { taken.0 } else { fallthrough.0 }];
            }
        }
        block.succs.iter().map(|s| s.0).collect()
    }

    fn read(&self, st: &State, op: &Operand, bits: u16) -> SmallVec<[Val; 4]> {
        let width = usize::from((bits / 32).max(1));
        let mut out = SmallVec::new();
        match op {
            Operand::Reg(r) | Operand::Half(r, _) => {
                for i in 0..usize::from(r.len) {
                    let idx = usize::from(r.base) + i;
                    out.push(match r.kind {
                        Kind::S => st.s.get(idx).copied().unwrap_or_default(),
                        Kind::Ttmp => st.t.get(idx).copied().unwrap_or_default(),
                        Kind::V => {
                            if idx < NV {
                                st.read_v(idx as u16)
                            } else {
                                Val::default()
                            }
                        }
                    });
                }
            }
            Operand::Special(sp) => {
                let one =
                    |v: Val| -> SmallVec<[Val; 4]> { std::iter::repeat(v).take(width).collect() };
                return match sp {
                    Special::Vcc => [st.s[VCC_LO], st.s[VCC_HI]]
                        .into_iter()
                        .take(width.max(1))
                        .collect(),
                    Special::VccLo => one(st.s[VCC_LO]),
                    Special::VccHi => one(st.s[VCC_HI]),
                    Special::Exec => st.exec.iter().copied().take(width).collect(),
                    Special::ExecLo => one(st.exec[0]),
                    Special::ExecHi => one(st.exec[1]),
                    Special::Scc => one(st.scc),
                    Special::M0 => one(st.m0),
                    Special::Null => one(Val::konst(0)),
                    Special::Ttmp(n) => one(st.t[usize::from(*n)]),
                    Special::SrcSharedBase
                    | Special::SrcSharedLimit
                    | Special::SrcPrivateBase
                    | Special::SrcPrivateLimit => one(Val::src(APERTURE)),
                    Special::Pc => one(Val::src(PCREL)),
                    Special::FlatScratch => one(Val::src(SCRATCH)),
                };
            }
            other => {
                let value = inline_value(other);
                out.push(value.map(Val::konst).unwrap_or_default());
                for _ in 1..width {
                    let hi = value.map(|v| {
                        if other_is_negative(other) && v >> 31 != 0 {
                            u32::MAX
                        } else {
                            0
                        }
                    });
                    out.push(hi.map(Val::konst).unwrap_or_default());
                }
            }
        }
        while out.len() < width {
            let last = out.last().copied().unwrap_or_default();
            out.push(last);
        }
        out
    }

    /// Write `vals` to `op`. VGPR writes are strong for the enabled lanes
    /// (disabled lanes live in `State::inactive`); half writes join.
    fn write(&self, st: &mut State, op: &Operand, vals: &[Val]) {
        let val_at = |i: usize| {
            vals.get(i)
                .copied()
                .or_else(|| vals.last().copied())
                .unwrap_or_default()
        };
        match op {
            Operand::Reg(r) => {
                for i in 0..usize::from(r.len) {
                    let idx = usize::from(r.base) + i;
                    let val = val_at(i);
                    match r.kind {
                        Kind::S => {
                            if idx < NS {
                                st.s[idx] = val;
                                st.masks[idx] = Mask::Unknown;
                            }
                        }
                        Kind::Ttmp => {
                            if idx < 16 {
                                st.t[idx] = val;
                            }
                        }
                        Kind::V => {
                            if idx < NV {
                                self.write_v(st, idx as u16, val, true);
                            }
                        }
                    }
                }
            }
            Operand::Half(r, _) => {
                let idx = usize::from(r.base);
                match r.kind {
                    Kind::V if idx < NV => self.write_v(st, idx as u16, val_at(0), false),
                    Kind::S if idx < NS => st.s[idx] = st.s[idx].join(val_at(0)),
                    _ => {}
                }
            }
            Operand::Special(sp) => match sp {
                Special::Vcc => {
                    st.s[VCC_LO] = val_at(0);
                    st.s[VCC_HI] = val_at(1);
                    st.masks[VCC_LO] = Mask::Unknown;
                    st.masks[VCC_HI] = Mask::Unknown;
                }
                Special::VccLo => {
                    st.s[VCC_LO] = val_at(0);
                    st.masks[VCC_LO] = Mask::Unknown;
                }
                Special::VccHi => {
                    st.s[VCC_HI] = val_at(0);
                    st.masks[VCC_HI] = Mask::Unknown;
                }
                Special::Exec => {
                    st.exec = [val_at(0), val_at(1)];
                }
                Special::ExecLo => st.exec[0] = val_at(0),
                Special::ExecHi => st.exec[1] = val_at(0),
                Special::M0 => st.m0 = val_at(0),
                Special::Scc => st.scc = val_at(0),
                Special::Ttmp(n) => st.t[usize::from(*n)] = val_at(0),
                _ => {}
            },
            _ => {}
        }
    }

    fn write_v(&self, st: &mut State, r: u16, val: Val, strong: bool) {
        let keys: SmallVec<[(u16, u8); 4]> = st
            .lanes
            .range((r, 0)..=(r, u8::MAX))
            .map(|(k, _)| *k)
            .collect();
        if strong {
            st.v[usize::from(r)] = val;
            for k in keys {
                st.lanes.remove(&k);
            }
        } else {
            st.v[usize::from(r)] = st.v[usize::from(r)].join(val);
            for k in keys {
                let old = st.lanes[&k];
                st.lanes.insert(k, old.join(val));
            }
        }
    }

    fn vals(&self, st: &State, f: &Field) -> SmallVec<[Val; 4]> {
        self.read(st, f.op, f.bits)
    }

    /// Transfer one instruction; with `rec`, also record facts.
    fn step(&self, st: &mut State, pos: usize, mut rec: Option<&mut Rec>, cyclic: bool) {
        let Some(dec) = &self.decoded[pos] else {
            if let Some(rec) = rec.as_deref_mut() {
                let id = self.body.layout[pos];
                rec.unresolved.push(Unresolved {
                    inst: Some(id),
                    rule: "undecodable",
                    detail: String::new(),
                });
            }
            return;
        };
        let inst = dec.inst;
        let name = dec.parts[0].name;
        if let Some(rec) = rec.as_deref_mut() {
            self.record_sinks(st, dec, rec, cyclic);
        }

        if name == "s_setpc_b64"
            || name == "s_swappc_b64"
            || name.starts_with("s_call")
            || name == "s_rfe_b64"
        {
            if let Some(rec) = rec.as_deref_mut() {
                rec.fail(self.arch, dec, "call");
            }
        }
        let fx = if self.sealed.is_some() {
            self.sealed_scalar(st, dec)
        } else {
            SealedFx::default()
        };
        let mut carry: Option<(CarryTo, Ix)> = None;
        let mut scc_konst: Option<bool> = None;
        if inst.effects.mem.is_some() || matches!(inst.form, Form::Vmem(_) | Form::Smem | Form::Ds)
        {
            self.memory(st, dec, rec.as_deref_mut());
        } else {
            // Evaluate every part against the pre-state (VOPD reads both halves first).
            let mut writes: SmallVec<[(&Operand, SmallVec<[Val; 4]>); 4]> = SmallVec::new();
            let mut lane_writes: SmallVec<[(u16, Option<u8>, Val); 1]> = SmallVec::new();
            scc_konst = self.cmp_konst(st, dec);
            for part in &dec.parts {
                self.eval(st, part, &mut writes, &mut lane_writes, &mut carry);
            }
            let plan = self.exec_plan(st, dec);
            let valu = matches!(
                inst.form,
                Form::Vop1
                    | Form::Vop1Dpp
                    | Form::Vop2
                    | Form::Vop2Dpp
                    | Form::Vop3
                    | Form::Vop3p
                    | Form::Vopc
                    | Form::Vopd
            );
            for (op, vals) in &writes {
                // A wave32 VALU SGPR destination (carry-out, compare mask) is
                // one SGPR even where the grammar types it 64-bit.
                match op {
                    Operand::Reg(r)
                        if valu && self.wave == Wave::Wave32 && r.kind == Kind::S && r.len == 2 =>
                    {
                        let low = Operand::Reg(crate::reg::RegRef { len: 1, ..*r });
                        self.write(st, &low, vals);
                    }
                    _ => self.write(st, op, vals),
                }
            }
            for (r, lane, val) in lane_writes {
                match lane {
                    Some(l) => {
                        st.lanes.insert((r, l), val);
                    }
                    None => self.write_v(st, r, val, false),
                }
            }
            self.apply_plan(st, plan);
        }
        if inst.effects.implicit.writes & ImplicitSet::SCC != 0 {
            st.scc = Val::default();
        }
        if inst.effects.implicit.writes & ImplicitSet::VCC != 0
            && !dec.parts.iter().any(|p| {
                p.fields.iter().any(|f| {
                    f.def
                        && matches!(
                            f.op,
                            Operand::Special(Special::Vcc | Special::VccLo | Special::VccHi)
                        )
                })
            })
        {
            st.s[VCC_LO] = Val::default();
            if self.wave == Wave::Wave64 {
                st.s[VCC_HI] = Val::default();
            }
            if !name.starts_with("v_cmp_") {
                st.masks[VCC_LO] = Mask::Unknown;
                st.masks[VCC_HI] = Mask::Unknown;
            }
        }
        if inst.effects.implicit.writes & ImplicitSet::M0 != 0
            && !dec.parts.iter().any(|p| {
                p.fields
                    .iter()
                    .any(|f| f.def && matches!(f.op, Operand::Special(Special::M0)))
            })
        {
            st.m0 = Val::default();
        }
        // Results the implicit SCC/VCC clearing above must not erase: a
        // compare of constants, and the carry mask of a matched add.
        if let Some(k) = scc_konst {
            st.scc.konst = Some(u32::from(k));
        }
        match carry {
            Some((CarryTo::Scc, ix)) => st.scc.ix = ix,
            Some((CarryTo::Vcc, ix)) => st.s[VCC_LO].ix = ix,
            None => {}
        }
        // Sealed analysis: exact scalar results (constants through logic ops
        // and decided selects) and the unsigned-compare witness of SCC.
        if let Some((i, vals)) = fx.dst_vals {
            for (k, v) in vals.into_iter().enumerate() {
                if i + k < NS {
                    st.s[i + k] = v;
                    st.masks[i + k] = Mask::Unknown;
                }
            }
        }
        if let Some((i, c)) = fx.dst_konst {
            if i < NS {
                st.s[i].konst = Some(c);
            }
        }
        if let Some(ix) = fx.scc_ix {
            st.scc.ix = ix;
        }
        if let Some(k) = fx.scc_konst {
            st.scc.konst = Some(u32::from(k));
        }
        if let Some((i, ix)) = fx.dst_ix {
            if i < NS {
                st.s[i].ix = ix;
            }
        }
        if self.sealed.is_some() && st.exec_nz {
            let writes_exec = inst.effects.implicit.writes & ImplicitSet::EXEC != 0
                || dec.parts.iter().any(|p| {
                    p.fields.iter().any(|f| {
                        f.def
                            && matches!(
                                f.op,
                                Operand::Special(Special::Exec | Special::ExecLo | Special::ExecHi)
                            )
                    })
                });
            // `s_or_b32 exec_lo, exec_lo, x` only adds lanes: still nonzero.
            let superset = dec.parts.len() == 1
                && norm_name(dec.parts[0].name) == "s_or_b32"
                && dec.parts[0]
                    .fields
                    .iter()
                    .any(|f| f.def && matches!(f.op, Operand::Special(Special::ExecLo)))
                && dec.parts[0]
                    .sources()
                    .any(|f| matches!(f.op, Operand::Special(Special::ExecLo)));
            if writes_exec && !superset {
                st.exec_nz = false;
            }
        }
        if fx.exec_nz {
            st.exec_nz = true;
        }
    }

    /// The mask value of an operand (for EXEC and saved-mask arithmetic).
    fn mask_of(&self, st: &State, op: &Operand) -> Mask {
        let sgprs = |base: usize, len: usize| -> Mask {
            let first = st.masks.get(base).copied().unwrap_or(Mask::Unknown);
            if (base..base + len).all(|i| st.masks.get(i) == Some(&first)) {
                first
            } else {
                Mask::Unknown
            }
        };
        match op {
            Operand::Special(Special::Exec | Special::ExecLo) => {
                let top = st.top();
                Mask::Within {
                    level: top,
                    exact: st.exec_exact == Some(top),
                }
            }
            Operand::Special(Special::Vcc | Special::VccLo) => {
                sgprs(VCC_LO, if self.wave == Wave::Wave64 { 2 } else { 1 })
            }
            Operand::Reg(r) if r.kind == Kind::S => {
                let len = if self.wave == Wave::Wave32 {
                    1
                } else {
                    usize::from(r.len)
                };
                sgprs(usize::from(r.base), len)
            }
            other => match inline_value(other) {
                Some(u32::MAX) => Mask::Full,
                Some(0) => Mask::Within {
                    level: st.top(),
                    exact: false,
                },
                _ => Mask::Unknown,
            },
        }
    }

    /// How this instruction changes EXEC and the mask facts of its SGPR
    /// destinations, computed on the pre-state.
    fn exec_plan(&self, st: &State, dec: &Dec) -> ExecPlan {
        let part = &dec.parts[0];
        let name = part.name;
        let is_exec = |op: &Operand| {
            matches!(
                op,
                Operand::Special(Special::Exec | Special::ExecLo | Special::ExecHi)
            )
        };
        let sgprs_of = |op: &Operand| -> SmallVec<[usize; 2]> {
            match op {
                Operand::Reg(r) if r.kind == Kind::S => {
                    let len = if self.wave == Wave::Wave32 {
                        1
                    } else {
                        usize::from(r.len)
                    };
                    (0..len)
                        .map(|i| usize::from(r.base) + i)
                        .filter(|&i| i < NS)
                        .collect()
                }
                Operand::Special(Special::VccLo) => smallvec::smallvec![VCC_LO],
                Operand::Special(Special::Vcc) => {
                    if self.wave == Wave::Wave64 {
                        smallvec::smallvec![VCC_LO, VCC_HI]
                    } else {
                        smallvec::smallvec![VCC_LO]
                    }
                }
                _ => SmallVec::new(),
            }
        };
        let srcs: SmallVec<[&Field; 2]> = part.sources().collect();
        let mask = |i: usize| {
            srcs.get(i)
                .map_or(Mask::Unknown, |f| self.mask_of(st, f.op))
        };
        let algebra = || -> Option<Mask> {
            Some(if name.starts_with("s_mov_b") {
                mask(0)
            } else if name.starts_with("s_and_b") {
                mask(0).and(mask(1))
            } else if name.starts_with("s_andn2_b") || name.starts_with("s_and_not1_b") {
                mask(0).and_not(mask(1))
            } else if name.starts_with("s_or_b") {
                mask(0).or(mask(1))
            } else if name.starts_with("s_xor_b") {
                mask(0).xor(mask(1))
            } else {
                return None;
            })
        };
        if name.contains("_saveexec_") {
            let dst = part
                .field("SDST")
                .map(|f| sgprs_of(f.op))
                .unwrap_or_default();
            let op = if name.starts_with("s_and_saveexec")
                || name.starts_with("s_and_not0_saveexec")
                || name.starts_with("s_andn1_saveexec")
            {
                SaveOp::Narrow
            } else if name.starts_with("s_or_saveexec") {
                SaveOp::Or
            } else if name.starts_with("s_xor_saveexec") {
                SaveOp::Xor
            } else if name.starts_with("s_and_not1_saveexec")
                || name.starts_with("s_andn2_saveexec")
            {
                SaveOp::SrcAndNotExec
            } else {
                SaveOp::Other
            };
            return ExecPlan::SaveExec {
                dst,
                op,
                src: mask(0),
            };
        }
        if part.fields.iter().any(|f| f.def && is_exec(f.op)) {
            let narrows = (name.starts_with("s_and_b") && srcs.iter().any(|f| is_exec(f.op)))
                || ((name.starts_with("s_andn2_b") || name.starts_with("s_and_not1_b"))
                    && srcs.first().is_some_and(|f| is_exec(f.op)));
            if narrows {
                return ExecPlan::Narrow;
            }
            return ExecPlan::Set(algebra().unwrap_or(Mask::Unknown));
        }
        if dec.inst.effects.implicit.writes & ImplicitSet::EXEC != 0 {
            return if name.starts_with("v_cmpx") {
                ExecPlan::Narrow
            } else {
                ExecPlan::Set(Mask::Unknown)
            };
        }
        // Mask facts for SGPR destinations.
        if name.starts_with("v_cmp_") {
            let dst = match part.fields.iter().find(|f| f.def) {
                Some(f) => sgprs_of(f.op),
                None => sgprs_of(&Operand::Special(Special::Vcc)),
            };
            return ExecPlan::MaskResult {
                dst,
                mask: Mask::Within {
                    level: st.top(),
                    exact: false,
                },
            };
        }
        let Some(dst) = part
            .field("SDST")
            .map(|f| sgprs_of(f.op))
            .filter(|d| !d.is_empty())
        else {
            return ExecPlan::None;
        };
        if name.starts_with("s_mov_b") && srcs.first().is_some_and(|f| is_exec(f.op)) {
            return ExecPlan::CopyExec { dst };
        }
        match algebra() {
            Some(mask) if mask != Mask::Unknown => ExecPlan::MaskResult { dst, mask },
            _ => ExecPlan::None,
        }
    }

    fn apply_plan(&self, st: &mut State, plan: ExecPlan) {
        match plan {
            ExecPlan::None => {}
            ExecPlan::MaskResult { dst, mask } => {
                for d in dst {
                    st.masks[d] = mask;
                }
            }
            ExecPlan::CopyExec { dst } => {
                let level = st.exec_copy_level();
                for d in dst {
                    st.masks[d] = Mask::Within { level, exact: true };
                }
            }
            ExecPlan::Narrow => st.exec_narrow(),
            ExecPlan::Set(mask) => st.exec_set(mask),
            ExecPlan::SaveExec { dst, op, src } => {
                let level = st.exec_copy_level();
                let saved = Mask::Within { level, exact: true };
                for d in dst {
                    st.masks[d] = saved;
                }
                match op {
                    SaveOp::Narrow => st.exec_narrow(),
                    SaveOp::Or => st.exec_set(src.or(saved)),
                    SaveOp::Xor => st.exec_set(src.xor(saved)),
                    SaveOp::SrcAndNotExec => st.exec_set(src.and_not(saved)),
                    SaveOp::Other => st.exec_set(Mask::Unknown),
                }
            }
        }
    }

    /// Values written by one opcode half. Every explicit destination gets a
    /// value: a destination the semantic rules below do not cover receives
    /// the provenance-free combination of the sources.
    fn eval<'a>(
        &self,
        st: &State,
        part: &Part<'a>,
        writes: &mut SmallVec<[(&'a Operand, SmallVec<[Val; 4]>); 4]>,
        lanes: &mut SmallVec<[(u16, Option<u8>, Val); 1]>,
        carry: &mut Option<(CarryTo, Ix)>,
    ) {
        let start = writes.len();
        let lane_start = lanes.len();
        self.eval_rules(st, part, writes, lanes);
        let uses: SmallVec<[Val; 8]> = part.sources().flat_map(|f| self.vals(st, f)).collect();
        for d in part.fields.iter().filter(|f| f.def) {
            let written = writes[start..]
                .iter()
                .any(|(op, _)| std::ptr::eq(*op, d.op))
                || lanes.len() > lane_start;
            if !written {
                writes.push((d.op, smallvec::smallvec![Val::kill(uses.iter())]));
            }
        }
        self.eval_ix(st, part, &mut writes[start..], carry);
    }

    /// The integer/address witness of this part's results (§A9.3 rule 2): exact
    /// bit-vector expressions through the modelled operations. An operation
    /// outside this list leaves its results unwitnessed (`Ix::NONE`), which is
    /// how unsupported arithmetic reaches `Unknown`.
    fn eval_ix<'a>(
        &self,
        st: &State,
        part: &Part<'a>,
        writes: &mut [(&'a Operand, SmallVec<[Val; 4]>)],
        carry: &mut Option<(CarryTo, Ix)>,
    ) {
        enum Res {
            One(Ix),
            Two(Ix),
        }
        let name = norm_name(part.name);
        let srcs: SmallVec<[&Field; 4]> = part.sources().collect();
        let (Some(&s0), Some(&s1)) = (srcs.first(), srcs.get(1)) else {
            if matches!(name.as_str(), "v_readfirstlane_b32" | "v_mov_b32_dpp") {
                if let Some(d) = part.fields.iter().find(|f| f.def) {
                    patch_ix(writes, d.op, &[Ix::NONE]);
                }
            }
            return;
        };
        let lvl = st.exec_exact.unwrap_or(EXEC_UNKNOWN);
        let mut it = self.ix.borrow_mut();
        let it = &mut *it;
        let one = |it: &mut Interner, f: &Field| -> Ix {
            ixof(it, self.vals(st, f).first().copied().unwrap_or_default())
        };
        let wide = |it: &mut Interner, f: &Field| -> Ix {
            let v = self.vals(st, f);
            if v.len() < 2 {
                return Ix::NONE;
            }
            let (lo, hi) = (ixof(it, v[0]), ixof(it, v[1]));
            it.as64(lo, hi)
        };
        let kon = |f: &Field| -> Option<u32> { self.vals(st, f).first().and_then(|v| v.konst) };
        let vdst = part
            .fields
            .iter()
            .find(|f| f.def && f.name == "VDST")
            .or_else(|| part.fields.iter().find(|f| f.def && f.name == "SDST"));
        let sdst = part.fields.iter().find(|f| f.def && f.name == "SDST");
        // High half of `A + B` given the matched carry and the high words.
        let hi_sum = |it: &mut Interner, cin: Ix, lvl: u8, ha: Ix, hb: Ix| -> Ix {
            let Some((ca, cb)) = it.carry_parts(cin, lvl) else {
                return Ix::NONE;
            };
            for (x, y) in [(ha, hb), (hb, ha)] {
                let (a, b) = (it.as64(ca, x), it.as64(cb, y));
                let w = it.add64(a, b);
                if !w.is_none() {
                    return it.hi_of(w);
                }
            }
            Ix::NONE
        };
        let res = match name.as_str() {
            "s_mul_i32" | "v_mul_lo_u32" => {
                let (x, y) = (one(it, s0), one(it, s1));
                Some(Res::One(it.mul32(x, y)))
            }
            "s_add_u32" | "s_add_co_u32" => {
                let (x, y) = (one(it, s0), one(it, s1));
                *carry = Some((CarryTo::Scc, it.carry(x, y, CARRY_SCALAR)));
                Some(Res::One(it.add32(x, y)))
            }
            "s_add_i32" | "s_add_co_i32" | "v_add_nc_u32" | "v_add_u32" => {
                let (x, y) = (one(it, s0), one(it, s1));
                Some(Res::One(it.add32(x, y)))
            }
            "s_sub_u32" | "s_sub_co_u32" | "s_sub_i32" | "s_sub_co_i32" | "v_sub_nc_u32"
            | "v_sub_u32" => {
                let (x, y) = (one(it, s0), one(it, s1));
                Some(Res::One(it.sub32(x, y)))
            }
            "v_subrev_nc_u32" | "v_subrev_u32" => {
                let (x, y) = (one(it, s0), one(it, s1));
                Some(Res::One(it.sub32(y, x)))
            }
            "v_add_co_u32" => {
                let (x, y) = (one(it, s0), one(it, s1));
                let c = if lvl == EXEC_UNKNOWN {
                    Ix::NONE
                } else {
                    it.carry(x, y, lvl)
                };
                match sdst {
                    Some(d) => patch_ix(writes, d.op, &[c]),
                    None => *carry = Some((CarryTo::Vcc, c)),
                }
                Some(Res::One(it.add32(x, y)))
            }
            "s_addc_u32" | "s_add_co_ci_u32" => {
                let (ha, hb) = (one(it, s0), one(it, s1));
                Some(Res::One(hi_sum(it, st.scc.ix.live(), CARRY_SCALAR, ha, hb)))
            }
            "v_add_co_ci_u32" => {
                let cin = part
                    .field("SRC2")
                    .and_then(|f| self.vals(st, f).first().map(|v| v.ix.live()))
                    .unwrap_or_else(|| st.s[VCC_LO].ix.live());
                let (ha, hb) = (one(it, s0), one(it, s1));
                Some(Res::One(if lvl == EXEC_UNKNOWN {
                    Ix::NONE
                } else {
                    hi_sum(it, cin, lvl, ha, hb)
                }))
            }
            "s_lshl_b32" => Some(Res::One(match kon(s1) {
                Some(k) => {
                    let x = one(it, s0);
                    it.shl32(x, k & 31)
                }
                None => Ix::NONE,
            })),
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let n = u32::from(name.as_bytes()[6] - b'0');
                let (x, y) = (one(it, s0), one(it, s1));
                let shifted = it.shl32(x, n);
                Some(Res::One(it.add32(shifted, y)))
            }
            "v_lshlrev_b32" => Some(Res::One(match kon(s0) {
                Some(k) => {
                    let x = one(it, s1);
                    it.shl32(x, k & 31)
                }
                None => Ix::NONE,
            })),
            "s_ashr_i32" => Some(Res::One(if kon(s1).map(|k| k & 31) == Some(31) {
                let x = one(it, s0);
                it.sra31(x)
            } else {
                Ix::NONE
            })),
            "v_ashrrev_i32" => Some(Res::One(if kon(s0).map(|k| k & 31) == Some(31) {
                let x = one(it, s1);
                it.sra31(x)
            } else {
                Ix::NONE
            })),
            "s_add_nc_u64" | "s_add_u64" => {
                let (x, y) = (wide(it, s0), wide(it, s1));
                Some(Res::Two(it.add64(x, y)))
            }
            "s_mul_u64" => {
                let (x, y) = (wide(it, s0), wide(it, s1));
                Some(Res::Two(it.mul64(x, y)))
            }
            "s_lshl_b64" => Some(Res::Two(match kon(s1) {
                Some(k) => {
                    let x = wide(it, s0);
                    it.shl64(x, k & 63)
                }
                None => Ix::NONE,
            })),
            "v_lshlrev_b64" => Some(Res::Two(match kon(s0) {
                Some(k) => {
                    let x = wide(it, s1);
                    it.shl64(x, k & 63)
                }
                None => Ix::NONE,
            })),
            "v_readfirstlane_b32" | "v_mov_b32_dpp" => Some(Res::One(Ix::NONE)),
            _ => None,
        };
        if let (Some(res), Some(d)) = (res, vdst) {
            match res {
                Res::One(x) => patch_ix(writes, d.op, &[x]),
                Res::Two(w) => {
                    let (lo, hi) = (it.lo_of(w), it.hi_of(w));
                    patch_ix(writes, d.op, &[lo, hi]);
                }
            }
        }
    }

    /// `Some(result)` when this is a scalar compare of two known constants.
    fn cmp_konst(&self, st: &State, dec: &Dec) -> Option<bool> {
        let part = &dec.parts[0];
        let rest = part.name.strip_prefix("s_cmp_")?;
        let (rel, ty) = rest.split_once('_')?;
        let srcs: SmallVec<[&Field; 2]> = part.sources().collect();
        if srcs.len() != 2 {
            return None;
        }
        let n = match ty {
            "u32" | "i32" => 1,
            "u64" => 2,
            _ => return None,
        };
        let get = |f: &Field| -> Option<u64> {
            let v = self.vals(st, f);
            let mut x = 0u64;
            for i in 0..n {
                x |= u64::from(v.get(i)?.konst?) << (32 * i);
            }
            Some(x)
        };
        let (x, y) = (get(srcs[0])?, get(srcs[1])?);
        let signed = ty == "i32";
        let (sx, sy) = (i64::from(x as u32 as i32), i64::from(y as u32 as i32));
        Some(match rel {
            "eq" => x == y,
            "lg" => x != y,
            "lt" if signed => sx < sy,
            "le" if signed => sx <= sy,
            "gt" if signed => sx > sy,
            "ge" if signed => sx >= sy,
            "lt" => x < y,
            "le" => x <= y,
            "gt" => x > y,
            "ge" => x >= y,
            _ => return None,
        })
    }

    fn eval_rules<'a>(
        &self,
        st: &State,
        part: &Part<'a>,
        writes: &mut SmallVec<[(&'a Operand, SmallVec<[Val; 4]>); 4]>,
        lanes: &mut SmallVec<[(u16, Option<u8>, Val); 1]>,
    ) {
        let name = part.name;
        let defs: SmallVec<[&Field<'a>; 2]> = part.fields.iter().filter(|f| f.def).collect();
        let uses: SmallVec<[&Field<'a>; 4]> = part.sources().collect();
        if defs.is_empty() {
            return;
        }
        match name {
            "v_writelane_b32" => {
                let (Some(dst), Some(src), Some(sel)) =
                    (part.field("VDST"), part.field("SRC0"), part.field("SRC1"))
                else {
                    return;
                };
                let Operand::Reg(r) = dst.op else { return };
                let val = self.vals(st, src)[0];
                lanes.push((r.base, lane_of(st, sel.op), val));
                return;
            }
            "v_readlane_b32" => {
                let (Some(dst), Some(src), Some(sel)) =
                    (part.field("VDST"), part.field("SRC0"), part.field("SRC1"))
                else {
                    return;
                };
                let val = match (src.op, lane_of(st, sel.op)) {
                    (Operand::Reg(r), Some(l)) if r.kind == Kind::V => {
                        st.lanes.get(&(r.base, l)).copied().unwrap_or_else(|| {
                            st.v[usize::from(r.base)].join(st.disabled_v(r.base))
                        })
                    }
                    (Operand::Reg(r), None) if r.kind == Kind::V => {
                        st.read_v(r.base).join(st.disabled_v(r.base))
                    }
                    _ => self.vals(st, src)[0],
                };
                writes.push((dst.op, smallvec::smallvec![val]));
                return;
            }
            "s_getpc_b64" => {
                writes.push((
                    defs[0].op,
                    smallvec::smallvec![Val::src(PCREL), Val::src(PCREL)],
                ));
                return;
            }
            // gfx12 PC-relative sequence: `s_getpc_b64; s_sext_i32_i16 hi, hi` re-canonicalises the 48-bit PC's
            // high half. Only a value that is purely PCREL (no kernarg roots, no taint) keeps it.
            "s_sext_i32_i16" => {
                if let Some(src) = part.field("SSRC0") {
                    let v = self.vals(st, src)[0];
                    if v.src == PCREL && v.roots.is_empty() && v.taint.is_empty() {
                        writes.push((defs[0].op, smallvec::smallvec![Val::src(PCREL)]));
                        return;
                    }
                }
            }
            _ => {}
        }
        if is_mov(name) {
            if let Some(src) = part
                .first_use()
                .or_else(|| part.fields.iter().find(|f| !f.def))
            {
                writes.push((defs[0].op, self.vals(st, src)));
            }
            return;
        }
        if is_select(name) {
            let cands: SmallVec<[SmallVec<[Val; 4]>; 2]> = uses
                .iter()
                .filter(|f| f.name != "SRC2")
                .map(|f| self.vals(st, f))
                .collect();
            let width = cands.iter().map(SmallVec::len).max().unwrap_or(1);
            let out = (0..width)
                .map(|i| {
                    cands
                        .iter()
                        .map(|c| c[i.min(c.len() - 1)])
                        .reduce(Val::join)
                        .unwrap_or_default()
                })
                .collect();
            writes.push((defs[0].op, out));
            return;
        }
        if name.starts_with("s_cmov_") {
            let (Some(dst), Some(src)) = (part.field("SDST"), part.field("SSRC0")) else {
                return;
            };
            let old = self.vals(st, dst);
            let new = self.vals(st, src);
            let out = old
                .iter()
                .enumerate()
                .map(|(i, a)| a.join(new[i.min(new.len() - 1)]))
                .collect();
            writes.push((dst.op, out));
            return;
        }
        let all_uses: SmallVec<[Val; 8]> = uses.iter().flat_map(|f| self.vals(st, f)).collect();
        if is_mad64(name) {
            let taint = Val::kill(all_uses.iter());
            let c = part
                .field("SRC2")
                .map(|f| self.vals(st, f))
                .unwrap_or_default();
            let none = Val::default();
            for d in &defs {
                if d.name == "SDST" {
                    writes.push((d.op, smallvec::smallvec![taint]));
                    continue;
                }
                let lo = Val::mix([&taint, c.first().unwrap_or(&none)]);
                let hi = Val::mix([&taint, c.get(1).unwrap_or(&none)]);
                writes.push((d.op, smallvec::smallvec![lo, hi]));
            }
            return;
        }
        if let Some(keep) = keep_field(name) {
            let taint = Val::kill(all_uses.iter());
            let kept = part
                .field(keep)
                .map(|f| self.vals(st, f))
                .unwrap_or_default();
            let out = Val::mix(std::iter::once(&taint).chain(kept.iter()));
            for d in &defs {
                writes.push((d.op, smallvec::smallvec![out]));
            }
            return;
        }
        if is_kill(name) {
            let mut out = Val::kill(all_uses.iter());
            if let [a, b] = all_uses.as_slice() {
                out.konst = match (a.konst, b.konst) {
                    (Some(x), Some(y)) if name.starts_with("s_lshl_b32") => Some(x.wrapping_shl(y)),
                    (Some(x), Some(y)) if name.starts_with("s_mul_i32") => Some(x.wrapping_mul(y)),
                    _ => None,
                };
            }
            for d in &defs {
                writes.push((d.op, smallvec::smallvec![out]));
            }
            return;
        }
        // Provenance-keeping arithmetic, dword by dword when widths agree.
        let use_vals: SmallVec<[SmallVec<[Val; 4]>; 4]> =
            uses.iter().map(|f| self.vals(st, f)).collect();
        let has_vdst = defs.iter().any(|f| f.name == "VDST");
        let carry_in = name.starts_with("s_addc_u32")
            || name.starts_with("s_add_co_ci_u32")
            || name.starts_with("v_add_co_ci_u32");
        for d in &defs {
            if d.name == "SDST" && has_vdst {
                // Carry-out of a VOP3b add/sub: a lane mask, never a pointer.
                writes.push((d.op, smallvec::smallvec![Val::kill(all_uses.iter())]));
                continue;
            }
            let width = match d.op {
                Operand::Reg(r) => usize::from(r.len),
                _ => usize::from((d.bits / 32).max(1)),
            };
            let per_dword = width > 1 && use_vals.iter().all(|v| v.len() == width);
            let mut out: SmallVec<[Val; 4]> = SmallVec::new();
            for i in 0..width {
                let lane: SmallVec<[&Val; 4]> = if per_dword {
                    use_vals.iter().map(|v| &v[i]).collect()
                } else {
                    all_uses.iter().collect()
                };
                let mut v = Val::mix(lane.iter().copied());
                if is_add(name) && lane.len() == 2 {
                    let (a, b) = (lane[0], lane[1]);
                    if i == 0 {
                        v.konst = a.konst.zip(b.konst).map(|(x, y)| x.wrapping_add(y));
                        v.koff = match (a.koff, b.koff) {
                            (Some(k), None) => b.konst.map(|c| k + i64::from(c as i32)),
                            (None, Some(k)) => a.konst.map(|c| k + i64::from(c as i32)),
                            _ => None,
                        };
                    } else {
                        v.koff = a.koff.xor(b.koff);
                    }
                } else if carry_in {
                    v.koff = lane.iter().filter_map(|x| x.koff).next();
                } else if let [a, b] = lane.as_slice() {
                    v.konst = match (a.konst, b.konst) {
                        (Some(x), Some(y)) if name.starts_with("s_and_b32") => Some(x & y),
                        (Some(x), Some(y)) if name.starts_with("s_or_b32") => Some(x | y),
                        (Some(x), Some(y)) if name.starts_with("s_sub") => Some(x.wrapping_sub(y)),
                        _ => None,
                    };
                }
                out.push(v);
            }
            writes.push((d.op, out));
        }
    }

    /// A4 sinks that do not depend on memory attribution.
    fn record_sinks(&self, st: &State, dec: &Dec, rec: &mut Rec, cyclic: bool) {
        for part in &dec.parts {
            let name = part.name;
            let uses: SmallVec<[SmallVec<[Val; 4]>; 4]> = part
                .fields
                .iter()
                .filter(|f| f.used && !is_mask(name, f.name))
                .map(|f| self.vals(st, f))
                .collect();
            if name.starts_with("s_cmp")
                || name.starts_with("v_cmp")
                || name.starts_with("s_bitcmp")
            {
                let bit = if cyclic {
                    Sinks::LOOP_BOUND
                } else {
                    Sinks::GUARD
                };
                for v in uses.iter().flatten() {
                    rec.sink(v.taint, v.tf, bit);
                }
                continue;
            }
            if dec.inst.effects.mem.is_some() {
                continue;
            }
            let all = Val::mix(uses.iter().flatten());
            if all.tf & (WI | WG) != 0 {
                let roots = uses
                    .iter()
                    .flatten()
                    .fold(DwSet::default(), |acc, v| acc.union(v.roots));
                rec.sink(all.taint.minus(roots), all.tf, Sinks::GRID_TERM);
            }
        }
    }

    fn memory(&self, st: &mut State, dec: &Dec, mut rec: Option<&mut Rec>) {
        let inst = dec.inst;
        let part = &dec.parts[0];
        let name = part.name;
        let op = mem_op(name);
        let cache = CacheBits {
            th: inst.mods.cpol.th,
            scope: inst.mods.cpol.scope,
            glc: inst.mods.cpol.glc,
            slc: inst.mods.cpol.slc,
            dlc: inst.mods.cpol.dlc,
            nv: inst.mods.cpol.nv,
        };
        let dst = part.fields.iter().find(|f| f.def);
        let data_fields: SmallVec<[&Field; 2]> = part
            .fields
            .iter()
            .filter(|f| {
                !f.def
                    && matches!(
                        f.name,
                        "VSRC" | "VDATA" | "DATA" | "DATA0" | "DATA1" | "SDATA"
                    )
                    && f.used
            })
            .collect();
        match inst.form {
            Form::Ds => {
                let passthrough = name.starts_with("ds_swizzle")
                    || name.starts_with("ds_bpermute")
                    || name.starts_with("ds_permute");
                if let Some(rec) = rec.as_deref_mut() {
                    if let Some(addr) = part.field("ADDR") {
                        for v in self.vals(st, addr) {
                            rec.sink(v.taint, v.tf, Sinks::LDS_ADDRESS);
                        }
                        self.sealed_flow(st, addr, dec, rec, "sealed-table-value-in-address");
                    }
                    if !passthrough {
                        for f in &data_fields {
                            for v in self.vals(st, f) {
                                rec.sink(v.taint, v.tf, Sinks::STORED);
                            }
                            self.sealed_flow(st, f, dec, rec, "sealed-table-value-stored");
                        }
                    }
                }
                if let Some(d) = dst {
                    let width = match d.op {
                        Operand::Reg(r) => usize::from(r.len),
                        _ => 1,
                    };
                    let val = if passthrough {
                        Val::mix(
                            data_fields
                                .iter()
                                .flat_map(|f| self.vals(st, f))
                                .collect::<SmallVec<[Val; 4]>>()
                                .iter(),
                        )
                    } else {
                        Val::loaded()
                    };
                    let vals: SmallVec<[Val; 4]> = std::iter::repeat(val).take(width).collect();
                    self.write(st, d.op, &vals);
                }
                return;
            }
            Form::Smem => {
                if op != MemOp::Load {
                    if let Some(d) = dst {
                        self.write(st, d.op, &[Val::flags(MEM)]);
                    }
                    return;
                }
                let Some(sbase) = part.field("SBASE") else {
                    return;
                };
                let base = self.vals(st, sbase);
                let width = dst.map_or(1, |d| match d.op {
                    Operand::Reg(r) => usize::from(r.len),
                    _ => 1,
                });
                let mut offset: Option<i64> = Some(0);
                let mut offset_vals: SmallVec<[Val; 2]> = SmallVec::new();
                for f in part.fields.iter().filter(|f| f.name == "SOFFSET") {
                    match f.op {
                        Operand::Imm(ImmField::SmemOffset(n)) => {
                            offset = offset.map(|o| o + i64::from(*n))
                        }
                        Operand::Special(Special::Null) => {}
                        other => {
                            let v = self.read(st, other, 32)[0];
                            offset_vals.push(v);
                            offset = match (offset, v.konst) {
                                (Some(o), Some(c)) => Some(o + i64::from(c)),
                                _ => None,
                            };
                        }
                    }
                }
                for e in &dec.extras {
                    if let Operand::Imm(ImmField::SmemDisplacement(n) | ImmField::SmemOffset(n)) =
                        **e
                    {
                        offset = offset.map(|o| o + i64::from(n));
                    }
                }
                let buffer = name.starts_with("s_buffer_load");
                let (lo, hi) = if buffer {
                    (
                        base.first().copied().unwrap_or_default(),
                        base.get(1).copied().unwrap_or_default(),
                    )
                } else {
                    (base[0], base.get(1).copied().unwrap_or_default())
                };
                let mut vals: SmallVec<[Val; 8]> =
                    std::iter::repeat(Val::loaded()).take(width).collect();
                let resolved = resolve_base(&lo, &hi);
                self.probe_hit(dec.id, &lo, &hi);
                if let Base::Side(side) = resolved {
                    if side == KSEG && !buffer {
                        match (lo.koff, offset) {
                            (Some(k), Some(o)) => {
                                let start = k + o;
                                for (i, v) in vals.iter_mut().enumerate() {
                                    let byte = start + 4 * i as i64;
                                    if byte < 0 {
                                        *v = Val::any_karg();
                                        continue;
                                    }
                                    let dw = (byte / 4) as u32;
                                    *v = if byte % 4 == 0 {
                                        let (word, used) = self.karg_word(dw);
                                        if let (Some(gi), Some(rec)) = (used, rec.as_deref_mut()) {
                                            rec.guards_used.insert(gi);
                                        }
                                        word
                                    } else {
                                        Val::karg(dw).join(Val::karg(dw + 1))
                                    };
                                    if let Some(rec) = rec.as_deref_mut() {
                                        for d in [dw, dw + u32::from(byte % 4 != 0)] {
                                            if d < MAX_DW {
                                                rec.loaded = rec.loaded.union(DwSet::one(d));
                                            } else {
                                                rec.loaded_ovf = true;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {
                                for v in vals.iter_mut() {
                                    *v = Val::any_karg();
                                }
                                if let Some(rec) = rec.as_deref_mut() {
                                    rec.dynamic = true;
                                }
                            }
                        }
                        if let Some(d) = dst {
                            self.write(st, d.op, &vals);
                        }
                        return;
                    }
                }
                let addr = if buffer {
                    Ix::NONE
                } else {
                    self.smem_addr(&lo, &hi, offset)
                };
                if width == 1 && !buffer && name == "s_load_b32" {
                    vals[0].ix = self.index_load(
                        dec,
                        addr,
                        &resolved,
                        AccessClass::SmemLoad,
                        MemPath::Smem,
                        cache,
                        rec.as_deref_mut(),
                    );
                }
                if self.sealed.is_some() && !buffer && !self.pos_cyclic[dec.pos] {
                    if width == 1 && name == "s_load_b32" {
                        // The identity of the value this load produces: the
                        // index leaf of a sealed-table entry address and of
                        // the compare that guards it.
                        vals[0].ix = self.ix.borrow_mut().load(dec.id);
                    }
                }
                if self.sealed.is_some() && !buffer && width == 2 && name == "s_load_b64" {
                    if let Some(v) = self.entry_load(
                        dec,
                        addr,
                        &resolved,
                        AccessClass::SmemLoad,
                        MemPath::Smem,
                        cache,
                        rec.as_deref_mut(),
                    ) {
                        vals[0] = v[0];
                        vals[1] = v[1];
                    }
                }
                if let Some(rec) = rec.as_deref_mut() {
                    let path = if buffer {
                        MemPath::SmemBuffer
                    } else {
                        MemPath::Smem
                    };
                    self.attribute(
                        rec,
                        dec,
                        resolved,
                        &lo,
                        &hi,
                        &offset_vals,
                        AccessClass::SmemLoad,
                        path,
                        cache,
                        addr,
                        access_width(name),
                    );
                }
                if let Some(d) = dst {
                    self.write(st, d.op, &vals);
                }
                return;
            }
            Form::Vmem(VmemForm::Image) | Form::Export => {
                if let Some(rec) = rec.as_deref_mut() {
                    if inst.form != Form::Export {
                        rec.fail(self.arch, dec, "unmodeled-image");
                    }
                }
                if let Some(d) = dst {
                    let w = match d.op {
                        Operand::Reg(r) => usize::from(r.len),
                        _ => 1,
                    };
                    let vals: SmallVec<[Val; 4]> =
                        std::iter::repeat(Val::loaded()).take(w).collect();
                    self.write(st, d.op, &vals);
                }
                return;
            }
            Form::Vmem(form) => {
                let loaded_width = dst.map_or(0, |d| match d.op {
                    Operand::Reg(r) => usize::from(r.len),
                    _ => 1,
                });
                let result: SmallVec<[Val; 4]> = std::iter::repeat(Val::loaded())
                    .take(loaded_width.max(1))
                    .collect();
                if op == MemOp::Other {
                    if let Some(d) = dst {
                        self.write(st, d.op, &result);
                    }
                    return;
                }
                if form == VmemForm::Scratch {
                    if let Some(rec) = rec.as_deref_mut() {
                        rec.side.scratch += 1;
                        for f in &data_fields {
                            for v in self.vals(st, f) {
                                rec.sink(v.taint, v.tf, Sinks::STORED);
                            }
                            self.sealed_flow(st, f, dec, rec, "sealed-table-value-stored");
                        }
                    }
                    if let Some(d) = dst {
                        self.write(st, d.op, &result);
                    }
                    return;
                }
                let (lo, hi, offsets) = match form {
                    VmemForm::Buffer => {
                        let rsrc = part
                            .field("RSRC")
                            .map(|f| self.vals(st, f))
                            .unwrap_or_default();
                        let mut offs: SmallVec<[Val; 4]> = SmallVec::new();
                        for n in ["VADDR", "SOFFSET"] {
                            if let Some(f) = part.field(n) {
                                offs.extend(self.vals(st, f));
                            }
                        }
                        if let Some(v) = rsrc.get(2) {
                            offs.push(*v);
                        }
                        (
                            rsrc.first().copied().unwrap_or_default(),
                            rsrc.get(1).copied().unwrap_or_default(),
                            offs,
                        )
                    }
                    _ => {
                        let saddr = part
                            .field("SADDR")
                            .filter(|f| matches!(f.op, Operand::Reg(_)));
                        let vaddr = part.field("VADDR").or_else(|| part.field("ADDR"));
                        match saddr {
                            Some(s) => {
                                let base = self.vals(st, s);
                                let offs = vaddr.map(|f| self.vals(st, f)).unwrap_or_default();
                                (base[0], base.get(1).copied().unwrap_or_default(), offs)
                            }
                            None => {
                                let base = vaddr.map(|f| self.vals(st, f)).unwrap_or_default();
                                (
                                    base.first().copied().unwrap_or_default(),
                                    base.get(1).copied().unwrap_or_default(),
                                    SmallVec::new(),
                                )
                            }
                        }
                    }
                };
                let path = match form {
                    VmemForm::Buffer => MemPath::Buffer,
                    VmemForm::Flat => MemPath::Flat,
                    _ => MemPath::Global,
                };
                let class = match op {
                    MemOp::Load => AccessClass::VmemLoad,
                    MemOp::Store => AccessClass::VmemStore,
                    MemOp::Atomic { commutative } => AccessClass::Atomic { commutative },
                    MemOp::LdsDma => AccessClass::LdsDma,
                    MemOp::Other => unreachable!(),
                };
                let addr = if form == VmemForm::Global {
                    self.vmem_addr(st, dec)
                } else {
                    Ix::NONE
                };
                let resolved = resolve_base(&lo, &hi);
                self.probe_hit(dec.id, &lo, &hi);
                let mut out = result;
                if let Base::Side(KSEG) = resolved {
                    if matches!(op, MemOp::Load) {
                        out = std::iter::repeat(Val::any_karg())
                            .take(loaded_width.max(1))
                            .collect();
                    }
                }
                if form == VmemForm::Global
                    && matches!(op, MemOp::Load)
                    && name == "global_load_b32"
                    && loaded_width == 1
                {
                    out[0].ix = self.index_load(
                        dec,
                        addr,
                        &resolved,
                        class,
                        path,
                        cache,
                        rec.as_deref_mut(),
                    );
                }
                if self.sealed.is_some()
                    && form == VmemForm::Global
                    && matches!(op, MemOp::Load)
                    && name == "global_load_b64"
                    && loaded_width == 2
                {
                    if let Some(v) = self.entry_load(
                        dec,
                        addr,
                        &resolved,
                        class,
                        path,
                        cache,
                        rec.as_deref_mut(),
                    ) {
                        out[0] = v[0];
                        out[1] = v[1];
                    }
                }
                if let Some(rec) = rec.as_deref_mut() {
                    for f in &data_fields {
                        for v in self.vals(st, f) {
                            rec.sink(v.taint, v.tf, Sinks::STORED);
                        }
                        self.sealed_flow(st, f, dec, rec, "sealed-table-value-stored");
                    }
                    self.attribute(
                        rec,
                        dec,
                        resolved,
                        &lo,
                        &hi,
                        &offsets,
                        class,
                        path,
                        cache,
                        addr,
                        access_width(name),
                    );
                }
                if let Some(d) = dst {
                    self.write(st, d.op, &out);
                }
            }
            _ => {
                if let Some(d) = dst {
                    self.write(st, d.op, &[Val::flags(MEM)]);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attribute(
        &self,
        rec: &mut Rec,
        dec: &Dec,
        resolved: Base,
        lo: &Val,
        hi: &Val,
        offsets: &[Val],
        class: AccessClass,
        path: MemPath,
        cache: CacheBits,
        addr: Ix,
        width: Option<u8>,
    ) {
        let writes = class.mode().writes();
        let offset_taint = offsets.iter().fold((DwSet::default(), 0u8), |(t, f), v| {
            (t.union(v.taint), f | v.tf)
        });
        let mut resolved = resolved;
        if self.guard.is_some() {
            // §A9.3 rules 3-5. The legacy lattice drops `OPAQUE` through
            // shifts, multiplies and sign extensions (its named assumption),
            // so an index-dependent address can look ordinarily resolved.
            // Under a guard every address that depends on loaded data - a
            // memory-derived base, or any address carrying loaded-data
            // influence - needs the complete witness, and the witness base
            // must be the ordinary resolution's only candidate. Every other
            // unresolved base keeps its refusal.
            let mem_data = (lo.tf | hi.tf | offset_taint.1) & MEM != 0;
            let memory_derived = matches!(resolved, Base::Unresolved("base-memory-derived"));
            if memory_derived || (mem_data && matches!(resolved, Base::Args { .. })) {
                let certified = self
                    .indexed_access(dec, addr, width, class, path, cache)
                    .filter(|p| match &resolved {
                        Base::Args { hi, .. } => *hi == DwSet::one(p.base_arg_offset / 4 + 1),
                        _ => true,
                    });
                match certified {
                    Some(pending) => {
                        let base_dw = pending.base_arg_offset / 4;
                        rec.pending.push(pending);
                        resolved = Base::Args {
                            hi: DwSet::one(base_dw + 1),
                            lo: DwSet::one(base_dw),
                        };
                    }
                    None => resolved = Base::Unresolved("base-memory-derived"),
                }
            }
        }
        if self.sealed.is_some() {
            let infl_addr = (lo.tf | hi.tf | offset_taint.1) & TBL != 0;
            match &resolved {
                Base::Table(mask) | Base::Both { mask, .. } => {
                    if offset_taint.1 & TBL != 0 {
                        rec.fail(self.arch, dec, "sealed-table-value-in-offset");
                    } else if !matches!(class, AccessClass::VmemLoad | AccessClass::SmemLoad) {
                        rec.fail(
                            self.arch,
                            dec,
                            if writes {
                                "sealed-table-write"
                            } else {
                                "sealed-table-access-class"
                            },
                        );
                    } else if let Some(width) = width {
                        rec.sealed_accesses.push(TableAccessRec {
                            inst: dec.id,
                            pc: dec.pos as u32,
                            mask: *mask,
                            class,
                            path,
                            cache,
                            width,
                        });
                    } else {
                        rec.fail(self.arch, dec, "sealed-table-access-width");
                    }
                }
                Base::Unresolved(_) => {}
                _ if infl_addr => rec.fail(self.arch, dec, "sealed-table-value-in-address"),
                _ => {}
            }
        }
        if let Base::Both { hi, lo, .. } = resolved {
            resolved = Base::Args { hi, lo };
        }
        match resolved {
            Base::Table(_) | Base::Both { .. } => {
                rec.sink(
                    lo.taint.union(hi.taint).union(offset_taint.0),
                    lo.tf | hi.tf | offset_taint.1,
                    Sinks::ADDRESS_OFFSET,
                );
            }
            Base::Unresolved(rule) => {
                rec.fail(self.arch, dec, rule);
                rec.sink(
                    lo.taint.union(hi.taint).union(offset_taint.0),
                    lo.tf | hi.tf | offset_taint.1,
                    Sinks::ADDRESS_OFFSET,
                );
            }
            Base::Side(side) => {
                match side {
                    KSEG => {
                        if writes {
                            rec.fail(self.arch, dec, "kernarg-write")
                        } else {
                            rec.side.kernarg_segment += 1;
                            rec.dynamic = true;
                        }
                    }
                    PCREL => {
                        if writes {
                            rec.fail(self.arch, dec, "module-global-write")
                        } else {
                            rec.side.module_global += 1
                        }
                    }
                    PACKET => {
                        if writes {
                            rec.fail(self.arch, dec, "abi-packet-write")
                        } else {
                            rec.side.abi_packet += 1
                        }
                    }
                    APERTURE => {
                        if path == MemPath::Flat {
                            rec.side.local_flat += 1
                        } else {
                            rec.fail(self.arch, dec, "aperture-non-flat")
                        }
                    }
                    SCRATCH => rec.side.scratch += 1,
                    _ => rec.fail(self.arch, dec, "base-mixed"),
                }
                rec.sink(
                    lo.taint.union(hi.taint).union(offset_taint.0),
                    lo.tf | hi.tf | offset_taint.1,
                    Sinks::ADDRESS_OFFSET,
                );
            }
            Base::Args {
                hi: hi_roots,
                lo: lo_roots,
            } => {
                let mut declared: SmallVec<[usize; 2]> = SmallVec::new();
                let mut synthetic: SmallVec<[u32; 2]> = SmallVec::new();
                for dw in hi_roots.iter() {
                    match self.layout.arg_of(dw) {
                        Some(i) => {
                            if !declared.contains(&i) {
                                declared.push(i)
                            }
                        }
                        None => {
                            let slot = (dw * 4) & !7;
                            if !synthetic.contains(&slot) {
                                synthetic.push(slot)
                            }
                        }
                    }
                }
                // A pointer argument among the high-dword candidates rules out
                // by-value candidates (they carry no allocation).
                if declared.iter().any(|&i| self.layout.is_pointer(i)) {
                    declared.retain(|&mut i| self.layout.is_pointer(i));
                    synthetic.clear();
                }
                for dw in lo_roots.iter() {
                    if let Some(i) = self.layout.arg_of(dw) {
                        if self.layout.is_pointer(i) && !declared.contains(&i) {
                            declared.push(i);
                        }
                    }
                }
                let mut targets: SmallVec<[Target; 4]> =
                    declared.iter().map(|&i| Target::Declared(i)).collect();
                targets.extend(synthetic.iter().map(|&s| Target::Synthetic(s)));
                let shared = targets.len() > 1;
                let mut base_dws = DwSet::default();
                for t in &targets {
                    let (off, size) = match *t {
                        Target::Declared(i) => {
                            (self.layout.args[i].offset, self.layout.args[i].size.max(4))
                        }
                        Target::Synthetic(s) => (s, 8),
                    };
                    for dw in off / 4..(off + size).div_ceil(4) {
                        if dw < MAX_DW {
                            base_dws = base_dws.union(DwSet::one(dw));
                        }
                    }
                    rec.accesses.entry(*t).or_default().push(ArgAccess {
                        inst: dec.id,
                        class,
                        path,
                        cache,
                        shared,
                    });
                }
                rec.sink(base_dws, 0, Sinks::ADDRESS_BASE);
                rec.sink(
                    lo.taint
                        .union(hi.taint)
                        .union(offset_taint.0)
                        .minus(base_dws),
                    lo.tf | hi.tf | offset_taint.1,
                    Sinks::ADDRESS_OFFSET,
                );
            }
        }
    }

    /// A kernarg dword loaded from the segment: the legacy value plus its
    /// exact witness. Under a guard the dword is the guarded constant (a null
    /// pointer guard is a constant with no provenance, never an allocation).
    fn karg_word(&self, dw: u32) -> (Val, Option<usize>) {
        let mut v = Val::karg(dw);
        if dw >= MAX_DW {
            return (v, None);
        }
        let mut it = self.ix.borrow_mut();
        if let Some(&(word, gi)) = self.guard.as_ref().and_then(|g| g.consts.get(&(dw * 4))) {
            v.roots = DwSet::default();
            v.konst = Some(word);
            v.ix = it.c32(word);
            return (v, Some(gi));
        }
        v.ix = it.k32(dw * 4);
        (v, None)
    }

    fn karg_kind(&self, offset: u32) -> KargKind {
        match self.layout.arg_of(offset / 4) {
            Some(i) if self.layout.is_pointer(i) => KargKind::Pointer {
                arg_offset: self.layout.args[i].offset,
            },
            Some(_) => KargKind::Scalar,
            None => KargKind::Other,
        }
    }

    /// `sbase + offset` of an SMEM access as a 64-bit witness.
    fn smem_addr(&self, lo: &Val, hi: &Val, offset: Option<i64>) -> Ix {
        let Some(offset) = offset else {
            return Ix::NONE;
        };
        let mut it = self.ix.borrow_mut();
        let (l, h) = (ixof(&mut it, *lo), ixof(&mut it, *hi));
        let base = it.as64(l, h);
        let imm = it.imm64(offset);
        it.add64(base, imm)
    }

    /// The 64-bit address witness of a global VMEM access: `saddr + zext(vaddr)`
    /// or the `vaddr` pair, plus the instruction immediate.
    fn vmem_addr(&self, st: &State, dec: &Dec) -> Ix {
        let part = &dec.parts[0];
        let saddr = part
            .field("SADDR")
            .filter(|f| matches!(f.op, Operand::Reg(_)));
        let vaddr = part.field("VADDR").or_else(|| part.field("ADDR"));
        let mut imm = 0i64;
        for e in &dec.extras {
            if let Operand::Imm(ImmField::VmemOffset(n)) = **e {
                imm += i64::from(n);
            }
        }
        let mut it = self.ix.borrow_mut();
        let it = &mut *it;
        let pair = |it: &mut Interner, v: &[Val]| -> Ix {
            if v.len() < 2 {
                return Ix::NONE;
            }
            let (l, h) = (ixof(it, v[0]), ixof(it, v[1]));
            it.as64(l, h)
        };
        let base = match (saddr, vaddr) {
            (Some(s), v) => {
                let b = pair(it, &self.vals(st, s));
                match v.map(|f| f.op) {
                    Some(Operand::Reg(r)) if r.kind == Kind::V && r.len == 1 => {
                        let x = ixof(it, self.vals(st, v.expect("vaddr"))[0]);
                        let z = it.zext(x);
                        it.add64(b, z)
                    }
                    Some(Operand::Special(Special::Null)) | None => b,
                    _ => Ix::NONE,
                }
            }
            (None, Some(f)) => match f.op {
                Operand::Reg(r) if r.kind == Kind::V && r.len == 2 => pair(it, &self.vals(st, f)),
                _ => Ix::NONE,
            },
            (None, None) => Ix::NONE,
        };
        let imm = it.imm64(imm);
        it.add64(base, imm)
    }

    /// §A9.3 rule 1: a qualifying ordinary 32-bit load from a directly rooted
    /// argument is an index leaf. Its own read must resolve through ordinary
    /// attribution and its address must be a complete direct-base/offset
    /// witness with no loaded term (no nested `P[Q[i]]`); loads inside a CFG
    /// cycle never qualify (the leaf names one dynamic instance). Guarded
    /// analysis only.
    #[allow(clippy::too_many_arguments)]
    fn index_load(
        &self,
        dec: &Dec,
        addr: Ix,
        resolved: &Base,
        class: AccessClass,
        path: MemPath,
        cache: CacheBits,
        rec: Option<&mut Rec>,
    ) -> Ix {
        if self.guard.is_none()
            || addr.is_none()
            || self.pos_cyclic[dec.pos]
            || !matches!(resolved, Base::Args { .. })
        {
            return Ix::NONE;
        }
        let mut it = self.ix.borrow_mut();
        let Ok((src_arg_offset, byte_offset)) = certify_source(&it, addr, &|o| self.karg_kind(o))
        else {
            return Ix::NONE;
        };
        if let Some(rec) = rec {
            rec.loads.insert(
                dec.id,
                LoadInfo {
                    pc: dec.pos as u32,
                    src_arg_offset,
                    class,
                    path,
                    cache,
                    byte_offset,
                },
            );
        }
        it.load(dec.id)
    }

    /// §A9.3 rules 3-5: the complete witness of an access whose ordinary base
    /// is memory-derived, when it is `direct pointer argument + integer
    /// displacement` with at least one qualifying index load. Guarded
    /// analysis only.
    fn indexed_access(
        &self,
        dec: &Dec,
        addr: Ix,
        width: Option<u8>,
        class: AccessClass,
        path: MemPath,
        cache: CacheBits,
    ) -> Option<Pending> {
        if self.guard.is_none() || addr.is_none() {
            return None;
        }
        let width_bytes = width?;
        let it = self.ix.borrow();
        let cert = certify(&it, addr, &|o| self.karg_kind(o)).ok()?;
        Some(Pending {
            inst: dec.id,
            pc: dec.pos as u32,
            base_arg_offset: cert.base_arg_offset,
            class,
            path,
            cache,
            width_bytes,
            disp: cert.disp,
            loads: cert.loads,
        })
    }

    // -----------------------------------------------------------------
    // Sealed-table analysis (`analyze_sealed`).
    // -----------------------------------------------------------------

    /// Replay every reachable block of `ins` once with recording on.
    fn record(&self, ins: &[Option<State>], rec: &mut Rec) {
        for (b, block) in self.body.blocks.iter().enumerate() {
            let Some(mut st) = ins[b].clone() else {
                continue;
            };
            for pos in block.range.0..block.range.1 {
                self.step(&mut st, pos, Some(&mut *rec), self.cyclic[b]);
            }
        }
    }

    /// Null-guard replay: remember the table bits the watched access's base
    /// carried.
    fn probe_hit(&self, id: InstId, lo: &Val, hi: &Val) {
        if self.sealed.is_none() {
            return;
        }
        let mut p = self.probe.borrow_mut();
        if p.0 == Some(id) {
            p.1 = hi.tbl | lo.tbl;
        }
    }

    /// Sealed analysis: a value influenced by a sealed-table entry must not
    /// reach `field` (store data, LDS address): the loaded pointer would
    /// escape the read-only use the certificate covers.
    fn sealed_flow(&self, st: &State, field: &Field, dec: &Dec, rec: &mut Rec, rule: &'static str) {
        if self.sealed.is_some() && self.vals(st, field).iter().any(|v| v.tf & TBL != 0) {
            rec.fail(self.arch, dec, rule);
        }
    }

    /// Exact scalar results of the sealed analysis, computed on the pre-state:
    /// the SCC witness of an unsigned compare, constants through scalar logic
    /// (including the annihilators `x & 0` and `x | -1`, so a flag that is
    /// zero on one path stays zero through `s_and_b32 s, flag, exec_lo`), and
    /// `s_cselect` with a decided SCC. Exec operands never have a constant
    /// here. Everything else is left to the ordinary transfer function.
    fn sealed_scalar(&self, st: &State, dec: &Dec) -> SealedFx {
        let mut fx = SealedFx::default();
        if dec.parts.len() != 1 {
            return fx;
        }
        let part = &dec.parts[0];
        let name = norm_name(part.name);
        let srcs: SmallVec<[&Field; 4]> = part.sources().collect();
        if let Some(rest) = name.strip_prefix("s_cmp_") {
            if let (Some((rel, ty)), 2) = (rest.split_once('_'), srcs.len()) {
                let mut it = self.ix.borrow_mut();
                let it = &mut *it;
                // `v == c`, where `v` may hold a flag (`Bit`/`Mask`).
                let eq_flag = |it: &mut Interner, v: Ix, c: Ix| -> Ix {
                    if let (Some((f, t)), Some(k)) = (it.flag_value(v), it.const32(c)) {
                        return if k == t {
                            f
                        } else if k == 0 {
                            it.flag_not(f)
                        } else {
                            Ix::NONE
                        };
                    }
                    it.eq32(v, c)
                };
                let flag = match ty {
                    "u32" | "i32" => {
                        let (vx, vy) = (self.vals(st, srcs[0])[0], self.vals(st, srcs[1])[0]);
                        let (x, y) = (ixof(it, vx), ixof(it, vy));
                        let signed = ty == "i32";
                        let lt = |it: &mut Interner, a: Ix, b: Ix| {
                            if signed {
                                it.lts(a, b)
                            } else {
                                it.ltu(a, b)
                            }
                        };
                        match rel {
                            "eq" | "lg" => {
                                let mut f = eq_flag(it, x, y);
                                if f.is_none() {
                                    f = eq_flag(it, y, x);
                                }
                                if rel == "lg" {
                                    it.flag_not(f)
                                } else {
                                    f
                                }
                            }
                            "lt" => lt(it, x, y),
                            "gt" => lt(it, y, x),
                            "ge" => {
                                let f = lt(it, x, y);
                                it.flag_not(f)
                            }
                            "le" => {
                                let f = lt(it, y, x);
                                it.flag_not(f)
                            }
                            _ => Ix::NONE,
                        }
                    }
                    "u64" if matches!(rel, "eq" | "lg") => {
                        // `w == 0` / `w != 0` of a matched 64-bit pair.
                        let (a, b) = (self.vals(st, srcs[0]), self.vals(st, srcs[1]));
                        let zero = |v: &SmallVec<[Val; 4]>| {
                            v.len() >= 2 && v[0].konst == Some(0) && v[1].konst == Some(0)
                        };
                        let wide = |it: &mut Interner, v: &SmallVec<[Val; 4]>| {
                            if v.len() < 2 {
                                return Ix::NONE;
                            }
                            let (lo, hi) = (ixof(it, v[0]), ixof(it, v[1]));
                            it.as64(lo, hi)
                        };
                        let w = if zero(&b) {
                            wide(it, &a)
                        } else if zero(&a) {
                            wide(it, &b)
                        } else {
                            Ix::NONE
                        };
                        let f = it.zero64(w);
                        if rel == "lg" {
                            it.flag_not(f)
                        } else {
                            f
                        }
                    }
                    _ => Ix::NONE,
                };
                if !flag.is_none() {
                    fx.scc_ix = Some(flag);
                    let mut atom = flag;
                    while let Some(Node::Not(g)) = it.node(atom) {
                        atom = g;
                    }
                    self.atom_pc
                        .borrow_mut()
                        .entry(atom)
                        .or_default()
                        .insert(dec.pos as u32);
                }
            }
            return fx;
        }
        let dst = part.fields.iter().find(|f| f.def && f.name == "SDST");
        let dst_idx = |len: usize| -> Option<usize> {
            match dst?.op {
                Operand::Reg(r) if r.kind == Kind::S && usize::from(r.len) == len => {
                    Some(usize::from(r.base))
                }
                Operand::Special(Special::VccLo) if len == 1 => Some(VCC_LO),
                _ => None,
            }
        };
        let kon = |f: &Field| -> Option<u32> {
            if matches!(
                f.op,
                Operand::Special(Special::Exec | Special::ExecLo | Special::ExecHi)
            ) {
                return None;
            }
            self.vals(st, f).first().and_then(|v| v.konst)
        };
        let is_exec = |f: &Field| matches!(f.op, Operand::Special(Special::Exec | Special::ExecLo));
        // The flag a value holds as an all-ones/zero mask.
        let mask_flag = |f: &Field| -> Option<Ix> {
            let v = self.vals(st, f).first().copied()?;
            let it = self.ix.borrow();
            it.flag_value(v.ix.live())
                .filter(|&(_, t)| t == u32::MAX)
                .map(|(f, _)| f)
        };
        match name.as_str() {
            // A saved copy of a nonzero EXEC is a nonzero value; OR-ing a
            // nonzero value into EXEC leaves EXEC nonzero.
            "s_mov_b32" if st.exec_nz && srcs.len() == 1 && is_exec(srcs[0]) => {
                if let Some(i) = dst_idx(1) {
                    let mut it = self.ix.borrow_mut();
                    let t = it.flag_true();
                    fx.dst_ix = Some((i, it.nz_flag(t)));
                }
            }
            "s_or_b32"
                if srcs.len() == 2
                    && matches!(dst.map(|d| d.op), Some(Operand::Special(Special::ExecLo)))
                    && (is_exec(srcs[0]) || is_exec(srcs[1])) =>
            {
                let other = if is_exec(srcs[0]) { srcs[1] } else { srcs[0] };
                let v = self.vals(st, other).first().copied().unwrap_or_default();
                let it = self.ix.borrow();
                let nz = v.konst.is_some_and(|k| k != 0)
                    || matches!(it.node(v.ix.live()),
                        Some(Node::NzFlag(f)) if matches!(it.node(f), Some(Node::True)));
                fx.exec_nz = nz;
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_and_not1_b32" | "s_or_not1_b32"
                if srcs.len() == 2 =>
            {
                let (a, b) = (kon(srcs[0]), kon(srcs[1]));
                let res = match (name.as_str(), a, b) {
                    ("s_and_b32", Some(x), Some(y)) => Some(x & y),
                    ("s_and_b32", Some(0), _) | ("s_and_b32", _, Some(0)) => Some(0),
                    ("s_or_b32", Some(x), Some(y)) => Some(x | y),
                    ("s_or_b32", Some(u32::MAX), _) | ("s_or_b32", _, Some(u32::MAX)) => {
                        Some(u32::MAX)
                    }
                    ("s_xor_b32", Some(x), Some(y)) => Some(x ^ y),
                    ("s_and_not1_b32", Some(x), Some(y)) => Some(x & !y),
                    ("s_and_not1_b32", Some(0), _) | ("s_and_not1_b32", _, Some(u32::MAX)) => {
                        Some(0)
                    }
                    ("s_or_not1_b32", Some(x), Some(y)) => Some(x | !y),
                    ("s_or_not1_b32", Some(u32::MAX), _) | ("s_or_not1_b32", _, Some(0)) => {
                        Some(u32::MAX)
                    }
                    _ => None,
                };
                if let Some(r) = res {
                    fx.scc_konst = Some(r != 0);
                    if let Some(i) = dst_idx(1) {
                        fx.dst_konst = Some((i, r));
                    }
                }
                // Flags: `exec & mask(F)` is nonzero exactly when `F` holds
                // while EXEC is nonzero; two masks combine bitwise.
                let (ma, mb) = (mask_flag(srcs[0]), mask_flag(srcs[1]));
                let mut it = self.ix.borrow_mut();
                let combined = match (name.as_str(), ma, mb) {
                    ("s_and_b32", Some(f), None) if st.exec_nz && is_exec(srcs[1]) => {
                        Some((f, it.nz_flag(f)))
                    }
                    ("s_and_b32", None, Some(f)) if st.exec_nz && is_exec(srcs[0]) => {
                        Some((f, it.nz_flag(f)))
                    }
                    ("s_and_b32", Some(f), Some(g)) => {
                        let a = it.flag_and(f, g);
                        Some((a, it.mask(a)))
                    }
                    ("s_or_b32", Some(f), Some(g)) => {
                        let o = it.flag_or(f, g);
                        Some((o, it.mask(o)))
                    }
                    _ => None,
                };
                if let Some((flag, value)) = combined {
                    if !flag.is_none() {
                        fx.scc_ix = Some(flag);
                    }
                    if let (Some(i), false) = (dst_idx(1), value.is_none()) {
                        fx.dst_ix = Some((i, value));
                    }
                }
            }
            "s_cselect_b32" | "s_cselect_b64" if srcs.len() == 2 => {
                if let Some(c) = st.scc.konst {
                    let pick = if c != 0 { srcs[0] } else { srcs[1] };
                    let len = if name.ends_with("b64") { 2 } else { 1 };
                    let vals = self.vals(st, pick);
                    if let (Some(i), true) = (dst_idx(len), vals.len() >= len) {
                        fx.dst_vals = Some((i, vals.into_iter().take(len).collect()));
                    }
                }
                // A flag-decided select: `-1/1` when SCC holds, else `0`.
                let scc = st.scc.ix.live();
                let mut it_ref = self.ix.borrow_mut();
                let it = &mut *it_ref;
                if it.is_flag(scc) && name == "s_cselect_b32" {
                    let value = match (kon(srcs[0]), kon(srcs[1])) {
                        (Some(u32::MAX), Some(0)) => it.mask(scc),
                        (Some(1), Some(0)) => it.bit(scc),
                        (Some(0), Some(u32::MAX)) => {
                            let n = it.flag_not(scc);
                            it.mask(n)
                        }
                        (Some(0), Some(1)) => {
                            let n = it.flag_not(scc);
                            it.bit(n)
                        }
                        _ => Ix::NONE,
                    };
                    if let (Some(i), false) = (dst_idx(1), value.is_none()) {
                        fx.dst_ix = Some((i, value));
                    }
                }
            }
            _ => {}
        }
        fx
    }

    /// A qualified entry load's result, or (collecting pass) record the load
    /// as a candidate when its shape is a sealed-table entry. `None` for every
    /// other load.
    #[allow(clippy::too_many_arguments)]
    fn entry_load(
        &self,
        dec: &Dec,
        addr: Ix,
        resolved: &Base,
        class: AccessClass,
        path: MemPath,
        cache: CacheBits,
        rec: Option<&mut Rec>,
    ) -> Option<[Val; 2]> {
        let ctx = self.sealed.as_ref()?;
        if let Some(qualified) = &ctx.qualified {
            let site = qualified.get(&dec.id)?;
            if let Some(rec) = rec {
                rec.sealed_entries.insert(
                    dec.id,
                    EntryRec {
                        pc: dec.pos as u32,
                        class,
                        path,
                        cache,
                        width: 8,
                    },
                );
            }
            let (lo, hi) = {
                let mut it = self.ix.borrow_mut();
                let ent = it.ent(dec.id);
                (it.lo_of(ent), it.hi_of(ent))
            };
            let half = |ix: Ix| Val {
                tbl: site.bit,
                tf: MEM | TBL,
                ix,
                ..Val::default()
            };
            return Some([half(lo), half(hi)]);
        }
        // Collecting pass: only the recording replay registers candidates.
        rec?;
        if addr.is_none() {
            return None;
        }
        let it = self.ix.borrow();
        let shape = table_entry(&it, addr, &|o| self.karg_kind(o)).ok()?;
        if !matches!(resolved, Base::Args { hi, .. } if *hi == DwSet::one(shape.table_arg_offset / 4 + 1))
        {
            return None;
        }
        let (index_inst, index_kernarg) = match it.node(shape.index) {
            Some(Node::Load(i)) => (Some(i), None),
            Some(Node::K32(o)) => (None, Some(o)),
            _ => return None,
        };
        ctx.cands.borrow_mut().insert(
            dec.id,
            SealedCand {
                inst: dec.id,
                pos: dec.pos,
                table_arg_offset: shape.table_arg_offset,
                index: shape.index,
                index_inst,
                index_kernarg,
            },
        );
        None
    }

    fn pos_of(&self, inst: InstId) -> u32 {
        self.body
            .layout
            .iter()
            .position(|&i| i == inst)
            .unwrap_or(0) as u32
    }

    /// The unsigned bound `n` of a compare `index < n` as a certificate bound.
    fn bound_of(&self, n: Ix) -> Option<SealedBound> {
        let it = self.ix.borrow();
        match it.node(n)? {
            Node::C32(c) => Some(SealedBound::Const(c)),
            Node::K32(o) if o % 4 == 0 && self.karg_kind(o) == KargKind::Scalar => {
                Some(SealedBound::Kernarg { offset: o })
            }
            _ => None,
        }
    }

    /// Prove the guard of every candidate entry load (see [`analyze_sealed`]).
    fn qualify(&self, cands: Vec<SealedCand>) -> (BTreeMap<InstId, SealedSite>, Vec<Unresolved>) {
        let mut out: BTreeMap<InstId, SealedSite> = BTreeMap::new();
        let mut rejected = Vec::new();
        for c in cands {
            let detail = self.decoded[c.pos]
                .as_ref()
                .and_then(|d| d.inst.text(self.arch).ok())
                .unwrap_or_default();
            if out.len() >= 8 {
                rejected.push(Unresolved {
                    inst: Some(c.inst),
                    rule: "sealed-too-many-entries",
                    detail,
                });
                continue;
            }
            match self.table_guard(&c) {
                Ok((bound, guard_pcs)) => {
                    let index = match (c.index_inst, c.index_kernarg) {
                        (Some(i), _) => SealedIndex::Load { pc: self.pos_of(i) },
                        (None, Some(offset)) => SealedIndex::Kernarg { offset },
                        (None, None) => unreachable!("a candidate has an index"),
                    };
                    let bit = 1u8 << out.len();
                    out.insert(
                        c.inst,
                        SealedSite {
                            bit,
                            table_arg_offset: c.table_arg_offset,
                            bound,
                            index,
                            guard_pcs,
                        },
                    );
                }
                Err(rule) => rejected.push(Unresolved {
                    inst: Some(c.inst),
                    rule,
                    detail,
                }),
            }
        }
        (out, rejected)
    }

    /// The blocks that can reach the block of layout position `pos` (all
    /// outside every CFG cycle, so their paths are finite), and that block.
    fn walk_setup(&self, pos: usize) -> Result<(usize, Vec<bool>), &'static str> {
        let blocks = &self.body.blocks;
        let site = blocks
            .iter()
            .position(|b| b.range.0 <= pos && pos < b.range.1)
            .ok_or("sealed-site-block")?;
        if self.cyclic[site] {
            return Err("sealed-entry-in-cycle");
        }
        let mut anc = vec![false; blocks.len()];
        anc[site] = true;
        let mut stack = vec![site];
        while let Some(b) = stack.pop() {
            for p in &blocks[b].preds {
                if !anc[p.0] {
                    anc[p.0] = true;
                    stack.push(p.0);
                }
            }
        }
        if !anc[0] {
            return Err("sealed-site-unreachable");
        }
        if (0..blocks.len()).any(|b| anc[b] && self.cyclic[b]) {
            return Err("sealed-guard-cyclic-prefix");
        }
        Ok((site, anc))
    }

    /// The bound `index < bound` is established on every feasible path from
    /// the kernel entry to the entry load, and the compares that do so.
    /// Every block that can reach the load must be outside every CFG cycle, so
    /// the paths are finite; each is replayed with the ordinary transfer
    /// function from the entry state, branch edges the replay decides are
    /// pruned, and a decided compare branch adds the compare facts the edge
    /// implies (through exact flag algebra: masks, `or`/`and`, `exec` known
    /// nonzero). `0 <=s i <s n` counts as `i <u n`.
    fn table_guard(&self, c: &SealedCand) -> Result<(SealedBound, Vec<u32>), &'static str> {
        let (site, anc) = self.walk_setup(c.pos)?;
        let mut walk = GuardWalk {
            site,
            anc,
            paths: 0,
            steps: 0,
            err: None,
            goal: Goal::Bound {
                index: c.index,
                common: None,
            },
        };
        self.walk(0, self.entry.clone(), &mut Vec::new(), &mut walk);
        if let Some(e) = walk.err {
            return Err(e);
        }
        let Goal::Bound { common, .. } = walk.goal else {
            unreachable!()
        };
        let Some(common) = common else {
            return Err("sealed-site-unreachable");
        };
        let (bound, pcs) = common.into_iter().next().ok_or("sealed-index-unguarded")?;
        Ok((bound, pcs.into_iter().collect()))
    }

    /// Every path to the access at `pos` on which its base derives from the
    /// entry load `ents[k].1` (site bit `ents[k].0`) knows that entry is
    /// nonzero. False when anything cannot be shown (a cycle before the
    /// access, an undecodable instruction, the path budget).
    fn access_nonnull(&self, id: InstId, pos: usize, ents: Vec<(u8, Ix)>) -> bool {
        let Ok((site, anc)) = self.walk_setup(pos) else {
            return false;
        };
        let mut walk = GuardWalk {
            site,
            anc,
            paths: 0,
            steps: 0,
            err: None,
            goal: Goal::NonNull {
                id,
                pos,
                ents,
                ok: true,
            },
        };
        self.walk(0, self.entry.clone(), &mut Vec::new(), &mut walk);
        walk.err.is_none() && matches!(walk.goal, Goal::NonNull { ok: true, .. })
    }

    /// Facts the decided edge of a conditional branch adds.
    fn edge_facts(
        &self,
        cond: crate::cfg::Cond,
        is_taken: bool,
        st: &State,
        facts: &mut Vec<(Ix, bool)>,
    ) {
        use crate::cfg::Cond;
        let it = self.ix.borrow();
        match cond {
            Cond::Scc0 | Cond::Scc1 => {
                let f = st.scc.ix.live();
                if it.is_flag(f) {
                    it.assume(f, (cond == Cond::Scc1) == is_taken, facts);
                }
            }
            Cond::Vccz | Cond::Vccnz if self.wave == Wave::Wave32 => {
                if let Some(Node::NzFlag(f)) = it.node(st.s[VCC_LO].ix.live()) {
                    it.assume(f, (cond == Cond::Vccnz) == is_taken, facts);
                }
            }
            _ => {}
        }
    }

    fn walk(&self, b: usize, mut st: State, facts: &mut Vec<(Ix, bool)>, w: &mut GuardWalk) {
        use crate::cfg::{Cond, Terminator};
        if w.err.is_some() {
            return;
        }
        let block = &self.body.blocks[b];
        if b == w.site {
            w.paths += 1;
            if w.paths > GUARD_PATHS {
                w.err = Some("sealed-guard-path-budget");
                return;
            }
            match &mut w.goal {
                Goal::Bound { index, common } => {
                    let it = self.ix.borrow();
                    let zero_half = facts.iter().any(|&(a, pol)| {
                        !pol && matches!(it.node(a),
                            Some(Node::Lts { a: x, b: z })
                                if x == *index && it.const32(z) == Some(0))
                    });
                    let pcs_of = |a: Ix| -> BTreeSet<u32> {
                        self.atom_pc.borrow().get(&a).cloned().unwrap_or_default()
                    };
                    let mut here: BTreeMap<SealedBound, BTreeSet<u32>> = BTreeMap::new();
                    for &(a, pol) in facts.iter().filter(|f| f.1) {
                        let n = match it.node(a) {
                            Some(Node::Ltu { a: x, b: n }) if x == *index => Some(n),
                            Some(Node::Lts { a: x, b: n }) if x == *index && zero_half => Some(n),
                            _ => None,
                        };
                        let Some(bound) = n.and_then(|n| self.bound_of(n)) else {
                            continue;
                        };
                        let pcs = here.entry(bound).or_default();
                        pcs.extend(pcs_of(a));
                        if matches!(it.node(a), Some(Node::Lts { .. })) {
                            for &(z, pol) in facts.iter().filter(|f| !f.1) {
                                if matches!(it.node(z),
                                    Some(Node::Lts { a: x, b: zz })
                                        if x == *index && it.const32(zz) == Some(0))
                                {
                                    pcs.extend(pcs_of(z));
                                    let _ = pol;
                                }
                            }
                        }
                    }
                    *common = Some(match common.take() {
                        None => here,
                        Some(mut prev) => {
                            prev.retain(|k, _| here.contains_key(k));
                            for (k, pcs) in prev.iter_mut() {
                                pcs.extend(here[k].iter().copied());
                            }
                            prev
                        }
                    });
                }
                Goal::NonNull { id, pos, ents, ok } => {
                    *self.probe.borrow_mut() = (Some(*id), 0);
                    for p in block.range.0..=*pos {
                        w.steps += 1;
                        if self.decoded[p].is_none() || w.steps > GUARD_STEPS {
                            w.err = Some("sealed-guard-undecoded");
                            return;
                        }
                        self.step(&mut st, p, None, false);
                    }
                    let tbl = self.probe.borrow().1;
                    let it = self.ix.borrow();
                    for &(bit, ent) in ents.iter().filter(|e| tbl & e.0 != 0) {
                        let known = facts.iter().any(|&(a, pol)| {
                            !pol && matches!(it.node(a), Some(Node::Zero64(w)) if w == ent)
                        });
                        *ok &= known;
                        let _ = bit;
                    }
                }
            }
            return;
        }
        for pos in block.range.0..block.range.1 {
            w.steps += 1;
            if w.steps > GUARD_STEPS {
                w.err = Some("sealed-guard-path-budget");
                return;
            }
            if self.decoded[pos].is_none() {
                w.err = Some("sealed-guard-undecoded");
                return;
            }
            self.step(&mut st, pos, None, false);
        }
        match &block.term {
            Terminator::Branch {
                cond,
                taken,
                fallthrough,
            } => {
                let decided = self.branch_outcome(*cond, &st);
                for (target, is_taken) in [(taken.0, true), (fallthrough.0, false)] {
                    if decided.is_some_and(|d| d != is_taken) || !w.anc[target] {
                        continue;
                    }
                    let mut next = facts.clone();
                    self.edge_facts(*cond, is_taken, &st, &mut next);
                    let mut child = st.clone();
                    if self.wave == Wave::Wave32
                        && matches!(
                            (cond, is_taken),
                            (Cond::Execz, false) | (Cond::Execnz, true)
                        )
                    {
                        child.exec_nz = true;
                    }
                    self.walk(target, child, &mut next, w);
                }
            }
            Terminator::FallThrough | Terminator::Jump(_) => {
                for s in &block.succs {
                    if w.anc[s.0] {
                        self.walk(s.0, st.clone(), facts, w);
                    }
                }
            }
            Terminator::EndPgm | Terminator::Unreachable => {}
        }
    }

    /// The sealed tables of the final (qualified) run: entry loads grouped by
    /// `(table argument, bound)` with every read whose base derives from them.
    fn assemble_sealed(&self, rec: &Rec) -> Vec<SealedTableUse> {
        let Some(qualified) = self.sealed.as_ref().and_then(|s| s.qualified.as_ref()) else {
            return Vec::new();
        };
        let mut groups: BTreeMap<(u32, SealedBound), Vec<(&InstId, &SealedSite)>> = BTreeMap::new();
        for (inst, site) in qualified {
            groups
                .entry((site.table_arg_offset, site.bound))
                .or_default()
                .push((inst, site));
        }
        let mut out = Vec::new();
        for ((table_arg_offset, bound), mut sites) in groups {
            sites.sort_by_key(|(inst, _)| rec.sealed_entries.get(*inst).map_or(0, |e| e.pc));
            let mask = sites.iter().fold(0u8, |m, (_, s)| m | s.bit);
            let mut entries = Vec::new();
            for (inst, site) in &sites {
                let Some(er) = rec.sealed_entries.get(*inst) else {
                    continue;
                };
                entries.push(SealedEntryLoad {
                    inst: **inst,
                    pc: er.pc,
                    class: er.class,
                    path: er.path,
                    cache: er.cache,
                    width_bytes: er.width,
                    index: site.index,
                    guard_pcs: site.guard_pcs.clone(),
                });
            }
            // `null_guarded`: every read through an entry of this table is
            // reachable only with that entry nonzero (proved path by path).
            let ent_of = |inst: &InstId| self.ix.borrow_mut().ent(*inst);
            let mut null_guarded = true;
            let mut accesses = Vec::new();
            for a in rec.sealed_accesses.iter().filter(|a| a.mask & mask != 0) {
                if null_guarded {
                    let ents: Vec<(u8, Ix)> = sites
                        .iter()
                        .filter(|(_, s)| a.mask & s.bit != 0)
                        .map(|(inst, s)| (s.bit, ent_of(inst)))
                        .collect();
                    null_guarded = self.access_nonnull(a.inst, a.pc as usize, ents);
                }
                let mut entry_pcs: Vec<u32> = sites
                    .iter()
                    .filter(|(_, s)| a.mask & s.bit != 0)
                    .filter_map(|(inst, _)| rec.sealed_entries.get(*inst).map(|e| e.pc))
                    .collect();
                entry_pcs.sort_unstable();
                accesses.push(SealedAccess {
                    inst: a.inst,
                    pc: a.pc,
                    class: a.class,
                    path: a.path,
                    cache: a.cache,
                    width_bytes: a.width,
                    entry_pcs,
                });
            }
            accesses.sort_by_key(|a| a.pc);
            out.push(SealedTableUse {
                table_arg_offset,
                bound,
                entry_bytes: 8,
                entries,
                accesses,
                null_guarded,
            });
        }
        out
    }
}

/// The exact scalar results the sealed analysis adds to one instruction's
/// ordinary transfer (see [`Analyzer::sealed_scalar`]).
#[derive(Default)]
struct SealedFx {
    scc_ix: Option<Ix>,
    scc_konst: Option<bool>,
    dst_konst: Option<(usize, u32)>,
    dst_vals: Option<(usize, SmallVec<[Val; 2]>)>,
    dst_ix: Option<(usize, Ix)>,
    /// `s_or_b32 exec_lo, exec_lo, x` with `x` known nonzero.
    exec_nz: bool,
}

/// Path budget of the entry-guard replay.
const GUARD_PATHS: usize = 4096;
const GUARD_STEPS: usize = 400_000;

/// What a path enumeration proves at its target block.
enum Goal {
    /// `index < bound` on every path, at an entry load.
    Bound {
        index: Ix,
        /// Bounds established on every path reached so far, with their
        /// compares.
        common: Option<BTreeMap<SealedBound, BTreeSet<u32>>>,
    },
    /// Every table-derived base at the access is a known-nonzero entry.
    NonNull {
        id: InstId,
        pos: usize,
        ents: Vec<(u8, Ix)>,
        ok: bool,
    },
}

/// State of one path enumeration.
struct GuardWalk {
    site: usize,
    anc: Vec<bool>,
    paths: usize,
    steps: usize,
    err: Option<&'static str>,
    goal: Goal,
}

fn other_is_negative(op: &Operand) -> bool {
    matches!(op, Operand::Inline(InlineConst::Integer(n)) if *n < 0)
}

/// Where an implicit carry-out lane mask is written.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CarryTo {
    Scc,
    Vcc,
}
/// `State::exec_exact` unknown: a VALU carry then names no EXEC level.
const EXEC_UNKNOWN: u8 = 0xfe;

/// Mnemonic without the encoding suffix, VOPD halves as their plain opcode.
fn norm_name(name: &str) -> String {
    let mut n = match name.strip_prefix("v_dual_") {
        Some(rest) => format!("v_{rest}"),
        None => name.to_owned(),
    };
    for suffix in ["_e32", "_e64"] {
        if let Some(stripped) = n.strip_suffix(suffix) {
            n = stripped.to_owned();
            break;
        }
    }
    n
}

/// The witness of a value: its own, or its known constant.
fn ixof(it: &mut Interner, v: Val) -> Ix {
    let ix = v.ix.live();
    if !ix.is_none() {
        ix
    } else if let Some(k) = v.konst {
        it.c32(k)
    } else {
        Ix::NONE
    }
}

/// Replace the witnesses of the value written to `target`, one per dword.
fn patch_ix(writes: &mut [(&Operand, SmallVec<[Val; 4]>)], target: &Operand, ixs: &[Ix]) {
    if let Some((_, vals)) = writes.iter_mut().find(|(op, _)| std::ptr::eq(*op, target)) {
        let base = vals.first().copied().unwrap_or_default();
        let mut out: SmallVec<[Val; 4]> = SmallVec::new();
        for (i, &ix) in ixs.iter().enumerate() {
            let mut v = vals.get(i).copied().unwrap_or(base);
            v.ix = ix;
            out.push(v);
        }
        *vals = out;
    }
}

/// Bytes one access moves per lane, from the mnemonic's size suffix.
fn access_width(name: &str) -> Option<u8> {
    Some(match name.rsplit('_').next()? {
        "b8" | "u8" | "i8" => 1,
        "b16" | "u16" | "i16" => 2,
        "b32" | "u32" | "i32" | "f32" => 4,
        "b64" | "u64" | "i64" | "f64" => 8,
        "b96" => 12,
        "b128" => 16,
        "b256" => 32,
        "b512" => 64,
        _ => return None,
    })
}

fn read_cache_of(accesses: &[ArgAccess]) -> ReadCacheClass {
    accesses.iter().fold(ReadCacheClass::NotRead, |acc, a| {
        acc.join(match a.class {
            AccessClass::SmemLoad => ReadCacheClass::ScalarOnly,
            AccessClass::VmemLoad | AccessClass::LdsDma => ReadCacheClass::VmemOnly,
            AccessClass::VmemStore | AccessClass::Atomic { .. } => ReadCacheClass::NotRead,
        })
    })
}

/// Turn the accesses resolved through the address witness into
/// [`IndexedArgEffect`]s: attach the index-load records their `Load(i)` leaves
/// name and check the entry registers they read are pinned by supplied
/// guards. Anything that cannot be completed is an unresolved access.
fn resolve_pending(
    rec: &Rec,
    supplied: &[Guard],
    unresolved: &mut Vec<Unresolved>,
) -> (Vec<IndexedArgEffect>, bool) {
    let mut out = Vec::new();
    let mut entry_used = false;
    for p in &rec.pending {
        let mut fail = |rule: &'static str, detail: String| {
            unresolved.push(Unresolved {
                inst: Some(p.inst),
                rule,
                detail,
            })
        };
        let mut index_loads = Vec::new();
        let mut complete = true;
        for &inst in &p.loads {
            match rec.loads.get(&inst) {
                Some(l) => index_loads.push(IndexLoad {
                    inst,
                    pc: l.pc,
                    src_arg_offset: l.src_arg_offset,
                    elem: IndexElem::I32,
                    class: l.class,
                    path: l.path,
                    cache: l.cache,
                    byte_offset: l.byte_offset.clone(),
                    ext: Vec::new(),
                }),
                None => {
                    fail("indexed-load-unqualified", format!("pc {}", p.pc));
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let mut ext = BTreeMap::new();
        load_ext(&p.disp, &mut ext);
        for (i, l) in index_loads.iter_mut().enumerate() {
            l.ext = ext.remove(&(i as u16)).unwrap_or_default();
        }
        // Entry registers hold packed ids: the expression is only evaluable
        // when the other axes are pinned by guards.
        let mut regs: Vec<EntryReg> = Vec::new();
        let mut collect = |e: &IdxExpr| {
            e.visit(&mut |n| {
                if let IdxExpr::Entry(r) = n {
                    if !regs.contains(r) {
                        regs.push(*r);
                    }
                }
            })
        };
        collect(&p.disp);
        for l in &index_loads {
            collect(&l.byte_offset);
        }
        for r in regs {
            let needed: &[Guard] = match r {
                EntryReg::WorkgroupIdX => &[],
                EntryReg::WorkgroupIdYZ => &[Guard::GridDim { axis: 2, value: 1 }],
                EntryReg::WorkitemIdPacked => &[
                    Guard::BlockDim { axis: 1, value: 1 },
                    Guard::BlockDim { axis: 2, value: 1 },
                ],
            };
            for g in needed {
                if supplied.contains(g) {
                    entry_used = true;
                } else {
                    fail("indexed-entry-guard", format!("{} needs {g:?}", r.name()));
                    complete = false;
                }
            }
        }
        if !complete {
            continue;
        }
        out.push(IndexedArgEffect {
            inst: p.inst,
            pc: p.pc,
            base_arg_offset: p.base_arg_offset,
            mode: p.class.mode(),
            class: p.class,
            path: p.path,
            cache: p.cache,
            width_bytes: p.width_bytes,
            displacement: p.disp.clone(),
            index_loads,
            guards: Vec::new(),
        });
    }
    (out, entry_used)
}

fn finish(
    mut facts: KernargFacts,
    layout: &Layout,
    rec: Option<Rec>,
    supplied: &[Guard],
    sealed: Vec<SealedTableUse>,
) -> KernargFacts {
    let rec = rec.unwrap_or_else(|| Rec::new(DwSet::default()));
    let (mut indexed, entry_used) = resolve_pending(&rec, supplied, &mut facts.unresolved);
    facts.status = if !facts.unresolved.is_empty() {
        KernelStatus::Unknown
    } else if !indexed.is_empty() || !rec.guards_used.is_empty() || entry_used || !sealed.is_empty()
    {
        KernelStatus::Conditional
    } else {
        KernelStatus::Proven
    };
    if facts.status == KernelStatus::Conditional {
        facts.sealed = sealed;
    }
    if facts.status != KernelStatus::Proven {
        facts.guards = supplied.to_vec();
        for e in &mut indexed {
            e.guards = supplied.to_vec();
        }
    }
    facts.indexed = indexed;
    facts.side = rec.side;
    facts.dynamic_kernarg_reads = rec.dynamic;
    let dword_count = layout.size.div_ceil(4);
    let loaded = |dw: u32| {
        rec.dynamic
            || if dw < MAX_DW {
                rec.loaded.contains(dw)
            } else {
                rec.loaded_ovf
            }
    };
    let sinks_of = |dw: u32| {
        if dw < MAX_DW {
            rec.sinks[dw as usize]
        } else {
            0
        }
    };
    facts.dwords = (0..dword_count)
        .map(|dw| DwordFacts {
            offset: dw * 4,
            loaded: loaded(dw),
            sinks: Sinks(sinks_of(dw)),
        })
        .collect();
    let span = |offset: u32, size: u32| offset / 4..(offset + size.max(1)).div_ceil(4);
    let make = |target: Target, arg: Option<(usize, &Kernarg)>| -> ArgFacts {
        let (offset, size, index, name, kind) = match (target, arg) {
            (_, Some((i, a))) => (
                a.offset,
                a.size,
                Some(i),
                a.name.clone(),
                a.value_kind.clone(),
            ),
            (Target::Synthetic(s), None) => {
                (s, 8, None, format!("synthetic@{s}"), "synthetic".to_owned())
            }
            (Target::Declared(_), None) => unreachable!(),
        };
        let accesses = rec.accesses.get(&target).cloned().unwrap_or_default();
        let dws = span(offset, size);
        let was_loaded = dws.clone().any(loaded);
        let sinks = Sinks(dws.clone().fold(0, |acc, dw| acc | sinks_of(dw)));
        let mode = accesses
            .iter()
            .map(|a| a.class.mode())
            .reduce(AccessMode::join);
        let role = if !accesses.is_empty() {
            ArgRole::Pointer
        } else if was_loaded {
            ArgRole::Integer
        } else {
            ArgRole::NotLoaded
        };
        ArgFacts {
            offset,
            size,
            index,
            name,
            value_kind: kind,
            role,
            mode,
            bound: mode.map(|_| RegionBound::AllocationWide),
            read_cache: read_cache_of(&accesses),
            accesses,
            sinks,
            loaded: was_loaded,
        }
    };
    let mut args: Vec<ArgFacts> = layout
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| make(Target::Declared(i), Some((i, a))))
        .collect();
    for target in rec.accesses.keys() {
        if let Target::Synthetic(_) = target {
            args.push(make(*target, None));
        }
    }
    let mut implicit = ImplicitUse::default();
    for a in &args {
        if !a.is_hidden() || !a.loaded {
            continue;
        }
        implicit.hidden_loaded.push(a.value_kind.clone());
        if HIDDEN_GRID_KINDS.contains(&a.value_kind.as_str()) {
            implicit.hidden_grid.push(a.value_kind.clone());
            if a.sinks.0 != 0 {
                implicit.hidden_grid_live.push(a.value_kind.clone());
            }
        }
    }
    facts.implicit = implicit;
    facts.read_cache = if facts.status == KernelStatus::Unknown {
        ReadCacheClass::Unknown
    } else {
        args.iter()
            .fold(ReadCacheClass::NotRead, |acc, a| acc.join(a.read_cache))
    };
    facts.taint_overflow = rec.taint_overflow;
    facts.args = args;
    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Body;
    use crate::descriptor::{
        KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3,
    };
    use crate::inst::{KernelOrigin, SymbolId};
    use crate::kernarg::IndexedArgEffect;
    use crate::metadata::{HsaKernelMetadata, KernelMeta};

    /// A gfx1201 wave32 HSA kernel from llvm-mc words (kernarg pointer in
    /// s[0:1]) with the given `.args` (name, offset, size, value kind).
    fn kernel(words: &[u32], args: &[(&str, u32, u32, &str)]) -> Kernel {
        let mut body = Body::default();
        let mut at = 0;
        while at < words.len() {
            let (inst, used) = crate::codec::gfx12::decode(&words[at..]).expect("decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            at += used;
        }
        crate::passes::cfg::build_blocks(&mut body, Arch::Gfx1201).expect("blocks");
        let size = args
            .iter()
            .map(|a| a.1 + a.2)
            .max()
            .unwrap_or(0)
            .div_ceil(8)
            * 8;
        let descriptor = KernelDescriptor {
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            kernarg_size: size,
            kernel_code_entry_byte_offset: 0,
            compute_pgm_rsrc3: Rsrc3(0),
            compute_pgm_rsrc1: Rsrc1(0x1f),
            compute_pgm_rsrc2: Rsrc2(0x384),
            kernel_code_properties: KernelCodeProperties(0x408),
            kernarg_preload: KernargPreload(0),
            reserved: [0; 28],
        };
        let parsed = KernelMeta {
            name: "k".into(),
            symbol: "k.kd".into(),
            kernarg_segment_size: size,
            kernarg_segment_align: 8,
            wavefront_size: 32,
            args: args
                .iter()
                .map(|&(name, offset, size, kind)| Kernarg {
                    name: name.into(),
                    offset,
                    size,
                    value_kind: kind.into(),
                    address_space: (kind == "global_buffer").then(|| "global".to_owned()),
                })
                .collect(),
            ..Default::default()
        };
        Kernel {
            symbol: SymbolId("k".into()),
            wave: Wave::Wave32,
            abi: Abi::Hsa {
                descriptor,
                metadata: HsaKernelMetadata {
                    raw_msgpack: Vec::new(),
                    parsed,
                },
            },
            body,
            origin: KernelOrigin::Authored {
                builder_crate: "test".into(),
                version: "0".into(),
                git: "0".into(),
            },
        }
    }
    fn facts(words: &[u32], args: &[(&str, u32, u32, &str)]) -> KernargFacts {
        analyze(&kernel(words, args), Arch::Gfx1201)
    }
    fn mode(f: &KernargFacts, offset: u32) -> Option<AccessMode> {
        f.arg_at(offset).and_then(|a| a.mode)
    }
    fn rules(f: &KernargFacts) -> Vec<&'static str> {
        f.unresolved.iter().map(|u| u.rule).collect()
    }
    const PTRS: &[(&str, u32, u32, &str)] =
        &[("a", 0, 8, "global_buffer"), ("b", 8, 8, "global_buffer")];

    #[test]
    fn pc_relative_base_survives_the_gfx12_high_half_sext() {
        // s_getpc_b64 s[4:5]; s_sext_i32_i16 s5, s5; global_load_b32 v1, v0, s[4:5]
        let f = facts(
            &[
                0xbe844700, 0xbe850f05, 0xee050004, 0x00000001, 0x00000000, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!(f.side.module_global, 1);
    }

    #[test]
    fn kernarg_pointer_does_not_survive_a_high_half_sext() {
        // s_load_b64 s[4:5], s[0:1], 0; s_wait_kmcnt 0; s_sext_i32_i16 s5, s5; global_load_b32 v1, v0, s[4:5]
        let f = facts(
            &[
                0xf4000100, 0xf8000000, 0xbfc70000, 0xbe850f05, 0xee050004, 0x00000001, 0x00000000,
                0xbfb00000,
            ],
            PTRS,
        );
        assert_ne!(f.status, KernelStatus::Proven);
    }

    #[test]
    fn base_saved_and_restored_through_lanes_is_attributed() {
        // s_load_b128 s[4:7], s[0:1]; v_writelane_b32 v40, s6, 0; v_writelane_b32 v40, s7, 1;
        // s_mov_b64 s[6:7], 0; v_readlane_b32 s10, v40, 0; v_readlane_b32 s11, v40, 1;
        // global_load_b32 v1, v0, s[4:5]; global_store_b32 v0, v1, s[10:11]
        let f = facts(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xd7610028, 0x02010006, 0xd7610028, 0x02010207,
                0xbe860180, 0xd760000a, 0x02010128, 0xd760000b, 0x02010328, 0x7e000280, 0xee050004,
                0x00000001, 0x00000000, 0xbfc00000, 0xee06800a, 0x00800000, 0x00000000, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!(
            (mode(&f, 0), mode(&f, 8)),
            (Some(AccessMode::Read), Some(AccessMode::Write))
        );
        assert!(f.args.iter().all(|a| a.accesses.iter().all(|x| !x.shared)));
        assert_eq!(f.arg_at(8).unwrap().role, ArgRole::Pointer);
    }

    #[test]
    fn restore_from_a_lane_never_written_is_unknown() {
        // Same save, but the restore reads lanes 2 and 3: the base has no provenance.
        let f = facts(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xd7610028, 0x02010006, 0xd7610028, 0x02010207,
                0xbe860180, 0xd760000a, 0x02010528, 0xd760000b, 0x02010728, 0x7e000280, 0xee06800a,
                0x00000000, 0x00000000, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(f.status, KernelStatus::Unknown);
        assert_eq!(rules(&f), ["base-no-provenance"]);
        assert_eq!(
            mode(&f, 8),
            None,
            "the clobbered-then-lost pointer is not credited with the store"
        );
    }

    #[test]
    fn cselect_between_two_argument_pointers_reaches_both() {
        // s_cmp_eq_u32 s8, 0; s_cselect_b64 s[2:3], s[4:5], s[6:7]; load via s[2:3]; store via s[12:13].
        let args = &[
            ("a", 0, 8, "global_buffer"),
            ("b", 8, 8, "global_buffer"),
            ("flag", 16, 4, "by_value"),
            ("out", 24, 8, "global_buffer"),
        ];
        let f = facts(
            &[
                0xf4004100, 0xf8000000, 0xf4000200, 0xf8000010, 0xf4002300, 0xf8000018, 0xbfc70000,
                0xbf068008, 0x98820604, 0x7e000280, 0xee050002, 0x00000001, 0x00000000, 0xbfc00000,
                0xee06800c, 0x00800000, 0x00000000, 0xbfb00000,
            ],
            args,
        );
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        for offset in [0, 8] {
            let a = f.arg_at(offset).unwrap();
            assert_eq!(a.mode, Some(AccessMode::Read));
            assert!(
                a.accesses.iter().all(|x| x.shared),
                "the selected load names both candidates"
            );
        }
        assert_eq!(mode(&f, 24), Some(AccessMode::Write));
        let flag = f.arg_at(16).unwrap();
        assert_eq!(
            (flag.role, flag.sinks.has(Sinks::GUARD)),
            (ArgRole::Integer, true)
        );
        assert_eq!(f.read_cache, ReadCacheClass::VmemOnly);
    }

    #[test]
    fn unattributable_bases_make_the_kernel_unknown() {
        // A pointer loaded from memory (indirect chain): arg a is read through the scalar cache.
        let f = facts(
            &[
                0xf4002100, 0xf8000000, 0xbfc70000, 0xf4002182, 0xf8000000, 0xbfc70000, 0x7e000280,
                0xee068006, 0x00000000, 0x00000000, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(
            (f.status, rules(&f)),
            (KernelStatus::Unknown, vec!["base-memory-derived"])
        );
        assert_eq!(f.arg_at(0).unwrap().read_cache, ReadCacheClass::ScalarOnly);
        assert_eq!(f.read_cache, ReadCacheClass::Unknown);
        // A buffer V# built from memory.
        let f = facts(
            &[
                0xf4002100, 0xf8000000, 0xbfc70000, 0xf4004202, 0xf8000000, 0x7e000280, 0xbfc70000,
                0xc406807c, 0x40801000, 0x00000000, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(
            (f.status, rules(&f)),
            (KernelStatus::Unknown, vec!["base-memory-derived"])
        );
        // A constant base: s_mov_b64 s[6:7], 0x1000.
        let f = facts(
            &[
                0xbe8601ff, 0x00001000, 0x7e000280, 0xee068006, 0x00000000, 0x00000000, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(
            (f.status, rules(&f)),
            (KernelStatus::Unknown, vec!["base-no-provenance"])
        );
    }

    #[test]
    fn disabled_lanes_keep_their_pointer_until_exec_widens() {
        // v[2:3] = a; load; s_mov_b32 s12, exec_lo; v_cmpx_gt_u32 16, v0; v[2:3] = b;
        // store (enabled lanes hold b only); s_or_b32 exec_lo, exec_lo, s12; store (a or b).
        let f = facts(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0x7e040204, 0x7e060205, 0xee05007c, 0x00000001,
                0x00000002, 0xbe8c007e, 0x7d980090, 0x7e040206, 0x7e060207, 0xbfc00000, 0xee06807c,
                0x00800000, 0x00000002, 0x8c7e0c7e, 0xee06807c, 0x00800000, 0x00000002, 0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        let stores_to_a = f
            .arg_at(0)
            .unwrap()
            .accesses
            .iter()
            .filter(|x| x.class == AccessClass::VmemStore)
            .count();
        let stores_to_b = f
            .arg_at(8)
            .unwrap()
            .accesses
            .iter()
            .filter(|x| x.class == AccessClass::VmemStore)
            .count();
        assert_eq!(
            (stores_to_a, stores_to_b),
            (1, 2),
            "narrowed store reaches b only; the store after the restore reaches both"
        );
        assert_eq!(mode(&f, 0), Some(AccessMode::ReadWrite));
    }

    #[test]
    fn divergent_writes_of_one_pointer_register_keep_both_candidates() {
        // if (v0 < 16) v[2:3] = a else v[2:3] = b (s_and_saveexec / s_xor exec / s_or exec); load v[2:3].
        let f = facts(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0x7c980090, 0xbe88206a, 0x7e040204, 0x7e060205,
                0x8d7e087e, 0x7e040206, 0x7e060207, 0x8c7e087e, 0xee05007c, 0x00000001, 0x00000002,
                0xbfb00000,
            ],
            PTRS,
        );
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!(
            (mode(&f, 0), mode(&f, 8)),
            (Some(AccessMode::Read), Some(AccessMode::Read))
        );
        assert!(f.arg_at(0).unwrap().accesses[0].shared);
    }

    #[test]
    fn implicit_block_count_loaded_as_loop_bound_and_scalar_roles() {
        // n = s_load(0x8); bc = s_load(0x10) (hidden_block_count_x); loop while ++i < bc;
        // store i at out + n * ttmp9 (a work-group id).
        let args = &[
            ("out", 0, 8, "global_buffer"),
            ("n", 8, 4, "by_value"),
            ("bcx", 16, 4, "hidden_block_count_x"),
            ("bcy", 20, 4, "hidden_block_count_y"),
            ("bcz", 24, 4, "hidden_block_count_z"),
            ("gsx", 28, 2, "hidden_group_size_x"),
        ];
        let f = facts(
            &[
                0xf4000080, 0xf8000010, 0xf4002100, 0xf8000000, 0xf4000200, 0xf8000008, 0xbfc70000,
                0xbe830080, 0x81038103, 0xbf0a0203, 0xbfa2fffd, 0x96067508, 0x7e000206, 0x7e020203,
                0xee068004, 0x00800000, 0x00000000, 0xbfb00000,
            ],
            args,
        );
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!(f.implicit.hidden_loaded, ["hidden_block_count_x"]);
        assert_eq!(f.implicit.hidden_grid_live, ["hidden_block_count_x"]);
        assert!(f.arg_at(16).unwrap().sinks.has(Sinks::LOOP_BOUND));
        let n = f.arg_at(8).unwrap();
        assert_eq!(n.role, ArgRole::Integer);
        assert!(
            n.sinks.has(Sinks::GRID_TERM)
                && n.sinks.has(Sinks::ADDRESS_OFFSET)
                && !n.sinks.has(Sinks::ADDRESS_BASE)
        );
        assert_eq!(f.arg_at(20).unwrap().role, ArgRole::NotLoaded);
        assert!(f.arg_at(0).unwrap().sinks.has(Sinks::ADDRESS_BASE));
    }

    #[test]
    fn analyzed_revision_carries_the_facts() {
        let program = crate::inst::Program {
            target: crate::inst::Target {
                arch: Arch::Gfx1201,
                xnack: crate::inst::Setting::Any,
                sramecc: crate::inst::Setting::Any,
                abi_version: 4,
            },
            kernels: vec![kernel(
                &[
                    0xf4004100, 0xf8000000, 0xbfc70000, 0x7c980090, 0xbe88206a, 0x7e040204,
                    0x7e060205, 0x8d7e087e, 0x7e040206, 0x7e060207, 0x8c7e087e, 0xee05007c,
                    0x00000001, 0x00000002, 0xbfb00000,
                ],
                PTRS,
            )],
            source: None,
        };
        let analyzed = crate::edit::analyze(program, &SymbolId("k".into())).expect("analyzes");
        assert_eq!(analyzed.facts.kernargs.status, KernelStatus::Proven);
        assert_eq!(
            analyzed.facts.kernargs.arg_at(8).and_then(|a| a.mode),
            Some(AccessMode::Read)
        );
    }

    // ---- Amendment 2026-10-25 §A9.5 fixtures 1-3 (synthetic gfx1201 ISA) ----

    /// dst +0, P (index source) +8, one scalar +24; kernarg segment s[0:1].
    const IDX: &[(&str, u32, u32, &str)] = &[
        ("dst", 0, 8, "global_buffer"),
        ("p", 8, 8, "global_buffer"),
        ("k", 24, 4, "by_value"),
    ];
    fn gfacts(words: &[u32], args: &[(&str, u32, u32, &str)]) -> KernargFacts {
        analyze_guarded(&kernel(words, args), Arch::Gfx1201, &[])
    }
    fn disp(e: &IndexedArgEffect) -> String {
        e.displacement.to_string()
    }

    // s_load_b128 s[4:7], s[0:1]; s_load_b32 s8, s[0:1], 0x18; s_load_b32 s9, s[6:7], 0x4;
    // s[10:11] = sext(s9); s[14:15] = sext(s8); s[12:13] = s[10:11] * s[14:15];
    // s[4:5] += s[12:13]; global_store_b32 v1, v2, s[4:5]
    const SEXT_MUL: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4000200, 0xf8000018, 0xbfc70000, 0xf4000243, 0xf8000004,
        0xbfc70000, 0x860b9f09, 0xbe8a0009, 0x860f9f08, 0xbe8e0008, 0xaa8c0e0a, 0xa9840c04,
        0x7e020280, 0x7e040287, 0xee068004, 0x01000000, 0x00000001, 0xbfb00000,
    ];

    #[test]
    fn sext_index_times_stride_is_a_conditional_indexed_effect() {
        let f = gfacts(SEXT_MUL, IDX);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        assert_eq!(f.indexed.len(), 1);
        let e = &f.indexed[0];
        assert_eq!(
            (e.base_arg_offset, e.mode, e.class, e.path, e.width_bytes),
            (
                0,
                AccessMode::Write,
                AccessClass::VmemStore,
                MemPath::Global,
                4
            )
        );
        assert_eq!(
            (e.pc, e.cache),
            (13, CacheBits::default()),
            "store pc, no cache bits"
        );
        assert_eq!(disp(e), "(mul (sext load:0) (sext karg32:24))");
        let il = &e.index_loads[0];
        assert_eq!(
            (il.pc, il.src_arg_offset, il.class, il.path, il.elem),
            (3, 8, AccessClass::SmemLoad, MemPath::Smem, IndexElem::I32)
        );
        assert_eq!(il.byte_offset.to_string(), "c64:4");
        assert_eq!(il.ext, [crate::kernarg::IndexExt::Sext64]);
        // dst write / P read, allocation-wide, scalar-cache read of P.
        assert_eq!(
            (mode(&f, 0), mode(&f, 8)),
            (Some(AccessMode::Write), Some(AccessMode::Read))
        );
        assert_eq!(f.arg_at(8).unwrap().read_cache, ReadCacheClass::ScalarOnly);
        assert!(f.guards.is_empty() && e.guards.is_empty());
        // The legacy lattice drops the loaded index through the sign extension
        // and multiply (its named assumption): unguarded results are unchanged.
        let legacy = analyze(&kernel(SEXT_MUL, IDX), Arch::Gfx1201);
        assert_eq!(legacy.status, KernelStatus::Proven);
        assert!(legacy.indexed.is_empty() && legacy.guards.is_empty());
    }

    // v1 = 0; global_load_b32 v2, v1, s[6:7] offset:8 scope:SCOPE_DEV; v3 = v2 << 2;
    // v[4:5] = s[4:5] + zext(v3) via v_add_co_u32 / v_add_co_ci_u32 (VOP2, vcc);
    // global_store_b32 v[4:5], v7, off
    const VMEM_ZEXT_VCC: &[u32] = &[
        0xf4004100, 0xf8000000, 0xbfc70000, 0x7e020280, 0xee050006, 0x00080002, 0x00000801,
        0xbfc00000, 0x30060482, 0xd7006a04, 0x02020604, 0x7e0c0280, 0x400a0c05, 0x7e0e0285,
        0xee06807c, 0x03800000, 0x00000004, 0xbfb00000,
    ];

    #[test]
    fn vmem_index_load_with_carry_chain_through_vcc() {
        let f = gfacts(VMEM_ZEXT_VCC, IDX);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        let e = &f.indexed[0];
        assert_eq!(disp(e), "(zext (shl load:0 2))");
        let il = &e.index_loads[0];
        assert_eq!(
            (il.class, il.path, il.cache.scope, il.src_arg_offset),
            (AccessClass::VmemLoad, MemPath::Global, 2, 8)
        );
        assert_eq!(il.byte_offset.to_string(), "c64:8");
        assert_eq!(il.ext, [crate::kernarg::IndexExt::Raw32]);
        assert_eq!(f.arg_at(8).unwrap().read_cache, ReadCacheClass::VmemOnly);
        // The 32-bit shift wraps at 32 bits: the checked evaluation refuses a
        // value whose shifted form would wrap, accepts one that does not.
        let at = |p: u64| {
            e.displacement.eval_checked(&|leaf| match leaf {
                crate::kernarg::IdxLeaf::Load(0) => Some(p),
                _ => None,
            })
        };
        assert_eq!(at(0x1000), Ok(0x4000));
        assert_eq!(at(0x4000_0000), Err(crate::kernarg::IdxEvalError::Wrap));
    }

    // s8 = load P; s[10:11] = zext(s8) << 2; s_add_u32 s4, s4, s10; s_addc_u32 s5, s5, s11
    const SMEM_ZEXT_SCALAR_CARRY: &[u32] = &[
        0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0xbe8a0008,
        0xbe8b0080, 0x848a820a, 0x80040a04, 0x82050b05, 0x7e020280, 0x7e040283, 0xee068004,
        0x01000000, 0x00000001, 0xbfb00000,
    ];

    #[test]
    fn zero_extended_index_through_scalar_carry_chain() {
        let f = gfacts(SMEM_ZEXT_SCALAR_CARRY, IDX);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        let e = &f.indexed[0];
        assert_eq!(disp(e), "(shl (zext load:0) 2)");
        assert_eq!(e.index_loads[0].ext, [crate::kernarg::IndexExt::Zext64]);
        assert_eq!(e.index_loads[0].byte_offset.to_string(), "c64:0");
    }

    // The same chain with the high-half add exchanged for a different high
    // word must not form an address: s_addc_u32 s5, s5, 0 loses the zext high.
    #[test]
    fn carry_chain_needs_the_matching_high_halves() {
        // s_load_b128 s[4:7]; s8 = P[0]; s10 = s8; s_add_u32 s4, s4, s10;
        // s_addc_u32 s5, s5, 0; store: the high half of the index is dropped.
        let f = gfacts(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0xbe8a0008,
                0x80040a04, 0x82058005, 0x7e020280, 0x7e040283, 0xee068004, 0x01000000, 0x00000001,
                0xbfb00000,
            ],
            IDX,
        );
        // s_addc_u32 with a zero high word is exactly zext(s10): it IS an address.
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        assert_eq!(disp(&f.indexed[0]), "(zext load:0)");
    }

    // Fixture 2: a loaded value used only as data/arithmetic is not a pointer effect.
    const LOAD_NOT_ADDRESS: &[u32] = &[
        0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x80098108,
        0x7e020280, 0x7e040209, 0xee068004, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    // SEXT_MUL with the roles of dst and P exchanged (index read from dst, store to P).
    const SEXT_MUL_SWAPPED: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4000200, 0xf8000018, 0xbfc70000, 0xf4000242, 0xf8000004,
        0xbfc70000, 0x860b9f09, 0xbe8a0009, 0x860f9f08, 0xbe8e0008, 0xaa8c0e0a, 0xa9860c06,
        0x7e020280, 0x7e040287, 0xee068006, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    // SEXT_MUL with a 64-bit store (global_store_b64 v1, v[2:3], s[4:5]).
    const SEXT_MUL_B64: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4000200, 0xf8000018, 0xbfc70000, 0xf4000243, 0xf8000004,
        0xbfc70000, 0x860b9f09, 0xbe8a0009, 0x860f9f08, 0xbe8e0008, 0xaa8c0e0a, 0xa9840c04,
        0x7e020280, 0x7e040287, 0x7e060288, 0xee06c004, 0x01000000, 0x00000001, 0xbfb00000,
    ];

    #[test]
    fn arithmetic_load_that_never_reaches_an_address_is_no_pointer_effect() {
        let f = gfacts(LOAD_NOT_ADDRESS, IDX);
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert!(f.indexed.is_empty() && f.guards.is_empty());
        assert_eq!(
            (mode(&f, 0), mode(&f, 8)),
            (Some(AccessMode::Write), Some(AccessMode::Read))
        );
    }

    #[test]
    fn exchanged_source_and_destination_change_the_facts() {
        let a = gfacts(SEXT_MUL, IDX);
        let b = gfacts(SEXT_MUL_SWAPPED, IDX);
        assert_eq!(b.status, KernelStatus::Conditional, "{:?}", b.unresolved);
        let (ea, eb) = (&a.indexed[0], &b.indexed[0]);
        assert_eq!(
            (ea.base_arg_offset, ea.index_loads[0].src_arg_offset),
            (0, 8)
        );
        assert_eq!(
            (eb.base_arg_offset, eb.index_loads[0].src_arg_offset),
            (8, 0)
        );
        assert_eq!(
            (mode(&b, 0), mode(&b, 8)),
            (Some(AccessMode::Read), Some(AccessMode::Write))
        );
        assert_ne!(a.indexed, b.indexed);
    }

    #[test]
    fn store_width_and_hidden_grid_dependency_are_reflected() {
        let narrow = gfacts(SEXT_MUL, IDX);
        let wide = gfacts(SEXT_MUL_B64, IDX);
        assert_eq!(
            wide.status,
            KernelStatus::Conditional,
            "{:?}",
            wide.unresolved
        );
        assert_eq!(
            (narrow.indexed[0].width_bytes, wide.indexed[0].width_bytes),
            (4, 8)
        );
        // The stride is the hidden block-count word: the expression reads it and
        // the implicit-use facts name it as grid-live.
        let hidden = gfacts(
            SEXT_MUL,
            &[
                ("dst", 0, 8, "global_buffer"),
                ("p", 8, 8, "global_buffer"),
                ("bcx", 24, 4, "hidden_block_count_x"),
            ],
        );
        assert_eq!(
            hidden.status,
            KernelStatus::Conditional,
            "{:?}",
            hidden.unresolved
        );
        assert_eq!(
            disp(&hidden.indexed[0]),
            "(mul (sext load:0) (sext karg32:24))"
        );
        assert_eq!(hidden.implicit.hidden_grid_live, ["hidden_block_count_x"]);
        assert!(narrow.implicit.hidden_grid_live.is_empty());
    }

    // Fixture 3: everything that must stay Unknown.
    fn unknown(words: &[u32], why: &str) {
        let f = gfacts(words, IDX);
        assert_eq!(f.status, KernelStatus::Unknown, "{why}");
        assert_eq!(
            rules(&f),
            ["base-memory-derived"],
            "{why}: {:?}",
            f.unresolved
        );
        assert!(f.indexed.iter().all(|e| e.index_loads.len() == 1), "{why}");
    }

    #[test]
    fn loaded_pointers_and_descriptors_stay_unknown_under_a_guard() {
        // s_load_b64 s[8:9], s[6:7]; store through s[8:9].
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4002203, 0xf8000000, 0xbfc70000, 0x7e020280,
                0x7e040287, 0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
            ],
            "loaded 64-bit pointer dereference",
        );
        // s_load_b128 s[8:11], s[6:7]; buffer_store through that V#.
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4004203, 0xf8000000, 0xbfc70000, 0x7e020280,
                0x7e040287, 0xc406807c, 0x40801002, 0x00000001, 0xbfb00000,
            ],
            "descriptor-built pointer base",
        );
        // Two loaded 32-bit halves reassembled as s[8:9].
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xf4000243, 0xf8000004,
                0xbfc70000, 0x7e020280, 0x7e040287, 0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
            ],
            "two loaded32 halves as a pointer",
        );
    }

    #[test]
    fn mixed_bases_and_stale_lanes_stay_unknown_under_a_guard() {
        // if (P[0] != 0) s[4:5] += sext(P[0]) << 2; store s[4:5]: direct base
        // joined with an indexed one.
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0xbf068008,
                0xbfa20005, 0x860b9f08, 0xbe8a0008, 0x848a820a, 0xa9840a04, 0xbf800000, 0x7e020280,
                0x7e040281, 0xee068004, 0x01000000, 0x00000001, 0xbfb00000,
            ],
            "direct/indexed base through a CFG join",
        );
        // v[2:3] = dst; under a narrowed EXEC v[2:3] = dst + (P[0] << 2); EXEC
        // restored; store through v[2:3]: the stale lanes keep the direct base.
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x7e040204,
                0x7e060205, 0xbe8c007e, 0x7d980090, 0x860b9f08, 0xbe8a0008, 0x848a820a, 0xa98e0a04,
                0x7e04020e, 0x7e06020f, 0x8c7e0c7e, 0x7e080289, 0xee06807c, 0x02000000, 0x00000002,
                0xbfb00000,
            ],
            "partial-EXEC stale base",
        );
    }

    #[test]
    fn full_exec_rewrite_of_a_pointer_register_pair_is_the_positive_control() {
        // The stale-lane fixture without the narrowed EXEC: v[2:3] = dst, then
        // v[2:3] = dst + (P[0] << 2) on all lanes, store through v[2:3].
        let f = gfacts(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x7e040204,
                0x7e060205, 0x860b9f08, 0xbe8a0008, 0x848a820a, 0xa98e0a04, 0x7e04020e, 0x7e06020f,
                0x7e080289, 0xee06807c, 0x02000000, 0x00000002, 0xbfb00000,
            ],
            IDX,
        );
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        assert_eq!(disp(&f.indexed[0]), "(shl (sext load:0) 2)");
    }

    #[test]
    fn pointers_in_the_index_and_unsupported_arithmetic_stay_unknown_under_a_guard() {
        // index = P[0] * dst_low (a pointer multiplied into an integer index).
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x96090408,
                0x860b9f09, 0xbe8a0009, 0xa9840a04, 0x7e020280, 0x7e040287, 0xee068004, 0x01000000,
                0x00000001, 0xbfb00000,
            ],
            "pointer multiplied into an index",
        );
        // index = P[0] ^ 0x55 (xor is not a modelled operation).
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x8d09ff08,
                0x00000055, 0x860b9f09, 0xbe8a0009, 0xa9840a04, 0x7e020280, 0x7e040287, 0xee068004,
                0x01000000, 0x00000001, 0xbfb00000,
            ],
            "unsupported xor",
        );
        // index = P[0] >> 3: the legacy lattice kills it, the witness does not model it.
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x85098308,
                0x860b9f09, 0xbe8a0009, 0xa9840a04, 0x7e020280, 0x7e040287, 0xee068004, 0x01000000,
                0x00000001, 0xbfb00000,
            ],
            "unsupported logical shift right",
        );
        // P[Q[0]]: the index load's own address contains a loaded term.
        unknown(
            &[
                0xf4004100, 0xf8000000, 0xbfc70000, 0xf4000203, 0xf8000000, 0xbfc70000, 0x860b9f08,
                0xbe8a0008, 0x848a820a, 0xa98c0a06, 0xf4000386, 0xf8000000, 0xbfc70000, 0x86119f0e,
                0xbe90000e, 0x84908210, 0xa9841004, 0x7e020280, 0x7e040287, 0xee068004, 0x01000000,
                0x00000001, 0xbfb00000,
            ],
            "nested P[Q[i]] indexing",
        );
    }
    const POSITIVE: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const CONST_BOUND: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0ac00b, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const STORE_THROUGH: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000c, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0x7e040280, 0xee06800e, 0x01000000, 0x00000001,
        0xbfb00000,
    ];
    const UNGUARDED: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000, 0xbfc70000,
        0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008, 0x01000000,
        0x00000001, 0xbfb00000,
    ];
    const WRONG_VALUE: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a11, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const SIGNED_GUARD: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf040a0b, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const REVERSED_GUARD: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0b0a, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const BYPASSED_GUARD: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf068010, 0xbfa20002, 0xbf0a0a0b, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c,
        0xa9920c04, 0xf4002389, 0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002,
        0x00000001, 0xbfc00000, 0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const FLAG_DIAMOND: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbe9400c1, 0xbf0a0a0b, 0xbfa20001, 0xbe940080, 0x8b157e14, 0x98158081, 0xbf078115,
        0xbfa2000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000,
        0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008,
        0x01000000, 0x00000001, 0xbfb00000,
    ];
    const FLAG_DIAMOND_BROKEN: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbe9400c1, 0xbf0a0a0b, 0xbfa20001, 0xbe9400c1, 0x8b157e14, 0x98158081, 0xbf078115,
        0xbfa2000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000,
        0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008,
        0x01000000, 0x00000001, 0xbfb00000,
    ];
    const POINTER_AS_INDEX: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10013, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05400e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee05007c, 0x00000004, 0x00000002, 0xbfc00000, 0xee068008, 0x02000000, 0x00000001,
        0xbfb00000,
    ];
    const POINTER_STORED: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000d, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0x7e04020e, 0x7e06020f, 0xee06c008, 0x01000000,
        0x00000001, 0xbfb00000,
    ];
    const POINTER_MIXED_WITH_ARG: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000c, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xa99c0e08, 0xee06801c, 0x00800000, 0x00000001,
        0xbfb00000,
    ];
    const ATOMIC_THROUGH: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000b, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee0d400e, 0x00800000, 0x00000001, 0xbfb00000,
    ];
    const SCALE4: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c820c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const ENTRY_OFFSET: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000008, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const LOOP_ENTRY: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10013, 0xbe9e0082, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04,
        0xf4002389, 0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001,
        0xbfc00000, 0xee068008, 0x01000000, 0x00000001, 0x819e811e, 0xbf07801e, 0xbfa2ffee,
        0xbfb00000,
    ];
    const TWO_LOADS_ONE_TABLE: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa1001d, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa99a0c04,
        0xf400258d, 0xf8000000, 0xbfc70000, 0xee050016, 0x00000003, 0x00000001, 0xbfc00000,
        0xee068008, 0x01800000, 0x00000001, 0xbfb00000,
    ];

    // ---------------------------------------------------------------------
    // Sealed-table analysis. Kernels assembled with llvm-mc (gfx1201): table
    // pointer +0, index pointer +8, output +16, bound `n` +24, flag +28.
    // ---------------------------------------------------------------------
    const SEALED_ARGS: &[(&str, u32, u32, &str)] = &[
        ("table", 0, 8, "global_buffer"),
        ("idx", 8, 8, "global_buffer"),
        ("out", 16, 8, "global_buffer"),
        ("n", 24, 4, "by_value"),
        ("flag", 28, 4, "by_value"),
    ];
    fn sealed(words: &[u32]) -> KernargFacts {
        analyze_sealed(&kernel(words, SEALED_ARGS), Arch::Gfx1201)
    }
    fn first_rule(f: &KernargFacts) -> &'static str {
        f.unresolved.first().map_or("", |u| u.rule)
    }
    fn refuses(words: &[u32], rule: &str) {
        let f = sealed(words);
        assert_eq!(f.status, KernelStatus::Unknown, "{:?}", f.unresolved);
        assert!(f.sealed.is_empty());
        assert_eq!(first_rule(&f), rule, "{:?}", f.unresolved);
    }

    #[test]
    fn guarded_table_entry_read_through_is_a_conditional_sealed_table() {
        let f = sealed(POSITIVE);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        assert!(f.guards.is_empty() && f.indexed.is_empty());
        let [t] = &f.sealed[..] else {
            panic!("one table: {:?}", f.sealed)
        };
        assert_eq!(t.table_arg_offset, 0);
        assert_eq!(t.bound, SealedBound::Kernarg { offset: 24 });
        assert_eq!(t.entry_bytes, 8);
        let [e] = &t.entries[..] else { panic!() };
        assert_eq!(
            (e.pc, e.class, e.path, e.width_bytes),
            (14, AccessClass::SmemLoad, MemPath::Smem, 8)
        );
        assert_eq!(e.index, SealedIndex::Load { pc: 5 });
        assert_eq!(e.guard_pcs, [8]);
        let [a] = &t.accesses[..] else { panic!() };
        assert_eq!(
            (a.class, a.path, a.width_bytes, a.entry_pcs.clone()),
            (AccessClass::VmemLoad, MemPath::Global, 4, vec![e.pc])
        );
        // The table argument is read (its entry load is an ordinary read);
        // the output argument is written; the loaded pointer is no argument.
        assert_eq!(mode(&f, 0), Some(AccessMode::Read));
        assert_eq!(mode(&f, 16), Some(AccessMode::Write));
        // The unguarded analysis is unchanged: Unknown, memory-derived.
        let base = analyze(&kernel(POSITIVE, SEALED_ARGS), Arch::Gfx1201);
        assert_eq!(base.status, KernelStatus::Unknown);
        assert_eq!(rules(&base), ["base-memory-derived"]);
        assert!(base.sealed.is_empty());
    }

    #[test]
    fn constant_bound_and_two_entry_loads_of_one_table() {
        let f = sealed(CONST_BOUND);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        assert_eq!(f.sealed[0].bound, SealedBound::Const(64));
        let f = sealed(TWO_LOADS_ONE_TABLE);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        let [t] = &f.sealed[..] else { panic!() };
        assert_eq!(t.entries.len(), 2);
        assert_eq!(t.accesses.len(), 2);
        assert_ne!(t.accesses[0].entry_pcs, t.accesses[1].entry_pcs);
    }

    #[test]
    fn a_flag_materialised_by_a_diamond_is_followed_not_guessed() {
        // `ok = idx < n ? -1 : 0; if (!(ok & exec)) return;` as the compiler
        // writes it: the guard holds only through the flag.
        let f = sealed(FLAG_DIAMOND);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        assert_eq!(f.sealed[0].entries[0].guard_pcs.len(), 1);
        // The same shape with the flag set on both paths guards nothing.
        refuses(FLAG_DIAMOND_BROKEN, "sealed-index-unguarded");
    }

    #[test]
    fn only_a_complete_unsigned_guard_of_the_loaded_index_qualifies() {
        refuses(UNGUARDED, "sealed-index-unguarded");
        refuses(WRONG_VALUE, "sealed-index-unguarded");
        refuses(SIGNED_GUARD, "sealed-index-unguarded");
        refuses(REVERSED_GUARD, "sealed-index-unguarded");
        refuses(BYPASSED_GUARD, "sealed-index-unguarded");
    }

    #[test]
    fn writes_and_escapes_through_the_loaded_base_stay_unknown() {
        refuses(STORE_THROUGH, "sealed-table-write");
        refuses(ATOMIC_THROUGH, "sealed-table-write");
        refuses(POINTER_STORED, "sealed-table-value-stored");
        // An argument candidate beside the entry makes the base `Both`; the
        // access is a store, so it is refused as a write through an entry.
        refuses(POINTER_MIXED_WITH_ARG, "sealed-table-write");
        // A loaded base used as an address again is memory-derived.
        refuses(POINTER_AS_INDEX, "base-memory-derived");
    }

    #[test]
    fn entry_shapes_the_certificate_does_not_cover_keep_the_plain_rule() {
        for words in [SCALE4, ENTRY_OFFSET] {
            let f = sealed(words);
            assert_eq!(f.status, KernelStatus::Unknown);
            assert!(f.unresolved.iter().all(|u| u.rule == "base-memory-derived"));
            assert!(f.sealed.is_empty());
        }
        refuses(LOOP_ENTRY, "sealed-entry-in-cycle");
    }
    const SIGNED_GUARD_FLAGS: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbfa50016, 0xbf04800b, 0x981480c1, 0xbf030a0b, 0x981580c1, 0x8c141514, 0x8b6a147e,
        0xbfa4000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000,
        0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008,
        0x01000000, 0x00000001, 0xbfb00000,
    ];
    const SIGNED_NO_EXEC_FACT: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf04800b, 0x981480c1, 0xbf030a0b, 0x981580c1, 0x8c141514, 0x8b6a147e, 0xbfa4000f,
        0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000, 0xbfc70000,
        0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008, 0x01000000,
        0x00000001, 0xbfb00000,
    ];
    const SIGNED_MISSING_NONNEG: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbfa50015, 0xbe940080, 0xbf030a0b, 0x981580c1, 0x8c141514, 0x8b6a147e, 0xbfa4000f,
        0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000, 0xbfc70000,
        0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008, 0x01000000,
        0x00000001, 0xbfb00000,
    ];
    const SIGNED_NONNEG_OF_OTHER: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbfa50016, 0xbf048011, 0x981480c1, 0xbf030a0b, 0x981580c1, 0x8c141514, 0x8b6a147e,
        0xbfa4000f, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389, 0xf8000000,
        0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008,
        0x01000000, 0x00000001, 0xbfb00000,
    ];
    const NULL_CHECKED: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10011, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xbf11800e, 0xbfa10007, 0xee05000e, 0x00000002,
        0x00000001, 0xbfc00000, 0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const DEREF_BEFORE_NULL_CHECK: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10011, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbf11800e, 0xbfa10000, 0xbfb00000,
    ];
    const NULL_CHECK_OTHER_REGISTER: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10011, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xbf118008, 0xbfa10007, 0xee05000e, 0x00000002,
        0x00000001, 0xbfc00000, 0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const NULL_CHECK_BYPASSED: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10013, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xbf068010, 0xbfa20002, 0xbf11800e, 0xbfa10007,
        0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008, 0x01000000, 0x00000001,
        0xbfb00000,
    ];
    const NULL_CHECK_FLAGS: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbfa50017, 0xbf0a0a0b, 0xbfa10015, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04,
        0xf4002389, 0xf8000000, 0xbfc70000, 0x7e020280, 0xbf11800e, 0x981480c1, 0x8b157e14,
        0x98158081, 0xbf078115, 0xbfa20007, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000,
        0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];
    const NULL_CHECK_FLAGS_NO_EXEC_FACT: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10015, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xbf11800e, 0x981480c1, 0x8b157e14, 0x98158081,
        0xbf078115, 0xbfa20007, 0xee05000e, 0x00000002, 0x00000001, 0xbfc00000, 0xee068008,
        0x01000000, 0x00000001, 0xbfb00000,
    ];
    const ARG_OR_ENTRY_BASE: &[u32] = &[
        0xf4004100, 0xf8000000, 0xf4002200, 0xf8000010, 0xf4000280, 0xf8000018, 0xf4000400,
        0xf800001c, 0xbfc70000, 0xf40002c3, 0xf8000000, 0xf4000443, 0xf8000004, 0xbfc70000,
        0xbf0a0a0b, 0xbfa10011, 0xbe8c000b, 0x860d9f0b, 0x848c830c, 0xa9920c04, 0xf4002389,
        0xf8000000, 0xbfc70000, 0x7e020280, 0xbf068110, 0x9896080e, 0xee050016, 0x00000002,
        0x00000001, 0xbfc00000, 0xee068008, 0x01000000, 0x00000001, 0xbfb00000,
    ];

    #[test]
    fn a_signed_pair_guard_through_flags_and_exec_is_followed() {
        // `if (e < 0 || e >= n) return;` as the compiler writes it: two
        // compares into masks, `or`, `exec & mask`, `vccnz`. It proves `e < n`
        // only because EXEC is known nonzero (the EXECZ edge before it).
        let f = sealed(SIGNED_GUARD_FLAGS);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        let [t] = &f.sealed[..] else { panic!() };
        assert_eq!(t.bound, SealedBound::Kernarg { offset: 24 });
        assert_eq!(t.entries[0].guard_pcs.len(), 2, "both signed compares");
        refuses(SIGNED_NO_EXEC_FACT, "sealed-index-unguarded");
        refuses(SIGNED_MISSING_NONNEG, "sealed-index-unguarded");
        refuses(SIGNED_NONNEG_OF_OTHER, "sealed-index-unguarded");
    }

    #[test]
    fn only_an_entry_checked_nonzero_before_every_read_is_null_guarded() {
        for (name, words, want) in [
            ("scc null check", NULL_CHECKED, true),
            ("flag null check with exec known", NULL_CHECK_FLAGS, true),
            ("no null check", POSITIVE, false),
            ("deref before the check", DEREF_BEFORE_NULL_CHECK, false),
            (
                "check of another register",
                NULL_CHECK_OTHER_REGISTER,
                false,
            ),
            ("check bypassed on a path", NULL_CHECK_BYPASSED, false),
            (
                "flag check without the exec fact",
                NULL_CHECK_FLAGS_NO_EXEC_FACT,
                false,
            ),
        ] {
            let f = sealed(words);
            assert_eq!(
                f.status,
                KernelStatus::Conditional,
                "{name}: {:?}",
                f.unresolved
            );
            assert_eq!(f.sealed[0].null_guarded, want, "{name}");
        }
    }

    #[test]
    fn a_base_that_is_an_argument_on_one_path_and_an_entry_on_another_reads_both() {
        let f = sealed(ARG_OR_ENTRY_BASE);
        assert_eq!(f.status, KernelStatus::Conditional, "{:?}", f.unresolved);
        let [t] = &f.sealed[..] else { panic!() };
        assert_eq!(t.accesses.len(), 1);
        assert!(!t.null_guarded);
        // The read is also an ordinary read of the argument it may name.
        let out = f.arg_at(16).expect("out argument");
        assert!(out
            .accesses
            .iter()
            .any(|a| a.class == AccessClass::VmemLoad));
    }
}
