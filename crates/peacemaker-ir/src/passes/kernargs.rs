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
use std::collections::BTreeMap;

use petgraph::graph::{DiGraph, NodeIndex};
use smallvec::SmallVec;

use crate::cfg::{Body, InstId};
use crate::effects::ImplicitSet;
use crate::inst::{Abi, Arch, Form, FormFields, Inst, Kernel, UserSgprRole, VmemForm, Wave};
use crate::isa::OpRow;
use crate::kernarg::{
    AccessClass, AccessMode, ArgAccess, ArgFacts, ArgRole, CacheBits, DwordFacts, ImplicitUse, KernargFacts,
    KernelStatus, MemPath, ReadCacheClass, RegionBound, SideReads, Sinks, Unresolved, HIDDEN_GRID_KINDS,
};
use crate::metadata::Kernarg;
use crate::operand::{ImmField, InlineConst, Operand, Special};
use crate::reg::Kind;

/// Kernarg dwords tracked individually (512 bytes). Kernels with a larger
/// segment fall back to the overflow flags: an overflowing pointer root is
/// `Unknown`, an overflowing taint is reported as `taint_overflow`.
const MAX_DW: u32 = 128;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
struct DwSet(u128);
impl DwSet {
    const ALL: DwSet = DwSet(u128::MAX);
    fn one(dw: u32) -> Self { DwSet(1u128 << dw) }
    fn is_empty(self) -> bool { self.0 == 0 }
    fn union(self, other: Self) -> Self { DwSet(self.0 | other.0) }
    fn minus(self, other: Self) -> Self { DwSet(self.0 & !other.0) }
    fn contains(self, dw: u32) -> bool { dw < MAX_DW && self.0 >> dw & 1 != 0 }
    fn iter(self) -> impl Iterator<Item = u32> { (0..MAX_DW).filter(move |&dw| self.0 >> dw & 1 != 0) }
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Val {
    roots: DwSet,
    src: u8,
    taint: DwSet,
    tf: u8,
    konst: Option<u32>,
    /// Byte offset from the kernarg segment base, for a `KSEG`-rooted value.
    koff: Option<i64>,
}

impl Val {
    fn konst(value: u32) -> Self { Val { konst: Some(value), ..Val::default() } }
    fn src(src: u8) -> Self { Val { src, ..Val::default() } }
    fn flags(tf: u8) -> Self { Val { tf, ..Val::default() } }
    fn karg(dw: u32) -> Self {
        if dw < MAX_DW { Val { roots: DwSet::one(dw), taint: DwSet::one(dw), ..Val::default() } }
        else { Val { src: ROOT_OVF, tf: TAINT_OVF, ..Val::default() } }
    }
    fn loaded() -> Self { Val { src: OPAQUE, tf: MEM, ..Val::default() } }
    /// A dword read from the kernarg segment at a dynamic offset: any argument.
    fn any_karg() -> Self { Val { src: OPAQUE, taint: DwSet::ALL, tf: TAINT_OVF, ..Val::default() } }
    fn join(self, other: Self) -> Self {
        Val {
            roots: self.roots.union(other.roots),
            src: self.src | other.src,
            taint: self.taint.union(other.taint),
            tf: self.tf | other.tf,
            konst: if self.konst == other.konst { self.konst } else { None },
            koff: if self.koff == other.koff { self.koff } else { None },
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
        }
        out
    }
    /// Arithmetic producing an integer: provenance dropped, influence kept.
    fn kill<'v>(vals: impl IntoIterator<Item = &'v Val>) -> Self {
        let mut out = Val::mix(vals);
        out.roots = DwSet::default();
        out.src = 0;
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
    Within { level: u8, exact: bool },
}
impl Mask {
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Mask::Full, x) | (x, Mask::Full) => x,
            (Mask::Within { level: a, .. }, Mask::Within { level: b, .. }) => Mask::Within { level: a.max(b), exact: false },
            (Mask::Within { level, .. }, Mask::Unknown) | (Mask::Unknown, Mask::Within { level, .. }) => Mask::Within { level, exact: false },
            _ => Mask::Unknown,
        }
    }
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Mask::Full, _) | (_, Mask::Full) => Mask::Full,
            (Mask::Within { level: a, exact: ea }, Mask::Within { level: b, exact: eb }) =>
                Mask::Within { level: a.min(b), exact: (ea && a <= b) || (eb && b <= a) },
            _ => Mask::Unknown,
        }
    }
    fn xor(self, other: Self) -> Self {
        match (self, other) {
            (Mask::Within { level: a, .. }, Mask::Within { level: b, .. }) => Mask::Within { level: a.min(b), exact: false },
            _ => Mask::Unknown,
        }
    }
    /// `self & !other`.
    fn and_not(self, _other: Self) -> Self {
        match self { Mask::Within { level, .. } => Mask::Within { level, exact: false }, _ => Mask::Unknown }
    }
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (a, b) if a == b => a,
            (Mask::Within { level: a, .. }, Mask::Within { level: b, .. }) => Mask::Within { level: a.min(b), exact: false },
            _ => Mask::Unknown,
        }
    }
    /// Levels above `depth - 1` were closed: a claim about them weakens to
    /// the deepest open level (`S_j ⊆ S_{depth-1}` for `j ≥ depth`).
    fn clamp(self, depth: usize) -> Self {
        match self {
            Mask::Within { level, .. } if usize::from(level) >= depth => Mask::Within { level: (depth - 1) as u8, exact: false },
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
}

impl State {
    fn empty() -> Self {
        State {
            s: vec![Val::default(); NS], t: vec![Val::default(); 16], v: vec![Val::default(); NV],
            lanes: BTreeMap::new(), levels: vec![vec![Val::default(); NV]], exec_exact: Some(0),
            masks: vec![Mask::Unknown; NS], exec: [Val::default(); 2], m0: Val::default(), scc: Val::default(),
        }
    }
    fn top(&self) -> u8 { (self.levels.len() - 1) as u8 }
    fn read_v(&self, r: u16) -> Val {
        let mut out = self.v[usize::from(r)];
        for (_, lane) in self.lanes.range((r, 0)..=(r, u8::MAX)) { out = out.join(*lane); }
        out
    }
    /// Values held by currently disabled lanes of VGPR `r` (cross-lane reads).
    fn disabled_v(&self, r: u16) -> Val {
        self.levels.iter().fold(Val::default(), |acc, level| acc.join(level[usize::from(r)]))
    }
    /// Close every level at or above `depth` into level `depth - 1`.
    fn collapse(&mut self, depth: usize) {
        while self.levels.len() > depth {
            let closed = self.levels.pop().expect("level");
            let below = self.levels.last_mut().expect("level 0");
            for (a, b) in below.iter_mut().zip(closed) { *a = a.join(b); }
        }
        if self.exec_exact.is_some_and(|k| usize::from(k) >= depth) { self.exec_exact = None; }
        for m in &mut self.masks { *m = m.clamp(depth); }
    }
    fn join_from(&mut self, o: &State) {
        for (a, b) in self.s.iter_mut().zip(&o.s) { *a = a.join(*b); }
        for (a, b) in self.t.iter_mut().zip(&o.t) { *a = a.join(*b); }
        let mut lanes = BTreeMap::new();
        for (&(r, l), &val) in &self.lanes {
            let other = o.lanes.get(&(r, l)).copied().unwrap_or(o.v[usize::from(r)]);
            lanes.insert((r, l), val.join(other));
        }
        for (&(r, l), &val) in &o.lanes {
            lanes.entry((r, l)).or_insert_with(|| val.join(self.v[usize::from(r)]));
        }
        self.lanes = lanes;
        for (a, b) in self.v.iter_mut().zip(&o.v) { *a = a.join(*b); }
        let depth = self.levels.len().min(o.levels.len());
        self.collapse(depth);
        let mut other = o.clone();
        other.collapse(depth);
        for (mine, theirs) in self.levels.iter_mut().zip(&other.levels) {
            for (a, b) in mine.iter_mut().zip(theirs) { *a = a.join(*b); }
        }
        if self.exec_exact != other.exec_exact { self.exec_exact = None; }
        for (a, b) in self.masks.iter_mut().zip(&other.masks) { *a = a.join(*b); }
        self.exec = [self.exec[0].join(o.exec[0]), self.exec[1].join(o.exec[1])];
        self.m0 = self.m0.join(o.m0);
        self.scc = self.scc.join(o.scc);
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
                    for (v, dead) in self.v.iter_mut().zip(level.iter_mut()) { *v = v.join(*dead); *dead = Val::default(); }
                }
                self.exec_exact = Some(0);
            }
            Mask::Within { level, exact } => {
                let b = usize::from(level).min(self.levels.len() - 1);
                let mut joined = vec![Val::default(); NV];
                for l in &self.levels[b..] { for (j, x) in joined.iter_mut().zip(l) { *j = j.join(*x); } }
                for (v, j) in self.v.iter_mut().zip(&joined) { *v = v.join(*j); }
                self.levels.truncate(b + 1);
                if exact {
                    self.levels[b] = vec![Val::default(); NV];
                    self.exec_exact = Some(b as u8);
                } else {
                    for r in 0..NV { joined[r] = joined[r].join(self.read_v(r as u16)); }
                    self.levels[b] = joined;
                    self.exec_exact = None;
                }
                for m in &mut self.masks { *m = m.clamp(b + 1); }
            }
            Mask::Unknown => self.exec_set(Mask::Within { level: 0, exact: false }),
        }
    }
    /// A copy of the current EXEC: the level it is exactly equal to, opening
    /// a new level when EXEC is not a known saved mask.
    fn exec_copy_level(&mut self) -> u8 {
        let top = self.top();
        if self.exec_exact == Some(top) { return top; }
        self.levels.push(vec![Val::default(); NV]);
        let new = self.top();
        self.exec_exact = Some(new);
        new
    }
}

/// One decoded operand in its grammar field.
struct Field<'a> { name: &'static str, bits: u16, op: &'a Operand, def: bool, used: bool }
/// One opcode half (two for VOPD).
struct Part<'a> { name: &'static str, fields: SmallVec<[Field<'a>; 6]> }
/// A VALU lane-mask source (carry-in, select mask). The grammar types it
/// 64-bit; in wave32 it is one SGPR, and it never carries a value.
fn is_mask(part: &str, field: &str) -> bool {
    field == "SRC2" && (part.starts_with("v_add_co_ci") || part.starts_with("v_sub_co_ci") || part.starts_with("v_subrev_co_ci") || part.starts_with("v_cndmask"))
}
impl<'a> Part<'a> {
    /// Value sources: used, not also defined, not a lane mask.
    fn sources(&self) -> impl Iterator<Item = &Field<'a>> { self.fields.iter().filter(move |f| f.used && !f.def && !is_mask(self.name, f.name)) }
    fn field(&self, name: &str) -> Option<&Field<'a>> { self.fields.iter().find(|f| f.name == name) }
    fn first_use(&self) -> Option<&Field<'a>> { self.sources().next() }
}
struct Dec<'a> { id: InstId, inst: &'a Inst, parts: SmallVec<[Part<'a>; 2]>, extras: SmallVec<[&'a Operand; 4]> }

fn map_fields<'a>(row: &'static OpRow, form: Form, returns: bool, ops: &'a [Operand], extras: &mut SmallVec<[&'a Operand; 4]>) -> SmallVec<[Field<'a>; 6]> {
    let mut names = row.slots(returns);
    if form == Form::Vop3 && row.name.starts_with("v_cmpx_") { names.next(); }
    let prefix = usize::from(form == Form::Vopc && row.name.starts_with("v_cmp_")
        && matches!(ops.first(), Some(Operand::Special(Special::Vcc | Special::VccLo))));
    let defs: SmallVec<[&str; 4]> = row.defs.split(',').collect();
    let uses: SmallVec<[&str; 6]> = row.uses.split(',').collect();
    let mut out = SmallVec::new();
    for op in ops.iter().skip(prefix) {
        match names.next() {
            Some((name, bits)) if !name.is_empty() => {
                out.push(Field { name, bits, op, def: defs.contains(&name), used: uses.contains(&name) });
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
        parts.push(Part { name: x.name, fields: map_fields(x, Form::Vopd, returns, &inst.operands[..split], &mut extras) });
        parts.push(Part { name: y.name, fields: map_fields(y, Form::Vopd, returns, &inst.operands[split..], &mut extras) });
    } else {
        let row = crate::isa::lookup(arch, inst.op, inst.form)?;
        parts.push(Part { name: row.name, fields: map_fields(row, inst.form, returns, &inst.operands, &mut extras) });
    }
    Some(Dec { id, inst, parts, extras })
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
        "s_mul", "v_mul", "s_lshl_b", "s_lshr", "s_ashr", "v_lshlrev", "v_lshrrev", "v_ashrrev", "v_dual_lshlrev",
        "s_bfe", "v_bfe", "s_bcnt", "s_ff", "s_flbit", "s_cls", "v_ffbh", "v_ffbl", "v_cls", "v_clz", "v_ctz",
        "v_cvt", "s_cvt", "v_rcp", "v_rsq", "v_sqrt", "v_exp", "v_log", "v_sin", "v_cos", "v_frac", "v_ldexp",
        "v_div", "v_mbcnt", "v_bcnt", "s_bitcmp", "v_cmp", "s_cmp", "v_dot", "v_wmma", "v_swmmac",
        "v_sad", "v_msad", "v_qsad", "v_mqsad", "s_sext", "v_bfrev", "s_brev", "s_abs", "s_absdiff",
        "v_trig", "v_frexp", "v_class", "s_bitset", "s_quadmask", "s_wqm", "s_bitreplicate",
    ];
    const FLOAT: &[&str] = &["_f32", "_f16", "_f64", "_bf16", "_fp8", "_bf8"];
    PREFIX.iter().any(|p| name.starts_with(p)) || FLOAT.iter().any(|f| name.contains(f))
}

fn is_mov(name: &str) -> bool {
    matches!(name, "s_mov_b32" | "s_mov_b64" | "s_movk_i32" | "v_mov_b32_e32" | "v_mov_b32_e64" | "v_mov_b32_dpp"
        | "v_dual_mov_b32" | "v_readfirstlane_b32" | "v_mov_b64_e32" | "v_mov_b64_e64" | "s_mov_fed_b32" | "v_mov_b16_e32" | "v_mov_b16_e64")
}
fn is_select(name: &str) -> bool {
    name.starts_with("s_cselect_") || name.starts_with("v_cndmask_b32") || name == "v_dual_cndmask_b32"
}
fn is_add(name: &str) -> bool {
    matches!(name, "s_add_u32" | "s_add_co_u32" | "s_add_i32" | "s_add_co_i32" | "v_add_co_u32" | "v_add_nc_u32_e32"
        | "v_add_nc_u32_e64" | "v_add_u32_e32" | "v_add_u32_e64" | "v_dual_add_nc_u32" | "s_add_nc_u64")
}
/// Shift-add / multiply-add: provenance of the addend field only.
fn keep_field(name: &str) -> Option<&'static str> {
    if name.starts_with("s_lshl") && name.ends_with("_add_u32") { return Some("SSRC1"); }
    match name {
        "v_lshl_add_u32" | "v_lshl_or_b32" | "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mad_u32_u16" | "v_mad_i32_i16"
        | "v_mad_u16" | "v_mad_i16" | "v_mad_nc_u64_u32" | "v_mad_nc_i64_i32" => Some("SRC2"),
        _ => None,
    }
}
fn is_mad64(name: &str) -> bool {
    matches!(name, "v_mad_u64_u32" | "v_mad_i64_i32" | "v_mad_co_u64_u32" | "v_mad_co_i64_i32" | "v_lshl_add_u64")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemOp { Load, Store, Atomic { commutative: bool }, LdsDma, Other }
fn mem_op(name: &str) -> MemOp {
    if name.contains("atomic") {
        let noncommutative = name.contains("cmpswap") || name.contains("swap") || name.contains("cond_sub") || name.contains("csub");
        return MemOp::Atomic { commutative: !noncommutative };
    }
    if name.contains("_lds_") || name.ends_with("_lds") { return MemOp::LdsDma; }
    if name.contains("_load") { return MemOp::Load; }
    if name.contains("_store") { return MemOp::Store; }
    MemOp::Other
}

/// Resolution of one access base.
enum Base { Args { hi: DwSet, lo: DwSet }, Side(u8), Unresolved(&'static str) }
fn resolve_base(lo: &Val, hi: &Val) -> Base {
    if hi.src & ROOT_OVF != 0 || lo.src & ROOT_OVF != 0 { return Base::Unresolved("base-root-overflow"); }
    if hi.src & OPAQUE != 0 { return Base::Unresolved("base-memory-derived"); }
    let side = hi.src & (KSEG | APERTURE | PCREL | PACKET | SCRATCH);
    let categories = u32::from(!hi.roots.is_empty()) + side.count_ones();
    if categories == 0 { return Base::Unresolved("base-no-provenance"); }
    if categories > 1 { return Base::Unresolved("base-mixed"); }
    if !hi.roots.is_empty() { Base::Args { hi: hi.roots, lo: lo.roots } } else { Base::Side(side) }
}

/// Kernarg segment layout: declared arguments plus the dword → argument map.
struct Layout { args: Vec<Kernarg>, size: u32 }
impl Layout {
    fn arg_of(&self, dw: u32) -> Option<usize> {
        let byte = dw * 4;
        self.args.iter().position(|a| byte + 4 > a.offset && byte < a.offset + a.size.max(1))
    }
    fn is_pointer(&self, index: usize) -> bool {
        let a = &self.args[index];
        a.address_space.is_some() || a.value_kind == "global_buffer" || matches!(a.value_kind.as_str(),
            "hidden_printf_buffer" | "hidden_hostcall_buffer" | "hidden_default_queue" | "hidden_completion_action"
            | "hidden_multigrid_sync_arg" | "hidden_heap_v1" | "hidden_queue_ptr" | "dynamic_shared_pointer")
    }
}

/// A resolved argument target: a declared argument or a synthetic slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Target { Declared(usize), Synthetic(u32) }

struct Rec {
    accesses: BTreeMap<Target, Vec<ArgAccess>>,
    unresolved: Vec<Unresolved>,
    side: SideReads,
    loaded: DwSet,
    loaded_ovf: bool,
    dynamic: bool,
    sinks: Vec<u8>,
    taint_overflow: bool,
}
impl Rec {
    fn sink(&mut self, taint: DwSet, tf: u8, bit: u8) {
        for dw in taint.iter() { self.sinks[dw as usize] |= bit; }
        if tf & TAINT_OVF != 0 { self.taint_overflow = true; }
    }
    fn fail(&mut self, arch: Arch, dec: &Dec, rule: &'static str) {
        let text = dec.inst.text(arch).unwrap_or_else(|_| format!("{:?}", dec.inst.op));
        self.unresolved.push(Unresolved { inst: Some(dec.id), rule, detail: text });
    }
}

struct Analyzer<'k> {
    arch: Arch,
    wave: Wave,
    body: &'k Body,
    layout: Layout,
    entry: State,
    /// Blocks inside a CFG cycle (A4 loop-bound vs guard).
    cyclic: Vec<bool>,
    decoded: Vec<Option<Dec<'k>>>,
}

/// A1/A3/A4 facts for `kernel`. Never fails: an unsupported arch or ABI, or
/// an instruction the pass does not model, yields `KernelStatus::Unknown`
/// with the reason in `unresolved`.
pub fn analyze(kernel: &Kernel, arch: Arch) -> KernargFacts {
    let (args, size) = match &kernel.abi {
        Abi::Hsa { metadata, descriptor } => {
            let size = if metadata.parsed.kernarg_segment_size != 0 { metadata.parsed.kernarg_segment_size } else { descriptor.kernarg_size };
            (metadata.parsed.args.clone(), size)
        }
        Abi::Raw { .. } => (Vec::new(), 0),
    };
    let layout = Layout { args, size };
    let mut facts = KernargFacts { kernarg_size: size, ..KernargFacts::default() };
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        facts.unresolved.push(Unresolved { inst: None, rule: "unsupported-arch", detail: format!("{arch:?}") });
        return finish(facts, &layout, None);
    }
    let body = &kernel.body;
    if body.blocks.is_empty() {
        facts.unresolved.push(Unresolved { inst: None, rule: "blocks-not-built", detail: String::new() });
        return finish(facts, &layout, None);
    }
    let mut decoded = Vec::with_capacity(body.layout.len());
    for &id in &body.layout {
        decoded.push(body.insts.get(id).and_then(|inst| decode(arch, id, inst)));
    }
    let (entry, entry_loaded) = entry_state(kernel, arch);
    let mut an = Analyzer { arch, wave: kernel.wave, body, layout, entry, cyclic: cyclic_blocks(body), decoded };
    let mut rec = Rec {
        accesses: BTreeMap::new(), unresolved: Vec::new(), side: SideReads::default(), loaded: entry_loaded,
        loaded_ovf: false, dynamic: false, sinks: vec![0; MAX_DW as usize], taint_overflow: false,
    };
    let ins = an.fixpoint();
    for (b, block) in body.blocks.iter().enumerate() {
        let Some(mut st) = ins[b].clone() else { continue };
        for pos in block.range.0..block.range.1 {
            an.step(&mut st, pos, Some(&mut rec), an.cyclic[b]);
        }
    }
    facts.unresolved = std::mem::take(&mut rec.unresolved);
    let layout = std::mem::replace(&mut an.layout, Layout { args: Vec::new(), size: 0 });
    finish(facts, &layout, Some(rec))
}

fn cyclic_blocks(body: &Body) -> Vec<bool> {
    let mut graph: DiGraph<(), ()> = DiGraph::new();
    let nodes: Vec<NodeIndex> = body.blocks.iter().map(|_| graph.add_node(())).collect();
    for block in &body.blocks {
        for succ in &block.succs { graph.add_edge(nodes[block.id.0], nodes[succ.0], ()); }
    }
    let mut out = vec![false; body.blocks.len()];
    for scc in petgraph::algo::tarjan_scc(&graph) {
        let looped = scc.len() > 1 || graph.contains_edge(scc[0], scc[0]);
        if looped { for n in scc { out[n.index()] = true; } }
    }
    out
}

/// ABI entry values: user SGPRs by role, preloaded kernargs, workgroup ids
/// (gfx11 system SGPRs, gfx12 TTMP7/TTMP9), work-item ids in v0.
fn entry_state(kernel: &Kernel, arch: Arch) -> (State, DwSet) {
    let mut st = State::empty();
    let mut loaded = DwSet::default();
    let mut roles: Vec<(UserSgprRole, usize)> = Vec::new();
    let (user_count, preload, rsrc2) = match &kernel.abi {
        Abi::Hsa { descriptor, .. } => {
            let props = descriptor.kernel_code_properties.0;
            let table = [
                (UserSgprRole::PrivateSegmentBuffer, 4), (UserSgprRole::DispatchPtr, 2), (UserSgprRole::QueuePtr, 2),
                (UserSgprRole::KernargSegmentPtr, 2), (UserSgprRole::DispatchId, 2), (UserSgprRole::FlatScratchInit, 2),
                (UserSgprRole::PrivateSegmentSize, 1),
            ];
            for (bit, (role, n)) in table.into_iter().enumerate() {
                if props >> bit & 1 != 0 { roles.push((role, n)); }
            }
            let pre = u32::from(descriptor.kernarg_preload.0);
            let rsrc2 = descriptor.compute_pgm_rsrc2.0;
            ((rsrc2 >> 1 & 0x1f) as usize, (pre & 0x7f, pre >> 7), rsrc2)
        }
        Abi::Raw { user_sgprs, .. } => {
            for role in user_sgprs {
                let n = match role { UserSgprRole::PrivateSegmentBuffer => 4, UserSgprRole::PrivateSegmentSize => 1, _ => 2 };
                roles.push((role.clone(), n));
            }
            // Raw kernels get workgroup ids X/Y/Z after the user SGPRs.
            (roles.iter().map(|(_, n)| n).sum(), (0, 0), 0x380)
        }
    };
    let mut next = 0usize;
    for (role, n) in &roles {
        let val = match role {
            UserSgprRole::PrivateSegmentBuffer => Val::src(SCRATCH),
            UserSgprRole::DispatchPtr | UserSgprRole::QueuePtr => Val::src(PACKET),
            UserSgprRole::KernargSegmentPtr => Val { koff: Some(0), ..Val::src(KSEG) },
            UserSgprRole::FlatScratchInit => Val::src(SCRATCH),
            _ => Val::default(),
        };
        for i in 0..*n { if next + i < NS { st.s[next + i] = val; } }
        next += n;
    }
    // Kernarg preload (KERNARG_PRELOAD_SPEC_LENGTH/OFFSET, in dwords).
    let (count, offset) = preload;
    for i in 0..count {
        let dw = offset + i;
        if next < NS { st.s[next] = Val::karg(dw); }
        if dw < MAX_DW { loaded = loaded.union(DwSet::one(dw)); }
        next += 1;
    }
    if arch == Arch::Gfx1201 {
        st.t[9] = Val::flags(WG);
        st.t[7] = Val::flags(WG);
    } else {
        let mut sys = user_count.max(next);
        for bit in 7..=10 {
            if rsrc2 >> bit & 1 != 0 {
                if sys < NS { st.s[sys] = Val::flags(WG); }
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
    MaskResult { dst: SmallVec<[usize; 2]>, mask: Mask },
    CopyExec { dst: SmallVec<[usize; 2]> },
    Narrow,
    Set(Mask),
    SaveExec { dst: SmallVec<[usize; 2]>, op: SaveOp, src: Mask },
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
            let Some(mut st) = ins[b].clone() else { continue };
            let block = &self.body.blocks[b];
            for pos in block.range.0..block.range.1 { self.step(&mut st, pos, None, false); }
            for succ in &block.succs {
                let target = succ.0;
                let changed = match &mut ins[target] {
                    Some(existing) => {
                        let before = existing.clone();
                        existing.join_from(&st);
                        *existing != before
                    }
                    slot @ None => { *slot = Some(st.clone()); true }
                };
                if changed && !queued[target] { queued[target] = true; work.push(target); }
            }
        }
        ins
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
                        Kind::V => if idx < NV { st.read_v(idx as u16) } else { Val::default() },
                    });
                }
            }
            Operand::Special(sp) => {
                let one = |v: Val| -> SmallVec<[Val; 4]> { std::iter::repeat(v).take(width).collect() };
                return match sp {
                    Special::Vcc => [st.s[VCC_LO], st.s[VCC_HI]].into_iter().take(width.max(1)).collect(),
                    Special::VccLo => one(st.s[VCC_LO]),
                    Special::VccHi => one(st.s[VCC_HI]),
                    Special::Exec => st.exec.iter().copied().take(width).collect(),
                    Special::ExecLo => one(st.exec[0]),
                    Special::ExecHi => one(st.exec[1]),
                    Special::Scc => one(st.scc),
                    Special::M0 => one(st.m0),
                    Special::Null => one(Val::konst(0)),
                    Special::Ttmp(n) => one(st.t[usize::from(*n)]),
                    Special::SrcSharedBase | Special::SrcSharedLimit | Special::SrcPrivateBase | Special::SrcPrivateLimit => one(Val::src(APERTURE)),
                    Special::Pc => one(Val::src(PCREL)),
                    Special::FlatScratch => one(Val::src(SCRATCH)),
                };
            }
            other => {
                let value = inline_value(other);
                out.push(value.map(Val::konst).unwrap_or_default());
                for _ in 1..width {
                    let hi = value.map(|v| if other_is_negative(other) && v >> 31 != 0 { u32::MAX } else { 0 });
                    out.push(hi.map(Val::konst).unwrap_or_default());
                }
            }
        }
        while out.len() < width { let last = out.last().copied().unwrap_or_default(); out.push(last); }
        out
    }

    /// Write `vals` to `op`. VGPR writes are strong for the enabled lanes
    /// (disabled lanes live in `State::inactive`); half writes join.
    fn write(&self, st: &mut State, op: &Operand, vals: &[Val]) {
        let val_at = |i: usize| vals.get(i).copied().or_else(|| vals.last().copied()).unwrap_or_default();
        match op {
            Operand::Reg(r) => {
                for i in 0..usize::from(r.len) {
                    let idx = usize::from(r.base) + i;
                    let val = val_at(i);
                    match r.kind {
                        Kind::S => if idx < NS { st.s[idx] = val; st.masks[idx] = Mask::Unknown; },
                        Kind::Ttmp => if idx < 16 { st.t[idx] = val; },
                        Kind::V => if idx < NV { self.write_v(st, idx as u16, val, true); },
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
                Special::Vcc => { st.s[VCC_LO] = val_at(0); st.s[VCC_HI] = val_at(1); st.masks[VCC_LO] = Mask::Unknown; st.masks[VCC_HI] = Mask::Unknown; }
                Special::VccLo => { st.s[VCC_LO] = val_at(0); st.masks[VCC_LO] = Mask::Unknown; }
                Special::VccHi => { st.s[VCC_HI] = val_at(0); st.masks[VCC_HI] = Mask::Unknown; }
                Special::Exec => { st.exec = [val_at(0), val_at(1)]; }
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
        let keys: SmallVec<[(u16, u8); 4]> = st.lanes.range((r, 0)..=(r, u8::MAX)).map(|(k, _)| *k).collect();
        if strong {
            st.v[usize::from(r)] = val;
            for k in keys { st.lanes.remove(&k); }
        } else {
            st.v[usize::from(r)] = st.v[usize::from(r)].join(val);
            for k in keys { let old = st.lanes[&k]; st.lanes.insert(k, old.join(val)); }
        }
    }

    fn vals(&self, st: &State, f: &Field) -> SmallVec<[Val; 4]> { self.read(st, f.op, f.bits) }

    /// Transfer one instruction; with `rec`, also record facts.
    fn step(&self, st: &mut State, pos: usize, mut rec: Option<&mut Rec>, cyclic: bool) {
        let Some(dec) = &self.decoded[pos] else {
            if let Some(rec) = rec.as_deref_mut() {
                let id = self.body.layout[pos];
                rec.unresolved.push(Unresolved { inst: Some(id), rule: "undecodable", detail: String::new() });
            }
            return;
        };
        let inst = dec.inst;
        let name = dec.parts[0].name;
        if let Some(rec) = rec.as_deref_mut() { self.record_sinks(st, dec, rec, cyclic); }

        if name == "s_setpc_b64" || name == "s_swappc_b64" || name.starts_with("s_call") || name == "s_rfe_b64" {
            if let Some(rec) = rec.as_deref_mut() { rec.fail(self.arch, dec, "call"); }
        }
        if inst.effects.mem.is_some() || matches!(inst.form, Form::Vmem(_) | Form::Smem | Form::Ds) {
            self.memory(st, dec, rec.as_deref_mut());
        } else {
            // Evaluate every part against the pre-state (VOPD reads both halves first).
            let mut writes: SmallVec<[(&Operand, SmallVec<[Val; 4]>); 4]> = SmallVec::new();
            let mut lane_writes: SmallVec<[(u16, Option<u8>, Val); 1]> = SmallVec::new();
            for part in &dec.parts { self.eval(st, part, &mut writes, &mut lane_writes); }
            let plan = self.exec_plan(st, dec);
            let valu = matches!(inst.form, Form::Vop1 | Form::Vop1Dpp | Form::Vop2 | Form::Vop2Dpp | Form::Vop3 | Form::Vop3p | Form::Vopc | Form::Vopd);
            for (op, vals) in &writes {
                // A wave32 VALU SGPR destination (carry-out, compare mask) is
                // one SGPR even where the grammar types it 64-bit.
                match op {
                    Operand::Reg(r) if valu && self.wave == Wave::Wave32 && r.kind == Kind::S && r.len == 2 => {
                        let low = Operand::Reg(crate::reg::RegRef { len: 1, ..*r });
                        self.write(st, &low, vals);
                    }
                    _ => self.write(st, op, vals),
                }
            }
            for (r, lane, val) in lane_writes {
                match lane {
                    Some(l) => { st.lanes.insert((r, l), val); }
                    None => self.write_v(st, r, val, false),
                }
            }
            self.apply_plan(st, plan);
        }
        if inst.effects.implicit.writes & ImplicitSet::SCC != 0 { st.scc = Val::default(); }
        if inst.effects.implicit.writes & ImplicitSet::VCC != 0 && !dec.parts.iter().any(|p| p.fields.iter().any(|f| f.def && matches!(f.op, Operand::Special(Special::Vcc | Special::VccLo | Special::VccHi)))) {
            st.s[VCC_LO] = Val::default();
            if self.wave == Wave::Wave64 { st.s[VCC_HI] = Val::default(); }
            if !name.starts_with("v_cmp_") { st.masks[VCC_LO] = Mask::Unknown; st.masks[VCC_HI] = Mask::Unknown; }
        }
        if inst.effects.implicit.writes & ImplicitSet::M0 != 0 && !dec.parts.iter().any(|p| p.fields.iter().any(|f| f.def && matches!(f.op, Operand::Special(Special::M0)))) {
            st.m0 = Val::default();
        }
    }

    /// The mask value of an operand (for EXEC and saved-mask arithmetic).
    fn mask_of(&self, st: &State, op: &Operand) -> Mask {
        let sgprs = |base: usize, len: usize| -> Mask {
            let first = st.masks.get(base).copied().unwrap_or(Mask::Unknown);
            if (base..base + len).all(|i| st.masks.get(i) == Some(&first)) { first } else { Mask::Unknown }
        };
        match op {
            Operand::Special(Special::Exec | Special::ExecLo) => {
                let top = st.top();
                Mask::Within { level: top, exact: st.exec_exact == Some(top) }
            }
            Operand::Special(Special::Vcc | Special::VccLo) => sgprs(VCC_LO, if self.wave == Wave::Wave64 { 2 } else { 1 }),
            Operand::Reg(r) if r.kind == Kind::S => {
                let len = if self.wave == Wave::Wave32 { 1 } else { usize::from(r.len) };
                sgprs(usize::from(r.base), len)
            }
            other => match inline_value(other) {
                Some(u32::MAX) => Mask::Full,
                Some(0) => Mask::Within { level: st.top(), exact: false },
                _ => Mask::Unknown,
            },
        }
    }

    /// How this instruction changes EXEC and the mask facts of its SGPR
    /// destinations, computed on the pre-state.
    fn exec_plan(&self, st: &State, dec: &Dec) -> ExecPlan {
        let part = &dec.parts[0];
        let name = part.name;
        let is_exec = |op: &Operand| matches!(op, Operand::Special(Special::Exec | Special::ExecLo | Special::ExecHi));
        let sgprs_of = |op: &Operand| -> SmallVec<[usize; 2]> {
            match op {
                Operand::Reg(r) if r.kind == Kind::S => {
                    let len = if self.wave == Wave::Wave32 { 1 } else { usize::from(r.len) };
                    (0..len).map(|i| usize::from(r.base) + i).filter(|&i| i < NS).collect()
                }
                Operand::Special(Special::VccLo) => smallvec::smallvec![VCC_LO],
                Operand::Special(Special::Vcc) => if self.wave == Wave::Wave64 { smallvec::smallvec![VCC_LO, VCC_HI] } else { smallvec::smallvec![VCC_LO] },
                _ => SmallVec::new(),
            }
        };
        let srcs: SmallVec<[&Field; 2]> = part.sources().collect();
        let mask = |i: usize| srcs.get(i).map_or(Mask::Unknown, |f| self.mask_of(st, f.op));
        let algebra = || -> Option<Mask> {
            Some(if name.starts_with("s_mov_b") { mask(0) }
                else if name.starts_with("s_and_b") { mask(0).and(mask(1)) }
                else if name.starts_with("s_andn2_b") || name.starts_with("s_and_not1_b") { mask(0).and_not(mask(1)) }
                else if name.starts_with("s_or_b") { mask(0).or(mask(1)) }
                else if name.starts_with("s_xor_b") { mask(0).xor(mask(1)) }
                else { return None })
        };
        if name.contains("_saveexec_") {
            let dst = part.field("SDST").map(|f| sgprs_of(f.op)).unwrap_or_default();
            let op = if name.starts_with("s_and_saveexec") || name.starts_with("s_and_not0_saveexec") || name.starts_with("s_andn1_saveexec") { SaveOp::Narrow }
                else if name.starts_with("s_or_saveexec") { SaveOp::Or }
                else if name.starts_with("s_xor_saveexec") { SaveOp::Xor }
                else if name.starts_with("s_and_not1_saveexec") || name.starts_with("s_andn2_saveexec") { SaveOp::SrcAndNotExec }
                else { SaveOp::Other };
            return ExecPlan::SaveExec { dst, op, src: mask(0) };
        }
        if part.fields.iter().any(|f| f.def && is_exec(f.op)) {
            let narrows = (name.starts_with("s_and_b") && srcs.iter().any(|f| is_exec(f.op)))
                || ((name.starts_with("s_andn2_b") || name.starts_with("s_and_not1_b")) && srcs.first().is_some_and(|f| is_exec(f.op)));
            if narrows { return ExecPlan::Narrow; }
            return ExecPlan::Set(algebra().unwrap_or(Mask::Unknown));
        }
        if dec.inst.effects.implicit.writes & ImplicitSet::EXEC != 0 {
            return if name.starts_with("v_cmpx") { ExecPlan::Narrow } else { ExecPlan::Set(Mask::Unknown) };
        }
        // Mask facts for SGPR destinations.
        if name.starts_with("v_cmp_") {
            let dst = match part.fields.iter().find(|f| f.def) {
                Some(f) => sgprs_of(f.op),
                None => sgprs_of(&Operand::Special(Special::Vcc)),
            };
            return ExecPlan::MaskResult { dst, mask: Mask::Within { level: st.top(), exact: false } };
        }
        let Some(dst) = part.field("SDST").map(|f| sgprs_of(f.op)).filter(|d| !d.is_empty()) else { return ExecPlan::None };
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
            ExecPlan::MaskResult { dst, mask } => for d in dst { st.masks[d] = mask; },
            ExecPlan::CopyExec { dst } => {
                let level = st.exec_copy_level();
                for d in dst { st.masks[d] = Mask::Within { level, exact: true }; }
            }
            ExecPlan::Narrow => st.exec_narrow(),
            ExecPlan::Set(mask) => st.exec_set(mask),
            ExecPlan::SaveExec { dst, op, src } => {
                let level = st.exec_copy_level();
                let saved = Mask::Within { level, exact: true };
                for d in dst { st.masks[d] = saved; }
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
    fn eval<'a>(&self, st: &State, part: &Part<'a>, writes: &mut SmallVec<[(&'a Operand, SmallVec<[Val; 4]>); 4]>, lanes: &mut SmallVec<[(u16, Option<u8>, Val); 1]>) {
        let start = writes.len();
        let lane_start = lanes.len();
        self.eval_rules(st, part, writes, lanes);
        let uses: SmallVec<[Val; 8]> = part.sources().flat_map(|f| self.vals(st, f)).collect();
        for d in part.fields.iter().filter(|f| f.def) {
            let written = writes[start..].iter().any(|(op, _)| std::ptr::eq(*op, d.op)) || lanes.len() > lane_start;
            if !written { writes.push((d.op, smallvec::smallvec![Val::kill(uses.iter())])); }
        }
    }

    fn eval_rules<'a>(&self, st: &State, part: &Part<'a>, writes: &mut SmallVec<[(&'a Operand, SmallVec<[Val; 4]>); 4]>, lanes: &mut SmallVec<[(u16, Option<u8>, Val); 1]>) {
        let name = part.name;
        let defs: SmallVec<[&Field<'a>; 2]> = part.fields.iter().filter(|f| f.def).collect();
        let uses: SmallVec<[&Field<'a>; 4]> = part.sources().collect();
        if defs.is_empty() { return; }
        match name {
            "v_writelane_b32" => {
                let (Some(dst), Some(src), Some(sel)) = (part.field("VDST"), part.field("SRC0"), part.field("SRC1")) else { return };
                let Operand::Reg(r) = dst.op else { return };
                let val = self.vals(st, src)[0];
                lanes.push((r.base, lane_of(st, sel.op), val));
                return;
            }
            "v_readlane_b32" => {
                let (Some(dst), Some(src), Some(sel)) = (part.field("VDST"), part.field("SRC0"), part.field("SRC1")) else { return };
                let val = match (src.op, lane_of(st, sel.op)) {
                    (Operand::Reg(r), Some(l)) if r.kind == Kind::V => st.lanes.get(&(r.base, l)).copied()
                        .unwrap_or_else(|| st.v[usize::from(r.base)].join(st.disabled_v(r.base))),
                    (Operand::Reg(r), None) if r.kind == Kind::V => st.read_v(r.base).join(st.disabled_v(r.base)),
                    _ => self.vals(st, src)[0],
                };
                writes.push((dst.op, smallvec::smallvec![val]));
                return;
            }
            "s_getpc_b64" => {
                writes.push((defs[0].op, smallvec::smallvec![Val::src(PCREL), Val::src(PCREL)]));
                return;
            }
            _ => {}
        }
        if is_mov(name) {
            if let Some(src) = part.first_use().or_else(|| part.fields.iter().find(|f| !f.def)) {
                writes.push((defs[0].op, self.vals(st, src)));
            }
            return;
        }
        if is_select(name) {
            let cands: SmallVec<[SmallVec<[Val; 4]>; 2]> = uses.iter().filter(|f| f.name != "SRC2").map(|f| self.vals(st, f)).collect();
            let width = cands.iter().map(SmallVec::len).max().unwrap_or(1);
            let out = (0..width).map(|i| {
                cands.iter().map(|c| c[i.min(c.len() - 1)]).reduce(Val::join).unwrap_or_default()
            }).collect();
            writes.push((defs[0].op, out));
            return;
        }
        if name.starts_with("s_cmov_") {
            let (Some(dst), Some(src)) = (part.field("SDST"), part.field("SSRC0")) else { return };
            let old = self.vals(st, dst);
            let new = self.vals(st, src);
            let out = old.iter().enumerate().map(|(i, a)| a.join(new[i.min(new.len() - 1)])).collect();
            writes.push((dst.op, out));
            return;
        }
        let all_uses: SmallVec<[Val; 8]> = uses.iter().flat_map(|f| self.vals(st, f)).collect();
        if is_mad64(name) {
            let taint = Val::kill(all_uses.iter());
            let c = part.field("SRC2").map(|f| self.vals(st, f)).unwrap_or_default();
            let none = Val::default();
            for d in &defs {
                if d.name == "SDST" { writes.push((d.op, smallvec::smallvec![taint])); continue; }
                let lo = Val::mix([&taint, c.first().unwrap_or(&none)]);
                let hi = Val::mix([&taint, c.get(1).unwrap_or(&none)]);
                writes.push((d.op, smallvec::smallvec![lo, hi]));
            }
            return;
        }
        if let Some(keep) = keep_field(name) {
            let taint = Val::kill(all_uses.iter());
            let kept = part.field(keep).map(|f| self.vals(st, f)).unwrap_or_default();
            let out = Val::mix(std::iter::once(&taint).chain(kept.iter()));
            for d in &defs { writes.push((d.op, smallvec::smallvec![out])); }
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
            for d in &defs { writes.push((d.op, smallvec::smallvec![out])); }
            return;
        }
        // Provenance-keeping arithmetic, dword by dword when widths agree.
        let use_vals: SmallVec<[SmallVec<[Val; 4]>; 4]> = uses.iter().map(|f| self.vals(st, f)).collect();
        let has_vdst = defs.iter().any(|f| f.name == "VDST");
        let carry_in = name.starts_with("s_addc_u32") || name.starts_with("s_add_co_ci_u32") || name.starts_with("v_add_co_ci_u32");
        for d in &defs {
            if d.name == "SDST" && has_vdst {
                // Carry-out of a VOP3b add/sub: a lane mask, never a pointer.
                writes.push((d.op, smallvec::smallvec![Val::kill(all_uses.iter())]));
                continue;
            }
            let width = match d.op { Operand::Reg(r) => usize::from(r.len), _ => usize::from((d.bits / 32).max(1)) };
            let per_dword = width > 1 && use_vals.iter().all(|v| v.len() == width);
            let mut out: SmallVec<[Val; 4]> = SmallVec::new();
            for i in 0..width {
                let lane: SmallVec<[&Val; 4]> = if per_dword { use_vals.iter().map(|v| &v[i]).collect() } else { all_uses.iter().collect() };
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
            let uses: SmallVec<[SmallVec<[Val; 4]>; 4]> = part.fields.iter().filter(|f| f.used && !is_mask(name, f.name)).map(|f| self.vals(st, f)).collect();
            if name.starts_with("s_cmp") || name.starts_with("v_cmp") || name.starts_with("s_bitcmp") {
                let bit = if cyclic { Sinks::LOOP_BOUND } else { Sinks::GUARD };
                for v in uses.iter().flatten() { rec.sink(v.taint, v.tf, bit); }
                continue;
            }
            if dec.inst.effects.mem.is_some() { continue; }
            let all = Val::mix(uses.iter().flatten());
            if all.tf & (WI | WG) != 0 {
                let roots = uses.iter().flatten().fold(DwSet::default(), |acc, v| acc.union(v.roots));
                rec.sink(all.taint.minus(roots), all.tf, Sinks::GRID_TERM);
            }
        }
    }

    fn memory(&self, st: &mut State, dec: &Dec, mut rec: Option<&mut Rec>) {
        let inst = dec.inst;
        let part = &dec.parts[0];
        let name = part.name;
        let op = mem_op(name);
        let cache = CacheBits { th: inst.mods.cpol.th, scope: inst.mods.cpol.scope, glc: inst.mods.cpol.glc, slc: inst.mods.cpol.slc, dlc: inst.mods.cpol.dlc, nv: inst.mods.cpol.nv };
        let dst = part.fields.iter().find(|f| f.def);
        let data_fields: SmallVec<[&Field; 2]> = part.fields.iter().filter(|f| !f.def && matches!(f.name, "VSRC" | "VDATA" | "DATA" | "DATA0" | "DATA1" | "SDATA") && f.used).collect();
        match inst.form {
            Form::Ds => {
                let passthrough = name.starts_with("ds_swizzle") || name.starts_with("ds_bpermute") || name.starts_with("ds_permute");
                if let Some(rec) = rec.as_deref_mut() {
                    if let Some(addr) = part.field("ADDR") { for v in self.vals(st, addr) { rec.sink(v.taint, v.tf, Sinks::LDS_ADDRESS); } }
                    if !passthrough { for f in &data_fields { for v in self.vals(st, f) { rec.sink(v.taint, v.tf, Sinks::STORED); } } }
                }
                if let Some(d) = dst {
                    let width = match d.op { Operand::Reg(r) => usize::from(r.len), _ => 1 };
                    let val = if passthrough { Val::mix(data_fields.iter().flat_map(|f| self.vals(st, f)).collect::<SmallVec<[Val; 4]>>().iter()) } else { Val::loaded() };
                    let vals: SmallVec<[Val; 4]> = std::iter::repeat(val).take(width).collect();
                    self.write(st, d.op, &vals);
                }
                return;
            }
            Form::Smem => {
                if op != MemOp::Load {
                    if let Some(d) = dst { self.write(st, d.op, &[Val::flags(MEM)]); }
                    return;
                }
                let Some(sbase) = part.field("SBASE") else { return };
                let base = self.vals(st, sbase);
                let width = dst.map_or(1, |d| match d.op { Operand::Reg(r) => usize::from(r.len), _ => 1 });
                let mut offset: Option<i64> = Some(0);
                let mut offset_vals: SmallVec<[Val; 2]> = SmallVec::new();
                for f in part.fields.iter().filter(|f| f.name == "SOFFSET") {
                    match f.op {
                        Operand::Imm(ImmField::SmemOffset(n)) => offset = offset.map(|o| o + i64::from(*n)),
                        Operand::Special(Special::Null) => {}
                        other => {
                            let v = self.read(st, other, 32)[0];
                            offset_vals.push(v);
                            offset = match (offset, v.konst) { (Some(o), Some(c)) => Some(o + i64::from(c)), _ => None };
                        }
                    }
                }
                for e in &dec.extras { if let Operand::Imm(ImmField::SmemDisplacement(n) | ImmField::SmemOffset(n)) = **e { offset = offset.map(|o| o + i64::from(n)); } }
                let buffer = name.starts_with("s_buffer_load");
                let (lo, hi) = if buffer { (base.first().copied().unwrap_or_default(), base.get(1).copied().unwrap_or_default()) }
                    else { (base[0], base.get(1).copied().unwrap_or_default()) };
                let mut vals: SmallVec<[Val; 8]> = std::iter::repeat(Val::loaded()).take(width).collect();
                let resolved = resolve_base(&lo, &hi);
                if let Base::Side(side) = resolved {
                    if side == KSEG && !buffer {
                        match (lo.koff, offset) {
                            (Some(k), Some(o)) => {
                                let start = k + o;
                                for (i, v) in vals.iter_mut().enumerate() {
                                    let byte = start + 4 * i as i64;
                                    if byte < 0 { *v = Val::any_karg(); continue; }
                                    let dw = (byte / 4) as u32;
                                    *v = if byte % 4 == 0 { Val::karg(dw) } else { Val::karg(dw).join(Val::karg(dw + 1)) };
                                    if let Some(rec) = rec.as_deref_mut() {
                                        for d in [dw, dw + u32::from(byte % 4 != 0)] {
                                            if d < MAX_DW { rec.loaded = rec.loaded.union(DwSet::one(d)); } else { rec.loaded_ovf = true; }
                                        }
                                    }
                                }
                            }
                            _ => {
                                for v in vals.iter_mut() { *v = Val::any_karg(); }
                                if let Some(rec) = rec.as_deref_mut() { rec.dynamic = true; }
                            }
                        }
                        if let Some(d) = dst { self.write(st, d.op, &vals); }
                        return;
                    }
                }
                if let Some(rec) = rec.as_deref_mut() {
                    let path = if buffer { MemPath::SmemBuffer } else { MemPath::Smem };
                    self.attribute(rec, dec, resolved, &lo, &hi, &offset_vals, AccessClass::SmemLoad, path, cache);
                }
                if let Some(d) = dst { self.write(st, d.op, &vals); }
                return;
            }
            Form::Vmem(VmemForm::Image) | Form::Export => {
                if let Some(rec) = rec.as_deref_mut() {
                    if inst.form != Form::Export { rec.fail(self.arch, dec, "unmodeled-image"); }
                }
                if let Some(d) = dst { let w = match d.op { Operand::Reg(r) => usize::from(r.len), _ => 1 }; let vals: SmallVec<[Val; 4]> = std::iter::repeat(Val::loaded()).take(w).collect(); self.write(st, d.op, &vals); }
                return;
            }
            Form::Vmem(form) => {
                let loaded_width = dst.map_or(0, |d| match d.op { Operand::Reg(r) => usize::from(r.len), _ => 1 });
                let result: SmallVec<[Val; 4]> = std::iter::repeat(Val::loaded()).take(loaded_width.max(1)).collect();
                if op == MemOp::Other {
                    if let Some(d) = dst { self.write(st, d.op, &result); }
                    return;
                }
                if form == VmemForm::Scratch {
                    if let Some(rec) = rec.as_deref_mut() { rec.side.scratch += 1; for f in &data_fields { for v in self.vals(st, f) { rec.sink(v.taint, v.tf, Sinks::STORED); } } }
                    if let Some(d) = dst { self.write(st, d.op, &result); }
                    return;
                }
                let (lo, hi, offsets) = match form {
                    VmemForm::Buffer => {
                        let rsrc = part.field("RSRC").map(|f| self.vals(st, f)).unwrap_or_default();
                        let mut offs: SmallVec<[Val; 4]> = SmallVec::new();
                        for n in ["VADDR", "SOFFSET"] { if let Some(f) = part.field(n) { offs.extend(self.vals(st, f)); } }
                        if let Some(v) = rsrc.get(2) { offs.push(*v); }
                        (rsrc.first().copied().unwrap_or_default(), rsrc.get(1).copied().unwrap_or_default(), offs)
                    }
                    _ => {
                        let saddr = part.field("SADDR").filter(|f| matches!(f.op, Operand::Reg(_)));
                        let vaddr = part.field("VADDR").or_else(|| part.field("ADDR"));
                        match saddr {
                            Some(s) => {
                                let base = self.vals(st, s);
                                let offs = vaddr.map(|f| self.vals(st, f)).unwrap_or_default();
                                (base[0], base.get(1).copied().unwrap_or_default(), offs)
                            }
                            None => {
                                let base = vaddr.map(|f| self.vals(st, f)).unwrap_or_default();
                                (base.first().copied().unwrap_or_default(), base.get(1).copied().unwrap_or_default(), SmallVec::new())
                            }
                        }
                    }
                };
                let path = match form { VmemForm::Buffer => MemPath::Buffer, VmemForm::Flat => MemPath::Flat, _ => MemPath::Global };
                let class = match op {
                    MemOp::Load => AccessClass::VmemLoad,
                    MemOp::Store => AccessClass::VmemStore,
                    MemOp::Atomic { commutative } => AccessClass::Atomic { commutative },
                    MemOp::LdsDma => AccessClass::LdsDma,
                    MemOp::Other => unreachable!(),
                };
                let resolved = resolve_base(&lo, &hi);
                let mut out = result;
                if let Base::Side(KSEG) = resolved {
                    if matches!(op, MemOp::Load) {
                        out = std::iter::repeat(Val::any_karg()).take(loaded_width.max(1)).collect();
                    }
                }
                if let Some(rec) = rec.as_deref_mut() {
                    for f in &data_fields { for v in self.vals(st, f) { rec.sink(v.taint, v.tf, Sinks::STORED); } }
                    self.attribute(rec, dec, resolved, &lo, &hi, &offsets, class, path, cache);
                }
                if let Some(d) = dst { self.write(st, d.op, &out); }
            }
            _ => {
                if let Some(d) = dst { self.write(st, d.op, &[Val::flags(MEM)]); }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attribute(&self, rec: &mut Rec, dec: &Dec, resolved: Base, lo: &Val, hi: &Val, offsets: &[Val], class: AccessClass, path: MemPath, cache: CacheBits) {
        let writes = class.mode().writes();
        let offset_taint = offsets.iter().fold((DwSet::default(), 0u8), |(t, f), v| (t.union(v.taint), f | v.tf));
        match resolved {
            Base::Unresolved(rule) => {
                rec.fail(self.arch, dec, rule);
                rec.sink(lo.taint.union(hi.taint).union(offset_taint.0), lo.tf | hi.tf | offset_taint.1, Sinks::ADDRESS_OFFSET);
            }
            Base::Side(side) => {
                match side {
                    KSEG => if writes { rec.fail(self.arch, dec, "kernarg-write") } else {
                        rec.side.kernarg_segment += 1;
                        rec.dynamic = true;
                    },
                    PCREL => if writes { rec.fail(self.arch, dec, "module-global-write") } else { rec.side.module_global += 1 },
                    PACKET => if writes { rec.fail(self.arch, dec, "abi-packet-write") } else { rec.side.abi_packet += 1 },
                    APERTURE => if path == MemPath::Flat { rec.side.local_flat += 1 } else { rec.fail(self.arch, dec, "aperture-non-flat") },
                    SCRATCH => rec.side.scratch += 1,
                    _ => rec.fail(self.arch, dec, "base-mixed"),
                }
                rec.sink(lo.taint.union(hi.taint).union(offset_taint.0), lo.tf | hi.tf | offset_taint.1, Sinks::ADDRESS_OFFSET);
            }
            Base::Args { hi: hi_roots, lo: lo_roots } => {
                let mut declared: SmallVec<[usize; 2]> = SmallVec::new();
                let mut synthetic: SmallVec<[u32; 2]> = SmallVec::new();
                for dw in hi_roots.iter() {
                    match self.layout.arg_of(dw) { Some(i) => if !declared.contains(&i) { declared.push(i) }, None => { let slot = (dw * 4) & !7; if !synthetic.contains(&slot) { synthetic.push(slot) } } }
                }
                // A pointer argument among the high-dword candidates rules out
                // by-value candidates (they carry no allocation).
                if declared.iter().any(|&i| self.layout.is_pointer(i)) {
                    declared.retain(|&mut i| self.layout.is_pointer(i));
                    synthetic.clear();
                }
                for dw in lo_roots.iter() {
                    if let Some(i) = self.layout.arg_of(dw) { if self.layout.is_pointer(i) && !declared.contains(&i) { declared.push(i); } }
                }
                let mut targets: SmallVec<[Target; 4]> = declared.iter().map(|&i| Target::Declared(i)).collect();
                targets.extend(synthetic.iter().map(|&s| Target::Synthetic(s)));
                let shared = targets.len() > 1;
                let mut base_dws = DwSet::default();
                for t in &targets {
                    let (off, size) = match *t { Target::Declared(i) => (self.layout.args[i].offset, self.layout.args[i].size.max(4)), Target::Synthetic(s) => (s, 8) };
                    for dw in off / 4..(off + size).div_ceil(4) { if dw < MAX_DW { base_dws = base_dws.union(DwSet::one(dw)); } }
                    rec.accesses.entry(*t).or_default().push(ArgAccess { inst: dec.id, class, path, cache, shared });
                }
                rec.sink(base_dws, 0, Sinks::ADDRESS_BASE);
                rec.sink(lo.taint.union(hi.taint).union(offset_taint.0).minus(base_dws), lo.tf | hi.tf | offset_taint.1, Sinks::ADDRESS_OFFSET);
            }
        }
    }
}

fn other_is_negative(op: &Operand) -> bool { matches!(op, Operand::Inline(InlineConst::Integer(n)) if *n < 0) }

fn read_cache_of(accesses: &[ArgAccess]) -> ReadCacheClass {
    accesses.iter().fold(ReadCacheClass::NotRead, |acc, a| acc.join(match a.class {
        AccessClass::SmemLoad => ReadCacheClass::ScalarOnly,
        AccessClass::VmemLoad | AccessClass::LdsDma => ReadCacheClass::VmemOnly,
        AccessClass::VmemStore | AccessClass::Atomic { .. } => ReadCacheClass::NotRead,
    }))
}

fn finish(mut facts: KernargFacts, layout: &Layout, rec: Option<Rec>) -> KernargFacts {
    let rec = rec.unwrap_or(Rec {
        accesses: BTreeMap::new(), unresolved: Vec::new(), side: SideReads::default(), loaded: DwSet::default(),
        loaded_ovf: false, dynamic: false, sinks: vec![0; MAX_DW as usize], taint_overflow: false,
    });
    facts.status = if facts.unresolved.is_empty() { KernelStatus::Proven } else { KernelStatus::Unknown };
    facts.side = rec.side;
    facts.dynamic_kernarg_reads = rec.dynamic;
    let dword_count = layout.size.div_ceil(4);
    let loaded = |dw: u32| rec.dynamic || if dw < MAX_DW { rec.loaded.contains(dw) } else { rec.loaded_ovf };
    let sinks_of = |dw: u32| if dw < MAX_DW { rec.sinks[dw as usize] } else { 0 };
    facts.dwords = (0..dword_count).map(|dw| DwordFacts { offset: dw * 4, loaded: loaded(dw), sinks: Sinks(sinks_of(dw)) }).collect();
    let span = |offset: u32, size: u32| offset / 4..(offset + size.max(1)).div_ceil(4);
    let make = |target: Target, arg: Option<(usize, &Kernarg)>| -> ArgFacts {
        let (offset, size, index, name, kind) = match (target, arg) {
            (_, Some((i, a))) => (a.offset, a.size, Some(i), a.name.clone(), a.value_kind.clone()),
            (Target::Synthetic(s), None) => (s, 8, None, format!("synthetic@{s}"), "synthetic".to_owned()),
            (Target::Declared(_), None) => unreachable!(),
        };
        let accesses = rec.accesses.get(&target).cloned().unwrap_or_default();
        let dws = span(offset, size);
        let was_loaded = dws.clone().any(loaded);
        let sinks = Sinks(dws.clone().fold(0, |acc, dw| acc | sinks_of(dw)));
        let mode = accesses.iter().map(|a| a.class.mode()).reduce(AccessMode::join);
        let role = if !accesses.is_empty() { ArgRole::Pointer } else if was_loaded { ArgRole::Integer } else { ArgRole::NotLoaded };
        ArgFacts {
            offset, size, index, name, value_kind: kind, role, mode,
            bound: mode.map(|_| RegionBound::AllocationWide), read_cache: read_cache_of(&accesses),
            accesses, sinks, loaded: was_loaded,
        }
    };
    let mut args: Vec<ArgFacts> = layout.args.iter().enumerate().map(|(i, a)| make(Target::Declared(i), Some((i, a)))).collect();
    for target in rec.accesses.keys() {
        if let Target::Synthetic(_) = target { args.push(make(*target, None)); }
    }
    let mut implicit = ImplicitUse::default();
    for a in &args {
        if !a.is_hidden() || !a.loaded { continue; }
        implicit.hidden_loaded.push(a.value_kind.clone());
        if HIDDEN_GRID_KINDS.contains(&a.value_kind.as_str()) {
            implicit.hidden_grid.push(a.value_kind.clone());
            if a.sinks.0 != 0 { implicit.hidden_grid_live.push(a.value_kind.clone()); }
        }
    }
    facts.implicit = implicit;
    facts.read_cache = if facts.status == KernelStatus::Unknown { ReadCacheClass::Unknown }
        else { args.iter().fold(ReadCacheClass::NotRead, |acc, a| acc.join(a.read_cache)) };
    facts.taint_overflow = rec.taint_overflow;
    facts.args = args;
    facts
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Body;
    use crate::descriptor::{KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3};
    use crate::inst::{KernelOrigin, SymbolId};
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
        let size = args.iter().map(|a| a.1 + a.2).max().unwrap_or(0).div_ceil(8) * 8;
        let descriptor = KernelDescriptor { group_segment_fixed_size: 0, private_segment_fixed_size: 0, kernarg_size: size,
            kernel_code_entry_byte_offset: 0, compute_pgm_rsrc3: Rsrc3(0), compute_pgm_rsrc1: Rsrc1(0x1f), compute_pgm_rsrc2: Rsrc2(0x384),
            kernel_code_properties: KernelCodeProperties(0x408), kernarg_preload: KernargPreload(0), reserved: [0; 28] };
        let parsed = KernelMeta {
            name: "k".into(), symbol: "k.kd".into(), kernarg_segment_size: size, kernarg_segment_align: 8, wavefront_size: 32,
            args: args.iter().map(|&(name, offset, size, kind)| Kernarg {
                name: name.into(), offset, size, value_kind: kind.into(),
                address_space: (kind == "global_buffer").then(|| "global".to_owned()),
            }).collect(),
            ..Default::default()
        };
        Kernel { symbol: SymbolId("k".into()), wave: Wave::Wave32,
            abi: Abi::Hsa { descriptor, metadata: HsaKernelMetadata { raw_msgpack: Vec::new(), parsed } }, body,
            origin: KernelOrigin::Authored { builder_crate: "test".into(), version: "0".into(), git: "0".into() } }
    }
    fn facts(words: &[u32], args: &[(&str, u32, u32, &str)]) -> KernargFacts { analyze(&kernel(words, args), Arch::Gfx1201) }
    fn mode(f: &KernargFacts, offset: u32) -> Option<AccessMode> { f.arg_at(offset).and_then(|a| a.mode) }
    fn rules(f: &KernargFacts) -> Vec<&'static str> { f.unresolved.iter().map(|u| u.rule).collect() }
    const PTRS: &[(&str, u32, u32, &str)] = &[("a", 0, 8, "global_buffer"), ("b", 8, 8, "global_buffer")];

    #[test]
    fn base_saved_and_restored_through_lanes_is_attributed() {
        // s_load_b128 s[4:7], s[0:1]; v_writelane_b32 v40, s6, 0; v_writelane_b32 v40, s7, 1;
        // s_mov_b64 s[6:7], 0; v_readlane_b32 s10, v40, 0; v_readlane_b32 s11, v40, 1;
        // global_load_b32 v1, v0, s[4:5]; global_store_b32 v0, v1, s[10:11]
        let f = facts(&[0xf4004100, 0xf8000000, 0xbfc70000, 0xd7610028, 0x02010006, 0xd7610028, 0x02010207, 0xbe860180,
            0xd760000a, 0x02010128, 0xd760000b, 0x02010328, 0x7e000280, 0xee050004, 0x00000001, 0x00000000, 0xbfc00000,
            0xee06800a, 0x00800000, 0x00000000, 0xbfb00000], PTRS);
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!((mode(&f, 0), mode(&f, 8)), (Some(AccessMode::Read), Some(AccessMode::Write)));
        assert!(f.args.iter().all(|a| a.accesses.iter().all(|x| !x.shared)));
        assert_eq!(f.arg_at(8).unwrap().role, ArgRole::Pointer);
    }

    #[test]
    fn restore_from_a_lane_never_written_is_unknown() {
        // Same save, but the restore reads lanes 2 and 3: the base has no provenance.
        let f = facts(&[0xf4004100, 0xf8000000, 0xbfc70000, 0xd7610028, 0x02010006, 0xd7610028, 0x02010207, 0xbe860180,
            0xd760000a, 0x02010528, 0xd760000b, 0x02010728, 0x7e000280, 0xee06800a, 0x00000000, 0x00000000, 0xbfb00000], PTRS);
        assert_eq!(f.status, KernelStatus::Unknown);
        assert_eq!(rules(&f), ["base-no-provenance"]);
        assert_eq!(mode(&f, 8), None, "the clobbered-then-lost pointer is not credited with the store");
    }

    #[test]
    fn cselect_between_two_argument_pointers_reaches_both() {
        // s_cmp_eq_u32 s8, 0; s_cselect_b64 s[2:3], s[4:5], s[6:7]; load via s[2:3]; store via s[12:13].
        let args = &[("a", 0, 8, "global_buffer"), ("b", 8, 8, "global_buffer"), ("flag", 16, 4, "by_value"), ("out", 24, 8, "global_buffer")];
        let f = facts(&[0xf4004100, 0xf8000000, 0xf4000200, 0xf8000010, 0xf4002300, 0xf8000018, 0xbfc70000, 0xbf068008,
            0x98820604, 0x7e000280, 0xee050002, 0x00000001, 0x00000000, 0xbfc00000, 0xee06800c, 0x00800000, 0x00000000,
            0xbfb00000], args);
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        for offset in [0, 8] {
            let a = f.arg_at(offset).unwrap();
            assert_eq!(a.mode, Some(AccessMode::Read));
            assert!(a.accesses.iter().all(|x| x.shared), "the selected load names both candidates");
        }
        assert_eq!(mode(&f, 24), Some(AccessMode::Write));
        let flag = f.arg_at(16).unwrap();
        assert_eq!((flag.role, flag.sinks.has(Sinks::GUARD)), (ArgRole::Integer, true));
        assert_eq!(f.read_cache, ReadCacheClass::VmemOnly);
    }

    #[test]
    fn unattributable_bases_make_the_kernel_unknown() {
        // A pointer loaded from memory (indirect chain): arg a is read through the scalar cache.
        let f = facts(&[0xf4002100, 0xf8000000, 0xbfc70000, 0xf4002182, 0xf8000000, 0xbfc70000, 0x7e000280, 0xee068006,
            0x00000000, 0x00000000, 0xbfb00000], PTRS);
        assert_eq!((f.status, rules(&f)), (KernelStatus::Unknown, vec!["base-memory-derived"]));
        assert_eq!(f.arg_at(0).unwrap().read_cache, ReadCacheClass::ScalarOnly);
        assert_eq!(f.read_cache, ReadCacheClass::Unknown);
        // A buffer V# built from memory.
        let f = facts(&[0xf4002100, 0xf8000000, 0xbfc70000, 0xf4004202, 0xf8000000, 0x7e000280, 0xbfc70000, 0xc406807c,
            0x40801000, 0x00000000, 0xbfb00000], PTRS);
        assert_eq!((f.status, rules(&f)), (KernelStatus::Unknown, vec!["base-memory-derived"]));
        // A constant base: s_mov_b64 s[6:7], 0x1000.
        let f = facts(&[0xbe8601ff, 0x00001000, 0x7e000280, 0xee068006, 0x00000000, 0x00000000, 0xbfb00000], PTRS);
        assert_eq!((f.status, rules(&f)), (KernelStatus::Unknown, vec!["base-no-provenance"]));
    }

    #[test]
    fn disabled_lanes_keep_their_pointer_until_exec_widens() {
        // v[2:3] = a; load; s_mov_b32 s12, exec_lo; v_cmpx_gt_u32 16, v0; v[2:3] = b;
        // store (enabled lanes hold b only); s_or_b32 exec_lo, exec_lo, s12; store (a or b).
        let f = facts(&[0xf4004100, 0xf8000000, 0xbfc70000, 0x7e040204, 0x7e060205, 0xee05007c, 0x00000001, 0x00000002,
            0xbe8c007e, 0x7d980090, 0x7e040206, 0x7e060207, 0xbfc00000, 0xee06807c, 0x00800000, 0x00000002, 0x8c7e0c7e,
            0xee06807c, 0x00800000, 0x00000002, 0xbfb00000], PTRS);
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        let stores_to_a = f.arg_at(0).unwrap().accesses.iter().filter(|x| x.class == AccessClass::VmemStore).count();
        let stores_to_b = f.arg_at(8).unwrap().accesses.iter().filter(|x| x.class == AccessClass::VmemStore).count();
        assert_eq!((stores_to_a, stores_to_b), (1, 2), "narrowed store reaches b only; the store after the restore reaches both");
        assert_eq!(mode(&f, 0), Some(AccessMode::ReadWrite));
    }

    #[test]
    fn divergent_writes_of_one_pointer_register_keep_both_candidates() {
        // if (v0 < 16) v[2:3] = a else v[2:3] = b (s_and_saveexec / s_xor exec / s_or exec); load v[2:3].
        let f = facts(&[0xf4004100, 0xf8000000, 0xbfc70000, 0x7c980090, 0xbe88206a, 0x7e040204, 0x7e060205, 0x8d7e087e,
            0x7e040206, 0x7e060207, 0x8c7e087e, 0xee05007c, 0x00000001, 0x00000002, 0xbfb00000], PTRS);
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!((mode(&f, 0), mode(&f, 8)), (Some(AccessMode::Read), Some(AccessMode::Read)));
        assert!(f.arg_at(0).unwrap().accesses[0].shared);
    }

    #[test]
    fn implicit_block_count_loaded_as_loop_bound_and_scalar_roles() {
        // n = s_load(0x8); bc = s_load(0x10) (hidden_block_count_x); loop while ++i < bc;
        // store i at out + n * ttmp9 (a work-group id).
        let args = &[("out", 0, 8, "global_buffer"), ("n", 8, 4, "by_value"),
            ("bcx", 16, 4, "hidden_block_count_x"), ("bcy", 20, 4, "hidden_block_count_y"), ("bcz", 24, 4, "hidden_block_count_z"),
            ("gsx", 28, 2, "hidden_group_size_x")];
        let f = facts(&[0xf4000080, 0xf8000010, 0xf4002100, 0xf8000000, 0xf4000200, 0xf8000008, 0xbfc70000, 0xbe830080,
            0x81038103, 0xbf0a0203, 0xbfa2fffd, 0x96067508, 0x7e000206, 0x7e020203, 0xee068004, 0x00800000, 0x00000000,
            0xbfb00000], args);
        assert_eq!(f.status, KernelStatus::Proven, "{:?}", f.unresolved);
        assert_eq!(f.implicit.hidden_loaded, ["hidden_block_count_x"]);
        assert_eq!(f.implicit.hidden_grid_live, ["hidden_block_count_x"]);
        assert!(f.arg_at(16).unwrap().sinks.has(Sinks::LOOP_BOUND));
        let n = f.arg_at(8).unwrap();
        assert_eq!(n.role, ArgRole::Integer);
        assert!(n.sinks.has(Sinks::GRID_TERM) && n.sinks.has(Sinks::ADDRESS_OFFSET) && !n.sinks.has(Sinks::ADDRESS_BASE));
        assert_eq!(f.arg_at(20).unwrap().role, ArgRole::NotLoaded);
        assert!(f.arg_at(0).unwrap().sinks.has(Sinks::ADDRESS_BASE));
    }

    #[test]
    fn analyzed_revision_carries_the_facts() {
        let program = crate::inst::Program {
            target: crate::inst::Target { arch: Arch::Gfx1201, xnack: crate::inst::Setting::Any, sramecc: crate::inst::Setting::Any, abi_version: 4 },
            kernels: vec![kernel(&[0xf4004100, 0xf8000000, 0xbfc70000, 0x7c980090, 0xbe88206a, 0x7e040204, 0x7e060205, 0x8d7e087e,
                0x7e040206, 0x7e060207, 0x8c7e087e, 0xee05007c, 0x00000001, 0x00000002, 0xbfb00000], PTRS)],
            source: None,
        };
        let analyzed = crate::edit::analyze(program, &SymbolId("k".into())).expect("analyzes");
        assert_eq!(analyzed.facts.kernargs.status, KernelStatus::Proven);
        assert_eq!(analyzed.facts.kernargs.arg_at(8).and_then(|a| a.mode), Some(AccessMode::Read));
    }
}
