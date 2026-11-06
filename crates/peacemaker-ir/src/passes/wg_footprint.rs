// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Workgroup write footprints (railgun task-graph, same-instance tiles).
//!
//! For one standalone gfx1201 kernel and one tile geometry this proves, per
//! pointer argument `P`, the fact railgun's host task-graph check consumes:
//!
//! > writes through `P` by two workgroups `w != w'` of the launch are
//! > disjoint whenever `w` and `w'` differ on some axis of `axes`.
//!
//! The facts rest on the exact address witnesses of the A1 pass
//! ([`super::kernargs`]): every write access attributed to `P` carries the
//! integer expression of its 64-bit address over the kernarg segment, the
//! hardware workgroup ids (`ttmp9` = x, `ttmp7` = `y | z << 16`), the work-item
//! id in `v0`, constants and uniform values. Each expression is evaluated in the
//! ring `Z/2^bits` as the affine form
//!
//! ```text
//! value = U + sum_a S_a * wg_a + r            (mod 2^bits)
//! ```
//!
//! with `U` a uniform part (constants plus opaque kernarg-derived atoms that
//! are equal for every lane of every workgroup), exact integer coefficients
//! `S_a` of the workgroup ids and a lane-dependent bounded part `r in
//! [rlo, rhi]` (the work-item id lies in `[0, block_x)`). A 32-bit form that is
//! sign- or zero-extended and scaled keeps its [`Ext`] structure: the
//! extension is injective on residues, so it needs no value range.
//!
//! The address of every write must be `P + (uniform) + tile term` with exactly
//! one tile term, and the proof is the circular separation argument:
//!
//! * the uniform part is the same for both tiles and cancels (it cancels
//!   modulo `2^bits`, so it needs no bound);
//! * with every tile id on axis `a` below `E_a`, two tiles that differ on an
//!   axis of the footprint put their windows at least `sepmin` apart, where
//!   `sepmin = min_j (|S_j| - sum_{i<j} |S_i| (E_i - 1))` over the axes sorted
//!   by increasing `|S|` (mixed-radix separation);
//! * the window of one tile is `r`'s spread plus the access width;
//!   `sepmin` at least that wide separates the windows, and for a 32-bit tile
//!   term `sum |S| (E - 1) + sepmin <= 2^32` keeps the separation circular (no
//!   wrap makes two tiles meet).
//!
//! Soundness rests on these named assumptions, which the result records and a
//! consumer must discharge:
//!
//! 1. the tile launches `block_x` work-items on x only (the descriptor enables
//!    no y/z work-item id), so `v0 in [0, block_x)`;
//! 2. tile ids lie below the recorded `bounds`;
//! 3. every [`WgRequirement`] holds of the kernarg block (a kernel that
//!    computes `blockIdx.x * blockDim.x` reads the hidden group size dword;
//!    its low half must equal `block_x`).
//!
//! Anything the evaluator cannot put in this form (a product of two
//! non-constants involving a tile id, a clamp of a tile id, a division by a
//! kernarg scalar, a pointer selected between arguments, a loaded value, ...)
//! kills the fact for the argument: `Unproven` with the address shape.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use crate::cfg::InstId;
use crate::inst::{Abi, Arch, Kernel};
use crate::kernarg::{AccessClass, ArgFacts, ArgRole, EntryReg, KernargFacts, KernelStatus};

use super::kernarg_ix::{Interner, Ix, Node};
use super::kernargs::{analyze_for_wg, WriteSite};

/// The tile geometry a footprint is proven for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WgLaunch {
    /// Work-items per workgroup on x (y and z are 1).
    pub block_x: u32,
}

/// The relation of a [`WgRequirement`] on a kernarg/record dword `d`.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum WgReqOp {
    /// `(d & mask) == value`.
    #[default]
    MaskEq,
    /// `d <= value` (unsigned); `mask` is `0xffff_ffff`.
    LeU,
    /// `d >= value` (unsigned); `mask` is `0xffff_ffff`.
    GeU,
    /// `value * grid_extent[axis] <= d` (unsigned, 64-bit arithmetic) over the
    /// task's tile grid; `mask` is the axis (0 x, 1 y, 2 z).
    GridLe,
}

/// The kernarg dword at `offset` satisfies `op` (default: masked with `mask`,
/// equals `value`); must hold of the record of every tile the fact is used for.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct WgRequirement {
    pub offset: u32,
    pub mask: u32,
    pub value: u32,
    pub op: WgReqOp,
}

mod guard;
#[cfg(test)]
mod guard_tests;

/// The proven footprint of one pointer argument.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WgDisjoint {
    /// Axes (0 = x, 1 = y, 2 = z) the footprint is disjoint along, ascending.
    pub axes: Vec<u8>,
    /// Exclusive upper bound of the tile id on each axis the proof assumes
    /// (the hardware limit [`HW_BOUNDS`] when the proof needs none).
    pub bounds: [u64; 3],
    pub block_x: u32,
    pub requires: Vec<WgRequirement>,
    /// Write accesses the proof covers.
    pub sites: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArgWgClass {
    /// The kernel never writes through the argument.
    NoWrites,
    /// Every write is a commutative atomic: tiles may overlap.
    AtomicOnly,
    Disjoint(WgDisjoint),
    /// No fact: why, with the address shape of the first offending write.
    Unproven(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgWg {
    /// Kernarg byte offset of the pointer argument.
    pub arg_offset: u32,
    pub class: ArgWgClass,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WgFacts {
    /// `Err(why)`: the kernel is outside the analysis (A1 not `Proven`/sealed
    /// only, a different arch/ABI, an unsupported geometry); `args` is empty.
    pub applicable: Result<(), String>,
    pub args: Vec<ArgWg>,
}
impl WgFacts {
    pub fn arg(&self, offset: u32) -> Option<&ArgWgClass> {
        self.args
            .iter()
            .find(|a| a.arg_offset == offset)
            .map(|a| &a.class)
    }
}

/// Tile id limits of the hardware: x is a 32-bit count, y and z the 16-bit
/// halves of `ttmp7`.
pub const HW_BOUNDS: [u64; 3] = [1 << 32, 1 << 16, 1 << 16];
const BOUND_CANDIDATES: [u64; 7] = [1 << 32, 1 << 28, 1 << 24, 1 << 20, 1 << 16, 1 << 12, 1 << 8];

/// Footprints of every pointer argument of `kernel`. `sealed` selects the A1
/// variant ([`super::kernargs::analyze_sealed`]) whose `Conditional` status is
/// admitted when its only condition is sealed tables.
pub fn analyze(kernel: &Kernel, arch: Arch, sealed: bool, launch: WgLaunch) -> WgFacts {
    let na = |why: String| WgFacts {
        applicable: Err(why),
        args: Vec::new(),
    };
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return na("not an HSA kernel".into());
    };
    if arch != Arch::Gfx1201 {
        return na(format!(
            "{arch:?}: the witness entry registers are gfx1201's"
        ));
    }
    if launch.block_x == 0 || launch.block_x > 1024 {
        return na(format!("block_x {} outside 1..=1024", launch.block_x));
    }
    let max_flat = metadata.parsed.max_flat_workgroup_size;
    if max_flat != 0 && launch.block_x > max_flat {
        return na(format!(
            "block_x {} exceeds the kernel's max flat workgroup size {max_flat}",
            launch.block_x
        ));
    }
    if (descriptor.compute_pgm_rsrc2.0 >> 11) & 3 != 0 {
        return na("the descriptor enables y/z work-item ids (v0 is packed)".into());
    }
    let (facts, input) = analyze_for_wg(kernel, arch, sealed);
    let Some(input) = input else {
        return na("A1 analysis did not run".into());
    };
    match facts.status {
        KernelStatus::Proven => {}
        KernelStatus::Conditional
            if sealed
                && !facts.sealed.is_empty()
                && facts.guards.is_empty()
                && facts.indexed.is_empty() => {}
        other => return na(format!("A1 status is {other:?}")),
    }
    let group_size_dword = metadata
        .parsed
        .args
        .iter()
        .find(|a| a.value_kind == "hidden_group_size_x" && a.offset % 4 == 0)
        .map(|a| a.offset);
    let pointers: BTreeSet<u32> = facts
        .args
        .iter()
        .filter(|a| a.role == ArgRole::Pointer)
        .map(|a| a.offset)
        .collect();
    let mut ev = Eval {
        it: input.it,
        block: i128::from(launch.block_x),
        lanes: (0..launch.block_x).collect(),
        wave32: kernel.wave == crate::inst::Wave::Wave32,
        group_size_dword,
        memo: HashMap::new(),
        full_memo: HashMap::new(),
        in_all: false,
    };
    let args = facts
        .args
        .iter()
        .filter(|a| {
            a.role == ArgRole::Pointer || a.accesses.iter().any(|x| x.class.mode().writes())
        })
        .map(|a| ArgWg {
            arg_offset: a.offset,
            class: check_arg(&mut ev, a, &input.sites, &pointers, launch, &facts),
        })
        .collect();
    WgFacts {
        applicable: Ok(()),
        args,
    }
}

// ---------------------------------------------------------------------------
// Affine forms.
// ---------------------------------------------------------------------------

/// The exact values of a lane-dependent part: one integer per work-item slot of
/// the evaluation ([`Eval::lanes`]).
type Tab = Rc<Vec<i128>>;

/// `U + sum S_a wg_a + r + sum_X X * G_X  (mod 2^bits)` plus at most one scaled
/// extension (64-bit only). `X` is a uniform atom and `G_X` a form that depends
/// on the tile id or the lane (the product `row stride * row index`).
#[derive(Clone, Debug, Eq, PartialEq)]
struct Aff {
    bits: u8,
    /// Uniform opaque atoms with their coefficients (nonzero).
    usym: BTreeMap<Ix, i128>,
    /// Uniform constant.
    uc: i128,
    s: [i128; 3],
    /// Lane-dependent part, an exact integer interval.
    rlo: i128,
    rhi: i128,
    /// Every value of `r` is a multiple of `2^rtz`.
    rtz: u8,
    /// The value of `r` of every work-item slot, when known exactly (`rlo`,
    /// `rhi`, `rtz` describe it).
    tab: Option<Tab>,
    ext: Option<Box<Ext>>,
    /// Atom times non-constant form; never an atom that also has a constant
    /// coefficient in `usym` (see [`normalize`]).
    prod: BTreeMap<Ix, Box<Aff>>,
    /// Uses the hidden-group-size substitution.
    req: bool,
}

/// `scale * ext(inner)` with `inner` a 32-bit form that depends on the tile id.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Ext {
    signed: bool,
    scale: i128,
    inner: Aff,
}

type Res = Result<Aff, String>;

fn modulus(bits: u8) -> i128 {
    1i128 << bits
}
fn ctz(x: i128) -> u32 {
    if x == 0 {
        127
    } else {
        x.unsigned_abs().trailing_zeros()
    }
}

/// `(min, max, common trailing zero bits)` of a lane table.
fn tab_stats(t: &[i128]) -> (i128, i128, u8) {
    let (mut lo, mut hi, mut tz) = (i128::MAX, i128::MIN, 127u32);
    for &v in t {
        lo = lo.min(v);
        hi = hi.max(v);
        tz = tz.min(ctz(v));
    }
    (lo, hi, tz as u8)
}

impl Aff {
    fn constant(bits: u8, v: i128) -> Aff {
        Aff {
            bits,
            usym: BTreeMap::new(),
            uc: v,
            s: [0; 3],
            rlo: 0,
            rhi: 0,
            rtz: 0,
            tab: None,
            ext: None,
            prod: BTreeMap::new(),
            req: false,
        }
    }
    fn atom(bits: u8, ix: Ix) -> Aff {
        let mut a = Aff::constant(bits, 0);
        a.usym.insert(ix, 1);
        a
    }
    fn range(bits: u8, lo: i128, hi: i128, rtz: u8) -> Aff {
        Aff {
            rlo: lo,
            rhi: hi,
            rtz,
            ..Aff::constant(bits, 0)
        }
    }
    /// A pure lane value with the exact per-slot integers `t` (a constant when
    /// they are all equal).
    fn from_tab(bits: u8, t: Vec<i128>) -> Aff {
        let (lo, hi, tz) = tab_stats(&t);
        if lo == hi {
            return Aff::constant(bits, lo);
        }
        Aff {
            rlo: lo,
            rhi: hi,
            rtz: tz,
            tab: Some(Rc::new(t)),
            ..Aff::constant(bits, 0)
        }
    }
    fn wg(axis: usize, coef: i128) -> Aff {
        let mut a = Aff::constant(32, 0);
        a.s[axis] = coef;
        a
    }
    fn is_lane_free(&self) -> bool {
        self.rlo == 0 && self.rhi == 0
    }
    /// A product term anywhere in the form (also inside the extension).
    fn has_prod_deep(&self) -> bool {
        !self.prod.is_empty() || self.ext.as_ref().is_some_and(|e| e.inner.has_prod_deep())
    }
    fn has_wg(&self) -> bool {
        self.s != [0; 3] || self.ext.is_some() || !self.prod.is_empty()
    }
    /// No workgroup id, no lane term, no extension: the same for every lane of
    /// every workgroup.
    fn is_uniform(&self) -> bool {
        !self.has_wg() && self.is_lane_free()
    }
    fn as_const(&self) -> Option<i128> {
        (self.is_uniform() && self.usym.is_empty()).then_some(self.uc)
    }
    /// The machine value's exact integer range when it is a pure constant or
    /// lane expression (no unknown, no workgroup id).
    fn exact_range(&self) -> Option<(i128, i128)> {
        if self.has_wg() || !self.usym.is_empty() {
            return None;
        }
        let m = modulus(self.bits);
        if let Some(t) = &self.tab {
            let mut lo = i128::MAX;
            let mut hi = i128::MIN;
            for &v in t.iter() {
                let r = (self.uc + v).rem_euclid(m);
                lo = lo.min(r);
                hi = hi.max(r);
            }
            return Some((lo, hi));
        }
        let (lo, hi) = (self.uc + self.rlo, self.uc + self.rhi);
        if lo >= 0 && hi < m {
            Some((lo, hi))
        } else if hi < 0 && lo >= -m {
            Some((lo + m, hi + m))
        } else {
            None
        }
    }
    /// The machine value of each of `n` lane slots of a pure constant or lane
    /// expression with an exact table.
    fn residues(&self, n: usize) -> Option<Vec<i128>> {
        if self.has_wg() || !self.usym.is_empty() {
            return None;
        }
        let m = modulus(self.bits);
        match &self.tab {
            Some(t) if t.len() == n => {
                Some(t.iter().map(|&v| (self.uc + v).rem_euclid(m)).collect())
            }
            Some(_) => None,
            None if self.is_lane_free() => Some(vec![self.uc.rem_euclid(m); n]),
            None => None,
        }
    }
    /// Low bits of the residue known to be zero.
    fn total_tz(&self) -> u32 {
        let mut t = 127u32;
        if self.uc != 0 {
            t = t.min(ctz(self.uc));
        }
        for &c in self.usym.values() {
            t = t.min(ctz(c));
        }
        for &s in &self.s {
            if s != 0 {
                t = t.min(ctz(s));
            }
        }
        if self.rlo == self.rhi {
            if self.rlo != 0 {
                t = t.min(ctz(self.rlo));
            }
        } else {
            t = t.min(u32::from(self.rtz));
        }
        if let Some(e) = &self.ext {
            t = t.min(ctz(e.scale));
        }
        for g in self.prod.values() {
            t = t.min(g.total_tz());
        }
        t.min(u32::from(self.bits))
    }
}

fn checked<T>(x: Option<T>) -> Result<T, String> {
    x.ok_or_else(|| "coefficient overflow".to_owned())
}

/// Fold a constant lane part into the uniform constant.
fn fold_lane_const(a: &mut Aff) -> Result<(), String> {
    if a.rlo == a.rhi && a.rlo != 0 {
        a.uc = checked(a.uc.checked_add(a.rlo))?;
        a.rlo = 0;
        a.rhi = 0;
        a.rtz = 0;
        a.tab = None;
    }
    Ok(())
}

/// Keep every atom either with a constant coefficient (`usym`) or with a
/// non-constant form (`prod`), never both.
fn normalize(a: &mut Aff) -> Result<(), String> {
    let keys: Vec<Ix> = a.prod.keys().copied().collect();
    for x in keys {
        if let Some(c) = a.usym.remove(&x) {
            let g = a.prod.get_mut(&x).expect("product");
            g.uc = checked(g.uc.checked_add(c))?;
        }
        if let Some(c) = a.prod[&x].as_const() {
            a.prod.remove(&x);
            if c != 0 {
                a.usym.insert(x, c);
            }
        }
    }
    Ok(())
}

fn add(a: &Aff, b: &Aff) -> Res {
    if a.bits != b.bits {
        return Err("width mismatch".into());
    }
    let mut usym = a.usym.clone();
    for (&k, &c) in &b.usym {
        let e = usym.entry(k).or_insert(0);
        *e = checked(e.checked_add(c))?;
        if *e == 0 {
            usym.remove(&k);
        }
    }
    let mut prod = a.prod.clone();
    for (&k, g) in &b.prod {
        match prod.get(&k) {
            Some(h) => {
                let sum = add(h, g)?;
                prod.insert(k, Box::new(sum));
            }
            None => {
                prod.insert(k, g.clone());
            }
        }
    }
    let mut s = [0i128; 3];
    for i in 0..3 {
        s[i] = checked(a.s[i].checked_add(b.s[i]))?;
    }
    let ext = match (&a.ext, &b.ext) {
        (None, None) => None,
        (Some(e), None) | (None, Some(e)) => Some(e.clone()),
        (Some(x), Some(y)) => {
            if x.signed == y.signed && x.inner == y.inner {
                let scale = checked(x.scale.checked_add(y.scale))?;
                (scale != 0).then(|| {
                    Box::new(Ext {
                        scale,
                        ..(**x).clone()
                    })
                })
            } else {
                return Err("two different workgroup-dependent extension terms".into());
            }
        }
    };
    if ext.is_some() && s != [0; 3] {
        return Err("a direct 64-bit workgroup term next to an extension term".into());
    }
    let (tab, rlo, rhi, rtz) = match (a.is_lane_free(), b.is_lane_free()) {
        (true, true) => (None, 0, 0, 0),
        (true, false) => (b.tab.clone(), b.rlo, b.rhi, b.rtz),
        (false, true) => (a.tab.clone(), a.rlo, a.rhi, a.rtz),
        (false, false) => match (&a.tab, &b.tab) {
            (Some(x), Some(y)) if x.len() == y.len() => {
                let mut t = Vec::with_capacity(x.len());
                for (p, q) in x.iter().zip(y.iter()) {
                    t.push(checked(p.checked_add(*q))?);
                }
                let (lo, hi, tz) = tab_stats(&t);
                if lo == 0 && hi == 0 {
                    (None, 0, 0, 0)
                } else {
                    (Some(Rc::new(t)), lo, hi, tz)
                }
            }
            _ => (
                None,
                checked(a.rlo.checked_add(b.rlo))?,
                checked(a.rhi.checked_add(b.rhi))?,
                a.rtz.min(b.rtz),
            ),
        },
    };
    let mut out = Aff {
        bits: a.bits,
        usym,
        uc: checked(a.uc.checked_add(b.uc))?,
        s,
        rlo,
        rhi,
        rtz,
        tab,
        ext,
        prod,
        req: a.req | b.req,
    };
    fold_lane_const(&mut out)?;
    normalize(&mut out)?;
    Ok(out)
}

fn mulc(a: &Aff, c: i128) -> Res {
    if c == 0 {
        return Ok(Aff {
            req: a.req,
            ..Aff::constant(a.bits, 0)
        });
    }
    let mut usym = BTreeMap::new();
    for (&k, &v) in &a.usym {
        usym.insert(k, checked(v.checked_mul(c))?);
    }
    let mut prod = BTreeMap::new();
    for (&k, g) in &a.prod {
        prod.insert(k, Box::new(mulc(g, c)?));
    }
    let mut s = [0i128; 3];
    for i in 0..3 {
        s[i] = checked(a.s[i].checked_mul(c))?;
    }
    let ext = match &a.ext {
        None => None,
        Some(e) => Some(Box::new(Ext {
            scale: checked(e.scale.checked_mul(c))?,
            ..(**e).clone()
        })),
    };
    let (tab, rlo, rhi, rtz) = match &a.tab {
        Some(t) => {
            let mut nt = Vec::with_capacity(t.len());
            for v in t.iter() {
                nt.push(checked(v.checked_mul(c))?);
            }
            let (lo, hi, tz) = tab_stats(&nt);
            (Some(Rc::new(nt)), lo, hi, tz)
        }
        None => {
            let (x, y) = (
                checked(a.rlo.checked_mul(c))?,
                checked(a.rhi.checked_mul(c))?,
            );
            let rtz = (u32::from(a.rtz) + ctz(c).min(64)).min(127) as u8;
            (None, x.min(y), x.max(y), rtz)
        }
    };
    let mut out = Aff {
        bits: a.bits,
        usym,
        uc: checked(a.uc.checked_mul(c))?,
        s,
        rlo,
        rhi,
        rtz,
        tab,
        ext,
        prod,
        req: a.req,
    };
    fold_lane_const(&mut out)?;
    Ok(out)
}

fn sub(a: &Aff, b: &Aff) -> Res {
    add(a, &mulc(b, -1)?)
}

fn pure_range(bits: u8, lo: i128, hi: i128, rtz: u8, req: bool) -> Aff {
    Aff {
        req,
        ..Aff::range(bits, lo, hi, rtz)
    }
}

// ---------------------------------------------------------------------------
// Evaluator.
// ---------------------------------------------------------------------------

struct Eval {
    it: Interner,
    block: i128,
    /// The lane identity of every work-item slot of the tile: slot `l` stands
    /// for work-item `lanes[l]`. A site whose EXEC predicate leaves only some
    /// work-items active maps every inactive slot to an active one, so that the
    /// lane tables of the evaluation describe the active work-items only.
    lanes: Vec<u32>,
    wave32: bool,
    /// Kernarg byte offset of `hidden_group_size_x` (4-aligned), if declared.
    group_size_dword: Option<u32>,
    memo: HashMap<Ix, Res>,
    /// The evaluation memo over every work-item (the first-lane values).
    full_memo: HashMap<Ix, Res>,
    /// Currently evaluating over every work-item ([`Eval::with_all_lanes`]).
    in_all: bool,
}

impl Eval {
    fn eval(&mut self, ix: Ix) -> Res {
        if let Some(r) = self.memo.get(&ix) {
            return r.clone();
        }
        let r = self.eval_node(ix).map_err(|e| {
            if e.contains('@') {
                e
            } else {
                format!("{e} @ {}", show(&self.it, ix, 4))
            }
        });
        self.memo.insert(ix, r.clone());
        r
    }

    /// The uniform atom of a node whose children are all uniform, or `None`.
    fn uniform_atom(&mut self, ix: Ix, bits: u8) -> Option<Aff> {
        self.it.uniform(ix).then(|| Aff::atom(bits, ix))
    }

    /// Run `f` with every work-item active (the identity lane map and its own
    /// memo).
    fn with_all_lanes<R>(&mut self, f: impl FnOnce(&mut Eval) -> R) -> R {
        if self.in_all {
            return f(self);
        }
        let identity: Vec<u32> = (0..self.block as u32).collect();
        let lanes = std::mem::replace(&mut self.lanes, identity);
        std::mem::swap(&mut self.memo, &mut self.full_memo);
        self.in_all = true;
        let r = f(self);
        self.in_all = false;
        std::mem::swap(&mut self.memo, &mut self.full_memo);
        self.lanes = lanes;
        r
    }

    /// Evaluate with the work-items `lanes` standing for the slots (the
    /// per-site EXEC restriction); the memo is kept when the map is unchanged.
    fn set_lanes(&mut self, lanes: Vec<u32>) {
        if self.lanes != lanes {
            self.lanes = lanes;
            self.memo.clear();
        }
    }

    fn n(&self) -> usize {
        self.lanes.len()
    }

    /// Pointwise 32-bit operation on two pure lane/constant values, at least
    /// one of them lane-dependent with an exact table.
    fn lane_map2(&self, x: &Aff, y: &Aff, f: impl Fn(i128, i128) -> i128) -> Option<Aff> {
        if x.is_lane_free() && y.is_lane_free() {
            return None;
        }
        let (vx, vy) = (x.residues(self.n())?, y.residues(self.n())?);
        let m = modulus(32);
        let t: Vec<i128> = vx
            .iter()
            .zip(&vy)
            .map(|(&p, &q)| f(p, q).rem_euclid(m))
            .collect();
        let mut a = Aff::from_tab(32, t);
        a.req = x.req | y.req;
        Some(a)
    }

    fn lane_map1(&self, x: &Aff, f: impl Fn(i128) -> i128) -> Option<Aff> {
        if x.is_lane_free() {
            return None;
        }
        let vx = x.residues(self.n())?;
        let m = modulus(32);
        let t: Vec<i128> = vx.iter().map(|&p| f(p).rem_euclid(m)).collect();
        let mut a = Aff::from_tab(32, t);
        a.req = x.req;
        Some(a)
    }

    /// The truth value of flag `f` for every lane slot, when it is a function
    /// of the work-item id alone (pure lane/constant operands).
    fn flag_lanes(&mut self, f: Ix) -> Option<Vec<bool>> {
        let n = self.n();
        let sgn = |v: i128, bits: u8| {
            let m = modulus(bits);
            if v >= m / 2 {
                v - m
            } else {
                v
            }
        };
        match self.it.node(f)? {
            Node::True => Some(vec![true; n]),
            Node::Not(g) => Some(self.flag_lanes(g)?.into_iter().map(|b| !b).collect()),
            Node::And(a, b) | Node::Or(a, b) => {
                let is_and = matches!(self.it.node(f), Some(Node::And(..)));
                let (x, y) = (self.flag_lanes(a)?, self.flag_lanes(b)?);
                Some(
                    x.into_iter()
                        .zip(y)
                        .map(|(p, q)| if is_and { p && q } else { p || q })
                        .collect(),
                )
            }
            Node::Ltu { a, b } | Node::Lts { a, b } | Node::Eq { a, b } => {
                let (x, y) = (self.eval(a).ok()?, self.eval(b).ok()?);
                let (vx, vy) = (x.residues(n)?, y.residues(n)?);
                let node = self.it.node(f)?;
                Some(
                    vx.iter()
                        .zip(&vy)
                        .map(|(&p, &q)| match node {
                            Node::Ltu { .. } => p < q,
                            Node::Lts { .. } => sgn(p, x.bits) < sgn(q, y.bits),
                            _ => p == q,
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }

    fn eval_node(&mut self, ix: Ix) -> Res {
        let Some(node) = self.it.node(ix) else {
            return Err("no witness".into());
        };
        match node {
            Node::C32(v) => Ok(Aff::constant(32, i128::from(v))),
            Node::C64(v) => Ok(Aff::constant(64, i128::from(v))),
            Node::K32(_) | Node::Opaque(..) => Ok(Aff::atom(32, ix)),
            Node::K64(_) => Ok(Aff::atom(64, ix)),
            Node::Entry(EntryReg::WorkgroupIdX) => Ok(Aff::wg(0, 1)),
            Node::Entry(EntryReg::WorkgroupIdYZ) => {
                let mut a = Aff::wg(1, 1);
                a.s[2] = 1 << 16;
                Ok(a)
            }
            Node::Entry(EntryReg::WorkitemIdPacked) => Ok(Aff::from_tab(
                32,
                self.lanes.iter().map(|&l| i128::from(l)).collect(),
            )),
            Node::FirstLane(a) => {
                if !self.wave32 {
                    return Err("the first lane of a wave64 wave".into());
                }
                // The value of lane 0 of the wave, over every work-item.
                let x = self.with_all_lanes(|e| e.eval(a))?;
                let Some(vals) = x.residues(self.block as usize) else {
                    return Err(
                        "the first lane of a value that is not a function of the work-item id"
                            .into(),
                    );
                };
                let t: Vec<i128> = self
                    .lanes
                    .iter()
                    .map(|&l| vals[(l as usize / 32) * 32])
                    .collect();
                let mut r = Aff::from_tab(32, t);
                r.req = x.req;
                Ok(r)
            }
            Node::Load(_) => Err("a loaded value".into()),
            Node::Ent(_) => Err("a sealed-table entry".into()),
            Node::MulSU24(..) => Err("a signed-by-unsigned 64-bit product (mad pair)".into()),
            Node::LaneMask(_) => Err("a lane mask".into()),
            Node::Mask(f) | Node::Bit(f) => {
                let Some(fl) = self.flag_lanes(f) else {
                    return Err("a flag/carry value in an address".into());
                };
                let on = if matches!(node, Node::Mask(_)) {
                    modulus(32) - 1
                } else {
                    1
                };
                Ok(Aff::from_tab(
                    32,
                    fl.into_iter().map(|b| if b { on } else { 0 }).collect(),
                ))
            }
            Node::Carry { .. }
            | Node::Ltu { .. }
            | Node::Lts { .. }
            | Node::Eq { .. }
            | Node::Zero64(_)
            | Node::True
            | Node::Not(_)
            | Node::Or(..)
            | Node::And(..)
            | Node::NzFlag(_) => Err("a flag/carry value in an address".into()),
            Node::Add32(a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                add(&x, &y)
            }
            Node::Add64(a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                add(&x, &y)
            }
            Node::Sub32(a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                sub(&x, &y)
            }
            Node::Shl32(a, k) => {
                let x = self.eval(a)?;
                mulc(&x, 1i128 << k)
            }
            Node::Shl64(a, k) => {
                let x = self.eval(a)?;
                mulc(&x, 1i128 << k)
            }
            Node::Mul32(a, b) | Node::Mul64(a, b) => {
                let bits = if matches!(node, Node::Mul32(..)) {
                    32
                } else {
                    64
                };
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                let sgn = |c: i128| {
                    let m = modulus(bits);
                    if c >= m / 2 {
                        c - m
                    } else {
                        c
                    }
                };
                if let Some(c) = y.as_const() {
                    let mut r = mulc(&x, sgn(c))?;
                    r.req |= y.req;
                    return Ok(r);
                }
                if let Some(c) = x.as_const() {
                    let mut r = mulc(&y, sgn(c))?;
                    r.req |= x.req;
                    return Ok(r);
                }
                if x.is_uniform() && y.is_uniform() {
                    if let Some(u) = self.uniform_atom(ix, bits) {
                        return Ok(u);
                    }
                }
                if bits == 32 {
                    if let Some(r) = self.lane_map2(&x, &y, |p, q| p * q) {
                        return Ok(r);
                    }
                    if let (Some((al, ah)), Some((bl, bh))) = (x.exact_range(), y.exact_range()) {
                        if let (Some(lo), Some(hi)) = (al.checked_mul(bl), ah.checked_mul(bh)) {
                            if hi < modulus(32) {
                                return Ok(pure_range(32, lo, hi, 0, x.req | y.req));
                            }
                        }
                    }
                }
                // A uniform factor `c + sum c_i X_i` times a form that depends
                // on the tile or the lane distributes into products of atoms
                // with that form.
                if x.is_uniform() != y.is_uniform() {
                    let (u, v) = if x.is_uniform() { (&x, &y) } else { (&y, &x) };
                    let mut out = mulc(v, sgn(u.uc))?;
                    for (&atom, &c) in &u.usym {
                        let mut term = Aff::constant(bits, 0);
                        term.prod.insert(atom, Box::new(mulc(v, sgn(c))?));
                        out = add(&out, &term)?;
                    }
                    out.req |= u.req;
                    return Ok(out);
                }
                Err("product of two non-constant values".into())
            }
            Node::Sext(a) | Node::Zext(a) => {
                let signed = matches!(node, Node::Sext(_));
                let x = self.eval(a)?;
                self.extend(ix, signed, x)
            }
            Node::Lo(_) | Node::Hi(_) => {
                if let Some(u) = self.uniform_atom(ix, 32) {
                    return Ok(u);
                }
                Err("half of a 64-bit value".into())
            }
            Node::Sra31(a) => {
                let x = self.eval(a)?;
                if let Some(r) = self.lane_map1(&x, |p| ((p as u32 as i32) >> 31) as u32 as i128) {
                    return Ok(r);
                }
                if let Some((lo, hi)) = x.exact_range() {
                    let m = modulus(32);
                    return Ok(if hi < m / 2 {
                        Aff {
                            req: x.req,
                            ..Aff::constant(32, 0)
                        }
                    } else if lo >= m / 2 {
                        Aff {
                            req: x.req,
                            ..Aff::constant(32, m - 1)
                        }
                    } else {
                        pure_range(32, 0, m - 1, 0, x.req)
                    });
                }
                self.uniform_atom(ix, 32)
                    .ok_or_else(|| "sign of a workgroup-dependent value".into())
            }
            Node::Shr32(a, k) => {
                let x = self.eval(a)?;
                if let Some(r) = self.lane_map1(&x, |p| p >> k) {
                    return Ok(r);
                }
                if let Some((lo, hi)) = x.exact_range() {
                    return Ok(pure_range(32, lo >> k, hi >> k, 0, x.req));
                }
                if is_yz(&x) && k == 16 {
                    return Ok(Aff::wg(2, 1));
                }
                self.uniform_atom(ix, 32)
                    .ok_or_else(|| "shift right of a workgroup-dependent value".into())
            }
            Node::Sra32(a, k) => {
                let x = self.eval(a)?;
                if let Some(r) = self.lane_map1(&x, |p| ((p as u32 as i32) >> k) as u32 as i128) {
                    return Ok(r);
                }
                if let Some((lo, hi)) = x.exact_range() {
                    let m = modulus(32);
                    if hi < m / 2 {
                        return Ok(pure_range(32, lo >> k, hi >> k, 0, x.req));
                    }
                    if lo >= m / 2 {
                        return Ok(pure_range(32, (lo - m) >> k, (hi - m) >> k, 0, x.req));
                    }
                }
                self.uniform_atom(ix, 32)
                    .ok_or_else(|| "arithmetic shift of a workgroup-dependent value".into())
            }
            Node::And32(a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                let norm = |c: i128| c.rem_euclid(modulus(32));
                // The hidden group size: low half of the dword, equal to the tile's
                // block (the tile launches `block_x` work-items on x).
                for (k, mask) in [(a, &y), (b, &x)] {
                    if let (Some(Node::K32(o)), Some(0xffff)) =
                        (self.it.node(k), mask.as_const().map(norm))
                    {
                        if Some(o) == self.group_size_dword {
                            return Ok(Aff {
                                req: true,
                                ..Aff::constant(32, self.block)
                            });
                        }
                    }
                }
                if (is_yz(&x) && y.as_const().map(norm) == Some(0xffff))
                    || (is_yz(&y) && x.as_const().map(norm) == Some(0xffff))
                {
                    return Ok(Aff::wg(1, 1));
                }
                if let Some(r) = self.lane_map2(&x, &y, |p, q| p & q) {
                    return Ok(r);
                }
                let masked = match (x.as_const(), y.as_const()) {
                    (_, Some(m)) => Some((norm(m), &x)),
                    (Some(m), _) => Some((norm(m), &y)),
                    _ => None,
                };
                if let Some((m, v)) = masked {
                    if let Some((_, hi)) = v.exact_range() {
                        if (m & (m + 1)) == 0 && hi <= m {
                            return Ok(v.clone());
                        }
                        let tz = v.total_tz().max(ctz(m)).min(31) as u8;
                        return Ok(pure_range(32, 0, hi.min(m), tz, x.req | y.req));
                    }
                } else if let (Some((_, ah)), Some((_, bh))) = (x.exact_range(), y.exact_range()) {
                    return Ok(pure_range(32, 0, ah.min(bh), 0, x.req | y.req));
                }
                self.uniform_atom(ix, 32)
                    .ok_or_else(|| "bit mask of a workgroup-dependent value".into())
            }
            Node::Or32(a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                if let Some(r) = self.lane_map2(&x, &y, |p, q| p | q) {
                    return Ok(r);
                }
                // Disjoint bit ranges: OR is ADD.
                for (hi_part, lo_part) in [(&x, &y), (&y, &x)] {
                    if let Some((_, h)) = lo_part.exact_range() {
                        let k = 128 - (h as u128).leading_zeros();
                        if hi_part.total_tz() >= k {
                            return add(hi_part, lo_part);
                        }
                    }
                }
                if let (Some((al, ah)), Some((bl, bh))) = (x.exact_range(), y.exact_range()) {
                    let top = ah.max(bh);
                    let bl_len = 128 - (top as u128).leading_zeros();
                    return Ok(pure_range(
                        32,
                        al.max(bl),
                        (1i128 << bl_len) - 1,
                        0,
                        x.req | y.req,
                    ));
                }
                self.uniform_atom(ix, 32)
                    .ok_or_else(|| "bit or of a workgroup-dependent value".into())
            }
            Node::MinS(a, b) | Node::MaxS(a, b) | Node::MinU(a, b) | Node::MaxU(a, b) => {
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                let signed = matches!(node, Node::MinS(..) | Node::MaxS(..));
                let max = matches!(node, Node::MaxS(..) | Node::MaxU(..));
                let key = |v: i128| {
                    if signed {
                        i128::from(v as u32 as i32)
                    } else {
                        v
                    }
                };
                if let Some(r) = self.lane_map2(&x, &y, |p, q| {
                    let (kp, kq) = (key(p), key(q));
                    if (kp < kq) == max {
                        q
                    } else {
                        p
                    }
                }) {
                    return Ok(r);
                }
                if let (Some((al, ah)), Some((bl, bh))) = (x.exact_range(), y.exact_range()) {
                    let m = modulus(32);
                    let sg = |lo: i128, hi: i128| -> Option<(i128, i128)> {
                        if !signed || hi < m / 2 {
                            Some((lo, hi))
                        } else if lo >= m / 2 {
                            Some((lo - m, hi - m))
                        } else {
                            None
                        }
                    };
                    if let (Some((a0, a1)), Some((b0, b1))) = (sg(al, ah), sg(bl, bh)) {
                        let (lo, hi) = if max {
                            (a0.max(b0), a1.max(b1))
                        } else {
                            (a0.min(b0), a1.min(b1))
                        };
                        if lo >= 0 {
                            return Ok(pure_range(32, lo, hi, 0, x.req | y.req));
                        }
                    }
                }
                self.uniform_atom(ix, 32)
                    .ok_or_else(|| "min/max involving a workgroup-dependent value".into())
            }
            Node::Sel(f, a, b) => {
                if let Some(u) = self.uniform_atom(ix, 32) {
                    return Ok(u);
                }
                let Some(fl) = self.flag_lanes(f) else {
                    return Err("select on a workgroup-dependent value".into());
                };
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                let n = self.n();
                let (Some(vx), Some(vy)) = (x.residues(n), y.residues(n)) else {
                    return Err("select of workgroup-dependent values".into());
                };
                let t: Vec<i128> = fl
                    .iter()
                    .zip(vx.iter().zip(&vy))
                    .map(|(&c, (&p, &q))| if c { p } else { q })
                    .collect();
                let mut r = Aff::from_tab(32, t);
                r.req = x.req | y.req;
                Ok(r)
            }
        }
    }

    fn extend(&mut self, ix: Ix, signed: bool, x: Aff) -> Res {
        if x.bits != 32 {
            return Err("extension of a non-32-bit value".into());
        }
        if let Some(t) = &x.tab {
            if !x.has_wg() && x.usym.is_empty() {
                let m = modulus(32);
                let vals: Vec<i128> = t
                    .iter()
                    .map(|&v| {
                        let r = (x.uc + v).rem_euclid(m);
                        if signed && r >= m / 2 {
                            r - m
                        } else {
                            r
                        }
                    })
                    .collect();
                let mut r = Aff::from_tab(64, vals);
                r.req = x.req;
                return Ok(r);
            }
        }
        if let Some((lo, hi)) = x.exact_range() {
            let m = modulus(32);
            let (lo, hi) = if !signed || hi < m / 2 {
                (lo, hi)
            } else if lo >= m / 2 {
                (lo - m, hi - m)
            } else {
                return Err("sign extension of a value spanning the sign bit".into());
            };
            let rtz = if lo == hi { 0 } else { x.rtz };
            let mut a = Aff {
                req: x.req,
                rlo: lo,
                rhi: hi,
                rtz,
                ..Aff::constant(64, 0)
            };
            fold_lane_const(&mut a)?;
            return Ok(a);
        }
        if x.is_uniform() {
            return self
                .uniform_atom(ix, 64)
                .ok_or_else(|| "extension of a non-uniform value".into());
        }
        if x.s == [0; 3] && x.prod.is_empty() {
            return Err("extension of a lane-dependent value with an unknown part".into());
        }
        let req = x.req;
        Ok(Aff {
            req,
            ext: Some(Box::new(Ext {
                signed,
                scale: 1,
                inner: x,
            })),
            ..Aff::constant(64, 0)
        })
    }
}

fn is_yz(a: &Aff) -> bool {
    a.bits == 32
        && a.usym.is_empty()
        && a.uc == 0
        && a.is_lane_free()
        && a.ext.is_none()
        && a.s == [0, 1, 1 << 16]
}

/// A compact rendering of the witness for diagnostics.
/// Diagnostics rendering of a witness (see [`show`]).
pub(super) fn show_ix(it: &Interner, ix: Ix, depth: u32) -> String {
    if ix.is_none() {
        return "NONE".into();
    }
    show(it, ix, depth)
}

fn show(it: &Interner, ix: Ix, depth: u32) -> String {
    let Some(n) = it.node(ix) else {
        return "?".into();
    };
    if depth == 0 {
        return "..".into();
    }
    let d = depth - 1;
    let s = |x: Ix| show(it, x, d);
    match n {
        Node::C32(v) => format!("{v}"),
        Node::C64(v) => format!("{v}"),
        Node::K32(o) | Node::K64(o) => format!("k[{o}]"),
        Node::Entry(EntryReg::WorkgroupIdX) => "wg_x".into(),
        Node::Entry(EntryReg::WorkgroupIdYZ) => "wg_yz".into(),
        Node::Entry(EntryReg::WorkitemIdPacked) => "lane".into(),
        Node::Load(_) => "load".into(),
        Node::Ent(_) => "entry".into(),
        Node::Opaque(..) => "opq".into(),
        Node::Lo(a) => format!("lo({})", s(a)),
        Node::Hi(a) => format!("hi({})", s(a)),
        Node::Sra31(a) => format!("sign({})", s(a)),
        Node::Sra32(a, k) => format!("({}>>s{k})", s(a)),
        Node::Shr32(a, k) => format!("({}>>{k})", s(a)),
        Node::Shl32(a, k) | Node::Shl64(a, k) => format!("({}<<{k})", s(a)),
        Node::Sext(a) => format!("sext({})", s(a)),
        Node::Zext(a) => format!("zext({})", s(a)),
        Node::Add32(a, b) | Node::Add64(a, b) => format!("({}+{})", s(a), s(b)),
        Node::Sub32(a, b) => format!("({}-{})", s(a), s(b)),
        Node::Mul32(a, b) | Node::Mul64(a, b) => format!("({}*{})", s(a), s(b)),
        Node::And32(a, b) => format!("({}&{})", s(a), s(b)),
        Node::Or32(a, b) => format!("({}|{})", s(a), s(b)),
        Node::MinS(a, b) | Node::MinU(a, b) => format!("min({},{})", s(a), s(b)),
        Node::MaxS(a, b) | Node::MaxU(a, b) => format!("max({},{})", s(a), s(b)),
        Node::Sel(_, a, b) => format!("sel({},{})", s(a), s(b)),
        Node::FirstLane(a) => format!("first({})", s(a)),
        _ => "flag".into(),
    }
}

// ---------------------------------------------------------------------------
// The per-argument proof.
// ---------------------------------------------------------------------------

/// One write access reduced to its tile term and window.
struct SiteForm {
    /// 64-bit uniform atoms other than the base pointer.
    usym: BTreeMap<Ix, i128>,
    uc: i128,
    s: [i128; 3],
    rlo: i128,
    rhi: i128,
    ext: Option<Ext>,
    width: i128,
    req: bool,
    /// The full address form without the base pointer.
    a: Aff,
    /// The dominating scalar branch facts, evaluated under the site's lanes.
    rels: Vec<guard::Rel>,
}

/// The work-items that can execute a site: every work-item that satisfies the
/// EXEC lane predicates (`None`: none can). Inactive slots map to the first
/// active work-item so the lane tables of the evaluation describe active ones.
fn lane_model(ev: &mut Eval, lane_facts: &[(Ix, bool)]) -> Option<Vec<u32>> {
    let block = ev.block as usize;
    let mut active = vec![true; block];
    ev.with_all_lanes(|e| {
        for &(atom, holds) in lane_facts {
            if let Some(v) = e.flag_lanes(atom) {
                for (a, b) in active.iter_mut().zip(v) {
                    *a &= b == holds;
                }
            }
        }
    });
    let first = active.iter().position(|&a| a)?;
    Some(
        (0..block)
            .map(|l| if active[l] { l as u32 } else { first as u32 })
            .collect(),
    )
}

fn check_arg(
    ev: &mut Eval,
    arg: &ArgFacts,
    sites: &BTreeMap<InstId, WriteSite>,
    pointers: &BTreeSet<u32>,
    launch: WgLaunch,
    facts: &KernargFacts,
) -> ArgWgClass {
    let writes: Vec<_> = arg
        .accesses
        .iter()
        .filter(|a| a.class.mode().writes())
        .collect();
    if writes.is_empty() {
        return ArgWgClass::NoWrites;
    }
    if writes.iter().all(|a| {
        matches!(
            a.class,
            AccessClass::Atomic { commutative: true } | AccessClass::AtomicRelease
        )
    }) {
        return ArgWgClass::AtomicOnly;
    }
    let un = |s: String| ArgWgClass::Unproven(s);
    let mut forms: Vec<SiteForm> = Vec::new();
    for w in &writes {
        let at = |m: &str| format!("{m} (pc {})", sites.get(&w.inst).map_or(0, |s| s.pos));
        if w.shared {
            return un(at("the write may go through several arguments"));
        }
        let Some(site) = sites.get(&w.inst) else {
            return un(at("the write has no recorded address"));
        };
        let Some(width) = site.width else {
            return un(at("the write has no known width"));
        };
        if site.addr.is_none() {
            return un(at("the address has no exact witness (not a global access, or an unmodelled operation)"));
        }
        let Some(lanes) = lane_model(ev, &site.lane_facts) else {
            // No work-item satisfies the EXEC predicates: the write never executes.
            continue;
        };
        ev.set_lanes(lanes);
        let a = match ev.eval(site.addr) {
            Ok(a) => a,
            Err(e) => return un(at(&format!("address not affine in the tile id: {e}"))),
        };
        let base = a
            .usym
            .iter()
            .find(|(&k, _)| matches!(ev.it.node(k), Some(Node::K64(o)) if o == arg.offset));
        let Some((&base_ix, &coef)) = base else {
            return un(at("the address is not based on this argument's pointer"));
        };
        if coef != 1 {
            return un(at("the base pointer is scaled"));
        }
        let mut usym = a.usym.clone();
        usym.remove(&base_ix);
        if usym
            .keys()
            .any(|&k| matches!(ev.it.node(k), Some(Node::K64(o)) if pointers.contains(&o)))
        {
            return un(at("another argument pointer enters the address"));
        }
        let rels = guard::rels(ev, &site.facts);
        let mut a_nb = a.clone();
        a_nb.usym = usym.clone();
        forms.push(SiteForm {
            usym,
            uc: a.uc,
            s: a.s,
            rlo: a.rlo,
            rhi: a.rhi,
            ext: a.ext.map(|e| *e),
            width: i128::from(width),
            req: a.req,
            a: a_nb,
            rels,
        });
    }
    let _ = facts;
    if forms.is_empty() {
        return un("every write is under an unsatisfiable lane predicate".into());
    }
    let req = forms.iter().any(|f| f.req);
    if std::env::var("WGDBG2").is_ok() {
        for w in &writes {
            if let Some(s) = sites.get(&w.inst) {
                let f: Vec<String> = s
                    .facts
                    .iter()
                    .map(|(a, p)| format!("{}{}", if *p { "" } else { "!" }, show(&ev.it, *a, 6)))
                    .collect();
                let l: Vec<String> = s
                    .lane_facts
                    .iter()
                    .map(|(a, p)| format!("{}{}", if *p { "" } else { "!" }, show(&ev.it, *a, 6)))
                    .collect();
                eprintln!(
                    "DBG arg@{} pc{} facts {:?} lane {:?} node {:?}",
                    arg.offset,
                    s.pos,
                    f,
                    l,
                    s.lane_facts
                        .iter()
                        .map(|(a, _)| ev.it.node(*a))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    let mut requires: Vec<WgRequirement> = match (req, ev.group_size_dword) {
        (true, Some(o)) => vec![WgRequirement {
            offset: o,
            mask: 0xffff,
            value: launch.block_x,
            op: WgReqOp::MaskEq,
        }],
        _ => Vec::new(),
    };
    if forms.iter().any(|f| f.a.has_prod_deep()) {
        let gsites: Vec<guard::Site<'_>> = forms
            .iter()
            .map(|f| guard::Site {
                a: &f.a,
                width: f.width,
                rels: &f.rels,
            })
            .collect();
        return match guard::prove(ev, &gsites) {
            Ok(p) => {
                requires.extend(p.needs);
                requires.sort();
                ArgWgClass::Disjoint(WgDisjoint {
                    axes: p.axes,
                    bounds: p.bounds,
                    block_x: launch.block_x,
                    requires,
                    sites: forms.len(),
                })
            }
            Err(e) => un(format!("guarded range proof failed: {e}")),
        };
    }
    let first = &forms[0];
    for f in &forms[1..] {
        if f.usym != first.usym {
            return un("sites differ in their uniform (kernarg) offsets".into());
        }
        if f.s != first.s {
            return un("sites have different tile coefficients".into());
        }
        match (&f.ext, &first.ext) {
            (None, None) => {}
            (Some(x), Some(y))
                if x.signed == y.signed
                    && x.scale == y.scale
                    && x.inner.s == y.inner.s
                    && x.inner.usym == y.inner.usym => {}
            _ => return un("sites differ in the shape of their tile term".into()),
        }
    }
    for cand in BOUND_CANDIDATES {
        let bounds = [
            cand.min(HW_BOUNDS[0]),
            cand.min(HW_BOUNDS[1]),
            cand.min(HW_BOUNDS[2]),
        ];
        if let Some(axes) = separated(&forms, bounds) {
            return ArgWgClass::Disjoint(WgDisjoint {
                axes,
                bounds,
                block_x: launch.block_x,
                requires,
                sites: forms.len(),
            });
        }
    }
    un(format!(
        "{} (tile term {}, windows too wide for the tile stride at every tile bound down to 256)",
        describe(&forms),
        if first.ext.is_some() {
            "extended"
        } else {
            "direct"
        }
    ))
}

fn describe(forms: &[SiteForm]) -> String {
    let f = &forms[0];
    let (s, win) = match &f.ext {
        Some(e) => (
            e.inner.s,
            format!(
                "c={} inner window {}..{}",
                e.scale, e.inner.rlo, e.inner.rhi
            ),
        ),
        None => (f.s, format!("window {}..{}", f.rlo, f.rhi)),
    };
    format!("strides x={} y={} z={}, {win}", s[0], s[1], s[2])
}

/// Mixed-radix separation of the tile windows under tile id bounds `b`; the
/// axes of the footprint when it holds.
fn separated(forms: &[SiteForm], b: [u64; 3]) -> Option<Vec<u8>> {
    let first = &forms[0];
    let stride = match &first.ext {
        Some(e) => e.inner.s,
        None => first.s,
    };
    let axes: Vec<u8> = (0..3u8).filter(|&a| stride[a as usize] != 0).collect();
    if axes.is_empty() {
        return None;
    }
    let mut ordered: Vec<(i128, i128)> = axes
        .iter()
        .map(|&a| (stride[a as usize].abs(), i128::from(b[a as usize]) - 1))
        .collect();
    ordered.sort();
    let mut acc: i128 = 0;
    let mut sepmin = i128::MAX;
    for &(s, e1) in &ordered {
        sepmin = sepmin.min(s.checked_sub(acc)?);
        acc = acc.checked_add(s.checked_mul(e1)?)?;
    }
    let tmax = acc;
    match &first.ext {
        None => {
            let lo = forms.iter().map(|f| f.uc + f.rlo).min()?;
            let hi = forms.iter().map(|f| f.uc + f.rhi + f.width - 1).max()?;
            let wd = hi - lo + 1;
            (sepmin >= wd && tmax.checked_add(wd)? <= 1i128 << 62).then_some(axes)
        }
        Some(e) => {
            let (ilo, ihi) = (
                forms
                    .iter()
                    .map(|f| f.ext.as_ref().map(|e| e.inner.uc + e.inner.rlo))
                    .min()??,
                forms
                    .iter()
                    .map(|f| f.ext.as_ref().map(|e| e.inner.uc + e.inner.rhi))
                    .max()??,
            );
            let rw = ihi - ilo;
            let m = sepmin - rw;
            let (qlo, qhi) = (
                forms.iter().map(|f| f.uc + f.rlo).min()?,
                forms.iter().map(|f| f.uc + f.rhi + f.width - 1).max()?,
            );
            let wb = qhi - qlo + 1;
            let c = e.scale.abs();
            (m >= 1
                && c < 1i128 << 30
                && tmax.checked_add(sepmin)? <= 1i128 << 32
                && c.checked_mul(m)? >= wb)
                .then_some(axes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Body;
    use crate::descriptor::{
        KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3,
    };
    use crate::inst::{KernelOrigin, SymbolId, Wave};
    use crate::metadata::{HsaKernelMetadata, Kernarg, KernelMeta};

    /// A gfx1201 wave32 HSA kernel from llvm-mc words with one pointer argument
    /// at offset 0 (kernarg pointer in s[0:1], one x work-item id).
    fn kernel(words: &[u32]) -> Kernel {
        let mut body = Body::default();
        let mut at = 0;
        while at < words.len() {
            let (inst, used) = crate::codec::gfx12::decode(&words[at..]).expect("decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            at += used;
        }
        crate::passes::cfg::build_blocks(&mut body, Arch::Gfx1201).expect("blocks");
        let descriptor = KernelDescriptor {
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            kernarg_size: 8,
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
            kernarg_segment_size: 8,
            kernarg_segment_align: 8,
            wavefront_size: 32,
            args: vec![Kernarg {
                name: "p".into(),
                offset: 0,
                size: 8,
                value_kind: "global_buffer".into(),
                address_space: Some("global".into()),
            }],
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

    fn class(words: &[u32]) -> ArgWgClass {
        let r = analyze(
            &kernel(words),
            Arch::Gfx1201,
            false,
            WgLaunch { block_x: 64 },
        );
        assert_eq!(r.applicable, Ok(()));
        r.arg(0).expect("pointer arg").clone()
    }

    // s_load_b64 s[2:3],s[0:1],0; v_lshl_add_u32 v1,ttmp9,6,v0; v_lshlrev_b32 v1,2,v1;
    // s_wait_kmcnt 0; v_mov_b32 v2,0; global_store_b32 v1,v2,s[2:3]; s_endpgm
    const ROW: [u32; 11] = [
        0xf4002080, 0xf8000000, 0xd6460001, 0x04010c75, 0x30020282, 0xbfc70000, 0x7e040280,
        0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
    ];

    #[test]
    fn row_times_stride_plus_lane_is_disjoint_along_x() {
        // address = P + 4 * (wg_x * 64 + lane), lane < 64.
        match class(&ROW) {
            ArgWgClass::Disjoint(d) => {
                assert_eq!(d.axes, [0]);
                assert_eq!(d.block_x, 64);
                // 256 * E <= 2^32 (32-bit offset arithmetic): the proof needs E <= 2^24.
                assert_eq!(d.bounds[0], 1 << 24);
                assert!(d.requires.is_empty());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn store_independent_of_the_workgroup_gets_no_fact() {
        // address = P + 4 * lane
        let k = [
            0xf4002080, 0xf8000000, 0x30020082, 0xbfc70000, 0x7e040280, 0xee068002, 0x01000000,
            0x00000001, 0xbfb00000,
        ];
        assert!(matches!(class(&k), ArgWgClass::Unproven(_)));
    }

    #[test]
    fn stride_smaller_than_the_lane_window_gets_no_fact() {
        // address = P + 4 * (wg_x * 16 + lane), lane < 64: windows overlap.
        let k = [
            0xf4002080, 0xf8000000, 0xd6460001, 0x04010875, 0x30020282, 0xbfc70000, 0x7e040280,
            0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
        ];
        assert!(matches!(class(&k), ArgWgClass::Unproven(_)));
    }

    #[test]
    fn atomic_only_argument_needs_no_fact() {
        // global_atomic_add_u32 v1,v2,s[2:3] (no return): commutative.
        let k = [
            0xf4002080, 0xf8000000, 0x7e020280, 0x7e040281, 0xbfc70000, 0xee0d4002, 0x01000000,
            0x00000001, 0xbfb00000,
        ];
        assert_eq!(class(&k), ArgWgClass::AtomicOnly);
    }

    #[test]
    fn unsupported_block_sizes_and_geometries_are_not_applicable() {
        let r = analyze(&kernel(&ROW), Arch::Gfx1201, false, WgLaunch { block_x: 0 });
        assert!(r.applicable.is_err());
        let r = analyze(
            &kernel(&ROW),
            Arch::Gfx1100,
            false,
            WgLaunch { block_x: 64 },
        );
        assert!(r.applicable.is_err());
    }

    fn direct(s: [i128; 3], rlo: i128, rhi: i128, width: i128) -> Vec<SiteForm> {
        vec![SiteForm {
            usym: BTreeMap::new(),
            uc: 0,
            s,
            rlo,
            rhi,
            ext: None,
            width,
            req: false,
            a: Aff::constant(64, 0),
            rels: Vec::new(),
        }]
    }

    #[test]
    fn two_dimensional_tiles_need_the_mixed_radix_condition() {
        // x stride 4 over a 256-id extent (1024 bytes), y stride 1024: tiles {x,y}.
        let f = direct([4, 1024, 0], 0, 0, 4);
        assert_eq!(separated(&f, [256, 256, 1]), Some(vec![0, 1]));
        // y stride too small for the x extent: no fact.
        let f = direct([4, 1020, 0], 0, 0, 4);
        assert_eq!(separated(&f, [256, 256, 1]), None);
        // a smaller x extent restores it.
        assert_eq!(separated(&f, [255, 256, 1]), Some(vec![0, 1]));
    }

    #[test]
    fn window_wider_than_the_stride_is_refused() {
        let f = direct([64, 0, 0], 0, 64, 4);
        assert_eq!(separated(&f, [16, 1, 1]), None);
        let f = direct([68, 0, 0], 0, 64, 4);
        assert_eq!(separated(&f, [16, 1, 1]), Some(vec![0]));
    }
}
