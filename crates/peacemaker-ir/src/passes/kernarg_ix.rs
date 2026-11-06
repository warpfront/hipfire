// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The separate integer/address witness of the kernarg pass (railgun design
//! A, amendment 2026-10-25 §A9.3).
//!
//! Every 32-bit location may additionally carry an [`Ix`]: a hash-consed node
//! naming the *exact* bit-vector expression the register holds. It is
//! independent of the legacy `roots`/`src`/`taint` lattice (a loaded `i32`
//! stays `OPAQUE` there). Nodes are 32-bit values or 64-bit values; a 64-bit
//! value lives in a register pair as `lo_of(w)`/`hi_of(w)`, and a pair names a
//! 64-bit value again only through [`Interner::as64`], which accepts exactly
//! matching halves (so a high half alone, mismatched halves, or two loaded
//! 32-bit halves never form an address).
//!
//! Unsupported arithmetic, widths above the caps, and joins of different
//! expressions produce [`Ix::NONE`]; nothing here widens to a guess.
use std::collections::HashMap;

use crate::cfg::InstId;
use crate::kernarg::{EntryReg, IdxExpr, IndexExt};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(super) struct Ix(u32);
impl Ix {
    /// No witness: the value is unknown to the integer/address domain.
    pub const NONE: Ix = Ix(0);
    /// Identity of [`Ix::join`]: the value of a location no lane holds (the
    /// disabled-lane stacks of `State`). Never a witness; reads treat it as
    /// `NONE`.
    pub const BOT: Ix = Ix(u32::MAX);
    pub fn is_none(self) -> bool {
        self.0 == 0 || self.0 == u32::MAX
    }
    /// Equal witnesses survive; anything else is unknown (guarded joins only:
    /// a join of different expressions never invents a common one).
    pub fn join(self, other: Ix) -> Ix {
        match (self, other) {
            (Ix::BOT, x) | (x, Ix::BOT) => x,
            (a, b) if a == b => a,
            _ => Ix::NONE,
        }
    }
    /// The witness as an operand: `BOT` reads as `NONE`.
    pub fn live(self) -> Ix {
        if self == Ix::BOT {
            Ix::NONE
        } else {
            self
        }
    }
}

/// `lvl` of a [`Node::Carry`] produced by a scalar add (no EXEC dependency).
pub(super) const CARRY_SCALAR: u8 = 0xff;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) enum Node {
    // 32-bit values.
    C32(u32),
    /// The kernarg dword at byte offset `o` (4-aligned).
    K32(u32),
    Entry(EntryReg),
    /// The `i32` loaded by that instruction (dynamic instance unique: loads in
    /// CFG cycles never produce one).
    Load(InstId),
    Lo(Ix),
    Hi(Ix),
    /// Arithmetic shift right by 31 (the sign word).
    Sra31(Ix),
    Add32(Ix, Ix),
    Sub32(Ix, Ix),
    Mul32(Ix, Ix),
    Shl32(Ix, u8),
    /// Carry-out lane mask of `a + b`; valid only as the carry-in of the
    /// matching high add, under the same EXEC level (`lvl`).
    Carry {
        a: Ix,
        b: Ix,
        lvl: u8,
    },
    // 64-bit values.
    C64(u64),
    /// The kernarg qword at byte offset `o` (8-aligned).
    K64(u32),
    Sext(Ix),
    Zext(Ix),
    Add64(Ix, Ix),
    Mul64(Ix, Ix),
    Shl64(Ix, u8),
    /// The 64-bit pointer a sealed-table entry load produced (dynamic instance
    /// unique: such loads never sit in CFG cycles).
    Ent(InstId),
    // Boolean flags (1 bit): SCC and lane-uniform masks the sealed analysis
    // follows exactly. Atoms compare exact 32-bit operands (or a 64-bit value
    // against zero); connectives combine flags.
    Ltu {
        a: Ix,
        b: Ix,
    },
    Lts {
        a: Ix,
        b: Ix,
    },
    Eq {
        a: Ix,
        b: Ix,
    },
    /// The 64-bit value is zero.
    Zero64(Ix),
    /// The flag that always holds.
    True,
    Not(Ix),
    Or(Ix, Ix),
    And(Ix, Ix),
    // 32-bit values holding a flag.
    /// `-1` if the flag holds, else `0`.
    Mask(Ix),
    /// `1` if the flag holds, else `0`.
    Bit(Ix),
    /// Some value that is nonzero exactly when the flag holds.
    NzFlag(Ix),
}
impl Node {
    fn bits(self) -> u8 {
        match self {
            Node::Ltu { .. }
            | Node::Lts { .. }
            | Node::Eq { .. }
            | Node::Zero64(_)
            | Node::True
            | Node::Not(_)
            | Node::Or(..)
            | Node::And(..) => 1,
            Node::Ent(_)
            | Node::C64(_)
            | Node::K64(_)
            | Node::Sext(_)
            | Node::Zext(_)
            | Node::Add64(..)
            | Node::Mul64(..)
            | Node::Shl64(..) => 64,
            _ => 32,
        }
    }
}

const MAX_DEPTH: u16 = 40;
const MAX_NODES: usize = 20_000;

#[derive(Default)]
pub(super) struct Interner {
    nodes: Vec<(Node, u16)>,
    map: HashMap<Node, Ix>,
}

impl Interner {
    pub fn node(&self, ix: Ix) -> Option<Node> {
        (!ix.is_none()).then(|| self.nodes[ix.0 as usize - 1].0)
    }
    fn depth(&self, ix: Ix) -> u16 {
        if ix.is_none() {
            0
        } else {
            self.nodes[ix.0 as usize - 1].1
        }
    }
    fn mk(&mut self, n: Node, children: &[Ix]) -> Ix {
        if let Some(&ix) = self.map.get(&n) {
            return ix;
        }
        let depth = children.iter().map(|&c| self.depth(c)).max().unwrap_or(0) + 1;
        if depth > MAX_DEPTH || self.nodes.len() >= MAX_NODES {
            return Ix::NONE;
        }
        self.nodes.push((n, depth));
        let ix = Ix(self.nodes.len() as u32);
        self.map.insert(n, ix);
        ix
    }
    fn bits(&self, ix: Ix) -> Option<u8> {
        self.node(ix).map(Node::bits)
    }
    fn is32(&self, ix: Ix) -> bool {
        self.bits(ix) == Some(32) && !matches!(self.node(ix), Some(Node::Carry { .. }))
    }
    fn is64(&self, ix: Ix) -> bool {
        self.bits(ix) == Some(64)
    }

    pub fn c32(&mut self, v: u32) -> Ix {
        self.mk(Node::C32(v), &[])
    }
    pub fn k32(&mut self, offset: u32) -> Ix {
        self.mk(Node::K32(offset), &[])
    }
    pub fn entry(&mut self, r: EntryReg) -> Ix {
        self.mk(Node::Entry(r), &[])
    }
    pub fn load(&mut self, inst: InstId) -> Ix {
        self.mk(Node::Load(inst), &[])
    }
    /// A 64-bit constant from a signed immediate (two's complement).
    pub fn imm64(&mut self, v: i64) -> Ix {
        self.c64(v as u64)
    }
    fn c64(&mut self, v: u64) -> Ix {
        self.mk(Node::C64(v), &[])
    }
    fn k64(&mut self, offset: u32) -> Ix {
        self.mk(Node::K64(offset), &[])
    }
    pub fn const32(&self, ix: Ix) -> Option<u32> {
        match self.node(ix)? {
            Node::C32(v) => Some(v),
            _ => None,
        }
    }
    fn const64(&self, ix: Ix) -> Option<u64> {
        match self.node(ix)? {
            Node::C64(v) => Some(v),
            _ => None,
        }
    }

    pub fn add32(&mut self, a: Ix, b: Ix) -> Ix {
        if !self.is32(a) || !self.is32(b) {
            return Ix::NONE;
        }
        match (self.const32(a), self.const32(b)) {
            (Some(x), Some(y)) => return self.c32(x.wrapping_add(y)),
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            _ => {}
        }
        let (a, b) = if b < a { (b, a) } else { (a, b) };
        self.mk(Node::Add32(a, b), &[a, b])
    }
    pub fn sub32(&mut self, a: Ix, b: Ix) -> Ix {
        if !self.is32(a) || !self.is32(b) {
            return Ix::NONE;
        }
        match (self.const32(a), self.const32(b)) {
            (Some(x), Some(y)) => return self.c32(x.wrapping_sub(y)),
            (_, Some(0)) => return a,
            _ => {}
        }
        self.mk(Node::Sub32(a, b), &[a, b])
    }
    pub fn mul32(&mut self, a: Ix, b: Ix) -> Ix {
        if !self.is32(a) || !self.is32(b) {
            return Ix::NONE;
        }
        match (self.const32(a), self.const32(b)) {
            (Some(x), Some(y)) => return self.c32(x.wrapping_mul(y)),
            (Some(1), _) => return b,
            (_, Some(1)) => return a,
            (Some(0), _) | (_, Some(0)) => return self.c32(0),
            _ => {}
        }
        let (a, b) = if b < a { (b, a) } else { (a, b) };
        self.mk(Node::Mul32(a, b), &[a, b])
    }
    pub fn shl32(&mut self, a: Ix, k: u32) -> Ix {
        if !self.is32(a) || k >= 32 {
            return Ix::NONE;
        }
        if k == 0 {
            return a;
        }
        if let Some(x) = self.const32(a) {
            return self.c32(x << k);
        }
        self.mk(Node::Shl32(a, k as u8), &[a])
    }
    pub fn sra31(&mut self, a: Ix) -> Ix {
        if !self.is32(a) {
            return Ix::NONE;
        }
        if let Some(x) = self.const32(a) {
            return self.c32(((x as i32) >> 31) as u32);
        }
        self.mk(Node::Sra31(a), &[a])
    }
    pub fn carry(&mut self, a: Ix, b: Ix, lvl: u8) -> Ix {
        if !self.is32(a) || !self.is32(b) {
            return Ix::NONE;
        }
        self.mk(Node::Carry { a, b, lvl }, &[a, b])
    }

    pub fn ent(&mut self, inst: InstId) -> Ix {
        self.mk(Node::Ent(inst), &[])
    }
    pub fn flag_true(&mut self) -> Ix {
        self.mk(Node::True, &[])
    }
    pub fn is_flag(&self, ix: Ix) -> bool {
        self.bits(ix) == Some(1)
    }
    fn flag_atom(&mut self, n: Node, a: Ix, b: Ix) -> Ix {
        if !self.is32(a) || !self.is32(b) {
            return Ix::NONE;
        }
        self.mk(n, &[a, b])
    }
    /// `a <u b`.
    pub fn ltu(&mut self, a: Ix, b: Ix) -> Ix {
        self.flag_atom(Node::Ltu { a, b }, a, b)
    }
    /// `a <s b`.
    pub fn lts(&mut self, a: Ix, b: Ix) -> Ix {
        self.flag_atom(Node::Lts { a, b }, a, b)
    }
    /// `a == b` (operands ordered: equality is symmetric).
    pub fn eq32(&mut self, a: Ix, b: Ix) -> Ix {
        let (a, b) = if b < a { (b, a) } else { (a, b) };
        self.flag_atom(Node::Eq { a, b }, a, b)
    }
    /// The 64-bit `w` is zero.
    pub fn zero64(&mut self, w: Ix) -> Ix {
        if !self.is64(w) {
            return Ix::NONE;
        }
        self.mk(Node::Zero64(w), &[w])
    }
    pub fn flag_not(&mut self, f: Ix) -> Ix {
        if !self.is_flag(f) {
            return Ix::NONE;
        }
        match self.node(f) {
            Some(Node::Not(g)) => g,
            _ => self.mk(Node::Not(f), &[f]),
        }
    }
    pub fn flag_or(&mut self, f: Ix, g: Ix) -> Ix {
        if !self.is_flag(f) || !self.is_flag(g) {
            return Ix::NONE;
        }
        let (f, g) = if g < f { (g, f) } else { (f, g) };
        self.mk(Node::Or(f, g), &[f, g])
    }
    pub fn flag_and(&mut self, f: Ix, g: Ix) -> Ix {
        if !self.is_flag(f) || !self.is_flag(g) {
            return Ix::NONE;
        }
        let (f, g) = if g < f { (g, f) } else { (f, g) };
        self.mk(Node::And(f, g), &[f, g])
    }
    pub fn mask(&mut self, f: Ix) -> Ix {
        if !self.is_flag(f) {
            return Ix::NONE;
        }
        self.mk(Node::Mask(f), &[f])
    }
    pub fn bit(&mut self, f: Ix) -> Ix {
        if !self.is_flag(f) {
            return Ix::NONE;
        }
        self.mk(Node::Bit(f), &[f])
    }
    pub fn nz_flag(&mut self, f: Ix) -> Ix {
        if !self.is_flag(f) {
            return Ix::NONE;
        }
        self.mk(Node::NzFlag(f), &[f])
    }
    /// `(flag, value when it holds)` of a flag-holding 32-bit value.
    pub fn flag_value(&self, ix: Ix) -> Option<(Ix, u32)> {
        match self.node(ix)? {
            Node::Mask(f) => Some((f, u32::MAX)),
            Node::Bit(f) => Some((f, 1)),
            _ => None,
        }
    }
    /// The atoms (with polarity) that must hold for `flag` to evaluate to
    /// `value`: through `Not`, a true `And`, a false `Or`. Nothing else is
    /// concluded (a false `And` or true `Or` implies no single atom).
    pub fn assume(&self, flag: Ix, value: bool, out: &mut Vec<(Ix, bool)>) {
        match self.node(flag) {
            Some(Node::Not(g)) => self.assume(g, !value, out),
            Some(Node::And(f, g)) if value => {
                self.assume(f, true, out);
                self.assume(g, true, out);
            }
            Some(Node::Or(f, g)) if !value => {
                self.assume(f, false, out);
                self.assume(g, false, out);
            }
            Some(Node::Ltu { .. } | Node::Lts { .. } | Node::Eq { .. } | Node::Zero64(_)) => {
                out.push((flag, value))
            }
            _ => {}
        }
    }

    fn sext(&mut self, a: Ix) -> Ix {
        if !self.is32(a) {
            return Ix::NONE;
        }
        if let Some(x) = self.const32(a) {
            return self.c64(x as i32 as i64 as u64);
        }
        self.mk(Node::Sext(a), &[a])
    }
    pub fn zext(&mut self, a: Ix) -> Ix {
        if !self.is32(a) {
            return Ix::NONE;
        }
        if let Some(x) = self.const32(a) {
            return self.c64(u64::from(x));
        }
        self.mk(Node::Zext(a), &[a])
    }
    pub fn add64(&mut self, a: Ix, b: Ix) -> Ix {
        if !self.is64(a) || !self.is64(b) {
            return Ix::NONE;
        }
        match (self.const64(a), self.const64(b)) {
            (Some(x), Some(y)) => return self.c64(x.wrapping_add(y)),
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            _ => {}
        }
        let (a, b) = if b < a { (b, a) } else { (a, b) };
        self.mk(Node::Add64(a, b), &[a, b])
    }
    pub fn mul64(&mut self, a: Ix, b: Ix) -> Ix {
        if !self.is64(a) || !self.is64(b) {
            return Ix::NONE;
        }
        match (self.const64(a), self.const64(b)) {
            (Some(x), Some(y)) => return self.c64(x.wrapping_mul(y)),
            (Some(1), _) => return b,
            (_, Some(1)) => return a,
            (Some(0), _) | (_, Some(0)) => return self.c64(0),
            _ => {}
        }
        let (a, b) = if b < a { (b, a) } else { (a, b) };
        self.mk(Node::Mul64(a, b), &[a, b])
    }
    pub fn shl64(&mut self, a: Ix, k: u32) -> Ix {
        if !self.is64(a) || k >= 64 {
            return Ix::NONE;
        }
        if k == 0 {
            return a;
        }
        if let Some(x) = self.const64(a) {
            return self.c64(x << k);
        }
        self.mk(Node::Shl64(a, k as u8), &[a])
    }

    /// The low dword of 64-bit `w` as a 32-bit node. Exact identities only:
    /// the low half of a sum/product/shift depends on the low halves alone.
    pub fn lo_of(&mut self, w: Ix) -> Ix {
        match self.node(w) {
            Some(Node::C64(v)) => self.c32(v as u32),
            Some(Node::K64(o)) => self.k32(o),
            Some(Node::Sext(x) | Node::Zext(x)) => x,
            Some(Node::Add64(a, b)) => {
                let (la, lb) = (self.lo_of(a), self.lo_of(b));
                self.add32(la, lb)
            }
            Some(Node::Mul64(a, b)) => {
                let (la, lb) = (self.lo_of(a), self.lo_of(b));
                self.mul32(la, lb)
            }
            Some(Node::Shl64(a, k)) if k < 32 => {
                let la = self.lo_of(a);
                self.shl32(la, u32::from(k))
            }
            Some(_) if self.is64(w) => self.mk(Node::Lo(w), &[w]),
            _ => Ix::NONE,
        }
    }
    pub fn hi_of(&mut self, w: Ix) -> Ix {
        match self.node(w) {
            Some(Node::C64(v)) => self.c32((v >> 32) as u32),
            Some(Node::K64(o)) => self.k32(o + 4),
            Some(Node::Sext(x)) => self.sra31(x),
            Some(Node::Zext(_)) => self.c32(0),
            Some(_) if self.is64(w) => self.mk(Node::Hi(w), &[w]),
            _ => Ix::NONE,
        }
    }
    /// The 64-bit value whose halves are exactly `lo`/`hi`, if they name one.
    pub fn as64(&mut self, lo: Ix, hi: Ix) -> Ix {
        if !self.is32(lo) || !self.is32(hi) {
            return Ix::NONE;
        }
        match (self.node(lo), self.node(hi)) {
            (Some(Node::C32(l)), Some(Node::C32(h))) => {
                return self.c64(u64::from(h) << 32 | u64::from(l))
            }
            (Some(Node::K32(a)), Some(Node::K32(b))) if a % 8 == 0 && b == a + 4 => {
                return self.k64(a)
            }
            (_, Some(Node::Hi(w))) => {
                if self.lo_of(w) == lo {
                    return w;
                }
            }
            (_, Some(Node::C32(0))) => return self.zext(lo),
            (_, Some(Node::Sra31(x))) if x == lo => return self.sext(lo),
            _ => {}
        }
        Ix::NONE
    }

    /// `(a, b)` such that `carry` is `Carry { a, b }` at EXEC level `lvl`.
    pub fn carry_parts(&self, carry: Ix, lvl: u8) -> Option<(Ix, Ix)> {
        match self.node(carry)? {
            Node::Carry { a, b, lvl: l } if l == lvl => Some((a, b)),
            _ => None,
        }
    }

    /// Terms of the top-level `Add64` chain of `w`.
    fn terms(&self, w: Ix, out: &mut Vec<Ix>) {
        match self.node(w) {
            Some(Node::Add64(a, b)) => {
                self.terms(a, out);
                self.terms(b, out);
            }
            _ => out.push(w),
        }
    }
}

/// What a kernarg dword offset is, for the certificate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KargKind {
    /// Inside the pointer argument starting at `arg_offset`.
    Pointer {
        arg_offset: u32,
    },
    /// Inside a declared non-pointer (typed scalar / hidden geometry) argument.
    Scalar,
    Other,
}

/// A complete address witness: one direct pointer argument plus an integer
/// displacement. `loads` lists the `Load(i)` leaves of `disp` in order.
pub(super) struct Cert {
    pub base_arg_offset: u32,
    pub disp: IdxExpr,
    pub loads: Vec<InstId>,
}

/// Qualify the 64-bit address `addr` (§A9.3 rule 3): exactly one live direct
/// pointer argument term, every other term an integer expression over
/// approved leaves. Pointer values multiplied/shifted/truncated into the
/// displacement, loaded address words, carries and unmatched halves refuse.
pub(super) fn certify(
    it: &Interner,
    addr: Ix,
    classify: &dyn Fn(u32) -> KargKind,
) -> Result<Cert, &'static str> {
    let cert = split(it, addr, classify)?;
    if cert.loads.is_empty() {
        return Err("witness-no-index-load");
    }
    Ok(cert)
}

/// Like [`certify`] for an index source: no loads allowed in the address (no
/// nested `P[Q[i]]`); the displacement may be the empty sum.
pub(super) fn certify_source(
    it: &Interner,
    addr: Ix,
    classify: &dyn Fn(u32) -> KargKind,
) -> Result<(u32, IdxExpr), &'static str> {
    let cert = split(it, addr, classify)?;
    if !cert.loads.is_empty() {
        return Err("witness-nested-index");
    }
    Ok((cert.base_arg_offset, cert.disp))
}

/// `addr` as one direct pointer argument plus a displacement.
fn split(
    it: &Interner,
    addr: Ix,
    classify: &dyn Fn(u32) -> KargKind,
) -> Result<Cert, &'static str> {
    if !it.is64(addr) {
        return Err("witness-not-64bit");
    }
    let mut terms = Vec::new();
    it.terms(addr, &mut terms);
    let mut base = None;
    let mut rest = Vec::new();
    for t in terms {
        if let Some(Node::K64(o)) = it.node(t) {
            if let KargKind::Pointer { arg_offset } = classify(o) {
                // The whole pointer value, exactly once: a second pointer term
                // is pointer + pointer arithmetic.
                if arg_offset != o || base.replace(o).is_some() {
                    return Err("witness-base-shape");
                }
                continue;
            }
        }
        rest.push(t);
    }
    let Some(base_arg_offset) = base else {
        return Err("witness-no-base");
    };
    let mut loads = Vec::new();
    let mut disp: Option<IdxExpr> = None;
    for t in rest {
        let e = to_expr(it, t, classify, &mut loads)?;
        disp = Some(match disp {
            None => e,
            Some(d) => IdxExpr::Add(Box::new(d), Box::new(e)),
        });
    }
    Ok(Cert {
        base_arg_offset,
        disp: disp.unwrap_or(IdxExpr::Const { value: 0, bits: 64 }),
        loads,
    })
}

fn to_expr(
    it: &Interner,
    ix: Ix,
    classify: &dyn Fn(u32) -> KargKind,
    loads: &mut Vec<InstId>,
) -> Result<IdxExpr, &'static str> {
    let node = it.node(ix).ok_or("witness-unknown")?;
    let b = |x: Ix, loads: &mut Vec<InstId>| -> Result<Box<IdxExpr>, &'static str> {
        Ok(Box::new(to_expr(it, x, classify, loads)?))
    };
    Ok(match node {
        Node::C32(v) => IdxExpr::Const {
            value: u64::from(v),
            bits: 32,
        },
        Node::C64(v) => IdxExpr::Const { value: v, bits: 64 },
        Node::K32(o) => {
            if classify(o) != KargKind::Scalar {
                return Err("witness-pointer-in-offset");
            }
            IdxExpr::Kernarg {
                offset: o,
                bits: 32,
            }
        }
        Node::K64(o) => {
            if classify(o) != KargKind::Scalar || classify(o + 4) != KargKind::Scalar {
                return Err("witness-pointer-in-offset");
            }
            IdxExpr::Kernarg {
                offset: o,
                bits: 64,
            }
        }
        Node::Entry(r) => IdxExpr::Entry(r),
        Node::Load(inst) => {
            let i = match loads.iter().position(|&l| l == inst) {
                Some(i) => i,
                None => {
                    loads.push(inst);
                    loads.len() - 1
                }
            };
            IdxExpr::Load(i as u16)
        }
        Node::Sext(a) => IdxExpr::Ext {
            signed: true,
            a: b(a, loads)?,
        },
        Node::Zext(a) => IdxExpr::Ext {
            signed: false,
            a: b(a, loads)?,
        },
        Node::Shl32(a, k) | Node::Shl64(a, k) => IdxExpr::Shl {
            a: b(a, loads)?,
            amount: k,
        },
        Node::Add32(x, y) | Node::Add64(x, y) => IdxExpr::Add(b(x, loads)?, b(y, loads)?),
        Node::Sub32(x, y) => IdxExpr::Sub(b(x, loads)?, b(y, loads)?),
        Node::Mul32(x, y) | Node::Mul64(x, y) => IdxExpr::Mul(b(x, loads)?, b(y, loads)?),
        Node::Lo(_) | Node::Hi(_) | Node::Sra31(_) | Node::Carry { .. } => {
            return Err("witness-unmatched-half")
        }
        Node::Ent(_)
        | Node::Ltu { .. }
        | Node::Lts { .. }
        | Node::Eq { .. }
        | Node::Zero64(_)
        | Node::True
        | Node::Not(_)
        | Node::Or(..)
        | Node::And(..)
        | Node::Mask(_)
        | Node::Bit(_)
        | Node::NzFlag(_) => return Err("witness-boolean-in-address"),
    })
}

/// Per `Load(i)` leaf of `e`, how its consumer uses it.
pub(super) fn load_ext(e: &IdxExpr, out: &mut std::collections::BTreeMap<u16, Vec<IndexExt>>) {
    fn go(
        e: &IdxExpr,
        parent: Option<IndexExt>,
        out: &mut std::collections::BTreeMap<u16, Vec<IndexExt>>,
    ) {
        match e {
            IdxExpr::Load(i) => {
                let v = out.entry(*i).or_default();
                let u = parent.unwrap_or(IndexExt::Raw32);
                if !v.contains(&u) {
                    v.push(u);
                    v.sort();
                }
            }
            IdxExpr::Ext { signed, a } => go(
                a,
                Some(if *signed {
                    IndexExt::Sext64
                } else {
                    IndexExt::Zext64
                }),
                out,
            ),
            IdxExpr::Shl { a, .. } => go(a, None, out),
            IdxExpr::Add(a, b) | IdxExpr::Sub(a, b) | IdxExpr::Mul(a, b) => {
                go(a, None, out);
                go(b, None, out);
            }
            _ => {}
        }
    }
    go(e, None, out);
}

/// The shape of a sealed-table entry address: `table pointer argument +
/// (ext(index) << 3)` with a 32-bit `index` that is the value loaded by one
/// instruction, or a kernarg scalar. The entry stride is the 8-byte pointer.
pub(super) struct TableShape {
    pub table_arg_offset: u32,
    /// The 32-bit index value (`Load(_)` or `K32(_)`).
    pub index: Ix,
}

/// Qualify the 64-bit address of a sealed-table entry load. Anything else
/// (an extra term, another scale, a second pointer, a loaded address word)
/// refuses; nothing widens to a guess.
pub(super) fn table_entry(
    it: &Interner,
    addr: Ix,
    classify: &dyn Fn(u32) -> KargKind,
) -> Result<TableShape, &'static str> {
    if !it.is64(addr) {
        return Err("sealed-entry-not-64bit");
    }
    let mut terms = Vec::new();
    it.terms(addr, &mut terms);
    if terms.len() != 2 {
        return Err("sealed-entry-shape");
    }
    let mut table = None;
    let mut scaled = None;
    for t in terms {
        match it.node(t) {
            Some(Node::K64(o)) => match classify(o) {
                KargKind::Pointer { arg_offset } if arg_offset == o && table.is_none() => {
                    table = Some(o)
                }
                _ => return Err("sealed-entry-shape"),
            },
            Some(Node::Shl64(x, 3)) if scaled.is_none() => scaled = Some(x),
            _ => return Err("sealed-entry-shape"),
        }
    }
    let (Some(table_arg_offset), Some(scaled)) = (table, scaled) else {
        return Err("sealed-entry-shape");
    };
    let Some(Node::Sext(index) | Node::Zext(index)) = it.node(scaled) else {
        return Err("sealed-entry-shape");
    };
    match it.node(index) {
        Some(Node::Load(_)) => {}
        Some(Node::K32(o)) if classify(o) == KargKind::Scalar => {}
        _ => return Err("sealed-entry-index"),
    }
    Ok(TableShape {
        table_arg_offset,
        index,
    })
}
