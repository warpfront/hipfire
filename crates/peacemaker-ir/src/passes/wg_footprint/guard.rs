// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The guard-aware range prover of [`super`]: writes of the shape
//!
//! ```text
//! address = P + U + X * G(w, lane) + C(w, lane)        (mod 2^64)
//! ```
//!
//! where `X` is the sign/zero extension of one kernarg dword `n` (the row
//! stride of a row-major output), `G` the row form and `C` the column form.
//! Two tile ids `w != w'` of a launch cannot write the same byte when
//!
//! * `u = 2^tz(G)` and every write lies in the column range `[0, u*n)`: the
//!   dominating scalar branches of the write prove it (`col < n`), and the
//!   decode `address = (u*n) * R + C` with `C in [0, u*n)`, `R = G / u >= 0` is
//!   then unique, so equal addresses have equal `R` **and** equal `C`;
//! * the row windows `R(w)` are disjoint for tiles that differ on a row axis,
//!   or the column windows `C(w)` are disjoint for the rest: the footprint axes
//!   are the union of the row-separated and column-separated axes;
//! * no 32-bit/64-bit wrap makes two machine values meet: every integer the
//!   proof reads (the inner 32-bit forms, `R`, `u*n*(R+1)`) is bounded under
//!   the tile id bounds and the recorded preconditions.
//!
//! The row form may itself be a radix pair `H * M + L` (`M` another kernarg
//! dword, `0 <= L < M` by a dominating guard), e.g. `row = k_top * token +
//! rank` with `rank < k_top`: such a form is injective in its tile ids.
//!
//! Every relation the proof needs of a kernarg dword beyond what the guards
//! enforce is returned as an explicit [`WgRequirement`]; nothing is assumed.
use std::collections::BTreeSet;

use super::{modulus, tab_stats, Aff, Eval, Ix, Node, WgReqOp, WgRequirement, HW_BOUNDS};

/// A scalar branch fact that holds on every path to a store: `lhs < rhs`
/// (`strict`) or `lhs <= rhs`, over signed or unsigned 32-bit values.
pub(super) struct Rel {
    pub signed: bool,
    pub strict: bool,
    pub lhs: Aff,
    pub rhs: Aff,
}

/// Evaluate the comparison facts of a store under the current lane model.
pub(super) fn rels(ev: &mut Eval, facts: &[(Ix, bool)]) -> Vec<Rel> {
    let mut out = Vec::new();
    for &(atom, holds) in facts {
        let (signed, a, b) = match ev.it.node(atom) {
            Some(Node::Ltu { a, b }) => (false, a, b),
            Some(Node::Lts { a, b }) => (true, a, b),
            _ => continue,
        };
        let (Ok(x), Ok(y)) = (ev.eval(a), ev.eval(b)) else {
            continue;
        };
        if x.bits != 32 || y.bits != 32 {
            continue;
        }
        // `holds`: x < y; else y <= x.
        out.push(if holds {
            Rel {
                signed,
                strict: true,
                lhs: x,
                rhs: y,
            }
        } else {
            Rel {
                signed,
                strict: false,
                lhs: y,
                rhs: x,
            }
        });
    }
    out
}

/// One write access.
pub(super) struct Site<'a> {
    /// The 64-bit address form with the base pointer removed.
    pub a: &'a Aff,
    /// Access width in bytes.
    pub width: i128,
    pub rels: &'a [Rel],
}

pub(super) struct Proof {
    pub axes: Vec<u8>,
    pub bounds: [u64; 3],
    pub needs: Vec<WgRequirement>,
}

type R<T> = Result<T, String>;

const TWO31: i128 = 1 << 31;
const TWO32: i128 = 1 << 32;
const TWO64: i128 = 1 << 64;

/// The tile id bounds tried, largest first (x, then y, then z).
fn candidates() -> Vec<[i128; 3]> {
    let xs: [i128; 7] = [1 << 32, 1 << 28, 1 << 24, 1 << 20, 1 << 16, 1 << 12, 1 << 8];
    let ys: [i128; 4] = [1 << 16, 1 << 12, 1 << 8, 1 << 4];
    let zs: [i128; 4] = [1 << 16, 1 << 12, 1 << 8, 1 << 4];
    let mut out = Vec::new();
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                out.push([x, y, z]);
            }
        }
    }
    out
}

fn add_need(needs: &mut Vec<WgRequirement>, n: WgRequirement) {
    for e in needs.iter_mut() {
        if e.offset == n.offset && e.op == n.op {
            match n.op {
                WgReqOp::MaskEq => {
                    e.mask |= n.mask;
                    return;
                }
                WgReqOp::LeU => {
                    e.value = e.value.min(n.value);
                    return;
                }
                WgReqOp::GeU => {
                    e.value = e.value.max(n.value);
                    return;
                }
                WgReqOp::GridLe if e.mask == n.mask => {
                    e.value = e.value.max(n.value);
                    return;
                }
                WgReqOp::GridLe => {}
            }
        }
    }
    needs.push(n);
}

fn need_le(offset: u32, cap: i128) -> WgRequirement {
    WgRequirement {
        offset,
        mask: u32::MAX,
        value: cap.clamp(0, i128::from(u32::MAX)) as u32,
        op: WgReqOp::LeU,
    }
}

/// A 32-bit form without unknown terms whose machine value is its integer value
/// (nonnegative, below `2^31`) under the tile bounds `b`: a signed compare of
/// it is the integer compare.
fn small_nonneg(a: &Aff, b: &[i128; 3]) -> bool {
    if !a.usym.is_empty() || !a.prod.is_empty() || a.ext.is_some() || a.s.iter().any(|&v| v < 0) {
        return false;
    }
    let (_, wmax) = wg_range(&a.s, b);
    a.uc + a.rlo >= 0 && wmax + a.uc + a.rhi < TWO31
}

/// `a - b` when the two forms agree in everything but the uniform constant.
fn same_but_const(a: &Aff, b: &Aff) -> Option<i128> {
    if a.bits != b.bits
        || a.usym != b.usym
        || a.s != b.s
        || a.ext != b.ext
        || a.prod != b.prod
        || a.rlo != b.rlo
        || a.rhi != b.rhi
        || a.rtz != b.rtz
    {
        return None;
    }
    if !a.is_lane_free() && (a.tab.is_none() || b.tab.is_none() || a.tab != b.tab) {
        return None;
    }
    Some(a.uc - b.uc)
}

/// Sum of the positive/negative stride contributions at the tile bounds `b`.
fn wg_range(s: &[i128; 3], b: &[i128; 3]) -> (i128, i128) {
    let (mut lo, mut hi) = (0i128, 0i128);
    for i in 0..3 {
        let top = s[i].saturating_mul(b[i] - 1);
        if s[i] >= 0 {
            hi = hi.saturating_add(top);
        } else {
            lo = lo.saturating_add(top);
        }
    }
    (lo, hi)
}

/// Mixed-radix separation of a form with strides `s` under tile bounds `b`:
/// `(axes, sepmin)`: any two tiles that differ on an axis put their values at
/// least `sepmin` apart.
fn radix_sep(s: &[i128; 3], b: &[i128; 3]) -> Option<(Vec<u8>, i128)> {
    let axes: Vec<u8> = (0..3u8).filter(|&a| s[a as usize] != 0).collect();
    if axes.is_empty() {
        return Some((axes, i128::MAX));
    }
    let mut ordered: Vec<(i128, i128)> = axes
        .iter()
        .map(|&a| (s[a as usize].abs(), b[a as usize] - 1))
        .collect();
    ordered.sort();
    let mut acc = 0i128;
    let mut sepmin = i128::MAX;
    for &(st, e1) in &ordered {
        sepmin = sepmin.min(st.checked_sub(acc)?);
        acc = acc.checked_add(st.checked_mul(e1)?)?;
    }
    Some((axes, sepmin))
}

/// The window `[lo, hi]` of `sum mult_p * part_p` over the lanes (the parts'
/// uniform constants and lane parts; the tile strides are summed separately).
fn window(parts: &[(i128, &Aff)]) -> (i128, i128) {
    let mut lo = 0i128;
    let mut hi = 0i128;
    for (m, p) in parts {
        lo += m * p.uc;
        hi += m * p.uc;
    }
    let lanes: Vec<&(i128, &Aff)> = parts.iter().filter(|(_, p)| !p.is_lane_free()).collect();
    let tabs: Option<Vec<(i128, &Vec<i128>)>> = lanes
        .iter()
        .map(|(m, p)| p.tab.as_ref().map(|t| (*m, &**t)))
        .collect();
    match tabs {
        Some(tabs) if !tabs.is_empty() && tabs.iter().all(|(_, t)| t.len() == tabs[0].1.len()) => {
            let n = tabs[0].1.len();
            let mut tot: Vec<i128> = vec![0; n];
            for (m, t) in &tabs {
                for (acc, v) in tot.iter_mut().zip(t.iter()) {
                    *acc += m * v;
                }
            }
            let (l, h, _) = tab_stats(&tot);
            (lo + l, hi + h)
        }
        _ => {
            for (m, p) in lanes.into_iter() {
                let (a, b) = (m * p.rlo, m * p.rhi);
                lo += a.min(b);
                hi += a.max(b);
            }
            (lo, hi)
        }
    }
}

/// The 32-bit form of a kernarg dword atom `n`: `atom(32, n)`.
fn is_atom(a: &Aff, n: Ix) -> bool {
    a.usym.len() == 1
        && a.usym.get(&n) == Some(&1)
        && a.uc == 0
        && a.s == [0; 3]
        && a.is_lane_free()
        && a.ext.is_none()
        && a.prod.is_empty()
}

struct Shape {
    /// The row stride atom `X = ext(n)`.
    x: Ix,
    signed: bool,
    n_ix: Ix,
    n_off: u32,
    u: i128,
}

/// The shape and bounds-independent facts of the product; then the proof under
/// each candidate tile id bounds.
pub(super) fn prove(ev: &Eval, sites: &[Site<'_>]) -> R<Proof> {
    let shape = Shape::of(ev, sites)?;
    let mut last = String::new();
    for b in candidates() {
        match shape.try_bounds(ev, sites, b) {
            Ok(p) => return Ok(p),
            Err(e) => last = e,
        }
    }
    Err(last)
}

impl Shape {
    fn of(ev: &Eval, sites: &[Site<'_>]) -> R<Shape> {
        let first = sites[0].a;
        if first.prod.len() != 1 {
            return Err(format!(
                "the address has {} row-stride products (one is supported)",
                first.prod.len()
            ));
        }
        let x = *first.prod.keys().next().expect("one product");
        for s in sites {
            if s.a.prod.len() != 1 || !s.a.prod.contains_key(&x) {
                return Err("sites differ in their row stride".into());
            }
            if s.a.usym != first.usym {
                return Err("sites differ in their uniform (kernarg) offsets".into());
            }
            if s.a.s != first.s {
                return Err("sites have different tile coefficients".into());
            }
        }
        if !first.usym.is_empty() {
            return Err(
                "an unknown uniform offset enters the address next to the row product".into(),
            );
        }
        let (signed, n_ix) = match ev.it.node(x) {
            Some(Node::Sext(n)) => (true, n),
            Some(Node::Zext(n)) => (false, n),
            _ => return Err("the row stride is not an extended 32-bit kernarg dword".into()),
        };
        let Some(Node::K32(n_off)) = ev.it.node(n_ix) else {
            return Err("the row stride is not a kernarg dword".into());
        };
        let mut tz = 127u32;
        for s in sites {
            let g = &s.a.prod[&x];
            if !g.usym.is_empty() {
                return Err("the row form has an unknown uniform term".into());
            }
            tz = tz.min(g.total_tz());
        }
        let tz = tz.min(32);
        Ok(Shape {
            x,
            signed,
            n_ix,
            n_off,
            u: 1i128 << tz,
        })
    }

    fn try_bounds(&self, ev: &Eval, sites: &[Site<'_>], b: [i128; 3]) -> R<Proof> {
        let mut needs = Vec::new();
        let (row_axes, rmax) = self.row(ev, sites, &b, &mut needs)?;
        let col_axes = self.col(ev, sites, &b, &mut needs)?;
        // 64-bit exactness of `u * n * R + C`.
        let nmax = if self.signed { TWO31 - 1 } else { TWO32 - 1 };
        if self.u.saturating_mul(nmax).saturating_mul(rmax + 1) > TWO64 {
            let cap = TWO64 / (self.u * (rmax + 1));
            if cap < 1 {
                return Err("the row index is too large for the 64-bit address".into());
            }
            add_need(&mut needs, need_le(self.n_off, cap));
        }
        let axes: BTreeSet<u8> = row_axes.into_iter().chain(col_axes).collect();
        if axes.is_empty() {
            return Err("no tile axis separates the writes".into());
        }
        let mut bounds = HW_BOUNDS;
        for &a in &axes {
            bounds[usize::from(a)] = (b[usize::from(a)] as u64).min(HW_BOUNDS[usize::from(a)]);
        }
        Ok(Proof {
            axes: axes.into_iter().collect(),
            bounds,
            needs,
        })
    }

    /// The row windows `R = G / u`: the axes along which two tiles' windows are
    /// disjoint and the maximum row index.
    fn row(
        &self,
        ev: &Eval,
        sites: &[Site<'_>],
        b: &[i128; 3],
        needs: &mut Vec<WgRequirement>,
    ) -> R<(Vec<u8>, i128)> {
        let gs: Vec<&Aff> = sites.iter().map(|s| &*s.a.prod[&self.x]).collect();
        let first = gs[0];
        let ext0 = first.ext.as_deref();
        for g in &gs {
            let same = match (&g.ext, ext0) {
                (None, None) => true,
                (Some(e), Some(f)) => e.signed == f.signed && e.scale == f.scale,
                _ => false,
            };
            if !same || g.s != first.s {
                return Err("sites differ in the shape of the row form".into());
            }
        }
        match ext0 {
            None => {
                // R = (sum s_i w_i + lane + uc) / u.
                let parts: Vec<Vec<(i128, &Aff)>> = gs.iter().map(|g| vec![(1, *g)]).collect();
                let (s, lo, hi) = flat(&parts)?;
                let s: [i128; 3] = [s[0] / self.u, s[1] / self.u, s[2] / self.u];
                let (lo, hi) = (lo / self.u, hi / self.u);
                if lo < 0 || s.iter().any(|&v| v < 0) {
                    return Err("the row index can be negative".into());
                }
                let (axes, sep) = radix_sep(&s, b).ok_or("row strides overflow")?;
                if sep < hi - lo + 1 {
                    return Err(format!(
                        "row windows overlap (sepmin {sep} < width {})",
                        hi - lo + 1
                    ));
                }
                let (_, wmax) = wg_range(&s, b);
                Ok((axes, wmax + hi))
            }
            Some(e0) => {
                let scale = e0.scale;
                if scale <= 0 || scale % self.u != 0 {
                    return Err("the row scale is not a multiple of the row unit".into());
                }
                let k = scale / self.u;
                let inners: Vec<&Aff> = gs
                    .iter()
                    .map(|g| &g.ext.as_ref().expect("ext").inner)
                    .collect();
                if inners.iter().any(|i| !i.prod.is_empty()) {
                    // A radix pair inside the extension: R = k * (H * M + L).
                    if gs.iter().any(|g| !g.is_lane_free() || g.uc != 0) {
                        return Err("a row form with a radix pair and an extra window".into());
                    }
                    let (axes, imax) = self.nested(ev, sites, &inners, e0.signed, b, needs)?;
                    return Ok((axes.into_iter().collect(), k * imax));
                }
                // R = k * inner + (direct lane + uc) / u.
                let direct: Vec<Aff> = gs.iter().map(|g| direct_part(g)).collect();
                let parts: Vec<Vec<(i128, &Aff)>> = direct
                    .iter()
                    .zip(&inners)
                    .map(|(d, i)| vec![(scale, *i), (1, d)])
                    .collect();
                let (s, lo, hi) = flat(&parts)?;
                let s: [i128; 3] = [s[0] / self.u, s[1] / self.u, s[2] / self.u];
                let (lo, hi) = (lo / self.u, hi / self.u);
                self.inner_exact(&inners, e0.signed, b)?;
                if lo < 0 || s.iter().any(|&v| v < 0) {
                    return Err("the row index can be negative".into());
                }
                let (axes, sep) = radix_sep(&s, b).ok_or("row strides overflow")?;
                if sep < hi - lo + 1 {
                    return Err(format!(
                        "row windows overlap (sepmin {sep} < width {})",
                        hi - lo + 1
                    ));
                }
                let (_, wmax) = wg_range(&s, b);
                Ok((axes, wmax + hi))
            }
        }
    }

    /// The integer value of a 32-bit inner form equals its sign/zero
    /// extension: it is nonnegative and below `2^31` (`2^32`).
    fn inner_exact(&self, inners: &[&Aff], signed: bool, b: &[i128; 3]) -> R<()> {
        let lim = if signed { TWO31 } else { TWO32 };
        for i in inners {
            if !i.usym.is_empty() || !i.prod.is_empty() || i.ext.is_some() {
                return Err("an unknown term in the 32-bit index".into());
            }
            if i.s.iter().any(|&v| v < 0) || i.uc + i.rlo < 0 {
                return Err("the 32-bit index can be negative".into());
            }
            let (_, wmax) = wg_range(&i.s, b);
            if wmax + i.uc + i.rhi >= lim {
                return Err("the 32-bit index can wrap".into());
            }
        }
        Ok(())
    }

    /// A radix pair `inner = rest + M * H` (32-bit, `M` a kernarg dword atom,
    /// `0 <= rest < M` by a dominating guard): `(axes, max inner)`.
    fn nested(
        &self,
        ev: &Eval,
        sites: &[Site<'_>],
        inners: &[&Aff],
        signed: bool,
        b: &[i128; 3],
        needs: &mut Vec<WgRequirement>,
    ) -> R<(BTreeSet<u8>, i128)> {
        let first = inners[0];
        if first.prod.len() != 1 {
            return Err("a row index with several radix products".into());
        }
        let m = *first.prod.keys().next().expect("one product");
        let Some(Node::K32(m_off)) = ev.it.node(m) else {
            return Err("the radix of the row index is not a kernarg dword".into());
        };
        let mut rests: Vec<Aff> = Vec::new();
        let mut hs: Vec<&Aff> = Vec::new();
        for i in inners {
            if i.prod.len() != 1 || !i.prod.contains_key(&m) {
                return Err("sites differ in the radix of the row index".into());
            }
            let mut rest = (*i).clone();
            rest.prod.clear();
            rests.push(rest);
            hs.push(&i.prod[&m]);
        }
        // `rest` window, strides and the guard `rest < M`.
        let rparts: Vec<Vec<(i128, &Aff)>> = rests.iter().map(|r| vec![(1, r)]).collect();
        let (rs, rlo, rhi) = flat(&rparts)?;
        if rs.iter().any(|&v| v < 0) || rlo < 0 {
            return Err("the low digit of the row index can be negative".into());
        }
        for (r, site) in rests.iter().zip(sites) {
            if !r.usym.is_empty() {
                return Err("an unknown term in the low digit of the row index".into());
            }
            let ok = site.rels.iter().any(|rel| {
                rel.strict
                    && rel.signed
                    && is_atom(&rel.rhs, m)
                    && same_but_const(&rel.lhs, r).is_some_and(|d| d >= 0)
                    && small_nonneg(&rel.lhs, b)
            });
            if !ok {
                return Err("no dominating guard bounds the low digit of the row index".into());
            }
        }
        // `H`.
        let (h_axes, hmax) = if hs.iter().any(|h| !h.prod.is_empty()) {
            self.nested(ev, sites, &hs, signed, b, needs)?
        } else {
            self.inner_exact(&hs, true, b)?;
            let parts: Vec<Vec<(i128, &Aff)>> = hs.iter().map(|h| vec![(1, *h)]).collect();
            let (hs_s, hlo, hhi) = flat(&parts)?;
            let (axes, sep) = radix_sep(&hs_s, b).ok_or("strides overflow")?;
            if sep < hhi - hlo + 1 {
                return Err("high digit windows overlap".into());
            }
            let (_, w) = wg_range(&hs_s, b);
            (axes.into_iter().collect(), w + hhi)
        };
        let (l_axes, sep) = radix_sep(&rs, b).ok_or("strides overflow")?;
        if sep < rhi - rlo + 1 {
            return Err("low digit windows overlap".into());
        }
        // `M * (hmax + 1) <= 2^31`: no 32-bit wrap.
        let lim = if signed { TWO31 } else { TWO32 };
        let cap = lim / (hmax + 1);
        let imax = if cap >= lim - 1 {
            lim - 1
        } else {
            cap * (hmax + 1) - 1
        };
        if cap < lim - 1 {
            add_need(needs, need_le(m_off, cap));
        }
        Ok((
            h_axes
                .union(&l_axes.into_iter().collect())
                .copied()
                .collect(),
            imax,
        ))
    }

    /// The column windows `C`: guards put every write in `[0, u*n)`; the axes
    /// along which two tiles' windows are disjoint.
    fn col(
        &self,
        ev: &Eval,
        sites: &[Site<'_>],
        b: &[i128; 3],
        needs: &mut Vec<WgRequirement>,
    ) -> R<Vec<u8>> {
        let cs: Vec<Aff> = sites
            .iter()
            .map(|s| {
                let mut c = s.a.clone();
                c.prod.clear();
                c
            })
            .collect();
        let ext0 = cs[0].ext.as_deref();
        for c in &cs {
            let same = match (&c.ext, ext0) {
                (None, None) => true,
                (Some(e), Some(f)) => e.signed == f.signed && e.scale == f.scale,
                _ => false,
            };
            if !same {
                return Err("sites differ in the shape of the column form".into());
            }
            if !c.usym.is_empty() {
                return Err("an unknown uniform term in the column form".into());
            }
        }
        let (scale, signed) = ext0.map_or((1, true), |e| (e.scale, e.signed));
        if cs[0].ext.is_none() && cs[0].s != [0; 3] {
            return Err("a direct 64-bit tile term in the column form".into());
        }
        if scale <= 0 {
            return Err("a negative column scale".into());
        }
        let inners: Vec<Option<&Aff>> = cs
            .iter()
            .map(|c| c.ext.as_ref().map(|e| &e.inner))
            .collect();
        let zero = Aff::constant(32, 0);
        let inner_refs: Vec<&Aff> = inners.iter().map(|i| i.unwrap_or(&zero)).collect();
        if ext0.is_some() {
            self.inner_exact(&inner_refs, signed, b)?;
        }
        let direct: Vec<Aff> = cs.iter().map(direct_part).collect();
        // Strides and window of the column bytes.
        let parts: Vec<Vec<(i128, &Aff)>> = direct
            .iter()
            .zip(&inner_refs)
            .map(|(d, i)| vec![(scale, *i), (1, d)])
            .collect();
        let (s, lo, hi) = flat(&parts)?;
        if s.iter().any(|&v| v < 0) {
            return Err("a negative column stride".into());
        }
        let width = sites.iter().map(|s| s.width).max().expect("sites");
        let (axes, sep) = radix_sep(&s, b).ok_or("column strides overflow")?;
        if sep < hi - lo + width {
            return Err(format!(
                "column windows overlap (sepmin {sep} < width {})",
                hi - lo + width
            ));
        }
        // Per-site: every byte lies in [0, u*n) by a dominating `col < n`.
        let mut guarded = needs.clone();
        let by_guard =
            sites
                .iter()
                .zip(&inner_refs)
                .zip(&direct)
                .try_for_each(|((site, inner), d)| {
                    self.col_in_range(site, inner, d, scale, b, &mut guarded)
                });
        match by_guard {
            Ok(()) => *needs = guarded,
            Err(why) => {
                // No dominating guard bounds the column: the tile grid must.
                // One column axis whose stride covers the whole write window
                // puts every write below `u * n` when `stride * grid <= n`
                // (an explicit certified precondition of the record).
                let one: Vec<u8> = axes.clone();
                if one.len() != 1 || lo < 0 || s[usize::from(one[0])] % self.u != 0 {
                    return Err(why);
                }
                let st = s[usize::from(one[0])];
                if hi + width > st {
                    return Err(why);
                }
                add_need(
                    needs,
                    WgRequirement {
                        offset: self.n_off,
                        mask: u32::from(one[0]),
                        value: (st / self.u).clamp(1, i128::from(u32::MAX)) as u32,
                        op: WgReqOp::GridLe,
                    },
                );
            }
        }
        Ok(axes)
    }

    /// `C + width - 1 < u * n` and `C >= 0` for one site.
    fn col_in_range(
        &self,
        site: &Site<'_>,
        inner: &Aff,
        direct: &Aff,
        scale: i128,
        b: &[i128; 3],
        needs: &mut Vec<WgRequirement>,
    ) -> R<()> {
        let u = self.u;
        if scale % u != 0 {
            return Err("the column scale is not a multiple of the row unit".into());
        }
        let k = scale / u;
        if k != 1 {
            return Err("the column scale is a proper multiple of the row unit".into());
        }
        // The element form `E` and the in-element byte window `[dlo, dhi]`,
        // in two decompositions: the lane part inside the element (a guard on
        // the lane-dependent element) or in the byte window (a guard on the
        // tile-origin element).
        let uc = direct.uc;
        let (q, beta) = (uc.div_euclid(u), uc.rem_euclid(u));
        let lane_ok = direct.is_lane_free()
            || (direct.tab.is_some()
                && u32::from(direct.rtz) >= u.trailing_zeros()
                && direct.rlo >= 0);
        let mut candidates: Vec<(Aff, i128, i128)> = Vec::new();
        // Lane in the byte window.
        {
            let mut e = inner.clone();
            e.uc += q;
            candidates.push((e, beta + direct.rlo, beta + direct.rhi));
        }
        // Lane in the element.
        if lane_ok && !direct.is_lane_free() {
            let t: Vec<i128> = direct
                .tab
                .as_ref()
                .expect("table")
                .iter()
                .map(|v| v / u)
                .collect();
            let lane = Aff::from_tab(32, t);
            let mut e = super::add(inner, &lane)?;
            e.uc += q;
            candidates.push((e, beta, beta));
        }
        for (e, dlo, dhi) in &candidates {
            if dlo < &0 {
                continue;
            }
            // The element is nonnegative and below `2^31` by the bounds.
            let (_, wmax) = wg_range(&e.s, b);
            if e.s.iter().any(|&v| v < 0) || e.uc + e.rlo < 0 || wmax + e.uc + e.rhi >= TWO31 {
                continue;
            }
            for rel in site.rels {
                if !rel.strict || !is_atom(&rel.rhs, self.n_ix) {
                    continue;
                }
                if self.signed && !rel.signed {
                    continue;
                }
                let Some(d) = same_but_const(&rel.lhs, e) else {
                    continue;
                };
                if d < 0 || !small_nonneg(&rel.lhs, b) {
                    continue;
                }
                // `E <= lhs <= n - 1`; a multiple-of-`2^j` lhs and dword leave
                // a gap of `2^j` elements.
                let need_span = dhi + site.width; // bytes above `u * E`
                let mut j = 0u32;
                while u << j < need_span {
                    j += 1;
                    if j > 20 {
                        break;
                    }
                }
                if j > 0 {
                    if rel.lhs.total_tz() < j {
                        continue;
                    }
                    add_need(
                        needs,
                        WgRequirement {
                            offset: self.n_off,
                            mask: (1u32 << j) - 1,
                            value: 0,
                            op: WgReqOp::MaskEq,
                        },
                    );
                }
                return Ok(());
            }
        }
        Err("no dominating guard bounds the column index by the row stride".into())
    }
}

/// `a` without its extension and products: the uniform constant, the direct
/// tile coefficients and the lane part.
fn direct_part(a: &Aff) -> Aff {
    let mut d = a.clone();
    d.ext = None;
    d.prod.clear();
    d
}

/// Strides and the `[lo, hi]` window of `sum mult * part` per site, equal in
/// the strides across sites; the union of the windows.
fn flat(parts: &[Vec<(i128, &Aff)>]) -> R<([i128; 3], i128, i128)> {
    let mut stride: Option<[i128; 3]> = None;
    let (mut lo, mut hi) = (i128::MAX, i128::MIN);
    for site in parts {
        let mut s = [0i128; 3];
        for (m, p) in site {
            if !p.usym.is_empty() {
                return Err("an unknown uniform term in the index".into());
            }
            for i in 0..3 {
                s[i] += m * p.s[i];
            }
        }
        match stride {
            None => stride = Some(s),
            Some(t) if t == s => {}
            Some(_) => return Err("sites have different tile coefficients".into()),
        }
        let (l, h) = window(site);
        lo = lo.min(l);
        hi = hi.max(h);
    }
    Ok((stride.expect("sites"), lo, hi))
}

#[allow(dead_code)]
fn unused(_: i128) -> i128 {
    modulus(32)
}
