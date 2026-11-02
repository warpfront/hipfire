// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The LDS-staged template (`bt{2,4,8}w{4,8}`), modelled on `mq4_verify`'s slab
//! pipeline: a wave owns 16 weight rows x `16*BT` tokens (BT accumulator
//! tiles, so one dequantized fragment feeds BT WMMAs), a workgroup is `W` waves.
//!
//! A 256-K group is 8 slabs; slab `s`: half `h = s / 4`, `u = s % 4`, fragments
//! `2u` and `2u + 1`. Per slab (2*BT WMMAs, fragment-major: consecutive WMMAs
//! are independent token tiles):
//!  * LDS: two slots of `16*BT` token rows x 48 bytes (32 data bytes: the kg0
//!    piece = K `[128h + 16u, +16)`, the kg1 piece = K `[128h + 64 + 16u, +16)`,
//!    then 16 pad). The wave reads its A operands with one `ds_load_b128` per
//!    token tile (row `16b + ml`, +16 bytes for kg), the fragments `2u` / `2u+1`
//!    being bytes 0..7 / 8..15. X is staged through registers loaded two slabs
//!    ahead (per-thread width `16*BT / W` bytes: b32/b64/b128/2xb128); one
//!    barrier per slab publishes the next slot (placed between the first and
//!    second fragment's WMMAs, so the A loads of the next slab trail the
//!    second fragment's reads of the same tile register by BT WMMAs).
//!  * Codes (b64: the slab's two dwords) are loaded three slabs ahead into one of
//!    four buffers; the next slab's two fragments are dequantized (17 VALU
//!    each) between this slab's WMMAs into the other of two B sets.
//!  * The `P` chain of a tile restarts from inline zero at each half's first
//!    fragment; the fold of tile `b` is emitted immediately before the next
//!    half's first WMMA of tile `b` (BT-1 WMMAs after the tile's last WMMA, so the
//!    accumulator latency is hidden) — for the group's last half this lands in
//!    the next group's slab 0, with a zero `P`, zero scale and zero `T`
//!    prologue state making the first such fold a no-op (`fma(0, 0, +0)`);
//!    the epilogue folds the last half.
//!  * The scales of group `g` are converted in slab 0 after the previous half's
//!    folds; the next group's header is loaded in slab 1.
//!
//! VGPR high water: `V0 + 8*BT (T) + 8*BT (P) + 4*BT (A) + 8 (B) + 8 (codes) +
//! 4 (header, scales) + 16 (temps) + 2 * X-stage`: 235 for bt8w4.
use super::*;

pub(super) const PITCH: u32 = 48; // 32 data bytes + 16 pad per token row
const SLABS: u8 = 8; // slabs per 256-K group

impl Spec {
    /// X bytes per thread per slab.
    fn xw(&self) -> u32 {
        16 * u32::from(self.bt) / u32::from(self.waves)
    }
    /// X stage registers per stage.
    fn xdw(&self) -> u8 {
        (self.xw() / 4) as u8
    }
    /// The slot-relative WMMA after which the slab barrier sits.
    fn bar_at(&self) -> usize {
        usize::from(self.bt) - 1
    }
}

struct Lay {
    t: u8,
    p: u8,
    a: u8,
    b: u8,
    code: u8,
    hdr: u8,
    sca: u8,
    scb: u8,
    dt: [u8; 2],
    xr: [u8; 2],
    xdw: u8,
    end: u16,
}
impl Lay {
    fn new(spec: &Spec) -> Lay {
        let bt = spec.bt;
        let t = V0;
        let p = t + 8 * bt;
        let a = p + 8 * bt;
        let b = a + 4 * bt;
        let code = b + 8;
        let hdr = code + 8;
        let (sca, scb) = (hdr + 2, hdr + 3);
        let dt = [hdr + 4, hdr + 12];
        let xdw = spec.xdw();
        let xr = [dt[1] + 8, dt[1] + 8 + xdw];
        Lay {
            t,
            p,
            a,
            b,
            code,
            hdr,
            sca,
            scb,
            dt,
            xr,
            xdw,
            end: u16::from(xr[1]) + u16::from(xdw),
        }
    }
    /// A buffer of token tile `bi` (fragment `f` = the pair at `+2f`).
    fn a(&self, bi: u8) -> u8 {
        self.a + 4 * bi
    }
    /// B fragment `f` of set `set`.
    fn bf(&self, set: u8, f: u8) -> u8 {
        self.b + 4 * (set % 2) + 2 * f
    }
    /// Code buffer (2 dwords) of slab `c`.
    fn code(&self, c: u8) -> u8 {
        self.code + 2 * (c % 4)
    }
}

fn plan(spec: &Spec, l: &Lay) -> Result<RegPlan, String> {
    let mut p = plan_common(spec, l.end)?;
    let w = || Live::Whole;
    for bi in 0..spec.bt {
        p.v::<8>("t_acc", l.t + 8 * bi, w())?;
        p.v::<8>("p_acc", l.p + 8 * bi, w())?;
        p.v::<4>("a_frag", l.a(bi), w())?;
    }
    for set in 0..2 {
        for f in 0..2 {
            p.v::<2>("b_frag", l.bf(set, f), w())?;
        }
    }
    for c in 0..4 {
        p.v::<2>("codes", l.code(c), w())?;
    }
    p.v::<2>("header", l.hdr, w())?;
    p.v::<1>("scale_a", l.sca, w())?;
    p.v::<1>("scale_b", l.scb, w())?;
    for d in l.dt {
        p.v::<8>("dequant_temp", d, w())?;
    }
    for x in l.xr {
        p.add_range("x_stage", Kind::V, x, l.xdw, w())?;
    }
    Ok(p)
}

/// K byte offset of slab `c` within its 256-K group: `128h + 16u`.
fn koff(c: u8) -> u32 {
    128 * u32::from(c / 4) + 16 * u32::from(c % 4)
}
/// Code byte offset (relative to the lane's `+8 + 32kg` pointer) of slab `c`: `64h + 8u`.
fn coff(c: u8) -> u32 {
    64 * u32::from(c / 4) + 8 * u32::from(c % 4)
}

/// X loads of absolute slab offset `ahead` (relative to this group's slab 0 =
/// group + `ahead / 8`, in-group slab `ahead % 8`) into stage register `dst`.
fn xloads(spec: &Spec, dst: u8, ahead: u8) -> Vec<Fill> {
    let (soff, c) = if ahead < SLABS {
        (GX0, ahead)
    } else {
        (GX1, ahead - SLABS)
    };
    let k = koff(c);
    if spec.xw() == 32 {
        vec![
            Fill::XLoad {
                dwords: 4,
                dst,
                soff,
                imm: k,
            },
            Fill::XLoad {
                dwords: 4,
                dst: dst + 4,
                soff,
                imm: k + 64,
            },
        ]
    } else {
        vec![Fill::XLoad {
            dwords: spec.xdw(),
            dst,
            soff,
            imm: k,
        }]
    }
}
/// LDS stores of the stage `src` into slot `slot`.
fn xstores(spec: &Spec, src: u8, slot: usize) -> Vec<Fill> {
    let base = slot as u32 * spec.slot_bytes();
    if spec.xw() == 32 {
        vec![
            Fill::DsStore {
                slot,
                dwords: 4,
                src,
                imm: base,
            },
            Fill::DsStore {
                slot,
                dwords: 4,
                src: src + 4,
                imm: base + 16,
            },
        ]
    } else {
        vec![Fill::DsStore {
            slot,
            dwords: spec.xdw(),
            src,
            imm: base,
        }]
    }
}
/// Dequantization of slab `c`'s two fragments (codes buffer of `c`) into B set `c`.
fn dequant_slab(l: &Lay, c: u8) -> (Vec<Fill>, Vec<Fill>) {
    let w = l.code(c);
    (
        dequant(w, l.bf(c, 0), l.dt[0]),
        dequant(w + 1, l.bf(c, 1), l.dt[1]),
    )
}

/// One slab `c` (0..7) of the group loop body. On entry LDS slot `c % 2` holds
/// X(s) (published), the stage `(c + 1) % 2` holds X(s + 1) and `c % 2` is
/// loading X(s + 2); A(s) is in the A registers, B set `c % 2` holds slab `c`'s
/// fragments, the code buffers `c + 1`, `c + 2` are loaded or in flight.
fn slab(b: &mut Builder, spec: &Spec, l: &Lay, c: u8) -> R {
    let (bt, bar) = (spec.bt, spec.bar_at());
    let wpc = 2 * usize::from(bt);
    let slot = usize::from(c % 2);
    let other = 1 - slot;
    let stage = l.xr[usize::from((c + 1) % 2)];
    let mut slots: Vec<Vec<Fill>> = vec![Vec::new(); wpc];
    let mut pre: Vec<Vec<Fill>> = vec![Vec::new(); wpc];
    // Codes three slabs ahead.
    let cc = c + 3;
    let (ptr, cs) = if cc < SLABS {
        (PC, cc)
    } else {
        (PCN, cc - SLABS)
    };
    slots[0].push(Fill::GLoad {
        dwords: 2,
        dst: l.code(cs),
        ptr,
        imm: coff(cs),
    });
    if c == 1 {
        slots[0].push(Fill::GLoad {
            dwords: 2,
            dst: l.hdr,
            ptr: PHN,
            imm: 0,
        });
    }
    // Next slab's dequant (frag 0 under the first fragment's WMMAs, frag 1 under the rest).
    let (d0, d1) = dequant_slab(l, c + 1);
    spread(&mut slots, d0, 0, usize::from(bt));
    spread(&mut slots, d1, usize::from(bt) / 2, wpc);
    // Stage X(s + 1) into the other slot before the barrier.
    for (i, f) in xstores(spec, stage, other).into_iter().enumerate() {
        slots[i.min(bar)].push(f);
    }
    // Folds of the previous half and the half's scales.
    if c % 4 == 0 {
        let sc = if c == 0 { l.scb } else { l.sca };
        for bi in 0..bt {
            pre[usize::from(bi)] = fold(l.t + 8 * bi, l.p + 8 * bi, sc);
        }
    }
    if c == 0 {
        slots[bar].push(cvt_scale(l.scb, l.hdr, 1));
        slots[bar].push(cvt_scale(l.sca, l.hdr, 0));
    }
    for k in 0..wpc {
        let (f, bi) = ((k / usize::from(bt)) as u8, (k % usize::from(bt)) as u8);
        for x in &pre[k] {
            emit_fill(b, x)?;
        }
        wmma(
            b,
            l.p + 8 * bi,
            l.a(bi) + 2 * f,
            l.bf(c, f),
            !(c % 4 == 0 && f == 0),
        )?;
        for x in &slots[k] {
            emit_fill(b, x)?;
        }
        if k == bar {
            // This wave's reads of `slot` have landed; publish the other slot, retire this one.
            b.wait(Counter::Ds, 0)?;
            b.barrier(&[Transition::Ready(other), Transition::Retire(slot)])?;
            for x in xloads(spec, stage, c + 3) {
                emit_fill(b, &x)?;
            }
        }
        if k >= usize::from(bt) {
            // A(s + 1) of tile `bi` from the published slot, after the last read of its register.
            let bi = (k - usize::from(bt)) as u8;
            emit_fill(
                b,
                &Fill::DsLoad {
                    slot: other,
                    dst: l.a(bi),
                    imm: other as u32 * spec.slot_bytes() + 16 * PITCH * u32::from(bi),
                },
            )?;
        }
    }
    Ok(())
}

fn body(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    group_scalars(b, false)?;
    mad64(b, PC, s(GO), "1", BASEC)?;
    mad64(b, PCN, s(NGO), "1", BASEC)?;
    mad64(b, PHN, s(NGO), "1", BASEH)?;
    for c in 0..SLABS {
        slab(b, spec, l, c)?;
    }
    sop(b, format!("s_add_co_i32 s{G}, s{G}, 1"), &[G], &[G])?;
    sop(b, format!("s_cmp_lt_u32 s{G}, s{GPR}"), &[], &[G, GPR])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "loop")), &[], &[])
}

fn prologue(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    common_prologue(b, spec)?;
    let bt = spec.bt;
    // A rows in LDS: ml * 48 + 16 * kg.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T2}, 4, v{T1}"),
        &[v(T2)],
        &[v(T1)],
    )?;
    op(
        b,
        format!("v_mad_u32_u24 v{LDRD}, v{T0}, {}, v{T2}", lit(PITCH)),
        &[v(LDRD)],
        &[v(T0), v(T2)],
    )?;
    // The thread's X piece: row R = tid * w / 32, byte O = tid * w % 32 within the row.
    let (xw, sk) = (spec.xw(), ka_k(spec));
    let lg = xw.trailing_zeros();
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T3}, {}, v{TID}", 5 - lg),
        &[v(T3)],
        &[v(TID)],
    )?;
    if xw < 32 {
        op(
            b,
            format!("v_and_b32_e32 v{T4}, {}, v{TID}", lit(32 / xw - 1)),
            &[v(T4)],
            &[v(TID)],
        )?;
        op(
            b,
            format!("v_lshlrev_b32_e32 v{T4}, {lg}, v{T4}"),
            &[v(T4)],
            &[v(T4)],
        )?;
    } else {
        op(b, format!("v_mov_b32_e32 v{T4}, 0"), &[v(T4)], &[])?;
    }
    op(
        b,
        format!("v_mad_u32_u24 v{LDST}, v{T3}, {}, v{T4}", lit(PITCH)),
        &[v(LDST)],
        &[v(T3), v(T4)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XOFF}, s{BS}, v{T3}"),
        &[v(XOFF)],
        &[s(BS), v(T3)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{XOFF}, v{XOFF}, s{sk}"),
        &[v(XOFF)],
        &[v(XOFF), s(sk)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{T5}, 16, v{T4}"),
        &[v(T5)],
        &[v(T4)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T5}, 2, v{T5}"),
        &[v(T5)],
        &[v(T5)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XOFF}, v{XOFF}, v{T5}"),
        &[v(XOFF)],
        &[v(XOFF), v(T5)],
    )?;
    if xw < 16 {
        op(
            b,
            format!("v_and_b32_e32 v{T5}, 15, v{T4}"),
            &[v(T5)],
            &[v(T4)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{XOFF}, v{XOFF}, v{T5}"),
            &[v(XOFF)],
            &[v(XOFF), v(T5)],
        )?;
    }
    for r in 0..8 * bt {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", l.t + r),
            &[v(l.t + r)],
            &[],
        )?;
    }
    for r in 0..8 * bt {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", l.p + r),
            &[v(l.p + r)],
            &[],
        )?;
    }
    op(b, format!("v_mov_b32_e32 v{}, 0", l.scb), &[v(l.scb)], &[])?;
    sop(b, format!("s_mov_b32 s{G}, 0"), &[G], &[])?;
    group_scalars(b, false)?;
    mad64(b, PC, s(GO), "1", BASEC)?;
    mad64(b, PCN, s(NGO), "1", BASEC)?;
    mad64(b, PHN, s(NGO), "1", BASEH)?;
    // Group 0: header, slab 0's codes and X; stage X(0) in slot 0.
    gload(b, 2, l.hdr, BASEH, 0)?;
    gload(b, 2, l.code(0), PC, coff(0))?;
    for f in xloads(spec, l.xr[0], 0) {
        emit_fill(b, &f)?;
    }
    for f in xstores(spec, l.xr[0], 0) {
        emit_fill(b, &f)?;
    }
    b.barrier(&[Transition::Ready(0)])?;
    // Slab 1 and 2's codes and X (the loop body's tail leaves exactly these in
    // flight), slab 0's dequant and its A operands from the published slot.
    gload(b, 2, l.code(1), PC, coff(1))?;
    for f in xloads(spec, l.xr[1], 1) {
        emit_fill(b, &f)?;
    }
    gload(b, 2, l.code(2), PC, coff(2))?;
    for f in xloads(spec, l.xr[0], 2) {
        emit_fill(b, &f)?;
    }
    let (d0, d1) = dequant_slab(l, 0);
    for f in d0.iter().chain(&d1) {
        emit_fill(b, f)?;
    }
    for bi in 0..bt {
        emit_fill(
            b,
            &Fill::DsLoad {
                slot: 0,
                dst: l.a(bi),
                imm: 16 * PITCH * u32::from(bi),
            },
        )?;
    }
    Ok(())
}

pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
    let l = Lay::new(&spec);
    let mut b = Builder::new(kernel_spec(&spec), plan(&spec, &l)?);
    b.enable_delay_alu();
    for (id, (name, base)) in [("x_slot0", 0u32), ("x_slot1", spec.slot_bytes())]
        .into_iter()
        .enumerate()
    {
        if b.lds_slot(name, base, spec.slot_bytes())? != id {
            return Err("LDS slot order".into());
        }
    }
    prologue(&mut b, &spec, &l)?;
    b.loop_(&lbl(&spec, "loop"), |b| body(b, &spec, &l))?;
    // Dead registers after the loop: the P registers themselves (S values), and the
    // A / B / code / temp / X-stage registers (Y_old).
    let n = 8 * usize::from(spec.bt);
    let mut pool: Vec<u8> = (0..4 * spec.bt).map(|r| l.a + r).collect();
    pool.extend(l.b..l.b + 8);
    pool.extend(l.code..l.code + 8);
    pool.extend(l.dt[0]..l.dt[0] + 16);
    for x in l.xr {
        pool.extend(x..x + l.xdw);
    }
    let epi = Epi {
        t: l.t,
        p: l.p,
        sc: l.scb,
        s_pool: (0..n as u8).map(|r| l.p + r).collect(),
        y_pool: pool.into_iter().take(n).collect(),
    };
    if epi.y_pool.len() < n {
        return Err("epilogue Y pool too small".into());
    }
    epilogue(&mut b, &spec, &epi)?;
    b.finish()
}
