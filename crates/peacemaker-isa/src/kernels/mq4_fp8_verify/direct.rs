// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The direct template (`bt1w1`): one wave per workgroup (16 weight rows x 16
//! tokens), no LDS, no barriers. Per 128-K half the wave loads its 8 code
//! dwords (4 x `global_load_b64`) and its 16 activation fragments' 64 bytes
//! (4 x `buffer_load_b128`), dequantizes 8 fragments and runs one chain of 8
//! WMMAs.
//!
//! Pipeline (half index `x = 2g + h`; stage `h` = register set of half `h`):
//!  * codes of half `x + 3` are loaded as soon as half `x + 1`'s codes have been
//!    dequantized, into the stage just consumed: two halves of codes (2 KB per
//!    wave) are in flight in steady state (a stage's code registers alternate
//!    1:1 with the halves);
//!  * during chain `x` the next half's fragments are dequantized (17 VALU each)
//!    between the chain's dependent WMMAs into the other `BF` set;
//!  * the previous half's fold (`T = fma(sc, P, T)`) is interleaved with this
//!    half's chain (`P` is double buffered per half: `P0`, `P1`), so the fold of
//!    half `x` happens in chain `x + 1` and the epilogue folds the last half;
//!  * the activation registers of a half are reloaded with the next group's same
//!    half as soon as the chain has consumed each 16-byte piece, so they are in
//!    flight for about one group (two chains) before their next use.
//!
//! The first iteration folds the all-zero `P1` with a zero scale into `T = +0`
//! (`fma(0, 0, +0) = +0`), which is the contract's "T starts at +0".
//!
//! VGPR high water: 147 (see `Lay`).
use super::*;

/// Direct-template VGPR layout.
struct Lay {
    t: u8,
    p: [u8; 2],
    bf: [u8; 2],
    cs: [u8; 2],
    ax: [u8; 2],
    hdr: u8,
    sca: u8,
    scb: u8,
    dt: [u8; 2],
    end: u16,
}
impl Lay {
    fn new() -> Lay {
        let t = V0;
        let p = [t + 8, t + 16];
        let bf = [t + 24, t + 40];
        let cs = [t + 56, t + 64];
        let ax = [t + 72, t + 88];
        let hdr = t + 104;
        let (sca, scb) = (hdr + 2, hdr + 3);
        let dt = [hdr + 4, hdr + 12];
        Lay {
            t,
            p,
            bf,
            cs,
            ax,
            hdr,
            sca,
            scb,
            dt,
            end: u16::from(dt[1]) + 8,
        }
    }
}

fn plan(spec: &Spec, l: &Lay) -> Result<RegPlan, String> {
    let mut p = plan_common(spec, l.end)?;
    let w = || Live::Whole;
    p.v::<8>("t_acc", l.t, w())?;
    for h in 0..2 {
        p.v::<8>("p_acc", l.p[h], w())?;
        for f in 0..8 {
            p.v::<2>("b_frag", l.bf[h] + 2 * f, w())?;
        }
        for q in 0..4 {
            p.v::<2>("codes", l.cs[h] + 2 * q, w())?;
        }
        for q in 0..4 {
            p.v::<4>("x_frag", l.ax[h] + 4 * q, w())?;
        }
        p.v::<8>("dequant_temp", l.dt[h], w())?;
    }
    p.v::<2>("header", l.hdr, w())?;
    p.v::<1>("scale_a", l.sca, w())?;
    p.v::<1>("scale_b", l.scb, w())?;
    Ok(p)
}

/// One half-chain `h` of the group body. `warm` emits only the loads and the
/// half-1 chain's dequantization, for group `-1` (the prologue's warm-up: the
/// group-0 loads then leave exactly the in-flight set the body's back edge
/// leaves, which the checked loop requires): pointers and K offsets are those
/// of group `-1` (next group = group 0, group after next = group 1).
fn chain(b: &mut Builder, l: &Lay, h: usize, warm: bool) -> R {
    let other = 1 - h;
    let mut slots: Vec<Vec<Fill>> = vec![Vec::new(); 8];
    // The previous group's second half (P1, scale SCB), then this group's two scales and the next group's header.
    let (pn, pc, ph, gx) = if warm {
        (BASEC, PCN, BASEH, GX0)
    } else {
        (PCN, PC, PHN, GX1)
    };
    // Dequantize the other half's fragments (the next half in sequence) between the chain's WMMAs.
    for t in 0..8u8 {
        if !warm || h == 1 {
            slots[usize::from(t)].extend(dequant(
                l.cs[other] + t,
                l.bf[other] + 2 * t,
                l.dt[usize::from(t % 2)],
            ));
        }
        if t % 2 == 1 {
            let q = t / 2;
            // Codes of the half three ahead (X+3) into the stage just consumed.
            let (ptr, imm) = if h == 0 {
                (pn, 64 + 8 * u32::from(q))
            } else {
                (pc, 8 * u32::from(q))
            };
            slots[usize::from(t)].push(Fill::GLoad {
                dwords: 2,
                dst: l.cs[other] + 2 * q,
                ptr,
                imm,
            });
        }
    }
    // This chain's activation registers, reloaded for the next group's same half as consumed.
    for q in 0..4u8 {
        slots[usize::from(2 * q + 1)].push(Fill::XLoad {
            dwords: 4,
            dst: l.ax[h] + 4 * q,
            soff: gx,
            imm: 128 * h as u32 + 16 * u32::from(q),
        });
    }
    if h == 0 {
        if !warm {
            slots[0].extend(fold(l.t, l.p[1], l.scb));
            slots[1].push(cvt_scale(l.scb, l.hdr, 1));
            slots[1].push(cvt_scale(l.sca, l.hdr, 0));
        }
        slots[1].push(Fill::GLoad {
            dwords: 2,
            dst: l.hdr,
            ptr: ph,
            imm: 0,
        });
    } else if !warm {
        slots[1].extend(fold(l.t, l.p[0], l.sca));
    }
    for t in 0..8u8 {
        if !warm {
            wmma(b, l.p[h], l.ax[h] + 2 * t, l.bf[h] + 2 * t, t != 0)?;
        }
        for f in &slots[usize::from(t)] {
            emit_fill(b, f)?;
        }
    }
    Ok(())
}

fn body(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    group_scalars(b, true)?;
    mad64(b, PCN, s(NGO), "1", BASEC)?;
    mad64(b, PC, s(NGO2), "1", BASEC)?;
    mad64(b, PHN, s(NGO), "1", BASEH)?;
    chain(b, l, 0, false)?;
    chain(b, l, 1, false)?;
    sop(b, format!("s_add_co_i32 s{G}, s{G}, 1"), &[G], &[G])?;
    sop(b, format!("s_cmp_lt_u32 s{G}, s{GPR}"), &[], &[G, GPR])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "loop")), &[], &[])
}

fn prologue(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    common_prologue(b, spec)?;
    // X voffset of the lane's token row: (bs + ml) * K + 64 * kg.
    op(
        b,
        format!("v_add_nc_u32_e32 v{T3}, s{BS}, v{T0}"),
        &[v(T3)],
        &[s(BS), v(T0)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{XOFF}, v{T3}, s{}", ka_k(spec)),
        &[v(XOFF)],
        &[v(T3), s(ka_k(spec))],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T3}, 6, v{T1}"),
        &[v(T3)],
        &[v(T1)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XOFF}, v{XOFF}, v{T3}"),
        &[v(XOFF)],
        &[v(XOFF), v(T3)],
    )?;
    for r in 0..8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", l.t + r),
            &[v(l.t + r)],
            &[],
        )?;
    }
    for r in 0..8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", l.p[1] + r),
            &[v(l.p[1] + r)],
            &[],
        )?;
    }
    op(b, format!("v_mov_b32_e32 v{}, 0", l.scb), &[v(l.scb)], &[])?;
    sop(b, format!("s_mov_b32 s{G}, 0"), &[G], &[])?;
    group_scalars(b, false)?;
    mad64(b, PCN, s(NGO), "1", BASEC)?;
    // Group 0's half 0 codes, then the warm-up of group -1: group 0's half-1
    // codes, activations and header, half 0 dequantized, group 1's half-0 codes.
    for q in 0..4u8 {
        gload(b, 2, l.cs[0] + 2 * q, BASEC, 8 * u32::from(q))?;
    }
    chain(b, l, 0, true)?;
    chain(b, l, 1, true)?;
    Ok(())
}

pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
    let l = Lay::new();
    let mut b = Builder::new(kernel_spec(&spec), plan(&spec, &l)?);
    b.enable_delay_alu();
    prologue(&mut b, &spec, &l)?;
    b.loop_(&lbl(&spec, "loop"), |b| body(b, &spec, &l))?;
    let epi = Epi {
        t: l.t,
        p: l.p[1],
        sc: l.scb,
        s_pool: (0..8).map(|r| l.p[0] + r).collect(),
        y_pool: (0..8).map(|r| l.bf[0] + r).collect(),
    };
    epilogue(&mut b, &spec, &epi)?;
    b.finish()
}
