// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 Down (MQ4) software-pipelined K loop: the five K128 epochs of K=640 fully unrolled with a
//! register-resident operand ring. Epoch `e + DIST` is issued before epoch `e` is computed, so the
//! global-load latency of the next epoch(s) hides under the WMMA chain and fold of the current one.
//! Every wait is the minimal `vmcnt` the ledger derives from the first use of a loaded register
//! (no `wait_all` between epochs).
//!
//! Numerics are untouched: each epoch runs exactly `fold::compute`'s instruction sequence
//! (f16 -> f32 scale, eight `ds_swizzle` broadcasts, in-place `^ 0x88888888` unpack, eight magic-seeded
//! K16 int WMMA steps, `fold_pass`), in ascending epoch order from +0.
//!
//! Only reached when the kernarg `K = 640` (`ECNT == 5`); any other K takes the incumbent loop
//! (`kloop::general`). The ABI admits every `K % 128 == 0` (`validate_launch`), so the guard is a
//! runtime compare, not an assumption.
use super::*;
use peacemaker_author::{lds::Lowering, Transitions};

/// Epochs handled by the unrolled path (`K = 640`).
pub(super) const EPOCHS: u32 = 5;
/// Operand sets in the ring, and how many epochs ahead of the compute each load is issued.
pub(super) const NSETS: u32 = 2;
pub(super) const DIST: u32 = 1;

/// First VGPR of operand set 1 (set 0 is the incumbent's `SET0` set at `v92`).
const SET_BASE: u8 = 126;
/// 16 code dwords, 16 activation dwords, scale dword, `d`.
const SET_WIDTH: u8 = 34;

#[derive(Clone, Copy)]
struct Set {
    a: u8,
    x: u8,
    hs: u8,
    d: u8,
}

fn set(g: &Gen, i: u32) -> Set {
    if i == 0 {
        Set {
            a: g.a(0),
            x: g.x(),
            hs: g.hs(0),
            d: g.d(),
        }
    } else {
        let b = SET_BASE + SET_WIDTH * (i as u8 - 1);
        Set {
            a: b,
            x: b + 16,
            hs: b + 32,
            d: b + 33,
        }
    }
}

/// The unrolled path replaces the K loop of the exact-gfx1100 BF16 MQ4 Down entry only.
pub(super) fn applies(g: &Gen) -> bool {
    g.spec.arch == Arch::Gfx1100
        && g.spec.kind == Kind::Down
        && g.spec.codec == Codec::Mq4
        && !g.spec.acc
        && g.spec.negative == Negative::None
}

/// Registers of operand sets `1..NSETS` (set 0 is registered by the generic plan).
pub(super) fn plan_sets(p: &mut RegPlan, g: &Gen, kernel: &dyn Fn() -> Live) -> Result<(), String> {
    for i in 1..NSETS {
        let st = set(g, i);
        for q in 0..4u8 {
            p.v::<4>("pipe_a", st.a + 4 * q, kernel())?;
        }
        for q in 0..4u8 {
            p.v::<4>("pipe_x", st.x + 4 * q, kernel())?;
        }
        p.v::<1>("pipe_scale", st.hs, kernel())?;
        p.v::<1>("pipe_d", st.d, kernel())?;
    }
    Ok(())
}

/// Issue epoch `e`'s loads into `st`: activation offset, scale + codes clause, X + d clause.
fn issue(b: &mut Builder, e: u32, st: Set) -> Result<(), String> {
    sop(
        b,
        format!("s_mul_i32 s{XEO}, s{XSTR}, {e}"),
        &[XEO],
        &[XSTR],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XE}, s{XEO}, v{XOFF}"),
        &[v(XE)],
        &[s(XEO), v(XOFF)],
    )?;
    // A padding slot stays out of range at every epoch (no wrap of OOB + epoch).
    op(
        b,
        format!(
            "v_cndmask_b32_e64 v{XE}, {}, v{XE}, s{LIVE}",
            lit(GATHER_OOB)
        ),
        &[v(XE)],
        &[v(XE), s(LIVE)],
    )?;
    // QT53 group `[f16 sc, f16 zp][64 nibble bytes]` per epoch: scale at 68e, codes at 68e + 4.
    let so = 68 * e;
    let co = 68 * e + 4;
    b.clause(|b| {
        let off = if so == 0 {
            String::new()
        } else {
            format!(" offset:{so}")
        };
        mem(
            b,
            format!(
                "buffer_load_u16 v{}, v{HOFF}, s[{SRD_W}:{}], 0 offen{off}",
                st.hs,
                SRD_W + 3
            ),
            &[v(st.hs)],
            &[v(HOFF), sr(SRD_W, 4)],
            MemoryClass::VmemLoad,
        )?;
        for q in 0..4u8 {
            bload(b, 4, st.a + 4 * q, WOFF, SRD_W, co + 16 * u32::from(q))?;
        }
        Ok(())
    })?;
    b.clause(|b| {
        for q in 0..4u8 {
            bload(b, 4, st.x + 4 * q, XE, SRD_X, 8 + 16 * u32::from(q))?;
        }
        bload(b, 1, st.d, XE, SRD_X, 0)
    })?;
    Ok(())
}

/// `fold::compute` over operand set `st` (one tensor).
fn compute(b: &mut Builder, g: &Gen, st: Set) -> Result<(), String> {
    let a = g.spec.arch;
    op(
        b,
        format!("v_cvt_f32_f16_e64 v{HF}, v{}.l", st.hs),
        &[v(HF)],
        &[v(st.hs)],
    )?;
    for j in 0..8u8 {
        let text = format!(
            "ds_swizzle_b32 v{}, v{HF} offset:swizzle(BROADCAST,16,{})",
            SCF + j,
            swizzle_lane(u32::from(j))
        );
        b.ds_crosslane(
            Instruction::new(text, vec![v(SCF + j)], vec![v(HF)]).memory(MemoryClass::DsLoad),
        )?;
    }
    for i in 0..16u8 {
        let r = st.a + i;
        op(
            b,
            format!("v_xor_b32_e32 v{r}, {}, v{r}", lit(REBIAS)),
            &[v(r)],
            &[v(r)],
        )?;
    }
    for i in 0..8u8 {
        iu4_fold::wmma_step(
            b,
            a,
            V::<8>(CACC),
            V::<2>(st.a + 2 * i),
            V::<2>(st.x + 2 * i),
            i == 0,
            V::<8>(MAGIC8),
        )?;
    }
    iu4_fold::fold_pass(b, CACC, SUM, SCF, st.d, TPROD)
}

/// The unrolled, software-pipelined five-epoch K loop. Leaves an empty wait ledger.
pub(super) fn emit<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen) -> Result<(), String> {
    if BAR == BAR_START {
        wg.barrier(Lockstep)?;
    }
    for e in 0..DIST {
        if BAR == BAR_TOP {
            wg.barrier(Lockstep)?;
        }
        issue(wg.isa(), e, set(g, e % NSETS))?;
        if BAR == BAR_POST_ISSUE {
            wg.barrier(Lockstep)?;
        }
    }
    for e in 0..EPOCHS {
        if e + DIST < EPOCHS {
            if BAR == BAR_TOP {
                wg.barrier(Lockstep)?;
            }
            issue(wg.isa(), e + DIST, set(g, (e + DIST) % NSETS))?;
            if BAR == BAR_POST_ISSUE {
                wg.barrier(Lockstep)?;
            }
        }
        compute(wg.isa(), g, set(g, e % NSETS))?;
    }
    wg.isa().wait_all()
}

/// Barrier placement: before every epoch's issue, after every issue, once before the first issue.
const BAR_TOP: u8 = 1;
const BAR_POST_ISSUE: u8 = 2;
const BAR_START: u8 = 3;
const BAR: u8 = BAR_POST_ISSUE;

/// A workgroup barrier that carries no LDS transitions (this kernel owns no LDS region): a pure
/// `s_barrier` so the waves of a CTA reach the same point of the K loop together.
struct Lockstep;
impl Transitions for Lockstep {
    type After = ();
    fn lower(&self, _l: &mut Lowering) -> Result<(), String> {
        Ok(())
    }
    fn after(self) {}
}
