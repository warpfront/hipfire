// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ring depth 3 (`Tune { ring: RingDepth::Three }`): the A/W payload rings are
//! three LDS slabs deep and publish two slabs ahead. The metadata rings
//! (`DS`, `SZ`) stay two deep.
//!
//! Schedule. A trip is still a 256-K group (two K128 blocks, plane `h` = 0, 1,
//! four 64-K slabs). Two prefetch register sets hold the next block's slabs:
//! `P0` its slab 0, `P1` its slab 1. At block `b`'s entry slab `2b` is the
//! ring's current buffer, slab `2b + 1` is published ahead, and `P0`/`P1`
//! hold the slabs of block `b + 1` (fetched, possibly still in flight):
//!
//! ```text
//! phase A   fragments(cur) ; WMMA groups ; publish P0 (b+1 slab 0) into next
//!           fetch_next_meta ; wait ; barrier rotate(A, W) ; fetch b+2 slab 0 -> P0
//! phase B   fragments(cur) ; d_loads ; WMMA groups ; publish P1 (b+1 slab 1)
//!           and the metadata ; wait ; signal rotate(A, W, DS, SZ)
//!           fetch b+2 slab 1 -> P1 ; fold ; wait_arrived
//! ```
//!
//! Block `b + 2` is the same plane one group on, so the activation
//! descriptors step at the head of their block (`srd_a[0]` in the even block,
//! `srd_a[1]` in the odd one) and the trip end advances only the group
//! offset. The last group is peeled: its even block (`T0`) publishes the odd
//! block's slabs and metadata and fetches nothing, its odd block (`T1`)
//! publishes and fetches nothing and moves its payload rings one slab on with
//! a store-free barrier ([`peacemaker_author::advance`]).
//!
//! LDS addresses stay immediates: with `n = K/256 - 1` full groups, `r = n mod
//! 3` and `q = n / 3`, slab `s` lives in slot `(s - r) mod 3`, so after `r`
//! groups (four slabs each) the next group starts on slot 0, and so does the
//! peeled group after the loop's `q` trips of three groups. The code is three
//! arms (rings start at slot `(3 - r) mod 3`, the prologue's ring part, `r`
//! full groups), the loop body (three groups), and the peeled group.
use super::{
    fold,
    kloop::{self, d_loads, fragments, scale_loads, scale_tail, trips_cmp, wmma_group},
    publish, Gen, Lds3, MetaD, MetaS, PayA, PayW, Staged, WaitPlace, Wg, EPI, K_BEGIN, K_LOOP,
    K_LOOP_END,
};
use crate::insn::Instruction;
use crate::kernels::common::{op, s};
use peacemaker_author::{advance, rotate, Free, Published, Ring, Scc, WgUniform};

/// `Spec::validate` refuses the combination; the match arms of the wait placements name it.
const WINDOW_RING3: &str = "wait placement window is generated at ring depth 2 only";

/// What a block does besides its own slabs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind3 {
    /// Publishes block `b + 1`'s slabs and metadata and fetches `b + 2`'s.
    Full,
    /// The peeled group's even block: publishes, fetches nothing.
    T0,
    /// The peeled group's odd block: publishes and fetches nothing.
    T1,
}

type StagedRings3 = (
    Staged<PayA, Published, 3>,
    Staged<PayW, Published, 3>,
    Staged<MetaD, Published>,
    Staged<MetaS, Published>,
);

/// Publish prefetch set 1 (the next block's slab 1) into the payload rings'
/// free next buffer and the next block's token/row metadata into the
/// metadata rings'.
fn publish_next(
    wg: &mut Wg,
    g: &Gen,
    ra: Ring<PayA, Published, Free, 3>,
    rw: Ring<PayW, Published, Free, 3>,
    rd: Ring<MetaD, Published, Free>,
    rs: Ring<MetaS, Published, Free>,
) -> Result<StagedRings3, String> {
    let (sa, sw) = publish::publish_slab_set(wg, g, 1, ra, rw)?;
    let (sd, ss) = publish::publish_meta(wg, g, rd, rs)?;
    Ok((sa, sw, sd, ss))
}

/// One dense K128 block of the depth-3 K-loop on plane `h`. `step` advances
/// that activation descriptor right after the block's head (the odd/even
/// block steps its own plane's descriptor: the fetch of block `b + 2` reads it
/// one group on).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block3(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    kind: Kind3,
    end: &str,
    entry_label: bool,
    step: Option<usize>,
    (ra, rw, rd, rs): Lds3,
) -> Result<Lds3, String> {
    kloop::block_begin(
        wg,
        g,
        tag,
        h,
        end,
        entry_label,
        ra.cur_index(),
        rd.cur_index(),
    )?;
    let (_, x0, x1) = kloop::sub_labels(tag);
    if let Some(which) = step {
        kloop::step_descriptor(wg.isa(), g, which)?;
    }
    let place = g.spec.tune.wait;
    // Phase A: slab 0 of the block from the current buffer; the next block's
    // slab 0 (prefetch set 0) is published into the free next buffer.
    fragments(wg, g, &ra, &rw)?;
    let (ra, rw) = if kind == Kind3::T1 {
        for sb in 0..2 {
            wmma_group(wg.isa(), g, sb, sb == 0, false)?;
        }
        // Both of this block's slabs were published by the even block; only
        // the current buffer moves on.
        wg.barrier((advance(ra), advance(rw)))?
    } else {
        let ((ra, pa), (rw, pw)) = match place {
            WaitPlace::AfterMma => {
                for sb in 0..2 {
                    wmma_group(wg.isa(), g, sb, sb == 0, false)?;
                }
                publish::publish_slab_set(wg, g, 0, ra, rw)?
            }
            WaitPlace::MidMma => {
                wmma_group(wg.isa(), g, 0, true, false)?;
                let staged = publish::publish_slab_set(wg, g, 0, ra, rw)?;
                wmma_group(wg.isa(), g, 1, false, false)?;
                staged
            }
            WaitPlace::BeforeMma => {
                let staged = publish::publish_slab_set(wg, g, 0, ra, rw)?;
                for sb in 0..2 {
                    wmma_group(wg.isa(), g, sb, sb == 0, false)?;
                }
                staged
            }
            WaitPlace::Window => return Err(WINDOW_RING3.into()),
        };
        publish::fetch_next_meta(wg.isa(), g, h)?;
        let (da, dw) = wg.wait_all((pa, pw))?;
        wg.barrier((rotate(ra, da), rotate(rw, dw)))?
    };
    // Set 0 was published (and drained at the barrier): it takes block b+2's
    // slab 0.
    if kind == Kind3::Full {
        publish::fetch_ahead(wg.isa(), g, h, 0)?;
    }

    // Phase B: the second slab from the (new) current buffer, the fold.
    fragments(wg, g, &ra, &rw)?;
    d_loads(wg, g, &rd)?;
    if kind == Kind3::T1 {
        wmma_group(wg.isa(), g, 0, false, false)?;
        scale_loads(wg, g, &x0, &rs)?;
        wmma_group(wg.isa(), g, 1, false, true)?;
        scale_tail(wg, g, &x1, &rs)?;
        fold::scale(wg.isa(), g)?;
        return Ok((ra, rw, rd, rs));
    }
    let ((ra, pa), (rw, pw), (rd, pd), (rs, ps)) = match place {
        WaitPlace::AfterMma => {
            wmma_group(wg.isa(), g, 0, false, false)?;
            scale_loads(wg, g, &x0, &rs)?;
            wmma_group(wg.isa(), g, 1, false, true)?;
            scale_tail(wg, g, &x1, &rs)?;
            publish_next(wg, g, ra, rw, rd, rs)?
        }
        WaitPlace::MidMma => {
            wmma_group(wg.isa(), g, 0, false, false)?;
            scale_loads(wg, g, &x0, &rs)?;
            let (sa, sw, sd, ss) = publish_next(wg, g, ra, rw, rd, rs)?;
            wmma_group(wg.isa(), g, 1, false, true)?;
            scale_tail(wg, g, &x1, &ss.0)?;
            (sa, sw, sd, ss)
        }
        WaitPlace::BeforeMma => {
            let (sa, sw, sd, ss) = publish_next(wg, g, ra, rw, rd, rs)?;
            wmma_group(wg.isa(), g, 0, false, false)?;
            scale_loads(wg, g, &x0, &ss.0)?;
            wmma_group(wg.isa(), g, 1, false, true)?;
            scale_tail(wg, g, &x1, &ss.0)?;
            (sa, sw, sd, ss)
        }
        WaitPlace::Window => return Err(WINDOW_RING3.into()),
    };
    let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
    let b2 = wg.signal((
        rotate(ra, da),
        rotate(rw, dw),
        rotate(rd, dd),
        rotate(rs, ds),
    ))?;
    // The signal drained set 1's stores: it takes block b+2's slab 1 while
    // this block folds.
    if kind == Kind3::Full {
        publish::fetch_ahead(wg.isa(), g, h, 1)?;
    }
    fold::scale(wg.isa(), g)?;
    wg.wait_arrived(b2)
}

/// The trip counter splits `n = K/256 - 1` full groups into `q = n / 3` loop
/// trips (left in the trip register) and `r = n mod 3` leading groups (left in
/// the first scratch SGPR, `s_tmp0`).
pub(crate) fn split_trips(wg: &mut Wg, g: &Gen) -> Result<(), String> {
    let b = wg.isa();
    let (r, q) = (g.tmp, g.tmp + 1);
    let o = |b: &mut crate::Builder, t: String, d: &[u8], u: &[u8]| {
        op(
            b,
            t,
            &d.iter().map(|&n| s(n)).collect::<Vec<_>>(),
            &u.iter().map(|&n| s(n)).collect::<Vec<_>>(),
        )
    };
    o(
        b,
        format!("s_mul_hi_u32 s{q}, s{}, 0xaaaaaaab", g.trips),
        &[q],
        &[g.trips],
    )?;
    o(b, format!("s_lshr_b32 s{q}, s{q}, 1"), &[q], &[q])?;
    o(b, format!("s_mul_i32 s{r}, s{q}, 3"), &[r], &[q])?;
    o(
        b,
        format!("s_sub_co_i32 s{r}, s{}, s{r}", g.trips),
        &[r],
        &[g.trips, r],
    )?;
    o(b, format!("s_mov_b32 s{}, s{q}", g.trips), &[g.trips], &[q])
}

/// Compare the leading-group count `r` (in `s_tmp0`, left by [`split_trips`])
/// with `value`: `s_cmp_lg_u32`, true when `r != value`.
pub(crate) fn ring_cmp(wg: &mut Wg, g: &Gen, value: u32) -> Result<WgUniform<Scc>, String> {
    let r = g.tmp;
    wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_lg_u32 s{r}, {value}"),
        vec![],
        vec![s(r)],
    ))
}

/// Group trip end: the group offset moves on; `count` also drops the loop's
/// trip counter (the last group of a loop trip).
pub(crate) fn trip_end3(wg: &mut Wg, g: &Gen, count: bool) -> Result<(), String> {
    let b = wg.isa();
    op(
        b,
        format!("s_add_co_i32 s{0}, s{0}, 0x88", g.goff),
        &[s(g.goff)],
        &[s(g.goff)],
    )?;
    if count {
        op(
            b,
            format!("s_add_co_i32 s{0}, s{0}, -1", g.trips),
            &[s(g.trips)],
            &[s(g.trips)],
        )?;
    }
    Ok(())
}

/// One full 256-K group: two `Full` blocks (each steps its own plane's
/// descriptor at its head) and the trip end. `end` closes the second block's
/// register lifetimes; `next` is the label of whatever block follows it.
fn group3(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    end: &str,
    count: bool,
    lds: Lds3,
) -> Result<Lds3, String> {
    let m1 = kloop::label(&format!("{tag}m1"));
    let lds = block3(
        wg,
        g,
        &format!("{tag}m0"),
        0,
        Kind3::Full,
        &m1,
        true,
        Some(0),
        lds,
    )?;
    let lds = block3(
        wg,
        g,
        &format!("{tag}m1"),
        1,
        Kind3::Full,
        end,
        true,
        Some(1),
        lds,
    )?;
    trip_end3(wg, g, count)?;
    Ok(lds)
}

/// Arm `r` of the entry branch: rings start at slot `(3 - r) mod 3`, the
/// prologue's ring part, then `r` full groups.
fn arm(wg: &mut Wg, g: &Gen, r: usize, lds: super::FreeRings3) -> Result<Lds3, String> {
    let (ra, rw, rd, rs) = lds;
    let k = (3 - r) % 3;
    let (ra, rw) = (ra.starting_at(k)?, rw.starting_at(k)?);
    let mut lds = super::prologue::ring_part3(wg, g, (ra, rw, rd, rs))?;
    for i in 0..r {
        let tag = format!("a{r}g{i}");
        // The last block's register lifetimes close at a label of its own arm:
        // lifetimes are positional, and the join label follows the other arms.
        let end = if i + 1 == r {
            kloop::label(&format!("a{r}_end"))
        } else {
            kloop::label(&format!("a{r}g{}m0", i + 1))
        };
        lds = group3(wg, g, &tag, &end, false, lds)?;
        if i + 1 == r {
            wg.label(&end)?;
        }
    }
    Ok(lds)
}

/// The depth-3 K-loop with its prologue's ring part and the peeled group.
pub(crate) fn emit(wg: &mut Wg, g: &Gen, lds: super::FreeRings3) -> Result<Lds3, String> {
    wg.label(K_BEGIN)?;
    split_trips(wg, g)?;
    let (join, join12) = (kloop::label("arms_join"), kloop::label("arms_join12"));
    let cmp = ring_cmp;
    let c0 = cmp(wg, g, 0)?;
    let lds = wg.wg_if_else(
        c0,
        &kloop::label("arms_else"),
        &join,
        lds,
        |wg, lds| arm(wg, g, 0, lds),
        |wg, lds| {
            let c1 = cmp(wg, g, 1)?;
            wg.wg_if_else(
                c1,
                &kloop::label("arms_else12"),
                &join12,
                lds,
                |wg, lds| arm(wg, g, 1, lds),
                |wg, lds| arm(wg, g, 2, lds),
            )
        },
    )?;
    // K == 256 or n a multiple of... no loop trip: straight to the peeled group.
    let no_trip = trips_cmp(wg, g, "s_cmp_eq_u32")?;
    let lds = wg.wg_skip_if(no_trip, &kloop::tail_label(0), lds, |wg, lds| {
        let lds = wg.loop_carried(K_LOOP, lds, |wg, lds| {
            let mut lds = lds;
            for i in 0..3 {
                let end = if i == 2 {
                    K_LOOP_END.to_owned()
                } else {
                    kloop::label(&format!("b{}m0", i + 1))
                };
                lds = group3(wg, g, &format!("b{i}"), &end, i == 2, lds)?;
            }
            Ok((lds, trips_cmp(wg, g, "s_cmp_lg_u32")?))
        })?;
        wg.label(K_LOOP_END)?;
        Ok(lds)
    })?;
    // The skip target is the peeled group's first block label.
    let t1 = kloop::label("t1");
    let lds = block3(wg, g, "t0", 0, Kind3::T0, &t1, false, None, lds)?;
    block3(wg, g, "t1", 1, Kind3::T1, EPI, true, None, lds)
}
