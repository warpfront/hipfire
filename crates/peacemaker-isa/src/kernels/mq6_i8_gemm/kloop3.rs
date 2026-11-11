// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The ring-depth-3 K-loop (`Tune { ring: Three }`): the A/W payload rings are three slab buffers
//! deep, so a slab is published **two slabs ahead** of the one being read, and the raw global
//! fetch is two slabs ahead too (two raw prefetch register sets, [`Gen::pf_set`]). Scheduling only:
//! every result bit equals the depth-2 kernel's.
//!
//! # Schedule (block `b` = plane `h` of group `g`; slabs `2b`, `2b + 1`)
//! Entry: slab `2b` in `cur`, slab `2b + 1` published in `ahead`, block `b + 1`'s slab 0 raw in
//! prefetch set `P0` and its slab 1 raw in `P1` (`P` = K32 half 0 and the weight dwords, fetched
//! two blocks of work ago); plane `h` of the DS / SZ rings published (they stay two-deep).
//! - Phase A: fragments from `cur`, interleaved with the publication of block `b + 1`'s slab 0 from
//!   `P0` into `next` (half 1 refetched into `P0` as soon as half 0 is stored); the next block's
//!   metadata fetch; B1 (`rotate` A, W); then block `b + 2`'s slab 0 into `P0`.
//! - Phase B: fragments from the new `cur` (slab `2b + 1`), interleaved with the publication of
//!   block `b + 1`'s slab 1 from `P1` and of its metadata; the fold; B2 (`rotate` A, W, DS, SZ);
//!   then block `b + 2`'s slab 1 into `P1`.
//! - Block `b + 2` is the same plane one group on: its activation descriptor has stepped at the
//!   start of block `b` (`srd_a[h]`), its weight offsets carry the group stride (`goff` moves at
//!   the group end).
//! - The peeled last group: `t0` publishes `t1`'s slabs and fetches nothing ahead; `t1`
//!   publishes and fetches nothing, so its B1 is the store-free `advance`.
//!
//! # Slot period
//! A trip of depth 2 is four slabs; with three buffers the slot of slab `s` repeats every twelve.
//! With `n = K/256 - 1` full groups before the peeled one, `r = n % 3` and `q = n / 3`, slab `s`
//! lives in buffer `(s - r) mod 3`, so three groups (the loop body, `q` trips) and the peeled
//! group all start at slot phase 0 and every LDS address stays an immediate. The `r` leading
//! groups run before the loop in one of three arms (`wg_if_else` on `r`; arm `r` starts its rings
//! at buffer `(3 - r) % 3`, runs the staging of slabs 0 and 1 and `r` full groups) that leave the
//! rings in one state. Code: 3 arm groups + 3 body groups + 1 peeled group.
use super::{fold_and_b2, label, seed_fill, tail_label, trips_cmp, Hooks, Pr, Stage, Work};
use crate::kernels::mq6_i8_gemm::{
    blocks, prologue, publish, spec, Gen, Lds, MetaD, MetaS, PayA, PayW, Wg, EPI, K_BEGIN, K_LOOP,
    K_LOOP_END,
};
use peacemaker_author::{advance, prime, rotate, End, Free, Ring};
use std::collections::VecDeque;

/// Join of the arm selector's first branch (after every arm).
pub const K_ARMS_END: &str = ".Lmq6i8_arms_end";
/// Labels of the arm selector: the second (`r != 0`) branch, the inner join (after arms 1 / 2) and
/// the inner else (arm 2).
pub const ARM12: &str = ".Lmq6i8_arm12";
pub const ARM12_END: &str = ".Lmq6i8_arm12_end";
pub const ARM2: &str = ".Lmq6i8_arm2";

/// The four rings before anything is staged.
pub(crate) type Free3 = (
    Ring<PayA, Free, Free, 3>,
    Ring<PayW, Free, Free, 3>,
    Ring<MetaD, Free, Free>,
    Ring<MetaS, Free, Free>,
);

/// One block of the unrolled schedule.
#[derive(Clone, Copy)]
struct Blk<'a> {
    tag: &'a str,
    /// Plane (K128 parity) of the block.
    h: usize,
    /// Block `b + 1` exists: publish its slabs (A: slab 0 from `P0`; B: slab 1 from `P1`) and
    /// metadata, rotate at B1 / B2.
    pub_next: bool,
    /// Block `b + 2` exists: fetch its slab 0 into `P0` after B1 and its slab 1 into `P1` after B2.
    fetch_ahead: bool,
    /// Activation descriptor to step at the block's start (`srd_a[h]` moves to group `g + 1`).
    step: Option<usize>,
    /// The label that closes the block (the fold registers live until there).
    end: &'a str,
    entry_label: bool,
}

/// Block `b + 2`'s slab `slab` (K32 half 0 and the weight dwords) into prefetch set `gs`: the same
/// plane one group on, so the weight offsets carry the group stride.
fn fetch_ahead(
    b: &mut crate::Builder,
    g: &Gen,
    gs: &Gen,
    slab: u32,
    h: usize,
) -> Result<(), String> {
    publish::fetch_a(b, gs, slab, 0, g.srd_a[h])?;
    publish::fetch_w(b, gs, spec::GROUP_BYTES + 8 + 96 * h as u32 + 48 * slab)
}

/// The publication of block `b + 1`'s slab `slab` from prefetch set `gs`, queued for the
/// interleaver. Phase A (`meta == false`) publishes the activations' half 0 first; phase B
/// finishes with the metadata publication.
fn queue_slab(st: &mut Stage<3>, gs: &Gen, slab: u32, srd: u8, meta: bool) -> Result<(), String> {
    let mut insns = Vec::new();
    let fetch_half1 = Work::FetchA { slab, sb: 1, srd };
    if !meta {
        st.work.push_back(Work::StoreA(0));
        publish::decode_round(&mut insns, gs, 0)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(fetch_half1);
        st.work.push_back(Work::StoreW(0));
    } else {
        publish::decode_round(&mut insns, gs, 0)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(Work::StoreW(0));
        st.work.push_back(Work::StoreA(0));
        st.work.push_back(fetch_half1);
    }
    // A tile with one weight round (128 rows on 512 threads) has no second round to decode.
    if gs.w_pf.len() > 1 {
        publish::decode_round(&mut insns, gs, 1)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(Work::StoreW(1));
    }
    st.work.push_back(Work::StoreA(1));
    if meta {
        st.work.push_back(Work::Meta);
    }
    Ok(())
}

fn block3(
    wg: &mut Wg,
    g: &Gen,
    sets: &[Gen; 2],
    blk: Blk,
    (ra, rw, rd, rs): Lds<3>,
) -> Result<Lds<3>, String> {
    let h = blk.h;
    if rd.cur_index() != h {
        return Err(format!(
            "block {}: LDS rings out of phase with plane {h}",
            blk.tag
        ));
    }
    let (x, xf) = (label(blk.tag), label(&format!("{}_fold", blk.tag)));
    let b = wg.isa();
    if blk.entry_label {
        b.label(&x)?;
    }
    blocks::declare_block_regs(b, g, &x, &xf, blk.end)?;
    if let Some(which) = blk.step {
        blocks::step_descriptor(b, g, which)?;
    }
    let srd_next = g.srd_a[h ^ 1];

    let slot_a = ra.cur_index();
    let mut st = Stage::<3> {
        slot: 0,
        work: VecDeque::new(),
        // The other prefetch set's loads are in flight: the builder's store-source hazard
        // waits for the stores of the half just published, nothing more.
        drain_before_fetch: false,
        a: Pr::Free(ra),
        w: Pr::Free(rw),
        d: Pr::Free(rd),
        s: Pr::Free(rs),
    };

    // ------------------------------------------------------------------ phase A
    if blk.pub_next {
        st.slot = st.a.begin(wg)?;
        st.w.begin(wg)?;
        queue_slab(&mut st, &sets[0], 0, srd_next, false)?;
    }
    // The seed octet is the decode scratch: decode starts after the last seeded WMMA.
    st.phase(wg, &sets[0], slot_a, true, 8, 3, Hooks::default())?;
    st.flush(wg, &sets[0])?;
    if blk.pub_next {
        // Next block's token word and packed row header (header word of the next half; across a
        // group boundary the group stride, since `goff` has not advanced).
        publish::fetch_meta(wg.isa(), g, h, if h == 0 { 4 } else { spec::GROUP_BYTES })?;
    }
    let (ra, rw) = if blk.pub_next {
        let ((ra, pa), (rw, pw)) = (st.a.staged()?, st.w.staged()?);
        let (da, dw) = wg.wait_all((pa, pw))?;
        let b1 = wg.signal((rotate(ra, da), rotate(rw, dw)))?;
        if blk.fetch_ahead {
            fetch_ahead(wg.isa(), g, &sets[0], 0, h)?;
        }
        wg.wait_arrived(b1)?
    } else {
        // The last block publishes nothing: the barrier only retires the slab just read and
        // makes the published `ahead` slab current.
        let (ra, rw) = (st.a.free()?, st.w.free()?);
        let b1 = wg.signal((advance(ra), advance(rw)))?;
        wg.wait_arrived(b1)?
    };
    let slot_b = ra.cur_index();
    st.a = Pr::Free(ra);
    st.w = Pr::Free(rw);

    // ------------------------------------------------------------------ phase B
    if blk.pub_next {
        st.slot = st.a.begin(wg)?;
        st.w.begin(wg)?;
        queue_slab(&mut st, &sets[1], 1, srd_next, true)?;
    }
    st.phase(wg, &sets[1], slot_b, false, 6, 3, Hooks::default())?;
    st.flush(wg, &sets[1])?;

    // ------------------------------------------------------------------ fold + B2
    let (fetch, ahead) = (blk.fetch_ahead, &sets[1]);
    fold_and_b2(
        wg,
        g,
        &mut st,
        h,
        &xf,
        blk.pub_next,
        &|b| {
            if fetch {
                fetch_ahead(b, g, ahead, 1, h)
            } else {
                Ok(())
            }
        },
        blk.pub_next,
    )
}

/// One full 256-K group: block plane 0 (steps `srd_a[0]`), block plane 1 (steps `srd_a[1]`), and
/// the group offsets' move. `gtag` names the blocks; `end` closes the second one.
fn group3(
    wg: &mut Wg,
    g: &Gen,
    sets: &[Gen; 2],
    gtag: &str,
    end: &str,
    lds: Lds<3>,
) -> Result<Lds<3>, String> {
    let (t0, t1) = (format!("{gtag}b0"), format!("{gtag}b1"));
    let l1 = label(&t1);
    let lds = block3(
        wg,
        g,
        sets,
        Blk {
            tag: &t0,
            h: 0,
            pub_next: true,
            fetch_ahead: true,
            step: Some(0),
            end: &l1,
            entry_label: true,
        },
        lds,
    )?;
    let lds = block3(
        wg,
        g,
        sets,
        Blk {
            tag: &t1,
            h: 1,
            pub_next: true,
            fetch_ahead: true,
            step: Some(1),
            end,
            entry_label: true,
        },
        lds,
    )?;
    blocks::group_end(wg.isa(), g)?;
    Ok(lds)
}

/// Staging of block 0's slabs 0 and 1 and block 0's metadata, and the raw fetch of block 1 into
/// both prefetch sets: the depth-3 prologue's ring part. The rings start at buffer `k`, so slab 0
/// lands in buffer `k`, slab 1 in `k + 1`; two barriers (the first primes A, W, DS, SZ with slab 0
/// and plane 0, the second rotates A, W over the still vacant `cur` to slab 0). Ends in the entry
/// state of block 0.
fn head(
    wg: &mut Wg,
    g: &Gen,
    sets: &[Gen; 2],
    (ra, rw, rd, rs): Free3,
    k: usize,
) -> Result<Lds<3>, String> {
    let (ra, rw) = (ra.starting_at(k)?, rw.starting_at(k)?);
    let (g0, g1) = (&sets[0], &sets[1]);
    let (a0r, a1r) = (g.srd_a[0], g.srd_a[1]);

    // Slab 0 (K32 half 0 and the weights were fetched before the arms) and block 0's metadata.
    let slot0 = ra.next_index();
    let (mut sa, mut sw) = (wg.begin_write(ra), wg.begin_write(rw));
    for rnd in 0..g0.w_pf.len() {
        publish::decode_round(wg.isa(), g0, rnd)?;
        sw = publish::store_w_round(wg, g0, sw, slot0, rnd)?;
    }
    sa = publish::store_a_half(wg, g0, sa, slot0, 0)?;
    // The raw staging registers are still sources of the half 0 stores: retire them first.
    wg.isa().wait_all()?;
    publish::fetch_a(wg.isa(), g0, 0, 1, a0r)?;
    sa = publish::store_a_half(wg, g0, sa, slot0, 1)?;
    let ((rd, pd), (rs, ps)) = publish::publish_meta(wg, g, rd, rs)?;
    let (pa, pw) = (sa.1, sw.1);
    let (ra, rw) = (sa.0, sw.0);
    let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
    let first = wg.signal((prime(ra, da), prime(rw, dw), prime(rd, dd), prime(rs, ds)))?;
    // Slab 1's raw half 0 and weights (set 1) and block 1's slab 0 (set 0, free since slab 0's
    // stores drained) fly while the first barrier completes. Block 1's fetch is older than the
    // slab-1 half-1 refetch below, so the loop's entry ledger is the steady state's: set 0's
    // fetch landed, set 1's in flight.
    let b = wg.isa();
    publish::fetch_a(b, g1, 1, 0, a0r)?;
    publish::fetch_w(b, g1, 8 + 48)?;
    publish::fetch_a(b, g0, 0, 0, a1r)?;
    publish::fetch_w(b, g0, 8 + 96)?;
    let (ra, rw, rd, rs) = wg.wait_arrived(first)?;

    // Slab 1 into the next buffer.
    let slot1 = ra.next_index();
    let (mut sa, mut sw) = (wg.begin_write(ra), wg.begin_write(rw));
    for rnd in 0..g1.w_pf.len() {
        publish::decode_round(wg.isa(), g1, rnd)?;
        sw = publish::store_w_round(wg, g1, sw, slot1, rnd)?;
    }
    sa = publish::store_a_half(wg, g1, sa, slot1, 0)?;
    wg.isa().wait_all()?;
    publish::fetch_a(wg.isa(), g1, 1, 1, a0r)?;
    sa = publish::store_a_half(wg, g1, sa, slot1, 1)?;
    let (pa, pw) = (sa.1, sw.1);
    let (ra, rw) = (sa.0, sw.0);
    let (da, dw) = wg.wait_all((pa, pw))?;
    let second = wg.signal((rotate(ra, da), rotate(rw, dw)))?;
    // Block 1's slab 1 into prefetch set 1 (block 1 is the odd plane of group 0) and the WMMA
    // seed.
    let b = wg.isa();
    publish::fetch_a(b, g1, 1, 0, a1r)?;
    publish::fetch_w(b, g1, 8 + 96 + 48)?;
    seed_fill(b, g)?;
    let (ra, rw) = wg.wait_arrived(second)?;
    Ok((ra, rw, rd, rs))
}

/// Arm `r`: the rings start at buffer `(3 - r) % 3`, block 0 is staged, `r` full groups run.
fn arm(wg: &mut Wg, g: &Gen, sets: &[Gen; 2], r: usize, rings: Free3) -> Result<Lds<3>, String> {
    let mut lds = head(wg, g, sets, rings, (3 - r) % 3)?;
    // The arm's last fold ends at a label of its own: another arm's blocks follow textually.
    let end = label(&format!("a{r}_end"));
    for j in 0..r {
        let next = label(&format!("a{r}g{}b0", j + 1));
        let gend = if j + 1 == r { &end } else { &next };
        lds = group3(wg, g, sets, &format!("a{r}g{j}"), gend, lds)?;
    }
    if r > 0 {
        wg.label(&end)?;
    }
    Ok(lds)
}

/// The depth-3 kernel body: prologue front, the `K` split, block 0's fetch, the arms, the loop
/// of three-group trips, the peeled group. Leaves the kernel at the epilogue label's position.
pub(crate) fn emit(wg: &mut Wg, g: &Gen, end: &End, rings: Free3) -> Result<(), String> {
    prologue::pre(wg, g, end)?;
    blocks::ring3_split(wg.isa(), g)?;
    prologue::fetch0(wg.isa(), g)?;
    prologue::zero_acc(wg.isa(), g)?;
    wg.label(K_BEGIN)?;
    let sets = [g.pf_set(0)?, g.pf_set(1)?];

    let lt =
        |wg: &mut Wg, imm: u32| wg.scmp_wg_uniform(blocks::ring3_cmp_insn(g, "s_cmp_lg_u32", imm)?);
    let sel = lt(wg, 0)?;
    let lds = wg.wg_if_else(
        sel,
        ARM12,
        K_ARMS_END,
        rings,
        |wg, rings| arm(wg, g, &sets, 0, rings),
        |wg, rings| {
            let sel = lt(wg, 1)?;
            wg.wg_if_else(
                sel,
                ARM2,
                ARM12_END,
                rings,
                |wg, rings| arm(wg, g, &sets, 1, rings),
                |wg, rings| arm(wg, g, &sets, 2, rings),
            )
        },
    )?;

    // `q == 0`: no full trip; go straight to the peeled group.
    let no_trip = trips_cmp(wg, g, "s_cmp_eq_u32")?;
    let lds = wg.wg_skip_if(no_trip, &tail_label(0), lds, |wg, lds| {
        let lds = wg.loop_carried(K_LOOP, lds, |wg, lds| {
            let mut lds = lds;
            for i in 0..3 {
                let next = label(&format!("m{}b0", i + 1));
                let gend = if i == 2 { K_LOOP_END } else { &next };
                lds = group3(wg, g, &sets, &format!("m{i}"), gend, lds)?;
            }
            blocks::trips_dec(wg.isa(), g)?;
            Ok((lds, trips_cmp(wg, g, "s_cmp_lg_u32")?))
        })?;
        wg.label(K_LOOP_END)?;
        Ok(lds)
    })?;

    // The skip target is the peeled group's first block label.
    let t1 = label("t1");
    let lds = block3(
        wg,
        g,
        &sets,
        Blk {
            tag: "t0",
            h: 0,
            pub_next: true,
            fetch_ahead: false,
            step: None,
            end: &t1,
            entry_label: false,
        },
        lds,
    )?;
    block3(
        wg,
        g,
        &sets,
        Blk {
            tag: "t1",
            h: 1,
            pub_next: false,
            fetch_ahead: false,
            step: None,
            end: EPI,
            entry_label: true,
        },
        lds,
    )?;
    Ok(())
}
