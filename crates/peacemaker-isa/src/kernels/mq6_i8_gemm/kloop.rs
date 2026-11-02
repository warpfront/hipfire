// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The rolled K-loop. One trip is a whole 256-K group (two 128-K blocks, plane parity 0 then 1),
//! so every LDS address is an immediate and the trip ends in the state it started in. The last
//! group is peeled: its second block fetches and publishes nothing.
//!
//! Block `b` (plane `h`, entry state: slab 0 published in slot 0, slab 1's raw A half 0 and
//! weight dwords in `a_pf` / `w_pf`, token/row planes `h` published):
//! - Phase A: four K16 steps from slot 0 on the two fragment sets (`F0`, `F1` ping-pong: the
//!   next-but-one step loads while the current one issues), interleaved with the decode and
//!   publication of slab 1 into slot 1; the next block's metadata fetch (issued with the half-1
//!   refetch after WMMA 7; `StoreA(1)` is held to after WMMA 24, see `Spec::stacked`); B1
//!   retires slot 0 and publishes slot 1.
//! - Phase B: next-block slab-0 fetch; four K16 steps from slot 1 interleaved with the decode and
//!   publication of the next block's slab 0 into slot 0 and of its metadata (published first);
//!   the fold of this
//!   block (row group 0 first; row group 1's `sc` lands in the already folded accumulators'
//!   dead registers, its `z'` in the decode scratch); B2 is signalled after the last LDS load and
//!   waited at the block end, overlapping the fold's remaining packets. The block's last
//!   instructions refill the scratch with the next block's WMMA seed.
use super::{
    blocks, fold, publish, spec, Gen, Lds, MetaD, MetaS, PayA, PayW, Staged, Wg, EPI, K_LOOP,
    K_LOOP_END,
};
use crate::{insn::Instruction, Builder};
use peacemaker_author::{rotate, Free, Published, Ring, Scc, WgUniform};
use std::collections::VecDeque;

/// The depth-3 ring K-loop (prologue split on `K`, three-group period; shares this module's
/// interleaver).
#[path = "kloop3.rs"]
pub mod ring3;

pub(crate) fn label(tag: &str) -> String {
    format!(".Lmq6i8_{tag}")
}
pub(crate) fn tail_label(h: usize) -> String {
    label(&format!("t{h}"))
}

pub(crate) fn trips_cmp(wg: &mut Wg, g: &Gen, cmp: &str) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(blocks::trips_cmp_insn(g, cmp)?)
}

pub(crate) fn emit(wg: &mut Wg, g: &Gen, lds: Lds) -> Result<Lds, String> {
    wg.label(super::K_BEGIN)?;
    // K == 256: no full trip; go straight to the peeled group.
    let no_trip = trips_cmp(wg, g, "s_cmp_eq_u32")?;
    let lds = wg.wg_skip_if(no_trip, &tail_label(0), lds, |wg, lds| {
        let lds = wg.loop_carried(K_LOOP, lds, |wg, lds| {
            let lds = block_with(wg, g, "m0", 0, true, &label("m1"), true, |_| Ok(()), lds)?;
            // The even-block descriptor was last read by block 0's slab-1 fetch; move it to block
            // 2g+2 before block 1's next-block fetch reads it.
            let lds = block_with(
                wg,
                g,
                "m1",
                1,
                true,
                K_LOOP_END,
                true,
                |b| blocks::step_descriptor(b, g, 0),
                lds,
            )?;
            // Trip end: the odd-block descriptor and the group offsets move on after every read of
            // them in this trip.
            blocks::trip_end(wg.isa(), g)?;
            Ok((lds, trips_cmp(wg, g, "s_cmp_lg_u32")?))
        })?;
        wg.label(K_LOOP_END)?;
        Ok(lds)
    })?;
    // The skip target is the peeled group's first block label.
    let lds = block_with(wg, g, "t0", 0, true, &label("t1"), false, |_| Ok(()), lds)?;
    block_with(wg, g, "t1", 1, false, EPI, true, |_| Ok(()), lds)
}

/// A ring as the interleaver holds it: not yet staging, or staging (next buffer `Writing`).
enum Pr<R: 'static, const D: usize = 2> {
    Free(Ring<R, Published, Free, D>),
    Stg(Staged<R, Published, D>),
    Taken,
}
impl<R: 'static, const D: usize> Pr<R, D> {
    fn ds_load(&self, wg: &mut Wg, insn: Instruction) -> Result<(), String> {
        match self {
            Self::Free(r) => wg.ds_load_cur(r, insn),
            Self::Stg((r, _)) => wg.ds_load_cur(r, insn),
            Self::Taken => Err("ring is taken".into()),
        }
    }
    fn begin(&mut self, wg: &mut Wg) -> Result<usize, String> {
        match std::mem::replace(self, Self::Taken) {
            Self::Free(r) => {
                let slot = r.next_index();
                *self = Self::Stg(wg.begin_write(r));
                Ok(slot)
            }
            _ => Err("begin on a ring that is not Free".into()),
        }
    }
    fn staged(&mut self) -> Result<Staged<R, Published, D>, String> {
        match std::mem::replace(self, Self::Taken) {
            Self::Stg(s) => Ok(s),
            _ => Err("ring is not staging".into()),
        }
    }
    fn free(&mut self) -> Result<Ring<R, Published, Free, D>, String> {
        match std::mem::replace(self, Self::Taken) {
            Self::Free(r) => Ok(r),
            _ => Err("ring is not Free".into()),
        }
    }
}

/// One deferred unit of the interleaved slab publication.
enum Work {
    Insn(Instruction),
    StoreW(usize),
    StoreA(u32),
    FetchA {
        slab: u32,
        sb: u32,
        srd: u8,
    },
    /// The next block's token word and packed row header (`fetch_meta(h, off)`).
    FetchMeta {
        h: usize,
        off: u32,
    },
    Meta,
}

/// Units a phase places at fixed WMMAs besides the steady `per`-per-WMMA drain: `burst` units
/// after WMMA `burst_at`, and `hold` appended and drained after WMMA `hold_at` (none by default).
struct Hooks {
    burst_at: usize,
    burst: usize,
    hold_at: usize,
    hold: Vec<Work>,
}
impl Default for Hooks {
    fn default() -> Self {
        Self {
            burst_at: usize::MAX,
            burst: 0,
            hold_at: usize::MAX,
            hold: Vec::new(),
        }
    }
}

/// The four rings plus the queue of work to interleave between WMMAs.
/// The queue is the slab publication of one block; `drain_before_fetch` keeps the shipped
/// ledger-wide wait before a half-1 refetch of the raw staging registers (depth 2), or leaves the
/// wait to the builder's store-source hazard (depth 3, where the other prefetch set's loads are
/// in flight and must not be drained).
struct Stage<const D: usize = 2> {
    slot: usize,
    work: VecDeque<Work>,
    drain_before_fetch: bool,
    a: Pr<PayA, D>,
    w: Pr<PayW, D>,
    d: Pr<MetaD>,
    s: Pr<MetaS>,
}
impl<const D: usize> Stage<D> {
    fn push_insns(&mut self, insns: Vec<Instruction>) {
        self.work.extend(insns.into_iter().map(Work::Insn));
    }
    fn pump(&mut self, wg: &mut Wg, g: &Gen, n: usize) -> Result<(), String> {
        for _ in 0..n {
            let Some(item) = self.work.pop_front() else {
                return Ok(());
            };
            match item {
                Work::Insn(i) => wg.isa().push(i)?,
                Work::StoreW(r) => {
                    let sw = self.w.staged()?;
                    self.w = Pr::Stg(publish::store_w_round(wg, g, sw, self.slot, r)?);
                }
                Work::StoreA(sb) => {
                    let sa = self.a.staged()?;
                    self.a = Pr::Stg(publish::store_a_half(wg, g, sa, self.slot, sb)?);
                }
                Work::FetchA { slab, sb, srd } => {
                    // The raw staging registers are still sources of the previous half's stores.
                    if self.drain_before_fetch {
                        blocks::wait_all(wg.isa())?;
                    }
                    publish::fetch_a(wg.isa(), g, slab, sb, srd)?
                }
                Work::FetchMeta { h, off } => publish::fetch_meta(wg.isa(), g, h, off)?,
                Work::Meta => {
                    let (rd, rs) = (self.d.free()?, self.s.free()?);
                    let (sd, ss) = publish::publish_meta(wg, g, rd, rs)?;
                    self.d = Pr::Stg(sd);
                    self.s = Pr::Stg(ss);
                }
            }
        }
        Ok(())
    }
    fn flush(&mut self, wg: &mut Wg, g: &Gen) -> Result<(), String> {
        let n = self.work.len();
        self.pump(wg, g, n)
    }

    /// `ds_load_b128` of one K16 step (`step` = `sb * 2 + e`) into fragment set `set` from slot
    /// `slot`: the weight row-group pair, then the two token-group pairs.
    fn step_loads(
        &self,
        wg: &mut Wg,
        g: &Gen,
        slot: usize,
        step: u32,
        set: usize,
    ) -> Result<(), String> {
        let [w, a0, a1] = blocks::step_load_insns(g, slot, step, set)?;
        self.w.ds_load(wg, w)?;
        self.a.ds_load(wg, a0)?;
        self.a.ds_load(wg, a1)
    }

    /// Four K16 steps from `slot`; after the `n`-th WMMA (`n >= start`) `per` work units run, plus
    /// the units `hooks` places at fixed WMMAs.
    fn phase(
        &mut self,
        wg: &mut Wg,
        g: &Gen,
        slot: usize,
        first: bool,
        start: usize,
        per: usize,
        mut hooks: Hooks,
    ) -> Result<(), String> {
        self.step_loads(wg, g, slot, 0, 0)?;
        self.step_loads(wg, g, slot, 1, 1)?;
        for step in 0..4u32 {
            let set = (step & 1) as usize;
            let base = 8 * step as usize;
            mma_step(wg, g, step, set, first && step == 0, &mut |wg, i| {
                if base + i == hooks.burst_at {
                    self.pump(wg, g, hooks.burst)?;
                }
                if base + i >= start {
                    self.pump(wg, g, per)?;
                }
                if base + i == hooks.hold_at {
                    // Held units (the last-queued consumer of the global refetch) are appended and
                    // run now: the refetch had the whole span in between to land.
                    self.work.extend(hooks.hold.drain(..));
                    let n = self.work.len();
                    self.pump(wg, g, n)?;
                }
                Ok(())
            })?;
            if step + 2 < 4 {
                self.step_loads(wg, g, slot, step + 2, set)?;
            }
        }
        Ok(())
    }
}

/// The eight WMMAs of one K16 step on fragment set `set`; `after(wg, i)` runs after the `i`-th.
fn mma_step(
    wg: &mut Wg,
    g: &Gen,
    step: u32,
    set: usize,
    first: bool,
    after: &mut dyn FnMut(&mut Wg, usize) -> Result<(), String>,
) -> Result<(), String> {
    for i in 0..fold::BUNDLE_ORDER.len() {
        blocks::mma(wg.isa(), g, step, set, first, i)?;
        after(wg, i)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn block_with(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end: &str,
    entry_label: bool,
    after_slab1_fetch: impl Fn(&mut Builder) -> Result<(), String>,
    (ra, rw, rd, rs): Lds,
) -> Result<Lds, String> {
    if rd.cur_index() != h || ra.cur_index() != 0 {
        return Err(format!(
            "block {tag}: LDS rings out of phase with plane {h}"
        ));
    }
    let (x, xf) = (label(tag), label(&format!("{tag}_fold")));
    let b = wg.isa();
    if entry_label {
        b.label(&x)?;
    }
    blocks::declare_block_regs(b, g, &x, &xf, end)?;
    after_slab1_fetch(b)?;

    let mut st = Stage {
        slot: 0,
        work: VecDeque::new(),
        drain_before_fetch: true,
        a: Pr::Free(ra),
        w: Pr::Free(rw),
        d: Pr::Free(rd),
        s: Pr::Free(rs),
    };

    // ------------------------------------------------------------------ phase A
    // The stacked schedule (every epilogue but `SetNr`, `Spec::stacked`) queues the half-1
    // refetch and the next block's metadata fetch first (issued after WMMA 7, so both are long
    // landed when phase B consumes them) and holds `StoreA(1)`, the last consumer of the
    // refetch, to after WMMA 24.
    let stacked = g.spec.stacked();
    st.slot = st.a.begin(wg)?;
    st.w.begin(wg)?;
    let mut insns = Vec::new();
    // Half 0 of this slab's activations was fetched a block ago: publish it first, so the
    // refill of the raw staging registers (which must wait for these stores to read them) finds
    // them long drained.
    st.work.push_back(Work::StoreA(0));
    let fetch_half1 = Work::FetchA {
        slab: 1,
        sb: 1,
        srd: g.srd_a[h],
    };
    if stacked {
        st.work.push_back(fetch_half1);
        if fetch_next {
            // Next block's token word and packed row header (header word of the next half; across
            // a group boundary the group stride, since `goff` has not advanced).
            st.work.push_back(Work::FetchMeta {
                h,
                off: if h == 0 { 4 } else { spec::GROUP_BYTES },
            });
        }
        publish::decode_round(&mut insns, g, 0)?;
        st.push_insns(std::mem::take(&mut insns));
    } else {
        publish::decode_round(&mut insns, g, 0)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(fetch_half1);
    }
    st.work.push_back(Work::StoreW(0));
    // A tile with one weight round (128 rows on 512 threads) has no second round to decode.
    if g.w_pf.len() > 1 {
        publish::decode_round(&mut insns, g, 1)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(Work::StoreW(1));
    }
    // The seed octet is the decode scratch: decode starts after the last seeded WMMA.
    let hooks = if stacked {
        Hooks {
            burst_at: 7,
            burst: if fetch_next { 3 } else { 2 },
            hold_at: 24,
            hold: vec![Work::StoreA(1)],
        }
    } else {
        st.work.push_back(Work::StoreA(1));
        Hooks::default()
    };
    st.phase(wg, g, 0, true, 8, 3, hooks)?;
    st.flush(wg, g)?;
    if fetch_next && !stacked {
        // Next block's token word and packed row header (header word of the next half; across a
        // group boundary the group stride, since `goff` has not advanced).
        publish::fetch_meta(wg.isa(), g, h, if h == 0 { 4 } else { spec::GROUP_BYTES })?;
    }
    // B1 is split: the next block's raw slab-0 fetch (VMEM, its registers are free once the
    // stores drained) overlaps the rendezvous.
    let ((ra, pa), (rw, pw)) = (st.a.staged()?, st.w.staged()?);
    let (da, dw) = wg.wait_all((pa, pw))?;
    let b1 = wg.signal((rotate(ra, da), rotate(rw, dw)))?;
    if fetch_next {
        let b = wg.isa();
        publish::fetch_a(b, g, 0, 0, g.srd_a[h ^ 1])?;
        publish::fetch_w(
            b,
            g,
            if h == 0 {
                8 + 96
            } else {
                spec::GROUP_BYTES + 8
            },
        )?;
    }
    let (ra, rw) = wg.wait_arrived(b1)?;
    st.a = Pr::Free(ra);
    st.w = Pr::Free(rw);

    // ------------------------------------------------------------------ phase B
    // Stacked: the metadata publication first (its words land ahead of the slab-0 raw fetch),
    // `StoreA(1)` held to after WMMA 24.
    let mut hooks = Hooks::default();
    if fetch_next {
        st.slot = st.a.begin(wg)?;
        st.w.begin(wg)?;
        if stacked {
            st.work.push_back(Work::Meta);
        }
        publish::decode_round(&mut insns, g, 0)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(Work::StoreW(0));
        st.work.push_back(Work::StoreA(0));
        st.work.push_back(Work::FetchA {
            slab: 0,
            sb: 1,
            srd: g.srd_a[h ^ 1],
        });
        if g.w_pf.len() > 1 {
            publish::decode_round(&mut insns, g, 1)?;
            st.push_insns(insns);
            st.work.push_back(Work::StoreW(1));
        }
        if stacked {
            hooks.hold_at = 24;
            hooks.hold = vec![Work::StoreA(1)];
        } else {
            st.work.push_back(Work::StoreA(1));
            st.work.push_back(Work::Meta);
        }
    }
    st.phase(wg, g, 1, false, 6, 3, hooks)?;
    st.flush(wg, g)?;

    // ------------------------------------------------------------------ fold
    fold_and_b2(
        wg,
        g,
        &mut st,
        h,
        &xf,
        fetch_next,
        &|b| {
            // Next block's slab 1 (K32 half 0 and the weight codes): `goff` has not advanced at
            // the odd-to-even boundary, so its offsets carry the group stride.
            publish::fetch_a(b, g, 1, 0, g.srd_a[h ^ 1])?;
            publish::fetch_w(
                b,
                g,
                if h == 0 {
                    8 + 96 + 48
                } else {
                    spec::GROUP_BYTES + 8 + 48
                },
            )
        },
        fetch_next,
    )
}

/// The fold of block plane `h` (label `xf`) and the block-end barrier B2: the plane's `(d, float(s))`
/// pairs and row `sc` / `z'` octets are loaded from the DS / SZ rings and folded into `acc`.
/// With `b2`, B2 is signalled once the fold's LDS reads are issued (`after_signal` issues the
/// fetch that overlaps the rendezvous) and waited at the block end; with `seed`, the next block's
/// WMMA seed octet is refilled after row group 1's fold. Returns the four rings.
fn fold_and_b2<const D: usize>(
    wg: &mut Wg,
    g: &Gen,
    st: &mut Stage<D>,
    h: usize,
    xf: &str,
    b2: bool,
    after_signal: &dyn Fn(&mut Builder) -> Result<(), String>,
    seed: bool,
) -> Result<Lds<D>, String> {
    wg.label(xf)?;
    let scratch = g.scratch;
    // Plane `h` addresses: token (d, float(s)) pairs via `msn` as the address temporary, rows' sc / z'
    // from `sc_addr`.
    blocks::fold_addr(wg.isa(), g, h)?;
    let sz_load = |wg: &mut Wg, st: &Stage<D>, dst: u8, rel: u32| -> Result<(), String> {
        st.s.ds_load(wg, blocks::fold_sz_load_insn(g, h, dst, rel)?)
    };
    for nb in 0..4usize {
        st.d.ds_load(wg, blocks::fold_token_load_insn(g, nb)?)?;
    }
    // Row group 0: sc octet then z' octet (planar, 8 rows of this lane's row half).
    sz_load(wg, st, fold::SC, 0)?;
    sz_load(wg, st, fold::SC + 4, 16)?;
    sz_load(wg, st, fold::ZP, 512)?;
    sz_load(wg, st, fold::ZP + 4, 512 + 16)?;
    // Row group 1's z' into the decode scratch (free until the next block's decode).
    sz_load(wg, st, scratch, 64 + 512)?;
    sz_load(wg, st, scratch + 4, 64 + 512 + 16)?;

    let frag = |wg: &mut Wg, nb: usize, rg: usize, sc: u8, zp: u8| -> Result<(), String> {
        let b = wg.isa();
        blocks::emit_msn(b, g, nb)?;
        blocks::fold_fragment(b, g, nb * 2 + rg, sc, zp)
    };
    frag(wg, 0, 0, fold::SC, fold::ZP)?;
    // Fragment k = 0 is folded: its int32 registers now hold row group 1's sc.
    sz_load(wg, st, g.cacc, 64)?;
    sz_load(wg, st, g.cacc + 4, 64 + 16)?;
    frag(wg, 1, 0, fold::SC, fold::ZP)?;

    let arrived = if b2 {
        let ((ra, pa), (rw, pw)) = (st.a.staged()?, st.w.staged()?);
        let ((rd, pd), (rs, ps)) = (st.d.staged()?, st.s.staged()?);
        let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
        let b2 = wg.signal((
            rotate(ra, da),
            rotate(rw, dw),
            rotate(rd, dd),
            rotate(rs, ds),
        ))?;
        after_signal(wg.isa())?;
        Some(b2)
    } else {
        None
    };
    frag(wg, 2, 0, fold::SC, fold::ZP)?;
    frag(wg, 3, 0, fold::SC, fold::ZP)?;
    for nb in 0..4 {
        frag(wg, nb, 1, g.cacc, scratch)?;
    }
    if seed {
        // Next block's WMMA seed octet (the scratch is dead after row group 1's fold).
        seed_fill(wg.isa(), g)?;
    }
    match arrived {
        Some(b2) => wg.wait_arrived(b2),
        None => Ok((st.a.free()?, st.w.free()?, st.d.free()?, st.s.free()?)),
    }
}

/// `scratch[0..8] = 0x4B400000`: the WMMA chain seed (`as_float` of `0x4B400000 + n` is exactly
/// `12582912 + n` for `|n| < 2^22`).
pub(crate) fn seed_fill(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for i in 0..8u8 {
        crate::kernels::common::op(
            b,
            format!(
                "v_mov_b32_e32 v{}, {}",
                g.scratch + i,
                crate::kernels::common::lit(0x4b40_0000)
            ),
            &[crate::kernels::common::v(g.scratch + i)],
            &[],
        )?;
    }
    Ok(())
}
