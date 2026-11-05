// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave-private LDS regions (`Driver::lds_private`, typed `WaveRegion`): one
//! wave's slice of an LDS window that no other wave touches. Publish and
//! retire are wave-local state steps, so they emit no `s_barrier` and no
//! counter wait and are legal under wave-uniform control (`skip_if`,
//! `exec_if`, `if_else`, `loop_until`): one wave's DS instructions execute in
//! issue order (RDNA3 ISA 12.5, RDNA4 ISA 12.5 and 5.7.1.4), so the owning
//! wave's load after its own store sees the stored bytes.
//!
//! Accept paths run on the reference backend (`trace::Trace`, which never
//! inserts a drain itself); reject paths check that every illegal request is
//! refused before it reaches the backend. The production backend's twin
//! tests (slot machine, Builder, certifier, emulator) are in
//! `peacemaker-isa`.
use peacemaker_author::runtime::{Driver, PrivatePhase, Transition};
use peacemaker_author::trace::Trace;
use peacemaker_author::{
    ready, Free, Gfx1100, Gfx1151, Gfx1201, Owned, Target, WaveRegion, Workgroup, Writing,
};

enum Scratch {}

const ST: &str = "ds_store_b32 v1, v2";
const LD: &str = "ds_load_b32 v3, v1";
const CMP: &str = "s_cmp_lg_u32 s4, 0";

fn err<T>(r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e,
    }
}

/// Store, publish, load, retire, then again: the wave-local protocol twice.
fn round_trips<T: Target>(d: &mut Driver<'_, T, Trace>, p: peacemaker_author::runtime::PrivateId) {
    for _ in 0..2 {
        d.private_store(p, ST.into()).unwrap();
        d.private_publish(p).unwrap();
        d.private_load(p, LD.into()).unwrap();
        d.private_retire(p).unwrap();
    }
}

fn no_barrier_and_no_wait(b: &Trace) {
    for m in [
        "s_barrier",
        "s_barrier_signal",
        "s_barrier_wait",
        "s_waitcnt",
        "s_wait_dscnt",
    ] {
        assert_eq!(b.count(m), 0, "{m} in {:?}", b.text);
    }
}

fn under_skip_if<T: Target>(arch: &'static str) -> Trace {
    let mut b = Trace::new(arch);
    let mut d = Driver::<T, Trace>::new(&mut b).unwrap();
    let p = d.lds_private("scratch", 4096, 1024, 8).unwrap();
    // Only some waves run the body: wave-uniform control.
    let c = d.scmp(CMP.into()).unwrap();
    d.skip_if(c, ".Lskip", |d| {
        round_trips(d, p);
        Ok(())
    })
    .unwrap();
    assert_eq!(d.private_phase(p).unwrap(), PrivatePhase::RtFree);
    b
}

#[test]
fn round_trip_under_wave_uniform_control_needs_no_barrier_and_no_wait() {
    for b in [
        under_skip_if::<Gfx1100>("gfx1100"),
        under_skip_if::<Gfx1151>("gfx1151"),
        under_skip_if::<Gfx1201>("gfx1201"),
    ] {
        no_barrier_and_no_wait(&b);
        assert_eq!((b.count("ds_store_b32"), b.count("ds_load_b32")), (2, 2));
        assert_eq!(
            b.text,
            [
                CMP,
                "s_cbranch_scc1 .Lskip",
                ST,
                LD,
                ST,
                LD,
                ".Lskip:"
            ]
        );
    }
}

#[test]
fn the_typed_core_emits_the_drivers_exact_trace() {
    let mut typed = Trace::new("gfx1201");
    {
        let mut wg = Workgroup::<Gfx1201, Trace>::new(&mut typed).unwrap();
        let p = wg.lds_private::<Scratch>("scratch", 4096, 1024, 8).unwrap();
        let c = wg.scmp(CMP.into()).unwrap();
        let _p = wg
            .skip_if(c, ".Lskip", p, |w, mut p| {
                for _ in 0..2 {
                    let writing = w.private_store(p, ST.into())?;
                    let owned = w.private_publish(writing)?;
                    w.private_load(&owned, LD.into())?;
                    p = w.private_retire(owned)?;
                }
                Ok(p)
            })
            .unwrap();
    }
    assert_eq!(typed.text, under_skip_if::<Gfx1201>("gfx1201").text);
}

#[test]
fn legal_under_exec_if_if_else_and_loop_until() {
    let mut b = Trace::new("gfx1151");
    let mut d = Driver::<Gfx1151, Trace>::new(&mut b).unwrap();
    let p = d.lds_private("scratch", 0, 512, 16).unwrap();
    let c = d.scmp(CMP.into()).unwrap();
    d.exec_if(c, |d| {
        round_trips(d, p);
        Ok(())
    })
    .unwrap();
    let c = d.scmp(CMP.into()).unwrap();
    d.if_else(
        c,
        ".Lelse",
        ".Ljoin",
        |d| {
            round_trips(d, p);
            Ok(())
        },
        |d| {
            d.private_store(p, ST.into())?;
            d.private_publish(p)?;
            d.private_load(p, LD.into())?;
            d.private_retire(p)
        },
    )
    .unwrap();
    // A loop body drains its own store (the back-edge ledger must equal the
    // entry's, so the stores before the loop are drained first); publishing
    // and retiring still emit nothing.
    d.private_drain(p).unwrap();
    d.loop_until(".Lhead", ".Lexit", |d, exit| {
        d.private_store(p, ST.into())?;
        d.private_drain(p)?;
        d.private_publish(p)?;
        d.private_load(p, LD.into())?;
        d.private_retire(p)?;
        let c = d.scmp(CMP.into())?;
        d.break_if(c, exit)
    })
    .unwrap();
    assert_eq!(b.count("s_barrier"), 0);
    // Every wait is a requested drain: one before the loop, one in the body
    // (emitted once per fixed-point run).
    assert!(b.count("s_waitcnt") >= 2, "{:?}", b.text);
}

#[test]
fn a_shared_region_still_publishes_by_barrier_beside_a_private_one() {
    // With the private store drained, the shared barrier goes through...
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let tile = d.lds("tile", 0, 2048).unwrap();
    let p = d.lds_private("scratch", 2048, 256, 4).unwrap();
    d.ds_store(tile, "ds_store_b128 v1, v[2:5]".into()).unwrap();
    d.wait(tile).unwrap();
    d.barrier(&[Transition::Ready(tile)]).unwrap();
    d.ds_load(tile, "ds_load_b128 v[8:11], v6".into()).unwrap();
    round_trips(&mut d, p);
    d.private_drain(p).unwrap();
    d.barrier(&[Transition::Retire(tile)]).unwrap();
    assert_eq!(b.count("s_wait_dscnt"), 2, "{:?}", b.text);
    assert_eq!(b.count("s_barrier_signal"), 2);
    // ...and without the drain the private store is an ordinary DS store to
    // a workgroup barrier (`s_barrier` waits for no memory counter), so the
    // reference backend refuses the barrier exactly as for a shared region.
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let tile = d.lds("tile", 0, 2048).unwrap();
    let p = d.lds_private("scratch", 2048, 256, 4).unwrap();
    d.ds_store(tile, "ds_store_b128 v1, v[2:5]".into()).unwrap();
    d.wait(tile).unwrap();
    d.barrier(&[Transition::Ready(tile)]).unwrap();
    d.ds_load(tile, "ds_load_b128 v[8:11], v6".into()).unwrap();
    round_trips(&mut d, p);
    let e = err(d.barrier(&[Transition::Retire(tile)]));
    assert!(e.contains("LDS store pending"), "{e}");
}

#[test]
fn protocol_violations_are_refused_before_the_backend() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let p = d.lds_private("scratch", 0, 1024, 8).unwrap();
    // Nothing to read, publish or retire in Free.
    assert!(err(d.private_load(p, LD.into())).contains("not published to its wave"));
    assert!(err(d.private_publish(p)).contains("needs a store first"));
    assert!(err(d.private_retire(p)).contains("only an Owned slice retires"));
    assert!(err(d.private_drain(p)).contains("no store of this wave is in flight"));
    d.private_store(p, ST.into()).unwrap();
    assert_eq!(d.private_phase(p).unwrap(), PrivatePhase::RtWriting);
    // Writing: readable only once published.
    assert!(err(d.private_load(p, LD.into())).contains("not published to its wave"));
    assert!(err(d.private_retire(p)).contains("only an Owned slice retires"));
    d.private_publish(p).unwrap();
    // Owned: not writable until retired, not publishable twice.
    assert!(err(d.private_store(p, ST.into())).contains("Owned"));
    assert!(err(d.private_publish(p)).contains("needs a store first"));
    assert!(err(d.relayout()).contains("retire it first"));
    d.private_load(p, LD.into()).unwrap();
    d.private_retire(p).unwrap();
    d.relayout().unwrap();
    // A relayout consumes the region.
    assert!(err(d.private_store(p, ST.into())).contains("relayout"));
    // No refused request emitted anything.
    assert_eq!((b.count("ds_store_b32"), b.count("ds_load_b32")), (1, 1));
}

#[test]
fn an_id_from_another_driver_is_refused() {
    let (mut a, mut c) = (Trace::new("gfx1201"), Trace::new("gfx1201"));
    let mut da = Driver::<Gfx1201, Trace>::new(&mut a).unwrap();
    let mut dc = Driver::<Gfx1201, Trace>::new(&mut c).unwrap();
    let p = da.lds_private("scratch", 0, 256, 2).unwrap();
    assert!(err(dc.private_store(p, ST.into())).contains("another Driver"));
}

#[test]
fn layouts_must_be_aligned_in_range_and_wave_scope_is_refused() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    for (base, stride, waves, want) in [
        (0, 1024, 0, "0 waves"),
        (0, 1024, 33, "33 waves"),
        (0, 0, 4, "multiple of 16"),
        (0, 1000, 4, "multiple of 16"),
        (8, 1024, 4, "multiple of 16"),
        (0, 4096, 17, "exceeds 65536 bytes"),
        (65536 - 16, 16, 2, "exceeds 65536 bytes"),
        (u32::MAX - 15, 16, 2, "exceeds 65536 bytes"),
        (0, u32::MAX - 15, 2, "exceeds 65536 bytes"),
    ] {
        let e = err(d.lds_private("scratch", base, stride, waves));
        assert!(e.contains(want), "({base},{stride},{waves}): {e}");
    }
    // The full window is fine.
    d.lds_private("full", 0, 2048, 32).unwrap();
    // Declaring a region is workgroup scope: not under wave-uniform control.
    let c = d.scmp(CMP.into()).unwrap();
    let e = err(d.skip_if(c, ".Lskip", |d| d.lds_private("late", 4096, 256, 2).map(drop)));
    assert!(e.contains("workgroup scope"), "{e}");
}

#[test]
fn paths_must_agree_on_a_private_regions_ownership() {
    // Owned on the body's path only: the waves that skipped it hold nothing.
    for (publish, want) in [(true, "Owned"), (false, "Writing")] {
        let mut b = Trace::new("gfx1201");
        let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
        let p = d.lds_private("scratch", 0, 256, 4).unwrap();
        let c = d.scmp(CMP.into()).unwrap();
        let e = err(d.skip_if(c, ".Lskip", |d| {
            d.private_store(p, ST.into())?;
            if publish {
                d.private_publish(p)?;
            }
            Ok(())
        }));
        assert!(e.contains("wave-private scratch"), "{e}");
        assert!(e.contains(want), "{e}");
    }
    // The if/else arms may not change ownership either.
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let p = d.lds_private("scratch", 0, 256, 4).unwrap();
    let c = d.scmp(CMP.into()).unwrap();
    let e = err(d.if_else(
        c,
        ".Lelse",
        ".Ljoin",
        |d| d.private_store(p, ST.into()),
        |_| Ok(()),
    ));
    assert!(e.contains("changes LDS ownership"), "{e}");
}

#[test]
fn typed_states_are_names_of_the_same_machine() {
    // The typed walk through Free -> Writing -> Owned -> Free.
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let p: WaveRegion<Scratch, Free> = wg.lds_private("scratch", 0, 256, 4).unwrap();
    let w: WaveRegion<Scratch, Writing> = wg.private_store(p, ST.into()).unwrap();
    let w = wg.private_drain(w).unwrap();
    let o: WaveRegion<Scratch, Owned> = wg.private_publish(w).unwrap();
    wg.private_load(&o, LD.into()).unwrap();
    let p: WaveRegion<Scratch, Free> = wg.private_retire(o).unwrap();
    // A workgroup barrier for a shared region beside it.
    let tile = wg.lds::<Scratch>("tile", 4096, 2048).unwrap();
    let (tile, st) = wg.ds_store(tile, "ds_store_b128 v1, v[2:5]".into()).unwrap();
    let st = wg.wait(st).unwrap();
    let (tile,) = wg.barrier((ready(tile, st),)).unwrap();
    wg.ds_load(&tile, "ds_load_b128 v[8:11], v6".into()).unwrap();
    // `AllFree`: the private region relayouts like a shared one.
    let (tile,) = wg.barrier((peacemaker_author::retire(tile),)).unwrap();
    wg.relayout((p, tile)).unwrap();
    assert_eq!(b.count("s_barrier"), 2);
}
