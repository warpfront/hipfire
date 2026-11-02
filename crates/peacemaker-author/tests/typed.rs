// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Driver::typed`: a typed body running inside a runtime driver with rings
//! lent across (`Lend::give` / `take` / `give_back`). A kernel split so its
//! middle runs in `typed` re-emits the pure typed kernel's exact trace; every
//! way of lending a ring wrongly is refused by the driver's own checks; and a
//! script loop around a typed block is checked by the driver's loop rule.
use peacemaker_author::runtime::{Driver, Transition};
use peacemaker_author::trace::Trace;
use peacemaker_author::{
    prime, retire_cur, rotate, Free, Gfx1201, Published, Ring, Wave, Workgroup,
};

enum Tile {}
enum Other {}

const STAGE: &str = "ds_store_b128 v1, v[2:5]";
const LOAD: &str = "ds_load_b128 v[8:11], v6";
const CMP: &str = "s_cmp_lg_u32 s4, 0";

fn err<T>(r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e,
    }
}

type Wg<'a> = Workgroup<'a, Gfx1201, Trace>;
type Steady = Ring<Tile, Published, Free>;

/// The block both kernels share: read the current buffer, stage the next one,
/// drain it and rotate.
fn middle(wg: &mut Wg, ring: Steady) -> Result<Steady, String> {
    wg.ds_load_cur(&ring, LOAD.into())?;
    let (ring, st) = wg.ds_store(ring, STAGE.into())?;
    let drained = wg.wait(st)?;
    let (ring,) = wg.barrier((rotate(ring, drained),))?;
    Ok(ring)
}

/// The block that retires the last published buffer.
fn tail(wg: &mut Wg, ring: Steady) -> Result<Ring<Tile, Free, Free>, String> {
    wg.ds_load_cur(&ring, LOAD.into())?;
    let (ring,) = wg.barrier((retire_cur(ring),))?;
    Ok(ring)
}

fn pure_typed(b: &mut Trace) -> Result<(), String> {
    let mut wg = Wg::new(b)?;
    let k0 = wg.lds::<Tile>("K0", 0, 4096)?;
    let k1 = wg.lds::<Tile>("K1", 4096, 4096)?;
    let (ring, st) = wg.ds_store(Ring::new(k0, k1), STAGE.into())?;
    let drained = wg.wait(st)?;
    let (ring,) = wg.barrier((prime(ring, drained),))?;
    let ring = middle(&mut wg, ring)?;
    tail(&mut wg, ring)?;
    Ok(())
}

/// Declare and fill the ring in the driver, run `middle` inside `typed`, end in
/// the driver.
fn split(b: &mut Trace) -> Result<(), String> {
    let mut d = Driver::<Gfx1201, Trace>::new(b)?;
    let id = d.typed(|wg, lend| {
        let k0 = wg.lds::<Tile>("K0", 0, 4096)?;
        let k1 = wg.lds::<Tile>("K1", 4096, 4096)?;
        Ok(lend.give(Ring::new(k0, k1), "K"))
    })?;
    d.ds_store(id, STAGE.into())?;
    d.wait(id)?;
    d.barrier(&[Transition::Prime(id)])?;
    d.typed(|wg, lend| {
        let ring = lend.take::<Tile, Published, Free, 2>(id)?;
        lend.give_back(id, middle(wg, ring)?)
    })?;
    d.ds_load_cur(id, LOAD.into())?;
    d.barrier(&[Transition::RetireCur(id)])?;
    Ok(())
}

#[test]
fn a_typed_middle_emits_the_pure_typed_kernels_trace_exactly() {
    let (mut typed, mut mixed) = (Trace::new("gfx1201"), Trace::new("gfx1201"));
    pure_typed(&mut typed).unwrap();
    split(&mut mixed).unwrap();
    assert_eq!(typed.text, mixed.text);
    // One rotate barrier and one prime: the split barriers of gfx12.
    assert_eq!(mixed.count("s_barrier_signal"), 3);
}

#[test]
fn ring_internals_round_trip_exactly() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let id = d
        .typed(|wg, lend| {
            let k0 = wg.lds::<Tile>("K0", 0, 4096)?;
            let k1 = wg.lds::<Tile>("K1", 4096, 4096)?;
            Ok(lend.give(Ring::new(k0, k1), "K"))
        })
        .unwrap();
    // A fresh ring: cur = 1, nothing published.
    assert_eq!(d.cur_index(id).unwrap(), 1);
    // A ring that crosses and returns unchanged keeps every internal.
    let (cur, next) = (d.cur_index(id).unwrap(), d.next_index(id).unwrap());
    d.typed(|_, lend| {
        let ring = lend.take::<Tile, Free, Free, 2>(id)?;
        assert_eq!((ring.cur_index(), ring.next_index()), (cur, next));
        lend.give_back(id, ring)
    })
    .unwrap();
    assert_eq!(
        (d.cur_index(id).unwrap(), d.next_index(id).unwrap()),
        (cur, next)
    );
    // After a typed rotate the driver agrees with the typed ring's `cur_index`.
    d.ds_store(id, STAGE.into()).unwrap();
    d.wait(id).unwrap();
    d.barrier(&[Transition::Prime(id)]).unwrap();
    let typed_cur = d
        .typed(|wg, lend| {
            let ring = middle(wg, lend.take::<Tile, Published, Free, 2>(id)?)?;
            let cur = ring.cur_index();
            lend.give_back(id, ring)?;
            Ok(cur)
        })
        .unwrap();
    assert_eq!(d.cur_index(id).unwrap(), typed_cur);
    // And it is readable again: `cur_live` survived.
    d.ds_load_cur(id, LOAD.into()).unwrap();
}

/// A driver whose ring `K` is `Published`/`Free` after a prime.
fn primed(
    b: &mut Trace,
) -> (
    Driver<'_, Gfx1201, Trace>,
    peacemaker_author::runtime::RingId,
) {
    let mut d = Driver::<Gfx1201, Trace>::new(b).unwrap();
    let id = d
        .typed(|wg, lend| {
            let k0 = wg.lds::<Tile>("K0", 0, 4096)?;
            let k1 = wg.lds::<Tile>("K1", 4096, 4096)?;
            Ok(lend.give(Ring::new(k0, k1), "K"))
        })
        .unwrap();
    d.ds_store(id, STAGE.into()).unwrap();
    d.wait(id).unwrap();
    d.barrier(&[Transition::Prime(id)]).unwrap();
    (d, id)
}

#[test]
fn take_in_the_wrong_phase_is_refused() {
    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    let e = err(d.typed(|_, lend| lend.take::<Tile, Free, Free, 2>(id).map(|_| ())));
    assert!(
        e.contains("[cur Published, next Free]") && e.contains("takes it as [cur Free, next Free]"),
        "{e}"
    );
}

#[test]
fn take_under_the_wrong_tag_is_refused() {
    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    let e = err(d.typed(|_, lend| lend.take::<Other, Published, Free, 2>(id).map(|_| ())));
    assert!(e.contains("different tag"), "{e}");
}

#[test]
fn a_script_made_ring_cannot_be_lent() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let (r0, r1) = (
        d.lds("R0", 0, 4096).unwrap(),
        d.lds("R1", 4096, 4096).unwrap(),
    );
    let ring = d.ring("R", r0, r1).unwrap();
    let e = err(d.typed(|_, lend| lend.take::<Tile, Free, Free, 2>(ring).map(|_| ())));
    assert!(e.contains("was not given by a typed body"), "{e}");
}

#[test]
fn a_ring_with_a_pending_token_is_refused() {
    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    // The next buffer is `Writing` with an undrained store.
    d.ds_store(id, STAGE.into()).unwrap();
    let e = err(d.typed(|_, lend| lend.take::<Tile, Published, Free, 2>(id).map(|_| ())));
    assert!(e.contains("Pending store"), "{e}");
    // Draining does not make a `Writing` ring lendable: it still is not `Free`.
    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    d.ds_store(id, STAGE.into()).unwrap();
    d.wait(id).unwrap();
    let e = err(d.typed(|_, lend| lend.take::<Tile, Published, Free, 2>(id).map(|_| ())));
    assert!(e.contains("next Writing"), "{e}");
}

#[test]
fn a_ring_taken_twice_or_not_returned_is_refused() {
    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    let e = err(d.typed(|_, lend| {
        let _a = lend.take::<Tile, Published, Free, 2>(id)?;
        lend.take::<Tile, Published, Free, 2>(id).map(|_| ())
    }));
    assert!(e.contains("already lent"), "{e}");

    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    let e = err(d.typed(|_, lend| lend.take::<Tile, Published, Free, 2>(id).map(|_| ())));
    // The error is the body's own `Ok(())`: the driver refuses to close the
    // typed scope with the ring still out.
    assert!(e.contains("still taken"), "{e}");
    // The lowering is dead: the ring stays lent, and the driver says so.
    let e = err(d.ds_load_cur(id, LOAD.into()));
    assert!(e.contains("lent to a typed body"), "{e}");
}

#[test]
fn a_ring_given_back_must_be_the_ring_taken() {
    let mut b = Trace::new("gfx1201");
    let (mut d, a) = primed(&mut b);
    // A second ring of the same tag over other slots.
    let other = d
        .typed(|wg, lend| {
            let k0 = wg.lds::<Tile>("L0", 8192, 4096)?;
            let k1 = wg.lds::<Tile>("L1", 12288, 4096)?;
            Ok(lend.give(Ring::new(k0, k1), "L"))
        })
        .unwrap();
    d.ds_store(other, STAGE.into()).unwrap();
    d.wait(other).unwrap();
    d.barrier(&[Transition::Prime(other)]).unwrap();
    let e = err(d.typed(|_, lend| {
        let ra = lend.take::<Tile, Published, Free, 2>(a)?;
        let rb = lend.take::<Tile, Published, Free, 2>(other)?;
        lend.give_back(a, rb)?;
        lend.give_back(other, ra)
    }));
    assert!(e.contains("not the one that was taken"), "{e}");
}

#[test]
fn typed_is_refused_under_wave_scope_and_after_a_wave_exit() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let c = d.scmp(CMP.into()).unwrap();
    let e = err(d.skip_if(c, ".Lskip", |d| d.typed(|_, _| Ok(()))));
    assert!(e.contains("typed is workgroup scope"), "{e}");

    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let end = d.exit(".Lend").unwrap();
    let c = d.scmp(CMP.into()).unwrap();
    d.wave_exit_unless(c, &end).unwrap();
    let e = err(d.typed(|_, _| Ok(())));
    assert!(e.contains("after a wave-uniform kernel exit"), "{e}");
}

#[test]
fn a_typed_wave_exit_is_recorded_by_the_driver() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let end = d.exit(".Lend").unwrap();
    d.typed(|wg, lend| {
        let tend = lend.end(&end)?;
        let c = wg.scmp(CMP.into())?;
        // The wave-scope form (`Workgroup::exit_unless` is the workgroup one).
        Wave::exit_unless(wg, c, &tend)
    })
    .unwrap();
    let e = err(d.typed(|_, _| Ok(())));
    assert!(e.contains("after a wave-uniform kernel exit"), "{e}");
}

#[test]
fn an_exit_viewed_through_lend_shares_its_branch_flag() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let end = d.exit(".Lend").unwrap();
    d.typed(|wg, lend| {
        let tend = lend.end(&end)?;
        assert_eq!(tend.label(), ".Lend");
        let c = wg.scmp_wg_uniform(CMP.into())?;
        wg.exit_if(c, &tend)
    })
    .unwrap();
    // A branch reaches the exit, so it takes no tail (the driver sees the flag
    // the typed branch set).
    let e = err(d.end_with(end, |d| d.isa().raw("s_nop 0")));
    assert!(e.contains("takes no tail"), "{e}");

    // An exit nothing branched to still takes one.
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let end = d.exit(".Lend").unwrap();
    d.typed(|_, lend| lend.end(&end).map(|_| ())).unwrap();
    d.end_with(end, |d| d.isa().raw("s_nop 0")).unwrap();
}

#[test]
fn an_exit_or_ring_of_another_driver_is_refused() {
    let (mut a, mut b) = (Trace::new("gfx1201"), Trace::new("gfx1201"));
    let mut da = Driver::<Gfx1201, Trace>::new(&mut a).unwrap();
    let mut db = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let foreign = db.exit(".Lend").unwrap();
    let e = err(da.typed(|_, lend| lend.end(&foreign).map(|_| ())));
    assert!(e.contains("belongs to another Driver"), "{e}");
    let id = da
        .typed(|wg, lend| {
            let k0 = wg.lds::<Tile>("K0", 0, 4096)?;
            let k1 = wg.lds::<Tile>("K1", 4096, 4096)?;
            Ok(lend.give(Ring::new(k0, k1), "K"))
        })
        .unwrap();
    let e = err(db.typed(|_, lend| lend.take::<Tile, Free, Free, 2>(id).map(|_| ())));
    assert!(e.contains("another Driver"), "{e}");
}

#[test]
fn a_typed_condition_is_a_driver_condition_that_goes_stale() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let end = d.exit(".Lend").unwrap();
    let c = d
        .typed(|wg, lend| Ok(lend.wg_cond(wg.scmp_wg_uniform(CMP.into())?)))
        .unwrap();
    d.exit_if(c, &end).unwrap();
    d.end(end).unwrap();

    // SCC redefined between the compare and the branch: refused like any cond.
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let end = d.exit(".Lend").unwrap();
    let c = d
        .typed(|wg, lend| Ok(lend.wg_cond(wg.scmp_wg_uniform(CMP.into())?)))
        .unwrap();
    let _ = d.scmp(CMP.into()).unwrap();
    let e = err(d.exit_if(c, &end));
    assert!(e.contains("SCC was redefined"), "{e}");
}

/// The script loop of `loop_carried` around a typed block that rotates the
/// ring: the driver's invariance rule covers it with no extra mechanism, and
/// the loop emits the pure typed loop's trace.
#[test]
fn a_script_loop_around_a_typed_block_passes_the_loop_check() {
    let mut mixed = Trace::new("gfx1201");
    {
        let (mut d, id) = primed(&mut mixed);
        d.loop_carried(".Lk", |d| {
            d.typed(|wg, lend| {
                let ring = lend.take::<Tile, Published, Free, 2>(id)?;
                lend.give_back(id, middle(wg, ring)?)
            })?;
            d.scmp_wg_uniform(CMP.into())
        })
        .unwrap();
    }
    let mut typed = Trace::new("gfx1201");
    {
        let mut wg = Wg::new(&mut typed).unwrap();
        let k0 = wg.lds::<Tile>("K0", 0, 4096).unwrap();
        let k1 = wg.lds::<Tile>("K1", 4096, 4096).unwrap();
        let (ring, st) = wg.ds_store(Ring::new(k0, k1), STAGE.into()).unwrap();
        let drained = wg.wait(st).unwrap();
        let (ring,) = wg.barrier((prime(ring, drained),)).unwrap();
        wg.loop_carried(".Lk", ring, |wg, ring| {
            let ring = middle(wg, ring)?;
            Ok((ring, wg.scmp_wg_uniform(CMP.into())?))
        })
        .unwrap();
    }
    assert_eq!(typed.text, mixed.text);
}

#[test]
fn a_typed_block_that_leaves_the_ring_in_another_state_is_refused_by_the_loop() {
    let mut b = Trace::new("gfx1201");
    let (mut d, id) = primed(&mut b);
    let e = err(d.loop_carried(".Lk", |d| {
        d.typed(|wg, lend| {
            let ring = lend.take::<Tile, Published, Free, 2>(id)?;
            // Retires the published buffer: the ring leaves `Free`/`Free`.
            lend.give_back(id, tail(wg, ring)?)
        })?;
        d.scmp_wg_uniform(CMP.into())
    }));
    assert!(
        e.contains("differs between the entry and the back edge"),
        "{e}"
    );
}

// ---- a ring that crosses with a pending `Writing` buffer ---------------------

fn primed_ring(d: &mut Driver<'_, Gfx1201, Trace>) -> peacemaker_author::runtime::RingId {
    let id = d
        .typed(|wg, lend| {
            let k0 = wg.lds::<Tile>("K0", 0, 4096)?;
            let k1 = wg.lds::<Tile>("K1", 4096, 4096)?;
            Ok(lend.give(Ring::new(k0, k1), "K"))
        })
        .unwrap();
    d.ds_store(id, STAGE.into()).unwrap();
    d.wait(id).unwrap();
    d.barrier(&[Transition::Prime(id)]).unwrap();
    id
}

/// `middle` split at its wait: the typed body stages the next buffer and hands
/// the ring back `Writing` with its token pending; the driver waits and rotates.
/// The trace is the pure typed `middle`'s.
#[test]
fn a_publish_split_from_its_wait_emits_the_same_trace() {
    let mut whole = Trace::new("gfx1201");
    {
        let mut d = Driver::<Gfx1201, Trace>::new(&mut whole).unwrap();
        let id = primed_ring(&mut d);
        d.typed(|wg, lend| {
            let ring = lend.take::<Tile, Published, Free, 2>(id)?;
            lend.give_back(id, middle(wg, ring)?)
        })
        .unwrap();
    }
    let mut split = Trace::new("gfx1201");
    {
        let mut d = Driver::<Gfx1201, Trace>::new(&mut split).unwrap();
        let id = primed_ring(&mut d);
        d.typed(|wg, lend| {
            let ring = lend.take::<Tile, Published, Free, 2>(id)?;
            wg.ds_load_cur(&ring, LOAD.into())?;
            lend.give_writing(id, wg.ds_store(ring, STAGE.into())?)
        })
        .unwrap();
        // Writing, pending: the driver knows it, and a barrier needs the wait.
        let v = d.typed(|_, lend| lend.view::<Tile>(id)).unwrap();
        assert!(v.pending);
        let e = err(d.barrier(&[Transition::Rotate(id)]));
        assert!(e.contains("undrained LDS store"), "{e}");
        d.wait(id).unwrap();
        d.barrier(&[Transition::Rotate(id)]).unwrap();
    }
    assert_eq!(whole.text, split.text);
}

/// A body that only reads a `Writing` ring gives it back as it was: still
/// pending before the wait, still drained after it.
#[test]
fn reading_a_writing_ring_keeps_its_token_state() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let id = primed_ring(&mut d);
    d.typed(|wg, lend| {
        let ring = lend.take::<Tile, Published, Free, 2>(id)?;
        lend.give_writing(id, wg.ds_store(ring, STAGE.into())?)
    })
    .unwrap();
    let read = |d: &mut Driver<'_, Gfx1201, Trace>| {
        d.typed(|wg, lend| {
            let (ring, tok) =
                lend.take_writing::<Tile, peacemaker_author::Gfx12Waits, Published, 2>(id)?;
            wg.ds_load_cur(&ring, LOAD.into())?;
            lend.give_writing(id, (ring, tok))
        })
        .unwrap()
    };
    read(&mut d);
    assert!(d.typed(|_, lend| lend.view::<Tile>(id)).unwrap().pending);
    d.wait(id).unwrap();
    read(&mut d);
    assert!(!d.typed(|_, lend| lend.view::<Tile>(id)).unwrap().pending);
    // Drained as it was: the rotate goes through without another wait.
    d.barrier(&[Transition::Rotate(id)]).unwrap();
}

/// The `Free` take still refuses a `Writing` ring, and `take_writing` refuses a
/// ring whose next buffer is `Free`.
#[test]
fn the_writing_lend_checks_the_next_buffer() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let id = primed_ring(&mut d);
    let e = err(d.typed(|_, lend| {
        lend.take_writing::<Tile, peacemaker_author::Gfx12Waits, Published, 2>(id)
            .map(|_| ())
    }));
    assert!(e.contains("next Writing"), "{e}");
    d.typed(|wg, lend| {
        let ring = lend.take::<Tile, Published, Free, 2>(id)?;
        lend.give_writing(id, wg.ds_store(ring, STAGE.into())?)
    })
    .unwrap();
    let e = err(d.typed(|_, lend| lend.take::<Tile, Published, Free, 2>(id).map(|_| ())));
    assert!(e.contains("Pending") || e.contains("next"), "{e}");
    let e = err(d.typed(|_, lend| lend.view::<Other>(id).map(|_| ())));
    assert!(e.contains("tag"), "{e}");
}
