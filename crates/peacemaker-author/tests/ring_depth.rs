// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! N-slot rings (depth 2 to 4): typed `Ring<R, C, N, D>` and the runtime
//! `Driver::ring_n` fill, rotate and advance over a `D`-buffer ring,
//! `starting_at`, `loop_carried` over a period of `D` rotations, `Lend`
//! carrying the depth, and `wg_if_else` (typed and runtime).
use peacemaker_author::runtime::{Driver, Transition};
use peacemaker_author::trace::Trace;
use peacemaker_author::{
    advance, prime, retire_cur, rotate, Free, Gfx1151, Published, Ring, Workgroup,
};

enum Tile {}

const STAGE: &str = "ds_store_b128 v1, v[2:5]";
const CMP: &str = "s_cmp_lg_u32 s4, 0";

fn err<T>(r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e,
    }
}

/// A load whose text names the physical buffer, so a body that does not
/// return to the buffer it started on re-emits different code.
fn load(cur: usize) -> String {
    format!("ds_load_b128 v[8:11], v6 offset:{}", cur * 4096)
}

type Wg<'a> = Workgroup<'a, Gfx1151, Trace>;
type Ring3<C, N> = Ring<Tile, C, N, 3>;

fn regions<const D: usize>(
    wg: &mut Wg,
) -> Result<[peacemaker_author::LdsRegion<Tile, Free>; D], String> {
    let mut v = Vec::new();
    for i in 0..D {
        v.push(wg.lds::<Tile>(&format!("K{i}"), (i as u32) * 4096, 4096)?);
    }
    v.try_into().map_err(|_| "region count".to_string())
}

/// Store into the ring's next buffer, drain it and publish with `prime`.
fn prime_one<const D: usize>(
    wg: &mut Wg,
    ring: Ring<Tile, Free, Free, D>,
) -> Result<Ring<Tile, Published, Free, D>, String> {
    let (ring, st) = wg.ds_store(ring, STAGE.into())?;
    let drained = wg.wait(st)?;
    let (ring,) = wg.barrier((prime(ring, drained),))?;
    Ok(ring)
}

/// Store into the next buffer, drain it and rotate.
fn rotate_one<const D: usize>(
    wg: &mut Wg,
    ring: Ring<Tile, Published, Free, D>,
) -> Result<Ring<Tile, Published, Free, D>, String> {
    let (ring, st) = wg.ds_store(ring, STAGE.into())?;
    let drained = wg.wait(st)?;
    let (ring,) = wg.barrier((rotate(ring, drained),))?;
    Ok(ring)
}

/// A depth-3 ring filled by one `prime` and one `rotate` over the vacant
/// current buffer: slab 0 current, slab 1 ahead, the third buffer next.
fn filled3(wg: &mut Wg) -> Result<Ring3<Published, Free>, String> {
    let ring = Ring::<Tile, Free, Free, 3>::new_n(regions::<3>(wg)?)?;
    let ring = prime_one(wg, ring)?;
    rotate_one(wg, ring)
}

// ---- typed ------------------------------------------------------------

#[test]
fn depth3_fill_primes_then_rotates_over_the_vacant_cur() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let ring = Ring::<Tile, Free, Free, 3>::new_n(regions::<3>(&mut wg).unwrap()).unwrap();
    assert_eq!(
        (ring.depth(), ring.cur_index(), ring.next_index()),
        (3, 1, 0)
    );
    // First publication: buffer 0 holds data, the current buffer is vacant.
    let ring = prime_one(&mut wg, ring).unwrap();
    assert_eq!((ring.cur_index(), ring.next_index()), (2, 1));
    // Rotating over the vacant current buffer retires nothing (the trace
    // backend would refuse a retire of a slot no barrier published).
    let ring = rotate_one(&mut wg, ring).unwrap();
    assert_eq!((ring.cur_index(), ring.next_index()), (0, 2));
    wg.ds_load_cur(&ring, load(ring.cur_index()).into())
        .unwrap();
    drop(b);
}

#[test]
fn depth3_steady_rotation_visits_every_buffer() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let mut ring = filled3(&mut wg).unwrap();
    let mut seen = vec![(ring.cur_index(), ring.next_index())];
    for _ in 0..3 {
        wg.ds_load_cur(&ring, load(ring.cur_index()).into())
            .unwrap();
        ring = rotate_one(&mut wg, ring).unwrap();
        seen.push((ring.cur_index(), ring.next_index()));
    }
    assert_eq!(seen, [(0, 2), (1, 0), (2, 1), (0, 2)]);
}

#[test]
fn a_vacant_cur_read_is_refused() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let ring = Ring::<Tile, Free, Free, 3>::new_n(regions::<3>(&mut wg).unwrap()).unwrap();
    let ring = prime_one(&mut wg, ring).unwrap();
    // `prime` of a fresh ring leaves `cur` vacant for depth 3.
    let e = err(wg.ds_load_cur(&ring, load(ring.cur_index()).into()));
    assert!(e.contains("no barrier has published"), "{e}");
    // The first rotation over it publishes and makes it readable.
    let ring = rotate_one(&mut wg, ring).unwrap();
    wg.ds_load_cur(&ring, load(ring.cur_index()).into())
        .unwrap();
}

/// A body of three rotations returns the ring to the buffer it started on;
/// one rotation re-emits different code on the second pass.
fn rotate_loop(trip: usize) -> Result<(), String> {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b)?;
    let ring = filled3(&mut wg)?;
    wg.loop_carried(".Lk", ring, |wg, mut ring| {
        for _ in 0..trip {
            wg.ds_load_cur(&ring, load(ring.cur_index()).into())?;
            ring = rotate_one(wg, ring)?;
        }
        Ok((ring, wg.scmp_wg_uniform(CMP.into())?))
    })
    .map(drop)
}

#[test]
fn loop_carried_needs_a_whole_period_of_rotations() {
    rotate_loop(3).unwrap();
    rotate_loop(6).unwrap();
    for trip in [1, 2, 4] {
        let e = err(rotate_loop(trip));
        assert!(
            e.contains("did not reach a fixed point"),
            "trip {trip}: {e}"
        );
    }
}

#[test]
fn starting_at_picks_the_first_store_buffer() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    for k in 0..3 {
        let regs = regions_named(&mut wg, k).unwrap();
        let ring = Ring::<Tile, Free, Free, 3>::new_n(regs).unwrap();
        let ring = ring.starting_at(k).unwrap();
        assert_eq!(ring.next_index(), k);
        assert_eq!(ring.cur_index(), (k + 1) % 3);
    }
    let regs = regions_named(&mut wg, 9).unwrap();
    let ring = Ring::<Tile, Free, Free, 3>::new_n(regs).unwrap();
    let e = err(ring.starting_at(3));
    assert!(e.contains("outside a depth-3 ring"), "{e}");
}

fn regions_named(
    wg: &mut Wg,
    tag: usize,
) -> Result<[peacemaker_author::LdsRegion<Tile, Free>; 3], String> {
    let mut v = Vec::new();
    for i in 0..3 {
        let base = ((tag * 3 + i) * 64) as u32;
        v.push(wg.lds::<Tile>(&format!("S{tag}.{i}"), base, 64)?);
    }
    v.try_into().map_err(|_| "region count".to_string())
}

#[test]
fn starting_at_is_refused_once_anything_was_published() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let ring = Ring::<Tile, Free, Free, 2>::new_n(regions::<2>(&mut wg).unwrap()).unwrap();
    let ring = prime_one(&mut wg, ring).unwrap();
    wg.ds_load_cur(&ring, load(ring.cur_index()).into())
        .unwrap();
    let (ring,) = wg.barrier((retire_cur(ring),)).unwrap();
    let e = err(ring.starting_at(0));
    assert!(e.contains("before anything is published"), "{e}");
}

#[test]
fn ring_depth_is_bounded_to_two_through_four() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let five = (0..5)
        .map(|i| wg.lds::<Tile>(&format!("B{i}"), 0, 64).unwrap())
        .collect::<Vec<_>>();
    let five: [_; 5] = five.try_into().unwrap_or_else(|_| unreachable!());
    let e = err(Ring::<Tile, Free, Free, 5>::new_n(five));
    assert!(e.contains("2 to 4 buffers, not 5"), "{e}");
    let one = [wg.lds::<Tile>("O", 0, 64).unwrap()];
    let e = err(Ring::<Tile, Free, Free, 1>::new_n(one));
    assert!(e.contains("2 to 4 buffers, not 1"), "{e}");
}

#[test]
fn advance_moves_to_the_published_ahead_buffer_without_a_store() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let ring = filled3(&mut wg).unwrap();
    assert_eq!(ring.cur_index(), 0);
    wg.ds_load_cur(&ring, load(ring.cur_index()).into())
        .unwrap();
    let (ring,) = wg.barrier((advance(ring),)).unwrap();
    assert_eq!((ring.cur_index(), ring.next_index()), (1, 0));
    wg.ds_load_cur(&ring, load(ring.cur_index()).into())
        .unwrap();
    // Nothing is published after buffer 1 any more.
    let e = err(wg.barrier((advance(ring),)));
    assert!(
        e.contains("needs a published buffer after the current one"),
        "{e}"
    );
}

#[test]
fn advance_of_a_depth_2_ring_is_always_refused() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Wg::new(&mut b).unwrap();
    let ring = Ring::<Tile, Free, Free, 2>::new_n(regions::<2>(&mut wg).unwrap()).unwrap();
    let ring = prime_one(&mut wg, ring).unwrap();
    let e = err(wg.barrier((advance(ring),)));
    assert!(e.contains("depth-2"), "{e}");
}

// ---- typed wg_if_else ---------------------------------------------------

/// Both arms fill the ring from `first`: they end equal only if they start
/// on the same buffer.
fn if_else_typed(then_start: usize, else_start: usize) -> Result<Trace, String> {
    let mut b = Trace::new("gfx1151");
    {
        let mut wg = Wg::new(&mut b)?;
        let ring = Ring::<Tile, Free, Free, 3>::new_n(regions::<3>(&mut wg)?)?;
        let cond = wg.scmp_wg_uniform(CMP.into())?;
        let fill = |start: usize| {
            move |wg: &mut Wg, ring: Ring<Tile, Free, Free, 3>| {
                let ring = ring.starting_at(start)?;
                prime_one(wg, ring)
            }
        };
        let ring = wg.wg_if_else(
            cond,
            ".Lelse",
            ".Ljoin",
            ring,
            fill(then_start),
            fill(else_start),
        )?;
        drop(ring);
    }
    Ok(b)
}

#[test]
fn wg_if_else_accepts_equal_arms_and_places_the_join() {
    let b = if_else_typed(1, 1).unwrap();
    let want = [
        CMP,
        "s_cbranch_scc1 .Lelse",
        STAGE,
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        "s_branch .Ljoin",
        ".Lelse:",
        STAGE,
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        ".Ljoin:",
    ];
    assert_eq!(b.text, want);
}

#[test]
fn wg_if_else_refuses_arms_that_end_on_different_buffers() {
    let e = err(if_else_typed(0, 1));
    assert!(
        e.contains("arms end in different ring or region values"),
        "{e}"
    );
}

// ---- runtime driver -----------------------------------------------------

fn driver_ring3(d: &mut Driver<'_, Gfx1151, Trace>) -> peacemaker_author::runtime::RingId {
    let a = d.lds("K0", 0, 4096).unwrap();
    let b2 = d.lds("K1", 4096, 4096).unwrap();
    let c = d.lds("K2", 8192, 4096).unwrap();
    d.ring_n("K", &[a, b2, c]).unwrap()
}

fn fill_step(
    d: &mut Driver<'_, Gfx1151, Trace>,
    g: peacemaker_author::runtime::RingId,
    t: Transition,
) {
    d.ds_store(g, STAGE.into()).unwrap();
    d.wait(g).unwrap();
    d.barrier(&[t]).unwrap();
}

#[test]
fn runtime_depth3_fill_steady_rotation_and_vacant_read() {
    let mut b = Trace::new("gfx1151");
    let mut d = Driver::<Gfx1151, Trace>::new(&mut b).unwrap();
    let g = driver_ring3(&mut d);
    assert_eq!(d.ring_depth(g).unwrap(), 3);
    assert_eq!((d.cur_index(g).unwrap(), d.next_index(g).unwrap()), (1, 0));
    fill_step(&mut d, g, Transition::Prime(g));
    assert_eq!((d.cur_index(g).unwrap(), d.next_index(g).unwrap()), (2, 1));
    let e = err(d.ds_load_cur(g, load(2).into()));
    assert!(e.contains("no barrier has published"), "{e}");
    fill_step(&mut d, g, Transition::Rotate(g));
    let mut seen = vec![(d.cur_index(g).unwrap(), d.next_index(g).unwrap())];
    for _ in 0..3 {
        let cur = d.cur_index(g).unwrap();
        d.ds_load_cur(g, load(cur).into()).unwrap();
        fill_step(&mut d, g, Transition::Rotate(g));
        seen.push((d.cur_index(g).unwrap(), d.next_index(g).unwrap()));
    }
    assert_eq!(seen, [(0, 2), (1, 0), (2, 1), (0, 2)]);
}

#[test]
fn runtime_loop_carried_needs_three_rotations_per_trip() {
    let run = |trip: usize| -> Result<(), String> {
        let mut b = Trace::new("gfx1151");
        let mut d = Driver::<Gfx1151, Trace>::new(&mut b)?;
        let g = driver_ring3(&mut d);
        fill_step(&mut d, g, Transition::Prime(g));
        fill_step(&mut d, g, Transition::Rotate(g));
        d.loop_carried(".Lk", |d| {
            for _ in 0..trip {
                let cur = d.cur_index(g)?;
                d.ds_load_cur(g, load(cur).into())?;
                d.ds_store(g, STAGE.into())?;
                d.wait(g)?;
                d.barrier(&[Transition::Rotate(g)])?;
            }
            d.scmp_wg_uniform(CMP.into())
        })
    };
    run(3).unwrap();
    let e = err(run(1));
    assert!(e.contains("did not reach a fixed point"), "{e}");
}

#[test]
fn runtime_start_at_and_advance() {
    let mut b = Trace::new("gfx1151");
    let mut d = Driver::<Gfx1151, Trace>::new(&mut b).unwrap();
    let g = driver_ring3(&mut d);
    d.ring_start_at(g, 2).unwrap();
    assert_eq!((d.cur_index(g).unwrap(), d.next_index(g).unwrap()), (0, 2));
    let e = err(d.ring_start_at(g, 3));
    assert!(e.contains("outside a depth-3 ring"), "{e}");
    fill_step(&mut d, g, Transition::Prime(g));
    fill_step(&mut d, g, Transition::Rotate(g));
    // buffer 2 then buffer 0 were published; cur = 2 (read before retiring).
    assert_eq!(d.cur_index(g).unwrap(), 2);
    let e = err(d.ring_start_at(g, 0));
    assert!(e.contains("before anything is published"), "{e}");
    d.ds_load_cur(g, load(2).into()).unwrap();
    d.barrier(&[Transition::Advance(g)]).unwrap();
    assert_eq!(d.cur_index(g).unwrap(), 0);
    let e = err(d.barrier(&[Transition::Advance(g)]));
    assert!(
        e.contains("no published buffer after the current one"),
        "{e}"
    );
}

fn if_else_runtime(then_start: usize, else_start: usize) -> Result<(), String> {
    let mut b = Trace::new("gfx1151");
    let mut d = Driver::<Gfx1151, Trace>::new(&mut b)?;
    let g = driver_ring3(&mut d);
    let cond = d.scmp_wg_uniform(CMP.into())?;
    d.wg_if_else(
        cond,
        ".Lelse",
        ".Ljoin",
        |d| {
            d.ring_start_at(g, then_start)?;
            fill_step_r(d, g)
        },
        |d| {
            d.ring_start_at(g, else_start)?;
            fill_step_r(d, g)
        },
    )
}

fn fill_step_r(
    d: &mut Driver<'_, Gfx1151, Trace>,
    g: peacemaker_author::runtime::RingId,
) -> Result<(), String> {
    d.ds_store(g, STAGE.into())?;
    d.wait(g)?;
    d.barrier(&[Transition::Prime(g)])
}

#[test]
fn runtime_wg_if_else_accepts_equal_arms_and_refuses_unequal_ones() {
    if_else_runtime(1, 1).unwrap();
    let e = err(if_else_runtime(0, 1));
    assert!(e.contains("arms end in different LDS states"), "{e}");
}

// ---- Lend carries the depth --------------------------------------------

#[test]
fn lend_round_trips_depth3_and_refuses_a_depth_mismatch() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<peacemaker_author::Gfx1201, Trace>::new(&mut b).unwrap();
    let id = d
        .typed(|wg, lend| {
            let regs = {
                let mut v = Vec::new();
                for i in 0..3 {
                    v.push(wg.lds::<Tile>(&format!("K{i}"), (i as u32) * 4096, 4096)?);
                }
                v
            };
            let regs: [_; 3] = regs.try_into().unwrap_or_else(|_| unreachable!());
            Ok(lend.give(Ring::<Tile, Free, Free, 3>::new_n(regs)?, "K"))
        })
        .unwrap();
    assert_eq!(d.ring_depth(id).unwrap(), 3);
    assert_eq!(d.typed(|_, lend| lend.view::<Tile>(id)).unwrap().depth, 3);
    // A typed body that takes it as a depth-2 ring is refused, naming both.
    let e = err(d.typed(|_, lend| lend.take::<Tile, Free, Free, 2>(id).map(|_| ())));
    assert!(e.contains("depth 3") && e.contains("depth 2"), "{e}");
    // Depth 3 round trip: take, fill it typed, give it back.
    d.typed(|wg, lend| {
        let ring = lend.take::<Tile, Free, Free, 3>(id)?;
        let (ring, st) = wg.ds_store(ring, STAGE.into())?;
        let drained = wg.wait(st)?;
        let (ring,) = wg.barrier((prime(ring, drained),))?;
        lend.give_back(id, ring)
    })
    .unwrap();
    assert_eq!(
        d.ring_phases(id).unwrap().0,
        peacemaker_author::runtime::Phase::RtPublished
    );
    // The driver now holds the typed ring's cur / live buffers.
    assert_eq!(
        (d.cur_index(id).unwrap(), d.next_index(id).unwrap()),
        (2, 1)
    );
    let e = err(d.ds_load_cur(id, load(2).into()));
    assert!(e.contains("no barrier has published"), "{e}");
    // Giving back a ring of another depth under the same id is refused.
    let e = err(d.typed(|wg, lend| {
        let _ = lend.take::<Tile, Published, Free, 3>(id)?;
        let a = wg.lds::<Tile>("X0", 12288, 64)?;
        let c = wg.lds::<Tile>("X1", 12352, 64)?;
        let two = Ring::<Tile, Free, Free, 2>::new(a, c).into_steady();
        lend.give_back(id, two)
    }));
    assert!(e.contains("depth 2") && e.contains("depth 3"), "{e}");
}
