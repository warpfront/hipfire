//! The runtime-checked driver (`runtime::Driver`) against the typed core:
//! the two shipped LDS races are rejected dynamically (twins of
//! `tests/ui/halo_verify_attn_race.rs` and `fa2_mailbox_race.rs`), their
//! fixes re-emit the typed twin's exact trace, and illegal transitions are
//! refused before anything reaches the backend.
use peacemaker_author::runtime::{Driver, Phase, Transition};
use peacemaker_author::trace::Trace;
use peacemaker_author::{join, ready, retire, rotate, Gfx1100, Gfx1151, Gfx1201, Ring, Workgroup};

enum KTile {}
enum VBody {}
enum Mailbox {}
enum VPlane {}
enum Gate {}

const STAGE: &str = "ds_store_b128 v1, v[2:5]";
const LOAD: &str = "ds_load_b128 v[8:11], v6";
const CMP: &str = "s_cmp_lg_u32 s4, 0";

fn err<T>(r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("expected the driver to refuse"),
        Err(e) => e,
    }
}

// ---- Halo VerifyAttn ------------------------------------------------------

/// Runtime twin of `tests/ui/halo_verify_attn_race.rs` (`drain` = false) and
/// `halo_verify_attn_fixed.rs` (`drain` = true).
fn k_tile_loop(b: &mut Trace, drain: bool) -> Result<(), String> {
    let mut d = Driver::<Gfx1151, Trace>::new(b)?;
    let k0 = d.lds("K0", 0, 8704)?;
    let k1 = d.lds("K1", 8704, 8704)?;
    let ring = d.ring("K", k0, k1)?;
    d.ds_store(ring, STAGE.into())?;
    d.ring_steady(ring)?;
    d.loop_carried(".Lk_tile", |d| {
        if drain {
            d.wait(ring)?;
        }
        d.barrier(&[Transition::Rotate(ring)])?;
        d.ds_load_cur(ring, LOAD.into())?;
        d.ds_store(ring, STAGE.into())?;
        d.scmp_wg_uniform(CMP.into())
    })
}

fn k_tile_loop_typed(b: &mut Trace) {
    let mut wg = Workgroup::<Gfx1151, Trace>::new(b).unwrap();
    let k0 = wg.lds::<KTile>("K0", 0, 8704).unwrap();
    let k1 = wg.lds::<KTile>("K1", 8704, 8704).unwrap();
    let (ring, st) = wg.ds_store(Ring::new(k0, k1), STAGE.into()).unwrap();
    let _ = wg.loop_carried(".Lk_tile", (ring.into_steady(), st), |wg, (ring, st)| {
        let drained = wg.wait(st)?;
        let (ring,) = wg.barrier((rotate(ring, drained),))?;
        wg.ds_load_cur(&ring, LOAD.into())?;
        let next = wg.ds_store(ring, STAGE.into())?;
        Ok((next, wg.scmp_wg_uniform(CMP.into())?))
    })
    .unwrap();
}

#[test]
fn halo_verify_attn_race_is_rejected_at_the_loop_head_barrier() {
    let mut b = Trace::new("gfx1151");
    let e = err(k_tile_loop(&mut b, false));
    assert!(e.contains("undrained LDS store"), "{e}");
    assert_eq!(b.count("s_barrier"), 0, "the racy barrier never reached the backend");
}

#[test]
fn halo_verify_attn_fix_emits_the_typed_twin_exactly() {
    let (mut dynamic, mut typed) = (Trace::new("gfx1151"), Trace::new("gfx1151"));
    k_tile_loop(&mut dynamic, true).unwrap();
    k_tile_loop_typed(&mut typed);
    assert_eq!(dynamic.text, typed.text);
    let barrier = dynamic.text.iter().position(|t| t == "s_barrier").unwrap();
    assert_eq!(dynamic.text[barrier - 1], "s_waitcnt lgkmcnt(0)");
    assert_eq!((dynamic.count("s_waitcnt"), dynamic.count("s_barrier")), (1, 1));
}

#[test]
fn loop_back_edge_must_return_the_store_state_it_entered_with() {
    // The loop is entered with an undrained tile (a wait at the head is
    // needed); a body that also drains its own tail store hands the head a
    // drained one, so the second trip is not the first: the types call this
    // `Drained` vs `Pending`.
    let mut b = Trace::new("gfx1151");
    let mut d = Driver::<Gfx1151, Trace>::new(&mut b).unwrap();
    let k0 = d.lds("K0", 0, 8704).unwrap();
    let k1 = d.lds("K1", 8704, 8704).unwrap();
    let ring = d.ring("K", k0, k1).unwrap();
    d.ds_store(ring, STAGE.into()).unwrap();
    d.ring_steady(ring).unwrap();
    let e = err(d.loop_carried(".Lk_tile", |d| {
        d.wait(ring)?;
        d.barrier(&[Transition::Rotate(ring)])?;
        d.ds_load_cur(ring, LOAD.into())?;
        d.ds_store(ring, STAGE.into())?;
        d.wait(ring)?;
        d.scmp_wg_uniform(CMP.into())
    }));
    assert!(e.contains("differs between the entry and the back edge"), "{e}");
}

// ---- gfx11 FA2 mailbox ----------------------------------------------------

const MBOX_STORE: &str = "ds_store_2addr_b32 v0, v1, v2 offset1:1";
const MBOX_LOAD: &str = "ds_load_2addr_b32 v[3:4], v0 offset1:1";
const V_FILL: &str = "ds_store_b128 v5, v[8:11]";

/// Runtime twin of `tests/ui/fa2_mailbox_race.rs` (`retire_first` = false)
/// and the non-fill body of `fa2_mailbox_fixed.rs` (`retire_first` = true).
fn fa2_mailbox(b: &mut Trace, retire_first: bool) -> Result<(), String> {
    let mut d = Driver::<Gfx1100, Trace>::new(b)?;
    let v_body = d.lds("V", 16384, 16376)?;
    let mbox = d.lds("mailbox", 32760, 8)?;
    d.ds_store(mbox, MBOX_STORE.into())?;
    d.wait(mbox)?;
    d.barrier(&[Transition::Ready(mbox)])?;
    d.ds_load(mbox, MBOX_LOAD.into())?;
    if retire_first {
        d.barrier(&[Transition::Retire(mbox)])?;
    }
    // First V fill of the tile loop: the whole plane, mailbox included.
    let v_plane = d.join(&[v_body, mbox])?;
    d.ds_store(v_plane, V_FILL.into())?;
    d.wait(v_plane)
}

fn fa2_non_fill_typed(b: &mut Trace) {
    let mut wg = Workgroup::<Gfx1100, Trace>::new(b).unwrap();
    let v_body = wg.lds::<VBody>("V", 16384, 16376).unwrap();
    let mbox = wg.lds::<Mailbox>("mailbox", 32760, 8).unwrap();
    let (mbox, st) = wg.ds_store(mbox, MBOX_STORE.into()).unwrap();
    let drained = wg.wait(st).unwrap();
    let (mbox,) = wg.barrier((ready(mbox, drained),)).unwrap();
    wg.ds_load(&mbox, MBOX_LOAD.into()).unwrap();
    let (mbox,) = wg.barrier((retire(mbox),)).unwrap();
    let v_plane = join::<VPlane, _, _>(v_body, mbox).unwrap();
    let (_v, st) = wg.ds_store(v_plane, V_FILL.into()).unwrap();
    let _ = wg.wait(st).unwrap();
}

#[test]
fn fa2_mailbox_race_is_rejected_at_the_join() {
    let mut b = Trace::new("gfx1100");
    let e = err(fa2_mailbox(&mut b, false));
    assert!(e.contains("cannot join mailbox") && e.contains("Published"), "{e}");
    assert_eq!(b.count("s_barrier"), 1, "only the publishing barrier was emitted");
    assert_eq!(b.count("ds_store_b128"), 0, "the V fill never reached the backend");
}

#[test]
fn fa2_mailbox_fix_emits_the_typed_twin_exactly() {
    let (mut dynamic, mut typed) = (Trace::new("gfx1100"), Trace::new("gfx1100"));
    fa2_mailbox(&mut dynamic, true).unwrap();
    fa2_non_fill_typed(&mut typed);
    assert_eq!(dynamic.text, typed.text);
    assert_eq!(dynamic.count("s_barrier"), 2);
}

#[test]
fn rejected_requests_emit_nothing_and_leave_the_driver_usable() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let v = d.lds("V", 0, 1024).unwrap();
    let mbox = d.lds("mailbox", 1024, 8).unwrap();
    d.ds_store(mbox, MBOX_STORE.into()).unwrap();
    let before = d.position();
    // Publishing without a wait, retiring a Writing region, reading it.
    assert!(err(d.barrier(&[Transition::Ready(mbox)])).contains("undrained LDS store"));
    assert!(err(d.barrier(&[Transition::Retire(mbox)])).contains("only a Published region"));
    assert!(err(d.ds_load(mbox, MBOX_LOAD.into())).contains("not published"));
    assert!(err(d.barrier(&[Transition::Ready(v)])).contains("only a Writing region"));
    assert_eq!(d.position(), before);
    assert_eq!((d.region_phase(mbox).unwrap(), d.region_phase(v).unwrap()), (Phase::RtWriting, Phase::RtFree));
    d.wait(mbox).unwrap();
    d.barrier(&[Transition::Ready(mbox)]).unwrap();
    // A published region takes no store and cannot be waited on again.
    assert!(err(d.ds_store(mbox, MBOX_STORE.into())).contains("Published"));
    assert!(err(d.wait(mbox)).contains("no store of this wave is in flight"));
    assert!(err(d.barrier(&[Transition::Ready(mbox)])).contains("only a Writing region"));
    d.ds_load(mbox, MBOX_LOAD.into()).unwrap();
    d.barrier(&[Transition::Retire(mbox)]).unwrap();
    assert!(err(d.barrier(&[Transition::Retire(mbox)])).contains("only a Published region"));
}

// ---- scope and wait rules -------------------------------------------------

#[test]
fn a_stale_wait_does_not_cover_a_later_store() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 64).unwrap();
    d.ds_store(r, STAGE.into()).unwrap();
    d.wait(r).unwrap();
    d.ds_store(r, STAGE.into()).unwrap();
    assert!(err(d.barrier(&[Transition::Ready(r)])).contains("undrained LDS store"));
    d.wait(r).unwrap();
    d.barrier(&[Transition::Ready(r)]).unwrap();
}

#[test]
fn barriers_and_layout_refuse_under_wave_uniform_control() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 64).unwrap();
    d.ds_store(r, STAGE.into()).unwrap();
    d.wait(r).unwrap();
    let c = d.scmp("s_cmp_lg_u32 s4, 0".into()).unwrap();
    let e = err(d.skip_if(c, ".Lskip", |d| d.barrier(&[Transition::Ready(r)])));
    assert!(e.contains("workgroup scope"), "{e}");
    let e = err(d.loop_until(".Lh", ".Lx", |d, _| d.barrier(&[])));
    assert!(e.contains("workgroup scope"), "{e}");
    assert_eq!(d.region_phase(r).unwrap(), Phase::RtWriting);
}

#[test]
fn conditional_bodies_must_not_change_region_ownership() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 64).unwrap();
    let c = d.scmp("s_cmp_lg_u32 s4, 0".into()).unwrap();
    let e = err(d.skip_if(c, ".Lskip", |d| d.ds_store(r, STAGE.into())));
    assert!(e.contains("disagree on LDS ownership"), "{e}");
}

#[test]
fn no_workgroup_barrier_follows_a_wave_uniform_exit() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 64).unwrap();
    let end = d.exit(".Lend").unwrap();
    let c = d.scmp("s_cmp_lg_u32 s4, 0".into()).unwrap();
    d.wave_exit_unless(c, &end).unwrap();
    d.ds_store(r, STAGE.into()).unwrap();
    d.wait(r).unwrap();
    assert!(err(d.barrier(&[Transition::Ready(r)])).contains("after a wave-uniform kernel exit"));
}

#[test]
fn ids_belong_to_their_driver_and_a_second_driver_cannot_seal_the_backend() {
    let (mut a, mut c) = (Trace::new("gfx1100"), Trace::new("gfx1100"));
    let mut da = Driver::<Gfx1100, Trace>::new(&mut a).unwrap();
    let mut dc = Driver::<Gfx1100, Trace>::new(&mut c).unwrap();
    let r = da.lds("R", 0, 64).unwrap();
    assert!(err(dc.ds_store(r, STAGE.into())).contains("another Driver"));
    drop((da, dc));
    let e = Driver::<Gfx1100, Trace>::new(&mut a).err().expect("sealed backend");
    assert!(e.contains("already owned"), "{e}");
}

#[test]
fn split_barrier_holds_its_regions_until_wait_arrived() {
    let mut b = Trace::new("gfx1201");
    let mut d = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 64).unwrap();
    d.ds_store(r, STAGE.into()).unwrap();
    d.wait(r).unwrap();
    let a = d.signal(&[Transition::Ready(r)]).unwrap();
    assert!(err(d.ds_load(r, LOAD.into())).contains("held by a split barrier"));
    assert!(err(d.signal(&[])).contains("already in flight"));
    d.wait_arrived(a).unwrap();
    assert_eq!(d.region_phase(r).unwrap(), Phase::RtPublished);
    d.ds_load(r, LOAD.into()).unwrap();
}

// ---- handoff parity ---------------------------------------------------------

#[test]
fn handoff_emits_the_typed_twin_exactly() {
    let mut typed = Trace::new("gfx1201");
    {
        let mut wg = Workgroup::<Gfx1201, Trace>::new(&mut typed).unwrap();
        let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
        let end = wg.exit(".Lend").unwrap();
        let readers = wg.scmp("s_cmp_ge_u32 s4, 2".into()).unwrap();
        wg.handoff(
            readers,
            ".Lup",
            end,
            gate,
            |w, gate| {
                let out = w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
                w.ds_store(out, "ds_store_b128 v1, v[6:9] offset:16".into())
            },
            |w, gate, _| {
                w.ds_load(&gate, "ds_load_b128 v[8:11], v1".into())?;
                w.ds_load(&gate, "ds_load_b128 v[12:15], v1 offset:16".into())
            },
        )
        .unwrap();
    }
    let mut dynamic = Trace::new("gfx1201");
    {
        let mut d = Driver::<Gfx1201, Trace>::new(&mut dynamic).unwrap();
        let gate = d.lds("gate", 0, 2048).unwrap();
        let end = d.exit(".Lend").unwrap();
        let readers = d.scmp("s_cmp_ge_u32 s4, 2".into()).unwrap();
        d.handoff(
            readers,
            ".Lup",
            end,
            gate,
            |d, gate| {
                d.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
                d.ds_store(gate, "ds_store_b128 v1, v[6:9] offset:16".into())
            },
            |d, gate, _| {
                d.ds_load(gate, "ds_load_b128 v[8:11], v1".into())?;
                d.ds_load(gate, "ds_load_b128 v[12:15], v1 offset:16".into())
            },
        )
        .unwrap();
    }
    dynamic.finish().unwrap();
    assert_eq!(dynamic.text, typed.text);
}

#[test]
fn a_begun_write_is_pending_even_before_its_first_store() {
    let mut b = Trace::new("gfx1100");
    let mut d = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 64).unwrap();
    d.begin_write(r).unwrap();
    // The token is `Pending` with no event yet: publishing needs a wait.
    assert!(err(d.barrier(&[Transition::Ready(r)])).contains("undrained LDS store"));
    // A conditional store keeps `Pending` as `Pending`: same ownership shape.
    let c = d.scmp("s_cmp_lg_u32 s4, 0".into()).unwrap();
    d.skip_if(c, ".Lskip", |d| d.ds_store(r, STAGE.into())).unwrap();
    // A wait inside the body would turn it `Drained` on one path only.
    let c = d.scmp("s_cmp_lg_u32 s4, 0".into()).unwrap();
    assert!(err(d.skip_if(c, ".Lskip2", |d| d.wait(r))).contains("disagree on LDS ownership"));
}

// ---- handle identity -------------------------------------------------------

#[test]
fn a_condition_from_another_driver_is_refused_at_a_matching_position() {
    let (mut a, mut b) = (Trace::new("gfx1100"), Trace::new("gfx1100"));
    let mut da = Driver::<Gfx1100, Trace>::new(&mut a).unwrap();
    let mut db = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let wg_cond = da.scmp_wg_uniform(CMP.into()).unwrap();
    let wave_cond = da.scmp(CMP.into()).unwrap();
    // B's own compare ends at the position A's handle was made at.
    let own = db.scmp(CMP.into()).unwrap();
    let before = db.position();
    // A wave-dependent SCC must not become workgroup control.
    let e = err(db.wg_skip_if(wg_cond, ".Lskip", |d| d.barrier(&[])));
    assert!(e.contains("another Driver"), "{e}");
    assert!(err(db.skip_if(wave_cond, ".Lskip", |_| Ok(()))).contains("another Driver"));
    let end = db.exit(".Lend").unwrap();
    assert!(err(db.exit_if(wg_cond, &end)).contains("another Driver"));
    assert_eq!(db.position(), before, "nothing was branched");
    // The local condition is still usable.
    db.skip_if(own, ".Lskip", |_| Ok(())).unwrap();
}

#[test]
fn an_exit_from_another_driver_is_refused_and_a_local_clone_still_forbids_a_tail() {
    let (mut a, mut b) = (Trace::new("gfx1100"), Trace::new("gfx1100"));
    let mut da = Driver::<Gfx1100, Trace>::new(&mut a).unwrap();
    let mut db = Driver::<Gfx1100, Trace>::new(&mut b).unwrap();
    let foreign = da.exit(".Lend").unwrap();
    let own = db.exit(".Lend").unwrap();
    let c = db.scmp_wg_uniform(CMP.into()).unwrap();
    let before = db.position();
    assert!(err(db.exit_if(c, &foreign)).contains("another Driver"));
    assert!(err(db.end(foreign.clone())).contains("another Driver"));
    assert_eq!(db.position(), before, "no branch and no exit placed");
    // A clone of the local exit shares the "a branch reaches it" flag.
    let c = db.scmp_wg_uniform(CMP.into()).unwrap();
    db.exit_if(c, &own.clone()).unwrap();
    let e = err(db.end_with(own, |d| d.raw(|b| b.raw("v_mov_b32 v0, 1"))));
    assert!(e.contains("takes no tail"), "{e}");
}

#[test]
fn a_stale_or_foreign_arrived_token_is_refused() {
    let (mut a, mut b) = (Trace::new("gfx1201"), Trace::new("gfx1201"));
    let mut da = Driver::<Gfx1201, Trace>::new(&mut a).unwrap();
    let mut db = Driver::<Gfx1201, Trace>::new(&mut b).unwrap();
    // A clone of a completed signal must not consume a newer identical one.
    let first = da.signal(&[]).unwrap();
    let stale = first.clone();
    da.wait_arrived(first).unwrap();
    let second = da.signal(&[]).unwrap();
    assert!(err(da.wait_arrived(stale)).contains("without the signal"));
    // Nor may another driver's token (an empty list matches anything by value).
    let foreign = db.signal(&[]).unwrap();
    assert!(err(da.wait_arrived(foreign.clone())).contains("without the signal"));
    da.wait_arrived(second).unwrap();
    db.wait_arrived(foreign).unwrap();
}
