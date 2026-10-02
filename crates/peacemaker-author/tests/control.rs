//! Wave-role handoff, joins, loop exits and scope ownership on the reference
//! backend.
use peacemaker_author::{ready, retire, trace::Trace, Gfx1100, Gfx1151, Gfx1201, LdsRegion, Published, Target, Workgroup};

enum Gate {}

fn handoff<T: Target>(arch: &'static str) -> Trace {
    let mut b = Trace::new(arch);
    let mut wg = Workgroup::<T, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let readers = wg.scmp("s_cmp_ge_u32 s4, 2".into()).unwrap();
    wg.handoff(
        readers,
        ".Lup",
        end,
        gate,
        |w, gate| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()),
        |w, gate, _| w.ds_load(&gate, "ds_load_b128 v[8:11], v1".into()),
    )
    .unwrap();
    b.finish().unwrap();
    b
}

/// Writers drain before their barrier and leave for the kernel exit;
/// readers resume from the branch point, meet them at their own barrier,
/// read, and end at the same exit. One barrier per wave on both barrier
/// models.
#[test]
fn handoff_publishes_to_the_reading_waves() {
    let gfx11 = ["s_waitcnt lgkmcnt(0)", "s_barrier"];
    let gfx12 = ["s_wait_dscnt 0x0", "s_barrier_signal -1", "s_barrier_wait 0xffff"];
    for (b, drain, barrier) in [
        (handoff::<Gfx1151>("gfx1151"), gfx11[0], &gfx11[1..]),
        (handoff::<Gfx1201>("gfx1201"), gfx12[0], &gfx12[1..]),
    ] {
        let mut want = vec!["s_cmp_ge_u32 s4, 2", "s_cbranch_scc1 .Lup", "ds_store_b128 v1, v[2:5]", drain];
        want.extend(barrier);
        want.extend(["s_branch .Lend", ".Lup:"]);
        want.extend(barrier);
        want.extend(["ds_load_b128 v[8:11], v1", ".Lend:", "s_endpgm"]);
        assert_eq!(b.text, want);
    }
}

/// The reviewer's post-handoff counterexample (handoff, reader load, a
/// retiring barrier only the readers reach): the reader continuation has no
/// barrier (`tests/ui/barrier_after_handoff.rs`), and once the handoff has
/// placed the exit nothing more is emitted.
#[test]
fn nothing_follows_a_handoff() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let readers = wg.scmp("s_cmp_ge_u32 s4, 2".into()).unwrap();
    let gate: LdsRegion<Gate, Published> = wg
        .handoff(readers, ".Lup", end, gate, |w, gate| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()), |w, gate, _| {
            w.ds_load(&gate, "ds_load_b128 v[8:11], v1".into())?;
            Ok(gate)
        })
        .unwrap();
    let err = wg.barrier((retire(gate),)).err().unwrap();
    assert!(err.contains("already ended"), "{err}");
    assert!(wg.label(".Lafter").is_err());
    assert_eq!(b.count("s_barrier"), 2);
}

/// A kernel exit is placed only by `Workgroup::end`, and a kernel that never
/// places it does not finish.
#[test]
fn exits_are_placed_by_end_only() {
    let mut b = Trace::new("gfx1100");
    let mut wg = Workgroup::<Gfx1100, Trace>::new(&mut b).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let outside = wg.scmp_wg_uniform("s_cmp_ge_u32 s2, s3".into()).unwrap();
    wg.exit_if(outside, &end).unwrap();
    assert!(wg.label(".Lend").is_err());
    drop(end);
    assert!(b.finish().is_err());
}

/// The reviewer's escalation counterexample: a wave-only body rebuilding a
/// `Workgroup` over the backend it reaches through `isa` (to place a
/// barrier under wave-dependent control) is refused.
#[test]
fn a_wave_scope_cannot_rebuild_its_workgroup() {
    let mut b = Trace::new("gfx1100");
    let mut wg = Workgroup::<Gfx1100, Trace>::new(&mut b).unwrap();
    let first_wave = wg.scmp("s_cmp_eq_u32 s18, 0".into()).unwrap();
    let err = wg
        .skip_if(first_wave, ".Lskip", (), |w, ()| Workgroup::<Gfx1100, Trace>::new(w.isa()).map(drop))
        .unwrap_err();
    assert!(err.contains("already owned"), "{err}");
    assert_eq!(b.count("s_barrier"), 0);
}

/// A store the skipped body made stays pending past the skip target: the
/// continuation cannot re-carve the LDS as if the body never ran.
#[test]
fn skip_target_joins_a_store_of_the_body() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let odd = wg.scmp("s_bitcmp1_b32 s19, 7".into()).unwrap();
    wg.skip_if(odd, ".Lskip", (), |w, ()| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).map(drop)).unwrap();
    let err = wg.relayout((other,)).unwrap_err();
    assert!(err.contains("gate is Publishing"), "{err}");
}

/// A store made on one arm only stays pending past the join: the
/// continuation cannot re-carve the LDS as if the arm never ran.
#[test]
fn if_else_joins_a_store_of_either_arm() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let odd = wg.scmp("s_bitcmp1_b32 s19, 7".into()).unwrap();
    wg.if_else(odd, ".Lodd", ".Ljoin", |w| w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).map(drop), |_| Ok(()))
        .unwrap();
    wg.label(".Ljoin").unwrap();
    let err = wg.relayout((other,)).unwrap_err();
    assert!(err.contains("gate is Publishing"), "{err}");
}

/// Paths that disagree on LDS ownership (published on one, untouched on the
/// other) do not join.
#[test]
fn skip_join_refuses_disagreeing_lds_ownership() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let skip = wg.scmp_wg_uniform("s_cmp_eq_u32 s2, 0".into()).unwrap();
    let err = wg
        .wg_skip_if(skip, ".Lskip", (), |wg, ()| {
            let (gate, pending) = wg.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
            let drained = wg.wait(pending)?;
            wg.barrier((ready(gate, drained),)).map(drop)
        })
        .unwrap_err();
    assert!(err.contains("gate Published on one and Free on the other"), "{err}");
}

/// A `loop_until` is left only through `break_if`.
#[test]
fn loop_until_needs_a_break() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let err = wg.loop_until(".Lwalk", ".Lfound", |_, _| Ok(())).unwrap_err();
    assert!(err.contains("no break_if"), "{err}");
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    wg.loop_until(".Lwalk", ".Lfound", |w, exit| {
        let hit = w.scmp("s_cmp_lg_u32 s4, 0".into())?;
        w.break_if(hit, exit)
    })
    .unwrap();
    assert_eq!(b.text, [".Lwalk:", "s_cmp_lg_u32 s4, 0", "s_cbranch_scc1 .Lfound", "s_branch .Lwalk", ".Lfound:"]);
}

/// Raw access through `isa` cannot emit what the typed core owns.
#[test]
fn raw_trace_refuses_lds_barriers_and_branches() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    for t in ["ds_store_b32 v1, v2", "s_barrier", "s_branch .Lx", "s_cbranch_scc1 .Lx", "s_endpgm"] {
        assert!(wg.isa().raw(t).is_err(), "{t}");
    }
    wg.isa().raw("v_mov_b32 v1, 0").unwrap();
}

/// `Forward` targets are placed once, after every branch to them, and
/// inside their block: no branch re-enters code.
#[test]
fn forward_branches_only_go_forward() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let err = wg
        .forward(|w, f| {
            let c = w.scmp("s_cmp_eq_u32 s2, 0".into())?;
            f.branch_if(w, c, ".Lx")?;
            f.place(w, ".Lx")?;
            f.goto(w, ".Lx")
        })
        .unwrap_err();
    assert!(err.contains("branches only go forward"), "{err}");
    let err = wg
        .forward(|w, f| {
            let c = w.scmp("s_cmp_eq_u32 s2, 0".into())?;
            f.branch_if(w, c, ".Ly")
        })
        .unwrap_err();
    assert!(err.contains(".Ly is never placed"), "{err}");
}

/// A target joins every branch to it: a slot stored only on the path of
/// the middle branch is still being written at the target.
#[test]
fn forward_target_joins_a_store_of_any_branch() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let _gate = wg
        .forward(|w, f| {
            let c = w.scmp("s_cmp_eq_u32 s2, 0".into())?;
            f.branch_if(w, c, ".Lout")?;
            let (gate, pending) = w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
            let c = w.scmp("s_cmp_eq_u32 s2, 1".into())?;
            f.branch_if(w, c, ".Lout")?;
            let _drained = w.wait(pending)?;
            f.place(w, ".Lout")?;
            Ok(gate)
        })
        .unwrap();
    assert!(wg.relayout((other,)).unwrap_err().contains("gate is Publishing"));
}

/// `exit_unless` continues at its target from the branch point: the
/// body's store leaves the kernel with it. An exit something branches to
/// takes no tail.
#[test]
fn exit_unless_target_starts_from_the_branch() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let c = wg.scmp_wg_uniform("s_cmp_eq_u32 s2, 0".into()).unwrap();
    wg.exit_unless(c, ".Lgo", &end, |wg| wg.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).map(drop)).unwrap();
    // Every slot is Free at the target: the path that stored left the kernel.
    wg.relayout((other,)).unwrap();
    let err = wg.end_with(end, |w| w.isa().raw("v_mov_b32 v1, 0")).unwrap_err();
    assert!(err.contains("takes no tail"), "{err}");
    assert_eq!(b.text[..4], ["s_cmp_eq_u32 s2, 0", "s_cbranch_scc1 .Lgo", "ds_store_b128 v1, v[2:5]", "s_branch .Lend"]);
}

#[test]
fn forward_unless_joins_pending_store_and_rejects_backward_target() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    wg.forward(|w, f| {
        let (gate, pending) = w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
        let c = w.scmp("s_cmp_gt_u32 s2, 1".into())?;
        f.branch_unless(w, c, ".Lout")?;
        let _ = w.wait(pending)?;
        f.place(w, ".Lout")?;
        let c = w.scmp("s_cmp_gt_u32 s2, 2".into())?;
        assert!(f.branch_unless(w, c, ".Lout").unwrap_err().contains("branches only go forward"));
        Ok(gate)
    }).unwrap();
    assert!(wg.relayout((other,)).unwrap_err().contains("gate is Publishing"));
    assert!(b.text.iter().any(|s| s == "s_cbranch_scc0 .Lout"));
}

#[test]
fn skip_unless_preserves_skipped_pending_store() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let other = wg.lds::<Gate>("other", 2048, 2048).unwrap();
    let (_, pending) = wg.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).unwrap();
    let c = wg.scmp("s_cmp_gt_u32 s2, 1".into()).unwrap();
    wg.skip_unless(c, ".Lout", (), |w, ()| w.wait(pending).map(drop)).unwrap();
    assert!(wg.relayout((other,)).unwrap_err().contains("gate is Publishing"));
    assert!(b.text.iter().any(|s| s == "s_cbranch_scc0 .Lout"));
}

#[test]
fn wave_exit_unless_refuses_later_workgroup_barrier_and_exit_tail() {
    let mut b = Trace::new("gfx1151");
    let mut wg = Workgroup::<Gfx1151, Trace>::new(&mut b).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 2048).unwrap();
    let (gate, pending) = wg.ds_store(gate, "ds_store_b128 v1, v[2:5]".into()).unwrap();
    let drained = wg.wait(pending).unwrap();
    let c = wg.scmp("s_cmp_gt_u32 s2, 1".into()).unwrap();
    peacemaker_author::Wave::exit_unless(&mut wg, c, &end).unwrap();
    assert!(wg.barrier((ready(gate, drained),)).err().unwrap().contains("wave-uniform kernel exit"));
    assert!(wg.end_with(end, |w| w.isa().raw("v_mov_b32 v1, 0")).unwrap_err().contains("takes no tail"));
    assert!(b.text.iter().any(|s| s == "s_cbranch_scc0 .Lend"));
}
