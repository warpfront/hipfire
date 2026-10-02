//! The typed core over the builder: a publish/read/rotate double buffer
//! written against `peacemaker_author` emits exactly what the same program
//! emits through the untyped builder (instructions, waits and barrier
//! transitions), and a sealed builder refuses untyped LDS and barrier calls,
//! raw branches and a second `Workgroup`. Every control join (skip targets,
//! `Forward` targets reached from several branches, loop heads reached from
//! the entry and the back edge) keeps the waits and hazard guards of every
//! path into it.
use hipfire_isa::{Arch, Builder, KernelSpec, KernargLayout, RegPlan};
use hipfire_isa::insn::{Instruction, MemoryClass};
use hipfire_isa::lds::Transition;
use hipfire_isa::reg::{Kind, Live, RegRef};
use peacemaker_author::{Gfx1100, Gfx1151, Gfx1201, Ring, Target, Workgroup, prime, retire, rotate};

fn probe(arch: Arch) -> Builder {
    let mut plan = RegPlan::new(16, 8).unwrap();
    for (name, n) in [("a", 0), ("b", 1), ("addr", 2), ("c", 3), ("e", 5)] { plan.v::<1>(name, n, Live::Whole).unwrap(); }
    for (name, n) in [("x", 5), ("y", 6)] { plan.s::<1>(name, n, Live::Whole).unwrap(); }
    Builder::new(KernelSpec {
        kernel_id: "probe".into(), variant: "default".into(), arch, symbol: "probe".into(), kernargs: KernargLayout::new(8),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 64, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, plan)
}
fn v(n: u8) -> RegRef { RegRef { kind: Kind::V, base: n, len: 1 } }
fn store(o: u32) -> Instruction { Instruction::new(format!("ds_store_b32 v2, v0 offset:{o}"), vec![], vec![v(2), v(0)]).memory(MemoryClass::DsStore) }
fn load(o: u32) -> Instruction { Instruction::new(format!("ds_load_b32 v1, v2 offset:{o}"), vec![v(1)], vec![v(2)]).memory(MemoryClass::DsLoad) }
fn mul() -> Instruction { Instruction::new("v_mul_f32_e32 v3, v1, v1", vec![v(3)], vec![v(1)]) }
fn sg(n: u8) -> RegRef { RegRef { kind: Kind::S, base: n, len: 1 } }
/// `v1 = global[v2]`: pending on VMcnt/LOADcnt until awaited.
fn gload() -> Instruction { Instruction::new("global_load_b32 v1, v2, s[0:1]", vec![v(1)], vec![v(2)]).memory(MemoryClass::VmemLoad) }
fn cmp(text: &str) -> Instruction { Instruction::new(text, vec![], vec![]) }
/// A VALU read of s5 (gfx12 tracks VALU-read SGPRs for SALU write hazards).
fn read_s5() -> Instruction { Instruction::new("v_add_nc_u32_e32 v3, s5, v0", vec![v(3)], vec![sg(5), v(0)]) }
fn write_s5() -> Instruction { Instruction::new("s_mov_b32 s5, 1", vec![sg(5)], vec![]) }
/// `v5 = global[v2]`, issued after `gload`.
fn gload5() -> Instruction { Instruction::new("global_load_b32 v5, v2, s[0:1]", vec![v(5)], vec![v(2)]).memory(MemoryClass::VmemLoad) }
/// Two VMEM stores whose sources stay locked until they are read.
fn gstores() -> [Instruction; 2] {
    [Instruction::new("global_store_b32 v2, v0, s[0:1]", vec![], vec![v(2), v(0)]).memory(MemoryClass::VmemStore),
     Instruction::new("global_store_b32 v2, v1, s[0:1] offset:4", vec![], vec![v(2), v(1)]).memory(MemoryClass::VmemStore)]
}
const SA_GUARD: &str = "s_wait_alu depctr_sa_sdst(0)";
/// The instructions after label `l`.
fn after(b: &Builder, l: &str) -> Vec<String> {
    let t = text(b);
    t[t.iter().position(|i| *i == format!("{l}:")).unwrap() + 1..].to_vec()
}

/// Two stores into buffer 0, publish; read it while staging buffer 1; rotate.
fn untyped(arch: Arch) -> Builder {
    let mut b = probe(arch);
    let s0 = b.lds_slot("B0", 0, 256).unwrap();
    let s1 = b.lds_slot("B1", 256, 256).unwrap();
    b.ds_store(s0, store(0)).unwrap();
    b.ds_store(s0, store(4)).unwrap();
    b.barrier(&[Transition::Ready(s0)]).unwrap();
    b.ds_load(s0, load(0)).unwrap();
    b.ds_store(s1, store(256)).unwrap();
    b.push(mul()).unwrap();
    b.barrier(&[Transition::Retire(s0), Transition::Ready(s1)]).unwrap();
    b.ds_load(s1, load(256)).unwrap();
    b
}
enum Buf {}
fn typed<T: Target>(arch: Arch) -> Builder {
    let mut b = probe(arch);
    let mut wg = Workgroup::<T, Builder>::new(&mut b).unwrap();
    let (b0, b1) = (wg.lds::<Buf>("B0", 0, 256).unwrap(), wg.lds::<Buf>("B1", 256, 256).unwrap());
    let st = wg.ds_store(Ring::new(b0, b1), store(0)).unwrap();
    let (ring, pending) = wg.ds_store(st, store(4)).unwrap();
    let drained = wg.wait(pending).unwrap();
    let (ring,) = wg.barrier((prime(ring, drained),)).unwrap();
    wg.ds_load_cur(&ring, load(0)).unwrap();
    let (ring, pending) = wg.ds_store(ring, store(256)).unwrap();
    wg.isa().push(mul()).unwrap();
    let drained = wg.wait(pending).unwrap();
    let (ring,) = wg.barrier((rotate(ring, drained),)).unwrap();
    wg.ds_load_cur(&ring, load(256)).unwrap();
    b
}
fn text(b: &Builder) -> Vec<String> { b.program().instructions.iter().map(|i| i.text.clone()).collect() }

#[test]
fn typed_double_buffer_emits_the_untyped_program() {
    for (arch, t) in [(Arch::Gfx1100, typed::<Gfx1100>(Arch::Gfx1100)), (Arch::Gfx1201, typed::<Gfx1201>(Arch::Gfx1201))] {
        let u = untyped(arch);
        assert_eq!(text(&t), text(&u), "{arch:?}");
        let tr = |b: &Builder| b.barriers.iter().map(|p| (p.pc_index, p.transitions.clone())).collect::<Vec<_>>();
        assert_eq!(tr(&t), tr(&u), "{arch:?}");
        let waits = |b: &Builder| b.waits.iter().map(|w| (w.pc_index, w.insn.clone())).collect::<Vec<_>>();
        assert_eq!(waits(&t), waits(&u), "{arch:?}");
    }
    // The drains are real: one per publishing barrier.
    assert_eq!(text(&untyped(Arch::Gfx1100)).iter().filter(|t| *t == "s_waitcnt lgkmcnt(0)").count(), 2);
}

#[test]
fn sealed_builder_refuses_untyped_lds_barriers_and_loops() {
    let mut b = probe(Arch::Gfx1100);
    let slot = b.lds_slot("S", 0, 256).unwrap();
    let wg = Workgroup::<Gfx1100, Builder>::new(&mut b).unwrap();
    drop(wg);
    assert!(b.ds_store(slot, store(0)).is_err());
    assert!(b.barrier(&[]).is_err());
    assert!(b.loop_(".Lx", |_| Ok(())).is_err());
    // Nor does it take a second Workgroup.
    assert!(Workgroup::<Gfx1100, Builder>::new(&mut b).is_err());
    // A backend for another architecture is refused outright.
    let mut b = probe(Arch::Gfx1201);
    assert!(Workgroup::<Gfx1100, Builder>::new(&mut b).is_err());
}

/// The reviewer's escalation counterexample on the production backend: a
/// wave-only body rebuilding a `Workgroup` from `isa` to reach a barrier.
#[test]
fn a_wave_scope_cannot_rebuild_its_workgroup() {
    let mut b = probe(Arch::Gfx1100);
    let mut wg = Workgroup::<Gfx1100, Builder>::new(&mut b).unwrap();
    let first_wave = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    let err = wg.skip_if(first_wave, ".Lskip", (), |w, ()| Workgroup::<Gfx1100, Builder>::new(w.isa()).map(drop)).unwrap_err();
    assert!(err.contains("already owned"), "{err}");
    assert!(b.barriers.is_empty());
}

enum Gate {}
fn handoff<T: Target>(arch: Arch) -> Builder {
    let mut b = probe(arch);
    let mut wg = Workgroup::<T, Builder>::new(&mut b).unwrap();
    let gate = wg.lds::<Gate>("gate", 0, 256).unwrap();
    let end = wg.exit(".Lend").unwrap();
    let readers = wg.scmp(cmp("s_cmp_ge_u32 s6, 2")).unwrap();
    let gate = wg.handoff(readers, ".Lup", end, gate, |w, gate| w.ds_store(gate, store(0)), |w, gate, _| {
        w.ds_load(&gate, load(0))?;
        w.isa().push(mul())?;
        Ok(gate)
    }).unwrap();
    // Readers alone run past the handoff: nothing follows its exit.
    assert!(wg.barrier((retire(gate),)).err().unwrap().contains("already ended"));
    assert!(wg.isa().push(mul()).is_err());
    b
}

/// Writers drain, publish at one barrier and leave for the exit; readers
/// meet them at one barrier of their own (`s_barrier` on gfx11, the split
/// signal/wait pair on gfx12), read and end at the same exit.
#[test]
fn handoff_on_both_barrier_models() {
    for (b, drain, wait_load, barrier) in [
        (handoff::<Gfx1151>(Arch::Gfx1151), "s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(0)", vec!["s_barrier"]),
        (handoff::<Gfx1201>(Arch::Gfx1201), "s_wait_dscnt 0x0", "s_wait_dscnt 0x0", vec!["s_barrier_signal -1", "s_barrier_wait 0xffff"]),
    ] {
        let mut want = vec!["s_cmp_ge_u32 s6, 2", "s_cbranch_scc1 .Lup", "ds_store_b32 v2, v0 offset:0", drain];
        want.extend(&barrier);
        want.extend(["s_branch .Lend", ".Lup:"]);
        want.extend(&barrier);
        want.extend(["ds_load_b32 v1, v2 offset:0", wait_load, "v_mul_f32_e32 v3, v1, v1", ".Lend:", "s_endpgm"]);
        assert_eq!(text(&b), want);
        assert_eq!(b.barriers.len(), 2);
        b.finish().unwrap();
    }
}

/// An `if_else` arm starts from the branch point's hazard state, and the
/// join keeps a guard either arm still owes. gfx12: s5 is VALU-read, then
/// SALU-written, so its next VALU read needs `depctr_sa_sdst`.
#[test]
fn if_else_arms_start_and_join_with_the_branch_hazards() {
    let guard = "s_wait_alu depctr_sa_sdst(0)";
    // Both arms read s5: each needs the guard (the else arm must not
    // inherit the then arm's cleared tracker).
    let mut b = probe(Arch::Gfx1201);
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
    wg.isa().push(read_s5()).unwrap();
    wg.isa().push(write_s5()).unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    wg.if_else(c, ".Lelse", ".Ljoin", |w| w.isa().push(read_s5()), |w| w.isa().push(read_s5())).unwrap();
    wg.label(".Ljoin").unwrap();
    assert_eq!(after(&b, ".Lelse"), [guard, "v_add_nc_u32_e32 v3, s5, v0", ".Ljoin:"]);
    assert_eq!(text(&b).iter().filter(|t| *t == guard).count(), 2);
    // Only the else arm reads s5: past the join the then path still owes it.
    let mut b = probe(Arch::Gfx1201);
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
    wg.isa().push(read_s5()).unwrap();
    wg.isa().push(write_s5()).unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    wg.if_else(c, ".Lelse", ".Ljoin", |_| Ok(()), |w| w.isa().push(read_s5())).unwrap();
    wg.label(".Ljoin").unwrap();
    wg.isa().push(read_s5()).unwrap();
    assert_eq!(after(&b, ".Ljoin"), [guard, "v_add_nc_u32_e32 v3, s5, v0"]);
}

/// A skip target joins the skipped path: a load the body awaited is still
/// pending on the path that skipped it.
#[test]
fn skip_target_keeps_the_skipped_paths_pending_load() {
    let mut b = probe(Arch::Gfx1100);
    let mut wg = Workgroup::<Gfx1100, Builder>::new(&mut b).unwrap();
    wg.isa().push(gload()).unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    wg.skip_if(c, ".Lskip", (), |w, ()| w.isa().push(mul())).unwrap();
    wg.isa().push(mul()).unwrap();
    assert_eq!(after(&b, ".Lskip"), ["s_waitcnt vmcnt(0)", "v_mul_f32_e32 v3, v1, v1"]);
}

/// The reviewer's handoff counterexample: wave-scope code reaching the
/// builder through `isa` pushes `s_branch` back to an earlier label (or the
/// same through a line break, a raw encoding, another spelling, the untyped
/// control entry point). Every form is refused and nothing is emitted.
#[test]
fn raw_isa_cannot_branch_end_or_barrier() {
    let only = [".Lreader:", "s_cmp_eq_u32 s6, 0", "s_cbranch_scc1 .Lskip"];
    let mut b = probe(Arch::Gfx1151);
    raw_attempts(&mut Workgroup::<Gfx1151, Builder>::new(&mut b).unwrap());
    assert_eq!(text(&b), only);
    let mut b = probe(Arch::Gfx1201);
    raw_attempts(&mut Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap());
    assert_eq!(text(&b), only);
}
fn raw_attempts<T: Target>(wg: &mut Workgroup<T, Builder>) {
    wg.label(".Lreader").unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    wg.skip_if(c, ".Lskip", (), |w, ()| {
        for text in ["s_branch .Lreader", "s_cbranch_scc1 .Lreader", "s_cbranch_execz .Lreader", "s_setpc_b64 s[0:1]", "s_swappc_b64 s[0:1], s[2:3]",
            "s_endpgm", "s_trap 2", "s_barrier", "s_barrier_signal -1", "s_barrier_wait 0xffff", "s_barrier_leave",
            "v_mov_b32 v3, 0\n\ts_branch .Lreader", "S_BRANCH .Lreader", ".long 0xbfa0fffe", ".Lagain:", "s_nop 0\r s_endpgm"] {
            assert!(w.isa().push(cmp(text)).is_err(), "raw {text:?} accepted");
        }
        assert!(w.isa().control(cmp("s_branch .Lreader")).is_err(), "untyped control accepted on a sealed builder");
        Err("probe done".into())
    }).unwrap_err();
}

/// A loop head is reached from the entry and from the back edge, so the
/// body's first emission is redone from their join. gfx12: s5 is read by a
/// VALU at the head and written by a SALU before the back edge; the next
/// iteration's read needs `depctr_sa_sdst`, the entry path does not (the
/// reviewer's executed counterexample was accepted without it).
#[test]
fn loop_head_keeps_the_back_edges_sgpr_guard() {
    let want = [SA_GUARD, "v_add_nc_u32_e32 v3, s5, v0", "s_mov_b32 s5, 1", "s_cmp_lg_u32 s6, 0", "s_cbranch_scc1 .Lloop"];
    let mut b = probe(Arch::Gfx1201);
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
    wg.loop_carried(".Lloop", (), |wg, ()| {
        wg.isa().push(read_s5())?;
        wg.isa().push(write_s5())?;
        Ok(((), wg.scmp_wg_uniform(cmp("s_cmp_lg_u32 s6, 0"))?))
    }).unwrap();
    assert_eq!(after(&b, ".Lloop"), want);
    // The untyped loop has the same fixed point.
    let mut b = probe(Arch::Gfx1201);
    b.loop_(".Lloop", |b| {
        b.push(read_s5())?;
        b.push(write_s5())?;
        b.push(cmp("s_cmp_lg_u32 s6, 0"))?;
        b.control(cmp("s_cbranch_scc1 .Lloop"))
    }).unwrap();
    assert_eq!(after(&b, ".Lloop"), want);
    assert_eq!(b.loop_fixpoints[0].iterations, 2);
}

/// Port of the branch-hazard regression: a `Forward` arm starts from the
/// branch point's hazard state, and the join keeps a guard either arm still
/// owes. gfx12: s5 is VALU-read, then SALU-written, so its next VALU read
/// needs `depctr_sa_sdst`.
#[test]
fn forward_arms_start_and_join_with_the_branch_hazards() {
    let if_else = |then_reads: bool| {
        let mut b = probe(Arch::Gfx1201);
        let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
        wg.isa().push(read_s5()).unwrap();
        wg.isa().push(write_s5()).unwrap();
        let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
        wg.forward(|w, f| {
            f.branch_if(w, c, ".Lelse")?;
            if then_reads { w.isa().push(read_s5())? }
            f.goto(w, ".Ljoin")?;
            f.place(w, ".Lelse")?;
            w.isa().push(read_s5())?;
            f.place(w, ".Ljoin")
        }).unwrap();
        wg.isa().push(read_s5()).unwrap();
        b
    };
    // Both arms read s5: each needs the guard (the else arm must not
    // inherit the then arm's cleared tracker).
    let b = if_else(true);
    assert_eq!(after(&b, ".Lelse"), [SA_GUARD, "v_add_nc_u32_e32 v3, s5, v0", ".Ljoin:", "v_add_nc_u32_e32 v3, s5, v0"]);
    assert_eq!(text(&b).iter().filter(|t| *t == SA_GUARD).count(), 2);
    // Only the else arm reads s5: past the join the then path still owes it.
    let b = if_else(false);
    assert_eq!(after(&b, ".Ljoin"), [SA_GUARD, "v_add_nc_u32_e32 v3, s5, v0"]);
}

/// A target reached from several branches joins every one of them, not
/// just the last: path 1 branches with nothing pending, path 2 with the v1
/// load and an s5 guard owed, the fall-through with both loads and no guard
/// owed. Past the target the v1 use drains LOADcnt (count 1 would leave v1
/// in flight on path 2) and the s5 read takes path 2's guard.
#[test]
fn forward_target_joins_every_break_state() {
    let mut b = probe(Arch::Gfx1201);
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
    wg.forward(|w, f| {
        let c = w.scmp(cmp("s_cmp_eq_u32 s6, 0"))?;
        f.branch_if(w, c, ".Lout")?;
        w.isa().push(gload())?;
        w.isa().push(read_s5())?;
        w.isa().push(write_s5())?;
        let c = w.scmp(cmp("s_cmp_eq_u32 s6, 1"))?;
        f.branch_if(w, c, ".Lout")?;
        w.isa().push(read_s5())?;
        w.isa().push(gload5())?;
        f.place(w, ".Lout")
    }).unwrap();
    wg.isa().push(read_s5()).unwrap();
    wg.isa().push(mul()).unwrap();
    assert_eq!(after(&b, ".Lout"), [SA_GUARD, "v_add_nc_u32_e32 v3, s5, v0", "s_wait_loadcnt 0x0", "v_mul_f32_e32 v3, v1, v1"]);
}

/// Equal-shape in-order loads on two paths with different ids: the join
/// keeps both paths' ids pending in one slot per position, so the waits stay
/// exact (v1, the older load, needs only `s_wait_loadcnt 0x1`).
#[test]
fn equal_shape_load_join_keeps_both_ids_and_exact_waits() {
    let mut b = probe(Arch::Gfx1201);
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    let (a, other) = wg.forward(|w, f| {
        f.branch_if(w, c, ".Lb")?;
        w.isa().push(gload())?;
        let a = w.isa().ledger().last_id().unwrap();
        w.isa().push(gload5())?;
        f.goto(w, ".Ljoin")?;
        f.place(w, ".Lb")?;
        w.isa().push(gload())?;
        let other = w.isa().ledger().last_id().unwrap();
        w.isa().push(gload5())?;
        f.place(w, ".Ljoin")?;
        Ok((a, other))
    }).unwrap();
    assert_ne!(a, other);
    assert!(wg.isa().ledger().is_pending(a) && wg.isa().ledger().is_pending(other));
    wg.isa().push(mul()).unwrap();
    assert!(!wg.isa().ledger().is_pending(a) && !wg.isa().ledger().is_pending(other));
    assert_eq!(after(&b, ".Ljoin"), ["s_wait_loadcnt 0x1", "v_mul_f32_e32 v3, v1, v1"]);
}

/// Stores retire out of order: an equal-shape join of two VMEM stores per
/// path keeps the conservative union, so redefining the older store's
/// source drains STOREcnt. Straight-line code (no join) needs only 0x1.
#[test]
fn equal_shape_store_join_forces_a_full_drain() {
    let redefine_v0 = || Instruction::new("v_mov_b32_e32 v0, 0", vec![v(0)], vec![]);
    let mut b = probe(Arch::Gfx1201);
    for s in gstores() { b.push(s).unwrap(); }
    b.push(redefine_v0()).unwrap();
    assert_eq!(text(&b)[2], "s_wait_storecnt 0x1");
    let mut b = probe(Arch::Gfx1201);
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
    let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    wg.forward(|w, f| {
        f.branch_if(w, c, ".Lb")?;
        for s in gstores() { w.isa().push(s)? }
        f.goto(w, ".Ljoin")?;
        f.place(w, ".Lb")?;
        for s in gstores() { w.isa().push(s)? }
        f.place(w, ".Ljoin")
    }).unwrap();
    wg.isa().push(redefine_v0()).unwrap();
    assert_eq!(after(&b, ".Ljoin"), ["s_wait_storecnt 0x0", "v_mov_b32_e32 v0, 0"]);
}

/// One LDS store per path at the same ledger position, under different ids:
/// `Wave::wait` on either path's token drains it (one `s_wait_dscnt 0x0`),
/// after which the other path's token has nothing left to wait for.
#[test]
fn either_paths_lds_store_token_retires_the_joined_slot() {
    for first_a in [true, false] {
        let mut b = probe(Arch::Gfx1201);
        let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b).unwrap();
        let (r1, r2) = (wg.lds::<Buf>("R1", 0, 256).unwrap(), wg.lds::<Buf>("R2", 256, 256).unwrap());
        let c = wg.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
        let (pa, pb) = wg.forward(|w, f| {
            f.branch_if(w, c, ".Lb")?;
            let (_r1, pa) = w.ds_store(r1, store(0))?;
            f.goto(w, ".Ljoin")?;
            f.place(w, ".Lb")?;
            let (_r2, pb) = w.ds_store(r2, store(256))?;
            f.place(w, ".Ljoin")?;
            Ok((pa, pb))
        }).unwrap();
        let (first, second) = if first_a { (pa, pb) } else { (pb, pa) };
        let _drained = (wg.wait(first).unwrap(), wg.wait(second).unwrap());
        assert!(wg.isa().ledger().is_empty());
        assert_eq!(after(&b, ".Ljoin"), ["s_wait_dscnt 0x0"]);
    }
}

/// A `loop_until` exit continues from its `break_if` state, not the back
/// edge's: a load issued before the break and awaited after it is still
/// pending at the exit.
fn walk<T: Target>(arch: Arch) -> Builder {
    let mut b = probe(arch);
    let mut wg = Workgroup::<T, Builder>::new(&mut b).unwrap();
    wg.loop_until(".Lwalk", ".Lfound", |w, exit| {
        w.isa().push(gload())?;
        let hit = w.scmp(cmp("s_cmp_lg_u32 s6, 0"))?;
        w.break_if(hit, exit)?;
        w.isa().push(mul())
    }).unwrap();
    wg.isa().push(mul()).unwrap();
    b
}
#[test]
fn loop_until_exit_continues_from_the_break() {
    for (b, wait) in [(walk::<Gfx1151>(Arch::Gfx1151), "s_waitcnt vmcnt(0)"), (walk::<Gfx1201>(Arch::Gfx1201), "s_wait_loadcnt 0x0")] {
        assert_eq!(after(&b, ".Lwalk"), ["global_load_b32 v1, v2, s[0:1]", "s_cmp_lg_u32 s6, 0", "s_cbranch_scc1 .Lfound", wait,
            "v_mul_f32_e32 v3, v1, v1", "s_branch .Lwalk", ".Lfound:", wait, "v_mul_f32_e32 v3, v1, v1"]);
    }
}
