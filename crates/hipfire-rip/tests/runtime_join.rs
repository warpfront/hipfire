//! `runtime::Driver` over the real `hipfire_isa::Builder`: a store that is
//! still outstanding on the skipped path of a `skip_if` must be drained by
//! the driver's explicit `wait`, even though the body's path issued a
//! younger store and retired both with a raw counter wait. `Trace` cannot
//! show this (its raw instructions never touch its ledger), and the
//! builder's own barrier repair would mask a missing wait, so the proof is
//! the ledger *before* the publishing barrier.
use hipfire_isa::insn::{Instruction, MemoryClass};
use hipfire_isa::ledger::Counter;
use hipfire_isa::reg::{Kind, Live, RegRef};
use hipfire_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};
use peacemaker_author::runtime::{Driver, Phase, Transition};
use peacemaker_author::{Gfx1151, Gfx1201, Target};

fn probe(arch: Arch) -> Builder {
    let mut plan = RegPlan::new(16, 8).unwrap();
    for (name, n) in [("a", 0), ("b", 1), ("addr", 2), ("c", 3), ("e", 5)] {
        plan.v::<1>(name, n, Live::Whole).unwrap();
    }
    for (name, n) in [("x", 5), ("y", 6)] {
        plan.s::<1>(name, n, Live::Whole).unwrap();
    }
    Builder::new(
        KernelSpec {
            kernel_id: "probe".into(),
            variant: "default".into(),
            arch,
            symbol: "probe".into(),
            kernargs: KernargLayout::new(8),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 64,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        plan,
    )
}
fn v(n: u8) -> RegRef {
    RegRef { kind: Kind::V, base: n, len: 1 }
}
fn store(o: u32) -> Instruction {
    Instruction::new(format!("ds_store_b32 v2, v0 offset:{o}"), vec![], vec![v(2), v(0)]).memory(MemoryClass::DsStore)
}
fn cmp(text: &str) -> Instruction {
    Instruction::new(text, vec![], vec![])
}

fn joined_frontier<T: Target>(arch: Arch, counter: Counter) {
    let mut b = probe(arch);
    let mut d = Driver::<T, Builder>::new(&mut b).unwrap();
    let r = d.lds("R", 0, 256).unwrap();
    d.ds_store(r, store(0)).unwrap(); // e0
    let c = d.scmp(cmp("s_cmp_eq_u32 s6, 0")).unwrap();
    d.skip_if(c, ".Lskip", |d| {
        d.ds_store(r, store(4))?; // e1
        d.isa().wait(counter, 0) // raw drain of this path only
    })
    .unwrap();
    assert!(d.isa().ledger().pending_stores(), "e0 is still outstanding on the skipped path");
    d.wait(r).unwrap();
    assert!(
        !d.isa().ledger().pending_stores(),
        "{arch:?}: the explicit wait must drain the skipped path's store before the publishing barrier"
    );
    d.barrier(&[Transition::Ready(r)]).unwrap();
    assert_eq!(d.region_phase(r).unwrap(), Phase::RtPublished);
}

#[test]
fn explicit_wait_after_a_skip_drains_the_skipped_paths_store_gfx1151() {
    joined_frontier::<Gfx1151>(Arch::Gfx1151, Counter::Lgkm);
}

#[test]
fn explicit_wait_after_a_skip_drains_the_skipped_paths_store_gfx1201() {
    joined_frontier::<Gfx1201>(Arch::Gfx1201, Counter::Ds);
}
