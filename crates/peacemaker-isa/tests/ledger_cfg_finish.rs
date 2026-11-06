// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Builder::verify_waits` checks every use/WAR/WAW has its wait on every control-flow
//! path of the emitted program (`ledger_cfg`), not only on the linear path the
//! builder's ledger follows across raw branches. Emission is unchanged and `finish`
//! does not run the check: these kernels differ only in what `verify_waits` finds.

use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::ledger::Counter;
use peacemaker_isa::ledger_cfg::Stats;
use peacemaker_isa::reg::{Kind, Live, RegRef};
use peacemaker_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan, S, V};

fn builder() -> Builder {
    let spec = KernelSpec {
        kernel_id: "cfg_wait".into(),
        variant: "t".into(),
        arch: Arch::Gfx1201,
        symbol: "cfg_wait".into(),
        kernargs: KernargLayout::new(16),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut regs = RegPlan::new(32, 32).unwrap();
    regs.add_range("s45", Kind::S, 4, 2, Live::Whole).unwrap();
    for b in [0u8, 2, 3] {
        regs.add_range(&format!("v{b}"), Kind::V, b, 1, Live::Whole)
            .unwrap();
    }
    Builder::new(spec, regs)
}
fn vr(n: u8) -> RegRef {
    V::<1>(n).reg()
}
fn load(b: &mut Builder) -> Result<(), String> {
    b.push(
        Instruction::new(
            "global_load_b32 v2, v0, s[4:5]",
            vec![vr(2)],
            vec![vr(0), S::<2>(4).reg()],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn use_load(b: &mut Builder) -> Result<(), String> {
    b.push(Instruction::new(
        "v_mov_b32 v3, v2",
        vec![vr(3)],
        vec![vr(2)],
    ))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}
/// End the kernel, then check its waits and finish it: `finish` accepts what
/// `verify_waits` rejects.
fn end(mut b: Builder) -> (Result<Stats, String>, peacemaker_isa::Emitted) {
    branch(&mut b, "s_endpgm").unwrap();
    let verdict = b.verify_waits();
    (verdict, b.finish().expect("finish does not verify waits"))
}

/// The wait precedes the branch, so every path to the label holds it.
#[test]
fn wait_before_the_branch_is_accepted() {
    let mut b = builder();
    load(&mut b).unwrap();
    b.wait(Counter::Load, 0).unwrap();
    branch(&mut b, "s_cbranch_scc1 SKIP").unwrap();
    b.label("SKIP").unwrap();
    use_load(&mut b).unwrap();
    end(b).0.unwrap();
}

/// The linear ledger sees the wait on the fall-through; the branch path to `SKIP`
/// reaches the use with the load still pending.
#[test]
fn wait_missing_on_a_branch_path_is_found_by_verify_waits() {
    let mut b = builder();
    load(&mut b).unwrap();
    branch(&mut b, "s_cbranch_scc1 SKIP").unwrap();
    b.wait(Counter::Load, 0).unwrap();
    b.label("SKIP").unwrap();
    use_load(&mut b).unwrap();
    let e = end(b).0.unwrap_err();
    assert!(e.contains("v_mov_b32 v3, v2"), "{e}");
    assert!(e.contains("s_cbranch_scc1 SKIP` taken"), "{e}");
}

/// A raw back edge to a non-`loop_` label: the first iteration enters the head with
/// nothing in flight, the second with the load issued below.
#[test]
fn raw_back_edge_carrying_a_pending_load_is_found_by_verify_waits() {
    let mut b = builder();
    b.label("HEAD").unwrap();
    use_load(&mut b).unwrap();
    load(&mut b).unwrap();
    branch(&mut b, "s_cbranch_scc1 HEAD").unwrap();
    b.wait(Counter::Load, 0).unwrap();
    let e = end(b).0.unwrap_err();
    assert!(e.contains("s_cbranch_scc1 HEAD` taken"), "{e}");
}

/// Same loop with the wait before the back edge.
#[test]
fn raw_back_edge_after_the_wait_is_accepted() {
    let mut b = builder();
    b.label("HEAD").unwrap();
    use_load(&mut b).unwrap();
    load(&mut b).unwrap();
    b.wait(Counter::Load, 0).unwrap();
    branch(&mut b, "s_cbranch_scc1 HEAD").unwrap();
    end(b).0.unwrap();
}

/// The load is issued only on a path that never reaches the label: after an
/// unconditional branch the linear ledger still holds it at `L` (the builder
/// then emits a wait there), while the CFG walk sees `L` reached only by the
/// conditional branch with nothing in flight. Emission keeps the builder's wait.
#[test]
fn load_on_a_dead_fall_through_is_not_a_ghost_pending_operation() {
    let mut b = builder();
    branch(&mut b, "s_cbranch_scc1 L").unwrap();
    branch(&mut b, "s_branch END").unwrap();
    load(&mut b).unwrap();
    b.label("L").unwrap();
    use_load(&mut b).unwrap();
    b.label("END").unwrap();
    let (verdict, e) = end(b);
    verdict.unwrap();
    assert!(
        e.s_text.contains("s_wait_loadcnt"),
        "the builder's emitted bytes are unchanged: the linear ledger still waits at L"
    );
}
