// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-level checks of the gfx12 VALU window scheduler and its `s_delay_alu` hints.

use peacemaker_isa::insn::Instruction;
use peacemaker_isa::reg::Live;
use peacemaker_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};

const NAMES: [&str; 8] = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"];

fn spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "valu_sched_probe".into(),
        variant: "default".into(),
        arch: Arch::Gfx1201,
        symbol: "valu_sched_probe".into(),
        kernargs: KernargLayout::new(8),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

fn texts(b: &Builder) -> Vec<String> {
    b.program()
        .instructions
        .iter()
        .map(|i| i.text.clone())
        .collect()
}

/// A merged hint (`instskip`/`instid1`) names a consumer after its own: the second consumer must not lie past a
/// branch, where peacemaker-lift's block check refuses it.
#[test]
fn delay_hint_does_not_merge_across_a_branch() {
    let mut plan = RegPlan::new(16, 8).unwrap();
    let v: Vec<_> = (0..8u8)
        .map(|i| plan.v::<1>(NAMES[i as usize], i, Live::Whole).unwrap())
        .collect();
    let mut b = Builder::new(spec(), plan);
    b.enable_delay_alu();
    b.enable_valu_schedule().unwrap();
    let add = |d: usize, x: usize, y: usize| {
        Instruction::new(
            format!("v_add_f32_e32 v{d}, v{x}, v{y}"),
            vec![v[d].reg()],
            vec![v[x].reg(), v[y].reg()],
        )
    };
    b.push(add(1, 2, 3)).unwrap();
    b.push(add(4, 1, 5)).unwrap(); // reads v1: the first hint (a merge target)
    b.control(Instruction::new("s_cbranch_scc1 .Lend", vec![], vec![]))
        .unwrap();
    b.push(add(6, 4, 7)).unwrap(); // reads v4 of the previous block: a hint of its own
    b.flush_valu().unwrap();
    b.label(".Lend").unwrap();
    b.control(Instruction::new("s_endpgm", vec![], vec![]))
        .unwrap();
    let t = texts(&b);
    let hints: Vec<&String> = t.iter().filter(|x| x.starts_with("s_delay_alu")).collect();
    assert_eq!(hints.len(), 2, "{t:?}");
    assert!(hints.iter().all(|h| !h.contains("instskip")), "{t:?}");
}
