// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-level checks of the gfx12 VALU window scheduler and its `s_delay_alu` hints.

use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::reg::{Kind, Live, RegRef};
use peacemaker_isa::regalloc::{compact_vgprs, declare_vgprs, Pass};
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

fn v(base: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len: 1,
    }
}

fn kernel_plan(pass: Pass) -> RegPlan {
    let mut regs = RegPlan::new(pass.vgprs().unwrap_or(16), 8).unwrap();
    match pass.vgprs() {
        None => {
            for b in 0..16u8 {
                regs.add_range(&format!("v{b}"), Kind::V, b, 1, Live::Whole)
                    .unwrap();
            }
        }
        Some(n) => declare_vgprs(&mut regs, n).unwrap(),
    }
    regs.add_range("srd", Kind::S, 0, 2, Live::Whole).unwrap();
    regs
}

/// The window scheduler pays one wait for the strongest requirement of the ready instructions, not the
/// weakest-first chain (four loads issued in order, four readers), and pass 2 of the VGPR compaction replays it.
#[test]
fn ready_instructions_share_one_memory_wait() {
    let build = |pass: Pass| {
        let mut b = Builder::new(spec(), kernel_plan(pass));
        b.vgpr_pass(pass);
        b.enable_valu_schedule().unwrap();
        for k in 0..4u8 {
            b.push(
                Instruction::new(
                    format!("global_load_b32 v{}, v1, s[0:1]", 4 + k),
                    vec![v(4 + k)],
                    vec![
                        v(1),
                        RegRef {
                            kind: Kind::S,
                            base: 0,
                            len: 2,
                        },
                    ],
                )
                .memory(MemoryClass::VmemLoad),
            )
            .unwrap();
        }
        for k in 0..4u8 {
            b.push(Instruction::new(
                format!("v_add_f32_e32 v{}, v{}, v{}", 8 + k, 4 + k, 4 + k),
                vec![v(8 + k)],
                vec![v(4 + k), v(4 + k)],
            ))
            .unwrap();
        }
        b.control(Instruction::new("s_endpgm", vec![], vec![]))
            .unwrap();
        b.finish().map(|e| ((), e))
    };
    let ((), kernel) = compact_vgprs(build).unwrap();
    let waits: Vec<&str> = kernel
        .program
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .filter(|t| t.starts_with("s_wait_loadcnt"))
        .collect();
    assert_eq!(waits, ["s_wait_loadcnt 0x0"]);
}
