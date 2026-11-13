// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder APIs that take registers outside an instruction, under the VGPR compaction's second pass.

use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::reg::{Kind, Live, RegRef};
use peacemaker_isa::regalloc::{compact_vgprs, declare_vgprs, Pass};
use peacemaker_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};

fn spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "vgpr_map_probe".into(),
        variant: "default".into(),
        arch: Arch::Gfx1201,
        symbol: "vgpr_map_probe".into(),
        kernargs: KernargLayout::new(8),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

fn v(base: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len: 1,
    }
}

fn plan(pass: Pass) -> RegPlan {
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

/// `settle_defs` names the emitter's registers: pass 2 must settle the renamed register that the next definition
/// writes, so the wait lands where pass 1 put it.
#[test]
fn settle_defs_translates_to_the_renamed_register() {
    let build = |pass: Pass| {
        let mut b = Builder::new(spec(), plan(pass));
        b.vgpr_pass(pass);
        b.push(Instruction::new("v_mov_b32 v1, 0", vec![v(1)], vec![]))
            .unwrap();
        b.push(
            Instruction::new(
                "global_load_b32 v9, v1, s[0:1]",
                vec![v(9)],
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
        b.settle_defs(&[v(9)]).unwrap();
        b.push(Instruction::new("v_mov_b32 v2, 0", vec![v(2)], vec![]))
            .unwrap();
        b.push(Instruction::new("v_mov_b32 v9, v2", vec![v(9)], vec![v(2)]))
            .unwrap();
        b.wait_all().unwrap();
        b.control(Instruction::new("s_endpgm", vec![], vec![]))
            .unwrap();
        b.finish().map(|e| ((), e))
    };
    let ((), kernel) = compact_vgprs(build).unwrap();
    let text: Vec<&str> = kernel
        .program
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    let load = text
        .iter()
        .position(|t| t.starts_with("global_load"))
        .unwrap();
    // the load's destination is renamed (v9 has no neighbour to keep it in place), yet the wait stays right after it
    assert!(!text[load].contains(" v9,"), "{text:?}");
    assert!(text[load + 1].starts_with("s_wait_loadcnt"), "{text:?}");
}
