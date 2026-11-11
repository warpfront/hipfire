// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared descriptor opt-in: no assembler, object generation or GPU required.
use peacemaker_isa::insn::Instruction;
use peacemaker_isa::reg::Live;
use peacemaker_isa::{emit, Arch, Builder, KernargLayout, KernelSpec, RegPlan};

fn probe(arch: Arch, y: bool, sgpr_budget: u16) -> Builder {
    let mut regs = RegPlan::new(1, sgpr_budget).unwrap();
    regs.v::<1>("workitem_id", 0, Live::Whole).unwrap();
    regs.s::<2>("kernarg", 0, Live::Whole).unwrap();
    Builder::new(
        KernelSpec {
            kernel_id: "workgroup_id_z".into(),
            variant: "probe".into(),
            arch,
            symbol: "workgroup_id_z_probe".into(),
            kernargs: KernargLayout::new(0),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: y,
            workgroup_size: 256,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    )
}

fn end(builder: &mut Builder) {
    builder
        .control(Instruction::new("s_endpgm", vec![], vec![]))
        .unwrap();
}

#[test]
fn default_descriptor_and_proof_keep_legacy_output() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        for y in [false, true] {
            let mut builder = probe(arch, y, 16);
            end(&mut builder);
            let legacy = emit::assembly(&builder.spec, &builder.regs, builder.program()).unwrap();
            let emitted = builder.finish().unwrap();
            assert_eq!(emitted.s_text, legacy, "{arch:?} y={y}");
            assert!(emitted
                .s_text
                .contains(".amdhsa_system_sgpr_workgroup_id_z 0\n"));
            assert!(!emitted.proof.system_sgpr_workgroup_id_z);
            assert!(serde_json::to_value(&emitted.proof)
                .unwrap()
                .get("system_sgpr_workgroup_id_z")
                .is_none());
        }
    }
}

#[test]
fn workgroup_z_accounts_for_architectural_input_footprint() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        for y in [false, true] {
            let mut builder = probe(arch, y, 16);
            builder.enable_workgroup_id_z();
            builder.enable_workgroup_id_z(); // Idempotent, with no instruction insertion.
            assert!(builder.program().instructions.is_empty());
            end(&mut builder);
            let emitted = builder.finish().unwrap();
            // gfx1100 pads user + system to 16 (LLVM FeatureUserSGPRInit16Bug), so the inputs end at s16.
            let high = if arch.gfx12() {
                2
            } else if arch == Arch::Gfx1100 {
                16
            } else {
                4 + u16::from(y)
            };
            assert!(emitted
                .s_text
                .contains(".amdhsa_system_sgpr_workgroup_id_z 1\n"));
            assert!(emitted
                .s_text
                .contains(&format!(".amdhsa_next_free_sgpr {high}\n")));
            assert!(emitted.s_text.contains(&format!(".sgpr_count: {high}\n")));
            assert_eq!(emitted.shape.next_free_sgpr, high);
            assert_eq!(emitted.proof.next_free_sgpr, high);
            assert!(emitted.proof.system_sgpr_workgroup_id_z);
        }
    }
}

#[test]
fn workgroup_z_only_changes_descriptor_when_inputs_are_already_planned() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        let mut legacy = probe(arch, true, 16);
        legacy.regs.s::<1>("last_scalar", 15, Live::Whole).unwrap();
        let mut opted_in = legacy.clone();
        opted_in.enable_workgroup_id_z();
        end(&mut legacy);
        end(&mut opted_in);
        let legacy = legacy.finish().unwrap();
        let opted_in = opted_in.finish().unwrap();
        let mut expected = legacy.s_text.replace(
            ".amdhsa_system_sgpr_workgroup_id_z 0\n",
            ".amdhsa_system_sgpr_workgroup_id_z 1\n",
        );
        if arch == Arch::Gfx1100 {
            // The padding that makes user + system 16 shrinks by the one more system SGPR.
            expected = expected.replace(
                ".amdhsa_user_sgpr_count 14\n",
                ".amdhsa_user_sgpr_count 13\n",
            );
        }
        assert_eq!(opted_in.s_text, expected, "{arch:?}");
    }
}

#[test]
fn workgroup_z_entry_sgprs_obey_budget_without_charging_gfx12_ttmp7() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        let mut builder = probe(arch, true, 4);
        builder.enable_workgroup_id_z();
        end(&mut builder);
        let result = builder.finish();
        if arch.gfx12() {
            assert!(result.is_ok());
        } else {
            assert_eq!(result.unwrap_err(), "register plan exceeds budget");
        }
    }
}

#[test]
fn final_scalar_operand_high_water_matches_shape_and_proof() {
    let mut builder = probe(Arch::Gfx1151, true, 8);
    builder.enable_workgroup_id_z();
    // Imported instructions may use operands beyond the authored RegPlan.
    builder
        .push(Instruction::new("s_mov_b32 s7, 0", vec![], vec![]))
        .unwrap();
    end(&mut builder);
    let emitted = builder.finish().unwrap();
    assert!(emitted.s_text.contains(".amdhsa_next_free_sgpr 8\n"));
    assert_eq!(emitted.shape.next_free_sgpr, 8);
    assert_eq!(emitted.proof.next_free_sgpr, 8);
}
