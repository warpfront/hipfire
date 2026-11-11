// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLVM `FeatureUserSGPRInit16Bug` (gfx1100/gfx1102, wave32 compute): "Bug requiring at least 16 user+system SGPRs
//! to be enabled". `SITargetLowering::allocateSystemSGPRs` pads the user SGPRs with dead inputs until
//! `user + (workgroup id X/Y/Z + info) >= 16` (the private-segment wave offset is not counted), so the system
//! SGPRs sit after the padding. The builder does the same from `KernelSpec::declared_user_sgprs`, and every
//! committed gfx1100 PM object must satisfy the rule.
use peacemaker_isa::arch::{WgIdSource, WgIds};
use peacemaker_isa::insn::Instruction;
use peacemaker_isa::reg::Live;
use peacemaker_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};

fn spec(arch: Arch, y: bool) -> KernelSpec {
    KernelSpec {
        kernel_id: "user_sgpr16".into(),
        variant: "probe".into(),
        arch,
        symbol: "user_sgpr16_probe".into(),
        kernargs: KernargLayout::new(8).pointer("arg", 0),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: y,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

fn emit(arch: Arch, y: bool, z: bool) -> String {
    let mut regs = RegPlan::new(1, 16).unwrap();
    regs.v::<1>("workitem_id", 0, Live::Whole).unwrap();
    regs.s::<2>("kernarg", 0, Live::Whole).unwrap();
    let mut b = Builder::new(spec(arch, y), regs);
    if z {
        b.enable_workgroup_id_z();
    }
    b.control(Instruction::new("s_endpgm", vec![], vec![]))
        .unwrap();
    b.finish().unwrap().s_text
}

#[test]
fn only_gfx1100_has_the_rule() {
    assert_eq!(Arch::Gfx1100.min_user_system_sgprs(), 16);
    for arch in [Arch::Gfx1151, Arch::Gfx1201, Arch::Gfx942] {
        assert_eq!(arch.min_user_system_sgprs(), 0, "{arch:?}");
        assert_eq!(arch.declared_user_sgprs(2, 3), 2, "{arch:?}");
    }
}

#[test]
fn declared_user_sgprs_pad_to_sixteen_like_llvm() {
    // user + system = 16 exactly; more user SGPRs than the padding needs are kept.
    for (user, system, declared) in [
        (2, 1, 15),
        (2, 2, 14),
        (2, 3, 13),
        (2, 4, 12),
        (6, 1, 15),
        (13, 3, 13),
        (14, 3, 14),
        (16, 1, 16),
    ] {
        assert_eq!(
            Arch::Gfx1100.declared_user_sgprs(user, system),
            declared,
            "user {user} system {system}"
        );
        assert!(Arch::Gfx1100.declared_user_sgprs(user, system) + system >= 16);
    }
}

#[test]
fn workgroup_ids_follow_the_declared_user_sgprs() {
    let ids = |arch, y, z| spec(arch, y).workgroup_ids(z).unwrap();
    // X, Y: s14, s15. X only: s15. X, Y, Z: s13, s14, s15 (z follows Y).
    assert_eq!(
        ids(Arch::Gfx1100, true, false),
        WgIds {
            x: WgIdSource::Sgpr(14),
            y: WgIdSource::Sgpr(15)
        }
    );
    assert_eq!(
        ids(Arch::Gfx1100, false, false).x,
        WgIdSource::Sgpr(15),
        "X only"
    );
    assert_eq!(
        ids(Arch::Gfx1100, true, true),
        WgIds {
            x: WgIdSource::Sgpr(13),
            y: WgIdSource::Sgpr(14)
        }
    );
    // Unchanged elsewhere: ids right after the two user SGPRs; gfx1201 delivers them in ttmps.
    assert_eq!(
        ids(Arch::Gfx1151, true, true),
        WgIds {
            x: WgIdSource::Sgpr(2),
            y: WgIdSource::Sgpr(3)
        }
    );
    assert_eq!(ids(Arch::Gfx1201, true, true).x, WgIdSource::Ttmp9);
}

#[test]
fn builder_declares_the_padded_user_sgpr_count() {
    for (y, z, declared) in [
        (false, false, 15),
        (true, false, 14),
        (true, true, 13),
        (false, true, 14),
    ] {
        let text = emit(Arch::Gfx1100, y, z);
        assert!(
            text.contains(&format!(".amdhsa_user_sgpr_count {declared}\n")),
            "y={y} z={z}"
        );
        // User + system SGPRs are live inputs: the descriptor covers all sixteen.
        assert!(text.contains(".amdhsa_next_free_sgpr 16\n"), "y={y} z={z}");
        let elf = peacemaker_isa::native::assemble(&text, Arch::Gfx1100).unwrap();
        assert!(!elf.is_empty());
    }
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        assert!(emit(arch, true, false).contains(".amdhsa_user_sgpr_count 2\n"));
    }
}

#[test]
fn native_assembly_refuses_a_gfx1100_descriptor_below_sixteen() {
    let good = emit(Arch::Gfx1100, true, false);
    let bad = good.replace(
        ".amdhsa_user_sgpr_count 14\n",
        ".amdhsa_user_sgpr_count 2\n",
    );
    assert_ne!(good, bad);
    let err = peacemaker_isa::native::assemble(&bad, Arch::Gfx1100).unwrap_err();
    assert!(
        err.contains("FeatureUserSGPRInit16Bug") && err.contains("2 user + 2 system = 4"),
        "{err}"
    );
    // The same descriptor is legal where the bug does not exist.
    let halo = emit(Arch::Gfx1151, true, false);
    assert!(halo.contains(".amdhsa_user_sgpr_count 2\n"));
    peacemaker_isa::native::assemble(&halo, Arch::Gfx1151).unwrap();
}

#[test]
fn descriptor_rule_counts_workgroup_ids_and_info_but_not_the_private_segment() {
    // rsrc2 bits 1..5 USER_SGPR_COUNT, 7/8/9/10 ENABLE_SGPR_WORKGROUP_ID_X/Y/Z/INFO, 0 PRIVATE_SEGMENT.
    let rsrc2 = |user: u32, system_bits: u32, private: u32| user << 1 | system_bits << 7 | private;
    let ok = |r| Arch::Gfx1100.check_user_system_sgprs("k", r).is_ok();
    assert!(ok(rsrc2(14, 0b11, 0)));
    assert!(ok(rsrc2(13, 0b111, 0)));
    assert!(ok(rsrc2(12, 0b1111, 0)));
    assert!(!ok(rsrc2(2, 0b11, 0)));
    assert!(!ok(rsrc2(14, 0b01, 0)));
    // LLVM does not count the private-segment wave offset toward 16.
    assert!(!ok(rsrc2(13, 0b11, 1)));
    assert!(Arch::Gfx1151
        .check_user_system_sgprs("k", rsrc2(2, 0b11, 0))
        .is_ok());
}

/// Every committed gfx1100 PM object (all kernels of every `kernels/*_pm_gfx1100*.hxaco`) declares >= 16 user+system
/// SGPRs, checked on the object bytes through the lifter's descriptor.
#[test]
fn every_committed_gfx1100_pm_object_declares_sixteen_user_and_system_sgprs() {
    use peacemaker_ir::inst::{Abi, Frontend};
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut objects = 0;
    let mut kernels = 0;
    let mut bad = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains("_pm_gfx1100") && n.ends_with(".hxaco"))
        })
        .collect();
    entries.sort();
    for path in entries {
        let bytes = std::fs::read(&path).unwrap();
        let lifted = peacemaker_lift::lift_object(
            &bytes,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
        objects += 1;
        for k in &lifted.program.kernels {
            let Abi::Hsa { descriptor, .. } = &k.abi else {
                continue;
            };
            kernels += 1;
            if let Err(e) =
                Arch::Gfx1100.check_user_system_sgprs(&k.symbol.0, descriptor.compute_pgm_rsrc2.0)
            {
                bad.push(format!("{}: {e}", path.display()));
            }
        }
    }
    assert!(
        objects >= 13 && kernels >= 30,
        "{objects} objects {kernels} kernels"
    );
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
