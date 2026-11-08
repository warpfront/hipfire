// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The unchanged committed hipcc objects `kernels/lane_leviathan_hipcc_gfx120{0,1}.co`: the metadata kernel
//! `lane_leviathan_verify` calls the out-of-line helper `llv_tail` through `s_getpc_b64`/`s_swappc_b64`. These tests
//! cover the helper loader (`peacemaker_ir::helpers`) and the emulator's construction over kernel plus helper; they do
//! not execute the kernel.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::helpers::{helper_extents, load_helpers};
use peacemaker_ir::inst::{Arch, Frontend, KernelOrigin, Program};
use peacemaker_ir::operand::{ImmField, Operand};
use peacemaker_ir::provenance::Source;
use peacemaker_lift::{lift_object, Options};

const SYMBOL: &str = "lane_leviathan_verify";
const TAIL: &str = "_ZL8llv_tailR9LlvSharedPjPfPKjiiiffifi";
const OBJECTS: [(&[u8], Arch); 2] = [
    (
        include_bytes!("../../../kernels/lane_leviathan_hipcc_gfx1200.co"),
        Arch::Gfx1200,
    ),
    (
        include_bytes!("../../../kernels/lane_leviathan_hipcc_gfx1201.co"),
        Arch::Gfx1201,
    ),
];

fn lift(object: &[u8]) -> Program {
    lift_object(
        object,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("the committed hipcc object lifts")
    .program
}

#[test]
fn leviathan_has_exactly_one_helper_extent() {
    for (object, arch) in OBJECTS {
        let program = lift(object);
        assert_eq!(program.target.arch, arch);
        let extents = helper_extents(&program.source.as_ref().expect("lifted source").elf).unwrap();
        assert_eq!(extents.len(), 1, "{arch:?}: {extents:?}");
        assert_eq!(extents[0].start, 0x1b00);
        assert_eq!(extents[0].size, 0x49b4);
        assert_eq!(extents[0].names.iter().collect::<Vec<_>>(), [TAIL]);
    }
}

#[test]
fn leviathan_helper_is_typed_decoded_at_its_original_addresses() {
    for (object, arch) in OBJECTS {
        let program = lift(object);
        let helpers = load_helpers(&program).unwrap();
        assert_eq!(helpers.len(), 1, "{arch:?}");
        let h = &helpers[0];
        assert_eq!(h.names.iter().collect::<Vec<_>>(), [TAIL]);
        assert_eq!((h.entry_va, h.size), (0x1b00, 0x49b4));
        assert_eq!(h.insts.len(), 3603, "{arch:?}");
        let kernel_sha = match &program.kernels[0].origin {
            KernelOrigin::Frontend { object_sha256, .. } => *object_sha256,
            KernelOrigin::Authored { .. } => panic!("lifted kernel"),
        };
        // Every instruction carries its address, its words, and the object digest.
        let mut va = h.entry_va;
        for inst in &h.insts {
            assert_eq!(inst.prov.pc, Some(va as u32));
            assert!(inst.prov.bytes.is_some());
            assert!(matches!(&inst.prov.source, Source::Object(sha) if sha.0 == kernel_sha));
            let n = peacemaker_ir::codec::gfx12::encode_for(arch, inst)
                .unwrap()
                .len();
            va += 4 * n as u64;
        }
        assert_eq!(
            va,
            h.entry_va + h.size,
            "instructions tile the extent exactly"
        );
        let name_at = |i: usize| h.insts[i].op.name(arch);
        assert_eq!(name_at(0), Some("s_wait_loadcnt_dscnt"));
        let last = h.insts.last().unwrap();
        assert_eq!(last.op.name(arch), Some("s_setpc_b64"));
        assert_eq!(last.prov.pc, Some(0x64b0));
        assert_eq!(
            last.prov.bytes.unwrap()[0],
            0xBE80_481E,
            "s_setpc_b64 s[30:31]"
        );
        // The helper's only branches are SOPP offsets as decoded (no CFG labels).
        for inst in &h.insts {
            let name = inst.op.name(arch).unwrap();
            if name == "s_branch" || name.starts_with("s_cbranch_") {
                assert!(
                    matches!(inst.operands.as_slice(), [Operand::Imm(ImmField::Sopp(_))]),
                    "{name}: {:?}",
                    inst.operands
                );
            }
        }
    }
}

#[test]
fn emulator_constructs_over_kernel_and_helper() {
    for (object, arch) in OBJECTS {
        let program = lift(object);
        let helpers = load_helpers(&program).unwrap();
        // Every opcode of kernel and helper has a semantics row, every helper branch lands in the helper.
        Emulator::with_helpers(&program, SYMBOL, &helpers)
            .unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        // The kernel alone also constructs: its call then fails closed at run time.
        Emulator::new(&program, SYMBOL).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    }
}

#[test]
fn gfx1200_object_runs_only_under_the_opt_in() {
    let (object, arch) = OBJECTS[0];
    assert_eq!(arch, Arch::Gfx1200);
    let program = lift(object);
    let helpers = load_helpers(&program).unwrap();
    let emu = Emulator::with_helpers(&program, SYMBOL, &helpers).unwrap();
    let launch = Launch {
        kernarg_ptr: 0,
        workgroup: [0, 0, 0],
        block: [32, 1, 1],
        dynamic_lds: 0,
    };
    let e = emu
        .run_workgroup(&launch, &mut Memory::default())
        .unwrap_err();
    assert!(e.contains("opt in with gfx1200_as_gfx1201"), "{e}");
    // Under the opt-in the first failure is the kernel's own (no captured kernargs/memory), not the admission one.
    let e = Emulator::with_helpers(&program, SYMBOL, &helpers)
        .unwrap()
        .gfx1200_as_gfx1201()
        .run_workgroup(&launch, &mut Memory::default())
        .unwrap_err();
    assert!(!e.contains("opt in with gfx1200_as_gfx1201"), "{e}");
}

#[test]
fn helpers_need_the_lifted_source() {
    let mut program = lift(OBJECTS[1].0);
    program.source = None;
    let e = load_helpers(&program).unwrap_err();
    assert!(e.contains("no source"), "{e}");
}
