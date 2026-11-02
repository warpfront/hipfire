// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Checked `ds_bpermute_b32` crosslane admission (gfx1201): the W1 decode
//! twins' shfl-down reductions. The form is a DS-load-class instruction
//! with an address (lane byte index) and a data VGPR use, counts on the DS
//! queue, carries no LDS slot, and must pass the native writer and M7.
#![cfg(feature = "toolchain")]
use peacemaker_isa::{
    insn::{Instruction, MemoryClass},
    ledger::Counter,
    reg::Live,
    Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};

const BPERMUTE16: &str = "ds_bpermute_b32 v2, v0, v1 offset:64";
/// llvm-objdump's canonical spelling of swizzle offset 527 (and 0x0f, or 0x10):
/// lane l reads lane 16 | (l & 15), the hipcc W1 shfl 16 step.
const SWIZZLE16: &str = "ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,\"1pppp\")";

/// `out[lane] = in[src(lane)]` through one checked crosslane exchange.
fn probe(crosslane: &str, with_wait: bool) -> Result<peacemaker_isa::Emitted, String> {
    probe_with(
        crosslane,
        with_wait,
        KernargLayout::new(16).pointer("in", 0).pointer("out", 8),
    )
}
fn probe_with(
    crosslane: &str,
    with_wait: bool,
    kernargs: KernargLayout,
) -> Result<peacemaker_isa::Emitted, String> {
    let mut regs = RegPlan::new(8, 8)?;
    let lane = regs.v::<1>("lane", 0, Live::Whole)?;
    let data = regs.v::<1>("data", 1, Live::Whole)?;
    let out = regs.v::<1>("out", 2, Live::Whole)?;
    let ptrs = regs.s::<4>("ptrs", 0, Live::Whole)?;
    let spec = KernelSpec {
        kernel_id: "bpermute_probe".into(),
        variant: "shfl16".into(),
        arch: Arch::Gfx1201,
        symbol: "bpermute_probe".into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, regs);
    b.push(
        Instruction::new(
            "s_load_b128 s[0:3], s[0:1], 0x0",
            vec![ptrs.reg()],
            vec![ptrs.reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v0, 2, v0",
        vec![lane.reg()],
        vec![lane.reg()],
    ))?;
    b.wait(Counter::Km, 0)?;
    b.push(
        Instruction::new(
            "global_load_b32 v1, v0, s[0:1]",
            vec![data.reg()],
            vec![lane.reg(), ptrs.reg()],
        )
        .memory(MemoryClass::VmemLoad),
    )?;
    b.wait(Counter::Load, 0)?;
    let uses = if crosslane.starts_with("ds_bpermute") {
        vec![lane.reg(), data.reg()]
    } else {
        vec![data.reg()]
    };
    b.ds_crosslane(Instruction::new(crosslane, vec![out.reg()], uses).memory(MemoryClass::DsLoad))?;
    if with_wait {
        b.wait(Counter::Ds, 0)?;
    }
    b.push(
        Instruction::new(
            "global_store_b32 v0, v2, s[2:3]",
            vec![],
            vec![lane.reg(), out.reg(), ptrs.reg()],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

fn assemble_and_m7(e: &peacemaker_isa::Emitted, tag: &str) -> serde_json::Value {
    assert_eq!(
        e.proof.lds_slots.len(),
        0,
        "a crosslane exchange carries no LDS slot"
    );
    let elf = peacemaker_isa::native::assemble(&e.s_text, Arch::Gfx1201).expect("native assemble");
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("crosslane-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bpermute_probe.co");
    std::fs::write(&path, &elf).unwrap();
    let summary = peacemaker_isa::pm_check::m7(&path, "gfx1201", "bpermute_probe")
        .expect("M7 obligations {}");
    eprintln!("M7 {tag} {summary}");
    assert_eq!(summary["obligations"], serde_json::json!({}));
    assert_eq!(summary["ambiguous_delays"], 0);
    peacemaker_isa::ledger_replay::replay_waits(&e.s_text, Arch::Gfx1201)
        .expect("independent wait replay");
    std::fs::remove_dir_all(&dir).ok();
    summary
}

#[test]
fn bpermute_is_admitted_assembled_and_m7_clean() {
    let e = probe(BPERMUTE16, true).expect("checked bpermute kernel");
    assert!(e.s_text.contains(BPERMUTE16));
    assemble_and_m7(&e, "bpermute");
}

/// The canonical BITMASK_PERM spelling assembles natively to offset 527, the
/// same encoding as the numeric form, so `.s` text and llvm-objdump parse-back agree.
#[test]
fn bitmask_perm_swizzle_is_canonical_and_m7_clean() {
    let canonical = probe(SWIZZLE16, true).expect("checked swizzle kernel");
    assert!(canonical.s_text.contains(SWIZZLE16));
    assemble_and_m7(&canonical, "swizzle");
    let numeric = probe("ds_swizzle_b32 v2, v1 offset:527", true).expect("numeric swizzle");
    let elf_c = peacemaker_isa::native::assemble(&canonical.s_text, Arch::Gfx1201).unwrap();
    let elf_n = peacemaker_isa::native::assemble(&numeric.s_text, Arch::Gfx1201).unwrap();
    assert_eq!(elf_c.len(), elf_n.len());
    let word = 0xd8d4020fu32.to_le_bytes();
    assert!(
        elf_c.windows(4).any(|w| w == word),
        "canonical form encodes offset 527"
    );
    assert!(
        elf_n.windows(4).any(|w| w == word),
        "numeric form encodes offset 527"
    );
}

#[test]
fn bpermute_result_is_never_read_before_its_ds_wait() {
    // Without an explicit wait the checked builder either refuses the read or
    // retires the DS queue itself; the independent wait replay must accept the text.
    match probe(BPERMUTE16, false) {
        Err(_) => {}
        Ok(e) => {
            let after = e.s_text.split("ds_bpermute_b32").nth(1).unwrap();
            let store = after.find("global_store_b32").unwrap();
            assert!(
                after[..store].contains("s_wait_dscnt"),
                "{}",
                &after[..store]
            );
            peacemaker_isa::ledger_replay::replay_waits(&e.s_text, Arch::Gfx1201).unwrap();
        }
    }
    // The replay itself refuses a read of the exchange result before the DS wait.
    assert!(peacemaker_isa::ledger_replay::replay_waits(
        "ds_bpermute_b32 v2, v0, v1 offset:64\nv_add_f32_e32 v3, v2, v2\n",
        Arch::Gfx1201
    )
    .is_err());
}

#[test]
fn crosslane_still_refuses_lds_memory_forms() {
    let mut regs = RegPlan::new(8, 8).unwrap();
    let a = regs.v::<1>("a", 0, Live::Whole).unwrap();
    let d = regs.v::<1>("d", 1, Live::Whole).unwrap();
    let spec = KernelSpec {
        kernel_id: "k".into(),
        variant: "v".into(),
        arch: Arch::Gfx1201,
        symbol: "k".into(),
        kernargs: KernargLayout::new(8),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, regs);
    for text in ["ds_load_b32 v1, v0", "ds_permute_b32 v1, v0, v0"] {
        assert!(
            b.ds_crosslane(
                Instruction::new(text, vec![d.reg()], vec![a.reg()]).memory(MemoryClass::DsLoad)
            )
            .is_err(),
            "{text}"
        );
    }
    assert!(
        b.ds_crosslane(Instruction::new(
            "ds_bpermute_b32 v1, v0, v0",
            vec![d.reg()],
            vec![a.reg()]
        ))
        .is_err(),
        "memory class required"
    );
}

/// The frozen `.actual_access` of a twinned hipcc pointer is emitted in LLVM's
/// key order, survives the native writer and M7's byte-exact re-emission, and
/// a plain pointer's metadata text is unchanged.
#[test]
fn pointer_actual_access_metadata() {
    use peacemaker_isa::plan::Access;
    let plain = probe(BPERMUTE16, true).unwrap();
    assert!(!plain.s_text.contains(".actual_access"));
    assert!(plain
        .s_text
        .contains("      - .address_space: global\n        .name: in\n"));
    let layout = KernargLayout::new(16)
        .pointer_access("in", 0, Access::ReadOnly)
        .pointer_access("out", 8, Access::WriteOnly);
    let e = probe_with(BPERMUTE16, true, layout).unwrap();
    assert!(e.s_text.contains("      - .actual_access: read_only\n        .address_space: global\n        .name: in\n"), "{}", e.s_text);
    assert!(e.s_text.contains(
        "      - .actual_access: write_only\n        .address_space: global\n        .name: out\n"
    ));
    let summary = assemble_and_m7(&e, "access");
    assert_eq!(summary["obligations"], serde_json::json!({}));
    let elf = peacemaker_isa::native::assemble(&e.s_text, Arch::Gfx1201).unwrap();
    // Golden against a real hipcc (clang) gfx1201 object: the NT_AMDGPU_METADATA
    // MessagePack of an access-annotated pointer arg is the same key/value byte
    // run in both: fixstr `.actual_access`, fixstr value, then `.address_space: global`.
    let hipcc = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../peacemaker-lift/tests/fixtures/kt48/hipcc.co"
    ))
    .unwrap();
    let run = |value: &str| -> Vec<u8> {
        let mut v = vec![0xa0 | 14];
        v.extend_from_slice(b".actual_access");
        v.push(0xa0 | value.len() as u8);
        v.extend_from_slice(value.as_bytes());
        v.push(0xa0 | 14);
        v.extend_from_slice(b".address_space");
        v.push(0xa0 | 6);
        v.extend_from_slice(b"global");
        v
    };
    for value in ["read_only", "write_only"] {
        let needle = run(value);
        assert!(
            hipcc.windows(needle.len()).any(|w| w == needle),
            "hipcc object lacks {value} run"
        );
        assert!(
            elf.windows(needle.len()).any(|w| w == needle),
            "native object differs from hipcc for {value}"
        );
    }
}
