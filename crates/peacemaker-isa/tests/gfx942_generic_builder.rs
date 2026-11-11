// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The generic `Builder` on gfx942 (wave64): source ownership and control flow
//! are checked over the text, the checked CDNA builder owns waits, hazards and
//! emission.

use peacemaker_isa::insn::Instruction;
use peacemaker_isa::plan::{KernargLayout, KernelSpec};
use peacemaker_isa::reg::{Kind, Live, RegPlan};
use peacemaker_isa::{native, Arch, Builder};

fn spec(wave32: bool) -> KernelSpec {
    KernelSpec {
        kernel_id: "g942".into(),
        variant: "smoke".into(),
        arch: Arch::Gfx942,
        symbol: "g942_smoke".into(),
        kernargs: KernargLayout::new(8).pointer("out", 0),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 64,
        group_segment_fixed_size: 0,
        wave32,
        cu_mode: false,
    }
}

fn plan() -> RegPlan {
    let mut regs = RegPlan::new(8, 16).unwrap();
    regs.add_range("kernarg", Kind::S, 0, 2, Live::Whole)
        .unwrap();
    regs.add_range("wg", Kind::S, 2, 1, Live::Whole).unwrap();
    regs.add_range("ptr", Kind::S, 4, 2, Live::Whole).unwrap();
    regs.add_range("sc", Kind::S, 8, 1, Live::Whole).unwrap();
    regs.add_range("tid", Kind::V, 0, 1, Live::Whole).unwrap();
    regs.add_range("vals", Kind::V, 1, 2, Live::Whole).unwrap();
    regs
}

fn insn(text: &str) -> Instruction {
    Instruction::new(text, vec![], vec![])
}

fn kernel(b: &mut Builder) -> Result<(), String> {
    b.push(insn("s_load_dwordx2 s[4:5], s[0:1], 0x0"))?;
    b.push(insn("v_mov_b32 v1, 5"))?;
    b.push(insn("v_mov_b32 v2, 0"))?;
    b.push(insn("global_store_dword v2, v1, s[4:5]"))?;
    b.control(insn("s_endpgm"))
}

#[test]
fn finishes_with_cdna_waits_and_native_assembles() {
    let mut b = Builder::new(spec(false), plan());
    kernel(&mut b).unwrap();
    let emitted = b.finish().unwrap();
    assert!(
        emitted.s_text.contains("s_waitcnt lgkmcnt(0)"),
        "{}",
        emitted.s_text
    );
    assert!(emitted.s_text.contains(".amdhsa_accum_offset"));
    assert!(!emitted.s_text.contains("wavefront_size32"));
    assert!(emitted.proof.waits.iter().all(|w| {
        emitted.program.instructions[w.pc_index]
            .mnemonic()
            .starts_with("s_waitcnt")
    }));
    let elf = native::assemble(&emitted.s_text, Arch::Gfx942).unwrap();
    assert_eq!(&elf[..4], b"\x7fELF");
}

#[test]
fn branches_and_labels_lower_through_checked_methods() {
    let mut b = Builder::new(spec(false), plan());
    b.push(insn("s_mov_b32 s8, 3")).unwrap();
    b.label("loop").unwrap();
    b.push(insn("s_sub_u32 s8, s8, 1")).unwrap();
    b.push(insn("s_cmp_lg_u32 s8, 0")).unwrap();
    b.control(insn("s_cbranch_scc1 loop")).unwrap();
    b.control(insn("s_endpgm")).unwrap();
    let emitted = b.finish().unwrap();
    assert!(emitted
        .s_text
        .contains("s_cbranch_scc1 .Lg942_smoke_rip_loop"));
    assert!(emitted
        .program
        .instructions
        .iter()
        .any(|i| i.text == "s_cbranch_scc1 rip_loop"));
    native::assemble(&emitted.s_text, Arch::Gfx942).unwrap();
}

#[test]
fn rejects_wave32_spec_and_raw_waits() {
    let mut b = Builder::new(spec(true), plan());
    assert!(b
        .push(insn("v_mov_b32 v1, 5"))
        .unwrap_err()
        .contains("wave32"));
    let mut b = Builder::new(spec(false), plan());
    assert!(b.push(insn("s_waitcnt lgkmcnt(0)")).is_err());
    assert!(b.push(insn("s_barrier")).is_err());
    // RDNA-only spelling does not parse as CDNA.
    assert!(b.push(insn("s_load_b64 s[4:5], s[0:1], 0x0")).is_err());
    assert!(b.push(insn("s_wait_alu depctr_va_vdst(0)")).is_err());
}

#[test]
fn register_outside_plan_is_refused() {
    let mut b = Builder::new(spec(false), plan());
    assert!(b.push(insn("v_mov_b32 v5, 1")).is_err());
}

#[test]
fn user_sgpr_count_must_be_the_kernarg_pointer() {
    let mut s = spec(false);
    s.user_sgpr_count = 6;
    let mut b = Builder::new(s, plan());
    kernel(&mut b).unwrap();
    assert!(b.finish().is_err());
}

#[test]
fn explicit_vop3_nop_without_an_encoding_receipt_is_refused() {
    let mut b = Builder::new(spec(false), plan());
    b.label(".Lend").unwrap();
    let error = b.push(insn("v_nop_e64")).unwrap_err();
    assert!(error.contains("v_nop"), "{error}");
    assert!(error.contains("unverified encoding"), "{error}");
}
