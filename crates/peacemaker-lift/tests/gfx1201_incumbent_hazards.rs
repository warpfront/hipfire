// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The two gfx1201 hipcc incumbent code objects (`lane_proj`, wave64; `res_lanes`, wave32) must
//! pass the static hazard checkers cleanly, and the checkers must actually *verify* the
//! `s_wait_alu` guards hipcc emitted rather than ignore them: neutralising a guard makes a
//! Required finding appear.
//!
//! The objects live outside the repo. `PM_INCUMBENT_DIR` overrides the default directory;
//! when an object is missing the test prints a SKIP line and passes.

use peacemaker_ir::inst::{Arch, Frontend, Kernel, Program, Wave};
use peacemaker_ir::operand::{ImmField, Operand};
use peacemaker_ir::passes::hazards::{self, SgprModel};
use peacemaker_lift::hazards_gfx11::{check_kernel, Severity};
use peacemaker_lift::{lift_object, Options};
use std::collections::BTreeMap;

const DEFAULT_DIR: &str = "/home/kaden/qcal/release-0.4.3/pm-krsi-lane/tmp/inc";

fn lift(name: &str) -> Option<Program> {
    let dir = std::env::var("PM_INCUMBENT_DIR").unwrap_or_else(|_| DEFAULT_DIR.into());
    let path = std::path::Path::new(&dir).join(name);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!(
            "SKIP: incumbent object {} is missing (set PM_INCUMBENT_DIR)",
            path.display()
        );
        return None;
    };
    let lifted = lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Some(lifted.program)
}

/// Every `s_wait_alu` of a kernel: (instruction, original `simm16`).
fn wait_alus(kernel: &Kernel, arch: Arch) -> Vec<(peacemaker_ir::cfg::InstId, i16)> {
    kernel
        .body
        .layout
        .iter()
        .filter_map(|id| {
            let inst = kernel.body.insts.get(*id)?;
            if inst.op.name(arch) != Some("s_wait_alu") {
                return None;
            }
            inst.operands.iter().find_map(|op| match op {
                Operand::Imm(ImmField::Sopp(bits)) => Some((*id, *bits)),
                _ => None,
            })
        })
        .collect()
}

fn set_wait(kernel: &mut Kernel, id: peacemaker_ir::cfg::InstId, bits: i16) {
    let inst = kernel.body.insts.get_mut(id).expect("live instruction");
    for op in &mut inst.operands {
        if let Operand::Imm(ImmField::Sopp(n)) = op {
            *n = bits;
        }
    }
}

/// `s_wait_alu 0xffff`: every counter at its no-wait value.
const NO_WAIT: i16 = -1;

fn check_clean(program: &Program, expect_wave: Wave, expect_kernels: usize) {
    assert_eq!(program.target.arch, Arch::Gfx1201);
    assert_eq!(program.kernels.len(), expect_kernels);
    let mut advisory = BTreeMap::<&str, usize>::new();
    for kernel in &program.kernels {
        assert_eq!(kernel.wave, expect_wave, "{}", kernel.symbol.0);
        let findings = check_kernel(kernel, Arch::Gfx1201, true);
        for f in &findings {
            match f.severity {
                Severity::Advisory => *advisory.entry(f.rule).or_default() += 1,
                _ => panic!(
                    "{}: {} finding {} at {:#x}: {} / {}",
                    kernel.symbol.0,
                    f.severity.as_str(),
                    f.rule,
                    f.pc,
                    f.required,
                    f.present
                ),
            }
        }
        // The pipeline-default replay (wave64: LLVM-exact, wave32: builder parity) agrees.
        let analysis = hazards::analyze(&kernel.body, Arch::Gfx1201, kernel.wave)
            .unwrap_or_else(|e| panic!("{}: {e}", kernel.symbol.0));
        assert!(
            analysis.missing.is_empty() && analysis.obligations.is_empty(),
            "{}: {analysis:?}",
            kernel.symbol.0
        );
    }
    eprintln!("advisory findings: {advisory:?}");
}

#[test]
fn gfx1201_incumbent_lane_proj_wave64_is_clean() {
    let Some(program) = lift("lane_proj.elf") else {
        return;
    };
    check_clean(&program, Wave::Wave64, 8);
}

#[test]
fn gfx1201_incumbent_res_lanes_wave32_is_clean() {
    let Some(program) = lift("res_lanes.elf") else {
        return;
    };
    check_clean(&program, Wave::Wave32, 1);
}

/// Neutralise every guard: the checker must now demand them, and name the kinds hipcc
/// emitted (`sa_sdst` and `va_sdst` for lane_proj; `sa_sdst` and, via the LLVM-exact VCC
/// model, `va_vcc` for res_lanes).
fn waits_are_verified(name: &str, kinds: &[&str]) {
    let Some(mut program) = lift(name) else {
        return;
    };
    for kernel in &mut program.kernels {
        let waits = wait_alus(kernel, Arch::Gfx1201);
        assert!(
            !waits.is_empty(),
            "{} carries no s_wait_alu",
            kernel.symbol.0
        );
        for (id, _) in &waits {
            set_wait(kernel, *id, NO_WAIT);
        }
        let found = hazards::analyze_with(
            &kernel.body,
            Arch::Gfx1201,
            kernel.wave,
            SgprModel::LlvmExact,
        )
        .unwrap();
        for kind in kinds {
            assert!(
                found.missing.iter().any(|m| m.wait == *kind),
                "{}: no {kind} demand after removing {} guards",
                kernel.symbol.0,
                waits.len()
            );
        }
        let checked = check_kernel(kernel, Arch::Gfx1201, false);
        assert!(
            checked
                .iter()
                .any(|f| f.rule == "gfx12-valu-read-sgpr" && f.severity == Severity::Required),
            "{}",
            kernel.symbol.0
        );
    }
}

#[test]
fn gfx1201_incumbent_lane_proj_waits_are_verified() {
    waits_are_verified("lane_proj.elf", &["depctr_sa_sdst(0)", "depctr_va_sdst(0)"]);
}

#[test]
fn gfx1201_incumbent_res_lanes_waits_are_verified() {
    waits_are_verified("res_lanes.elf", &["depctr_sa_sdst(0)", "depctr_va_vcc(0)"]);
}

/// Removing guards one at a time: LLVM's pass emits only waits it demands, so a faithful
/// replay finds every one of hipcc's guards load-bearing (27 of 27 in the n1 kernel).
#[test]
fn gfx1201_incumbent_lane_proj_single_guards_are_load_bearing() {
    let Some(mut program) = lift("lane_proj.elf") else {
        return;
    };
    let kernel = &mut program.kernels[0];
    let waits = wait_alus(kernel, Arch::Gfx1201);
    let mut needed = 0usize;
    for (id, bits) in &waits {
        set_wait(kernel, *id, NO_WAIT);
        let found = hazards::analyze_with(
            &kernel.body,
            Arch::Gfx1201,
            kernel.wave,
            SgprModel::LlvmExact,
        )
        .unwrap();
        needed += usize::from(!found.missing.is_empty());
        set_wait(kernel, *id, *bits);
    }
    eprintln!(
        "{}: {needed} of {} s_wait_alu are individually load-bearing",
        kernel.symbol.0,
        waits.len()
    );
    assert_eq!(needed, waits.len(), "load-bearing guards");
}
