// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Offline lift probe for hipcc code objects (PM-KRSI-1100 LiftProbe).
//! Usage: x1_lift_probe OBJ.hsaco...
//! For each object: `lift_object` verdict (the Rule and VA on failure), then per kernel
//! the instruction census (s_delay_alu, VOPD, s_clause, waits, barriers, WMMA ...),
//! block count, descriptor resources, `emit::checked` (re-emit + relift) and `analyze`
//! obligations (the facts the edit API re-establishes after every edit).

use peacemaker_ir::edit::analyze;
use peacemaker_ir::inst::{Abi, Frontend};
use peacemaker_lift::{emit, lift_object, Options};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        println!(
            "== {path} ({} bytes, sha256 {:x})",
            bytes.len(),
            Sha256::digest(&bytes)
        );
        let lifted = match lift_object(
            &bytes,
            Options {
                frontend: Frontend::Hipcc,
            },
        ) {
            Ok(l) => l,
            Err(e) => {
                println!("LIFT FAIL: {e}");
                continue;
            }
        };
        let program = lifted.program;
        let arch = program.target.arch;
        println!(
            "LIFT OK: {} kernel(s), arch {arch:?}",
            program.kernels.len()
        );
        match emit::checked(
            &program,
            Options {
                frontend: Frontend::Hipcc,
            },
        ) {
            Ok((out, _)) => println!("emit::checked OK, identical to input: {}", out == bytes),
            Err(e) => println!("emit::checked FAIL: {e}"),
        }
        let only = std::env::var("X1_ONLY").ok();
        for kernel in &program.kernels {
            if only.as_deref().is_some_and(|o| kernel.symbol.0 != o) {
                continue;
            }
            let mut hist: BTreeMap<&'static str, usize> = BTreeMap::new();
            for id in &kernel.body.layout {
                let inst = kernel.body.insts.get(*id).unwrap();
                let name = inst.op.name(arch).unwrap_or("?");
                *hist.entry(name).or_default() += 1;
            }
            let sum = |f: &dyn Fn(&str) -> bool| -> usize {
                hist.iter().filter(|(k, _)| f(k)).map(|(_, v)| *v).sum()
            };
            let (vgpr, lds) = match &kernel.abi {
                Abi::Hsa { descriptor, .. } => (
                    descriptor.compute_pgm_rsrc1.next_free_vgpr(true),
                    descriptor.group_segment_fixed_size,
                ),
                _ => (0, 0),
            };
            println!(
                "- {}: insts {} blocks {} vgprs {} lds {} | delay_alu {} vopd {} clause {} waitcnt {} (lgkm/vm/other) barrier {} wmma {} branches {}",
                kernel.symbol.0,
                kernel.body.layout.len(),
                kernel.body.blocks.len(),
                vgpr,
                lds,
                sum(&|n| n == "s_delay_alu"),
                sum(&|n| n.starts_with("v_dual_")),
                sum(&|n| n == "s_clause"),
                sum(&|n| n.starts_with("s_waitcnt") || n.starts_with("s_wait_")),
                sum(&|n| n.starts_with("s_barrier")),
                sum(&|n| n.starts_with("v_wmma")),
                sum(&|n| n.starts_with("s_cbranch") || n == "s_branch"),
            );
            if std::env::var("X1_HIST").is_ok() {
                for (k, v) in &hist {
                    println!("    {v:6} {k}");
                }
            }
            if std::env::var("X1_ANALYZE").is_ok() {
                match analyze(program.clone(), &kernel.symbol) {
                    Ok(a) => {
                        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
                        for o in &a.obligations {
                            *kinds
                                .entry(format!("{:?}/{}", o.kind, o.rule_id))
                                .or_default() += 1;
                        }
                        println!(
                            "    analyze OK, {} obligations {kinds:?}",
                            a.obligations.len()
                        );
                    }
                    Err(e) => println!("    analyze FAIL: {e}"),
                }
            }
        }
    }
}
