// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Offline ISA-gap audit of hipcc gfx1100 code objects (PM-KRSI-1100 DecodeAudit/IsaGaps). No device.
//! Usage: isa_gap_audit OBJ.hsaco...
//! Per object: `lift_object` verdict, `emit::checked` identity, and per kernel the opcodes of the lifted body that
//! have no `isa/semantics.tbl` row (the emulator refuses them at load). One summary line per object, details
//! only for failures: `OK` means lifted, re-emitted identically and every instruction has a semantics row.

use peacemaker_ir::inst::Frontend;
use peacemaker_lift::{emit, lift_object, Options};
use std::collections::BTreeMap;

fn main() {
    let mut bad = 0usize;
    for path in std::env::args().skip(1) {
        let name = path.rsplit('/').next().unwrap_or(&path).to_owned();
        let bytes = std::fs::read(&path).unwrap();
        let program = match lift_object(
            &bytes,
            Options {
                frontend: Frontend::Hipcc,
            },
        ) {
            Ok(l) => l.program,
            Err(e) => {
                println!("{name}: LIFT FAIL: {e}");
                bad += 1;
                continue;
            }
        };
        let arch = program.target.arch;
        let identical = match emit::checked(
            &program,
            Options {
                frontend: Frontend::Hipcc,
            },
        ) {
            Ok((out, _)) => {
                if out == bytes {
                    None
                } else {
                    Some("emit::checked output differs from input".to_owned())
                }
            }
            Err(e) => Some(format!("emit::checked FAIL: {e}")),
        };
        let mut missing: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        let mut total = 0usize;
        for k in &program.kernels {
            total += k.body.layout.len();
            for id in &k.body.layout {
                let i = k.body.insts.get(*id).unwrap();
                let n = i.op.name(arch).unwrap_or("?");
                if peacemaker_ir::emu::semantic_class(n).is_none() {
                    *missing
                        .entry(k.symbol.0.clone())
                        .or_default()
                        .entry(n.to_owned())
                        .or_default() += 1;
                }
            }
        }
        let ok = identical.is_none() && missing.is_empty();
        println!(
            "{name}: {} ({} kernels, {total} insts){}",
            if ok { "OK" } else { "GAPS" },
            program.kernels.len(),
            identical.map(|e| format!(" {e}")).unwrap_or_default()
        );
        for (k, ops) in &missing {
            bad += 1;
            println!("    {k}: {ops:?}");
        }
    }
    if bad != 0 {
        std::process::exit(1);
    }
}
