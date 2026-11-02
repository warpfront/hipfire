// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Static hardware-hazard check of lifted code objects (gfx1100 / gfx1151 / gfx1201, wave32 and
//! wave64; see `peacemaker_lift::hazards_gfx11` for the rules and their LLVM citations).
//!
//! Usage: `hazard_check [--no-advisory] [--summary] [--rules] OBJECT[:SYMBOL] ...`
//!
//! Prints one JSON document on stdout: per object, per kernel, the findings
//! (`rule`, `pc`, the instructions involved, `required` vs `present` separation) and, for
//! cross-checking, the obligations of PM's own `peacemaker_ir::passes::hazards::analyze`
//! replay. `--summary` drops the per-finding arrays and keeps counts. `--rules` prints the
//! rule table and exits. `--dump OBJECT:SYMBOL [LO HI]` prints the lifted instruction text of a
//! kernel (optionally only pcs in `[LO, HI]`, hex) and exits. Exit status: 0 = no required/unchecked finding in any object,
//! 1 = at least one, 2 = usage / lift failure.

use peacemaker_ir::inst::{Frontend, Program, Wave};
use peacemaker_ir::passes::hazards;
use peacemaker_lift::hazards_gfx11::{check_kernel, Severity, RULES};
use peacemaker_lift::{lift_object, Options};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn rules_json() -> Value {
    Value::Array(
        RULES
            .iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "applies": r.applies,
                    "severity": r.severity.as_str(),
                    "implemented": r.implemented,
                    "llvm": r.llvm,
                    "gate": r.gate,
                    "mitigation": r.mitigation,
                })
            })
            .collect(),
    )
}

fn check_object(path: &str, symbol: Option<&str>, advisory: bool, summary: bool) -> (Value, bool) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return (json!({"object": path, "error": format!("read: {e}")}), true),
    };
    let sha = format!("{:x}", Sha256::digest(&bytes));
    let program: Program = match lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    ) {
        Ok(l) => l.program,
        Err(e) => {
            return (
                json!({"object": path, "sha256": sha, "lift_error": e.to_string()}),
                true,
            )
        }
    };
    let arch = program.target.arch;
    let mut failed = false;
    let mut kernels = Vec::new();
    for kernel in &program.kernels {
        if symbol.is_some_and(|s| kernel.symbol.0 != s) {
            continue;
        }
        let findings = check_kernel(kernel, arch, advisory);
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut per_rule: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &findings {
            *counts.entry(f.severity.as_str()).or_default() += 1;
            *per_rule.entry(f.rule).or_default() += 1;
            failed |= f.severity != Severity::Advisory;
        }
        let pm = hazards::analyze(&kernel.body, arch, kernel.wave)
            .map(|a| {
                a.obligations
                    .iter()
                    .map(|o| {
                        let pc = o
                            .insts
                            .first()
                            .and_then(|id| kernel.body.insts.get(*id))
                            .and_then(|i| i.prov.pc);
                        json!({"rule_id": o.rule_id, "pc": pc.map(|p| format!("{p:#x}")), "text": o.text})
                    })
                    .collect::<Vec<_>>()
            })
            .map_err(|e| e.to_string());
        let mut k = json!({
            "symbol": kernel.symbol.0,
            "wave": match kernel.wave { Wave::Wave32 => 32, Wave::Wave64 => 64 },
            "insts": kernel.body.layout.len(),
            "blocks": kernel.body.blocks.len(),
            "counts": counts,
            "per_rule": per_rule,
        });
        match &pm {
            Ok(v) => {
                k["pm_ir_hazard_replay"] = json!({"count": v.len(), "obligations": if summary { Value::Null } else { Value::Array(v.clone()) }});
                failed |= !v.is_empty();
            }
            Err(e) => {
                k["pm_ir_hazard_replay"] = json!({"error": e});
                failed = true;
            }
        }
        if !summary {
            k["findings"] = Value::Array(findings.iter().map(|f| f.to_json()).collect());
        }
        kernels.push(k);
    }
    (
        json!({"object": path, "sha256": sha, "arch": format!("{arch:?}"), "kernels": kernels}),
        failed,
    )
}

fn dump(args: &[String]) {
    let target = args.first().expect("--dump OBJECT:SYMBOL [LO HI]");
    let (path, symbol) = target.rsplit_once(':').expect("OBJECT:SYMBOL");
    let parse = |s: &String| u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex pc");
    let lo = args.get(1).map(parse).unwrap_or(0);
    let hi = args.get(2).map(parse).unwrap_or(u32::MAX);
    let bytes = std::fs::read(path).expect("read object");
    let program = lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift")
    .program;
    let arch = program.target.arch;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .expect("symbol not in object");
    for id in &kernel.body.layout {
        let inst = kernel.body.insts.get(*id).unwrap();
        let pc = inst.prov.pc.unwrap_or(0);
        if (lo..=hi).contains(&pc) {
            println!(
                "{pc:#x}: {}",
                inst.text(arch).unwrap_or_else(|_| "?".into())
            );
        }
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("--dump") {
        dump(&raw[1..]);
        return;
    }
    let mut advisory = true;
    let mut summary = false;
    let mut targets = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--no-advisory" => advisory = false,
            "--summary" => summary = true,
            "--rules" => {
                println!("{}", serde_json::to_string_pretty(&rules_json()).unwrap());
                return;
            }
            _ => targets.push(arg),
        }
    }
    if targets.is_empty() {
        eprintln!("usage: hazard_check [--no-advisory] [--summary] [--rules] OBJECT[:SYMBOL] ...");
        std::process::exit(2);
    }
    let mut objects = Vec::new();
    let mut failed = false;
    let mut broken = false;
    for target in &targets {
        // `path:symbol`, but paths may contain ':' only before the last one; split once from the right
        // when the suffix is not an existing file.
        let (path, symbol) = match target.rsplit_once(':') {
            Some((p, s)) if !std::path::Path::new(target).exists() => (p, Some(s)),
            _ => (target.as_str(), None),
        };
        let (value, bad) = check_object(path, symbol, advisory, summary);
        broken |= value.get("error").is_some() || value.get("lift_error").is_some();
        failed |= bad;
        objects.push(value);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "checker": "peacemaker_lift::hazards_gfx11",
            "llvm": "ROCm/llvm-project d6f6cb691863e9258034ab48c1dfce21689d5606",
            "objects": objects,
        }))
        .unwrap()
    );
    std::process::exit(if broken { 2 } else { i32::from(failed) });
}
