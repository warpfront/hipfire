// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Offline production compile + lift status of gfx1100 Flash-Next decode modules (PM-KRSI-1100
//! DecodeAudit). No device. Every module goes through `kernel_registry::lookup("gfx1100", ..)`
//! (the source the runtime registry hands the JIT) and `KernelCompiler::compile_to_paths` (the
//! runtime JIT's argv: core flags, ROCm flags, module flags, scheduler profile, per-source flag
//! comments). The effective-source sha256 is compared with the closed route plan row
//! (`route_plans/qwen3.8-flash-next.gfx1100.tsv`, column 5) so the object is provably the module
//! the XTX daemon loaded.
//!
//! Usage: decode_audit_compile OUT_DIR PLAN_TSV [MODULE...]
//!   no MODULE: every `ar` hip row of the plan.
//! Writes OUT_DIR/objs/<module>.hsaco + .hip, prints one JSON line per module and appends to
//! OUT_DIR/compile.jsonl: module, plan symbols, source sha256 vs plan sha256, object sha256, argv,
//! lift verdict (`lift_object`, hipcc frontend) and per kernel symbol: instruction count, vgprs,
//! LDS, opcodes without emulator semantics.

use peacemaker_ir::inst::{Abi, Frontend};
use rdna_compute::{kernel_registry, KernelCompiler};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .expect("usage: decode_audit_compile OUT_DIR PLAN_TSV [MODULE...]"),
    );
    let plan = args.next().expect("PLAN_TSV");
    let wanted: Vec<String> = args.collect();
    let objs = out.join("objs");
    std::fs::create_dir_all(&objs).unwrap();
    let cache = out.join("kernel-cache");
    std::fs::create_dir_all(&cache).unwrap();
    std::env::set_var("HIPFIRE_KERNEL_CACHE", &cache);
    std::env::set_var("HIPFIRE_HOME", out.join("home"));
    // plan rows: route kind module symbols sha256
    let mut rows: BTreeMap<String, (Vec<String>, String)> = BTreeMap::new();
    for line in std::fs::read_to_string(&plan).unwrap().lines() {
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() == 5 && f[0] == "ar" && f[1] == "hip" {
            rows.insert(
                f[2].to_owned(),
                (
                    f[3].split(',').map(str::to_owned).collect(),
                    f[4].to_owned(),
                ),
            );
        }
    }
    let modules: Vec<String> = if wanted.is_empty() {
        rows.keys().cloned().collect()
    } else {
        wanted
    };
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("compile.jsonl"))
        .unwrap();
    for module in modules {
        let rec = one(&module, &rows, &objs);
        let line = rec.to_string();
        println!("{line}");
        writeln!(log, "{line}").unwrap();
    }
}

fn one(module: &str, rows: &BTreeMap<String, (Vec<String>, String)>, objs: &PathBuf) -> Value {
    let (plan_symbols, plan_sha) = rows.get(module).cloned().unwrap_or_default();
    let entry = match kernel_registry::lookup("gfx1100", module, "") {
        Ok(e) => e,
        Err(e) => {
            return json!({"module": module, "error": format!("registry: {e:?}"), "plan_symbols": plan_symbols})
        }
    };
    let source = entry.source().to_owned();
    let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
    let kc = KernelCompiler::new("gfx1100", String::new()).expect("compiler");
    let src = objs.join(format!("{module}.hip"));
    let obj = objs.join(format!("{module}.hsaco"));
    let _ = std::fs::remove_file(&obj);
    if let Err(e) = kc.compile_to_paths(module, &source, &src, &obj) {
        return json!({"module": module, "error": format!("compile: {e}"), "source_sha256": source_sha});
    }
    let argv = kc.jit_argv(module, &source, &src, &obj);
    let bytes = std::fs::read(&obj).unwrap();
    let mut rec = json!({
        "module": module,
        "registry_symbols": entry.symbols,
        "plan_symbols": plan_symbols,
        "source_sha256": source_sha,
        "plan_sha256": plan_sha,
        "source_matches_plan": !plan_sha.is_empty() && plan_sha == source_sha,
        "registry_flags": entry.flags,
        "scheduler_profile": entry.scheduler_profile,
        "object": obj.display().to_string(),
        "object_bytes": bytes.len(),
        "object_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "argv": argv,
    });
    match peacemaker_lift::lift_object(
        &bytes,
        peacemaker_lift::Options {
            frontend: Frontend::Hipcc,
        },
    ) {
        Err(e) => {
            rec["lift_ok"] = json!(false);
            rec["lift_error"] = json!(format!("{e}"));
        }
        Ok(l) => {
            rec["lift_ok"] = json!(true);
            let program = l.program;
            let arch = program.target.arch;
            let mut kernels = Vec::new();
            for k in &program.kernels {
                let mut missing: BTreeMap<String, usize> = BTreeMap::new();
                for id in &k.body.layout {
                    let i = k.body.insts.get(*id).unwrap();
                    let name = i.op.name(arch).unwrap_or("?").to_owned();
                    if peacemaker_ir::emu::semantic_class(&name).is_none() {
                        *missing.entry(name).or_default() += 1;
                    }
                }
                let (vgpr, lds, kargs, kseg) = match &k.abi {
                    Abi::Hsa { descriptor, metadata } => (
                        descriptor.compute_pgm_rsrc1.next_free_vgpr(true),
                        descriptor.group_segment_fixed_size,
                        metadata
                            .parsed
                            .args
                            .iter()
                            .map(|a| json!({"name": a.name, "offset": a.offset, "size": a.size, "kind": a.value_kind}))
                            .collect::<Vec<_>>(),
                        metadata.parsed.kernarg_segment_size,
                    ),
                    _ => (0, 0, Vec::new(), 0),
                };
                kernels.push(json!({
                    "symbol": k.symbol.0,
                    "insts": k.body.layout.len(),
                    "vgprs": vgpr,
                    "static_lds": lds,
                    "kernarg_segment_size": kseg,
                    "kernargs": kargs,
                    "no_emu_semantics": missing,
                }));
            }
            rec["kernels"] = json!(kernels);
        }
    }
    rec
}
