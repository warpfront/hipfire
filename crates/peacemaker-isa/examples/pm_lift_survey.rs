// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab: region-conversion survey of the rows-6 layer bodies. For every template
//! of `templates_<R>.json` (MegaM0) whose eager symbol is found in one of the
//! standalone objects under `DIR/*/sa.co`, run
//! `peacemaker_lift::rewrite::region::lift_region` at the template's blockDim
//! and print the receipt (or the refused obligation) as one JSON line each.
//!
//! `pm_lift_survey TEMPLATES.json STANDALONE_DIR`

use peacemaker_ir::inst::Frontend;
use peacemaker_lift::rewrite::region::{lift_region, RegionInputs};
use serde_json::{json, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(templates), Some(dir)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: pm_lift_survey TEMPLATES.json STANDALONE_DIR");
        std::process::exit(2);
    };
    let t: Value = serde_json::from_str(&std::fs::read_to_string(templates).expect("templates"))
        .expect("json");
    let mut objects = Vec::new();
    for entry in std::fs::read_dir(dir).expect("dir").flatten() {
        let path = entry.path().join("sa.co");
        if let Ok(bytes) = std::fs::read(&path) {
            let symbols: Vec<String> = match peacemaker_lift::lift_object(
                &bytes,
                peacemaker_lift::Options {
                    frontend: Frontend::Hipcc,
                },
            ) {
                Ok(l) => l
                    .program
                    .kernels
                    .iter()
                    .map(|k| k.symbol.0.clone())
                    .collect(),
                Err(e) => {
                    println!(
                        "{}",
                        json!({"object": path, "status": "lift failed", "reason": e.to_string()})
                    );
                    vec![]
                }
            };
            objects.push((path, bytes, symbols));
        }
    }
    for tmpl in t.as_array().expect("array") {
        let Some(symbol) = tmpl["symbol"].as_str() else {
            continue;
        };
        let block = tmpl["B"].as_u64().unwrap_or(0) as u32;
        let Some((path, bytes, _)) = objects
            .iter()
            .find(|(_, _, s)| s.iter().any(|k| k == symbol))
        else {
            println!(
                "{}",
                json!({"tid": tmpl["tid"], "symbol": symbol, "status": "no standalone object"})
            );
            continue;
        };
        let inputs = RegionInputs {
            prefix: format!(".Lt{}", tmpl["tid"]),
            workgroup_x: 104,
            workgroup_yz: 105,
            block,
            dynamic_lds: 0,
        };
        let line = match lift_region(bytes, symbol, &inputs) {
            Ok(r) => {
                let pool = if r.sgprs <= peacemaker_isa::kernels::lifted_dispatch_gfx1201::WG_X {
                    let sym = format!("pm_lift_t{}", tmpl["tid"]);
                    let lifted = RegionInputs {
                        workgroup_x: peacemaker_isa::kernels::lifted_dispatch_gfx1201::WG_X,
                        workgroup_yz: peacemaker_isa::kernels::lifted_dispatch_gfx1201::WG_YZ,
                        ..inputs.clone()
                    };
                    lift_region(bytes, symbol, &lifted)
                        .map_err(|e| e.to_string())
                        .and_then(|r| {
                            let e = peacemaker_isa::kernels::lifted_dispatch_gfx1201::pool(
                                &sym,
                                block,
                                &[peacemaker_isa::kernels::lifted_dispatch_gfx1201::Hosted {
                                    tmpl: tmpl["tid"].as_u64().unwrap_or(0) as u32,
                                    region: &r,
                                }],
                                None,
                            )?;
                            let elf = peacemaker_isa::native::assemble_with_regions(
                                &e.s_text,
                                peacemaker_isa::Arch::Gfx1201,
                                &e.proof.regions,
                            )?;
                            peacemaker_lift::rewrite::region::check_carried(&elf, &sym, &r)
                                .map(|n| format!("ok: {n} carried"))
                        })
                        .unwrap_or_else(|e| format!("pool failed: {e}"))
                } else {
                    format!("{} SGPRs reach the region inputs", r.sgprs)
                };
                let c = &r.receipt;
                json!({"tid": tmpl["tid"], "symbol": symbol, "object": path, "status": "converted", "pool": pool, "B": block,
                    "waves": c.waves, "lds": r.lds_bytes, "kernarg": r.kernarg_bytes, "sgprs": r.sgprs, "vgprs": r.vgprs,
                    "exits": c.exits_converted, "dealloc": c.dealloc_removed, "barriers": c.barriers,
                    "uniform_branches": c.uniform_branches, "divergent_branches": c.divergent_branches,
                    "participation": c.barrier_participation, "live_in": c.live_in, "undefined_live_in": c.undefined_live_in,
                    "rewrites": c.input_rewrites, "hidden": c.hidden_args.iter().map(|h| format!("{}@{}", h.kind, h.offset)).collect::<Vec<_>>(),
                    "verbatim": c.carried_verbatim, "substituted": c.carried_substituted, "rsrc1_modes": format!("{:#x}", c.rsrc1_modes)})
            }
            Err(e) => {
                json!({"tid": tmpl["tid"], "symbol": symbol, "object": path, "status": "refused", "B": block, "reason": e.reason})
            }
        };
        println!("{line}");
    }
}
