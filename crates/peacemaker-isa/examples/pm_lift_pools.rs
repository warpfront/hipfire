// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab, CPU only: convert every rows-6 template (MegaM0 `templates_r6.json`) from
//! its standalone object and emit one P0 pool dispatcher per blockDim, each
//! hosting every converted template of that blockDim as a region.
//!
//! `pm_lift_pools TEMPLATES.json STANDALONE_DIR OUT_DIR [CARD_VERIFIED_TIDS]`
//! writes `OUT_DIR/p0_r6_pool_b<B>.co` and `OUT_DIR/templates_r6.json` (MegaM0's
//! fields plus `pool`, `pool_symbol`, `hidden_args`, `receipt`, and
//! `card_verified` for the comma-separated tids proven byte-identical on a card).

#[path = "lift_common/mod.rs"]
mod lift_common;

use serde_json::{json, Value};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(templates), Some(standalone), Some(out)) = (args.get(1), args.get(2), args.get(3))
    else {
        eprintln!(
            "usage: pm_lift_pools TEMPLATES.json STANDALONE_DIR OUT_DIR [CARD_VERIFIED_TIDS]"
        );
        std::process::exit(2);
    };
    let verified: Vec<u32> = args
        .get(4)
        .map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_default();
    let t: Value = serde_json::from_str(&std::fs::read_to_string(templates).expect("templates"))
        .expect("json");
    let out = Path::new(out);
    std::fs::create_dir_all(out).expect("out");
    let converted = lift_common::convert_all(&t, Path::new(standalone));
    let pools = lift_common::build_pools(&converted);
    let mut pool_of = std::collections::BTreeMap::new();
    for (b, symbol, elf, tids) in &pools {
        match elf {
            Ok(elf) => {
                let file = format!("p0_r6_pool_b{b}.co");
                std::fs::write(out.join(&file), elf).expect("write pool");
                println!("{symbol}: {} hosted {tids:?} -> {file}", tids.len());
                for tid in tids {
                    pool_of.insert(*tid, (format!("b{b}"), symbol.clone(), file.clone()));
                }
            }
            Err(e) => println!("{symbol}: FAILED {e}"),
        }
    }
    let mut rows = Vec::new();
    for tmpl in t.as_array().expect("array") {
        let tid = tmpl["tid"].as_u64().unwrap_or(0) as u32;
        let mut row = tmpl.clone();
        let Some(c) = converted.iter().find(|c| c.tid == tid) else {
            rows.push(row);
            continue;
        };
        let (mode, legal, reason) = match (&c.region, pool_of.get(&tid)) {
            (Ok(_), Some(_)) => ("LIFT_NATIVE", true, String::new()),
            (Ok(_), None) => ("ILLEGAL", false, "pool emission failed".to_owned()),
            (Err(e), _) => ("ILLEGAL", false, e.clone()),
        };
        row["mode"] = json!(mode);
        row["legal"] = json!(legal);
        row["reason"] = json!(reason);
        row["card_verified"] = json!(verified.contains(&tid));
        if let Some((pool, symbol, file)) = pool_of.get(&tid) {
            row["pool"] = json!(pool);
            row["pool_symbol"] = json!(symbol);
            row["kernel"] = json!(file);
        }
        if let Ok(r) = &c.region {
            let rc = &r.receipt;
            row["lds_bytes"] = json!(r.lds_bytes);
            row["kernarg_bytes"] = json!(r.kernarg_bytes);
            row["hidden_args"] = json!(rc
                .hidden_args
                .iter()
                .map(|h| json!({"kind": h.kind, "offset": h.offset, "size": h.size}))
                .collect::<Vec<_>>());
            row["receipt"] = json!({
                "base_object": c.object, "base_object_sha256": rc.base_object_sha256.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "waves": rc.waves, "exits_converted": rc.exits_converted, "dealloc_removed": rc.dealloc_removed,
                "input_rewrites": rc.input_rewrites, "barriers": rc.barriers, "uniform_branches": rc.uniform_branches,
                "divergent_branches": rc.divergent_branches, "barrier_participation": rc.barrier_participation,
                "live_in": rc.live_in, "undefined_live_in": rc.undefined_live_in, "rsrc1_modes": format!("{:#x}", rc.rsrc1_modes),
                "carried_verbatim": rc.carried_verbatim, "carried_substituted": rc.carried_substituted,
                "branches_retargeted": rc.branches_retargeted, "sgprs": r.sgprs, "vgprs": r.vgprs
            });
        }
        rows.push(row);
    }
    std::fs::write(
        out.join("templates_r6.json"),
        serde_json::to_string_pretty(&rows).unwrap(),
    )
    .expect("write templates");
}
