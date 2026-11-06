// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared by the lab `pm_lift_*` examples: converting MegaM0's rows-6 templates
//! from their standalone objects and building one pool dispatcher per blockDim.

use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::lifted_dispatch_gfx1201 as ld;
use peacemaker_isa::{native, Arch};
use peacemaker_lift::rewrite::region::{check_carried, lift_region, LiftedRegion, RegionInputs};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// MegaM0's COPY template id.
pub const COPY_TMPL: u32 = 2;

/// One template: its id, MegaM0 name, eager symbol, blockDim, conversion (or the
/// refused obligation) and the object it came from.
pub struct Converted {
    pub tid: u32,
    pub name: String,
    pub symbol: String,
    pub block: u32,
    pub object: Option<std::path::PathBuf>,
    pub region: Result<LiftedRegion, String>,
}

/// Convert every template of `templates` found in `standalone/*/sa.co`. The
/// template's `lds_bytes` above the descriptor's static size is dynamic LDS.
pub fn convert_all(templates: &Value, standalone: &Path) -> Vec<Converted> {
    let mut objects = Vec::new();
    for entry in std::fs::read_dir(standalone)
        .expect("standalone dir")
        .flatten()
    {
        let path = entry.path().join("sa.co");
        if let Ok(bytes) = std::fs::read(&path) {
            let symbols: Vec<String> = peacemaker_lift::lift_object(
                &bytes,
                peacemaker_lift::Options {
                    frontend: Frontend::Hipcc,
                },
            )
            .map(|l| {
                l.program
                    .kernels
                    .iter()
                    .map(|k| k.symbol.0.clone())
                    .collect()
            })
            .unwrap_or_default();
            objects.push((path, bytes, symbols));
        }
    }
    let mut out = Vec::new();
    for t in templates.as_array().expect("templates array") {
        let Some(symbol) = t["symbol"].as_str() else {
            continue;
        };
        let tid = t["tid"].as_u64().unwrap_or(0) as u32;
        let block = t["B"].as_u64().unwrap_or(0) as u32;
        let name = t["name"].as_str().unwrap_or(symbol).to_owned();
        let found = objects
            .iter()
            .find(|(_, _, s)| s.iter().any(|k| k == symbol));
        let region = match found {
            None => Err("no standalone object".to_owned()),
            Some((_, bytes, _)) => {
                let inputs = |dynamic_lds| RegionInputs {
                    prefix: format!(".Lt{tid}b"),
                    workgroup_x: ld::WG_X,
                    workgroup_yz: ld::WG_YZ,
                    block,
                    dynamic_lds,
                };
                lift_region(bytes, symbol, &inputs(0))
                    .map_err(|e| e.reason)
                    .and_then(|r| {
                        let want = t["lds_bytes"].as_u64().unwrap_or(0) as u32;
                        if want > r.lds_bytes {
                            lift_region(bytes, symbol, &inputs(want - r.lds_bytes))
                                .map_err(|e| e.reason)
                        } else {
                            Ok(r)
                        }
                    })
            }
        };
        out.push(Converted {
            tid,
            name,
            symbol: symbol.to_owned(),
            block,
            object: found.map(|(p, _, _)| p.clone()),
            region,
        });
    }
    out
}

/// One pool per blockDim over the converted templates: `(B, symbol, elf, hosted tids)`;
/// every hosted region is re-lifted from the pool object (`check_carried`).
pub fn build_pools(
    converted: &[Converted],
) -> Vec<(u32, String, Result<Vec<u8>, String>, Vec<u32>)> {
    let mut by_block: BTreeMap<u32, Vec<&Converted>> = BTreeMap::new();
    for c in converted.iter().filter(|c| c.region.is_ok()) {
        by_block.entry(c.block).or_default().push(c);
    }
    by_block
        .into_iter()
        .map(|(b, cs)| {
            let symbol = format!("pm_pool_b{b}");
            let tids: Vec<u32> = cs.iter().map(|c| c.tid).collect();
            let hosted: Vec<ld::Hosted> = cs
                .iter()
                .map(|c| ld::Hosted {
                    tmpl: c.tid,
                    region: c.region.as_ref().expect("converted"),
                })
                .collect();
            // COPY (tmpl 2, M0CopyArgs) is hosted by the 256-thread pool
            let copy = (b == 256).then_some(COPY_TMPL);
            let elf = ld::pool(&symbol, b, &hosted, copy)
                .and_then(|e| {
                    native::assemble_with_regions(&e.s_text, Arch::Gfx1201, &e.proof.regions)
                })
                .and_then(|elf| {
                    for h in &hosted {
                        check_carried(&elf, &symbol, h.region)?;
                    }
                    Ok(elf)
                });
            (b, symbol, elf, tids)
        })
        .collect()
}
