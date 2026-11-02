// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use serde_json::Value;
use std::fmt::Write;
fn number(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap()
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let mut text = String::new();
    if args[1] == "--payoff" {
        let root: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
        println!("| Mode | Arch/remainder | NPU raw TOPS | GPU fold ratio | bytes/kop | share% | gain% (ESTIMATE) |\n|---|---|---:|---:|---:|---:|---:|");
        for row in root["rows"].as_array().unwrap() {
            println!(
                "| {} | {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.2} |",
                row["contract_mode"].as_str().unwrap(),
                row["arch"].as_str().unwrap(),
                number(row, "npu_raw_tops_assumed"),
                number(row, "gpu_fold_ratio"),
                number(row, "npu_bytes_per_kop"),
                100.0 * number(row, "optimal_ops_share"),
                number(row, "gain_pct")
            );
        }
        return;
    }
    for path in &args[2..] {
        let root: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        writeln!(text, "### {}\n", path).unwrap();
        writeln!(text,"Sample bytes {}; weight/activation FNV64 {}/{}; scale rounding loss15/16 {}/{} of {}.\n",root["sampled_bytes"],root["sample_weight_fnv64"],root["sample_activation_fnv64"],root["weight_scale_loss_15"],root["weight_scale_loss_16"],root["weight_scales"]).unwrap();
        writeln!(text,"| Contract | mean/max rel current | mean/max rel f64 | NRMSE current/f64 | max abs current |\n|---|---:|---:|---:|---:|").unwrap();
        for name in ["ief15", "ief16", "bf16mac", "group4", "once"] {
            let s = &root["total"][name];
            writeln!(
                text,
                "| {name} | {:.6e} / {:.6e} | {:.6e} / {:.6e} | {:.6e} / {:.6e} | {:.6e} |",
                number(s, "mean_rel_current"),
                number(s, "max_rel_current"),
                number(s, "mean_rel_f64"),
                number(s, "max_rel_f64"),
                number(s, "nrmse_current"),
                number(s, "nrmse_f64"),
                number(s, "max_abs")
            )
            .unwrap();
        }
        writeln!(
            text,
            "\nCurrent NRMSE vs f64: {:.6e}; outputs {}; zero mismatches {}.\n",
            number(&root["total"]["ief15"], "current_nrmse_f64"),
            root["total"]["ief15"]["n"],
            root["total"]["ief15"]["zero_mismatch"]
        )
        .unwrap();
        for key in ["gemm_kinds", "layers"] {
            writeln!(text,"#### IEF15 {key}\n\n| Group | mean/max rel current | mean/max rel f64 | NRMSE current/f64 |\n|---|---:|---:|---:|").unwrap();
            let mut rows = root[key].as_object().unwrap().iter().collect::<Vec<_>>();
            if key == "layers" {
                rows.sort_by_key(|(k, _)| k.parse::<usize>().unwrap());
            }
            for (name, v) in rows {
                let s = &v["ief15"];
                writeln!(
                    text,
                    "| {name} | {:.6e} / {:.6e} | {:.6e} / {:.6e} | {:.6e} / {:.6e} |",
                    number(s, "mean_rel_current"),
                    number(s, "max_rel_current"),
                    number(s, "mean_rel_f64"),
                    number(s, "max_rel_f64"),
                    number(s, "nrmse_current"),
                    number(s, "nrmse_f64")
                )
                .unwrap();
            }
            writeln!(text).unwrap();
        }
    }
    std::fs::write(&args[1], text).unwrap();
}
