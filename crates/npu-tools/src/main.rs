// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use npu_tools::{aie_status, gpu_bw, m4, soc_metrics, tblgen, txn_dump};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        eprintln!(
            "usage: npu-tools aie-status|txn-dump|soc-metrics|m4|tblgen2rs|gpu-bw|fclk ARGS..."
        );
        std::process::exit(2);
    };
    if matches!(command.as_str(), "--help" | "-h") {
        println!(
            "usage: npu-tools aie-status|txn-dump|soc-metrics|m4|tblgen2rs|gpu-bw|fclk ARGS..."
        );
        return;
    }
    let result = match command.as_str() {
        "aie-status" => aie_status::run(rest).map(|()| 0),
        "txn-dump" => txn_dump::run(rest).map(|()| 0),
        "soc-metrics" => soc_metrics::run(rest).map(|()| 0),
        "m4" => m4::run(rest),
        "tblgen2rs" => tblgen::run(rest).map(|()| 0),
        "gpu-bw" => gpu_bw::run(rest).map(|()| 0),
        "fclk" => railgun::npu::fclk::cli(rest),
        _ => {
            eprintln!("unknown npu-tools subcommand {command:?}");
            std::process::exit(2)
        }
    };
    match result {
        Ok(status) => std::process::exit(status),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1)
        }
    }
}
