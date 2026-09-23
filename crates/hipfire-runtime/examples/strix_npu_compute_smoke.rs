use hipfire_runtime::npu::{
    run_strix_npu_smoke_report, AcceleratorNode, StrixNpuAcceleratorNode, StrixNpuSmokeConfig,
};
use std::path::PathBuf;

fn main() {
    let mut cfg = StrixNpuSmokeConfig::default_compute();
    let mut json = false;
    let mut summary_json = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--summary-json" => summary_json = true,
            "--probe-only" => cfg.run_mlir_aie_smoke = false,
            "--script" => {
                cfg.script_path = PathBuf::from(args.next().expect("--script requires a path"));
            }
            "--mlir-aie-dir" => {
                cfg.mlir_aie_dir = Some(PathBuf::from(
                    args.next().expect("--mlir-aie-dir requires a path"),
                ));
            }
            "--device-name" => {
                cfg.device_name = args.next().expect("--device-name requires a value");
            }
            "--min-memlock-bytes" => {
                let v = args.next().expect("--min-memlock-bytes requires a value");
                cfg.min_memlock_bytes = v.parse().expect("--min-memlock-bytes must be an integer");
            }
            "-h" | "--help" => {
                println!(
                    "usage: strix_npu_compute_smoke [--json] [--probe-only] [--script PATH] \\
                     [--summary-json] [--mlir-aie-dir PATH] [--device-name NAME] \\
                     [--min-memlock-bytes N]"
                );
                return;
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    if summary_json {
        let node = StrixNpuAcceleratorNode {
            min_memlock_bytes: cfg.min_memlock_bytes,
            smoke_config: cfg,
            ..Default::default()
        };
        match node.validate_compute() {
            Ok(summary) => {
                println!("{}", serde_json::to_string_pretty(&summary).unwrap());
                if !summary.daemon_usable {
                    std::process::exit(2);
                }
                return;
            }
            Err(e) => {
                eprintln!("strix_npu_compute_smoke: {e}");
                std::process::exit(1);
            }
        }
    }

    let result = run_strix_npu_smoke_report(&cfg);
    if json {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        print!("{}", result.command.output);
    }

    if result.command.status != Some(0) {
        std::process::exit(result.command.status.unwrap_or(1));
    }
}
