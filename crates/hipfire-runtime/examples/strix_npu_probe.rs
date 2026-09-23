use hipfire_runtime::npu::{probe_strix_npu, DEFAULT_ACCEL_PATH, DEFAULT_MIN_MEMLOCK_BYTES};

fn main() {
    let mut json = false;
    let mut accel_path = DEFAULT_ACCEL_PATH.to_string();
    let mut min_memlock = DEFAULT_MIN_MEMLOCK_BYTES;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--accel" => {
                accel_path = args.next().expect("--accel requires a path");
            }
            "--min-memlock-bytes" => {
                let v = args.next().expect("--min-memlock-bytes requires a value");
                min_memlock = v.parse().expect("--min-memlock-bytes must be an integer");
            }
            "-h" | "--help" => {
                println!("usage: strix_npu_probe [--json] [--accel PATH] [--min-memlock-bytes N]");
                return;
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    let report = match probe_strix_npu(&accel_path, min_memlock) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("strix_npu_probe: {e}");
            std::process::exit(1);
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("Strix/XDNA NPU probe");
        println!("  accel node: {}", report.accel.path.display());
        println!("  exists: {}", report.accel.exists);
        println!("  char device: {}", report.accel.is_char_device);
        println!(
            "  mode: {:04o} uid={} gid={}",
            report.accel.mode, report.accel.uid, report.accel.gid
        );
        match report.memlock.soft_bytes {
            Some(v) => println!("  memlock soft: {v} bytes"),
            None => println!("  memlock soft: unlimited"),
        }
        println!("  xrt-smi found: {}", report.xrt_smi.found);
        println!("  xrt-smi status: {:?}", report.xrt_smi.status);
        println!("  readiness: {:?}", report.readiness);
        for line in &report.advice {
            println!("  advice: {line}");
        }
        if !report.xrt_smi.output.trim().is_empty() {
            println!("\nxrt-smi output:\n{}", report.xrt_smi.output.trim());
        }
    }

    if !report.ready() {
        std::process::exit(2);
    }
}
