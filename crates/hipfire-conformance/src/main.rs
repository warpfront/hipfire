//! `hipfire-conformance`: `--self-test` (CPU-only, host fake state) and
//! `--classes` (print the class table ids). Card rows follow B2.

use hipfire_conformance::selftest;
use hipfire_conformance::verdict::{peak_rss_kib, ClassTable};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("--self-test") => {
            let t0 = Instant::now();
            let cases = selftest::run_all();
            let mut bad = 0;
            for c in &cases {
                println!("{} {}: {}", if c.ok { "ok  " } else { "FAIL" }, c.name, c.detail);
                bad += usize::from(!c.ok);
            }
            println!(
                "wall_ms {:.1} peak_rss_kib {}",
                t0.elapsed().as_secs_f64() * 1e3,
                peak_rss_kib().map_or("unknown".into(), |k| k.to_string())
            );
            if bad == 0 {
                println!("self-test: pass");
                ExitCode::SUCCESS
            } else {
                println!("self-test: {bad} case(s) failed");
                ExitCode::FAILURE
            }
        }
        Some("--classes") => {
            for i in ClassTable::embedded().invariant {
                println!("{}\t{}\t{}", i.id, i.bounded_difference.as_deref().unwrap_or("exact only"), i.name);
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: hipfire-conformance --self-test | --classes");
            ExitCode::from(2)
        }
    }
}
