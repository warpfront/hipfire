//! Routing skew of a Qwen4 (Flash-Next) route trace: how much routed traffic
//! the hottest experts of each layer carry.
//!
//! ```text
//! HIPFIRE_QWEN4_ROUTE_TRACE=/tmp/route.txt hipfire run qwen3.8:flash-next ...
//! cargo run -p hipfire-runtime --example hetero_route_skew -- /tmp/route.txt [fractions...] \
//!     [--budget-gib 23] [--expert-bytes N]
//! ```

use hipfire_runtime::hetero_placement::{
    coverage_of_top, parse_route_trace, plan_hot_set, RouteShape,
};

/// Flash-Next routing geometry.
const FLASH_NEXT: RouteShape = RouteShape { n_layers: 48, top_k: 10, n_experts: 512 };
/// Flash-Next MQ4 routed experts: 61 GiB over 48 x 512 experts.
const DEFAULT_EXPERT_BYTES: u64 = (61u64 << 30) / (48 * 512);
const GIB: f64 = (1u64 << 30) as f64;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take_opt = |name: &str| -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        let v = args.get(i + 1).cloned().unwrap_or_else(|| panic!("{name} needs a value"));
        args.drain(i..=i + 1);
        Some(v)
    };
    let budget_gib: Option<f64> = take_opt("--budget-gib").map(|v| v.parse().expect("--budget-gib"));
    let expert_bytes: u64 =
        take_opt("--expert-bytes").map_or(DEFAULT_EXPERT_BYTES, |v| v.parse().expect("--expert-bytes"));
    let Some(path) = args.first() else {
        eprintln!("usage: hetero_route_skew <trace-file> [fractions...]");
        std::process::exit(2);
    };
    let fractions: Vec<f64> = if args.len() > 1 {
        args[1..].iter().map(|a| a.parse().expect("fraction must be a number")).collect()
    } else {
        vec![0.25, 0.35, 0.5]
    };
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let freq = parse_route_trace(&text, FLASH_NEXT).unwrap_or_else(|e| panic!("{e}"));
    println!("tokens {} (dropped partial lines {})", freq.tokens, freq.dropped_partial_lines);
    if freq.tokens == 0 {
        return;
    }
    for f in fractions {
        let mut cov = coverage_of_top(&freq, f);
        cov.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "top {:>4.1}% of experts per layer carry: min {:.3}  median {:.3}  max {:.3}",
            f * 100.0,
            cov[0],
            cov[cov.len() / 2],
            cov[cov.len() - 1]
        );
    }
    if let Some(gib) = budget_gib {
        let bytes = vec![vec![expert_bytes; FLASH_NEXT.n_experts]; FLASH_NEXT.n_layers];
        let p = plan_hot_set(&freq, &bytes, (gib * GIB) as u64);
        println!(
            "budget {gib:.1} GiB: hot traffic share {:.3}, hot {:.2} GiB, cold {:.2} GiB",
            p.hot_traffic_share,
            p.hot_bytes as f64 / GIB,
            p.cold_bytes as f64 / GIB
        );
    }
}
