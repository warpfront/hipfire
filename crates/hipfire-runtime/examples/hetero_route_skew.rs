//! Routing skew of a Qwen4 (Flash-Next) route trace: how much routed traffic
//! the hottest experts of each layer carry.
//!
//! ```text
//! HIPFIRE_QWEN4_ROUTE_TRACE=/tmp/route.txt hipfire run qwen3.8:flash-next ...
//! cargo run -p hipfire-runtime --example hetero_route_skew -- /tmp/route.txt [fractions...]
//! ```

use hipfire_runtime::hetero_placement::{coverage_of_top, parse_route_trace, RouteShape};

/// Flash-Next routing geometry.
const FLASH_NEXT: RouteShape = RouteShape { n_layers: 48, top_k: 10, n_experts: 512 };

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
}
