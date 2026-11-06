// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `pm-census` CLI. Every subcommand prints one JSON document on stdout.

fn usage() -> ! {
    eprintln!("pm-census — PM/railgun bedrock developer tooling");
    eprintln!();
    eprintln!("  pm-census inventory [--repo DIR]                  embedded PM object inventory");
    eprintln!("  pm-census census ...                              see `pm-census census --help`");
    eprintln!("  pm-census ledger ...                              retained-PM4 dispatch ledger -> CensusV1 (see `pm-census ledger --help`)");
    eprintln!("  pm-census extract ...                             see `pm-census extract --help`");
    eprintln!("  pm-census extract nodes ...                       trace + config + hfq -> node corpus (see `pm-census extract nodes --help`)");
    eprintln!("  pm-census roofline <corpus.json> [--what-if K=PIPE,...]   per-token MACs by op class/weight group, peak ceilings");
    eprintln!("  pm-census queue <corpus.json>... [--top N] [--md NODE-QUEUE.md]   leaves ranked by share x (1 - floor/measured)");
    eprintln!("  pm-census noise ...                               see `pm-census noise --help`");
    eprintln!("  pm-census tune plan|run|export ...                R4-R6 autotuner (see `pm-census tune`)");
    eprintln!("  pm-census tune kld|promote|trial-table ...        R7 KLD/state gate (see `pm-census tune`)");
    eprintln!("  pm-census tune kld --stat token-bootstrap ...     R9 token-level KLD statistic over HFKTOK per-token files");
    eprintln!("  pm-census tune kld-power --sd SD --margin M ...   R9 chunks needed for a margin / minimum resolvable margin");
    eprintln!("  pm-census tune selector-receipt --record R.json ... --constant N --path F --value V --out O  pm.selector_receipt.v1 from hip_loop --rowclasses records");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        usage();
    };
    let result = match cmd.as_str() {
        "inventory" => pm_census::inventory::cli(rest),
        "census" => pm_census::census::cli(rest),
        "extract" => pm_census::extract::cli(rest),
        "roofline" => pm_census::ingest::cli_roofline(rest),
        "queue" => pm_census::ingest::cli_queue(rest),
        "ledger" => pm_census::ledger::cli(rest),
        "noise" => pm_census::noise::cli(rest),
        "tune" => match rest.split_first() {
            Some((sub, tail)) if sub == "export" => pm_census::tune::store::cli(tail),
            Some((sub, _)) if sub == "selector-receipt" => pm_census::tune::rowclass::cli(rest),
            Some((sub, _))
                if matches!(
                    sub.as_str(),
                    "kld" | "kld-power" | "promote" | "trial-table"
                ) =>
            {
                pm_census::tune::kld::cli(rest)
            }
            _ => pm_census::tune::run::cli(rest),
        },
        _ => usage(),
    };
    if let Err(e) = result {
        eprintln!("pm-census {cmd}: {e}");
        std::process::exit(1);
    }
}
