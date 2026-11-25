//! `hipfire-conformance`: `--self-test` (CPU-only, host fake state),
//! `--classes` (print the class table ids), and
//! `--card ROWS.json [REPORT.json]` (card rows through the real daemon).

use hipfire_conformance::verdict::{peak_rss_kib, ClassTable};
use hipfire_conformance::{card, selftest};
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
                println!(
                    "{} {}: {}",
                    if c.ok { "ok  " } else { "FAIL" },
                    c.name,
                    c.detail
                );
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
                println!(
                    "{}\t{}\t{}",
                    i.id,
                    i.bounded_difference.as_deref().unwrap_or("exact only"),
                    i.name
                );
            }
            ExitCode::SUCCESS
        }
        Some("--card") => {
            let args: Vec<String> = std::env::args().skip(2).collect();
            let Some(rows) = args.first() else {
                eprintln!("usage: hipfire-conformance --card ROWS.json [REPORT.json]");
                return ExitCode::from(2);
            };
            let file: card::CardFile = match std::fs::read_to_string(rows)
                .map_err(|e| e.to_string())
                .and_then(|t| card::expand_env(&t))
                .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("{rows}: {e}");
                    return ExitCode::from(2);
                }
            };
            let t0 = Instant::now();
            let results = card::run_card_file(&ClassTable::embedded(), &file);
            let mut bad = 0;
            for r in &results {
                let verdict = r.report.as_ref().map_or("refused", |x| x.verdict.name());
                let where_ = r
                    .report
                    .as_ref()
                    .and_then(|x| x.first_difference.as_ref())
                    .map(|d| {
                        format!(
                            " at step {} {} row {:?} owner {}",
                            d.step, d.component, d.row, d.owner
                        )
                    })
                    .unwrap_or_default();
                println!(
                    "{} {}: {verdict}{where_}{}",
                    if r.ok { "ok  " } else { "FAIL" },
                    r.name,
                    r.refusal
                        .as_deref()
                        .map(|e| format!(" ({e})"))
                        .unwrap_or_default()
                );
                bad += usize::from(!r.ok);
            }
            println!(
                "wall_ms {:.1} peak_rss_kib {}",
                t0.elapsed().as_secs_f64() * 1e3,
                peak_rss_kib().map_or("unknown".into(), |k| k.to_string())
            );
            if let Some(out) = args.get(1) {
                let text = serde_json::to_string_pretty(&results).expect("report serializes");
                if let Err(e) = std::fs::write(out, text) {
                    eprintln!("{out}: {e}");
                    return ExitCode::FAILURE;
                }
            }
            if bad == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        _ => {
            eprintln!(
                "usage: hipfire-conformance --self-test | --classes | --card ROWS.json [REPORT.json]"
            );
            ExitCode::from(2)
        }
    }
}
