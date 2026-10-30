//! Developer qualification of native MTP with exact Flash-Next LRU.
//! Default PLAN_ONLY. --cpu-controls never opens a GPU. --execute --phase
//! collector uses tiny fixtures and the production route collector, not a model.
//! Model proof is explicit and cannot pass from AR fallback or counters alone.
#[path = "fn_mtp_lru/collector.rs"]
mod collector;
#[path = "fn_mtp_lru/model.rs"]
mod model;
#[path = "fn_mtp_lru/receipts.rs"]
mod receipts;
use receipts::{ensure, R};
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

struct Args {
    execute: bool,
    phase: String,
    out: Option<PathBuf>,
    model: Option<PathBuf>,
    prompts: Vec<PathBuf>,
    max_seq: usize,
    tokens: usize,
    k: usize,
}
fn parse() -> R<Args> {
    let mut a = Args {
        execute: false,
        phase: "collector".into(),
        out: None,
        model: None,
        prompts: Vec::new(),
        max_seq: 16384,
        tokens: 512,
        k: 3,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--execute" => a.execute = true,
            "--cpu-controls" => {
                a.phase = "cpu".into();
                a.execute = true;
            }
            "--phase" => a.phase = it.next().ok_or("--phase needs collector|model")?,
            "--out" => a.out = Some(it.next().ok_or("--out needs path")?.into()),
            "--model" => a.model = Some(it.next().ok_or("--model needs path")?.into()),
            "--prompt-files" => {
                a.prompts = it
                    .next()
                    .ok_or("--prompt-files needs comma paths")?
                    .split(',')
                    .map(PathBuf::from)
                    .collect()
            }
            "--max-seq" => {
                a.max_seq = it
                    .next()
                    .ok_or("--max-seq needs integer")?
                    .parse()
                    .map_err(|_| "invalid max-seq")?
            }
            "--tokens" => {
                a.tokens = it
                    .next()
                    .ok_or("--tokens needs integer")?
                    .parse()
                    .map_err(|_| "invalid tokens")?
            }
            "--k" => {
                a.k = it
                    .next()
                    .ok_or("--k needs integer")?
                    .parse()
                    .map_err(|_| "invalid k")?
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    ensure(
        matches!(a.phase.as_str(), "collector" | "model" | "cpu"),
        "invalid phase",
    )?;
    ensure(
        (2..8).contains(&a.k) && a.tokens >= 16,
        "strong acceptance coverage requires K2..7 and at least16 generated tokens",
    )?;
    if a.execute && a.phase == "model" {
        ensure(
            a.model.is_some() && !a.prompts.is_empty(),
            "model phase requires --model and --prompt-files",
        )?;
    }
    Ok(a)
}
fn main() {
    let result = (|| -> R<(Value, Option<PathBuf>)> {
        let a = parse()?;
        // Claim output before GPU/model initialization; never overwrite old proof.
        let mut file = match &a.out {
            Some(path) => Some(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(|e| format!("output {}: {e}", path.display()))?,
            ),
            None => None,
        };
        let mut report = json!({"schema":"hipfire-fn-mtp-lru-oracle-v1","phase":a.phase,"performance_claim":false,"complete":false,"qualification":"none","status":"PLAN_ONLY"});
        let outcome = if !a.execute {
            Ok(())
        } else if a.phase == "cpu" {
            receipts::cpu_controls().map(|v| {
                report["evidence"] = v;
            })
        } else if a.phase == "collector" {
            match rdna_compute::Gpu::init() {
                Ok(mut gpu) => match collector::run(&mut gpu) {
                    Ok(v) => {
                        report["evidence"] = v;
                        Ok(())
                    }
                    Err(e) => {
                        std::mem::forget(gpu);
                        Err(e)
                    }
                },
                Err(error) => Err(error.to_string()),
            }
        } else {
            model::run(&a).map(|v| {
                report["evidence"] = v;
            })
        };
        if a.execute {
            report["status"] = json!(if outcome.is_ok() { "PASS" } else { "FAIL" });
            report["complete"] = json!(outcome.is_ok());
            report["qualification"] = json!(if outcome.is_ok() {
                a.phase.as_str()
            } else {
                "none"
            });
        }
        if let Err(e) = &outcome {
            report["error"] = json!(e);
        }
        if let Some(f) = file.as_mut() {
            serde_json::to_writer_pretty(&mut *f, &report).map_err(|e| e.to_string())?;
            f.write_all(b"\n").map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
        }
        // Failure evidence has already been published before returning nonzero.
        println!("{report}");
        outcome?;
        Ok((report, a.out))
    })();
    if let Err(e) = result {
        eprintln!("fn_mtp_lru_oracle: {e}");
        std::process::exit(1);
    }
}
