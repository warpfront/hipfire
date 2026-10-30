// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon oracle CLI (PLAN §6.2): module | state | logits | bench. Owner: slice O.
//!
//! This file is argument parsing, dispatch and verdict logic only. Each
//! subcommand calls `raon_oracle::<module>::run(fixtures, &opts)` in
//! `examples/raon_oracle/<module>.rs`, which drives the production launchers
//! (never a duplicate test kernel). The dispatcher refuses any module that
//! reports "not built", and refuses a result that does not carry an explicit
//! per-case pass/fail verdict: there is no placeholder pass.
//!
//! ```text
//! raon_oracle module --module mhc|rda|mla|moe|norm|head --fixtures DIR --out FILE [--negative] [--k v ...]
//! raon_oracle state  --model M --tokens F --mode serial|batched|graph|state-only --out FILE [--k v ...]
//! raon_oracle logits --model M --tokens F (--teacher F | --kld REF.kldref.bin) --out FILE [--k v ...]
//! raon_oracle generate --model M [--k v ...]   (generate::run streams text to stdout, exit 0/1; `--json 1` also prints the result object; no --out)
//! raon_oracle bench  --model M --pp 512,8192 --tg 128 --out FILE [--k v ...]
//! ```
//!
//! `--negative` points the module at `<fixtures>/negative_controls` and
//! requires every case there to FAIL (deliberately wrong references produced
//! by `scripts/raon_reference.py fixtures --negative-controls`).
//! Extra `--key value` pairs are forwarded verbatim in `opts` (module-owned
//! knobs such as `--case`, `--route`).
//!
//! Exit status: 0 verdict ok, 1 verdict failed / module error, 2 usage.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod raon_oracle {
    pub mod bench;
    pub mod generate;
    pub mod logits;
    pub mod mhc;
    pub mod mla;
    pub mod moe;
    pub mod rda;
    pub mod state;
}

type RunFn = fn(&Path, &BTreeMap<String, String>) -> Result<Value, String>;

const USAGE: &str = "usage:
  raon_oracle module --module mhc|rda|mla|moe|norm|head --fixtures DIR --out FILE [--negative]
  raon_oracle state  --model M --tokens F --mode serial|batched|graph|state-only --out FILE
  raon_oracle logits --model M (--tokens F --teacher F | --kld REF.kldref.bin) --out FILE
  raon_oracle generate --model M ... (streams text to stdout; no --out)
  raon_oracle bench  --model M --pp 512,8192 --tg 128 [--reps 3] [--warmup 1] [--profile] --out FILE
  raon_oracle bench  --bucket-json PROFILE.json --out FILE   (CPU-only re-bucketing of a saved --profile result)
(extra `--key value` pairs are forwarded to the module as opts)";

struct Cli {
    sub: String,
    opts: BTreeMap<String, String>,
    negative: bool,
}

fn parse_cli() -> Result<Cli, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let sub = argv.first().cloned().ok_or("missing subcommand")?;
    if !matches!(
        sub.as_str(),
        "module" | "state" | "logits" | "bench" | "generate"
    ) {
        return Err(format!("unknown subcommand `{sub}`"));
    }
    let mut opts = BTreeMap::new();
    let mut negative = false;
    let mut i = 1;
    while i < argv.len() {
        let a = &argv[i];
        let key = a
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument `{a}`"))?;
        if key == "negative" {
            negative = true;
            i += 1;
            continue;
        }
        if key == "profile" {
            opts.insert("profile".to_string(), "1".to_string());
            i += 1;
            continue;
        }
        let val = argv
            .get(i + 1)
            .ok_or_else(|| format!("`--{key}` needs a value"))?;
        if opts.insert(key.to_string(), val.clone()).is_some() {
            return Err(format!("`--{key}` given twice"));
        }
        i += 2;
    }
    if negative && sub != "module" {
        return Err("--negative applies to `module` only".into());
    }
    Ok(Cli {
        sub,
        opts,
        negative,
    })
}

fn require<'a>(opts: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required `--{key}`"))
}

fn parse_pp_list(s: &str) -> Result<Vec<usize>, String> {
    let v: Result<Vec<usize>, _> = s.split(',').map(|t| t.trim().parse::<usize>()).collect();
    let v = v.map_err(|e| format!("--pp `{s}`: {e}"))?;
    if v.is_empty() || v.contains(&0) {
        return Err(format!("--pp `{s}`: need positive lengths"));
    }
    Ok(v)
}

/// Resolve the verdict of a per-case result: `(ok, summary lines)`.
/// A module result must be `{"cases":[{"name":..,"pass":bool,..},..]}` with at
/// least one case; a missing/ill-typed verdict is a failure, never a pass.
/// `negative`: every case must have `pass == false`.
fn case_verdict(result: &Value, negative: bool) -> (bool, Vec<String>) {
    let mut lines = Vec::new();
    let cases = match result.get("cases").and_then(Value::as_array) {
        Some(c) if !c.is_empty() => c,
        _ => {
            lines.push("result has no non-empty `cases` array".to_string());
            return (false, lines);
        }
    };
    let mut ok = true;
    for c in cases {
        let name = c.get("name").and_then(Value::as_str).unwrap_or("<unnamed>");
        let Some(pass) = c.get("pass").and_then(Value::as_bool) else {
            lines.push(format!("  {name}: MISSING verdict `pass`"));
            ok = false;
            continue;
        };
        let good = pass != negative;
        ok &= good;
        let detail = c
            .get("detail")
            .or_else(|| c.get("first_mismatch"))
            .map(|d| format!(" {d}"))
            .unwrap_or_default();
        let tag = match (negative, pass) {
            (false, true) => "PASS",
            (false, false) => "FAIL",
            (true, false) => "NEGATIVE-CONTROL-FAILED-AS-REQUIRED",
            (true, true) => "NEGATIVE-CONTROL-WRONGLY-PASSED",
        };
        lines.push(format!("  {name}: {tag}{detail}"));
    }
    if let Some(d) = result.get("declared_cases").and_then(Value::as_u64) {
        if d as usize != cases.len() {
            lines.push(format!(
                "manifest declares {d} cases but result has {}",
                cases.len()
            ));
            ok = false;
        }
    }
    (ok, lines)
}

/// Verdict for state/logits: explicit top-level boolean `pass` (plus optional cases).
fn scalar_verdict(result: &Value) -> (bool, Vec<String>) {
    match result.get("pass").and_then(Value::as_bool) {
        Some(p) => (p, vec![format!("  pass={p}")]),
        None => (false, vec!["result lacks boolean `pass`".to_string()]),
    }
}

fn module_fn(name: &str) -> Result<RunFn, String> {
    Ok(match name {
        "mhc" => raon_oracle::mhc::run,
        "norm" => raon_oracle::mhc::run_norm,
        "head" => raon_oracle::mhc::run_head,
        "rda" => raon_oracle::rda::run,
        "mla" => raon_oracle::mla::run,
        "moe" => raon_oracle::moe::run,
        other => {
            return Err(format!(
                "unknown --module `{other}` (mhc|rda|mla|moe|norm|head)"
            ))
        }
    })
}

fn run_generate(cli: &Cli) -> Result<bool, String> {
    require(&cli.opts, "model")?;
    let fixtures = PathBuf::from(cli.opts.get("fixtures").map(String::as_str).unwrap_or("."));
    let result =
        raon_oracle::generate::run(&fixtures, &cli.opts).map_err(|e| format!("generate: {e}"))?;
    // generate::run streams its own text; `--json 1` additionally prints the result object.
    if cli.opts.get("json").map(String::as_str) == Some("1") {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        );
    }
    Ok(result.get("pass").and_then(Value::as_bool).unwrap_or(true))
}

fn run(cli: &Cli) -> Result<(bool, Value), String> {
    let out = require(&cli.opts, "out")?.to_string();
    let (label, run_fn, fixtures): (String, RunFn, PathBuf);
    let mut opts = cli.opts.clone();
    opts.remove("out");
    if cli.negative {
        opts.insert("negative".into(), "1".into());
    }
    match cli.sub.as_str() {
        "module" => {
            let m = require(&cli.opts, "module")?.to_string();
            run_fn = module_fn(&m)?;
            let base = PathBuf::from(require(&cli.opts, "fixtures")?);
            fixtures = if cli.negative {
                base.join("negative_controls")
            } else {
                base
            };
            opts.remove("fixtures");
            opts.remove("module");
            opts.insert("module".into(), m.clone());
            label = format!("module:{m}");
        }
        "state" => {
            require(&cli.opts, "model")?;
            require(&cli.opts, "tokens")?;
            let mode = require(&cli.opts, "mode")?;
            if !matches!(mode, "serial" | "batched" | "graph" | "state-only") {
                return Err(format!("--mode `{mode}`: serial|batched|graph|state-only"));
            }
            run_fn = raon_oracle::state::run;
            fixtures = PathBuf::from(cli.opts.get("fixtures").map(String::as_str).unwrap_or("."));
            label = format!("state:{mode}");
        }
        "logits" => {
            require(&cli.opts, "model")?;
            if !cli.opts.contains_key("kld") {
                require(&cli.opts, "tokens")?;
            }
            if !cli.opts.contains_key("teacher") && !cli.opts.contains_key("kld") {
                return Err("logits needs --teacher F or --kld REF.kldref.bin".into());
            }
            run_fn = raon_oracle::logits::run;
            fixtures = PathBuf::from(cli.opts.get("fixtures").map(String::as_str).unwrap_or("."));
            label = "logits".into();
        }
        "bench" => {
            if !cli.opts.contains_key("bucket-json") {
                require(&cli.opts, "model")?;
                parse_pp_list(require(&cli.opts, "pp")?)?;
                require(&cli.opts, "tg")?
                    .parse::<usize>()
                    .map_err(|e| format!("--tg: {e}"))?;
            }
            run_fn = raon_oracle::bench::run;
            fixtures = PathBuf::from(cli.opts.get("fixtures").map(String::as_str).unwrap_or("."));
            label = "bench".into();
        }
        _ => unreachable!(),
    }

    let result = run_fn(&fixtures, &opts).map_err(|e| format!("{label}: {e}"))?;
    let (ok, lines) = match cli.sub.as_str() {
        "module" => case_verdict(&result, cli.negative),
        "bench" => {
            // A measurement probe: verdict is "ran without error and did not
            // self-report failure".
            let ok = result.get("pass").and_then(Value::as_bool).unwrap_or(true);
            (ok, vec![format!("  pass={ok}")])
        }
        _ => scalar_verdict(&result),
    };
    println!("raon_oracle {label}: {}", if ok { "OK" } else { "FAILED" });
    for l in &lines {
        println!("{l}");
    }
    let doc = json!({
        "oracle": label,
        "negative": cli.negative,
        "fixtures": fixtures.display().to_string(),
        "ok": ok,
        "verdict": lines,
        "result": result,
    });
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(&out, text).map_err(|e| format!("write {out}: {e}"))?;
    Ok((ok, doc))
}

fn main() {
    let cli = match parse_cli() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("raon_oracle: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if cli.sub == "generate" {
        match run_generate(&cli) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("raon_oracle: ERROR {e}");
                std::process::exit(1);
            }
        }
    }
    match run(&cli) {
        Ok((true, _)) => {}
        Ok((false, _)) => std::process::exit(1),
        Err(e) => {
            eprintln!("raon_oracle: ERROR {e}");
            std::process::exit(1);
        }
    }
}
