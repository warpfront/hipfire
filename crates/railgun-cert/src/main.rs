//! `railgun-cert certify [--out DIR] OBJECT...` writes one receipt per code
//! object to `DIR/receipts/<arch>/<object_sha256>.receipt.json`.
//!
//! `railgun-cert coverage --manifest FILE --out DIR` certifies every object a
//! launch-sequence manifest names and writes `DIR/coverage.jsonl`, one row per
//! (sequence, kernel). Manifest lines: `sequence<TAB>kernel<TAB>object|-[<TAB>note]`;
//! `#` starts a comment. A runtime sidecar `<stem>.radiowave.json` next to a
//! `<stem>.hsaco` is used for the differential when present.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use railgun_cert::{certify, Receipt};
use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;

fn sidecar_of(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".hsaco").or_else(|| name.strip_suffix(".co"))?;
    std::fs::read_to_string(path.with_file_name(format!("{stem}.radiowave.json"))).ok()
}

fn certify_path(path: &Path, out: &Path) -> Result<(Receipt, PathBuf)> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let receipt = certify(&bytes, sidecar_of(path).as_deref()).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = out.join("receipts").join(&receipt.arch);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(format!("{}.receipt.json", receipt.object_sha256));
    std::fs::write(&file, serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok((receipt, file))
}

fn row(sequence: &str, kernel: &str, object: &str, note: &str, result: Option<&Result<(Receipt, PathBuf)>>) -> Value {
    let base = json!({ "sequence": sequence, "kernel": kernel, "object": object, "note": note });
    let Some(result) = result else { return merge(base, json!({ "status": "missing_object" })) };
    let (receipt, file) = match result {
        Ok(r) => r,
        Err(e) => return merge(base, json!({ "status": "lift_rejected", "error": e })),
    };
    let Some(k) = receipt.railgun.kernels.iter().find(|k| k.symbol == kernel) else {
        let symbols: Vec<&str> = receipt.railgun.kernels.iter().map(|k| k.symbol.as_str()).collect();
        return merge(base, json!({ "status": "symbol_missing", "object_sha256": receipt.object_sha256, "symbols": symbols }));
    };
    let mut rules: BTreeMap<&str, usize> = BTreeMap::new();
    for u in &k.unresolved { *rules.entry(u.rule.as_str()).or_insert(0) += 1; }
    let first: Vec<Value> = k.unresolved.iter().take(3).map(|u| json!({ "pc": u.pc, "rule": u.rule, "detail": u.detail })).collect();
    let fails: Vec<Value> = k.differential.args.iter().filter(|a| a.status == "fail").map(|a| json!({ "offset": a.offset, "name": a.name, "a1": a.a1, "metadata": a.metadata, "reason": a.reason })).collect();
    let args: Vec<Value> = k.args.iter().filter(|a| a.mode.is_some() || a.value_kind == "global_buffer").map(|a| json!({
        "name": a.name, "offset": a.offset, "kind": a.value_kind, "mode": a.mode, "role": a.role, "read_cache": a.read_cache,
        "classes": a.classes, "shared": a.shared,
    })).collect();
    let scalars: Vec<Value> = k.args.iter().filter(|a| a.role != "pointer" && a.value_kind != "global_buffer" && a.loaded).map(|a| json!({
        "name": a.name, "offset": a.offset, "kind": a.value_kind, "role": a.role, "sinks": a.sinks,
    })).collect();
    merge(base, json!({
        "status": k.status, "arch": receipt.arch, "object_sha256": receipt.object_sha256, "receipt": file,
        "unresolved_rules": rules, "first_unresolved": first,
        "differential_source": k.differential.source, "differential_status": k.differential.status, "differential_fails": fails,
        "radiowave_mutable_read_cache": k.differential.radiowave_mutable_read_cache,
        "read_cache": k.read_cache, "dynamic_kernarg_reads": k.dynamic_kernarg_reads, "taint_overflow": k.taint_overflow,
        "hidden_loaded": k.implicit.hidden_loaded, "hidden_grid_live": k.implicit.hidden_grid_live,
        "loads_block_count": k.implicit.loads_block_count, "loads_group_size": k.implicit.loads_group_size,
        "side": k.side, "args": args, "scalars": scalars, "pm_analyze_error": k.pm_analyze_error,
    }))
}

fn merge(mut a: Value, b: Value) -> Value {
    if let (Some(a), Value::Object(b)) = (a.as_object_mut(), b) { a.extend(b); }
    a
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("usage: railgun-cert certify|coverage ...")?;
    let mut out = PathBuf::from("railgun-cert-out");
    let mut manifest = None;
    let mut objects = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out needs a value")?),
            "--manifest" => manifest = Some(PathBuf::from(args.next().ok_or("--manifest needs a value")?)),
            other => objects.push(PathBuf::from(other)),
        }
    }
    match command.as_str() {
        "certify" => {
            for path in objects {
                match certify_path(&path, &out) {
                    Ok((receipt, file)) => {
                        for k in &receipt.railgun.kernels {
                            println!("{}\t{}\t{}\tdiff={}\t{}", receipt.arch, k.symbol, k.status, k.differential.status, file.display());
                        }
                    }
                    Err(e) => println!("REJECT\t{e}"),
                }
            }
            Ok(())
        }
        "coverage" => {
            let manifest = manifest.ok_or("coverage needs --manifest")?;
            let text = std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
            let mut cache: HashMap<PathBuf, Result<(Receipt, PathBuf)>> = HashMap::new();
            let mut lines = Vec::new();
            for line in text.lines() {
                let line = line.split('#').next().unwrap_or("").trim_end();
                if line.trim().is_empty() { continue; }
                let cols: Vec<&str> = line.split('\t').collect();
                let (sequence, kernel, object) = match cols.as_slice() {
                    [s, k, o, ..] => (*s, *k, *o),
                    _ => return Err(format!("malformed manifest line: {line}")),
                };
                let note = cols.get(3).copied().unwrap_or("");
                let result = if object == "-" { None } else {
                    let path = PathBuf::from(object);
                    if !cache.contains_key(&path) {
                        let r = certify_path(&path, &out);
                        cache.insert(path.clone(), r);
                    }
                    cache.get(&path)
                };
                lines.push(serde_json::to_string(&row(sequence, kernel, object, note, result)).map_err(|e| e.to_string())?);
            }
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            std::fs::write(out.join("coverage.jsonl"), lines.join("\n") + "\n").map_err(|e| e.to_string())?;
            eprintln!("{} rows, {} objects -> {}", lines.len(), cache.len(), out.display());
            Ok(())
        }
        "recording-inventory" => railgun_cert::recording::cli(&objects),
        other => Err(format!("unknown command {other}")),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("railgun-cert: {e}"); ExitCode::FAILURE }
    }
}
