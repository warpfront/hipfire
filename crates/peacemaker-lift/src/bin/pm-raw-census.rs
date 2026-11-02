// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline census and identity gate for one raw ACO artifact (no GPU, no Mesa, no tools).
//! Usage: pm-raw-census <shader.rawspec.json>
//!
//! The spec file is the `pm-raw-kernel-spec-v1` JSON (`peacemaker_lift::raw::parse_spec_file`);
//! its `binary` and `sidecar` names are resolved next to it. The sidecar must hash to the
//! declared `sidecar_sha256`; the executable range goes through `lift_raw` (typed decode,
//! word-for-word re-encode, CFG/window checks, explicit entry facts). Prints one JSON verdict
//! on stdout: `"verdict":"pass"` (exit 0) only when the whole stream lifts, otherwise
//! `"verdict":"kill"` (exit 1) naming the causal rule, byte offset and word. The `census`
//! lists every decoded opcode/form and every undecodable word; the first failure is exactly
//! what the lift reports, later ones are best effort (see `raw::StreamScan`).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use peacemaker_ir::inst::{Abi, Frontend};
use peacemaker_lift::raw::{parse_spec_file, scan_stream, verify_sidecar, SpecFile};
use peacemaker_lift::{emit, lift_raw, LiftError, Options};
use serde_json::{json, Value};

const SCHEMA: &str = "pm-raw-census-v1";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn kill(stage: &str, reason: impl ToString) -> Value {
    json!({"schema": SCHEMA, "verdict": "kill", "stage": stage, "reason": reason.to_string()})
}

fn word_at(bytes: &[u8], offset: u64) -> Value {
    usize::try_from(offset)
        .ok()
        .and_then(|at| bytes.get(at..at + 4))
        .map_or(Value::Null, |w| {
            json!(format!(
                "{:#010x}",
                u32::from_le_bytes(w.try_into().expect("four bytes"))
            ))
        })
}

fn lift_kill(stage: &str, error: &LiftError, bytes: &[u8]) -> Value {
    let LiftError::Rejected {
        kernel,
        offset,
        rule,
        reason,
    } = error;
    json!({
        "schema": SCHEMA, "verdict": "kill", "stage": stage, "rule": format!("{rule:?}"),
        "kernel": kernel, "offset": offset, "word": word_at(bytes, *offset), "reason": reason,
    })
}

fn census(bytes: &[u8], file: &SpecFile) -> Value {
    let spec = &file.spec;
    let range = spec
        .extent
        .offset
        .checked_add(spec.extent.len)
        .and_then(|end| {
            Some((
                usize::try_from(spec.extent.offset).ok()?,
                usize::try_from(end).ok()?,
            ))
        });
    let Some(code) = range.and_then(|(start, end)| bytes.get(start..end)) else {
        return json!({"error": "executable extent is outside the binary"});
    };
    let scan = scan_stream(code, spec.target.arch);
    let mut opcodes: BTreeMap<&str, usize> = BTreeMap::new();
    let mut forms: BTreeMap<&str, usize> = BTreeMap::new();
    for inst in &scan.decoded {
        *opcodes.entry(inst.opcode.as_str()).or_default() += 1;
        *forms.entry(inst.form.as_str()).or_default() += 1;
    }
    let failures: Vec<Value> = scan
        .failures
        .iter()
        .map(|f| {
            json!({
                "offset": spec.extent.offset + f.offset, "word": format!("{:#010x}", f.word),
                "rule": format!("{:?}", f.rule), "reason": f.reason,
            })
        })
        .collect();
    json!({
        "words": scan.words, "instructions": scan.decoded.len(), "opcodes": opcodes, "forms": forms,
        "failures": failures, "failures_after_first_are_best_effort": true,
    })
}

fn run(spec_path: &Path) -> (Value, bool) {
    let fail = |value: Value| (value, false);
    let text = match fs::read_to_string(spec_path) {
        Ok(text) => text,
        Err(e) => return fail(kill("read-spec", format!("{}: {e}", spec_path.display()))),
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(e) => return fail(kill("parse-spec", e)),
    };
    let file = match parse_spec_file(&value) {
        Ok(file) => file,
        Err(e) => return fail(kill("parse-spec", e)),
    };
    let dir = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let bytes = match fs::read(dir.join(&file.binary)) {
        Ok(bytes) => bytes,
        Err(e) => return fail(kill("read-binary", format!("{}: {e}", file.binary))),
    };
    let sidecar = match fs::read(dir.join(&file.sidecar)) {
        Ok(sidecar) => sidecar,
        Err(e) => return fail(kill("read-sidecar", format!("{}: {e}", file.sidecar))),
    };
    let census = census(&bytes, &file);
    if let Err(e) = verify_sidecar(&sidecar, &file.spec) {
        let mut verdict = lift_kill("sidecar", &e, &bytes);
        verdict["census"] = census;
        return fail(verdict);
    }
    match lift_raw(
        &bytes,
        &file.spec,
        Options {
            frontend: Frontend::Aco,
        },
    ) {
        Err(e) => {
            let mut verdict = lift_kill("lift", &e, &bytes);
            verdict["census"] = census;
            fail(verdict)
        }
        Ok(lifted) => {
            let kernel = &lifted.program.kernels[0];
            let arch = lifted.program.target.arch;
            let range = file.spec.extent.offset as usize
                ..(file.spec.extent.offset + file.spec.extent.len) as usize;
            let identical = emit::bytes(kernel, arch).is_ok_and(|code| code == bytes[range]);
            let Abi::Raw(raw) = &kernel.abi else {
                return fail(kill("lift", "lifted kernel has no raw ABI"));
            };
            let streams: Vec<Value> = lifted
                .proof
                .streams
                .iter()
                .map(|s| {
                    json!({
                        "entry": s.entry, "size": s.size, "stream_sha256": hex(&s.stream_sha256),
                    })
                })
                .collect();
            (
                json!({
                    "schema": SCHEMA, "verdict": if identical { "pass" } else { "kill" },
                    "stage": if identical { "complete" } else { "identity" },
                    "kernel": kernel.symbol.0, "arch": format!("{arch:?}"), "wave": format!("{:?}", kernel.wave),
                    "instructions": kernel.body.layout.len(), "blocks": kernel.body.blocks.len(),
                    "entry_facts": raw.entry.len(), "relocations": raw.relocations.len(),
                    "identity": {
                        "emit_bytes_equals_executable": identical,
                        "input_sha256": hex(&lifted.proof.input_sha256),
                        "sidecar_sha256": lifted.proof.sidecar_sha256.as_ref().map(|h| hex(h.as_slice())),
                        "streams": streams,
                    },
                    "census": census,
                }),
                identical,
            )
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        eprintln!("usage: pm-raw-census <shader.rawspec.json>");
        return ExitCode::from(2);
    };
    let (verdict, pass) = run(Path::new(path));
    println!(
        "{}",
        serde_json::to_string_pretty(&verdict).expect("verdict serializes")
    );
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
