// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone emitter + certifier of the gfx1100 row-local R128 / K640 A4 producer
//! `mq_i4_a4_r128_k640_gfx1100` (module `mq_i4_a4_pm_gfx1100`). The kernel module is included by
//! path so `src/lib.rs` is untouched; everything it uses is `peacemaker_isa`'s public API.
//!
//! ```text
//! cargo run -p peacemaker-isa --features toolchain --example mq_i4_a4_gfx1100_emit -- <out-dir> [--host-target TRIPLE]
//! ```
//!
//! Writes, under `<out-dir>`:
//!  * `mq_i4_a4_pm_gfx1100.s`, `.proof.json`, `.co` (native writer), `.m7.json` (M7 receipts of the
//!    native and the ROCm-oracle-linked object), `.hazards.json` (independent wait / hazard replay
//!    plus the builder's inserted hazard guards), `.audit.json` / `.audit.md` / `.audit.default.isa.txt`
//!    (`audit::run` on the native object), `.certificate.json` (identities + every stage's result);
//!  * `kernels/mq_i4_a4_pm_gfx1100.hxaco` (HIP offload bundle of the native object) and
//!    `kernels/mq_i4_a4.gfx1100.r128_k640.contract.json` (the shape contract);
//!  * `oracle/` (ROCm `llvm-mc` + `ld.lld` build of the same `.s`, and the radiowave `manifest.json`).
//!
//! Every stage is a hard stop: wait replay (per kernel and module), empty hazard replay, native
//! assembly, M7 (byte-exact lift, no obligations, no ambiguous delays) on the native and the
//! oracle object, ROCm build with parse-back, `toolchain::certify` with the contract, and a
//! finding-free `audit::run`. On any failure it prints `{"verdict":"kill","stage":..,"reason":..}`
//! and exits 1; on success one JSON summary and exit 0. Nothing is asserted by hand.
#![allow(dead_code)]

#[path = "../src/kernels/mq_i4_a4_gfx1100.rs"]
#[allow(dead_code)]
mod mq_i4_a4_gfx1100;

#[cfg(not(feature = "toolchain"))]
fn main() {
    println!(
        "{}",
        serde_json::json!({"verdict": "kill", "stage": "features", "reason": "build with --features toolchain (M7, ROCm oracle build, certify, audit)"})
    );
    std::process::exit(1)
}

#[cfg(feature = "toolchain")]
use peacemaker_isa::{
    audit, ledger_replay, native, pm_check,
    toolchain::{build, certify, IsaShapeContract, Toolchain},
    Arch,
};
#[cfg(feature = "toolchain")]
use serde_json::{json, Value};
#[cfg(feature = "toolchain")]
use sha2::{Digest, Sha256};
#[cfg(feature = "toolchain")]
use std::path::{Path, PathBuf};

#[cfg(feature = "toolchain")]
const ARCH: Arch = Arch::Gfx1100;

#[cfg(feature = "toolchain")]
fn kill(stage: &str, reason: impl std::fmt::Display) -> ! {
    println!(
        "{}",
        json!({"verdict": "kill", "stage": stage, "reason": reason.to_string()})
    );
    std::process::exit(1)
}

#[cfg(feature = "toolchain")]
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "toolchain")]
fn put(stage: &str, path: &Path, bytes: impl AsRef<[u8]>) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| kill(stage, format!("{}: {e}", parent.display())))
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| kill(stage, format!("{}: {e}", path.display())));
}

#[cfg(feature = "toolchain")]
fn pretty(stage: &str, value: &impl serde::Serialize) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(value).unwrap_or_else(|e| kill(stage, e));
    bytes.push(b'\n');
    bytes
}

#[cfg(feature = "toolchain")]
fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: mq_i4_a4_gfx1100_emit <out-dir> [--host-target TRIPLE]";
    let out = PathBuf::from(args.next().unwrap_or_else(|| kill("usage", usage)));
    let mut host = native::DEFAULT_HOST_TARGET.to_owned();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--host-target" => host = args.next().unwrap_or_else(|| kill("usage", usage)),
            _ => kill("usage", usage),
        }
    }
    let module = mq_i4_a4_gfx1100::Spec::module(ARCH);
    let symbol = mq_i4_a4_gfx1100::Spec { arch: ARCH }.symbol();
    let tag = mq_i4_a4_gfx1100::Spec { arch: ARCH }.tag();

    // Emission: the checked builder, then the independent wait / hazard replay of the source text.
    let (emitted, text, proof) =
        mq_i4_a4_gfx1100::emit_module(ARCH).unwrap_or_else(|e| kill("emit", e));
    for e in &emitted {
        ledger_replay::replay_waits(&e.s_text, ARCH).unwrap_or_else(|e| kill("wait_replay", e))
    }
    ledger_replay::replay_waits(&text, ARCH).unwrap_or_else(|e| kill("wait_replay", e));
    let replay_hazards =
        ledger_replay::replay_hazards(&text, ARCH).unwrap_or_else(|e| kill("hazard_replay", e));
    if !replay_hazards.is_empty() {
        kill(
            "hazard_replay",
            format!("{} hazards: {replay_hazards:?}", replay_hazards.len()),
        )
    }

    let s_path = out.join(format!("{module}.s"));
    let proof_bytes = pretty("proof", &proof);
    put("output", &s_path, &text);
    put(
        "output",
        &out.join(format!("{module}.proof.json")),
        &proof_bytes,
    );

    // Native code object and bundle (no ROCm tools).
    let elf = native::assemble(&text, ARCH).unwrap_or_else(|e| kill("native_assemble", e));
    let co_path = out.join(format!("{module}.co"));
    let bundle = native::bundle(&elf, ARCH, &host);
    let hxaco_path = out.join("kernels").join(format!("{module}.hxaco"));
    put("output", &co_path, &elf);
    put("output", &hxaco_path, &bundle);

    // Shape contract, derived from the fresh emission.
    let [spec] = mq_i4_a4_gfx1100::module_specs(ARCH)[..] else {
        kill("emit", "one entry expected")
    };
    let contract_value = mq_i4_a4_gfx1100::contract(spec, &emitted[0]);
    let contract: IsaShapeContract =
        serde_json::from_value(contract_value.clone()).unwrap_or_else(|e| kill("contract", e));
    let contract_path = out
        .join("kernels")
        .join(format!("mq_i4_a4.{}.{tag}.contract.json", ARCH.name()));
    put(
        "output",
        &contract_path,
        pretty("contract", &contract_value),
    );

    // M7 on the natively linked object.
    let m7_native = pm_check::m7(&co_path, ARCH.name(), &symbol)
        .unwrap_or_else(|e| kill("m7_native", format!("{symbol}: {e}")));
    if m7_native["lift"].as_str() != Some("byte-exact")
        || m7_native["obligations"] != json!({})
        || m7_native["ambiguous_delays"] != json!(0)
    {
        kill("m7_native", format!("{symbol}: {m7_native}"));
    }

    // ROCm oracle build of the same source (llvm-mc + ld.lld + bundler, parse-back mandatory).
    let toolchain = Toolchain::oracle();
    let oracle_dir = out.join("oracle");
    std::fs::create_dir_all(&oracle_dir).unwrap_or_else(|e| kill("output", e));
    let oracle_s = oracle_dir.join(format!("{module}.s"));
    put("output", &oracle_s, &text);
    let built = build(
        &toolchain,
        &oracle_s,
        &oracle_dir.join(format!("{module}.hsaco")),
        ARCH.name(),
    )
    .unwrap_or_else(|e| kill("oracle_build", e));
    let m7_oracle = pm_check::m7(&built.elf, ARCH.name(), &symbol)
        .unwrap_or_else(|e| kill("m7_oracle", format!("{symbol}: {e}")));
    if m7_oracle["lift"].as_str() != Some("byte-exact")
        || m7_oracle["obligations"] != json!({})
        || m7_oracle["ambiguous_delays"] != json!(0)
    {
        kill("m7_oracle", format!("{symbol}: {m7_oracle}"));
    }
    let manifest = oracle_dir.join("manifest.json");
    certify(
        &toolchain,
        &built,
        &oracle_s,
        ARCH.name(),
        &manifest,
        Some(&contract),
        &sha256(&proof_bytes),
        &proof.builder_git_sha,
    )
    .unwrap_or_else(|e| kill("certify", format!("{symbol}: {e}")));

    // Native-object audit: resource / spill / scratch / barrier findings must be empty.
    let audit_json = out.join(format!("{module}.audit.json"));
    let report = audit::run(audit::Options {
        input: co_path.clone(),
        arch: ARCH.name().into(),
        json: Some(audit_json.clone()),
        markdown: Some(out.join(format!("{module}.audit.md"))),
        ..Default::default()
    })
    .unwrap_or_else(|e| kill("audit", e));
    let mut findings = 0usize;
    let mut audited = false;
    for kernel in report.profiles.iter().flat_map(|p| &p.kernels) {
        if kernel.symbol == symbol {
            audited = true
        }
        findings +=
            kernel.findings.len() + kernel.loops.iter().map(|l| l.findings.len()).sum::<usize>();
    }
    if !audited {
        kill("audit", format!("{symbol} absent from the audit report"))
    }
    if findings != 0 {
        kill(
            "audit",
            format!("{findings} findings: see {}", audit_json.display()),
        )
    }

    // Receipts.
    let by_symbol = |receipt: &Value| {
        let mut m = serde_json::Map::new();
        m.insert(symbol.clone(), receipt.clone());
        Value::Object(m)
    };
    let m7_receipts = json!({ "native": by_symbol(&m7_native), "oracle": by_symbol(&m7_oracle) });
    put(
        "output",
        &out.join(format!("{module}.m7.json")),
        pretty("m7", &m7_receipts),
    );
    let hazards = json!({
        "arch": ARCH.name(), "symbol": symbol,
        "wait_replay": "pass", "hazard_replay": replay_hazards, "hazard_replay_count": replay_hazards.len(),
        "builder_inserted_hazards": proof.kernels.iter().map(|k| json!({"variant": k.variant, "hazards": k.hazards, "waits": k.waits.len()})).collect::<Vec<_>>(),
        "m7_obligations": m7_native["obligations"], "m7_ambiguous_delays": m7_native["ambiguous_delays"],
        "m7_delays": m7_native["delays"], "m7_clauses": m7_native["clauses"],
    });
    put(
        "output",
        &out.join(format!("{module}.hazards.json")),
        pretty("hazards", &hazards),
    );
    let file = |p: &Path| -> Value {
        let bytes =
            std::fs::read(p).unwrap_or_else(|e| kill("output", format!("{}: {e}", p.display())));
        json!({ "path": p.strip_prefix(&out).unwrap_or(p).display().to_string(), "sha256": sha256(&bytes), "bytes": bytes.len() })
    };
    let certificate = json!({
        "verdict": "pass", "arch": ARCH.name(), "module": module, "symbol": symbol,
        "builder_crate_version": proof.builder_crate_version, "builder_git_sha": proof.builder_git_sha,
        "s_text_sha256": proof.s_text_sha256, "proof_sha256": sha256(&proof_bytes),
        "stages": {
            "wait_replay": "pass", "hazard_replay": "pass (0 hazards)", "native_assemble": "pass",
            "m7_native": "pass", "oracle_build_parse_back": "pass", "m7_oracle": "pass",
            "toolchain_certify": "pass", "audit": "pass (0 findings)",
        },
        "files": {
            "s": file(&s_path), "co": file(&co_path), "hxaco": file(&hxaco_path), "contract": file(&contract_path),
            "oracle_hsaco": file(&built.hsaco), "oracle_elf": file(&built.elf), "manifest": file(&manifest), "audit": file(&audit_json),
        },
    });
    let certificate_path = out.join(format!("{module}.certificate.json"));
    put(
        "output",
        &certificate_path,
        pretty("certificate", &certificate),
    );
    println!(
        "{}",
        json!({
            "verdict": "pass", "module": module, "symbol": symbol, "out": out.display().to_string(),
            "hxaco": hxaco_path.display().to_string(), "certificate": certificate_path.display().to_string(),
            "co_sha256": sha256(&elf), "hxaco_sha256": sha256(&bundle), "s_text_sha256": proof.s_text_sha256,
        })
    );
}
