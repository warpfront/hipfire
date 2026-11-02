// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline F1→F3 gate: a producer artifact directory (`artifact.json`, `shader.rawspec.json`,
//! `shader.bin`, `shader.abi.json`, `shader.spv`, ...) and an `AcoHipContract` JSON in, an HSA
//! code object around the untouched ACO body out.
//!
//! ```text
//! cargo run -p peacemaker-isa --features toolchain --example aco_hsa_emit -- <artifact-dir>/shader.rawspec.json <contract.json> <out-dir>
//! ```
//!
//! Stages, each a hard stop: `artifact.json` verification (every listed file present and hashed to
//! its recorded SHA-256, no unlisted file, provenance present, recorded executable range = hashed
//! range of `shader.bin`) → spec/binary/sidecar/contract identities equal the artifact's → sidecar
//! hash → `lift_raw` (typed decode, exact re-encode, entry facts) → raw static certificate (shared
//! analyses, no open obligation) → `adapt_to_hsa` (prologue, native link, relift, HSA certificate).
//! A hand-edited rawspec / FP / resource declaration is refused at the first stage, before any
//! lift. On success it writes `candidate.co`, `candidate.hxaco` (HIP bundle) and the
//! `aco-pm-candidate-v1` manifest for each (`candidate.co.manifest.json`,
//! `candidate.hxaco.manifest.json`) into `<out-dir>` and prints one JSON summary (exit 0). On any
//! rejection it prints `{"verdict":"kill","stage":..,"reason":..}` and exits 1: no partial output.
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::aco_cert::{certify_program, LaunchContract};
use peacemaker_isa::aco_hsa::{adapt_to_hsa, verify_artifact_dir, AcoHipContract, HipArgKind};
use peacemaker_lift::raw::{lift_raw, parse_spec_file, verify_sidecar};
use peacemaker_lift::Options;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn kill(stage: &str, reason: impl std::fmt::Display) -> ! {
    println!(
        "{}",
        json!({"verdict": "kill", "stage": stage, "reason": reason.to_string()})
    );
    std::process::exit(1)
}

fn read(path: &Path, what: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| kill("input", format!("{what} {}: {e}", path.display())))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [spec_path, contract_path, out_dir] = args.as_slice() else {
        kill(
            "usage",
            "aco_hsa_emit <shader.rawspec.json> <contract.json> <out-dir>",
        );
    };
    let spec_path = PathBuf::from(spec_path);
    let base = spec_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let base = if base.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        base
    };

    // Static import chain first: artifact.json must list and match every file in the directory.
    let binding = verify_artifact_dir(&base).unwrap_or_else(|e| kill("artifact", e));
    if spec_path.file_name().and_then(|n| n.to_str()) != Some("shader.rawspec.json") {
        kill(
            "artifact",
            "the raw spec must be the artifact's shader.rawspec.json",
        );
    }
    let spec_bytes = read(&spec_path, "spec");
    if sha256(&spec_bytes) != binding.rawspec_sha256 {
        kill(
            "artifact",
            "shader.rawspec.json changed between verification and use",
        )
    }
    let doc: Value = serde_json::from_slice(&spec_bytes).unwrap_or_else(|e| kill("spec", e));
    let file = parse_spec_file(&doc).unwrap_or_else(|e| kill("spec", e));
    if file.binary != "shader.bin" || file.sidecar != "shader.abi.json" {
        kill(
            "artifact",
            format!(
                "the spec names {} / {}, not the artifact's shader.bin / shader.abi.json",
                file.binary, file.sidecar
            ),
        );
    }
    let binary = read(&base.join(&file.binary), "binary");
    let sidecar = read(&base.join(&file.sidecar), "sidecar");
    if sha256(&binary) != binding.binary_sha256 || sha256(&sidecar) != binding.sidecar_sha256 {
        kill(
            "artifact",
            "shader.bin or shader.abi.json changed between verification and use",
        );
    }
    binding
        .check_raw_spec(&file.spec)
        .unwrap_or_else(|e| kill("artifact", e));
    verify_sidecar(&sidecar, &file.spec).unwrap_or_else(|e| kill("sidecar", e));

    let contract_text = String::from_utf8(read(Path::new(contract_path), "contract"))
        .unwrap_or_else(|e| kill("contract", e));
    let contract =
        AcoHipContract::from_json(&contract_text).unwrap_or_else(|e| kill("contract", e));
    contract.validate().unwrap_or_else(|e| kill("contract", e));
    binding
        .check_contract(&contract)
        .unwrap_or_else(|e| kill("artifact", e));

    let lifted = lift_raw(
        &binary,
        &file.spec,
        Options {
            frontend: Frontend::Aco,
        },
    )
    .unwrap_or_else(|e| kill("lift_raw", e));
    let launch = LaunchContract {
        grid: contract.shape.grid,
        block: contract.shape.block,
        dynamic_lds_max: contract.dynamic_lds_max,
        bound_inputs: contract
            .args
            .iter()
            .filter_map(|a| match &a.kind {
                HipArgKind::GlobalPointer { extent_bytes, .. } => {
                    Some((a.name.clone(), *extent_bytes))
                }
                HipArgKind::ByValue => None,
            })
            .collect(),
    };
    let raw = certify_program(lifted, &file.spec.symbol, &launch)
        .unwrap_or_else(|e| kill("raw_certificate", e));
    let emission = adapt_to_hsa(&raw, &contract).unwrap_or_else(|e| kill("adapt_to_hsa", e));

    let out = Path::new(out_dir);
    std::fs::create_dir_all(out).unwrap_or_else(|e| kill("output", e));
    for (name, image) in [
        ("candidate.co", &emission.elf),
        ("candidate.hxaco", &emission.bundle),
    ] {
        std::fs::write(out.join(name), image).unwrap_or_else(|e| kill("output", e));
        let manifest = serde_json::to_string_pretty(&emission.candidate_manifest(image))
            .expect("manifest serialises");
        std::fs::write(out.join(format!("{name}.manifest.json")), manifest + "\n")
            .unwrap_or_else(|e| kill("output", e));
    }
    let h = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let domain = |r: &peacemaker_isa::aco_cert::CertificationReport| {
        json!({
            "domain": format!("{:?}", r.domain),
            "frontend": format!("{:?}", r.frontend),
            "symbol": r.symbol,
            "input_sha256": h(&r.input_sha256),
            "stream_sha256": r.stream_sha256.iter().map(|s| h(s)).collect::<Vec<_>>(),
            "sidecar_sha256": r.sidecar_sha256.map(|s| h(&s)),
            "open_obligations": r.obligations.len(),
            "obligations": r.obligations.iter().map(|o| json!({"rule": o.rule_id, "kind": format!("{:?}", o.kind), "text": o.text})).collect::<Vec<_>>(),
            "launch": {"grid": r.launch.grid, "block": r.launch.block, "dynamic_lds_max": r.launch.dynamic_lds_max, "bound_inputs": r.launch.bound_inputs},
            "max_vgpr": r.max_vgpr,
            "max_sgpr": r.max_sgpr,
        })
    };
    let certificate = json!({
        "schema": "pm-aco-certificate-v1",
        "frontend": "Aco",
        "static_only": true,
        "gpu_receipt": null,
        "identities": {
            "artifact_sha256": h(&binding.artifact_sha256),
            "rawspec_sha256": h(&binding.rawspec_sha256),
            "spirv_sha256": h(&binding.spirv_sha256),
            "raw_binary_sha256": h(&binding.binary_sha256),
            "raw_exec_sha256": h(&binding.exec_sha256),
            "raw_exec_range": {"offset": binding.exec_offset, "len": binding.exec_len},
            "sidecar_sha256": h(&binding.sidecar_sha256),
            "compiler_sha256": h(&binding.compiler_sha256),
            "mesa_patch_sha256": h(&binding.mesa_patch_sha256),
            "mesa_commit": binding.mesa_commit,
            "contract_sha256": h(&emission.contract_sha256),
            "adapter_version": peacemaker_isa::aco_hsa::ADAPTER_VERSION,
            "hsa_elf_sha256": h(&emission.elf_sha256),
            "hsa_bundle_sha256": h(&sha256(&emission.bundle)),
            "prologue_sha256": h(&emission.prologue.sha256),
            "prologue_range": {"offset": emission.prologue.offset, "len": emission.prologue.len},
            "body_sha256": h(&emission.body.sha256),
            "body_range": {"offset": emission.body.offset, "len": emission.body.len},
        },
        "raw": domain(&raw.report),
        "hsa": domain(&emission.hsa_certificate.report),
        "adapter": {
            "kernarg_base": emission.proof.kernarg_base.to_string(),
            "kernarg_base_copied": emission.proof.kernarg_base_copied,
            "temps": emission.proof.temps.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
            "prologue_text": emission.proof.prologue_text,
            "hardware_entry": emission.proof.hardware_entry,
            "prologue_words": emission.proof.prologue_words,
            "body_words": emission.proof.body_words,
        },
        // Each item was executed on this run; reaching this point means every one passed (any failure exits with a kill verdict first).
        "checks_performed": [
            "artifact.json lists exactly the files on disk; every file hash recomputed",
            "provenance hashes present; recorded executable range hashes to the stated sha256",
            "rawspec / binary / sidecar / contract identities equal the artifact's",
            "lift_raw: typed decode, word-for-word typed re-encode of the executable range, CFG and window checks, entry-fact definedness",
            "raw static certificate: shared analyses (waits, hazards, barriers, definedness, LDS, resources) closed with zero obligations",
            "re-encoded raw body hash equals contract exec hash and the round-trip proof stream hash",
            "contract FP/resources/entry bindings/aperture equal the raw shader config; unbound entry SGPRs proven dead by typed liveness",
            "native link of prologue ++ body; .text starts with prologue ++ body; extracted body equals raw body",
            "lift_object of the emitted ELF (Frontend::Aco): typed words equal prologue ++ body, block ends equal raw ends shifted by the prologue, HSA target v6",
            "HSA static certificate: shared analyses closed with zero obligations",
        ],
    });
    std::fs::write(
        out.join("certificate.json"),
        serde_json::to_string_pretty(&certificate).expect("certificate serialises") + "\n",
    )
    .unwrap_or_else(|e| kill("output", e));
    println!(
        "{}",
        json!({
            "verdict": "emitted",
            "symbol": emission.symbol,
            "prologue_text": emission.proof.prologue_text,
            "prologue_range": {"offset": emission.prologue.offset, "len": emission.prologue.len},
            "body_range": {"offset": emission.body.offset, "len": emission.body.len},
            "temps": emission.proof.temps.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
            "hardware_entry": emission.proof.hardware_entry,
            "hsa_obligations": emission.hsa_certificate.report.obligations.len(),
            "contract_sha256": emission.contract_sha256.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        })
    );
}
