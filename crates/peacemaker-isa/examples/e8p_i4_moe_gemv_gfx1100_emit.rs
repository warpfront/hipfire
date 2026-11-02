// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone emission of the E8P-I4 K-E2 decode GEMV module for gfx1100 (default), gfx1151 or gfx1201
//! (`kernels/e8p_i4_moe_gemv.rs`, library module `kernels::e8p_i4_moe_gemv`; the gfx1201 emitted
//! text is unchanged by the gfx11 arms, so `--arch gfx1201` re-emits the committed object).
//!
//! ```text
//! cargo run -p peacemaker-isa --features toolchain --example e8p_i4_moe_gemv_gfx1100_emit -- \
//!     <output-dir> [--arch gfx1100|gfx1151|gfx1201] [--host-target TRIPLE]
//! ```
//!
//! Files written to `<output-dir>` (`<module>` is `e8p_i4_moe_gemv_pm_<arch>`):
//!  * `<module>.s`: the two-symbol module assembly (gfx1100 / gfx1151: `e8p_i4_moe_gemv_{r256_gate_up,r128_down}_pm_<arch>`;
//!    gfx1201: `e8p_i4_moe_gemv_{r256_gate_up,r128_down}`);
//!  * `<module>.proof.json`: the `ModuleProof` (per-kernel register plan, wait ledger, hazard guards, loop
//!    fixpoints, `s_text_sha256`, module digest);
//!  * `<module>.co`: the native code object `native::assemble` links from that source;
//!  * `<module>.hxaco`: `native::bundle` of that object (`__CLANG_OFFLOAD_BUNDLE__`, host entry
//!    `host-x86_64-unknown-linux-gnu` unless `--host-target` is given), the committed-bundle packaging;
//!  * `<module>.m7.json`: `pm_check::m7` receipts keyed by symbol (needs `--features toolchain`; without it
//!    the files above and the contracts are written, then the run errors instead of emitting a receipt);
//!  * `e8p_i4_moe_gemv.<arch>.<tag>.contract.json`, `<tag>` in `r256_gate_up`, `r128_down`.
//!
//! Prints the SHA-256 of the object and of the bundle. Nothing here runs a GPU or touches the registry.

use std::{fs, path::PathBuf};

use peacemaker_isa::kernels::e8p_i4_moe_gemv::{contract, emit_module, module_specs, Spec};
use peacemaker_isa::{native, Arch};
use sha2::{Digest, Sha256};

const USAGE: &str = "usage: e8p_i4_moe_gemv_gfx1100_emit <output-dir> [--arch gfx1100|gfx1151|gfx1201] [--host-target TRIPLE]";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut dir, mut target, mut host) = (
        None::<PathBuf>,
        Arch::Gfx1100,
        native::DEFAULT_HOST_TARGET.to_owned(),
    );
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(v) = arg.strip_prefix("--arch=") {
            target = v.parse()?
        } else if arg == "--arch" {
            target = args.next().ok_or(USAGE)?.parse()?
        } else if let Some(v) = arg.strip_prefix("--host-target=") {
            host = v.to_owned()
        } else if arg == "--host-target" {
            host = args.next().ok_or(USAGE)?
        } else if arg.starts_with("--") || dir.is_some() {
            return Err(USAGE.into());
        } else {
            dir = Some(PathBuf::from(arg))
        }
    }
    let dir = dir.ok_or(USAGE)?;
    if !matches!(target, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(format!("{USAGE}: {} is not an E8P K-E2 target", target.name()).into());
    }
    fs::create_dir_all(&dir)?;

    let (emitted, text, proof) = emit_module(target)?;
    let specs = module_specs(target);
    if emitted.len() != specs.len() {
        return Err("module emission returned a different number of kernels".into());
    }
    let module = Spec::module(target);

    let source = dir.join(format!("{module}.s"));
    fs::write(&source, &text)?;
    fs::write(
        dir.join(format!("{module}.proof.json")),
        serde_json::to_vec_pretty(&proof)?,
    )?;
    let elf = native::assemble(&text, target)?;
    let object = dir.join(format!("{module}.co"));
    fs::write(&object, &elf)?;
    let bundle = native::bundle(&elf, target, &host);
    let bundle_path = dir.join(format!("{module}.hxaco"));
    fs::write(&bundle_path, &bundle)?;

    let mut summary = Vec::new();
    for (spec, e) in specs.iter().zip(&emitted) {
        let file = dir.join(format!(
            "e8p_i4_moe_gemv.{}.{}.contract.json",
            target.name(),
            spec.tag()
        ));
        fs::write(&file, serde_json::to_vec_pretty(&contract(*spec, e))?)?;
        summary.push(serde_json::json!({
            "symbol": spec.symbol(), "contract": file, "vgpr": e.shape.next_free_vgpr, "sgpr": e.shape.next_free_sgpr,
            "waits": e.shape.waits, "hazard_guards": e.proof.hazards.len(),
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "module": module, "arch": target.name(), "source": source, "object": object, "bundle": bundle_path,
            "s_text_sha256": proof.s_text_sha256, "object_sha256": sha(&elf), "bundle_sha256": sha(&bundle),
            "object_bytes": elf.len(), "bundle_bytes": bundle.len(), "entries": summary,
        }))?
    );

    m7(&dir, &module, &object, target, &specs)
}

/// M7 byte obligations of every symbol of the linked object, written to `<module>.m7.json`.
#[cfg(feature = "toolchain")]
fn m7(
    dir: &std::path::Path,
    module: &str,
    object: &std::path::Path,
    target: Arch,
    specs: &[Spec],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut receipts = std::collections::BTreeMap::new();
    for spec in specs {
        let symbol = spec.symbol();
        let receipt = peacemaker_isa::pm_check::m7(object, target.name(), &symbol)?;
        receipts.insert(symbol, receipt);
    }
    let receipts = serde_json::to_vec_pretty(&receipts)?;
    fs::write(dir.join(format!("{module}.m7.json")), &receipts)?;
    println!("{}", String::from_utf8_lossy(&receipts));
    Ok(())
}

#[cfg(not(feature = "toolchain"))]
fn m7(
    _dir: &std::path::Path,
    module: &str,
    _object: &std::path::Path,
    _target: Arch,
    _specs: &[Spec],
) -> Result<(), Box<dyn std::error::Error>> {
    Err(format!("M7 certification requires --features toolchain (pm_check is feature-gated); no {module}.m7.json was written").into())
}
