// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone emission of the Consumer24 B2 K1 gfx1100 decode GEMV module
//! `mq_i4_moe_gemv_pm_gfx1100` (`kernels/mq_i4_moe_gemv_gfx1100.rs`, included below with `#[path]`
//! because the emitter is deliberately not in the kernel registry).
//!
//! ```text
//! cargo run -p peacemaker-isa --features toolchain --example mq_i4_moe_gemv_gfx1100_emit -- <output-dir>
//! ```
//!
//! Files written to `<output-dir>`:
//!  * `b2.k1.s`: the four-symbol module assembly (`mq_i4_moe_gemv_{mq2i4,mq4sym}_{r256_gate_up,r128_down}_pm_gfx1100`);
//!  * `b2.k1.proof.json`: the `ModuleProof` (per-kernel `BuilderProof`s: register plan, wait ledger, gfx1100
//!    hazard guards, loop fixpoints, `s_text_sha256`, and the module digest of the combined source);
//!  * `b2.k1.co`: the native code object `native::assemble` links from that source (the object M7 certifies);
//!  * `b2.k1.m7.json`: `pm_check::m7` receipts keyed by symbol (lift byte-exact identity, whole-program wait
//!    replay, gfx1100 TRANS-use hazard replay, `s_clause`/`s_delay_alu` windows, resources). Any obligation fails
//!    the run. M7 lives behind the `toolchain` feature: without it the example still writes the four files
//!    above and the contracts, then exits with an error instead of a receipt;
//!  * `mq_i4_moe_gemv.gfx1100.<tag>.contract.json`, `<tag>` in `mq2i4_r256_gate_up`, `mq2i4_r128_down`,
//!    `mq4sym_r256_gate_up`, `mq4sym_r128_down`: the strict shape contract of each entry
//!    (`IsaShapeContract`, `peacemaker custom build --contract`), derived from the emission.
//!
//! Nothing here runs a GPU. The frozen ABI, the host-mapped meaning and the x4 caveat are documented on
//! the emitter module.

#![allow(dead_code)]

use std::{fs, path::PathBuf};

// Facade for the emitter's `crate::` / `super::` dependencies: the file is written against the library's
// module tree, so the example re-exports exactly those names at its own crate root.
use peacemaker_isa::kernels::iu4_gemm;
use peacemaker_isa::{
    arch, insn, plan, reg, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
mod kernels {
    pub use peacemaker_isa::kernels::common;
}
#[path = "../src/kernels/mq_i4_moe_gemv_gfx1100.rs"]
mod mq_i4_moe_gemv_gfx1100;

use mq_i4_moe_gemv_gfx1100::{contract, emit_module, module_specs, Spec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: mq_i4_moe_gemv_gfx1100_emit <output-dir>")?,
    );
    fs::create_dir_all(&dir)?;
    let target = Arch::Gfx1100;
    let (emitted, text, proof): (Vec<Emitted>, String, iu4_gemm::ModuleProof) =
        emit_module(target)?;
    let specs = module_specs(target);
    if emitted.len() != specs.len() {
        return Err("module emission returned a different number of kernels".into());
    }

    let source = dir.join("b2.k1.s");
    fs::write(&source, &text)?;
    fs::write(
        dir.join("b2.k1.proof.json"),
        serde_json::to_vec_pretty(&proof)?,
    )?;
    let object = dir.join("b2.k1.co");
    fs::write(&object, peacemaker_isa::native::assemble(&text, target)?)?;

    let mut summary = Vec::new();
    for (spec, e) in specs.iter().zip(&emitted) {
        let file = dir.join(format!(
            "mq_i4_moe_gemv.{}.{}.contract.json",
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
            "module": Spec::module(target), "source": source, "object": object, "s_text_sha256": proof.s_text_sha256, "entries": summary,
        }))?
    );

    m7(&dir, &object, &specs)
}

/// M7 byte obligations of every symbol of the linked object, written to `b2.k1.m7.json`.
#[cfg(feature = "toolchain")]
fn m7(
    dir: &std::path::Path,
    object: &std::path::Path,
    specs: &[Spec],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut receipts = std::collections::BTreeMap::new();
    for spec in specs {
        let symbol = spec.symbol();
        let receipt = peacemaker_isa::pm_check::m7(object, spec.arch.name(), &symbol)?;
        receipts.insert(symbol, receipt);
    }
    let receipts = serde_json::to_vec_pretty(&receipts)?;
    fs::write(dir.join("b2.k1.m7.json"), &receipts)?;
    println!("{}", String::from_utf8_lossy(&receipts));
    Ok(())
}

#[cfg(not(feature = "toolchain"))]
fn m7(
    _dir: &std::path::Path,
    _object: &std::path::Path,
    _specs: &[Spec],
) -> Result<(), Box<dyn std::error::Error>> {
    Err("M7 certification requires --features toolchain (pm_check is feature-gated); no b2.k1.m7.json was written".into())
}
