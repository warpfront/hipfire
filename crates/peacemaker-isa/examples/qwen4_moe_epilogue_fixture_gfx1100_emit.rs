// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emits the gfx1100 K2 epilogue fixture module (B4-A): `qwen4_moe_epilogue_fixture_pm_gfx1100`,
//! entries `qwen4_moe_epilogue_fixture_{gate_up,down}{,_wrongsel}_gfx1100`.
//!
//! `cargo run -p peacemaker-isa --features lab --example qwen4_moe_epilogue_fixture_gfx1100_emit -- OUT_DIR`
//! writes `OUT_DIR/<module>.{s,co,hxaco,sha256.json}`; ABI and launch geometry are documented in
//! `epilogue_fixture.rs`. CPU only (no ROCm tools, no device).

#[path = "../src/kernels/qwen4_moe_sym_gfx1100/epilogue.rs"]
mod epilogue;
#[path = "../src/kernels/qwen4_moe_sym_gfx1100/epilogue_fixture.rs"]
mod epilogue_fixture;

use peacemaker_isa::{kernels::qwen4_moe_sym::Kind, native, Arch};
use sha2::{Digest, Sha256};

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn main() -> Result<(), String> {
    let out = std::env::args()
        .nth(1)
        .ok_or("usage: qwen4_moe_epilogue_fixture_gfx1100_emit OUT_DIR")?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let (emitted, text) = epilogue_fixture::emit_module()?;
    let elf = native::assemble(&text, Arch::Gfx1100)?;
    let bundle = native::bundle(&elf, Arch::Gfx1100, native::DEFAULT_HOST_TARGET);
    let name = epilogue_fixture::MODULE;
    for (ext, bytes) in [
        ("s", text.as_bytes()),
        ("co", elf.as_slice()),
        ("hxaco", bundle.as_slice()),
    ] {
        std::fs::write(format!("{out}/{name}.{ext}"), bytes).map_err(|e| e.to_string())?;
    }
    let entries: Vec<String> = [false, true]
        .into_iter()
        .flat_map(|w| [Kind::GateUp, Kind::Down].map(|k| epilogue_fixture::symbol(k, w)))
        .collect();
    let meta = serde_json::json!({
        "module": name, "arch": "gfx1100", "entries": entries,
        "sha256": { "s": hex(text.as_bytes()), "co": hex(&elf), "hxaco": hex(&bundle) },
    });
    std::fs::write(
        format!("{out}/{name}.sha256.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("{meta}");
    Ok(())
}
