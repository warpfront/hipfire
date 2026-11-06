// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cargo run -p peacemaker-isa --release --features lab --example ceil_probe_emit -- --out-dir DIR`: emits the Builder `v_ceil_f32` device probe
//! (`kernels/ceil_probe.rs`) for gfx1100, gfx1151 and gfx1201 with the native assembler and
//! linker (no ROCm tools): `DIR/pm_ceil_probe_<arch>.{s,co,hxaco}` plus one JSON line per arch with
//! the sha256 of each. The device runner is
//! `crates/peacemaker-ir/isa/emu/numerics-probe/ceil/ceil_pm_run.hip`.
use peacemaker_isa::{kernels::ceil_probe, native, Arch};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut out = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--out-dir" => out = Some(PathBuf::from(args.next().ok_or("--out-dir needs a value")?)),
            _ => return Err("usage: ceil_probe_emit --out-dir DIR".into()),
        }
    }
    let dir = out.ok_or("usage: ceil_probe_emit --out-dir DIR")?;
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        let emitted = ceil_probe::emit(arch)?;
        let co = native::assemble(&emitted.s_text, arch)?;
        let hxaco = native::bundle(&co, arch, native::DEFAULT_HOST_TARGET);
        let stem = ceil_probe::symbol(arch);
        for (ext, bytes) in [
            ("s", emitted.s_text.as_bytes()),
            ("co", &co),
            ("hxaco", &hxaco),
        ] {
            let p = dir.join(format!("{stem}.{ext}"));
            fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        println!(
            "{}",
            serde_json::json!({
                "arch": arch.name(), "symbol": stem, "s_sha256": sha(emitted.s_text.as_bytes()),
                "co_sha256": sha(&co), "hxaco_sha256": sha(&hxaco),
                "builder_git_sha": emitted.proof.builder_git_sha,
            })
        );
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ceil_probe_emit: {e}");
        std::process::exit(1);
    }
}
