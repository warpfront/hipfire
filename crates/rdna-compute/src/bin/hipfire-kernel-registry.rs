// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Export exact runtime HIP sources for the sidecar packaging script.
//! Usage: hipfire-kernel-registry --arch gfx1201 --out-dir ./kernel-sources
//!        [--extra-flags "-D..."].
//! registry.tsv: arch TAB module TAB comma-separated symbols TAB absolute
//! source path TAB space-separated core hipcc flags TAB scheduler profile.

use rdna_compute::kernel_registry;
use std::{env, fs, path::PathBuf, process::ExitCode};

fn run() -> Result<(), String> {
    let mut arch = None;
    let mut out = None;
    let mut extra_flags: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arch" => arch = args.next(),
            "--out-dir" => out = args.next().map(PathBuf::from),
            "--extra-flags" => extra_flags = Some(args.next().ok_or("missing --extra-flags value")?),
            "--help" | "-h" => {
                println!("Usage: hipfire-kernel-registry --arch <gfx> --out-dir <dir> [--extra-flags <flags>]");
                return Ok(());
            }
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    let arch = arch.ok_or("missing --arch")?;
    let out = out.ok_or("missing --out-dir")?;
    // Match Gpu::init's per-arch derived compiler flags, not only the raw
    // HIPFIRE_HIPCC_EXTRA_FLAGS override.
    let extra_flags = extra_flags.unwrap_or_else(|| {
        rdna_compute::FeatureFlags::from_active_config(&arch).hipcc_extra_flags
    });
    let mut lines = String::new();
    let mut entries = kernel_registry::entries(&arch, &extra_flags)
        .map_err(|e| format!("unsupported registry architecture: {e:?}"))?;
    fs::create_dir_all(&out).map_err(|e| format!("creating {}: {e}", out.display()))?;
    let out = out.canonicalize().map_err(|e| format!("resolving {}: {e}", out.display()))?;
    entries.sort_by_key(|entry| entry.module);
    fs::write(out.join("extra-flags.txt"), &extra_flags)
        .map_err(|e| format!("writing extra-flags.txt: {e}"))?;
    for entry in entries {
        let path = out.join(format!("{}.hip", entry.module));
        fs::write(&path, entry.source.as_bytes())
            .map_err(|e| format!("writing {}: {e}", path.display()))?;
        lines.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            entry.arch,
            entry.module,
            entry.symbols.join(","),
            path.display(),
            entry.flags.join(" "),
            entry.scheduler_profile.as_deref().unwrap_or("")
        ));
    }
    fs::write(out.join("registry.tsv"), lines).map_err(|e| format!("writing registry.tsv: {e}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR: {error}");
            ExitCode::FAILURE
        }
    }
}
