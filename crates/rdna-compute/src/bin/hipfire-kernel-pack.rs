// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compile explicit (module, symbol, effective-source) jobs to indexed objects.
//! The source must be byte-identical to the string passed to ensure_kernel.
//!
//! hipfire-kernel-pack --arch gfx1201 --output bin/kernels/compiled/gfx1201 \
//!   --extra-flags '-DIU4_A4_CANDIDATES=2' \
//!   --kernel rmsnorm_f32:rmsnorm_f32:kernels/src/rmsnorm.hip

use std::path::PathBuf;
use std::process::ExitCode;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut arch = None;
    let mut output = None;
    let mut extra_flags = String::new();
    let mut registry = None;
    let mut jobs: Vec<(String, Vec<String>, PathBuf)> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arch" => arch = Some(args.next().ok_or("--arch needs a value")?),
            "--output" => output = Some(PathBuf::from(args.next().ok_or("--output needs a value")?)),
            "--extra-flags" => extra_flags = args.next().ok_or("--extra-flags needs a value")?,
            "--registry" => registry = Some(PathBuf::from(args.next().ok_or("--registry needs a path")?)),
            "--kernel" => {
                let value = args.next().ok_or("--kernel needs module:symbol[,symbol...]:source")?;
                let (module, rest) = value.split_once(':').ok_or("--kernel needs module:symbol[,symbol...]:source")?;
                let (symbols, source) = rest.split_once(':').ok_or("--kernel needs module:symbol[,symbol...]:source")?;
                let symbols = symbols.split(',').map(str::to_owned).collect::<Vec<_>>();
                if module.is_empty() || symbols.iter().any(|s| s.is_empty()) || source.is_empty() {
                    return Err(format!("invalid --kernel: {value}"));
                }
                jobs.push((module.to_owned(), symbols, PathBuf::from(source)));
            }
            "--help" | "-h" => {
                eprintln!("Usage: hipfire-kernel-pack --arch gfx1201 --output <arch-dir> [--registry registry.tsv] [--kernel module:symbol[,symbol...]:effective-source.hip ...] [--extra-flags flags]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    let arch = arch.ok_or("missing --arch")?;
    let output = output.ok_or("missing --output")?;
    if let Some(path) = registry {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        for (line_no, line) in text.lines().enumerate() {
            let columns = line.split('\t').collect::<Vec<_>>();
            if columns.len() != 6 {
                return Err(format!("{}:{}: expected six TSV columns", path.display(), line_no + 1));
            }
            let [entry_arch, module, symbols, source_path, flags, profile] = columns.as_slice() else {
                unreachable!()
            };
            if *entry_arch != arch || module.is_empty() || source_path.is_empty() {
                return Err(format!("{}:{}: invalid arch/module/source", path.display(), line_no + 1));
            }
            let source = std::fs::read_to_string(source_path)
                .map_err(|e| format!("{}: {e}", source_path))?;
            let recipe = rdna_compute::KernelCompiler::recipe_for_source(&arch, module, &source, &extra_flags);
            if recipe.flags.join(" ") != *flags || recipe.scheduler_profile.as_deref() != Some(profile) {
                return Err(format!("{}:{}: registry flags/profile differ from compiler recipe", path.display(), line_no + 1));
            }
            jobs.push((
                (*module).to_owned(),
                symbols.split(',').map(str::to_owned).collect(),
                PathBuf::from(source_path),
            ));
        }
    }
    if jobs.is_empty() {
        return Err("at least one --kernel is required".into());
    }
    // Never silently publish an object built with one arch under another arch's path.
    if output.file_name().and_then(|s| s.to_str()) != Some(arch.as_str()) {
        return Err(format!("--output must end in architecture directory {arch}"));
    }
    let mut compiler = rdna_compute::KernelCompiler::new(&arch, extra_flags)
        .map_err(|e| e.to_string())?;
    for (module, symbols, source_path) in jobs {
        let source = std::fs::read_to_string(&source_path)
            .map_err(|e| format!("{}: {e}", source_path.display()))?;
        let object = compiler.pack_to(&module, &source, &symbols, &output)
            .map_err(|e| format!("{module}: {e}"))?;
        eprintln!("packaged {module} [{}] at {}", symbols.join(","), object.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hipfire-kernel-pack: {error}");
            ExitCode::FAILURE
        }
    }
}
