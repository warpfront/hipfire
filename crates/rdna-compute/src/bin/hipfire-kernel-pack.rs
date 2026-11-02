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
use std::sync::atomic::{AtomicUsize, Ordering};

use rdna_compute::compile_jobs::{
    available_memory, compile_job_budget, parse_job_override, BYTES_PER_JOB,
};

fn worker_count(
    cores: usize,
    available_bytes: Option<u64>,
    override_jobs: Option<&str>,
) -> Result<usize, String> {
    let jobs = parse_job_override("HIPFIRE_PACK_JOBS", override_jobs)?;
    // Packaging historically retains its last worker even with no free RAM.
    // Unlike overlapping model loads it has no caller-owned weight reserve.
    compile_job_budget(
        cores,
        available_bytes.map(|bytes| bytes.max(BYTES_PER_JOB)),
        0,
        jobs,
    )
}

fn host_jobs() -> Result<usize, String> {
    let cores = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let memory = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            let zoneinfo = rdna_compute::uma_memory::read_zoneinfo();
            available_memory(&text, zoneinfo.as_deref())
        });
    let override_jobs = hipfire_config::developer_var("HIPFIRE_PACK_JOBS").ok();
    worker_count(cores, memory, override_jobs.as_deref())
}

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
            "--output" => {
                output = Some(PathBuf::from(args.next().ok_or("--output needs a value")?))
            }
            "--extra-flags" => extra_flags = args.next().ok_or("--extra-flags needs a value")?,
            "--registry" => {
                registry = Some(PathBuf::from(args.next().ok_or("--registry needs a path")?))
            }
            "--kernel" => {
                let value = args
                    .next()
                    .ok_or("--kernel needs module:symbol[,symbol...]:source")?;
                let (module, rest) = value
                    .split_once(':')
                    .ok_or("--kernel needs module:symbol[,symbol...]:source")?;
                let (symbols, source) = rest
                    .split_once(':')
                    .ok_or("--kernel needs module:symbol[,symbol...]:source")?;
                let symbols = symbols.split(',').map(str::to_owned).collect::<Vec<_>>();
                if module.is_empty() || symbols.iter().any(|s| s.is_empty()) || source.is_empty() {
                    return Err(format!("invalid --kernel: {value}"));
                }
                jobs.push((module.to_owned(), symbols, PathBuf::from(source)));
            }
            "--print-jobs" => {
                println!("{}", host_jobs()?);
                return Ok(());
            }
            "--help" | "-h" => {
                eprintln!("Usage: hipfire-kernel-pack --arch gfx1201 --output <arch-dir> [--registry registry.tsv] [--kernel module:symbol[,symbol...]:effective-source.hip ...] [--extra-flags flags]\nHIPFIRE_PACK_JOBS overrides host-scaled workers (bounded by cores and available RAM); --print-jobs prints this budget.");
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    let arch = arch.ok_or("missing --arch")?;
    let output = output.ok_or("missing --output")?;
    if let Some(path) = registry {
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (line_no, line) in text.lines().enumerate() {
            let columns = line.split('\t').collect::<Vec<_>>();
            if columns.len() != 6 {
                return Err(format!(
                    "{}:{}: expected six TSV columns",
                    path.display(),
                    line_no + 1
                ));
            }
            let [entry_arch, module, symbols, source_path, flags, profile] = columns.as_slice()
            else {
                unreachable!()
            };
            if *entry_arch != arch || module.is_empty() || source_path.is_empty() {
                return Err(format!(
                    "{}:{}: invalid arch/module/source",
                    path.display(),
                    line_no + 1
                ));
            }
            let source = std::fs::read_to_string(source_path)
                .map_err(|e| format!("{}: {e}", source_path))?;
            let recipe = rdna_compute::KernelCompiler::recipe_for_source(
                &arch,
                module,
                &source,
                &extra_flags,
            );
            if recipe.flags.join(" ") != *flags
                || recipe.scheduler_profile.as_deref() != Some(profile)
            {
                return Err(format!(
                    "{}:{}: registry flags/profile differ from compiler recipe",
                    path.display(),
                    line_no + 1
                ));
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
        return Err(format!(
            "--output must end in architecture directory {arch}"
        ));
    }
    // Each module has its own atomic object/hash/index publication. Keep repeat
    // requests on one worker to preserve their serial last-writer semantics.
    let mut module_groups = std::collections::HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, (module, _, _)) in jobs.iter().enumerate() {
        let group = *module_groups.entry(module.as_str()).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(index);
    }
    let workers = host_jobs()?.min(groups.len());
    eprintln!("packaging {} modules with {workers} workers", jobs.len());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(|| -> Result<(), String> {
                let mut compiler = rdna_compute::KernelCompiler::new(&arch, extra_flags.clone())
                    .map_err(|e| e.to_string())?;
                while let Some(group) = groups.get(next.fetch_add(1, Ordering::Relaxed)) {
                    for &index in group {
                        let (module, symbols, source_path) = &jobs[index];
                        let source = std::fs::read_to_string(source_path)
                            .map_err(|e| format!("{}: {e}", source_path.display()))?;
                        let object = compiler
                            .pack_to(module, &source, symbols, &output)
                            .map_err(|e| format!("{module}: {e}"))?;
                        eprintln!(
                            "packaged {module} [{}] at {}",
                            symbols.join(","),
                            object.display()
                        );
                    }
                }
                Ok(())
            }));
        }
        let mut error = None;
        // Join every worker even after failure; no background compile survives
        // the error. The release wrapper only manifests a successful arch.
        for handle in handles {
            match handle
                .join()
                .map_err(|_| "pack worker panicked".to_owned())
                .and_then(|result| result)
            {
                Ok(()) => {}
                Err(message) => {
                    error.get_or_insert(message);
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_reserve_scales_and_never_removes_last_worker() {
        for (cores, expected) in [(1, 1), (8, 7), (16, 14), (64, 56)] {
            assert_eq!(worker_count(cores, Some(u64::MAX), None).unwrap(), expected);
        }
    }

    #[test]
    fn memory_and_cpu_bound_overrides() {
        assert_eq!(worker_count(64, Some(3 * BYTES_PER_JOB), None).unwrap(), 3);
        assert_eq!(worker_count(64, Some(0), None).unwrap(), 1);
        assert_eq!(worker_count(64, None, None).unwrap(), 1);
        assert_eq!(worker_count(8, Some(u64::MAX), Some("100")).unwrap(), 8);
        assert_eq!(
            worker_count(64, Some(3 * BYTES_PER_JOB), Some("20")).unwrap(),
            3
        );
        assert_eq!(worker_count(64, Some(u64::MAX), Some("1")).unwrap(), 1);
        for value in ["0", "-1", "no", ""] {
            assert!(worker_count(64, Some(u64::MAX), Some(value)).is_err());
        }
    }

    #[test]
    fn parses_available_not_unused_memory() {
        assert_eq!(
            available_memory("MemFree: 1 kB\nMemAvailable: 3145728 kB\n", None),
            Some(2 * BYTES_PER_JOB)
        );
        assert_eq!(available_memory("MemFree: 100 kB\n", None), None);
        assert_eq!(available_memory("MemAvailable: invalid kB\n", None), None);
    }
}
