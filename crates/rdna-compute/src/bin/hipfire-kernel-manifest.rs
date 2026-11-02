// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Verify a packaged `<arch>` directory against this build's registry and
//! write the release pack manifest (see `scripts/build-kernel-pack.sh`).
//!
//! hipfire-kernel-manifest --arch gfx1201 --dir kernels/compiled/gfx1201 \
//!   --tag v0.4.0 --commit <sha> --rocm-root /opt/rocm --out manifest.json \
//!   [--extra-flags F] [--rocm-min 7.2 --rocm-max-exclusive 8.0]

use rdna_compute::kernel_pack::{self, PackManifest, KERNEL_CACHE_ABI, PACK_FORMAT};
use std::path::PathBuf;
use std::process::ExitCode;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut arch, mut dir, mut tag, mut commit, mut rocm_root, mut out) =
        (None, None, None, None, None, None);
    let (mut extra_flags, mut rocm_min, mut rocm_max) = (None, None, None);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--arch" => arch = Some(value()?),
            "--dir" => dir = Some(PathBuf::from(value()?)),
            "--tag" => tag = Some(value()?),
            "--commit" => commit = Some(value()?),
            "--rocm-root" => rocm_root = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--extra-flags" => extra_flags = Some(value()?),
            "--rocm-min" => rocm_min = Some(value()?),
            "--rocm-max-exclusive" => rocm_max = Some(value()?),
            "--help" | "-h" => {
                eprintln!("Usage: hipfire-kernel-manifest --arch <gfx> --dir <arch-dir> --tag <tag> --commit <sha> --rocm-root <root> --out <manifest.json> [--extra-flags F] [--rocm-min X.Y --rocm-max-exclusive X.Y]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    let arch = arch.ok_or("missing --arch")?;
    let dir = dir.ok_or("missing --dir")?;
    let tag = tag.ok_or("missing --tag")?;
    let commit = commit.ok_or("missing --commit")?;
    let rocm_root = rocm_root.ok_or("missing --rocm-root")?;
    let out = out.ok_or("missing --out")?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "--commit must be a full 40-hex commit, got {commit}"
        ));
    }
    // Same derivation as hipfire-kernel-registry and Gpu::init.
    let extra_flags = extra_flags
        .unwrap_or_else(|| rdna_compute::FeatureFlags::from_active_config(&arch).hipcc_extra_flags);
    // Same fallback as hipfire-rocm-resolve and `hipfire setup`: a TheRock
    // `/opt/rocm` holds no `.info/version`; its versioned `core-X.Y` root does.
    let rocm_version = hipfire_config::rocm::version_for_root(&rocm_root)
        .or_else(hipfire_config::rocm::version)
        .ok_or_else(|| format!("{}: no ROCm version (.info/version)", rocm_root.display()))?;
    let (default_min, default_max) = kernel_pack::default_rocm_range(&rocm_version)
        .ok_or_else(|| format!("unparseable ROCm version {rocm_version}"))?;
    let rocm_min = rocm_min.unwrap_or(default_min);
    let rocm_max_exclusive = rocm_max.unwrap_or(default_max);
    if !kernel_pack::rocm_admitted(&rocm_version, &rocm_min, &rocm_max_exclusive) {
        return Err(format!(
            "build ROCm {rocm_version} is outside the declared range [{rocm_min}, {rocm_max_exclusive})"
        ));
    }

    // Every object must come from this machine's compiler: the manifest
    // describes the toolchain that actually built the pack.
    let local = rdna_compute::KernelCompiler::local_toolchain_id().ok_or(
        "no device compiler resolves; a pack must be verified against its build toolchain",
    )?;
    let verified = kernel_pack::verify_dir(&arch, &extra_flags, &dir, Some(&local))?;
    let hip_version = verified
        .toolchain_id
        .strip_prefix("HIP version:")
        .map(str::trim)
        .unwrap_or(&verified.toolchain_id)
        .to_owned();
    let manifest = PackManifest {
        format: PACK_FORMAT,
        tag,
        commit,
        arch,
        modules: verified.modules,
        kernel_cache_abi: KERNEL_CACHE_ABI,
        extra_flags,
        toolchain_id: verified.toolchain_id.clone(),
        hip_version,
        rocm_version,
        code_object_version: verified.code_object_version,
        rocm_min,
        rocm_max_exclusive,
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    std::fs::write(&out, bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "manifest: {} {} modules, code object v{}, `{}`, ROCm [{}, {})",
        manifest.arch,
        manifest.modules,
        manifest.code_object_version,
        manifest.toolchain_id,
        manifest.rocm_min,
        manifest.rocm_max_exclusive
    );
    Ok(())
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
