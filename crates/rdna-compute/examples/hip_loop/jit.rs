// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The offline compile of `hip_loop`: a source goes through rdna-compute's own
//! `KernelCompiler` for gfx1100 (`compile_to_paths`: `core_hipcc_flags`, the
//! ROCm include/`--hip-path` flags, the module flags of `module_flags_for` and
//! the scheduler profile of the module name, per-source
//! `HIPFIRE_COMPILER_FLAGS:` comments), exactly the runtime JIT's argv, no
//! device. The incumbent built from the shipped source is therefore the object
//! production JITs.
//!
//! Included with `#[path = "hip_loop/jit.rs"] mod jit;`.
#![allow(dead_code)]

use crate::elf::{self, KernelInfo};
use crate::node::{err, Incumbent, Result, ARCH};
use rdna_compute::KernelCompiler;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One compiled object.
pub struct Built {
    pub tag: String,
    /// The object as the compiler wrote it (offload bundle): what the loader gets.
    pub image: Vec<u8>,
    pub image_path: PathBuf,
    pub image_sha256: String,
    pub kernels: BTreeMap<String, KernelInfo>,
    /// The complete hipcc invocation, compiler first.
    pub argv: Vec<String>,
    pub source_path: PathBuf,
}

impl Built {
    pub fn kernel(&self, symbol: &str) -> Result<&KernelInfo> {
        self.kernels.get(symbol).ok_or_else(|| {
            let have: Vec<&str> = self.kernels.keys().map(String::as_str).collect();
            format!(
                "{}: the object does not define kernel `{symbol}` (defines {} kernel(s): {})",
                self.tag,
                have.len(),
                have.iter().take(8).copied().collect::<Vec<_>>().join(", ")
            )
        })
    }
}

/// Keep the runtime compiler's side effects (hot cache seeding, cache dir) inside
/// `workdir` instead of `$HOME`, unless the caller already chose `HIPFIRE_KERNEL_CACHE` /
/// `HIPFIRE_HOME` (then those are honoured as set). Offline mode only: in GPU mode
/// the device runtime keeps its normal cache. Call before the first compile, before
/// other threads start.
pub fn isolate_caches(workdir: &Path) -> Result<()> {
    for (var, sub) in [
        ("HIPFIRE_KERNEL_CACHE", "kernel-cache"),
        ("HIPFIRE_HOME", "home"),
    ] {
        if std::env::var_os(var).is_none() {
            let dir = workdir.join(sub);
            std::fs::create_dir_all(&dir).map_err(err)?;
            std::env::set_var(var, &dir);
        }
    }
    Ok(())
}

/// Compile `source` as the incumbent's module (same module name, so the same
/// module flags, same `extra_flags`) into `workdir/<tag>/`, for gfx1100.
pub fn compile(workdir: &Path, tag: &str, inc: &Incumbent, source: &str) -> Result<Built> {
    compile_for(ARCH, workdir, tag, inc, source)
}

/// [`compile`] for `arch` (the node's own arch).
pub fn compile_for(
    arch: &str,
    workdir: &Path,
    tag: &str,
    inc: &Incumbent,
    source: &str,
) -> Result<Built> {
    let dir = workdir.join(tag);
    std::fs::create_dir_all(&dir).map_err(err)?;
    let src = dir.join(format!("{}.hip", inc.module));
    let obj = dir.join(format!("{}.hsaco", inc.module));
    let _ = std::fs::remove_file(&obj);
    let kc = KernelCompiler::new(arch, inc.extra_flags.clone()).map_err(err)?;
    kc.compile_to_paths(&inc.module, source, &src, &obj)
        .map_err(|e| format!("{tag}: {e}"))?;
    let argv = kc.jit_argv(&inc.module, source, &src, &obj);
    let image = std::fs::read(&obj).map_err(|e| format!("{tag}: read {}: {e}", obj.display()))?;
    if image.is_empty() {
        return Err(format!("{tag}: the compiler wrote an empty object"));
    }
    let kernels = elf::kernels(&image, arch).map_err(|e| format!("{tag}: {e}"))?;
    Ok(Built {
        tag: tag.to_owned(),
        image_sha256: elf::sha256_hex(&image),
        image,
        image_path: obj,
        kernels,
        argv,
        source_path: src,
    })
}

/// Compile the baseline and candidate sources concurrently (independent
/// compiler instances, separate directories).
pub fn compile_pair(workdir: &Path, inc: &Incumbent, cand_source: &str) -> Result<(Built, Built)> {
    std::thread::scope(|s| {
        let b = s.spawn(|| compile(workdir, "baseline", inc, &inc.source));
        let c = s.spawn(|| compile(workdir, "candidate", inc, cand_source));
        let b = b
            .join()
            .map_err(|_| "baseline compile thread panicked".to_owned())?;
        let c = c
            .join()
            .map_err(|_| "candidate compile thread panicked".to_owned())?;
        Ok((b?, c?))
    })
}

pub fn hipcc_available() -> bool {
    KernelCompiler::local_toolchain_id().is_some()
}

/// A prebuilt candidate code object (ELF `.co` or offload bundle): inspected,
/// not compiled. `argv` is empty and `source_path` is the object itself.
pub fn from_object(tag: &str, path: &Path) -> Result<Built> {
    from_object_for(ARCH, tag, path)
}

/// [`from_object`] for `arch` (the node's own arch).
pub fn from_object_for(arch: &str, tag: &str, path: &Path) -> Result<Built> {
    let image = std::fs::read(path).map_err(|e| format!("{tag}: read {}: {e}", path.display()))?;
    if image.is_empty() {
        return Err(format!("{tag}: {} is empty", path.display()));
    }
    let kernels =
        elf::kernels(&image, arch).map_err(|e| format!("{tag}: {}: {e}", path.display()))?;
    Ok(Built {
        tag: tag.to_owned(),
        image_sha256: elf::sha256_hex(&image),
        image,
        image_path: path.to_path_buf(),
        kernels,
        argv: Vec::new(),
        source_path: path.to_path_buf(),
    })
}

/// `FILE.co` -> `FILE.co.json`.
pub fn sidecar_path(co: &Path) -> PathBuf {
    let mut s = co.as_os_str().to_owned();
    s.push(".json");
    PathBuf::from(s)
}

/// What a compiled object certifies about itself: the module, the sha256 of the
/// SOURCE it was compiled from, the toolchain, the object's own sha256 and, per
/// kernel, the `.text` sha256 and explicit kernarg layout. `--base-object` checks
/// a code object against this.
pub fn sidecar_json(
    node: &str,
    inc: &Incumbent,
    b: &Built,
    source: &str,
    header: &BTreeMap<String, String>,
) -> serde_json::Value {
    sidecar_json_for(ARCH, node, inc, b, source, header)
}

/// [`sidecar_json`] for `arch` (the node's own arch).
pub fn sidecar_json_for(
    arch: &str,
    node: &str,
    inc: &Incumbent,
    b: &Built,
    source: &str,
    header: &BTreeMap<String, String>,
) -> serde_json::Value {
    let kernels: serde_json::Map<String, serde_json::Value> = b
        .kernels
        .iter()
        .map(|(name, k)| {
            (
                name.clone(),
                serde_json::json!({
                    "code_sha256": k.code_sha256,
                    "kernarg_bytes": k.kernarg_bytes,
                    "explicit_args": k.json()["explicit_args"],
                }),
            )
        })
        .collect();
    serde_json::json!({
        "schema": 1,
        "tool": "hip_loop",
        "node": node,
        "module": inc.module,
        "symbol": inc.symbol,
        "arch": arch,
        "source_sha256": elf::sha256_hex(source.as_bytes()),
        "toolchain_id": KernelCompiler::local_toolchain_id(),
        "object_sha256": b.image_sha256,
        "header": header,
        "hipcc_argv": b.argv,
        "kernels": kernels,
    })
}

/// Copy the compiled object to `co` and write its sidecar next to it.
pub fn publish(co: &Path, b: &Built, sidecar: &serde_json::Value) -> Result<()> {
    std::fs::copy(&b.image_path, co).map_err(|e| format!("write {}: {e}", co.display()))?;
    let text = serde_json::to_string_pretty(sidecar).map_err(err)?;
    let side = sidecar_path(co);
    std::fs::write(&side, text).map_err(|e| format!("write {}: {e}", side.display()))
}
