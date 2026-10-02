// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Release kernel packs: one architecture's indexed `kernels/compiled/<arch>`
//! tree built from an exact commit, plus the manifest an installer checks
//! before trusting it. `scripts/build-kernel-pack.sh` writes packs through
//! `hipfire-kernel-manifest`; `hipfire setup` and `install.ps1` install them
//! through `hipfire kernel-pack install`. Both sides use this module, and
//! per-object checks go through the runtime's own index check, so the
//! packager, the installer and the loader cannot disagree about validity.

use crate::compiler::KernelCompiler;
pub use crate::compiler::KERNEL_CACHE_ABI;
use crate::kernel_registry::{self, KernelEntry};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

/// Manifest schema version. Bump on any field change.
pub const PACK_FORMAT: u32 = 1;
/// Manifest path inside the tarball, beside the `<arch>/` directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Provenance and admission range of one pack. Serialized as `manifest.json`
/// inside the tarball (covered by the published SHA-256) and as a loose
/// `.manifest.json` release asset for inspection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackManifest {
    pub format: u32,
    pub tag: String,
    /// Full commit the objects were compiled from.
    pub commit: String,
    pub arch: String,
    /// Registry module count; every module has `.hsaco`, `.hash` and `.index.json`.
    pub modules: usize,
    pub kernel_cache_abi: u32,
    /// Registry extra hipcc flags (default `FeatureFlags` for `arch`).
    pub extra_flags: String,
    /// `hipcc --version` first line, as recorded in every index.
    pub toolchain_id: String,
    pub hip_version: String,
    /// `<rocm root>/.info/version` of the build toolchain.
    pub rocm_version: String,
    /// AMDGPU code-object version of every object (e.g. 6).
    pub code_object_version: u32,
    /// Oldest ROCm release the pack admits (inclusive).
    pub rocm_min: String,
    /// First ROCm release the pack no longer admits (exclusive).
    pub rocm_max_exclusive: String,
}

/// Release asset stem: `<stem>.tar.gz`, `<stem>.tar.gz.sha256`, `<stem>.manifest.json`.
pub fn asset_stem(tag: &str, arch: &str) -> String {
    format!("hipfire-kernels-{tag}-{arch}")
}

/// What a verified pack directory contains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPack {
    pub modules: usize,
    pub toolchain_id: String,
    pub code_object_version: u32,
}

/// Verify an installed-layout `<arch>` directory against the registry this
/// build compiles for `arch` and `extra_flags`.
///
/// Every registry module must be present with an index that matches the
/// checked-out source, recipe and object bytes; every object must share one
/// toolchain and one code-object version; and the directory must hold nothing
/// else. `local_toolchain_id` is the machine's device compiler identity, if
/// any: a pack from a different compiler build would be rejected by the
/// runtime and recompiled, so it is rejected here too.
pub fn verify_dir(
    arch: &str,
    extra_flags: &str,
    dir: &Path,
    local_toolchain_id: Option<&str>,
) -> Result<VerifiedPack, String> {
    let entries = kernel_registry::entries(arch, extra_flags)
        .map_err(|error| format!("no indexed kernel registry for {arch}: {error:?}"))?;
    verify_entries(&entries, arch, extra_flags, dir, local_toolchain_id)
}

pub(crate) fn verify_entries(
    entries: &[KernelEntry],
    arch: &str,
    extra_flags: &str,
    dir: &Path,
    local_toolchain_id: Option<&str>,
) -> Result<VerifiedPack, String> {
    let mut expected = BTreeSet::new();
    let mut toolchain: Option<String> = None;
    let mut code_object: Option<u32> = None;
    for entry in entries {
        let key =
            KernelCompiler::packaging_hash_for(arch, entry.module, entry.source(), extra_flags);
        let (object, recorded) = KernelCompiler::check_indexed_object(
            dir,
            arch,
            entry.module,
            entry.source(),
            entry.symbols,
            extra_flags,
            &key,
            local_toolchain_id.unwrap_or(""),
        )?
        .ok_or_else(|| format!("{}: module missing from pack", entry.module))?;
        match &toolchain {
            None => toolchain = Some(recorded),
            Some(first) if *first != recorded => {
                return Err(format!(
                    "{}: built by `{recorded}`, other modules by `{first}`",
                    entry.module
                ))
            }
            Some(_) => {}
        }
        let bytes = std::fs::read(&object).map_err(|e| format!("{}: {e}", object.display()))?;
        let version =
            code_object_version(&bytes).map_err(|e| format!("{}: {e}", object.display()))?;
        match code_object {
            None => code_object = Some(version),
            Some(first) if first != version => {
                return Err(format!(
                    "{}: code object v{version}, other modules v{first}",
                    entry.module
                ))
            }
            Some(_) => {}
        }
        for suffix in ["hsaco", "hash", "index.json"] {
            expected.insert(format!("{}.{suffix}", entry.module));
        }
    }
    // A stray unindexed pair would be seeded into the hot cache and loaded
    // without any index check, so the directory is exactly the registry.
    for item in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let item = item.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = item.file_name().to_string_lossy().into_owned();
        let kind = item.file_type().map_err(|e| format!("{name}: {e}"))?;
        if !kind.is_file() || !expected.contains(&name) {
            return Err(format!("{name}: not part of the {arch} registry"));
        }
    }
    Ok(VerifiedPack {
        modules: entries.len(),
        toolchain_id: toolchain.ok_or_else(|| format!("{arch}: empty registry"))?,
        code_object_version: code_object.ok_or_else(|| format!("{arch}: empty registry"))?,
    })
}

impl PackManifest {
    /// Admission checks that need no object bytes: schema, identity and the
    /// ROCm range. `local_rocm` is `<rocm root>/.info/version`, if known.
    pub fn check(&self, arch: &str, commit: &str, local_rocm: Option<&str>) -> Result<(), String> {
        if self.format != PACK_FORMAT {
            return Err(format!(
                "manifest format {} (installer reads {PACK_FORMAT})",
                self.format
            ));
        }
        if self.arch != arch {
            return Err(format!("pack is for {}, GPU is {arch}", self.arch));
        }
        if self.commit != commit {
            return Err(format!(
                "pack was built from {}, source is {commit}",
                self.commit
            ));
        }
        if self.kernel_cache_abi != KERNEL_CACHE_ABI {
            return Err(format!(
                "pack kernel cache ABI {} (runtime uses {KERNEL_CACHE_ABI})",
                self.kernel_cache_abi
            ));
        }
        let local = local_rocm.ok_or("local ROCm version unknown (no <root>/.info/version)")?;
        if !rocm_admitted(local, &self.rocm_min, &self.rocm_max_exclusive) {
            return Err(format!(
                "ROCm {local} is outside the pack's supported range [{}, {})",
                self.rocm_min, self.rocm_max_exclusive
            ));
        }
        Ok(())
    }

    /// The directory verification must agree with what the manifest claims.
    pub fn check_verified(&self, verified: &VerifiedPack) -> Result<(), String> {
        if self.modules != verified.modules
            || self.toolchain_id != verified.toolchain_id
            || self.code_object_version != verified.code_object_version
        {
            return Err(format!(
                "manifest claims {} modules/`{}`/v{}, objects are {}/`{}`/v{}",
                self.modules,
                self.toolchain_id,
                self.code_object_version,
                verified.modules,
                verified.toolchain_id,
                verified.code_object_version
            ));
        }
        Ok(())
    }
}

/// Numeric components of a dotted version, ignoring a non-numeric tail in each
/// component (`7.15.26333-0000000` → `[7, 15, 26333]`).
fn version_parts(version: &str) -> Option<Vec<u64>> {
    let parts = version
        .trim()
        .split('.')
        .map(|part| {
            let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
            digits.parse::<u64>().ok()
        })
        .collect::<Option<Vec<_>>>()?;
    (!parts.is_empty()).then_some(parts)
}

fn compare_versions(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    let len = a.len().max(b.len());
    let at = |v: &[u64], i: usize| v.get(i).copied().unwrap_or(0);
    (0..len)
        .map(|i| at(a, i).cmp(&at(b, i)))
        .find(|order| order.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// `min <= local < max_exclusive`, comparing dotted numeric versions.
pub fn rocm_admitted(local: &str, min: &str, max_exclusive: &str) -> bool {
    match (
        version_parts(local),
        version_parts(min),
        version_parts(max_exclusive),
    ) {
        (Some(local), Some(min), Some(max)) => {
            compare_versions(&local, &min).is_ge() && compare_versions(&local, &max).is_lt()
        }
        _ => false,
    }
}

/// Default admission range for a pack built with ROCm `version`: the build's
/// `major.minor` up to the next major release.
pub fn default_rocm_range(version: &str) -> Option<(String, String)> {
    let parts = version_parts(version)?;
    let major = parts[0];
    let minor = parts.get(1).copied().unwrap_or(0);
    Some((format!("{major}.{minor}"), format!("{}.0", major + 1)))
}

/// AMDGPU code-object version of a `--genco` object: an uncompressed clang
/// offload bundle holding one amdgcn ELF, or that ELF itself.
pub fn code_object_version(object: &[u8]) -> Result<u32, String> {
    const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    const ELFOSABI_AMDGPU_HSA: u8 = 64;
    let read_u64 = |at: usize| -> Result<usize, String> {
        let bytes = object
            .get(at..at.checked_add(8).ok_or("offload bundle offset overflow")?)
            .ok_or("truncated offload bundle")?;
        usize::try_from(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
            .map_err(|_| "offload bundle offset overflow".to_owned())
    };
    let elf = if object.starts_with(BUNDLE_MAGIC) {
        let count = read_u64(BUNDLE_MAGIC.len())?;
        let mut cursor = BUNDLE_MAGIC.len() + 8;
        let mut device = None;
        for _ in 0..count {
            let (offset, size, triple_len) = (
                read_u64(cursor)?,
                read_u64(cursor + 8)?,
                read_u64(cursor + 16)?,
            );
            let start = cursor + 24;
            let end = start
                .checked_add(triple_len)
                .ok_or("offload bundle offset overflow")?;
            let triple = object.get(start..end).ok_or("truncated offload bundle")?;
            cursor = end;
            if size > 0 && triple.windows(6).any(|window| window == b"amdgcn") {
                let stop = offset
                    .checked_add(size)
                    .ok_or("offload bundle offset overflow")?;
                device = Some(
                    object
                        .get(offset..stop)
                        .ok_or("truncated offload bundle entry")?,
                );
            }
        }
        device.ok_or("offload bundle has no amdgcn object")?
    } else {
        object
    };
    if elf.get(..4) != Some(b"\x7fELF".as_slice()) || elf.len() < 16 {
        return Err("not an ELF code object".to_owned());
    }
    if elf[7] != ELFOSABI_AMDGPU_HSA {
        return Err(format!("ELF OS/ABI {} is not AMDGPU HSA", elf[7]));
    }
    // EI_ABIVERSION: 2 → code object v4, 3 → v5, 4 → v6.
    Ok(u32::from(elf[8]) + 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_range_is_half_open_and_numeric() {
        assert!(rocm_admitted("10.0.0", "10.0", "11.0"));
        assert!(rocm_admitted("10.12.1", "10.0", "11.0"));
        assert!(!rocm_admitted("11.0.0", "10.0", "11.0"));
        assert!(!rocm_admitted("9.9.9", "10.0", "11.0"));
        // Lexical comparison would admit 7.10 below 7.2.
        assert!(rocm_admitted("7.10.0", "7.2", "8.0"));
        assert!(!rocm_admitted("7.1.1", "7.2", "8.0"));
        assert!(!rocm_admitted("unknown", "7.2", "8.0"));
        assert_eq!(
            default_rocm_range("10.0.0"),
            Some(("10.0".into(), "11.0".into()))
        );
        assert_eq!(
            default_rocm_range("7.2.4"),
            Some(("7.2".into(), "8.0".into()))
        );
    }
}
