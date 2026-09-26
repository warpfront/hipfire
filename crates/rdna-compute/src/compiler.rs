// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use radiowave::{CodeObjectCertification, ExistingCodeObjectRequest, SchedulerProfile};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// True when `a` and `b` name the same filesystem object. Lexical equality is
/// checked first; if both paths exist, canonicalization also equates aliases
/// (symlinks, `.` / `..` segments) so two-directory logic never treats one
/// underlying dir as two targets.
fn same_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// True when `path` is an existing nonempty regular file (usable HSACO blob).
fn nonempty_blob(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// Unique same-dir staging token (pid + counter + nanos) for parallel-safe temps.
fn unique_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}_{n}_{nanos}", std::process::id())
}

/// Shared on-disk kernel cache root: `$HIPFIRE_KERNEL_CACHE` when set,
/// otherwise `$HOME/.hipfire_kernels` so every worktree and daemon on the
/// machine shares one content-keyed store instead of recompiling per CWD.
/// Falls back to the legacy CWD-relative dir only when `HOME` is unset.
/// Pre-existing CWD `.hipfire_kernels` dirs are left alone — they simply go
/// unused, never deleted or migrated by this code.
fn default_cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_KERNEL_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".hipfire_kernels");
    }
    PathBuf::from(".hipfire_kernels")
}

/// Content-keyed stem for hot entries: module name plus the full cache hash
/// (source + flags + arch + toolchain + ABI). Distinct triples map to
/// distinct paths, so a nonempty hit needs no sidecar validation and two
/// daemons from different builds sharing one root never collide on a path.
fn hot_stem(name: &str, src_hash: &str) -> String {
    format!("{name}.{src_hash}")
}

/// Final hot object filename for a (`name`, `src_hash`) pair.
fn hot_object_name(name: &str, src_hash: &str) -> String {
    format!("{}.hsaco", hot_stem(name, src_hash))
}

/// Validated pair: nonempty regular HSACO **and** matching hash sidecar.
/// A hash alone never certifies a missing/empty/directory blob. A blob alone
/// is never treated as hash-validated.
fn pair_valid(hsaco: &Path, hash: &Path, src_hash: &str) -> bool {
    if !nonempty_blob(hsaco) {
        return false;
    }
    hash.exists() && {
        let stored = std::fs::read_to_string(hash).unwrap_or_default();
        stored.trim() == src_hash
    }
}

const PACK_INDEX_VERSION: u32 = 1;

/// One object per module; the index authenticates the bytes and records the
/// producing compiler separately from the portable lookup key.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackIndex {
    version: u32,
    module: String,
    symbols: Vec<String>,
    arch: String,
    source_sha256: String,
    flags: Vec<String>,
    scheduler_profile: String,
    cache_abi: u32,
    packaging_key: String,
    object_sha256: String,
    toolchain_id: String,
    toolchain_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileRecipe {
    pub arch: String,
    pub flags: Vec<String>,
    pub scheduler_profile: Option<String>,
    pub cache_abi: u32,
}

/// Transactionally publish a blob+hash pair into `dest_dir`.
///
/// Staging uses same-directory temps so `rename` is atomic on one volume.
/// Order: stage blob → verify nonempty → stage hash → **invalidate old hash**
/// → publish blob → publish hash. A crash mid-publish never leaves a hash that
/// certifies a missing/wrong-generation blob. `force` replaces even when the
/// destination pair already matches `src_hash`.
///
/// When `src_blob` is already the destination blob path, only the hash sidecar
/// is repaired (still via stage+rename). Never copies a path onto itself.
fn publish_pair(
    dest_dir: &Path,
    name: &str,
    src_blob: &Path,
    src_hash: &str,
    force: bool,
) -> Result<(), String> {
    let dest_hsaco = dest_dir.join(format!("{name}.hsaco"));
    let dest_hash = dest_dir.join(format!("{name}.hash"));

    if !force && pair_valid(&dest_hsaco, &dest_hash, src_hash) {
        return Ok(());
    }

    if !nonempty_blob(src_blob) {
        return Err(format!("{name}: source blob missing or empty"));
    }

    let token = unique_token();
    let tmp_hsaco = dest_dir.join(format!(".{name}.{token}.hsaco.tmp"));
    let tmp_hash = dest_dir.join(format!(".{name}.{token}.hash.tmp"));
    let cleanup = || {
        let _ = std::fs::remove_file(&tmp_hsaco);
        let _ = std::fs::remove_file(&tmp_hash);
    };

    let same_blob = same_path(src_blob, &dest_hsaco);

    if !same_blob {
        if let Err(e) = std::fs::copy(src_blob, &tmp_hsaco) {
            cleanup();
            return Err(format!("{name}: blob stage failed ({e})"));
        }
        if !nonempty_blob(&tmp_hsaco) {
            cleanup();
            return Err(format!("{name}: staged blob empty"));
        }
    }

    if let Err(e) = std::fs::write(&tmp_hash, src_hash) {
        cleanup();
        return Err(format!("{name}: hash stage failed ({e})"));
    }

    // Invalidate old hash before publishing a replacement blob so an old
    // sidecar can never certify a new generation.
    let _ = std::fs::remove_file(&dest_hash);

    if !same_blob {
        if let Err(e) = std::fs::rename(&tmp_hsaco, &dest_hsaco) {
            cleanup();
            return Err(format!("{name}: blob publish failed ({e})"));
        }
    }

    if let Err(e) = std::fs::rename(&tmp_hash, &dest_hash) {
        let _ = std::fs::remove_file(&tmp_hash);
        return Err(format!("{name}: hash publish failed ({e})"));
    }

    Ok(())
}

/// Publish only a nonempty blob into `dest_dir`, removing any orphan hash so
/// it cannot certify the copied content. Used for hashless cold seed fallback.
fn publish_blob_only(dest_dir: &Path, name: &str, src_blob: &Path) -> Result<(), String> {
    let dest_hsaco = dest_dir.join(format!("{name}.hsaco"));
    let dest_hash = dest_dir.join(format!("{name}.hash"));

    if !nonempty_blob(src_blob) {
        return Err(format!("{name}: source blob missing or empty"));
    }
    if same_path(src_blob, &dest_hsaco) {
        // Drop orphan hash that would certify this blob under the wrong key.
        let _ = std::fs::remove_file(&dest_hash);
        return Ok(());
    }

    let token = unique_token();
    let tmp_hsaco = dest_dir.join(format!(".{name}.{token}.hsaco.tmp"));
    if let Err(e) = std::fs::copy(src_blob, &tmp_hsaco) {
        let _ = std::fs::remove_file(&tmp_hsaco);
        return Err(format!("{name}: blob stage failed ({e})"));
    }
    if !nonempty_blob(&tmp_hsaco) {
        let _ = std::fs::remove_file(&tmp_hsaco);
        return Err(format!("{name}: staged blob empty"));
    }

    // Remove hash before blob publish so it cannot certify the new content.
    let _ = std::fs::remove_file(&dest_hash);

    if let Err(e) = std::fs::rename(&tmp_hsaco, &dest_hsaco) {
        let _ = std::fs::remove_file(&tmp_hsaco);
        return Err(format!("{name}: blob publish failed ({e})"));
    }
    Ok(())
}

fn publish_index(dir: &Path, index: &PackIndex) -> Result<(), String> {
    let path = dir.join(format!("{}.index.json", index.module));
    let temp = dir.join(format!(".{}.{}.index.tmp", index.module, unique_token()));
    let bytes = serde_json::to_vec_pretty(index).map_err(|e| e.to_string())?;
    std::fs::write(&temp, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&temp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        e.to_string()
    })
}

/// Seed hot from cold using **complete kernel pairs**, never independent
/// entry-by-entry copies.
///
/// - Orphan cold hashes and empty/directory blobs are ignored.
/// - Skip only when hot is already a nonempty pair with the **same** cold hash.
/// - Pair replacement publishes blob then hash as a unit (via `publish_pair`).
/// - Hashless nonempty cold may seed only when hot has no validated pair, and
///   any orphan hot hash is removed before the blob is published.
fn seed_hot_from_cold(cold: &Path, hot: &Path) -> std::io::Result<()> {
    if same_path(cold, hot) {
        return Ok(());
    }
    std::fs::create_dir_all(hot)?;

    for entry in std::fs::read_dir(cold)? {
        let entry = entry?;
        let src = entry.path();
        let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("");
        // Blob-centric: never iterate orphan .hash files as seed units.
        if ext != "hsaco" {
            continue;
        }
        if !nonempty_blob(&src) {
            continue;
        }
        let stem = match src.file_stem().and_then(|s| s.to_str()) {
            Some(s) if !s.is_empty() && !s.starts_with('.') => s,
            _ => continue,
        };
        // Index-managed objects must be checked at their original path: a
        // seed of only the pair would discard object SHA and toolchain identity.
        if cold.join(format!("{stem}.index.json")).exists() {
            continue;
        }

        let cold_hash_path = cold.join(format!("{stem}.hash"));
        let hot_hsaco = hot.join(format!("{stem}.hsaco"));
        let hot_hash = hot.join(format!("{stem}.hash"));

        let cold_hash = std::fs::read_to_string(&cold_hash_path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        match &cold_hash {
            Some(h) => {
                // Complete cold pair: skip only when hot already matches that hash.
                if pair_valid(&hot_hsaco, &hot_hash, h) {
                    continue;
                }
                if let Err(e) = publish_pair(hot, stem, &src, h, true) {
                    return Err(std::io::Error::other(e));
                }
            }
            None => {
                // Hashless cold fallback: only when hot has no validated pair
                // (nonempty blob + any hash sidecar written by a prior compile).
                let hot_has_validated_pair = nonempty_blob(&hot_hsaco) && hot_hash.exists();
                if hot_has_validated_pair {
                    continue;
                }
                if let Err(e) = publish_blob_only(hot, stem, &src) {
                    return Err(std::io::Error::other(e));
                }
            }
        }
    }
    Ok(())
}

/// Cache-key version. Bump when the kernel ABI or hipcc invocation changes in a
/// way that makes previously-cached `.hsaco` blobs incompatible, to force a clean
/// recompile instead of loading a stale "invalid device image".
const KERNEL_CACHE_ABI: u32 = 4;

/// Compiles HIP kernel sources to code objects, with caching.
///
/// Two-directory contract:
/// - **hot** (`cache_dir`): shared JIT workspace
///   (`$HOME/.hipfire_kernels/{arch}` by default). Hot filenames are
///   content-keyed (`{name}.{hash}.hsaco`), so a nonempty hit is by
///   construction the right blob. Preferred for lookup after seeding.
/// - **cold** (`cold_dir`): persistent install location
///   (`kernels/compiled/{arch}` under the installed binary). Writeback target
///   so install-time / post-update recompiles refresh the blobs that seed the
///   next process. Distinct from hot; must not be collapsed into the lookup dir.
///
/// `precompiled_dir` is the lookup view: hot when it has blobs, otherwise cold.
/// Tries that first (with hash validation), falls back to hipcc into hot, then
/// writeback to cold.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
    /// Lookup directory for pre-compiled blobs (hot when seeded, else cold).
    precompiled_dir: Option<PathBuf>,
    /// Persistent install dir (`kernels/compiled/{arch}`). Writeback target;
    /// kept even when lookup prefers the hot path.
    cold_dir: Option<PathBuf>,
    has_hipcc: bool,
    pub extra_flags: String,
    /// Exact gfx1151 JIT module names compiled in CU mode. This is an
    /// experiment boundary for per-code-object admission; other arches and
    /// unlisted modules retain their existing WGP-mode code objects.
    gfx1151_cumode_modules: HashSet<String>,
    /// Toolchain fingerprint (hipcc --version first line). Folded into the cache
    /// hash so blobs built by a different compiler/ROCm don't get reused across
    /// builds sharing one `.hipfire_kernels` dir (the "invalid device image" trap).
    toolchain_id: String,
    /// Resolved device-compiler binary. Bare "hipcc" when PATH provides it;
    /// otherwise an absolute path discovered under a resolved ROCm root.
    hipcc_bin: std::path::PathBuf,
    /// ROCM_PATH handed to the spawned compiler so it can find its own LLVM.
    /// None when the environment already sets it.
    rocm_env_root: Option<std::path::PathBuf>,
    /// Override scheduler profile applied to all kernels when set via
    /// `HIPFIRE_SCHED_PROFILE`. `None` selects the per-kernel table.
    sched_profile_override: Option<SchedulerProfile>,
}

impl KernelCompiler {
    pub fn new(arch: &str, extra_flags: String) -> HipResult<Self> {
        // Hot JIT cache defaults to the shared `$HOME/.hipfire_kernels` root
        // (overridable via `HIPFIRE_KERNEL_CACHE`), so parallel worktrees and
        // daemons on one machine share compiled kernels instead of each
        // recompiling into its own CWD dir. Sharing is safe because hot
        // filenames are content-keyed (`{name}.{hash}.hsaco`, hash covering
        // source + flags + arch + toolchain + ABI): distinct builds map to
        // distinct paths and never clobber each other's blobs, and all entry
        // writes are atomic (same-dir temp + rename) so a concurrent reader
        // only ever sees a complete entry. Per-arch subdirs still isolate
        // hetero processes (gfx906 + gfx1031) as before.
        let cache_root = default_cache_root();
        let cache_dir = cache_root.join(arch);
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;

        // Probe for pre-compiled kernels: exe-relative → CWD-relative → ~/.hipfire/bin/
        let precompiled_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .map(|dir| dir.join("kernels").join("compiled").join(arch))
            .filter(|p| p.is_dir())
            .or_else(|| {
                let cwd_path = PathBuf::from("kernels/compiled").join(arch);
                if cwd_path.is_dir() {
                    Some(cwd_path)
                } else {
                    None
                }
            })
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| {
                        PathBuf::from(h)
                            .join(".hipfire/bin/kernels/compiled")
                            .join(arch)
                    })
                    .filter(|p| p.is_dir())
            });

        // Seed the hot path from the persistent install location. Cold survives
        // reboots / process restarts; hot may not (or may be worktree-local).
        // Copy is incremental — only copies files not already present as a
        // JIT-validated hot pair (see seed_hot_from_cold skip rule).
        // Stale pairs are not wiped by update: hash mismatch in compile()
        // triggers hipcc and writeback refreshes cold in place.
        // cache_dir is already arch-keyed; the hot dir IS the cache dir.
        let cold_dir = precompiled_dir;
        let hot_dir = cache_dir.clone();
        if let Some(cold) = &cold_dir {
            if let Err(e) = seed_hot_from_cold(cold, &hot_dir) {
                eprintln!(
                    "  hot-path seed failed at {} ({e}) — falling back to install dir reads",
                    hot_dir.display()
                );
            }
        }
        // Prefer the hot-path dir when it exists and has contents.
        // This is what the `compile()` lookup uses from here on. Cold stays
        // on `cold_dir` for writeback even when lookup points at hot.
        let effective_precompiled = if hot_dir.is_dir()
            && std::fs::read_dir(&hot_dir)
                .map(|mut it| {
                    it.any(|e| {
                        e.map(|e| e.path().extension().map(|x| x == "hsaco").unwrap_or(false))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        {
            Some(hot_dir.clone())
        } else {
            cold_dir.clone()
        };

        if let Some(ref dir) = effective_precompiled {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }
        let precompiled_dir = effective_precompiled;

        // Probe for hipcc once at init, not per-kernel. Capture its version line
        // as a toolchain fingerprint for the cache hash (Fix #1).
        //
        // Resolve hipcc through the same selected ROCm root as the runtime and
        // headers. Probing a bare PATH hipcc first could pair a configured
        // ROCM_PATH from one version with another version's compiler.
        let resolved_hipcc = hipfire_config::rocm::tool("hipcc");
        let hipcc_bin = resolved_hipcc
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("hipcc"));
        // hipcc resolves its LLVM as $ROCM_PATH/lib/llvm/bin/clang++ and
        // ROCM_PATH defaults to /opt/rocm. On a root elsewhere the probe below
        // still succeeds while every real compile fails, so hand the child the
        // selected compiler's installation root unless the operator already
        // set one that matches.
        let rocm_env_root = hipfire_config::rocm::compiler_env_root(&hipcc_bin);

        // HIPFIRE_NO_DEVICE_COMPILER=1 makes the engine behave as if no device
        // compiler exists, so pre-compiled blobs are used verbatim instead of
        // being recompiled locally.
        //
        // This has to be an explicit switch. It used to be achievable by
        // removing the compiler from PATH, because the probe was a bare
        // `Command::new("hipcc")`. Resolved-root discovery (added so a ROCm
        // rooted outside /opt/rocm still works) finds the compiler regardless
        // of PATH, which silently defeated PATH-shadowing and made
        // "pin these exact code objects" quietly recompile instead. Verified:
        // a pinned cross-machine run reported 0 recompiles while actually
        // executing locally-built blobs.
        let compiler_disabled = hipfire_config::developer_var_os("HIPFIRE_NO_DEVICE_COMPILER")
            .is_some_and(|v| v != "0" && !v.is_empty());

        let hipcc_out = if compiler_disabled {
            eprintln!(
                "  HIPFIRE_NO_DEVICE_COMPILER set — treating the device compiler as absent; \
                 pre-compiled blobs will be used verbatim"
            );
            None
        } else if resolved_hipcc.is_none() {
            None
        } else {
            let mut probe = Command::new(&hipcc_bin);
            probe.arg("--version");
            if let Some(ref root) = rocm_env_root {
                probe.env("ROCM_PATH", root);
            }
            probe.output().ok()
        };
        let has_hipcc = hipcc_out
            .as_ref()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let toolchain_id = hipcc_out
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            })
            .unwrap_or_default();

        let gfx1151_cumode_modules = if arch == "gfx1151" {
            hipfire_config::developer_var("HIPFIRE_GFX1151_CUMODE_MODULES")
                .unwrap_or_default()
                .split(|character: char| {
                    character == ',' || character == ';' || character.is_whitespace()
                })
                .filter(|module| !module.is_empty())
                .map(str::to_owned)
                .collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };
        if !gfx1151_cumode_modules.is_empty() {
            let mut modules = gfx1151_cumode_modules.iter().cloned().collect::<Vec<_>>();
            modules.sort();
            eprintln!("  gfx1151 CU-mode modules: {}", modules.join(","));
        }

        let sched_profile_override = hipfire_config::developer_var("HIPFIRE_SCHED_PROFILE")
            .ok()
            .and_then(|value| SchedulerProfile::parse(&value));

        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
            precompiled_dir,
            cold_dir,
            has_hipcc,
            hipcc_bin,
            rocm_env_root,
            extra_flags,
            gfx1151_cumode_modules,
            toolchain_id,
            sched_profile_override,
        })
    }

    /// Returns a reference to all compiled kernel paths (name → .hsaco path).
    pub fn compiled_kernels(&self) -> &HashMap<String, PathBuf> {
        &self.compiled
    }

    /// Register `func_name` as an additional key pointing at an already-compiled
    /// artifact. `compiled_kernels()` is keyed by MODULE name, but the retained-PM4
    /// capture resolves a launched FUNCTION name; for arch-variant kernels (e.g.
    /// `gemv_hfq4g256_residual`, whose module is `gemv_hfq4g256_residual_rdna3` on
    /// RDNA3) the two differ, so without this alias the capture cannot find the
    /// owning `.hsaco` and the retained replay route fails closed. Additive and
    /// idempotent: never overwrites an existing key, so module-name lookups and
    /// the default-arch (module == func) case are unchanged.
    pub fn register_func_artifact(&mut self, func_name: &str, path: PathBuf) {
        self.compiled.entry(func_name.to_string()).or_insert(path);
    }

    fn module_flags_for(
        arch: &str,
        name: &str,
        gfx1151_cumode_modules: &HashSet<String>,
    ) -> Vec<String> {
        if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201")
            && matches!(
                name,
                "gdn_chunk_prep"
                    | "gdn_chunk_kkt_solve"
                    | "gdn_chunk_scan"
                    | "gdn_chunk_prep_gfx11"
                    | "gdn_chunk_kkt_solve_gfx1100"
            )
        {
            let mut flags = vec!["-ffp-contract=off".to_owned()];
            if arch == "gfx1201" && name == "gdn_chunk_scan" {
                flags.push("-mcumode".to_owned());
            }
            flags
        } else if arch == "gfx1151" && gfx1151_cumode_modules.contains(name) {
            vec!["-mcumode".to_owned()]
        } else {
            Vec::new()
        }
    }

    fn module_flags(&self, name: &str) -> Vec<String> {
        Self::module_flags_with_profile(
            &self.arch,
            name,
            &self.gfx1151_cumode_modules,
            self.scheduler_profile_for(name),
        )
    }

    fn module_flags_with_profile(
        arch: &str,
        name: &str,
        gfx1151_cumode_modules: &HashSet<String>,
        profile: SchedulerProfile,
    ) -> Vec<String> {
        let mut flags = Self::module_flags_for(arch, name, gfx1151_cumode_modules);
        flags.extend(profile.llvm_args().iter().map(|flag| flag.to_string()));
        flags
    }

    /// Select the profile registered for this module; the developer-only
    /// `HIPFIRE_SCHED_PROFILE` overrides it for whole-run experiments.
    fn scheduler_profile_for(&self, name: &str) -> SchedulerProfile {
        if let Some(profile) = self.sched_profile_override {
            return profile;
        }
        crate::kernels::scheduler_profile_for_module(&self.arch, name)
    }

    /// Single hashing sequence for all kernel cache keys. Every caller must
    /// go through this so KERNEL_CACHE_ABI or field order changes cannot
    /// silently diverge between JIT and packaging paths.
    fn hash_parts(
        source: &str,
        arch: &str,
        extra_flags: &str,
        module_flags: &[String],
        toolchain_id: &str,
        scheduler_profile: SchedulerProfile,
        arm: &str,
        builder_version: &str,
        kernel_id_variant: &str,
        assembler_identity: &str,
    ) -> String {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        arch.hash(&mut hasher);
        extra_flags.hash(&mut hasher);
        if !module_flags.is_empty() {
            "module-flags-v1".hash(&mut hasher);
            module_flags.hash(&mut hasher);
        }
        toolchain_id.hash(&mut hasher);
        scheduler_profile.as_str().hash(&mut hasher);
        arm.hash(&mut hasher);
        builder_version.hash(&mut hasher);
        kernel_id_variant.hash(&mut hasher);
        assembler_identity.hash(&mut hasher);
        KERNEL_CACHE_ABI.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Distinct identity for a builder-authored object. This computes the key
    /// only; custom `.hxaco` publication is owned by the later admission path.
    pub fn custom_isa_hash_for(
        source: &str,
        arch: &str,
        builder_version: &str,
        kernel_id_variant: &str,
        assembler_identity: &str,
    ) -> String {
        Self::hash_parts(
            source, arch, "", &[], "", SchedulerProfile::Default,
            "custom-isa", builder_version, kernel_id_variant, assembler_identity,
        )
    }

    fn cache_hash(&self, name: &str, source: &str) -> String {
        Self::hash_parts(
            source,
            &self.arch,
            &self.extra_flags,
            &self.module_flags(name),
            &self.toolchain_id,
            self.scheduler_profile_for(name),
            "hipcc",
            "",
            "",
            "",
        )
    }

    /// Hash a kernel as a **packaged** blob will be validated on a
    /// compiler-free runtime. Identical to `cache_hash` except
    /// `toolchain_id` is empty — the value `KernelCompiler::new` computes
    /// when `hipcc --version` cannot be run (`unwrap_or_default()`).
    ///
    /// Dropping `toolchain_id` for shipped blobs is sound: that field exists
    /// to stop different builds sharing one mutable `.hipfire_kernels` dir from
    /// reusing each other's blobs (the "invalid device image" trap). A baked,
    /// read-only, shipped-as-a-unit `kernels/compiled/{arch}` directory has no
    /// such sharing, and the key still binds source + arch + flags + ABI.
    pub fn packaging_hash(&self, name: &str, source: &str) -> String {
        Self::hash_parts(
            source,
            &self.arch,
            &self.extra_flags,
            &self.module_flags(name),
            "",
            self.scheduler_profile_for(name),
            "hipcc",
            "",
            "",
            "",
        )
    }

    /// Static packaging hash for the `hipfire-kernel-hash` binary: same inputs
    /// as `packaging_hash` but without needing a `KernelCompiler` instance.
    /// Resolves `gfx1151` CU-mode modules from the environment so the key
    /// matches what a runtime constructed with the same env would compute.
    pub fn packaging_hash_for(arch: &str, name: &str, source: &str, extra_flags: &str) -> String {
        let gfx1151_cumode_modules = if arch == "gfx1151" {
            hipfire_config::developer_var("HIPFIRE_GFX1151_CUMODE_MODULES")
                .unwrap_or_default()
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .filter(|m| !m.is_empty())
                .map(str::to_owned)
                .collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };
        let sched_profile_override = hipfire_config::developer_var("HIPFIRE_SCHED_PROFILE")
            .ok()
            .and_then(|value| SchedulerProfile::parse(&value));
        let scheduler_profile = sched_profile_override
            .unwrap_or_else(|| crate::kernels::scheduler_profile_for_module(arch, name));
        let module_flags = Self::module_flags_with_profile(
            arch,
            name,
            &gfx1151_cumode_modules,
            scheduler_profile,
        );
        Self::hash_parts(
            source,
            arch,
            extra_flags,
            &module_flags,
            "",
            scheduler_profile,
            "hipcc",
            "",
            "",
            "",
        )
    }

    /// Portable compile recipe. Source-specific flags are added by
    /// `recipe_for_source`; neither path requires hipcc or a GPU.
    pub fn recipe_for(arch: &str, module: &str, extra_flags: &[String]) -> CompileRecipe {
        let selected = if arch == "gfx1151" {
            hipfire_config::developer_var("HIPFIRE_GFX1151_CUMODE_MODULES")
                .unwrap_or_default()
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect()
        } else {
            HashSet::new()
        };
        let profile = hipfire_config::developer_var("HIPFIRE_SCHED_PROFILE")
            .ok()
            .and_then(|value| SchedulerProfile::parse(&value))
            .unwrap_or_else(|| crate::kernels::scheduler_profile_for_module(arch, module));
        let mut flags = vec![
            "--genco".to_owned(),
            format!("--offload-arch={arch}"),
            "-O3".to_owned(),
            "--no-offload-compress".to_owned(),
        ];
        flags.extend(extra_flags.iter().cloned());
        flags.extend(Self::module_flags_with_profile(arch, module, &selected, profile));
        CompileRecipe {
            arch: arch.to_owned(),
            flags,
            scheduler_profile: Some(profile.as_str().to_owned()),
            cache_abi: KERNEL_CACHE_ABI,
        }
    }

    pub fn recipe_for_source(
        arch: &str,
        module: &str,
        source: &str,
        extra_flags: &str,
    ) -> CompileRecipe {
        let extra = extra_flags.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        let mut recipe = Self::recipe_for(arch, module, &extra);
        recipe.flags.extend(Self::per_kernel_flags(source));
        recipe
    }

    pub fn packaging_flags_for(
        arch: &str,
        module: &str,
        source: &str,
        extra_flags: &str,
    ) -> Vec<String> {
        Self::recipe_for_source(arch, module, source, extra_flags).flags
    }

    fn pack_index(
        &self,
        module: &str,
        source: &str,
        symbols: Vec<String>,
        object: &Path,
    ) -> Result<PackIndex, String> {
        if !valid_identifier(module) || symbols.is_empty() || !symbols.iter().all(|s| valid_identifier(s)) {
            return Err(format!("invalid module/symbol for package: {module}"));
        }
        if self.toolchain_id.is_empty() {
            return Err(format!("{module}: missing hipcc toolchain identity"));
        }
        let object_sha256 = sha256_hex(&std::fs::read(object).map_err(|e| e.to_string())?);
        let recipe = Self::recipe_for_source(&self.arch, module, source, &self.extra_flags);
        Ok(PackIndex {
            version: PACK_INDEX_VERSION,
            module: module.to_owned(),
            symbols,
            arch: self.arch.clone(),
            source_sha256: sha256_hex(source.as_bytes()),
            flags: recipe.flags,
            scheduler_profile: recipe.scheduler_profile.unwrap_or_default(),
            cache_abi: KERNEL_CACHE_ABI,
            packaging_key: self.packaging_hash(module, source),
            object_sha256,
            toolchain_id: self.toolchain_id.clone(),
            toolchain_sha256: sha256_hex(self.toolchain_id.as_bytes()),
        })
    }

    fn indexed_object(
        &self,
        dir: &Path,
        module: &str,
        source: &str,
        symbol: &str,
    ) -> Result<Option<PathBuf>, String> {
        let index_path = dir.join(format!("{module}.index.json"));
        if !index_path.exists() {
            if dir.join(format!("{module}.hsaco")).exists() {
                return Err(format!("{}: index missing for installed object", index_path.display()));
            }
            return Ok(None);
        }
        let fail = |reason: &str| format!("{}: {reason}", index_path.display());
        let bytes = std::fs::read(&index_path).map_err(|e| fail(&e.to_string()))?;
        let index: PackIndex = serde_json::from_slice(&bytes).map_err(|e| fail(&e.to_string()))?;
        let recipe = Self::recipe_for_source(&self.arch, module, source, &self.extra_flags);
        if index.version != PACK_INDEX_VERSION
            || index.module != module
            || index.arch != self.arch
            || !index.symbols.iter().any(|entry| entry == symbol)
            || index.source_sha256 != sha256_hex(source.as_bytes())
            || index.flags != recipe.flags
            || index.scheduler_profile != recipe.scheduler_profile.unwrap_or_default()
            || index.cache_abi != KERNEL_CACHE_ABI
            || index.packaging_key != self.packaging_hash(module, source)
            || index.toolchain_id.is_empty()
            || index.toolchain_sha256 != sha256_hex(index.toolchain_id.as_bytes())
            || (!self.toolchain_id.is_empty() && index.toolchain_id != self.toolchain_id)
        {
            return Err(fail("index identity, symbol, source, flags, profile, ABI or toolchain mismatch"));
        }
        let object = dir.join(format!("{module}.hsaco"));
        if !pair_valid(&object, &dir.join(format!("{module}.hash")), &index.packaging_key) {
            return Err(fail("object/hash pair missing or stale"));
        }
        let bytes = std::fs::read(&object).map_err(|e| fail(&e.to_string()))?;
        if sha256_hex(&bytes) != index.object_sha256 {
            return Err(fail("object SHA-256 mismatch"));
        }
        Ok(Some(object))
    }

    fn writeback_package(&self, module: &str, source: &str, symbol: &str, object: &Path, force: bool) {
        let Some(dir) = self.writeback_dir() else { return };
        let key = self.packaging_hash(module, source);
        let index_path = dir.join(format!("{module}.index.json"));
        let cold_object = dir.join(format!("{module}.hsaco"));
        let cold_hash = dir.join(format!("{module}.hash"));
        let mut index = match self.pack_index(module, source, vec![symbol.to_owned()], object) {
            Ok(index) => index,
            Err(error) => {
                eprintln!("  WARNING: {module}: cold index unavailable ({error})");
                return;
            }
        };
        let existing = std::fs::read(&index_path).ok()
            .and_then(|bytes| serde_json::from_slice::<PackIndex>(&bytes).ok());
        let same_bytes = existing.as_ref().is_some_and(|prior| {
            prior.version == index.version
                && prior.module == index.module
                && prior.arch == index.arch
                && prior.source_sha256 == index.source_sha256
                && prior.flags == index.flags
                && prior.scheduler_profile == index.scheduler_profile
                && prior.cache_abi == index.cache_abi
                && prior.packaging_key == index.packaging_key
                && prior.object_sha256 == index.object_sha256
                && prior.toolchain_id == index.toolchain_id
                && prior.toolchain_sha256 == index.toolchain_sha256
        });
        if same_bytes && pair_valid(&cold_object, &cold_hash, &key) {
            index.symbols.extend(existing.unwrap().symbols.into_iter().filter(|s| s != symbol));
            // Even if the old index says its digest matches, check the actual
            // cold bytes; a corrupted blob must be repaired from the hot object.
            if std::fs::read(&cold_object)
                .ok()
                .is_some_and(|bytes| sha256_hex(&bytes) == index.object_sha256)
                && !force
            {
                if let Err(error) = publish_index(dir, &index) {
                    eprintln!("  WARNING: {module}: cold index writeback failed ({error})");
                }
                return;
            }
        }
        let _ = std::fs::remove_file(&index_path);
        if let Err(error) = publish_pair(dir, module, object, &key, true) {
            eprintln!("  WARNING: {module}: cold writeback failed ({error})");
            return;
        }
        if let Err(error) = publish_index(dir, &index) {
            eprintln!("  WARNING: {module}: cold index writeback failed ({error})");
        }
    }

    /// Build or retrieve one object, publishing a verified install package.
    pub fn pack_to(&mut self, module: &str, source: &str, symbols: &[String], dir: &Path) -> HipResult<PathBuf> {
        if symbols.is_empty() || !self.has_hipcc {
            return Err(self.compiler_unavailable_error(module, "packaging requires hipcc and at least one symbol"));
        }
        std::fs::create_dir_all(dir).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        let object = self.compile_for_symbol(module, source, &symbols[0])?.to_path_buf();
        let index = self.pack_index(module, source, symbols.to_vec(), &object)
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        let _ = std::fs::remove_file(dir.join(format!("{module}.index.json")));
        publish_pair(dir, module, &object, &index.packaging_key, true)
            .and_then(|_| publish_index(dir, &index))
            .map_err(|e| hip_bridge::HipError::new(0, &e))?;
        Ok(dir.join(format!("{module}.hsaco")))
    }

    /// Persistent install dir for writeback. None when no cold location was
    /// probed, or when it coincides with the hot cache (nothing extra to sync).
    fn writeback_dir(&self) -> Option<&Path> {
        self.cold_dir
            .as_deref()
            .filter(|d| !same_path(d, self.cache_dir.as_path()))
    }

    fn compiler_unavailable_error(&self, name: &str, action: &str) -> hip_bridge::HipError {
        let tried = vec![self.hipcc_bin.display().to_string()];
        let guidance =
            hipfire_config::rocm::resolution_failure("the ROCm HIP compiler (hipcc)", &tried);
        hip_bridge::HipError::new(
            0,
            &format!("{name}: {action}, but hipcc is unavailable.\n{guidance}"),
        )
    }

    fn unverified_object_error(
        &self,
        name: &str,
        expected_key: &str,
        found: &Path,
    ) -> hip_bridge::HipError {
        self.compiler_unavailable_error(
            name,
            &format!(
                "unverified kernel image for arch {}: expected key {expected_key} for {} \
                 (missing or mismatched .hash, or empty/non-file object); \
                 reinstall kernels with matching .hash sidecars or install hipcc to recompile",
                self.arch,
                found.display()
            ),
        )
    }

    /// Attach a hash-bound Radiowave inspection to an already validated
    /// Hipfire kernel-cache artifact. Retained PM4 consumes this adjacent
    /// manifest for cache-scope and argument-effect proofs; without it the
    /// replay path correctly fails closed to broad acquires and unknown
    /// resource effects even when the code object itself is already safe.
    ///
    /// Certification never recompiles or replaces the HSACO. It is best-effort
    /// because packaged installs may intentionally have no ROCm inspection
    /// tools; those installs retain the conservative replay policy.
    fn ensure_radiowave_certification(&self, name: &str, artifact: &Path) {
        if !self.has_hipcc {
            return;
        }
        let manifest = artifact.with_extension("radiowave.json");
        let already_valid = std::fs::read(artifact)
            .ok()
            .zip(std::fs::read_to_string(&manifest).ok())
            .is_some_and(|(code, encoded)| {
                CodeObjectCertification::from_json(&code, &encoded).is_ok_and(|certification| {
                    certification.manifest().scheduler_profile == self.scheduler_profile_for(name)
                })
            });
        if already_valid {
            return;
        }

        let source = artifact.with_extension("hip");
        if !source.is_file() {
            return;
        }
        let request = ExistingCodeObjectRequest::new(&source, artifact, &self.arch)
            .hipcc(&self.hipcc_bin)
            .command(vec![
                self.hipcc_bin.display().to_string(),
                "hipfire-kernel-cache".to_owned(),
                name.to_owned(),
            ])
            .manifest(&manifest)
            .scheduler_profile(self.scheduler_profile_for(name));
        if let Err(error) = radiowave::Compiler.certify_existing(&request) {
            eprintln!(
                "  WARNING: {name}: Radiowave could not certify existing cache artifact: {error}"
            );
        }
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Tries pre-compiled blob first (with hash validation), falls back to hipcc.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        self.compile_for_symbol(name, source, name)
    }

    /// Check the exact exported symbol before a packaged object reaches HIP.
    pub fn compile_for_symbol(&mut self, name: &str, source: &str, symbol: &str) -> HipResult<&Path> {
        if !valid_identifier(name) || !valid_identifier(symbol) {
            return Err(hip_bridge::HipError::new(0, "invalid kernel module or symbol"));
        }
        if let Some(dir) = &self.cold_dir {
            match self.indexed_object(dir, name, source, symbol) {
                Ok(Some(object)) => {
                    eprintln!("  {name}: indexed package verified: {}", object.display());
                    self.compiled.insert(name.to_owned(), object);
                    return Ok(&self.compiled[name]);
                }
                Ok(None) => {}
                Err(error) if !self.has_hipcc => {
                    return Err(hip_bridge::HipError::new(0, &format!(
                        "{name} {}: packaged object rejected ({error}); reinstall kernels or install hipcc",
                        self.arch
                    )));
                }
                Err(error) => eprintln!("  {name}: packaged object rejected ({error}); falling back to JIT"),
            }
        }

        // Hash source + arch + flags + toolchain + ABI for cache validation (used by
        // both pre-compiled and runtime paths). Flags and toolchain matter: identical
        // source compiled with different hipcc flags / ROCm versions yields a different
        // .hsaco, and reusing the wrong one surfaces as "device kernel image is invalid".
        let module_flags = self.module_flags(name);
        let src_hash = self.cache_hash(name, source);

        // Only a nonempty object with a matching sidecar can be selected.
        // Keep the rejected path for a useful compiler-free error, but try
        // content-keyed hot and validated legacy entries before failing.
        let mut unverified_object = None;
        let mut stale_precompiled = false;
        if let Some(dir) = &self.precompiled_dir {
            let precompiled = dir.join(format!("{name}.hsaco"));
            if precompiled.exists() {
                if nonempty_blob(&precompiled) {
                    // A content-keyed hot entry may still hit below.
                    stale_precompiled = true;
                }
                unverified_object = Some(precompiled);
            }
        }

        // Hot entries are content-keyed (`{name}.{hash}.hsaco`): the filename
        // IS the key, so a nonempty hit is by construction correct — no
        // sidecar check, and no collision between builds sharing this root.
        let stem = hot_stem(name, &src_hash);
        let obj_path = self.cache_dir.join(hot_object_name(name, &src_hash));

        if nonempty_blob(&obj_path) {
            self.writeback_package(name, source, symbol, &obj_path, false);
            self.ensure_radiowave_certification(name, &obj_path);
            self.compiled.insert(name.to_string(), obj_path);
            return Ok(&self.compiled[name]);
        }

        // Legacy `{name}.hsaco` + hash fallback: entries written before
        // content-keying, plus cold-seeded pairs. On a validated hit, promote
        // once into the content-keyed name so later lookups take the direct
        // path; if promotion fails the legacy path itself is still a valid hit.
        let legacy_obj = self.cache_dir.join(format!("{name}.hsaco"));
        let legacy_hash = self.cache_dir.join(format!("{name}.hash"));
        if pair_valid(&legacy_obj, &legacy_hash, &src_hash) {
            let hit_path = if publish_pair(&self.cache_dir, &stem, &legacy_obj, &src_hash, true)
                .is_ok()
            {
                obj_path
            } else {
                legacy_obj
            };
            self.writeback_package(name, source, symbol, &hit_path, false);
            self.ensure_radiowave_certification(name, &hit_path);
            self.compiled.insert(name.to_string(), hit_path);
            return Ok(&self.compiled[name]);
        }

        if !self.has_hipcc {
            if unverified_object.is_none() && legacy_obj.exists() {
                unverified_object = Some(legacy_obj);
            }
            return Err(match unverified_object {
                Some(found) => self.unverified_object_error(name, &src_hash, &found),
                None => self.compiler_unavailable_error(name, "no usable cached kernel image exists"),
            });
        }

        if stale_precompiled {
            eprintln!("  {name}: pre-compiled blob hash is stale and no cached build matches; recompiling");
        }
        Self::hipcc_compile_publish(
            &self.hipcc_bin,
            self.rocm_env_root.as_deref(),
            &self.arch,
            &self.cache_dir,
            name,
            source,
            &src_hash,
            &self.extra_flags,
            &module_flags,
        )?;
        self.ensure_radiowave_certification(name, &obj_path);

        // Hot keys keep the toolchain ID; cold pairs use the packaging key
        // with toolchain identity bound in the checked index.
        self.writeback_package(name, source, symbol, &obj_path, false);

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }

    /// Force a fresh hipcc recompile after the driver rejects a loaded image.
    ///
    /// Does **not** pre-delete hot/cold artifacts (safe when hot==cold or the
    /// dirs are aliased). Bypasses all lookup and invokes transactional hipcc
    /// publication directly. Cold is force-synced only after success. A failed
    /// hipcc leaves prior durable pairs intact.
    pub(crate) fn recompile(&mut self, name: &str, source: &str, symbol: &str) -> HipResult<PathBuf> {
        if !self.has_hipcc {
            return Err(self.compiler_unavailable_error(
                name,
                "the cached kernel image is invalid and must be recompiled",
            ));
        }

        self.compiled.remove(name);

        let module_flags = self.module_flags(name);
        let src_hash = self.cache_hash(name, source);
        let obj_path = self.cache_dir.join(hot_object_name(name, &src_hash));

        Self::hipcc_compile_publish(
            &self.hipcc_bin,
            self.rocm_env_root.as_deref(),
            &self.arch,
            &self.cache_dir,
            name,
            source,
            &src_hash,
            &self.extra_flags,
            &module_flags,
        )?;
        self.ensure_radiowave_certification(name, &obj_path);

        // Force-sync only after the new object is complete.
        self.writeback_package(name, source, symbol, &obj_path, true);

        self.compiled.insert(name.to_string(), obj_path.clone());
        Ok(obj_path)
    }

    /// Extract per-kernel hipcc flags from magic comments in the source.
    /// The marker must be the dominant content of a comment line — i.e. a
    /// line whose non-whitespace starts with `//` followed (possibly after
    /// more whitespace) by `HIPFIRE_COMPILER_FLAGS:`. Flags after the colon
    /// are split on whitespace and appended to the hipcc invocation.
    /// Lines that merely *mention* the tag in prose (e.g. in a docstring
    /// explaining how to use it) are ignored, so we don't accidentally turn
    /// documentation into command-line arguments.
    fn per_kernel_flags(source: &str) -> Vec<String> {
        const TAG: &str = "HIPFIRE_COMPILER_FLAGS:";
        let mut out = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            let after_slashes = match trimmed.strip_prefix("//") {
                Some(rest) => rest.trim_start(),
                None => continue,
            };
            if let Some(rest) = after_slashes.strip_prefix(TAG) {
                for tok in rest.split_whitespace() {
                    out.push(tok.to_string());
                }
            }
        }
        out
    }

    /// On Windows, convert a path containing spaces to its 8.3 short-path
    /// form (e.g. `C:\Program Files\AMD\ROCm\6.4\include` to
    /// `C:\PROGRA~1\AMD\ROCm\6.4\include`) so it can be embedded as a single
    /// argv element to hipcc.bat without being split by the inner clang.exe
    /// re-tokenisation. Falls back to the original path on any error or on
    /// non-Windows hosts. Reported as #82.
    #[cfg(target_os = "windows")]
    fn win_short_path_if_needed(p: &str) -> String {
        if !p.contains(' ') {
            return p.to_string();
        }
        // Use cmd.exe's `for %A in (LONG) do echo %~sA` to ask the OS for the
        // 8.3 alias. Subprocess approach avoids pulling in a winapi crate dep
        // for this single call site.
        let out = Command::new("cmd")
            .raw_arg("/c")
            .raw_arg(&format!("for %A in (\"{}\") do @echo %~sA", p))
            .output();
        match out {
            Ok(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !s.is_empty() && !s.contains(' ') {
                    s
                } else {
                    p.to_string()
                }
            }
            _ => p.to_string(),
        }
    }

    /// No-op on non-Windows: POSIX argv handling preserves embedded spaces
    /// and ROCm's standard `/opt/rocm/include` has no spaces anyway.
    #[cfg(not(target_os = "windows"))]
    fn win_short_path_if_needed(p: &str) -> String {
        p.to_string()
    }

    /// Root-scoped flags passed through hipcc to the device compiler.
    ///
    /// `hipcc.bat` re-tokenises its arguments on Windows, so both flags must
    /// use the same space-free path policy as the explicit include directory.
    /// Keep the resolver injectable so the Windows-shaped argv contract can be
    /// tested on hosts without a Windows SDK.
    fn rocm_root_flags_with(
        root: &Path,
        resolve_for_hipcc: impl FnOnce(&str) -> String,
    ) -> Vec<String> {
        let root = root.to_string_lossy();
        let resolved = resolve_for_hipcc(&root);
        vec![
            format!("--rocm-path={resolved}"),
            format!("--hip-path={resolved}"),
        ]
    }

    fn rocm_root_flags(root: &Path) -> Vec<String> {
        Self::rocm_root_flags_with(root, Self::win_short_path_if_needed)
    }

    /// Core hipcc argv: genco/arch/O3, then passthrough, then -o out src.
    fn direct_hipcc_args(
        arch: &str,
        src_path: &Path,
        obj_path: &Path,
        passthrough: Vec<String>,
    ) -> Vec<String> {
        // clang ≥19 compresses offload bundles by default. HIP's loader accepts the
        // compressed container; the public HSA reader Redline uses does not, so a
        // CCOB blob demotes every retained route to plain HIP.
        let mut args: Vec<String> = vec![
            "--genco".into(),
            format!("--offload-arch={arch}"),
            "-O3".into(),
            "--no-offload-compress".into(),
        ];
        args.extend(passthrough);
        args.push("-o".into());
        args.push(obj_path.to_str().unwrap().into());
        args.push(src_path.to_str().unwrap().into());
        args
    }

    /// Build hipcc passthrough flags (include paths, extras, module, per-kernel).
    fn hipcc_passthrough(
        name: &str,
        source: &str,
        extra_flags: &str,
        module_flags: &[String],
    ) -> Vec<String> {
        let per_kernel = Self::per_kernel_flags(source);
        let mut passthrough: Vec<String> = Vec::new();
        // Some hipcc installs (notably V620's CachyOS build of ROCm 7.2) do not
        // auto-inject the HIP include path, so `#include <hip/hip_runtime.h>`
        // fails with "file not found". Add well-known candidates as -I flags;
        // existence-checked so wrong paths on other distros don't leak in.
        //
        // The include path comes from the same selected root as hipcc and the
        // runtime. Do not separately prepend ambient HIP_PATH or /opt/rocm:
        // either could belong to another side-by-side ROCm version.
        //
        // Tell the device compiler where ROCm is rather than relying on its
        // self-relative probe. That probe needs marker files (`bin/.hipVersion`
        // and friends) which some installs omit; without them clang reports
        // "cannot find ROCm device library" and, once headers are supplied by
        // -I alone, still resolves device math like `rsqrtf` against the host
        // `/usr/include/math.h` instead of the HIP runtime wrapper. Only a
        // complete root is passed, so a shim directory can never be handed to
        // clang as authoritative; on a conventional install this resolves to
        // the same path clang would have found by itself.
        let selected_root = hipfire_config::rocm::root();
        if let Some(root) = selected_root.as_ref() {
            if hipfire_config::rocm::is_complete_root(root) {
                passthrough.extend(Self::rocm_root_flags(root));
            }
        }
        if let Some(candidate) =
            selected_root.map(|root| root.join("include").to_string_lossy().into_owned())
        {
            if Path::new(&candidate).join("hip/hip_runtime.h").exists() {
                // Windows hipcc (hipcc.bat) re-tokenises its argv on the inner
                // clang.exe command line WITHOUT preserving quoting around
                // embedded spaces, so an include path inside `Program Files`
                // gets split at the space and clang sees the half before the
                // split. Convert to the 8.3 short-path form (e.g.
                // C:\PROGRA~1\AMD\ROCm\6.4\include) which contains no spaces.
                // Reported in #82.
                let resolved = Self::win_short_path_if_needed(&candidate);
                passthrough.push(format!("-I{resolved}"));
            }
        }
        for flag in extra_flags.split_whitespace() {
            passthrough.push(flag.to_string());
        }
        for flag in module_flags {
            passthrough.push(flag.clone());
        }
        for flag in &per_kernel {
            passthrough.push(flag.clone());
        }
        if !module_flags.is_empty() || !per_kernel.is_empty() {
            let flags = module_flags
                .iter()
                .chain(per_kernel.iter())
                .cloned()
                .collect::<Vec<_>>();
            eprintln!("  {name}: per-kernel flags: {}", flags.join(" "));
        }
        passthrough
    }

    /// Low-level hipcc invocation writing the object to `obj_path` (may be a temp).
    /// Does not mutate hash sidecars. Shared by the transactional publisher.
    fn hipcc_compile_to(
        hipcc_bin: &Path,
        rocm_env_root: Option<&Path>,
        arch: &str,
        src_path: &Path,
        obj_path: &Path,
        name: &str,
        source: &str,
        extra_flags: &str,
        module_flags: &[String],
    ) -> HipResult<()> {
        std::fs::write(src_path, source).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
        })?;

        let passthrough = Self::hipcc_passthrough(name, source, extra_flags, module_flags);
        let args = Self::direct_hipcc_args(arch, src_path, obj_path, passthrough);

        let mut cmd = Command::new(hipcc_bin);
        cmd.args(&args);
        if let Some(root) = rocm_env_root {
            cmd.env("ROCM_PATH", root);
        }
        let output = cmd.output().map_err(|e| {
            let tried = vec![hipcc_bin.display().to_string()];
            let guidance =
                hipfire_config::rocm::resolution_failure("the ROCm HIP compiler (hipcc)", &tried);
            hip_bridge::HipError::new(
                0,
                &format!("failed to run {}: {e}\n{guidance}", hipcc_bin.display()),
            )
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // A missing hip_runtime.h is a missing PACKAGE, not a missing flag.
            // clang reports it as a bare "file not found" that names neither the
            // ROCm root it searched nor the component that would supply it, so
            // spell both out rather than leaving the user to guess.
            let hint = if stderr.contains("hip/hip_runtime.h") {
                let root = hipfire_config::rocm::root();
                let guidance = hipfire_config::rocm::install_guidance()
                    .into_iter()
                    .map(|line| format!("\n  {line}"))
                    .collect::<String>();
                format!(
                    "\n\nThe HIP headers are not installed under the ROCm root ({}).\n\
                     A working hipcc does not imply them — they ship in a separate\n\
                     package.{guidance}\n\
                     Run `hipfire diag` for the per-root component inventory.",
                    root.map(|r| r.display().to_string())
                        .unwrap_or_else(|| "none resolved".into()),
                )
            } else {
                String::new()
            };
            return Err(hip_bridge::HipError::new(
                0,
                &format!("hipcc compilation failed for {name}:\n{stderr}{hint}"),
            ));
        }
        Ok(())
    }

    /// Compile via hipcc into a unique same-dir temp object, then transactionally
    /// publish the final hot blob+hash. Never writes final hashes directly.
    /// On any failure, temps are cleaned and prior hot pairs remain intact.
    /// Hash publication errors are **not** ignored.
    fn hipcc_compile_publish(
        hipcc_bin: &Path,
        rocm_env_root: Option<&Path>,
        arch: &str,
        cache_dir: &Path,
        name: &str,
        source: &str,
        src_hash: &str,
        extra_flags: &str,
        module_flags: &[String],
    ) -> HipResult<()> {
        // Content-keyed outputs: the source file, final blob, and temps all
        // carry the hash, so concurrent builds compiling different sources
        // under one module name never share a path. `name` stays the human
        // label in messages. The hash sidecar is still published (same stem)
        // for tooling that validates pairs; lookup needs only the blob.
        let stem = hot_stem(name, src_hash);
        let src_path = cache_dir.join(format!("{stem}.hip"));
        let final_hsaco = cache_dir.join(format!("{stem}.hsaco"));
        let token = unique_token();
        let tmp_obj = cache_dir.join(format!(".{stem}.{token}.hsaco.tmp"));

        let cleanup_tmp = || {
            let _ = std::fs::remove_file(&tmp_obj);
        };

        let compile_result = Self::hipcc_compile_to(
            hipcc_bin,
            rocm_env_root,
            arch,
            &src_path,
            &tmp_obj,
            name,
            source,
            extra_flags,
            module_flags,
        );

        if let Err(e) = compile_result {
            cleanup_tmp();
            return Err(e);
        }

        if !nonempty_blob(&tmp_obj) {
            cleanup_tmp();
            return Err(hip_bridge::HipError::new(
                0,
                &format!("hipcc produced empty object for {name}"),
            ));
        }

        // Publish blob then hash via the shared transactional helper.
        // On failure, leave prior durable pair intact (temps cleaned inside).
        if let Err(e) = publish_pair(cache_dir, &stem, &tmp_obj, src_hash, true) {
            cleanup_tmp();
            // If publish moved the temp already, final may exist without hash —
            // that is the safe invalid state (blob without certifying hash).
            let _ = std::fs::remove_file(&tmp_obj);
            return Err(hip_bridge::HipError::new(
                0,
                &format!("failed to publish compiled kernel {name}: {e}"),
            ));
        }

        // publish_pair copies then leaves the source temp behind when src != dest.
        cleanup_tmp();

        // Defensive: final pair must be valid after publication.
        let final_hash = cache_dir.join(format!("{stem}.hash"));
        if !pair_valid(&final_hsaco, &final_hash, src_hash) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("post-publish pair invalid for {name}"),
            ));
        }

        Ok(())
    }

    /// Compile multiple kernels in parallel. Returns paths to .hsaco files.
    /// Kernels already compiled or cached are skipped.
    pub fn compile_batch(&mut self, kernels: &[(&str, &str)]) -> HipResult<()> {
        let with_symbols = kernels.iter().map(|&(name, source)| (name, source, name)).collect::<Vec<_>>();
        self.compile_batch_for_symbols(&with_symbols)
    }

    /// Preserve parallel JIT builds while validating the symbol for packaged hits.
    pub fn compile_batch_for_symbols(&mut self, kernels: &[(&str, &str, &str)]) -> HipResult<()> {
        let mut to_compile: Vec<(String, String, String, String, Vec<String>)> = Vec::new();

        for &(name, source, symbol) in kernels {
            if !valid_identifier(name) || !valid_identifier(symbol) {
                return Err(hip_bridge::HipError::new(0, "invalid kernel module or symbol"));
            }
            if let Some(dir) = &self.cold_dir {
                match self.indexed_object(dir, name, source, symbol) {
                    Ok(Some(object)) => {
                        eprintln!("  {name}: indexed package verified: {}", object.display());
                        self.compiled.insert(name.to_owned(), object);
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) if !self.has_hipcc => {
                        return Err(hip_bridge::HipError::new(0, &format!(
                            "{name} {}: packaged object rejected ({error}); reinstall kernels or install hipcc",
                            self.arch
                        )));
                    }
                    Err(error) => eprintln!("  {name}: packaged object rejected ({error}); falling back to JIT"),
                }
            }

            let module_flags = self.module_flags(name);
            let src_hash = self.cache_hash(name, source);
            let mut unverified_object = None;

            // Check precompiled with valid pair (nonempty blob + matching hash).
            if let Some(dir) = &self.precompiled_dir {
                let precompiled = dir.join(format!("{name}.hsaco"));
                if precompiled.exists() {
                    unverified_object = Some(precompiled);
                }
            }

            // Hot lookup mirrors compile(): content-keyed hit first (correct
            // by construction), then legacy pair with one-time promotion.
            let stem = hot_stem(name, &src_hash);
            let obj_path = self.cache_dir.join(hot_object_name(name, &src_hash));

            if nonempty_blob(&obj_path) {
                self.writeback_package(name, source, symbol, &obj_path, false);
                self.compiled.insert(name.to_string(), obj_path);
                continue;
            }

            let legacy_obj = self.cache_dir.join(format!("{name}.hsaco"));
            let legacy_hash = self.cache_dir.join(format!("{name}.hash"));
            if pair_valid(&legacy_obj, &legacy_hash, &src_hash) {
                let hit_path =
                    if publish_pair(&self.cache_dir, &stem, &legacy_obj, &src_hash, true).is_ok() {
                        obj_path
                    } else {
                        legacy_obj
                    };
                self.writeback_package(name, source, symbol, &hit_path, false);
                self.compiled.insert(name.to_string(), hit_path);
                continue;
            }

            if !self.has_hipcc {
                if unverified_object.is_none() && legacy_obj.exists() {
                    unverified_object = Some(legacy_obj);
                }
                return Err(match unverified_object {
                    Some(found) => self.unverified_object_error(name, &src_hash, &found),
                    None => self.compiler_unavailable_error(name, "no usable cached kernel image exists"),
                });
            }

            to_compile.push((name.to_string(), source.to_string(), symbol.to_string(), src_hash, module_flags));
        }

        if to_compile.is_empty() {
            return Ok(());
        }

        let n = to_compile.len();
        eprintln!("  compiling {n} kernels in parallel...");
        let arch = self.arch.clone();
        let cache_dir = self.cache_dir.clone();
        // Writeback happens on the joining thread after each successful build.
        let hipcc_bin = self.hipcc_bin.clone();
        let rocm_env_root = self.rocm_env_root.clone();

        // Shared counter so parallel threads can report "[i/N] name" as each one
        // completes. Ordering follows completion (not launch) — matches the pace
        // of hipcc finishing.
        let done = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn hipcc in parallel threads — each uses transactional publication.
        let results: Vec<_> = to_compile
            .into_iter()
            .map(|(name, source, symbol, src_hash, module_flags)| {
                let arch = arch.clone();
                let cache_dir = cache_dir.clone();
                let hipcc_bin = hipcc_bin.clone();
                let rocm_env_root = rocm_env_root.clone();
                let extra_flags = self.extra_flags.clone();
                let done = std::sync::Arc::clone(&done);
                let handle = thread::spawn(move || {
                    let result = Self::hipcc_compile_publish(
                        &hipcc_bin,
                        rocm_env_root.as_deref(),
                        &arch,
                        &cache_dir,
                        &name,
                        &source,
                        &src_hash,
                        &extra_flags,
                        &module_flags,
                    );

                    let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let marker = if result.is_ok() { "✓" } else { "✗" };
                    eprintln!("  [{i:>3}/{n}] {marker} {name}");
                    let obj_path = cache_dir.join(hot_object_name(&name, &src_hash));
                    (name, source, symbol, obj_path, result)
                });
                handle
            })
            .collect();

        let mut errors = Vec::new();
        for handle in results {
            let (name, source, symbol, obj_path, result) = handle.join().unwrap();
            match result {
                Ok(()) => {
                    self.writeback_package(&name, &source, &symbol, &obj_path, false);
                    self.compiled.insert(name, obj_path);
                }
                Err(e) => errors.push(e),
            }
        }
        eprintln!("  done ({n} kernels).");

        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_compiler(extra_flags: &str, toolchain_id: &str) -> KernelCompiler {
        KernelCompiler {
            cache_dir: PathBuf::from(".test-cache"),
            arch: "gfx1151".to_string(),
            compiled: HashMap::new(),
            precompiled_dir: None,
            cold_dir: None,
            has_hipcc: false,
            extra_flags: extra_flags.to_string(),
            gfx1151_cumode_modules: HashSet::new(),
            toolchain_id: toolchain_id.to_string(),
            hipcc_bin: PathBuf::from("hipcc"),
            rocm_env_root: None,
            sched_profile_override: None,
        }
    }

    #[test]
    fn custom_arm_cache_key_cannot_alias_hipcc() {
        let baseline = KernelCompiler::hash_parts(
            "source", "gfx1201", "", &[], "llvm", SchedulerProfile::Default,
            "hipcc", "", "", "",
        );
        let custom = KernelCompiler::custom_isa_hash_for(
            "source", "gfx1201", "0.1.0", "iu4_k1:cacc2=0", "llvm-mc 23",
        );
        assert_ne!(baseline, custom);
        let other_variant = KernelCompiler::custom_isa_hash_for(
            "source", "gfx1201", "0.1.0", "iu4_k1:cacc2=1", "llvm-mc 23",
        );
        assert_ne!(custom, other_variant);
    }

    #[test]
    fn two_dir_writeback_stays_distinct() {
        // Lookup may prefer hot while cold remains the install writeback target.
        // Defends the directory-selection invariant without invoking hipcc.
        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = PathBuf::from("/tmp/hipfire_hot/gfx1201");
        c.precompiled_dir = Some(PathBuf::from("/tmp/hipfire_hot/gfx1201"));
        c.cold_dir = Some(PathBuf::from(
            "/home/user/.hipfire/bin/kernels/compiled/gfx1201",
        ));

        assert_eq!(
            c.writeback_dir(),
            Some(Path::new(
                "/home/user/.hipfire/bin/kernels/compiled/gfx1201"
            )),
            "writeback must target cold, not the hot lookup view"
        );

        // When cold coincides with hot there is nothing extra to sync.
        c.cold_dir = Some(c.cache_dir.clone());
        assert_eq!(c.writeback_dir(), None);
    }

    #[test]
    fn same_path_equates_lexical_identity() {
        let p = Path::new("/tmp/hipfire_hot/gfx1201");
        assert!(same_path(p, p));
        assert!(!same_path(
            Path::new("/tmp/hipfire_hot/gfx1201"),
            Path::new("/home/user/.hipfire/bin/kernels/compiled/gfx1201"),
        ));
    }

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn pair_valid_requires_nonempty_blob() {
        let root = temp_root("pair_valid");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let hsaco = root.join("k.hsaco");
        let hash = root.join("k.hash");
        std::fs::write(&hash, "deadbeef").unwrap();

        // Missing blob.
        assert!(!pair_valid(&hsaco, &hash, "deadbeef"));

        // Empty blob.
        std::fs::write(&hsaco, b"").unwrap();
        assert!(!nonempty_blob(&hsaco));
        assert!(!pair_valid(&hsaco, &hash, "deadbeef"));

        // Nonempty blob + matching hash.
        std::fs::write(&hsaco, b"BLOB").unwrap();
        assert!(pair_valid(&hsaco, &hash, "deadbeef"));

        // Hash mismatch.
        assert!(!pair_valid(&hsaco, &hash, "other"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn validated_hot_lookup_writeback_refreshes_stale_cold() {
        // With a compiler available, a hot-keyed build repairs a stale,
        // unindexed cold install pair without invoking hipcc again.
        let root = temp_root("hot_lookup_wb");
        let hot = root.join("hot");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::create_dir_all(&cold).unwrap();

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = Some(hot.clone());
        c.cold_dir = Some(cold.clone());
        c.has_hipcc = true;

        let name = "add_inplace";
        let source = "__global__ void add_inplace() {}";
        let src_hash = c.cache_hash(name, source);

        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_BLOB_V1").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();
        std::fs::write(cold.join(format!("{name}.hsaco")), b"OLD_COLD_BLOB").unwrap();
        std::fs::write(cold.join(format!("{name}.hash")), "stale").unwrap();

        let path = c
            .compile(name, source)
            .expect("validated hot lookup must refresh stale installed object");
        assert_eq!(std::fs::read(path).unwrap(), b"HOT_BLOB_V1");

        let cold_hash = std::fs::read_to_string(cold.join(format!("{name}.hash"))).unwrap();
        assert_eq!(
            cold_hash.trim(),
            c.packaging_hash(name, source),
            "stale cold hash must be refreshed from validated hot lookup"
        );
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_BLOB_V1",
            "cold blob must be copied from the validated hot pair"
        );
        // Hot pair must remain the source of truth (no self-clobber).
        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_BLOB_V1"
        );
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn validated_hot_lookup_restores_missing_cold_blob() {
        // End-to-end RED: hot hit + cold hash present + cold hsaco missing
        // must restore the cold blob (install daemon --precompile path).
        let root = temp_root("hot_lookup_missing_cold");
        let hot = root.join("hot");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::create_dir_all(&cold).unwrap();

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = Some(hot.clone());
        c.cold_dir = Some(cold.clone());
        c.has_hipcc = false;

        let name = "add_inplace";
        let source = "__global__ void add_inplace() {}";
        let src_hash = c.cache_hash(name, source);

        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_BLOB_V1").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();
        // Cold hash matches, but blob is gone (the verified RED fixture).
        std::fs::write(cold.join(format!("{name}.hash")), &src_hash).unwrap();

        let path = c
            .compile(name, source)
            .expect("validated hot lookup must succeed without hipcc");
        assert_eq!(std::fs::read(path).unwrap(), b"HOT_BLOB_V1");

        assert!(
            cold.join(format!("{name}.hsaco")).exists(),
            "missing cold hsaco must be restored from validated hot"
        );
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_BLOB_V1"
        );
        assert_eq!(
            std::fs::read_to_string(cold.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            c.packaging_hash(name, source)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_hash_includes_flags_and_toolchain() {
        let source = "__global__ void kernel() {}";
        let base = test_compiler("", "hipcc 7.2").cache_hash("kernel", source);
        let flags_changed = test_compiler("-mllvm -amdgpu-enable-flat-scratch=false", "hipcc 7.2")
            .cache_hash("kernel", source);
        let toolchain_changed = test_compiler("", "hipcc 7.3").cache_hash("kernel", source);

        assert_ne!(
            base, flags_changed,
            "cache key must change when hipcc flags change"
        );
        assert_ne!(
            base, toolchain_changed,
            "cache key must change when hipcc toolchain changes"
        );
    }

    #[test]
    fn cache_key_covers_source_flags_and_arch() {
        // The content key must be a pure function of (source, flags, arch):
        // same triple → same key; changing any one of the three → new key.
        // A hit is then by construction the right blob for this build.
        let source = "__global__ void kernel() {}";
        let base = test_compiler("", "hipcc 7.2").cache_hash("kernel", source);
        assert_eq!(
            base,
            test_compiler("", "hipcc 7.2").cache_hash("kernel", source),
            "same source + flags + arch must produce the same key"
        );
        assert_ne!(
            base,
            test_compiler("", "hipcc 7.2").cache_hash("kernel", "__global__ void kernel2() {}"),
            "cache key must change when the source changes"
        );
        assert_ne!(
            base,
            test_compiler("-O0", "hipcc 7.2").cache_hash("kernel", source),
            "cache key must change when compile flags change"
        );
        let mut other_arch = test_compiler("", "hipcc 7.2");
        other_arch.arch = "gfx1100".to_string();
        assert_ne!(
            base,
            other_arch.cache_hash("kernel", source),
            "cache key must change when the target arch changes"
        );
    }

    #[test]
    fn hot_names_isolate_distinct_builds() {
        // Content-keyed filenames: same (name, hash) → same path; any change
        // → a different path, so daemons from different builds sharing one
        // root never collide on an entry (the `named symbol not found` race).
        assert_eq!(
            hot_object_name("gemv_hfq4g256", "abc123"),
            hot_object_name("gemv_hfq4g256", "abc123")
        );
        assert_ne!(
            hot_object_name("gemv_hfq4g256", "abc123"),
            hot_object_name("gemv_hfq4g256", "def456"),
            "different source/flags/arch must map to a different entry path"
        );
        assert_ne!(
            hot_object_name("gemv_hfq4g256", "abc123"),
            hot_object_name("gemm_hfq4g128", "abc123"),
            "different kernels must map to different entry paths"
        );
        let name = hot_object_name("gemv_hfq4g256", "abc123");
        assert!(
            name.contains("gemv_hfq4g256") && name.contains("abc123"),
            "entry layout must stay self-describing: {name}"
        );
    }

    #[test]
    fn atomic_publish_never_shows_partial() {
        // A concurrent reader must only ever observe a complete entry.
        // Publish cycles several generations (distinct sizes + contents)
        // while a reader thread samples the destination path; any truncation
        // or interleave — the half-written object behind
        // `named symbol not found` — fails the exact-equality check.
        // No GPU needed: this exercises the same stage-temp + rename path
        // every hot entry write uses.
        let root = temp_root("atomic_publish");
        let _ = std::fs::remove_dir_all(&root);
        let dest = root.join("dest");
        let src_dir = root.join("src");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::create_dir_all(&src_dir).unwrap();

        let name = "k";
        let gens: Vec<Vec<u8>> = (0..8u8).map(|g| vec![g; 1024 * (g as usize + 1)]).collect();
        let seed = src_dir.join("seed.hsaco");
        std::fs::write(&seed, &gens[0]).unwrap();
        publish_pair(&dest, name, &seed, "gen0", true).unwrap();

        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader_done = std::sync::Arc::clone(&done);
        let dest_path = dest.join(format!("{name}.hsaco"));

        // The reader owns its observations locally and returns them through
        // the scoped join — no shared lock between the threads.
        let (seen, missings) = std::thread::scope(|scope| {
            let reader = scope.spawn(move || {
                let mut seen: Vec<Vec<u8>> = Vec::new();
                let mut missings = 0;
                let mut samples = 0;
                while samples < 500 {
                    match std::fs::read(&dest_path) {
                        Ok(bytes) => {
                            if !seen.contains(&bytes) {
                                seen.push(bytes);
                            }
                        }
                        Err(_) => missings += 1,
                    }
                    samples += 1;
                    if reader_done.load(std::sync::atomic::Ordering::SeqCst) && samples >= 100 {
                        break;
                    }
                }
                (seen, missings)
            });
            for (g, content) in gens.iter().enumerate().skip(1) {
                let src = src_dir.join(format!("gen{g}.hsaco"));
                std::fs::write(&src, content).unwrap();
                publish_pair(&dest, name, &src, &format!("gen{g}"), true).unwrap();
            }
            done.store(true, std::sync::atomic::Ordering::SeqCst);
            reader.join().unwrap()
        });

        assert_eq!(
            missings, 0,
            "atomic rename must never expose a missing entry once seeded"
        );
        assert!(
            seen.iter().all(|b| gens.contains(b)),
            "reader saw a partial/mixed entry: sizes {:?}",
            seen.iter().map(|b| b.len()).collect::<Vec<_>>()
        );
        assert!(
            seen.iter().any(|b| *b != gens[0]),
            "reader never observed a new generation — test sampled nothing"
        );
        assert!(
            leftover_temps(&dest).is_empty(),
            "publish series must leave no temps: {:?}",
            leftover_temps(&dest)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn default_cache_root_honors_env_and_shared_default() {
        // Race-free (no env mutation): assert the ambient resolution.
        match std::env::var_os("HIPFIRE_KERNEL_CACHE") {
            Some(v) => assert_eq!(
                default_cache_root(),
                PathBuf::from(v),
                "explicit HIPFIRE_KERNEL_CACHE must win"
            ),
            None => {
                let root = default_cache_root();
                assert_eq!(
                    root.file_name().and_then(|n| n.to_str()),
                    Some(".hipfire_kernels"),
                    "default must be a single shared .hipfire_kernels root, not the CWD"
                );
                if let Some(home) = std::env::var_os("HOME") {
                    assert_eq!(
                        root,
                        PathBuf::from(home).join(".hipfire_kernels"),
                        "default must live under HOME so all worktrees share it"
                    );
                }
            }
        }
    }

    #[test]
    fn direct_hipcc_args_keep_rocm_as_the_default_compiler() {
        let args = KernelCompiler::direct_hipcc_args(
            "gfx1100",
            Path::new("kernel.hip"),
            Path::new("kernel.hsaco"),
            vec!["-I/opt/rocm/include".to_owned()],
        );

        assert_eq!(
            args,
            vec![
                "--genco",
                "--offload-arch=gfx1100",
                "-O3",
                "--no-offload-compress",
                "-I/opt/rocm/include",
                "-o",
                "kernel.hsaco",
                "kernel.hip",
            ]
        );
        assert!(
            args.iter().all(|arg| !arg.contains("RADIOWAVE")),
            "the product compiler must not inject Radiowave"
        );
    }

    #[test]
    fn windows_shaped_rocm_root_flags_share_the_short_path_policy() {
        let long = Path::new(r"C:\Program Files\AMD\ROCm\7.2");
        let flags = KernelCompiler::rocm_root_flags_with(long, |path| {
            assert_eq!(path, r"C:\Program Files\AMD\ROCm\7.2");
            r"C:\PROGRA~1\AMD\ROCm\7.2".to_owned()
        });

        assert_eq!(
            flags,
            vec![
                r"--rocm-path=C:\PROGRA~1\AMD\ROCm\7.2",
                r"--hip-path=C:\PROGRA~1\AMD\ROCm\7.2",
            ]
        );
        assert!(
            flags.iter().all(|arg| !arg.contains(' ')),
            "hipcc.bat must receive each root-scoped flag without spaces"
        );
    }

    #[test]
    fn gfx1151_cumode_is_module_exact_and_cache_keyed() {
        let source = "__global__ void kernel() {}";
        let control = test_compiler("", "hipcc 7.2");
        let mut candidate = test_compiler("", "hipcc 7.2");
        candidate
            .gfx1151_cumode_modules
            .insert("selected".to_owned());

        assert_eq!(candidate.module_flags("selected"), vec!["-mcumode"]);
        assert!(candidate.module_flags("other").is_empty());
        assert_ne!(
            control.cache_hash("selected", source),
            candidate.cache_hash("selected", source)
        );
        assert_eq!(
            control.cache_hash("other", source),
            candidate.cache_hash("other", source)
        );

        candidate.arch = "gfx1100".to_owned();
        assert!(candidate.module_flags("selected").is_empty());
    }

    #[test]
    fn gfx1100_muse_rm_bt_uses_iterative_ilp_and_is_cache_keyed() {
        let source = "__global__ void kernel() {}";
        let module = "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt";
        let mut gfx1100 = test_compiler("", "hipcc 7.2");
        gfx1100.arch = "gfx1100".to_owned();
        let control = test_compiler("", "hipcc 7.2");

        assert_eq!(
            gfx1100.module_flags(module),
            vec!["-mllvm".to_owned(), "-misched=gcn-iterative-ilp".to_owned(),]
        );
        assert!(gfx1100.module_flags("other").is_empty());
        assert!(control.module_flags(module).is_empty());
        assert_ne!(
            control.cache_hash(module, source),
            gfx1100.cache_hash(module, source)
        );
    }

    #[test]
    fn gfx1201_k1_scheduler_is_module_exact_and_cache_keyed() {
        let source = "__global__ void kernel() {}";
        let module = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_v3";
        let shipping = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_symfold_g12r";
        let mut compiler = test_compiler("", "hipcc 7.2");
        compiler.arch = "gfx1201".to_owned();

        assert_eq!(
            compiler.module_flags(module),
            SchedulerProfile::IterativeIlp
                .llvm_args()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            compiler.scheduler_profile_for(module),
            SchedulerProfile::IterativeIlp
        );
        assert!(compiler.module_flags(shipping).is_empty());
        assert!(compiler.module_flags("gemv_mq4g256v2_mq4v2").is_empty());
        assert_ne!(
            compiler.cache_hash(module, source),
            KernelCompiler::hash_parts(
                source,
                "gfx1201",
                "",
                &[],
                "hipcc 7.2",
                SchedulerProfile::Default,
                "hipcc", "", "", "",
            )
        );
        assert_eq!(
            compiler.cache_hash(shipping, source),
            KernelCompiler::hash_parts(
                source,
                "gfx1201",
                "",
                &[],
                "hipcc 7.2",
                SchedulerProfile::Default,
                "hipcc", "", "", "",
            )
        );
    }

    /// Write an executable fake hipcc that copies a canned blob to `-o` path,
    /// or exits nonzero when `fail` is true.
    fn install_fake_hipcc(bin_dir: &Path, canned_blob: &[u8], fail: bool) -> PathBuf {
        std::fs::create_dir_all(bin_dir).unwrap();
        let blob_path = bin_dir.join("canned.hsaco");
        std::fs::write(&blob_path, canned_blob).unwrap();
        let hipcc = bin_dir.join("fake_hipcc");
        let script = if fail {
            "#!/bin/sh\necho 'fake hipcc failure' >&2\nexit 1\n".to_string()
        } else {
            format!(
                "#!/bin/sh\nout=\"\"\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then out=\"$2\"; shift 2; continue; fi\n  shift\ndone\ncp \"{}\" \"$out\"\n",
                blob_path.display()
            )
        };
        std::fs::write(&hipcc, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&hipcc).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&hipcc, perms).unwrap();
        }
        hipcc
    }

    fn leftover_temps(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.contains(".tmp"))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn seed_orphan_cold_hash_cannot_certify_stale_hot_blob() {
        // CRITICAL: orphan cold .hash must never be copied beside a stale hot
        // blob and make compile() accept it as current.
        let root = temp_root("seed_orphan_hash");
        let _ = std::fs::remove_dir_all(&root);
        let cold = root.join("cold");
        let hot = root.join("hot");
        std::fs::create_dir_all(&cold).unwrap();
        std::fs::create_dir_all(&hot).unwrap();

        let name = "k";
        let current = "current_hash";
        // Cold has only the current hash (missing blob — repair case).
        std::fs::write(cold.join(format!("{name}.hash")), current).unwrap();
        // Hot has a stale-but-loadable blob with no hash.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"STALE_BLOB").unwrap();

        seed_hot_from_cold(&cold, &hot).unwrap();

        assert!(
            !hot.join(format!("{name}.hash")).exists(),
            "orphan cold hash must not be seeded beside a stale hot blob"
        );
        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"STALE_BLOB"
        );
        // pair_valid must still reject (no matching hash on hot).
        assert!(!pair_valid(
            &hot.join(format!("{name}.hsaco")),
            &hot.join(format!("{name}.hash")),
            current
        ));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn seed_pair_replaces_mismatched_hot_pair_as_unit() {
        let root = temp_root("seed_pair_replace");
        let _ = std::fs::remove_dir_all(&root);
        let cold = root.join("cold");
        let hot = root.join("hot");
        std::fs::create_dir_all(&cold).unwrap();
        std::fs::create_dir_all(&hot).unwrap();

        let name = "k";
        std::fs::write(cold.join(format!("{name}.hsaco")), b"COLD_BLOB").unwrap();
        std::fs::write(cold.join(format!("{name}.hash")), "cold_hash").unwrap();
        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_OLD").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "hot_old_hash").unwrap();

        seed_hot_from_cold(&cold, &hot).unwrap();

        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"COLD_BLOB"
        );
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            "cold_hash"
        );
        assert!(pair_valid(
            &hot.join(format!("{name}.hsaco")),
            &hot.join(format!("{name}.hash")),
            "cold_hash"
        ));
        assert!(
            leftover_temps(&hot).is_empty(),
            "no temp residue after pair seed: {:?}",
            leftover_temps(&hot)
        );

        // Matching hot pair must not be overwritten.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_MATCH").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "cold_hash").unwrap();
        seed_hot_from_cold(&cold, &hot).unwrap();
        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_MATCH",
            "matching hot pair must be preserved"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn empty_or_directory_blob_rejected_without_hipcc_single_and_batch() {
        let root = temp_root("empty_dir_blob");
        let _ = std::fs::remove_dir_all(&root);
        let hot = root.join("hot");
        std::fs::create_dir_all(&hot).unwrap();

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = Some(hot.clone());
        c.has_hipcc = false;

        let name = "add_inplace";
        let source = "__global__ void add_inplace() {}";
        let src_hash = c.cache_hash(name, source);

        // Empty file with matching hash.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();

        c.compile(name, source).unwrap_err();
        assert!(!c.compiled.contains_key(name));

        // Directory named like a blob with matching hash.
        c.compiled.clear();
        let _ = std::fs::remove_file(hot.join(format!("{name}.hsaco")));
        std::fs::create_dir(hot.join(format!("{name}.hsaco"))).unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();

        c.compile(name, source).unwrap_err();

        // Batch path must agree.
        c.compiled.clear();
        c.compile_batch(&[(name, source)]).unwrap_err();
        assert!(!c.compiled.contains_key(name));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_recompile_preserves_pair_when_hot_equals_cold() {
        let root = temp_root("recompile_hot_eq_cold");
        let _ = std::fs::remove_dir_all(&root);
        let shared = root.join("shared");
        std::fs::create_dir_all(&shared).unwrap();
        let hipcc = install_fake_hipcc(&root.join("bin"), b"NEW", true);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = shared.clone();
        c.precompiled_dir = Some(shared.clone());
        c.cold_dir = Some(shared.clone());
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        let src_hash = c.cache_hash(name, source);
        std::fs::write(shared.join(format!("{name}.hsaco")), b"PRIOR_PAIR").unwrap();
        std::fs::write(shared.join(format!("{name}.hash")), &src_hash).unwrap();

        let err = c.recompile(name, source, name).unwrap_err();
        assert!(err.to_string().contains("hipcc"), "{err}");

        assert_eq!(
            std::fs::read(shared.join(format!("{name}.hsaco"))).unwrap(),
            b"PRIOR_PAIR",
            "failed recompile must not delete hot==cold blob"
        );
        assert_eq!(
            std::fs::read_to_string(shared.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash,
            "failed recompile must not delete hot==cold hash"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_recompile_preserves_pair_when_dirs_aliased() {
        let root = temp_root("recompile_alias");
        let _ = std::fs::remove_dir_all(&root);
        let real = root.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let alias = root.join("alias");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &alias).unwrap();
        #[cfg(not(unix))]
        {
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        let hipcc = install_fake_hipcc(&root.join("bin"), b"NEW", true);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = real.clone();
        c.precompiled_dir = Some(alias.clone());
        c.cold_dir = Some(alias.clone());
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        let src_hash = c.cache_hash(name, source);
        std::fs::write(real.join(format!("{name}.hsaco")), b"ALIAS_PRIOR").unwrap();
        std::fs::write(real.join(format!("{name}.hash")), &src_hash).unwrap();

        let _ = c.recompile(name, source, name).unwrap_err();

        assert_eq!(
            std::fs::read(real.join(format!("{name}.hsaco"))).unwrap(),
            b"ALIAS_PRIOR"
        );
        assert_eq!(
            std::fs::read_to_string(real.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );
        // writeback_dir must treat alias as same path → no distinct cold.
        assert_eq!(c.writeback_dir(), None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_transactional_hipcc_leaves_old_hot_pair_intact() {
        let root = temp_root("tx_hipcc_fail");
        let _ = std::fs::remove_dir_all(&root);
        let hot = root.join("hot");
        std::fs::create_dir_all(&hot).unwrap();
        let hipcc = install_fake_hipcc(&root.join("bin"), b"NEW", true);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = None;
        c.cold_dir = None;
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        // Prior pair with a different hash so compile must attempt hipcc.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"OLD_HOT").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "old_hash").unwrap();

        let err = c.compile(name, source).unwrap_err();
        assert!(err.to_string().contains("hipcc"), "{err}");

        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"OLD_HOT"
        );
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            "old_hash"
        );
        assert!(
            leftover_temps(&hot).is_empty(),
            "failed hipcc must clean temps: {:?}",
            leftover_temps(&hot)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn successful_publish_pair_has_no_temp_residue() {
        let root = temp_root("publish_ok");
        let _ = std::fs::remove_dir_all(&root);
        let dest = root.join("dest");
        let src_dir = root.join("src");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::create_dir_all(&src_dir).unwrap();

        let name = "k";
        let src_blob = src_dir.join("blob.hsaco");
        std::fs::write(&src_blob, b"FRESH_BLOB").unwrap();
        // Prior pair to replace.
        std::fs::write(dest.join(format!("{name}.hsaco")), b"OLD").unwrap();
        std::fs::write(dest.join(format!("{name}.hash")), "old").unwrap();

        publish_pair(&dest, name, &src_blob, "new_hash", true).unwrap();

        assert_eq!(
            std::fs::read(dest.join(format!("{name}.hsaco"))).unwrap(),
            b"FRESH_BLOB"
        );
        assert_eq!(
            std::fs::read_to_string(dest.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            "new_hash"
        );
        assert!(pair_valid(
            &dest.join(format!("{name}.hsaco")),
            &dest.join(format!("{name}.hash")),
            "new_hash"
        ));
        assert!(
            leftover_temps(&dest).is_empty(),
            "successful publish must leave no temps: {:?}",
            leftover_temps(&dest)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn successful_hipcc_publish_yields_matching_pair_no_temps() {
        let root = temp_root("hipcc_publish_ok");
        let _ = std::fs::remove_dir_all(&root);
        let hot = root.join("hot");
        std::fs::create_dir_all(&hot).unwrap();
        let hipcc = install_fake_hipcc(&root.join("bin"), b"COMPILED_OK", false);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = None;
        c.cold_dir = None;
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        let src_hash = c.cache_hash(name, source);

        // Stale prior pair forces recompile path.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"STALE").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "stale").unwrap();

        let path = c.compile(name, source).expect("fake hipcc must succeed");
        // Hot outputs are content-keyed: the blob carries the full cache hash.
        let keyed = hot.join(hot_object_name(name, &src_hash));
        assert_eq!(path, &keyed);
        assert_eq!(std::fs::read(path).unwrap(), b"COMPILED_OK");
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.{src_hash}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );
        assert!(pair_valid(
            &hot.join(hot_object_name(name, &src_hash)),
            &hot.join(format!("{name}.{src_hash}.hash")),
            &src_hash
        ));
        assert!(
            leftover_temps(&hot).is_empty(),
            "successful hipcc publish must leave no temps: {:?}",
            leftover_temps(&hot)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn packaging_hash_matches_without_a_local_compiler() {
        let name = "packaged_kernel";
        let source = "__global__ void packaged_kernel() {}";
        let arch = "gfx1201";


        // Builder has a real toolchain, but packaging must use empty id.
        let mut builder = test_compiler("", "hipcc 7.2");
        builder.arch = arch.to_string();
        let packaging_via_instance = builder.packaging_hash(name, source);
        let packaging_via_static = KernelCompiler::packaging_hash_for(arch, name, source, "");
        assert_eq!(
            packaging_via_instance, packaging_via_static,
            "instance and static packaging hashes must agree (same hash_parts)"
        );

        // Compiler-free runtime computes with empty toolchain_id.
        let mut runtime = test_compiler("", "");
        runtime.arch = arch.to_string();
        let runtime_hash = runtime.cache_hash(name, source);
        assert_eq!(
            packaging_via_instance, runtime_hash,
            "packaging hash must equal what a hipcc-free runtime computes"
        );
        // And must differ from a builder's normal cache_hash (which folds in toolchain).
        let builder_hash = builder.cache_hash(name, source);
        assert_ne!(
            packaging_via_instance, builder_hash,
            "packaging (empty toolchain) must differ from normal cache hash"
        );
    }
    // Exercise both public lookup paths with separate instances so an earlier
    // successful lookup cannot hide a subsequent batch lookup.
    fn prebuilt_gate_compiler(root: &Path, with_hipcc: bool) -> KernelCompiler {
        let mut compiler = test_compiler("", "");
        compiler.arch = "gfx1201".to_owned();
        compiler.cache_dir = root.join("hot");
        std::fs::create_dir_all(&compiler.cache_dir).unwrap();
        let cold = root.join("cold");
        std::fs::create_dir_all(&cold).unwrap();
        compiler.precompiled_dir = Some(cold.clone());
        compiler.cold_dir = Some(cold);
        compiler.has_hipcc = with_hipcc;
        if with_hipcc {
            compiler.toolchain_id = "hipcc 7.2".to_owned();
            compiler.hipcc_bin = install_fake_hipcc(&root.join("bin"), b"FRESH", false);
        }
        compiler
    }


    #[test]
    fn prebuilt_gate_missing_or_wrong_hash_rejected() {
        for hash in [None, Some("wrong-key")] {
            let root = temp_root("gate_bad_hash");
            let name = "gate";
            let source = "__global__ void gate() {}";
            let cold = root.join("cold");
            let mut producer = prebuilt_gate_compiler(&root, true);
            producer.compile_for_symbol(name, source, name).unwrap();
            let single = prebuilt_gate_compiler(&root, false);
            let hash_path = cold.join("gate.hash");
            match hash {
                None => { std::fs::remove_file(&hash_path).unwrap(); }
                Some(bad) => { std::fs::write(&hash_path, bad).unwrap(); }
            }
            for (batch, mut compiler) in [single, prebuilt_gate_compiler(&root, false)]
                .into_iter()
                .enumerate()
            {
                let err = if batch == 0 {
                    compiler.compile(name, source).unwrap_err()
                } else {
                    compiler.compile_batch(&[(name, source)]).unwrap_err()
                };
                assert!(err.to_string().contains("pair missing or stale"), "{err}");
                assert!(!compiler.compiled.contains_key(name));
            }
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn prebuilt_gate_truncated_to_empty_and_legacy_without_hash_rejected() {
        for legacy in [false, true] {
            let root = temp_root("gate_empty_or_legacy");
            let name = "gate";
            let source = "__global__ void gate() {}";
            let single = prebuilt_gate_compiler(&root, false);
            let dir = if legacy { root.join("hot") } else { root.join("cold") };
            let object = dir.join("gate.hsaco");
            std::fs::write(&object, b"PREBUILT").unwrap();
            if legacy {
                // A legacy hot entry without its sidecar is not content-keyed.
            } else {
                std::fs::write(dir.join("gate.hash"), single.cache_hash(name, source)).unwrap();
                // Simulate an interrupted/truncated copy of a formerly valid pair.
                std::fs::write(&object, b"").unwrap();
            }
            for (batch, mut compiler) in [single, prebuilt_gate_compiler(&root, false)]
                .into_iter()
                .enumerate()
            {
                if batch == 0 {
                    compiler.compile(name, source).unwrap_err();
                } else {
                    compiler.compile_batch(&[(name, source)]).unwrap_err();
                }
                assert!(!compiler.compiled.contains_key(name));
            }
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn prebuilt_gate_bad_cold_falls_through_to_hipcc() {
        for batch in [false, true] {
            let root = temp_root("gate_recompile");
            let name = "gate";
            let source = "__global__ void gate() {}";
            let mut compiler = prebuilt_gate_compiler(&root, true);
            let old = root.join("cold/gate.hsaco");
            std::fs::write(&old, b"STALE").unwrap();
            std::fs::write(root.join("cold/gate.hash"), "wrong-key").unwrap();
            let key = compiler.cache_hash(name, source);
            if batch {
                compiler.compile_batch(&[(name, source)]).unwrap();
            } else {
                compiler.compile(name, source).unwrap();
            }
            let hot = root.join("hot").join(hot_object_name(name, &key));
            assert_eq!(compiler.compiled[name], hot);
            assert_eq!(std::fs::read(&hot).unwrap(), b"FRESH");
            assert_eq!(std::fs::read(&old).unwrap(), b"FRESH");
            assert_eq!(std::fs::read_to_string(root.join("cold/gate.hash")).unwrap(), compiler.packaging_hash(name, source));
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn stale_indexed_cold_cannot_fall_through_to_hot_without_compiler() {
        for batch in [false, true] {
            let root = temp_root("gate_hot_hit");
            let name = "gate";
            let source = "__global__ void gate() {}";
            let mut producer = prebuilt_gate_compiler(&root, true);
            producer.compile_for_symbol(name, source, name).unwrap();
            let mut compiler = prebuilt_gate_compiler(&root, false);
            std::fs::write(root.join("cold/gate.hsaco"), b"CORRUPTED").unwrap();
            let key = compiler.cache_hash(name, source);
            let hot = root.join("hot").join(hot_object_name(name, &key));
            std::fs::write(&hot, b"HOT_VALID").unwrap();
            if batch {
                compiler.compile_batch(&[(name, source)]).unwrap_err();
            } else {
                compiler.compile(name, source).unwrap_err();
            }
            assert!(!compiler.compiled.contains_key(name));
            assert_eq!(std::fs::read(&hot).unwrap(), b"HOT_VALID");
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn precompile_indexed_pair_is_accepted_without_compiler_for_both_lookups() {
        let root = temp_root("indexed_precompile");
        let mut producer = prebuilt_gate_compiler(&root, true);
        let source = "__global__ void rmsnorm_f32() {}";
        let name = "rmsnorm";
        let symbol = "rmsnorm_f32";
        let hot_key = producer.cache_hash(name, source);
        let portable_key = producer.packaging_hash(name, source);
        assert_ne!(hot_key, portable_key);
        producer.compile_batch_for_symbols(&[(name, source, symbol)]).unwrap();
        let cold = root.join("cold");
        let object = cold.join("rmsnorm.hsaco");
        assert!(pair_valid(&object, &cold.join("rmsnorm.hash"), &portable_key));
        let index: PackIndex = serde_json::from_slice(&std::fs::read(cold.join("rmsnorm.index.json")).unwrap()).unwrap();
        assert_eq!(index.toolchain_id, "hipcc 7.2");
        assert_eq!(index.object_sha256, sha256_hex(b"FRESH"));
        assert_eq!(index.symbols, [symbol]);

        for batch in [false, true] {
            let mut consumer = prebuilt_gate_compiler(&root, false);
            if batch {
                consumer.compile_batch_for_symbols(&[(name, source, symbol)]).unwrap();
                assert_eq!(consumer.compiled[name], object);
            } else {
                assert_eq!(consumer.compile_for_symbol(name, source, symbol).unwrap(), object);
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn indexed_writeback_preserves_all_symbols_in_shared_module() {
        let root = temp_root("shared_module_symbols");
        let source = "__global__ void a() {} __global__ void b() {}";
        let mut producer = prebuilt_gate_compiler(&root, true);
        producer.compile_for_symbol("shared", source, "a").unwrap();
        producer.compile_for_symbol("shared", source, "b").unwrap();
        let mut consumer = prebuilt_gate_compiler(&root, false);
        assert!(consumer.compile_for_symbol("shared", source, "a").is_ok());
        assert!(consumer.compile_for_symbol("shared", source, "b").is_ok());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn indexed_package_rejects_wrong_object_symbol_source_and_identity() {
        let root = temp_root("indexed_negative");
        let source = "__global__ void rmsnorm_f32() {}";
        let name = "rmsnorm";
        let symbol = "rmsnorm_f32";
        let mut producer = prebuilt_gate_compiler(&root, true);
        producer.compile_for_symbol(name, source, symbol).unwrap();
        let cold = root.join("cold");
        let object = cold.join("rmsnorm.hsaco");
        let index_path = cold.join("rmsnorm.index.json");
        let original = std::fs::read(&index_path).unwrap();

        for field in ["arch", "source_sha256", "flags", "scheduler_profile", "cache_abi", "packaging_key", "object_sha256", "toolchain_id"] {
            let mut mutated: serde_json::Value = serde_json::from_slice(&original).unwrap();
            mutated[field] = match field {
                "flags" => serde_json::json!(["-O0"]),
                "cache_abi" => serde_json::json!(0),
                _ => serde_json::json!("wrong"),
            };
            std::fs::write(&index_path, serde_json::to_vec(&mutated).unwrap()).unwrap();
            let mut consumer = prebuilt_gate_compiler(&root, false);
            let err = consumer.compile_for_symbol(name, source, symbol).unwrap_err();
            assert!(err.to_string().contains("packaged object rejected"), "{field}: {err}");
        }
        std::fs::write(&index_path, &original).unwrap();
        let mut consumer = prebuilt_gate_compiler(&root, false);
        assert!(consumer.compile_for_symbol(name, source, "other_symbol").is_err());
        assert!(consumer.compile_for_symbol(name, "__global__ void changed() {}", symbol).is_err());
        std::fs::write(&object, b"CORRUPTED").unwrap();
        let mut consumer = prebuilt_gate_compiler(&root, false);
        assert!(consumer.compile_batch_for_symbols(&[(name, source, symbol)]).unwrap_err().to_string().contains("SHA-256"));
        let _ = std::fs::remove_dir_all(root);
    }
}
