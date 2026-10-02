// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Install a release kernel pack instead of compiling the registry locally.
//!
//! `hipfire setup --tag/--ref` and `install.ps1` both come through
//! [`install`]. Every failure is a reason to fall back to local compilation,
//! never a partial install: the pack is fetched, SHA-256 checked, unpacked
//! into a same-volume staging directory, checked against the manifest, the
//! local ROCm and compiler, and every index is re-verified against the
//! checked-out sources before the directory replaces the installed one.

use anyhow::Result;
use flate2::read::GzDecoder;
use rdna_compute::kernel_pack::{self, PackManifest, MANIFEST_FILE};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
};

/// Release download base for `tag` on the official repository.
pub(crate) fn default_base_url(tag: &str) -> String {
    format!("https://github.com/warpfront/hipfire/releases/download/{tag}")
}

/// Largest compressed pack accepted (current packs are a few MiB).
const MAX_PACK_BYTES: u64 = 512 << 20;

pub(crate) struct PackRequest<'a> {
    pub tag: &'a str,
    pub arch: &'a str,
    /// Checkout the daemon was built from; its HEAD must equal the pack commit.
    pub commit: &'a str,
    /// Installed `.../kernels/compiled/<arch>` directory to replace.
    pub dest: &'a Path,
    /// Directory or URL holding the release assets (`https://`, `file://`, or a path).
    pub base: &'a str,
    /// `<rocm root>/.info/version` of the machine's ROCm, if known.
    pub rocm_version: Option<&'a str>,
}

/// Download, verify and install one pack. `Err` carries the reason the caller
/// falls back to local compilation; the installed directory is untouched then.
pub(crate) fn install(request: &PackRequest<'_>) -> std::result::Result<PackManifest, String> {
    if !request.arch.starts_with("gfx") || !request.arch[3..].bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(format!("invalid GPU arch {}", request.arch));
    }
    let stem = kernel_pack::asset_stem(request.tag, request.arch);
    let tarball_name = format!("{stem}.tar.gz");
    let sha_name = format!("{tarball_name}.sha256");
    let base = request.base.trim_end_matches('/');
    let sha_text = fetch(&format!("{base}/{sha_name}"))?
        .ok_or_else(|| format!("no kernel pack {sha_name} at {base}"))?;
    let expected = parse_sha256_line(&String::from_utf8_lossy(&sha_text), &tarball_name)?;
    let tarball = fetch(&format!("{base}/{tarball_name}"))?
        .ok_or_else(|| format!("no kernel pack {tarball_name} at {base}"))?;
    let actual = format!("{:x}", Sha256::digest(&tarball));
    if actual != expected {
        return Err(format!(
            "{tarball_name}: SHA-256 {actual} does not match published {expected}"
        ));
    }

    let parent = request
        .dest
        .parent()
        .ok_or_else(|| format!("{}: no parent directory", request.dest.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let nonce = unique_suffix();
    let staging = parent.join(format!(".{}.pack-{nonce}", request.arch));
    let result = (|| {
        unpack(&tarball, request.arch, &staging)?;
        let manifest_bytes =
            fs::read(staging.join(MANIFEST_FILE)).map_err(|e| format!("{MANIFEST_FILE}: {e}"))?;
        let manifest: PackManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|e| format!("{MANIFEST_FILE}: {e}"))?;
        if manifest.tag != request.tag {
            return Err(format!(
                "pack is tagged {}, requested {}",
                manifest.tag, request.tag
            ));
        }
        manifest.check(request.arch, request.commit, request.rocm_version)?;
        // Validate first: the snapshot below panics on an invalid config.
        hipfire_config::load_local_process_config()
            .map_err(|e| format!("invalid hipfire configuration: {e}"))?;
        let local = rdna_compute::KernelCompiler::local_toolchain_id();
        if let Some(local) = local
            .as_deref()
            .filter(|local| *local != manifest.toolchain_id)
        {
            return Err(format!(
                "pack was compiled by `{}` but the local compiler is `{local}`; the runtime would \
                 recompile every kernel",
                manifest.toolchain_id
            ));
        }
        let extra_flags =
            rdna_compute::FeatureFlags::from_active_config(request.arch).hipcc_extra_flags;
        if extra_flags != manifest.extra_flags {
            return Err(format!(
                "this configuration selects kernel flags `{extra_flags}`, the pack was built with `{}`",
                manifest.extra_flags
            ));
        }
        let arch_dir = staging.join(request.arch);
        let verified =
            kernel_pack::verify_dir(request.arch, &extra_flags, &arch_dir, local.as_deref())?;
        manifest.check_verified(&verified)?;
        replace_dir(&arch_dir, request.dest, &nonce)?;
        Ok(manifest)
    })();
    let _ = fs::remove_dir_all(&staging);
    result
}

fn unique_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{nanos}", std::process::id())
}

/// `https://…`/`http://…` via HTTP, `file://…` or a plain path from disk.
/// `Ok(None)` means the asset does not exist (HTTP 404 or missing file).
fn fetch(location: &str) -> std::result::Result<Option<Vec<u8>>, String> {
    if location.starts_with("https://") || location.starts_with("http://") {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_global(Some(Duration::from_secs(10 * 60)))
            .http_status_as_error(false)
            .build()
            .into();
        let mut response = agent
            .get(location)
            .call()
            .map_err(|error| format!("{location}: {error}"))?;
        let status = response.status();
        if status == 404 {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(format!("{location}: HTTP {status}"));
        }
        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .take(MAX_PACK_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("{location}: {error}"))?;
        if bytes.len() as u64 > MAX_PACK_BYTES {
            return Err(format!("{location}: larger than {MAX_PACK_BYTES} bytes"));
        }
        return Ok(Some(bytes));
    }
    let mut local = location.strip_prefix("file://").unwrap_or(location);
    // `file:///C:/packs` names the Windows path `C:/packs`.
    if cfg!(windows) && local.as_bytes().get(2) == Some(&b':') && local.starts_with('/') {
        local = &local[1..];
    }
    let path = PathBuf::from(local);
    match fs::read(&path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

/// `sha256sum` output: `<64 hex>  <name>` (a leading `*` marks binary mode).
fn parse_sha256_line(text: &str, expected_name: &str) -> std::result::Result<String, String> {
    let mut fields = text.split_whitespace();
    let digest = fields.next().unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{expected_name}.sha256: no SHA-256 digest"));
    }
    if let Some(name) = fields.next() {
        if name.trim_start_matches('*') != expected_name {
            return Err(format!("{expected_name}.sha256 names {name}"));
        }
    }
    Ok(digest.to_ascii_lowercase())
}

/// Unpack a gzip'd ustar pack holding exactly `manifest.json` and flat
/// regular files under `<arch>/`. Links, devices, nested paths and duplicate
/// members are rejected rather than interpreted.
fn unpack(tarball: &[u8], arch: &str, staging: &Path) -> std::result::Result<(), String> {
    let mut archive = Vec::new();
    GzDecoder::new(tarball)
        .take(MAX_PACK_BYTES * 8)
        .read_to_end(&mut archive)
        .map_err(|e| format!("pack is not valid gzip: {e}"))?;
    fs::create_dir(staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let arch_dir = staging.join(arch);
    fs::create_dir(&arch_dir).map_err(|e| format!("{}: {e}", arch_dir.display()))?;
    let mut seen = BTreeSet::new();
    let mut offset = 0usize;
    loop {
        let header = archive
            .get(offset..offset + 512)
            .ok_or("pack tar ends without an end-of-archive block")?;
        if header.iter().all(|&b| b == 0) {
            break;
        }
        let stored_sum = octal(&header[148..156]).ok_or("pack tar header has a bad checksum")?;
        let computed: u64 = header
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                if (148..156).contains(&i) {
                    u64::from(b' ')
                } else {
                    u64::from(b)
                }
            })
            .sum();
        if stored_sum != computed {
            return Err("pack tar header checksum mismatch".into());
        }
        let field = |range: std::ops::Range<usize>| {
            let raw = &header[range];
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            String::from_utf8(raw[..end].to_vec())
        };
        let name = field(0..100).map_err(|_| "pack tar member name is not UTF-8")?;
        let prefix = field(345..500).map_err(|_| "pack tar member prefix is not UTF-8")?;
        let path = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let size =
            usize::try_from(octal(&header[124..136]).ok_or("pack tar member has a bad size")?)
                .map_err(|_| "pack tar member too large")?;
        let data_start = offset + 512;
        let data = archive
            .get(data_start..data_start + size)
            .ok_or("pack tar member is truncated")?;
        offset = data_start + size.div_ceil(512) * 512;
        if !seen.insert(path.clone()) {
            return Err(format!("pack tar repeats {path}"));
        }
        match header[156] {
            b'5' if path == format!("{arch}/") || path == arch => continue,
            b'0' | 0 => {}
            kind => {
                return Err(format!(
                    "pack tar member {path} has unsupported type {}",
                    kind as char
                ))
            }
        }
        let target = if path == MANIFEST_FILE {
            staging.join(MANIFEST_FILE)
        } else {
            let file = path
                .strip_prefix(&format!("{arch}/"))
                .filter(|file| {
                    !file.is_empty()
                        && !file.starts_with('.')
                        && file.bytes().all(|b| {
                            b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-'
                        })
                })
                .ok_or_else(|| format!("pack tar member {path} is outside {arch}/"))?;
            arch_dir.join(file)
        };
        fs::write(&target, data).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    if !seen.contains(MANIFEST_FILE) {
        return Err(format!("pack has no {MANIFEST_FILE}"));
    }
    Ok(())
}

/// NUL/space-terminated octal tar number.
fn octal(field: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(field).ok()?;
    let digits = text.trim_matches(|c: char| c == '\0' || c == ' ');
    if digits.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(digits, 8).ok()
}

/// Swap the verified directory into place; the prior directory is restored if
/// the swap fails and removed once it succeeds.
fn replace_dir(verified: &Path, dest: &Path, nonce: &str) -> std::result::Result<(), String> {
    let backup = dest.with_file_name(format!(
        ".{}.replaced-{nonce}",
        dest.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("kernels")
    ));
    let had_prior = dest.exists();
    if had_prior {
        fs::rename(dest, &backup).map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    if let Err(error) = fs::rename(verified, dest) {
        if had_prior {
            let _ = fs::rename(&backup, dest);
        }
        return Err(format!("{}: {error}", dest.display()));
    }
    if had_prior {
        let _ = fs::remove_dir_all(&backup);
    }
    Ok(())
}

/// `hipfire kernel-pack install`: the same path `hipfire setup` takes, for
/// `install.ps1` and for repairing an install without a device compiler.
pub(crate) fn install_command(
    paths: &crate::Paths,
    args: crate::KernelPackInstallArgs,
) -> Result<()> {
    let commit = crate::git_output(&args.source, &["rev-parse", "HEAD"])?;
    let dest = args.dest.unwrap_or_else(|| {
        paths
            .root
            .join("bin")
            .join("kernels")
            .join("compiled")
            .join(&args.arch)
    });
    let rocm_version = args.rocm_version.or_else(|| {
        hipfire_config::rocm::root()
            .and_then(|root| hipfire_config::rocm::version_for_root(&root))
            .or_else(hipfire_config::rocm::version)
    });
    let base = args.url.unwrap_or_else(|| default_base_url(&args.tag));
    let manifest = install(&PackRequest {
        tag: &args.tag,
        arch: &args.arch,
        commit: &commit,
        dest: &dest,
        base: &base,
        rocm_version: rocm_version.as_deref(),
    })
    .map_err(|reason| anyhow::anyhow!("kernel pack not installed: {reason}"))?;
    println!(
        "installed kernel pack {} for {} ({} modules, {}) at {}",
        manifest.tag,
        manifest.arch,
        manifest.modules,
        manifest.toolchain_id,
        dest.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ustar(members: &[(&str, u8, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, kind, data) in members {
            let mut header = [0u8; 512];
            header[..name.len()].copy_from_slice(name.as_bytes());
            header[100..107].copy_from_slice(b"0000644");
            header[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
            header[156] = *kind;
            header[257..263].copy_from_slice(b"ustar\0");
            header[148..156].fill(b' ');
            let sum: u64 = header.iter().map(|&b| u64::from(b)).sum();
            header[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
            out.extend_from_slice(&header);
            out.extend_from_slice(data);
            out.resize(out.len().div_ceil(512) * 512, 0);
        }
        out.resize(out.len() + 1024, 0);
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut gz, &out).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn unpack_accepts_only_flat_regular_members_under_arch() {
        let root = std::env::temp_dir().join(format!("hipfire-pack-unpack-{}", unique_suffix()));
        fs::create_dir_all(&root).unwrap();
        let good = ustar(&[
            ("gfx1201/", b'5', b""),
            ("gfx1201/rmsnorm.hsaco", b'0', b"OBJ"),
            ("manifest.json", b'0', b"{}"),
        ]);
        unpack(&good, "gfx1201", &root.join("ok")).unwrap();
        assert_eq!(
            fs::read(root.join("ok/gfx1201/rmsnorm.hsaco")).unwrap(),
            b"OBJ"
        );
        for (label, bad) in [
            (
                "traversal",
                ustar(&[
                    ("gfx1201/../x.hsaco", b'0', b"x"),
                    ("manifest.json", b'0', b"{}"),
                ]),
            ),
            (
                "other arch",
                ustar(&[
                    ("gfx1100/x.hsaco", b'0', b"x"),
                    ("manifest.json", b'0', b"{}"),
                ]),
            ),
            (
                "symlink",
                ustar(&[
                    ("gfx1201/x.hsaco", b'2', b""),
                    ("manifest.json", b'0', b"{}"),
                ]),
            ),
            (
                "nested",
                ustar(&[
                    ("gfx1201/sub/x.hsaco", b'0', b"x"),
                    ("manifest.json", b'0', b"{}"),
                ]),
            ),
            (
                "duplicate",
                ustar(&[
                    ("manifest.json", b'0', b"{}"),
                    ("manifest.json", b'0', b"{}"),
                ]),
            ),
            ("no manifest", ustar(&[("gfx1201/x.hsaco", b'0', b"x")])),
        ] {
            let staging = root.join(label.replace(' ', "-"));
            assert!(
                unpack(&bad, "gfx1201", &staging).is_err(),
                "{label} accepted"
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sha256_line_must_name_the_tarball() {
        let digest = "ab".repeat(32);
        assert_eq!(
            parse_sha256_line(&format!("{digest}  p.tar.gz\n"), "p.tar.gz").unwrap(),
            digest
        );
        assert_eq!(
            parse_sha256_line(&format!("{digest} *p.tar.gz"), "p.tar.gz").unwrap(),
            digest
        );
        assert!(parse_sha256_line(&format!("{digest}  other.tar.gz"), "p.tar.gz").is_err());
        assert!(parse_sha256_line("not-a-digest  p.tar.gz", "p.tar.gz").is_err());
    }
}
