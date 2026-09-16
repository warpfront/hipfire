// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Opt-in gfx1151 XDNA NPU spillover sidecar (`.xdna.zip`), manifest v1.
//!
//! A sidecar is a **stored** (uncompressed) ZIP archive with a `manifest.json`
//! plus one flat `main.pdi` and one flat `insts.bin` per profile. The runtime
//! consumes the flat PDI/insts bytes only; the source xclbin may be retained
//! beside them as provenance but is never executed.
//!
//! Manifest v1 schema (frozen; unknown fields are rejected):
//!
//! ```json
//! {
//!   "version": 1,
//!   "model_sha256": "<64 hex>",
//!   "quant": "mq4g256v2",
//!   "arch": "gfx1151",
//!   "npu": "npu5",
//!   "toolchain": {"mlir_aie": "<rev>", "peano": "<version>"},
//!   "profiles": [{
//!     "tensor_role": "gate_up",
//!     "M": 4096, "K": 5120, "N": 512, "row_count": 4096,
//!     "tile_m": 128, "tile_k": 64, "tile_n": 64,
//!     "cols": 8, "k_mt": 512,
//!     "pdi": "profiles/gate_up/main.pdi",
//!     "insts": "profiles/gate_up/insts.bin",
//!     "arg_layout": [{"name": "a", "offset": 0, "size": 4096}],
//!     "sha256_pdi": "<64 hex>",
//!     "sha256_insts": "<64 hex>"
//!   }]
//! }
//! ```
//!
//! [`XdnaSidecarDescriptor::load_verified`] checks every offset/range and
//! payload hash once at load: manifest shape, exact `arch == "gfx1151"` and
//! `npu == "npu5"`, per-profile payload presence, and the SHA-256 of each
//! payload against the manifest. Anything else fails closed. Device admission
//! (exact `gfx1151` GPU string) is [`XdnaSidecarDescriptor::admit_for_arch`];
//! the process opt-in flag (`kernel.npu_spillover`) is checked by the caller.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
use thiserror::Error;

/// Manifest schema version this parser accepts.
pub const XDNA_MANIFEST_VERSION: u32 = 1;
/// Only this exact GPU architecture string is admittable.
pub const XDNA_ADMIT_ARCH: &str = "gfx1151";
/// Only this NPU identifier is admittable.
pub const XDNA_ADMIT_NPU: &str = "npu5";
/// Manifest entry name inside every `.xdna.zip`.
pub const XDNA_MANIFEST_NAME: &str = "manifest.json";
/// Refuse any single stored entry larger than this (PDI/insts are ~10^5 B).
const MAX_STORED_ENTRY_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum XdnaError {
    #[error("xdna sidecar I/O for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("xdna sidecar {path}: invalid manifest: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("xdna sidecar {path}: payload hash mismatch for {entry}: manifest {expected}, file {actual}")]
    ShaMismatch {
        path: PathBuf,
        entry: String,
        expected: String,
        actual: String,
    },
    #[error("xdna sidecar {path}: malformed stored zip: {message}")]
    Zip { path: PathBuf, message: String },
}

pub type Result<T, E = XdnaError> = std::result::Result<T, E>;

/// One named kernel-argument slot: byte offset and size within the ERT packet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XdnaArgLayout {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

/// One admitted (M, K, N) tile shape with validated payload bindings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XdnaProfile {
    pub tensor_role: String,
    #[serde(rename = "M")]
    pub m: u64,
    #[serde(rename = "K")]
    pub k: u64,
    #[serde(rename = "N")]
    pub n: u64,
    pub row_count: u64,
    pub tile_m: u64,
    pub tile_k: u64,
    pub tile_n: u64,
    pub cols: u64,
    pub k_mt: u64,
    pub pdi: String,
    pub insts: String,
    pub arg_layout: Vec<XdnaArgLayout>,
    pub sha256_pdi: String,
    pub sha256_insts: String,
}

/// Manifest v1 root. `toolchain` carries offline toolchain ids (e.g.
/// `mlir_aie` revision, `peano` version) as provenance, never executed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct XdnaManifest {
    pub version: u32,
    pub model_sha256: String,
    pub quant: String,
    pub arch: String,
    pub npu: String,
    pub toolchain: BTreeMap<String, String>,
    pub profiles: Vec<XdnaProfile>,
}

/// A load-time verified sidecar: the archive path plus its checked manifest.
/// `verified` is true only when every payload hash matched at load; the flag
/// exists so a future unverified-construct path cannot be mistaken for this one.
#[derive(Clone, Debug, PartialEq)]
pub struct XdnaSidecarDescriptor {
    pub path: PathBuf,
    pub manifest: XdnaManifest,
    pub verified: bool,
}

impl XdnaManifest {
    /// Check every range, identity binding, and hash shape once at load.
    /// Payload *bytes* are checked by [`XdnaSidecarDescriptor::load_verified`].
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.version != XDNA_MANIFEST_VERSION {
            return Err(format!("unsupported manifest version {}", self.version));
        }
        validate_hex_digest(&self.model_sha256, "model_sha256")?;
        if self.quant.trim().is_empty() {
            return Err("quant is empty".into());
        }
        if self.arch != XDNA_ADMIT_ARCH {
            return Err(format!(
                "arch '{}' is not admittable (expected '{XDNA_ADMIT_ARCH}')",
                self.arch
            ));
        }
        if self.npu != XDNA_ADMIT_NPU {
            return Err(format!(
                "npu '{}' is not admittable (expected '{XDNA_ADMIT_NPU}')",
                self.npu
            ));
        }
        if self.toolchain.is_empty() {
            return Err("toolchain ids are empty".into());
        }
        if self.profiles.is_empty() {
            return Err("no profiles declared".into());
        }
        for (index, profile) in self.profiles.iter().enumerate() {
            profile
                .validate()
                .map_err(|message| format!("profile {index}: {message}"))?;
        }
        Ok(())
    }
}

impl XdnaProfile {
    fn validate(&self) -> std::result::Result<(), String> {
        if self.tensor_role.trim().is_empty() {
            return Err("tensor_role is empty".into());
        }
        for (name, value) in [
            ("M", self.m),
            ("K", self.k),
            ("N", self.n),
            ("row_count", self.row_count),
            ("tile_m", self.tile_m),
            ("tile_k", self.tile_k),
            ("tile_n", self.tile_n),
            ("cols", self.cols),
            ("k_mt", self.k_mt),
        ] {
            if value == 0 {
                return Err(format!("{name} must be nonzero"));
            }
        }
        if self.row_count > self.m {
            return Err(format!(
                "row_count {} exceeds M {}",
                self.row_count, self.m
            ));
        }
        validate_archive_name(&self.pdi, "pdi")?;
        validate_archive_name(&self.insts, "insts")?;
        if self.arg_layout.is_empty() {
            return Err("arg_layout is empty".into());
        }
        for arg in &self.arg_layout {
            if arg.name.trim().is_empty() {
                return Err("arg_layout entry has an empty name".into());
            }
            if arg.size == 0 {
                return Err(format!("arg_layout '{}' has zero size", arg.name));
            }
            arg.offset.checked_add(arg.size).ok_or_else(|| {
                format!("arg_layout '{}' offset+size overflows", arg.name)
            })?;
        }
        validate_hex_digest(&self.sha256_pdi, "sha256_pdi")?;
        validate_hex_digest(&self.sha256_insts, "sha256_insts")?;
        Ok(())
    }
}

impl XdnaSidecarDescriptor {
    /// Open `path`, parse and validate `manifest.json`, then SHA-256 every
    /// profile payload against the manifest. Fails closed on any mismatch.
    pub fn load_verified(path: &Path) -> Result<Self> {
        let read = |entry: &str| read_stored_entry(path, entry);
        let manifest_bytes = read(XDNA_MANIFEST_NAME)?;
        let manifest: XdnaManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
            XdnaError::Invalid {
                path: path.to_owned(),
                message: format!("{XDNA_MANIFEST_NAME}: {error}"),
            }
        })?;
        manifest.validate().map_err(|message| XdnaError::Invalid {
            path: path.to_owned(),
            message,
        })?;
        // Bind each payload to its hash before anyone can consume the manifest.
        let mut want = BTreeSet::new();
        for profile in &manifest.profiles {
            want.insert(profile.pdi.clone());
            want.insert(profile.insts.clone());
        }
        let payloads = read_stored_entries(path, &want)?;
        for profile in &manifest.profiles {
            for (entry, expected) in [
                (&profile.pdi, &profile.sha256_pdi),
                (&profile.insts, &profile.sha256_insts),
            ] {
                let bytes = payloads.get(entry).expect("entry was requested");
                let actual = sha256_hex(bytes);
                if actual != *expected {
                    return Err(XdnaError::ShaMismatch {
                        path: path.to_owned(),
                        entry: entry.clone(),
                        expected: expected.clone(),
                        actual,
                    });
                }
            }
        }
        Ok(Self {
            path: path.to_owned(),
            manifest,
            verified: true,
        })
    }

    /// Exact-architecture admission: only `gfx1151` passes. The caller checks
    /// the `kernel.npu_spillover` process flag first; a rejection here carries
    /// the reason so the loader can log why an opted-in load stays GPU-only.
    pub fn admit_for_arch(&self, gpu_arch: &str) -> std::result::Result<(), String> {
        if gpu_arch == XDNA_ADMIT_ARCH {
            return Ok(());
        }
        Err(format!(
            "xdna sidecar '{}' requires exact arch '{XDNA_ADMIT_ARCH}', host is '{gpu_arch}'; staying GPU-only",
            self.path.display()
        ))
    }
}

fn validate_hex_digest(digest: &str, field: &str) -> std::result::Result<(), String> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} is not a 64-hex SHA-256"));
    }
    Ok(())
}

/// Reject absolute paths, parent escapes, and empty names so a hostile
/// manifest cannot steer extraction outside the archive's namespace.
fn validate_archive_name(name: &str, field: &str) -> std::result::Result<(), String> {
    if name.trim().is_empty() {
        return Err(format!("{field} entry name is empty"));
    }
    let path = Path::new(name);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(format!("{field} entry name '{name}' escapes the archive"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(hex(byte >> 4));
        out.push(hex(byte & 0x0F));
    }
    out
}

fn hex(nibble: u8) -> char {
    (if nibble < 10 { b'0' + nibble } else { b'a' + nibble - 10 }) as char
}

/// Read one stored entry from the archive.
fn read_stored_entry(path: &Path, entry: &str) -> Result<Vec<u8>> {
    let mut want = BTreeSet::new();
    want.insert(entry.to_owned());
    let mut entries = read_stored_entries(path, &want)?;
    entries.remove(entry).ok_or_else(|| XdnaError::Zip {
        path: path.to_owned(),
        message: format!("entry '{entry}' vanished after central-directory parse"),
    })
}

/// Minimal stored-ZIP reader: the sidecar contract mandates stored
/// (uncompressed) entries, so anything else — deflate, encryption, archives
/// without a central directory — is rejected rather than decoded.
fn read_stored_entries(path: &Path, want: &BTreeSet<String>) -> Result<BTreeMap<String, Vec<u8>>> {
    let io = |source: std::io::Error| XdnaError::Io {
        path: path.to_owned(),
        source,
    };
    let zip = |message: String| XdnaError::Zip {
        path: path.to_owned(),
        message,
    };
    let mut file = File::open(path).map_err(io)?;
    let file_len = file.metadata().map_err(io)?.len();
    if file_len < 22 {
        return Err(zip("file is too small to be a zip archive".into()));
    }
    // Locate the end-of-central-directory record (it may carry a comment, so
    // scan backwards over the last min(len, 64 KiB + 22) bytes).
    let tail_len = file_len.min(65557 + 22);
    file.seek(SeekFrom::End(-(tail_len as i64))).map_err(io)?;
    let mut tail = vec![0u8; tail_len as usize];
    file.read_exact(&mut tail).map_err(io)?;
    let eocd = tail
        .windows(4)
        .rposition(|window| window == b"PK\x05\x06")
        .ok_or_else(|| zip("end-of-central-directory record not found".into()))?;
    let eocd = &tail[eocd..];
    if eocd.len() < 22 {
        return Err(zip("truncated end-of-central-directory record".into()));
    }
    let central_count = u16_at(eocd, 10) as u64;
    let central_size = u32_at(eocd, 12) as u64;
    let central_offset = u32_at(eocd, 16) as u64;
    if central_offset.checked_add(central_size).is_none_or(|end| end > file_len) {
        return Err(zip("central directory runs past end of file".into()));
    }
    // Walk the central directory, collecting the wanted stored entries.
    file.seek(SeekFrom::Start(central_offset)).map_err(io)?;
    let mut central = vec![0u8; central_size as usize];
    file.read_exact(&mut central).map_err(io)?;
    let mut cursor = 0usize;
    // (name, local-header offset, size)
    let mut hits: Vec<(String, u64, u64)> = Vec::new();
    for _ in 0..central_count {
        let rest = central.get(cursor..).ok_or_else(|| zip("truncated central directory".into()))?;
        if rest.len() < 46 || &rest[0..4] != b"PK\x01\x02" {
            return Err(zip("bad central-directory entry signature".into()));
        }
        let flags = u16_at(rest, 8);
        let method = u16_at(rest, 10);
        let size = u32_at(rest, 24) as u64;
        let name_len = u16_at(rest, 28) as usize;
        let extra_len = u16_at(rest, 30) as usize;
        let comment_len = u16_at(rest, 32) as usize;
        let local_offset = u32_at(rest, 42) as u64;
        let end = cursor
            .checked_add(46 + name_len + extra_len + comment_len)
            .filter(|end| *end <= central.len())
            .ok_or_else(|| zip("central-directory entry overruns".into()))?;
        let name = std::str::from_utf8(&central[cursor + 46..cursor + 46 + name_len])
            .map_err(|_| zip("non-UTF8 entry name".into()))?;
        cursor = end;
        if !want.contains(name) {
            continue;
        }
        if flags & 0x01 != 0 {
            return Err(zip(format!("entry '{name}' is encrypted")));
        }
        if method != 0 {
            return Err(zip(format!(
                "entry '{name}' uses compression method {method}; sidecars must be stored"
            )));
        }
        if size > MAX_STORED_ENTRY_BYTES {
            return Err(zip(format!("entry '{name}' exceeds the size cap")));
        }
        hits.push((name.to_owned(), local_offset, size));
    }
    for name in want {
        if !hits.iter().any(|(hit, _, _)| hit == name) {
            return Err(zip(format!("entry '{name}' not found in archive")));
        }
    }
    // Extract each hit through its local header (names must agree).
    let mut out = BTreeMap::new();
    for (name, local_offset, size) in hits {
        file.seek(SeekFrom::Start(local_offset)).map_err(io)?;
        let mut header = [0u8; 30];
        file.read_exact(&mut header).map_err(io)?;
        if &header[0..4] != b"PK\x03\x04" {
            return Err(zip(format!("entry '{name}' has a bad local header")));
        }
        if u16_at(&header, 8) != 0 {
            return Err(zip(format!(
                "entry '{name}' is compressed; sidecars must be stored"
            )));
        }
        let name_len = u16_at(&header, 26) as usize;
        let extra_len = u16_at(&header, 28) as usize;
        let header_len = 30 + name_len + extra_len;
        let mut header_rest = vec![0u8; header_len - 30];
        file.read_exact(&mut header_rest).map_err(io)?;
        let local_name = std::str::from_utf8(&header_rest[..name_len])
            .map_err(|_| zip(format!("entry '{name}' has a non-UTF8 local name")))?;
        if local_name != name {
            return Err(zip(format!(
                "entry '{name}' local header names '{local_name}'"
            )));
        }
        let mut bytes = vec![0u8; size as usize];
        file.read_exact(&mut bytes).map_err(io)?;
        out.insert(name, bytes);
    }
    Ok(out)
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only stored-ZIP writer: local headers + central directory + EOCD,
    /// method 0 throughout. Mirrors the real sidecar layout (stored entries).
    fn write_stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        fn u16le(out: &mut Vec<u8>, value: u16) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        fn u32le(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let mut bytes = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let local_offset = bytes.len() as u32;
            bytes.extend_from_slice(b"PK\x03\x04");
            u16le(&mut bytes, 20); // version needed
            u16le(&mut bytes, 0); // flags
            u16le(&mut bytes, 0); // method: stored
            u16le(&mut bytes, 0); // time
            u16le(&mut bytes, 0); // date
            u32le(&mut bytes, 0); // crc (unchecked by the reader)
            u32le(&mut bytes, data.len() as u32);
            u32le(&mut bytes, data.len() as u32);
            u16le(&mut bytes, name.len() as u16);
            u16le(&mut bytes, 0); // extra
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(data);
            central.extend_from_slice(b"PK\x01\x02");
            u16le(&mut central, 20); // version made by
            u16le(&mut central, 20); // version needed
            u16le(&mut central, 0); // flags
            u16le(&mut central, 0); // method: stored
            u16le(&mut central, 0); // time
            u16le(&mut central, 0); // date
            u32le(&mut central, 0); // crc
            u32le(&mut central, data.len() as u32);
            u32le(&mut central, data.len() as u32);
            u16le(&mut central, name.len() as u16);
            u16le(&mut central, 0); // extra
            u16le(&mut central, 0); // comment
            u16le(&mut central, 0); // disk
            u16le(&mut central, 0); // internal attrs
            u32le(&mut central, 0); // external attrs
            u32le(&mut central, local_offset);
            central.extend_from_slice(name.as_bytes());
        }
        let central_offset = bytes.len() as u32;
        bytes.extend_from_slice(&central);
        let central_size = central.len() as u32;
        bytes.extend_from_slice(b"PK\x05\x06");
        u16le(&mut bytes, 0); // disk
        u16le(&mut bytes, 0); // central disk
        u16le(&mut bytes, entries.len() as u16);
        u16le(&mut bytes, entries.len() as u16);
        u32le(&mut bytes, central_size);
        u32le(&mut bytes, central_offset);
        u16le(&mut bytes, 0); // comment
        bytes
    }

    fn manifest_json(pdi: &[u8], insts: &[u8]) -> String {
        serde_json::json!({
            "version": 1,
            "model_sha256": "a".repeat(64),
            "quant": "mq4g256v2",
            "arch": "gfx1151",
            "npu": "npu5",
            "toolchain": {"mlir_aie": "f50bef7", "peano": "22.0.0.2026090701"},
            "profiles": [{
                "tensor_role": "gate_up",
                "M": 1024, "K": 5120, "N": 512, "row_count": 1024,
                "tile_m": 128, "tile_k": 64, "tile_n": 64,
                "cols": 8, "k_mt": 512,
                "pdi": "profiles/gate_up/main.pdi",
                "insts": "profiles/gate_up/insts.bin",
                "arg_layout": [
                    {"name": "a", "offset": 0, "size": 64},
                    {"name": "mask", "offset": 20, "size": 4}
                ],
                "sha256_pdi": sha256_hex(pdi),
                "sha256_insts": sha256_hex(insts)
            }]
        })
        .to_string()
    }

    fn write_sidecar(dir: &Path, name: &str, manifest: &str, pdi: &[u8], insts: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(
            &path,
            write_stored_zip(&[
                ("manifest.json", manifest.as_bytes()),
                ("profiles/gate_up/main.pdi", pdi),
                ("profiles/gate_up/insts.bin", insts),
            ]),
        )
        .unwrap();
        path
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hipfire-xdna-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn valid_sidecar_verifies() {
        let dir = test_dir("valid");
        let pdi = b"fake-pdi-bytes";
        let insts = b"fake-insts-bytes";
        let path = write_sidecar(&dir, "m.xdna.zip", &manifest_json(pdi, insts), pdi, insts);
        let descriptor = XdnaSidecarDescriptor::load_verified(&path).expect("valid sidecar");
        assert!(descriptor.verified);
        assert_eq!(descriptor.path, path);
        assert_eq!(descriptor.manifest.version, 1);
        assert_eq!(descriptor.manifest.arch, "gfx1151");
        assert_eq!(descriptor.manifest.profiles.len(), 1);
        assert_eq!(descriptor.manifest.profiles[0].tensor_role, "gate_up");
        assert!(descriptor.admit_for_arch("gfx1151").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_payload_fails_closed() {
        let dir = test_dir("tamper");
        let pdi = b"fake-pdi-bytes";
        let insts = b"fake-insts-bytes";
        // Manifest binds the ORIGINAL bytes; the archive carries a corrupt PDI.
        let path = write_sidecar(&dir, "m.xdna.zip", &manifest_json(pdi, insts), b"corrupt!", insts);
        let error = XdnaSidecarDescriptor::load_verified(&path).expect_err("tamper must fail");
        let message = format!("{error}");
        assert!(message.contains("hash mismatch"), "{message}");
        assert!(message.contains("profiles/gate_up/main.pdi"), "{message}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_arch_manifest_is_rejected() {
        let dir = test_dir("arch");
        let pdi = b"fake-pdi-bytes";
        let insts = b"fake-insts-bytes";
        let mut manifest: serde_json::Value =
            serde_json::from_str(&manifest_json(pdi, insts)).unwrap();
        manifest["arch"] = serde_json::json!("gfx1201");
        let path = write_sidecar(&dir, "m.xdna.zip", &manifest.to_string(), pdi, insts);
        let error = XdnaSidecarDescriptor::load_verified(&path).expect_err("arch must fail");
        assert!(format!("{error}").contains("gfx1201"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_payload_entry_is_rejected() {
        let dir = test_dir("missing");
        let pdi = b"fake-pdi-bytes";
        let insts = b"fake-insts-bytes";
        let path = dir.join("m.xdna.zip");
        // Archive omits the PDI the manifest binds.
        std::fs::write(
            &path,
            write_stored_zip(&[
                ("manifest.json", manifest_json(pdi, insts).as_bytes()),
                ("profiles/gate_up/insts.bin", insts),
            ]),
        )
        .unwrap();
        let error = XdnaSidecarDescriptor::load_verified(&path).expect_err("missing entry");
        assert!(format!("{error}").contains("main.pdi"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deflated_entry_is_rejected() {
        let dir = test_dir("deflate");
        let pdi = b"fake-pdi-bytes";
        let insts = b"fake-insts-bytes";
        let mut archive = write_stored_zip(&[
            ("manifest.json", manifest_json(pdi, insts).as_bytes()),
            ("profiles/gate_up/main.pdi", pdi),
            ("profiles/gate_up/insts.bin", insts),
        ]);
        // Rewrite the PDI entry's central-directory method to deflate (8) by
        // scanning for central signatures; the reader must reject it.
        let marker = b"profiles/gate_up/main.pdi";
        let mut patched = false;
        let mut cursor = 0usize;
        while cursor + 46 <= archive.len() {
            if &archive[cursor..cursor + 4] == b"PK\x01\x02" {
                let name_len =
                    u16::from_le_bytes([archive[cursor + 28], archive[cursor + 29]]) as usize;
                let start = cursor + 46;
                if start + name_len <= archive.len() && &archive[start..start + name_len] == marker
                {
                    archive[cursor + 10] = 8; // method: deflate
                    archive[cursor + 11] = 0;
                    patched = true;
                }
                let extra_len =
                    u16::from_le_bytes([archive[cursor + 30], archive[cursor + 31]]) as usize;
                let comment_len =
                    u16::from_le_bytes([archive[cursor + 32], archive[cursor + 33]]) as usize;
                cursor += 46 + name_len + extra_len + comment_len;
            } else if &archive[cursor..cursor + 4] == b"PK\x05\x06" {
                break;
            } else {
                cursor += 1;
            }
        }
        assert!(patched, "test setup must patch the method");
        let path = dir.join("m.xdna.zip");
        std::fs::write(&path, &archive).unwrap();
        let error = XdnaSidecarDescriptor::load_verified(&path).expect_err("deflate must fail");
        assert!(format!("{error}").contains("stored"), "got: {error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_gfx1151_host_is_not_admitted() {
        let dir = test_dir("admit");
        let pdi = b"fake-pdi-bytes";
        let insts = b"fake-insts-bytes";
        let path = write_sidecar(&dir, "m.xdna.zip", &manifest_json(pdi, insts), pdi, insts);
        let descriptor = XdnaSidecarDescriptor::load_verified(&path).expect("valid sidecar");
        for arch in ["gfx1100", "gfx1201", "gfx942", ""] {
            let error = descriptor
                .admit_for_arch(arch)
                .expect_err("non-gfx1151 must not admit");
            assert!(error.contains("gfx1151"), "{error}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
