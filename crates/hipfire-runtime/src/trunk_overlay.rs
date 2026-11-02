//! Trunk overlay sidecar: pre-encoded QT44 (or, for the Q8F16 measurement
//! overlay, QT3) tensors that replace a base model's MQ6 tensors at load.
//!
//! The sidecar is an HFQM container (the model-file layout, written with
//! [`write_hfqm_package_streaming`]) whose `arch_id` is the base's and whose
//! tensors are QT44 matrices (`quant_type` 44, `group_size` 256, shape
//! `[m, k]` of the base tensor it replaces, `data_size = m * (k / 256) * 136`)
//! or, when the manifest entry says `"format": "qt3"`, Q8F16 matrices
//! (`quant_type` 3, `group_size` 32, `data_size = m * (k / 32) * 34`: an f16
//! scale and 32 int8 per block, the engine's `DType::Q8_0` rows), in base
//! index order. Its metadata JSON is exactly one object key,
//! [`TRUNK_OVERLAY_METADATA_KEY`], holding a [`TrunkOverlayManifest`]. An
//! entry with no `format` key is QT44, so every sidecar written before the
//! field existed binds and reads unchanged.
//!
//! # Binding
//!
//! [`bind_trunk_overlay`] is the cheap load-time check: arch id, base file
//! length, sha256 of the base bytes `[0, data_offset)` (the PLE sidecar
//! convention, [`HfqFile::index_sha256`]), manifest/index name agreement, and
//! per-tensor agreement with the base (`quant_type`, `data_size`, shape) and
//! with the size formula of the entry's format (QT44 or QT3). The full-file `sha256` of the base is recorded
//! (verified by [`write_trunk_overlay`] at build time and by `hipfire pull` on
//! download) and never recomputed at load. Payload sha256s are verified by
//! [`BoundTrunkOverlay::read_verified`] when the payloads are read for upload.
//!
//! # API
//!
//! ```text
//! TRUNK_OVERLAY_METADATA_KEY, TRUNK_OVERLAY_VERSION
//! TrunkOverlayManifest / TrunkOverlayBase / TrunkOverlayTensor
//! bind_trunk_overlay(base, path) -> Result<BoundTrunkOverlay, String>
//! BoundTrunkOverlay::{path, manifest, marker, tensor_count, payload_bytes, read_verified}
//! TrunkOverlayPayload { name, shape, format, bytes }
//! write_trunk_overlay(base_path, payload_dir, suffixes, marker, format, output)
//!     -> Result<TrunkOverlayWriteReport, String>
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry, HfqTensorInfo};
use crate::ple_sidecar::{hex_lower, short_hex};

/// The one metadata key of a trunk overlay sidecar.
pub const TRUNK_OVERLAY_METADATA_KEY: &str = "hipfire_trunk_overlay";
/// The manifest version this build writes and accepts.
pub const TRUNK_OVERLAY_VERSION: u32 = 1;

/// Quant type byte of QT44 (MQ4G256V2) tensors.
const QT44_QUANT_TYPE: u8 = 44;
/// QT44 group size (columns per group).
const QT44_GROUP_SIZE: u32 = 256;
/// Quant type byte of QT3 (Q8F16, `DType::Q8_0`) tensors.
const QT3_QUANT_TYPE: u8 = 3;
/// QT3 group size (columns per 34-byte block).
const QT3_GROUP_SIZE: u32 = 32;
/// Bytes of one QT3 block: an f16 scale and 32 int8 values.
const QT3_BLOCK_BYTES: usize = 34;

/// The payload format of one overlay tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrunkOverlayFormat {
    /// MQ4G256V2: 136 bytes per 256 weights, `K % 256 == 0`.
    #[default]
    Qt44,
    /// Q8F16 (`DType::Q8_0`): 34 bytes per 32 weights, `K % 32 == 0`.
    Qt3,
}

impl TrunkOverlayFormat {
    /// The HFQM `quant_type` byte the sidecar index carries.
    pub const fn quant_type(self) -> u8 {
        match self {
            Self::Qt44 => QT44_QUANT_TYPE,
            Self::Qt3 => QT3_QUANT_TYPE,
        }
    }

    /// The HFQM `group_size` the sidecar index carries.
    pub const fn group_size(self) -> u32 {
        match self {
            Self::Qt44 => QT44_GROUP_SIZE,
            Self::Qt3 => QT3_GROUP_SIZE,
        }
    }

    /// The name used in refusal messages.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Qt44 => "QT44",
            Self::Qt3 => "QT3",
        }
    }

    /// Exact payload bytes of an `[m, k]` matrix; `None` unless `k` is a
    /// positive multiple of the group size, or on overflow.
    pub fn payload_bytes(self, m: usize, k: usize) -> Option<usize> {
        let group = self.group_size() as usize;
        if k == 0 || k % group != 0 {
            return None;
        }
        let groups = m.checked_mul(k / group)?;
        match self {
            Self::Qt44 => groups.checked_mul(rdna_compute::MQ4V2_GROUP_BYTES),
            Self::Qt3 => groups.checked_mul(QT3_BLOCK_BYTES),
        }
    }

    fn is_qt44(&self) -> bool {
        *self == Self::Qt44
    }
}

/// Streaming buffer for hashing the base file.
const HASH_BUFFER_BYTES: usize = 8 << 20;

/// The base file a sidecar binds to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrunkOverlayBase {
    pub file_bytes: u64,
    /// sha256 of the base bytes `[0, data_offset)`, lowercase hex.
    pub index_sha256: String,
    /// sha256 of the whole base file, lowercase hex.
    pub sha256: String,
}

/// One overlay tensor: its payload digest and the base tensor it replaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrunkOverlayTensor {
    pub name: String,
    /// sha256 of the payload bytes, lowercase hex.
    pub sha256: String,
    pub base_quant_type: u8,
    pub base_data_size: u64,
    /// Payload format; absent (QT44) in every sidecar written before QT3.
    #[serde(default, skip_serializing_if = "TrunkOverlayFormat::is_qt44")]
    pub format: TrunkOverlayFormat,
}

/// The `hipfire_trunk_overlay` metadata object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrunkOverlayManifest {
    pub version: u32,
    pub qwen4_trunk_iu4_default: String,
    pub base: TrunkOverlayBase,
    pub tensors: Vec<TrunkOverlayTensor>,
}

/// The whole metadata JSON of a sidecar: exactly one object key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrunkOverlayEnvelope {
    hipfire_trunk_overlay: TrunkOverlayManifest,
}

/// One verified overlay payload, ready for upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrunkOverlayPayload {
    pub name: String,
    pub shape: [usize; 2],
    pub format: TrunkOverlayFormat,
    pub bytes: Vec<u8>,
}

/// A sidecar that passed every [`bind_trunk_overlay`] check against its base.
pub struct BoundTrunkOverlay {
    path: PathBuf,
    sidecar: HfqFile,
    manifest: TrunkOverlayManifest,
}

impl std::fmt::Debug for BoundTrunkOverlay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundTrunkOverlay")
            .field("path", &self.path)
            .field("marker", &self.manifest.qwen4_trunk_iu4_default)
            .field("tensors", &self.manifest.tensors.len())
            .finish()
    }
}

fn is_hex64_lower(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `[m, k]` of a rank-2 tensor.
fn matrix_shape(info: &HfqTensorInfo) -> Option<[usize; 2]> {
    match info.shape.as_slice() {
        &[m, k] => Some([m as usize, k as usize]),
        _ => None,
    }
}

/// Bind the sidecar at `path` to `base`. See the module docs for the checks.
/// The error is a one-line refusal reason (no path prefix).
pub fn bind_trunk_overlay(base: &HfqFile, path: &Path) -> Result<BoundTrunkOverlay, String> {
    // Offset 0 on purpose: `HfqFile::open` would auto-attach a REAP overlay.
    let mut sidecar =
        HfqFile::open_at_offset(path, 0).map_err(|e| format!("cannot open the sidecar: {e}"))?;
    // Payloads are read positionally; nothing needs the mapping after the index.
    sidecar.drop_mmap();

    if sidecar.arch_id != base.arch_id {
        return Err(format!(
            "overlay is arch {}, base is arch {}",
            sidecar.arch_id, base.arch_id
        ));
    }
    let envelope: TrunkOverlayEnvelope = serde_json::from_str(&sidecar.metadata_json)
        .map_err(|e| format!("metadata is not a `{TRUNK_OVERLAY_METADATA_KEY}` object: {e}"))?;
    let manifest = envelope.hipfire_trunk_overlay;
    if manifest.version != TRUNK_OVERLAY_VERSION {
        return Err(format!(
            "overlay version is {}, this build reads {TRUNK_OVERLAY_VERSION}",
            manifest.version
        ));
    }
    for (what, digest) in [
        ("base.index_sha256", &manifest.base.index_sha256),
        ("base.sha256", &manifest.base.sha256),
    ] {
        if !is_hex64_lower(digest) {
            return Err(format!("{what} is not 64 lowercase hex digits"));
        }
    }
    if manifest.base.file_bytes != base.file_len() {
        return Err(format!(
            "base is {} bytes, overlay binds {}",
            base.file_len(),
            manifest.base.file_bytes
        ));
    }
    let base_index = hex_lower(
        &base
            .index_sha256()
            .map_err(|e| format!("cannot hash the base index: {e}"))?,
    );
    if base_index != manifest.base.index_sha256 {
        return Err(format!(
            "base index sha256 {} differs from the overlay's {}",
            short_hex(&base_index),
            short_hex(&manifest.base.index_sha256)
        ));
    }

    let infos = sidecar.tensors();
    if manifest.tensors.is_empty() {
        return Err("overlay holds no tensors".to_string());
    }
    if manifest.tensors.len() != infos.len() {
        return Err(format!(
            "manifest lists {} tensors, the sidecar index holds {}",
            manifest.tensors.len(),
            infos.len()
        ));
    }
    let base_by_name: HashMap<&str, &HfqTensorInfo> = base
        .tensors()
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    for (i, (entry, info)) in manifest.tensors.iter().zip(infos).enumerate() {
        if entry.name != info.name {
            return Err(format!(
                "manifest tensor {i} is {}, the sidecar index has {}",
                entry.name, info.name
            ));
        }
        if !is_hex64_lower(&entry.sha256) {
            return Err(format!(
                "tensor {}: sha256 is not 64 lowercase hex digits",
                entry.name
            ));
        }
        let Some([m, k]) = matrix_shape(info) else {
            return Err(format!(
                "tensor {} has rank {}, not 2",
                info.name,
                info.shape.len()
            ));
        };
        let format = entry.format;
        if info.quant_type != format.quant_type() || info.group_size != format.group_size() {
            return Err(format!(
                "tensor {} is quant type {} group {}, not {} ({}) group {}",
                info.name,
                info.quant_type,
                info.group_size,
                format.label(),
                format.quant_type(),
                format.group_size()
            ));
        }
        let Some(expected) = format.payload_bytes(m, k) else {
            return Err(format!(
                "tensor {} has K={k}, not a positive multiple of {}",
                info.name,
                format.group_size()
            ));
        };
        if info.data_size != expected {
            return Err(format!(
                "tensor {} is {} bytes, {} [{m}, {k}] is {expected}",
                info.name,
                info.data_size,
                format.label()
            ));
        }
        let Some(base_info) = base_by_name.get(entry.name.as_str()) else {
            return Err(format!("tensor {} is not in the base", entry.name));
        };
        if base_info.quant_type != entry.base_quant_type {
            return Err(format!(
                "tensor {}: base quant type is {}, overlay binds {}",
                entry.name, base_info.quant_type, entry.base_quant_type
            ));
        }
        if base_info.data_size as u64 != entry.base_data_size {
            return Err(format!(
                "tensor {}: base holds {} bytes, overlay binds {}",
                entry.name, base_info.data_size, entry.base_data_size
            ));
        }
        if base_info.shape != info.shape {
            return Err(format!(
                "tensor {}: base shape is {:?}, overlay shape is {:?}",
                entry.name, base_info.shape, info.shape
            ));
        }
    }

    Ok(BoundTrunkOverlay {
        path: path.to_path_buf(),
        sidecar,
        manifest,
    })
}

impl BoundTrunkOverlay {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn manifest(&self) -> &TrunkOverlayManifest {
        &self.manifest
    }

    /// The trunk default marker the overlay arms (`gfx1151-zab-v1`).
    pub fn marker(&self) -> &str {
        &self.manifest.qwen4_trunk_iu4_default
    }

    pub fn tensor_count(&self) -> usize {
        self.manifest.tensors.len()
    }

    /// Total payload bytes of every overlay tensor.
    pub fn payload_bytes(&self) -> u64 {
        self.sidecar
            .tensors()
            .iter()
            .map(|t| t.data_size as u64)
            .sum()
    }

    /// Resident device bytes the overlay adds on top of the base weights:
    /// the payload bytes minus the base bytes of the tensors it replaces
    /// (`base_data_size`). Positive for a Q8F16 (QT3) overlay (34 B per 32
    /// weights replaces the six-bit trunk), negative for the QT44 zab overlay.
    /// Placement budgets that run before the overlay applies do not see this;
    /// a caller sizing free VRAM charges `max(0, net_extra_bytes())`.
    pub fn net_extra_bytes(&self) -> i64 {
        let replaced: u64 = self
            .manifest
            .tensors
            .iter()
            .map(|tensor| tensor.base_data_size)
            .sum();
        self.payload_bytes() as i64 - replaced as i64
    }

    /// Read every payload with positional I/O and verify its sha256 against
    /// the manifest. All-or-nothing: the first mismatch names its tensor.
    pub fn read_verified(&self) -> Result<Vec<TrunkOverlayPayload>, String> {
        let mut payloads = Vec::with_capacity(self.manifest.tensors.len());
        for (entry, info) in self.manifest.tensors.iter().zip(self.sidecar.tensors()) {
            let Some((_, bytes)) = self.sidecar.tensor_data_vec(&entry.name) else {
                return Err(format!(
                    "tensor {}: cannot read its {} payload bytes",
                    entry.name, info.data_size
                ));
            };
            let digest = hex_lower(&Sha256::digest(&bytes));
            if digest != entry.sha256 {
                return Err(format!(
                    "tensor {}: payload sha256 {} differs from the manifest's {}",
                    entry.name,
                    short_hex(&digest),
                    short_hex(&entry.sha256)
                ));
            }
            let Some(shape) = matrix_shape(info) else {
                return Err(format!("tensor {} is not rank 2", entry.name));
            };
            payloads.push(TrunkOverlayPayload {
                name: entry.name.clone(),
                shape,
                format: entry.format,
                bytes,
            });
        }
        Ok(payloads)
    }
}

/// Base file summary in a [`TrunkOverlayWriteReport`].
#[derive(Debug, Clone, Serialize)]
pub struct TrunkOverlayReportBase {
    pub path: String,
    pub file_bytes: u64,
    pub sha256: String,
    pub index_sha256: String,
}

/// What [`write_trunk_overlay`] produced.
#[derive(Debug, Clone, Serialize)]
pub struct TrunkOverlayWriteReport {
    pub output: String,
    pub file_bytes: u64,
    /// sha256 of the written sidecar file.
    pub file_sha256: String,
    pub tensors: usize,
    pub payload_bytes: u64,
    pub marker: String,
    pub base: TrunkOverlayReportBase,
}

/// sha256 of a whole file, streamed through an 8 MiB buffer.
fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUFFER_BYTES];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex_lower(&hasher.finalize()), total))
}

/// Build a trunk overlay sidecar at `output` for the base at `base_path`.
///
/// Selects the base tensors whose names end with any of `suffixes` (base
/// index order; none is an error). Each needs `payload_dir/<name>.bin` of
/// exactly the `format` size (for QT44 the
/// `HIPFIRE_QWEN4_REQUANT=...=qt44:<dir>` layout). All tensors of one sidecar
/// share `format`.
/// An existing `output` is refused; the file is written to
/// `<output>.partial`, fsynced and renamed, then re-bound and re-read as a
/// self-check (the output is deleted if that fails).
pub fn write_trunk_overlay(
    base_path: &Path,
    payload_dir: &Path,
    suffixes: &[String],
    marker: &str,
    format: TrunkOverlayFormat,
    output: &Path,
) -> Result<TrunkOverlayWriteReport, String> {
    if suffixes.is_empty() {
        return Err("at least one tensor name suffix is required".to_string());
    }
    if output.exists() {
        return Err(format!(
            "output {} already exists; refusing to overwrite",
            output.display()
        ));
    }
    let base = HfqFile::open_at_offset(base_path, 0)
        .map_err(|e| format!("open base {}: {e}", base_path.display()))?;

    let selected: Vec<&HfqTensorInfo> = base
        .tensors()
        .iter()
        .filter(|t| suffixes.iter().any(|s| t.name.ends_with(s.as_str())))
        .collect();
    if selected.is_empty() {
        return Err(format!("no base tensor name ends with any of {suffixes:?}"));
    }

    let mut entries = Vec::with_capacity(selected.len());
    let mut manifest_tensors = Vec::with_capacity(selected.len());
    let mut payload_paths = Vec::with_capacity(selected.len());
    let mut payload_total = 0u64;
    for info in &selected {
        let &[m, k] = info.shape.as_slice() else {
            return Err(format!(
                "base tensor {} has rank {}, not 2",
                info.name,
                info.shape.len()
            ));
        };
        let expected = format
            .payload_bytes(m as usize, k as usize)
            .ok_or_else(|| {
                format!(
                    "base tensor {} has K={k}, not a positive multiple of {}",
                    info.name,
                    format.group_size()
                )
            })?;
        let bin = payload_dir.join(format!("{}.bin", info.name));
        let metadata = std::fs::metadata(&bin)
            .map_err(|e| format!("payload for {}: {}: {e}", info.name, bin.display()))?;
        if metadata.len() != expected as u64 {
            return Err(format!(
                "payload {} is {} bytes, {} [{m}, {k}] is {expected}",
                bin.display(),
                metadata.len(),
                format.label()
            ));
        }
        let (sha256, _) = sha256_file(&bin)?;
        manifest_tensors.push(TrunkOverlayTensor {
            name: info.name.clone(),
            sha256,
            base_quant_type: info.quant_type,
            base_data_size: info.data_size as u64,
            format,
        });
        entries.push(HfqStreamEntry {
            name: info.name.clone(),
            quant_type: format.quant_type(),
            shape: info.shape.clone(),
            group_size: format.group_size(),
            data_len: expected as u64,
        });
        payload_total += expected as u64;
        payload_paths.push(bin);
    }

    let index_sha256 = hex_lower(
        &base
            .index_sha256()
            .map_err(|e| format!("hash base index: {e}"))?,
    );
    let (base_sha256, base_bytes) = sha256_file(base_path)?;
    if base_bytes != base.file_len() {
        return Err(format!(
            "base {} changed while hashing: read {base_bytes} bytes, opened {}",
            base_path.display(),
            base.file_len()
        ));
    }
    let manifest = TrunkOverlayManifest {
        version: TRUNK_OVERLAY_VERSION,
        qwen4_trunk_iu4_default: marker.to_string(),
        base: TrunkOverlayBase {
            file_bytes: base_bytes,
            index_sha256: index_sha256.clone(),
            sha256: base_sha256.clone(),
        },
        tensors: manifest_tensors,
    };
    let metadata_json = serde_json::to_string(&TrunkOverlayEnvelope {
        hipfire_trunk_overlay: manifest,
    })
    .map_err(|e| format!("serialize manifest: {e}"))?;

    let mut partial_name = output.as_os_str().to_os_string();
    partial_name.push(".partial");
    let partial = PathBuf::from(partial_name);
    let written =
        write_hfqm_package_streaming(&partial, base.arch_id, &metadata_json, &entries, |i, w| {
            let mut file = File::open(&payload_paths[i])?;
            let copied = std::io::copy(&mut file, w)?;
            if copied != entries[i].data_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "payload {} changed while writing: copied {copied} bytes, expected {}",
                        payload_paths[i].display(),
                        entries[i].data_len
                    ),
                ));
            }
            Ok(())
        })
        .and_then(|()| File::open(&partial)?.sync_all());
    if let Err(e) = written {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("write {}: {e}", partial.display()));
    }
    if let Err(e) = std::fs::rename(&partial, output) {
        let _ = std::fs::remove_file(&partial);
        return Err(format!(
            "rename {} -> {}: {e}",
            partial.display(),
            output.display()
        ));
    }

    let self_check = bind_trunk_overlay(&base, output)
        .and_then(|bound| bound.read_verified().map(|_| bound.payload_bytes()));
    if let Err(e) = self_check {
        let _ = std::fs::remove_file(output);
        return Err(format!("self-check of {} failed: {e}", output.display()));
    }
    let (file_sha256, file_bytes) = sha256_file(output)?;

    Ok(TrunkOverlayWriteReport {
        output: output.display().to_string(),
        file_bytes,
        file_sha256,
        tensors: entries.len(),
        payload_bytes: payload_total,
        marker: marker.to_string(),
        base: TrunkOverlayReportBase {
            path: base_path.display().to_string(),
            file_bytes: base_bytes,
            sha256: base_sha256,
            index_sha256,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hfq::{write_hfqm_package_mem, HfqMemTensor};

    const ARCH: u32 = 16;
    const SUFFIX: &str = ".in_proj_z.weight";

    fn base_tensor(name: &str, quant_type: u8, shape: &[u32], data_len: usize) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: shape.to_vec(),
            group_size: 256,
            data: (0..data_len).map(|i| (i % 251) as u8).collect(),
        }
    }

    fn payload_bytes(seed: u8, m: usize, k: usize) -> Vec<u8> {
        (0..m * (k / 256) * 136)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    struct Fixture {
        dir: tempfile::TempDir,
        base: PathBuf,
        payloads: PathBuf,
        z0: Vec<u8>,
        z1: Vec<u8>,
    }

    /// Base: two overlay targets (MQ6-ish type 15, [4, 256]) plus one other
    /// tensor; payload dir holds the matching `.bin` files.
    fn fixture(metadata: &str) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.hfq");
        let tensors = vec![
            base_tensor("model.layers.0.in_proj_z.weight", 15, &[4, 256], 4 * 192),
            base_tensor("model.layers.0.other.weight", 13, &[4, 256], 4 * 136),
            base_tensor(
                "model.layers.1.in_proj_z.weight",
                15,
                &[2, 512],
                2 * 2 * 192,
            ),
        ];
        write_hfqm_package_mem(&base, ARCH, metadata, &tensors).unwrap();
        let payloads = dir.path().join("payloads");
        std::fs::create_dir(&payloads).unwrap();
        let z0 = payload_bytes(1, 4, 256);
        let z1 = payload_bytes(9, 2, 512);
        std::fs::write(payloads.join("model.layers.0.in_proj_z.weight.bin"), &z0).unwrap();
        std::fs::write(payloads.join("model.layers.1.in_proj_z.weight.bin"), &z1).unwrap();
        Fixture {
            dir,
            base,
            payloads,
            z0,
            z1,
        }
    }

    fn build(fx: &Fixture, name: &str) -> (PathBuf, TrunkOverlayWriteReport) {
        let out = fx.dir.path().join(name);
        let report = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[SUFFIX.to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt44,
            &out,
        )
        .unwrap();
        (out, report)
    }

    fn open(path: &Path) -> HfqFile {
        HfqFile::open_at_offset(path, 0).unwrap()
    }

    /// Sidecar built by hand from a manifest/tensor list, bound against `base`.
    struct Hand {
        manifest: TrunkOverlayManifest,
        tensors: Vec<HfqMemTensor>,
        arch: u32,
    }

    /// A valid hand-built sidecar matching the fixture base.
    fn hand(fx: &Fixture) -> Hand {
        let base = open(&fx.base);
        let index = hex_lower(&base.index_sha256().unwrap());
        let tensors = vec![
            HfqMemTensor {
                name: "model.layers.0.in_proj_z.weight".into(),
                quant_type: 44,
                shape: vec![4, 256],
                group_size: 256,
                data: fx.z0.clone(),
            },
            HfqMemTensor {
                name: "model.layers.1.in_proj_z.weight".into(),
                quant_type: 44,
                shape: vec![2, 512],
                group_size: 256,
                data: fx.z1.clone(),
            },
        ];
        let entries = vec![
            TrunkOverlayTensor {
                name: tensors[0].name.clone(),
                sha256: hex_lower(&Sha256::digest(&fx.z0)),
                base_quant_type: 15,
                base_data_size: 4 * 192,
                format: TrunkOverlayFormat::Qt44,
            },
            TrunkOverlayTensor {
                name: tensors[1].name.clone(),
                sha256: hex_lower(&Sha256::digest(&fx.z1)),
                base_quant_type: 15,
                base_data_size: 2 * 2 * 192,
                format: TrunkOverlayFormat::Qt44,
            },
        ];
        Hand {
            manifest: TrunkOverlayManifest {
                version: 1,
                qwen4_trunk_iu4_default: "gfx1151-zab-v1".into(),
                base: TrunkOverlayBase {
                    file_bytes: base.file_len(),
                    index_sha256: index,
                    sha256: "0".repeat(64),
                },
                tensors: entries,
            },
            tensors,
            arch: ARCH,
        }
    }

    fn write_hand(fx: &Fixture, h: &Hand, name: &str) -> PathBuf {
        let path = fx.dir.path().join(name);
        let json = serde_json::to_string(&TrunkOverlayEnvelope {
            hipfire_trunk_overlay: h.manifest.clone(),
        })
        .unwrap();
        write_hfqm_package_mem(&path, h.arch, &json, &h.tensors).unwrap();
        path
    }

    fn refusal(fx: &Fixture, h: &Hand) -> String {
        let path = write_hand(fx, h, "hand.trunk-overlay");
        let err = bind_trunk_overlay(&open(&fx.base), &path)
            .expect_err("bind must refuse")
            .to_string();
        std::fs::remove_file(&path).unwrap();
        err
    }

    #[test]
    fn round_trip_write_bind_read() {
        let fx = fixture(r#"{"architecture":"qwen4"}"#);
        let (out, report) = build(&fx, "m.trunk-overlay");
        assert_eq!(report.tensors, 2);
        assert_eq!(report.payload_bytes, (fx.z0.len() + fx.z1.len()) as u64);
        assert_eq!(report.marker, "gfx1151-zab-v1");
        assert_eq!(report.file_bytes, std::fs::metadata(&out).unwrap().len());
        assert_eq!(report.file_sha256, sha256_file(&out).unwrap().0);
        assert_eq!(report.base.sha256, sha256_file(&fx.base).unwrap().0);
        assert!(!fx.dir.path().join("m.trunk-overlay.partial").exists());

        let base = open(&fx.base);
        let bound = bind_trunk_overlay(&base, &out).unwrap();
        assert_eq!(bound.marker(), "gfx1151-zab-v1");
        assert_eq!(bound.tensor_count(), 2);
        assert_eq!(bound.payload_bytes(), report.payload_bytes);
        assert_eq!(bound.path(), out);
        assert_eq!(bound.manifest().base.sha256, report.base.sha256);
        let payloads = bound.read_verified().unwrap();
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].name, "model.layers.0.in_proj_z.weight");
        assert_eq!(payloads[0].shape, [4, 256]);
        assert_eq!(payloads[0].bytes, fx.z0);
        assert_eq!(payloads[1].shape, [2, 512]);
        assert_eq!(payloads[1].bytes, fx.z1);
    }

    #[test]
    fn hand_built_sidecar_binds() {
        let fx = fixture("{}");
        let h = hand(&fx);
        let path = write_hand(&fx, &h, "ok.trunk-overlay");
        let bound = bind_trunk_overlay(&open(&fx.base), &path).unwrap();
        assert_eq!(bound.read_verified().unwrap().len(), 2);
    }

    #[test]
    fn refuses_base_length_mismatch() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        h.manifest.base.file_bytes += 1;
        let base_len = open(&fx.base).file_len();
        assert_eq!(
            refusal(&fx, &h),
            format!("base is {base_len} bytes, overlay binds {}", base_len + 1)
        );
    }

    #[test]
    fn refuses_index_sha_mismatch() {
        let fx = fixture(r#"{"a":1}"#);
        let mut h = hand(&fx);
        // Same length, different metadata: the index digest differs.
        let other = fixture(r#"{"a":2}"#);
        h.manifest.base.index_sha256 = hex_lower(&open(&other.base).index_sha256().unwrap());
        assert_eq!(open(&other.base).file_len(), open(&fx.base).file_len());
        let err = refusal(&fx, &h);
        assert!(err.starts_with("base index sha256 "), "{err}");
    }

    #[test]
    fn refuses_arch_mismatch() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        h.arch = 3;
        assert_eq!(refusal(&fx, &h), "overlay is arch 3, base is arch 16");
    }

    #[test]
    fn refuses_tensor_absent_from_base() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        h.manifest.tensors[1].name = "model.layers.9.in_proj_z.weight".into();
        h.tensors[1].name = "model.layers.9.in_proj_z.weight".into();
        assert_eq!(
            refusal(&fx, &h),
            "tensor model.layers.9.in_proj_z.weight is not in the base"
        );
    }

    #[test]
    fn refuses_base_quant_type_and_size_mismatch() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        h.manifest.tensors[0].base_quant_type = 13;
        assert!(refusal(&fx, &h).contains("base quant type is 15, overlay binds 13"));
        let mut h = hand(&fx);
        h.manifest.tensors[0].base_data_size += 1;
        assert!(refusal(&fx, &h).contains("overlay binds 769"));
    }

    #[test]
    fn refuses_shape_mismatch() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        // Same byte count (K=512 [2] vs [4, 256]) but a different shape.
        h.tensors[0].shape = vec![2, 512];
        h.tensors[0].data = payload_bytes(1, 2, 512);
        h.manifest.tensors[0].sha256 = hex_lower(&Sha256::digest(&h.tensors[0].data));
        let err = refusal(&fx, &h);
        assert!(
            err.contains("base shape is [4, 256], overlay shape is [2, 512]"),
            "{err}"
        );
    }

    #[test]
    fn refuses_wrong_quant_type_or_size_formula() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        h.tensors[0].quant_type = 13;
        assert!(refusal(&fx, &h).contains("not QT44"));

        let mut h = hand(&fx);
        h.tensors[0].group_size = 128;
        assert!(refusal(&fx, &h).contains("not QT44"));

        let mut h = hand(&fx);
        h.tensors[0].data.push(0);
        h.manifest.tensors[0].sha256 = hex_lower(&Sha256::digest(&h.tensors[0].data));
        assert!(refusal(&fx, &h).contains("QT44 [4, 256] is 544"));
    }

    #[test]
    fn refuses_rank_not_two() {
        let fx = fixture("{}");
        let mut h = hand(&fx);
        h.tensors[0].shape = vec![4, 1, 256];
        assert!(refusal(&fx, &h).contains("has rank 3, not 2"));
    }

    #[test]
    fn refuses_manifest_index_mismatch() {
        let fx = fixture("{}");
        // Order swapped in the manifest only.
        let mut h = hand(&fx);
        h.manifest.tensors.swap(0, 1);
        assert!(refusal(&fx, &h).contains("manifest tensor 0 is model.layers.1"));
        // Count differs.
        let mut h = hand(&fx);
        h.manifest.tensors.pop();
        assert_eq!(
            refusal(&fx, &h),
            "manifest lists 1 tensors, the sidecar index holds 2"
        );
        // A duplicate name in the sidecar index is refused by the HFQM open.
        let mut h = hand(&fx);
        h.manifest.tensors[1] = h.manifest.tensors[0].clone();
        h.tensors[1] = HfqMemTensor {
            name: h.tensors[0].name.clone(),
            quant_type: 44,
            shape: vec![4, 256],
            group_size: 256,
            data: fx.z0.clone(),
        };
        assert!(refusal(&fx, &h).contains("duplicate"));
    }

    #[test]
    fn refuses_unknown_metadata_shape() {
        let fx = fixture("{}");
        let h = hand(&fx);
        let path = fx.dir.path().join("bad.trunk-overlay");
        let mut value = serde_json::to_value(&TrunkOverlayEnvelope {
            hipfire_trunk_overlay: h.manifest.clone(),
        })
        .unwrap();
        value["extra"] = serde_json::json!(1);
        write_hfqm_package_mem(&path, ARCH, &value.to_string(), &h.tensors).unwrap();
        let err = bind_trunk_overlay(&open(&fx.base), &path).unwrap_err();
        assert!(err.contains("hipfire_trunk_overlay"), "{err}");

        let mut h2 = hand(&fx);
        h2.manifest.version = 2;
        assert!(refusal(&fx, &h2).contains("overlay version is 2"));
    }

    #[test]
    fn corrupted_payload_binds_but_read_verified_names_the_tensor() {
        let fx = fixture("{}");
        let (out, _) = build(&fx, "m.trunk-overlay");
        let sidecar = open(&out);
        let second = sidecar.tensors()[1].clone();
        drop(sidecar);
        let mut bytes = std::fs::read(&out).unwrap();
        bytes[second.data_offset + 5] ^= 0x40;
        std::fs::write(&out, &bytes).unwrap();
        let bound = bind_trunk_overlay(&open(&fx.base), &out).unwrap();
        let err = bound.read_verified().unwrap_err();
        assert!(
            err.starts_with("tensor model.layers.1.in_proj_z.weight: payload sha256 "),
            "{err}"
        );
    }

    #[test]
    fn writer_refuses_existing_output_and_missing_payload() {
        let fx = fixture("{}");
        let (out, _) = build(&fx, "m.trunk-overlay");
        let again = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[SUFFIX.to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt44,
            &out,
        )
        .unwrap_err();
        assert!(again.contains("already exists"), "{again}");

        std::fs::remove_file(fx.payloads.join("model.layers.1.in_proj_z.weight.bin")).unwrap();
        let missing = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[SUFFIX.to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt44,
            &fx.dir.path().join("other.trunk-overlay"),
        )
        .unwrap_err();
        assert!(
            missing.contains("payload for model.layers.1.in_proj_z.weight"),
            "{missing}"
        );
        assert!(!fx.dir.path().join("other.trunk-overlay").exists());
    }

    #[test]
    fn writer_refuses_wrong_size_payload_and_no_match() {
        let fx = fixture("{}");
        std::fs::write(
            fx.payloads.join("model.layers.0.in_proj_z.weight.bin"),
            &fx.z0[1..],
        )
        .unwrap();
        let err = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[SUFFIX.to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt44,
            &fx.dir.path().join("a.trunk-overlay"),
        )
        .unwrap_err();
        assert!(err.contains("QT44 [4, 256] is 544"), "{err}");

        let err = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[".nope".to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt44,
            &fx.dir.path().join("b.trunk-overlay"),
        )
        .unwrap_err();
        assert!(err.contains("no base tensor name ends with"), "{err}");

        let err = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt44,
            &fx.dir.path().join("c.trunk-overlay"),
        )
        .unwrap_err();
        assert!(err.contains("at least one"), "{err}");
    }

    fn qt3_bytes(seed: u8, m: usize, k: usize) -> Vec<u8> {
        (0..m * (k / 32) * 34)
            .map(|i| (i as u8).wrapping_mul(11).wrapping_add(seed))
            .collect()
    }

    /// A valid hand-built sidecar whose two tensors are QT3 (Q8F16).
    fn hand_qt3(fx: &Fixture) -> Hand {
        let mut h = hand(fx);
        let (z0, z1) = (qt3_bytes(3, 4, 256), qt3_bytes(5, 2, 512));
        for (i, (data, shape)) in [(z0, vec![4u32, 256]), (z1, vec![2, 512])]
            .into_iter()
            .enumerate()
        {
            h.manifest.tensors[i].sha256 = hex_lower(&Sha256::digest(&data));
            h.manifest.tensors[i].format = TrunkOverlayFormat::Qt3;
            h.tensors[i].quant_type = 3;
            h.tensors[i].group_size = 32;
            h.tensors[i].shape = shape;
            h.tensors[i].data = data;
        }
        h
    }

    #[test]
    fn qt3_format_parse_and_size_formula() {
        let q3 = TrunkOverlayFormat::Qt3;
        assert_eq!(
            (q3.quant_type(), q3.group_size(), q3.label()),
            (3, 32, "QT3")
        );
        // 34 bytes per 32 weights; K only needs to be a multiple of 32.
        assert_eq!(q3.payload_bytes(4, 256), Some(4 * 8 * 34));
        assert_eq!(q3.payload_bytes(7, 2560), Some(7 * 80 * 34));
        assert_eq!(q3.payload_bytes(3, 96), Some(3 * 3 * 34));
        assert_eq!(q3.payload_bytes(3, 48), None);
        assert_eq!(q3.payload_bytes(3, 0), None);
        assert_eq!(q3.payload_bytes(usize::MAX, 64), None);
        let q44 = TrunkOverlayFormat::Qt44;
        assert_eq!(q44.payload_bytes(4, 256), Some(544));
        assert_eq!(q44.payload_bytes(4, 96), None);
        // JSON spelling and default.
        assert_eq!(serde_json::to_string(&q3).unwrap(), "\"qt3\"");
        let old: TrunkOverlayTensor = serde_json::from_str(
            r#"{"name":"t","sha256":"","base_quant_type":15,"base_data_size":1}"#,
        )
        .unwrap();
        assert_eq!(old.format, TrunkOverlayFormat::Qt44);
        // A QT44 entry serializes with no `format` key (flag-off identity of
        // every sidecar written before QT3); a QT3 entry carries it.
        assert!(!serde_json::to_string(&old).unwrap().contains("format"));
        let mut new = old.clone();
        new.format = q3;
        assert!(serde_json::to_string(&new)
            .unwrap()
            .contains("\"format\":\"qt3\""));
    }

    #[test]
    fn qt3_hand_built_sidecar_binds_and_carries_the_format() {
        let fx = fixture("{}");
        let h = hand_qt3(&fx);
        let path = write_hand(&fx, &h, "q8.trunk-overlay");
        let bound = bind_trunk_overlay(&open(&fx.base), &path).unwrap();
        assert_eq!(bound.payload_bytes(), 2 * 4 * 8 * 34);
        let payloads = bound.read_verified().unwrap();
        assert_eq!(payloads.len(), 2);
        for payload in &payloads {
            assert_eq!(payload.format, TrunkOverlayFormat::Qt3);
        }
        assert_eq!(payloads[0].shape, [4, 256]);
        assert_eq!(payloads[0].bytes, qt3_bytes(3, 4, 256));
        assert_eq!(payloads[1].bytes, qt3_bytes(5, 2, 512));
        // The QT44 sidecar of the same base still reads as QT44.
        let qt44 = write_hand(&fx, &hand(&fx), "q44.trunk-overlay");
        let payloads = bind_trunk_overlay(&open(&fx.base), &qt44)
            .unwrap()
            .read_verified()
            .unwrap();
        assert!(payloads
            .iter()
            .all(|p| p.format == TrunkOverlayFormat::Qt44));
    }

    #[test]
    fn qt3_writer_round_trip() {
        let fx = fixture("{}");
        let (z0, z1) = (qt3_bytes(3, 4, 256), qt3_bytes(5, 2, 512));
        std::fs::write(fx.payloads.join("model.layers.0.in_proj_z.weight.bin"), &z0).unwrap();
        std::fs::write(fx.payloads.join("model.layers.1.in_proj_z.weight.bin"), &z1).unwrap();
        let out = fx.dir.path().join("q8.trunk-overlay");
        let report = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[SUFFIX.to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt3,
            &out,
        )
        .unwrap();
        assert_eq!(report.payload_bytes, (z0.len() + z1.len()) as u64);
        let sidecar = open(&out);
        for info in sidecar.tensors() {
            assert_eq!((info.quant_type, info.group_size), (3, 32));
        }
        let bound = bind_trunk_overlay(&open(&fx.base), &out).unwrap();
        let payloads = bound.read_verified().unwrap();
        assert_eq!(payloads[0].bytes, z0);
        assert_eq!(payloads[1].format, TrunkOverlayFormat::Qt3);
        // The QT44-sized payloads are the wrong size for a QT3 build.
        std::fs::write(
            fx.payloads.join("model.layers.0.in_proj_z.weight.bin"),
            &fx.z0,
        )
        .unwrap();
        let err = write_trunk_overlay(
            &fx.base,
            &fx.payloads,
            &[SUFFIX.to_string()],
            "gfx1151-zab-v1",
            TrunkOverlayFormat::Qt3,
            &fx.dir.path().join("again.trunk-overlay"),
        )
        .unwrap_err();
        assert!(err.contains("QT3 [4, 256] is 1088"), "{err}");
    }

    #[test]
    fn qt3_refusals_name_the_tensor_and_format() {
        let fx = fixture("{}");
        // Wrong size for the QT3 formula (QT44 size under a qt3 entry).
        let mut h = hand_qt3(&fx);
        h.tensors[0].data = fx.z0.clone();
        h.manifest.tensors[0].sha256 = hex_lower(&Sha256::digest(&h.tensors[0].data));
        let err = refusal(&fx, &h);
        assert!(
            err.contains(
                "tensor model.layers.0.in_proj_z.weight is 544 bytes, QT3 [4, 256] is 1088"
            ),
            "{err}"
        );
        // Index says qt44 while the manifest entry says qt3, and the reverse.
        let mut h = hand_qt3(&fx);
        h.tensors[0].quant_type = 44;
        h.tensors[0].group_size = 256;
        let err = refusal(&fx, &h);
        assert!(
            err.contains("is quant type 44 group 256, not QT3 (3) group 32"),
            "{err}"
        );
        let mut h = hand(&fx);
        h.tensors[0].quant_type = 3;
        h.tensors[0].group_size = 32;
        let err = refusal(&fx, &h);
        assert!(
            err.contains("is quant type 3 group 32, not QT44 (44) group 256"),
            "{err}"
        );
        // A group size the format does not carry.
        let mut h = hand_qt3(&fx);
        h.tensors[1].group_size = 256;
        assert!(refusal(&fx, &h).contains("not QT3 (3) group 32"));
        // K not a multiple of 32 (shape differs from the base too, but the
        // K check comes first).
        let mut h = hand_qt3(&fx);
        h.tensors[0].shape = vec![4, 48];
        h.tensors[0].data = vec![0; 4 * 34];
        h.manifest.tensors[0].sha256 = hex_lower(&Sha256::digest(&h.tensors[0].data));
        let err = refusal(&fx, &h);
        assert!(err.contains("K=48, not a positive multiple of 32"), "{err}");
    }
}
