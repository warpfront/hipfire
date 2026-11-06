//! HC8 sidecar: OFFLINE-requantized 8-bit replacements (OCP E4M3, per-output-row
//! power-of-two f32 scale) for the Flash-Next hyper-connection mixer BF16 weights
//! (`*.input_mix_weight_down.weight` / `*.input_mix_weight_up.weight`).
//!
//! The sidecar (`<model>.hc8`) is an HFQM container (written with
//! [`write_hfqm_package_streaming`]) whose `arch_id` is the base's and whose tensors
//! are rank-2 `[m, k]` (the shape of the base tensor they replace), `quant_type`
//! [`HC8_QUANT_TYPE`] (57, `HC8E4M3Row`), `group_size` 32 (the K tile), in base index
//! order, `data_size = hc8_payload_bytes(m, k)`: the swizzled E4M3 bytes `q`
//! (`m * k`) followed by little-endian `f32 s[m]`. Its metadata JSON is exactly one
//! object key, [`HC8_SIDECAR_METADATA_KEY`], holding an [`Hc8Manifest`]. See
//! `docs/quant-formats/hc8-e4m3-row.md`.
//!
//! # Binding
//!
//! [`bind_hc8_sidecar`] is the cheap load-time check (the trunk overlay convention):
//! arch id, base file length, sha256 of the base bytes `[0, data_offset)`
//! ([`HfqFile::index_sha256`]), manifest/index name agreement, and per-tensor
//! agreement with the base (BF16 `quant_type` 16, `data_size`, shape) and with the
//! HC8 payload size. Payload sha256s are verified by [`BoundHc8Sidecar::read_verified`].
//!
//! # API
//!
//! ```text
//! HC8_QUANT_TYPE, HC8_SIDECAR_METADATA_KEY, HC8_SIDECAR_VERSION
//! is_hc8_target(name) -> bool
//! Hc8Manifest / Hc8Base / Hc8Tensor
//! bind_hc8_sidecar(base, path) -> Result<BoundHc8Sidecar, String>
//! BoundHc8Sidecar::{path, manifest, tensor_count, payload_bytes, replaced_bytes, net_bytes, read_verified}
//! Hc8Payload { name, shape, bytes }
//! build_hc8_payload(bf16_le, m, k) -> Result<Vec<u8>, String>
//! write_hc8_sidecar(base_path, output) -> Result<Hc8WriteReport, String>
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry, HfqTensorInfo};
use crate::ple_sidecar::{hex_lower, short_hex};

/// HFQM `quant_type` of every sidecar tensor (`HC8E4M3Row`).
pub const HC8_QUANT_TYPE: u8 = 57;
/// The one metadata key of an HC8 sidecar.
pub const HC8_SIDECAR_METADATA_KEY: &str = "hipfire_hc8_sidecar";
/// The manifest version this build writes and accepts.
pub const HC8_SIDECAR_VERSION: u32 = 1;
/// The manifest `format` string.
pub const HC8_FORMAT: &str = "e4m3_row_pow2";
/// HFQM `group_size` field of a sidecar tensor: the K tile of the swizzle.
const HC8_GROUP_SIZE: u32 = 32;
/// Base quant type of a BF16 tensor.
const BF16_QUANT_TYPE: u8 = 16;
/// Streaming buffer for hashing the base file.
const HASH_BUFFER_BYTES: usize = 8 << 20;

/// Whether `name` is a hyper-connection mixer weight the HC8 sidecar replaces.
pub fn is_hc8_target(name: &str) -> bool {
    name.contains("hyper_connection")
        && (name.ends_with(".input_mix_weight_down.weight")
            || name.ends_with(".input_mix_weight_up.weight"))
}

/// The base file a sidecar binds to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hc8Base {
    pub file_bytes: u64,
    /// sha256 of the base bytes `[0, data_offset)`, lowercase hex.
    pub index_sha256: String,
    /// sha256 of the whole base file, lowercase hex.
    pub sha256: String,
}

/// One sidecar tensor: its payload digest and the base tensor it replaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hc8Tensor {
    pub name: String,
    /// sha256 of the payload bytes, lowercase hex.
    pub sha256: String,
    pub base_quant_type: u8,
    pub base_data_size: u64,
}

/// The `hipfire_hc8_sidecar` metadata object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hc8Manifest {
    pub version: u32,
    /// Always [`HC8_FORMAT`] (`e4m3_row_pow2`).
    pub format: String,
    pub base: Hc8Base,
    pub tensors: Vec<Hc8Tensor>,
}

/// The whole metadata JSON of a sidecar: exactly one object key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Hc8Envelope {
    hipfire_hc8_sidecar: Hc8Manifest,
}

/// One verified sidecar payload, ready for upload: swizzled `q` then f32 LE `s`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hc8Payload {
    pub name: String,
    pub shape: [usize; 2],
    pub bytes: Vec<u8>,
}

/// A sidecar that passed every [`bind_hc8_sidecar`] check against its base.
pub struct BoundHc8Sidecar {
    path: PathBuf,
    sidecar: HfqFile,
    manifest: Hc8Manifest,
}

impl std::fmt::Debug for BoundHc8Sidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundHc8Sidecar")
            .field("path", &self.path)
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

fn payload_len(m: usize, k: usize) -> Option<usize> {
    rdna_compute::hc8_weight::hc8_payload_bytes(m, k)
}

/// Quantize a little-endian BF16 `[m, k]` matrix to the sidecar payload: swizzled
/// E4M3 bytes then f32 LE row scales. The one host quantizer wrapper shared by the
/// builder and the tests.
pub fn build_hc8_payload(bf16_le: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    let Some(total) = payload_len(m, k) else {
        return Err(format!(
            "[{m}, {k}] is not an HC8 shape (rows a positive multiple of 16, K a positive multiple of 32)"
        ));
    };
    if bf16_le.len() != m * k * 2 {
        return Err(format!(
            "BF16 input is {} bytes, [{m}, {k}] is {}",
            bf16_le.len(),
            m * k * 2
        ));
    }
    let bits: Vec<u16> = bf16_le
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let (mut payload, scales) = rdna_compute::hc8::quantize_e4m3_rows(&bits, m, k);
    payload.reserve(4 * m);
    for s in &scales {
        payload.extend_from_slice(&s.to_le_bytes());
    }
    debug_assert_eq!(payload.len(), total);
    Ok(payload)
}

/// Bind the sidecar at `path` to `base`. The error is a one-line refusal reason
/// (no path prefix).
pub fn bind_hc8_sidecar(base: &HfqFile, path: &Path) -> Result<BoundHc8Sidecar, String> {
    // Offset 0 on purpose: `HfqFile::open` would auto-attach a REAP overlay.
    let mut sidecar =
        HfqFile::open_at_offset(path, 0).map_err(|e| format!("cannot open the sidecar: {e}"))?;
    // Payloads are read positionally; nothing needs the mapping after the index.
    sidecar.drop_mmap();

    if sidecar.arch_id != base.arch_id {
        return Err(format!(
            "sidecar is arch {}, base is arch {}",
            sidecar.arch_id, base.arch_id
        ));
    }
    let envelope: Hc8Envelope = serde_json::from_str(&sidecar.metadata_json)
        .map_err(|e| format!("metadata is not a `{HC8_SIDECAR_METADATA_KEY}` object: {e}"))?;
    let manifest = envelope.hipfire_hc8_sidecar;
    if manifest.version != HC8_SIDECAR_VERSION {
        return Err(format!(
            "sidecar version is {}, this build reads {HC8_SIDECAR_VERSION}",
            manifest.version
        ));
    }
    if manifest.format != HC8_FORMAT {
        return Err(format!(
            "sidecar format is {:?}, this build reads {HC8_FORMAT:?}",
            manifest.format
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
            "base is {} bytes, sidecar binds {}",
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
            "base index sha256 {} differs from the sidecar's {}",
            short_hex(&base_index),
            short_hex(&manifest.base.index_sha256)
        ));
    }

    let infos = sidecar.tensors();
    if manifest.tensors.is_empty() {
        return Err("sidecar holds no tensors".to_string());
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
        if !is_hc8_target(&entry.name) {
            return Err(format!("tensor {} is not an HC8 target", entry.name));
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
        if info.quant_type != HC8_QUANT_TYPE || info.group_size != HC8_GROUP_SIZE {
            return Err(format!(
                "tensor {} is quant type {} group {}, not HC8E4M3Row ({HC8_QUANT_TYPE}) group {HC8_GROUP_SIZE}",
                info.name, info.quant_type, info.group_size
            ));
        }
        let Some(expected) = payload_len(m, k) else {
            return Err(format!(
                "tensor {} [{m}, {k}] is not an HC8 shape (rows % 16, K % 32)",
                info.name
            ));
        };
        if info.data_size != expected {
            return Err(format!(
                "tensor {} is {} bytes, HC8 [{m}, {k}] is {expected}",
                info.name, info.data_size
            ));
        }
        let Some(base_info) = base_by_name.get(entry.name.as_str()) else {
            return Err(format!("tensor {} is not in the base", entry.name));
        };
        if base_info.quant_type != BF16_QUANT_TYPE {
            return Err(format!(
                "tensor {}: base quant type is {}, not BF16 ({BF16_QUANT_TYPE})",
                entry.name, base_info.quant_type
            ));
        }
        if base_info.quant_type != entry.base_quant_type {
            return Err(format!(
                "tensor {}: base quant type is {}, sidecar binds {}",
                entry.name, base_info.quant_type, entry.base_quant_type
            ));
        }
        if base_info.data_size as u64 != entry.base_data_size {
            return Err(format!(
                "tensor {}: base holds {} bytes, sidecar binds {}",
                entry.name, base_info.data_size, entry.base_data_size
            ));
        }
        if base_info.shape != info.shape {
            return Err(format!(
                "tensor {}: base shape is {:?}, sidecar shape is {:?}",
                entry.name, base_info.shape, info.shape
            ));
        }
    }

    Ok(BoundHc8Sidecar {
        path: path.to_path_buf(),
        sidecar,
        manifest,
    })
}

impl BoundHc8Sidecar {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn manifest(&self) -> &Hc8Manifest {
        &self.manifest
    }

    pub fn tensor_count(&self) -> usize {
        self.manifest.tensors.len()
    }

    /// Total payload bytes of every sidecar tensor.
    pub fn payload_bytes(&self) -> u64 {
        self.sidecar
            .tensors()
            .iter()
            .map(|t| t.data_size as u64)
            .sum()
    }

    /// Total base (BF16) bytes of the tensors the sidecar replaces.
    pub fn replaced_bytes(&self) -> u64 {
        self.manifest
            .tensors
            .iter()
            .map(|t| t.base_data_size)
            .sum()
    }

    /// Resident device bytes the sidecar adds on top of the base: payload minus
    /// replaced bytes (negative: HC8 is a saving).
    pub fn net_bytes(&self) -> i64 {
        self.payload_bytes() as i64 - self.replaced_bytes() as i64
    }

    /// Read every payload with positional I/O and verify its sha256 against the
    /// manifest. All-or-nothing: the first mismatch names its tensor.
    pub fn read_verified(&self) -> Result<Vec<Hc8Payload>, String> {
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
            payloads.push(Hc8Payload {
                name: entry.name.clone(),
                shape,
                bytes,
            });
        }
        Ok(payloads)
    }
}

/// Base file summary in a [`Hc8WriteReport`].
#[derive(Debug, Clone, Serialize)]
pub struct Hc8ReportBase {
    pub path: String,
    pub file_bytes: u64,
    pub sha256: String,
    pub index_sha256: String,
}

/// What [`write_hc8_sidecar`] produced.
#[derive(Debug, Clone, Serialize)]
pub struct Hc8WriteReport {
    pub output: String,
    pub file_bytes: u64,
    /// sha256 of the written sidecar file.
    pub file_sha256: String,
    pub tensors: usize,
    pub payload_bytes: u64,
    pub replaced_bytes: u64,
    pub base: Hc8ReportBase,
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

/// Build an HC8 sidecar at `output` for the base at `base_path`.
///
/// Selects every [`is_hc8_target`] base tensor stored as BF16 (base index order;
/// none is an error), quantizes it with [`build_hc8_payload`] and writes the HFQM
/// container to `<output>.partial`, fsyncs and renames it, then re-binds and
/// re-reads it as a self-check (the output is deleted if that fails). An existing
/// `output` is refused.
///
/// The payload sha256 sits in the metadata, which precedes the payloads in the HFQM
/// layout, so every payload is built before writing. All payloads are held in RAM
/// (about 330 MB for the whole Flash-Next set: ~200 tensors of ~6.5 MB), which is
/// the simplest way to keep the single quantization pass.
pub fn write_hc8_sidecar(base_path: &Path, output: &Path) -> Result<Hc8WriteReport, String> {
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
        .filter(|t| is_hc8_target(&t.name) && t.quant_type == BF16_QUANT_TYPE)
        .collect();
    if selected.is_empty() {
        return Err("no BF16 hyper-connection mixer tensor in the base".to_string());
    }

    let mut entries = Vec::with_capacity(selected.len());
    let mut manifest_tensors = Vec::with_capacity(selected.len());
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(selected.len());
    let mut payload_total = 0u64;
    let mut replaced_total = 0u64;
    for info in &selected {
        let Some([m, k]) = matrix_shape(info) else {
            return Err(format!(
                "base tensor {} has rank {}, not 2",
                info.name,
                info.shape.len()
            ));
        };
        let Some((_, bf16)) = base.tensor_data_vec(&info.name) else {
            return Err(format!("cannot read base tensor {}", info.name));
        };
        let payload = build_hc8_payload(&bf16, m, k)
            .map_err(|e| format!("base tensor {}: {e}", info.name))?;
        manifest_tensors.push(Hc8Tensor {
            name: info.name.clone(),
            sha256: hex_lower(&Sha256::digest(&payload)),
            base_quant_type: info.quant_type,
            base_data_size: info.data_size as u64,
        });
        entries.push(HfqStreamEntry {
            name: info.name.clone(),
            quant_type: HC8_QUANT_TYPE,
            shape: info.shape.clone(),
            group_size: HC8_GROUP_SIZE,
            data_len: payload.len() as u64,
        });
        payload_total += payload.len() as u64;
        replaced_total += info.data_size as u64;
        payloads.push(payload);
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
    let manifest = Hc8Manifest {
        version: HC8_SIDECAR_VERSION,
        format: HC8_FORMAT.to_string(),
        base: Hc8Base {
            file_bytes: base_bytes,
            index_sha256: index_sha256.clone(),
            sha256: base_sha256.clone(),
        },
        tensors: manifest_tensors,
    };
    let metadata_json = serde_json::to_string(&Hc8Envelope {
        hipfire_hc8_sidecar: manifest,
    })
    .map_err(|e| format!("serialize manifest: {e}"))?;

    let mut partial_name = output.as_os_str().to_os_string();
    partial_name.push(".partial");
    let partial = PathBuf::from(partial_name);
    let written =
        write_hfqm_package_streaming(&partial, base.arch_id, &metadata_json, &entries, |i, w| {
            w.write_all(&payloads[i])
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

    let self_check =
        bind_hc8_sidecar(&base, output).and_then(|bound| bound.read_verified().map(|_| ()));
    if let Err(e) = self_check {
        let _ = std::fs::remove_file(output);
        return Err(format!("self-check of {} failed: {e}", output.display()));
    }
    let (file_sha256, file_bytes) = sha256_file(output)?;

    Ok(Hc8WriteReport {
        output: output.display().to_string(),
        file_bytes,
        file_sha256,
        tensors: entries.len(),
        payload_bytes: payload_total,
        replaced_bytes: replaced_total,
        base: Hc8ReportBase {
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
    use rdna_compute::hc8::{dequant_rowmajor, quantize_e4m3_rows, Hc8Fmt};
    use rdna_compute::hc8_weight::hc8_swizzled_offset;

    const ARCH: u32 = 16;
    const DOWN: &str = "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight";
    const UP: &str = "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight";
    const OTHER: &str = "model.language_model.layers.0.other.weight";

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        /// Roughly unit-scale f32 in [-2, 2).
        fn f(&mut self) -> f32 {
            (self.next() % 4096) as f32 / 1024.0 - 2.0
        }
    }

    fn bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// Row 0 all zero, row 1 tiny, row 2 huge, the rest random.
    fn random_bf16(seed: u64, m: usize, k: usize) -> Vec<u16> {
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(m * k);
        for r in 0..m {
            let mul = match r {
                1 => 1e-20,
                2 => 1e30,
                _ => 1.0,
            };
            for _ in 0..k {
                out.push(if r == 0 { 0 } else { bf16_bits(rng.f() * mul) });
            }
        }
        out
    }

    fn to_le(bits: &[u16]) -> Vec<u8> {
        bits.iter().flat_map(|b| b.to_le_bytes()).collect()
    }

    fn expected_payload(bits: &[u16], m: usize, k: usize) -> Vec<u8> {
        let (mut q, s) = quantize_e4m3_rows(bits, m, k);
        for v in s {
            q.extend_from_slice(&v.to_le_bytes());
        }
        q
    }

    #[test]
    fn target_names() {
        assert!(is_hc8_target(DOWN));
        assert!(is_hc8_target(UP));
        assert!(is_hc8_target(
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"
        ));
        assert!(!is_hc8_target(OTHER));
        assert!(!is_hc8_target("model.layers.0.input_mix_weight_up.weight"));
        assert!(!is_hc8_target(
            "model.layers.0.attn_hyper_connection.input_mix_weight_up.bias"
        ));
    }

    #[test]
    fn quantizer_round_trip_is_exact_bf16_with_pow2_scales() {
        let (m, k) = (32usize, 64usize);
        let bits = random_bf16(7, m, k);
        let payload = build_hc8_payload(&to_le(&bits), m, k).unwrap();
        assert_eq!(payload.len(), m * k + 4 * m);

        let (q_sw, s_bytes) = payload.split_at(m * k);
        let s: Vec<f32> = s_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut q_rm = vec![0u8; m * k];
        for r in 0..m {
            for c in 0..k {
                q_rm[r * k + c] = q_sw[hc8_swizzled_offset(r, c, k)];
            }
        }
        let got = dequant_rowmajor(Hc8Fmt::E4m3, &q_rm, &s, m, k);
        let (q_ref, s_ref) = rdna_compute::hc8::quantize_e4m3_rows_rowmajor(&bits, m, k);
        assert_eq!(s, s_ref);
        assert_eq!(q_rm, q_ref);
        assert_eq!(got, dequant_rowmajor(Hc8Fmt::E4m3, &q_ref, &s_ref, m, k));

        for v in &got {
            assert_eq!(v.to_bits() & 0xffff, 0, "{v} is not exactly a BF16");
        }
        for r in 0..m {
            let amax = bits[r * k..(r + 1) * k]
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16).abs())
                .fold(0.0f32, f32::max);
            if amax == 0.0 {
                assert_eq!(s[r], 1.0, "row {r}");
                continue;
            }
            let e = (amax / 448.0).log2().ceil() as i32;
            assert_eq!(s[r], 2.0f32.powi(e), "row {r}");
            assert_eq!(s[r].to_bits() & 0x007f_ffff, 0, "row {r} scale is not a power of two");
        }
    }

    #[test]
    fn payload_builder_refuses_bad_shapes() {
        assert!(build_hc8_payload(&[0; 2 * 8 * 64], 8, 64).is_err());
        assert!(build_hc8_payload(&[0; 2 * 16 * 48], 16, 48).is_err());
        assert!(build_hc8_payload(&[0; 10], 16, 32).is_err());
    }

    fn mem(name: &str, quant_type: u8, shape: &[u32], data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type,
            shape: shape.to_vec(),
            group_size: 0,
            data,
        }
    }

    struct Fixture {
        dir: tempfile::TempDir,
        base: PathBuf,
        down: Vec<u16>,
        up: Vec<u16>,
    }

    const DOWN_SHAPE: [usize; 2] = [16, 64];
    const UP_SHAPE: [usize; 2] = [64, 32];

    fn fixture_with(down_qt: u8, up_shape: [usize; 2]) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.hfq");
        let down = random_bf16(11, DOWN_SHAPE[0], DOWN_SHAPE[1]);
        let up = random_bf16(23, up_shape[0], up_shape[1]);
        let tensors = vec![
            mem(
                DOWN,
                down_qt,
                &[DOWN_SHAPE[0] as u32, DOWN_SHAPE[1] as u32],
                to_le(&down),
            ),
            mem(OTHER, 13, &[4, 8], vec![5; 4 * 8]),
            mem(UP, 16, &[up_shape[0] as u32, up_shape[1] as u32], to_le(&up)),
        ];
        write_hfqm_package_mem(&base, ARCH, "{}", &tensors).unwrap();
        Fixture {
            dir,
            base,
            down,
            up,
        }
    }

    fn fixture() -> Fixture {
        fixture_with(16, UP_SHAPE)
    }

    fn open(path: &Path) -> HfqFile {
        HfqFile::open_at_offset(path, 0).unwrap()
    }

    #[test]
    fn write_bind_read_round_trip() {
        let fx = fixture();
        let out = fx.dir.path().join("base.hfq.hc8");
        let report = write_hc8_sidecar(&fx.base, &out).unwrap();
        assert_eq!(report.tensors, 2);

        let base = open(&fx.base);
        let bound = bind_hc8_sidecar(&base, &out).unwrap();
        assert_eq!(bound.tensor_count(), 2);
        assert_eq!(bound.path(), out.as_path());
        let payloads = bound.read_verified().unwrap();
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].name, DOWN);
        assert_eq!(payloads[0].shape, DOWN_SHAPE);
        assert_eq!(payloads[0].bytes, expected_payload(&fx.down, 16, 64));
        assert_eq!(payloads[1].name, UP);
        assert_eq!(payloads[1].shape, UP_SHAPE);
        assert_eq!(payloads[1].bytes, expected_payload(&fx.up, 64, 32));

        let manifest = bound.manifest();
        assert_eq!(manifest.version, HC8_SIDECAR_VERSION);
        assert_eq!(manifest.format, HC8_FORMAT);
        assert_eq!(manifest.base.file_bytes, base.file_len());
        assert_eq!(
            manifest.base.index_sha256,
            hex_lower(&base.index_sha256().unwrap())
        );
        let payload_total = (16 * 64 + 4 * 16 + 64 * 32 + 4 * 64) as u64;
        let replaced = (16 * 64 * 2 + 64 * 32 * 2) as u64;
        assert_eq!(bound.payload_bytes(), payload_total);
        assert_eq!(bound.replaced_bytes(), replaced);
        assert_eq!(bound.net_bytes(), payload_total as i64 - replaced as i64);
        assert!(bound.net_bytes() < 0);
        assert!(!out.with_extension("hc8.partial").exists());
    }

    #[test]
    fn existing_output_is_refused() {
        let fx = fixture();
        let out = fx.dir.path().join("o.hc8");
        std::fs::write(&out, b"x").unwrap();
        let err = write_hc8_sidecar(&fx.base, &out).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(std::fs::read(&out).unwrap(), b"x");
    }

    #[test]
    fn builder_refuses_a_base_without_bf16_targets() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("b.hfq");
        write_hfqm_package_mem(&base, ARCH, "{}", &[mem(OTHER, 13, &[4, 8], vec![0; 32])])
            .unwrap();
        let err = write_hc8_sidecar(&base, &dir.path().join("o.hc8")).unwrap_err();
        assert!(err.contains("no BF16 hyper-connection"), "{err}");
    }

    /// Hand-built sidecar over the fixture base; `edit` mutates the manifest.
    fn hand(
        fx: &Fixture,
        arch: u32,
        edit: impl FnOnce(&mut Hc8Manifest, &mut Vec<HfqMemTensor>),
    ) -> PathBuf {
        let base = open(&fx.base);
        let mut tensors = vec![
            mem_hc8(DOWN, &fx.down, DOWN_SHAPE),
            mem_hc8(UP, &fx.up, UP_SHAPE),
        ];
        let mut manifest = Hc8Manifest {
            version: HC8_SIDECAR_VERSION,
            format: HC8_FORMAT.to_string(),
            base: Hc8Base {
                file_bytes: base.file_len(),
                index_sha256: hex_lower(&base.index_sha256().unwrap()),
                sha256: "0".repeat(64),
            },
            tensors: tensors
                .iter()
                .zip([16 * 64 * 2, 64 * 32 * 2])
                .map(|(t, bytes)| Hc8Tensor {
                    name: t.name.clone(),
                    sha256: hex_lower(&Sha256::digest(&t.data)),
                    base_quant_type: 16,
                    base_data_size: bytes,
                })
                .collect(),
        };
        edit(&mut manifest, &mut tensors);
        let json = serde_json::to_string(&Hc8Envelope {
            hipfire_hc8_sidecar: manifest,
        })
        .unwrap();
        let out = fx.dir.path().join(format!("hand-{arch}.hc8"));
        let _ = std::fs::remove_file(&out);
        write_hfqm_package_mem(&out, arch, &json, &tensors).unwrap();
        out
    }

    fn mem_hc8(name: &str, bits: &[u16], shape: [usize; 2]) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: HC8_QUANT_TYPE,
            shape: vec![shape[0] as u32, shape[1] as u32],
            group_size: HC8_GROUP_SIZE,
            data: expected_payload(bits, shape[0], shape[1]),
        }
    }

    #[test]
    fn hand_built_sidecar_binds() {
        let fx = fixture();
        let path = hand(&fx, ARCH, |_, _| {});
        let bound = bind_hc8_sidecar(&open(&fx.base), &path).unwrap();
        assert_eq!(bound.read_verified().unwrap().len(), 2);
    }

    #[test]
    fn refuses_wrong_arch() {
        let fx = fixture();
        let path = hand(&fx, ARCH + 1, |_, _| {});
        let err = bind_hc8_sidecar(&open(&fx.base), &path).unwrap_err();
        assert!(err.contains("arch"), "{err}");
    }

    #[test]
    fn refuses_base_length_change() {
        let fx = fixture();
        let out = fx.dir.path().join("o.hc8");
        write_hc8_sidecar(&fx.base, &out).unwrap();
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&fx.base)
            .unwrap();
        f.write_all(&[0]).unwrap();
        drop(f);
        let err = bind_hc8_sidecar(&open(&fx.base), &out).unwrap_err();
        assert!(err.contains("bytes"), "{err}");
    }

    #[test]
    fn refuses_tampered_payload_sha() {
        let fx = fixture();
        let path = hand(&fx, ARCH, |m, _| m.tensors[1].sha256 = "1".repeat(64));
        let bound = bind_hc8_sidecar(&open(&fx.base), &path).unwrap();
        let err = bound.read_verified().unwrap_err();
        assert!(err.contains("payload sha256"), "{err}");
        assert!(err.contains(UP), "{err}");
    }

    #[test]
    fn refuses_tampered_payload_bytes() {
        let fx = fixture();
        let path = hand(&fx, ARCH, |_, t| t[0].data[3] ^= 0x40);
        let bound = bind_hc8_sidecar(&open(&fx.base), &path).unwrap();
        assert!(bound.read_verified().is_err());
    }

    #[test]
    fn refuses_base_tensor_not_bf16() {
        let fx = fixture_with(3, UP_SHAPE);
        // The base DOWN is now qt 3 (data length unchanged); a sidecar naming it must refuse.
        let path = hand(&fx, ARCH, |_, _| {});
        let err = bind_hc8_sidecar(&open(&fx.base), &path).unwrap_err();
        assert!(err.contains("not BF16"), "{err}");
        // And the builder skips it, writing only the BF16 UP.
        let out = fx.dir.path().join("o.hc8");
        assert_eq!(write_hc8_sidecar(&fx.base, &out).unwrap().tensors, 1);
    }

    #[test]
    fn refuses_shape_mismatch() {
        // Base UP is [32, 64] (same element count as [64, 32]); the sidecar says [64, 32].
        let fx = fixture_with(16, [32, 64]);
        let path = hand(&fx, ARCH, |_, _| {});
        let err = bind_hc8_sidecar(&open(&fx.base), &path).unwrap_err();
        assert!(err.contains("shape"), "{err}");
    }

    #[test]
    fn refuses_wrong_quant_type_and_unknown_manifest_fields() {
        let fx = fixture();
        let path = hand(&fx, ARCH, |_, t| t[0].quant_type = 44);
        let err = bind_hc8_sidecar(&open(&fx.base), &path).unwrap_err();
        assert!(err.contains("quant type"), "{err}");
        let path = hand(&fx, ARCH, |m, _| m.format = "i8_row".into());
        let err = bind_hc8_sidecar(&open(&fx.base), &path).unwrap_err();
        assert!(err.contains("format"), "{err}");
    }
}
