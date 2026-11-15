// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The stock-FP8 MTP head sidecar (`<model>.mtp`) of a Qwen4 Flash-Next model.
//!
//! The stock-FP8 artifact omits the native MTP head.  The sidecar is an HFQM
//! container (arch 16) holding the head's 31 entries (two qt=55 `FP8E4M3B128`
//! stacked expert tables and 29 verbatim BF16 tensors), bound to the base by
//! its recipe id, file length, index sha256 and the source `fp8.sha256` pin.
//!
//! [`bind_stock_mtp_sidecar`] is the one load-time check (every refusal is one
//! line, raised before any allocation proportional to the payload; the payload
//! itself is sha256-verified here, once).  The producer re-runs it on what it
//! wrote, so the writer and the reader cannot disagree.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
use rdna_compute::DType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{Qwen4Config, ARCH_ID};
use crate::weights::{Qwen4Manifest, Qwen4TierRecipe, RECIPE_METADATA_KEY, STOCK_FP8_RECIPE_ID};

/// The one top-level metadata key of a sidecar.
pub const STOCK_MTP_SIDECAR_KEY: &str = "hipfire_qwen4_mtp_sidecar";
/// `format` of the manifest object.
pub const STOCK_MTP_SIDECAR_FORMAT: &str = "stock-fp8-mtp";
/// The only manifest version this build reads.
pub const STOCK_MTP_SIDECAR_VERSION: u32 = 1;
/// Entries of the production sidecar: 2 qt=55 expert tables + 29 BF16 tensors.
pub const STOCK_MTP_SIDECAR_TENSORS: usize = 31;

const QT_BF16: u8 = 16;
const QT_FP8E4M3B128: u8 = 55;
const MTP_PREFIX: &str = "mtp.";
/// Log prefix of every sidecar line.
const LOG_PREFIX: &str = "  qwen4 mtp sidecar: ";

/// `<full model file name>.mtp`: appended to the whole name, never
/// `Path::with_extension` (that spelling is the `params.mtp` of other arches'
/// head formats).
pub fn qwen4_stock_mtp_sidecar_path(model: &Path) -> PathBuf {
    let mut name = model.as_os_str().to_os_string();
    name.push(".mtp");
    PathBuf::from(name)
}

/// The base the sidecar extends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StockMtpBaseBinding {
    pub recipe_id: String,
    pub file_bytes: u64,
    pub index_sha256: String,
    pub sha256: String,
}

/// The source checkpoint pin the base was converted from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StockMtpSourceBinding {
    pub manifest_sha256: String,
    pub config_sha256: String,
}

/// One sidecar entry's payload digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StockMtpTensorDigest {
    pub name: String,
    pub sha256: String,
}

/// The `hipfire_qwen4_mtp_sidecar` object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StockMtpSidecarManifest {
    pub version: u32,
    pub format: String,
    pub base: StockMtpBaseBinding,
    pub source: StockMtpSourceBinding,
    pub tensors: Vec<StockMtpTensorDigest>,
}

/// The whole metadata JSON object: exactly one key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StockMtpSidecarEnvelope {
    pub hipfire_qwen4_mtp_sidecar: StockMtpSidecarManifest,
}

/// The sidecar entries the stock-FP8 manifest declares for a headed `config`,
/// in manifest order: every `mtp.*` weight except the tied embedding alias.
/// `config` is the headed source config (profile `FlashNext`).
pub fn expected_stock_mtp_entries(config: &Qwen4Config) -> Result<Vec<WeightEntry>, String> {
    let config = config.clone().into_stock_mtp()?;
    let manifest = Qwen4Manifest::build_with_recipe(&config, Qwen4TierRecipe::StockFp8)
        .map_err(|error| format!("qwen4: stock-FP8 MTP manifest: {error}"))?;
    Ok(manifest
        .weights
        .into_iter()
        .filter(|entry| {
            entry.name.starts_with(MTP_PREFIX) && !matches!(entry.policy, ShardPolicy::Tied { .. })
        })
        .collect())
}

/// A sidecar that passed every [`bind_stock_mtp_sidecar`] check.  The mmap is
/// kept: payloads are served `Borrowed` from it.
pub struct BoundStockMtpSidecar {
    pub path: PathBuf,
    pub hfq: HfqFile,
    pub manifest: StockMtpSidecarManifest,
    /// sha256 of the sidecar's own index bytes (`HfqFile::index_sha256`).
    pub index_sha256: [u8; 32],
    /// Sum of every entry's `data_size`.
    pub payload_bytes: u64,
}

impl std::fmt::Debug for BoundStockMtpSidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundStockMtpSidecar")
            .field("path", &self.path)
            .field("tensors", &self.manifest.tensors.len())
            .field("payload_bytes", &self.payload_bytes)
            .finish()
    }
}

/// Two-file metadata view of a headed stock-FP8 model: the total routing rule
/// is `name.starts_with("mtp.")` ⇒ the sidecar, every other name ⇒ the base.
#[derive(Clone, Copy)]
pub struct Qwen4StockSources<'a> {
    pub base: &'a HfqFile,
    pub mtp: Option<&'a HfqFile>,
}

impl Qwen4StockSources<'_> {
    /// The index record of `name` in whichever file holds it.
    pub fn find_tensor_info(&self, name: &str) -> Option<&HfqTensorInfo> {
        match self.mtp {
            Some(mtp) if name.starts_with(MTP_PREFIX) => mtp.find_tensor_info(name),
            _ => self.base.find_tensor_info(name),
        }
    }
}

/// `  qwen4 mtp sidecar: attached <path> (31 tensors, N payload bytes, every
/// payload sha256-verified; base <12> index <12>, source fp8.sha256 <12>)`.
pub fn stock_mtp_attached_line(b: &BoundStockMtpSidecar) -> String {
    format!(
        "{LOG_PREFIX}attached {} ({} tensors, {} payload bytes, every payload sha256-verified; \
         base {} index {}, source fp8.sha256 {})",
        b.path.display(),
        b.manifest.tensors.len(),
        b.payload_bytes,
        short_hex(&b.manifest.base.sha256),
        short_hex(&b.manifest.base.index_sha256),
        short_hex(&b.manifest.source.manifest_sha256),
    )
}

/// `  qwen4 mtp sidecar: REFUSED <path>: <reason>; AR only`.
pub fn stock_mtp_refusal_line(path: &Path, reason: &str) -> String {
    format!("{LOG_PREFIX}REFUSED {}: {reason}; AR only", path.display())
}

/// The first 12 characters of a digest.
pub fn short_hex(text: &str) -> &str {
    text.get(..12).unwrap_or(text)
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 15)] as char);
    }
    out
}

fn is_hex64_lower(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// What the stock-FP8 recipe declares for one sidecar entry: HFQ quant type,
/// group size, shape and exact payload bytes.
struct ExpectedRecord {
    qt: u8,
    group: u32,
    shape: Vec<u32>,
    bytes: u64,
    format: String,
}

fn expected_record(entry: &WeightEntry) -> Result<ExpectedRecord, String> {
    let shape = entry
        .logical_shape
        .iter()
        .map(|&dimension| {
            u32::try_from(dimension)
                .map_err(|_| format!("manifest dimension of {} does not fit HFQ", entry.name))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let format = format!("{:?}", entry.dtype);
    let (qt, group, bytes) = match entry.dtype {
        DType::BF16 => (
            QT_BF16,
            0u32,
            entry.logical_shape.iter().product::<usize>() * 2,
        ),
        DType::FP8E4M3B128 => {
            let [experts, rows, k] = entry.logical_shape[..] else {
                return Err(format!(
                    "manifest shape of {} is not [experts, rows, K]",
                    entry.name
                ));
            };
            let bytes = rdna_compute::fp8_e4m3::b128_tensor_bytes(experts, rows, k)
                .ok_or_else(|| format!("manifest shape of {} has no exact extent", entry.name))?;
            (QT_FP8E4M3B128, 128u32, bytes)
        }
        other => {
            return Err(format!(
                "manifest declares {} as {other:?}, which a stock-FP8 MTP sidecar cannot hold",
                entry.name
            ))
        }
    };
    Ok(ExpectedRecord {
        qt,
        group,
        shape,
        bytes: bytes as u64,
        format,
    })
}

/// §4.2 checks 1-16, in order.  The error is one line without a path prefix.
/// Opens with `HfqFile::open_at_offset(path, 0)` (never `open`: that
/// auto-attaches a REAP overlay).  The sidecar's expected entries come from the
/// base's own config.
pub fn bind_stock_mtp_sidecar(base: &HfqFile, path: &Path) -> Result<BoundStockMtpSidecar, String> {
    bind_against(base, path, || {
        let config = crate::artifact::parse_artifact_config(&base.metadata_json)
            .map_err(|error| format!("base config: {error}"))?;
        expected_stock_mtp_entries(&config)
    })
}

/// The check sequence over an explicit expected entry list (production: the 31
/// entries of the headed stock manifest; the unit tests pass a tiny list).
fn bind_against(
    base: &HfqFile,
    path: &Path,
    expected: impl FnOnce() -> Result<Vec<WeightEntry>, String>,
) -> Result<BoundStockMtpSidecar, String> {
    // 1
    let hfq = HfqFile::open_at_offset(path, 0)
        .map_err(|error| format!("cannot open the sidecar: {error}"))?;
    // 2
    if hfq.arch_id != base.arch_id {
        return Err(format!(
            "sidecar is arch {}, base is arch {}",
            hfq.arch_id, base.arch_id
        ));
    }
    // 3
    let envelope: StockMtpSidecarEnvelope = serde_json::from_str(&hfq.metadata_json)
        .map_err(|error| format!("metadata is not a `{STOCK_MTP_SIDECAR_KEY}` object: {error}"))?;
    let manifest = envelope.hipfire_qwen4_mtp_sidecar;
    // 4
    if manifest.version != STOCK_MTP_SIDECAR_VERSION {
        return Err(format!(
            "sidecar version is {}, this build reads {STOCK_MTP_SIDECAR_VERSION}",
            manifest.version
        ));
    }
    // 5
    if manifest.format != STOCK_MTP_SIDECAR_FORMAT {
        return Err(format!(
            "sidecar format is {:?}, this build reads {STOCK_MTP_SIDECAR_FORMAT:?}",
            manifest.format
        ));
    }
    // 6
    let base_meta: serde_json::Value = serde_json::from_str(&base.metadata_json)
        .map_err(|error| format!("base metadata is not JSON: {error}"))?;
    let recipe = base_meta.get(RECIPE_METADATA_KEY);
    let base_recipe_id = recipe
        .and_then(|recipe| recipe.get("id"))
        .and_then(|id| id.as_str());
    if base_recipe_id != Some(STOCK_FP8_RECIPE_ID) || manifest.base.recipe_id != STOCK_FP8_RECIPE_ID
    {
        return Err(format!(
            "base recipe is {:?}, the sidecar extends {STOCK_FP8_RECIPE_ID:?}",
            base_recipe_id.unwrap_or("<absent>")
        ));
    }
    // 7
    let base_source = recipe
        .and_then(|recipe| recipe.get("source_manifest_sha256"))
        .and_then(|sha| sha.as_str())
        .unwrap_or("<absent>");
    if base_source != manifest.source.manifest_sha256 {
        return Err(format!(
            "base was converted from source manifest {base_source}, the sidecar from {}",
            manifest.source.manifest_sha256
        ));
    }
    // 8
    if base.file_len() != manifest.base.file_bytes {
        return Err(format!(
            "base is {} bytes, sidecar binds {}",
            base.file_len(),
            manifest.base.file_bytes
        ));
    }
    // 9
    let base_index = hex_lower(
        &base
            .index_sha256()
            .map_err(|error| format!("cannot hash the base index: {error}"))?,
    );
    if base_index != manifest.base.index_sha256 {
        return Err(format!(
            "base index sha256 {} differs from the sidecar's {}",
            short_hex(&base_index),
            short_hex(&manifest.base.index_sha256)
        ));
    }
    // 10
    for (what, digest) in [
        ("base.index_sha256", &manifest.base.index_sha256),
        ("base.sha256", &manifest.base.sha256),
        ("source.manifest_sha256", &manifest.source.manifest_sha256),
        ("source.config_sha256", &manifest.source.config_sha256),
    ] {
        if !is_hex64_lower(digest) {
            return Err(format!("{what} is not 64 lowercase hex digits"));
        }
    }
    for entry in &manifest.tensors {
        if !is_hex64_lower(&entry.sha256) {
            return Err(format!(
                "tensor {} sha256 is not 64 lowercase hex digits",
                entry.name
            ));
        }
    }
    // 11
    let expected = expected()?;
    let index = hfq.tensors();
    if index.len() != expected.len() || manifest.tensors.len() != expected.len() {
        return Err(format!(
            "sidecar holds {} tensors, the stock-FP8 MTP head declares {}",
            index.len().max(manifest.tensors.len()),
            expected.len()
        ));
    }
    // 12
    for (position, (info, want)) in index.iter().zip(&expected).enumerate() {
        if info.name != want.name || manifest.tensors[position].name != want.name {
            return Err(format!(
                "sidecar tensor {position} is {}, the manifest's is {}",
                info.name, want.name
            ));
        }
    }
    // 13
    for info in index {
        if base.find_tensor_info(&info.name).is_some() {
            return Err(format!("tensor {} is also in the base", info.name));
        }
    }
    if let Some(stray) = base
        .tensors()
        .iter()
        .find(|info| info.name.starts_with(MTP_PREFIX))
    {
        return Err(format!(
            "base holds {}; the stock-FP8 base carries no {MTP_PREFIX}* records",
            stray.name
        ));
    }
    // 14, 15
    let mut records = Vec::with_capacity(expected.len());
    for want in &expected {
        records.push(expected_record(want)?);
    }
    for (info, record) in index.iter().zip(&records) {
        if info.quant_type != record.qt
            || info.group_size != record.group
            || info.shape != record.shape
        {
            return Err(format!(
                "tensor {} is quant type {} group {} shape {:?}, the manifest declares {} group {} shape {:?}",
                info.name,
                info.quant_type,
                info.group_size,
                info.shape,
                record.qt,
                record.group,
                record.shape
            ));
        }
        if info.data_size as u64 != record.bytes {
            return Err(format!(
                "tensor {} is {} bytes, {} {:?} is {}",
                info.name, info.data_size, record.format, record.shape, record.bytes
            ));
        }
    }
    // 16
    let mut payloads = Vec::with_capacity(index.len());
    for info in index {
        let Some((_, bytes)) = hfq.tensor_data(&info.name) else {
            return Err(format!(
                "tensor {}: cannot read its {} payload bytes",
                info.name, info.data_size
            ));
        };
        payloads.push(bytes);
    }
    let digests = hash_payloads(&payloads);
    for (digest, entry) in digests.iter().zip(&manifest.tensors) {
        let digest = hex_lower(digest);
        if digest != entry.sha256 {
            return Err(format!(
                "tensor {}: payload sha256 {} differs from the manifest's {}",
                entry.name,
                short_hex(&digest),
                short_hex(&entry.sha256)
            ));
        }
    }
    let sidecar_index = hfq
        .index_sha256()
        .map_err(|error| format!("cannot hash the sidecar index: {error}"))?;
    let payload_bytes = index.iter().map(|info| info.data_size as u64).sum();
    debug_assert_eq!(hfq.arch_id, ARCH_ID);
    Ok(BoundStockMtpSidecar {
        path: path.to_path_buf(),
        hfq,
        manifest,
        index_sha256: sidecar_index,
        payload_bytes,
    })
}

/// sha256 of each payload, in order, over every available core.
fn hash_payloads(payloads: &[&[u8]]) -> Vec<[u8; 32]> {
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(payloads.len())
        .max(1);
    let mut done: Vec<(usize, [u8; 32])> = Vec::with_capacity(payloads.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut local: Vec<(usize, [u8; 32])> = Vec::new();
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(bytes) = payloads.get(at) else { break };
                        local.push((at, Sha256::digest(bytes).into()));
                    }
                    local
                })
            })
            .collect();
        for handle in handles {
            done.extend(handle.join().expect("sidecar hash worker panicked"));
        }
    });
    done.sort_by_key(|(at, _)| *at);
    done.into_iter().map(|(_, digest)| digest).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use hipfire_runtime::weight_manifest::DTypeConstraint;

    fn source_pin() -> String {
        format!("2f9001c5f2bb{}", "0".repeat(52))
    }

    fn sha_hex(bytes: &[u8]) -> String {
        hex_lower(&Sha256::digest(bytes))
    }

    /// Per-test scratch directory (removed on drop).
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-qwen4-mtp-sidecar-{name}-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self, file: &str) -> PathBuf {
            self.0.join(file)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn base_meta(recipe_id: &str, source: &str) -> String {
        let mut value = serde_json::Map::new();
        value.insert(
            RECIPE_METADATA_KEY.to_string(),
            serde_json::json!({"id": recipe_id, "source_manifest_sha256": source}),
        );
        serde_json::Value::Object(value).to_string()
    }

    fn bf16(name: &str, shape: Vec<u32>, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: QT_BF16,
            shape,
            group_size: 0,
            data,
        }
    }

    fn write_base(path: &Path, meta: &str, arch: u32, tensors: &[HfqMemTensor]) {
        write_hfqm_package_mem(path, arch, meta, tensors).unwrap();
    }

    /// The tiny expected list: one BF16 `[2,2]` and one qt=55 `[2,128,128]`.
    fn tiny_expected() -> Vec<WeightEntry> {
        vec![
            WeightEntry::model_with_dtype_constraint(
                "mtp.a.weight",
                vec![2, 2],
                DType::BF16,
                DTypeConstraint::source_exact(DType::BF16),
                ShardPolicy::Replicate,
            ),
            WeightEntry::model_with_dtype_constraint(
                "mtp.layers.0.mlp.experts.gate_up_proj",
                vec![2, 128, 128],
                DType::FP8E4M3B128,
                DTypeConstraint::source_exact(DType::FP8E4M3B128),
                ShardPolicy::Replicate,
            ),
        ]
    }

    fn tiny_tensors() -> Vec<HfqMemTensor> {
        // qt=55 [2,128,128]: 2 blocks of round_up(128*128 + 2, 256) = 16640 bytes.
        vec![
            bf16("mtp.a.weight", vec![2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]),
            HfqMemTensor {
                name: "mtp.layers.0.mlp.experts.gate_up_proj".to_string(),
                quant_type: QT_FP8E4M3B128,
                shape: vec![2, 128, 128],
                group_size: 128,
                data: (0..33_280u32).map(|i| (i % 251) as u8).collect(),
            },
        ]
    }

    struct Fixture {
        _scratch: Scratch,
        base: PathBuf,
        side: PathBuf,
        base_tensors: Vec<HfqMemTensor>,
        base_meta: String,
        envelope: StockMtpSidecarEnvelope,
        tensors: Vec<HfqMemTensor>,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let scratch = Scratch::new(name);
            let base = scratch.path("model.hfq");
            let side = qwen4_stock_mtp_sidecar_path(&base);
            let base_tensors = vec![bf16("lm_head.weight", vec![2], vec![9, 9, 9, 9])];
            let base_meta = base_meta(STOCK_FP8_RECIPE_ID, &source_pin());
            write_base(&base, &base_meta, ARCH_ID, &base_tensors);
            let hfq = HfqFile::open_at_offset(&base, 0).unwrap();
            let tensors = tiny_tensors();
            let envelope = StockMtpSidecarEnvelope {
                hipfire_qwen4_mtp_sidecar: StockMtpSidecarManifest {
                    version: STOCK_MTP_SIDECAR_VERSION,
                    format: STOCK_MTP_SIDECAR_FORMAT.to_string(),
                    base: StockMtpBaseBinding {
                        recipe_id: STOCK_FP8_RECIPE_ID.to_string(),
                        file_bytes: hfq.file_len(),
                        index_sha256: hex_lower(&hfq.index_sha256().unwrap()),
                        sha256: "0".repeat(64),
                    },
                    source: StockMtpSourceBinding {
                        manifest_sha256: source_pin(),
                        config_sha256: "1".repeat(64),
                    },
                    tensors: tensors
                        .iter()
                        .map(|tensor| StockMtpTensorDigest {
                            name: tensor.name.clone(),
                            sha256: sha_hex(&tensor.data),
                        })
                        .collect(),
                },
            };
            Self {
                _scratch: scratch,
                base,
                side,
                base_tensors,
                base_meta,
                envelope,
                tensors,
            }
        }

        fn write_side(&self) {
            let meta = serde_json::to_string(&self.envelope).unwrap();
            write_hfqm_package_mem(&self.side, ARCH_ID, &meta, &self.tensors).unwrap();
        }

        fn bind(&self) -> Result<BoundStockMtpSidecar, String> {
            let base = HfqFile::open_at_offset(&self.base, 0).unwrap();
            bind_against(&base, &self.side, || Ok(tiny_expected()))
        }

        fn refused(&self, needle: &str) {
            let error = self.bind().expect_err(needle);
            assert!(error.contains(needle), "{needle:?} missing from: {error}");
        }
    }

    #[test]
    fn u5_path_appends_to_the_full_file_name() {
        assert_eq!(
            qwen4_stock_mtp_sidecar_path(Path::new("/m/qwen3.8-flash-next.stock-fp8.hfq")),
            PathBuf::from("/m/qwen3.8-flash-next.stock-fp8.hfq.mtp")
        );
    }

    #[test]
    fn u5_envelope_is_strict() {
        let f = Fixture::new("envelope");
        let json = serde_json::to_string(&f.envelope).unwrap();
        let round: StockMtpSidecarEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(round, f.envelope);
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["extra"] = serde_json::json!(1);
        assert!(serde_json::from_value::<StockMtpSidecarEnvelope>(value).is_err());
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value[STOCK_MTP_SIDECAR_KEY]["base"]["surprise"] = serde_json::json!(1);
        assert!(serde_json::from_value::<StockMtpSidecarEnvelope>(value).is_err());
    }

    #[test]
    fn u5_binds_a_good_sidecar_and_logs_the_proof_line() {
        let f = Fixture::new("good");
        f.write_side();
        let bound = f.bind().expect("binds");
        assert_eq!(bound.payload_bytes, 8 + 33_280);
        assert_eq!(bound.manifest.tensors.len(), 2);
        let line = stock_mtp_attached_line(&bound);
        assert!(line.starts_with("  qwen4 mtp sidecar: attached "), "{line}");
        assert!(line.contains("(2 tensors, 33288 payload bytes, every payload sha256-verified; base 000000000000 index "));
        assert!(line.ends_with(&format!("source fp8.sha256 {})", short_hex(&source_pin()))));
        assert_eq!(
            stock_mtp_refusal_line(Path::new("/m/x.mtp"), "why"),
            "  qwen4 mtp sidecar: REFUSED /m/x.mtp: why; AR only"
        );
        let tensors = Qwen4StockSources {
            base: &HfqFile::open_at_offset(&f.base, 0).unwrap(),
            mtp: Some(&bound.hfq),
        };
        assert!(tensors.find_tensor_info("mtp.a.weight").is_some());
        assert!(tensors.find_tensor_info("lm_head.weight").is_some());
        assert!(tensors.find_tensor_info("mtp.nope").is_none());
    }

    #[test]
    fn u5_reason_1_cannot_open() {
        let f = Fixture::new("r1");
        f.refused("cannot open the sidecar: ");
    }

    #[test]
    fn u5_reason_2_arch() {
        let f = Fixture::new("r2");
        let meta = serde_json::to_string(&f.envelope).unwrap();
        write_hfqm_package_mem(&f.side, ARCH_ID + 1, &meta, &f.tensors).unwrap();
        f.refused(&format!(
            "sidecar is arch {}, base is arch {ARCH_ID}",
            ARCH_ID + 1
        ));
    }

    #[test]
    fn u5_reason_3_not_an_envelope() {
        let f = Fixture::new("r3");
        write_hfqm_package_mem(&f.side, ARCH_ID, "{\"other\":1}", &f.tensors).unwrap();
        f.refused("metadata is not a `hipfire_qwen4_mtp_sidecar` object: ");
    }

    #[test]
    fn u5_reason_4_version() {
        let mut f = Fixture::new("r4");
        f.envelope.hipfire_qwen4_mtp_sidecar.version = 2;
        f.write_side();
        f.refused("sidecar version is 2, this build reads 1");
    }

    #[test]
    fn u5_reason_5_format() {
        let mut f = Fixture::new("r5");
        f.envelope.hipfire_qwen4_mtp_sidecar.format = "other".into();
        f.write_side();
        f.refused("sidecar format is \"other\", this build reads \"stock-fp8-mtp\"");
    }

    #[test]
    fn u5_reason_6_base_recipe() {
        let f = Fixture::new("r6");
        f.write_side();
        // A packed base (no recipe id) and a different recipe id both refuse.
        write_base(&f.base, "{}", ARCH_ID, &f.base_tensors);
        f.refused("base recipe is \"<absent>\", the sidecar extends \"stock-fp8\"");
        write_base(
            &f.base,
            &base_meta("other", &source_pin()),
            ARCH_ID,
            &f.base_tensors,
        );
        f.refused("base recipe is \"other\", the sidecar extends \"stock-fp8\"");
    }

    #[test]
    fn u5_reason_7_source_manifest() {
        let f = Fixture::new("r7");
        f.write_side();
        let other = "a".repeat(64);
        write_base(
            &f.base,
            &base_meta(STOCK_FP8_RECIPE_ID, &other),
            ARCH_ID,
            &f.base_tensors,
        );
        f.refused(&format!(
            "base was converted from source manifest {other}, the sidecar from {}",
            source_pin()
        ));
    }

    #[test]
    fn u5_reason_8_base_length() {
        let mut f = Fixture::new("r8");
        let real = f.envelope.hipfire_qwen4_mtp_sidecar.base.file_bytes;
        f.envelope.hipfire_qwen4_mtp_sidecar.base.file_bytes = real + 1;
        f.write_side();
        f.refused(&format!("base is {real} bytes, sidecar binds {}", real + 1));
    }

    #[test]
    fn u5_reason_9_base_index() {
        let mut f = Fixture::new("r9");
        f.envelope.hipfire_qwen4_mtp_sidecar.base.index_sha256 = "b".repeat(64);
        f.write_side();
        f.refused("differs from the sidecar's bbbbbbbbbbbb");
    }

    #[test]
    fn u5_reason_10_not_hex() {
        let mut f = Fixture::new("r10");
        f.envelope.hipfire_qwen4_mtp_sidecar.base.sha256 = "ABC".into();
        f.write_side();
        f.refused("base.sha256 is not 64 lowercase hex digits");
        let mut f = Fixture::new("r10b");
        f.envelope.hipfire_qwen4_mtp_sidecar.tensors[0].sha256 = "z".repeat(64);
        f.write_side();
        f.refused("is not 64 lowercase hex digits");
    }

    #[test]
    fn u5_reason_11_tensor_count() {
        let mut f = Fixture::new("r11");
        f.tensors.pop();
        f.envelope.hipfire_qwen4_mtp_sidecar.tensors.pop();
        f.write_side();
        f.refused("sidecar holds 1 tensors, the stock-FP8 MTP head declares 2");
    }

    #[test]
    fn u5_reason_12_tensor_order() {
        let mut f = Fixture::new("r12");
        f.tensors.swap(0, 1);
        f.envelope.hipfire_qwen4_mtp_sidecar.tensors.swap(0, 1);
        f.write_side();
        f.refused(
            "sidecar tensor 0 is mtp.layers.0.mlp.experts.gate_up_proj, the manifest's is mtp.a.weight",
        );
    }

    #[test]
    fn u5_reason_13_name_in_base() {
        let mut f = Fixture::new("r13");
        f.base_tensors
            .push(bf16("mtp.a.weight", vec![2, 2], vec![0; 8]));
        write_base(&f.base, &f.base_meta, ARCH_ID, &f.base_tensors);
        // The sidecar binds the rewritten base's length and index.
        let base = HfqFile::open_at_offset(&f.base, 0).unwrap();
        f.envelope.hipfire_qwen4_mtp_sidecar.base.file_bytes = base.file_len();
        f.envelope.hipfire_qwen4_mtp_sidecar.base.index_sha256 =
            hex_lower(&base.index_sha256().unwrap());
        drop(base);
        f.write_side();
        f.refused("tensor mtp.a.weight is also in the base");
    }

    #[test]
    fn u5_reason_14_geometry() {
        let mut f = Fixture::new("r14");
        f.tensors[0].shape = vec![4, 1];
        f.write_side();
        f.refused(
            "tensor mtp.a.weight is quant type 16 group 0 shape [4, 1], the manifest declares 16 group 0 shape [2, 2]",
        );
        let mut f = Fixture::new("r14b");
        f.tensors[1].quant_type = QT_BF16;
        f.write_side();
        f.refused("tensor mtp.layers.0.mlp.experts.gate_up_proj is quant type 16 group 128");
    }

    #[test]
    fn u5_reason_15_bytes() {
        let mut f = Fixture::new("r15");
        f.tensors[0].data = vec![0; 6];
        f.envelope.hipfire_qwen4_mtp_sidecar.tensors[0].sha256 = sha_hex(&f.tensors[0].data);
        f.write_side();
        f.refused("tensor mtp.a.weight is 6 bytes, BF16 [2, 2] is 8");
    }

    #[test]
    fn u5_reason_16_payload_sha() {
        let mut f = Fixture::new("r16");
        f.tensors[1].data[100] ^= 1;
        f.write_side();
        let error = f.bind().unwrap_err();
        assert!(
            error.starts_with("tensor mtp.layers.0.mlp.experts.gate_up_proj: payload sha256 "),
            "{error}"
        );
        assert!(error.contains(" differs from the manifest's "), "{error}");
    }

    #[test]
    fn u5_production_expected_entries_are_the_31_head_records() {
        let entries =
            expected_stock_mtp_entries(&crate::config::compact_test_config()).expect("headed");
        assert_eq!(entries.len(), STOCK_MTP_SIDECAR_TENSORS);
        let mut experts: Vec<_> = entries
            .iter()
            .filter(|entry| entry.dtype == DType::FP8E4M3B128)
            .collect();
        // Manifest order is authoritative; pin the geometry by name.
        experts.sort_by(|a, b| b.name.cmp(&a.name));
        assert_eq!(experts.len(), 2);
        assert_eq!(experts[0].name, "mtp.layers.0.mlp.experts.gate_up_proj");
        assert_eq!(experts[0].logical_shape, [512, 1280, 2560]);
        assert_eq!(experts[1].name, "mtp.layers.0.mlp.experts.down_proj");
        assert_eq!(experts[1].logical_shape, [512, 2560, 640]);
        assert!(entries
            .iter()
            .filter(|entry| entry.dtype != DType::FP8E4M3B128)
            .all(|entry| entry.dtype == DType::BF16));
        assert!(entries
            .iter()
            .all(|entry| entry.name != "mtp.embed_tokens.weight"));
        let sizes: Vec<u64> = experts
            .iter()
            .map(|entry| expected_record(entry).unwrap().bytes)
            .collect();
        assert_eq!(sizes, [1_677_983_744, 838_991_872]);
        let total: u64 = entries
            .iter()
            .map(|entry| expected_record(entry).unwrap().bytes)
            .sum();
        assert_eq!(total, 2_698_112_512);
    }
}
