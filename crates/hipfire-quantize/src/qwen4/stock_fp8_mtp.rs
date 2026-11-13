// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-quantize --recipe stock-fp8-mtp-sidecar`: the MTP head sidecar of
//! a stock-FP8 artifact.
//!
//! ```text
//! hipfire-quantize --recipe stock-fp8-mtp-sidecar --input <fp8-checkpoint-dir> \
//!     --base <model.hfq> [--base-sha256 <hex>] [--output <path>]
//! ```
//!
//! The AR artifact omits every `mtp.*` source tensor.  This producer reads
//! exactly those tensors from the same pinned checkpoint and writes them as a
//! second HFQM container (`<model>.hfq.mtp`, arch 16): the two stacked
//! `FP8E4M3B128` expert tables of the head (the AR producer's
//! `assemble_layer`, prefix `mtp.layers.0`) and the BF16 tensors verbatim.
//! The container is bound to the base by recipe id, file length, index
//! sha256 and whole-file sha256, and to the source by the pinned
//! `fp8.sha256` and its `config.json` line (design M0 §3.3, §4.1).
//!
//! # Gates (all before the first output byte unless noted)
//!
//! 1. Production: `<input>/fp8.sha256` hashes to the pin, shard names and
//!    sidecar files match it.
//! 2. The source census equals the production census (`STOCK_F8_MTP` = 1,536
//!    included) and the sidecar entries equal the manifest's non-tied `mtp.*`
//!    set exactly.
//! 3. The base is HFQM arch 16 with `qwen4_recipe.id == "stock-fp8"` and a
//!    `source_manifest_sha256` equal to the pin; its whole file is hashed in
//!    one streaming read and must equal `--base-sha256` when given.
//! 4. Every `mtp.*` F8_E4M3 tensor is scanned for NaN codes.
//! 5. Temporary write, fsync, reopen, payload sha256 re-read; production also
//!    rebinds with `bind_stock_mtp_sidecar`; no-replace publication.
//!
//! # Receipt
//!
//! `<sidecar>.receipt.json` carries the same per-source provenance as the
//! stock-FP8 receipt (shard, byte range, sha256, `count_0x80`), the base
//! binding, the census (`census.fp8.mtp`) and each entry's payload sha256.

use super::*;
use hipfire_arch_qwen4::stock_mtp_sidecar::{
    bind_stock_mtp_sidecar, qwen4_stock_mtp_sidecar_path, StockMtpBaseBinding,
    StockMtpSidecarEnvelope, StockMtpSidecarManifest, StockMtpSourceBinding, StockMtpTensorDigest,
    STOCK_MTP_SIDECAR_FORMAT, STOCK_MTP_SIDECAR_KEY, STOCK_MTP_SIDECAR_TENSORS,
    STOCK_MTP_SIDECAR_VERSION,
};
use hipfire_runtime::hfq::HfqFile;

const MTP_SIDECAR_RECIPE: &str = "stock-fp8-mtp-sidecar";
const MTP_LAYER_PREFIX: &str = "mtp.layers.0";

#[derive(Debug)]
pub(crate) struct StockFp8MtpOptions<'a> {
    pub(crate) input: &'a Path,
    /// The stock-FP8 AR artifact the sidecar extends.
    pub(crate) base: &'a Path,
    pub(crate) output: &'a Path,
    pub(crate) mode: Qwen4Mode,
    /// Pinned SHA-256 of `<input>/fp8.sha256` (required in both modes: it is
    /// the sidecar's source binding).  `Production` requires exactly
    /// [`STOCK_FP8_PINNED_MANIFEST_SHA256`].
    pub(crate) pin: Option<&'a str>,
    /// Lowercase hex sha256 the base file must hash to.
    pub(crate) base_sha256: Option<&'a str>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StockFp8MtpSummary {
    pub(crate) entries: usize,
    pub(crate) expert_entries: usize,
    pub(crate) mtp_fp8_scanned: usize,
    pub(crate) payload_bytes: u64,
    pub(crate) sidecar_bytes: u64,
    pub(crate) sidecar_sha256: String,
    pub(crate) base_sha256: String,
    pub(crate) output: PathBuf,
    pub(crate) receipt_path: PathBuf,
    pub(crate) elapsed_seconds: f64,
}

/// CLI entry point (production only).  `output` defaults to
/// `qwen4_stock_mtp_sidecar_path(base)`.
pub(crate) fn run_stock_fp8_mtp_sidecar_cli(
    input: &Path,
    base: &Path,
    output: Option<&Path>,
    base_sha256: Option<&str>,
) -> Result<StockFp8MtpSummary, Qwen4Error> {
    let base_sha256 = match base_sha256 {
        Some(hex) => Some(normalize_hex64(hex, "--base-sha256")?),
        None => None,
    };
    let output = match output {
        Some(output) => output.to_path_buf(),
        None => qwen4_stock_mtp_sidecar_path(base),
    };
    write_stock_fp8_mtp_sidecar(&StockFp8MtpOptions {
        input,
        base,
        output: &output,
        mode: Qwen4Mode::Production,
        pin: Some(STOCK_FP8_PINNED_MANIFEST_SHA256),
        base_sha256: base_sha256.as_deref(),
    })
}

fn normalize_hex64(value: &str, what: &str) -> Result<String, Qwen4Error> {
    let lower = value.to_ascii_lowercase();
    if lower.len() != 64 || !lower.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return invalid(format!("{what} is not 64 hex digits: {value:?}"));
    }
    Ok(lower)
}

// -------------------------------------------------------------------- planning

/// Plan the sidecar entries from the `mtp.*` source tensors the AR planner
/// omitted: the head's experts stacked by [`assemble_layer`], every other
/// tensor BF16 verbatim.  With the production `index` the entry set must equal
/// the manifest's non-tied `mtp.*` records and is ordered by the manifest.
fn plan_mtp_entries(
    tensors: Vec<SourceTensor>,
    mode: Qwen4Mode,
    index: Option<&ManifestIndex>,
) -> Result<Vec<StockEntry>, Qwen4Error> {
    let mut layers = BTreeMap::<String, LayerExperts>::new();
    let mut bf16 = Vec::<SourceTensor>::new();
    for tensor in tensors {
        let name = tensor.name.clone();
        if name.contains(".mlp.experts.") {
            let parsed = parse_expert_name(&name).ok_or_else(|| {
                Qwen4Error::Invalid(format!(
                    "{name}: not a separate routed-expert tensor \
                     (<layer>.mlp.experts.E.{{gate,up,down}}_proj.{{weight,weight_scale_inv}})"
                ))
            })?;
            if parsed.scale {
                require_dtype(&tensor, "BF16", "mtp expert scale")?;
            } else {
                require_dtype(&tensor, FP8_DTYPE, "mtp expert weight")?;
            }
            let layer = layers.entry(parsed.prefix.to_string()).or_default();
            let slot = if parsed.scale {
                &mut layer.scales
            } else {
                &mut layer.weights
            };
            if slot[parsed.proj as usize]
                .insert(parsed.expert, tensor)
                .is_some()
            {
                return invalid(format!("duplicate expert tensor {name}"));
            }
        } else {
            require_dtype(&tensor, "BF16", "mtp tensor")?;
            bf16.push(tensor);
        }
    }
    if layers.len() != 1 {
        return invalid(format!(
            "the stock-fp8 MTP head has exactly one expert layer, found {}",
            layers.len()
        ));
    }
    let (prefix, layer) = layers.into_iter().next().expect("one layer");
    if mode == Qwen4Mode::Production && prefix != MTP_LAYER_PREFIX {
        return invalid(format!(
            "the stock-fp8 MTP expert layer is {prefix}, production requires {MTP_LAYER_PREFIX}"
        ));
    }
    let (pair, _geometry) = assemble_layer(&prefix, layer, mode)?;
    let mut entries: Vec<StockEntry> = pair.into_iter().collect();
    for tensor in bf16 {
        let data_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
        if tensor.data_len() != data_len {
            return invalid(format!(
                "{} has {} payload bytes, expected {data_len}",
                tensor.name,
                tensor.data_len()
            ));
        }
        entries.push(StockEntry {
            name: tensor.name.clone(),
            quant_type: 16,
            shape: shape_u32(&tensor.shape, &tensor.name)?,
            group_size: 0,
            data_len,
            payload: StockPayload::Verbatim(tensor),
        });
    }

    if let Some(index) = index {
        // The inverse of the AR planner's filter: only `mtp.*`, and the tied
        // aliases are not in `required`, so they are not expected here.
        let expected: BTreeMap<&str, &[u64]> = index
            .required
            .iter()
            .filter(|(_, entry)| entry.is_mtp)
            .map(|(name, entry)| (name.as_str(), entry.shape.as_slice()))
            .collect();
        let actual: BTreeMap<&str, Vec<u64>> = entries
            .iter()
            .map(|entry| {
                (
                    entry.name.as_str(),
                    entry.shape.iter().map(|&d| u64::from(d)).collect(),
                )
            })
            .collect();
        let missing: Vec<String> = expected
            .keys()
            .filter(|name| !actual.contains_key(*name))
            .map(|name| name.to_string())
            .collect();
        let extra: Vec<String> = actual
            .keys()
            .filter(|name| !expected.contains_key(*name))
            .map(|name| name.to_string())
            .collect();
        if !missing.is_empty() || !extra.is_empty() {
            return invalid(format!(
                "stock-fp8 MTP source disagrees with the Flash-Next manifest: {} missing {}, {} undeclared {}",
                missing.len(),
                list_sample(&missing),
                extra.len(),
                list_sample(&extra)
            ));
        }
        for (name, shape) in &actual {
            if expected[name] != shape.as_slice() {
                return invalid(format!(
                    "{name} has shape {shape:?}, manifest expects {:?}",
                    expected[name]
                ));
            }
        }
        entries.sort_by_key(|entry| {
            index
                .manifest_order
                .get(&entry.name)
                .copied()
                .unwrap_or(usize::MAX)
        });
        if entries.len() != STOCK_MTP_SIDECAR_TENSORS {
            return invalid(format!(
                "the stock-fp8 MTP sidecar plans {} entries, the contract is {STOCK_MTP_SIDECAR_TENSORS}",
                entries.len()
            ));
        }
    } else {
        entries.sort_by(|left, right| left.name.cmp(&right.name));
    }
    Ok(entries)
}

// ------------------------------------------------------------------------ base

fn open_base(base: &Path, pin: &str) -> Result<HfqFile, Qwen4Error> {
    if !base.is_file() {
        return invalid(format!(
            "--base {} is not a file (the stock-fp8 artifact the sidecar extends)",
            base.display()
        ));
    }
    let file = HfqFile::open_at_offset(base, 0)
        .map_err(|error| Qwen4Error::io(format!("open base {}", base.display()), error))?;
    if file.arch_id != QWEN4_ARCH_ID {
        return invalid(format!(
            "base {} is arch {}, expected {QWEN4_ARCH_ID}",
            base.display(),
            file.arch_id
        ));
    }
    let metadata: Value = serde_json::from_str(&file.metadata_json)
        .map_err(|error| Qwen4Error::json("parse base metadata", error))?;
    let recipe = &metadata["qwen4_recipe"];
    if recipe["id"] != STOCK_FP8_RECIPE_ID {
        return invalid(format!(
            "base recipe is {}, the sidecar extends {STOCK_FP8_RECIPE_ID:?}",
            recipe["id"]
        ));
    }
    if recipe["source_manifest_sha256"] != pin {
        return invalid(format!(
            "base was converted from source manifest {}, the input's pin is {pin:?}",
            recipe["source_manifest_sha256"]
        ));
    }
    Ok(file)
}

// --------------------------------------------------------------------- hashing

/// Counts and hashes bytes written to it; used to digest an entry's payload
/// before the header (which carries the digests) is written.
#[derive(Default)]
struct HashingSink {
    hasher: Sha256,
    bytes: u64,
}

impl Write for HashingSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.hasher.update(buf);
        self.bytes += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Re-read every payload of the finished sidecar and compare it with the
/// envelope's digests (reason wording of design §7.3 item 16).
pub(super) fn verify_sidecar_payloads(path: &Path) -> Result<Qwen4ReopenPlan, Qwen4Error> {
    let plan = reopen_stock_artifact(path)?;
    let value: Value = serde_json::from_str(&plan.metadata_json)
        .map_err(|error| Qwen4Error::json("parse sidecar metadata", error))?;
    check_single_key(&value)?;
    let envelope: StockMtpSidecarEnvelope = serde_json::from_value(value)
        .map_err(|error| Qwen4Error::json("parse sidecar envelope", error))?;
    let manifest = envelope.hipfire_qwen4_mtp_sidecar;
    if manifest.tensors.len() != plan.entries.len() {
        return invalid(format!(
            "sidecar holds {} tensors, its manifest digests {}",
            plan.entries.len(),
            manifest.tensors.len()
        ));
    }
    let file =
        File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
    let mut buf = vec![0u8; IO_CHUNK_BYTES];
    for (entry, digest) in plan.entries.iter().zip(&manifest.tensors) {
        if entry.name != digest.name {
            return invalid(format!(
                "sidecar tensor is {}, the manifest's is {}",
                entry.name, digest.name
            ));
        }
        let mut hasher = Sha256::new();
        let mut done = 0u64;
        while done < entry.data_len {
            let n = (buf.len() as u64).min(entry.data_len - done) as usize;
            read_exact_at(&file, entry.data_offset + done, &mut buf[..n]).map_err(|error| {
                Qwen4Error::io(
                    format!("re-read {} of {}", entry.name, path.display()),
                    error,
                )
            })?;
            hasher.update(&buf[..n]);
            done += n as u64;
        }
        let got = hex_lower(hasher.finalize().as_slice());
        if got != digest.sha256 {
            return invalid(format!(
                "tensor {}: payload sha256 {got} differs from the manifest's {}",
                entry.name, digest.sha256
            ));
        }
    }
    Ok(plan)
}

fn check_single_key(value: &Value) -> Result<(), Qwen4Error> {
    match value.as_object() {
        Some(object) if object.len() == 1 && object.contains_key(STOCK_MTP_SIDECAR_KEY) => Ok(()),
        _ => invalid(format!(
            "sidecar metadata must be exactly one {STOCK_MTP_SIDECAR_KEY:?} object"
        )),
    }
}

// ---------------------------------------------------------------------- driver

/// Write the MTP sidecar for `options.base` from the pinned checkpoint.
pub(crate) fn write_stock_fp8_mtp_sidecar(
    options: &StockFp8MtpOptions<'_>,
) -> Result<StockFp8MtpSummary, Qwen4Error> {
    let started = Instant::now();
    if options
        .input
        .to_str()
        .is_some_and(|input| input.starts_with("hf://"))
    {
        return invalid(
            "stock-fp8-mtp-sidecar needs a local checkpoint directory (fp8.sha256 must be hashed)",
        );
    }
    let pin = options.pin.ok_or_else(|| {
        Qwen4Error::Invalid(
            "stock-fp8-mtp-sidecar requires the source manifest pin (it is the source binding)"
                .to_string(),
        )
    })?;
    if options.mode == Qwen4Mode::Production && pin != STOCK_FP8_PINNED_MANIFEST_SHA256 {
        return invalid(
            "stock-fp8-mtp-sidecar production mode requires the pinned manifest SHA-256",
        );
    }
    let receipt_path = receipt_path_for(options.output);
    for path in [options.output, receipt_path.as_path()] {
        if path.exists() {
            return invalid(format!(
                "refusing to replace existing stock-fp8 MTP sidecar output {}",
                path.display()
            ));
        }
    }

    // Gate 1: the manifest pin, before any source tensor or header is read.
    if !options.input.is_dir() {
        return invalid(format!(
            "stock-fp8-mtp-sidecar needs the checkpoint directory holding {STOCK_FP8_MANIFEST_NAME}: {}",
            options.input.display()
        ));
    }
    let manifest = verify_manifest(options.input, pin)?;
    let config_sha256 = manifest.files.get("config.json").cloned().ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "{} does not list config.json",
            manifest.path.display()
        ))
    })?;

    let source = source_paths(options.input)?;
    let SourceSet::Local { paths, .. } = &source else {
        return invalid("stock-fp8-mtp-sidecar needs a local source");
    };
    validate_source_shard_count(&source, options.mode)?;
    verify_manifest_files(options.input, &manifest, paths)?;
    let tensors = load_inventory(&source)?;
    let config_value = load_optional_config(&source)?.ok_or_else(|| {
        Qwen4Error::Invalid(
            "stock-fp8-mtp-sidecar input is missing required config.json".to_string(),
        )
    })?;
    let index = if options.mode == Qwen4Mode::Production {
        Some(production_index(&config_value)?)
    } else {
        None
    };

    // Gate 2: the AR planner's full census and manifest cross-check, then the
    // sidecar plan over exactly the `mtp.*` tensors it omitted.
    let inventory = plan_stock_entries(tensors, options.mode, index.as_ref())?;
    let mtp_tensors: Vec<SourceTensor> = inventory
        .omitted
        .iter()
        .filter(|omitted| omitted.reason == "mtp")
        .map(|omitted| omitted.tensor.clone())
        .collect();
    let mtp_bytes: u64 = mtp_tensors.iter().map(SourceTensor::data_len).sum();
    let mtp_count = mtp_tensors.len();
    let mtp_fp8: Vec<SourceTensor> = mtp_tensors
        .iter()
        .filter(|tensor| tensor.dtype == FP8_DTYPE)
        .cloned()
        .collect();
    let entries = plan_mtp_entries(mtp_tensors, options.mode, index.as_ref())?;
    let expert_entries = entries
        .iter()
        .filter(|entry| matches!(entry.payload, StockPayload::Experts { .. }))
        .count();

    // Gate 3: the base.
    let base = open_base(options.base, pin)?;
    let base_file_bytes = base.file_len();
    let base_index_sha256 = hex_lower(
        &base
            .index_sha256()
            .map_err(|error| Qwen4Error::io("hash base index", error))?,
    );
    let envelope_for = |base_sha256: &str, digests: &[String]| -> StockMtpSidecarEnvelope {
        StockMtpSidecarEnvelope {
            hipfire_qwen4_mtp_sidecar: StockMtpSidecarManifest {
                version: STOCK_MTP_SIDECAR_VERSION,
                format: STOCK_MTP_SIDECAR_FORMAT.to_string(),
                base: StockMtpBaseBinding {
                    recipe_id: STOCK_FP8_RECIPE_ID.to_string(),
                    file_bytes: base_file_bytes,
                    index_sha256: base_index_sha256.clone(),
                    sha256: base_sha256.to_string(),
                },
                source: StockMtpSourceBinding {
                    manifest_sha256: manifest.sha256.clone(),
                    config_sha256: config_sha256.clone(),
                },
                tensors: entries
                    .iter()
                    .zip(digests)
                    .map(|(entry, sha256)| StockMtpTensorDigest {
                        name: entry.name.clone(),
                        sha256: sha256.clone(),
                    })
                    .collect(),
            },
        }
    };
    let metadata_of = |envelope: &StockMtpSidecarEnvelope| -> Result<String, Qwen4Error> {
        serde_json::to_string(envelope)
            .map_err(|error| Qwen4Error::json("serialize stock-fp8 MTP sidecar metadata", error))
    };
    let stream_entries: Vec<hipfire_runtime::hfq::HfqStreamEntry> = entries
        .iter()
        .map(|entry| hipfire_runtime::hfq::HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        })
        .collect();
    let placeholder_digests = vec!["0".repeat(64); entries.len()];
    let placeholder_metadata = metadata_of(&envelope_for(&"0".repeat(64), &placeholder_digests))?;
    let predicted_output_bytes = predicted_output_bytes(&placeholder_metadata, &stream_entries)?;
    let source_records: u64 = entries
        .iter()
        .map(|entry| entry.sources().len() as u64)
        .sum();
    let receipt_estimate = RECEIPT_FIXED_BYTES + RECEIPT_BYTES_PER_SOURCE * source_records;
    capacity_preflight(
        options.output,
        predicted_output_bytes
            .checked_add(receipt_estimate)
            .ok_or_else(|| Qwen4Error::Invalid("predicted output size overflows".to_string()))?,
    )?;

    // The base's whole-file sha256: one streaming read.
    let (base_digest, base_hashed_bytes) = hash_file_streaming(options.base)?;
    let base_sha256 = hex_lower(&base_digest);
    if base_hashed_bytes != base_file_bytes {
        return invalid(format!(
            "base {} is {base_hashed_bytes} bytes on read, {base_file_bytes} when opened; it changed",
            options.base.display()
        ));
    }
    if let Some(want) = options.base_sha256 {
        if want != base_sha256 {
            return invalid(format!(
                "base {} sha256 {base_sha256} refused: --base-sha256 is {want}",
                options.base.display()
            ));
        }
    }

    // Gate 4: every `mtp.*` FP8 tensor scanned before any output exists.
    let scan = scan_fp8_sources(&mtp_fp8)?;
    source.verify_identity()?;

    // Entry digests: one pass over the sources into a hashing sink, so the
    // header can carry them.  The copier cross-checks every FP8 range against
    // the scan, then the write pass below must reproduce the same digests.
    let mut copier = Copier {
        buf: vec![0u8; IO_CHUNK_BYTES],
        copied: HashMap::new(),
        scan: &scan,
    };
    let mut digests = Vec::with_capacity(entries.len());
    for entry in &entries {
        let mut sink = HashingSink::default();
        copier.write_entry(entry, &mut sink)?;
        if sink.bytes != entry.data_len {
            return invalid(format!(
                "{} produced {} payload bytes, planned {}",
                entry.name, sink.bytes, entry.data_len
            ));
        }
        digests.push(hex_lower(sink.hasher.finalize().as_slice()));
    }
    let envelope = envelope_for(&base_sha256, &digests);
    let metadata_json = metadata_of(&envelope)?;

    // Write, fsync, verify.
    let temporary = temporary_output_path(options.output)?;
    let mut cleanup = TempCleanup(vec![temporary.clone()]);
    hipfire_runtime::hfq::write_hfqm_package_streaming(
        &temporary,
        QWEN4_ARCH_ID,
        &metadata_json,
        &stream_entries,
        |index, writer| {
            copier
                .write_entry(&entries[index], writer)
                .map_err(Qwen4Error::into_io)
        },
    )
    .map_err(|error| {
        Qwen4Error::io(
            format!("write stock-fp8 MTP sidecar {}", options.output.display()),
            error,
        )
    })?;
    let copied = copier.copied;
    sync_file(&temporary)?;
    source.verify_identity()?;
    let reopened = reopen_stock_artifact(&temporary)?;
    reopened.validate_against(&metadata_json, &stream_entries)?;
    verify_sidecar_payloads(&temporary)?;
    if options.mode == Qwen4Mode::Production {
        // The loader's own bind: every §4.2 check, payload sha256 included.
        bind_stock_mtp_sidecar(&base, &temporary).map_err(|reason| {
            Qwen4Error::Invalid(format!(
                "written sidecar fails its own bind check: {reason}"
            ))
        })?;
    }
    let (sidecar_digest, sidecar_bytes) = hash_file_streaming(&temporary)?;
    let sidecar_sha256 = hex_lower(&sidecar_digest);

    // Receipt.
    let omitted_fp8_count = mtp_fp8.len();
    let count80_total: u64 = scan.values().map(|digest| digest.count_0x80).sum();
    let pad_bytes: u64 = entries
        .iter()
        .map(|entry| match &entry.payload {
            StockPayload::Experts { blocks, pad } => pad * blocks.len() as u64,
            _ => 0,
        })
        .sum();
    let payload_bytes: u64 = entries.iter().map(|entry| entry.data_len).sum();
    let source_files: Vec<Value> = paths
        .iter()
        .map(|path| {
            json!({
                "name": path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "path": absolute(path),
                "bytes": fs::metadata(path).map(|m| m.len()).ok(),
                "manifest_sha256": path
                    .file_name()
                    .and_then(|name| manifest.files.get(&*name.to_string_lossy())),
            })
        })
        .collect();
    let tensors_json: Vec<Value> = entries
        .iter()
        .zip(&reopened.entries)
        .zip(&digests)
        .map(|((entry, placed), digest)| {
            let sources: Vec<Value> = entry
                .sources()
                .iter()
                .map(|use_| source_json(use_, &scan, &copied))
                .collect();
            json!({
                "name": entry.name,
                "quant_type": entry.quant_type,
                "shape": entry.shape,
                "group_size": entry.group_size,
                "artifact_offset": placed.data_offset,
                "artifact_len": placed.data_len,
                "payload_sha256": digest,
                "sources": sources,
            })
        })
        .collect();
    let receipt = json!({
        "receipt_version": 1,
        "recipe": MTP_SIDECAR_RECIPE,
        "component_mode": match options.mode {
            Qwen4Mode::Production => "production",
            Qwen4Mode::CompactFixture => "compact-fixture",
        },
        "artifact_sha256": sidecar_sha256,
        "artifact": {
            "path": absolute(options.output),
            "bytes": sidecar_bytes,
            "sha256": sidecar_sha256,
            "entries": entries.len(),
        },
        "base": {
            "path": absolute(options.base),
            "recipe_id": STOCK_FP8_RECIPE_ID,
            "file_bytes": base_file_bytes,
            "index_sha256": base_index_sha256,
            "sha256": base_sha256,
            "sha256_expected": options.base_sha256,
        },
        "manifest_sha256": manifest.sha256,
        "config_sha256": config_sha256,
        "pinned_manifest_sha256": pin,
        "manifest_verified": true,
        "input": absolute(options.input),
        "source_files": source_files,
        "census": {
            "source_tensors": inventory.census.total,
            "dtypes": inventory.census.by_dtype,
            "fp8": {
                "total": omitted_fp8_count,
                "scanned": scan.len(),
                "nan_codes": 0,
                "mtp": inventory.census.f8_mtp,
                "count_0x80": {"total": count80_total},
            },
            "mtp": {
                "tensors": mtp_count,
                "bytes": mtp_bytes,
            },
            "emitted_entries": entries.len(),
            "expert_entries": expert_entries,
        },
        "bytes": {
            "source_mtp": mtp_bytes,
            "artifact_total": sidecar_bytes,
            "artifact_payload": payload_bytes,
            "artifact_header_and_index": reopened.entries.first().map(|e| e.data_offset),
            "artifact_expert_pad_zero": pad_bytes,
            "predicted_artifact": predicted_output_bytes,
        },
        "tensors": tensors_json,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    let receipt_temporary = temporary_output_path(&receipt_path)?;
    cleanup.0.push(receipt_temporary.clone());
    {
        let file = File::create(&receipt_temporary).map_err(|error| {
            Qwen4Error::io(format!("create {}", receipt_temporary.display()), error)
        })?;
        let mut writer = io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, &receipt)
            .map_err(|error| Qwen4Error::json("serialize stock-fp8 MTP receipt", error))?;
        writer.write_all(b"\n").map_err(|error| {
            Qwen4Error::io(format!("write {}", receipt_temporary.display()), error)
        })?;
        writer.flush().map_err(|error| {
            Qwen4Error::io(format!("flush {}", receipt_temporary.display()), error)
        })?;
    }
    sync_file(&receipt_temporary)?;

    // Publish: sidecar first, then receipt; a failed receipt takes the
    // just-published sidecar back so none exists without its receipt.
    publish_qwen4(&temporary, options.output)?;
    if let Err(error) = publish_qwen4(&receipt_temporary, &receipt_path) {
        let _ = fs::remove_file(options.output);
        return Err(error);
    }

    Ok(StockFp8MtpSummary {
        entries: entries.len(),
        expert_entries,
        mtp_fp8_scanned: scan.len(),
        payload_bytes,
        sidecar_bytes,
        sidecar_sha256,
        base_sha256,
        output: options.output.to_path_buf(),
        receipt_path,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}
