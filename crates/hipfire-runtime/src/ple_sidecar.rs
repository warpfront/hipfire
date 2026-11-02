// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Flash-Next PLE row sidecar: the `ple-table` flavour of the `HFPLEHP1`
//! (PACK_VERSION 3) container that holds the PLE n-gram table next to a model
//! file (`<model>.ple-q8` / `<model>.ple-q4`), its strict open, the sealed
//! single-file [`SourceIdentity`] used for bounded shard reads of a row-ordered
//! table, and the load-time resolution that picks and binds a sidecar to an
//! [`HfqFile`].
//!
//! The container, its header type ([`PleHotPackHeader`] with
//! [`PleTableFields`]) and its reader/writer live in [`crate::ple_hot_pack`];
//! this module owns what is specific to a table that replaces the model's own
//! PLE shard tensors:
//!
//! * **row order** (`.ple-q8` from a split; a `.ple-q4` derived without a
//!   rank): shard `s` is the contiguous byte range
//!   `data_offset + s * rows_per_shard * encoded_row_bytes`, served by
//!   [`PleRowsFile::shard_descriptors`].
//! * **rank order** (a `.ple-q4` derived with a hot rank): rows are stored in
//!   tier order after an embedded rank, served by
//!   [`PleRowsFile::open_table_store`].
//!
//! # Binding
//!
//! A table's header records a base file binding (`PleTableFields::{base_index_sha256,
//! base_file_bytes}`: the sha256 of the base bytes `[0, data_offset)` — header,
//! metadata JSON, tensor index, padding — and the base file length). How
//! [`HfqFile::attach_ple_rows`] uses it depends on the base:
//!
//! * a **monolithic** base (embedded PLE shards) is bound to exactly one file:
//!   both fields must match it;
//! * a **split** base carries the [`PLE_BASE_MARKER_KEY`] metadata entry and is
//!   bound by row identity: the marker's geometry must equal the table's and its
//!   `q8_rows_sha256` must equal the table's `source_rows_sha256`. The two file
//!   binding fields are not compared, so one table serves every split base
//!   marked for the same rows.
//!
//! # API
//!
//! ```text
//! ple_shard_name(shard) -> String
//! merkle_sha256(&[[u8; 32]]) -> [u8; 32]
//! ple_fingerprint_from_metadata(metadata_json) -> Result<String, String>
//! PleRowsFile::{open, path, header, table, order, encoding, shard_count, rows_per_shard,
//!               row_width, file_bytes, shard_offset, shard_descriptors, open_table_store}
//! resolve_and_attach_ple_rows(&mut HfqFile, model_path, PleFormatRequest, gpu_arch,
//!                             ResolvePolicy) -> Result<PleRowsChoice, String>
//! ```

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use hipfire_config::memory::{ple_sidecar_path, PleFormat, PleFormatRequest};
use sha2::{Digest, Sha256};

use crate::external_rows::RowEncoding;
use crate::hfq::{HfqFile, HfqTensorInfo};
use crate::model_source::{
    capture_file_identity, read_file_exact_at, read_file_nowait_at, SourceError,
    SourceFileIdentity, SourceFormat, SourceIdentity, SourceRangeDescriptor, SourceRangeIdentity,
    SourceReader, SourceReaderImpl,
};
use crate::ple_convert::{derive_q4, PleProgress};
use crate::ple_hot_pack::{
    read_ple_hot_pack_header, PackKind, PackOrder, PleHotPackHeader, PleTableFields,
};
use crate::ple_hot_store::{HotRowStore, NowaitRead};

/// Name of PLE shard `s` is `PLE_SHARD_PREFIX + s + PLE_SHARD_SUFFIX`.
pub const PLE_SHARD_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
pub const PLE_SHARD_SUFFIX: &str = ".weight";
/// Top-level metadata key a split base carries (see the module docs).
pub const PLE_BASE_MARKER_KEY: &str = "hipfire_ple_rows";
/// Version of the base marker object.
const PLE_BASE_MARKER_VERSION: u64 = 1;
/// HFQ `quant_type` byte of a Q8F16 tensor.
pub(crate) const Q8_QUANT_TYPE: u8 = 3;
/// Registry download sizes quoted in refusal messages.
const Q8_SIZE_HINT: &str = "54.4 GB";
const Q4_SIZE_HINT: &str = "28.8 GB";
/// Multipliers / heads of the canonical `qwen4_ple` metadata object.
const PLE_MULTIPLIER_COUNT: usize = 3;
const PLE_HEAD_COUNT: usize = 16;

/// Tensor name of PLE shard `shard`.
pub fn ple_shard_name(shard: usize) -> String {
    format!("{PLE_SHARD_PREFIX}{shard}{PLE_SHARD_SUFFIX}")
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
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

/// The first 12 characters of a digest, for error messages.
pub(crate) fn short_hex(text: &str) -> &str {
    text.get(..12).unwrap_or(text)
}

/// `sha256(concat(shard_digests))`: the source-rows digest of a table.
pub fn merkle_sha256(shard_digests: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for digest in shard_digests {
        hasher.update(digest);
    }
    hasher.finalize().into()
}

/// Identity of the PLE row layout of a Qwen4 model, lowercase hex: sha256 over
/// the little-endian bytes of `padded_rows` (u64), the three multipliers (i64),
/// the sixteen head vocabulary sizes (u64) and the sixteen head offsets (u64),
/// read from the canonical `qwen4_ple` metadata object — exactly
/// `PleHashMetadata::fingerprint` of the Qwen4 architecture crate. A rank or a
/// table built for one hash layout is therefore never applied to another.
///
/// Only the fields the digest covers are read and typed (integers, never
/// through floating point); the layout invariants (prefix offsets, padding)
/// are the Qwen4 admission's to check.
pub fn ple_fingerprint_from_metadata(metadata_json: &str) -> Result<String, String> {
    let metadata: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("ple fingerprint: metadata JSON: {e}"))?;
    let object = metadata
        .get("qwen4_ple")
        .ok_or_else(|| "ple fingerprint: metadata lacks `qwen4_ple`".to_string())?
        .as_object()
        .ok_or_else(|| "ple fingerprint: `qwen4_ple` is not an object".to_string())?;
    let field = |name: &str| {
        object
            .get(name)
            .ok_or_else(|| format!("ple fingerprint: `qwen4_ple.{name}` is missing"))
    };
    let padded_rows = field("padded_rows")?
        .as_u64()
        .ok_or_else(|| "ple fingerprint: `qwen4_ple.padded_rows` is not a u64".to_string())?;
    fn array<'a>(
        object: &'a serde_json::Map<String, serde_json::Value>,
        name: &str,
        len: usize,
    ) -> Result<&'a Vec<serde_json::Value>, String> {
        let values = object
            .get(name)
            .ok_or_else(|| format!("ple fingerprint: `qwen4_ple.{name}` is missing"))?
            .as_array()
            .ok_or_else(|| format!("ple fingerprint: `qwen4_ple.{name}` is not an array"))?;
        if values.len() != len {
            return Err(format!(
                "ple fingerprint: `qwen4_ple.{name}` has {} values, expected {len}",
                values.len()
            ));
        }
        Ok(values)
    }
    let mut hasher = Sha256::new();
    hasher.update(padded_rows.to_le_bytes());
    for value in array(object, "multipliers", PLE_MULTIPLIER_COUNT)? {
        let value = value.as_i64().ok_or_else(|| {
            "ple fingerprint: `qwen4_ple.multipliers` holds a non-i64 value".to_string()
        })?;
        hasher.update(value.to_le_bytes());
    }
    for name in ["head_vocab_sizes", "head_offsets"] {
        for value in array(object, name, PLE_HEAD_COUNT)? {
            let value = value.as_u64().ok_or_else(|| {
                format!("ple fingerprint: `qwen4_ple.{name}` holds a non-u64 value")
            })?;
            hasher.update(value.to_le_bytes());
        }
    }
    Ok(hex_lower(&hasher.finalize()))
}

// ---------------------------------------------------------------------------
// Opened table + sealed source identity
// ---------------------------------------------------------------------------

/// Positional reader for the table file. Every read re-checks the sealed file
/// identity (one `stat`) exactly like the HFQ reader.
struct PleRowsReader {
    identity: Arc<SourceIdentity>,
    file: File,
    file_identity: SourceFileIdentity,
}

impl SourceReaderImpl for PleRowsReader {
    fn identity(&self) -> &SourceIdentity {
        self.identity.as_ref()
    }

    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        read_file_exact_at(&self.file, &self.file_identity, offset, dst)
    }

    fn read_nowait_at(&self, offset: u64, dst: &mut [u8]) -> Result<NowaitRead, SourceError> {
        read_file_nowait_at(&self.file, &self.file_identity, offset, dst)
    }
}

/// Sealed single-file identity and reader of a row-ordered table.
#[derive(Debug, Clone)]
struct SealedRows {
    identity: Arc<SourceIdentity>,
    reader: SourceReader,
}

/// Geometry derived from a validated `ple-table` header.
#[derive(Debug)]
struct TableGeometry {
    shard_count: usize,
    rows_per_shard: usize,
    row_width: usize,
}

/// The `ple-table` invariants this module relies on, beyond the container
/// reader's own checks: the header is a table of an encoding PLE rows can use,
/// its rows are the model's natural Q8F16 rows in the declared encoding, and
/// its shards tile `padded_rows` exactly.
fn check_table_header(header: &PleHotPackHeader) -> Result<TableGeometry, String> {
    if header.kind != PackKind::PleTable {
        return Err("this is a hot-row pack, not a PLE table".to_string());
    }
    let table = header
        .table
        .as_ref()
        .ok_or_else(|| "a PLE table header carries no table fields".to_string())?;
    let encoding = header.row_encoding;
    if !matches!(encoding, RowEncoding::Q8F16 | RowEncoding::Q4Lm) {
        return Err(format!(
            "row encoding {} is not a PLE table encoding (expected Q8_0 or Q4LM)",
            encoding.dtype_name()
        ));
    }
    if table.source_encoding != RowEncoding::Q8F16 {
        return Err(format!(
            "source encoding {} is not Q8_0",
            table.source_encoding.dtype_name()
        ));
    }
    let overflow = |what: &str| format!("{what} overflows this platform's usize");
    let row_width = usize::try_from(header.row_width).map_err(|_| overflow("row_width"))?;
    let rows_per_shard =
        usize::try_from(header.rows_per_shard).map_err(|_| overflow("rows_per_shard"))?;
    let expected_row_bytes = encoding
        .checked_encoded_row_bytes(row_width)
        .ok_or_else(|| {
            format!(
                "row_width {row_width} is not a valid {} row width",
                encoding.dtype_name()
            )
        })?;
    if header.encoded_row_bytes != expected_row_bytes as u64 {
        return Err(format!(
            "encoded_row_bytes {}, but {} rows of {row_width} values are {expected_row_bytes} bytes",
            header.encoded_row_bytes,
            encoding.dtype_name()
        ));
    }
    if rows_per_shard == 0 || header.padded_rows % header.rows_per_shard != 0 {
        return Err(format!(
            "padded_rows {} is not a whole number of {rows_per_shard}-row shards",
            header.padded_rows
        ));
    }
    let shard_count = usize::try_from(header.padded_rows / header.rows_per_shard)
        .map_err(|_| overflow("shard_count"))?;
    let data_bytes = header
        .padded_rows
        .checked_mul(header.encoded_row_bytes)
        .ok_or_else(|| "data_bytes overflow".to_string())?;
    if header.data_bytes != data_bytes {
        return Err(format!(
            "data_bytes {} != padded_rows {} * encoded_row_bytes {} = {data_bytes}",
            header.data_bytes, header.padded_rows, header.encoded_row_bytes
        ));
    }
    let q4lm = encoding == RowEncoding::Q4Lm;
    if q4lm != table.codebook_f32_bits.is_some() || q4lm != table.scale_grid_f32_bits.is_some() {
        return Err(format!(
            "codebook/scale are {} for a {} table",
            if q4lm { "absent" } else { "present" },
            encoding.dtype_name()
        ));
    }
    for (name, digest) in [
        ("source_rows_sha256", &table.source_rows_sha256),
        ("base_index_sha256", &table.base_index_sha256),
    ] {
        if !is_hex64_lower(digest) {
            return Err(format!("{name} is not 64 lowercase hex digits"));
        }
    }
    match header.order {
        PackOrder::Row if header.rank_bytes != 0 => {
            return Err("a row-ordered table carries a rank section".to_string())
        }
        PackOrder::Rank if header.rank_bytes != header.padded_rows => {
            return Err(format!(
                "rank_bytes {} != padded_rows {}",
                header.rank_bytes, header.padded_rows
            ))
        }
        _ => {}
    }
    Ok(TableGeometry {
        shard_count,
        rows_per_shard,
        row_width,
    })
}

/// The raw header JSON text of a validated container (the sealed
/// [`SourceIdentity::metadata_json`]).
fn read_header_json(path: &Path) -> Result<String, String> {
    let io_err = |what: &str, e: io::Error| format!("ple table {}: {what}: {e}", path.display());
    let mut file = File::open(path).map_err(|e| io_err("open", e))?;
    let mut prefix = [0u8; 16];
    file.read_exact(&mut prefix)
        .map_err(|e| io_err("read preamble", e))?;
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes"));
    let mut bytes = vec![
        0u8;
        usize::try_from(header_len).map_err(|_| format!(
            "ple table {}: header length {header_len} overflows",
            path.display()
        ))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|e| io_err("read header", e))?;
    String::from_utf8(bytes)
        .map_err(|_| format!("ple table {}: header is not valid UTF-8", path.display()))
}

/// An opened, validated PLE table sidecar (`HFPLEHP1` v3, kind `ple-table`).
///
/// A **row-ordered** table additionally carries a sealed single-file
/// [`SourceIdentity`] (format [`SourceFormat::PleRows`], one manifest range per
/// shard named [`ple_shard_name`]) from which [`Self::shard_descriptors`] hands
/// out bounded, identity-checked reads. A **rank-ordered** table is served by
/// [`Self::open_table_store`] instead.
#[derive(Clone)]
pub struct PleRowsFile {
    path: PathBuf,
    header: PleHotPackHeader,
    shard_count: usize,
    rows_per_shard: usize,
    row_width: usize,
    /// `Some` iff the table is row-ordered.
    sealed: Option<SealedRows>,
    /// The rank-ordered table's store, opened at most once and shared by every
    /// clone of this handle (the rank and slot index are built once).
    store: Arc<OnceLock<Result<Arc<HotRowStore>, String>>>,
}

impl std::fmt::Debug for PleRowsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PleRowsFile")
            .field("path", &self.path)
            .field("order", &self.header.order)
            .field("encoding", &self.header.row_encoding)
            .field("shard_count", &self.shard_count)
            .field("rows_per_shard", &self.rows_per_shard)
            .field("row_width", &self.row_width)
            .finish_non_exhaustive()
    }
}

impl PleRowsFile {
    /// Open and strictly validate `path`: the container reader's checks
    /// ([`read_ple_hot_pack_header`]: magic, version, every field, offsets,
    /// exact file length), then this module's (`kind` is `ple-table`, table
    /// fields present, Q8_0/Q4LM rows of a natural Q8_0 source, whole shards).
    /// A row-ordered table also seals its file identity.
    pub fn open(path: &Path) -> Result<Self, String> {
        let header = read_ple_hot_pack_header(path)?;
        let geometry = check_table_header(&header)
            .map_err(|e| format!("ple table {}: {e}", path.display()))?;
        let sealed = match header.order {
            PackOrder::Rank => None,
            PackOrder::Row => Some(Self::seal_rows(path, &header, &geometry)?),
        };
        Ok(Self {
            path: path.to_path_buf(),
            header,
            shard_count: geometry.shard_count,
            rows_per_shard: geometry.rows_per_shard,
            row_width: geometry.row_width,
            sealed,
            store: Arc::new(OnceLock::new()),
        })
    }

    fn seal_rows(
        path: &Path,
        header: &PleHotPackHeader,
        geometry: &TableGeometry,
    ) -> Result<SealedRows, String> {
        let err = |what: &str, e: &dyn std::fmt::Display| {
            format!("ple table {}: {what}: {e}", path.display())
        };
        let header_json = read_header_json(path)?;
        let file = File::open(path).map_err(|e| err("open", &e))?;
        let file_identity = capture_file_identity(path).map_err(|e| err("seal identity", &e))?;
        let expected_len = header.data_offset + header.data_bytes;
        if file_identity.len != expected_len {
            return Err(format!(
                "ple table {}: file is {} bytes, expected {expected_len} (changed while opening?)",
                path.display(),
                file_identity.len
            ));
        }
        let shard_len = header.rows_per_shard * header.encoded_row_bytes;
        let manifest = (0..geometry.shard_count)
            .map(|shard| SourceRangeIdentity {
                name: ple_shard_name(shard),
                file_index: 0,
                offset: header.data_offset + shard as u64 * shard_len,
                length: shard_len,
                dtype: header.row_encoding.dtype_name().to_string(),
                logical_shape: vec![geometry.rows_per_shard, geometry.row_width],
            })
            .collect();
        let identity = Arc::new(SourceIdentity {
            canonical_path: file_identity.canonical_path.clone(),
            format: SourceFormat::PleRows,
            files: vec![file_identity.clone()],
            metadata_json: header_json,
            manifest,
        });
        let reader = SourceReader::from_inner(PleRowsReader {
            identity: identity.clone(),
            file,
            file_identity,
        });
        Ok(SealedRows { identity, reader })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The container header (kind `ple-table`).
    pub fn header(&self) -> &PleHotPackHeader {
        &self.header
    }

    /// The table fields of the header: base binding, source digest, codebook
    /// and scale grid, data digests.
    pub fn table(&self) -> &PleTableFields {
        self.header
            .table
            .as_ref()
            .expect("open() validated that a PLE table carries table fields")
    }

    /// Row order (shard-contiguous) or rank order (tier-ordered).
    pub fn order(&self) -> PackOrder {
        self.header.order
    }

    pub fn encoding(&self) -> RowEncoding {
        self.header.row_encoding
    }

    /// Number of shards (`padded_rows / rows_per_shard`).
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }

    pub fn rows_per_shard(&self) -> usize {
        self.rows_per_shard
    }

    /// Values per row.
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    /// Total file length (`data_offset + data_bytes`).
    pub fn file_bytes(&self) -> u64 {
        self.header.data_offset + self.header.data_bytes
    }

    /// Absolute file offset of shard `shard`'s first row. `None` for a
    /// rank-ordered table (its rows are not stored per shard) or a shard out
    /// of range.
    pub fn shard_offset(&self, shard: usize) -> Option<u64> {
        if self.sealed.is_none() || shard >= self.shard_count {
            return None;
        }
        Some(
            self.header.data_offset
                + shard as u64 * self.header.rows_per_shard * self.header.encoded_row_bytes,
        )
    }

    /// One checked descriptor per shard, in shard order: dtype
    /// `encoding.dtype_name()`, logical shape `[rows_per_shard, row_width]`,
    /// all sharing one sealed identity and one identity-checking reader. Row
    /// order only; a rank-ordered table is an error here.
    pub fn shard_descriptors(&self) -> Result<Vec<SourceRangeDescriptor>, SourceError> {
        let sealed = self.sealed.as_ref().ok_or_else(|| SourceError::InvalidSource {
            reason: format!(
                "PLE table {} is rank-ordered; its rows are not shard-contiguous (open it with open_table_store)",
                self.path.display()
            ),
        })?;
        let length = self.header.rows_per_shard * self.header.encoded_row_bytes;
        let dtype = self.header.row_encoding.dtype_name();
        (0..self.shard_count)
            .map(|shard| {
                SourceRangeDescriptor::from_parts(
                    sealed.identity.clone(),
                    self.header.data_offset + shard as u64 * length,
                    length,
                    dtype.to_string(),
                    vec![self.rows_per_shard, self.row_width],
                    sealed.reader.clone(),
                )
            })
            .collect()
    }

    /// The rank-ordered table as a hot-row store that serves immediately
    /// ([`HotRowStore::open_table`]). Rank order only; a row-ordered table is
    /// an error here. Memoized: the store is opened on the first call and every
    /// call (on this handle or any clone) returns the same `Arc`, so the rank
    /// and slot index are built once per file.
    pub fn open_table_store(&self) -> Result<Arc<HotRowStore>, String> {
        if self.header.order != PackOrder::Rank {
            return Err(format!(
                "PLE table {} is row-ordered; read it through shard_descriptors",
                self.path.display()
            ));
        }
        self.store
            .get_or_init(|| HotRowStore::open_table(&self.path))
            .clone()
    }
}

// ---------------------------------------------------------------------------
// Base-file helpers (used by `HfqFile::attach_ple_rows` and the resolver)
// ---------------------------------------------------------------------------

/// Number of PLE shard tensors embedded in an HFQM index. Shard names must be
/// canonical (`shard_<decimal>`) and, if present, form the dense range
/// `0..n`; anything else is an error.
pub(crate) fn embedded_shard_count(tensors: &[HfqTensorInfo]) -> Result<usize, String> {
    let mut count = 0usize;
    let mut max_index = 0usize;
    for tensor in tensors {
        let Some(middle) = tensor
            .name
            .strip_prefix(PLE_SHARD_PREFIX)
            .and_then(|rest| rest.strip_suffix(PLE_SHARD_SUFFIX))
        else {
            continue;
        };
        let index = middle
            .parse::<usize>()
            .ok()
            .filter(|index| index.to_string() == middle)
            .ok_or_else(|| {
                format!(
                    "PLE shard tensor `{}` has a non-canonical name",
                    tensor.name
                )
            })?;
        count += 1;
        max_index = max_index.max(index);
    }
    if count > 0 && max_index + 1 != count {
        return Err(format!(
            "the index holds {count} PLE shard tensors but their numbering is not 0..{count}"
        ));
    }
    Ok(count)
}

/// The base marker (`hipfire_ple_rows`) of a split base, parsed strictly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PleBaseMarker {
    pub shard_count: usize,
    pub rows_per_shard: usize,
    pub row_width: usize,
    pub q8_rows_sha256: String,
}

pub(crate) fn parse_base_marker(value: &serde_json::Value) -> Result<PleBaseMarker, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("`{PLE_BASE_MARKER_KEY}` is not an object"))?;
    let get_u64 = |name: &str| {
        object.get(name).and_then(|v| v.as_u64()).ok_or_else(|| {
            format!("`{PLE_BASE_MARKER_KEY}.{name}` is missing or not an unsigned integer")
        })
    };
    let get_usize = |name: &str| {
        usize::try_from(get_u64(name)?)
            .map_err(|_| format!("`{PLE_BASE_MARKER_KEY}.{name}` overflows"))
    };
    let version = get_u64("version")?;
    if version != PLE_BASE_MARKER_VERSION {
        return Err(format!(
            "`{PLE_BASE_MARKER_KEY}.version` is {version}, expected {PLE_BASE_MARKER_VERSION}"
        ));
    }
    if object.get("location").and_then(|v| v.as_str()) != Some("sidecar") {
        return Err(format!(
            "`{PLE_BASE_MARKER_KEY}.location` is not \"sidecar\""
        ));
    }
    let q8_rows_sha256 = object
        .get("q8_rows_sha256")
        .and_then(|v| v.as_str())
        .filter(|text| is_hex64_lower(text))
        .ok_or_else(|| {
            format!(
                "`{PLE_BASE_MARKER_KEY}.q8_rows_sha256` is missing or not 64 lowercase hex digits"
            )
        })?
        .to_string();
    Ok(PleBaseMarker {
        shard_count: get_usize("shard_count")?,
        rows_per_shard: get_usize("rows_per_shard")?,
        row_width: get_usize("row_width")?,
        q8_rows_sha256,
    })
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Where the PLE rows of a loaded model come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PleRowsOrigin {
    /// The 128 shard tensors embedded in the model file (today's path).
    Embedded,
    /// A bound `<model>.ple-q4` / `.ple-q8` sidecar.
    Sidecar(PathBuf),
}

/// Outcome of [`resolve_and_attach_ple_rows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleRowsChoice {
    pub format: PleFormat,
    pub origin: PleRowsOrigin,
    /// The q4 sidecar was derived by this call.
    pub derived: bool,
    /// `auto` fell back to the non-preferred format: how to get the preferred.
    pub note: Option<String>,
}

/// What resolution may do besides opening existing files.
pub struct ResolvePolicy<'a> {
    /// Derive a missing `.ple-q4` (from the embedded rows or the q8 sidecar).
    /// Never honoured for `auto`. Explicit API consumers only: the serve/load
    /// path never sets it, because `memory.ple_format = "q4"` alone is not
    /// consent to write a ~29 GB file.
    pub derive_missing_q4: bool,
    /// Progress sink for a derivation.
    pub progress: Option<&'a mut (dyn FnMut(&PleProgress) + Send)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BaseKind {
    /// The PLE shard tensors are in the index.
    Monolithic,
    /// The shards are absent and the base carries the marker.
    Split,
}

fn classify_base(hfq: &HfqFile) -> Result<BaseKind, String> {
    let embedded = embedded_shard_count(hfq.tensors())?;
    let marker = hfq.ple_rows_marker().is_some();
    match (embedded > 0, marker) {
        (true, false) => Ok(BaseKind::Monolithic),
        (false, true) => Ok(BaseKind::Split),
        (true, true) => Err(format!(
            "{}: the base carries `{PLE_BASE_MARKER_KEY}` (rows in a sidecar) but its index also holds {embedded} embedded PLE shard tensors",
            hfq.path().display()
        )),
        (false, false) => Err(format!(
            "{}: not a PLE model: no embedded PLE shard tensors and no `{PLE_BASE_MARKER_KEY}` marker",
            hfq.path().display()
        )),
    }
}

fn format_of(encoding: RowEncoding) -> Result<PleFormat, String> {
    match encoding {
        RowEncoding::Q8F16 => Ok(PleFormat::Q8),
        RowEncoding::Q4Lm => Ok(PleFormat::Q4),
        other => Err(format!(
            "PLE rows encoding {} is neither q8 nor q4",
            other.dtype_name()
        )),
    }
}

fn size_hint(format: PleFormat) -> &'static str {
    match format {
        PleFormat::Q8 => Q8_SIZE_HINT,
        PleFormat::Q4 => Q4_SIZE_HINT,
    }
}

fn model_file_name(model_path: &Path) -> String {
    model_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.display().to_string())
}

/// `hipfire pull <model tag> --ple <format>` (size; model file name).
fn pull_hint(model_path: &Path, format: PleFormat) -> String {
    format!(
        "`hipfire pull <model tag> --ple {}` ({}; the registry tag of {})",
        format.name(),
        size_hint(format),
        model_file_name(model_path)
    )
}

fn derive_hint(model_path: &Path) -> String {
    format!("`hipfire ple derive --format q4 {}`", model_path.display())
}

/// True when `path` exists. Any error other than "not found" is a hard error:
/// a sidecar we cannot even stat must not silently fall back to another source.
fn sidecar_present(path: &Path) -> Result<bool, String> {
    match std::fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("PLE sidecar {}: {e}", path.display())),
    }
}

/// Open `path`, check it holds the encoding its name selects, bind it to
/// `hfq`. Any failure is a hard error.
fn open_and_attach(hfq: &mut HfqFile, path: &Path, format: PleFormat) -> Result<(), String> {
    let rows = PleRowsFile::open(path)?;
    let expected = match format {
        PleFormat::Q4 => RowEncoding::Q4Lm,
        PleFormat::Q8 => RowEncoding::Q8F16,
    };
    if rows.encoding() != expected {
        return Err(format!(
            "PLE sidecar {} holds {} rows but its name selects {}; delete or rename it and re-create it with {} or re-pull the matching file",
            path.display(),
            rows.encoding().dtype_name(),
            expected.dtype_name(),
            derive_hint(hfq.path()),
        ));
    }
    hfq.attach_ple_rows(rows)
}

fn sidecar_choice(
    format: PleFormat,
    path: PathBuf,
    derived: bool,
    note: Option<String>,
) -> PleRowsChoice {
    PleRowsChoice {
        format,
        origin: PleRowsOrigin::Sidecar(path),
        derived,
        note,
    }
}

fn derive_and_attach(
    hfq: &mut HfqFile,
    model_path: &Path,
    q4_path: &Path,
    progress: Option<&mut (dyn FnMut(&PleProgress) + Send)>,
) -> Result<(), String> {
    let derived = match progress {
        Some(progress) => derive_q4(model_path, None, progress),
        None => derive_q4(model_path, None, &mut |_: &PleProgress| {}),
    };
    derived.map_err(|e| format!("deriving {}: {e}", q4_path.display()))?;
    open_and_attach(hfq, q4_path, PleFormat::Q4)
}

/// A sidecar found by [`locate_sidecar`]: the file and the base path whose
/// `<base>.ple-*` name it carries.
struct FoundSidecar {
    sidecar: PathBuf,
    base: PathBuf,
}

/// Base paths whose sidecars are looked up, in order: `model_path` exactly as
/// given (a symlink keeps the sidecars published beside the link), then its
/// canonical target when that differs.
fn sidecar_bases(model_path: &Path) -> Vec<PathBuf> {
    let mut bases = vec![model_path.to_path_buf()];
    if let Ok(canonical) = std::fs::canonicalize(model_path) {
        if canonical != model_path {
            bases.push(canonical);
        }
    }
    bases
}

/// The first existing sidecar of `format` across `bases`. A sidecar that
/// exists but cannot be stat'ed is a hard error ([`sidecar_present`]).
fn locate_sidecar(bases: &[PathBuf], format: PleFormat) -> Result<Option<FoundSidecar>, String> {
    for base in bases {
        let sidecar = ple_sidecar_path(base, format);
        if sidecar_present(&sidecar)? {
            return Ok(Some(FoundSidecar {
                sidecar,
                base: base.clone(),
            }));
        }
    }
    Ok(None)
}

/// Every location [`locate_sidecar`] probes for `format`, for messages.
fn searched(bases: &[PathBuf], format: PleFormat) -> String {
    bases
        .iter()
        .map(|base| ple_sidecar_path(base, format).display().to_string())
        .collect::<Vec<_>>()
        .join(" and ")
}

/// Resolve existing PLE rows for validation tools, without deriving missing q4.
/// Oracle, parity and replay gates share this entrypoint and attachment policy.
pub fn attach_validation_ple_rows(
    hfq: &mut HfqFile,
    model_path: &Path,
    request: PleFormatRequest,
    gpu_arch: &str,
) -> Result<PleRowsChoice, String> {
    resolve_and_attach_ple_rows(
        hfq,
        model_path,
        request,
        gpu_arch,
        ResolvePolicy {
            derive_missing_q4: false,
            progress: None,
        },
    )
}

/// Pick the PLE rows source for a Qwen4 model and bind it.
///
/// Base kinds: **monolithic** (the shard tensors are in the index) and
/// **split** (they are absent and the base carries the
/// [`PLE_BASE_MARKER_KEY`] marker); anything else is an error.
///
/// | request | monolithic | split base |
/// |---|---|---|
/// | `q8` | embedded rows | `<base>.ple-q8`, else refuse |
/// | `q4` | `<model>.ple-q4`, else derive (if allowed), else refuse | `<base>.ple-q4`, else derive from `<base>.ple-q8` (if allowed), else refuse |
/// | `auto` | preferred (q4 on gfx1151, else q8) if available, else the other with a `note`; **never derives, never downloads**; monolithic always has q8 | same; neither available → refuse |
///
/// Every `<model>.ple-*` lookup probes `model_path` exactly as given first
/// (so sidecars published beside a symlink are found), then the canonical
/// target of `model_path` (`<target>.ple-*`). Derivation happens only when
/// `policy.derive_missing_q4` is set by the caller; the load path leaves it
/// off, so a missing q4 sidecar is an error naming both searched locations.
/// A present sidecar that fails to open or bind is always a hard error. The
/// call is idempotent: an already attached sidecar is reported, not reopened
/// (an explicit request for the other format is an error).
pub fn resolve_and_attach_ple_rows(
    hfq: &mut HfqFile,
    model_path: &Path,
    request: PleFormatRequest,
    gpu_arch: &str,
    mut policy: ResolvePolicy<'_>,
) -> Result<PleRowsChoice, String> {
    let auto = request == PleFormatRequest::Auto;
    let wanted = request.resolve(gpu_arch);

    if let Some(rows) = hfq.ple_rows() {
        let attached = format_of(rows.encoding())?;
        if !auto && attached != wanted {
            return Err(format!(
                "PLE rows are already attached from {} ({}), but {} was requested",
                rows.path().display(),
                attached.name(),
                wanted.name()
            ));
        }
        return Ok(sidecar_choice(
            attached,
            rows.path().to_path_buf(),
            false,
            None,
        ));
    }

    let kind = classify_base(hfq)?;
    let bases = sidecar_bases(model_path);
    let q4_at = locate_sidecar(&bases, PleFormat::Q4)?;
    let q8_at = locate_sidecar(&bases, PleFormat::Q8)?;
    let at = |format: PleFormat| match format {
        PleFormat::Q4 => &q4_at,
        PleFormat::Q8 => &q8_at,
    };
    let searched_for = |format: PleFormat| searched(&bases, format);

    match kind {
        BaseKind::Monolithic => {
            let embedded = |note: Option<String>| PleRowsChoice {
                format: PleFormat::Q8,
                origin: PleRowsOrigin::Embedded,
                derived: false,
                note,
            };
            if wanted == PleFormat::Q8 {
                return Ok(embedded(None));
            }
            if let Some(found) = &q4_at {
                open_and_attach(hfq, &found.sidecar, PleFormat::Q4)?;
                return Ok(sidecar_choice(
                    PleFormat::Q4,
                    found.sidecar.clone(),
                    false,
                    None,
                ));
            }
            if auto {
                return Ok(embedded(Some(format!(
                    "q4 PLE sidecar not found (searched {}); using the embedded q8 rows. Create the q4 rows with {}",
                    searched_for(PleFormat::Q4),
                    derive_hint(model_path)
                ))));
            }
            if policy.derive_missing_q4 {
                let q4_path = ple_sidecar_path(model_path, PleFormat::Q4);
                derive_and_attach(hfq, model_path, &q4_path, policy.progress.take())?;
                return Ok(sidecar_choice(PleFormat::Q4, q4_path, true, None));
            }
            Err(format!(
                "PLE q4 sidecar is missing (searched {}) and derivation was not requested. Fetch it with {} or create it with {}, or set memory.ple_format = \"q8\" to load the embedded q8 rows",
                searched_for(PleFormat::Q4),
                pull_hint(model_path, PleFormat::Q4),
                derive_hint(model_path)
            ))
        }
        BaseKind::Split => {
            if auto {
                let (preferred, other) = match wanted {
                    PleFormat::Q4 => (PleFormat::Q4, PleFormat::Q8),
                    PleFormat::Q8 => (PleFormat::Q8, PleFormat::Q4),
                };
                if let Some(found) = at(preferred) {
                    let path = found.sidecar.clone();
                    open_and_attach(hfq, &path, preferred)?;
                    return Ok(sidecar_choice(preferred, path, false, None));
                }
                if let Some(found) = at(other) {
                    let path = found.sidecar.clone();
                    open_and_attach(hfq, &path, other)?;
                    let how = match preferred {
                        PleFormat::Q4 => format!(
                            "{} (derives from the q8 rows) or {}",
                            derive_hint(model_path),
                            pull_hint(model_path, PleFormat::Q4)
                        ),
                        PleFormat::Q8 => pull_hint(model_path, PleFormat::Q8),
                    };
                    let note = format!(
                        "{} PLE sidecar not found (searched {}); using the {} rows. Get the {} rows with {how}",
                        preferred.name(),
                        searched_for(preferred),
                        other.name(),
                        preferred.name()
                    );
                    return Ok(sidecar_choice(other, path, false, Some(note)));
                }
                return Err(format!(
                    "{} is a split base without embedded PLE rows, and neither the q8 sidecar (searched {}) nor the q4 sidecar (searched {}) exists. Fetch one with {} or {} ({} derives q4 once the q8 file is present)",
                    model_path.display(),
                    searched_for(PleFormat::Q8),
                    searched_for(PleFormat::Q4),
                    pull_hint(model_path, PleFormat::Q4),
                    pull_hint(model_path, PleFormat::Q8),
                    derive_hint(model_path)
                ));
            }
            match wanted {
                PleFormat::Q8 => {
                    if let Some(found) = &q8_at {
                        open_and_attach(hfq, &found.sidecar, PleFormat::Q8)?;
                        return Ok(sidecar_choice(
                            PleFormat::Q8,
                            found.sidecar.clone(),
                            false,
                            None,
                        ));
                    }
                    Err(format!(
                        "PLE q8 sidecar is missing (searched {}; {} is a split base without embedded PLE rows). Fetch it with {}, or set memory.ple_format = \"q4\" and use the q4 rows ({} or {} once the q8 file is present)",
                        searched_for(PleFormat::Q8),
                        model_path.display(),
                        pull_hint(model_path, PleFormat::Q8),
                        pull_hint(model_path, PleFormat::Q4),
                        derive_hint(model_path)
                    ))
                }
                PleFormat::Q4 => {
                    if let Some(found) = &q4_at {
                        open_and_attach(hfq, &found.sidecar, PleFormat::Q4)?;
                        return Ok(sidecar_choice(
                            PleFormat::Q4,
                            found.sidecar.clone(),
                            false,
                            None,
                        ));
                    }
                    if let (Some(q8), true) = (&q8_at, policy.derive_missing_q4) {
                        // Derive beside the base whose q8 rows were found.
                        let q4_path = ple_sidecar_path(&q8.base, PleFormat::Q4);
                        derive_and_attach(hfq, &q8.base, &q4_path, policy.progress.take())?;
                        return Ok(sidecar_choice(PleFormat::Q4, q4_path, true, None));
                    }
                    if let Some(q8) = &q8_at {
                        return Err(format!(
                            "PLE q4 sidecar is missing (searched {}) and derivation was not requested. Derive it from {} with {}, fetch it with {}, or set memory.ple_format = \"q8\"",
                            searched_for(PleFormat::Q4),
                            q8.sidecar.display(),
                            derive_hint(&q8.base),
                            pull_hint(model_path, PleFormat::Q4)
                        ));
                    }
                    Err(format!(
                        "PLE q4 sidecar is missing (searched {}; {} is a split base without embedded PLE rows). Fetch it with {} ({} to derive it needs the q8 file, searched {}, first), or set memory.ple_format = \"q8\" and fetch {}",
                        searched_for(PleFormat::Q4),
                        model_path.display(),
                        pull_hint(model_path, PleFormat::Q4),
                        derive_hint(model_path),
                        searched_for(PleFormat::Q8),
                        pull_hint(model_path, PleFormat::Q8)
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use crate::ple_hot_pack::{encode_ple_table_header, TABLE_SEGMENT_BYTES};
    use crate::ple_q4::{lm4_codebook_bits, lm4_scale_grid_bits};

    const SHARDS: usize = 3;
    const ROWS: usize = 4;
    const WIDTH: usize = 64;
    const Q8_ROW: usize = WIDTH / 32 * 34;
    const Q4_ROW: usize = WIDTH / 32 * 18;

    fn row_bytes(encoding: RowEncoding) -> usize {
        match encoding {
            RowEncoding::Q8F16 => Q8_ROW,
            RowEncoding::Q4Lm => Q4_ROW,
            _ => unreachable!(),
        }
    }

    /// Deterministic distinct shard bytes of `rows` rows. Q8F16 shards are
    /// well-formed blocks (finite f16 scale + 32 i8), so a real derivation can
    /// consume them.
    fn shard_rows(encoding: RowEncoding, shard: usize, salt: u8, rows: usize) -> Vec<u8> {
        let mix = salt.wrapping_add(shard as u8 * 17);
        match encoding {
            RowEncoding::Q8F16 => {
                let mut out = Vec::with_capacity(rows * Q8_ROW);
                for block in 0..rows * WIDTH / 32 {
                    let scale = half::f16::from_f32(0.01 * (1 + (block + shard) % 5) as f32);
                    out.extend_from_slice(&scale.to_bits().to_le_bytes());
                    out.extend(
                        (0..32usize).map(|i| ((block * 7 + i * 13) as u8).wrapping_add(mix)),
                    );
                }
                out
            }
            _ => (0..rows * row_bytes(encoding))
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(mix))
                .collect(),
        }
    }

    fn all_shards(encoding: RowEncoding, salt: u8) -> Vec<Vec<u8>> {
        (0..SHARDS)
            .map(|s| shard_rows(encoding, s, salt, ROWS))
            .collect()
    }

    /// Merkle digest of natural-order shard bytes: a table's `source_rows_sha256`
    /// when the shards are Q8F16, and what a split base's marker records.
    fn merkle_of(shards: &[Vec<u8>]) -> String {
        let digests: Vec<[u8; 32]> = shards.iter().map(|s| Sha256::digest(s).into()).collect();
        hex_lower(&merkle_sha256(&digests))
    }

    fn q8_rows_sha(salt: u8) -> String {
        merkle_of(&all_shards(RowEncoding::Q8F16, salt))
    }

    /// Per-segment digests of the data section and their Merkle (`data_sha256`).
    fn segments(data: &[u8]) -> (Vec<String>, String) {
        let digests: Vec<[u8; 32]> = data
            .chunks(TABLE_SEGMENT_BYTES as usize)
            .map(|chunk| Sha256::digest(chunk).into())
            .collect();
        (
            digests.iter().map(|d| hex_lower(d)).collect(),
            hex_lower(&merkle_sha256(&digests)),
        )
    }

    /// A consistent `ple-table` header over `data` (the data section as it is
    /// laid out in the file). `buckets` selects rank order. The base binding is
    /// a placeholder until [`bind`].
    fn table_header(
        encoding: RowEncoding,
        rows_per_shard: usize,
        data: &[u8],
        buckets: Option<&[u8]>,
        tier_group: usize,
        source_rows: &str,
    ) -> PleHotPackHeader {
        let q4 = encoding == RowEncoding::Q4Lm;
        let padded_rows = (data.len() / row_bytes(encoding)) as u64;
        let (segment_sha256, data_sha256) = segments(data);
        let (order, tier_group, tier_rows, rank_payload) = match buckets {
            Some(buckets) => {
                let tiers = 256usize.div_ceil(tier_group);
                let mut tier_rows = vec![0u64; tiers];
                for &bucket in buckets {
                    tier_rows[usize::from(bucket) / tier_group] += 1;
                }
                (
                    PackOrder::Rank,
                    tier_group,
                    tier_rows,
                    hex_lower(&Sha256::digest(buckets)),
                )
            }
            None => (PackOrder::Row, 256, vec![padded_rows], "0".repeat(64)),
        };
        PleHotPackHeader {
            padded_rows,
            rows_per_shard: rows_per_shard as u64,
            row_encoding: encoding,
            row_width: WIDTH as u64,
            encoded_row_bytes: row_bytes(encoding) as u64,
            // The fixture models' own fingerprint, so derive accepts the table.
            ple_fingerprint: ple_fingerprint_from_metadata(&format!("{{{}}}", qwen4_ple_json()))
                .expect("fixture qwen4_ple metadata"),
            rank_payload_sha256: rank_payload,
            tier_group,
            kept_buckets: 256,
            tier_rows,
            data_offset: 0,
            data_bytes: data.len() as u64,
            shard_offsets: Vec::new(),
            model_file_bytes: 0,
            version: 3,
            kind: PackKind::PleTable,
            order,
            rank_offset: 0,
            rank_bytes: buckets.map_or(0, |b| b.len() as u64),
            table: Some(PleTableFields {
                codebook_f32_bits: q4.then(lm4_codebook_bits),
                scale_grid_f32_bits: q4.then(lm4_scale_grid_bits),
                source_encoding: RowEncoding::Q8F16,
                source_rows_sha256: source_rows.to_string(),
                segment_bytes: TABLE_SEGMENT_BYTES,
                segment_sha256,
                data_sha256,
                base_index_sha256: "ab".repeat(32),
                base_file_bytes: 12345,
            }),
            generator: "hipfire ple derive".to_string(),
        }
    }

    fn table_mut(header: &mut PleHotPackHeader) -> &mut PleTableFields {
        header.table.as_mut().expect("test header has table fields")
    }

    fn bind(header: &mut PleHotPackHeader, base: &HfqFile) {
        let table = table_mut(header);
        table.base_index_sha256 = hex_lower(&base.index_sha256().unwrap());
        table.base_file_bytes = base.file_len();
    }

    /// Write `header` (`encode_ple_table_header` sets the offsets), the embedded
    /// rank if any, and the data section.
    fn write_table(
        path: &Path,
        header: &mut PleHotPackHeader,
        buckets: Option<&[u8]>,
        data: &[u8],
    ) {
        let mut bytes = encode_ple_table_header(header).unwrap();
        if let Some(buckets) = buckets {
            assert_eq!(bytes.len() as u64, header.rank_offset);
            bytes.extend_from_slice(buckets);
            bytes.resize(header.data_offset as usize, 0);
        }
        assert_eq!(bytes.len() as u64, header.data_offset);
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes).unwrap();
    }

    /// A row-ordered table of `shards` (natural order), bound to nothing yet.
    fn row_table(
        encoding: RowEncoding,
        shards: &[Vec<u8>],
        rows_per_shard: usize,
        source_rows: &str,
    ) -> (PleHotPackHeader, Vec<u8>) {
        let data = shards.concat();
        (
            table_header(encoding, rows_per_shard, &data, None, 256, source_rows),
            data,
        )
    }

    /// Buckets for `padded_rows` rows spreading over both tiers of a
    /// `tier_group` 128 layout.
    fn buckets_for(padded_rows: usize) -> Vec<u8> {
        (0..padded_rows)
            .map(|r| ((r * 97 + 13) % 256) as u8)
            .collect()
    }

    /// The data section of a rank-ordered table: rows in tier order (tier `t` =
    /// rows with `bucket / tier_group == t`, ascending global row order).
    fn rank_data(natural: &[u8], row_len: usize, buckets: &[u8], tier_group: usize) -> Vec<u8> {
        let rows: Vec<&[u8]> = natural.chunks(row_len).collect();
        let mut out = Vec::with_capacity(natural.len());
        for tier in 0..256usize.div_ceil(tier_group) {
            for (row, &bucket) in buckets.iter().enumerate() {
                if usize::from(bucket) / tier_group == tier {
                    out.extend_from_slice(rows[row]);
                }
            }
        }
        out
    }

    /// Write a bound table for `base` in row order.
    fn write_bound_sidecar(
        sidecar: &Path,
        base: &Path,
        encoding: RowEncoding,
        source_rows: &str,
        salt: u8,
    ) -> PleHotPackHeader {
        let hfq = HfqFile::open(base).unwrap();
        let shards = all_shards(encoding, salt);
        let (mut header, data) = row_table(encoding, &shards, ROWS, source_rows);
        bind(&mut header, &hfq);
        write_table(sidecar, &mut header, None, &data);
        header
    }

    /// Write a bound table for `base` built by `mutate` over a consistent
    /// row-ordered table of `shards` of `rows_per_shard` rows.
    fn write_mutated_sidecar(
        sidecar: &Path,
        base: &HfqFile,
        encoding: RowEncoding,
        rows_per_shard: usize,
        source_rows: &str,
        mutate: impl FnOnce(&mut PleHotPackHeader),
    ) {
        let shards: Vec<Vec<u8>> = (0..SHARDS)
            .map(|s| shard_rows(encoding, s, 1, rows_per_shard))
            .collect();
        let (mut header, data) = row_table(encoding, &shards, rows_per_shard, source_rows);
        bind(&mut header, base);
        mutate(&mut header);
        write_table(sidecar, &mut header, None, &data);
    }

    // -- small pieces -------------------------------------------------------

    #[test]
    fn shard_names_match_the_model_tensor_names() {
        assert_eq!(
            ple_shard_name(0),
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
        );
        assert_eq!(
            ple_shard_name(127),
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight"
        );
    }

    #[test]
    fn merkle_is_sha_of_concatenated_digests() {
        let a: [u8; 32] = Sha256::digest(b"a").into();
        let b: [u8; 32] = Sha256::digest(b"b").into();
        let mut concat = a.to_vec();
        concat.extend_from_slice(&b);
        assert_eq!(
            merkle_sha256(&[a, b]),
            <[u8; 32]>::from(Sha256::digest(&concat))
        );
        assert_ne!(merkle_sha256(&[a, b]), merkle_sha256(&[b, a]));
        assert_eq!(merkle_sha256(&[]), <[u8; 32]>::from(Sha256::digest(b"")));
    }

    fn qwen4_ple_metadata(padded_rows: u64) -> serde_json::Value {
        let sizes: Vec<u64> = (0..16).map(|i| 100 + 3 * i).collect();
        let mut offsets = Vec::new();
        let mut at = 0u64;
        for size in &sizes {
            offsets.push(at);
            at += size;
        }
        serde_json::json!({
            "arch_id": 16,
            "qwen4_ple": {
                "version": 1,
                "multipliers": [-3i64, 5, 23_703_573_157_769i64],
                "head_vocab_sizes": sizes,
                "head_offsets": offsets,
                "padded_rows": padded_rows,
            }
        })
    }

    #[test]
    fn fingerprint_hashes_padded_rows_multipliers_sizes_and_offsets_in_order() {
        let metadata = qwen4_ple_metadata(4096);
        let mut hasher = Sha256::new();
        hasher.update(4096u64.to_le_bytes());
        for m in [-3i64, 5, 23_703_573_157_769i64] {
            hasher.update(m.to_le_bytes());
        }
        let ple = &metadata["qwen4_ple"];
        for name in ["head_vocab_sizes", "head_offsets"] {
            for v in ple[name].as_array().unwrap() {
                hasher.update(v.as_u64().unwrap().to_le_bytes());
            }
        }
        let expected = hex_lower(&hasher.finalize());
        let got = ple_fingerprint_from_metadata(&metadata.to_string()).unwrap();
        assert_eq!(got, expected);
        assert!(is_hex64_lower(&got));

        // every covered field moves the digest
        let base = got;
        let mut variants = Vec::new();
        variants.push(qwen4_ple_metadata(4097));
        let mut v = qwen4_ple_metadata(4096);
        v["qwen4_ple"]["multipliers"][1] = serde_json::json!(6);
        variants.push(v);
        let mut v = qwen4_ple_metadata(4096);
        v["qwen4_ple"]["head_vocab_sizes"][15] = serde_json::json!(1);
        variants.push(v);
        let mut v = qwen4_ple_metadata(4096);
        v["qwen4_ple"]["head_offsets"][2] = serde_json::json!(1);
        variants.push(v);
        for variant in variants {
            assert_ne!(
                ple_fingerprint_from_metadata(&variant.to_string()).unwrap(),
                base
            );
        }
        // fields the digest does not cover (other metadata, the version) do not
        let mut v = qwen4_ple_metadata(4096);
        v["arch_id"] = serde_json::json!(99);
        assert_eq!(ple_fingerprint_from_metadata(&v.to_string()).unwrap(), base);
    }

    #[test]
    fn fingerprint_rejects_malformed_metadata() {
        let good = qwen4_ple_metadata(4096);
        assert!(ple_fingerprint_from_metadata("{")
            .unwrap_err()
            .contains("metadata JSON"));
        assert!(ple_fingerprint_from_metadata("{}")
            .unwrap_err()
            .contains("qwen4_ple"));
        assert!(ple_fingerprint_from_metadata(r#"{"qwen4_ple":1}"#)
            .unwrap_err()
            .contains("not an object"));
        let cases: Vec<(&str, serde_json::Value, &str)> = vec![
            ("padded_rows", serde_json::json!(1.5), "padded_rows"),
            ("padded_rows", serde_json::json!(-1), "padded_rows"),
            ("multipliers", serde_json::json!([1, 2]), "has 2 values"),
            ("multipliers", serde_json::json!([1, 2, 1.5]), "non-i64"),
            ("multipliers", serde_json::json!("x"), "not an array"),
            ("head_vocab_sizes", serde_json::json!([1]), "has 1 values"),
            (
                "head_offsets",
                serde_json::json!(vec![-1i64; 16]),
                "non-u64",
            ),
        ];
        for (field, value, what) in cases {
            let mut bad = good.clone();
            bad["qwen4_ple"][field] = value;
            let err = ple_fingerprint_from_metadata(&bad.to_string()).unwrap_err();
            assert!(err.contains(what), "`{what}` in: {err}");
        }
        for field in [
            "padded_rows",
            "multipliers",
            "head_vocab_sizes",
            "head_offsets",
        ] {
            let mut bad = good.clone();
            bad["qwen4_ple"].as_object_mut().unwrap().remove(field);
            let err = ple_fingerprint_from_metadata(&bad.to_string()).unwrap_err();
            assert!(err.contains(field) && err.contains("missing"), "{err}");
        }
    }

    // -- opening tables -----------------------------------------------------

    #[test]
    fn tables_round_trip_in_row_and_rank_order() {
        let dir = tempfile::tempdir().unwrap();
        let q8 = all_shards(RowEncoding::Q8F16, 3);
        let source = merkle_of(&q8);
        for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            let shards = all_shards(encoding, 3);
            let natural = shards.concat();
            let padded_rows = SHARDS * ROWS;
            for rank in [false, true] {
                let path = dir
                    .path()
                    .join(format!("{}-{rank}.table", encoding.dtype_name()));
                let buckets = buckets_for(padded_rows);
                let (mut header, data) = if rank {
                    let data = rank_data(&natural, row_bytes(encoding), &buckets, 128);
                    (
                        table_header(encoding, ROWS, &data, Some(&buckets), 128, &source),
                        data,
                    )
                } else {
                    row_table(encoding, &shards, ROWS, &source)
                };
                write_table(&path, &mut header, rank.then_some(&buckets[..]), &data);

                let file = PleRowsFile::open(&path).unwrap();
                assert_eq!(file.header(), &header);
                assert_eq!(file.path(), path);
                assert_eq!(file.encoding(), encoding);
                assert_eq!(
                    file.order(),
                    if rank {
                        PackOrder::Rank
                    } else {
                        PackOrder::Row
                    }
                );
                assert_eq!(file.shard_count(), SHARDS);
                assert_eq!(file.rows_per_shard(), ROWS);
                assert_eq!(file.row_width(), WIDTH);
                assert_eq!(file.file_bytes(), std::fs::metadata(&path).unwrap().len());
                assert_eq!(file.table().source_rows_sha256, source);
                assert_eq!(file.header().data_offset % 4096, 0);
                if rank {
                    assert_eq!(file.shard_offset(0), None);
                } else {
                    let shard_bytes = (ROWS * row_bytes(encoding)) as u64;
                    assert_eq!(file.shard_offset(0), Some(header.data_offset));
                    assert_eq!(
                        file.shard_offset(2),
                        Some(header.data_offset + 2 * shard_bytes)
                    );
                    assert_eq!(file.shard_offset(SHARDS), None);
                }
            }
        }
    }

    fn mutated_header(mutate: impl FnOnce(&mut PleHotPackHeader)) -> PleHotPackHeader {
        let shards = all_shards(RowEncoding::Q8F16, 1);
        let (mut header, _) = row_table(RowEncoding::Q8F16, &shards, ROWS, &q8_rows_sha(1));
        mutate(&mut header);
        header
    }

    #[test]
    fn table_header_check_rejects_everything_a_sidecar_must_not_be() {
        let good = mutated_header(|_| {});
        let geometry = check_table_header(&good).unwrap();
        assert_eq!(
            (
                geometry.shard_count,
                geometry.rows_per_shard,
                geometry.row_width
            ),
            (SHARDS, ROWS, WIDTH)
        );
        let q4_shards = all_shards(RowEncoding::Q4Lm, 1);
        let (q4, _) = row_table(RowEncoding::Q4Lm, &q4_shards, ROWS, &q8_rows_sha(1));
        check_table_header(&q4).unwrap();

        type Mutation = Box<dyn Fn(&mut PleHotPackHeader)>;
        let cases: Vec<(&str, Mutation)> = vec![
            ("hot-row pack", Box::new(|h| h.kind = PackKind::HotPack)),
            ("no table fields", Box::new(|h| h.table = None)),
            (
                "not a PLE table encoding",
                Box::new(|h| h.row_encoding = RowEncoding::Bf16),
            ),
            (
                "source encoding",
                Box::new(|h| table_mut(h).source_encoding = RowEncoding::Q4Lm),
            ),
            ("valid Q8_0 row width", Box::new(|h| h.row_width = 33)),
            ("encoded_row_bytes", Box::new(|h| h.encoded_row_bytes += 1)),
            ("whole number", Box::new(|h| h.rows_per_shard = 5)),
            ("whole number", Box::new(|h| h.rows_per_shard = 0)),
            ("data_bytes", Box::new(|h| h.data_bytes += 1)),
            (
                "codebook/scale are present",
                Box::new(|h| table_mut(h).codebook_f32_bits = Some(lm4_codebook_bits())),
            ),
            (
                "codebook/scale are present",
                Box::new(|h| table_mut(h).scale_grid_f32_bits = Some(lm4_scale_grid_bits())),
            ),
            (
                "source_rows_sha256",
                Box::new(|h| table_mut(h).source_rows_sha256 = "AB".repeat(32)),
            ),
            (
                "base_index_sha256",
                Box::new(|h| table_mut(h).base_index_sha256 = "ab".to_string()),
            ),
            ("carries a rank section", Box::new(|h| h.rank_bytes = 1)),
            ("rank_bytes", Box::new(|h| h.order = PackOrder::Rank)),
        ];
        for (what, mutate) in cases {
            let header = mutated_header(|h| mutate(h));
            let err = check_table_header(&header).err().unwrap_or_else(|| {
                panic!("expected `{what}` to be refused");
            });
            assert!(err.contains(what), "expected `{what}` in: {err}");
        }
        let mut q4_missing = q4.clone();
        table_mut(&mut q4_missing).codebook_f32_bits = None;
        assert!(check_table_header(&q4_missing)
            .unwrap_err()
            .contains("codebook/scale are absent"));
    }

    #[test]
    fn open_refuses_damaged_files() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.table");
        let shards = all_shards(RowEncoding::Q4Lm, 5);
        let (mut header, data) = row_table(RowEncoding::Q4Lm, &shards, ROWS, &q8_rows_sha(5));
        write_table(&good, &mut header, None, &data);
        PleRowsFile::open(&good).unwrap();
        let bad = dir.path().join("bad.table");
        let bytes = std::fs::read(&good).unwrap();

        let mut flipped = bytes.clone();
        flipped[0] ^= 0xFF;
        std::fs::write(&bad, &flipped).unwrap();
        assert!(PleRowsFile::open(&bad).is_err());
        std::fs::write(&bad, &bytes[..bytes.len() - 1]).unwrap();
        assert!(PleRowsFile::open(&bad).is_err());
        let mut extended = bytes.clone();
        extended.push(0);
        std::fs::write(&bad, &extended).unwrap();
        assert!(PleRowsFile::open(&bad).is_err());
        std::fs::write(&bad, &bytes[..10]).unwrap();
        assert!(PleRowsFile::open(&bad).is_err());
        assert!(PleRowsFile::open(&dir.path().join("missing")).is_err());
    }

    // -- descriptors --------------------------------------------------------

    #[test]
    fn shard_descriptors_read_exact_shard_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.ple-q8");
        let shards = all_shards(RowEncoding::Q8F16, 9);
        let (mut header, data) = row_table(RowEncoding::Q8F16, &shards, ROWS, &merkle_of(&shards));
        write_table(&path, &mut header, None, &data);
        let file = PleRowsFile::open(&path).unwrap();
        let descriptors = file.shard_descriptors().unwrap();
        assert_eq!(descriptors.len(), SHARDS);
        let identity = descriptors[0].source_identity().clone();
        assert_eq!(identity.format, SourceFormat::PleRows);
        assert_eq!(identity.files.len(), 1);
        assert_eq!(
            identity.files[0].canonical_path,
            std::fs::canonicalize(&path).unwrap()
        );
        assert_eq!(identity.files[0].len, file.file_bytes());
        assert_eq!(identity.manifest.len(), SHARDS);
        assert!(identity.metadata_json.contains("\"ple-table\""));
        for (s, descriptor) in descriptors.iter().enumerate() {
            assert_eq!(descriptor.source_identity(), &identity);
            assert_eq!(identity.manifest[s].name, ple_shard_name(s));
            assert_eq!(Some(identity.manifest[s].offset), file.shard_offset(s));
            assert_eq!(identity.manifest[s].offset, descriptor.offset);
            assert_eq!(descriptor.length as usize, ROWS * Q8_ROW);
            assert_eq!(descriptor.dtype(), "Q8_0");
            assert_eq!(descriptor.logical_shape(), [ROWS, WIDTH]);
            let mut buf = vec![0u8; ROWS * Q8_ROW];
            descriptor.read_exact(&mut buf).unwrap();
            assert_eq!(buf, shards[s]);
            // sub-range read at an absolute offset
            let mut row = vec![0u8; Q8_ROW];
            descriptor
                .read_exact_at(descriptor.offset + Q8_ROW as u64, &mut row)
                .unwrap();
            assert_eq!(row, shards[s][Q8_ROW..2 * Q8_ROW]);
        }
        // reads outside the shard are refused even though the file holds them
        let mut one = [0u8; 1];
        assert!(matches!(
            descriptors[0].read_exact_at(descriptors[1].offset, &mut one),
            Err(SourceError::RangeOutOfBounds { .. })
        ));
        assert!(matches!(
            descriptors[1].read_exact_at(descriptors[0].offset, &mut one),
            Err(SourceError::RangeOutOfBounds { .. })
        ));
        // a row-ordered table has no store
        assert!(file
            .open_table_store()
            .err()
            .expect("a row-ordered table has no store")
            .contains("row-ordered"));

        // q4 tables advertise the Q4LM dtype
        let path4 = dir.path().join("m.ple-q4");
        let q4 = all_shards(RowEncoding::Q4Lm, 9);
        let (mut header4, data4) = row_table(RowEncoding::Q4Lm, &q4, ROWS, &merkle_of(&shards));
        write_table(&path4, &mut header4, None, &data4);
        let q4_file = PleRowsFile::open(&path4).unwrap();
        let q4_descriptors = q4_file.shard_descriptors().unwrap();
        assert_eq!(q4_descriptors[1].dtype(), "Q4LM");
        assert_eq!(q4_descriptors[1].length as usize, ROWS * Q4_ROW);
        let mut buf = vec![0u8; ROWS * Q4_ROW];
        q4_descriptors[2].read_exact(&mut buf).unwrap();
        assert_eq!(buf, q4[2]);
    }

    #[test]
    fn rank_ordered_tables_have_no_descriptors_and_memoize_their_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.ple-q4");
        let shards = all_shards(RowEncoding::Q4Lm, 2);
        let buckets = buckets_for(SHARDS * ROWS);
        let data = rank_data(&shards.concat(), Q4_ROW, &buckets, 128);
        let mut header = table_header(
            RowEncoding::Q4Lm,
            ROWS,
            &data,
            Some(&buckets),
            128,
            &q8_rows_sha(2),
        );
        assert_eq!(header.tier_rows.iter().sum::<u64>(), (SHARDS * ROWS) as u64);
        write_table(&path, &mut header, Some(&buckets), &data);
        let file = PleRowsFile::open(&path).unwrap();
        assert_eq!(file.order(), PackOrder::Rank);
        match file.shard_descriptors() {
            Err(SourceError::InvalidSource { reason }) => {
                assert!(reason.contains("rank-ordered"), "{reason}")
            }
            other => panic!("expected InvalidSource, got {other:?}"),
        }
        let store = file.open_table_store().unwrap();
        // memoized: the same handle, a clone, and a repeat all share one store
        assert!(Arc::ptr_eq(&store, &file.open_table_store().unwrap()));
        assert!(Arc::ptr_eq(
            &store,
            &file.clone().open_table_store().unwrap()
        ));
    }

    #[test]
    fn descriptor_reads_detect_file_replacement_and_growth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.ple-q8");
        let shards = all_shards(RowEncoding::Q8F16, 2);
        let (mut header, data) = row_table(RowEncoding::Q8F16, &shards, ROWS, &merkle_of(&shards));
        write_table(&path, &mut header, None, &data);
        let file = PleRowsFile::open(&path).unwrap();
        let descriptors = file.shard_descriptors().unwrap();
        let mut buf = vec![0u8; ROWS * Q8_ROW];
        descriptors[0].read_exact(&mut buf).unwrap();

        // growth
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(matches!(
            descriptors[0].read_exact(&mut buf),
            Err(SourceError::IdentityChanged { .. })
        ));
        // a re-open now fails the strict length check
        assert!(PleRowsFile::open(&path).is_err());

        // replacement by a different inode
        let other = dir.path().join("other");
        std::fs::write(&other, vec![0u8; file.file_bytes() as usize]).unwrap();
        std::fs::rename(&other, &path).unwrap();
        assert!(matches!(
            descriptors[1].read_exact(&mut buf),
            Err(SourceError::IdentityChanged { .. })
        ));
    }

    // -- attach + resolution -------------------------------------------------

    /// A `qwen4_ple` object for the fixture geometry (`padded_rows` =
    /// `SHARDS * ROWS`), which derive needs for the table fingerprint.
    fn qwen4_ple_json() -> String {
        format!(
            "\"qwen4_ple\":{{\"version\":1,\"multipliers\":[-3,5,7],\"head_vocab_sizes\":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\"head_offsets\":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\"padded_rows\":{}}}",
            SHARDS * ROWS
        )
    }

    fn marker_json(q8_rows_sha256: &str) -> String {
        format!(
            "{{{},\"{PLE_BASE_MARKER_KEY}\":{{\"version\":1,\"location\":\"sidecar\",\"shard_count\":{SHARDS},\"rows_per_shard\":{ROWS},\"row_width\":{WIDTH},\"q8_rows_sha256\":\"{q8_rows_sha256}\"}}}}",
            qwen4_ple_json()
        )
    }

    fn i64_tensor(name: &str) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: 54,
            shape: vec![3],
            group_size: 0,
            data: (0u8..24).collect(),
        }
    }

    /// Monolithic base: the PLE shards embedded as qt 3 `[ROWS, WIDTH]`.
    fn write_monolithic(path: &Path, q8_shards: &[Vec<u8>]) {
        let mut tensors: Vec<HfqMemTensor> = q8_shards
            .iter()
            .enumerate()
            .map(|(s, data)| HfqMemTensor {
                name: ple_shard_name(s),
                quant_type: Q8_QUANT_TYPE,
                shape: vec![ROWS as u32, WIDTH as u32],
                group_size: 32,
                data: data.clone(),
            })
            .collect();
        tensors.push(i64_tensor(
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        ));
        write_hfqm_package_mem(path, 16, &format!("{{{}}}", qwen4_ple_json()), &tensors).unwrap();
    }

    /// Split base: no shards, the marker in the metadata.
    fn write_split(path: &Path, q8_rows_sha256: &str) {
        let tensors = [i64_tensor(
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        )];
        write_hfqm_package_mem(path, 16, &marker_json(q8_rows_sha256), &tensors).unwrap();
    }

    fn policy(derive: bool) -> ResolvePolicy<'static> {
        ResolvePolicy {
            derive_missing_q4: derive,
            progress: None,
        }
    }

    fn sidecar_of(base: &Path, format: PleFormat) -> PathBuf {
        ple_sidecar_path(base, format)
    }

    #[test]
    fn attach_binds_tables_to_split_and_monolithic_bases() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m-base.hfqm");
        let sha = q8_rows_sha(1);
        write_split(&base, &sha);
        let q8 = sidecar_of(&base, PleFormat::Q8);
        let header = write_bound_sidecar(&q8, &base, RowEncoding::Q8F16, &sha, 1);
        assert_eq!(header.table.as_ref().unwrap().source_rows_sha256, sha);

        let mut hfq = HfqFile::open(&base).unwrap();
        let marker = hfq.ple_rows_marker().unwrap();
        assert_eq!(marker["q8_rows_sha256"], sha.as_str());
        assert_eq!(marker["shard_count"], SHARDS);
        assert!(hfq.ple_rows().is_none());
        hfq.attach_ple_rows(PleRowsFile::open(&q8).unwrap())
            .unwrap();
        assert_eq!(hfq.ple_rows().unwrap().path(), q8);
        // one attach only
        let err = hfq
            .attach_ple_rows(PleRowsFile::open(&q8).unwrap())
            .unwrap_err();
        assert!(err.contains("already"), "{err}");
        // the effective source identity is untouched
        let identity = hfq.source_identity().unwrap();
        assert_eq!(identity.format, SourceFormat::Hfq);
        assert!(identity
            .manifest
            .iter()
            .all(|m| !m.name.contains("ngram_embedding")));

        // a q4 table binds to a monolithic base
        let mono = dir.path().join("m.hfqm");
        write_monolithic(&mono, &all_shards(RowEncoding::Q8F16, 4));
        let q4 = sidecar_of(&mono, PleFormat::Q4);
        write_bound_sidecar(&q4, &mono, RowEncoding::Q4Lm, &q8_rows_sha(4), 4);
        let mut hfq = HfqFile::open(&mono).unwrap();
        assert!(hfq.ple_rows_marker().is_none());
        hfq.attach_ple_rows(PleRowsFile::open(&q4).unwrap())
            .unwrap();
        assert_eq!(hfq.ple_rows().unwrap().encoding(), RowEncoding::Q4Lm);
    }

    #[test]
    fn rank_ordered_tables_bind_like_row_ordered_ones() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m.hfqm");
        write_monolithic(&base, &all_shards(RowEncoding::Q8F16, 4));
        let q4 = sidecar_of(&base, PleFormat::Q4);
        let shards = all_shards(RowEncoding::Q4Lm, 4);
        let buckets = buckets_for(SHARDS * ROWS);
        let data = rank_data(&shards.concat(), Q4_ROW, &buckets, 128);
        let mut header = table_header(
            RowEncoding::Q4Lm,
            ROWS,
            &data,
            Some(&buckets),
            128,
            &q8_rows_sha(4),
        );
        bind(&mut header, &HfqFile::open(&base).unwrap());
        write_table(&q4, &mut header, Some(&buckets), &data);
        let mut hfq = HfqFile::open(&base).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(false),
        )
        .unwrap();
        assert_eq!(choice.origin, PleRowsOrigin::Sidecar(q4));
        assert_eq!(hfq.ple_rows().unwrap().order(), PackOrder::Rank);
    }

    #[test]
    fn attach_rejects_every_binding_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m-base.hfqm");
        let sha = q8_rows_sha(1);
        write_split(&base, &sha);
        write_bound_sidecar(
            &dir.path().join("side.ple-q8"),
            &base,
            RowEncoding::Q8F16,
            &sha,
            1,
        );
        let bad = dir.path().join("bad.ple-q8");
        let attach = |mutate: &dyn Fn(&mut PleHotPackHeader), rows_per_shard: usize| {
            let hfq = HfqFile::open(&base).unwrap();
            write_mutated_sidecar(&bad, &hfq, RowEncoding::Q8F16, rows_per_shard, &sha, |h| {
                mutate(h)
            });
            HfqFile::open(&base)
                .unwrap()
                .attach_ple_rows(PleRowsFile::open(&bad).unwrap())
        };
        attach(&|_| {}, ROWS).unwrap();

        // a split base is bound by row identity, not file bytes: a different
        // index sha or file length does not refuse it
        attach(&|h| table_mut(h).base_index_sha256 = "cd".repeat(32), ROWS).unwrap();
        attach(&|h| table_mut(h).base_file_bytes += 1, ROWS).unwrap();

        // marker sha differs from the table's source digest
        let err = attach(&|h| table_mut(h).source_rows_sha256 = "ef".repeat(32), ROWS).unwrap_err();
        assert!(err.contains("efefefefefef"), "{err}");
        assert!(err.contains(short_hex(&sha)), "{err}");

        // geometry differs from the marker
        let err = attach(&|_| {}, ROWS + 1).unwrap_err();
        assert!(err.contains("rows_per_shard"), "{err}");

        // a q4 table built from different rows mentions the derive remedy
        let q4_path = dir.path().join("bad.ple-q4");
        let hfq = HfqFile::open(&base).unwrap();
        write_mutated_sidecar(&q4_path, &hfq, RowEncoding::Q4Lm, ROWS, &sha, |h| {
            table_mut(h).source_rows_sha256 = "ef".repeat(32)
        });
        let err = HfqFile::open(&base)
            .unwrap()
            .attach_ple_rows(PleRowsFile::open(&q4_path).unwrap())
            .unwrap_err();
        assert!(err.contains("hipfire ple derive --format q4"), "{err}");

        // a base without marker and without shards
        let plain = dir.path().join("plain.hfqm");
        write_hfqm_package_mem(&plain, 16, "{}", &[i64_tensor("t")]).unwrap();
        let plain_side = dir.path().join("plain.ple-q8");
        write_bound_sidecar(&plain_side, &plain, RowEncoding::Q8F16, &sha, 1);
        let err = HfqFile::open(&plain)
            .unwrap()
            .attach_ple_rows(PleRowsFile::open(&plain_side).unwrap())
            .unwrap_err();
        assert!(
            err.contains("neither embedded PLE shard tensors nor"),
            "{err}"
        );
        assert!(err.contains(PLE_BASE_MARKER_KEY), "{err}");
    }

    /// A split base carrying `marker` and an extra tensor + metadata key, so its
    /// file length and index differ from every other base built with a
    /// different `extra`.
    fn write_split_variant(path: &Path, marker: &str, extra: usize) {
        let mut tensors = vec![i64_tensor(
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        )];
        for i in 0..extra {
            tensors.push(i64_tensor(&format!("extra.{i}")));
        }
        let metadata =
            marker.replacen('{', &format!("{{\"pad\":\"{}\",", "x".repeat(extra * 7)), 1);
        write_hfqm_package_mem(path, 16, &metadata, &tensors).unwrap();
    }

    #[test]
    fn one_table_binds_two_split_bases_of_different_length_and_index() {
        let dir = tempfile::tempdir().unwrap();
        let sha = q8_rows_sha(1);
        let marker = marker_json(&sha);
        let a = dir.path().join("a-base.hfqm");
        let b = dir.path().join("b-base.hfqm");
        write_split_variant(&a, &marker, 0);
        write_split_variant(&b, &marker, 3);
        let (ha, hb) = (HfqFile::open(&a).unwrap(), HfqFile::open(&b).unwrap());
        assert_ne!(ha.file_len(), hb.file_len());
        assert_ne!(ha.index_sha256().unwrap(), hb.index_sha256().unwrap());

        // one shared table, bound (for its header fields only) to base `a`
        let shared = dir.path().join("shared.ple-q8");
        let header = write_bound_sidecar(&shared, &a, RowEncoding::Q8F16, &sha, 1);
        assert_eq!(
            header.table.as_ref().unwrap().base_file_bytes,
            ha.file_len()
        );

        for base in [&a, &b] {
            let mut hfq = HfqFile::open(base).unwrap();
            hfq.attach_ple_rows(PleRowsFile::open(&shared).unwrap())
                .unwrap();
            assert_eq!(hfq.ple_rows().unwrap().path(), shared);
        }

        // a base marked for different rows (R') refuses the same table
        let other = dir.path().join("other-base.hfqm");
        write_split_variant(&other, &marker_json(&q8_rows_sha(2)), 1);
        let err = HfqFile::open(&other)
            .unwrap()
            .attach_ple_rows(PleRowsFile::open(&shared).unwrap())
            .unwrap_err();
        assert!(err.contains("was split from q8 rows"), "{err}");
        assert!(err.contains(short_hex(&q8_rows_sha(2))), "{err}");
        assert!(err.contains(short_hex(&sha)), "{err}");

        // a base marked R with different geometry refuses it
        for (from, to) in [
            (
                format!("\"row_width\":{WIDTH}"),
                format!("\"row_width\":{}", WIDTH + 32),
            ),
            (
                format!("\"rows_per_shard\":{ROWS}"),
                format!("\"rows_per_shard\":{}", ROWS + 1),
            ),
            (
                format!("\"shard_count\":{SHARDS}"),
                format!("\"shard_count\":{}", SHARDS + 1),
            ),
        ] {
            let skewed = dir.path().join("skewed-base.hfqm");
            let json = marker.replace(&from, &to);
            assert_ne!(json, marker, "{from}");
            write_split_variant(&skewed, &json, 2);
            let err = HfqFile::open(&skewed)
                .unwrap()
                .attach_ple_rows(PleRowsFile::open(&shared).unwrap())
                .unwrap_err();
            let field = from.split('"').nth(1).unwrap();
            assert!(err.contains(field) && err.contains("declares"), "{err}");
        }
    }

    #[test]
    fn monolithic_attach_still_requires_file_length_and_index() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m.hfqm");
        write_monolithic(&base, &all_shards(RowEncoding::Q8F16, 6));
        let bad = dir.path().join("m.ple-q8");
        let attach = |mutate: &dyn Fn(&mut PleHotPackHeader)| {
            let hfq = HfqFile::open(&base).unwrap();
            write_mutated_sidecar(&bad, &hfq, RowEncoding::Q8F16, ROWS, &q8_rows_sha(6), |h| {
                mutate(h)
            });
            HfqFile::open(&base)
                .unwrap()
                .attach_ple_rows(PleRowsFile::open(&bad).unwrap())
        };
        attach(&|_| {}).unwrap();

        let err = attach(&|h| table_mut(h).base_file_bytes += 1).unwrap_err();
        assert!(err.contains("bytes"), "{err}");
        assert!(err.contains(bad.to_str().unwrap()), "{err}");

        let err = attach(&|h| table_mut(h).base_index_sha256 = "cd".repeat(32)).unwrap_err();
        assert!(err.contains("cdcdcdcdcdcd"), "{err}");
        assert!(err.contains("index"), "{err}");
        assert!(err.contains("pull"), "{err}");

        // same length, one flipped byte inside the index padding: index sha differs
        let side = dir.path().join("side.ple-q8");
        write_bound_sidecar(&side, &base, RowEncoding::Q8F16, &q8_rows_sha(1), 1);
        let data_offset = HfqFile::open(&base).unwrap().data_offset() as usize;
        let mut raw = std::fs::read(&base).unwrap();
        raw[data_offset - 1] ^= 0x01;
        std::fs::write(&base, raw).unwrap();
        let err = HfqFile::open(&base)
            .unwrap()
            .attach_ple_rows(PleRowsFile::open(&side).unwrap())
            .unwrap_err();
        assert!(err.contains("index"), "{err}");
    }

    #[test]
    fn absent_marker_still_refuses_even_with_matching_length_and_index() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("plain.hfqm");
        write_hfqm_package_mem(&plain, 16, "{}", &[i64_tensor("t")]).unwrap();
        // bound to this very file's length and index: only the missing marker is wrong
        let side = dir.path().join("plain.ple-q8");
        write_bound_sidecar(&side, &plain, RowEncoding::Q8F16, &q8_rows_sha(1), 1);
        let err = HfqFile::open(&plain)
            .unwrap()
            .attach_ple_rows(PleRowsFile::open(&side).unwrap())
            .unwrap_err();
        assert!(
            err.contains("neither embedded PLE shard tensors nor"),
            "{err}"
        );
        assert!(err.contains(PLE_BASE_MARKER_KEY), "{err}");

        // a table bound to some other marked split base does not help
        let marked = dir.path().join("marked-base.hfqm");
        write_split(&marked, &q8_rows_sha(1));
        let shared = dir.path().join("shared.ple-q8");
        write_bound_sidecar(&shared, &marked, RowEncoding::Q8F16, &q8_rows_sha(1), 1);
        let err = HfqFile::open(&plain)
            .unwrap()
            .attach_ple_rows(PleRowsFile::open(&shared).unwrap())
            .unwrap_err();
        assert!(
            err.contains("neither embedded PLE shard tensors nor"),
            "{err}"
        );
    }

    #[test]
    fn attach_checks_embedded_shard_geometry_and_marker_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m.hfqm");
        let shards = all_shards(RowEncoding::Q8F16, 6);
        write_monolithic(&base, &shards);
        let path = dir.path().join("m.ple-q4");
        let hfq = HfqFile::open(&base).unwrap();

        // 3 shards of ROWS+1 rows against embedded [ROWS, WIDTH]
        write_mutated_sidecar(
            &path,
            &hfq,
            RowEncoding::Q4Lm,
            ROWS + 1,
            &q8_rows_sha(6),
            |_| {},
        );
        let mut hfq = HfqFile::open(&base).unwrap();
        let err = hfq
            .attach_ple_rows(PleRowsFile::open(&path).unwrap())
            .unwrap_err();
        assert!(
            err.contains(&ple_shard_name(0)) || err.contains("shape"),
            "{err}"
        );

        // a table of 2 shards against 3 embedded
        let hfq = HfqFile::open(&base).unwrap();
        let two: Vec<Vec<u8>> = (0..2)
            .map(|s| shard_rows(RowEncoding::Q4Lm, s, 6, ROWS))
            .collect();
        let (mut header, data) = row_table(RowEncoding::Q4Lm, &two, ROWS, &q8_rows_sha(6));
        bind(&mut header, &hfq);
        write_table(&path, &mut header, None, &data);
        let mut hfq = HfqFile::open(&base).unwrap();
        let err = hfq
            .attach_ple_rows(PleRowsFile::open(&path).unwrap())
            .unwrap_err();
        assert!(err.contains("shard"), "{err}");

        // marker and embedded shards at once is inconsistent
        let both = dir.path().join("both.hfqm");
        let tensors = vec![HfqMemTensor {
            name: ple_shard_name(0),
            quant_type: Q8_QUANT_TYPE,
            shape: vec![ROWS as u32, WIDTH as u32],
            group_size: 0,
            data: shards[0].clone(),
        }];
        write_hfqm_package_mem(&both, 16, &marker_json(&q8_rows_sha(6)), &tensors).unwrap();
        let hfq = HfqFile::open(&both).unwrap();
        assert!(classify_base(&hfq).is_err());
    }

    #[test]
    fn embedded_shard_count_requires_dense_canonical_names() {
        let info = |name: String| HfqTensorInfo {
            name,
            quant_type: Q8_QUANT_TYPE,
            shape: vec![1, 32],
            group_size: 0,
            data_offset: 0,
            data_size: 0,
        };
        let dense: Vec<_> = (0..3).map(|s| info(ple_shard_name(s))).collect();
        assert_eq!(embedded_shard_count(&dense).unwrap(), 3);
        assert_eq!(embedded_shard_count(&[]).unwrap(), 0);
        let gap = [info(ple_shard_name(0)), info(ple_shard_name(2))];
        assert!(embedded_shard_count(&gap).is_err());
        let padded = [info(format!("{PLE_SHARD_PREFIX}01{PLE_SHARD_SUFFIX}"))];
        assert!(embedded_shard_count(&padded).is_err());
        assert_eq!(
            embedded_shard_count(&[info("other".to_string())]).unwrap(),
            0
        );
    }

    #[test]
    fn marker_parsing_is_strict() {
        let good: serde_json::Value = serde_json::from_str(&marker_json(&"ab".repeat(32))).unwrap();
        let marker = parse_base_marker(&good[PLE_BASE_MARKER_KEY]).unwrap();
        assert_eq!(marker.shard_count, SHARDS);
        assert_eq!(marker.rows_per_shard, ROWS);
        assert_eq!(marker.row_width, WIDTH);
        for (key, value) in [
            ("version", serde_json::json!(2)),
            ("location", serde_json::json!("embedded")),
            ("shard_count", serde_json::json!("3")),
            ("q8_rows_sha256", serde_json::json!("AB")),
        ] {
            let mut bad = good[PLE_BASE_MARKER_KEY].clone();
            bad[key] = value;
            assert!(parse_base_marker(&bad).is_err(), "{key}");
        }
        let mut missing = good[PLE_BASE_MARKER_KEY].clone();
        missing.as_object_mut().unwrap().remove("row_width");
        assert!(parse_base_marker(&missing).is_err());
        assert!(parse_base_marker(&serde_json::json!(1)).is_err());
    }

    #[test]
    fn resolve_monolithic_follows_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m.hfqm");
        write_monolithic(&base, &all_shards(RowEncoding::Q8F16, 4));
        let q4 = sidecar_of(&base, PleFormat::Q4);

        // q8: embedded rows, no sidecar touched
        let mut hfq = HfqFile::open(&base).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Q8,
            "gfx1100",
            policy(true),
        )
        .unwrap();
        assert_eq!(
            choice,
            PleRowsChoice {
                format: PleFormat::Q8,
                origin: PleRowsOrigin::Embedded,
                derived: false,
                note: None
            }
        );
        assert!(hfq.ple_rows().is_none());

        // q4 without the file and without derivation: refuse naming both remedies
        let err = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(false),
        )
        .unwrap_err();
        assert!(err.contains(q4.to_str().unwrap()), "{err}");
        assert!(err.contains("hipfire ple derive --format q4"), "{err}");
        assert!(err.contains("memory.ple_format"), "{err}");
        assert!(hfq.ple_rows().is_none());

        // auto never derives: preferred q4 (gfx1151) falls back to embedded q8 with a note
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Auto,
            "gfx1151",
            policy(true),
        )
        .unwrap();
        assert_eq!(choice.format, PleFormat::Q8);
        assert_eq!(choice.origin, PleRowsOrigin::Embedded);
        assert!(!choice.derived);
        let note = choice.note.unwrap();
        assert!(
            note.contains(q4.to_str().unwrap()) && note.contains("hipfire ple derive"),
            "{note}"
        );
        assert!(!q4.exists());
        // auto on another arch: q8 preferred, silently
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Auto,
            "gfx1100",
            policy(true),
        )
        .unwrap();
        assert_eq!(
            (choice.format, choice.origin, choice.note),
            (PleFormat::Q8, PleRowsOrigin::Embedded, None)
        );

        // with the sidecar present: q4 and auto(gfx1151) bind it
        write_bound_sidecar(&q4, &base, RowEncoding::Q4Lm, &q8_rows_sha(4), 4);
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(false),
        )
        .unwrap();
        assert_eq!(
            choice,
            PleRowsChoice {
                format: PleFormat::Q4,
                origin: PleRowsOrigin::Sidecar(q4.clone()),
                derived: false,
                note: None
            }
        );
        assert_eq!(hfq.ple_rows().unwrap().path(), q4);
        // idempotent: reported, not reopened; an explicit conflicting format errors
        let again = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Auto,
            "gfx1100",
            policy(false),
        )
        .unwrap();
        assert_eq!(again.origin, PleRowsOrigin::Sidecar(q4.clone()));
        assert_eq!(again.format, PleFormat::Q4);
        let err = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Q8,
            "gfx1100",
            policy(false),
        )
        .unwrap_err();
        assert!(err.contains("already attached"), "{err}");

        let mut hfq = HfqFile::open(&base).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Auto,
            "gfx1151",
            policy(false),
        )
        .unwrap();
        assert_eq!((choice.format, choice.note), (PleFormat::Q4, None));
        // auto on a non-q4 arch keeps the embedded q8 rows even with a q4 file present
        let mut hfq = HfqFile::open(&base).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Auto,
            "gfx1100",
            policy(false),
        )
        .unwrap();
        assert_eq!(choice.origin, PleRowsOrigin::Embedded);
        assert!(hfq.ple_rows().is_none());

        // a present sidecar that does not bind is a hard error, never a fallback
        let mut raw = std::fs::read(&q4).unwrap();
        let offset = raw.len() - 1;
        raw.truncate(offset);
        std::fs::write(&q4, raw).unwrap();
        for request in [PleFormatRequest::Q4, PleFormatRequest::Auto] {
            let mut hfq = HfqFile::open(&base).unwrap();
            let err =
                resolve_and_attach_ple_rows(&mut hfq, &base, request, "gfx1151", policy(true))
                    .unwrap_err();
            assert!(err.contains(q4.to_str().unwrap()), "{err}");
        }
    }

    #[test]
    fn resolve_rejects_sidecar_with_the_wrong_encoding_for_its_name() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m-base.hfqm");
        let sha = q8_rows_sha(2);
        write_split(&base, &sha);
        // a q8 table sitting where the q4 sidecar belongs
        write_bound_sidecar(
            &sidecar_of(&base, PleFormat::Q4),
            &base,
            RowEncoding::Q8F16,
            &sha,
            2,
        );
        let mut hfq = HfqFile::open(&base).unwrap();
        let err = resolve_and_attach_ple_rows(
            &mut hfq,
            &base,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(false),
        )
        .unwrap_err();
        assert!(err.contains("Q8_0") && err.contains("Q4LM"), "{err}");
    }

    #[test]
    fn validation_split_q4_policy_refuses_without_deriving() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("validation-base.hfqm");
        let sha = q8_rows_sha(2);
        write_split(&base, &sha);
        let q8 = sidecar_of(&base, PleFormat::Q8);
        let q4 = sidecar_of(&base, PleFormat::Q4);
        write_bound_sidecar(&q8, &base, RowEncoding::Q8F16, &sha, 2);
        let mut hfq = HfqFile::open(&base).unwrap();
        let error = attach_validation_ple_rows(&mut hfq, &base, PleFormatRequest::Q4, "gfx1201")
            .unwrap_err();
        assert!(
            !q4.exists(),
            "validation must never derive a missing q4 table"
        );
        assert!(error.contains(q4.to_str().unwrap()));
        assert!(hfq.ple_rows().is_none());
        write_bound_sidecar(&q4, &base, RowEncoding::Q4Lm, &sha, 2);
        let choice =
            attach_validation_ple_rows(&mut hfq, &base, PleFormatRequest::Q4, "gfx1201").unwrap();
        assert_eq!(
            choice,
            PleRowsChoice {
                format: PleFormat::Q4,
                origin: PleRowsOrigin::Sidecar(q4),
                derived: false,
                note: None,
            }
        );
        assert!(hfq.ple_rows().is_some());
        let mut replay = HfqFile::open(&base).unwrap();
        let replay_choice = resolve_and_attach_ple_rows(
            &mut replay,
            &base,
            PleFormatRequest::Q4,
            "gfx1201",
            ResolvePolicy {
                derive_missing_q4: false,
                progress: None,
            },
        )
        .unwrap();
        assert_eq!(choice, replay_choice);
    }

    #[test]
    fn resolve_split_base_follows_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("qwen3.8-flash-next-base.mq4-xts");
        let sha = q8_rows_sha(2);
        write_split(&base, &sha);
        let q8 = sidecar_of(&base, PleFormat::Q8);
        let q4 = sidecar_of(&base, PleFormat::Q4);
        let open = || HfqFile::open(&base).unwrap();
        let resolve = |hfq: &mut HfqFile, request, arch| {
            resolve_and_attach_ple_rows(hfq, &base, request, arch, policy(false))
        };

        // nothing present: every request refuses with the pull hint (name + both remedies)
        let err = resolve(&mut open(), PleFormatRequest::Q8, "gfx1100").unwrap_err();
        assert!(err.contains(q8.to_str().unwrap()), "{err}");
        assert!(err.contains("hipfire pull <model tag> --ple q8"), "{err}");
        assert!(err.contains("qwen3.8-flash-next-base.mq4-xts"), "{err}");
        assert!(err.contains("54.4 GB"), "{err}");
        assert!(err.contains("memory.ple_format = \"q4\""), "{err}");
        let err = resolve(&mut open(), PleFormatRequest::Q4, "gfx1100").unwrap_err();
        assert!(err.contains(q4.to_str().unwrap()), "{err}");
        assert!(err.contains("hipfire pull <model tag> --ple q4"), "{err}");
        assert!(err.contains("28.8 GB"), "{err}");
        assert!(err.contains("memory.ple_format = \"q8\""), "{err}");
        for arch in ["gfx1100", "gfx1151"] {
            let err = resolve(&mut open(), PleFormatRequest::Auto, arch).unwrap_err();
            assert!(
                err.contains(q8.to_str().unwrap()) && err.contains(q4.to_str().unwrap()),
                "{err}"
            );
            assert!(
                err.contains("--ple q4") && err.contains("--ple q8"),
                "{err}"
            );
            assert!(
                err.contains("hipfire ple derive --format q4")
                    && err.contains("q8 file is present"),
                "{err}"
            );
        }

        // q8 present
        write_bound_sidecar(&q8, &base, RowEncoding::Q8F16, &sha, 2);
        let mut hfq = open();
        let choice = resolve(&mut hfq, PleFormatRequest::Q8, "gfx1151").unwrap();
        assert_eq!(
            choice,
            PleRowsChoice {
                format: PleFormat::Q8,
                origin: PleRowsOrigin::Sidecar(q8.clone()),
                derived: false,
                note: None
            }
        );
        // q4 refused without derivation, naming the derive remedy; q8 stays unattached
        let mut hfq = open();
        let err = resolve(&mut hfq, PleFormatRequest::Q4, "gfx1100").unwrap_err();
        assert!(
            err.contains(q4.to_str().unwrap()) && err.contains("hipfire ple derive --format q4"),
            "{err}"
        );
        assert!(hfq.ple_rows().is_none());
        // auto: preferred q8 on gfx1100 silently; on gfx1151 falls back to q8 with a note
        let choice = resolve(&mut open(), PleFormatRequest::Auto, "gfx1100").unwrap();
        assert_eq!((choice.format, choice.note), (PleFormat::Q8, None));
        let choice = resolve(&mut open(), PleFormatRequest::Auto, "gfx1151").unwrap();
        assert_eq!(choice.format, PleFormat::Q8);
        assert!(!choice.derived);
        let note = choice.note.unwrap();
        assert!(
            note.contains("hipfire ple derive") && note.contains("--ple q4"),
            "{note}"
        );
        assert!(!q4.exists(), "auto must never derive");

        // q4 present too: both preferences resolve without a note
        write_bound_sidecar(&q4, &base, RowEncoding::Q4Lm, &sha, 2);
        let choice = resolve(&mut open(), PleFormatRequest::Auto, "gfx1151").unwrap();
        assert_eq!((choice.format, choice.note), (PleFormat::Q4, None));
        let choice = resolve(&mut open(), PleFormatRequest::Auto, "gfx1100").unwrap();
        assert_eq!((choice.format, choice.note), (PleFormat::Q8, None));
        let choice = resolve(&mut open(), PleFormatRequest::Q4, "gfx1100").unwrap();
        assert_eq!(choice.origin, PleRowsOrigin::Sidecar(q4.clone()));

        // only q4 present: auto(gfx1100) falls back to q4 with a pull hint for q8
        std::fs::remove_file(&q8).unwrap();
        let choice = resolve(&mut open(), PleFormatRequest::Auto, "gfx1100").unwrap();
        assert_eq!(choice.format, PleFormat::Q4);
        let note = choice.note.unwrap();
        assert!(note.contains("hipfire pull <model tag> --ple q8"), "{note}");
        let err = resolve(&mut open(), PleFormatRequest::Q8, "gfx1100").unwrap_err();
        assert!(err.contains(q8.to_str().unwrap()), "{err}");

        // a present sidecar whose source digest differs from the base marker is a
        // hard error for every request that reaches it
        let other = dir.path().join("other-base.hfqm");
        write_split(&other, &"77".repeat(32));
        let other_q8 = sidecar_of(&other, PleFormat::Q8);
        write_bound_sidecar(&other_q8, &other, RowEncoding::Q8F16, &q8_rows_sha(2), 2);
        for request in [PleFormatRequest::Q8, PleFormatRequest::Auto] {
            let mut hfq = HfqFile::open(&other).unwrap();
            let err =
                resolve_and_attach_ple_rows(&mut hfq, &other, request, "gfx1100", policy(false))
                    .unwrap_err();
            assert!(err.contains(other_q8.to_str().unwrap()), "{err}");
            assert!(err.contains("777777777777"), "{err}");
            assert!(hfq.ple_rows().is_none());
        }
    }

    #[test]
    fn resolve_refuses_a_file_that_is_not_a_ple_model() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("plain.hfqm");
        write_hfqm_package_mem(&plain, 16, "{}", &[i64_tensor("t")]).unwrap();
        let mut hfq = HfqFile::open(&plain).unwrap();
        let err = resolve_and_attach_ple_rows(
            &mut hfq,
            &plain,
            PleFormatRequest::Q8,
            "gfx1100",
            policy(false),
        )
        .unwrap_err();
        assert!(err.contains("not a PLE model"), "{err}");
    }

    #[test]
    fn resolve_derives_missing_q4_on_both_base_kinds() {
        // Monolithic: derived from the embedded rows.
        let dir = tempfile::tempdir().unwrap();
        let mono = dir.path().join("m.hfqm");
        write_monolithic(&mono, &all_shards(RowEncoding::Q8F16, 4));
        let mut phases = Vec::new();
        let mut progress = |p: &PleProgress| phases.push(p.phase);
        let mut hfq = HfqFile::open(&mono).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &mono,
            PleFormatRequest::Q4,
            "gfx1100",
            ResolvePolicy {
                derive_missing_q4: true,
                progress: Some(&mut progress),
            },
        )
        .unwrap();
        assert!(choice.derived);
        assert_eq!(choice.format, PleFormat::Q4);
        assert_eq!(
            choice.origin,
            PleRowsOrigin::Sidecar(sidecar_of(&mono, PleFormat::Q4))
        );
        assert!(!phases.is_empty());
        assert_eq!(hfq.ple_rows().unwrap().encoding(), RowEncoding::Q4Lm);
        // a second resolve on a fresh handle finds it, not deriving again
        let mut hfq = HfqFile::open(&mono).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &mono,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(true),
        )
        .unwrap();
        assert!(!choice.derived);

        // Split base: derived from <base>.ple-q8; refused without it even when allowed.
        let split = dir.path().join("m-base.hfqm");
        let sha = q8_rows_sha(8);
        write_split(&split, &sha);
        let mut hfq = HfqFile::open(&split).unwrap();
        let err = resolve_and_attach_ple_rows(
            &mut hfq,
            &split,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(true),
        )
        .unwrap_err();
        assert!(err.contains("--ple q4"), "{err}");
        write_bound_sidecar(
            &sidecar_of(&split, PleFormat::Q8),
            &split,
            RowEncoding::Q8F16,
            &sha,
            8,
        );
        let mut hfq = HfqFile::open(&split).unwrap();
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            &split,
            PleFormatRequest::Q4,
            "gfx1100",
            policy(true),
        )
        .unwrap();
        assert!(choice.derived);
        assert_eq!(hfq.ple_rows().unwrap().encoding(), RowEncoding::Q4Lm);
    }

    /// `<tmp>/models/<name>` is a symlink to `<tmp>/store/<name>`: the layout
    /// of a models directory that points at an artifact kept elsewhere.
    #[cfg(unix)]
    struct Linked {
        _dir: tempfile::TempDir,
        /// The symlink, as a user selects it.
        link: PathBuf,
        /// Its canonical target.
        real: PathBuf,
    }

    #[cfg(unix)]
    fn linked_base(name: &str, write: impl FnOnce(&Path)) -> Linked {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        let models = dir.path().join("models");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        let target = store.join(name);
        write(&target);
        let link = models.join(name);
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let real = std::fs::canonicalize(&link).unwrap();
        assert_ne!(real, link, "fixture must be a real symlink");
        Linked {
            _dir: dir,
            link,
            real,
        }
    }

    #[cfg(unix)]
    fn dir_entries(path: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[cfg(unix)]
    fn resolve_at(
        base: &Path,
        request: PleFormatRequest,
        arch: &str,
        derive: bool,
    ) -> (HfqFile, Result<PleRowsChoice, String>) {
        let mut hfq = HfqFile::open(base).unwrap();
        let result = resolve_and_attach_ple_rows(&mut hfq, base, request, arch, policy(derive));
        (hfq, result)
    }

    #[cfg(unix)]
    #[test]
    fn monolithic_symlink_prefers_the_sidecar_beside_the_link() {
        let linked = linked_base("m.hfqm", |p| {
            write_monolithic(p, &all_shards(RowEncoding::Q8F16, 4))
        });
        let link_q4 = sidecar_of(&linked.link, PleFormat::Q4);
        let real_q4 = sidecar_of(&linked.real, PleFormat::Q4);
        let sha = q8_rows_sha(4);
        write_bound_sidecar(&link_q4, &linked.real, RowEncoding::Q4Lm, &sha, 4);
        // The canonical target has its own, different, bound sidecar: the one
        // beside the selected path must still win.
        write_bound_sidecar(&real_q4, &linked.real, RowEncoding::Q4Lm, &sha, 9);
        let real_before = std::fs::read(&real_q4).unwrap();

        // Explicit API consumers may allow derivation; an existing sidecar
        // must not be derived over or replaced regardless.
        for derive in [false, true] {
            let (hfq, choice) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", derive);
            let choice = choice.unwrap();
            assert_eq!(choice.origin, PleRowsOrigin::Sidecar(link_q4.clone()));
            assert!(!choice.derived);
            assert_eq!(hfq.ple_rows().unwrap().path(), link_q4);
        }
        let (hfq, choice) = resolve_at(&linked.link, PleFormatRequest::Auto, "gfx1151", false);
        assert_eq!(
            choice.unwrap().origin,
            PleRowsOrigin::Sidecar(link_q4.clone())
        );
        assert_eq!(hfq.ple_rows().unwrap().path(), link_q4);
        assert_eq!(std::fs::read(&real_q4).unwrap(), real_before);
    }

    #[cfg(unix)]
    #[test]
    fn monolithic_symlink_falls_back_to_the_sidecar_beside_the_canonical_target() {
        let linked = linked_base("m.hfqm", |p| {
            write_monolithic(p, &all_shards(RowEncoding::Q8F16, 4))
        });
        let link_q4 = sidecar_of(&linked.link, PleFormat::Q4);
        let real_q4 = sidecar_of(&linked.real, PleFormat::Q4);
        write_bound_sidecar(
            &real_q4,
            &linked.real,
            RowEncoding::Q4Lm,
            &q8_rows_sha(4),
            4,
        );

        let (hfq, choice) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", false);
        let choice = choice.unwrap();
        assert_eq!(choice.origin, PleRowsOrigin::Sidecar(real_q4.clone()));
        assert!(!choice.derived);
        assert_eq!(hfq.ple_rows().unwrap().path(), real_q4);
        assert!(!link_q4.exists(), "nothing may be created beside the link");

        // A dangling sidecar link beside the selected path is "not found",
        // not an error: the canonical lookup still runs.
        std::os::unix::fs::symlink(linked.real.with_extension("nowhere"), &link_q4).unwrap();
        let (_, choice) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", false);
        assert_eq!(choice.unwrap().origin, PleRowsOrigin::Sidecar(real_q4));
    }

    #[cfg(unix)]
    #[test]
    fn missing_q4_is_refused_naming_both_locations_and_never_derived() {
        let linked = linked_base("m.hfqm", |p| {
            write_monolithic(p, &all_shards(RowEncoding::Q8F16, 4))
        });
        let link_q4 = sidecar_of(&linked.link, PleFormat::Q4);
        let real_q4 = sidecar_of(&linked.real, PleFormat::Q4);
        let link_dir = dir_entries(&linked.link);
        let real_dir = dir_entries(&linked.real);

        // The loader's policy (`derive_missing_q4: false`): a q4 request with
        // no sidecar anywhere is an error, not a ~29 GB derivation.
        let (hfq, err) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", false);
        let err = err.unwrap_err();
        assert!(err.contains(link_q4.to_str().unwrap()), "{err}");
        assert!(err.contains(real_q4.to_str().unwrap()), "{err}");
        assert!(err.contains("hipfire pull"), "{err}");
        assert!(err.contains("hipfire ple derive --format q4"), "{err}");
        assert!(err.contains("derivation was not requested"), "{err}");
        assert!(hfq.ple_rows().is_none());

        // auto never derives either and says where it looked.
        let (_, choice) = resolve_at(&linked.link, PleFormatRequest::Auto, "gfx1151", true);
        let choice = choice.unwrap();
        assert_eq!(choice.origin, PleRowsOrigin::Embedded);
        let note = choice.note.unwrap();
        assert!(
            note.contains(link_q4.to_str().unwrap()) && note.contains(real_q4.to_str().unwrap()),
            "{note}"
        );
        assert!(note.contains("hipfire ple derive"), "{note}");

        assert_eq!(dir_entries(&linked.link), link_dir);
        assert_eq!(dir_entries(&linked.real), real_dir);
        assert!(!link_q4.exists() && !real_q4.exists());
    }

    #[cfg(unix)]
    #[test]
    fn unlinked_missing_q4_names_the_single_location_once() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m.hfqm");
        write_monolithic(&base, &all_shards(RowEncoding::Q8F16, 4));
        let q4 = sidecar_of(&base, PleFormat::Q4);
        let (_, err) = resolve_at(&base, PleFormatRequest::Q4, "gfx1100", false);
        let err = err.unwrap_err();
        assert!(err.contains("searched "), "{err}");
        // `searched` joins locations with " and ": a plain file has one.
        assert!(
            err.contains(&format!("searched {})", q4.display())),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn split_symlink_prefers_the_link_side_then_the_canonical_target() {
        let sha = q8_rows_sha(2);
        let linked = linked_base("flash-next-base.mq4-xts", |p| write_split(p, &sha));
        let link_q8 = sidecar_of(&linked.link, PleFormat::Q8);
        let real_q8 = sidecar_of(&linked.real, PleFormat::Q8);
        let link_q4 = sidecar_of(&linked.link, PleFormat::Q4);
        let real_q4 = sidecar_of(&linked.real, PleFormat::Q4);

        // Canonical only: found there.
        write_bound_sidecar(&real_q8, &linked.real, RowEncoding::Q8F16, &sha, 2);
        let (hfq, choice) = resolve_at(&linked.link, PleFormatRequest::Q8, "gfx1100", false);
        assert_eq!(
            choice.unwrap().origin,
            PleRowsOrigin::Sidecar(real_q8.clone())
        );
        assert_eq!(hfq.ple_rows().unwrap().path(), real_q8);

        // Both: the one beside the selected link wins.
        write_bound_sidecar(&link_q8, &linked.real, RowEncoding::Q8F16, &sha, 2);
        for request in [PleFormatRequest::Q8, PleFormatRequest::Auto] {
            let (hfq, choice) = resolve_at(&linked.link, request, "gfx1100", false);
            assert_eq!(
                choice.unwrap().origin,
                PleRowsOrigin::Sidecar(link_q8.clone())
            );
            assert_eq!(hfq.ple_rows().unwrap().path(), link_q8);
        }

        // q4 likewise: canonical only, then link-side preferred.
        write_bound_sidecar(&real_q4, &linked.real, RowEncoding::Q4Lm, &sha, 2);
        let (_, choice) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", false);
        assert_eq!(
            choice.unwrap().origin,
            PleRowsOrigin::Sidecar(real_q4.clone())
        );
        write_bound_sidecar(&link_q4, &linked.real, RowEncoding::Q4Lm, &sha, 2);
        let (_, choice) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", false);
        assert_eq!(choice.unwrap().origin, PleRowsOrigin::Sidecar(link_q4));
    }

    #[cfg(unix)]
    #[test]
    fn split_symlink_missing_sidecars_name_both_locations_and_derive_only_on_request() {
        let sha = q8_rows_sha(2);
        let linked = linked_base("flash-next-base.mq4-xts", |p| write_split(p, &sha));
        let q8 = |base: &Path| sidecar_of(base, PleFormat::Q8).to_str().unwrap().to_owned();
        let q4 = |base: &Path| sidecar_of(base, PleFormat::Q4).to_str().unwrap().to_owned();
        let link_dir = dir_entries(&linked.link);
        let real_dir = dir_entries(&linked.real);

        // Nothing anywhere: every request names both locations of the file it
        // needs plus the pull remedy, and `q4` also the derive command.
        for (request, wanted) in [
            (PleFormatRequest::Q8, q8 as fn(&Path) -> String),
            (PleFormatRequest::Q4, q4 as fn(&Path) -> String),
        ] {
            let (_, err) = resolve_at(&linked.link, request, "gfx1100", false);
            let err = err.unwrap_err();
            assert!(err.contains(&wanted(&linked.link)), "{err}");
            assert!(err.contains(&wanted(&linked.real)), "{err}");
            assert!(err.contains("hipfire pull"), "{err}");
            assert!(err.contains("hipfire ple derive --format q4"), "{err}");
        }
        let (_, err) = resolve_at(&linked.link, PleFormatRequest::Auto, "gfx1151", false);
        let err = err.unwrap_err();
        for base in [&linked.link, &linked.real] {
            assert!(err.contains(&q8(base)) && err.contains(&q4(base)), "{err}");
        }
        assert!(err.contains("hipfire pull"), "{err}");
        assert!(err.contains("hipfire ple derive --format q4"), "{err}");

        // q8 only beside the canonical target: a q4 request is refused (no
        // implicit derive), naming both q4 locations and the derive command.
        write_bound_sidecar(
            &sidecar_of(&linked.real, PleFormat::Q8),
            &linked.real,
            RowEncoding::Q8F16,
            &sha,
            2,
        );
        let real_entries = dir_entries(&linked.real);
        let (hfq, err) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", false);
        let err = err.unwrap_err();
        assert!(err.contains("derivation was not requested"), "{err}");
        assert!(err.contains(&q4(&linked.link)), "{err}");
        assert!(err.contains(&q4(&linked.real)), "{err}");
        assert!(err.contains("hipfire ple derive --format q4"), "{err}");
        assert!(err.contains("hipfire pull"), "{err}");
        assert!(hfq.ple_rows().is_none());
        assert_eq!(dir_entries(&linked.link), link_dir);
        assert_eq!(dir_entries(&linked.real), real_entries);
        assert_ne!(real_entries, real_dir);

        // auto with only q8 around (gfx1151 prefers q4) falls back silently
        // of derivation, noting both searched q4 locations.
        let (_, choice) = resolve_at(&linked.link, PleFormatRequest::Auto, "gfx1151", true);
        let choice = choice.unwrap();
        assert_eq!(choice.format, PleFormat::Q8);
        let note = choice.note.unwrap();
        assert!(
            note.contains(&q4(&linked.link)) && note.contains(&q4(&linked.real)),
            "{note}"
        );
        assert_eq!(dir_entries(&linked.real), real_entries);

        // An explicit-API caller that does opt in derives beside the base
        // whose q8 rows were found.
        let (hfq, choice) = resolve_at(&linked.link, PleFormatRequest::Q4, "gfx1100", true);
        let choice = choice.unwrap();
        assert!(choice.derived);
        assert_eq!(
            choice.origin,
            PleRowsOrigin::Sidecar(sidecar_of(&linked.real, PleFormat::Q4))
        );
        assert_eq!(hfq.ple_rows().unwrap().encoding(), RowEncoding::Q4Lm);
        assert_eq!(dir_entries(&linked.link), link_dir);
    }
}
