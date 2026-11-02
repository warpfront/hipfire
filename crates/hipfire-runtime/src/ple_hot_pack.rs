// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE hot-row pack (`<model>.plehotrows`): the hot rows of the `.plehot` rank
//! sidecar re-laid out into a separate file so the runtime can read them
//! through the kernel page cache (clean, file-backed, never swapped).
//!
//! Rows are copied *verbatim* from the model file's PLE shards, in the exact
//! order `ple_hot_store::HotRowStore` keeps them in RAM, so the bytes a request
//! returns never change.  This module is the file FORMAT: the header reader
//! and the writer (driven offline by `hipfire-quantize ple-hot-pack`).
//!
//! # Layout (all integers little-endian)
//!
//! ```text
//! 0..8     magic b"HFPLEHP1"
//! 8..16    u64 header_len
//! 16..     header: UTF-8 JSON, header_len bytes
//! pad      zero bytes up to data_offset (data_offset = round_up(16 + header_len, 4096))
//! data     tier 0 rows, tier 1 rows, ..., tier T-1 rows; each row `encoded_row_bytes` verbatim
//! EOF      file length == data_offset + data_bytes
//! ```
//!
//! * `kept_buckets` = K: buckets `0..K` of the sidecar are packed;
//!   `tier_group` = G (default `ple_hot_store::DEFAULT_TIER_BUCKETS`).
//! * Tier `t` holds every padded row `r` with `bucket[r] / G == t` and
//!   `bucket[r] < K`, in ascending global row order.  `T = ceil(K / G)`.
//!   Tier byte base = `sum(tier_rows[..t]) * encoded_row_bytes`.
//! * Global row `r = shard * rows_per_shard + local_row`; its model-file bytes
//!   are at `shard_offsets[shard] + local_row * encoded_row_bytes` (padding rows included).
//! * Pack offset of a kept row = `data_offset + tier_base[t] + slot(r) * encoded_row_bytes`,
//!   `slot(r)` = number of rows of tier `t` with a smaller global row index
//!   (exactly `HotRowStore::slot`).
//!
//! Header JSON fields (all required): `version` (2), `padded_rows`,
//! `rows_per_shard`, `row_encoding` (the artifact dtype name of the row encoding,
//! `RowEncoding::dtype_name()`, parsed back with `RowEncoding::from_dtype`: `"Q8_0"`
//! for 170-byte PLE rows, `"BF16"`, ...), `row_width` (values per row; PLE is 160),
//! `encoded_row_bytes` (must equal `row_encoding.encoded_row_bytes(row_width)`; the
//! pack layout never assumes a particular size), `ple_fingerprint` (64 lowercase
//! hex), `rank_payload_sha256` (the sidecar's `payload_sha256`), `tier_group`,
//! `kept_buckets`, `tier_rows` (u64 per tier), `data_offset`, `data_bytes`
//! (= sum(tier_rows) * `encoded_row_bytes`), `shard_offsets` (model-file offset of each PLE
//! shard), `model_file_bytes` (size of the model the pack was cut from),
//! `generator` (`"hipfire-quantize ple-hot-pack"`).
//!
//! # Public API (exact signatures)
//!
//! ```text
//! PACK_MAGIC: &[u8; 8]; PACK_VERSION: u64 (3); HOT_PACK_VERSION: u64 (2); TABLE_SEGMENT_BYTES: u64
//! PackKind { HotPack, PleTable } / PackOrder { Rank, Row }  // name() -> &str, parse(&str) -> Option<Self>
//! PleTableFields { codebook_f32_bits: Option<[u32;16]>, scale_grid_f32_bits: Option<[u32;9]>,
//!                  source_encoding, source_rows_sha256, segment_bytes, segment_sha256, data_sha256,
//!                  base_index_sha256, base_file_bytes }
//! PleHotPackHeader { version, kind, order, padded_rows, rows_per_shard, row_encoding, row_width,
//!                    encoded_row_bytes, ple_fingerprint, rank_payload_sha256, tier_group,
//!                    kept_buckets, tier_rows, rank_offset, rank_bytes, data_offset, data_bytes,
//!                    shard_offsets, model_file_bytes, table: Option<PleTableFields>, generator }
//! read_ple_hot_pack_header(path: &Path) -> Result<PleHotPackHeader, String>   // v2 and v3
//! encode_ple_table_header(header: &mut PleHotPackHeader) -> Result<Vec<u8>, String>
//! table_segment_count(data_bytes: u64, segment_bytes: u64) -> usize
//! checked_row_bytes(encoding: RowEncoding, width: usize) -> Result<usize, String>
//! write_ple_hot_pack(model_path: &Path, shard_offsets: &[u64], rows_per_shard: usize,
//!                    row_encoding: RowEncoding, row_width: usize,
//!                    fingerprint_hex: &str, rank: &PleHotRank, kept_buckets: usize,
//!                    tier_group: usize, out: &Path) -> Result<PleHotPackHeader, String>
//! write_ple_hot_pack_with_progress(.., out: &Path, progress: &mut dyn FnMut(u64, u64))
//!     -> Result<PleHotPackHeader, String>      // progress(rows_done, rows_total)
//! ```
//!
//! On non-Unix hosts the two writers return an unsupported error before touching
//! any file (they need O_DIRECT positional reads and positional writes); header
//! reading and encoding are platform-independent.
//!
//! # Version 3: hot packs and PLE tables in one container
//!
//! The PLE row sidecars (`<model>.ple-q8`, `<model>.ple-q4`) are `HFPLEHP1` files of
//! `kind = "ple-table"`: every row of the table, either Q8_0 or Q4LM.  The reader accepts
//! v2 (a hot pack: implied `kind hot-pack`, `order rank`, no embedded rank) and v3.  The
//! hot-pack writer keeps writing v2.
//!
//! ```text
//! 0..8     magic b"HFPLEHP1"
//! 8..16    u64 header_len
//! 16..     header: UTF-8 JSON, header_len bytes
//! pad      zeros up to rank_offset (order rank) or data_offset (order row)
//! rank     (order rank only) padded_rows bucket bytes at rank_offset, zeros up to data_offset
//! data     per `order`, data_bytes
//! EOF      file length == data_offset + data_bytes
//! ```
//!
//! * `rank_offset = round_up(16 + header_len, 4096)`, `data_offset =
//!   round_up(rank_offset + rank_bytes, 4096)`; row order has `rank_offset = rank_bytes = 0` and
//!   `data_offset = round_up(16 + header_len, 4096)`.  Both are a fixed point of the JSON length.
//! * `order rank`: tiers as in v2 (tier `t` = rows with `bucket / tier_group == t`, ascending
//!   global row order, hottest first), the bucket bytes embedded (`rank_bytes = padded_rows`,
//!   `rank_payload_sha256` = sha256 of them); `kept_buckets` is 256, so every row is in a tier.
//! * `order row`: global row order, shard `s` at `data_offset + s * rows_per_shard *
//!   encoded_row_bytes`; `rank_payload_sha256` is 64 zeros, `tier_group` 256, `tier_rows`
//!   `[padded_rows]`.
//! * A ple-table has `shard_offsets = []`, `model_file_bytes = 0`, `source_encoding "Q8_0"`,
//!   `codebook` / `scale` objects exactly for Q4LM rows, a base binding (`base_index_sha256`,
//!   `base_file_bytes`; checked against the base only for a monolithic base, ignored when
//!   attaching to a split base, which binds by row identity), `segment_bytes` > 0 (writers use
//!   [`TABLE_SEGMENT_BYTES`]) with one `segment_sha256` per segment of the data, and
//!   `data_sha256` = sha256 of the concatenated raw segment digests.  v3 JSON key order is the
//!   one `json_table_header` writes.
//!
//! `check_header_fields` lists every rule; [`read_ple_hot_pack_header`] checks all of them
//! plus the offsets and the exact file length (data and rank bytes are not read).

use sha2::{Digest, Sha256};
use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;

use crate::external_rows::RowEncoding;
#[cfg(unix)]
use crate::ple_hot_store::{open_direct, AlignedBuf, ShardRead, READ_ALIGN, READ_CHUNK};
use crate::ple_hot_store::{PleHotRank, BUCKETS};

pub const PACK_MAGIC: &[u8; 8] = b"HFPLEHP1";
/// The newest container version (hot pack or ple-table).
pub const PACK_VERSION: u64 = 3;
/// The version the hot-pack writer emits and the reader still accepts.
pub const HOT_PACK_VERSION: u64 = 2;
/// Bytes per data segment of a ple-table (its `segment_sha256` granularity).
pub const TABLE_SEGMENT_BYTES: u64 = 1 << 30;

const CODEBOOK_NAME: &str = "lloyd-max-normal-4bit";
const SCALE_DTYPE: &str = "f16";
const SCALE_GROUP: u64 = 32;
const SCALE_RULE: &str = "rms-mse-grid";
const ZERO_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Sanity bound on the JSON header.
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// The data section starts on this boundary.
const DATA_ALIGN: u64 = 4096;
/// A tier's output buffer is written out when it reaches this size.
#[cfg(unix)]
const FLUSH_BYTES: usize = 8 << 20;
/// `tier_of` value of a bucket that is not kept.
#[cfg(unix)]
const NO_TIER: u8 = u8::MAX;
#[cfg_attr(not(unix), allow(dead_code))]
const GENERATOR: &str = "hipfire-quantize ple-hot-pack";

/// What a `HFPLEHP1` file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackKind {
    /// The hottest rows of a model's PLE table, cut from the model file.
    HotPack,
    /// A whole PLE row table (every row), the `.ple-q8` / `.ple-q4` sidecars.
    PleTable,
}

impl PackKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::HotPack => "hot-pack",
            Self::PleTable => "ple-table",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "hot-pack" => Some(Self::HotPack),
            "ple-table" => Some(Self::PleTable),
            _ => None,
        }
    }
}

/// How the data section is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackOrder {
    /// Tier by tier, each tier's rows in ascending global row order (the
    /// layout `HotRowStore` indexes with a rank).
    Rank,
    /// Global row order: shard `s` at `data_offset + s * rows_per_shard *
    /// encoded_row_bytes`.
    Row,
}

impl PackOrder {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rank => "rank",
            Self::Row => "row",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "rank" => Some(Self::Rank),
            "row" => Some(Self::Row),
            _ => None,
        }
    }
}

/// The fields only a `ple-table` carries (all nullable / zero in a hot pack).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleTableFields {
    /// `LM4_CODEBOOK` bits; present exactly for Q4LM rows.
    pub codebook_f32_bits: Option<[u32; 16]>,
    /// `LM4_SCALE_GRID` bits; present exactly for Q4LM rows.
    pub scale_grid_f32_bits: Option<[u32; 9]>,
    /// Encoding of the rows the table was made from (always Q8_0).
    pub source_encoding: RowEncoding,
    /// Merkle digest of the natural-order Q8F16 shards the rows came from.
    pub source_rows_sha256: String,
    /// Bytes per data segment ([`TABLE_SEGMENT_BYTES`]).
    pub segment_bytes: u64,
    /// sha256 of each data segment.
    pub segment_sha256: Vec<String>,
    /// sha256 of the concatenated raw segment digests.
    pub data_sha256: String,
    /// sha256 of the base HFQM file's bytes `[0, data_offset)`. Checked on attach to a
    /// monolithic base only; a split base is bound by row identity instead.
    pub base_index_sha256: String,
    /// Length of that base file (same attach rule as `base_index_sha256`).
    pub base_file_bytes: u64,
}

/// A validated header (see the module docs for the field meanings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHotPackHeader {
    /// Container version: 2 (hot pack) or 3 (hot pack or ple-table).
    pub version: u64,
    pub kind: PackKind,
    pub order: PackOrder,
    pub padded_rows: u64,
    pub rows_per_shard: u64,
    /// Row encoding of every packed row (the model's PLE shard encoding).
    pub row_encoding: RowEncoding,
    /// Values per row.
    pub row_width: u64,
    /// Bytes per row; equals `row_encoding.encoded_row_bytes(row_width)`.
    pub encoded_row_bytes: u64,
    pub ple_fingerprint: String,
    pub rank_payload_sha256: String,
    pub tier_group: usize,
    pub kept_buckets: usize,
    pub tier_rows: Vec<u64>,
    /// Offset of the embedded rank (rank-ordered ple-table), else 0.
    pub rank_offset: u64,
    /// Bytes of the embedded rank (`padded_rows` for a rank-ordered ple-table), else 0.
    pub rank_bytes: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub shard_offsets: Vec<u64>,
    pub model_file_bytes: u64,
    /// `Some` exactly for [`PackKind::PleTable`].
    pub table: Option<PleTableFields>,
    pub generator: String,
}

fn is_hex64_lower(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Lowercase hex of `bytes`.
fn hex_lower(bytes: &[u8]) -> String {
    crate::ple_hot_store::hex(bytes)
}

/// The 32 bytes of a 64-lowercase-hex digest.
fn unhex32(text: &str) -> Option<[u8; 32]> {
    if !is_hex64_lower(text) {
        return None;
    }
    let mut out = [0u8; 32];
    for (byte, pair) in out.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        let digit = |c: u8| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
        *byte = digit(pair[0]) << 4 | digit(pair[1]);
    }
    Some(out)
}

/// Bytes of one `width`-value row of `encoding`, or an error for a width the
/// encoding cannot hold (zero, or a partial quantisation block).
pub fn checked_row_bytes(encoding: RowEncoding, width: usize) -> Result<usize, String> {
    encoding.checked_encoded_row_bytes(width).ok_or_else(|| {
        format!(
            "row width {width} is not a valid {} row width",
            encoding.dtype_name()
        )
    })
}

fn round_up_data(n: u64) -> u64 {
    n.next_multiple_of(DATA_ALIGN)
}

fn tier_count(kept_buckets: usize, tier_group: usize) -> usize {
    kept_buckets.div_ceil(tier_group)
}

#[cfg_attr(not(unix), allow(dead_code))]
fn json_header(h: &PleHotPackHeader) -> serde_json::Value {
    serde_json::json!({
        "version": HOT_PACK_VERSION,
        "padded_rows": h.padded_rows,
        "rows_per_shard": h.rows_per_shard,
        "row_encoding": h.row_encoding.dtype_name(),
        "row_width": h.row_width,
        "encoded_row_bytes": h.encoded_row_bytes,
        "ple_fingerprint": h.ple_fingerprint,
        "rank_payload_sha256": h.rank_payload_sha256,
        "tier_group": h.tier_group,
        "kept_buckets": h.kept_buckets,
        "tier_rows": h.tier_rows,
        "data_offset": h.data_offset,
        "data_bytes": h.data_bytes,
        "shard_offsets": h.shard_offsets,
        "model_file_bytes": h.model_file_bytes,
        "generator": h.generator,
    })
}

/// Serialise `header`, setting its `data_offset` to the fixed point of
/// `round_up(16 + len(json(data_offset)), 4096)`.
#[cfg_attr(not(unix), allow(dead_code))]
fn encode_header(header: &mut PleHotPackHeader) -> Result<Vec<u8>, String> {
    let mut candidate = DATA_ALIGN;
    loop {
        header.data_offset = candidate;
        let bytes = serde_json::to_vec(&json_header(header))
            .map_err(|e| format!("ple hot pack header: {e}"))?;
        if bytes.len() as u64 > MAX_HEADER_BYTES {
            return Err(format!("ple hot pack header is {} bytes", bytes.len()));
        }
        let needed = round_up_data(16 + bytes.len() as u64);
        if needed == candidate {
            return Ok(bytes);
        }
        candidate = needed;
    }
}

/// The v3 JSON of a `ple-table` header, in the design's field order.
fn json_table_header(h: &PleHotPackHeader, t: &PleTableFields) -> serde_json::Value {
    let codebook = t.codebook_f32_bits.map_or(
        serde_json::Value::Null,
        |bits| serde_json::json!({"name": CODEBOOK_NAME, "f32_bits": bits}),
    );
    let scale = t
        .scale_grid_f32_bits
        .map_or(serde_json::Value::Null, |bits| {
            serde_json::json!({
                "dtype": SCALE_DTYPE,
                "group": SCALE_GROUP,
                "rule": SCALE_RULE,
                "grid_f32_bits": bits,
            })
        });
    serde_json::json!({
        "version": PACK_VERSION,
        "kind": h.kind.name(),
        "order": h.order.name(),
        "padded_rows": h.padded_rows,
        "rows_per_shard": h.rows_per_shard,
        "row_encoding": h.row_encoding.dtype_name(),
        "row_width": h.row_width,
        "encoded_row_bytes": h.encoded_row_bytes,
        "ple_fingerprint": h.ple_fingerprint,
        "rank_payload_sha256": h.rank_payload_sha256,
        "tier_group": h.tier_group,
        "kept_buckets": h.kept_buckets,
        "tier_rows": h.tier_rows,
        "rank_offset": h.rank_offset,
        "rank_bytes": h.rank_bytes,
        "data_offset": h.data_offset,
        "data_bytes": h.data_bytes,
        "shard_offsets": h.shard_offsets,
        "model_file_bytes": h.model_file_bytes,
        "codebook": codebook,
        "scale": scale,
        "source_encoding": t.source_encoding.dtype_name(),
        "source_rows_sha256": t.source_rows_sha256,
        "segment_bytes": t.segment_bytes,
        "segment_sha256": t.segment_sha256,
        "data_sha256": t.data_sha256,
        "base_index_sha256": t.base_index_sha256,
        "base_file_bytes": t.base_file_bytes,
        "generator": h.generator,
    })
}

/// Serialise a `ple-table` header (design §8) and fix its layout: sets
/// `version`, `rank_bytes` (`padded_rows` for rank order, else 0), `rank_offset`
/// and `data_offset` to the fixed point of the 4096-alignment rules, after
/// checking every ple-table rule that does not depend on file bytes.  Returns
/// the file bytes `[0, pad_to)`: magic, header length, JSON and zero padding up
/// to `rank_offset` (rank order; the caller appends the rank bytes and pads to
/// `data_offset`) or `data_offset` (row order; the data follows).
pub fn encode_ple_table_header(header: &mut PleHotPackHeader) -> Result<Vec<u8>, String> {
    let fail = |what: String| format!("ple table header: {what}");
    if header.kind != PackKind::PleTable {
        return Err(fail("kind is not ple-table".to_string()));
    }
    header.version = PACK_VERSION;
    header.rank_bytes = match header.order {
        PackOrder::Rank => header.padded_rows,
        PackOrder::Row => {
            header.rank_offset = 0;
            0
        }
    };
    check_header_fields(header).map_err(&fail)?;
    let mut candidate = DATA_ALIGN;
    let json = loop {
        match header.order {
            PackOrder::Rank => {
                header.rank_offset = candidate;
                header.data_offset = round_up_data(candidate + header.rank_bytes);
            }
            PackOrder::Row => {
                header.rank_offset = 0;
                header.data_offset = candidate;
            }
        }
        let table = header
            .table
            .as_ref()
            .expect("checked by check_header_fields");
        let bytes = serde_json::to_vec(&json_table_header(header, table))
            .map_err(|e| fail(e.to_string()))?;
        if bytes.len() as u64 > MAX_HEADER_BYTES {
            return Err(fail(format!("header is {} bytes", bytes.len())));
        }
        let needed = round_up_data(16 + bytes.len() as u64);
        if needed == candidate {
            break bytes;
        }
        candidate = needed;
    };
    let pad_to = match header.order {
        PackOrder::Rank => header.rank_offset,
        PackOrder::Row => header.data_offset,
    };
    let mut out = Vec::with_capacity(pad_to as usize);
    out.extend_from_slice(PACK_MAGIC);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.resize(pad_to as usize, 0);
    Ok(out)
}

/// Segments of `data_bytes` at `segment_bytes` each (the last one partial); 0
/// when `segment_bytes` is 0.
pub fn table_segment_count(data_bytes: u64, segment_bytes: u64) -> usize {
    if segment_bytes == 0 {
        return 0;
    }
    data_bytes.div_ceil(segment_bytes) as usize
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read and validate a `HFPLEHP1` header (v2 hot pack or v3 hot pack /
/// ple-table): magic, version, header bound, field presence and types, row
/// size, tier count, `data_bytes`, the offset layout rules, every ple-table
/// rule (see [`check_header_fields`]) and the exact file length.  The data and
/// rank sections are not read.
pub fn read_ple_hot_pack_header(path: &Path) -> Result<PleHotPackHeader, String> {
    let bad = |what: String| format!("ple hot pack {}: {what}", path.display());
    let io_err = |what: &str, e: io::Error| format!("ple hot pack {}: {what}: {e}", path.display());
    let mut file = File::open(path).map_err(|e| io_err("open", e))?;
    let file_len = file.metadata().map_err(|e| io_err("stat", e))?.len();
    let mut prefix = [0u8; 16];
    file.read_exact(&mut prefix)
        .map_err(|e| io_err("read preamble", e))?;
    if &prefix[..8] != PACK_MAGIC {
        return Err(bad("bad magic".to_string()));
    }
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes"));
    if header_len == 0 || header_len > MAX_HEADER_BYTES || 16 + header_len > file_len {
        return Err(bad(format!("implausible header length {header_len}")));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|e| io_err("read header", e))?;
    let json: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|e| bad(format!("parse header: {e}")))?;
    let fields = HeaderJson { json: &json, path };

    let version = fields.u64("version")?;
    if version != HOT_PACK_VERSION && version != PACK_VERSION {
        return Err(bad(format!(
            "version {version}, expected {HOT_PACK_VERSION} or {PACK_VERSION}"
        )));
    }
    let generator = fields
        .field("generator")?
        .as_str()
        .ok_or_else(|| bad("header `generator` is not a string".to_string()))?
        .to_string();
    let (kind, order) = if version == PACK_VERSION {
        let kind = fields.str("kind")?;
        let order = fields.str("order")?;
        (
            PackKind::parse(kind).ok_or_else(|| {
                bad(format!(
                    "header `kind` `{kind}` is not hot-pack or ple-table"
                ))
            })?,
            PackOrder::parse(order)
                .ok_or_else(|| bad(format!("header `order` `{order}` is not rank or row")))?,
        )
    } else {
        (PackKind::HotPack, PackOrder::Rank)
    };
    let mut header = PleHotPackHeader {
        version,
        kind,
        order,
        padded_rows: fields.u64("padded_rows")?,
        rows_per_shard: fields.u64("rows_per_shard")?,
        row_encoding: {
            let text = fields.str("row_encoding")?;
            RowEncoding::from_dtype(text).ok_or_else(|| {
                bad(format!(
                    "header `row_encoding` `{text}` is not a known row encoding"
                ))
            })?
        },
        row_width: fields.u64("row_width")?,
        encoded_row_bytes: fields.u64("encoded_row_bytes")?,
        ple_fingerprint: fields.hex("ple_fingerprint")?,
        rank_payload_sha256: fields.hex("rank_payload_sha256")?,
        tier_group: fields.usize("tier_group")?,
        kept_buckets: fields.usize("kept_buckets")?,
        tier_rows: fields.u64_array("tier_rows")?,
        rank_offset: 0,
        rank_bytes: 0,
        data_offset: fields.u64("data_offset")?,
        data_bytes: fields.u64("data_bytes")?,
        shard_offsets: fields.u64_array("shard_offsets")?,
        model_file_bytes: fields.u64("model_file_bytes")?,
        table: None,
        generator,
    };
    if version == PACK_VERSION {
        header.rank_offset = fields.u64("rank_offset")?;
        header.rank_bytes = fields.u64("rank_bytes")?;
        header.table = match kind {
            PackKind::PleTable => Some(fields.table_fields()?),
            PackKind::HotPack => {
                fields.require_no_table_fields()?;
                None
            }
        };
    }
    check_header_fields(&header).map_err(&bad)?;

    let rank_ordered_table = header.kind == PackKind::PleTable && header.order == PackOrder::Rank;
    if rank_ordered_table {
        let expected_rank = round_up_data(16 + header_len);
        if header.rank_offset != expected_rank {
            return Err(bad(format!(
                "rank_offset {} != round_up(16 + header_len, {DATA_ALIGN}) = {expected_rank}",
                header.rank_offset
            )));
        }
        let expected_data = header
            .rank_offset
            .checked_add(header.rank_bytes)
            .map(round_up_data)
            .ok_or_else(|| bad("rank section overflows".to_string()))?;
        if header.data_offset != expected_data {
            return Err(bad(format!(
                "data_offset {} != round_up(rank_offset + rank_bytes, {DATA_ALIGN}) = \
                 {expected_data}",
                header.data_offset
            )));
        }
    } else {
        let expected_offset = round_up_data(16 + header_len);
        if header.data_offset != expected_offset {
            return Err(bad(format!(
                "data_offset {} != round_up(16 + header_len, {DATA_ALIGN}) = {expected_offset}",
                header.data_offset
            )));
        }
    }
    let expected_len = header
        .data_offset
        .checked_add(header.data_bytes)
        .ok_or_else(|| bad("file length overflow".to_string()))?;
    if file_len != expected_len {
        return Err(bad(format!(
            "file is {file_len} bytes, expected {expected_len}"
        )));
    }
    Ok(header)
}

/// Typed access to a parsed header object; every error names the field.
struct HeaderJson<'a> {
    json: &'a serde_json::Value,
    path: &'a Path,
}

impl HeaderJson<'_> {
    fn bad(&self, what: String) -> String {
        format!("ple hot pack {}: {what}", self.path.display())
    }

    fn field(&self, name: &str) -> Result<&serde_json::Value, String> {
        self.json
            .get(name)
            .ok_or_else(|| self.bad(format!("header lacks `{name}`")))
    }

    fn u64(&self, name: &str) -> Result<u64, String> {
        self.field(name)?
            .as_u64()
            .ok_or_else(|| self.bad(format!("header `{name}` is not an unsigned integer")))
    }

    fn usize(&self, name: &str) -> Result<usize, String> {
        usize::try_from(self.u64(name)?).map_err(|_| self.bad(format!("header `{name}` overflows")))
    }

    fn str(&self, name: &str) -> Result<&str, String> {
        self.field(name)?
            .as_str()
            .ok_or_else(|| self.bad(format!("header `{name}` is not a string")))
    }

    fn hex(&self, name: &str) -> Result<String, String> {
        let text = self.str(name)?;
        if is_hex64_lower(text) {
            Ok(text.to_string())
        } else {
            Err(self.bad(format!("header `{name}` is not 64 lowercase hex digits")))
        }
    }

    fn u64_array(&self, name: &str) -> Result<Vec<u64>, String> {
        self.field(name)?
            .as_array()
            .ok_or_else(|| self.bad(format!("header `{name}` is not an array")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| self.bad(format!("header `{name}` holds a non-integer")))
            })
            .collect()
    }

    /// `value[key]` as exactly `N` u32s.
    fn bits<const N: usize>(
        &self,
        owner: &str,
        value: &serde_json::Value,
        key: &str,
    ) -> Result<[u32; N], String> {
        let items = value
            .get(key)
            .and_then(serde_json::Value::as_array)
            .filter(|items| items.len() == N)
            .ok_or_else(|| self.bad(format!("header `{owner}.{key}` is not {N} integers")))?;
        let mut out = [0u32; N];
        for (slot, item) in out.iter_mut().zip(items) {
            *slot = item
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| self.bad(format!("header `{owner}.{key}` holds a non-u32")))?;
        }
        Ok(out)
    }

    fn codebook(&self) -> Result<Option<[u32; 16]>, String> {
        let value = self.field("codebook")?;
        if value.is_null() {
            return Ok(None);
        }
        if value.get("name").and_then(serde_json::Value::as_str) != Some(CODEBOOK_NAME) {
            return Err(self.bad(format!("header `codebook.name` is not `{CODEBOOK_NAME}`")));
        }
        self.bits::<16>("codebook", value, "f32_bits").map(Some)
    }

    fn scale(&self) -> Result<Option<[u32; 9]>, String> {
        let value = self.field("scale")?;
        if value.is_null() {
            return Ok(None);
        }
        let dtype = value.get("dtype").and_then(serde_json::Value::as_str);
        let rule = value.get("rule").and_then(serde_json::Value::as_str);
        let group = value.get("group").and_then(serde_json::Value::as_u64);
        if dtype != Some(SCALE_DTYPE) || rule != Some(SCALE_RULE) || group != Some(SCALE_GROUP) {
            return Err(self.bad(format!(
                "header `scale` is not {{dtype `{SCALE_DTYPE}`, group {SCALE_GROUP}, rule \
                 `{SCALE_RULE}`}}"
            )));
        }
        self.bits::<9>("scale", value, "grid_f32_bits").map(Some)
    }

    /// The nullable v3 fields of a `ple-table`.
    fn table_fields(&self) -> Result<PleTableFields, String> {
        let source = self.str("source_encoding")?;
        let source_encoding = RowEncoding::from_dtype(source).ok_or_else(|| {
            self.bad(format!(
                "header `source_encoding` `{source}` is not a known row encoding"
            ))
        })?;
        let segment_sha256 = self
            .field("segment_sha256")?
            .as_array()
            .ok_or_else(|| self.bad("header `segment_sha256` is not an array".to_string()))?
            .iter()
            .map(|item| match item.as_str() {
                Some(text) if is_hex64_lower(text) => Ok(text.to_string()),
                _ => Err(self.bad("header `segment_sha256` holds a non-64-hex entry".to_string())),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PleTableFields {
            codebook_f32_bits: self.codebook()?,
            scale_grid_f32_bits: self.scale()?,
            source_encoding,
            source_rows_sha256: self.hex("source_rows_sha256")?,
            segment_bytes: self.u64("segment_bytes")?,
            segment_sha256,
            data_sha256: self.hex("data_sha256")?,
            base_index_sha256: self.hex("base_index_sha256")?,
            base_file_bytes: self.u64("base_file_bytes")?,
        })
    }

    /// A v3 `hot-pack` carries the ple-table fields as nulls / zeros.
    fn require_no_table_fields(&self) -> Result<(), String> {
        for name in [
            "codebook",
            "scale",
            "source_encoding",
            "source_rows_sha256",
            "data_sha256",
            "base_index_sha256",
            "base_file_bytes",
        ] {
            if !self.field(name)?.is_null() {
                return Err(self.bad(format!("header `{name}` must be null in a hot pack")));
            }
        }
        if self.u64("segment_bytes")? != 0 {
            return Err(self.bad("header `segment_bytes` must be 0 in a hot pack".to_string()));
        }
        let segments = self.field("segment_sha256")?.as_array();
        if !segments.is_some_and(|segments| segments.is_empty()) {
            return Err(self.bad("header `segment_sha256` must be empty in a hot pack".to_string()));
        }
        Ok(())
    }
}

/// Every rule of a parsed header that does not depend on the file's bytes:
/// row size, tier counts, `data_bytes`, shard coverage, and for each kind its
/// own rules (see the module docs).  The reader runs it on every header and
/// [`encode_ple_table_header`] before it writes one.
fn check_header_fields(h: &PleHotPackHeader) -> Result<(), String> {
    let width =
        usize::try_from(h.row_width).map_err(|_| "header `row_width` overflows".to_string())?;
    let declared_bytes = checked_row_bytes(h.row_encoding, width)?;
    if h.encoded_row_bytes != declared_bytes as u64 {
        return Err(format!(
            "encoded_row_bytes {}, but {} rows of {} values are {declared_bytes} bytes",
            h.encoded_row_bytes,
            h.row_encoding.dtype_name(),
            h.row_width
        ));
    }
    if !(2..=BUCKETS).contains(&h.tier_group) {
        return Err(format!("tier_group {} outside 2..={BUCKETS}", h.tier_group));
    }
    if !(1..=BUCKETS).contains(&h.kept_buckets) {
        return Err(format!(
            "kept_buckets {} outside 1..={BUCKETS}",
            h.kept_buckets
        ));
    }
    let tiers = tier_count(h.kept_buckets, h.tier_group);
    if h.tier_rows.len() != tiers {
        return Err(format!(
            "tier_rows has {} entries, expected {tiers}",
            h.tier_rows.len()
        ));
    }
    let total_rows = h
        .tier_rows
        .iter()
        .try_fold(0u64, |sum, &n| sum.checked_add(n))
        .ok_or_else(|| "tier_rows overflow".to_string())?;
    let expected_bytes = total_rows
        .checked_mul(h.encoded_row_bytes)
        .ok_or_else(|| "data_bytes overflow".to_string())?;
    if h.data_bytes != expected_bytes {
        return Err(format!(
            "data_bytes {} != sum(tier_rows) * {} = {expected_bytes}",
            h.data_bytes, h.encoded_row_bytes
        ));
    }
    if total_rows > h.padded_rows {
        return Err(format!(
            "tier_rows sum {total_rows} exceeds padded_rows {}",
            h.padded_rows
        ));
    }
    match h.kind {
        PackKind::HotPack => {
            if h.rows_per_shard == 0
                || h.shard_offsets.len() as u64 * h.rows_per_shard != h.padded_rows
            {
                return Err(format!(
                    "{} shards of {} rows do not cover padded_rows {}",
                    h.shard_offsets.len(),
                    h.rows_per_shard,
                    h.padded_rows
                ));
            }
            if h.order != PackOrder::Rank || h.rank_offset != 0 || h.rank_bytes != 0 {
                return Err(
                    "a hot pack is rank-ordered (order rank, rank_offset 0, rank_bytes 0): its \
                     rank is the separate `.plehot` sidecar"
                        .to_string(),
                );
            }
            if h.table.is_some() {
                return Err("a hot pack carries no ple-table fields".to_string());
            }
            Ok(())
        }
        PackKind::PleTable => check_table_fields(h, total_rows),
    }
}

/// The `ple-table` rules of design §8.
fn check_table_fields(h: &PleHotPackHeader, total_rows: u64) -> Result<(), String> {
    if h.version != PACK_VERSION {
        return Err(format!(
            "a ple-table is a version {PACK_VERSION} file, header says {}",
            h.version
        ));
    }
    let table = h
        .table
        .as_ref()
        .ok_or_else(|| "a ple-table carries its table fields".to_string())?;
    if !matches!(h.row_encoding, RowEncoding::Q8F16 | RowEncoding::Q4Lm) {
        return Err(format!(
            "a ple-table holds Q8_0 or Q4LM rows, not {}",
            h.row_encoding.dtype_name()
        ));
    }
    let q4 = h.row_encoding == RowEncoding::Q4Lm;
    if table.codebook_f32_bits.is_some() != q4 {
        return Err(format!(
            "`codebook` must be {} for {} rows",
            if q4 { "present" } else { "null" },
            h.row_encoding.dtype_name()
        ));
    }
    if table.scale_grid_f32_bits.is_some() != q4 {
        return Err(format!(
            "`scale` must be {} for {} rows",
            if q4 { "present" } else { "null" },
            h.row_encoding.dtype_name()
        ));
    }
    if table.source_encoding != RowEncoding::Q8F16 {
        return Err(format!(
            "source_encoding must be Q8_0, not {}",
            table.source_encoding.dtype_name()
        ));
    }
    for (name, digest) in [
        ("source_rows_sha256", &table.source_rows_sha256),
        ("data_sha256", &table.data_sha256),
        ("base_index_sha256", &table.base_index_sha256),
    ] {
        if !is_hex64_lower(digest) {
            return Err(format!("`{name}` is not 64 lowercase hex digits"));
        }
    }
    if table.base_file_bytes == 0 {
        return Err("base_file_bytes is zero".to_string());
    }
    if table.segment_bytes == 0 {
        return Err("segment_bytes is zero".to_string());
    }
    let segments = table_segment_count(h.data_bytes, table.segment_bytes);
    if table.segment_sha256.len() != segments {
        return Err(format!(
            "segment_sha256 has {} entries, {} data bytes need {segments}",
            table.segment_sha256.len(),
            h.data_bytes
        ));
    }
    let mut raw = Vec::with_capacity(segments * 32);
    for digest in &table.segment_sha256 {
        raw.extend_from_slice(
            &unhex32(digest)
                .ok_or_else(|| "segment_sha256 holds a non-64-hex entry".to_string())?,
        );
    }
    let expected = hex_lower(&Sha256::digest(&raw));
    if table.data_sha256 != expected {
        return Err(format!(
            "data_sha256 {} != sha256 of the segment digests {expected}",
            table.data_sha256
        ));
    }
    if !is_hex64_lower(&h.ple_fingerprint) {
        return Err("ple_fingerprint is not 64 lowercase hex digits".to_string());
    }
    if h.kept_buckets != BUCKETS {
        return Err(format!(
            "kept_buckets {} != {BUCKETS}: a ple-table keeps every row",
            h.kept_buckets
        ));
    }
    if !h.shard_offsets.is_empty() {
        return Err("shard_offsets must be empty in a ple-table".to_string());
    }
    if h.model_file_bytes != 0 {
        return Err("model_file_bytes must be 0 in a ple-table".to_string());
    }
    if h.rows_per_shard == 0 || h.padded_rows == 0 || h.padded_rows % h.rows_per_shard != 0 {
        return Err(format!(
            "padded_rows {} is not a whole number of shards of {} rows",
            h.padded_rows, h.rows_per_shard
        ));
    }
    if total_rows != h.padded_rows {
        return Err(format!(
            "tier_rows sum {total_rows} != padded_rows {}: a ple-table keeps every row",
            h.padded_rows
        ));
    }
    match h.order {
        PackOrder::Rank => {
            if h.rank_bytes != h.padded_rows {
                return Err(format!(
                    "rank_bytes {} != padded_rows {} (a rank-ordered table embeds its rank)",
                    h.rank_bytes, h.padded_rows
                ));
            }
        }
        PackOrder::Row => {
            if h.rank_offset != 0 || h.rank_bytes != 0 {
                return Err(format!(
                    "rank_offset {} / rank_bytes {} must be 0 in a row-ordered table",
                    h.rank_offset, h.rank_bytes
                ));
            }
            if h.rank_payload_sha256 != ZERO_SHA {
                return Err(
                    "rank_payload_sha256 must be 64 zeros in a row-ordered table".to_string(),
                );
            }
            if h.tier_group != BUCKETS || h.tier_rows != [h.padded_rows] {
                return Err(format!(
                    "a row-ordered table has tier_group {BUCKETS} and tier_rows [padded_rows], \
                     not tier_group {} and tier_rows {:?}",
                    h.tier_group, h.tier_rows
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

#[cfg(unix)]
/// Model file read through O_DIRECT (or buffered where the filesystem refuses
/// the flag).
struct DirectRead(File);

#[cfg(unix)]
impl ShardRead for DirectRead {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.0.read_at(dst, offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                other => return other,
            }
        }
    }
}

/// Cut the pack `out` from the model file `model_path` (O_DIRECT reads).
/// See [`write_ple_hot_pack_with_progress`].
#[allow(clippy::too_many_arguments)]
pub fn write_ple_hot_pack(
    model_path: &Path,
    shard_offsets: &[u64],
    rows_per_shard: usize,
    row_encoding: RowEncoding,
    row_width: usize,
    fingerprint_hex: &str,
    rank: &PleHotRank,
    kept_buckets: usize,
    tier_group: usize,
    out: &Path,
) -> Result<PleHotPackHeader, String> {
    write_ple_hot_pack_with_progress(
        model_path,
        shard_offsets,
        rows_per_shard,
        row_encoding,
        row_width,
        fingerprint_hex,
        rank,
        kept_buckets,
        tier_group,
        out,
        &mut |_, _| {},
    )
}

/// [`write_ple_hot_pack`] reporting `progress(rows_done, rows_total)` after
/// every model window read.  The pack is written to `<out>.tmp.<pid>` and
/// renamed into place after `sync_all`; the tmp file is removed on error.
#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
pub fn write_ple_hot_pack_with_progress(
    model_path: &Path,
    shard_offsets: &[u64],
    rows_per_shard: usize,
    row_encoding: RowEncoding,
    row_width: usize,
    fingerprint_hex: &str,
    rank: &PleHotRank,
    kept_buckets: usize,
    tier_group: usize,
    out: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PleHotPackHeader, String> {
    let file = open_direct(model_path)
        .map_err(|e| format!("ple hot pack: open {}: {e}", model_path.display()))?;
    let model_file_bytes = file
        .metadata()
        .map_err(|e| format!("ple hot pack: stat {}: {e}", model_path.display()))?
        .len();
    write_pack_with(
        &DirectRead(file),
        model_file_bytes,
        shard_offsets,
        rows_per_shard,
        row_encoding,
        row_width,
        fingerprint_hex,
        rank,
        kept_buckets,
        tier_group,
        out,
        progress,
    )
}

/// Unsupported off Unix: the writer reads the model with O_DIRECT positional
/// reads and writes the pack with positional writes.  Fails before any file
/// is opened, created or renamed.
#[cfg(not(unix))]
#[allow(clippy::too_many_arguments)]
pub fn write_ple_hot_pack_with_progress(
    _model_path: &Path,
    _shard_offsets: &[u64],
    _rows_per_shard: usize,
    _row_encoding: RowEncoding,
    _row_width: usize,
    _fingerprint_hex: &str,
    _rank: &PleHotRank,
    _kept_buckets: usize,
    _tier_group: usize,
    _out: &Path,
    _progress: &mut dyn FnMut(u64, u64),
) -> Result<PleHotPackHeader, String> {
    Err(
        "ple hot pack: writing a PLE hot pack is unsupported on Windows (it needs Unix \
         O_DIRECT positional reads and positional writes); use a Unix host"
            .to_string(),
    )
}

/// Writer core over any positional model reader (tests drive it with an
/// in-memory model).
#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_pack_with(
    reader: &dyn ShardRead,
    model_file_bytes: u64,
    shard_offsets: &[u64],
    rows_per_shard: usize,
    row_encoding: RowEncoding,
    row_width: usize,
    fingerprint_hex: &str,
    rank: &PleHotRank,
    kept_buckets: usize,
    tier_group: usize,
    out: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PleHotPackHeader, String> {
    let fail = |what: String| format!("ple hot pack: {what}");
    let row_bytes = checked_row_bytes(row_encoding, row_width).map_err(&fail)? as u64;
    if !is_hex64_lower(fingerprint_hex) {
        return Err(fail(format!(
            "fingerprint `{fingerprint_hex}` is not 64 lowercase hex digits"
        )));
    }
    if !(2..=BUCKETS).contains(&tier_group) {
        return Err(fail(format!(
            "tier group {tier_group} outside 2..={BUCKETS}"
        )));
    }
    if !(1..=BUCKETS).contains(&kept_buckets) {
        return Err(fail(format!(
            "kept buckets {kept_buckets} outside 1..={BUCKETS}"
        )));
    }
    let rows_per_shard_u64 = rows_per_shard as u64;
    let padded_rows = shard_offsets.len() as u64 * rows_per_shard_u64;
    if rows_per_shard == 0 || shard_offsets.is_empty() || rank.padded_rows() != padded_rows {
        return Err(fail(format!(
            "rank covers {} rows but {} shards of {rows_per_shard} rows cover {padded_rows}",
            rank.padded_rows(),
            shard_offsets.len()
        )));
    }
    let buckets = rank.buckets();
    if buckets.len() as u64 != padded_rows {
        return Err(fail(format!(
            "rank holds {} buckets for {padded_rows} rows",
            buckets.len()
        )));
    }
    let rank_sha = rank
        .header()
        .get("payload_sha256")
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase)
        .filter(|sha| is_hex64_lower(sha))
        .ok_or_else(|| fail("rank header lacks a 64-hex payload_sha256".to_string()))?;
    for (shard, &offset) in shard_offsets.iter().enumerate() {
        let end = rows_per_shard_u64
            .checked_mul(row_bytes)
            .and_then(|len| offset.checked_add(len));
        if end.is_none_or(|end| end > model_file_bytes) {
            return Err(fail(format!(
                "shard {shard} at {offset} is not inside the {model_file_bytes}-byte model file"
            )));
        }
    }

    // Tier of every bucket, rows per tier, tier byte bases.
    let tiers = tier_count(kept_buckets, tier_group);
    let mut tier_of = [NO_TIER; BUCKETS];
    for (bucket, slot) in tier_of.iter_mut().enumerate().take(kept_buckets) {
        *slot = (bucket / tier_group) as u8;
    }
    let mut tier_rows = vec![0u64; tiers];
    for &bucket in buckets {
        let tier = tier_of[usize::from(bucket)];
        if tier != NO_TIER {
            tier_rows[usize::from(tier)] += 1;
        }
    }
    let mut tier_base = Vec::with_capacity(tiers);
    let mut total_rows = 0u64;
    for &rows in &tier_rows {
        tier_base.push(total_rows * row_bytes);
        total_rows += rows;
    }
    let data_bytes = total_rows * row_bytes;

    let mut header = PleHotPackHeader {
        version: HOT_PACK_VERSION,
        kind: PackKind::HotPack,
        order: PackOrder::Rank,
        padded_rows,
        rows_per_shard: rows_per_shard_u64,
        row_encoding,
        row_width: row_width as u64,
        encoded_row_bytes: row_bytes,
        ple_fingerprint: fingerprint_hex.to_string(),
        rank_payload_sha256: rank_sha,
        tier_group,
        kept_buckets,
        tier_rows,
        rank_offset: 0,
        rank_bytes: 0,
        data_offset: 0,
        data_bytes,
        shard_offsets: shard_offsets.to_vec(),
        model_file_bytes,
        table: None,
        generator: GENERATOR.to_string(),
    };
    let header_bytes = encode_header(&mut header)?;

    let mut tmp_name = out.as_os_str().to_owned();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp_name);
    let result = write_tmp(
        reader,
        &tmp,
        &header,
        &header_bytes,
        &tier_of,
        &tier_base,
        buckets,
        progress,
    )
    .and_then(|()| {
        std::fs::rename(&tmp, out).map_err(|e| {
            fail(format!(
                "rename {} -> {}: {e}",
                tmp.display(),
                out.display()
            ))
        })
    });
    match result {
        Ok(()) => Ok(header),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(error)
        }
    }
}

#[cfg(unix)]
/// One tier's pending output.
struct TierOut {
    buf: Vec<u8>,
    /// Absolute file offset the next flush writes at.
    cursor: u64,
    /// Rows appended so far (flushed or buffered).
    rows: u64,
}

#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
fn write_tmp(
    reader: &dyn ShardRead,
    tmp: &Path,
    header: &PleHotPackHeader,
    header_bytes: &[u8],
    tier_of: &[u8; BUCKETS],
    tier_base: &[u64],
    buckets: &[u8],
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    let tmp_err = |what: &str, e: io::Error| format!("ple hot pack: {what} {}: {e}", tmp.display());
    let output = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp)
        .map_err(|e| tmp_err("create", e))?;
    output
        .set_len(header.data_offset + header.data_bytes)
        .map_err(|e| tmp_err("set_len", e))?;
    let mut preamble = Vec::with_capacity(16 + header_bytes.len());
    preamble.extend_from_slice(PACK_MAGIC);
    preamble.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    preamble.extend_from_slice(header_bytes);
    output
        .write_all_at(&preamble, 0)
        .map_err(|e| tmp_err("write header", e))?;

    let row_bytes = header.encoded_row_bytes as usize;
    let rows_per_shard = header.rows_per_shard;
    let mut tiers: Vec<TierOut> = header
        .tier_rows
        .iter()
        .zip(tier_base)
        .map(|(&rows, &base)| TierOut {
            buf: Vec::with_capacity(((rows * row_bytes as u64).min(FLUSH_BYTES as u64)) as usize),
            cursor: header.data_offset + base,
            rows: 0,
        })
        .collect();
    let flush = |tier: &mut TierOut| -> Result<(), String> {
        if !tier.buf.is_empty() {
            output
                .write_all_at(&tier.buf, tier.cursor)
                .map_err(|e| tmp_err("write rows to", e))?;
            tier.cursor += tier.buf.len() as u64;
            tier.buf.clear();
        }
        Ok(())
    };

    let mut window = AlignedBuf::new(READ_CHUNK);
    for (shard, &file_offset) in header.shard_offsets.iter().enumerate() {
        let first_global = shard as u64 * rows_per_shard;
        let shard_end = file_offset + rows_per_shard * row_bytes as u64;
        let mut next = 0u64;
        while next < rows_per_shard {
            // Aligned window starting at or before the first unread row.
            let start = (file_offset + next * row_bytes as u64) & !(READ_ALIGN - 1);
            let end = (start + READ_CHUNK as u64).min(shard_end.next_multiple_of(READ_ALIGN));
            let want = (end - start) as usize;
            let dst = &mut window.as_mut_slice()[..want];
            let mut filled = 0usize;
            while filled < want {
                let n = reader
                    .read_at(start + filled as u64, &mut dst[filled..])
                    .map_err(|e| {
                        format!("ple hot pack: read model at {}: {e}", start + filled as u64)
                    })?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            let available_end = start + filled as u64;
            let rows_end =
                (available_end.saturating_sub(file_offset) / row_bytes as u64).min(rows_per_shard);
            if rows_end <= next {
                return Err(format!(
                    "ple hot pack: unexpected end of model file at byte {available_end} \
                     (shard {shard} rows {next}..{rows_per_shard})"
                ));
            }
            for row in next..rows_end {
                let tier = tier_of[usize::from(buckets[(first_global + row) as usize])];
                if tier == NO_TIER {
                    continue;
                }
                let tier = &mut tiers[usize::from(tier)];
                if tier.buf.len() + row_bytes > FLUSH_BYTES {
                    flush(tier)?;
                }
                let at = (file_offset + row * row_bytes as u64 - start) as usize;
                tier.buf.extend_from_slice(&dst[at..at + row_bytes]);
                tier.rows += 1;
            }
            next = rows_end;
            progress(first_global + next, header.padded_rows);
        }
    }
    for (index, tier) in tiers.iter_mut().enumerate() {
        flush(tier)?;
        if tier.rows != header.tier_rows[index] {
            return Err(format!(
                "ple hot pack: tier {index} got {} rows, expected {}",
                tier.rows, header.tier_rows[index]
            ));
        }
        let expected_end = header.data_offset + tier_base[index] + tier.rows * row_bytes as u64;
        if tier.cursor != expected_end {
            return Err(format!(
                "ple hot pack: tier {index} cursor {} != {expected_end}",
                tier.cursor
            ));
        }
    }
    output.sync_all().map_err(|e| tmp_err("sync", e))
}

/// Table-file fixtures for the unit tests of this module and its consumers.
#[cfg(test)]
pub(crate) mod table_support {
    use super::*;
    use crate::ple_hot_store::test_support::FINGERPRINT;
    use crate::ple_q4::{lm4_codebook_bits, lm4_scale_grid_bits};
    use sha2::{Digest, Sha256};

    /// A valid header for a table of `shard_count` shards (offsets / digests of
    /// the data are filled by [`write_test_table`]).
    pub(crate) fn table_header(
        order: PackOrder,
        encoding: RowEncoding,
        rows_per_shard: usize,
        row_width: usize,
        shard_count: usize,
        buckets: Option<&[u8]>,
        tier_group: usize,
    ) -> PleHotPackHeader {
        let row_bytes = checked_row_bytes(encoding, row_width).unwrap() as u64;
        let padded_rows = (rows_per_shard * shard_count) as u64;
        let (tier_group, tier_rows, rank_payload_sha256) = match order {
            PackOrder::Rank => {
                let buckets = buckets.expect("rank order needs the buckets");
                assert_eq!(buckets.len() as u64, padded_rows);
                let mut tier_rows = vec![0u64; tier_count(BUCKETS, tier_group)];
                for &b in buckets {
                    tier_rows[usize::from(b) / tier_group] += 1;
                }
                (tier_group, tier_rows, hex_lower(&Sha256::digest(buckets)))
            }
            PackOrder::Row => (BUCKETS, vec![padded_rows], "0".repeat(64)),
        };
        let q4 = encoding == RowEncoding::Q4Lm;
        let data_bytes = padded_rows * row_bytes;
        let segments = vec![[0x11u8; 32]; table_segment_count(data_bytes, TABLE_SEGMENT_BYTES)];
        PleHotPackHeader {
            version: PACK_VERSION,
            kind: PackKind::PleTable,
            order,
            padded_rows,
            rows_per_shard: rows_per_shard as u64,
            row_encoding: encoding,
            row_width: row_width as u64,
            encoded_row_bytes: row_bytes,
            ple_fingerprint: FINGERPRINT.to_string(),
            rank_payload_sha256,
            tier_group,
            kept_buckets: BUCKETS,
            tier_rows,
            rank_offset: 0,
            rank_bytes: 0,
            data_offset: 0,
            data_bytes: padded_rows * row_bytes,
            shard_offsets: Vec::new(),
            model_file_bytes: 0,
            table: Some(PleTableFields {
                codebook_f32_bits: q4.then(lm4_codebook_bits),
                scale_grid_f32_bits: q4.then(lm4_scale_grid_bits),
                source_encoding: RowEncoding::Q8F16,
                source_rows_sha256: "5a".repeat(32),
                segment_bytes: TABLE_SEGMENT_BYTES,
                segment_sha256: segments.iter().map(|d| hex_lower(d)).collect(),
                data_sha256: hex_lower(&Sha256::digest(segments.concat())),
                base_index_sha256: "7e".repeat(32),
                base_file_bytes: 123_456_789,
            }),
            generator: "hipfire ple test".to_string(),
        }
    }

    /// Write a v3 ple-table file holding `shards` (natural-order encoded rows
    /// of `encoding`): row order verbatim, rank order re-laid out by tier with
    /// the rank embedded.  Returns the header as written.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_test_table(
        path: &Path,
        order: PackOrder,
        encoding: RowEncoding,
        row_width: usize,
        rows_per_shard: usize,
        shards: &[Vec<u8>],
        buckets: Option<&[u8]>,
        tier_group: usize,
    ) -> PleHotPackHeader {
        let row_bytes = checked_row_bytes(encoding, row_width).unwrap();
        let mut header = table_header(
            order,
            encoding,
            rows_per_shard,
            row_width,
            shards.len(),
            buckets,
            tier_group,
        );
        let data: Vec<u8> = match order {
            PackOrder::Row => shards.concat(),
            PackOrder::Rank => {
                let buckets = buckets.unwrap();
                let mut tiers = vec![Vec::new(); header.tier_rows.len()];
                for (row, &bucket) in buckets.iter().enumerate() {
                    let shard = &shards[row / rows_per_shard];
                    let local = row % rows_per_shard;
                    tiers[usize::from(bucket) / tier_group]
                        .extend_from_slice(&shard[local * row_bytes..(local + 1) * row_bytes]);
                }
                tiers.concat()
            }
        };
        assert_eq!(data.len() as u64, header.data_bytes);
        let digests: Vec<[u8; 32]> = data
            .chunks(TABLE_SEGMENT_BYTES as usize)
            .map(|segment| Sha256::digest(segment).into())
            .collect();
        let shard_digests: Vec<[u8; 32]> =
            shards.iter().map(|s| Sha256::digest(s).into()).collect();
        let table = header.table.as_mut().unwrap();
        table.segment_sha256 = digests.iter().map(|d| hex_lower(d)).collect();
        table.data_sha256 = hex_lower(&Sha256::digest(digests.concat()));
        table.source_rows_sha256 = hex_lower(&Sha256::digest(shard_digests.concat()));
        let mut file = encode_ple_table_header(&mut header).unwrap();
        if order == PackOrder::Rank {
            file.extend_from_slice(buckets.unwrap());
            file.resize(header.data_offset as usize, 0);
        }
        file.extend_from_slice(&data);
        std::fs::write(path, file).unwrap();
        header
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::ple_hot_store::test_support::*;

    struct Fixture {
        model: Vec<u8>,
        offsets: Vec<u64>,
        rows_per_shard: usize,
        encoding: RowEncoding,
        row_width: usize,
        row_bytes: usize,
        buckets: Vec<u8>,
        rank: PleHotRank,
    }

    /// PLE-shaped fixture: Q8F16 rows of 160 values (170 B).
    fn fixture(shards: usize, rows_per_shard: usize, seed: u64) -> Fixture {
        fixture_with(
            synthetic_shards(shards, rows_per_shard),
            rows_per_shard,
            seed,
            RowEncoding::Q8F16,
            TEST_ROW_WIDTH,
        )
    }

    /// Fixture whose shards hold deterministic rows of any encoding/width.
    fn encoded_fixture(
        shards: usize,
        rows_per_shard: usize,
        seed: u64,
        encoding: RowEncoding,
        row_width: usize,
    ) -> Fixture {
        let row_bytes = checked_row_bytes(encoding, row_width).unwrap();
        let data = (0..shards)
            .map(|s| {
                (0..rows_per_shard * row_bytes)
                    .map(|i| {
                        let global = (s * rows_per_shard + i / row_bytes) as u64;
                        (global
                            .wrapping_mul(0x9E37_79B9)
                            .wrapping_add((i % row_bytes) as u64 * 31)
                            >> 3) as u8
                    })
                    .collect()
            })
            .collect();
        fixture_with(data, rows_per_shard, seed, encoding, row_width)
    }

    fn fixture_with(
        shard_data: Vec<Vec<u8>>,
        rows_per_shard: usize,
        seed: u64,
        encoding: RowEncoding,
        row_width: usize,
    ) -> Fixture {
        let shards = shard_data.len();
        let (model, offsets) = layout_shards(&shard_data);
        let rows = shards * rows_per_shard;
        let buckets = synthetic_buckets(rows, seed);
        let rank = PleHotRank::from_buckets(rows as u64, FINGERPRINT, buckets.clone());
        Fixture {
            model,
            offsets,
            rows_per_shard,
            encoding,
            row_width,
            row_bytes: checked_row_bytes(encoding, row_width).unwrap(),
            buckets,
            rank,
        }
    }

    fn pack(
        fx: &Fixture,
        kept: usize,
        group: usize,
        out: &Path,
    ) -> Result<PleHotPackHeader, String> {
        let mut last = (0u64, 0u64);
        let header = write_pack_with(
            &SliceReader(fx.model.clone()),
            fx.model.len() as u64,
            &fx.offsets,
            fx.rows_per_shard,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            kept,
            group,
            out,
            &mut |done, total| {
                assert!(done >= last.0 && done <= total);
                last = (done, total);
            },
        )?;
        assert_eq!(
            last,
            (header.padded_rows, header.padded_rows),
            "progress reaches 100%"
        );
        Ok(header)
    }

    /// Every kept row lands at its slot, byte for byte; the file length is exact.
    fn verify(fx: &Fixture, header: &PleHotPackHeader, out: &Path) {
        let bytes = std::fs::read(out).unwrap();
        assert_eq!(bytes.len() as u64, header.data_offset + header.data_bytes);
        assert_eq!(&bytes[..8], PACK_MAGIC);
        let kept = header.kept_buckets;
        let group = header.tier_group;
        let tiers = tier_count(kept, group);
        let mut base = vec![0u64; tiers];
        for t in 1..tiers {
            base[t] = base[t - 1] + header.tier_rows[t - 1] * fx.row_bytes as u64;
        }
        let mut seen = vec![0u64; tiers];
        let mut padding_zero = true;
        for &b in &bytes[16 + u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize
            ..header.data_offset as usize]
        {
            padding_zero &= b == 0;
        }
        assert!(padding_zero, "padding between header and data is zero");
        for (row, &bucket) in fx.buckets.iter().enumerate() {
            if usize::from(bucket) >= kept {
                continue;
            }
            let tier = usize::from(bucket) / group;
            let slot = seen[tier];
            seen[tier] += 1;
            let shard = row / fx.rows_per_shard;
            let local = row % fx.rows_per_shard;
            let model_at = fx.offsets[shard] as usize + local * fx.row_bytes;
            let pack_at =
                header.data_offset as usize + base[tier] as usize + slot as usize * fx.row_bytes;
            assert_eq!(
                &bytes[pack_at..pack_at + fx.row_bytes],
                &fx.model[model_at..model_at + fx.row_bytes],
                "row {row} (tier {tier}, slot {slot})"
            );
        }
        assert_eq!(seen, header.tier_rows, "tier row counts");
    }

    #[test]
    fn round_trip_group_5_and_2() {
        let fx = fixture(3, 4096, 7);
        assert_ne!(fx.offsets[0] % 4096, 0, "shard start is not sector aligned");
        let dir = tempfile::tempdir().unwrap();
        for (kept, group) in [(100usize, 5usize), (100, 2), (7, 5), (1, 2)] {
            let out = dir.path().join(format!("pack-{kept}-{group}.plehotrows"));
            let written = pack(&fx, kept, group, &out).unwrap();
            assert_eq!(written.tier_rows.len(), tier_count(kept, group));
            assert_eq!(written.data_offset % 4096, 0);
            assert_eq!(written.model_file_bytes, fx.model.len() as u64);
            assert_eq!(written.shard_offsets, fx.offsets);
            assert_eq!(written.rank_payload_sha256.len(), 64);
            let read = read_ple_hot_pack_header(&out).unwrap();
            assert_eq!(read, written, "header reads back equal");
            verify(&fx, &read, &out);
        }
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "tmp files left: {leftovers:?}");
    }

    #[test]
    fn keep_all_buckets_packs_every_row() {
        let fx = fixture(3, 4096, 11);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("all.plehotrows");
        let written = pack(&fx, 256, 5, &out).unwrap();
        assert_eq!(
            written.data_bytes,
            written.padded_rows * fx.row_bytes as u64
        );
        assert_eq!(written.tier_rows.iter().sum::<u64>(), written.padded_rows);
        assert_eq!(written.tier_rows.len(), 52);
        verify(&fx, &read_ple_hot_pack_header(&out).unwrap(), &out);
    }

    /// Shards larger than one 16 MiB read window, at unaligned file offsets.
    #[test]
    fn multi_window_unaligned_shards() {
        let fx = fixture(2, 100_000, 3);
        const { assert!(100_000 * 170 > READ_CHUNK) };
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("big.plehotrows");
        let written = pack(&fx, 64, 4, &out).unwrap();
        verify(&fx, &read_ple_hot_pack_header(&out).unwrap(), &out);
        assert!(written.data_bytes > 0);
    }

    #[test]
    fn writer_rejects_bad_inputs() {
        let fx = fixture(2, 1024, 5);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("x.plehotrows");
        assert!(pack(&fx, 0, 5, &out).is_err());
        assert!(pack(&fx, 257, 5, &out).is_err());
        assert!(pack(&fx, 10, 1, &out).is_err());
        assert!(pack(&fx, 10, 257, &out).is_err());
        let err = write_pack_with(
            &SliceReader(fx.model.clone()),
            fx.model.len() as u64,
            &fx.offsets,
            1000,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            10,
            5,
            &out,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("cover"), "{err}");
        let err = write_pack_with(
            &SliceReader(fx.model.clone()),
            fx.offsets[1] + (1024 * fx.row_bytes) as u64 - 1,
            &fx.offsets,
            1024,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            10,
            5,
            &out,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("not inside"), "{err}");
        assert!(!out.exists());
    }

    #[test]
    fn writer_cleans_tmp_on_read_failure() {
        let fx = fixture(2, 1024, 5);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("short.plehotrows");
        // The reader's file is shorter than the model_file_bytes claims.
        let mut truncated = fx.model.clone();
        truncated.truncate(fx.offsets[1] as usize + 1000);
        let err = write_pack_with(
            &SliceReader(truncated),
            fx.model.len() as u64,
            &fx.offsets,
            fx.rows_per_shard,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            100,
            5,
            &out,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("unexpected end"), "{err}");
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "tmp removed"
        );
    }

    /// The pack layout is row-size agnostic: BF16 rows of small widths (16 B,
    /// 48 B, an odd 34 B) round-trip through the writer and the header reader,
    /// and the writer refuses a width its encoding cannot hold.
    #[test]
    fn non_q8_row_encodings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        for (encoding, width, row_bytes) in [
            (RowEncoding::Bf16, 8usize, 16usize),
            (RowEncoding::Bf16, 24, 48),
            (RowEncoding::Bf16, 17, 34),
            (RowEncoding::Q8F16, 32, 34),
        ] {
            let fx = encoded_fixture(3, 2048, 13, encoding, width);
            assert_eq!(fx.row_bytes, row_bytes);
            assert_ne!(row_bytes, 170);
            let out = dir
                .path()
                .join(format!("{}-{width}.plehotrows", encoding.dtype_name()));
            let written = pack(&fx, 100, 5, &out).unwrap();
            assert_eq!(written.row_encoding, encoding);
            assert_eq!(written.row_width, width as u64);
            assert_eq!(written.encoded_row_bytes, row_bytes as u64);
            assert_eq!(
                written.data_bytes,
                written.tier_rows.iter().sum::<u64>() * row_bytes as u64
            );
            let read = read_ple_hot_pack_header(&out).unwrap();
            assert_eq!(read, written, "header reads back equal");
            verify(&fx, &read, &out);
        }

        let fx = encoded_fixture(2, 1024, 5, RowEncoding::Bf16, 8);
        let out = dir.path().join("bad-width.plehotrows");
        for (encoding, width) in [
            (RowEncoding::Q8F16, 33usize),
            (RowEncoding::Q8F16, 0),
            (RowEncoding::Bf16, 0),
        ] {
            let err = write_pack_with(
                &SliceReader(fx.model.clone()),
                fx.model.len() as u64,
                &fx.offsets,
                fx.rows_per_shard,
                encoding,
                width,
                FINGERPRINT,
                &fx.rank,
                10,
                5,
                &out,
                &mut |_, _| {},
            )
            .unwrap_err();
            assert!(err.contains("row width"), "{err}");
        }
        assert!(!out.exists());
    }

    /// Rewrite a good pack's header through `mutate`, recomputing a consistent
    /// `data_offset`, and keep the original data bytes.
    fn forge(good: &Path, bad: &Path, mutate: impl Fn(&mut serde_json::Value)) {
        let bytes = std::fs::read(good).unwrap();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let old_offset = round_up_data(16 + header_len as u64) as usize;
        let mut json: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
        mutate(&mut json);
        let mut offset = 4096u64;
        let new_header = loop {
            json["data_offset"] = serde_json::json!(offset);
            let text = serde_json::to_vec(&json).unwrap();
            let needed = round_up_data(16 + text.len() as u64);
            if needed == offset {
                break text;
            }
            offset = needed;
        };
        let mut forged = Vec::new();
        forged.extend_from_slice(PACK_MAGIC);
        forged.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
        forged.extend_from_slice(&new_header);
        forged.resize(offset as usize, 0);
        forged.extend_from_slice(&bytes[old_offset..]);
        std::fs::write(bad, forged).unwrap();
    }

    #[test]
    fn reader_rejects_malformed_packs() {
        let fx = fixture(2, 1024, 9);
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.plehotrows");
        pack(&fx, 100, 5, &good).unwrap();
        read_ple_hot_pack_header(&good).unwrap();
        let bad = dir.path().join("bad.plehotrows");
        let expect = |what: &str| {
            let err = read_ple_hot_pack_header(&bad).unwrap_err();
            assert!(err.contains(what), "expected `{what}` in: {err}");
        };

        // forge round-trips unchanged
        forge(&good, &bad, |_| {});
        read_ple_hot_pack_header(&bad).unwrap();

        let mut bytes = std::fs::read(&good).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&bad, &bytes).unwrap();
        expect("bad magic");

        let bytes = std::fs::read(&good).unwrap();
        std::fs::write(&bad, &bytes[..bytes.len() - 1]).unwrap();
        expect("file is");
        std::fs::write(&bad, &bytes[..10]).unwrap();
        assert!(read_ple_hot_pack_header(&bad).is_err());
        std::fs::write(&bad, &bytes[..20]).unwrap();
        expect("header length");
        let mut extended = bytes.clone();
        extended.push(0);
        std::fs::write(&bad, &extended).unwrap();
        expect("file is");

        forge(&good, &bad, |j| {
            j["data_bytes"] = serde_json::json!(j["data_bytes"].as_u64().unwrap() + 170)
        });
        expect("data_bytes");
        forge(&good, &bad, |j| {
            j["tier_rows"].as_array_mut().unwrap().pop();
        });
        expect("tier_rows has");
        forge(&good, &bad, |j| j["version"] = serde_json::json!(1));
        expect("version 1");
        forge(&good, &bad, |j| {
            j["encoded_row_bytes"] = serde_json::json!(34)
        });
        expect("encoded_row_bytes");
        // Encoding and size must agree: 170 B rows are not BF16 rows of 160 values.
        forge(&good, &bad, |j| {
            j["row_encoding"] = serde_json::json!("BF16")
        });
        expect("encoded_row_bytes");
        forge(&good, &bad, |j| j["row_width"] = serde_json::json!(96));
        expect("encoded_row_bytes");
        forge(&good, &bad, |j| j["row_width"] = serde_json::json!(0));
        expect("row width");
        forge(&good, &bad, |j| j["row_width"] = serde_json::json!(161));
        expect("row width");
        forge(&good, &bad, |j| {
            j["row_encoding"] = serde_json::json!("Q3_K")
        });
        expect("row_encoding");
        forge(&good, &bad, |j| j["row_encoding"] = serde_json::json!(8));
        expect("row_encoding");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("row_encoding");
        });
        expect("row_encoding");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("row_width");
        });
        expect("row_width");
        forge(&good, &bad, |j| j["tier_group"] = serde_json::json!(1));
        expect("tier_group");
        forge(&good, &bad, |j| j["kept_buckets"] = serde_json::json!(0));
        expect("kept_buckets");
        forge(&good, &bad, |j| j["kept_buckets"] = serde_json::json!(257));
        expect("kept_buckets");
        forge(&good, &bad, |j| {
            j["ple_fingerprint"] = serde_json::json!("ABC")
        });
        expect("ple_fingerprint");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("generator");
        });
        expect("generator");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("model_file_bytes");
        });
        expect("model_file_bytes");
        forge(&good, &bad, |j| {
            j["padded_rows"] = serde_json::json!("many")
        });
        expect("padded_rows");

        // data_offset that is not round_up(16 + header_len, 4096)
        let bytes = std::fs::read(&good).unwrap();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut json: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
        json["data_offset"] = serde_json::json!(8192);
        let text = serde_json::to_vec(&json).unwrap();
        assert_eq!(text.len(), header_len, "same-width edit");
        let mut forged = bytes.clone();
        forged[16..16 + header_len].copy_from_slice(&text);
        std::fs::write(&bad, &forged).unwrap();
        expect("data_offset");
    }

    // ---- v3 container: ple-table ------------------------------------------

    use super::table_support::{table_header, write_test_table};
    use crate::ple_hot_store::DEFAULT_TIER_BUCKETS;
    use crate::ple_q4::{lm4_codebook_bits, lm4_scale_grid_bits};

    const T_SHARDS: usize = 3;
    const T_RPS: usize = 640;
    const T_ROWS: usize = T_SHARDS * T_RPS;

    /// The v3 key order of the design (§8).
    const TABLE_KEYS: [&str; 29] = [
        "version",
        "kind",
        "order",
        "padded_rows",
        "rows_per_shard",
        "row_encoding",
        "row_width",
        "encoded_row_bytes",
        "ple_fingerprint",
        "rank_payload_sha256",
        "tier_group",
        "kept_buckets",
        "tier_rows",
        "rank_offset",
        "rank_bytes",
        "data_offset",
        "data_bytes",
        "shard_offsets",
        "model_file_bytes",
        "codebook",
        "scale",
        "source_encoding",
        "source_rows_sha256",
        "segment_bytes",
        "segment_sha256",
        "data_sha256",
        "base_index_sha256",
        "base_file_bytes",
        "generator",
    ];

    fn table_shards(encoding: RowEncoding) -> Vec<Vec<u8>> {
        let row_bytes = checked_row_bytes(encoding, TEST_ROW_WIDTH).unwrap();
        (0..T_SHARDS)
            .map(|s| {
                (0..T_RPS * row_bytes)
                    .map(|i| ((s * 131 + i * 7 + i / row_bytes) >> 1) as u8)
                    .collect()
            })
            .collect()
    }

    fn table_buckets() -> Vec<u8> {
        synthetic_buckets(T_ROWS, 21)
    }

    fn write_table(
        dir: &Path,
        order: PackOrder,
        encoding: RowEncoding,
    ) -> (PathBuf, PleHotPackHeader) {
        let path = dir.join(format!(
            "{}-{}.ple-table",
            order.name(),
            encoding.dtype_name()
        ));
        let buckets = table_buckets();
        let header = write_test_table(
            &path,
            order,
            encoding,
            TEST_ROW_WIDTH,
            T_RPS,
            &table_shards(encoding),
            (order == PackOrder::Rank).then_some(buckets.as_slice()),
            DEFAULT_TIER_BUCKETS,
        );
        (path, header)
    }

    fn header_len_of(bytes: &[u8]) -> usize {
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize
    }

    #[test]
    fn v3_table_headers_round_trip_for_both_orders_and_encodings() {
        let dir = tempfile::tempdir().unwrap();
        let buckets = table_buckets();
        for order in [PackOrder::Rank, PackOrder::Row] {
            for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
                let (path, written) = write_table(dir.path(), order, encoding);
                let read = read_ple_hot_pack_header(&path).unwrap();
                assert_eq!(read, written, "{order:?} {encoding:?}");
                assert_eq!(
                    (read.version, read.kind, read.order),
                    (PACK_VERSION, PackKind::PleTable, order)
                );
                let table = read.table.as_ref().unwrap();
                let q4 = encoding == RowEncoding::Q4Lm;
                assert_eq!(table.codebook_f32_bits, q4.then(lm4_codebook_bits));
                assert_eq!(table.scale_grid_f32_bits, q4.then(lm4_scale_grid_bits));
                assert_eq!(table.source_encoding, RowEncoding::Q8F16);
                assert_eq!(table.segment_bytes, TABLE_SEGMENT_BYTES);
                assert_eq!(
                    table.segment_sha256.len(),
                    table_segment_count(read.data_bytes, TABLE_SEGMENT_BYTES)
                );
                assert_eq!(read.kept_buckets, BUCKETS);
                assert!(read.shard_offsets.is_empty());
                assert_eq!(read.model_file_bytes, 0);
                assert_eq!(read.data_bytes, T_ROWS as u64 * read.encoded_row_bytes);

                let bytes = std::fs::read(&path).unwrap();
                let header_len = header_len_of(&bytes);
                assert_eq!(&bytes[..8], PACK_MAGIC);
                assert_eq!(bytes.len() as u64, read.data_offset + read.data_bytes);
                assert_eq!(read.data_offset % DATA_ALIGN, 0);
                let json: serde_json::Value =
                    serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
                let keys: Vec<&str> = json
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect();
                assert_eq!(keys, TABLE_KEYS, "field order");
                assert_eq!(json["kind"], "ple-table");
                assert_eq!(json["order"], order.name());
                let data = &bytes[read.data_offset as usize..];
                match order {
                    PackOrder::Rank => {
                        assert_eq!(read.rank_offset, round_up_data(16 + header_len as u64));
                        assert_eq!(read.rank_bytes, T_ROWS as u64);
                        assert_eq!(
                            read.data_offset,
                            round_up_data(read.rank_offset + read.rank_bytes)
                        );
                        let rank = &bytes[read.rank_offset as usize..][..T_ROWS];
                        assert_eq!(rank, buckets.as_slice());
                        assert_eq!(hex_lower(&Sha256::digest(rank)), read.rank_payload_sha256);
                        assert!(bytes[16 + header_len..read.rank_offset as usize]
                            .iter()
                            .all(|&b| b == 0));
                        assert!(bytes[(read.rank_offset + read.rank_bytes) as usize
                            ..read.data_offset as usize]
                            .iter()
                            .all(|&b| b == 0));
                        assert_eq!(read.tier_group, DEFAULT_TIER_BUCKETS);
                        assert_eq!(
                            read.tier_rows.len(),
                            tier_count(BUCKETS, DEFAULT_TIER_BUCKETS)
                        );
                        assert_eq!(read.tier_rows.iter().sum::<u64>(), T_ROWS as u64);
                    }
                    PackOrder::Row => {
                        assert_eq!((read.rank_offset, read.rank_bytes), (0, 0));
                        assert_eq!(read.data_offset, round_up_data(16 + header_len as u64));
                        assert_eq!(read.rank_payload_sha256, "0".repeat(64));
                        assert_eq!(
                            (read.tier_group, read.tier_rows.clone()),
                            (BUCKETS, vec![T_ROWS as u64])
                        );
                        assert!(bytes[16 + header_len..read.data_offset as usize]
                            .iter()
                            .all(|&b| b == 0));
                        assert_eq!(data, table_shards(encoding).concat().as_slice());
                    }
                }

                // Re-encoding the read header reproduces the file's header bytes.
                let mut again = read.clone();
                let encoded = encode_ple_table_header(&mut again).unwrap();
                assert_eq!(again, read);
                assert_eq!(&bytes[..encoded.len()], encoded.as_slice());
                assert_eq!(
                    encoded.len() as u64,
                    if order == PackOrder::Rank {
                        read.rank_offset
                    } else {
                        read.data_offset
                    }
                );
            }
        }
    }

    /// A v2 pack reads with the implied v3 fields; the writer still emits v2.
    #[test]
    fn v2_packs_read_with_implied_kind_and_order() {
        let fx = fixture(2, 1024, 9);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("v2.plehotrows");
        let written = pack(&fx, 100, 5, &out).unwrap();
        let read = read_ple_hot_pack_header(&out).unwrap();
        assert_eq!(read, written);
        assert_eq!(
            (
                read.version,
                read.kind,
                read.order,
                read.rank_offset,
                read.rank_bytes
            ),
            (HOT_PACK_VERSION, PackKind::HotPack, PackOrder::Rank, 0, 0)
        );
        assert!(read.table.is_none());
        assert_eq!(read.generator, GENERATOR);
        let bytes = std::fs::read(&out).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len_of(&bytes)]).unwrap();
        assert_eq!(json["version"], 2);
        assert!(json.get("kind").is_none());
    }

    /// Same-width text edit of a file's header (layout rules stay untouched).
    fn patch_header_text(good: &Path, bad: &Path, from: &str, to: &str) {
        assert_eq!(from.len(), to.len());
        let mut bytes = std::fs::read(good).unwrap();
        let header_len = header_len_of(&bytes);
        let text = String::from_utf8(bytes[16..16 + header_len].to_vec()).unwrap();
        assert!(text.contains(from), "{from} not in header");
        let text = text.replacen(from, to, 1);
        bytes[16..16 + header_len].copy_from_slice(text.as_bytes());
        std::fs::write(bad, bytes).unwrap();
    }

    /// Rewrite a good table's header through `mutate` with the layout offsets
    /// recomputed from the (mutated) `order` / `rank_bytes`, keeping the
    /// original rank and data bytes.
    fn forge_table(good: &Path, bad: &Path, mutate: impl Fn(&mut serde_json::Value)) {
        let bytes = std::fs::read(good).unwrap();
        let header_len = header_len_of(&bytes);
        let mut json: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
        let old_rank_offset = json["rank_offset"].as_u64().unwrap() as usize;
        let old_rank_bytes = json["rank_bytes"].as_u64().unwrap() as usize;
        let old_data_offset = json["data_offset"].as_u64().unwrap() as usize;
        mutate(&mut json);
        let rank_order = json["order"] == "rank";
        let mut candidate = DATA_ALIGN;
        let (text, rank_offset, data_offset) = loop {
            if rank_order {
                json["rank_offset"] = serde_json::json!(candidate);
                let rank_bytes = json["rank_bytes"].as_u64().unwrap_or(0);
                json["data_offset"] = serde_json::json!(round_up_data(candidate + rank_bytes));
            } else {
                json["data_offset"] = serde_json::json!(candidate);
            }
            let text = serde_json::to_vec(&json).unwrap();
            let needed = round_up_data(16 + text.len() as u64);
            if needed == candidate {
                break (
                    text,
                    json["rank_offset"].as_u64().unwrap(),
                    json["data_offset"].as_u64().unwrap(),
                );
            }
            candidate = needed;
        };
        let mut forged = Vec::new();
        forged.extend_from_slice(PACK_MAGIC);
        forged.extend_from_slice(&(text.len() as u64).to_le_bytes());
        forged.extend_from_slice(&text);
        if rank_order {
            forged.resize(rank_offset as usize, 0);
            forged.extend_from_slice(&bytes[old_rank_offset..old_rank_offset + old_rank_bytes]);
        }
        forged.resize(data_offset as usize, 0);
        forged.extend_from_slice(&bytes[old_data_offset..]);
        std::fs::write(bad, forged).unwrap();
    }

    type Mutation = Box<dyn Fn(&mut serde_json::Value)>;

    /// Box a header mutation (the bound gives the closure its signature).
    fn m(mutation: impl Fn(&mut serde_json::Value) + 'static) -> Mutation {
        Box::new(mutation)
    }

    #[test]
    fn table_reader_checks_every_ple_table_rule() {
        use serde_json::{json, Value};
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("bad.ple-table");
        let expect = |what: &str| {
            let err = read_ple_hot_pack_header(&bad).unwrap_err();
            assert!(err.contains(what), "expected `{what}` in: {err}");
        };
        for order in [PackOrder::Rank, PackOrder::Row] {
            for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
                let (good, _) = write_table(dir.path(), order, encoding);
                let q4 = encoding == RowEncoding::Q4Lm;
                // forge round-trips unchanged
                forge_table(&good, &bad, |_| {});
                read_ple_hot_pack_header(&bad).unwrap();

                let mut cases: Vec<(&str, Mutation, &str)> = vec![
                    (
                        "kept",
                        m(|j| j["kept_buckets"] = json!(257)),
                        "kept_buckets",
                    ),
                    (
                        "shard_offsets",
                        m(|j| j["shard_offsets"] = json!([0])),
                        "shard_offsets",
                    ),
                    (
                        "model_file_bytes",
                        m(|j| j["model_file_bytes"] = json!(1)),
                        "model_file_bytes",
                    ),
                    ("kind", m(|j| j["kind"] = json!("bogus")), "kind"),
                    ("order", m(|j| j["order"] = json!("bogus")), "order"),
                    (
                        "kind hot-pack",
                        m(|j| j["kind"] = json!("hot-pack")),
                        "must be null",
                    ),
                    (
                        "BF16 rows",
                        m(|j| {
                            j["row_encoding"] = json!("BF16");
                            j["encoded_row_bytes"] = json!(320);
                            j["data_bytes"] = json!(T_ROWS * 320);
                        }),
                        "Q8_0 or Q4LM",
                    ),
                    (
                        "source_encoding BF16",
                        m(|j| j["source_encoding"] = json!("BF16")),
                        "source_encoding",
                    ),
                    (
                        "source_encoding Q4LM",
                        m(|j| j["source_encoding"] = json!("Q4LM")),
                        "source_encoding",
                    ),
                    (
                        "source_encoding null",
                        m(|j| j["source_encoding"] = Value::Null),
                        "source_encoding",
                    ),
                    (
                        "source_rows_sha256",
                        m(|j| j["source_rows_sha256"] = json!("xyz")),
                        "source_rows_sha256",
                    ),
                    (
                        "segment_bytes 0",
                        m(|j| j["segment_bytes"] = json!(0)),
                        "segment_bytes",
                    ),
                    (
                        "segment_bytes mismatch",
                        m(|j| j["segment_bytes"] = json!(4096)),
                        "segment_sha256 has",
                    ),
                    (
                        "segment count",
                        m(|j| {
                            j["segment_sha256"].as_array_mut().unwrap().pop();
                        }),
                        "segment_sha256",
                    ),
                    (
                        "segment digest",
                        m(|j| j["segment_sha256"][0] = json!("ab".repeat(32))),
                        "data_sha256",
                    ),
                    (
                        "segment hex",
                        m(|j| j["segment_sha256"][0] = json!("QQ")),
                        "segment_sha256",
                    ),
                    (
                        "data_sha256",
                        m(|j| j["data_sha256"] = json!("ab".repeat(32))),
                        "data_sha256",
                    ),
                    (
                        "base_index_sha256",
                        m(|j| j["base_index_sha256"] = Value::Null),
                        "base_index_sha256",
                    ),
                    (
                        "base_file_bytes 0",
                        m(|j| j["base_file_bytes"] = json!(0)),
                        "base_file_bytes",
                    ),
                    (
                        "base_file_bytes null",
                        m(|j| j["base_file_bytes"] = Value::Null),
                        "base_file_bytes",
                    ),
                    (
                        "tier_group",
                        m(|j| j["tier_group"] = json!(1)),
                        "tier_group",
                    ),
                    (
                        "missing field",
                        m(|j| {
                            j.as_object_mut().unwrap().remove("segment_bytes");
                        }),
                        "segment_bytes",
                    ),
                    (
                        "data_bytes",
                        m(|j| j["data_bytes"] = json!(j["data_bytes"].as_u64().unwrap() + 90)),
                        "data_bytes",
                    ),
                    (
                        "tier_rows short",
                        m(|j| {
                            j["tier_rows"].as_array_mut().unwrap().pop();
                        }),
                        "tier_rows has",
                    ),
                    (
                        "fingerprint",
                        m(|j| j["ple_fingerprint"] = json!("ABC")),
                        "ple_fingerprint",
                    ),
                    // Codebook / scale are present exactly for Q4LM.
                ];
                if q4 {
                    cases.extend([
                        (
                            "no codebook",
                            m(|j| j["codebook"] = Value::Null),
                            "codebook",
                        ),
                        ("no scale", m(|j| j["scale"] = Value::Null), "scale"),
                        (
                            "codebook name",
                            m(|j| j["codebook"]["name"] = json!("other")),
                            "codebook",
                        ),
                        (
                            "codebook bits",
                            m(|j| {
                                j["codebook"]["f32_bits"].as_array_mut().unwrap().pop();
                            }),
                            "f32_bits",
                        ),
                        (
                            "scale group",
                            m(|j| j["scale"]["group"] = json!(64)),
                            "scale",
                        ),
                        (
                            "scale rule",
                            m(|j| j["scale"]["rule"] = json!("max")),
                            "scale",
                        ),
                        (
                            "scale dtype",
                            m(|j| j["scale"]["dtype"] = json!("f32")),
                            "scale",
                        ),
                        (
                            "scale grid",
                            m(|j| {
                                j["scale"]["grid_f32_bits"].as_array_mut().unwrap().pop();
                            }),
                            "grid_f32_bits",
                        ),
                    ]);
                } else {
                    cases.extend([
                        (
                            "codebook on Q8",
                            m(|j| {
                                j["codebook"] = json!({
                                    "name": "lloyd-max-normal-4bit",
                                    "f32_bits": lm4_codebook_bits(),
                                })
                            }),
                            "codebook",
                        ),
                        (
                            "scale on Q8",
                            m(|j| {
                                j["scale"] = json!({
                                    "dtype": "f16", "group": 32, "rule": "rms-mse-grid",
                                    "grid_f32_bits": lm4_scale_grid_bits(),
                                })
                            }),
                            "scale",
                        ),
                    ]);
                }
                match order {
                    PackOrder::Rank => cases.extend([
                        (
                            "rank_bytes",
                            m(|j| j["rank_bytes"] = json!(T_ROWS - 1)),
                            "rank_bytes",
                        ),
                        (
                            "rank sha",
                            m(|j| j["rank_payload_sha256"] = json!("zz")),
                            "rank_payload_sha256",
                        ),
                        (
                            "tier_rows sum",
                            m(|j| {
                                let rows = j["tier_rows"][0].as_u64().unwrap();
                                j["tier_rows"][0] = json!(rows - 1);
                                let row_bytes = j["encoded_row_bytes"].as_u64().unwrap();
                                j["data_bytes"] =
                                    json!(j["data_bytes"].as_u64().unwrap() - row_bytes);
                            }),
                            "keeps every row",
                        ),
                    ]),
                    PackOrder::Row => cases.extend([
                        (
                            "row kept",
                            m(|j| j["kept_buckets"] = json!(255)),
                            "kept_buckets",
                        ),
                        (
                            "row tier_group",
                            m(|j| j["tier_group"] = json!(2)),
                            "tier_rows has",
                        ),
                        (
                            "row rank sha",
                            m(|j| j["rank_payload_sha256"] = json!("1".repeat(64))),
                            "rank_payload_sha256",
                        ),
                        (
                            "row rank_offset",
                            m(|j| j["rank_offset"] = json!(4096)),
                            "rank_offset",
                        ),
                        (
                            "row rank_bytes",
                            m(|j| j["rank_bytes"] = json!(1)),
                            "rank_bytes",
                        ),
                    ]),
                }
                for (name, mutate, want) in &cases {
                    forge_table(&good, &bad, mutate);
                    let err = read_ple_hot_pack_header(&bad).expect_err(&format!(
                        "{order:?} {encoding:?}: `{name}` must be rejected"
                    ));
                    assert!(
                        err.contains(want),
                        "{order:?} {encoding:?} `{name}`: expected `{want}` in: {err}"
                    );
                }

                // Layout: rank at round_up(16 + header_len), data after the rank.
                match order {
                    PackOrder::Rank => {
                        patch_header_text(
                            &good,
                            &bad,
                            "\"rank_offset\":4096",
                            "\"rank_offset\":8192",
                        );
                        expect("rank_offset");
                        patch_header_text(
                            &good,
                            &bad,
                            "\"data_offset\":8192",
                            "\"data_offset\":4096",
                        );
                        expect("data_offset");
                    }
                    PackOrder::Row => {
                        patch_header_text(
                            &good,
                            &bad,
                            "\"data_offset\":4096",
                            "\"data_offset\":8192",
                        );
                        expect("data_offset");
                    }
                }

                // File length is exact.
                let bytes = std::fs::read(&good).unwrap();
                std::fs::write(&bad, &bytes[..bytes.len() - 1]).unwrap();
                expect("file is");
                let mut longer = bytes.clone();
                longer.push(0);
                std::fs::write(&bad, &longer).unwrap();
                expect("file is");
            }
        }
    }

    /// A v3 `hot-pack` (nobody writes one yet) reads: rank-ordered, no embedded
    /// rank, every ple-table field null.
    #[test]
    fn v3_hot_pack_headers_read_and_require_null_table_fields() {
        let fx = fixture(2, 1024, 9);
        let dir = tempfile::tempdir().unwrap();
        let v2 = dir.path().join("v2.plehotrows");
        pack(&fx, 100, 5, &v2).unwrap();
        let v3 = dir.path().join("v3.plehotrows");
        let to_v3 = |j: &mut serde_json::Value| {
            let old = j.as_object().unwrap().clone();
            let mut new = serde_json::Map::new();
            new.insert("version".into(), serde_json::json!(3));
            new.insert("kind".into(), serde_json::json!("hot-pack"));
            new.insert("order".into(), serde_json::json!("rank"));
            for (key, value) in old {
                if key != "version" {
                    new.insert(key, value);
                }
            }
            for key in ["rank_offset", "rank_bytes"] {
                new.insert(key.into(), serde_json::json!(0));
            }
            for key in [
                "codebook",
                "scale",
                "source_encoding",
                "source_rows_sha256",
                "data_sha256",
                "base_index_sha256",
                "base_file_bytes",
            ] {
                new.insert(key.into(), serde_json::Value::Null);
            }
            new.insert("segment_bytes".into(), serde_json::json!(0));
            new.insert("segment_sha256".into(), serde_json::json!([]));
            *j = serde_json::Value::Object(new);
        };
        forge(&v2, &v3, to_v3);
        let read = read_ple_hot_pack_header(&v3).unwrap();
        assert_eq!(
            (
                read.version,
                read.kind,
                read.order,
                read.rank_offset,
                read.rank_bytes
            ),
            (PACK_VERSION, PackKind::HotPack, PackOrder::Rank, 0, 0)
        );
        assert!(read.table.is_none());
        assert!(!read.shard_offsets.is_empty());

        let bad = dir.path().join("bad.plehotrows");
        let cases: Vec<(Mutation, &str)> = vec![
            (m(|j| j["order"] = serde_json::json!("row")), "rank-ordered"),
            (m(|j| j["rank_bytes"] = serde_json::json!(5)), "rank"),
            (
                m(|j| j["source_encoding"] = serde_json::json!("Q8_0")),
                "must be null",
            ),
            (
                m(|j| j["base_file_bytes"] = serde_json::json!(1)),
                "must be null",
            ),
            (
                m(|j| j["segment_bytes"] = serde_json::json!(1)),
                "segment_bytes",
            ),
            (
                m(|j| j["segment_sha256"] = serde_json::json!(["ab".repeat(32)])),
                "segment_sha256",
            ),
        ];
        for (mutate, want) in &cases {
            forge(&v3, &bad, mutate);
            let err = read_ple_hot_pack_header(&bad).unwrap_err();
            assert!(err.contains(want), "expected `{want}` in: {err}");
        }
    }

    #[test]
    fn segment_count_rounds_up() {
        assert_eq!(table_segment_count(0, TABLE_SEGMENT_BYTES), 0);
        assert_eq!(table_segment_count(1, TABLE_SEGMENT_BYTES), 1);
        assert_eq!(
            table_segment_count(TABLE_SEGMENT_BYTES, TABLE_SEGMENT_BYTES),
            1
        );
        assert_eq!(
            table_segment_count(TABLE_SEGMENT_BYTES + 1, TABLE_SEGMENT_BYTES),
            2
        );
        assert_eq!(table_segment_count(28_800_000_000, TABLE_SEGMENT_BYTES), 27);
        assert_eq!(table_segment_count(10, 0), 0);
    }

    #[test]
    fn table_header_encoder_refuses_other_kinds_and_invalid_fields() {
        let buckets = table_buckets();
        let mut header = table_header(
            PackOrder::Rank,
            RowEncoding::Q8F16,
            T_RPS,
            TEST_ROW_WIDTH,
            T_SHARDS,
            Some(&buckets),
            DEFAULT_TIER_BUCKETS,
        );
        let mut hot = header.clone();
        hot.kind = PackKind::HotPack;
        assert!(encode_ple_table_header(&mut hot).is_err());
        let mut no_table = header.clone();
        no_table.table = None;
        assert!(encode_ple_table_header(&mut no_table).is_err());
        let mut bad_kept = header.clone();
        bad_kept.kept_buckets = 100;
        assert!(encode_ple_table_header(&mut bad_kept).is_err());
        // A valid one sets its offsets and the version.
        header.version = 0;
        let bytes = encode_ple_table_header(&mut header).unwrap();
        assert_eq!(header.version, PACK_VERSION);
        assert_eq!(header.rank_bytes, T_ROWS as u64);
        assert_eq!(bytes.len() as u64, header.rank_offset);
        assert_eq!(header.data_offset % DATA_ALIGN, 0);
    }
}
