// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE table conversion: derive a Q4 (Lloyd-Max) PLE table (`<model>.ple-q4`)
//! from Q8F16 rows, and split a monolithic Qwen4 HFQM into a base file plus a
//! Q8 PLE table (`<base>.ple-q8`).
//!
//! Both sidecars are `HFPLEHP1` version 3 `ple-table` containers
//! (`crate::ple_hot_pack`): the Q8 table written by [`split`] is **row order**
//! (shard `s` at `data_offset + s * rows_per_shard * 170`, exactly the model's
//! natural layout); the Q4 table written by [`derive_q4`] is **rank order**
//! when a `.plehot` rank is available (rows laid out tier by tier, hottest
//! first, with the rank embedded) and row order otherwise.
//!
//! Both operations stream the 54.4 GB (Flash-Next) PLE n-gram table with
//! positional reads/writes over `available_parallelism()` workers, hash every
//! byte they read and write, and publish their outputs atomically:
//!
//! * outputs are written to `<out>.tmp` (`<base>.ple-q8.tmp` for a split),
//!   each guarded by an exclusive non-blocking `flock` — a second converter on
//!   the same output fails with a clear error, a stale tmp left by a crash is
//!   truncated and restarted;
//! * the container header is written **last** (a half-written file is never
//!   mistaken for a valid one), the file is `sync_all`ed, renamed into place and
//!   the directory is fsynced;
//! * free space is checked with `statvfs` before anything is written
//!   (output bytes + 1 GiB);
//! * tmp files are removed on every error path;
//! * output bytes are deterministic (no paths, times or thread-count effects).
//!
//! # Derive
//!
//! The Q8 rows come from `<model>.ple-q8` when that exists (it must bind to the
//! model and the digests recomputed while reading must match its recorded
//! `source_rows_sha256`), otherwise from the model's embedded shards. The
//! encoding is [`crate::ple_q4::encode_q4lm_row_from_q8`], bit-identical in
//! decode to the `lm4` PLE emulation diagnostic (`diag/ple-lm-emulate`).
//!
//! Rank order places the row of global index `r` with bucket `b` at
//! `data_offset + (tier_base[t] + slot(r)) * 90` where `t = b / tier_group` and
//! `slot(r)` counts the tier-`t` rows before `r` (`HotRowStore::slot`). Workers
//! take shards, walk them in global-row chunks, encode each chunk, group its
//! rows by tier and write each tier's run with one contiguous `pwrite`; the
//! per-chunk per-tier slot bases come from one pass over the buckets.
//! `data_sha256` is computed by re-reading the written data in 1 GiB segments
//! from disk, which doubles as the write verification.
//!
//! # Split
//!
//! The base keeps the model's metadata JSON byte for byte with the
//! `hipfire_ple_rows` marker inserted before the final `}`, the original tensor
//! index minus the PLE shards (same order, same bytes), and the kept payloads
//! copied verbatim in index order. The base `data_offset` is the smallest value
//! at or past the index end that is congruent to the original offset of the
//! first kept tensor modulo 4096, so every kept payload keeps the 4 KiB
//! alignment it had in the source. Both outputs are re-read from disk and
//! verified against the source before they are renamed into place; the segment
//! digests of the Q8 table are computed in that re-read.

// Non-Unix builds compile only the platform-independent planning/format logic
// and explicit "unsupported" entry points; the I/O implementation (positional
// reads/writes, flock, statvfs, fadvise, sync_file_range) is `cfg(unix)`, so
// the planning code it alone consumes is intentionally unreferenced there.
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use hipfire_config::memory::{ple_hot_rank_file, ple_sidecar_path, PleFormat};
use sha2::{Digest, Sha256};

use crate::external_rows::RowEncoding;
use crate::hfq::{HfqFile, HfqTensorInfo};
use crate::ple_hot_pack::{
    encode_ple_table_header, read_ple_hot_pack_header, table_segment_count, PackKind, PackOrder,
    PleHotPackHeader, PleTableFields, TABLE_SEGMENT_BYTES,
};
use crate::ple_hot_store::{PleHotRank, BUCKETS, DEFAULT_TIER_BUCKETS};
use crate::ple_q4::{
    encode_q4lm_row_from_q8, lm4_codebook_bits, lm4_scale_grid_bits, q4lm_row_bytes,
};
use crate::ple_sidecar::{
    merkle_sha256, ple_fingerprint_from_metadata, ple_shard_name, PLE_BASE_MARKER_KEY,
    PLE_SHARD_PREFIX, PLE_SHARD_SUFFIX,
};

/// `HFPLEHP1` container version of a `ple-table`.
const TABLE_VERSION: u64 = 3;
/// HFQM quant-type byte of a Q8F16 (`Q8_0`) tensor.
const Q8_QUANT_TYPE: u8 = 3;
/// Quantisation group of a Q8F16 tensor.
const Q8_GROUP: u32 = 32;
/// Weights per Q8F16 block.
const Q8_BLOCK_WEIGHTS: usize = 32;
/// Bytes per Q8F16 block: an f16 scale plus 32 signed bytes.
const Q8_BLOCK_BYTES: usize = 34;
/// HFQM fixed header bytes.
const HFQM_HEADER_BYTES: u64 = 32;
/// HFQM metadata-overlay bit in the version word.
const HFQM_OVERLAY_FLAG: u32 = 1 << 31;
/// HFQM base format version.
const HFQM_VERSION: u32 = 1;
/// Alignment of the HFQM data section and of the kept payloads.
const DATA_ALIGN: u64 = 4096;
/// Free space required beyond the output bytes.
const SPACE_MARGIN: u64 = 1 << 30;
/// Default positional-I/O chunk per worker.
const DEFAULT_CHUNK_BYTES: usize = 32 << 20;
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// One progress report. `phase` is `"derive"`, `"copy-ple"`, `"copy-base"` or
/// `"verify"`; the byte counters are phase-relative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleProgress {
    pub phase: &'static str,
    pub done_bytes: u64,
    pub total_bytes: u64,
}

/// Result of [`derive_q4`].
#[derive(Debug, Clone)]
pub struct DeriveReport {
    /// The `<model>.ple-q4` that was written.
    pub output: PathBuf,
    /// The file the Q8 rows were read from (the q8 sidecar, or the model).
    pub source: PathBuf,
    /// Bytes of the output file.
    pub bytes: u64,
    /// `data_sha256` of the table (sha256 of its segment digests).
    pub payload_sha256: String,
    pub source_rows_sha256: String,
    pub elapsed: Duration,
    pub threads: usize,
    /// Row order of the table: `"rank"` or `"row"`.
    pub order: &'static str,
}

/// Result of [`split`].
#[derive(Debug, Clone)]
pub struct SplitReport {
    pub base: PathBuf,
    pub base_bytes: u64,
    pub base_index_sha256: String,
    pub ple_q8: PathBuf,
    pub ple_q8_bytes: u64,
    pub q8_rows_sha256: String,
    pub elapsed: Duration,
}

/// Parallelism and I/O granularity; tests shrink them to exercise multi-chunk,
/// multi-segment, multi-worker paths on tiny models.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tuning {
    pub threads: usize,
    pub chunk_bytes: usize,
    pub segment_bytes: u64,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            segment_bytes: TABLE_SEGMENT_BYTES,
        }
    }
}

type Progress<'a> = &'a mut (dyn FnMut(&PleProgress) + Send);

/// Error of every conversion entry point on a non-Unix host, raised before any
/// file is opened or created.
#[cfg(not(unix))]
fn unix_only(ctx: &str, command: &str) -> String {
    format!(
        "{ctx}: PLE table conversion requires a Unix (Linux) host: it needs Unix positional \
         file I/O, flock and statvfs, which this build does not implement. Run \
         `hipfire ple {command}` on a Unix (Linux) host"
    )
}

/// `qwen3.8-flash-next.mq4-xts` → `qwen3.8-flash-next-base.mq4-xts`: "-base"
/// goes before the extension (appended when there is none).
pub fn default_split_base_path(model: &Path) -> PathBuf {
    let Some(stem) = model.file_stem() else {
        return append_suffix(model, "-base");
    };
    let mut name = OsString::from(stem);
    name.push("-base");
    if let Some(ext) = model.extension() {
        name.push(".");
        name.push(ext);
    }
    model.with_file_name(name)
}

// ─── table header ────────────────────────────────────────────────────────────

/// Everything of a `ple-table` header except the digests that are only known
/// after the data is written.
struct TableParams<'a> {
    layout: PleShardLayout,
    encoding: RowEncoding,
    row_bytes: usize,
    rank_order: bool,
    fingerprint: &'a str,
    /// Hex sha256 of the embedded rank bytes (zeros for row order).
    rank_sha256: &'a str,
    tier_group: usize,
    tier_rows: &'a [u64],
    rank_bytes: u64,
    data_bytes: u64,
    segment_bytes: u64,
    source_rows_sha256: &'a str,
    base_index_sha256: &'a str,
    base_file_bytes: u64,
    generator: &'static str,
}

impl TableParams<'_> {
    fn header(&self, segment_sha256: Vec<String>, data_sha256: &str) -> PleHotPackHeader {
        let q4 = self.encoding == RowEncoding::Q4Lm;
        PleHotPackHeader {
            version: TABLE_VERSION,
            kind: PackKind::PleTable,
            order: if self.rank_order {
                PackOrder::Rank
            } else {
                PackOrder::Row
            },
            padded_rows: self.layout.padded_rows(),
            rows_per_shard: self.layout.rows_per_shard as u64,
            row_encoding: self.encoding,
            row_width: self.layout.row_width as u64,
            encoded_row_bytes: self.row_bytes as u64,
            ple_fingerprint: self.fingerprint.to_string(),
            rank_payload_sha256: self.rank_sha256.to_string(),
            tier_group: self.tier_group,
            kept_buckets: BUCKETS,
            tier_rows: self.tier_rows.to_vec(),
            rank_offset: 0,
            rank_bytes: self.rank_bytes,
            data_offset: 0,
            data_bytes: self.data_bytes,
            shard_offsets: Vec::new(),
            model_file_bytes: 0,
            table: Some(PleTableFields {
                codebook_f32_bits: q4.then(lm4_codebook_bits),
                scale_grid_f32_bits: q4.then(lm4_scale_grid_bits),
                source_encoding: RowEncoding::Q8F16,
                source_rows_sha256: self.source_rows_sha256.to_string(),
                segment_bytes: self.segment_bytes,
                segment_sha256,
                data_sha256: data_sha256.to_string(),
                base_index_sha256: self.base_index_sha256.to_string(),
                base_file_bytes: self.base_file_bytes,
            }),
            generator: self.generator.to_string(),
        }
    }

    /// A header with zero segment digests (and the matching `data_sha256`, so
    /// the encoder's consistency check holds): its encoded length (hence
    /// `rank_offset` and `data_offset`) is already final.
    fn probe(&self) -> Result<(PleHotPackHeader, Vec<u8>), String> {
        let segments = table_segment_count(self.data_bytes, self.segment_bytes);
        let zero_digests = vec![[0u8; 32]; segments];
        let data_sha256 = hex(&merkle_sha256(&zero_digests));
        let mut header = self.header(vec![ZERO_SHA256.to_string(); segments], &data_sha256);
        let bytes = encode_ple_table_header(&mut header)?;
        Ok((header, bytes))
    }

    /// The final header; its offsets must equal the probe's.
    fn finish(
        &self,
        probe: &PleHotPackHeader,
        segment_digests: &[[u8; 32]],
    ) -> Result<(PleHotPackHeader, Vec<u8>), String> {
        let data_sha256 = hex(&merkle_sha256(segment_digests));
        let mut header = self.header(
            segment_digests.iter().map(|d| hex(d)).collect(),
            &data_sha256,
        );
        let bytes = encode_ple_table_header(&mut header)?;
        if header.rank_offset != probe.rank_offset || header.data_offset != probe.data_offset {
            return Err(format!(
                "header layout changed between probe and final (rank_offset {} vs {}, \
                 data_offset {} vs {})",
                header.rank_offset, probe.rank_offset, header.data_offset, probe.data_offset
            ));
        }
        Ok((header, bytes))
    }
}

/// Re-read `[data_offset, data_offset + data_bytes)` of `file` in
/// `segment_bytes` segments over the worker pool and return the segment
/// digests. This is the content digest of a table and, because the caller
/// flushed and dropped the page cache first, the verification that the bytes
/// reached the disk.
#[cfg(unix)]
fn segment_digests(
    file: &File,
    path: &Path,
    data_offset: u64,
    data_bytes: u64,
    tuning: &Tuning,
    progress: Progress<'_>,
) -> Result<Vec<[u8; 32]>, String> {
    let seg = tuning.segment_bytes;
    let buf_len = (tuning.chunk_bytes as u64).min(seg).max(1) as usize;
    run_parallel(
        "verify",
        data_bytes,
        table_segment_count(data_bytes, seg),
        tuning.threads,
        progress,
        || vec![0u8; buf_len],
        |buf, i, done| {
            let start = i as u64 * seg;
            let end = (start + seg).min(data_bytes);
            let mut hash = Sha256::new();
            let mut off = start;
            while off < end {
                let n = (end - off).min(buf_len as u64) as usize;
                file.read_exact_at(&mut buf[..n], data_offset + off)
                    .map_err(|e| format!("verify read {} at {}: {e}", path.display(), off))?;
                drop_cache(file, data_offset + off, n as u64);
                hash.update(&buf[..n]);
                done.fetch_add(n as u64, Ordering::Relaxed);
                off += n as u64;
            }
            Ok(<[u8; 32]>::from(hash.finalize()))
        },
    )
}

// ─── derive ──────────────────────────────────────────────────────────────────

/// Rebind an existing Q4 table after proving the embedded Q8 PLE shards are
/// unchanged. The old base must bind the source table. Publication replaces
/// the destination directory entry, never its inode (including hard links).
#[cfg(unix)]
pub fn rebind_q4(
    old_base: &Path,
    new_base: &Path,
    source: &Path,
    progress: Progress<'_>,
) -> Result<PathBuf, String> {
    let mut old = HfqFile::open(old_base).map_err(|e| e.to_string())?;
    let mut new = HfqFile::open(new_base).map_err(|e| e.to_string())?;
    old.drop_mmap();
    new.drop_mmap();
    let rows = crate::ple_sidecar::PleRowsFile::open(source)?;
    if rows.encoding() != RowEncoding::Q4Lm {
        return Err("ple rebind: source is not Q4".into());
    }
    let mut header = rows.header().clone();
    old.attach_ple_rows(rows)?;
    if ple_fingerprint_from_metadata(&old.metadata_json)?
        != ple_fingerprint_from_metadata(&new.metadata_json)?
    {
        return Err("ple rebind: PLE metadata changed".into());
    }
    let old_shards: Vec<_> = old
        .tensor_infos()
        .iter()
        .filter(|t| t.name.starts_with(PLE_SHARD_PREFIX) && t.name.ends_with(PLE_SHARD_SUFFIX))
        .collect();
    let new_shards: Vec<_> = new
        .tensor_infos()
        .iter()
        .filter(|t| t.name.starts_with(PLE_SHARD_PREFIX) && t.name.ends_with(PLE_SHARD_SUFFIX))
        .collect();
    if old_shards.is_empty() || old_shards.len() != new_shards.len() {
        return Err("ple rebind: requires matching embedded PLE shards".into());
    }
    let old_file = File::open(old_base).map_err(|e| e.to_string())?;
    let new_file = File::open(new_base).map_err(|e| e.to_string())?;
    let mut left = vec![0u8; DEFAULT_CHUNK_BYTES];
    let mut right = vec![0u8; DEFAULT_CHUNK_BYTES];
    for (a, b) in old_shards.iter().zip(&new_shards) {
        if a.name != b.name
            || a.shape != b.shape
            || a.quant_type != b.quant_type
            || a.group_size != b.group_size
            || a.data_size != b.data_size
        {
            return Err("ple rebind: PLE descriptor changed".into());
        }
        let mut offset = 0;
        while offset < a.data_size {
            let n = left.len().min(a.data_size - offset);
            old_file
                .read_exact_at(&mut left[..n], (a.data_offset + offset) as u64)
                .map_err(|e| e.to_string())?;
            new_file
                .read_exact_at(&mut right[..n], (b.data_offset + offset) as u64)
                .map_err(|e| e.to_string())?;
            if left[..n] != right[..n] {
                return Err(format!("ple rebind: source rows changed in {}", a.name));
            }
            offset += n;
        }
    }
    let input = File::open(source).map_err(|e| e.to_string())?;
    let tuning = Tuning {
        threads: 1,
        segment_bytes: header
            .table
            .as_ref()
            .ok_or("ple rebind: missing table")?
            .segment_bytes,
        ..Tuning::default()
    };
    let segments = segment_digests(
        &input,
        source,
        header.data_offset,
        header.data_bytes,
        &tuning,
        progress,
    )?;
    if segments.iter().map(|s| hex(s)).collect::<Vec<_>>()
        != header
            .table
            .as_ref()
            .ok_or("ple rebind: missing table")?
            .segment_sha256
    {
        return Err("ple rebind: source payload digest mismatch".into());
    }
    let original = header.clone();
    let table = header.table.as_mut().ok_or("ple rebind: missing table")?;
    table.base_file_bytes = new.file_len();
    table.base_index_sha256 = hex(&new.index_sha256().map_err(|e| e.to_string())?);
    let bytes = encode_ple_table_header(&mut header)?;
    if header.rank_offset != original.rank_offset || header.data_offset != original.data_offset {
        return Err("ple rebind: binding changes container layout".into());
    }
    let out = ple_sidecar_path(new_base, PleFormat::Q4);
    if out == source {
        return Err("ple rebind: source and destination must differ".into());
    }
    let tmp = append_suffix(&out, ".tmp");
    let mut cleanup = TmpCleanup::default();
    let output = open_locked_tmp(&tmp, "ple rebind")?;
    cleanup.add(&tmp);
    let size = input.metadata().map_err(|e| e.to_string())?.len();
    check_space(parent_dir(&out), size, "ple rebind")?;
    output.set_len(size).map_err(|e| e.to_string())?;
    let mut offset = 0;
    while offset < size {
        let n = left.len().min((size - offset) as usize);
        input
            .read_exact_at(&mut left[..n], offset)
            .map_err(|e| e.to_string())?;
        output
            .write_all_at(&left[..n], offset)
            .map_err(|e| e.to_string())?;
        output
            .read_exact_at(&mut right[..n], offset)
            .map_err(|e| e.to_string())?;
        if left[..n] != right[..n] {
            return Err("ple rebind: copied bytes differ".into());
        }
        offset += n as u64;
    }
    output.write_all_at(&bytes, 0).map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    if read_ple_hot_pack_header(&tmp)? != header {
        return Err("ple rebind: written header differs".into());
    }
    let mut check = HfqFile::open(new_base).map_err(|e| e.to_string())?;
    check.attach_ple_rows(crate::ple_sidecar::PleRowsFile::open(&tmp)?)?;
    fs::rename(&tmp, &out).map_err(|e| e.to_string())?;
    cleanup.disarm();
    sync_dir(parent_dir(&out), "ple rebind")?;
    Ok(out)
}

/// Rebind an existing Q4 table. Unsupported on this platform: the
/// implementation needs Unix positional file I/O and `flock`.
#[cfg(not(unix))]
pub fn rebind_q4(
    _old_base: &Path,
    _new_base: &Path,
    _source: &Path,
    _progress: Progress<'_>,
) -> Result<PathBuf, String> {
    Err(unix_only("ple rebind", "rebind"))
}

/// Derive `<model>.ple-q4` (a v3 `ple-table`) from the model's Q8F16 PLE rows
/// (its `.ple-q8` table when present, else the embedded shards).
///
/// `rank` selects the table order: the given `.plehot` path, else
/// `HIPFIRE_PLE_HOT_RANK_FILE`, else `<model>.plehot` if it exists produce a
/// rank-order table (rows in tier order, rank embedded); with none of them the
/// table is row order. A rank whose fingerprint or padded rows do not match
/// the model, or whose payload hash is wrong, is refused. See the module docs.
#[cfg(unix)]
pub fn derive_q4(
    model_path: &Path,
    rank: Option<&Path>,
    progress: &mut (dyn FnMut(&PleProgress) + Send),
) -> Result<DeriveReport, String> {
    derive_q4_with(model_path, rank, &Tuning::default(), progress)
}

/// Derive `<model>.ple-q4`. Unsupported on this platform: the implementation
/// needs Unix positional file I/O, `flock` and `statvfs`.
#[cfg(not(unix))]
pub fn derive_q4(
    _model_path: &Path,
    _rank: Option<&Path>,
    _progress: &mut (dyn FnMut(&PleProgress) + Send),
) -> Result<DeriveReport, String> {
    Err(unix_only("ple derive", "derive"))
}

#[cfg(unix)]
pub(crate) fn derive_q4_with(
    model_path: &Path,
    rank: Option<&Path>,
    tuning: &Tuning,
    progress: Progress<'_>,
) -> Result<DeriveReport, String> {
    let started = Instant::now();
    let fail = |msg: String| format!("ple derive: {msg}");
    let out_path = ple_sidecar_path(model_path, PleFormat::Q4);
    if exists(&out_path) {
        return Err(fail(format!(
            "{} already exists; delete it first to re-derive",
            out_path.display()
        )));
    }
    let hfq = HfqFile::open(model_path)
        .map_err(|e| fail(format!("open {}: {e}", model_path.display())))?;
    let model_index_sha = hex(&hfq
        .index_sha256()
        .map_err(|e| fail(format!("hash the index of {}: {e}", model_path.display())))?);
    let model_bytes = hfq.file_len();
    let fingerprint = ple_fingerprint_from_metadata(&hfq.metadata_json)
        .map_err(|e| fail(format!("{}: {e}", model_path.display())))?;
    let source = select_rows_source(
        &hfq,
        model_path,
        &model_index_sha,
        model_bytes,
        &fingerprint,
    )
    .map_err(fail)?;
    let layout = source.layout;
    let rank = resolve_rank(model_path, rank, layout.padded_rows(), &fingerprint).map_err(fail)?;
    let q8_row_bytes = layout.q8_row_bytes();
    let q4_row_bytes = q4lm_row_bytes(layout.row_width).ok_or_else(|| {
        fail(format!(
            "row width {} is not a whole number of 32-weight blocks",
            layout.row_width
        ))
    })?;
    let data_bytes = checked_mul(layout.padded_rows(), q4_row_bytes as u64).map_err(fail)?;
    let total_in = layout.shard_count as u64 * layout.q8_shard_bytes();

    let chunk_rows = (tuning.chunk_bytes / q8_row_bytes).max(1);
    let rank_buckets = rank.as_ref().map(|(_, r)| r.buckets());
    let tier_group = if rank.is_some() {
        DEFAULT_TIER_BUCKETS
    } else {
        BUCKETS
    };
    let plan = TierPlan::new(rank_buckets, tier_group, &layout, chunk_rows);
    let rank_sha256 =
        rank_buckets.map_or_else(|| ZERO_SHA256.to_string(), |b| hex(&Sha256::digest(b)));
    let zero = ZERO_SHA256;
    let params = TableParams {
        layout,
        encoding: RowEncoding::Q4Lm,
        row_bytes: q4_row_bytes,
        rank_order: rank.is_some(),
        fingerprint: &fingerprint,
        rank_sha256: &rank_sha256,
        tier_group,
        tier_rows: &plan.tier_rows,
        rank_bytes: rank_buckets.map_or(0, |b| b.len() as u64),
        data_bytes,
        segment_bytes: tuning.segment_bytes,
        source_rows_sha256: zero,
        base_index_sha256: &model_index_sha,
        base_file_bytes: model_bytes,
        generator: "hipfire ple derive",
    };
    let (probe, _) = params.probe().map_err(fail)?;
    let data_offset = probe.data_offset;
    let total_out = checked_add(data_offset, data_bytes).map_err(fail)?;

    let src_file = File::open(&source.path)
        .map_err(|e| fail(format!("open {}: {e}", source.path.display())))?;
    advise_sequential(&src_file);
    let tmp = append_suffix(&out_path, ".tmp");
    let mut cleanup = TmpCleanup::default();
    let out_file = open_locked_tmp(&tmp, "ple derive")?;
    cleanup.add(&tmp);
    check_space(parent_dir(&out_path), total_out, "ple derive")?;
    out_file
        .set_len(total_out)
        .map_err(|e| fail(format!("size {}: {e}", tmp.display())))?;
    if let Some(buckets) = rank_buckets {
        out_file
            .write_all_at(buckets, probe.rank_offset)
            .map_err(|e| fail(format!("write the rank into {}: {e}", tmp.display())))?;
    }

    let rows = layout.rows_per_shard;
    let chunks_per_shard = rows.div_ceil(chunk_rows);
    let tiers = plan.tiers;
    let digests = run_parallel(
        "derive",
        total_in,
        layout.shard_count,
        tuning.threads,
        &mut *progress,
        || DeriveState {
            input: vec![0u8; chunk_rows.min(rows) * q8_row_bytes],
            out: vec![0u8; chunk_rows.min(rows) * q4_row_bytes],
            counts: vec![0usize; tiers],
            starts: vec![0usize; tiers],
            cursor: vec![0usize; tiers],
            behind: WriteBehind::default(),
        },
        |st, s, done| {
            let DeriveState {
                input,
                out,
                counts,
                starts,
                cursor,
                behind,
            } = st;
            let src_base = source.shard_offsets[s];
            let mut hash_in = Sha256::new();
            for k in 0..chunks_per_shard {
                let row0 = k * chunk_rows;
                let n = chunk_rows.min(rows - row0);
                let global0 = s * rows + row0;
                let chunk_index = s * chunks_per_shard + k;
                let in_len = n * q8_row_bytes;
                let src_off = src_base + (row0 * q8_row_bytes) as u64;
                src_file
                    .read_exact_at(&mut input[..in_len], src_off)
                    .map_err(|e| {
                        format!(
                            "ple derive: read {} shard {s} rows {row0}..{}: {e}",
                            source.path.display(),
                            row0 + n
                        )
                    })?;
                drop_cache(&src_file, src_off, in_len as u64);
                hash_in.update(&input[..in_len]);
                let encode = |r: usize, dst: &mut [u8]| {
                    encode_q4lm_row_from_q8(&input[r * q8_row_bytes..(r + 1) * q8_row_bytes], dst)
                        .map_err(|e| format!("ple derive: shard {s} row {}: {e}", row0 + r))
                };
                match rank_buckets {
                    None => {
                        for r in 0..n {
                            encode(r, &mut out[r * q4_row_bytes..(r + 1) * q4_row_bytes])?;
                        }
                        let dst = data_offset + (global0 * q4_row_bytes) as u64;
                        out_file
                            .write_all_at(&out[..n * q4_row_bytes], dst)
                            .map_err(|e| format!("ple derive: write {}: {e}", tmp.display()))?;
                        behind.wrote(&out_file, dst, (n * q4_row_bytes) as u64);
                    }
                    Some(buckets) => {
                        let chunk_buckets = &buckets[global0..global0 + n];
                        counts.fill(0);
                        for &b in chunk_buckets {
                            counts[usize::from(plan.tier_of[usize::from(b)])] += 1;
                        }
                        let mut acc = 0usize;
                        for t in 0..tiers {
                            starts[t] = acc;
                            cursor[t] = acc;
                            acc += counts[t];
                        }
                        for (r, &b) in chunk_buckets.iter().enumerate() {
                            let t = usize::from(plan.tier_of[usize::from(b)]);
                            let pos = cursor[t];
                            cursor[t] += 1;
                            encode(r, &mut out[pos * q4_row_bytes..(pos + 1) * q4_row_bytes])?;
                        }
                        for t in 0..tiers {
                            if counts[t] == 0 {
                                continue;
                            }
                            let run = &out
                                [starts[t] * q4_row_bytes..(starts[t] + counts[t]) * q4_row_bytes];
                            let slot =
                                plan.tier_base_rows[t] + plan.prefix[chunk_index * tiers + t];
                            let dst = data_offset + slot * q4_row_bytes as u64;
                            out_file
                                .write_all_at(run, dst)
                                .map_err(|e| format!("ple derive: write {}: {e}", tmp.display()))?;
                            start_writeback(&out_file, dst, run.len() as u64);
                        }
                    }
                }
                done.fetch_add(in_len as u64, Ordering::Relaxed);
            }
            behind.finish(&out_file);
            Ok(<[u8; 32]>::from(hash_in.finalize()))
        },
    )?;

    let source_rows_sha256 = hex(&merkle_sha256(&digests));
    if let Some(expected) = &source.expected_rows_sha256 {
        if source_rows_sha256 != *expected {
            return Err(fail(format!(
                "{} is corrupt: its rows hash to {source_rows_sha256} but its header records \
                 {expected}; delete it and re-pull or re-split",
                source.path.display()
            )));
        }
    }

    // Flush, drop the cache and re-read the data from disk for the digests.
    out_file
        .sync_all()
        .map_err(|e| fail(format!("sync {}: {e}", tmp.display())))?;
    drop_cache(&out_file, 0, 0);
    let segments = segment_digests(
        &out_file,
        &tmp,
        data_offset,
        data_bytes,
        tuning,
        &mut *progress,
    )
    .map_err(fail)?;
    if let Some(buckets) = rank_buckets {
        let mut written = vec![0u8; buckets.len()];
        out_file
            .read_exact_at(&mut written, probe.rank_offset)
            .map_err(|e| fail(format!("verify read {}: {e}", tmp.display())))?;
        if written != buckets {
            return Err(fail(format!(
                "verification failed: the rank embedded in {} does not match the rank file",
                tmp.display()
            )));
        }
    }

    let params = TableParams {
        source_rows_sha256: &source_rows_sha256,
        ..params
    };
    let (header, header_bytes) = params.finish(&probe, &segments).map_err(fail)?;
    out_file
        .write_all_at(&header_bytes, 0)
        .map_err(|e| fail(format!("write header {}: {e}", tmp.display())))?;
    out_file
        .sync_all()
        .map_err(|e| fail(format!("sync {}: {e}", tmp.display())))?;
    let read_back =
        read_ple_hot_pack_header(&tmp).map_err(|e| fail(format!("verification failed: {e}")))?;
    if read_back != header {
        return Err(fail(format!(
            "verification failed: the header of {} does not read back as written",
            tmp.display()
        )));
    }
    fs::rename(&tmp, &out_path).map_err(|e| {
        fail(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            out_path.display()
        ))
    })?;
    cleanup.disarm();
    sync_dir(parent_dir(&out_path), "ple derive")?;
    let payload_sha256 = header
        .table
        .as_ref()
        .map(|t| t.data_sha256.clone())
        .unwrap_or_default();
    Ok(DeriveReport {
        output: out_path,
        source: source.path,
        bytes: total_out,
        payload_sha256,
        source_rows_sha256,
        elapsed: started.elapsed(),
        threads: tuning.threads.min(layout.shard_count).max(1),
        order: if rank.is_some() { "rank" } else { "row" },
    })
}

#[cfg(unix)]
struct DeriveState {
    input: Vec<u8>,
    /// Encoded rows of one chunk (tier order in rank mode).
    out: Vec<u8>,
    counts: Vec<usize>,
    starts: Vec<usize>,
    cursor: Vec<usize>,
    behind: WriteBehind,
}

/// Resolve and validate the hot rank: the explicit path, else
/// `HIPFIRE_PLE_HOT_RANK_FILE`, else `<model>.plehot` when present.
fn resolve_rank(
    model_path: &Path,
    rank: Option<&Path>,
    padded_rows: u64,
    fingerprint: &str,
) -> Result<Option<(PathBuf, PleHotRank)>, String> {
    let explicit = rank
        .map(Path::to_path_buf)
        .or_else(|| ple_hot_rank_file().map(PathBuf::from));
    let path = match explicit {
        Some(path) => {
            if !exists(&path) {
                return Err(format!(
                    "the PLE hot rank {} does not exist",
                    path.display()
                ));
            }
            path
        }
        None => {
            let default = append_suffix(model_path, ".plehot");
            if !exists(&default) {
                return Ok(None);
            }
            default
        }
    };
    let rank = PleHotRank::open(&path, padded_rows, fingerprint)?;
    rank.verify()?;
    Ok(Some((path, rank)))
}

/// Tier structure of a table: `bucket → tier`, rows per tier, the row index of
/// each tier's first row, and for every (shard, chunk) the number of rows of
/// each tier before the chunk (flattened `[chunk][tier]`). Row order is the
/// degenerate one-tier plan (`tier_group == BUCKETS`, no buckets).
struct TierPlan {
    tier_of: [u8; BUCKETS],
    tiers: usize,
    tier_rows: Vec<u64>,
    tier_base_rows: Vec<u64>,
    prefix: Vec<u64>,
}

impl TierPlan {
    fn new(
        buckets: Option<&[u8]>,
        tier_group: usize,
        layout: &PleShardLayout,
        chunk_rows: usize,
    ) -> Self {
        let tiers = BUCKETS.div_ceil(tier_group);
        let mut tier_of = [0u8; BUCKETS];
        for (bucket, slot) in tier_of.iter_mut().enumerate() {
            *slot = (bucket / tier_group) as u8;
        }
        let chunks_per_shard = layout.rows_per_shard.div_ceil(chunk_rows);
        let mut prefix = Vec::with_capacity(layout.shard_count * chunks_per_shard * tiers);
        let mut counts = vec![0u64; tiers];
        for s in 0..layout.shard_count {
            for k in 0..chunks_per_shard {
                prefix.extend_from_slice(&counts);
                let row0 = k * chunk_rows;
                let n = chunk_rows.min(layout.rows_per_shard - row0);
                match buckets {
                    Some(b) => {
                        let first = s * layout.rows_per_shard + row0;
                        for &bucket in &b[first..first + n] {
                            counts[usize::from(tier_of[usize::from(bucket)])] += 1;
                        }
                    }
                    None => counts[0] += n as u64,
                }
            }
        }
        let mut tier_base_rows = Vec::with_capacity(tiers);
        let mut base = 0u64;
        for &rows in &counts {
            tier_base_rows.push(base);
            base += rows;
        }
        Self {
            tier_of,
            tiers,
            tier_rows: counts,
            tier_base_rows,
            prefix,
        }
    }
}

/// Where the Q8 rows of a derive come from.
struct RowsSource {
    path: PathBuf,
    layout: PleShardLayout,
    /// Absolute file offset of each shard in `path`.
    shard_offsets: Vec<u64>,
    /// `source_rows_sha256` recorded by a `.ple-q8` table, which the digests
    /// recomputed while reading must reproduce.
    expected_rows_sha256: Option<String>,
}

fn select_rows_source(
    hfq: &HfqFile,
    model_path: &Path,
    index_sha: &str,
    file_bytes: u64,
    fingerprint: &str,
) -> Result<RowsSource, String> {
    let q8_path = ple_sidecar_path(model_path, PleFormat::Q8);
    let embedded = find_shards(hfq.tensor_infos())?;
    let marker = hfq.ple_rows_marker();
    if exists(&q8_path) {
        let q8 = read_ple_hot_pack_header(&q8_path)
            .map_err(|e| format!("{}: {e}", q8_path.display()))?;
        let table = match (&q8.kind, &q8.table) {
            (PackKind::PleTable, Some(table)) => table,
            _ => {
                return Err(format!(
                    "{} is not a PLE table; delete it",
                    q8_path.display()
                ))
            }
        };
        if q8.order != PackOrder::Row || q8.row_encoding != RowEncoding::Q8F16 {
            return Err(format!(
                "{} is not a row-order Q8 table (order {:?}, encoding {:?}); delete it",
                q8_path.display(),
                q8.order,
                q8.row_encoding
            ));
        }
        let q8_len = fs::metadata(&q8_path)
            .map_err(|e| format!("stat {}: {e}", q8_path.display()))?
            .len();
        if q8_len != q8.data_offset + q8.data_bytes {
            return Err(format!(
                "{} is {q8_len} bytes but its header describes {}; it is truncated or has \
                 trailing data — delete it and re-pull",
                q8_path.display(),
                q8.data_offset + q8.data_bytes
            ));
        }
        if q8.ple_fingerprint != fingerprint {
            return Err(format!(
                "{} was built for PLE layout {} but {} has {fingerprint}; delete it and \
                 re-pull the matching sidecar",
                q8_path.display(),
                q8.ple_fingerprint,
                model_path.display()
            ));
        }
        if table.base_index_sha256 != index_sha || table.base_file_bytes != file_bytes {
            return Err(format!(
                "{} is bound to a different base file (it records index sha256 {} and {} bytes; \
                 {} has {index_sha} and {file_bytes} bytes); delete it or re-pull the matching \
                 base + sidecar pair",
                q8_path.display(),
                table.base_index_sha256,
                table.base_file_bytes,
                model_path.display()
            ));
        }
        let rows_per_shard = usize::try_from(q8.rows_per_shard)
            .ok()
            .filter(|&r| r > 0 && q8.padded_rows % r as u64 == 0)
            .ok_or_else(|| {
                format!(
                    "{} has rows_per_shard {} that does not divide padded_rows {}",
                    q8_path.display(),
                    q8.rows_per_shard,
                    q8.padded_rows
                )
            })?;
        let layout = PleShardLayout {
            shard_count: (q8.padded_rows / rows_per_shard as u64) as usize,
            rows_per_shard,
            row_width: q8.row_width as usize,
        };
        if q8.encoded_row_bytes != layout.q8_row_bytes() as u64
            || q8.data_bytes != layout.padded_rows() * layout.q8_row_bytes() as u64
        {
            return Err(format!(
                "{} geometry is inconsistent: {} bytes per row, {} data bytes for {} rows of \
                 width {}",
                q8_path.display(),
                q8.encoded_row_bytes,
                q8.data_bytes,
                layout.padded_rows(),
                q8.row_width
            ));
        }
        if let Some(m) = &marker {
            let num = |k: &str| m.get(k).and_then(|v| v.as_u64());
            let ok = m.get("q8_rows_sha256").and_then(|v| v.as_str())
                == Some(table.source_rows_sha256.as_str())
                && num("shard_count") == Some(layout.shard_count as u64)
                && num("rows_per_shard") == Some(layout.rows_per_shard as u64)
                && num("row_width") == Some(layout.row_width as u64);
            if !ok {
                return Err(format!(
                    "{} does not match the `{PLE_BASE_MARKER_KEY}` marker of {}; delete it and \
                     re-pull the matching sidecar",
                    q8_path.display(),
                    model_path.display()
                ));
            }
        }
        if let Some((embedded_layout, _)) = &embedded {
            if *embedded_layout != layout {
                return Err(format!(
                    "{} geometry differs from the PLE shards embedded in {}",
                    q8_path.display(),
                    model_path.display()
                ));
            }
        }
        let shard_bytes = layout.q8_shard_bytes();
        let shard_offsets = (0..layout.shard_count)
            .map(|s| q8.data_offset + s as u64 * shard_bytes)
            .collect();
        return Ok(RowsSource {
            path: q8_path,
            layout,
            shard_offsets,
            expected_rows_sha256: Some(table.source_rows_sha256.clone()),
        });
    }
    if let Some((layout, tensor_index)) = embedded {
        let infos = hfq.tensor_infos();
        let shard_offsets = tensor_index
            .iter()
            .map(|&i| infos[i].data_offset as u64)
            .collect();
        return Ok(RowsSource {
            path: model_path.to_path_buf(),
            layout,
            shard_offsets,
            expected_rows_sha256: None,
        });
    }
    if marker.is_some() {
        Err(format!(
            "{} is a split base and its Q8 rows sidecar {} is missing; fetch it with \
             `hipfire pull <model> --ple q8` (54.4 GB)",
            model_path.display(),
            q8_path.display()
        ))
    } else {
        Err(format!(
            "{} has no PLE shard tensors and no `{PLE_BASE_MARKER_KEY}` marker; \
             it is not a Flash-Next model",
            model_path.display()
        ))
    }
}

// ─── split ───────────────────────────────────────────────────────────────────

/// Split a monolithic Qwen4 HFQM into `<out_base>` (no PLE shards, marker in
/// the metadata) and `<out_base>.ple-q8` (the exact Q8F16 rows as a row-order
/// `ple-table`). `out_base` defaults to [`default_split_base_path`]. The
/// original is not touched.
#[cfg(unix)]
pub fn split(
    model_path: &Path,
    out_base: Option<&Path>,
    progress: &mut (dyn FnMut(&PleProgress) + Send),
) -> Result<SplitReport, String> {
    split_with(model_path, out_base, &Tuning::default(), progress)
}

/// Split a monolithic Qwen4 HFQM. Unsupported on this platform: the
/// implementation needs Unix positional file I/O, `flock` and `statvfs`.
#[cfg(not(unix))]
pub fn split(
    _model_path: &Path,
    _out_base: Option<&Path>,
    _progress: &mut (dyn FnMut(&PleProgress) + Send),
) -> Result<SplitReport, String> {
    Err(unix_only("ple split", "split"))
}

#[cfg(unix)]
pub(crate) fn split_with(
    model_path: &Path,
    out_base: Option<&Path>,
    tuning: &Tuning,
    progress: Progress<'_>,
) -> Result<SplitReport, String> {
    let started = Instant::now();
    let fail = |msg: String| format!("ple split: {msg}");
    let out_base = out_base
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_split_base_path(model_path));
    let out_q8 = ple_sidecar_path(&out_base, PleFormat::Q8);
    if out_base == model_path {
        return Err(fail(format!(
            "the output base {} is the input model",
            out_base.display()
        )));
    }
    for out in [&out_base, &out_q8] {
        if exists(out) {
            return Err(fail(format!(
                "{} already exists; remove it or choose another --out",
                out.display()
            )));
        }
    }

    // Raw header preflight: the overlay flag is hidden by the HfqFile reader.
    let model_file =
        File::open(model_path).map_err(|e| fail(format!("open {}: {e}", model_path.display())))?;
    let mut fixed = [0u8; HFQM_HEADER_BYTES as usize];
    model_file.read_exact_at(&mut fixed, 0).map_err(|e| {
        fail(format!(
            "read the HFQM header of {}: {e}",
            model_path.display()
        ))
    })?;
    if &fixed[0..4] != b"HFQM" {
        return Err(fail(format!(
            "{} is not an HFQM container",
            model_path.display()
        )));
    }
    let raw_version = u32::from_le_bytes(fixed[4..8].try_into().unwrap());
    if raw_version & HFQM_OVERLAY_FLAG != 0 {
        return Err(fail(format!(
            "{} carries a metadata overlay; splitting overlaid containers is not supported",
            model_path.display()
        )));
    }
    if raw_version != HFQM_VERSION {
        return Err(fail(format!(
            "{} has unsupported HFQM version {raw_version}",
            model_path.display()
        )));
    }
    let arch_id = u32::from_le_bytes(fixed[8..12].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
    if metadata_offset != HFQM_HEADER_BYTES {
        return Err(fail(format!(
            "{} has metadata at offset {metadata_offset}, expected {HFQM_HEADER_BYTES}",
            model_path.display()
        )));
    }

    let hfq = HfqFile::open(model_path)
        .map_err(|e| fail(format!("open {}: {e}", model_path.display())))?;
    if hfq.ple_rows_marker().is_some() {
        return Err(fail(format!(
            "{} is already a split base",
            model_path.display()
        )));
    }
    let infos = hfq.tensor_infos();
    let (layout, tensor_index) = find_shards(infos).map_err(&fail)?.ok_or_else(|| {
        fail(format!(
            "{} has no PLE shard tensors to split",
            model_path.display()
        ))
    })?;
    let src_len = hfq.file_len();
    let src_data_offset = hfq.data_offset();
    let last_end = infos
        .last()
        .map_or(src_data_offset, |t| (t.data_offset + t.data_size) as u64);
    if src_len != last_end {
        return Err(fail(format!(
            "{} is {src_len} bytes but its tensors end at {last_end} ({} trailing bytes); \
             refusing to split a container with trailing data",
            model_path.display(),
            src_len.saturating_sub(last_end)
        )));
    }
    let fingerprint = ple_fingerprint_from_metadata(&hfq.metadata_json)
        .map_err(|e| fail(format!("{}: {e}", model_path.display())))?;

    // The original head: header + metadata + index + padding.
    let mut src_head = vec![0u8; src_data_offset as usize];
    model_file
        .read_exact_at(&mut src_head, 0)
        .map_err(|e| fail(format!("read the head of {}: {e}", model_path.display())))?;
    let metadata = hfq.metadata_json.as_str();
    let index_start = HFQM_HEADER_BYTES as usize + metadata.len();
    if src_head.get(HFQM_HEADER_BYTES as usize..index_start) != Some(metadata.as_bytes())
        || !metadata.ends_with('}')
    {
        return Err(fail(format!(
            "{}: metadata JSON is not stored as plain brace-terminated UTF-8",
            model_path.display()
        )));
    }
    let all: Vec<&HfqTensorInfo> = infos.iter().collect();
    let original_index = serialize_index(&all).map_err(&fail)?;
    if src_head.get(index_start..index_start + original_index.len())
        != Some(original_index.as_slice())
    {
        return Err(fail(format!(
            "{}: the tensor index does not round-trip byte for byte",
            model_path.display()
        )));
    }
    match serde_json::from_str::<serde_json::Value>(metadata) {
        Ok(serde_json::Value::Object(map)) => {
            if map.contains_key(PLE_BASE_MARKER_KEY) {
                return Err(fail(format!(
                    "{} already has a `{PLE_BASE_MARKER_KEY}` metadata key",
                    model_path.display()
                )));
            }
        }
        _ => {
            return Err(fail(format!(
                "{}: metadata is not a JSON object",
                model_path.display()
            )))
        }
    }
    drop(src_head);

    let shard_set: std::collections::HashSet<usize> = tensor_index.iter().copied().collect();
    let kept: Vec<&HfqTensorInfo> = infos
        .iter()
        .enumerate()
        .filter(|(i, _)| !shard_set.contains(i))
        .map(|(_, t)| t)
        .collect();
    let plan = BasePlan::new(arch_id, metadata, &layout, &kept).map_err(&fail)?;

    // Q8 table geometry. The header length is digest-independent, so a probe
    // with zero digests (and the final base byte count) fixes data_offset.
    let q8_shard_bytes = layout.q8_shard_bytes();
    let q8_data_bytes = layout.shard_count as u64 * q8_shard_bytes;
    let one_tier = [layout.padded_rows()];
    let zero = ZERO_SHA256;
    let q8_base = TableParams {
        layout,
        encoding: RowEncoding::Q8F16,
        row_bytes: layout.q8_row_bytes(),
        rank_order: false,
        fingerprint: &fingerprint,
        rank_sha256: ZERO_SHA256,
        tier_group: BUCKETS,
        tier_rows: &one_tier,
        rank_bytes: 0,
        data_bytes: q8_data_bytes,
        segment_bytes: tuning.segment_bytes,
        source_rows_sha256: zero,
        base_index_sha256: zero,
        base_file_bytes: plan.file_bytes,
        generator: "hipfire ple split",
    };
    let (probe, _) = q8_base.probe().map_err(&fail)?;
    let q8_data_offset = probe.data_offset;
    let q8_bytes = q8_data_offset + q8_data_bytes;

    let q8_tmp = append_suffix(&out_q8, ".tmp");
    let base_tmp = append_suffix(&out_base, ".tmp");
    let mut cleanup = TmpCleanup::default();
    let q8_file = open_locked_tmp(&q8_tmp, "ple split")?;
    cleanup.add(&q8_tmp);
    let base_file = open_locked_tmp(&base_tmp, "ple split")?;
    cleanup.add(&base_tmp);
    check_space(
        parent_dir(&out_base),
        q8_bytes + plan.file_bytes,
        "ple split",
    )?;
    q8_file
        .set_len(q8_bytes)
        .map_err(|e| fail(format!("size {}: {e}", q8_tmp.display())))?;
    base_file
        .set_len(plan.file_bytes)
        .map_err(|e| fail(format!("size {}: {e}", base_tmp.display())))?;
    advise_sequential(&model_file);

    let chunk = tuning.chunk_bytes.max(1) as u64;
    let segment_bytes = tuning.segment_bytes;
    // Copy/verify pieces of the q8 data: cut at every chunk, shard and segment
    // boundary so a piece never straddles a shard or a segment.
    let pieces = q8_pieces(q8_data_bytes, q8_shard_bytes, chunk, segment_bytes);

    // 1. PLE rows → ple-q8: per-shard source digests (shard-sequential) and a
    //    digest per piece for the verify re-read.
    let copied = run_parallel(
        "copy-ple",
        q8_data_bytes,
        layout.shard_count,
        tuning.threads,
        &mut *progress,
        || CopyState::new(chunk as usize),
        |st, s, done| {
            let src_base = infos[tensor_index[s]].data_offset as u64;
            let shard_start = s as u64 * q8_shard_bytes;
            let lo = pieces.partition_point(|p| p.0 < shard_start);
            let hi = pieces.partition_point(|p| p.0 < shard_start + q8_shard_bytes);
            let mut shard_hash = Sha256::new();
            let mut piece_digests = Vec::with_capacity(hi - lo);
            for &(start, len) in &pieces[lo..hi] {
                let n = len as usize;
                let buf = &mut st.buf[..n];
                let src_off = src_base + (start - shard_start);
                model_file.read_exact_at(buf, src_off).map_err(|e| {
                    format!(
                        "ple split: read {} shard {s} at {src_off}: {e}",
                        model_path.display()
                    )
                })?;
                drop_cache(&model_file, src_off, len);
                shard_hash.update(&*buf);
                piece_digests.push(<[u8; 32]>::from(Sha256::digest(&*buf)));
                let dst = q8_data_offset + start;
                q8_file
                    .write_all_at(buf, dst)
                    .map_err(|e| format!("ple split: write {}: {e}", q8_tmp.display()))?;
                st.behind.wrote(&q8_file, dst, len);
                done.fetch_add(len, Ordering::Relaxed);
            }
            st.behind.finish(&q8_file);
            Ok((<[u8; 32]>::from(shard_hash.finalize()), piece_digests))
        },
    )?;
    let shard_digests: Vec<[u8; 32]> = copied.iter().map(|(d, _)| *d).collect();
    let piece_digests: Vec<[u8; 32]> = copied.iter().flat_map(|(_, p)| p.iter().copied()).collect();
    let q8_rows_sha256 = hex(&merkle_sha256(&shard_digests));

    // 2. The base head (the marker carries the rows digest) and its binding.
    let base_head = plan.head(&q8_rows_sha256);
    let base_index_sha256 = hex(&Sha256::digest(&base_head));
    base_file
        .write_all_at(&base_head, 0)
        .map_err(|e| fail(format!("write head {}: {e}", base_tmp.display())))?;

    // 3. Kept payloads → base, hashing fixed-size chunks of the source.
    let chunks = plan.chunks(chunk);
    let chunk_digests = run_parallel(
        "copy-base",
        plan.kept_bytes,
        chunks.len(),
        tuning.threads,
        &mut *progress,
        || CopyState::new(chunk as usize),
        |st, c, done| {
            let ch = &chunks[c];
            let buf = &mut st.buf[..ch.len as usize];
            model_file.read_exact_at(buf, ch.src).map_err(|e| {
                format!(
                    "ple split: read {} at {}: {e}",
                    model_path.display(),
                    ch.src
                )
            })?;
            drop_cache(&model_file, ch.src, ch.len);
            let digest = <[u8; 32]>::from(Sha256::digest(&*buf));
            base_file
                .write_all_at(buf, ch.dst)
                .map_err(|e| format!("ple split: write {}: {e}", base_tmp.display()))?;
            st.behind.wrote(&base_file, ch.dst, ch.len);
            done.fetch_add(ch.len, Ordering::Relaxed);
            st.behind.finish(&base_file);
            Ok(digest)
        },
    )?;

    // 4. Flush to disk, drop the cache, and verify what the disk now holds.
    //    The q8 re-read also yields the table's segment digests.
    for (file, path) in [(&q8_file, &q8_tmp), (&base_file, &base_tmp)] {
        file.sync_all()
            .map_err(|e| fail(format!("sync {}: {e}", path.display())))?;
        drop_cache(file, 0, 0);
    }
    let segments = table_segment_count(q8_data_bytes, segment_bytes);
    let verified = run_parallel(
        "verify",
        q8_data_bytes + plan.kept_bytes,
        segments + chunks.len(),
        tuning.threads,
        &mut *progress,
        || CopyState::new(chunk as usize),
        |st, i, done| {
            if i < segments {
                let seg_start = i as u64 * segment_bytes;
                let seg_end = (seg_start + segment_bytes).min(q8_data_bytes);
                let lo = pieces.partition_point(|p| p.0 < seg_start);
                let hi = pieces.partition_point(|p| p.0 < seg_end);
                let mut seg_hash = Sha256::new();
                for (idx, &(start, len)) in pieces[lo..hi].iter().enumerate() {
                    let buf = &mut st.buf[..len as usize];
                    q8_file
                        .read_exact_at(buf, q8_data_offset + start)
                        .map_err(|e| format!("ple split: verify read {}: {e}", q8_tmp.display()))?;
                    drop_cache(&q8_file, q8_data_offset + start, len);
                    if <[u8; 32]>::from(Sha256::digest(&*buf)) != piece_digests[lo + idx] {
                        return Err(format!(
                            "ple split: verification failed: {len} bytes at data offset {start} \
                             of {} do not match the source (disk or filesystem error?)",
                            q8_tmp.display()
                        ));
                    }
                    seg_hash.update(&*buf);
                    done.fetch_add(len, Ordering::Relaxed);
                }
                Ok(Some(<[u8; 32]>::from(seg_hash.finalize())))
            } else {
                let c = i - segments;
                let ch = &chunks[c];
                let buf = &mut st.buf[..ch.len as usize];
                base_file
                    .read_exact_at(buf, ch.dst)
                    .map_err(|e| format!("ple split: verify read {}: {e}", base_tmp.display()))?;
                drop_cache(&base_file, ch.dst, ch.len);
                let digest = <[u8; 32]>::from(Sha256::digest(&*buf));
                done.fetch_add(ch.len, Ordering::Relaxed);
                if digest != chunk_digests[c] {
                    return Err(format!(
                        "ple split: verification failed: {} bytes at base offset {} of {} do not \
                         match the source (disk or filesystem error?)",
                        ch.len,
                        ch.dst,
                        base_tmp.display()
                    ));
                }
                Ok(None)
            }
        },
    )?;
    let segment_digests: Vec<[u8; 32]> =
        verified.iter().take(segments).flatten().copied().collect();

    // 5. The q8 header, written last, bound to the base just written.
    let params = TableParams {
        source_rows_sha256: &q8_rows_sha256,
        base_index_sha256: &base_index_sha256,
        ..q8_base
    };
    let (q8_hdr, q8_header_bytes) = params.finish(&probe, &segment_digests).map_err(&fail)?;
    q8_file
        .write_all_at(&q8_header_bytes, 0)
        .map_err(|e| fail(format!("write header {}: {e}", q8_tmp.display())))?;
    q8_file
        .sync_all()
        .map_err(|e| fail(format!("sync {}: {e}", q8_tmp.display())))?;

    // Headers and structure of both outputs, read back from the files.
    let mut written_head = vec![0u8; base_head.len()];
    base_file
        .read_exact_at(&mut written_head, 0)
        .map_err(|e| fail(format!("verify read {}: {e}", base_tmp.display())))?;
    if written_head != base_head {
        return Err(fail(format!(
            "verification failed: the head of {} does not match what was written",
            base_tmp.display()
        )));
    }
    let q8_read =
        read_ple_hot_pack_header(&q8_tmp).map_err(|e| fail(format!("verification failed: {e}")))?;
    if q8_read != q8_hdr {
        return Err(fail(format!(
            "verification failed: the header of {} does not read back as written",
            q8_tmp.display()
        )));
    }
    {
        let base = HfqFile::open(&base_tmp).map_err(|e| {
            fail(format!(
                "verification failed: the base {} does not open: {e}",
                base_tmp.display()
            ))
        })?;
        verify_base(
            &base,
            &kept,
            &layout,
            &q8_rows_sha256,
            &base_index_sha256,
            plan.file_bytes,
        )
        .map_err(|e| fail(format!("verification failed: {} {e}", base_tmp.display())))?;
    }

    // 6. Publish: q8 first, then the base (the base is what names the model).
    fs::rename(&q8_tmp, &out_q8).map_err(|e| {
        fail(format!(
            "rename {} -> {}: {e}",
            q8_tmp.display(),
            out_q8.display()
        ))
    })?;
    if let Err(e) = fs::rename(&base_tmp, &out_base) {
        let _ = fs::remove_file(&out_q8);
        return Err(fail(format!(
            "rename {} -> {}: {e}",
            base_tmp.display(),
            out_base.display()
        )));
    }
    cleanup.disarm();
    sync_dir(parent_dir(&out_base), "ple split")?;
    Ok(SplitReport {
        base: out_base,
        base_bytes: plan.file_bytes,
        base_index_sha256,
        ple_q8: out_q8,
        ple_q8_bytes: q8_bytes,
        q8_rows_sha256,
        elapsed: started.elapsed(),
    })
}

/// `[start, len)` pieces of `[0, total)` cut at every multiple of `chunk`,
/// `shard_bytes` and `segment_bytes`.
fn q8_pieces(total: u64, shard_bytes: u64, chunk: u64, segment_bytes: u64) -> Vec<(u64, u64)> {
    let cuts = [chunk, shard_bytes, segment_bytes];
    let mut out = Vec::new();
    let mut at = 0u64;
    while at < total {
        let end = cuts
            .iter()
            .map(|&g| (at / g + 1) * g)
            .min()
            .unwrap_or(total)
            .min(total);
        out.push((at, end - at));
        at = end;
    }
    out
}

/// One contiguous copy unit of the kept payloads.
struct Chunk {
    src: u64,
    dst: u64,
    len: u64,
}

/// The base container layout: head bytes (header + metadata + index + pad),
/// its data offset and the kept payload run list.
struct BasePlan<'a> {
    arch_id: u32,
    metadata: &'a str,
    marker_fields: String,
    index: Vec<u8>,
    n_kept: usize,
    data_offset: u64,
    kept_bytes: u64,
    file_bytes: u64,
    /// (source offset, destination offset, length) of maximal contiguous runs.
    runs: Vec<(u64, u64, u64)>,
}

impl<'a> BasePlan<'a> {
    fn new(
        arch_id: u32,
        metadata: &'a str,
        layout: &PleShardLayout,
        kept: &[&HfqTensorInfo],
    ) -> Result<Self, String> {
        let index = serialize_index(kept)?;
        let marker_fields = format!(
            "\"version\":1,\"location\":\"sidecar\",\"shard_count\":{},\"rows_per_shard\":{},\
             \"row_width\":{},\"q8_rows_sha256\":\"",
            layout.shard_count, layout.rows_per_shard, layout.row_width
        );
        // Placeholder digest: the head length does not depend on digest values.
        let meta_len = insert_marker(metadata, &marker_fields, ZERO_SHA256).len() as u64;
        let index_end = HFQM_HEADER_BYTES + meta_len + index.len() as u64;
        let first_kept = kept.first().map(|t| t.data_offset as u64);
        let data_offset = base_data_offset(index_end, first_kept);
        let mut runs: Vec<(u64, u64, u64)> = Vec::new();
        let mut kept_bytes = 0u64;
        for t in kept {
            let (src, len) = (t.data_offset as u64, t.data_size as u64);
            if len == 0 {
                continue;
            }
            match runs.last_mut() {
                Some(run) if run.0 + run.2 == src => run.2 += len,
                _ => runs.push((src, data_offset + kept_bytes, len)),
            }
            kept_bytes += len;
        }
        Ok(Self {
            arch_id,
            metadata,
            marker_fields,
            index,
            n_kept: kept.len(),
            data_offset,
            kept_bytes,
            file_bytes: data_offset + kept_bytes,
            runs,
        })
    }

    /// `[0, data_offset)` of the base for the given q8 rows digest.
    fn head(&self, q8_rows_sha256: &str) -> Vec<u8> {
        let meta = insert_marker(self.metadata, &self.marker_fields, q8_rows_sha256);
        let mut head = Vec::with_capacity(self.data_offset as usize);
        head.extend_from_slice(b"HFQM");
        head.extend_from_slice(&HFQM_VERSION.to_le_bytes());
        head.extend_from_slice(&self.arch_id.to_le_bytes());
        head.extend_from_slice(&(self.n_kept as u32).to_le_bytes());
        head.extend_from_slice(&HFQM_HEADER_BYTES.to_le_bytes());
        head.extend_from_slice(&self.data_offset.to_le_bytes());
        head.extend_from_slice(meta.as_bytes());
        head.extend_from_slice(&self.index);
        head.resize(self.data_offset as usize, 0);
        head
    }

    /// The kept payload runs cut into `chunk`-byte copy units.
    fn chunks(&self, chunk: u64) -> Vec<Chunk> {
        let mut out = Vec::new();
        for &(src, dst, len) in &self.runs {
            let mut off = 0u64;
            while off < len {
                let n = (len - off).min(chunk);
                out.push(Chunk {
                    src: src + off,
                    dst: dst + off,
                    len: n,
                });
                off += n;
            }
        }
        out
    }
}

/// Smallest offset ≥ `index_end` congruent to the first kept payload's source
/// offset modulo 4096 (so every kept payload keeps its alignment).
fn base_data_offset(index_end: u64, first_kept: Option<u64>) -> u64 {
    let residue = first_kept.map_or(0, |o| o % DATA_ALIGN);
    index_end + (residue + DATA_ALIGN - index_end % DATA_ALIGN) % DATA_ALIGN
}

/// `metadata` with `,"hipfire_ple_rows":{<fields><sha>"}` inserted before its
/// final `}` (no comma when the object is empty).
fn insert_marker(metadata: &str, marker_fields: &str, sha: &str) -> String {
    let body = &metadata[..metadata.len() - 1];
    let comma = if body.trim_end().ends_with('{') {
        ""
    } else {
        ","
    };
    format!("{body}{comma}\"{PLE_BASE_MARKER_KEY}\":{{{marker_fields}{sha}\"}}}}")
}

/// Index bytes (count + entries) for `entries`, exactly as the HFQM writer.
fn serialize_index(entries: &[&HfqTensorInfo]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for t in entries {
        let name = t.name.as_bytes();
        let name_len =
            u16::try_from(name.len()).map_err(|_| format!("tensor name too long: {}", t.name))?;
        let n_dims = u8::try_from(t.shape.len())
            .map_err(|_| format!("tensor {} has too many dims", t.name))?;
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(name);
        out.push(t.quant_type);
        out.push(n_dims);
        for d in &t.shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.group_size.to_le_bytes());
        out.extend_from_slice(&(t.data_size as u64).to_le_bytes());
    }
    Ok(out)
}

/// Structural comparison of the written base against the source's kept tensors.
fn verify_base(
    base: &HfqFile,
    kept: &[&HfqTensorInfo],
    layout: &PleShardLayout,
    q8_rows_sha256: &str,
    base_index_sha256: &str,
    file_bytes: u64,
) -> Result<(), String> {
    let got = base.tensor_infos();
    if got.len() != kept.len() {
        return Err(format!(
            "has {} tensors, expected {}",
            got.len(),
            kept.len()
        ));
    }
    for (index, (g, k)) in got.iter().zip(kept).enumerate() {
        if g.name != k.name
            || g.quant_type != k.quant_type
            || g.shape != k.shape
            || g.group_size != k.group_size
            || g.data_size != k.data_size
        {
            return Err(format!("tensor {} differs from the source", k.name));
        }
        // `base_data_offset` keeps the first kept payload's 4 KiB residue, so
        // every kept payload of the same contiguous run keeps its alignment
        // (all of them when the shards are one prefix or suffix of the data,
        // as in the Flash-Next artifact); payloads past a removed run that is
        // not a whole number of pages cannot, whatever the data offset.
        if index == 0 && g.data_offset as u64 % DATA_ALIGN != k.data_offset as u64 % DATA_ALIGN {
            return Err(format!(
                "tensor {} lost its 4 KiB alignment (offset {} vs source {})",
                k.name, g.data_offset, k.data_offset
            ));
        }
    }
    let expected = serde_json::json!({
        "version": 1,
        "location": "sidecar",
        "shard_count": layout.shard_count,
        "rows_per_shard": layout.rows_per_shard,
        "row_width": layout.row_width,
        "q8_rows_sha256": q8_rows_sha256,
    });
    if base.ple_rows_marker().as_ref() != Some(&expected) {
        return Err(format!(
            "does not carry the expected `{PLE_BASE_MARKER_KEY}` marker"
        ));
    }
    let sha = base
        .index_sha256()
        .map_err(|e| format!("index hash: {e}"))?;
    if hex(&sha) != base_index_sha256 {
        return Err("index hash differs from the one bound into the sidecar".to_string());
    }
    if base.file_len() != file_bytes {
        return Err(format!(
            "is {} bytes, expected {file_bytes}",
            base.file_len()
        ));
    }
    Ok(())
}

// ─── PLE shard discovery ─────────────────────────────────────────────────────

/// Canonical PLE row geometry (Q8F16 source rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PleShardLayout {
    shard_count: usize,
    rows_per_shard: usize,
    row_width: usize,
}

impl PleShardLayout {
    fn q8_row_bytes(&self) -> usize {
        self.row_width / Q8_BLOCK_WEIGHTS * Q8_BLOCK_BYTES
    }

    fn q8_shard_bytes(&self) -> u64 {
        self.rows_per_shard as u64 * self.q8_row_bytes() as u64
    }

    fn padded_rows(&self) -> u64 {
        self.shard_count as u64 * self.rows_per_shard as u64
    }
}

/// The `shard_{0..n}` PLE tensors of an index: their geometry and the index
/// position of shard `s`. `None` when no tensor has a shard name.
fn find_shards(infos: &[HfqTensorInfo]) -> Result<Option<(PleShardLayout, Vec<usize>)>, String> {
    let mut found: Vec<(usize, usize)> = Vec::new();
    for (i, t) in infos.iter().enumerate() {
        let Some(num) = t
            .name
            .strip_prefix(PLE_SHARD_PREFIX)
            .and_then(|rest| rest.strip_suffix(PLE_SHARD_SUFFIX))
        else {
            continue;
        };
        if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let s: usize = num
            .parse()
            .map_err(|_| format!("PLE shard tensor {} has an unparsable shard number", t.name))?;
        if ple_shard_name(s) != t.name {
            return Err(format!(
                "PLE shard tensor {} has a non-canonical name",
                t.name
            ));
        }
        found.push((s, i));
    }
    if found.is_empty() {
        return Ok(None);
    }
    found.sort_unstable();
    for (pos, &(s, _)) in found.iter().enumerate() {
        if s != pos {
            return Err(format!(
                "PLE shard tensors are not exactly shard_0..shard_{}: shard {pos} is missing",
                found.len() - 1
            ));
        }
    }
    let tensor_index: Vec<usize> = found.iter().map(|&(_, i)| i).collect();
    let first = &infos[tensor_index[0]];
    let (rows, width) = match first.shape.as_slice() {
        &[r, w] if r > 0 && w > 0 => (r as usize, w as usize),
        other => {
            return Err(format!(
                "PLE shard tensor {} has shape {other:?}, expected [rows, width]",
                first.name
            ))
        }
    };
    if width % Q8_BLOCK_WEIGHTS != 0 {
        return Err(format!(
            "PLE row width {width} is not a whole number of {Q8_BLOCK_WEIGHTS}-weight Q8 blocks"
        ));
    }
    let layout = PleShardLayout {
        shard_count: found.len(),
        rows_per_shard: rows,
        row_width: width,
    };
    let expect_bytes = layout.q8_shard_bytes();
    for &i in &tensor_index {
        let t = &infos[i];
        if t.quant_type != Q8_QUANT_TYPE
            || t.group_size != Q8_GROUP
            || t.shape != first.shape
            || t.data_size as u64 != expect_bytes
        {
            return Err(format!(
                "PLE shard tensor {} is not a Q8F16 [{rows}, {width}] tensor of {expect_bytes} \
                 bytes (qt {}, shape {:?}, group {}, {} bytes)",
                t.name, t.quant_type, t.shape, t.group_size, t.data_size
            ));
        }
    }
    Ok(Some((layout, tensor_index)))
}

// ─── parallel runner ─────────────────────────────────────────────────────────

/// Decrements the live-worker count even when a worker panics, so the polling
/// thread always terminates (the scope then re-raises the panic).
struct ActiveGuard<'a>(&'a AtomicUsize);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// Run `tasks` units of work over up to `threads` workers pulling task indices
/// from an atomic counter. Each worker builds one `init()` state. Progress is
/// reported from the calling thread only: at phase start, roughly every 0.5% of
/// `total_bytes` (workers add to the shared `done` counter) and at completion.
/// Results come back in task order; the first error aborts the rest.
fn run_parallel<S, T, I, W>(
    phase: &'static str,
    total_bytes: u64,
    tasks: usize,
    threads: usize,
    progress: Progress<'_>,
    init: I,
    work: W,
) -> Result<Vec<T>, String>
where
    T: Send + Sync,
    I: Fn() -> S + Sync,
    W: Fn(&mut S, usize, &AtomicU64) -> Result<T, String> + Sync,
{
    let workers = threads.min(tasks).max(1);
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(0);
    let active = AtomicUsize::new(workers);
    let abort = AtomicBool::new(false);
    let failure: OnceLock<String> = OnceLock::new();
    let results: Vec<OnceLock<T>> = (0..tasks).map(|_| OnceLock::new()).collect();
    progress(&PleProgress {
        phase,
        done_bytes: 0,
        total_bytes,
    });
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let _guard = ActiveGuard(&active);
                let mut state = init();
                while !abort.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= tasks {
                        break;
                    }
                    match work(&mut state, i, &done) {
                        Ok(value) => {
                            let _ = results[i].set(value);
                        }
                        Err(e) => {
                            abort.store(true, Ordering::Relaxed);
                            // The first error wins; later ones are dropped.
                            let _ = failure.set(e);
                            break;
                        }
                    }
                }
            });
        }
        let step = (total_bytes / 200).max(1);
        let mut last = 0u64;
        while active.load(Ordering::Acquire) != 0 {
            std::thread::sleep(Duration::from_millis(20));
            let now = done.load(Ordering::Relaxed).min(total_bytes);
            if now >= last + step {
                last = now;
                progress(&PleProgress {
                    phase,
                    done_bytes: now,
                    total_bytes,
                });
            }
        }
    });
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    progress(&PleProgress {
        phase,
        done_bytes: total_bytes,
        total_bytes,
    });
    results
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .ok_or_else(|| format!("{phase}: a task produced no result"))
        })
        .collect()
}

#[cfg(unix)]
/// Per-worker copy buffer and write-behind state.
struct CopyState {
    buf: Vec<u8>,
    behind: WriteBehind,
}

#[cfg(unix)]
impl CopyState {
    fn new(chunk_bytes: usize) -> Self {
        Self {
            buf: vec![0u8; chunk_bytes],
            behind: WriteBehind::default(),
        }
    }
}

// ─── file helpers ────────────────────────────────────────────────────────────

/// Removes the listed tmp files on drop unless disarmed. A path is only added
/// once its lock is held, so a competing converter's tmp is never deleted.
#[derive(Default)]
struct TmpCleanup {
    paths: Vec<PathBuf>,
}

impl TmpCleanup {
    fn add(&mut self, path: &Path) {
        self.paths.push(path.to_path_buf());
    }

    fn disarm(&mut self) {
        self.paths.clear();
    }
}

impl Drop for TmpCleanup {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = fs::remove_file(path);
        }
    }
}

/// Create/open `path`, take an exclusive non-blocking `flock` and truncate any
/// stale content. The lock lives as long as the returned handle.
#[cfg(unix)]
fn open_locked_tmp(path: &Path, ctx: &str) -> Result<File, String> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("{ctx}: open {}: {e}", path.display()))?;
    // SAFETY: the fd is valid for the lifetime of `file`.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let e = io::Error::last_os_error();
        return Err(if e.kind() == io::ErrorKind::WouldBlock {
            format!(
                "{ctx}: {} is locked by another process (a concurrent `hipfire ple` conversion \
                 writing the same output?); wait for it to finish",
                path.display()
            )
        } else {
            format!("{ctx}: lock {}: {e}", path.display())
        });
    }
    file.set_len(0)
        .map_err(|e| format!("{ctx}: truncate {}: {e}", path.display()))?;
    Ok(file)
}

/// Free bytes available to unprivileged writers on the filesystem of `dir`.
#[cfg(unix)]
pub fn available_bytes(dir: &Path) -> Result<u64, String> {
    use std::os::unix::ffi::OsStrExt;
    let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes())
        .map_err(|_| format!("path {} contains a NUL byte", dir.display()))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: c_dir is NUL-terminated and stats is writable statvfs storage.
    if unsafe { libc::statvfs(c_dir.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(format!(
            "statvfs {}: {}",
            dir.display(),
            io::Error::last_os_error()
        ));
    }
    // SAFETY: statvfs initialised `stats` on success.
    let stats = unsafe { stats.assume_init() };
    #[allow(clippy::unnecessary_cast)]
    Ok((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
}

/// Refuse when `dir` cannot hold `bytes` plus the safety margin.
#[cfg(unix)]
fn check_space(dir: &Path, bytes: u64, ctx: &str) -> Result<(), String> {
    let available = available_bytes(dir).map_err(|e| format!("{ctx}: {e}"))?;
    let need = bytes.saturating_add(SPACE_MARGIN);
    if available < need {
        return Err(format!(
            "{ctx}: not enough free space in {}: need {:.1} GB ({:.1} GB of output + 1 GiB \
             margin), {:.1} GB available",
            dir.display(),
            need as f64 / 1e9,
            bytes as f64 / 1e9,
            available as f64 / 1e9
        ));
    }
    Ok(())
}

fn sync_dir(dir: &Path, ctx: &str) -> Result<(), String> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| format!("{ctx}: fsync directory {}: {e}", dir.display()))
}

fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn exists(path: &Path) -> bool {
    path.symlink_metadata().is_ok()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn checked_mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("size {a} x {b} overflows u64"))
}

fn checked_add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("size {a} + {b} overflows u64"))
}

#[cfg(unix)]
fn advise_sequential(file: &File) {
    // SAFETY: advisory call on a valid fd.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
    }
}

/// Drop the clean cached pages of `[off, off + len)` (`len == 0`: to EOF).
#[cfg(unix)]
fn drop_cache(file: &File, off: u64, len: u64) {
    // SAFETY: advisory call on a valid fd.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            off as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

/// Keeps a writer's dirty page-cache footprint to about one chunk: each new
/// range starts asynchronous writeback; the previous range is then waited on
/// and dropped from the cache.
#[cfg(unix)]
#[derive(Default)]
struct WriteBehind {
    prev: Option<(u64, u64)>,
}

#[cfg(unix)]
impl WriteBehind {
    fn wrote(&mut self, file: &File, off: u64, len: u64) {
        start_writeback(file, off, len);
        if let Some((prev_off, prev_len)) = self.prev.replace((off, len)) {
            settle(file, prev_off, prev_len);
        }
    }

    fn finish(&mut self, file: &File) {
        if let Some((off, len)) = self.prev.take() {
            settle(file, off, len);
        }
    }
}

#[cfg(target_os = "linux")]
fn start_writeback(file: &File, off: u64, len: u64) {
    // SAFETY: advisory call on a valid fd.
    unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            off as libc::off64_t,
            len as libc::off64_t,
            libc::SYNC_FILE_RANGE_WRITE,
        );
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn start_writeback(_file: &File, _off: u64, _len: u64) {}

#[cfg(target_os = "linux")]
fn settle(file: &File, off: u64, len: u64) {
    // SAFETY: advisory call on a valid fd.
    unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            off as libc::off64_t,
            len as libc::off64_t,
            libc::SYNC_FILE_RANGE_WAIT_BEFORE
                | libc::SYNC_FILE_RANGE_WRITE
                | libc::SYNC_FILE_RANGE_WAIT_AFTER,
        );
    }
    drop_cache(file, off, len);
}

#[cfg(all(unix, not(target_os = "linux")))]
fn settle(file: &File, off: u64, len: u64) {
    drop_cache(file, off, len);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::hfq::{write_hfqm_package_mem, HfqMemTensor};
    #[cfg(unix)]
    use crate::ple_hot_store::write_ple_hot_rank;
    #[cfg(unix)]
    use crate::ple_q4::decode_q4lm_row_bf16;

    const SHARDS: usize = 4;
    const ROWS: usize = 8;
    const WIDTH: usize = 64;
    const PADDED: usize = SHARDS * ROWS;
    const Q8_ROW: usize = WIDTH / 32 * 34;
    #[cfg(unix)]
    const Q4_ROW: usize = WIDTH / 32 * 18;
    #[cfg(unix)]
    /// 16 heads of 2 rows each = 32 valid rows = `padded_rows`.
    const METADATA: &str = r#"{"architecture":"qwen4","note":"ple convert test","qwen4_ple":{"version":1,"multipliers":[-3,5,7],"head_vocab_sizes":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],"head_offsets":[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30],"padded_rows":32}}"#;

    #[cfg(unix)]
    /// Small tuning that forces many chunks, several segments and workers.
    fn tuning() -> Tuning {
        Tuning {
            threads: 3,
            chunk_bytes: 200,
            segment_bytes: 1000,
        }
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[cfg(unix)]
    /// Q8F16 rows with zero / negative / subnormal / large scales and ±127
    /// extremes mixed in (never NaN/inf, never overflowing an f16 scale).
    fn q8_shard(seed: u64) -> Vec<u8> {
        const SCALES: [u16; 8] = [
            0x0000, 0x3c00, 0xb400, 0x0001, 0x2e66, 0x5640, 0x0400, 0xbc00,
        ];
        let mut state = seed;
        let mut out = Vec::new();
        for _ in 0..ROWS * WIDTH / 32 {
            let scale = SCALES[lcg(&mut state) as usize % SCALES.len()];
            out.extend_from_slice(&scale.to_le_bytes());
            for _ in 0..32 {
                let r = lcg(&mut state);
                let v: i8 = match r % 16 {
                    0 => 127,
                    1 => -127,
                    2 => 0,
                    _ => (r >> 8) as i8,
                };
                out.push(v as u8);
            }
        }
        out
    }

    #[cfg(unix)]
    fn payload(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len).map(|_| (lcg(&mut state) >> 5) as u8).collect()
    }

    #[cfg(unix)]
    struct Fixture {
        dir: tempfile::TempDir,
        model: PathBuf,
        shards: Vec<Vec<u8>>,
        kept: Vec<(String, Vec<u8>)>,
    }

    #[cfg(unix)]
    fn mem(name: &str, qt: u8, shape: Vec<u32>, group: u32, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: qt,
            shape,
            group_size: group,
            data,
        }
    }

    #[cfg(unix)]
    /// `shards_first`: shards at index 0..4 (the real layout); otherwise one
    /// kept tensor precedes them so kept payloads form two source runs.
    fn fixture(shards_first: bool) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("flash.mq4-xts");
        let shards: Vec<Vec<u8>> = (0..SHARDS).map(|s| q8_shard(100 + s as u64)).collect();
        let kept = vec![
            ("model.embed_tokens.weight".to_string(), payload(1, 5000)),
            (
                "model.layers.0.norm.weight".to_string(),
                payload(2, 4096 * 2 + 100),
            ),
            ("lm_head.weight".to_string(), payload(3, 77)),
        ];
        let mut tensors = Vec::new();
        let kept_tensor = |i: usize| {
            mem(
                &kept[i].0,
                1,
                vec![kept[i].1.len() as u32],
                0,
                kept[i].1.clone(),
            )
        };
        if !shards_first {
            tensors.push(kept_tensor(0));
        }
        for (s, data) in shards.iter().enumerate() {
            tensors.push(mem(
                &ple_shard_name(s),
                3,
                vec![ROWS as u32, WIDTH as u32],
                32,
                data.clone(),
            ));
        }
        if shards_first {
            tensors.push(kept_tensor(0));
        }
        tensors.push(kept_tensor(1));
        tensors.push(kept_tensor(2));
        write_hfqm_package_mem(&model, 16, METADATA, &tensors).unwrap();
        Fixture {
            dir,
            model,
            shards,
            kept,
        }
    }

    #[cfg(unix)]
    fn no_progress() -> impl FnMut(&PleProgress) + Send {
        |_: &PleProgress| {}
    }

    #[cfg(unix)]
    fn read_all(path: &Path) -> Vec<u8> {
        fs::read(path).unwrap()
    }

    #[cfg(unix)]
    fn q4_path(model: &Path) -> PathBuf {
        ple_sidecar_path(model, PleFormat::Q4)
    }

    #[cfg(unix)]
    fn q8_path(model: &Path) -> PathBuf {
        ple_sidecar_path(model, PleFormat::Q8)
    }

    #[cfg(unix)]
    fn fingerprint() -> String {
        ple_fingerprint_from_metadata(METADATA).unwrap()
    }

    #[cfg(unix)]
    fn derive(model: &Path, rank: Option<&Path>, tuning: &Tuning) -> Result<DeriveReport, String> {
        derive_q4_with(model, rank, tuning, &mut no_progress())
    }

    #[cfg(unix)]
    #[test]
    fn rebind_breaks_hardlink_and_preserves_payload() {
        use std::os::unix::fs::MetadataExt;
        let fx = fixture(true);
        derive(&fx.model, None, &tuning()).unwrap();
        let source = q4_path(&fx.model);
        let stock = read_all(&source);
        let base = fx.dir.path().join("changed.mq4");
        fs::copy(&fx.model, &base).unwrap();
        crate::hfq::append_hfq_metadata_overlay(&base, METADATA).unwrap();
        let out = q4_path(&base);
        fs::hard_link(&source, &out).unwrap();
        rebind_q4(&fx.model, &base, &source, &mut no_progress()).unwrap();
        assert_ne!(
            fs::metadata(&source).unwrap().ino(),
            fs::metadata(&out).unwrap().ino()
        );
        assert_eq!(read_all(&source), stock);
        let a = read_ple_hot_pack_header(&source).unwrap();
        let mut b = read_ple_hot_pack_header(&out).unwrap();
        assert_eq!(
            &stock[a.data_offset as usize..],
            &read_all(&out)[b.data_offset as usize..]
        );
        assert_eq!(
            b.table.as_ref().unwrap().base_file_bytes,
            fs::metadata(&base).unwrap().len()
        );
        b.table.as_mut().unwrap().base_file_bytes = a.table.as_ref().unwrap().base_file_bytes;
        b.table.as_mut().unwrap().base_index_sha256 =
            a.table.as_ref().unwrap().base_index_sha256.clone();
        assert_eq!(a, b);
    }

    #[cfg(unix)]
    #[test]
    fn rebind_refuses_changed_source_rows_and_corrupt_q4() {
        let fx = fixture(true);
        derive(&fx.model, None, &tuning()).unwrap();
        let source = q4_path(&fx.model);
        let base = fx.dir.path().join("changed.mq4");
        fs::copy(&fx.model, &base).unwrap();
        let hfq = HfqFile::open(&base).unwrap();
        let shard = hfq
            .tensor_infos()
            .iter()
            .find(|t| t.name == ple_shard_name(0))
            .unwrap();
        let file = OpenOptions::new().write(true).open(&base).unwrap();
        file.write_all_at(&[255], shard.data_offset as u64).unwrap();
        assert!(rebind_q4(&fx.model, &base, &source, &mut no_progress())
            .unwrap_err()
            .contains("source rows changed"));
        fs::copy(&fx.model, &base).unwrap();
        let header = read_ple_hot_pack_header(&source).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&source)
            .unwrap()
            .write_all_at(&[255], header.data_offset)
            .unwrap();
        assert!(rebind_q4(&fx.model, &base, &source, &mut no_progress())
            .unwrap_err()
            .contains("payload digest mismatch"));
        assert!(!q4_path(&base).exists());
    }

    #[cfg(unix)]
    fn do_split(model: &Path, out: Option<&Path>, tuning: &Tuning) -> Result<SplitReport, String> {
        split_with(model, out, tuning, &mut no_progress())
    }

    fn random_buckets(seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..PADDED).map(|_| (lcg(&mut state) >> 3) as u8).collect()
    }

    #[cfg(unix)]
    fn write_rank(path: &Path, buckets: &[u8]) {
        write_ple_hot_rank(
            path,
            buckets.len() as u64,
            &fingerprint(),
            buckets,
            serde_json::json!({}),
        )
        .unwrap();
    }

    #[cfg(unix)]
    /// The Q4 row of global `row` of a derived table, found through the layout
    /// rules (row order: contiguous; rank order: tier base + slot).
    fn table_row(bytes: &[u8], h: &PleHotPackHeader, row: usize, buckets: &[u8]) -> Vec<u8> {
        let at = match h.order {
            PackOrder::Row => row,
            PackOrder::Rank => {
                let g = h.tier_group;
                let tier = usize::from(buckets[row]) / g;
                let base: u64 = h.tier_rows[..tier].iter().sum();
                let slot = (0..row)
                    .filter(|&r| usize::from(buckets[r]) / g == tier)
                    .count() as u64;
                (base + slot) as usize
            }
        };
        let off = h.data_offset as usize + at * Q4_ROW;
        bytes[off..off + Q4_ROW].to_vec()
    }

    #[cfg(unix)]
    fn read_table(path: &Path) -> PleHotPackHeader {
        read_ple_hot_pack_header(path).unwrap()
    }

    #[cfg(unix)]
    /// Recompute the segment digests / data digest of `bytes` and compare.
    fn assert_digests(bytes: &[u8], h: &PleHotPackHeader) {
        let table = h.table.as_ref().unwrap();
        assert_eq!(bytes.len() as u64, h.data_offset + h.data_bytes);
        let data = &bytes[h.data_offset as usize..];
        let seg = table.segment_bytes as usize;
        let digests: Vec<[u8; 32]> = data.chunks(seg).map(|c| Sha256::digest(c).into()).collect();
        assert_eq!(
            table.segment_sha256,
            digests.iter().map(|d| hex(d)).collect::<Vec<_>>()
        );
        assert_eq!(table.data_sha256, hex(&merkle_sha256(&digests)));
    }

    #[test]
    fn default_split_base_path_inserts_before_extension() {
        assert_eq!(
            default_split_base_path(Path::new("/m/qwen3.8-flash-next.mq4-xts")),
            Path::new("/m/qwen3.8-flash-next-base.mq4-xts")
        );
        assert_eq!(
            default_split_base_path(Path::new("model")),
            Path::new("model-base")
        );
    }

    #[cfg(unix)]
    #[test]
    fn derive_row_order_matches_row_encoding() {
        for shards_first in [true, false] {
            let fx = fixture(shards_first);
            let mut seen: Vec<PleProgress> = Vec::new();
            let mut cb = |p: &PleProgress| seen.push(*p);
            let report = derive_q4_with(&fx.model, None, &tuning(), &mut cb).unwrap();
            assert_eq!(report.output, q4_path(&fx.model));
            assert_eq!(report.source, fx.model);
            assert_eq!(report.threads, 3);
            assert_eq!(report.order, "row");
            let phases: Vec<&str> = seen.iter().map(|p| p.phase).collect();
            assert!(phases.starts_with(&["derive"]) && phases.ends_with(&["verify"]));
            let derive_last = seen.iter().rfind(|p| p.phase == "derive").unwrap();
            assert_eq!(derive_last.done_bytes, derive_last.total_bytes);
            assert_eq!(derive_last.total_bytes, (SHARDS * ROWS * Q8_ROW) as u64);

            let h = read_table(&report.output);
            assert_eq!(h.version, 3);
            assert_eq!(h.kind, PackKind::PleTable);
            assert_eq!(h.order, PackOrder::Row);
            assert_eq!(h.generator, "hipfire ple derive");
            assert_eq!(h.row_encoding, RowEncoding::Q4Lm);
            assert_eq!(h.row_width, WIDTH as u64);
            assert_eq!(h.encoded_row_bytes, Q4_ROW as u64);
            assert_eq!(h.rows_per_shard, ROWS as u64);
            assert_eq!(h.padded_rows, PADDED as u64);
            assert_eq!(h.ple_fingerprint, fingerprint());
            assert_eq!(h.kept_buckets, 256);
            assert_eq!(h.tier_group, 256);
            assert_eq!(h.tier_rows, vec![PADDED as u64]);
            assert_eq!((h.rank_offset, h.rank_bytes), (0, 0));
            assert!(h.shard_offsets.is_empty() && h.model_file_bytes == 0);
            let t = h.table.as_ref().unwrap();
            assert_eq!(t.codebook_f32_bits, Some(lm4_codebook_bits()));
            assert_eq!(t.scale_grid_f32_bits, Some(lm4_scale_grid_bits()));
            assert_eq!(t.source_encoding, RowEncoding::Q8F16);
            assert_eq!(t.segment_bytes, 1000);
            let hfq = HfqFile::open(&fx.model).unwrap();
            assert_eq!(t.base_index_sha256, hex(&hfq.index_sha256().unwrap()));
            assert_eq!(t.base_file_bytes, fs::metadata(&fx.model).unwrap().len());

            let bytes = read_all(&report.output);
            assert_eq!(report.bytes, bytes.len() as u64);
            assert_digests(&bytes, &h);
            assert_eq!(report.payload_sha256, t.data_sha256);
            let mut src_digests = Vec::new();
            for (s, shard) in fx.shards.iter().enumerate() {
                for r in 0..ROWS {
                    let mut expect = [0u8; Q4_ROW];
                    encode_q4lm_row_from_q8(&shard[r * Q8_ROW..(r + 1) * Q8_ROW], &mut expect)
                        .unwrap();
                    let got = table_row(&bytes, &h, s * ROWS + r, &[]);
                    assert_eq!(got, expect, "shard {s} row {r}");
                    let mut a = vec![0u8; WIDTH * 2];
                    let mut b = vec![0u8; WIDTH * 2];
                    decode_q4lm_row_bf16(&got, &mut a).unwrap();
                    decode_q4lm_row_bf16(&expect, &mut b).unwrap();
                    assert_eq!(a, b);
                }
                src_digests.push(<[u8; 32]>::from(Sha256::digest(shard)));
            }
            assert_eq!(t.source_rows_sha256, hex(&merkle_sha256(&src_digests)));
            assert_eq!(report.source_rows_sha256, t.source_rows_sha256);
            assert!(!append_suffix(&report.output, ".tmp").exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn derive_rank_order_places_rows_in_tier_order() {
        let fx = fixture(false);
        let buckets = random_buckets(7);
        let rank = fx.dir.path().join("rank.plehot");
        write_rank(&rank, &buckets);

        let row_report = derive(&fx.model, None, &tuning()).unwrap();
        let row_bytes = read_all(&row_report.output);
        let row_h = read_table(&row_report.output);
        fs::remove_file(&row_report.output).unwrap();

        let report = derive(&fx.model, Some(rank.as_path()), &tuning()).unwrap();
        assert_eq!(report.order, "rank");
        let h = read_table(&report.output);
        let bytes = read_all(&report.output);
        assert_eq!(h.order, PackOrder::Rank);
        assert_eq!(h.tier_group, DEFAULT_TIER_BUCKETS);
        assert_eq!(h.kept_buckets, 256);
        assert_eq!(h.tier_rows.len(), 256usize.div_ceil(DEFAULT_TIER_BUCKETS));
        assert_eq!(h.tier_rows.iter().sum::<u64>(), PADDED as u64);
        assert_eq!(h.rank_bytes, PADDED as u64);
        assert!(h.rank_offset > 0 && h.rank_offset % 4096 == 0 && h.data_offset % 4096 == 0);
        assert!(h.data_offset >= h.rank_offset + h.rank_bytes);
        assert_eq!(h.rank_payload_sha256, hex(&Sha256::digest(&buckets)));
        let rank_at = h.rank_offset as usize;
        assert_eq!(&bytes[rank_at..rank_at + PADDED], buckets.as_slice());
        assert_eq!(h.ple_fingerprint, fingerprint());
        assert_digests(&bytes, &h);
        // Same source rows and the same encoded rows, just permuted.
        let (t, row_t) = (h.table.as_ref().unwrap(), row_h.table.as_ref().unwrap());
        assert_eq!(t.source_rows_sha256, row_t.source_rows_sha256);
        for row in 0..PADDED {
            let want = table_row(&row_bytes, &row_h, row, &[]);
            let got = table_row(&bytes, &h, row, &buckets);
            assert_eq!(got, want, "row {row}");
            let mut a = vec![0u8; WIDTH * 2];
            let mut b = vec![0u8; WIDTH * 2];
            decode_q4lm_row_bf16(&got, &mut a).unwrap();
            decode_q4lm_row_bf16(&want, &mut b).unwrap();
            assert_eq!(a, b);
        }
    }

    #[cfg(unix)]
    #[test]
    fn derive_is_deterministic_across_threads_and_chunking() {
        let fx = fixture(true);
        let rank = fx.dir.path().join("rank.plehot");
        write_rank(&rank, &random_buckets(11));
        for rank_arg in [None, Some(rank.as_path())] {
            let out = q4_path(&fx.model);
            let run = |t: &Tuning| {
                derive(&fx.model, rank_arg, t).unwrap();
                let bytes = read_all(&out);
                let header = read_table(&out);
                fs::remove_file(&out).unwrap();
                (bytes, header)
            };
            let (a_bytes, a_h) = run(&tuning());
            // Same segment size, different threading and chunking: identical.
            let (b_bytes, _) = run(&Tuning {
                threads: 8,
                chunk_bytes: 1,
                segment_bytes: 1000,
            });
            assert_eq!(a_bytes, b_bytes);
            let (c_bytes, _) = run(&Tuning {
                threads: 1,
                chunk_bytes: 1 << 20,
                segment_bytes: 1000,
            });
            assert_eq!(a_bytes, c_bytes);
            // The segment size only changes header digests, never the rows.
            let (d_bytes, d_h) = run(&Tuning {
                threads: 2,
                chunk_bytes: 64,
                segment_bytes: TABLE_SEGMENT_BYTES,
            });
            assert_eq!(d_h.table.as_ref().unwrap().segment_sha256.len(), 1);
            assert_eq!(
                &a_bytes[a_h.data_offset as usize..],
                &d_bytes[d_h.data_offset as usize..]
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn derive_refuses_a_mismatched_rank() {
        let fx = fixture(true);
        let out = q4_path(&fx.model);
        // Wrong padded_rows.
        let short = fx.dir.path().join("short.plehot");
        write_ple_hot_rank(
            &short,
            16,
            &fingerprint(),
            &[0u8; 16],
            serde_json::json!({}),
        )
        .unwrap();
        let err = derive(&fx.model, Some(short.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("padded_rows"), "{err}");
        // Wrong fingerprint.
        let other = fx.dir.path().join("other.plehot");
        write_ple_hot_rank(
            &other,
            PADDED as u64,
            &"ab".repeat(32),
            &[0u8; PADDED],
            serde_json::json!({}),
        )
        .unwrap();
        let err = derive(&fx.model, Some(other.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("fingerprint"), "{err}");
        // Missing explicit rank.
        let missing = fx.dir.path().join("nope.plehot");
        let err = derive(&fx.model, Some(missing.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
        assert!(!out.exists() && !append_suffix(&out, ".tmp").exists());
    }

    #[cfg(unix)]
    #[test]
    fn derive_picks_up_the_default_rank_next_to_the_model() {
        let fx = fixture(true);
        let buckets = random_buckets(5);
        write_rank(&append_suffix(&fx.model, ".plehot"), &buckets);
        let report = derive(&fx.model, None, &tuning()).unwrap();
        assert_eq!(report.order, "rank");
        assert_eq!(read_table(&report.output).order, PackOrder::Rank);
    }

    #[cfg(unix)]
    #[test]
    fn split_writes_a_verified_base_and_q8_table() {
        for shards_first in [true, false] {
            let fx = fixture(shards_first);
            let out_base = fx.dir.path().join("flash-base.mq4-xts");
            let mut phases: Vec<&'static str> = Vec::new();
            let mut cb = |p: &PleProgress| {
                if phases.last() != Some(&p.phase) {
                    phases.push(p.phase);
                }
            };
            let report =
                split_with(&fx.model, Some(out_base.as_path()), &tuning(), &mut cb).unwrap();
            assert_eq!(phases, ["copy-ple", "copy-base", "verify"]);
            assert_eq!(report.base, out_base);
            assert_eq!(report.ple_q8, q8_path(&out_base));
            assert!(!append_suffix(&report.base, ".tmp").exists());
            assert!(!append_suffix(&report.ple_q8, ".tmp").exists());
            assert!(fx.model.exists(), "the original is never removed");

            let src = HfqFile::open(&fx.model).unwrap();
            let base = HfqFile::open(&out_base).unwrap();
            assert_eq!(report.base_bytes, base.file_len());
            assert_eq!(report.base_index_sha256, hex(&base.index_sha256().unwrap()));
            let kept_src: Vec<&HfqTensorInfo> = src
                .tensor_infos()
                .iter()
                .filter(|t| !t.name.starts_with(PLE_SHARD_PREFIX))
                .collect();
            assert_eq!(base.tensor_infos().len(), kept_src.len());
            assert_eq!(kept_src.len(), fx.kept.len());
            for (index, (g, k)) in base.tensor_infos().iter().zip(&kept_src).enumerate() {
                assert_eq!(
                    (&g.name, g.quant_type, &g.shape, g.group_size, g.data_size),
                    (&k.name, k.quant_type, &k.shape, k.group_size, k.data_size)
                );
                // The first kept payload always keeps its 4 KiB residue; with the
                // shards as one prefix (the Flash-Next layout) every kept one does.
                if shards_first || index == 0 {
                    assert_eq!(g.data_offset % 4096, k.data_offset % 4096, "{}", g.name);
                }
                let (_, data) = base.tensor_data_vec(&g.name).unwrap();
                let want = fx.kept.iter().find(|(n, _)| *n == g.name).unwrap();
                assert_eq!(data, want.1);
            }
            assert!(base
                .tensor_infos()
                .iter()
                .all(|t| !t.name.starts_with(PLE_SHARD_PREFIX)));
            // Minimal data offset: less than one 4 KiB step past the index end.
            let first = &base.tensor_infos()[0];
            let index_end = HFQM_HEADER_BYTES as usize
                + base.metadata_json.len()
                + serialize_index(&base.tensor_infos().iter().collect::<Vec<_>>())
                    .unwrap()
                    .len();
            assert!(first.data_offset >= index_end && first.data_offset - index_end < 4096);
            assert_eq!(base.data_offset() as usize, first.data_offset);

            // The q8 table: row order, the exact rows, bound to the new base.
            let h = read_table(&report.ple_q8);
            let t = h.table.as_ref().unwrap();
            assert_eq!(
                (h.version, h.kind, h.order),
                (3, PackKind::PleTable, PackOrder::Row)
            );
            assert_eq!(h.generator, "hipfire ple split");
            assert_eq!(h.row_encoding, RowEncoding::Q8F16);
            assert_eq!(h.encoded_row_bytes, Q8_ROW as u64);
            assert_eq!(h.ple_fingerprint, fingerprint());
            assert_eq!(h.rank_payload_sha256, ZERO_SHA256);
            assert_eq!((h.tier_group, h.kept_buckets), (256, 256));
            assert_eq!(h.tier_rows, vec![PADDED as u64]);
            assert_eq!(t.codebook_f32_bits, None);
            assert_eq!(t.scale_grid_f32_bits, None);
            assert_eq!(t.base_index_sha256, report.base_index_sha256);
            assert_eq!(t.base_file_bytes, report.base_bytes);
            assert_eq!(
                report.ple_q8_bytes,
                fs::metadata(&report.ple_q8).unwrap().len()
            );
            let q8_bytes = read_all(&report.ple_q8);
            assert_digests(&q8_bytes, &h);
            let mut digests = Vec::new();
            for (s, shard) in fx.shards.iter().enumerate() {
                let at = h.data_offset as usize + s * shard.len();
                assert_eq!(&q8_bytes[at..at + shard.len()], shard.as_slice());
                digests.push(<[u8; 32]>::from(Sha256::digest(shard)));
            }
            assert_eq!(t.source_rows_sha256, hex(&merkle_sha256(&digests)));
            assert_eq!(report.q8_rows_sha256, t.source_rows_sha256);

            // Metadata: the original text with the marker inserted at the end.
            let expected_meta = format!(
                "{},\"{PLE_BASE_MARKER_KEY}\":{{\"version\":1,\"location\":\"sidecar\",\
                 \"shard_count\":{SHARDS},\"rows_per_shard\":{ROWS},\"row_width\":{WIDTH},\
                 \"q8_rows_sha256\":\"{}\"}}}}",
                &METADATA[..METADATA.len() - 1],
                t.source_rows_sha256
            );
            assert_eq!(base.metadata_json, expected_meta);
        }
    }

    #[cfg(unix)]
    #[test]
    fn split_defaults_the_base_name_and_is_deterministic() {
        let fx = fixture(true);
        let report = do_split(&fx.model, None, &tuning()).unwrap();
        assert_eq!(report.base, fx.dir.path().join("flash-base.mq4-xts"));
        let (base_a, q8_a) = (read_all(&report.base), read_all(&report.ple_q8));
        fs::remove_file(&report.base).unwrap();
        fs::remove_file(&report.ple_q8).unwrap();
        let again = do_split(
            &fx.model,
            None,
            &Tuning {
                threads: 1,
                chunk_bytes: 1 << 20,
                segment_bytes: 1000,
            },
        )
        .unwrap();
        assert_eq!(read_all(&again.base), base_a);
        assert_eq!(read_all(&again.ple_q8), q8_a);
    }

    #[cfg(unix)]
    #[test]
    fn derive_from_split_base_equals_derive_from_monolithic() {
        let fx = fixture(true);
        let rank = fx.dir.path().join("rank.plehot");
        write_rank(&rank, &random_buckets(3));
        let split_dir = tempfile::tempdir().unwrap();
        let base_path = split_dir.path().join("flash-base.mq4-xts");
        do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap();
        for rank_arg in [None, Some(rank.as_path())] {
            let mono = derive(&fx.model, rank_arg, &tuning()).unwrap();
            let derived = derive(&base_path, rank_arg, &tuning()).unwrap();
            assert_eq!(derived.source, q8_path(&base_path));
            assert_eq!(derived.payload_sha256, mono.payload_sha256);
            assert_eq!(derived.source_rows_sha256, mono.source_rows_sha256);
            assert_eq!(derived.order, mono.order);

            let (hm, hs) = (read_table(&mono.output), read_table(&derived.output));
            assert_ne!(
                hm.table.as_ref().unwrap().base_index_sha256,
                hs.table.as_ref().unwrap().base_index_sha256
            );
            let (mut a, mut b) = (hm.clone(), hs.clone());
            for h in [&mut a, &mut b] {
                let t = h.table.as_mut().unwrap();
                t.base_index_sha256.clear();
                t.base_file_bytes = 0;
                h.data_offset = 0;
                h.rank_offset = 0;
            }
            assert_eq!(a, b);
            let (bm, bs) = (read_all(&mono.output), read_all(&derived.output));
            assert_eq!(
                &bm[hm.data_offset as usize..],
                &bs[hs.data_offset as usize..]
            );
            // The derived table binds to the split base, not to the original.
            let base = HfqFile::open(&base_path).unwrap();
            let t = hs.table.as_ref().unwrap();
            assert_eq!(t.base_index_sha256, hex(&base.index_sha256().unwrap()));
            assert_eq!(t.base_file_bytes, base.file_len());
            fs::remove_file(&mono.output).unwrap();
            fs::remove_file(&derived.output).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn derive_rejects_a_corrupt_q8_sidecar_and_leaves_no_tmp() {
        let fx = fixture(true);
        let base_path = fx.dir.path().join("flash-base.mq4-xts");
        let report = do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap();
        let mut bytes = read_all(&report.ple_q8);
        let header = read_table(&report.ple_q8);
        bytes[header.data_offset as usize + 3] ^= 0x40;
        fs::write(&report.ple_q8, bytes).unwrap();
        let err = derive(&base_path, None, &tuning()).unwrap_err();
        assert!(err.contains("corrupt"), "{err}");
        assert!(!q4_path(&base_path).exists());
        assert!(!append_suffix(&q4_path(&base_path), ".tmp").exists());
    }

    #[cfg(unix)]
    #[test]
    fn derive_rejects_a_q8_sidecar_bound_to_another_base() {
        let fx = fixture(true);
        let base_path = fx.dir.path().join("flash-base.mq4-xts");
        let report = do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap();
        fs::copy(&report.ple_q8, q8_path(&fx.model)).unwrap();
        let err = derive(&fx.model, None, &tuning()).unwrap_err();
        assert!(err.contains("different base"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn derive_refuses_existing_output_and_split_base_without_q8() {
        let fx = fixture(true);
        derive(&fx.model, None, &tuning()).unwrap();
        let err = derive(&fx.model, None, &tuning()).unwrap_err();
        assert!(
            err.contains("already exists") && err.contains("delete"),
            "{err}"
        );

        let base_path = fx.dir.path().join("flash-base.mq4-xts");
        let report = do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap();
        fs::remove_file(&report.ple_q8).unwrap();
        let err = derive(&base_path, None, &tuning()).unwrap_err();
        assert!(
            err.contains("Q8 rows sidecar") && err.contains("missing"),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn derive_refuses_when_the_tmp_is_locked_and_restarts_stale_tmp() {
        let fx = fixture(true);
        let out = q4_path(&fx.model);
        let tmp = append_suffix(&out, ".tmp");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&tmp)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let err = derive(&fx.model, None, &tuning()).unwrap_err();
        assert!(err.contains("locked by another process"), "{err}");
        assert!(
            tmp.exists(),
            "a competing converter's tmp must not be deleted"
        );
        assert!(!out.exists());
        drop(holder);

        // A stale (unlocked) tmp of a crashed run is truncated and restarted.
        fs::write(&tmp, vec![0xAAu8; 1 << 16]).unwrap();
        derive(&fx.model, None, &tuning()).unwrap();
        assert!(!tmp.exists());
        let h = read_table(&out);
        assert_eq!(read_all(&out).len() as u64, h.data_offset + h.data_bytes);
    }

    #[cfg(unix)]
    #[test]
    fn split_refuses_overlay_trailing_bytes_existing_outputs_and_locks() {
        // Metadata overlay flag in the version word.
        let fx = fixture(true);
        let mut bytes = read_all(&fx.model);
        bytes[7] |= 0x80;
        let overlay = fx.dir.path().join("overlay.mq4-xts");
        fs::write(&overlay, &bytes).unwrap();
        let err = do_split(&overlay, None, &tuning()).unwrap_err();
        assert!(err.contains("overlay"), "{err}");

        // Trailing bytes.
        let mut bytes = read_all(&fx.model);
        bytes.extend_from_slice(b"junk");
        let trailing = fx.dir.path().join("trailing.mq4-xts");
        fs::write(&trailing, &bytes).unwrap();
        let err = do_split(&trailing, None, &tuning()).unwrap_err();
        assert!(err.contains("trailing"), "{err}");

        // Existing outputs.
        let base_path = fx.dir.path().join("out-base.mq4-xts");
        fs::write(q8_path(&base_path), b"x").unwrap();
        let err = do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        fs::remove_file(q8_path(&base_path)).unwrap();
        fs::write(&base_path, b"x").unwrap();
        let err = do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        fs::remove_file(&base_path).unwrap();

        // Output equal to the input.
        let err = do_split(&fx.model, Some(fx.model.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("is the input"), "{err}");

        // Locked tmp.
        let tmp = append_suffix(&base_path, ".tmp");
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&tmp)
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let err = do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap_err();
        assert!(err.contains("locked by another process"), "{err}");
        assert!(tmp.exists());
        assert!(!append_suffix(&q8_path(&base_path), ".tmp").exists());
        drop(holder);
        fs::remove_file(&tmp).unwrap();
        do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn split_refuses_a_split_base_and_models_without_shards() {
        let fx = fixture(true);
        let base_path = fx.dir.path().join("flash-base.mq4-xts");
        do_split(&fx.model, Some(base_path.as_path()), &tuning()).unwrap();
        let err = do_split(&base_path, None, &tuning()).unwrap_err();
        assert!(err.contains("already a split base"), "{err}");

        let plain = fx.dir.path().join("plain.mq4-xts");
        write_hfqm_package_mem(
            &plain,
            16,
            METADATA,
            &[mem(
                "model.embed_tokens.weight",
                1,
                vec![4],
                0,
                vec![1, 2, 3, 4],
            )],
        )
        .unwrap();
        let err = do_split(&plain, None, &tuning()).unwrap_err();
        assert!(err.contains("no PLE shard tensors"), "{err}");
        let err = derive(&plain, None, &tuning()).unwrap_err();
        assert!(err.contains("no PLE shard tensors"), "{err}");
    }

    #[test]
    fn q8_pieces_cut_at_chunk_shard_and_segment_boundaries() {
        let (total, shard, chunk, seg) = (4 * 544, 544, 200, 1000);
        let pieces = q8_pieces(total, shard, chunk, seg);
        let mut at = 0;
        for &(start, len) in &pieces {
            assert_eq!(start, at);
            assert!(len > 0 && len <= chunk);
            for g in [chunk, shard, seg] {
                assert_eq!(
                    start / g,
                    (start + len - 1) / g,
                    "piece {start}+{len} straddles {g}"
                );
            }
            at += len;
        }
        assert_eq!(at, total);
    }

    #[test]
    fn tier_plan_prefix_counts_match_a_direct_count() {
        let layout = PleShardLayout {
            shard_count: SHARDS,
            rows_per_shard: ROWS,
            row_width: WIDTH,
        };
        let buckets = random_buckets(21);
        for chunk_rows in [1usize, 3, ROWS, 100] {
            let plan = TierPlan::new(Some(&buckets), DEFAULT_TIER_BUCKETS, &layout, chunk_rows);
            assert_eq!(plan.tiers, 256usize.div_ceil(DEFAULT_TIER_BUCKETS));
            let cps = ROWS.div_ceil(chunk_rows);
            for s in 0..SHARDS {
                for k in 0..cps {
                    let first = s * ROWS + k * chunk_rows;
                    for t in 0..plan.tiers {
                        let before = buckets[..first]
                            .iter()
                            .filter(|&&b| usize::from(plan.tier_of[usize::from(b)]) == t)
                            .count() as u64;
                        assert_eq!(plan.prefix[(s * cps + k) * plan.tiers + t], before);
                    }
                }
            }
            assert_eq!(plan.tier_rows.iter().sum::<u64>(), PADDED as u64);
            assert_eq!(plan.tier_base_rows[0], 0);
        }
        let row = TierPlan::new(None, BUCKETS, &layout, 3);
        assert_eq!((row.tiers, row.tier_rows.clone()), (1, vec![PADDED as u64]));
        assert_eq!(row.prefix[ROWS.div_ceil(3)], ROWS as u64);
    }

    #[test]
    fn split_marker_handles_an_empty_metadata_object() {
        assert_eq!(
            insert_marker("{}", "\"a\":1,\"s\":\"", "ff"),
            "{\"hipfire_ple_rows\":{\"a\":1,\"s\":\"ff\"}}"
        );
        assert_eq!(
            insert_marker("{\"k\":1 }", "\"a\":1,\"s\":\"", "ff"),
            "{\"k\":1 ,\"hipfire_ple_rows\":{\"a\":1,\"s\":\"ff\"}}"
        );
    }

    #[test]
    fn base_data_offset_preserves_alignment_residue() {
        for index_end in [32u64, 100, 4095, 4096, 4097, 14_400_123] {
            for first in [None, Some(0), Some(1), Some(4095), Some(54_400_000_123u64)] {
                let d = base_data_offset(index_end, first);
                assert!(d >= index_end && d - index_end < DATA_ALIGN);
                assert_eq!(d % DATA_ALIGN, first.map_or(0, |o| o % DATA_ALIGN));
            }
        }
    }

    #[test]
    fn shard_discovery_requires_a_complete_canonical_set() {
        let info = |name: String, qt: u8, shape: Vec<u32>, size: usize| HfqTensorInfo {
            name,
            quant_type: qt,
            shape,
            group_size: 32,
            data_offset: 0,
            data_size: size,
        };
        let good = |s: usize| {
            info(
                ple_shard_name(s),
                3,
                vec![ROWS as u32, WIDTH as u32],
                ROWS * Q8_ROW,
            )
        };
        let other = info("lm_head.weight".to_string(), 1, vec![4], 4);
        assert!(find_shards(&[other.clone()]).unwrap().is_none());
        let (layout, index) = find_shards(&[other.clone(), good(1), good(0)])
            .unwrap()
            .unwrap();
        assert_eq!(
            layout,
            PleShardLayout {
                shard_count: 2,
                rows_per_shard: ROWS,
                row_width: WIDTH
            }
        );
        assert_eq!(index, vec![2, 1]);
        assert!(find_shards(&[good(0), good(2)])
            .unwrap_err()
            .contains("missing"));
        let bad_qt = info(
            ple_shard_name(1),
            13,
            vec![ROWS as u32, WIDTH as u32],
            ROWS * Q8_ROW,
        );
        assert!(find_shards(&[good(0), bad_qt]).is_err());
        let bad_size = info(ple_shard_name(1), 3, vec![ROWS as u32, WIDTH as u32], 5);
        assert!(find_shards(&[good(0), bad_size]).is_err());
    }
}
