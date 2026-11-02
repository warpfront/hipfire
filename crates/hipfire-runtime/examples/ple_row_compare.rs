// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only PLE table row-payload comparison: are the *encoded row bytes* of
//! two `ple-table` sidecars (`HFPLEHP1` v3) identical, row for row in canonical
//! logical row order (row 0 .. `padded_rows - 1`), regardless of whether each
//! file stores its rows in rank (tier) order or in row order?
//!
//! ```text
//! cargo run --release -p hipfire-runtime --example ple_row_compare -- \
//!     --left A.ple-q4 --right B.ple-q4 [--out report.json] \
//!     [--chunk-rows 65536] [--sample-limit 8]
//! ```
//!
//! * Row -> file-offset mapping is the product's own, never re-derived here:
//!   - rank order: `HotRowStore::pack_offset(row)` of the store
//!     `PleRowsFile::open_table_store` opens (embedded rank sha256-checked,
//!     `tier_of[bucket[row]]`, tier base + slot * `encoded_row_bytes`; an
//!     absolute file offset);
//!   - row order: `PleRowsFile::shard_offset(row / rows_per_shard)` +
//!     `(row % rows_per_shard) * encoded_row_bytes`.
//!     Every returned offset is range/alignment checked against the header's
//!     data section and marked in a bitset, so a side is accepted only if its
//!     mapping is an exact bijection onto the `padded_rows` row slots.
//! * Rows are borrowed straight out of a read-only `mmap` (no per-row read
//!   syscall, no per-row allocation, no file-sized copy). Row equality is
//!   `[u8] ==` (memcmp) on the raw encoded bytes; no decoding, no tolerance.
//! * Each side's `canonical_logical_order_sha256` is the standard SHA-256 over
//!   the concatenation of its encoded rows in logical row order, nothing else.
//!   It is NOT the header `data_sha256` / `segment_sha256` (those hash the
//!   stored order and are reported only as informational, order-dependent
//!   fields), and it is not a commutative surrogate.
//! * Comparison is refused (`not_comparable`) unless the gated header fields
//!   agree: row encoding, row width, encoded row bytes, padded rows, rows per
//!   shard, data bytes, PLE fingerprint, source encoding, `source_rows_sha256`,
//!   codebook bits, scale-grid bits.
//!
//! Verdict / exit code: `byte_identical` 0, `error` 1, `not_comparable` 2,
//! `rows_differ` 3. The report is printed to stdout (and to `--out`).
//!
//! Memory: anonymous memory is the rank index of each rank-ordered side
//! (`padded_rows` bucket bytes + `tiers * ceil(padded_rows / 512) * 4` slot
//! prefix bytes), one coverage bitset (`padded_rows / 8` bytes) per side, and
//! two offset buffers of `chunk_rows * 8` bytes. The mapped table pages are
//! file-backed and reclaimable; they are unmapped (`MADV_DONTNEED`, no data
//! loss, page cache untouched) every `DROP_EVERY_CHUNKS` chunks so resident
//! mapped bytes stay bounded. Measured `/proc/self/status` figures are in the
//! report.
//!
//! Local bytes only: this example makes no claim about any remote / published
//! artifact.

use hipfire_runtime::ple_hot_pack::{PackKind, PackOrder, PleHotPackHeader, PleTableFields};
use hipfire_runtime::ple_hot_store::HotRowStore;
use hipfire_runtime::ple_sidecar::PleRowsFile;
use memmap2::{Mmap, UncheckedAdvice};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Rows per prefix-count block of the store's slot index
/// (`ple_hot_store::BLOCK_ROWS`; used only for the memory estimate).
const SLOT_INDEX_BLOCK_ROWS: u64 = 512;
/// Unmap resident table pages (and report progress) every this many chunks.
const DROP_EVERY_CHUNKS: u64 = 64;
const DEFAULT_CHUNK_ROWS: usize = 65_536;
const DEFAULT_SAMPLE_LIMIT: usize = 8;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn canon(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

/// One `/proc/self/status` field in KiB.
fn status_kib(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))
        .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
}

fn mem_snapshot() -> Value {
    json!({
        "vm_hwm_kib": status_kib("VmHWM"),
        "vm_rss_kib": status_kib("VmRSS"),
        "rss_anon_kib": status_kib("RssAnon"),
        "rss_file_kib": status_kib("RssFile"),
    })
}

/// Where a row's encoded bytes live in the file.
enum Mapper {
    /// `shard_offsets[s]` from `PleRowsFile::shard_offset(s)`.
    Row {
        shard_offsets: Vec<u64>,
        rows_per_shard: u64,
        encoded_row_bytes: u64,
    },
    /// `HotRowStore::pack_offset`.
    Rank(Arc<HotRowStore>),
}

impl Mapper {
    fn offset(&self, row: u64) -> Result<u64, String> {
        match self {
            Mapper::Row {
                shard_offsets,
                rows_per_shard,
                encoded_row_bytes,
            } => {
                let shard = usize::try_from(row / rows_per_shard)
                    .map_err(|_| format!("row {row}: shard index overflows"))?;
                let base = shard_offsets
                    .get(shard)
                    .ok_or_else(|| format!("row {row}: shard {shard} is out of range"))?;
                Ok(base + (row % rows_per_shard) * encoded_row_bytes)
            }
            Mapper::Rank(store) => store
                .pack_offset(row)
                .ok_or_else(|| format!("row {row}: the rank store has no pack offset for it")),
        }
    }
}

struct Side {
    label: &'static str,
    path: PathBuf,
    header: PleHotPackHeader,
    table: PleTableFields,
    map: Mmap,
    mapper: Mapper,
    erb: usize,
    /// One bit per data row slot, set when a row mapped onto it.
    seen: Vec<u64>,
    distinct: u64,
    /// Absolute file offsets of the current chunk's rows.
    offsets: Vec<u64>,
    hasher: Sha256,
}

impl Side {
    fn open(label: &'static str, path: &Path, chunk_rows: usize) -> Result<Side, String> {
        let err = |what: &str| format!("{label} {}: {what}", path.display());
        let file = PleRowsFile::open(path)?;
        let header = file.header().clone();
        let table = file.table().clone();
        if header.kind != PackKind::PleTable {
            return Err(err("is not a ple-table"));
        }
        let erb =
            usize::try_from(header.encoded_row_bytes).map_err(|_| err("row bytes overflow"))?;
        let rows = header.padded_rows;
        if erb == 0 || rows == 0 || header.rows_per_shard == 0 {
            return Err(err("empty geometry"));
        }
        if rows.checked_mul(header.encoded_row_bytes) != Some(header.data_bytes) {
            return Err(err("data_bytes != padded_rows * encoded_row_bytes"));
        }
        let mapper = match header.order {
            PackOrder::Row => {
                let mut shard_offsets = Vec::with_capacity(file.shard_count());
                for shard in 0..file.shard_count() {
                    shard_offsets.push(
                        file.shard_offset(shard)
                            .ok_or_else(|| err(&format!("no offset for shard {shard}")))?,
                    );
                }
                if shard_offsets.len() as u64 * header.rows_per_shard != rows {
                    return Err(err("shards * rows_per_shard != padded_rows"));
                }
                Mapper::Row {
                    shard_offsets,
                    rows_per_shard: header.rows_per_shard,
                    encoded_row_bytes: header.encoded_row_bytes,
                }
            }
            PackOrder::Rank => {
                let store = file.open_table_store()?;
                if store.padded_rows() != rows || store.encoded_row_bytes() != erb {
                    return Err(err("rank store geometry differs from the header"));
                }
                Mapper::Rank(store)
            }
        };
        drop(file);
        let handle = File::open(path).map_err(|e| err(&format!("open for mmap: {e}")))?;
        // SAFETY: read-only mapping of a table that is not modified while this
        // example runs (the same assumption every table reader makes); the
        // length is checked against the header below.
        let map = unsafe { Mmap::map(&handle) }.map_err(|e| err(&format!("mmap: {e}")))?;
        if map.len() as u64 != header.data_offset + header.data_bytes {
            return Err(err("file length != data_offset + data_bytes"));
        }
        let words = usize::try_from(rows.div_ceil(64)).map_err(|_| err("row count overflows"))?;
        Ok(Side {
            label,
            path: path.to_path_buf(),
            header,
            table,
            map,
            mapper,
            erb,
            seen: vec![0; words],
            distinct: 0,
            offsets: vec![0; chunk_rows],
            hasher: Sha256::new(),
        })
    }

    /// Map rows `start..start + n` to offsets, check them, and feed the rows to
    /// the canonical hasher in logical order.
    fn prepare(&mut self, start: u64, n: usize) -> Result<(), String> {
        let erb = self.erb as u64;
        let (data_offset, data_bytes) = (self.header.data_offset, self.header.data_bytes);
        for i in 0..n {
            let row = start + i as u64;
            let off = self.mapper.offset(row)?;
            let rel = off
                .checked_sub(data_offset)
                .filter(|rel| rel % erb == 0 && rel + erb <= data_bytes)
                .ok_or_else(|| {
                    format!(
                        "{} row {row}: offset {off} is not a row slot of the data section \
                         [{data_offset}, {})",
                        self.label,
                        data_offset + data_bytes
                    )
                })?;
            let slot = (rel / erb) as usize;
            let (word, bit) = (slot / 64, 1u64 << (slot % 64));
            if self.seen[word] & bit != 0 {
                return Err(format!(
                    "{} row {row}: offset {off} (slot {slot}) was already mapped by an earlier row",
                    self.label
                ));
            }
            self.seen[word] |= bit;
            self.distinct += 1;
            self.offsets[i] = off;
            let at = off as usize;
            self.hasher.update(&self.map[at..at + self.erb]);
        }
        Ok(())
    }

    fn drop_resident(&self) {
        // SAFETY: MADV_DONTNEED on a read-only shared file mapping only unmaps
        // its pages (they fault back in from the page cache); no slice of the
        // map is alive between chunks.
        unsafe { self.map.unchecked_advise(UncheckedAdvice::DontNeed) }.ok();
    }

    fn order_name(&self) -> &'static str {
        self.header.order.name()
    }

    fn mapping_source(&self) -> &'static str {
        match self.header.order {
            PackOrder::Rank => {
                "HotRowStore::pack_offset via PleRowsFile::open_table_store \
                 (ple_hot_store.rs: tier_base[tier_of[bucket[row]]] + slot(row) * encoded_row_bytes)"
            }
            PackOrder::Row => {
                "PleRowsFile::shard_offset(row / rows_per_shard) + (row % rows_per_shard) * \
                 encoded_row_bytes (ple_sidecar.rs)"
            }
        }
    }

    fn rank_index_bytes(&self) -> u64 {
        match self.header.order {
            PackOrder::Rank => {
                let blocks = self.header.padded_rows.div_ceil(SLOT_INDEX_BLOCK_ROWS);
                self.header.padded_rows + self.header.tier_rows.len() as u64 * blocks * 4
            }
            PackOrder::Row => 0,
        }
    }

    fn describe(&self, canonical_sha: Option<&str>) -> Value {
        let (h, t) = (&self.header, &self.table);
        json!({
            "path": canon(&self.path),
            "file_bytes": h.data_offset + h.data_bytes,
            "order": self.order_name(),
            "mapping_source": self.mapping_source(),
            "canonical_logical_order_sha256": canonical_sha,
            "coverage": {
                "rows_mapped": self.distinct,
                "distinct_row_slots": self.distinct,
                "row_slots_in_file": h.padded_rows,
                "exact_bijection": self.distinct == h.padded_rows,
            },
            "header": {
                "version": h.version,
                "encoding": h.row_encoding.dtype_name(),
                "source_encoding": t.source_encoding.dtype_name(),
                "padded_rows": h.padded_rows,
                "rows_per_shard": h.rows_per_shard,
                "row_width": h.row_width,
                "encoded_row_bytes": h.encoded_row_bytes,
                "data_offset": h.data_offset,
                "data_bytes": h.data_bytes,
                "rank_offset": h.rank_offset,
                "rank_bytes": h.rank_bytes,
                "tier_group": h.tier_group,
                "tier_rows": h.tier_rows,
                "ple_fingerprint": h.ple_fingerprint,
                "source_rows_sha256": t.source_rows_sha256,
                "generator": h.generator,
                "base_index_sha256": t.base_index_sha256,
                "base_file_bytes": t.base_file_bytes,
                "rank_payload_sha256": h.rank_payload_sha256,
            },
            "order_dependent_header_digests_not_proof": {
                "header_data_sha256": t.data_sha256,
                "header_segment_count": t.segment_sha256.len(),
            },
        })
    }
}

/// Gated header equality: every entry must be equal for the rows to be
/// comparable.
fn identity_checks(l: &Side, r: &Side) -> (bool, Vec<Value>) {
    let mut all = true;
    let mut out = Vec::new();
    let mut check = |name: &str, left: Value, right: Value| {
        let equal = left == right;
        all &= equal;
        out.push(json!({"field": name, "equal": equal, "left": left, "right": right}));
    };
    let (lh, rh) = (&l.header, &r.header);
    let (lt, rt) = (&l.table, &r.table);
    check(
        "row_encoding",
        json!(lh.row_encoding.dtype_name()),
        json!(rh.row_encoding.dtype_name()),
    );
    check("row_width", json!(lh.row_width), json!(rh.row_width));
    check(
        "encoded_row_bytes",
        json!(lh.encoded_row_bytes),
        json!(rh.encoded_row_bytes),
    );
    check("padded_rows", json!(lh.padded_rows), json!(rh.padded_rows));
    check(
        "rows_per_shard",
        json!(lh.rows_per_shard),
        json!(rh.rows_per_shard),
    );
    check("data_bytes", json!(lh.data_bytes), json!(rh.data_bytes));
    check(
        "ple_fingerprint",
        json!(lh.ple_fingerprint),
        json!(rh.ple_fingerprint),
    );
    check(
        "source_encoding",
        json!(lt.source_encoding.dtype_name()),
        json!(rt.source_encoding.dtype_name()),
    );
    check(
        "source_rows_sha256",
        json!(lt.source_rows_sha256),
        json!(rt.source_rows_sha256),
    );
    check(
        "codebook_f32_bits",
        json!(lt.codebook_f32_bits),
        json!(rt.codebook_f32_bits),
    );
    check(
        "scale_grid_f32_bits",
        json!(lt.scale_grid_f32_bits),
        json!(rt.scale_grid_f32_bits),
    );
    (all, out)
}

struct Args {
    left: PathBuf,
    right: PathBuf,
    out: Option<PathBuf>,
    chunk_rows: usize,
    sample_limit: usize,
}

fn parse_args() -> Result<Args, String> {
    let usage = "usage: ple_row_compare --left PATH --right PATH [--out PATH] \
                 [--chunk-rows N] [--sample-limit N]";
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut left, mut right, mut out) = (None, None, None);
    let (mut chunk_rows, mut sample_limit) = (DEFAULT_CHUNK_ROWS, DEFAULT_SAMPLE_LIMIT);
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{usage}"))
        };
        match flag.as_str() {
            "--left" => left = Some(PathBuf::from(value("--left")?)),
            "--right" => right = Some(PathBuf::from(value("--right")?)),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--chunk-rows" => {
                chunk_rows = value("--chunk-rows")?
                    .parse()
                    .map_err(|e| format!("--chunk-rows: {e}"))?
            }
            "--sample-limit" => {
                sample_limit = value("--sample-limit")?
                    .parse()
                    .map_err(|e| format!("--sample-limit: {e}"))?
            }
            other => return Err(format!("unknown argument {other}\n{usage}")),
        }
    }
    if chunk_rows == 0 {
        return Err("--chunk-rows must be > 0".to_string());
    }
    Ok(Args {
        left: left.ok_or(usage)?,
        right: right.ok_or(usage)?,
        out,
        chunk_rows,
        sample_limit,
    })
}

fn run(args: &Args) -> Result<(Value, i32), String> {
    let started = Instant::now();
    let mut left = Side::open("left", &args.left, args.chunk_rows)?;
    let mut right = Side::open("right", &args.right, args.chunk_rows)?;
    let after_open = mem_snapshot();
    eprintln!(
        "ple_row_compare: opened both tables in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let (comparable, checks) = identity_checks(&left, &right);
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    if !comparable {
        let report = json!({
            "verdict": "not_comparable",
            "command": command,
            "reason": "gated header fields differ; row payloads are not compared",
            "header_identity_checks": checks,
            "left": left.describe(None),
            "right": right.describe(None),
            "note": "local bytes only; no remote/HF identity claimed",
        });
        return Ok((report, 2));
    }

    let rows = left.header.padded_rows;
    let erb = left.erb;
    let mut mismatches = 0u64;
    let mut first_mismatch: Option<Value> = None;
    let mut samples: Vec<Value> = Vec::new();
    let mut peak_anon_kib = status_kib("RssAnon").unwrap_or(0);
    let mut start = 0u64;
    let mut chunk_index = 0u64;
    let stream_started = Instant::now();

    while start < rows {
        let n = (rows - start).min(args.chunk_rows as u64) as usize;
        let (done_left, done_right) = std::thread::scope(|scope| {
            let worker = scope.spawn(|| left.prepare(start, n));
            let here = right.prepare(start, n);
            (worker.join().expect("left mapping thread panicked"), here)
        });
        done_left?;
        done_right?;

        for i in 0..n {
            let (lo, ro) = (left.offsets[i] as usize, right.offsets[i] as usize);
            let (l, r) = (&left.map[lo..lo + erb], &right.map[ro..ro + erb]);
            if l != r {
                mismatches += 1;
                if first_mismatch.is_some() && samples.len() >= args.sample_limit {
                    continue;
                }
                let sample = json!({
                    "row": start + i as u64,
                    "left_file_offset": lo,
                    "right_file_offset": ro,
                    "first_differing_byte_in_row": l.iter().zip(r).position(|(a, b)| a != b),
                    "left_row_hex": hex(l),
                    "right_row_hex": hex(r),
                });
                if first_mismatch.is_none() {
                    first_mismatch = Some(sample.clone());
                }
                if samples.len() < args.sample_limit {
                    samples.push(sample);
                }
            }
        }

        start += n as u64;
        chunk_index += 1;
        if chunk_index % DROP_EVERY_CHUNKS == 0 || start == rows {
            left.drop_resident();
            right.drop_resident();
            peak_anon_kib = peak_anon_kib.max(status_kib("RssAnon").unwrap_or(0));
            let secs = stream_started.elapsed().as_secs_f64();
            eprintln!(
                "ple_row_compare: {start}/{rows} rows ({:.1}%), {:.1}s, {:.2}M rows/s, \
                 {mismatches} mismatching rows, anon {:.0} MiB, file-mapped {:.0} MiB",
                start as f64 * 100.0 / rows as f64,
                secs,
                start as f64 / secs.max(1e-9) / 1e6,
                status_kib("RssAnon").unwrap_or(0) as f64 / 1024.0,
                status_kib("RssFile").unwrap_or(0) as f64 / 1024.0,
            );
        }
    }

    let left_sha = hex(&std::mem::take(&mut left.hasher).finalize());
    let right_sha = hex(&std::mem::take(&mut right.hasher).finalize());
    let coverage_exact = left.distinct == rows && right.distinct == rows;
    let sha_equal = left_sha == right_sha;
    let identical = mismatches == 0 && sha_equal && coverage_exact;
    let chunk = args.chunk_rows as u64;
    let bitset_bytes = left.seen.len() as u64 * 8;
    let report = json!({
        "verdict": if identical { "byte_identical" } else { "rows_differ" },
        "command": command,
        "comparison": {
            "rows_compared": rows,
            "row_bytes": erb,
            "payload_bytes_per_side": rows * erb as u64,
            "mismatching_rows": mismatches,
            "first_mismatch": first_mismatch,
            "mismatch_samples": samples,
            "canonical_sha256_equal": sha_equal,
            "coverage_exact_on_both_sides": coverage_exact,
            "method": "raw encoded row bytes, memcmp ([u8] ==), canonical logical row order; \
                       no decode, no tolerance",
            "canonical_sha256_definition": "SHA-256 over the concatenation of the encoded rows \
                       0..padded_rows-1 in logical row order, nothing else",
        },
        "header_identity_checks": checks,
        "left": left.describe(Some(&left_sha)),
        "right": right.describe(Some(&right_sha)),
        "memory_accounting": {
            "chunk_rows": chunk,
            "anonymous_estimate_bytes": {
                "left_rank_index": left.rank_index_bytes(),
                "right_rank_index": right.rank_index_bytes(),
                "coverage_bitsets": 2 * bitset_bytes,
                "offset_buffers": 2 * chunk * 8,
            },
            "file_mapped": "both tables are mmapped read-only (virtual only); resident table \
                            pages are unmapped every 64 chunks and are reclaimable page cache",
            "measured_after_open": after_open,
            "measured_at_end": mem_snapshot(),
            "peak_rss_anon_kib_sampled": peak_anon_kib,
        },
        "timing": {
            "open_and_index_secs": stream_started.duration_since(started).as_secs_f64(),
            "stream_secs": stream_started.elapsed().as_secs_f64(),
            "threads": "2 per chunk (one mapping+hashing thread per side), compare on main",
        },
        "proposed_execution": "nice -n 19 ionice -c3 cargo run --release -p hipfire-runtime \
            --example ple_row_compare -- --left LEFT --right RIGHT --out report.json",
        "note": "local bytes only; no remote/HF identity claimed",
    });
    Ok((report, if identical { 0 } else { 3 }))
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("ple_row_compare: {e}");
            std::process::exit(1);
        }
    };
    let (report, code) = match run(&args) {
        Ok(done) => done,
        Err(e) => (json!({"verdict": "error", "error": e}), 1),
    };
    let text = serde_json::to_string_pretty(&report).expect("report serializes");
    if let Some(out) = &args.out {
        if let Err(e) = std::fs::write(out, &text) {
            eprintln!("ple_row_compare: write {}: {e}", out.display());
            std::process::exit(1);
        }
    }
    println!("{text}");
    std::process::exit(code);
}
