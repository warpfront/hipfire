// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE hot-row store: the most frequently used rows of the external PLE row
//! table, read from a rank-ordered copy of them (the hot-row pack) through the
//! kernel page cache, so on unified-memory GPUs most rows a request needs come
//! from RAM without SSD reads, and the memory they use stays reclaimable.
//!
//! The Qwen4 PLE table is ~320M padded rows of 170 B (Q8F16) stored in shards
//! inside the model file.  A frequency rank of the rows (the `.plehot`
//! sidecar, one bucket byte per row, bucket 0 = hottest) selects which rows
//! are hot.  `hipfire-quantize ple-hot-pack` copies the rows of the hottest
//! buckets *verbatim from the model file* into `<model>.plehotrows`
//! ([`crate::ple_hot_pack`]), grouped in tiers of buckets with each tier's
//! rows in ascending row order, so the hottest rows share OS pages (24 rows
//! per 4 KiB page instead of 24 random-rank rows).  Rows are decoded by the
//! same `RowEncoding::decode_row` the model-file path uses, so the store never
//! changes the bytes a request returns, only where they come from.
//!
//! The pack is ordinary file data: page-cache pages of it are clean, so under
//! memory pressure the kernel drops them (and reads them again from the pack
//! when needed) instead of writing anything to swap.  After the model is
//! ready, a prewarm reads the pack's data from its start (hottest tier
//! first), then reads tier 0 a second time so its pages are promoted to the
//! active list and outlive the colder ones when the cache is reclaimed.  It
//! runs while no request is being served: every request holds the row store's
//! [`FetchActivity`] busy for its whole duration (a `ServingGuard`, and each PLE
//! fetch its own guard; bound by `RowStore::attach_hot_store`), and the prewarm
//! resumes `FETCH_QUIET` after the last guard dropped.
//!
//! A rank-ordered table's worker can also start while the model is still
//! loading ([`LoadFill`]): it reads the hottest prefix of the table at a
//! capped rate until the model is ready.  The weight upload streams the model
//! file through the same page cache and evicts the oldest pages first (the
//! hottest ones read here), so the fill watches the prefix's residency
//! (`mincore`) and re-reads evicted pages, hottest first, before reading on.
//! The prewarm after ready then continues at the end of the resident prefix.
//! Ready never waits for the fill.
//!
//! # Public API (exact signatures)
//!
//! ```text
//! PleHotRank::open(path: &Path, expected_padded_rows: u64, expected_fingerprint_hex: &str)
//!     -> Result<PleHotRank, String>          // everything validated except the payload sha
//! PleHotRank::from_buckets(padded_rows: u64, fingerprint_hex: &str, buckets: Vec<u8>) -> PleHotRank
//! PleHotRank::{padded_rows() -> u64, buckets() -> &[u8], header() -> &serde_json::Value,
//!              verify() -> Result<(), String>}
//! write_ple_hot_rank(path: &Path, padded_rows: u64, fingerprint_hex: &str, buckets: &[u8],
//!                    rank_meta: serde_json::Value) -> Result<(), String>
//! HotRowStore::open(rank: PleHotRank, pack_path: &Path, rows_per_shard: usize,
//!                   encoding: RowEncoding, encoded_row_bytes: usize, shard_offsets: &[u64],
//!                   model_file_bytes: u64)
//!     -> Result<Arc<HotRowStore>, String>
//! HotRowStore::open_table(path: &Path) -> Result<Arc<HotRowStore>, String>
//!                                            // rank-ordered ple-table: serves at once
//! HotRowStore::spawn_worker(self: &Arc<Self>, model_path: &Path, shard_offsets: Vec<u64>)
//!     -> JoinHandle<()>                      // verify, then prewarm once a budget is set
//! HotRowStore::spawn_table_worker(self: &Arc<Self>, load_fill: Option<LoadFill>) -> JoinHandle<()>
//!                                            // table: the same prewarm, no verification;
//!                                            // `load_fill` reads the hottest prefix first
//! HotRowStore::{set_prewarm_budget(bytes: u64), stop(), is_serving() -> bool,
//!               verified() -> Option<bool>, pack_offset(row: u64) -> Option<u64>,
//!               read_pack(offset: u64, dst: &mut [u8]) -> Result<(), SourceError>,
//!               tier_count(), kept_buckets(), data_bytes(), prewarmed_bytes(),
//!               load_filled_bytes(), prewarm_complete(), hot_rows(), hot_pages(),
//!               hot_reads(), row_width()}
//! LoadFill { bytes: u64, bytes_per_sec: u64 }
//! mem_available_bytes() -> Option<u64>
//! ```
//!
//! # Sidecar format (`<model>.plehot`)
//!
//! ```text
//! 0..8    magic b"HFPLEHT1"
//! 8..16   u64 LE header_len
//! header  UTF-8 JSON {version:1, padded_rows, buckets:256, encoded_row_bytes:170,
//!                     ple_fingerprint, payload_sha256, rank:{..}}
//! payload padded_rows bytes, one bucket per global PLE row (0 = hottest)
//! ```

// Outside Unix the store cannot be constructed: its pack reads, `posix_fadvise`
// advice, `mincore` residency view and `O_DIRECT` model reads have no
// implementation here, so the constructors and the worker chain are gated and
// what only they use is dead code.
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{FileExt, OpenOptionsExt};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::external_rows::{FetchActivity, RowEncoding, FETCH_QUIET};
use crate::model_source::SourceError;
use crate::ple_hot_pack::{read_ple_hot_pack_header, PackKind, PackOrder};

const MAGIC: &[u8; 8] = b"HFPLEHT1";
const VERSION: u64 = 1;
/// Frequency buckets in a sidecar.
pub const BUCKETS: usize = 256;
/// Encoded bytes of one Q8F16 PLE row (160 values = 5 blocks of 34 B): the
/// informational `encoded_row_bytes` a sidecar records and the row size of the
/// test fixtures.  Neither the sidecar reader nor the pack depends on it.
pub const ENCODED_ROW_BYTES: usize = 170;
/// Buckets per tier of a hot-row pack.
pub const DEFAULT_TIER_BUCKETS: usize = 5;
/// Rows per prefix-count block of the slot index.
const BLOCK_ROWS: usize = 512;
/// One aligned O_DIRECT read of the pack writer.
pub(crate) const READ_CHUNK: usize = 16 << 20;
pub(crate) const READ_ALIGN: u64 = 4096;
/// The prewarm resumes once no PLE fetch ran for this long
/// ([`FETCH_QUIET`]), so it only reads while no request is being served.
const BUSY_QUIET: Duration = FETCH_QUIET;
/// Bytes per prewarm read.
const PREWARM_CHUNK: usize = 8 << 20;
/// A [`LoadFill`] reads about this much time's worth of its rate per read, so
/// the rate cap holds over short spans too.
const LOAD_FILL_SLICE: Duration = Duration::from_millis(50);
/// Smallest [`LoadFill`] read.
const LOAD_FILL_MIN_CHUNK: usize = 4096;
/// A [`LoadFill`] looks for evicted pages of its prefix this often.
const LOAD_FILL_REPAIR_PERIOD: Duration = Duration::from_secs(1);
/// Pages per `mincore` call of a residency scan.
const RESIDENCY_SCAN_PAGES: usize = 16 << 10;
/// Kept rows spot-checked against the model file before the store serves.
const VERIFY_SAMPLES: usize = 4096;
/// Threads of the spot check (its reads are random 170-B reads).
const VERIFY_THREADS: usize = 16;
/// Sanity bound on a sidecar header.
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// `tier_of` value of a bucket that is not kept.
const NO_TIER: u8 = u8::MAX;
/// `prewarm_budget` before [`HotRowStore::set_prewarm_budget`].
const BUDGET_UNSET: u64 = u64::MAX;

const STATE_VERIFYING: u8 = 0;
const STATE_SERVING: u8 = 1;
const STATE_FAILED: u8 = 2;

const TAG: &str = "qwen4 ple hot store";

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 15)] as char);
    }
    out
}

/// The error text of a hot-row store path that needs Unix-only file I/O.
#[cfg(not(unix))]
fn unsupported(what: &str) -> String {
    format!(
        "{TAG}: {what} is unsupported on Windows: the hot-row store needs Unix \
         positional reads (pread), posix_fadvise and mincore; run on Linux"
    )
}

fn is_hex64(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

// ---------------------------------------------------------------------------
// Sidecar: reader + writer
// ---------------------------------------------------------------------------

/// A validated `.plehot` rank: one bucket byte per padded PLE row.
pub struct PleHotRank {
    header: serde_json::Value,
    padded_rows: u64,
    payload_sha256: String,
    buckets: Vec<u8>,
}

impl PleHotRank {
    /// Read a sidecar and validate magic, version, `padded_rows`, bucket
    /// count, row size, fingerprint, and payload length.  The payload sha256
    /// is NOT checked here (hashing ~320 MB belongs on the fill thread): call
    /// [`PleHotRank::verify`], which `HotRowStore::spawn_fill` does itself.
    pub fn open(
        path: &Path,
        expected_padded_rows: u64,
        expected_fingerprint_hex: &str,
    ) -> Result<PleHotRank, String> {
        let describe = |what: &str, error: &dyn std::fmt::Display| {
            format!("ple hot rank {}: {what}: {error}", path.display())
        };
        let mut file = File::open(path).map_err(|e| describe("open", &e))?;
        let file_len = file.metadata().map_err(|e| describe("stat", &e))?.len();
        let mut prefix = [0u8; 16];
        file.read_exact(&mut prefix)
            .map_err(|e| describe("read preamble", &e))?;
        if &prefix[..8] != MAGIC {
            return Err(format!("ple hot rank {}: bad magic", path.display()));
        }
        let header_len = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes"));
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(format!(
                "ple hot rank {}: implausible header length {header_len}",
                path.display()
            ));
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| describe("read header", &e))?;
        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).map_err(|e| describe("parse header", &e))?;

        let field_u64 = |name: &str| {
            header
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("ple hot rank {}: header lacks `{name}`", path.display()))
        };
        let version = field_u64("version")?;
        if version != VERSION {
            return Err(format!(
                "ple hot rank {}: version {version}, expected {VERSION}",
                path.display()
            ));
        }
        let padded_rows = field_u64("padded_rows")?;
        if padded_rows != expected_padded_rows {
            return Err(format!(
                "ple hot rank {}: padded_rows {padded_rows} != model's {expected_padded_rows}",
                path.display()
            ));
        }
        let buckets = field_u64("buckets")?;
        if buckets != BUCKETS as u64 {
            return Err(format!(
                "ple hot rank {}: {buckets} buckets, expected {BUCKETS}",
                path.display()
            ));
        }
        let row_bytes = field_u64("encoded_row_bytes")?;
        if row_bytes == 0 {
            return Err(format!(
                "ple hot rank {}: encoded_row_bytes is zero",
                path.display()
            ));
        }
        let fingerprint = header
            .get("ple_fingerprint")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("ple hot rank {}: header lacks fingerprint", path.display()))?;
        if !fingerprint.eq_ignore_ascii_case(expected_fingerprint_hex) {
            return Err(format!(
                "ple hot rank {}: PLE fingerprint {fingerprint} does not match the model's \
                 {expected_fingerprint_hex}",
                path.display()
            ));
        }
        let payload_sha256 = header
            .get("payload_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|sha| is_hex64(sha))
            .ok_or_else(|| {
                format!(
                    "ple hot rank {}: header lacks a 64-hex payload_sha256",
                    path.display()
                )
            })?
            .to_ascii_lowercase();
        let expected_len = 16u64
            .checked_add(header_len)
            .and_then(|n| n.checked_add(padded_rows))
            .ok_or_else(|| format!("ple hot rank {}: length overflow", path.display()))?;
        if file_len != expected_len {
            return Err(format!(
                "ple hot rank {}: file is {file_len} bytes, expected {expected_len}",
                path.display()
            ));
        }
        let rows = usize::try_from(padded_rows)
            .map_err(|_| format!("ple hot rank {}: padded_rows overflow", path.display()))?;
        let mut payload = vec![0u8; rows];
        file.read_exact(&mut payload)
            .map_err(|e| describe("read payload", &e))?;
        Ok(PleHotRank {
            header,
            padded_rows,
            payload_sha256,
            buckets: payload,
        })
    }

    /// An in-memory rank (tests, tools).  Its payload hash is computed here,
    /// so [`PleHotRank::verify`] passes.
    pub fn from_buckets(padded_rows: u64, fingerprint_hex: &str, buckets: Vec<u8>) -> PleHotRank {
        assert_eq!(buckets.len() as u64, padded_rows, "one bucket per row");
        let payload_sha256 = hex(&Sha256::digest(&buckets));
        let header = serde_json::json!({
            "version": VERSION,
            "padded_rows": padded_rows,
            "buckets": BUCKETS,
            "encoded_row_bytes": ENCODED_ROW_BYTES,
            "ple_fingerprint": fingerprint_hex,
            "payload_sha256": payload_sha256,
        });
        PleHotRank {
            header,
            padded_rows,
            payload_sha256,
            buckets,
        }
    }

    pub fn padded_rows(&self) -> u64 {
        self.padded_rows
    }

    /// One bucket per padded row (0 = hottest).
    pub fn buckets(&self) -> &[u8] {
        &self.buckets
    }

    /// The parsed sidecar header.
    pub fn header(&self) -> &serde_json::Value {
        &self.header
    }

    /// Check the payload against the header's `payload_sha256`.
    pub fn verify(&self) -> Result<(), String> {
        let actual = hex(&Sha256::digest(&self.buckets));
        if actual == self.payload_sha256 {
            Ok(())
        } else {
            Err(format!(
                "ple hot rank payload sha256 {actual} != header {}",
                self.payload_sha256
            ))
        }
    }
}

/// Write a `.plehot` sidecar atomically (tmp file + rename).  `buckets` holds
/// one byte per padded row; `rank_meta` is stored verbatim as the header's
/// `rank` object.
pub fn write_ple_hot_rank(
    path: &Path,
    padded_rows: u64,
    fingerprint_hex: &str,
    buckets: &[u8],
    rank_meta: serde_json::Value,
) -> Result<(), String> {
    if buckets.len() as u64 != padded_rows {
        return Err(format!(
            "ple hot rank: {} buckets for {padded_rows} padded rows",
            buckets.len()
        ));
    }
    if !is_hex64(fingerprint_hex) {
        return Err(format!(
            "ple hot rank: fingerprint `{fingerprint_hex}` is not 64 hex digits"
        ));
    }
    let header = serde_json::json!({
        "version": VERSION,
        "padded_rows": padded_rows,
        "buckets": BUCKETS,
        "encoded_row_bytes": ENCODED_ROW_BYTES,
        "ple_fingerprint": fingerprint_hex.to_ascii_lowercase(),
        "payload_sha256": hex(&Sha256::digest(buckets)),
        "rank": rank_meta,
    });
    let header_bytes =
        serde_json::to_vec(&header).map_err(|e| format!("ple hot rank header: {e}"))?;
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp_name);
    let write = || -> io::Result<()> {
        let mut file = File::create(&tmp)?;
        file.write_all(MAGIC)?;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&header_bytes)?;
        file.write_all(buckets)?;
        file.flush()?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("ple hot rank {}: write: {e}", path.display())
    })
}

// ---------------------------------------------------------------------------
// O_DIRECT model reads (the pack writer)
// ---------------------------------------------------------------------------

/// A sector-aligned read buffer for O_DIRECT.
pub(crate) struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

impl AlignedBuf {
    pub(crate) fn new(len: usize) -> AlignedBuf {
        let layout = Layout::from_size_align(len, READ_ALIGN as usize).expect("read buffer layout");
        // SAFETY: non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        AlignedBuf {
            ptr: NonNull::new(ptr).expect("read buffer allocation"),
            len,
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: owned, initialised (zeroed) allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe {
            dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align_unchecked(self.len, READ_ALIGN as usize),
            )
        }
    }
}

// SAFETY: exclusively owned bytes.
unsafe impl Send for AlignedBuf {}

/// Positional reader of the model file (the file's own offsets).
pub(crate) trait ShardRead: Sync {
    /// Read up to `dst.len()` bytes at absolute `offset`; 0 at EOF.
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize>;
}

#[cfg(unix)]
/// Open `path` for O_DIRECT reads, falling back to buffered reads on a
/// filesystem that refuses the flag.
pub(crate) fn open_direct(path: &Path) -> io::Result<File> {
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {
            eprintln!(
                "{TAG}: O_DIRECT unsupported on {}; using buffered reads",
                path.display()
            );
            File::open(path)
        }
        other => other,
    }
}

/// `/proc/meminfo` `MemAvailable - CMA` in bytes (page cache counts as
/// available, so a prewarmed pack does not lower it; CMA pageblocks do not
/// serve pinned allocations).
pub fn mem_available_bytes() -> Option<u64> {
    rdna_compute::uma_memory::mem_available_bytes()
}

#[cfg(unix)]
/// Read exactly `dst.len()` bytes at `offset`, retrying interrupted and short
/// reads; a read that hits end of file is a `ShortRead`.
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
    let mut done = 0;
    while done < dst.len() {
        let at = offset + done as u64;
        match file.read_at(&mut dst[done..], at) {
            Ok(0) => {
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: done,
                })
            }
            Ok(n) => done += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(SourceError::Io { offset: at, source }),
        }
    }
    Ok(())
}

/// Unsupported outside Unix: errs without reading.
#[cfg(not(unix))]
fn read_exact_at(_file: &File, offset: u64, _dst: &mut [u8]) -> Result<(), SourceError> {
    Err(SourceError::Io {
        offset,
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            unsupported("positional pack read"),
        ),
    })
}

/// Result of a non-blocking positional read ([`read_nowait_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NowaitRead {
    /// `dst` was filled completely from data already in memory.
    Done,
    /// Nothing usable was read: the bytes are not in the page cache (`EAGAIN`),
    /// the read was short (including end of file), or the kernel/filesystem has
    /// no `RWF_NOWAIT` support.  `dst` content is unspecified; the caller redoes
    /// the whole read through its blocking path, which also reports a genuine
    /// short read or error.
    WouldBlock,
}

/// Set once the kernel or filesystem reported no `RWF_NOWAIT` support, so later
/// reads skip the doomed syscall.  Capability only: real I/O errors never set it.
static NOWAIT_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

/// Read `dst.len()` bytes at `offset` with `preadv2(RWF_NOWAIT)`: the read
/// completes only when every byte is already in the page cache, and never
/// waits for the device.  `Ok(WouldBlock)` (see [`NowaitRead`]) covers a cache
/// miss, a short read and missing `RWF_NOWAIT` support (`ENOSYS`,
/// `EOPNOTSUPP`, `EINVAL`); every other errno is a real I/O error and is
/// returned.
#[cfg(target_os = "linux")]
pub(crate) fn read_nowait_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<NowaitRead> {
    if dst.is_empty() {
        return Ok(NowaitRead::Done);
    }
    if NOWAIT_UNSUPPORTED.load(Ordering::Relaxed) {
        return Ok(NowaitRead::WouldBlock);
    }
    let Ok(at) = libc::off_t::try_from(offset) else {
        return Ok(NowaitRead::WouldBlock);
    };
    match nowait_probe(file, at, dst)? {
        NowaitProbe::Read(read) => Ok(read),
        NowaitProbe::Unsupported => {
            NOWAIT_UNSUPPORTED.store(true, Ordering::Relaxed);
            Ok(NowaitRead::WouldBlock)
        }
    }
}

/// One `preadv2(RWF_NOWAIT)` of a non-empty `dst` (see [`read_nowait_at`]),
/// without the process-wide capability flag.
#[cfg(target_os = "linux")]
fn nowait_probe(file: &File, at: libc::off_t, dst: &mut [u8]) -> io::Result<NowaitProbe> {
    let iov = libc::iovec {
        iov_base: dst.as_mut_ptr().cast(),
        iov_len: dst.len(),
    };
    loop {
        // SAFETY: `iov` describes `dst`, exclusively borrowed for the call; the
        // descriptor is open for the duration of the borrow of `file`.
        let n = unsafe { libc::preadv2(file.as_raw_fd(), &iov, 1, at, libc::RWF_NOWAIT) };
        if n >= 0 {
            return Ok(NowaitProbe::Read(if n as usize == dst.len() {
                NowaitRead::Done
            } else {
                NowaitRead::WouldBlock
            }));
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINTR) => {}
            Some(libc::EAGAIN) => return Ok(NowaitProbe::Read(NowaitRead::WouldBlock)),
            Some(libc::ENOSYS | libc::EOPNOTSUPP | libc::EINVAL) => {
                return Ok(NowaitProbe::Unsupported)
            }
            _ => return Err(error),
        }
    }
}

/// Outcome of [`nowait_probe`].
#[cfg(target_os = "linux")]
#[derive(Debug, PartialEq, Eq)]
enum NowaitProbe {
    Read(NowaitRead),
    /// The kernel or filesystem has no `RWF_NOWAIT` (`ENOSYS`, `EOPNOTSUPP`,
    /// `EINVAL`).
    Unsupported,
}

/// `RWF_NOWAIT` is Linux-only: elsewhere every read takes the blocking path.
#[cfg(not(target_os = "linux"))]
pub(crate) fn read_nowait_at(_: &File, _: u64, _: &mut [u8]) -> io::Result<NowaitRead> {
    Ok(NowaitRead::WouldBlock)
}

/// Test hook of [`HotRowStore::read_pack_nowait`]: `policy(offset)` true =
/// served as if cached, false = would block.
#[cfg(test)]
pub(crate) type NowaitPolicy = std::sync::Arc<dyn Fn(u64) -> bool + Send + Sync>;

#[cfg(unix)]
fn fadvise(file: &File, advice: libc::c_int) {
    // SAFETY: a valid open descriptor; advice only.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, advice) };
}

#[cfg(unix)]
/// Page-cache residency of a file's first bytes, through `mincore` on a
/// shared read-only mapping that is never read through (mapping a file does
/// not read it).
struct Residency {
    addr: NonNull<libc::c_void>,
    len: usize,
    page: u64,
    pages: Vec<u8>,
}

#[cfg(unix)]
impl Residency {
    /// Map the first `len` bytes of `file`.
    fn map(file: &File, len: u64) -> io::Result<Residency> {
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = u64::try_from(page)
            .ok()
            .filter(|&page| page > 0)
            .ok_or_else(io::Error::last_os_error)?;
        let len = usize::try_from(len)
            .ok()
            .filter(|&len| len > 0)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: a fresh read-only shared mapping of an open descriptor; it
        // is only passed to `mincore` and unmapped on drop.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let addr = NonNull::new(addr).ok_or_else(|| io::Error::from(io::ErrorKind::Other))?;
        Ok(Residency {
            addr,
            len,
            page,
            pages: Vec::new(),
        })
    }

    /// The first offset in `from..to` whose page is not in the page cache,
    /// or `to` (clipped to the mapping) when every page is.
    fn first_hole(&mut self, from: u64, to: u64) -> io::Result<u64> {
        let to = to.min(self.len as u64);
        if from >= to {
            return Ok(to);
        }
        let page = self.page;
        let mut first = from / page;
        let last = (to - 1) / page + 1;
        while first < last {
            let count = (last - first).min(RESIDENCY_SCAN_PAGES as u64) as usize;
            self.pages.resize(count, 0);
            // SAFETY: `first..first + count` pages lie inside the mapping
            // (`last` <= its page count), and `pages` holds `count` bytes.
            let rc = unsafe {
                libc::mincore(
                    self.addr
                        .as_ptr()
                        .cast::<u8>()
                        .add((first * page) as usize)
                        .cast(),
                    count * page as usize,
                    self.pages.as_mut_ptr(),
                )
            };
            if rc != 0 {
                return Err(io::Error::last_os_error());
            }
            if let Some(index) = self.pages.iter().position(|&state| state & 1 == 0) {
                return Ok(((first + index as u64) * page).max(from));
            }
            first += count as u64;
        }
        Ok(to)
    }
}

#[cfg(unix)]
impl Drop for Residency {
    fn drop(&mut self) {
        // SAFETY: the mapping `map` created, unmapped once.
        unsafe { libc::munmap(self.addr.as_ptr(), self.len) };
    }
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// The row -> tier index of a rank: bucket -> tier, per-tier row counts before
/// every 512-row block, and the rows of each tier.
struct TierIndex {
    tier_count: usize,
    tier_of: [u8; BUCKETS],
    /// `prefix[t][block]` = rows of tier `t` in rows `[0, block * BLOCK_ROWS)`.
    prefix: Vec<Vec<u32>>,
    tier_rows: Vec<u64>,
}

/// Index `buckets` for tiers of `group` buckets over buckets `0..kept`.
fn build_tier_index(buckets: &[u8], group: usize, kept: usize) -> TierIndex {
    let tier_count = kept.div_ceil(group);
    let mut tier_of = [NO_TIER; BUCKETS];
    for (bucket, slot) in tier_of.iter_mut().enumerate().take(kept) {
        *slot = (bucket / group) as u8;
    }
    let blocks = buckets.len().div_ceil(BLOCK_ROWS);
    let mut prefix: Vec<Vec<u32>> = (0..tier_count)
        .map(|_| Vec::with_capacity(blocks))
        .collect();
    let mut counts = [0u32; BUCKETS];
    for block in buckets.chunks(BLOCK_ROWS) {
        for (tier, column) in prefix.iter_mut().enumerate() {
            column.push(counts[tier]);
        }
        for &b in block {
            counts[usize::from(tier_of[usize::from(b)])] += 1;
        }
    }
    let tier_rows = counts[..tier_count].iter().map(|&n| u64::from(n)).collect();
    TierIndex {
        tier_count,
        tier_of,
        prefix,
        tier_rows,
    }
}

/// Absolute file offset of each tier's first row: tiers follow each other from
/// `data_offset`, `tier_rows[t] * row_bytes` bytes each.
fn tier_bases(data_offset: u64, tier_rows: &[u64], row_bytes: u64) -> Vec<u64> {
    let mut base = data_offset;
    tier_rows
        .iter()
        .map(|&rows| {
            let at = base;
            base += rows * row_bytes;
            at
        })
        .collect()
}

/// The hot rows of the PLE table: where each kept row sits in the hot-row
/// pack, and the open pack.
///
/// A row is kept when its sidecar bucket is below the pack's `kept_buckets`;
/// its tier is `bucket / tier_group`, and its pack offset is the tier's base
/// plus its slot (the tier's rows with a smaller global row index) times the
/// row size.  Nothing is served until the worker thread has checked the
/// sidecar payload hash and spot-checked [`VERIFY_SAMPLES`] kept rows against
/// the model file.
pub struct HotRowStore {
    padded_rows: u64,
    rows_per_shard: usize,
    /// Values per row.
    row_width: usize,
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    /// Buckets per tier.
    group: usize,
    /// Kept buckets `K`: buckets `0..K` are in the pack.
    kept: usize,
    tier_count: usize,
    /// Bucket -> tier (`NO_TIER` when not kept).
    tier_of: [u8; BUCKETS],
    /// The sidecar payload.
    bucket: Box<[u8]>,
    /// `prefix[t][block]` = rows of tier `t` in rows `[0, block * 512)`.
    prefix: Vec<Box<[u32]>>,
    /// Absolute pack offset of each tier's first row.
    tier_base: Vec<u64>,
    /// The pack, opened for random reads (no readahead).
    pack: File,
    pack_path: PathBuf,
    data_offset: u64,
    data_bytes: u64,
    expected_sha: String,
    state: AtomicU8,
    stop: AtomicBool,
    hot_rows: AtomicU64,
    hot_pages: AtomicU64,
    hot_reads: AtomicU64,
    prewarm_budget: AtomicU64,
    prewarmed: AtomicU64,
    prewarm_done: AtomicBool,
    /// Bytes of the data read by the [`LoadFill`] before the model was ready.
    load_filled: AtomicU64,
    /// Bytes of tier 0 re-read by the prewarm's second phase.
    retouched: AtomicU64,
    /// The row store's fetch activity (see [`HotRowStore::bind_activity`]).
    activity: OnceLock<std::sync::Arc<FetchActivity>>,
    /// Test hook: once set, [`HotRowStore::read_pack_nowait`] answers from the
    /// policy instead of the kernel.
    #[cfg(test)]
    nowait_policy: OnceLock<NowaitPolicy>,
}

impl HotRowStore {
    #[cfg(unix)]
    /// Open the hot-row pack at `pack_path` for `rank` and build the row ->
    /// pack-offset index.  Errors when the pack was not cut from this model
    /// (PLE fingerprint, shard offsets, model file size), from this rank
    /// (sidecar payload hash, per-tier row counts), or for this geometry.
    /// Serves nothing until [`HotRowStore::spawn_worker`] has verified it.
    pub fn open(
        rank: PleHotRank,
        pack_path: &Path,
        rows_per_shard: usize,
        encoding: RowEncoding,
        encoded_row_bytes: usize,
        shard_offsets: &[u64],
        model_file_bytes: u64,
    ) -> Result<std::sync::Arc<HotRowStore>, String> {
        let fail = |what: String| format!("ple hot pack {}: {what}", pack_path.display());
        let header = read_ple_hot_pack_header(pack_path)?;
        if header.kind != PackKind::HotPack {
            return Err(fail(
                "is a PLE row table, not a hot pack (open it with HotRowStore::open_table)"
                    .to_string(),
            ));
        }
        let PleHotRank {
            header: rank_header,
            padded_rows,
            payload_sha256,
            buckets,
        } = rank;
        if encoded_row_bytes == 0
            || header.row_encoding != encoding
            || header.encoded_row_bytes != encoded_row_bytes as u64
        {
            return Err(fail(format!(
                "rows are {} B {} in the pack and {encoded_row_bytes} B {} in the row store",
                header.encoded_row_bytes,
                header.row_encoding.dtype_name(),
                encoding.dtype_name()
            )));
        }
        if rows_per_shard == 0
            || header.rows_per_shard != rows_per_shard as u64
            || header.padded_rows != padded_rows
            || padded_rows != shard_offsets.len() as u64 * rows_per_shard as u64
        {
            return Err(fail(format!(
                "geometry {} rows in shards of {}, model {padded_rows} rows in {} shards of \
                 {rows_per_shard}",
                header.padded_rows,
                header.rows_per_shard,
                shard_offsets.len()
            )));
        }
        if padded_rows > u64::from(u32::MAX) {
            return Err(fail(format!("{padded_rows} rows exceed the index range")));
        }
        let rank_fingerprint = rank_header
            .get("ple_fingerprint")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !header
            .ple_fingerprint
            .eq_ignore_ascii_case(rank_fingerprint)
        {
            return Err(fail(format!(
                "PLE fingerprint {} is not the model's {rank_fingerprint}",
                header.ple_fingerprint
            )));
        }
        if header.rank_payload_sha256 != payload_sha256 {
            return Err(fail(format!(
                "cut from rank {} but the sidecar is {payload_sha256}",
                header.rank_payload_sha256
            )));
        }
        if header.shard_offsets != shard_offsets {
            return Err(fail(
                "PLE shard offsets differ from the model's".to_string(),
            ));
        }
        if header.model_file_bytes != model_file_bytes {
            return Err(fail(format!(
                "cut from a {} B model file, this one is {model_file_bytes} B",
                header.model_file_bytes
            )));
        }

        let group = header.tier_group;
        let kept = header.kept_buckets;
        let index = build_tier_index(&buckets, group, kept);
        if index.tier_rows != header.tier_rows {
            return Err(fail(
                "per-tier row counts differ from the sidecar's".to_string(),
            ));
        }
        let TierIndex {
            tier_count,
            tier_of,
            prefix,
            tier_rows,
        } = index;
        let tier_base = tier_bases(header.data_offset, &tier_rows, encoded_row_bytes as u64);
        let row_width = usize::try_from(header.row_width)
            .map_err(|_| fail(format!("row_width {} overflows", header.row_width)))?;
        let pack = File::open(pack_path).map_err(|e| fail(format!("open: {e}")))?;
        fadvise(&pack, libc::POSIX_FADV_RANDOM);
        let index_bytes =
            buckets.len() as u64 + prefix.iter().map(|c| c.len() as u64 * 4).sum::<u64>();
        eprintln!(
            "{TAG}: pack {}: kept {kept}/{BUCKETS} buckets in {tier_count} tiers ({:.2} GiB, index \
             {:.0} MiB)",
            pack_path.display(),
            gib(header.data_bytes),
            mib(index_bytes),
        );
        Ok(std::sync::Arc::new(HotRowStore {
            padded_rows,
            row_width,
            rows_per_shard,
            encoding,
            encoded_row_bytes,
            group,
            kept,
            tier_count,
            tier_of,
            bucket: buckets.into_boxed_slice(),
            prefix: prefix.into_iter().map(Vec::into_boxed_slice).collect(),
            tier_base,
            pack,
            pack_path: pack_path.to_path_buf(),
            data_offset: header.data_offset,
            data_bytes: header.data_bytes,
            expected_sha: payload_sha256,
            state: AtomicU8::new(STATE_VERIFYING),
            stop: AtomicBool::new(false),
            hot_rows: AtomicU64::new(0),
            hot_pages: AtomicU64::new(0),
            hot_reads: AtomicU64::new(0),
            prewarm_budget: AtomicU64::new(BUDGET_UNSET),
            prewarmed: AtomicU64::new(0),
            prewarm_done: AtomicBool::new(false),
            load_filled: AtomicU64::new(0),
            retouched: AtomicU64::new(0),
            activity: OnceLock::new(),
            #[cfg(test)]
            nowait_policy: OnceLock::new(),
        }))
    }

    /// Unsupported outside Unix: errs before opening or reading anything.
    #[cfg(not(unix))]
    pub fn open(
        _rank: PleHotRank,
        pack_path: &Path,
        _rows_per_shard: usize,
        _encoding: RowEncoding,
        _encoded_row_bytes: usize,
        _shard_offsets: &[u64],
        _model_file_bytes: u64,
    ) -> Result<std::sync::Arc<HotRowStore>, String> {
        Err(format!(
            "ple hot pack {}: {}",
            pack_path.display(),
            unsupported("opening a hot-row pack")
        ))
    }

    #[cfg(unix)]
    /// Open a rank-ordered PLE table (a `ple-table` of `order rank`, the
    /// `<model>.ple-q4` derived with a rank): every row of the PLE table in
    /// tier order, with the rank embedded.  Reads the header and the rank
    /// (its sha256 must be the header's `rank_payload_sha256`), builds the same
    /// row -> pack-offset index as [`HotRowStore::open`], and **serves at
    /// once**: there is no model file to spot-check against, and the table's
    /// own digests are the sidecar loader's business.  Every row has a pack
    /// offset.  Errs on a hot pack or a row-ordered table, and on a rank that
    /// is not the table's.
    pub fn open_table(path: &Path) -> Result<std::sync::Arc<HotRowStore>, String> {
        let fail = |what: String| format!("ple table {}: {what}", path.display());
        let header = read_ple_hot_pack_header(path)?;
        if header.kind != PackKind::PleTable {
            return Err(fail(
                "is a hot pack, not a PLE table (open it with HotRowStore::open)".to_string(),
            ));
        }
        if header.order != PackOrder::Rank {
            return Err(fail(
                "is row-ordered: read it through the row store's shard descriptors".to_string(),
            ));
        }
        let padded_rows = header.padded_rows;
        if padded_rows > u64::from(u32::MAX) {
            return Err(fail(format!("{padded_rows} rows exceed the index range")));
        }
        let to_usize = |name: &str, value: u64| {
            usize::try_from(value).map_err(|_| fail(format!("`{name}` {value} overflows")))
        };
        let rows_per_shard = to_usize("rows_per_shard", header.rows_per_shard)?;
        let row_width = to_usize("row_width", header.row_width)?;
        let encoded_row_bytes = to_usize("encoded_row_bytes", header.encoded_row_bytes)?;
        let pack = File::open(path).map_err(|e| fail(format!("open: {e}")))?;
        let mut buckets = vec![0u8; to_usize("padded_rows", padded_rows)?];
        read_exact_at(&pack, header.rank_offset, &mut buckets)
            .map_err(|e| fail(format!("reading the embedded rank: {e}")))?;
        let actual = hex(&Sha256::digest(&buckets));
        if actual != header.rank_payload_sha256 {
            return Err(fail(format!(
                "embedded rank sha256 {actual} != header {}",
                header.rank_payload_sha256
            )));
        }
        let group = header.tier_group;
        let kept = header.kept_buckets;
        let index = build_tier_index(&buckets, group, kept);
        if index.tier_rows != header.tier_rows {
            return Err(fail(
                "per-tier row counts differ from the embedded rank's".to_string(),
            ));
        }
        let TierIndex {
            tier_count,
            tier_of,
            prefix,
            tier_rows,
        } = index;
        let tier_base = tier_bases(header.data_offset, &tier_rows, encoded_row_bytes as u64);
        fadvise(&pack, libc::POSIX_FADV_RANDOM);
        let index_bytes =
            buckets.len() as u64 + prefix.iter().map(|c| c.len() as u64 * 4).sum::<u64>();
        eprintln!(
            "{TAG}: table {}: {} rows in {tier_count} tiers ({:.2} GiB, index {:.0} MiB); serving",
            path.display(),
            padded_rows,
            gib(header.data_bytes),
            mib(index_bytes),
        );
        Ok(std::sync::Arc::new(HotRowStore {
            padded_rows,
            row_width,
            rows_per_shard,
            encoding: header.row_encoding,
            encoded_row_bytes,
            group,
            kept,
            tier_count,
            tier_of,
            bucket: buckets.into_boxed_slice(),
            prefix: prefix.into_iter().map(Vec::into_boxed_slice).collect(),
            tier_base,
            pack,
            pack_path: path.to_path_buf(),
            data_offset: header.data_offset,
            data_bytes: header.data_bytes,
            expected_sha: header.rank_payload_sha256,
            state: AtomicU8::new(STATE_SERVING),
            stop: AtomicBool::new(false),
            hot_rows: AtomicU64::new(0),
            hot_pages: AtomicU64::new(0),
            hot_reads: AtomicU64::new(0),
            prewarm_budget: AtomicU64::new(BUDGET_UNSET),
            prewarmed: AtomicU64::new(0),
            prewarm_done: AtomicBool::new(false),
            load_filled: AtomicU64::new(0),
            retouched: AtomicU64::new(0),
            activity: OnceLock::new(),
            #[cfg(test)]
            nowait_policy: OnceLock::new(),
        }))
    }

    /// Unsupported outside Unix: errs before opening or reading anything.
    #[cfg(not(unix))]
    pub fn open_table(path: &Path) -> Result<std::sync::Arc<HotRowStore>, String> {
        Err(format!(
            "ple table {}: {}",
            path.display(),
            unsupported("opening a rank-ordered PLE table")
        ))
    }

    /// Values per row (the row store's `row_width`).
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    /// Tiers in the pack.
    pub fn tier_count(&self) -> usize {
        self.tier_count
    }

    /// Buckets (of 256) in the pack.
    pub fn kept_buckets(&self) -> usize {
        self.kept
    }

    pub fn padded_rows(&self) -> u64 {
        self.padded_rows
    }

    pub fn rows_per_shard(&self) -> usize {
        self.rows_per_shard
    }

    pub fn encoded_row_bytes(&self) -> usize {
        self.encoded_row_bytes
    }

    /// Row encoding of the pack (equal to the row store's).
    pub fn encoding(&self) -> RowEncoding {
        self.encoding
    }

    /// Row bytes in the pack.
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    /// The pack passed verification and serves rows.
    pub fn is_serving(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_SERVING
    }

    /// `None` while the worker is still verifying, then whether it passed.
    pub fn verified(&self) -> Option<bool> {
        match self.state.load(Ordering::Acquire) {
            STATE_SERVING => Some(true),
            STATE_FAILED => Some(false),
            _ => None,
        }
    }

    /// Bytes of the pack read by the prewarm so far.
    pub fn prewarmed_bytes(&self) -> u64 {
        self.prewarmed.load(Ordering::Relaxed)
    }

    /// Bytes of the data the [`LoadFill`] read while the model loaded.
    pub fn load_filled_bytes(&self) -> u64 {
        self.load_filled.load(Ordering::Relaxed)
    }

    /// The prewarm read its budget (phase 1) and re-touched tier 0 (phase 2).
    pub fn prewarm_complete(&self) -> bool {
        self.prewarm_done.load(Ordering::Acquire)
    }

    /// Bytes of tier 0 the prewarm's second phase has re-read so far.
    pub fn retouched_bytes(&self) -> u64 {
        self.retouched.load(Ordering::Relaxed)
    }

    /// Rows served from the pack.
    pub fn hot_rows(&self) -> u64 {
        self.hot_rows.load(Ordering::Relaxed)
    }

    /// Pages whose every requested row the pack served.
    pub fn hot_pages(&self) -> u64 {
        self.hot_pages.load(Ordering::Relaxed)
    }

    /// Reads issued against the pack (a read covers a run of consecutive
    /// pack rows).
    pub fn hot_reads(&self) -> u64 {
        self.hot_reads.load(Ordering::Relaxed)
    }

    pub(crate) fn note_served(&self, rows: u64, pages: u64, reads: u64) {
        self.hot_rows.fetch_add(rows, Ordering::Relaxed);
        self.hot_pages.fetch_add(pages, Ordering::Relaxed);
        self.hot_reads.fetch_add(reads, Ordering::Relaxed);
    }

    /// Use `activity` (the row store's [`FetchActivity`]) to pause the
    /// prewarm: it waits while a request is being served or a PLE fetch (a
    /// prefill chunk or a decode step) is in flight, and for [`BUSY_QUIET`]
    /// after the last guard dropped, so it never
    /// competes with a request for the SSD.  Binding the same activity again
    /// is a no-op; another one errs.  Until bound, nothing pauses the
    /// prewarm.
    pub(crate) fn bind_activity(
        &self,
        activity: std::sync::Arc<FetchActivity>,
    ) -> Result<(), String> {
        let bound = self.activity.get_or_init(|| activity.clone());
        if std::sync::Arc::ptr_eq(bound, &activity) {
            Ok(())
        } else {
            Err("the hot store is already bound to another row store".to_string())
        }
    }

    /// Sleep while a PLE fetch is in flight or finished less than
    /// [`BUSY_QUIET`] ago; `false` when the store was stopped meanwhile.
    fn wait_until_idle(&self) -> bool {
        match self.activity.get() {
            Some(activity) => activity.wait_idle(BUSY_QUIET, &self.stop),
            None => !self.stopped(),
        }
    }

    /// Ask the worker to stop at its next read; pair with joining the handle
    /// [`HotRowStore::spawn_worker`] returned.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// Absolute pack offset of global `row`'s encoded bytes when the pack
    /// holds it and serves.
    pub fn pack_offset(&self, row: u64) -> Option<u64> {
        if row >= self.padded_rows || !self.is_serving() {
            return None;
        }
        let tier = self.tier_of[usize::from(self.bucket[row as usize])];
        if tier == NO_TIER {
            return None;
        }
        let tier = usize::from(tier);
        Some(self.tier_base[tier] + self.slot(row, tier) * self.encoded_row_bytes as u64)
    }

    /// Read `dst.len()` bytes of the pack at absolute `offset`.
    pub fn read_pack(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        self.check_pack_range(offset, dst.len())?;
        read_exact_at(&self.pack, offset, dst)
    }

    /// [`HotRowStore::read_pack`] that only completes from the page cache
    /// (`RWF_NOWAIT`, see [`read_nowait_at`]): the same range check, then
    /// `Done` with `dst` filled or `WouldBlock` with `dst` unspecified.
    pub(crate) fn read_pack_nowait(
        &self,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<NowaitRead, SourceError> {
        self.check_pack_range(offset, dst.len())?;
        #[cfg(test)]
        if let Some(policy) = self.nowait_policy.get() {
            return if policy(offset) {
                read_exact_at(&self.pack, offset, dst).map(|()| NowaitRead::Done)
            } else {
                dst.fill(0xA5);
                Ok(NowaitRead::WouldBlock)
            };
        }
        read_nowait_at(&self.pack, offset, dst).map_err(|source| SourceError::Io { offset, source })
    }

    fn check_pack_range(&self, offset: u64, len: usize) -> Result<(), SourceError> {
        let end = offset
            .checked_add(len as u64)
            .ok_or(SourceError::Overflow {
                offset,
                length: len as u64,
            })?;
        if offset < self.data_offset || end > self.data_offset + self.data_bytes {
            return Err(SourceError::RangeOutOfBounds {
                offset,
                length: len as u64,
                source_len: self.data_offset + self.data_bytes,
            });
        }
        Ok(())
    }

    /// Install the test hook of [`HotRowStore::read_pack_nowait`] (once).
    #[cfg(test)]
    pub(crate) fn set_nowait_policy(&self, policy: NowaitPolicy) {
        assert!(
            self.nowait_policy.set(policy).is_ok(),
            "nowait policy already installed"
        );
    }

    /// Index of `row` inside tier `tier` (rows of the tier in ascending global
    /// row order): the prefix count of its 512-row block plus the tier's rows
    /// earlier in the block.
    pub(crate) fn slot(&self, row: u64, tier: usize) -> u64 {
        let row = row as usize;
        let block = row / BLOCK_ROWS;
        let lo = (tier * self.group) as u16;
        let hi = ((tier + 1) * self.group).min(self.kept) as u16;
        let before = self.bucket[block * BLOCK_ROWS..row]
            .iter()
            .filter(|&&b| (lo..hi).contains(&u16::from(b)))
            .count();
        u64::from(self.prefix[tier][block]) + before as u64
    }

    /// Bytes of the pack to prewarm once the model is ready (capped at the
    /// pack's size; 0 = none).  The worker starts reading at once if it has
    /// verified the pack, else right after.
    pub fn set_prewarm_budget(&self, bytes: u64) {
        self.prewarm_budget
            .store(bytes.min(self.data_bytes), Ordering::Release);
    }

    #[cfg(unix)]
    /// Start the worker: check the sidecar payload hash and spot-check
    /// [`VERIFY_SAMPLES`] kept rows of the pack against the model file
    /// (`shard_offsets[s]` = file offset of PLE shard `s`), then serve; once
    /// [`HotRowStore::set_prewarm_budget`] was called, read that many bytes of
    /// the pack's hottest tiers sequentially into the page cache while no
    /// request is being served.  Call [`HotRowStore::stop`] then join it
    /// before unloading the row store.
    pub fn spawn_worker(
        self: &std::sync::Arc<Self>,
        model_path: &Path,
        shard_offsets: Vec<u64>,
    ) -> JoinHandle<()> {
        let store = self.clone();
        let model_path = model_path.to_path_buf();
        std::thread::Builder::new()
            .name("ple-hot-pack".to_string())
            .spawn(move || store.run_worker(&model_path, &shard_offsets))
            .expect("spawn ple hot pack thread")
    }

    #[cfg(unix)]
    fn run_worker(&self, model_path: &Path, shard_offsets: &[u64]) {
        match self.verify(model_path, shard_offsets) {
            Ok(summary) => {
                eprintln!("{TAG}: {summary}; serving");
                self.state.store(STATE_SERVING, Ordering::Release);
            }
            Err(error) => {
                eprintln!("{TAG}: {error}; the pack is not used");
                self.state.store(STATE_FAILED, Ordering::Release);
                return;
            }
        }
        let Some(budget) = self.wait_for_budget() else {
            return;
        };
        self.prewarm(budget, 0);
    }

    #[cfg(unix)]
    /// Start the worker of a table store ([`HotRowStore::open_table`]): it
    /// verifies nothing (the table serves from the start).  With `load_fill`
    /// it first reads the table's hottest prefix while the model loads (see
    /// [`LoadFill`]); once [`HotRowStore::set_prewarm_budget`] was called it
    /// runs the same prewarm as [`HotRowStore::spawn_worker`] over the
    /// table's tiers, from where the fill stopped.  Call
    /// [`HotRowStore::stop`] then join it before unloading the row store.
    pub fn spawn_table_worker(
        self: &std::sync::Arc<Self>,
        load_fill: Option<LoadFill>,
    ) -> JoinHandle<()> {
        let store = self.clone();
        // The load this worker was spawned in: a fill that outlives it must
        // not record into a later load's receipt.
        let load_generation = rdna_compute::load_phases::generation();
        std::thread::Builder::new()
            .name("ple-table".to_string())
            .spawn(move || {
                let filled = load_fill.map_or(0, |fill| store.load_fill(fill, load_generation));
                if let Some(budget) = store.wait_for_budget() {
                    store.prewarm(budget, filled);
                }
            })
            .expect("spawn ple table thread")
    }

    #[cfg(unix)]
    /// Read the first `fill.bytes` of the data (hottest tier first) into the
    /// page cache at no more than `fill.bytes_per_sec`, until the prewarm
    /// budget is set (the model is ready) or the store stops.  The weight
    /// upload streams the model file through the same cache and evicts the
    /// oldest pages first, which are the hottest ones read here, so every
    /// [`LOAD_FILL_REPAIR_PERIOD`] a pass over the prefix read so far
    /// re-reads its evicted pages, hottest first, before the fill reads on.
    /// Once the whole prefix is resident it keeps repairing it until ready.
    /// Returns the resident prefix (from `data_offset`) at the end, where the
    /// prewarm after ready continues.
    fn load_fill(&self, fill: LoadFill, load_generation: u64) -> u64 {
        let target = fill.bytes.min(self.data_bytes);
        if target == 0 || fill.bytes_per_sec == 0 {
            return 0;
        }
        let file = match File::open(&self.pack_path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("{TAG}: load fill: opening the table failed: {error}");
                return 0;
            }
        };
        fadvise(&file, libc::POSIX_FADV_SEQUENTIAL);
        let begin = self.data_offset;
        let end = begin + target;
        // Without a residency view the fill reads the prefix once, in order.
        let mut residency = match Residency::map(&file, end) {
            Ok(residency) => Some(residency),
            Err(error) => {
                eprintln!("{TAG}: load fill: no residency view ({error}); reading the prefix once");
                None
            }
        };
        let chunk = fill.chunk();
        let mut buffer = vec![0u8; chunk];
        let started = Instant::now();
        // Next unread offset; bytes read so far (the rate's count), and how
        // many of them re-read evicted pages.
        let mut cursor = begin;
        let mut read_bytes = 0u64;
        let mut repaired = 0u64;
        // Where the running repair pass looks for the next hole; `None`
        // between passes, the next one starting at `next_pass`.
        let mut scan_from = None;
        let mut next_pass = started;
        let mut ended = "stopped at ready";
        loop {
            if self.stopped() {
                return 0;
            }
            if self.prewarm_budget.load(Ordering::Acquire) != BUDGET_UNSET {
                break;
            }
            let due = Duration::from_secs_f64(read_bytes as f64 / fill.bytes_per_sec as f64);
            let elapsed = started.elapsed();
            if due > elapsed {
                std::thread::sleep((due - elapsed).min(LOAD_FILL_SLICE));
                continue;
            }
            if scan_from.is_none() && residency.is_some() && Instant::now() >= next_pass {
                scan_from = Some(begin);
            }
            let mut at = None;
            if let (Some(view), Some(from)) = (residency.as_mut(), scan_from) {
                match view.first_hole(from, cursor) {
                    Ok(hole) if hole < cursor => {
                        if from == begin {
                            self.load_filled.store(hole - begin, Ordering::Relaxed);
                        }
                        at = Some(hole);
                    }
                    Ok(_) => {
                        if from == begin {
                            self.load_filled.store(cursor - begin, Ordering::Relaxed);
                        }
                        scan_from = None;
                        next_pass = Instant::now() + LOAD_FILL_REPAIR_PERIOD;
                    }
                    Err(error) => {
                        eprintln!("{TAG}: load fill: residency scan failed ({error}); reading on in order");
                        residency = None;
                        scan_from = None;
                    }
                }
            }
            let Some(at) = at.or((cursor < end).then_some(cursor)) else {
                // The whole prefix is resident: wait for the next pass.
                std::thread::sleep(LOAD_FILL_SLICE);
                continue;
            };
            if !self.wait_until_idle() {
                return 0;
            }
            let len = ((end - at) as usize).min(chunk);
            if let Err(error) = read_exact_at(&file, at, &mut buffer[..len]) {
                eprintln!(
                    "{TAG}: load fill: read at {:.2} GiB failed: {error}",
                    gib(at - begin)
                );
                ended = "stopped by a read error";
                break;
            }
            read_bytes += len as u64;
            if at < cursor {
                repaired += len as u64;
                scan_from = Some(at + len as u64);
                // A view that still shows the page just read as a hole cannot
                // see the page cache (`mincore` needs the file's owner or
                // write access): read on in order without it.
                let blind = residency.as_mut().is_some_and(
                    |view| !matches!(view.first_hole(at, at + 1), Ok(hole) if hole > at),
                );
                if blind {
                    eprintln!("{TAG}: load fill: the residency view is blind; reading on in order");
                    residency = None;
                    scan_from = None;
                }
            } else {
                cursor += len as u64;
                if scan_from.is_none() {
                    self.load_filled.store(cursor - begin, Ordering::Relaxed);
                }
            }
        }
        let filled = match residency
            .as_mut()
            .map(|view| view.first_hole(begin, cursor))
        {
            Some(Ok(hole)) => hole,
            _ => cursor,
        } - begin;
        self.load_filled.store(filled, Ordering::Relaxed);
        let secs = started.elapsed().as_secs_f64();
        rdna_compute::load_phases::add_overlapped_for(
            load_generation,
            "ple_load_fill",
            (secs * 1e9) as u64,
            read_bytes,
        );
        eprintln!(
            "{TAG}: load fill {ended}: {:.2} GiB resident from the hottest row on ({:.2} GiB read \
             of {:.2}); {:.2} GiB read in {secs:.1} s, {:.2} of it re-reading evicted pages, {:.2} GB/s \
             (cap {:.2})",
            gib(filled),
            gib(cursor - begin),
            gib(target),
            gib(read_bytes),
            gib(repaired),
            read_bytes as f64 / 1e9 / secs.max(1e-9),
            fill.bytes_per_sec as f64 / 1e9,
        );
        filled
    }

    /// Sleep until a prewarm budget is set; `None` when the store was stopped
    /// first.
    fn wait_for_budget(&self) -> Option<u64> {
        loop {
            if self.stopped() {
                return None;
            }
            match self.prewarm_budget.load(Ordering::Acquire) {
                BUDGET_UNSET => std::thread::sleep(Duration::from_millis(50)),
                budget => return Some(budget),
            }
        }
    }

    #[cfg(unix)]
    /// Sidecar payload hash, then [`VERIFY_SAMPLES`] kept rows compared byte
    /// for byte between the pack and the model file.
    fn verify(&self, model_path: &Path, shard_offsets: &[u64]) -> Result<String, String> {
        let started = Instant::now();
        let actual = hex(&Sha256::digest(&self.bucket));
        if actual != self.expected_sha {
            return Err(format!(
                "sidecar payload sha256 {actual} != {}",
                self.expected_sha
            ));
        }
        let model = File::open(model_path).map_err(|e| {
            format!(
                "opening {} for the spot check failed: {e}",
                model_path.display()
            )
        })?;
        fadvise(&model, libc::POSIX_FADV_RANDOM);
        // Deterministic sample: pseudo-random rows, kept ones only.
        let mut rows = Vec::with_capacity(VERIFY_SAMPLES);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..VERIFY_SAMPLES * 64 {
            if rows.len() == VERIFY_SAMPLES {
                break;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let row = state % self.padded_rows;
            if self.tier_of[usize::from(self.bucket[row as usize])] != NO_TIER {
                rows.push(row);
            }
        }
        let row_bytes = self.encoded_row_bytes;
        let per_thread = rows.len().div_ceil(VERIFY_THREADS).max(1);
        let mismatch = std::thread::scope(|scope| {
            let handles: Vec<_> = rows
                .chunks(per_thread)
                .map(|chunk| {
                    let model = &model;
                    scope.spawn(move || -> Result<(), String> {
                        let mut from_pack = vec![0u8; row_bytes];
                        let mut from_model = vec![0u8; row_bytes];
                        for &row in chunk {
                            let tier =
                                usize::from(self.tier_of[usize::from(self.bucket[row as usize])]);
                            let offset =
                                self.tier_base[tier] + self.slot(row, tier) * row_bytes as u64;
                            read_exact_at(&self.pack, offset, &mut from_pack)
                                .map_err(|e| format!("pack read of row {row}: {e}"))?;
                            let shard = (row / self.rows_per_shard as u64) as usize;
                            let local = row % self.rows_per_shard as u64;
                            read_exact_at(
                                model,
                                shard_offsets[shard] + local * row_bytes as u64,
                                &mut from_model,
                            )
                            .map_err(|e| format!("model read of row {row}: {e}"))?;
                            if from_pack != from_model {
                                return Err(format!(
                                    "row {row} differs between the pack and the model"
                                ));
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("ple hot pack verify thread panicked"))
                .collect::<Result<Vec<()>, String>>()
        });
        mismatch?;
        Ok(format!(
            "sidecar sha256 ok, {} pack rows match the model ({:.0} ms)",
            rows.len(),
            started.elapsed().as_secs_f64() * 1e3
        ))
    }

    #[cfg(unix)]
    /// Prewarm the pack's hottest `budget` bytes into the page cache while no
    /// request is being served.  Reads only the pack (`pack_path`,
    /// `data_offset`, `tier_base`), never the source model file.
    ///
    /// Phase 1 reads the pack's data from its start, hottest tier first, so
    /// early requests hit sooner after load; it skips the first `filled`
    /// bytes, which a [`LoadFill`] already read.  Phase 2 then reads tier 0 (the
    /// first `tier_base[1]` bytes, clipped to the budget) a second time: a
    /// page touched twice is promoted to the kernel's active list, so the
    /// hottest rows stay resident when the cache is reclaimed.  Both phases
    /// pause while a PLE fetch runs ([`HotRowStore::busy`]).
    fn prewarm(&self, budget: u64, filled: u64) {
        if budget == 0 {
            self.prewarm_done.store(true, Ordering::Release);
            return;
        }
        let file = match File::open(&self.pack_path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("{TAG}: prewarm: opening the pack failed: {error}");
                return;
            }
        };
        fadvise(&file, libc::POSIX_FADV_SEQUENTIAL);
        let started = Instant::now();
        let mut paused = Duration::ZERO;
        let mut buffer = vec![0u8; PREWARM_CHUNK];
        let budget = budget.min(self.data_bytes);
        let skipped = filled.min(budget);
        self.prewarmed.store(skipped, Ordering::Relaxed);
        let tier0_end = self
            .tier_base
            .get(1)
            .copied()
            .unwrap_or(self.data_offset + self.data_bytes);
        let (fill, touch) = prewarm_spans(self.data_offset, tier0_end, budget, skipped);
        if !self.read_span(&file, &mut buffer, &mut paused, fill, &self.prewarmed) {
            return;
        }
        let fill_time = started.elapsed();
        if !self.read_span(&file, &mut buffer, &mut paused, touch, &self.retouched) {
            return;
        }
        self.prewarm_done.store(true, Ordering::Release);
        let reading = started.elapsed().saturating_sub(paused).as_secs_f64();
        let retouched = self.retouched.load(Ordering::Relaxed);
        eprintln!(
            "{TAG}: prewarm complete: {:.2} GiB of {:.2} GiB (hottest first; {:.2} GiB of it read \
             during load) in {:.1} s, then tier 0 re-touched ({:.2} GiB, {:.1} s); {:.2} GB/s \
             reading, {:.1} s paused for requests, MemAvailable - CMA {:.1} GiB",
            gib(budget),
            gib(self.data_bytes),
            gib(skipped),
            fill_time.as_secs_f64(),
            gib(retouched),
            (started.elapsed() - fill_time).as_secs_f64(),
            (budget - skipped + retouched) as f64 / 1e9 / reading.max(1e-9),
            paused.as_secs_f64(),
            mem_available_bytes().map_or(f64::NAN, gib),
        );
    }

    #[cfg(unix)]
    /// Read the pack bytes `span` (absolute offsets) in [`PREWARM_CHUNK`]
    /// reads, publishing the bytes of the data read so far (from
    /// `data_offset`) in `progress`, pausing while a
    /// fetch runs.  `false` = stopped by [`HotRowStore::stop`] or a read
    /// failed.
    fn read_span(
        &self,
        file: &File,
        buffer: &mut [u8],
        paused: &mut Duration,
        span: std::ops::Range<u64>,
        progress: &AtomicU64,
    ) -> bool {
        let mut offset = span.start;
        while offset < span.end {
            let idle_from = Instant::now();
            if !self.wait_until_idle() {
                return false;
            }
            *paused += idle_from.elapsed();
            let len = ((span.end - offset) as usize).min(PREWARM_CHUNK);
            if let Err(error) = read_exact_at(file, offset, &mut buffer[..len]) {
                eprintln!(
                    "{TAG}: prewarm stopped at {:.2} GiB: {error}",
                    gib(offset - self.data_offset)
                );
                return false;
            }
            offset += len as u64;
            progress.store(offset - self.data_offset, Ordering::Relaxed);
        }
        true
    }
}

/// The prewarm's two spans (absolute pack offsets): phase 1 the first
/// `budget` bytes of the data from `data_offset`, less the first `skipped`
/// (read during load); phase 2 tier 0 (`data_offset..tier0_end`), clipped to
/// the budget.
fn prewarm_spans(
    data_offset: u64,
    tier0_end: u64,
    budget: u64,
    skipped: u64,
) -> (std::ops::Range<u64>, std::ops::Range<u64>) {
    let end = data_offset + budget;
    (
        data_offset + skipped.min(budget)..end,
        data_offset..tier0_end.min(end),
    )
}

/// A table worker's fill of the table's hottest prefix while the model loads
/// ([`HotRowStore::spawn_table_worker`]): the first `bytes` of the data at
/// no more than `bytes_per_sec`, so it leaves the drive to the weight upload.
/// It runs until the model is ready, re-reading pages of the prefix the
/// cache evicted meanwhile, hottest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadFill {
    pub bytes: u64,
    pub bytes_per_sec: u64,
}

impl LoadFill {
    /// Bytes per read: [`LOAD_FILL_SLICE`] worth of the rate, at least
    /// [`LOAD_FILL_MIN_CHUNK`] and at most [`PREWARM_CHUNK`].
    fn chunk(self) -> usize {
        let slice = (self.bytes_per_sec as f64 * LOAD_FILL_SLICE.as_secs_f64()) as u64;
        usize::try_from(slice)
            .unwrap_or(usize::MAX)
            .clamp(LOAD_FILL_MIN_CHUNK, PREWARM_CHUNK)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    #[cfg(unix)]
    use std::sync::Arc;

    /// A model "file" in memory.
    pub(crate) struct SliceReader(pub Vec<u8>);

    impl ShardRead for SliceReader {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
            let begin = usize::try_from(offset).unwrap().min(self.0.len());
            let n = dst.len().min(self.0.len() - begin);
            dst[..n].copy_from_slice(&self.0[begin..begin + n]);
            Ok(n)
        }
    }

    /// Concatenate shards behind a 13-byte prefix and 7 bytes of padding
    /// between shards, so shard ranges are not sector aligned.
    pub(crate) fn layout_shards(shards: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
        let mut file = vec![0xEEu8; 13];
        let mut offsets = Vec::new();
        for shard in shards {
            offsets.push(file.len() as u64);
            file.extend_from_slice(shard);
            file.extend_from_slice(&[0xEE; 7]);
        }
        (file, offsets)
    }

    /// Deterministic encoded rows: every byte derives from the global row.
    pub(crate) fn synthetic_shards(shards: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        (0..shards)
            .map(|s| {
                let mut bytes = Vec::with_capacity(rows_per_shard * ENCODED_ROW_BYTES);
                for r in 0..rows_per_shard {
                    let global = (s * rows_per_shard + r) as u64;
                    for i in 0..ENCODED_ROW_BYTES as u64 {
                        bytes.push(
                            (global.wrapping_mul(0x9E37_79B9).wrapping_add(i * 31) >> 3) as u8,
                        );
                    }
                }
                bytes
            })
            .collect()
    }

    /// Buckets of a pseudo-random ranking: `bucket = rank * 256 / rows`.
    pub(crate) fn synthetic_buckets(rows: usize, seed: u64) -> Vec<u8> {
        let mut order: Vec<u32> = (0..rows as u32).collect();
        let mut state = seed | 1;
        for i in (1..rows).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            order.swap(i, (state % (i as u64 + 1)) as usize);
        }
        let mut buckets = vec![0u8; rows];
        for (rank, &row) in order.iter().enumerate() {
            buckets[row as usize] = (rank as u64 * BUCKETS as u64 / rows as u64) as u8;
        }
        buckets
    }

    /// Values per row of the test fixtures' Q8F16 rows ([`ENCODED_ROW_BYTES`] bytes).
    pub(crate) const TEST_ROW_WIDTH: usize = 160;

    pub(crate) const FINGERPRINT: &str =
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    #[cfg(unix)]
    /// A model file and its hot-row pack on disk, and the opened store.
    pub(crate) struct PackFixture {
        pub dir: tempfile::TempDir,
        pub model_path: PathBuf,
        pub pack_path: PathBuf,
        pub offsets: Vec<u64>,
        pub buckets: Vec<u8>,
        pub store: Arc<HotRowStore>,
        pub worker: Option<JoinHandle<()>>,
    }

    #[cfg(unix)]
    impl PackFixture {
        /// Write `shards` as a model file, pack its rows of buckets
        /// `0..kept` (tiers of `group`) under a pseudo-random rank, open the
        /// store and wait until its worker verified the pack.
        pub(crate) fn new(
            shards: &[Vec<u8>],
            rows_per_shard: usize,
            kept: usize,
            group: usize,
            seed: u64,
        ) -> Self {
            Self::with_encoding(
                shards,
                rows_per_shard,
                kept,
                group,
                seed,
                RowEncoding::Q8F16,
                TEST_ROW_WIDTH,
            )
        }

        /// [`PackFixture::new`] for shards of `encoding` rows of `row_width`
        /// values (the shard bytes are opaque to the pack).
        pub(crate) fn with_encoding(
            shards: &[Vec<u8>],
            rows_per_shard: usize,
            kept: usize,
            group: usize,
            seed: u64,
            encoding: RowEncoding,
            row_width: usize,
        ) -> Self {
            let row_bytes = crate::ple_hot_pack::checked_row_bytes(encoding, row_width).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let (file, offsets) = layout_shards(shards);
            let model_path = dir.path().join("model.mq4");
            std::fs::write(&model_path, &file).unwrap();
            let rows = shards.len() * rows_per_shard;
            let buckets = synthetic_buckets(rows, seed);
            let rank = PleHotRank::from_buckets(rows as u64, FINGERPRINT, buckets.clone());
            let pack_path = dir.path().join("model.mq4.plehotrows");
            crate::ple_hot_pack::write_pack_with(
                &SliceReader(file.clone()),
                file.len() as u64,
                &offsets,
                rows_per_shard,
                encoding,
                row_width,
                FINGERPRINT,
                &rank,
                kept,
                group,
                &pack_path,
                &mut |_, _| {},
            )
            .unwrap();
            let store = HotRowStore::open(
                rank,
                &pack_path,
                rows_per_shard,
                encoding,
                row_bytes,
                &offsets,
                file.len() as u64,
            )
            .unwrap();
            let worker = store.spawn_worker(&model_path, offsets.clone());
            while store.verified().is_none() {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(store.verified(), Some(true));
            PackFixture {
                dir,
                model_path,
                pack_path,
                offsets,
                buckets,
                store,
                worker: Some(worker),
            }
        }
    }

    #[cfg(unix)]
    impl Drop for PackFixture {
        fn drop(&mut self) {
            self.store.stop();
            if let Some(worker) = self.worker.take() {
                worker.join().unwrap();
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::test_support::*;
    use super::*;

    const ROWS_PER_SHARD: usize = 1000;
    const SHARDS: usize = 7;
    const ROWS: usize = ROWS_PER_SHARD * SHARDS;

    fn model_row(shards: &[Vec<u8>], row: u64) -> &[u8] {
        let shard = row as usize / ROWS_PER_SHARD;
        let local = row as usize % ROWS_PER_SHARD;
        &shards[shard][local * ENCODED_ROW_BYTES..(local + 1) * ENCODED_ROW_BYTES]
    }

    /// Every kept row's pack bytes are the model's bytes; rows of buckets the
    /// pack does not keep have no pack offset.
    #[test]
    fn pack_offsets_return_the_model_rows() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        for (kept, group) in [(100, 5), (256, 4), (7, 2)] {
            let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, kept, group, 41);
            let store = &fixture.store;
            assert_eq!(store.kept_buckets(), kept);
            assert_eq!(store.tier_count(), kept.div_ceil(group));
            let mut held = 0u64;
            let mut bytes = vec![0u8; ENCODED_ROW_BYTES];
            for row in 0..ROWS as u64 {
                match store.pack_offset(row) {
                    Some(offset) => {
                        assert!(usize::from(fixture.buckets[row as usize]) < kept);
                        store.read_pack(offset, &mut bytes).unwrap();
                        assert_eq!(bytes, model_row(&shards, row), "row {row}");
                        held += 1;
                    }
                    None => assert!(usize::from(fixture.buckets[row as usize]) >= kept),
                }
            }
            assert_eq!(held * ENCODED_ROW_BYTES as u64, store.data_bytes());
            assert_eq!(store.pack_offset(ROWS as u64), None);
        }
    }

    /// Reads outside the pack's row region are refused.
    #[test]
    fn read_pack_is_bounded_to_the_rows() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let store = &fixture.store;
        let mut bytes = vec![0u8; ENCODED_ROW_BYTES];
        assert!(
            store.read_pack(0, &mut bytes).is_err(),
            "header bytes are not rows"
        );
        let end = store.data_offset + store.data_bytes();
        assert!(store.read_pack(end - 10, &mut bytes).is_err());
        assert!(store
            .read_pack(end - ENCODED_ROW_BYTES as u64, &mut bytes)
            .is_ok());
    }

    /// `preadv2(RWF_NOWAIT)` completes a read with the file's bytes or reports
    /// `WouldBlock` (a cache miss, or no kernel/filesystem support: the host
    /// decides which); it never returns other bytes and never completes a read
    /// that runs past the end of the file.
    #[test]
    fn read_nowait_at_completes_with_the_file_bytes_or_would_block() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let data: Vec<u8> = (0..20_000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) as u8)
            .collect();
        std::fs::write(&path, &data).unwrap();
        let file = File::open(&path).unwrap();
        for (offset, len) in [(0usize, 1usize), (3, 4093), (4096, 8192), (19_000, 1000)] {
            let mut dst = vec![0xA5u8; len];
            match read_nowait_at(&file, offset as u64, &mut dst).unwrap() {
                NowaitRead::Done => assert_eq!(dst, data[offset..offset + len], "{offset}+{len}"),
                NowaitRead::WouldBlock => {}
            }
        }
        let mut dst = vec![0u8; 100];
        assert_eq!(
            read_nowait_at(&file, 19_950, &mut dst).unwrap(),
            NowaitRead::WouldBlock,
            "50 bytes remain: a short read is not a completed one"
        );
        assert_eq!(
            read_nowait_at(&file, u64::MAX, &mut dst).unwrap(),
            NowaitRead::WouldBlock
        );
        assert_eq!(read_nowait_at(&file, 0, &mut []).unwrap(), NowaitRead::Done);
    }

    /// A failure that is not "no `RWF_NOWAIT` here" is returned, not turned
    /// into a fallback that would hide it (`EBADF`: a descriptor not open for
    /// reading).
    #[cfg(target_os = "linux")]
    #[test]
    fn nowait_probe_returns_real_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, [7u8; 64]).unwrap();
        let write_only = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        let mut dst = [0u8; 16];
        let error = nowait_probe(&write_only, 0, &mut dst).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EBADF));
        let readable = File::open(&path).unwrap();
        assert!(nowait_probe(&readable, 0, &mut dst).is_ok());
    }

    /// `read_pack_nowait` bounds-checks like `read_pack`, and under the test
    /// hook serves (with the pack's bytes) or would block as the policy says.
    #[test]
    fn read_pack_nowait_checks_the_range_and_follows_the_policy() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let store = &fixture.store;
        let mut bytes = vec![0u8; ENCODED_ROW_BYTES];
        assert!(store.read_pack_nowait(0, &mut bytes).is_err());
        let end = store.data_offset + store.data_bytes();
        assert!(store.read_pack_nowait(end - 10, &mut bytes).is_err());

        let serve = std::sync::Arc::new(AtomicBool::new(true));
        let seen = std::sync::Arc::new(AtomicU64::new(u64::MAX));
        {
            let (serve, seen) = (serve.clone(), seen.clone());
            store.set_nowait_policy(std::sync::Arc::new(move |offset: u64| {
                seen.store(offset, Ordering::Release);
                serve.load(Ordering::Acquire)
            }));
        }
        let at = end - ENCODED_ROW_BYTES as u64;
        let mut want = vec![0u8; ENCODED_ROW_BYTES];
        store.read_pack(at, &mut want).unwrap();
        assert_eq!(
            store.read_pack_nowait(at, &mut bytes).unwrap(),
            NowaitRead::Done
        );
        assert_eq!(bytes, want);
        assert_eq!(seen.load(Ordering::Acquire), at);
        serve.store(false, Ordering::Release);
        assert_eq!(
            store.read_pack_nowait(at, &mut bytes).unwrap(),
            NowaitRead::WouldBlock
        );
    }

    /// A pack cut from another rank, another model size, or other shard
    /// offsets is refused at open.
    #[test]
    fn open_refuses_a_pack_from_another_rank_or_model() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let model_bytes = std::fs::metadata(&fixture.model_path).unwrap().len();
        let other_rank =
            || PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, synthetic_buckets(ROWS, 99));
        let same_rank =
            || PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, fixture.buckets.clone());
        let open = |rank, offsets: &[u64], bytes| {
            HotRowStore::open(
                rank,
                &fixture.pack_path,
                ROWS_PER_SHARD,
                RowEncoding::Q8F16,
                ENCODED_ROW_BYTES,
                offsets,
                bytes,
            )
        };
        assert!(open(same_rank(), &fixture.offsets, model_bytes).is_ok());
        assert!(open(other_rank(), &fixture.offsets, model_bytes).is_err());
        assert!(open(same_rank(), &fixture.offsets, model_bytes + 1).is_err());
        let mut moved = fixture.offsets.clone();
        moved[1] += 1;
        assert!(open(same_rank(), &moved, model_bytes).is_err());
    }

    /// A pack whose row encoding or row size differs from the row store's is
    /// refused at open.
    #[test]
    fn open_refuses_a_pack_of_another_row_encoding_or_size() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let model_bytes = std::fs::metadata(&fixture.model_path).unwrap().len();
        let open = |encoding, row_bytes| {
            HotRowStore::open(
                PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, fixture.buckets.clone()),
                &fixture.pack_path,
                ROWS_PER_SHARD,
                encoding,
                row_bytes,
                &fixture.offsets,
                model_bytes,
            )
        };
        let store = open(RowEncoding::Q8F16, ENCODED_ROW_BYTES).unwrap();
        assert_eq!(store.encoding(), RowEncoding::Q8F16);
        assert_eq!(store.encoded_row_bytes(), ENCODED_ROW_BYTES);
        // Right size, wrong encoding.
        let err = open(RowEncoding::Bf16, ENCODED_ROW_BYTES).err().unwrap();
        assert!(err.contains("row store"), "{err}");
        // Right encoding, wrong size.
        let err = open(RowEncoding::Q8F16, 160).err().unwrap();
        assert!(err.contains("row store"), "{err}");
        let err = open(RowEncoding::Q8F16, 0).err().unwrap();
        assert!(err.contains("row store"), "{err}");
        // A BF16 row store of 160 values is 320 B rows: not this pack either.
        assert!(open(RowEncoding::Bf16, 2 * TEST_ROW_WIDTH).is_err());
    }

    /// A pack whose rows were altered after it was written fails the spot
    /// check and never serves.
    #[test]
    fn a_corrupted_pack_fails_verification_and_serves_nothing() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 256, 4, 5);
        // Flip one byte in every row of the pack.
        let header = read_ple_hot_pack_header(&fixture.pack_path).unwrap();
        let mut bytes = std::fs::read(&fixture.pack_path).unwrap();
        for row in 0..(header.data_bytes / ENCODED_ROW_BYTES as u64) {
            bytes[(header.data_offset + row * ENCODED_ROW_BYTES as u64) as usize] ^= 0x5A;
        }
        std::fs::write(&fixture.pack_path, &bytes).unwrap();
        let rank = PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, fixture.buckets.clone());
        let model_bytes = std::fs::metadata(&fixture.model_path).unwrap().len();
        let store = HotRowStore::open(
            rank,
            &fixture.pack_path,
            ROWS_PER_SHARD,
            RowEncoding::Q8F16,
            ENCODED_ROW_BYTES,
            &fixture.offsets,
            model_bytes,
        )
        .unwrap();
        assert!(!store.is_serving());
        assert_eq!(
            store.pack_offset(0),
            None,
            "nothing is served before verification"
        );
        store
            .spawn_worker(&fixture.model_path, fixture.offsets.clone())
            .join()
            .unwrap();
        assert_eq!(store.verified(), Some(false));
        assert!((0..ROWS as u64).all(|row| store.pack_offset(row).is_none()));
    }

    /// The prewarm waits for a budget, pauses while a fetch is in flight,
    /// then reads exactly that many bytes (phase 1, from the start of the
    /// pack's data) and re-reads tier 0 once (phase 2).
    #[test]
    fn prewarm_reads_its_budget_while_idle() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 200, 5, 8);
        let store = &fixture.store;
        let tier0 = store.tier_base[1] - store.data_offset;
        // The row store's activity, as `RowStore::attach_hot_store` binds it.
        let activity = std::sync::Arc::new(FetchActivity::new());
        store.bind_activity(activity.clone()).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(store.prewarmed_bytes(), 0, "no budget, no prewarm");
        let guard = activity.busy();
        let budget = store.data_bytes() / 2;
        assert!(budget > tier0, "the budget spans tier 0 and more");
        store.set_prewarm_budget(budget);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            store.prewarmed_bytes(),
            0,
            "a fetch in flight pauses the prewarm"
        );
        assert_eq!(store.retouched_bytes(), 0);
        drop(guard);
        // Resumes BUSY_QUIET after the fetch; wait it out.
        let deadline = Instant::now() + BUSY_QUIET + Duration::from_secs(10);
        while !store.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(store.prewarm_complete());
        assert_eq!(store.prewarmed_bytes(), budget, "phase 1 reads the budget");
        assert_eq!(store.retouched_bytes(), tier0, "phase 2 reads tier 0");
    }

    /// A whole-request serving guard pauses the prewarm across the compute
    /// gaps between its fetches; a request that fails releases it on every
    /// path, and the prewarm then completes after [`BUSY_QUIET`].
    #[test]
    fn prewarm_pauses_for_a_whole_request_then_completes() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 200, 5, 8);
        let store = &fixture.store;
        let tier0 = store.tier_base[1] - store.data_offset;
        let activity = std::sync::Arc::new(FetchActivity::new());
        store.bind_activity(activity.clone()).unwrap();
        let budget = store.data_bytes() / 2;
        // A request that fails after its fetches; the guard unwinds with it.
        // The budget is set while the request is being served.
        let request = |activity: &std::sync::Arc<FetchActivity>| -> Result<(), String> {
            let _serving = activity.serving();
            store.set_prewarm_budget(budget);
            drop(activity.busy());
            // A compute gap with no fetch in flight.
            std::thread::sleep(Duration::from_millis(250));
            assert_eq!(store.prewarmed_bytes(), 0, "paused across the gap");
            drop(activity.busy());
            Err("request failed".to_string())
        };
        assert!(request(&activity).is_err());
        assert_eq!(activity.in_flight(), 0, "guard released on the error path");
        assert_eq!(store.prewarmed_bytes(), 0, "quiet period not over yet");
        let deadline = Instant::now() + BUSY_QUIET + Duration::from_secs(10);
        while !store.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(store.prewarm_complete());
        assert_eq!(store.prewarmed_bytes(), budget);
        assert_eq!(store.retouched_bytes(), tier0);
    }

    /// A budget inside tier 0 re-reads only that much of it; a budget of 0
    /// reads nothing.
    #[test]
    fn prewarm_retouch_is_clipped_to_the_budget() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 200, 5, 8);
        let store = &fixture.store;
        let tier0 = store.tier_base[1] - store.data_offset;
        let budget = tier0 / 2;
        store.set_prewarm_budget(budget);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !store.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(store.prewarm_complete());
        assert_eq!(store.prewarmed_bytes(), budget);
        assert_eq!(store.retouched_bytes(), budget);

        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 200, 5, 8);
        let store = &fixture.store;
        store.set_prewarm_budget(0);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !store.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(store.prewarm_complete());
        assert_eq!((store.prewarmed_bytes(), store.retouched_bytes()), (0, 0));
    }

    #[test]
    fn prewarm_spans_cover_the_budget_and_tier_0() {
        assert_eq!(
            prewarm_spans(4096, 4096 + 1000, 5000, 0),
            (4096..9096, 4096..5096)
        );
        assert_eq!(
            prewarm_spans(4096, 4096 + 1000, 600, 0),
            (4096..4696, 4096..4696)
        );
        // A load fill's prefix is skipped by phase 1, never by phase 2.
        assert_eq!(
            prewarm_spans(4096, 4096 + 1000, 5000, 3000),
            (7096..9096, 4096..5096)
        );
        assert_eq!(
            prewarm_spans(4096, 4096 + 1000, 600, 3000),
            (4696..4696, 4096..4696)
        );
    }

    #[test]
    fn load_fill_reads_about_a_slice_of_its_rate_per_read() {
        let chunk = |bytes_per_sec| {
            LoadFill {
                bytes: 1,
                bytes_per_sec,
            }
            .chunk()
        };
        assert_eq!(chunk(200_000_000), PREWARM_CHUNK.min(10_000_000));
        assert_eq!(chunk(100_000), 5_000);
        assert_eq!(chunk(1), LOAD_FILL_MIN_CHUNK);
        assert_eq!(chunk(u64::MAX), PREWARM_CHUNK);
    }

    /// An activity binds once; rebinding the same one is fine.
    #[test]
    fn bind_activity_accepts_one_activity() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 32, 4, 2);
        let store = &fixture.store;
        let first = std::sync::Arc::new(FetchActivity::new());
        store.bind_activity(first.clone()).unwrap();
        store.bind_activity(first).unwrap();
        let err = store
            .bind_activity(std::sync::Arc::new(FetchActivity::new()))
            .unwrap_err();
        assert!(err.contains("another row store"), "{err}");
    }

    /// `stop` ends a worker that is still waiting for its budget.
    #[test]
    fn stop_ends_the_worker() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let mut fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 32, 4, 2);
        fixture.store.stop();
        fixture.worker.take().unwrap().join().unwrap();
        assert!(!fixture.store.prewarm_complete());
    }

    // ---- rank-ordered PLE table store --------------------------------------

    use crate::ple_hot_pack::table_support::write_test_table;
    use crate::ple_hot_pack::{read_ple_hot_pack_header, PackOrder};

    /// 90-byte (Q4LM-sized) rows cut from the 170-byte synthetic ones.
    fn q4_sized_shards(shards: &[Vec<u8>]) -> Vec<Vec<u8>> {
        shards
            .iter()
            .map(|shard| {
                shard
                    .chunks_exact(ENCODED_ROW_BYTES)
                    .flat_map(|row| row[..90].iter().copied())
                    .collect()
            })
            .collect()
    }

    fn table_shards(encoding: RowEncoding) -> Vec<Vec<u8>> {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        match encoding {
            RowEncoding::Q4Lm => q4_sized_shards(&shards),
            _ => shards,
        }
    }

    fn write_table(
        dir: &Path,
        order: PackOrder,
        encoding: RowEncoding,
        shards: &[Vec<u8>],
    ) -> PathBuf {
        let path = dir.join(format!(
            "{}-{}.ple-table",
            order.name(),
            encoding.dtype_name()
        ));
        let buckets = synthetic_buckets(ROWS, 31);
        write_test_table(
            &path,
            order,
            encoding,
            TEST_ROW_WIDTH,
            ROWS_PER_SHARD,
            shards,
            (order == PackOrder::Rank).then_some(buckets.as_slice()),
            DEFAULT_TIER_BUCKETS,
        );
        path
    }

    /// A table store serves at once, holds every row at a pack offset, and the
    /// bytes at that offset are the natural-order row's.
    #[test]
    fn table_store_serves_every_row_immediately() {
        let dir = tempfile::tempdir().unwrap();
        for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            let shards = table_shards(encoding);
            let row_bytes = shards[0].len() / ROWS_PER_SHARD;
            let path = write_table(dir.path(), PackOrder::Rank, encoding, &shards);
            let store = HotRowStore::open_table(&path).unwrap();
            assert!(store.is_serving(), "serves before any worker ran");
            assert_eq!(store.verified(), Some(true));
            assert_eq!(store.padded_rows(), ROWS as u64);
            assert_eq!(store.rows_per_shard(), ROWS_PER_SHARD);
            assert_eq!(store.row_width(), TEST_ROW_WIDTH);
            assert_eq!(store.encoding(), encoding);
            assert_eq!(store.encoded_row_bytes(), row_bytes);
            assert_eq!(store.kept_buckets(), BUCKETS);
            assert_eq!(store.tier_count(), BUCKETS.div_ceil(DEFAULT_TIER_BUCKETS));
            assert_eq!(store.data_bytes(), (ROWS * row_bytes) as u64);
            let mut bytes = vec![0u8; row_bytes];
            for row in 0..ROWS as u64 {
                let offset = store.pack_offset(row).expect("every row is in a table");
                store.read_pack(offset, &mut bytes).unwrap();
                let shard = row as usize / ROWS_PER_SHARD;
                let local = row as usize % ROWS_PER_SHARD;
                assert_eq!(
                    bytes,
                    shards[shard][local * row_bytes..(local + 1) * row_bytes],
                    "{encoding:?} row {row}"
                );
            }
            assert_eq!(store.pack_offset(ROWS as u64), None);
        }
    }

    /// A table is never taken for a hot pack and vice versa; a row-ordered
    /// table has no rank to index; a rank that is not the table's is refused.
    #[test]
    fn table_and_hot_pack_openers_refuse_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let shards = table_shards(RowEncoding::Q8F16);
        let table = write_table(dir.path(), PackOrder::Rank, RowEncoding::Q8F16, &shards);

        let rank = PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, synthetic_buckets(ROWS, 31));
        let err = HotRowStore::open(
            rank,
            &table,
            ROWS_PER_SHARD,
            RowEncoding::Q8F16,
            ENCODED_ROW_BYTES,
            &[0u64; SHARDS],
            0,
        )
        .err()
        .unwrap();
        assert!(err.contains("PLE row table"), "{err}");

        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 100, 5, 8);
        let err = HotRowStore::open_table(&fixture.pack_path).err().unwrap();
        assert!(err.contains("hot pack"), "{err}");

        let row_ordered = write_table(dir.path(), PackOrder::Row, RowEncoding::Q8F16, &shards);
        let err = HotRowStore::open_table(&row_ordered).err().unwrap();
        assert!(err.contains("row-ordered"), "{err}");

        let header = read_ple_hot_pack_header(&table).unwrap();
        let mut bytes = std::fs::read(&table).unwrap();
        bytes[header.rank_offset as usize + 3] ^= 0x55;
        let tampered = dir.path().join("tampered.ple-table");
        std::fs::write(&tampered, bytes).unwrap();
        let err = HotRowStore::open_table(&tampered).err().unwrap();
        assert!(err.contains("embedded rank sha256"), "{err}");

        assert!(HotRowStore::open_table(&dir.path().join("absent")).is_err());
    }

    /// The table worker runs the shared prewarm: nothing before a budget, then
    /// exactly the budget (capped at the data size); `stop` ends a worker that
    /// is still waiting.
    #[test]
    fn table_worker_prewarms_under_its_budget() {
        let dir = tempfile::tempdir().unwrap();
        let shards = table_shards(RowEncoding::Q4Lm);
        let path = write_table(dir.path(), PackOrder::Rank, RowEncoding::Q4Lm, &shards);

        let store = HotRowStore::open_table(&path).unwrap();
        let worker = store.spawn_table_worker(None);
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(store.prewarmed_bytes(), 0, "no budget, no prewarm");
        assert!(!store.prewarm_complete());
        let budget = store.data_bytes() / 2;
        store.set_prewarm_budget(budget);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !store.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(store.prewarm_complete());
        assert_eq!(store.prewarmed_bytes(), budget);
        store.stop();
        worker.join().unwrap();
        assert!(
            store.is_serving(),
            "the worker never changes the serving state"
        );

        let capped = HotRowStore::open_table(&path).unwrap();
        let worker = capped.spawn_table_worker(None);
        capped.set_prewarm_budget(u64::MAX - 1);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !capped.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(capped.prewarmed_bytes(), capped.data_bytes());
        capped.stop();
        worker.join().unwrap();

        let waiting = HotRowStore::open_table(&path).unwrap();
        let worker = waiting.spawn_table_worker(None);
        waiting.stop();
        worker.join().unwrap();
        assert!(!waiting.prewarm_complete());
    }

    fn wait_for(mut what: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !what() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A load fill reads its prefix before any budget is set; the prewarm
    /// after ready continues from there and reads the budget in total.
    #[test]
    fn table_worker_load_fill_reads_the_prefix_before_ready() {
        let dir = tempfile::tempdir().unwrap();
        let shards = table_shards(RowEncoding::Q4Lm);
        let path = write_table(dir.path(), PackOrder::Rank, RowEncoding::Q4Lm, &shards);
        let store = HotRowStore::open_table(&path).unwrap();
        let prefix = store.data_bytes() / 3;
        let worker = store.spawn_table_worker(Some(LoadFill {
            bytes: prefix,
            bytes_per_sec: u64::MAX,
        }));
        wait_for(|| store.load_filled_bytes() == prefix);
        assert_eq!(store.load_filled_bytes(), prefix);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(store.prewarmed_bytes(), 0, "no budget, no prewarm");
        assert!(!store.prewarm_complete());
        let budget = store.data_bytes() / 2;
        store.set_prewarm_budget(budget);
        wait_for(|| store.prewarm_complete());
        assert!(store.prewarm_complete());
        assert_eq!(store.prewarmed_bytes(), budget);
        assert_eq!(store.load_filled_bytes(), prefix);
        store.stop();
        worker.join().unwrap();

        // A budget below the filled prefix reads nothing more in phase 1.
        let small = HotRowStore::open_table(&path).unwrap();
        let worker = small.spawn_table_worker(Some(LoadFill {
            bytes: u64::MAX,
            bytes_per_sec: u64::MAX,
        }));
        wait_for(|| small.load_filled_bytes() == small.data_bytes());
        assert_eq!(
            small.load_filled_bytes(),
            small.data_bytes(),
            "capped at the data"
        );
        small.set_prewarm_budget(prefix);
        wait_for(|| small.prewarm_complete());
        assert_eq!(small.prewarmed_bytes(), prefix);
        small.stop();
        worker.join().unwrap();
    }

    /// Pages of the prefix the cache drops during the load are read again.
    /// (Skipped where the file system keeps the pages, e.g. tmpfs.)
    #[test]
    fn table_worker_load_fill_repairs_evicted_pages() {
        let dir = tempfile::tempdir().unwrap();
        let shards = table_shards(RowEncoding::Q4Lm);
        let path = write_table(dir.path(), PackOrder::Rank, RowEncoding::Q4Lm, &shards);
        let file = File::open(&path).unwrap();
        file.sync_all().unwrap();
        let store = HotRowStore::open_table(&path).unwrap();
        let end = store.data_offset + store.data_bytes();
        let mut view = Residency::map(&file, end).unwrap();
        let evict = || {
            // SAFETY: a valid open descriptor; advice only.
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        };
        evict();
        if view.first_hole(0, end).unwrap() == end {
            eprintln!("the file system keeps the pages; nothing to repair");
            return;
        }
        let worker = store.spawn_table_worker(Some(LoadFill {
            bytes: u64::MAX,
            bytes_per_sec: u64::MAX,
        }));
        let full = store.data_bytes();
        wait_for(|| store.load_filled_bytes() == full);
        assert_eq!(store.load_filled_bytes(), full);
        assert_eq!(view.first_hole(store.data_offset, end).unwrap(), end);
        evict();
        // The next pass (within LOAD_FILL_REPAIR_PERIOD) sees the holes and
        // reads them again.
        std::thread::sleep(LOAD_FILL_REPAIR_PERIOD + Duration::from_millis(500));
        wait_for(|| view.first_hole(store.data_offset, end).unwrap() == end);
        assert_eq!(view.first_hole(store.data_offset, end).unwrap(), end);
        store.set_prewarm_budget(0);
        wait_for(|| store.prewarm_complete());
        assert_eq!(store.load_filled_bytes(), full);
        store.stop();
        worker.join().unwrap();
    }

    /// The fill holds its rate, ends when the model is ready (the prewarm
    /// then reads the rest), and `stop` ends a worker mid-fill.
    #[test]
    fn table_worker_load_fill_is_throttled_and_yields_at_ready() {
        let dir = tempfile::tempdir().unwrap();
        let shards = table_shards(RowEncoding::Q4Lm);
        let path = write_table(dir.path(), PackOrder::Rank, RowEncoding::Q4Lm, &shards);
        // 4 KiB reads at 80 KB/s: the whole table would take ~8 s.
        let slow = LoadFill {
            bytes: u64::MAX,
            bytes_per_sec: 80_000,
        };
        let store = HotRowStore::open_table(&path).unwrap();
        let worker = store.spawn_table_worker(Some(slow));
        std::thread::sleep(Duration::from_millis(200));
        store.set_prewarm_budget(store.data_bytes());
        wait_for(|| store.prewarm_complete());
        assert!(store.prewarm_complete());
        let filled = store.load_filled_bytes();
        assert!(filled > 0, "the first read is not delayed");
        assert!(
            filled < store.data_bytes() / 2,
            "{filled} B in ~0.2 s exceeds the rate"
        );
        assert_eq!(store.prewarmed_bytes(), store.data_bytes());
        store.stop();
        worker.join().unwrap();

        let stopped = HotRowStore::open_table(&path).unwrap();
        let worker = stopped.spawn_table_worker(Some(slow));
        std::thread::sleep(Duration::from_millis(100));
        let started = Instant::now();
        stopped.stop();
        worker.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(!stopped.prewarm_complete());
        assert!(stopped.load_filled_bytes() < stopped.data_bytes());
    }

    #[test]
    fn sidecar_round_trip_and_rejections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.plehot");
        let buckets = synthetic_buckets(ROWS, 7);
        write_ple_hot_rank(
            &path,
            ROWS as u64,
            FINGERPRINT,
            &buckets,
            serde_json::json!({"k": 1}),
        )
        .unwrap();
        let rank = PleHotRank::open(&path, ROWS as u64, FINGERPRINT).unwrap();
        assert_eq!(rank.buckets(), buckets.as_slice());
        assert_eq!(rank.header()["rank"]["k"], 1);
        rank.verify().unwrap();
        assert!(PleHotRank::open(&path, ROWS as u64 + 1, FINGERPRINT).is_err());
        let other = FINGERPRINT.replace('0', "f");
        assert!(PleHotRank::open(&path, ROWS as u64, &other).is_err());
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        let tampered = PleHotRank::open(&path, ROWS as u64, FINGERPRINT).unwrap();
        assert!(tampered.verify().is_err());
        bytes.pop();
        std::fs::write(&path, &bytes).unwrap();
        assert!(PleHotRank::open(&path, ROWS as u64, FINGERPRINT).is_err());
    }
}
