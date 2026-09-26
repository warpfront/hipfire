// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded storage for SSD-resident embedding rows that are too large to keep
//! resident (e.g. Qwen4's hashed n-gram per-layer embeddings).
//!
//! `RowStore` deliberately owns the reader worker and its bounded resources.  It
//! does not know how row ids are derived, schedule generation, mutate model
//! state, or perform GPU work.  A caller computes the global row ids it needs,
//! enqueues them with [`RowStore::prefetch`] as early as they are known, waits
//! for the completed [`RowLease`] where the rows are consumed, uploads the
//! contiguous staging bytes, and drops the lease after the device has consumed
//! them.  [`RowFetch`] owns that request-local ticket/lease lifecycle.
//!
//! The production source is a vector of sealed [`SourceRangeDescriptor`]s,
//! one for each physical shard of one `[rows, width]` table.  Reads are
//! positional and descriptor checked; no mmap, source-wide `Vec`, or
//! file-backed GPU pointer is used.  Leases always publish rows as
//! little-endian BF16, whatever tier the shards store.

use crate::model_source::{SourceError, SourceRangeDescriptor};
use rdna_compute::DType;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Userspace cache budget.  This is deliberately fixed rather than a public
/// runtime knob: the model's SSD path must remain bounded on every request.
pub const PAGE_CACHE_BYTES: usize = 256 * 1024 * 1024;
/// Physical rows per userspace page.
///
/// A page is both the cache unit and the read unit, so the row *count* decides
/// how much of an OS page one requested row drags in. Requested rows are drawn
/// at random over the whole 100 GB physical table, so a page read almost never
/// serves any other requested row and the cost of one row is the whole page.
/// Measured on gfx1151/ext4 with a 2 MiB page: a 291-token chunk requested 4656
/// rows in 4483 distinct windows and pulled ~9.4 GB for ~1.5 MB of useful bytes
/// — 492 ms of page-cache copy (1810 ms cold) spent with the GPU idle.
///
/// The device cost of a window is `1 + (window - 1) / 4096` pages, so the row
/// count is a real lever and it pulls two ways: fewer rows per page means fewer
/// pages fetched per row (~1.04 at one row, ~1.17 at four, ~1.5 at twelve,
/// ~2.0 at twenty-four) but more cache entries, and every entry carries index
/// overhead outside the byte budget. Four rows keeps the page inside a fifth of
/// an OS page and the index overhead near a quarter of the payload.
const ROWS_PER_PAGE: usize = 4;
/// Byte budget of one coalesced read or one staging buffer.  The effective
/// size is rounded down to a whole number of decoded rows.
const STAGING_BUDGET_BYTES: usize = 8 * 1024 * 1024;
/// At most two staging buffers exist: one completed output and one read buffer.
pub const MAX_STAGING_BUFFERS: usize = 2;
/// Bounded queue of request ids.  A completed request also occupies one ticket
/// until its lease is consumed or the request is cancelled.
pub const READER_QUEUE_CAPACITY: usize = 64;

/// Stored representation of one physical row.
///
/// The artifact declares this through the shard dtype, so both tiers can be
/// read by one reader: an artifact written before the Q8F16 tier is BF16 rows,
/// and a Q8F16 artifact is five 34-byte blocks per row (`qf16` scale + 32 i8,
/// [`Q8_BLOCK_BYTES`]).  Decoding happens on the reader
/// thread while the requested rows are copied out of a cached page, so the
/// cache keeps the compact form and the device path is untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowEncoding {
    /// Raw little-endian BF16: `2 * width` bytes per row.
    Bf16,
    /// Q8F16: `width / 32` blocks of `[f16 scale][32 x i8]`.
    Q8F16,
}

/// Bytes of one Q8F16 block: an f16 scale followed by 32 signed bytes.
const Q8_BLOCK_BYTES: usize = 34;
/// Weights per Q8F16 block.
const Q8_BLOCK_WEIGHTS: usize = 32;

impl RowEncoding {
    /// The storage dtype of this encoding.
    pub const fn dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::BF16,
            Self::Q8F16 => DType::Q8_0,
        }
    }

    /// Encoded bytes of one physical row of `width` values.
    ///
    /// # Panics
    /// If `width` is zero or, for Q8F16, not a whole number of 32-value blocks.
    pub fn encoded_row_bytes(self, width: usize) -> usize {
        crate::weight_store::external_row_stride(self.dtype(), width)
            .expect("row width is a whole number of blocks")
    }

    /// Bytes of one userspace page of `width`-value rows.
    pub fn page_bytes(self, width: usize) -> usize {
        ROWS_PER_PAGE * self.encoded_row_bytes(width)
    }

    /// The artifact dtype this encoding is declared as.
    pub const fn dtype_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Q8F16 => "Q8_0",
        }
    }

    /// The encoding a shard dtype declares, or `None` when the reader has no
    /// decoder for it.
    ///
    /// The declared dtype is the contract. Nothing infers the tier from the
    /// extent, so a mislabelled shard is refused instead of decoded at the
    /// wrong width.
    pub fn from_dtype(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => Some(Self::Bf16),
            "Q8_0" | "Q8F16" | "Q8" | "Q8_F16" => Some(Self::Q8F16),
            _ => None,
        }
    }

    /// Decode one encoded row into `out`, the row's little-endian BF16 form.
    ///
    /// `out` fixes the row width; `encoded` must be exactly
    /// [`Self::encoded_row_bytes`] of that width, so a short row can never be
    /// silently padded.
    fn decode_row(self, encoded: &[u8], out: &mut [u8]) -> Result<(), RowStoreError> {
        match self {
            Self::Bf16 => {
                if encoded.len() != out.len() {
                    return Err(RowStoreError::InvalidLease {
                        reason: "BF16 row length mismatch".to_string(),
                    });
                }
                out.copy_from_slice(encoded);
            }
            Self::Q8F16 => {
                if encoded.len() != self.encoded_row_bytes(out.len() / 2) {
                    return Err(RowStoreError::InvalidLease {
                        reason: "Q8F16 row length mismatch".to_string(),
                    });
                }
                for (block, chunk) in encoded.chunks_exact(Q8_BLOCK_BYTES).enumerate() {
                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    for index in 0..Q8_BLOCK_WEIGHTS {
                        let value = scale * (chunk[2 + index] as i8) as f32;
                        let cell = (block * Q8_BLOCK_WEIGHTS + index) * 2;
                        out[cell..cell + 2].copy_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
                    }
                }
            }
        }
        Ok(())
    }
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    // The decoder the runtime uses for f16 payloads (half-backed), not a local
    // hand-rolled version: f16 scale decode must be exact, including subnormals.
    crate::llama::f16_to_f32(bits)
}

/// Round-to-nearest-even f32 -> BF16 bits, the narrowing the parity fixture
/// uses (`hipfire-arch-qwen4/examples/qwen4_parity.rs`), so a decoded row is
/// bit-identical to the
/// narrowing the reference does.
#[inline]
fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// A page in the physical shard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct PageKey {
    shard: usize,
    page: usize,
}

/// Checked location of one global row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLocation {
    /// Global row across all shards.
    pub global_row: u64,
    /// Physical shard containing the row.
    pub shard: usize,
    /// Row number within that shard.
    pub local_row: usize,
    /// Page number within that shard.
    pub page: usize,
    /// Byte offset within the page.
    pub page_byte_offset: usize,
}

impl RowLocation {
    fn page_key(self) -> PageKey {
        PageKey {
            shard: self.shard,
            page: self.page,
        }
    }
}

/// Counters and bounded-resource measurements for diagnostics and unload
/// proofs.  These are snapshots; they do not grant access to mutable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCacheStats {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub resident_pages: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub reads: u64,
    pub coalesced_reads: u64,
    pub read_bytes: u64,
    pub evictions: u64,
    pub queue_depth: usize,
    pub outstanding_readers: usize,
    pub outstanding_leases: usize,
    pub staging_in_use: usize,
    pub staging_high_water: usize,
}

/// Proof returned by [`RowStore::quiesce`] and [`RowStore::unload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowQuiesceReport {
    pub queue_depth: usize,
    pub outstanding_readers: usize,
    pub outstanding_leases: usize,
    pub staging_in_use: usize,
    pub cache_bytes: usize,
    pub cache_pages: usize,
}

impl RowQuiesceReport {
    pub const fn is_clean(self) -> bool {
        self.queue_depth == 0
            && self.outstanding_readers == 0
            && self.outstanding_leases == 0
            && self.staging_in_use == 0
    }
}

/// Failure modes of the bounded row reader.  In particular, source read
/// failures are preserved as [`SourceError`] and never become zero rows.
#[derive(Debug)]
pub enum RowStoreError {
    Descriptor {
        index: usize,
        reason: String,
    },
    Source(SourceError),
    GlobalRowOutOfRange {
        row: u64,
        padded_rows: u64,
    },
    PaddingRow {
        row: u64,
        valid_rows: u64,
    },
    RequestTooLarge {
        rows: usize,
        max_rows: usize,
    },
    QueueFull,
    StagingUnavailable,
    Quiescing,
    EpochMismatch {
        requested: u64,
        current: u64,
    },
    EpochNotMonotonic {
        requested: u64,
        current: u64,
    },
    Canceled,
    UnknownTicket(u64),
    AlreadyConsumed(u64),
    WorkerStopped,
    InvalidLease {
        reason: String,
    },
    LeaseStale {
        lease_epoch: u64,
        current_epoch: u64,
    },
    QuiesceTimeout(RowQuiesceReport),
    WorkerJoin,
}

impl fmt::Display for RowStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor { index, reason } => {
                write!(f, "row shard descriptor {index} is invalid: {reason}")
            }
            Self::Source(error) => write!(f, "row source read failed: {error}"),
            Self::GlobalRowOutOfRange { row, padded_rows } => {
                write!(f, "row {row} is outside {padded_rows} physical rows")
            }
            Self::PaddingRow { row, valid_rows } => {
                write!(f, "row {row} is padding (valid rows end at {valid_rows})")
            }
            Self::RequestTooLarge { rows, max_rows } => {
                write!(f, "row prefetch has {rows} rows; maximum is {max_rows}")
            }
            Self::QueueFull => f.write_str("row reader queue is full"),
            Self::StagingUnavailable => f.write_str("row staging buffers are all in use"),
            Self::Quiescing => f.write_str("row store is quiescing or unloaded"),
            Self::EpochMismatch { requested, current } => {
                write!(
                    f,
                    "row request epoch {requested} does not match current epoch {current}"
                )
            }
            Self::EpochNotMonotonic { requested, current } => {
                write!(
                    f,
                    "row store epoch {requested} is not newer than current epoch {current}"
                )
            }
            Self::Canceled => f.write_str("row prefetch was canceled"),
            Self::UnknownTicket(id) => write!(f, "unknown row prefetch ticket {id}"),
            Self::AlreadyConsumed(id) => write!(f, "row prefetch ticket {id} was already consumed"),
            Self::WorkerStopped => f.write_str("row reader worker has stopped"),
            Self::InvalidLease { reason } => write!(f, "invalid row lease: {reason}"),
            Self::LeaseStale {
                lease_epoch,
                current_epoch,
            } => write!(
                f,
                "row lease epoch {lease_epoch} is stale at current epoch {current_epoch}"
            ),
            Self::QuiesceTimeout(report) => write!(f, "row store quiesce timed out: {report:?}"),
            Self::WorkerJoin => f.write_str("row reader worker did not join cleanly"),
        }
    }
}

impl std::error::Error for RowStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SourceError> for RowStoreError {
    fn from(error: SourceError) -> Self {
        Self::Source(error)
    }
}

/// A completed, epoch-bound set of rows in request order.
///
/// The lease owns exactly one of the two fixed staging buffers until dropped.
/// It intentionally exposes a borrowed byte slice rather than moving the
/// buffer out: callers cannot accidentally retain an unbounded third staging
/// allocation past model unload.
pub struct RowLease {
    inner: Arc<RowStoreInner>,
    epoch: u64,
    ids: Vec<u64>,
    bytes: Vec<u8>,
}

impl fmt::Debug for RowLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowLease")
            .field("epoch", &self.epoch)
            .field("rows", &self.ids.len())
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl RowLease {
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn row_count(&self) -> usize {
        self.ids.len()
    }

    /// Global row ids in the order they were requested.
    pub fn row_ids(&self) -> &[u64] {
        &self.ids
    }

    /// Borrow the contiguous `[rows, width]` BF16 staging bytes only
    /// while this lease's epoch is current.  A fallible view is intentional:
    /// returning an unconditional slice would let a reset publish stale rows.
    pub fn as_bytes(&self) -> Result<&[u8], RowStoreError> {
        self.try_bytes()
    }

    pub fn validate(&self) -> Result<(), RowStoreError> {
        self.inner.validate_lease_epoch(self.epoch)
    }

    /// Re-check the reader epoch after the caller's device upload has
    /// completed.  The upload itself is intentionally owned by the caller;
    /// this method closes the second half of the publication boundary so a
    /// reset cannot silently publish stale rows.
    pub fn validate_after_upload(&self) -> Result<(), RowStoreError> {
        self.validate()
    }

    pub fn try_bytes(&self) -> Result<&[u8], RowStoreError> {
        self.validate()?;
        Ok(&self.bytes)
    }

    /// The decoded BF16 bytes of the `index`-th requested row.
    pub fn row_bytes(&self, index: usize) -> Result<&[u8], RowStoreError> {
        self.validate()?;
        if index >= self.ids.len() {
            return Err(RowStoreError::InvalidLease {
                reason: format!("row index {index} of {}", self.ids.len()),
            });
        }
        let row_bytes = self.inner.decoded_row_bytes;
        let begin = index * row_bytes;
        Ok(&self.bytes[begin..begin + row_bytes])
    }

    /// Copy all rows in one contiguous operation into caller-owned upload
    /// storage.  This is the CPU staging boundary; callers must not copy one
    /// row at a time into separate device allocations.  The epoch is checked
    /// both before and after the copy so a reset racing this operation is
    /// reported instead of publishing stale bytes.
    pub fn stage_into(&self, destination: &mut [u8]) -> Result<(), RowStoreError> {
        self.validate()?;
        if destination.len() != self.bytes.len() {
            return Err(RowStoreError::InvalidLease {
                reason: format!(
                    "staging destination has {} bytes, expected {}",
                    destination.len(),
                    self.bytes.len()
                ),
            });
        }
        destination.copy_from_slice(&self.bytes);
        self.validate_after_upload()
    }
}

impl Drop for RowLease {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.bytes);
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        self.inner.return_staging_locked(&mut state, bytes);
        state.active_leases = state.active_leases.saturating_sub(1);
        self.inner.cv.notify_all();
    }
}

/// Handle for a queued row read.  Dropping it invalidates an unfinished or
/// completed result and returns its staging buffer once the reader is safe.
pub struct RowTicket {
    inner: Weak<RowStoreInner>,
    id: u64,
    epoch: u64,
    canceled: Arc<AtomicBool>,
    consumed: AtomicBool,
}

impl fmt::Debug for RowTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowTicket")
            .field("id", &self.id)
            .field("epoch", &self.epoch)
            .field("canceled", &self.canceled.load(Ordering::Acquire))
            .field("consumed", &self.consumed.load(Ordering::Acquire))
            .finish()
    }
}

impl RowTicket {
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn row_ids(&self) -> Result<Vec<u64>, RowStoreError> {
        let inner = self.inner.upgrade().ok_or(RowStoreError::WorkerStopped)?;
        let state = inner.state.lock().expect("row store state mutex poisoned");
        let ticket = state.tickets.get(&self.id).ok_or_else(|| {
            if self.canceled.load(Ordering::Acquire) {
                RowStoreError::Canceled
            } else {
                RowStoreError::UnknownTicket(self.id)
            }
        })?;
        Ok(ticket.ids.clone())
    }

    pub fn cancel(&self) -> Result<(), RowStoreError> {
        if self.consumed.load(Ordering::Acquire) {
            return Err(RowStoreError::AlreadyConsumed(self.id));
        }
        self.canceled.store(true, Ordering::Release);
        let inner = self.inner.upgrade().ok_or(RowStoreError::WorkerStopped)?;
        inner.cancel_ticket(self.id)
    }
}

impl Drop for RowTicket {
    fn drop(&mut self) {
        if !self.consumed.load(Ordering::Acquire) {
            self.canceled.store(true, Ordering::Release);
            if let Some(inner) = self.inner.upgrade() {
                let _ = inner.cancel_ticket(self.id);
            }
        }
    }
}

/// Model-owned bounded row reader/cache.
pub struct RowStore {
    inner: Arc<RowStoreInner>,
    worker: Option<JoinHandle<()>>,
}

impl fmt::Debug for RowStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.inner.report();
        f.debug_struct("RowStore")
            .field("shards", &self.inner.shard_count)
            .field("rows_per_shard", &self.inner.rows_per_shard)
            .field("report", &report)
            .finish()
    }
}

impl RowStore {
    /// Open production row storage from sealed shard descriptors.
    ///
    /// All descriptors must refer to one immutable source identity, share one
    /// `[rows_per_shard, width]` shape and dtype, either BF16 (`2 * width`-byte
    /// rows) or Q8F16 (`width / 32` 34-byte blocks), and be exactly
    /// `rows * encoded_row_bytes` bytes.  Global rows at or past `valid_rows`
    /// are padding: requests for them are refused, so padding can never be
    /// returned as an embedding.  `name` labels the reader thread.
    pub fn new(
        name: &str,
        descriptors: Vec<SourceRangeDescriptor>,
        valid_rows: u64,
    ) -> Result<Self, RowStoreError> {
        let descriptors: Arc<[SourceRangeDescriptor]> = descriptors.into();
        let layout = validate_descriptors(&descriptors, valid_rows)?;
        let source: Arc<dyn PositionalRowSource> = Arc::new(DescriptorRowSource {
            descriptors: descriptors.clone(),
        });
        Self::spawn(name, source, descriptors, layout)
    }

    fn spawn(
        name: &str,
        source: Arc<dyn PositionalRowSource>,
        descriptors: Arc<[SourceRangeDescriptor]>,
        layout: Layout,
    ) -> Result<Self, RowStoreError> {
        let decoded_row_bytes = layout.row_width * 2;
        let staging_bytes = (STAGING_BUDGET_BYTES / decoded_row_bytes) * decoded_row_bytes;
        let inner = Arc::new(RowStoreInner {
            source,
            stopped: AtomicBool::new(false),
            shard_count: layout.shard_count,
            _descriptors: descriptors,
            valid_rows: layout.valid_rows,
            padded_rows: layout.shard_count as u64 * layout.rows_per_shard as u64,
            encoding: layout.encoding,
            encoded_row_bytes: layout.encoding.encoded_row_bytes(layout.row_width),
            decoded_row_bytes,
            staging_bytes,
            rows_per_shard: layout.rows_per_shard,
            rows_per_page: layout.rows_per_page,
            current_epoch: AtomicU64::new(0),
            epoch_started: AtomicBool::new(false),
            state: Mutex::new(RowStoreState::new(staging_bytes)),
            cv: Condvar::new(),
        });
        let weak = Arc::downgrade(&inner);
        let worker = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || worker_loop(weak))
            .map_err(|error| RowStoreError::Descriptor {
                index: 0,
                reason: format!("cannot start bounded reader: {error}"),
            })?;
        Ok(Self {
            inner,
            worker: Some(worker),
        })
    }

    pub fn shard_count(&self) -> usize {
        self.inner.shard_count
    }

    pub fn rows_per_shard(&self) -> usize {
        self.inner.rows_per_shard
    }

    pub const fn rows_per_page(&self) -> usize {
        ROWS_PER_PAGE
    }

    /// Decoded BF16 bytes of one row, as leases publish it.
    pub fn decoded_row_bytes(&self) -> usize {
        self.inner.decoded_row_bytes
    }

    /// Most rows one prefetch may request: one staging buffer's worth.
    pub fn max_rows_per_prefetch(&self) -> usize {
        self.inner.staging_bytes / self.inner.decoded_row_bytes
    }

    pub fn current_epoch(&self) -> u64 {
        self.inner.current_epoch.load(Ordering::Acquire)
    }

    /// Start a model/request epoch and invalidate every prior ticket.
    ///
    /// Epochs are single-use and strictly increasing after the first epoch.
    /// The first caller may choose epoch zero.  Invalidated queued tickets are
    /// removed immediately; an in-flight source read is allowed to finish
    /// under cancellation but can never publish a lease.
    pub fn begin_epoch(&self, epoch: u64) -> Result<(), RowStoreError> {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(RowStoreError::WorkerStopped);
        }
        let current = self.current_epoch();
        let started = self.inner.epoch_started.load(Ordering::Acquire);
        if started && epoch <= current {
            return Err(RowStoreError::EpochNotMonotonic {
                requested: epoch,
                current,
            });
        }
        self.inner.current_epoch.store(epoch, Ordering::Release);
        self.inner.epoch_started.store(true, Ordering::Release);
        invalidate_tickets_locked(&self.inner, &mut state);
        self.inner.cv.notify_all();
        Ok(())
    }

    /// Invalidate the current request epoch, drain all queued/readers/leases,
    /// then resume the same model-owned cache for the next epoch.  Immutable
    /// cached pages survive; request tickets and output buffers do not.
    pub fn reset_epoch(&self, timeout: Duration) -> Result<u64, RowStoreError> {
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(RowStoreError::WorkerStopped);
        }
        let next = {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            let current = self.current_epoch();
            let next = current
                .checked_add(1)
                .ok_or(RowStoreError::EpochNotMonotonic {
                    requested: u64::MAX,
                    current: u64::MAX,
                })?;
            let started = self.inner.epoch_started.load(Ordering::Acquire);
            if started && next <= current {
                return Err(RowStoreError::EpochNotMonotonic {
                    requested: next,
                    current,
                });
            }
            self.inner.current_epoch.store(next, Ordering::Release);
            self.inner.epoch_started.store(true, Ordering::Release);
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
            next
        };
        self.quiesce(timeout)?;
        self.resume()?;
        Ok(next)
    }

    /// Enqueue a bounded prefetch of `row_ids`, in the order the lease will
    /// publish them.  Every id is checked against the valid rows before the
    /// ticket exists.
    pub fn prefetch(&self, epoch: u64, row_ids: Vec<u64>) -> Result<RowTicket, RowStoreError> {
        let max_rows = self.max_rows_per_prefetch();
        if row_ids.len() > max_rows {
            return Err(RowStoreError::RequestTooLarge {
                rows: row_ids.len(),
                max_rows,
            });
        }
        let current = self.current_epoch();
        if epoch != current {
            return Err(RowStoreError::EpochMismatch {
                requested: epoch,
                current,
            });
        }
        let (locations, pages) = self.plan_rows(&row_ids)?;
        let ids = row_ids;
        let required_bytes = locations.len() * self.inner.decoded_row_bytes;

        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        if !state.accepting || state.stop {
            return Err(RowStoreError::Quiescing);
        }
        let current = self.current_epoch();
        if epoch != current {
            return Err(RowStoreError::EpochMismatch {
                requested: epoch,
                current,
            });
        }
        if state.tickets.len() >= READER_QUEUE_CAPACITY {
            return Err(RowStoreError::QueueFull);
        }
        let id = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        let canceled = Arc::new(AtomicBool::new(false));
        state.tickets.insert(
            id,
            TicketState {
                epoch,
                canceled: canceled.clone(),
                ids,
                locations,
                pages,
                required_bytes,
                output: None,
                status: TicketStatus::Pending,
            },
        );
        state.queue.push_back(id);
        self.inner.cv.notify_one();
        Ok(RowTicket {
            inner: Arc::downgrade(&self.inner),
            id,
            epoch,
            canceled,
            consumed: AtomicBool::new(false),
        })
    }

    /// Wait for a ticket's completed rows.  This is the layer-1 consumption
    /// boundary; all source reads and cache fills are complete on return.
    pub fn wait_completed_lease(&self, ticket: &RowTicket) -> Result<RowLease, RowStoreError> {
        let inner = ticket.inner.upgrade().ok_or(RowStoreError::WorkerStopped)?;
        if !Arc::ptr_eq(&inner, &self.inner) {
            return Err(RowStoreError::InvalidLease {
                reason: "ticket belongs to a different row store".to_string(),
            });
        }
        if ticket
            .consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(RowStoreError::AlreadyConsumed(ticket.id));
        }
        let mut state = inner.state.lock().expect("row store state mutex poisoned");
        loop {
            let done = match state.tickets.get(&ticket.id) {
                Some(ticket_state) => !matches!(ticket_state.status, TicketStatus::Pending),
                None => {
                    return Err(if ticket.canceled.load(Ordering::Acquire) {
                        RowStoreError::Canceled
                    } else {
                        RowStoreError::UnknownTicket(ticket.id)
                    })
                }
            };
            if done {
                break;
            }
            if state.stop && state.active_readers == 0 {
                return Err(RowStoreError::WorkerStopped);
            }
            state = inner
                .cv
                .wait(state)
                .expect("row store state mutex poisoned");
        }
        let ticket_state = state
            .tickets
            .remove(&ticket.id)
            .ok_or(RowStoreError::UnknownTicket(ticket.id))?;
        let ticket_canceled = ticket_state.canceled.load(Ordering::Acquire)
            || ticket_state.epoch != inner.current_epoch.load(Ordering::Acquire);
        match ticket_state.status {
            TicketStatus::Pending => unreachable!("completed wait observed pending ticket"),
            TicketStatus::Completed(Ok(())) if !ticket_canceled => {
                let bytes = ticket_state
                    .output
                    .ok_or_else(|| RowStoreError::InvalidLease {
                        reason: "completed ticket has no staging buffer".to_string(),
                    })?;
                state.active_leases += 1;
                let lease_inner = Arc::clone(&inner);
                drop(state);
                Ok(RowLease {
                    inner: lease_inner,
                    epoch: ticket_state.epoch,
                    ids: ticket_state.ids,
                    bytes,
                })
            }
            TicketStatus::Completed(Ok(())) => {
                if let Some(bytes) = ticket_state.output {
                    inner.return_staging_locked(&mut state, bytes);
                }
                inner.cv.notify_all();
                Err(RowStoreError::Canceled)
            }
            TicketStatus::Completed(Err(error)) => {
                if let Some(bytes) = ticket_state.output {
                    inner.return_staging_locked(&mut state, bytes);
                }
                inner.cv.notify_all();
                Err(error)
            }
        }
    }

    pub fn cancel(&self, ticket: &RowTicket) -> Result<(), RowStoreError> {
        ticket.cancel()
    }
    /// Stop accepting tickets, invalidate pending work, and wait for readers
    /// and leases to drain.  A timeout returns an unload proof showing exactly
    /// which owner is still live; it never pretends quiescence succeeded.
    pub fn quiesce(&self, timeout: Duration) -> Result<RowQuiesceReport, RowStoreError> {
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
        }
        let deadline = Instant::now() + timeout;
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        loop {
            if state.queue.is_empty() && state.active_readers == 0 && state.active_leases == 0 {
                purge_tickets_locked(&self.inner, &mut state);
                let report = self.inner.report_locked(&state);
                self.inner.cv.notify_all();
                return Ok(report);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(RowStoreError::QuiesceTimeout(
                    self.inner.report_locked(&state),
                ));
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next, wait) = self
                .inner
                .cv
                .wait_timeout(state, remaining)
                .expect("row store state mutex poisoned");
            state = next;
            if wait.timed_out() {
                return Err(RowStoreError::QuiesceTimeout(
                    self.inner.report_locked(&state),
                ));
            }
        }
    }

    fn quiesce_blocking(&self) -> Result<RowQuiesceReport, RowStoreError> {
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
        }
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        loop {
            if state.queue.is_empty() && state.active_readers == 0 && state.active_leases == 0 {
                purge_tickets_locked(&self.inner, &mut state);
                let report = self.inner.report_locked(&state);
                self.inner.cv.notify_all();
                return Ok(report);
            }
            state = self
                .inner
                .cv
                .wait(state)
                .expect("row store state mutex poisoned");
        }
    }

    pub fn resume(&self) -> Result<(), RowStoreError> {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        if state.stop || self.inner.stopped.load(Ordering::Acquire) {
            return Err(RowStoreError::WorkerStopped);
        }
        state.accepting = true;
        self.inner.cv.notify_all();
        Ok(())
    }

    /// Quiesce and join the model-owned reader with explicit blocking
    /// semantics.  Regular-file positional reads cannot be safely canceled,
    /// so this API never returns a timeout while a worker can still outlive
    /// its source owner.
    pub fn unload(mut self) -> Result<RowQuiesceReport, RowStoreError> {
        let report = self.quiesce_blocking()?;
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.current_epoch.fetch_add(1, Ordering::AcqRel);
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            state.stop = true;
            self.inner.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| RowStoreError::WorkerJoin)?;
        }
        Ok(report)
    }

    pub fn cache_stats(&self) -> RowCacheStats {
        self.inner.stats()
    }

    /// Drop all immutable userspace pages.  The source reader intentionally
    /// has no mmap or fd-pinning contract; dropping these pages is therefore a
    /// portable best-effort eviction of this cache, while the counters make
    /// kernel page-cache growth observable to the outer admission harness.
    pub fn evict_cached_pages(&self) -> usize {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        let dropped = state.cache.clear();
        self.inner.cv.notify_all();
        dropped
    }

    pub fn locate_row(&self, row: u64) -> Result<RowLocation, RowStoreError> {
        self.inner.locate_row(row)
    }

    fn plan_rows(
        &self,
        row_ids: &[u64],
    ) -> Result<(Vec<RowLocation>, Vec<PageKey>), RowStoreError> {
        let mut locations = Vec::with_capacity(row_ids.len());
        let mut pages = Vec::with_capacity(row_ids.len());
        for &row in row_ids {
            let location = self.inner.locate_row(row)?;
            pages.push(location.page_key());
            locations.push(location);
        }
        pages.sort_unstable();
        pages.dedup();
        Ok((locations, pages))
    }
}

impl Drop for RowStore {
    fn drop(&mut self) {
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.current_epoch.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut state) = self.inner.state.lock() {
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            state.stop = true;
            self.inner.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Bound on draining readers and leases when a [`RowFetch`] aborts.
const FETCH_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Owns one request's rows until the request either commits or aborts.
///
/// [`RowFetch::begin`] starts a fresh epoch, [`RowFetch::prefetch`] enqueues
/// the rows, and [`RowFetch::wait`] turns the ticket into the lease.  Immutable
/// page-cache entries belong to the [`RowStore`] and survive an abort; this
/// guard only owns the exact ticket and completed lease of its epoch.  Dropping
/// it without [`RowFetch::complete`] aborts.
pub struct RowFetch<'a> {
    store: &'a RowStore,
    epoch: u64,
    ticket: Option<RowTicket>,
    lease: Option<RowLease>,
    armed: bool,
}

impl<'a> RowFetch<'a> {
    /// Start the next request epoch, invalidating every earlier ticket.
    pub fn begin(store: &'a RowStore) -> Result<Self, RowStoreError> {
        let current = store.current_epoch();
        let epoch = current
            .checked_add(1)
            .ok_or(RowStoreError::EpochNotMonotonic {
                requested: u64::MAX,
                current,
            })?;
        store.begin_epoch(epoch)?;
        Ok(Self {
            store,
            epoch,
            ticket: None,
            lease: None,
            armed: true,
        })
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Enqueue this request's rows.  Call once, as soon as the ids are known.
    pub fn prefetch(&mut self, row_ids: Vec<u64>) -> Result<(), RowStoreError> {
        debug_assert!(self.ticket.is_none() && self.lease.is_none());
        self.ticket = Some(self.store.prefetch(self.epoch, row_ids)?);
        Ok(())
    }

    /// The completed lease, if [`Self::wait`] has already produced it.
    pub fn lease(&self) -> Option<&RowLease> {
        self.lease.as_ref()
    }

    /// Block until the prefetched rows are complete and return the lease.
    pub fn wait(&mut self) -> Result<&RowLease, RowStoreError> {
        if self.lease.is_none() {
            let ticket = self
                .ticket
                .as_ref()
                .ok_or_else(|| RowStoreError::InvalidLease {
                    reason: "rows awaited before they were prefetched".to_string(),
                })?;
            let lease = self.store.wait_completed_lease(ticket)?;
            // `wait_completed_lease` marks the ticket consumed.  Drop that
            // handle before retaining the lease so the only live owner is
            // explicit.
            self.ticket = None;
            self.lease = Some(lease);
        }
        Ok(self.lease.as_ref().expect("lease installed above"))
    }

    /// Release the ticket and lease after the request committed.
    pub fn complete(&mut self) {
        self.armed = false;
        drop(self.ticket.take());
        drop(self.lease.take());
    }

    /// Cancel outstanding work and drain the store into a fresh epoch.  Returns
    /// that epoch, or every cleanup failure joined into one message.
    pub fn abort(&mut self) -> Result<u64, String> {
        if !self.armed {
            return Ok(self.store.current_epoch());
        }
        // Disarm first: if cleanup itself reports an error, Drop must not
        // issue a second reset against a later epoch.
        self.armed = false;
        let mut errors = Vec::new();
        if let Some(ticket) = self.ticket.take() {
            if let Err(error) = self.store.cancel(&ticket) {
                // wait_completed_lease marks a ticket consumed before waiting;
                // source-read and cancellation errors therefore legitimately
                // report AlreadyConsumed during abort.
                if !matches!(
                    error,
                    RowStoreError::AlreadyConsumed(_)
                        | RowStoreError::Canceled
                        | RowStoreError::UnknownTicket(_)
                ) {
                    errors.push(format!("cancel epoch {} ticket: {error}", self.epoch));
                }
            }
            drop(ticket);
        }
        // A lease holds one of the bounded staging buffers.  It must be
        // returned before reset_epoch waits for readers/leases to drain.
        drop(self.lease.take());
        let next_epoch = match self.store.reset_epoch(FETCH_CLEANUP_TIMEOUT) {
            Ok(next) => Some(next),
            Err(error) => {
                errors.push(format!("drain epoch {}: {error}", self.epoch));
                None
            }
        };
        if errors.is_empty() {
            Ok(next_epoch.expect("successful reset returns its next epoch"))
        } else {
            Err(errors.join("; "))
        }
    }
}

impl Drop for RowFetch<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.abort();
        }
    }
}

struct RowStoreInner {
    source: Arc<dyn PositionalRowSource>,
    shard_count: usize,
    _descriptors: Arc<[SourceRangeDescriptor]>,
    valid_rows: u64,
    padded_rows: u64,
    /// Stored row representation, fixed by the sealed shard dtype.
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    /// BF16 width of one published row.
    decoded_row_bytes: usize,
    /// Whole-row size of one staging buffer / coalesced read.
    staging_bytes: usize,
    rows_per_shard: usize,
    rows_per_page: usize,
    epoch_started: AtomicBool,
    current_epoch: AtomicU64,
    stopped: AtomicBool,
    state: Mutex<RowStoreState>,
    cv: Condvar,
}

impl RowStoreInner {
    fn locate_row(&self, row: u64) -> Result<RowLocation, RowStoreError> {
        if row >= self.padded_rows {
            return Err(RowStoreError::GlobalRowOutOfRange {
                row,
                padded_rows: self.padded_rows,
            });
        }
        if row >= self.valid_rows {
            return Err(RowStoreError::PaddingRow {
                row,
                valid_rows: self.valid_rows,
            });
        }
        let shard_rows = self.rows_per_shard as u64;
        let shard = usize::try_from(row / shard_rows).map_err(|_| RowStoreError::InvalidLease {
            reason: "shard index overflow".to_string(),
        })?;
        let local_row =
            usize::try_from(row % shard_rows).map_err(|_| RowStoreError::InvalidLease {
                reason: "local row overflow".to_string(),
            })?;
        if shard >= self.shard_count {
            return Err(RowStoreError::GlobalRowOutOfRange {
                row,
                padded_rows: self.padded_rows,
            });
        }
        let page = local_row / self.rows_per_page;
        let row_in_page = local_row % self.rows_per_page;
        let page_byte_offset =
            row_in_page
                .checked_mul(self.encoded_row_bytes)
                .ok_or_else(|| RowStoreError::InvalidLease {
                    reason: "page byte offset overflow".to_string(),
                })?;
        Ok(RowLocation {
            global_row: row,
            shard,
            local_row,
            page,
            page_byte_offset,
        })
    }

    fn validate_lease_epoch(&self, lease_epoch: u64) -> Result<(), RowStoreError> {
        let current = self.current_epoch.load(Ordering::Acquire);
        if self.stopped.load(Ordering::Acquire) || current != lease_epoch {
            Err(RowStoreError::LeaseStale {
                lease_epoch,
                current_epoch: current,
            })
        } else {
            Ok(())
        }
    }

    fn report(&self) -> RowQuiesceReport {
        let state = self.state.lock().expect("row store state mutex poisoned");
        self.report_locked(&state)
    }

    fn report_locked(&self, state: &RowStoreState) -> RowQuiesceReport {
        RowQuiesceReport {
            queue_depth: state.queue.len(),
            outstanding_readers: state.active_readers,
            outstanding_leases: state.active_leases,
            staging_in_use: state.staging_in_use,
            cache_bytes: state.cache.resident_bytes,
            cache_pages: state.cache.pages.len(),
        }
    }

    fn stats(&self) -> RowCacheStats {
        let state = self.state.lock().expect("row store state mutex poisoned");
        RowCacheStats {
            capacity_bytes: PAGE_CACHE_BYTES,
            resident_bytes: state.cache.resident_bytes,
            resident_pages: state.cache.pages.len(),
            cache_hits: state.cache.hits,
            cache_misses: state.cache.misses,
            reads: state.cache.reads,
            coalesced_reads: state.cache.coalesced_reads,
            read_bytes: state.cache.read_bytes,
            evictions: state.cache.evictions,
            queue_depth: state.queue.len(),
            outstanding_readers: state.active_readers,
            outstanding_leases: state.active_leases,
            staging_in_use: state.staging_in_use,
            staging_high_water: state.staging_high_water,
        }
    }

    fn take_staging_locked(&self, state: &mut RowStoreState) -> Option<Vec<u8>> {
        let slot = state.staging_pool.iter().position(Option::is_some)?;
        let buffer = state.staging_pool[slot].take();
        state.staging_in_use += 1;
        state.staging_high_water = state.staging_high_water.max(state.staging_in_use);
        buffer
    }

    fn return_staging_locked(&self, state: &mut RowStoreState, mut buffer: Vec<u8>) {
        buffer.clear();
        if let Some(slot) = state.staging_pool.iter().position(Option::is_none) {
            state.staging_pool[slot] = Some(buffer);
        }
        state.staging_in_use = state.staging_in_use.saturating_sub(1);
    }

    fn cancel_ticket(&self, id: u64) -> Result<(), RowStoreError> {
        let mut state = self.state.lock().expect("row store state mutex poisoned");
        let Some(ticket) = state.tickets.get(&id) else {
            return Ok(());
        };
        ticket.canceled.store(true, Ordering::Release);

        let queued = state.queue.iter().position(|queued_id| *queued_id == id);
        let completed = matches!(
            state.tickets.get(&id).map(|ticket| &ticket.status),
            Some(TicketStatus::Completed(_))
        );
        if let Some(position) = queued {
            state.queue.remove(position);
        }
        if queued.is_some() || completed {
            if let Some(ticket) = state.tickets.remove(&id) {
                if let Some(buffer) = ticket.output {
                    self.return_staging_locked(&mut state, buffer);
                }
            }
        }
        self.cv.notify_all();
        Ok(())
    }
    fn process_ticket(&self, id: u64, read_buffer: Vec<u8>) -> ProcessedTicket {
        let (epoch, canceled, pages, locations, output) = {
            let mut state = self.state.lock().expect("row store state mutex poisoned");
            let Some(ticket) = state.tickets.get_mut(&id) else {
                return ProcessedTicket {
                    output: None,
                    read_buffer,
                    result: Err(RowStoreError::Canceled),
                };
            };
            (
                ticket.epoch,
                ticket.canceled.clone(),
                ticket.pages.clone(),
                ticket.locations.clone(),
                ticket.output.take(),
            )
        };
        let mut output = output.unwrap_or_default();
        let mut read_buffer = read_buffer;
        if canceled.load(Ordering::Acquire) {
            return ProcessedTicket {
                output: Some(output),
                read_buffer,
                result: Err(RowStoreError::Canceled),
            };
        }
        let current = self.current_epoch.load(Ordering::Acquire);
        if epoch != current {
            return ProcessedTicket {
                output: Some(output),
                read_buffer,
                result: Err(RowStoreError::EpochMismatch {
                    requested: epoch,
                    current,
                }),
            };
        }
        let result = self.fill_output(&locations, &pages, &mut output, &mut read_buffer, &canceled);
        ProcessedTicket {
            output: Some(output),
            read_buffer,
            result,
        }
    }

    /// Fill the final bounded lease directly while each page group is
    /// available.  Requested page sets may be much larger than the cache;
    /// no second pass assumes all pages remain resident.
    fn fill_output(
        &self,
        locations: &[RowLocation],
        pages: &[PageKey],
        output: &mut [u8],
        read_staging: &mut Vec<u8>,
        canceled: &AtomicBool,
    ) -> Result<(), RowStoreError> {
        let expected = locations
            .len()
            .checked_mul(self.decoded_row_bytes)
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "staging size overflow".to_string(),
            })?;
        if output.len() != expected {
            return Err(RowStoreError::InvalidLease {
                reason: "staging size does not match row plan".to_string(),
            });
        }
        let mut requested: HashMap<PageKey, Vec<RowCopy>> = HashMap::new();
        for (index, location) in locations.iter().copied().enumerate() {
            requested
                .entry(location.page_key())
                .or_default()
                .push(RowCopy {
                    output_offset: index * self.decoded_row_bytes,
                    page_offset: location.page_byte_offset,
                });
        }

        let mut missing = Vec::new();
        for &page in pages {
            if canceled.load(Ordering::Acquire) {
                return Err(RowStoreError::Canceled);
            }
            let mut state = self.state.lock().expect("row store state mutex poisoned");
            if let Some(cached) = state.cache.pages.get(&page) {
                copy_requested_rows(
                    self.encoding,
                    self.encoded_row_bytes,
                    self.decoded_row_bytes,
                    page,
                    &cached.bytes,
                    requested.get(&page).map(Vec::as_slice).unwrap_or(&[]),
                    output,
                )?;
                state.cache.touch(page);
                state.cache.hits += 1;
            } else {
                state.cache.misses += 1;
                missing.push(page);
            }
        }

        let groups = self.coalesce_pages(&missing)?;
        for group in groups {
            if canceled.load(Ordering::Acquire) {
                return Err(RowStoreError::Canceled);
            }
            self.read_group(&group, &requested, output, read_staging, canceled)?;
        }
        Ok(())
    }

    fn coalesce_pages(&self, pages: &[PageKey]) -> Result<Vec<Vec<PageKey>>, RowStoreError> {
        let mut groups: Vec<Vec<PageKey>> = Vec::new();
        let mut current_bytes = 0usize;
        for &page in pages {
            let page_bytes = self.page_len(page)?;
            let append = groups.last().is_some_and(|group| {
                let previous = group[group.len() - 1];
                previous.shard == page.shard
                    && previous.page.checked_add(1) == Some(page.page)
                    && current_bytes
                        .checked_add(page_bytes)
                        .is_some_and(|total| total <= self.staging_bytes)
            });
            if append {
                groups.last_mut().expect("group exists").push(page);
                current_bytes = current_bytes.checked_add(page_bytes).ok_or_else(|| {
                    RowStoreError::InvalidLease {
                        reason: "coalesced length overflow".to_string(),
                    }
                })?;
            } else {
                groups.push(vec![page]);
                current_bytes = page_bytes;
            }
        }
        Ok(groups)
    }

    fn page_len(&self, page: PageKey) -> Result<usize, RowStoreError> {
        if page.shard >= self.shard_count {
            return Err(RowStoreError::GlobalRowOutOfRange {
                row: self.padded_rows,
                padded_rows: self.padded_rows,
            });
        }
        let first_row = page.page.checked_mul(self.rows_per_page).ok_or_else(|| {
            RowStoreError::InvalidLease {
                reason: "page row offset overflow".to_string(),
            }
        })?;
        if first_row >= self.rows_per_shard {
            return Err(RowStoreError::InvalidLease {
                reason: "page exceeds physical shard".to_string(),
            });
        }
        let rows = (self.rows_per_shard - first_row).min(self.rows_per_page);
        rows.checked_mul(self.encoded_row_bytes)
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "page byte length overflow".to_string(),
            })
    }
    fn read_group(
        &self,
        group: &[PageKey],
        requested: &HashMap<PageKey, Vec<RowCopy>>,
        output: &mut [u8],
        read_staging: &mut Vec<u8>,
        canceled: &AtomicBool,
    ) -> Result<(), RowStoreError> {
        let first = *group.first().ok_or_else(|| RowStoreError::InvalidLease {
            reason: "empty coalesced page group".to_string(),
        })?;
        let first_offset = first
            .page
            .checked_mul(self.rows_per_page)
            .and_then(|row| row.checked_mul(self.encoded_row_bytes))
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "coalesced offset overflow".to_string(),
            })? as u64;
        let lengths: Vec<usize> = group
            .iter()
            .map(|&page| self.page_len(page))
            .collect::<Result<_, _>>()?;
        let total = lengths
            .iter()
            .try_fold(0usize, |sum, len| sum.checked_add(*len))
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "coalesced length overflow".to_string(),
            })?;
        if total > self.staging_bytes {
            return Err(RowStoreError::InvalidLease {
                reason: "coalesced read exceeds staging budget".to_string(),
            });
        }

        read_staging.resize(total, 0);
        self.source
            .read_at(first.shard, first_offset, read_staging)
            .map_err(RowStoreError::Source)?;
        if canceled.load(Ordering::Acquire) {
            return Err(RowStoreError::Canceled);
        }

        let mut state = self.state.lock().expect("row store state mutex poisoned");
        let mut cursor = 0usize;
        for (&page, &length) in group.iter().zip(&lengths) {
            let page_bytes = &read_staging[cursor..cursor + length];
            if let Some(rows) = requested.get(&page) {
                copy_requested_rows(
                    self.encoding,
                    self.encoded_row_bytes,
                    self.decoded_row_bytes,
                    page,
                    page_bytes,
                    rows,
                    output,
                )?;
            }
            state.cache.insert(page, page_bytes.to_vec());
            cursor += length;
        }
        state.cache.reads += 1;
        state.cache.coalesced_reads += 1;
        state.cache.read_bytes += total as u64;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct RowCopy {
    output_offset: usize,
    page_offset: usize,
}

/// Copy the requested rows out of one cached page into the lease staging
/// buffer, decoding from the stored representation to `decoded_row_bytes` of
/// BF16.
///
/// The cache keeps the *encoded* page, so a decode is paid per access instead
/// of per distinct row; at 160 weights per row that is five Q8 blocks, and the
/// alternative (caching decoded rows) would cut the cache's row capacity by the
/// same encoding ratio.
fn copy_requested_rows(
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    decoded_row_bytes: usize,
    page: PageKey,
    page_bytes: &[u8],
    rows: &[RowCopy],
    output: &mut [u8],
) -> Result<(), RowStoreError> {
    for row in rows {
        let page_end = row
            .page_offset
            .checked_add(encoded_row_bytes)
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: format!("page {page:?} row offset overflow"),
            })?;
        let output_end = row
            .output_offset
            .checked_add(decoded_row_bytes)
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "output row offset overflow".to_string(),
            })?;
        if page_end > page_bytes.len() || output_end > output.len() {
            return Err(RowStoreError::InvalidLease {
                reason: format!("requested row is outside page {page:?} or output"),
            });
        }
        encoding.decode_row(
            &page_bytes[row.page_offset..page_end],
            &mut output[row.output_offset..output_end],
        )?;
    }
    Ok(())
}

struct ProcessedTicket {
    output: Option<Vec<u8>>,
    read_buffer: Vec<u8>,
    result: Result<(), RowStoreError>,
}

struct TicketState {
    epoch: u64,
    canceled: Arc<AtomicBool>,
    ids: Vec<u64>,
    locations: Vec<RowLocation>,
    pages: Vec<PageKey>,
    required_bytes: usize,
    output: Option<Vec<u8>>,
    status: TicketStatus,
}

enum TicketStatus {
    Pending,
    Completed(Result<(), RowStoreError>),
}

struct RowStoreState {
    queue: VecDeque<u64>,
    tickets: HashMap<u64, TicketState>,
    next_ticket: u64,
    accepting: bool,
    stop: bool,
    active_readers: usize,
    active_leases: usize,
    staging_pool: Vec<Option<Vec<u8>>>,
    staging_in_use: usize,
    staging_high_water: usize,
    cache: PageCache,
}

impl RowStoreState {
    fn new(staging_bytes: usize) -> Self {
        let staging_pool = (0..MAX_STAGING_BUFFERS)
            .map(|_| Some(Vec::with_capacity(staging_bytes)))
            .collect();
        Self {
            queue: VecDeque::new(),
            tickets: HashMap::new(),
            next_ticket: 1,
            accepting: true,
            stop: false,
            active_readers: 0,
            active_leases: 0,
            staging_pool,
            staging_in_use: 0,

            staging_high_water: 0,
            cache: PageCache::new(PAGE_CACHE_BYTES),
        }
    }
}
fn invalidate_tickets_locked(inner: &RowStoreInner, state: &mut RowStoreState) {
    for ticket in state.tickets.values() {
        ticket.canceled.store(true, Ordering::Release);
    }
    let queued: Vec<u64> = state.queue.drain(..).collect();
    for id in queued {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
    let completed: Vec<u64> = state
        .tickets
        .iter()
        .filter_map(|(&id, ticket)| {
            matches!(ticket.status, TicketStatus::Completed(_)).then_some(id)
        })
        .collect();
    for id in completed {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
}

fn purge_tickets_locked(inner: &RowStoreInner, state: &mut RowStoreState) {
    let ids: Vec<u64> = state.tickets.keys().copied().collect();
    for id in ids {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
}

fn worker_loop(weak: Weak<RowStoreInner>) {
    loop {
        let Some(inner) = weak.upgrade() else {
            break;
        };
        let work = {
            let mut state = inner.state.lock().expect("row store state mutex poisoned");
            loop {
                if state.stop {
                    break None;
                }
                let Some(id) = state.queue.front().copied() else {
                    state = inner
                        .cv
                        .wait(state)
                        .expect("row store state mutex poisoned");
                    continue;
                };
                let Some(mut output) = inner.take_staging_locked(&mut state) else {
                    state = inner
                        .cv
                        .wait(state)
                        .expect("row store state mutex poisoned");
                    continue;
                };
                let Some(read_buffer) = inner.take_staging_locked(&mut state) else {
                    inner.return_staging_locked(&mut state, output);
                    state = inner
                        .cv
                        .wait(state)
                        .expect("row store state mutex poisoned");
                    continue;
                };
                state.queue.pop_front();
                let Some(ticket) = state.tickets.get_mut(&id) else {
                    inner.return_staging_locked(&mut state, output);
                    inner.return_staging_locked(&mut state, read_buffer);
                    continue;
                };
                output.resize(ticket.required_bytes, 0);
                ticket.output = Some(output);
                state.active_readers += 1;
                break Some((id, read_buffer));
            }
        };
        let Some((id, read_buffer)) = work else {
            break;
        };
        let ProcessedTicket {
            mut output,
            read_buffer,
            result,
        } = inner.process_ticket(id, read_buffer);
        let mut state = inner.state.lock().expect("row store state mutex poisoned");
        state.active_readers = state.active_readers.saturating_sub(1);
        inner.return_staging_locked(&mut state, read_buffer);
        let Some((ticket_canceled, ticket_epoch)) = state
            .tickets
            .get(&id)
            .map(|ticket| (ticket.canceled.load(Ordering::Acquire), ticket.epoch))
        else {
            if let Some(buffer) = output.take() {
                inner.return_staging_locked(&mut state, buffer);
            }
            inner.cv.notify_all();
            continue;
        };
        let canceled =
            ticket_canceled || ticket_epoch != inner.current_epoch.load(Ordering::Acquire);
        if canceled {
            if let Some(ticket) = state.tickets.remove(&id) {
                if let Some(buffer) = ticket.output {
                    inner.return_staging_locked(&mut state, buffer);
                }
            }
            if let Some(buffer) = output.take() {
                inner.return_staging_locked(&mut state, buffer);
            }
            inner.cv.notify_all();
            continue;
        }
        let mut return_buffer = None;
        {
            let ticket = state.tickets.get_mut(&id).expect("ticket disappeared");
            match result {
                Ok(()) => {
                    ticket.output = output;
                    ticket.status = TicketStatus::Completed(Ok(()));
                }
                Err(error) => {
                    return_buffer = output.take();
                    ticket.output = None;
                    ticket.status = TicketStatus::Completed(Err(error));
                }
            }
        }
        if let Some(buffer) = return_buffer {
            inner.return_staging_locked(&mut state, buffer);
        }
        inner.cv.notify_all();
    }
}

trait PositionalRowSource: Send + Sync {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError>;
}

struct DescriptorRowSource {
    descriptors: Arc<[SourceRangeDescriptor]>,
}

impl PositionalRowSource for DescriptorRowSource {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        let descriptor = self
            .descriptors
            .get(shard)
            .ok_or_else(|| SourceError::InvalidSource {
                reason: format!("row shard {shard} is absent"),
            })?;
        let absolute =
            descriptor
                .offset
                .checked_add(local_offset)
                .ok_or(SourceError::Overflow {
                    offset: descriptor.offset,
                    length: local_offset,
                })?;
        descriptor.read_exact_at(absolute, dst)
    }
}

struct CachedPage {
    bytes: Vec<u8>,
    last_used: u64,
}

struct PageCache {
    capacity_bytes: usize,
    pages: HashMap<PageKey, CachedPage>,
    /// Recency index over [`CachedPage::last_used`]: stamp -> page.  A page is
    /// small, so the cache holds tens of thousands of entries and the least
    /// recently used page must be found without scanning the map.
    by_stamp: BTreeMap<u64, PageKey>,
    resident_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    reads: u64,
    coalesced_reads: u64,
    read_bytes: u64,
    evictions: u64,
}

impl PageCache {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            pages: HashMap::new(),
            by_stamp: BTreeMap::new(),
            resident_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            reads: 0,
            coalesced_reads: 0,
            read_bytes: 0,
            evictions: 0,
        }
    }

    fn touch(&mut self, key: PageKey) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(page) = self.pages.get_mut(&key) {
            self.by_stamp.remove(&page.last_used);
            page.last_used = self.clock;
            self.by_stamp.insert(self.clock, key);
        }
    }

    fn insert(&mut self, key: PageKey, bytes: Vec<u8>) {
        if bytes.len() > self.capacity_bytes {
            return;
        }
        if let Some(previous) = self.pages.remove(&key) {
            self.by_stamp.remove(&previous.last_used);
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.bytes.len());
        }
        while self.resident_bytes.saturating_add(bytes.len()) > self.capacity_bytes {
            let Some((&stamp, &victim)) = self.by_stamp.iter().next() else {
                break;
            };
            self.by_stamp.remove(&stamp);
            if let Some(page) = self.pages.remove(&victim) {
                self.resident_bytes = self.resident_bytes.saturating_sub(page.bytes.len());
                self.evictions += 1;
            }
        }
        self.clock = self.clock.wrapping_add(1);
        self.resident_bytes += bytes.len();
        self.by_stamp.insert(self.clock, key);
        self.pages.insert(
            key,
            CachedPage {
                bytes,
                last_used: self.clock,
            },
        );
    }

    fn clear(&mut self) -> usize {
        let dropped = self.resident_bytes;
        self.pages.clear();
        self.by_stamp.clear();
        self.resident_bytes = 0;
        dropped
    }
}

/// Physical layout of one validated shard set.
#[derive(Debug, Clone, Copy)]
struct Layout {
    shard_count: usize,
    rows_per_shard: usize,
    rows_per_page: usize,
    row_width: usize,
    valid_rows: u64,
    encoding: RowEncoding,
}

fn validate_descriptors(
    descriptors: &[SourceRangeDescriptor],
    valid_rows: u64,
) -> Result<Layout, RowStoreError> {
    let first = descriptors
        .first()
        .ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: "at least one physical shard is required".to_string(),
        })?;
    // The declared dtype is the contract: BF16 shards are `2 * width`-byte raw
    // rows and Q8F16 shards are `width / 32` 34-byte blocks per row. Nothing
    // infers the tier from the byte count, so a mislabelled artifact is refused
    // rather than decoded as the wrong width.
    let encoding =
        RowEncoding::from_dtype(first.dtype()).ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: format!("rows must be BF16 or Q8_0, got {}", first.dtype()),
        })?;
    let shape = first.logical_shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(RowStoreError::Descriptor {
            index: 0,
            reason: format!("expected a non-empty [rows, width] shape, got {shape:?}"),
        });
    }
    let (rows, width) = (shape[0], shape[1]);
    if encoding == RowEncoding::Q8F16 && width % Q8_BLOCK_WEIGHTS != 0 {
        return Err(RowStoreError::Descriptor {
            index: 0,
            reason: format!("Q8F16 row width {width} is not a multiple of {Q8_BLOCK_WEIGHTS}"),
        });
    }
    let expected_length = (rows as u64)
        .checked_mul(encoding.encoded_row_bytes(width) as u64)
        .ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: "descriptor length overflow".to_string(),
        })?;
    let identity = first.source_identity();
    for (index, descriptor) in descriptors.iter().enumerate() {
        if descriptor.source_identity() != identity {
            return Err(RowStoreError::Descriptor {
                index,
                reason: "descriptor source identity differs from shard 0".to_string(),
            });
        }
        if descriptor.logical_shape() != [rows, width] {
            return Err(RowStoreError::Descriptor {
                index,
                reason: format!("all shards must have shape [{rows}, {width}]"),
            });
        }
        if descriptor.length != expected_length {
            return Err(RowStoreError::Descriptor {
                index,
                reason: format!(
                    "expected {expected_length} bytes for {} rows, got {}",
                    encoding.dtype_name(),
                    descriptor.length
                ),
            });
        }
        if RowEncoding::from_dtype(descriptor.dtype()) != Some(encoding) {
            return Err(RowStoreError::Descriptor {
                index,
                reason: format!(
                    "all shards must be {}, got {}",
                    encoding.dtype_name(),
                    descriptor.dtype()
                ),
            });
        }
    }
    let physical_rows = (rows as u64)
        .checked_mul(descriptors.len() as u64)
        .ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: "physical row count overflow".to_string(),
        })?;
    if valid_rows == 0 || valid_rows > physical_rows {
        return Err(RowStoreError::Descriptor {
            index: 0,
            reason: format!("valid rows {valid_rows} outside 1..={physical_rows} physical rows"),
        });
    }
    Ok(Layout {
        shard_count: descriptors.len(),
        rows_per_shard: rows,
        rows_per_page: ROWS_PER_PAGE,
        row_width: width,
        valid_rows,
        encoding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::atomic::AtomicUsize;

    struct MemoryRowSource {
        shards: Vec<Vec<u8>>,
        reads: AtomicUsize,
        fail: Option<(usize, u64)>,
    }

    impl PositionalRowSource for MemoryRowSource {
        fn read_at(&self, shard: usize, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
            self.reads.fetch_add(1, Ordering::AcqRel);
            if self.fail == Some((shard, offset)) {
                return Err(SourceError::Io {
                    offset,
                    source: io::Error::new(io::ErrorKind::Other, "fixture read failure"),
                });
            }
            let source = self
                .shards
                .get(shard)
                .ok_or_else(|| SourceError::InvalidSource {
                    reason: "fixture shard missing".to_string(),
                })?;
            let begin = usize::try_from(offset).map_err(|_| SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            let end = begin.checked_add(dst.len()).ok_or(SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            if end > source.len() {
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: source.len().saturating_sub(begin),
                });
            }
            dst.copy_from_slice(&source[begin..end]);
            Ok(())
        }
    }

    /// The Qwen4 PLE shape these fixtures were written against: sixteen
    /// 160-value rows per token.
    const ROWS_PER_TOKEN: usize = 16;
    const ROW_WIDTH: usize = 160;
    const ROW_BYTES: usize = ROW_WIDTH * 2;

    /// Valid/padded row counts of a fixture table made of sixteen equal
    /// per-head sub-tables.
    #[derive(Clone, Copy)]
    struct Fixture {
        valid_rows: u64,
        padded_rows: u64,
    }

    fn metadata_with_head_size(head_size: u64, padded_rows: u64) -> Fixture {
        Fixture {
            valid_rows: head_size * ROWS_PER_TOKEN as u64,
            padded_rows,
        }
    }

    fn metadata(padded_rows: u64) -> Fixture {
        metadata_with_head_size(2, padded_rows)
    }

    fn rows_source(shard_count: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        (0..shard_count)
            .map(|shard| {
                let mut bytes = vec![0u8; rows_per_shard * ROW_BYTES];
                for row in 0..rows_per_shard {
                    let global = shard * rows_per_shard + row;
                    let value = (global as u16).to_le_bytes();
                    for cell in bytes[row * ROW_BYTES..(row + 1) * ROW_BYTES].chunks_exact_mut(2) {
                        cell.copy_from_slice(&value);
                    }
                }
                bytes
            })
            .collect()
    }

    impl RowStore {
        /// Prefetch sixteen rows per token, one from each per-head sub-table,
        /// with ids scattered like a multiplicative hash of `(seed, token,
        /// head)`.  A repeated token repeats its ids.
        fn prefetch_tokens(
            &self,
            epoch: u64,
            seed: u32,
            tokens: &[u32],
        ) -> Result<RowTicket, RowStoreError> {
            let head_rows = self.inner.valid_rows / ROWS_PER_TOKEN as u64;
            let ids = tokens
                .iter()
                .flat_map(|&token| {
                    (0..ROWS_PER_TOKEN as u64).map(move |head| {
                        let mut x = (u64::from(token) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ u64::from(seed).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                            ^ (head + 1).wrapping_mul(0x1656_67B1_9E37_79F9);
                        x ^= x >> 29;
                        head * head_rows + x % head_rows
                    })
                })
                .collect();
            self.prefetch(epoch, ids)
        }

        fn from_test_source(
            metadata: Fixture,
            rows_per_shard: usize,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, RowStoreError> {
            Self::from_test_source_with_page_rows(metadata, rows_per_shard, ROWS_PER_PAGE, source)
        }

        fn from_test_source_with_page_rows(
            metadata: Fixture,
            rows_per_shard: usize,
            rows_per_page: usize,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, RowStoreError> {
            Self::from_test_source_encoded(
                metadata,
                rows_per_shard,
                rows_per_page,
                RowEncoding::Bf16,
                source,
            )
        }

        fn from_test_source_encoded(
            metadata: Fixture,
            rows_per_shard: usize,
            rows_per_page: usize,
            encoding: RowEncoding,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, RowStoreError> {
            if rows_per_shard == 0 || rows_per_page == 0 {
                return Err(RowStoreError::Descriptor {
                    index: 0,
                    reason: "invalid fixture shard geometry".to_string(),
                });
            }
            let shard_count = usize::try_from(metadata.padded_rows / rows_per_shard as u64)
                .map_err(|_| RowStoreError::Descriptor {
                    index: 0,
                    reason: "test shard count overflow".to_string(),
                })?;
            if shard_count == 0
                || (shard_count as u64) * rows_per_shard as u64 != metadata.padded_rows
            {
                return Err(RowStoreError::Descriptor {
                    index: 0,
                    reason: "fixture rows do not cover the padded table".to_string(),
                });
            }
            Self::spawn(
                "external-rows-test-reader",
                source,
                Vec::new().into(),
                Layout {
                    shard_count,
                    rows_per_shard,
                    rows_per_page,
                    row_width: ROW_WIDTH,
                    valid_rows: metadata.valid_rows,
                    encoding,
                },
            )
        }
    }

    /// The production constructor reads real sealed HFQ ranges: rows come back
    /// in request order, padding past `valid_rows` is refused, and a
    /// `valid_rows` beyond the physical table cannot open.
    #[test]
    fn production_store_reads_sealed_ranges_and_refuses_padding() {
        use crate::hfq::hfq_test_fixture::{
            write_compact_qwen4_ple_hfq, COMPACT_PLE_NAMES, COMPACT_PLE_ROW_BYTES,
        };
        use crate::hfq::HfqFile;
        use crate::model_source::{ModelSource, SourcePayload};

        let dir = tempfile::tempdir().expect("fixture directory");
        let path = dir.path().join("rows.hfq");
        write_compact_qwen4_ple_hfq(&path).expect("write compact fixture");
        let hfq = HfqFile::open(&path).expect("open compact fixture");
        let descriptors: Vec<SourceRangeDescriptor> = COMPACT_PLE_NAMES
            .iter()
            .map(
                |name| match (&hfq as &dyn ModelSource).tensor_payload(name) {
                    Ok(Some(SourcePayload::Range(descriptor))) => descriptor,
                    _ => panic!("fixture tensor {name} is not a lazy range"),
                },
            )
            .collect();
        let physical_rows = descriptors
            .iter()
            .map(|descriptor| descriptor.logical_shape()[0] as u64)
            .sum::<u64>();
        let mut expected = Vec::new();
        for (shard, descriptor) in descriptors.iter().enumerate() {
            let mut row = vec![0u8; COMPACT_PLE_ROW_BYTES];
            descriptor
                .read_exact_at(descriptor.offset, &mut row)
                .unwrap();
            expected.push((shard as u64 * descriptor.logical_shape()[0] as u64, row));
        }

        let too_many = RowStore::new("t", descriptors.clone(), physical_rows + 1);
        assert!(matches!(too_many, Err(RowStoreError::Descriptor { .. })));

        let store = RowStore::new("t", descriptors, physical_rows - 1).unwrap();
        assert!(matches!(
            store.prefetch(0, vec![physical_rows - 1]),
            Err(RowStoreError::PaddingRow { .. })
        ));
        let ids: Vec<u64> = expected.iter().rev().map(|(row, _)| *row).collect();
        let ticket = store.prefetch(0, ids.clone()).unwrap();
        let lease = store.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), ids.as_slice());
        for (index, (_, row)) in expected.iter().rev().enumerate() {
            assert_eq!(lease.row_bytes(index).unwrap(), row.as_slice());
        }
        drop(lease);
        assert!(store.unload().unwrap().is_clean());
    }

    // These tests exercise bounded cache and deterministic page planning.
    #[test]
    fn cache_is_bounded_and_evicts_lru() {
        let mut cache = PageCache::new(ROW_BYTES * 2);
        cache.insert(PageKey { shard: 0, page: 0 }, vec![1; ROW_BYTES]);
        cache.insert(PageKey { shard: 0, page: 1 }, vec![2; ROW_BYTES]);
        cache.touch(PageKey { shard: 0, page: 0 });
        cache.insert(PageKey { shard: 0, page: 2 }, vec![3; ROW_BYTES]);
        assert_eq!(cache.resident_bytes, ROW_BYTES * 2);
        assert_eq!(cache.pages.len(), 2);
        assert!(!cache.pages.contains_key(&PageKey { shard: 0, page: 1 }));
        assert_eq!(cache.evictions, 1);
    }

    #[test]
    fn location_rejects_padding_and_maps_shard_boundary() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        assert_eq!(
            rows.locate_row(0).unwrap(),
            RowLocation {
                global_row: 0,
                shard: 0,
                local_row: 0,
                page: 0,
                page_byte_offset: 0,
            }
        );
        let error = rows.locate_row(32).unwrap_err();
        assert!(matches!(error, RowStoreError::PaddingRow { row: 32, .. }));
        let full = RowStore::from_test_source(
            metadata_with_head_size(8, 128),
            64,
            Arc::new(MemoryRowSource {
                shards: rows_source(2, 64),
                reads: AtomicUsize::new(0),
                fail: None,
            }),
        )
        .unwrap();
        assert_eq!(full.locate_row(63).unwrap().shard, 0);
        assert_eq!(full.locate_row(64).unwrap().shard, 1);
        assert!(full.unload().unwrap().is_clean());
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn prefetch_reads_once_and_preserves_token_head_order() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source.clone()).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0, 1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!((lease.row_count() / ROWS_PER_TOKEN), 2);
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        for token in 0..2 {
            for head in 0..ROWS_PER_TOKEN {
                let row = lease.row_bytes(token * ROWS_PER_TOKEN + head).unwrap();
                let value = u16::from_le_bytes([row[0], row[1]]) as u64;
                assert_eq!(value, lease.row_ids()[token * ROWS_PER_TOKEN + head]);
            }
        }
        let mut copied = vec![0; lease.as_bytes().unwrap().len()];
        lease.stage_into(&mut copied).unwrap();
        assert_eq!(copied, lease.as_bytes().unwrap());
        drop(lease);
        // Every distinct physical page is read exactly once for one ticket;
        // adjacent pages may coalesce into one call.
        let mut distinct = std::collections::HashSet::new();
        let mut expected_bytes = 0usize;
        for &row in expected_ids.iter() {
            let key = rows.locate_row(row).unwrap().page_key();
            if distinct.insert(key) {
                expected_bytes += rows.inner.page_len(key).unwrap();
            }
        }
        let stats = rows.cache_stats();
        assert_eq!(stats.read_bytes as usize, expected_bytes);
        assert!(stats.reads as usize <= distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);
        let second = rows.prefetch_tokens(0, 99, &[0, 1]).unwrap();
        let lease = rows.wait_completed_lease(&second).unwrap();
        drop(lease);
        assert_eq!(rows.cache_stats().read_bytes, stats.read_bytes);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn source_error_and_epoch_invalidation_are_explicit() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: Some((0, 0)),
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let error = rows.wait_completed_lease(&ticket).unwrap_err();
        assert!(matches!(
            error,
            RowStoreError::Source(SourceError::Io { .. })
        ));
        // This is the forward-attempt abort boundary: the consumed ticket
        // may already report a source error, but reset_epoch must still drain
        // the exact epoch before a caller returns to its request loop.
        assert_eq!(rows.reset_epoch(Duration::from_secs(5)).unwrap(), 1);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_readers, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());

        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        rows.begin_epoch(1).unwrap();
        let error = rows.wait_completed_lease(&ticket).unwrap_err();
        assert!(matches!(
            error,
            RowStoreError::Canceled | RowStoreError::EpochMismatch { .. }
        ));
        // A canceled read follows the same abort path and must leave the
        // reader reusable at a later epoch rather than leaking its ticket.
        assert_eq!(rows.reset_epoch(Duration::from_secs(5)).unwrap(), 2);
        let retry = rows.prefetch_tokens(2, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&retry).unwrap();
        drop(lease);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_readers, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn coalesced_page_plan_is_ordered_and_deduplicated() {
        let pages = vec![
            PageKey { shard: 1, page: 2 },
            PageKey { shard: 0, page: 1 },
            PageKey { shard: 0, page: 0 },
            PageKey { shard: 0, page: 1 },
        ];
        let mut pages = pages;
        pages.sort_unstable();
        pages.dedup();
        assert_eq!(
            pages,
            vec![
                PageKey { shard: 0, page: 0 },
                PageKey { shard: 0, page: 1 },
                PageKey { shard: 1, page: 2 },
            ]
        );
    }
    /// One requested 320-byte row must not drag a multi-megabyte window out of
    /// the source.  The page unit is also the read unit, so this pins the
    /// per-row amplification that the lease wait depends on: with a 2 MiB
    /// window a 291-token chunk read ~9.4 GB for ~1.5 MB of rows, which cost
    /// 492 ms of page-cache copy with the GPU idle.
    #[test]
    fn read_window_stays_within_the_row_page() {
        // Big enough that hashed rows land in distinct windows the way the
        // production table spreads them.
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 32768),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_with_page_rows(
            metadata_with_head_size(4096, 65536),
            32768,
            ROWS_PER_PAGE,
            source.clone(),
        )
        .unwrap();
        let tokens: Vec<u32> = (0..4).collect();
        let ticket = rows.prefetch_tokens(0, 7, &tokens).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (index, &row_id) in expected_ids.iter().enumerate() {
            let row = lease.row_bytes(index).unwrap();
            assert_eq!(u16::from_le_bytes([row[0], row[1]]) as u64, row_id);
        }
        drop(lease);

        let mut distinct = std::collections::HashSet::new();
        let mut expected_bytes = 0usize;
        for &row in expected_ids.iter() {
            let key = rows.locate_row(row).unwrap().page_key();
            if distinct.insert(key) {
                expected_bytes += rows.inner.page_len(key).unwrap();
            }
        }
        let stats = rows.cache_stats();
        assert_eq!(stats.read_bytes as usize, expected_bytes);
        assert!(stats.reads as usize <= distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);

        let needed = expected_ids.iter().count() * ROW_BYTES;
        assert!(
            expected_bytes <= needed * 16,
            "requested {needed} bytes and read {expected_bytes} bytes"
        );
        assert!(rows.unload().unwrap().is_clean());
    }

    /// Q8F16 rows for the same fixture layout: five `[f16 scale][32 x i8]`
    /// blocks per row, each block a distinct constant so a mis-ordered block or
    /// a mis-scaled value is visible.
    fn rows_source_q8(shard_count: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        let blocks = ROW_WIDTH / Q8_BLOCK_WEIGHTS;
        (0..shard_count)
            .map(|shard| {
                let mut bytes = vec![0u8; rows_per_shard * blocks * Q8_BLOCK_BYTES];
                for row in 0..rows_per_shard {
                    let global = (shard * rows_per_shard + row) as f32;
                    for block in 0..blocks {
                        let offset = (row * blocks + block) * Q8_BLOCK_BYTES;
                        // scale = 1.0 for even blocks, 0.5 for odd ones.
                        let scale: f32 = if block % 2 == 0 { 1.0 } else { 0.5 };
                        let bits = f32_to_f16(scale);
                        bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                        for index in 0..Q8_BLOCK_WEIGHTS {
                            // Distinct, in-range quantized values per row/head.
                            let value = ((index as f32) - 16.0 + global) as i8;
                            bytes[offset + 2 + index] = value as u8;
                        }
                    }
                }
                bytes
            })
            .collect()
    }

    fn expected_q8_row(shard: usize, row: usize, rows_per_shard: usize) -> Vec<u8> {
        let blocks = ROW_WIDTH / Q8_BLOCK_WEIGHTS;
        let global = (shard * rows_per_shard + row) as f32;
        let mut out = vec![0u8; ROW_BYTES];
        for block in 0..blocks {
            let scale: f32 = if block % 2 == 0 { 1.0 } else { 0.5 };
            for index in 0..Q8_BLOCK_WEIGHTS {
                let value = scale * (((index as f32) - 16.0 + global) as i8) as f32;
                let cell = (block * Q8_BLOCK_WEIGHTS + index) * 2;
                out[cell..cell + 2].copy_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
            }
        }
        out
    }

    /// f16 encode for the fixture only (the reader decodes f16; this gives the
    /// fixture exact scales rather than rounded ones).
    fn f32_to_f16(value: f32) -> u16 {
        crate::llama::f32_to_f16(value)
    }

    #[test]
    fn q8_rows_decode_to_bf16_lease_rows() {
        let rows_per_shard = 64;
        let source = Arc::new(MemoryRowSource {
            shards: rows_source_q8(2, rows_per_shard),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_encoded(
            metadata(128),
            rows_per_shard,
            ROWS_PER_PAGE,
            RowEncoding::Q8F16,
            source.clone(),
        )
        .unwrap();
        assert_eq!(
            RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH) * ROWS_PER_PAGE,
            RowEncoding::Q8F16.page_bytes(ROW_WIDTH)
        );
        let ticket = rows.prefetch_tokens(0, 99, &[0, 1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (index, &row_id) in expected_ids.iter().enumerate() {
            let location = rows.locate_row(row_id).unwrap();
            let want = expected_q8_row(location.shard, location.local_row, rows_per_shard);
            assert_eq!(
                lease.row_bytes(index).unwrap(),
                want.as_slice(),
                "row {row_id} (request index {index}) decoded differently"
            );
        }
        drop(lease);
        // Encoded pages: the lease is still the 320-byte BF16 row shape the
        // device consumes, while the cache and the reads use the 170-byte form.
        let stats = rows.cache_stats();
        assert!(stats.read_bytes > 0);
        for row in expected_ids.iter() {
            let location = rows.locate_row(*row).unwrap();
            assert_eq!(
                rows.inner.page_len(location.page_key()).unwrap(),
                RowEncoding::Q8F16.page_bytes(ROW_WIDTH)
            );
        }
        assert!(rows.unload().unwrap().is_clean());
    }

    /// CPU cost of the Q8F16 row decode, which now sits on the reader thread.
    ///
    ///   cargo test -p hipfire-arch-qwen4 --lib -- q8_decode_cost_probe --ignored --nocapture
    ///
    /// Prints ns per row both ways: straight through the decoder, and inside a
    /// real prefetch over an in-memory shard set (which also carries the page
    /// copy the decode replaces for BF16).
    #[test]
    #[ignore = "CPU timing probe; no GPU or artifact needed"]
    fn q8_decode_cost_probe() {
        let encoded = rows_source_q8(1, 1).remove(0);
        let mut out = vec![0u8; ROW_BYTES];
        const DECODES: u32 = 200_000;
        let started = std::time::Instant::now();
        for _ in 0..DECODES {
            RowEncoding::Q8F16
                .decode_row(&encoded, &mut out)
                .expect("decode");
            std::hint::black_box(&out);
        }
        let ns_per_row = started.elapsed().as_secs_f64() * 1e9 / DECODES as f64;

        let mut arms = Vec::new();
        for encoding in [RowEncoding::Bf16, RowEncoding::Q8F16] {
            let rows_per_shard = 4096;
            let shards = match encoding {
                RowEncoding::Bf16 => rows_source(2, rows_per_shard),
                RowEncoding::Q8F16 => rows_source_q8(2, rows_per_shard),
            };
            let tokens: Vec<u32> = (0..256).collect();
            let rows = RowStore::from_test_source_encoded(
                metadata_with_head_size(512, (2 * rows_per_shard) as u64),
                rows_per_shard,
                ROWS_PER_PAGE,
                encoding,
                Arc::new(MemoryRowSource {
                    shards,
                    reads: AtomicUsize::new(0),
                    fail: None,
                }),
            )
            .unwrap();
            // Warm the page cache so the arm measures copy + decode, not reads.
            let ticket = rows.prefetch_tokens(0, 99, &tokens).unwrap();
            drop(rows.wait_completed_lease(&ticket).unwrap());
            let started = std::time::Instant::now();
            let ticket = rows.prefetch_tokens(0, 99, &tokens).unwrap();
            drop(rows.wait_completed_lease(&ticket).unwrap());
            let elapsed = started.elapsed().as_secs_f64();
            arms.push((
                encoding,
                elapsed * 1e6 / tokens.len() as f64,
                rows.cache_stats(),
            ));
            assert!(rows.unload().unwrap().is_clean());
        }
        println!(
            "q8-decode-probe decode_row={ns_per_row:.1}ns/row ({:.1}ns per 16-row token)",
            ns_per_row * f64::from(ROWS_PER_TOKEN as u32)
        );
        for (encoding, us_per_token, stats) in &arms {
            println!(
                "q8-decode-probe {:?} prefetch={us_per_token:.2}us/token (warm pages, {} reads, \
                 {} read_bytes)",
                encoding, stats.reads, stats.read_bytes
            );
        }
    }

    #[test]
    fn row_encoding_is_declared_by_dtype_not_inferred_from_extent() {
        assert_eq!(RowEncoding::from_dtype("BF16"), Some(RowEncoding::Bf16));
        assert_eq!(RowEncoding::from_dtype("Q8_0"), Some(RowEncoding::Q8F16));
        for foreign in ["F32", "F16", "MQ4G256V2", "MQ6G256V2", "I64", ""] {
            assert_eq!(RowEncoding::from_dtype(foreign), None, "{foreign}");
        }
        // The two tiers differ exactly by the row stride the shard must cover:
        // 320 bytes for a BF16 shard, 5 x 34 for a Q8F16 one.
        assert_eq!(RowEncoding::Bf16.encoded_row_bytes(ROW_WIDTH), ROW_BYTES);
        assert_eq!(
            RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH),
            (ROW_WIDTH / Q8_BLOCK_WEIGHTS) * Q8_BLOCK_BYTES
        );
        assert_eq!(RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH), 170);
        assert_eq!(RowEncoding::Q8F16.dtype_name(), "Q8_0");
    }

    #[test]
    fn direct_fill_handles_partial_pages_duplicates_and_cache_pressure() {
        for encoding in [RowEncoding::Bf16, RowEncoding::Q8F16] {
            assert_eq!(
                encoding.page_bytes(ROW_WIDTH) % encoding.encoded_row_bytes(ROW_WIDTH),
                0
            );
        }

        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_with_page_rows(
            metadata_with_head_size(8, 128),
            64,
            5,
            source.clone(),
        )
        .unwrap();
        assert_eq!(rows.inner.staging_bytes % ROW_BYTES, 0);
        {
            let mut state = rows.inner.state.lock().expect("test state mutex poisoned");
            state.cache.capacity_bytes = ROW_BYTES * 10;
        }

        let ticket = rows.prefetch_tokens(0, 999, &[0, 0]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        for token in 0..2 {
            for head in 0..ROWS_PER_TOKEN {
                let row = lease.row_bytes(token * ROWS_PER_TOKEN + head).unwrap();
                let value = u16::from_le_bytes([row[0], row[1]]) as u64;
                assert_eq!(value, lease.row_ids()[token * ROWS_PER_TOKEN + head]);
            }
        }
        let first_stats = rows.cache_stats();
        assert!(first_stats.reads > 1);
        assert_eq!(
            source.reads.load(Ordering::Acquire) as u64,
            first_stats.reads
        );
        assert!(first_stats.resident_bytes <= ROW_BYTES * 10);
        drop(lease);

        let ticket = rows.prefetch_tokens(0, 999, &[1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (head, &row_id) in lease.row_ids()[..ROWS_PER_TOKEN].iter().enumerate() {
            let row = lease.row_bytes(head).unwrap();
            let value = u16::from_le_bytes([row[0], row[1]]) as u64;
            assert_eq!(value, row_id);
        }
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        drop(lease);

        let stats = rows.cache_stats();
        assert!(stats.reads > first_stats.reads);
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);
        assert!(stats.cache_hits > 0);
        assert!(stats.cache_misses > 0);
        assert!(stats.evictions > 0);
        assert!(stats.resident_bytes <= ROW_BYTES * 10);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn epoch_reset_stales_lease_and_enforces_monotonic_epochs() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        rows.begin_epoch(0).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();

        std::thread::scope(|scope| {
            let resetter = scope.spawn(|| rows.reset_epoch(Duration::from_secs(1)));
            while rows.current_epoch() != 1 {
                std::thread::yield_now();
            }
            assert!(matches!(
                lease.validate(),
                Err(RowStoreError::LeaseStale { .. })
            ));
            assert!(matches!(
                lease.as_bytes(),
                Err(RowStoreError::LeaseStale { .. })
            ));
            let mut destination =
                vec![0u8; (lease.row_count() / ROWS_PER_TOKEN) * ROWS_PER_TOKEN * ROW_BYTES];
            assert!(matches!(
                lease.stage_into(&mut destination),
                Err(RowStoreError::LeaseStale { .. })
            ));
            assert!(matches!(
                lease.validate_after_upload(),
                Err(RowStoreError::LeaseStale { .. })
            ));
            drop(lease);
            assert_eq!(resetter.join().unwrap().unwrap(), 1);
        });

        assert!(matches!(
            rows.begin_epoch(1),
            Err(RowStoreError::EpochNotMonotonic {
                requested: 1,
                current: 1
            })
        ));
        rows.begin_epoch(2).unwrap();
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn cancel_releases_queued_ticket_slot_while_lease_holds_staging() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let first = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&first).unwrap();

        let mut queued = Vec::with_capacity(READER_QUEUE_CAPACITY);
        for _ in 0..READER_QUEUE_CAPACITY {
            queued.push(rows.prefetch_tokens(0, 99, &[0]).unwrap());
        }
        assert!(matches!(
            rows.prefetch_tokens(0, 99, &[0]),
            Err(RowStoreError::QueueFull)
        ));
        rows.cancel(&queued[0]).unwrap();
        let replacement = rows.prefetch_tokens(0, 99, &[0]).unwrap();

        drop(queued);
        drop(replacement);
        drop(lease);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn unload_waits_for_a_live_lease_before_releasing_the_reader() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();

        let unload = std::thread::spawn(move || rows.unload());
        for _ in 0..128 {
            assert!(!unload.is_finished());
            std::thread::yield_now();
        }
        drop(lease);
        assert!(unload.join().unwrap().unwrap().is_clean());
    }
}
