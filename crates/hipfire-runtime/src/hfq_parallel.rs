// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded, ordered HFQ reads for production model loading.
//!
//! Reader lanes fill caller-declared final host buffers concurrently. Results
//! are returned in plan order, so architecture loaders can preserve their
//! existing GPU allocation and pointer-table order exactly. Callers deliberately
//! submit small windows (at most four large jobs); this bounds live host memory
//! while still saturating the local NVMe on `hipx`/Strix Halo.
//!
//! Attached HFQ overlays are refused. Their tensor offsets belong to a second
//! file, while this reader owns one base path per lane. Architecture loaders must
//! retain their existing overlay-aware path as the fallback.

use crate::hfq::HfqFile;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Production default established by the DS4-shaped storage and canonical
/// upload screens. Four lanes reached 1.76-1.91 GB/s without changing GPU
/// allocation order; wider lane counts regressed on the same NVMe.
pub const HFQ_READER_LANES: usize = 4;

const DIRECT_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const DIRECT_ALIGNMENT: usize = 4096;

/// One model file, with a mapping used only to query page residency.
pub struct DirectSource {
    buffered: File,
    direct: Option<File>,
    view: Option<memmap2::Mmap>,
    identity: crate::model_source::SourceFileIdentity,
    // Retain the bounded pinned ring across tensors; repinning it per tensor
    // would put hipHostMalloc back on the critical path. The mutex also keeps
    // concurrent callers from reusing slots whose H2D is still pending.
    staging: Mutex<Option<(i32, Vec<hip_bridge::PinnedTransferBuffer>)>>,
}

impl DirectSource {
    pub fn open(hfq: &HfqFile) -> io::Result<Self> {
        if hfq.has_overlay() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "direct reads cannot cross HFQ overlays",
            ));
        }
        Self::open_path(hfq.path())
    }

    pub fn open_path(path: &Path) -> io::Result<Self> {
        let identity =
            crate::model_source::capture_file_identity(path).map_err(io::Error::other)?;
        let buffered = File::open(path)?;
        #[cfg(target_os = "linux")]
        let direct = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
                .ok()
        };
        #[cfg(not(target_os = "linux"))]
        let direct = None;
        // SAFETY: model files are immutable for the lifetime of their source.
        // No mapped byte is dereferenced: this view is only passed to mincore.
        let view = unsafe { memmap2::MmapOptions::new().map(&buffered) }.ok();
        crate::model_source::verify_path_identity(&identity).map_err(io::Error::other)?;
        Ok(Self {
            buffered,
            direct,
            view,
            identity,
            staging: Mutex::new(None),
        })
    }

    pub fn is_direct_available(&self) -> bool {
        self.direct.is_some()
    }

    pub fn file_identity(&self) -> &crate::model_source::SourceFileIdentity {
        &self.identity
    }

    fn validate(&self, offset: u64, len: usize) -> io::Result<()> {
        if offset
            .checked_add(len as u64)
            .is_none_or(|end| end > self.identity.len)
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "HFQ range at offset {offset}, length {len} exceeds EOF {}",
                    self.identity.len
                ),
            ));
        }
        Ok(())
    }

    fn resident(&self, offset: u64, len: usize) -> bool {
        #[cfg(target_os = "linux")]
        {
            let Some(view) = &self.view else { return false };
            if len == 0 {
                return true;
            }
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page <= 0 {
                return false;
            }
            let page = page as usize;
            let start = offset as usize / page * page;
            let end = (offset as usize + len).div_ceil(page);
            let pages = end - start / page;
            // At most one byte per page in a 16 MiB chunk.
            let mut residency = [0u8; DIRECT_CHUNK_BYTES / DIRECT_ALIGNMENT + 1];
            if pages > residency.len() {
                return false;
            }
            let result = unsafe {
                libc::mincore(
                    view.as_ptr().add(start) as *mut libc::c_void,
                    offset as usize + len - start,
                    residency.as_mut_ptr(),
                )
            };
            result == 0
                && residency[..pages].iter().filter(|&&v| v & 1 != 0).count() * 10 >= pages * 9
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (offset, len);
            false
        }
    }
}

/// Caller-owned writable host bytes, valid until the read returns.
pub struct HostRangeCopy {
    pub file_offset: u64,
    pub len: usize,
    pub dst: *mut u8,
}

/// Device destination, including an offset into its existing owner.
pub struct DeviceRangeCopy<'a> {
    pub file_offset: u64,
    pub len: usize,
    pub dst: &'a hip_bridge::DeviceBuffer,
    pub dst_offset: usize,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct DirectReadStats {
    pub bytes_direct: u64,
    pub bytes_buffered: u64,
    pub read_ns: u64,
    pub h2d_ns: u64,
    pub direct_read_ns: u64,
    pub buffered_read_ns: u64,
}

impl DirectReadStats {
    fn merge(&mut self, other: Self) {
        self.bytes_direct += other.bytes_direct;
        self.bytes_buffered += other.bytes_buffered;
        self.read_ns += other.read_ns;
        self.h2d_ns += other.h2d_ns;
        self.direct_read_ns += other.direct_read_ns;
        self.buffered_read_ns += other.buffered_read_ns;
    }
}

struct AlignedBounce {
    ptr: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedBounce {
    fn new() -> io::Result<Self> {
        let layout =
            std::alloc::Layout::from_size_align(DIRECT_CHUNK_BYTES, DIRECT_ALIGNMENT).unwrap();
        let ptr = std::ptr::NonNull::new(unsafe { std::alloc::alloc(layout) })
            .ok_or_else(|| io::Error::other("allocate direct read bounce buffer"))?;
        Ok(Self { ptr, layout })
    }
}

impl Drop for AlignedBounce {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[derive(Clone, Copy)]
struct DirectChunk {
    offset: u64,
    len: usize,
    dst: usize,
}

fn buffered_chunk(
    src: &DirectSource,
    dst: &mut [u8],
    offset: u64,
    stats: &mut DirectReadStats,
) -> io::Result<()> {
    let started = std::time::Instant::now();
    read_exact_at(&src.buffered, dst, offset)?;
    let ns = rdna_compute::load_phases::ns_since(started);
    stats.bytes_buffered += dst.len() as u64;
    stats.buffered_read_ns += ns;
    stats.read_ns += ns;
    Ok(())
}

fn read_direct_chunk(
    src: &DirectSource,
    chunk: DirectChunk,
    bounce: &mut Option<AlignedBounce>,
) -> io::Result<DirectReadStats> {
    let mut stats = DirectReadStats::default();
    // SAFETY: the host API validates ranges; its caller guarantees disjoint,
    // live writable destinations. Each chunk is handed to exactly one lane.
    let dst = unsafe { std::slice::from_raw_parts_mut(chunk.dst as *mut u8, chunk.len) };
    if src.direct.is_none() || src.resident(chunk.offset, chunk.len) {
        buffered_chunk(src, dst, chunk.offset, &mut stats)?;
        return Ok(stats);
    }
    let head = ((DIRECT_ALIGNMENT as u64 - chunk.offset % DIRECT_ALIGNMENT as u64)
        % DIRECT_ALIGNMENT as u64) as usize;
    let head = head.min(chunk.len);
    let middle = (chunk.len - head) / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
    if head != 0 {
        buffered_chunk(src, &mut dst[..head], chunk.offset, &mut stats)?;
    }
    if middle != 0 {
        let offset = chunk.offset + head as u64;
        let target = &mut dst[head..head + middle];
        let started = std::time::Instant::now();
        // O_DIRECT always lands in the lane's anonymous aligned bounce buffer,
        // never in the caller's destination: destinations are pinned
        // hipHostMalloc/GTT mappings the kernel cannot pin for direct I/O
        // (`pread` -> EFAULT, measured on gfx1201 hot-set owners). The copy
        // out of the bounce buffer runs on the lane, in parallel.
        if bounce.is_none() {
            *bounce = Some(AlignedBounce::new()?);
        }
        let buf = unsafe {
            std::slice::from_raw_parts_mut(bounce.as_mut().unwrap().ptr.as_ptr(), middle)
        };
        read_exact_at(src.direct.as_ref().unwrap(), buf, offset)?;
        target.copy_from_slice(buf);
        let ns = rdna_compute::load_phases::ns_since(started);
        stats.bytes_direct += middle as u64;
        stats.direct_read_ns += ns;
        stats.read_ns += ns;
    }
    let tail = head + middle;
    if tail < chunk.len {
        buffered_chunk(
            src,
            &mut dst[tail..],
            chunk.offset + tail as u64,
            &mut stats,
        )?;
    }
    Ok(stats)
}

/// Read with up to four lanes. Destinations must be writable, nonoverlapping,
/// and valid for the complete call. No read crosses the source EOF.
///
/// # Safety
/// Every `dst` must designate `len` writable bytes, exclusive to this call.
pub unsafe fn read_ranges_into_host(
    src: &DirectSource,
    copies: &[HostRangeCopy],
    lanes: usize,
) -> io::Result<DirectReadStats> {
    crate::model_source::verify_path_identity(&src.identity).map_err(io::Error::other)?;
    let mut chunks = Vec::new();
    for copy in copies {
        src.validate(copy.file_offset, copy.len)?;
        if copy.len != 0 && copy.dst.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "null host read destination",
            ));
        }
        for start in (0..copy.len).step_by(DIRECT_CHUNK_BYTES) {
            chunks.push(DirectChunk {
                offset: copy.file_offset + start as u64,
                len: (copy.len - start).min(DIRECT_CHUNK_BYTES),
                dst: copy.dst as usize + start,
            });
        }
    }
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..lanes.clamp(1, HFQ_READER_LANES).min(chunks.len()) {
            let chunks = &chunks;
            let cursor = &cursor;
            workers.push(scope.spawn(move || {
                let mut bounce = None;
                let mut stats = DirectReadStats::default();
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&chunk) = chunks.get(index) else {
                        break;
                    };
                    stats.merge(read_direct_chunk(src, chunk, &mut bounce)?);
                }
                Ok::<_, io::Error>(stats)
            }));
        }
        let mut stats = DirectReadStats::default();
        let mut error = None;
        for worker in workers {
            match worker
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("direct reader lane panicked")))
            {
                Ok(part) => stats.merge(part),
                Err(err) => {
                    if error.is_none() {
                        error = Some(err);
                    }
                }
            }
        }
        error.map_or(Ok(stats), Err)
    })
}

/// Double-buffered pinned batches: current reads overlap the preceding batch's
/// async copies. A batch's completion event is drained before reusing its slots.
pub fn upload_ranges_to_device(
    gpu: &mut rdna_compute::Gpu,
    src: &DirectSource,
    copies: &[DeviceRangeCopy<'_>],
    lanes: usize,
) -> Result<DirectReadStats, String> {
    let lanes = lanes.clamp(1, HFQ_READER_LANES);
    let mut chunks = Vec::new();
    for copy in copies {
        src.validate(copy.file_offset, copy.len)
            .map_err(|e| e.to_string())?;
        if copy
            .dst_offset
            .checked_add(copy.len)
            .is_none_or(|end| end > copy.dst.size())
        {
            return Err("direct upload exceeds device destination".into());
        }
        for start in (0..copy.len).step_by(DIRECT_CHUNK_BYTES) {
            chunks.push((
                copy.file_offset + start as u64,
                (copy.len - start).min(DIRECT_CHUNK_BYTES),
                copy.dst,
                copy.dst_offset + start,
            ));
        }
    }
    if chunks.is_empty() {
        return Ok(DirectReadStats::default());
    }
    gpu.bind_thread().map_err(|e| e.to_string())?;
    let hip = &gpu.hip;
    let device = hip.current_device().map_err(|e| e.to_string())?;
    let mut staging = src
        .staging
        .lock()
        .map_err(|_| "direct staging lock poisoned")?;
    if staging.as_ref().is_some_and(|(owner, _)| *owner != device) {
        *staging = None;
    }
    let (_, buffers) = staging.get_or_insert_with(|| (device, Vec::new()));
    while buffers.len() < 2 * lanes.min(chunks.len()) {
        buffers.push(
            hip.transfer_buffer(DIRECT_CHUNK_BYTES)
                .map_err(|e| e.to_string())?,
        );
    }
    let stream = hip.stream_create().map_err(|e| e.to_string())?;
    let mut events = Vec::new();
    let result = (|| {
        for _ in 0..2 {
            let start = hip.event_create().map_err(|e| e.to_string())?;
            let stop = match hip.event_create() {
                Ok(event) => event,
                Err(error) => {
                    let _ = hip.event_destroy(start);
                    return Err(error.to_string());
                }
            };
            events.push((start, stop, false));
        }
        let width = lanes.min(chunks.len());
        let mut stats = DirectReadStats::default();
        for (batch, window) in chunks.chunks(width).enumerate() {
            let bank = batch % 2;
            let (start_event, stop_event, pending) = &mut events[bank];
            if *pending {
                hip.event_synchronize(stop_event)
                    .map_err(|e| e.to_string())?;
                stats.h2d_ns += (hip
                    .event_elapsed_ms(start_event, stop_event)
                    .map_err(|e| e.to_string())? as f64
                    * 1_000_000.0) as u64;
                *pending = false;
            }
            let reads: Vec<_> = window
                .iter()
                .enumerate()
                .map(|(slot, &(offset, len, _, _))| HostRangeCopy {
                    file_offset: offset,
                    len,
                    dst: buffers[bank * width + slot].as_bytes_mut().as_mut_ptr(),
                })
                .collect();
            stats.merge(
                unsafe { read_ranges_into_host(src, &reads, lanes) }.map_err(|e| e.to_string())?,
            );
            hip.event_record(start_event, Some(&stream))
                .map_err(|e| e.to_string())?;
            for (slot, &(_, len, dst, offset)) in window.iter().enumerate() {
                // SAFETY: the non-owning view stays inside the checked destination.
                let view = unsafe {
                    hip_bridge::DeviceBuffer::from_host_mapped(
                        (dst.as_ptr() as *mut u8).add(offset).cast(),
                        len,
                    )
                };
                hip.memcpy_htod_async(
                    &view,
                    &buffers[bank * width + slot].as_bytes_mut()[..len],
                    &stream,
                )
                .map_err(|e| e.to_string())?;
            }
            hip.event_record(stop_event, Some(&stream))
                .map_err(|e| e.to_string())?;
            *pending = true;
        }
        for (start, stop, pending) in &mut events {
            if *pending {
                hip.event_synchronize(stop).map_err(|e| e.to_string())?;
                stats.h2d_ns += (hip
                    .event_elapsed_ms(start, stop)
                    .map_err(|e| e.to_string())? as f64
                    * 1_000_000.0) as u64;
                *pending = false;
            }
        }
        Ok(stats)
    })();
    // Even failed enqueues may leave copies outstanding. Never free pinned
    // storage until the stream is drained; retain it on an unrecoverable error.
    let drained = hip.stream_synchronize(&stream);
    if drained.is_err() {
        std::mem::forget(std::mem::take(buffers));
    }
    for (start, stop, _) in events {
        let _ = hip.event_destroy(start);
        let _ = hip.event_destroy(stop);
    }
    let _ = hip.stream_destroy(stream);
    drained.map_err(|e| e.to_string())?;
    result
}

#[derive(Clone, Debug)]
struct HfqReadSegment {
    source_offset: usize,
    len: usize,
    destination_offset: usize,
}

/// One canonical host output buffer. A job may be a single tensor or a packed
/// concatenation of many source tensors (for example DS4 expert `w1 || w3`).
#[derive(Clone, Debug)]
pub struct HfqReadJob {
    label: String,
    output_len: usize,
    segments: Vec<HfqReadSegment>,
}

impl HfqReadJob {
    /// Build a direct read job for one tensor.
    pub fn tensor(hfq: &HfqFile, name: &str) -> io::Result<Self> {
        Self::packed(hfq, name, [name])
    }

    /// Build one final buffer by concatenating `tensor_names` in the supplied
    /// order. Source tensors are validated before any reader thread starts.
    pub fn packed<I, S>(
        hfq: &HfqFile,
        label: impl Into<String>,
        tensor_names: I,
    ) -> io::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if hfq.has_overlay() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "parallel HFQ reads do not cross attached overlay files",
            ));
        }
        let mut output_len = 0usize;
        let mut segments = Vec::new();
        for name in tensor_names {
            let name = name.as_ref();
            let info = hfq.find_tensor_info(name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("HFQ tensor not found: {name}"),
                )
            })?;
            let next_len = output_len.checked_add(info.data_size).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "packed HFQ job size overflow")
            })?;
            segments.push(HfqReadSegment {
                source_offset: info.data_offset,
                len: info.data_size,
                destination_offset: output_len,
            });
            output_len = next_len;
        }
        if segments.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "parallel HFQ read job has no tensors",
            ));
        }
        Ok(Self {
            label: label.into(),
            output_len,
            segments,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn output_len(&self) -> usize {
        self.output_len
    }
}

#[derive(Debug)]
pub struct HfqReadResult {
    pub label: String,
    pub data: Vec<u8>,
}

/// Fill all jobs with up to four independent file handles and return them in
/// the exact order supplied. Callers then upload/install in their pre-existing
/// canonical order. The function never allocates GPU memory or changes model
/// state.
pub fn read_hfq_jobs_ordered(hfq: &HfqFile, jobs: &[HfqReadJob]) -> io::Result<Vec<HfqReadResult>> {
    if hfq.has_overlay() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "parallel HFQ reads do not cross attached overlay files",
        ));
    }
    // Honor the model's page-cache policy: on discrete GPUs the carrier keeps
    // pages resident across loads (fadvise(DONTNEED) here would force a full
    // disk re-read on every restart); UMA / opt-in eviction keeps dropping.
    read_jobs_from_path(hfq.path(), jobs, HFQ_READER_LANES, hfq.evicts_page_cache())
}

fn read_jobs_from_path(
    path: &Path,
    jobs: &[HfqReadJob],
    lanes: usize,
    evict_page_cache: bool,
) -> io::Result<Vec<HfqReadResult>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let lanes = lanes.clamp(1, jobs.len());
    let next_job = AtomicUsize::new(0);
    let slots = Arc::new(Mutex::new(
        (0..jobs.len())
            .map(|_| None)
            .collect::<Vec<Option<io::Result<HfqReadResult>>>>(),
    ));
    let path = PathBuf::from(path);

    std::thread::scope(|scope| {
        for _ in 0..lanes {
            let slots = Arc::clone(&slots);
            let path = path.clone();
            let next_job = &next_job;
            scope.spawn(move || {
                let file = match File::open(&path) {
                    Ok(file) => file,
                    Err(error) => {
                        loop {
                            let index = next_job.fetch_add(1, Ordering::Relaxed);
                            if index >= jobs.len() {
                                break;
                            }
                            slots.lock().expect("HFQ result slots poisoned")[index] =
                                Some(Err(io::Error::new(
                                    error.kind(),
                                    format!("open {}: {error}", path.display()),
                                )));
                        }
                        return;
                    }
                };
                advise_sequential(&file);
                loop {
                    let index = next_job.fetch_add(1, Ordering::Relaxed);
                    if index >= jobs.len() {
                        break;
                    }
                    let result = read_job(&file, &jobs[index], evict_page_cache);
                    slots.lock().expect("HFQ result slots poisoned")[index] = Some(result);
                }
            });
        }
    });

    let mut slots = slots.lock().expect("HFQ result slots poisoned");
    let mut ordered = Vec::with_capacity(jobs.len());
    for (index, slot) in slots.iter_mut().enumerate() {
        let result = slot.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("parallel HFQ reader did not fill job {index}"),
            )
        })??;
        debug_assert_eq!(result.label, jobs[index].label);
        ordered.push(result);
    }
    Ok(ordered)
}

fn read_job(file: &File, job: &HfqReadJob, evict_page_cache: bool) -> io::Result<HfqReadResult> {
    let mut data = vec![0u8; job.output_len];
    for segment in &job.segments {
        let end = segment
            .destination_offset
            .checked_add(segment.len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "HFQ segment overflow"))?;
        read_exact_at(
            file,
            &mut data[segment.destination_offset..end],
            segment.source_offset as u64,
        )?;
        if evict_page_cache {
            advise_dontneed(file, segment.source_offset, segment.len);
        }
    }
    Ok(HfqReadResult {
        label: job.label.clone(),
        data,
    })
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        let read = match file.read_at(buffer, offset) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result.map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("HFQ pread at offset {offset}: {error}"),
                )
            })?,
        };
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short parallel HFQ pread at offset {offset}"),
            ));
        }
        offset += read as u64;
        buffer = &mut buffer[read..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buffer)
}

#[cfg(unix)]
fn advise_sequential(file: &File) {
    use std::os::fd::AsRawFd;
    let _ = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
}

#[cfg(not(unix))]
fn advise_sequential(_file: &File) {}

#[cfg(unix)]
fn advise_dontneed(file: &File, offset: usize, len: usize) {
    use std::os::fd::AsRawFd;
    let _ = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
}

#[cfg(not(unix))]
fn advise_dontneed(_file: &File, _offset: usize, _len: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn job(label: &str, segments: &[(usize, usize)]) -> HfqReadJob {
        let mut destination_offset = 0usize;
        let segments = segments
            .iter()
            .map(|&(source_offset, len)| {
                let segment = HfqReadSegment {
                    source_offset,
                    len,
                    destination_offset,
                };
                destination_offset += len;
                segment
            })
            .collect();
        HfqReadJob {
            label: label.to_string(),
            output_len: destination_offset,
            segments,
        }
    }

    #[test]
    fn parallel_reads_preserve_job_and_segment_order() {
        let mut source = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0u8..=127).collect::<Vec<_>>();
        source.write_all(&bytes).unwrap();
        source.flush().unwrap();
        let jobs = vec![
            job("first", &[(16, 8), (0, 4)]),
            job("second", &[(80, 12)]),
            job("third", &[(40, 3), (60, 5), (120, 8)]),
            job("fourth", &[(8, 8)]),
        ];
        let results = read_jobs_from_path(source.path(), &jobs, 4, false).unwrap();
        assert_eq!(results[0].label, "first");
        assert_eq!(results[0].data, [&bytes[16..24], &bytes[0..4]].concat());
        assert_eq!(results[1].label, "second");
        assert_eq!(results[1].data, bytes[80..92]);
        assert_eq!(results[2].label, "third");
        assert_eq!(
            results[2].data,
            [&bytes[40..43], &bytes[60..65], &bytes[120..128]].concat()
        );
        assert_eq!(results[3].label, "fourth");
        assert_eq!(results[3].data, bytes[8..16]);
    }

    #[test]
    fn short_read_fails_closed() {
        let mut source = tempfile::NamedTempFile::new().unwrap();
        source.write_all(&[1, 2, 3, 4]).unwrap();
        source.flush().unwrap();
        let error =
            read_jobs_from_path(source.path(), &[job("short", &[(2, 8)])], 1, false).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    fn direct_fixture(dir: Option<&Path>) -> (tempfile::NamedTempFile, Vec<u8>) {
        let mut file = match dir {
            Some(dir) => tempfile::NamedTempFile::new_in(dir).unwrap(),
            None => tempfile::NamedTempFile::new().unwrap(),
        };
        let bytes: Vec<_> = (0..DIRECT_CHUNK_BYTES * 3 + 193)
            .map(|i| (i.wrapping_mul(37) ^ (i >> 13)) as u8)
            .collect();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();
        (file, bytes)
    }

    fn assert_ranges(source: &DirectSource, bytes: &[u8], lanes: usize) -> DirectReadStats {
        let ranges = [
            (3, 4191),
            (DIRECT_CHUNK_BYTES - 17, DIRECT_CHUNK_BYTES + 95),
            (bytes.len() - 177, 177),
            (0, 0),
        ];
        let mut destinations: Vec<_> = ranges.iter().map(|&(_, len)| vec![0xa5; len + 2]).collect();
        let copies: Vec<_> = ranges
            .iter()
            .zip(&mut destinations)
            .map(|(&(offset, len), dst)| HostRangeCopy {
                file_offset: offset as u64,
                len,
                dst: unsafe { dst.as_mut_ptr().add(1) },
            })
            .collect();
        let stats = unsafe { read_ranges_into_host(source, &copies, lanes) }.unwrap();
        for (&(offset, len), dst) in ranges.iter().zip(destinations) {
            assert_eq!(&dst[1..len + 1], &bytes[offset..offset + len]);
            assert_eq!(dst[0], 0xa5);
            assert_eq!(dst[len + 1], 0xa5);
        }
        assert_eq!(
            stats.bytes_direct + stats.bytes_buffered,
            ranges.iter().map(|&(_, len)| len as u64).sum::<u64>()
        );
        stats
    }

    #[test]
    fn direct_ranges_odd_chunk_boundary_eof_and_lane_counts() {
        let (file, bytes) = direct_fixture(None);
        let source = DirectSource::open_path(file.path()).unwrap();
        for lanes in [1, 4] {
            // Newly written pages are resident; buffered path must retain them.
            let stats = assert_ranges(&source, &bytes, lanes);
            #[cfg(target_os = "linux")]
            assert_eq!(stats.bytes_direct, 0);
            advise_dontneed(file.as_file(), 0, bytes.len());
            assert_ranges(&source, &bytes, lanes);
            // Rewarm all pages for the next resident arm.
            let mut warm = vec![0; bytes.len()];
            read_exact_at(file.as_file(), &mut warm, 0).unwrap();
        }
    }

    #[test]
    fn direct_ranges_mixed_residency() {
        let (file, bytes) = direct_fixture(None);
        let source = DirectSource::open_path(file.path()).unwrap();
        advise_dontneed(file.as_file(), 0, bytes.len());
        let cold = !source.resident(DIRECT_CHUNK_BYTES as u64, DIRECT_CHUNK_BYTES);
        let mut first = vec![0; DIRECT_CHUNK_BYTES];
        read_exact_at(file.as_file(), &mut first, 0).unwrap();
        // Drop the second and third chunks again, independent of readahead.
        advise_dontneed(file.as_file(), DIRECT_CHUNK_BYTES, DIRECT_CHUNK_BYTES * 2);
        for lanes in [1, 4] {
            let mut dst = vec![0; DIRECT_CHUNK_BYTES * 3];
            let copies = [HostRangeCopy {
                file_offset: 0,
                len: dst.len(),
                dst: dst.as_mut_ptr(),
            }];
            let stats = unsafe { read_ranges_into_host(&source, &copies, lanes) }.unwrap();
            assert_eq!(&dst, &bytes[..dst.len()]);
            #[cfg(target_os = "linux")]
            if cold && source.is_direct_available() {
                assert!(stats.bytes_direct >= DIRECT_CHUNK_BYTES as u64);
                assert!(stats.bytes_buffered >= DIRECT_CHUNK_BYTES as u64);
            }
        }
    }

    #[test]
    fn direct_ranges_buffered_only_and_shm() {
        let dir = Path::new("/dev/shm");
        if !dir.is_dir() {
            return;
        }
        let mut file = tempfile::NamedTempFile::new_in(dir).unwrap();
        let bytes: Vec<_> = (0..16387).map(|i| (i * 19) as u8).collect();
        file.write_all(&bytes).unwrap();
        file.as_file().sync_all().unwrap();
        let mut source = DirectSource::open_path(file.path()).unwrap();
        // Some tmpfs kernels accept O_DIRECT as a no-op. Force its absence as
        // well, so the fallback is exercised independently of kernel behavior.
        source.direct = None;
        for lanes in [1, 4] {
            let mut dst = vec![0; bytes.len() - 3];
            let stats = unsafe {
                read_ranges_into_host(
                    &source,
                    &[HostRangeCopy {
                        file_offset: 3,
                        len: dst.len(),
                        dst: dst.as_mut_ptr(),
                    }],
                    lanes,
                )
            }
            .unwrap();
            assert_eq!(&dst, &bytes[3..]);
            assert_eq!(stats.bytes_direct, 0);
            assert_eq!(stats.bytes_buffered, dst.len() as u64);
        }
    }

    #[test]
    fn direct_ranges_reject_eof_and_identity_change() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[1, 2, 3, 4]).unwrap();
        file.as_file().sync_all().unwrap();
        let source = DirectSource::open_path(file.path()).unwrap();
        let mut dst = [0xa5; 5];
        let error = unsafe {
            read_ranges_into_host(
                &source,
                &[HostRangeCopy {
                    file_offset: 1,
                    len: 5,
                    dst: dst.as_mut_ptr(),
                }],
                4,
            )
        }
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert!(error.to_string().contains("offset 1"));
        assert_eq!(dst, [0xa5; 5]);
        file.as_file().set_len(2).unwrap();
        assert!(unsafe {
            read_ranges_into_host(
                &source,
                &[HostRangeCopy {
                    file_offset: 0,
                    len: 1,
                    dst: dst.as_mut_ptr(),
                }],
                1,
            )
        }
        .is_err());
    }
}
