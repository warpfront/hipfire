// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Idle-only page-cache prewarm of the PLE rows of a sidecar file.
//!
//! When the PLE rows live in a `.ple-q4` / `.ple-q8` sidecar (and no hot-row
//! pack serves them), reading the sidecar's data region sequentially once puts
//! it in the kernel page cache, so the scattered row reads of a request hit RAM
//! instead of the SSD.  The pages are ordinary clean file pages: under memory
//! pressure the kernel drops them rather than swapping.
//!
//! [`PleRowsPrewarm`] reads `[data_offset, data_offset + min(budget, data_bytes))`
//! in 8 MiB chunks with plain buffered `pread`s into one reused buffer (the data
//! is discarded), and only while the row store is idle: before every chunk it
//! waits on the row store's [`FetchActivity`] until no request is being served
//! (no whole-request `ServingGuard` and no per-fetch guard held) and the last
//! guard dropped [`FETCH_QUIET`] ago (the rule of the hot-row prewarm).  The
//! budget can be set after spawn ([`PleRowsPrewarm::set_budget`]); the worker
//! reads nothing before it.  Dropping the handle stops and joins the worker.
//!
//! # Public API (exact signatures)
//!
//! ```text
//! PleRowsPrewarm::spawn(path: &Path, data_offset: u64, data_bytes: u64,
//!                       activity: Arc<FetchActivity>) -> std::io::Result<PleRowsPrewarm>
//! PleRowsPrewarm::{set_budget(bytes: u64), stop(), prewarmed_bytes() -> u64,
//!                  complete() -> bool, failed() -> bool}
//! ```
//!
//! Off Unix `spawn` returns `ErrorKind::Unsupported` without opening the file
//! (the prewarm needs positional reads and `posix_fadvise`).
//!
//! Log lines (stderr, harness scripts key on the prefixes):
//! `  qwen4 ple prewarm: complete, <MiB> MiB in <secs> s ...`,
//! `  qwen4 ple prewarm: stopped, ...`, `  qwen4 ple prewarm: failed, ...`.

#[cfg(unix)]
use std::fs::File;
use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

use crate::external_rows::{FetchActivity, FETCH_QUIET};

/// Bytes per prewarm read.
#[cfg(unix)]
const PREWARM_CHUNK: usize = 8 << 20;
/// `budget` before [`PleRowsPrewarm::set_budget`].
#[cfg(unix)]
const BUDGET_UNSET: u64 = u64::MAX;
#[cfg(unix)]
const TAG: &str = "  qwen4 ple prewarm:";

#[cfg(unix)]
fn mib(bytes: u64) -> u64 {
    bytes >> 20
}

// Off Unix no worker is ever spawned, so the struct is only read by the
// platform-independent handle methods.
#[cfg_attr(not(unix), allow(dead_code))]
struct Shared {
    data_offset: u64,
    data_bytes: u64,
    budget: AtomicU64,
    prewarmed: AtomicU64,
    done: AtomicBool,
    failed: AtomicBool,
    stop: AtomicBool,
}

/// Background reader that warms the page cache with a sidecar's rows while the
/// row store is idle.  Stops and joins its worker on drop.
pub struct PleRowsPrewarm {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl PleRowsPrewarm {
    /// Open `path` and start the worker over its data region
    /// `[data_offset, data_offset + data_bytes)`.  It reads nothing until
    /// [`PleRowsPrewarm::set_budget`], and only while `activity` (the row
    /// store's, [`crate::external_rows::RowStore::activity`]) is idle.
    pub fn spawn(
        path: &Path,
        data_offset: u64,
        data_bytes: u64,
        activity: Arc<FetchActivity>,
    ) -> io::Result<PleRowsPrewarm> {
        Self::spawn_with_quiet(path, data_offset, data_bytes, activity, FETCH_QUIET)
    }

    /// [`PleRowsPrewarm::spawn`] with an explicit quiet period (tests).
    #[cfg(unix)]
    pub(crate) fn spawn_with_quiet(
        path: &Path,
        data_offset: u64,
        data_bytes: u64,
        activity: Arc<FetchActivity>,
        quiet: Duration,
    ) -> io::Result<PleRowsPrewarm> {
        data_offset.checked_add(data_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "prewarm range overflows u64")
        })?;
        let file = File::open(path)?;
        // Plain buffered reads; the hint only makes the kernel read ahead.
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }
        let shared = Arc::new(Shared {
            data_offset,
            data_bytes,
            budget: AtomicU64::new(BUDGET_UNSET),
            prewarmed: AtomicU64::new(0),
            done: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let worker_shared = shared.clone();
        let worker = std::thread::Builder::new()
            .name("ple-rows-prewarm".to_string())
            .spawn(move || worker_shared.run(&file, &activity, quiet))?;
        Ok(PleRowsPrewarm {
            shared,
            worker: Some(worker),
        })
    }

    /// Unsupported off Unix: the prewarm needs positional reads and
    /// `posix_fadvise`.  Fails before the file is touched and starts no worker.
    #[cfg(not(unix))]
    pub(crate) fn spawn_with_quiet(
        _path: &Path,
        _data_offset: u64,
        _data_bytes: u64,
        _activity: Arc<FetchActivity>,
        _quiet: Duration,
    ) -> io::Result<PleRowsPrewarm> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "PLE sidecar page-cache prewarm is unsupported on Windows (it needs Unix \
             positional reads and posix_fadvise); use a Unix host",
        ))
    }

    /// Bytes of the data region to read (capped at its size; 0 = none).  The
    /// worker starts reading at once, subject to idleness.
    pub fn set_budget(&self, bytes: u64) {
        self.shared
            .budget
            .store(bytes.min(self.shared.data_bytes), Ordering::Release);
    }

    /// Ask the worker to stop at its next chunk.  Dropping the handle does this
    /// and joins.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
    }

    /// Bytes of the data region read so far.
    pub fn prewarmed_bytes(&self) -> u64 {
        self.shared.prewarmed.load(Ordering::Relaxed)
    }

    /// The worker read its whole budget.
    pub fn complete(&self) -> bool {
        self.shared.done.load(Ordering::Acquire)
    }

    /// A read failed and the worker gave up.
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Acquire)
    }
}

impl Drop for PleRowsPrewarm {
    fn drop(&mut self) {
        self.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(unix)]
impl Shared {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn run(&self, file: &File, activity: &FetchActivity, quiet: Duration) {
        let budget = loop {
            if self.stopped() {
                return;
            }
            match self.budget.load(Ordering::Acquire) {
                BUDGET_UNSET => std::thread::sleep(Duration::from_millis(50)),
                budget => break budget,
            }
        };
        if budget == 0 {
            self.done.store(true, Ordering::Release);
            return;
        }
        let started = Instant::now();
        let mut paused = Duration::ZERO;
        let mut buffer = vec![0u8; PREWARM_CHUNK];
        let end = self.data_offset + budget;
        let mut offset = self.data_offset;
        while offset < end {
            let idle_from = Instant::now();
            if !activity.wait_idle(quiet, &self.stop) {
                eprintln!(
                    "{TAG} stopped, {} MiB of {} MiB in {:.1} s",
                    mib(offset - self.data_offset),
                    mib(budget),
                    started.elapsed().as_secs_f64()
                );
                return;
            }
            paused += idle_from.elapsed();
            let len = ((end - offset) as usize).min(PREWARM_CHUNK);
            if let Err(error) = file.read_exact_at(&mut buffer[..len], offset) {
                self.failed.store(true, Ordering::Release);
                eprintln!(
                    "{TAG} failed, read error at {} MiB of {} MiB: {error}",
                    mib(offset - self.data_offset),
                    mib(budget)
                );
                return;
            }
            offset += len as u64;
            self.prewarmed
                .store(offset - self.data_offset, Ordering::Relaxed);
        }
        self.done.store(true, Ordering::Release);
        let elapsed = started.elapsed();
        let reading = elapsed.saturating_sub(paused).as_secs_f64();
        eprintln!(
            "{TAG} complete, {} MiB in {:.1} s ({:.2} GB/s reading, {:.1} s paused for requests)",
            mib(budget),
            elapsed.as_secs_f64(),
            budget as f64 / 1e9 / reading.max(1e-9),
            paused.as_secs_f64(),
        );
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;

    const QUIET: Duration = Duration::from_millis(60);
    const OFFSET: u64 = 4096;
    /// Two full chunks and a partial one.
    const DATA: u64 = 2 * PREWARM_CHUNK as u64 + (4 << 20);

    fn data_file(len: u64) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.ple-q4");
        let mut file = File::create(&path).unwrap();
        file.write_all(&vec![0xA5u8; len as usize]).unwrap();
        (dir, path)
    }

    fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn spawn(path: &Path, activity: &Arc<FetchActivity>) -> PleRowsPrewarm {
        PleRowsPrewarm::spawn_with_quiet(path, OFFSET, DATA, activity.clone(), QUIET).unwrap()
    }

    /// Nothing is read before a budget; then exactly the budget (capped at the
    /// data size) is, once idle.
    #[test]
    fn reads_exactly_its_budget() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let activity = Arc::new(FetchActivity::new());
        let prewarm = spawn(&path, &activity);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!((prewarm.prewarmed_bytes(), prewarm.complete()), (0, false));
        let budget = PREWARM_CHUNK as u64 + (1 << 20);
        prewarm.set_budget(budget);
        wait_until("a partial budget", || prewarm.complete());
        assert_eq!(prewarm.prewarmed_bytes(), budget);
        assert!(!prewarm.failed());

        let activity = Arc::new(FetchActivity::new());
        let all = spawn(&path, &activity);
        all.set_budget(u64::MAX - 1);
        wait_until("the whole region", || all.complete());
        assert_eq!(all.prewarmed_bytes(), DATA);

        let activity = Arc::new(FetchActivity::new());
        let none = spawn(&path, &activity);
        none.set_budget(0);
        wait_until("a zero budget", || none.complete());
        assert_eq!(none.prewarmed_bytes(), 0);
    }

    /// A fetch in flight pauses the worker; it resumes `quiet` after the last
    /// fetch finished.
    #[test]
    fn pauses_while_a_fetch_is_in_flight() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let activity = Arc::new(FetchActivity::new());
        let guard = activity.busy();
        let prewarm = spawn(&path, &activity);
        prewarm.set_budget(DATA);
        std::thread::sleep(Duration::from_millis(250));
        assert_eq!(prewarm.prewarmed_bytes(), 0, "a fetch in flight pauses it");
        let released = Instant::now();
        drop(guard);
        std::thread::sleep(QUIET / 3);
        assert_eq!(prewarm.prewarmed_bytes(), 0, "quiet period not over yet");
        wait_until("resume", || prewarm.complete());
        assert!(released.elapsed() >= QUIET);
        assert_eq!(prewarm.prewarmed_bytes(), DATA);

        // A fetch between chunks pauses it again.
        let activity = Arc::new(FetchActivity::new());
        let prewarm = spawn(&path, &activity);
        let guard = activity.busy();
        prewarm.set_budget(DATA);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(prewarm.prewarmed_bytes(), 0);
        drop(guard);
        wait_until("second resume", || prewarm.complete());
    }

    /// A whole-request serving guard pauses the worker across the compute
    /// gaps between that request's fetches (longer than `quiet`, where the
    /// per-fetch guards alone would let it run); it resumes `quiet` after the
    /// request ends and completes.
    #[test]
    fn pauses_across_a_whole_request_and_resumes_after_it() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let activity = Arc::new(FetchActivity::new());
        let serving = activity.serving();
        let prewarm = spawn(&path, &activity);
        prewarm.set_budget(DATA);
        for _ in 0..3 {
            // A fetch of the request, then a compute gap far over `quiet`.
            drop(activity.busy());
            std::thread::sleep(QUIET * 3);
            assert_eq!(
                prewarm.prewarmed_bytes(),
                0,
                "no progress while a request is being served"
            );
        }
        assert!(!prewarm.complete());
        let released = Instant::now();
        drop(serving);
        std::thread::sleep(QUIET / 3);
        assert_eq!(prewarm.prewarmed_bytes(), 0, "quiet period not over yet");
        wait_until("resume after the request", || prewarm.complete());
        assert!(released.elapsed() >= QUIET);
        assert_eq!(prewarm.prewarmed_bytes(), DATA);
        assert!(!prewarm.failed());
    }

    /// A request that ends in an error or a cancellation releases its serving
    /// guard on every path, so the paused worker resumes and completes.
    #[test]
    fn resumes_after_a_failed_or_canceled_request() {
        let (_dir, path) = data_file(OFFSET + DATA);
        for outcome in ["error", "panic"] {
            let activity = Arc::new(FetchActivity::new());
            let prewarm = spawn(&path, &activity);
            let (held_tx, held_rx) = std::sync::mpsc::channel();
            let (finish_tx, finish_rx) = std::sync::mpsc::channel();
            let request = {
                let activity = activity.clone();
                std::thread::spawn(move || -> Result<(), String> {
                    let _serving = activity.serving();
                    let _fetch = activity.busy();
                    held_tx.send(()).unwrap();
                    finish_rx.recv().unwrap();
                    if outcome == "panic" {
                        panic!("request canceled mid-flight");
                    }
                    Err("request failed".to_string())
                })
            };
            held_rx.recv_timeout(Duration::from_secs(20)).unwrap();
            prewarm.set_budget(DATA);
            std::thread::sleep(QUIET * 3);
            assert_eq!(prewarm.prewarmed_bytes(), 0, "{outcome}: paused in flight");
            finish_tx.send(()).unwrap();
            let joined = request.join();
            assert_eq!(joined.is_err(), outcome == "panic");
            assert_eq!(activity.in_flight(), 0, "{outcome}: guards released");
            wait_until("resume after the failed request", || prewarm.complete());
            assert_eq!(prewarm.prewarmed_bytes(), DATA, "{outcome}");
        }
    }

    /// Dropping the handle ends a worker that is paused or waiting for a budget.
    #[test]
    fn drop_stops_and_joins() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let activity = Arc::new(FetchActivity::new());
        let waiting = spawn(&path, &activity);
        drop(waiting);

        let guard = activity.busy();
        let paused = spawn(&path, &activity);
        paused.set_budget(DATA);
        std::thread::sleep(Duration::from_millis(100));
        let started = Instant::now();
        drop(paused);
        assert!(started.elapsed() < Duration::from_secs(2));
        drop(guard);
    }

    /// A file shorter than the declared data region fails the worker instead of
    /// reporting completion.
    #[test]
    fn a_short_file_fails() {
        let (_dir, path) = data_file(OFFSET + PREWARM_CHUNK as u64 + 12345);
        let activity = Arc::new(FetchActivity::new());
        let prewarm = spawn(&path, &activity);
        prewarm.set_budget(DATA);
        wait_until("the read error", || prewarm.failed());
        assert!(!prewarm.complete());
        assert_eq!(prewarm.prewarmed_bytes(), PREWARM_CHUNK as u64);
    }

    #[test]
    fn spawn_reports_a_missing_file_and_a_bad_range() {
        let dir = tempfile::tempdir().unwrap();
        let activity = Arc::new(FetchActivity::new());
        let missing = dir.path().join("absent");
        assert!(PleRowsPrewarm::spawn(&missing, 0, 1, activity.clone()).is_err());
        let (_dir, path) = data_file(16);
        assert!(PleRowsPrewarm::spawn(&path, u64::MAX, 2, activity).is_err());
    }
}
