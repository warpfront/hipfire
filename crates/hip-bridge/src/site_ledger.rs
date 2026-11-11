// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-blocking HIP call-site ledger (`HIPFIRE_HOST_SITE_LEDGER=<dir>`, developer, lab census).
//!
//! Every host synchronization (`hipStreamSynchronize`, `hipEventSynchronize`,
//! `hipDeviceSynchronize`), copy (`hipMemcpy*`, `hipMemset*`) and allocation (`hipMalloc`,
//! `hipExtMallocWithFlags`, `hipFree`, `hipHostMalloc`/`hipHostFree`, VMM create/map/unmap/
//! release) issued through [`crate::HipRuntime`] appends one event to `<dir>/ledger-<pid>.tsv`:
//! start, duration (the host time blocked inside the call), API, bytes, the calling thread and the
//! return-address stack of the call. The scheduler marks served step boundaries
//! ([`mark_step`]), so a census divides every site's count and blocked time per decode step.
//! `<dir>/maps-<pid>.txt` is a copy of `/proc/self/maps` taken when the ledger opens; an
//! offline symbolizer maps every stack frame to `file:line` of a binary built with line tables.
//!
//! Off (the default) every hook is one relaxed atomic load. On, a recorded call also pays one
//! `backtrace(3)` walk *after* its timed region, so the blocked time is the call's own; the walk
//! still delays the host, which lets the device run further ahead, so on-ledger sync waits are a
//! lower bound of the unrecorded ones. Never enable it for a timing measurement.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// The ledger switch (developer): a directory to write the ledger into.
pub const HOST_SITE_LEDGER_ENV: &str = "HIPFIRE_HOST_SITE_LEDGER";

/// The recorded HIP API family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Api {
    StreamSync,
    EventSync,
    DeviceSync,
    /// Blocking `hipMemcpy` / `hipMemcpyPeer`.
    Memcpy,
    /// `hipMemcpyAsync` / `hipMemcpyPeerAsync` / `hipMemcpy2DAsync`.
    MemcpyAsync,
    Memset,
    MemsetAsync,
    Malloc,
    Free,
    HostMalloc,
    HostFree,
    VmmCreate,
    VmmMap,
    VmmUnmap,
    VmmRelease,
    /// A host-side wait that is not a HIP call (e.g. the EP AR host arrival gate).
    HostWait,
}

impl Api {
    fn name(self) -> &'static str {
        match self {
            Self::StreamSync => "hipStreamSynchronize",
            Self::EventSync => "hipEventSynchronize",
            Self::DeviceSync => "hipDeviceSynchronize",
            Self::Memcpy => "hipMemcpy",
            Self::MemcpyAsync => "hipMemcpyAsync",
            Self::Memset => "hipMemset",
            Self::MemsetAsync => "hipMemsetAsync",
            Self::Malloc => "hipMalloc",
            Self::Free => "hipFree",
            Self::HostMalloc => "hipHostMalloc",
            Self::HostFree => "hipHostFree",
            Self::VmmCreate => "hipMemCreate",
            Self::VmmMap => "hipMemMap",
            Self::VmmUnmap => "hipMemUnmap",
            Self::VmmRelease => "hipMemRelease",
            Self::HostWait => "hostWait",
        }
    }
}

/// Copy direction (`-` for non-copies).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dir {
    None,
    HtoD,
    DtoH,
    DtoD,
    Peer,
}

impl Dir {
    fn name(self) -> &'static str {
        match self {
            Self::None => "-",
            Self::HtoD => "h2d",
            Self::DtoH => "d2h",
            Self::DtoD => "d2d",
            Self::Peer => "peer",
        }
    }
}

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static NEXT_THREAD: AtomicU32 = AtomicU32::new(1);

struct Ledger {
    epoch: Instant,
    out: std::io::BufWriter<std::fs::File>,
    stacks: HashMap<u64, u32>,
}

static LEDGER: Mutex<Option<Ledger>> = Mutex::new(None);

thread_local! {
    static THREAD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[inline]
fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => open(),
    }
}

#[cold]
fn open() -> bool {
    // Only an installed process config is read (as `null_stream`): a call before
    // `install_process_config` records nothing and the next call re-checks.
    let Some(config) = hipfire_config::active_process_config() else {
        return false;
    };
    let dir = config
        .legacy_value(HOST_SITE_LEDGER_ENV)
        .filter(|d| !d.is_empty());
    let Some(dir) = dir else {
        STATE.store(OFF, Ordering::Relaxed);
        return false;
    };
    let mut guard = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        let pid = std::process::id();
        let dir = std::path::PathBuf::from(dir);
        let opened = std::fs::create_dir_all(&dir)
            .and_then(|()| {
                std::fs::copy("/proc/self/maps", dir.join(format!("maps-{pid}.txt")))?;
                std::fs::File::create(dir.join(format!("ledger-{pid}.tsv")))
            })
            .map(|file| Ledger {
                epoch: Instant::now(),
                out: std::io::BufWriter::with_capacity(1 << 20, file),
                stacks: HashMap::new(),
            });
        match opened {
            Ok(mut ledger) => {
                let wall = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                let _ = writeln!(ledger.out, "H\t{pid}\t{wall}");
                *guard = Some(ledger);
            }
            Err(error) => {
                eprintln!("[host-site-ledger] cannot open {}: {error}", dir.display());
                STATE.store(OFF, Ordering::Relaxed);
                return false;
            }
        }
    }
    STATE.store(ON, Ordering::Relaxed);
    true
}

/// A started recorded call: `None` when the ledger is off.
#[must_use]
pub struct Span(Option<Instant>);

/// Start timing a call (one relaxed load when the ledger is off).
#[inline]
pub fn begin() -> Span {
    if enabled() {
        Span(Some(Instant::now()))
    } else {
        Span(None)
    }
}

/// Finish a call begun with [`begin`] and attribute it to the caller's stack.
#[inline]
pub fn end(span: Span, api: Api, dir: Dir, bytes: usize) {
    if let Some(start) = span.0 {
        let dur = start.elapsed();
        record(start, dur.as_nanos() as u64, api, dir, bytes, None);
    }
}

/// Finish a call begun with [`begin`] under a fixed site tag (no stack walk): for
/// high-frequency host waits whose site is already known.
#[inline]
pub fn end_tagged(span: Span, api: Api, tag: &'static str) {
    if let Some(start) = span.0 {
        let dur = start.elapsed();
        record(start, dur.as_nanos() as u64, api, Dir::None, 0, Some(tag));
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn stack(frames: &mut [*mut c_void; 48]) -> usize {
    extern "C" {
        fn backtrace(buffer: *mut *mut c_void, size: std::ffi::c_int) -> std::ffi::c_int;
    }
    // SAFETY: glibc `backtrace(3)` writes at most `size` return addresses into `buffer`.
    let n = unsafe { backtrace(frames.as_mut_ptr(), frames.len() as std::ffi::c_int) };
    n.max(0) as usize
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn stack(_frames: &mut [*mut c_void; 48]) -> usize {
    0
}

fn thread_id(ledger: &mut Ledger) -> u32 {
    THREAD.with(|cell| {
        let id = cell.get();
        if id != 0 {
            return id;
        }
        let id = NEXT_THREAD.fetch_add(1, Ordering::Relaxed);
        cell.set(id);
        let name = std::thread::current().name().unwrap_or("?").to_owned();
        let _ = writeln!(ledger.out, "T\t{id}\t{name}");
        id
    })
}

#[cold]
fn record(start: Instant, dur_ns: u64, api: Api, dir: Dir, bytes: usize, tag: Option<&str>) {
    let mut frames = [std::ptr::null_mut(); 48];
    let depth = if tag.is_some() { 0 } else { stack(&mut frames) };
    // The stack includes the ledger's and hip-bridge's own frames; offline analysis skips
    // them, so the hash only needs to be stable per call path.
    let ips = &frames[..depth];
    let hash = ips.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &ip| {
        (h ^ ip as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let mut guard = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
    let Some(ledger) = guard.as_mut() else {
        return;
    };
    let tid = thread_id(ledger);
    let site = match tag {
        Some(tag) => format!("tag:{tag}"),
        None => {
            let next = ledger.stacks.len() as u32 + 1;
            let id = *ledger.stacks.entry(hash).or_insert(next);
            if id == next {
                let _ = write!(ledger.out, "S\t{id}\t");
                for (i, ip) in ips.iter().enumerate() {
                    let _ = write!(
                        ledger.out,
                        "{}{:x}",
                        if i == 0 { "" } else { "," },
                        *ip as usize
                    );
                }
                let _ = writeln!(ledger.out);
            }
            id.to_string()
        }
    };
    let t_ns = start.saturating_duration_since(ledger.epoch).as_nanos();
    let _ = writeln!(
        ledger.out,
        "E\t{t_ns}\t{tid}\t{}\t{}\t{dur_ns}\t{bytes}\t{site}",
        api.name(),
        dir.name()
    );
}

/// Mark the end of one served scheduler step: `lanes` decoding mesh lanes, `prefill_rows`
/// prompt rows in the step, `tokens` committed. Flushes the ledger.
pub fn mark_step(step: u64, lanes: usize, prefill_rows: usize, tokens: usize) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let mut guard = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
    let Some(ledger) = guard.as_mut() else {
        return;
    };
    let t_ns = now.saturating_duration_since(ledger.epoch).as_nanos();
    let _ = writeln!(
        ledger.out,
        "M\t{t_ns}\t{step}\t{lanes}\t{prefill_rows}\t{tokens}"
    );
    let _ = ledger.out.flush();
}
