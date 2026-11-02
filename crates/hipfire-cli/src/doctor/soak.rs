// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire doctor --soak`: a multi-hour soak of the Strix Halo NPU trunk prefill route (GPU and
//! NPU busy together). One `npu_trunk=auto` daemon takes a loop of long cold prompts while the
//! kernel log is followed and an fsync'd heartbeat is written. The soak STOPS at the first fatal
//! kernel line, request hang, daemon exit or low memory: it never keeps stressing a machine that
//! has started to fault. The only writes are under the output directory; no setting changes.
//! Ported from the lab script `npu-trunk-soak.sh`.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use hipfire_client::Engine;
use serde_json::json;

use super::calibrate::{DaemonSession, Sizer};
use super::checks::Target;

/// Prompt sizes (rows) of one burst.
const SIZES: [usize; 4] = [9000, 16000, 32000, 57000];
/// Idle gap between bursts: clocks drop, so every burst also exercises the ramp.
const IDLE_BETWEEN_BURSTS: Duration = Duration::from_secs(20);
/// A request with no completion within this long is a hang.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1200);
const HEARTBEAT_PERIOD: Duration = Duration::from_secs(10);
/// Heartbeat gaps above this many seconds are reported.
const HEARTBEAT_GAP_S: i64 = 30;
/// Stop below this MemAvailable (4 GiB).
const MEM_FLOOR_KB: u64 = 4 * 1024 * 1024;
/// Late kernel lines (daemon teardown) arrive after the last request.
const SETTLE: Duration = Duration::from_secs(3);
const TAIL_LINES: usize = 40;
const MAX_HOURS: f64 = 24.0;
/// Context of the soak daemon; the largest prompt is 57000 rows.
const MAX_SEQ: u64 = 65536;

// ---------------------------------------------------------------------------------------------
// Kernel log classification
// ---------------------------------------------------------------------------------------------

/// Known benign message of hipfire <= 0.4.2.x on Linux >= 7.1: amdxdna no longer mmaps
/// user-pointer buffers. Counted, never fatal.
const BENIGN: [&str; 2] = [
    "amdxdna_insert_pages: Failed to mmap dma buf -22",
    "amdxdna_gem_obj_mmap: Failed insert pages, ret -22",
];
/// After an `amdgpu` token on the line.
const AMDGPU_ERRORS: [&str; 7] = [
    "page fault",
    "timeout",
    "GPU reset",
    "device lost",
    "MES failed",
    "sq_intr",
    "Fence fallback",
];
const SYSTEM_ERRORS: [&str; 7] = [
    "BUG:",
    "Oops",
    "hung_task",
    "soft lockup",
    "hard LOCKUP",
    "Out of memory",
    "watchdog: ",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FatalKind {
    Amdgpu,
    Amdxdna,
    /// BUG/Oops/hung task/lockup/RCU stall/OOM/watchdog.
    System,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LineClass {
    Fatal(FatalKind),
    Benign,
    Other,
}

/// `amdgpu.*(…|ring .* reset)` with plain string matching.
fn amdgpu_error(line: &str) -> bool {
    let Some(at) = line.find("amdgpu") else {
        return false;
    };
    let rest = &line[at + "amdgpu".len()..];
    AMDGPU_ERRORS.iter().any(|p| rest.contains(p))
        || rest
            .find("ring ")
            .is_some_and(|i| rest[i + "ring ".len()..].contains(" reset"))
}

/// `rcu_.*stall`.
fn rcu_stall(line: &str) -> bool {
    line.find("rcu_")
        .is_some_and(|i| line[i + "rcu_".len()..].contains("stall"))
}

pub(super) fn classify_line(line: &str) -> LineClass {
    if BENIGN.iter().any(|b| line.contains(b)) {
        LineClass::Benign
    } else if amdgpu_error(line) {
        LineClass::Fatal(FatalKind::Amdgpu)
    } else if line.contains("amdxdna") {
        LineClass::Fatal(FatalKind::Amdxdna)
    } else if SYSTEM_ERRORS.iter().any(|p| line.contains(p)) || rcu_stall(line) {
        LineClass::Fatal(FatalKind::System)
    } else {
        LineClass::Other
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    benign: u64,
    other: u64,
    amdgpu: u64,
    amdxdna: u64,
    system: u64,
}

impl Counts {
    fn add(&mut self, class: LineClass) {
        match class {
            LineClass::Benign => self.benign += 1,
            LineClass::Other => self.other += 1,
            LineClass::Fatal(FatalKind::Amdgpu) => self.amdgpu += 1,
            LineClass::Fatal(FatalKind::Amdxdna) => self.amdxdna += 1,
            LineClass::Fatal(FatalKind::System) => self.system += 1,
        }
    }

    fn fatal(&self) -> u64 {
        self.amdgpu + self.amdxdna + self.system
    }

    fn lines(&self) -> Vec<String> {
        vec![
            format!("kmsg benign amdxdna -22 lines: {}", self.benign),
            format!("kmsg other amdxdna lines:      {}", self.amdxdna),
            format!("kmsg amdgpu errors:            {}", self.amdgpu),
            format!("kmsg hangs/lockups/oops/oom:   {}", self.system),
        ]
    }
}

/// Live state fed by the kernel-log reader thread.
#[derive(Default)]
struct KernelWatch {
    state: Mutex<WatchState>,
    /// Set on the first fatal line; never cleared.
    fatal: AtomicBool,
    /// The follower's stdout reached EOF: the kernel is no longer being watched.
    ended: AtomicBool,
}

#[derive(Default)]
struct WatchState {
    counts: Counts,
    first_fatal: Option<String>,
    /// Last non-benign lines.
    tail: VecDeque<String>,
}

impl KernelWatch {
    fn observe(&self, line: &str) {
        let class = classify_line(line);
        let mut state = self.state.lock().unwrap();
        state.counts.add(class);
        if class == LineClass::Benign {
            return;
        }
        if state.tail.len() == TAIL_LINES {
            state.tail.pop_front();
        }
        state.tail.push_back(line.to_owned());
        if matches!(class, LineClass::Fatal(_)) && state.first_fatal.is_none() {
            state.first_fatal = Some(line.to_owned());
            self.fatal.store(true, Ordering::SeqCst);
        }
    }

    fn counts(&self) -> Counts {
        self.state.lock().unwrap().counts
    }

    fn first_fatal(&self) -> Option<String> {
        if !self.fatal.load(Ordering::SeqCst) {
            return None;
        }
        self.state.lock().unwrap().first_fatal.clone()
    }

    fn tail(&self) -> Vec<String> {
        self.state.lock().unwrap().tail.iter().cloned().collect()
    }
}

/// Drops the host name and PCI addresses of a kernel log line; keeps the timestamp prefix.
/// `2026-10-16T10:00:00.1+00:00 host kernel: amdgpu 0000:c0:00.0: x` ->
/// `2026-10-16T10:00:00.1+00:00 kernel: amdgpu <pci>: x`.
fn redact_line(line: &str) -> String {
    let line = line.trim_end();
    let scrubbed = match (line.split_once(' '), line.find(" kernel: ")) {
        (Some((stamp, _)), Some(at)) if stamp.starts_with(|c: char| c.is_ascii_digit()) => {
            format!("{stamp} {}", &line[at + 1..])
        }
        _ => line.to_owned(),
    };
    scrubbed
        .split(' ')
        .map(|token| {
            let core = token.trim_matches(|c: char| matches!(c, ':' | ',' | '(' | ')' | '[' | ']'));
            if is_pci_address(core) {
                token.replace(core, "<pci>")
            } else {
                token.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `0000:c0:00.0` or `c0:00.0`.
fn is_pci_address(s: &str) -> bool {
    let hex = |p: &str, n: usize| p.len() == n && p.bytes().all(|b| b.is_ascii_hexdigit());
    let parts: Vec<&str> = s.split(':').collect();
    let (bus, devfn) = match parts.as_slice() {
        [domain, bus, devfn] if hex(domain, 4) => (bus, devfn),
        [bus, devfn] => (bus, devfn),
        _ => return false,
    };
    let Some((dev, func)) = devfn.split_once('.') else {
        return false;
    };
    hex(bus, 2) && hex(dev, 2) && func.len() == 1 && func.bytes().all(|b| b.is_ascii_digit())
}

// ---------------------------------------------------------------------------------------------
// Time and /proc helpers
// ---------------------------------------------------------------------------------------------

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// (year, month, day) of a day count since 1970-01-01 (proleptic Gregorian).
pub(super) fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// `2026-10-16T12:34:56Z`.
fn fmt_utc(epoch: i64) -> String {
    let (y, m, d) = civil_from_days(epoch.div_euclid(86_400));
    let s = epoch.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        s / 3600,
        s % 3600 / 60,
        s % 60
    )
}

/// Inverse of [`fmt_utc`]; `None` for anything else.
fn parse_utc(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, sec) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    Some(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec)
}

/// Value in kB of `key:` in a /proc/meminfo text.
fn parse_meminfo_kb(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.strip_prefix(':')?;
        rest.split_whitespace().next()?.parse().ok()
    })
}

fn mem_available_kb() -> Option<u64> {
    parse_meminfo_kb(&fs::read_to_string("/proc/meminfo").ok()?, "MemAvailable")
}

fn load_1min() -> String {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn heartbeat_line(epoch: i64, load: &str, mem_kb: Option<u64>) -> String {
    let mem = mem_kb.map_or_else(|| "unknown".to_owned(), |kb| kb.to_string());
    format!("{} load={load} memavail_kb={mem}", fmt_utc(epoch))
}

/// Number of consecutive heartbeat lines more than `limit_s` seconds apart.
fn heartbeat_gaps(text: &str, limit_s: i64) -> usize {
    let mut gaps = 0;
    let mut prev: Option<i64> = None;
    for t in text
        .lines()
        .filter_map(|l| parse_utc(l.split_whitespace().next()?))
    {
        if prev.is_some_and(|p| t - p > limit_s) {
            gaps += 1;
        }
        prev = Some(t);
    }
    gaps
}

/// The run start: the timestamp of the first heartbeat line.
fn parse_start_epoch(heartbeat: &str) -> Option<i64> {
    parse_utc(
        heartbeat
            .lines()
            .find(|l| !l.trim().is_empty())?
            .split_whitespace()
            .next()?,
    )
}

fn write_synced(path: &Path, text: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

// ---------------------------------------------------------------------------------------------
// Kernel log follower and heartbeat
// ---------------------------------------------------------------------------------------------

fn probe(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

struct Follower {
    child: Child,
}

impl Follower {
    fn spawn(out: &Path, watch: &Arc<KernelWatch>) -> Result<Follower> {
        let mut command = if probe("journalctl", &["-k", "-n", "0", "--no-pager"]) {
            let mut c = Command::new("journalctl");
            c.args([
                "-k",
                "-f",
                "-n",
                "0",
                "-o",
                "short-iso-precise",
                "--no-pager",
            ]);
            c
        } else if probe("sudo", &["-n", "dmesg", "-T", "--since", "now"]) {
            // `--since now`: without it dmesg replays the whole ring buffer and old errors
            // would stop the soak at once.
            let mut c = Command::new("sudo");
            c.args(["-n", "dmesg", "-w", "-T", "--since", "now"]);
            c
        } else {
            bail!(
                "cannot read the kernel log: neither `journalctl -k` nor `sudo -n dmesg` works \
                 (add yourself to the systemd-journal group, or allow passwordless sudo dmesg)"
            );
        };
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("start the kernel log follower")?;
        let stdout = child
            .stdout
            .take()
            .context("kernel log follower has no stdout")?;
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(out.join("kmsg.log"))
            .context("create kmsg.log")?;
        let watch = Arc::clone(watch);
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut buf = Vec::new();
            while matches!(reader.read_until(b'\n', &mut buf), Ok(n) if n > 0) {
                let line = String::from_utf8_lossy(&buf).trim_end().to_owned();
                buf.clear();
                let _ = writeln!(log, "{line}");
                watch.observe(&line);
            }
            watch.ended.store(true, Ordering::SeqCst);
        });
        Ok(Follower { child })
    }

    fn stop(mut self) {
        // SIGTERM first: `sudo` relays it to dmesg, a SIGKILL would orphan dmesg.
        // SAFETY: plain kill(2) on our own child.
        unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM) };
        let end = Instant::now() + Duration::from_secs(2);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < end => thread::sleep(Duration::from_millis(50)),
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
    }
}

struct Heartbeat {
    stop: mpsc::Sender<()>,
    handle: thread::JoinHandle<()>,
}

impl Heartbeat {
    fn spawn(path: &Path) -> Result<Heartbeat> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .context("create heartbeat.log")?;
        let beat = |file: &mut File| {
            let line = heartbeat_line(unix_now(), &load_1min(), mem_available_kb());
            // A failed write must not end the soak: the gap shows in the summary.
            let _ = writeln!(file, "{line}");
            let _ = file.sync_all();
        };
        // The first line is the run start that `report` reads after a freeze.
        beat(&mut file);
        let (stop, rx) = mpsc::channel::<()>();
        let handle = thread::spawn(move || {
            while matches!(
                rx.recv_timeout(HEARTBEAT_PERIOD),
                Err(mpsc::RecvTimeoutError::Timeout)
            ) {
                beat(&mut file);
            }
        });
        Ok(Heartbeat { stop, handle })
    }

    fn stop(self) {
        drop(self.stop);
        let _ = self.handle.join();
    }
}

// ---------------------------------------------------------------------------------------------
// Request loop
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Request {
    rows: usize,
    ok: bool,
    wall_s: f64,
    prompt_tokens: Option<u64>,
}

#[derive(Debug)]
enum StopReason {
    KernelFatal(String),
    Hang { i: u64, rows: usize },
    DaemonExited,
    RequestFailed { i: u64, rows: usize, error: String },
    LowMemory(u64),
    FollowerEnded,
    StartFailed(String),
    Harness(String),
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelFatal(line) => write!(f, "kernel error: {}", redact_line(line)),
            Self::Hang { i, rows } => write!(
                f,
                "request {i} ({rows} rows) did not complete within {} s (hang); daemon killed",
                REQUEST_TIMEOUT.as_secs()
            ),
            Self::DaemonExited => f.write_str("daemon exited"),
            Self::RequestFailed { i, rows, error } => {
                write!(f, "request {i} ({rows} rows) failed: {error}")
            }
            Self::LowMemory(kb) => write!(
                f,
                "MemAvailable {} MiB < {} MiB",
                kb / 1024,
                MEM_FLOOR_KB / 1024
            ),
            Self::FollowerEnded => f.write_str(
                "the kernel log follower exited; the soak cannot watch the kernel any more",
            ),
            Self::StartFailed(e) => write!(f, "the daemon did not start: {e}"),
            Self::Harness(e) => write!(f, "soak harness error: {e}"),
        }
    }
}

/// `(pass, reason)`; `None` = the deadline was reached.
fn verdict_reason(stop: Option<&StopReason>, hours: f64) -> (bool, String) {
    match stop {
        None => (true, format!("ran {hours} h")),
        Some(reason) => (false, reason.to_string()),
    }
}

fn check_stop(engine: &Engine, watch: &KernelWatch) -> Option<StopReason> {
    if let Some(line) = watch.first_fatal() {
        return Some(StopReason::KernelFatal(line));
    }
    if watch.ended.load(Ordering::SeqCst) {
        return Some(StopReason::FollowerEnded);
    }
    if engine.exited() {
        return Some(StopReason::DaemonExited);
    }
    mem_available_kb()
        .filter(|kb| *kb < MEM_FLOOR_KB)
        .map(StopReason::LowMemory)
}

/// Runs `work`; if it takes longer than `timeout`, kills the daemon so `work` returns (an error)
/// instead of blocking forever. Returns the result and whether the watchdog fired.
fn with_watchdog<T>(
    engine: &Engine,
    timeout: Duration,
    work: impl FnOnce() -> Result<T>,
) -> (Result<T>, bool) {
    let hung = AtomicBool::new(false);
    let hung_ref = &hung;
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let result = thread::scope(|scope| {
        scope.spawn(move || {
            if matches!(
                done_rx.recv_timeout(timeout),
                Err(mpsc::RecvTimeoutError::Timeout)
            ) {
                hung_ref.store(true, Ordering::SeqCst);
                engine.terminate();
            }
        });
        let result = work();
        drop(done_tx);
        result
    });
    (result, hung.load(Ordering::SeqCst))
}

fn request_nonce(i: u64) -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos ^ i.rotate_left(48)
}

/// Sleeps `gap` in 1 s steps so a kernel error during the idle time stops the soak at once.
fn idle(
    gap: Duration,
    deadline: Instant,
    engine: &Engine,
    watch: &KernelWatch,
) -> Option<StopReason> {
    let end = (Instant::now() + gap).min(deadline);
    while Instant::now() < end {
        thread::sleep(Duration::from_secs(1).min(end.saturating_duration_since(Instant::now())));
        if let Some(reason) = check_stop(engine, watch) {
            return Some(reason);
        }
    }
    None
}

fn request_loop(
    session: &mut DaemonSession,
    engine: &Engine,
    watch: &KernelWatch,
    out: &Path,
    hours: f64,
    requests: &mut Vec<Request>,
) -> Result<Option<StopReason>> {
    let deadline = Instant::now() + Duration::from_secs_f64(hours * 3600.0);
    let mut tsv = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("requests.tsv"))
        .context("create requests.tsv")?;
    writeln!(tsv, "i\tutc\trows_req\tok\twall_s\tprompt_tokens")?;
    tsv.sync_all()?;
    let mut i = 0u64;
    let mut sizer = Sizer::new();
    loop {
        for rows in SIZES {
            if Instant::now() >= deadline {
                return Ok(None);
            }
            if let Some(reason) = check_stop(engine, watch) {
                return Ok(Some(reason));
            }
            i += 1;
            let (prompt, words) = sizer.prompt(request_nonce(i), rows);
            let started = Instant::now();
            let (result, hung) =
                with_watchdog(engine, REQUEST_TIMEOUT, || session.generate(&prompt, 1));
            let wall_s = started.elapsed().as_secs_f64();
            let prompt_tokens = result.as_ref().ok().map(|g| g.prompt_tokens);
            if let Some(tokens) = prompt_tokens {
                // Only the first observation lands: it fixes the tokens-per-word ratio.
                sizer.observe(tokens, words);
            }
            let ok = result.is_ok();
            writeln!(
                tsv,
                "{i}\t{}\t{rows}\t{}\t{wall_s:.2}\t{}",
                fmt_utc(unix_now()),
                u8::from(ok),
                prompt_tokens.map_or_else(|| "-".to_owned(), |t| t.to_string())
            )?;
            tsv.sync_all()?;
            write_synced(&out.join("npu-trunk.log"), &session.npu_lines().join("\n"))?;
            requests.push(Request {
                rows,
                ok,
                wall_s,
                prompt_tokens,
            });
            eprintln!(
                "soak: request {i} rows={rows} {} in {wall_s:.1} s",
                if ok { "ok" } else { "FAILED" }
            );
            if let Err(error) = result {
                // The kernel line is the cause of a failed request, when there is one.
                if let Some(line) = watch.first_fatal() {
                    return Ok(Some(StopReason::KernelFatal(line)));
                }
                return Ok(Some(if hung {
                    StopReason::Hang { i, rows }
                } else if engine.exited() {
                    StopReason::DaemonExited
                } else {
                    StopReason::RequestFailed {
                        i,
                        rows,
                        error: format!("{error:#}"),
                    }
                }));
            }
            if let Some(reason) = check_stop(engine, watch) {
                return Ok(Some(reason));
            }
        }
        if let Some(reason) = idle(IDLE_BETWEEN_BURSTS, deadline, engine, watch) {
            return Ok(Some(reason));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
struct SizeStat {
    rows: usize,
    requests: usize,
    median_wall_s: f64,
    prompt_tokens: Option<u64>,
}

/// Median of the ok requests of one size (the lower median, like the lab script).
fn size_stats(requests: &[Request]) -> Vec<SizeStat> {
    SIZES
        .iter()
        .filter_map(|&rows| {
            let ok: Vec<&Request> = requests.iter().filter(|r| r.ok && r.rows == rows).collect();
            let mut walls: Vec<f64> = ok.iter().map(|r| r.wall_s).collect();
            walls.sort_by(f64::total_cmp);
            let median_wall_s = *walls.get(((walls.len() + 1) / 2).checked_sub(1)?)?;
            Some(SizeStat {
                rows,
                requests: walls.len(),
                median_wall_s,
                prompt_tokens: ok.last().and_then(|r| r.prompt_tokens),
            })
        })
        .collect()
}

fn is_route_line(line: &str) -> bool {
    [
        "npu-trunk: mode",
        "npu-trunk: off",
        "npu-trunk: built-in",
        "npu-trunk: guard",
        "WARN npu-trunk",
    ]
    .iter()
    .any(|p| line.contains(p))
}

struct Summary<'a> {
    pass: bool,
    reason: &'a str,
    hours: f64,
    requests: &'a [Request],
    counts: Counts,
    first_fatal: Option<&'a str>,
    tail: &'a [String],
    gaps: usize,
    npu_lines: &'a [String],
}

impl Summary<'_> {
    fn text(&self) -> String {
        let mut out = vec![
            format!(
                "verdict: {} ({})",
                if self.pass { "PASS" } else { "FAIL" },
                self.reason
            ),
            format!(
                "mode: auto  hours: {}  requests: {}",
                self.hours,
                self.requests.len()
            ),
            "route lines:".to_owned(),
        ];
        let mut route: BTreeMap<String, usize> = BTreeMap::new();
        for line in self.npu_lines.iter().filter(|l| is_route_line(l)) {
            *route.entry(line.chars().take(240).collect()).or_default() += 1;
        }
        out.extend(
            route
                .iter()
                .take(12)
                .map(|(line, n)| format!("{n:>7} {line}")),
        );
        let planned = self
            .npu_lines
            .iter()
            .filter(|l| l.contains("npu-trunk: rows="))
            .count();
        out.push(format!("planned requests (npu-trunk: rows=...): {planned}"));
        out.extend(self.counts.lines());
        out.push("per size (median wall s, prompt tokens):".to_owned());
        for s in size_stats(self.requests) {
            out.push(format!(
                "  {}: median {:.2} s over {} requests, {} prompt tokens",
                s.rows,
                s.median_wall_s,
                s.requests,
                s.prompt_tokens
                    .map_or_else(|| "?".to_owned(), |t| t.to_string())
            ));
        }
        out.push(format!(
            "heartbeat gaps > {HEARTBEAT_GAP_S} s: {}",
            self.gaps
        ));
        if !self.pass {
            out.push("kernel log tail:".to_owned());
            out.extend(self.tail.iter().cloned());
        }
        out.join("\n") + "\n"
    }

    fn json(&self, host: &str) -> serde_json::Value {
        let verdict = if self.pass { "PASS" } else { "FAIL" };
        json!({
            "schema": "hipfire-soak/1",
            "verdict": verdict,
            "reason": self.reason,
            "hours": self.hours,
            "requests": self.requests.len(),
            "failed_requests": self.requests.iter().filter(|r| !r.ok).count(),
            "per_size": size_stats(self.requests).iter().map(|s| json!({
                "rows": s.rows,
                "requests": s.requests,
                "median_wall_s": (s.median_wall_s * 100.0).round() / 100.0,
                "prompt_tokens": s.prompt_tokens,
            })).collect::<Vec<_>>(),
            "benign_count": self.counts.benign,
            "fatal_count": self.counts.fatal(),
            "heartbeat_gaps": self.gaps,
            "fatal_line": self.first_fatal.map(redact_line),
            "host": host,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

pub(crate) fn plan_lines(hours: f64) -> Vec<String> {
    let sizes = SIZES.map(|s| s.to_string()).join("/");
    vec![
        format!("Soak plan: {hours} h of NPU trunk prefill under load."),
        "  - starts one npu_trunk=auto daemon and loads the model".to_owned(),
        format!(
            "  - loops cold prompts of {sizes} rows, {} s idle between bursts, for {hours} h",
            IDLE_BETWEEN_BURSTS.as_secs()
        ),
        "  - uses the GPU and the NPU together".to_owned(),
        "  - watches the kernel log and writes a heartbeat every 10 s".to_owned(),
        format!(
            "  - stops at the first kernel error, request hang (> {} s), daemon exit or \
             MemAvailable < 4 GiB",
            REQUEST_TIMEOUT.as_secs()
        ),
        "  - the amdxdna `-22` mmap message pair is counted as benign, never fatal".to_owned(),
        "  - writes only into the output directory; changes no setting".to_owned(),
        "  - after a freeze and reboot: hipfire doctor --soak-report <dir>".to_owned(),
    ]
}

fn prepare_out(out: &Path) -> Result<()> {
    if out.exists() {
        if !out.is_dir() {
            bail!("{} exists and is not a directory", out.display());
        }
        if fs::read_dir(out)?.next().is_some() {
            bail!(
                "output directory {} is not empty; choose a new --out",
                out.display()
            );
        }
        Ok(())
    } else {
        fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))
    }
}

/// Runs the soak; `Ok(true)` = PASS.
pub(crate) fn run(paths: &crate::Paths, target: &Target, hours: f64, out: &Path) -> Result<bool> {
    if !(hours > 0.0 && hours <= MAX_HOURS) {
        bail!("--hours must be in (0, {MAX_HOURS}]");
    }
    prepare_out(out)?;
    let watch = Arc::new(KernelWatch::default());
    let follower = Follower::spawn(out, &watch)?;
    let heartbeat = match Heartbeat::spawn(&out.join("heartbeat.log")) {
        Ok(h) => h,
        Err(e) => {
            follower.stop();
            return Err(e);
        }
    };
    eprintln!("soak: following the kernel log; loading the model with npu_trunk=auto ...");

    let mut requests = Vec::new();
    let mut npu_lines = Vec::new();
    let outcome: Result<Option<StopReason>> =
        match DaemonSession::start(paths, target, "auto", false, MAX_SEQ) {
            Err(e) => Ok(Some(StopReason::StartFailed(format!("{e:#}")))),
            Ok(mut session) => {
                let engine = session.engine_handle();
                let result = request_loop(&mut session, &engine, &watch, out, hours, &mut requests);
                npu_lines = session.npu_lines();
                let _ = session.finish();
                result
            }
        };
    let mut stop = outcome.unwrap_or_else(|e| Some(StopReason::Harness(format!("{e:#}"))));
    let _ = write_synced(&out.join("npu-trunk.log"), &npu_lines.join("\n"));

    // Late kernel lines (daemon teardown) still count.
    thread::sleep(SETTLE);
    if stop.is_none() {
        stop = watch.first_fatal().map(StopReason::KernelFatal);
    }
    follower.stop();
    heartbeat.stop();

    let (pass, reason) = verdict_reason(stop.as_ref(), hours);
    let heartbeat_text = fs::read_to_string(out.join("heartbeat.log")).unwrap_or_default();
    let first_fatal = watch.first_fatal();
    let tail = watch.tail();
    let summary = Summary {
        pass,
        reason: &reason,
        hours,
        requests: &requests,
        counts: watch.counts(),
        first_fatal: first_fatal.as_deref(),
        tail: &tail,
        gaps: heartbeat_gaps(&heartbeat_text, HEARTBEAT_GAP_S),
        npu_lines: &npu_lines,
    };
    let text = summary.text();
    let host = hipfire_arch_qwen4::npu_trunk::host_facts(Some(target.gpu_pci.as_str()));
    let json = serde_json::to_string_pretty(&summary.json(&host))? + "\n";
    write_synced(&out.join("summary.txt"), &text)?;
    write_synced(&out.join("summary.json"), &json)?;
    eprint!("{text}");
    Ok(pass)
}

fn build_report(
    start: i64,
    now: i64,
    last_heartbeat: Option<&str>,
    last_request: Option<&str>,
    kmsg: Option<&str>,
) -> String {
    let mut out = vec![
        format!(
            "report {}: previous boot after a soak started {}",
            fmt_utc(now),
            fmt_utc(start)
        ),
        format!("last heartbeat: {}", last_heartbeat.unwrap_or("none")),
        format!("last request: {}", last_request.unwrap_or("none")),
    ];
    let Some(kmsg) = kmsg else {
        out.push("previous-boot kernel log: unavailable (volatile journal? run as a user that can read it)".to_owned());
        return out.join("\n") + "\n";
    };
    let mut counts = Counts::default();
    let mut fatal = Vec::new();
    let mut lines = 0usize;
    for line in kmsg.lines().filter(|l| !l.starts_with("-- ")) {
        lines += 1;
        let class = classify_line(line);
        counts.add(class);
        if matches!(class, LineClass::Fatal(_)) {
            fatal.push(line);
        }
    }
    out.push(format!(
        "previous-boot kernel lines since the start: {lines}"
    ));
    out.extend(counts.lines());
    out.push(format!("fatal kernel lines: {}", fatal.len()));
    out.extend(fatal.iter().take(20).map(|l| format!("  {l}")));
    out.push("last kernel lines:".to_owned());
    let all: Vec<&str> = kmsg.lines().collect();
    out.extend(
        all[all.len().saturating_sub(TAIL_LINES)..]
            .iter()
            .map(|l| (*l).to_owned()),
    );
    out.join("\n") + "\n"
}

fn last_line(text: &str) -> Option<&str> {
    text.lines().rev().find(|l| !l.trim().is_empty())
}

/// After a freeze and reboot: collects the previous boot's kernel log into `dir`.
pub(crate) fn report(dir: &Path) -> Result<()> {
    let heartbeat = fs::read_to_string(dir.join("heartbeat.log")).with_context(|| {
        format!(
            "{} is not a soak directory (no heartbeat.log)",
            dir.display()
        )
    })?;
    let start = parse_start_epoch(&heartbeat)
        .with_context(|| "heartbeat.log has no timestamped first line")?;
    let since = format!("@{start}");
    let journal = Command::new("journalctl")
        .args(["-k", "-b", "-1", "--since", since.as_str()])
        .args(["-o", "short-iso-precise", "--no-pager"])
        .stdin(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    write_synced(
        &dir.join("prev-boot-kmsg.log"),
        journal
            .as_deref()
            .unwrap_or("journalctl -b -1 unavailable (volatile journal?)\n"),
    )?;
    let requests = fs::read_to_string(dir.join("requests.tsv")).unwrap_or_default();
    let text = build_report(
        start,
        unix_now(),
        last_line(&heartbeat),
        last_line(&requests),
        journal.as_deref(),
    );
    write_synced(&dir.join("report.txt"), &text)?;
    print!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOST: &str = "2026-10-16T10:00:00.123456+00:00 hipx kernel:";

    fn line(msg: &str) -> String {
        format!("{HOST} {msg}")
    }

    #[test]
    fn every_fatal_alternative_is_fatal() {
        let amdgpu = [
            "amdgpu 0000:c0:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:40 vmid:8)",
            "amdgpu 0000:c0:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=12",
            "amdgpu 0000:c0:00.0: amdgpu: GPU reset begin!",
            "amdgpu 0000:c0:00.0: amdgpu: device lost from bus!",
            "amdgpu 0000:c0:00.0: amdgpu: MES failed to respond to msg=REMOVE_QUEUE",
            "amdgpu 0000:c0:00.0: amdgpu: sq_intr: error, detail 0x1",
            "amdgpu 0000:c0:00.0: amdgpu: ring comp_1.0.0 reset succeeded",
            "amdgpu 0000:c0:00.0: amdgpu: Fence fallback timer expired on ring sdma0",
        ];
        for msg in amdgpu {
            assert_eq!(
                classify_line(&line(msg)),
                LineClass::Fatal(FatalKind::Amdgpu),
                "{msg}"
            );
        }
        assert_eq!(
            classify_line(&line("amdxdna 0000:c1:00.1: [E] aie2_sched: job timed out")),
            LineClass::Fatal(FatalKind::Amdxdna)
        );
        let system = [
            "BUG: unable to handle page fault for address: 0000000000001234",
            "Oops: general protection fault, probably for non-canonical address",
            "Kernel panic - not syncing: hung_task: blocked tasks",
            "watchdog: BUG: soft lockup - CPU#3 stuck for 26s! [swapper/3:0]",
            "watchdog: Watchdog detected hard LOCKUP on cpu 4",
            "rcu: INFO: rcu_sched self-detected stall on CPU",
            "Out of memory: Killed process 4242 (hipfire-daemon)",
            "watchdog: something else",
        ];
        for msg in system {
            assert_eq!(
                classify_line(&line(msg)),
                LineClass::Fatal(FatalKind::System),
                "{msg}"
            );
        }
    }

    #[test]
    fn benign_pair_is_never_fatal() {
        for msg in [
            "amdxdna 0000:c1:00.1: amdxdna_insert_pages: Failed to mmap dma buf -22",
            "amdxdna 0000:c1:00.1: amdxdna_gem_obj_mmap: Failed insert pages, ret -22",
        ] {
            assert_eq!(classify_line(&line(msg)), LineClass::Benign, "{msg}");
        }
        // Another -22 amdxdna message is not in the pair.
        assert_eq!(
            classify_line(&line(
                "amdxdna 0000:c1:00.1: amdxdna_gem_obj_mmap: other, ret -22"
            )),
            LineClass::Fatal(FatalKind::Amdxdna)
        );
    }

    #[test]
    fn unrelated_lines_are_other() {
        for msg in [
            "amdgpu 0000:c0:00.0: amdgpu: SMU is initialized successfully!",
            "amdgpu 0000:c0:00.0: amdgpu: ring gfx_0.0.0 uses VM inv eng 0 on hub 0",
            "usb 1-1: new high-speed USB device number 3 using xhci_hcd",
        ] {
            assert_eq!(classify_line(&line(msg)), LineClass::Other, "{msg}");
        }
    }

    #[test]
    fn watch_counts_and_first_fatal() {
        let watch = KernelWatch::default();
        watch.observe(&line(
            "amdxdna 0000:c1:00.1: amdxdna_insert_pages: Failed to mmap dma buf -22",
        ));
        watch.observe(&line(
            "amdgpu 0000:c0:00.0: amdgpu: SMU is initialized successfully!",
        ));
        assert!(watch.first_fatal().is_none());
        watch.observe(&line("amdgpu 0000:c0:00.0: amdgpu: GPU reset begin!"));
        watch.observe(&line("BUG: later"));
        let counts = watch.counts();
        assert_eq!(
            (counts.benign, counts.other, counts.amdgpu, counts.system),
            (1, 1, 1, 1)
        );
        assert!(watch.first_fatal().unwrap().contains("GPU reset"));
        assert_eq!(watch.tail().len(), 3, "benign lines stay out of the tail");
    }

    #[test]
    fn redaction_drops_host_and_pci_keeps_timestamp() {
        let red = redact_line(&line("amdgpu 0000:c0:00.0: amdgpu: GPU reset begin!"));
        assert_eq!(
            red,
            "2026-10-16T10:00:00.123456+00:00 kernel: amdgpu <pci>: amdgpu: GPU reset begin!"
        );
        assert!(!red.contains("hipx"));
    }

    #[test]
    fn utc_round_trip_and_known_value() {
        assert_eq!(fmt_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(parse_utc("2026-10-16T12:34:56Z"), Some(1_792_154_096));
        assert_eq!(fmt_utc(1_792_154_096), "2026-10-16T12:34:56Z");
        for t in [0, 951_782_400, 1_709_251_199, 1_792_154_096] {
            assert_eq!(parse_utc(&fmt_utc(t)), Some(t));
        }
        assert_eq!(parse_utc("2026-10-16 12:34:56"), None);
        assert_eq!(parse_utc("garbage"), None);
    }

    #[test]
    fn heartbeat_gap_detection() {
        let text = "2026-10-16T10:00:00Z load=1 memavail_kb=1\n\
                    2026-10-16T10:00:10Z load=1 memavail_kb=1\n\
                    2026-10-16T10:00:41Z load=1 memavail_kb=1\n\
                    2026-10-16T10:00:51Z load=1 memavail_kb=1\n\
                    2026-10-16T10:02:00Z load=1 memavail_kb=1\n";
        assert_eq!(heartbeat_gaps(text, 30), 2);
        assert_eq!(heartbeat_gaps(text, 100), 0);
        assert_eq!(heartbeat_gaps("", 30), 0);
    }

    #[test]
    fn heartbeat_line_format() {
        assert_eq!(
            heartbeat_line(1_792_154_096, "0.52", Some(98_765_432)),
            "2026-10-16T12:34:56Z load=0.52 memavail_kb=98765432"
        );
        assert!(heartbeat_line(0, "unknown", None).ends_with("memavail_kb=unknown"));
    }

    #[test]
    fn meminfo_parsing() {
        let sample = "MemTotal:       126976192 kB\nMemFree:         4096000 kB\n\
                      MemAvailable:   100663296 kB\nBuffers:           12345 kB\n";
        assert_eq!(parse_meminfo_kb(sample, "MemAvailable"), Some(100_663_296));
        assert_eq!(parse_meminfo_kb(sample, "MemTotal"), Some(126_976_192));
        assert_eq!(parse_meminfo_kb(sample, "Cached"), None);
        assert_eq!(
            parse_meminfo_kb("MemAvailableX: 5 kB\n", "MemAvailable"),
            None
        );
    }

    #[test]
    fn verdict_and_reason() {
        assert_eq!(verdict_reason(None, 4.0), (true, "ran 4 h".to_owned()));
        let hang = StopReason::Hang { i: 7, rows: 32000 };
        let (pass, reason) = verdict_reason(Some(&hang), 4.0);
        assert!(!pass);
        assert!(reason.contains("request 7 (32000 rows)") && reason.contains("1200 s"));
        let kernel = StopReason::KernelFatal(line("amdgpu 0000:c0:00.0: amdgpu: GPU reset begin!"));
        let (pass, reason) = verdict_reason(Some(&kernel), 4.0);
        assert!(!pass && reason.contains("GPU reset") && !reason.contains("hipx"));
        let (_, mem) = verdict_reason(Some(&StopReason::LowMemory(3 * 1024 * 1024)), 1.0);
        assert_eq!(mem, "MemAvailable 3072 MiB < 4096 MiB");
        assert_eq!(MEM_FLOOR_KB, 4_194_304);
    }

    fn req(_i: u64, rows: usize, ok: bool, wall_s: f64) -> Request {
        Request {
            rows,
            ok,
            wall_s,
            prompt_tokens: ok.then_some(rows as u64 + 5),
        }
    }

    #[test]
    fn per_size_median_ignores_failures() {
        let requests = [
            req(1, 9000, true, 3.0),
            req(2, 16000, true, 9.0),
            req(3, 9000, true, 1.0),
            req(4, 9000, true, 2.0),
            req(5, 9000, false, 99.0),
        ];
        let stats = size_stats(&requests);
        assert_eq!(stats.len(), 2);
        assert_eq!(
            stats[0],
            SizeStat {
                rows: 9000,
                requests: 3,
                median_wall_s: 2.0,
                prompt_tokens: Some(9005)
            }
        );
        assert_eq!(stats[1].rows, 16000);
    }

    #[test]
    fn summary_text_and_json() {
        let requests = [req(1, 9000, true, 3.0)];
        let tail = vec![line("amdgpu 0000:c0:00.0: amdgpu: GPU reset begin!")];
        let fatal = tail[0].clone();
        let npu = vec![
            "npu-trunk: mode auto".to_owned(),
            "npu-trunk: rows=9000 route=pair".to_owned(),
        ];
        let summary = Summary {
            pass: false,
            reason: "kernel error: x",
            hours: 4.0,
            requests: &requests,
            counts: Counts {
                benign: 2,
                amdgpu: 1,
                ..Counts::default()
            },
            first_fatal: Some(&fatal),
            tail: &tail,
            gaps: 1,
            npu_lines: &npu,
        };
        let text = summary.text();
        assert!(text.starts_with("verdict: FAIL (kernel error: x)\n"));
        assert!(text.contains("kmsg benign amdxdna -22 lines: 2"));
        assert!(text.contains("9000: median 3.00 s over 1 requests, 9005 prompt tokens"));
        assert!(text.contains("heartbeat gaps > 30 s: 1"));
        assert!(text.contains("planned requests (npu-trunk: rows=...): 1"));
        assert!(text.contains("kernel log tail:") && text.contains("GPU reset"));
        let json = summary
            .json("kernel=7.3.0 iommu=gpu:identity,npu:identity amdxdna=in-tree npu_fw=1.1.2.65");
        assert_eq!(json["verdict"], "FAIL");
        assert_eq!(json["benign_count"], 2);
        assert_eq!(json["requests"], 1);
        assert_eq!(
            json["fatal_line"],
            "2026-10-16T10:00:00.123456+00:00 kernel: amdgpu <pci>: amdgpu: GPU reset begin!"
        );
        assert!(!json.to_string().contains("hipx"));
    }

    #[test]
    fn passing_summary_has_no_kernel_tail() {
        let summary = Summary {
            pass: true,
            reason: "ran 1 h",
            hours: 1.0,
            requests: &[],
            counts: Counts::default(),
            first_fatal: None,
            tail: &[],
            gaps: 0,
            npu_lines: &[],
        };
        let text = summary.text();
        assert!(text.starts_with("verdict: PASS (ran 1 h)\n") && !text.contains("kernel log tail"));
        assert!(summary.json("h")["fatal_line"].is_null());
    }

    #[test]
    fn start_epoch_from_first_heartbeat_line() {
        let hb = "2026-10-16T12:34:56Z load=0.1 memavail_kb=5\n2026-10-16T12:35:06Z load=0.2 memavail_kb=5\n";
        assert_eq!(parse_start_epoch(hb), Some(1_792_154_096));
        assert_eq!(
            last_line(hb),
            Some("2026-10-16T12:35:06Z load=0.2 memavail_kb=5")
        );
        assert_eq!(parse_start_epoch(""), None);
        assert_eq!(parse_start_epoch("not a time load=1\n"), None);
    }

    #[test]
    fn report_text_lists_fatal_lines() {
        let kmsg = format!(
            "-- Boot abc --\n{}\n{}\n",
            line("amdxdna 0000:c1:00.1: amdxdna_insert_pages: Failed to mmap dma buf -22"),
            line("amdgpu 0000:c0:00.0: amdgpu: GPU reset begin!")
        );
        let text = build_report(
            1_792_154_096,
            1_792_160_000,
            Some("2026-10-16T13:00:00Z load=9 memavail_kb=7"),
            None,
            Some(&kmsg),
        );
        assert!(text.contains("last heartbeat: 2026-10-16T13:00:00Z load=9 memavail_kb=7"));
        assert!(text.contains("previous boot after a soak started 2026-10-16T12:34:56Z"));
        assert!(text.contains("kmsg benign amdxdna -22 lines: 1"));
        assert!(text.contains("fatal kernel lines: 1"));
        let unavailable = build_report(0, 0, None, None, None);
        assert!(unavailable.contains("unavailable"));
    }

    #[test]
    fn plan_names_the_contract_items() {
        let plan = plan_lines(4.0).join("\n");
        for needle in [
            "npu_trunk=auto",
            "9000/16000/32000/57000",
            "20 s idle",
            "4 h",
            "first kernel error",
            "MemAvailable < 4 GiB",
            "GPU and the NPU",
            "only into the output directory",
            "changes no setting",
        ] {
            assert!(plan.contains(needle), "{needle}");
        }
    }

    #[test]
    fn pci_address_shapes() {
        assert!(is_pci_address("0000:c0:00.0"));
        assert!(is_pci_address("c1:00.1"));
        assert!(!is_pci_address("12:34:56"));
        assert!(!is_pci_address("0000:c0:00.10"));
    }
}
