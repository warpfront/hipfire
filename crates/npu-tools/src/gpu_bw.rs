// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Strix Halo (gfx1151) GPU bandwidth probe using PM-native code objects.
//!
//! The kernels are precompiled into `native/gpu_bw_gfx1151.co` and loaded through the shared lazy
//! HIP runtime (`railgun::npu::hip_runtime`, `dlopen` only); nothing here needs the ROCm compiler. The
//! runtime selects the Halo iGPU (`railgun::npu::fclk::igpu_pci`, env `NPU_IGPU_PCI`, default `0000:bf:00.0`) before the module is loaded and the arch is gated to
//! `gfx1151` before the (gfx1151-only) code object is touched. `HIP_VISIBLE_DEVICES` is not needed.
//!
//! Module symbols `read_bw` / `write_bw` take the same 32-byte kernarg blob: `ptr@0`, `count@8`
//! (number of 16-byte elements), `sink@16` (per-thread XOR output, ignored by write) and
//! `stride@24` (launched threads). Launch is `CUs * 16` blocks of 256 threads.
//!
//! Verification is entirely CPU-side and outside the timed region. Write buffers get a full
//! poison upload (the lane-wise complement of the pattern), are proven poisoned by readback, are
//! launched, and are compared byte-exact against the pattern. Read sources get the full pattern
//! uploaded and proven by readback; the per-thread sink is set to 0xA5, the kernel runs and the
//! sink is compared with CPU-computed per-thread four-lane XOR checksums of every loaded element
//! (a checksum verification, not a byte-exact image of the source).

use railgun::npu::hip_runtime::{
    HipBuffer, HipEvent, HipFunction, HipMemoryKind, HipRuntime, HostRegistration,
};
use railgun::npu::{Bo, Device};
use std::ffi::c_void;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ARCH: &str = "gfx1151";
const CODE_OBJECT: &[u8] = include_bytes!("../native/gpu_bw_gfx1151.co");
const THREADS: u32 = 256;
const BLOCKS_PER_CU: u32 = 16;
const SINK_FILL: u8 = 0xA5;
const ELEMENT: usize = 16;
const DEFAULT_BYTES: usize = 1 << 30;
const DEFAULT_ITERS: u32 = 10;

const BACKINGS: [&str; 7] = [
    "vram",
    "vmm-host",
    "vmm-host-uncached",
    "host-registered",
    "host-registered-mapped",
    "host-registered-portable",
    "host-registered-userptr",
];
const OPS: [&str; 3] = ["read", "write", "copy"];

const USAGE: &str = "\
Usage: gpu-bw [--buf LIST] [--op LIST] [--bytes N] [--iters N]
       gpu-bw --duration SECONDS --buf host-registered,vram --op read,write [--bytes N]
  --buf LIST  Comma-separated backings (default: all):
              vram,vmm-host,vmm-host-uncached,host-registered,
              host-registered-mapped,host-registered-portable,host-registered-userptr
              vram: ordinary hipMalloc; vmm-host: pinned host VMM;
              vmm-host-uncached: uncached host VMM.
              host-registered: MAP_POPULATE anonymous pages, 2 MiB aligned, flags=0;
              -mapped: flags=2; -portable: flags=3 (Portable | Mapped);
              -userptr: flags=0, also pinned by an amdxdna userptr BO.
  --op LIST   read,write,copy (default: all); copy traffic counts read + write bytes
  --bytes N   Bytes per buffer, positive multiple of 16 (default: 1073741824)
  --iters N   Timed launches after two warm-ups (default: 10)
  --duration SECONDS  Run each selected backing/op for this duration.
  --help      Show usage without making any HIP call.
Verification (outside timing, on the CPU): write/copy destinations are poisoned by a full
CPU upload, launched, and compared byte-exact against a deterministic pattern before and
after timing and after a final poisoned launch. Copy sources are uploaded and proven present.
Read sources get the full pattern uploaded and proven present; the per-thread sink is set to
0xA5 and compared with CPU-computed per-thread four-lane XOR checksums of all loaded elements
(checksum verification, not full-byte-exact). Failure =>
verify_fail line and an error exit. Result rows carry verification=checksum (read) or
verification=byte-exact (write/copy). An unsupported allocation prints alloc_fail and continues;
a run that completes nothing is an error.
Device is selected by PCI bus ID $NPU_IGPU_PCI, default 0000:bf:00.0 (gfx1151 only); HIP_VISIBLE_DEVICES is not needed.
GB/s = 1e9 bytes/s. Background timestamps are CLOCK_REALTIME nanoseconds.";

// ---------------------------------------------------------------------------------------------
// CPU reference semantics (shared by upload generation and verification).
// ---------------------------------------------------------------------------------------------

/// Four u32 lanes of 16-byte element `i`. Index truncates to 32 bits exactly like the kernel.
#[inline]
fn pattern(i: usize) -> [u32; 4] {
    let x = i as u32;
    [
        x,
        x ^ 0x1234_5678,
        x.wrapping_mul(1_664_525),
        x.wrapping_add(1_013_904_223),
    ]
}

/// Lane-wise complement: every lane of the poison differs from the same lane of the pattern.
#[inline]
fn poison(i: usize) -> [u32; 4] {
    pattern(i).map(|lane| !lane)
}

fn store(chunk: &mut [u8], lanes: [u32; 4]) {
    for (dst, lane) in chunk.chunks_exact_mut(4).zip(lanes) {
        dst.copy_from_slice(&lane.to_le_bytes());
    }
}

fn load(chunk: &[u8]) -> [u32; 4] {
    let mut lanes = [0u32; 4];
    for (lane, src) in lanes.iter_mut().zip(chunk.chunks_exact(4)) {
        *lane = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
    }
    lanes
}

fn fill_pattern(buf: &mut [u8]) {
    for (i, chunk) in buf.chunks_exact_mut(ELEMENT).enumerate() {
        store(chunk, pattern(i));
    }
}

fn fill_poison(buf: &mut [u8]) {
    for (i, chunk) in buf.chunks_exact_mut(ELEMENT).enumerate() {
        store(chunk, poison(i));
    }
}

/// Elements whose bytes differ from `pattern(i)`.
fn pattern_mismatches(buf: &[u8]) -> u64 {
    buf.chunks_exact(ELEMENT)
        .enumerate()
        .filter(|&(i, chunk)| load(chunk) != pattern(i))
        .count() as u64
}

/// Elements whose bytes differ from `poison(i)`.
fn poison_mismatches(buf: &[u8]) -> u64 {
    buf.chunks_exact(ELEMENT)
        .enumerate()
        .filter(|&(i, chunk)| load(chunk) != poison(i))
        .count() as u64
}

/// Per-thread XOR `read_bw` must produce: thread `t` folds elements `t, t+threads, ...`.
fn expected_read(count: usize, threads: usize) -> Vec<[u32; 4]> {
    let mut sums = vec![[0u32; 4]; threads];
    let mut t = 0;
    for i in 0..count {
        let p = pattern(i);
        for (acc, lane) in sums[t].iter_mut().zip(p) {
            *acc ^= lane;
        }
        t += 1;
        if t == threads {
            t = 0;
        }
    }
    sums
}

/// Threads whose sink bytes differ from the expected checksum (a missing/short sink counts all).
fn sink_mismatches(sink: &[u8], expected: &[[u32; 4]]) -> u64 {
    let mut bad = 0u64;
    for (t, want) in expected.iter().enumerate() {
        match sink.get(t * ELEMENT..(t + 1) * ELEMENT) {
            Some(chunk) if load(chunk) == *want => {}
            _ => bad += 1,
        }
    }
    bad
}

// ---------------------------------------------------------------------------------------------
// Argument parsing.
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct Options {
    backings: Vec<String>,
    ops: Vec<String>,
    bytes: usize,
    iters: u32,
    duration: Option<f64>,
}

fn parse_list(option: &str, text: &str, names: &[&str]) -> Result<Vec<String>, String> {
    let mut result: Vec<String> = Vec::new();
    for item in text.split(',') {
        if !names.contains(&item) {
            return Err(format!(
                "{option}: unknown list entry {item:?}; expected one of {}",
                names.join(",")
            ));
        }
        if result.iter().any(|seen| seen == item) {
            return Err(format!("{option}: duplicate list entry {item:?}"));
        }
        result.push(item.to_string());
    }
    Ok(result)
}

fn parse_positive(option: &str, text: &str) -> Result<u64, String> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{option}: invalid number {text:?}"));
    }
    let value: u64 = text
        .parse()
        .map_err(|_| format!("{option}: number {text:?} overflows 64 bits"))?;
    if value == 0 {
        return Err(format!("{option}: must be a positive integer"));
    }
    Ok(value)
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut backings = None;
    let mut ops = None;
    let mut bytes = None;
    let mut iters = None;
    let mut duration = None;
    let mut it = args.iter();
    while let Some(option) = it.next() {
        let option = option.as_str();
        if !matches!(
            option,
            "--buf" | "--op" | "--bytes" | "--iters" | "--duration"
        ) {
            return Err(format!("unknown option: {option}"));
        }
        let value = it
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?
            .as_str();
        let seen = match option {
            "--buf" => backings.is_some(),
            "--op" => ops.is_some(),
            "--bytes" => bytes.is_some(),
            "--iters" => iters.is_some(),
            _ => duration.is_some(),
        };
        if seen {
            return Err(format!("duplicate option: {option}"));
        }
        match option {
            "--buf" => backings = Some(parse_list(option, value, &BACKINGS)?),
            "--op" => ops = Some(parse_list(option, value, &OPS)?),
            "--bytes" => {
                let n = parse_positive(option, value)?;
                let n = usize::try_from(n)
                    .ok()
                    .filter(|&n| n <= usize::MAX / 2)
                    .ok_or_else(|| format!("{option}: {value} is too large for this platform"))?;
                if n % ELEMENT != 0 {
                    return Err(format!("{option}: {n} is not a multiple of 16"));
                }
                bytes = Some(n);
            }
            "--iters" => {
                let n = parse_positive(option, value)?;
                iters = Some(
                    u32::try_from(n)
                        .map_err(|_| format!("{option}: {value} exceeds {}", u32::MAX))?,
                );
            }
            _ => {
                let seconds: f64 = value
                    .parse()
                    .map_err(|_| format!("{option}: invalid number {value:?}"))?;
                if !seconds.is_finite() || seconds <= 0.0 {
                    return Err(format!("{option}: must be positive finite seconds"));
                }
                duration = Some(seconds);
            }
        }
    }
    let options = Options {
        backings: backings.unwrap_or_else(|| BACKINGS.iter().map(|s| s.to_string()).collect()),
        ops: ops.unwrap_or_else(|| OPS.iter().map(|s| s.to_string()).collect()),
        bytes: bytes.unwrap_or(DEFAULT_BYTES),
        iters: iters.unwrap_or(DEFAULT_ITERS),
        duration,
    };
    // Duration applies independently to each selected backing/op.
    Ok(options)
}

// ---------------------------------------------------------------------------------------------
// Benchmark.
// ---------------------------------------------------------------------------------------------

fn kernarg(ptr: u64, count: u64, sink: u64, stride: u64) -> [u8; 32] {
    let mut blob = [0u8; 32];
    for (slot, value) in blob.chunks_exact_mut(8).zip([ptr, count, sink, stride]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    blob
}

fn realtime_ns() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .map_err(|e| format!("clock_gettime(CLOCK_REALTIME) before epoch: {e}"))
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Read compares per-thread XOR checksums; write compares every byte.
fn verification_label(op: &str) -> &'static str {
    if op == "read" {
        "checksum"
    } else {
        "byte-exact"
    }
}

// Linux anonymous mmap, deliberately local to the probe: no new allocator/runtime API.
unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
        -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}
struct Pages {
    ptr: *mut u8,
    len: usize,
}
impl Pages {
    fn new(bytes: usize) -> Result<Self, String> {
        const ALIGN: usize = 2 << 20;
        let len = bytes.checked_add(ALIGN - 1).ok_or("mmap size overflow")? & !(ALIGN - 1);
        let reserve = len.checked_add(ALIGN).ok_or("mmap size overflow")?;
        // PROT_READ|WRITE, MAP_PRIVATE|ANONYMOUS|POPULATE.
        let raw = unsafe { mmap(std::ptr::null_mut(), reserve, 3, 0x8022, -1, 0) };
        if raw as usize == usize::MAX {
            return Err(format!("mmap: {}", std::io::Error::last_os_error()));
        }
        let aligned = (raw as usize + ALIGN - 1) & !(ALIGN - 1);
        let prefix = aligned - raw as usize;
        let suffix = reserve - prefix - len;
        unsafe {
            if prefix != 0 {
                munmap(raw, prefix);
            }
            if suffix != 0 {
                munmap((aligned + len) as *mut c_void, suffix);
            }
        }
        Ok(Self {
            ptr: aligned as *mut u8,
            len,
        })
    }
}
impl Drop for Pages {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.cast(), self.len);
        }
    }
}
enum Buffer {
    Hip(HipBuffer),
    // Field drop order: unpin both devices before unmapping; BO before its device fd.
    Registered {
        reg: HostRegistration,
        _bo: Option<Bo>,
        _device: Option<Device>,
        pages: Pages,
    },
}
impl Buffer {
    fn allocate(rt: &HipRuntime, bytes: usize, backing: &str) -> Result<Self, String> {
        if !backing.starts_with("host-registered") {
            return rt.allocate(bytes, kind_for(backing)).map(Self::Hip);
        }
        let pages = Pages::new(bytes)?;
        let device = if backing == "host-registered-userptr" {
            Some(Device::open()?)
        } else {
            None
        };
        let bo = device
            .as_ref()
            .map(|d| d.userptr_bo(pages.ptr, pages.len))
            .transpose()?;
        let flags = match backing {
            "host-registered-mapped" => 2,
            "host-registered-portable" => 3,
            _ => 0,
        };
        let reg = rt.register_host_flags(pages.ptr, pages.len, flags)?;
        Ok(Self::Registered {
            reg,
            _bo: bo,
            _device: device,
            pages,
        })
    }
    fn ptr(&self) -> u64 {
        match self {
            Self::Hip(b) => b.ptr(),
            Self::Registered { reg, .. } => reg.ptr(),
        }
    }
    fn upload(&self, rt: &HipRuntime, bytes: &[u8]) -> Result<(), String> {
        match self {
            Self::Hip(b) => rt.upload(b, bytes),
            Self::Registered { pages, .. } => {
                rt.synchronize()?;
                assert!(bytes.len() <= pages.len);
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), pages.ptr, bytes.len());
                    railgun::npu::clflush(pages.ptr, bytes.len());
                }
                Ok(())
            }
        }
    }
    fn download(&self, rt: &HipRuntime, bytes: &mut [u8]) -> Result<(), String> {
        match self {
            Self::Hip(b) => rt.download(b, bytes),
            Self::Registered { pages, .. } => {
                rt.synchronize()?;
                assert!(bytes.len() <= pages.len);
                unsafe {
                    railgun::npu::clflush(pages.ptr, bytes.len());
                    std::ptr::copy_nonoverlapping(pages.ptr, bytes.as_mut_ptr(), bytes.len());
                }
                Ok(())
            }
        }
    }
}

struct Bench {
    rt: HipRuntime,
    read_fn: HipFunction,
    write_fn: HipFunction,
    sink: HipBuffer,
    start: HipEvent,
    stop: HipEvent,
    blocks: u32,
    total_threads: usize,
    count: usize,
    bytes: usize,
    /// One bytes-sized host buffer reused for every upload and readback.
    host: Vec<u8>,
    sink_host: Vec<u8>,
    expected: Option<Vec<[u32; 4]>>,
    copy_source: Option<Buffer>,
}

/// Why a verified step stopped: a verify mismatch is reported and the op skipped, a HIP error aborts.
enum Stop {
    Verify {
        phase: &'static str,
        mismatches: u64,
    },
    Fatal(String),
}

impl From<String> for Stop {
    fn from(error: String) -> Self {
        Stop::Fatal(error)
    }
}

impl Bench {
    fn launch(&self, op: &str, buffer: &Buffer) -> Result<(), String> {
        if op == "copy" {
            return self.rt.copy_dtod(
                buffer.ptr(),
                self.copy_source
                    .as_ref()
                    .ok_or("missing copy source")?
                    .ptr(),
                self.bytes,
            );
        }
        let function = if op == "read" {
            &self.read_fn
        } else {
            &self.write_fn
        };
        let mut args = kernarg(
            buffer.ptr(),
            self.count as u64,
            self.sink.ptr(),
            self.total_threads as u64,
        );
        self.rt.launch(function, self.blocks, THREADS, &mut args)
    }

    fn launch_sync(&self, op: &str, buffer: &Buffer) -> Result<(), String> {
        self.launch(op, buffer)?;
        self.rt.synchronize()
    }

    fn download_host(&mut self, buffer: &Buffer) -> Result<(), String> {
        buffer.download(&self.rt, &mut self.host)
    }

    /// Op-specific initial state; the target is proven to hold it by readback.
    fn init(&mut self, op: &str, buffer: &Buffer) -> Result<(), Stop> {
        if op == "copy" {
            fill_pattern(&mut self.host);
            let source = self
                .copy_source
                .as_ref()
                .ok_or("missing copy source".to_string())?;
            source.upload(&self.rt, &self.host)?;
            self.host.fill(0);
            source.download(&self.rt, &mut self.host)?;
            let bad = pattern_mismatches(&self.host);
            if bad != 0 {
                return Err(Stop::Verify {
                    phase: "copy-source",
                    mismatches: bad,
                });
            }
        }
        if op == "read" {
            fill_pattern(&mut self.host);
        } else {
            fill_poison(&mut self.host);
        }
        buffer.upload(&self.rt, &self.host)?;
        self.rt.synchronize()?;
        self.host.fill(0);
        self.download_host(buffer)?;
        let bad = if op == "read" {
            pattern_mismatches(&self.host)
        } else {
            poison_mismatches(&self.host)
        };
        if bad != 0 {
            return Err(Stop::Verify {
                phase: "init",
                mismatches: bad,
            });
        }
        Ok(())
    }

    /// Prepares the op's output so a no-op launch cannot pass: sink 0xA5 for read, poison for write.
    fn poison_output(
        &mut self,
        op: &str,
        buffer: &Buffer,
        phase: &'static str,
    ) -> Result<(), Stop> {
        if op == "read" {
            self.rt.memset(&self.sink, SINK_FILL)?;
            self.rt.synchronize()?;
            return Ok(());
        }
        fill_poison(&mut self.host);
        buffer.upload(&self.rt, &self.host)?;
        self.rt.synchronize()?;
        self.host.fill(0);
        self.download_host(buffer)?;
        let bad = poison_mismatches(&self.host);
        if bad != 0 {
            return Err(Stop::Verify {
                phase,
                mismatches: bad,
            });
        }
        Ok(())
    }

    fn verify(&mut self, op: &str, buffer: &Buffer, phase: &'static str) -> Result<(), Stop> {
        let bad = if op == "read" {
            if self.expected.is_none() {
                self.expected = Some(expected_read(self.count, self.total_threads));
            }
            self.rt.download(&self.sink, &mut self.sink_host)?;
            sink_mismatches(&self.sink_host, self.expected.as_deref().unwrap_or(&[]))
        } else {
            self.download_host(buffer)?;
            pattern_mismatches(&self.host)
        };
        if bad != 0 {
            return Err(Stop::Verify {
                phase,
                mismatches: bad,
            });
        }
        Ok(())
    }

    fn timed_iteration(&self, op: &str, buffer: &Buffer) -> Result<f32, String> {
        self.rt.record(&self.start)?;
        self.launch(op, buffer)?;
        self.rt.record(&self.stop)?;
        self.rt.synchronize()?;
        let ms = self.rt.elapsed_ms(&self.start, &self.stop)?;
        if !(ms > 0.0) {
            return Err(format!("non-positive HIP event duration {ms}"));
        }
        Ok(ms)
    }

    /// Runs one verified op on `buffer`; prints the result row on success.
    fn run_op(
        &mut self,
        options: &Options,
        backing: &str,
        op: &str,
        buffer: &Buffer,
    ) -> Result<(), Stop> {
        self.init(op, buffer)?;
        // Warm-up 1 is verified, warm-up 2 is unverified, as in the original tool.
        self.poison_output(op, buffer, "pre-poison")?;
        self.launch_sync(op, buffer)?;
        self.verify(op, buffer, "pre")?;
        self.launch_sync(op, buffer)?;
        let traffic = self.bytes * if op == "copy" { 2 } else { 1 };
        if let Some(seconds) = options.duration {
            let wall_start = realtime_ns()?;
            println!("gpubw-bg start {wall_start}");
            flush();
            let begin = Instant::now();
            let mut launches = 0u64;
            let elapsed = loop {
                self.launch_sync(op, buffer)?;
                launches += 1;
                let elapsed = begin.elapsed().as_secs_f64();
                if elapsed >= seconds {
                    break elapsed;
                }
            };
            println!("gpubw-bg end {}", realtime_ns()?);
            self.verify(op, buffer, "post")?;
            self.poison_output(op, buffer, "final-poison")?;
            self.launch_sync(op, buffer)?;
            self.verify(op, buffer, "final-poisoned")?;
            println!(
                "gpubw-bg buf={backing} op={op} bytes={traffic} launches={launches} GBps={:.2} seconds={elapsed:.6} verify=ok verification={}",
                traffic as f64 * launches as f64 / elapsed / 1e9,
                verification_label(op)
            );
        } else {
            let mut times = Vec::with_capacity(options.iters as usize);
            for _ in 0..options.iters {
                times.push(self.timed_iteration(op, buffer)?);
            }
            self.verify(op, buffer, "post")?;
            self.poison_output(op, buffer, "final-poison")?;
            self.launch_sync(op, buffer)?;
            self.verify(op, buffer, "final-poisoned")?;
            times.sort_by(f32::total_cmp);
            let best = f64::from(times[0]);
            let mid = times.len() / 2;
            let median = if times.len() % 2 == 1 {
                f64::from(times[mid])
            } else {
                (f64::from(times[mid - 1]) + f64::from(times[mid])) / 2.0
            };
            println!(
                "gpubw buf={backing} op={op} bytes={traffic} best_GBps={:.2} median_GBps={:.2} best_ms={best:.4} verify=ok verification={}",
                traffic as f64 / best / 1e6,
                traffic as f64 / median / 1e6,
                verification_label(op)
            );
        }
        flush();
        Ok(())
    }
}

fn flush() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

fn kind_for(backing: &str) -> Option<HipMemoryKind> {
    match backing {
        "vmm-host" => Some(HipMemoryKind::VmmHost),
        "vmm-host-uncached" => Some(HipMemoryKind::VmmHostUncached),
        _ => None,
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    // Even when other arguments are present, --help never initializes HIP.
    if args.iter().any(|arg| arg == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let options = parse_args(args)?;

    let pci = railgun::npu::fclk::igpu_pci();
    let rt = HipRuntime::load_for_pci(&pci)?;
    let arch = rt.device_arch()?;
    if arch != ARCH {
        return Err(format!(
            "device {pci} reports arch {arch:?}; this gpu-bw code object is {ARCH}-only"
        ));
    }
    let cus = rt.compute_units()?;
    if cus == 0 {
        return Err("device reports no compute units".to_string());
    }
    println!("device gcnArchName={arch} pci={pci} CUs={cus}");

    let blocks = cus
        .checked_mul(BLOCKS_PER_CU)
        .ok_or("grid size overflows u32")?;
    let total_threads = usize::try_from(u64::from(blocks) * u64::from(THREADS))
        .map_err(|_| "thread count overflows usize")?;
    let sink_bytes = total_threads
        .checked_mul(ELEMENT)
        .ok_or("sink size overflows usize")?;
    let module = rt.load_module(CODE_OBJECT)?;
    let read_fn = module.function("read_bw")?;
    let write_fn = module.function("write_bw")?;
    let sink = rt.allocate(sink_bytes, None)?;
    let start = rt.event()?;
    let stop = rt.event()?;

    let mut host: Vec<u8> = Vec::new();
    host.try_reserve_exact(options.bytes).map_err(|e| {
        format!(
            "cannot allocate {} byte host staging buffer: {e}",
            options.bytes
        )
    })?;
    host.resize(options.bytes, 0);

    let mut bench = Bench {
        rt,
        read_fn,
        write_fn,
        sink,
        start,
        stop,
        blocks,
        total_threads,
        count: options.bytes / ELEMENT,
        bytes: options.bytes,
        host,
        sink_host: vec![0u8; sink_bytes],
        expected: None,
        copy_source: None,
    };

    let mut completed = 0u32;
    let mut verify_failed = false;
    for backing in &options.backings {
        let allocation = (|| {
            let buffer = Buffer::allocate(&bench.rt, options.bytes, backing)?;
            bench.copy_source = if options.ops.iter().any(|op| op == "copy") {
                Some(Buffer::allocate(&bench.rt, options.bytes, backing)?)
            } else {
                None
            };
            Ok::<_, String>(buffer)
        })();
        let buffer = match allocation {
            Ok(buffer) => buffer,
            Err(error) => {
                println!(
                    "alloc_fail buf={backing} call=allocate err={}",
                    one_line(&error)
                );
                flush();
                continue;
            }
        };
        for op in &options.ops {
            match bench.run_op(&options, backing, op, &buffer) {
                Ok(()) => completed += 1,
                Err(Stop::Verify { phase, mismatches }) => {
                    println!("verify_fail buf={backing} op={op} phase={phase} mismatches={mismatches} verification={}", verification_label(op));
                    flush();
                    verify_failed = true;
                }
                Err(Stop::Fatal(error)) => return Err(format!("buf={backing} op={op}: {error}")),
            }
        }
        bench.copy_source = None;
    }
    // Unsupported backings are expected; a completely empty run or any verification failure is not.
    if verify_failed {
        return Err("verification failed".to_string());
    }
    if completed == 0 {
        return Err("no backing/op combination completed".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poison_differs_from_pattern_in_every_lane_and_is_detected() {
        for i in [
            0usize,
            1,
            2,
            0x1234_5678,
            u32::MAX as usize,
            u32::MAX as usize + 1,
        ] {
            let (p, q) = (pattern(i), poison(i));
            assert!(p.iter().zip(q).all(|(a, b)| *a != b), "i={i}");
        }
        let mut buf = vec![0u8; 5 * ELEMENT];
        fill_poison(&mut buf);
        // A launch that wrote nothing leaves poison, which must not pass as the pattern.
        assert_eq!(pattern_mismatches(&buf), 5);
        assert_eq!(poison_mismatches(&buf), 0);
        fill_pattern(&mut buf);
        assert_eq!(pattern_mismatches(&buf), 0);
        buf[3 * ELEMENT + 15] ^= 1;
        assert_eq!(pattern_mismatches(&buf), 1);
    }

    #[test]
    fn expected_checksums_fold_lanes_per_thread_with_stride_boundary() {
        // count = 6, 4 threads: t0 = e0^e4, t1 = e1^e5, t2 = e2, t3 = e3.
        let sums = expected_read(6, 4);
        for lane in 0..4 {
            assert_eq!(sums[0][lane], pattern(0)[lane] ^ pattern(4)[lane]);
            assert_eq!(sums[1][lane], pattern(1)[lane] ^ pattern(5)[lane]);
            assert_eq!(sums[2][lane], pattern(2)[lane]);
            assert_eq!(sums[3][lane], pattern(3)[lane]);
        }
        // Fewer elements than threads: the surplus threads store zero.
        let sums = expected_read(2, 4);
        assert_eq!(sums[2], [0; 4]);
        assert_eq!(sums[3], [0; 4]);
        // Mismatch counting flags a wrong lane and a missing thread slot.
        let mut sink = vec![0u8; 4 * ELEMENT];
        for (t, s) in sums.iter().enumerate() {
            store(&mut sink[t * ELEMENT..(t + 1) * ELEMENT], *s);
        }
        assert_eq!(sink_mismatches(&sink, &sums), 0);
        sink[ELEMENT + 12] ^= 0x80;
        assert_eq!(sink_mismatches(&sink, &sums), 1);
        assert_eq!(sink_mismatches(&sink[..ELEMENT], &sums), 3);
    }

    #[test]
    fn argument_validation() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(parse_args(&args(&["--bytes", "16", "--iters", "1"])).is_ok());
        for bad in [
            &["--bytes", "24"][..],
            &["--bytes", "0"],
            &["--bytes", "99999999999999999999999"],
            &["--iters", "4294967296"],
            &["--buf", "vram,vram"],
            &["--buf", "host"],
            &["--bytes"],
            &["--bytes", "16", "--bytes", "32"],
            &["--duration", "inf"],
            &["--duration", "0"],
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
        assert!(parse_args(&args(&[
            "--duration",
            "0.5",
            "--buf",
            "vram",
            "--op",
            "read"
        ]))
        .is_ok());
        assert!(parse_args(&args(&[
            "--duration",
            "6",
            "--buf",
            "host-registered,vram",
            "--op",
            "read,write,copy"
        ]))
        .is_ok());
    }

    #[test]
    fn anonymous_pages_are_populated_and_two_mib_aligned() {
        for bytes in [16usize, 4096, (2 << 20) + 16] {
            let pages = Pages::new(bytes).unwrap();
            assert_eq!(pages.ptr as usize % (2 << 20), 0);
            assert!(pages.len >= bytes);
            assert_eq!(pages.len % (2 << 20), 0);
            unsafe {
                assert_eq!(*pages.ptr, 0);
                assert_eq!(*pages.ptr.add(pages.len - 1), 0);
                *pages.ptr.add(bytes - 1) = 0x5a;
                assert_eq!(*pages.ptr.add(bytes - 1), 0x5a);
            }
        }
    }
}
