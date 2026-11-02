// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! G0 proof, Rust port: cross-process sharing of device VRAM through HIP VMM
//! POSIX-FD handles, passed with `SCM_RIGHTS`.
//!
//! This is the hip-bridge port of the standalone C++ G0 probe
//! (`g0-probe/g0_probe.cpp`, result `G0-RESULT.md`). It exercises only the
//! hip-bridge API surface that the weight-host design freezes:
//! `VmmArena::reserve_exportable` / `map_next` / `publish_read_only` /
//! `export_posix` (exporter) and `VmmArena::import_posix` (importer), plus
//! `HipRuntime` modules, `launch_kernel_blob`, streams and graph capture.
//!
//! # Invocation
//!
//! ```text
//! cargo build --release -p hip-bridge --example shared_vmm_fd_probe
//! # default role = orchestrate: spawns every other role itself
//! target/release/examples/shared_vmm_fd_probe
//! target/release/examples/shared_vmm_fd_probe --exporter-exit abrupt   # exporter _exit(1)
//! target/release/examples/shared_vmm_fd_probe --exporter-exit sigterm  # SIGTERM to the recorded exporter PID
//! ```
//!
//! Options (all roles accept the common ones; the orchestrator forwards them):
//!
//! | option | default | meaning |
//! |---|---|---|
//! | `--device N` / `HIPFIRE_SHARED_VMM_DEVICE` | 0 | HIP device ordinal (use `ROCR_VISIBLE_DEVICES` to pick the card) |
//! | `--exporter-exit clean\|abrupt\|sigterm` | clean | how the exporter dies while both importers hold the weights |
//! | `--hsaco PATH` | compiled | pre-built code object (skips the hipcc compile) |
//! | `--cycles N` | 3 | repeated import/unload cycles per importer |
//! | `--replays N` | 10 | graph replays per graph step |
//! | `--seed N` | fixed | pattern seed |
//! | `--accounting gate\|report\|off` | gate | physical VRAM counter (`mem_info_vram_used`) gates |
//! | `--step-timeout-s N` | 180 | upper bound for any single orchestrator wait |
//! | `--ipc-timeout-ms N` | 60000 | upper bound for any single socket send/recv |
//!
//! Roles (`--role`): `orchestrate` (default), `identity`, `exporter`,
//! `importer`. The last three are exec'd by the orchestrator as separate
//! processes (`current_exe()`), they are not forks. A role can also be started
//! by hand but needs inherited socket FDs (`--fds`/`--fd`) and `--hsaco`.
//!
//! The exit status is 0 only when every step passed and every subprocess ended
//! as expected. Any HIP error, data mismatch, protocol violation, unexpected
//! exit status or timeout prints `FAIL ...` and exits 1. Evidence lines are
//! `PASS <STEP> key=value ...`; the last line is
//! `shared_vmm_fd_probe: PASS (<n> steps, exporter-exit=<mode>)`. No timing or
//! performance claims are made.
//!
//! # What is exercised (G0 items)
//!
//! * 9 physical allocations in 9 arenas (requested 4 KiB, 64 KiB, 1 MiB,
//!   2 MiB+4 KiB, 5 MiB, 16 MiB, 33 MiB, 64 MiB, 64 MiB; each rounds up to the
//!   recommended granularity, the rounded sizes are the physical sizes that
//!   cross the wire). Each handle holds a deterministic 64-bit pattern
//!   (splitmix) written by a GPU fill kernel.
//! * Exporter: RW mapping verified, then `publish_read_only`, access query
//!   (`hipMemGetAccess`: RW = 3 before, RO = 1 after), re-verified.
//! * Separate exec'd importers B and C: `import_posix` maps RO at offset 0;
//!   access query must be RO.
//! * Verification is real GPU reads (a compiled `verify_kernel` over the
//!   shared mapping: mismatch count and word checksum) AND a full host
//!   readback of every byte, plus a wrong-seed negative control. Eager and
//!   hipGraph (capture of memset + verify kernels + async D2H of the result
//!   slots; replayed N times with the host buffer poisoned before every replay,
//!   then a fresh capture).
//! * Received FDs are closed after import (`close(2)` return checked; the
//!   process FD table is scanned for any FD still naming the same dma-buf
//!   inode, reported as `residual_same_inode_fds`), then everything is
//!   verified again.
//! * Exporter exits (clean unmap/release, `_exit(1)`, or SIGTERM) while B and C
//!   keep verifying (eager, replay of the pre-exit graph, fresh capture).
//! * Repeated import/unload cycles in each importer: each cycle imports every
//!   handle again into a fresh reservation, verifies, then releases; the VA of
//!   a released arena is retired, never reused (non-overlap with every earlier
//!   range and `retired_va_bytes()` growth are checked).
//! * Importer B exits first and independently, C keeps verifying, then C exits.
//! * Physical accounting from sysfs `mem_info_vram_used` (device found by PCI
//!   bus id): exporter allocation adds ~W; importing adds well under 0.25 W
//!   (no second physical copy); W stays resident after the exporter and after
//!   B exit; usage returns to the baseline after the last exit.
//! * Attribution diagnostics (never gating, nothing is signalled): at every
//!   accounting sample `DIAG` lines list per-PID VRAM on the target card from
//!   KFD (`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`) and from
//!   `/proc/<pid>/fdinfo` (`drm-memory-vram`, deduplicated by client id), split
//!   into our recorded PIDs (`who=A|B|C|I`) and every other accessible process
//!   (`who=OTHER`, with comm). After each reaped child `PROC_ABSENT` confirms
//!   `/proc/<pid>` is gone. The final sample is taken immediately and once more
//!   after one fixed 100 ms delay (`ATTRIBUTION` lines compare both against the
//!   baseline and against the change in other processes' VRAM); the final gate
//!   still uses the immediate sample only.
//!
//! # Process/IPC ABI
//!
//! The orchestrator creates one `AF_UNIX`/`SOCK_SEQPACKET|SOCK_CLOEXEC`
//! `socketpair` per importer, before spawning anything; each role inherits only
//! its own end(s) (CLOEXEC cleared in the child between fork and exec). The
//! orchestrator drives every role through the role's stdin (one command per
//! line) and reads its stdout (`EVT <name> k=v ...` lines or `FAIL <msg>`).
//!
//! Wire frame (64 bytes, 8 little-endian u64, one `sendmsg` = one SEQPACKET
//! record): `[magic, version=1, kind, index, count, bytes, requested, extra]`.
//!
//! * `MANIFEST(1)` exporter -> importer: `count` = number of handles, `bytes` =
//!   total physical bytes, `extra` = pattern seed. No FD.
//! * `ALLOC(2)`: `index`, `bytes` = physical (rounded) size, `requested`; carries
//!   EXACTLY ONE FD in `SCM_RIGHTS` (sender FD numbers are never trusted;
//!   `MSG_CMSG_CLOEXEC` on receive, `FD_CLOEXEC` verified).
//! * `ACK(3)` importer -> exporter after import+map+RO check: `count`, `bytes`
//!   imported.
//!
//! Receivers reject bad magic/version/kind/index, short or truncated frames
//! (`MSG_TRUNC`/`MSG_CTRUNC`), missing/extra FDs, byte totals that do not add
//! up. Every `sendmsg`/`recvmsg` is preceded by one bounded `poll`; there are
//! no retry loops anywhere (a timeout is a failure).
//!
//! # FD and handle lifecycle
//!
//! 1. Exporter: `reserve_exportable` -> one `map_next(reserved_bytes)` (RW) ->
//!    GPU fill + verify -> `publish_read_only` -> `export_posix` (one FD per
//!    handle) -> `SERVE`: send every FD to both importers by `SCM_RIGHTS`, await
//!    both ACKs, then close the exporter's FD copies. The physical handle is
//!    kept alive by the exporter's own handle until it exits/unmaps.
//! 2. Importer: receive FDs, `import_posix` (the runtime does not consume the
//!    FD; the importer keeps it open for the repeated-import cycles), ACK.
//!    `CLOSE_FDS` closes them all; the imported physical references live on in
//!    the mappings. Teardown: destroy graphs, `release` every arena (unmap +
//!    release; VA retired).
//! 3. The shared physical memory is freed by the kernel only after the last
//!    process drops its handle/mapping, so the exporter dying never invalidates
//!    importer mappings.
//!
//! Only SIGTERM, only to PIDs recorded at spawn (start time re-checked in
//! `/proc/<pid>/stat` before signalling), only for the sigterm exit mode and
//! for failure cleanup. No SIGKILL, no process-name matching.

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
compile_error!("shared_vmm_fd_probe is Linux 64-bit only (SCM_RIGHTS via raw glibc FFI)");

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use hip_bridge::{
    retired_va_bytes, DeviceBuffer, Function, Graph, GraphExec, HipRuntime, KernargBlob, Module,
    Stream, VmmArena,
};

// ---------------------------------------------------------------------------
// Raw glibc FFI (std has no stable SCM_RIGHTS; no extra crate dependency).
// ---------------------------------------------------------------------------
mod sys {
    use std::os::raw::{c_int, c_short, c_ulong, c_void};

    pub const AF_UNIX: c_int = 1;
    pub const SOCK_SEQPACKET: c_int = 5;
    pub const SOCK_CLOEXEC: c_int = 0o2_000_000;
    pub const SOL_SOCKET: c_int = 1;
    pub const SCM_RIGHTS: c_int = 1;
    pub const MSG_NOSIGNAL: c_int = 0x4000;
    pub const MSG_CMSG_CLOEXEC: c_int = 0x4000_0000;
    pub const MSG_TRUNC: c_int = 0x20;
    pub const MSG_CTRUNC: c_int = 0x8;
    pub const POLLIN: c_short = 1;
    pub const POLLOUT: c_short = 4;
    pub const F_GETFD: c_int = 1;
    pub const F_SETFD: c_int = 2;
    pub const FD_CLOEXEC: c_int = 1;
    pub const SIGTERM: c_int = 15;

    #[repr(C)]
    pub struct IoVec {
        pub base: *mut c_void,
        pub len: usize,
    }

    /// glibc `struct msghdr` (Linux 64-bit).
    #[repr(C)]
    pub struct MsgHdr {
        pub name: *mut c_void,
        pub namelen: u32,
        pub iov: *mut IoVec,
        pub iovlen: usize,
        pub control: *mut c_void,
        pub controllen: usize,
        pub flags: c_int,
    }

    #[repr(C)]
    pub struct PollFd {
        pub fd: c_int,
        pub events: c_short,
        pub revents: c_short,
    }

    extern "C" {
        pub fn socketpair(domain: c_int, ty: c_int, proto: c_int, sv: *mut c_int) -> c_int;
        pub fn sendmsg(fd: c_int, msg: *const MsgHdr, flags: c_int) -> isize;
        pub fn recvmsg(fd: c_int, msg: *mut MsgHdr, flags: c_int) -> isize;
        pub fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
        pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        pub fn close(fd: c_int) -> c_int;
        pub fn kill(pid: c_int, sig: c_int) -> c_int;
        pub fn _exit(code: c_int) -> !;
    }
}

// ---------------------------------------------------------------------------
// Constants and wire format
// ---------------------------------------------------------------------------
const MAGIC: u64 = 0x5348_5644_4d46_4431; // "SHVDMFD1"
const WIRE_VERSION: u64 = 1;
const KIND_MANIFEST: u64 = 1;
const KIND_ALLOC: u64 = 2;
const KIND_ACK: u64 = 3;
const FRAME_BYTES: usize = 64;
type Frame = [u64; 8];

/// hipMemAccessFlags: `HIP_MEM_ACCESS_FLAGS_PROT_*` (G0 values).
const ACCESS_RO: u32 = 1;
const ACCESS_RW: u32 = 3;

/// Requested sizes of the 9 physical handles (mixed small/large, two 64 MiB).
const REQUESTED_BYTES: [u64; 9] = [
    4096,
    65_536,
    1 << 20,
    (2 << 20) + 4096,
    5 << 20,
    16 << 20,
    33 << 20,
    64 << 20,
    64 << 20,
];
const DEFAULT_SEED: u64 = 0x5348_4152_4544_564d;
const BLOCK: u32 = 256;
const MAX_GRID: u64 = 2048;

const KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static __device__ __forceinline__ uint64_t pat(uint64_t i, uint64_t sm) {
    return splitmix((i * 0xD6E8FEB86659FD93ull) ^ sm);
}

extern "C" __global__ void fill_kernel(uint64_t* p, size_t n, uint64_t sm) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = pat(i, sm);
}

// out[0] = mismatching words, out[1] = wrapping sum of all words read.
extern "C" __global__ void verify_kernel(const uint64_t* p, size_t n, uint64_t sm,
                                         unsigned long long* out) {
    unsigned long long mm = 0, s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint64_t v = p[i];
        if (v != pat(i, sm)) ++mm;
        s += v;
    }
    if (mm) atomicAdd(&out[0], (unsigned long long)mm);
    atomicAdd(&out[1], (unsigned long long)s);
}
"#;

// ---------------------------------------------------------------------------
// Host-side pattern (must match the kernel)
// ---------------------------------------------------------------------------
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn pat(i: u64, sm: u64) -> u64 {
    splitmix(i.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ sm)
}

fn seedmix(seed: u64, id: u32) -> u64 {
    splitmix(seed ^ ((u64::from(id) + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)))
}

fn expected_sum(words: u64, sm: u64) -> u64 {
    (0..words).fold(0u64, |acc, i| acc.wrapping_add(pat(i, sm)))
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    id: u32,
    requested: u64,
    /// Physical (granularity-rounded) bytes.
    bytes: u64,
    sm: u64,
    expect_sum: u64,
}

impl Spec {
    fn new(seed: u64, id: u32, requested: u64, bytes: u64) -> Spec {
        let sm = seedmix(seed, id);
        Spec {
            id,
            requested,
            bytes,
            sm,
            expect_sum: expected_sum(bytes / 8, sm),
        }
    }

    fn words(&self) -> u64 {
        self.bytes / 8
    }
}

struct Mapped {
    spec: Spec,
    arena: VmmArena,
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExitMode {
    Clean,
    Abrupt,
    Sigterm,
}

impl ExitMode {
    fn as_str(self) -> &'static str {
        match self {
            ExitMode::Clean => "clean",
            ExitMode::Abrupt => "abrupt",
            ExitMode::Sigterm => "sigterm",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AcctMode {
    Gate,
    Report,
    Off,
}

struct Args {
    role: String,
    name: String,
    device: i32,
    exit_mode: ExitMode,
    hsaco: Option<String>,
    fds: Vec<RawFd>,
    seed: u64,
    ipc_timeout_ms: i32,
    step_timeout: Duration,
    cycles: u32,
    replays: u32,
    acct: AcctMode,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        role: "orchestrate".into(),
        name: String::new(),
        device: std::env::var("HIPFIRE_SHARED_VMM_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        exit_mode: ExitMode::Clean,
        hsaco: None,
        fds: Vec::new(),
        seed: DEFAULT_SEED,
        ipc_timeout_ms: 60_000,
        step_timeout: Duration::from_secs(180),
        cycles: 3,
        replays: 10,
        acct: AcctMode::Gate,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = |what: &str| it.next().ok_or_else(|| format!("{what} needs a value"));
        match flag.as_str() {
            "--role" => a.role = val("--role")?,
            "--name" => a.name = val("--name")?,
            "--device" => a.device = parse_num(&val("--device")?, "--device")?,
            "--exporter-exit" => {
                a.exit_mode = match val("--exporter-exit")?.as_str() {
                    "clean" => ExitMode::Clean,
                    "abrupt" => ExitMode::Abrupt,
                    "sigterm" => ExitMode::Sigterm,
                    other => return Err(format!("--exporter-exit: unknown mode {other}")),
                }
            }
            "--hsaco" => a.hsaco = Some(val("--hsaco")?),
            "--fds" => {
                for part in val("--fds")?.split(',').filter(|s| !s.is_empty()) {
                    a.fds.push(parse_num(part, "--fds")?);
                }
            }
            "--fd" => a.fds.push(parse_num(&val("--fd")?, "--fd")?),
            "--seed" => a.seed = parse_num(&val("--seed")?, "--seed")?,
            "--ipc-timeout-ms" => {
                a.ipc_timeout_ms = parse_num(&val("--ipc-timeout-ms")?, "--ipc-timeout-ms")?
            }
            "--step-timeout-s" => {
                a.step_timeout =
                    Duration::from_secs(parse_num(&val("--step-timeout-s")?, "--step-timeout-s")?)
            }
            "--cycles" => a.cycles = parse_num(&val("--cycles")?, "--cycles")?,
            "--replays" => a.replays = parse_num(&val("--replays")?, "--replays")?,
            "--accounting" => {
                a.acct = match val("--accounting")?.as_str() {
                    "gate" => AcctMode::Gate,
                    "report" => AcctMode::Report,
                    "off" => AcctMode::Off,
                    other => return Err(format!("--accounting: unknown mode {other}")),
                }
            }
            "-h" | "--help" => {
                println!("see the module documentation at the top of shared_vmm_fd_probe.rs");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.ipc_timeout_ms <= 0 {
        return Err("--ipc-timeout-ms must be positive".into());
    }
    if a.cycles == 0 || a.replays == 0 {
        return Err("--cycles and --replays must be at least 1".into());
    }
    Ok(a)
}

fn parse_num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
    s.trim()
        .parse::<T>()
        .map_err(|_| format!("{what}: invalid number {s:?}"))
}

// ---------------------------------------------------------------------------
// Event output (role -> orchestrator)
// ---------------------------------------------------------------------------
fn emit(name: &str, kv: &[(&str, String)]) {
    let mut line = format!("EVT {name}");
    for (k, v) in kv {
        let _ = write!(line, " {k}={v}");
    }
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{line}");
    let _ = lock.flush();
}

fn fail_line(msg: &str) {
    let one_line = msg.replace(['\n', '\r'], " ");
    eprintln!("FAIL {one_line}");
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "FAIL {one_line}");
    let _ = lock.flush();
}

fn hip_err<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |e| format!("{what}: {e}")
}

// ---------------------------------------------------------------------------
// Bounded SCM_RIGHTS socket I/O
// ---------------------------------------------------------------------------
fn wait_ready(fd: RawFd, events: i16, timeout_ms: i32, what: &str) -> Result<(), String> {
    let mut p = sys::PollFd {
        fd,
        events,
        revents: 0,
    };
    // SAFETY: `p` is a live stack PollFd and nfds == 1.
    let r = unsafe { sys::poll(&mut p, 1, timeout_ms) };
    match r {
        0 => Err(format!("{what}: timed out after {timeout_ms} ms")),
        r if r < 0 => Err(format!("{what}: poll: {}", std::io::Error::last_os_error())),
        _ => Ok(()),
    }
}

fn encode(frame: &Frame) -> [u8; FRAME_BYTES] {
    let mut out = [0u8; FRAME_BYTES];
    for (i, w) in frame.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn decode(buf: &[u8; FRAME_BYTES]) -> Frame {
    let mut f = [0u64; 8];
    for (i, w) in f.iter_mut().enumerate() {
        *w = u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
    }
    f
}

fn mk_frame(kind: u64, index: u64, count: u64, bytes: u64, requested: u64, extra: u64) -> Frame {
    [
        MAGIC,
        WIRE_VERSION,
        kind,
        index,
        count,
        bytes,
        requested,
        extra,
    ]
}

#[repr(C, align(8))]
struct Ctl([u8; 64]);

/// One bounded `poll(POLLOUT)` then one `sendmsg`. At most one FD is attached.
fn send_frame(
    sock: RawFd,
    frame: &Frame,
    fd: Option<BorrowedFd<'_>>,
    timeout_ms: i32,
    what: &str,
) -> Result<(), String> {
    let mut bytes = encode(frame);
    let mut iov = sys::IoVec {
        base: bytes.as_mut_ptr().cast(),
        len: FRAME_BYTES,
    };
    let mut ctl = Ctl([0u8; 64]);
    let mut hdr = sys::MsgHdr {
        name: std::ptr::null_mut(),
        namelen: 0,
        iov: &mut iov,
        iovlen: 1,
        control: std::ptr::null_mut(),
        controllen: 0,
        flags: 0,
    };
    if let Some(fd) = fd {
        // cmsghdr { size_t len; int level; int type; } then the FD; CMSG_LEN(4)=20,
        // CMSG_SPACE(4)=24 on Linux 64-bit.
        ctl.0[0..8].copy_from_slice(&20u64.to_ne_bytes());
        ctl.0[8..12].copy_from_slice(&sys::SOL_SOCKET.to_ne_bytes());
        ctl.0[12..16].copy_from_slice(&sys::SCM_RIGHTS.to_ne_bytes());
        ctl.0[16..20].copy_from_slice(&fd.as_raw_fd().to_ne_bytes());
        hdr.control = ctl.0.as_mut_ptr().cast();
        hdr.controllen = 24;
    }
    wait_ready(sock, sys::POLLOUT, timeout_ms, what)?;
    // SAFETY: hdr points at live iov/ctl buffers valid for the call.
    let n = unsafe { sys::sendmsg(sock, &hdr, sys::MSG_NOSIGNAL) };
    if n != FRAME_BYTES as isize {
        return Err(format!(
            "{what}: sendmsg: {}",
            if n < 0 {
                std::io::Error::last_os_error().to_string()
            } else {
                format!("short send {n}")
            }
        ));
    }
    Ok(())
}

/// One bounded `poll(POLLIN)` then one `recvmsg`. Returns the frame and every
/// FD that arrived (already owned, so they close on any later error).
fn recv_frame(sock: RawFd, timeout_ms: i32, what: &str) -> Result<(Frame, Vec<OwnedFd>), String> {
    wait_ready(sock, sys::POLLIN, timeout_ms, what)?;
    let mut bytes = [0u8; FRAME_BYTES];
    let mut iov = sys::IoVec {
        base: bytes.as_mut_ptr().cast(),
        len: FRAME_BYTES,
    };
    let mut ctl = Ctl([0u8; 64]);
    let mut hdr = sys::MsgHdr {
        name: std::ptr::null_mut(),
        namelen: 0,
        iov: &mut iov,
        iovlen: 1,
        control: ctl.0.as_mut_ptr().cast(),
        controllen: 64,
        flags: 0,
    };
    // SAFETY: hdr points at live iov/ctl buffers valid for the call.
    let n = unsafe { sys::recvmsg(sock, &mut hdr, sys::MSG_CMSG_CLOEXEC) };
    let err = std::io::Error::last_os_error();
    let mut fds: Vec<OwnedFd> = Vec::new();
    if n >= 0 {
        let len = hdr.controllen.min(64);
        let mut off = 0usize;
        while off + 16 <= len {
            let cmsg_len = u64::from_ne_bytes(ctl.0[off..off + 8].try_into().expect("8")) as usize;
            let level = i32::from_ne_bytes(ctl.0[off + 8..off + 12].try_into().expect("4"));
            let ty = i32::from_ne_bytes(ctl.0[off + 12..off + 16].try_into().expect("4"));
            if cmsg_len < 16 || off + cmsg_len > len {
                break;
            }
            if level == sys::SOL_SOCKET && ty == sys::SCM_RIGHTS {
                for chunk in ctl.0[off + 16..off + cmsg_len].chunks_exact(4) {
                    let raw = i32::from_ne_bytes(chunk.try_into().expect("4"));
                    // SAFETY: the kernel installed `raw` in this process for us.
                    fds.push(unsafe { OwnedFd::from_raw_fd(raw) });
                }
            }
            off += (cmsg_len + 7) & !7;
        }
    }
    if n < 0 {
        return Err(format!("{what}: recvmsg: {err}"));
    }
    if n == 0 {
        return Err(format!("{what}: peer closed the connection"));
    }
    if n as usize != FRAME_BYTES || hdr.flags & (sys::MSG_TRUNC | sys::MSG_CTRUNC) != 0 {
        return Err(format!(
            "{what}: short/truncated frame or control data (n={n})"
        ));
    }
    let frame = decode(&bytes);
    if frame[0] != MAGIC || frame[1] != WIRE_VERSION {
        return Err(format!(
            "{what}: bad magic/version {:#x}/{}",
            frame[0], frame[1]
        ));
    }
    Ok((frame, fds))
}

fn close_checked(fd: OwnedFd) -> Result<(), String> {
    let raw = fd.into_raw_fd();
    // SAFETY: `raw` was owned by us and is not used again.
    let rc = unsafe { sys::close(raw) };
    if rc != 0 {
        return Err(format!("close({raw}): {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

fn fd_is_cloexec(fd: RawFd) -> bool {
    // SAFETY: plain fcntl query on an FD number.
    let flags = unsafe { sys::fcntl(fd, sys::F_GETFD) };
    flags >= 0 && flags & sys::FD_CLOEXEC != 0
}

fn fd_identity(fd: RawFd) -> Result<(u64, u64), String> {
    let md = std::fs::metadata(format!("/proc/self/fd/{fd}"))
        .map_err(|e| format!("stat /proc/self/fd/{fd}: {e}"))?;
    Ok((md.dev(), md.ino()))
}

/// Count process FDs that still name one of `idents` (dev, inode).
fn count_fds_with_identity(idents: &[(u64, u64)]) -> Result<usize, String> {
    let mut hits = 0usize;
    let rd =
        std::fs::read_dir("/proc/self/fd").map_err(|e| format!("read_dir /proc/self/fd: {e}"))?;
    for entry in rd.flatten() {
        if let Ok(md) = std::fs::metadata(entry.path()) {
            if idents.contains(&(md.dev(), md.ino())) {
                hits += 1;
            }
        }
    }
    Ok(hits)
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------
struct Gpu {
    hip: HipRuntime,
    device: i32,
    _module: Module,
    fill: Function,
    verify: Function,
}

fn grid_for(words: u64) -> u32 {
    words.div_ceil(u64::from(BLOCK)).clamp(1, MAX_GRID) as u32
}

impl Gpu {
    fn init(device: i32, hsaco: &str) -> Result<Gpu, String> {
        let hip = HipRuntime::load().map_err(hip_err("load HIP runtime"))?;
        hip.set_device(device).map_err(hip_err("hipSetDevice"))?;
        let module = hip.module_load(hsaco).map_err(hip_err("hipModuleLoad"))?;
        let fill = hip
            .module_get_function(&module, "fill_kernel")
            .map_err(hip_err("fill_kernel"))?;
        let verify = hip
            .module_get_function(&module, "verify_kernel")
            .map_err(hip_err("verify_kernel"))?;
        Ok(Gpu {
            hip,
            device,
            _module: module,
            fill,
            verify,
        })
    }

    fn launch(
        &self,
        func: &Function,
        words: u64,
        blob: &mut KernargBlob,
        stream: Option<&Stream>,
    ) -> Result<(), String> {
        // SAFETY: the blob is the kernel's explicit argument struct
        // (ptr, u64, u64[, ptr]); all pointers are live device addresses.
        let launched = unsafe {
            self.hip.launch_kernel_blob(
                func,
                [grid_for(words), 1, 1],
                [BLOCK, 1, 1],
                0,
                stream,
                blob.as_mut_slice(),
            )
        };
        launched.map_err(hip_err("hipModuleLaunchKernel"))
    }

    fn fill(&self, ptr: *mut c_void, words: u64, sm: u64) -> Result<(), String> {
        let mut blob = KernargBlob::new();
        blob.push_ptr(ptr);
        blob.push_u64(words);
        blob.push_u64(sm);
        self.launch(&self.fill, words, &mut blob, None)
    }

    fn verify_blob(ptr: *mut c_void, words: u64, sm: u64, out: *mut c_void) -> KernargBlob {
        let mut blob = KernargBlob::new();
        blob.push_ptr(ptr);
        blob.push_u64(words);
        blob.push_u64(sm);
        blob.push_ptr(out);
        blob
    }

    fn sync(&self) -> Result<(), String> {
        self.hip
            .device_synchronize()
            .map_err(hip_err("hipDeviceSynchronize"))
    }
}

struct View {
    spec: Spec,
    buf: DeviceBuffer,
}

fn views(set: &[Mapped]) -> Result<Vec<View>, String> {
    set.iter()
        .map(|m| {
            Ok(View {
                spec: m.spec,
                buf: m
                    .arena
                    .buffer(m.spec.bytes as usize)
                    .map_err(hip_err("arena.buffer"))?,
            })
        })
        .collect()
}

fn res_slot(base: *mut c_void, i: usize) -> *mut c_void {
    (base as *mut u8).wrapping_add(i * 16).cast()
}

fn parse_res(bytes: &[u8], i: usize) -> (u64, u64) {
    let o = i * 16;
    (
        u64::from_ne_bytes(bytes[o..o + 8].try_into().expect("8")),
        u64::from_ne_bytes(bytes[o + 8..o + 16].try_into().expect("8")),
    )
}

fn check_res(bytes: &[u8], vs: &[View], ctx: &str) -> Result<(), String> {
    for (i, v) in vs.iter().enumerate() {
        let (mm, sum) = parse_res(bytes, i);
        if mm != 0 || sum != v.spec.expect_sum {
            return Err(format!(
                "{ctx}: GPU verify mismatch alloc {} ({} B): mismatches={mm} sum={sum:#x} expect={:#x}",
                v.spec.id, v.spec.bytes, v.spec.expect_sum
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct VerifyStats {
    gpu_words: u64,
    readback_bytes: u64,
}

/// Eager GPU verification (real kernel reads of the mapping) and optional full
/// host readback of every byte.
fn verify_eager(
    gpu: &Gpu,
    set: &[Mapped],
    ctx: &str,
    readback: bool,
) -> Result<VerifyStats, String> {
    let vs = views(set)?;
    let res = gpu
        .hip
        .malloc(vs.len() * 16)
        .map_err(hip_err("malloc result slots"))?;
    gpu.hip
        .memset(&res, 0, vs.len() * 16)
        .map_err(hip_err("memset result slots"))?;
    let mut stats = VerifyStats::default();
    for (i, v) in vs.iter().enumerate() {
        let mut blob = Gpu::verify_blob(
            v.buf.as_ptr(),
            v.spec.words(),
            v.spec.sm,
            res_slot(res.as_ptr(), i),
        );
        gpu.launch(&gpu.verify, v.spec.words(), &mut blob, None)?;
        stats.gpu_words += v.spec.words();
    }
    gpu.sync()?;
    let mut host = vec![0u8; vs.len() * 16];
    gpu.hip
        .memcpy_dtoh(&mut host, &res)
        .map_err(hip_err("D2H result slots"))?;
    gpu.hip.free(res).map_err(hip_err("free result slots"))?;
    check_res(&host, &vs, ctx)?;
    if readback {
        let max = vs.iter().map(|v| v.spec.bytes as usize).max().unwrap_or(0);
        let mut bytes = vec![0u8; max];
        for v in &vs {
            let n = v.spec.bytes as usize;
            gpu.hip
                .memcpy_dtoh(&mut bytes[..n], &v.buf)
                .map_err(hip_err("D2H readback"))?;
            for (i, w) in bytes[..n].chunks_exact(8).enumerate() {
                let got = u64::from_le_bytes(w.try_into().expect("8"));
                if got != pat(i as u64, v.spec.sm) {
                    return Err(format!(
                        "{ctx}: host readback mismatch alloc {} word {i}: {got:#x} != {:#x}",
                        v.spec.id,
                        pat(i as u64, v.spec.sm)
                    ));
                }
            }
            stats.readback_bytes += v.spec.bytes;
        }
    }
    Ok(stats)
}

/// The verify kernel must actually detect corruption: run it with a wrong seed
/// and require mismatches.
fn negative_control(gpu: &Gpu, set: &[Mapped]) -> Result<u64, String> {
    let m = set.last().ok_or("negative control: empty set")?;
    let buf = m
        .arena
        .buffer(m.spec.bytes as usize)
        .map_err(hip_err("arena.buffer"))?;
    let res = gpu.hip.malloc(16).map_err(hip_err("malloc"))?;
    gpu.hip.memset(&res, 0, 16).map_err(hip_err("memset"))?;
    let mut blob = Gpu::verify_blob(
        buf.as_ptr(),
        m.spec.words(),
        m.spec.sm ^ 0xDEAD_BEEF,
        res.as_ptr(),
    );
    gpu.launch(&gpu.verify, m.spec.words(), &mut blob, None)?;
    gpu.sync()?;
    let mut host = [0u8; 16];
    gpu.hip
        .memcpy_dtoh(&mut host, &res)
        .map_err(hip_err("D2H"))?;
    gpu.hip.free(res).map_err(hip_err("free"))?;
    let (mm, _) = parse_res(&host, 0);
    if mm < m.spec.words() / 2 {
        return Err(format!(
            "negative control: wrong seed produced only {mm} mismatches of {} words",
            m.spec.words()
        ));
    }
    Ok(mm)
}

/// Captured graph: memset(result slots) + one verify kernel per allocation +
/// async D2H of the slots into pinned host memory.
struct GraphState {
    stream: Stream,
    graph: Graph,
    exec: GraphExec,
    res: DeviceBuffer,
    host: *mut c_void,
    host_len: usize,
    _blobs: Vec<KernargBlob>,
    specs: Vec<Spec>,
}

impl GraphState {
    fn capture(gpu: &Gpu, set: &[Mapped]) -> Result<GraphState, String> {
        let vs = views(set)?;
        let slots = vs.len() * 16;
        let res = gpu
            .hip
            .malloc(slots)
            .map_err(hip_err("malloc graph result slots"))?;
        let host = gpu
            .hip
            .host_malloc(slots, 0)
            .map_err(hip_err("hipHostMalloc"))?;
        let stream = gpu
            .hip
            .stream_create_non_blocking()
            .map_err(hip_err("stream_create"))?;
        let mut blobs: Vec<KernargBlob> = Vec::with_capacity(vs.len());
        let captured = (|| -> Result<(), String> {
            // Mode 1 = hipStreamCaptureModeThreadLocal (rdna-compute convention).
            gpu.hip
                .stream_begin_capture(&stream, 1)
                .map_err(hip_err("hipStreamBeginCapture"))?;
            gpu.hip
                .memset_async(&res, 0, slots, &stream)
                .map_err(hip_err("memset_async"))?;
            for (i, v) in vs.iter().enumerate() {
                let mut blob = Gpu::verify_blob(
                    v.buf.as_ptr(),
                    v.spec.words(),
                    v.spec.sm,
                    res_slot(res.as_ptr(), i),
                );
                gpu.launch(&gpu.verify, v.spec.words(), &mut blob, Some(&stream))?;
                blobs.push(blob);
            }
            // SAFETY: `host` is pinned host memory of `slots` bytes that outlives the graph.
            let dst = unsafe { std::slice::from_raw_parts_mut(host.cast::<u8>(), slots) };
            gpu.hip
                .memcpy_dtoh_async(dst, &res, Some(&stream))
                .map_err(hip_err("memcpy_dtoh_async"))?;
            Ok(())
        })();
        let ended = gpu.hip.stream_end_capture(&stream);
        if let Err(e) = captured {
            let _ = ended;
            return Err(format!("graph capture: {e}"));
        }
        let graph = ended.map_err(hip_err("hipStreamEndCapture"))?;
        let exec = gpu
            .hip
            .graph_instantiate(&graph)
            .map_err(hip_err("hipGraphInstantiate"))?;
        Ok(GraphState {
            stream,
            graph,
            exec,
            res,
            host,
            host_len: slots,
            _blobs: blobs,
            specs: vs.iter().map(|v| v.spec).collect(),
        })
    }

    /// Replay `n` times; the pinned host buffer is poisoned (0xFF) before every
    /// replay so only a real re-execution can produce the expected results.
    fn replay(&self, gpu: &Gpu, n: u32, ctx: &str) -> Result<(), String> {
        for r in 0..n {
            // SAFETY: host is a live pinned allocation of host_len bytes; the
            // stream is idle (synchronized) between replays.
            let host =
                unsafe { std::slice::from_raw_parts_mut(self.host.cast::<u8>(), self.host_len) };
            host.fill(0xFF);
            gpu.hip
                .graph_launch(&self.exec, &self.stream)
                .map_err(hip_err("hipGraphLaunch"))?;
            gpu.hip
                .stream_synchronize(&self.stream)
                .map_err(hip_err("hipStreamSynchronize"))?;
            for (i, spec) in self.specs.iter().enumerate() {
                let (mm, sum) = parse_res(host, i);
                if mm != 0 || sum != spec.expect_sum {
                    return Err(format!(
                        "{ctx}: graph replay {r} alloc {}: mismatches={mm} sum={sum:#x} expect={:#x}",
                        spec.id, spec.expect_sum
                    ));
                }
            }
        }
        Ok(())
    }

    fn destroy(self, gpu: &Gpu) -> Result<(), String> {
        gpu.hip
            .stream_synchronize(&self.stream)
            .map_err(hip_err("stream sync"))?;
        gpu.hip
            .graph_exec_destroy(self.exec)
            .map_err(hip_err("graph_exec_destroy"))?;
        gpu.hip
            .graph_destroy(self.graph)
            .map_err(hip_err("graph_destroy"))?;
        gpu.hip
            .stream_destroy(self.stream)
            .map_err(hip_err("stream_destroy"))?;
        gpu.hip.host_free(self.host).map_err(hip_err("host_free"))?;
        gpu.hip
            .free(self.res)
            .map_err(hip_err("free graph result slots"))?;
        Ok(())
    }
}

fn release_all(gpu: &Gpu, set: &mut Vec<Mapped>) -> Result<(), String> {
    let mut first: Option<String> = None;
    for m in set.iter_mut() {
        if let Err(e) = m.arena.release(&gpu.hip) {
            first.get_or_insert(format!("release alloc {}: {e}", m.spec.id));
        }
    }
    set.clear();
    first.map_or(Ok(()), Err)
}

// ---------------------------------------------------------------------------
// Role event-loop plumbing
// ---------------------------------------------------------------------------
fn read_command() -> Option<(String, Option<u64>)> {
    let mut line = String::new();
    match std::io::stdin().lock().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => {
            let mut it = line.split_whitespace();
            let cmd = it.next().unwrap_or("").to_string();
            let arg = it.next().and_then(|a| a.parse().ok());
            Some((cmd, arg))
        }
    }
}

fn inherited_sock(fd: RawFd) -> Result<OwnedFd, String> {
    // SAFETY: `fd` was inherited from the orchestrator for this role only.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    if fd_identity(fd).is_err() {
        return Err(format!("inherited socket fd {fd} is not open"));
    }
    Ok(owned)
}

fn hsaco_arg(a: &Args) -> Result<&str, String> {
    a.hsaco
        .as_deref()
        .ok_or_else(|| "--hsaco is required for this role".to_string())
}

// ---------------------------------------------------------------------------
// Role: identity
// ---------------------------------------------------------------------------
fn role_identity(a: &Args) -> Result<(), String> {
    let hip = HipRuntime::load().map_err(hip_err("load HIP runtime"))?;
    hip.set_device(a.device).map_err(hip_err("hipSetDevice"))?;
    let (maj, min) = hip.runtime_version().unwrap_or((0, 0));
    let arch = hip.get_arch(a.device).map_err(hip_err("get_arch"))?;
    let bdf = hip
        .device_pci_bus_id(a.device)
        .map_err(hip_err("pci bus id"))?;
    let uuid = hip
        .device_uuid(a.device)
        .unwrap_or_else(|_| "unknown".into());
    emit(
        "identity",
        &[
            ("arch", arch),
            ("bdf", bdf.to_lowercase()),
            ("uuid", uuid.replace(' ', "_")),
            ("hip_runtime", format!("{maj}.{min}")),
            (
                "devices",
                hip.device_count()
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ),
        ],
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Role: exporter
// ---------------------------------------------------------------------------
fn role_exporter(a: &Args) -> Result<(), String> {
    if a.fds.is_empty() {
        return Err("exporter needs --fds <socket,...>".into());
    }
    let socks: Vec<OwnedFd> = a
        .fds
        .iter()
        .map(|&f| inherited_sock(f))
        .collect::<Result<_, _>>()?;
    let gpu = Gpu::init(a.device, hsaco_arg(a)?)?;
    let hip = &gpu.hip;
    let granularity_note = {
        let arch = hip.get_arch(a.device).unwrap_or_default();
        let bdf = hip
            .device_pci_bus_id(a.device)
            .map_err(hip_err("pci bus id"))?;
        (arch, bdf.to_lowercase())
    };

    // Build: reserve exportable, map once (RW), fill by GPU kernel.
    let mut set: Vec<Mapped> = Vec::new();
    let built = (|| -> Result<(), String> {
        for (id, &req) in REQUESTED_BYTES.iter().enumerate() {
            let mut arena = VmmArena::reserve_exportable(hip, a.device, req as usize)
                .map_err(hip_err("reserve_exportable"))?;
            let reserved = arena.reserved_bytes();
            let mapped = arena
                .map_next(hip, reserved, &[])
                .map_err(hip_err("map_next"));
            if let Err(e) = mapped {
                let _ = arena.release(hip);
                return Err(e);
            }
            let bytes = arena.mapped_bytes() as u64;
            let spec = Spec::new(a.seed, id as u32, req, bytes);
            set.push(Mapped { spec, arena });
            let m = set.last().expect("pushed");
            let flags = m
                .arena
                .owner_access_flags(hip)
                .map_err(hip_err("hipMemGetAccess"))?;
            if flags != ACCESS_RW {
                return Err(format!(
                    "alloc {id}: pre-publish access {flags}, expected RW={ACCESS_RW}"
                ));
            }
            let buf = m
                .arena
                .buffer(bytes as usize)
                .map_err(hip_err("arena.buffer"))?;
            gpu.fill(buf.as_ptr(), spec.words(), spec.sm)?;
        }
        gpu.sync()
    })();
    if let Err(e) = built {
        let _ = release_all(&gpu, &mut set);
        return Err(e);
    }

    let result = (|| -> Result<(), String> {
        let total: u64 = set.iter().map(|m| m.spec.bytes).sum();
        let pre = verify_eager(&gpu, &set, "exporter RW", true)?;
        let neg = negative_control(&gpu, &set)?;
        for m in set.iter_mut() {
            m.arena
                .publish_read_only(hip)
                .map_err(hip_err("publish_read_only"))?;
            let flags = m
                .arena
                .owner_access_flags(hip)
                .map_err(hip_err("hipMemGetAccess"))?;
            if flags != ACCESS_RO || !m.arena.is_read_only() {
                return Err(format!(
                    "alloc {}: post-publish access {flags}, expected RO={ACCESS_RO}",
                    m.spec.id
                ));
            }
        }
        let post = verify_eager(&gpu, &set, "exporter RO", true)?;
        let mut fds: Vec<OwnedFd> = Vec::new();
        for m in &set {
            fds.push(m.arena.export_posix(hip).map_err(hip_err("export_posix"))?);
        }
        let cloexec = fds.iter().filter(|f| fd_is_cloexec(f.as_raw_fd())).count();
        emit(
            "ready",
            &[
                ("allocs", set.len().to_string()),
                ("physical_bytes", total.to_string()),
                ("rw_pre_publish", ACCESS_RW.to_string()),
                ("ro_post_publish", ACCESS_RO.to_string()),
                (
                    "gpu_words_verified",
                    (pre.gpu_words + post.gpu_words).to_string(),
                ),
                (
                    "readback_bytes",
                    (pre.readback_bytes + post.readback_bytes).to_string(),
                ),
                ("negative_control_mismatches", neg.to_string()),
                ("export_fds_cloexec", cloexec.to_string()),
                ("arch", granularity_note.0.clone()),
                ("bdf", granularity_note.1.clone()),
                (
                    "sizes",
                    set.iter()
                        .map(|m| format!("{}>{}", m.spec.requested, m.spec.bytes))
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
        );
        exporter_loop(a, &gpu, &mut set, &socks, &mut fds)
    })();
    if result.is_err() {
        let _ = release_all(&gpu, &mut set);
    }
    result
}

fn exporter_loop(
    a: &Args,
    gpu: &Gpu,
    set: &mut Vec<Mapped>,
    socks: &[OwnedFd],
    fds: &mut Vec<OwnedFd>,
) -> Result<(), String> {
    let total: u64 = set.iter().map(|m| m.spec.bytes).sum();
    loop {
        let Some((cmd, _)) = read_command() else {
            return Err("orchestrator closed stdin".into());
        };
        match cmd.as_str() {
            "SERVE" => {
                for (si, sock) in socks.iter().enumerate() {
                    let raw = sock.as_raw_fd();
                    let to = a.ipc_timeout_ms;
                    send_frame(
                        raw,
                        &mk_frame(KIND_MANIFEST, 0, set.len() as u64, total, 0, a.seed),
                        None,
                        to,
                        &format!("send MANIFEST to importer {si}"),
                    )?;
                    for (i, m) in set.iter().enumerate() {
                        send_frame(
                            raw,
                            &mk_frame(
                                KIND_ALLOC,
                                i as u64,
                                set.len() as u64,
                                m.spec.bytes,
                                m.spec.requested,
                                0,
                            ),
                            Some(fds[i].as_fd_borrowed()),
                            to,
                            &format!("send ALLOC {i} to importer {si}"),
                        )?;
                    }
                }
                for (si, sock) in socks.iter().enumerate() {
                    let (f, extra_fds) = recv_frame(
                        sock.as_raw_fd(),
                        a.ipc_timeout_ms,
                        &format!("recv ACK from importer {si}"),
                    )?;
                    if !extra_fds.is_empty() {
                        return Err(format!(
                            "importer {si} ACK carried {} unexpected FDs",
                            extra_fds.len()
                        ));
                    }
                    if f[2] != KIND_ACK || f[4] != set.len() as u64 || f[5] != total {
                        return Err(format!(
                            "importer {si}: bad ACK {f:?} (want count={} bytes={total})",
                            set.len()
                        ));
                    }
                }
                let n = fds.len();
                for fd in fds.drain(..) {
                    close_checked(fd)?;
                }
                emit(
                    "served",
                    &[
                        ("importers", socks.len().to_string()),
                        ("acks", socks.len().to_string()),
                        ("fds_sent_each", n.to_string()),
                        ("exporter_fds_closed", n.to_string()),
                    ],
                );
            }
            "VERIFY" => {
                let s = verify_eager(gpu, set, "exporter after FD close", true)?;
                emit(
                    "verified",
                    &[
                        ("gpu_words", s.gpu_words.to_string()),
                        ("readback_bytes", s.readback_bytes.to_string()),
                    ],
                );
            }
            "EXIT_CLEAN" => {
                release_all(gpu, set)?;
                emit("exiting", &[("mode", "clean".into())]);
                std::process::exit(0);
            }
            "EXIT_ABRUPT" => {
                emit("exiting", &[("mode", "abrupt".into())]);
                // SAFETY: deliberate hard exit: no unmap/release/FD close/HIP teardown.
                unsafe { sys::_exit(1) }
            }
            "IDLE" => emit("idle", &[]),
            other => return Err(format!("exporter: unknown command {other:?}")),
        }
    }
}

trait AsFdBorrowed {
    fn as_fd_borrowed(&self) -> BorrowedFd<'_>;
}

impl AsFdBorrowed for OwnedFd {
    fn as_fd_borrowed(&self) -> BorrowedFd<'_> {
        std::os::fd::AsFd::as_fd(self)
    }
}

// ---------------------------------------------------------------------------
// Role: importer
// ---------------------------------------------------------------------------
struct Importer {
    gpu: Gpu,
    set: Vec<Mapped>,
    specs: Vec<Spec>,
    fds: Vec<OwnedFd>,
    idents: Vec<(u64, u64)>,
    graph: Option<GraphState>,
    ranges: Vec<(usize, usize)>,
}

fn import_all(gpu: &Gpu, specs: &[Spec], fds: &[OwnedFd]) -> Result<Vec<Mapped>, String> {
    let mut set: Vec<Mapped> = Vec::new();
    for (spec, fd) in specs.iter().zip(fds) {
        match VmmArena::import_posix(
            &gpu.hip,
            gpu.device,
            fd.as_fd_borrowed(),
            spec.bytes as usize,
        ) {
            Ok(arena) => set.push(Mapped { spec: *spec, arena }),
            Err(e) => {
                let _ = release_all(gpu, &mut set);
                return Err(format!("import_posix alloc {}: {e}", spec.id));
            }
        }
        let m = set.last().expect("pushed");
        let flags = m
            .arena
            .owner_access_flags(&gpu.hip)
            .map_err(hip_err("hipMemGetAccess"))?;
        if flags != ACCESS_RO || !m.arena.is_read_only() {
            let _ = release_all(gpu, &mut set);
            return Err(format!(
                "alloc {}: imported access {flags}, expected RO={ACCESS_RO}",
                spec.id
            ));
        }
        if m.arena.mapped_bytes() as u64 != spec.bytes {
            let got = m.arena.mapped_bytes();
            let _ = release_all(gpu, &mut set);
            return Err(format!(
                "alloc {}: imported {got} B, manifest says {}",
                spec.id, spec.bytes
            ));
        }
    }
    Ok(set)
}

fn role_importer(a: &Args) -> Result<(), String> {
    if a.fds.len() != 1 {
        return Err("importer needs exactly one --fd <socket>".into());
    }
    let sock = inherited_sock(a.fds[0])?;
    let gpu = Gpu::init(a.device, hsaco_arg(a)?)?;

    // Warm-up: force lazy HIP context/queue/graph memory before the "no second
    // physical copy" accounting window opens.
    {
        let words = 1u64 << 18;
        let buf = gpu
            .hip
            .malloc((words * 8) as usize)
            .map_err(hip_err("warmup malloc"))?;
        gpu.fill(buf.as_ptr(), words, 7)?;
        gpu.sync()?;
        gpu.hip.free(buf).map_err(hip_err("warmup free"))?;
        let s = gpu
            .hip
            .stream_create_non_blocking()
            .map_err(hip_err("warmup stream"))?;
        gpu.hip
            .stream_destroy(s)
            .map_err(hip_err("warmup stream destroy"))?;
    }
    let mut imp = Importer {
        gpu,
        set: Vec::new(),
        specs: Vec::new(),
        fds: Vec::new(),
        idents: Vec::new(),
        graph: None,
        ranges: Vec::new(),
    };
    emit("warm", &[("name", a.name.clone())]);

    loop {
        let Some((cmd, arg)) = read_command() else {
            return Err("orchestrator closed stdin".into());
        };
        match cmd.as_str() {
            "IMPORT" => importer_import(a, &mut imp, &sock)?,
            "VERIFY" => {
                if imp.set.is_empty() {
                    return Err("VERIFY before IMPORT".into());
                }
                let neg = negative_control(&imp.gpu, &imp.set)?;
                let s = verify_eager(
                    &imp.gpu,
                    &imp.set,
                    &format!("importer {} eager", a.name),
                    true,
                )?;
                emit(
                    "verified",
                    &[
                        ("gpu_words", s.gpu_words.to_string()),
                        ("readback_bytes", s.readback_bytes.to_string()),
                        ("negative_control_mismatches", neg.to_string()),
                    ],
                );
            }
            "CYCLES" => importer_cycles(a, &mut imp, arg.unwrap_or(u64::from(a.cycles)) as u32)?,
            "GRAPH" | "FRESH_GRAPH" => {
                let n = arg.unwrap_or(u64::from(a.replays)) as u32;
                if let Some(old) = imp.graph.take() {
                    old.destroy(&imp.gpu)?;
                }
                let g = GraphState::capture(&imp.gpu, &imp.set)?;
                g.replay(
                    &imp.gpu,
                    n,
                    &format!("importer {} {}", a.name, cmd.to_lowercase()),
                )?;
                imp.graph = Some(g);
                emit(
                    &cmd.to_lowercase(),
                    &[
                        ("replays", n.to_string()),
                        ("allocs", imp.set.len().to_string()),
                    ],
                );
            }
            "REPLAY" => {
                let n = arg.unwrap_or(u64::from(a.replays)) as u32;
                let g = imp
                    .graph
                    .as_ref()
                    .ok_or("REPLAY without a captured graph")?;
                g.replay(
                    &imp.gpu,
                    n,
                    &format!("importer {} replay of earlier graph", a.name),
                )?;
                emit("replay", &[("replays", n.to_string())]);
            }
            "CLOSE_FDS" => {
                let n = imp.fds.len();
                for fd in imp.fds.drain(..) {
                    close_checked(fd)?;
                }
                let residual = count_fds_with_identity(&imp.idents)?;
                emit(
                    "closed_fds",
                    &[
                        ("closed", n.to_string()),
                        ("residual_same_inode_fds", residual.to_string()),
                    ],
                );
            }
            "EXIT" => {
                if let Some(g) = imp.graph.take() {
                    g.destroy(&imp.gpu)?;
                }
                release_all(&imp.gpu, &mut imp.set)?;
                emit(
                    "exiting",
                    &[("retired_va_bytes", retired_va_bytes().to_string())],
                );
                std::process::exit(0);
            }
            other => return Err(format!("importer: unknown command {other:?}")),
        }
    }
}

fn importer_import(a: &Args, imp: &mut Importer, sock: &OwnedFd) -> Result<(), String> {
    if !imp.set.is_empty() {
        return Err("IMPORT twice".into());
    }
    let raw = sock.as_raw_fd();
    let to = a.ipc_timeout_ms;
    let (m, mfds) = recv_frame(raw, to, "recv MANIFEST")?;
    if !mfds.is_empty() {
        return Err("MANIFEST carried FDs".into());
    }
    if m[2] != KIND_MANIFEST {
        return Err(format!("expected MANIFEST, got kind {}", m[2]));
    }
    let count = m[4];
    let total = m[5];
    let seed = m[7];
    if count == 0 || count > 32 {
        return Err(format!("manifest count {count} outside 1..=32"));
    }
    let mut specs = Vec::new();
    let mut fds: Vec<OwnedFd> = Vec::new();
    let mut sum = 0u64;
    for i in 0..count {
        let (f, mut got) = recv_frame(raw, to, &format!("recv ALLOC {i}"))?;
        if f[2] != KIND_ALLOC || f[3] != i || f[4] != count {
            return Err(format!("ALLOC {i}: bad frame {f:?}"));
        }
        if got.len() != 1 {
            return Err(format!(
                "ALLOC {i}: expected exactly 1 FD, got {}",
                got.len()
            ));
        }
        let fd = got.pop().expect("one fd");
        if !fd_is_cloexec(fd.as_raw_fd()) {
            return Err(format!(
                "ALLOC {i}: received FD lacks FD_CLOEXEC (MSG_CMSG_CLOEXEC ignored?)"
            ));
        }
        let (bytes, requested) = (f[5], f[6]);
        if bytes == 0 || bytes % 8 != 0 || bytes < requested {
            return Err(format!(
                "ALLOC {i}: implausible sizes bytes={bytes} requested={requested}"
            ));
        }
        sum = sum.checked_add(bytes).ok_or("byte total overflow")?;
        specs.push(Spec::new(seed, i as u32, requested, bytes));
        imp.idents.push(fd_identity(fd.as_raw_fd())?);
        fds.push(fd);
    }
    if sum != total {
        return Err(format!(
            "manifest total {total} != sum of allocations {sum}"
        ));
    }
    let set = import_all(&imp.gpu, &specs, &fds)?;
    for m in &set {
        imp.ranges
            .push((m.arena.base_address(), m.arena.reserved_bytes()));
    }
    imp.set = set;
    imp.specs = specs;
    imp.fds = fds;
    send_frame(
        raw,
        &mk_frame(KIND_ACK, 0, count, total, 0, 0),
        None,
        to,
        "send ACK",
    )?;
    emit(
        "imported",
        &[
            ("name", a.name.clone()),
            ("allocs", count.to_string()),
            ("physical_bytes", total.to_string()),
            ("access_ro", ACCESS_RO.to_string()),
            ("fds_cloexec", imp.fds.len().to_string()),
        ],
    );
    Ok(())
}

fn importer_cycles(a: &Args, imp: &mut Importer, cycles: u32) -> Result<(), String> {
    let mut retired_total = 0usize;
    let rv_start = retired_va_bytes();
    for k in 1..=cycles {
        let mut set = import_all(&imp.gpu, &imp.specs, &imp.fds)?;
        for m in &set {
            let (lo, len) = (m.arena.base_address(), m.arena.reserved_bytes());
            for &(blo, blen) in &imp.ranges {
                if lo < blo + blen && blo < lo + len {
                    let _ = release_all(&imp.gpu, &mut set);
                    return Err(format!(
                        "cycle {k}: VA 0x{lo:x}+{len} overlaps earlier range 0x{blo:x}+{blen} (retired VA reused)"
                    ));
                }
            }
            imp.ranges.push((lo, len));
        }
        let reserved: usize = set.iter().map(|m| m.arena.reserved_bytes()).sum();
        let verified = verify_eager(
            &imp.gpu,
            &set,
            &format!("importer {} cycle {k}", a.name),
            true,
        );
        let rv0 = retired_va_bytes();
        let released = release_all(&imp.gpu, &mut set);
        verified?;
        released?;
        let rv1 = retired_va_bytes();
        if rv1 < rv0 + reserved {
            return Err(format!(
                "cycle {k}: retired VA grew {} B, expected at least {reserved} B",
                rv1.saturating_sub(rv0)
            ));
        }
        retired_total += reserved;
    }
    emit(
        "cycles",
        &[
            ("cycles", cycles.to_string()),
            ("retired_ranges_recorded", imp.ranges.len().to_string()),
            ("va_retired_by_cycles", retired_total.to_string()),
            ("retired_va_bytes", retired_va_bytes().to_string()),
            (
                "retired_va_growth",
                retired_va_bytes().saturating_sub(rv_start).to_string(),
            ),
        ],
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------
struct Role {
    name: &'static str,
    pid: i32,
    start_ticks: u64,
    stdin: Option<ChildStdin>,
    events: Receiver<String>,
    exit: Receiver<ExitStatus>,
    exited: Option<ExitStatus>,
}

fn read_start_ticks(pid: i32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &stat[stat.rfind(')')? + 1..];
    // fields after ")" start at field 3 (state); starttime is field 22.
    rest.split_whitespace().nth(19)?.parse().ok()
}

fn describe(st: &ExitStatus) -> String {
    match (st.code(), st.signal()) {
        (Some(c), _) => format!("exit code {c}"),
        (None, Some(s)) => format!("killed by signal {s}"),
        _ => "unknown status".into(),
    }
}

type Evt = HashMap<String, String>;

impl Role {
    fn spawn(
        name: &'static str,
        exe: &Path,
        role: &str,
        args: &[String],
        inherit: &[RawFd],
    ) -> Result<Role, String> {
        let mut cmd = Command::new(exe);
        cmd.arg("--role")
            .arg(role)
            .arg("--name")
            .arg(name)
            .args(args);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let fds = inherit.to_vec();
        // SAFETY: only async-signal-safe fcntl calls run between fork and exec.
        unsafe {
            cmd.pre_exec(move || {
                for &fd in &fds {
                    let fl = sys::fcntl(fd, sys::F_GETFD);
                    if fl < 0 || sys::fcntl(fd, sys::F_SETFD, fl & !sys::FD_CLOEXEC) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let mut child: Child = cmd
            .spawn()
            .map_err(|e| format!("spawn {name} ({role}): {e}"))?;
        let pid = child.id() as i32;
        let start_ticks = read_start_ticks(pid).unwrap_or(0);
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or("child stdout not piped")?;
        let (etx, events) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if etx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let (xtx, exit) = mpsc::channel::<ExitStatus>();
        std::thread::spawn(move || {
            if let Ok(st) = child.wait() {
                let _ = xtx.send(st);
            }
        });
        println!("SPAWN {name} role={role} pid={pid} starttime={start_ticks}");
        Ok(Role {
            name,
            pid,
            start_ticks,
            stdin,
            events,
            exit,
            exited: None,
        })
    }

    fn poll_exit(&mut self) -> Option<ExitStatus> {
        if self.exited.is_none() {
            self.exited = self.exit.try_recv().ok();
        }
        self.exited
    }

    fn wait_exit(&mut self, timeout: Duration) -> Result<ExitStatus, String> {
        if let Some(st) = self.exited {
            return Ok(st);
        }
        match self.exit.recv_timeout(timeout) {
            Ok(st) => {
                self.exited = Some(st);
                Ok(st)
            }
            Err(_) => Err(format!("{}: did not exit within {timeout:?}", self.name)),
        }
    }

    fn send(&mut self, cmd: &str) -> Result<(), String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| format!("{}: stdin closed", self.name))?;
        writeln!(stdin, "{cmd}")
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("{}: send {cmd}: {e}", self.name))
    }

    fn expect(&mut self, want: &str, timeout: Duration) -> Result<Evt, String> {
        loop {
            match self.events.recv_timeout(timeout) {
                Ok(line) => {
                    if let Some(msg) = line.strip_prefix("FAIL ") {
                        return Err(format!("{} reported failure: {msg}", self.name));
                    }
                    let Some(rest) = line.strip_prefix("EVT ") else {
                        println!("[{}] {line}", self.name);
                        continue;
                    };
                    let mut parts = rest.split_whitespace();
                    let name = parts.next().unwrap_or("");
                    let kv: Evt = parts
                        .filter_map(|p| {
                            p.split_once('=')
                                .map(|(k, v)| (k.to_string(), v.to_string()))
                        })
                        .collect();
                    if name != want {
                        return Err(format!("{}: expected event {want}, got {name}", self.name));
                    }
                    return Ok(kv);
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(format!(
                        "{}: timed out after {timeout:?} waiting for {want}",
                        self.name
                    ))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let st = self
                        .wait_exit(Duration::from_secs(10))
                        .map(|s| describe(&s))
                        .unwrap_or_else(|e| e);
                    return Err(format!(
                        "{}: output closed while waiting for {want} ({st})",
                        self.name
                    ));
                }
            }
        }
    }

    /// SIGTERM to the recorded PID, only if the PID still has the recorded start time.
    fn sigterm_recorded(&mut self) -> Result<(), String> {
        if self.poll_exit().is_some() {
            return Ok(());
        }
        match read_start_ticks(self.pid) {
            Some(t) if t == self.start_ticks && self.start_ticks != 0 => {
                // SAFETY: PID and start time match the process we spawned.
                if unsafe { sys::kill(self.pid, sys::SIGTERM) } != 0 {
                    return Err(format!(
                        "{}: kill(SIGTERM): {}",
                        self.name,
                        std::io::Error::last_os_error()
                    ));
                }
                println!(
                    "SIGTERM {} pid={} starttime={} (recorded child; no SIGKILL)",
                    self.name, self.pid, self.start_ticks
                );
                Ok(())
            }
            Some(_) => Err(format!(
                "{}: pid {} start time changed; refusing to signal",
                self.name, self.pid
            )),
            None => Ok(()),
        }
    }
}

struct Orch {
    roles: Vec<Role>,
    steps: u32,
    tmp: Option<PathBuf>,
    /// Other recorded child PIDs (the identity process) for attribution/absence checks.
    extra: Vec<(&'static str, i32)>,
}

impl Orch {
    fn pass(&mut self, step: &str, detail: &str) {
        self.steps += 1;
        println!("PASS {step} {detail}");
    }

    fn role(&mut self, name: &str) -> Result<&mut Role, String> {
        self.roles
            .iter_mut()
            .find(|r| r.name == name)
            .ok_or_else(|| format!("no role {name}"))
    }

    fn cmd(&mut self, role: &str, cmd: &str, want: &str, to: Duration) -> Result<Evt, String> {
        let r = self.role(role)?;
        r.send(cmd)?;
        r.expect(want, to)
    }

    /// Send `cmd` to B and C first so they run concurrently, then collect both events.
    fn cmd_bc(&mut self, cmd: &str, want: &str, to: Duration) -> Result<(Evt, Evt), String> {
        self.role("B")?.send(cmd)?;
        self.role("C")?.send(cmd)?;
        let b = self.role("B")?.expect(want, to)?;
        let c = self.role("C")?.expect(want, to)?;
        Ok((b, c))
    }

    fn expect_exit(
        &mut self,
        role: &str,
        to: Duration,
        want_code: Option<i32>,
        want_signal: Option<i32>,
    ) -> Result<String, String> {
        let r = self.role(role)?;
        let st = r.wait_exit(to)?;
        proc_absent(r.name, r.pid, r.start_ticks)?;
        let ok = match (want_code, want_signal) {
            (Some(c), _) => st.code() == Some(c),
            (None, Some(s)) => st.signal() == Some(s),
            _ => false,
        };
        if !ok {
            return Err(format!("{role}: unexpected exit status: {}", describe(&st)));
        }
        r.stdin = None;
        Ok(describe(&st))
    }

    fn cleanup(&mut self) {
        for r in &mut self.roles {
            if r.poll_exit().is_none() {
                if let Err(e) = r.sigterm_recorded() {
                    eprintln!("cleanup: {e}");
                }
            }
        }
        for r in &mut self.roles {
            if r.exited.is_none() {
                match r.wait_exit(Duration::from_secs(15)) {
                    Ok(st) => println!("CLEANUP {} {}", r.name, describe(&st)),
                    Err(e) => eprintln!(
                        "cleanup: {e} (no SIGKILL sent; pid {} may still be running)",
                        r.pid
                    ),
                }
            }
        }
        if let Some(dir) = self.tmp.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn get<'a>(e: &'a Evt, k: &str) -> Result<&'a str, String> {
    e.get(k)
        .map(String::as_str)
        .ok_or_else(|| format!("event missing field {k}"))
}

fn get_u64(e: &Evt, k: &str) -> Result<u64, String> {
    get(e, k)?
        .parse()
        .map_err(|_| format!("event field {k} is not a number"))
}

fn socketpair_seqpacket() -> Result<(OwnedFd, OwnedFd), String> {
    let mut sv = [0i32; 2];
    // SAFETY: sv is a live 2-int array.
    let rc = unsafe {
        sys::socketpair(
            sys::AF_UNIX,
            sys::SOCK_SEQPACKET | sys::SOCK_CLOEXEC,
            0,
            sv.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("socketpair: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: socketpair returned two fresh FDs owned by us.
    Ok(unsafe { (OwnedFd::from_raw_fd(sv[0]), OwnedFd::from_raw_fd(sv[1])) })
}

fn compile_kernel(arch: &str, dir: &Path) -> Result<PathBuf, String> {
    let src = dir.join("shared_vmm_fd_probe.hip");
    let obj = dir.join("shared_vmm_fd_probe.hsaco");
    std::fs::write(&src, KERNEL_SRC).map_err(|e| format!("write kernel source: {e}"))?;
    // Same toolchain resolution and codegen flags as rdna-compute's JIT compiler.
    let hipcc = hipfire_config::rocm::tool("hipcc").unwrap_or_else(|| PathBuf::from("hipcc"));
    let mut cmd = Command::new(&hipcc);
    cmd.args([
        "--genco",
        &format!("--offload-arch={arch}"),
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
        "-o",
    ])
    .arg(&obj)
    .arg(&src);
    if let Some(root) = hipfire_config::rocm::compiler_env_root(&hipcc) {
        cmd.env("ROCM_PATH", root);
    }
    let out = cmd
        .output()
        .map_err(|e| format!("run {}: {e}", hipcc.display()))?;
    if !out.status.success() {
        return Err(format!(
            "{} failed ({}): {}",
            hipcc.display(),
            describe(&out.status),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(obj)
}

struct Accounting {
    mode: AcctMode,
    path: Option<PathBuf>,
    baseline: Option<u64>,
    bdf: String,
    /// KFD gpu_id of the target card (for per-PID KFD VRAM), if resolvable.
    gpu_id: Option<String>,
}

impl Accounting {
    fn new(mode: AcctMode, bdf: &str) -> Accounting {
        let path = PathBuf::from(format!("/sys/bus/pci/devices/{bdf}/mem_info_vram_used"));
        let path = (mode != AcctMode::Off && path.exists()).then_some(path);
        let gpu_id = if mode == AcctMode::Off {
            None
        } else {
            match kfd_gpu_id(bdf) {
                Ok(id) => Some(id),
                Err(e) => {
                    println!("NOTE per-PID KFD attribution unavailable: {e}");
                    None
                }
            }
        };
        Accounting {
            mode,
            path,
            baseline: None,
            bdf: bdf.to_string(),
            gpu_id,
        }
    }

    fn read(&self) -> Option<u64> {
        let p = self.path.as_ref()?;
        std::fs::read_to_string(p).ok()?.trim().parse().ok()
    }

    fn sample(&self, label: &str) -> Option<u64> {
        let v = self.read();
        match v {
            Some(v) => println!(
                "ACCT {label} vram_used={v} delta_baseline={}",
                v as i64 - self.baseline.unwrap_or(v) as i64
            ),
            None if self.mode != AcctMode::Off => println!("ACCT {label} unavailable"),
            None => {}
        }
        v
    }

    fn gating(&self) -> bool {
        self.mode == AcctMode::Gate && self.path.is_some() && self.baseline.is_some()
    }

    fn gate(&self, name: &str, ok: bool, detail: String) -> Result<(), String> {
        if !self.gating() {
            println!(
                "REPORT {name} {detail} ({})",
                if ok {
                    "within gate"
                } else {
                    "outside gate, not gating"
                }
            );
            return Ok(());
        }
        if ok {
            println!("PASS {name} {detail}");
            Ok(())
        } else {
            Err(format!("{name}: {detail}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Per-PID VRAM attribution (diagnostics only; never gates, never signals)
// ---------------------------------------------------------------------------

/// After a role has been reaped (`Child::wait`), its `/proc/<pid>` must be gone
/// (or the PID reused by a process with a different start time).
fn proc_absent(name: &str, pid: i32, start_ticks: u64) -> Result<(), String> {
    match read_start_ticks(pid) {
        None => {
            println!("PROC_ABSENT {name} pid={pid}");
            Ok(())
        }
        Some(t) if t != start_ticks => {
            println!(
                "PROC_ABSENT {name} pid={pid} (pid reused by a different process, start differs)"
            );
            Ok(())
        }
        Some(_) => Err(format!(
            "{name}: /proc/{pid} still present with the recorded start time after wait"
        )),
    }
}

/// (domain, KFD location_id) for a `dddd:bb:dd.f` PCI address.
fn bdf_location(bdf: &str) -> Option<(u64, u64)> {
    let mut p = bdf.split(':');
    let dom = u64::from_str_radix(p.next()?, 16).ok()?;
    let bus = u64::from_str_radix(p.next()?, 16).ok()?;
    let (dev, func) = p.next()?.split_once('.')?;
    let dev = u64::from_str_radix(dev, 16).ok()?;
    let func = u64::from_str_radix(func, 16).ok()?;
    Some((dom, (bus << 8) | (dev << 3) | func))
}

/// KFD `gpu_id` of the node whose PCI domain/location match `bdf`.
fn kfd_gpu_id(bdf: &str) -> Result<String, String> {
    let (dom, loc) = bdf_location(bdf).ok_or_else(|| format!("bad PCI address {bdf}"))?;
    let nodes = "/sys/class/kfd/kfd/topology/nodes";
    let rd = std::fs::read_dir(nodes).map_err(|e| format!("read_dir {nodes}: {e}"))?;
    for entry in rd.flatten() {
        let Ok(props) = std::fs::read_to_string(entry.path().join("properties")) else {
            continue;
        };
        let mut kv: HashMap<&str, &str> = HashMap::new();
        for line in props.lines() {
            let mut it = line.split_whitespace();
            if let (Some(k), Some(v), None) = (it.next(), it.next(), it.next()) {
                kv.insert(k, v);
            }
        }
        if kv.get("location_id").copied() == Some(loc.to_string().as_str())
            && kv.get("domain").copied() == Some(dom.to_string().as_str())
        {
            let id = std::fs::read_to_string(entry.path().join("gpu_id"))
                .map_err(|e| format!("read gpu_id: {e}"))?;
            return Ok(id.trim().to_string());
        }
    }
    Err(format!(
        "no KFD node for PCI {bdf} (domain {dom}, location_id {loc})"
    ))
}

/// Per-PID KFD VRAM on the target GPU: `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` (bytes).
fn kfd_vram_by_pid(gpu_id: &str) -> Result<Vec<(i32, u64)>, String> {
    let dir = "/sys/class/kfd/kfd/proc";
    let rd = std::fs::read_dir(dir).map_err(|e| format!("read_dir {dir}: {e}"))?;
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<i32>().ok())
        else {
            continue;
        };
        let v = std::fs::read_to_string(entry.path().join(format!("vram_{gpu_id}")))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        if let Some(v) = v {
            out.push((pid, v));
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// Per-PID `drm-memory-vram` (bytes) from `/proc/<pid>/fdinfo/*` entries whose
/// `drm-pdev` is the target BDF, deduplicated by `drm-client-id`. Also returns
/// how many processes' fdinfo directories were not readable (EACCES).
fn fdinfo_vram_by_pid(bdf: &str) -> (Vec<(i32, u64)>, usize) {
    let mut out = Vec::new();
    let mut denied = 0usize;
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return (out, denied);
    };
    for entry in rd.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<i32>().ok())
        else {
            continue;
        };
        let fdinfo = match std::fs::read_dir(entry.path().join("fdinfo")) {
            Ok(d) => d,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    denied += 1;
                }
                continue;
            }
        };
        let mut clients: HashMap<String, u64> = HashMap::new();
        for f in fdinfo.flatten() {
            let Ok(txt) = std::fs::read_to_string(f.path()) else {
                continue;
            };
            if !txt.contains("drm-pdev:") {
                continue;
            }
            let (mut pdev, mut client, mut vram_kib) = (None, None, 0u64);
            for line in txt.lines() {
                let Some((k, v)) = line.split_once(':') else {
                    continue;
                };
                let v = v.trim();
                match k.trim() {
                    "drm-pdev" => pdev = Some(v.to_lowercase()),
                    "drm-client-id" => client = Some(v.to_string()),
                    "drm-memory-vram" => {
                        vram_kib = v
                            .split_whitespace()
                            .next()
                            .and_then(|n| n.parse().ok())
                            .unwrap_or(0)
                    }
                    _ => {}
                }
            }
            if pdev.as_deref() == Some(bdf) {
                let key = client.unwrap_or_else(|| f.file_name().to_string_lossy().into_owned());
                let e = clients.entry(key).or_insert(0);
                *e = (*e).max(vram_kib);
            }
        }
        if !clients.is_empty() {
            out.push((pid, clients.values().sum::<u64>() * 1024));
        }
    }
    out.sort_unstable();
    (out, denied)
}

fn proc_comm(pid: i32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|s| s.trim().replace(' ', "_"))
        .unwrap_or_else(|_| "?".into())
}

/// Print per-PID VRAM for own recorded PIDs and for every other process with
/// VRAM on the target BDF (accessible ones only; nothing is signalled).
/// Returns the KFD (own_sum, others_sum) when the KFD source is available.
fn diag(orch: &mut Orch, acct: &Accounting, label: &str) -> Option<(u64, u64)> {
    if acct.mode == AcctMode::Off {
        return None;
    }
    let mut known: Vec<(String, i32, bool)> = Vec::new();
    for r in &mut orch.roles {
        let alive = r.poll_exit().is_none();
        known.push((r.name.to_string(), r.pid, alive));
    }
    for (n, p) in &orch.extra {
        known.push(((*n).to_string(), *p, false));
    }
    let who = |pid: i32| known.iter().find(|(_, p, _)| *p == pid);
    let mut kfd_result = None;
    match acct.gpu_id.as_deref() {
        Some(gid) => match kfd_vram_by_pid(gid) {
            Ok(list) => {
                let (mut own, mut others, mut n_others) = (0u64, 0u64, 0usize);
                for &(pid, v) in &list {
                    match who(pid) {
                        Some((name, _, alive)) => {
                            own += v;
                            println!("DIAG {label} src=kfd gpu_id={gid} who={name} pid={pid} alive={alive} vram_bytes={v}");
                        }
                        None => {
                            others += v;
                            if v > 0 {
                                n_others += 1;
                                println!("DIAG {label} src=kfd gpu_id={gid} who=OTHER pid={pid} comm={} vram_bytes={v}", proc_comm(pid));
                            }
                        }
                    }
                }
                let listed_exited: Vec<String> = known
                    .iter()
                    .filter(|(_, p, alive)| !*alive && list.iter().any(|(lp, _)| lp == p))
                    .map(|(n, p, _)| format!("{n}:{p}"))
                    .collect();
                println!(
                    "DIAG {label} src=kfd own_sum={own} others_sum={others} others_with_vram={n_others} own_exited_but_still_listed=[{}]",
                    listed_exited.join(",")
                );
                kfd_result = Some((own, others));
            }
            Err(e) => println!("DIAG {label} src=kfd unavailable: {e}"),
        },
        None => println!(
            "DIAG {label} src=kfd unavailable (no KFD node for {})",
            acct.bdf
        ),
    }
    let (list, denied) = fdinfo_vram_by_pid(&acct.bdf);
    let (mut own, mut others) = (0u64, 0u64);
    for &(pid, v) in &list {
        match who(pid) {
            Some((name, _, alive)) => {
                own += v;
                println!("DIAG {label} src=fdinfo who={name} pid={pid} alive={alive} drm_memory_vram_bytes={v}");
            }
            None => {
                others += v;
                println!(
                    "DIAG {label} src=fdinfo who=OTHER pid={pid} comm={} drm_memory_vram_bytes={v}",
                    proc_comm(pid)
                );
            }
        }
    }
    println!(
        "DIAG {label} src=fdinfo own_sum={own} others_sum={others} processes_not_readable={denied}"
    );
    kfd_result
}

fn run(orch: &mut Orch, a: &Args) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let to = a.step_timeout;
    let common = |hsaco: Option<&Path>| -> Vec<String> {
        let mut v = vec![
            "--device".into(),
            a.device.to_string(),
            "--seed".into(),
            a.seed.to_string(),
            "--ipc-timeout-ms".into(),
            a.ipc_timeout_ms.to_string(),
            "--cycles".into(),
            a.cycles.to_string(),
            "--replays".into(),
            a.replays.to_string(),
        ];
        if let Some(h) = hsaco {
            v.push("--hsaco".into());
            v.push(h.display().to_string());
        }
        v
    };

    // --- identity (separate short-lived process; also yields arch/PCI id) ----
    let mut ident = Role::spawn("I", &exe, "identity", &common(None), &[])?;
    let id_evt = ident.expect("identity", to);
    let id_exit = ident.wait_exit(Duration::from_secs(30));
    let id_evt = id_evt?;
    if !id_exit?.success() {
        return Err("identity process failed".into());
    }
    proc_absent("I", ident.pid, ident.start_ticks)?;
    orch.extra.push(("I", ident.pid));
    let arch = get(&id_evt, "arch")?.to_string();
    let bdf = get(&id_evt, "bdf")?.to_string();
    orch.pass(
        "IDENTITY",
        &format!(
            "arch={arch} bdf={bdf} uuid={} hip_runtime={}",
            get(&id_evt, "uuid")?,
            get(&id_evt, "hip_runtime")?
        ),
    );

    // --- kernel code object -------------------------------------------------
    let dir = std::env::temp_dir().join(format!("shared_vmm_fd_probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    orch.tmp = Some(dir.clone());
    let hsaco = match &a.hsaco {
        Some(p) => PathBuf::from(p),
        None => compile_kernel(&arch, &dir)?,
    };
    orch.pass(
        "KERNEL_COMPILED",
        &format!("hsaco={} arch={arch}", hsaco.display()),
    );

    let mut acct = Accounting::new(a.acct, &bdf);
    acct.baseline = acct.read();
    let baseline = acct.sample("baseline");
    let base_diag = diag(orch, &acct, "baseline");
    if a.acct == AcctMode::Gate && !acct.gating() {
        println!("NOTE accounting gates skipped: sysfs mem_info_vram_used for {bdf} unavailable");
    }

    // --- sockets, exporter ---------------------------------------------------
    let (ab_a, ab_b) = socketpair_seqpacket()?;
    let (ac_a, ac_c) = socketpair_seqpacket()?;
    let mut a_args = common(Some(&hsaco));
    a_args.push("--fds".into());
    a_args.push(format!("{},{}", ab_a.as_raw_fd(), ac_a.as_raw_fd()));
    let a_role = Role::spawn(
        "A",
        &exe,
        "exporter",
        &a_args,
        &[ab_a.as_raw_fd(), ac_a.as_raw_fd()],
    )?;
    orch.roles.push(a_role);
    drop((ab_a, ac_a));
    let ready = orch.role("A")?.expect("ready", to)?;
    let w = get_u64(&ready, "physical_bytes")?;
    orch.pass(
        "EXPORTER_READY_PATTERN_RW_RO",
        &format!(
            "allocs={} W={w} rw_pre_publish={} ro_post_publish={} gpu_words={} readback_bytes={} negctl_mismatches={} sizes(requested>physical)={}",
            get(&ready, "allocs")?,
            get(&ready, "rw_pre_publish")?,
            get(&ready, "ro_post_publish")?,
            get(&ready, "gpu_words_verified")?,
            get(&ready, "readback_bytes")?,
            get(&ready, "negative_control_mismatches")?,
            get(&ready, "sizes")?
        ),
    );
    let after_ready = acct.sample("exporter_ready");
    diag(orch, &acct, "exporter_ready");
    if let (Some(b), Some(r)) = (baseline, after_ready) {
        acct.gate(
            "ACCT_EXPORT_ALLOC_PHYSICAL",
            (r as f64 - b as f64) >= 0.9 * w as f64,
            format!("delta={} W={w}", r as i64 - b as i64),
        )?;
    }

    // --- importers -----------------------------------------------------------
    let b_role = Role::spawn(
        "B",
        &exe,
        "importer",
        &common(Some(&hsaco))
            .into_iter()
            .chain(["--fd".into(), ab_b.as_raw_fd().to_string()])
            .collect::<Vec<_>>(),
        &[ab_b.as_raw_fd()],
    )?;
    orch.roles.push(b_role);
    let c_role = Role::spawn(
        "C",
        &exe,
        "importer",
        &common(Some(&hsaco))
            .into_iter()
            .chain(["--fd".into(), ac_c.as_raw_fd().to_string()])
            .collect::<Vec<_>>(),
        &[ac_c.as_raw_fd()],
    )?;
    orch.roles.push(c_role);
    drop((ab_b, ac_c));
    orch.role("B")?.expect("warm", to)?;
    orch.role("C")?.expect("warm", to)?;
    let after_warm = acct.sample("importers_warm");
    diag(orch, &acct, "importers_warm");

    // --- serve / import ------------------------------------------------------
    orch.role("A")?.send("SERVE")?;
    orch.role("B")?.send("IMPORT")?;
    orch.role("C")?.send("IMPORT")?;
    let served = orch.role("A")?.expect("served", to)?;
    let ib = orch.role("B")?.expect("imported", to)?;
    let ic = orch.role("C")?.expect("imported", to)?;
    orch.pass(
        "SCM_RIGHTS_SERVE_IMPORT_RO",
        &format!(
            "importers={} acks={} fds_sent_each={} exporter_fds_closed={} B_allocs={} C_allocs={} B_bytes={} C_bytes={} access_ro={} B_fds_cloexec={} C_fds_cloexec={}",
            get(&served, "importers")?, get(&served, "acks")?, get(&served, "fds_sent_each")?, get(&served, "exporter_fds_closed")?,
            get(&ib, "allocs")?, get(&ic, "allocs")?, get(&ib, "physical_bytes")?, get(&ic, "physical_bytes")?,
            get(&ib, "access_ro")?, get(&ib, "fds_cloexec")?, get(&ic, "fds_cloexec")?
        ),
    );
    let after_import = acct.sample("importers_imported");
    diag(orch, &acct, "importers_imported");
    if let (Some(w0), Some(i1)) = (after_warm, after_import) {
        let d = i1 as i64 - w0 as i64;
        acct.gate(
            "ACCT_IMPORT_NO_SECOND_W",
            (d as f64) < 0.25 * w as f64,
            format!(
                "delta_both_importers={d} limit={} (second W would be {w} each)",
                (0.25 * w as f64) as u64
            ),
        )?;
    }

    let ev = orch.cmd("A", "VERIFY", "verified", to)?;
    orch.pass(
        "EXPORTER_VERIFY_AFTER_FD_CLOSE",
        &format!("readback_bytes={}", get(&ev, "readback_bytes")?),
    );

    // --- importer verification: eager, cycles, graph, FD close ---------------
    let (vb, vc) = orch.cmd_bc("VERIFY", "verified", to)?;
    orch.pass(
        "IMPORTERS_EAGER_GPU_AND_READBACK",
        &format!(
            "B_gpu_words={} C_gpu_words={} B_readback={} C_readback={} negctl_B={} negctl_C={}",
            get(&vb, "gpu_words")?,
            get(&vc, "gpu_words")?,
            get(&vb, "readback_bytes")?,
            get(&vc, "readback_bytes")?,
            get(&vb, "negative_control_mismatches")?,
            get(&vc, "negative_control_mismatches")?
        ),
    );
    let (cb, cc) = orch.cmd_bc(&format!("CYCLES {}", a.cycles), "cycles", to)?;
    orch.pass(
        "REPEATED_IMPORT_UNLOAD_RETIRED_VA",
        &format!("cycles={} B_va_retired={} C_va_retired={} B_ranges={} C_ranges={} (fresh VA each cycle, no overlap with any earlier range)",
            get(&cb, "cycles")?, get(&cb, "va_retired_by_cycles")?, get(&cc, "va_retired_by_cycles")?,
            get(&cb, "retired_ranges_recorded")?, get(&cc, "retired_ranges_recorded")?),
    );
    orch.cmd_bc(&format!("GRAPH {}", a.replays), "graph", to)?;
    orch.pass(
        "IMPORTERS_GRAPH_CAPTURE_REPLAY",
        &format!(
            "replays={} each (host result buffer poisoned before every replay)",
            a.replays
        ),
    );
    let (xb, xc) = orch.cmd_bc("CLOSE_FDS", "closed_fds", to)?;
    orch.pass(
        "IMPORTERS_CLOSE_RECEIVED_FDS",
        &format!(
            "B_closed={} C_closed={} B_residual_same_inode_fds={} C_residual_same_inode_fds={}",
            get(&xb, "closed")?,
            get(&xc, "closed")?,
            get(&xb, "residual_same_inode_fds")?,
            get(&xc, "residual_same_inode_fds")?
        ),
    );
    orch.cmd_bc("VERIFY", "verified", to)?;
    orch.cmd_bc(&format!("REPLAY {}", a.replays), "replay", to)?;
    orch.pass(
        "IMPORTERS_VERIFY_AFTER_FD_CLOSE",
        "eager + graph replay on mappings whose FDs are closed",
    );

    // --- exporter exits -------------------------------------------------------
    let exit_desc = match a.exit_mode {
        ExitMode::Clean => {
            orch.cmd("A", "EXIT_CLEAN", "exiting", to)?;
            orch.expect_exit("A", to, Some(0), None)?
        }
        ExitMode::Abrupt => {
            orch.cmd("A", "EXIT_ABRUPT", "exiting", to)?;
            orch.expect_exit("A", to, Some(1), None)?
        }
        ExitMode::Sigterm => {
            orch.role("A")?.sigterm_recorded()?;
            orch.expect_exit("A", to, None, Some(sys::SIGTERM))?
        }
    };
    orch.pass(
        "EXPORTER_EXIT",
        &format!("mode={} status={exit_desc}", a.exit_mode.as_str()),
    );
    let after_a_exit = acct.sample("exporter_exited");
    diag(orch, &acct, "exporter_exited");
    if let (Some(b), Some(x)) = (baseline, after_a_exit) {
        acct.gate(
            "ACCT_EXPORTER_EXIT_W_RETAINED",
            (x as f64 - b as f64) >= 0.9 * w as f64,
            format!(
                "used_minus_baseline={} W={w} (importers still hold W)",
                x as i64 - b as i64
            ),
        )?;
    }
    orch.cmd_bc("VERIFY", "verified", to)?;
    orch.cmd_bc(&format!("REPLAY {}", a.replays), "replay", to)?;
    orch.cmd_bc(&format!("FRESH_GRAPH {}", a.replays), "fresh_graph", to)?;
    orch.pass(
        "POST_EXPORTER_EXIT_EAGER_GRAPH_REPLAY_FRESH_CAPTURE_BC",
        "B and C: eager+readback, replay of pre-exit graph, fresh capture+replay",
    );

    // --- independent importer exit -------------------------------------------
    orch.cmd("B", "EXIT", "exiting", to)?;
    let bst = orch.expect_exit("B", to, Some(0), None)?;
    let after_b_exit = acct.sample("importer_B_exited");
    diag(orch, &acct, "importer_B_exited");
    for cmd in [
        "VERIFY".to_string(),
        format!("REPLAY {}", a.replays),
        format!("FRESH_GRAPH {}", a.replays),
    ] {
        let want = cmd.split_whitespace().next().unwrap_or("").to_lowercase();
        let want_evt = if want == "verify" {
            "verified"
        } else {
            want.as_str()
        };
        orch.cmd("C", &cmd, want_evt, to)?;
    }
    orch.pass(
        "IMPORTER_B_EXIT_INDEPENDENT_C_VERIFIES",
        &format!("B {bst}; C eager+replay+fresh capture still pass"),
    );
    if let (Some(b), Some(x)) = (baseline, after_b_exit) {
        acct.gate(
            "ACCT_B_EXIT_W_RETAINED",
            (x as f64 - b as f64) >= 0.9 * w as f64,
            format!(
                "used_minus_baseline={} W={w} (C still holds W)",
                x as i64 - b as i64
            ),
        )?;
    }
    orch.cmd("C", "EXIT", "exiting", to)?;
    let cst = orch.expect_exit("C", to, Some(0), None)?;
    orch.pass("IMPORTER_C_EXIT", &format!("C {cst}"));
    let fin = acct.sample("final");
    let fin_diag = diag(orch, &acct, "final");
    // One fixed, bounded delay (not a retry loop) to tell asynchronous driver
    // reclamation from a persistent difference; the gate below uses `fin` only.
    std::thread::sleep(Duration::from_millis(100));
    let fin_delayed = acct.sample("final_delayed_100ms");
    let fin_delayed_diag = diag(orch, &acct, "final_delayed_100ms");
    if let (Some(b), Some(f), Some(fd)) = (baseline, fin, fin_delayed) {
        println!(
            "ATTRIBUTION final_minus_baseline={} delayed_100ms_minus_baseline={} async_reclaim_observed={}",
            f as i64 - b as i64,
            fd as i64 - b as i64,
            f != fd
        );
        if let (Some((_, o0)), Some((own1, o1)), Some((own2, o2))) =
            (base_diag, fin_diag, fin_delayed_diag)
        {
            println!(
                "ATTRIBUTION kfd: others_sum baseline={o0} final={o1} delayed={o2}; others_delta_final={} others_delta_delayed={}; own_sum_at_final={own1} own_sum_at_delayed={own2}; final_minus_baseline_minus_others_delta={} (nonzero own_sum => our PID still listed)",
                o1 as i64 - o0 as i64,
                o2 as i64 - o0 as i64,
                (f as i64 - b as i64) - (o1 as i64 - o0 as i64)
            );
        }
    }
    if let (Some(b), Some(f)) = (baseline, fin) {
        let diff = f as i64 - b as i64;
        acct.gate(
            "ACCT_FINAL_PHYSICAL_RELEASED",
            diff.unsigned_abs() <= 32 << 20,
            format!("final_minus_baseline={diff} tolerance={}", 32u64 << 20),
        )?;
    }
    for r in &mut orch.roles {
        if r.poll_exit().is_none() {
            return Err(format!("{} is still running at the end", r.name));
        }
    }
    Ok(())
}

fn role_main(a: &Args) -> Result<(), String> {
    match a.role.as_str() {
        "identity" => role_identity(a),
        "exporter" => role_exporter(a),
        "importer" => role_importer(a),
        other => Err(format!("unknown role {other}")),
    }
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("FAIL {e}");
            std::process::exit(2);
        }
    };
    if a.role != "orchestrate" {
        match role_main(&a) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                fail_line(&e);
                std::process::exit(1);
            }
        }
    }
    let mut orch = Orch {
        roles: Vec::new(),
        steps: 0,
        tmp: None,
        extra: Vec::new(),
    };
    let result = run(&mut orch, &a);
    orch.cleanup();
    match result {
        Ok(()) => println!(
            "shared_vmm_fd_probe: PASS ({} steps, exporter-exit={})",
            orch.steps,
            a.exit_mode.as_str()
        ),
        Err(e) => {
            eprintln!("shared_vmm_fd_probe: FAIL {e}");
            println!(
                "shared_vmm_fd_probe: FAIL after {} passed steps: {e}",
                orch.steps
            );
            std::process::exit(1);
        }
    }
}
