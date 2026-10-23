// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Fixed-size G80 BF16 panel transport. See docs/npu/railgun-npu.md.
//! The relay owns all DRM objects; HIP resources remain on the calling thread.
use super::hip_runtime::{HipEvent, HipRuntime, HostRegistration, RelayApi};
use super::{Bo, Chain, Device, HwCtx, Prepared, ERT_STATE_COMPLETED};
use core::ffi::c_void;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};
use std::{
    cell::RefCell,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

pub const WAIT_TIMEOUT: Duration = Duration::from_secs(3);
/// Five times the measured ~60 ms full G80 chain envelope. GPU-ready polling
/// has its separate WAIT_TIMEOUT budget; queued producer work is not NPU time.
pub const EXPECTED_CHAIN_TIME: Duration = Duration::from_millis(60);
pub const CHAIN_WATCHDOG: Duration = Duration::from_millis(300);
pub const PANEL_N: usize = 1280;
pub const PANEL_K: usize = 2560;
/// Out-projection panels: five 512-column panels of K = 6144 (the G80 N512 program, its own PDI / hardware context).
pub const OUT_PANEL_N: usize = 512;
pub const OUT_PANEL_K: usize = 6144;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    GdnIn,
    QsaIn,
    GdnOut,
    QsaO,
}
impl Shape {
    pub const ALL: [Shape; 4] = [Self::GdnIn, Self::QsaIn, Self::GdnOut, Self::QsaO];
    pub fn is_out(self) -> bool {
        matches!(self, Self::GdnOut | Self::QsaO)
    }
    pub fn panels(self) -> usize {
        match self {
            Self::GdnIn => 13,
            Self::QsaIn => 11,
            Self::GdnOut | Self::QsaO => 5,
        }
    }
    /// Columns of one panel: 1280 (input shapes) or 512 (out shapes).
    pub fn panel_n(self) -> usize {
        if self.is_out() {
            OUT_PANEL_N
        } else {
            PANEL_N
        }
    }
    pub fn pitch(self) -> usize {
        self.panels() * self.panel_n()
    }
    fn index(self) -> usize {
        match self {
            Self::GdnIn => 0,
            Self::QsaIn => 1,
            Self::GdnOut => 2,
            Self::QsaO => 3,
        }
    }
}

/// One row-major design per shape, with col0=0; every panel reuses it.
/// Supply build_panel_program_rowmajor(rows_max, pitch, 0)'s bytes (input shapes) or
/// build_panel_program_out(rows_max, 2560, 0)'s bytes (out shapes: a different PDI, one hardware context per distinct PDI).
pub struct PanelProgram {
    pub shape: Shape,
    pub pdi: Vec<u8>,
    pub txn: Vec<u8>,
    pub padded_rows: usize,
    pub pitch: usize,
}
/// The program of one shape at one row count: the G80 PDI (its 32 cores' wave loop count is part of it) and the TXN.
pub struct PanelBuild {
    pub pdi: Vec<u8>,
    pub txn: Vec<u8>,
}
/// Builds a shape's program for a padded row count (a whole number of the shape's waves; pure CPU, tens of ms), called on
/// the relay thread the first time a (shape, rows) chain is kicked and cached after.
pub type TxnBuilder = Arc<dyn Fn(Shape, usize) -> Result<PanelBuild, String> + Send + Sync>;
/// Hardware contexts the row-adaptive chains may hold beyond the full-row ones: the (class, rows) variants in use. A kick
/// whose variant would exceed it runs the full-row chain.
pub const MAX_ADAPTIVE_CONTEXTS: usize = 4;
/// Rows of a bank of `rows` live rows rounded up to the whole waves of the shape's program: 256-row waves (input
/// shapes), 512-row waves (out shapes). The A arena is packed for exactly this many rows and the chain runs exactly this
/// many waves.
pub fn padded_rows_of(shape: Shape, rows: usize) -> usize {
    let wave = if shape.is_out() { 512 } else { 256 };
    rows.div_ceil(wave) * wave
}
/// Every entry is already pack_b_bf16 encoded, including Sm/Se sidecars.
pub struct LayerPanels {
    pub layer: usize,
    pub shape: Shape,
    pub panels: Vec<Vec<u8>>,
}

/// Per-bank arenas: input shapes use `a` (`rows * 2880` B) and the C arenas of GdnIn / QsaIn (256-row padded); out shapes use
/// `a_out` (`rows * 6912` B: K = 6144 packed A, 512-row padded) and their own BF16 C arenas (`rows * 2560 * 2` B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout {
    pub padded_rows: usize,
    pub a_bytes: usize,
    pub b_bytes: usize,
    /// `rows_max` rounded up to 512-row N512 waves.
    pub out_padded_rows: usize,
    pub a_out_bytes: usize,
    pub b_out_bytes: usize,
    pitches: [usize; 4],
}
impl ArenaLayout {
    pub fn new(rows_max: usize) -> Result<Self, String> {
        if !(1..=65536).contains(&rows_max) {
            return Err("rows_max outside 1..=65536".into());
        }
        let padded_rows = rows_max.div_ceil(256) * 256;
        let out_padded_rows = rows_max.div_ceil(512) * 512;
        Ok(Self {
            padded_rows,
            a_bytes: padded_rows * 2880,
            b_bytes: 8 * 40 * 2 * 5504,
            out_padded_rows,
            a_out_bytes: out_padded_rows * 6912,
            b_out_bytes: 8 * 96 * 4352,
            pitches: [
                Shape::GdnIn.pitch(),
                Shape::QsaIn.pitch(),
                Shape::GdnOut.pitch(),
                Shape::QsaO.pitch(),
            ],
        })
    }
    /// Padded rows of the shape's program: 256-row waves (input shapes), 512-row waves (out shapes).
    pub fn padded_rows_for(self, shape: Shape) -> usize {
        if shape.is_out() {
            self.out_padded_rows
        } else {
            self.padded_rows
        }
    }
    /// Packed A bytes of one bank for the shape's program.
    pub fn a_bytes_for(self, shape: Shape) -> usize {
        if shape.is_out() {
            self.a_out_bytes
        } else {
            self.a_bytes
        }
    }
    /// Bytes of one resident B panel of the shape.
    pub fn b_bytes_for(self, shape: Shape) -> usize {
        if shape.is_out() {
            self.b_out_bytes
        } else {
            self.b_bytes
        }
    }
    pub fn c_bytes(self, shape: Shape) -> usize {
        self.padded_rows_for(shape) * self.pitches[shape.index()] * 2
    }
    pub fn c_offset(self, shape: Shape, panel: usize) -> Result<usize, String> {
        if panel >= self.pitches[shape.index()] / shape.panel_n() {
            Err("panel index outside C arena".into())
        } else {
            Ok(panel * shape.panel_n() * 2)
        }
    }
    /// Addressed span, not the full arg2 declaration: offset panels write only
    /// their panel columns (1280 / 512) on each row, including the final padded row.
    pub fn c_span(self, shape: Shape) -> usize {
        ((self.padded_rows_for(shape) - 1) * self.pitches[shape.index()] + shape.panel_n()) * 2
    }
    pub fn set_pitch(&mut self, shape: Shape, pitch: usize) {
        self.pitches[shape.index()] = pitch;
    }
    /// Arena bytes of one bank, `[a_in, a_out, c GdnIn, c QsaIn, c GdnOut, c QsaO]`. An absent shape (or A class)
    /// keeps only one registration page for the fixed binding table, never a full arena.
    pub fn bank_arena_bytes(self, has_in: bool, has_out: bool) -> [usize; 6] {
        let c = |shape: Shape| self.c_bytes(shape).max(4096);
        [
            if has_in { self.a_bytes } else { 0 }.max(4096),
            if has_out { self.a_out_bytes } else { 0 }.max(4096),
            c(Shape::GdnIn),
            c(Shape::QsaIn),
            c(Shape::GdnOut),
            c(Shape::QsaO),
        ]
    }
    /// Bytes `banks` banks of arenas pin against `RLIMIT_MEMLOCK`: every arena rounded up to whole pages.
    pub fn pinned_bytes(self, banks: usize, has_in: bool, has_out: bool) -> usize {
        banks
            * self
                .bank_arena_bytes(has_in, has_out)
                .iter()
                .map(|&n| n.div_ceil(4096) * 4096)
                .sum::<usize>()
    }
    /// [`Self::pinned_bytes`] for a trunk of `banks` banks opened at `rows_max` with one `(shape, compacted pitch)`
    /// program per shape: exactly what [`NpuTrunkRuntime::open_with_builder`] settles before it allocates.
    pub fn trunk_pinned_bytes(
        rows_max: usize,
        banks: usize,
        programs: &[(Shape, usize)],
    ) -> Result<usize, String> {
        let mut layout = Self::new(rows_max)?;
        layout.pitches = [0; 4];
        for &(shape, pitch) in programs {
            layout.set_pitch(shape, pitch);
        }
        let (has_in, has_out) = (
            programs.iter().any(|(s, _)| !s.is_out()),
            programs.iter().any(|(s, _)| s.is_out()),
        );
        Ok(layout.pinned_bytes(banks, has_in, has_out))
    }
}

/// A view, not ownership: valid only until runtime teardown. The BO itself is
/// relay-owned; handle/address expose its NPU binding without transferring it.
#[derive(Clone, Copy, Debug)]
pub struct HostRegistered {
    pub npu_handle: u32,
    pub npu_address: u64,
    pub host: *mut u8,
    pub device: u64,
    pub len: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seq {
    pub bank: usize,
    pub value: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct Completion {
    pub kick_to_done: Duration,
    pub npu_busy: Duration,
    pub submit_ns: u64,
    pub done_ns: u64,
    /// CLOCK_MONOTONIC immediately after publishing/flushing the terminal word.
    pub signal_write_ns: u64,
    /// Host time the relay spent resolving this chain's program/commands/chain on this kick: building a row-adaptive
    /// program, its panel commands, or a panel-range sub-chain, or looking one up in the cache (zero for the full-row
    /// full-range chain).
    pub chain_build: Duration,
}
/// Relay statistics of the chains of one shape completed since the previous [`NpuTrunkRuntime::take_chain_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeChainStats {
    pub chains: u64,
    pub npu_busy_ns: u64,
    pub kick_to_done_ns: u64,
}
/// Per-shape relay chain statistics, indexed in [`Shape::ALL`] order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChainStats {
    pub shapes: [ShapeChainStats; 4],
}
impl ChainStats {
    pub fn shape(&self, shape: Shape) -> ShapeChainStats {
        self.shapes[shape.index()]
    }
    fn record(&mut self, shape: Shape, completion: &Completion) {
        let s = &mut self.shapes[shape.index()];
        s.chains += 1;
        s.npu_busy_ns = s
            .npu_busy_ns
            .saturating_add(completion.npu_busy.as_nanos() as u64);
        s.kick_to_done_ns = s
            .kick_to_done_ns
            .saturating_add(completion.kick_to_done.as_nanos() as u64);
    }
}
/// A kick's panel range is non-empty and within the layer's `count` resident panels.
fn validate_panels(range: &std::ops::Range<usize>, count: usize) -> Result<(), String> {
    if range.start >= range.end {
        return Err(format!("empty panel range {}..{}", range.start, range.end));
    }
    if range.end > count {
        return Err(format!(
            "panel range {}..{} exceeds the layer's {count} resident panels",
            range.start, range.end
        ));
    }
    Ok(())
}

/// Linux CLOCK_MONOTONIC timestamp shared by relay and forward trace.
pub fn monotonic_ns() -> u64 {
    let mut time = [0i64; 2];
    // CLOCK_MONOTONIC is always available on the Linux-only NPU backend.
    assert_eq!(unsafe { super::sys::clock_gettime(1, &mut time) }, 0);
    time[0] as u64 * 1_000_000_000 + time[1] as u64
}

struct Pages {
    ptr: *mut u8,
    len: usize,
}
/// Transparent-huge-page size on x86_64.
const HUGE: usize = 2 << 20;
impl Pages {
    /// Arenas of at least one huge page start on a 2 MiB boundary and are `MADV_HUGEPAGE` before first touch, so
    /// under `transparent_hugepage=madvise` their full 2 MiB extents fault as THPs: fewer IOMMU (SVA) and GPUVM
    /// translations for the NPU DMA and the GPU arena reads. The length stays 4 KiB-rounded (the tail is 4 KiB
    /// pages), so pinned_vm / RLIMIT_MEMLOCK accounting is unchanged; the driver pins THP subpages like any page.
    fn new(bytes: usize) -> Result<Self, String> {
        use super::sys::{
            madvise, mmap, munmap, MADV_HUGEPAGE, MAP_ANON, MAP_FAILED, MAP_PRIVATE, PROT_RW,
        };
        let len = bytes.div_ceil(4096) * 4096;
        let span = if len >= HUGE { len + HUGE } else { len };
        let raw = unsafe {
            mmap(
                core::ptr::null_mut(),
                span,
                PROT_RW,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if raw == MAP_FAILED {
            return Err(format!("trunk mmap: errno {}", super::errno()));
        }
        if span == len {
            return Ok(Self {
                ptr: raw.cast(),
                len,
            });
        }
        let base = raw as usize;
        let start = base.next_multiple_of(HUGE);
        let end = start + len;
        unsafe {
            if start > base {
                munmap(raw, start - base);
            }
            if base + span > end {
                munmap(end as *mut _, base + span - end);
            }
            // A hint: without THP support the arena keeps 4 KiB pages, which is today's layout.
            madvise(start as *mut _, len, MADV_HUGEPAGE);
        }
        Ok(Self {
            ptr: start as *mut u8,
            len,
        })
    }
}
impl Drop for Pages {
    fn drop(&mut self) {
        unsafe {
            super::sys::munmap(self.ptr.cast(), self.len);
        }
    }
}
// Field order enforces HIP unregister before pages are unmapped, including open errors.
struct RegisteredPages<R, P> {
    registration: R,
    pages: P,
}
type Arena = RegisteredPages<HostRegistration, Pages>;
impl Arena {
    fn new(hip: &HipRuntime, bytes: usize) -> Result<Self, String> {
        let pages = Pages::new(bytes)?;
        let registration = hip.register_host(pages.ptr, pages.len)?;
        Ok(Self {
            registration,
            pages,
        })
    }
}
/// `a[0]` / `c[0..2]`: input shapes; `a[1]` / `c[2..4]`: out shapes (one page when no out shape is open).
struct Bank {
    a: [Arena; 2],
    c: [Arena; 4],
    control: Arena,
    ready: HipEvent,
    acquire: HipEvent,
    retired: HipEvent,
}
#[derive(Clone, Copy)]
struct Binding {
    handle: u32,
    addr: u64,
}
/// Registered-page bindings of one bank: `[a_in, a_out, c0, c1, c2, c3]`.
type BankBindings = [Binding; 6];
#[derive(Clone, Copy)]
struct BankPages {
    a: [usize; 2],
    a_len: [usize; 2],
    c: [usize; 4],
    c_len: [usize; 4],
}
struct Job {
    seq: Seq,
    layer: usize,
    shape: Shape,
    rows: usize,
    panels: std::ops::Range<usize>,
    event: usize,
    start: Instant,
}
struct Pending {
    value: u32,
    result: Option<Result<Completion, String>>,
}
struct Status {
    pending: Vec<Pending>,
    failure: Option<String>,
    stopping: bool,
}
struct Shared {
    status: Mutex<Status>,
    changed: Condvar,
    signals: Vec<usize>,
    stats: Mutex<ChainStats>,
}
impl Shared {
    fn record_chain(&self, shape: Shape, completion: &Completion) {
        self.stats.lock().record(shape, completion);
    }
    fn take_chain_stats(&self) -> ChainStats {
        std::mem::take(&mut *self.stats.lock())
    }
    fn validate(&self, seq: Seq) -> Result<(), String> {
        let status = self.status.lock();
        let p = status
            .pending
            .get(seq.bank)
            .ok_or("invalid sequence bank")?;
        if seq.value == 0 || p.value != seq.value {
            return Err("unknown/stale sequence".into());
        }
        Ok(())
    }
    fn check_error(&self) -> Result<(), String> {
        match &self.status.lock().failure {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }
    fn wait(&self, seq: Seq, timeout: Duration) -> Result<Completion, String> {
        let deadline = Instant::now() + timeout;
        let mut status = self.status.lock();
        loop {
            let p = status
                .pending
                .get(seq.bank)
                .ok_or("invalid sequence bank")?;
            if seq.value == 0 || p.value != seq.value {
                return Err("unknown/stale sequence".into());
            }
            if let Some(result) = &p.result {
                return result.clone();
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!("trunk host wait timed out: {seq:?}"));
            }
            self.changed.wait_for(&mut status, left);
        }
    }
    // First terminal result wins: shutdown cannot be overwritten by a late
    // successful timeline observation. Record failure BEFORE releasing the GPU.
    fn finish(&self, seq: Seq, mut result: Result<Completion, String>) {
        let mut status = self.status.lock();
        let Some(p) = status.pending.get(seq.bank) else {
            return;
        };
        if p.value != seq.value || p.result.is_some() {
            return;
        }
        if let Err(e) = &result {
            if status.failure.is_none() {
                status.failure = Some(format!("trunk {seq:?}: {e}"));
            }
        }
        let signal_write_ns = self.publish(seq);
        if let Ok(completion) = &mut result {
            completion.signal_write_ns = signal_write_ns;
        }
        status.pending[seq.bank].result = Some(result);
        self.changed.notify_all();
    }
    fn publish(&self, seq: Seq) -> u64 {
        // No generation wrap or skipped reuse: both success and failure are
        // terminal for this generation. The registered coherent line is alive
        // until the relay has joined and all queued GPU consumers have retired.
        let done = self.signals[seq.bank] as *mut AtomicU32;
        unsafe {
            (*done).store(seq.value, Ordering::Release);
            super::clflush(done.cast(), 64);
        }
        monotonic_ns()
    }
    fn stop(&self, reason: &str) {
        let mut status = self.status.lock();
        status.stopping = true;
        for bank in 0..status.pending.len() {
            let p = &mut status.pending[bank];
            if p.value == 0 || p.result.is_some() {
                continue;
            }
            let seq = Seq {
                bank,
                value: p.value,
            };
            p.result = Some(Err(reason.into()));
            if status.failure.is_none() {
                status.failure = Some(format!("trunk {seq:?}: {reason}"));
            }
            self.publish(seq);
        }
        self.changed.notify_all();
    }
}
struct Watchdog {
    deadline: Instant,
}
impl Watchdog {
    fn new(start: Instant, timeout: Duration) -> Self {
        Self {
            deadline: start + timeout,
        }
    }
    fn remaining(&self, now: Instant) -> Result<Duration, String> {
        let remaining = self.deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            Err("NPU chain watchdog timed out".into())
        } else {
            Ok(remaining)
        }
    }
}
struct BankState {
    generation: u32,
    kicked: bool,
    retired: bool,
    prepared: bool,
}

/// Explicit diagnostic access and chain watchdog; production never touches
/// A/C from the CPU after open. CPU oracle users must opt into arena invalidation.
#[derive(Clone, Copy, Debug)]
pub struct OpenOptions {
    pub watchdog: Duration,
    pub cpu_reads: bool,
    /// Developer-only fault injection: the relay forces a 1 ns chain watchdog on the 1-based Nth chain it runs, counting
    /// every job it receives in order. This exercises the real "NPU chain watchdog timed out" path with the chain in flight.
    pub fault_chain: Option<u64>,
}
impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            watchdog: CHAIN_WATCHDOG,
            cpu_reads: false,
            fault_chain: None,
        }
    }
}
impl OpenOptions {
    fn flush_for_cpu_reads(self, flush: impl FnOnce()) {
        if self.cpu_reads {
            flush();
        }
    }
}

/// Async kick and enqueue-ahead GPU rejoin, with terminal failure signalling.
/// No rows/addresses/commands are patched per kick; padded A MUST be zeroed.
pub struct NpuTrunkRuntime {
    hip: HipRuntime,
    api: RelayApi,
    banks: Vec<Bank>,
    bindings: Vec<BankBindings>,
    layout: ArenaLayout,
    rows_max: usize,
    layers: Vec<(usize, Shape)>,
    /// Resident panel count of each `layers` entry, same order.
    panel_counts: Vec<usize>,
    state: RefCell<Vec<BankState>>,
    builder: bool,
    variants: RefCell<Vec<(bool, usize)>>,
    shared: Arc<Shared>,
    tx: Option<mpsc::SyncSender<Job>>,
    relay: Option<thread::JoinHandle<()>>,
}
impl NpuTrunkRuntime {
    /// Uses the existing dlopen HIP backend, without a link-time ROCm dependency
    /// or a second GPU runtime crate. pci identifies the same GPU as producer.
    pub fn open(
        programs: Vec<PanelProgram>,
        layers: Vec<LayerPanels>,
        banks: usize,
        rows_max: usize,
        pci: &str,
    ) -> Result<Self, String> {
        Self::open_with_options(
            programs,
            layers,
            banks,
            rows_max,
            pci,
            OpenOptions::default(),
        )
    }
    /// CPU oracle access requires `cpu_reads`; a 1 ns watchdog forces the real
    /// timeout path in the transport gate without a production environment flag.
    pub fn open_with_options(
        programs: Vec<PanelProgram>,
        layers: Vec<LayerPanels>,
        banks: usize,
        rows_max: usize,
        pci: &str,
        options: OpenOptions,
    ) -> Result<Self, String> {
        Self::open_with_builder(programs, layers, banks, rows_max, pci, options, None)
    }
    /// [`Self::open_with_options`] with a row-adaptive program builder: a kick of fewer rows than the full program runs
    /// the program of its own wave count (see [`TxnBuilder`]); without one every kick runs the full-row chain.
    pub fn open_with_builder(
        programs: Vec<PanelProgram>,
        layers: Vec<LayerPanels>,
        banks: usize,
        rows_max: usize,
        pci: &str,
        options: OpenOptions,
        builder: Option<TxnBuilder>,
    ) -> Result<Self, String> {
        if options.watchdog.is_zero() || options.watchdog > WAIT_TIMEOUT {
            return Err("chain watchdog outside (0, WAIT_TIMEOUT]".into());
        }
        let mut layout = ArenaLayout::new(rows_max)?;
        layout.pitches = [0; 4];
        if banks == 0 {
            return Err("trunk needs at least one bank".into());
        }
        let mut shapes = Vec::new();
        for program in &programs {
            if shapes.contains(&program.shape) {
                return Err("duplicate shape program".into());
            }
            if program.pdi.is_empty() || program.txn.is_empty() {
                return Err("empty panel program".into());
            }
            if program.padded_rows != layout.padded_rows_for(program.shape) {
                return Err("panel program padded_rows mismatch".into());
            }
            if program.pitch == 0
                || program.pitch % program.shape.panel_n() != 0
                || program.pitch > program.shape.pitch()
            {
                return Err("invalid compacted panel pitch".into());
            }
            // One hardware context per distinct PDI (input shapes and out shapes run different G80 programs); shapes of
            // the same program share a context. Nothing requires the PDIs of different shape classes to be equal.
            layout.set_pitch(program.shape, program.pitch);
            shapes.push(program.shape);
        }
        let mut keys = Vec::new();
        for l in &layers {
            if !shapes.contains(&l.shape) {
                return Err("missing shape program".into());
            }
            if keys.contains(&(l.layer, l.shape)) {
                return Err("duplicate layer/shape".into());
            }
            if l.panels.len() != layout.pitches[l.shape.index()] / l.shape.panel_n()
                || l.panels
                    .iter()
                    .any(|b| b.len() != layout.b_bytes_for(l.shape))
            {
                return Err(format!(
                    "layer {} {:?}: invalid BF16 B panel count/size",
                    l.layer, l.shape
                ));
            }
            keys.push((l.layer, l.shape));
        }
        if keys.is_empty() {
            return Err("no resident B panels".into());
        }
        let panel_counts: Vec<usize> = layers.iter().map(|l| l.panels.len()).collect();
        let (has_in, has_out) = (
            shapes.iter().any(|s| !s.is_out()),
            shapes.iter().any(|s| s.is_out()),
        );
        let sizes = layout.bank_arena_bytes(has_in, has_out);
        // Every arena becomes a userptr BO whose pages the driver pins against RLIMIT_MEMLOCK; settle the limit
        // before any HIP load, allocation or registration.
        super::ensure_pin_headroom(layout.pinned_bytes(banks, has_in, has_out))?;
        let hip = HipRuntime::load_for_pci(pci)?;
        if hip.device_arch()? != "gfx1151" {
            return Err("trunk transport currently requires gfx1151".into());
        }
        let api = hip.relay_api()?;
        let mut storage = Vec::with_capacity(banks);
        for _ in 0..banks {
            storage.push(Bank {
                a: [Arena::new(&hip, sizes[0])?, Arena::new(&hip, sizes[1])?],
                c: [
                    Arena::new(&hip, sizes[2])?,
                    Arena::new(&hip, sizes[3])?,
                    Arena::new(&hip, sizes[4])?,
                    Arena::new(&hip, sizes[5])?,
                ],
                control: Arena::new(&hip, 4096)?,
                ready: hip.system_event()?,
                acquire: hip.system_event()?,
                retired: hip.system_event()?,
            });
        }
        let pages = storage
            .iter()
            .map(|b| BankPages {
                a: [b.a[0].pages.ptr as usize, b.a[1].pages.ptr as usize],
                a_len: [b.a[0].pages.len, b.a[1].pages.len],
                c: [
                    b.c[0].pages.ptr as usize,
                    b.c[1].pages.ptr as usize,
                    b.c[2].pages.ptr as usize,
                    b.c[3].pages.ptr as usize,
                ],
                c_len: [
                    b.c[0].pages.len,
                    b.c[1].pages.len,
                    b.c[2].pages.len,
                    b.c[3].pages.len,
                ],
            })
            .collect();
        let shared = Arc::new(Shared {
            status: Mutex::new(Status {
                pending: (0..banks)
                    .map(|_| Pending {
                        value: 0,
                        result: None,
                    })
                    .collect(),
                failure: None,
                stopping: false,
            }),
            changed: Condvar::new(),
            signals: storage
                .iter()
                .map(|b| b.control.pages.ptr as usize)
                .collect(),
            stats: Mutex::new(ChainStats::default()),
        });
        let (tx, rx) = mpsc::sync_channel(banks);
        let (init_tx, init_rx) = mpsc::sync_channel(1);
        let relay_shared = Arc::clone(&shared);
        let has_builder = builder.is_some();
        let relay = thread::Builder::new()
            .name("npu-trunk-relay".into())
            .spawn(move || {
                relay_main(
                    programs,
                    layers,
                    pages,
                    api,
                    layout,
                    builder,
                    rx,
                    init_tx,
                    relay_shared,
                    options,
                );
            })
            .map_err(|e| format!("spawn trunk relay: {e}"))?;
        // Construct the teardown guard before receiving: all early returns join
        // the relay (contexts/BOs) before unregistering/unmapping caller pages.
        let mut rt = Self {
            hip,
            api,
            banks: storage,
            bindings: Vec::new(),
            layout,
            rows_max,
            layers: keys,
            panel_counts,
            state: RefCell::new(
                (0..banks)
                    .map(|_| BankState {
                        generation: 0,
                        kicked: false,
                        retired: false,
                        prepared: false,
                    })
                    .collect(),
            ),
            builder: has_builder,
            variants: RefCell::new(Vec::new()),
            shared,
            tx: Some(tx),
            relay: Some(relay),
        };
        rt.bindings = init_rx
            .recv()
            .map_err(|_| "trunk relay exited during open")??;
        Ok(rt)
    }
    pub fn layout(&self) -> ArenaLayout {
        self.layout
    }
    pub fn hip(&self) -> &HipRuntime {
        &self.hip
    }
    /// `shape = Some(s)`: the C arena of `s`; `None` + `out`: the packed-A arena of the input (`false`) or out (`true`) shapes.
    fn view(&self, bank: usize, shape: Option<Shape>, out: bool) -> Result<HostRegistered, String> {
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let arena = shape.map_or(&b.a[usize::from(out)], |s| &b.c[s.index()]);
        let binding = self.bindings[bank][shape.map_or(usize::from(out), |s| 2 + s.index())];
        let a_len = if out {
            self.layout.a_out_bytes
        } else {
            self.layout.a_bytes
        };
        Ok(HostRegistered {
            npu_handle: binding.handle,
            npu_address: binding.addr,
            host: arena.pages.ptr,
            device: arena.registration.ptr(),
            len: shape.map_or(a_len, |s| self.layout.c_bytes(s)),
        })
    }
    /// Packed-A arena of the input shapes (`GdnIn` / `QsaIn`).
    pub fn a_arena(&self, bank: usize) -> Result<HostRegistered, String> {
        self.view(bank, None, false)
    }
    /// Packed-A arena of the out shapes (`GdnOut` / `QsaO`, K = 6144).
    pub fn a_out_arena(&self, bank: usize) -> Result<HostRegistered, String> {
        self.view(bank, None, true)
    }
    pub fn c_arena(&self, bank: usize, shape: Shape) -> Result<HostRegistered, String> {
        self.view(bank, Some(shape), false)
    }
    pub fn done_word(&self, bank: usize) -> Result<u64, String> {
        Ok(self
            .banks
            .get(bank)
            .ok_or("invalid trunk bank")?
            .control
            .registration
            .ptr())
    }

    /// Nonblocking sticky error check. A terminal failure releases its GPU wait,
    /// but never licenses using that pair's outputs or committing request state.
    pub fn check_error(&self) -> Result<(), String> {
        self.shared.check_error()
    }
    /// Enqueue retirement before writing the next A. Must precede producer pack.
    /// # Safety
    /// `stream` is live, belongs to this GPU, and carries the following producer.
    pub unsafe fn prepare_bank(&self, bank: usize, stream: *mut c_void) -> Result<(), String> {
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let mut states = self.state.borrow_mut();
        let state = &mut states[bank];
        if state.kicked {
            if !state.retired {
                return Err("bank consumer not retired".into());
            }
            self.wait_host(Seq {
                bank,
                value: state.generation,
            })?;
            unsafe {
                self.hip.wait_raw(&b.retired, stream)?;
            }
        }
        state.prepared = true;
        Ok(())
    }
    /// Padded rows the producer packs A for and the chain of a `rows`-row bank of `shape` runs: the bank's whole waves when
    /// the runtime has a program builder and the (class, rows) variant fits the context budget
    /// ([`MAX_ADAPTIVE_CONTEXTS`]), else the full-row program's rows. Deterministic per runtime (the admitted variants are
    /// remembered), so the pack and the kick of one bank always agree.
    pub fn chain_rows(&self, shape: Shape, rows: usize) -> usize {
        let full = self.layout.padded_rows_for(shape);
        let padded = padded_rows_of(shape, rows);
        if !self.builder || padded >= full {
            return full;
        }
        let class = (shape.is_out(), padded);
        let mut variants = self.variants.borrow_mut();
        if variants.contains(&class) {
            return padded;
        }
        if variants.len() >= MAX_ADAPTIVE_CONTEXTS {
            return full;
        }
        variants.push(class);
        padded
    }
    /// Records a system-release event after GPU A writes on stream, then queues
    /// relay work. Caller must not touch A/C until completion/consumer retirement.
    /// `stream` must be live and on the selected GPU; null selects its default stream.
    /// Runs the layer's whole chain (every resident panel); see [`Self::kick_panels`].
    pub unsafe fn kick(
        &self,
        layer: usize,
        shape: Shape,
        bank: usize,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<Seq, String> {
        let panels = self
            .layer_panels(layer, shape)
            .ok_or("unknown layer/shape")?;
        unsafe { self.kick_panels(layer, shape, bank, rows, 0..panels, stream) }
    }
    /// Resident panel count of (`layer`, `shape`); `None` if the pair is unknown.
    pub fn layer_panels(&self, layer: usize, shape: Shape) -> Option<usize> {
        self.layers
            .iter()
            .position(|&k| k == (layer, shape))
            .map(|i| self.panel_counts[i])
    }
    /// Relay statistics of the chains completed since the previous call, then reset. Takes one short lock; no device sync.
    pub fn take_chain_stats(&self) -> ChainStats {
        self.shared.take_chain_stats()
    }
    /// [`Self::kick`] running only the resident panels `panels` (non-empty, `end` <= the layer's panel count) of the
    /// layer's chain, in panel order. Only the C columns of those panels are written; the C arena columns of the other
    /// panels keep their previous bytes. The full range runs the layer's full chain; any other range runs a chain over
    /// the same panel commands (same TXN, B BO and C offsets), built on first use and cached. `Err` before any bank state
    /// changes when the range is invalid. Same safety contract as [`Self::kick`].
    pub unsafe fn kick_panels(
        &self,
        layer: usize,
        shape: Shape,
        bank: usize,
        rows: usize,
        panels: std::ops::Range<usize>,
        stream: *mut c_void,
    ) -> Result<Seq, String> {
        self.check_error()?;
        if rows == 0 || rows > self.rows_max {
            return Err("rows outside admitted extent".into());
        }
        let count = self
            .layer_panels(layer, shape)
            .ok_or("unknown layer/shape")?;
        validate_panels(&panels, count)?;
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let mut states = self.state.borrow_mut();
        let state = &mut states[bank];
        if state.kicked {
            if !state.retired || (!state.prepared && !self.api.query(b.retired.raw())?) {
                return Err("bank consumer not retired".into());
            }
            // prepare_bank already waited for this generation before queuing
            // retirement and the next producer. A kick without prepare still
            // needs the host completion check before reusing the bank.
            if !state.prepared {
                self.wait_host(Seq {
                    bank,
                    value: state.generation,
                })?;
            }
        }
        // signed-difference GEQ implementations must never see generation wrap.
        if state.generation == i32::MAX as u32 {
            return Err("bank sequence exhausted; reopen runtime".into());
        }
        let seq = Seq {
            bank,
            value: state.generation + 1,
        };
        let start = Instant::now();
        {
            let mut status = self.shared.status.lock();
            if status.stopping {
                return Err("trunk runtime stopping".into());
            }
            status.pending[bank] = Pending {
                value: seq.value,
                result: None,
            };
        }
        // Mark ownership before any fallible enqueue. Even on HIP/relay failure,
        // GPU A writes may already be queued by the caller: teardown must not
        // unmap that operand just because no job reached the relay.
        state.generation = seq.value;
        state.kicked = true;
        state.retired = false;
        state.prepared = false;
        let enqueue = (|| {
            unsafe {
                self.hip.record_raw(&b.ready, stream)?;
            }
            self.tx
                .as_ref()
                .ok_or("runtime closed")?
                .try_send(Job {
                    seq,
                    layer,
                    shape,
                    rows: self.chain_rows(shape, rows),
                    panels,
                    event: b.ready.raw(),
                    start,
                })
                .map_err(|e| format!("trunk relay queue: {e}"))
        })();
        if let Err(e) = enqueue {
            self.shared.finish(seq, Err(e.clone()));
            return Err(e);
        }
        Ok(seq)
    }
    pub fn wait_host(&self, seq: Seq) -> Result<Completion, String> {
        self.shared.wait(seq, WAIT_TIMEOUT)
    }
    /// Enqueue the terminal GEQ signal wait immediately, without waiting for
    /// host completion, then a system-fenced event before GPU C loads. Failures
    /// also release this wait; callers MUST check errors before using pair outputs.
    pub unsafe fn wait_gpu(&self, seq: Seq, stream: *mut c_void) -> Result<(), String> {
        self.shared.validate(seq)?;
        let bank = &self.banks[seq.bank];
        unsafe {
            self.api
                .wait_value(stream, self.done_word(seq.bank)?, seq.value)?;
            self.hip.record_raw(&bank.acquire, stream)?;
        }
        Ok(())
    }
    /// Call after enqueuing the LAST C consumer. Required before bank reuse or
    /// teardown. The event bounds retirement without synchronizing unrelated streams.
    pub unsafe fn retire_bank(&self, seq: Seq, stream: *mut c_void) -> Result<(), String> {
        self.shared.validate(seq)?;
        unsafe {
            self.hip.record_raw(&self.banks[seq.bank].retired, stream)?;
        }
        self.state.borrow_mut()[seq.bank].retired = true;
        self.check_error()
    }
    /// Checked teardown. If a GPU consumer does not retire within 3 s, keeps
    /// registrations/pages alive rather than unmapping memory still in use.
    pub fn close(mut self) -> Result<(), String> {
        self.shutdown()
    }
    /// Non-blocking, idempotent: releases every outstanding GPU wait by publishing the terminal signal of every pending
    /// bank, records the sticky failure (the first reason wins) and stops the relay from running further chains, so later
    /// kicks fail. Use on any fault in the pair executor before the GPU continues on the same stream.
    pub fn abort(&self, reason: &str) {
        self.shared.stop(reason);
    }
    fn shutdown(&mut self) -> Result<(), String> {
        // Release every outstanding GPU wait BEFORE joining/destructing the
        // NPU context. A late completion cannot replace the cancellation result.
        self.shared.stop("runtime shutdown before NPU completion");
        self.tx.take();
        let mut failure = None;
        if let Some(relay) = self.relay.take() {
            if let Err(e) = join_within(relay, WAIT_TIMEOUT) {
                failure = Some(e);
            }
        }
        let deadline = Instant::now() + WAIT_TIMEOUT;
        for (b, state) in self.banks.iter().zip(self.state.borrow().iter()) {
            if !state.kicked {
                continue;
            }
            if !state.retired {
                failure = Some("GPU bank not retired before teardown".to_string());
                break;
            }
            loop {
                match self.api.query(b.retired.raw()) {
                    Ok(true) => break,
                    Ok(false) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_micros(50))
                    }
                    Ok(false) => {
                        failure = Some("GPU retirement timed out".into());
                        break;
                    }
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }
        }
        if let Some(e) = failure {
            // Exceptional safe quarantine: cannot destroy events, unregister or
            // unmap a live GPU operand. OS cleanup happens at process exit.
            std::mem::forget(std::mem::take(&mut self.banks));
            return Err(e);
        }
        self.banks.clear();
        self.check_error()?;
        Ok(())
    }
}
impl Drop for NpuTrunkRuntime {
    fn drop(&mut self) {
        // close() already joined and either freed or quarantined every bank.
        if self.banks.is_empty() && self.relay.is_none() {
            return;
        }
        if let Err(e) = self.shutdown() {
            eprintln!("npu-trunk teardown: {e}");
        }
    }
}

// The order is intentional and tested below: contexts (one per distinct PDI), chains, commands, operands,
// instruction/PDI BOs; the row-adaptive extras sit beside their fixed counterparts. The enclosing Device drops only
// after this owner.
struct Resources<C, Ch = Chain, P = Prepared, B = Bo> {
    ctxs: Vec<C>,
    extra_ctxs: Vec<C>,
    chains: Vec<Ch>,
    extra_chains: Vec<Ch>,
    commands: Vec<P>,
    extra_commands: Vec<P>,
    b: Vec<B>,
    a: Vec<B>,
    a_out: Vec<B>,
    c: Vec<B>,
    insts: Vec<B>,
    extra_insts: Vec<B>,
    pdis: Vec<B>,
    extra_pdis: Vec<B>,
}
/// Distinct PDIs in first-use order: `(first program of each, program -> PDI index)`. Programs with byte-identical PDIs
/// (every input shape; both out shapes) share one hardware context, programs with different PDIs get their own.
fn distinct_pdis(programs: &[PanelProgram]) -> (Vec<usize>, Vec<usize>) {
    let mut firsts: Vec<usize> = Vec::new();
    let mut of_program = Vec::with_capacity(programs.len());
    for (i, program) in programs.iter().enumerate() {
        match firsts.iter().position(|&f| programs[f].pdi == program.pdi) {
            Some(index) => of_program.push(index),
            None => {
                of_program.push(firsts.len());
                firsts.push(i);
            }
        }
    }
    (firsts, of_program)
}
fn prepared_panel(
    dev: &Device,
    insts: &Bo,
    a: &Bo,
    b: &Bo,
    c: &Bo,
    c_offset: usize,
    c_bytes: usize,
) -> Result<Prepared, String> {
    if c_offset.checked_add(c_bytes).is_none_or(|end| end > c.len) {
        return Err("C panel outside arena".into());
    }
    let mut command = Prepared::new(dev, insts, &[a, b, c])?;
    // Initialization only, not a hot-path patch: retains the real C handle for
    // pinning while binding a panel's fixed offset in that BO.
    command.addresses[2] = c
        .dev_addr()
        .checked_add(c_offset as u64)
        .ok_or("C address overflow")?;
    let mut words = [0; 17];
    super::encode_start_raw(&mut words, insts.xdna, insts.len, &command.addresses[..3]);
    super::write_words(&command.cmd, &words);
    Ok(command)
}
/// Row-adaptive state of the relay: which (class, padded rows) contexts, (shape, padded rows) TXN BOs,
/// (layer, bank, padded rows) panel-command sets and (layer, bank, padded rows, first, end panel) chains exist, as indices
/// into the `extra_*` vectors of [`Resources`]. `chains` also holds the sub-range chains of the full-row program
/// (`padded` = the full program rows, which never collides with an adaptive row count).
#[derive(Default)]
struct Adaptive {
    ctx: std::collections::HashMap<(bool, usize), usize>,
    pdi: std::collections::HashMap<(bool, usize), Vec<u8>>,
    insts: std::collections::HashMap<(usize, usize), usize>,
    /// (layer, bank, padded) -> index of the first of the layer's panel commands in `extra_commands`.
    commands: std::collections::HashMap<(usize, usize, usize), usize>,
    chains: std::collections::HashMap<(usize, usize, usize, usize, usize), usize>,
}
/// The chain over `panels` and hardware context of (`layer`, `bank`) for a bank of `padded` rows (a whole number of the
/// shape's waves, smaller than the full program), built and cached on first use: the PDI/TXN of that row count from
/// `build`, one hardware context per distinct (class, rows) PDI (the 32 cores' loop count is part of the PDI), the layer's
/// panel commands against the bank's own A/B/C arenas (built once per (layer, bank, rows), whatever the range). The host
/// admits at most [`MAX_ADAPTIVE_CONTEXTS`] variants ([`NpuTrunkRuntime::chain_rows`]), so a larger set is an error.
#[allow(clippy::too_many_arguments)]
fn adaptive_chain<'d>(
    dev: &'d Device,
    build: &TxnBuilder,
    layout: ArenaLayout,
    layers: &[LayerPanels],
    b_starts: &[usize],
    layer: usize,
    bank: usize,
    shape: Shape,
    padded: usize,
    panels: std::ops::Range<usize>,
    r: &mut Resources<HwCtx<'d>>,
    ad: &mut Adaptive,
) -> Result<(usize, usize), String> {
    let class = (shape.is_out(), padded);
    let key = (layer, bank, padded, panels.start, panels.end);
    if let Some(&chain) = ad.chains.get(&key) {
        return Ok((chain, ad.ctx[&class]));
    }
    if !ad.ctx.contains_key(&class) && ad.ctx.len() >= MAX_ADAPTIVE_CONTEXTS {
        return Err(format!(
            "row-adaptive context budget {MAX_ADAPTIVE_CONTEXTS} exceeded at {padded} rows"
        ));
    }
    let command_key = (layer, bank, padded);
    let start = match ad.commands.get(&command_key) {
        Some(&start) => start,
        None => {
            let insts_key = (shape.index(), padded);
            if !ad.insts.contains_key(&insts_key) {
                let PanelBuild { pdi, txn } = build(shape, padded)?;
                if txn.is_empty() || pdi.is_empty() {
                    return Err(format!(
                        "empty row-adaptive program for {shape:?} at {padded} rows"
                    ));
                }
                match ad.pdi.get(&class) {
                    Some(have) if *have != pdi => {
                        return Err(format!(
                            "{shape:?} and its class built different PDIs at {padded} rows"
                        ))
                    }
                    Some(_) => {}
                    None => {
                        let bo = dev.dev_bo(&pdi)?;
                        let ctx = HwCtx::create(dev, 32, 1)?;
                        ctx.config_cu(&bo)?;
                        r.extra_pdis.push(bo);
                        r.extra_ctxs.push(ctx);
                        ad.ctx.insert(class, r.extra_ctxs.len() - 1);
                        ad.pdi.insert(class, pdi);
                    }
                }
                r.extra_insts.push(dev.dev_bo(&txn)?);
                ad.insts.insert(insts_key, r.extra_insts.len() - 1);
            }
            let (insts, l) = (ad.insts[&insts_key], &layers[layer]);
            let start = r.extra_commands.len();
            let a = if shape.is_out() {
                &r.a_out[bank]
            } else {
                &r.a[bank]
            };
            for panel in 0..l.panels.len() {
                let command = prepared_panel(
                    dev,
                    &r.extra_insts[insts],
                    a,
                    &r.b[b_starts[layer] + panel],
                    &r.c[bank * 4 + shape.index()],
                    layout.c_offset(shape, panel)?,
                    layout.c_span(shape),
                )?;
                r.extra_commands.push(command);
            }
            ad.commands.insert(command_key, start);
            start
        }
    };
    let refs: Vec<_> = r.extra_commands[start + panels.start..start + panels.end]
        .iter()
        .collect();
    r.extra_chains.push(Chain::new(dev, &refs)?);
    let chain = r.extra_chains.len() - 1;
    ad.chains.insert(key, chain);
    Ok((chain, ad.ctx[&class]))
}
/// The chain over the sub-range `panels` (not the whole layer) of the full-row program's panel commands of (`layer`,
/// `bank`), built on first use from commands `relay_main` already prepared (`cmd_starts[layer * banks + bank]`); no new
/// command, TXN or context. Index into `extra_chains`; runs on the full-row program's context.
#[allow(clippy::too_many_arguments)]
fn range_chain<'d>(
    dev: &'d Device,
    cmd_starts: &[usize],
    banks: usize,
    layer: usize,
    bank: usize,
    padded: usize,
    panels: std::ops::Range<usize>,
    r: &mut Resources<HwCtx<'d>>,
    ad: &mut Adaptive,
) -> Result<usize, String> {
    let key = (layer, bank, padded, panels.start, panels.end);
    if let Some(&chain) = ad.chains.get(&key) {
        return Ok(chain);
    }
    let start = cmd_starts[layer * banks + bank];
    let refs: Vec<_> = r.commands[start + panels.start..start + panels.end]
        .iter()
        .collect();
    r.extra_chains.push(Chain::new(dev, &refs)?);
    let chain = r.extra_chains.len() - 1;
    ad.chains.insert(key, chain);
    Ok(chain)
}
fn relay_main(
    programs: Vec<PanelProgram>,
    layers: Vec<LayerPanels>,
    pages: Vec<BankPages>,
    api: RelayApi,
    layout: ArenaLayout,
    build: Option<TxnBuilder>,
    rx: mpsc::Receiver<Job>,
    init: mpsc::SyncSender<Result<Vec<BankBindings>, String>>,
    shared: Arc<Shared>,
    options: OpenOptions,
) {
    let dev = match Device::open().and_then(|mut d| {
        d.map_heap(64 << 20)?;
        Ok(d)
    }) {
        Ok(d) => d,
        Err(e) => {
            let _ = init.send(Err(e));
            return;
        }
    };
    let (pdi_firsts, pdi_of_program) = distinct_pdis(&programs);
    let setup = (|| {
        api.select()?;
        let mut r = Resources {
            ctxs: Vec::new(),
            extra_ctxs: Vec::new(),
            chains: Vec::new(),
            extra_chains: Vec::new(),
            commands: Vec::new(),
            extra_commands: Vec::new(),
            b: Vec::new(),
            a: Vec::new(),
            a_out: Vec::new(),
            c: Vec::new(),
            insts: Vec::new(),
            extra_insts: Vec::new(),
            pdis: Vec::new(),
            extra_pdis: Vec::new(),
        };
        // One context per distinct PDI, created and configured in first-use order; the single relay thread submits to
        // them serially, so at most one chain is in flight.
        for &first in &pdi_firsts {
            let pdi = dev.dev_bo(&programs[first].pdi)?;
            let ctx = HwCtx::create(&dev, 32, 1)?;
            ctx.config_cu(&pdi)?;
            r.pdis.push(pdi);
            r.ctxs.push(ctx);
        }
        for program in &programs {
            r.insts.push(dev.dev_bo(&program.txn)?);
        }
        let mut bindings = Vec::with_capacity(pages.len());
        for p in &pages {
            let a = dev.userptr_bo(p.a[0] as *mut u8, p.a_len[0])?;
            let a_out = dev.userptr_bo(p.a[1] as *mut u8, p.a_len[1])?;
            let mut binding = [
                Binding {
                    handle: a.handle,
                    addr: a.dev_addr(),
                },
                Binding {
                    handle: a_out.handle,
                    addr: a_out.dev_addr(),
                },
                Binding { handle: 0, addr: 0 },
                Binding { handle: 0, addr: 0 },
                Binding { handle: 0, addr: 0 },
                Binding { handle: 0, addr: 0 },
            ];
            for s in 0..4 {
                let c = dev.userptr_bo(p.c[s] as *mut u8, p.c_len[s])?;
                binding[2 + s] = Binding {
                    handle: c.handle,
                    addr: c.dev_addr(),
                };
                r.c.push(c);
            }
            bindings.push(binding);
            r.a.push(a);
            r.a_out.push(a_out);
        }
        let mut b_starts = Vec::with_capacity(layers.len());
        let mut cmd_starts = Vec::with_capacity(layers.len() * pages.len());
        for l in &layers {
            let b_start = r.b.len();
            b_starts.push(b_start);
            let program = programs
                .iter()
                .position(|p| p.shape == l.shape)
                .ok_or("missing relay shape program")?;
            for bytes in &l.panels {
                let mut b = dev.shmem_bo(bytes.len())?;
                b.as_mut_slice().copy_from_slice(bytes);
                b.flush();
                r.b.push(b);
            }
            for bank in 0..pages.len() {
                let start = r.commands.len();
                cmd_starts.push(start);
                let a = if l.shape.is_out() {
                    &r.a_out[bank]
                } else {
                    &r.a[bank]
                };
                for panel in 0..l.panels.len() {
                    r.commands.push(prepared_panel(
                        &dev,
                        &r.insts[program],
                        a,
                        &r.b[b_start + panel],
                        &r.c[bank * 4 + l.shape.index()],
                        layout.c_offset(l.shape, panel)?,
                        layout.c_span(l.shape),
                    )?);
                }
                let refs: Vec<_> = r.commands[start..].iter().collect();
                r.chains.push(Chain::new(&dev, &refs)?);
            }
        }
        // Anonymous pages start zeroed; registrations/userptr setup is complete.
        // Evict any initialization lines once, before handing arenas to devices.
        // Production CPU code never accesses A/C again.
        for arena in r.a.iter().chain(&r.a_out).chain(&r.c) {
            arena.flush();
        }
        Ok::<_, String>((r, bindings, b_starts, cmd_starts))
    })();
    let (mut resources, bindings, b_starts, cmd_starts) = match setup {
        Ok(r) => r,
        Err(e) => {
            let _ = init.send(Err(e));
            return;
        }
    };
    let mut ad = Adaptive::default();
    if init.send(Ok(bindings)).is_err() {
        return;
    }
    // Declared after resources, so unwinding releases waits before context/BO
    // destruction (which may itself wait for a device job to be cancelled).
    struct RelayExit<'a>(&'a Shared);
    impl Drop for RelayExit<'_> {
        fn drop(&mut self) {
            self.0.stop("trunk relay exited before completion");
        }
    }
    let _exit = RelayExit(&shared);
    let mut fatal: Option<String> = None;
    // 1-based relay chain number (`OpenOptions::fault_chain`).
    for (chain_no, job) in (1u64..).zip(rx) {
        let result = (|| {
            if let Some(e) = &fatal {
                return Err(format!("trunk relay poisoned: {e}"));
            }
            if shared.status.lock().stopping {
                return Err("runtime shutdown before NPU completion".into());
            }
            let deadline = job.start + WAIT_TIMEOUT;
            loop {
                if Instant::now() >= deadline {
                    return Err("GPU A-ready event timed out".into());
                }
                if shared.status.lock().stopping {
                    return Err("runtime shutdown before GPU A-ready".into());
                }
                if api.query(job.event)? {
                    break;
                }
                thread::sleep(Duration::from_micros(20));
            }
            // GPU system release makes A writes visible to NPU DMA in DRAM.
            // Production has no CPU A lines after the open-time flush; only the
            // diagnostic oracle can have cached aliases. Never SYNC_BO.
            let layer = layers
                .iter()
                .position(|l| l.layer == job.layer && l.shape == job.shape)
                .ok_or("unknown relay layer")?;
            options.flush_for_cpu_reads(|| {
                if job.shape.is_out() {
                    resources.a_out[job.seq.bank].flush()
                } else {
                    resources.a[job.seq.bank].flush()
                }
            });
            // The shape's PDI selects its hardware context (the programs list holds one entry per shape).
            let program = programs
                .iter()
                .position(|p| p.shape == job.shape)
                .ok_or("missing relay shape program")?;
            // Row-adaptive chain: the program of the bank's own wave count (its PDI loops over exactly that many waves),
            // built on first use. `job.rows` is the padded row count the host chose ([`NpuTrunkRuntime::chain_rows`]); the
            // full-row count (or a runtime without a builder) runs the full-row chain, exactly as before.
            let padded = job.rows;
            let count = layers[layer].panels.len();
            let (lo, hi) = (job.panels.start, job.panels.end);
            if lo >= hi || hi > count {
                return Err(format!("invalid panel range {lo}..{hi} of {count} panels"));
            }
            let build_started = Instant::now();
            let adaptive = match &build {
                Some(build) if padded != programs[program].padded_rows => Some(adaptive_chain(
                    &dev,
                    build,
                    layout,
                    &layers,
                    &b_starts,
                    layer,
                    job.seq.bank,
                    job.shape,
                    padded,
                    lo..hi,
                    &mut resources,
                    &mut ad,
                )?),
                _ => None,
            };
            // Full-row program on a partial panel range: a chain over that sub-slice of the (layer, bank)'s own commands.
            let sub = match adaptive {
                None if lo != 0 || hi != count => Some(range_chain(
                    &dev,
                    &cmd_starts,
                    pages.len(),
                    layer,
                    job.seq.bank,
                    programs[program].padded_rows,
                    lo..hi,
                    &mut resources,
                    &mut ad,
                )?),
                _ => None,
            };
            let chain_build = if adaptive.is_some() || sub.is_some() {
                build_started.elapsed()
            } else {
                Duration::ZERO
            };
            let (chain, ctx) = match (adaptive, sub) {
                (Some((chain, ctx)), _) => {
                    (&resources.extra_chains[chain], &resources.extra_ctxs[ctx])
                }
                (None, Some(chain)) => (
                    &resources.extra_chains[chain],
                    &resources.ctxs[pdi_of_program[program]],
                ),
                (None, None) => (
                    &resources.chains[layer * pages.len() + job.seq.bank],
                    &resources.ctxs[pdi_of_program[program]],
                ),
            };
            let start = Instant::now();
            let watchdog = chain_watchdog(&options, chain_no);
            let chain_watchdog = Watchdog::new(start, watchdog);
            let submit_ns = monotonic_ns();
            let point = ctx.submit_chain(chain)?;
            let left = chain_watchdog.remaining(Instant::now())?;
            let state = ctx
                .wait_chain(chain, point, left.as_millis().max(1) as u64)
                .map_err(|e| format!("NPU chain completion failed (watchdog {watchdog:?}): {e}"))?;
            let done_ns = monotonic_ns();
            chain_watchdog.remaining(Instant::now())?;
            let npu_busy = start.elapsed();
            if state != ERT_STATE_COMPLETED {
                return Err(format!(
                    "NPU chain state {state}, error index {}",
                    chain.error_index()
                ));
            }
            // The transaction tail synchronizes all NPU C DMA writes to DRAM.
            // GPU consumers wait on the terminal word then system-acquire; CPU
            // caches hold no C lines in production after the open-time flush.
            options
                .flush_for_cpu_reads(|| resources.c[job.seq.bank * 4 + job.shape.index()].flush());
            Ok(Completion {
                kick_to_done: job.start.elapsed(),
                npu_busy,
                submit_ns,
                done_ns,
                signal_write_ns: 0,
                chain_build,
            })
        })();
        if let Err(e) = &result {
            fatal = Some(e.clone());
        }
        if let Ok(completion) = &result {
            shared.record_chain(job.shape, completion);
        }
        shared.finish(job.seq, result);
    }
    drop(resources); // destroys context before any operand BO, even on timeout
}
/// Bounded join: a relay stuck in an NPU driver ioctl is detached (never blocks teardown); the caller must quarantine
/// everything the relay may still reference.
fn join_within(handle: thread::JoinHandle<()>, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            std::mem::forget(handle);
            return Err(format!("trunk relay did not exit within {timeout:?} (stuck in the NPU driver?); runtime quarantined"));
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle
        .join()
        .map_err(|_| "trunk relay panicked".to_string())
}
/// Chain watchdog of the 1-based `chain`th job the relay received: 1 ns on the `fault_chain` one, else `options.watchdog`.
fn chain_watchdog(options: &OpenOptions, chain: u64) -> Duration {
    if options.fault_chain == Some(chain) {
        Duration::from_nanos(1)
    } else {
        options.watchdog
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arena_flush_policy_keeps_production_device_only() {
        for cpu_reads in [false, true] {
            let options = OpenOptions {
                cpu_reads,
                ..OpenOptions::default()
            };
            let mut flushes = 0;
            for _ in 0..20 {
                options.flush_for_cpu_reads(|| flushes += 1); // A before submit
                options.flush_for_cpu_reads(|| flushes += 1); // C after completion
            }
            assert_eq!(flushes, if cpu_reads { 40 } else { 0 });
        }
        assert!(!OpenOptions::default().cpu_reads);
    }
    #[test]
    fn bank_rows_map_to_whole_waves() {
        // In-projection waves are 256 rows, out-projection waves 512; a bank's chain runs exactly these.
        for (shape, rows, padded) in [
            (Shape::GdnIn, 1, 256),
            (Shape::GdnIn, 256, 256),
            (Shape::QsaIn, 1127, 1280),
            (Shape::GdnIn, 4647, 4864),
            (Shape::GdnIn, 4672, 4864),
            (Shape::QsaIn, 4180, 4352),
            (Shape::GdnIn, 8192, 8192),
            (Shape::QsaO, 1127, 1536),
            (Shape::QsaO, 4672, 5120),
            (Shape::GdnOut, 8192, 8192),
        ] {
            assert_eq!(padded_rows_of(shape, rows), padded, "{shape:?} {rows}");
            assert!(padded >= rows && padded - rows < shape_wave(shape));
        }
        // The packed A of a bank is a prefix of the arena: 2880 B per padded row (K2560), 6912 B (K6144).
        let layout = ArenaLayout::new(8192).unwrap();
        for rows in [1127usize, 4647, 4672, 8192] {
            assert!(padded_rows_of(Shape::GdnIn, rows) * 2880 <= layout.a_bytes);
            assert!(padded_rows_of(Shape::QsaO, rows) * 6912 <= layout.a_out_bytes);
        }
        assert!(
            MAX_ADAPTIVE_CONTEXTS >= 2,
            "a rebalanced request needs one in-projection and one out variant"
        );
    }
    fn shape_wave(shape: Shape) -> usize {
        if shape.is_out() {
            512
        } else {
            256
        }
    }
    #[test]
    fn compacted_qkv_arena_has_eight_panel_windows() {
        let mut l = ArenaLayout::new(8192).unwrap();
        l.pitches = [10240, 0, 0, 0];
        assert_eq!(l.c_bytes(Shape::GdnIn), 8192 * 10240 * 2);
        assert_eq!(l.c_bytes(Shape::QsaIn), 0);
        for panel in 0..8 {
            assert_eq!(l.c_offset(Shape::GdnIn, panel).unwrap(), panel * 1280 * 2);
            assert!(
                l.c_offset(Shape::GdnIn, panel).unwrap() + l.c_span(Shape::GdnIn)
                    <= l.c_bytes(Shape::GdnIn)
            );
        }
        assert!(l.c_offset(Shape::GdnIn, 8).is_err());
        assert!(l.c_offset(Shape::QsaIn, 0).is_err());
    }
    #[test]
    fn production_arena_sizes_and_offsets() {
        let l = ArenaLayout::new(8192).unwrap();
        assert_eq!(l.a_bytes, 23_592_960);
        assert_eq!(l.b_bytes, 3_522_560);
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            assert_eq!(l.c_bytes(shape), 8192 * shape.pitch() * 2);
            for p in 0..shape.panels() {
                let offset = l.c_offset(shape, p).unwrap();
                assert_eq!(offset, 2 * p * PANEL_N);
                assert!(offset + l.c_span(shape) <= l.c_bytes(shape));
            }
            assert!(l.c_offset(shape, shape.panels()).is_err());
            assert_eq!(
                l.c_offset(shape, shape.panels() - 1).unwrap() + l.c_span(shape),
                l.c_bytes(shape)
            );
        }
        assert_eq!(l.c_bytes(Shape::GdnIn), 272_629_760);
        assert_eq!(l.c_bytes(Shape::QsaIn), 230_686_720);
        // Two banks x (A 8192x2880, GdnIn + QsaIn C, three 4 KiB placeholders): the shipped trunk's pin.
        let shipped = [
            (Shape::GdnIn, Shape::GdnIn.pitch()),
            (Shape::QsaIn, Shape::QsaIn.pitch()),
        ];
        assert_eq!(
            ArenaLayout::trunk_pinned_bytes(8192, 2, &shipped).unwrap(),
            1_053_843_456
        );
        assert!(ArenaLayout::trunk_pinned_bytes(0, 2, &shipped).is_err());
        assert_eq!(ArenaLayout::new(257).unwrap().padded_rows, 512);
        assert!(ArenaLayout::new(0).is_err());
        assert!(ArenaLayout::new(65537).is_err());
    }
    #[test]
    fn shape_chains_preserve_panel_order() {
        for shape in Shape::ALL {
            let handles: Vec<u32> = (100..100 + shape.panels() as u32).collect();
            let words = super::super::encode_chain(&handles).unwrap();
            assert_eq!(words[1] as usize, shape.panels());
            for (p, h) in handles.iter().enumerate() {
                assert_eq!(words[7 + 2 * p], *h);
            }
            assert_eq!((words[0] >> 23) & 31, 19);
        }
    }
    #[test]
    fn out_shapes_are_five_512_column_panels_with_their_own_arenas() {
        for shape in [Shape::GdnOut, Shape::QsaO] {
            assert!(shape.is_out());
            assert_eq!(
                (shape.panels(), shape.panel_n(), shape.pitch()),
                (5, OUT_PANEL_N, 2560)
            );
        }
        assert!(!Shape::GdnIn.is_out() && !Shape::QsaIn.is_out());
        let l = ArenaLayout::new(8192).unwrap();
        // K = 6144 packed A: 8 segments of waves * 96 * 4608 B = rows * 6912; B panel 8 * 96 * 4352; BF16 C 8192 x 2560.
        assert_eq!(
            (l.out_padded_rows, l.a_out_bytes, l.b_out_bytes),
            (8192, 56_623_104, 3_342_336)
        );
        assert_eq!(l.a_out_bytes, 8 * (8192 / 512) * 96 * 4608);
        for shape in [Shape::GdnOut, Shape::QsaO] {
            assert_eq!(l.c_bytes(shape), 41_943_040);
            assert_eq!(
                (
                    l.a_bytes_for(shape),
                    l.b_bytes_for(shape),
                    l.padded_rows_for(shape)
                ),
                (56_623_104, 3_342_336, 8192)
            );
            for p in 0..5 {
                assert_eq!(l.c_offset(shape, p).unwrap(), 2 * p * OUT_PANEL_N);
                assert!(l.c_offset(shape, p).unwrap() + l.c_span(shape) <= l.c_bytes(shape));
            }
            assert!(l.c_offset(shape, 5).is_err());
            assert_eq!(
                l.c_offset(shape, 4).unwrap() + l.c_span(shape),
                l.c_bytes(shape)
            );
        }
        // Input shapes keep their sizes; 512-row padding is independent of the 256-row padding.
        assert_eq!(
            (
                l.a_bytes_for(Shape::GdnIn),
                l.b_bytes_for(Shape::QsaIn),
                l.padded_rows_for(Shape::QsaIn)
            ),
            (23_592_960, 3_522_560, 8192)
        );
        let odd = ArenaLayout::new(257).unwrap();
        assert_eq!((odd.padded_rows, odd.out_padded_rows), (512, 512));
        let odd = ArenaLayout::new(700).unwrap();
        assert_eq!(
            (odd.padded_rows, odd.out_padded_rows, odd.a_out_bytes),
            (768, 1024, 1024 * 6912)
        );
    }
    #[test]
    fn one_context_per_distinct_pdi_in_first_use_order() {
        let program = |shape, pdi: &[u8]| PanelProgram {
            shape,
            pdi: pdi.to_vec(),
            txn: vec![1],
            padded_rows: 256,
            pitch: shape.pitch(),
        };
        let (firsts, of) =
            distinct_pdis(&[program(Shape::GdnIn, b"in"), program(Shape::QsaIn, b"in")]);
        assert_eq!((firsts, of), (vec![0], vec![0, 0]));
        let (firsts, of) = distinct_pdis(&[
            program(Shape::GdnIn, b"in"),
            program(Shape::GdnOut, b"out"),
            program(Shape::QsaIn, b"in"),
            program(Shape::QsaO, b"out"),
        ]);
        assert_eq!((firsts, of), (vec![0, 1], vec![0, 1, 0, 1]));
        let (firsts, of) =
            distinct_pdis(&[program(Shape::QsaO, b"out"), program(Shape::GdnOut, b"out")]);
        assert_eq!((firsts, of), (vec![0], vec![0, 0]));
    }
    #[repr(align(64))]
    struct Signal(AtomicU32);
    fn shared(signals: &[Signal], value: u32) -> Shared {
        Shared {
            status: Mutex::new(Status {
                pending: signals
                    .iter()
                    .map(|_| Pending {
                        value,
                        result: None,
                    })
                    .collect(),
                failure: None,
                stopping: false,
            }),
            changed: Condvar::new(),
            signals: signals
                .iter()
                .map(|s| &s.0 as *const AtomicU32 as usize)
                .collect(),
            stats: Mutex::new(ChainStats::default()),
        }
    }
    fn completion() -> Completion {
        Completion {
            kick_to_done: Duration::ZERO,
            npu_busy: Duration::ZERO,
            submit_ns: 1,
            done_ns: 2,
            signal_write_ns: 0,
            chain_build: Duration::ZERO,
        }
    }
    #[test]
    fn invalid_timeout_and_failed_sequences_are_bounded() {
        let signals = [Signal(AtomicU32::new(2))];
        let pending = shared(&signals, 3);
        let seq = Seq { bank: 0, value: 3 };
        assert!(pending
            .wait(
                Seq {
                    bank: 0,
                    value: 999
                },
                Duration::ZERO
            )
            .unwrap_err()
            .contains("unknown"));
        assert!(pending
            .wait(Seq { bank: 1, value: 3 }, Duration::ZERO)
            .is_err());
        assert!(pending
            .wait(seq, Duration::from_millis(1))
            .unwrap_err()
            .contains("timed out"));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 2);
        pending.finish(seq, Err("NPU timeout".into()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 3);
        assert_eq!(
            pending.wait(seq, Duration::ZERO).unwrap_err(),
            "NPU timeout"
        );
        assert!(pending.check_error().unwrap_err().contains("NPU timeout"));
        pending.finish(seq, Ok(completion()));
        assert_eq!(
            pending.wait(seq, Duration::ZERO).unwrap_err(),
            "NPU timeout"
        );
    }
    #[test]
    fn terminal_signals_are_monotonic_and_stale_completions_are_ignored() {
        let signals = [Signal(AtomicU32::new(0))];
        let pending = shared(&signals, 1);
        let seq = Seq { bank: 0, value: 1 };
        pending.validate(seq).unwrap(); // enqueue-ahead validation needs no completion
        assert_eq!(signals[0].0.load(Ordering::Acquire), 0);
        pending.finish(seq, Ok(completion()));
        assert_eq!(pending.wait(seq, Duration::ZERO).unwrap().done_ns, 2);
        assert!(pending.wait(seq, Duration::ZERO).unwrap().signal_write_ns >= 2);
        pending.status.lock().pending[0] = Pending {
            value: 2,
            result: None,
        };
        pending.finish(seq, Err("stale".into()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 1);
        pending.check_error().unwrap();
        pending.finish(Seq { bank: 0, value: 2 }, Ok(completion()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 2);
        pending.stop("shutdown");
        assert_eq!(signals[0].0.load(Ordering::Acquire), 2);
        pending.check_error().unwrap();
    }
    #[test]
    fn shutdown_releases_every_bank_and_late_success_cannot_clear_failure() {
        let signals = [Signal(AtomicU32::new(6)), Signal(AtomicU32::new(6))];
        let pending = shared(&signals, 7);
        pending.stop("runtime dropped");
        for (bank, signal) in signals.iter().enumerate() {
            let seq = Seq { bank, value: 7 };
            assert_eq!(signal.0.load(Ordering::Acquire), 7);
            pending.finish(seq, Ok(completion()));
            assert_eq!(
                pending.wait(seq, Duration::ZERO).unwrap_err(),
                "runtime dropped"
            );
        }
        assert!(pending.status.lock().stopping);
        assert!(pending
            .check_error()
            .unwrap_err()
            .contains("runtime dropped"));
    }
    #[test]
    fn watchdog_deadline_is_chain_only_and_timeout_releases_wait() {
        let start = Instant::now();
        let watchdog = Watchdog::new(start, CHAIN_WATCHDOG);
        assert_eq!(CHAIN_WATCHDOG, EXPECTED_CHAIN_TIME * 5);
        assert_eq!(watchdog.remaining(start).unwrap(), CHAIN_WATCHDOG);
        assert_eq!(
            watchdog
                .remaining(start + CHAIN_WATCHDOG - Duration::from_nanos(1))
                .unwrap(),
            Duration::from_nanos(1)
        );
        let signals = [Signal(AtomicU32::new(0))];
        let pending = shared(&signals, 1);
        let error = watchdog.remaining(start + CHAIN_WATCHDOG).unwrap_err();
        pending.finish(Seq { bank: 0, value: 1 }, Err(error));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 1);
        assert!(pending
            .check_error()
            .unwrap_err()
            .contains("watchdog timed out"));
        assert!(watchdog.remaining(start + WAIT_TIMEOUT).is_err());
    }
    #[test]
    fn owning_fields_teardown_in_declared_order() {
        use std::rc::Rc;
        struct Trace(&'static str, Rc<RefCell<Vec<&'static str>>>);
        impl Drop for Trace {
            fn drop(&mut self) {
                self.1.borrow_mut().push(self.0);
            }
        }
        let log = Rc::new(RefCell::new(Vec::new()));
        let t = |name| Trace(name, Rc::clone(&log));
        // Instantiate the actual production ownership aggregates with tracing
        // resources, exercising their field order without opening either device.
        let npu = Resources {
            ctxs: vec![t("context"), t("context2")],
            extra_ctxs: vec![t("adaptive-context")],
            chains: vec![t("chain")],
            extra_chains: vec![t("adaptive-chain")],
            commands: vec![t("command")],
            extra_commands: vec![t("adaptive-command")],
            b: vec![t("bo")],
            a: vec![],
            a_out: vec![t("a_out")],
            c: vec![],
            insts: vec![t("insts")],
            extra_insts: vec![t("adaptive-insts")],
            pdis: vec![t("pdi"), t("pdi2")],
            extra_pdis: vec![t("adaptive-pdi")],
        };
        let arena = RegisteredPages {
            registration: t("unregister"),
            pages: t("unmap"),
        };
        drop((npu, arena));
        assert_eq!(
            &*log.borrow(),
            &[
                "context",
                "context2",
                "adaptive-context",
                "chain",
                "adaptive-chain",
                "command",
                "adaptive-command",
                "bo",
                "a_out",
                "insts",
                "adaptive-insts",
                "pdi",
                "pdi2",
                "adaptive-pdi",
                "unregister",
                "unmap"
            ]
        );
    }
    #[test]
    fn panel_commands_bind_fixed_arena_offsets() {
        let layout = ArenaLayout::new(8192).unwrap();
        for panel in 0..13 {
            let c = 0x1234_0000_0000u64 + layout.c_offset(Shape::GdnIn, panel).unwrap() as u64;
            let mut words = [0; 17];
            super::super::encode_start_raw(
                &mut words,
                0x1000,
                4096,
                &[0x2000, 0x3000 + panel as u64 * layout.b_bytes as u64, c],
            );
            assert_eq!(u64::from(words[11]) | (u64::from(words[12]) << 32), c);
            assert_eq!(words[7], 0x2000);
            assert_eq!(words[6], 4096);
        }
    }
    #[test]
    fn panel_ranges_are_validated_against_the_resident_count() {
        for (range, count) in [(0..13, 13), (8..13, 13), (0..8, 13), (4..5, 5), (0..1, 1)] {
            validate_panels(&range, count).unwrap();
        }
        assert!(validate_panels(&(3..3), 13).unwrap_err().contains("empty"));
        assert!(validate_panels(&std::ops::Range { start: 5, end: 4 }, 13)
            .unwrap_err()
            .contains("empty"));
        assert!(validate_panels(&(0..0), 0).unwrap_err().contains("empty"));
        assert!(validate_panels(&(0..14), 13)
            .unwrap_err()
            .contains("exceeds"));
        assert!(validate_panels(&(12..14), 13)
            .unwrap_err()
            .contains("exceeds"));
        assert!(validate_panels(&(0..1), 0).is_err());
    }
    fn timed(busy_us: u64, kick_to_done_us: u64) -> Completion {
        Completion {
            kick_to_done: Duration::from_micros(kick_to_done_us),
            npu_busy: Duration::from_micros(busy_us),
            ..completion()
        }
    }
    #[test]
    fn chain_stats_accumulate_per_shape_then_reset_on_take() {
        let signals = [Signal(AtomicU32::new(0))];
        let s = shared(&signals, 1);
        assert_eq!(s.take_chain_stats(), ChainStats::default());
        s.record_chain(Shape::GdnIn, &timed(10, 12));
        s.record_chain(Shape::QsaO, &timed(7, 9));
        s.record_chain(Shape::GdnIn, &timed(20, 25));
        let taken = s.take_chain_stats();
        assert_eq!(
            taken.shape(Shape::GdnIn),
            ShapeChainStats {
                chains: 2,
                npu_busy_ns: 30_000,
                kick_to_done_ns: 37_000
            }
        );
        assert_eq!(
            taken.shape(Shape::QsaO),
            ShapeChainStats {
                chains: 1,
                npu_busy_ns: 7_000,
                kick_to_done_ns: 9_000
            }
        );
        assert_eq!(taken.shape(Shape::QsaIn), ShapeChainStats::default());
        assert_eq!(taken.shape(Shape::GdnOut), ShapeChainStats::default());
        assert_eq!(s.take_chain_stats(), ChainStats::default());
        s.record_chain(Shape::GdnOut, &timed(3, 4));
        assert_eq!(s.take_chain_stats().shape(Shape::GdnOut).chains, 1);
    }
    #[test]
    fn chain_stats_index_in_shape_all_order() {
        let mut stats = ChainStats::default();
        for (i, shape) in Shape::ALL.into_iter().enumerate() {
            stats.record(shape, &timed(i as u64 + 1, 0));
        }
        for (i, shape) in Shape::ALL.into_iter().enumerate() {
            assert_eq!(stats.shapes[i], stats.shape(shape));
            assert_eq!(stats.shape(shape).npu_busy_ns, (i as u64 + 1) * 1000);
        }
    }
    // `NpuTrunkRuntime` needs HIP, so abort() is exercised at the `Shared` level it delegates to (`abort` = `Shared::stop`).
    #[test]
    fn abort_releases_every_pending_wait_once() {
        let signals = [
            Signal(AtomicU32::new(0)),
            Signal(AtomicU32::new(0)),
            Signal(AtomicU32::new(0)),
        ];
        let s = shared(&signals, 7);
        let seq = |bank| Seq { bank, value: 7 };
        s.finish(seq(0), Ok(completion()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 7);
        signals[0].0.store(99, Ordering::Release); // a republish would overwrite this
        s.stop("first reason");
        assert_eq!(signals[0].0.load(Ordering::Acquire), 99);
        assert_eq!(signals[1].0.load(Ordering::Acquire), 7);
        assert_eq!(signals[2].0.load(Ordering::Acquire), 7);
        assert!(s.status.lock().stopping);
        assert!(s.wait(seq(0), Duration::from_millis(10)).is_ok());
        assert!(s
            .wait(seq(1), Duration::from_millis(10))
            .unwrap_err()
            .contains("first reason"));
        assert!(s
            .wait(seq(2), Duration::from_millis(10))
            .unwrap_err()
            .contains("first reason"));
        assert!(s.check_error().unwrap_err().contains("first reason"));
        s.stop("second reason");
        s.finish(seq(1), Ok(completion()));
        s.finish(seq(2), Err("late failure".into()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 99);
        let err = s.check_error().unwrap_err();
        assert!(
            err.contains("first reason")
                && !err.contains("second reason")
                && !err.contains("late failure")
        );
        assert!(s.wait(seq(0), Duration::from_millis(10)).is_ok());
        for bank in [1, 2] {
            assert!(s
                .wait(seq(bank), Duration::from_millis(10))
                .unwrap_err()
                .contains("first reason"));
        }
    }
    #[test]
    fn fault_chain_forces_only_the_named_chain() {
        assert_eq!(OpenOptions::default().fault_chain, None);
        let plain = OpenOptions::default();
        let faulted = OpenOptions {
            fault_chain: Some(3),
            ..OpenOptions::default()
        };
        for chain in 1..=5u64 {
            assert_eq!(chain_watchdog(&plain, chain), plain.watchdog);
            let expect = if chain == 3 {
                Duration::from_nanos(1)
            } else {
                faulted.watchdog
            };
            assert_eq!(chain_watchdog(&faulted, chain), expect);
        }
    }
    #[test]
    fn join_within_is_bounded() {
        let (tx, rx) = mpsc::channel::<()>();
        let blocked = thread::spawn(move || {
            let _ = rx.recv();
        });
        let start = Instant::now();
        let err = join_within(blocked, Duration::from_millis(50)).unwrap_err();
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "join was not bounded"
        );
        assert!(err.contains("did not exit"), "{err}");
        drop(tx); // lets the detached thread exit
        assert!(join_within(thread::spawn(|| {}), Duration::from_secs(2)).is_ok());
        let err = join_within(
            thread::spawn(|| {
                panic!("relay test panic");
            }),
            Duration::from_secs(2),
        )
        .unwrap_err();
        assert!(err.contains("panicked"), "{err}");
    }
}
