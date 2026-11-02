// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! D2 mapped-return transport pilot (PLAN.md §2 B1 kill microexperiment).
//!
//! Prices the transport that returns CPU-computed cold-expert outputs to the
//! GPU combine, plus the readiness cells (stream sync vs event polling), on a
//! real GPU. Compiled everywhere; it only *runs* with a HIP device.
//!
//! Combine arms (`--arms copy,mapped`), per generation `g`, per row count `R`:
//!   * copy:   CPU writes `[R,10,2560]` F32 into a `PinnedTransferBuffer`,
//!             `upload_async` into a device tensor, then the production
//!             `Gpu::moe_down_combine_top10_batched`.
//!   * mapped: CPU writes the same values into a `MappedHostBuffer`, `publish`,
//!             the same combine kernel reads the device alias in place,
//!             `mark_gpu_reading`, `retire`.
//! The CPU first busy-computes `--cpu-work-us` microseconds (the GPU sits idle,
//! as during the D2 CPU arm) and writes RAW F32 values depending on
//! `(g, slot, element)`, so stale data is detectable. Every generation the
//! residual is read back and compared BYTE-for-byte with
//! `hipfire_dispatch::cpu_exec::moe_canonical_combine_reference`, copy is
//! compared bitwise with mapped, and the 4096 sentinel guard bytes behind the
//! mapped payload must be unchanged.
//!
//! Readiness arms (`ready-sync`, `ready-event`): one small GPU producer
//! (`add_inplace_f32` of a generation-dependent `[R,2560]` addend) followed by the three
//! production-shaped D2H copies (`R*2560*4`, `R*10*4`, `R*10*4` bytes, like
//! cpu_split.rs `prepare`) into one `PinnedTransferBuffer`. `ready-sync` waits
//! with `PinnedTransferBuffer::complete`; `ready-event` records one event
//! after the third copy and polls `event_query` in a tight loop. The
//! downloaded bytes are verified against the producer output every generation.
//!
//! The idle gap before each timed region is the busy compute PLUS untimed
//! per-generation housekeeping (residual reset, oracle reference, previous
//! generation's verification), which is longer than `--cpu-work-us`.
//!
//! CP-wait arms (need `hipDeviceAttributeCanUseStreamWaitValue`; they REFUSE,
//! reported in the cell and the JSON header, when it is 0 or signal memory
//! cannot be allocated):
//!   * cpwait:     enqueue `WaitValue32(join_word >= gen)` then the combine on
//!                 the mapped view; the host busy-works `--cpu-work-us`, writes
//!                 the slots, publishes, stores `gen`. `wall` is the
//!                 flag-store -> combine-complete time (compare with the
//!                 `mapped` arm's CPU-done -> combine-complete). Byte oracle
//!                 every generation.
//!   * ready-word: the readiness cell with a `WriteValue32` ready word after the
//!                 three D2H copies and the host volatile poll
//!                 (`arm_ready_word` / `try_complete_ready_word`); compare with
//!                 `ready-sync` / `ready-event`.
//!   * abort:      a wait that is never satisfied, `SIGNAL_ABORT` stored after
//!                 50 ms; the queue must drain within 1 s (`released=true`),
//!                 the result is discarded. 5 generations per row count.
//! The JSON header carries `hipDeviceAttributeCanUseStreamWaitValue`.
//!
//! Usage:
//!   d2_mapped_return_pilot [--rows 1,4,8] [--gens 200] [--arms copy,mapped]
//!       [--json out.json] [--cpu-load 0] [--cpu-work-us 400] [--seed 1]
//! Arms: copy, mapped, ready-sync, ready-event, cpwait, ready-word, abort.
//! Exit: 0 success; 1 failure (details printed); 2 usage; 3 no GPU/HIP.

use hip_bridge::{
    Event, MappedHostBuffer, PinnedTransferBuffer, SignalGen, SignalWord,
    HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE, HIP_EVENT_DISABLE_TIMING, SIGNAL_ABORT,
};
use hipfire_dispatch::cpu_exec::moe_canonical_combine_reference;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DIM: usize = 2560;
const TOP_K: usize = 10;
const N_EXPERTS: usize = 512;
const GUARD_BYTES: usize = 4096;
const SENTINEL: u8 = 0xA5;
const MAX_FAILURE_LINES: usize = 12;

type Res<T> = Result<T, String>;

fn he<E: std::fmt::Display>(ctx: &'static str) -> impl FnOnce(E) -> String {
    move |e| format!("{ctx}: {e}")
}

// ───────────────────────── small utilities ─────────────────────────

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}
extern "C" {
    fn clock_gettime(clk: i32, ts: *mut Timespec) -> i32;
}
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

/// CPU time consumed by the calling thread, in ns.
fn thread_cpu_ns() -> u64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid out-pointer to a #[repr(C)] timespec; Linux x86_64 layout.
    let rc = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return 0;
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = mix(self.0);
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16_777_216.0
    }
}

/// RAW F32 value for `(generation, slot, element)`; exactly representable,
/// range about ±8.
fn slot_value(seed: u64, g: u64, slot: u64, e: u64) -> f32 {
    let base = mix(seed ^ g.wrapping_mul(0xD6E8_FEB8_6659_FD93))
        ^ slot.wrapping_mul(0xA076_1D64_78BD_642F);
    let h = mix(base.wrapping_add(e.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    (((h >> 40) as i32) - (1 << 23)) as f32 / 1_048_576.0
}

fn fill_f32(seed: u64, g: u64, out: &mut [f32]) {
    for (slot, row) in out.chunks_mut(DIM).enumerate() {
        for (e, v) in row.iter_mut().enumerate() {
            *v = slot_value(seed, g, slot as u64, e as u64);
        }
    }
}

fn fill_bytes(seed: u64, g: u64, out: &mut [u8]) {
    for (slot, row) in out.chunks_mut(DIM * 4).enumerate() {
        for (e, v) in row.chunks_exact_mut(4).enumerate() {
            v.copy_from_slice(&slot_value(seed, g, slot as u64, e as u64).to_ne_bytes());
        }
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: any f32 slice is valid as bytes; length is exact.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn i32_bytes(v: &[i32]) -> &[u8] {
    // SAFETY: any i32 slice is valid as bytes; length is exact.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Busy host compute for `us` microseconds (the D2 CPU arm stand-in).
fn busy_work(us: u64) {
    if us == 0 {
        return;
    }
    let end = Instant::now() + std::time::Duration::from_micros(us);
    let mut acc = 1.000_1f64;
    loop {
        for _ in 0..64 {
            acc = std::hint::black_box(acc * 1.000_000_1 + 1e-9);
        }
        if Instant::now() >= end {
            break;
        }
    }
    std::hint::black_box(acc);
}

#[derive(Default, Clone)]
struct Samples(Vec<f64>);

impl Samples {
    fn push(&mut self, v: f64) {
        self.0.push(v);
    }
    fn stats(&self) -> (usize, f64, f64, f64) {
        if self.0.is_empty() {
            return (0, f64::NAN, f64::NAN, f64::NAN);
        }
        let mut s = self.0.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let n = s.len();
        let pct = |p: f64| s[((p * n as f64).ceil() as usize).clamp(1, n) - 1];
        (n, pct(0.5), pct(0.9), s.iter().sum::<f64>() / n as f64)
    }
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "null".to_string()
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Concurrent memory-streaming load on `n` threads for the whole run.
struct CpuLoad {
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<u64>>,
}

impl CpuLoad {
    fn start(n: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handles = (0..n)
            .map(|t| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut buf = vec![t as u64 + 1; 32 << 20]; // 256 MiB per thread
                    let mut sink = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        for w in buf.iter_mut() {
                            *w = w.wrapping_add(sink | 1);
                            sink ^= *w;
                        }
                    }
                    std::hint::black_box(sink)
                })
            })
            .collect();
        Self { stop, handles }
    }
    fn finish(self) {
        self.stop.store(true, Ordering::Relaxed);
        for h in self.handles {
            let _ = h.join();
        }
    }
}

// ───────────────────────── fixtures ─────────────────────────

struct Fixture {
    rows: usize,
    ids: Vec<i32>,
    weights: Vec<f32>,
    pristine: Vec<f32>,
    d_resid: GpuTensor,
    d_ids: GpuTensor,
    d_weights: GpuTensor,
    /// Readiness-cell producer input/output, `[R,2560]`.
    d_work: GpuTensor,
    /// Generation-dependent addend of the readiness producer, `[R,2560]`.
    d_addend: GpuTensor,
}

fn alloc_f32(gpu: &mut Gpu, n: usize) -> Res<GpuTensor> {
    gpu.alloc_tensor(&[n], DType::F32)
        .map_err(he("alloc_tensor"))
}

impl Fixture {
    fn new(gpu: &mut Gpu, rows: usize, seed: u64) -> Res<Self> {
        let mut rng = Rng(seed ^ (rows as u64).wrapping_mul(0x1234_5678_9ABC_DEF1));
        let mut ids = Vec::with_capacity(rows * TOP_K);
        let mut weights = Vec::with_capacity(rows * TOP_K);
        for _ in 0..rows {
            // Distinct valid ids in random (unsorted) order: partial Fisher-Yates.
            let mut pool: Vec<i32> = (0..N_EXPERTS as i32).collect();
            for j in 0..TOP_K {
                let k = j + (rng.next() as usize) % (N_EXPERTS - j);
                pool.swap(j, k);
                ids.push(pool[j]);
                weights.push(0.02 + 0.2 * rng.unit());
            }
        }
        let pristine: Vec<f32> = (0..rows * DIM).map(|_| (rng.unit() - 0.5) * 4.0).collect();
        let d_resid = alloc_f32(gpu, rows * DIM)?;
        let d_ids = alloc_f32(gpu, rows * TOP_K)?;
        let d_weights = alloc_f32(gpu, rows * TOP_K)?;
        let d_work = alloc_f32(gpu, rows * DIM)?;
        let d_addend = alloc_f32(gpu, rows * DIM)?;
        gpu.hip
            .memcpy_htod(&d_ids.buf, i32_bytes(&ids))
            .map_err(he("upload ids"))?;
        gpu.hip
            .memcpy_htod(&d_weights.buf, f32_bytes(&weights))
            .map_err(he("upload weights"))?;
        Ok(Self {
            rows,
            ids,
            weights,
            pristine,
            d_resid,
            d_ids,
            d_weights,
            d_work,
            d_addend,
        })
    }

    fn reset_residual(&self, gpu: &Gpu) -> Res<()> {
        gpu.hip
            .memcpy_htod(&self.d_resid.buf, f32_bytes(&self.pristine))
            .map_err(he("reset residual"))
    }

    /// Canonical oracle for generation `g`.
    fn reference(&self, seed: u64, g: u64) -> Vec<f32> {
        let mut expanded = vec![0f32; self.rows * TOP_K * DIM];
        fill_f32(seed, g, &mut expanded);
        let mut target = self.pristine.clone();
        moe_canonical_combine_reference(
            DIM,
            self.rows,
            TOP_K,
            N_EXPERTS,
            &self.ids,
            &self.weights,
            &expanded,
            &mut target,
        );
        target
    }
}

// ───────────────────────── per-cell records ─────────────────────────

#[derive(Default, Clone)]
struct Metrics {
    /// CPU-done → combine-complete (or enqueue → ready) wall, µs.
    wall: Samples,
    /// Host thread CPU time over the same interval, µs.
    cpu: Samples,
    /// H2D API call wall (copy arm), µs.
    h2d_api: Samples,
    /// Combine kernel time from events, µs.
    kernel: Samples,
    /// `retire()` wall (mapped arm), µs.
    retire: Samples,
    /// Polls that returned "not ready" per generation (ready-event).
    polls: Samples,
}

#[derive(Default)]
struct Cell {
    arm: &'static str,
    rows: usize,
    gens: u64,
    first: Option<Vec<(&'static str, f64)>>,
    metrics: Metrics,
    oracle_ok: u64,
    oracle_fail: u64,
    guard_fail: u64,
    cross_ok: u64,
    cross_fail: u64,
    failures: Vec<String>,
    /// Why the cell did not run (capability refusal).
    refused: Option<String>,
    /// `abort` cell: every generation's queue drained after the abort store.
    released: Option<bool>,
}

impl Cell {
    fn new(arm: &'static str, rows: usize, gens: u64) -> Self {
        Self {
            arm,
            rows,
            gens,
            ..Self::default()
        }
    }
    /// A cell the device cannot run: reported, never a silent success or failure.
    fn refused(arm: &'static str, rows: usize, gens: u64, why: String) -> Self {
        Self {
            refused: Some(why),
            ..Self::new(arm, rows, gens)
        }
    }
    fn fail(&mut self, msg: String) {
        if self.failures.len() < MAX_FAILURE_LINES {
            self.failures.push(msg);
        }
    }
    fn failed(&self) -> bool {
        self.oracle_fail + self.guard_fail + self.cross_fail > 0 || !self.failures.is_empty()
    }
}

fn first_bit_diff(got: &[f32], want: &[f32]) -> Option<(usize, f32, f32)> {
    got.iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, got[i], want[i]))
}

fn us(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

// ───────────────────────── combine cells ─────────────────────────

struct Sample {
    wall: f64,
    cpu: f64,
    h2d_api: Option<f64>,
    kernel: f64,
    retire: Option<f64>,
}

fn record(cell: &mut Cell, g: u64, s: &Sample) {
    if g == 0 {
        let mut first = vec![
            ("wall_us", s.wall),
            ("cpu_us", s.cpu),
            ("kernel_us", s.kernel),
        ];
        if let Some(v) = s.h2d_api {
            first.push(("h2d_api_us", v));
        }
        if let Some(v) = s.retire {
            first.push(("retire_us", v));
        }
        cell.first = Some(first);
        return;
    }
    cell.metrics.wall.push(s.wall);
    cell.metrics.cpu.push(s.cpu);
    cell.metrics.kernel.push(s.kernel);
    if let Some(v) = s.h2d_api {
        cell.metrics.h2d_api.push(v);
    }
    if let Some(v) = s.retire {
        cell.metrics.retire.push(v);
    }
}

fn check_oracle(cell: &mut Cell, fx: &Fixture, gpu: &Gpu, g: u64, want: &[f32]) -> Res<Vec<f32>> {
    let got = gpu
        .download_f32(&fx.d_resid)
        .map_err(he("download residual"))?;
    match first_bit_diff(&got, want) {
        None => cell.oracle_ok += 1,
        Some((i, a, b)) => {
            cell.oracle_fail += 1;
            let n = got
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            cell.fail(format!(
                "{} R={} gen {g}: residual != canonical reference at element {i} (row {} col {}): gpu {a:e} ({:#010x}) ref {b:e} ({:#010x}); {n} elements differ",
                cell.arm, cell.rows, i / DIM, i % DIM, a.to_bits(), b.to_bits()
            ));
        }
    }
    Ok(got)
}

fn run_combine_cell(
    gpu: &mut Gpu,
    fx: &Fixture,
    use_copy: bool,
    use_mapped: bool,
    gens: u64,
    work_us: u64,
    seed: u64,
) -> Res<Vec<Cell>> {
    let rows = fx.rows;
    let payload_f32 = rows * TOP_K * DIM;
    let payload_bytes = payload_f32 * 4;
    let ev_start = gpu.hip.event_create().map_err(he("event_create"))?;
    let ev_stop = gpu.hip.event_create().map_err(he("event_create"))?;

    // Copy arm resources.
    let mut copy_state: Option<(PinnedTransferBuffer, GpuTensor)> = if use_copy {
        let pinned = gpu
            .hip
            .transfer_buffer(payload_bytes)
            .map_err(he("transfer_buffer"))?;
        let dev = alloc_f32(gpu, payload_f32)?;
        Some((pinned, dev))
    } else {
        None
    };
    // Mapped arm resources.
    let mut mapped: Option<MappedHostBuffer> = if use_mapped {
        let mut m = gpu
            .hip
            .mapped_host_buffer(payload_bytes + GUARD_BYTES)
            .map_err(he("mapped_host_buffer"))?;
        {
            let mut w = m.begin_write().map_err(he("begin_write"))?;
            w.bytes_mut()[payload_bytes..].fill(SENTINEL);
            // Dropped unpublished: back to Retired, generation unchanged.
        }
        Some(m)
    } else {
        None
    };
    let mapped_tensor: Option<GpuTensor> = mapped.as_ref().map(|m| GpuTensor {
        // SAFETY: non-owning view of the mapped allocation; `m` outlives every
        // use (it is dropped after this tensor at the end of the function) and
        // the tensor is never passed to a free path.
        buf: unsafe { m.device_buffer().alias() },
        shape: vec![payload_f32],
        dtype: DType::F32,
    });
    let mut copy_cell = Cell::new("copy", rows, gens);
    let mut mapped_cell = Cell::new("mapped", rows, gens);

    for g in 0..gens {
        // Untimed housekeeping first (previous generation is fully retired).
        let want = fx.reference(seed, g);
        let copy_first = g % 2 == 0;
        let mut copy_out: Option<Vec<f32>> = None;
        let mut mapped_out: Option<Vec<f32>> = None;
        for phase in 0..2 {
            let is_copy = (phase == 0) == copy_first;
            if is_copy && use_copy {
                let (pinned, dev) = copy_state.as_mut().expect("copy state");
                fx.reset_residual(gpu)?;
                // Retire the previous upload before reusing the pinned buffer.
                pinned.complete(&gpu.hip).map_err(he("pinned complete"))?;
                busy_work(work_us);
                fill_bytes(seed, g, &mut pinned.as_bytes_mut()[..payload_bytes]);
                let t_done = Instant::now();
                let c0 = thread_cpu_ns();
                pinned
                    .upload_async(
                        &gpu.hip,
                        &dev.buf,
                        0,
                        payload_bytes,
                        gpu.active_stream.as_ref(),
                    )
                    .map_err(he("upload_async"))?;
                let h2d = us(t_done.elapsed());
                gpu.hip
                    .event_record(&ev_start, gpu.active_stream.as_ref())
                    .map_err(he("record start"))?;
                gpu.moe_down_combine_top10_batched(
                    dev,
                    &fx.d_ids,
                    &fx.d_weights,
                    &fx.d_resid,
                    DIM,
                    rows,
                    N_EXPERTS,
                )
                .map_err(he("combine (copy)"))?;
                gpu.hip
                    .event_record(&ev_stop, gpu.active_stream.as_ref())
                    .map_err(he("record stop"))?;
                gpu.hip
                    .event_synchronize(&ev_stop)
                    .map_err(he("sync stop"))?;
                let wall = us(t_done.elapsed());
                let cpu = (thread_cpu_ns().saturating_sub(c0)) as f64 / 1e3;
                let kernel = gpu
                    .hip
                    .event_elapsed_ms(&ev_start, &ev_stop)
                    .map_err(he("elapsed"))? as f64
                    * 1e3;
                record(
                    &mut copy_cell,
                    g,
                    &Sample {
                        wall,
                        cpu,
                        h2d_api: Some(h2d),
                        kernel,
                        retire: None,
                    },
                );
                copy_out = Some(check_oracle(&mut copy_cell, fx, gpu, g, &want)?);
            } else if !is_copy && use_mapped {
                let m = mapped.as_mut().expect("mapped state");
                let tensor = mapped_tensor.as_ref().expect("mapped tensor");
                fx.reset_residual(gpu)?;
                busy_work(work_us);
                let token = {
                    let mut w = m.begin_write().map_err(he("begin_write"))?;
                    if w.bytes_mut()[payload_bytes..]
                        .iter()
                        .any(|&b| b != SENTINEL)
                    {
                        mapped_cell.guard_fail += 1;
                        mapped_cell.fail(format!(
                            "mapped R={rows} gen {g}: guard bytes were modified"
                        ));
                        w.bytes_mut()[payload_bytes..].fill(SENTINEL);
                    }
                    let f = w.f32_mut().map_err(he("f32_mut"))?;
                    fill_f32(seed, g, &mut f[..payload_f32]);
                    w.publish()
                };
                let t_done = Instant::now();
                let c0 = thread_cpu_ns();
                gpu.hip
                    .event_record(&ev_start, gpu.active_stream.as_ref())
                    .map_err(he("record start"))?;
                gpu.moe_down_combine_top10_batched(
                    tensor,
                    &fx.d_ids,
                    &fx.d_weights,
                    &fx.d_resid,
                    DIM,
                    rows,
                    N_EXPERTS,
                )
                .map_err(he("combine (mapped)"))?;
                gpu.hip
                    .event_record(&ev_stop, gpu.active_stream.as_ref())
                    .map_err(he("record stop"))?;
                m.mark_gpu_reading(&gpu.hip, token, gpu.active_stream.as_ref())
                    .map_err(he("mark_gpu_reading"))?;
                let t_retire = Instant::now();
                m.retire(&gpu.hip).map_err(he("retire"))?;
                let wall = us(t_done.elapsed());
                let retire = us(t_retire.elapsed());
                let cpu = (thread_cpu_ns().saturating_sub(c0)) as f64 / 1e3;
                let kernel = gpu
                    .hip
                    .event_elapsed_ms(&ev_start, &ev_stop)
                    .map_err(he("elapsed"))? as f64
                    * 1e3;
                record(
                    &mut mapped_cell,
                    g,
                    &Sample {
                        wall,
                        cpu,
                        h2d_api: None,
                        kernel,
                        retire: Some(retire),
                    },
                );
                mapped_out = Some(check_oracle(&mut mapped_cell, fx, gpu, g, &want)?);
            }
        }
        if let (Some(a), Some(b)) = (&copy_out, &mapped_out) {
            match first_bit_diff(a, b) {
                None => {
                    copy_cell.cross_ok += 1;
                    mapped_cell.cross_ok += 1;
                }
                Some((i, x, y)) => {
                    mapped_cell.cross_fail += 1;
                    mapped_cell.fail(format!(
                        "R={rows} gen {g}: copy vs mapped residual differ at element {i}: copy {x:e} mapped {y:e}"
                    ));
                }
            }
        }
    }
    // The last generation's guard check needs a final (abandoned) write window.
    if let Some(m) = mapped.as_mut() {
        let mut w = m.begin_write().map_err(he("final begin_write"))?;
        if w.bytes_mut()[payload_bytes..]
            .iter()
            .any(|&b| b != SENTINEL)
        {
            mapped_cell.guard_fail += 1;
            mapped_cell.fail(format!(
                "mapped R={rows} after gen {}: guard bytes were modified",
                gens - 1
            ));
        }
    }
    drop(mapped_tensor);
    drop(mapped);
    drop(copy_state);
    let _ = gpu.hip.event_destroy(ev_start);
    let _ = gpu.hip.event_destroy(ev_stop);
    let mut cells = Vec::new();
    if use_copy {
        cells.push(copy_cell);
    }
    if use_mapped {
        cells.push(mapped_cell);
    }
    Ok(cells)
}

// ───────────────────────── readiness cells ─────────────────────────

fn run_ready_cell(
    gpu: &mut Gpu,
    fx: &Fixture,
    event_mode: bool,
    gens: u64,
    work_us: u64,
    seed: u64,
) -> Res<Cell> {
    let rows = fx.rows;
    let nx = rows * DIM;
    let slots = rows * TOP_K;
    let mut cell = Cell::new(
        if event_mode {
            "ready-event"
        } else {
            "ready-sync"
        },
        rows,
        gens,
    );
    let mut pinned = gpu
        .hip
        .transfer_buffer((nx + 2 * slots) * 4)
        .map_err(he("transfer_buffer"))?;
    let ev: Event = gpu
        .hip
        .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
        .map_err(he("event_create"))?;
    for g in 0..gens {
        pinned.complete(&gpu.hip).map_err(he("pinned complete"))?;
        let mut addend = vec![0f32; nx];
        fill_f32(seed ^ 0x5EED_ADD, g, &mut addend);
        gpu.hip
            .memcpy_htod(&fx.d_work.buf, f32_bytes(&fx.pristine))
            .map_err(he("reset work"))?;
        gpu.hip
            .memcpy_htod(&fx.d_addend.buf, f32_bytes(&addend))
            .map_err(he("upload addend"))?;
        busy_work(work_us);
        let t0 = Instant::now();
        let c0 = thread_cpu_ns();
        gpu.add_inplace_f32(&fx.d_work, &fx.d_addend)
            .map_err(he("add_inplace_f32"))?;
        let stream = gpu.active_stream.as_ref();
        pinned
            .download_async(&gpu.hip, &fx.d_work.buf, 0, nx * 4, stream)
            .map_err(he("download x"))?;
        pinned
            .download_async(&gpu.hip, &fx.d_ids.buf, nx * 4, slots * 4, stream)
            .map_err(he("download ids"))?;
        pinned
            .download_async(
                &gpu.hip,
                &fx.d_weights.buf,
                (nx + slots) * 4,
                slots * 4,
                stream,
            )
            .map_err(he("download weights"))?;
        let mut not_ready = 0u64;
        if event_mode {
            gpu.hip
                .event_record(&ev, gpu.active_stream.as_ref())
                .map_err(he("record ready"))?;
            while !gpu.hip.event_query(&ev).map_err(he("event_query"))? {
                not_ready += 1;
            }
        } else {
            pinned.complete(&gpu.hip).map_err(he("pinned complete"))?;
        }
        let wall = us(t0.elapsed());
        let cpu = (thread_cpu_ns().saturating_sub(c0)) as f64 / 1e3;
        // Untimed: leave the pinned owner's pending flag consistent, then verify.
        pinned.complete(&gpu.hip).map_err(he("pinned complete"))?;
        if g == 0 {
            cell.first = Some(vec![
                ("wall_us", wall),
                ("cpu_us", cpu),
                ("polls", not_ready as f64),
            ]);
        } else {
            cell.metrics.wall.push(wall);
            cell.metrics.cpu.push(cpu);
            if event_mode {
                cell.metrics.polls.push(not_ready as f64);
            }
        }
        let bytes = pinned.as_bytes_mut();
        let mut bad: Option<String> = None;
        for (i, c) in bytes[..nx * 4].chunks_exact(4).enumerate() {
            let want = (fx.pristine[i] + addend[i]).to_bits();
            let got = u32::from_ne_bytes(c.try_into().unwrap());
            if got != want {
                bad = Some(format!("x[{i}] {got:#010x} != producer {want:#010x}"));
                break;
            }
        }
        if bad.is_none() && bytes[nx * 4..(nx + slots) * 4] != *i32_bytes(&fx.ids) {
            bad = Some("downloaded ids differ from device ids".to_string());
        }
        if bad.is_none() && bytes[(nx + slots) * 4..] != *f32_bytes(&fx.weights) {
            bad = Some("downloaded weights differ from device weights".to_string());
        }
        if let Some(why) = bad {
            cell.oracle_fail += 1;
            cell.fail(format!("{} R={rows} gen {g}: {why}", cell.arm));
        } else {
            cell.oracle_ok += 1;
        }
    }
    let _ = gpu.hip.event_destroy(ev);
    Ok(cell)
}

// ───────────────────────── CP-wait cells ─────────────────────────

/// `hipDeviceAttributeCanUseStreamWaitValue` of the device, and whether the
/// bridge accepts stream value operations on it.
struct WaitCap {
    /// Raw attribute value; `None` when the query failed.
    attribute: Option<i32>,
    /// `HipRuntime::can_use_stream_wait_value`; `None` when the query failed.
    usable: Option<bool>,
    query_error: Option<String>,
}

impl WaitCap {
    /// Why the cpwait, ready-word and abort cells must refuse, or `None`.
    fn refusal(&self) -> Option<String> {
        match (self.usable, &self.query_error) {
            (Some(true), _) => None,
            (Some(false), _) => Some(format!(
                "hipDeviceAttributeCanUseStreamWaitValue is {}: stream value waits are unsupported on this device",
                self.attribute.map_or("0".to_string(), |v| v.to_string())
            )),
            (None, Some(e)) => Some(format!("capability query failed: {e}")),
            (None, None) => Some("capability unknown".to_string()),
        }
    }
}

fn probe_wait_cap(gpu: &Gpu) -> WaitCap {
    let device = gpu.device_id;
    let attribute = gpu
        .hip
        .get_device_attribute(HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE, device);
    let usable = gpu.hip.can_use_stream_wait_value(device);
    WaitCap {
        attribute: attribute.as_ref().ok().copied(),
        usable: usable.as_ref().ok().copied(),
        query_error: usable.err().or(attribute.err()).map(|e| e.to_string()),
    }
}

/// Generations of the abort cell (each waits 50 ms before its abort store).
const ABORT_GENS: u64 = 5;
/// How long the never-satisfied wait must hold before the abort store.
const ABORT_HOLD: std::time::Duration = std::time::Duration::from_millis(50);
/// How long the queue gets to drain after the abort store.
const ABORT_DRAIN_LIMIT: std::time::Duration = std::time::Duration::from_secs(1);
/// Bound of a ready-word spin before the cell gives up.
const READY_WORD_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);

/// One mapped result allocation with a guard region and its device view.
struct MappedRig {
    tensor: GpuTensor,
    buf: MappedHostBuffer,
    payload_f32: usize,
}

impl MappedRig {
    fn new(gpu: &Gpu, payload_f32: usize) -> Res<Self> {
        let payload_bytes = payload_f32 * 4;
        let mut buf = gpu
            .hip
            .mapped_host_buffer(payload_bytes + GUARD_BYTES)
            .map_err(he("mapped_host_buffer"))?;
        {
            let mut w = buf.begin_write().map_err(he("begin_write"))?;
            w.bytes_mut()[payload_bytes..].fill(SENTINEL);
        }
        let tensor = GpuTensor {
            // SAFETY: non-owning view of the mapped allocation owned by `buf`
            // in the same struct (dropped together, tensor first); the tensor
            // is never passed to a free path.
            buf: unsafe { buf.device_buffer().alias() },
            shape: vec![payload_f32],
            dtype: DType::F32,
        };
        Ok(Self {
            tensor,
            buf,
            payload_f32,
        })
    }

    /// Write generation `g`'s slots (guard bytes checked first) and publish.
    fn write_generation(
        &mut self,
        seed: u64,
        g: u64,
        cell: &mut Cell,
    ) -> Res<hip_bridge::MappedPublished> {
        let payload_bytes = self.payload_f32 * 4;
        let mut w = self.buf.begin_write().map_err(he("begin_write"))?;
        if w.bytes_mut()[payload_bytes..]
            .iter()
            .any(|&b| b != SENTINEL)
        {
            cell.guard_fail += 1;
            cell.fail(format!(
                "{} R={} gen {g}: guard bytes were modified",
                cell.arm, cell.rows
            ));
            w.bytes_mut()[payload_bytes..].fill(SENTINEL);
        }
        let f = w.f32_mut().map_err(he("f32_mut"))?;
        fill_f32(seed, g, &mut f[..self.payload_f32]);
        Ok(w.publish())
    }
}

/// One cpwait generation, from the wait enqueue to the combine's completion.
/// EVERY error between the wait enqueue and the flag store stores
/// `SIGNAL_ABORT` before returning: a device synchronize with a pending wait
/// never returns.
#[allow(clippy::too_many_arguments)]
fn cpwait_generation(
    gpu: &mut Gpu,
    fx: &Fixture,
    rig: &mut MappedRig,
    word: &SignalWord,
    ev_start: &Event,
    ev_stop: &Event,
    gen: u32,
    g: u64,
    work_us: u64,
    seed: u64,
    cell: &mut Cell,
) -> Res<Sample> {
    let rows = fx.rows;
    let stored = std::cell::Cell::new(false);
    let abort_unless_stored = |why: String| {
        if !stored.get() {
            word.store_release(SIGNAL_ABORT);
        }
        why
    };
    gpu.hip
        .stream_wait_signal_gte(gpu.active_stream.as_ref(), word, gen)
        .map_err(|e| abort_unless_stored(format!("wait enqueue: {e}")))?;
    let mut body = || -> Res<Sample> {
        gpu.hip
            .event_record(ev_start, gpu.active_stream.as_ref())
            .map_err(he("record start"))?;
        gpu.moe_down_combine_top10_batched(
            &rig.tensor,
            &fx.d_ids,
            &fx.d_weights,
            &fx.d_resid,
            DIM,
            rows,
            N_EXPERTS,
        )
        .map_err(he("combine (cpwait)"))?;
        gpu.hip
            .event_record(ev_stop, gpu.active_stream.as_ref())
            .map_err(he("record stop"))?;
        // The combine is queued behind the wait. The CPU arm runs now.
        busy_work(work_us);
        let token = rig.write_generation(seed, g, cell)?;
        rig.buf
            .mark_gpu_reading(&gpu.hip, token, gpu.active_stream.as_ref())
            .map_err(he("mark_gpu_reading"))?;
        let c0 = thread_cpu_ns();
        let t_store = Instant::now();
        stored.set(true);
        word.store_release(gen);
        gpu.hip
            .event_synchronize(ev_stop)
            .map_err(he("sync stop"))?;
        let wall = us(t_store.elapsed());
        let cpu = (thread_cpu_ns().saturating_sub(c0)) as f64 / 1e3;
        let kernel = gpu
            .hip
            .event_elapsed_ms(ev_start, ev_stop)
            .map_err(he("elapsed"))? as f64
            * 1e3;
        Ok(Sample {
            wall,
            cpu,
            h2d_api: None,
            kernel,
            retire: None,
        })
    };
    body().map_err(abort_unless_stored)
}

/// `cpwait`: enqueue `WaitValue32(gen)` then the combine on the mapped view;
/// the host busy-works `work_us`, writes the slots, stores `gen`, and the cell
/// measures the flag-store -> combine-complete wall. Byte oracle every
/// generation.
fn run_cpwait_cell(gpu: &mut Gpu, fx: &Fixture, gens: u64, work_us: u64, seed: u64) -> Res<Cell> {
    let rows = fx.rows;
    let mut cell = Cell::new("cpwait", rows, gens);
    let word = match gpu.hip.signal_word() {
        Ok(w) => w,
        Err(e) => {
            return Ok(Cell::refused(
                "cpwait",
                rows,
                gens,
                format!("signal word allocation failed: {e}"),
            ))
        }
    };
    let mut rig = MappedRig::new(gpu, rows * TOP_K * DIM)?;
    let ev_start = gpu.hip.event_create().map_err(he("event_create"))?;
    let ev_stop = gpu.hip.event_create().map_err(he("event_create"))?;
    let mut siggen = SignalGen::new();
    let outcome = (|| -> Res<()> {
        for g in 0..gens {
            let want = fx.reference(seed, g);
            let gen = siggen.next().map_err(he("signal generation"))?;
            fx.reset_residual(gpu)?;
            rig.buf.retire(&gpu.hip).map_err(he("retire"))?;
            let sample = cpwait_generation(
                gpu, fx, &mut rig, &word, &ev_start, &ev_stop, gen, g, work_us, seed, &mut cell,
            )?;
            record(&mut cell, g, &sample);
            check_oracle(&mut cell, fx, gpu, g, &want)?;
        }
        Ok(())
    })();
    if outcome.is_ok() {
        // The last generation's guard check needs a final (abandoned) window.
        rig.buf.retire(&gpu.hip).map_err(he("final retire"))?;
        let payload_bytes = rig.payload_f32 * 4;
        let mut w = rig.buf.begin_write().map_err(he("final begin_write"))?;
        if w.bytes_mut()[payload_bytes..]
            .iter()
            .any(|&b| b != SENTINEL)
        {
            cell.guard_fail += 1;
            cell.fail(format!("cpwait R={rows}: guard bytes were modified"));
        }
    }
    let _ = gpu.hip.event_destroy(ev_start);
    let _ = gpu.hip.event_destroy(ev_stop);
    outcome?;
    Ok(cell)
}

/// `ready-word`: the readiness cell with a `WriteValue32` ready word after the
/// three D2H copies and the production `arm_ready_word` /
/// `try_complete_ready_word` host poll instead of a stream sync or an event.
fn run_ready_word_cell(
    gpu: &mut Gpu,
    fx: &Fixture,
    gens: u64,
    work_us: u64,
    seed: u64,
) -> Res<Cell> {
    let rows = fx.rows;
    let nx = rows * DIM;
    let slots = rows * TOP_K;
    let mut cell = Cell::new("ready-word", rows, gens);
    let word = match gpu.hip.signal_word() {
        Ok(w) => w,
        Err(e) => {
            return Ok(Cell::refused(
                "ready-word",
                rows,
                gens,
                format!("signal word allocation failed: {e}"),
            ))
        }
    };
    let mut pinned = gpu
        .hip
        .transfer_buffer((nx + 2 * slots) * 4)
        .map_err(he("transfer_buffer"))?;
    let mut siggen = SignalGen::new();
    for g in 0..gens {
        pinned.complete(&gpu.hip).map_err(he("pinned complete"))?;
        let gen = siggen.next().map_err(he("signal generation"))?;
        let mut addend = vec![0f32; nx];
        fill_f32(seed ^ 0x5EED_ADD, g, &mut addend);
        gpu.hip
            .memcpy_htod(&fx.d_work.buf, f32_bytes(&fx.pristine))
            .map_err(he("reset work"))?;
        gpu.hip
            .memcpy_htod(&fx.d_addend.buf, f32_bytes(&addend))
            .map_err(he("upload addend"))?;
        busy_work(work_us);
        let t0 = Instant::now();
        let c0 = thread_cpu_ns();
        gpu.add_inplace_f32(&fx.d_work, &fx.d_addend)
            .map_err(he("add_inplace_f32"))?;
        let stream = gpu.active_stream.as_ref();
        pinned
            .download_async(&gpu.hip, &fx.d_work.buf, 0, nx * 4, stream)
            .map_err(he("download x"))?;
        pinned
            .download_async(&gpu.hip, &fx.d_ids.buf, nx * 4, slots * 4, stream)
            .map_err(he("download ids"))?;
        pinned
            .download_async(
                &gpu.hip,
                &fx.d_weights.buf,
                (nx + slots) * 4,
                slots * 4,
                stream,
            )
            .map_err(he("download weights"))?;
        let mut proof = pinned
            .arm_ready_word(&gpu.hip, &word, gen)
            .map_err(he("arm_ready_word"))?;
        let mut not_ready = 0u64;
        while !pinned
            .try_complete_ready_word(&mut proof, &word)
            .map_err(he("try_complete_ready_word"))?
        {
            not_ready += 1;
            if t0.elapsed() > READY_WORD_LIMIT {
                return Err(format!(
                    "ready-word R={rows} gen {g}: the ready word never reached {gen} (stuck at {})",
                    word.load()
                ));
            }
            std::hint::spin_loop();
        }
        let wall = us(t0.elapsed());
        let cpu = (thread_cpu_ns().saturating_sub(c0)) as f64 / 1e3;
        if g == 0 {
            cell.first = Some(vec![
                ("wall_us", wall),
                ("cpu_us", cpu),
                ("polls", not_ready as f64),
            ]);
        } else {
            cell.metrics.wall.push(wall);
            cell.metrics.cpu.push(cpu);
            cell.metrics.polls.push(not_ready as f64);
        }
        let bytes = pinned.as_bytes_mut();
        let mut bad: Option<String> = None;
        for (i, c) in bytes[..nx * 4].chunks_exact(4).enumerate() {
            let want = (fx.pristine[i] + addend[i]).to_bits();
            let got = u32::from_ne_bytes(c.try_into().unwrap());
            if got != want {
                bad = Some(format!("x[{i}] {got:#010x} != producer {want:#010x}"));
                break;
            }
        }
        if bad.is_none() && bytes[nx * 4..(nx + slots) * 4] != *i32_bytes(&fx.ids) {
            bad = Some("downloaded ids differ from device ids".to_string());
        }
        if bad.is_none() && bytes[(nx + slots) * 4..] != *f32_bytes(&fx.weights) {
            bad = Some("downloaded weights differ from device weights".to_string());
        }
        if let Some(why) = bad {
            cell.oracle_fail += 1;
            cell.fail(format!("ready-word R={rows} gen {g}: {why}"));
        } else {
            cell.oracle_ok += 1;
        }
    }
    Ok(cell)
}

/// `abort`: enqueue a wait that is never satisfied, store `SIGNAL_ABORT` after
/// 50 ms, and verify the queue drains within 1 s (`released=true`). The
/// consumer ran on garbage, so its output is never compared. Each generation
/// uses a fresh word: a word that held the abort value releases every later wait.
fn run_abort_cell(gpu: &mut Gpu, fx: &Fixture) -> Res<Cell> {
    let rows = fx.rows;
    let mut cell = Cell::new("abort", rows, ABORT_GENS);
    let rig = MappedRig::new(gpu, rows * TOP_K * DIM)?;
    let ev_start = gpu.hip.event_create().map_err(he("event_create"))?;
    let ev_stop = gpu.hip.event_create().map_err(he("event_create"))?;
    for g in 0..ABORT_GENS {
        let word = match gpu.hip.signal_word() {
            Ok(w) => w,
            Err(e) => {
                let _ = gpu.hip.event_destroy(ev_start);
                let _ = gpu.hip.event_destroy(ev_stop);
                return Ok(Cell::refused(
                    "abort",
                    rows,
                    ABORT_GENS,
                    format!("signal word allocation failed: {e}"),
                ));
            }
        };
        fx.reset_residual(gpu)?;
        // Generation 1 is never written: the word stays 0.
        gpu.hip
            .stream_wait_signal_gte(gpu.active_stream.as_ref(), &word, 1)
            .map_err(|e| {
                word.store_release(SIGNAL_ABORT);
                format!("abort cell: wait enqueue: {e}")
            })?;
        gpu.hip
            .event_record(&ev_start, gpu.active_stream.as_ref())
            .map_err(|e| {
                word.store_release(SIGNAL_ABORT);
                format!("record start: {e}")
            })?;
        gpu.moe_down_combine_top10_batched(
            &rig.tensor,
            &fx.d_ids,
            &fx.d_weights,
            &fx.d_resid,
            DIM,
            rows,
            N_EXPERTS,
        )
        .map_err(|e| {
            word.store_release(SIGNAL_ABORT);
            format!("combine (abort): {e}")
        })?;
        gpu.hip
            .event_record(&ev_stop, gpu.active_stream.as_ref())
            .map_err(|e| {
                word.store_release(SIGNAL_ABORT);
                format!("record stop: {e}")
            })?;
        std::thread::sleep(ABORT_HOLD);
        let held = !gpu.hip.event_query(&ev_stop).map_err(|e| {
            word.store_release(SIGNAL_ABORT);
            format!("event_query: {e}")
        })?;
        let t_abort = Instant::now();
        word.store_release(SIGNAL_ABORT);
        let mut released = false;
        while t_abort.elapsed() < ABORT_DRAIN_LIMIT {
            if gpu.hip.event_query(&ev_stop).map_err(he("event_query"))? {
                released = true;
                break;
            }
            std::hint::spin_loop();
        }
        let drain = us(t_abort.elapsed());
        if !released {
            cell.oracle_fail += 1;
            cell.released = Some(false);
            cell.fail(format!(
                "abort R={rows} gen {g}: the queue did not drain within {} ms after the abort store",
                ABORT_DRAIN_LIMIT.as_millis()
            ));
            // A stuck queue would hang every drain: retain the owners and stop.
            std::mem::forget(word);
            std::mem::forget(rig);
            return Err(format!(
                "abort R={rows} gen {g}: released=false; the queue stayed blocked after the abort store"
            ));
        }
        if held {
            cell.oracle_ok += 1;
        } else {
            cell.oracle_fail += 1;
            cell.fail(format!(
                "abort R={rows} gen {g}: the unsatisfied wait did not hold the queue for {} ms (the combine ran before the abort store)",
                ABORT_HOLD.as_millis()
            ));
        }
        cell.metrics.wall.push(drain);
        // Released: the drain below cannot hang. The garbage result is dropped.
        gpu.hip
            .device_synchronize()
            .map_err(he("drain after abort"))?;
    }
    // Every generation drained after its abort store (a stuck queue returns
    // `released=false` as a hard error above).
    cell.released = Some(true);
    let _ = gpu.hip.event_destroy(ev_start);
    let _ = gpu.hip.event_destroy(ev_stop);
    Ok(cell)
}

// ───────────────────────── driver ─────────────────────────

struct Args {
    rows: Vec<usize>,
    gens: u64,
    arms: Vec<String>,
    json: Option<String>,
    cpu_load: usize,
    cpu_work_us: u64,
    seed: u64,
}

fn parse_args() -> Res<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        rows: vec![1, 4, 8],
        gens: 200,
        arms: vec!["copy".into(), "mapped".into()],
        json: None,
        cpu_load: 0,
        cpu_work_us: 400,
        seed: 1,
    };
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv
            .get(i + 1)
            .ok_or_else(|| format!("{key} needs a value"))?;
        match key {
            "--rows" => {
                a.rows = val
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("--rows {s:?}: {e}"))
                    })
                    .collect::<Res<_>>()?;
            }
            "--gens" => a.gens = val.parse().map_err(|e| format!("--gens: {e}"))?,
            "--arms" => a.arms = val.split(',').map(|s| s.trim().to_string()).collect(),
            "--json" => a.json = Some(val.clone()),
            "--cpu-load" => a.cpu_load = val.parse().map_err(|e| format!("--cpu-load: {e}"))?,
            "--cpu-work-us" => {
                a.cpu_work_us = val.parse().map_err(|e| format!("--cpu-work-us: {e}"))?
            }
            "--seed" => a.seed = val.parse().map_err(|e| format!("--seed: {e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if a.rows.is_empty() || a.rows.iter().any(|&r| r == 0 || r > 8) {
        return Err("--rows must be 1..=8 values".into());
    }
    if a.gens < 2 {
        return Err("--gens must be >= 2 (generation 0 is reported separately)".into());
    }
    for arm in &a.arms {
        if ![
            "copy",
            "mapped",
            "ready-sync",
            "ready-event",
            "cpwait",
            "ready-word",
            "abort",
        ]
        .contains(&arm.as_str())
        {
            return Err(format!("unknown arm {arm:?}"));
        }
    }
    Ok(a)
}

fn metric_json(s: &Samples) -> String {
    let (n, p50, p90, mean) = s.stats();
    format!(
        "{{\"n\":{n},\"p50\":{},\"p90\":{},\"mean\":{}}}",
        json_num(p50),
        json_num(p90),
        json_num(mean)
    )
}

fn cell_json(c: &Cell) -> String {
    let mut o = String::new();
    let _ = write!(
        o,
        "{{\"arm\":{},\"rows\":{},\"gens\":{}",
        json_str(c.arm),
        c.rows,
        c.gens
    );
    o.push_str(",\"first_use\":");
    match &c.first {
        Some(f) => {
            o.push('{');
            for (i, (k, v)) in f.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                let _ = write!(o, "{}:{}", json_str(k), json_num(*v));
            }
            o.push('}');
        }
        None => o.push_str("null"),
    }
    let m = &c.metrics;
    o.push_str(",\"stats_us\":{");
    let _ = write!(
        o,
        "\"wall\":{},\"host_thread_cpu\":{}",
        metric_json(&m.wall),
        metric_json(&m.cpu)
    );
    match c.arm {
        "copy" => {
            let _ = write!(
                o,
                ",\"h2d_api\":{},\"combine_kernel\":{}",
                metric_json(&m.h2d_api),
                metric_json(&m.kernel)
            );
        }
        "mapped" => {
            let _ = write!(
                o,
                ",\"retire\":{},\"combine_kernel\":{}",
                metric_json(&m.retire),
                metric_json(&m.kernel)
            );
        }
        "ready-event" | "ready-word" => {
            let _ = write!(o, ",\"polls_per_gen\":{}", metric_json(&m.polls));
        }
        "cpwait" => {
            let _ = write!(o, ",\"combine_kernel\":{}", metric_json(&m.kernel));
        }
        _ => {}
    }
    o.push('}');
    let _ = write!(
        o,
        ",\"checks\":{{\"oracle_ok\":{},\"oracle_fail\":{},\"guard_fail\":{},\"copy_vs_mapped_ok\":{},\"copy_vs_mapped_fail\":{},\"failures\":[",
        c.oracle_ok, c.oracle_fail, c.guard_fail, c.cross_ok, c.cross_fail
    );
    for (i, f) in c.failures.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&json_str(f));
    }
    o.push_str("]}");
    match &c.refused {
        Some(why) => {
            let _ = write!(o, ",\"refused\":{}", json_str(why));
        }
        None => o.push_str(",\"refused\":null"),
    }
    match c.released {
        Some(r) => {
            let _ = write!(o, ",\"released\":{r}");
        }
        None => o.push_str(",\"released\":null"),
    }
    o.push('}');
    o
}

fn run(args: &Args, cap: &WaitCap, gpu: &mut Gpu) -> Res<Vec<Cell>> {
    let want = |a: &str| args.arms.iter().any(|x| x == a);
    let (use_copy, use_mapped) = (want("copy"), want("mapped"));
    let fixtures: Vec<Fixture> = args
        .rows
        .iter()
        .map(|&r| Fixture::new(gpu, r, args.seed))
        .collect::<Res<_>>()?;
    // Compile the kernels outside every measurement (first-use transport costs
    // such as page population are still reported separately per cell).
    {
        let fx = &fixtures[0];
        let scratch = alloc_f32(gpu, fx.rows * TOP_K * DIM)?;
        gpu.hip
            .memcpy_htod(&scratch.buf, &vec![0u8; fx.rows * TOP_K * DIM * 4])
            .map_err(he("warm scratch"))?;
        fx.reset_residual(gpu)?;
        gpu.moe_down_combine_top10_batched(
            &scratch,
            &fx.d_ids,
            &fx.d_weights,
            &fx.d_resid,
            DIM,
            fx.rows,
            N_EXPERTS,
        )
        .map_err(he("warm combine"))?;
        gpu.add_inplace_f32(&fx.d_work, &fx.d_addend)
            .map_err(he("warm producer"))?;
        gpu.hip.device_synchronize().map_err(he("warm sync"))?;
    }
    let load = (args.cpu_load > 0).then(|| CpuLoad::start(args.cpu_load));
    let mut cells = Vec::new();
    let outcome = (|| -> Res<()> {
        for fx in &fixtures {
            eprintln!("rows={}", fx.rows);
            if use_copy || use_mapped {
                cells.extend(run_combine_cell(
                    gpu,
                    fx,
                    use_copy,
                    use_mapped,
                    args.gens,
                    args.cpu_work_us,
                    args.seed,
                )?);
            }
            if want("ready-sync") {
                cells.push(run_ready_cell(
                    gpu,
                    fx,
                    false,
                    args.gens,
                    args.cpu_work_us,
                    args.seed,
                )?);
            }
            if want("ready-event") {
                cells.push(run_ready_cell(
                    gpu,
                    fx,
                    true,
                    args.gens,
                    args.cpu_work_us,
                    args.seed,
                )?);
            }
            if want("cpwait") {
                cells.push(match cap.refusal() {
                    Some(why) => Cell::refused("cpwait", fx.rows, args.gens, why),
                    None => run_cpwait_cell(gpu, fx, args.gens, args.cpu_work_us, args.seed)?,
                });
            }
            if want("ready-word") {
                cells.push(match cap.refusal() {
                    Some(why) => Cell::refused("ready-word", fx.rows, args.gens, why),
                    None => run_ready_word_cell(gpu, fx, args.gens, args.cpu_work_us, args.seed)?,
                });
            }
            if want("abort") {
                cells.push(match cap.refusal() {
                    Some(why) => Cell::refused("abort", fx.rows, ABORT_GENS, why),
                    None => run_abort_cell(gpu, fx)?,
                });
            }
        }
        Ok(())
    })();
    if let Some(load) = load {
        load.finish();
    }
    outcome.map(|()| cells)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("usage error: {e}");
            std::process::exit(2);
        }
    };
    let gpu = std::panic::catch_unwind(Gpu::init);
    let mut gpu = match gpu {
        Ok(Ok(g)) => g,
        Ok(Err(e)) => {
            eprintln!("REFUSED: no usable HIP GPU ({e}); nothing measured, no success reported");
            std::process::exit(3);
        }
        Err(_) => {
            eprintln!(
                "REFUSED: HIP/GPU initialization panicked; nothing measured, no success reported"
            );
            std::process::exit(3);
        }
    };
    let arch = gpu.arch.clone();
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    eprintln!(
        "host={hostname} arch={arch} rows={:?} gens={} arms={:?} cpu_work_us={} cpu_load={}",
        args.rows, args.gens, args.arms, args.cpu_work_us, args.cpu_load
    );

    let cap = probe_wait_cap(&gpu);
    eprintln!(
        "hipDeviceAttributeCanUseStreamWaitValue={} usable={}{}",
        cap.attribute
            .map_or("unknown".to_string(), |v| v.to_string()),
        cap.usable.map_or("unknown".to_string(), |v| v.to_string()),
        cap.refusal().map_or(String::new(), |r| format!(
            " ({r}; cpwait, ready-word and abort cells refuse)"
        ))
    );
    let result = run(&args, &cap, &mut gpu);
    let (cells, hard_error) = match result {
        Ok(c) => (c, None),
        Err(e) => (Vec::new(), Some(e)),
    };

    let any_failed = cells.iter().any(Cell::failed);
    let mapped_cells: Vec<&Cell> = cells.iter().filter(|c| c.arm == "mapped").collect();
    let visibility_proved = hard_error.is_none()
        && !any_failed
        && mapped_cells.len() == args.rows.len()
        && mapped_cells
            .iter()
            .all(|c| c.oracle_ok == c.gens && c.guard_fail == 0);
    let cpwait_cells: Vec<&Cell> = cells.iter().filter(|c| c.arm == "cpwait").collect();
    let cpwait_proved = hard_error.is_none()
        && !any_failed
        && !cpwait_cells.is_empty()
        && cpwait_cells
            .iter()
            .all(|c| c.refused.is_none() && c.oracle_ok == c.gens);

    let mut json = String::new();
    let _ = write!(
        json,
        "{{\"hostname\":{},\"arch\":{},\"hipDeviceAttributeCanUseStreamWaitValue\":{},\"can_use_stream_wait_value\":{},\"rows\":[{}],\"gens\":{},\"seed\":{},\"cpu_work_us\":{},\"cpu_load_threads\":{},\"arms\":[{}],\"dim\":{DIM},\"top_k\":{TOP_K},\"n_experts\":{N_EXPERTS},\"guard_bytes\":{GUARD_BYTES}",
        json_str(&hostname),
        json_str(&arch),
        cap.attribute.map_or("null".to_string(), |v| v.to_string()),
        cap.usable.map_or("null".to_string(), |v| v.to_string()),
        args.rows.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(","),
        args.gens,
        args.seed,
        args.cpu_work_us,
        args.cpu_load,
        args.arms.iter().map(|a| json_str(a)).collect::<Vec<_>>().join(",")
    );
    json.push_str(",\"cells\":[");
    for (i, c) in cells.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&cell_json(c));
    }
    let _ = write!(
        json,
        "],\"visibility_proved\":{visibility_proved},\"cpwait_proved\":{cpwait_proved}"
    );
    match &hard_error {
        Some(e) => {
            let _ = write!(json, ",\"error\":{}", json_str(e));
        }
        None => json.push_str(",\"error\":null"),
    }
    json.push('}');

    if let Some(path) = &args.json {
        if let Err(e) = std::fs::write(path, &json) {
            eprintln!("cannot write {path}: {e}");
            std::process::exit(1);
        }
    }

    for c in &cells {
        if let Some(why) = &c.refused {
            eprintln!("{:<11} R={} REFUSED: {why}", c.arm, c.rows);
            continue;
        }
        let (n, p50, p90, mean) = c.metrics.wall.stats();
        let (_, c50, _, _) = c.metrics.cpu.stats();
        eprintln!(
            "{:<11} R={} n={n} wall p50 {:.1} p90 {:.1} mean {:.1} us; host cpu p50 {:.1} us; oracle ok {}/{}{}",
            c.arm,
            c.rows,
            p50,
            p90,
            mean,
            c50,
            c.oracle_ok,
            c.gens,
            c.released.map_or(String::new(), |r| format!("; released={r}"))
        );
    }
    if let Some(e) = hard_error {
        eprintln!("FAILED: {e}");
        std::process::exit(1);
    }
    if any_failed {
        for c in &cells {
            for f in &c.failures {
                eprintln!("FAILURE: {f}");
            }
        }
        eprintln!("FAILED: oracle/guard/cross-arm check failed; visibility_proved=false");
        std::process::exit(1);
    }
    eprintln!("OK visibility_proved={visibility_proved}");
}
