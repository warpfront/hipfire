// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-model held-staging regression for the retained gfx1201 Qwen3.5-dense
//! PM4 decode. Priority route: railgun's lowering
//! (`HIPFIRE_RAILGUN_SHADOW=<corpus>` + `HIPFIRE_RAILGUN_BACKEND=railgun`);
//! the Redline planner's tape is exercised when those variables are unset, and
//! no claim is made about which one a run used beyond the `submit lowering:`
//! line it prints.
//!
//! The production decode step stages two inputs through HIP before the PM4
//! replay: the token embedding (`embedding_lookup*` into `scratch.x`) and the
//! position word (`memcpy_htod(scratch.pos_buf)`, a synchronous hipMemcpy; see
//! `qwen35/forward.rs`, `forward_scratch`). The PM4 tape runs on its own queue.
//! Whether the embedding is ordered before the tape is therefore decided by
//! the stream the embedding runs on and by what the position copy waits for.
//! This test measures it instead of assuming it:
//!
//! * The default model route is installed exactly as the daemon does after a
//!   load: `configure_model_default(true)` + the gfx1201 dense NOP pacing.
//!   The first eligible decode records, the next ones replay PM4.
//! * Golden: `step(B)` through the prepared PM4 route, unheld, from a restored
//!   state. `step(A)` (a different token) leaves `emb(A)` in `scratch.x`.
//! * Held: state restored again, then a host-mapped release word is queued as
//!   `hipStreamWaitValue32` on a holder stream so the *embedding kernel of
//!   `step(B)`* cannot run until a host worker releases it after `HOLD`, and
//!   the real `forward_scratch(B)` is called. If the PM4 replay does not wait
//!   for the embedding it consumes the stale `emb(A)` and its logits equal the
//!   `A` logits instead of the golden `B` logits.
//!
//! Variants (both blocking, as production installs):
//! * `Default stream` (`active_stream == None`): the embedding goes to the NULL
//!   stream (Q8_0 embedding passes NULL explicitly, HFQ4G256/HFQ4G128 pass
//!   `active_stream`, which is `None`). It is held transitively by a *blocking*
//!   holder stream that carries the wait (`hip_bridge::Stream` has no NULL
//!   constructor, so the wait cannot be queued on the NULL stream directly).
//!   `memcpy_htod(pos_buf)` is also a NULL-stream operation; if it waits for
//!   the held embedding the forward blocks until the release and the logits
//!   equal golden. The test reports `waited_for_release`, the empirical answer
//!   to "is the position copy the barrier".
//! * `Installed blocking active stream` (what `ensure_capture_stream`
//!   installs): the embedding runs on the held blocking `active_stream`; the
//!   NULL-stream position copy must wait for it.
//!
//! The held phase releases the stream from a worker thread; every path
//! releases before joining, destroying a stream or freeing the signal
//! (including panics). Nothing is killed from inside the test: a wedged run
//! needs SIGTERM from outside.
//!
//! Run (gfx1201, card locked, check mode off, no replay env):
//!
//! ```bash
//! PM4_HOLD_MODEL=/abs/qwen3.x-27b.mq4 \
//! cargo test --release -p hipfire-arch-qwen35 --test pm4_staging_hold_hw \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Railgun lowering of the same step:
//!
//! ```bash
//! HIPFIRE_RAILGUN_SHADOW=<railgun-jit-corpus dir> HIPFIRE_RAILGUN_BACKEND=railgun \
//! PM4_HOLD_MODEL=... cargo test ... (as above)
//! ```
//!
//! `PM4_HOLD_CTX` (default 256) is the primed context length.

use hip_bridge::{DeviceBuffer, Stream, HIP_HOST_MALLOC_MAPPED};
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use rdna_compute::replay::{ReplayBackendRequest, ReplayController};
use rdna_compute::{Gpu, GpuTensor};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const WAIT_EQ: u32 = 0x1;
const WAIT_MASK: u32 = u32::MAX;
const RELEASED: u32 = 1;
/// Hold between the forward call and the release. A decode step's host side
/// (embedding launch, state-independent PM4 submit of one IB) takes
/// milliseconds; the hold is far longer.
const HOLD: Duration = Duration::from_millis(1500);
const KEEPALIVE_CAP: Duration = Duration::from_secs(60);
const KEEPALIVE_PERIOD: Duration = Duration::from_millis(2);
const SIGNAL_BYTES: usize = 4096;
/// Stale/new token pairs: step(stale) leaves its embedding in `scratch.x`
/// before step(new) is held. Distinct tokens give distinct logits.
const TOKEN_PAIRS: [(u32, u32); 3] = [(101, 202), (303, 404), (505, 606)];
const WARM_TOKEN: u32 = 101;

static LOCK: Mutex<()> = Mutex::new(());

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

// ---------------------------------------------------------------------------
// Host-mapped release word and worker (same contract as
// rdna-compute/tests/pm4_replay_staging_dependency.rs)
// ---------------------------------------------------------------------------

struct HostSignal {
    host: *mut c_void,
    buf: DeviceBuffer,
}

impl HostSignal {
    fn new(gpu: &Gpu) -> Self {
        let host = gpu
            .hip
            .host_malloc(SIGNAL_BYTES, HIP_HOST_MALLOC_MAPPED)
            .expect("HARNESS: hipHostMalloc(mapped) release word");
        let device = gpu
            .hip
            .host_get_device_pointer(host, 0)
            .expect("HARNESS: hipHostGetDevicePointer for the release word");
        assert!(
            !device.is_null(),
            "HARNESS: null device alias for the release word"
        );
        // SAFETY: device alias of the live hipHostMalloc allocation `host`,
        // freed once through `host` by `HostSignal::free`.
        let buf = unsafe { DeviceBuffer::from_host_mapped(device, SIGNAL_BYTES) };
        let signal = Self { host, buf };
        write_word(signal.addr(), 0);
        signal
    }

    fn addr(&self) -> usize {
        self.host as usize
    }

    fn free(self, gpu: &Gpu) {
        gpu.hip
            .host_free(self.host)
            .expect("HARNESS: hipHostFree of the release word");
    }
}

fn write_word(addr: usize, value: u32) {
    // SAFETY: `addr` is the 4-byte-aligned host pointer of a live SIGNAL_BYTES
    // hipHostMalloc allocation that outlives every `Releaser` over it.
    unsafe { (addr as *mut u32).write_volatile(value) };
    fence(Ordering::SeqCst);
}

struct Releaser {
    addr: usize,
    cancel: Option<mpsc::Sender<()>>,
    released: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    released_at: Arc<Mutex<Option<Instant>>>,
    worker: Option<JoinHandle<()>>,
    finished: bool,
}

impl Releaser {
    fn new(addr: usize) -> Self {
        Self {
            addr,
            cancel: None,
            released: Arc::new(AtomicBool::new(false)),
            done: Arc::new(AtomicBool::new(false)),
            released_at: Arc::new(Mutex::new(None)),
            worker: None,
            finished: false,
        }
    }

    /// Release after `hold` (or immediately on cancel). The worker stores
    /// `released` BEFORE the release word, so a call that returns after the
    /// stream really waited observes `released == true`.
    fn arm(&mut self, hold: Duration) {
        assert!(self.worker.is_none(), "HARNESS: release worker armed twice");
        let (cancel, wait) = mpsc::channel::<()>();
        self.cancel = Some(cancel);
        let addr = self.addr;
        let released = Arc::clone(&self.released);
        let done = Arc::clone(&self.done);
        let released_at = Arc::clone(&self.released_at);
        self.worker = Some(
            thread::Builder::new()
                .name("pm4-hold-release".to_owned())
                .spawn(move || {
                    let _ = wait.recv_timeout(hold);
                    *released_at
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
                    released.store(true, Ordering::SeqCst);
                    fence(Ordering::SeqCst);
                    let give_up = Instant::now() + KEEPALIVE_CAP;
                    loop {
                        write_word(addr, RELEASED);
                        if done.load(Ordering::SeqCst) || Instant::now() >= give_up {
                            break;
                        }
                        thread::sleep(KEEPALIVE_PERIOD);
                    }
                })
                .expect("HARNESS: spawn release worker"),
        );
    }

    fn release_now(&mut self) {
        if self.worker.is_some() {
            self.cancel = None;
        } else {
            write_word(self.addr, RELEASED);
        }
    }

    fn released(&self) -> bool {
        self.released.load(Ordering::SeqCst)
    }

    fn finish(&mut self) -> Option<Instant> {
        if !self.finished {
            self.cancel = None;
            match self.worker.take() {
                Some(worker) => {
                    self.done.store(true, Ordering::SeqCst);
                    worker.join().expect("HARNESS: release worker panicked");
                }
                None => write_word(self.addr, RELEASED),
            }
            self.finished = true;
        }
        *self
            .released_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Drop for Releaser {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.cancel = None;
        match self.worker.take() {
            Some(worker) => {
                self.done.store(true, Ordering::SeqCst);
                let _ = worker.join();
            }
            None => write_word(self.addr, RELEASED),
        }
    }
}

// ---------------------------------------------------------------------------
// Model state helpers (same restore contract as tests/pm_decode_twins_hw.rs)
// ---------------------------------------------------------------------------

struct State {
    tensors: Vec<Vec<u8>>,
}

fn mapped_len(gpu: &Gpu, t: &GpuTensor) -> usize {
    gpu.vmm_mapped_bytes(t)
        .map_or(t.buf.size(), |m| m.min(t.buf.size()))
}

fn state_tensors(slot: &ModelSlot) -> Vec<&GpuTensor> {
    let kv = &slot.kv_cache;
    let dn = &slot.dn_state;
    kv.k_gpu
        .iter()
        .chain(kv.v_gpu.iter())
        .chain(kv.k_scales.iter())
        .chain(kv.v_scales.iter())
        .chain(dn.s_matrices.iter())
        .chain(dn.s_scales.iter())
        .chain(dn.conv_states.iter())
        .chain(dn.s_ef_residual.iter())
        .collect()
}

fn snapshot(gpu: &Gpu, slot: &ModelSlot) -> State {
    gpu.hip.device_synchronize().expect("sync");
    let tensors = state_tensors(slot)
        .into_iter()
        .map(|t| {
            let mut b = vec![0u8; mapped_len(gpu, t)];
            gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("dtoh state");
            b
        })
        .collect();
    State { tensors }
}

fn restore(gpu: &Gpu, slot: &ModelSlot, s: &State) {
    for (t, b) in state_tensors(slot).into_iter().zip(&s.tensors) {
        gpu.hip.memcpy_htod(&t.buf, b).expect("htod state");
    }
    gpu.hip.device_synchronize().expect("sync");
}

fn logits(gpu: &Gpu, slot: &ModelSlot) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let t = &slot.scratch.logits;
    let mut b = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("dtoh logits");
    b
}

fn forward(gpu: &mut Gpu, slot: &mut ModelSlot, token: u32, pos: usize) -> Result<(), String> {
    let ModelSlot {
        weights,
        config,
        kv_cache,
        dn_state,
        scratch,
        ..
    } = slot;
    qwen35::forward_scratch(
        gpu, weights, config, token, pos, kv_cache, dn_state, scratch,
    )
    .map_err(|error| error.to_string())
}

// ---------------------------------------------------------------------------
// Bring-up through the production default route
// ---------------------------------------------------------------------------

struct Rig {
    _guard: MutexGuard<'static, ()>,
    gpu: Gpu,
    slot: ModelSlot,
    before: State,
    frame: u32,
    pos: usize,
    signal: HostSignal,
    embedding: String,
}

fn assert_env() {
    for (key, allowed) in [
        ("HIPFIRE_REPLAY_BACKEND", &["", "auto"][..]),
        ("HIPFIRE_REPLAY_MANUAL_CAPTURE", &["", "0"][..]),
        ("HIPFIRE_REPLAY_TRANSPORT", &["", "auto", "pm4"][..]),
        ("HIPFIRE_RAILGUN_CHECK", &["", "off"][..]),
        ("HIPFIRE_RAILGUN_DIGEST", &["", "0"][..]),
    ] {
        let value = std::env::var(key).unwrap_or_default();
        assert!(
            allowed.contains(&value.trim()),
            "{key}={value:?}: the production default route (and an unfenced step) needs it \
             unset or one of {allowed:?}"
        );
    }
}

fn submit_lowering() -> &'static str {
    let backend = std::env::var("HIPFIRE_RAILGUN_BACKEND").unwrap_or_default();
    let shadow = std::env::var("HIPFIRE_RAILGUN_SHADOW").unwrap_or_default();
    if backend == "railgun" && !shadow.is_empty() && shadow != "0" {
        "railgun (HIPFIRE_RAILGUN_BACKEND=railgun)"
    } else {
        "redline planner"
    }
}

fn bring_up() -> Option<Rig> {
    let guard = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_env();
    let model = PathBuf::from(std::env::var("PM4_HOLD_MODEL").unwrap_or_else(|_| {
        "/home/kaden/qcal/release-0.4.0/fp8-tp-ep-geo/models/qwen3.8-27b.mq4-xts".to_owned()
    }));
    let ctx: usize = std::env::var("PM4_HOLD_CTX")
        .ok()
        .map_or(256, |v| v.parse().expect("PM4_HOLD_CTX"));

    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: no GPU ({error})");
            return None;
        }
    };
    if gpu.arch != "gfx1201" {
        eprintln!(
            "skip: the dense retained PM4 default is gfx1201-only, found {}",
            gpu.arch
        );
        return None;
    }
    assert!(
        model.is_file(),
        "model not found: {} (set PM4_HOLD_MODEL)",
        model.display()
    );

    let mut cfg = ModelSlotConfig::default();
    // Pinned 27B artifact runs at max_seq 512; keep at least that.
    cfg.max_seq = (ctx + 64).max(512);
    let mut slot = ModelSlot::load(&mut gpu, &model, "pm4-hold", cfg).expect("load model");
    let embedding = format!("{:?}", slot.weights.embd_format);

    // The daemon's post-load default for Qwen3.5 dense on gfx1201.
    assert!(
        gpu.replay.configure_model_default(true),
        "configure_model_default refused: an explicit replay backend/capture setting is in effect"
    );
    gpu.replay
        .set_pm4_gfx12_dispatch_pacing(rdna_compute::replay::gfx1201_pm4_pacing_from_config());
    assert_eq!(
        gpu.replay.transport_name(),
        "pm4",
        "the default dense route must be PM4"
    );

    let prompt: Vec<u32> = (0..ctx as u32).map(|i| 10 + (i % 1000)).collect();
    {
        let ModelSlot {
            weights,
            config,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = &mut slot;
        qwen35::forward_prefill_batch(
            &mut gpu, weights, config, &prompt, 0, kv_cache, dn_state, scratch, None, None, None,
            None,
        )
        .expect("prefill");
    }
    // One warm decode that must NOT record (an ineligible call), so lazy
    // allocations and first-use kernel loads stay out of the recorded step.
    gpu.graphs.ar_graph_eligible = false;
    forward(&mut gpu, &mut slot, WARM_TOKEN, ctx).expect("warm decode");
    assert!(
        !gpu.replay.is_recording() && !gpu.replay.prepared_pm4_plan_ready(),
        "HARNESS: the warm decode must not have recorded or prepared a tape"
    );
    gpu.hip.device_synchronize().expect("sync warm");

    let pos = ctx + 1;
    let before = snapshot(&gpu, &slot);
    let frame = rdna_compute::norm::gdn_requant_frame_checkpoint();

    // The first eligible decode records on HIP and prepares the PM4 tape.
    forward(&mut gpu, &mut slot, TOKEN_PAIRS[0].1, pos).expect("recording decode");
    gpu.hip.device_synchronize().expect("sync record");
    assert!(
        gpu.replay.prepared_pm4_plan_ready() && gpu.replay.should_route_pm4(),
        "HARNESS: the default lifecycle did not prepare a PM4 route ({:?})",
        gpu.replay.fallback_reason()
    );
    eprintln!(
        "PM4 default route prepared on {}; embedding={embedding}; submit lowering: {}",
        gpu.arch,
        submit_lowering()
    );
    let signal = HostSignal::new(&gpu);
    Some(Rig {
        _guard: guard,
        gpu,
        slot,
        before,
        frame,
        pos,
        signal,
        embedding,
    })
}

impl Rig {
    /// Unheld PM4 decode: restore, then one ordinary decode through the
    /// prepared PM4 route; proves exactly one retained replay ran at `pos`.
    fn step(&mut self, token: u32) -> Vec<u8> {
        restore(&self.gpu, &self.slot, &self.before);
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(self.frame);
        assert!(
            self.gpu.replay.should_route_pm4(),
            "HARNESS: route left PM4 ({:?})",
            self.gpu.replay.fallback_reason()
        );
        let before = self.gpu.replay.replay_observation().count;
        forward(&mut self.gpu, &mut self.slot, token, self.pos).expect("unheld PM4 decode");
        let observation = self.gpu.replay.replay_observation();
        assert_eq!(
            (observation.count - before, observation.last_position),
            (1, Some(self.pos)),
            "HARNESS: the unheld decode did not run exactly one retained PM4 replay at the \
             decode position (HIP fallback?): {observation:?}"
        );
        logits(&self.gpu, &self.slot)
    }

    /// Ordinary HIP oracle: same restored state and GDN frame, retained route
    /// swapped out for a plain HIP controller (and no HIP graph), then
    /// restored. Leaves `scratch.x` = emb(token) like any decode.
    fn hip_step(&mut self, token: u32) -> Vec<u8> {
        restore(&self.gpu, &self.slot, &self.before);
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(self.frame);
        let retained = std::mem::replace(
            &mut self.gpu.replay,
            ReplayController::new(ReplayBackendRequest::Hip),
        );
        self.gpu.graphs.ar_graph_eligible = false;
        let result = forward(&mut self.gpu, &mut self.slot, token, self.pos);
        self.gpu.replay = retained;
        result.expect("HIP oracle decode");
        assert!(
            self.gpu.active_stream.is_none(),
            "HARNESS: the HIP oracle installed an active stream"
        );
        logits(&self.gpu, &self.slot)
    }

    fn teardown(self) {
        let Rig {
            mut gpu,
            slot,
            signal,
            ..
        } = self;
        gpu.hip
            .device_synchronize()
            .expect("HARNESS: sync teardown");
        gpu.replay.shutdown().expect("retained replay quiescence");
        signal.free(&gpu);
        drop(slot);
    }
}

// ---------------------------------------------------------------------------
// Held runs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Hold {
    /// `active_stream == None`; the wait sits on a blocking holder stream the
    /// NULL-stream work is ordered behind.
    DefaultStream,
    /// A held blocking stream (what `ensure_capture_stream` installs) as
    /// `gpu.active_stream`: HFQ4 embeddings launch on it, the NULL-stream
    /// position copy waits for it.
    InstalledBlocking,
}

/// What a held `forward_scratch` did.
struct Held {
    hold: Hold,
    forward: Result<(), String>,
    /// The worker had released the stream when `forward_scratch` returned,
    /// i.e. the call waited for the held producer (the position copy is the
    /// only host-synchronous step before the PM4 submit).
    waited_for_release: bool,
    returned_after: Duration,
    /// Retained PM4 replays observed by `gpu.replay` between the call and its
    /// return (must be exactly 1) and the position of the last one.
    pm4_replays: u64,
    pm4_position: Option<usize>,
    logits: Vec<u8>,
}

fn open_hold(rig: &mut Rig, hold: Hold) -> (Releaser, Option<Stream>) {
    let holder = match hold {
        Hold::DefaultStream | Hold::InstalledBlocking => rig.gpu.hip.stream_create(),
    }
    .expect("HARNESS: create holder stream");
    let releaser = Releaser::new(rig.signal.addr());
    rig.gpu
        .hip
        .stream_wait_value32(&holder, &rig.signal.buf, RELEASED, WAIT_EQ, WAIT_MASK)
        .expect("HARNESS: queue hipStreamWaitValue32 on the holder stream");
    match hold {
        Hold::DefaultStream => {
            assert!(
                rig.gpu.active_stream.is_none(),
                "HARNESS: active stream installed"
            );
            (releaser, Some(holder))
        }
        Hold::InstalledBlocking => {
            assert!(
                rig.gpu.active_stream.is_none(),
                "HARNESS: active stream installed"
            );
            rig.gpu.active_stream = Some(holder);
            (releaser, None)
        }
    }
}

/// Release, drain and destroy the holder; returns after the worker joined.
fn close_hold(rig: &mut Rig, mut releaser: Releaser, held_default: Option<Stream>) {
    releaser.release_now();
    let holder = match held_default {
        Some(holder) => holder,
        None => rig
            .gpu
            .active_stream
            .take()
            .expect("installed holder stream"),
    };
    rig.gpu
        .hip
        .stream_synchronize(&holder)
        .expect("HARNESS: drain the holder after release");
    releaser.finish();
    rig.gpu
        .hip
        .device_synchronize()
        .expect("HARNESS: sync after hold");
    rig.gpu
        .hip
        .stream_destroy(holder)
        .expect("HARNESS: destroy holder");
}

/// Restore, hold the staging stream, run the real decode, release, drain.
fn run_held(rig: &mut Rig, hold: Hold, token: u32) -> Held {
    restore(&rig.gpu, &rig.slot, &rig.before);
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(rig.frame);
    assert!(rig.gpu.replay.should_route_pm4(), "HARNESS: route left PM4");
    write_word(rig.signal.addr(), 0);

    let (mut releaser, default_holder) = open_hold(rig, hold);
    let before_replays = rig.gpu.replay.replay_observation().count;
    let started = Instant::now();
    releaser.arm(HOLD);
    let (gpu, slot, pos) = (&mut rig.gpu, &mut rig.slot, rig.pos);
    let forward = forward(gpu, slot, token, pos);
    let waited_for_release = releaser.released();
    let returned_after = started.elapsed();
    // The PM4 replay blocks the host until the tape completes, so this is the
    // retained submission's own observation, taken before the release (no
    // host readback has happened yet).
    let observation = rig.gpu.replay.replay_observation();
    let pm4_replays = observation.count - before_replays;
    let pm4_position = observation.last_position;
    close_hold(rig, releaser, default_holder);

    Held {
        hold,
        forward,
        waited_for_release,
        returned_after,
        pm4_replays,
        pm4_position,
        logits: logits(&rig.gpu, &rig.slot),
    }
}

/// Direct probe of the synchronous position copy: does `hipMemcpy` H2D on the
/// NULL stream wait for a held producer? (Writes the same position the
/// following steps use, so it is harmless.)
fn probe_pos_copy(rig: &mut Rig, hold: Hold) -> (bool, Duration) {
    let (mut releaser, default_holder) = open_hold(rig, hold);
    let started = Instant::now();
    releaser.arm(HOLD);
    rig.gpu
        .hip
        .memcpy_htod(&rig.slot.scratch.pos_buf, &(rig.pos as i32).to_ne_bytes())
        .expect("position copy");
    let waited = releaser.released();
    let elapsed = started.elapsed();
    close_hold(rig, releaser, default_holder);
    (waited, elapsed)
}

fn first_difference(a: &[u8], b: &[u8]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(index) => format!("first differing logits byte {index} (of {})", a.len()),
        None => "identical".to_owned(),
    }
}

/// Run the pairs for `hold`, compare against the unheld golden, tear down,
/// then judge.
fn held_oracle(hold: Hold) {
    let Some(mut rig) = bring_up() else { return };

    let (waited, elapsed) = probe_pos_copy(&mut rig, hold);
    eprintln!(
        "EMPIRICAL position copy [{hold:?}]: waited_for_release={waited} after {:.1}ms \
         (hold {:.0}ms)",
        ms(elapsed),
        ms(HOLD)
    );

    let mut runs = Vec::new();
    for (stale, new) in TOKEN_PAIRS {
        // Ordinary HIP oracle (same state, same GDN frame, no retained route),
        // then the same step through the prepared PM4 route, unheld.
        let oracle_new = rig.hip_step(new);
        let pm4_new = rig.step(new);
        // `step(stale)` leaves emb(stale) in scratch.x; the held decode of
        // `new` starts from there.
        let pm4_stale = rig.step(stale);
        let held = run_held(&mut rig, hold, new);
        runs.push((stale, new, oracle_new, pm4_new, pm4_stale, held));
    }
    let rig_pos = rig.pos;
    let embedding = rig.embedding.clone();
    rig.teardown();

    for (stale, new, golden_new, pm4_new, golden_stale, held) in &runs {
        assert!(
            pm4_new == golden_new,
            "HARNESS: the unheld PM4 decode of token {new} is not byte-identical to the ordinary \
             HIP oracle ({}); the route is not the certified one, so a held comparison would be \
             meaningless",
            first_difference(pm4_new, golden_new)
        );
        assert_eq!(
            held.pm4_replays, 1,
            "HARNESS: the held decode of token {new} did not perform exactly one PM4 replay \
             before the release (replay observation delta {}): the retained PM4 transport was not \
             what ran",
            held.pm4_replays
        );
        assert_eq!(
            held.pm4_position,
            Some(rig_pos),
            "HARNESS: the held decode's retained replay was not at the decode position"
        );
        assert!(
            golden_new != golden_stale,
            "HARNESS: tokens {stale} and {new} produce identical logits; the stale input would be \
             invisible"
        );
        if let Err(reason) = &held.forward {
            panic!(
                "held real decode failed ({:?}, tokens {stale}->{new}): {reason}",
                held.hold
            );
        }
        eprintln!(
            "EMPIRICAL held decode [{:?}, embedding {embedding}, tokens {stale}->{new}]: \
             waited_for_release={} returned_after={:.1}ms logits_match_golden={} \
             logits_match_stale_input={}",
            held.hold,
            held.waited_for_release,
            ms(held.returned_after),
            held.logits == *golden_new,
            held.logits == *golden_stale,
        );
        assert!(
            held.logits == *golden_new,
            "PM4/HIP staging race in the real decode ({:?}, embedding {embedding}, tokens \
             {stale}->{new}): forward_scratch returned (waited_for_release={}, \
             returned_after={:.1}ms of a {:.0}ms hold) and the PM4 replay consumed {}; \
             {} against the golden new-token logits",
            held.hold,
            held.waited_for_release,
            ms(held.returned_after),
            ms(HOLD),
            if held.logits == *golden_stale {
                "the STALE token's embedding (logits equal the stale-token logits)"
            } else {
                "inputs that match neither token"
            },
            first_difference(&held.logits, golden_new),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Production-shaped regression: the real default-stream decode with the
/// embedding held behind a blocking stream must still produce the golden
/// logits. Passes when the synchronous position copy is a real barrier for
/// NULL-stream staging (and passes after any boundary fence).
#[test]
#[ignore = "requires gfx1201 + a Qwen3.5 dense model (PM4_HOLD_MODEL)"]
fn held_default_stream_embedding_real_decode_matches_unheld() {
    held_oracle(Hold::DefaultStream);
}

/// Production-shaped regression: the installed *blocking* stream
/// (`ensure_capture_stream` installs `stream_create`, flags 0) carries the
/// embedding (HFQ4G256/HFQ4G128 launch on `active_stream`; Q8_0 on NULL,
/// ordered behind it), held; the NULL-stream position copy must wait for it.
#[test]
#[ignore = "requires gfx1201 + a Qwen3.5 dense model (PM4_HOLD_MODEL)"]
fn held_installed_blocking_stream_embedding_real_decode_matches_unheld() {
    held_oracle(Hold::InstalledBlocking);
}
