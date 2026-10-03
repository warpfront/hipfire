//! railgun M2 check mode: the HIP twin (design §2.4, §5 G2).
//!
//! `HIPFIRE_RAILGUN_CHECK = off | sample(N) | always` (the `railgun.check`
//! knob; serve default `off`). On a checked step of the prepared PM4 route:
//!
//! 1. snapshot every comparison surface (device→device);
//! 2. run the PM4 lowering; read every surface back;
//! 3. restore the snapshot; run the **HIP twin** — the same program's nodes
//!    launched through HIP (`hipModuleLaunchKernel`, the objects HIP loaded)
//!    with the exact kernarg bytes the PM4 lowering submitted for this
//!    position; read every surface back;
//! 4. byte-compare. Any difference poisons the retained route (the forward
//!    keeps the twin's result, which is already in place) and is reported.
//!
//! The snapshot and the restore are each fenced with `hipDeviceSynchronize`:
//! `hipMemcpy` device→device returns before the copy completes, and the PM4
//! lowering runs on its own queue, outside HIP's stream order. Unfenced, the
//! PM4 run raced the pre-image copies and the first checked step differed on
//! 249 of 392 surfaces under Redline's lowering as well as railgun's.
//!
//! Surfaces. `always` compares **every live `State` allocation of the
//! allocator registry** (`hip_bridge::registry`: every hipMalloc and every
//! mapped VMM chunk, not the effect list). `Weights` are immutable during
//! decode and outside the surface set: `always` digests them before and after
//! the first checked step of each prepared program (and against the first
//! program's digest), not per step.
//! `sample(N)` checks one step in N and compares only the allocations the
//! prepared railgun program writes (allocation-wide, M1 facts), refusing a
//! program with any non-PM4 node — weaker by design.
//!
//! `HIPFIRE_RAILGUN_NEGATIVE_CONTROL=flip_byte` (developer) XORs one byte of
//! one surface into the PM4 arm's result before read-back; the checked step
//! must come back unequal on exactly that byte (`negative_control.caught`).
//!
//! `HIPFIRE_RAILGUN_DIGEST=1` writes, after every replayed step, a digest of
//! every live `State` allocation paired by allocation order: the G2 third
//! arm compares these lines between a railgun process and the pre-railgun
//! default process run on identical inputs. `HIPFIRE_RAILGUN_CHECK_OUT`
//! receives one JSON line per replayed step.
use std::collections::HashMap;
use std::io::Write as _;
use std::time::Instant;

use hip_bridge::registry::{self, AllocationRole, LiveAllocation};
use hip_bridge::DeviceBuffer;
use serde_json::json;

use crate::Gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckMode {
    Off,
    Sample(u64),
    Always,
}

impl CheckMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        let v = value.trim();
        if v.is_empty() || v == "off" {
            return Ok(Self::Off);
        }
        if v == "always" {
            return Ok(Self::Always);
        }
        if let Some(n) = v.strip_prefix("sample(").and_then(|r| r.strip_suffix(')')) {
            let n: u64 = n.trim().parse().map_err(|_| format!("railgun.check: bad sample period in {v:?}"))?;
            if n == 0 {
                return Err("railgun.check: sample(0)".into());
            }
            return Ok(Self::Sample(n));
        }
        Err(format!("railgun.check: expected off | sample(N) | always, got {v:?}"))
    }

    fn name(self) -> String {
        match self {
            Self::Off => "off".into(),
            Self::Sample(n) => format!("sample({n})"),
            Self::Always => "always".into(),
        }
    }
}

pub struct CheckState {
    mode: CheckMode,
    digest: bool,
    flip_byte: bool,
    out: Option<std::fs::File>,
    step: u64,
    checked: u64,
    mismatched_steps: u64,
    snapshots: HashMap<(usize, usize), DeviceBuffer>,
    after_pm4: Vec<Vec<u8>>,
    after_twin: Vec<Vec<u8>>,
    weights_baseline: Option<Vec<(u64, usize, u64)>>,
    /// Prepared program whose first checked step verified the weights.
    weights_program: Option<u64>,
}

impl CheckState {
    pub fn from_config() -> Option<Self> {
        let var = |name: &str| hipfire_config::developer_var(name).ok().filter(|v| !v.is_empty());
        let mode = match var("HIPFIRE_RAILGUN_CHECK").map(|v| CheckMode::parse(&v)) {
            None => CheckMode::Off,
            Some(Ok(mode)) => mode,
            Some(Err(e)) => panic!("{e}"),
        };
        let digest = var("HIPFIRE_RAILGUN_DIGEST").as_deref() == Some("1");
        if mode == CheckMode::Off && !digest {
            return None;
        }
        let flip_byte = mode != CheckMode::Off && var("HIPFIRE_RAILGUN_NEGATIVE_CONTROL").as_deref() == Some("flip_byte");
        let out = var("HIPFIRE_RAILGUN_CHECK_OUT").map(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| panic!("HIPFIRE_RAILGUN_CHECK_OUT {path}: {e}"))
        });
        eprintln!("[railgun] check mode {} digest={digest}", mode.name());
        if flip_byte {
            eprintln!("[railgun] NEGATIVE CONTROL: one byte of the PM4 arm's result flipped at every checked step");
        }
        Some(Self {
            mode,
            digest,
            flip_byte,
            out,
            step: 0,
            checked: 0,
            mismatched_steps: 0,
            snapshots: HashMap::new(),
            after_pm4: Vec::new(),
            after_twin: Vec::new(),
            weights_baseline: None,
            weights_program: None,
        })
    }

    fn emit(&mut self, line: serde_json::Value) {
        if let Some(out) = self.out.as_mut() {
            let _ = writeln!(out, "{line}");
        }
    }

    /// One replayed step. `Err` leaves the route to the caller's poison path.
    unsafe fn step(&mut self, gpu: &mut Gpu, pos: usize) -> Result<(), String> {
        let index = self.step;
        self.step += 1;
        let due = match self.mode {
            CheckMode::Off => false,
            CheckMode::Always => true,
            CheckMode::Sample(n) => index % n == 0,
        };
        if !due {
            let started = Instant::now();
            // SAFETY: forwarded from the model owner.
            unsafe { gpu.replay.replay_pm4(pos) }?;
            let pm4_ms = ms(started);
            let digest = if self.digest { Some(self.digest_surfaces(gpu)?) } else { None };
            self.emit(json!({"step": index, "pos": pos, "checked": false, "ms": {"pm4": pm4_ms}, "digest": digest}));
            return Ok(());
        }
        // SAFETY: forwarded from the model owner.
        unsafe { self.checked_step(gpu, index, pos) }
    }

    unsafe fn checked_step(&mut self, gpu: &mut Gpu, index: u64, pos: usize) -> Result<(), String> {
        let t_start = Instant::now();
        // The forward's HIP work before the replay (embedding, pos word) may
        // be in flight: settle it before the pre-image is taken.
        sync(gpu)?;
        let surfaces = self.surfaces(gpu)?;
        let total_bytes: usize = surfaces.iter().map(|s| s.bytes).sum();
        let program = gpu.replay.prepared_pm4_generation();
        let weights_before = if self.mode == CheckMode::Always && self.weights_program != Some(program) {
            Some(weights_digest(gpu)?)
        } else {
            None
        };
        let weights_before_ms = ms(t_start);

        // 1. pre-image
        let t = Instant::now();
        {
            let _internal = registry::role_scope(AllocationRole::Internal);
            let live: std::collections::HashSet<(usize, usize)> = surfaces.iter().map(|s| (s.base, s.bytes)).collect();
            let stale: Vec<_> = self.snapshots.keys().copied().filter(|k| !live.contains(k)).collect();
            for key in stale {
                if let Some(buf) = self.snapshots.remove(&key) {
                    let _ = gpu.hip.free(buf);
                }
            }
            for s in &surfaces {
                if !self.snapshots.contains_key(&(s.base, s.bytes)) {
                    let buf = gpu.hip.malloc(s.bytes).map_err(|e| format!("check snapshot {} bytes: {e}", s.bytes))?;
                    self.snapshots.insert((s.base, s.bytes), buf);
                }
            }
        }
        for s in &surfaces {
            let snap = &self.snapshots[&(s.base, s.bytes)];
            // SAFETY: registry entries are live device allocations.
            let live = unsafe { DeviceBuffer::from_raw(s.base as *mut _, s.bytes) };
            gpu.hip.memcpy_dtod(snap, &live, s.bytes).map_err(|e| format!("check snapshot: {e}"))?;
        }
        sync(gpu)?;
        let snapshot_ms = ms(t);

        // 2. PM4 lowering
        let t = Instant::now();
        // SAFETY: forwarded from the model owner.
        unsafe { gpu.replay.replay_pm4(pos) }?;
        let pm4_ms = ms(t);
        let flipped = if self.flip_byte { Some(flip_one_byte(gpu, &surfaces)?) } else { None };
        let t = Instant::now();
        read_back(gpu, &surfaces, &mut self.after_pm4)?;
        let readback_pm4_ms = ms(t);

        // 3. restore, twin
        let t = Instant::now();
        for s in &surfaces {
            let snap = &self.snapshots[&(s.base, s.bytes)];
            // SAFETY: as above.
            let live = unsafe { DeviceBuffer::from_raw(s.base as *mut _, s.bytes) };
            gpu.hip.memcpy_dtod(&live, snap, s.bytes).map_err(|e| format!("check restore: {e}"))?;
        }
        sync(gpu)?;
        let restore_ms = ms(t);
        let t = Instant::now();
        let twin_launches = unsafe { run_twin(gpu) }?;
        let twin_ms = ms(t);
        let t = Instant::now();
        read_back(gpu, &surfaces, &mut self.after_twin)?;
        let readback_twin_ms = ms(t);

        // 4. compare
        let t = Instant::now();
        let mut mismatches = Vec::new();
        let mut flip_seen = false;
        for (k, s) in surfaces.iter().enumerate() {
            let (a, b) = (&self.after_pm4[k], &self.after_twin[k]);
            if a != b {
                let first = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap_or(0);
                let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                flip_seen |= flipped == Some((s.seq, first)) && differing == 1;
                mismatches.push(json!({"seq": s.seq, "base": format!("{:#x}", s.base), "bytes": s.bytes,
                    "kind": format!("{:?}", s.kind), "first_diff": first, "differing_bytes": differing}));
            }
        }
        let compare_ms = ms(t);

        // 5. weights unchanged across the step, once per prepared program
        let t = Instant::now();
        let weights = match weights_before {
            None => None,
            Some(before) => {
                let after = weights_digest(gpu)?;
                let baseline = self.weights_baseline.get_or_insert_with(|| before.clone());
                if before != after || *baseline != after {
                    return Err("railgun check: a Weights allocation changed (or the weight set changed) across the checked step".into());
                }
                self.weights_program = Some(program);
                let bytes: usize = after.iter().map(|w| w.1).sum();
                Some(json!({"program": program, "allocations": after.len(), "bytes": bytes, "equal": true}))
            }
        };
        let weights_ms = if weights.is_some() { weights_before_ms + ms(t) } else { 0.0 };

        let digest = if self.digest { Some(digest_of(&surfaces, &self.after_twin)) } else { None };
        self.checked += 1;
        let equal = mismatches.is_empty();
        if !equal {
            self.mismatched_steps += 1;
        }
        let negative_control = flipped.map(|(seq, offset)| {
            json!({"kind": "flip_byte", "seq": seq, "offset": offset, "caught": flip_seen && mismatches.len() == 1})
        });
        let line = json!({
            "step": index, "pos": pos, "checked": true, "mode": self.mode.name(), "equal": equal,
            "surfaces": surfaces.len(), "bytes": total_bytes, "twin_launches": twin_launches,
            "mismatches": mismatches, "weights": weights, "negative_control": negative_control,
            "ms": {"weights": weights_ms, "snapshot": snapshot_ms, "pm4": pm4_ms, "readback_pm4": readback_pm4_ms,
                   "restore": restore_ms, "twin": twin_ms, "readback_twin": readback_twin_ms, "compare": compare_ms,
                   "total": ms(t_start)},
            "checked_steps": self.checked, "mismatched_steps": self.mismatched_steps, "digest": digest,
        });
        self.emit(line);
        if !equal {
            let reason = format!("railgun check: PM4 lowering and HIP twin differ at step {index} (pos {pos})");
            eprintln!("[railgun] {reason}; route poisoned, the twin's result stands");
            gpu.replay.poison(reason);
        }
        Ok(())
    }

    fn surfaces(&self, gpu: &Gpu) -> Result<Vec<LiveAllocation>, String> {
        let state = registry::live_allocations_with_role(AllocationRole::State);
        match self.mode {
            CheckMode::Always | CheckMode::Off => Ok(state),
            CheckMode::Sample(_) => {
                let program = gpu
                    .replay
                    .railgun_program_surfaces()
                    .ok_or("railgun.check=sample needs the railgun program (HIPFIRE_RAILGUN_SHADOW)")?;
                if program.non_pm4_nodes != 0 {
                    return Err(format!("railgun.check=sample refused: {} non-PM4 nodes", program.non_pm4_nodes));
                }
                Ok(state
                    .into_iter()
                    .filter(|a| {
                        let (lo, hi) = (a.base as u64, (a.base + a.bytes) as u64);
                        program.written.iter().any(|w| w.base < hi && lo < w.base + w.bytes)
                    })
                    .collect())
            }
        }
    }

    fn digest_surfaces(&mut self, gpu: &Gpu) -> Result<serde_json::Value, String> {
        let surfaces = registry::live_allocations_with_role(AllocationRole::State);
        read_back(gpu, &surfaces, &mut self.after_twin)?;
        Ok(digest_of(&surfaces, &self.after_twin))
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn sync(gpu: &Gpu) -> Result<(), String> {
    gpu.hip.device_synchronize().map_err(|e| format!("check sync: {e}"))
}

fn read_back(gpu: &Gpu, surfaces: &[LiveAllocation], into: &mut Vec<Vec<u8>>) -> Result<(), String> {
    sync(gpu)?;
    into.resize_with(surfaces.len(), Vec::new);
    for (k, s) in surfaces.iter().enumerate() {
        into[k].resize(s.bytes, 0);
        // SAFETY: registry entries are live device allocations.
        let live = unsafe { DeviceBuffer::from_raw(s.base as *mut _, s.bytes) };
        gpu.hip.memcpy_dtoh(&mut into[k], &live).map_err(|e| format!("check read-back: {e}"))?;
    }
    Ok(())
}

/// `[seq, bytes, hash]` of every `Weights` allocation, in allocation order.
/// Allocations are read back in batches of up to 1 GiB and every batch is
/// hashed in 16 MiB slices across threads; an allocation's hash combines its
/// slice hashes. (Read-back per allocation with one core hashing took 3.3 s
/// for the 15 GB H2 weight set; the serial hash was 2.4 s of it.)
fn weights_digest(gpu: &Gpu) -> Result<Vec<(u64, usize, u64)>, String> {
    const BATCH: usize = 1 << 30;
    const SLICE: usize = 16 << 20;
    let weights = registry::live_allocations_with_role(AllocationRole::Weights);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(32);
    let mut host = Vec::new();
    let mut digests = Vec::with_capacity(weights.len());
    let mut first = 0;
    while first < weights.len() {
        let mut end = first;
        let mut bytes = 0;
        while end < weights.len() && (end == first || bytes + weights[end].bytes <= BATCH) {
            bytes += weights[end].bytes;
            end += 1;
        }
        let batch = &weights[first..end];
        host.resize(bytes, 0);
        let mut at = 0;
        for w in batch {
            // SAFETY: registry entries are live device allocations.
            let live = unsafe { DeviceBuffer::from_raw(w.base as *mut _, w.bytes) };
            gpu.hip.memcpy_dtoh(&mut host[at..at + w.bytes], &live).map_err(|e| format!("weights read-back: {e}"))?;
            at += w.bytes;
        }
        // Slices never straddle two allocations; `owner[i]` is the batch
        // index of slice i's allocation.
        let mut slices = Vec::new();
        let mut owner = Vec::new();
        let mut at = 0;
        for (k, w) in batch.iter().enumerate() {
            for s in host[at..at + w.bytes].chunks(SLICE) {
                slices.push(s);
                owner.push(k);
            }
            at += w.bytes;
        }
        let mut hashes = vec![0u64; slices.len()];
        let per = slices.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (out, input) in hashes.chunks_mut(per).zip(slices.chunks(per)) {
                scope.spawn(move || {
                    for (h, s) in out.iter_mut().zip(input) {
                        *h = fast_hash(s);
                    }
                });
            }
        });
        for (k, w) in batch.iter().enumerate() {
            let combined: Vec<u8> =
                hashes.iter().zip(&owner).filter(|(_, o)| **o == k).flat_map(|(h, _)| h.to_le_bytes()).collect();
            digests.push((w.seq, w.bytes, fast_hash(&combined)));
        }
        first = end;
    }
    Ok(digests)
}

/// Negative control: XOR 0xff into the middle byte of the middle surface.
/// Returns `(seq, offset)`.
fn flip_one_byte(gpu: &Gpu, surfaces: &[LiveAllocation]) -> Result<(u64, usize), String> {
    let s = surfaces.get(surfaces.len() / 2).ok_or("negative control: no surfaces")?;
    let offset = s.bytes / 2;
    // SAFETY: one byte inside a live registry allocation.
    let cell = unsafe { DeviceBuffer::from_raw((s.base + offset) as *mut _, 1) };
    let mut byte = [0u8];
    gpu.hip.memcpy_dtoh(&mut byte, &cell).map_err(|e| format!("negative control read: {e}"))?;
    byte[0] ^= 0xff;
    gpu.hip.memcpy_htod(&cell, &byte).map_err(|e| format!("negative control write: {e}"))?;
    sync(gpu)?;
    eprintln!("[railgun] NEGATIVE CONTROL: flipped byte {offset} of surface seq {}", s.seq);
    Ok((s.seq, offset))
}

/// Per-surface `[seq, bytes, hash]` plus one combined hash.
fn digest_of(surfaces: &[LiveAllocation], bytes: &[Vec<u8>]) -> serde_json::Value {
    let rows: Vec<(u64, usize, u64)> = surfaces.iter().zip(bytes).map(|(s, b)| (s.seq, s.bytes, fast_hash(b))).collect();
    let mut all = Vec::with_capacity(rows.len() * 24);
    for (seq, n, h) in &rows {
        all.extend_from_slice(&seq.to_le_bytes());
        all.extend_from_slice(&(*n as u64).to_le_bytes());
        all.extend_from_slice(&h.to_le_bytes());
    }
    json!({"combined": format!("{:016x}", fast_hash(&all)), "surfaces": rows.iter().map(|(s, n, h)| json!([s, n, format!("{h:016x}")])).collect::<Vec<_>>()})
}

/// 64-bit multiply–xorshift hash over 8-byte words (not cryptographic: it
/// pairs surfaces across arms; byte equality is decided by `==`).
fn fast_hash(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (bytes.len() as u64).wrapping_mul(K);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in chunks.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// Launch the prepared program's nodes through HIP, in order, with the
/// kernarg bytes the PM4 lowering submitted (explicit prefix only; HIP
/// supplies its own hidden arguments). Returns the launch count.
unsafe fn run_twin(gpu: &mut Gpu) -> Result<usize, String> {
    let images = gpu.replay.prepared_pm4_submitted_kernargs().map_err(|e| format!("HIP twin refused: {e}"))?;
    let launches: Vec<(String, [u32; 3], [u32; 3], u32, Vec<u8>)> = gpu
        .replay
        .recorded_launches()
        .iter()
        .take(images.len())
        .map(|l| (l.kernel.clone(), l.grid, l.block, l.shared_mem, l.kernarg.clone()))
        .collect();
    if launches.len() != images.len() {
        return Err(format!("HIP twin: {} prepared dispatches, {} recorded launches", images.len(), launches.len()));
    }
    for (k, (name, grid, block, shared, recorded)) in launches.iter().enumerate() {
        let func = gpu.functions.get(name).ok_or_else(|| format!("HIP twin: no HIP function {name}"))?;
        // The recorded blob is the explicit segment padded to 16 bytes; the
        // loader segment may end before that padding (no hidden arguments).
        // Every byte the loader segment covers comes from the submitted image.
        let mut blob = recorded.clone();
        let covered = blob.len().min(images[k].len());
        blob[..covered].copy_from_slice(&images[k][..covered]);
        // SAFETY: the blob is the recorded launch's explicit kernarg segment
        // with this position's bindings applied; every pointer is a live
        // model allocation (the same contract as the PM4 replay).
        unsafe { gpu.hip.launch_kernel_blob(func, *grid, *block, *shared, gpu.active_stream.as_ref(), &mut blob) }
            .map_err(|e| format!("HIP twin launch {k} {name}: {e}"))?;
    }
    gpu.hip.device_synchronize().map_err(|e| format!("HIP twin sync: {e}"))?;
    Ok(launches.len())
}

impl Gpu {
    /// The forward's PM4 replay, through railgun check mode when it is on.
    ///
    /// # Safety
    ///
    /// Same contract as [`crate::replay::ReplayController::replay_pm4`].
    pub unsafe fn replay_pm4_routed(&mut self, pos: usize) -> Result<(), String> {
        let Some(mut state) = self.replay.railgun_check.take() else {
            // SAFETY: forwarded.
            return unsafe { self.replay.replay_pm4(pos) }.map(|_| ());
        };
        // SAFETY: forwarded.
        let result = unsafe { state.step(self, pos) };
        if let Err(reason) = &result {
            eprintln!("[railgun] check step at pos {pos} failed: {reason}");
            state.emit(json!({"pos": pos, "error": reason}));
        }
        self.replay.railgun_check = Some(state);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::CheckMode;

    #[test]
    fn check_mode_grammar() {
        assert_eq!(CheckMode::parse("off"), Ok(CheckMode::Off));
        assert_eq!(CheckMode::parse("always"), Ok(CheckMode::Always));
        assert_eq!(CheckMode::parse("sample(64)"), Ok(CheckMode::Sample(64)));
        assert!(CheckMode::parse("sample(0)").is_err());
        assert!(CheckMode::parse("sample(x)").is_err());
        assert!(CheckMode::parse("sometimes").is_err());
    }
}
