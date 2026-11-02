// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K0c probe for `hipfire_runtime::ep_row_gather` (Flash-Next expert-parallel
//! row all-gather) over `--ranks` GPUs.
//!
//! ```sh
//! cargo build --release -p hipfire-runtime --features lab --example ep_row_gather_probe
//! ep_row_gather_probe bytes  --ranks 4 [--stream real|null]
//! ep_row_gather_probe timing --ranks 4 --reps 200 --warmups 10 [--map balanced|skewed|recorded]
//! ```
//! Device selection is the caller's (fabric exports `HIPFIRE_DEVICES` /
//! `ROCR_VISIBLE_DEVICES`); rank r = logical device r.
//!
//! # `bytes`
//!
//! 10 slots x 10240 B rows, two parity buffers per rank, slot->expert maps
//! (balanced / one-owner-skewed / a recorded route / a map with id < 0 slots).
//! Each rank's OWNED rows hold rank/slot/round-distinct patterns (+0, -0, quiet
//! and signalling NaN payloads, denormals, infinities, random words); non-owned
//! rows are poisoned. Every round the producers are a real kernel on the rank's
//! launch stream (`copy_f32_buffer` staging -> parity buffer, which also
//! re-poisons the non-owned rows), then publish/gather, then a device-side
//! snapshot of the parity buffer into a per-round tensor. Rounds alternate
//! parity and are issued back to back with NO host sync, so write-after-read
//! hazards between rounds would corrupt a snapshot. All snapshots are
//! downloaded afterwards and memcmp'd against the expected owner bytes. Two
//! issue paths: single host thread (all publish, then all gather) and one host
//! thread per rank through `exchange`. Negative controls (a flipped sign bit in
//! an owner's source row after publish, and one rank skipping its gather) must
//! be DETECTED as mismatches.
//!
//! # `timing`
//!
//! The K0 EP4 decode schedule (10 slots x 10240 B; balanced map => 30 peer row
//! copies per round). Round = publish on every rank, gather on every rank (one
//! host thread), then a stream synchronize of every rank; timed with the host
//! monotonic clock. Output is JSONL: a `meta` line, one `timing` line per rep
//! and a `summary` line. Eager only: no graph capture/replay is involved.

use hipfire_runtime::ep_row_gather::{EpRowGather, EpRowGatherSpec, EpSlotOwners};
use rdna_compute::{Gpu, GpuTensor};
use std::time::{Duration, Instant};

type R<T> = Result<T, String>;

const SLOTS: usize = 10;
/// Expert-id bound of the modulo gather (the Flash-Next layer's expert count).
const N_EXPERTS: usize = 512;
const ROW_BYTES: usize = 10_240;
const BUF_BYTES: usize = SLOTS * ROW_BYTES;
const BYTES_ROUNDS: usize = 8;
const RECORDED_ROUTE: [i32; SLOTS] = [3, 17, 42, 101, 255, 256, 300, 411, 480, 511];
const MAX_PROBE_BUFFERS: usize = 3;
const WARM_RING_BYTES: usize = 64 << 20;

fn ctx<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> R<T> {
    r.map_err(|e| format!("{what}: {e}"))
}

// ───────────────────────────── patterns ─────────────────────────────

const SPECIAL: [u32; 16] = [
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x7fc0_0000, // quiet NaN
    0x7f80_0001, // signalling NaN
    0xffc1_2345, // negative quiet NaN with payload
    0xff80_0001, // negative signalling NaN
    0x7fa5_5a5a, // signalling NaN, wide payload
    0x0000_0001, // smallest denormal
    0x007f_ffff, // largest denormal
    0x8000_0001, // negative smallest denormal
    0x807f_ffff, // negative largest denormal
    0x7f7f_ffff, // max finite
    0xff7f_ffff, // -max finite
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x3f80_0000, // 1.0
];

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// One row of owner-distinct bytes: every third word is a special value.
fn fill_pattern(seed: u64, out: &mut [u8]) {
    for (i, w) in out.chunks_exact_mut(4).enumerate() {
        let v = if i % 3 == 0 {
            SPECIAL[(i / 3 + seed as usize) % SPECIAL.len()]
        } else {
            splitmix(seed.wrapping_mul(0x1_0000_0001).wrapping_add(i as u64)) as u32
        };
        w.copy_from_slice(&v.to_le_bytes());
    }
}

fn pattern_seed(round: usize, rank: usize, slot: usize) -> u64 {
    ((round as u64 * 16 + rank as u64) * 64 + slot as u64) + 1
}

fn owner_of(expert: i32, ranks: usize) -> Option<usize> {
    (expert >= 0).then(|| expert as usize % ranks)
}

/// Bytes rank `rank` stages into parity buffer in `round`: pattern on owned
/// rows, per-rank poison elsewhere (including negative-id slots).
fn staging_bytes(rank: usize, round: usize, experts: &[i32], ranks: usize) -> Vec<u8> {
    let mut out = vec![0u8; BUF_BYTES];
    for (slot, row) in out.chunks_exact_mut(ROW_BYTES).enumerate() {
        if owner_of(experts[slot], ranks) == Some(rank) {
            fill_pattern(pattern_seed(round, rank, slot), row);
        } else {
            row.fill(0xA5 ^ rank as u8);
        }
    }
    out
}

/// Bytes rank `rank` must hold after the gather of `round`.
fn expected_bytes(rank: usize, round: usize, experts: &[i32], ranks: usize) -> Vec<u8> {
    let mut out = staging_bytes(rank, round, experts, ranks);
    for (slot, row) in out.chunks_exact_mut(ROW_BYTES).enumerate() {
        if let Some(owner) = owner_of(experts[slot], ranks) {
            if owner != rank {
                fill_pattern(pattern_seed(round, owner, slot), row);
            }
        }
    }
    out
}

// ───────────────────────────── maps ─────────────────────────────

fn map_experts(name: &str, ranks: usize) -> R<[i32; SLOTS]> {
    let mut m = [0i32; SLOTS];
    match name {
        // slot j owned by rank j % ranks (K0 allgather schedule)
        "balanced" => (0..SLOTS).for_each(|j| m[j] = j as i32),
        // every slot owned by rank 1
        "skewed" => (0..SLOTS).for_each(|j| m[j] = (1 + ranks * j) as i32),
        "recorded" => m = RECORDED_ROUTE,
        // balanced with two negative-id slots (never written by the gather)
        "negative" => {
            (0..SLOTS).for_each(|j| m[j] = j as i32);
            m[2] = -1;
            m[7] = -5;
        }
        other => return Err(format!("unknown map {other:?}")),
    }
    Ok(m)
}

fn peer_copies(experts: &[i32], ranks: usize) -> usize {
    experts
        .iter()
        .filter_map(|&e| owner_of(e, ranks))
        .map(|_| ranks - 1)
        .sum()
}

// ───────────────────────────── rig ─────────────────────────────

struct Rig {
    n: usize,
    gpus: Vec<Gpu>,
    /// bufs[rank][buffer index]; `MAX_PROBE_BUFFERS` allocated, the first K registered.
    bufs: Vec<Vec<GpuTensor>>,
    real_stream: bool,
    warm: Vec<(GpuTensor, GpuTensor)>,
}

impl Rig {
    fn new(n: usize, real_stream: bool, warm_bytes: usize) -> R<Self> {
        let mut gpus = Vec::with_capacity(n);
        for r in 0..n {
            gpus.push(ctx(
                Gpu::init_with_device(r as i32),
                &format!("init rank {r}"),
            )?);
        }
        let mut bufs = Vec::with_capacity(n);
        let mut warm = Vec::new();
        for (r, gpu) in gpus.iter_mut().enumerate() {
            ctx(gpu.bind_thread(), "bind_thread")?;
            if real_stream {
                let stream = ctx(gpu.hip.stream_create(), "stream_create")?;
                gpu.active_stream = Some(stream);
            }
            let zeros = vec![0u8; BUF_BYTES];
            let mut set = Vec::new();
            for i in 0..MAX_PROBE_BUFFERS {
                set.push(ctx(
                    gpu.upload_raw(&zeros, &[BUF_BYTES]),
                    &format!("alloc buffer {i}"),
                )?);
            }
            bufs.push(set);
            if warm_bytes > 0 {
                let z = vec![0u8; warm_bytes];
                let a = ctx(
                    gpu.upload_raw(&z, &[warm_bytes]),
                    &format!("alloc warm src rank {r}"),
                )?;
                let b = ctx(
                    gpu.upload_raw(&z, &[warm_bytes]),
                    &format!("alloc warm dst rank {r}"),
                )?;
                warm.push((a, b));
            }
        }
        Ok(Rig {
            n,
            gpus,
            bufs,
            real_stream,
            warm,
        })
    }

    fn sync_rank(&self, r: usize) -> R<()> {
        let g = &self.gpus[r];
        ctx(g.bind_thread(), "bind_thread")?;
        match g.active_stream.as_ref() {
            Some(s) => ctx(
                g.hip.stream_synchronize(s),
                &format!("stream_synchronize rank {r}"),
            ),
            None => ctx(
                g.hip.device_synchronize(),
                &format!("device_synchronize rank {r}"),
            ),
        }
    }

    fn sync_all(&self) -> R<()> {
        (0..self.n).try_for_each(|r| self.sync_rank(r))
    }

    fn ids_tensor(&mut self, rank: usize, experts: &[i32]) -> R<GpuTensor> {
        let bytes: Vec<u8> = experts.iter().flat_map(|e| e.to_le_bytes()).collect();
        ctx(
            self.gpus[rank].upload_raw(&bytes, &[bytes.len()]),
            "upload slot_experts",
        )
    }

    fn build_transport(&mut self, k: usize) -> R<std::sync::Arc<EpRowGather>> {
        let mut b = EpRowGather::builder(EpRowGatherSpec {
            ranks: self.n,
            slots: SLOTS,
            row_bytes: ROW_BYTES,
        })
        .device_waits(true);
        for r in 0..self.n {
            let set: Vec<&GpuTensor> = self.bufs[r].iter().take(k).collect();
            ctx(
                b.register(r, &mut self.gpus[r], &set),
                &format!("register rank {r}"),
            )?;
        }
        let t = ctx(b.build(), "build")?;
        // After every peer-read allocation (parity buffers + warm ring) exists.
        for r in 0..self.n {
            ctx(
                t.enable_peers(r, &mut self.gpus[r]),
                &format!("enable_peers rank {r}"),
            )?;
        }
        Ok(t)
    }

    fn release_transport(&mut self, t: &EpRowGather) -> R<()> {
        self.sync_all()?;
        for r in 0..self.n {
            ctx(
                t.release(r, &mut self.gpus[r]),
                &format!("release rank {r}"),
            )?;
        }
        Ok(())
    }

    /// K0-style DPM warm: a 64 MB peer-copy ring (rank r -> r+1) for `secs`.
    fn warm_ring(&self, secs: u64) -> R<()> {
        if secs == 0 || self.warm.is_empty() {
            return Ok(());
        }
        let t = Instant::now();
        let mut rounds = 0u64;
        while t.elapsed() < Duration::from_secs(secs) {
            for r in 0..self.n {
                let d = (r + 1) % self.n;
                let g = &self.gpus[d];
                ctx(g.bind_thread(), "bind_thread")?;
                let stream = g
                    .active_stream
                    .as_ref()
                    .ok_or("warm ring needs real streams")?;
                ctx(
                    g.hip.memcpy_peer_async(
                        &self.warm[d].1.buf,
                        self.gpus[d].device_id,
                        &self.warm[r].0.buf,
                        self.gpus[r].device_id,
                        WARM_RING_BYTES.min(self.warm[r].0.buf.size()),
                        stream,
                    ),
                    "warm memcpy_peer_async",
                )?;
            }
            self.sync_all()?;
            rounds += 1;
        }
        eprintln!("ep_row_gather_probe: warm ring {WARM_RING_BYTES} B x{} ranks: {rounds} rounds in {secs}s", self.n);
        Ok(())
    }

    fn teardown(mut self) -> R<()> {
        self.sync_all()?;
        for r in 0..self.n {
            let g = &mut self.gpus[r];
            ctx(g.bind_thread(), "bind_thread")?;
            for t in std::mem::take(&mut self.bufs[r]) {
                ctx(g.hip.free(t.buf), "free buffer")?;
            }
            if let Some((a, b)) = self.warm.pop() {
                ctx(g.hip.free(a.buf), "free warm src")?;
                ctx(g.hip.free(b.buf), "free warm dst")?;
            }
            if let Some(stream) = g.active_stream.take() {
                ctx(g.hip.stream_destroy(stream), "stream_destroy")?;
            }
        }
        Ok(())
    }
}

// ───────────────────────────── bytes mode ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    /// Flip the sign bit of word 0 of the owner's row for `slot` after publish,
    /// before any gather reads it.
    FlipOwnerSource {
        slot: usize,
    },
    /// `rank` never gathers.
    SkipGather {
        rank: usize,
    },
}

struct CaseOut {
    mismatched_bufs: usize,
    compared_bytes: usize,
    first: Option<String>,
}

/// One rank's exclusive view for its worker thread. `Gpu` and device handles are
/// `!Send`/`!Sync` by contract (thread-affine scratch), so the probe hands each
/// worker raw pointers instead.
///
/// SAFETY (`Send`): during the thread scope exactly one worker dereferences a
/// given job; the spawning thread touches none of the pointees until the scope
/// joins; every worker re-binds its device (`bind_thread`) inside each call.
struct RankJob {
    gpu: *mut Gpu,
    bufs: *const Vec<GpuTensor>,
    data: *const RankData,
}
unsafe impl Send for RankJob {}

struct RankData {
    staging: Vec<GpuTensor>,
    snaps: Vec<GpuTensor>,
    ids: GpuTensor,
}

fn first_diff(got: &[u8], want: &[u8]) -> Option<usize> {
    got.iter().zip(want).position(|(a, b)| a != b)
}

fn run_case(
    rig: &mut Rig,
    experts: &[i32; SLOTS],
    threaded: bool,
    fault: Fault,
    rounds: usize,
    k: usize,
) -> R<CaseOut> {
    let n = rig.n;
    let words = BUF_BYTES / 4;
    let mut data: Vec<RankData> = Vec::with_capacity(n);
    for r in 0..n {
        let ids = rig.ids_tensor(r, experts)?;
        let mut staging = Vec::new();
        let mut snaps = Vec::new();
        for round in 0..rounds {
            let host = staging_bytes(r, round, experts, n);
            staging.push(ctx(
                rig.gpus[r].upload_raw(&host, &[BUF_BYTES]),
                "upload staging",
            )?);
            snaps.push(ctx(
                rig.gpus[r].upload_raw(&vec![0xEEu8; BUF_BYTES], &[BUF_BYTES]),
                "alloc snapshot",
            )?);
        }
        data.push(RankData {
            staging,
            snaps,
            ids,
        });
    }
    rig.sync_all()?;
    let transport = rig.build_transport(k)?;

    let issue_err: R<()> = if threaded {
        let transport = &transport;
        let jobs: Vec<RankJob> = rig
            .gpus
            .iter_mut()
            .zip(rig.bufs.iter())
            .zip(data.iter())
            .map(|((gpu, bufs), d)| RankJob { gpu, bufs, data: d })
            .collect();
        let results: Vec<R<()>> = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .into_iter()
                .enumerate()
                .map(|(rank, job)| {
                    scope.spawn(move || -> R<()> {
                        let job = job;
                        // SAFETY: see `RankJob`.
                        let (gpu, bufs, d) = unsafe { (&mut *job.gpu, &*job.bufs, &*job.data) };
                        let body = (|| -> R<()> {
                            for round in 0..rounds {
                                let parity = round % k;
                                ctx(
                                    gpu.copy_f32_buffer(&bufs[parity], &d.staging[round], words),
                                    "producer",
                                )?;
                                ctx(
                                    transport.exchange(
                                        gpu,
                                        rank,
                                        parity,
                                        EpSlotOwners::ExpertModulo(&d.ids, N_EXPERTS),
                                        SLOTS,
                                    ),
                                    "exchange",
                                )?;
                                ctx(
                                    gpu.copy_f32_buffer(&d.snaps[round], &bufs[parity], words),
                                    "snapshot",
                                )?;
                            }
                            Ok(())
                        })();
                        if body.is_err() {
                            transport.abort();
                        }
                        body
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("rank thread panicked".to_string()))
                })
                .collect()
        });
        results.into_iter().collect()
    } else {
        (|| -> R<()> {
            for round in 0..rounds {
                let parity = round % k;
                for r in 0..n {
                    ctx(
                        rig.gpus[r].copy_f32_buffer(
                            &rig.bufs[r][parity],
                            &data[r].staging[round],
                            words,
                        ),
                        "producer",
                    )?;
                }
                for r in 0..n {
                    ctx(transport.publish(&mut rig.gpus[r], r, parity), "publish")?;
                }
                if let Fault::FlipOwnerSource { slot } = fault {
                    let owner = owner_of(experts[slot], n).ok_or("fault slot has no owner")?;
                    rig.sync_all()?;
                    let g = &rig.gpus[owner];
                    ctx(g.bind_thread(), "bind_thread")?;
                    let view = rig.bufs[owner][parity].buf.byte_view(slot * ROW_BYTES, 4);
                    let mut word = [0u8; 4];
                    ctx(g.hip.memcpy_dtoh(&mut word, &view), "fault dtoh")?;
                    word[3] ^= 0x80;
                    ctx(g.hip.memcpy_htod(&view, &word), "fault htod")?;
                }
                for r in 0..n {
                    if fault == (Fault::SkipGather { rank: r }) {
                        continue;
                    }
                    ctx(
                        transport.gather(
                            &mut rig.gpus[r],
                            r,
                            parity,
                            EpSlotOwners::ExpertModulo(&data[r].ids, N_EXPERTS),
                            SLOTS,
                        ),
                        "gather",
                    )?;
                }
                for r in 0..n {
                    ctx(
                        rig.gpus[r].copy_f32_buffer(
                            &data[r].snaps[round],
                            &rig.bufs[r][parity],
                            words,
                        ),
                        "snapshot",
                    )?;
                }
            }
            Ok(())
        })()
    };
    let sync_err = rig.sync_all();
    let rel = rig.release_transport(&transport);
    issue_err?;
    sync_err?;
    rel?;

    let mut out = CaseOut {
        mismatched_bufs: 0,
        compared_bytes: 0,
        first: None,
    };
    for r in 0..n {
        for round in 0..rounds {
            let got = ctx(
                rig.gpus[r].download_raw_bytes(&data[r].snaps[round]),
                "download snapshot",
            )?;
            let want = expected_bytes(r, round, experts, n);
            out.compared_bytes += want.len();
            if got != want {
                out.mismatched_bufs += 1;
                if out.first.is_none() {
                    let at = first_diff(&got, &want).unwrap_or(0);
                    out.first = Some(format!(
                        "rank {r} round {round} buffer {} slot {} byte {at}: got 0x{:02x} want 0x{:02x}",
                        round % k,
                        at / ROW_BYTES,
                        got[at],
                        want[at]
                    ));
                }
            }
        }
    }
    for (r, d) in data.into_iter().enumerate() {
        let g = &mut rig.gpus[r];
        ctx(g.bind_thread(), "bind_thread")?;
        for t in d
            .staging
            .into_iter()
            .chain(d.snaps)
            .chain(std::iter::once(d.ids))
        {
            ctx(g.hip.free(t.buf), "free case tensor")?;
        }
    }
    Ok(out)
}

fn bytes_mode(n: usize, real_stream: bool) -> R<bool> {
    let mut rig = Rig::new(n, real_stream, 0)?;
    let stream = if real_stream { "real" } else { "null" };
    let mut all_ok = true;
    let mut report = |ok: bool, line: String| {
        println!("{} {line}", if ok { "PASS" } else { "FAIL" });
        all_ok &= ok;
    };
    for map in ["balanced", "skewed", "recorded", "negative"] {
        let experts = map_experts(map, n)?;
        for threaded in [false, true] {
            let path = if threaded {
                "exchange-threads"
            } else {
                "publish-gather-1thread"
            };
            let o = run_case(&mut rig, &experts, threaded, Fault::None, BYTES_ROUNDS, 2)?;
            report(
                o.mismatched_bufs == 0,
                format!(
                    "bytes ranks={n} map={map} path={path} stream={stream} rounds={BYTES_ROUNDS} buffers=2(alternating) \
                     peer_copies/round={} compared={}B mismatched_snapshots={}{}",
                    peer_copies(&experts, n),
                    o.compared_bytes,
                    o.mismatched_bufs,
                    o.first.map(|f| format!(" first: {f}")).unwrap_or_default()
                ),
            );
        }
    }
    // Three registered buffers cycled 0,1,2 (trunk parity pair + a dedicated one).
    for map in ["balanced", "recorded"] {
        let experts = map_experts(map, n)?;
        for threaded in [false, true] {
            let path = if threaded {
                "exchange-threads"
            } else {
                "publish-gather-1thread"
            };
            let o = run_case(&mut rig, &experts, threaded, Fault::None, 9, 3)?;
            report(
                o.mismatched_bufs == 0,
                format!(
                    "bytes ranks={n} map={map} path={path} stream={stream} rounds=9 buffers=3(cycling) \
                     compared={}B mismatched_snapshots={}{}",
                    o.compared_bytes,
                    o.mismatched_bufs,
                    o.first.map(|f| format!(" first: {f}")).unwrap_or_default()
                ),
            );
        }
    }
    // Negative controls: the comparison MUST report mismatches.
    let balanced = map_experts("balanced", n)?;
    let controls: [(&str, Fault); 2] = [
        (
            "flip-owner-source-sign-bit",
            Fault::FlipOwnerSource { slot: 3 },
        ),
        ("rank-skips-gather", Fault::SkipGather { rank: n - 1 }),
    ];
    for (name, fault) in controls {
        let o = run_case(&mut rig, &balanced, false, fault, 2, 2)?;
        report(
            o.mismatched_bufs > 0,
            format!(
                "negative-control {name} ranks={n} stream={stream}: mismatched_snapshots={} (must be > 0){}",
                o.mismatched_bufs,
                o.first.map(|f| format!(" first: {f}")).unwrap_or_default()
            ),
        );
    }
    rig.teardown()?;
    Ok(all_ok)
}

// ───────────────────────────── timing mode ─────────────────────────────

struct TimingCfg {
    n: usize,
    reps: usize,
    warmups: usize,
    map: String,
    real_stream: bool,
    warm_secs: u64,
    label: String,
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    let idx = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[idx.clamp(1, sorted.len()) - 1]
}

fn jstr(s: &str) -> String {
    format!("{:?}", s)
}

fn timing_mode(cfg: &TimingCfg) -> R<()> {
    let n = cfg.n;
    let experts = map_experts(&cfg.map, n)?;
    let mut rig = Rig::new(
        n,
        cfg.real_stream,
        if cfg.warm_secs > 0 {
            WARM_RING_BYTES
        } else {
            0
        },
    )?;
    // Owned rows hold stable patterns in both parities; other rows poisoned.
    for r in 0..n {
        let host = {
            let mut b = staging_bytes(r, 0, &experts, n);
            // round 0 pattern for owned rows
            b.truncate(BUF_BYTES);
            b
        };
        ctx(rig.gpus[r].bind_thread(), "bind_thread")?;
        for p in 0..2 {
            ctx(
                rig.gpus[r].hip.memcpy_htod(&rig.bufs[r][p].buf, &host),
                "fill buffer",
            )?;
        }
    }
    let ids: Vec<GpuTensor> = (0..n)
        .map(|r| rig.ids_tensor(r, &experts))
        .collect::<R<_>>()?;
    rig.sync_all()?;
    let transport = rig.build_transport(2)?;

    let devices: Vec<String> = (0..n)
        .map(|r| {
            let g = &rig.gpus[r];
            format!(
                "{{\"rank\":{r},\"device_id\":{},\"arch\":{},\"pci\":{}}}",
                g.device_id,
                jstr(&g.hip.get_arch(g.device_id).unwrap_or_default()),
                jstr(&g.hip.device_pci_bus_id(g.device_id).unwrap_or_default())
            )
        })
        .collect();
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    println!(
        "{{\"kind\":\"meta\",\"probe\":\"ep_row_gather_probe\",\"label\":{},\"impl\":\"ep_row_gather_fused\",\"W\":{n},\
         \"map\":{},\"experts\":{:?},\"slots\":{SLOTS},\"row_bytes\":{ROW_BYTES},\"peer_copies_per_round\":{},\
         \"reps\":{},\"warmups\":{},\"warm_seconds\":{},\"stream\":{},\"devices\":[{}],\
         \"fabric_cards\":{},\"hipfire_devices\":{},\"rocr_visible_devices\":{},\
         \"timing_definition\":\"round = publish on all ranks + gather on all ranks (one host thread) + stream sync of all ranks; host monotonic clock; eager, no graph\"}}",
        jstr(&cfg.label),
        jstr(&cfg.map),
        experts,
        peer_copies(&experts, n),
        cfg.reps,
        cfg.warmups,
        cfg.warm_secs,
        jstr(if cfg.real_stream { "real" } else { "null" }),
        devices.join(","),
        jstr(&env("FABRIC_CARDS")),
        jstr(&env("HIPFIRE_DEVICES")),
        jstr(&env("ROCR_VISIBLE_DEVICES")),
    );

    rig.warm_ring(cfg.warm_secs)?;

    let mut us = Vec::with_capacity(cfg.reps);
    for i in 0..(cfg.warmups + cfg.reps) {
        let parity = i % 2;
        let t = Instant::now();
        for r in 0..n {
            ctx(transport.publish(&mut rig.gpus[r], r, parity), "publish")?;
        }
        for r in 0..n {
            ctx(
                transport.gather(
                    &mut rig.gpus[r],
                    r,
                    parity,
                    EpSlotOwners::ExpertModulo(&ids[r], N_EXPERTS),
                    SLOTS,
                ),
                "gather",
            )?;
        }
        rig.sync_all()?;
        let dt = t.elapsed().as_secs_f64() * 1e6;
        if i >= cfg.warmups {
            us.push(dt);
        }
    }

    // Validate: every rank's parity buffers hold the owner bytes.
    let mut bad = 0usize;
    for r in 0..n {
        let want = expected_bytes(r, 0, &experts, n);
        for p in 0..2 {
            let got = ctx(
                rig.gpus[r].download_raw_bytes(&rig.bufs[r][p]),
                "validate download",
            )?;
            if got != want {
                bad += 1;
                eprintln!(
                    "VALIDATION FAILED rank {r} buffer {p}: first mismatch at byte {:?}",
                    first_diff(&got, &want)
                );
            }
        }
    }
    if bad > 0 {
        return Err(format!(
            "post-timing validation failed on {bad} buffers; timings invalid"
        ));
    }

    for (i, v) in us.iter().enumerate() {
        println!(
            "{{\"kind\":\"timing\",\"mode\":\"decode\",\"impl\":\"ep_row_gather_fused\",\"W\":{n},\"map\":{},\"rep\":{i},\"us\":{v:.3}}}",
            jstr(&cfg.map)
        );
    }
    let mut sorted = us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    let median = pct(&sorted, 50.0);
    let p95 = pct(&sorted, 95.0);
    println!(
        "{{\"kind\":\"summary\",\"mode\":\"decode\",\"impl\":\"ep_row_gather_fused\",\"W\":{n},\"map\":{},\"reps\":{},\
         \"mean_us\":{mean:.3},\"median_us\":{median:.3},\"p95_us\":{p95:.3},\"min_us\":{:.3},\"max_us\":{:.3},\
         \"peer_copies_per_round\":{},\"validate\":\"pass\"}}",
        jstr(&cfg.map),
        us.len(),
        sorted[0],
        sorted[sorted.len() - 1],
        peer_copies(&experts, n)
    );
    eprintln!(
        "ep_row_gather_probe timing ranks={n} map={} reps={}: mean={mean:.2}us median={median:.2}us p95={p95:.2}us min={:.2} max={:.2}",
        cfg.map,
        us.len(),
        sorted[0],
        sorted[sorted.len() - 1]
    );

    rig.release_transport(&transport)?;
    for (r, t) in ids.into_iter().enumerate() {
        ctx(rig.gpus[r].bind_thread(), "bind_thread")?;
        ctx(rig.gpus[r].hip.free(t.buf), "free ids")?;
    }
    rig.teardown()
}

// ───────────────────────────── cli ─────────────────────────────

fn usage() -> &'static str {
    "usage: ep_row_gather_probe bytes  --ranks N [--stream real|null]\n       \
     ep_row_gather_probe timing --ranks N --reps N --warmups N [--map balanced|skewed|recorded] [--stream real|null] [--warm-seconds S] [--label S]"
}

fn run() -> R<bool> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().ok_or_else(|| usage().to_string())?;
    let mut cfg = TimingCfg {
        n: 4,
        reps: 200,
        warmups: 10,
        map: "balanced".into(),
        real_stream: true,
        warm_secs: 10,
        label: String::new(),
    };
    let mut i = 1;
    while i < args.len() {
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("{} needs a value\n{}", args[i], usage()))?;
        let num = |v: &String| v.parse::<usize>().map_err(|e| format!("{}: {e}", args[i]));
        match args[i].as_str() {
            "--ranks" => cfg.n = num(v)?,
            "--reps" => cfg.reps = num(v)?,
            "--warmups" => cfg.warmups = num(v)?,
            "--map" => cfg.map = v.clone(),
            "--warm-seconds" => cfg.warm_secs = num(v)? as u64,
            "--label" => cfg.label = v.clone(),
            "--stream" => {
                cfg.real_stream = match v.as_str() {
                    "real" => true,
                    "null" => false,
                    _ => return Err(format!("--stream must be real|null\n{}", usage())),
                }
            }
            other => return Err(format!("unknown option {other}\n{}", usage())),
        }
        i += 2;
    }
    if !(2..=4).contains(&cfg.n) {
        return Err("--ranks must be 2..=4".into());
    }
    match mode.as_str() {
        "bytes" => bytes_mode(cfg.n, cfg.real_stream),
        "timing" => {
            if cfg.reps == 0 {
                return Err("--reps must be >= 1".into());
            }
            if !cfg.real_stream {
                cfg.warm_secs = 0;
            }
            timing_mode(&cfg).map(|()| true)
        }
        _ => Err(usage().to_string()),
    }
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("ep_row_gather_probe: FAIL");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("ep_row_gather_probe: ERROR: {e}");
            std::process::exit(2);
        }
    }
}
