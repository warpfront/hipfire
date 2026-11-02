// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Synthetic ONE-device oracle for the Flash-Next expert-parallel transport
//! launchers at N = 2..=8 ranks. No peer access and no model: every "rank" is
//! a set of buffers on the same physical card, so the launchers' routing,
//! bounds and bit-exactness are checked against a host reference.
//!
//! ```sh
//! cargo build --release -p hipfire-runtime --features lab --example ep_transport_oracle
//! HIPFIRE_DEVICES=0 target/release/examples/ep_transport_oracle \
//!     [--ranks 2,3,6] [--seed N] [--slots 37] [--device 0] \
//!     [--suites rows,tile] [--json receipt.json]
//! ```
//!
//! Exit status: 0 only when every case passed; 1 on any mismatch or launcher
//! error; 2 on usage errors / no device.
//!
//! For every `n` in `--ranks` and every `self_rank` in `0..n`:
//!
//! * `rows_by_owner` / `rows_by_slot_owner` (`Gpu::ep_gather_rows_by_owner` /
//!   `ep_gather_rows_by_slot_owner`, `src: [usize; 8]`): `n` source row buffers
//!   (rank `r`'s buffer = seeded random bytes with NaN payloads, -0.0,
//!   denormals, infinities in every third word) and a destination with its own
//!   seeded bytes. Slots carry random expert ids (including negative ids) or
//!   explicit owners (including `-1`). Expected destination: row `s` equals
//!   `src[owner]` row `s` when `owner` is a real peer, else the destination's
//!   ORIGINAL bytes (owned rows, negative ids / owners are never written).
//!   Two row widths (`n16` = 640 and 257 uint4) cover the grid tail.
//! * `tile_mask` (`Gpu::ep_tile_mask`): 1000 tile ids over 512 experts with
//!   `-1` sentinels and a padding tail; owner table `e % n`. Expected
//!   `owned[t] = ids[t]` when the tile's expert is owned by `self_rank`, else
//!   `-1`; the source ids must be unchanged.
//! * `tile_pull` (`Gpu::ep_gather_grouped_rows_by_tile_owner`,
//!   `sources: [usize; 8]`, owner table `e % n`): `n` grouped-row buffers of
//!   40 tiles x 16 rows (two row widths); expected destination: every live
//!   tile whose expert another rank owns is the owner's tile bytes, every other
//!   tile (owned, `-1` sentinel, padding tail) is the destination's original
//!   bytes.
//!
//! Every destination byte is compared. Cases print `PASS <case>` or
//! `FAIL <case>: <first difference>`; `--json` writes a receipt.
//!
//! # Distinct-device mode (`--devices`, no GPU peer access required)
//!
//! ```sh
//! ep_transport_oracle --devices 0,1[,..] [--transport auto|host|peer|both]
//!     [--host-flags mapped|coherent|all]
//!     [--suites xrows,xtiles,xctl,xtime,xhop,xedge] [--rounds 6] [--reps 200]
//!     [--layers 48] [--seed N] [--json receipt.json]
//! ```
//!
//! `--devices`: 2..=8 distinct HIP device indices (0..N-1 of the exported set).
//! Rank r = one `Gpu::init_with_device` on its OWN host thread, one launch stream
//! (`gpu.active_stream`); every kernel, event record and async copy of a rank is on
//! that stream. `--transport` defaults to env `HIPFIRE_QWEN4_EP_TRANSPORT`
//! (`auto|peer|host`, unset = `auto`); `auto` = `host` when any directed pair
//! is not byte-qualified (below), else `peer`; `both` runs every suite per transport
//! (the peer arm is skipped with a `SKIP` line + receipt note when a pair is not
//! byte-qualified). In `host` transport the oracle never calls `enable_peer_access`
//! and every kernel pointer is checked against the rank's own allocations + its alias
//! of each host bank (a violation is an error, never a launch). `--host-flags`:
//! `mapped` = `HIP_HOST_MALLOC_MAPPED`, `coherent` = mapped | `HIP_HOST_MALLOC_COHERENT`,
//! `all` = both (rank suites only; the `xedge` service chooses its own bank flags).
//!
//! ## Peer byte qualification (preflight)
//!
//! Peer is never chosen from `can_access_peer` alone. Unless `--transport host`, a
//! sequential coordinator runs BEFORE any rank thread: for every unordered device
//! pair it initializes exactly two `Gpu`s, calls
//! `multi_gpu::transport::qualify_peer` (peer-pointer kernel reads in BOTH directions)
//! and drops them. The receipt's `peer_qualification` holds the per-pair rows and the
//! directed `matrix` (`true|false`, `null` on the diagonal). `auto` picks the host arm
//! when any directed pair fails; `peer` runs only fully qualified sets.
//!
//! Explicit `--transport host` does NOT probe: nothing in the process touches
//! `enable_peer_access`. `auto`/`both` (and `peer`) probe, and `qualify_peer` may
//! enable peer access at process level (it cannot be undone). The receipt keeps the
//! two facts apart: `peer_qualification.performed` / `per_rank[].peer_access_enabled.
//! probe_ran` say a probe ran (so peer access MAY be enabled by `qualify_peer`), while
//! `peer_access_enabled.host` / `.peer` record whether the oracle's own arms called
//! `enable_peer_access` (always `false` for the host arm).
//!
//! ## Host bank and exchange protocol (host transport)
//!
//! Two banks (`K = 2`, bank `b = k % 2` for layer/round `k`), each ONE `host_malloc`
//! on rank 0 aliased into every rank with `host_get_device_pointer`. Per round:
//! 1. reuse guard: `k >= 2` host-syncs every rank's `done[b]`;
//! 2. (correctness) rank 0 poisons bank `b` with `0xDEADBEEF`, barrier;
//! 3. producer: stage `y_r`, publish OWNED units into `alias[r]` through the existing
//!    gather kernels with a host-computed publish mask (rows: slot-owner kernel with
//!    `n_ranks=2, self=1`, owners `0|-1`; tiles: tile-owner kernel with table `0|1`),
//!    record `ready[b]`;
//! 4. host barrier; 5. host-sync every OTHER rank's `ready[b]` (`-sw` variant:
//!    `hipStreamWaitEvent` on the rank stream instead);
//! 6. consumer: real gather into `y_r` with `src[q] = alias[r]` (`q != r`);
//! 7. record `done[b]`.
//! The peer arm keeps `y_r` local, gathers from `addr(y_q)` and applies the same
//! "readers finished before rewrite" rule.
//!
//! ## Suites
//!
//! * `xrows`, `xtiles`: `--rounds` seeded rounds (data differs per round and rank),
//!   uneven ownership passes, the modulo kernel pass, sentinel/padding tiles. Every
//!   destination byte and (rank 0, after step 5) the bank bytes are checked; live
//!   units must equal the owner's staged bytes, dead units must still be poison.
//!   Mismatches are classified `POISON` / `STALE` / `OTHER`.
//! * `xctl` (host only): gather before publication must read back `POISON` for every
//!   non-owned live unit; a normal round on the same bank afterwards must pass.
//! * `xtime`: shapes `decode`, `fewrow`, `prefill` (+`prefill-solo`), each with a
//!   verified warm round, an exchange-free `-baseline`, a kernel-event pass
//!   (publish/gather) and a wall-clock pass; `decode`/`fewrow` also run `-sw`.
//! * `xhop`: pipeline-boundary hop rank 0 -> rank 1 (one-way and ping-pong/2) for
//!   10,240 B / 40,960 B / 15,728,640 B with methods `mapped`, `mapped-sw`,
//!   `mapped-sw-chain` (5 hops, total/5), `staged` (pinned buffer) and `peer`.
//!
//! * `xedge`: the shared edge transport service (`multi_gpu::transport`: `provision`,
//!   `EdgeLease::{publish, receive, poison, drain}`, lab `inject_fault`) driven directly
//!   by a dedicated coordinator thread BEFORE the rank threads (it needs two `Gpu`s on
//!   one thread, so it does not go through the two-thread `Env`). For every ADJACENT
//!   device pair of `--devices` and every selected edge transport (`host` -> HostMapped;
//!   `peer` -> Peer only for a byte-qualified pair; `auto` -> Peer when the pair is
//!   qualified else HostMapped; `both` -> HostMapped + Peer when qualified). Payloads
//!   are the HC row shapes F32 (40,960 B/row) and BF16 (20,480 B/row). Cases per
//!   `xedge <transport> dev<A>-dev<B>` prefix: `fixed <dt> rows=<1|8|1536|4096>`
//!   (unique seeded bytes per generation, every byte compared, source unchanged);
//!   `chain6 <dt> rows=<R>` (six alternating-device hops over two leases, two
//!   iterations, every hop destination compared); `reuse <dt> dir=fwd|rev` (mixed
//!   row counts across k+2 bank reuse, 256 B poisoned destination tail and random
//!   source tail must stay untouched); `delayed-producer|delayed-consumer <shape>`
//!   (a GPU stall of ~100+ ms is enqueued on the producer/consumer stream so a
//!   premature bank read/overwrite shows up as STALE data); `poison <FaultPoint>` for
//!   all nine lab hooks (primes generations 0 and 1, injects at generation 2 so real
//!   bank reuse is exercised, requires the faulting call plus every later publish and
//!   receive to fail closed, `drain` to succeed, then a reprovisioned lease to run
//!   clean); `poison explicit`; `reject <short-view|row-align|dtype-bytes|rows-zero|
//!   over-max|foreign-view|recv-op|recv-rows|recv-dtype|recv-epoch>`; `reprovision`
//!   (fresh epoch, restarted generations, stale-epoch descriptor refused). Every lease
//!   is drained before any tensor, stream or device is freed; a failed drain
//!   quarantines (leaks) the lease and tensors and stops that pair.
//!
//! Output: `PASS <case>` / `FAIL <case>: <detail>`, `PROBE ...` (peer qualification),
//! `TIME <suite> <shape>
//! <transport>/<flags> median_us=.. p10_us=.. p90_us=.. [GBps=..] publish_us=[..]
//! gather_us=[..]`, final `RESULT:` line; `--json` writes the receipt (arch, device
//! ids, PCI bus id, resolved transports, `can_access_peer` matrix, `peer_qualification`
//! (probe matrix), `xedge` arms, host flags, alias==host per rank, cases, timings,
//! `peer_access_enabled` per rank).
//! Exit status: 0 only when every correctness case passed (timing has no thresholds);
//! 1 on any FAIL or rank error; 2 on usage errors / no device / no runnable transport.

use hip_bridge::{
    DeviceBuffer, Event, HIP_EVENT_DISABLE_TIMING, HIP_EVENT_RELEASE_TO_SYSTEM,
    HIP_HOST_MALLOC_COHERENT, HIP_HOST_MALLOC_MAPPED,
};
use rdna_compute::moe::{ep_tile_pm_arch, EP_TILE_ROWS};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{fence, AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use hipfire_runtime::mesh_placement::EdgeTransport;
use hipfire_runtime::multi_gpu::transport::{
    provision, qualify_peer, EdgeDescriptor, EdgeId, EdgeLease, EdgeSpec, FaultPoint,
};

type R<T> = Result<T, String>;

const EXPERTS: usize = 512;
const MASK_TILES: usize = 1000;
const PULL_TILES: usize = 40;
const MASK_POISON: i32 = 0x5A5A_5A5A;

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

/// Deterministic stream of u64 from a seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = splitmix(self.0);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `len` bytes of seeded random words; every third word is a special bit
/// pattern (rotating through [`SPECIAL`] by seed).
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for (i, w) in out.chunks_exact_mut(4).enumerate() {
        let v = if i % 3 == 0 {
            SPECIAL[(i / 3 + seed as usize) % SPECIAL.len()]
        } else {
            splitmix(seed.wrapping_mul(0x1_0000_0001).wrapping_add(i as u64)) as u32
        };
        w.copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// First differing byte of two slices, as a report string.
fn first_diff(label: &str, expected: &[u8], got: &[u8], row_bytes: usize) -> Option<String> {
    if expected.len() != got.len() {
        return Some(format!(
            "{label}: length {} vs {}",
            expected.len(),
            got.len()
        ));
    }
    let offset = expected.iter().zip(got).position(|(a, b)| a != b)?;
    let bad = expected.iter().zip(got).filter(|(a, b)| a != b).count();
    let word = offset & !3;
    let end = (word + 4).min(expected.len());
    Some(format!(
        "{label}: {bad} differing byte(s); first at byte {offset} (row {}, word@{word} expected {} got {})",
        offset / row_bytes.max(1),
        hex(&expected[word..end]),
        hex(&got[word..end]),
    ))
}

fn ctx<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> R<T> {
    r.map_err(|e| format!("{what}: {e}"))
}

struct Rig {
    gpu: Gpu,
}

impl Rig {
    fn upload(&self, bytes: &[u8], what: &str) -> R<GpuTensor> {
        ctx(self.gpu.upload_raw(bytes, &[bytes.len()]), what)
    }
    fn sync(&self) -> R<()> {
        ctx(self.gpu.bind_thread(), "bind_thread")?;
        ctx(self.gpu.hip.device_synchronize(), "device_synchronize")
    }
    fn download(&self, t: &GpuTensor, what: &str) -> R<Vec<u8>> {
        self.sync()?;
        ctx(self.gpu.download_raw_bytes(t), what)
    }
}

fn addr(t: &GpuTensor) -> usize {
    t.buf.as_ptr() as usize
}

fn sources8(tensors: &[GpuTensor]) -> [usize; 8] {
    let mut out = [0usize; 8];
    for (slot, t) in out.iter_mut().zip(tensors) {
        *slot = addr(t);
    }
    out
}

// ───────────────────────────── row gathers ─────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum RowKind {
    ByOwner,
    BySlotOwner,
}

fn row_case(
    rig: &mut Rig,
    kind: RowKind,
    n: usize,
    self_rank: usize,
    slots: usize,
    n16: usize,
    seed: u64,
) -> R<Option<String>> {
    let row_bytes = n16 * 16;
    let len = slots * row_bytes;
    let case_seed = splitmix(
        seed ^ ((n as u64) << 48) ^ ((self_rank as u64) << 40) ^ ((n16 as u64) << 8) ^ kind as u64,
    );
    // Slot -> expert id (modulo kernel) or explicit owner (slot-owner kernel).
    let mut rng = Rng(case_seed);
    let ids: Vec<i32> = (0..slots)
        .map(|s| match kind {
            RowKind::ByOwner => {
                if s % 7 == 3 {
                    -1 - rng.below(5) as i32
                } else {
                    rng.below(EXPERTS) as i32
                }
            }
            RowKind::BySlotOwner => {
                if s % 7 == 3 {
                    -1
                } else {
                    rng.below(n) as i32
                }
            }
        })
        .collect();
    let owner_of = |id: i32| -> Option<usize> {
        match kind {
            RowKind::ByOwner => (id >= 0 && (id as usize) < EXPERTS).then(|| id as usize % n),
            RowKind::BySlotOwner => (id >= 0 && (id as usize) < n).then_some(id as usize),
        }
    };
    let src_host: Vec<Vec<u8>> = (0..n)
        .map(|r| random_bytes(case_seed ^ splitmix(r as u64 + 1), len))
        .collect();
    let dst_host = random_bytes(case_seed ^ 0xD57D_57D5, len);
    let mut expected = dst_host.clone();
    for (s, id) in ids.iter().enumerate() {
        if let Some(owner) = owner_of(*id) {
            if owner != self_rank {
                expected[s * row_bytes..(s + 1) * row_bytes]
                    .copy_from_slice(&src_host[owner][s * row_bytes..(s + 1) * row_bytes]);
            }
        }
    }
    let mut src = Vec::with_capacity(n);
    for (r, host) in src_host.iter().enumerate() {
        src.push(rig.upload(host, &format!("upload source {r}"))?);
    }
    let dst = rig.upload(&dst_host, "upload destination")?;
    let id_tensor = rig.upload(&i32_bytes(&ids), "upload slot ids")?;
    ctx(rig.gpu.bind_thread(), "bind_thread")?;
    let sources = sources8(&src);
    let launch = match kind {
        RowKind::ByOwner => rig.gpu.ep_gather_rows_by_owner(
            &id_tensor,
            sources,
            addr(&dst),
            slots,
            n,
            self_rank,
            n16,
            EXPERTS,
        ),
        RowKind::BySlotOwner => rig.gpu.ep_gather_rows_by_slot_owner(
            &id_tensor,
            sources,
            addr(&dst),
            slots,
            n,
            self_rank,
            n16,
        ),
    };
    ctx(launch, "launch")?;
    let got = rig.download(&dst, "download destination")?;
    let mut problem = first_diff("destination", &expected, &got, row_bytes);
    if problem.is_none() {
        // Sources and ids are read-only inputs.
        for (r, host) in src_host.iter().enumerate() {
            let after = rig.download(&src[r], "download source")?;
            if let Some(p) = first_diff(&format!("source {r} (read-only)"), host, &after, row_bytes)
            {
                problem = Some(p);
                break;
            }
        }
    }
    Ok(problem)
}

// ───────────────────────────── tile mask / pull ─────────────────────────────

/// Tile ids over `EXPERTS` with ~20% `-1` sentinels and a `-1` padding tail.
fn tile_ids(seed: u64, tiles: usize) -> Vec<i32> {
    let mut rng = Rng(seed);
    let live = tiles - tiles / 5;
    (0..tiles)
        .map(|t| {
            if t >= live || t % 5 == 2 {
                -1
            } else {
                rng.below(EXPERTS) as i32
            }
        })
        .collect()
}

fn owner_table(n: usize) -> Vec<i32> {
    (0..EXPERTS).map(|e| (e % n) as i32).collect()
}

fn mask_case(rig: &mut Rig, n: usize, self_rank: usize, seed: u64) -> R<Option<String>> {
    let case_seed = splitmix(seed ^ 0x3A5C ^ ((n as u64) << 48) ^ ((self_rank as u64) << 40));
    let ids = tile_ids(case_seed, MASK_TILES);
    let table = owner_table(n);
    let expected: Vec<i32> = ids
        .iter()
        .map(|&id| {
            if id >= 0 && table[id as usize] as usize == self_rank {
                id
            } else {
                -1
            }
        })
        .collect();
    let ids_t = rig.upload(&i32_bytes(&ids), "upload tile ids")?;
    let owned_t = rig.upload(
        &i32_bytes(&vec![MASK_POISON; MASK_TILES]),
        "upload owned ids",
    )?;
    let table_t = rig.upload(&i32_bytes(&table), "upload owner table")?;
    ctx(rig.gpu.bind_thread(), "bind_thread")?;
    ctx(
        rig.gpu
            .ep_tile_mask(&ids_t, &owned_t, &table_t, MASK_TILES, EXPERTS, self_rank),
        "launch",
    )?;
    let got = rig.download(&owned_t, "download owned ids")?;
    if let Some(p) = first_diff("tile_ids_owned", &i32_bytes(&expected), &got, 4) {
        return Ok(Some(p));
    }
    let ids_after = rig.download(&ids_t, "download tile ids")?;
    Ok(first_diff(
        "tile_ids (read-only)",
        &i32_bytes(&ids),
        &ids_after,
        4,
    ))
}

fn pull_case(
    rig: &mut Rig,
    n: usize,
    self_rank: usize,
    row_u4: usize,
    seed: u64,
) -> R<Option<String>> {
    let tile_bytes = EP_TILE_ROWS * row_u4 * 16;
    let len = PULL_TILES * tile_bytes;
    let case_seed = splitmix(
        seed ^ 0x9111 ^ ((n as u64) << 48) ^ ((self_rank as u64) << 40) ^ ((row_u4 as u64) << 8),
    );
    let ids = tile_ids(case_seed, PULL_TILES);
    let table = owner_table(n);
    let src_host: Vec<Vec<u8>> = (0..n)
        .map(|r| random_bytes(case_seed ^ splitmix(r as u64 + 1), len))
        .collect();
    let dst_host = random_bytes(case_seed ^ 0xD57D_57D5, len);
    let mut expected = dst_host.clone();
    for (t, &id) in ids.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let owner = table[id as usize] as usize;
        if owner != self_rank {
            expected[t * tile_bytes..(t + 1) * tile_bytes]
                .copy_from_slice(&src_host[owner][t * tile_bytes..(t + 1) * tile_bytes]);
        }
    }
    let mut src = Vec::with_capacity(n);
    for (r, host) in src_host.iter().enumerate() {
        src.push(rig.upload(host, &format!("upload source {r}"))?);
    }
    let dst = rig.upload(&dst_host, "upload destination")?;
    let ids_t = rig.upload(&i32_bytes(&ids), "upload tile ids")?;
    let table_t = rig.upload(&i32_bytes(&table), "upload owner table")?;
    ctx(rig.gpu.bind_thread(), "bind_thread")?;
    ctx(
        rig.gpu.ep_gather_grouped_rows_by_tile_owner(
            &ids_t,
            &table_t,
            sources8(&src),
            &dst,
            PULL_TILES,
            EXPERTS,
            n,
            self_rank,
            row_u4,
        ),
        "launch",
    )?;
    let got = rig.download(&dst, "download destination")?;
    let mut problem = first_diff("destination", &expected, &got, tile_bytes);
    if problem.is_none() {
        for (r, host) in src_host.iter().enumerate() {
            let after = rig.download(&src[r], "download source")?;
            if let Some(p) =
                first_diff(&format!("source {r} (read-only)"), host, &after, tile_bytes)
            {
                problem = Some(p);
                break;
            }
        }
    }
    Ok(problem)
}

// ───────────────────────────── driver ─────────────────────────────

const USAGE: &str = "usage: ep_transport_oracle [--ranks 2,3,6] [--seed N] [--slots 37] [--device 0] [--suites rows,tile] [--json FILE]\n       ep_transport_oracle --devices 0,1[,..] [--transport auto|host|peer|both] [--host-flags mapped|coherent|all] [--suites xrows,xtiles,xctl,xtime,xhop,xedge] [--rounds 6] [--reps 200] [--layers 48] [--seed N] [--json FILE]";
const X_SUITES: [&str; 6] = ["xrows", "xtiles", "xctl", "xtime", "xhop", "xedge"];

struct Args {
    ranks: Vec<usize>,
    seed: u64,
    slots: usize,
    device: i32,
    suites: Option<String>,
    json: Option<String>,
    devices: Option<Vec<i32>>,
    transport: Option<String>,
    host_flags: String,
    rounds: usize,
    reps: usize,
    layers: usize,
    /// One-device-only flags seen (rejected next to `--devices`).
    one_device_flags: Vec<&'static str>,
    /// Distinct-device-only flags seen (rejected without `--devices`).
    multi_flags: Vec<&'static str>,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        ranks: vec![2, 3, 6],
        seed: 0x5EED_0E9A,
        slots: 37,
        device: 0,
        suites: None,
        json: None,
        devices: None,
        transport: None,
        host_flags: "all".into(),
        rounds: 6,
        reps: 200,
        layers: 48,
        one_device_flags: Vec::new(),
        multi_flags: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--ranks" => {
                a.ranks = value("--ranks")?
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("--ranks {s:?}: {e}"))
                    })
                    .collect::<R<_>>()?;
                a.one_device_flags.push("--ranks");
            }
            "--seed" => {
                a.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--slots" => {
                a.slots = value("--slots")?
                    .parse()
                    .map_err(|e| format!("--slots: {e}"))?;
                a.one_device_flags.push("--slots");
            }
            "--device" => {
                a.device = value("--device")?
                    .parse()
                    .map_err(|e| format!("--device: {e}"))?;
                a.one_device_flags.push("--device");
            }
            "--json" => a.json = Some(value("--json")?),
            "--suites" => a.suites = Some(value("--suites")?),
            "--devices" => {
                a.devices = Some(
                    value("--devices")?
                        .split(',')
                        .map(|s| {
                            s.trim()
                                .parse::<i32>()
                                .map_err(|e| format!("--devices {s:?}: {e}"))
                        })
                        .collect::<R<_>>()?,
                );
            }
            "--transport" => {
                a.transport = Some(value("--transport")?);
                a.multi_flags.push("--transport");
            }
            "--host-flags" => {
                a.host_flags = value("--host-flags")?;
                a.multi_flags.push("--host-flags");
            }
            "--rounds" => {
                a.rounds = value("--rounds")?
                    .parse()
                    .map_err(|e| format!("--rounds: {e}"))?;
                a.multi_flags.push("--rounds");
            }
            "--reps" => {
                a.reps = value("--reps")?
                    .parse()
                    .map_err(|e| format!("--reps: {e}"))?;
                a.multi_flags.push("--reps");
            }
            "--layers" => {
                a.layers = value("--layers")?
                    .parse()
                    .map_err(|e| format!("--layers: {e}"))?;
                a.multi_flags.push("--layers");
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if let Some(devs) = &a.devices {
        if !a.one_device_flags.is_empty() {
            return Err(format!(
                "{} are one-device options and cannot be combined with --devices",
                a.one_device_flags.join(", ")
            ));
        }
        if !(2..=8).contains(&devs.len()) {
            return Err("--devices takes 2..=8 distinct device indices".into());
        }
        let mut seen = HashSet::new();
        if devs.iter().any(|&d| d < 0 || !seen.insert(d)) {
            return Err("--devices entries must be distinct and non-negative".into());
        }
        if let Some(t) = &a.transport {
            if !matches!(t.as_str(), "auto" | "host" | "peer" | "both") {
                return Err(format!("--transport takes host|peer|both; got {t:?}"));
            }
        }
        if !matches!(a.host_flags.as_str(), "mapped" | "coherent" | "all") {
            return Err(format!(
                "--host-flags takes mapped|coherent|all; got {:?}",
                a.host_flags
            ));
        }
        if let Some(v) = &a.suites {
            if let Some(bad) = v.split(',').find(|s| !X_SUITES.contains(s)) {
                return Err(format!(
                    "--suites takes {}; got {bad:?}",
                    X_SUITES.join(",")
                ));
            }
        }
        if a.rounds == 0 || a.reps == 0 || a.layers == 0 {
            return Err("--rounds, --reps and --layers must be non-zero".into());
        }
    } else {
        if !a.multi_flags.is_empty() {
            return Err(format!("{} require --devices", a.multi_flags.join(", ")));
        }
        if let Some(v) = &a.suites {
            if let Some(bad) = v.split(',').find(|s| !matches!(*s, "rows" | "tile")) {
                return Err(format!("--suites takes rows,tile; got {bad:?}"));
            }
        }
        if a.ranks.is_empty() || a.ranks.iter().any(|n| !(2..=8).contains(n)) {
            return Err("--ranks entries must be within 2..=8".into());
        }
        if a.slots == 0 {
            return Err("--slots must be non-zero".into());
        }
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if args.devices.is_some() {
        multi_main(&args);
    } else {
        legacy_main(&args);
    }
}

/// The original one-device oracle (no `--devices`).
fn legacy_main(args: &Args) -> ! {
    let (rows, tile) = match &args.suites {
        None => (true, true),
        Some(v) => (
            v.split(',').any(|s| s == "rows"),
            v.split(',').any(|s| s == "tile"),
        ),
    };
    let gpu = match Gpu::init_with_device(args.device) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device: {e}");
            std::process::exit(2);
        }
    };
    let arch = gpu.arch.clone();
    println!(
        "ep_transport_oracle device={} arch={arch} tile_pm_arch={} ranks={:?} seed={:#x}",
        args.device,
        ep_tile_pm_arch(&arch),
        args.ranks,
        args.seed
    );
    let mut rig = Rig { gpu };
    let (mut pass, mut fail) = (0usize, 0usize);
    let mut cases: Vec<Value> = Vec::new();
    let mut record = |name: String, outcome: R<Option<String>>| {
        let (ok, detail) = match outcome {
            Ok(None) => (true, None),
            Ok(Some(p)) => (false, Some(p)),
            Err(e) => (false, Some(format!("error: {e}"))),
        };
        match &detail {
            None => println!("PASS {name}"),
            Some(d) => println!("FAIL {name}: {d}"),
        }
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
        cases.push(json!({"case": name, "pass": ok, "detail": detail}));
    };
    for &n in &args.ranks {
        for self_rank in 0..n {
            if rows {
                for (kind, label) in [
                    (RowKind::ByOwner, "rows_by_owner"),
                    (RowKind::BySlotOwner, "rows_by_slot_owner"),
                ] {
                    for n16 in [640usize, 257] {
                        let name = format!(
                            "{label} n={n} self={self_rank} slots={} n16={n16}",
                            args.slots
                        );
                        let outcome =
                            row_case(&mut rig, kind, n, self_rank, args.slots, n16, args.seed);
                        record(name, outcome);
                    }
                }
            }
            if tile {
                record(
                    format!("tile_mask n={n} self={self_rank} tiles={MASK_TILES}"),
                    mask_case(&mut rig, n, self_rank, args.seed),
                );
                for row_u4 in [64usize, 128] {
                    record(
                        format!(
                            "tile_pull n={n} self={self_rank} tiles={PULL_TILES} row_u4={row_u4}"
                        ),
                        pull_case(&mut rig, n, self_rank, row_u4, args.seed),
                    );
                }
            }
        }
    }
    println!(
        "RESULT: {} ({pass} passed, {fail} failed)",
        if fail == 0 { "PASS" } else { "FAIL" }
    );
    if let Some(path) = &args.json {
        let receipt = json!({
            "arch": arch,
            "device": args.device,
            "tile_pm_arch": ep_tile_pm_arch(&arch),
            "ranks": args.ranks,
            "seed": args.seed,
            "slots": args.slots,
            "passed": pass,
            "failed": fail,
            "cases": cases,
        });
        if let Err(e) = std::fs::write(
            path,
            serde_json::to_string_pretty(&receipt).unwrap_or_default(),
        ) {
            eprintln!("--json {path}: {e}");
            std::process::exit(2);
        }
    }
    std::process::exit(if fail == 0 && pass > 0 { 0 } else { 1 });
}

// ═══════════════════════ distinct-device mode (`--devices`) ═══════════════════════
//
// One host thread per rank; every rank owns its `Gpu`, launch stream and events.
// All ranks run the SAME deterministic script (arms x suites x passes x rounds), so
// collective calls (barriers, bank create/destroy, votes) always line up.

const POISON_WORD: u32 = 0xDEAD_BEEF;
const BARRIER_TIMEOUT: Duration = Duration::from_secs(30);
const WARMUPS: usize = 10;
const HOP_ROW: usize = 10_240;
const HOP_PAYLOADS: [usize; 3] = [10_240, 40_960, 15_728_640];

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ───────────────────────────── barrier ─────────────────────────────

/// Generation-counting spin(+yield) barrier with an abort flag and a timeout.
/// The teardown instance ignores the abort flag so every rank can still meet.
struct Barrier {
    parties: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
    aborted: AtomicBool,
    honor_abort: bool,
}

impl Barrier {
    fn new(parties: usize, honor_abort: bool) -> Self {
        Self {
            parties,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            aborted: AtomicBool::new(false),
            honor_abort,
        }
    }
    fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
    }
    fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }
    fn wait(&self) -> R<()> {
        if self.honor_abort && self.is_aborted() {
            return Err("barrier aborted".into());
        }
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.parties {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
            return Ok(());
        }
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            if self.generation.load(Ordering::Acquire) != generation {
                return Ok(());
            }
            if self.honor_abort && self.is_aborted() {
                return Err("barrier aborted".into());
            }
            if spins < 2048 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            if start.elapsed() >= BARRIER_TIMEOUT {
                if self.honor_abort {
                    self.abort();
                }
                return Err("barrier timed out".into());
            }
            std::thread::yield_now();
        }
    }
}

// ───────────────────────────── configuration ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostFlags {
    Mapped,
    Coherent,
}

impl HostFlags {
    fn bits(self) -> u32 {
        match self {
            HostFlags::Mapped => HIP_HOST_MALLOC_MAPPED,
            HostFlags::Coherent => HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_COHERENT,
        }
    }
    fn name(self) -> &'static str {
        match self {
            HostFlags::Mapped => "mapped",
            HostFlags::Coherent => "coherent",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Host(HostFlags),
    Peer,
}

impl Arm {
    fn is_host(self) -> bool {
        matches!(self, Arm::Host(_))
    }
    fn transport(self) -> &'static str {
        if self.is_host() {
            "host"
        } else {
            "peer"
        }
    }
    fn flags_name(self) -> &'static str {
        match self {
            Arm::Host(f) => f.name(),
            Arm::Peer => "none",
        }
    }
    fn tag(self) -> String {
        format!("{}/{}", self.transport(), self.flags_name())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TransportReq {
    Auto,
    Host,
    Peer,
    Both,
}

struct Cfg {
    devices: Vec<i32>,
    transport: TransportReq,
    transport_src: String,
    host_flags: Vec<HostFlags>,
    xrows: bool,
    xtiles: bool,
    xctl: bool,
    xtime: bool,
    xhop: bool,
    xedge: bool,
    rounds: usize,
    reps: usize,
    layers: usize,
    seed: u64,
}

// ───────────────────────────── shared state ─────────────────────────────

/// One host allocation aliased into every rank's device context.
struct HostBank {
    host: AtomicUsize,
    bytes: AtomicUsize,
    /// Per-rank device alias (`hipHostGetDevicePointer`).
    alias: Vec<AtomicUsize>,
}

impl HostBank {
    fn new(ranks: usize) -> Self {
        Self {
            host: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            alias: (0..ranks).map(|_| AtomicUsize::new(0)).collect(),
        }
    }
}

#[derive(Default)]
struct RankInfo {
    arch: String,
    pci: Option<String>,
    can_access: Vec<Option<bool>>,
    peer_enabled_host: bool,
    peer_enabled_final: bool,
}

#[derive(Default, Clone)]
struct TimedOut {
    durs: Vec<f64>,
    pubs: Vec<f64>,
    gats: Vec<f64>,
}

#[derive(Default)]
struct Rec {
    cases: Vec<Value>,
    pass: usize,
    fail: usize,
    timings: Vec<Value>,
    banks: Vec<Value>,
    notes: Vec<String>,
    errors: Vec<String>,
    init_err: Option<String>,
    resolved: Vec<String>,
    arms: Vec<String>,
    xedge_arms: Vec<Value>,
}

impl Rec {
    fn case(&mut self, name: String, detail: Option<String>) {
        match &detail {
            None => {
                println!("PASS {name}");
                self.pass += 1;
            }
            Some(d) => {
                println!("FAIL {name}: {d}");
                self.fail += 1;
            }
        }
        self.cases
            .push(json!({"case": name, "pass": detail.is_none(), "detail": detail}));
    }
    fn note(&mut self, s: String) {
        println!("NOTE {s}");
        self.notes.push(s);
    }
}

struct Shared {
    cfg: Cfg,
    n: usize,
    barrier: Barrier,
    td: Barrier,
    init_lock: Mutex<()>,
    infos: Vec<Mutex<RankInfo>>,
    ready: Vec<[AtomicPtr<Event>; 2]>,
    done: Vec<[AtomicPtr<Event>; 2]>,
    /// Per-rank address of the rank's local `y` (peer transport reads it).
    yaddr: Vec<AtomicUsize>,
    banks: [HostBank; 2],
    alloc_err: Mutex<Option<String>>,
    vote: AtomicUsize,
    rec: Mutex<Rec>,
    times: Vec<Mutex<TimedOut>>,
    hop_seq: AtomicUsize,
    hop_t0: AtomicU64,
    hop_t1: AtomicU64,
    epoch: Instant,
    pre: OnceLock<Preflight>,
}

impl Shared {
    fn new(cfg: Cfg) -> Self {
        let n = cfg.devices.len();
        let ev_pair = || {
            [
                AtomicPtr::new(std::ptr::null_mut()),
                AtomicPtr::new(std::ptr::null_mut()),
            ]
        };
        Shared {
            cfg,
            n,
            barrier: Barrier::new(n, true),
            td: Barrier::new(n, false),
            init_lock: Mutex::new(()),
            infos: (0..n).map(|_| Mutex::new(RankInfo::default())).collect(),
            ready: (0..n).map(|_| ev_pair()).collect(),
            done: (0..n).map(|_| ev_pair()).collect(),
            yaddr: (0..n).map(|_| AtomicUsize::new(0)).collect(),
            banks: [HostBank::new(n), HostBank::new(n)],
            alloc_err: Mutex::new(None),
            vote: AtomicUsize::new(0),
            rec: Mutex::new(Rec::default()),
            times: (0..n).map(|_| Mutex::new(TimedOut::default())).collect(),
            hop_seq: AtomicUsize::new(0),
            hop_t0: AtomicU64::new(0),
            hop_t1: AtomicU64::new(0),
            epoch: Instant::now(),
            pre: OnceLock::new(),
        }
    }
    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }
    fn pre(&self) -> &Preflight {
        self.pre.get().expect("preflight not run")
    }
}

// ───────────────────────────── geometry / plans / data ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    SlotOwner,
    Modulo,
    Tiles,
}

#[derive(Clone, Copy)]
struct Geom {
    kind: Kind,
    units: usize,
    unit_bytes: usize,
    n16: usize,
    row_u4: usize,
}

impl Geom {
    fn rows(kind: Kind, slots: usize, n16: usize) -> Self {
        Geom {
            kind,
            units: slots,
            unit_bytes: n16 * 16,
            n16,
            row_u4: 0,
        }
    }
    fn tiles(tiles: usize, row_u4: usize) -> Self {
        Geom {
            kind: Kind::Tiles,
            units: tiles,
            unit_bytes: EP_TILE_ROWS * row_u4 * 16,
            n16: 0,
            row_u4,
        }
    }
    fn total(&self) -> usize {
        self.units * self.unit_bytes
    }
}

/// Per-unit ownership for one round. `dev0`/`dev1` are the consumer's device tables:
/// rows = owners (slot-owner kernel) or expert ids (modulo kernel), tiles = tile ids and
/// the expert owner table.
#[derive(Clone)]
struct Plan {
    owner: Vec<Option<usize>>,
    dev0: Vec<i32>,
    dev1: Vec<i32>,
}

fn plan_rows(kind: Kind, slots: usize, n: usize, uneven: bool, seed: u64) -> Plan {
    let mut rng = Rng(seed);
    let mut owner = Vec::with_capacity(slots);
    let mut dev0 = Vec::with_capacity(slots);
    for s in 0..slots {
        if kind == Kind::Modulo {
            let id = if s % 7 == 3 {
                -1 - rng.below(5) as i32
            } else {
                rng.below(EXPERTS) as i32
            };
            dev0.push(id);
            owner.push((id >= 0).then(|| id as usize % n));
        } else {
            let o = if s % 7 == 3 {
                -1
            } else if uneven {
                if rng.below(10) < 7 {
                    0
                } else {
                    1 + rng.below(n - 1) as i32
                }
            } else {
                rng.below(n) as i32
            };
            dev0.push(o);
            owner.push((o >= 0).then(|| o as usize));
        }
    }
    Plan {
        owner,
        dev0,
        dev1: Vec::new(),
    }
}

fn plan_balanced(slots: usize, n: usize) -> Plan {
    Plan {
        owner: (0..slots).map(|s| Some(s % n)).collect(),
        dev0: (0..slots).map(|s| (s % n) as i32).collect(),
        dev1: Vec::new(),
    }
}

fn table_even(n: usize) -> Vec<i32> {
    owner_table(n)
}

/// Rank 0 owns `e % 4 != 3`; the remainder is shared round-robin by ranks `1..n`.
fn table_uneven(n: usize) -> Vec<i32> {
    (0..EXPERTS)
        .map(|e| {
            if e % 4 != 3 {
                0
            } else {
                (1 + (e / 4) % (n - 1)) as i32
            }
        })
        .collect()
}

fn plan_tiles(ids: Vec<i32>, table: Vec<i32>) -> Plan {
    let owner = ids
        .iter()
        .map(|&id| (id >= 0 && (id as usize) < EXPERTS).then(|| table[id as usize] as usize))
        .collect();
    Plan {
        owner,
        dev0: ids,
        dev1: table,
    }
}

/// 90% live tiles (every tenth is a `-1` sentinel).
fn tile_ids_live90(seed: u64, tiles: usize) -> Vec<i32> {
    let mut rng = Rng(seed);
    (0..tiles)
        .map(|t| {
            if t % 10 == 9 {
                -1
            } else {
                rng.below(EXPERTS) as i32
            }
        })
        .collect()
}

fn data_seed(seed: u64, tag: u64, round: usize, rank: usize) -> u64 {
    splitmix(seed ^ splitmix(tag) ^ ((round as u64) << 32) ^ ((rank as u64) << 48) ^ 0x57A6_ED00)
}

fn unit_data(dseed: u64, u: usize, ub: usize) -> Vec<u8> {
    random_bytes(
        splitmix(dseed ^ (u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        ub,
    )
}

fn staged_bytes(dseed: u64, g: &Geom) -> Vec<u8> {
    let mut v = Vec::with_capacity(g.total());
    for u in 0..g.units {
        v.extend_from_slice(&unit_data(dseed, u, g.unit_bytes));
    }
    v
}

fn poison_unit(ub: usize) -> Vec<u8> {
    POISON_WORD
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(ub)
        .collect()
}

fn classify(got: &[u8], exp: &[u8], stale: Option<&[u8]>) -> &'static str {
    let pw = POISON_WORD.to_le_bytes();
    if got
        .chunks_exact(4)
        .zip(exp.chunks_exact(4))
        .any(|(g, e)| g == pw.as_slice() && e != pw.as_slice())
    {
        "POISON"
    } else if stale == Some(got) {
        "STALE"
    } else {
        "OTHER"
    }
}

fn unit_diff(u: usize, g: &[u8], e: &[u8]) -> String {
    let off = g.iter().zip(e).position(|(a, b)| a != b).unwrap_or(0);
    let w = off & !3;
    let end = (w + 4).min(e.len());
    format!(
        "unit {u} byte {off}: word expected {} got {}",
        hex(&e[w..end]),
        hex(&g[w..end])
    )
}

/// Compare unit by unit; mismatches are classified POISON / STALE / OTHER.
fn check_units(
    units: usize,
    ub: usize,
    got: &[u8],
    exp: &[u8],
    stale: &dyn Fn(usize) -> Option<Vec<u8>>,
) -> Option<String> {
    if got.len() != exp.len() || got.len() != units * ub {
        return Some(format!(
            "length got {} expected {} (want {})",
            got.len(),
            exp.len(),
            units * ub
        ));
    }
    let (mut bad, mut p, mut s, mut o) = (0usize, 0usize, 0usize, 0usize);
    let mut first: Option<String> = None;
    for u in 0..units {
        let g = &got[u * ub..(u + 1) * ub];
        let e = &exp[u * ub..(u + 1) * ub];
        if g == e {
            continue;
        }
        bad += 1;
        let st = stale(u);
        let class = classify(g, e, st.as_deref());
        match class {
            "POISON" => p += 1,
            "STALE" => s += 1,
            _ => o += 1,
        }
        if first.is_none() {
            first = Some(format!("{class} {}", unit_diff(u, g, e)));
        }
    }
    (bad > 0).then(|| {
        format!(
            "{bad} bad unit(s) of {units} [POISON={p} STALE={s} OTHER={o}]; first: {}",
            first.unwrap_or_default()
        )
    })
}

/// Negative control: every non-owned live unit must read back as a full poison unit
/// (the destination was gathered BEFORE any publication); the rest must be untouched.
fn check_control(
    g: &Geom,
    owner: &[Option<usize>],
    r: usize,
    got: &[u8],
    staged: &[u8],
) -> Option<String> {
    let ub = g.unit_bytes;
    let pw = POISON_WORD.to_le_bytes();
    let (mut detected, mut undetected, mut corrupt) = (0usize, 0usize, 0usize);
    let mut first: Option<String> = None;
    for u in 0..g.units {
        let gu = &got[u * ub..(u + 1) * ub];
        match owner[u] {
            Some(o) if o != r => {
                if gu.chunks_exact(4).all(|w| w == pw.as_slice()) {
                    detected += 1;
                } else {
                    undetected += 1;
                    if first.is_none() {
                        first = Some(format!(
                            "not POISON: {}",
                            unit_diff(u, gu, &poison_unit(ub))
                        ));
                    }
                }
            }
            _ => {
                if gu != &staged[u * ub..(u + 1) * ub] {
                    corrupt += 1;
                    if first.is_none() {
                        first = Some(format!(
                            "untouched unit changed: {}",
                            unit_diff(u, gu, &staged[u * ub..(u + 1) * ub])
                        ));
                    }
                }
            }
        }
    }
    if detected == 0 && undetected == 0 && corrupt == 0 {
        return Some("no non-owned live unit to detect".into());
    }
    (undetected > 0 || corrupt > 0).then(|| format!("{undetected} non-owned unit(s) not detected as POISON, {corrupt} owned/dead unit(s) corrupted ({detected} detected); first: {}", first.unwrap_or_default()))
}

fn fmt_us(v: &[f64]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn pctile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// (median, p10, p90)
fn summarize(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (pctile(&s, 0.5), pctile(&s, 0.1), pctile(&s, 0.9))
}

fn median(v: &[f64]) -> f64 {
    summarize(v).0
}

fn src_at(q: usize, a: usize) -> [usize; 8] {
    let mut s = [0usize; 8];
    s[q] = a;
    s
}

// ───────────────────────────── pass specs ─────────────────────────────

struct PassSpec {
    geom: Geom,
    uneven: bool,
    tag: u64,
    desc: String,
}

impl PassSpec {
    fn plan(&self, n: usize, seed: u64, round: usize) -> Plan {
        let s = splitmix(seed ^ self.tag.wrapping_mul(0x9E37_79B9) ^ ((round as u64) << 24));
        match self.geom.kind {
            Kind::Tiles => {
                let table = if self.uneven {
                    table_uneven(n)
                } else {
                    table_even(n)
                };
                plan_tiles(tile_ids(s, self.geom.units), table)
            }
            kind => plan_rows(kind, self.geom.units, n, self.uneven, s),
        }
    }
}

fn rows_specs() -> Vec<PassSpec> {
    let mk = |kind: Kind, slots: usize, n16: usize, uneven: bool, tag: u64| {
        let kname = match (kind, uneven) {
            (Kind::Modulo, _) => "modulo",
            (_, true) => "slot_owner-uneven",
            _ => "slot_owner",
        };
        PassSpec {
            geom: Geom::rows(kind, slots, n16),
            uneven,
            tag,
            desc: format!("slots={slots} n16={n16} {kname}"),
        }
    };
    vec![
        mk(Kind::SlotOwner, 37, 640, false, 0x1001),
        mk(Kind::SlotOwner, 37, 257, false, 0x1002),
        mk(Kind::SlotOwner, 10, 640, false, 0x1003),
        mk(Kind::Modulo, 37, 257, false, 0x1004),
        mk(Kind::SlotOwner, 37, 640, true, 0x1005),
    ]
}

fn tiles_specs() -> Vec<PassSpec> {
    let mut v = Vec::new();
    let mut tag = 0x2001;
    for row_u4 in [64usize, 320] {
        for uneven in [false, true] {
            v.push(PassSpec {
                geom: Geom::tiles(PULL_TILES, row_u4),
                uneven,
                tag,
                desc: format!(
                    "tiles={PULL_TILES} row_u4={row_u4} table={}",
                    if uneven { "uneven" } else { "even" }
                ),
            });
            tag += 1;
        }
    }
    v
}

// ───────────────────────────── rank environment ─────────────────────────────

#[derive(Clone, Copy)]
enum Which {
    Ready,
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Baseline,
    Wall,
    Kernel,
    /// Step 5 via `hipStreamWaitEvent` on the rank stream instead of host event syncs.
    WallSw,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    Mapped,
    MappedSw,
    MappedChain,
    Staged,
    Peer,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Mapped => "mapped",
            Method::MappedSw => "mapped-sw",
            Method::MappedChain => "mapped-sw-chain",
            Method::Staged => "staged",
            Method::Peer => "peer",
        }
    }
}

struct Pass {
    geom: Geom,
    y: GpuTensor,
    /// rows: owners / expert ids; tiles: tile ids.
    t0: GpuTensor,
    /// rows: publish owners; tiles: expert owner table.
    t1: GpuTensor,
    /// tiles: publish table (rows: unused).
    t2: GpuTensor,
    /// Host transport: non-owning tensors over this rank's alias of each bank.
    bank_t: Vec<GpuTensor>,
    alias: [usize; 2],
}

struct HopBufs {
    a: GpuTensor,
    b: GpuTensor,
    own0: GpuTensor,
    own1: GpuTensor,
}

struct Env<'a> {
    sh: &'a Shared,
    rank: usize,
    device: i32,
    gpu: Gpu,
    ready: Vec<Box<Event>>,
    done: Vec<Box<Event>>,
    /// Timing events (default flags): publish start/stop, gather start/stop.
    kt: Vec<Event>,
    /// Host transport: every pointer a kernel of this rank may be handed.
    allowed: HashSet<usize>,
    guard_on: bool,
    peers_enabled: bool,
    tile_ok: bool,
}

impl<'a> Env<'a> {
    fn init(sh: &'a Shared, rank: usize, slot: &mut Option<Env<'a>>) -> R<()> {
        let device = sh.cfg.devices[rank];
        let _g = lock(&sh.init_lock);
        let mut gpu = ctx(
            Gpu::init_with_device(device),
            &format!("init device {device} (rank {rank})"),
        )?;
        ctx(gpu.bind_thread(), "bind_thread")?;
        let stream = ctx(gpu.hip.stream_create(), "stream_create")?;
        gpu.active_stream = Some(stream);
        let mut can_access = Vec::with_capacity(sh.n);
        for (q, &d) in sh.cfg.devices.iter().enumerate() {
            can_access.push(if q == rank {
                None
            } else {
                Some(ctx(gpu.hip.can_access_peer(device, d), "can_access_peer")?)
            });
        }
        {
            let mut info = lock(&sh.infos[rank]);
            info.arch = gpu.arch.clone();
            info.pci = gpu.hip.device_pci_bus_id(device).ok();
            info.can_access = can_access;
        }
        *slot = Some(Env {
            sh,
            rank,
            device,
            gpu,
            ready: Vec::new(),
            done: Vec::new(),
            kt: Vec::new(),
            allowed: HashSet::new(),
            guard_on: false,
            peers_enabled: false,
            tile_ok: false,
        });
        let env = slot.as_mut().ok_or("env slot")?;
        let flags = HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM;
        for b in 0..2 {
            let ev = Box::new(ctx(
                env.gpu.hip.event_create_with_flags(flags),
                "event_create ready",
            )?);
            sh.ready[rank][b].store(&*ev as *const Event as *mut Event, Ordering::Release);
            env.ready.push(ev);
        }
        for b in 0..2 {
            let ev = Box::new(ctx(
                env.gpu.hip.event_create_with_flags(flags),
                "event_create done",
            )?);
            sh.done[rank][b].store(&*ev as *const Event as *mut Event, Ordering::Release);
            env.done.push(ev);
        }
        for _ in 0..4 {
            let ev = ctx(env.gpu.hip.event_create(), "event_create timing")?;
            env.kt.push(ev);
        }
        Ok(())
    }

    fn bind(&self) -> R<()> {
        ctx(self.gpu.bind_thread(), "bind_thread")
    }

    fn stream_sync(&self) -> R<()> {
        self.bind()?;
        let s = self
            .gpu
            .active_stream
            .as_ref()
            .ok_or("rank has no launch stream")?;
        ctx(
            self.gpu.hip.stream_synchronize(s),
            &format!("stream_synchronize rank {}", self.rank),
        )
    }

    fn vote(&self, local_fail: bool) -> R<bool> {
        let sh = self.sh;
        if local_fail {
            sh.vote.fetch_add(1, Ordering::SeqCst);
        }
        sh.barrier.wait()?;
        let any = sh.vote.load(Ordering::SeqCst) > 0;
        sh.barrier.wait()?;
        if self.rank == 0 {
            sh.vote.store(0, Ordering::SeqCst);
        }
        sh.barrier.wait()?;
        Ok(any)
    }

    // ── memory ──

    fn upload(&mut self, bytes: &[u8], what: &str) -> R<GpuTensor> {
        self.bind()?;
        let t = ctx(self.gpu.upload_raw(bytes, &[bytes.len()]), what)?;
        self.allowed.insert(addr(&t));
        Ok(t)
    }

    fn free_tensor(&mut self, t: GpuTensor) -> R<()> {
        self.allowed.remove(&addr(&t));
        self.bind()?;
        ctx(self.gpu.hip.free(t.buf), "free tensor")
    }

    fn htod(&self, t: &GpuTensor, data: &[u8]) -> R<()> {
        self.bind()?;
        let s = self
            .gpu
            .active_stream
            .as_ref()
            .ok_or("rank has no launch stream")?;
        ctx(
            self.gpu.hip.memcpy_htod_async(&t.buf, data, s),
            "memcpy_htod_async",
        )
    }

    fn dtoh(&self, t: &GpuTensor, len: usize) -> R<Vec<u8>> {
        self.stream_sync()?;
        let mut v = vec![0u8; len];
        ctx(
            self.gpu
                .hip
                .memcpy_dtoh_async(&mut v, &t.buf, self.gpu.active_stream.as_ref()),
            "memcpy_dtoh_async",
        )?;
        self.stream_sync()?;
        Ok(v)
    }

    // ── events ──

    fn peer_event(slot: &AtomicPtr<Event>) -> R<&Event> {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            return Err("peer event not created".into());
        }
        // SAFETY: the owning rank keeps the boxed event alive until the teardown
        // barrier, which every rank passes only after its last protocol step.
        Ok(unsafe { &*p })
    }

    fn slot(&self, w: Which, q: usize, b: usize) -> &AtomicPtr<Event> {
        match w {
            Which::Ready => &self.sh.ready[q][b],
            Which::Done => &self.sh.done[q][b],
        }
    }

    fn record(&self, w: Which, b: usize) -> R<()> {
        self.bind()?;
        let ev: &Event = match w {
            Which::Ready => &self.ready[b],
            Which::Done => &self.done[b],
        };
        ctx(
            self.gpu
                .hip
                .event_record(ev, self.gpu.active_stream.as_ref()),
            "event_record",
        )
    }

    fn sync_event(&self, w: Which, q: usize, b: usize) -> R<()> {
        self.bind()?;
        let ev = Self::peer_event(self.slot(w, q, b))?;
        ctx(self.gpu.hip.event_synchronize(ev), "event_synchronize")
    }

    /// Host-synchronize every rank's event (ascending), optionally including this rank.
    fn sync_set(&self, w: Which, b: usize, include_self: bool) -> R<()> {
        for q in 0..self.sh.n {
            if q != self.rank || include_self {
                self.sync_event(w, q, b)?;
            }
        }
        Ok(())
    }

    /// Device-side wait: enqueue `hipStreamWaitEvent` for every OTHER rank's event.
    fn wait_set(&self, w: Which, b: usize) -> R<()> {
        self.bind()?;
        let s = self
            .gpu
            .active_stream
            .as_ref()
            .ok_or("rank has no launch stream")?;
        for q in 0..self.sh.n {
            if q != self.rank {
                let ev = Self::peer_event(self.slot(w, q, b))?;
                ctx(self.gpu.hip.stream_wait_event(s, ev), "stream_wait_event")?;
            }
        }
        Ok(())
    }

    fn wait_one(&self, w: Which, q: usize, b: usize, device_wait: bool) -> R<()> {
        if device_wait {
            self.bind()?;
            let s = self
                .gpu
                .active_stream
                .as_ref()
                .ok_or("rank has no launch stream")?;
            let ev = Self::peer_event(self.slot(w, q, b))?;
            ctx(self.gpu.hip.stream_wait_event(s, ev), "stream_wait_event")
        } else {
            self.sync_event(w, q, b)
        }
    }

    fn wait_seq(&self, target: usize) -> R<()> {
        let sh = self.sh;
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            if sh.hop_seq.load(Ordering::Acquire) >= target {
                return Ok(());
            }
            if sh.barrier.is_aborted() {
                return Err("hop wait aborted".into());
            }
            if spins < 2048 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            if start.elapsed() >= BARRIER_TIMEOUT {
                sh.barrier.abort();
                return Err("hop wait timed out".into());
            }
            std::thread::yield_now();
        }
    }

    // ── guarded launches ──

    fn guard(&self, what: &str, ptrs: &[usize]) -> R<()> {
        if !self.guard_on {
            return Ok(());
        }
        for &p in ptrs {
            if p != 0 && !self.allowed.contains(&p) {
                return Err(format!(
                    "GUARD {what}: pointer {p:#x} is neither a rank-{} allocation nor its host-bank alias; launch refused",
                    self.rank
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rows_slot_owner(
        &mut self,
        owner: &GpuTensor,
        src: [usize; 8],
        dst: usize,
        slots: usize,
        ranks: usize,
        self_rank: usize,
        n16: usize,
    ) -> R<()> {
        let mut p = vec![addr(owner), dst];
        p.extend_from_slice(&src);
        self.guard("ep_gather_rows_by_slot_owner", &p)?;
        ctx(
            self.gpu
                .ep_gather_rows_by_slot_owner(owner, src, dst, slots, ranks, self_rank, n16),
            "ep_gather_rows_by_slot_owner",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rows_modulo(
        &mut self,
        ids: &GpuTensor,
        src: [usize; 8],
        dst: usize,
        slots: usize,
        ranks: usize,
        self_rank: usize,
        n16: usize,
    ) -> R<()> {
        let mut p = vec![addr(ids), dst];
        p.extend_from_slice(&src);
        self.guard("ep_gather_rows_by_owner", &p)?;
        ctx(
            self.gpu
                .ep_gather_rows_by_owner(ids, src, dst, slots, ranks, self_rank, n16, EXPERTS),
            "ep_gather_rows_by_owner",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn tiles(
        &mut self,
        ids: &GpuTensor,
        table: &GpuTensor,
        sources: [usize; 8],
        dst: &GpuTensor,
        tiles: usize,
        ranks: usize,
        rank: usize,
        row_u4: usize,
    ) -> R<()> {
        let mut p = vec![addr(ids), addr(table), addr(dst)];
        p.extend_from_slice(&sources);
        self.guard("ep_gather_grouped_rows_by_tile_owner", &p)?;
        ctx(
            self.gpu.ep_gather_grouped_rows_by_tile_owner(
                ids, table, sources, dst, tiles, EXPERTS, ranks, rank, row_u4,
            ),
            "ep_gather_grouped_rows_by_tile_owner",
        )
    }

    /// Publish mask: copy this rank's OWNED units from `y` into bank `b` (host transport).
    fn publish(&mut self, pass: &Pass, b: usize, kt: bool) -> R<()> {
        let g = pass.geom;
        let src = src_at(0, addr(&pass.y));
        if kt {
            self.record_kt(0)?;
        }
        match g.kind {
            Kind::Tiles => self.tiles(
                &pass.t0,
                &pass.t2,
                src,
                &pass.bank_t[b],
                g.units,
                2,
                1,
                g.row_u4,
            )?,
            _ => self.rows_slot_owner(&pass.t1, src, pass.alias[b], g.units, 2, 1, g.n16)?,
        }
        if kt {
            self.record_kt(1)?;
        }
        Ok(())
    }

    /// The real consumer gather into `y`.
    fn consume(&mut self, pass: &Pass, arm: Arm, b: usize, kt: bool) -> R<()> {
        let sh = self.sh;
        let (r, n) = (self.rank, sh.n);
        let g = pass.geom;
        let mut src = [0usize; 8];
        for (q, s) in src.iter_mut().enumerate().take(n) {
            *s = match arm {
                Arm::Host(_) => {
                    if q == r {
                        addr(&pass.y)
                    } else {
                        pass.alias[b]
                    }
                }
                Arm::Peer => sh.yaddr[q].load(Ordering::Acquire),
            };
        }
        if kt {
            self.record_kt(2)?;
        }
        match g.kind {
            Kind::SlotOwner => {
                self.rows_slot_owner(&pass.t0, src, addr(&pass.y), g.units, n, r, g.n16)?
            }
            Kind::Modulo => self.rows_modulo(&pass.t0, src, addr(&pass.y), g.units, n, r, g.n16)?,
            Kind::Tiles => self.tiles(&pass.t0, &pass.t1, src, &pass.y, g.units, n, r, g.row_u4)?,
        }
        if kt {
            self.record_kt(3)?;
        }
        Ok(())
    }

    fn record_kt(&self, i: usize) -> R<()> {
        ctx(
            self.gpu
                .hip
                .event_record(&self.kt[i], self.gpu.active_stream.as_ref()),
            "event_record timing",
        )
    }

    fn kt_ms(&self, a: usize, b: usize) -> R<f64> {
        Ok(ctx(
            self.gpu.hip.event_elapsed_ms(&self.kt[a], &self.kt[b]),
            "event_elapsed_ms",
        )? as f64)
    }

    // ── plan tables / staging ──

    fn upload_plan(&mut self, pass: &Pass, plan: &Plan, keep: &mut Vec<Vec<u8>>) -> R<()> {
        let r = self.rank;
        match pass.geom.kind {
            Kind::Tiles => {
                keep.push(i32_bytes(&plan.dev0));
                self.htod(&pass.t0, &keep[keep.len() - 1])?;
                keep.push(i32_bytes(&plan.dev1));
                self.htod(&pass.t1, &keep[keep.len() - 1])?;
                let pubt: Vec<i32> = plan
                    .dev1
                    .iter()
                    .map(|&o| if o as usize == r { 0 } else { 1 })
                    .collect();
                keep.push(i32_bytes(&pubt));
                self.htod(&pass.t2, &keep[keep.len() - 1])?;
            }
            _ => {
                keep.push(i32_bytes(&plan.dev0));
                self.htod(&pass.t0, &keep[keep.len() - 1])?;
                let pubo: Vec<i32> = plan
                    .owner
                    .iter()
                    .map(|o| if *o == Some(r) { 0 } else { -1 })
                    .collect();
                keep.push(i32_bytes(&pubo));
                self.htod(&pass.t1, &keep[keep.len() - 1])?;
            }
        }
        Ok(())
    }

    /// `keep[0]` is left holding this rank's staged bytes.
    fn stage(&mut self, pass: &Pass, plan: &Plan, dseed: u64, keep: &mut Vec<Vec<u8>>) -> R<()> {
        keep.push(staged_bytes(dseed, &pass.geom));
        self.htod(&pass.y, &keep[0])?;
        self.upload_plan(pass, plan, keep)
    }

    // ── host banks ──

    fn bank_host(&self, b: usize) -> (*mut u8, usize) {
        (
            self.sh.banks[b].host.load(Ordering::Acquire) as *mut u8,
            self.sh.banks[b].bytes.load(Ordering::Acquire),
        )
    }

    fn poison_bank(&self, b: usize) {
        let (p, n) = self.bank_host(b);
        // SAFETY: `p` is a live pinned allocation of `n` bytes (a multiple of 4) owned by
        // the bank; the reuse guard guarantees no GPU kernel is using it.
        let words = unsafe { std::slice::from_raw_parts_mut(p as *mut u32, n / 4) };
        words.fill(POISON_WORD.to_le());
        fence(Ordering::SeqCst);
    }

    fn read_bank(&self, b: usize) -> Vec<u8> {
        fence(Ordering::SeqCst);
        let (p, n) = self.bank_host(b);
        // SAFETY: live pinned allocation of `n` bytes; the caller synchronized every producer.
        let v = unsafe { std::slice::from_raw_parts(p as *const u8, n) }.to_vec();
        fence(Ordering::SeqCst);
        v
    }

    /// Collective. rank 0 `host_malloc`s both banks; every rank maps them (`mapped`).
    /// Returns `false` (identically on all ranks) after recording a FAIL case.
    fn create_banks(&mut self, bytes: usize, flags: u32, mapped: bool, tag: &str) -> R<bool> {
        let sh = self.sh;
        let (r, n) = (self.rank, sh.n);
        if r == 0 {
            *lock(&sh.alloc_err) = None;
            self.bind()?;
            for i in 0..2 {
                sh.banks[i].host.store(0, Ordering::Release);
                match self.gpu.hip.host_malloc(bytes, flags) {
                    Ok(p) => {
                        // SAFETY: fresh pinned allocation of `bytes` bytes.
                        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
                        sh.banks[i].bytes.store(bytes, Ordering::Release);
                        sh.banks[i].host.store(p as usize, Ordering::Release);
                    }
                    Err(e) => {
                        *lock(&sh.alloc_err) =
                            Some(format!("host_malloc({bytes}, {flags:#x}): {e}"));
                        break;
                    }
                }
            }
            fence(Ordering::SeqCst);
        }
        sh.barrier.wait()?;
        let mut bad = lock(&sh.alloc_err).clone();
        if bad.is_none() && mapped {
            self.bind()?;
            for i in 0..2 {
                let host = sh.banks[i].host.load(Ordering::Acquire) as *mut c_void;
                let a = self
                    .gpu
                    .hip
                    .host_get_device_pointer(host, 0)
                    .map(|p| p as usize)
                    .unwrap_or(0);
                sh.banks[i].alias[r].store(a, Ordering::Release);
            }
        }
        sh.barrier.wait()?;
        if bad.is_none() && mapped {
            'chk: for i in 0..2 {
                for q in 0..n {
                    if sh.banks[i].alias[q].load(Ordering::Acquire) == 0 {
                        bad = Some(format!(
                            "host_get_device_pointer returned null/failed for rank {q} (bank {i})"
                        ));
                        break 'chk;
                    }
                }
            }
        }
        if let Some(msg) = bad {
            if r == 0 {
                lock(&sh.rec).case(
                    format!("xalloc {tag} n={n} bytes={bytes} flags={flags:#x}"),
                    Some(msg),
                );
            }
            sh.barrier.wait()?;
            if r == 0 {
                self.free_banks()?;
            }
            sh.barrier.wait()?;
            return Ok(false);
        }
        if r == 0 {
            let mut entries = Vec::new();
            for i in 0..2 {
                let host = sh.banks[i].host.load(Ordering::Acquire);
                let alias: Vec<usize> = (0..n)
                    .map(|q| sh.banks[i].alias[q].load(Ordering::Acquire))
                    .collect();
                entries.push(json!({
                    "bank": i,
                    "host": format!("{host:#x}"),
                    "alias": alias.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>(),
                    "alias_eq_host": alias.iter().map(|&a| mapped && a == host).collect::<Vec<_>>(),
                }));
            }
            lock(&sh.rec).banks.push(json!({"tag": tag, "bytes": bytes, "flags": format!("{flags:#x}"), "mapped": mapped, "banks": entries}));
        }
        Ok(true)
    }

    /// rank 0 only: free whatever bank memory is live.
    fn free_banks(&mut self) -> R<()> {
        self.bind()?;
        let mut first: Option<String> = None;
        for i in 0..2 {
            let host = self.sh.banks[i].host.swap(0, Ordering::AcqRel);
            if host != 0 {
                if let Err(e) = self.gpu.hip.host_free(host as *mut c_void) {
                    first.get_or_insert(format!("host_free: {e}"));
                }
            }
        }
        first.map_or(Ok(()), Err)
    }

    fn enable_peers(&mut self) -> R<()> {
        self.bind()?;
        for (q, &dev) in self.sh.cfg.devices.iter().enumerate() {
            if q == self.rank {
                continue;
            }
            if !ctx(
                self.gpu.hip.can_access_peer(self.device, dev),
                "can_access_peer",
            )? {
                return Err(format!(
                    "device {} cannot access peer device {dev}",
                    self.device
                ));
            }
            ctx(self.gpu.hip.enable_peer_access(dev), "enable_peer_access")?;
        }
        self.peers_enabled = true;
        Ok(())
    }

    // ── pass lifecycle ──

    fn free_pass(&mut self, pass: Pass) -> R<()> {
        let Pass {
            y,
            t0,
            t1,
            t2,
            bank_t,
            alias,
            ..
        } = pass;
        drop(bank_t);
        for a in alias {
            self.allowed.remove(&a);
        }
        let mut first: R<()> = Ok(());
        for t in [y, t0, t1, t2] {
            let r = self.free_tensor(t);
            if first.is_ok() {
                first = r;
            }
        }
        first
    }

    fn setup_pass(&mut self, geom: &Geom, arm: Arm, tag: &str) -> R<Option<Pass>> {
        let total = geom.total();
        let (s0, s1, s2) = match geom.kind {
            Kind::Tiles => (geom.units * 4, EXPERTS * 4, EXPERTS * 4),
            _ => (geom.units * 4, geom.units * 4, 16),
        };
        let mut ts: Vec<GpuTensor> = Vec::new();
        for len in [total, s0, s1, s2] {
            match self.upload(&vec![0u8; len], "alloc pass tensor") {
                Ok(t) => ts.push(t),
                Err(e) => {
                    for t in ts.drain(..) {
                        let _ = self.free_tensor(t);
                    }
                    return Err(e);
                }
            }
        }
        let mut it = ts.into_iter();
        let (Some(y), Some(t0), Some(t1), Some(t2)) = (it.next(), it.next(), it.next(), it.next())
        else {
            return Err("internal: pass tensors".into());
        };
        let mut pass = Pass {
            geom: *geom,
            y,
            t0,
            t1,
            t2,
            bank_t: Vec::new(),
            alias: [0; 2],
        };
        match self.finish_setup(&mut pass, arm, tag) {
            Ok(true) => Ok(Some(pass)),
            Ok(false) => {
                self.free_pass(pass)?;
                Ok(None)
            }
            Err(e) => {
                let _ = self.free_pass(pass);
                Err(e)
            }
        }
    }

    fn finish_setup(&mut self, pass: &mut Pass, arm: Arm, tag: &str) -> R<bool> {
        let sh = self.sh;
        let r = self.rank;
        let total = pass.geom.total();
        if let Arm::Host(f) = arm {
            if !self.create_banks(total, f.bits(), true, tag)? {
                return Ok(false);
            }
            for b in 0..2 {
                let a = sh.banks[b].alias[r].load(Ordering::Acquire);
                pass.alias[b] = a;
                self.allowed.insert(a);
                // SAFETY: non-owning view over this rank's alias of the pinned bank; never
                // freed through this tensor and never outlives the bank (freed after the
                // teardown barrier).
                pass.bank_t.push(GpuTensor {
                    buf: unsafe { DeviceBuffer::from_raw(a as *mut c_void, total) },
                    shape: vec![total],
                    dtype: DType::Raw,
                });
            }
        }
        sh.yaddr[r].store(addr(&pass.y), Ordering::Release);
        sh.barrier.wait()?;
        if arm == Arm::Peer {
            self.enable_peers()?;
        }
        Ok(true)
    }

    fn teardown_pass(&mut self, pass: Pass, arm: Arm) -> R<()> {
        let sh = self.sh;
        let sync = self.stream_sync();
        let b1 = sh.barrier.wait();
        let mut res = sync.and(b1.clone());
        if b1.is_ok() {
            if arm.is_host() {
                if self.rank == 0 {
                    let f = self.free_banks();
                    res = res.and(f);
                }
                res = res.and(sh.barrier.wait());
            }
            res = res.and(self.free_pass(pass));
        }
        // On an aborted barrier peers may still read these buffers: leave them to process exit.
        res
    }

    fn run_pass<F>(&mut self, geom: &Geom, arm: Arm, tag: &str, body: F) -> R<()>
    where
        F: FnOnce(&mut Self, &Pass) -> R<()>,
    {
        let Some(pass) = self.setup_pass(geom, arm, tag)? else {
            return Ok(());
        };
        let res = body(self, &pass);
        let td = self.teardown_pass(pass, arm);
        res.and(td)
    }

    // ── correctness round ──

    #[allow(clippy::too_many_arguments)]
    fn verify_bank(
        &self,
        pass: &Pass,
        b: usize,
        plan: &Plan,
        dseeds: &[u64],
        prev: Option<(&Plan, &[u64])>,
    ) -> Option<String> {
        let g = pass.geom;
        let ub = g.unit_bytes;
        let got = self.read_bank(b);
        let mut exp = Vec::with_capacity(g.total());
        for u in 0..g.units {
            match plan.owner[u] {
                Some(o) => exp.extend_from_slice(&unit_data(dseeds[o], u, ub)),
                None => exp.extend_from_slice(&poison_unit(ub)),
            }
        }
        let stale =
            |u: usize| prev.and_then(|(pp, pd)| pp.owner[u].map(|o| unit_data(pd[o], u, ub)));
        check_units(g.units, ub, &got, &exp, &stale)
    }

    /// One exchange with full verification. `sw`: step 5 via stream waits.
    #[allow(clippy::too_many_arguments)]
    fn correctness_round(
        &mut self,
        pass: &Pass,
        arm: Arm,
        k: usize,
        sw: bool,
        plan: &Plan,
        dseeds: &[u64],
        prev: Option<(&Plan, &[u64])>,
        label: &str,
        desc: &str,
        out: &mut Vec<(String, Option<String>)>,
    ) -> R<()> {
        let sh = self.sh;
        let r = self.rank;
        let b = k % 2;
        let g = pass.geom;
        let mut keep: Vec<Vec<u8>> = Vec::new();
        match arm {
            Arm::Host(_) => {
                if k >= 2 {
                    self.sync_set(Which::Done, b, true)?;
                }
                if r == 0 {
                    self.poison_bank(b);
                }
                sh.barrier.wait()?;
                self.stage(pass, plan, dseeds[r], &mut keep)?;
                self.publish(pass, b, false)?;
                self.record(Which::Ready, b)?;
            }
            Arm::Peer => {
                if k >= 1 {
                    self.sync_set(Which::Done, (k - 1) % 2, false)?;
                }
                self.stage(pass, plan, dseeds[r], &mut keep)?;
                self.record(Which::Ready, b)?;
            }
        }
        sh.barrier.wait()?;
        if sw {
            self.wait_set(Which::Ready, b)?;
        } else {
            self.sync_set(Which::Ready, b, false)?;
        }
        if arm.is_host() && r == 0 {
            // The host-side bank read needs EVERY publication complete on the host
            // (also in the `-sw` variant, where the other ranks only wait device-side).
            self.sync_set(Which::Ready, b, true)?;
            let d = self.verify_bank(pass, b, plan, dseeds, prev);
            out.push((format!("{label} round={k} rank=bank {desc}"), d));
        }
        self.consume(pass, arm, b, false)?;
        self.record(Which::Done, b)?;
        let got = self.dtoh(&pass.y, g.total())?;
        let ub = g.unit_bytes;
        let mut exp = keep[0].clone();
        for u in 0..g.units {
            if let Some(o) = plan.owner[u] {
                if o != r {
                    exp[u * ub..(u + 1) * ub].copy_from_slice(&unit_data(dseeds[o], u, ub));
                }
            }
        }
        let stale =
            |u: usize| prev.and_then(|(pp, pd)| pp.owner[u].map(|o| unit_data(pd[o], u, ub)));
        let d = check_units(g.units, ub, &got, &exp, &stale);
        out.push((format!("{label} round={k} rank={r} {desc}"), d));
        Ok(())
    }

    /// xctl negative control: gather BEFORE any publication (bank freshly poisoned).
    fn control_round(
        &mut self,
        pass: &Pass,
        plan: &Plan,
        dseeds: &[u64],
        label: &str,
        desc: &str,
        out: &mut Vec<(String, Option<String>)>,
    ) -> R<()> {
        let sh = self.sh;
        let r = self.rank;
        let g = pass.geom;
        let mut keep: Vec<Vec<u8>> = Vec::new();
        let arm = Arm::Host(HostFlags::Mapped); // only `is_host` matters to `consume`
        if r == 0 {
            self.poison_bank(0);
        }
        sh.barrier.wait()?;
        self.stage(pass, plan, dseeds[r], &mut keep)?;
        self.consume(pass, arm, 0, false)?;
        let got = self.dtoh(&pass.y, g.total())?;
        // Every gather has completed before any publication is launched.
        sh.barrier.wait()?;
        self.publish(pass, 0, false)?;
        self.record(Which::Ready, 0)?;
        sh.barrier.wait()?;
        self.sync_set(Which::Ready, 0, false)?;
        self.record(Which::Done, 0)?;
        self.stream_sync()?;
        let d = check_control(&g, &plan.owner, r, &got, &keep[0]);
        out.push((format!("{label} control rank={r} {desc}"), d));
        Ok(())
    }

    // ── timing loops ──

    fn timed_loop(
        &mut self,
        pass: &Pass,
        arm: Arm,
        k0: usize,
        total: usize,
        mode: Mode,
    ) -> R<TimedOut> {
        let sh = self.sh;
        let kernels = mode != Mode::Baseline;
        let kt = mode == Mode::Kernel;
        let sw = mode == Mode::WallSw;
        let mut out = TimedOut::default();
        for i in 0..WARMUPS + total {
            let k = k0 + i;
            let b = k % 2;
            sh.barrier.wait()?;
            let t0 = Instant::now();
            match arm {
                Arm::Host(_) => {
                    if k >= 2 {
                        self.sync_set(Which::Done, b, true)?;
                    }
                    if kernels {
                        self.publish(pass, b, kt)?;
                    }
                    self.record(Which::Ready, b)?;
                    sh.barrier.wait()?;
                    if sw {
                        self.wait_set(Which::Ready, b)?;
                    } else {
                        self.sync_set(Which::Ready, b, false)?;
                    }
                    if kernels {
                        self.consume(pass, arm, b, kt)?;
                    }
                    self.record(Which::Done, b)?;
                }
                Arm::Peer => {
                    if k >= 1 {
                        self.sync_set(Which::Done, (k - 1) % 2, false)?;
                    }
                    self.record(Which::Ready, b)?;
                    sh.barrier.wait()?;
                    if sw {
                        self.wait_set(Which::Ready, b)?;
                    } else {
                        self.sync_set(Which::Ready, b, false)?;
                    }
                    if kernels {
                        self.consume(pass, arm, b, kt)?;
                    }
                    self.record(Which::Done, b)?;
                }
            }
            self.stream_sync()?;
            let dt = t0.elapsed().as_secs_f64() * 1e6;
            if i >= WARMUPS {
                out.durs.push(dt);
                if kt {
                    if arm.is_host() {
                        out.pubs.push(self.kt_ms(0, 1)? * 1e3);
                    }
                    out.gats.push(self.kt_ms(2, 3)? * 1e3);
                }
            }
        }
        Ok(out)
    }

    /// Only this rank publishes + gathers (bank 0); others idle at the caller's barriers.
    fn solo_loop(&mut self, pass: &Pass, arm: Arm, total: usize, kt: bool) -> R<TimedOut> {
        let mut out = TimedOut::default();
        for i in 0..WARMUPS + total {
            let t0 = Instant::now();
            if arm.is_host() {
                self.publish(pass, 0, kt)?;
                self.record(Which::Ready, 0)?;
                self.sync_event(Which::Ready, self.rank, 0)?;
            }
            self.consume(pass, arm, 0, kt)?;
            self.record(Which::Done, 0)?;
            self.stream_sync()?;
            let dt = t0.elapsed().as_secs_f64() * 1e6;
            if i >= WARMUPS {
                out.durs.push(dt);
                if kt {
                    if arm.is_host() {
                        out.pubs.push(self.kt_ms(0, 1)? * 1e3);
                    }
                    out.gats.push(self.kt_ms(2, 3)? * 1e3);
                }
            }
        }
        Ok(out)
    }

    /// Collective: rank 0 receives every rank's vectors.
    fn collect(&self, t: TimedOut) -> R<Option<Vec<TimedOut>>> {
        let sh = self.sh;
        *lock(&sh.times[self.rank]) = t;
        sh.barrier.wait()?;
        let res = (self.rank == 0).then(|| {
            (0..sh.n)
                .map(|q| lock(&sh.times[q]).clone())
                .collect::<Vec<_>>()
        });
        sh.barrier.wait()?;
        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_time(
        &self,
        suite: &str,
        shape: &str,
        tag: &str,
        durs: &[f64],
        pubs: &[f64],
        gats: &[f64],
        gbps: Option<f64>,
        tokens: &str,
        extra: Value,
    ) {
        let (med, p10, p90) = summarize(durs);
        let mut line = format!(
            "TIME {suite} {shape} {tag} median_us={med:.1} p10_us={p10:.1} p90_us={p90:.1}"
        );
        if let Some(g) = gbps {
            line.push_str(&format!(" GBps={g:.3}"));
        }
        line.push_str(&format!(
            " publish_us={} gather_us={}",
            fmt_us(pubs),
            fmt_us(gats)
        ));
        if !tokens.is_empty() {
            line.push(' ');
            line.push_str(tokens);
        }
        let mut rec = lock(&self.sh.rec);
        println!("{line}");
        rec.timings.push(json!({
            "suite": suite, "shape": shape, "transport": tag,
            "median_us": med, "p10_us": p10, "p90_us": p90, "samples": durs.len(),
            "GBps": gbps, "publish_us": pubs, "gather_us": gats, "extra": extra,
        }));
    }

    // ── kernel warm-up (serialized JIT) ──

    fn warm_once(&mut self) -> R<()> {
        let n = self.sh.n;
        let ids = self.upload(&i32_bytes(&[-1; 64]), "warm ids")?;
        let tbl = self.upload(&i32_bytes(&vec![0; EXPERTS]), "warm table")?;
        let tid = self.upload(&i32_bytes(&[-1]), "warm tile ids")?;
        let dst = self.upload(&vec![0u8; 16384], "warm dst")?;
        let src = self.upload(&vec![0u8; 16384], "warm src")?;
        let (d, s) = (addr(&dst), addr(&src));
        let res = (|| -> R<()> {
            self.rows_modulo(&ids, src_at(1, s), d, 1, 2, 0, 1)?;
            self.rows_slot_owner(&ids, src_at(1, s), d, 1, 2, 0, 1)?;
            if self.tile_ok {
                let mut srcs = [s; 8];
                srcs[0] = d;
                self.tiles(&tid, &tbl, srcs, &dst, 1, 2, 0, 64)?;
                if n > 2 {
                    self.tiles(&tid, &tbl, srcs, &dst, 1, n, 0, 64)?;
                }
            }
            self.stream_sync()
        })();
        for t in [ids, tbl, tid, dst, src] {
            let f = self.free_tensor(t);
            if res.is_ok() {
                f?;
            }
        }
        res
    }

    fn warm_kernels(&mut self) -> R<()> {
        for t in 0..self.sh.n {
            if t == self.rank {
                self.warm_once()?;
            }
            self.sh.barrier.wait()?;
        }
        Ok(())
    }
}

// ───────────────────────────── suites ─────────────────────────────

fn rounds_pass(env: &mut Env<'_>, suite: &str, spec: &PassSpec, arm: Arm) -> R<()> {
    let sh = env.sh;
    let n = sh.n;
    let label = format!("{suite} {} n={n}", arm.tag());
    env.run_pass(&spec.geom, arm, &label, |env, pass| {
        let mut hist: Vec<(Plan, Vec<u64>)> = Vec::new();
        for k in 0..sh.cfg.rounds {
            let plan = spec.plan(n, sh.cfg.seed, k);
            let dseeds: Vec<u64> = (0..n)
                .map(|q| data_seed(sh.cfg.seed, spec.tag, k, q))
                .collect();
            let prev = (k >= 2).then(|| (&hist[k - 2].0, &hist[k - 2].1[..]));
            let mut out = Vec::new();
            env.correctness_round(
                pass, arm, k, false, &plan, &dseeds, prev, &label, &spec.desc, &mut out,
            )?;
            for (name, detail) in out {
                lock(&sh.rec).case(name, detail);
            }
            hist.push((plan, dseeds));
        }
        Ok(())
    })
}

fn ctl_pass(env: &mut Env<'_>, spec: &PassSpec, arm: Arm) -> R<()> {
    let sh = env.sh;
    let n = sh.n;
    let label = format!("xctl {} n={n}", arm.tag());
    env.run_pass(&spec.geom, arm, &label, |env, pass| {
        let mut hist: Vec<(Plan, Vec<u64>)> = Vec::new();
        for k in 0..3usize {
            let plan = spec.plan(n, sh.cfg.seed, k);
            let dseeds: Vec<u64> = (0..n)
                .map(|q| data_seed(sh.cfg.seed, spec.tag, k, q))
                .collect();
            let mut out = Vec::new();
            if k == 0 {
                env.control_round(pass, &plan, &dseeds, &label, &spec.desc, &mut out)?;
            } else {
                // Round 2 reuses bank 0, the bank the control round left half-published.
                let prev = (k >= 2).then(|| (&hist[k - 2].0, &hist[k - 2].1[..]));
                env.correctness_round(
                    pass,
                    arm,
                    k,
                    false,
                    &plan,
                    &dseeds,
                    prev,
                    &format!("{label} recovery"),
                    &spec.desc,
                    &mut out,
                )?;
            }
            for (name, detail) in out {
                lock(&sh.rec).case(name, detail);
            }
            hist.push((plan, dseeds));
        }
        Ok(())
    })
}

struct Shape {
    name: &'static str,
    geom: Geom,
    plan: Plan,
    total: usize,
    desc: String,
    /// Also run the stream-wait (`-sw`) arm.
    sw: bool,
    solo: bool,
}

fn time_shape(env: &mut Env<'_>, pass: &Pass, arm: Arm, s: &Shape, idx: usize) -> R<()> {
    let sh = env.sh;
    let (n, r) = (sh.n, env.rank);
    let tag = arm.tag();
    let label = format!("xtime {tag} n={n} shape={}", s.name);
    let dseeds: Vec<u64> = (0..n)
        .map(|q| data_seed(sh.cfg.seed, 0x7100 + idx as u64, 0, q))
        .collect();
    // Verified warm round(s): must PASS or the shape is FAIL.
    let mut k = 0usize;
    let mut bad = false;
    for sw in [false, true] {
        if sw && !s.sw {
            break;
        }
        let mut out = Vec::new();
        let lab = format!("{label} warm{}", if sw { "-sw" } else { "" });
        env.correctness_round(
            pass, arm, k, sw, &s.plan, &dseeds, None, &lab, &s.desc, &mut out,
        )?;
        k += 1;
        for (name, detail) in out {
            bad |= detail.is_some();
            lock(&sh.rec).case(name, detail);
        }
    }
    if env.vote(bad)? {
        if r == 0 {
            lock(&sh.rec).note(format!(
                "{label}: verified warm round FAILED; timing skipped"
            ));
        }
        return Ok(());
    }
    let total = s.total;
    let base = env.timed_loop(pass, arm, k, total, Mode::Baseline)?;
    k += WARMUPS + total;
    let kern = env.timed_loop(pass, arm, k, total, Mode::Kernel)?;
    k += WARMUPS + total;
    let wall = env.timed_loop(pass, arm, k, total, Mode::Wall)?;
    k += WARMUPS + total;
    let swp = if s.sw {
        Some(env.timed_loop(pass, arm, k, total, Mode::WallSw)?)
    } else {
        None
    };
    let base_all = env.collect(base)?;
    let kern_all = env.collect(kern)?;
    let wall_all = env.collect(wall)?;
    let sw_all = match swp {
        Some(t) => env.collect(t)?,
        None => None,
    };
    let ub = s.geom.unit_bytes;
    let live = s.plan.owner.iter().flatten().count();
    let (hostward, deviceward) = if arm.is_host() {
        (live * ub, (n - 1) * live * ub)
    } else {
        (0, (n - 1) * live * ub)
    };
    let is_tiles = s.geom.kind == Kind::Tiles;
    let bytes = json!({"hostward": hostward, "deviceward": deviceward, "live_units": live, "unit_bytes": ub});
    if r == 0 {
        let max_over = |all: &[TimedOut]| -> Vec<f64> {
            (0..all[0].durs.len())
                .map(|i| all.iter().map(|t| t.durs[i]).fold(0.0, f64::max))
                .collect()
        };
        let per_rank_med = |all: &[TimedOut], f: fn(&TimedOut) -> &Vec<f64>| -> Vec<f64> {
            all.iter()
                .filter(|t| !f(t).is_empty())
                .map(|t| median(f(t)))
                .collect()
        };
        if let (Some(base_all), Some(kern_all), Some(wall_all)) = (&base_all, &kern_all, &wall_all)
        {
            let pubs = per_rank_med(kern_all, |t| &t.pubs);
            let gats = per_rank_med(kern_all, |t| &t.gats);
            let bd = max_over(base_all);
            env.emit_time(
                "xtime",
                &format!("{}-baseline", s.name),
                &tag,
                &bd,
                &[],
                &[],
                None,
                "",
                json!({"note": "protocol only, no kernels"}),
            );
            let wd = max_over(wall_all);
            let gbps =
                is_tiles.then(|| (hostward + deviceward) as f64 / (median(&wd) * 1e-6) / 1e9);
            env.emit_time(
                "xtime",
                s.name,
                &tag,
                &wd,
                &pubs,
                &gats,
                gbps,
                "",
                json!({"bytes": bytes, "baseline_median_us": median(&bd), "per_rank_median_us": per_rank_med(wall_all, |t| &t.durs), "exchanges": s.total}),
            );
            if let Some(sw_all) = &sw_all {
                let sd = max_over(sw_all);
                env.emit_time(
                    "xtime",
                    &format!("{}-sw", s.name),
                    &tag,
                    &sd,
                    &pubs,
                    &gats,
                    None,
                    "",
                    json!({"bytes": bytes, "exchanges": s.total, "step5": "hipStreamWaitEvent"}),
                );
            }
        }
    }
    if s.solo {
        let mut solo_res: Vec<Option<Vec<TimedOut>>> = Vec::new();
        for kt in [true, false] {
            let mut mine = TimedOut::default();
            for a in 0..n {
                sh.barrier.wait()?;
                if r == a {
                    mine = env.solo_loop(pass, arm, total, kt)?;
                }
                sh.barrier.wait()?;
            }
            solo_res.push(env.collect(mine)?);
        }
        if r == 0 {
            if let (Some(Some(kern)), Some(Some(wall))) = (solo_res.first(), solo_res.get(1)) {
                let pubs: Vec<f64> = kern
                    .iter()
                    .filter(|t| !t.pubs.is_empty())
                    .map(|t| median(&t.pubs))
                    .collect();
                let gats: Vec<f64> = kern
                    .iter()
                    .filter(|t| !t.gats.is_empty())
                    .map(|t| median(&t.gats))
                    .collect();
                let pooled: Vec<f64> = wall.iter().flat_map(|t| t.durs.iter().copied()).collect();
                let rank_med: Vec<f64> = wall.iter().map(|t| median(&t.durs)).collect();
                let rank_gbps: Vec<f64> = (0..n)
                    .map(|q| {
                        let owned = s.plan.owner.iter().filter(|o| **o == Some(q)).count();
                        let host_q = if arm.is_host() { owned * ub } else { 0 };
                        let dev_q = (live - owned) * ub;
                        (host_q + dev_q) as f64 / (rank_med[q] * 1e-6) / 1e9
                    })
                    .collect();
                let gbps = median(&rank_gbps);
                let tokens = format!(
                    "rank_median_us={} rank_GBps={}",
                    fmt_us(&rank_med),
                    fmt_us(&rank_gbps)
                );
                env.emit_time(
                    "xtime",
                    &format!("{}-solo", s.name),
                    &tag,
                    &pooled,
                    &pubs,
                    &gats,
                    Some(gbps),
                    &tokens,
                    json!({"rank_median_us": rank_med, "rank_GBps": rank_gbps, "bytes_per_rank": "owned live units hostward, other live units deviceward"}),
                );
            }
        }
    }
    Ok(())
}

fn suite_xtime(env: &mut Env<'_>, arm: Arm) -> R<()> {
    let sh = env.sh;
    let n = sh.n;
    let steps = (sh.cfg.reps / 10).max(1);
    let shapes = vec![
        Shape {
            name: "decode",
            geom: Geom::rows(Kind::SlotOwner, 10, 640),
            plan: plan_balanced(10, n),
            total: sh.cfg.layers * steps,
            desc: "slots=10 n16=640 slot_owner".into(),
            sw: true,
            solo: false,
        },
        Shape {
            name: "fewrow",
            geom: Geom::rows(Kind::SlotOwner, 80, 640),
            plan: plan_balanced(80, n),
            total: sh.cfg.layers * steps,
            desc: "slots=80 n16=640 slot_owner".into(),
            sw: true,
            solo: false,
        },
        Shape {
            name: "prefill",
            geom: Geom::tiles(960, 320),
            plan: plan_tiles(
                tile_ids_live90(splitmix(sh.cfg.seed ^ 0x9F11), 960),
                table_even(n),
            ),
            total: steps,
            desc: "tiles=960 row_u4=320 table=even".into(),
            sw: false,
            solo: true,
        },
    ];
    for (idx, s) in shapes.iter().enumerate() {
        if s.geom.kind == Kind::Tiles && !env.tile_ok {
            if env.rank == 0 {
                println!(
                    "SKIP xtime {} shape={}: tile pull needs gfx1100/gfx1201",
                    arm.tag(),
                    s.name
                );
            }
            continue;
        }
        let label = format!("xtime {} n={n} shape={}", arm.tag(), s.name);
        env.run_pass(&s.geom, arm, &label, |env, pass| {
            time_shape(env, pass, arm, s, idx)
        })?;
    }
    Ok(())
}

// ── xhop ──

fn hop_modes(method: Method) -> &'static [&'static str] {
    match method {
        Method::MappedChain => &["chain"],
        _ => &["oneway", "pingpong"],
    }
}

fn hop_body(
    env: &mut Env<'_>,
    method: Method,
    payload: usize,
    bufs: &Option<HopBufs>,
    data: &[u8],
    tag: &str,
) -> R<()> {
    let sh = env.sh;
    let (n, r) = (sh.n, env.rank);
    let big = payload > 1 << 20;
    let (warm, reps) = if big {
        (2usize, (sh.cfg.reps / 10).max(1))
    } else {
        (WARMUPS, sh.cfg.reps)
    };
    let slots = payload / HOP_ROW;
    let mut seq = 0usize;
    for &mode in hop_modes(method) {
        let mut samples: Vec<f64> = Vec::new();
        let mut aborted_mode = false;
        for i in 0..warm + reps {
            sh.barrier.wait()?;
            env.hop_iter(bufs, method, mode, seq, payload, slots)?;
            seq += 6;
            sh.barrier.wait()?;
            if i == 0 {
                // Verify the payload bytes once per method/mode.
                let mut detail: Option<String> = None;
                let mut verifier = false;
                if let Some(h) = bufs {
                    let (is_v, t) = if mode == "pingpong" {
                        (r == 0, &h.b)
                    } else {
                        (r == 1, &h.a)
                    };
                    if is_v {
                        verifier = true;
                        let got = env.dtoh(t, payload)?;
                        detail = first_diff("payload", data, &got, HOP_ROW);
                    }
                }
                let failed = detail.is_some();
                if verifier {
                    lock(&sh.rec).case(
                        format!(
                            "xhop {tag} n={n} payload={payload} method={} mode={mode}",
                            method.name()
                        ),
                        detail,
                    );
                }
                if env.vote(failed)? {
                    aborted_mode = true;
                    break;
                }
            }
            if r == 0 && i >= warm {
                let ns = sh
                    .hop_t1
                    .load(Ordering::Acquire)
                    .saturating_sub(sh.hop_t0.load(Ordering::Acquire))
                    as f64;
                let div = match mode {
                    "pingpong" => 2.0,
                    "chain" => 5.0,
                    _ => 1.0,
                };
                samples.push(ns / 1e3 / div);
            }
        }
        if aborted_mode || r != 0 {
            continue;
        }
        let (med, _, _) = summarize(&samples);
        let gbps = payload as f64 / (med * 1e-6) / 1e9;
        env.emit_time(
            "xhop",
            &format!("{payload}B-{}-{mode}", method.name()),
            tag,
            &samples,
            &[],
            &[],
            Some(gbps),
            "",
            json!({"payload_bytes": payload, "method": method.name(), "mode": mode, "per": if mode == "pingpong" { "half round trip" } else if mode == "chain" { "hop (total/5)" } else { "one way" }}),
        );
    }
    Ok(())
}

impl<'a> Env<'a> {
    fn hop_iter(
        &mut self,
        bufs: &Option<HopBufs>,
        method: Method,
        mode: &str,
        seq: usize,
        payload: usize,
        slots: usize,
    ) -> R<()> {
        let Some(h) = bufs else {
            return Ok(());
        };
        let sh = self.sh;
        let r = self.rank;
        let n16 = HOP_ROW / 16;
        let al = |q: usize, b: usize| sh.banks[b].alias[q].load(Ordering::Acquire);
        let pingpong = mode == "pingpong";
        let t1 = |sh: &Shared| sh.hop_t1.store(sh.now_ns(), Ordering::Release);
        match method {
            Method::Mapped | Method::MappedSw => {
                let sw = method == Method::MappedSw;
                if r == 0 {
                    sh.hop_t0.store(sh.now_ns(), Ordering::Release);
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, addr(&h.a)),
                        al(0, 0),
                        slots,
                        2,
                        1,
                        n16,
                    )?;
                    self.record(Which::Ready, 0)?;
                    sh.hop_seq.store(seq + 1, Ordering::Release);
                    if pingpong {
                        self.wait_seq(seq + 2)?;
                        self.wait_one(Which::Ready, 1, 0, sw)?;
                        self.rows_slot_owner(
                            &h.own1,
                            src_at(1, al(0, 1)),
                            addr(&h.b),
                            slots,
                            2,
                            0,
                            n16,
                        )?;
                        self.stream_sync()?;
                        t1(sh);
                    }
                } else {
                    self.wait_seq(seq + 1)?;
                    self.wait_one(Which::Ready, 0, 0, sw)?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, al(1, 0)),
                        addr(&h.a),
                        slots,
                        2,
                        1,
                        n16,
                    )?;
                    if pingpong {
                        self.rows_slot_owner(
                            &h.own0,
                            src_at(0, addr(&h.a)),
                            al(1, 1),
                            slots,
                            2,
                            1,
                            n16,
                        )?;
                        self.record(Which::Ready, 0)?;
                        sh.hop_seq.store(seq + 2, Ordering::Release);
                        self.stream_sync()?;
                    } else {
                        self.stream_sync()?;
                        t1(sh);
                    }
                }
            }
            Method::MappedChain => {
                // 0->1->0->1->0->1: five hops, enqueued with stream waits only (no host event syncs).
                if r == 0 {
                    sh.hop_t0.store(sh.now_ns(), Ordering::Release);
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, addr(&h.a)),
                        al(0, 0),
                        slots,
                        2,
                        1,
                        n16,
                    )?; // hop 1 send
                    self.record(Which::Ready, 0)?;
                    sh.hop_seq.store(seq + 1, Ordering::Release);
                    self.wait_seq(seq + 2)?; // hop 2 recv
                    self.wait_one(Which::Ready, 1, 0, true)?;
                    self.rows_slot_owner(
                        &h.own1,
                        src_at(1, al(0, 1)),
                        addr(&h.b),
                        slots,
                        2,
                        0,
                        n16,
                    )?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, addr(&h.b)),
                        al(0, 0),
                        slots,
                        2,
                        1,
                        n16,
                    )?; // hop 3 send
                    self.record(Which::Ready, 1)?;
                    sh.hop_seq.store(seq + 3, Ordering::Release);
                    self.wait_seq(seq + 4)?; // hop 4 recv
                    self.wait_one(Which::Ready, 1, 1, true)?;
                    self.rows_slot_owner(
                        &h.own1,
                        src_at(1, al(0, 1)),
                        addr(&h.b),
                        slots,
                        2,
                        0,
                        n16,
                    )?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, addr(&h.b)),
                        al(0, 0),
                        slots,
                        2,
                        1,
                        n16,
                    )?; // hop 5 send
                    self.record(Which::Ready, 0)?;
                    sh.hop_seq.store(seq + 5, Ordering::Release);
                    self.stream_sync()?;
                } else {
                    self.wait_seq(seq + 1)?; // hop 1 recv
                    self.wait_one(Which::Ready, 0, 0, true)?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, al(1, 0)),
                        addr(&h.a),
                        slots,
                        2,
                        1,
                        n16,
                    )?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, addr(&h.a)),
                        al(1, 1),
                        slots,
                        2,
                        1,
                        n16,
                    )?; // hop 2 send
                    self.record(Which::Ready, 0)?;
                    sh.hop_seq.store(seq + 2, Ordering::Release);
                    self.wait_seq(seq + 3)?; // hop 3 recv
                    self.wait_one(Which::Ready, 0, 1, true)?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, al(1, 0)),
                        addr(&h.a),
                        slots,
                        2,
                        1,
                        n16,
                    )?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, addr(&h.a)),
                        al(1, 1),
                        slots,
                        2,
                        1,
                        n16,
                    )?; // hop 4 send
                    self.record(Which::Ready, 1)?;
                    sh.hop_seq.store(seq + 4, Ordering::Release);
                    self.wait_seq(seq + 5)?; // hop 5 recv
                    self.wait_one(Which::Ready, 0, 0, true)?;
                    self.rows_slot_owner(
                        &h.own0,
                        src_at(0, al(1, 0)),
                        addr(&h.a),
                        slots,
                        2,
                        1,
                        n16,
                    )?;
                    self.stream_sync()?;
                    t1(sh);
                }
            }
            Method::Staged => {
                let pin = |i: usize| sh.banks[i].host.load(Ordering::Acquire) as *mut u8;
                self.bind()?;
                if r == 0 {
                    sh.hop_t0.store(sh.now_ns(), Ordering::Release);
                    // SAFETY: pinned staging buffer of `payload` bytes; only this thread touches it
                    // until it publishes `hop_seq`.
                    let sl = unsafe { std::slice::from_raw_parts_mut(pin(0), payload) };
                    ctx(
                        self.gpu.hip.memcpy_dtoh_async(
                            sl,
                            &h.a.buf,
                            self.gpu.active_stream.as_ref(),
                        ),
                        "staged memcpy_dtoh_async",
                    )?;
                    self.stream_sync()?;
                    sh.hop_seq.store(seq + 1, Ordering::Release);
                    if pingpong {
                        self.wait_seq(seq + 2)?;
                        // SAFETY: as above; rank 1 finished writing before publishing `hop_seq`.
                        let sl = unsafe { std::slice::from_raw_parts(pin(1), payload) };
                        self.htod(&h.b, sl)?;
                        self.stream_sync()?;
                        t1(sh);
                    }
                } else {
                    self.wait_seq(seq + 1)?;
                    // SAFETY: rank 0 finished writing before publishing `hop_seq`.
                    let sl = unsafe { std::slice::from_raw_parts(pin(0), payload) };
                    self.htod(&h.a, sl)?;
                    if pingpong {
                        // SAFETY: only this thread touches buffer 1 until it publishes `hop_seq`.
                        let sl = unsafe { std::slice::from_raw_parts_mut(pin(1), payload) };
                        ctx(
                            self.gpu.hip.memcpy_dtoh_async(
                                sl,
                                &h.a.buf,
                                self.gpu.active_stream.as_ref(),
                            ),
                            "staged memcpy_dtoh_async",
                        )?;
                        self.stream_sync()?;
                        sh.hop_seq.store(seq + 2, Ordering::Release);
                    } else {
                        self.stream_sync()?;
                        t1(sh);
                    }
                }
            }
            Method::Peer => {
                let y0 = sh.yaddr[0].load(Ordering::Acquire);
                let y1 = sh.yaddr[1].load(Ordering::Acquire);
                if r == 0 {
                    sh.hop_t0.store(sh.now_ns(), Ordering::Release);
                    self.record(Which::Ready, 0)?;
                    sh.hop_seq.store(seq + 1, Ordering::Release);
                    if pingpong {
                        self.wait_seq(seq + 2)?;
                        self.sync_event(Which::Ready, 1, 0)?;
                        self.rows_slot_owner(&h.own1, src_at(1, y1), addr(&h.b), slots, 2, 0, n16)?;
                        self.stream_sync()?;
                        t1(sh);
                    }
                } else {
                    self.wait_seq(seq + 1)?;
                    self.sync_event(Which::Ready, 0, 0)?;
                    self.rows_slot_owner(&h.own0, src_at(0, y0), addr(&h.a), slots, 2, 1, n16)?;
                    if pingpong {
                        self.record(Which::Ready, 0)?;
                        sh.hop_seq.store(seq + 2, Ordering::Release);
                        self.stream_sync()?;
                    } else {
                        self.stream_sync()?;
                        t1(sh);
                    }
                }
            }
        }
        Ok(())
    }

    fn hop_pass(&mut self, arm: Arm, method: Method, payload: usize) -> R<()> {
        let sh = self.sh;
        let r = self.rank;
        let slots = payload / HOP_ROW;
        let tag = if method == Method::Staged {
            "host/pinned".to_string()
        } else {
            arm.tag()
        };
        let data = random_bytes(splitmix(sh.cfg.seed ^ payload as u64 ^ 0x40F), payload);
        let mut bufs: Option<HopBufs> = None;
        if r < 2 {
            let a = self.upload(
                &if r == 0 {
                    data.clone()
                } else {
                    vec![0xA5u8; payload]
                },
                "hop a",
            )?;
            let b = self.upload(&vec![0u8; payload], "hop b")?;
            let own0 = self.upload(&vec![0u8; slots * 4], "hop owner0")?;
            let own1 = self.upload(&i32_bytes(&vec![1; slots]), "hop owner1")?;
            bufs = Some(HopBufs { a, b, own0, own1 });
        }
        sh.yaddr[r].store(bufs.as_ref().map_or(0, |h| addr(&h.a)), Ordering::Release);
        let mut banked = false;
        let setup: R<bool> = (|| {
            match (method, arm) {
                (Method::Mapped | Method::MappedSw | Method::MappedChain, Arm::Host(f)) => {
                    if !self.create_banks(
                        payload,
                        f.bits(),
                        true,
                        &format!("xhop {tag} payload={payload}"),
                    )? {
                        return Ok(false);
                    }
                    banked = true;
                    for b in 0..2 {
                        let a = sh.banks[b].alias[r].load(Ordering::Acquire);
                        self.allowed.insert(a);
                    }
                }
                (Method::Staged, _) => {
                    if !self.create_banks(
                        payload,
                        0,
                        false,
                        &format!("xhop {tag} payload={payload}"),
                    )? {
                        return Ok(false);
                    }
                    banked = true;
                }
                (Method::Peer, Arm::Peer) => {
                    sh.barrier.wait()?;
                    self.enable_peers()?;
                }
                _ => return Err("internal: hop method/arm mismatch".into()),
            }
            if r == 0 {
                sh.hop_seq.store(0, Ordering::Release);
            }
            sh.barrier.wait()?;
            Ok(true)
        })();
        let res = match setup {
            Ok(true) => hop_body(self, method, payload, &bufs, &data, &tag),
            Ok(false) => Ok(()),
            Err(e) => Err(e),
        };
        // Teardown (same discipline as passes).
        let sync = self.stream_sync();
        let b1 = sh.barrier.wait();
        let mut tail = sync.and(b1.clone());
        if b1.is_ok() {
            if banked {
                if r == 0 {
                    let f = self.free_banks();
                    tail = tail.and(f);
                }
                tail = tail.and(sh.barrier.wait());
                for b in 0..2 {
                    let a = sh.banks[b].alias[r].load(Ordering::Acquire);
                    self.allowed.remove(&a);
                }
            }
            if let Some(h) = bufs {
                for t in [h.a, h.b, h.own0, h.own1] {
                    let f = self.free_tensor(t);
                    tail = tail.and(f);
                }
            }
        }
        res.and(tail)
    }
}

fn suite_xhop(env: &mut Env<'_>, arm: Arm, first_host: bool) -> R<()> {
    for &payload in &HOP_PAYLOADS {
        match arm {
            Arm::Host(_) => {
                for m in [Method::Mapped, Method::MappedSw, Method::MappedChain] {
                    env.hop_pass(arm, m, payload)?;
                }
                if first_host {
                    env.hop_pass(arm, Method::Staged, payload)?;
                }
            }
            Arm::Peer => env.hop_pass(arm, Method::Peer, payload)?,
        }
    }
    Ok(())
}

// ── driver for one rank ──

fn resolve_arms(sh: &Shared, announce: bool) -> Vec<Arm> {
    let all_peer = sh.pre().all_ok(sh.n);
    let (want_host, want_peer) = match sh.cfg.transport {
        TransportReq::Auto => (!all_peer, all_peer),
        TransportReq::Host => (true, false),
        TransportReq::Peer => (false, true),
        TransportReq::Both => (true, true),
    };
    let mut arms = Vec::new();
    if want_host {
        arms.extend(sh.cfg.host_flags.iter().map(|&f| Arm::Host(f)));
    }
    if want_peer {
        if all_peer {
            arms.push(Arm::Peer);
        } else {
            if announce {
                println!("SKIP peer transport: not every directed device pair is byte-qualified (qualify_peer)");
                lock(&sh.rec)
                    .notes
                    .push("peer arm skipped: a directed pair is not byte-qualified".into());
            }
        }
    }
    arms
}

fn run_all(env: &mut Env<'_>) -> R<()> {
    let sh = env.sh;
    sh.barrier.wait()?; // every rank initialized
    env.tile_ok = (0..sh.n).all(|q| ep_tile_pm_arch(&lock(&sh.infos[q]).arch));
    let arms = resolve_arms(sh, env.rank == 0);
    if env.rank == 0 {
        let infos: Vec<String> = (0..sh.n)
            .map(|q| {
                format!(
                    "rank{q}=dev{}:{}",
                    sh.cfg.devices[q],
                    lock(&sh.infos[q]).arch
                )
            })
            .collect();
        let mut rec = lock(&sh.rec);
        println!(
            "ep_transport_oracle devices={:?} {} transport={} ({}) arms={:?} seed={:#x}",
            sh.cfg.devices,
            infos.join(" "),
            match sh.cfg.transport {
                TransportReq::Auto => "auto",
                TransportReq::Host => "host",
                TransportReq::Peer => "peer",
                TransportReq::Both => "both",
            },
            sh.cfg.transport_src,
            arms.iter().map(|a| a.tag()).collect::<Vec<_>>(),
            sh.cfg.seed
        );
        rec.arms = arms.iter().map(|a| a.tag()).collect();
        let mut resolved: Vec<String> = Vec::new();
        for a in &arms {
            if !resolved.contains(&a.transport().to_string()) {
                resolved.push(a.transport().to_string());
            }
        }
        rec.resolved = resolved;
        if sh.pre().probed && arms.iter().any(|a| a.is_host()) {
            rec.note(
                "host arm ran after a qualify_peer probe: the probe may have enabled peer access at process level; the host arm itself never calls enable_peer_access (peer_access_enabled.host=false)"
                    .into(),
            );
        }
    }
    env.warm_kernels()?;
    let mut first_host = true;
    for &arm in &arms {
        env.guard_on = arm.is_host();
        let cfg = &sh.cfg;
        if cfg.xrows {
            for spec in rows_specs() {
                rounds_pass(env, "xrows", &spec, arm)?;
            }
        }
        if cfg.xtiles {
            if env.tile_ok {
                for spec in tiles_specs() {
                    rounds_pass(env, "xtiles", &spec, arm)?;
                }
            } else if env.rank == 0 {
                println!("SKIP xtiles {}: tile pull needs gfx1100/gfx1201", arm.tag());
            }
        }
        if cfg.xctl && arm.is_host() {
            let specs = [rows_specs().remove(0), tiles_specs().remove(0)];
            for spec in &specs {
                if spec.geom.kind == Kind::Tiles && !env.tile_ok {
                    continue;
                }
                ctl_pass(env, spec, arm)?;
            }
        }
        if cfg.xtime {
            suite_xtime(env, arm)?;
        }
        if cfg.xhop {
            suite_xhop(env, arm, first_host)?;
        }
        if arm.is_host() {
            first_host = false;
            lock(&sh.infos[env.rank]).peer_enabled_host = env.peers_enabled;
        }
    }
    lock(&sh.infos[env.rank]).peer_enabled_final = env.peers_enabled;
    if !arms.iter().any(|a| a.is_host()) {
        lock(&sh.infos[env.rank]).peer_enabled_host = false;
    }
    sh.barrier.wait()?;
    if env.rank == 0 && arms.is_empty() {
        lock(&sh.rec)
            .errors
            .push("no runnable transport (peer access missing and peer requested)".into());
    }
    Ok(())
}

fn rank_body<'a>(sh: &'a Shared, rank: usize, slot: &mut Option<Env<'a>>) -> R<()> {
    Env::init(sh, rank, slot).map_err(|e| {
        lock(&sh.rec).init_err.get_or_insert(e.clone());
        e
    })?;
    let env = slot.as_mut().ok_or("env slot")?;
    run_all(env)
}

fn rank_thread(sh: &Shared, rank: usize) {
    let mut slot: Option<Env<'_>> = None;
    if let Err(e) = rank_body(sh, rank, &mut slot) {
        let mut rec = lock(&sh.rec);
        rec.case(
            format!("xrank rank={rank} device={}", sh.cfg.devices[rank]),
            Some(format!("error: {e}")),
        );
        rec.errors.push(format!("rank {rank}: {e}"));
        drop(rec);
        sh.barrier.abort();
    }
    // Cleanup: every rank meets here exactly once.
    if let Some(env) = slot.as_ref() {
        let _ = env.gpu.bind_thread();
        let _ = env.gpu.hip.device_synchronize();
    }
    let td_ok = sh.td.wait().is_ok();
    if let Some(mut env) = slot {
        if rank == 0 && td_ok {
            let _ = env.free_banks();
        }
        let _ = env.gpu.bind_thread();
        for ev in env.ready.drain(..) {
            let _ = env.gpu.hip.event_destroy(*ev);
        }
        for ev in env.done.drain(..) {
            let _ = env.gpu.hip.event_destroy(*ev);
        }
        for ev in env.kt.drain(..) {
            let _ = env.gpu.hip.event_destroy(ev);
        }
        if let Some(s) = env.gpu.active_stream.take() {
            let _ = env.gpu.hip.stream_destroy(s);
        }
    }
}

// ───────────────────────────── peer qualification preflight ─────────────────────────────

/// Byte qualification of every directed device pair, produced once by the
/// sequential coordinator before any rank thread starts.
struct Preflight {
    /// `false` for explicit `--transport host`: nothing was probed.
    probed: bool,
    /// `qual[p][q]` (`p != q`): rank `p` -> rank `q` byte-qualified by `qualify_peer`.
    /// `qualify_peer` qualifies both directions of an unordered pair in one call, so
    /// `qual[a][b] == qual[b][a]`.
    qual: Vec<Vec<Option<bool>>>,
    pairs: Vec<Value>,
}

impl Preflight {
    fn directed_ok(&self, p: usize, q: usize) -> bool {
        self.probed && self.qual.get(p).and_then(|r| r.get(q)).copied().flatten() == Some(true)
    }
    fn pair_ok(&self, a: usize, b: usize) -> bool {
        self.directed_ok(a, b) && self.directed_ok(b, a)
    }
    fn all_ok(&self, n: usize) -> bool {
        (0..n).all(|p| (0..n).all(|q| p == q || self.directed_ok(p, q)))
    }
}

impl Cfg {
    /// Any suite that runs on the two-thread rank `Env`.
    fn rank_suites(&self) -> bool {
        self.xrows || self.xtiles || self.xctl || self.xtime || self.xhop
    }
}

fn quiesce(g: &Gpu) -> R<()> {
    ctx(g.bind_thread(), "bind_thread")?;
    ctx(g.hip.device_synchronize(), "device_synchronize")
}

/// Raw capability queries, `qualify_peer`, then a device-wide quiesce of both sides.
/// Any error is returned (the caller leaks both `Gpu`s: their state is unknown).
/// Returns (byte-qualified both directions, can_access a->b, can_access b->a).
fn probe_pair(
    sh: &Shared,
    a: (usize, i32, &mut Gpu),
    b: (usize, i32, &mut Gpu),
) -> R<(bool, bool, bool)> {
    let (ra, da, ga) = a;
    let (rb, db, gb) = b;
    ctx(ga.bind_thread(), "bind_thread")?;
    let can_ab = ctx(ga.hip.can_access_peer(da, db), "can_access_peer")?;
    ctx(gb.bind_thread(), "bind_thread")?;
    let can_ba = ctx(gb.hip.can_access_peer(db, da), "can_access_peer")?;
    for (r, g, d) in [(ra, &*ga, da), (rb, &*gb, db)] {
        let mut info = lock(&sh.infos[r]);
        info.arch = g.arch.clone();
        info.pci = g.hip.device_pci_bus_id(d).ok();
    }
    // Only Ok(false) may select the host arm; an Err fails the whole preflight closed.
    let ok = qualify_peer(ga, gb).map_err(|e| format!("qualify_peer dev{da}<->dev{db}: {e}"))?;
    quiesce(gb)?;
    quiesce(ga)?;
    Ok((ok, can_ab, can_ba))
}

/// Sequential coordinator: two `Gpu`s at a time, never concurrent with rank threads.
fn preflight(sh: &Shared) -> R<()> {
    let n = sh.n;
    let mut pre = Preflight {
        probed: false,
        qual: vec![vec![None; n]; n],
        pairs: Vec::new(),
    };
    if sh.cfg.transport != TransportReq::Host {
        pre.probed = true;
        for a in 0..n {
            for b in a + 1..n {
                let (da, db) = (sh.cfg.devices[a], sh.cfg.devices[b]);
                let mut ga = ctx(
                    Gpu::init_with_device(da),
                    &format!("preflight init device {da}"),
                )?;
                let mut gb = match ctx(
                    Gpu::init_with_device(db),
                    &format!("preflight init device {db}"),
                ) {
                    Ok(g) => g,
                    Err(e) => {
                        std::mem::forget(ga);
                        return Err(e);
                    }
                };
                let (ok, can_ab, can_ba) = match probe_pair(sh, (a, da, &mut ga), (b, db, &mut gb))
                {
                    Ok(v) => v,
                    Err(e) => {
                        // Fail closed: never drop/free Gpus in an unknown state.
                        std::mem::forget(gb);
                        std::mem::forget(ga);
                        return Err(e);
                    }
                };
                drop(gb);
                drop(ga);
                pre.qual[a][b] = Some(ok);
                pre.qual[b][a] = Some(ok);
                println!(
                    "PROBE rank{a}=dev{da} <-> rank{b}=dev{db}: byte_qualified={ok} can_access_peer={can_ab}/{can_ba}"
                );
                pre.pairs.push(json!({
                    "rank_a": a, "rank_b": b, "device_a": da, "device_b": db,
                    "qualified_both_directions": ok,
                    "can_access_peer_a_to_b": can_ab, "can_access_peer_b_to_a": can_ba,
                }));
            }
        }
        lock(&sh.rec).note(
            "peer probe ran via qualify_peer (kernel reads, both directions); qualify_peer may enable peer access at process level, which is not undone"
                .into(),
        );
    }
    sh.pre
        .set(pre)
        .map_err(|_| "preflight already ran".to_string())
}

// ───────────────────────────── xedge: transport service oracle ─────────────────────────────

#[derive(Clone, Copy)]
struct Payload {
    name: &'static str,
    dtype_bytes: usize,
    row_bytes: usize,
}

/// Qwen4 HC row: 4 branches x 2560 hidden, F32 / BF16.
const HC_F32: Payload = Payload {
    name: "f32",
    dtype_bytes: 4,
    row_bytes: 40_960,
};
const HC_BF16: Payload = Payload {
    name: "bf16",
    dtype_bytes: 2,
    row_bytes: 20_480,
};
const XEDGE_PAYLOADS: [Payload; 2] = [HC_F32, HC_BF16];
const XEDGE_ROWS: [usize; 4] = [1, 8, 1536, 4096];
const STALL_BYTES: usize = 128 << 20;
const STALL_ITERS: usize = 600;
const FAIL_CLOSED_LIMIT: Duration = Duration::from_secs(20);
const TAIL_BYTES: usize = 256;
const TAIL_SEED: u64 = 0x7A11_7A11_5EED_0001;

fn poison_bytes(len: usize) -> Vec<u8> {
    POISON_WORD
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(len)
        .collect()
}

/// Unique, reproducible seed per (run seed, case label, generation).
fn xseed(seed: u64, label: &str, k: usize) -> u64 {
    let mut h = splitmix(seed);
    for b in label.bytes() {
        h = splitmix(h ^ b as u64);
    }
    splitmix(h ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

struct XSide {
    device: i32,
    gpu: Gpu,
    scratch: Option<DeviceBuffer>,
    stall_ms: f64,
}

impl XSide {
    fn new(sh: &Shared, rank: usize) -> R<Self> {
        let device = sh.cfg.devices[rank];
        let mut gpu = ctx(
            Gpu::init_with_device(device),
            &format!("xedge init device {device} (rank {rank})"),
        )?;
        ctx(gpu.bind_thread(), "bind_thread")?;
        let stream = ctx(gpu.hip.stream_create(), "stream_create")?;
        gpu.active_stream = Some(stream);
        {
            let mut info = lock(&sh.infos[rank]);
            info.arch = gpu.arch.clone();
            info.pci = gpu.hip.device_pci_bus_id(device).ok();
        }
        let mut side = XSide {
            device,
            gpu,
            scratch: None,
            stall_ms: 0.0,
        };
        if let Err(e) = side.arm_stall() {
            side.teardown(false);
            return Err(e);
        }
        Ok(side)
    }

    fn bind(&self) -> R<()> {
        ctx(self.gpu.bind_thread(), "bind_thread")
    }

    fn sync(&self) -> R<()> {
        self.bind()?;
        let s = self
            .gpu
            .active_stream
            .as_ref()
            .ok_or("xedge side has no active stream")?;
        ctx(
            self.gpu.hip.stream_synchronize(s),
            &format!("stream_synchronize dev{}", self.device),
        )
    }

    /// Enqueue a long chain of memsets on this side's active stream (GPU-side delay).
    fn stall(&self) -> R<()> {
        self.bind()?;
        let s = self
            .gpu
            .active_stream
            .as_ref()
            .ok_or("xedge side has no active stream")?;
        let buf = self.scratch.as_ref().ok_or("no stall scratch")?;
        for _ in 0..STALL_ITERS {
            ctx(
                self.gpu.hip.memset_async(buf, 0, STALL_BYTES, s),
                "stall memset_async",
            )?;
        }
        Ok(())
    }

    fn arm_stall(&mut self) -> R<()> {
        self.bind()?;
        self.scratch = Some(ctx(
            self.gpu.hip.malloc(STALL_BYTES),
            "stall scratch malloc",
        )?);
        let t = Instant::now();
        self.stall()?;
        self.sync()?;
        self.stall_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Drain, free scratch, destroy the stream. A quarantined side is leaked: its
    /// streams may still reference leased resources.
    fn teardown(mut self, quarantined: bool) {
        if quarantined || self.sync().is_err() || self.bind().is_err() {
            // Quarantined, or the device cannot be proven idle: leak, never free.
            std::mem::forget(self);
            return;
        }
        if let Some(b) = self.scratch.take() {
            let _ = self.gpu.hip.free(b);
        }
        if let Some(s) = self.gpu.active_stream.take() {
            let _ = self.gpu.hip.stream_destroy(s);
        }
    }
}

/// Tensors of one case; freed only after the lease drained.
#[derive(Default)]
struct Held {
    items: Vec<(bool, GpuTensor)>,
}

impl Held {
    fn push(&mut self, on_a: bool, t: GpuTensor) -> usize {
        self.items.push((on_a, t));
        self.items.len() - 1
    }
    fn t(&self, i: usize) -> &GpuTensor {
        &self.items[i].1
    }
}

/// Two devices on the coordinator thread. Side `a` = logical edge device 0, side `b`
/// = logical edge device 1.
struct XPair {
    a: XSide,
    b: XSide,
    quarantined: bool,
}

impl XPair {
    fn init(sh: &Shared, ra: usize, rb: usize) -> R<Self> {
        let a = XSide::new(sh, ra)?;
        let b = match XSide::new(sh, rb) {
            Ok(b) => b,
            Err(e) => {
                a.teardown(false);
                return Err(e);
            }
        };
        Ok(XPair {
            a,
            b,
            quarantined: false,
        })
    }

    fn teardown(self) {
        let q = self.quarantined;
        self.a.teardown(q);
        self.b.teardown(q);
    }

    fn side(&self, on_a: bool) -> &XSide {
        if on_a {
            &self.a
        } else {
            &self.b
        }
    }

    /// (producer, consumer) for the edge direction.
    fn gpus(&mut self, fwd: bool) -> (&mut Gpu, &mut Gpu) {
        if fwd {
            (&mut self.a.gpu, &mut self.b.gpu)
        } else {
            (&mut self.b.gpu, &mut self.a.gpu)
        }
    }

    fn up(&self, on_a: bool, bytes: &[u8], what: &str) -> R<GpuTensor> {
        let s = self.side(on_a);
        s.bind()?;
        ctx(s.gpu.upload_raw(bytes, &[bytes.len()]), what)
    }

    fn dl(&self, on_a: bool, t: &GpuTensor) -> R<Vec<u8>> {
        let s = self.side(on_a);
        s.sync()?;
        ctx(s.gpu.download_raw_bytes(t), "download_raw_bytes")
    }

    fn sync_both(&self) -> R<()> {
        let x = self.a.sync();
        let y = self.b.sync();
        x.and(y)
    }

    fn publish(&mut self, fwd: bool, lease: &EdgeLease, t: &GpuTensor, d: EdgeDescriptor) -> R<()> {
        let (src, _) = self.gpus(fwd);
        ctx(src.bind_thread(), "bind_thread")?;
        lease.publish(src, t, d)
    }

    fn receive(&mut self, fwd: bool, lease: &EdgeLease, t: &GpuTensor, d: EdgeDescriptor) -> R<()> {
        let (_, dst) = self.gpus(fwd);
        ctx(dst.bind_thread(), "bind_thread")?;
        lease.receive(dst, t, d)
    }

    /// Drain (stops submissions, syncs both devices, releases banks/events), then drop
    /// the lease. A failed drain quarantines: the lease is leaked, never freed.
    fn finish(&mut self, fwd: bool, mut lease: EdgeLease) -> R<()> {
        let drained = {
            let (src, dst) = self.gpus(fwd);
            lease.drain(src, dst)
        };
        match drained {
            Ok(()) => {
                drop(lease);
                let synced = self.sync_both();
                if synced.is_err() {
                    self.quarantined = true;
                }
                synced.map_err(|e| format!("stream error after drain, pair quarantined: {e}"))
            }
            Err(e) => {
                self.quarantined = true;
                std::mem::forget(lease);
                Err(format!("drain failed, resources quarantined: {e}"))
            }
        }
    }

    fn free_held(&mut self, held: Held) -> R<()> {
        if self.quarantined {
            for (_, t) in held.items {
                std::mem::forget(t);
            }
            return Ok(());
        }
        let mut first: Option<String> = None;
        for (on_a, t) in held.items {
            let gpu = if on_a {
                &mut self.a.gpu
            } else {
                &mut self.b.gpu
            };
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(format!("free_tensor: {e}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn provision_edge(
    pair: &mut XPair,
    fwd: bool,
    tr: EdgeTransport,
    max_bytes: usize,
) -> R<EdgeLease> {
    let edge = if fwd {
        EdgeId {
            src_device: 0,
            dst_device: 1,
        }
    } else {
        EdgeId {
            src_device: 1,
            dst_device: 0,
        }
    };
    let result = (|| {
        let (src, dst) = pair.gpus(fwd);
        ctx(dst.bind_thread(), "bind_thread")?;
        ctx(src.bind_thread(), "bind_thread")?;
        provision(
            src,
            dst,
            EdgeSpec {
                edge,
                transport: tr,
                max_bytes,
            },
        )
        .map_err(|e| format!("provision: {e}"))
    })();
    // A failed provision may leave a quarantined lease referencing both devices.
    if result.is_err() {
        pair.quarantined = true;
    }
    result
}

/// Provision, run `f`, then ALWAYS drain, drop the lease, and only then free tensors.
fn with_lease<F>(
    pair: &mut XPair,
    fwd: bool,
    tr: EdgeTransport,
    max_bytes: usize,
    f: F,
) -> R<Vec<String>>
where
    F: FnOnce(&mut XPair, &mut EdgeLease, &mut Held) -> R<Vec<String>>,
{
    let mut lease = provision_edge(pair, fwd, tr, max_bytes)?;
    let mut held = Held::default();
    let mut problems = match f(pair, &mut lease, &mut held) {
        Ok(p) => p,
        Err(e) => vec![format!("error: {e}")],
    };
    if let Err(e) = pair.finish(fwd, lease) {
        problems.push(e);
    }
    if let Err(e) = pair.free_held(held) {
        problems.push(e);
    }
    Ok(problems)
}

#[derive(Clone, Copy)]
struct SeqGen {
    p: Payload,
    rows: usize,
}

#[derive(Clone, Copy)]
struct SeqOpts {
    /// Extra bytes after the live extent on both views (destination = poison, source
    /// = random); the service must neither read nor write them.
    tail: usize,
    stall_prod: bool,
    stall_cons: bool,
    sleep_ms: u64,
}

const SEQ_PLAIN: SeqOpts = SeqOpts {
    tail: 0,
    stall_prod: false,
    stall_cons: false,
    sleep_ms: 0,
};

#[derive(Clone, Copy)]
struct XSlot {
    si: usize,
    di: usize,
    len: usize,
    seed: u64,
    rows: usize,
    row_bytes: usize,
    desc: EdgeDescriptor,
}

#[allow(clippy::too_many_arguments)]
fn make_slots(
    sh: &Shared,
    pair: &XPair,
    lease: &EdgeLease,
    held: &mut Held,
    fwd: bool,
    label: &str,
    gens: &[SeqGen],
    tail: usize,
) -> R<Vec<XSlot>> {
    let (prod_a, cons_a) = (fwd, !fwd);
    let mut slots = Vec::with_capacity(gens.len());
    for (k, g) in gens.iter().enumerate() {
        let len = g.rows * g.p.row_bytes;
        let seed = xseed(sh.cfg.seed, label, k);
        let mut src = random_bytes(seed, len);
        src.extend_from_slice(&random_bytes(seed ^ TAIL_SEED, tail));
        let si = held.push(prod_a, pair.up(prod_a, &src, "upload source view")?);
        drop(src);
        let di = held.push(
            cons_a,
            pair.up(
                cons_a,
                &poison_bytes(len + tail),
                "upload poisoned destination",
            )?,
        );
        slots.push(XSlot {
            si,
            di,
            len,
            seed,
            rows: g.rows,
            row_bytes: g.p.row_bytes,
            desc: EdgeDescriptor {
                epoch: lease.epoch(),
                // Arbitrary, non-monotonic: the service's generation is independent of op.
                op: (seed & 0xFFFF_FFFF) + 1,
                rows: g.rows,
                dtype_bytes: g.p.dtype_bytes,
                bytes: len,
            },
        });
    }
    Ok(slots)
}

fn consume(
    pair: &mut XPair,
    fwd: bool,
    lease: &EdgeLease,
    held: &Held,
    s: &XSlot,
    o: &SeqOpts,
) -> R<()> {
    if o.stall_cons {
        pair.side(!fwd).stall()?;
    }
    if o.sleep_ms > 0 {
        std::thread::sleep(Duration::from_millis(o.sleep_ms));
    }
    pair.receive(fwd, lease, held.t(s.di), s.desc)
}

/// Depth-2 pipeline: publish(k), receive(k-1); both banks stay in flight and every
/// generation >= 2 reuses a bank. Destinations are verified after the whole run, so
/// nothing host-synchronizes between transfers.
#[allow(clippy::too_many_arguments)]
fn run_seq(
    sh: &Shared,
    pair: &mut XPair,
    fwd: bool,
    lease: &EdgeLease,
    held: &mut Held,
    label: &str,
    gens: &[SeqGen],
    o: &SeqOpts,
) -> R<Vec<String>> {
    let (prod_a, cons_a) = (fwd, !fwd);
    let slots = make_slots(sh, pair, lease, held, fwd, label, gens, o.tail)?;
    let n = slots.len();
    for k in 0..n {
        if o.stall_prod {
            pair.side(prod_a).stall()?;
        }
        pair.publish(fwd, lease, held.t(slots[k].si), slots[k].desc)?;
        if k >= 1 {
            consume(pair, fwd, lease, held, &slots[k - 1], o)?;
        }
    }
    if n > 0 {
        consume(pair, fwd, lease, held, &slots[n - 1], o)?;
    }
    pair.sync_both()?;
    let mut problems = Vec::new();
    for (k, s) in slots.iter().enumerate() {
        let want = random_bytes(s.seed, s.len);
        let got = pair.dl(cons_a, held.t(s.di))?;
        if let Some(d) = first_diff(
            &format!("gen {k} rows={} destination", s.rows),
            &want,
            &got[..s.len],
            s.row_bytes,
        ) {
            problems.push(d);
        }
        if o.tail > 0 {
            if let Some(d) = first_diff(
                &format!("gen {k} destination tail"),
                &poison_bytes(o.tail),
                &got[s.len..],
                16,
            ) {
                problems.push(d);
            }
        }
        let src = pair.dl(prod_a, held.t(s.si))?;
        if let Some(d) = first_diff(
            &format!("gen {k} source view mutated"),
            &want,
            &src[..s.len],
            s.row_bytes,
        ) {
            problems.push(d);
        }
        if o.tail > 0 {
            if let Some(d) = first_diff(
                &format!("gen {k} source tail mutated"),
                &random_bytes(s.seed ^ TAIL_SEED, o.tail),
                &src[s.len..],
                16,
            ) {
                problems.push(d);
            }
        }
    }
    Ok(problems)
}

fn seq_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    fwd: bool,
    label: &str,
    gens: &[SeqGen],
    o: &SeqOpts,
) -> R<Vec<String>> {
    let max = gens
        .iter()
        .map(|g| g.rows * g.p.row_bytes)
        .max()
        .unwrap_or(16);
    with_lease(pair, fwd, tr, max, |pair, lease, held| {
        run_seq(sh, pair, fwd, lease, held, label, gens, o)
    })
}

fn fixed_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
    p: Payload,
    rows: usize,
) -> R<Vec<String>> {
    // >= 4 generations: both banks are reused at least once.
    let n = if rows >= 4096 { 4 } else { 5 };
    let gens = vec![SeqGen { p, rows }; n];
    let label = format!("{pfx} fixed {} rows={rows}", p.name);
    seq_case(sh, pair, tr, true, &label, &gens, &SEQ_PLAIN)
}

fn reuse_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
    p: Payload,
    fwd: bool,
) -> R<Vec<String>> {
    // Mixed live sizes on every bank: a short generation lands on a bank that held a
    // longer one, so stale bank bytes beyond the live extent would show up.
    let gens: Vec<SeqGen> = [1536usize, 1, 4096, 8, 1, 4096, 1536, 8]
        .iter()
        .map(|&rows| SeqGen { p, rows })
        .collect();
    let label = format!(
        "{pfx} reuse {} dir={}",
        p.name,
        if fwd { "fwd" } else { "rev" }
    );
    let o = SeqOpts {
        tail: TAIL_BYTES,
        ..SEQ_PLAIN
    };
    seq_case(sh, pair, tr, fwd, &label, &gens, &o)
}

fn delayed_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
    producer: bool,
    p: Payload,
    rows: usize,
) -> R<Vec<String>> {
    let gens = vec![SeqGen { p, rows }; 6];
    let label = format!(
        "{pfx} delayed-{} {} rows={rows}",
        if producer { "producer" } else { "consumer" },
        p.name
    );
    let o = SeqOpts {
        tail: 0,
        stall_prod: producer,
        stall_cons: !producer,
        sleep_ms: if producer { 0 } else { 25 },
    };
    seq_case(sh, pair, tr, true, &label, &gens, &o)
}

#[allow(clippy::too_many_arguments)]
fn chain_body(
    sh: &Shared,
    pair: &mut XPair,
    l_ab: &EdgeLease,
    l_ba: &EdgeLease,
    held: &mut Held,
    label: &str,
    p: Payload,
    rows: usize,
) -> R<Vec<String>> {
    let bytes = rows * p.row_bytes;
    let mut problems = Vec::new();
    for it in 0..2usize {
        let seed = xseed(sh.cfg.seed, label, it);
        let want = random_bytes(seed, bytes);
        let mut cur = held.push(true, pair.up(true, &want, "upload chain source")?);
        let mut hops: Vec<(usize, bool)> = Vec::new();
        for h in 0..6usize {
            let fwd = h % 2 == 0;
            let lease = if fwd { l_ab } else { l_ba };
            let dst = held.push(
                !fwd,
                pair.up(!fwd, &poison_bytes(bytes), "upload hop destination")?,
            );
            let desc = EdgeDescriptor {
                epoch: lease.epoch(),
                op: (splitmix(seed ^ h as u64) & 0xFFFF_FFFF) + 1,
                rows,
                dtype_bytes: p.dtype_bytes,
                bytes,
            };
            pair.publish(fwd, lease, held.t(cur), desc)?;
            pair.receive(fwd, lease, held.t(dst), desc)?;
            hops.push((dst, !fwd));
            cur = dst;
        }
        pair.sync_both()?;
        for (h, (idx, on_a)) in hops.iter().enumerate() {
            let got = pair.dl(*on_a, held.t(*idx))?;
            if let Some(d) = first_diff(
                &format!(
                    "iteration {it} hop {h} ({})",
                    if h % 2 == 0 { "A>B" } else { "B>A" }
                ),
                &want,
                &got,
                p.row_bytes,
            ) {
                problems.push(d);
            }
        }
    }
    Ok(problems)
}

/// Six alternating-device hops (A>B, B>A, ...) over two leases; hop h+1 publishes the
/// tensor hop h received. All six hops are enqueued before anything is read back.
fn chain_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
    p: Payload,
    rows: usize,
) -> R<Vec<String>> {
    let bytes = rows * p.row_bytes;
    let label = format!("{pfx} chain6 {} rows={rows}", p.name);
    let l_ab = provision_edge(pair, true, tr, bytes)?;
    let l_ba = match provision_edge(pair, false, tr, bytes) {
        Ok(l) => l,
        Err(e) => {
            return Err(match pair.finish(true, l_ab) {
                Ok(()) => e,
                Err(d) => format!("{e}; {d}"),
            });
        }
    };
    let mut held = Held::default();
    let mut problems = match chain_body(sh, pair, &l_ab, &l_ba, &mut held, &label, p, rows) {
        Ok(v) => v,
        Err(e) => vec![format!("error: {e}")],
    };
    if let Err(e) = pair.finish(true, l_ab) {
        problems.push(e);
    }
    if let Err(e) = pair.finish(false, l_ba) {
        problems.push(e);
    }
    if let Err(e) = pair.free_held(held) {
        problems.push(e);
    }
    Ok(problems)
}

/// (hook, receipt name, fires inside `publish`)
fn fault_points() -> [(FaultPoint, &'static str, bool); 9] {
    [
        (FaultPoint::BeforeReuseWait, "BeforeReuseWait", true),
        (FaultPoint::AfterReuseWait, "AfterReuseWait", true),
        (FaultPoint::AfterPublishCopy, "AfterPublishCopy", true),
        (FaultPoint::AfterReadyRecord, "AfterReadyRecord", true),
        (
            FaultPoint::AfterReadyPublication,
            "AfterReadyPublication",
            true,
        ),
        (FaultPoint::AfterReadyWait, "AfterReadyWait", false),
        (FaultPoint::AfterReceiveCopy, "AfterReceiveCopy", false),
        (FaultPoint::AfterDoneRecord, "AfterDoneRecord", false),
        (
            FaultPoint::AfterDonePublication,
            "AfterDonePublication",
            false,
        ),
    ]
}

fn expect_err(problems: &mut Vec<String>, what: &str, r: R<()>) {
    if r.is_ok() {
        problems.push(format!("{what} returned Ok"));
    }
}

fn poison_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
    point: FaultPoint,
    name: &str,
    publish_side: bool,
) -> R<Vec<String>> {
    let label = format!("{pfx} poison {name}");
    let p = HC_F32;
    let rows = 8usize;
    let mut problems = with_lease(pair, true, tr, rows * p.row_bytes, |pair, lease, held| {
        let gens = vec![SeqGen { p, rows }; 4];
        let s = make_slots(sh, pair, lease, held, true, &label, &gens, 0)?;
        // Prime generations 0 and 1 through both banks, then inject so generation 2 is
        // a real bank reuse (done-generation wait) for every hook.
        for k in 0..2 {
            pair.publish(true, lease, held.t(s[k].si), s[k].desc)?;
        }
        for k in 0..2 {
            pair.receive(true, lease, held.t(s[k].di), s[k].desc)?;
        }
        lease.inject_fault(point);
        let t0 = Instant::now();
        let mut problems = Vec::new();
        let first = pair.publish(true, lease, held.t(s[2].si), s[2].desc);
        if publish_side {
            expect_err(&mut problems, "publish(g2) at the fault point", first);
        } else {
            match first {
                Err(e) => problems.push(format!(
                    "publish(g2) failed before the receive-side fault: {e}"
                )),
                Ok(()) => expect_err(
                    &mut problems,
                    "receive(g2) at the fault point",
                    pair.receive(true, lease, held.t(s[2].di), s[2].desc),
                ),
            }
        }
        // Poisoned: every later call fails closed, immediately.
        expect_err(
            &mut problems,
            "publish after the fault",
            pair.publish(true, lease, held.t(s[3].si), s[3].desc),
        );
        expect_err(
            &mut problems,
            "receive after the fault",
            pair.receive(true, lease, held.t(s[2].di), s[2].desc),
        );
        let el = t0.elapsed();
        if el > FAIL_CLOSED_LIMIT {
            problems.push(format!(
                "fail-closed path took {:.1}s (limit {}s)",
                el.as_secs_f64(),
                FAIL_CLOSED_LIMIT.as_secs()
            ));
        }
        Ok(problems)
    })?;
    if !pair.quarantined {
        let gens = vec![SeqGen { p, rows }; 3];
        let rec = seq_case(
            sh,
            pair,
            tr,
            true,
            &format!("{label} recovery"),
            &gens,
            &SEQ_PLAIN,
        )?;
        problems.extend(
            rec.into_iter()
                .map(|r| format!("recovery after poison: {r}")),
        );
    }
    Ok(problems)
}

fn poison_explicit_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
) -> R<Vec<String>> {
    let label = format!("{pfx} poison explicit");
    let p = HC_F32;
    let rows = 8usize;
    with_lease(pair, true, tr, rows * p.row_bytes, |pair, lease, held| {
        let gens = vec![SeqGen { p, rows }; 3];
        let s = make_slots(sh, pair, lease, held, true, &label, &gens, 0)?;
        pair.publish(true, lease, held.t(s[0].si), s[0].desc)?;
        lease.poison();
        let mut problems = Vec::new();
        expect_err(
            &mut problems,
            "receive after explicit poison",
            pair.receive(true, lease, held.t(s[0].di), s[0].desc),
        );
        expect_err(
            &mut problems,
            "publish after explicit poison",
            pair.publish(true, lease, held.t(s[1].si), s[1].desc),
        );
        Ok(problems)
    })
}

fn up_view(pair: &XPair, held: &mut Held, on_a: bool, len: usize, seed: u64) -> R<usize> {
    Ok(held.push(
        on_a,
        pair.up(on_a, &random_bytes(seed, len), "upload view")?,
    ))
}

fn reject_case(
    sh: &Shared,
    pair: &mut XPair,
    tr: EdgeTransport,
    pfx: &str,
    which: &str,
) -> R<Vec<String>> {
    let label = format!("{pfx} reject {which}");
    let seed = xseed(sh.cfg.seed, &label, 0);
    let row = HC_F32.row_bytes;
    let small = 8 * row;
    with_lease(pair, true, tr, small, |pair, lease, held| {
        let ok = EdgeDescriptor {
            epoch: lease.epoch(),
            op: 7,
            rows: 8,
            dtype_bytes: 4,
            bytes: small,
        };
        let one = EdgeDescriptor {
            rows: 1,
            bytes: row,
            ..ok
        };
        let mut problems = Vec::new();
        match which {
            "short-view" => {
                let v = up_view(pair, held, true, row - 16, seed)?;
                let r = pair.publish(true, lease, held.t(v), one);
                expect_err(
                    &mut problems,
                    "publish with a view shorter than the descriptor",
                    r,
                );
            }
            "row-align" => {
                let v = up_view(pair, held, true, 64, seed)?;
                let d = EdgeDescriptor {
                    rows: 1,
                    bytes: 40,
                    ..ok
                };
                let r = pair.publish(true, lease, held.t(v), d);
                expect_err(
                    &mut problems,
                    "publish with a row size that is not a multiple of 16",
                    r,
                );
            }
            "dtype-bytes" => {
                let v = up_view(pair, held, true, row, seed)?;
                let d = EdgeDescriptor {
                    dtype_bytes: 3,
                    ..one
                };
                let r = pair.publish(true, lease, held.t(v), d);
                expect_err(&mut problems, "publish with dtype_bytes=3", r);
            }
            "rows-zero" => {
                let v = up_view(pair, held, true, row, seed)?;
                let d = EdgeDescriptor { rows: 0, ..one };
                let r = pair.publish(true, lease, held.t(v), d);
                expect_err(&mut problems, "publish with rows=0", r);
            }
            "over-max" => {
                let v = up_view(pair, held, true, small + 16, seed)?;
                let d = EdgeDescriptor {
                    rows: 1,
                    bytes: small + 16,
                    ..ok
                };
                let r = pair.publish(true, lease, held.t(v), d);
                expect_err(
                    &mut problems,
                    "publish larger than the provisioned max_bytes",
                    r,
                );
            }
            "foreign-view" => {
                // A view that lives on the OTHER device must be refused by the producer.
                let v = up_view(pair, held, false, small, seed)?;
                let r = pair.publish(true, lease, held.t(v), ok);
                expect_err(
                    &mut problems,
                    "publish of a view owned by the consumer device",
                    r,
                );
            }
            "recv-op" | "recv-rows" | "recv-dtype" | "recv-epoch" => {
                let s = up_view(pair, held, true, small, seed)?;
                let d = up_view(pair, held, false, small, seed ^ 1)?;
                pair.publish(true, lease, held.t(s), ok)?;
                let bad = match which {
                    "recv-op" => EdgeDescriptor {
                        op: ok.op + 1,
                        ..ok
                    },
                    "recv-rows" => EdgeDescriptor { rows: 16, ..ok },
                    "recv-dtype" => EdgeDescriptor {
                        dtype_bytes: 2,
                        ..ok
                    },
                    _ => EdgeDescriptor {
                        epoch: ok.epoch + 1,
                        ..ok
                    },
                };
                let r = pair.receive(true, lease, held.t(d), bad);
                expect_err(&mut problems, "receive with a mismatching descriptor", r);
            }
            other => return Err(format!("unknown reject case {other:?}")),
        }
        Ok(problems)
    })
}

fn reprovision_case(sh: &Shared, pair: &mut XPair, tr: EdgeTransport, pfx: &str) -> R<Vec<String>> {
    let label = format!("{pfx} reprovision");
    let p = HC_F32;
    let (small, big) = (8 * p.row_bytes, 1536 * p.row_bytes);
    let mut problems = Vec::new();

    let l1 = provision_edge(pair, true, tr, small)?;
    let e1 = l1.epoch();
    let mut held1 = Held::default();
    match run_seq(
        sh,
        pair,
        true,
        &l1,
        &mut held1,
        &format!("{label} lease1"),
        &vec![SeqGen { p, rows: 8 }; 3],
        &SEQ_PLAIN,
    ) {
        Ok(v) => problems.extend(v),
        Err(e) => problems.push(format!("error: {e}")),
    }
    if let Err(e) = pair.finish(true, l1) {
        problems.push(e);
    }
    if let Err(e) = pair.free_held(held1) {
        problems.push(e);
    }
    if pair.quarantined {
        return Ok(problems);
    }

    // New lease on the same edge (different capacity): fresh epoch, generations restart.
    let l2 = provision_edge(pair, true, tr, big)?;
    if l2.epoch() == e1 {
        problems.push(format!("reprovisioned lease reused transport epoch {e1}"));
    }
    let mut held2 = Held::default();
    match run_seq(
        sh,
        pair,
        true,
        &l2,
        &mut held2,
        &format!("{label} lease2"),
        &vec![SeqGen { p, rows: 1536 }; 3],
        &SEQ_PLAIN,
    ) {
        Ok(v) => problems.extend(v),
        Err(e) => problems.push(format!("error: {e}")),
    }
    // Stale descriptor from the previous epoch must be refused (and poisons l2).
    match up_view(pair, &mut held2, true, small, xseed(sh.cfg.seed, &label, 9)) {
        Ok(v) => {
            let stale = EdgeDescriptor {
                epoch: e1,
                op: 1,
                rows: 8,
                dtype_bytes: 4,
                bytes: small,
            };
            let r = pair.publish(true, &l2, held2.t(v), stale);
            expect_err(&mut problems, "publish with the previous lease's epoch", r);
        }
        Err(e) => problems.push(format!("error: {e}")),
    }
    if let Err(e) = pair.finish(true, l2) {
        problems.push(e);
    }
    if let Err(e) = pair.free_held(held2) {
        problems.push(e);
    }
    Ok(problems)
}

fn record_x(sh: &Shared, name: &str, r: R<Vec<String>>) {
    let detail = match r {
        Ok(v) if v.is_empty() => None,
        Ok(v) => Some(v.into_iter().take(4).collect::<Vec<_>>().join("; ")),
        Err(e) => Some(format!("error: {e}")),
    };
    lock(&sh.rec).case(name.to_string(), detail);
}

fn edge_suite(sh: &Shared, pair: &mut XPair, tr: EdgeTransport, d0: i32, d1: i32) {
    let pfx = format!("xedge {tr} dev{d0}-dev{d1}");
    macro_rules! run_case {
        ($name:expr, $body:expr) => {{
            if pair.quarantined {
                return;
            }
            let name: String = $name;
            let r: R<Vec<String>> = $body;
            record_x(sh, &name, r);
        }};
    }
    for p in XEDGE_PAYLOADS {
        for rows in XEDGE_ROWS {
            run_case!(
                format!("{pfx} fixed {} rows={rows}", p.name),
                fixed_case(sh, pair, tr, &pfx, p, rows)
            );
        }
    }
    for p in XEDGE_PAYLOADS {
        for rows in XEDGE_ROWS {
            run_case!(
                format!("{pfx} chain6 {} rows={rows}", p.name),
                chain_case(sh, pair, tr, &pfx, p, rows)
            );
        }
    }
    for p in XEDGE_PAYLOADS {
        for fwd in [true, false] {
            run_case!(
                format!(
                    "{pfx} reuse {} dir={}",
                    p.name,
                    if fwd { "fwd" } else { "rev" }
                ),
                reuse_case(sh, pair, tr, &pfx, p, fwd)
            );
        }
    }
    for producer in [true, false] {
        for (p, rows) in [(HC_F32, 8usize), (HC_BF16, 1536usize)] {
            run_case!(
                format!(
                    "{pfx} delayed-{} {} rows={rows}",
                    if producer { "producer" } else { "consumer" },
                    p.name
                ),
                delayed_case(sh, pair, tr, &pfx, producer, p, rows)
            );
        }
    }
    for (point, name, publish_side) in fault_points() {
        run_case!(
            format!("{pfx} poison {name}"),
            poison_case(sh, pair, tr, &pfx, point, name, publish_side)
        );
    }
    run_case!(
        format!("{pfx} poison explicit"),
        poison_explicit_case(sh, pair, tr, &pfx)
    );
    for which in [
        "short-view",
        "row-align",
        "dtype-bytes",
        "rows-zero",
        "over-max",
        "foreign-view",
        "recv-op",
        "recv-rows",
        "recv-dtype",
        "recv-epoch",
    ] {
        run_case!(
            format!("{pfx} reject {which}"),
            reject_case(sh, pair, tr, &pfx, which)
        );
    }
    run_case!(
        format!("{pfx} reprovision"),
        reprovision_case(sh, pair, tr, &pfx)
    );
}

/// Dedicated coordinator for `xedge`: adjacent device pairs, on this thread, before
/// any rank thread exists.
fn xedge_main(sh: &Shared) {
    let pre = sh.pre();
    for i in 0..sh.n - 1 {
        let j = i + 1;
        let (d0, d1) = (sh.cfg.devices[i], sh.cfg.devices[j]);
        let qualified = pre.pair_ok(i, j);
        let mut arms: Vec<EdgeTransport> = Vec::new();
        match sh.cfg.transport {
            TransportReq::Host => arms.push(EdgeTransport::HostMapped),
            TransportReq::Peer => {
                if qualified {
                    arms.push(EdgeTransport::Peer);
                }
            }
            TransportReq::Both => {
                arms.push(EdgeTransport::HostMapped);
                if qualified {
                    arms.push(EdgeTransport::Peer);
                }
            }
            TransportReq::Auto => arms.push(if qualified {
                EdgeTransport::Peer
            } else {
                EdgeTransport::HostMapped
            }),
        }
        if matches!(sh.cfg.transport, TransportReq::Peer | TransportReq::Both) && !qualified {
            println!("SKIP xedge peer dev{d0}-dev{d1}: pair is not byte-qualified by qualify_peer");
            lock(&sh.rec).notes.push(format!(
                "xedge peer dev{d0}-dev{d1} skipped: pair not byte-qualified"
            ));
        }
        if arms.is_empty() {
            continue;
        }
        let mut pair = match XPair::init(sh, i, j) {
            Ok(p) => p,
            Err(e) => {
                let mut rec = lock(&sh.rec);
                rec.init_err.get_or_insert(e.clone());
                rec.case(format!("xedge dev{d0}-dev{d1} init"), Some(e.clone()));
                rec.errors.push(format!("xedge dev{d0}-dev{d1}: {e}"));
                continue;
            }
        };
        {
            let (sa, sb) = (pair.a.stall_ms, pair.b.stall_ms);
            let mut rec = lock(&sh.rec);
            rec.note(format!(
                "xedge dev{d0}-dev{d1}: stall calibration dev{d0}={sa:.1} ms dev{d1}={sb:.1} ms (delayed-producer/consumer enqueue one stall per transfer)"
            ));
            if sa < 5.0 || sb < 5.0 {
                rec.note(format!(
                    "xedge dev{d0}-dev{d1}: stall under 5 ms; delayed cases are weak on this device"
                ));
            }
        }
        for tr in arms {
            lock(&sh.rec).xedge_arms.push(json!({
                "ranks": [i, j],
                "devices": [d0, d1],
                "transport": tr.to_string(),
                "pair_byte_qualified": qualified,
            }));
            if tr == EdgeTransport::HostMapped && pre.probed {
                lock(&sh.rec).note(format!(
                    "xedge host-mapped dev{d0}-dev{d1} ran after a qualify_peer probe: the probe may have enabled peer access at process level; the HostMapped service itself never queries or enables peer access"
                ));
            }
            edge_suite(sh, &mut pair, tr, d0, d1);
            if pair.quarantined {
                lock(&sh.rec).errors.push(format!(
                    "xedge dev{d0}-dev{d1}: pair quarantined after a failed provision, drain or post-drain sync; remaining cases skipped"
                ));
                break;
            }
        }
        pair.teardown();
    }
}

fn multi_main(args: &Args) -> ! {
    let devices = args.devices.clone().unwrap_or_default();
    let env_transport = std::env::var("HIPFIRE_QWEN4_EP_TRANSPORT").ok();
    let (req_str, src) = match (&args.transport, &env_transport) {
        (Some(t), _) => (t.clone(), "--transport".to_string()),
        (None, Some(e)) if !e.trim().is_empty() => (
            e.trim().to_string(),
            "HIPFIRE_QWEN4_EP_TRANSPORT".to_string(),
        ),
        _ => ("auto".to_string(), "default".to_string()),
    };
    let transport = match req_str.as_str() {
        "auto" => TransportReq::Auto,
        "host" => TransportReq::Host,
        "peer" => TransportReq::Peer,
        "both" if args.transport.is_some() => TransportReq::Both,
        other => {
            eprintln!("transport {other:?} (from {src}) must be auto|peer|host (both only via --transport)");
            std::process::exit(2);
        }
    };
    let host_flags = match args.host_flags.as_str() {
        "mapped" => vec![HostFlags::Mapped],
        "coherent" => vec![HostFlags::Coherent],
        _ => vec![HostFlags::Mapped, HostFlags::Coherent],
    };
    let suites: Vec<String> = args
        .suites
        .clone()
        .unwrap_or_else(|| X_SUITES.join(","))
        .split(',')
        .map(str::to_string)
        .collect();
    let has = |s: &str| suites.iter().any(|x| x == s);
    let cfg = Cfg {
        devices,
        transport,
        transport_src: src,
        host_flags,
        xrows: has("xrows"),
        xtiles: has("xtiles"),
        xctl: has("xctl"),
        xtime: has("xtime"),
        xhop: has("xhop"),
        xedge: has("xedge"),
        rounds: args.rounds,
        reps: args.reps,
        layers: args.layers,
        seed: args.seed,
    };
    let n = cfg.devices.len();
    let sh = Shared::new(cfg);
    // Sequential coordinator phases run on this thread, before any rank thread.
    let mut coordinated = true;
    if let Err(e) = preflight(&sh) {
        lock(&sh.rec).init_err.get_or_insert(e);
        coordinated = false;
    }
    if coordinated && sh.cfg.xedge {
        xedge_main(&sh);
    }
    if coordinated && sh.cfg.rank_suites() {
        std::thread::scope(|s| {
            for rank in 0..n {
                let sh = &sh;
                s.spawn(move || rank_thread(sh, rank));
            }
        });
    }
    let rec = lock(&sh.rec);
    if let Some(e) = &rec.init_err {
        eprintln!("no device: {e}");
        std::process::exit(2);
    }
    let (pass, fail) = (rec.pass, rec.fail);
    println!(
        "RESULT: {} ({pass} passed, {fail} failed)",
        if fail == 0 && rec.errors.is_empty() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    for e in &rec.errors {
        eprintln!("error: {e}");
    }
    let no_transport = (sh.cfg.rank_suites() && rec.arms.is_empty())
        || (sh.cfg.xedge && rec.xedge_arms.is_empty());
    if let Some(path) = &args.json {
        let pre = sh.pre();
        let ranks: Vec<Value> = (0..n)
            .map(|q| {
                let info = lock(&sh.infos[q]);
                json!({
                    "rank": q,
                    "device": sh.cfg.devices[q],
                    "arch": info.arch,
                    "pci_bus_id": info.pci,
                    "peer_access_enabled": {
                        "host": info.peer_enabled_host,
                        "peer": info.peer_enabled_final,
                        "probe_ran": pre.probed,
                    },
                })
            })
            .collect();
        let matrix: Vec<Value> = (0..n)
            .map(|q| json!(lock(&sh.infos[q]).can_access))
            .collect();
        let receipt = json!({
            "mode": "distinct-device",
            "devices": sh.cfg.devices,
            "ranks": n,
            "seed": sh.cfg.seed,
            "rounds": sh.cfg.rounds,
            "reps": sh.cfg.reps,
            "layers": sh.cfg.layers,
            "transport_source": sh.cfg.transport_src,
            "transport_resolved": rec.resolved,
            "arms": rec.arms,
            "can_access_peer": matrix,
            "peer_qualification": {
                "performed": pre.probed,
                "api": "multi_gpu::transport::qualify_peer (kernel reads, both directions)",
                "probe_may_enable_peer_access": pre.probed,
                "matrix": pre.qual,
                "pairs": pre.pairs,
            },
            "xedge": {
                "enabled": sh.cfg.xedge,
                "arms": rec.xedge_arms,
            },
            "host_flags": sh.cfg.host_flags.iter().map(|f| f.name()).collect::<Vec<_>>(),
            "per_rank": ranks,
            "host_banks": rec.banks,
            "notes": rec.notes,
            "errors": rec.errors,
            "passed": pass,
            "failed": fail,
            "cases": rec.cases,
            "timings": rec.timings,
        });
        if let Err(e) = std::fs::write(
            path,
            serde_json::to_string_pretty(&receipt).unwrap_or_default(),
        ) {
            eprintln!("--json {path}: {e}");
            std::process::exit(2);
        }
    }
    if no_transport {
        eprintln!("no runnable transport");
        std::process::exit(2);
    }
    std::process::exit(if fail == 0 && pass > 0 && rec.errors.is_empty() {
        0
    } else {
        1
    });
}
