// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle (and bench) of the sequence-parallel exchange launches
//! (`sp_hc_rs_d8_*` reduce-scatter fused into the HC write, `sp_ag_rows_*` row all-gather,
//! `rdna_compute::ar_hc_write::sp_*`) against the product all-reduce launch
//! (`ar_hc_write_d8_*`) on two gfx1201 cards, both on the DENSE wire of one `ArHcExchange`.
//!
//! ```text
//! sp_rs_ag_oracle [--rows 1,2,7,16,255,256,300,511,512,613] [--iters 200] [--case-iters 3]
//!                 [--devices 0,1] [--out DIR] [--stream bf16|f32|both] [--no-fault] [--bench]
//! ```
//!
//! Per (stream kind, row count, split) case, `--case-iters` steps; then a chained loop of
//! `--iters` steps rotating over every case, alternating the launch order (AR, RS, AG... on
//! even steps, AG, RS, AR, AG on odd ones) on the SAME exchange (generation consistency).
//! Splits of the tile between the two ranks: `even` (rank 0 takes the extra row), `reverse`
//! (rank 1 takes it; one rank owns zero rows when `rows == 1`) and `skew` (rank 0 owns
//! `rows / 4`). Each step checks, on both ranks:
//!  (i) RS vs AR: identical random partials (distinct per rank), gates and streams; copy A gets
//!      the product `ar_hc_write_d8_*` launch on all rows, copy B the RS launch. Own rows of B
//!      are byte-equal to the same rows of A, the other rows of B are byte-equal to the initial
//!      streams, the debug `m` of own rows is bit-equal to A's and to the host fold, `m` of the
//!      other rows was never written;
//!  (ii) AG: each rank fills only its own rows of a `[rows][row_elems]` buffer with random
//!      finite bf16 (garbage elsewhere; a NaN-pair dword is planted at random). After the
//!      gather both ranks' buffers equal the concatenation, a NaN-pair dword `0xFFFF_FFFF`
//!      arriving as `0xFFFF_FFFE` (`sp_wire_dword`). Formats: bf16 bits and f32 words holding
//!      bf16-exact values (`F32Bf16`), at row_elems 2560 and 640;
//!  (iii) after every launch the dense bank it used is all-EMPTY (0xFF) in the rank's own mailbox;
//!  (v) unless `--no-fault`: one rank skips an RS launch, the other's bounded wait must record
//!      `SP_HC_RS_GFX1201` (kernel `sp_hc_rs_d8_gfx1201`) and fail; the same for an AG launch
//!      (`sp_ag_rows_bf16_d8_gfx1201`).
//! `--bench`: per 4096-row tile, median of 20 (host wall after sync) of AR (16 launches x 256
//! rows), RS (8 launches covering 512 rows each: 256 own + 256 peer), AG of a `[4096][2560]`
//! buffer (same slicing; bf16 and f32bf16) and the RS+AG sequence.
//! Exit code 3 on any failure; a JSON receipt lands in `--out` (default `.`).

use hip_bridge::DeviceBuffer;
use hipfire_runtime::ar_hc_exchange::{ArHcExchange, ArHcMode};
use rdna_compute::ar_hc_write::{
    ar_hc_fold, bf16_bits, bf16_round, sp_ag_rows_per_launch, sp_even_split, sp_slices,
    sp_wire_dword, ArHcWire, SpAgElem, StreamKind, AR_HC_HIDDEN, SP_ROW_CAP,
};
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::{Arc, Barrier};
use std::time::Instant;

const H: usize = AR_HC_HIDDEN;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A finite bf16 bit pattern (the exponent is never all ones).
    fn finite_bf16(&mut self) -> u16 {
        let b = (self.next() >> 48) as u16;
        if b & 0x7F80 == 0x7F80 {
            b ^ 0x4000
        } else {
            b
        }
    }
}

fn es<E: ToString>(e: E) -> String {
    e.to_string()
}
fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).position(|(x, y)| x != y)
}
fn off(p: *mut c_void, bytes: usize) -> *mut c_void {
    // SAFETY: only used to form in-bounds sub-pointers of device allocations.
    unsafe { p.cast::<u8>().add(bytes).cast::<c_void>() }
}

#[derive(Clone, Debug)]
struct Split {
    name: &'static str,
    ranges: [(usize, usize); 2],
}

fn splits_for(rows: usize) -> Vec<Split> {
    let mut v = vec![Split {
        name: "even",
        ranges: sp_even_split(rows),
    }];
    let floor = rows / 2;
    let candidates = [
        Split {
            name: "reverse",
            ranges: [(0, floor), (floor, rows - floor)],
        },
        Split {
            name: "skew",
            ranges: [(0, rows / 4), (rows / 4, rows - rows / 4)],
        },
    ];
    for c in candidates {
        if v.iter().all(|s| s.ranges != c.ranges) {
            v.push(c);
        }
    }
    v
}

/// One rank's device state: the largest tile of the run.
struct Dev {
    rank: usize,
    gpu: Gpu,
    partial: DeviceBuffer,
    streams_a: DeviceBuffer,
    streams_b: DeviceBuffer,
    gates: DeviceBuffer,
    dbg_a: DeviceBuffer,
    dbg_b: DeviceBuffer,
    ag16: DeviceBuffer,
    ag32: DeviceBuffer,
}

impl Dev {
    fn new(rank: usize, gpu: Gpu, max_rows: usize) -> Result<Self, String> {
        let alloc = |n: usize| gpu.hip.malloc(n).map_err(es);
        let partial = alloc(max_rows * H * 4)?;
        let streams_a = alloc(max_rows * StreamKind::F32.row_bytes())?;
        let streams_b = alloc(max_rows * StreamKind::F32.row_bytes())?;
        let gates = alloc(max_rows * 16)?;
        let dbg_a = alloc(max_rows * H * 4)?;
        let dbg_b = alloc(max_rows * H * 4)?;
        let ag16 = alloc(max_rows * H * 2)?;
        let ag32 = alloc(max_rows * H * 4)?;
        Ok(Self {
            rank,
            gpu,
            partial,
            streams_a,
            streams_b,
            gates,
            dbg_a,
            dbg_b,
            ag16,
            ag32,
        })
    }

    /// Free every buffer and hand the `Gpu` back.
    fn into_gpu(self) -> Result<Gpu, String> {
        let Dev {
            gpu,
            partial,
            streams_a,
            streams_b,
            gates,
            dbg_a,
            dbg_b,
            ag16,
            ag32,
            ..
        } = self;
        for b in [
            partial, streams_a, streams_b, gates, dbg_a, dbg_b, ag16, ag32,
        ] {
            gpu.hip.free(b).map_err(es)?;
        }
        Ok(gpu)
    }
}

/// Drain the rank, fail on a recorded wait timeout, and check that the dense bank the launch
/// used is back to all-EMPTY. Returns a failure note for a bank that was not re-armed.
fn check_launch(
    x: &ArHcExchange,
    gpu: &mut Gpu,
    rank: usize,
    what: &str,
) -> Result<Option<String>, String> {
    gpu.hip.device_synchronize().map_err(es)?;
    gpu.take_device_wait_error()
        .map_err(|e| format!("{what}: {e}"))?;
    let gen = x.generation(rank, gpu).map_err(es)?;
    let mb = x.download_mailbox(rank, gpu, ArHcWire::Dense).map_err(es)?;
    let (stride, bank) = (x.bank_stride(ArHcWire::Dense), (gen & 1) as usize);
    Ok(mb[bank * stride..(bank + 1) * stride]
        .iter()
        .position(|&b| b != 0xFF)
        .map(|at| {
            format!(
                "{what}: mailbox bank {bank} byte {at} not re-armed ({:#04x})",
                mb[bank * stride + at]
            )
        }))
}

/// The product all-reduce on `rows` rows in <= 256-row launches, drained and bank-checked
/// after each.
#[allow(clippy::too_many_arguments)]
fn ar_all(
    x: &ArHcExchange,
    d: &mut Dev,
    kind: StreamKind,
    streams: *mut c_void,
    dbg: *mut c_void,
    rows: usize,
    fails: &mut Vec<String>,
) -> Result<(), String> {
    let partial = d.partial.as_ptr();
    let gates = d.gates.as_ptr();
    let mut done = 0;
    while done < rows {
        let n = SP_ROW_CAP.min(rows - done);
        // SAFETY: slices of live full-tile allocations of this rank's device.
        unsafe {
            x.write(
                d.rank,
                &mut d.gpu,
                off(partial, done * H * 4),
                off(streams, done * kind.row_bytes()),
                kind,
                off(gates, done * 16),
                n,
                off(dbg, done * H * 4),
            )
        }
        .map_err(es)?;
        let what = format!("AR rows {done}..{}", done + n);
        if let Some(f) = check_launch(x, &mut d.gpu, d.rank, &what)? {
            fails.push(f);
        }
        done += n;
    }
    Ok(())
}

/// The RS launches of one tile (<= 256 rows per range per launch).
#[allow(clippy::too_many_arguments)]
fn rs_all(
    x: &ArHcExchange,
    d: &mut Dev,
    kind: StreamKind,
    streams: *mut c_void,
    dbg: *mut c_void,
    own: (usize, usize),
    peer: (usize, usize),
    fails: &mut Vec<String>,
) -> Result<(), String> {
    let partial = d.partial.as_ptr();
    let gates = d.gates.as_ptr();
    for ((oo, on), (po, pn)) in sp_slices(own.1, peer.1, SP_ROW_CAP) {
        // SAFETY: as above.
        unsafe {
            x.sp_rs_write(
                d.rank,
                &mut d.gpu,
                partial,
                streams,
                kind,
                gates,
                (own.0 + oo, on),
                (peer.0 + po, pn),
                dbg,
            )
        }
        .map_err(es)?;
        let what = format!("RS own {:?} peer {:?}", (own.0 + oo, on), (peer.0 + po, pn));
        if let Some(f) = check_launch(x, &mut d.gpu, d.rank, &what)? {
            fails.push(f);
        }
    }
    Ok(())
}

/// The AG launches of one buffer.
#[allow(clippy::too_many_arguments)]
fn ag_all(
    x: &ArHcExchange,
    d: &mut Dev,
    buf: *mut c_void,
    elem: SpAgElem,
    row_elems: usize,
    own: (usize, usize),
    peer: (usize, usize),
    fails: &mut Vec<String>,
) -> Result<(), String> {
    let cap = sp_ag_rows_per_launch(row_elems, x.bank_stride(ArHcWire::Dense)).min(SP_ROW_CAP);
    for ((oo, on), (po, pn)) in sp_slices(own.1, peer.1, cap) {
        // SAFETY: as above.
        unsafe {
            x.sp_ag_rows(
                d.rank,
                &mut d.gpu,
                buf,
                elem,
                row_elems,
                (own.0 + oo, on),
                (peer.0 + po, pn),
            )
        }
        .map_err(es)?;
        let what = format!(
            "AG {} x{row_elems} own {:?} peer {:?}",
            elem.name(),
            (own.0 + oo, on),
            (peer.0 + po, pn)
        );
        if let Some(f) = check_launch(x, &mut d.gpu, d.rank, &what)? {
            fails.push(f);
        }
    }
    Ok(())
}

fn init_streams(kind: StreamKind, rows: usize) -> Vec<u8> {
    let mut rng = Rng::new(0x5eed_0000 + rows as u64);
    let wide = rows * 4 * H;
    match kind {
        StreamKind::Bf16 => bytes_u16(
            &(0..wide)
                .map(|_| bf16_bits((rng.unit() - 0.5) * 4.0))
                .collect::<Vec<u16>>(),
        ),
        StreamKind::F32 => bytes_f32(
            &(0..wide)
                .map(|i| {
                    let v = (rng.unit() - 0.5) * 4.0;
                    // bf16-exact like the forward stores, every 16th unrounded
                    if i % 16 == 0 {
                        v
                    } else {
                        bf16_round(v)
                    }
                })
                .collect::<Vec<f32>>(),
        ),
    }
}

fn gen_partial(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let mag = 2f32.powi(-6 + rng.below(10) as i32);
            (rng.unit() - 0.5) * 2.0 * mag
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Phase {
    Ar,
    Rs,
    Ag(SpAgElem, usize),
}

/// One full step: RS vs AR over `rows` rows of `kind` at `split`, then the four AG rounds.
#[allow(clippy::too_many_arguments)]
fn step(
    d: &mut Dev,
    x: &ArHcExchange,
    bar: &Barrier,
    kind: StreamKind,
    rows: usize,
    split: &Split,
    init: &[u8],
    seed: u64,
    flip: bool,
) -> Result<Vec<String>, String> {
    let rank = d.rank;
    let n = rows * H;
    let rb = kind.row_bytes();
    let mut rng = Rng::new(seed);
    let p = [gen_partial(&mut rng, n), gen_partial(&mut rng, n)];
    let gates_h: Vec<f32> = (0..rows * 4).map(|_| (rng.unit() - 0.5) * 8.0).collect();
    let (own, peer) = (split.ranges[rank], split.ranges[1 - rank]);
    let hip = &d.gpu.hip;
    hip.memcpy_htod(&d.partial, &bytes_f32(&p[rank]))
        .map_err(es)?;
    hip.memcpy_htod(&d.gates, &bytes_f32(&gates_h))
        .map_err(es)?;
    hip.memcpy_htod(&d.streams_a, &init[..rows * rb])
        .map_err(es)?;
    hip.memcpy_htod(&d.streams_b, &init[..rows * rb])
        .map_err(es)?;
    hip.memset(&d.dbg_a, 0, n * 4).map_err(es)?;
    hip.memset(&d.dbg_b, 0, n * 4).map_err(es)?;
    let mut fails: Vec<String> = Vec::new();
    let order: [Phase; 6] = if flip {
        [
            Phase::Ag(SpAgElem::Bf16, 2560),
            Phase::Ag(SpAgElem::F32Bf16, 2560),
            Phase::Rs,
            Phase::Ar,
            Phase::Ag(SpAgElem::Bf16, 640),
            Phase::Ag(SpAgElem::F32Bf16, 640),
        ]
    } else {
        [
            Phase::Ar,
            Phase::Rs,
            Phase::Ag(SpAgElem::Bf16, 2560),
            Phase::Ag(SpAgElem::F32Bf16, 2560),
            Phase::Ag(SpAgElem::Bf16, 640),
            Phase::Ag(SpAgElem::F32Bf16, 640),
        ]
    };
    let (sa, sb) = (d.streams_a.as_ptr(), d.streams_b.as_ptr());
    let (da, db) = (d.dbg_a.as_ptr(), d.dbg_b.as_ptr());
    for phase in order {
        bar.wait();
        match phase {
            Phase::Ar => ar_all(x, d, kind, sa, da, rows, &mut fails)?,
            Phase::Rs => rs_all(x, d, kind, sb, db, own, peer, &mut fails)?,
            Phase::Ag(elem, row_elems) => {
                ag_phase(d, x, elem, row_elems, rows, split, &mut rng, &mut fails)?
            }
        }
    }
    // (i) RS vs AR
    let mut got_a = vec![0u8; rows * rb];
    let mut got_b = vec![0u8; rows * rb];
    d.gpu
        .hip
        .memcpy_dtoh(&mut got_a, &d.streams_a)
        .map_err(es)?;
    d.gpu
        .hip
        .memcpy_dtoh(&mut got_b, &d.streams_b)
        .map_err(es)?;
    let (o0, o1) = (own.0, own.0 + own.1);
    if let Some(i) = first_diff(&got_a[o0 * rb..o1 * rb], &got_b[o0 * rb..o1 * rb]) {
        fails.push(format!(
            "RS own rows != AR at stream byte {}: {:#04x} vs {:#04x}",
            o0 * rb + i,
            got_b[o0 * rb + i],
            got_a[o0 * rb + i]
        ));
    }
    for (lo, hi) in [(0, o0), (o1, rows)] {
        if let Some(i) = first_diff(&got_b[lo * rb..hi * rb], &init[lo * rb..hi * rb]) {
            fails.push(format!(
                "RS wrote a row outside own {own:?} (stream byte {})",
                lo * rb + i
            ));
        }
    }
    let mut dbg_a = vec![0u8; n * 4];
    let mut dbg_b = vec![0u8; n * 4];
    d.gpu.hip.memcpy_dtoh(&mut dbg_a, &d.dbg_a).map_err(es)?;
    d.gpu.hip.memcpy_dtoh(&mut dbg_b, &d.dbg_b).map_err(es)?;
    let (e0, e1) = (o0 * H * 4, o1 * H * 4);
    if let Some(i) = first_diff(&dbg_a[e0..e1], &dbg_b[e0..e1]) {
        fails.push(format!("RS m != AR m at byte {}", e0 + i));
    }
    let m_bits = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    for i in o0 * H..o1 * H {
        if m_bits(&dbg_b, i) != ar_hc_fold(p[0][i], p[1][i]).to_bits() {
            fails.push(format!("RS m != host fold at element {i}"));
            break;
        }
    }
    for (lo, hi) in [(0, e0), (e1, n * 4)] {
        if dbg_b[lo..hi].iter().any(|&b| b != 0) {
            fails.push("RS wrote mixed_dbg outside the own rows".into());
        }
    }
    Ok(fails)
}

/// One AG round: own rows of a `[rows][row_elems]` buffer hold random finite bf16 (a NaN-pair
/// dword planted at random), the peer rows garbage; after the gather the buffer must equal the
/// own rows plus the wire image of the peer rows.
#[allow(clippy::too_many_arguments)]
fn ag_phase(
    d: &mut Dev,
    x: &ArHcExchange,
    elem: SpAgElem,
    row_elems: usize,
    rows: usize,
    split: &Split,
    rng: &mut Rng,
    fails: &mut Vec<String>,
) -> Result<(), String> {
    let rank = d.rank;
    let (own, peer) = (split.ranges[rank], split.ranges[1 - rank]);
    let total = rows * row_elems;
    let eb = elem.element_bytes();
    // `full`: the concatenation both ranks agree on; `wire`: what a RECEIVING rank sees.
    let mut bits: Vec<u16> = (0..total).map(|_| rng.finite_bf16()).collect();
    for _ in 0..2 {
        // NaN pair: the two elements of one wire dword
        let row = rng.below(rows as u64) as usize;
        let pair = rng.below((row_elems / 2) as u64) as usize;
        let at = row * row_elems + 2 * pair;
        bits[at] = 0xFFFF;
        bits[at + 1] = 0xFFFF;
    }
    let (full, wire): (Vec<u8>, Vec<u8>) = match elem {
        SpAgElem::Bf16 => {
            let wire: Vec<u16> = bits
                .chunks_exact(2)
                .flat_map(|c| {
                    let w = sp_wire_dword(u32::from(c[0]) | (u32::from(c[1]) << 16));
                    [w as u16, (w >> 16) as u16]
                })
                .collect();
            (bytes_u16(&bits), bytes_u16(&wire))
        }
        SpAgElem::F32Bf16 => {
            let f: Vec<u32> = bits.iter().map(|&b| u32::from(b) << 16).collect();
            let wire: Vec<u32> = f
                .chunks_exact(2)
                .flat_map(|c| {
                    let w = sp_wire_dword((c[0] >> 16) | (c[1] & 0xFFFF_0000));
                    [w << 16, w & 0xFFFF_0000]
                })
                .collect();
            (bytes_u32(&f), bytes_u32(&wire))
        }
    };
    let row_b = row_elems * eb;
    let mut init = vec![0u8; rows * row_b];
    let mut want = vec![0u8; rows * row_b];
    // Garbage for the non-own rows comes from a rank-local stream: the shared `rng` must draw the
    // same count on both ranks (an uneven split has different non-own row counts per rank), or the
    // next phase's `bits` would differ between the ranks.
    let mut junk =
        Rng::new(0x5eed ^ ((rank as u64) << 32) ^ (rows as u64) ^ ((row_elems as u64) << 16));
    for row in 0..rows {
        let r = row * row_b..(row + 1) * row_b;
        if row >= own.0 && row < own.0 + own.1 {
            init[r.clone()].copy_from_slice(&full[r.clone()]);
            want[r.clone()].copy_from_slice(&full[r]);
        } else {
            for b in init[r.clone()].iter_mut() {
                *b = junk.next() as u8;
            }
            want[r.clone()].copy_from_slice(&wire[r]);
        }
    }
    let buf = match elem {
        SpAgElem::Bf16 => d.ag16.as_ptr(),
        SpAgElem::F32Bf16 => d.ag32.as_ptr(),
    };
    let target = match elem {
        SpAgElem::Bf16 => &d.ag16,
        SpAgElem::F32Bf16 => &d.ag32,
    };
    d.gpu.hip.memcpy_htod(target, &init).map_err(es)?;
    ag_all(x, d, buf, elem, row_elems, own, peer, fails)?;
    let target = match elem {
        SpAgElem::Bf16 => &d.ag16,
        SpAgElem::F32Bf16 => &d.ag32,
    };
    let mut got = vec![0u8; rows * row_b];
    d.gpu.hip.memcpy_dtoh(&mut got, target).map_err(es)?;
    if let Some(i) = first_diff(&got, &want) {
        fails.push(format!(
            "AG {} x{row_elems}: buffer differs at row {} byte {}: {:#04x} vs {:#04x}",
            elem.name(),
            i / row_b,
            i % row_b,
            got[i],
            want[i]
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Case {
    kind: &'static str,
    rows: usize,
    split: String,
    steps: usize,
    fails: usize,
    first: Option<String>,
}

impl Case {
    fn new(kind: &'static str, rows: usize, split: &str) -> Self {
        Self {
            kind,
            rows,
            split: split.to_string(),
            steps: 0,
            fails: 0,
            first: None,
        }
    }
    fn record(&mut self, rank: usize, it: usize, fails: Vec<String>) {
        self.steps += 1;
        self.fails += fails.len();
        if self.first.is_none() {
            if let Some(f) = fails.into_iter().next() {
                self.first = Some(format!(
                    "{} rows {} split {} iter {it} rank {rank}: {f}",
                    self.kind, self.rows, self.split
                ));
            }
        }
    }
}

#[derive(Clone, Debug)]
struct BenchRow {
    kind: &'static str,
    ar_us: f64,
    rs_us: f64,
    ag_bf16_us: f64,
    ag_f32bf16_us: f64,
    rs_ag_us: f64,
}

struct RankOut {
    cases: Vec<Case>,
    chained: Case,
    faults: Vec<String>,
    bench: Vec<BenchRow>,
}

struct Config {
    rows_list: Vec<usize>,
    iters: usize,
    case_iters: usize,
    kinds: Vec<StreamKind>,
    fault: bool,
    bench: bool,
}

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Rows of the bench tile.
const BENCH_ROWS: usize = 4096;

/// Median us of 20 runs (after 3 warm-ups) of `f`, each started after a barrier and drained.
fn time_us(
    d: &mut Dev,
    bar: &Barrier,
    f: &mut dyn FnMut(&mut Dev) -> Result<(), String>,
) -> Result<f64, String> {
    let (warm, samples) = (3, 20);
    let mut v = Vec::new();
    for s in 0..warm + samples {
        bar.wait();
        let t0 = Instant::now();
        f(d)?;
        d.gpu.hip.device_synchronize().map_err(es)?;
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        d.gpu.take_device_wait_error().map_err(es)?;
        if s >= warm {
            v.push(dt);
        }
    }
    Ok(median_us(v))
}

fn bench_rank(
    d: &mut Dev,
    kinds: &[StreamKind],
    bar: &Barrier,
    x: &ArHcExchange,
) -> Result<Vec<BenchRow>, String> {
    const ROWS: usize = BENCH_ROWS;
    let rank = d.rank;
    let mut rng = Rng::new(0xbe9c);
    let part: Vec<f32> = gen_partial(&mut rng, ROWS * H);
    d.gpu
        .hip
        .memcpy_htod(&d.partial, &bytes_f32(&part))
        .map_err(es)?;
    d.gpu.hip.memset(&d.gates, 0, ROWS * 16).map_err(es)?;
    d.gpu.hip.memset(&d.ag16, 0x3c, ROWS * H * 2).map_err(es)?;
    d.gpu.hip.memset(&d.ag32, 0x3c, ROWS * H * 4).map_err(es)?;
    let split = sp_even_split(ROWS);
    let (own, peer) = (split[rank], split[1 - rank]);
    let mut out = Vec::new();
    for &kind in kinds {
        d.gpu
            .hip
            .memset(&d.streams_a, 0, ROWS * kind.row_bytes())
            .map_err(es)?;
        let (sa, null) = (d.streams_a.as_ptr(), std::ptr::null_mut::<c_void>());
        let gates = d.gates.as_ptr();
        let partial = d.partial.as_ptr();
        let ag16 = d.ag16.as_ptr();
        let ag32 = d.ag32.as_ptr();
        // SAFETY (all closures): full 4096-row tiles of live allocations on this device.
        let ar_us = time_us(d, bar, &mut |d: &mut Dev| unsafe {
            x.write(rank, &mut d.gpu, partial, sa, kind, gates, ROWS, null)
                .map_err(es)
        })?;
        let rs_us = time_us(d, bar, &mut |d: &mut Dev| unsafe {
            x.sp_rs_write(rank, &mut d.gpu, partial, sa, kind, gates, own, peer, null)
                .map_err(es)
        })?;
        let ag_bf16_us = time_us(d, bar, &mut |d: &mut Dev| unsafe {
            x.sp_ag_rows(rank, &mut d.gpu, ag16, SpAgElem::Bf16, H, own, peer)
                .map_err(es)
        })?;
        let ag_f32bf16_us = time_us(d, bar, &mut |d: &mut Dev| unsafe {
            x.sp_ag_rows(rank, &mut d.gpu, ag32, SpAgElem::F32Bf16, H, own, peer)
                .map_err(es)
        })?;
        let rs_ag_us = time_us(d, bar, &mut |d: &mut Dev| unsafe {
            x.sp_rs_write(rank, &mut d.gpu, partial, sa, kind, gates, own, peer, null)
                .map_err(es)?;
            x.sp_ag_rows(rank, &mut d.gpu, ag16, SpAgElem::Bf16, H, own, peer)
                .map_err(es)
        })?;
        out.push(BenchRow {
            kind: kind.name(),
            ar_us,
            rs_us,
            ag_bf16_us,
            ag_f32bf16_us,
            rs_ag_us,
        });
    }
    Ok(out)
}

fn run_rank(
    rank: usize,
    device: i32,
    cfg: Arc<Config>,
    x: Arc<ArHcExchange>,
    bar: Arc<Barrier>,
) -> Result<RankOut, String> {
    let mut gpu = Gpu::init_with_device(device).map_err(es)?;
    gpu.ensure_device_wait_block().map_err(es)?;
    x.register(rank, &mut gpu).map_err(es)?;
    bar.wait();
    x.enable_peers(rank, &mut gpu).map_err(es)?;
    bar.wait();
    x.sp_ensure_kernels(&mut gpu).map_err(es)?;
    x.qualify(rank, &mut gpu, 8, 16)
        .map_err(|e| format!("qualification: {e}"))?;
    bar.wait();
    let max_rows = *cfg.rows_list.iter().max().unwrap();
    let dev_rows = if cfg.bench {
        max_rows.max(BENCH_ROWS)
    } else {
        max_rows
    };
    let mut d = Dev::new(rank, gpu, dev_rows)?;
    let inits: Vec<(StreamKind, Vec<u8>)> = cfg
        .kinds
        .iter()
        .map(|&k| (k, init_streams(k, max_rows)))
        .collect();
    let init_of = |kind: StreamKind| &inits.iter().find(|(k, _)| *k == kind).unwrap().1;
    // per-case phase, then the chained loop over the same cases
    let mut all: Vec<(StreamKind, usize, Split)> = Vec::new();
    for &kind in &cfg.kinds {
        for &rows in &cfg.rows_list {
            for split in splits_for(rows) {
                all.push((kind, rows, split));
            }
        }
    }
    let mut cases = Vec::new();
    for (ci, (kind, rows, split)) in all.iter().enumerate() {
        let mut case = Case::new(kind.name(), *rows, split.name);
        for it in 0..cfg.case_iters {
            let seed = 0x1000_0000 + (ci * 64 + it) as u64;
            let fails = step(
                &mut d,
                &x,
                &bar,
                *kind,
                *rows,
                split,
                init_of(*kind),
                seed,
                it % 2 == 1,
            )?;
            case.record(rank, it, fails);
        }
        cases.push(case);
    }
    let mut chained = Case::new("chained", 0, "rotating");
    for it in 0..cfg.iters {
        let (kind, rows, split) = &all[it % all.len()];
        let seed = 0x2000_0000 + it as u64;
        let fails = step(
            &mut d,
            &x,
            &bar,
            *kind,
            *rows,
            split,
            init_of(*kind),
            seed,
            it % 2 == 1,
        )?;
        if !fails.is_empty() && chained.first.is_none() {
            chained.first = Some(format!(
                "{} rows {rows} split {} iter {it} rank {rank}: {}",
                kind.name(),
                split.name,
                fails[0]
            ));
        }
        chained.steps += 1;
        chained.fails += fails.len();
    }
    // bounded-wait faults: rank 1 skips the launch, rank 0's wait must time out and say so
    let mut faults = Vec::new();
    if cfg.fault {
        for (label, symbol) in [
            ("RS", "sp_hc_rs_d8_gfx1201"),
            ("AG", "sp_ag_rows_bf16_d8_gfx1201"),
        ] {
            bar.wait();
            if rank == 0 {
                let (partial, streams, gates, ag) = (
                    d.partial.as_ptr(),
                    d.streams_b.as_ptr(),
                    d.gates.as_ptr(),
                    d.ag16.as_ptr(),
                );
                // SAFETY: tiles of two rows inside the allocations.
                let launched = unsafe {
                    if label == "RS" {
                        x.sp_rs_write(
                            0,
                            &mut d.gpu,
                            partial,
                            streams,
                            StreamKind::Bf16,
                            gates,
                            (0, 1),
                            (1, 1),
                            std::ptr::null_mut(),
                        )
                    } else {
                        x.sp_ag_rows(0, &mut d.gpu, ag, SpAgElem::Bf16, H, (0, 1), (1, 1))
                    }
                };
                launched.map_err(es)?;
                d.gpu.hip.device_synchronize().map_err(es)?;
                faults.push(match d.gpu.take_device_wait_error() {
                    Ok(()) => format!("FAIL: the unmatched {label} launch did not time out"),
                    Err(e) => {
                        let s = e.to_string();
                        if s.contains(symbol) {
                            format!("ok: {label}: {s}")
                        } else {
                            format!("FAIL: {label}: wrong site: {s}")
                        }
                    }
                });
            }
            bar.wait();
            // both ranks reset after a wait error before reuse
            x.reset(rank, &mut d.gpu).map_err(es)?;
        }
    }
    bar.wait();
    let bench = if cfg.bench {
        bench_rank(&mut d, &cfg.kinds, &bar, &x)?
    } else {
        Vec::new()
    };
    bar.wait();
    let mut gpu = d.into_gpu()?;
    bar.wait();
    x.release(rank, &mut gpu).map_err(es)?;
    Ok(RankOut {
        cases,
        chained,
        faults,
        bench,
    })
}

fn main() {
    let mut cfg = Config {
        rows_list: vec![1, 2, 7, 16, 255, 256, 300, 511, 512, 613],
        iters: 200,
        case_iters: 3,
        kinds: StreamKind::ALL.to_vec(),
        fault: true,
        bench: false,
    };
    let mut devices = [0i32, 1i32];
    let mut out = ".".to_string();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rows" => {
                i += 1;
                cfg.rows_list = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("rows"))
                    .collect();
            }
            "--iters" => {
                i += 1;
                cfg.iters = args[i].parse().expect("iters");
            }
            "--case-iters" => {
                i += 1;
                cfg.case_iters = args[i].parse().expect("case-iters");
            }
            "--devices" => {
                i += 1;
                let d: Vec<i32> = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("device"))
                    .collect();
                devices = [d[0], d[1]];
            }
            "--out" => {
                i += 1;
                out = args[i].clone();
            }
            "--stream" => {
                i += 1;
                cfg.kinds = match args[i].as_str() {
                    "bf16" => vec![StreamKind::Bf16],
                    "f32" => vec![StreamKind::F32],
                    "both" => StreamKind::ALL.to_vec(),
                    other => panic!("--stream takes f32|bf16|both, got {other}"),
                };
            }
            "--no-fault" => cfg.fault = false,
            "--bench" => cfg.bench = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    assert!(!cfg.rows_list.is_empty() && cfg.rows_list.iter().all(|&r| r > 0));
    let x = Arc::new(ArHcExchange::with_mode(SP_ROW_CAP, ArHcMode::Dense).expect("exchange"));
    let bar = Arc::new(Barrier::new(2));
    let cfg = Arc::new(cfg);
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (x, b, c) = (x.clone(), bar.clone(), cfg.clone());
            let dev = devices[rank];
            std::thread::spawn(move || {
                match run_rank(rank, dev, c, x, b) {
                    Ok(r) => Ok(r),
                    Err(e) => {
                        // the peer thread may be parked on a barrier: do not wait for it
                        eprintln!("rank {rank}: FATAL {e}");
                        eprintln!("sp_rs_ag_oracle: FAIL (rank {rank} fatal)");
                        std::process::exit(3);
                    }
                }
            })
        })
        .collect();
    let results: Vec<Result<RankOut, String>> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    let mut failed = false;
    let mut json_rows = Vec::new();
    let mut json_bench = Vec::new();
    let mut checks = 0usize;
    for (rank, r) in results.iter().enumerate() {
        let res = r.as_ref().expect("fatal errors exit in the thread");
        for c in res.cases.iter().chain(std::iter::once(&res.chained)) {
            checks += c.steps;
            let bad = c.fails != 0;
            failed |= bad;
            if bad || c.kind == "chained" {
                eprintln!(
                    "rank {rank} {} rows {} split {}: {} steps, {} failures{}",
                    c.kind,
                    c.rows,
                    c.split,
                    c.steps,
                    c.fails,
                    c.first
                        .as_deref()
                        .map_or(String::new(), |f| format!(" FIRST: {f}"))
                );
            }
            json_rows.push(format!(
                "{{\"rank\":{rank},\"stream\":\"{}\",\"rows\":{},\"split\":\"{}\",\"steps\":{},\"failures\":{},\"pass\":{}}}",
                c.kind, c.rows, c.split, c.steps, c.fails, !bad
            ));
        }
        for f in &res.faults {
            eprintln!("rank {rank} fault check: {f}");
            failed |= f.starts_with("FAIL");
        }
        for b in &res.bench {
            eprintln!(
                "rank {rank} bench {} (4096-row tile): AR {:.1} us  RS {:.1}  AG bf16 {:.1}  AG f32bf16 {:.1}  RS+AG {:.1}  (RS+AG / AR = {:.3})",
                b.kind, b.ar_us, b.rs_us, b.ag_bf16_us, b.ag_f32bf16_us, b.rs_ag_us, b.rs_ag_us / b.ar_us
            );
            json_bench.push(format!(
                "{{\"rank\":{rank},\"stream\":\"{}\",\"tile_rows\":4096,\"ar_us\":{:.2},\"rs_us\":{:.2},\"ag_bf16_us\":{:.2},\"ag_f32bf16_us\":{:.2},\"rs_ag_us\":{:.2}}}",
                b.kind, b.ar_us, b.rs_us, b.ag_bf16_us, b.ag_f32bf16_us, b.rs_ag_us
            ));
        }
    }
    let receipt = format!(
        "{{\"oracle\":\"sp_rs_ag_oracle\",\"pass\":{},\"iters\":{},\"case_iters\":{},\"streams\":[{}],\"devices\":[{},{}],\"results\":[{}],\"bench\":[{}]}}",
        !failed,
        cfg.iters,
        cfg.case_iters,
        cfg.kinds.iter().map(|k| format!("\"{}\"", k.name())).collect::<Vec<_>>().join(","),
        devices[0],
        devices[1],
        json_rows.join(","),
        json_bench.join(",")
    );
    let _ = std::fs::create_dir_all(&out);
    let path = format!("{out}/sp_rs_ag_oracle.json");
    let _ = std::fs::write(&path, &receipt);
    eprintln!("receipt: {path}");
    println!(
        "sp_rs_ag_oracle: {} ({checks} rank-steps, rows {:?}, streams {:?}, iters {})",
        if failed { "FAIL" } else { "PASS" },
        cfg.rows_list,
        cfg.kinds.iter().map(|k| k.name()).collect::<Vec<_>>(),
        cfg.iters
    );
    std::process::exit(if failed { 3 } else { 0 });
}
