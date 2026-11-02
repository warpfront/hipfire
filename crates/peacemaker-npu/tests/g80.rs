// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! G80 (`gemm_g80::design_g80`) in the whole-array simulator: a 256-row x 1280-column wave (128x80 core tile, a vertical core
//! pair shares one 128-row A block through E/O halves, N fixed at 1280, int8 SRS epilogue, `K = 64*kc`, `kc <= 40`).
//!
//! Proven here, for the generated PDI/TXN only:
//!   * every submit is CPU-exact (`clamp(floor(c / 2^shift), -128, 127)`, computed independently of the design), leaves the
//!     original and the packed operands unchanged and moves exactly the per-channel shim bytes of the layout
//!     (MM2S0 = B `kc*10240` per column, MM2S1 = A `waves*kc*4096` on columns 0..4 only, S2MM0/1 = `waves*2*10240`);
//!   * every core tile of every wave is delivered; the tile ranges tile the C argument exactly;
//!   * padded `M`, odd and even `kc*waves`, three seeds on one reused `Config`, signed extremes under every shift;
//!   * `no_compute` / `repeat` probes are not GEMMs but complete, with the same finite byte traffic;
//!   * explicit shape / epilogue / probe-layout boundaries are rejected, accepted boundaries build;
//!   * the grouped launch API (`append_run_body` with arena offsets): three one-wave experts M = 64 / 160 / 200 in ONE TXN
//!     equal their eager fresh-`Config` bytes and leave the retained state of a single full run;
//!   * lean submits (`lean_insts`, `lean_insts_requeue`, `append_lean_run_body`): full -> 3 lean submits with fresh operands
//!     are exact, byte-equal to a fresh full run with the same operands and leave `Config::state_snapshot` equal to a
//!     fresh full run; full/lean interleaves; the E3 group with lean tail bodies equals the all-full group (bytes and state).
//!   * `Epilogue::I32` (direct half-stripe int32 C, 40960 B per core tile, S2MM0/1 = `waves*2*40960`): every full / lean /
//!     forced-requeue / grouped submit is exact against the independent `cpu_reference` AND its complete padded packed C arena
//!     is rebuilt word by word by a test-local oracle (poisoned arena, padded rows included, every raw int32 bounded by
//!     `k*16384`), operands unchanged, tile ranges tile the arena once, memtile descriptors / locks / queues parsed from the TXN
//!     (`g80_i32_*`). The simulator timing comparison int8 vs I32 reports functional ticks only, never silicon cycles.
//!
//! Simulator limitations (not faked): `Config` has no consumer-delay / asymmetric drain pacing knob and no mid-run state
//! inspection (`state_snapshot` is a between-submit fingerprint, `submit_with`'s producer only publishes host bytes), and
//! no TXN op can inject into the middle of a body. The two-stream B readiness ordering (even / odd N tile, K chunk, replay
//! per M wave) is therefore proven through *distinct per-parity, per-chunk B patterns* whose every misordering changes C
//! (`pattern_b_*`), at one and two M waves, reusing one `Config` with the parity pattern flipped between submits. A
//! negative perturbed ready-credit run is not accessible for the same reason.
use peacemaker_npu::dma::Direction as DmaDirection;
use peacemaker_npu::kernels::{
    experts::{grouped, grouped_all_full, ExpertPlan, GroupedDesign},
    ring::{RingLayout, SENTINEL},
};
use peacemaker_npu::kernels::{
    gemm_array::ArrayDesign,
    gemm_core::{Control, Epilogue, Probe},
    gemm_g80::design_g80,
    gemm_i8::{cpu_reference, ArgKind, ArgSpec},
};
use peacemaker_npu::sim::{
    config::{parse_txn, Config, TxnOp},
    dma::Direction,
};
use peacemaker_npu::txn::Txn;
use std::cell::Cell;
use std::{
    hash::Hasher,
    ops::Range,
    panic::{catch_unwind, AssertUnwindSafe},
};

const N: usize = 1280;
const WAVE_M: usize = 256;
const COLS: usize = 8;
const A_COLS: usize = 4;
/// One K chunk of the A half a column injects (128 rows x 64 / 2), of the B segment of one column (two 80-wide N tiles x 64),
/// and one core C tile (128 x 80 int8).
const A_CHUNK: usize = 4096;
const B_CHUNK: usize = 10240;
const C_TILE: usize = 10240;
const POISON: u8 = 0xa5;
const GUARD: u8 = 0xcc;
const GAP: usize = 256;
const SEEDS: [u32; 3] = [0x12345678, 0x98765432, 0x2468ace1];
const BOTH: [Control; 2] = [Control::Fast, Control::Slow];
/// (m, k) with kc*waves = 1, 2, 3, 8, 9, 12 (odd and even) plus padded 200 rows (2) and 513 rows (3 waves, kc 1).
const REUSE_SHAPES: [(usize, usize); 8] = [
    (256, 64),
    (256, 128),
    (768, 64),
    (512, 256),
    (768, 192),
    (1024, 192),
    (200, 128),
    (513, 64),
];

fn int8(shift: u8) -> Epilogue {
    Epilogue::Int8 { shift }
}
fn design(m: usize, k: usize, ctl: Control, shift: u8) -> ArrayDesign {
    design_g80(m, N, k, int8(shift), ctl)
}
fn signature(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    (0..a.len()).find(|&i| a[i] != b[i])
}
/// Independent reference: exact int32 GEMM, arithmetic shift (floor), saturate to int8.
fn want(a: &[i8], b: &[i8], m: usize, k: usize, shift: u8) -> Vec<i32> {
    cpu_reference(a, b, m, N, k)
        .into_iter()
        .map(|c| ((c as i64) >> shift).clamp(-128, 127) as i32)
        .collect()
}

// ---------------------------------------------------------------- operands
fn random(m: usize, k: usize, seed: u32) -> (Vec<i8>, Vec<i8>) {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as i8
    };
    let mut a: Vec<_> = (0..m * k).map(|_| next()).collect();
    let mut b: Vec<_> = (0..k * N).map(|_| next()).collect();
    a[0] = -128;
    b[0] = -128;
    (a, b)
}
/// Row and column classes at the signed extremes: -128 / 127 / -1 / 1 / 0 / alternating / pseudo-random, so products reach
/// positive and negative saturation, exact boundaries and small negative sums whose floor differs from truncation.
fn extremes(m: usize, k: usize, rot: usize) -> (Vec<i8>, Vec<i8>) {
    let a = (0..m * k)
        .map(|x| {
            let (i, kk) = (x / k, x % k);
            match (i + rot) % 8 {
                0 => -128,
                1 => 127,
                2 => -1,
                3 => 1,
                4 => {
                    if kk % 2 == 0 {
                        -128
                    } else {
                        127
                    }
                }
                5 => {
                    if kk % 3 == 0 {
                        127
                    } else {
                        -1
                    }
                }
                6 => (kk * 37 + i) as i8,
                _ => 0,
            }
        })
        .collect();
    let b = (0..k * N)
        .map(|x| {
            let (kk, j) = (x / N, x % N);
            match (j + rot) % 8 {
                0 => -128,
                1 => 127,
                2 => -1,
                3 => 1,
                4 => {
                    if kk % 2 == 0 {
                        127
                    } else {
                        -128
                    }
                }
                5 => {
                    if kk % 3 == 1 {
                        -128
                    } else {
                        1
                    }
                }
                6 => (kk * 11 + j * 5) as i8,
                _ => 0,
            }
        })
        .collect();
    (a, b)
}
/// Every K chunk and each of the two N-tile parities of a core column (tile = j/80, parity = tile%2, flipped by `flip`) gets
/// its own value stream, so a consumer that takes the wrong parity, the wrong chunk or a stale chunk produces another C.
fn pattern_b(m: usize, k: usize, flip: bool) -> (Vec<i8>, Vec<i8>) {
    let a = (0..m * k)
        .map(|x| {
            let (i, kk) = (x / k, x % k);
            (((kk / 64) * 29 + (kk % 64) * 3 + i * 7 + (i / 128) * 11) % 255) as i32 - 127
        })
        .map(|v| v as i8)
        .collect();
    let b = (0..k * N)
        .map(|x| {
            let (kk, j) = (x / N, x % N);
            let t = j / 80;
            let parity = (t % 2) ^ usize::from(flip);
            ((((kk / 64) * 53 + parity * 101 + (t / 2) * 17 + (kk % 64) * 5 + (j % 80) * 3 + 1)
                % 255) as i32
                - 127) as i8
        })
        .collect();
    (a, b)
}

// ---------------------------------------------------------------- layout and traffic
type Traffic = Vec<(u8, Direction, u8, u64)>;
fn traffic(sim: &Config) -> Traffic {
    sim.shim_stats()
        .map(|c| (c.col, c.direction, c.channel, c.dram_bytes()))
        .collect()
}
fn core_sum(sim: &Config) -> (u64, u64) {
    sim.core_stats().fold((0, 0), |(busy, wait), c| {
        (busy + c.busy, wait + c.lock_wait)
    })
}
/// Expected bytes of one shim channel over `runs` bodies moving `waves` waves in total.
fn expected_bytes(dir: Direction, ch: u8, col: u8, runs: usize, waves: usize, kc: usize) -> u64 {
    (match (dir, ch) {
        (Direction::Mm2s, 0) => runs * kc * B_CHUNK,
        (Direction::Mm2s, 1) => {
            if (col as usize) < A_COLS {
                waves * kc * A_CHUNK
            } else {
                0
            }
        }
        (Direction::S2mm, 0) | (Direction::S2mm, 1) => waves * 2 * C_TILE,
        _ => panic!("unexpected shim channel {dir:?} {ch}"),
    }) as u64
}
/// Per-channel deltas equal the layout formula; returns (read, written) bytes of the delta.
fn assert_traffic(
    before: &Traffic,
    after: &Traffic,
    runs: usize,
    waves: usize,
    kc: usize,
    label: &str,
) -> (u64, u64) {
    assert_eq!(before.len(), after.len(), "{label}");
    let (mut read, mut written) = (0, 0);
    for (&(col, dir, ch, was), &(col1, dir1, ch1, now)) in before.iter().zip(after) {
        assert_eq!((col, dir, ch), (col1, dir1, ch1), "{label}");
        let delta = now - was;
        assert_eq!(
            delta,
            expected_bytes(dir, ch, col, runs, waves, kc),
            "{label}: shim col {col} {dir:?} ch {ch} bytes"
        );
        match dir {
            Direction::Mm2s => read += delta,
            Direction::S2mm => written += delta,
        }
    }
    (read, written)
}
/// Argument sizes, host byte totals, wave grid and the exact tiling of C by the per-tile ranges.
fn check_layout(d: &ArrayDesign, m: usize, k: usize, label: &str) {
    let (kc, waves) = (k / 64, m.div_ceil(WAVE_M));
    assert_eq!(d.waves(), waves, "{label}: waves");
    assert_eq!(d.wave_m(), WAVE_M, "{label}: wave_m");
    let sizes = [
        A_COLS * waves * kc * A_CHUNK,
        COLS * kc * B_CHUNK,
        COLS * waves * 4 * C_TILE,
    ];
    assert_eq!(
        d.args,
        vec![
            ArgSpec {
                bytes: sizes[0],
                kind: ArgKind::In
            },
            ArgSpec {
                bytes: sizes[1],
                kind: ArgKind::In
            },
            ArgSpec {
                bytes: sizes[2],
                kind: ArgKind::Out
            }
        ],
        "{label}: args"
    );
    assert_eq!(d.host_bytes_read(), (sizes[0] + sizes[1]) as u64, "{label}");
    assert_eq!(d.host_bytes_written(), sizes[2] as u64, "{label}");
    let tiles = d.output_tile_ranges();
    assert_eq!(tiles.len(), COLS * 4, "{label}: output tiles");
    let mut all: Vec<Range<usize>> = Vec::new();
    for ((col, row), ranges) in &tiles {
        assert_eq!(ranges.len(), waves, "{label}: tile ({col},{row}) ranges");
        for r in ranges {
            assert_eq!(r.len(), C_TILE, "{label}: tile ({col},{row}) range length");
            all.push(r.clone())
        }
    }
    all.sort_by_key(|r| r.start);
    let mut end = 0;
    for r in &all {
        assert_eq!(
            r.start, end,
            "{label}: C tile ranges must tile arg2 exactly"
        );
        end = r.end
    }
    assert_eq!(end, sizes[2], "{label}: C tile ranges cover arg2");
}

// ---------------------------------------------------------------- one checked submit
struct Case {
    label: String,
    m: usize,
    k: usize,
    shift: u8,
}
fn case(label: impl Into<String>, m: usize, k: usize, shift: u8) -> Case {
    Case {
        label: label.into(),
        m,
        k,
        shift,
    }
}
struct Run {
    ticks: usize,
    c: Vec<u8>,
}
/// One submit of `insts` (a full or lean TXN of `d`) on a possibly reused `Config`. `values`: C must equal the CPU reference
/// (false for timing probes: only completion, traffic and delivery are checked). Counters are cumulative, so everything is
/// a per-submit delta.
fn submit_checked(
    sim: &mut Config,
    d: &ArrayDesign,
    insts: &[u8],
    c: &Case,
    a: &[i8],
    b: &[i8],
    values: bool,
) -> Run {
    let Case { label, m, k, shift } = c;
    let (m, k, shift) = (*m, *k, *shift);
    let (kc, waves) = (k / 64, m.div_ceil(WAVE_M));
    let [a_host, b_host] = d.pack_in(a, b);
    assert_eq!(
        (a_host.len(), b_host.len()),
        (d.args[0].bytes, d.args[1].bytes),
        "{label}: packed sizes"
    );
    let (a0, b0) = (a.to_vec(), b.to_vec());
    let before = [signature(&a_host), signature(&b_host)];
    let mut args = vec![a_host, b_host, vec![POISON; d.args[2].bytes]];
    let (t0, ticks0, (busy0, wait0)) = (traffic(sim), sim.ticks, core_sum(sim));
    sim.submit(insts, &mut args)
        .unwrap_or_else(|e| panic!("{label} {m}x{N}x{k}: submit failed: {e}"));
    let ticks = sim.ticks - ticks0;
    let (busy1, wait1) = core_sum(sim);
    if values {
        let w = want(a, b, m, k, shift);
        assert_eq!(
            d.reference(a, b),
            w,
            "{label}: design reference != independent reference"
        );
        let got = d.unpack_out(&args[2]);
        assert_eq!(got.len(), w.len(), "{label}");
        if let Some(i) = (0..got.len()).find(|&i| got[i] != w[i]) {
            panic!(
                "{label} {m}x{N}x{k} shift {shift}: first mismatch at ({},{}) got {} want {}",
                i / N,
                i % N,
                got[i],
                w[i]
            )
        }
    }
    assert_eq!(
        [signature(&args[0]), signature(&args[1])],
        before,
        "{label}: packed inputs changed"
    );
    assert!(a == &a0[..] && b == &b0[..], "{label}: operands changed");
    let (read, written) = assert_traffic(&t0, &traffic(sim), 1, waves, kc, label);
    assert_eq!(
        (read, written),
        (d.host_bytes_read(), d.host_bytes_written()),
        "{label}: host byte totals"
    );
    for ((col, row), ranges) in d.output_tile_ranges() {
        for (w, r) in ranges.iter().enumerate() {
            assert!(
                args[2][r.clone()].iter().any(|&x| x != POISON),
                "{label}: tile ({col},{row}) wave {w} not delivered"
            );
        }
    }
    eprintln!("PASS {label} {m}x{N}x{k} shift {shift}: {}; {ticks} functional ticks (not hardware cycles); core busy={} lock_wait={}; DRAM read={read} write={written} \
        (per column MM2S0 B={} MM2S1 A={}(cols<4) S2MM0/1={} B)",if values {format!("{} CPU-exact words",m*N)} else {"timing probe, completion + bytes only".into()},
        busy1-busy0,wait1-wait0,kc*B_CHUNK,waves*kc*A_CHUNK,waves*2*C_TILE);
    Run {
        ticks,
        c: args.swap_remove(2),
    }
}
fn fresh(d: &ArrayDesign) -> Config {
    Config::from_pdi(&d.pdi).unwrap()
}
/// One fresh-`Config` full submit.
fn once(d: &ArrayDesign, c: &Case, a: &[i8], b: &[i8]) -> Run {
    submit_checked(&mut fresh(d), d, &d.insts, c, a, b, true)
}

// ---------------------------------------------------------------- exact shapes
fn shape(m: usize, k: usize, ctl: Control) {
    let d = design(m, k, ctl, 12);
    check_layout(&d, m, k, "layout");
    let (a, b) = random(m, k, 0xc001d00d);
    let run = once(&d, &case(format!("G80 {ctl:?}"), m, k, 12), &a, &b);
    let lean = d
        .lean_insts()
        .map(|l| format!("{} B", l.len()))
        .unwrap_or_else(|e| format!("n/a ({e})"));
    eprintln!("MEASURED G80 {ctl:?} {m}x{N}x{k}: {} ticks; PDI {} B, full TXN {} B, lean TXN {lean}; args A={} B={} C={} B; DRAM read {} write {}",
        run.ticks,d.pdi.len(),d.insts.len(),d.args[0].bytes,d.args[1].bytes,d.args[2].bytes,d.host_bytes_read(),d.host_bytes_written());
}
fn shape_both(m: usize, k: usize) {
    for ctl in BOTH {
        shape(m, k, ctl)
    }
}
#[test]
fn g80_exact_256_1280_64() {
    shape_both(256, 64)
}
#[test]
fn g80_exact_256_1280_128() {
    shape_both(256, 128)
}
#[test]
fn g80_exact_160_1280_2560() {
    shape_both(160, 2560)
}
#[test]
fn g80_exact_64_1280_2560() {
    shape_both(64, 2560)
}
#[test]
fn g80_exact_512_1280_640() {
    shape_both(512, 640)
}
#[test]
fn g80_exact_4096_1280_2560_fast() {
    shape(4096, 2560, Control::Fast)
}
#[test]
fn g80_exact_4096_1280_2560_slow() {
    shape(4096, 2560, Control::Slow)
}
/// Padded M: partial last wave (200, 255, 257, 513), a single row, and exact wave multiples.
#[test]
fn g80_exact_padded_m() {
    for (m, k) in [
        (200, 192),
        (513, 128),
        (255, 64),
        (257, 64),
        (1, 64),
        (200, 2560),
    ] {
        shape_both(m, k)
    }
}
/// Int8 shift sweep on random data (floor + saturate), one padded wave.
#[test]
fn g80_exact_random_shift_sweep() {
    for shift in [0u8, 1, 6, 12, 20, 31] {
        for ctl in BOTH {
            let (m, k) = (200, 128);
            let d = design(m, k, ctl, shift);
            let (a, b) = random(m, k, 0xfeed0001);
            once(
                &d,
                &case(format!("G80 {ctl:?} random"), m, k, shift),
                &a,
                &b,
            );
        }
    }
}
/// Signed extremes: floor of small negative sums, exact saturation boundary (shift 14, K = 128: 2^21 / 2^14 = 128) and both
/// saturations, through the reverse int8 epilogue overlaying its own C accumulator, at every shift, two padded waves.
#[test]
fn g80_extremes_floor_shift_saturation() {
    for shift in [0u8, 1, 6, 12, 14, 20, 31] {
        for ctl in BOTH {
            for rot in [0usize, 3] {
                let (m, k) = (300, 128);
                let d = design(m, k, ctl, shift);
                let (a, b) = extremes(m, k, rot);
                let raw = cpu_reference(&a, &b, m, N, k);
                if shift <= 12 {
                    assert!(
                        raw.iter().any(|&c| (c as i64) >> shift >= 128)
                            && raw.iter().any(|&c| (c as i64) >> shift <= -129),
                        "data must saturate both ways at shift {shift}"
                    )
                }
                if shift >= 1 {
                    assert!(raw.iter().any(|&c|c<0&&(c as i64)&((1i64<<shift)-1)!=0),"data must hold negative non-multiples (floor != truncation) at shift {shift}")
                }
                assert!(
                    want(&a, &b, m, k, shift).contains(&-128) || shift > 12,
                    "shift {shift}: negative saturation must be exercised"
                );
                once(
                    &d,
                    &case(format!("G80 {ctl:?} extremes rot {rot}"), m, k, shift),
                    &a,
                    &b,
                );
            }
        }
    }
}

// ---------------------------------------------------------------- context reuse
/// Three seeds on one `Config`: finite queues, lock initial values, iteration counters and ring parities must reset every
/// submit, for odd and even `kc*waves`.
#[test]
fn g80_reuses_context_three_seeds_odd_and_even() {
    for (m, k) in REUSE_SHAPES {
        for ctl in BOTH {
            let d = design(m, k, ctl, 12);
            let mut sim = fresh(&d);
            for seed in SEEDS {
                let (a, b) = random(m, k, seed);
                submit_checked(
                    &mut sim,
                    &d,
                    &d.insts,
                    &case(format!("G80 reuse {ctl:?} seed {seed:#x}"), m, k, 12),
                    &a,
                    &b,
                    true,
                );
            }
            eprintln!("PASS G80 same-context {ctl:?} {m}x{N}x{k} (kc*waves={}) three submits, {} functional ticks total",k/64*d.waves(),sim.ticks);
        }
    }
}

// ---------------------------------------------------------------- B stream readiness / ordering
/// Distinct B per parity and chunk, at one and two M waves (the second M wave replays B from the memtile), same `Config`
/// with the parity pattern flipped and flipped back.
fn pattern_b_case(kc: usize, mw: usize, ctl: Control) {
    let (m, k) = (WAVE_M * mw, 64 * kc);
    let d = design(m, k, ctl, 12);
    let mut sim = fresh(&d);
    let (a, b0) = pattern_b(m, k, false);
    let (_, b1) = pattern_b(m, k, true);
    assert_ne!(
        want(&a, &b0, m, k, 12),
        want(&a, &b1, m, k, 12),
        "pattern must be parity sensitive"
    );
    for (i, flip) in [false, true, false].into_iter().enumerate() {
        let (a, b) = pattern_b(m, k, flip);
        submit_checked(
            &mut sim,
            &d,
            &d.insts,
            &case(
                format!("G80 patternB {ctl:?} kc{kc} mw{mw} flip{i}"),
                m,
                k,
                12,
            ),
            &a,
            &b,
            true,
        );
    }
}
#[test]
fn g80_pattern_b_parities_and_chunks_at_one_and_two_m_waves() {
    for kc in [1, 2, 3, 5, 8] {
        for mw in [1, 2] {
            for ctl in BOTH {
                pattern_b_case(kc, mw, ctl)
            }
        }
    }
}
/// Full 40-chunk segment (5120 B per parity half-queue, K = 2560) at one and two M waves.
#[test]
fn g80_pattern_b_full_kc40() {
    for mw in [1, 2] {
        pattern_b_case(40, mw, Control::Fast)
    }
}

// ---------------------------------------------------------------- timing probes
/// Not GEMMs (`no_compute` / `repeat`): the lock / DMA protocol is unchanged, so the run completes, moves the same bytes,
/// delivers every tile and leaves the operands untouched; two submits on one `Config` prove the finite queues requeue.
fn probe_case(m: usize, k: usize, p: Probe) {
    let d = design_g80(m, N, k, int8(12), Control::Probe(p));
    check_layout(&d, m, k, "probe layout");
    let mut sim = fresh(&d);
    let mut ticks = Vec::new();
    for seed in [SEEDS[0], SEEDS[1]] {
        let (a, b) = random(m, k, seed);
        ticks.push(
            submit_checked(
                &mut sim,
                &d,
                &d.insts,
                &case(format!("G80 probe {p:?} seed {seed:#x}"), m, k, 12),
                &a,
                &b,
                false,
            )
            .ticks,
        );
    }
    let (a, b) = random(m, k, SEEDS[0]);
    let fast = once(
        &design(m, k, Control::Fast, 12),
        &case("G80 Fast reference", m, k, 12),
        &a,
        &b,
    );
    eprintln!(
        "MEASURED G80 probe {p:?} {m}x{N}x{k}: {} / {} ticks per submit vs Fast {} ticks",
        ticks[0], ticks[1], fast.ticks
    );
}
#[test]
fn g80_probe_no_compute_completes_with_exact_bytes() {
    for (m, k) in [(256, 128), (512, 192)] {
        probe_case(
            m,
            k,
            Probe {
                no_compute: true,
                ..Probe::default()
            },
        );
        probe_case(
            m,
            k,
            Probe {
                no_compute: true,
                serial: true,
                ..Probe::default()
            },
        )
    }
}
#[test]
fn g80_probe_repeat_completes_with_exact_bytes() {
    for (m, k) in [(256, 128), (300, 64)] {
        probe_case(
            m,
            k,
            Probe {
                repeat: 3,
                ..Probe::default()
            },
        )
    }
}
/// The probe spellings of the two plain controls are the plain controls.
#[test]
fn g80_probe_default_and_serial_are_still_exact_gemms() {
    let (m, k) = (200, 128);
    let (a, b) = random(m, k, SEEDS[2]);
    for p in [
        Probe::default(),
        Probe {
            serial: true,
            ..Probe::default()
        },
    ] {
        let d = design_g80(m, N, k, int8(12), Control::Probe(p));
        submit_checked(
            &mut fresh(&d),
            &d,
            &d.insts,
            &case(format!("G80 {p:?}"), m, k, 12),
            &a,
            &b,
            true,
        );
    }
}

// ---------------------------------------------------------------- unsupported shapes and boundaries
fn rejects(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}
fn assert_all_rejected(cases: Vec<(String, bool)>) {
    let accepted: Vec<_> = cases
        .iter()
        .filter(|c| !c.1)
        .map(|c| c.0.as_str())
        .collect();
    assert!(
        accepted.is_empty(),
        "unsupported but accepted: {accepted:?}"
    );
}
#[test]
fn g80_rejects_n_other_than_1280() {
    assert_all_rejected(
        [0usize, 1, 640, 1279, 1281, 2560, 2561]
            .into_iter()
            .map(|n| {
                (
                    format!("N={n}"),
                    rejects(|| {
                        design_g80(256, n, 128, int8(12), Control::Fast);
                    }),
                )
            })
            .collect(),
    );
}
#[test]
fn g80_rejects_k_outside_multiples_of_64_up_to_2560() {
    assert_all_rejected(
        [0usize, 1, 63, 65, 100, 2559, 2561, 2624, 5120]
            .into_iter()
            .map(|k| {
                (
                    format!("K={k}"),
                    rejects(|| {
                        design_g80(256, N, k, int8(12), Control::Fast);
                    }),
                )
            })
            .collect(),
    );
}
#[test]
fn g80_rejects_empty_and_more_than_256_waves() {
    assert_all_rejected(
        [0usize, 256 * 256 + 1, 256 * 257]
            .into_iter()
            .map(|m| {
                (
                    format!("M={m}"),
                    rejects(|| {
                        design_g80(m, N, 64, int8(12), Control::Fast);
                    }),
                )
            })
            .collect(),
    );
}
/// `Epilogue::I32` is a supported direct half-stripe epilogue (see the I32 section): the old rejection is gone and every
/// boundary shape builds with the 40960 B/core tile layout; other bounds still reject for it (`g80_i32_rejects_*`).
#[test]
fn g80_accepts_i32_epilogue_boundary_shapes() {
    for (m, k) in [
        (1, 64),
        (1, 2560),
        (256, 2560),
        (257, 64),
        (256 * 256, 64),
        (256 * 256, 2560),
    ] {
        for ctl in BOTH {
            check_layout32(&design32(m, k, ctl), m, k, &format!("I32 {ctl:?} boundary"))
        }
    }
}
#[test]
fn g80_rejects_probe_layouts_other_than_0() {
    assert_all_rejected(
        (1u8..=3)
            .map(|layout| {
                (
                    format!("layout={layout}"),
                    rejects(|| {
                        design_g80(
                            256,
                            N,
                            64,
                            int8(12),
                            Control::Probe(Probe {
                                layout,
                                ..Probe::default()
                            }),
                        );
                    }),
                )
            })
            .collect(),
    );
}
/// `pack_in` checks the operand lengths.
#[test]
fn g80_pack_rejects_wrong_operand_lengths() {
    let (m, k) = (200, 128);
    let d = design(m, k, Control::Fast, 12);
    let (a, b) = random(m, k, SEEDS[0]);
    assert_all_rejected(vec![
        (
            "A short".into(),
            rejects(|| {
                d.pack_in(&a[..a.len() - 1], &b);
            }),
        ),
        (
            "A long".into(),
            rejects(|| {
                let mut x = a.clone();
                x.push(0);
                d.pack_in(&x, &b);
            }),
        ),
        (
            "B short".into(),
            rejects(|| {
                d.pack_in(&a, &b[..b.len() - 1]);
            }),
        ),
        (
            "B long".into(),
            rejects(|| {
                let mut x = b.clone();
                x.push(0);
                d.pack_in(&a, &x);
            }),
        ),
        (
            "A empty".into(),
            rejects(|| {
                d.pack_in(&[], &b);
            }),
        ),
    ]);
}
/// Accepted extremes build with the documented argument sizes: one row, kc = 1, kc = 40, exactly 256 waves.
#[test]
fn g80_accepts_boundary_shapes() {
    for (m, k) in [
        (1, 64),
        (1, 2560),
        (256, 2560),
        (257, 64),
        (256 * 256, 64),
        (256 * 256, 2560),
    ] {
        for ctl in BOTH {
            check_layout(&design(m, k, ctl, 12), m, k, &format!("{ctl:?} boundary"))
        }
    }
}

// ---------------------------------------------------------------- grouped launch (`append_run_body` with arena offsets)
struct Expert {
    m: usize,
    k: usize,
    shift: u8,
    a: Vec<i8>,
    b: Vec<i8>,
    d: ArrayDesign,
    packed: [Vec<u8>; 2],
    eager_c: Vec<u8>,
    eager: Config,
}
/// Experts with distinct M that all pad to the same wave count as expert 0: operands and the eager fresh-`Config` full run
/// (itself CPU-exact).
fn experts(k: usize, ms: &[usize], ctl: Control, shift: u8, seed0: u32) -> Vec<Expert> {
    let waves = ms[0].div_ceil(WAVE_M);
    ms.iter()
        .enumerate()
        .map(|(e, &m)| {
            let d = design(m, k, ctl, shift);
            check_layout(&d, m, k, "expert");
            assert_eq!(d.waves(), waves, "expert M={m} must pad to {waves} wave(s)");
            let (a, b) = random(m, k, seed0.wrapping_add(e as u32 * 0x9e37));
            let mut eager = fresh(&d);
            let run = submit_checked(
                &mut eager,
                &d,
                &d.insts,
                &case(format!("G80 eager expert {e} {ctl:?}"), m, k, shift),
                &a,
                &b,
                true,
            );
            let packed = d.pack_in(&a, &b);
            Expert {
                m,
                k,
                shift,
                a,
                b,
                d,
                packed,
                eager_c: run.c,
                eager,
            }
        })
        .collect()
}
/// Arena layout with `GAP` guard bytes around every region; A forward, B and C reversed (non-monotonic).
fn arena_off(ex: &[Expert], e: usize, arg: usize) -> u64 {
    let stride = ex[0].d.args[arg].bytes + GAP;
    let pos = if arg == 0 { e } else { ex.len() - 1 - e };
    (GAP + pos * stride) as u64
}
fn arena_len(ex: &[Expert], arg: usize) -> usize {
    GAP + ex.len() * (ex[0].d.args[arg].bytes + GAP)
}
fn group_txn(ex: &[Expert], lean_tail: bool) -> Vec<u8> {
    let mut t = Txn::aie2p_8col();
    for (e, x) in ex.iter().enumerate() {
        let base = [
            arena_off(ex, e, 0),
            arena_off(ex, e, 1),
            arena_off(ex, e, 2),
        ];
        if lean_tail && e > 0 {
            x.d.append_lean_run_body(&mut t, base, &[])
                .unwrap_or_else(|err| panic!("lean body of expert {e}: {err}"))
        } else {
            x.d.append_run_body(&mut t, base)
        }
    }
    t.to_bytes()
}
fn group_args(ex: &[Expert]) -> Vec<Vec<u8>> {
    let mut args = vec![
        vec![GUARD; arena_len(ex, 0)],
        vec![GUARD; arena_len(ex, 1)],
        vec![POISON; arena_len(ex, 2)],
    ];
    for (e, x) in ex.iter().enumerate() {
        for arg in 0..2 {
            let at = arena_off(ex, e, arg) as usize;
            args[arg][at..at + x.packed[arg].len()].copy_from_slice(&x.packed[arg]);
        }
    }
    args
}
/// A fresh `Config` after ONE full body of the last expert at its arena offsets (the state a group must end in: the shim DDR
/// patch registers of the final body are part of the retained state, so a standalone zero-base run is not comparable).
fn last_body_only(ex: &[Expert]) -> Config {
    let e = ex.len() - 1;
    let mut t = Txn::aie2p_8col();
    ex[e].d.append_run_body(
        &mut t,
        [
            arena_off(ex, e, 0),
            arena_off(ex, e, 1),
            arena_off(ex, e, 2),
        ],
    );
    let mut sim = fresh(&ex[0].d);
    let mut args = group_args(ex);
    sim.submit(&t.to_bytes(), &mut args).unwrap();
    sim
}
/// Submits the group TXN on `sim`: every expert's C region is byte-equal to its eager fresh-`Config` bytes and equals the
/// independent reference, operands are untouched, every byte outside the C regions is still poison, and the shim traffic is
/// the sum of the experts'. Returns the final args.
fn run_group(sim: &mut Config, ex: &[Expert], txn: &[u8], what: &str) -> Vec<Vec<u8>> {
    let mut args = group_args(ex);
    let before = args.clone();
    let (t0, ticks0) = (traffic(sim), sim.ticks);
    sim.submit(txn, &mut args)
        .unwrap_or_else(|e| panic!("{what}: group submit failed: {e}"));
    let kc = ex[0].k / 64;
    let (read, written) = assert_traffic(&t0, &traffic(sim), ex.len(), ex.len(), kc, what);
    let mut owned = vec![false; args[2].len()];
    for (e, x) in ex.iter().enumerate() {
        let at = arena_off(ex, e, 2) as usize;
        let r = at..at + x.d.args[2].bytes;
        if let Some(i) = first_diff(&args[2][r.clone()], &x.eager_c) {
            panic!(
                "{what}: expert {e} (M={}) C bytes != eager at byte {i}",
                x.m
            )
        }
        assert_eq!(
            x.d.unpack_out(&args[2][r.clone()]),
            want(&x.a, &x.b, x.m, x.k, x.shift),
            "{what}: expert {e} (M={}) != CPU reference",
            x.m
        );
        for o in &mut owned[r] {
            *o = true
        }
    }
    assert!(
        args[0] == before[0] && args[1] == before[1],
        "{what}: operand arenas (incl. guards) changed"
    );
    for (i, &o) in owned.iter().enumerate() {
        if !o {
            assert_eq!(
                args[2][i], POISON,
                "{what}: stray write outside every C region at byte {i}"
            )
        }
    }
    assert_eq!(
        (read, written),
        (
            ex.iter().map(|x| x.d.host_bytes_read()).sum::<u64>(),
            ex.iter().map(|x| x.d.host_bytes_written()).sum::<u64>()
        ),
        "{what}: summed host bytes"
    );
    eprintln!("PASS {what}: E={} M={:?}: every expert byte-equal to its eager fresh-Config C and CPU-exact; {} group ticks; DRAM read={read} write={written}",
        ex.len(),ex.iter().map(|x|x.m).collect::<Vec<_>>(),sim.ticks-ticks0);
    args
}
/// E3 mixed M = 64 / 160 / 200 (one wave each, shared PDI core wave counter) in ONE TXN of full bodies.
fn group_exact(k: usize, ctl: Control) {
    let ms = [64, 160, 200];
    let ex = experts(k, &ms, ctl, 12, 0xc0ffee01);
    for x in &ex[1..] {
        assert_eq!(x.d.pdi, ex[0].d.pdi, "one-wave experts must share one PDI")
    }
    let txn = group_txn(&ex, false);
    let mut sim = fresh(&ex[0].d);
    let what = format!("G80 group {ctl:?} K={k}");
    run_group(&mut sim, &ex, &txn, &what);
    let eager_ticks: usize = ex.iter().map(|x| x.eager.ticks).sum();
    eprintln!("MEASURED {what}: group {} ticks vs eager sum {eager_ticks} ticks; TXN {} B vs standalone sum {} B",sim.ticks,txn.len(),ex.iter().map(|x|x.d.insts.len()).sum::<usize>());
    // The group leaves the retained state of one full run.
    assert!(sim.state_snapshot()==last_body_only(&ex).state_snapshot(),"{what}: retained state after the group != one fresh full run of the last body at its arena offsets");
    // The same Config again with new operands (finite queues and iteration counters reset), then a standalone full TXN.
    let ex2 = experts(k, &ms, ctl, 12, 0xbadc0de1);
    assert_eq!(
        group_txn(&ex2, false),
        txn,
        "{what}: group TXN depends on operands"
    );
    run_group(&mut sim, &ex2, &txn, &format!("{what} resubmit"));
    let y = &ex2[0];
    submit_checked(
        &mut sim,
        &y.d,
        &y.d.insts,
        &case(format!("{what} standalone after group"), y.m, y.k, 12),
        &y.a,
        &y.b,
        true,
    );
    assert!(
        sim.state_snapshot() == ex[0].eager.state_snapshot(),
        "{what}: retained state after the standalone run != after one full run"
    );
}
#[test]
fn g80_group_e3_mixed_m_exact_fast_k192() {
    group_exact(192, Control::Fast)
}
#[test]
fn g80_group_e3_mixed_m_exact_slow_k192() {
    group_exact(192, Control::Slow)
}
#[test]
fn g80_group_e3_mixed_m_exact_fast_k64() {
    group_exact(64, Control::Fast)
}
#[test]
fn g80_group_e3_mixed_m_exact_fast_k2560() {
    group_exact(2560, Control::Fast)
}

// ---------------------------------------------------------------- lean submits
fn zero_args(d: &ArrayDesign) -> Vec<Vec<u8>> {
    vec![
        vec![0; d.args[0].bytes],
        vec![0; d.args[1].bytes],
        vec![POISON; d.args[2].bytes],
    ]
}
/// A fresh `Config` after one completed full run with zero operands: the state every lean submit assumes.
fn primed(d: &ArrayDesign) -> Config {
    let mut sim = fresh(d);
    let mut args = zero_args(d);
    sim.submit(&d.insts, &mut args).unwrap();
    assert!(
        args[2].iter().all(|&x| x == 0),
        "zero operands must give zero C"
    );
    sim
}
fn lean_of(d: &ArrayDesign, what: &str) -> Vec<u8> {
    d.lean_insts()
        .unwrap_or_else(|e| panic!("{what}: lean_insts: {e}"))
}
/// One lean submit of `insts` on `sim` for seed operands: exact, byte-equal to a fresh full run (whose own retained state
/// also equals `initial`), and the retained state afterwards equals `initial`.
macro_rules! lean_step {
    ($sim:expr,$d:expr,$insts:expr,$initial:expr,$m:expr,$k:expr,$seed:expr,$what:expr) => {{
        let (a, b) = random($m, $k, $seed);
        let c = case($what.to_string(), $m, $k, 12);
        let mut full = fresh($d);
        let reference = submit_checked(&mut full, $d, &$d.insts, &c, &a, &b, true);
        assert!(
            full.state_snapshot() == $initial,
            "{}: fresh full run state != initial full state",
            $what
        );
        let run = submit_checked($sim, $d, $insts, &c, &a, &b, true);
        if let Some(i) = first_diff(&run.c, &reference.c) {
            panic!("{}: C bytes != fresh full TXN bytes at byte {i}", $what)
        }
        assert!(
            $sim.state_snapshot() == $initial,
            "{}: retained state != fresh full state",
            $what
        );
        eprintln!(
            "MEASURED {}: lean {} ticks vs full {} ticks",
            $what, run.ticks, reference.ticks
        );
    }};
}
/// Full (zero operands) -> three lean submits with fresh operands on the SAME Config.
fn lean_repeat(m: usize, k: usize, ctl: Control) {
    let d = design(m, k, ctl, 12);
    let what = format!("G80 {ctl:?} {m}x{N}x{k}");
    let lean = lean_of(&d, &what);
    let mut sim = primed(&d);
    let initial = sim.state_snapshot();
    for (i, seed) in SEEDS.into_iter().enumerate() {
        lean_step!(
            &mut sim,
            &d,
            &lean,
            initial,
            m,
            k,
            seed,
            format!("{what} lean #{i}")
        )
    }
    eprintln!("PASS lean x3 {what}: exact + byte-equal to fresh full + state == fresh full; full {} B lean {} B",d.insts.len(),lean.len());
}
/// full -> lean -> full -> lean on one Config.
fn lean_interleave(m: usize, k: usize, ctl: Control) {
    let d = design(m, k, ctl, 12);
    let what = format!("G80 {ctl:?} {m}x{N}x{k}");
    let lean = lean_of(&d, &what);
    let mut sim = primed(&d);
    let initial = sim.state_snapshot();
    for (i, (seed, is_lean)) in [
        (SEEDS[0], false),
        (SEEDS[1], true),
        (SEEDS[2], false),
        (0xdeadbeef, true),
    ]
    .into_iter()
    .enumerate()
    {
        let label = format!(
            "{what} interleave #{i} {}",
            if is_lean { "lean" } else { "full" }
        );
        lean_step!(
            &mut sim,
            &d,
            if is_lean { &lean[..] } else { &d.insts[..] },
            initial,
            m,
            k,
            seed,
            label
        );
    }
    eprintln!("PASS full->lean->full->lean {what}");
}
#[test]
fn g80_lean_three_submits_exact_and_state_restored() {
    for (m, k) in REUSE_SHAPES {
        for ctl in BOTH {
            lean_repeat(m, k, ctl)
        }
    }
}
#[test]
fn g80_lean_full_lean_interleave_exact_and_state_restored() {
    for (m, k) in REUSE_SHAPES {
        for ctl in BOTH {
            lean_interleave(m, k, ctl)
        }
    }
}
#[test]
fn g80_lean_large_k_and_waves() {
    for (m, k) in [(160, 2560), (512, 640)] {
        lean_repeat(m, k, Control::Fast);
        lean_interleave(m, k, Control::Fast)
    }
}
const FORCE_B: (u32, DmaDirection, u32) = (0, DmaDirection::S2mm, 4);
const FORCE_C: (u32, DmaDirection, u32) = (0, DmaDirection::Mm2s, 0);
/// The persistent ring's POLL / DONE clobber column 0 memtile S2MM4 (B fill) and MM2S0 (C drain): the forced reset +
/// requeue keeps every finite queue and still gives exact, byte-equal results and the fresh full retained state.
fn forced_requeue(m: usize, k: usize, ctl: Control) {
    let d = design(m, k, ctl, 12);
    let what = format!("G80 forced {ctl:?} {m}x{N}x{k}");
    let plain = lean_of(&d, &what);
    let sets: [(&str, Vec<(u32, DmaDirection, u32)>); 3] = [
        ("B S2MM4", vec![FORCE_B]),
        ("C MM2S0", vec![FORCE_C]),
        ("B S2MM4 + C MM2S0", vec![FORCE_B, FORCE_C]),
    ];
    for (label, extra) in sets {
        let forced = d
            .lean_insts_requeue(&extra)
            .unwrap_or_else(|e| panic!("{what} {label}: lean_insts_requeue: {e}"));
        let mut sim = primed(&d);
        let initial = sim.state_snapshot();
        for (i, seed) in SEEDS.into_iter().enumerate() {
            lean_step!(
                &mut sim,
                &d,
                &forced,
                initial,
                m,
                k,
                seed,
                format!("{what} {label} #{i}")
            )
        }
        lean_step!(
            &mut sim,
            &d,
            &plain,
            initial,
            m,
            k,
            0xdeadbeef,
            format!("{what} plain lean after {label}")
        );
        eprintln!(
            "PASS forced requeue {label} {what}: lean {} B forced {} B",
            plain.len(),
            forced.len()
        );
    }
}
#[test]
fn g80_lean_forced_requeue_one_and_two_waves() {
    for (m, k) in [(256, 64), (256, 128), (512, 192), (200, 128)] {
        for ctl in BOTH {
            forced_requeue(m, k, ctl)
        }
    }
}
/// Unsupported forced channels are rejected before any op is appended.
#[test]
fn g80_lean_forced_requeue_rejects_unsupported_channels() {
    let d = design(256, 128, Control::Fast, 12);
    for extra in [
        (8u32, DmaDirection::S2mm, 4u32),
        (0, DmaDirection::S2mm, 5),
        (0, DmaDirection::Mm2s, 4),
    ] {
        assert!(
            d.lean_insts_requeue(&[extra]).is_err(),
            "forced {extra:?} must be rejected"
        );
        let mut t = Txn::aie2p_8col();
        let empty = t.to_bytes();
        assert!(d.append_lean_run_body(&mut t, [0; 3], &[extra]).is_err());
        assert_eq!(t.to_bytes(), empty, "txn must be unchanged on Err");
    }
}
/// E3 mixed M group: expert 0 full body then lean bodies for experts 1, 2 equals the all-full group in every output byte and
/// in the retained state, and is shorter.
fn group_lean(k: usize, ctl: Control) {
    let ex = experts(k, &[64, 160, 200], ctl, 12, 0xc0ffee02);
    let (full, lean) = (group_txn(&ex, false), group_txn(&ex, true));
    let what = format!("G80 lean group {ctl:?} K={k}");
    let (mut sim_l, mut sim_f) = (fresh(&ex[0].d), fresh(&ex[0].d));
    let args_l = run_group(&mut sim_l, &ex, &lean, &what);
    let args_f = run_group(&mut sim_f, &ex, &full, &format!("{what} (all-full oracle)"));
    for i in 0..3 {
        if let Some(at) = first_diff(&args_l[i], &args_f[i]) {
            panic!("{what}: arg{i} differs from the all-full group at byte {at}")
        }
    }
    assert!(
        sim_l.state_snapshot() == sim_f.state_snapshot(),
        "{what}: retained state after the lean group != the all-full group"
    );
    assert!(
        sim_l.state_snapshot() == last_body_only(&ex).state_snapshot(),
        "{what}: retained state != one fresh full run of the last body at its arena offsets"
    );
    // Reuse: a second lean group on the same Config with new operands (every body, including expert 0, starts after a completed run).
    let ex2 = experts(k, &[64, 160, 200], ctl, 12, 0xbadc0de2);
    run_group(&mut sim_l, &ex2, &lean, &format!("{what} resubmit"));
    assert!(
        sim_l.state_snapshot() == sim_f.state_snapshot(),
        "{what}: state after the second lean group"
    );
    eprintln!("MEASURED {what}: lean group TXN {} B vs all-full {} B; cumulative ticks after two lean groups {} vs one all-full group {} (one group = {})",lean.len(),full.len(),sim_l.ticks,sim_f.ticks,sim_l.ticks/2);
}
#[test]
fn g80_group_lean_tail_equals_all_full_fast_k192() {
    group_lean(192, Control::Fast)
}
#[test]
fn g80_group_lean_tail_equals_all_full_slow_k128() {
    group_lean(128, Control::Slow)
}
#[test]
fn g80_group_lean_tail_equals_all_full_fast_k2560() {
    group_lean(2560, Control::Fast)
}

// ---------------------------------------------------------------- `experts::grouped` with a G80 design
// The production grouped API (expert 0 FULL body, experts 1.. LEAN bodies, optional persistent ring once around the group)
// built from ONE one-wave G80 design (`wave_m = 256`), against `grouped_all_full` (every body FULL) and the eager designs.
const DONE_MAGIC: u32 = 0x454e4f44;
const LINE: usize = 64;
const FIRST_AT: usize = 37;
const OPERAND_TO_SEQ: usize = 3;
fn wr32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes())
}
fn wr64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes())
}
fn done_bytes(slot: usize, seq: u32) -> [u8; LINE] {
    let mut l = [0; LINE];
    wr32(&mut l, 0, seq);
    wr32(&mut l, 4, slot as u32);
    wr32(&mut l, 8, DONE_MAGIC);
    l
}
fn plans_of(ex: &[Expert]) -> Vec<ExpertPlan> {
    (0..ex.len())
        .map(|e| ExpertPlan {
            m: ex[e].m,
            a_off: arena_off(ex, e, 0),
            b_off: arena_off(ex, e, 1),
            c_off: arena_off(ex, e, 2),
        })
        .collect()
}
/// `(grouped, grouped_all_full)` of the same plans on the G80 design of `round_up(max m, 256)` rows (one or more waves),
/// passed with `wave_m` = all its rows (the npu-experts construction).
fn g80_groups(
    ex: &[Expert],
    ctl: Control,
    k: usize,
    ring: Option<(RingLayout, usize, u32)>,
) -> (GroupedDesign, GroupedDesign) {
    let plans = plans_of(ex);
    let rows = ex.iter().map(|x| x.m).max().unwrap().div_ceil(WAVE_M) * WAVE_M;
    let d = design(rows, k, ctl, 12);
    let wave_m = d.waves() * d.wave_m();
    assert_eq!(wave_m, rows);
    (
        grouped(d, wave_m, N, k, &plans, ring),
        grouped_all_full(design(rows, k, ctl, 12), wave_m, N, k, &plans, ring),
    )
}
/// Guard-filled A / B arenas, poison C arena (and an initialised ring arg).
fn blank_g(g: &GroupedDesign, layout: Option<RingLayout>) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = g.args.iter().map(|s| vec![GUARD; s.bytes]).collect();
    v[2].fill(POISON);
    if let Some(l) = layout {
        assert_eq!(v.len(), 4);
        assert_eq!(v[3].len(), l.bytes());
        l.initialize(&mut v[3])
    } else {
        assert_eq!(v.len(), 3)
    }
    v
}
fn put_g(args: &mut [Vec<u8>], plans: &[ExpertPlan], ex: &[Expert]) {
    for (p, x) in plans.iter().zip(ex) {
        let (a, b) = (p.a_off as usize, p.b_off as usize);
        args[0][a..a + x.packed[0].len()].copy_from_slice(&x.packed[0]);
        args[1][b..b + x.packed[1].len()].copy_from_slice(&x.packed[1]);
    }
}
fn c_range_g(g: &GroupedDesign, p: &ExpertPlan, e: usize) -> Range<usize> {
    p.c_off as usize..p.c_off as usize + g.c_bytes(e)
}
/// Every expert's C is byte-equal to its eager fresh-`Config` bytes and equals the independent reference, the operand arenas
/// (incl. guards) are untouched and no byte outside the C regions was written.
fn check_group_g(
    g: &GroupedDesign,
    plans: &[ExpertPlan],
    ex: &[Expert],
    args: &[Vec<u8>],
    layout: Option<RingLayout>,
    what: &str,
) {
    let mut want_args = blank_g(g, layout);
    put_g(&mut want_args, plans, ex);
    if let Some(i) = first_diff(&args[0], &want_args[0]) {
        panic!("{what}: A arena modified at byte {i}")
    }
    if let Some(i) = first_diff(&args[1], &want_args[1]) {
        panic!("{what}: B arena modified at byte {i}")
    }
    let mut covered = vec![false; args[2].len()];
    for (e, (p, x)) in plans.iter().zip(ex).enumerate() {
        let c = c_range_g(g, p, e);
        if let Some(i) = first_diff(&args[2][c.clone()], &x.eager_c) {
            panic!(
                "{what}: expert {e} (m={}) C != eager G80 bytes of a fresh Config (byte {i})",
                p.m
            )
        }
        assert_eq!(
            g.unpack_out(e, &args[2][c.clone()]),
            want(&x.a, &x.b, x.m, x.k, x.shift),
            "{what}: expert {e} C != CPU reference"
        );
        covered[c].fill(true);
    }
    assert!(
        args[2]
            .iter()
            .zip(&covered)
            .all(|(&b, &c)| c || b == POISON),
        "{what}: C arena written outside the expert regions"
    );
}
/// Packing, reference, unpacking, sizes and image of the group equal the per-expert eager design.
fn check_api_g(g: &GroupedDesign, ex: &[Expert]) {
    assert_eq!(g.experts(), ex.len());
    for (e, x) in ex.iter().enumerate() {
        assert_eq!(
            g.pack_in(e, &x.a, &x.b),
            x.packed,
            "expert {e} (m={}): grouped packing == eager packing",
            x.m
        );
        assert_eq!(
            g.reference(e, &x.a, &x.b),
            want(&x.a, &x.b, x.m, x.k, x.shift),
            "expert {e}: grouped reference == independent reference"
        );
        assert_eq!(
            g.unpack_out(e, &x.eager_c),
            want(&x.a, &x.b, x.m, x.k, x.shift),
            "expert {e}: grouped unpack of the eager C"
        );
        assert_eq!(
            (g.a_bytes(e), g.b_bytes(e), g.c_bytes(e)),
            (x.d.args[0].bytes, x.d.args[1].bytes, x.d.args[2].bytes),
            "expert {e}: sizes"
        );
        assert!(g.pdi == x.d.pdi, "expert {e}: one shared PDI");
    }
}
/// Lean group vs all-full group on the same `Config` type, two rounds of fresh operands (the second submit starts from the
/// end state of the first): exact C, equal arenas, per-channel shim bytes of `E` bodies, equal retained state.
fn grouped_api_case(ms: &[usize], k: usize, ctl: Control) {
    let ex = experts(k, ms, ctl, 12, 0xc0ffee11);
    let plans = plans_of(&ex);
    let (g, f) = g80_groups(&ex, ctl, k, None);
    assert!(g.patch_sites.is_empty() && f.patch_sites.is_empty());
    assert!(g.pdi == f.pdi, "lean and all-full group load the same PDI");
    check_api_g(&g, &ex);
    let (mut sg, mut sf) = (
        Config::from_pdi(&g.pdi).unwrap(),
        Config::from_pdi(&f.pdi).unwrap(),
    );
    let ex2 = experts(k, ms, ctl, 12, 0xbadc0de3);
    for (round, x) in [&ex, &ex2].into_iter().enumerate() {
        let (mut ag, mut af) = (blank_g(&g, None), blank_g(&f, None));
        put_g(&mut ag, &plans, x);
        put_g(&mut af, &plans, x);
        let (tg, tf, (g0, f0)) = (traffic(&sg), traffic(&sf), (sg.ticks, sf.ticks));
        sg.submit(&g.insts, &mut ag)
            .unwrap_or_else(|e| panic!("lean group round {round}: {e}"));
        sf.submit(&f.insts, &mut af)
            .unwrap_or_else(|e| panic!("all-full group round {round}: {e}"));
        check_group_g(
            &g,
            &plans,
            x,
            &ag,
            None,
            &format!("lean group round {round}"),
        );
        check_group_g(
            &f,
            &plans,
            x,
            &af,
            None,
            &format!("all-full group round {round}"),
        );
        for i in 0..3 {
            if let Some(at) = first_diff(&ag[i], &af[i]) {
                panic!("round {round}: lean group arg{i} != all-full group arg{i} at byte {at}")
            }
        }
        let (kc, w) = (k / 64, ex[0].d.waves());
        let (rg, wg) = assert_traffic(
            &tg,
            &traffic(&sg),
            ms.len(),
            w * ms.len(),
            kc,
            &format!("lean group round {round}"),
        );
        let (rf, wf) = assert_traffic(
            &tf,
            &traffic(&sf),
            ms.len(),
            w * ms.len(),
            kc,
            &format!("all-full group round {round}"),
        );
        assert_eq!((rg, wg), (rf, wf));
        assert!(
            sg.state_snapshot() == sf.state_snapshot(),
            "round {round}: retained state after the lean group != the all-full group"
        );
        eprintln!("MEASURED G80 grouped API {ctl:?} E={} M={ms:?} K={k} round {round}: PDI {} B; TXN lean {} B vs all-full {} B; {} vs {} functional ticks; DRAM read={rg} write={wg}; \
            per column MM2S0 B={} MM2S1 A={}(cols<4) S2MM0/1={} B",ms.len(),g.pdi.len(),g.insts.len(),f.insts.len(),sg.ticks-g0,sf.ticks-f0,ms.len()*kc*B_CHUNK,w*ms.len()*kc*A_CHUNK,w*ms.len()*2*C_TILE);
    }
}
#[test]
fn g80_grouped_api_e3_mixed_m_fast_k192() {
    grouped_api_case(&[64, 160, 200], 192, Control::Fast)
}
#[test]
fn g80_grouped_api_e3_mixed_m_slow_k128() {
    grouped_api_case(&[64, 160, 200], 128, Control::Slow)
}
#[test]
fn g80_grouped_api_e3_mixed_m_fast_k64_kc1() {
    grouped_api_case(&[64, 160, 200], 64, Control::Fast)
}
#[test]
fn g80_grouped_api_e3_mixed_m_fast_k2560() {
    grouped_api_case(&[64, 160, 200], 2560, Control::Fast)
}
#[test]
fn g80_grouped_api_e4_boundary_m_fast_k128() {
    grouped_api_case(&[1, 256, 200, 64], 128, Control::Fast)
}
// Two-wave groups (npu-experts `--design G80 --m 257..=512`: `design_g80(512, ..)`, `wave_m = 512`): lean tails exact and
// state-equal to the all-full group at kc 1 / 2 / 3 / 40, Fast and Slow, mixed M, with and without the ring.
#[test]
fn g80_grouped_api_two_waves_e3_fast_k192() {
    grouped_api_case(&[300, 512, 257], 192, Control::Fast)
}
#[test]
fn g80_grouped_api_two_waves_e3_slow_k128() {
    grouped_api_case(&[512, 300, 400], 128, Control::Slow)
}
#[test]
fn g80_grouped_api_two_waves_e2_fast_k2560() {
    grouped_api_case(&[512, 384], 2560, Control::Fast)
}
#[test]
fn g80_grouped_ring_two_waves_e3_late_producer() {
    grouped_ring_case(
        &[512, 300, 257],
        64,
        Control::Fast,
        RingLayout { nslots: 2 },
        1,
        5,
        2_000,
    )
}
/// The persistent ring once around the whole group; the producer publishes operands, then `seq`, late; C must stay poison
/// until then; the ring's DONE line is exact and the lean ring group equals the all-full ring group (bytes and state).
fn grouped_ring_case(
    ms: &[usize],
    k: usize,
    ctl: Control,
    layout: RingLayout,
    slot: usize,
    seq: u32,
    delay: usize,
) {
    let ex = experts(k, ms, ctl, 12, 0xc0ffee21);
    let plans = plans_of(&ex);
    let ring = Some((layout, slot, seq));
    let (g, f) = g80_groups(&ex, ctl, k, ring);
    assert_eq!(g.args.len(), 4);
    assert_eq!(g.args[3].bytes, layout.bytes());
    check_api_g(&g, &ex);
    let run = |d: &GroupedDesign| -> (Vec<Vec<u8>>, Config) {
        let mut args = blank_g(d, Some(layout));
        let mut sim = Config::from_pdi(&d.pdi).unwrap();
        let mut shadow = args[3].clone();
        let (mut stage, mut at, mut calls) = (0u8, FIRST_AT, 0usize);
        sim.submit_with(&d.insts,&mut args,&mut |t,a|{
            calls+=1;
            if stage<2 && calls%16==1 {for (e,p) in plans.iter().enumerate() {
                assert!(a[2][c_range_g(d,p,e)].iter().all(|&b|b==POISON),"expert {e} C written before the group was published (tick {t})");
            }}
            if stage==2 || t<at {return}
            if stage==0 {put_g(a,&plans,&ex);stage=1;at=t+OPERAND_TO_SEQ+delay;return}
            for (e,p) in plans.iter().enumerate() {assert!(a[2][c_range_g(d,p,e)].iter().all(|&b|b==POISON),"expert {e}: C computed from operands whose seq was not yet published (tick {t})")}
            let line=layout.slot_line(slot);let p=&plans[0];
            let mut fields=[0u8;LINE-4];wr32(&mut fields,0,9);wr32(&mut fields,4,p.m as u32);wr32(&mut fields,8,0);
            wr64(&mut fields,12,p.a_off);wr64(&mut fields,20,p.b_off);wr64(&mut fields,28,p.c_off);
            for arg in [&mut a[3],&mut shadow] {arg[line+4..line+LINE].copy_from_slice(&fields);arg[line..line+4].copy_from_slice(&seq.to_le_bytes())}
            stage=2;
        }).unwrap();
        assert_eq!(stage, 2, "the producer never published");
        assert!(
            sim.ticks > delay,
            "the group finished ({} ticks) before the late publication ({delay})",
            sim.ticks
        );
        if let Some(i) = first_diff(
            &args[3][..layout.done_line(0)],
            &shadow[..layout.done_line(0)],
        ) {
            panic!("ring header / slot lines != initial + producer writes at byte {i}")
        }
        for s in 0..layout.nslots {
            let at = layout.done_line(s);
            let want_line = if s == slot {
                done_bytes(slot, seq)
            } else {
                [0; LINE]
            };
            assert_eq!(
                &args[3][at..at + LINE],
                &want_line[..],
                "done line of slot {s}"
            );
        }
        (args, sim)
    };
    let ((ag, sg), (af, sf)) = (run(&g), run(&f));
    check_group_g(&g, &plans, &ex, &ag, Some(layout), "lean ring group");
    check_group_g(&f, &plans, &ex, &af, Some(layout), "all-full ring group");
    for i in 0..4 {
        if let Some(at) = first_diff(&ag[i], &af[i]) {
            panic!("lean ring group arg{i} != all-full ring group arg{i} at byte {at}")
        }
    }
    assert!(
        sg.state_snapshot() == sf.state_snapshot(),
        "retained state after the lean ring group != the all-full ring group"
    );
    eprintln!("MEASURED G80 grouped ring {ctl:?} E={} M={ms:?} K={k} slot={slot} seq={seq} delay={delay}: PDI {} B; TXN lean {} B vs all-full {} B; {} vs {} functional ticks",
        ms.len(),g.pdi.len(),g.insts.len(),f.insts.len(),sg.ticks,sf.ticks);
}
#[test]
fn g80_grouped_ring_e3_late_producer() {
    grouped_ring_case(
        &[64, 160, 200],
        128,
        Control::Fast,
        RingLayout { nslots: 2 },
        0,
        1,
        20_000,
    )
}
#[test]
fn g80_grouped_ring_e4_late_producer_slot_seq() {
    grouped_ring_case(
        &[160, 200, 64, 64],
        192,
        Control::Slow,
        RingLayout { nslots: 4 },
        3,
        7,
        400,
    )
}
/// Group boundaries: M = 1 and one full wave are exact; empty, M = 0, M above the wave and unaligned offsets are rejected.
#[test]
fn g80_grouped_boundaries() {
    grouped_api_case(&[1, 256], 64, Control::Fast);
    let (k, wave_m) = (64usize, WAVE_M);
    let one = |plans: &[ExpertPlan]| {
        grouped(
            design(WAVE_M, k, Control::Fast, 12),
            wave_m,
            N,
            k,
            plans,
            None,
        );
    };
    let plan = |m, a, b, c| ExpertPlan {
        m,
        a_off: a,
        b_off: b,
        c_off: c,
    };
    assert_all_rejected(vec![
        ("empty group".into(), rejects(|| one(&[]))),
        ("m=0".into(), rejects(|| one(&[plan(0, 0, 0, 0)]))),
        ("m=257".into(), rejects(|| one(&[plan(257, 0, 0, 0)]))),
        ("unaligned A".into(), rejects(|| one(&[plan(64, 2, 0, 0)]))),
        (
            "unaligned C".into(),
            rejects(|| one(&[plan(64, 0, 0, 0), plan(64, 0, 0, 6)])),
        ),
        (
            "multi-wave design".into(),
            rejects(|| {
                grouped(
                    design(512, k, Control::Fast, 12),
                    wave_m,
                    N,
                    k,
                    &[plan(64, 0, 0, 0)],
                    None,
                );
            }),
        ),
    ]);
}

// ---------------------------------------------------------------- adversarial B fill order (parity 1 ready before parity 0)
/// TEST ONLY. The memtile B fill is S2MM4: BD4 (parity 0 region, releases ready lock 4) -> BD5 (parity 1 region, releases ready
/// lock 6), each one 5120 B chunk half, repeated `kc` times. The two guarded readers each acquire their own ready lock and read
/// their own parity region, so the guard order is independent of the fill order. Here the fill order is reversed: BD4 and BD5
/// exchange address and lock words in every column's descriptor rewrite of the full TXN (BD4 keeps `next = 5`, BD5 none, both
/// keep length and iteration), so parity 1 is filled and signalled first, and the host B halves `[even, odd]` of every chunk
/// are exchanged to `[odd, even]` so every byte still lands in its original region. The result must stay CPU-exact.
const B_HALF: usize = 5120;
const BD_FILL: [u32; 2] = [4, 5];
fn mem_bd_addr(col: u32, id: u32) -> u32 {
    peacemaker_npu::regs::tile(col, 1, 0xa0000 + id * 0x20)
}
/// `d.insts` with the B fill descriptors of all 8 columns exchanged (panics if a column's BD4 / BD5 rewrite is not found).
fn reverse_fill_ops(d: &ArrayDesign) -> Vec<TxnOp> {
    let mut ops = parse_txn(&d.insts).unwrap();
    for col in 0..COLS as u32 {
        let at = BD_FILL.map(|id| {
            ops.iter()
                .position(
                    |o| matches!(o,TxnOp::BlockWrite(a,w) if *a==mem_bd_addr(col,id)&&w.len()==8),
                )
                .unwrap_or_else(|| {
                    panic!("col {col}: no 8-word BlockWrite of memtile BD{id} in the full TXN")
                })
        });
        let words = at.map(|i| match &ops[i] {
            TxnOp::BlockWrite(_, w) => w.clone(),
            _ => unreachable!(),
        });
        let (w4, w5) = (&words[0], &words[1]);
        assert!(
            w4[1] >> 19 & 1 == 1 && (w4[1] >> 20) & 0x3f == BD_FILL[1] && w5[1] >> 19 & 1 == 0,
            "col {col}: BD4 must chain to BD5 and BD5 end the chain"
        );
        assert!(
            w4[0] == w5[0] && w4[6] == w5[6] && w4[2..6] == w5[2..6],
            "col {col}: BD4/BD5 differ in more than address and locks"
        );
        assert!(
            w4[1] & 0x7ffff != w5[1] & 0x7ffff && w4[7] != w5[7],
            "col {col}: BD4/BD5 must differ in address and locks"
        );
        let (mut n4, mut n5) = (w4.clone(), w5.clone());
        n4[1] = (w4[1] & !0x7ffff) | (w5[1] & 0x7ffff);
        n5[1] = (w5[1] & !0x7ffff) | (w4[1] & 0x7ffff);
        n4[7] = w5[7];
        n5[7] = w4[7];
        ops[at[0]] = TxnOp::BlockWrite(mem_bd_addr(col, BD_FILL[0]), n4);
        ops[at[1]] = TxnOp::BlockWrite(mem_bd_addr(col, BD_FILL[1]), n5);
    }
    ops
}
/// Host B with the `[even, odd]` halves of every column chunk exchanged to `[odd, even]`.
fn swap_b_halves(b: &mut [u8], kc: usize) {
    for col in 0..COLS {
        for ch in 0..kc {
            let base = (col * kc + ch) * 2 * B_HALF;
            for i in 0..B_HALF {
                b.swap(base + i, base + B_HALF + i)
            }
        }
    }
}
/// Reversed-order TXN: CPU-exact, A / reversed B unchanged, exact per-channel shim bytes, on a REUSED `Config` (two submits).
/// `control`: the same reversed descriptors with the unexchanged host B must NOT give the CPU result (the patch is live).
fn reversed_fill_case(m: usize, k: usize, ctl: Control) {
    let d = design(m, k, ctl, 12);
    let (kc, waves) = (k / 64, m.div_ceil(WAVE_M));
    let ops = reverse_fill_ops(&d);
    let what = format!("G80 reversed B fill {ctl:?} {m}x{N}x{k}");
    let mut sim = fresh(&d);
    for (i, seed) in [SEEDS[0], SEEDS[1]].into_iter().enumerate() {
        let (a, b) = random(m, k, seed);
        let [a_host, mut b_host] = d.pack_in(&a, &b);
        swap_b_halves(&mut b_host, kc);
        let before = [signature(&a_host), signature(&b_host)];
        let mut args = vec![a_host, b_host, vec![POISON; d.args[2].bytes]];
        let (t0, ticks0) = (traffic(&sim), sim.ticks);
        sim.execute(&ops, &mut args)
            .unwrap_or_else(|e| panic!("{what} #{i}: {e}"));
        let w = want(&a, &b, m, k, 12);
        let got = d.unpack_out(&args[2]);
        if let Some(j) = (0..w.len()).find(|&j| got[j] != w[j]) {
            panic!(
                "{what} #{i}: first mismatch at ({},{}) got {} want {}",
                j / N,
                j % N,
                got[j],
                w[j]
            )
        }
        assert_eq!(
            [signature(&args[0]), signature(&args[1])],
            before,
            "{what} #{i}: reversed inputs changed"
        );
        let (read, written) = assert_traffic(&t0, &traffic(&sim), 1, waves, kc, &what);
        eprintln!(
            "MEASURED {what} #{i}: CPU-exact; {} ticks; DRAM read={read} write={written}",
            sim.ticks - ticks0
        );
    }
    let (a, b) = random(m, k, SEEDS[2]);
    let [a_host, b_host] = d.pack_in(&a, &b);
    let mut args = vec![a_host, b_host, vec![POISON; d.args[2].bytes]];
    let mut control = fresh(&d);
    let outcome = control.execute(&ops, &mut args);
    assert!(
        outcome.is_err() || d.unpack_out(&args[2]) != want(&a, &b, m, k, 12),
        "{what}: control (reversed descriptors, unexchanged B) must differ from the CPU result"
    );
}
#[test]
fn g80_reversed_b_fill_order_exact_two_parities_and_chunks() {
    for (m, k) in [
        (256, 128),
        (256, 192),
        (512, 192),
        (768, 64),
        (512, 640),
        (300, 320),
    ] {
        for ctl in BOTH {
            reversed_fill_case(m, k, ctl)
        }
    }
}
#[test]
fn g80_reversed_b_fill_order_exact_kc40_two_m_waves() {
    reversed_fill_case(512, 2560, Control::Fast)
}

// ---------------------------------------------------------------- negative: the finite requeues / forced resets are load-bearing
// The lean body must requeue every finite memtile task (B fill BD4, guarded readers BD30 / BD42, whole-segment replays BD31 /
// BD43 for MW > 1) after a completed full run, and a ring POLL's self-looping S2MM4 task must be reset by the group's first
// body. Each omission below is a TEST-ONLY TXN derived from valid ops; the run must hang (bounded `tick_limit`) or give a wrong
// C. The unmodified ops are the control in every case.
/// Memtile DMA start-queue register offset: S2MM `ch` at `0xa0604 + 8*ch`, MM2S `ch` after the six S2MM channels.
fn mem_queue_off(mm2s: bool, ch: u32) -> u32 {
    0xa0604 + 8 * (if mm2s { 6 + ch } else { ch })
}
/// Removes only the FIRST write of the task queue register `off` of column `col`'s memtile that queues BD `bd`.
fn drop_one_queue_write(col: u32, off: u32, bd: u32) -> impl Fn(&TxnOp) -> bool {
    let dropped = Cell::new(false);
    move |op| match *op {
        TxnOp::Write(a, v)
            if !dropped.get()
                && a >> 25 == col
                && (a >> 20) & 31 == 1
                && a & 0xfffff == off
                && v & 0xffff == bd =>
        {
            dropped.set(true);
            true
        }
        _ => false,
    }
}
#[derive(Debug)]
#[allow(dead_code)]
enum Detected {
    Failed(String),
    WrongC(usize),
}
/// The valid lean ops of `d` minus the ops selected by `drop`; shim programming, core control and every SYNC are retained.
fn forced_ops(d: &ArrayDesign, drop: &dyn Fn(&TxnOp) -> bool) -> (Vec<TxnOp>, usize) {
    let lean = parse_txn(&d.lean_insts().unwrap()).unwrap();
    let kept: Vec<TxnOp> = lean.iter().filter(|o| !drop(*o)).cloned().collect();
    let removed = lean.len() - kept.len();
    assert_eq!(
        kept.iter()
            .filter(|o| matches!(o, TxnOp::Sync { .. }))
            .count(),
        lean.iter()
            .filter(|o| matches!(o, TxnOp::Sync { .. }))
            .count()
    );
    assert!(kept.iter().any(|o| matches!(o, TxnOp::DdrPatch { .. })));
    (kept, removed)
}
/// Runs `ops` up to three times on a Config that completed one full zero-operand run, with the deadlock limit tightened to a
/// multiple of that run so a stuck SYNC reports quickly. `None` = it behaved like a lean TXN.
fn run_forced(d: &ArrayDesign, m: usize, k: usize, ops: &[TxnOp]) -> Option<Detected> {
    let mut sim = primed(d);
    sim.tick_limit = sim.ticks * 2 + 4096;
    for seed in SEEDS {
        let (a, b) = random(m, k, seed);
        let [a_host, b_host] = d.pack_in(&a, &b);
        let mut args = vec![a_host, b_host, vec![POISON; d.args[2].bytes]];
        if let Err(e) = sim.execute(ops, &mut args) {
            return Some(Detected::Failed(e));
        }
        let (got, w) = (d.unpack_out(&args[2]), want(&a, &b, m, k, 12));
        if let Some(i) = (0..got.len()).find(|&i| got[i] != w[i]) {
            return Some(Detected::WrongC(i));
        }
    }
    None
}
fn omitted_queue_detected(
    m: usize,
    k: usize,
    ctl: Control,
    mm2s: bool,
    ch: u32,
    bd: u32,
    what_dropped: &str,
) {
    let d = design(m, k, ctl, 12);
    let what = format!("G80 {ctl:?} {m}x{N}x{k}");
    let (control, removed) = forced_ops(&d, &|_| false);
    assert_eq!(removed, 0);
    assert!(
        run_forced(&d, m, k, &control).is_none(),
        "{what}: the unmodified lean ops must behave like lean"
    );
    let (ops, removed) = forced_ops(&d, &drop_one_queue_write(0, mem_queue_off(mm2s, ch), bd));
    assert_eq!(
        removed, 1,
        "{what}: expected exactly one {what_dropped} queue write to omit"
    );
    let got = run_forced(&d, m, k, &ops);
    assert!(got.is_some(),"{what}: omitting the col 0 {what_dropped} queue write still behaves like lean: the requeue is not load-bearing");
    eprintln!(
        "NEGATIVE {what}: omitting col 0 {what_dropped} queue write -> {:?}",
        got.unwrap()
    );
}
/// One M wave (no whole-segment replay) and two M waves.
const OMIT_SHAPES: [(usize, usize); 2] = [(256, 128), (512, 192)];
#[test]
fn g80_omitting_b_fill_bd4_requeue_is_detected() {
    for (m, k) in OMIT_SHAPES {
        for ctl in BOTH {
            omitted_queue_detected(m, k, ctl, false, 4, 4, "B fill BD4")
        }
    }
}
#[test]
fn g80_omitting_guard_bd30_requeue_is_detected() {
    for (m, k) in OMIT_SHAPES {
        for ctl in BOTH {
            omitted_queue_detected(m, k, ctl, true, 1, 30, "guarded reader BD30")
        }
    }
}
#[test]
fn g80_omitting_guard_bd42_requeue_is_detected() {
    for (m, k) in OMIT_SHAPES {
        for ctl in BOTH {
            omitted_queue_detected(m, k, ctl, true, 5, 42, "guarded reader BD42")
        }
    }
}
#[test]
fn g80_omitting_whole_bd31_requeue_is_detected_mw2() {
    for ctl in BOTH {
        omitted_queue_detected(512, 192, ctl, true, 1, 31, "whole-segment replay BD31")
    }
}
#[test]
fn g80_omitting_whole_bd43_requeue_is_detected_mw2() {
    for ctl in BOTH {
        omitted_queue_detected(512, 192, ctl, true, 5, 43, "whole-segment replay BD43")
    }
}
/// Real grouped ring (E3 mixed M, one wave each): the POLL leaves a self-looping task on column 0 memtile S2MM4. The group's
/// first (full) body must reset that channel; with its S2MM4 CTRL (reset) writes omitted the fill never starts and the run
/// must hang within the bounded `tick_limit` or give a wrong C. The unmodified ring group is the control.
fn ring_poll_reset_omission_detected(
    ms: &[usize],
    k: usize,
    ctl: Control,
    layout: RingLayout,
    slot: usize,
    seq: u32,
) {
    let ex = experts(k, ms, ctl, 12, 0xc0ffee31);
    let plans = plans_of(&ex);
    let (g, _) = g80_groups(&ex, ctl, k, Some((layout, slot, seq)));
    let what = format!("G80 grouped ring {ctl:?} E={} M={ms:?} K={k}", ms.len());
    let run =
        |ops: &[TxnOp], tick_limit: Option<usize>| -> (Result<(), String>, Vec<Vec<u8>>, usize) {
            let mut args = blank_g(&g, Some(layout));
            let mut sim = Config::from_pdi(&g.pdi).unwrap();
            if let Some(l) = tick_limit {
                sim.tick_limit = l
            }
            let (mut stage, mut next) = (0u8, FIRST_AT);
            let res = sim.execute_with(ops, &mut args, &mut |t, a| {
                if stage == 2 || t < next {
                    return;
                }
                if stage == 0 {
                    put_g(a, &plans, &ex);
                    stage = 1;
                    next = t + OPERAND_TO_SEQ;
                    return;
                }
                let line = layout.slot_line(slot);
                let p = &plans[0];
                let mut fields = [0u8; LINE - 4];
                wr32(&mut fields, 0, 9);
                wr32(&mut fields, 4, p.m as u32);
                wr32(&mut fields, 8, 0);
                wr64(&mut fields, 12, p.a_off);
                wr64(&mut fields, 20, p.b_off);
                wr64(&mut fields, 28, p.c_off);
                a[3][line + 4..line + LINE].copy_from_slice(&fields);
                a[3][line..line + 4].copy_from_slice(&seq.to_le_bytes());
                stage = 2;
            });
            (res, args, sim.ticks)
        };
    let ops = parse_txn(&g.insts).unwrap();
    let (res, args, ticks) = run(&ops, None);
    res.unwrap_or_else(|e| panic!("{what}: control ring group failed: {e}"));
    check_group_g(
        &g,
        &plans,
        &ex,
        &args,
        Some(layout),
        &format!("{what} control"),
    );
    let drain = ops
        .iter()
        .position(|o| matches!(o,TxnOp::MaskPoll(_,_,v) if *v==SENTINEL))
        .expect("ring drain poll");
    let ctrl = peacemaker_npu::regs::tile(0, 1, 0xa0600 + 4 * 8);
    let mut removed = 0usize;
    let kept: Vec<TxnOp> = ops
        .iter()
        .enumerate()
        .filter(|(i, o)| {
            let hit = *i > drain && matches!(o,TxnOp::MaskWrite(a,..) if *a==ctrl);
            if hit {
                removed += 1
            }
            !hit
        })
        .map(|(_, o)| o.clone())
        .collect();
    assert!(
        removed > 0,
        "{what}: no col 0 S2MM4 CTRL write in the group body to omit"
    );
    let (res, args, _) = run(&kept, Some(ticks * 2 + 4096));
    let wrong = plans
        .iter()
        .zip(&ex)
        .enumerate()
        .any(|(e, (p, x))| args[2][c_range_g(&g, p, e)] != x.eager_c[..]);
    assert!(res.is_err()||wrong,"{what}: omitting the {removed} col 0 S2MM4 reset writes of the group body still gave the exact result: the POLL self-loop reset is not load-bearing");
    eprintln!(
        "NEGATIVE {what}: omitting {removed} col 0 S2MM4 CTRL writes of the group body -> {}",
        match &res {
            Err(e) => format!("failed: {e}"),
            Ok(()) => "wrong C".into(),
        }
    );
}
#[test]
fn g80_grouped_ring_omitting_col0_s2mm4_reset_is_detected_e3() {
    for ctl in BOTH {
        ring_poll_reset_omission_detected(&[64, 160, 200], 128, ctl, RingLayout { nslots: 2 }, 0, 1)
    }
}

// ================================================================ Epilogue::I32: direct half-stripe int32 C
// The host C arena of an I32 design holds `COLS * waves * 4` core tiles of 128x80 little-endian i32 (40960 B, `(mb 0..16,
// nb 0..10)` 8x8 blocks) in the order column -> stream (core row pair) -> wave -> row parity -> tile. Everything below
// rebuilds that layout and the exact values independently of `ArrayDesign::unpack_out` / `pair_c_offset`.
const T32: usize = 4 * C_TILE;
const POISONS: [u8; 2] = [0xa5, 0x5a];
const MEM_C0: u32 = 0x4000;
const MEM_C_ROW: u32 = 0x5000;
const STRIPE_WORDS: u32 = 5120;
const PROD_BD: [[u32; 2]; 4] = [[0, 1], [24, 25], [2, 3], [26, 27]];
const DRAIN_CH: [u32; 2] = [0, 3];
const DRAIN_BD0: [u32; 2] = [6, 32];
fn design32(m: usize, k: usize, ctl: Control) -> ArrayDesign {
    design_g80(m, N, k, Epilogue::I32, ctl)
}

/// Argument sizes, host byte totals, and an independent check that the per-tile ranges tile the C arena exactly once with
/// the documented offsets.
fn check_layout32(d: &ArrayDesign, m: usize, k: usize, label: &str) {
    let (kc, waves) = (k / 64, m.div_ceil(WAVE_M));
    assert_eq!(d.waves(), waves, "{label}: waves");
    assert_eq!(d.wave_m(), WAVE_M, "{label}: wave_m");
    let sizes = [
        A_COLS * waves * kc * A_CHUNK,
        COLS * kc * B_CHUNK,
        COLS * waves * 4 * T32,
    ];
    assert_eq!(
        d.args,
        vec![
            ArgSpec {
                bytes: sizes[0],
                kind: ArgKind::In
            },
            ArgSpec {
                bytes: sizes[1],
                kind: ArgKind::In
            },
            ArgSpec {
                bytes: sizes[2],
                kind: ArgKind::Out
            }
        ],
        "{label}: args"
    );
    assert_eq!(d.host_bytes_read(), (sizes[0] + sizes[1]) as u64, "{label}");
    assert_eq!(d.host_bytes_written(), sizes[2] as u64, "{label}");
    let tiles = d.output_tile_ranges();
    assert_eq!(tiles.len(), COLS * 4, "{label}: output tiles");
    let mut all: Vec<Range<usize>> = Vec::new();
    for ((col, row), ranges) in &tiles {
        assert_eq!(ranges.len(), waves, "{label}: tile ({col},{row}) ranges");
        let (col, rho) = (*col as usize, *row as usize - 2);
        for (w, r) in ranges.iter().enumerate() {
            assert_eq!(r.len(), T32, "{label}: tile ({col},{rho}) range length");
            assert_eq!(
                r.start,
                col * waves * 4 * T32 + (rho / 2) * waves * 2 * T32 + (w * 2 + rho % 2) * T32,
                "{label}: tile ({col},{rho}) wave {w} offset"
            );
            all.push(r.clone())
        }
    }
    all.sort_by_key(|r| r.start);
    let mut end = 0;
    for r in &all {
        assert_eq!(
            r.start, end,
            "{label}: C tile ranges must tile arg2 exactly once"
        );
        end = r.end
    }
    assert_eq!(end, sizes[2], "{label}: C tile ranges cover arg2");
}

/// The complete padded packed C arena for the row-major `m*N` result `c`: every word of every tile of every wave, rows beyond
/// `m` (the zero padding of the last wave) being 0. Panics if the construction does not write every word exactly once.
fn oracle_arena(c: &[i32], m: usize, waves: usize) -> Vec<u8> {
    assert_eq!(c.len(), m * N);
    let mut out = vec![0u8; COLS * waves * 4 * T32];
    let mut words = 0usize;
    for col in 0..COLS {
        for w in 0..waves {
            for rho in 0..4 {
                let (m0, n0) = (w * WAVE_M + (rho / 2) * 128, (2 * col + rho % 2) * 80);
                let base =
                    col * waves * 4 * T32 + (rho / 2) * waves * 2 * T32 + (w * 2 + rho % 2) * T32;
                for mb in 0..16 {
                    for nb in 0..10 {
                        for i in 0..8 {
                            for j in 0..8 {
                                let (row, cc) = (m0 + mb * 8 + i, n0 + nb * 8 + j);
                                let v = if row < m { c[row * N + cc] } else { 0 };
                                let at = base + 4 * ((mb * 10 + nb) * 64 + i * 8 + j);
                                out[at..at + 4].copy_from_slice(&v.to_le_bytes());
                                words += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        words * 4,
        out.len(),
        "oracle must write every word of the arena exactly once"
    );
    out
}

/// Per-channel deltas of `runs` bodies moving `waves` waves in total: the int8 formula for everything but S2MM0/1, which
/// move `waves*2*40960` B per column; 16 S2MM channels exist. Returns (read, written) of the delta.
fn assert_traffic32(
    before: &Traffic,
    after: &Traffic,
    runs: usize,
    waves: usize,
    kc: usize,
    label: &str,
) -> (u64, u64) {
    assert_eq!(before.len(), after.len(), "{label}");
    let (mut read, mut written, mut s2mm) = (0, 0, 0usize);
    for (&(col, dir, ch, was), &(col1, dir1, ch1, now)) in before.iter().zip(after) {
        assert_eq!((col, dir, ch), (col1, dir1, ch1), "{label}");
        let delta = now - was;
        let want = match (dir, ch) {
            (Direction::S2mm, 0) | (Direction::S2mm, 1) => (waves * 2 * T32) as u64,
            _ => expected_bytes(dir, ch, col, runs, waves, kc),
        };
        assert_eq!(delta, want, "{label}: shim col {col} {dir:?} ch {ch} bytes");
        match dir {
            Direction::Mm2s => read += delta,
            Direction::S2mm => {
                written += delta;
                s2mm += 1
            }
        }
    }
    assert_eq!(s2mm, 2 * COLS, "{label}: S2MM channels");
    (read, written)
}

struct Run32 {
    ticks: usize,
    busy: u64,
    wait: u64,
    read: u64,
    written: u64,
    c: Vec<u8>,
}
/// One submit of `insts` (full or lean TXN of the I32 design `d`) on a possibly reused `Config`, C arena filled with `poison`.
/// Proves: every raw int32 of the arena overwritten (none is the poison word) and bounded by `k*16384`; the arena equals the
/// oracle word for word (padded rows 0); `unpack_out` and `d.reference` equal the independent `cpu_reference`; operands and
/// packed inputs unchanged; exact per-channel shim bytes; shim written bytes == arena bytes (with the exact tile tiling of
/// `check_layout32` no word can have been written twice).
fn submit32(
    sim: &mut Config,
    d: &ArrayDesign,
    insts: &[u8],
    label: &str,
    m: usize,
    k: usize,
    a: &[i8],
    b: &[i8],
    poison: u8,
) -> Run32 {
    let (kc, waves) = (k / 64, m.div_ceil(WAVE_M));
    let [a_host, b_host] = d.pack_in(a, b);
    assert_eq!(
        (a_host.len(), b_host.len()),
        (d.args[0].bytes, d.args[1].bytes),
        "{label}: packed sizes"
    );
    let (a0, b0) = (a.to_vec(), b.to_vec());
    let before = [signature(&a_host), signature(&b_host)];
    let mut args = vec![a_host, b_host, vec![poison; d.args[2].bytes]];
    let (t0, ticks0, (busy0, wait0)) = (traffic(sim), sim.ticks, core_sum(sim));
    sim.submit(insts, &mut args)
        .unwrap_or_else(|e| panic!("{label} {m}x{N}x{k}: submit failed: {e}"));
    let ticks = sim.ticks - ticks0;
    let (busy1, wait1) = core_sum(sim);
    let (pw, bound) = (i32::from_le_bytes([poison; 4]), k as i64 * 16384);
    assert!(
        (pw as i64).abs() > bound,
        "{label}: the poison word must lie outside the accumulator bound"
    );
    assert_eq!(args[2].len() % 4, 0, "{label}");
    for (i, w) in args[2].chunks_exact(4).enumerate() {
        let v = i32::from_le_bytes(w.try_into().unwrap());
        assert!(
            v != pw,
            "{label}: raw word {i} of the C arena was never written (still poison)"
        );
        assert!(
            (v as i64).abs() <= bound,
            "{label}: raw word {i} = {v} exceeds the accumulator bound {bound}"
        );
    }
    let cpu = cpu_reference(a, b, m, N, k);
    assert_eq!(
        d.reference(a, b),
        cpu,
        "{label}: design reference != independent reference"
    );
    let oracle = oracle_arena(&cpu, m, waves);
    assert_eq!(oracle.len(), args[2].len(), "{label}: arena size");
    if let Some(at) = first_diff(&args[2], &oracle) {
        let (word, tile) = (at / 4, at / T32);
        panic!("{label} {m}x{N}x{k}: raw arena != oracle at byte {at} (word {word}, tile {tile}): got {} want {}",
            i32::from_le_bytes(args[2][at&!3..(at&!3)+4].try_into().unwrap()),i32::from_le_bytes(oracle[at&!3..(at&!3)+4].try_into().unwrap()))
    }
    let got = d.unpack_out(&args[2]);
    assert_eq!(got.len(), cpu.len(), "{label}");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != cpu[i]) {
        panic!(
            "{label} {m}x{N}x{k}: unpack_out mismatch at ({},{}) got {} want {}",
            i / N,
            i % N,
            got[i],
            cpu[i]
        )
    }
    assert_eq!(
        [signature(&args[0]), signature(&args[1])],
        before,
        "{label}: packed inputs changed"
    );
    assert!(a == &a0[..] && b == &b0[..], "{label}: operands changed");
    let (read, written) = assert_traffic32(&t0, &traffic(sim), 1, waves, kc, label);
    assert_eq!(
        (read, written),
        (d.host_bytes_read(), d.host_bytes_written()),
        "{label}: host byte totals"
    );
    assert_eq!(
        written,
        args[2].len() as u64,
        "{label}: shim S2MM traffic must equal the arena bytes (no duplicate writes)"
    );
    let padded = (waves * WAVE_M - m) * N;
    eprintln!("PASS {label} {m}x{N}x{k} I32: {} raw words == oracle ({padded} padded-row words overwritten with 0, all |v|<={bound}); {ticks} functional ticks (not hardware cycles), \
        {:.1}/wave; core busy={} lock_wait={}; DRAM read={read} write={written} (per column S2MM0/1={} B)",args[2].len()/4,ticks as f64/waves as f64,busy1-busy0,wait1-wait0,waves*2*T32);
    Run32 {
        ticks,
        busy: busy1 - busy0,
        wait: wait1 - wait0,
        read,
        written,
        c: args.swap_remove(2),
    }
}
fn once32(d: &ArrayDesign, label: &str, m: usize, k: usize, a: &[i8], b: &[i8]) -> Run32 {
    submit32(&mut fresh(d), d, &d.insts, label, m, k, a, b, POISONS[0])
}

// ---------------------------------------------------------------- operands
/// Rows alternate -128 / +127 (`flip` exchanges them), as do the columns: C = k*16384 (-128 x -128), -k*128*127 (mixed) and
/// k*127*127 (+127 x +127) all occur, so K = 2560 reaches +41943040 and -41615360.
fn signed_extremes(m: usize, k: usize, flip: bool) -> (Vec<i8>, Vec<i8>) {
    let a = (0..m * k)
        .map(|x| if (x / k % 2 == 0) ^ flip { -128 } else { 127 })
        .collect();
    let b = (0..k * N)
        .map(|x| if (x % N % 2 == 0) ^ flip { -128 } else { 127 })
        .collect();
    (a, b)
}
type Operands = (Vec<i8>, Vec<i8>);

// ---------------------------------------------------------------- exact full submits, two different operands per Config
/// Two different operand sets on ONE `Config` (the second submit starts from the end state of the first: wave parity
/// `waves%2`, `kc*waves%2`), each with a different poison byte; the two arenas differ and the second equals a fresh-`Config`
/// run of the same operands byte for byte.
fn pair32(label: &str, m: usize, k: usize, ctl: Control, ops: [Operands; 2]) {
    let d = design32(m, k, ctl);
    check_layout32(&d, m, k, label);
    let mut sim = fresh(&d);
    let waves = d.waves();
    let what = format!("{label} {ctl:?}");
    let runs: Vec<Run32> = ops
        .iter()
        .enumerate()
        .map(|(i, (a, b))| {
            submit32(
                &mut sim,
                &d,
                &d.insts,
                &format!(
                    "{what} submit{i} (waves {waves}, kc*waves {})",
                    k / 64 * waves
                ),
                m,
                k,
                a,
                b,
                POISONS[i],
            )
        })
        .collect();
    assert!(
        runs[0].c != runs[1].c,
        "{what}: different operands must give different C arenas"
    );
    let (a, b) = &ops[1];
    let fresh_run = once32(
        &d,
        &format!("{what} fresh Config, operands of submit1"),
        m,
        k,
        a,
        b,
    );
    if let Some(at) = first_diff(&runs[1].c, &fresh_run.c) {
        panic!("{what}: reused-Config arena != fresh-Config arena at byte {at}")
    }
}
fn random_pair(m: usize, k: usize) -> [Operands; 2] {
    [random(m, k, SEEDS[0]), random(m, k, SEEDS[1])]
}
#[test]
fn g80_i32_exact_one_wave_m1_200_256_k64_128() {
    for m in [1, 200, 256] {
        for k in [64, 128] {
            for ctl in BOTH {
                pair32("G80 I32 one wave", m, k, ctl, random_pair(m, k))
            }
        }
    }
}
#[test]
fn g80_i32_exact_multi_wave_m300_513_k64_128() {
    for m in [300, 513] {
        for k in [64, 128] {
            for ctl in BOTH {
                pair32("G80 I32 multi wave", m, k, ctl, random_pair(m, k))
            }
        }
    }
}
#[test]
fn g80_i32_exact_k2560_small_m() {
    for m in [8, 64] {
        for ctl in BOTH {
            pair32("G80 I32 K2560 small M", m, 2560, ctl, random_pair(m, 2560))
        }
    }
}
/// Waves 1..4 and padded M back to back: odd and even `waves` / `kc*waves` on one `Config` each.
#[test]
fn g80_i32_consecutive_submits_odd_and_even_wave_parity() {
    for (m, k) in [
        (256, 64),
        (512, 64),
        (768, 64),
        (1024, 64),
        (200, 128),
        (300, 128),
        (513, 64),
    ] {
        for ctl in BOTH {
            pair32("G80 I32 parity", m, k, ctl, random_pair(m, k))
        }
    }
}
/// K = 2560 with -128 / +127 rows and columns reaches the accumulator bounds exactly (+41943040 / -41615360) in BOTH operand
/// polarities, the second submit following the first on the same `Config`.
#[test]
fn g80_i32_signed_extremes_k2560_reach_exact_bounds() {
    for m in [8, 64] {
        let k = 2560;
        for flip in [false, true] {
            let (a, b) = signed_extremes(m, k, flip);
            let c = cpu_reference(&a, &b, m, N, k);
            assert_eq!(
                (
                    c.iter().copied().max().unwrap(),
                    c.iter().copied().min().unwrap()
                ),
                (41943040, -41615360),
                "M={m} flip={flip}: extremes must reach the bounds"
            );
        }
        for ctl in BOTH {
            pair32(
                "G80 I32 K2560 extremes",
                m,
                k,
                ctl,
                [signed_extremes(m, k, false), signed_extremes(m, k, true)],
            )
        }
    }
}
/// The mixed -128 / 127 / -1 / 1 / 0 / alternating / pseudo-random classes of the int8 extremes test at raw int32 precision.
#[test]
fn g80_i32_extreme_classes_full_range() {
    for (m, k) in [(200, 128), (300, 64), (256, 640)] {
        for ctl in BOTH {
            pair32(
                "G80 I32 extreme classes",
                m,
                k,
                ctl,
                [extremes(m, k, 0), extremes(m, k, 3)],
            )
        }
    }
}
/// The int8 and I32 designs of one shape pack identical operands and differ only in the C stream (4x the bytes).
#[test]
fn g80_i32_layout_is_four_times_int8_c_and_same_inputs() {
    for (m, k) in [(1, 64), (200, 128), (300, 64), (513, 128), (64, 2560)] {
        let (a, b) = random(m, k, SEEDS[2]);
        let (d8, d32) = (
            design(m, k, Control::Fast, 12),
            design32(m, k, Control::Fast),
        );
        check_layout32(&d32, m, k, "I32 layout");
        assert_eq!(
            d8.pack_in(&a, &b),
            d32.pack_in(&a, &b),
            "M={m} K={k}: packed A / B must not depend on the epilogue"
        );
        assert_eq!(
            d32.args[2].bytes,
            4 * d8.args[2].bytes,
            "M={m} K={k}: C bytes"
        );
        assert_eq!(
            (d8.args[0].bytes, d8.args[1].bytes),
            (d32.args[0].bytes, d32.args[1].bytes)
        );
    }
}
#[test]
fn g80_i32_rejects_unsupported_shapes_and_probe_layouts() {
    let mut cases: Vec<(String, bool)> = Vec::new();
    for n in [0usize, 1, 640, 1279, 1281, 2560] {
        cases.push((
            format!("N={n}"),
            rejects(|| {
                design_g80(256, n, 128, Epilogue::I32, Control::Fast);
            }),
        ))
    }
    for k in [0usize, 1, 63, 65, 2559, 2561, 5120] {
        cases.push((
            format!("K={k}"),
            rejects(|| {
                design_g80(256, N, k, Epilogue::I32, Control::Fast);
            }),
        ))
    }
    for m in [0usize, 256 * 256 + 1] {
        cases.push((
            format!("M={m}"),
            rejects(|| {
                design_g80(m, N, 64, Epilogue::I32, Control::Fast);
            }),
        ))
    }
    for layout in 1u8..=3 {
        cases.push((
            format!("layout={layout}"),
            rejects(|| {
                design_g80(
                    256,
                    N,
                    64,
                    Epilogue::I32,
                    Control::Probe(Probe {
                        layout,
                        ..Probe::default()
                    }),
                );
            }),
        ))
    }
    assert_all_rejected(cases);
}

// ---------------------------------------------------------------- memtile descriptors, locks and queues parsed from the TXN
// Contract: per column and core row `r` ONE 20480 B slot at memtile offset 0x4000 + r*0x5000 holds each of the two 20480 B
// stripes of a wave in turn. Producer S2MM `r` runs the finite chain PROD_BD[r][0] -> [1] (5120 words each, `repeat = mw`),
// the finite drain chain DRAIN_BD0[g]..+3 (MM2S0 rows 0, 1 / MM2S3 rows 2, 3: row0, row0, row1, row1, `repeat = mw`)
// releases the slot back. Both chains of a row use the same empty / full lock pair (initial 1 / 0).
#[derive(Debug, PartialEq, Clone, Copy)]
struct MemBd {
    len: u32,
    addr: u32,
    next: Option<u32>,
    acq: Option<(u32, i32)>,
    rel: Option<(u32, i32)>,
}
fn s7(v: u32) -> i32 {
    (((v & 0x7f) << 25) as i32) >> 25
}
fn mem_bd_of(ops: &[TxnOp], col: u32, id: u32, what: &str) -> MemBd {
    let at = mem_bd_addr(col, id);
    let found: Vec<&Vec<u32>> = ops
        .iter()
        .filter_map(|o| match o {
            TxnOp::BlockWrite(a, w) if *a == at && w.len() == 8 => Some(w),
            _ => None,
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{what}: col {col} memtile BD{id} must be written exactly once"
    );
    let w = found[0];
    MemBd {
        len: w[0] & 0x1ffff,
        addr: (w[1] & 0x7ffff) * 4,
        next: if w[1] >> 19 & 1 == 1 {
            Some(w[1] >> 20 & 0x3f)
        } else {
            None
        },
        acq: if w[7] >> 15 & 1 == 1 {
            Some((w[7] & 0xff, s7(w[7] >> 8)))
        } else {
            None
        },
        rel: Some((w[7] >> 16 & 0xff, s7(w[7] >> 24))),
    }
}
/// Memtile (row 1) start-queue writes of `(col, mm2s, ch)` as `(first BD, repeat)`.
fn queue_writes(ops: &[TxnOp], col: u32, mm2s: bool, ch: u32) -> Vec<(u32, u32)> {
    let off = mem_queue_off(mm2s, ch);
    ops.iter()
        .filter_map(|o| match *o {
            TxnOp::Write(a, v) if a >> 25 == col && (a >> 20) & 31 == 1 && a & 0xfffff == off => {
                Some((v & 0xffff, (v >> 16 & 0xff) + 1))
            }
            _ => None,
        })
        .collect()
}
/// Every memtile start-queue write in the ops (all columns) queues a BD of the bank of its channel (even channel => id < 24,
/// odd channel => id >= 24) and no BD id reaches 48; every memtile BD block write targets an id < 48.
fn assert_bank_aware(ops: &[TxnOp], what: &str) -> usize {
    let mut n = 0;
    for o in ops {
        match *o {
            TxnOp::Write(a, v)
                if (a >> 20) & 31 == 1
                    && (0xa0600..0xa0660).contains(&(a & 0xfffff))
                    && (a & 0xfffff) % 8 == 4 =>
            {
                let idx = ((a & 0xfffff) - 0xa0604) / 8;
                assert!(idx < 12, "{what}: queue register {a:#x}");
                let ch = if idx < 6 { idx } else { idx - 6 };
                let bd = v & 0xffff;
                assert!(bd < 48, "{what}: queue write {a:#x} queues BD{bd}");
                assert_eq!(
                    bd >= 24,
                    ch % 2 == 1,
                    "{what}: memtile {} ch {ch} queued BD{bd}: BD bank != channel bank",
                    if idx < 6 { "S2MM" } else { "MM2S" }
                );
                n += 1;
            }
            TxnOp::BlockWrite(a, ref w)
                if (a >> 20) & 31 == 1
                    && w.len() == 8
                    && (0xa0000..0xa0600).contains(&(a & 0xfffff)) =>
            {
                assert!(
                    ((a & 0xfffff) - 0xa0000) / 0x20 < 48,
                    "{what}: memtile BD {a:#x}"
                )
            }
            _ => {}
        }
    }
    n
}
/// The C-channel queues of every column: S2MM `r` queues PROD_BD[r][0] and MM2S 0 / 3 queue BD 6 / 32 with `repeat = mw`;
/// every such write is that one (`once`: exactly one write, as in the full TXN).
fn assert_c_queues(ops: &[TxnOp], mw: usize, once: bool, what: &str) {
    let check = |got: Vec<(u32, u32)>, want: (u32, u32), label: String| {
        assert!(
            !got.is_empty() && got.iter().all(|&q| q == want),
            "{what}: {label}: queue writes {got:?}, want only {want:?}"
        );
        if once {
            assert_eq!(got.len(), 1, "{what}: {label}: exactly one queue write")
        }
    };
    for col in 0..COLS as u32 {
        for r in 0..4 {
            check(
                queue_writes(ops, col, false, r as u32)
                    .into_iter()
                    .filter(|q| PROD_BD[r].contains(&q.0))
                    .collect(),
                (PROD_BD[r][0], mw as u32),
                format!("col {col} C producer S2MM{r}"),
            )
        }
        for g in 0..2 {
            check(
                queue_writes(ops, col, true, DRAIN_CH[g]),
                (DRAIN_BD0[g], mw as u32),
                format!("col {col} C drain MM2S{}", DRAIN_CH[g]),
            )
        }
    }
}
fn assert_i32_descriptors(m: usize, k: usize, ctl: Control) {
    let d = design32(m, k, ctl);
    let mw = m.div_ceil(WAVE_M);
    let what = format!("G80 I32 descriptors {ctl:?} {m}x{N}x{k}");
    let ops = parse_txn(&d.insts).unwrap();
    assert!(
        assert_bank_aware(&ops, &what) > 0,
        "{what}: no memtile queue writes"
    );
    let mem_end = 0x80000 + 0x18000;
    for col in 0..COLS as u32 {
        let mut pairs = Vec::new();
        for r in 0..4usize {
            let addr = 0x80000 + MEM_C0 + r as u32 * MEM_C_ROW;
            let w = format!("{what} col {col} row {r}");
            let p = PROD_BD[r].map(|id| mem_bd_of(&ops, col, id, &w));
            for (i, b) in p.iter().enumerate() {
                assert_eq!(
                    (b.len, b.addr),
                    (STRIPE_WORDS, addr),
                    "{w}: producer BD{} length / address",
                    PROD_BD[r][i]
                );
                assert!(
                    b.addr + 4 * b.len <= mem_end,
                    "{w}: C slot overlaps A / B regions"
                );
            }
            assert_eq!(
                (p[0].next, p[1].next),
                (Some(PROD_BD[r][1]), None),
                "{w}: finite producer chain"
            );
            assert_eq!(p[0].acq, p[1].acq, "{w}");
            assert_eq!(
                p[0].rel, p[1].rel,
                "{w}: both stripes use one empty / full pair"
            );
            let (empty, full) = (
                p[0].acq.expect("producer acquires empty"),
                p[0].rel.expect("producer releases full"),
            );
            assert_eq!((empty.1, full.1), (-1, 1), "{w}: producer lock values");
            assert!(
                (72..80).contains(&empty.0) && (72..80).contains(&full.0),
                "{w}: only memtile locks 8..15 carry C: empty {} full {}",
                empty.0,
                full.0
            );
            let (g, rr) = (r / 2, (r % 2) as u32);
            for t in 0..2u32 {
                let id = DRAIN_BD0[g] + 2 * rr + t;
                let b = mem_bd_of(&ops, col, id, &w);
                assert_eq!(
                    (b.len, b.addr),
                    (STRIPE_WORDS, addr),
                    "{w}: drain BD{id} length / address"
                );
                assert_eq!(b.acq, Some((full.0, -1)), "{w}: drain BD{id} acquires full");
                assert_eq!(
                    b.rel,
                    Some((empty.0, 1)),
                    "{w}: drain BD{id} releases empty"
                );
            }
            assert_eq!(PROD_BD[r][0] >= 24, r % 2 == 1, "{w}: producer bank");
            assert_eq!(DRAIN_BD0[g] >= 24, DRAIN_CH[g] % 2 == 1, "{w}: drain bank");
            let local = (empty.0 - 64, full.0 - 64);
            for (id, want) in [(local.0, 1u32), (local.1, 0)] {
                let at = peacemaker_npu::regs::tile(col, 1, 0xc0000 + id * 0x10);
                let vals: Vec<u32> = ops
                    .iter()
                    .filter_map(|o| match *o {
                        TxnOp::Write(a, v) if a == at => Some(v),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    vals.last().copied(),
                    Some(want),
                    "{w}: memtile lock {id} initial value (writes {vals:?})"
                );
            }
            pairs.push(local);
        }
        for g in 0..2 {
            for t in 0..4u32 {
                let b = mem_bd_of(&ops, col, DRAIN_BD0[g] + t, &format!("{what} col {col}"));
                assert_eq!(
                    b.next,
                    if t < 3 {
                        Some(DRAIN_BD0[g] + t + 1)
                    } else {
                        None
                    },
                    "{what} col {col}: drain chain {g} BD{} next",
                    DRAIN_BD0[g] + t
                );
            }
        }
        let mut ids: Vec<u32> = pairs.iter().flat_map(|p| [p.0, p.1]).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 8, "{what} col {col}: four distinct lock pairs");
        for id in 16..24u32 {
            let at = peacemaker_npu::regs::tile(col, 1, 0xc0000 + id * 0x10);
            assert!(ops.iter().all(|o|!matches!(o,TxnOp::Write(a,_) if *a==at)),"{what} col {col}: memtile lock {id} (the removed second slot) must not be initialised");
        }
    }
    assert_c_queues(&ops, mw, true, &what);
    // lean and forced-requeue TXNs queue the same finite C tasks and obey the same bank rule
    let lean = parse_txn(&lean_of(&d, &what)).unwrap();
    assert!(assert_bank_aware(&lean, &format!("{what} lean")) > 0);
    assert_c_queues(&lean, mw, false, &format!("{what} lean"));
    for extra in [vec![FORCE_B], vec![FORCE_C], vec![FORCE_B, FORCE_C]] {
        let f = parse_txn(
            &d.lean_insts_requeue(&extra)
                .unwrap_or_else(|e| panic!("{what}: lean_insts_requeue: {e}")),
        )
        .unwrap();
        assert_bank_aware(&f, &format!("{what} forced {extra:?}"));
        assert_c_queues(&f, mw, false, &format!("{what} forced {extra:?}"));
    }
    eprintln!("PASS {what}: per column 4 single 20480 B slots, producers {PROD_BD:?}, drains {DRAIN_BD0:?} (+3), repeat mw={mw}, locks parsed, banks checked");
}
#[test]
fn g80_i32_memtile_descriptors_locks_and_queues_one_wave() {
    for (m, k) in [(256, 64), (200, 128), (1, 64)] {
        for ctl in BOTH {
            assert_i32_descriptors(m, k, ctl)
        }
    }
}
#[test]
fn g80_i32_memtile_descriptors_locks_and_queues_multi_wave() {
    for (m, k) in [(512, 64), (513, 128), (768, 192)] {
        for ctl in BOTH {
            assert_i32_descriptors(m, k, ctl)
        }
    }
}
/// The int8 C descriptors are untouched: two 10240 B slots, cyclic producer rings and a cyclic four BD drain chain.
#[test]
fn g80_int8_c_descriptors_unchanged_next_to_i32() {
    for (m, k) in [(256, 64), (512, 128)] {
        let d = design(m, k, Control::Fast, 12);
        let ops = parse_txn(&d.insts).unwrap();
        assert_bank_aware(&ops, "int8");
        for col in 0..COLS as u32 {
            for r in 0..4usize {
                let off = |slot: u32| 0x80000 + MEM_C0 + slot * 0xa000 + r as u32 * 0x2800;
                let w = format!("int8 col {col} row {r}");
                let p = PROD_BD[r].map(|id| mem_bd_of(&ops, col, id, &w));
                assert_eq!(
                    (p[0].len, p[0].addr, p[1].len, p[1].addr),
                    (2560, off(0), 2560, off(1)),
                    "{w}: producer slots"
                );
                assert_eq!(
                    (p[0].next, p[1].next),
                    (Some(PROD_BD[r][1]), Some(PROD_BD[r][0])),
                    "{w}: cyclic ring"
                );
                let (g, rr) = (r / 2, (r % 2) as u32);
                for slot in 0..2u32 {
                    let id = DRAIN_BD0[g] + slot * 2 + rr;
                    let b = mem_bd_of(&ops, col, id, &w);
                    assert_eq!(
                        (b.len, b.addr, b.next),
                        (
                            2560,
                            off(slot),
                            Some(DRAIN_BD0[g] + (slot * 2 + rr + 1) % 4)
                        ),
                        "{w}: drain BD{id}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------- lean submits and forced requeues (I32 has no C parity repair)
/// One submit of `insts` on `sim` for seed operands: exact (full oracle arena), byte-equal to a fresh full run (whose own
/// retained state equals `initial`), and the retained state afterwards equals `initial`.
macro_rules! lean_step32 {
    ($sim:expr,$d:expr,$insts:expr,$initial:expr,$m:expr,$k:expr,$seed:expr,$what:expr) => {{
        let (a, b) = random($m, $k, $seed);
        let what: String = $what;
        let mut full = fresh($d);
        let reference = submit32(
            &mut full,
            $d,
            &$d.insts,
            &format!("{what} (fresh full oracle)"),
            $m,
            $k,
            &a,
            &b,
            POISONS[0],
        );
        assert!(
            full.state_snapshot() == $initial,
            "{what}: fresh full run state != initial full state"
        );
        let run = submit32($sim, $d, $insts, &what, $m, $k, &a, &b, POISONS[1]);
        if let Some(i) = first_diff(&run.c, &reference.c) {
            panic!("{what}: C bytes != fresh full TXN bytes at byte {i}")
        }
        assert!(
            $sim.state_snapshot() == $initial,
            "{what}: retained state != fresh full state"
        );
        eprintln!(
            "MEASURED {what}: {} ticks vs fresh full {} ticks",
            run.ticks, reference.ticks
        );
    }};
}
/// Full (zero operands) -> three lean submits with fresh operands on the SAME Config.
fn lean_repeat32(m: usize, k: usize, ctl: Control) {
    let d = design32(m, k, ctl);
    let what = format!("G80 I32 {ctl:?} {m}x{N}x{k}");
    let lean = lean_of(&d, &what);
    let mut sim = primed(&d);
    let initial = sim.state_snapshot();
    for (i, seed) in SEEDS.into_iter().enumerate() {
        lean_step32!(
            &mut sim,
            &d,
            &lean,
            initial,
            m,
            k,
            seed,
            format!("{what} lean #{i}")
        )
    }
    eprintln!("PASS I32 lean x3 {what} (waves {}): exact oracle arenas, byte-equal to fresh full, state == fresh full; full {} B lean {} B",d.waves(),d.insts.len(),lean.len());
}
/// full -> lean -> full -> lean on one Config.
fn lean_interleave32(m: usize, k: usize, ctl: Control) {
    let d = design32(m, k, ctl);
    let what = format!("G80 I32 {ctl:?} {m}x{N}x{k}");
    let lean = lean_of(&d, &what);
    let mut sim = primed(&d);
    let initial = sim.state_snapshot();
    for (i, (seed, is_lean)) in [
        (SEEDS[0], false),
        (SEEDS[1], true),
        (SEEDS[2], false),
        (0xdeadbeef, true),
    ]
    .into_iter()
    .enumerate()
    {
        let label = format!(
            "{what} interleave #{i} {}",
            if is_lean { "lean" } else { "full" }
        );
        lean_step32!(
            &mut sim,
            &d,
            if is_lean { &lean[..] } else { &d.insts[..] },
            initial,
            m,
            k,
            seed,
            label
        );
    }
    eprintln!("PASS I32 full->lean->full->lean {what}");
}
/// Forced MM2S0 / S2MM4 requeues keep every finite queue incl. the C tasks (`repeat = mw`) and give the fresh full state.
fn forced_requeue32(m: usize, k: usize, ctl: Control) {
    let d = design32(m, k, ctl);
    let what = format!("G80 I32 forced {ctl:?} {m}x{N}x{k}");
    let plain = lean_of(&d, &what);
    let sets: [(&str, Vec<(u32, DmaDirection, u32)>); 3] = [
        ("B S2MM4", vec![FORCE_B]),
        ("C MM2S0", vec![FORCE_C]),
        ("B S2MM4 + C MM2S0", vec![FORCE_B, FORCE_C]),
    ];
    for (label, extra) in sets {
        let forced = d
            .lean_insts_requeue(&extra)
            .unwrap_or_else(|e| panic!("{what} {label}: lean_insts_requeue: {e}"));
        let mut sim = primed(&d);
        let initial = sim.state_snapshot();
        for (i, seed) in SEEDS.into_iter().enumerate() {
            lean_step32!(
                &mut sim,
                &d,
                &forced,
                initial,
                m,
                k,
                seed,
                format!("{what} {label} #{i}")
            )
        }
        lean_step32!(
            &mut sim,
            &d,
            &plain,
            initial,
            m,
            k,
            0xdeadbeef,
            format!("{what} plain lean after {label}")
        );
        eprintln!(
            "PASS I32 forced requeue {label} {what}: lean {} B forced {} B",
            plain.len(),
            forced.len()
        );
    }
}
/// Waves 1..3 (odd and even): the lean body needs no C parity repair, every wave count behaves like the full state.
const LEAN32_SHAPES: [(usize, usize); 6] = [
    (256, 64),
    (256, 128),
    (512, 64),
    (768, 64),
    (200, 128),
    (513, 64),
];
#[test]
fn g80_i32_lean_three_submits_exact_and_state_restored() {
    for (m, k) in LEAN32_SHAPES {
        for ctl in BOTH {
            lean_repeat32(m, k, ctl)
        }
    }
}
#[test]
fn g80_i32_lean_full_lean_interleave_exact_and_state_restored() {
    for (m, k) in LEAN32_SHAPES {
        for ctl in BOTH {
            lean_interleave32(m, k, ctl)
        }
    }
}
#[test]
fn g80_i32_lean_forced_requeue_one_to_three_waves() {
    for (m, k) in [(256, 64), (512, 128), (768, 64), (200, 128)] {
        for ctl in BOTH {
            forced_requeue32(m, k, ctl)
        }
    }
}
#[test]
fn g80_i32_lean_large_k() {
    for (m, k) in [(160, 2560), (512, 640)] {
        lean_repeat32(m, k, Control::Fast);
        lean_interleave32(m, k, Control::Fast)
    }
}

// ---------------------------------------------------------------- grouped launch (`append_run_body`) with I32 experts
/// Experts with distinct M, all padding to the same wave count: operands, packed operands and the eager fresh-`Config` full run
/// (itself a full oracle check). `shift` is unused for I32 (0).
fn experts32(k: usize, ms: &[usize], ctl: Control, seed0: u32) -> Vec<Expert> {
    let waves = ms[0].div_ceil(WAVE_M);
    ms.iter()
        .enumerate()
        .map(|(e, &m)| {
            let d = design32(m, k, ctl);
            check_layout32(&d, m, k, "I32 expert");
            assert_eq!(d.waves(), waves, "expert M={m} must pad to {waves} wave(s)");
            let (a, b) = random(m, k, seed0.wrapping_add(e as u32 * 0x9e37));
            let mut eager = fresh(&d);
            let run = submit32(
                &mut eager,
                &d,
                &d.insts,
                &format!("G80 I32 eager expert {e} {ctl:?}"),
                m,
                k,
                &a,
                &b,
                POISONS[0],
            );
            let packed = d.pack_in(&a, &b);
            Expert {
                m,
                k,
                shift: 0,
                a,
                b,
                d,
                packed,
                eager_c: run.c,
                eager,
            }
        })
        .collect()
}
/// Group TXN on `sim`: every expert's C region is byte-equal to its eager bytes, equals the independent reference, holds no
/// poison word and stays bounded; operands untouched, no byte outside the C regions written, S2MM bytes = the experts' arenas.
fn run_group32(sim: &mut Config, ex: &[Expert], txn: &[u8], what: &str) -> Vec<Vec<u8>> {
    let mut args = group_args(ex);
    let before = args.clone();
    let (t0, ticks0) = (traffic(sim), sim.ticks);
    let waves = ex[0].d.waves();
    sim.submit(txn, &mut args)
        .unwrap_or_else(|e| panic!("{what}: group submit failed: {e}"));
    let kc = ex[0].k / 64;
    let (read, written) =
        assert_traffic32(&t0, &traffic(sim), ex.len(), ex.len() * waves, kc, what);
    let mut owned = vec![false; args[2].len()];
    let pw = i32::from_le_bytes([POISON; 4]);
    for (e, x) in ex.iter().enumerate() {
        let at = arena_off(ex, e, 2) as usize;
        let r = at..at + x.d.args[2].bytes;
        let bound = x.k as i64 * 16384;
        for (i, w) in args[2][r.clone()].chunks_exact(4).enumerate() {
            let v = i32::from_le_bytes(w.try_into().unwrap());
            assert!(
                v != pw && (v as i64).abs() <= bound,
                "{what}: expert {e} (M={}) raw word {i} = {v}: unwritten or unbounded",
                x.m
            );
        }
        if let Some(i) = first_diff(&args[2][r.clone()], &x.eager_c) {
            panic!(
                "{what}: expert {e} (M={}) C bytes != eager at byte {i}",
                x.m
            )
        }
        let cpu = cpu_reference(&x.a, &x.b, x.m, N, x.k);
        assert_eq!(
            x.d.unpack_out(&args[2][r.clone()]),
            cpu,
            "{what}: expert {e} (M={}) != CPU reference",
            x.m
        );
        assert!(
            args[2][r.clone()] == oracle_arena(&cpu, x.m, waves)[..],
            "{what}: expert {e} (M={}) arena != padded oracle",
            x.m
        );
        for o in &mut owned[r] {
            *o = true
        }
    }
    assert!(
        args[0] == before[0] && args[1] == before[1],
        "{what}: operand arenas (incl. guards) changed"
    );
    for (i, &o) in owned.iter().enumerate() {
        if !o {
            assert_eq!(
                args[2][i], POISON,
                "{what}: stray write outside every C region at byte {i}"
            )
        }
    }
    assert_eq!(
        (read, written),
        (
            ex.iter().map(|x| x.d.host_bytes_read()).sum::<u64>(),
            ex.iter().map(|x| x.d.host_bytes_written()).sum::<u64>()
        ),
        "{what}: summed host bytes"
    );
    eprintln!("PASS {what}: E={} M={:?}: every expert byte-equal to its eager fresh-Config C, CPU-exact and oracle-exact; {} group ticks; DRAM read={read} write={written}",
        ex.len(),ex.iter().map(|x|x.m).collect::<Vec<_>>(),sim.ticks-ticks0);
    args
}
/// E3 group of full bodies: exact, retained state of one full run, resubmit with new operands, then a standalone run.
fn group_exact32(ms: &[usize], k: usize, ctl: Control) {
    let ex = experts32(k, ms, ctl, 0xc0ffee21);
    let txn = group_txn(&ex, false);
    let mut sim = fresh(&ex[0].d);
    let what = format!("G80 I32 group {ctl:?} M={ms:?} K={k}");
    run_group32(&mut sim, &ex, &txn, &what);
    assert!(
        sim.state_snapshot() == last_body_only(&ex).state_snapshot(),
        "{what}: retained state after the group != one fresh full run of the last body"
    );
    let ex2 = experts32(k, ms, ctl, 0xbadc0de4);
    assert_eq!(
        group_txn(&ex2, false),
        txn,
        "{what}: group TXN depends on operands"
    );
    run_group32(&mut sim, &ex2, &txn, &format!("{what} resubmit"));
    let y = &ex2[0];
    submit32(
        &mut sim,
        &y.d,
        &y.d.insts,
        &format!("{what} standalone after group"),
        y.m,
        y.k,
        &y.a,
        &y.b,
        POISONS[1],
    );
    assert!(
        sim.state_snapshot() == ex[0].eager.state_snapshot(),
        "{what}: retained state after the standalone run != after one full run"
    );
}
/// Expert 0 full body then lean bodies equals the all-full group (bytes, state), shorter, and reusable.
fn group_lean32(ms: &[usize], k: usize, ctl: Control) {
    let ex = experts32(k, ms, ctl, 0xc0ffee22);
    let (full, lean) = (group_txn(&ex, false), group_txn(&ex, true));
    let what = format!("G80 I32 lean group {ctl:?} M={ms:?} K={k}");
    let (mut sim_l, mut sim_f) = (fresh(&ex[0].d), fresh(&ex[0].d));
    let args_l = run_group32(&mut sim_l, &ex, &lean, &what);
    let args_f = run_group32(&mut sim_f, &ex, &full, &format!("{what} (all-full oracle)"));
    for i in 0..3 {
        if let Some(at) = first_diff(&args_l[i], &args_f[i]) {
            panic!("{what}: arg{i} differs from the all-full group at byte {at}")
        }
    }
    assert!(
        sim_l.state_snapshot() == sim_f.state_snapshot(),
        "{what}: retained state after the lean group != the all-full group"
    );
    assert!(
        sim_l.state_snapshot() == last_body_only(&ex).state_snapshot(),
        "{what}: retained state != one fresh full run of the last body"
    );
    let ex2 = experts32(k, ms, ctl, 0xbadc0de5);
    run_group32(&mut sim_l, &ex2, &lean, &format!("{what} resubmit"));
    assert!(
        sim_l.state_snapshot() == sim_f.state_snapshot(),
        "{what}: state after the second lean group"
    );
    eprintln!(
        "MEASURED {what}: lean group TXN {} B vs all-full {} B",
        lean.len(),
        full.len()
    );
}
#[test]
fn g80_i32_group_e3_mixed_m_exact() {
    for (k, ctl) in [
        (192, Control::Fast),
        (128, Control::Slow),
        (64, Control::Fast),
    ] {
        group_exact32(&[64, 160, 200], k, ctl)
    }
}
#[test]
fn g80_i32_group_e3_mixed_m_exact_k2560() {
    group_exact32(&[64, 160, 200], 2560, Control::Fast)
}
#[test]
fn g80_i32_group_two_waves_e3_exact() {
    group_exact32(&[300, 512, 257], 128, Control::Fast)
}
#[test]
fn g80_i32_group_lean_tail_equals_all_full() {
    for (k, ctl) in [
        (192, Control::Fast),
        (128, Control::Slow),
        (64, Control::Fast),
    ] {
        group_lean32(&[64, 160, 200], k, ctl)
    }
}
#[test]
fn g80_i32_group_lean_tail_two_waves_equals_all_full() {
    group_lean32(&[300, 512, 257], 128, Control::Fast);
    group_lean32(&[512, 300], 64, Control::Slow)
}
#[test]
fn g80_i32_group_lean_tail_k2560() {
    group_lean32(&[64, 160, 200], 2560, Control::Fast)
}

// ---------------------------------------------------------------- `experts::grouped` with an I32 G80 design
fn g80_groups32(ex: &[Expert], ctl: Control, k: usize) -> (GroupedDesign, GroupedDesign) {
    let plans = plans_of(ex);
    let rows = ex.iter().map(|x| x.m).max().unwrap().div_ceil(WAVE_M) * WAVE_M;
    let d = design32(rows, k, ctl);
    let wave_m = d.waves() * d.wave_m();
    assert_eq!(wave_m, rows);
    (
        grouped(d, wave_m, N, k, &plans, None),
        grouped_all_full(design32(rows, k, ctl), wave_m, N, k, &plans, None),
    )
}
fn check_group_g32(
    g: &GroupedDesign,
    plans: &[ExpertPlan],
    ex: &[Expert],
    args: &[Vec<u8>],
    what: &str,
) {
    let mut want_args = blank_g(g, None);
    put_g(&mut want_args, plans, ex);
    if let Some(i) = first_diff(&args[0], &want_args[0]) {
        panic!("{what}: A arena modified at byte {i}")
    }
    if let Some(i) = first_diff(&args[1], &want_args[1]) {
        panic!("{what}: B arena modified at byte {i}")
    }
    let mut covered = vec![false; args[2].len()];
    for (e, (p, x)) in plans.iter().zip(ex).enumerate() {
        let c = c_range_g(g, p, e);
        if let Some(i) = first_diff(&args[2][c.clone()], &x.eager_c) {
            panic!(
                "{what}: expert {e} (m={}) C != eager I32 bytes of a fresh Config (byte {i})",
                p.m
            )
        }
        assert_eq!(
            g.unpack_out(e, &args[2][c.clone()]),
            cpu_reference(&x.a, &x.b, x.m, N, x.k),
            "{what}: expert {e} C != CPU reference"
        );
        covered[c].fill(true);
    }
    assert!(
        args[2]
            .iter()
            .zip(&covered)
            .all(|(&b, &c)| c || b == POISON),
        "{what}: C arena written outside the expert regions"
    );
}
fn grouped_api_case32(ms: &[usize], k: usize, ctl: Control) {
    let ex = experts32(k, ms, ctl, 0xc0ffee31);
    let plans = plans_of(&ex);
    let (g, f) = g80_groups32(&ex, ctl, k);
    assert!(g.patch_sites.is_empty() && f.patch_sites.is_empty());
    assert!(g.pdi == f.pdi, "lean and all-full group load the same PDI");
    assert_eq!(g.experts(), ex.len());
    for (e, x) in ex.iter().enumerate() {
        assert_eq!(
            g.pack_in(e, &x.a, &x.b),
            x.packed,
            "expert {e}: grouped packing == eager packing"
        );
        assert_eq!(
            g.reference(e, &x.a, &x.b),
            cpu_reference(&x.a, &x.b, x.m, N, x.k),
            "expert {e}: grouped reference == independent reference"
        );
        assert_eq!(
            g.unpack_out(e, &x.eager_c),
            cpu_reference(&x.a, &x.b, x.m, N, x.k),
            "expert {e}: grouped unpack of the eager C"
        );
        assert_eq!(
            (g.a_bytes(e), g.b_bytes(e), g.c_bytes(e)),
            (x.d.args[0].bytes, x.d.args[1].bytes, x.d.args[2].bytes),
            "expert {e}: sizes"
        );
    }
    let (mut sg, mut sf) = (
        Config::from_pdi(&g.pdi).unwrap(),
        Config::from_pdi(&f.pdi).unwrap(),
    );
    let ex2 = experts32(k, ms, ctl, 0xbadc0de6);
    for (round, x) in [&ex, &ex2].into_iter().enumerate() {
        let (mut ag, mut af) = (blank_g(&g, None), blank_g(&f, None));
        put_g(&mut ag, &plans, x);
        put_g(&mut af, &plans, x);
        let (tg, tf, (g0, f0)) = (traffic(&sg), traffic(&sf), (sg.ticks, sf.ticks));
        sg.submit(&g.insts, &mut ag)
            .unwrap_or_else(|e| panic!("lean group round {round}: {e}"));
        sf.submit(&f.insts, &mut af)
            .unwrap_or_else(|e| panic!("all-full group round {round}: {e}"));
        check_group_g32(&g, &plans, x, &ag, &format!("I32 lean group round {round}"));
        check_group_g32(
            &f,
            &plans,
            x,
            &af,
            &format!("I32 all-full group round {round}"),
        );
        for i in 0..3 {
            if let Some(at) = first_diff(&ag[i], &af[i]) {
                panic!("round {round}: lean group arg{i} != all-full group arg{i} at byte {at}")
            }
        }
        let (kc, w) = (k / 64, ex[0].d.waves());
        let (rg, wg) = assert_traffic32(
            &tg,
            &traffic(&sg),
            ms.len(),
            w * ms.len(),
            kc,
            &format!("I32 lean group round {round}"),
        );
        let (rf, wf) = assert_traffic32(
            &tf,
            &traffic(&sf),
            ms.len(),
            w * ms.len(),
            kc,
            &format!("I32 all-full group round {round}"),
        );
        assert_eq!((rg, wg), (rf, wf));
        assert!(
            sg.state_snapshot() == sf.state_snapshot(),
            "round {round}: retained state after the lean group != the all-full group"
        );
        eprintln!("MEASURED G80 I32 grouped API {ctl:?} E={} M={ms:?} K={k} round {round}: TXN lean {} B vs all-full {} B; {} vs {} functional ticks (not hardware cycles); DRAM read={rg} write={wg}",
            ms.len(),g.insts.len(),f.insts.len(),sg.ticks-g0,sf.ticks-f0);
    }
}
#[test]
fn g80_i32_grouped_api_e3_mixed_m() {
    for (k, ctl) in [
        (192, Control::Fast),
        (128, Control::Slow),
        (64, Control::Fast),
    ] {
        grouped_api_case32(&[64, 160, 200], k, ctl)
    }
}
#[test]
fn g80_i32_grouped_api_e4_boundary_m() {
    grouped_api_case32(&[1, 256, 200, 64], 128, Control::Fast)
}
#[test]
fn g80_i32_grouped_api_two_waves_e3() {
    grouped_api_case32(&[300, 512, 257], 192, Control::Fast);
    grouped_api_case32(&[512, 300, 400], 64, Control::Slow)
}
#[test]
fn g80_i32_grouped_api_e3_k2560() {
    grouped_api_case32(&[64, 160, 200], 2560, Control::Fast)
}

// ---------------------------------------------------------------- simulator timing, int8 vs I32 (functional ticks, NOT silicon cycles)
/// Same operands, same shape and control: int8 (shift 12) and I32 on fresh `Config`s. Both are checked exact; prints ticks per
/// wave, core busy / lock_wait and the I32 - int8 deltas. The simulator's tick is a functional event step: the numbers say
/// nothing about hardware clock cycles, only how the modelled lock / DMA protocol behaves.
fn timing32(m: usize, k: usize, ctl: Control) {
    let (kc, waves) = (k / 64, m.div_ceil(WAVE_M));
    let (a, b) = random(m, k, SEEDS[0]);
    let what = format!("G80 {ctl:?} {m}x{N}x{k}");
    let d8 = design(m, k, ctl, 12);
    let d32 = design32(m, k, ctl);
    let mut s8 = fresh(&d8);
    let [a_host, b_host] = d8.pack_in(&a, &b);
    let mut args = vec![a_host, b_host, vec![POISON; d8.args[2].bytes]];
    let (t0, ticks0, (busy0, wait0)) = (traffic(&s8), s8.ticks, core_sum(&s8));
    s8.submit(&d8.insts, &mut args)
        .unwrap_or_else(|e| panic!("{what} int8: submit failed: {e}"));
    let (t8, (busy1, wait1)) = (s8.ticks - ticks0, core_sum(&s8));
    let (b8, w8) = (busy1 - busy0, wait1 - wait0);
    assert_eq!(
        d8.unpack_out(&args[2]),
        want(&a, &b, m, k, 12),
        "{what}: int8 exact"
    );
    assert_traffic(&t0, &traffic(&s8), 1, waves, kc, &what);
    let r32 = once32(&d32, &format!("{what} timing"), m, k, &a, &b);
    assert!(t8 > 0 && r32.ticks > 0, "{what}: both submits must run");
    assert_eq!(
        r32.written,
        4 * d8.host_bytes_written(),
        "{what}: I32 writes 4x the int8 C bytes"
    );
    assert_eq!(r32.read, d8.host_bytes_read(), "{what}: same input bytes");
    let (p8, p32) = (t8 as f64 / waves as f64, r32.ticks as f64 / waves as f64);
    eprintln!("MEASURED TIMING (simulator functional ticks, NOT silicon cycles) {what} waves={waves}: int8 {t8} ticks ({p8:.1}/wave) core busy={b8} lock_wait={w8}; \
        I32 {} ticks ({p32:.1}/wave) core busy={} lock_wait={}; delta I32-int8 {:+} ticks ({:+.1}/wave) busy {:+} lock_wait {:+}; C bytes int8 {} I32 {}",
        r32.ticks,r32.busy,r32.wait,r32.ticks as i64-t8 as i64,p32-p8,r32.busy as i64-b8 as i64,r32.wait as i64-w8 as i64,d8.host_bytes_written(),r32.written);
}
#[test]
fn g80_i32_timing_vs_int8_k64() {
    for m in [512, 768] {
        for ctl in BOTH {
            timing32(m, 64, ctl)
        }
    }
}
#[test]
fn g80_i32_timing_vs_int8_k2560() {
    timing32(64, 2560, Control::Fast);
    timing32(512, 2560, Control::Fast)
}

// ---------------------------------------------------------------- odd / even mw reuse and estimate (I32)
/// full -> lean -> full -> lean and every forced requeue (S2MM4, MM2S0, both) for M = 200 (mw 1, odd) and M = 300 (mw 2, even)
/// at kc 1 and 3: the finite C tasks (`repeat = mw`) leave no wave-parity state.
#[test]
fn g80_i32_lean_reuse_odd_and_even_mw_k64_k192() {
    for (m, k) in [(200, 64), (200, 192), (300, 64), (300, 192)] {
        for ctl in BOTH {
            lean_interleave32(m, k, ctl);
            forced_requeue32(m, k, ctl)
        }
    }
}
/// Model numbers, not measurements: the I32 hand-off is `6P` per wave plus the final tail `P` with `P = 20480 / 4 = 5120`
/// (`output_dma_cycles = (6*waves + 1) * P`); int8 Fast is `(663 + 2560) * waves`.
#[test]
fn g80_i32_estimate_drain_model_vs_int8() {
    for waves in 1..=3usize {
        let m = waves * WAVE_M;
        let (e32, e8) = (
            design32(m, 128, Control::Fast).estimate(1_000_000_000),
            design(m, 128, Control::Fast, 12).estimate(1_000_000_000),
        );
        assert_eq!(
            e32.output_dma_cycles,
            (6 * waves as u64 + 1) * 5120,
            "I32 waves {waves}"
        );
        assert_eq!(
            e8.output_dma_cycles,
            (663 + 2560) * waves as u64,
            "int8 waves {waves}"
        );
        assert!(e32.estimated_cycles>e8.estimated_cycles,"waves {waves}: the serial single-slot I32 hand-off must cost more than int8 in the model");
        eprintln!("MODEL (estimate, not measured) waves {waves}: I32 drain {} cycles, int8 {} cycles; estimated total I32 {} vs int8 {}",e32.output_dma_cycles,e8.output_dma_cycles,e32.estimated_cycles,e8.estimated_cycles);
    }
}

// ---------------------------------------------------------------- integer direct BF16
fn bf16_scales(count: usize, seed: usize) -> Vec<(u16, i16)> {
    (0..count)
        .map(|i| {
            if (i + seed) % 17 == 0 {
                (0, 0)
            } else {
                (
                    (16384 + (i * 7919 + seed * 1237) % 16384) as u16,
                    [-163, -110, -90, -70, -35, -18, 0, 45, 110][(i + seed) % 9],
                )
            }
        })
        .collect()
}

fn submit_bf16(
    sim: &mut Config,
    d: &ArrayDesign,
    insts: &[u8],
    m: usize,
    k: usize,
    a: &[i8],
    b: &[i8],
    ds: &[(u16, i16)],
    ss: &[(u16, i16)],
) -> usize {
    use peacemaker_npu::kernels::{
        gemm_g80::{pack_a_bf16, pack_b_bf16},
        w8a8_epi::bf16_direct,
    };
    let ah = pack_a_bf16(d, a, ds);
    let bh = pack_b_bf16(d, b, ss);
    let before_inputs = [signature(&ah), signature(&bh)];
    let mut args = vec![ah, bh, vec![POISON; d.args[2].bytes]];
    let before = traffic(sim);
    let t0 = sim.ticks;
    sim.submit(insts, &mut args).unwrap();
    assert_eq!([signature(&args[0]), signature(&args[1])], before_inputs);
    let raw = cpu_reference(a, b, m, N, k);
    let want: Vec<_> = raw
        .iter()
        .enumerate()
        .map(|(at, &i)| {
            let (sm, se) = ss[at % N];
            let (dm, de) = ds[at / N];
            i32::from(bf16_direct(i, sm, se, dm, de))
        })
        .collect();
    let got = d.unpack_out(&args[2]);
    if let Some(at) = got.iter().zip(&want).position(|(g, w)| g != w) {
        panic!(
            "BF16 m={m} k={k} at ({},{}): I={} S={:?} D={:?} got {:04x} want {:04x}",
            at / N,
            at % N,
            raw[at],
            ss[at % N],
            ds[at / N],
            got[at],
            want[at]
        );
    }
    // Rebuild the complete native arena, including every padded token.
    let mut arena = vec![0u8; d.args[2].bytes];
    for ((col, row), ranges) in d.output_tile_ranges() {
        let rho = row as usize - 2;
        for (wave, range) in ranges.iter().enumerate() {
            assert_eq!(range.len(), 20480);
            let (m0, n0) = (
                wave * 256 + rho / 2 * 128,
                (2 * col as usize + rho % 2) * 80,
            );
            for mb in 0..16 {
                for nb in 0..10 {
                    for i in 0..8 {
                        for j in 0..8 {
                            let token = m0 + mb * 8 + i;
                            let feature = n0 + nb * 8 + j;
                            let bits = if token < m {
                                want[token * N + feature] as u16
                            } else {
                                0
                            };
                            let off = range.start + ((mb * 10 + nb) * 64 + i * 8 + j) * 2;
                            arena[off..off + 2].copy_from_slice(&bits.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    assert_eq!(args[2], arena, "BF16 complete poisoned arena");
    let (waves, kc) = (m.div_ceil(256), k / 64);
    let sizes = [
        4 * waves * kc * 4608,
        8 * kc * 2 * 5504,
        8 * waves * 4 * 20480,
    ];
    assert_eq!(d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(), sizes);
    let mut totals = [0u64; 2];
    for (&(col, dir, ch, was), (_, _, _, now)) in before.iter().zip(traffic(sim)) {
        let bytes = now - was;
        let expected = match (dir, ch) {
            (Direction::Mm2s, 0) => kc * 2 * 5504,
            (Direction::Mm2s, 1) => {
                if col < 4 {
                    waves * kc * 4608
                } else {
                    0
                }
            }
            (Direction::S2mm, 0 | 1) => waves * 2 * 20480,
            _ => panic!("unexpected shim channel"),
        };
        assert_eq!(bytes, expected as u64);
        totals[usize::from(dir == Direction::S2mm)] += bytes;
    }
    assert_eq!(totals, [d.host_bytes_read(), d.host_bytes_written()]);
    let ticks = sim.ticks - t0;
    eprintln!(
        "BF16 {m}x{N}x{k}: {ticks} simulator-modelled cycles; read={} write={}",
        totals[0], totals[1]
    );
    ticks
}

#[test]
fn g80_bf16_one_and_multiple_waves_reuse_changed_operands() {
    for ctl in BOTH {
        for m in [200, 300] {
            let k = 64;
            let d = design_g80(m, N, k, Epilogue::Bf16, ctl);
            let mut sim = fresh(&d);
            let lean = d.lean_insts().unwrap();
            for seed in 0..3 {
                let (a, b) = random(m, k, SEEDS[seed]);
                submit_bf16(
                    &mut sim,
                    &d,
                    if seed == 0 { &d.insts } else { &lean },
                    m,
                    k,
                    &a,
                    &b,
                    &bf16_scales(m, seed),
                    &bf16_scales(N, seed + 3),
                );
            }
        }
    }
}

#[test]
fn g80_bf16_exact_ties_carries_subnormals_and_zero_scales() {
    let (m, k) = (256, 64);
    let d = design_g80(m, N, k, Epilogue::Bf16, Control::Fast);
    let mut a = vec![0i8; m * k];
    let mut b = vec![0i8; k * N];
    // I = 257,259,511 and their negatives. With Sm=Dm=2^14 and
    // Se+De=-36 these are exact BF16 halfway points (even/odd/carry).
    for token in 0..m {
        let total: i32 = [257, 259, 511, -257, -259, -511, 0][token % 7];
        let magnitude = total.unsigned_abs() as usize;
        for at in 0..5 {
            a[token * k + at] = (magnitude.saturating_sub(at * 127).min(127) as i8)
                * if total < 0 { -1 } else { 1 };
        }
    }
    for kk in 0..5 {
        b[kk * N..(kk + 1) * N].fill(1);
    }
    let ds = vec![(16384, -18); m];
    let mut ss = vec![(16384, -18); N];
    // The same exact ties at the subnormal grid, plus overflow and Sm=0.
    for (feature, exponent) in [(3, -144), (4, -143), (5, 109)] {
        ss[feature] = (16384, exponent);
    }
    ss[6] = (0, 0);
    let mut sim = fresh(&d);
    submit_bf16(&mut sim, &d, &d.insts, m, k, &a, &b, &ds, &ss);
}

#[test]
fn g80_bf16_k2560_signed_extremes_and_modelled_cycles() {
    let (m, k) = (256, 2560);
    let (a, b) = extremes(m, k, 0);
    let ds = bf16_scales(m, 1);
    let ss = bf16_scales(N, 2);
    let bf = design_g80(m, N, k, Epilogue::Bf16, Control::Fast);
    let ticks_bf = submit_bf16(&mut fresh(&bf), &bf, &bf.insts, m, k, &a, &b, &ds, &ss);
    let int8 = design(m, k, Control::Fast, 12);
    let ticks_i8 = once(&int8, &case("BF16 cycle baseline int8", m, k, 12), &a, &b).ticks;
    let i32 = design32(m, k, Control::Fast);
    let ticks_i32 = submit32(
        &mut fresh(&i32),
        &i32,
        &i32.insts,
        "BF16 cycle baseline I32",
        m,
        k,
        &a,
        &b,
        POISON,
    )
    .ticks;
    eprintln!(
        "MODELLED wave cycles int8={ticks_i8} I32={ticks_i32} BF16={ticks_bf}; \
        scaled 33.1 TOPS: int8=33.1 I32={:.4} BF16={:.4}; arithmetic group={} cycles/32",
        33.1 * ticks_i8 as f64 / ticks_i32 as f64,
        33.1 * ticks_i8 as f64 / ticks_bf as f64,
        peacemaker_npu::kernels::gemm_core_g80::bf16_group_cycles()
    );
}

#[test]
fn g80_bf16_descriptor_bounds_and_8192_host_bytes() {
    let d = design_g80(8192, N, 2560, Epilogue::Bf16, Control::Fast);
    let ops = parse_txn(&d.insts).unwrap();
    assert_bank_aware(&ops, "BF16");
    assert_c_queues(&ops, 32, true, "BF16");
    for col in 0..8u32 {
        for r in 0..4 {
            let prod = mem_bd_of(&ops, col, PROD_BD[r][0], "BF16");
            let drain = mem_bd_of(&ops, col, DRAIN_BD0[r / 2] + (r % 2) as u32, "BF16");
            assert_eq!(prod.len, 5120);
            assert_eq!(prod.addr, 0x80000 + 0x4000 + (r / 2 * 20480) as u32);
            assert_eq!(drain.addr, prod.addr);
            assert_eq!(drain.rel, Some((72 + 2 * (r as u32 ^ 1), 1)));
            assert_eq!(prod.next, None);
        }
        let last_b = mem_bd_of(&ops, col, 5, "BF16");
        assert!(last_b.addr + 40 * last_b.len * 4 <= 0x100000);
    }
    let bytes = d.host_bytes_read() + d.host_bytes_written();
    let bkop = bytes as f64 * 1000.0 / d.useful_ops() as f64;
    assert!(bkop <= 0.90, "{bkop}");
    eprintln!(
        "BF16 8192x1280x2560 host read={} write={} total={bytes}, B/kop={bkop:.9}",
        d.host_bytes_read(),
        d.host_bytes_written()
    );
}

#[test]
fn g80_bf16_pack_near_i26_j56_and_random_grid() {
    use peacemaker_npu::kernels::{gemm_core_g80::bf16_pack_debug_program, w8a8_epi::bf16_direct};
    let (program, offsets) = bf16_pack_debug_program();
    let decoder = peacemaker_npu::sim::decode::PreparedDecoder::new(&program).unwrap();
    let mut state = 0x4781deabu32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for iteration in 0..128 {
        let mut tile = peacemaker_npu::sim::Tile::new(program.clone()).unwrap();
        let mut wants = [0u16; 32];
        for lane in 0..32 {
            let i = if lane < 2 {
                if lane == 0 {
                    (1 << 26) - 1
                } else {
                    1 - (1 << 26)
                }
            } else {
                (next() as i32) >> 5
            };
            let sm = if lane < 2 {
                32767
            } else if lane == 2 {
                0
            } else {
                (16384 + (next() & 16383)) as u16
            };
            let dm = if lane < 2 {
                32767
            } else if lane == 3 {
                0
            } else {
                (16384 + (next() & 16383)) as u16
            };
            let se = (next() % 278) as i16 - 163;
            let de = (next() % 278) as i16 - 163;
            wants[lane] = bf16_direct(i, sm, se, dm, de);
            tile.memory[4 * lane..4 * lane + 4].copy_from_slice(&i.to_le_bytes());
            for (off, bits) in offsets.into_iter().zip([dm, de as u16, sm, se as u16]) {
                let off = off as usize + 2 * lane;
                tile.memory[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let result = tile.run(&decoder, 100_000);
        assert!(
            matches!(result, Ok(peacemaker_npu::sim::Step::Done)),
            "{result:?}"
        );
        for (lane, want) in wants.into_iter().enumerate() {
            let off = 0x5000 + 2 * lane;
            let got = u16::from_le_bytes(tile.memory[off..off + 2].try_into().unwrap());
            assert_eq!(got, want, "BF16 pack iteration={iteration} lane={lane}");
        }
    }
}

#[test]
fn g80_bf16_multiple_k2560_waves_reuse() {
    let (m, k) = (300, 2560);
    let d = design_g80(m, N, k, Epilogue::Bf16, Control::Fast);
    let mut sim = fresh(&d);
    let lean = d.lean_insts().unwrap();
    for seed in 0..2 {
        let (a, b) = pattern_b(m, k, seed != 0);
        submit_bf16(
            &mut sim,
            &d,
            if seed == 0 { &d.insts } else { &lean },
            m,
            k,
            &a,
            &b,
            &bf16_scales(m, seed),
            &bf16_scales(N, seed + 3),
        );
    }
}

#[test]
fn g80_bf16_rowmajor_poison_traffic_cycles_and_reuse() {
    use peacemaker_npu::kernels::gemm_g80::{
        design_g80_bf16_layout, pack_a_bf16, pack_b_bf16, Bf16Layout,
    };
    use peacemaker_npu::kernels::w8a8_epi::bf16_direct;
    for (m, k, pitch, col0) in [14080, 16640]
        .into_iter()
        .flat_map(|pitch| [200, 300, 1545, 1801].map(move |m| (m, 64, pitch, 1280)))
        .chain([(200, 2560, 14080, 2560), (300, 2560, 2624, 32)])
    {
        let native = design_g80(m, N, k, Epilogue::Bf16, Control::Fast);
        let d = design_g80_bf16_layout(
            m,
            N,
            k,
            Epilogue::Bf16,
            Control::Fast,
            Bf16Layout::RowMajor {
                pitch_elems: pitch,
                col0_elems: col0,
            },
        );
        assert_eq!(d.host_bytes_written(), native.host_bytes_written());
        assert_eq!(
            d.estimate(1_000_000_000).estimated_cycles,
            native.estimate(1_000_000_000).estimated_cycles
        );
        let mut sim = fresh(&d);
        let lean = d.lean_insts().unwrap();
        for seed in 0..2 {
            let (a, b) = pattern_b(m, k, seed != 0);
            let ds = bf16_scales(m, seed);
            let ss = bf16_scales(N, seed + 3);
            let raw = cpu_reference(&a, &b, m, N, k);
            let mut expected = vec![POISON; d.args[2].bytes];
            for row in 0..d.waves() * WAVE_M {
                for col in 0..N {
                    let bits = if row < m {
                        let (sm, se) = ss[col];
                        let (dm, de) = ds[row];
                        bf16_direct(raw[row * N + col], sm, se, dm, de)
                    } else {
                        0
                    };
                    let off = 2 * (row * pitch + col0 + col);
                    expected[off..off + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
            let mut args = vec![
                pack_a_bf16(&d, &a, &ds),
                pack_b_bf16(&d, &b, &ss),
                vec![POISON; d.args[2].bytes],
            ];
            let before_inputs = [signature(&args[0]), signature(&args[1])];
            let before = traffic(&sim);
            let start = sim.ticks;
            sim.submit(if seed == 0 { &d.insts } else { &lean }, &mut args)
                .unwrap();
            assert_eq!(
                args[2], expected,
                "rowmajor complete poison arena M={m} K={k} seed={seed}"
            );
            assert_eq!([signature(&args[0]), signature(&args[1])], before_inputs);
            for (&(col, dir, ch, was), (_, _, _, now)) in before.iter().zip(traffic(&sim)) {
                let expected = match (dir, ch) {
                    (Direction::Mm2s, 0) => k / 64 * 2 * 5504,
                    (Direction::Mm2s, 1) => {
                        if col < 4 {
                            d.waves() * k / 64 * 4608
                        } else {
                            0
                        }
                    }
                    (Direction::S2mm, 0 | 1) => d.waves() * 2 * 20480,
                    _ => panic!("unexpected shim channel"),
                };
                assert_eq!(now - was, expected as u64);
            }
            if seed == 0 {
                let tile_ticks = submit_bf16(
                    &mut fresh(&native),
                    &native,
                    &native.insts,
                    m,
                    k,
                    &a,
                    &b,
                    &ds,
                    &ss,
                );
                eprintln!(
                    "rowmajor {m}x{N}x{k} pitch={pitch}: {} modelled ticks vs native {tile_ticks}",
                    sim.ticks - start
                );
            }
        }
    }
}

#[test]
fn g80_bf16_rowmajor_adjacent_panels_one_design_base_offset() {
    use peacemaker_npu::kernels::gemm_g80::{
        design_g80_bf16_layout, pack_a_bf16, pack_b_bf16, Bf16Layout,
    };
    use peacemaker_npu::kernels::w8a8_epi::bf16_direct;
    for pitch in [14080, 16640] {
        let (m, k, col0) = (2041, 64, 32);
        let d = design_g80_bf16_layout(
            m,
            N,
            k,
            Epilogue::Bf16,
            Control::Fast,
            Bf16Layout::RowMajor {
                pitch_elems: pitch,
                col0_elems: 0,
            },
        );
        let (a, b0) = pattern_b(m, k, false);
        let (_, b1) = pattern_b(m, k, true);
        let ds = bf16_scales(m, 0);
        let mut ah = Vec::new();
        let mut bh = Vec::new();
        let mut expected = vec![POISON; d.args[2].bytes + GAP];
        let mut txn = Txn::aie2p_8col();
        for (panel, b) in [&b0, &b1].into_iter().enumerate() {
            let ss = bf16_scales(N, panel + 3);
            let base = [
                ah.len() as u64,
                bh.len() as u64,
                (2 * (col0 + panel * N)) as u64,
            ];
            ah.extend(pack_a_bf16(&d, &a, &ds));
            bh.extend(pack_b_bf16(&d, b, &ss));
            if panel == 0 {
                d.append_run_body(&mut txn, base);
            } else {
                d.append_lean_run_body_same_design(&mut txn, base, &[])
                    .unwrap();
            }
            let raw = cpu_reference(&a, b, m, N, k);
            for row in 0..d.waves() * WAVE_M {
                for col in 0..N {
                    let bits = if row < m {
                        let (sm, se) = ss[col];
                        let (dm, de) = ds[row];
                        bf16_direct(raw[row * N + col], sm, se, dm, de)
                    } else {
                        0
                    };
                    let off = 2 * (row * pitch + col0 + panel * N + col);
                    expected[off..off + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
        }
        let mut args = vec![ah, bh, vec![POISON; expected.len()]];
        let mut sim = fresh(&d);
        sim.submit(&txn.to_bytes(), &mut args).unwrap();
        assert_eq!(args[2], expected, "adjacent panels and trailing guard");
        let read: u64 = traffic(&sim)
            .iter()
            .filter(|s| s.1 == Direction::Mm2s)
            .map(|s| s.3)
            .sum();
        let written: u64 = traffic(&sim)
            .iter()
            .filter(|s| s.1 == Direction::S2mm)
            .map(|s| s.3)
            .sum();
        assert_eq!(
            [read, written],
            [2 * d.host_bytes_read(), 2 * d.host_bytes_written()]
        );
    }
}

#[test]
fn g80_bf16_rowmajor_builder_bounds() {
    use peacemaker_npu::kernels::gemm_g80::build_panel_program_rowmajor;
    for (rows, pitch, col0) in [
        (0, 16640, 0),
        (65537, 16640, 0),
        (256, 1278, 0),
        (256, 16641, 0),
        (256, 16640, 1),
        (256, 16640, 16640),
    ] {
        assert!(build_panel_program_rowmajor(rows, pitch, col0).is_err());
    }
    for pitch in [14080, 16640] {
        let d = build_panel_program_rowmajor(65536, pitch, 0).unwrap();
        assert_eq!(d.args[2].bytes, 65536 * pitch * 2);
        assert_eq!(d.waves(), 256);
    }
}

/// Exercise the maximal 256-wave deployment with one output allocation, not
/// a second multi-GiB expected arena. Constant K rows give an independent
/// exact dot oracle while varying every token/feature scale and wave's data.
#[test]
fn g80_bf16_rowmajor_256_waves_both_pitches() {
    use peacemaker_npu::kernels::gemm_g80::{
        design_g80_bf16_layout, pack_a_bf16, pack_b_bf16, Bf16Layout,
    };
    use peacemaker_npu::kernels::w8a8_epi::bf16_direct;
    let (m, k, col0) = (65529, 64, 1280);
    let av = |row: usize| (row % 127) as i32 - 63;
    let bv = |col: usize| (col % 31) as i32 - 15;
    let a: Vec<i8> = (0..m * k).map(|i| av(i / k) as i8).collect();
    let b: Vec<i8> = (0..k * N).map(|i| bv(i % N) as i8).collect();
    let ds = bf16_scales(m, 2);
    let ss = bf16_scales(N, 5);
    for pitch in [14080, 16640] {
        let d = design_g80_bf16_layout(
            m,
            N,
            k,
            Epilogue::Bf16,
            Control::Fast,
            Bf16Layout::RowMajor {
                pitch_elems: pitch,
                col0_elems: col0,
            },
        );
        let mut sim = fresh(&d);
        let mut args = vec![
            pack_a_bf16(&d, &a, &ds),
            pack_b_bf16(&d, &b, &ss),
            vec![POISON; d.args[2].bytes + GAP],
        ];
        let inputs = [signature(&args[0]), signature(&args[1])];
        let before = traffic(&sim);
        sim.submit(&d.insts, &mut args).unwrap();
        for row in 0..65536 {
            let out = &args[2][2 * row * pitch..2 * (row + 1) * pitch];
            assert!(out[..2 * col0].iter().all(|&v| v == POISON));
            assert!(out[2 * (col0 + N)..].iter().all(|&v| v == POISON));
            for col in 0..N {
                let want = if row < m {
                    let (dm, de) = ds[row];
                    let (sm, se) = ss[col];
                    bf16_direct(k as i32 * av(row) * bv(col), sm, se, dm, de)
                } else {
                    0
                };
                let at = 2 * (col0 + col);
                assert_eq!(
                    u16::from_le_bytes(out[at..at + 2].try_into().unwrap()),
                    want,
                    "256 waves pitch={pitch} row={row} col={col}"
                );
            }
        }
        assert!(args[2][d.args[2].bytes..].iter().all(|&v| v == POISON));
        assert_eq!([signature(&args[0]), signature(&args[1])], inputs);
        for (&(col, dir, ch, was), (_, _, _, now)) in before.iter().zip(traffic(&sim)) {
            let bytes = match (dir, ch) {
                (Direction::Mm2s, 0) => 2 * 5504,
                (Direction::Mm2s, 1) => {
                    if col < 4 {
                        256 * 4608
                    } else {
                        0
                    }
                }
                (Direction::S2mm, 0 | 1) => 256 * 2 * 20480,
                _ => panic!("unexpected shim channel"),
            };
            assert_eq!(now - was, bytes as u64);
        }
    }
}

#[test]
fn g80_bf16_rowmajor_batch_descriptor_budget_and_instruction_counts() {
    use peacemaker_npu::kernels::gemm_g80::{design_g80_bf16_layout, Bf16Layout};
    let ids = [[1, 4, 6, 7, 8, 9, 10], [3, 5, 11, 12, 13, 14, 15]];
    for pitch in [14080, 16640] {
        for waves in [1usize, 2, 7, 8, 32, 256] {
            let d = design_g80_bf16_layout(
                waves * WAVE_M,
                N,
                64,
                Epilogue::Bf16,
                Control::Fast,
                Bf16Layout::RowMajor {
                    pitch_elems: pitch,
                    col0_elems: 1280,
                },
            );
            let native = design_g80(waves * WAVE_M, N, 64, Epilogue::Bf16, Control::Fast);
            let ops = parse_txn(&d.insts).unwrap();
            let native_ops = parse_txn(&native.insts).unwrap();
            let batches = waves.div_ceil(7);
            let mut c_blocks = 0;
            let mut c_patches = 0;
            let mut c_tasks = 0;
            let mut c_cuts = 0;
            for op in &ops {
                match op {
                    TxnOp::BlockWrite(a, words)
                        if a & 0x1ffffff >= 0x1d000 && a & 0x1ffffff < 0x1d200 =>
                    {
                        let id = ((a & 0x1ffffff) - 0x1d000) / 32;
                        assert!(id < 16, "shared shim BD pool");
                        if ids.iter().flatten().any(|&v| v == id) {
                            c_blocks += 1;
                            assert_eq!(words[0], 10240);
                            assert_eq!(words[3], 40 << 20);
                            assert_eq!(words[4] & 0xfffff, pitch as u32 / 2 - 1);
                            assert_eq!((words[4] >> 20) & 1023, 128);
                            assert_eq!(words[5], (2 << 24) | 39);
                            assert_eq!(words[6], 0, "no mutable iteration current");
                            let (pair, index) = ids
                                .iter()
                                .enumerate()
                                .find_map(|(pair, ids)| {
                                    ids.iter().position(|&v| v == id).map(|index| (pair, index))
                                })
                                .unwrap();
                            let next = if index + 1 < waves.min(7) {
                                (1 << 26) | (ids[pair][index + 1] << 27)
                            } else {
                                0
                            };
                            assert_eq!(words[7], (1 << 25) | next);
                        } else {
                            assert!(id == 0 || id == 2, "A/B only");
                        }
                    }
                    TxnOp::DdrPatch { arg: 2, plus, .. } => {
                        c_patches += 1;
                        let elem = *plus as usize / 2;
                        let row = elem / pitch;
                        let col = elem % pitch;
                        assert!(row < waves * WAVE_M);
                        assert!(row % 128 == 0);
                        assert!((1280..2560).contains(&col) && (col - 1280) % 160 == 0);
                    }
                    TxnOp::Write(a, _) if a & 0x1ffffff == 0x1d204 || a & 0x1ffffff == 0x1d20c => {
                        c_tasks += 1
                    }
                    TxnOp::Write(a, _) if a & 0x1ffffff >= 0x1d000 && a & 0x1ffffff < 0x1d200 => {
                        c_cuts += 1
                    }
                    _ => {}
                }
            }
            assert_eq!(c_blocks, 16 * waves.min(7));
            assert_eq!(c_patches, 16 * waves);
            assert_eq!(c_tasks, 16 * batches);
            let cuts = usize::from(waves > 7 && waves % 7 != 0) * 16;
            assert_eq!(c_cuts, cuts);
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, TxnOp::Sync { .. }))
                    .count(),
                2 * batches
            );
            let row_ops = 16 * waves.min(7) + 16 * waves + 18 * batches + cuts;
            // Native shim C: 16 descriptors, patches, tasks and two SYNCs.
            let old_ops = native_ops.len() - 50 + 82 * waves;
            assert_eq!(ops.len(), native_ops.len() - 50 + row_ops);
            assert!(ops.len() <= old_ops);
            eprintln!("rowmajor pitch={pitch} W={waves}: {} ops / {} B; previous {old_ops} ops; native {} ops / {} B",
                ops.len(), d.insts.len(), native_ops.len(), native.insts.len());
        }
    }
}

// ---------------------------------------------------------------- N512 / K6144 direct BF16 (gate K0)
// `gemm_g80_n512`: the V8 wave (512 rows x 512 columns, 128x64 core tiles, B panel resident) with the integer BF16 epilogue.
// The CPU twin is `bf16_direct` over the exact int32 whole-K sum; golden bytes means the complete padded arena.
use peacemaker_npu::kernels::{
    gemm_g80::Bf16Layout as OutLayout,
    gemm_g80_n512::{
        build_panel_program_n512, design_g80_n512_layout, pack_a_bf16 as out_pack_a,
        pack_b_bf16 as out_pack_b,
    },
    w8a8_epi::bf16_direct as out_twin,
};
const ON: usize = 512;
const OUT_WAVE_M: usize = 512;
const OUT_K: usize = 6144;
const OUT_A_CHUNK: usize = 4608;
const OUT_B_CHUNK: usize = 4352;
const OUT_C_TILE: usize = 16384;

fn out_design(m: usize, k: usize, layout: OutLayout) -> ArrayDesign {
    design_g80_n512_layout(m, ON, k, Epilogue::Bf16, Control::Fast, layout)
}
fn out_rowmajor(pitch: usize, col0: usize) -> OutLayout {
    OutLayout::RowMajor {
        pitch_elems: pitch,
        col0_elems: col0,
    }
}

/// Token scales: every third row has the exact tie scale `Dm = 2^14, De = -18`, some zero scales, the rest on the exponent range.
fn out_token_scales(m: usize, seed: usize) -> Vec<(u16, i16)> {
    (0..m)
        .map(|t| {
            if (t + seed) % 3 == 0 {
                (16384, -18)
            } else if (t + seed) % 17 == 5 {
                (0, 0)
            } else {
                (
                    (16384 + (t * 7919 + seed * 1237) % 16384) as u16,
                    [-163, -110, -90, -70, -35, -18, 0, 45, 110][(t + seed) % 9],
                )
            }
        })
        .collect()
}
fn out_feature_scales(n: usize, seed: usize) -> Vec<(u16, i16)> {
    (0..n)
        .map(|j| {
            if (j + seed) % 2 == 0 {
                (16384, -18)
            } else if (j + seed) % 19 == 3 {
                (0, 0)
            } else {
                (
                    (16384 + (j * 104729 + seed * 7717) % 16384) as u16,
                    [-163, -110, -90, -70, -35, -18, 0, 45, 110][(j * 5 + seed) % 9],
                )
            }
        })
        .collect()
}
/// B column `j` is the constant `OUT_BCOL[j % 8]` down all of K, A row `t` sums to `limb_sum(t)`: the exact int32 result is
/// `OUT_BCOL[j % 8] * limb_sum(t)`, so every limb length (|I| from 1 to 2^26), both signs, exact BF16 ties, carries and
/// all-ones patterns are constructed, not hoped for.
const OUT_BCOL: [i32; 8] = [1, 127, -127, 3, -1, 64, -64, 5];
fn limb_sum(t: usize, k: usize) -> i32 {
    let (idx, sign) = (t % 120, if (t / 120) % 2 == 0 { 1i64 } else { -1 });
    let (l, flavor) = (idx / 6, idx % 6);
    let tie = |extra: i64| {
        if l >= 8 {
            (0x101 + extra) << (l - 8)
        } else {
            (1i64 << l) + 1
        }
    };
    let magnitude: i64 = match flavor {
        0 => 1 << l,
        1 => tie(0),
        2 => {
            if l >= 8 {
                0x1ff << (l - 8)
            } else {
                (1 << (l + 1)) - 1
            }
        }
        3 => tie(2),
        4 => (1 << (l + 1)) - 1,
        _ => ((t as i64 * 2654435761) % (1 << (l + 1)).max(1)) | (1 << l),
    };
    (sign * magnitude).clamp(-127 * k as i64, 127 * k as i64) as i32
}
fn out_limb_fixture(m: usize, k: usize) -> (Vec<i8>, Vec<i8>, Vec<i32>) {
    let mut a = vec![0i8; m * k];
    for t in 0..m {
        let s = limb_sum(t, k) as i64;
        let base = s / k as i64;
        let rem = (s - base * k as i64).unsigned_abs() as usize;
        let step = if s < 0 { -1 } else { 1 };
        for kk in 0..k {
            a[t * k + kk] = (base + if kk < rem { step } else { 0 }) as i8;
        }
    }
    let mut b = vec![0i8; k * ON];
    for kk in 0..k {
        for j in 0..ON {
            b[kk * ON + j] = OUT_BCOL[j % 8] as i8;
        }
    }
    let raw = (0..m * ON)
        .map(|x| OUT_BCOL[(x % ON) % 8] * limb_sum(x / ON, k))
        .collect();
    (a, b, raw)
}
fn out_random(m: usize, k: usize, seed: u32) -> (Vec<i8>, Vec<i8>) {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as i8
    };
    let mut a: Vec<_> = (0..m * k).map(|_| next()).collect();
    let mut b: Vec<_> = (0..k * ON).map(|_| next()).collect();
    a[0] = -128;
    b[0] = -128;
    (a, b)
}
/// Expected complete arena: native tiles are rebuilt tile by tile (padding rows zero), row-major is poison everywhere except the
/// `[col0, col0+512)` window of the padded rows, which holds the twin (padding rows zero).
fn out_expected_arena(
    d: &ArrayDesign,
    layout: OutLayout,
    raw: &[i32],
    ds: &[(u16, i16)],
    ss: &[(u16, i16)],
    m: usize,
    extra: usize,
) -> Vec<u8> {
    let twin = |row: usize, col: usize| {
        if row < m {
            out_twin(
                raw[row * ON + col],
                ss[col].0,
                ss[col].1,
                ds[row].0,
                ds[row].1,
            )
        } else {
            0
        }
    };
    let mut arena = vec![POISON; d.args[2].bytes + extra];
    match layout {
        OutLayout::RowMajor {
            pitch_elems: pitch,
            col0_elems: col0,
        } => {
            for row in 0..d.waves() * OUT_WAVE_M {
                for col in 0..ON {
                    let off = 2 * (row * pitch + col0 + col);
                    arena[off..off + 2].copy_from_slice(&twin(row, col).to_le_bytes());
                }
            }
        }
        OutLayout::NativeTiles => {
            for ((col, prow), ranges) in d.output_tile_ranges() {
                let (c, rho) = (col as usize, prow as usize - 2);
                for (w, range) in ranges.iter().enumerate() {
                    assert_eq!(range.len(), OUT_C_TILE);
                    let (m0, n0) = (
                        w * OUT_WAVE_M + (2 * (rho / 2) + c % 2) * 128,
                        (c / 2) * 128 + (rho % 2) * 64,
                    );
                    for mb in 0..16 {
                        for nb in 0..8 {
                            for i in 0..8 {
                                for j in 0..8 {
                                    let off = range.start + ((mb * 8 + nb) * 64 + i * 8 + j) * 2;
                                    arena[off..off + 2].copy_from_slice(
                                        &twin(m0 + mb * 8 + i, n0 + nb * 8 + j).to_le_bytes(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    arena
}
/// One checked submit of `insts` (full or lean) of `d`; returns the simulator ticks. Golden bytes: the complete arena
/// (poisoned, padding rows zero, nothing outside the window written) equals the `bf16_direct` twin of `raw`.
fn out_submit(
    sim: &mut Config,
    d: &ArrayDesign,
    insts: &[u8],
    a: &[i8],
    b: &[i8],
    raw: &[i32],
    ds: &[(u16, i16)],
    ss: &[(u16, i16)],
    m: usize,
    layout: OutLayout,
    what: &str,
) -> usize {
    let (ah, bh) = (out_pack_a(d, a, ds), out_pack_b(d, b, ss));
    assert_eq!(
        (ah.len(), bh.len()),
        (d.args[0].bytes, d.args[1].bytes),
        "{what}: packed sizes"
    );
    let inputs = [signature(&ah), signature(&bh)];
    let mut args = vec![ah, bh, vec![POISON; d.args[2].bytes]];
    let (before, t0) = (traffic(sim), sim.ticks);
    sim.submit(insts, &mut args)
        .unwrap_or_else(|e| panic!("{what}: submit failed: {e}"));
    let ticks = sim.ticks - t0;
    assert_eq!(
        [signature(&args[0]), signature(&args[1])],
        inputs,
        "{what}: packed inputs changed"
    );
    let want = out_expected_arena(d, layout, raw, ds, ss, m, 0);
    if let Some(at) = first_diff(&args[2], &want) {
        let got = d.unpack_out(&args[2]);
        let (elem, row, col) = (at / 2, 0, 0);
        let _ = (elem, row, col);
        let bad = (0..m * ON).find(|&i| {
            got[i]
                != i32::from(out_twin(
                    raw[i],
                    ss[i % ON].0,
                    ss[i % ON].1,
                    ds[i / ON].0,
                    ds[i / ON].1,
                ))
        });
        panic!(
            "{what}: arena differs at byte {at}; first wrong element {:?} ({})",
            bad.map(|i| (i / ON, i % ON, raw[i], got[i])),
            if bad.is_none() {
                "all live elements match: padding / poison / window violation"
            } else {
                "value mismatch"
            }
        );
    }
    let (waves, kc) = (m.div_ceil(OUT_WAVE_M), d.args[1].bytes / (8 * OUT_B_CHUNK));
    for (&(_, dir, ch, was), (_, _, _, now)) in before.iter().zip(traffic(sim)) {
        let bytes = match (dir, ch) {
            (Direction::Mm2s, 0) => kc * OUT_B_CHUNK,
            (Direction::Mm2s, 1) => waves * kc * OUT_A_CHUNK,
            (Direction::S2mm, 0 | 1) => waves * 2 * OUT_C_TILE,
            _ => panic!("unexpected shim channel"),
        };
        assert_eq!(
            now - was,
            bytes as u64,
            "{what}: shim {dir:?} ch {ch} bytes"
        );
    }
    eprintln!("PASS {what}: {m}x{ON}x{} {ticks} functional ticks (simulator ticks, not hardware cycles), complete arena equal to the bf16_direct twin",kc*64);
    ticks
}

fn out_case(m: usize, k: usize, layout: OutLayout, seed: usize, what: &str) -> usize {
    let d = out_design(m, k, layout);
    let (a, b) = out_random(m, k, SEEDS[seed % 3]);
    let raw = cpu_reference(&a, &b, m, ON, k);
    let (ds, ss) = (out_token_scales(m, seed), out_feature_scales(ON, seed + 3));
    out_submit(
        &mut fresh(&d),
        &d,
        &d.insts,
        &a,
        &b,
        &raw,
        &ds,
        &ss,
        m,
        layout,
        what,
    )
}
#[test]
fn g80_n512_smoke_native_k128_one_wave() {
    out_case(
        512,
        128,
        OutLayout::NativeTiles,
        0,
        "N512 smoke native K128 W1",
    );
}

fn out_limb_case(m: usize, k: usize, layout: OutLayout, seed: usize, what: &str) -> usize {
    let d = out_design(m, k, layout);
    let (a, b, raw) = out_limb_fixture(m, k);
    let (ds, ss) = (out_token_scales(m, seed), out_feature_scales(ON, seed + 3));
    out_submit(
        &mut fresh(&d),
        &d,
        &d.insts,
        &a,
        &b,
        &raw,
        &ds,
        &ss,
        m,
        layout,
        what,
    )
}
#[test]
fn g80_n512_k6144_limb_fixture_one_wave_native() {
    out_limb_case(
        512,
        OUT_K,
        OutLayout::NativeTiles,
        0,
        "N512 limb K6144 W1 native",
    );
}
#[test]
fn g80_n512_k6144_limb_fixture_one_wave_rowmajor_panel4() {
    out_limb_case(
        512,
        OUT_K,
        out_rowmajor(2560, 2048),
        1,
        "N512 limb K6144 W1 rowmajor pitch 2560 col0 2048",
    );
}
#[test]
fn g80_n512_k6144_random_one_wave_rowmajor() {
    out_case(
        512,
        OUT_K,
        out_rowmajor(2560, 512),
        2,
        "N512 random K6144 W1 rowmajor panel1",
    );
}

/// 1, 2 and many waves of the limb fixture at K = 6144 (many = 8192 rows = 16 waves, the H3 shape), both destinations.
#[test]
fn g80_n512_k6144_limb_fixture_two_waves_rowmajor() {
    out_limb_case(
        1024,
        OUT_K,
        out_rowmajor(2560, 0),
        3,
        "N512 limb K6144 W2 rowmajor col0 0",
    );
}
#[test]
fn g80_n512_k6144_limb_fixture_two_waves_native() {
    out_limb_case(
        1024,
        OUT_K,
        OutLayout::NativeTiles,
        4,
        "N512 limb K6144 W2 native",
    );
}
#[test]
fn g80_n512_k6144_limb_fixture_many_waves_rowmajor_8192() {
    out_limb_case(
        8192,
        OUT_K,
        out_rowmajor(2560, 1536),
        5,
        "N512 limb K6144 W16 rowmajor col0 1536",
    );
}
/// Padded rows: 700 rows = 2 waves with 324 padding rows (zero scales), and a single row; random operands, poisoned arena.
#[test]
fn g80_n512_k6144_random_padded_rows_rowmajor() {
    out_case(
        700,
        OUT_K,
        out_rowmajor(2560, 1024),
        6,
        "N512 random K6144 M700 rowmajor",
    );
    out_case(
        1,
        OUT_K,
        out_rowmajor(2562, 2),
        7,
        "N512 random K6144 M1 rowmajor pitch 2562 col0 2",
    );
}
#[test]
fn g80_n512_k6144_random_padded_rows_native() {
    out_case(
        700,
        OUT_K,
        OutLayout::NativeTiles,
        8,
        "N512 random K6144 M700 native",
    );
}
/// qwen3.8-27b dense in-projections: K = 5120 (kc 80: two B groups of 40, credit window 63) at the dense arena pitches.
/// GDN = 33 panels at pitch 16896 (last panel col0 16384), FA = 28 panels at pitch 14336.
const DENSE_K: usize = 5120;
#[test]
fn g80_n512_k5120_limb_fixture_two_waves_dense_gdn_last_panel() {
    out_limb_case(
        1024,
        DENSE_K,
        out_rowmajor(16896, 16384),
        9,
        "N512 limb K5120 W2 rowmajor pitch 16896 col0 16384",
    );
}
#[test]
fn g80_n512_k5120_random_padded_rows_dense_fa() {
    out_case(
        700,
        DENSE_K,
        out_rowmajor(14336, 13824),
        10,
        "N512 random K5120 M700 rowmajor pitch 14336 col0 13824",
    );
}
/// Real qwen3.8-27b tiles (W8 codes of 512 fused in-projection columns, A8 activation rows, grid15-up scales) dumped by the
/// HIPF-161 kit into `$N512_REAL_TILE` (`meta` = "m k pitch col0", `a.i8` [m][k], `b.i8` [k][512], `ds.bin` / `ss.bin`
/// LE u16 Dm/Sm + i16 De/Se pairs). The complete arena must equal the `bf16_direct` twin of the whole-K int32 sums.
#[test]
#[ignore = "needs a real-tile dump in $N512_REAL_TILE"]
fn g80_n512_real_27b_tile() {
    let dir = std::path::PathBuf::from(std::env::var("N512_REAL_TILE").expect("N512_REAL_TILE"));
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let meta: Vec<usize> = String::from_utf8(read("meta"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let [m, k, pitch, col0] = meta[..] else {
        panic!("meta must be `m k pitch col0`")
    };
    let codes = |name: &str| read(name).into_iter().map(|b| b as i8).collect::<Vec<_>>();
    let pairs = |name: &str| {
        read(name)
            .chunks_exact(4)
            .map(|c| {
                (
                    u16::from_le_bytes([c[0], c[1]]),
                    i16::from_le_bytes([c[2], c[3]]),
                )
            })
            .collect::<Vec<_>>()
    };
    let (a, b, ds, ss) = (
        codes("a.i8"),
        codes("b.i8"),
        pairs("ds.bin"),
        pairs("ss.bin"),
    );
    assert_eq!(
        (a.len(), b.len(), ds.len(), ss.len()),
        (m * k, k * ON, m, ON)
    );
    let layout = out_rowmajor(pitch, col0);
    let d = out_design(m, k, layout);
    let raw = cpu_reference(&a, &b, m, ON, k);
    out_submit(
        &mut fresh(&d),
        &d,
        &d.insts,
        &a,
        &b,
        &raw,
        &ds,
        &ss,
        m,
        layout,
        &format!(
            "N512 real 27B tile {} K{k} M{m} pitch {pitch} col0 {col0}",
            dir.display()
        ),
    );
}
/// Short K (kc 1, 2, 3, 63, 64, 65): the B credit window and the two-group B split at every boundary.
#[test]
fn g80_n512_short_k_and_b_group_boundaries() {
    for kc in [1usize, 2, 3, 63, 64, 65, 99] {
        for m in [512usize, 1024] {
            out_case(
                m,
                kc * 64,
                out_rowmajor(2560, 512),
                kc + m,
                &format!("N512 random kc={kc} M{m} rowmajor"),
            );
        }
    }
}
/// Five adjacent 512-column panels of one 2560-pitch arena through ONE design (col0 = 0) with per-panel arena offsets; poison
/// guard before the first and after the last element of the arena.
#[test]
fn g80_n512_k6144_adjacent_panels_one_design_one_buffer() {
    let (m, pitch) = (700, 2560);
    let d = out_design(m, OUT_K, out_rowmajor(pitch, 0));
    let rows = d.waves() * OUT_WAVE_M;
    let mut ah = Vec::new();
    let mut bh = Vec::new();
    let mut expected = vec![POISON; rows * pitch * 2 + GAP];
    let mut txn = Txn::aie2p_8col();
    let (a, _) = out_random(m, OUT_K, SEEDS[0]);
    let ds = out_token_scales(m, 1);
    for panel in 0..5 {
        let (_, b) = out_random(m, OUT_K, SEEDS[1] ^ (panel as u32 + 1));
        let ss = out_feature_scales(ON, panel + 2);
        let base = [ah.len() as u64, bh.len() as u64, (2 * panel * ON) as u64];
        ah.extend(out_pack_a(&d, &a, &ds));
        bh.extend(out_pack_b(&d, &b, &ss));
        d.append_run_body(&mut txn, base);
        let raw = cpu_reference(&a, &b, m, ON, OUT_K);
        for row in 0..rows {
            for col in 0..ON {
                let bits = if row < m {
                    out_twin(
                        raw[row * ON + col],
                        ss[col].0,
                        ss[col].1,
                        ds[row].0,
                        ds[row].1,
                    )
                } else {
                    0
                };
                let off = 2 * (row * pitch + panel * ON + col);
                expected[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
    let mut args = vec![ah, bh, vec![POISON; expected.len()]];
    let mut sim = fresh(&d);
    sim.submit(&txn.to_bytes(), &mut args).unwrap();
    assert_eq!(
        first_diff(&args[2], &expected),
        None,
        "adjacent panels, padded rows and trailing guard"
    );
    let read: u64 = traffic(&sim)
        .iter()
        .filter(|s| s.1 == Direction::Mm2s)
        .map(|s| s.3)
        .sum();
    let written: u64 = traffic(&sim)
        .iter()
        .filter(|s| s.1 == Direction::S2mm)
        .map(|s| s.3)
        .sum();
    assert_eq!(
        [read, written],
        [5 * d.host_bytes_read(), 5 * d.host_bytes_written()]
    );
    eprintln!(
        "PASS N512 5 adjacent panels M{m}: {} ticks total",
        sim.ticks
    );
}
/// Argument sizes, tile ranges tiling the native arena exactly, row-major window ranges, builder bounds.
#[test]
fn g80_n512_layout_ranges_and_builder_bounds() {
    for (m, k) in [
        (512usize, 64usize),
        (512, 6144),
        (1, 6144),
        (1025, 6144),
        (8192, 6144),
    ] {
        let d = out_design(m, k, OutLayout::NativeTiles);
        let (waves, kc) = (m.div_ceil(OUT_WAVE_M), k / 64);
        assert_eq!(d.waves(), waves);
        assert_eq!(d.wave_m(), OUT_WAVE_M);
        let sizes = [
            8 * waves * kc * OUT_A_CHUNK,
            8 * kc * OUT_B_CHUNK,
            8 * waves * 4 * OUT_C_TILE,
        ];
        assert_eq!(d.args.iter().map(|x| x.bytes).collect::<Vec<_>>(), sizes);
        assert_eq!(d.host_bytes_read(), (sizes[0] + sizes[1]) as u64);
        assert_eq!(d.host_bytes_written(), sizes[2] as u64);
        let tiles = d.output_tile_ranges();
        assert_eq!(tiles.len(), 32);
        let mut all: Vec<Range<usize>> = tiles.iter().flat_map(|(_, r)| r.clone()).collect();
        all.sort_by_key(|r| r.start);
        let mut end = 0;
        for r in &all {
            assert_eq!(r.start, end);
            assert_eq!(r.len(), OUT_C_TILE);
            end = r.end
        }
        assert_eq!(end, sizes[2]);
    }
    let d = out_design(700, OUT_K, out_rowmajor(2560, 1024));
    assert_eq!(d.args[2].bytes, 2 * OUT_WAVE_M * 2560 * 2);
    let ranges = d.output_tile_ranges();
    let mut live = vec![0u8; d.args[2].bytes];
    for (_, rs) in &ranges {
        for r in rs {
            assert_eq!(r.len(), 128);
            for b in &mut live[r.clone()] {
                *b += 1
            }
        }
    }
    for row in 0..2 * OUT_WAVE_M {
        for col in 0..2560 {
            let inside = (1024..1536).contains(&col);
            for b in 0..2 {
                assert_eq!(
                    live[2 * (row * 2560 + col) + b],
                    u8::from(inside),
                    "row {row} col {col}"
                );
            }
        }
    }
    for (rows, pitch, col0) in [
        (0, 2560, 0),
        (131073, 2560, 0),
        (512, 2559, 0),
        (512, 2561, 1),
        (512, 2560, 1),
        (512, 2560, 2560),
        (512, 2560, 2049),
    ] {
        assert!(
            build_panel_program_n512(rows, pitch, col0, OUT_K).is_err(),
            "rows {rows} pitch {pitch} col0 {col0}"
        );
    }
    let p = build_panel_program_n512(8192, 2560, 0, OUT_K).unwrap();
    assert_eq!(
        (p.args[0].bytes, p.args[1].bytes, p.args[2].bytes),
        (
            8 * 16 * 96 * OUT_A_CHUNK,
            8 * 96 * OUT_B_CHUNK,
            8192 * 2560 * 2
        )
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| out_design(
            512,
            OUT_K + 64 * 4,
            OutLayout::NativeTiles
        )))
        .is_err(),
        "K beyond the memtile bound"
    );
    assert!(catch_unwind(AssertUnwindSafe(|| design_g80_n512_layout(
        512,
        1280,
        64,
        Epilogue::Bf16,
        Control::Fast,
        OutLayout::NativeTiles
    )))
    .is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| design_g80_n512_layout(
        512,
        ON,
        64,
        Epilogue::I32,
        Control::Fast,
        OutLayout::NativeTiles
    )))
    .is_err());
}

/// Gate K0 (design round verdict section 2). Simulator-first numbers, never silicon: incremental ticks per 512-row wave at
/// K = 6144 (W = 2 minus W = 1, and the 16-wave mean), the TOPS scaled from the measured 33.1 TOPS int8 wave through the SAME
/// incremental-tick ratio the BF16 docs use (`docs/npu/npu-trunk.md`), bytes per kilo-op of the 8192-row program, and the
/// 128x64 (V8 pair) against the 128x80 (G80) K-loop MAC efficiency from per-chunk tick fits. Gate: golden mismatch (any
/// sub-test above), or alone rate < 16 TOPS, or bytes > 1.4 B/kop kills Option 1 for 0.4.2.
#[test]
fn g80_n512_gate_k0_ticks_per_wave_tops_bytes_and_tile_efficiency() {
    let ticks = |m: usize, k: usize| -> usize {
        let d = out_design(m, k, out_rowmajor(2560, 512));
        let (a, b, raw) = out_limb_fixture(m, k);
        let (ds, ss) = (out_token_scales(m, 1), out_feature_scales(ON, 4));
        out_submit(
            &mut fresh(&d),
            &d,
            &d.insts,
            &a,
            &b,
            &raw,
            &ds,
            &ss,
            m,
            out_rowmajor(2560, 512),
            &format!("K0 {m}x{ON}x{k}"),
        )
    };
    let (t1, t2, t16) = (ticks(512, OUT_K), ticks(1024, OUT_K), ticks(8192, OUT_K));
    let inc = t2 - t1;
    let mean16 = (t16 - t1) as f64 / 15.0;
    let (c1, c2) = (ticks(512, 2048), ticks(1024, 2048));
    let inc32 = c2 - c1;
    let chunk64 = (inc - inc32) as f64 / (96.0 - 32.0);
    let wave_const64 = inc as f64 - 96.0 * chunk64;
    // int8 baseline wave (K2560 N1280, 256 x 1280): the 33.1 TOPS reference of the BF16 docs.
    let i8_ticks = |m: usize| {
        once(
            &design(m, 2560, Control::Fast, 12),
            &case("K0 int8 baseline", m, 2560, 12),
            &extremes(m, 2560, 0).0,
            &extremes(m, 2560, 0).1,
        )
        .ticks
    };
    let i8_inc = i8_ticks(512) - i8_ticks(256);
    // 128x80 BF16 per-chunk fit: kc 8 against 40 at the same two wave counts.
    let g80_ticks = |m: usize, k: usize| {
        let d = design_g80(m, N, k, Epilogue::Bf16, Control::Fast);
        let (a, b) = pattern_b(m, k, false);
        submit_bf16(
            &mut fresh(&d),
            &d,
            &d.insts,
            m,
            k,
            &a,
            &b,
            &bf16_scales(m, 1),
            &bf16_scales(N, 2),
        )
    };
    let g80_inc = |k: usize| g80_ticks(512, k) - g80_ticks(256, k);
    let chunk80 = (g80_inc(2560) - g80_inc(512)) as f64 / (40.0 - 8.0);
    let ops_wave = 2.0 * OUT_WAVE_M as f64 * ON as f64 * OUT_K as f64;
    let i8_ops_wave = 2.0 * 256.0 * 1280.0 * 2560.0;
    let tops = |inc_ticks: f64| 33.1 * (ops_wave / inc_ticks) / (i8_ops_wave / i8_inc as f64);
    let (alone, alone16) = (tops(inc as f64), tops(mean16));
    let (m8192, a_bytes, b_bytes, c_bytes) = (
        8192usize,
        8 * 16 * 96 * OUT_A_CHUNK,
        8 * 96 * OUT_B_CHUNK,
        8192 * ON * 2,
    );
    let bytes = (a_bytes + b_bytes + c_bytes) as f64;
    let kop = 2.0 * m8192 as f64 * ON as f64 * OUT_K as f64 / 1000.0;
    let eff64 = 524288.0 / chunk64 / 512.0;
    let eff80 = 655360.0 / chunk80 / 512.0;
    eprintln!("K0 REPORT ticks: W1={t1} W2={t2} W16={t16}; incremental per wave {inc} (W2-W1) / {mean16:.0} (16-wave mean); \
        K-loop {chunk64:.1} ticks/chunk, per-wave epilogue+drain {wave_const64:.0} ticks");
    eprintln!("K0 REPORT rate: int8 baseline incremental wave {i8_inc} ticks = 33.1 TOPS; N512 K6144 alone {alone:.2} TOPS \
        (16-wave mean {alone16:.2}); gate >= 16");
    eprintln!("K0 REPORT bytes: A {a_bytes} + B {b_bytes} + C {c_bytes} = {bytes} per 8192 rows = {:.4} B/kop; gate <= 1.4",bytes/kop);
    eprintln!("K0 REPORT tile: 128x64 K-loop {chunk64:.1} ticks/chunk = {:.1}% of MAC peak; 128x80 {chunk80:.1} ticks/chunk = {:.1}%; \
        128x64 loses {:.1}% of the 128x80 MAC efficiency (gate: > 15% triggers Option 2)",100.0*eff64,100.0*eff80,100.0*(1.0-eff64/eff80));
    assert!(
        alone >= 16.0,
        "K0: simulated rate alone {alone:.2} TOPS < 16"
    );
    assert!(bytes / kop <= 1.4, "K0: {:.4} B/kop > 1.4", bytes / kop);
    assert!(
        1.0 - eff64 / eff80 <= 0.15,
        "K0: 128x64 tile loses more than 15% MAC efficiency vs 128x80"
    );
}

// ---------------------------------------------------------------- N512 lean submits
/// Full (zero operands) -> lean submits with fresh operands on one `Config`: exact, byte-equal to a fresh full run, and the
/// retained state equals the fresh full state, for the plain lean TXN and every forced-channel set.
fn out_lean_case(
    m: usize,
    k: usize,
    layout: OutLayout,
    forced: &[(u32, DmaDirection, u32)],
    full_between: bool,
    what: &str,
) {
    let d = out_design(m, k, layout);
    let lean = d
        .lean_insts_requeue(forced)
        .unwrap_or_else(|e| panic!("{what}: lean_insts_requeue: {e}"));
    let mut sim = fresh(&d);
    {
        let mut args = zero_args(&d);
        sim.submit(&d.insts, &mut args).unwrap();
    }
    let initial = sim.state_snapshot();
    for (i, seed) in SEEDS.into_iter().enumerate() {
        let (a, b) = out_random(m, k, seed);
        let raw = cpu_reference(&a, &b, m, ON, k);
        let (ds, ss) = (out_token_scales(m, i), out_feature_scales(ON, i + 1));
        let mut full = fresh(&d);
        out_submit(
            &mut full,
            &d,
            &d.insts,
            &a,
            &b,
            &raw,
            &ds,
            &ss,
            m,
            layout,
            &format!("{what} fresh full #{i}"),
        );
        assert!(
            full.state_snapshot() == initial,
            "{what}: fresh full state != primed state"
        );
        let use_full = full_between && i == 1;
        out_submit(
            &mut sim,
            &d,
            if use_full { &d.insts } else { &lean },
            &a,
            &b,
            &raw,
            &ds,
            &ss,
            m,
            layout,
            &format!("{what} {} #{i}", if use_full { "full" } else { "lean" }),
        );
        assert!(
            sim.state_snapshot() == initial,
            "{what}: retained state != fresh full state after #{i}"
        );
    }
    eprintln!(
        "PASS {what}: full {} B lean {} B",
        d.insts.len(),
        lean.len()
    );
}
#[test]
fn g80_n512_lean_exact_and_state_restored() {
    for (m, k) in [
        (512usize, 64usize),
        (512, 192),
        (1024, 128),
        (700, 6144),
        (1024, 4160),
    ] {
        out_lean_case(
            m,
            k,
            out_rowmajor(2560, 512),
            &[],
            false,
            &format!("N512 lean {m}x{ON}x{k} rowmajor"),
        );
    }
    out_lean_case(
        512,
        192,
        OutLayout::NativeTiles,
        &[],
        true,
        "N512 lean 512x512x192 native full->lean interleave",
    );
}
#[test]
fn g80_n512_lean_forced_requeue() {
    for (m, k) in [(512usize, 64usize), (1024, 192)] {
        for (label, extra) in [
            ("B S2MM4", vec![FORCE_B]),
            ("C MM2S0", vec![FORCE_C]),
            ("both", vec![FORCE_B, FORCE_C]),
        ] {
            out_lean_case(
                m,
                k,
                out_rowmajor(2560, 0),
                &extra,
                false,
                &format!("N512 forced {label} {m}x{ON}x{k}"),
            );
        }
    }
}
/// Lean bodies append behind a full body with per-panel arena offsets (the transport / relay pattern): five panels, one buffer.
#[test]
fn g80_n512_lean_panels_after_full_body_one_buffer() {
    let (m, k, pitch) = (512, 192, 2560);
    let d = out_design(m, k, out_rowmajor(pitch, 0));
    let rows = d.waves() * OUT_WAVE_M;
    let (a, _) = out_random(m, k, SEEDS[0]);
    let ds = out_token_scales(m, 1);
    let (mut ah, mut bh, mut expected) =
        (Vec::new(), Vec::new(), vec![POISON; rows * pitch * 2 + GAP]);
    let mut txn = Txn::aie2p_8col();
    for panel in 0..5 {
        let (_, b) = out_random(m, k, SEEDS[1] ^ (panel as u32 + 7));
        let ss = out_feature_scales(ON, panel + 2);
        let base = [ah.len() as u64, bh.len() as u64, (2 * panel * ON) as u64];
        ah.extend(out_pack_a(&d, &a, &ds));
        bh.extend(out_pack_b(&d, &b, &ss));
        if panel == 0 {
            d.append_run_body(&mut txn, base);
        } else {
            d.append_lean_run_body_same_design(&mut txn, base, &[])
                .unwrap();
        }
        let raw = cpu_reference(&a, &b, m, ON, k);
        for row in 0..rows {
            for col in 0..ON {
                let bits = if row < m {
                    out_twin(
                        raw[row * ON + col],
                        ss[col].0,
                        ss[col].1,
                        ds[row].0,
                        ds[row].1,
                    )
                } else {
                    0
                };
                let off = 2 * (row * pitch + panel * ON + col);
                expected[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
    let mut args = vec![ah, bh, vec![POISON; expected.len()]];
    let mut sim = fresh(&d);
    sim.submit(&txn.to_bytes(), &mut args).unwrap();
    assert_eq!(first_diff(&args[2], &expected), None);
    assert!(
        d.append_lean_run_body_b_resident(&mut Txn::aie2p_8col(), [0; 3], &[])
            .is_err(),
        "B-resident lean body is not derived for N512"
    );
}
