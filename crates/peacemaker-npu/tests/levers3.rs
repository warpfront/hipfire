// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Lever 3 proofs: the V10 B-prefix option (`design_v10_prefix`). It reuses the V10 host layout, so every submit must be
//! CPU-exact with unchanged packed inputs and exactly the V10 per-channel shim traffic. The simulator's shim statistics and
//! tick counter are cumulative over the life of a `Config`, so every submit is checked on the per-submit delta.
//!
//! Simulator limitation: `Config` exposes no consumer-delay / asymmetric pacing knob (only `tick_limit` and `ticks`); channel
//! and core stepping order and the finite FIFO / lock back-pressure are fixed, so these tests exercise that one functional
//! schedule (Fast and Slow core control), not arbitrary producer/consumer skew.
use peacemaker_npu::kernels::{
    gemm_array::{self, ArrayDesign},
    gemm_core::{Control, Epilogue},
};
use peacemaker_npu::sim::{config::Config, dma::Direction};
use std::hash::Hasher;
const INT8: Epilogue = Epilogue::Int8 { shift: 12 };
const CONTROLS: [Control; 2] = [Control::Fast, Control::Slow];
const SEEDS: [u32; 3] = [0x12345678, 0x98765432, 0x2468ace1];
/// (m,n,k) with kc*waves = 1, 2, 3, 8, 9, 12, 9 (odd and even, NW = 1, 2, 3, MW = 1..3).
const REUSE_SHAPES: [(usize, usize, usize); 7] = [
    (512, 512, 64),
    (512, 512, 128),
    (1536, 512, 64),
    (1024, 1024, 128),
    (1536, 512, 192),
    (1024, 1536, 128),
    (1536, 1536, 64),
];
fn matrices(m: usize, n: usize, k: usize, seed: u32) -> (Vec<i8>, Vec<i8>) {
    let mut state = seed;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as i8
    };
    let mut a: Vec<_> = (0..m * k).map(|_| random()).collect();
    let mut b: Vec<_> = (0..k * n).map(|_| random()).collect();
    a[0] = -128;
    b[0] = -128;
    (a, b)
}
fn signature(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
type Counters = Vec<(u8, Direction, u8, u64)>;
fn shim_bytes(sim: &Config) -> Counters {
    sim.shim_stats()
        .map(|c| (c.col, c.direction, c.channel, c.dram_bytes()))
        .collect()
}
fn busy(sim: &Config) -> u64 {
    sim.core_stats().map(|c| c.busy).sum()
}
/// One submit on a (possibly reused) `Config`: C equal to `d.reference` (already CPU-exact), unchanged packed inputs, and the
/// exact per-channel and total shim traffic of *this* submit (delta of the cumulative counters). Returns the functional tick count.
fn submit_checked(
    sim: &mut Config,
    d: &ArrayDesign,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
    seed: u32,
) -> usize {
    let (a, b) = matrices(m, n, k, seed);
    let [a_host, b_host] = d.pack_in(&a, &b);
    let before = [signature(&a_host), signature(&b_host)];
    let (kc, mw, nw) = (k / 64, m.div_ceil(512), n.div_ceil(512));
    let waves = mw * nw;
    assert_eq!(d.waves(), waves, "{label}");
    let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
    let (traffic_before, ticks_before, busy_before) = (shim_bytes(sim), sim.ticks, busy(sim));
    sim.submit(&d.insts, &mut args).unwrap();
    let (ticks, core_busy) = (sim.ticks - ticks_before, busy(sim) - busy_before);
    let want = d.reference(&a, &b);
    let got = d.unpack_out(&args[2]);
    assert_eq!(got.len(), want.len(), "{label}");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{label} {m}x{n}x{k} seed {seed:#x}: first mismatch at ({},{}) got {} want {}",
            i / n,
            i % n,
            got[i],
            want[i]
        )
    }
    assert_eq!(
        [signature(&args[0]), signature(&args[1])],
        before,
        "{label} packed inputs changed"
    );
    let a_tiles = mw;
    let (mut read, mut written) = (0u64, 0u64);
    let after = shim_bytes(sim);
    assert_eq!(after.len(), traffic_before.len());
    for (&(col, dir, ch, now), &(col0, dir0, ch0, was)) in after.iter().zip(&traffic_before) {
        assert_eq!((col, dir, ch), (col0, dir0, ch0));
        let delta = now - was;
        let expected = match (dir, ch) {
            (Direction::Mm2s, 0) => nw * kc * 4096,
            (Direction::Mm2s, 1) => a_tiles * kc * 4096,
            (Direction::S2mm, 0) | (Direction::S2mm, 1) => waves * 2 * gemm_array::COUT_BYTES,
            _ => panic!("unexpected shim channel"),
        } as u64;
        assert_eq!(
            delta, expected,
            "{label} {m}x{n}x{k} seed {seed:#x}: shim col {col} {dir:?} ch {ch} per-submit bytes"
        );
        match dir {
            Direction::Mm2s => read += delta,
            Direction::S2mm => written += delta,
        }
    }
    assert_eq!(
        read,
        d.host_bytes_read(),
        "{label} {m}x{n}x{k}: total DDR read bytes"
    );
    assert_eq!(
        written,
        d.host_bytes_written(),
        "{label} {m}x{n}x{k}: total DDR write bytes"
    );
    let expected_ab = (a_tiles * kc * 4096 + nw * kc * 4096) as u64 * 8;
    assert_eq!(read, expected_ab, "{label}: summed read traffic");
    assert_eq!(
        written,
        (waves * 2 * gemm_array::COUT_BYTES) as u64 * 16,
        "{label}: summed write traffic"
    );
    for ((col, row), ranges) in d.output_tile_ranges() {
        assert_eq!(ranges.len(), waves);
        for (w, r) in ranges.iter().enumerate() {
            assert!(
                args[2][r.clone()].iter().any(|&x| x != 0xa5),
                "{label} tile ({col},{row}) wave {w} not written"
            );
        }
    }
    eprintln!("PASS {label} {m}x{n}x{k} seed {seed:#x}: {m}x{n} CPU-exact words; {ticks} functional ticks (not hardware cycles); core busy={core_busy}; \
        expected/observed DRAM read={expected_ab}/{read} write={}/{written} (per-channel MM2S0={} MM2S1={} S2MM0/1={} B per column)",
        waves*2*gemm_array::COUT_BYTES*16,nw*kc*4096,a_tiles*kc*4096,waves*2*gemm_array::COUT_BYTES);
    ticks
}
fn fresh(d: &ArrayDesign) -> Config {
    Config::from_pdi(&d.pdi).unwrap()
}
fn prefix(m: usize, n: usize, k: usize, ctl: Control) -> ArrayDesign {
    gemm_array::design_v10_prefix(m, n, k, INT8, ctl)
}
// ---------------------------------------------------------------- V10 B-prefix
fn prefix_all(m: usize, n: usize, k: usize) {
    for ctl in CONTROLS {
        let d = prefix(m, n, k, ctl);
        let mut sim = fresh(&d);
        let ticks = submit_checked(
            &mut sim,
            &d,
            &format!("V10 B-prefix {ctl:?}"),
            m,
            n,
            k,
            0xc001d00d,
        );
        let v10 = gemm_array::design_v10(m, n, k, INT8, ctl);
        assert_eq!(
            (d.host_bytes_read(), d.host_bytes_written()),
            (v10.host_bytes_read(), v10.host_bytes_written()),
            "host byte totals differ from V10"
        );
        assert_eq!(d.args, v10.args, "argument layout differs from V10");
        eprintln!("PASS V10 B-prefix {ctl:?} {m}x{n}x{k}: {ticks} ticks, host bytes equal V10");
    }
}
#[test]
fn prefix_exact_512_512_64_mw1() {
    prefix_all(512, 512, 64)
}
#[test]
fn prefix_exact_1024_1024_128() {
    prefix_all(1024, 1024, 128)
}
#[test]
fn prefix_exact_1536_1024_192() {
    prefix_all(1536, 1024, 192)
}
#[test]
fn prefix_exact_2048_2560_640() {
    prefix_all(2048, 2560, 640)
}
#[test]
fn prefix_exact_4096_2560_640() {
    prefix_all(4096, 2560, 640)
}
/// Flash-Next routed-expert down at the average row count (M = 160 padded into one 512-row wave, MW = 1).
#[test]
fn prefix_exact_expert_down_m160() {
    prefix_all(160, 2560, 640)
}
/// Same `Config` reused for three submits (the guarded first pass must restart every submit).
#[test]
fn prefix_reuses_context_three_seeds() {
    for (m, n, k) in REUSE_SHAPES {
        for ctl in CONTROLS {
            let d = prefix(m, n, k, ctl);
            let mut sim = fresh(&d);
            for seed in SEEDS {
                submit_checked(
                    &mut sim,
                    &d,
                    &format!("V10 B-prefix {ctl:?}"),
                    m,
                    n,
                    k,
                    seed,
                );
            }
            eprintln!("PASS V10 B-prefix same-context {ctl:?} {m}x{n}x{k} (kc*waves={}) three submits, {} functional ticks total",k/64*d.waves(),sim.ticks);
        }
    }
}
/// Boundary NW = 8, kc = 11 (88 B chunks, kc*(2+NW) = 110 <= 112 slots): 88-chunk prefix fill exceeds one 64-iteration window,
/// so the iteration count must be split and the ready-credit count must stay within its bound; MW = 1 (no unguarded pass)
/// and MW = 2 (one unguarded pass), same `Config` reused.
#[test]
fn prefix_exact_512_4096_704_iteration_wrap_and_credit_bound() {
    for (m, n, k) in [(512, 4096, 704), (1024, 4096, 704)] {
        for ctl in CONTROLS {
            let d = prefix(m, n, k, ctl);
            assert_eq!(k / 64 * n.div_ceil(512), 88);
            let mut sim = fresh(&d);
            for seed in [SEEDS[0], SEEDS[1]] {
                submit_checked(
                    &mut sim,
                    &d,
                    &format!("V10 B-prefix {ctl:?}"),
                    m,
                    n,
                    k,
                    seed,
                );
            }
        }
    }
}
/// The prefix design keeps V10's capacity bound and int8-only epilogue.
#[test]
#[should_panic]
fn prefix_rejects_i32_epilogue() {
    gemm_array::design_v10_prefix(512, 512, 64, Epilogue::I32, Control::Fast);
}
#[test]
#[should_panic]
fn prefix_rejects_oversize_gate_up_shape() {
    gemm_array::design_v10_prefix(4096, 1280, 2560, INT8, Control::Fast);
}
