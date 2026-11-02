// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Lean (slim) per-run TXN of the pair designs in the whole-array simulator.
//!
//! A lean TXN omits the work a full TXN does that is already true after a previous COMPLETED full run of the same design
//! on the same `Config` (core/ring/lock restarts and descriptor rewrites that the previous run left in place), and instead
//! repairs only the state a run advances (memtile channels and BD iterations of the resident V9/V10 designs). Contract
//! proven here:
//!   * after one full run with zero operands, any number of lean submits with other operands produce C bytes equal to the
//!     CPU reference AND equal to a fresh `Config` running the full TXN with the same operands;
//!   * `Config::state_snapshot` (registers/BD iteration words, bindings, every lock, channel queues/engine positions,
//!     FIFOs; no data/counters) after every lean (and full) run equals the snapshot after the initial full run;
//!   * full -> lean -> full -> lean interleaves keep both properties;
//!   * the repair operations are load-bearing: a test-only TXN derived from the valid lean ops with the memtile phase repairs
//!     filtered out is detected (SYNC timeout / wrong C / snapshot mismatch) on shapes that need repair.
//!   * V10 B-prefix designs (finite chunk-group fill / guarded reader BDs, optional lockfree replay) obey the same properties,
//!     including the >63-chunk two-group prefix and forced col 0 S2MM4 / MM2S0 requeue (no finite task or repeat count lost);
//!     omitting one prefix task queue write is detected.
use peacemaker_npu::dma::Direction;
use peacemaker_npu::kernels::{
    gemm_array::{self, ArrayDesign, Variant},
    gemm_core::{Control, Epilogue},
};
use peacemaker_npu::sim::config::{parse_txn, Config, TxnOp};
use std::cell::Cell;

const INT8: Epilogue = Epilogue::Int8 { shift: 12 };
const POISON: u8 = 0xa5;
const SEEDS: [u32; 3] = [0x12345678, 0x98765432, 0x2468ace1];

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
fn design(v: Variant, m: usize, n: usize, k: usize) -> ArrayDesign {
    match v {
        Variant::V9 => gemm_array::design_v9(m, n, k, INT8, Control::Fast),
        Variant::V10 => gemm_array::design_v10(m, n, k, INT8, Control::Fast),
        _ => gemm_array::design_variant(m, n, k, v),
    }
}
fn host_args(d: &ArrayDesign, a: &[i8], b: &[i8]) -> Vec<Vec<u8>> {
    let [ah, bh] = d.pack_in(a, b);
    vec![ah, bh, vec![POISON; d.args[2].bytes]]
}
fn zero_args(d: &ArrayDesign) -> Vec<Vec<u8>> {
    vec![
        vec![0; d.args[0].bytes],
        vec![0; d.args[1].bytes],
        vec![POISON; d.args[2].bytes],
    ]
}
/// A fresh `Config` that has completed one full run with all-zero operands: the state every lean submit assumes.
fn primed(d: &ArrayDesign) -> Config {
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    let mut args = zero_args(d);
    sim.submit(&d.insts, &mut args).unwrap();
    assert!(
        args[2].iter().all(|&x| x == 0),
        "zero operands must give zero C"
    );
    sim
}
/// C bytes of a FRESH `Config` running the full TXN on `args`.
fn fresh_full(d: &ArrayDesign, args: &[Vec<u8>]) -> Vec<u8> {
    let mut args = args.to_vec();
    Config::from_pdi(&d.pdi)
        .unwrap()
        .submit(&d.insts, &mut args)
        .unwrap();
    args.swap_remove(2)
}
fn first_byte_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    (0..a.len()).find(|&i| a[i] != b[i])
}
/// `c` (packed C bytes) must equal the design's CPU reference, the fresh-full bytes, and A/B must be untouched.
fn assert_run(
    d: &ArrayDesign,
    a: &[i8],
    b: &[i8],
    before: &[Vec<u8>],
    args: &[Vec<u8>],
    fresh: &[u8],
    what: &str,
) {
    let (got, want) = (d.unpack_out(&args[2]), d.reference(a, b));
    assert_eq!(got.len(), want.len(), "{what}");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{what}: C != reference at word {i}: got {} want {}",
            got[i], want[i]
        )
    }
    if let Some(i) = first_byte_diff(&args[2], fresh) {
        panic!("{what}: C bytes != fresh-Config full TXN bytes at byte {i}")
    }
    assert!(
        args[0] == before[0] && args[1] == before[1],
        "{what}: operands modified"
    );
}
fn lean_of(d: &ArrayDesign, what: &str) -> Vec<u8> {
    d.lean_insts()
        .unwrap_or_else(|e| panic!("{what}: lean_insts: {e}"))
}

/// One full zero-operand run, snapshot, then three lean submits with different operands on the SAME Config.
fn lean_repeat(v: Variant, m: usize, n: usize, k: usize) {
    lean_repeat_design(&design(v, m, n, k), &format!("{v} {m}x{n}x{k}"), m, n, k)
}
fn lean_repeat_design(d: &ArrayDesign, what: &str, m: usize, n: usize, k: usize) {
    let lean = lean_of(d, what);
    let mut sim = primed(d);
    let initial = sim.state_snapshot();
    for (i, seed) in SEEDS.into_iter().enumerate() {
        let (a, b) = matrices(m, n, k, seed);
        let args0 = host_args(d, &a, &b);
        let fresh = fresh_full(d, &args0);
        let mut args = args0.clone();
        sim.submit(&lean, &mut args).unwrap();
        assert_run(
            d,
            &a,
            &b,
            &args0,
            &args,
            &fresh,
            &format!("{what} lean #{i} seed {seed:#x}"),
        );
        assert!(
            sim.state_snapshot() == initial,
            "{what}: retained state after lean #{i} != state after the initial full run"
        );
    }
    eprintln!("PASS lean x3 {what}: reference-exact + byte-equal to fresh full; state == initial full state; full {} B lean {} B; {} ticks total",d.insts.len(),lean.len(),sim.ticks);
}
/// full -> lean -> full -> lean on one Config, each with its own operands.
fn lean_interleave(v: Variant, m: usize, n: usize, k: usize) {
    lean_interleave_design(&design(v, m, n, k), &format!("{v} {m}x{n}x{k}"), m, n, k)
}
fn lean_interleave_design(d: &ArrayDesign, what: &str, m: usize, n: usize, k: usize) {
    let lean = lean_of(d, what);
    let mut sim = primed(d);
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
        let (a, b) = matrices(m, n, k, seed);
        let args0 = host_args(d, &a, &b);
        let fresh = fresh_full(d, &args0);
        let mut args = args0.clone();
        sim.submit(if is_lean { &lean } else { &d.insts }, &mut args)
            .unwrap();
        let kind = if is_lean { "lean" } else { "full" };
        assert_run(
            d,
            &a,
            &b,
            &args0,
            &args,
            &fresh,
            &format!("{what} interleave #{i} {kind}"),
        );
        assert!(
            sim.state_snapshot() == initial,
            "{what}: retained state after interleave #{i} ({kind}) != initial full state"
        );
    }
    eprintln!("PASS full->lean->full->lean {what}: exact C + byte-equal to fresh full + identical retained state after every submit");
}

/// Wave grids covering mw/nw/kc parities: 1x1 kc1 (odd chunks), 1x1 kc2, 2x1 kc2, 1x2 kc4, plus the periodic 1x2 kc2.
const V9_SHAPES: [(usize, usize, usize); 5] = [
    (512, 512, 64),
    (512, 512, 128),
    (1024, 512, 128),
    (512, 1024, 256),
    (512, 1024, 128),
];
/// V10: 2x1 kc2, odd MW (1x2 kc2) and odd chunks (1x1 kc1).
const V10_SHAPES: [(usize, usize, usize); 3] = [(1024, 512, 128), (512, 1024, 128), (512, 512, 64)];
#[test]
fn v9_lean_three_submits_exact_and_state_restored() {
    for (m, n, k) in V9_SHAPES {
        lean_repeat(Variant::V9, m, n, k)
    }
}
#[test]
fn v9_lean_full_lean_interleave_exact_and_state_restored() {
    for (m, n, k) in V9_SHAPES {
        lean_interleave(Variant::V9, m, n, k)
    }
}
#[test]
fn v10_lean_three_submits_exact_and_state_restored() {
    for (m, n, k) in V10_SHAPES {
        lean_repeat(Variant::V10, m, n, k)
    }
}
#[test]
fn v10_lean_full_lean_interleave_exact_and_state_restored() {
    for (m, n, k) in V10_SHAPES {
        lean_interleave(Variant::V10, m, n, k)
    }
}

// ---------------------------------------------------------------- V10 B prefix (`design_v10_prefix`)
fn prefix(m: usize, n: usize, k: usize, ctl: Control) -> ArrayDesign {
    gemm_array::design_v10_prefix(m, n, k, INT8, ctl)
}
/// Full + 3 lean submits and full -> lean -> full -> lean on prefix designs of each requested control discipline.
fn prefix_lean_all(m: usize, n: usize, k: usize, controls: &[Control]) {
    for &ctl in controls {
        let d = prefix(m, n, k, ctl);
        let what = format!("V10-prefix {ctl:?} {m}x{n}x{k}");
        lean_repeat_design(&d, &what, m, n, k);
        lean_interleave_design(&d, &what, m, n, k);
    }
}
const BOTH: [Control; 2] = [Control::Fast, Control::Slow];
// The shapes of levers3.rs (REUSE_SHAPES and the production / >63-chunk prefix cases). MW = ceil(m/512): 512x512x64 and
// 512x4096x704 are MW1 (no lockfree replay task), 1024x.. MW2, 1536x.. MW3, 2048x.. MW4, 4096x.. MW8. The 704 shapes have
// kc*NW = 88 chunks: two fill / reader groups (64 + 24 chunks) with the B_EMPTY credit window.
#[test]
fn v10_prefix_lean_512_512_64_mw1() {
    prefix_lean_all(512, 512, 64, &BOTH)
}
#[test]
fn v10_prefix_lean_512_512_128() {
    prefix_lean_all(512, 512, 128, &BOTH)
}
#[test]
fn v10_prefix_lean_1536_512_64() {
    prefix_lean_all(1536, 512, 64, &BOTH)
}
#[test]
fn v10_prefix_lean_1024_1024_128() {
    prefix_lean_all(1024, 1024, 128, &BOTH)
}
#[test]
fn v10_prefix_lean_1536_512_192() {
    prefix_lean_all(1536, 512, 192, &BOTH)
}
#[test]
fn v10_prefix_lean_1024_1536_128() {
    prefix_lean_all(1024, 1536, 128, &BOTH)
}
#[test]
fn v10_prefix_lean_1536_1536_64() {
    prefix_lean_all(1536, 1536, 64, &BOTH)
}
#[test]
fn v10_prefix_lean_1536_1024_192() {
    prefix_lean_all(1536, 1024, 192, &BOTH)
}
#[test]
fn v10_prefix_lean_2048_2560_640() {
    prefix_lean_all(2048, 2560, 640, &[Control::Fast])
}
#[test]
fn v10_prefix_lean_4096_2560_640() {
    prefix_lean_all(4096, 2560, 640, &[Control::Fast])
}
#[test]
fn v10_prefix_lean_512_4096_704_mw1_two_groups() {
    prefix_lean_all(512, 4096, 704, &BOTH)
}
#[test]
fn v10_prefix_lean_1024_4096_704_mw2_two_groups() {
    prefix_lean_all(1024, 4096, 704, &[Control::Fast])
}

const FORCE_B: (u32, Direction, u32) = (0, Direction::S2mm, 4);
const FORCE_C: (u32, Direction, u32) = (0, Direction::Mm2s, 0);
/// The persistent ring's POLL / DONE clobber column 0 memtile S2MM4 (B fill) and MM2S0 (C drain): a forced reset + requeue of
/// them on a two-group prefix must still queue BOTH finite fill groups (64 + 24 chunks, repeat counts 64 / 24) and every reader
/// task, so the consumer-visible results are CPU-exact, byte-equal to a fresh full run and leave the initial retained state.
fn prefix_forced_requeue(m: usize, n: usize, k: usize) {
    let d = prefix(m, n, k, Control::Fast);
    let what = format!("V10-prefix {m}x{n}x{k}");
    let plain = lean_of(&d, &what);
    let runs: Vec<_> = SEEDS
        .into_iter()
        .map(|seed| {
            let (a, b) = matrices(m, n, k, seed);
            let args0 = host_args(&d, &a, &b);
            let fresh = fresh_full(&d, &args0);
            (a, b, args0, fresh)
        })
        .collect();
    let sets: [(&str, Vec<(u32, Direction, u32)>); 3] = [
        ("B S2MM4", vec![FORCE_B]),
        ("C MM2S0", vec![FORCE_C]),
        ("B S2MM4 + C MM2S0", vec![FORCE_B, FORCE_C]),
    ];
    for (label, extra) in sets {
        let forced = d
            .lean_insts_requeue(&extra)
            .unwrap_or_else(|e| panic!("{what} forced {label}: lean_insts_requeue: {e}"));
        let mut sim = primed(&d);
        let initial = sim.state_snapshot();
        for (i, (a, b, args0, fresh)) in runs.iter().enumerate() {
            let mut args = args0.clone();
            sim.submit(&forced, &mut args).unwrap();
            assert_run(
                &d,
                a,
                b,
                args0,
                &args,
                fresh,
                &format!("{what} forced {label} #{i}"),
            );
            assert!(
                sim.state_snapshot() == initial,
                "{what} forced {label}: retained state after forced #{i} != initial full state"
            );
        }
        // A plain lean submit directly after forced ones is still exact.
        let (a, b) = matrices(m, n, k, 0xdeadbeef);
        let args0 = host_args(&d, &a, &b);
        let fresh = fresh_full(&d, &args0);
        let mut args = args0.clone();
        sim.submit(&plain, &mut args).unwrap();
        assert_run(
            &d,
            &a,
            &b,
            &args0,
            &args,
            &fresh,
            &format!("{what} plain lean after forced {label}"),
        );
        assert!(
            sim.state_snapshot() == initial,
            "{what} forced {label}: state after the trailing plain lean != initial full state"
        );
        eprintln!("PASS forced requeue {label} {what}: lean {} B forced {} B, exact + byte-equal + state restored",plain.len(),forced.len());
    }
}
#[test]
fn v10_prefix_forced_requeue_512_4096_704_mw1() {
    prefix_forced_requeue(512, 4096, 704)
}
#[test]
fn v10_prefix_forced_requeue_1024_4096_704_mw2() {
    prefix_forced_requeue(1024, 4096, 704)
}

/// Pair A-sharing V8: fast/int8 is the primary combination; i32 and the slow discipline change the core program, the C
/// descriptor size and the token flow, so all four epilogue x control combinations are required to be supported and exact.
#[test]
fn v8_lean_all_epilogues_and_controls() {
    let (m, n) = (512, 512);
    let combos = [
        (INT8, Control::Fast),
        (Epilogue::I32, Control::Fast),
        (INT8, Control::Slow),
        (Epilogue::I32, Control::Slow),
    ];
    for (k, (epi, ctl)) in [64usize, 128]
        .into_iter()
        .flat_map(|k| combos.into_iter().map(move |c| (k, c)))
    {
        let what = format!("V8 {epi:?} {ctl:?} {m}x{n}x{k}");
        let d = gemm_array::design_v8(m, n, k, epi, ctl);
        let lean = lean_of(&d, &what);
        let mut sim = primed(&d);
        let initial = sim.state_snapshot();
        for (i, seed) in SEEDS.into_iter().enumerate() {
            let (a, b) = matrices(m, n, k, seed);
            let args0 = host_args(&d, &a, &b);
            let fresh = fresh_full(&d, &args0);
            let mut args = args0.clone();
            sim.submit(&lean, &mut args).unwrap();
            assert_run(
                &d,
                &a,
                &b,
                &args0,
                &args,
                &fresh,
                &format!("{what} lean #{i}"),
            );
            assert!(
                sim.state_snapshot() == initial,
                "{what}: state after lean #{i} != initial full state"
            );
        }
        eprintln!(
            "PASS lean x3 {what}: full {} B lean {} B",
            d.insts.len(),
            lean.len()
        );
    }
}

/// The lean stream is pair-only: static V5 (configure-once TXN) and the non-pair dynamic V6 are rejected with a message.
#[test]
fn non_pair_variants_lean_rejected() {
    for (v, what) in [(Variant::V5, "V5"), (Variant::V6, "V6")] {
        let d = gemm_array::design_variant(512, 512, 64, v);
        let e = d
            .lean_insts()
            .expect_err(&format!("{what} must not have a lean stream"));
        assert!(!e.is_empty());
        eprintln!("{what} lean rejected: {e}");
    }
}

// ---------------------------------------------------------------- negative: the phase repairs are load-bearing
/// Tile row of an op's register address (`col<<25 | row<<20 | offset`).
fn op_row(op: &TxnOp) -> Option<u32> {
    let a = match *op {
        TxnOp::Write(a, _)
        | TxnOp::BlockWrite(a, _)
        | TxnOp::MaskWrite(a, ..)
        | TxnOp::MaskPoll(a, ..)
        | TxnOp::DdrPatch { address: a, .. } => a,
        TxnOp::Sync { .. } => return None,
    };
    Some((a >> 20) & 31)
}
/// The phase repairs: compute/memtile DMA channel CTRL writes (RESET assert/deassert), compute/memtile lock writes, and memtile
/// BD word 6 mask writes (iteration current). Offsets: aie-rt AIE2P compute (0x1de00 CTRL, 0x1f000 locks) and memtile
/// (0xa0000 BDs of 0x20 B, 0xa0600 CTRL, 0xc0000 locks).
fn is_phase_repair(op: &TxnOp) -> bool {
    let (Some(row), Some(off)) = (op_row(op), op_offset(op)) else {
        return false;
    };
    match *op {
        TxnOp::MaskWrite(..) => match row {
            1 => {
                ((0xa0600..0xa0660).contains(&off) && off % 8 == 0)
                    || ((0xa0000..0xa0600).contains(&off) && off % 0x20 == 0x18)
            }
            r if r >= 2 => (0x1de00..0x1de20).contains(&off) && off % 8 == 0,
            _ => false,
        },
        TxnOp::Write(..) => match row {
            1 => (0xc0000..0xc0400).contains(&off),
            r if r >= 2 => (0x1f000..0x1f100).contains(&off),
            _ => false,
        },
        _ => false,
    }
}
fn op_offset(op: &TxnOp) -> Option<u32> {
    match *op {
        TxnOp::Write(a, _)
        | TxnOp::BlockWrite(a, _)
        | TxnOp::MaskWrite(a, ..)
        | TxnOp::MaskPoll(a, ..)
        | TxnOp::DdrPatch { address: a, .. } => Some(a & 0xfffff),
        TxnOp::Sync { .. } => None,
    }
}
/// What a forced (un-repaired) body did on a Config that satisfies the lean precondition.
#[derive(Debug)]
#[allow(dead_code)]
enum Detected {
    Failed(String),
    WrongC(usize),
    StateMismatch,
}
/// Runs `ops` (a test-only TXN built from valid lean ops, in order) up to three times on a primed Config, with the sim's
/// deadlock limit tightened to a multiple of one full run so a stuck SYNC reports quickly. `None` = it behaved like a lean TXN.
fn run_forced(d: &ArrayDesign, m: usize, n: usize, k: usize, ops: &[TxnOp]) -> Option<Detected> {
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    let mut zero = zero_args(d);
    sim.submit(&d.insts, &mut zero).unwrap();
    let initial = sim.state_snapshot();
    sim.tick_limit = sim.ticks * 2 + 4096;
    for seed in SEEDS {
        let (a, b) = matrices(m, n, k, seed);
        let mut args = host_args(d, &a, &b);
        if let Err(e) = sim.execute(ops, &mut args) {
            return Some(Detected::Failed(e));
        }
        let (got, want) = (d.unpack_out(&args[2]), d.reference(&a, &b));
        if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
            return Some(Detected::WrongC(i));
        }
        if sim.state_snapshot() != initial {
            return Some(Detected::StateMismatch);
        }
    }
    None
}
fn forced_ops(d: &ArrayDesign, drop: &dyn Fn(&TxnOp) -> bool) -> (Vec<TxnOp>, usize) {
    let lean = parse_txn(&d.lean_insts().unwrap()).unwrap();
    let kept: Vec<TxnOp> = lean.iter().filter(|o| !drop(*o)).cloned().collect();
    let removed = lean.len() - kept.len();
    // Shim programming, core control and every token SYNC are retained, so the forced body still starts the run and waits for C.
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
/// Dropping exactly the phase-repair ops of the lean TXN (channel RESET pulses, repair lock writes, memtile BD iteration-current
/// rewrites) while keeping the shim programming, channel task queues, core control and every SYNC must be caught on nonperiodic
/// shapes: their channel/iteration/lock state does not return to its start after one run, so each repair is load-bearing.
/// (A periodic shape needs none of them.)
#[test]
fn omitting_phase_repairs_is_detected_on_nonperiodic_shapes() {
    for (v, m, n, k) in [
        (Variant::V9, 512, 512, 128),
        (Variant::V9, 1024, 512, 128),
        (Variant::V10, 512, 1024, 128),
    ] {
        let d = design(v, m, n, k);
        let (ops, removed) = forced_ops(&d, &is_phase_repair);
        assert!(
            removed > 0,
            "{v} {m}x{n}x{k}: lean TXN has no phase-repair ops: nothing to omit"
        );
        let got = run_forced(&d, m, n, k, &ops);
        assert!(got.is_some(),"{v} {m}x{n}x{k}: lean TXN minus its {removed} phase-repair ops still behaves like lean: the repairs are not load-bearing");
        eprintln!(
            "NEGATIVE {v} {m}x{n}x{k}: omitting {removed} phase-repair ops of {} lean ops -> {:?}",
            removed + ops.len(),
            got.unwrap()
        );
    }
}

/// Removes only the FIRST op writing the channel task queue register `off` of column `col`'s memtile with BD `bd`.
fn drop_one_queue_write(col: u32, off: u32, bd: u32) -> impl Fn(&TxnOp) -> bool {
    let dropped = Cell::new(false);
    move |op| match *op {
        TxnOp::Write(a, v)
            if !dropped.get()
                && a >> 25 == col
                && op_row(op) == Some(1)
                && a & 0xfffff == off
                && v & 0xffff == bd =>
        {
            dropped.set(true);
            true
        }
        _ => false,
    }
}
/// V10 prefix finite tasks are NOT self-restoring: each finite BD runs `len` times, its iteration `current` wraps back to 0
/// and the channel is idle with an empty queue afterwards (sim `iteration_loaded`: `(current+1)%wrap`), and the balanced
/// finite FULL / EMPTY credits return to their initial values, so omitting the iteration-current clears is harmless by
/// construction and is NOT tested. Omitting ONE column's task queue write (here memtile queue registers, col 0: S2MM4 at
/// 0xa0624, MM2S1 at 0xa063c) leaves that column's fill or reader idle, and the run must be detected. The unmodified lean
/// ops are the control: they behave like lean on the same shape.
fn prefix_omitted_queue_detected(
    m: usize,
    n: usize,
    k: usize,
    off: u32,
    bd: u32,
    what_dropped: &str,
) {
    let d = prefix(m, n, k, Control::Fast);
    let what = format!("V10-prefix {m}x{n}x{k}");
    let (control, removed) = forced_ops(&d, &|_| false);
    assert_eq!(removed, 0);
    assert!(
        run_forced(&d, m, n, k, &control).is_none(),
        "{what}: the unmodified lean ops must behave like lean"
    );
    let (ops, removed) = forced_ops(&d, &drop_one_queue_write(0, off, bd));
    assert_eq!(
        removed, 1,
        "{what}: expected exactly one {what_dropped} queue write to omit"
    );
    let got = run_forced(&d, m, n, k, &ops);
    assert!(got.is_some(),"{what}: omitting the col 0 {what_dropped} queue write still behaves like lean: the requeue is not load-bearing");
    eprintln!(
        "NEGATIVE {what}: omitting col 0 {what_dropped} queue write -> {:?}",
        got.unwrap()
    );
}
#[test]
fn omitting_prefix_reader_bd30_requeue_is_detected() {
    prefix_omitted_queue_detected(1024, 1024, 128, 0xa063c, 30, "guarded reader BD30")
}
#[test]
fn omitting_prefix_second_group_reader_bd31_requeue_is_detected() {
    prefix_omitted_queue_detected(512, 4096, 704, 0xa063c, 31, "second-group reader BD31")
}
#[test]
fn omitting_prefix_second_group_fill_bd5_requeue_is_detected() {
    prefix_omitted_queue_detected(512, 4096, 704, 0xa0624, 5, "second-group fill BD5")
}
#[test]
fn omitting_prefix_lockfree_bd36_requeue_is_detected() {
    prefix_omitted_queue_detected(1024, 1024, 128, 0xa063c, 36, "lockfree replay BD36")
}

// ---------------------------------------------------------------- sizes
/// Prints full vs lean TXN bytes (design only, no simulation) for a small V9 and the production gate_up / down shapes.
#[test]
fn txn_sizes_full_vs_lean() {
    for (v, m, n, k, label) in [
        (Variant::V9, 512, 512, 128, "V9 small"),
        (Variant::V9, 4096, 1280, 2560, "V9 gateup"),
        (Variant::V9, 4096, 2560, 640, "V9 down"),
        (Variant::V10, 4096, 2560, 640, "V10 down"),
    ] {
        let d = design(v, m, n, k);
        let full = d.insts.len();
        let lean = d
            .lean_insts()
            .unwrap_or_else(|e| panic!("{label} {m}x{n}x{k}: lean_insts: {e}"))
            .len();
        assert!(
            lean < full,
            "{label}: lean {lean} B is not smaller than full {full} B"
        );
        eprintln!("TXN {label} {m}x{n}x{k}: full={full} B lean={lean} B ({:.1}% of full, ops full={} lean={})",100.0*lean as f64/full as f64,
            parse_txn(&d.insts).unwrap().len(),parse_txn(&d.lean_insts().unwrap()).unwrap().len());
    }
}
/// Prints full vs lean TXN bytes (design only, no simulation) for the V10 B-prefix down shape and a 160-row (MW1, padded) one.
#[test]
fn txn_sizes_v10_prefix_full_vs_lean() {
    for (m, n, k) in [(4096, 2560, 640), (160, 2560, 640)] {
        let d = prefix(m, n, k, Control::Fast);
        let full = d.insts.len();
        let lean = lean_of(&d, &format!("V10-prefix {m}x{n}x{k}"));
        assert!(
            lean.len() < full,
            "V10-prefix {m}x{n}x{k}: lean {} B is not smaller than full {full} B",
            lean.len()
        );
        eprintln!("TXN V10-prefix {m}x{n}x{k}: full={full} B lean={} B ({:.1}% of full, ops full={} lean={})",lean.len(),100.0*lean.len() as f64/full as f64,
            parse_txn(&d.insts).unwrap().len(),parse_txn(&lean).unwrap().len());
    }
}
