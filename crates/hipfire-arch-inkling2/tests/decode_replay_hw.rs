// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HIP-graph executor (`decode_replay::DecodeGraph`) against the frozen eager
//! decode on the typed-XTS artifact, gfx1201.
//!
//! Same lockstep design as `decode_step_hw` (whose helpers it reuses through a
//! `#[path]` module): every compared step starts both arms from the IDENTICAL
//! pre-step state (conv, the K/V rows at the write slot, host cursors saved and
//! restored), the arena slab and the FA2 partials are poisoned with a changing
//! pattern before each arm, and after the frozen-eager arm and the graph arm
//! the following is compared bit for bit:
//!
//! * the full unpadded-vocab logits (`to_bits`);
//! * all eight K and eight V F16 planes of all 48 visits at the write slot;
//! * the F32 `[3, 6656]` conv history of every visit;
//! * every `visit_cap`, `visit_pos_tags`, `seq_pos`;
//! * every named arena row (x ping-pong, prelude/previous-pass stream rows,
//!   the pre / prev / cur block banks, the head hidden row, the logits row);
//! * (witness steps) a hash of every live K/V row except the write slot.
//!
//! The graph arm is the real `DecodeGraph::forward`. Per step the test proves
//! which route actually ran: a replay issues exactly ONE `hipGraphLaunch`
//! (hip-bridge counter) and ZERO module launches, one H2D and one D2H; the
//! capture step runs the eager token (1024 launches) plus the recording
//! (1024 more, executing nothing) and issues no graph launch. The captured
//! node count (= graph-owned kernarg blobs = launches observed while recording)
//! must equal the eager census.
//!
//! ```sh
//! INKLING2_MQ4=.../inkling2-mini-xts-c.hfq HIPFIRE_GRAPH=0 \
//!   cargo test --release -p hipfire-arch-inkling2 --test decode_replay_hw -- \
//!   --ignored --nocapture --test-threads=1 --exact <name>
//! ```

#[path = "decode_step_hw.rs"]
#[allow(dead_code)]
mod oracle;

use hip_bridge::{launch_counters, memory_effects};
use hipfire_arch_inkling2::arch_model::Inkling2Bundle;
use hipfire_arch_inkling2::decode_replay::{
    graph_dev_opt_in, pm4_dev_opt_in, DecodeGraph, DecodePm4, EagerWhy, Pm4Route, StepRoute,
};
use hipfire_arch_inkling2::{ensure_decode_arena, forward_decode_frozen, plan_decode_step};
use oracle::{
    compare_post, f32_bytes, load, poison, post_state, prefill, reset, restore, save_pre, sha,
    tokens, witness, Receipt,
};
use rdna_compute::Gpu;

const EXPECT_NODES: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expect {
    /// Replay iff the test's own copy of the last captured key equals the plan's.
    Auto,
    Replay,
    Captured,
}

struct Rig {
    dg: DecodeGraph,
    rec: Receipt,
    salt: u32,
    /// The key the test saw captured last (its own prediction of the tape).
    key: Option<hipfire_arch_inkling2::DecodeStepKey>,
    replays: usize,
    captured: usize,
}

impl Rig {
    fn new(name: &str) -> Self {
        Self {
            dg: DecodeGraph::new(true),
            rec: Receipt::new(name),
            salt: 0,
            key: None,
            replays: 0,
            captured: 0,
        }
    }
}

struct Opts {
    witness: bool,
    /// Positive control: flip a conv bit in the graph arm only, on a step that
    /// is expected to replay.
    plant: bool,
}

/// Named arena rows (banks, ping-pong, head rows) as raw bytes.
fn banks(b: &Inkling2Bundle, gpu: &Gpu) -> Vec<(String, Vec<u8>)> {
    let a = b.gpu_state.as_ref().unwrap().decode_arena.as_ref().unwrap();
    a.named_tensors()
        .into_iter()
        .map(|(n, t)| (n, gpu.download_raw_bytes(t).expect("bank")))
        .collect()
}

fn first_bank_diff(want: &[(String, Vec<u8>)], got: &[(String, Vec<u8>)]) -> Option<String> {
    for ((n, a), (m, g)) in want.iter().zip(got) {
        assert_eq!(n, m);
        if let Some(o) = a.iter().zip(g).position(|(x, y)| x != y) {
            return Some(format!(
                "arena row {n} byte {o}: eager {:#04x} vs graph {:#04x}",
                a[o], g[o]
            ));
        }
    }
    None
}

fn route_name(r: &StepRoute) -> &'static str {
    match r {
        StepRoute::Replay => "replay",
        StepRoute::EagerCaptured => "eager+capture",
        StepRoute::Eager(EagerWhy::Disabled) => "eager(disabled)",
        StepRoute::Eager(EagerWhy::Sticky(_)) => "eager(sticky)",
        StepRoute::Eager(EagerWhy::CaptureFailed(_)) => "eager(capture-failed)",
    }
}

/// One compared step at `pos`; leaves the state post-step (the graph arm's).
fn step(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rig: &mut Rig,
    tok: u32,
    pos: usize,
    o: &Opts,
    expect: Expect,
) -> StepRoute {
    ensure_decode_arena(gpu, b).expect("arena");
    let plan = plan_decode_step(b, gpu, pos).expect("plan");
    let bucket = (plan.key.bucket.lo, plan.key.bucket.hi);
    let want_replay = match expect {
        Expect::Auto => rig.key.as_ref() == Some(&plan.key),
        Expect::Replay => true,
        Expect::Captured => false,
    };
    let mut next_salt = || {
        rig.salt += 1;
        rig.salt
    };
    let pre = save_pre(b, gpu, pos);
    let w_pre = o.witness.then(|| witness(b, gpu, pos));

    // Arm 1: frozen eager (the comparator), with its launch census.
    let s = next_salt();
    poison(b, gpu, s);
    let fx0 = memory_effects::snapshot();
    let mut le = Vec::new();
    forward_decode_frozen(b, gpu, tok, pos, &mut le).expect("eager arm");
    let fe = memory_effects::snapshot().since(fx0);
    let eager_launches = fe.launches;
    assert_eq!(
        (fe.htod, fe.dtoh, fe.dtod, fe.memset),
        (1, 1, 0, 0),
        "pos {pos}: eager step effects {fe:?}"
    );
    let pe = post_state(b, gpu, pos, le);
    let be = banks(b, gpu);
    let w_e = o.witness.then(|| witness(b, gpu, pos));

    // Arm 2: the graph executor, from the same pre-state.
    restore(b, gpu, &pre);
    let s = next_salt();
    poison(b, gpu, s);
    if o.plant && want_replay {
        let gs = b.gpu_state.as_ref().unwrap();
        let mut bytes = gpu.download_raw_bytes(&gs.conv[20]).unwrap();
        bytes[6656 * 4 + 5 * 4 + 3] ^= 0x20;
        gpu.memcpy_htod_auto(&gs.conv[20].buf, &bytes).unwrap();
    }
    let stats0 = rig.dg.stats().clone();
    let g0 = launch_counters::graph_launch::count();
    let fx1 = memory_effects::snapshot();
    let mut lg = Vec::new();
    let rep = rig
        .dg
        .forward(b, gpu, tok, pos, &mut lg)
        .unwrap_or_else(|e| panic!("pos {pos}: graph arm failed: {e}"));
    let fxg = memory_effects::snapshot().since(fx1);
    let glaunch = launch_counters::graph_launch::count() - g0;
    let stats1 = rig.dg.stats().clone();

    // Prove which route ran.
    if want_replay {
        assert_eq!(
            rep.route,
            StepRoute::Replay,
            "pos {pos} bucket {bucket:?}: expected a replay of the retained graph"
        );
        assert_eq!(
            glaunch, 1,
            "pos {pos}: a replay is exactly one hipGraphLaunch"
        );
        assert_eq!(
            fxg.launches, 0,
            "pos {pos}: a replay issues no module launch (the graph executed the nodes)"
        );
        assert_eq!(
            (fxg.htod, fxg.dtoh, fxg.dtod, fxg.memset),
            (1, 1, 0, 0),
            "pos {pos}: replay effects {fxg:?}"
        );
        assert_eq!(stats1.graph_launches, stats0.graph_launches + 1);
        assert_eq!(stats1.replays, stats0.replays + 1);
        rig.replays += 1;
    } else {
        assert_eq!(
            rep.route,
            StepRoute::EagerCaptured,
            "pos {pos} bucket {bucket:?}: expected eager token + capture"
        );
        assert_eq!(glaunch, 0, "pos {pos}: capture launches nothing");
        assert_eq!(
            fxg.launches,
            2 * eager_launches,
            "pos {pos}: eager token + recording launch counts"
        );
        assert_eq!(stats1.captures, stats0.captures + 1);
        assert_eq!(
            stats1.nodes, eager_launches as usize,
            "pos {pos}: captured nodes == eager census"
        );
        assert_eq!(stats1.nodes, EXPECT_NODES, "pos {pos}: 1024-launch body");
        assert_eq!(stats1.eager_launches, eager_launches);
        rig.key = Some(plan.key.clone());
        rig.captured += 1;
        eprintln!(
            "    capture: {} nodes, {:.2} ms (capture + instantiate)",
            stats1.nodes,
            rep.capture_ns as f64 / 1e6
        );
    }
    assert_eq!(
        rig.dg.active_key(),
        Some(&plan.key),
        "pos {pos}: the active graph is for this key"
    );

    let pg = post_state(b, gpu, pos, lg);
    let bg = banks(b, gpu);
    let w_g = o.witness.then(|| witness(b, gpu, pos));
    if let Some(d) = compare_post(&pe, &pg) {
        let d = d.replace("stock", "eager").replace("frozen", "graph");
        panic!(
            "MISMATCH pos {pos} ({}): graph vs frozen eager first differing site: {d}",
            route_name(&rep.route)
        );
    }
    if let Some(d) = first_bank_diff(&be, &bg) {
        panic!(
            "MISMATCH pos {pos} ({}): graph vs frozen eager first differing site: {d}",
            route_name(&rep.route)
        );
    }
    if o.witness {
        let (wp, we, wg) = (w_pre.unwrap(), w_e.unwrap(), w_g.unwrap());
        assert_eq!(
            we, wp,
            "pos {pos}: eager arm changed a non-write-slot KV row"
        );
        assert_eq!(
            wg, wp,
            "pos {pos}: graph arm changed a non-write-slot KV row"
        );
    }
    rig.rec.line(format!(
        "{{\"pos\":{pos},\"bucket\":[{},{}],\"route\":\"{}\",\"graph_launches\":{glaunch},\"module_launches\":{},\"logits_sha256\":\"{}\",\"ok\":true}}",
        bucket.0,
        bucket.1,
        route_name(&rep.route),
        fxg.launches,
        sha(f32_bytes(&pe.logits))
    ));
    eprintln!(
        "  pos {pos:5} bucket [{},{}] {:13} bitwise-equal (logits+KV slot+conv+tags+banks){}",
        bucket.0,
        bucket.1,
        route_name(&rep.route),
        if o.witness { " witness" } else { "" }
    );
    rep.route
}

fn segment(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rig: &mut Rig,
    toks: &[u32],
    (start, end, first): (usize, usize, Expect),
    wit: &dyn Fn(usize) -> bool,
) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..start]);
    eprintln!("segment {start}..={end} (first step expects {first:?})");
    for pos in start..=end {
        let o = Opts {
            witness: wit(pos) || pos == start,
            plant: false,
        };
        let e = if pos == start { first } else { Expect::Auto };
        step(b, gpu, rig, toks[pos], pos, &o, e);
    }
}

/// Refusals before submission leave host state and the retained graph alone.
fn refusals(b: &mut Inkling2Bundle, gpu: &mut Gpu, rig: &mut Rig, last_legal: usize) {
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(gs.seq_pos, last_legal + 1);
    let before = (gs.seq_pos, gs.visit_cap.clone(), gs.visit_pos_tags.clone());
    let key_before = rig.dg.active_key().cloned();
    let stats_before = rig.dg.stats().clone();
    let mut l = Vec::new();
    let e = rig
        .dg
        .forward(b, gpu, 7, last_legal + 1, &mut l)
        .expect_err("next position past the cap");
    eprintln!("  refused pos {} (cap): {e}", last_legal + 1);
    let e = rig
        .dg
        .forward(b, gpu, 7, last_legal, &mut l)
        .expect_err("out-of-order position");
    eprintln!("  refused pos {last_legal} (out-of-order): {e}");
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(
        (gs.seq_pos, &gs.visit_cap, &gs.visit_pos_tags),
        (before.0, &before.1, &before.2)
    );
    assert_eq!(
        rig.dg.active_key().cloned(),
        key_before,
        "refusal kept the tape"
    );
    let s = rig.dg.stats();
    assert_eq!(
        (s.replays, s.captures, s.eager_steps, s.destroys),
        (
            stats_before.replays,
            stats_before.captures,
            stats_before.eager_steps,
            stats_before.destroys
        ),
        "a refusal executed nothing"
    );
}

/// Bad cursors / tags are refused before any submission, and the retained
/// graph stays usable for the very next valid step.
fn bad_state_refusals(b: &mut Inkling2Bundle, gpu: &mut Gpu, rig: &mut Rig, toks: &[u32]) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..40]);
    let mut l = Vec::new();
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] += 1;
    let e = rig
        .dg
        .forward(b, gpu, toks[40], 40, &mut l)
        .expect_err("bad visit_cap");
    eprintln!("  refused (visit_cap): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] -= 1;
    let t = gs.visit_pos_tags[3][10];
    gs.visit_pos_tags[3][10] = 999;
    let e = rig
        .dg
        .forward(b, gpu, toks[40], 40, &mut l)
        .expect_err("bad tag");
    eprintln!("  refused (tag): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_pos_tags[3][10] = t;
    rig.dg
        .forward(b, gpu, toks[40], 41, &mut l)
        .expect_err("ahead");
    rig.dg
        .forward(b, gpu, toks[40], 39, &mut l)
        .expect_err("behind");
    // Restored state steps normally (compared).
    step(
        b,
        gpu,
        rig,
        toks[40],
        40,
        &Opts {
            witness: false,
            plant: false,
        },
        Expect::Auto,
    );
}

fn finish(b: &mut Inkling2Bundle, gpu: &mut Gpu, rig: &mut Rig, name: &str) {
    let s = rig.dg.stats().clone();
    eprintln!(
        "decode_replay_hw {name}: ALL BITWISE-EQUAL; {} replays, {} captures ({} nodes each, last {:.2} ms), {} destroys, {} eager, {} graph launches",
        s.replays,
        s.captures,
        s.nodes,
        s.last_capture_ns as f64 / 1e6,
        s.destroys,
        s.eager_steps,
        s.graph_launches
    );
    assert_eq!(s.replays as usize, rig.replays);
    assert_eq!(s.captures as usize, rig.captured);
    assert_eq!(s.graph_launches, s.replays, "one graph launch per replay");
    assert_eq!(
        s.destroys + 1,
        s.captures,
        "every key change destroyed the previous graph before rebinding"
    );
    assert_eq!(s.capture_failures, 0);
    assert!(rig.dg.sticky_reason().is_none());
    rig.dg.release(gpu).expect("release");
    assert!(rig.dg.active_key().is_none());
    assert_eq!(rig.dg.stats().destroys, s.captures);
    let _ = b;
}

/// Prime the shared scratch to its prefill high-water before the first capture,
/// so a later (longer) prefill cannot legitimately move a pointee between
/// steps; the key's layout epoch is checked by the dedicated test.
fn prewarm(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32], max_seq: usize) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..max_seq.saturating_sub(1).min(2052)]);
}

fn run_matrix(
    max_seq: usize,
    segments: &[(usize, usize, Expect)],
    name: &str,
    wit: &dyn Fn(usize) -> bool,
) {
    let mut gpu = Gpu::init().expect("gpu");
    assert!(
        !graph_dev_opt_in(&gpu),
        "the oracle runs with HIPFIRE_GRAPH=0; the dev opt-in must read false"
    );
    let mut b = load(&mut gpu, max_seq);
    let toks = tokens(max_seq + 16);
    prewarm(&mut b, &mut gpu, &toks, max_seq);
    let mut rig = Rig::new(name);
    for &s in segments {
        segment(&mut b, &mut gpu, &mut rig, &toks, s, wit);
        let last = max_seq.min(8192) - 1;
        if s.1 == last {
            refusals(&mut b, &mut gpu, &mut rig, s.1);
        }
    }
    bad_state_refusals(&mut b, &mut gpu, &mut rig, &toks);
    finish(&mut b, &mut gpu, &mut rig, name);
}

fn wit_default(p: usize) -> bool {
    matches!(p, 1023 | 1024 | 1025 | 2047 | 2048 | 4095 | 4096)
}

/// Clipped non-power-of-two capacity 2053. First warm/capture/replay from
/// ctx 128 for 64 consecutive same-bucket steps, then the 190..194 crossing, an
/// 81-step same-bucket run (reset into the retained bucket), the 1022..1026 SWA
/// wrap, the 2046..2050 wrap + 8->16 split crossing and the last legal row,
/// then reset into a different bucket, a crossing, and a reset into the
/// retained bucket again.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn graph_vs_eager_clipped_2053() {
    use Expect::*;
    run_matrix(
        2053,
        &[
            (128, 191, Captured), // 64 steps of bucket [129,192]: 1 capture, 63 replays
            (190, 194, Replay),   // reset into the retained bucket; 192 crosses to [193,2048]
            (300, 380, Replay),   // reset into the retained bucket: 81 consecutive replays
            (1022, 1026, Replay), // SWA wrap, one bucket
            (2046, 2052, Replay), // 2048 -> bucket [2049,2053]; last legal 2052
            (96, 134, Captured),  // reset into a DIFFERENT bucket; 127 -> 128 crosses
            (186, 198, Replay),   // reset into the retained [129,192]; 192 crosses
        ],
        "graph-clipped2053.jsonl",
        &wit_default,
    );
}

/// 4094..4098 across the 4095 -> 4096 16 -> 8 split-band change of a 4100-row
/// load; the last legal row 4099 and the refused next one.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn graph_vs_eager_4100() {
    run_matrix(
        4100,
        &[(4090, 4099, Expect::Captured)],
        "graph-cap4100.jsonl",
        &wit_default,
    );
}

/// Capacity exactly at the 1024 ring break (SWA ring == full rows).
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn graph_vs_eager_1024() {
    run_matrix(
        1024,
        &[(1010, 1023, Expect::Captured), (0, 30, Expect::Captured)],
        "graph-cap1024.jsonl",
        &wit_default,
    );
}

fn planted() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 300);
    let toks = tokens(320);
    prewarm(&mut b, &mut gpu, &toks, 300);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..130]);
    let mut rig = Rig::new("graph-planted.jsonl");
    for pos in 130..136 {
        let o = Opts {
            witness: false,
            // Only a step that really replays: the first (capture) step is
            // clean so the planted fault is reported on a replay.
            plant: pos >= 131,
        };
        step(&mut b, &mut gpu, &mut rig, toks[pos], pos, &o, Expect::Auto);
    }
}

/// Positive control: a flipped conv bit in the graph arm of a REPLAY step must
/// be reported at its first differing site.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
#[should_panic(expected = "MISMATCH")]
fn planted_conv_fault_in_a_replay_is_caught() {
    planted();
}

/// Sticky eager fallback, foreign-capture refusal, disabled executor.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn sticky_fallback_and_foreign_capture_refusal() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    let mut logits = Vec::new();

    // A disabled executor is the frozen eager step, stated as such.
    let mut off = DecodeGraph::new(false);
    let r = off
        .forward(&mut b, &mut gpu, toks[140], 140, &mut logits)
        .unwrap();
    assert_eq!(r.route, StepRoute::Eager(EagerWhy::Disabled));
    assert!(off.active_key().is_none());
    assert_eq!(
        gpu.graphs.graph_exec.is_some(),
        false,
        "no graph was captured"
    );
    let want = {
        // Reference logits of the next step from the same state.
        let pre = save_pre(&b, &gpu, 141);
        let mut l = Vec::new();
        forward_decode_frozen(&mut b, &mut gpu, toks[141], 141, &mut l).unwrap();
        restore(&mut b, &gpu, &pre);
        l
    };

    // A foreign capture in flight: refused before anything is submitted.
    let mut dg = DecodeGraph::new(true);
    let (cap, cursors) = {
        let gs = b.gpu_state.as_ref().unwrap();
        (gs.seq_pos, gs.visit_cap.clone())
    };
    gpu.graphs.capture_mode = true;
    let e = dg
        .forward(&mut b, &mut gpu, toks[141], 141, &mut logits)
        .expect_err("foreign capture");
    gpu.graphs.capture_mode = false;
    eprintln!("  refused (foreign capture): {e}");
    assert!(e.contains("foreign"));
    {
        let gs = b.gpu_state.as_ref().unwrap();
        assert_eq!((gs.seq_pos, &gs.visit_cap), (cap, &cursors));
    }

    // Capture failure after the successful eager token: the token stands, the
    // executor goes sticky-eager, and a reset or bucket change does not clear it.
    gpu.graphs.host_cpu_weights = true; // begin_graph_capture refuses (injected)
    let r = dg
        .forward(&mut b, &mut gpu, toks[141], 141, &mut logits)
        .expect("the eager token is returned despite the capture failure");
    gpu.graphs.host_cpu_weights = false;
    match &r.route {
        StepRoute::Eager(EagerWhy::CaptureFailed(why)) => {
            eprintln!("  capture failed (injected): {why}")
        }
        other => panic!("expected a capture failure, got {other:?}"),
    }
    assert_eq!(
        logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "the step whose capture failed returned the exact eager logits"
    );
    assert_eq!(
        b.gpu_state.as_ref().unwrap().seq_pos,
        142,
        "and committed once"
    );
    assert!(dg.sticky_reason().is_some());
    assert!(dg.active_key().is_none());
    assert!(
        gpu.graphs.graph_exec.is_none(),
        "the failed capture left no graph"
    );
    assert!(!gpu.graphs.capture_mode, "capture mode cleared");
    // Later steps, a reset and another bucket stay eager (and exact).
    for pos in 142..146 {
        let pre = save_pre(&b, &gpu, pos);
        let mut l = Vec::new();
        forward_decode_frozen(&mut b, &mut gpu, toks[pos], pos, &mut l).unwrap();
        restore(&mut b, &gpu, &pre);
        let r = dg
            .forward(&mut b, &mut gpu, toks[pos], pos, &mut logits)
            .unwrap();
        assert!(
            matches!(r.route, StepRoute::Eager(EagerWhy::Sticky(_))),
            "{r:?}"
        );
        assert_eq!(
            sha(f32_bytes(&l)),
            sha(f32_bytes(&logits)),
            "sticky eager step at {pos} is the frozen eager step"
        );
    }
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..100]);
    let r = dg
        .forward(&mut b, &mut gpu, toks[100], 100, &mut logits)
        .unwrap();
    assert!(
        matches!(r.route, StepRoute::Eager(EagerWhy::Sticky(_))),
        "{r:?}"
    );
    assert_eq!(dg.stats().replays, 0);
    assert_eq!(dg.stats().graph_launches, 0);
    eprintln!(
        "decode_replay_hw sticky_fallback: sticky reason kept across steps and a reset into another bucket: {}",
        dg.sticky_reason().unwrap()
    );
    dg.release(&mut gpu).unwrap();
}

/// A new layout epoch (shared scratch growth) and a new arena owner (reload)
/// each make the retained tape unusable: it is destroyed (quiesced) and a new
/// one recorded, and the new tape is bitwise the eager step.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn scratch_growth_and_reload_invalidate_the_tape() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 2053);
    let toks = tokens(2100);
    prewarm(&mut b, &mut gpu, &toks, 2053);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..300]);
    let mut rig = Rig::new("graph-invalidate.jsonl");
    let o = Opts {
        witness: false,
        plant: false,
    };
    use Expect::*;
    step(&mut b, &mut gpu, &mut rig, toks[300], 300, &o, Captured);
    step(&mut b, &mut gpu, &mut rig, toks[301], 301, &o, Replay);
    // Shared scratch growth: new layout epoch => new key. The prewarm already
    // sized the A8 scratch to the prefill high-water, so ask for more until the
    // epoch really moves.
    let e0 = rdna_compute::scratch::scratch_growth_epoch();
    let mut slots = 1usize << 13;
    while rdna_compute::scratch::scratch_growth_epoch() == e0 {
        assert!(slots <= 1 << 19, "scratch never grew");
        gpu.reserve_int8_mmq(2304, slots).unwrap();
        slots *= 2;
    }
    step(&mut b, &mut gpu, &mut rig, toks[302], 302, &o, Captured);
    step(&mut b, &mut gpu, &mut rig, toks[303], 303, &o, Replay);
    // Reload: the arena (and every pointee it owns) is a new owner. The old
    // graph still names the freed allocation; it must never launch.
    let arena = b.gpu_state.as_mut().unwrap().decode_arena.take().unwrap();
    arena.free(&mut gpu);
    step(&mut b, &mut gpu, &mut rig, toks[304], 304, &o, Captured);
    step(&mut b, &mut gpu, &mut rig, toks[305], 305, &o, Replay);
    step(&mut b, &mut gpu, &mut rig, toks[306], 306, &o, Replay);
    finish(&mut b, &mut gpu, &mut rig, "invalidate");
}

/// Two recordings of the same bucket at different positions, with different
/// staged contents, on the same allocations, carry byte-identical kernargs:
/// no position (or token) is baked into the tape.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn same_bucket_recordings_have_identical_kernarg_bytes() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    let mut logits = Vec::new();
    let mut a = DecodeGraph::new(true);
    let r = a
        .forward(&mut b, &mut gpu, toks[140], 140, &mut logits)
        .unwrap();
    assert_eq!(r.route, StepRoute::EagerCaptured);
    let blobs_a = gpu.graphs.ar_forward_blobs.clone();
    a.release(&mut gpu).unwrap();
    assert!(
        gpu.graphs.ar_forward_blobs.is_empty(),
        "blobs freed with the graph"
    );
    let mut c = DecodeGraph::new(true);
    let r = c
        .forward(&mut b, &mut gpu, toks[141], 141, &mut logits)
        .unwrap();
    assert_eq!(r.route, StepRoute::EagerCaptured);
    let blobs_c = gpu.graphs.ar_forward_blobs.clone();
    assert_eq!(blobs_a.len(), EXPECT_NODES);
    assert_eq!(blobs_a.len(), blobs_c.len());
    for (i, (x, y)) in blobs_a.iter().zip(&blobs_c).enumerate() {
        assert_eq!(
            x, y,
            "kernarg blob {i} differs between recordings at pos 140 and 141"
        );
    }
    let bytes: usize = blobs_a.iter().map(Vec::len).sum();
    eprintln!(
        "decode_replay_hw kernarg witness: {} blobs / {bytes} B identical across recordings at pos 140 and 141 (different token, position and staged record)",
        blobs_a.len()
    );
    c.release(&mut gpu).unwrap();
}

/// The Gpu-level graph dropped under the executor (fail-closed rollback / a
/// daemon invalidation calls `graph_destroy`) is a miss: the executor
/// re-captures through the eager warm token and never launches a missing graph.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn graph_dropped_under_the_executor_is_a_miss() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    let mut rig = Rig::new("graph-dropped.jsonl");
    let o = Opts {
        witness: false,
        plant: false,
    };
    use Expect::*;
    step(&mut b, &mut gpu, &mut rig, toks[140], 140, &o, Captured);
    step(&mut b, &mut gpu, &mut rig, toks[141], 141, &o, Replay);
    gpu.hip.device_synchronize().unwrap();
    gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
    assert!(gpu.graphs.graph_exec.is_none());
    // Same key and layout, but the Gpu no longer holds the graph: a miss.
    step(&mut b, &mut gpu, &mut rig, toks[142], 142, &o, Captured);
    step(&mut b, &mut gpu, &mut rig, toks[143], 143, &o, Replay);
    finish(&mut b, &mut gpu, &mut rig, "dropped");
}

// ---------------------------------------------------------------------------
// PM4 (retained) executor
//
// ```sh
// INKLING2_MQ4=.../inkling2-mini-xts-c.hfq HIPFIRE_GRAPH=0 \
//   HIPFIRE_REPLAY_BACKEND=redline HIPFIRE_REPLAY_TRANSPORT=pm4 \
//   cargo test --release -p hipfire-arch-inkling2 --test decode_replay_hw -- \
//   --ignored --nocapture --test-threads=1 --exact <pm4 test name>
// ```
//
// Per compared step three arms run from the IDENTICAL pre-state (conv, write-
// slot K/V rows, host cursors restored; FA2 partials + the arena slab poisoned
// with a changing pattern before each arm):
//   1. frozen eager (the comparator);
//   2. the exact recorded-HIP-blob replay of the held tape
//      (`replay_recorded_hip_prefix_at`), when a tape for the key is held;
//   3. the PM4 executor (warm eager / recording token / retained replay).
// Compared bit for bit after each: full unpadded logits, write-slot K/V of all
// 48 visits, conv x48, tags/cursors/seq_pos, every named arena row.
// ---------------------------------------------------------------------------

struct Rig4 {
    p: DecodePm4,
    rec: Receipt,
    salt: u32,
    /// The key the test saw recorded last (its own prediction of the tape).
    key: Option<hipfire_arch_inkling2::DecodeStepKey>,
    warm: bool,
    replays: usize,
    blob_replays: usize,
    recorded: usize,
}

impl Rig4 {
    fn new(name: &str) -> Self {
        Self {
            p: DecodePm4::new(
                true,
                std::env::var("INKLING2_PM4_STREAM").as_deref() == Ok("1"),
            ),
            rec: Receipt::new(name),
            salt: 0,
            key: None,
            warm: false,
            replays: 0,
            blob_replays: 0,
            recorded: 0,
        }
    }
}

fn pm4_route_name(r: &Pm4Route) -> &'static str {
    match r {
        Pm4Route::Replay => "pm4-replay",
        Pm4Route::Recorded => "eager+record",
        Pm4Route::Warm => "eager(warm)",
        Pm4Route::Eager(EagerWhy::Disabled) => "eager(disabled)",
        Pm4Route::Eager(EagerWhy::Sticky(_)) => "eager(sticky)",
        Pm4Route::Eager(EagerWhy::CaptureFailed(_)) => "eager(record-failed)",
    }
}

/// One compared step at `pos`; leaves the state post-step.
fn step4(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rig: &mut Rig4,
    tok: u32,
    pos: usize,
    o: &Opts,
) -> Pm4Route {
    ensure_decode_arena(gpu, b).expect("arena");
    let plan = plan_decode_step(b, gpu, pos).expect("plan");
    let bucket = (plan.key.bucket.lo, plan.key.bucket.hi);
    let want = if rig.key.as_ref() == Some(&plan.key) {
        Pm4Route::Replay
    } else if rig.warm {
        Pm4Route::Recorded
    } else {
        Pm4Route::Warm
    };
    let mut next_salt = || {
        rig.salt += 1;
        rig.salt
    };
    let pre = save_pre(b, gpu, pos);
    let w_pre = o.witness.then(|| witness(b, gpu, pos));

    // Arm 1: frozen eager.
    let s = next_salt();
    poison(b, gpu, s);
    let fx0 = memory_effects::snapshot();
    let mut le = Vec::new();
    forward_decode_frozen(b, gpu, tok, pos, &mut le).expect("eager arm");
    let fe = memory_effects::snapshot().since(fx0);
    assert_eq!(
        (fe.launches as usize, fe.htod, fe.dtoh, fe.dtod, fe.memset),
        (EXPECT_NODES, 1, 1, 0, 0),
        "pos {pos}: eager step effects {fe:?}"
    );
    let pe = post_state(b, gpu, pos, le);
    let be = banks(b, gpu);
    let w_e = o.witness.then(|| witness(b, gpu, pos));

    // Arm 2: the exact recorded HIP blobs of the held tape.
    if want == Pm4Route::Replay {
        restore(b, gpu, &pre);
        let s = next_salt();
        poison(b, gpu, s);
        let fx = memory_effects::snapshot();
        let mut lb = Vec::new();
        let r = rig
            .p
            .forward_blob_oracle(b, gpu, tok, pos, &mut lb)
            .unwrap_or_else(|e| panic!("pos {pos}: blob oracle failed: {e}"));
        let fxb = memory_effects::snapshot().since(fx);
        assert_eq!(r.route, Pm4Route::Replay);
        assert_eq!(
            (
                fxb.launches as usize,
                fxb.htod,
                fxb.dtoh,
                fxb.dtod,
                fxb.memset
            ),
            (EXPECT_NODES, 1, 1, 0, 0),
            "pos {pos}: blob-oracle effects {fxb:?}"
        );
        rig.blob_replays += 1;
        let pb = post_state(b, gpu, pos, lb);
        if let Some(d) = compare_post(&pe, &pb) {
            let d = d.replace("stock", "eager").replace("frozen", "blob");
            panic!(
                "MISMATCH pos {pos} (blob oracle): blob vs frozen eager first differing site: {d}"
            );
        }
        if let Some(d) = first_bank_diff(&be, &banks(b, gpu)) {
            panic!(
                "MISMATCH pos {pos} (blob oracle): blob vs frozen eager first differing site: {d}"
            );
        }
        if o.witness {
            assert_eq!(witness(b, gpu, pos), *w_pre.as_ref().unwrap());
        }
    }

    // Arm 3: the PM4 executor.
    restore(b, gpu, &pre);
    let s = next_salt();
    poison(b, gpu, s);
    if o.plant && want == Pm4Route::Replay {
        let gs = b.gpu_state.as_ref().unwrap();
        let mut bytes = gpu.download_raw_bytes(&gs.conv[20]).unwrap();
        bytes[6656 * 4 + 5 * 4 + 3] ^= 0x20;
        gpu.memcpy_htod_auto(&gs.conv[20].buf, &bytes).unwrap();
    }
    gpu.replay.begin_replay_observation_window();
    let stats0 = rig.p.stats().clone();
    let fx1 = memory_effects::snapshot();
    let mut lp = Vec::new();
    let rep = rig
        .p
        .forward(b, gpu, tok, pos, &mut lp)
        .unwrap_or_else(|e| panic!("pos {pos}: pm4 arm failed: {e}"));
    let fxp = memory_effects::snapshot().since(fx1);
    let stats1 = rig.p.stats().clone();
    let obs = gpu.replay.replay_observation();
    assert_eq!(
        rep.route,
        want,
        "pos {pos} bucket {bucket:?}: route {} != expected {}",
        pm4_route_name(&rep.route),
        pm4_route_name(&want)
    );
    match want {
        Pm4Route::Replay => {
            // Retirement proof: one checked submission, observed at this pos,
            // no fault; the body issued no HIP module launch.
            assert_eq!(
                fxp.launches, 0,
                "pos {pos}: a PM4 replay issues no HIP launch"
            );
            assert_eq!(
                (fxp.htod, fxp.dtoh, fxp.dtod, fxp.memset),
                (1, 1, 0, 0),
                "pos {pos}: pm4 replay effects {fxp:?}"
            );
            assert_eq!(
                (obs.count, obs.first_position, obs.last_position, obs.failed),
                (1, Some(pos), Some(pos), false),
                "pos {pos}: retained-replay observation {obs:?}"
            );
            assert_eq!(stats1.replays, stats0.replays + 1);
            rig.replays += 1;
        }
        Pm4Route::Recorded => {
            assert_eq!(
                fxp.launches as usize, EXPECT_NODES,
                "pos {pos}: the recording token is a real 1024-launch HIP token"
            );
            assert_eq!(obs.count, 0, "a recording token is not a retained replay");
            assert_eq!(stats1.recordings, stats0.recordings + 1);
            assert_eq!(
                stats1.nodes, EXPECT_NODES,
                "pos {pos}: recorded == body count"
            );
            assert!(stats1.unique_symbols > 0);
            assert_eq!(stats1.nodes as u64, stats1.eager_launches);
            rig.key = Some(plan.key.clone());
            rig.recorded += 1;
            eprintln!(
                "    record: {} launches, {} unique symbols, record {:.2} ms, prepare {:.2} ms, {} cmd dwords",
                stats1.nodes,
                stats1.unique_symbols,
                rep.record_ns as f64 / 1e6,
                rep.prepare_ns as f64 / 1e6,
                stats1.command_dwords
            );
        }
        Pm4Route::Warm => {
            assert_eq!(fxp.launches as usize, EXPECT_NODES);
            rig.warm = true;
        }
        Pm4Route::Eager(_) => unreachable!(),
    }
    if want != Pm4Route::Warm {
        assert_eq!(
            rig.p.active_key(),
            Some(&plan.key),
            "pos {pos}: tape is for this key"
        );
    }
    let pp = post_state(b, gpu, pos, lp);
    let bp = banks(b, gpu);
    let w_p = o.witness.then(|| witness(b, gpu, pos));
    if let Some(d) = compare_post(&pe, &pp) {
        let d = d.replace("stock", "eager").replace("frozen", "pm4");
        panic!(
            "MISMATCH pos {pos} ({}): pm4 vs frozen eager first differing site: {d}",
            pm4_route_name(&rep.route)
        );
    }
    if let Some(d) = first_bank_diff(&be, &bp) {
        panic!(
            "MISMATCH pos {pos} ({}): pm4 vs frozen eager first differing site: {d}",
            pm4_route_name(&rep.route)
        );
    }
    if o.witness {
        let (wp, we, wq) = (w_pre.unwrap(), w_e.unwrap(), w_p.unwrap());
        assert_eq!(
            we, wp,
            "pos {pos}: eager arm changed a non-write-slot KV row"
        );
        assert_eq!(wq, wp, "pos {pos}: pm4 arm changed a non-write-slot KV row");
    }
    rig.rec.line(format!(
        "{{\"pos\":{pos},\"bucket\":[{},{}],\"route\":\"{}\",\"hip_launches\":{},\"logits_sha256\":\"{}\",\"ok\":true}}",
        bucket.0,
        bucket.1,
        pm4_route_name(&rep.route),
        fxp.launches,
        sha(f32_bytes(&pe.logits))
    ));
    eprintln!(
        "  pos {pos:5} bucket [{},{}] {:13} bitwise-equal (logits+KV slot+conv+tags+banks){}{}",
        bucket.0,
        bucket.1,
        pm4_route_name(&rep.route),
        if want == Pm4Route::Replay {
            " +blob"
        } else {
            ""
        },
        if o.witness { " witness" } else { "" }
    );
    rep.route
}

fn segment4(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rig: &mut Rig4,
    toks: &[u32],
    (start, end): (usize, usize),
    wit: &dyn Fn(usize) -> bool,
) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..start]);
    eprintln!("segment {start}..={end}");
    for pos in start..=end {
        let o = Opts {
            witness: wit(pos) || pos == start,
            plant: false,
        };
        step4(b, gpu, rig, toks[pos], pos, &o);
    }
}

/// Refusals before submission leave host state and the held tape alone.
fn refusals4(b: &mut Inkling2Bundle, gpu: &mut Gpu, rig: &mut Rig4, last_legal: usize) {
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(gs.seq_pos, last_legal + 1);
    let before = (gs.seq_pos, gs.visit_cap.clone(), gs.visit_pos_tags.clone());
    let key_before = rig.p.active_key().cloned();
    let s0 = rig.p.stats().clone();
    let mut l = Vec::new();
    let e = rig
        .p
        .forward(b, gpu, 7, last_legal + 1, &mut l)
        .expect_err("next position past the cap");
    eprintln!("  refused pos {} (cap): {e}", last_legal + 1);
    let e = rig
        .p
        .forward(b, gpu, 7, last_legal, &mut l)
        .expect_err("out-of-order position");
    eprintln!("  refused pos {last_legal} (out-of-order): {e}");
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(
        (gs.seq_pos, &gs.visit_cap, &gs.visit_pos_tags),
        (before.0, &before.1, &before.2)
    );
    assert_eq!(
        rig.p.active_key().cloned(),
        key_before,
        "refusal kept the tape"
    );
    let s = rig.p.stats();
    assert_eq!(
        (s.replays, s.recordings, s.eager_steps, s.releases),
        (s0.replays, s0.recordings, s0.eager_steps, s0.releases),
        "a refusal executed nothing"
    );
}

fn bad_state_refusals4(b: &mut Inkling2Bundle, gpu: &mut Gpu, rig: &mut Rig4, toks: &[u32]) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..40]);
    let mut l = Vec::new();
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] += 1;
    let e = rig
        .p
        .forward(b, gpu, toks[40], 40, &mut l)
        .expect_err("bad visit_cap");
    eprintln!("  refused (visit_cap): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] -= 1;
    let t = gs.visit_pos_tags[3][10];
    gs.visit_pos_tags[3][10] = 999;
    let e = rig
        .p
        .forward(b, gpu, toks[40], 40, &mut l)
        .expect_err("bad tag");
    eprintln!("  refused (tag): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_pos_tags[3][10] = t;
    rig.p
        .forward(b, gpu, toks[40], 41, &mut l)
        .expect_err("ahead");
    rig.p
        .forward(b, gpu, toks[40], 39, &mut l)
        .expect_err("behind");
    step4(
        b,
        gpu,
        rig,
        toks[40],
        40,
        &Opts {
            witness: false,
            plant: false,
        },
    );
}

fn finish4(gpu: &mut Gpu, rig: &mut Rig4, name: &str) {
    let s = rig.p.stats().clone();
    eprintln!(
        "decode_replay_hw {name}: ALL BITWISE-EQUAL (frozen eager == recorded-blob == PM4); {} pm4 replays, {} blob replays, {} recordings ({} launches, {} unique symbols, last record {:.2} ms, last prepare {:.2} ms), {} releases, {} eager (warm {})",
        s.replays,
        s.blob_replays,
        s.recordings,
        s.nodes,
        s.unique_symbols,
        s.last_record_ns as f64 / 1e6,
        s.last_prepare_ns as f64 / 1e6,
        s.releases,
        s.eager_steps,
        s.warm_steps
    );
    assert_eq!(s.replays as usize, rig.replays);
    assert_eq!(s.blob_replays as usize, rig.blob_replays);
    assert_eq!(s.recordings as usize, rig.recorded);
    assert_eq!(
        s.releases + 1,
        s.recordings,
        "every key change released the previous tape before re-recording"
    );
    assert_eq!(s.record_failures, 0);
    assert!(rig.p.sticky_reason().is_none());
    assert!(!rig.p.quarantined());
    rig.p.release(gpu).expect("release");
    assert!(rig.p.active_key().is_none());
}

fn run_matrix4(
    max_seq: usize,
    segments: &[(usize, usize)],
    name: &str,
    wit: &dyn Fn(usize) -> bool,
) {
    let mut gpu = Gpu::init().expect("gpu");
    assert!(
        pm4_dev_opt_in(&gpu),
        "run with HIPFIRE_REPLAY_BACKEND=redline HIPFIRE_REPLAY_TRANSPORT=pm4 on gfx1201"
    );
    assert!(
        !graph_dev_opt_in(&gpu),
        "the PM4 oracle runs with HIPFIRE_GRAPH=0"
    );
    let mut b = load(&mut gpu, max_seq);
    let toks = tokens(max_seq + 16);
    prewarm(&mut b, &mut gpu, &toks, max_seq);
    let mut rig = Rig4::new(name);
    for &s in segments {
        segment4(&mut b, &mut gpu, &mut rig, &toks, s, wit);
        let last = max_seq.min(8192) - 1;
        if s.1 == last {
            refusals4(&mut b, &mut gpu, &mut rig, s.1);
        }
    }
    bad_state_refusals4(&mut b, &mut gpu, &mut rig, &toks);
    finish4(&mut gpu, &mut rig, name);
}

/// The S4 kill cell: a fresh load's warm token (pos 128), the recording token
/// (pos 129), then ctx 128 + 48 consecutive steps (pos 128..=175, keys
/// 129..=176, bucket end 192) as pure replays of the retained tape after a
/// reset (same key), each bitwise the frozen eager arm AND the exact recorded
/// HIP blobs. Then 190..194 (192 crosses to [193,2048]: release + re-record),
/// a 81-step same-bucket run, the 1022..1026 SWA wrap, 2046..2052 (wrap + 8->16
/// split crossing, last legal row, refusals), a reset into a different bucket
/// (127 -> 128) and back into the retained one.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn pm4_vs_eager_and_blob_clipped_2053() {
    run_matrix4(
        2053,
        &[
            (128, 129), // warm + record
            (128, 175), // 48 consecutive retained replays (keys 129..=176)
            (190, 194), // 192 crosses to [193,2048]
            (300, 380), // 81 consecutive replays of the [193,2048] tape
            (1022, 1026),
            (2046, 2052),
            (96, 134),  // reset into a different bucket; 127 -> 128 crosses
            (186, 198), // back into the retained [129,192]; 192 crosses
        ],
        "pm4-clipped2053.jsonl",
        &wit_default,
    );
}

#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn pm4_vs_eager_and_blob_4100() {
    run_matrix4(4100, &[(4090, 4099)], "pm4-cap4100.jsonl", &wit_default);
}

#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn pm4_vs_eager_and_blob_1024() {
    run_matrix4(
        1024,
        &[(1010, 1023), (0, 30)],
        "pm4-cap1024.jsonl",
        &wit_default,
    );
}

/// Positive control: a flipped conv bit in the PM4 arm of a REPLAY step must
/// be reported at its first differing site.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
#[should_panic(expected = "MISMATCH")]
fn pm4_planted_conv_fault_in_a_replay_is_caught() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut b = load(&mut gpu, 300);
    let toks = tokens(320);
    prewarm(&mut b, &mut gpu, &toks, 300);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..130]);
    let mut rig = Rig4::new("pm4-planted.jsonl");
    for pos in 130..137 {
        let o = Opts {
            witness: false,
            // The warm (130) and record (131) tokens are clean; the fault is
            // planted on a step that really replays.
            plant: pos >= 132,
        };
        step4(&mut b, &mut gpu, &mut rig, toks[pos], pos, &o);
    }
}

/// Without the opt-in the executor refuses before submitting anything and
/// leaves the load untouched; a disabled executor is the frozen eager step.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn pm4_refuses_without_opt_in_and_disabled_is_eager() {
    let mut gpu = Gpu::init().expect("gpu");
    if pm4_dev_opt_in(&gpu) {
        eprintln!("skipped: run this test WITHOUT HIPFIRE_REPLAY_BACKEND=redline");
        return;
    }
    let mut b = load(&mut gpu, 300);
    let toks = tokens(320);
    prewarm(&mut b, &mut gpu, &toks, 300);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..130]);
    let mut l = Vec::new();
    let mut p = DecodePm4::new(true, false);
    let e = p
        .forward(&mut b, &mut gpu, toks[130], 130, &mut l)
        .expect_err("no opt-in");
    eprintln!("  refused: {e}");
    assert_eq!(b.gpu_state.as_ref().unwrap().seq_pos, 130);
    let mut d = DecodePm4::new(false, false);
    let r = d
        .forward(&mut b, &mut gpu, toks[130], 130, &mut l)
        .expect("disabled executor is eager");
    assert_eq!(r.route, Pm4Route::Eager(EagerWhy::Disabled));
    assert_eq!(b.gpu_state.as_ref().unwrap().seq_pos, 131);
}

type RecordedSig = Vec<(String, [u32; 3], [u32; 3], u32, Vec<u8>)>;

fn recorded_sig(gpu: &Gpu) -> RecordedSig {
    gpu.replay
        .recorded_launches()
        .iter()
        .map(|l| {
            (
                l.kernel.clone(),
                l.grid,
                l.block,
                l.shared_mem,
                l.kernarg.clone(),
            )
        })
        .collect()
}

/// Two recordings of the same bucket at different positions, with different
/// staged contents, on the same allocations: identical kernel symbol, grid,
/// block, LDS and padded kernarg bytes for all 1024 launches (no position or
/// token baked), plus the capture census with the unique-symbol set.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn pm4_recordings_are_byte_identical_and_census() {
    let mut gpu = Gpu::init().expect("gpu");
    assert!(pm4_dev_opt_in(&gpu));
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    let mut l = Vec::new();
    let mut p = DecodePm4::new(true, false);
    let r = p.forward(&mut b, &mut gpu, toks[140], 140, &mut l).unwrap();
    assert_eq!(r.route, Pm4Route::Warm);
    let r = p.forward(&mut b, &mut gpu, toks[141], 141, &mut l).unwrap();
    assert_eq!(r.route, Pm4Route::Recorded);
    let sig_a = recorded_sig(&gpu);
    let syms_a: Vec<String> = p.recorded_symbols().to_vec();
    assert_eq!(sig_a.len(), EXPECT_NODES);
    assert_eq!(gpu.replay.retained_body_launches(), EXPECT_NODES);
    // Kernels whose by-value structs embed device pointers (FzStates/FzQRows,
    // the K/V plane tables) record with unknown effects: untyped, serialized.
    let embedded = [
        "inkling2_fz_mix_norm_rot_f32",
        "inkling2_attn_fa2_partial_pos_gfx1201",
        "inkling2_fz_attn_post_decode_f32",
    ];
    let mut n_embedded = 0;
    for l in gpu.replay.recorded_launches() {
        if embedded.contains(&l.kernel.as_str()) {
            n_embedded += 1;
            assert!(
                l.binding_layout.is_none(),
                "{} must not carry a typed (partial) effect set",
                l.kernel
            );
        }
    }
    assert!(
        n_embedded > 0,
        "tape has none of the embedded-pointer kernels"
    );
    let (typed, untyped) = gpu.replay.binding_layout_summary();
    eprintln!(
        "decode_replay_hw pm4 effects: {typed} typed / {untyped} untyped launches ({n_embedded} embedded-pointer launches forced untyped)"
    );
    p.release(&mut gpu).unwrap();
    assert!(
        gpu.replay.recorded_launches().is_empty(),
        "re-armed: recorder cleared"
    );
    // Different position, token and staged record; the same load and bucket.
    let r = p.forward(&mut b, &mut gpu, toks[142], 142, &mut l).unwrap();
    assert_eq!(r.route, Pm4Route::Recorded);
    let sig_c = recorded_sig(&gpu);
    assert_eq!(sig_c.len(), sig_a.len());
    for (i, (x, y)) in sig_a.iter().zip(&sig_c).enumerate() {
        assert_eq!(x.0, y.0, "launch {i}: kernel symbol");
        assert_eq!(x.1, y.1, "launch {i} {}: grid", x.0);
        assert_eq!(x.2, y.2, "launch {i} {}: block", x.0);
        assert_eq!(x.3, y.3, "launch {i} {}: LDS", x.0);
        assert_eq!(x.4, y.4, "launch {i} {}: kernarg bytes", x.0);
    }
    assert_eq!(syms_a, p.recorded_symbols());
    let bytes: usize = sig_a.iter().map(|s| s.4.len()).sum();
    eprintln!(
        "decode_replay_hw pm4 witness: {} launches / {bytes} B kernargs + geometry identical across recordings at pos 141 and 142; {} unique symbols:",
        sig_a.len(),
        syms_a.len()
    );
    for s in &syms_a {
        eprintln!("    {s}");
    }
    p.release(&mut gpu).unwrap();
}
