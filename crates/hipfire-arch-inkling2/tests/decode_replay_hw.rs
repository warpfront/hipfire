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
use hipfire_arch_inkling2::decode_replay::{graph_dev_opt_in, DecodeGraph, EagerWhy, StepRoute};
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
