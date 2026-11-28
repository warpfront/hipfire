// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The served decode wrapper (`decode_serve::forward_plain_ar_gpu`) on the
//! typed-XTS artifact, gfx1201: the replay-vs-eager state oracle (C1) through
//! the entry the generation loop and `bench_decode` call.
//!
//! Both arms are the SAME wrapper: the eager arm runs it with the existing
//! kill switch (`graph_forward = Some(false)`, i.e. `HIPFIRE_GRAPH=0`), which
//! must select the incumbent `forward_decode_gpu`; the graph arm runs it with
//! the product default (`graph_forward = None`). Every compared step starts
//! both arms from the IDENTICAL pre-step state (conv, K/V rows at the write
//! slot, host cursors saved and restored; FA2 partials and the whole arena
//! slab poisoned with a changing pattern before each arm). Compared bit for
//! bit after each step: the full unpadded-vocab logits, all eight K and eight
//! V F16 planes of all 48 visits at the write slot, the `[3, 6656]` conv of
//! every visit, every `visit_cap`, `visit_pos_tags`, `seq_pos`, and (witness
//! steps) a hash of every other live K/V row. Route proof per step from the
//! wrapper's own ledger plus the hip-bridge counters: a replay is exactly one
//! `hipGraphLaunch` and zero module launches; a bucket arm is the eager warm
//! token plus the recording (no graph launch).
//!
//! ```sh
//! INKLING2_MQ4=.../inkling2-mini-xts-c.hfq \
//!   cargo test --release -p hipfire-arch-inkling2 --test decode_serve_hw -- \
//!   --ignored --nocapture --test-threads=1 --exact <name>
//! ```
//! (`--exact`: the shared `oracle` module also registers `decode_step_hw`'s
//! own tests in this binary; and `HIPFIRE_GRAPH` / `HIPFIRE_AR_GRAPH` must not
//! be `0` in the environment: the product default is what is under test.)

#[path = "decode_step_hw.rs"]
#[allow(dead_code)]
mod oracle;

use hip_bridge::{launch_counters, memory_effects};
use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::decode_replay::{pm4_dev_opt_in, EagerWhy, Pm4Route, StepRoute};
use hipfire_arch_inkling2::forward::{forward_token_cached, lm_head_logits, CpuDecodeState};
use hipfire_arch_inkling2::gpu::upload;
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{
    admit_row, ensure_decode_arena, forward_decode_gpu, forward_plain_ar_gpu, forward_prefill_gpu,
    plan_decode_step, release_decode_graph, serve_stats, DecodeBackend, DecodeStepKey, EagerReason,
    PrefillLogits, PrefillRoutes, ServeRoute,
};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use oracle::{
    compare_post, f32_bytes, poison, post_state, prefill, reset, restore, save_pre, sha, tokens,
    witness, Receipt,
};
use rdna_compute::Gpu;

const EXPECT_NODES: usize = 1024;

/// `HIPFIRE_AR_GRAPH` / `HIPFIRE_GRAPH` as the product resolves them, replaced
/// on the device's shared flag set.
fn set_graph_flags(gpu: &mut Gpu, graph_ar: bool, graph_forward: Option<bool>) {
    let mut f = (*gpu.flags).clone();
    f.graph_ar = graph_ar;
    f.graph_forward = graph_forward;
    gpu.flags = std::sync::Arc::new(f);
}

/// The graph tests run with the replay controller on HIP
/// (`HIPFIRE_REPLAY_BACKEND` unset / `hip`): the route the graph owns.
fn init_gpu() -> Gpu {
    let gpu = init_gpu_any();
    assert!(
        !pm4_dev_opt_in(&gpu),
        "the graph tests need HIPFIRE_REPLAY_BACKEND unset or hip"
    );
    gpu
}

/// The PM4 tests run with `HIPFIRE_REPLAY_BACKEND=redline` (PM4 transport).
fn init_gpu_pm4() -> Gpu {
    let gpu = init_gpu_any();
    assert!(
        pm4_dev_opt_in(&gpu),
        "the PM4 tests need HIPFIRE_REPLAY_BACKEND=redline (PM4 transport) on gfx1201"
    );
    gpu
}

fn init_gpu_any() -> Gpu {
    for var in ["HIPFIRE_GRAPH", "HIPFIRE_AR_GRAPH"] {
        assert_ne!(
            std::env::var(var).as_deref(),
            Ok("0"),
            "{var}=0 is the kill switch; this oracle tests the product default"
        );
    }
    let gpu = Gpu::init().expect("gpu");
    assert!(
        gpu.flags.graph_ar && gpu.flags.graph_forward != Some(false),
        "the product default must not be killed: graph_ar {} graph_forward {:?}",
        gpu.flags.graph_ar,
        gpu.flags.graph_forward
    );
    gpu
}

/// The same loader as the oracle's, without its `HIPFIRE_GRAPH=0` requirement.
fn load(gpu: &mut Gpu, max_seq: usize) -> Inkling2Bundle {
    let art = std::env::var("INKLING2_MQ4").expect("set INKLING2_MQ4 to the typed-XTS .hfq");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("weights");
    let (gw, gs) = upload(&weights, &cfg, gpu, max_seq).expect("upload");
    Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    }
}

/// The product wrapper with the kill switch on (eager) or off (graph).
fn served(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    tok: u32,
    pos: usize,
    graph: bool,
) -> Result<Vec<f32>, String> {
    set_graph_flags(gpu, true, if graph { None } else { Some(false) });
    let mut l = Vec::new();
    let r = forward_plain_ar_gpu(b, gpu, tok, pos, &mut l);
    set_graph_flags(gpu, true, None);
    r.map(|()| l)
}

fn last_route(b: &Inkling2Bundle) -> ServeRoute {
    serve_stats(b)
        .expect("a served step ran")
        .last_route
        .expect("last route")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expect {
    /// Replay iff the test's own copy of the last captured key equals the plan's.
    Auto,
    Replay,
    Captured,
}

struct Rig {
    rec: Receipt,
    salt: u32,
    key: Option<DecodeStepKey>,
    replays: usize,
    captured: usize,
    /// The PM4 route is under test (warm / record / replay), not the graph.
    pm4: bool,
    /// The PM4 executor's eager warm token ran.
    warmed: bool,
}

impl Rig {
    fn new(name: &str) -> Self {
        Self {
            rec: Receipt::new(name),
            salt: 0,
            key: None,
            replays: 0,
            captured: 0,
            pm4: false,
            warmed: false,
        }
    }

    fn new_pm4(name: &str) -> Self {
        Self {
            pm4: true,
            ..Self::new(name)
        }
    }
}

fn route_name(r: &ServeRoute) -> &'static str {
    match r {
        ServeRoute::Graph(StepRoute::Replay) => "replay",
        ServeRoute::Graph(StepRoute::EagerCaptured) => "eager+capture",
        ServeRoute::Graph(StepRoute::Eager(EagerWhy::Sticky(_))) => "graph-eager(sticky)",
        ServeRoute::Graph(StepRoute::Eager(_)) => "graph-eager",
        ServeRoute::Pm4(Pm4Route::Replay) => "pm4-replay",
        ServeRoute::Pm4(Pm4Route::Recorded) => "pm4-record",
        ServeRoute::Pm4(Pm4Route::Warm) => "pm4-warm",
        ServeRoute::Pm4(Pm4Route::Eager(_)) => "pm4-eager",
        ServeRoute::Eager(EagerReason::Killed) => "eager(killed)",
        ServeRoute::Eager(_) => "eager",
    }
}

/// One compared step at `pos`; leaves the state post-step (the graph arm's).
fn step(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rig: &mut Rig,
    tok: u32,
    pos: usize,
    wit: bool,
    expect: Expect,
) -> ServeRoute {
    ensure_decode_arena(gpu, b).expect("arena");
    let plan = plan_decode_step(b, gpu, pos).expect("plan");
    let bucket = (plan.key.bucket.lo, plan.key.bucket.hi);
    let want_replay = match expect {
        Expect::Auto => rig.key.as_ref() == Some(&plan.key),
        Expect::Replay => true,
        Expect::Captured => false,
    };
    let pre = save_pre(b, gpu, pos);
    let w_pre = wit.then(|| witness(b, gpu, pos));

    // Arm 1: the wrapper with the kill switch = the incumbent eager decode.
    rig.salt += 1;
    poison(b, gpu, rig.salt);
    let g0 = launch_counters::graph_launch::count();
    let le = served(b, gpu, tok, pos, false)
        .unwrap_or_else(|e| panic!("pos {pos}: eager arm failed: {e}"));
    assert_eq!(
        last_route(b),
        ServeRoute::Eager(EagerReason::Killed),
        "pos {pos}: the kill switch must select the incumbent eager decode"
    );
    assert_eq!(
        launch_counters::graph_launch::count(),
        g0,
        "pos {pos}: the kill switch launched a graph"
    );
    let pe = post_state(b, gpu, pos, le);
    let w_e = wit.then(|| witness(b, gpu, pos));

    // Arm 2: the wrapper, product default, from the same pre-state.
    restore(b, gpu, &pre);
    rig.salt += 1;
    poison(b, gpu, rig.salt);
    let s0 = serve_stats(b).unwrap();
    let g0 = launch_counters::graph_launch::count();
    let obs0 = gpu.replay.replay_observation();
    let fx0 = memory_effects::snapshot();
    let lg = served(b, gpu, tok, pos, true)
        .unwrap_or_else(|e| panic!("pos {pos}: graph arm failed: {e}"));
    let fxg = memory_effects::snapshot().since(fx0);
    let glaunch = launch_counters::graph_launch::count() - g0;
    let s1 = serve_stats(b).unwrap();
    let route = last_route(b);
    if rig.pm4 {
        let obs = gpu.replay.replay_observation();
        assert_eq!(glaunch, 0, "pos {pos}: the PM4 route launches no graph");
        if want_replay {
            assert_eq!(
                route,
                ServeRoute::Pm4(Pm4Route::Replay),
                "pos {pos} bucket {bucket:?}: expected a replay of the retained PM4 tape"
            );
            assert_eq!(route.backend(), DecodeBackend::Pm4);
            assert_eq!(
                fxg.launches, 0,
                "pos {pos}: a PM4 replay issues no HIP module launch"
            );
            assert_eq!(
                (fxg.htod, fxg.dtoh, fxg.dtod, fxg.memset),
                (1, 1, 0, 0),
                "pos {pos}: PM4 replay effects {fxg:?}"
            );
            assert_eq!(s1.counts.pm4_replays, s0.counts.pm4_replays + 1);
            assert_eq!(
                (obs.count, obs.last_position, obs.failed),
                (obs0.count + 1, Some(pos), false),
                "pos {pos}: retained replay observation {obs:?}"
            );
            rig.replays += 1;
        } else if !rig.warmed {
            assert_eq!(
                route,
                ServeRoute::Pm4(Pm4Route::Warm),
                "pos {pos}: the load's first PM4 step is the eager warm token"
            );
            assert_eq!(s1.counts.pm4_warm_steps, s0.counts.pm4_warm_steps + 1);
            rig.warmed = true;
        } else {
            assert_eq!(
                route,
                ServeRoute::Pm4(Pm4Route::Recorded),
                "pos {pos} bucket {bucket:?}: expected the recording token of a new key"
            );
            assert_eq!(route.backend(), DecodeBackend::Eager);
            assert_eq!(
                fxg.launches, EXPECT_NODES as u64,
                "pos {pos}: the recording token is one real 1024-launch body"
            );
            assert_eq!(s1.counts.pm4_record_steps, s0.counts.pm4_record_steps + 1);
            assert_eq!(s1.pm4.recordings, s0.pm4.recordings + 1);
            assert_eq!(s1.pm4.nodes, EXPECT_NODES, "pos {pos}: 1024-dispatch tape");
            rig.key = Some(plan.key.clone());
            rig.captured += 1;
        }
        assert_eq!(s1.pm4.record_failures, 0, "pos {pos}: no record failure");
    } else if want_replay {
        assert_eq!(
            route,
            ServeRoute::Graph(StepRoute::Replay),
            "pos {pos} bucket {bucket:?}: expected a replay of the retained graph"
        );
        assert_eq!(route.backend(), DecodeBackend::Graph);
        assert_eq!(
            glaunch, 1,
            "pos {pos}: a replay is exactly one hipGraphLaunch"
        );
        assert_eq!(
            fxg.launches, 0,
            "pos {pos}: a replay issues no module launch"
        );
        assert_eq!(
            (fxg.htod, fxg.dtoh, fxg.dtod, fxg.memset),
            (1, 1, 0, 0),
            "pos {pos}: replay effects {fxg:?}"
        );
        assert_eq!(s1.counts.graph_replays, s0.counts.graph_replays + 1);
        rig.replays += 1;
    } else {
        assert_eq!(
            route,
            ServeRoute::Graph(StepRoute::EagerCaptured),
            "pos {pos} bucket {bucket:?}: expected eager token + capture"
        );
        assert_eq!(route.backend(), DecodeBackend::Eager);
        assert_eq!(glaunch, 0, "pos {pos}: capture launches nothing");
        assert_eq!(s1.counts.graph_arm_steps, s0.counts.graph_arm_steps + 1);
        assert_eq!(s1.graph.captures, s0.graph.captures + 1);
        assert_eq!(s1.graph.nodes, EXPECT_NODES, "pos {pos}: 1024-launch body");
        rig.key = Some(plan.key.clone());
        rig.captured += 1;
    }
    assert_eq!(
        serve_stats(b).unwrap().graph.capture_failures,
        0,
        "pos {pos}: no capture failure"
    );
    let pg = post_state(b, gpu, pos, lg);
    let w_g = wit.then(|| witness(b, gpu, pos));
    if let Some(d) = compare_post(&pe, &pg) {
        let d = d.replace("stock", "eager").replace("frozen", "served");
        panic!(
            "MISMATCH pos {pos} ({}): served graph vs served eager first differing site: {d}",
            route_name(&route)
        );
    }
    if wit {
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
        route_name(&route),
        fxg.launches,
        sha(f32_bytes(&pe.logits))
    ));
    eprintln!(
        "  pos {pos:5} bucket [{},{}] {:13} bitwise-equal to the eager arm (logits+KV slot+conv+tags){}",
        bucket.0,
        bucket.1,
        route_name(&route),
        if wit { " witness" } else { "" }
    );
    route
}

fn segment(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    rig: &mut Rig,
    toks: &[u32],
    (start, end, first): (usize, usize, Expect),
) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..start]);
    eprintln!("segment {start}..={end} (first step expects {first:?})");
    for pos in start..=end {
        let wit = matches!(pos, 1023 | 1024 | 1025 | 2047 | 2048 | 4095 | 4096) || pos == start;
        let e = if pos == start { first } else { Expect::Auto };
        step(b, gpu, rig, toks[pos], pos, wit, e);
    }
}

/// Prime the shared scratch to its prefill high-water before the first capture.
fn prewarm(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32], max_seq: usize) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..max_seq.saturating_sub(1).min(2052)]);
}

fn run_matrix(max_seq: usize, segments: &[(usize, usize, Expect)], name: &str, pm4: bool) {
    let mut gpu = if pm4 { init_gpu_pm4() } else { init_gpu() };
    let mut b = load(&mut gpu, max_seq);
    admit_row(&b, &gpu).expect("the artifact must be in the admitted row");
    let toks = tokens(max_seq + 16);
    prewarm(&mut b, &mut gpu, &toks, max_seq);
    let mut rig = if pm4 {
        Rig::new_pm4(name)
    } else {
        Rig::new(name)
    };
    for &s in segments {
        segment(&mut b, &mut gpu, &mut rig, &toks, s);
        // The last legal row's successor and an out-of-order step are refused
        // with the state and the tape untouched.
        let last = max_seq.min(8192) - 1;
        if s.1 == last {
            refusals(&mut b, &mut gpu, s.1);
        }
    }
    bad_state_refusals(&mut b, &mut gpu, &toks);
    let st = serve_stats(&b).unwrap();
    eprintln!(
        "decode_serve_hw {name}: ALL BITWISE-EQUAL; pm4 {} replays / {} recordings / {} releases, graph {} replays / {} captures / {} destroys / {} graph launches, eager steps {} (killed arm)",
        st.counts.pm4_replays,
        st.pm4.recordings,
        st.pm4.releases,
        st.counts.graph_replays,
        st.graph.captures,
        st.graph.destroys,
        st.graph.graph_launches,
        st.counts.eager,
    );
    if pm4 {
        assert_eq!(st.counts.pm4_replays as usize, rig.replays);
        assert_eq!(st.pm4.recordings as usize, rig.captured);
        assert_eq!(st.counts.graph_replays + st.counts.graph_arm_steps, 0);
        assert!(st.pm4_sticky.is_none() && !st.pm4_quarantined);
        assert!(release_decode_graph(&mut b, &mut gpu));
        assert!(
            !gpu.replay.prepared_pm4_plan_ready(),
            "release dropped the prepared tape"
        );
    } else {
        assert_eq!(st.counts.graph_replays as usize, rig.replays);
        assert_eq!(st.counts.graph_arm_steps as usize, rig.captured);
        assert_eq!(st.graph.graph_launches, st.counts.graph_replays);
        assert_eq!(st.graph.capture_failures, 0);
        assert!(st.sticky.is_none());
        assert_eq!(st.counts.pm4_replays + st.counts.pm4_warm_steps, 0);
        assert!(release_decode_graph(&mut b, &mut gpu));
        assert!(
            gpu.graphs.graph_exec.is_none(),
            "release destroyed the graph"
        );
        assert!(
            gpu.graphs.ar_forward_blobs.is_empty(),
            "release freed the graph-owned blobs"
        );
    }
}

fn refusals(b: &mut Inkling2Bundle, gpu: &mut Gpu, last_legal: usize) {
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(gs.seq_pos, last_legal + 1);
    let before = (gs.seq_pos, gs.visit_cap.clone(), gs.visit_pos_tags.clone());
    let s0 = serve_stats(b).unwrap();
    let e = served(b, gpu, 7, last_legal + 1, true).expect_err("next position past the cap");
    eprintln!("  refused pos {} (cap): {e}", last_legal + 1);
    let e = served(b, gpu, 7, last_legal, true).expect_err("out-of-order position");
    eprintln!("  refused pos {last_legal} (out-of-order): {e}");
    let gs = b.gpu_state.as_ref().unwrap();
    assert_eq!(
        (gs.seq_pos, &gs.visit_cap, &gs.visit_pos_tags),
        (before.0, &before.1, &before.2),
        "a refusal advanced the host state"
    );
    let s1 = serve_stats(b).unwrap();
    assert_eq!(s0.counts, s1.counts, "a refusal executed nothing");
    assert_eq!(s0.graph.graph_launches, s1.graph.graph_launches);
}

fn bad_state_refusals(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32]) {
    reset(b, gpu);
    prefill(b, gpu, &toks[..40]);
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] += 1;
    let e = served(b, gpu, toks[40], 40, true).expect_err("bad visit_cap");
    eprintln!("  refused (visit_cap): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_cap[7] -= 1;
    let t = gs.visit_pos_tags[3][10];
    gs.visit_pos_tags[3][10] = 999;
    let e = served(b, gpu, toks[40], 40, true).expect_err("bad tag");
    eprintln!("  refused (tag): {e}");
    let gs = b.gpu_state.as_mut().unwrap();
    gs.visit_pos_tags[3][10] = t;
    served(b, gpu, toks[40], 41, true).expect_err("ahead");
    served(b, gpu, toks[40], 39, true).expect_err("behind");
}

/// Clipped non-power-of-two capacity 2053 through the served wrapper: 64
/// consecutive same-bucket steps from ctx 128, the 190..194 crossing, an
/// 81-step run, the 1022..1026 SWA wrap, the 2046..2052 wrap + 8->16 split
/// crossing and last legal row, a reset into a different bucket (127 -> 128),
/// and a reset into the retained bucket again.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn served_graph_vs_eager_clipped_2053() {
    use Expect::*;
    run_matrix(
        2053,
        &[
            (128, 191, Captured),
            (190, 194, Replay),
            (300, 380, Replay),
            (1022, 1026, Replay),
            (2046, 2052, Replay),
            (96, 134, Captured),
            (186, 198, Replay),
        ],
        "serve-clipped2053.jsonl",
        false,
    );
}

/// 4094..4098 across the 4095 -> 4096 16 -> 8 split-band change; 4100 refused.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn served_graph_vs_eager_4100() {
    run_matrix(
        4100,
        &[(4090, 4099, Expect::Captured)],
        "serve-cap4100.jsonl",
        false,
    );
}

/// Capacity exactly at the 1024 ring break.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn served_graph_vs_eager_1024() {
    run_matrix(
        1024,
        &[(1010, 1023, Expect::Captured), (0, 30, Expect::Captured)],
        "serve-cap1024.jsonl",
        false,
    );
}

/// Positive control: a flipped conv bit in the graph arm of a replay step is
/// reported at its first differing site (the oracle is able to fail).
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
#[should_panic(expected = "MISMATCH")]
fn planted_conv_fault_in_a_served_replay_is_caught() {
    let mut gpu = init_gpu();
    let mut b = load(&mut gpu, 300);
    let toks = tokens(320);
    prewarm(&mut b, &mut gpu, &toks, 300);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..130]);
    let mut rig = Rig::new("serve-planted.jsonl");
    step(
        &mut b,
        &mut gpu,
        &mut rig,
        toks[130],
        130,
        false,
        Expect::Captured,
    );
    // The step itself restores the pre-state before the graph arm; flip a conv
    // bit between the arms by hand in a replay-only variant of the lockstep.
    let pos = 131;
    let pre = save_pre(&b, &gpu, pos);
    poison(&mut b, &mut gpu, 5);
    let le = served(&mut b, &mut gpu, toks[pos], pos, false).unwrap();
    let pe = post_state(&b, &gpu, pos, le);
    restore(&mut b, &gpu, &pre);
    let gs = b.gpu_state.as_ref().unwrap();
    let mut bytes = gpu.download_raw_bytes(&gs.conv[20]).unwrap();
    bytes[6656 * 4 + 5 * 4 + 3] ^= 0x20;
    gpu.memcpy_htod_auto(&gs.conv[20].buf, &bytes).unwrap();
    let lg = served(&mut b, &mut gpu, toks[pos], pos, true).unwrap();
    assert_eq!(last_route(&b), ServeRoute::Graph(StepRoute::Replay));
    let pg = post_state(&b, &gpu, pos, lg);
    if let Some(d) = compare_post(&pe, &pg) {
        panic!("MISMATCH pos {pos} (replay): first differing site: {d}");
    }
}

/// The existing kill switches select the incumbent eager decode and never
/// touch a graph; an out-of-row load (here: F16 expert tier) is refused with a
/// typed reason and runs eager; flipping back re-admits.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn kill_switches_and_ineligible_rows_run_eager() {
    let mut gpu = init_gpu();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    let g0 = launch_counters::graph_launch::count();
    let mut pos = 140;

    // HIPFIRE_GRAPH=0 (flags.graph_forward = Some(false)).
    served(&mut b, &mut gpu, toks[pos], pos, false).unwrap();
    pos += 1;
    assert_eq!(last_route(&b), ServeRoute::Eager(EagerReason::Killed));
    // HIPFIRE_AR_GRAPH=0 (flags.graph_ar = false).
    set_graph_flags(&mut gpu, false, None);
    let mut l = Vec::new();
    forward_plain_ar_gpu(&mut b, &mut gpu, toks[pos], pos, &mut l).unwrap();
    set_graph_flags(&mut gpu, true, None);
    pos += 1;
    assert_eq!(last_route(&b), ServeRoute::Eager(EagerReason::Killed));
    // An explicit HIPFIRE_GRAPH=1 still admits the row (not the kill switch).
    set_graph_flags(&mut gpu, true, Some(true));
    forward_plain_ar_gpu(&mut b, &mut gpu, toks[pos], pos, &mut l).unwrap();
    set_graph_flags(&mut gpu, true, None);
    pos += 1;
    assert_eq!(
        last_route(&b),
        ServeRoute::Graph(StepRoute::EagerCaptured),
        "an explicit HIPFIRE_GRAPH=1 arms the graph"
    );
    assert_eq!(launch_counters::graph_launch::count(), g0);

    // Out-of-row: the F16 expert tier is not A8. Typed reason, exact eager
    // logits (it IS the incumbent), the armed tape untouched.
    let armed = serve_stats(&b).unwrap().graph.clone();
    let epoch0 = rdna_compute::scratch::scratch_growth_epoch();
    let saved = b.gpu_state.as_ref().unwrap().prefill_routes;
    b.gpu_state.as_mut().unwrap().prefill_routes =
        PrefillRoutes::parse("dense=f16,experts=f16").unwrap();
    let want = {
        let pre = save_pre(&b, &gpu, pos);
        let l = forward_decode_gpu(&mut b, &mut gpu, toks[pos], pos).unwrap();
        restore(&mut b, &gpu, &pre);
        l
    };
    forward_plain_ar_gpu(&mut b, &mut gpu, toks[pos], pos, &mut l).unwrap();
    b.gpu_state.as_mut().unwrap().prefill_routes = saved;
    match last_route(&b) {
        ServeRoute::Eager(EagerReason::Ineligible(why)) => {
            eprintln!("  F16 expert tier refused: {why}");
            assert!(why.contains("A8"), "{why}");
        }
        other => panic!("expected a typed ineligible reason, got {other:?}"),
    }
    assert_eq!(sha(f32_bytes(&want)), sha(f32_bytes(&l)));
    assert_eq!(
        serve_stats(&b).unwrap().graph.graph_launches,
        armed.graph_launches
    );
    pos += 1;

    // Re-admitted. The F16-tier steps may have grown the shared scratch (a new
    // layout epoch = a new key): then the tape re-arms, otherwise the retained
    // tape replays. Either way the graph backend is back and the step is exact.
    let want = {
        let pre = save_pre(&b, &gpu, pos);
        let l = served(&mut b, &mut gpu, toks[pos], pos, false).unwrap();
        restore(&mut b, &gpu, &pre);
        l
    };
    let epoch_moved = rdna_compute::scratch::scratch_growth_epoch() != epoch0;
    forward_plain_ar_gpu(&mut b, &mut gpu, toks[pos], pos, &mut l).unwrap();
    assert_eq!(
        last_route(&b),
        ServeRoute::Graph(if epoch_moved {
            StepRoute::EagerCaptured
        } else {
            StepRoute::Replay
        }),
        "re-admission: the key rule (scratch epoch moved: {epoch_moved})"
    );
    assert_eq!(sha(f32_bytes(&want)), sha(f32_bytes(&l)));
    pos += 1;
    if epoch_moved {
        forward_plain_ar_gpu(&mut b, &mut gpu, toks[pos], pos, &mut l).unwrap();
        assert_eq!(last_route(&b), ServeRoute::Graph(StepRoute::Replay));
    }
    assert!(release_decode_graph(&mut b, &mut gpu));
}

/// A prompt prefill whose last block is ONE row (the one-row tail) runs the
/// incumbent blocked prefill and never selects, arms, replays or disturbs the
/// graph; decode after it matches the eager arm.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn prefill_one_row_tail_never_selects_the_graph() {
    let mut gpu = init_gpu();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    let mut rig = Rig::new("serve-prefill-tail.jsonl");
    segment(
        &mut b,
        &mut gpu,
        &mut rig,
        &toks,
        (60, 64, Expect::Captured),
    );
    let s0 = serve_stats(&b).unwrap();
    assert!(s0.counts.graph_replays >= 3);
    let g0 = launch_counters::graph_launch::count();

    // Same prompt, prefilled with a final one-row block (forward_prefill_gpu,
    // PrefillLogits::Last), exactly as the generation loop's prompt prefill.
    reset(&mut b, &mut gpu);
    forward_prefill_gpu(&mut b, &mut gpu, &toks[..60], 0, PrefillLogits::None).unwrap();
    let fx0 = memory_effects::snapshot();
    let mut tail = Vec::new();
    forward_prefill_gpu(
        &mut b,
        &mut gpu,
        &toks[60..61],
        60,
        PrefillLogits::Last(&mut tail),
    )
    .unwrap();
    let fx = memory_effects::snapshot().since(fx0);
    assert!(fx.launches > 0, "the one-row tail ran module launches");
    assert_eq!(
        launch_counters::graph_launch::count(),
        g0,
        "the one-row prefill tail launched a graph"
    );
    let s1 = serve_stats(&b).unwrap();
    assert_eq!(s0.counts, s1.counts, "no served step: nothing selected");
    assert_eq!(s0.graph.graph_launches, s1.graph.graph_launches);
    assert_eq!(s0.graph.captures, s1.graph.captures);
    // The tail is the incumbent one-row decode, bit for bit.
    reset(&mut b, &mut gpu);
    forward_prefill_gpu(&mut b, &mut gpu, &toks[..60], 0, PrefillLogits::None).unwrap();
    let eager = forward_decode_gpu(&mut b, &mut gpu, toks[60], 60).unwrap();
    assert_eq!(sha(f32_bytes(&tail)), sha(f32_bytes(&eager)));
    // A prefill of ONE row from position 0 (a one-token prompt) likewise.
    reset(&mut b, &mut gpu);
    let g1 = launch_counters::graph_launch::count();
    forward_prefill_gpu(
        &mut b,
        &mut gpu,
        &toks[..1],
        0,
        PrefillLogits::Last(&mut tail),
    )
    .unwrap();
    assert_eq!(launch_counters::graph_launch::count(), g1);
    // Decode after the prefill is exact against the eager arm, whichever route
    // (retained tape or re-arm) the wrapper picks.
    reset(&mut b, &mut gpu);
    forward_prefill_gpu(&mut b, &mut gpu, &toks[..60], 0, PrefillLogits::None).unwrap();
    forward_prefill_gpu(&mut b, &mut gpu, &toks[60..61], 60, PrefillLogits::None).unwrap();
    for pos in 61..66 {
        step(
            &mut b,
            &mut gpu,
            &mut rig,
            toks[pos],
            pos,
            false,
            Expect::Auto,
        );
    }
    assert!(release_decode_graph(&mut b, &mut gpu));
}

/// What the fail-closed rollback does to the Gpu (`invalidate_graph_state`)
/// and the explicit release never leave the wrapper replaying a destroyed
/// graph: the next step re-arms (an eager token + a recording) and is exact.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn rollback_and_release_invalidate_the_tape_without_panic() {
    let mut gpu = init_gpu();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    let mut rig = Rig::new("serve-rollback.jsonl");
    use Expect::*;
    segment(&mut b, &mut gpu, &mut rig, &toks, (150, 153, Captured));
    // The production rollback epilogue: reset + invalidate captured graphs.
    gpu.invalidate_graph_state();
    reset(&mut b, &mut gpu);
    segment(&mut b, &mut gpu, &mut rig, &toks, (150, 153, Captured));
    // Explicit release (what free_gpu does), then the same step range.
    assert!(release_decode_graph(&mut b, &mut gpu));
    assert!(gpu.graphs.graph_exec.is_none());
    segment(&mut b, &mut gpu, &mut rig, &toks, (150, 153, Captured));
    assert!(release_decode_graph(&mut b, &mut gpu));
}

/// A capture failure after the warm token through the wrapper: that step's
/// exact eager logits, one commit, sticky eager afterwards (not cleared by a
/// reset or another bucket), the kill-free wrapper still reports why.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn capture_failure_is_sticky_through_the_wrapper() {
    let mut gpu = init_gpu();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    let want = {
        let pre = save_pre(&b, &gpu, 140);
        let l = forward_decode_gpu(&mut b, &mut gpu, toks[140], 140).unwrap();
        restore(&mut b, &gpu, &pre);
        l
    };
    gpu.graphs.host_cpu_weights = true; // begin_graph_capture refuses (injected)
    let l = served(&mut b, &mut gpu, toks[140], 140, true)
        .expect("the eager token is returned despite the capture failure");
    gpu.graphs.host_cpu_weights = false;
    match last_route(&b) {
        ServeRoute::Graph(StepRoute::Eager(EagerWhy::CaptureFailed(why))) => {
            eprintln!("  capture failed (injected): {why}")
        }
        other => panic!("expected a capture failure, got {other:?}"),
    }
    assert_eq!(sha(f32_bytes(&want)), sha(f32_bytes(&l)));
    assert_eq!(b.gpu_state.as_ref().unwrap().seq_pos, 141, "committed once");
    assert!(gpu.graphs.graph_exec.is_none());
    for pos in 141..145 {
        let pre = save_pre(&b, &gpu, pos);
        let want = forward_decode_gpu(&mut b, &mut gpu, toks[pos], pos).unwrap();
        restore(&mut b, &gpu, &pre);
        let l = served(&mut b, &mut gpu, toks[pos], pos, true).unwrap();
        assert!(matches!(
            last_route(&b),
            ServeRoute::Eager(EagerReason::Sticky(_))
        ));
        assert_eq!(sha(f32_bytes(&want)), sha(f32_bytes(&l)));
    }
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..100]);
    served(&mut b, &mut gpu, toks[100], 100, true).unwrap();
    assert!(matches!(
        last_route(&b),
        ServeRoute::Eager(EagerReason::Sticky(_))
    ));
    let st = serve_stats(&b).unwrap();
    assert_eq!((st.graph.replays, st.graph.graph_launches), (0, 0));
    assert!(release_decode_graph(&mut b, &mut gpu));
}

/// Reload: freeing the load (`ArchModel::free_gpu`) quiesces and destroys its
/// graph before the weights/state go; a fresh load starts a fresh executor
/// (no stale tape, no inherited sticky) and is bitwise the eager arm.
#[test]
#[ignore = "needs the typed-XTS artifact on a gfx1201 card"]
fn reload_releases_the_graph_and_starts_a_fresh_executor() {
    let mut gpu = init_gpu();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    let mut rig = Rig::new("serve-reload.jsonl");
    segment(
        &mut b,
        &mut gpu,
        &mut rig,
        &toks,
        (150, 154, Expect::Captured),
    );
    assert!(gpu.graphs.graph_exec.is_some(), "a graph is held");
    let blobs = gpu.graphs.ar_forward_blobs.len();
    assert_eq!(blobs, EXPECT_NODES);
    Box::new(b).free_gpu(&mut gpu);
    assert!(
        gpu.graphs.graph_exec.is_none(),
        "free_gpu destroyed the graph before the weights"
    );
    assert!(gpu.graphs.ar_forward_blobs.is_empty());

    let mut b2 = load(&mut gpu, 400);
    assert!(
        serve_stats(&b2).is_none(),
        "a new load has no inherited serving state"
    );
    prewarm(&mut b2, &mut gpu, &toks, 400);
    let mut rig2 = Rig::new("serve-reload2.jsonl");
    segment(
        &mut b2,
        &mut gpu,
        &mut rig2,
        &toks,
        (150, 154, Expect::Captured),
    );
    assert!(release_decode_graph(&mut b2, &mut gpu));
    Box::new(b2).free_gpu(&mut gpu);
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (*x as f64 - ma, *y as f64 - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    (sxy / (sxx * syy).sqrt()) as f32
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// Gate 3b (tokens 1..N logits against the CPU cached decode; same tokens,
/// reference and bars as `gpu_parity::gate3b`) through the SERVED wrapper at
/// the product default: position 0 arms the graph (eager warm token + capture),
/// positions 1..3 are graph replays, and every one must clear the same bars.
fn gate3b_served(pm4: bool) {
    const TOKENS: [u32; 4] = [5000, 12345, 777, 31274];
    const R_MIN: f32 = 0.99;
    let art = std::env::var("INKLING2_MQ4").expect("set INKLING2_MQ4");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("weights");
    let vocab = cfg.unpadded_vocab_size;
    let e = cfg.n_embd;
    let mut want = Vec::new();
    {
        let (_, emb) = weights.f32("model.llm.embed.weight").expect("embed");
        let mut st = CpuDecodeState::new(&cfg, 16).expect("cpu state");
        for (pos, &t) in TOKENS.iter().enumerate() {
            let x = &emb[t as usize * e..(t as usize + 1) * e];
            let hidden =
                forward_token_cached(&weights, &cfg, &mut st, x, pos).expect("cpu forward");
            let l = lm_head_logits(&weights, &cfg, &hidden).expect("cpu logits");
            want.push(l[..vocab].to_vec());
        }
    }
    let mut gpu = if pm4 { init_gpu_pm4() } else { init_gpu() };
    let (gw, gs) = upload(&weights, &cfg, &mut gpu, 64).expect("upload");
    let mut b = Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    };
    admit_row(&b, &gpu).expect("the artifact must be in the admitted row");
    let mut fails = Vec::new();
    for (pos, &t) in TOKENS.iter().enumerate() {
        let got = served(&mut b, &mut gpu, t, pos, true).expect("served logits");
        let route = last_route(&b);
        let w = &want[pos];
        assert_eq!(got.len(), w.len());
        let r = pearson(&got, w);
        let (am_got, am_want) = (argmax(&got), argmax(w));
        eprintln!(
            "gate3b-served: pos {pos} tok {t} [{}]: r={r:.7} argmax gpu={am_got} cpu={am_want}",
            route_name(&route)
        );
        if am_got != am_want {
            fails.push(format!("pos {pos}: argmax gpu {am_got} vs cpu {am_want}"));
        }
        if r < R_MIN {
            fails.push(format!("pos {pos}: r {r} below {R_MIN}"));
        }
    }
    let st = serve_stats(&b).unwrap();
    if pm4 {
        assert_eq!(
            (
                st.counts.pm4_warm_steps,
                st.counts.pm4_record_steps,
                st.counts.pm4_replays
            ),
            (1, 1, 2),
            "the gate must exercise PM4: warm + record + 2 replays"
        );
    } else {
        assert_eq!(
            (st.counts.graph_arm_steps, st.counts.graph_replays),
            (1, 3),
            "the gate must exercise the graph: 1 arm + 3 replays"
        );
    }
    assert!(release_decode_graph(&mut b, &mut gpu));
    Box::new(b).free_gpu(&mut gpu);
    assert!(fails.is_empty(), "gate3b-served: {}", fails.join("; "));
    eprintln!("gate3b-served: PASS");
}

#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201; slow CPU reference"]
fn gate3b_multi_token_logits_served_graph() {
    gate3b_served(false);
}

#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline; slow CPU reference"]
fn gate3b_multi_token_logits_served_pm4() {
    gate3b_served(true);
}

/// The served PM4 route through the same lockstep oracle: warm token, recording
/// token, retained PM4 replays (bucket crossings, wraps, resets), against the
/// incumbent eager decode.
#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline"]
fn served_pm4_vs_eager_clipped_2053() {
    use Expect::*;
    run_matrix(
        2053,
        &[
            (128, 191, Captured),
            (190, 194, Replay),
            (300, 380, Replay),
            (1022, 1026, Replay),
            (2046, 2052, Replay),
            (96, 134, Captured),
            (186, 198, Replay),
        ],
        "serve-pm4-clipped2053.jsonl",
        true,
    );
}

#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline"]
fn served_pm4_vs_eager_4100() {
    run_matrix(
        4100,
        &[(4090, 4099, Expect::Captured)],
        "serve-pm4-cap4100.jsonl",
        true,
    );
}

#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline"]
fn served_pm4_vs_eager_1024() {
    run_matrix(
        1024,
        &[(1010, 1023, Expect::Captured), (0, 30, Expect::Captured)],
        "serve-pm4-cap1024.jsonl",
        true,
    );
}

/// A refused PM4 record (injected: the replay controller is poisoned before
/// the first step) returns that step's exact eager token once; the load then
/// leaves PM4 (tape released, controller poisoned) and runs the HIP graph: arm,
/// then replays, every step exact against the incumbent eager decode.
#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline"]
fn pm4_refusal_falls_back_to_the_graph() {
    let mut gpu = init_gpu_pm4();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    reset(&mut b, &mut gpu);
    prefill(&mut b, &mut gpu, &toks[..140]);
    gpu.replay
        .poison("injected: controller poisoned before the first PM4 step");
    let mut pos = 140;
    // 1) the PM4 executor sees the poisoned controller: a sticky eager token.
    let want = {
        let pre = save_pre(&b, &gpu, pos);
        let l = served(&mut b, &mut gpu, toks[pos], pos, false).unwrap();
        restore(&mut b, &gpu, &pre);
        l
    };
    let l = served(&mut b, &mut gpu, toks[pos], pos, true).unwrap();
    assert!(
        matches!(
            last_route(&b),
            ServeRoute::Pm4(Pm4Route::Eager(EagerWhy::Sticky(_)))
        ),
        "{:?}",
        last_route(&b)
    );
    assert_eq!(sha(f32_bytes(&want)), sha(f32_bytes(&l)));
    assert_eq!(
        b.gpu_state.as_ref().unwrap().seq_pos,
        pos + 1,
        "committed once"
    );
    pos += 1;
    // 2) the load leaves PM4: graph arm, then replays; all exact.
    let mut rig = Rig::new("serve-pm4-fallback.jsonl");
    for (i, p) in (pos..pos + 6).enumerate() {
        let expect = if i == 0 {
            Expect::Captured
        } else {
            Expect::Auto
        };
        let r = step(&mut b, &mut gpu, &mut rig, toks[p], p, false, expect);
        assert!(matches!(r, ServeRoute::Graph(_)), "{r:?}");
    }
    let st = serve_stats(&b).unwrap();
    assert!(st.pm4_sticky.is_some());
    assert!(!st.pm4_quarantined);
    assert_eq!((st.counts.graph_arm_steps, st.counts.graph_replays), (1, 5));
    assert_eq!(st.counts.pm4_replays, 0);
    assert!(release_decode_graph(&mut b, &mut gpu));
}

/// Prompt prefill (including a final one-row block) never selects PM4: no
/// retained replay observation, no served step, and decode afterwards is exact.
#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline"]
fn pm4_prefill_one_row_tail_never_selects_the_route() {
    let mut gpu = init_gpu_pm4();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    let mut rig = Rig::new_pm4("serve-pm4-prefill-tail.jsonl");
    segment(
        &mut b,
        &mut gpu,
        &mut rig,
        &toks,
        (60, 65, Expect::Captured),
    );
    let s0 = serve_stats(&b).unwrap();
    let obs0 = gpu.replay.replay_observation();
    assert!(s0.counts.pm4_replays >= 3);
    reset(&mut b, &mut gpu);
    forward_prefill_gpu(&mut b, &mut gpu, &toks[..60], 0, PrefillLogits::None).unwrap();
    let mut tail = Vec::new();
    forward_prefill_gpu(
        &mut b,
        &mut gpu,
        &toks[60..61],
        60,
        PrefillLogits::Last(&mut tail),
    )
    .unwrap();
    let s1 = serve_stats(&b).unwrap();
    assert_eq!(s0.counts, s1.counts, "no served step: nothing selected");
    let obs1 = gpu.replay.replay_observation();
    assert_eq!(
        (obs0.count, obs0.last_position),
        (obs1.count, obs1.last_position),
        "the prefill tail issued a retained PM4 replay"
    );
    reset(&mut b, &mut gpu);
    forward_prefill_gpu(&mut b, &mut gpu, &toks[..60], 0, PrefillLogits::None).unwrap();
    let eager = forward_decode_gpu(&mut b, &mut gpu, toks[60], 60).unwrap();
    assert_eq!(sha(f32_bytes(&tail)), sha(f32_bytes(&eager)));
    reset(&mut b, &mut gpu);
    forward_prefill_gpu(&mut b, &mut gpu, &toks[..61], 0, PrefillLogits::None).unwrap();
    for pos in 61..66 {
        step(
            &mut b,
            &mut gpu,
            &mut rig,
            toks[pos],
            pos,
            false,
            Expect::Auto,
        );
    }
    assert!(release_decode_graph(&mut b, &mut gpu));
}

/// Freeing the load (`ArchModel::free_gpu`) proves the queue idle and drops the
/// prepared tape before the weights go; a fresh load starts a fresh executor
/// (warm, record, replay) and is exact.
#[test]
#[ignore = "needs the typed-XTS artifact + gfx1201 + HIPFIRE_REPLAY_BACKEND=redline"]
fn pm4_reload_releases_the_tape_and_starts_fresh() {
    let mut gpu = init_gpu_pm4();
    let mut b = load(&mut gpu, 400);
    let toks = tokens(420);
    prewarm(&mut b, &mut gpu, &toks, 400);
    let mut rig = Rig::new_pm4("serve-pm4-reload.jsonl");
    segment(
        &mut b,
        &mut gpu,
        &mut rig,
        &toks,
        (150, 155, Expect::Captured),
    );
    assert!(gpu.replay.prepared_pm4_plan_ready(), "a PM4 tape is held");
    Box::new(b).free_gpu(&mut gpu);
    assert!(
        !gpu.replay.prepared_pm4_plan_ready(),
        "free_gpu released the tape before the weights"
    );
    let mut b2 = load(&mut gpu, 400);
    assert!(serve_stats(&b2).is_none(), "no inherited serving state");
    prewarm(&mut b2, &mut gpu, &toks, 400);
    let mut rig2 = Rig::new_pm4("serve-pm4-reload2.jsonl");
    segment(
        &mut b2,
        &mut gpu,
        &mut rig2,
        &toks,
        (150, 155, Expect::Captured),
    );
    assert!(release_decode_graph(&mut b2, &mut gpu));
    Box::new(b2).free_gpu(&mut gpu);
}
