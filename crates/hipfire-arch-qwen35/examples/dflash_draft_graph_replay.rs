// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate: `dflash_draft_graph_replay`.
//!
//! Proves the stream-ordered DFlash draft-body D2D copies
//! (`HIPFIRE_DFLASH_ORDERED_DRAFT`, default ON, `=0` restores the synchronous
//! null-stream `hipMemcpy` copies) are legal hipGraph copy nodes that replay
//! byte-identically, and that the switch changes no bytes:
//!
//! 1. ON/OFF eager identity: a singleton `draft_forward_opts` (b=16, l=48)
//!    and a 3-lane `draft_forward_lanes` (b=16 each, contexts 24/40/56) produce
//!    byte-identical planes (final x, K/V concat, projected context, every
//!    per-layer K/V context cache) in both arms.
//! 2. lanes == singleton: each lane's rows of the shared `x` equal a singleton
//!    `draft_forward_opts` of the same inputs, byte for byte, in both arms.
//! 3. Capture/replay: the forward is captured once on the explicit stream
//!    (`stream_begin_capture` .. `graph_instantiate`) and replayed three times
//!    (the block's noise rows are restored before every replay); every
//!    replay's planes must equal the eager planes byte for byte. Singleton and
//!    lanes are each captured. Arm ON must capture and replay clean; arm OFF
//!    is expected to refuse the capture (synchronous null-stream copies are
//!    illegal while capturing) and is only reported.
//!
//! Capture bakes the HOST addresses of the position arrays into the graph
//! (`memcpy_htod_auto` -> async H2D from the caller's slice), so the harness
//! keeps every `pos_q`/`pos_k` Vec alive and unchanged until the last replay.
//!
//! The batched lane draft requires the non-fused draft route. On gfx1100 run
//! with `HIPFIRE_DRAFT_COLLAPSE_OFF=1` (inherited by the children), otherwise
//! `draft_forward_lanes` refuses and the lanes step is reported as failed.
//!
//! Commands (parent re-execs itself twice, ON then OFF, sequentially):
//!
//! ```text
//! cargo run --release -p hipfire-arch-qwen35 --features deltanet \
//!     --example dflash_draft_graph_replay -- <draft.hfq> <outdir>
//! # single arm (internal; arm picked from HIPFIRE_DFLASH_ORDERED_DRAFT):
//! HIPFIRE_DFLASH_ORDERED_DRAFT=1 dflash_draft_graph_replay <draft.hfq> <outdir> child
//! ```
//!
//! Output: `<outdir>/{on,off}/{single_eager,lanes_eager}/<plane>.f32` + MANIFEST
//! (`name len fnv1a64`), `<outdir>/<arm>/result.json`, `<outdir>/summary.json`.
//! Exit 0 on PASS, 1 on FAIL.

use hip_bridge::{Graph, GraphExec, HipResult};
use hipfire_runtime::dflash::{
    draft_forward_lanes, draft_forward_opts, ring_segments, DflashConfig, DflashScratch,
    DflashWeights, DraftLane,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use std::path::Path;

const ENV_FLAG: &str = "HIPFIRE_DFLASH_ORDERED_DRAFT";
const REPLAYS: usize = 3;
/// Singleton geometry: b=16 as the collapse gate. Every context is below 64
/// rows so no GEMM (the fc projection runs over all `l` rows) reaches the
/// >= 64-row eager-only FP8/IU4 routes that a capture swaps for the FP16
/// WMMA route (railgun-cert `prefill-producer-fusions`, `eager-only-refusals`):
/// eager and replay then launch the same kernels, as the steady draft does
/// (fc over the accepted rows, <= 16, plus <= 48 block rows).
const B: usize = 16;
const CTX_CAP: usize = 256;
const L_SINGLE: usize = 48;
/// Lane geometry: three 16-row lanes; shared scratch = DFLASH_DRAFT_BATCH_MAX_ROWS (48).
const LANE_CTX: [usize; 3] = [24, 40, 56];
const SHARED_ROWS: usize = 48;

type Planes = Vec<(String, Vec<f32>)>;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_f32(state: &mut u64) -> f32 {
    // Uniform in [-1, 1).
    let u = (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64;
    (u * 2.0 - 1.0) as f32
}

fn f32_slice_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ── Scratch / inputs ────────────────────────────────────────────────────

/// Per-lane / singleton scratch, same construction as the collapse gate and
/// the production slot (`dflash_slot.rs`).
fn build_scratch(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
) -> HipResult<DflashScratch> {
    if let Some(w) = cfg.declared_window {
        let w_full = if cfg.all_layers_sliding { w } else { CTX_CAP };
        DflashScratch::new_windowed(gpu, cfg, B, w, w_full, CTX_CAP, weights.has_mq)
    } else {
        DflashScratch::new_with_mq(gpu, cfg, B, CTX_CAP, weights.has_mq)
    }
}

/// Shared row scratch of the batched lane draft, as the production
/// continuous-batching store builds it (`dflash_cb.rs`:
/// `new_with_mq(DFLASH_DRAFT_BATCH_MAX_ROWS, 1, true)`).
fn build_shared(gpu: &mut Gpu, cfg: &DflashConfig) -> HipResult<DflashScratch> {
    DflashScratch::new_with_mq(gpu, cfg, SHARED_ROWS, 1, true)
}

// Poison every data tensor in a scratch with 0xCD bytes (thlog/graph
// caches intentionally untouched — structural state, not data).
fn poison_scratch(gpu: &Gpu, s: &DflashScratch) -> HipResult<()> {
    let mut all: Vec<&GpuTensor> = vec![
        &s.x,
        &s.x_norm,
        &s.q,
        &s.k_noise,
        &s.v_noise,
        &s.gate,
        &s.up,
        &s.gate_up,
        &s.attn_out,
        &s.residual,
        &s.target_hidden,
        &s.target_hidden_proj,
        &s.k_cat,
        &s.v_cat,
        &s.positions_q,
        &s.positions_k,
        &s.noise_tokens,
    ];
    for t in [&s.mq_x_rot, &s.mq_x_rot_f16].into_iter().flatten() {
        all.push(t);
    }
    for t in [
        &s.conv_temp,
        &s.conv_dynamic,
        &s.selector_proj,
        &s.topk_ids,
        &s.topk_vals,
    ]
    .into_iter()
    .flatten()
    {
        all.push(t);
    }
    for t in s.k_ctx_cached.iter().chain(s.v_ctx_cached.iter()) {
        all.push(t);
    }
    for t in [
        &s.k_full_cached,
        &s.v_full_cached,
        &s.k_cat_full,
        &s.v_cat_full,
    ]
    .into_iter()
    .flatten()
    {
        all.push(t);
    }
    for t in all {
        gpu.hip.memset(&t.buf, 0xCD, t.buf.size())?;
    }
    Ok(())
}

/// Deterministic synthetic inputs of one block (singleton or lane).
struct Inputs {
    b: usize,
    l: usize,
    noise: Vec<f32>,
    th: Vec<f32>,
    // Kept alive and unchanged for the whole run: a captured graph bakes the
    // host addresses of these arrays (and of the slices the draft forward
    // uploads from them).
    pos_q: Vec<i32>,
    pos_k: Vec<i32>,
}

fn gen_inputs(seed: u64, b: usize, l: usize, base: i32, h: usize, ne: usize) -> Inputs {
    let mut rng = seed;
    let noise: Vec<f32> = (0..b * h).map(|_| rand_f32(&mut rng)).collect();
    let th: Vec<f32> = (0..l * ne * h).map(|_| rand_f32(&mut rng) * 0.5).collect();
    let pos_q: Vec<i32> = (0..b).map(|i| base + i as i32).collect();
    let pos_k: Vec<i32> = (0..l + b).map(|i| base - l as i32 + i as i32).collect();
    Inputs {
        b,
        l,
        noise,
        th,
        pos_q,
        pos_k,
    }
}

/// Upload `th` rows `[0, l)` into the scratch's target-hidden buffer. A
/// windowed scratch holds only a `cap_rows`-row ring (slot = row % cap_rows):
/// then only the last `cap_rows` rows are uploaded, wrap-split, exactly like
/// `draft_forward_opts`' own upload.
fn upload_th(gpu: &Gpu, s: &DflashScratch, th: &[f32], l: usize, row: usize) -> HipResult<()> {
    let cap_rows = s.target_hidden.buf.size() / (row * 4);
    if l <= cap_rows {
        return gpu
            .hip
            .memcpy_htod(&s.target_hidden.buf, f32_slice_bytes(&th[..l * row]));
    }
    for (row0, slot0, len) in ring_segments(l - cap_rows, l, cap_rows) {
        gpu.hip.memcpy_htod_offset(
            &s.target_hidden.buf,
            slot0 * row * 4,
            f32_slice_bytes(&th[row0 * row..(row0 + len) * row]),
        )?;
    }
    Ok(())
}

/// Entry planes of the singleton: noise into `x`, context into `target_hidden`.
/// Positions are NOT pre-uploaded: `draft_forward_opts` uploads them itself
/// (and a capture must see that upload).
fn upload_inputs(gpu: &Gpu, s: &DflashScratch, inp: &Inputs, row: usize) -> HipResult<()> {
    gpu.hip.memcpy_htod(&s.x.buf, f32_slice_bytes(&inp.noise))?;
    upload_th(gpu, s, &inp.th, inp.l, row)
}

// ── Plane collection / comparison / manifests ───────────────────────────

fn collect_scratch_planes(
    gpu: &Gpu,
    s: &DflashScratch,
    prefix: &str,
    with_final_x: bool,
    out: &mut Planes,
) -> HipResult<()> {
    let mut add = |name: String, t: &GpuTensor| -> HipResult<()> {
        out.push((name, gpu.download_f32(t)?));
        Ok(())
    };
    if with_final_x {
        add(format!("{prefix}final_x"), &s.x)?;
    }
    add(format!("{prefix}k_cat"), &s.k_cat)?;
    add(format!("{prefix}v_cat"), &s.v_cat)?;
    add(format!("{prefix}target_hidden_proj"), &s.target_hidden_proj)?;
    for (li, t) in s.k_ctx_cached.iter().enumerate() {
        add(format!("{prefix}k_ctx_cached_{li}"), t)?;
    }
    for (li, t) in s.v_ctx_cached.iter().enumerate() {
        add(format!("{prefix}v_ctx_cached_{li}"), t)?;
    }
    for (name, t) in [
        ("k_full_cached", &s.k_full_cached),
        ("v_full_cached", &s.v_full_cached),
        ("k_cat_full", &s.k_cat_full),
        ("v_cat_full", &s.v_cat_full),
    ] {
        if let Some(t) = t {
            add(format!("{prefix}{name}"), t)?;
        }
    }
    Ok(())
}

fn collect_single(gpu: &Gpu, s: &DflashScratch) -> HipResult<Planes> {
    let mut out = Planes::new();
    collect_scratch_planes(gpu, s, "", true, &mut out)?;
    Ok(out)
}

fn collect_lanes(gpu: &Gpu, shared: &DflashScratch, lanes: &[DflashScratch]) -> HipResult<Planes> {
    let mut out = Planes::new();
    out.push(("shared_x".to_string(), gpu.download_f32(&shared.x)?));
    for (i, s) in lanes.iter().enumerate() {
        collect_scratch_planes(gpu, s, &format!("lane{i}_"), false, &mut out)?;
    }
    Ok(out)
}

fn find_plane<'a>(planes: &'a Planes, name: &str) -> &'a [f32] {
    &planes
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("plane {name} missing"))
        .1
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Names of planes of `a` that are missing, resized or differ bitwise in `b`
/// (plus planes only present in `b`).
fn diff_planes(a: &Planes, b: &Planes) -> Vec<String> {
    let mut bad = Vec::new();
    for (name, va) in a {
        match b.iter().find(|(n, _)| n == name) {
            Some((_, vb)) if bits_equal(va, vb) => {}
            _ => bad.push(name.clone()),
        }
    }
    for (name, _) in b {
        if !a.iter().any(|(n, _)| n == name) {
            bad.push(name.clone());
        }
    }
    bad
}

fn planes_json(planes: &Planes) -> Value {
    Value::Array(
        planes
            .iter()
            .map(|(name, v)| {
                json!({
                    "name": name,
                    "len": v.len(),
                    "fnv1a64": format!("{:016x}", fnv1a64(f32_slice_bytes(v))),
                })
            })
            .collect(),
    )
}

fn write_planes(dir: &Path, planes: &Planes) {
    std::fs::create_dir_all(dir).expect("mkdir plane dir");
    let mut manifest = String::new();
    for (name, v) in planes {
        std::fs::write(dir.join(format!("{name}.f32")), f32_slice_bytes(v)).expect("write plane");
        manifest.push_str(&format!(
            "{name} {} {:016x}\n",
            v.len(),
            fnv1a64(f32_slice_bytes(v))
        ));
    }
    std::fs::write(dir.join("MANIFEST"), manifest).expect("write manifest");
}

// ── Eager runs ──────────────────────────────────────────────────────────

fn run_single_eager(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    inp: &Inputs,
) -> HipResult<Planes> {
    let row = cfg.num_extract() * cfg.hidden;
    let mut s = build_scratch(gpu, weights, cfg)?;
    poison_scratch(gpu, &s)?;
    upload_inputs(gpu, &s, inp, row)?;
    draft_forward_opts(
        gpu, weights, cfg, None, None, &inp.pos_q, &inp.pos_k, inp.b, inp.l, &mut s, false,
    )?;
    gpu.hip.device_synchronize()?;
    let planes = collect_single(gpu, &s)?;
    s.free_gpu(gpu);
    Ok(planes)
}

/// Fresh lane scratches + shared scratch holding the lanes' entry state:
/// per-lane target hidden uploaded, shared `x` rows `[off_i, off_i + b)` =
/// lane `i`'s noise. Positions are left to `draft_forward_lanes`.
fn build_lane_set(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    ins: &[Inputs],
) -> HipResult<(DflashScratch, Vec<DflashScratch>)> {
    let row = cfg.num_extract() * cfg.hidden;
    let shared = build_shared(gpu, cfg)?;
    poison_scratch(gpu, &shared)?;
    let mut lanes = Vec::with_capacity(ins.len());
    for inp in ins {
        let s = build_scratch(gpu, weights, cfg)?;
        poison_scratch(gpu, &s)?;
        upload_th(gpu, &s, &inp.th, inp.l, row)?;
        lanes.push(s);
    }
    restore_shared_x(gpu, &shared, ins)?;
    Ok((shared, lanes))
}

fn restore_shared_x(gpu: &Gpu, shared: &DflashScratch, ins: &[Inputs]) -> HipResult<()> {
    let noise: Vec<f32> = ins.iter().flat_map(|i| i.noise.iter().copied()).collect();
    gpu.hip.memcpy_htod(&shared.x.buf, f32_slice_bytes(&noise))
}

fn call_lanes(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    shared: &mut DflashScratch,
    scratches: &mut [DflashScratch],
    ins: &[Inputs],
) -> HipResult<()> {
    let mut lanes: Vec<DraftLane<'_>> = scratches
        .iter_mut()
        .zip(ins)
        .map(|(s, i)| DraftLane {
            scratch: s,
            b: i.b,
            ctx_len: i.l,
            positions_q: &i.pos_q,
            positions_k: &i.pos_k,
        })
        .collect();
    draft_forward_lanes(gpu, weights, cfg, shared, &mut lanes)
}

fn run_lanes_eager(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    ins: &[Inputs],
) -> HipResult<Planes> {
    let (mut shared, mut lanes) = build_lane_set(gpu, weights, cfg, ins)?;
    call_lanes(gpu, weights, cfg, &mut shared, &mut lanes, ins)?;
    gpu.hip.device_synchronize()?;
    let planes = collect_lanes(gpu, &shared, &lanes)?;
    shared.free_gpu(gpu);
    for s in lanes {
        s.free_gpu(gpu);
    }
    Ok(planes)
}

// ── Capture / replay ────────────────────────────────────────────────────

/// A captured + instantiated graph. `_blobs` owns the kernarg blobs the
/// graph's kernel nodes point at: it must outlive every replay.
struct Captured {
    graph: Graph,
    exec: GraphExec,
    _blobs: Vec<Vec<u8>>,
}

/// Mirrors `abort_draft_ffn_graph_capture` (crates/hipfire-runtime/src/dflash.rs).
fn abort_capture(gpu: &mut Gpu) {
    if gpu.graphs.capture_mode {
        if let Some(stream) = gpu.active_stream.as_ref() {
            if let Ok(g) = gpu.hip.stream_end_capture(stream) {
                let _ = gpu.hip.graph_destroy(g);
            }
        }
        gpu.graphs.capture_mode = false;
    }
    gpu.graphs.capture_blobs.clear();
}

/// Mirrors `begin/end_draft_ffn_graph_capture`: `call` runs between
/// begin/end capture on `gpu.active_stream`.
fn capture_graph(
    gpu: &mut Gpu,
    call: impl FnOnce(&mut Gpu) -> HipResult<()>,
) -> Result<Captured, String> {
    gpu.graphs.capture_blobs.clear();
    gpu.graphs.capture_mode = true;
    let begin = {
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("graph capture requires an explicit stream");
        gpu.hip.stream_begin_capture(stream, 0)
    };
    if let Err(e) = begin {
        gpu.graphs.capture_mode = false;
        gpu.graphs.capture_blobs.clear();
        return Err(format!("stream_begin_capture: {e}"));
    }
    if let Err(e) = call(gpu) {
        abort_capture(gpu);
        return Err(format!("draft call during capture: {e}"));
    }
    gpu.graphs.capture_mode = false;
    let graph = {
        let stream = gpu.active_stream.as_ref().unwrap();
        match gpu.hip.stream_end_capture(stream) {
            Ok(g) => g,
            Err(e) => {
                gpu.graphs.capture_blobs.clear();
                return Err(format!("stream_end_capture: {e}"));
            }
        }
    };
    let exec = match gpu.hip.graph_instantiate(&graph) {
        Ok(x) => x,
        Err(e) => {
            let _ = gpu.hip.graph_destroy(graph);
            gpu.graphs.capture_blobs.clear();
            return Err(format!("graph_instantiate: {e}"));
        }
    };
    let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
    Ok(Captured {
        graph,
        exec,
        _blobs: blobs,
    })
}

/// Replay `REPLAYS` times. Before each replay `restore` re-establishes the
/// entry planes (synchronous H2D, outside capture); after it `collect`
/// downloads the planes and they are compared byte-for-byte with `baseline`.
fn replay_all(
    gpu: &Gpu,
    cap: &Captured,
    mut restore: impl FnMut(&Gpu) -> HipResult<()>,
    mut collect: impl FnMut(&Gpu) -> HipResult<Planes>,
    baseline: &Planes,
) -> Vec<Value> {
    let mut out = Vec::new();
    for i in 0..REPLAYS {
        let r: HipResult<Vec<String>> = (|| {
            restore(gpu)?;
            let stream = gpu.active_stream.as_ref().unwrap();
            gpu.hip.graph_launch(&cap.exec, stream)?;
            gpu.hip.stream_synchronize(stream)?;
            let planes = collect(gpu)?;
            Ok(diff_planes(baseline, &planes))
        })();
        match r {
            Ok(bad) => out.push(json!({"replay": i, "error": null, "mismatched_planes": bad})),
            Err(e) => {
                out.push(json!({
                    "replay": i,
                    "error": e.to_string(),
                    "mismatched_planes": [],
                }));
                break;
            }
        }
    }
    out
}

fn destroy_captured(gpu: &Gpu, cap: Captured) {
    let _ = gpu.hip.device_synchronize();
    let _ = gpu.hip.graph_exec_destroy(cap.exec);
    let _ = gpu.hip.graph_destroy(cap.graph);
}

fn capture_failed(err: String) -> Value {
    eprintln!("capture refused/failed: {err}");
    json!({"ok": false, "error": err, "replays": []})
}

fn capture_single(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    inp: &Inputs,
    baseline: &Planes,
) -> HipResult<Value> {
    let row = cfg.num_extract() * cfg.hidden;
    let mut s = build_scratch(gpu, weights, cfg)?;
    poison_scratch(gpu, &s)?;
    upload_inputs(gpu, &s, inp, row)?;
    let res = capture_graph(gpu, |gpu| {
        draft_forward_opts(
            gpu, weights, cfg, None, None, &inp.pos_q, &inp.pos_k, inp.b, inp.l, &mut s, false,
        )
    });
    let value = match res {
        Err(e) => {
            let _ = gpu.hip.device_synchronize();
            capture_failed(e)
        }
        Ok(cap) => {
            let replays = replay_all(
                gpu,
                &cap,
                // Poison every plane (positions and caches included) and
                // re-upload only the entry inputs: each replay must redo the
                // whole forward, position uploads included.
                |gpu| {
                    poison_scratch(gpu, &s)?;
                    upload_inputs(gpu, &s, inp, row)
                },
                |gpu| collect_single(gpu, &s),
                baseline,
            );
            destroy_captured(gpu, cap);
            json!({"ok": true, "error": null, "replays": replays})
        }
    };
    s.free_gpu(gpu);
    Ok(value)
}

fn capture_lanes(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    ins: &[Inputs],
    baseline: &Planes,
) -> HipResult<Value> {
    let row = cfg.num_extract() * cfg.hidden;
    let (mut shared, mut lanes) = build_lane_set(gpu, weights, cfg, ins)?;
    let res = capture_graph(gpu, |gpu| {
        call_lanes(gpu, weights, cfg, &mut shared, &mut lanes, ins)
    });
    let value = match res {
        Err(e) => {
            let _ = gpu.hip.device_synchronize();
            capture_failed(e)
        }
        Ok(cap) => {
            let replays = replay_all(
                gpu,
                &cap,
                |gpu| {
                    poison_scratch(gpu, &shared)?;
                    for (s, inp) in lanes.iter().zip(ins) {
                        poison_scratch(gpu, s)?;
                        upload_th(gpu, s, &inp.th, inp.l, row)?;
                    }
                    restore_shared_x(gpu, &shared, ins)
                },
                |gpu| collect_lanes(gpu, &shared, &lanes),
                baseline,
            );
            destroy_captured(gpu, cap);
            json!({"ok": true, "error": null, "replays": replays})
        }
    };
    shared.free_gpu(gpu);
    for s in lanes {
        s.free_gpu(gpu);
    }
    Ok(value)
}

// ── Child (one arm) ─────────────────────────────────────────────────────

fn child(draft_path: &str, outdir: &Path) {
    let flag = std::env::var(ENV_FLAG).ok();
    let ordered = flag.as_deref().map_or(true, |v| v.trim() != "0");
    let arm = if ordered { "on" } else { "off" };
    let dir = outdir.join(arm);
    std::fs::create_dir_all(&dir).expect("mkdir arm dir");

    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream_create"));
    }
    let fused = gpu.draft_collapse_fused_enabled();
    eprintln!(
        "[{arm}] gpu={} {ENV_FLAG}={:?} ordered={ordered} fused_collapse_route={fused}",
        gpu.arch, flag
    );

    let hfq = HfqFile::open(Path::new(draft_path)).expect("open draft artifact");
    let cfg = DflashConfig::from_hfq(&hfq).expect("parse DflashConfig");
    let weights = DflashWeights::load(&mut gpu, &hfq, &cfg).expect("load draft");
    let h = cfg.hidden;
    let ne = cfg.num_extract();

    // ── 1. Singleton eager ──
    let single_in = gen_inputs(0x2b7e_1516_28ae_d2a6, B, L_SINGLE, 1000, h, ne);
    let single = run_single_eager(&mut gpu, &weights, &cfg, &single_in).expect("single eager");
    write_planes(&dir.join("single_eager"), &single);
    eprintln!("[{arm}] single_eager: {} planes", single.len());

    // ── 2. Lanes eager (+ lanes vs singleton) ──
    let lane_ins: Vec<Inputs> = LANE_CTX
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let seed =
                0x2b7e_1516_28ae_d2a6u64 ^ 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1);
            gen_inputs(seed, B, l, 2000 + 500 * i as i32, h, ne)
        })
        .collect();
    let lanes_eager = run_lanes_eager(&mut gpu, &weights, &cfg, &lane_ins);
    let (lanes_eager_json, lanes_vs_singleton, lanes_planes) = match lanes_eager {
        Err(e) => {
            let err = format!("draft_forward_lanes: {e}");
            eprintln!("[{arm}] lanes_eager FAILED: {err}");
            (
                json!({"ok": false, "error": err, "planes": []}),
                json!({"ok": false, "error": "lanes eager failed", "lanes": []}),
                None,
            )
        }
        Ok(planes) => {
            write_planes(&dir.join("lanes_eager"), &planes);
            let shared_x = find_plane(&planes, "shared_x");
            let mut per_lane = Vec::new();
            let mut all_equal = true;
            let mut off = 0usize;
            for (i, inp) in lane_ins.iter().enumerate() {
                let single_i = run_single_eager(&mut gpu, &weights, &cfg, inp)
                    .unwrap_or_else(|e| panic!("lane {i} singleton eager: {e}"));
                let fx = find_plane(&single_i, "final_x");
                let rows = &shared_x[off * h..(off + inp.b) * h];
                let equal = bits_equal(fx, rows);
                all_equal &= equal;
                per_lane.push(json!({"lane": i, "rows": [off, off + inp.b], "equal": equal}));
                off += inp.b;
            }
            eprintln!("[{arm}] lanes_vs_singleton: {all_equal}");
            (
                json!({"ok": true, "error": null, "planes": planes_json(&planes)}),
                json!({"ok": all_equal, "error": null, "lanes": per_lane}),
                Some(planes),
            )
        }
    };

    // ── 3. Capture / replay (never fatal: the parent judges) ──
    let cap_single = capture_single(&mut gpu, &weights, &cfg, &single_in, &single)
        .unwrap_or_else(|e| capture_failed(format!("setup: {e}")));
    let cap_lanes = match &lanes_planes {
        Some(planes) => capture_lanes(&mut gpu, &weights, &cfg, &lane_ins, planes)
            .unwrap_or_else(|e| capture_failed(format!("setup: {e}"))),
        None => capture_failed("skipped: lanes eager failed".to_string()),
    };
    eprintln!(
        "[{arm}] capture single ok={} lanes ok={}",
        cap_single["ok"], cap_lanes["ok"]
    );

    // ── 4. result.json ──
    let result = json!({
        "arm": arm,
        "ordered": ordered,
        "env": flag,
        "gpu_arch": gpu.arch,
        "fused_collapse_route": fused,
        "single_eager": {"ok": true, "planes": planes_json(&single)},
        "lanes_eager": lanes_eager_json,
        "lanes_vs_singleton": lanes_vs_singleton,
        "capture_single": cap_single,
        "capture_lanes": cap_lanes,
    });
    std::fs::write(
        dir.join("result.json"),
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .expect("write result.json");
}

// ── Parent ──────────────────────────────────────────────────────────────

fn read_result(outdir: &Path, arm: &str) -> Result<Value, String> {
    let p = outdir.join(arm).join("result.json");
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

fn replays_clean(v: &Value) -> bool {
    let Some(r) = v["replays"].as_array() else {
        return false;
    };
    r.len() == REPLAYS
        && r.iter().all(|x| {
            x["error"].is_null()
                && x["mismatched_planes"]
                    .as_array()
                    .is_some_and(|m| m.is_empty())
        })
}

fn parent(draft_path: &str, outdir: &Path) -> bool {
    std::fs::create_dir_all(outdir).expect("mkdir outdir");
    let exe = std::env::current_exe().expect("current exe");
    let mut fails: Vec<String> = Vec::new();

    for (arm, val) in [("on", "1"), ("off", "0")] {
        eprintln!("== arm {arm} ({ENV_FLAG}={val}) ==");
        let st = std::process::Command::new(&exe)
            .arg(draft_path)
            .arg(outdir)
            .arg("child")
            .env(ENV_FLAG, val)
            .status()
            .expect("re-exec child");
        if !st.success() {
            fails.push(format!("arm {arm}: child exited with {st}"));
        }
    }

    let on = read_result(outdir, "on");
    let off = read_result(outdir, "off");
    let (on, off) = match (on, off) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            for e in [a.err(), b.err()].into_iter().flatten() {
                fails.push(format!("missing result: {e}"));
            }
            return finish(outdir, fails, Value::Null, Value::Null);
        }
    };

    if on["ordered"] != json!(true) {
        fails.push("arm on: effective ordered flag is not true".into());
    }
    if off["ordered"] != json!(false) {
        fails.push("arm off: effective ordered flag is not false".into());
    }

    // (a) eager identity across arms (manifests + plane bytes), (b) lanes==singleton.
    for step in ["single_eager", "lanes_eager"] {
        for (arm, r) in [("on", &on), ("off", &off)] {
            if r[step]["ok"] != json!(true) {
                fails.push(format!("arm {arm}: {step} failed: {}", r[step]["error"]));
            }
        }
        if on[step]["planes"] != off[step]["planes"] {
            fails.push(format!("{step}: plane manifests differ between arms"));
        }
        if let Some(planes) = on[step]["planes"].as_array() {
            for p in planes {
                let name = p["name"].as_str().unwrap_or("");
                let f = |arm: &str| {
                    std::fs::read(outdir.join(arm).join(step).join(format!("{name}.f32")))
                };
                match (f("on"), f("off")) {
                    (Ok(a), Ok(b)) if a == b => {}
                    _ => fails.push(format!("{step}/{name}: bytes differ between arms")),
                }
            }
        }
    }
    for (arm, r) in [("on", &on), ("off", &off)] {
        if r["lanes_vs_singleton"]["ok"] != json!(true) {
            fails.push(format!(
                "arm {arm}: lanes_vs_singleton false: {}",
                r["lanes_vs_singleton"]
            ));
        }
    }

    // (c) arm on: both captures succeed and replay clean.
    for key in ["capture_single", "capture_lanes"] {
        let c = &on[key];
        if c["ok"] != json!(true) {
            fails.push(format!("arm on: {key} failed: {}", c["error"]));
        } else if !replays_clean(c) {
            fails.push(format!("arm on: {key} replays not clean: {}", c["replays"]));
        }
    }

    // (d) arm off: report only.
    let off_cap = json!({
        "single": {"ok": off["capture_single"]["ok"], "error": off["capture_single"]["error"]},
        "lanes": {"ok": off["capture_lanes"]["ok"], "error": off["capture_lanes"]["error"]},
    });
    let on_cap = json!({
        "single": on["capture_single"],
        "lanes": on["capture_lanes"],
    });
    finish(outdir, fails, on_cap, off_cap)
}

fn finish(outdir: &Path, fails: Vec<String>, on_cap: Value, off_cap: Value) -> bool {
    let pass = fails.is_empty();
    let summary = json!({
        "pass": pass,
        "failures": fails,
        "on_capture": on_cap,
        "off_capture_report_only": off_cap,
    });
    std::fs::write(
        outdir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary.json");
    if pass {
        println!(
            "PASS dflash_draft_graph_replay: ON/OFF eager identical, lanes==singleton, \
             ON capture+{REPLAYS} replays byte-identical (singleton+lanes); OFF capture \
             single ok={} lanes ok={} [{}]",
            off_cap["single"]["ok"],
            off_cap["lanes"]["ok"],
            outdir.join("summary.json").display()
        );
    } else {
        println!(
            "FAIL dflash_draft_graph_replay: {} failure(s): {} [{}]",
            summary["failures"].as_array().map_or(0, |f| f.len()),
            summary["failures"]
                .as_array()
                .map(|f| f
                    .iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join("; "))
                .unwrap_or_default(),
            outdir.join("summary.json").display()
        );
    }
    pass
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dflash_draft_graph_replay <draft.hfq> <outdir> [child]");
        std::process::exit(2);
    }
    let (draft, outdir) = (&args[1], Path::new(&args[2]));
    if args.get(3).map(|s| s.as_str()) == Some("child") {
        child(draft, outdir);
        return;
    }
    if !parent(draft, outdir) {
        std::process::exit(1);
    }
}
