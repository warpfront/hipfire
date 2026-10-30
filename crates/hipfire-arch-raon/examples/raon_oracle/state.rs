// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `state` subcommand (PLAN §6.2, slice S gate).
//!
//! State-only checks that exercise the production owners in
//! `hipfire_arch_raon::state` on the real device:
//!
//! * `slot_isolation`      two slots, unequal lengths, interleaved decode-style
//!                         writes, poisoned rows beyond each length, per-slot
//!                         reset/abort, snapshot/restore, failed admission;
//! * `ctx_reservation_*`   virtual reservations at ctx 512/8192/32768/65536/98304,
//!                         boundary growth with data integrity across the chunk
//!                         edge, physical bytes versus the ledger, recurrent bytes
//!                         constant in max_seq;
//! * `admission_*`         clean refusals (oversized reservation, 262144 admitted
//!                         only if the device backs it, growth refused under
//!                         memory pressure without mutating the slot);
//! * `scratch_*`           bounded scratch at 1/512/1024 rows vs plan bytes;
//! * `expanded_mirror`     admission limits and ledger;
//! * `vmm_owner_table`     every VMM owner released.
//!
//! `--mode serial|batched` additionally runs the integrated forward through
//! [`run_forward_mode`] (route parity + same-route determinism on the real
//! model); `--mode graph` runs eager-vs-graph greedy decode and gates every
//! logit byte and the final state bytes bitwise (see [`fwd_graph`]).

use std::collections::BTreeMap;
use std::path::Path;

use hip_bridge::DeviceBuffer;
use hipfire_arch_raon::bundle::{DecodeMode, PrefillMode, RaonBundle};
use hipfire_arch_raon::forward::forward_token;
use hipfire_runtime::loader_api::ModelSource;
use std::time::Instant;

use super::logits::{check_tokens, load_bundle, read_tokens};
use hipfire_arch_raon::state::{
    auto_backend, expanded_mirror_bytes, scratch_plan, state_budget, LatentBackend,
    RaonPrefillScratch, RaonState, RaonStateDims, ADMISSION_HEADROOM_BYTES,
};
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};

const DEFAULT_CTXS: [usize; 5] = [512, 8192, 32768, 65536, 98304];
const MIB: usize = 1 << 20;

type Check = Result<Value, String>;

/// Entry point shared with `raon_oracle state`.
pub fn run(_fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let mode = opts.get("mode").map(String::as_str).unwrap_or("state-only");
    let forward_mode = match mode {
        "state-only" | "state" => None,
        "serial" | "batched" | "graph" => Some(mode),
        other => {
            return Err(format!(
                "state: unknown --mode {other} (serial|batched|graph|state-only)"
            ))
        }
    };
    let ctxs = match opts.get("ctxs") {
        Some(s) => s
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<usize>()
                    .map_err(|e| format!("state: bad ctx {x:?}: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => DEFAULT_CTXS.to_vec(),
    };
    let dims = RaonStateDims::raon();
    dims.validate()?;
    let mut gpu = Gpu::init().map_err(|e| format!("state: gpu init: {e:?}"))?;

    let mut checks: Vec<Value> = Vec::new();
    let mut record = |name: String, r: Check| {
        checks.push(match r {
            Ok(detail) => json!({"name": name, "pass": true, "detail": detail}),
            Err(e) => json!({"name": name, "pass": false, "error": e}),
        });
    };

    record(
        "slot_isolation".into(),
        check_slot_isolation(&mut gpu, &dims),
    );
    let mut recurrent_bytes: Vec<(usize, usize)> = Vec::new();
    for &ctx in &ctxs {
        let r = check_ctx_reservation(&mut gpu, &dims, ctx, &mut recurrent_bytes);
        record(format!("ctx_reservation_{ctx}"), r);
    }
    record(
        "rda_bytes_constant_in_max_seq".into(),
        check_recurrent_constant(&recurrent_bytes),
    );
    record(
        "admission_refusal".into(),
        check_admission_refusal(&mut gpu, &dims),
    );
    record(
        "admission_262144".into(),
        check_admission_262144(&mut gpu, &dims),
    );
    record(
        "admission_growth_under_pressure".into(),
        check_growth_pressure(&mut gpu, &dims),
    );
    for rows in [1usize, 512, 1024] {
        record(
            format!("scratch_{rows}"),
            check_scratch(&mut gpu, &dims, rows),
        );
    }
    record(
        "scratch_rows_limit".into(),
        check_scratch_limit(&mut gpu, &dims),
    );
    record("expanded_mirror".into(), check_mirror(&mut gpu, &dims));
    gpu.drain_pool();
    record("vmm_owner_table".into(), check_vmm_clean(&mut gpu));

    let state_pass = checks.iter().all(|c| c["pass"].as_bool() == Some(true));
    let mut out = json!({
        "module": "state",
        "arch": gpu.arch,
        "mode": mode,
        "ctxs": ctxs,
        "state_checks_pass": state_pass,
        "checks": checks,
    });
    let mut pass = state_pass;
    if let Some(m) = forward_mode {
        match run_forward_mode(&mut gpu, m, opts) {
            Ok(v) => {
                pass &= v["pass"].as_bool() == Some(true);
                out["forward"] = json!({"status": "ok", "result": v});
            }
            Err(e) => {
                pass = false;
                out["forward"] = json!({"status": "error", "error": e});
            }
        }
    }
    out["pass"] = json!(pass);
    Ok(out)
}

/// Full-forward route parity (`serial|batched|graph`).
///
/// Loads the production bundle and drives the integrated forward over the
/// `--tokens` list through the SERIAL route (`forward_token` per token, state
/// advanced per token) and/or the BATCHED route (`bundle.prefill`, chunked),
/// fingerprinting after each run every RDA state, every conv history plane
/// (q/k/v), the live latent rows `[0, n_tokens)` of every MLA cache and the
/// final logits.
///
/// * `serial`:  serial twice; same-route determinism is bitwise.
/// * `batched`: serial once, batched twice; batched determinism is bitwise,
///   cross-route logits must have cosine >= 0.999 and equal top-1. Cross-route
///   state differences are reported, never gated bitwise.
/// * `graph`:   [`fwd_graph`]: `--steps` (default 256) greedy decode steps after
///   the same batched prefill, eager (`HIPFIRE_GRAPH=0` route) vs graph replay;
///   every step's logit bytes, the device `seq_len`, every RDA state / conv
///   plane / live latent row at the end must be bitwise equal, at two request
///   lengths through one recorded graph (reset in between).
///
/// Opts: `model` (HFQ), `tokens` (u32le binary or text ids, see `logits.rs`),
/// `steps` (graph mode only).
pub fn run_forward_mode(
    gpu: &mut Gpu,
    mode: &str,
    opts: &BTreeMap<String, String>,
) -> Result<Value, String> {
    if mode != "serial" && mode != "batched" && mode != "graph" {
        return Err(format!("forward mode `{mode}`: serial|batched|graph"));
    }
    let model = opts
        .get("model")
        .ok_or("state --mode serial|batched|graph: missing --model")?;
    let tok_path = opts
        .get("tokens")
        .ok_or("state --mode serial|batched|graph: missing --tokens")?;
    let tokens = read_tokens(tok_path)?;
    let steps = match opts.get("steps") {
        Some(s) => s
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("state: bad --steps {s:?}: {e}"))?,
        None => 256,
    };
    if mode == "graph" && steps == 0 {
        return Err("state --mode graph: --steps must be >= 1".into());
    }
    let extra = if mode == "graph" { steps } else { 0 };
    let max_seq = (tokens.len() + extra + 64).max(512);
    let src = ModelSource::from_path(model)?;
    let mut bundle = load_bundle(gpu, src, model, max_seq)?;
    let result = if mode == "graph" {
        fwd_graph(gpu, &mut bundle, &tokens, steps)
    } else {
        fwd_mode_inner(gpu, &mut bundle, mode, &tokens)
    };
    bundle.free_all(gpu);
    result.map(|mut v| {
        v["model"] = json!(model);
        v
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FwdElem {
    F32,
    Bf16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FwdRoute {
    Serial,
    Batched,
}

struct FwdItem {
    name: String,
    elem: FwdElem,
    bytes: Vec<u8>,
}

struct FwdSnap {
    items: Vec<FwdItem>,
    n_tokens: usize,
    secs: f64,
}

fn fwd_mode_inner(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    mode: &str,
    tokens: &[u32],
) -> Result<Value, String> {
    check_tokens(tokens, bundle.config.vocab, "tokens")?;
    let n = tokens.len();
    let mut out = json!({
        "n_tokens": n,
        "mode": mode,
        "definitions": {
            "bitwise_equal": "every captured byte identical",
            "max_abs": "max |a-b| over the tensor",
            "max_rel": "max_abs / max |b| (peak-relative)",
            "rel_l2": "||a-b||2 / ||b||2",
        },
    });
    let mut pass = true;
    match mode {
        "serial" => {
            let s1 = fwd_run_route(gpu, bundle, tokens, FwdRoute::Serial)?;
            let s2 = fwd_run_route(gpu, bundle, tokens, FwdRoute::Serial)?;
            let det = fwd_compare(&s1, &s2)?;
            let finite = fwd_logits_finite(&s1)?;
            pass &= det["bitwise_equal"].as_bool() == Some(true) && finite;
            out["serial"] = fwd_route_json(&s1);
            out["serial_repeat_secs"] = json!(s2.secs);
            out["serial_vs_serial"] = det;
            out["logits_finite"] = json!(finite);
        }
        _ => {
            let s = fwd_run_route(gpu, bundle, tokens, FwdRoute::Serial)?;
            let b1 = fwd_run_route(gpu, bundle, tokens, FwdRoute::Batched)?;
            let b2 = fwd_run_route(gpu, bundle, tokens, FwdRoute::Batched)?;
            let det = fwd_compare(&b1, &b2)?;
            let cross = fwd_compare(&s, &b1)?;
            let (cos, top1_equal, finite) = fwd_logits_cross(&s, &b1)?;
            let cross_ok = cos >= 0.999 && top1_equal;
            pass &= det["bitwise_equal"].as_bool() == Some(true) && cross_ok && finite;
            out["serial"] = fwd_route_json(&s);
            out["batched"] = fwd_route_json(&b1);
            out["batched_repeat_secs"] = json!(b2.secs);
            out["batched_vs_batched"] = det;
            out["serial_vs_batched"] = cross;
            out["cross_route_logits"] = json!({
                "cosine": cos,
                "top1_equal": top1_equal,
                "band": {"min_cosine": 0.999, "top1_equal": true},
                "pass": cross_ok,
            });
            out["logits_finite"] = json!(finite);
        }
    }
    out["pass"] = json!(pass);
    Ok(out)
}

/// Reset, run `tokens` through `route`, synchronize and capture the state.
fn fwd_run_route(
    gpu: &mut Gpu,
    b: &mut RaonBundle,
    tokens: &[u32],
    route: FwdRoute,
) -> Result<FwdSnap, String> {
    b.reset(gpu)?;
    b.state.ensure_capacity(gpu, tokens.len())?;
    let t0 = Instant::now();
    match route {
        FwdRoute::Serial => {
            let last = tokens.len() - 1;
            for (i, &t) in tokens.iter().enumerate() {
                forward_token(
                    &b.config,
                    &b.weights,
                    &mut b.state,
                    &mut b.scratch,
                    &mut b.work,
                    gpu,
                    t,
                    i as u32,
                    i == last,
                )?;
            }
        }
        FwdRoute::Batched => {
            b.set_prefill_mode(PrefillMode::Batched);
            b.prefill(gpu, tokens)?;
        }
    }
    hip_err(gpu.hip.device_synchronize(), "synchronize after route")?;
    let secs = t0.elapsed().as_secs_f64();
    if b.state.n_tokens() != tokens.len() {
        return Err(format!(
            "route advanced state to {} tokens, expected {}",
            b.state.n_tokens(),
            tokens.len()
        ));
    }
    fwd_capture(gpu, b, secs)
}

/// One greedy decode run: reset, batched prefill of `prompt`, then `steps`
/// decode steps each fed the previous step's argmax. Logits (index 0 = after
/// the prefill, then one per step) are either kept (`reference == None`) or
/// compared byte-for-byte against `reference` as they are produced.
struct DecodeArm {
    mode: &'static str,
    kept_logits: Vec<Vec<u8>>,
    tokens: Vec<u32>,
    logit_sets: usize,
    logit_mismatch_sets: usize,
    first_logit_mismatch: Option<usize>,
    seq_len_mismatches: usize,
    captures: u64,
    replays: u64,
    decode_secs: f64,
    snap: FwdSnap,
}

fn first_argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn decode_arm(
    gpu: &mut Gpu,
    b: &mut RaonBundle,
    prompt: &[u32],
    steps: usize,
    mode: DecodeMode,
    fresh_mirror: bool,
    reference: Option<&[Vec<u8>]>,
) -> Result<DecodeArm, String> {
    b.reset(gpu)?;
    if fresh_mirror {
        // Same mirror history for every arm of a pair: growth re-expands the
        // live prefix through the batched kv_b route, so an arm that starts with
        // an already-grown mirror is not the same route as one that grows.
        b.state.release_expanded_mirror(gpu)?;
    }
    b.set_prefill_mode(PrefillMode::Batched);
    b.set_decode_mode(mode);
    let (c0, r0) = b.decode_graph_stats();
    b.prefill(gpu, prompt)?;
    let mut arm = DecodeArm {
        mode: mode.as_str(),
        kept_logits: Vec::new(),
        tokens: Vec::with_capacity(steps),
        logit_sets: 0,
        logit_mismatch_sets: 0,
        first_logit_mismatch: None,
        seq_len_mismatches: 0,
        captures: 0,
        replays: 0,
        decode_secs: 0.0,
        snap: FwdSnap {
            items: Vec::new(),
            n_tokens: 0,
            secs: 0.0,
        },
    };
    let record = |arm: &mut DecodeArm, logits: &[f32]| -> Vec<u8> {
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let idx = arm.logit_sets;
        arm.logit_sets += 1;
        if let Some(r) = reference {
            if r.get(idx).is_none_or(|e| *e != bytes) {
                arm.logit_mismatch_sets += 1;
                arm.first_logit_mismatch.get_or_insert(idx);
            }
        }
        bytes
    };
    let mut logits = hip_err(
        gpu.download_f32(&b.scratch.logits),
        "download prefill logits",
    )?;
    let bytes = record(&mut arm, &logits);
    if reference.is_none() {
        arm.kept_logits.push(bytes);
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        let tok = first_argmax(&logits);
        arm.tokens.push(tok);
        b.decode_step(gpu, tok)?;
        logits = hip_err(
            gpu.download_f32(&b.scratch.logits),
            "download decode logits",
        )?;
        let bytes = record(&mut arm, &logits);
        if reference.is_none() {
            arm.kept_logits.push(bytes);
        }
        if device_seq_len(gpu, &b.state)? as usize != b.state.n_tokens() {
            arm.seq_len_mismatches += 1;
        }
    }
    hip_err(gpu.hip.device_synchronize(), "synchronize after decode arm")?;
    arm.decode_secs = t0.elapsed().as_secs_f64();
    let (c1, r1) = b.decode_graph_stats();
    arm.captures = c1 - c0;
    arm.replays = r1 - r0;
    if b.state.n_tokens() != prompt.len() + steps {
        return Err(format!(
            "decode arm advanced state to {} tokens, expected {}",
            b.state.n_tokens(),
            prompt.len() + steps
        ));
    }
    arm.snap = fwd_capture(gpu, b, arm.decode_secs)?;
    Ok(arm)
}

/// Informational decode-loop wall time: `n` steps of the same token with no
/// per-step readback, one synchronize at the end (2 untimed warm steps first, so
/// the graph arm is already recorded). Not a performance claim.
fn decode_loop_secs(
    gpu: &mut Gpu,
    b: &mut RaonBundle,
    prompt: &[u32],
    n: usize,
    mode: DecodeMode,
) -> Result<f64, String> {
    b.reset(gpu)?;
    b.set_prefill_mode(PrefillMode::Batched);
    b.set_decode_mode(mode);
    b.prefill(gpu, prompt)?;
    let logits = hip_err(gpu.download_f32(&b.scratch.logits), "download loop logits")?;
    let tok = first_argmax(&logits);
    for _ in 0..2 {
        b.decode_step(gpu, tok)?;
    }
    hip_err(
        gpu.hip.device_synchronize(),
        "synchronize before timed loop",
    )?;
    let t0 = Instant::now();
    for _ in 0..n {
        b.decode_step(gpu, tok)?;
    }
    hip_err(gpu.hip.device_synchronize(), "synchronize after timed loop")?;
    Ok(t0.elapsed().as_secs_f64())
}

fn arm_json(a: &DecodeArm) -> Value {
    json!({
        "decode_mode": a.mode,
        "logit_sets": a.logit_sets,
        "logit_sets_mismatching": a.logit_mismatch_sets,
        "first_logit_mismatch": a.first_logit_mismatch,
        "seq_len_mismatches": a.seq_len_mismatches,
        "graph_captures": a.captures,
        "graph_replays": a.replays,
        "decode_secs_informational": a.decode_secs,
    })
}

/// `--mode graph`: eager vs graph decode, same route, bitwise (PLAN §7 G gate).
///
/// Each case runs an eager arm and a graph arm from the same prompt with the
/// same mirror history; every logit byte of every step, the device `seq_len`,
/// every RDA state, conv plane and live latent row must be bitwise equal.
///
/// * `full_fresh_mirror`: the expanded mirror is released before each arm, so
///   both arms admit it at prefill and grow it (and the graph arm re-records)
///   at the same context length when `steps` crosses a growth boundary.
/// * `full_reuse_recording`: after a reset, the recording from the previous case
///   must replay (zero new captures when the bucket did not change).
/// * `short_after_reset`: half the prompt, `min(steps, 64)` steps, same
///   recording; a stale position or length would diverge the logits.
fn fwd_graph(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    tokens: &[u32],
    steps: usize,
) -> Result<Value, String> {
    check_tokens(tokens, bundle.config.vocab, "tokens")?;
    let mut cases = Vec::new();
    let mut pass = true;
    let half = (tokens.len() / 2).max(1);
    let plan: [(&str, &[u32], usize, bool); 3] = [
        ("full_fresh_mirror", tokens, steps, true),
        ("full_reuse_recording", tokens, steps, false),
        ("short_after_reset", &tokens[..half], steps.min(64), false),
    ];
    for (name, prompt, n_steps, fresh) in plan {
        let eager = decode_arm(gpu, bundle, prompt, n_steps, DecodeMode::Eager, fresh, None)?;
        let arm = decode_arm(
            gpu,
            bundle,
            prompt,
            n_steps,
            DecodeMode::Graph,
            fresh,
            Some(&eager.kept_logits),
        )?;
        let state = fwd_compare(&eager.snap, &arm.snap)?;
        let state_equal = state["bitwise_equal"].as_bool() == Some(true);
        let tokens_equal = arm.tokens == eager.tokens;
        let logits_equal = arm.logit_mismatch_sets == 0 && arm.logit_sets == eager.logit_sets;
        // Not a placeholder pass: the arm must really have been served by a
        // recording, and a reused recording must not have been re-captured.
        let served_by_graph = arm.replays > 0;
        let reuse_ok = fresh || arm.captures == 0;
        let case_pass = state_equal
            && tokens_equal
            && logits_equal
            && arm.seq_len_mismatches == 0
            && eager.seq_len_mismatches == 0
            && served_by_graph
            && reuse_ok;
        pass &= case_pass;
        let mut graph_json = arm_json(&arm);
        graph_json["state_bitwise_equal"] = json!(state_equal);
        graph_json["tokens_equal"] = json!(tokens_equal);
        graph_json["logits_bitwise_equal_every_step"] = json!(logits_equal);
        graph_json["served_by_graph"] = json!(served_by_graph);
        graph_json["recording_reused_without_recapture"] = json!(reuse_ok);
        graph_json["state_vs_eager"] = state;
        let mut case_json = json!({
            "case": name,
            "prompt_tokens": prompt.len(),
            "steps": n_steps,
            "fresh_mirror": fresh,
            "eager": arm_json(&eager),
            "graph": graph_json,
            "pass": case_pass,
        });
        if name == "full_reuse_recording" {
            let loop_n = n_steps.min(128);
            let eager_secs = decode_loop_secs(gpu, bundle, prompt, loop_n, DecodeMode::Eager)?;
            let graph_secs = decode_loop_secs(gpu, bundle, prompt, loop_n, DecodeMode::Graph)?;
            case_json["decode_loop_informational"] = json!({
                "steps": loop_n,
                "eager_secs": eager_secs,
                "graph_secs": graph_secs,
                "graph_kernel_nodes": bundle.decode_graph_kernel_nodes(),
                "note": "no per-step readback; correct-class job, not a timing claim",
            });
        }
        cases.push(case_json);
    }
    Ok(json!({
        "mode": "graph",
        "n_tokens": tokens.len(),
        "steps": steps,
        "env_hipfire_graph": std::env::var("HIPFIRE_GRAPH").ok(),
        "definitions": {
            "bitwise_equal": "every captured byte identical to the eager arm of the same route",
            "logits": "full [vocab] F32 logit vector after the prefill and after every decode step",
        },
        "cases": cases,
        "pass": pass,
    }))
}

fn fwd_capture(gpu: &Gpu, b: &RaonBundle, secs: f64) -> Result<FwdSnap, String> {
    let st = &b.state;
    let n = st.n_tokens();
    let mut items = Vec::new();
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        items.push(FwdItem {
            name: format!("rda[{i}]"),
            elem: FwdElem::F32,
            bytes: read_bytes(gpu, r, 0, r.size(), &format!("capture rda[{i}]"))?,
        });
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]]
            .into_iter()
            .enumerate()
        {
            items.push(FwdItem {
                name: format!("conv_{}[{i}]", ["q", "k", "v"][p]),
                elem: FwdElem::F32,
                bytes: read_bytes(
                    gpu,
                    &c.buf,
                    0,
                    c.buf.size(),
                    &format!("capture conv{p}[{i}]"),
                )?,
            });
        }
    }
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        items.push(FwdItem {
            name: format!("latent[{l}]"),
            elem: FwdElem::Bf16,
            bytes: read_bytes(
                gpu,
                &t.buf,
                0,
                n * rb,
                &format!("capture latent[{l}] rows [0,{n})"),
            )?,
        });
    }
    let logits = hip_err(gpu.download_f32(&b.scratch.logits), "capture logits")?;
    items.push(FwdItem {
        name: "logits".to_string(),
        elem: FwdElem::F32,
        bytes: logits.iter().flat_map(|x| x.to_le_bytes()).collect(),
    });
    Ok(FwdSnap {
        items,
        n_tokens: n,
        secs,
    })
}

fn fwd_fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fwd_values(elem: FwdElem, bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    let w = if elem == FwdElem::F32 { 4 } else { 2 };
    bytes.chunks_exact(w).map(move |c| match elem {
        FwdElem::F32 => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        FwdElem::Bf16 => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
    })
}

fn fwd_route_json(s: &FwdSnap) -> Value {
    let mut fp = serde_json::Map::new();
    for it in &s.items {
        fp.insert(
            it.name.clone(),
            json!(format!("{:016x}", fwd_fnv1a(&it.bytes))),
        );
    }
    json!({
        "n_tokens": s.n_tokens,
        "secs": s.secs,
        "tok_per_s": s.n_tokens as f64 / s.secs.max(1e-9),
        "fingerprint_fnv1a64": Value::Object(fp),
    })
}

fn fwd_logits(s: &FwdSnap) -> Result<Vec<f32>, String> {
    let it = s
        .items
        .iter()
        .find(|i| i.name == "logits")
        .ok_or("snapshot lacks logits")?;
    Ok(fwd_values(it.elem, &it.bytes).collect())
}

fn fwd_logits_finite(s: &FwdSnap) -> Result<bool, String> {
    Ok(fwd_logits(s)?.iter().all(|x| x.is_finite()))
}

/// `(cosine, top-1 equal (first-max argmax), both finite)` between two routes.
fn fwd_logits_cross(a: &FwdSnap, b: &FwdSnap) -> Result<(f64, bool, bool), String> {
    let (la, lb) = (fwd_logits(a)?, fwd_logits(b)?);
    if la.len() != lb.len() {
        return Err("logits length differs between routes".into());
    }
    let (mut aa, mut bb, mut ab) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in la.iter().zip(&lb) {
        let (x, y) = (x as f64, y as f64);
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    let cos = ab / (aa.sqrt() * bb.sqrt()).max(1e-300);
    let argmax = |v: &[f32]| {
        let mut best = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    };
    let finite = la.iter().all(|x| x.is_finite()) && lb.iter().all(|x| x.is_finite());
    Ok((cos, argmax(&la) == argmax(&lb), finite))
}

/// Per-tensor comparison of two snapshots (names/sizes must match).
fn fwd_compare(a: &FwdSnap, b: &FwdSnap) -> Result<Value, String> {
    if a.items.len() != b.items.len() {
        return Err(format!(
            "snapshots differ in tensor count: {} vs {}",
            a.items.len(),
            b.items.len()
        ));
    }
    let mut all_equal = true;
    let mut differing = 0usize;
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;
    let mut tensors = Vec::with_capacity(a.items.len());
    for (x, y) in a.items.iter().zip(&b.items) {
        if x.name != y.name || x.elem != y.elem || x.bytes.len() != y.bytes.len() {
            return Err(format!(
                "snapshot tensor mismatch: {} ({} B) vs {} ({} B)",
                x.name,
                x.bytes.len(),
                y.name,
                y.bytes.len()
            ));
        }
        let equal = x.bytes == y.bytes;
        let (mut max_abs, mut peak, mut dd, mut bb, mut nonfinite) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        if !equal {
            for (p, q) in fwd_values(x.elem, &x.bytes).zip(fwd_values(y.elem, &y.bytes)) {
                if !p.is_finite() || !q.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                let (p, q) = (p as f64, q as f64);
                max_abs = max_abs.max((p - q).abs());
                peak = peak.max(q.abs());
                dd += (p - q) * (p - q);
                bb += q * q;
            }
        }
        let max_rel = if peak > 0.0 { max_abs / peak } else { 0.0 };
        let rel_l2 = if bb > 0.0 { (dd / bb).sqrt() } else { 0.0 };
        all_equal &= equal;
        if !equal {
            differing += 1;
        }
        worst_abs = worst_abs.max(max_abs);
        worst_rel = worst_rel.max(max_rel);
        tensors.push(json!({
            "name": x.name,
            "bitwise_equal": equal,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "rel_l2": rel_l2,
            "nonfinite_elems": nonfinite,
        }));
    }
    Ok(json!({
        "bitwise_equal": all_equal,
        "tensors_compared": a.items.len(),
        "tensors_differing": differing,
        "worst_max_abs": worst_abs,
        "worst_max_rel": worst_rel,
        "tensors": tensors,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte patterns
// ─────────────────────────────────────────────────────────────────────────────

fn pat_byte(seed: u64, idx: usize) -> u8 {
    let mut x = seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0xFF) as u8
}

fn view(buf: &DeviceBuffer, off: usize, len: usize, label: &str) -> Result<DeviceBuffer, String> {
    if off.checked_add(len).is_none_or(|e| e > buf.size()) {
        return Err(format!(
            "{label}: range [{off}, {off}+{len}) exceeds device extent {}",
            buf.size()
        ));
    }
    Ok(buf.byte_view(off, len))
}

fn hip_err<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e:?}"))
}

/// Write `len` bytes at `off` where byte `k` is `pat(seed, idx0 + k)`.
fn write_pat(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    seed: u64,
    idx0: usize,
    label: &str,
) -> Result<(), String> {
    let host: Vec<u8> = (0..len).map(|k| pat_byte(seed, idx0 + k)).collect();
    let v = view(buf, off, len, label)?;
    hip_err(gpu.hip.memcpy_htod(&v, &host), label)
}

fn fill_const(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    val: u8,
    label: &str,
) -> Result<(), String> {
    let host = vec![val; len];
    let v = view(buf, off, len, label)?;
    hip_err(gpu.hip.memcpy_htod(&v, &host), label)
}

fn read_bytes(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    let v = view(buf, off, len, label)?;
    let mut host = vec![0u8; len];
    hip_err(gpu.hip.memcpy_dtoh(&mut host, &v), label)?;
    Ok(host)
}

fn check_pat(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    seed: u64,
    idx0: usize,
    label: &str,
) -> Result<(), String> {
    let host = read_bytes(gpu, buf, off, len, label)?;
    for (k, &got) in host.iter().enumerate() {
        let want = pat_byte(seed, idx0 + k);
        if got != want {
            return Err(format!(
                "{label}: first mismatch at byte {k} (abs {}): got 0x{got:02x} want 0x{want:02x}",
                off + k
            ));
        }
    }
    Ok(())
}

fn check_const(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    val: u8,
    label: &str,
) -> Result<(), String> {
    let host = read_bytes(gpu, buf, off, len, label)?;
    if let Some(k) = host.iter().position(|&b| b != val) {
        return Err(format!(
            "{label}: first mismatch at byte {k} (abs {}): got 0x{:02x} want 0x{val:02x}",
            off + k,
            host[k]
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Slot content helpers
// ─────────────────────────────────────────────────────────────────────────────

fn rda_seed(slot: u64, layer: usize) -> u64 {
    slot ^ 0xA5A5_0000 ^ ((layer as u64) << 8)
}
fn conv_seed(slot: u64, layer: usize) -> u64 {
    slot ^ 0x5A5A_0000 ^ ((layer as u64) << 8) ^ 0x77
}
fn row_seed(slot: u64, layer: usize) -> u64 {
    slot ^ 0x1234_0000 ^ ((layer as u64) << 8) ^ 0x31
}

fn fill_recurrent(gpu: &Gpu, st: &RaonState, slot: u64) -> Result<(), String> {
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        write_pat(
            gpu,
            r,
            0,
            r.size(),
            rda_seed(slot, i),
            0,
            &format!("fill rda[{i}]"),
        )?;
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]]
            .into_iter()
            .enumerate()
        {
            let c = &c.buf;
            write_pat(
                gpu,
                c,
                0,
                c.size(),
                conv_seed(slot, i) ^ ((p as u64) << 40),
                0,
                &format!("fill conv{p}[{i}]"),
            )?;
        }
    }
    Ok(())
}

fn verify_recurrent(gpu: &Gpu, st: &RaonState, slot: u64, label: &str) -> Result<(), String> {
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        check_pat(
            gpu,
            r,
            0,
            r.size(),
            rda_seed(slot, i),
            0,
            &format!("{label}: rda[{i}]"),
        )?;
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]]
            .into_iter()
            .enumerate()
        {
            let c = &c.buf;
            check_pat(
                gpu,
                c,
                0,
                c.size(),
                conv_seed(slot, i) ^ ((p as u64) << 40),
                0,
                &format!("{label}: conv{p}[{i}]"),
            )?;
        }
    }
    Ok(())
}

fn verify_recurrent_zero(gpu: &Gpu, st: &RaonState, label: &str) -> Result<(), String> {
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        check_const(gpu, r, 0, r.size(), 0, &format!("{label}: rda[{i}]"))?;
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]]
            .into_iter()
            .enumerate()
        {
            let c = &c.buf;
            check_const(gpu, c, 0, c.size(), 0, &format!("{label}: conv{p}[{i}]"))?;
        }
    }
    Ok(())
}

fn write_rows(
    gpu: &Gpu,
    st: &RaonState,
    slot: u64,
    row0: usize,
    nrows: usize,
) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        write_pat(
            gpu,
            &t.buf,
            row0 * rb,
            nrows * rb,
            row_seed(slot, l),
            row0 * rb,
            &format!("write latent l{l} rows [{row0},+{nrows})"),
        )?;
    }
    Ok(())
}

fn verify_rows(
    gpu: &Gpu,
    st: &RaonState,
    slot: u64,
    row0: usize,
    nrows: usize,
    label: &str,
) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        check_pat(
            gpu,
            &t.buf,
            row0 * rb,
            nrows * rb,
            row_seed(slot, l),
            row0 * rb,
            &format!("{label}: latent l{l} rows [{row0},+{nrows})"),
        )?;
    }
    Ok(())
}

fn poison_rows(gpu: &Gpu, st: &RaonState, row0: usize, row1: usize) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        fill_const(
            gpu,
            &t.buf,
            row0 * rb,
            (row1 - row0) * rb,
            0xFF,
            &format!("poison l{l}"),
        )?;
    }
    Ok(())
}

fn verify_poison(
    gpu: &Gpu,
    st: &RaonState,
    row0: usize,
    row1: usize,
    label: &str,
) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        check_const(
            gpu,
            &t.buf,
            row0 * rb,
            (row1 - row0) * rb,
            0xFF,
            &format!("{label}: poison l{l}"),
        )?;
    }
    Ok(())
}

fn device_seq_len(gpu: &Gpu, st: &RaonState) -> Result<i32, String> {
    let b = read_bytes(gpu, &st.seq_len.buf, 0, 4, "read seq_len")?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn expect_seq(gpu: &Gpu, st: &RaonState, want: usize, label: &str) -> Result<(), String> {
    let dev = device_seq_len(gpu, st)?;
    if dev as usize != want || st.n_tokens() != want {
        return Err(format!(
            "{label}: seq_len device {dev} / host {} != {want}",
            st.n_tokens()
        ));
    }
    Ok(())
}

fn mapped_bytes(gpu: &Gpu, st: &RaonState) -> Vec<usize> {
    st.mla_cache
        .iter()
        .map(|t| gpu.vmm_mapped_bytes(t).unwrap_or(t.buf.size()))
        .collect()
}

fn free_bytes(gpu: &Gpu) -> Result<usize, String> {
    hip_err(gpu.device_mem_info(), "device_mem_info").map(|(f, _)| f)
}

/// Physical delta must cover the ledger and not exceed it by more than
/// allocator rounding. Skipped on unified-memory parts whose free figure is a
/// host-RAM estimate.
fn check_delta(
    gpu: &Gpu,
    before: usize,
    after: usize,
    ledger_physical: usize,
    label: &str,
) -> Result<Value, String> {
    let delta = before.saturating_sub(after);
    if gpu.arch == "gfx1151" {
        return Ok(
            json!({"label": label, "delta_bytes": delta, "ledger_bytes": ledger_physical, "enforced": false}),
        );
    }
    let slack = 64 * MIB;
    if delta + 4 * MIB < ledger_physical || delta > ledger_physical + slack {
        return Err(format!(
            "{label}: physical delta {delta} B vs ledger {ledger_physical} B (allowed -4 MiB..+64 MiB)"
        ));
    }
    Ok(
        json!({"label": label, "delta_bytes": delta, "ledger_bytes": ledger_physical, "enforced": true}),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Checks
// ─────────────────────────────────────────────────────────────────────────────

fn check_slot_isolation(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    const A: u64 = 0xA000;
    const B: u64 = 0xB000;
    let max_seq = 4096;
    let mut a = RaonState::new(gpu, dims, max_seq)?;
    let mut b = match RaonState::new(gpu, dims, max_seq) {
        Ok(b) => b,
        Err(e) => {
            let _ = a.free_gpu(gpu);
            return Err(e);
        }
    };
    let r = (|| -> Check {
        let (la, lb) = (700usize, 3usize);
        fill_recurrent(gpu, &a, A)?;
        fill_recurrent(gpu, &b, B)?;
        a.ensure_capacity(gpu, la)?;
        b.ensure_capacity(gpu, lb)?;
        // Poison every mapped row past each slot's length first, then write
        // the live rows over the front of it.
        let (ca, cb) = (a.capacity_rows(), b.capacity_rows());
        poison_rows(gpu, &a, 0, ca)?;
        poison_rows(gpu, &b, 0, cb)?;
        write_rows(gpu, &a, A, 0, la)?;
        write_rows(gpu, &b, B, 0, lb)?;
        a.set_n_tokens(gpu, la)?;
        b.set_n_tokens(gpu, lb)?;
        expect_seq(gpu, &a, la, "a initial")?;
        expect_seq(gpu, &b, lb, "b initial")?;

        // Interleaved decode-style steps: each slot appends one row and bumps
        // its own length; a's recurrent block is rewritten each step.
        let steps = 8usize;
        let mut la_now = la;
        let mut lb_now = lb;
        for s in 0..steps {
            write_rows(gpu, &a, A, la_now, 1)?;
            la_now += 1;
            a.set_n_tokens(gpu, la_now)?;
            write_pat(
                gpu,
                &a.rda[0].buf,
                0,
                a.rda[0].buf.size(),
                rda_seed(A, 0) ^ (s as u64 + 1),
                0,
                "a rda step",
            )?;
            write_rows(gpu, &b, B, lb_now, 1)?;
            lb_now += 1;
            b.set_n_tokens(gpu, lb_now)?;
            if s % 2 == 1 {
                // extra b-only step: unequal progress
                write_rows(gpu, &b, B, lb_now, 1)?;
                lb_now += 1;
                b.set_n_tokens(gpu, lb_now)?;
            }
        }
        // restore a's layer-0 expected content for the plain verify below.
        write_pat(
            gpu,
            &a.rda[0].buf,
            0,
            a.rda[0].buf.size(),
            rda_seed(A, 0),
            0,
            "a rda restore",
        )?;
        expect_seq(gpu, &a, la_now, "a after interleave")?;
        expect_seq(gpu, &b, lb_now, "b after interleave")?;
        if la_now == lb_now {
            return Err("isolation scenario degenerated to equal lengths".into());
        }
        verify_recurrent(gpu, &a, A, "a after interleave")?;
        verify_recurrent(gpu, &b, B, "b after interleave")?;
        verify_rows(gpu, &a, A, 0, la_now, "a live rows")?;
        verify_rows(gpu, &b, B, 0, lb_now, "b live rows")?;
        verify_poison(gpu, &a, la_now, ca, "a beyond length")?;
        verify_poison(gpu, &b, lb_now, cb, "b beyond length")?;

        // Snapshot / restore on b (graph-warmup contract).
        let mut snap = b.snapshot_alloc(gpu)?;
        b.snapshot_into(gpu, &mut snap)?;
        fill_recurrent(gpu, &b, B ^ 0xFFFF)?;
        b.set_n_tokens(gpu, lb_now + 2)?;
        b.restore(gpu, &snap)?;
        expect_seq(gpu, &b, lb_now, "b after restore")?;
        verify_recurrent(gpu, &b, B, "b after restore")?;
        snap.free_gpu(gpu)?;

        // Failed admission: beyond the reservation. Slot is untouched.
        let mapped_a = mapped_bytes(gpu, &a);
        let ptrs_a: Vec<usize> = a
            .mla_cache
            .iter()
            .map(|t| t.buf.as_ptr() as usize)
            .collect();
        match a.ensure_capacity(gpu, max_seq + 1) {
            Err(e) if e.contains("max_seq") => {}
            other => {
                return Err(format!(
                    "ensure_capacity beyond reservation must be refused, got {other:?}"
                ))
            }
        }
        if a.set_n_tokens(gpu, max_seq + 1).is_ok() {
            return Err("set_n_tokens beyond reservation must be refused".into());
        }
        if mapped_bytes(gpu, &a) != mapped_a || a.capacity_rows() != ca {
            return Err("failed admission changed mapped extent".into());
        }
        expect_seq(gpu, &a, la_now, "a after failed admission")?;
        verify_recurrent(gpu, &a, A, "a after failed admission")?;
        verify_rows(gpu, &a, A, 0, la_now, "a rows after failed admission")?;

        // Abort/reset b: recurrent+conv zero, length 0, mappings and addresses
        // unchanged, neighbor a bit-identical.
        let mapped_b = mapped_bytes(gpu, &b);
        let ptrs_b: Vec<usize> = b
            .mla_cache
            .iter()
            .map(|t| t.buf.as_ptr() as usize)
            .collect();
        b.reset(gpu)?;
        expect_seq(gpu, &b, 0, "b after reset")?;
        verify_recurrent_zero(gpu, &b, "b after reset")?;
        if mapped_bytes(gpu, &b) != mapped_b
            || b.mla_cache
                .iter()
                .map(|t| t.buf.as_ptr() as usize)
                .collect::<Vec<_>>()
                != ptrs_b
        {
            return Err("reset unmapped or moved a latent cache".into());
        }
        verify_recurrent(gpu, &a, A, "a after b reset")?;
        verify_rows(gpu, &a, A, 0, la_now, "a rows after b reset")?;
        expect_seq(gpu, &a, la_now, "a after b reset")?;
        if a.mla_cache
            .iter()
            .map(|t| t.buf.as_ptr() as usize)
            .collect::<Vec<_>>()
            != ptrs_a
        {
            return Err("a latent cache address moved".into());
        }
        // Reuse b for a short new request and make sure nothing of the old
        // request is visible through the recurrent block, then scrub.
        b.scrub_latent(gpu)?;
        let cb2 = b.capacity_rows();
        check_const(
            gpu,
            &b.mla_cache[0].buf,
            0,
            cb2 * dims.latent_row_bytes(),
            0,
            "b scrub",
        )?;
        write_rows(gpu, &b, B ^ 0x99, 0, 2)?;
        b.set_n_tokens(gpu, 2)?;
        verify_rows(gpu, &b, B ^ 0x99, 0, 2, "b new request")?;
        verify_recurrent_zero(gpu, &b, "b new request")?;
        verify_recurrent(gpu, &a, A, "a after b reuse")?;

        // Abort a mid-request too.
        a.reset(gpu)?;
        expect_seq(gpu, &a, 0, "a after reset")?;
        verify_recurrent_zero(gpu, &a, "a after reset")?;
        Ok(json!({
            "slot_a_len": la_now,
            "slot_b_len": lb_now,
            "capacity_rows_a": ca,
            "capacity_rows_b": cb,
            "interleaved_steps": steps,
            "backend": a.backend().as_str(),
        }))
    })();
    let fa = a.free_gpu(gpu);
    let fb = b.free_gpu(gpu);
    match (r, fa, fb) {
        (Ok(v), Ok(()), Ok(())) => Ok(v),
        (Err(e), _, _) => Err(e),
        (Ok(_), Err(e), _) | (Ok(_), _, Err(e)) => Err(e),
    }
}

fn check_ctx_reservation(
    gpu: &mut Gpu,
    dims: &RaonStateDims,
    ctx: usize,
    recurrent: &mut Vec<(usize, usize)>,
) -> Check {
    gpu.drain_pool();
    let free0 = free_bytes(gpu)?;
    let mut st = RaonState::new(gpu, dims, ctx)?;
    let r = (|| -> Check {
        let free1 = free_bytes(gpu)?;
        let ledger = st.bytes_ledger(gpu);
        let backend = st.backend();
        let gran = match backend {
            LatentBackend::Vmm => gpu.vmm_granularity(&st.mla_cache[0]).unwrap_or(0),
            LatentBackend::Dense => 1,
        };
        let budget = state_budget(dims, ctx, backend, gran.max(1))?;
        let initial = ledger.physical_total();
        if initial != budget.initial_physical(dims) {
            return Err(format!(
                "initial ledger {initial} B != budget {} B",
                budget.initial_physical(dims)
            ));
        }
        let delta_initial = check_delta(gpu, free0, free1, initial, "initial")?;
        let rda_conv =
            ledger.physical_with_prefix("rda_state") + ledger.physical_with_prefix("conv_history");
        recurrent.push((ctx, rda_conv));
        let latent_virtual: usize = (0..dims.n_mla)
            .map(|i| {
                ledger
                    .entries
                    .iter()
                    .find(|e| e.owner == format!("latent_cache.mla{i}"))
                    .map_or(0, |e| e.virtual_bytes)
            })
            .sum();
        if latent_virtual < ctx * dims.latent_row_bytes() * dims.n_mla {
            return Err(format!(
                "virtual reservation {latent_virtual} B < {} B needed for {ctx} tokens",
                ctx * dims.latent_row_bytes() * dims.n_mla
            ));
        }

        // Boundary growth with data integrity across the chunk edge.
        let cap0 = st.capacity_rows();
        let mut growth = json!({"grew": false});
        if backend == LatentBackend::Vmm && cap0 < ctx {
            let head = cap0.min(16);
            write_rows(gpu, &st, 0x51, 0, head)?;
            write_rows(gpu, &st, 0x51, cap0 - 1, 1)?;
            if st.ensure_capacity(gpu, cap0)? {
                return Err("ensure_capacity(cap) must not grow".into());
            }
            if !st.ensure_capacity(gpu, cap0 + 1)? {
                return Err("ensure_capacity(cap+1) must grow".into());
            }
            let cap1 = st.capacity_rows();
            if cap1 <= cap0 {
                return Err(format!("capacity did not increase: {cap0} -> {cap1}"));
            }
            verify_rows(gpu, &st, 0x51, 0, head, "head after first growth")?;
            verify_rows(gpu, &st, 0x51, cap0 - 1, 1, "boundary row after growth")?;
            write_rows(gpu, &st, 0x52, cap0, 1)?;
            verify_rows(gpu, &st, 0x52, cap0, 1, "first new row")?;
            st.ensure_capacity(gpu, ctx)?;
            if st.capacity_rows() != ctx {
                return Err(format!(
                    "capacity {} != ctx {ctx} after full growth",
                    st.capacity_rows()
                ));
            }
            write_rows(gpu, &st, 0x53, ctx - 1, 1)?;
            verify_rows(gpu, &st, 0x53, ctx - 1, 1, "last row")?;
            verify_rows(gpu, &st, 0x51, 0, head, "head after full growth")?;
            verify_rows(
                gpu,
                &st,
                0x51,
                cap0 - 1,
                1,
                "boundary row after full growth",
            )?;
            verify_rows(gpu, &st, 0x52, cap0, 1, "first new row after full growth")?;
            if st.ensure_capacity(gpu, ctx + 1).is_ok() {
                return Err("ensure_capacity(ctx+1) must be refused".into());
            }
            let full_ledger = st.bytes_ledger(gpu);
            if full_ledger.physical_total() != budget.full_physical(dims) {
                return Err(format!(
                    "full ledger {} B != budget {} B",
                    full_ledger.physical_total(),
                    budget.full_physical(dims)
                ));
            }
            let free2 = free_bytes(gpu)?;
            let delta_full = check_delta(gpu, free0, free2, full_ledger.physical_total(), "full")?;
            growth = json!({
                "grew": true,
                "initial_capacity_rows": cap0,
                "after_first_growth_rows": cap1,
                "full_ledger_bytes": full_ledger.physical_total(),
                "delta_full": delta_full,
            });
        } else if !st.ensure_capacity(gpu, ctx)? && st.capacity_rows() < ctx {
            return Err("capacity short of ctx without growth".into());
        }
        Ok(json!({
            "ctx": ctx,
            "backend": backend.as_str(),
            "granularity": gran,
            "virtual_latent_bytes": latent_virtual,
            "initial_physical_bytes": initial,
            "rda_plus_conv_bytes": rda_conv,
            "delta_initial": delta_initial,
            "growth": growth,
            "ledger_initial": ledger.to_json(),
        }))
    })();
    let freed = st.free_gpu(gpu);
    gpu.drain_pool();
    match (r, freed) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_recurrent_constant(per_ctx: &[(usize, usize)]) -> Check {
    let Some(&(_, first)) = per_ctx.first() else {
        return Err("no ctx reservation produced a recurrent byte count".into());
    };
    let d = RaonStateDims::raon();
    let expected_rda = d.n_rda * d.rda_state_elems() * 4;
    if first < expected_rda {
        return Err(format!("recurrent bytes {first} < 45 MiB ({expected_rda})"));
    }
    for &(ctx, b) in per_ctx {
        if b != first {
            return Err(format!(
                "recurrent bytes at ctx {ctx} = {b}, at ctx {} = {first}",
                per_ctx[0].0
            ));
        }
    }
    Ok(
        json!({"rda_plus_conv_bytes": first, "rda_bytes": expected_rda, "ctxs": per_ctx.iter().map(|c| c.0).collect::<Vec<_>>()}),
    )
}

fn check_admission_refusal(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    let free0 = free_bytes(gpu)?;
    let row_all = dims.latent_row_bytes() * dims.n_mla;
    let ctx = (free0 / row_all).saturating_mul(2).saturating_add(4096);
    match RaonState::new(gpu, dims, ctx) {
        Ok(st) => {
            let _ = st.free_gpu(gpu);
            Err(format!(
                "max_seq {ctx} ({} MiB latent) must be refused with {} MiB free",
                ctx * row_all / MIB,
                free0 / MIB
            ))
        }
        Err(e) => {
            if !e.contains("refused") {
                return Err(format!("refusal must be a clean named refusal, got: {e}"));
            }
            let free1 = free_bytes(gpu)?;
            if free0.abs_diff(free1) > 16 * MIB {
                return Err(format!(
                    "refused admission changed free memory {free0} -> {free1}"
                ));
            }
            Ok(json!({"max_seq": ctx, "message": e}))
        }
    }
}

fn check_admission_262144(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    let ctx = 262_144;
    let backend = auto_backend(gpu);
    let gran = match backend {
        LatentBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|e| format!("granularity: {e:?}"))?,
        LatentBackend::Dense => 1,
    };
    let budget = state_budget(dims, ctx, backend, gran)?;
    let need = budget.full_physical(dims) + ADMISSION_HEADROOM_BYTES;
    gpu.drain_pool();
    let free = free_bytes(gpu)?;
    let supported = need <= free;
    match RaonState::new(gpu, dims, ctx) {
        Ok(mut st) => {
            if !supported && gpu.arch != "gfx1151" {
                let _ = st.free_gpu(gpu);
                return Err(format!(
                    "262144 admitted although need {need} B > free {free} B"
                ));
            }
            let r = (|| -> Check {
                st.ensure_capacity(gpu, ctx)?;
                if st.capacity_rows() != ctx {
                    return Err("full growth to 262144 incomplete".into());
                }
                write_rows(gpu, &st, 0x262, ctx - 1, 1)?;
                verify_rows(gpu, &st, 0x262, ctx - 1, 1, "row 262143")?;
                Ok(
                    json!({"outcome": "admitted", "need_bytes": need, "free_bytes": free, "latent_mib": st.bytes_ledger(gpu).physical_with_prefix("latent_cache") / MIB}),
                )
            })();
            let f = st.free_gpu(gpu);
            gpu.drain_pool();
            match (r, f) {
                (Ok(v), Ok(())) => Ok(v),
                (Err(e), _) => Err(e),
                (Ok(_), Err(e)) => Err(e),
            }
        }
        Err(e) => {
            if supported {
                return Err(format!(
                    "262144 refused although need {need} B <= free {free} B: {e}"
                ));
            }
            if !e.contains("refused") {
                return Err(format!("262144 refusal is not a named refusal: {e}"));
            }
            Ok(json!({"outcome": "refused", "need_bytes": need, "free_bytes": free, "message": e}))
        }
    }
}

fn check_growth_pressure(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    let ctx = 65_536;
    gpu.drain_pool();
    let mut st = RaonState::new(gpu, dims, ctx)?;
    if st.backend() != LatentBackend::Vmm {
        let f = st.free_gpu(gpu);
        f?;
        return Ok(json!({"skipped": "dense latent backend has no growth"}));
    }
    let r = (|| -> Check {
        let budget = state_budget(
            dims,
            ctx,
            LatentBackend::Vmm,
            gpu.vmm_granularity(&st.mla_cache[0]).unwrap_or(1),
        )?;
        let needed = budget.full_physical(dims) - st.bytes_ledger(gpu).physical_total();
        let free_now = free_bytes(gpu)?;
        let keep = needed / 2 + 64 * MIB;
        if free_now <= keep + 64 * MIB || gpu.arch == "gfx1151" {
            return Ok(
                json!({"skipped": "not enough free memory for a ballast test or UMA device", "free_bytes": free_now}),
            );
        }
        let ballast = hip_err(
            gpu.alloc_tensor_exact(&[free_now - keep], DType::Raw),
            "alloc ballast",
        )?;
        let outcome = (|| -> Check {
            let before = mapped_bytes(gpu, &st);
            let cap = st.capacity_rows();
            write_rows(gpu, &st, 0x77, 0, 4)?;
            match st.ensure_capacity(gpu, ctx) {
                Err(e) if e.contains("refused") => {}
                other => {
                    return Err(format!(
                        "growth under pressure must be refused, got {other:?}"
                    ))
                }
            }
            if mapped_bytes(gpu, &st) != before || st.capacity_rows() != cap {
                return Err("refused growth mutated the mapped extent".into());
            }
            verify_rows(gpu, &st, 0x77, 0, 4, "rows after refused growth")?;
            st.reset(gpu)?;
            Ok(json!({"needed_bytes": needed, "free_with_ballast": free_bytes(gpu)?}))
        })();
        gpu.bind_thread().map_err(|e| format!("bind: {e:?}"))?;
        hip_err(gpu.hip.free(ballast.buf), "free ballast")?;
        let v = outcome?;
        if !st.ensure_capacity(gpu, ctx)? || st.capacity_rows() != ctx {
            return Err("growth must succeed after pressure is released".into());
        }
        verify_rows(gpu, &st, 0x77, 0, 4, "rows after retried growth")?;
        Ok(v)
    })();
    let f = st.free_gpu(gpu);
    gpu.drain_pool();
    match (r, f) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_scratch(gpu: &mut Gpu, dims: &RaonStateDims, rows: usize) -> Check {
    gpu.drain_pool();
    let plan = scratch_plan(dims, rows)?;
    let free0 = free_bytes(gpu)?;
    let sc = RaonPrefillScratch::new(gpu, dims, rows)?;
    let r = (|| -> Check {
        let free1 = free_bytes(gpu)?;
        if sc.bytes() != plan.total_bytes
            || sc.bytes() != RaonPrefillScratch::required_bytes(dims, rows)?
        {
            return Err("scratch bytes != plan bytes".into());
        }
        let delta = check_delta(gpu, free0, free1, sc.bytes(), "scratch")?;
        // Views follow the plan and start zeroed.
        let base = sc.token_ids.buf.as_ptr() as usize;
        for (name, ptr, len) in [
            (
                "positions",
                sc.positions.buf.as_ptr() as usize,
                sc.positions.buf.size(),
            ),
            (
                "stream_a",
                sc.stream_a.buf.as_ptr() as usize,
                sc.stream_a.buf.size(),
            ),
            (
                "stream_b",
                sc.stream_b.buf.as_ptr() as usize,
                sc.stream_b.buf.size(),
            ),
            (
                "mla_kv_tile",
                sc.mla_kv_tile.buf.as_ptr() as usize,
                sc.mla_kv_tile.buf.size(),
            ),
            (
                "logits",
                sc.logits.buf.as_ptr() as usize,
                sc.logits.buf.size(),
            ),
        ] {
            let e = plan
                .entry(name)
                .ok_or_else(|| format!("plan lacks {name}"))?;
            if ptr != base + e.offset || len != e.bytes {
                return Err(format!(
                    "{name}: view does not match plan (ptr off {} vs {}, len {len} vs {})",
                    ptr.wrapping_sub(base),
                    e.offset,
                    e.bytes
                ));
            }
        }
        check_const(
            gpu,
            &sc.logits.buf,
            0,
            sc.logits.buf.size(),
            0,
            "logits zero-init",
        )?;
        check_const(
            gpu,
            &sc.mla_kv_tile.buf,
            0,
            sc.mla_kv_tile.buf.size(),
            0,
            "mla_kv_tile zero-init",
        )?;
        // A write to one buffer must not leak into a neighbor.
        fill_const(
            gpu,
            &sc.stream_a.buf,
            0,
            sc.stream_a.buf.size(),
            0xAB,
            "stream_a fill",
        )?;
        check_const(
            gpu,
            &sc.stream_b.buf,
            0,
            sc.stream_b.buf.size(),
            0,
            "stream_b untouched",
        )?;
        check_const(
            gpu,
            &sc.positions.buf,
            0,
            sc.positions.buf.size(),
            0,
            "positions untouched",
        )?;
        Ok(
            json!({"rows": rows, "bytes": sc.bytes(), "mib": sc.bytes() / MIB, "delta": delta, "buffers": plan.entries.len()}),
        )
    })();
    let f = sc.free_gpu(gpu);
    gpu.drain_pool();
    match (r, f) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_scratch_limit(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    if RaonPrefillScratch::new(gpu, dims, 0).is_ok() {
        return Err("0-row scratch must be refused".into());
    }
    if RaonPrefillScratch::new(gpu, dims, 1 << 20).is_ok() {
        return Err("oversized scratch must be refused".into());
    }
    Ok(json!({"zero_rows": "refused", "oversized": "refused"}))
}

fn check_mirror(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    gpu.drain_pool();
    let mut st = RaonState::new(gpu, dims, 16_384)?;
    let r = (|| -> Check {
        let budget = st.mirror_budget_rows();
        if budget == 0 || budget > 16_384 {
            return Err(format!(
                "mirror admission budget {budget} rows outside 1..=16384"
            ));
        }
        // A forced low budget makes everything above it a clear capacity refusal.
        st.set_mirror_budget_rows(4096);
        match st.ensure_mirror_rows(gpu, 4097) {
            Ok(_) => return Err("mirror above the admission budget must be refused".into()),
            Err(e) if e.contains("maximum context of 4096 tokens") => {}
            Err(e) => {
                return Err(format!(
                    "capacity refusal does not state the maximum context: {e}"
                ))
            }
        }
        if st.ensure_mirror_rows(gpu, 0)? {
            return Err("0-row mirror request must be a no-op".into());
        }
        if st.expanded_mirror().is_some() {
            return Err("refused/no-op requests must not create a mirror".into());
        }
        let rows = 2048;
        let bytes = expanded_mirror_bytes(dims, rows)?;
        let before = st.bytes_ledger(gpu).physical_total();
        st.ensure_mirror_rows(gpu, rows)?;
        let after = st.bytes_ledger(gpu);
        let mirror = after.physical_with_prefix("expanded_mirror");
        if mirror < bytes
            || mirror > bytes + 64 * 1024 * 1024
            || after.physical_total() != before + mirror
        {
            return Err(format!(
                "mirror ledger {mirror} B not within one growth chunk of {bytes} B"
            ));
        }
        let ptrs: Vec<usize> = {
            let m = st.expanded_mirror().ok_or("mirror missing after ensure")?;
            if m.knope.len() != dims.n_mla || m.vmir.len() != dims.n_mla || m.rows < rows {
                return Err("mirror view count/rows mismatch".into());
            }
            if m.knope[0].shape != vec![16_384, dims.heads, 128]
                || m.vmir[0].shape != vec![16_384, dims.heads, 128]
            {
                return Err("mirror views are not token-major [rows, heads, 128]".into());
            }
            let tok = dims.heads * 128 * 2;
            write_pat(
                gpu,
                &m.knope[3].buf,
                0,
                rows * tok,
                0x3C,
                0,
                "mirror knope write",
            )?;
            check_const(
                gpu,
                &m.vmir[3].buf,
                0,
                rows * tok,
                0,
                "mirror vmir untouched",
            )?;
            check_const(
                gpu,
                &m.knope[4].buf,
                0,
                rows * tok,
                0,
                "mirror next-layer knope untouched",
            )?;
            m.knope
                .iter()
                .chain(&m.vmir)
                .map(|t| t.buf.as_ptr() as usize)
                .collect()
        };
        // Growth maps in place: no address moves, earlier rows survive.
        st.ensure_mirror_rows(gpu, 4096)?;
        {
            let m = st.expanded_mirror().ok_or("mirror missing after growth")?;
            let now: Vec<usize> = m
                .knope
                .iter()
                .chain(&m.vmir)
                .map(|t| t.buf.as_ptr() as usize)
                .collect();
            if now != ptrs {
                return Err("mirror growth moved a plane address".into());
            }
            if m.rows < 4096 {
                return Err(format!("mirror backs {} rows after growth to 4096", m.rows));
            }
            let tok = dims.heads * 128 * 2;
            check_pat(
                gpu,
                &m.knope[3].buf,
                0,
                rows * tok,
                0x3C,
                0,
                "mirror rows survive growth",
            )?;
        }
        st.release_expanded_mirror(gpu)?;
        if st.expanded_mirror().is_some() || st.bytes_ledger(gpu).physical_total() != before {
            return Err("mirror release did not return the ledger".into());
        }
        Ok(
            json!({"rows": rows, "bytes": bytes, "per_token_bytes": bytes / rows, "budget_rows_default": budget}),
        )
    })();
    let f = st.free_gpu(gpu);
    gpu.drain_pool();
    match (r, f) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_vmm_clean(gpu: &mut Gpu) -> Check {
    hip_err(gpu.ensure_vmm_cleaned(), "ensure_vmm_cleaned")?;
    let n = gpu.vmm_allocation_count();
    if n != 0 {
        return Err(format!("{n} VMM owner(s) still registered"));
    }
    Ok(json!({"vmm_allocation_count": n}))
}
