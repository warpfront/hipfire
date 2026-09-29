// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Split-prefill numerics probe (decide investigation).
//!
//! For one fixed prompt, computes last-position logits through the production
//! decide hook (`Carrier::decide_prefill_logits`) under different segmentations
//! of the same token sequence, then compares log-softmax over the full vocab:
//!
//! - `A`      one prefill of all N tokens from position 0;
//! - `B(k)`   prefill `[0,k)` from 0, then `[k,N)` at `start_pos=k`;
//! - `C`      token-by-token (each call has 1 token → per-token/decode path);
//! - `S(c)`   N tokens in equal segments of `c` (several start_pos>0 calls).
//!
//! It also downloads the per-layer K/V cache after A and after each B(k) and
//! reports, per layer, how many cache positions differ byte-wise inside the
//! prefix `[0,k)` and the suffix `[k,N)` — this separates "the prefix rows are
//! computed differently when batched with more rows" from "the suffix rows are
//! computed differently at start_pos>0".
//!
//! Usage:
//!   cargo run --release -p hipfire-generate --example split_prefill_probe -- \
//!       ~/.hipfire/models/qwen3-0.6b.hf4 [N]
//!
//! Env: `PROBE_SPLITS=k1,k2,..` overrides the split points; `PROBE_KV=0`
//! skips the KV-cache byte diff; `PROBE_SEGS=c1,c2` overrides segment sizes;
//! `PROBE_KV_MODE` (default q8) and `PROBE_STATE_QUANT` (qwen35 DeltaNet
//! state: q8|fp32|q4) pass through to `load_model`.
//!
//! Decide noise-floor mode: `PROBE_DECIDE_REQUEST=<file.json>` (a decide
//! request `{"state":..,"questions":..}`) renders every question's prompt
//! exactly as `run_decide` does, runs it as A (one call) and C (token by
//! token), and prints one line `DECIDE_FLOOR {json}` with, per question, the
//! max |Δ log p| between A's and C's label distributions (softmax over the
//! option-code logits, the quantity decide answers are built from) and A's and
//! C's answers assembled exactly as the daemon's (`a_answer`, `c_answer`). Used by
//! `scripts/jev_eval/gates.py` gate 1b as the per-model path-noise floor.

use hipfire_loader::{Carrier, LoadedModel};
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};

const TEXT: &str = "The customer wrote in on Tuesday afternoon to say that the order \
they placed three weeks ago has still not arrived, even though the tracking page \
has shown it as out for delivery since last Friday. They have already contacted \
the courier twice and were told to speak to the retailer. They are polite but \
clearly frustrated, and they mention that the parcel contains a birthday present \
for their daughter whose party is this Saturday. They would like to know whether \
a replacement can be sent by express shipping or whether they should ask for a \
refund instead. Classify the main topic of this message as one of billing, \
shipping, technical or general. Answer with a single word:";

fn logsoftmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = x.iter().map(|&v| ((v as f64) - m).exp()).sum();
    let l = m + s.ln();
    x.iter().map(|&v| v as f64 - l).collect()
}

fn top5(lp: &[f64]) -> Vec<(usize, f64)> {
    let mut idx: Vec<usize> = (0..lp.len()).collect();
    idx.sort_by(|&a, &b| lp[b].partial_cmp(&lp[a]).unwrap());
    idx.into_iter().take(5).map(|i| (i, lp[i])).collect()
}

struct Cmp {
    max_abs: f64,
    max_abs_top5: f64,
    kl: f64,
    argmax_same: bool,
}

fn compare(a: &[f64], b: &[f64]) -> Cmp {
    let mut max_abs = 0.0f64;
    let mut kl = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        kl += a[i].exp() * (a[i] - b[i]);
    }
    let ta = top5(a);
    let tb = top5(b);
    let max_abs_top5 = ta
        .iter()
        .map(|&(i, _)| (a[i] - b[i]).abs())
        .fold(0.0, f64::max);
    Cmp {
        max_abs,
        max_abs_top5,
        kl,
        argmax_same: ta[0].0 == tb[0].0,
    }
}

fn rollback(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) {
    let ep = hipfire_generate::common::production_fail_closed_rollback(m, gpu, None, None);
    assert!(ep.rolled_back, "rollback failed: {:?}", ep.context);
}

/// Run the sequence as consecutive segments via the production decide hook.
fn run_segments(
    carrier: &dyn Carrier,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    toks: &[u32],
    seg_lens: &[usize],
) -> Vec<f32> {
    rollback(m, gpu);
    let mut pos = 0usize;
    let mut logits = Vec::new();
    for &len in seg_lens {
        logits = carrier
            .decide_prefill_logits(m, gpu, &toks[pos..pos + len], pos, true)
            .expect("hook")
            .expect("prefill");
        pos += len;
    }
    assert_eq!(pos, toks.len());
    gpu.hip.device_synchronize().expect("sync");
    logits
}

/// Per-layer raw K and V cache bytes (layers with empty buffers skipped).
fn kv_bytes(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Vec<(usize, Vec<u8>, Vec<u8>, usize)> {
    let kv = if let Some(b) = m.llama_mut() {
        &b.kv
    } else if let Some(b) = m.qwen35_mut() {
        &b.kv_cache
    } else {
        return Vec::new();
    };
    let cap = kv.physical_cap;
    let mut out = Vec::new();
    for l in 0..kv.k_gpu.len() {
        let kt = &kv.k_gpu[l];
        let vt = &kv.v_gpu[l];
        let (ks, vs) = (kt.byte_size(), vt.byte_size());
        if ks == 0 || vs == 0 || ks % cap != 0 {
            continue;
        }
        let mut kb = vec![0u8; ks];
        let mut vb = vec![0u8; vs];
        gpu.hip.memcpy_dtoh(&mut kb, &kt.buf).expect("dl k");
        gpu.hip.memcpy_dtoh(&mut vb, &vt.buf).expect("dl v");
        out.push((l, kb, vb, cap));
    }
    out
}

/// Count differing positions in [lo,hi) given a position-major layout.
fn diff_positions(a: &[u8], b: &[u8], cap: usize, lo: usize, hi: usize) -> (usize, Option<usize>) {
    let stride = a.len() / cap;
    let mut n = 0;
    let mut first = None;
    for p in lo..hi {
        if a[p * stride..(p + 1) * stride] != b[p * stride..(p + 1) * stride] {
            n += 1;
            if first.is_none() {
                first = Some(p);
            }
        }
    }
    (n, first)
}

fn parse_list(var: &str) -> Option<Vec<usize>> {
    std::env::var(var).ok().map(|s| {
        s.split(',')
            .filter(|t| !t.is_empty())
            .map(|t| t.trim().parse().expect("usize"))
            .collect()
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: split_prefill_probe <model> [N]");
    let want_n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);
    let do_kv = std::env::var("PROBE_KV").as_deref() != Ok("0");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let spec = SpecLoadCfg {
        ngram_draft: Some(false),
        dflash: Some(false),
        mtp: Some(false),
        dspark: Some(false),
        ..Default::default()
    };
    let kv_mode = std::env::var("PROBE_KV_MODE").unwrap_or_else(|_| "q8".into());
    let state_quant = std::env::var("PROBE_STATE_QUANT").ok();
    println!("kv_mode={kv_mode} state_quant={state_quant:?}");
    let mut m = hipfire_loader::load_model(
        path,
        PROBE_MAX_SEQ,
        None,
        Some(kv_mode.as_str()),
        None,
        state_quant.as_deref(),
        &CaskConfig::default(),
        1,
        spec,
        &mut gpu,
    )
    .expect("load");
    let carrier = hipfire_loader::carrier_for(m.arch_id).expect("carrier");
    assert!(carrier.decide_supported(), "carrier has no decide hooks");

    if let Ok(req_path) = std::env::var("PROBE_DECIDE_REQUEST") {
        decide_floor(carrier, &mut m, &mut gpu, &req_path);
        rollback(&mut m, &mut gpu);
        return;
    }

    let mut toks = m.tokenizer.as_ref().expect("tokenizer").encode(TEXT);
    // Repeat the text if short, then truncate to N.
    while toks.len() < want_n {
        let more = toks.clone();
        toks.extend(more);
    }
    toks.truncate(want_n);
    let n = toks.len();
    check_fits(n);
    println!(
        "model={path} arch_id={} carrier={} N={n}",
        m.arch_id,
        carrier.name()
    );
    if let Some(b) = m.llama_mut() {
        let l = &b.weights.layers[0];
        println!(
            "llama layer0 dtypes: wq={:?} wo={:?} w_gate={:?} w_down={:?}",
            l.wq.gpu_dtype, l.wo.gpu_dtype, l.w_gate.gpu_dtype, l.w_down.gpu_dtype
        );
    }

    let splits = parse_list("PROBE_SPLITS").unwrap_or_else(|| {
        let mut v = vec![
            n - 1,
            n - 2,
            n - 3,
            n - 4,
            n - 8,
            n - 32,
            n / 2,
            64,
            63,
            65,
            32,
            4,
            1,
        ];
        v.retain(|&k| k > 0 && k < n);
        v.dedup();
        v
    });
    let segs = parse_list("PROBE_SEGS").unwrap_or_else(|| vec![4, 8, 16, 32]);

    // Reference A (run twice: determinism).
    let a1 = run_segments(carrier, &mut m, &mut gpu, &toks, &[n]);
    let kv_a = if do_kv {
        kv_bytes(&mut m, &mut gpu)
    } else {
        Vec::new()
    };
    let a2 = run_segments(carrier, &mut m, &mut gpu, &toks, &[n]);
    let la = logsoftmax(&a1);
    let det = compare(&la, &logsoftmax(&a2));
    println!("determinism A vs A: max|dlogp|={:.3e}", det.max_abs);
    let t = top5(&la);
    println!(
        "A top5: {:?}",
        t.iter()
            .map(|&(i, l)| (i, format!("{:.4}", l.exp())))
            .collect::<Vec<_>>()
    );

    // C: token-by-token.
    let ones = vec![1usize; n];
    let lc = logsoftmax(&run_segments(carrier, &mut m, &mut gpu, &toks, &ones));
    let lc_t = top5(&lc);
    println!(
        "C top5: {:?}",
        lc_t.iter()
            .map(|&(i, l)| (i, format!("{:.4}", l.exp())))
            .collect::<Vec<_>>()
    );
    let ac = compare(&la, &lc);
    println!(
        "PAIR A-C            max|dlogp|={:.4} top5={:.4} KL={:.2e} argmax_same={}",
        ac.max_abs, ac.max_abs_top5, ac.kl, ac.argmax_same
    );

    println!("--- B(k): prefill [0,k) then [k,N) at start_pos=k ---");
    for &k in &splits {
        let lb = logsoftmax(&run_segments(carrier, &mut m, &mut gpu, &toks, &[k, n - k]));
        let ab = compare(&la, &lb);
        let bc = compare(&lb, &lc);
        println!(
            "B k={k:4} sfx={:4}  A-B max={:.4} top5={:.4} KL={:.2e} | B-C max={:.4} top5={:.4} KL={:.2e}",
            n - k,
            ab.max_abs,
            ab.max_abs_top5,
            ab.kl,
            bc.max_abs,
            bc.max_abs_top5,
            bc.kl
        );
        if do_kv {
            let kv_b = kv_bytes(&mut m, &mut gpu);
            let mut line = String::new();
            let mut tot = [0usize; 4];
            for ((l, ka, va, cap), (_, kb, vb, _)) in kv_a.iter().zip(kv_b.iter()) {
                let (kp, kpf) = diff_positions(ka, kb, *cap, 0, k);
                let (ks, _) = diff_positions(ka, kb, *cap, k, n);
                let (vp, _) = diff_positions(va, vb, *cap, 0, k);
                let (vs, _) = diff_positions(va, vb, *cap, k, n);
                tot[0] += kp;
                tot[1] += ks;
                tot[2] += vp;
                tot[3] += vs;
                if line.len() < 400 && (kp + ks + vp + vs) > 0 {
                    line.push_str(&format!(" L{l}:K{kp}/{ks}(first {:?}) V{vp}/{vs};", kpf));
                }
            }
            println!(
                "    KV diff A vs B (positions differing, prefix/suffix, summed over {} layers): K {}/{}  V {}/{}  first layers:{}",
                kv_a.len(),
                tot[0],
                tot[1],
                tot[2],
                tot[3],
                if line.is_empty() { " none".into() } else { line }
            );
        }
    }

    println!("--- S(c): equal segments of c tokens ---");
    for &c in &segs {
        let mut v = Vec::new();
        let mut r = n;
        while r > 0 {
            let s = c.min(r);
            v.push(s);
            r -= s;
        }
        let ls = logsoftmax(&run_segments(carrier, &mut m, &mut gpu, &toks, &v));
        let a_s = compare(&la, &ls);
        let sc = compare(&ls, &lc);
        println!(
            "S c={c:4}  A-S max={:.4} top5={:.4} KL={:.2e} | S-C max={:.4} top5={:.4} KL={:.2e}",
            a_s.max_abs, a_s.max_abs_top5, a_s.kl, sc.max_abs, sc.max_abs_top5, sc.kl
        );
    }
    rollback(&mut m, &mut gpu);
}

/// The probe loads with this `max_seq`. The carriers' prefill has no bounds
/// check, so a longer prompt is an illegal GPU memory access, not an error.
const PROBE_MAX_SEQ: usize = 512;

/// Refuse a prompt the probe's KV cannot hold, before it reaches the GPU.
fn check_fits(n: usize) {
    assert!(
        n < PROBE_MAX_SEQ,
        "prompt is {n} tokens; the probe loads with max_seq {PROBE_MAX_SEQ} (needs n + 1 <= max_seq)"
    );
}

/// Decide noise floor: per question, A (one call) vs C (token by token) on the
/// exact prompt `run_decide` renders, compared over the label distribution.
fn decide_floor(
    carrier: &dyn Carrier,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    req_path: &str,
) {
    use hipfire_engine::decide as d;
    let req: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(req_path).expect("read request"))
            .expect("request json");
    let parsed = d::parse_request(&req).expect("parse_request");
    // Exactly the daemon's rendering (shared with run_decide).
    let rendered = {
        let tok = m.tokenizer.as_ref().expect("tokenizer");
        hipfire_generate::decide::render_question_prompts(tok, m.chat_template.as_ref(), &parsed)
            .expect("render_question_prompts")
    };
    let mut per = serde_json::Map::new();
    let mut worst = 0.0f64;
    let prompts = parsed.questions.iter().zip(&rendered.label_ids);
    for ((q, ids), toks) in prompts.zip(&rendered.seqs) {
        let n = toks.len();
        check_fits(n);
        let pick = |l: &[f32]| ids.iter().map(|&i| l[i as usize]).collect::<Vec<f32>>();
        let la = pick(&run_segments(carrier, m, gpu, toks, &[n]));
        let lc = pick(&run_segments(carrier, m, gpu, toks, &vec![1usize; n]));
        let (a, c) = (d::softmax(&la), d::softmax(&lc));
        let dl = a
            .iter()
            .zip(&c)
            .map(|(p, q)| (p.max(1e-12).ln() - q.max(1e-12).ln()).abs())
            .fold(0.0, f64::max);
        worst = worst.max(dl);
        // Assembled exactly as the daemon's answers, so gates.py can compare
        // them by answer key against the daemon's full-prefill reply.
        per.insert(
            q.name.clone(),
            serde_json::json!({
                "tokens": n,
                "a_vs_c": dl,
                "a_answer": d::assemble_answer(q, &la),
                "c_answer": d::assemble_answer(q, &lc),
            }),
        );
    }
    println!(
        "DECIDE_FLOOR {}",
        serde_json::json!({"arch_id": m.arch_id, "max": worst, "per_question": per})
    );
}
