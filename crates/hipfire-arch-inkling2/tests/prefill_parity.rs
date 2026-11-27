// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Blocked-prefill oracle (`forward_prefill_gpu`) against the scalar loop
//! (`forward_token_gpu` per row) on the typed-XTS artifact, gfx1201.
//!
//! One process, three arms over the same `N_TOK` prompt rows on one load:
//! 1. scalar: one `forward_token_gpu` per row (visit traces and logits at the
//!    probe rows), then the final state and `DECODE` continuation logits;
//! 2. row-exact prefill (`PrefillMode::RowExact`, odd block size so blocks end
//!    mid-sequence and the last block is partial): must equal arm 1 **bit for
//!    bit**: every probe row's 48 `visit_out`, every probe row's logits, KV
//!    bytes, conv history, position tags, cursors, and the decode logits;
//! 3. blocked prefill (`PrefillMode::Blocked`, product arithmetic): the same
//!    structural state exactly (tags, cursors, `seq_pos`), numeric state and
//!    logits within the reported budget (GEMM / grouped / FA2 arithmetic).
//!
//! Probe rows span block boundaries, positions 1023/1024/1025 (relative-bias
//! extent and the first SWA ring wrap at 1024 rows) and the partial tail.
//!
//! ```sh
//! INKLING2_MQ4=.../inkling2-mini-xts-c.hfq HIPFIRE_GRAPH=0 \
//!   cargo test --release -p hipfire-arch-inkling2 --test prefill_parity -- \
//!   --ignored --nocapture --exact blocked_prefill_oracle
//! ```

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::gpu::upload;
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{
    forward_prefill_gpu_with, forward_token_gpu, forward_token_gpu_trace, prefill_supported,
    PrefillLogits, PrefillMode, PrefillRoutes,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

const N_TOK: usize = 1100;
const MAX_SEQ: usize = 1152;
const DECODE: usize = 6;
const EXACT_BLOCK: usize = 200;
const PROBES: [usize; 16] = [
    0, 1, 2, 3, 199, 200, 201, 511, 512, 999, 1000, 1022, 1023, 1024, 1025, 1099,
];

fn tokens() -> Vec<u32> {
    // Deterministic ids inside the vocabulary (LCG); content is irrelevant to parity.
    let mut s = 0x1234_5678_9abc_def0u64;
    (0..N_TOK)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 199_000) as u32 + 100
        })
        .collect()
}

fn load(gpu: &mut Gpu) -> Inkling2Bundle {
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "the oracle requires HIPFIRE_GRAPH=0"
    );
    let art = std::env::var("INKLING2_MQ4").expect("set INKLING2_MQ4 to the typed-XTS .hfq");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("weights");
    let (gw, gs) = upload(&weights, &cfg, gpu, MAX_SEQ).expect("upload");
    Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    }
}

#[derive(Default)]
struct Arm {
    /// Probe row -> logits.
    logits: Vec<Vec<f32>>,
    /// Probe row -> 48 visit outputs.
    visits: Vec<Vec<Vec<f32>>>,
    /// Argmax per prompt row.
    argmax: Vec<usize>,
    /// Live KV bytes per visit (K then V, head-major).
    kv: Vec<Vec<u8>>,
    conv: Vec<Vec<f32>>,
    tags: Vec<Vec<u32>>,
    caps: Vec<usize>,
    seq_pos: usize,
    decode: Vec<Vec<f32>>,
    seconds: f64,
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn reset(b: &mut Inkling2Bundle, gpu: &mut Gpu) {
    b.state.seq_pos = 0;
    b.gpu_state
        .as_mut()
        .unwrap()
        .reset_session_state(gpu)
        .expect("reset");
}

fn snapshot(b: &Inkling2Bundle, gpu: &Gpu, arm: &mut Arm) {
    let gs = b.gpu_state.as_ref().unwrap();
    let hd = b.config.head_dim;
    for v in 0..gs.k_cache.len() {
        let live = gs.visit_cap[v].min(gs.visit_rows[v]);
        let mut bytes = Vec::new();
        for planes in [&gs.k_cache[v], &gs.v_cache[v]] {
            for t in planes.iter() {
                let all = gpu.download_raw_bytes(t).expect("kv download");
                bytes.extend_from_slice(&all[..live * hd * 2]);
            }
        }
        arm.kv.push(bytes);
        arm.conv.push(gpu.download_f32(&gs.conv[v]).expect("conv download"));
    }
    arm.tags = gs.visit_pos_tags.clone();
    arm.caps = gs.visit_cap.clone();
    arm.seq_pos = gs.seq_pos;
}

fn decode(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32], arm: &mut Arm) {
    for (d, &t) in toks.iter().enumerate() {
        let l = forward_token_gpu(b, gpu, t, N_TOK + d).expect("decode");
        arm.decode.push(l);
    }
}

fn run_scalar(b: &mut Inkling2Bundle, gpu: &mut Gpu, toks: &[u32]) -> Arm {
    reset(b, gpu);
    let mut arm = Arm::default();
    let t0 = std::time::Instant::now();
    for (pos, &t) in toks.iter().enumerate() {
        let l = if PROBES.contains(&pos) {
            let (l, tr) = forward_token_gpu_trace(b, gpu, t, pos).expect("scalar trace");
            arm.visits.push(tr);
            arm.logits.push(l.clone());
            l
        } else {
            forward_token_gpu(b, gpu, t, pos).expect("scalar")
        };
        arm.argmax.push(argmax(&l));
    }
    arm.seconds = t0.elapsed().as_secs_f64();
    snapshot(b, gpu, &mut arm);
    arm
}

fn run_prefill(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    toks: &[u32],
    mode: PrefillMode,
    block: usize,
) -> Arm {
    reset(b, gpu);
    let mut arm = Arm::default();
    let mut trace = Vec::new();
    let e = b.config.n_embd;
    let mut probe_logits = Vec::new();
    let mut argmaxes = Vec::new();
    let t0 = std::time::Instant::now();
    {
        let mut sink = |pos: usize, row: &[f32]| -> Result<(), String> {
            if PROBES.contains(&pos) {
                probe_logits.push(row.to_vec());
            }
            argmaxes.push(argmax(row));
            Ok(())
        };
        forward_prefill_gpu_with(
            b,
            gpu,
            toks,
            0,
            PrefillLogits::Rows {
                from: 0,
                sink: &mut sink,
            },
            mode,
            PrefillRoutes::F16,
            block,
            Some(&mut trace),
        )
        .expect("prefill");
    }
    arm.seconds = t0.elapsed().as_secs_f64();
    arm.logits = probe_logits;
    arm.argmax = argmaxes;
    for &p in &PROBES {
        arm.visits
            .push(trace.iter().map(|v| v[p * e..(p + 1) * e].to_vec()).collect());
    }
    drop(trace);
    snapshot(b, gpu, &mut arm);
    arm
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (*x as f64 - ma, *y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa * sbb).sqrt()
}

/// KL(p_ref || p_got) over the full row, nats.
fn kl(reference: &[f32], got: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lr, lg) = (lse(reference), lse(got));
    reference
        .iter()
        .zip(got)
        .map(|(&r, &g)| {
            let lp = r as f64 - lr;
            lp.exp() * (lp - (g as f64 - lg))
        })
        .sum()
}

fn f16_bits_max_diff(a: &[u8], b: &[u8]) -> f32 {
    a.chunks_exact(2)
        .zip(b.chunks_exact(2))
        .map(|(x, y)| {
            let fx = half::f16::from_le_bytes([x[0], x[1]]).to_f32();
            let fy = half::f16::from_le_bytes([y[0], y[1]]).to_f32();
            (fx - fy).abs()
        })
        .fold(0.0f32, f32::max)
}

/// Bitwise comparison; returns the first mismatch description.
fn exact_mismatch(s: &Arm, x: &Arm) -> Option<String> {
    for (i, &p) in PROBES.iter().enumerate() {
        for v in 0..s.visits[i].len() {
            if s.visits[i][v] != x.visits[i][v] {
                return Some(format!(
                    "row {p} visit {v}: visit_out max|d| {:.3e}",
                    max_abs(&s.visits[i][v], &x.visits[i][v])
                ));
            }
        }
        if s.logits[i] != x.logits[i] {
            return Some(format!(
                "row {p}: logits max|d| {:.3e}",
                max_abs(&s.logits[i], &x.logits[i])
            ));
        }
    }
    for v in 0..s.kv.len() {
        if s.kv[v] != x.kv[v] {
            return Some(format!("visit {v}: KV bytes differ"));
        }
        if s.conv[v] != x.conv[v] {
            return Some(format!("visit {v}: conv history differs"));
        }
    }
    if s.tags != x.tags || s.caps != x.caps || s.seq_pos != x.seq_pos {
        return Some("tags / cursors / seq_pos differ".into());
    }
    for d in 0..s.decode.len() {
        if s.decode[d] != x.decode[d] {
            return Some(format!(
                "decode {d}: logits max|d| {:.3e}",
                max_abs(&s.decode[d], &x.decode[d])
            ));
        }
    }
    None
}

/// Metrics of `got` against the scalar reference; structural mismatches and
/// gate breaches are appended to `fails`.
fn report(scalar: &Arm, got: &Arm, label: &str, fails: &mut Vec<String>) {
    if got.tags != scalar.tags || got.caps != scalar.caps || got.seq_pos != scalar.seq_pos {
        fails.push(format!("{label}: tags / cursors / seq_pos differ"));
    }
    let mut worst_visit = (0.0f64, 0usize, 0usize);
    for (i, &p) in PROBES.iter().enumerate() {
        let (vr, vi) = (0..scalar.visits[i].len())
            .map(|v| (rel_rms(&got.visits[i][v], &scalar.visits[i][v]), v))
            .fold((0.0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
        if vr > worst_visit.0 {
            worst_visit = (vr, p, vi);
        }
        let (s, g) = (&scalar.logits[i], &got.logits[i]);
        let (r, k, am) = (pearson(g, s), kl(s, g), argmax(g) == argmax(s));
        eprintln!(
            "  {label} row {p:4}: worst visit {vi:2} rel-rms {vr:.3e}  logits r={r:.7} KL={k:.3e} max|d|={:.3e} argmax {}",
            max_abs(g, s),
            if am { "=" } else { "DIFF" }
        );
        if r < 0.995 {
            fails.push(format!("{label} row {p}: logits r {r}"));
        }
    }
    let agree = scalar
        .argmax
        .iter()
        .zip(&got.argmax)
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "  {label}: argmax agreement {agree}/{N_TOK}; worst probe visit rel-rms {:.3e} (row {}, visit {})",
        worst_visit.0, worst_visit.1, worst_visit.2
    );
    if agree * 100 < N_TOK * 95 {
        fails.push(format!("{label}: argmax agreement {agree}/{N_TOK}"));
    }
    let kv_worst = (0..scalar.kv.len())
        .map(|v| f16_bits_max_diff(&scalar.kv[v], &got.kv[v]))
        .fold(0.0f32, f32::max);
    let conv_worst = (0..scalar.conv.len())
        .map(|v| rel_rms(&got.conv[v], &scalar.conv[v]))
        .fold(0.0f64, f64::max);
    eprintln!("  {label}: state KV max|d| {kv_worst:.3e}, conv worst rel-rms {conv_worst:.3e}");
    let mut klsum = 0.0;
    for d in 0..DECODE {
        let (s, g) = (&scalar.decode[d], &got.decode[d]);
        let (r, k) = (pearson(g, s), kl(s, g));
        klsum += k;
        eprintln!(
            "  {label} decode {d}: r={r:.7} KL={k:.3e} argmax {}",
            if argmax(g) == argmax(s) { "=" } else { "DIFF" }
        );
        if r < 0.995 {
            fails.push(format!("{label} decode {d}: r {r}"));
        }
    }
    eprintln!("  {label}: decode mean KL {:.3e}", klsum / DECODE as f64);
}

#[test]
#[ignore = "needs the typed-XTS Inkling-2 artifact + gfx1201"]
fn blocked_prefill_oracle() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut b = load(&mut gpu);
    assert!(prefill_supported(&b, &gpu), "blocked prefill not supported on this load");
    let toks = tokens();

    let mut scalar = run_scalar(&mut b, &mut gpu, &toks);
    let cont: Vec<u32> = {
        // Fixed continuation: the scalar arm's greedy picks.
        let mut v = vec![*scalar.argmax.last().unwrap() as u32];
        let mut tmp = Arm::default();
        for d in 0..DECODE {
            let l = forward_token_gpu(&mut b, &mut gpu, v[d], N_TOK + d).expect("decode");
            v.push(argmax(&l) as u32);
            tmp.decode.push(l);
        }
        scalar.decode = tmp.decode;
        v.truncate(DECODE);
        v
    };
    eprintln!("scalar: {N_TOK} rows in {:.1}s", scalar.seconds);

    let mut exact = run_prefill(&mut b, &mut gpu, &toks, PrefillMode::ROW_EXACT, EXACT_BLOCK);
    decode(&mut b, &mut gpu, &cont, &mut exact);
    eprintln!("row-exact prefill: {:.1}s (block {EXACT_BLOCK})", exact.seconds);
    let exact_fail = exact_mismatch(&scalar, &exact);
    match &exact_fail {
        None => eprintln!("row-exact: BITWISE equal to the scalar loop (visits, logits, KV, conv, tags, decode)"),
        Some(m) => eprintln!("row-exact: MISMATCH {m}"),
    }

    let mut fails = Vec::new();
    if let Some(m) = exact_fail {
        fails.push(format!("row-exact: {m}"));
    }
    // Noise reference: the accepted decode-route change (FA2 vs the scalar
    // attention kernel, `attn_fa2 = false`) on the same prompt, same metrics.
    {
        b.gpu_state.as_mut().unwrap().attn_fa2 = false;
        let mut s0 = run_scalar(&mut b, &mut gpu, &toks);
        decode(&mut b, &mut gpu, &cont, &mut s0);
        b.gpu_state.as_mut().unwrap().attn_fa2 = true;
        eprintln!("reference: scalar loop with the scalar attention kernel vs FA2 decode");
        report(&scalar, &s0, "noise-ref", &mut Vec::new());
    }
    let only = |gemm, grouped, blocked_attn| PrefillMode {
        gemm,
        grouped,
        blocked_attn,
    };
    let arms = [
        ("attn-only", only(false, false, true), 200usize, false),
        ("gemm-only", only(true, false, false), 200, false),
        ("grouped-only", only(false, true, false), 200, false),
        ("blocked", PrefillMode::BLOCKED, 512, true),
        ("blocked", PrefillMode::BLOCKED, 200, true),
    ];
    for (name, mode, block, gate) in arms {
        let mut blk = run_prefill(&mut b, &mut gpu, &toks, mode, block);
        decode(&mut b, &mut gpu, &cont, &mut blk);
        eprintln!(
            "{name} prefill (block {block}): {:.2}s = {:.0} rows/s",
            blk.seconds,
            N_TOK as f64 / blk.seconds
        );
        let label = format!("{name}/{block}");
        let mut arm_fails = Vec::new();
        report(&scalar, &blk, &label, &mut arm_fails);
        if gate {
            fails.extend(arm_fails);
        }
    }
    if let Some(gw) = b.gpu_weights.take() {
        gw.free_gpu(&mut gpu);
    }
    if let Some(gs) = b.gpu_state.take() {
        gs.free_gpu(&mut gpu);
    }
    assert!(fails.is_empty(), "prefill oracle: {}", fails.join("; "));
    eprintln!("prefill oracle: PASS");
}
