// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Blocked-prefill oracle (`forward_prefill_gpu`) against the scalar loop
//! (`forward_token_gpu` per row) on the typed-XTS artifact, gfx1201.
//!
//! One process, three arms over the same prompt rows on one load:
//! 1. scalar: one `forward_token_gpu` per row (visit traces and logits at the
//!    probe rows), then the final state and `DECODE` continuation logits;
//! 2. row-exact prefill (`PrefillMode::RowExact`): must equal arm 1 **bit for
//!    bit**: every probe row's 48 `visit_out`, every probe row's logits, KV
//!    bytes, conv history, position tags, cursors, and the decode logits;
//! 3. blocked prefill (`PrefillMode::Blocked`, product arithmetic): the same
//!    structural state exactly (tags, cursors, `seq_pos`), numeric state and
//!    logits within an envelope relative to accepted arithmetic (below).
//!
//! Two tests, each its own process (`--exact`):
//! * `blocked_prefill_oracle` (1100 rows): row-exact at an odd block (200, so
//!   blocks end mid-sequence and the last one is partial), the decode-route
//!   noise reference, the single-site arms, and blocked/512 and blocked/200.
//! * `blocked_prefill_oracle_rungs` (2600 rows): the load's block-size rungs
//!   (`PREFILL_BLOCK_RUNGS`). Row-exact at 1024 and 2048 (bitwise, as arm 2),
//!   the grouped-only reference, then gated blocked arms at 512, 1024 and
//!   2048. The 1024 and 2048 arms prefill in two calls, `[0, SPLIT)` then
//!   `[SPLIT, 2600)`, so a block starts mid-ring with pre-block keys in the
//!   SWA cache: at 2048 the second call's first block spans `[300, 2348)`,
//!   wider than the 1024-row SWA ring (keys before the block from the cache,
//!   in-block keys from staging, only the trailing 1024 rows written back),
//!   and both rungs end on a partial tail (`[2348, 2600)`). Each wide blocked
//!   arm also reports whether it equals blocked/512 bit for bit
//!   (informational: the split moves the FA2 key tiling, so it need not). The
//!   scalar-attention noise reference stays in the first test: its VRAM grows
//!   with context and 2600 rows do not fit beside the other arms.
//!
//! **Blocked-arm envelope.** The prompt is random tokens, so a probe row can sit
//! on a near-tie that any accepted arithmetic change flips, and different
//! arithmetic flips different rows: the ungated grouped-only arm (Stage-1
//! grouped experts) alone reaches r 0.99458 / KL 1.0e-2 at row 999 (visit 45;
//! `blocked/runs/oracle3`, 2026-11-27), while rows 0..3 sit near KL 1e-7. So no
//! absolute per-row r and no per-row KL ratio is a bar. The gated arm is held
//! to the grouped-only reference arm, measured in this process on the same
//! prompt (accepted arithmetic without the dense / projection sites the gated
//! arm adds): per probe row `r >= r_ref(row) - 0.001`; pooled over the probe
//! rows, `max KL <= max(1e-4, 1.5 * max KL_ref)` and
//! `mean KL <= 1.5 * mean KL_ref` (the 1e-4 floor covers near-zero rows).
//! Absolute floors catch gross breakage regardless: finite logits and
//! r >= 0.99 at every probe row, argmax agreement >= 95%, decode r >= 0.995.
//! The rungs test gates the pooled bars and the floors; its per-row r bar is
//! printed, not gated: over 25 probes on 2600 rows it fails the accepted
//! blocked/512 arithmetic itself (row 2347: r 0.99827 vs 0.99899) while 1024
//! and 2048 flip a different row (2047: r 0.99884 vs 0.99898), with pooled KL
//! max 1.042e-2 / mean 1.078e-3 (512) and 1.042e-2 / 1.049e-3 (wide) against
//! bars 1.537e-2 / 1.172e-3; the planted fault lands at 5.2e-2 / 1.8e-2
//! (InkBlockPolicy o1, 2026-11-28).
//!
//! **Planted fault.** `INKLING2_ORACLE_PLANT=ctrl-swap` swaps the gain and forget
//! weights of every controller of the first core layer before the gated arms
//! (the scalar reference keeps the true weights); the gate must then fail on
//! every gated blocked arm.
//!
//! Probe rows span block boundaries (200, 300, 512, 1324, 2048, 2348),
//! positions 1023/1024/1025 (relative-bias extent and the first SWA ring wrap
//! at 1024 rows) and the partial tail.
//!
//! ```sh
//! INKLING2_MQ4=.../inkling2-mini-xts-c.hfq HIPFIRE_GRAPH=0 \
//!   cargo test --release -p hipfire-arch-inkling2 --test prefill_parity -- \
//!   --ignored --nocapture --exact blocked_prefill_oracle   # or blocked_prefill_oracle_rungs
//! ```

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::gpu::upload;
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{
    forward_prefill_gpu_with, forward_token_gpu, forward_token_gpu_trace, prefill_chunk_rows,
    prefill_supported, PrefillLogits, PrefillMode, PrefillRoutes, PREFILL_BLOCK_RUNGS,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

const DECODE: usize = 6;
const EXACT_BLOCK: usize = 200;
/// First call's rows for the two-call (1024 / 2048 rung) arms.
const SPLIT: usize = 300;

/// One oracle's prompt length, KV rows and probe rows (ascending).
struct Case {
    n_tok: usize,
    max_seq: usize,
    probes: &'static [usize],
}

const ORIGINAL: Case = Case {
    n_tok: 1100,
    max_seq: 1152,
    probes: &[
        0, 1, 2, 3, 199, 200, 201, 511, 512, 999, 1000, 1022, 1023, 1024, 1025, 1099,
    ],
};

const RUNGS: Case = Case {
    n_tok: 2600,
    max_seq: 2688,
    probes: &[
        0, 1, 2, 3, 299, 300, 301, 511, 512, 999, 1000, 1022, 1023, 1024, 1025, 1323, 1324, 1325,
        2047, 2048, 2049, 2347, 2348, 2349, 2599,
    ],
};

fn tokens(c: &Case) -> Vec<u32> {
    // Deterministic ids inside the vocabulary (LCG); content is irrelevant to parity.
    let mut s = 0x1234_5678_9abc_def0u64;
    (0..c.n_tok)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 199_000) as u32 + 100
        })
        .collect()
}

fn load(gpu: &mut Gpu, c: &Case) -> Inkling2Bundle {
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "the oracle requires HIPFIRE_GRAPH=0"
    );
    let art = std::env::var("INKLING2_MQ4").expect("set INKLING2_MQ4 to the typed-XTS .hfq");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("weights");
    let (gw, gs) = upload(&weights, &cfg, gpu, c.max_seq).expect("upload");
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
        arm.conv
            .push(gpu.download_f32(&gs.conv[v]).expect("conv download"));
    }
    arm.tags = gs.visit_pos_tags.clone();
    arm.caps = gs.visit_cap.clone();
    arm.seq_pos = gs.seq_pos;
}

fn decode(b: &mut Inkling2Bundle, gpu: &mut Gpu, c: &Case, toks: &[u32], arm: &mut Arm) {
    for (d, &t) in toks.iter().enumerate() {
        let l = forward_token_gpu(b, gpu, t, c.n_tok + d).expect("decode");
        arm.decode.push(l);
    }
}

fn run_scalar(b: &mut Inkling2Bundle, gpu: &mut Gpu, c: &Case, toks: &[u32]) -> Arm {
    reset(b, gpu);
    let mut arm = Arm::default();
    let t0 = std::time::Instant::now();
    for (pos, &t) in toks.iter().enumerate() {
        let l = if c.probes.contains(&pos) {
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

/// Prefill `toks` in `block`-row blocks: one call, or with `split > 0` two
/// calls `[0, split)` and `[split, len)` (the second call's blocks start
/// mid-ring).
fn run_prefill(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    c: &Case,
    toks: &[u32],
    mode: PrefillMode,
    block: usize,
    split: usize,
) -> Arm {
    reset(b, gpu);
    let mut arm = Arm::default();
    let mut trace = Vec::new();
    let e = b.config.n_embd;
    let mut probe_logits = Vec::new();
    let mut argmaxes = Vec::new();
    let calls: Vec<(usize, usize)> = if split == 0 {
        vec![(0, toks.len())]
    } else {
        vec![(0, split), (split, toks.len())]
    };
    let t0 = std::time::Instant::now();
    for (start, end) in calls {
        {
            let mut sink = |pos: usize, row: &[f32]| -> Result<(), String> {
                if c.probes.contains(&pos) {
                    probe_logits.push(row.to_vec());
                }
                argmaxes.push(argmax(row));
                Ok(())
            };
            forward_prefill_gpu_with(
                b,
                gpu,
                &toks[start..end],
                start,
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
        // The trace holds this call's rows only.
        for &p in c.probes.iter().filter(|&&p| (start..end).contains(&p)) {
            let r = p - start;
            arm.visits.push(
                trace
                    .iter()
                    .map(|v| v[r * e..(r + 1) * e].to_vec())
                    .collect(),
            );
        }
    }
    arm.seconds = t0.elapsed().as_secs_f64();
    arm.logits = probe_logits;
    arm.argmax = argmaxes;
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
fn exact_mismatch(c: &Case, s: &Arm, x: &Arm) -> Option<String> {
    for (i, &p) in c.probes.iter().enumerate() {
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

/// Prints the arm's metrics, pushes structural and absolute-floor breaches to
/// `fails`, and returns `(r, KL)` per probe row for the envelope gate.
fn report(
    c: &Case,
    scalar: &Arm,
    got: &Arm,
    label: &str,
    fails: &mut Vec<String>,
) -> Vec<(f64, f64)> {
    let n_tok = c.n_tok;
    if got.tags != scalar.tags || got.caps != scalar.caps || got.seq_pos != scalar.seq_pos {
        fails.push(format!("{label}: tags / cursors / seq_pos differ"));
    }
    let mut rows = Vec::with_capacity(c.probes.len());
    let mut worst_visit = (0.0f64, 0usize, 0usize);
    for (i, &p) in c.probes.iter().enumerate() {
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
        if !g.iter().all(|v| v.is_finite()) || r.is_nan() || r < 0.99 {
            fails.push(format!(
                "{label} row {p}: logits r {r} (floor 0.99, finite)"
            ));
        }
        rows.push((r, k));
    }
    let agree = scalar
        .argmax
        .iter()
        .zip(&got.argmax)
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "  {label}: argmax agreement {agree}/{n_tok}; worst probe visit rel-rms {:.3e} (row {}, visit {})",
        worst_visit.0, worst_visit.1, worst_visit.2
    );
    if agree * 100 < n_tok * 95 {
        fails.push(format!("{label}: argmax agreement {agree}/{n_tok}"));
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
    rows
}

/// The envelope gate of a blocked arm against the grouped-only reference
/// (`rows` and `grouped` are [`report`]'s per-probe `(r, KL)`): the pooled KL
/// bars, and with `per_row` the per-probe `r >= r_ref - 0.001` bar (otherwise
/// its breaches are printed, not gated).
fn envelope(
    c: &Case,
    label: &str,
    rows: &[(f64, f64)],
    grouped: &[(f64, f64)],
    per_row: bool,
    fails: &mut Vec<String>,
) {
    assert_eq!(
        grouped.len(),
        c.probes.len(),
        "grouped-only runs before the gated arms"
    );
    for (i, &p) in c.probes.iter().enumerate() {
        let (r, r_min) = (rows[i].0, grouped[i].0 - 0.001);
        if !(r >= r_min) {
            let m = format!("{label} row {p}: r {r:.7} < {r_min:.7}");
            if per_row {
                fails.push(m);
            } else {
                eprintln!("  (per-row, not gated) {m}");
            }
        }
    }
    let max_kl = |v: &[(f64, f64)]| v.iter().map(|x| x.1).fold(0.0f64, f64::max);
    let mean_kl = |v: &[(f64, f64)]| v.iter().map(|x| x.1).sum::<f64>() / v.len() as f64;
    let (kmax, kmax_bar) = (max_kl(rows), 1e-4f64.max(1.5 * max_kl(grouped)));
    let (kmean, kmean_bar) = (mean_kl(rows), 1.5 * mean_kl(grouped));
    eprintln!(
        "  {label}: pooled KL max {kmax:.3e} (bar {kmax_bar:.3e}), mean {kmean:.3e} (bar {kmean_bar:.3e})"
    );
    if !(kmax <= kmax_bar && kmean <= kmean_bar) {
        fails.push(format!(
            "{label}: pooled KL max {kmax:.3e} / mean {kmean:.3e} over bars {kmax_bar:.3e} / {kmean_bar:.3e}"
        ));
    }
}

/// Swaps gain and forget on every controller of the first core layer when
/// `INKLING2_ORACLE_PLANT=ctrl-swap` (once; it stays for every later arm).
fn plant(b: &mut Inkling2Bundle, armed: &mut Option<String>) {
    if armed.take().as_deref() != Some("ctrl-swap") {
        return;
    }
    let gw = b.gpu_weights.as_mut().unwrap();
    if let Some(cs) = gw.layers.iter_mut().find_map(|l| l.controllers.as_mut()) {
        eprintln!("PLANTED: gain/forget swapped on every controller of the first core layer");
        for ctrl in cs.iter_mut() {
            std::mem::swap(&mut ctrl.gain, &mut ctrl.forget);
        }
    }
}

/// Load, the scalar arm and its fixed greedy continuation.
fn setup(c: &Case) -> (Gpu, Inkling2Bundle, Vec<u32>, Arm, Vec<u32>) {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut b = load(&mut gpu, c);
    assert!(
        prefill_supported(&b, &gpu),
        "blocked prefill not supported on this load"
    );
    let toks = tokens(c);
    // The load's rung (VRAM-budgeted), then the scratch for the widest rung.
    let load_rows = prefill_chunk_rows(&b, &gpu);
    let widest = PREFILL_BLOCK_RUNGS[0];
    let cfg = b.config.clone();
    b.gpu_state
        .as_mut()
        .unwrap()
        .resize_prefill_scratch(&mut gpu, &cfg, widest)
        .expect("prefill scratch at the widest rung");
    eprintln!("load rung: {load_rows}-row blocks; oracle scratch: {widest} rows");

    let mut scalar = run_scalar(&mut b, &mut gpu, c, &toks);
    let cont: Vec<u32> = {
        // Fixed continuation: the scalar arm's greedy picks.
        let mut v = vec![*scalar.argmax.last().unwrap() as u32];
        let mut tmp = Arm::default();
        for d in 0..DECODE {
            let l = forward_token_gpu(&mut b, &mut gpu, v[d], c.n_tok + d).expect("decode");
            v.push(argmax(&l) as u32);
            tmp.decode.push(l);
        }
        scalar.decode = tmp.decode;
        v.truncate(DECODE);
        v
    };
    eprintln!("scalar: {} rows in {:.1}s", c.n_tok, scalar.seconds);
    (gpu, b, toks, scalar, cont)
}

/// Row-exact prefill at `block` (two calls when `split > 0`): bitwise the scalar loop.
#[allow(clippy::too_many_arguments)]
fn row_exact(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    c: &Case,
    toks: &[u32],
    cont: &[u32],
    scalar: &Arm,
    block: usize,
    split: usize,
    fails: &mut Vec<String>,
) {
    let mut exact = run_prefill(b, gpu, c, toks, PrefillMode::ROW_EXACT, block, split);
    decode(b, gpu, c, cont, &mut exact);
    eprintln!(
        "row-exact prefill: {:.1}s (block {block}, split {split})",
        exact.seconds
    );
    match exact_mismatch(c, scalar, &exact) {
        None => eprintln!(
            "row-exact/{block}: BITWISE equal to the scalar loop (visits, logits, KV, conv, tags, decode)"
        ),
        Some(m) => {
            eprintln!("row-exact/{block}: MISMATCH {m}");
            fails.push(format!("row-exact/{block}: {m}"));
        }
    }
}

fn only(gemm: bool, grouped: bool, blocked_attn: bool) -> PrefillMode {
    PrefillMode {
        gemm,
        grouped,
        blocked_attn,
    }
}

/// One prefill arm, decoded and reported: `(arm, per-probe (r, KL), floor breaches)`.
fn arm(
    b: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    c: &Case,
    toks: &[u32],
    cont: &[u32],
    scalar: &Arm,
    (name, mode, block, split): (&str, PrefillMode, usize, usize),
) -> (Arm, Vec<(f64, f64)>, Vec<String>) {
    let mut blk = run_prefill(b, gpu, c, toks, mode, block, split);
    decode(b, gpu, c, cont, &mut blk);
    eprintln!(
        "{name} prefill (block {block}, split {split}): {:.2}s = {:.0} rows/s",
        blk.seconds,
        c.n_tok as f64 / blk.seconds
    );
    let mut fails = Vec::new();
    let rows = report(c, scalar, &blk, &format!("{name}/{block}"), &mut fails);
    (blk, rows, fails)
}

fn finish(mut gpu: Gpu, mut b: Inkling2Bundle, fails: Vec<String>) {
    if let Some(gw) = b.gpu_weights.take() {
        gw.free_gpu(&mut gpu);
    }
    if let Some(gs) = b.gpu_state.take() {
        gs.free_gpu(&mut gpu);
    }
    assert!(fails.is_empty(), "prefill oracle: {}", fails.join("; "));
    eprintln!("prefill oracle: PASS");
}

#[test]
#[ignore = "needs the typed-XTS Inkling-2 artifact + gfx1201"]
fn blocked_prefill_oracle() {
    let c = &ORIGINAL;
    let (mut gpu, mut b, toks, scalar, cont) = setup(c);
    let mut fails = Vec::new();
    row_exact(
        &mut b,
        &mut gpu,
        c,
        &toks,
        &cont,
        &scalar,
        EXACT_BLOCK,
        0,
        &mut fails,
    );
    // Noise reference: the accepted decode-route change (FA2 vs the scalar
    // attention kernel, `attn_fa2 = false`) on the same prompt, same metrics.
    // (printed for scale; the envelope reference is the grouped-only arm)
    {
        b.gpu_state.as_mut().unwrap().attn_fa2 = false;
        let mut s0 = run_scalar(&mut b, &mut gpu, c, &toks);
        decode(&mut b, &mut gpu, c, &cont, &mut s0);
        b.gpu_state.as_mut().unwrap().attn_fa2 = true;
        eprintln!("reference: scalar loop with the scalar attention kernel vs FA2 decode");
        report(c, &scalar, &s0, "noise-ref", &mut Vec::new());
    }
    let arms = [
        ("attn-only", only(false, false, true), 200, false),
        ("gemm-only", only(true, false, false), 200, false),
        ("grouped-only", only(false, true, false), 200, false),
        ("blocked", PrefillMode::BLOCKED, 512, true),
        ("blocked", PrefillMode::BLOCKED, 200, true),
    ];
    let mut grouped: Vec<(f64, f64)> = Vec::new();
    let mut planted = std::env::var("INKLING2_ORACLE_PLANT").ok();
    for (name, mode, block, gate) in arms {
        if gate {
            plant(&mut b, &mut planted);
        }
        let (_, rows, arm_fails) = arm(
            &mut b,
            &mut gpu,
            c,
            &toks,
            &cont,
            &scalar,
            (name, mode, block, 0),
        );
        if name == "grouped-only" {
            grouped = rows.clone();
        }
        if gate {
            fails.extend(arm_fails);
            envelope(
                c,
                &format!("{name}/{block}"),
                &rows,
                &grouped,
                true,
                &mut fails,
            );
        }
    }
    finish(gpu, b, fails);
}

#[test]
#[ignore = "needs the typed-XTS Inkling-2 artifact + gfx1201"]
fn blocked_prefill_oracle_rungs() {
    let c = &RUNGS;
    let (mut gpu, mut b, toks, scalar, cont) = setup(c);
    let mut fails = Vec::new();
    for block in [1024, 2048] {
        row_exact(
            &mut b, &mut gpu, c, &toks, &cont, &scalar, block, SPLIT, &mut fails,
        );
    }
    let (_, grouped, _) = arm(
        &mut b,
        &mut gpu,
        c,
        &toks,
        &cont,
        &scalar,
        ("grouped-only", only(false, true, false), 200, 0),
    );
    let mut planted = std::env::var("INKLING2_ORACLE_PLANT").ok();
    plant(&mut b, &mut planted);
    // blocked/512 first: the wide rungs are compared with it bit for bit.
    let mut blocked512: Option<Arm> = None;
    for (block, split) in [(512, 0), (1024, SPLIT), (2048, SPLIT)] {
        let label = format!("blocked/{block}");
        let (blk, rows, arm_fails) = arm(
            &mut b,
            &mut gpu,
            c,
            &toks,
            &cont,
            &scalar,
            ("blocked", PrefillMode::BLOCKED, block, split),
        );
        fails.extend(arm_fails);
        envelope(c, &label, &rows, &grouped, false, &mut fails);
        match blocked512.as_ref() {
            None => blocked512 = Some(blk),
            Some(b512) => match exact_mismatch(c, b512, &blk) {
                None => eprintln!("  {label}: BITWISE equal to blocked/512"),
                Some(m) => eprintln!("  {label}: differs from blocked/512: {m}"),
            },
        }
    }
    finish(gpu, b, fails);
}
