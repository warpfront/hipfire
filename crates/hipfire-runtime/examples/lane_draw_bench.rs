// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GPU oracle + bench of `hipfire_runtime::lane_draw::lane_draws` (the
// lane-wide sampled verify, `kernels/src/lane_leviathan.hip`).
//
// 1. p tables: every row's device p (nucleus tokens, unnormalized probs, mass)
//    against a host model of the shared device sampler's law (blocked -> -inf,
//    recency-weighted repeat / presence / frequency penalty over the row scope,
//    top-G by (value desc, index asc), softmax, cap, min-p, nucleus). Tokens and
//    lengths exact, probs to a few ulp (device expf).
// 2. Verdicts: the host replica of the lane verdict on the DEVICE p tables
//    (same f32 ops) must give the same tokens / accepted / rng bit for bit.
// 3. Distribution: 1-draft lanes with drafts drawn from q (and point-mass q,
//    the copy-lane case); the first emitted token's histogram must be p
//    (chi-square), the acceptance rate sum min(p, q); an AR control.
// 4. Timing: 8 lanes x 6 rows of vocab 248320 (48 rows): one `lane_draws`
//    call (H2D + kernel + D2H + sync), GPU events and wall clock.
//
// Run on one gfx1201 card through the fabric:
//   cargo build --release -p hipfire-runtime --features lab --example lane_draw_bench

use hipfire_runtime::lane_draw::{
    lane_draws, reference_p, reference_verdict, LaneDraw, LaneDrawArena, LaneDrawJob, ReferenceP,
};
use hipfire_runtime::sampler::SamplerConfig;
use rdna_compute::lane_leviathan::LLV_P_WORDS;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const VOCAB: usize = 248_320;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) * (1.0 / 16_777_216.0)
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone)]
struct Cfg {
    temp: f32,
    top_p: f32,
    top_k_req: i32,
    min_p: f32,
    rep: f32,
    pres: f32,
    freq: f32,
    window: usize,
    blocked: Vec<u32>,
}

impl Cfg {
    fn plain(temp: f32, top_p: f32, top_k_req: i32) -> Self {
        Cfg {
            temp,
            top_p,
            top_k_req,
            min_p: 0.0,
            rep: 1.0,
            pres: 0.0,
            freq: 0.0,
            window: 0,
            blocked: Vec::new(),
        }
    }
    fn penalty_active(&self) -> bool {
        self.rep > 1.0 || self.pres > 0.0 || self.freq > 0.0
    }
}

#[derive(Clone)]
struct Lane {
    /// Global row of row 0 inside the logits tensor (rows are `vocab` floats).
    logits_row0: usize,
    rows: usize,
    vocab: usize,
    drafts: Vec<u32>,
    q: Vec<Vec<(u32, f32)>>,
    history: Vec<u32>,
    rng: u32,
    eos: Option<u32>,
    cfg: Cfg,
}

impl Lane {
    /// Row j's penalty scope: trailing `window` of history ++ drafts[..j].
    fn scope(&self, j: usize) -> Vec<u32> {
        if !self.cfg.penalty_active() || self.cfg.window == 0 {
            return Vec::new();
        }
        let mut all = self.history.clone();
        all.extend_from_slice(&self.drafts[..j]);
        let start = all.len().saturating_sub(self.cfg.window);
        all[start..].to_vec()
    }
}

fn host_p(row: &[f32], scope: &[u32], cfg: &Cfg) -> ReferenceP {
    reference_p(row, scope, &sampler_cfg(cfg))
}

fn dev_ptab(words: &[f32], row: usize) -> ReferenceP {
    let p = &words[row * LLV_P_WORDS..(row + 1) * LLV_P_WORDS];
    let n = p[0].to_bits() as usize;
    let point = (p[2].to_bits() != 0).then(|| p[3].to_bits());
    ReferenceP {
        point,
        entries: (0..n)
            .map(|i| (p[4 + i].to_bits(), p[4 + 64 + i]))
            .collect(),
        mass: p[1],
    }
}

// ---------------- harness ----------------

fn sampler_cfg(c: &Cfg) -> SamplerConfig {
    SamplerConfig {
        temperature: c.temp,
        top_p: c.top_p,
        repeat_penalty: c.rep,
        repeat_window: c.window,
        presence_penalty: c.pres,
        frequency_penalty: c.freq,
        blocked_tokens: c.blocked.clone(),
        top_k: Some(c.top_k_req as u32),
        min_p: (c.min_p > 0.0).then_some(c.min_p),
    }
}

/// One `lane_draws` call over `lanes`, plus the call's p tables.
fn run(
    gpu: &mut Gpu,
    arena: &mut LaneDrawArena,
    lanes: &[Lane],
    logits: &GpuTensor,
) -> (Vec<LaneDraw>, Vec<f32>) {
    let cfgs: Vec<SamplerConfig> = lanes.iter().map(|l| sampler_cfg(&l.cfg)).collect();
    let views: Vec<GpuTensor> = lanes
        .iter()
        .map(|l| logits.sub_offset(l.logits_row0 * l.vocab, l.rows * l.vocab))
        .collect();
    let jobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .enumerate()
        .map(|(i, l)| LaneDrawJob {
            logits: &views[i],
            rows: l.rows,
            vocab: l.vocab,
            drafts: &l.drafts,
            q: &l.q,
            history: &l.history,
            cfg: &cfgs[i],
            rng_state: l.rng,
            eos: l.eos,
        })
        .collect();
    let draws = lane_draws(gpu, arena, &jobs).expect("lane_draws");
    let rows: usize = lanes.iter().map(|l| l.rows).sum();
    let mut pt = gpu
        .download_f32(arena.ptab().expect("p tables"))
        .expect("p tables");
    pt.truncate(rows * LLV_P_WORDS);
    (draws, pt)
}

fn make_logits(rng: &mut Rng, rows: usize, vocab: usize, peaked: bool) -> Vec<f32> {
    let mut v = vec![0f32; rows * vocab];
    for r in 0..rows {
        let row = &mut v[r * vocab..(r + 1) * vocab];
        for x in row.iter_mut() {
            *x = rng.normal() * 2.0;
        }
        if peaked {
            for k in 0..12 {
                let t = rng.below(vocab);
                row[t] = 9.0 + rng.normal() - k as f32 * 0.4;
            }
        }
    }
    v
}

/// Draft q from `(token, logit)` candidates: softmax over them (as drawn).
fn q_from(rng: &mut Rng, row: &[f32], n: usize, temp: f32) -> Vec<(u32, f32)> {
    // Candidates: the row's top-n with a little noise (the head's ranking).
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    let c: Vec<(u32, f32)> = idx[..n]
        .iter()
        .map(|&t| (t, row[t as usize] + 0.3 * rng.normal()))
        .collect();
    let m = c.iter().map(|e| e.1).fold(f32::NEG_INFINITY, f32::max);
    let mut q: Vec<(u32, f32)> = c
        .iter()
        .map(|&(t, l)| (t, ((l - m) / temp).exp()))
        .collect();
    let s: f32 = q.iter().map(|e| e.1).sum();
    for e in &mut q {
        e.1 /= s;
    }
    q
}

fn sample_q(rng: &mut Rng, q: &[(u32, f32)]) -> u32 {
    let u = rng.unit();
    let mut acc = 0.0;
    for &(t, p) in q {
        acc += p;
        if u < acc {
            return t;
        }
    }
    q.last().unwrap().0
}

fn check_case(
    gpu: &mut Gpu,
    arena: &mut LaneDrawArena,
    name: &str,
    lanes: &[Lane],
    host_logits: &[f32],
    logits: &GpuTensor,
    fails: &mut usize,
) {
    let (draws, pt) = run(gpu, arena, lanes, logits);
    let mut grow = 0usize;
    let mut bad = 0usize;
    for (li, lane) in lanes.iter().enumerate() {
        let mut ps = Vec::new();
        for j in 0..lane.rows {
            let r0 = (lane.logits_row0 + j) * lane.vocab;
            let hp = host_p(&host_logits[r0..r0 + lane.vocab], &lane.scope(j), &lane.cfg);
            let dp = dev_ptab(&pt, grow + j);
            let same =
                hp.point == dp.point
                    && hp.entries.len() == dp.entries.len()
                    && hp.entries.iter().zip(&dp.entries).all(|(a, b)| {
                        a.0 == b.0 && (a.1 - b.1).abs() <= 4e-6 * a.1.abs().max(1e-30)
                    })
                    && (hp.mass - dp.mass).abs() <= 1e-5 * hp.mass.abs().max(1e-30);
            if !same {
                bad += 1;
                if bad <= 3 {
                    eprintln!(
                        "[{name}] lane {li} row {j}: p mismatch host n={} point={:?} mass={} | dev n={} point={:?} mass={}\n  host {:?}\n  dev  {:?}",
                        hp.entries.len(), hp.point, hp.mass, dp.entries.len(), dp.point, dp.mass,
                        &hp.entries[..hp.entries.len().min(6)], &dp.entries[..dp.entries.len().min(6)]
                    );
                }
            }
            ps.push(dp);
        }
        let want = reference_verdict(&ps, &lane.drafts, &lane.q, lane.rng, lane.eos);
        let d = &draws[li];
        if *d != want {
            bad += 1;
            if bad <= 6 {
                eprintln!("[{name}] lane {li}: verdict dev {d:?} | host {want:?}");
            }
        }
        grow += lane.rows;
    }
    if bad == 0 {
        println!("PASS {name} ({} lanes, {} rows)", lanes.len(), grow);
    } else {
        println!("FAIL {name}: {bad} mismatches");
        *fails += 1;
    }
}

fn lanes_for(
    rng: &mut Rng,
    host: &[f32],
    n_lanes: usize,
    rows: usize,
    vocab: usize,
    cfg: &Cfg,
    draft_from_q: bool,
) -> Vec<Lane> {
    (0..n_lanes)
        .map(|li| {
            let row0 = li * rows;
            let mut drafts = Vec::new();
            let mut qs = Vec::new();
            for j in 0..rows - 1 {
                let r = &host[(row0 + j) * vocab..(row0 + j + 1) * vocab];
                let q = q_from(rng, r, 8, cfg.temp.max(0.05));
                let d = if draft_from_q {
                    sample_q(rng, &q)
                } else {
                    // A deterministic draft (argmax): point-mass q.
                    let d = q[0].0;
                    qs.push(vec![(d, 1.0)]);
                    drafts.push(d);
                    continue;
                };
                drafts.push(d);
                qs.push(q);
            }
            let history: Vec<u32> = (0..40).map(|_| rng.below(vocab) as u32).collect();
            Lane {
                logits_row0: row0,
                rows,
                vocab,
                eos: if li % 3 == 0 {
                    drafts.first().copied()
                } else {
                    None
                },
                drafts,
                q: qs,
                history,
                rng: (rng.next() as u32) | 1,
                cfg: cfg.clone(),
            }
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!("arch {}", gpu.arch);
    assert!(gpu.lane_leviathan_supported(), "needs gfx12");
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut arena = LaneDrawArena::new();
    let mut fails = 0usize;

    // ---- 1+2: p tables and verdicts across sampler settings (vocab 248320, 8x6). ----
    let host = make_logits(&mut rng, 48, VOCAB, true);
    let logits = gpu.upload_f32(&host, &[48 * VOCAB]).unwrap();
    let mut pen = Cfg::plain(1.0, 0.95, 20);
    pen.rep = 1.3;
    pen.pres = 0.4;
    pen.freq = 0.2;
    pen.window = 64;
    let mut blk = Cfg::plain(0.8, 0.9, 40);
    blk.blocked = vec![1, 2, 3, 77, VOCAB as u32 + 5];
    let mut minp = Cfg::plain(1.0, 1.0, 64);
    minp.min_p = 0.05;
    let cases: Vec<(&str, Cfg)> = vec![
        ("T1 p.95 k20", Cfg::plain(1.0, 0.95, 20)),
        ("T.6 p.95 k20", Cfg::plain(0.6, 0.95, 20)),
        ("T1 p1 k64", Cfg::plain(1.0, 1.0, 64)),
        ("T1 p.8 k5", Cfg::plain(1.0, 0.8, 5)),
        ("T1 k0", Cfg::plain(1.0, 0.95, 0)),
        ("penalties w64", pen.clone()),
        ("blocked k40", blk),
        ("min_p", minp),
        ("greedy", Cfg::plain(0.0, 1.0, 20)),
    ];
    for (name, cfg) in &cases {
        for draft_from_q in [true, false] {
            let lanes = lanes_for(&mut rng, &host, 8, 6, VOCAB, cfg, draft_from_q);
            let tag = format!(
                "{name} {}",
                if draft_from_q { "q-drafts" } else { "argmax" }
            );
            check_case(
                &mut gpu, &mut arena, &tag, &lanes, &host, &logits, &mut fails,
            );
        }
    }
    // Drafts equal to the bonus-row winners: force long accepts through p-greedy drafts.
    {
        let cfg = Cfg::plain(0.3, 0.95, 20);
        let mut lanes = lanes_for(&mut rng, &host, 8, 6, VOCAB, &cfg, false);
        for l in &mut lanes {
            for j in 0..l.rows - 1 {
                let r = (l.logits_row0 + j) * VOCAB;
                let hp = host_p(&host[r..r + VOCAB], &l.scope(j), &l.cfg);
                l.drafts[j] = hp.entries[0].0;
                l.q[j] = vec![(hp.entries[0].0, 1.0)];
            }
        }
        check_case(
            &mut gpu,
            &mut arena,
            "long accepts",
            &lanes,
            &host,
            &logits,
            &mut fails,
        );
    }
    // AR draws (one row, no draft) and mixed lane widths in one launch.
    {
        let cfg = Cfg::plain(1.0, 0.95, 20);
        let mut lanes = Vec::new();
        let mut row0 = 0;
        for (i, rows) in [1usize, 6, 1, 3, 9, 2, 1, 6].iter().enumerate() {
            let mut l = lanes_for(
                &mut rng,
                &host[row0 * VOCAB..],
                1,
                *rows,
                VOCAB,
                &cfg,
                i % 2 == 0,
            )
            .pop()
            .unwrap();
            l.logits_row0 = row0;
            row0 += rows;
            lanes.push(l);
        }
        check_case(
            &mut gpu,
            &mut arena,
            "AR + mixed widths",
            &lanes,
            &host,
            &logits,
            &mut fails,
        );
    }
    gpu.free_tensor(logits).unwrap();

    // Exact ties: an all-equal row (tile fallback), planted ties at the cap boundary.
    {
        let v = 4096usize;
        let mut h = vec![0f32; 4 * v];
        for x in &mut h[v..2 * v] {
            *x = (rng.below(4)) as f32;
        }
        for x in &mut h[2 * v..4 * v] {
            *x = rng.normal();
        }
        for t in 0..30 {
            h[2 * v + t * 7] = 5.0;
        }
        let lg = gpu.upload_f32(&h, &[4 * v]).unwrap();
        for (name, cfg) in [
            ("ties k20", Cfg::plain(1.0, 0.95, 20)),
            ("ties k64", Cfg::plain(1.0, 1.0, 64)),
        ] {
            let lanes = lanes_for(&mut rng, &h, 1, 4, v, &cfg, true);
            check_case(&mut gpu, &mut arena, name, &lanes, &h, &lg, &mut fails);
        }
        // Big vocab all-equal row: tile fallback at size.
        gpu.free_tensor(lg).unwrap();
        let hz = vec![0.5f32; 2 * VOCAB];
        let lg = gpu.upload_f32(&hz, &[2 * VOCAB]).unwrap();
        let lanes = lanes_for(&mut rng, &hz, 1, 2, VOCAB, &Cfg::plain(1.0, 0.95, 20), true);
        let t0 = Instant::now();
        check_case(
            &mut gpu,
            &mut arena,
            "all-equal 248320 (tile fallback)",
            &lanes,
            &hz,
            &lg,
            &mut fails,
        );
        println!(
            "  all-equal case wall {:.1} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
        gpu.free_tensor(lg).unwrap();
    }

    // ---- 3: distribution (1-draft lanes: draft ~ q, a fixed copy draft with
    // q = its point mass, and an AR control). ----
    {
        let v = 2048usize;
        let mut h = make_logits(&mut rng, 2, v, true);
        for x in &mut h {
            *x *= 0.6;
        }
        let lg = gpu.upload_f32(&h, &[2 * v]).unwrap();
        let mut cfgp = Cfg::plain(1.0, 0.95, 20);
        cfgp.rep = 1.2;
        cfgp.window = 16;
        // mode 0: draft ~ q; 1: fixed copy draft, q = delta(draft); 2: AR.
        for (name, cfg, mode) in [
            ("dist T1 k20", Cfg::plain(1.0, 0.95, 20), 0u8),
            ("dist T.6 pen", cfgp, 0),
            ("dist copy point-mass T1 k20", Cfg::plain(1.0, 0.95, 20), 1),
            ("dist AR T1 k20", Cfg::plain(1.0, 0.95, 20), 2),
        ] {
            let n_lanes = 8192usize;
            let q0 = {
                let mut r2 = Rng(77);
                let mut q = q_from(&mut r2, &h[..v], 8, cfg.temp);
                // Mis-calibrated q: flatten it so rejections are frequent.
                for e in &mut q {
                    e.1 = e.1.sqrt();
                }
                let s: f32 = q.iter().map(|e| e.1).sum();
                for e in &mut q {
                    e.1 /= s;
                }
                q
            };
            let history: Vec<u32> = (0..10).map(|i| (i * 13) as u32).collect();
            let probe = Lane {
                logits_row0: 0,
                rows: 2,
                vocab: v,
                drafts: vec![0],
                q: vec![q0.clone()],
                history: history.clone(),
                rng: 1,
                eos: None,
                cfg: cfg.clone(),
            };
            let p0 = host_p(&h[..v], &probe.scope(0), &cfg);
            // The copy draft: p's second most likely token.
            let copy = p0.entries[1].0;
            let mut counts = std::collections::HashMap::<u32, u64>::new();
            let mut accepts = 0u64;
            let mut total = 0u64;
            for _ in 0..32 {
                let lanes: Vec<Lane> = (0..n_lanes)
                    .map(|_| {
                        let d = sample_q(&mut rng, &q0);
                        let rng_state = (rng.next() as u32) | 1;
                        match mode {
                            2 => Lane {
                                rows: 1,
                                drafts: vec![],
                                q: vec![],
                                rng: rng_state,
                                ..probe.clone()
                            },
                            1 => Lane {
                                drafts: vec![copy],
                                q: vec![vec![(copy, 1.0)]],
                                rng: rng_state,
                                ..probe.clone()
                            },
                            _ => Lane {
                                drafts: vec![d],
                                rng: rng_state,
                                ..probe.clone()
                            },
                        }
                    })
                    .collect();
                let (draws, _) = run(&mut gpu, &mut arena, &lanes, &lg);
                for d in &draws {
                    *counts.entry(d.tokens[0]).or_default() += 1;
                    accepts += d.accepted as u64;
                    total += 1;
                }
            }
            let mut chi = 0.0f64;
            let mut dof = 0usize;
            let mut outside = 0u64;
            for (&t, &c) in &counts {
                if !p0.entries.iter().any(|e| e.0 == t) {
                    outside += c;
                }
            }
            for &(t, x) in &p0.entries {
                let e = (x / p0.mass) as f64 * total as f64;
                let o = *counts.get(&t).unwrap_or(&0) as f64;
                if e >= 5.0 {
                    chi += (o - e) * (o - e) / e;
                    dof += 1;
                }
            }
            let dof = dof.saturating_sub(1).max(1);
            // Wilson-Hilferty z of chi-square.
            let k = dof as f64;
            let z =
                ((chi / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
            let expect_acc: f64 = match mode {
                2 => 0.0,
                1 => p0
                    .entries
                    .iter()
                    .find(|e| e.0 == copy)
                    .map_or(0.0, |e| e.1 / p0.mass) as f64,
                _ => q0
                    .iter()
                    .map(|&(t, qx)| {
                        let px = p0
                            .entries
                            .iter()
                            .find(|e| e.0 == t)
                            .map_or(0.0, |e| e.1 / p0.mass);
                        (px.min(qx)) as f64
                    })
                    .sum(),
            };
            let acc_rate = accepts as f64 / total as f64;
            let sd = (expect_acc * (1.0 - expect_acc) / total as f64).sqrt();
            let ok = z < 3.5 && outside == 0 && (acc_rate - expect_acc).abs() <= 4.0 * sd;
            println!(
                "{} {name}: n={total} chi2={chi:.1} dof={dof} z={z:.2} outside={outside} accept={acc_rate:.4} expected={expect_acc:.4} (+-{:.4})",
                if ok { "PASS" } else { "FAIL" },
                4.0 * sd
            );
            if !ok {
                fails += 1;
            }
        }
        gpu.free_tensor(lg).unwrap();
    }

    // ---- 4: timing, 8 lanes x 6 rows, vocab 248320. ----
    {
        let host = make_logits(&mut rng, 48, VOCAB, true);
        let logits = gpu.upload_f32(&host, &[48 * VOCAB]).unwrap();
        for (name, cfg) in [
            ("T1 p.95 k20", Cfg::plain(1.0, 0.95, 20)),
            ("T1 k64", Cfg::plain(1.0, 1.0, 64)),
            ("penalties w64", pen.clone()),
        ] {
            let lanes = lanes_for(&mut rng, &host, 8, 6, VOCAB, &cfg, true);
            let _ = run(&mut gpu, &mut arena, &lanes, &logits);
            // One product call (H2D + kernel + D2H + sync): GPU events around
            // it and the host wall clock.
            let n = 200;
            let start = gpu.hip.event_create().unwrap();
            let stop = gpu.hip.event_create().unwrap();
            let cfgs: Vec<SamplerConfig> = lanes.iter().map(|l| sampler_cfg(&l.cfg)).collect();
            let views: Vec<GpuTensor> = lanes
                .iter()
                .map(|l| logits.sub_offset(l.logits_row0 * l.vocab, l.rows * l.vocab))
                .collect();
            let jobs: Vec<LaneDrawJob<'_>> = lanes
                .iter()
                .enumerate()
                .map(|(i, l)| LaneDrawJob {
                    logits: &views[i],
                    rows: l.rows,
                    vocab: l.vocab,
                    drafts: &l.drafts,
                    q: &l.q,
                    history: &l.history,
                    cfg: &cfgs[i],
                    rng_state: l.rng,
                    eos: l.eos,
                })
                .collect();
            let mut ks = Vec::with_capacity(n);
            let mut rts = Vec::with_capacity(n);
            for _ in 0..n {
                let t = Instant::now();
                gpu.hip.event_record(&start, None).unwrap();
                let draws = lane_draws(&mut gpu, &mut arena, &jobs).unwrap();
                gpu.hip.event_record(&stop, None).unwrap();
                rts.push(t.elapsed().as_secs_f64() * 1e6);
                gpu.hip.device_synchronize().unwrap();
                assert_eq!(draws.len(), 8);
                ks.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1e3);
            }
            ks.sort_by(f64::total_cmp);
            rts.sort_by(f64::total_cmp);
            println!(
                "TIMING {name}: 48 rows x {VOCAB}: lane_draws GPU (H2D+kernel+D2H) median {:.1} us p90 {:.1} us; wall median {:.1} us p90 {:.1} us (n={n})",
                ks[n / 2],
                ks[n * 9 / 10],
                rts[n / 2],
                rts[n * 9 / 10]
            );
        }
        gpu.free_tensor(logits).unwrap();
    }

    if fails > 0 {
        println!("FAILED {fails} case(s)");
        std::process::exit(1);
    }
    println!("ALL PASS");
}
