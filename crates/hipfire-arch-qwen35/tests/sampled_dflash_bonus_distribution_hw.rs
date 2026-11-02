// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampled speculative target law against AR sampling, in distribution, on
//! fixed logits.
//!
//! `#[ignore]`d: it needs a real HIP GPU (no model). Every sampled
//! speculative target read on Qwen3.5-family targets truncates its rows with
//! `softmax_temp_topp_batched_f32`'s `tau`/`Z`, then draws by one of:
//!
//! - `host`: `apply_topp_trunc` + a host categorical draw — the DFlash host
//!   accept loop (acceptance rows and the full-accept / budget bonus, via
//!   [`sampled_target_bonus`]), the qwen35 MTP verify and draft rows, and the
//!   n-gram `spec_impl` sampled picks;
//! - `chain`: `chain_accept_spec_f32`'s all-accepted bonus (the DFlash GPU
//!   accept path), run with zero drafts;
//! - `bcat`: `batched_categorical_sample_f32` (the DFlash C8 draft sampler),
//!   `BCAT_STEPS` windows of `BCAT_ROWS` rows seeded as DFlash seeds them
//!   (`spec_draft_row_seed(stream, position, row)`).
//!
//! Without FAST_SAMPLE the DFlash rows are built on the host instead
//! (`hostrow`: [`sampled_target_bonus`] without GPU rows).
//!
//! Each case's top_k is the AR candidate cap (`llama::ar_candidate_cap`, what
//! the speculators pass). The reference is `llama::sample_top_k_p`, the AR
//! producer's host sampler (`sampler::sample_cpu` without penalties): its
//! exact law (top-k pool, temperature softmax, min_p, nucleus over the pool's
//! mass, crossing token included) is computed on the host, and `ar` draws
//! `TRIALS` samples from the sampler itself. For every arm:
//!
//! - the kernel's kept set (`p >= tau`) equals the AR support exactly;
//! - no draw lands outside the AR support;
//! - chi-square goodness-of-fit against the AR law and two-sample homogeneity
//!   against the `ar` draws both have p > `ALPHA`.
//!
//! The `bcat` draws must also be uncoupled: the rank correlation of rows 2k
//! and 2k+1 in a window, and of one row across consecutive windows, stays
//! within 5/sqrt(n).
//!
//! Negative controls, which must fail: `legacy_bonus` (the pre-fix DFlash
//! bonus, a draw from the untruncated softmax row), `legacy_kernel` (a host
//! replica of the pre-fix kernel's tau: 512 buckets over p_max, nucleus over
//! the FULL vocab mass, top_k as a bucket count, `tau = max(tau_p, tau_k)`),
//! and `legacy_bcat` (the pre-fix `(seed ^ row) | 1` row seeding, replayed
//! through the seed buffer), whose rows 2k and 2k+1 must show coupling.
//!
//! Cases: the serve config on a Zipf row where top_k binds (AR keeps 11 of
//! the 20), a ranked-past-k row whose 20th and 21st probabilities share one
//! old-kernel bucket, a min_p case (the MTP path passes min_p), and the
//! absent-top_k pool (cap 20, top_p 1.0).

use hipfire_arch_qwen35::speculative::{sampled_target_bonus, FastTargetRows};
use hipfire_runtime::llama::ar_candidate_cap;
use rdna_compute::sampling::spec_draft_row_seed;
use rdna_compute::{DType, Gpu};
use std::collections::{HashMap, HashSet};

const VOCAB: usize = 32_768;
const TRIALS: usize = 40_000;
/// `chain_accept_spec_f32` draws one bonus per launch.
const CHAIN_TRIALS: usize = 8_000;
/// The FAST_SAMPLE-off host row sorts the vocab per draw.
const HOST_ROW_TRIALS: usize = 6_000;
/// `batched_categorical_sample_f32` rows per launch (one draft window) and
/// windows per case; `BCAT_ROWS * BCAT_STEPS == TRIALS`.
const BCAT_ROWS: usize = 1_000;
const BCAT_STEPS: usize = 40;
const ALPHA: f64 = 1e-3;

struct Case {
    name: &'static str,
    /// Zipf slope of the fixture row.
    slope: f32,
    /// Make the ranks 20 and 21 nearly tied (one old-kernel bucket).
    near_tie_at_20: bool,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
}

const CASES: &[Case] = &[
    Case {
        name: "zipf_binding_topk",
        slope: 1.2,
        near_tie_at_20: false,
        temp: 0.7,
        top_p: 0.95,
        top_k: Some(20),
        min_p: 0.0,
    },
    Case {
        name: "ranked_past_k",
        slope: 0.6,
        near_tie_at_20: true,
        temp: 0.7,
        top_p: 1.0,
        top_k: Some(20),
        min_p: 0.0,
    },
    Case {
        name: "min_p",
        slope: 1.0,
        near_tie_at_20: false,
        temp: 0.8,
        top_p: 0.95,
        top_k: Some(40),
        min_p: 0.05,
    },
    Case {
        name: "absent_top_k",
        slope: 0.9,
        near_tie_at_20: false,
        temp: 1.0,
        top_p: 1.0,
        top_k: None,
        min_p: 0.0,
    },
];

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f32 {
    (splitmix(state) >> 40) as f32 / (1u64 << 24) as f32
}

/// Zipf-shaped logits over shuffled ids with distinct values.
fn fixture_logits(case: &Case, salt: u64) -> Vec<f32> {
    let mut rng = 0x5EED_DF1A_5B0B_u64 ^ salt;
    let mut ids: Vec<usize> = (0..VOCAB).collect();
    for i in (1..VOCAB).rev() {
        let j = (splitmix(&mut rng) % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    let mut logits = vec![0.0f32; VOCAB];
    for (rank, &id) in ids.iter().enumerate() {
        let jitter = (unit(&mut rng) - 0.5) * 0.02;
        logits[id] = 12.0 - case.slope * ((rank + 1) as f32).ln() + jitter;
    }
    if case.near_tie_at_20 {
        logits[ids[20]] = logits[ids[19]] - 2e-3;
    }
    logits
}

/// `llama::sample_top_k_p`'s law, computed in f64.
fn ar_law(logits: &[f32], c: &Case) -> Vec<(u32, f64)> {
    let cap = ar_candidate_cap(c.top_k);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    order.truncate(cap);
    let max = logits[order[0]] as f64;
    let w: Vec<f64> = order
        .iter()
        .map(|&i| ((logits[i] as f64 - max) / c.temp as f64).exp())
        .collect();
    let mut kept_cap = cap;
    if c.min_p > 0.0 {
        if let Some(i) = (1..cap).find(|&i| w[i] < c.min_p as f64 * w[0]) {
            kept_cap = i;
        }
    }
    let pool: f64 = w[..kept_cap].iter().sum();
    let mut cum = 0.0;
    let mut cut = kept_cap;
    for (r, &x) in w[..kept_cap].iter().enumerate() {
        cum += x;
        if cum >= c.top_p.clamp(0.0, 1.0) as f64 * pool {
            cut = r + 1;
            break;
        }
    }
    let kept: f64 = w[..cut].iter().sum();
    order[..cut]
        .iter()
        .zip(&w[..cut])
        .map(|(&i, &x)| (i as u32, x / kept))
        .collect()
}

/// Host replica of the pre-fix `softmax_temp_topp_batched_f32` threshold.
fn legacy_kernel_tau(probs: &[f32], top_p: f32, top_k: usize, min_p: f32) -> f32 {
    if top_p >= 1.0 && top_k == 0 && min_p <= 0.0 {
        return 0.0;
    }
    const NBINS: usize = 512;
    let pmax = probs.iter().cloned().fold(0.0f32, f32::max);
    let floor = pmax * 1e-5;
    let (mut mass, mut cnt) = (vec![0.0f32; NBINS], vec![0.0f32; NBINS]);
    for &p in probs {
        if p < floor {
            continue;
        }
        let b = ((p * (NBINS as f32 / pmax)) as usize).min(NBINS - 1);
        mass[b] += p;
        cnt[b] += 1.0;
    }
    let (mut cm, mut cc, mut tau_p, mut tau_k) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut fp, mut fk) = (top_p >= 1.0, top_k == 0);
    for b in (0..NBINS).rev() {
        if fp && fk {
            break;
        }
        cm += mass[b];
        cc += cnt[b];
        if !fp && cm >= top_p {
            tau_p = b as f32 / NBINS as f32 * pmax;
            fp = true;
        }
        if !fk && cc >= top_k as f32 {
            tau_k = b as f32 / NBINS as f32 * pmax;
            fk = true;
        }
    }
    let tau_minp = if min_p > 0.0 { min_p * pmax } else { 0.0 };
    tau_p.max(tau_k).max(tau_minp)
}

fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma Q(a, x).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_pre = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        (1.0 - sum * ln_pre.exp()).max(0.0)
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        ln_pre.exp() * h
    }
}

fn counts(draws: &[u32]) -> HashMap<u32, f64> {
    let mut c = HashMap::new();
    for &t in draws {
        *c.entry(t).or_insert(0.0) += 1.0;
    }
    c
}

/// Chi-square goodness-of-fit of `draws` against `law` (cells expecting
/// fewer than 5 pooled). A draw outside the support yields p = 0.
fn goodness_of_fit(draws: &[u32], law: &[(u32, f64)]) -> f64 {
    let n = draws.len() as f64;
    let c = counts(draws);
    let support: HashSet<u32> = law.iter().map(|&(t, _)| t).collect();
    if c.keys().any(|t| !support.contains(t)) {
        return 0.0;
    }
    let mut cells: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    for &(t, p) in law {
        let obs = c.get(&t).copied().unwrap_or(0.0);
        if p * n < 5.0 {
            pooled.0 += obs;
            pooled.1 += p * n;
        } else {
            cells.push((obs, p * n));
        }
    }
    if pooled.1 > 0.0 {
        cells.push(pooled);
    }
    let stat: f64 = cells.iter().map(|&(o, e)| (o - e).powi(2) / e).sum();
    let df = cells.len().saturating_sub(1).max(1);
    gamma_q(df as f64 / 2.0, stat / 2.0)
}

/// Two-sample chi-square homogeneity p-value (cells expecting fewer than 5
/// in either sample pooled).
fn homogeneity(a: &[u32], b: &[u32]) -> f64 {
    let (ca, cb) = (counts(a), counts(b));
    let keys: HashSet<u32> = ca.keys().chain(cb.keys()).copied().collect();
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let n = na + nb;
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    for k in keys {
        let (x, y) = (
            ca.get(&k).copied().unwrap_or(0.0),
            cb.get(&k).copied().unwrap_or(0.0),
        );
        if (x + y) * na.min(nb) / n < 5.0 {
            pooled.0 += x;
            pooled.1 += y;
        } else {
            kept.push((x, y));
        }
    }
    if pooled.0 + pooled.1 > 0.0 {
        kept.push(pooled);
    }
    let mut stat = 0.0;
    for &(x, y) in &kept {
        let t = x + y;
        let (ea, eb) = (t * na / n, t * nb / n);
        stat += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    let df = kept.len().saturating_sub(1).max(1);
    gamma_q(df as f64 / 2.0, stat / 2.0)
}

/// Pearson correlation of paired samples (0 when either side is constant).
fn pearson(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let (mx, my) = pairs
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx == 0.0 || syy == 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

/// Categorical draw over `row` truncated at `tau` (no truncation at 0).
fn draw_truncated(row: &[f32], tau: f32, u: f32) -> u32 {
    let mass: f32 = row.iter().filter(|&&p| p >= tau).sum();
    let target = u * mass;
    let mut acc = 0.0f32;
    for (i, &p) in row.iter().enumerate() {
        if p >= tau {
            acc += p;
            if target < acc {
                return i as u32;
            }
        }
    }
    (row.len() - 1) as u32
}

#[test]
#[ignore = "requires a real HIP GPU"]
fn sampled_spec_target_law_matches_ar_distribution() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let mut failures = Vec::new();
    for (ci, c) in CASES.iter().enumerate() {
        let logits = fixture_logits(c, ci as u64);
        let cap = ar_candidate_cap(c.top_k);
        let logits_gpu = gpu.upload_f32(&logits, &[VOCAB]).unwrap();
        let probs_gpu = gpu.alloc_tensor(&[VOCAB], DType::F32).unwrap();
        let tau_gpu = gpu.alloc_tensor(&[1], DType::F32).unwrap();
        let z_gpu = gpu.alloc_tensor(&[1], DType::F32).unwrap();
        gpu.softmax_temp_topp_batched_into_f32(
            &logits_gpu,
            &probs_gpu,
            &tau_gpu,
            &z_gpu,
            VOCAB,
            1,
            c.temp,
            c.top_p,
            cap,
            c.min_p,
        )
        .expect("softmax_temp_topp_batched_into_f32");
        let fast = FastTargetRows::dense(
            gpu.download_f32(&probs_gpu).unwrap(),
            gpu.download_f32(&tau_gpu).unwrap(),
            gpu.download_f32(&z_gpu).unwrap(),
        );

        let law = ar_law(&logits, c);
        let support: HashSet<u32> = law.iter().map(|&(t, _)| t).collect();
        let kept: HashSet<u32> = (0..VOCAB as u32)
            .filter(|&i| fast.probs[i as usize] >= fast.tau[0])
            .collect();
        let legacy_tau = legacy_kernel_tau(
            &fast.probs,
            c.top_p,
            c.top_k.map_or(0, |k| k as usize),
            c.min_p,
        );
        let legacy_kept = fast.probs.iter().filter(|&&p| p >= legacy_tau).count();
        println!(
            "case {}: T{} top_p {} top_k {:?} (cap {cap}) min_p {}: AR keeps {}, kernel keeps {}, pre-fix kernel kept {legacy_kept}",
            c.name,
            c.temp,
            c.top_p,
            c.top_k,
            c.min_p,
            law.len(),
            kept.len()
        );
        if kept != support {
            failures.push(format!(
                "{}: kernel kept set ({}) != AR support ({})",
                c.name,
                kept.len(),
                support.len()
            ));
        }

        let mut rng = 0xB0_0005_u64 ^ ((ci as u64) << 32);
        let mut scratch = Vec::with_capacity(VOCAB);
        let host: Vec<u32> = (0..TRIALS)
            .map(|_| {
                sampled_target_bonus(
                    &mut scratch,
                    0,
                    VOCAB,
                    Some(&fast),
                    &logits,
                    c.temp,
                    cap,
                    c.top_p,
                    unit(&mut rng),
                )
            })
            .collect();
        // DFlash's host rows carry no min_p (the daemon warns and DFlash
        // ignores it); each draw sorts the vocab, so fewer trials.
        let host_row_trials = if c.min_p > 0.0 { 0 } else { HOST_ROW_TRIALS };
        let hostrow: Vec<u32> = (0..host_row_trials)
            .map(|_| {
                sampled_target_bonus(
                    &mut scratch,
                    0,
                    VOCAB,
                    None,
                    &logits,
                    c.temp,
                    cap,
                    c.top_p,
                    unit(&mut rng),
                )
            })
            .collect();

        // chain_accept_spec_f32 with zero drafts: the all-accepted bonus.
        let dummy = gpu.zeros(&[1], DType::F32).unwrap();
        let out = gpu.alloc_tensor(&[4], DType::F32).unwrap();
        let mut chain = Vec::with_capacity(CHAIN_TRIALS);
        for _ in 0..CHAIN_TRIALS {
            let seed = splitmix(&mut rng) as u32;
            gpu.chain_accept_spec_f32(
                &probs_gpu, &dummy, &dummy, &dummy, &tau_gpu, &z_gpu, &dummy, &dummy, &out, 0,
                VOCAB, seed, 0.0,
            )
            .expect("chain_accept_spec_f32");
            let raw = gpu.download_f32(&out).unwrap();
            chain.push(raw[1].to_bits());
        }

        // batched_categorical_sample_f32: BCAT_STEPS draft windows of
        // BCAT_ROWS rows (copies of the row), seeded as DFlash does
        // (`spec_draft_row_seed(stream at window start, position, row)`), and
        // the same windows with the pre-fix in-kernel seeding replayed through
        // the seed buffer (`(stream_hi ^ stream_lo ^ row) | 1`).
        let rows: Vec<f32> = fast.probs.repeat(BCAT_ROWS);
        let rows_gpu = gpu.upload_f32(&rows, &[BCAT_ROWS * VOCAB]).unwrap();
        let taus_gpu = gpu
            .upload_f32(&vec![fast.tau[0]; BCAT_ROWS], &[BCAT_ROWS])
            .unwrap();
        let zs_gpu = gpu
            .upload_f32(&vec![fast.z[0]; BCAT_ROWS], &[BCAT_ROWS])
            .unwrap();
        let tok_gpu = gpu.alloc_tensor(&[BCAT_ROWS], DType::F32).unwrap();
        let pat_gpu = gpu.alloc_tensor(&[BCAT_ROWS], DType::F32).unwrap();
        let bcat_draw = |gpu: &mut Gpu, seeds: &[u32]| -> Vec<u32> {
            let bytes: Vec<u8> = seeds.iter().flat_map(|s| s.to_ne_bytes()).collect();
            let seeds_gpu = gpu.upload_raw(&bytes, &[bytes.len()]).unwrap();
            gpu.batched_categorical_sample_f32(
                &rows_gpu, &taus_gpu, &zs_gpu, &seeds_gpu, &tok_gpu, &pat_gpu, VOCAB, BCAT_ROWS,
            )
            .expect("batched_categorical_sample_f32");
            let raw = gpu.download_f32(&tok_gpu).unwrap();
            let _ = gpu.free_tensor(seeds_gpu);
            raw.iter().map(|v| v.to_bits()).collect()
        };
        let mut stream = splitmix(&mut rng);
        let mut position = 4096u64;
        let (mut bcat, mut legacy_bcat) = (Vec::new(), Vec::new());
        for _ in 0..BCAT_STEPS {
            let seeds: Vec<u32> = (0..BCAT_ROWS as u32)
                .map(|r| spec_draft_row_seed(stream, position, r))
                .collect();
            bcat.extend(bcat_draw(&mut gpu, &seeds));
            let old = (stream >> 32) as u32 ^ stream as u32;
            let seeds: Vec<u32> = (0..BCAT_ROWS as u32).map(|r| (old ^ r) | 1).collect();
            legacy_bcat.extend(bcat_draw(&mut gpu, &seeds));
            stream = splitmix(&mut stream.clone());
            position += 3;
        }
        for t in [
            rows_gpu, taus_gpu, zs_gpu, tok_gpu, pat_gpu, dummy, out, probs_gpu, tau_gpu, z_gpu,
            logits_gpu,
        ] {
            let _ = gpu.free_tensor(t);
        }
        // Coupling: rank (in the AR law's order) correlation of rows 2k vs
        // 2k+1 in one window, and of one row in consecutive windows.
        let rank: HashMap<u32, f64> = law
            .iter()
            .enumerate()
            .map(|(i, &(t, _))| (t, i as f64))
            .collect();
        let r_of = |d: &[u32], i: usize| rank.get(&d[i]).copied().unwrap_or(law.len() as f64);
        for (arm, d, control) in [("bcat", &bcat, false), ("legacy_bcat", &legacy_bcat, true)] {
            let mut within = Vec::new();
            let mut across = Vec::new();
            for s in 0..BCAT_STEPS {
                for k in 0..BCAT_ROWS / 2 {
                    let i = s * BCAT_ROWS + 2 * k;
                    within.push((r_of(d, i), r_of(d, i + 1)));
                }
                if s + 1 < BCAT_STEPS {
                    for row in 0..BCAT_ROWS {
                        let i = s * BCAT_ROWS + row;
                        across.push((r_of(d, i), r_of(d, i + BCAT_ROWS)));
                    }
                }
            }
            let (rw, ra) = (pearson(&within), pearson(&across));
            let (lw, la) = (
                5.0 / (within.len() as f64).sqrt(),
                5.0 / (across.len() as f64).sqrt(),
            );
            let coupled = rw.abs() > lw || ra.abs() > la;
            println!(
                "  {arm:>13}: row-pair corr {rw:+.4} (|r| <= {lw:.4}), cross-window corr {ra:+.4} (|r| <= {la:.4})"
            );
            if coupled != control {
                failures.push(format!(
                    "{}: {arm} coupling row-pair {rw:+.4} cross-window {ra:+.4} (control {control})",
                    c.name
                ));
            }
        }

        let legacy_bonus: Vec<u32> = (0..TRIALS)
            .map(|_| draw_truncated(&fast.probs, 0.0, unit(&mut rng)))
            .collect();
        let legacy_kernel: Vec<u32> = (0..TRIALS)
            .map(|_| draw_truncated(&fast.probs, legacy_tau, unit(&mut rng)))
            .collect();
        hipfire_runtime::llama::reset_cpu_sampler_rng(0x5A11_0000 ^ ci as u32);
        let min_p = (c.min_p > 0.0).then_some(c.min_p);
        let ar: Vec<u32> = (0..TRIALS)
            .map(|_| {
                hipfire_runtime::llama::sample_top_k_p(&logits, c.temp, c.top_p, c.top_k, min_p)
            })
            .collect();

        for (arm, draws, control) in [
            ("ar", &ar, false),
            ("host", &host, false),
            ("hostrow", &hostrow, false),
            ("chain", &chain, false),
            ("bcat", &bcat, false),
            ("legacy_bonus", &legacy_bonus, true),
            ("legacy_kernel", &legacy_kernel, true),
        ] {
            if draws.is_empty() {
                continue;
            }
            let outside = draws.iter().filter(|t| !support.contains(t)).count();
            let p_fit = goodness_of_fit(draws, &law);
            let p_ar = homogeneity(draws, &ar);
            println!(
                "  {arm:>13}: n {:>5}  outside {outside:>5}  gof p {p_fit:.3e}  vs ar p {p_ar:.3e}",
                draws.len()
            );
            let pass = outside == 0 && p_fit > ALPHA && p_ar > ALPHA;
            if control {
                // Pre-fix callers passed top_k_cut() (absent → 0): with
                // top_p 1.0 the old kernel did not truncate at all.
                if pass {
                    failures.push(format!("{}: negative control {arm} passed", c.name));
                }
            } else if !pass {
                failures.push(format!(
                    "{} {arm}: outside {outside}, gof p {p_fit:.3e}, vs ar p {p_ar:.3e}",
                    c.name
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
