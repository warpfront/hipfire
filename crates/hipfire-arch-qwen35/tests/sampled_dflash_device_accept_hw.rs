// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware oracle of the singleton device sampled accept
//! (`HIPFIRE_DFLASH_DEVICE_ACCEPT`, `dflash_sampled_accept_sparse_q_f32`).
//!
//! `#[ignore]`d: it needs a real HIP GPU (no model). Run explicitly:
//!
//!   cargo test -p hipfire-arch-qwen35 --release --test sampled_dflash_device_accept_hw -- --ignored --nocapture
//!   (production vocabulary: prefix `HIPFIRE_DSA_HW_VOCAB=248320 HIPFIRE_DSA_HW_TRIALS=300`)
//!
//! The kernel is compared, decision for decision and bit for bit, with its host
//! twin `dflash_sampled_accept_reference` (same 256-segment CDF association) on
//! the SAME device softmax rows, uniforms and sparse q:
//!
//! - `serve`: the serve config (top_p 0.9, top_k cap 20) on Zipf rows, a
//!   selector-shaped sparse q (top-8 support, proposals drawn from q);
//! - `point_mass`: the PLD q (one entry, `q = 1`);
//! - `full_support`: synthetic `tau = 0` rows (identity truncation, kept set =
//!   the whole vocabulary: no compaction, no overflow row, no dense download);
//! - `budget`: `max_accept` 0 / 2 (bonus drawn from the boundary row);
//! - `degenerate`: q mass above p everywhere (argmax fallback, code 1);
//! - `tie_rows`: a row whose kept set is a flat run of equal probabilities.
//!
//! and with the exact law: with one proposed row, x ~ q then accept / residual,
//! the committed token is distributed as the truncated target row (6-sigma cell
//! bounds over 200k windows, support never exceeded). The end-to-end entry
//! `dflash_sampled_accept_device` (softmax + upload + launch + 16-byte
//! readback) must equal the twin on the uniforms it pre-draws from its RNG and
//! be deterministic per seed.

use hipfire_arch_qwen35::speculative::{
    dflash_sampled_accept_device, DflashSampleCfg, DflashSampledDraft,
};
use rdna_compute::dflash_sampled_accept::{
    dflash_accept_pack_payload, dflash_sampled_accept_reference, DflashAcceptScratch,
    DflashDeviceDecision,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Vocabulary of the synthetic rows: 32768 by default; `HIPFIRE_DSA_HW_VOCAB=248320`
/// runs every case at the production vocabulary (segment 970 instead of 128).
fn vocab() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("HIPFIRE_DSA_HW_VOCAB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32_768)
    });
    *V
}
/// Windows per case: 3000 by default; `HIPFIRE_DSA_HW_TRIALS` overrides.
fn trials() -> usize {
    static T: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("HIPFIRE_DSA_HW_TRIALS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3_000)
    });
    *T
}
const LAW_TRIALS: usize = 200_000;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0
    }
}

/// Zipf-shaped logits over shuffled ids with distinct values.
fn logits(rows: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg(0xACCE_97 ^ salt);
    let mut out = Vec::with_capacity(rows * vocab());
    for _ in 0..rows {
        let mut ids: Vec<usize> = (0..vocab()).collect();
        for i in (1..vocab()).rev() {
            ids.swap(i, rng.below(i + 1));
        }
        let mut row = vec![0f32; vocab()];
        for (rank, &id) in ids.iter().enumerate() {
            row[id] = 9.0 - 1.1 * ((rank + 1) as f32).ln() - 1e-4 * rank as f32 + 0.02 * rng.unit();
        }
        out.extend(row);
    }
    out
}

struct Rows {
    probs: GpuTensor,
    tau: GpuTensor,
    z: GpuTensor,
    h_probs: Vec<f32>,
    h_tau: Vec<f32>,
    h_z: Vec<f32>,
}

fn softmax_rows(gpu: &mut Gpu, lg: &[f32], rows: usize, top_p: f32, top_k: usize) -> Rows {
    let l = gpu.upload_f32(lg, &[rows * vocab()]).unwrap();
    let probs = gpu.alloc_tensor(&[rows * vocab()], DType::F32).unwrap();
    let tau = gpu.alloc_tensor(&[rows], DType::F32).unwrap();
    let z = gpu.alloc_tensor(&[rows], DType::F32).unwrap();
    gpu.softmax_temp_topp_batched_into_f32(
        &l,
        &probs,
        &tau,
        &z,
        vocab(),
        rows,
        0.8,
        top_p,
        top_k,
        0.0,
    )
    .unwrap();
    resident(gpu, probs, tau, z)
}

fn resident(gpu: &mut Gpu, probs: GpuTensor, tau: GpuTensor, z: GpuTensor) -> Rows {
    Rows {
        h_probs: gpu.download_f32(&probs).unwrap(),
        h_tau: gpu.download_f32(&tau).unwrap(),
        h_z: gpu.download_f32(&z).unwrap(),
        probs,
        tau,
        z,
    }
}

/// Synthetic rows uploaded as is (any `tau` / `z`).
fn synthetic_rows(gpu: &mut Gpu, probs: &[f32], tau: &[f32], z: &[f32]) -> Rows {
    let p = gpu.upload_f32(probs, &[probs.len()]).unwrap();
    let t = gpu.upload_f32(tau, &[tau.len()]).unwrap();
    let zz = gpu.upload_f32(z, &[z.len()]).unwrap();
    resident(gpu, p, t, zz)
}

/// Sparse selector-shaped q rows: `support` ids per row (top ranks of the
/// row's own probabilities plus random ids), weights summing to 1.
fn selector_q(rows: &Rows, n: usize, support: usize, rng: &mut Lcg) -> Vec<Vec<(u32, f32)>> {
    (0..n)
        .map(|r| {
            let row = &rows.h_probs[r * vocab()..(r + 1) * vocab()];
            let mut order: Vec<usize> = (0..vocab()).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let mut ids: Vec<u32> = order[..support / 2].iter().map(|&i| i as u32).collect();
            ids.extend((0..support / 2).map(|_| rng.below(vocab()) as u32));
            ids.sort_unstable();
            ids.dedup();
            let w: Vec<f32> = ids.iter().map(|_| rng.unit() + 0.05).collect();
            let s: f32 = w.iter().sum();
            ids.into_iter().zip(w).map(|(i, w)| (i, w / s)).collect()
        })
        .collect()
}

fn draw(q: &[(u32, f32)], u: f32) -> (u32, f32) {
    let mut acc = 0.0;
    for &(i, v) in q {
        acc += v;
        if u < acc {
            return (i, v);
        }
    }
    *q.last().unwrap()
}

fn pack(
    tokens: &[u32],
    us: &[f32],
    u_fin: f32,
    max_accept: usize,
    selq: &[f32],
    q: &[Vec<(u32, f32)>],
) -> Vec<u32> {
    let mut off = vec![0u32];
    let (mut ids, mut vals) = (Vec::new(), Vec::new());
    for row in q {
        for &(i, v) in row {
            ids.push(i);
            vals.push(v);
        }
        off.push(ids.len() as u32);
    }
    let mut out = Vec::new();
    dflash_accept_pack_payload(
        &mut out, tokens, us, u_fin, max_accept, selq, &off, &ids, &vals,
    )
    .unwrap();
    out
}

/// Kernel vs twin over `trials()` windows of `n` rows. Returns the decisions.
fn kernel_equals_twin(
    gpu: &mut Gpu,
    scratch: &DflashAcceptScratch,
    name: &str,
    rows: &Rows,
    n: usize,
    mut make_q: impl FnMut(&mut Lcg) -> Vec<Vec<(u32, f32)>>,
    max_accept: impl Fn(usize) -> usize,
) -> Vec<DflashDeviceDecision> {
    let mut rng = Lcg(0x0_AC_0DE ^ name.len() as u64);
    let mut decisions = Vec::new();
    for k in 0..trials() {
        let q = make_q(&mut rng);
        let mut tokens = Vec::new();
        let mut selq = Vec::new();
        for row in q.iter() {
            let (t, v) = draw(row, rng.unit());
            tokens.push(t);
            selq.push(v);
        }
        let us: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
        let u_fin = rng.unit();
        let payload = pack(&tokens, &us, u_fin, max_accept(k), &selq, &q);
        let dev = gpu
            .dflash_sampled_accept_sparse_q_f32(
                scratch,
                &rows.probs,
                &rows.tau,
                &rows.z,
                &payload,
                n,
                vocab(),
            )
            .unwrap_or_else(|e| panic!("{name} #{k}: {e}"));
        let twin = dflash_sampled_accept_reference(
            &rows.h_probs,
            &rows.h_tau,
            &rows.h_z,
            &payload,
            n,
            vocab(),
        );
        assert_eq!(dev, twin, "{name} #{k}: kernel != twin");
        // The retained hipcc kernel is the non-production oracle of the builder kernel.
        let oracle = gpu
            .dflash_sampled_accept_sparse_q_f32_hip_oracle(
                scratch,
                &rows.probs,
                &rows.tau,
                &rows.z,
                &payload,
                n,
                vocab(),
            )
            .unwrap_or_else(|e| panic!("{name} #{k} (hip oracle): {e}"));
        assert_eq!(dev, oracle, "{name} #{k}: builder kernel != hipcc oracle");
        decisions.push(dev);
    }
    let rej = decisions.iter().filter(|d| d.rejected).count();
    let by_len = (0..=n)
        .map(|l| decisions.iter().filter(|d| d.accept_len == l).count())
        .collect::<Vec<_>>();
    println!(
        "{name}: {} windows, {rej} rejected, accept_len histogram {by_len:?}",
        trials()
    );
    decisions
}

#[test]
#[ignore = "requires a real HIP GPU"]
fn device_accept_kernel_equals_host_twin() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let n = 4;
    let scratch = DflashAcceptScratch::new(&mut gpu, 1, 1, 4 * n + 4 + 2 * 8 * n + 64).unwrap();

    // serve config rows (tau > 0, kept set <= cap).
    let lg = logits(n + 1, 1);
    let rows = softmax_rows(&mut gpu, &lg, n + 1, 0.9, 20);
    let d = kernel_equals_twin(
        &mut gpu,
        &scratch,
        "serve",
        &rows,
        n,
        |r| selector_q(&rows, n, 8, r),
        |_| n,
    );
    assert!(d.iter().any(|d| d.rejected) && d.iter().any(|d| !d.rejected));
    assert!(d.iter().all(|d| d.fallback == 0));

    // PLD point mass: q = delta(token) on tokens of the target's own top ranks.
    let tops: Vec<Vec<u32>> = (0..n)
        .map(|r| {
            let row = &rows.h_probs[r * vocab()..(r + 1) * vocab()];
            let mut o: Vec<u32> = (0..vocab() as u32).collect();
            o.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap());
            o.truncate(12);
            o
        })
        .collect();
    let d = kernel_equals_twin(
        &mut gpu,
        &scratch,
        "point_mass",
        &rows,
        n,
        |r| (0..n).map(|i| vec![(tops[i][r.below(12)], 1.0)]).collect(),
        |_| n,
    );
    assert!(
        d.iter().any(|w| w.rejected),
        "point-mass windows must reject sometimes"
    );
    for w in d.iter().filter(|w| w.rejected) {
        let p = rows.h_probs[w.accept_len * vocab() + w.bonus as usize];
        assert!(
            p >= rows.h_tau[w.accept_len],
            "residual left the kept set: {w:?}"
        );
    }

    // Budget bound: the bonus is drawn from the boundary row.
    kernel_equals_twin(
        &mut gpu,
        &scratch,
        "budget",
        &rows,
        n,
        |r| selector_q(&rows, n, 8, r),
        |k| [0, 2, n][k % 3],
    );

    // Full support (tau = 0): the identity row, no kept-set cut at all.
    let mut probs = Vec::new();
    let mut rng = Lcg(77);
    for _ in 0..=n {
        let mut r: Vec<f32> = (0..vocab()).map(|_| rng.unit().powi(4) + 1e-6).collect();
        let s: f32 = r.iter().sum();
        r.iter_mut().for_each(|v| *v /= s);
        probs.extend(r);
    }
    let full = synthetic_rows(&mut gpu, &probs, &vec![0.0; n + 1], &vec![1.0; n + 1]);
    let d = kernel_equals_twin(
        &mut gpu,
        &scratch,
        "full_support",
        &full,
        n,
        |r| selector_q(&full, n, 8, r),
        |_| n,
    );
    assert!(
        d.iter().any(|w| w.bonus as usize > vocab() / 2),
        "draws must reach the whole vocabulary"
    );

    // Degenerate residual: q puts more mass than p on the proposed id (p is a
    // point mass there), so p - q <= 0 everywhere.
    let mut probs = vec![0f32; (n + 1) * vocab()];
    for r in 0..=n {
        probs[r * vocab() + 7 + r] = 1.0;
    }
    let spike = synthetic_rows(&mut gpu, &probs, &vec![1.0; n + 1], &vec![1.0; n + 1]);
    let d = kernel_equals_twin(
        &mut gpu,
        &scratch,
        "degenerate",
        &spike,
        n,
        |_| (0..n).map(|r| vec![((7 + r) as u32, 2.0)]).collect(),
        |_| n,
    );
    for w in d.iter().filter(|w| w.rejected) {
        assert_eq!((w.fallback, w.bonus as usize), (1, 7 + w.accept_len));
    }

    // Flat run of equal probabilities at the kept boundary (tie kept set).
    let mut probs = vec![1e-7f32; (n + 1) * vocab()];
    for r in 0..=n {
        for j in 0..40 {
            probs[r * vocab() + 800 * j + r] = 0.025;
        }
    }
    let flat = synthetic_rows(&mut gpu, &probs, &vec![0.025; n + 1], &vec![1.0; n + 1]);
    kernel_equals_twin(
        &mut gpu,
        &scratch,
        "tie_rows",
        &flat,
        n,
        |r| selector_q(&flat, n, 8, r),
        |_| n,
    );
}

/// With one proposed row the committed token is the truncated target row's law.
#[test]
#[ignore = "requires a real HIP GPU"]
fn device_accept_committed_token_law() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let scratch = DflashAcceptScratch::new(&mut gpu, 1, 1, 64).unwrap();
    let lg = logits(2, 9);
    let rows = softmax_rows(&mut gpu, &lg, 2, 0.9, 20);
    let inv = 1.0f32 / rows.h_z[0].max(f32::MIN_POSITIVE);
    let target: Vec<f32> = rows.h_probs[..vocab()]
        .iter()
        .map(|&p| if p >= rows.h_tau[0] { p * inv } else { 0.0 })
        .collect();
    let mut rng = Lcg(31337);
    let q = selector_q(&rows, 1, 12, &mut rng).remove(0);
    let mut counts = std::collections::HashMap::<u32, u32>::new();
    for _ in 0..LAW_TRIALS {
        let (x, qx) = draw(&q, rng.unit());
        let payload = pack(&[x], &[rng.unit()], rng.unit(), 1, &[qx], &[q.clone()]);
        let d = gpu
            .dflash_sampled_accept_sparse_q_f32(
                &scratch,
                &rows.probs,
                &rows.tau,
                &rows.z,
                &payload,
                1,
                vocab(),
            )
            .unwrap();
        *counts
            .entry(if d.rejected { d.bonus } else { x })
            .or_default() += 1;
    }
    for (&tok, _) in &counts {
        assert!(
            target[tok as usize] > 0.0,
            "token {tok} is outside the kept set"
        );
    }
    let (mut chi, mut cells) = (0.0f64, 0usize);
    for c in 0..vocab() {
        let p = target[c] as f64;
        let e = p * LAW_TRIALS as f64;
        let got = *counts.get(&(c as u32)).unwrap_or(&0) as f64;
        let sd = (e * (1.0 - p)).sqrt();
        assert!(
            (got - e).abs() <= 6.0 * sd + 3.0,
            "token {c}: {got} draws, expected {e:.1} (sd {sd:.1})"
        );
        if e >= 5.0 {
            chi += (got - e) * (got - e) / e;
            cells += 1;
        }
    }
    // Pearson chi-square over the kept cells (every one expects >= 5): the
    // Wilson-Hilferty normal score must stay below 4 (one-sided p > 3e-5).
    let k = (cells - 1) as f64;
    let z = ((chi / k).cbrt() - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    println!("committed-token law: chi2 {chi:.1} over {cells} cells, WH z {z:.2}");
    assert!(
        z < 4.0,
        "chi-square rejects the target law: chi2 {chi:.1}, {cells} cells, z {z:.2}"
    );
}

/// End to end: softmax + pre-drawn uniforms + upload + launch + readback.
#[test]
#[ignore = "requires a real HIP GPU"]
fn device_accept_entry_is_twin_exact_and_deterministic() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let b = 5;
    let n = b - 1;
    let lg = logits(b, 5);
    let logits_gpu = gpu.upload_f32(&lg, &[b * vocab()]).unwrap();
    let cfg = DflashSampleCfg {
        temp: 0.8,
        top_p: 0.9,
        top_k: 20,
        cactus_delta: 0.0,
    };
    let rows = softmax_rows(&mut gpu, &lg, b, 0.9, 20);

    let mut rng = Lcg(2024);
    // Starts empty and tiny: the entry must grow it on demand.
    let mut slot: Option<DflashAcceptScratch> = None;
    for case in 0..200 {
        let q = selector_q(&rows, n, 8, &mut rng);
        let mut draft = DflashSampledDraft::default();
        draft.sparse = true;
        draft.sparse_q.clear();
        draft.block = vec![123];
        for row in &q {
            let (t, v) = draw(row, rng.unit());
            let ids: Vec<u32> = row.iter().map(|e| e.0).collect();
            let vals: Vec<f32> = row.iter().map(|e| e.1).collect();
            draft.sparse_q.push_row(&ids, &vals, vocab()).unwrap();
            draft.selected_q.push(v);
            draft.block.push(t);
        }
        let max_accept = [usize::MAX, 2, 0][case % 3];
        let seed = 0x9E37_79B9_7F4A_7C15u64 ^ case as u64;

        // The uniforms the entry pre-draws: n accept ones, then the final one.
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) * (1.0 / 16_777_216.0)
        };
        let us: Vec<f32> = (0..n).map(|_| next()).collect();
        let u_fin = next();
        let mut payload = Vec::new();
        dflash_accept_pack_payload(
            &mut payload,
            &draft.block[1..],
            &us,
            u_fin,
            max_accept,
            &draft.selected_q,
            draft.sparse_q.offsets(),
            &draft.sparse_q.idx,
            &draft.sparse_q.val,
        )
        .unwrap();
        let twin = dflash_sampled_accept_reference(
            &rows.h_probs,
            &rows.h_tau,
            &rows.h_z,
            &payload,
            n,
            vocab(),
        );

        let mut r1 = seed;
        let a = dflash_sampled_accept_device(
            &mut gpu,
            &logits_gpu,
            &draft.block,
            &draft,
            vocab(),
            cfg,
            max_accept,
            &mut r1,
            &mut slot,
        )
        .unwrap();
        let mut r2 = seed;
        let c = dflash_sampled_accept_device(
            &mut gpu,
            &logits_gpu,
            &draft.block,
            &draft,
            vocab(),
            cfg,
            max_accept,
            &mut r2,
            &mut slot,
        )
        .unwrap();
        assert_eq!(
            (a.accepted_drafts, a.bonus_token),
            (twin.accept_len, twin.bonus),
            "case {case}"
        );
        assert_eq!(
            (a.accepted_drafts, a.bonus_token, r1),
            (c.accepted_drafts, c.bonus_token, r2),
            "case {case}: not deterministic"
        );
        assert_ne!(r1, seed, "the entry must advance the host RNG");
    }
}
