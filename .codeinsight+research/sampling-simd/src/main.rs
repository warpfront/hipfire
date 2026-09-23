//! Standalone CPU experiment: AVX2 block-rejection candidate for the fixed-20
//! top-K scan (`sample_top_p`) and greedy argmax in
//! `crates/hipfire-runtime/src/llama.rs` (production base 77cd6ce7d).
//!
//! Scope rules for this file:
//! - Production is untouched. The scalar reference below is copied unchanged
//!   from `llama.rs` (`argmax`, the `sample_top_p` top-K scan, and the
//!   softmax/sort/RNG remainder) and serves as an independent oracle.
//! - No approximate exp/sum, no reassociation, no RNG changes. The stochastic
//!   remainder (softmax over exactly the 20 winners, insertion sort, xorshift
//!   draw) runs bit-identical scalar code for both paths; only the *scan*
//!   that selects the 20 candidates (and the greedy argmax) has an AVX2 form.
//! - AVX2 is pure block *rejection*: an 8-lane finite-AND-above-threshold
//!   compare skips whole blocks when the mask is zero; any nonzero block
//!   falls back to the exact scalar body lane-by-lane in original index
//!   order (rechecking the live threshold, since it is monotonic within the
//!   block). No out-of-bounds loads: tails run the scalar loop.
//! - Inputs are explicitly synthetic logits sized to the measured tokenizer
//!   vocab (248144) and the possibly-padded logit row (248320). The sampler
//!   scans `logits.len()`, so both sizes are benchmarked and labelled.
//!
//! Run (parent measures):
//!   cargo run --release --manifest-path \
//!     .codeinsight+research/sampling-simd/Cargo.toml -- result.json

use std::hint::black_box;
use std::time::Instant;

use hipfire_runtime::llama as prod;

const TOP_K: usize = 20;
// Logit-row sizes under test: measured tokenizer vocab and possibly-padded row.
const SIZES: [usize; 2] = [248_144, 248_320];
const BIG_REPS_SCAN: u64 = 30;
const BIG_REPS_ARGMAX: u64 = 30;
const SMALL_REPS: u64 = 300;

// ---------------------------------------------------------------------------
// Local RNG: bit-exact replica of the `llama.rs` xorshift32 stream.
// `reset_cpu_sampler_rng(seed)` maps 0 -> 1, then each `simple_rand` call
// advances `s ^= s<<13; s ^= s>>17; s ^= s<<5` and yields `s / u32::MAX`.
// With a nonzero seed the time-seeded branch is never taken, so driving this
// local RNG from the same seed reproduces the production draw stream exactly.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct LocalRng(u32);

impl LocalRng {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next_f32(&mut self) -> f32 {
        let mut s = self.0;
        if s == 0 {
            s = 1;
        }
        // xorshift32 (copied from llama.rs `simple_rand`)
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.0 = s;
        (s as f32) / (u32::MAX as f32)
    }
}

// ---------------------------------------------------------------------------
// Scalar reference: scan + argmax copied unchanged from llama.rs.
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct ScanOut {
    val: [f32; TOP_K],
    idx: [u32; TOP_K],
    max: f32,
}

fn scalar_scan(logits: &[f32]) -> ScanOut {
    let mut topk_val = [f32::NEG_INFINITY; TOP_K];
    let mut topk_idx = [0u32; TOP_K];
    let mut min_pos = 0usize; // index of smallest element in topk
    let mut min_val = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() {
            continue;
        }
        if l > max_logit {
            max_logit = l;
        }
        if l > min_val {
            topk_val[min_pos] = l;
            topk_idx[min_pos] = i as u32;
            // Find new min
            min_val = f32::INFINITY;
            for j in 0..TOP_K {
                if topk_val[j] < min_val {
                    min_val = topk_val[j];
                    min_pos = j;
                }
            }
        }
    }
    ScanOut {
        val: topk_val,
        idx: topk_idx,
        max: max_logit,
    }
}

fn scalar_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v.is_finite() && v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0 as u32
}

// Softmax/sort/RNG remainder copied unchanged from `sample_top_p`; operates
// on a finished scan so both the scalar and AVX2 paths share it bit-exactly.
fn finish_sample(
    logits: &[f32],
    scan: &ScanOut,
    temperature: f32,
    top_p: f32,
    rng: &mut LocalRng,
) -> u32 {
    if temperature <= 0.0 {
        return scalar_argmax(logits);
    }
    let top_p = top_p.clamp(0.0, 1.0);
    let inv_temp = 1.0 / temperature;
    let max_logit = scan.max;

    let mut probs = [0.0f32; TOP_K];
    let mut sum = 0.0f32;
    for i in 0..TOP_K {
        let p = if scan.val[i].is_finite() {
            let pp = ((scan.val[i] - max_logit) * inv_temp).exp();
            if pp.is_finite() {
                pp
            } else {
                0.0
            }
        } else {
            0.0
        };
        probs[i] = p;
        sum += p;
    }

    if sum <= 0.0 || !sum.is_finite() {
        return scalar_argmax(logits);
    }

    // Sort descending by probability (insertion sort on 20 elements)
    let mut order: [usize; TOP_K] = core::array::from_fn(|i| i);
    for i in 1..TOP_K {
        let mut j = i;
        while j > 0 && probs[order[j]] > probs[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }

    // Top-p filtering + sampling in one pass
    let r = rng.next_f32() * sum; // pre-scale by total sum
    let mut cumulative = 0.0f32;
    let mut sample_acc = 0.0f32;
    let threshold = top_p * sum;
    for &k in &order {
        cumulative += probs[k];
        sample_acc += probs[k];
        if sample_acc >= r {
            return scan.idx[k];
        }
        if cumulative >= threshold {
            // Past top_p — sample from what we have
            let r2 = rng.next_f32() * cumulative;
            let mut acc2 = 0.0f32;
            for &k2 in &order {
                acc2 += probs[k2];
                if acc2 >= r2 {
                    return scan.idx[k2];
                }
                if acc2 >= cumulative {
                    break;
                }
            }
            return scan.idx[order[0]];
        }
    }
    scan.idx[order[0]]
}

// ---------------------------------------------------------------------------
// AVX2 block-rejection candidate (x86_64 only).
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::ScanOut;
    use super::TOP_K;
    use std::arch::x86_64::*;

    /// 8-lane scan: skip a whole block when no lane is both finite and
    /// above the live `min_val` threshold. Any nonzero mask is processed
    /// lane-by-lane in original index order with the exact scalar body
    /// (finite recheck, live-threshold recheck, strict-`>` max/slot updates,
    /// lowest-slot-wins min rescan), so selection order, ties, signed-zero
    /// bits and `max_logit` match the scalar oracle bit-exactly.
    ///
    /// Skipping is sound for `max_logit` too: the invariant
    /// `max_logit >= min_val` holds from the shared NEG_INFINITY seed, so a
    /// block with every lane `<= min_val` (or non-finite) cannot hold a new
    /// max. Finite test is the ordered `abs(v) <= f32::MAX` compare (NaN and
    /// +-Inf both fail it, exactly like `is_finite`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn scan(logits: &[f32]) -> ScanOut {
        let mut topk_val = [f32::NEG_INFINITY; TOP_K];
        let mut topk_idx = [0u32; TOP_K];
        let mut min_pos = 0usize;
        let mut min_val = f32::NEG_INFINITY;
        let mut max_logit = f32::NEG_INFINITY;

        let n = logits.len();
        let ptr = logits.as_ptr();
        let fmax_b = _mm256_set1_ps(f32::MAX);
        let sign_b = _mm256_set1_ps(-0.0f32);
        // Full 8-lane blocks only; the tail uses the scalar loop, so SIMD
        // never reads out of bounds.
        let blocked = n & !7usize;
        let mut base = 0usize;
        while base < blocked {
            let v = _mm256_loadu_ps(ptr.add(base));
            let abs = _mm256_andnot_ps(sign_b, v);
            let finite = _mm256_cmp_ps(abs, fmax_b, _CMP_LE_OQ);
            let gt = _mm256_cmp_ps(v, _mm256_set1_ps(min_val), _CMP_GT_OQ);
            let both = _mm256_and_ps(finite, gt);
            let mask = _mm256_movemask_ps(both) as u32;
            if mask != 0 {
                // Candidate lanes in ORIGINAL increasing index order.
                let mut lane = 0u32;
                let mut m = mask;
                while m != 0 {
                    if (m & 1) != 0 {
                        let l = *ptr.add(base + lane as usize);
                        // Exact scalar body, rechecked against the live
                        // threshold (monotonic within the block: a lane that
                        // passed at block entry but no longer passes cannot
                        // be a new max either, since max >= min always).
                        if l.is_finite() {
                            if l > max_logit {
                                max_logit = l;
                            }
                            if l > min_val {
                                topk_val[min_pos] = l;
                                topk_idx[min_pos] = (base + lane as usize) as u32;
                                min_val = f32::INFINITY;
                                for j in 0..TOP_K {
                                    if topk_val[j] < min_val {
                                        min_val = topk_val[j];
                                        min_pos = j;
                                    }
                                }
                            }
                        }
                    }
                    lane += 1;
                    m >>= 1;
                }
            }
            base += 8;
        }
        while base < n {
            let l = *ptr.add(base);
            if l.is_finite() {
                if l > max_logit {
                    max_logit = l;
                }
                if l > min_val {
                    topk_val[min_pos] = l;
                    topk_idx[min_pos] = base as u32;
                    min_val = f32::INFINITY;
                    for j in 0..TOP_K {
                        if topk_val[j] < min_val {
                            min_val = topk_val[j];
                            min_pos = j;
                        }
                    }
                }
            }
            base += 1;
        }
        ScanOut {
            val: topk_val,
            idx: topk_idx,
            max: max_logit,
        }
    }

    /// Greedy argmax with block skip against the live best: a block whose
    /// lanes are all non-finite or `<= best` cannot change the fold, so it
    /// is skipped; otherwise lanes run in order with the exact scalar
    /// predicate (first-max-wins ties, all-non-finite falls back to 0).
    #[target_feature(enable = "avx2")]
    pub unsafe fn argmax(logits: &[f32]) -> u32 {
        let n = logits.len();
        let ptr = logits.as_ptr();
        let mut best_i = 0usize;
        let mut best = f32::NEG_INFINITY;
        let fmax_b = _mm256_set1_ps(f32::MAX);
        let sign_b = _mm256_set1_ps(-0.0f32);
        let blocked = n & !7usize;
        let mut base = 0usize;
        while base < blocked {
            let v = _mm256_loadu_ps(ptr.add(base));
            let abs = _mm256_andnot_ps(sign_b, v);
            let finite = _mm256_cmp_ps(abs, fmax_b, _CMP_LE_OQ);
            let gt = _mm256_cmp_ps(v, _mm256_set1_ps(best), _CMP_GT_OQ);
            let both = _mm256_and_ps(finite, gt);
            let mask = _mm256_movemask_ps(both) as u32;
            if mask != 0 {
                for lane in 0..8usize {
                    let l = *ptr.add(base + lane);
                    if l.is_finite() && l > best {
                        best = l;
                        best_i = base + lane;
                    }
                }
            }
            base += 8;
        }
        while base < n {
            let l = *ptr.add(base);
            if l.is_finite() && l > best {
                best = l;
                best_i = base;
            }
            base += 1;
        }
        best_i as u32
    }
}

fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn dispatch_scan(logits: &[f32]) -> ScanOut {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // Safe wrapper: `avx2::scan` only issues in-bounds 8-lane loads
            // plus the scalar tail, and requires AVX2 which we just detected.
            return unsafe { avx2::scan(logits) };
        }
    }
    scalar_scan(logits)
}

fn dispatch_argmax(logits: &[f32]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return unsafe { avx2::argmax(logits) };
        }
    }
    scalar_argmax(logits)
}

/// Direct AVX2 entry points so the harness explicitly exercises the SIMD
/// path (not just dispatch). `None` when the host lacks AVX2.
fn direct_avx2_scan(logits: &[f32]) -> Option<ScanOut> {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return Some(unsafe { avx2::scan(logits) });
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = logits;
        None
    }
}

fn direct_avx2_argmax(logits: &[f32]) -> Option<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return Some(unsafe { avx2::argmax(logits) });
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = logits;
        None
    }
}

// ---------------------------------------------------------------------------
// Deterministic synthetic input generator (independent of the sampler RNG).
// ---------------------------------------------------------------------------
struct Gen(u64);

impl Gen {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 32) as u32 as f32) / (u32::MAX as f32)
    }
}

fn gen_random_uniform(n: usize, seed: u64) -> Vec<f32> {
    let mut g = Gen(seed);
    (0..n).map(|_| g.next_unit_f32() * 20.0 - 10.0).collect()
}

fn gen_ascending(n: usize) -> Vec<f32> {
    // Ordered finite ramp spanning roughly [-248, +248] at N=248144.
    (0..n).map(|i| i as f32 * 0.002 - 248.0).collect()
}

fn gen_descending(n: usize) -> Vec<f32> {
    let mut v = gen_ascending(n);
    v.reverse();
    v
}

fn gen_ties(n: usize) -> Vec<f32> {
    // Every logit identical: maximal tie pressure on slot order.
    vec![2.5f32; n]
}

fn gen_tie_spikes(n: usize) -> Vec<f32> {
    // Flat base with many *equal* maxima: max tie must keep index order.
    let mut v = vec![1.0f32; n];
    let mut i = 0usize;
    while i < n {
        v[i] = 9.0;
        i += 5000;
    }
    v
}

fn gen_sparse_mask(n: usize) -> Vec<f32> {
    // Grammar-style mask: almost everything -Inf, a few finite spikes.
    let mut v = vec![f32::NEG_INFINITY; n];
    for k in 0..48u32 {
        let pos = ((k as usize).wrapping_mul(7919).wrapping_add(13)) % n;
        v[pos] = 5.0 - k as f32 * 0.1;
    }
    v
}

fn gen_nonfinite_mix(n: usize, seed: u64) -> Vec<f32> {
    let mut v = gen_random_uniform(n, seed);
    for i in (0..n).step_by(997) {
        v[i] = f32::NAN;
    }
    for i in (0..n).step_by(1331) {
        v[i] = f32::INFINITY;
    }
    for i in (0..n).step_by(7919) {
        v[i] = f32::NEG_INFINITY;
    }
    v
}

fn gen_signed_zeros(n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        if i % 10_000 == 0 {
            v.push(3.0);
        } else if i % 2 == 0 {
            v.push(0.0);
        } else {
            v.push(-0.0);
        }
    }
    v
}

fn gen_small(len: usize) -> Vec<f32> {
    // Deterministic pseudo-random small input.
    (0..len)
        .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(97) % 1000) as f32) * 0.01 - 5.0)
        .collect()
}

struct BigCase {
    /// e.g. "random_uniform@n248144" — size always labelled per parent note.
    name: String,
    logits: Vec<f32>,
}

fn build_big_cases() -> Vec<BigCase> {
    let mut out = Vec::new();
    for &n in &SIZES {
        let tag = format!("n{}", n);
        out.push(BigCase {
            name: format!("random_uniform@{}", tag),
            logits: gen_random_uniform(n, 0x1234_5678_9ABC_DEF0),
        });
        out.push(BigCase {
            name: format!("ascending@{}", tag),
            logits: gen_ascending(n),
        });
        out.push(BigCase {
            name: format!("descending@{}", tag),
            logits: gen_descending(n),
        });
        out.push(BigCase {
            name: format!("ties@{}", tag),
            logits: gen_ties(n),
        });
        out.push(BigCase {
            name: format!("tie_spikes@{}", tag),
            logits: gen_tie_spikes(n),
        });
        out.push(BigCase {
            name: format!("sparse_mask@{}", tag),
            logits: gen_sparse_mask(n),
        });
        out.push(BigCase {
            name: format!("nonfinite_mix@{}", tag),
            logits: gen_nonfinite_mix(n, 0x0BAD_F00D_CAFE_1234),
        });
        out.push(BigCase {
            name: format!("signed_zeros@{}", tag),
            logits: gen_signed_zeros(n),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Timing helper: warmup + black_box in and out; counters stay outside.
// ---------------------------------------------------------------------------
fn time_ns(reps: u64, mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e9 / (reps as f64)
}

fn cpu_model() -> String {
    // Best-effort host label; never fails the experiment.
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some(name) = rest.split_once(':') {
                    return name.1.trim().to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn opt_num(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.3}", x),
        None => "null".to_string(),
    }
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "result.json".to_string());
    let simd = avx2_available();

    let cases = build_big_cases();

    // End-to-end sampler settings: greedy plus three stochastic corners.
    // Seeds are nonzero so the production reset mapping (0 -> 1) is identity.
    const E2E: &[(f32, f32, u32)] = &[
        (0.0, 0.9, 12345),
        (0.7, 0.8, 777),
        (0.7, 1.0, 424_242),
        (1.0, 0.5, 999),
    ];

    let mut scan_cases = 0u64;
    let mut scan_mismatches = 0u64;
    let mut argmax_cases = 0u64;
    let mut argmax_mismatches = 0u64;
    let mut e2e_cases = 0u64;
    let mut e2e_mismatches = 0u64;
    let mut e2e_state_mismatches = 0u64;
    let mut small_len_cases = 0u64;
    let mut small_len_mismatches = 0u64;

    // ---- parity: scan bits + argmax + production end-to-end (untimed) ----
    for c in &cases {
        let s_ref = scalar_scan(&c.logits);
        if let Some(s_simd) = direct_avx2_scan(&c.logits) {
            scan_cases += 1;
            let mut same = s_ref.max.to_bits() == s_simd.max.to_bits() && s_ref.idx == s_simd.idx;
            if same {
                for i in 0..TOP_K {
                    if s_ref.val[i].to_bits() != s_simd.val[i].to_bits() {
                        same = false;
                        break;
                    }
                }
            }
            if !same {
                scan_mismatches += 1;
                eprintln!("SCAN MISMATCH {}", c.name);
            }
            // Dispatch must agree too.
            let s_disp = dispatch_scan(&c.logits);
            let mut dsame = s_ref.max.to_bits() == s_disp.max.to_bits() && s_ref.idx == s_disp.idx;
            if dsame {
                for i in 0..TOP_K {
                    if s_ref.val[i].to_bits() != s_disp.val[i].to_bits() {
                        dsame = false;
                        break;
                    }
                }
            }
            if !dsame {
                scan_mismatches += 1;
                eprintln!("DISPATCH SCAN MISMATCH {}", c.name);
            }
        }

        let a_ref = scalar_argmax(&c.logits);
        let a_prod = prod::argmax(&c.logits);
        argmax_cases += 1;
        if a_ref != a_prod {
            argmax_mismatches += 1;
            eprintln!(
                "ARGMAX ORACLE MISMATCH {} scalar={} prod={}",
                c.name, a_ref, a_prod
            );
        }
        if let Some(a_simd) = direct_avx2_argmax(&c.logits) {
            argmax_cases += 1;
            if a_simd != a_ref {
                argmax_mismatches += 1;
                eprintln!(
                    "ARGMAX SIMD MISMATCH {} simd={} ref={}",
                    c.name, a_simd, a_ref
                );
            }
        }
        {
            let a_disp = dispatch_argmax(&c.logits);
            argmax_cases += 1;
            if a_disp != a_ref {
                argmax_mismatches += 1;
                eprintln!(
                    "ARGMAX DISPATCH MISMATCH {} disp={} ref={}",
                    c.name, a_disp, a_ref
                );
            }
        }

        // End-to-end: production `sample_top_p` at a fixed RNG seed vs the
        // local full pipeline (AVX2 scan + unchanged scalar remainder) and
        // the pure-scalar local pipeline.
        for &(temp, top_p, seed) in E2E {
            e2e_cases += 1;
            prod::reset_cpu_sampler_rng(seed);
            let t_prod = prod::sample_top_p(&c.logits, temp, top_p);
            let prod_state = prod::sampler_rng_snapshot();
            let mut r1 = LocalRng::new(seed);
            let t_scalar = finish_sample(&c.logits, &s_ref, temp, top_p, &mut r1);
            let scan_for_simd = direct_avx2_scan(&c.logits).unwrap_or_else(|| s_ref.clone());
            let mut r2 = LocalRng::new(seed);
            let t_simd = finish_sample(&c.logits, &scan_for_simd, temp, top_p, &mut r2);
            if t_prod != t_scalar || t_prod != t_simd {
                e2e_mismatches += 1;
                eprintln!(
                    "E2E MISMATCH {} temp={} top_p={} seed={} prod={} scalar={} simd={}",
                    c.name, temp, top_p, seed, t_prod, t_scalar, t_simd
                );
            }
            // RNG-stream parity: the local replica must leave its state exactly
            // where the production global sits, draw-for-draw — including the
            // greedy path, which must not advance either stream.
            if prod_state != r1.0 || prod_state != r2.0 {
                e2e_state_mismatches += 1;
                eprintln!(
                    "E2E RNG STATE MISMATCH {} temp={} top_p={} seed={} prod_state={} scalar_state={} simd_state={}",
                    c.name, temp, top_p, seed, prod_state, r1.0, r2.0
                );
            }
        }
    }

    // ---- parity: small sizes 0..=65 (scan + argmax, incl. adversarial) ----
    for len in 0..=65usize {
        let variants: Vec<Vec<f32>> = vec![
            gen_small(len),
            vec![f32::NEG_INFINITY; len],
            vec![f32::NAN; len],
            vec![0.0f32; len],
        ];
        for v in &variants {
            small_len_cases += 1;
            let s_ref = scalar_scan(v);
            let a_ref = scalar_argmax(v);
            if a_ref != prod::argmax(v) {
                small_len_mismatches += 1;
                eprintln!("SMALL ARGMAX ORACLE MISMATCH len={}", len);
            }
            if let Some(s_simd) = direct_avx2_scan(v) {
                let mut same =
                    s_ref.max.to_bits() == s_simd.max.to_bits() && s_ref.idx == s_simd.idx;
                if same {
                    for i in 0..TOP_K {
                        if s_ref.val[i].to_bits() != s_simd.val[i].to_bits() {
                            same = false;
                            break;
                        }
                    }
                }
                if !same {
                    small_len_mismatches += 1;
                    eprintln!("SMALL SCAN MISMATCH len={}", len);
                }
                if let Some(a_simd) = direct_avx2_argmax(v) {
                    if a_simd != a_ref {
                        small_len_mismatches += 1;
                        eprintln!("SMALL ARGMAX SIMD MISMATCH len={}", len);
                    }
                }
            }
        }
    }
    // Empty slice end-to-end guard (production greedy path on empty logits).
    {
        let empty: Vec<f32> = vec![];
        small_len_cases += 1;
        if scalar_argmax(&empty) != prod::argmax(&empty) {
            small_len_mismatches += 1;
            eprintln!("EMPTY ARGMAX MISMATCH");
        }
    }

    // ---- timing: greedy argmax and top-20 scan separately (black_box) ----
    struct Timed {
        name: String,
        scan_scalar: f64,
        scan_avx2: Option<f64>,
        scan_dispatch: f64,
        argmax_scalar: f64,
        argmax_avx2: Option<f64>,
        argmax_dispatch: f64,
        argmax_prod: f64,
    }
    let mut timed: Vec<Timed> = Vec::new();
    for c in &cases {
        // Warmup (outside the clock).
        black_box(scalar_scan(black_box(c.logits.as_slice())));
        black_box(dispatch_scan(black_box(c.logits.as_slice())));
        black_box(scalar_argmax(black_box(c.logits.as_slice())));
        black_box(dispatch_argmax(black_box(c.logits.as_slice())));
        black_box(prod::argmax(black_box(c.logits.as_slice())));
        if simd {
            black_box(direct_avx2_scan(black_box(c.logits.as_slice())));
            black_box(direct_avx2_argmax(black_box(c.logits.as_slice())));
        }
        let scan_scalar = time_ns(BIG_REPS_SCAN, || {
            let s = scalar_scan(black_box(c.logits.as_slice()));
            black_box(s);
        });
        let scan_dispatch = time_ns(BIG_REPS_SCAN, || {
            let s = dispatch_scan(black_box(c.logits.as_slice()));
            black_box(s);
        });
        let scan_avx2 = if simd {
            Some(time_ns(BIG_REPS_SCAN, || {
                let s = direct_avx2_scan(black_box(c.logits.as_slice()));
                black_box(s);
            }))
        } else {
            None
        };
        let argmax_scalar = time_ns(BIG_REPS_ARGMAX, || {
            let a = scalar_argmax(black_box(c.logits.as_slice()));
            black_box(a);
        });
        let argmax_dispatch = time_ns(BIG_REPS_ARGMAX, || {
            let a = dispatch_argmax(black_box(c.logits.as_slice()));
            black_box(a);
        });
        let argmax_prod = time_ns(BIG_REPS_ARGMAX, || {
            let a = prod::argmax(black_box(c.logits.as_slice()));
            black_box(a);
        });
        let argmax_avx2 = if simd {
            Some(time_ns(BIG_REPS_ARGMAX, || {
                let a = direct_avx2_argmax(black_box(c.logits.as_slice()));
                black_box(a);
            }))
        } else {
            None
        };
        timed.push(Timed {
            name: c.name.clone(),
            scan_scalar,
            scan_avx2,
            scan_dispatch,
            argmax_scalar,
            argmax_avx2,
            argmax_dispatch,
            argmax_prod,
        });
    }

    // ---- timing: small-size overhead probes ----
    const PROBES: [usize; 10] = [0, 1, 7, 8, 9, 16, 20, 32, 64, 65];
    struct SmallTimed {
        len: usize,
        scan_scalar: f64,
        scan_avx2: Option<f64>,
        scan_dispatch: f64,
        argmax_scalar: f64,
        argmax_avx2: Option<f64>,
        argmax_dispatch: f64,
    }
    let mut small_timed: Vec<SmallTimed> = Vec::new();
    let (mut mean_ss, mut mean_sd, mut mean_as, mut mean_ad) = (0.0, 0.0, 0.0, 0.0);
    let (mut mean_sa, mut mean_aa) = (0.0, 0.0);
    let mut mean_n = 0u32;
    // Means over every length 0..=65 (scalar + dispatch always run).
    for len in 0..=65usize {
        let v = gen_small(len);
        mean_ss += time_ns(SMALL_REPS, || {
            let s = scalar_scan(black_box(v.as_slice()));
            black_box(s);
        });
        mean_sd += time_ns(SMALL_REPS, || {
            let s = dispatch_scan(black_box(v.as_slice()));
            black_box(s);
        });
        mean_as += time_ns(SMALL_REPS, || {
            let a = scalar_argmax(black_box(v.as_slice()));
            black_box(a);
        });
        mean_ad += time_ns(SMALL_REPS, || {
            let a = dispatch_argmax(black_box(v.as_slice()));
            black_box(a);
        });
        if simd {
            mean_sa += time_ns(SMALL_REPS, || {
                let s = direct_avx2_scan(black_box(v.as_slice()));
                black_box(s);
            });
            mean_aa += time_ns(SMALL_REPS, || {
                let a = direct_avx2_argmax(black_box(v.as_slice()));
                black_box(a);
            });
        }
        mean_n += 1;
    }
    for &len in &PROBES {
        let v = gen_small(len);
        let scan_scalar = time_ns(SMALL_REPS, || {
            let s = scalar_scan(black_box(v.as_slice()));
            black_box(s);
        });
        let scan_dispatch = time_ns(SMALL_REPS, || {
            let s = dispatch_scan(black_box(v.as_slice()));
            black_box(s);
        });
        let argmax_scalar = time_ns(SMALL_REPS, || {
            let a = scalar_argmax(black_box(v.as_slice()));
            black_box(a);
        });
        let argmax_dispatch = time_ns(SMALL_REPS, || {
            let a = dispatch_argmax(black_box(v.as_slice()));
            black_box(a);
        });
        let (scan_avx2, argmax_avx2) = if simd {
            (
                Some(time_ns(SMALL_REPS, || {
                    let s = direct_avx2_scan(black_box(v.as_slice()));
                    black_box(s);
                })),
                Some(time_ns(SMALL_REPS, || {
                    let a = direct_avx2_argmax(black_box(v.as_slice()));
                    black_box(a);
                })),
            )
        } else {
            (None, None)
        };
        small_timed.push(SmallTimed {
            len,
            scan_scalar,
            scan_avx2,
            scan_dispatch,
            argmax_scalar,
            argmax_avx2,
            argmax_dispatch,
        });
    }

    let total_mismatches = scan_mismatches + argmax_mismatches + e2e_mismatches + e2e_state_mismatches + small_len_mismatches;
    let result = if total_mismatches == 0 { "pass" } else { "FAIL" };

    // ---- JSON evidence (hand-rolled; no extra dependencies) ----
    let mut j = String::new();
    j.push_str("{\n");
    j.push_str("  \"experiment\": \"sampling-simd-avx2-block-rejection\",\n");
    j.push_str(&format!(
        "  \"host\": {{\"arch\": \"{}\", \"avx2_detected\": {}, \"cpu_model\": \"{}\"}},\n",
        std::env::consts::ARCH,
        simd,
        cpu_model().replace('\\', "\\\\").replace('"', "'")
    ));
    j.push_str("  \"production_base\": \"77cd6ce7d\",\n");
    j.push_str("  \"note\": \"inputs are synthetic logits sized to measured vocab 248144 and padded row 248320; sampler scans logits.len()\",\n");
    j.push_str("  \"logit_sizes\": [248144, 248320],\n");
    j.push_str(&format!(
        "  \"paths_exercised\": {{\"scalar\": true, \"avx2\": {}, \"dispatch\": true}},\n",
        simd
    ));
    j.push_str(&format!(
        "  \"parity\": {{\"scan_cases\": {}, \"scan_mismatches\": {}, \"argmax_cases\": {}, \"argmax_mismatches\": {}, \"e2e_cases\": {}, \"e2e_mismatches\": {}, \"e2e_state_mismatches\": {}, \"small_len_cases\": {}, \"small_len_mismatches\": {}}},\n",
        scan_cases,
        scan_mismatches,
        argmax_cases,
        argmax_mismatches,
        e2e_cases,
        e2e_mismatches,
        e2e_state_mismatches,
        small_len_cases,
        small_len_mismatches
    ));
    j.push_str(&format!(
        "  \"reps\": {{\"big_scan\": {}, \"big_argmax\": {}, \"small\": {}}},\n",
        BIG_REPS_SCAN, BIG_REPS_ARGMAX, SMALL_REPS
    ));
    j.push_str("  \"cases\": [\n");
    for (i, t) in timed.iter().enumerate() {
        j.push_str(&format!(
            "    {{\"name\": \"{}\", \"scan_ns_per_call\": {{\"scalar\": {:.3}, \"avx2\": {}, \"dispatch\": {:.3}}}, \"argmax_ns_per_call\": {{\"scalar\": {:.3}, \"avx2\": {}, \"dispatch\": {:.3}, \"production\": {:.3}}}}}{}\n",
            t.name,
            t.scan_scalar,
            opt_num(t.scan_avx2),
            t.scan_dispatch,
            t.argmax_scalar,
            opt_num(t.argmax_avx2),
            t.argmax_dispatch,
            t.argmax_prod,
            if i + 1 == timed.len() { "" } else { "," }
        ));
    }
    j.push_str("  ],\n");
    j.push_str("  \"small_sizes\": {\n");
    j.push_str(&format!(
        "    \"mean_ns_per_call_len0_65\": {{\"scan_scalar\": {:.3}, \"scan_avx2\": {}, \"scan_dispatch\": {:.3}, \"argmax_scalar\": {:.3}, \"argmax_avx2\": {}, \"argmax_dispatch\": {:.3}}},\n",
        mean_ss / (mean_n as f64),
        if simd {
            format!("{:.3}", mean_sa / (mean_n as f64))
        } else {
            "null".to_string()
        },
        mean_sd / (mean_n as f64),
        mean_as / (mean_n as f64),
        if simd {
            format!("{:.3}", mean_aa / (mean_n as f64))
        } else {
            "null".to_string()
        },
        mean_ad / (mean_n as f64)
    ));
    j.push_str("    \"probes\": [\n");
    for (i, s) in small_timed.iter().enumerate() {
        j.push_str(&format!(
            "      {{\"len\": {}, \"scan_ns_per_call\": {{\"scalar\": {:.3}, \"avx2\": {}, \"dispatch\": {:.3}}}, \"argmax_ns_per_call\": {{\"scalar\": {:.3}, \"avx2\": {}, \"dispatch\": {:.3}}}}}{}\n",
            s.len,
            s.scan_scalar,
            opt_num(s.scan_avx2),
            s.scan_dispatch,
            s.argmax_scalar,
            opt_num(s.argmax_avx2),
            s.argmax_dispatch,
            if i + 1 == small_timed.len() { "" } else { "," }
        ));
    }
    j.push_str("    ]\n");
    j.push_str("  },\n");
    j.push_str(&format!("  \"result\": \"{}\"\n", result));
    j.push_str("}\n");

    if let Err(e) = std::fs::write(&out_path, &j) {
        eprintln!("failed to write {}: {}", out_path, e);
        std::process::exit(2);
    }
    println!(
        "sampling-simd: avx2={} scan_cases={} argmax_cases={} e2e_cases={} e2e_state_mismatches={} small_len_cases={} mismatches={} -> {}",
        simd, scan_cases, argmax_cases, e2e_cases, e2e_state_mismatches, small_len_cases, total_mismatches, out_path
    );
    for t in &timed {
        println!(
            "  {:22} scan ns scalar={:10.1} avx2={:>10} disp={:10.1} | argmax ns scalar={:9.1} avx2={:>10} prod={:9.1}",
            t.name,
            t.scan_scalar,
            opt_num(t.scan_avx2),
            t.scan_dispatch,
            t.argmax_scalar,
            opt_num(t.argmax_avx2),
            t.argmax_prod
        );
    }

    if total_mismatches != 0 {
        std::process::exit(1);
    }
}
