// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware gate (gfx1201) for the Inkling-2 relative-bias extent rule on every GPU
//! attention path: past the 1024-column table the relative term is ZERO (upstream
//! `bin >= 0 && bin < n_bins ? table[bin] : 0`, the HF reference and
//! `forward.rs::rel_bias_for_dist`), never a clamp to column 1023.
//!
//! * the scalar gather (`inkling2_rel_bias_gather_f32`, which feeds both the scalar and
//!   the FA2 decode attention) at distances 1022/1023/1024/1025/2048 against the CPU
//!   reference `inkling2_rel_bias_f32`: bitwise `-sink` past the extent;
//! * the blocked FA2 prefill's in-tile bias on a global (full-causal) visit, query
//!   positions 1018..=1027 (so 1023/1024/1025 and key distances up to 1027), against an
//!   f64 attention reference built on that CPU bias reference. The table's column 1023
//!   is made large, so a clamp would move the rows at `pos >= 1024` far outside the bar
//!   (asserted on the reference itself).
//!
//! `#[ignore]`d: needs gfx1201. Run explicitly (through the GPU fabric):
//!
//!   cargo test -p rdna-compute --release --test inkling2_rel_extent_hw -- --ignored --nocapture

use rdna_compute::inkling2_attention_prefill::Inkling2PrefillAttn;
use rdna_compute::inkling2_rel_bias::inkling2_rel_bias_f32;
use rdna_compute::{DType, Gpu};

const H: usize = 24;
const KVH: usize = 8;
const D: usize = 128;
const NREL: usize = 16;
const EXT: usize = 1024;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

fn gfx1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("GPU required");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: {} is not gfx1201", gpu.arch);
        return None;
    }
    Some(gpu)
}

/// F16 bytes of `v` and the F16-rounded values.
fn f16(v: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let bits: Vec<u16> = v
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    (
        bits.iter().flat_map(|b| b.to_le_bytes()).collect(),
        bits.iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect(),
    )
}

#[test]
#[ignore = "requires gfx1201"]
fn gather_zero_term_past_extent_gfx1201() {
    let Some(mut gpu) = gfx1201() else { return };
    let mut r = Lcg(0x1023_1024_1025);
    let rel = r.vec(H * NREL, 1.0);
    let table = r.vec(NREL * EXT, 0.5);
    let sink = r.vec(H, 1.0);
    let dists = [0usize, 1022, 1023, 1024, 1025, 2048];
    let n_kv = dists.len();
    let d_rel = gpu.upload_f32(&rel, &[H, NREL]).unwrap();
    let d_table = gpu.upload_f32(&table, &[NREL, EXT]).unwrap();
    let d_sink = gpu.upload_f32(&sink, &[H]).unwrap();
    let dist_bytes: Vec<u8> = dists
        .iter()
        .flat_map(|&d| (d as i32).to_le_bytes())
        .collect();
    let d_dist = gpu.upload_raw(&dist_bytes, &[dist_bytes.len()]).unwrap();
    let out = gpu.alloc_tensor(&[H, n_kv], DType::F32).unwrap();
    gpu.inkling2_rel_bias_gather_f32(&d_rel, &d_table, &d_sink, &d_dist, &out, NREL, H, EXT, n_kv)
        .unwrap();
    let got = gpu.download_f32(&out).unwrap();
    let want = inkling2_rel_bias_f32(&rel, &table, &sink, &dists, NREL, H, EXT);
    for h in 0..H {
        for (k, &d) in dists.iter().enumerate() {
            let (g, w) = (got[h * n_kv + k], want[h * n_kv + k]);
            if d >= EXT {
                assert_eq!(
                    g.to_bits(),
                    (-sink[h]).to_bits(),
                    "head {h} d {d}: {g} != -sink"
                );
            }
            assert!((g - w).abs() <= 1e-5, "head {h} d {d}: gpu {g} cpu {w}");
        }
        // The column-1023 term is non-trivial, so a clamp would differ at d >= 1024.
        assert!(
            (want[h * n_kv + 2] - want[h * n_kv + 3]).abs() > 1e-4,
            "head {h}: degenerate table"
        );
    }
}

/// f64 reference of one query row: keys `0..=pos`, the synthetic zero-score zero-value
/// sink key, scores `q.k / sqrt(128) + bias` with `bias = rel_bias(h, pos - j)`.
fn attend_ref(
    q: &[f32],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    bias: impl Fn(usize, usize) -> f32,
    pos: usize,
    out: &mut [f64],
) {
    for h in 0..H {
        let kv = h / 3;
        let qh = &q[h * D..(h + 1) * D];
        let s: Vec<f64> = (0..=pos)
            .map(|j| {
                let dot: f64 = (0..D).map(|d| qh[d] as f64 * k[kv][j * D + d] as f64).sum();
                dot / (D as f64).sqrt() + bias(h, pos - j) as f64
            })
            .collect();
        let m = s.iter().copied().fold(0.0f64, f64::max);
        let mut l = (-m).exp();
        let o = &mut out[h * D..(h + 1) * D];
        o.fill(0.0);
        for (j, &sj) in s.iter().enumerate() {
            let e = (sj - m).exp();
            l += e;
            for d in 0..D {
                o[d] += e * v[kv][j * D + d] as f64;
            }
        }
        for x in o.iter_mut() {
            *x /= l;
        }
    }
}

#[test]
#[ignore = "requires gfx1201"]
fn prefill_fa2_zero_term_past_extent_global_visit() {
    let Some(mut gpu) = gfx1201() else { return };
    let (pos0, rows) = (1018usize, 10usize);
    let last = pos0 + rows - 1;
    let ring = 1088usize;
    let mut r = Lcg(0x5eed_1024);
    // Table: small random, column 1023 large and positive against positive rel rows,
    // so the d = 1023 key is strongly preferred and a clamp (d >= 1024 -> column 1023)
    // would hand the same weight to the keys past the extent.
    let mut table = r.vec(NREL * EXT, 0.05);
    for rr in 0..NREL {
        table[rr * EXT + EXT - 1] = 0.4;
    }
    let q = r.vec(rows * H * D, 0.5);
    let rel: Vec<f32> = (0..rows * H * NREL).map(|_| 0.4 + 0.2 * r.unit()).collect();
    let sinks = r.vec(rows * H, 0.5);
    // K/V for every key 0..=last per KV head, F16-rounded for the reference.
    let mut kf = Vec::new();
    let mut vf = Vec::new();
    let (mut k_cache, mut v_cache) = (Vec::new(), Vec::new());
    let mut k_stage = Vec::new();
    let mut v_stage = Vec::new();
    for _ in 0..KVH {
        let (kb, kr) = f16(&r.vec((last + 1) * D, 1.0));
        let (vb, vr) = f16(&r.vec((last + 1) * D, 1.0));
        let split = pos0 * D * 2;
        let mut kc = kb[..split].to_vec();
        kc.resize(ring * D * 2, 0);
        let mut vc = vb[..split].to_vec();
        vc.resize(ring * D * 2, 0);
        k_cache.push(gpu.upload_raw(&kc, &[kc.len()]).unwrap());
        v_cache.push(gpu.upload_raw(&vc, &[vc.len()]).unwrap());
        k_stage.extend_from_slice(&kb[split..]);
        v_stage.extend_from_slice(&vb[split..]);
        kf.push(kr);
        vf.push(vr);
    }
    let a = Inkling2PrefillAttn {
        q: &gpu.upload_f32(&q, &[rows, H, D]).unwrap(),
        rel: &gpu.upload_f32(&rel, &[rows, H, NREL]).unwrap(),
        sinks: &gpu.upload_f32(&sinks, &[rows, H]).unwrap(),
        table: &gpu.upload_f32(&table, &[NREL, EXT]).unwrap(),
        k_cache: &k_cache,
        v_cache: &v_cache,
        k_stage: &gpu.upload_raw(&k_stage, &[k_stage.len()]).unwrap(),
        v_stage: &gpu.upload_raw(&v_stage, &[v_stage.len()]).unwrap(),
        out: &gpu.alloc_tensor(&[rows, H, D], DType::F32).unwrap(),
        rows,
        pos0,
        window: 0,
        ring_rows: ring,
        n_extent: EXT,
    };
    gpu.inkling2_attn_prefill_fa2(&a).unwrap();
    let got = gpu.download_f32(a.out).unwrap();

    // hi/lo F16 Q and P keep the kernel within ~2e-7 of the f64 reference on F16 K/V.
    const BAR: f64 = 1e-5;
    let mut worst = 0.0f64;
    for i in 0..rows {
        let pos = pos0 + i;
        let qi = &q[i * H * D..(i + 1) * H * D];
        let rel_i = &rel[i * H * NREL..(i + 1) * H * NREL];
        let sink_i = &sinks[i * H..(i + 1) * H];
        let dists: Vec<usize> = (0..=pos).map(|j| pos - j).collect();
        // CPU reference bias [H, pos + 1] (zero term past the extent).
        let b = inkling2_rel_bias_f32(rel_i, &table, sink_i, &dists, NREL, H, EXT);
        let n_kv = pos + 1;
        let mut want = vec![0.0f64; H * D];
        attend_ref(qi, &kf, &vf, |h, d| b[h * n_kv + (pos - d)], pos, &mut want);
        // The clamp rule's output, to prove the bar discriminates.
        let clamp = |h: usize, d: usize| {
            let c = d.min(EXT - 1);
            (0..NREL)
                .map(|rr| table[rr * EXT + c] * rel_i[h * NREL + rr])
                .sum::<f32>()
                - sink_i[h]
        };
        let mut clamped = vec![0.0f64; H * D];
        attend_ref(qi, &kf, &vf, clamp, pos, &mut clamped);
        let sep = want
            .iter()
            .zip(&clamped)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if pos >= EXT {
            assert!(
                sep > 20.0 * BAR,
                "pos {pos}: clamp vs zero only {sep:.2e} apart"
            );
        } else {
            assert_eq!(sep, 0.0, "pos {pos}: no key is past the extent");
        }
        let err = got[i * H * D..(i + 1) * H * D]
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (g as f64 - w).abs())
            .fold(0.0, f64::max);
        eprintln!("pos {pos}: max |gpu - ref| {err:.2e}, clamp separation {sep:.2e}");
        assert!(
            err <= BAR,
            "pos {pos}: prefill FA2 vs zero-term reference {err:.2e} > {BAR:.0e}"
        );
        worst = worst.max(err);
    }
    eprintln!("prefill FA2 global visit, positions {pos0}..={last}: worst {worst:.2e}");
}
