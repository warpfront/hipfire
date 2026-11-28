// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware gate (gfx1201) for the Inkling-2 blocked-prefill FA2 attention
//! (`Gpu::inkling2_attn_prefill_fa2`) against an f64 reference on F16 K/V:
//!
//! * SWA and full-causal visits, ragged row counts (1, 37, 70, 300, 512) and
//!   `pos0` off the 16/32/64 grids, so partial query tiles, partial key tiles,
//!   the ring wrap and keys past the 1024-column extent are all exercised;
//! * bar `1e-5` absolute on every head of the sampled rows (the hi/lo FA2
//!   budget of `inkling2_rel_extent_hw`), finite everywhere, bitwise run to run.
//!
//! `#[ignore]`d: needs gfx1201. Run explicitly (through the GPU fabric):
//!
//!   cargo test -p rdna-compute --release --test inkling2_attn_prefill_hw -- --ignored --nocapture

use rdna_compute::inkling2_attention_prefill::Inkling2PrefillAttn;
use rdna_compute::{DType, Gpu};

const H: usize = 24;
const KVH: usize = 8;
const D: usize = 128;
const NREL: usize = 16;
const EXT: usize = 1024;
const BAR: f64 = 1e-5;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * s).collect()
    }
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

struct Ops {
    q: Vec<f32>,
    rel: Vec<f32>,
    sinks: Vec<f32>,
    table: Vec<f32>,
    kf: Vec<Vec<f32>>,
    vf: Vec<Vec<f32>>,
}

/// f64 reference of block row `i`: keys `[max(0, pos - window + 1), pos]`, the
/// zero-score zero-value sink key, relative term zero past the extent.
fn reference(o: &Ops, pos0: usize, window: usize, i: usize) -> Vec<f64> {
    let pos = pos0 + i;
    let lo = if window > 0 {
        (pos + 1).saturating_sub(window)
    } else {
        0
    };
    let mut out = vec![0.0f64; H * D];
    for h in 0..H {
        let kv = h / 3;
        let qh = &o.q[(i * H + h) * D..(i * H + h + 1) * D];
        let rh = &o.rel[(i * H + h) * NREL..(i * H + h + 1) * NREL];
        let sink = o.sinks[i * H + h] as f64;
        let s: Vec<f64> = (lo..=pos)
            .map(|j| {
                let dot: f64 = (0..D)
                    .map(|d| qh[d] as f64 * o.kf[kv][j * D + d] as f64)
                    .sum();
                let dist = pos - j;
                let b: f64 = if dist < EXT {
                    (0..NREL)
                        .map(|r| o.table[r * EXT + dist] as f64 * rh[r] as f64)
                        .sum()
                } else {
                    0.0
                };
                dot / (D as f64).sqrt() + b - sink
            })
            .collect();
        let m = s.iter().copied().fold(0.0f64, f64::max);
        let mut l = (-m).exp();
        let oh = &mut out[h * D..(h + 1) * D];
        for (n, &sj) in s.iter().enumerate() {
            let e = (sj - m).exp();
            l += e;
            for d in 0..D {
                oh[d] += e * o.vf[kv][(lo + n) * D + d] as f64;
            }
        }
        oh.iter_mut().for_each(|x| *x /= l);
    }
    out
}

fn case(gpu: &mut Gpu, rows: usize, pos0: usize, window: usize) {
    let ring = if window > 0 { window } else { 2112 };
    let last = pos0 + rows - 1;
    let mut r = Lcg(0x9e37_79b9 ^ (rows * 131 + pos0 * 7 + window) as u64);
    let mut o = Ops {
        q: r.vec(rows * H * D, 2.0),
        rel: r.vec(rows * H * NREL, 1.0),
        sinks: r.vec(rows * H, 1.0),
        table: r.vec(NREL * EXT, 0.5),
        kf: vec![],
        vf: vec![],
    };
    let (mut kc, mut vc, mut ksb, mut vsb) = (vec![], vec![], vec![], vec![]);
    for _ in 0..KVH {
        let (kb, kr) = f16(&r.vec((last + 1) * D, 1.0));
        let (vb, vr) = f16(&r.vec((last + 1) * D, 1.0));
        let mut kcb = vec![0u8; ring * D * 2];
        let mut vcb = vec![0u8; ring * D * 2];
        let lo = if window > 0 {
            pos0.saturating_sub(window)
        } else {
            0
        };
        for j in lo..pos0 {
            let s = (j % ring) * D * 2;
            kcb[s..s + D * 2].copy_from_slice(&kb[j * D * 2..(j + 1) * D * 2]);
            vcb[s..s + D * 2].copy_from_slice(&vb[j * D * 2..(j + 1) * D * 2]);
        }
        kc.push(gpu.upload_raw(&kcb, &[kcb.len()]).unwrap());
        vc.push(gpu.upload_raw(&vcb, &[vcb.len()]).unwrap());
        ksb.extend_from_slice(&kb[pos0 * D * 2..]);
        vsb.extend_from_slice(&vb[pos0 * D * 2..]);
        o.kf.push(kr);
        o.vf.push(vr);
    }
    let a = Inkling2PrefillAttn {
        q: &gpu.upload_f32(&o.q, &[rows, H, D]).unwrap(),
        rel: &gpu.upload_f32(&o.rel, &[rows, H, NREL]).unwrap(),
        sinks: &gpu.upload_f32(&o.sinks, &[rows, H]).unwrap(),
        table: &gpu.upload_f32(&o.table, &[NREL, EXT]).unwrap(),
        k_cache: &kc,
        v_cache: &vc,
        k_stage: &gpu.upload_raw(&ksb, &[ksb.len()]).unwrap(),
        v_stage: &gpu.upload_raw(&vsb, &[vsb.len()]).unwrap(),
        out: &gpu.alloc_tensor(&[rows, H, D], DType::F32).unwrap(),
        rows,
        pos0,
        window,
        ring_rows: ring,
        n_extent: EXT,
    };
    gpu.inkling2_attn_prefill_fa2(&a).unwrap();
    let got = gpu.download_f32(a.out).unwrap();
    gpu.inkling2_attn_prefill_fa2(&a).unwrap();
    let again = gpu.download_f32(a.out).unwrap();
    assert!(
        got.iter()
            .zip(&again)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "rows {rows} pos0 {pos0} window {window}: not bitwise run to run"
    );
    assert!(got.iter().all(|x| x.is_finite()), "non-finite output");

    let mut rows_s: Vec<usize> = [0, 1, 15, 16, 31, 32, 63, 64, 65, 127, 255, 300, 447, 511]
        .into_iter()
        .filter(|&i| i < rows)
        .collect();
    rows_s.push(rows - 1);
    rows_s.dedup();
    let mut worst = 0.0f64;
    for &i in &rows_s {
        let want = reference(&o, pos0, window, i);
        for (k, w) in want.iter().enumerate() {
            worst = worst.max((got[i * H * D + k] as f64 - w).abs());
        }
    }
    eprintln!(
        "rows {rows} pos0 {pos0} window {window}: max |fa2 - f64| {worst:.2e} (bar {BAR:.0e})"
    );
    assert!(
        worst <= BAR,
        "rows {rows} pos0 {pos0} window {window}: {worst:.2e} > {BAR:.0e}"
    );
    for t in kc.into_iter().chain(vc) {
        gpu.free_tensor(t).unwrap();
    }
}

#[test]
#[ignore = "requires gfx1201"]
fn prefill_fa2_matches_f64_reference_gfx1201() {
    let mut gpu = Gpu::init().expect("GPU required");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: {} is not gfx1201", gpu.arch);
        return;
    }
    // (rows, pos0, window): SWA blocks incl. the ring wrap, full causal past the extent,
    // ragged tiles and pos0 off every tile grid.
    for (rows, pos0, window) in [
        (512, 1024, 1024),
        (512, 0, 1024),
        (37, 1000, 1024),
        (300, 1501, 1024),
        (1, 0, 1024),
        (512, 1536, 0),
        (70, 2040, 0),
        (300, 1, 0),
        (1, 777, 0),
    ] {
        case(&mut gpu, rows, pos0, window);
    }
}
