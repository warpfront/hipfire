// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! 256-block FWHT rotation matching the device transform:
//! x' = 0.0625 * s2 (.) FWHT256(s1 (.) u)   (forward, u = x/awq)
//! u  = s1 (.) (H/16)(s2 (.) x')            (inverse; H symmetric, H*H = 256 I)

pub fn fwht256(x: &mut [f32]) {
    let n = 256;
    let mut stride = 1;
    while stride < n {
        for base in (0..n).step_by(stride * 2) {
            for j in 0..stride {
                let a = x[base + j];
                let b = x[base + j + stride];
                x[base + j] = a + b;
                x[base + j + stride] = a - b;
            }
        }
        stride *= 2;
    }
}
/// Forward rotation of every 256 block of `x` (len % 256 == 0).
pub fn rotate_fwd(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    for c in x.chunks_mut(256) {
        for i in 0..256 {
            c[i] *= s1[i];
        }
        fwht256(c);
        for i in 0..256 {
            c[i] *= 0.0625 * s2[i];
        }
    }
}
/// Inverse rotation (rotated -> original/awq domain) of every 256 block.
pub fn rotate_inv(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    for c in x.chunks_mut(256) {
        for i in 0..256 {
            c[i] *= s2[i];
        }
        fwht256(c);
        for i in 0..256 {
            c[i] *= 0.0625 * s1[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        }
    }
    fn signs(r: &mut impl FnMut() -> f32) -> Vec<f32> {
        (0..256)
            .map(|_| if r() > 0.0 { 1.0 } else { -1.0 })
            .collect()
    }
    #[test]
    fn inverse_undoes_forward() {
        let mut r = rng(7);
        let s1 = signs(&mut r);
        let s2 = signs(&mut r);
        let u: Vec<f32> = (0..512).map(|_| r()).collect();
        let mut x = u.clone();
        rotate_fwd(&mut x, &s1, &s2);
        rotate_inv(&mut x, &s1, &s2);
        for i in 0..512 {
            assert!((x[i] - u[i]).abs() < 1e-5);
        }
    }
    #[test]
    fn w_times_r_equals_wo_times_ro() {
        // W' r' == W'_o r_o with W'_o rows and r_o both mapped by the inverse rotation.
        let mut r = rng(11);
        let s1 = signs(&mut r);
        let s2 = signs(&mut r);
        let (n, k) = (5, 512);
        let w: Vec<f32> = (0..n * k).map(|_| r()).collect();
        let rp: Vec<f32> = (0..k).map(|_| r()).collect();
        let mut wo = w.clone();
        rotate_inv(&mut wo, &s1, &s2);
        let mut ro = rp.clone();
        rotate_inv(&mut ro, &s1, &s2);
        for i in 0..n {
            let a: f64 = (0..k).map(|j| w[i * k + j] as f64 * rp[j] as f64).sum();
            let b: f64 = (0..k).map(|j| wo[i * k + j] as f64 * ro[j] as f64).sum();
            assert!((a - b).abs() < 1e-4 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}
