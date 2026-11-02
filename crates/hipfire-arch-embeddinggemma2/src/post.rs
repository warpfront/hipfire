// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Post-projection host math: sentence-transformers modules 1_Pooling (mean)
//! and 2_Normalize (L2), plus Matryoshka truncation — pure f32, unit-testable
//! without a GPU.
//!
//! Pipeline order (matching `modules.json`): Transformer (which ends at the
//! 768-d `embedding_projection`) → mean pooling over all tokens → L2
//! normalize. Matryoshka use truncates the 768-d vector to 512/256/128 and
//! re-normalizes — implemented as pool → truncate → normalize so one code
//! path serves both.

/// Mean-pool the `[n, dim]` row-major token matrix into one `[dim]` vector.
/// `n == 0` returns zeros (the encode paths refuse empty inputs before this).
pub fn mean_pool_rows(m: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    if n == 0 || m.len() < n * dim {
        return out;
    }
    for row in 0..n {
        let r = &m[row * dim..(row + 1) * dim];
        for (o, x) in out.iter_mut().zip(r) {
            *o += x;
        }
    }
    let inv = 1.0 / n as f32;
    for o in out.iter_mut() {
        *o *= inv;
    }
    out
}

/// Matryoshka truncation: keep the first `dims` components. `dims >= dim`
/// or `dims == 0` returns the vector unchanged (a caller asking for more
/// than the model has gets the model's full width; 0 is "no opinion").
pub fn mrl_truncate(v: &[f32], dims: usize) -> &[f32] {
    if dims == 0 || dims >= v.len() {
        v
    } else {
        &v[..dims]
    }
}

/// L2-normalize one vector in place. Zero vectors pass through as zero; a
/// non-finite norm zeroes the vector rather than shipping NaN downstream
/// (same contract as the qwen35-embed `l2_normalize_rows`).
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum();
    if norm > 0.0 && norm.is_finite() {
        let inv = 1.0 / norm.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    } else if !norm.is_finite() {
        v.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// L2-normalize each row of a `[n, dim]` row-major matrix in place.
pub fn l2_normalize_rows(m: &mut [f32], dim: usize) {
    for row in m.chunks_exact_mut(dim) {
        l2_normalize(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_rows() {
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
        assert_eq!(mean_pool_rows(&m, 3, 2), vec![3.0, 4.0]);
        assert_eq!(mean_pool_rows(&m[..2], 1, 2), vec![1.0, 2.0]);
        // Guarded degenerate shapes return zeros, not garbage.
        assert_eq!(mean_pool_rows(&m, 0, 2), vec![0.0, 0.0]);
        assert_eq!(mean_pool_rows(&m[..3], 2, 2), vec![0.0, 0.0]);
    }

    #[test]
    fn truncate_then_normalize_is_matryoshka() {
        // A 768-scale vector faked at 8: truncation keeps the prefix, and
        // the truncated vector re-normalizes to unit length.
        let v: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let t = mrl_truncate(&v, 4).to_vec();
        assert_eq!(t, vec![1.0, 2.0, 3.0, 4.0]);
        let mut t2 = t.clone();
        l2_normalize(&mut t2);
        let norm: f32 = t2.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-6);
        // dims past the width is identity; 0 means "no opinion".
        assert_eq!(mrl_truncate(&v, 8).len(), 8);
        assert_eq!(mrl_truncate(&v, 999).len(), 8);
        assert_eq!(mrl_truncate(&v, 0).len(), 8);
    }

    #[test]
    fn normalize_handles_zero_and_nonfinite() {
        let mut z = vec![0.0f32; 3];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0, 0.0]);
        let mut bad = vec![f32::NAN, 1.0];
        l2_normalize(&mut bad);
        assert!(bad.iter().all(|x| *x == 0.0));
        let mut m = vec![3.0, 4.0, 1.0, 0.0];
        l2_normalize_rows(&mut m, 2);
        assert!((m[0] - 0.6).abs() < 1e-6 && (m[1] - 0.8).abs() < 1e-6);
        assert_eq!(m[2..], vec![1.0, 0.0]);
    }
}
