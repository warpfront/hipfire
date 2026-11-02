// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SIMD dispatch and vector-vs-scalar agreement.

use super::*;
use crate::gemv::{
    gemm_with_simd, gemv_shared_sourced, gemv_with_simd, row_bytes, tensor_bytes, SharedJobSource,
};
use crate::quant::{decode_planar_row, CpuQuant, InputRotation};
use crate::testfix;

/// The dispatch predicate is a pure function of `(available, requested)`: a
/// forced request must never turn into an illegal instruction on hardware that
/// lacks the feature, and `None` must follow detection. This is the test that
/// runs (and covers the branch) on non-AVX2 CI runners.
#[test]
fn dispatch_predicate_is_forced_only_where_supported() {
    assert!(use_avx2(true, None), "detected -> on");
    assert!(!use_avx2(false, None), "undetected -> off");
    assert!(
        !use_avx2(true, Some(false)),
        "forced off wins over detection"
    );
    assert!(use_avx2(true, Some(true)), "forced on, supported");
    assert!(
        !use_avx2(false, Some(true)),
        "forced on, unsupported: must fall back rather than fault"
    );
}

/// Every format this build has a kernel for, paired with whether that kernel
/// widens `fp16` metadata (and therefore needs F16C on top of AVX2).
///
/// Kept in step with `simd::features_for` by
/// [`row_dot_enabled_matches_each_kernels_feature_gate`].
fn kernel_formats() -> Vec<(CpuQuant, bool)> {
    vec![
        // Flat f32 header.
        (CpuQuant::Mq4G256, false),
        (CpuQuant::Hfq4G256, false),
        (CpuQuant::Hfq6G256, false),
        (CpuQuant::Mq6G256, false),
        (CpuQuant::Mq5G256, false),
        (CpuQuant::Hfq3G256, false),
        (CpuQuant::Mq3G256, false),
        (CpuQuant::Hfq2G256, false),
        (CpuQuant::Mq2G256, false),
        (CpuQuant::Hfq4G128, false),
        (CpuQuant::Hfq3G128, false),
        (CpuQuant::Hfq2G128, false),
        // fp16 header.
        (CpuQuant::Mq4CG256, true),
        (CpuQuant::Tq2G128, true),
        (CpuQuant::Bq1G128, true),
        (CpuQuant::Mfp4G32E8, true),
        // Per-128 fp16 header (the V2 family).
        (CpuQuant::Mq4G256V2, true),
        (CpuQuant::Mq6G256V2, true),
        (CpuQuant::Mq5G256V2, true),
        (CpuQuant::Mq3G256V2, true),
        (CpuQuant::Mq2G256V2, true),
        (CpuQuant::Mq4G128V2, true),
        // Planar slab formats: codes plane + f16 scale plane (qt 131/132/135/136).
        (CpuQuant::Mq3I4G128R256, true),
        (CpuQuant::Mq3I4G128R128, true),
        (CpuQuant::E8pI4G128R256, true),
        (CpuQuant::E8pI4G128R128, true),
        // Per-group fp16 codebook.
        (CpuQuant::Mq2G256Lloyd, true),
        (CpuQuant::Mq2G256LloydU, true),
        (CpuQuant::Mq3G256Lloyd, true),
        (CpuQuant::Mq4G256Lloyd, true),
        // Element formats.
        (CpuQuant::F16, true),
        (CpuQuant::Bf16, false),
        (CpuQuant::F32, false),
        (CpuQuant::Q8F16, true),
    ]
}

/// Every format the crate can name, taken from its own qt map rather than from
/// a hand-written list — `from_quant_type` over the whole `u8` space.
fn all_formats() -> Vec<CpuQuant> {
    let mut out: Vec<CpuQuant> = Vec::new();
    for qt in 0u8..=u8::MAX {
        if let Some(q) = CpuQuant::from_quant_type(qt) {
            if !out.contains(&q) {
                out.push(q);
            }
        }
    }
    out
}

/// The kernel table must name every CPU-decodable format, and nothing else.
///
/// This is what keeps [`kernel_formats`] honest as formats are ported: a
/// format that gains a kernel but not a row here shows up as a
/// `row_dot_avx2`-without-a-kernel miss in
/// [`avx2_and_scalar_agree_within_tolerance`] *and* as a gap here, and a stale
/// row for a format the crate no longer decodes fails the reverse check.
#[test]
fn every_cpu_decodable_format_has_a_kernel() {
    let all = all_formats();
    let listed: Vec<CpuQuant> = kernel_formats().iter().map(|(q, _)| *q).collect();
    for q in &all {
        assert!(
            listed.contains(q),
            "{q:?} decodes on the CPU but has no kernel row in this test — add it \
             (and its kernel) or explain why it stays scalar"
        );
    }
    assert_eq!(
        all.len(),
        listed.len(),
        "kernel_formats lists a format the qt map does not name"
    );
    for q in &listed {
        assert!(
            all.contains(q),
            "{q:?} has a kernel row but is not reachable through from_quant_type"
        );
    }
}

/// The per-format dispatch is a pure function of `(format, available,
/// requested)`: a format's gate is exactly the features its kernel needs, a
/// forced `Some(true)` must fall back to scalar — not fault — on hardware
/// without them, and a forced `Some(false)` must always lose. The logged kernel
/// name ([`row_dot_kernel`]) is the unforced decision, spelled by those
/// features.
#[test]
fn row_dot_enabled_matches_each_kernels_feature_gate() {
    for (q, needs_f16c) in kernel_formats() {
        let available = if needs_f16c {
            avx2_f16c_available()
        } else {
            avx2_available()
        };
        assert_eq!(row_dot_enabled(q, None), available, "{q:?}: detection");
        assert_eq!(
            row_dot_enabled(q, Some(true)),
            available,
            "{q:?}: forced on"
        );
        assert!(!row_dot_enabled(q, Some(false)), "{q:?}: forced off");
        let expected = match (available, needs_f16c) {
            (false, _) => "scalar",
            (true, true) => "avx2-fma-f16c",
            (true, false) => "avx2-fma",
        };
        assert_eq!(row_dot_kernel(q), expected, "{q:?}: logged kernel name");
    }
}

/// The shipped fixture headers are exact powers of two (`0.0625`, `-0.25`, …),
/// which makes every decoded value, product and partial sum exactly
/// representable — the two summation orders then agree *bit for bit* and the
/// relative bound below would never be exercised. Real quantized weights carry
/// arbitrary fp16/f32 headers, so this rewrites each group's header with
/// awkward (non-power-of-two) values before the two paths are compared.
fn awkward_headers(q: CpuQuant, packed: &mut [u8], m: usize, k: usize) {
    let (ge, gb) = (q.group_elems(), q.group_bytes());
    let groups = k / ge;
    // f32: 0.0313, -0.4921. fp16: 0.031311, -0.122986, 0.270996, -0.088684.
    const F16X4: [u16; 4] = [0x2802, 0xafdf, 0x3456, 0xadad];
    if q == CpuQuant::Mfp4G32E8 {
        let rb = row_bytes(q, k);
        for row in 0..m {
            packed[row * rb..row * rb + 2].copy_from_slice(&F16X4[row % F16X4.len()].to_le_bytes());
        }
        return;
    }
    if q.is_planar() {
        // The scales are their own plane after all `m` code rows: one f16 per
        // (row, group), at `codes_bytes + 2 * (row * groups + group)`. Clear
        // sign bits keep them finite and positive, as the writer guarantees.
        let scales_at = m * row_bytes(q, k);
        for i in 0..m * groups {
            let bits = AWKWARD_F16[i % AWKWARD_F16.len()] & 0x7fff;
            packed[scales_at + 2 * i..scales_at + 2 * i + 2].copy_from_slice(&bits.to_le_bytes());
        }
        return;
    }
    for row in 0..m {
        for g in 0..groups {
            let at = (row * groups + g) * gb;
            match q {
                CpuQuant::Mq4G256
                | CpuQuant::Hfq4G256
                | CpuQuant::Hfq6G256
                | CpuQuant::Mq6G256
                | CpuQuant::Mq5G256
                | CpuQuant::Hfq3G256
                | CpuQuant::Mq3G256
                | CpuQuant::Hfq2G256
                | CpuQuant::Mq2G256
                | CpuQuant::Hfq4G128
                | CpuQuant::Hfq3G128
                | CpuQuant::Hfq2G128 => {
                    packed[at..at + 4].copy_from_slice(&0.0313f32.to_le_bytes());
                    packed[at + 4..at + 8].copy_from_slice(&(-0.4921f32).to_le_bytes());
                }
                CpuQuant::Mq4CG256 | CpuQuant::Tq2G128 | CpuQuant::Bq1G128 => {
                    packed[at..at + 2].copy_from_slice(&F16X4[0].to_le_bytes());
                    if q == CpuQuant::Mq4CG256 {
                        packed[at + 2..at + 4].copy_from_slice(&F16X4[1].to_le_bytes());
                    }
                }
                CpuQuant::Mq4G128V2 => {
                    // fp16 [scale][zero] at the group's start.
                    packed[at..at + 2].copy_from_slice(&F16X4[0].to_le_bytes());
                    packed[at + 2..at + 4].copy_from_slice(&F16X4[1].to_le_bytes());
                }
                CpuQuant::Mq4G256V2
                | CpuQuant::Mq6G256V2
                | CpuQuant::Mq5G256V2
                | CpuQuant::Mq2G256V2
                | CpuQuant::Mq3G256V2 => {
                    for (i, bits) in F16X4.iter().enumerate() {
                        packed[at + 2 * i..at + 2 * i + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                // The Lloyd tier's header *is* the decode, so it is the
                // codebook that has to carry awkward, pairwise-distinct values:
                // the fixture's books are powers of two, and qt 30's repeats an
                // eight-entry book into sixteen lanes, which would hide an
                // off-by-eight table error.
                CpuQuant::Mq2G256Lloyd | CpuQuant::Mq2G256LloydU => write_cb(packed, at, 4),
                CpuQuant::Mq3G256Lloyd => write_cb(packed, at, 8),
                CpuQuant::Mq4G256Lloyd => write_cb(packed, at, 16),
                // Element formats: the fixture's weights are small powers of
                // two (`0, 1, -1, 2, -2, 3, -3, 5`), so every product and
                // partial sum would be exact and the tolerance would never be
                // exercised. Rewrite the group's elements — and, for Q8F16,
                // the block's scale, which is the only inexact part there.
                CpuQuant::F16 | CpuQuant::Bf16 | CpuQuant::F32 => {
                    for i in 0..ge {
                        match q {
                            CpuQuant::F16 => packed[at + 2 * i..at + 2 * i + 2]
                                .copy_from_slice(&AWKWARD_F16[i % 8].to_le_bytes()),
                            CpuQuant::Bf16 => packed[at + 2 * i..at + 2 * i + 2]
                                .copy_from_slice(&AWKWARD_BF16[i % 8].to_le_bytes()),
                            _ => packed[at + 4 * i..at + 4 * i + 4]
                                .copy_from_slice(&AWKWARD_F32[i % 8].to_le_bytes()),
                        }
                    }
                }
                CpuQuant::Q8F16 => {
                    packed[at..at + 2].copy_from_slice(&AWKWARD_F16[0].to_le_bytes())
                }
                CpuQuant::Mfp4G32E8 => unreachable!("row-header format handled above"),
                CpuQuant::Mq3I4G128R256
                | CpuQuant::Mq3I4G128R128
                | CpuQuant::E8pI4G128R256
                | CpuQuant::E8pI4G128R128 => unreachable!("planar scale plane handled above"),
            }
        }
    }
}

/// Eight `fp16` patterns with distinct magnitudes, alternating signs and
/// non-power-of-two mantissas: 0.0313, -0.123, 0.271, -0.0887, 16.7, -16.7,
/// 0.0172, -0.0172.
const AWKWARD_F16: [u16; 8] = [
    0x2802, 0xafdf, 0x3456, 0xadad, 0x4c2c, 0xcc2c, 0x2467, 0xa467,
];

/// The same eight magnitudes in `bf16`: 1.04, -1.04, 3.08, -3.08, 0.26, -0.26,
/// 12.3, -12.3.
const AWKWARD_BF16: [u16; 8] = [
    0x3f85, 0xbf85, 0x4045, 0xc045, 0x3e85, 0xbe85, 0x4145, 0xc145,
];

/// And in `f32` — the exact decimal forms of the four magnitudes above.
const AWKWARD_F32: [f32; 8] = [
    0.031311, -0.122986, 0.270996, -0.088684, 16.6875, -16.6875, 0.0171966, -0.0171966,
];

/// `n` `fp16` codebook entries at `at`: distinct values across four exponent
/// classes, and non-power-of-two mantissas wherever the fixture's own tables
/// would otherwise be exact.
fn write_cb(packed: &mut [u8], at: usize, n: usize) {
    /// 0.0625, -0.0708, 0.0876, -0.1073, 0.125, -0.1416, 0.1753, -0.2146,
    /// 0.25, -0.2832, 0.3506, -0.4292, 0.5, -0.5664, 0.7012, -0.8584 — an
    /// alternating sign and a distinct magnitude per entry, so a wrong index, a
    /// dropped sign or a full-table shift all show up.
    const CB16: [u16; 16] = [
        0x2c00, 0xac89, 0x2d9b, 0xaedd, 0x3000, 0xb089, 0x319b, 0xb2dd, 0x3400, 0xb489, 0x359b,
        0xb6dd, 0x3800, 0xb889, 0x399b, 0xbadd,
    ];
    for (i, bits) in CB16[..n].iter().enumerate() {
        packed[at + 2 * i..at + 2 * i + 2].copy_from_slice(&bits.to_le_bytes());
    }
}

/// The AVX2 kernels and the scalar reference are different summations of the
/// same products, so they are compared on a *relative* tolerance, not for
/// equality. `1e-4` bounds f32 accumulation noise, not a decode error: the two
/// paths sum the same values in different orders (eight lanes plus a horizontal
/// sum versus one running scalar), and a `k = 4096` row whose result is small
/// relative to its terms amplifies that difference even though both paths are
/// exact per operation. Measured across every format at `k <= 12288` the worst
/// case is `2.3e-5` (`F32`, whose fixture carries the widest magnitudes). A
/// real decode error is orders of magnitude larger — dropping a codebook table
/// half lands at `7.1e-1`, moving a payload offset at `3.5e+1` — so this still
/// catches what it exists to catch.
///
/// The comparison cannot pass vacuously for want of a kernel: the dispatcher is
/// exhaustive over [`CpuQuant`] (a format without a kernel does not compile) and
/// the gate this test skips on — the runner's CPU features — is asserted per
/// format by [`row_dot_enabled_matches_each_kernels_feature_gate`].
///
/// The fixture gives each group distinct, non-power-of-two header values (or
/// codebook entries, or elements), so mixing up a V2 group's 128-element halves
/// — or a header's scale/zero order, or an index's table half — is a difference
/// far above the bound, not a rounding difference.
#[test]
fn avx2_and_scalar_agree_within_tolerance() {
    for (q, needs_f16c) in kernel_formats() {
        if !row_dot_enabled(q, Some(true)) {
            let missing = if needs_f16c { "F16C" } else { "AVX2" };
            eprintln!("skip {q:?}: {missing} absent on this runner");
            continue;
        }
        for (m, k) in [(1usize, 256usize), (3, 512), (2, 4096), (1, 12288)] {
            let mut packed = testfix::weight_bytes(q, m, k);
            awkward_headers(q, &mut packed, m, k);
            let x: Vec<f32> = (0..k)
                .map(|i| ((i as u64 * 2654435761) % 4096) as f32 * 0.001 - 2.0)
                .collect();
            let mut simd = vec![0.0f32; m];
            let mut scalar = vec![0.0f32; m];
            gemv_with_simd(q, &packed, m, k, &x, &mut simd, Some(true));
            gemv_with_simd(q, &packed, m, k, &x, &mut scalar, Some(false));
            let floor = cancellation_floor(q, &packed, m, k, &x);
            for row in 0..m {
                let scale = scalar[row].abs().max(1e-6).max(floor[row]);
                let rel = (simd[row] - scalar[row]).abs() / scale;
                assert!(
                    rel <= 1e-4,
                    "{q:?} {m}x{k} row {row}: simd {} vs scalar {} (rel {rel:.3e})",
                    simd[row],
                    scalar[row]
                );
            }
        }
    }
}

/// Per-row slack for rows whose dot is a random-sign sum that cancels almost
/// completely: `0.05 * sqrt(Σ (w·x)²)` for the planar formats, `0` for the rest
/// (their fixtures keep the original normalisation).
///
/// The planar fixtures use uniformly random codes, so a row's value is a
/// zero-mean sum whose magnitude `|S|` is occasionally far below its terms'
/// scale `σ`, while the f32 difference between two summation orders stays near
/// `5e-7·σ`. Normalising by `|S|` alone then fails a few rows per thousand for
/// reasons that are not decode errors. A decode error is a whole weight off
/// (`~σ/sqrt(k)` ≫ `1e-5·σ`), so this floor still catches it.
fn cancellation_floor(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32]) -> Vec<f32> {
    if !q.is_planar() {
        return vec![0.0; m];
    }
    let (rb, spr) = (row_bytes(q, k), 2 * (k / 128));
    let mut weights = vec![0.0f32; k];
    (0..m)
        .map(|r| {
            let codes = &packed[r * rb..(r + 1) * rb];
            let at = m * rb + r * spr;
            decode_planar_row(q, codes, &packed[at..at + spr], k, &mut weights);
            let ss: f64 = weights
                .iter()
                .zip(x)
                .map(|(&w, &xi)| (w as f64 * xi as f64).powi(2))
                .sum();
            (0.05 * ss.sqrt()) as f32
        })
        .collect()
}

/// The forced-scalar path must be *exactly* `gemv`'s historical arithmetic, so
/// the S1 expectation tables and the GPU parity numbers keep meaning what they
/// were measured against.
#[test]
fn forced_scalar_path_is_the_s1_arithmetic() {
    let q = CpuQuant::Mq4G256;
    let (m, k) = (4usize, 512usize);
    let packed = testfix::weight_bytes(q, m, k);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.25 - 32.0).collect();
    let mut forced = vec![0.0f32; m];
    gemv_with_simd(q, &packed, m, k, &x, &mut forced, Some(false));
    // Hand-rolled over the same decode, in the same order.
    let ge = q.group_elems();
    let gb = q.group_bytes();
    let rb = row_bytes(q, k);
    for (row, got) in forced.iter().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..k / ge {
            let mut codes = vec![0.0f32; ge];
            crate::quant::decode_group_codes(q, &packed[row * rb + g * gb..], &mut codes);
            let mut partial = 0.0f32;
            for i in 0..ge {
                partial += codes[i] * x[g * ge + i];
            }
            acc += partial;
        }
        assert_eq!(*got, acc, "row {row}");
    }
}

/// Prepared sums must preserve each affine format's arithmetic. Awkward
/// metadata and activations expose rounding differences hidden by the
/// power-of-two shared-job fixtures.
#[test]
#[cfg(target_arch = "x86_64")]
fn cached_and_uncached_agree_on_awkward_inputs() {
    for q in [CpuQuant::Mq4G256, CpuQuant::Mq6G256] {
        if !row_dot_enabled(q, None) {
            continue;
        }
        for k in [256, 512, 2048, 12288] {
            let m = 7;
            let mut packed = testfix::weight_bytes(q, m, k);
            awkward_headers(q, &mut packed, m, k);
            for token in 0..4 {
                let x: Vec<f32> = (0..k)
                    .map(|i| (((i + token * 71) as u64 * 2654435761) % 4096) as f32 * 0.001 - 2.0)
                    .collect();
                let kind = activation_sum_kind(q).unwrap();
                let mut sums = vec![0.0; kind.len(k)];
                prepare_activation_sums(kind, &x, &mut sums);
                for row in 0..m {
                    let weights = &packed[row * row_bytes(q, k)..];
                    let uncached = row_dot_avx2(q, weights, &[], k, &x);
                    let cached = row_dot_with_activation_sums(q, weights, &[], k, &x, &sums);
                    assert_eq!(
                        cached.to_bits(),
                        uncached.to_bits(),
                        "{q:?} k={k} token={token} row={row}"
                    );
                }
            }
        }
    }
}

/// qt 53 takes any `k % 128`, including the Flash-Next down shape `k = 640`
/// (5 groups, 340 B per row) that no 256-group format can express. The shared
/// `kernel_formats` loop above only visits 256-multiples, so this walks the
/// 128-granular shapes — on awkward fp16 headers and on rows the (transcribed)
/// real encoder produced.
#[test]
fn mq4g128v2_avx2_and_scalar_agree_on_128_granular_shapes() {
    let q = CpuQuant::Mq4G128V2;
    if !row_dot_enabled(q, Some(true)) {
        eprintln!("skip {q:?}: AVX2/F16C absent on this runner");
        return;
    }
    let check = |packed: &[u8], m: usize, k: usize, what: &str| {
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as u64 * 2654435761) % 4096) as f32 * 0.001 - 2.0)
            .collect();
        let mut simd = vec![0.0f32; m];
        let mut scalar = vec![0.0f32; m];
        gemv_with_simd(q, packed, m, k, &x, &mut simd, Some(true));
        gemv_with_simd(q, packed, m, k, &x, &mut scalar, Some(false));
        for row in 0..m {
            let scale = scalar[row].abs().max(1e-6);
            let rel = (simd[row] - scalar[row]).abs() / scale;
            assert!(
                rel <= 1e-4,
                "{what} {m}x{k} row {row}: simd {} vs scalar {} (rel {rel:.3e})",
                simd[row],
                scalar[row]
            );
        }
    };
    for (m, k) in [(1usize, 128usize), (3, 384), (5, 640), (2, 1280), (1, 5120)] {
        assert_eq!(row_bytes(q, k), k / 128 * 68);
        let mut packed = testfix::weight_bytes(q, m, k);
        awkward_headers(q, &mut packed, m, k);
        check(&packed, m, k, "awkward headers");
        let w: Vec<f32> = (0..m * k)
            .map(|i| ((i as u64 * 40503 + 17) % 1999) as f32 * 0.001 - 1.0)
            .collect();
        let encoded = testfix::encode_mq4g128v2_ref(&w, m, k);
        check(&encoded, m, k, "encoder output");
    }
}

// ---- planar slab formats (qt 131/132/135/136) ------------------------------------

/// The four planar slab formats.
fn planar_formats() -> [CpuQuant; 4] {
    [
        CpuQuant::Mq3I4G128R256,
        CpuQuant::Mq3I4G128R128,
        CpuQuant::E8pI4G128R256,
        CpuQuant::E8pI4G128R128,
    ]
}

fn planar_x(k: usize, salt: usize) -> Vec<f32> {
    (0..k)
        .map(|i| (((i + salt * 131) as u64 * 2654435761) % 4096) as f32 * 0.001 - 2.0)
        .collect()
}

/// `(dot, Σ|w·x|)` of each row in `f64` over the crate's exact scalar decode
/// ([`decode_planar_row`]): an oracle whose error budget does not depend on how
/// much a random-sign row cancels.
fn planar_oracle(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32]) -> Vec<(f64, f64)> {
    let (rb, spr) = (row_bytes(q, k), 2 * (k / 128));
    let mut weights = vec![0.0f32; k];
    (0..m)
        .map(|r| {
            let at = m * rb + r * spr;
            decode_planar_row(
                q,
                &packed[r * rb..(r + 1) * rb],
                &packed[at..at + spr],
                k,
                &mut weights,
            );
            weights
                .iter()
                .zip(x)
                .fold((0.0f64, 0.0f64), |(d, l1), (&w, &xi)| {
                    let t = w as f64 * xi as f64;
                    (d + t, l1 + t.abs())
                })
        })
        .collect()
}

/// Format metadata is the wire contract: qt map, planar flag, 128-groups, code
/// bytes per group, no row header, and the rotation / `k` granularity per profile.
#[test]
fn planar_metadata_matches_the_wire_contract() {
    let table = [
        (131u8, CpuQuant::Mq3I4G128R256, 48usize, true),
        (132, CpuQuant::Mq3I4G128R128, 48, false),
        (135, CpuQuant::E8pI4G128R256, 32, true),
        (136, CpuQuant::E8pI4G128R128, 32, false),
    ];
    for (qt, q, gb, r256) in table {
        assert_eq!(CpuQuant::from_quant_type(qt), Some(q), "qt {qt}");
        assert!(q.is_planar(), "{q:?}");
        assert_eq!((q.group_elems(), q.group_bytes()), (128, gb), "{q:?}");
        assert_eq!(q.row_header_bytes(), 0, "{q:?}");
        assert_eq!(q.is_fwht_g256(), r256, "{q:?} g256");
        assert_eq!(q.is_fwht_g128(), !r256, "{q:?} g128");
        assert_eq!(
            q.input_rotation(),
            if r256 {
                InputRotation::FwhtG256
            } else {
                InputRotation::FwhtG128V2
            },
            "{q:?}"
        );
        assert_eq!(q.k_multiple(), if r256 { 256 } else { 128 }, "{q:?}");
    }
    for q in all_formats() {
        assert_eq!(q.is_planar(), planar_formats().contains(&q), "{q:?}");
    }
}

/// Whole-tensor sizes of the real Flash-Next expert shapes (gate_up
/// `[1280, 2560]`, down `[2560, 640]`) and the planar `row_bytes` meaning (codes
/// plane stride only).
#[test]
fn planar_tensor_bytes_match_the_wire_sizes() {
    // E8P: codes M*K/4 + scales 2*M*(K/128).
    assert_eq!(tensor_bytes(CpuQuant::E8pI4G128R256, 1280, 2560), 870_400);
    assert_eq!(tensor_bytes(CpuQuant::E8pI4G128R128, 2560, 640), 435_200);
    // W3: codes M*K*3/8 + scales 2*M*(K/128).
    assert_eq!(
        tensor_bytes(CpuQuant::Mq3I4G128R256, 1280, 2560),
        1280 * 2560 * 3 / 8 + 51_200
    );
    assert_eq!(tensor_bytes(CpuQuant::Mq3I4G128R256, 1280, 2560), 1_280_000);
    assert_eq!(
        tensor_bytes(CpuQuant::Mq3I4G128R128, 2560, 640),
        2560 * 640 * 3 / 8 + 2560 * 5 * 2
    );
    assert_eq!(tensor_bytes(CpuQuant::Mq3I4G128R128, 2560, 640), 640_000);
    // `row_bytes` is the codes-plane row stride, not tensor_bytes / m.
    assert_eq!(row_bytes(CpuQuant::Mq3I4G128R256, 2560), 2560 * 3 / 8);
    assert_eq!(row_bytes(CpuQuant::E8pI4G128R256, 2560), 2560 / 4);
    assert_eq!(row_bytes(CpuQuant::E8pI4G128R128, 640), 640 / 4);
    // Row formats keep `m * row_bytes`.
    assert_eq!(
        tensor_bytes(CpuQuant::Mq4G128V2, 7, 640),
        7 * row_bytes(CpuQuant::Mq4G128V2, 640)
    );
    assert_eq!(tensor_bytes(CpuQuant::Mq4G256, 3, 512), 3 * 2 * 136);
}

/// The planar kernels against the forced-scalar path on the real expert shapes
/// and the small ones, on random codes (every 16-bit E8P word and every 3-bit W3
/// code is legal) with finite positive f16 scales from subnormal to `2^15`.
/// Two checks: the same relative contract as
/// [`avx2_and_scalar_agree_within_tolerance`] (with the cancellation floor, see
/// [`cancellation_floor`]) and an absolute `1e-5·Σ|w·x|` bound against the exact
/// `f64` decode, for both the vector and the scalar path.
#[test]
fn planar_avx2_and_scalar_agree_within_tolerance() {
    for q in planar_formats() {
        if !row_dot_enabled(q, Some(true)) {
            eprintln!("skip {q:?}: AVX2/F16C absent on this runner");
            continue;
        }
        for (m, k) in [(1280usize, 2560usize), (2560, 640), (8, 256), (3, 128)] {
            // R256 profiles need k % 256 (640 and 128 are R128-only shapes).
            if k % q.k_multiple() != 0 {
                continue;
            }
            let packed = testfix::planar_tensor(q, m, k, 0xC0DE_0000 ^ (m * k) as u64);
            assert_eq!(packed.len(), tensor_bytes(q, m, k), "{q:?} {m}x{k}");
            let x = planar_x(k, 0);
            let mut simd = vec![0.0f32; m];
            let mut scalar = vec![0.0f32; m];
            gemv_with_simd(q, &packed, m, k, &x, &mut simd, Some(true));
            gemv_with_simd(q, &packed, m, k, &x, &mut scalar, Some(false));
            let floor = cancellation_floor(q, &packed, m, k, &x);
            let oracle = planar_oracle(q, &packed, m, k, &x);
            for row in 0..m {
                let scale = scalar[row].abs().max(1e-6).max(floor[row]);
                let rel = (simd[row] - scalar[row]).abs() / scale;
                assert!(
                    rel <= 1e-4,
                    "{q:?} {m}x{k} row {row}: simd {} vs scalar {} (rel {rel:.3e})",
                    simd[row],
                    scalar[row]
                );
                let (dot, l1) = oracle[row];
                for (what, got) in [("simd", simd[row]), ("scalar", scalar[row])] {
                    let err = (got as f64 - dot).abs();
                    assert!(
                        err <= 1e-5 * l1,
                        "{q:?} {m}x{k} row {row}: {what} {got} vs f64 oracle {dot} \
                         (err {err:.3e}, Σ|wx| {l1:.3e})"
                    );
                }
            }
        }
    }
}

/// Activation with mixed magnitudes (`2^±10`) and signs, so the word sums and
/// the signed-magnitude FMAs round differently from the benign [`planar_x`].
#[cfg(target_arch = "x86_64")]
fn e8p_awkward_x(k: usize, salt: usize) -> Vec<f32> {
    planar_x(k, salt)
        .into_iter()
        .enumerate()
        .map(|(i, v)| match (i + salt) % 5 {
            0 => v * 1024.0,
            1 => v / 1024.0,
            2 => -v,
            _ => v,
        })
        .collect()
}

/// Row `r`'s `(codes, scales)` of a planar `[m, k]` tensor.
#[cfg(target_arch = "x86_64")]
fn planar_row(q: CpuQuant, packed: &[u8], m: usize, k: usize, r: usize) -> (&[u8], &[u8]) {
    let (rb, spr) = (row_bytes(q, k), 2 * (k / 128));
    let at = m * rb + r * spr;
    (&packed[r * rb..(r + 1) * rb], &packed[at..at + spr])
}

/// E8P: the unprepared kernel (word sums inline), the prepared kernel (word
/// sums from [`prepare_activation_sums`]) and the two-row kernel must agree to
/// the bit, row by row, on random codes (every 16-bit word, so every shift bit
/// and sign pattern) with random f16 scales and awkward activations. The pair
/// kernel is also exercised on non-adjacent and identical rows.
#[test]
#[cfg(target_arch = "x86_64")]
fn e8p_prepared_and_pair_kernels_are_bit_identical_to_unprepared() {
    for q in [CpuQuant::E8pI4G128R256, CpuQuant::E8pI4G128R128] {
        if !row_dot_enabled(q, Some(true)) {
            eprintln!("skip {q:?}: AVX2/F16C absent on this runner");
            continue;
        }
        assert!(has_row_pair_kernel(q), "{q:?}");
        let kind = activation_sum_kind(q).expect("E8P has prepared sums");
        for k in [128usize, 640, 2560] {
            // R256 profiles need k % 256 (640 and 128 are R128-only shapes).
            if k % q.k_multiple() != 0 {
                continue;
            }
            let m = 7;
            let packed = testfix::planar_tensor(q, m, k, 0xE8B0_0000 ^ k as u64);
            for salt in 0..4 {
                for x in [planar_x(k, salt), e8p_awkward_x(k, salt)] {
                    let mut sums = vec![f32::NAN; kind.len(k)];
                    assert_eq!(sums.len(), k / 8, "{q:?} k={k}");
                    prepare_activation_sums(kind, &x, &mut sums);
                    for r in 0..m {
                        let (row, srow) = planar_row(q, &packed, m, k, r);
                        let uncached = row_dot_avx2(q, row, srow, k, &x);
                        let cached = row_dot_with_activation_sums(q, row, srow, k, &x, &sums);
                        assert_eq!(
                            cached.to_bits(),
                            uncached.to_bits(),
                            "{q:?} k={k} salt={salt} row={r}: cached vs uncached"
                        );
                    }
                    for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 0), (3, 3)] {
                        let (row0, srow0) = planar_row(q, &packed, m, k, a);
                        let (row1, srow1) = planar_row(q, &packed, m, k, b);
                        let [p0, p1] = row_pair_dot_with_activation_sums(
                            q, row0, srow0, row1, srow1, k, &x, &sums,
                        );
                        let s0 = row_dot_with_activation_sums(q, row0, srow0, k, &x, &sums);
                        let s1 = row_dot_with_activation_sums(q, row1, srow1, k, &x, &sums);
                        assert_eq!(
                            p0.to_bits(),
                            s0.to_bits(),
                            "{q:?} k={k} salt={salt} pair ({a},{b}) row {a}"
                        );
                        assert_eq!(
                            p1.to_bits(),
                            s1.to_bits(),
                            "{q:?} k={k} salt={salt} pair ({a},{b}) row {b}"
                        );
                    }
                }
            }
        }
    }
}

/// E8P: the folded kernels (unprepared, prepared, pair) against the exact `f64`
/// decode ([`planar_oracle`]; absolute `1e-5·Σ|w·x|`) and against the
/// forced-scalar path under the same relative contract and cancellation floor as
/// [`planar_avx2_and_scalar_agree_within_tolerance`].
#[test]
#[cfg(target_arch = "x86_64")]
fn e8p_folded_kernels_match_the_f64_decode_within_tolerance() {
    for q in [CpuQuant::E8pI4G128R256, CpuQuant::E8pI4G128R128] {
        if !row_dot_enabled(q, Some(true)) {
            eprintln!("skip {q:?}: AVX2/F16C absent on this runner");
            continue;
        }
        let kind = activation_sum_kind(q).expect("E8P has prepared sums");
        for k in [128usize, 640, 2560] {
            if k % q.k_multiple() != 0 {
                continue;
            }
            let m = 8;
            let packed = testfix::planar_tensor(q, m, k, 0xE8B1_0000 ^ k as u64);
            for x in [planar_x(k, 0), e8p_awkward_x(k, 1)] {
                let mut sums = vec![0.0f32; kind.len(k)];
                prepare_activation_sums(kind, &x, &mut sums);
                let mut scalar = vec![0.0f32; m];
                gemv_with_simd(q, &packed, m, k, &x, &mut scalar, Some(false));
                let floor = cancellation_floor(q, &packed, m, k, &x);
                let oracle = planar_oracle(q, &packed, m, k, &x);
                for r in 0..m {
                    let (row, srow) = planar_row(q, &packed, m, k, r);
                    let (prow, psrow) = planar_row(q, &packed, m, k, r ^ 1);
                    let pair = if r % 2 == 0 {
                        row_pair_dot_with_activation_sums(q, row, srow, prow, psrow, k, &x, &sums)
                            [0]
                    } else {
                        row_pair_dot_with_activation_sums(q, prow, psrow, row, srow, k, &x, &sums)
                            [1]
                    };
                    let kernels = [
                        ("uncached", row_dot_avx2(q, row, srow, k, &x)),
                        (
                            "cached",
                            row_dot_with_activation_sums(q, row, srow, k, &x, &sums),
                        ),
                        ("pair", pair),
                    ];
                    let (dot, l1) = oracle[r];
                    for (what, got) in kernels {
                        let err = (got as f64 - dot).abs();
                        assert!(
                            err <= 1e-5 * l1,
                            "{q:?} k={k} row {r}: {what} {got} vs f64 oracle {dot} \
                             (err {err:.3e}, Σ|wx| {l1:.3e})"
                        );
                        let scale = scalar[r].abs().max(1e-6).max(floor[r]);
                        let rel = (got - scalar[r]).abs() / scale;
                        assert!(
                            rel <= 1e-4,
                            "{q:?} k={k} row {r}: {what} {got} vs scalar {} (rel {rel:.3e})",
                            scalar[r]
                        );
                    }
                }
            }
        }
    }
}

/// A batch of planar jobs over plain slices (the CPU MoE splice's shape).
struct PlanarBatch<'a> {
    q: CpuQuant,
    packs: Vec<&'a [u8]>,
    xs: Vec<Vec<&'a [f32]>>,
}

impl SharedJobSource for PlanarBatch<'_> {
    fn jobs(&self) -> usize {
        self.packs.len()
    }
    fn n(&self, j: usize) -> usize {
        self.xs[j].len()
    }
    fn quant(&self, _j: usize) -> CpuQuant {
        self.q
    }
    fn packed(&self, j: usize) -> &[u8] {
        self.packs[j]
    }
    fn xs_row(&self, j: usize, t: usize) -> &[f32] {
        self.xs[j][t]
    }
}

/// The D2 AR == MTP contract for the planar formats: a job with `n = 1..=4`
/// activation rows through `gemv_shared_sourced` (alone and with other jobs in
/// one batch) and through `gemm` is bit-identical to `n` single-row `gemv` calls,
/// on the vector path, the runtime-detected path and the forced-scalar path.
/// Every planar format, at one gate_up-like (`k % 256`) and one down-like
/// (`k = 640`) shape.
#[test]
fn planar_shared_sourced_is_bit_identical_to_per_row_gemv() {
    for q in planar_formats() {
        let (m, k) = if q.k_multiple() == 256 {
            (40usize, 512usize)
        } else {
            (40, 640)
        };
        let tensors: Vec<Vec<u8>> = (0..2)
            .map(|s| testfix::planar_tensor(q, m, k, 0xFEED_0000 + s))
            .collect();
        // Job `j` has `j + 1` activation rows over weight tensor `j % 2`.
        let rows: Vec<Vec<Vec<f32>>> = (0..4)
            .map(|j| (0..=j).map(|t| planar_x(k, 10 * j + t)).collect())
            .collect();
        let x_refs: Vec<Vec<&[f32]>> = rows
            .iter()
            .map(|r| r.iter().map(|v| v.as_slice()).collect())
            .collect();
        for requested in [Some(true), None, Some(false)] {
            // Per-row `gemv` reference: `want[j][t][r]`.
            let want: Vec<Vec<Vec<f32>>> = (0..4)
                .map(|j| {
                    rows[j]
                        .iter()
                        .map(|x| {
                            let mut y = vec![0.0f32; m];
                            gemv_with_simd(q, &tensors[j % 2], m, k, x, &mut y, requested);
                            y
                        })
                        .collect()
                })
                .collect();
            // One batch of all four jobs, then each job alone.
            let mut selections: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3]];
            selections.extend((0..4).map(|j| vec![j]));
            for sel in selections {
                let batch = PlanarBatch {
                    q,
                    packs: sel.iter().map(|&j| tensors[j % 2].as_slice()).collect(),
                    xs: sel.iter().map(|&j| x_refs[j].clone()).collect(),
                };
                let total: usize = sel.iter().map(|&j| m * (j + 1)).sum();
                let mut out = vec![f32::NAN; total];
                gemv_shared_sourced(m, k, &mut out, &batch, requested);
                let mut base = 0usize;
                for &j in &sel {
                    let n = j + 1;
                    for r in 0..m {
                        for t in 0..n {
                            assert_eq!(
                                out[base + r * n + t].to_bits(),
                                want[j][t][r].to_bits(),
                                "{q:?} {requested:?} job {j} n={n} row {r} token {t}"
                            );
                        }
                    }
                    base += m * n;
                }
            }
            for j in 0..4 {
                let n = j + 1;
                let flat: Vec<f32> = rows[j].iter().flatten().copied().collect();
                let mut out = vec![0.0f32; n * m];
                gemm_with_simd(q, &tensors[j % 2], m, k, &flat, n, &mut out, requested);
                for t in 0..n {
                    for r in 0..m {
                        assert_eq!(
                            out[t * m + r].to_bits(),
                            want[j][t][r].to_bits(),
                            "{q:?} {requested:?} gemm n={n} token {t} row {r}"
                        );
                    }
                }
            }
        }
    }
}

// ── AVX-512 tier / V2 cached Σx: exactness against the AVX2 kernels ──────────
//
// These tests hold the tier to *bit* equality with the AVX2 kernels (not the
// scalar tolerance above): the AVX-512 tier is a wider execution of the same
// arithmetic, and the V2 prepared sums are the same reductions run ahead of
// time. Anything that reassociates a sum fails them on the adversarial inputs.

#[cfg(target_arch = "x86_64")]
mod exact {
    use super::*;
    use crate::gemv::{gemv_shared_sourced, SharedJobSource};

    /// xorshift64* — deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    /// A finite `fp16` bit pattern: any sign/exponent/mantissa (zeros and
    /// subnormals included) except the Inf/NaN exponent.
    fn finite_f16(rng: &mut Rng) -> u16 {
        let bits = rng.next() as u16;
        if (bits >> 10) & 0x1f == 0x1f {
            bits & !(1 << 10)
        } else {
            bits
        }
    }

    /// `m` rows of random payload under random finite `fp16` headers, for
    /// `Mq4G256V2` (four halves-headers per 136 B group) or `Mq4G128V2` (one
    /// pair per 68 B group).
    fn random_weights(q: CpuQuant, m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
        let rb = row_bytes(q, k);
        let gb = q.group_bytes();
        let mut packed: Vec<u8> = (0..m * rb).map(|_| rng.next() as u8).collect();
        let header_halves = match q {
            CpuQuant::Mq4G256V2 => 4,
            CpuQuant::Mq4G128V2 => 2,
            _ => unreachable!(),
        };
        for g in 0..m * (k / q.group_elems()) {
            for h in 0..header_halves {
                let bits = finite_f16(rng);
                packed[g * gb + 2 * h..g * gb + 2 * h + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        packed
    }

    /// Number of activation styles [`activations`] produces.
    const MODES: usize = 7;

    /// Adversarial activations: magnitude mixes, cancellation across both
    /// chains and within a chunk, subnormals, signed zeros, absorption.
    fn activations(k: usize, mode: usize, rng: &mut Rng) -> Vec<f32> {
        let mut x: Vec<f32> = (0..k).map(|_| rng.unit() * 2.0).collect();
        match mode {
            0 => {}
            1 => {
                const SCALES: [f32; 6] = [1e-30, 1e-12, 1e-4, 1.0, 1e6, 1e15];
                for v in &mut x {
                    *v = rng.unit() * SCALES[(rng.next() % 6) as usize];
                }
            }
            2 => {
                // Exact cancellation across chains (i, i ^ 8), within a chunk
                // (i, i ^ 1) and across chunk pairs (i, i ^ 16).
                for i in 0..k {
                    let j = match i % 3 {
                        0 => i ^ 8,
                        1 => i ^ 1,
                        _ => i ^ 16,
                    };
                    if j < i && j < k {
                        x[i] = -x[j];
                    }
                }
            }
            3 => {
                for v in &mut x {
                    let sign = ((rng.next() & 1) as u32) << 31;
                    *v = f32::from_bits(sign | (rng.next() as u32 & 0x007f_ffff));
                }
            }
            4 => {
                for v in &mut x {
                    *v = match rng.next() % 4 {
                        0 => 0.0,
                        1 => -0.0,
                        _ => *v,
                    };
                }
            }
            5 => {
                // Absorption: a huge value followed by unit values that only
                // survive in the right accumulation order.
                for (i, v) in x.iter_mut().enumerate() {
                    *v = match i % 16 {
                        0 => 1.0e18,
                        1 => -1.0e18,
                        _ => *v,
                    };
                }
            }
            6 => {
                // Subnormal/zero/normal soup around cancellation.
                for (i, v) in x.iter_mut().enumerate() {
                    *v = match i % 5 {
                        0 => f32::from_bits(1 + (rng.next() as u32 & 0xff)),
                        1 => -0.0,
                        2 => 3.0e-39,
                        3 => -3.0e-39,
                        _ => *v * 1e-20,
                    };
                }
            }
            _ => unreachable!(),
        }
        x
    }

    fn bits_eq(a: f32, b: f32, what: &str) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: {a:e} ({:#010x}) vs {b:e} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }

    /// `(format, k)` shapes: the D2 gate/up (QT44, k=2560) and down (QT53,
    /// k=640) shapes plus smaller/larger multiples that move the group count.
    const SHAPES: [(CpuQuant, usize); 8] = [
        (CpuQuant::Mq4G256V2, 256),
        (CpuQuant::Mq4G256V2, 512),
        (CpuQuant::Mq4G256V2, 2560),
        (CpuQuant::Mq4G256V2, 5120),
        (CpuQuant::Mq4G128V2, 128),
        (CpuQuant::Mq4G128V2, 640),
        (CpuQuant::Mq4G128V2, 2560),
        (CpuQuant::Mq4G128V2, 1408),
    ];

    #[test]
    fn tier_resolves_to_avx2_when_avx512_is_unavailable() {
        for (q, _) in kernel_formats() {
            assert_eq!(
                resolve_tier(q, true, false),
                SimdTier::Avx2,
                "{q:?}: no AVX-512"
            );
            assert_eq!(
                resolve_tier(q, false, false),
                SimdTier::Scalar,
                "{q:?}: no AVX2"
            );
            assert_eq!(
                resolve_tier(q, false, true),
                SimdTier::Scalar,
                "{q:?}: forced off"
            );
            let want = if matches!(q, CpuQuant::Mq4G256V2 | CpuQuant::Mq4G128V2) {
                SimdTier::Avx512
            } else {
                SimdTier::Avx2
            };
            assert_eq!(resolve_tier(q, true, true), want, "{q:?}: AVX-512 present");
            assert_eq!(
                tier_for(q, Some(false)),
                SimdTier::Scalar,
                "{q:?}: forced scalar"
            );
        }
        // The detected tier is consistent with the detected features.
        for q in [CpuQuant::Mq4G256V2, CpuQuant::Mq4G128V2] {
            let want = match (row_dot_enabled(q, None), avx512_available()) {
                (false, _) => SimdTier::Scalar,
                (true, true) => SimdTier::Avx512,
                (true, false) => SimdTier::Avx2,
            };
            assert_eq!(tier_for(q, None), want, "{q:?}");
        }
    }

    /// With AVX-512 reported absent the AVX2 kernel is what `row_dot_tier`
    /// runs for the V2 formats — i.e. the explicit AVX2 tier is the old kernel.
    #[test]
    fn avx2_tier_runs_the_avx2_kernels_when_avx512_is_absent() {
        let mut rng = Rng(0x1234_5678);
        for (q, k) in SHAPES {
            if !row_dot_enabled(q, Some(true)) {
                continue;
            }
            let packed = random_weights(q, 3, k, &mut rng);
            let x = activations(k, 0, &mut rng);
            let tier = resolve_tier(q, true, false);
            assert_eq!(tier, SimdTier::Avx2);
            let rb = row_bytes(q, k);
            for r in 0..3 {
                let via_tier = row_dot_tier(tier, q, &packed[r * rb..], &[], k, &x);
                let direct = unsafe {
                    match q {
                        CpuQuant::Mq4G256V2 => {
                            x86::mq4g256v2_row_dot(packed[r * rb..].as_ptr(), k, x.as_ptr())
                        }
                        _ => x86::mq4g128v2_row_dot(packed[r * rb..].as_ptr(), k, x.as_ptr()),
                    }
                };
                bits_eq(via_tier, direct, &format!("{q:?} k={k} row {r}"));
            }
        }
    }

    #[test]
    fn avx512_row_dot_is_bit_identical_to_avx2() {
        if !avx512_available() {
            eprintln!("skip: AVX-512F absent on this runner");
            return;
        }
        let mut rng = Rng(0xA5A5_0001);
        let m = 40;
        for (q, k) in SHAPES {
            let rb = row_bytes(q, k);
            for seed in 0..2 {
                let packed = random_weights(q, m, k, &mut rng);
                for mode in 0..MODES {
                    let x = activations(k, mode, &mut rng);
                    for r in 0..m {
                        let row = &packed[r * rb..];
                        let a2 = row_dot_tier(SimdTier::Avx2, q, row, &[], k, &x);
                        let a5 = row_dot_tier(SimdTier::Avx512, q, row, &[], k, &x);
                        let disp = row_dot_avx2(q, row, &[], k, &x);
                        let what = format!("{q:?} k={k} seed={seed} mode={mode} row={r}");
                        bits_eq(a5, a2, &what);
                        bits_eq(disp, a2, &format!("dispatch {what}"));
                    }
                }
            }
        }
    }

    #[test]
    fn v2_cached_sums_are_bit_identical_to_uncached_on_both_tiers() {
        let tiers: Vec<SimdTier> = if avx512_available() {
            vec![SimdTier::Avx2, SimdTier::Avx512]
        } else {
            vec![SimdTier::Avx2]
        };
        let mut rng = Rng(0xC0DE_0002);
        let m = 40;
        for (q, k) in SHAPES {
            if !row_dot_enabled(q, Some(true)) {
                continue;
            }
            let kind = activation_sum_kind(q).expect("V2 MQ4 formats cache Σx");
            assert!(kind == ActivationSumKind::V2Halves);
            assert_eq!(kind.len(k), k / 128);
            let rb = row_bytes(q, k);
            let packed = random_weights(q, m, k, &mut rng);
            for mode in 0..MODES {
                let x = activations(k, mode, &mut rng);
                let mut sums = vec![0.0f32; kind.len(k)];
                prepare_activation_sums(kind, &x, &mut sums);
                for r in 0..m {
                    let row = &packed[r * rb..];
                    let reference = row_dot_tier(SimdTier::Avx2, q, row, &[], k, &x);
                    for &tier in &tiers {
                        let uncached = row_dot_tier(tier, q, row, &[], k, &x);
                        let cached =
                            row_dot_with_activation_sums_tier(tier, q, row, &[], k, &x, &sums);
                        let what = format!("{q:?} k={k} mode={mode} row={r} {tier:?}");
                        bits_eq(uncached, reference, &format!("uncached {what}"));
                        bits_eq(cached, reference, &format!("cached {what}"));
                    }
                    bits_eq(
                        row_dot_with_activation_sums(q, row, &[], k, &x, &sums),
                        reference,
                        &format!("dispatch {q:?} k={k} mode={mode} row={r}"),
                    );
                }
            }
        }
    }

    /// The decode-once multi-token row dot must reproduce, bit for bit, the
    /// single-token prepared kernel (and the AVX2 uncached reference) for every
    /// token count — below, at and above each tier's group width and odd
    /// counts — on both tiers and both V2 formats, over adversarial activations
    /// (a different style per token) and random weights with finite `fp16`
    /// headers.
    #[test]
    fn multi_token_row_dot_is_bit_identical_to_per_token_on_both_tiers() {
        let tiers: Vec<SimdTier> = if avx512_available() {
            vec![SimdTier::Avx2, SimdTier::Avx512]
        } else {
            vec![SimdTier::Avx2]
        };
        if !row_dot_enabled(CpuQuant::Mq4G256V2, Some(true)) {
            eprintln!("skip: AVX2/F16C absent on this runner");
            return;
        }
        let mut rng = Rng(0x3317_0008);
        let m = 12;
        for (q, k) in SHAPES {
            let kind = activation_sum_kind(q).unwrap();
            let sl = kind.len(k);
            let rb = row_bytes(q, k);
            for n in (1..=9usize).chain([15, 16, 17]) {
                let packed = random_weights(q, m, k, &mut rng);
                // Token t uses style (t + n) % MODES, so every style meets
                // every neighbour across the n sweep.
                let xs_own: Vec<Vec<f32>> = (0..n)
                    .map(|t| activations(k, (t + n) % MODES, &mut rng))
                    .collect();
                let sums_own: Vec<Vec<f32>> = xs_own
                    .iter()
                    .map(|x| {
                        let mut s = vec![0.0f32; sl];
                        prepare_activation_sums(kind, x, &mut s);
                        s
                    })
                    .collect();
                let xs: Vec<&[f32]> = xs_own.iter().map(|v| v.as_slice()).collect();
                let ss: Vec<&[f32]> = sums_own.iter().map(|v| v.as_slice()).collect();
                for &tier in &tiers {
                    for r in 0..m {
                        let row = &packed[r * rb..];
                        let mut got = vec![0.0f32; n];
                        row_dot_multi_with_activation_sums_tier(
                            tier, q, row, k, &xs, &ss, &mut got,
                        );
                        for t in 0..n {
                            let what = format!("{q:?} k={k} n={n} {tier:?} row={r} token={t}");
                            let single = row_dot_with_activation_sums_tier(
                                tier,
                                q,
                                row,
                                &[],
                                k,
                                xs[t],
                                ss[t],
                            );
                            bits_eq(got[t], single, &format!("multi vs per-token {what}"));
                            let reference = row_dot_tier(SimdTier::Avx2, q, row, &[], k, xs[t]);
                            bits_eq(got[t], reference, &format!("multi vs avx2 uncached {what}"));
                        }
                    }
                }
                // The dispatching entry resolves the same tier as the
                // single-token dispatcher.
                let row = &packed[..];
                let mut got = vec![0.0f32; n];
                row_dot_multi_with_activation_sums(q, row, k, &xs, &ss, &mut got);
                for t in 0..n {
                    bits_eq(
                        got[t],
                        row_dot_with_activation_sums(q, row, &[], k, xs[t], ss[t]),
                        &format!("dispatch {q:?} k={k} n={n} token={t}"),
                    );
                }
            }
        }
    }

    /// Every token of a group gets the same answer whichever neighbours share
    /// the decode: a token's result must not depend on its slot or on the other
    /// tokens (no cross-token state), including when the same activation row
    /// appears several times and when the sums slices alias.
    #[test]
    fn multi_token_result_is_independent_of_slot_and_neighbours() {
        if !row_dot_enabled(CpuQuant::Mq4G256V2, Some(true)) {
            return;
        }
        let tiers: Vec<SimdTier> = if avx512_available() {
            vec![SimdTier::Avx2, SimdTier::Avx512]
        } else {
            vec![SimdTier::Avx2]
        };
        let mut rng = Rng(0x5107_0009);
        for (q, k) in [(CpuQuant::Mq4G256V2, 2560usize), (CpuQuant::Mq4G128V2, 640)] {
            let kind = activation_sum_kind(q).unwrap();
            let packed = random_weights(q, 4, k, &mut rng);
            let rb = row_bytes(q, k);
            let base: Vec<Vec<f32>> = (0..MODES)
                .map(|mode| activations(k, mode, &mut rng))
                .collect();
            let base_sums: Vec<Vec<f32>> = base
                .iter()
                .map(|x| {
                    let mut s = vec![0.0f32; kind.len(k)];
                    prepare_activation_sums(kind, x, &mut s);
                    s
                })
                .collect();
            for &tier in &tiers {
                for r in 0..4 {
                    let row = &packed[r * rb..];
                    let alone: Vec<f32> = (0..MODES)
                        .map(|i| {
                            row_dot_with_activation_sums_tier(
                                tier,
                                q,
                                row,
                                &[],
                                k,
                                &base[i],
                                &base_sums[i],
                            )
                        })
                        .collect();
                    // A rotated, repeated schedule over all styles.
                    for shift in 0..MODES {
                        let order: Vec<usize> = (0..11).map(|t| (t * 3 + shift) % MODES).collect();
                        let xs: Vec<&[f32]> = order.iter().map(|&i| base[i].as_slice()).collect();
                        let ss: Vec<&[f32]> =
                            order.iter().map(|&i| base_sums[i].as_slice()).collect();
                        let mut got = vec![0.0f32; order.len()];
                        row_dot_multi_with_activation_sums_tier(
                            tier, q, row, k, &xs, &ss, &mut got,
                        );
                        for (t, &i) in order.iter().enumerate() {
                            bits_eq(
                                got[t],
                                alone[i],
                                &format!("{q:?} {tier:?} row={r} shift={shift} slot={t} style={i}"),
                            );
                        }
                    }
                }
            }
        }
    }

    /// The two prepared-sum owners (Mq4G256V2's halves and Mq4G128V2's groups)
    /// share one value per 128 activations: it must equal the AVX2 two-chain
    /// reduction `codes_dot_sum` runs per half.
    #[test]
    fn prepared_v2_sum_equals_the_uncached_half_reduction() {
        if !avx2_f16c_available() {
            return;
        }
        let mut rng = Rng(0x5EED_0003);
        for mode in 0..MODES {
            let k = 1024;
            let x = activations(k, mode, &mut rng);
            let mut sums = vec![0.0f32; k / 128];
            prepare_activation_sums(ActivationSumKind::V2Halves, &x, &mut sums);
            let gbuf = vec![0u8; 8 + 128];
            for (b, got) in sums.iter().enumerate() {
                let (_, want) = unsafe {
                    x86::codes_dot_sum::<4, 8, false>(
                        gbuf.as_ptr(),
                        x[b * 128..].as_ptr(),
                        0,
                        16,
                        0.0,
                    )
                };
                bits_eq(*got, want, &format!("mode={mode} block={b}"));
            }
        }
    }

    /// `codes_dot_sum512` against AVX2 `codes_dot_sum` for *every* `first..last`
    /// chunk range of a 256-element group — including the odd lengths whose
    /// tail must land in chain 0 only — cached and uncached.
    #[test]
    fn avx512_chunk_ranges_incl_odd_tails_match_avx2() {
        if !avx512_available() {
            eprintln!("skip: AVX-512F absent on this runner");
            return;
        }
        let mut rng = Rng(0x7A11_0004);
        for mode in 0..MODES {
            let gbuf: Vec<u8> = (0..8 + 128).map(|_| rng.next() as u8).collect();
            let x = activations(256, mode, &mut rng);
            for first in 0..=32usize {
                for last in first..=32usize {
                    let cached_sum = rng.unit();
                    let (a2, s2) = unsafe {
                        x86::codes_dot_sum::<4, 8, false>(
                            gbuf.as_ptr(),
                            x.as_ptr(),
                            first,
                            last,
                            0.0,
                        )
                    };
                    let (a5, s5) = unsafe {
                        x86_512::codes_dot_sum512::<8, false>(
                            gbuf.as_ptr(),
                            x.as_ptr(),
                            first,
                            last,
                            0.0,
                        )
                    };
                    let what = format!("mode={mode} {first}..{last}");
                    bits_eq(a5, a2, &format!("dot {what}"));
                    bits_eq(s5, s2, &format!("sum {what}"));
                    let (c2, cs2) = unsafe {
                        x86::codes_dot_sum::<4, 8, true>(
                            gbuf.as_ptr(),
                            x.as_ptr(),
                            first,
                            last,
                            cached_sum,
                        )
                    };
                    let (c5, cs5) = unsafe {
                        x86_512::codes_dot_sum512::<8, true>(
                            gbuf.as_ptr(),
                            x.as_ptr(),
                            first,
                            last,
                            cached_sum,
                        )
                    };
                    bits_eq(c5, c2, &format!("cached dot {what}"));
                    bits_eq(cs5, cs2, &format!("cached sum {what}"));
                    bits_eq(c5, a2, &format!("cached == uncached dot {what}"));
                }
            }
        }
    }

    struct Job {
        q: CpuQuant,
        packed: Vec<u8>,
        xs: Vec<Vec<f32>>,
    }
    struct Src(Vec<Job>);
    impl SharedJobSource for Src {
        fn jobs(&self) -> usize {
            self.0.len()
        }
        fn n(&self, j: usize) -> usize {
            self.0[j].xs.len()
        }
        fn quant(&self, j: usize) -> CpuQuant {
            self.0[j].q
        }
        fn packed(&self, j: usize) -> &[u8] {
            &self.0[j].packed
        }
        fn xs_row(&self, j: usize, t: usize) -> &[f32] {
            &self.0[j].xs[t]
        }
    }

    /// FNV-1a over the bit patterns of `out`.
    fn digest(out: &[f32]) -> u64 {
        out.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
            (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Old-vs-new byte equality on the real D2 arm shapes. The digests were
    /// produced by the pre-D1 tree (`138e20f1b3`: nibble-plane decode, one
    /// 128-block at a time) on these seeded weights/activations through the
    /// shared sourced GEMV: Flash-Next gate_up `[1280, 2560]` (qt 44) and down
    /// `[2560, 640]` (qt 53), four experts carrying 1, 2, 3 and 4 activation
    /// rows (the R1..R4 windows). Any change to a per-output FMA order, chain
    /// split, lane join or affine association moves a digest.
    #[test]
    fn flash_next_arm_shapes_match_the_pre_d1_golden_digests() {
        if !row_dot_enabled(CpuQuant::Mq4G256V2, None) {
            eprintln!("skip: AVX2/F16C absent on this runner");
            return;
        }
        let mut got = Vec::new();
        for (q, m, k) in [
            (CpuQuant::Mq4G256V2, 1280usize, 2560usize),
            (CpuQuant::Mq4G128V2, 2560, 640),
        ] {
            let mut rng = Rng(0xD1_0000 + k as u64);
            let jobs: Vec<Job> = (1..=4usize)
                .map(|n| Job {
                    q,
                    packed: random_weights(q, m, k, &mut rng),
                    xs: (0..n)
                        .map(|t| activations(k, (n + t) % MODES, &mut rng))
                        .collect(),
                })
                .collect();
            let src = Src(jobs);
            let total: usize = (1..=4usize).map(|n| m * n).sum();
            let mut out = vec![0.0f32; total];
            gemv_shared_sourced(m, k, &mut out, &src, None);
            got.push(digest(&out));
        }
        assert_eq!(
            got, GOLDEN_FLASH_NEXT_DIGESTS,
            "pre-D1 golden digests moved: {got:#x?}"
        );
    }

    /// Digests of [`flash_next_arm_shapes_match_the_pre_d1_golden_digests`]
    /// (qt 44 gate_up, qt 53 down) from the pre-D1 tree.
    const GOLDEN_FLASH_NEXT_DIGESTS: [u64; 2] = [0x5c98_d582_2f42_3f75, 0x581f_60e2_3a00_487c];

    /// The whole sourced path (Σx prep + dedup, rayon blocking, tier dispatch)
    /// reproduces, bit for bit, the AVX2 uncached row dots, for 1..=9, 16 and 17
    /// activation rows per job (past each tier's multi-token group width, odd
    /// and even) on 1/2/8/16-thread pools. Jobs alias activation
    /// rows (so the Σx dedup is exercised) and mix the two V2 formats, which
    /// share one prepared sum per 128 activations.
    #[test]
    fn sourced_gemv_is_bit_identical_to_avx2_uncached_across_pools() {
        if !row_dot_enabled(CpuQuant::Mq4G256V2, None) {
            eprintln!("skip: AVX2/F16C absent on this runner");
            return;
        }
        eprintln!(
            "tier QT44={:?} QT53={:?}",
            tier_for(CpuQuant::Mq4G256V2, None),
            tier_for(CpuQuant::Mq4G128V2, None)
        );
        let mut rng = Rng(0xFEED_0005);
        // (k, m): D2 gate_up shape and the down shape, plus a 128-granular k
        // for the G256 job's neighbours.
        for (k, m) in [(2560usize, 37usize), (640, 53)] {
            for n in (1..=9usize).chain([16, 17]) {
                let shared_a: Vec<Vec<f32>> = (0..n)
                    .map(|t| activations(k, (t + 1) % MODES, &mut rng))
                    .collect();
                let own: Vec<Vec<f32>> = (0..n)
                    .map(|t| activations(k, (t * 3) % MODES, &mut rng))
                    .collect();
                let mut jobs = Vec::new();
                if k % 256 == 0 {
                    for _ in 0..2 {
                        jobs.push(Job {
                            q: CpuQuant::Mq4G256V2,
                            packed: random_weights(CpuQuant::Mq4G256V2, m, k, &mut rng),
                            xs: shared_a.clone(),
                        });
                    }
                }
                jobs.push(Job {
                    q: CpuQuant::Mq4G128V2,
                    packed: random_weights(CpuQuant::Mq4G128V2, m, k, &mut rng),
                    xs: shared_a.clone(),
                });
                jobs.push(Job {
                    q: CpuQuant::Mq4G128V2,
                    packed: random_weights(CpuQuant::Mq4G128V2, m, k, &mut rng),
                    xs: own,
                });
                // Reference: per element, the AVX2 uncached kernel.
                let mut want = Vec::new();
                for job in &jobs {
                    let rb = row_bytes(job.q, k);
                    for r in 0..m {
                        for x in &job.xs {
                            want.push(row_dot_tier(
                                SimdTier::Avx2,
                                job.q,
                                &job.packed[r * rb..],
                                &[],
                                k,
                                x,
                            ));
                        }
                    }
                }
                let src = Src(jobs);
                for threads in [1usize, 2, 8, 16] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let mut got = vec![0.0f32; want.len()];
                    pool.install(|| gemv_shared_sourced(m, k, &mut got, &src, None));
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        bits_eq(*g, *w, &format!("k={k} n={n} threads={threads} out[{i}]"));
                    }
                }
            }
        }
    }

    /// hiptrx-indicative micro-benchmark: ns per row dot, single thread,
    /// AVX2 vs AVX-512, with Σx recomputed per row (uncached) or prepared
    /// (cached). Run:
    /// `cargo test -p hipfire-cpu --lib --release -- --ignored v2_row_dot_microbench --nocapture`
    #[test]
    #[ignore]
    fn v2_row_dot_microbench_hiptrx_indicative() {
        use std::hint::black_box;
        use std::time::Instant;
        if !row_dot_enabled(CpuQuant::Mq4G256V2, None) {
            eprintln!("skip: AVX2/F16C absent");
            return;
        }
        let have512 = avx512_available();
        let mut rng = Rng(0xBE4C_0006);
        println!("[hiptrx-indicative] ns per row dot, single thread (AVX-512F present: {have512})");
        println!(
            "{:<24} {:>12} {:>12} {:>12} {:>12}",
            "shape", "avx2", "avx2+cachedΣ", "avx512", "avx512+cachedΣ"
        );
        for (q, k, m) in [
            (CpuQuant::Mq4G256V2, 2560usize, 256usize),
            (CpuQuant::Mq4G128V2, 640, 512),
        ] {
            let rb = row_bytes(q, k);
            let packed = random_weights(q, m, k, &mut rng);
            let x = activations(k, 0, &mut rng);
            let kind = activation_sum_kind(q).unwrap();
            let mut sums = vec![0.0f32; kind.len(k)];
            prepare_activation_sums(kind, &x, &mut sums);
            let passes = 600usize;
            let time = |f: &dyn Fn(&[u8]) -> f32| -> f64 {
                for r in 0..m {
                    black_box(f(&packed[r * rb..]));
                }
                let t0 = Instant::now();
                let mut acc = 0.0f32;
                for _ in 0..passes {
                    for r in 0..m {
                        acc += f(black_box(&packed[r * rb..]));
                    }
                }
                black_box(acc);
                t0.elapsed().as_nanos() as f64 / (passes * m) as f64
            };
            let a2 = time(&|row| row_dot_tier(SimdTier::Avx2, q, row, &[], k, black_box(&x)));
            let a2c = time(&|row| {
                row_dot_with_activation_sums_tier(
                    SimdTier::Avx2,
                    q,
                    row,
                    &[],
                    k,
                    black_box(&x),
                    &sums,
                )
            });
            let (a5, a5c) = if have512 {
                (
                    time(&|row| row_dot_tier(SimdTier::Avx512, q, row, &[], k, black_box(&x))),
                    time(&|row| {
                        row_dot_with_activation_sums_tier(
                            SimdTier::Avx512,
                            q,
                            row,
                            &[],
                            k,
                            black_box(&x),
                            &sums,
                        )
                    }),
                )
            } else {
                (f64::NAN, f64::NAN)
            };
            println!(
                "{:<24} {a2:>12.1} {a2c:>12.1} {a5:>12.1} {a5c:>12.1}",
                format!("{q:?} k={k}")
            );
        }
    }

    /// hiptrx-indicative micro-benchmark of the decode-once multi-token row
    /// dot: ns per *weight row* (all `n` activation rows of it), single thread,
    /// per tier, `n` calls of the single-token prepared kernel vs one
    /// multi-token call. Run:
    /// `cargo test -p hipfire-cpu --lib --release -- --ignored v2_multi_token_microbench --nocapture`
    #[test]
    #[ignore]
    fn v2_multi_token_microbench_hiptrx_indicative() {
        use std::hint::black_box;
        use std::time::Instant;
        if !row_dot_enabled(CpuQuant::Mq4G256V2, None) {
            eprintln!("skip: AVX2/F16C absent");
            return;
        }
        let tiers: Vec<SimdTier> = if avx512_available() {
            vec![SimdTier::Avx2, SimdTier::Avx512]
        } else {
            vec![SimdTier::Avx2]
        };
        let mut rng = Rng(0xBE4C_0007);
        println!("[hiptrx-indicative] ns per weight row (all n activation rows), single thread");
        println!(
            "{:<22} {:<8} {:>3} {:>12} {:>12} {:>8}",
            "shape", "tier", "n", "per-token", "multi", "speedup"
        );
        for (q, k, m) in [
            (CpuQuant::Mq4G256V2, 2560usize, 256usize),
            (CpuQuant::Mq4G128V2, 640, 512),
        ] {
            let rb = row_bytes(q, k);
            let packed = random_weights(q, m, k, &mut rng);
            let kind = activation_sum_kind(q).unwrap();
            let sl = kind.len(k);
            let xs_all: Vec<Vec<f32>> = (0..8)
                .map(|t| activations(k, t % MODES, &mut rng))
                .collect();
            let sums_all: Vec<Vec<f32>> = xs_all
                .iter()
                .map(|x| {
                    let mut s = vec![0.0f32; sl];
                    prepare_activation_sums(kind, x, &mut s);
                    s
                })
                .collect();
            for &tier in &tiers {
                for n in [1usize, 2, 4, 8] {
                    let xs: Vec<&[f32]> = xs_all[..n].iter().map(|v| v.as_slice()).collect();
                    let ss: Vec<&[f32]> = sums_all[..n].iter().map(|v| v.as_slice()).collect();
                    let mut out = vec![0.0f32; n];
                    let passes = 400usize;
                    let mut time = |f: &mut dyn FnMut(&[u8], &mut [f32])| -> f64 {
                        for r in 0..m {
                            f(&packed[r * rb..], &mut out);
                        }
                        let t0 = Instant::now();
                        let mut acc = 0.0f32;
                        for _ in 0..passes {
                            for r in 0..m {
                                f(black_box(&packed[r * rb..]), &mut out);
                                acc += out[0];
                            }
                        }
                        black_box(acc);
                        t0.elapsed().as_nanos() as f64 / (passes * m) as f64
                    };
                    let per_token = time(&mut |row, out| {
                        for t in 0..n {
                            out[t] = row_dot_with_activation_sums_tier(
                                tier,
                                q,
                                row,
                                &[],
                                k,
                                black_box(xs[t]),
                                ss[t],
                            );
                        }
                    });
                    let multi = time(&mut |row, out| {
                        row_dot_multi_with_activation_sums_tier(tier, q, row, k, &xs, &ss, out)
                    });
                    println!(
                        "{:<22} {:<8} {n:>3} {per_token:>12.1} {multi:>12.1} {:>7.2}x",
                        format!("{q:?} k={k}"),
                        format!("{tier:?}"),
                        per_token / multi
                    );
                }
            }
        }
    }
}
