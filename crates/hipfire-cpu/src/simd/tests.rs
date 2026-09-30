// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SIMD dispatch and vector-vs-scalar agreement.

use super::*;
use crate::gemv::{gemv_with_simd, row_bytes};
use crate::quant::CpuQuant;
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
        // Per-128 fp16 header (the V2 family).
        (CpuQuant::Mq4G256V2, true),
        (CpuQuant::Mq6G256V2, true),
        (CpuQuant::Mq5G256V2, true),
        (CpuQuant::Mq3G256V2, true),
        (CpuQuant::Mq2G256V2, true),
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
/// without them, and a forced `Some(false)` must always lose.
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
            for row in 0..m {
                let scale = scalar[row].abs().max(1e-6);
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
