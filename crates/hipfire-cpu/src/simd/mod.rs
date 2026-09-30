// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SIMD kernels for the CPU matmul core.
//!
//! `std::simd` is nightly-only (`portable_simd`) and this crate targets stable,
//! so the vector kernels are `core::arch` intrinsics behind `#[target_feature]`
//! with runtime detection, plus the plain scalar path as both the fallback and
//! the reference the vector path is tested against.
//!
//! The vector path is a *throughput* choice, not a numerical one: both compute
//! ordinary `f32` sequences and are compared within a relative tolerance (see
//! `simd::tests`), never bit-for-bit — the offload feature's contract is
//! llama.cpp-level coherence, not device parity, and the vector path is what
//! makes the CPU side competitive with reading the same bytes over PCIe.
//!
//! Parallelism stays in [`crate::gemv`] (rayon over output rows); the kernels
//! here compute a single row so the two compose without either owning the other.
//!
//! Every format the crate decodes has a kernel ([`x86`]), so [`row_dot_enabled`]
//! is a pure CPU-feature question — see `docs/quant-formats/cpu-simd-coverage.md`
//! for the per-format map.

#[cfg(target_arch = "x86_64")]
mod x86;

use crate::quant::CpuQuant;

/// Whether the CPU reports AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_available() -> bool {
    false
}

/// Whether the CPU additionally reports F16C, which the kernels over an `fp16`
/// header, `fp16` weights or an `fp16` codebook need to widen those values to
/// `f32`.
///
/// Every AVX2 part in practice has F16C, but "in practice" is not a hardware
/// guarantee and the conversion is the only `fp16` step in those kernels, so it
/// is detected rather than assumed.
#[cfg(target_arch = "x86_64")]
pub fn avx2_f16c_available() -> bool {
    avx2_available() && std::arch::is_x86_feature_detected!("f16c")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_f16c_available() -> bool {
    false
}

/// Whether `q` has a vector kernel on this CPU.
///
/// Every kernel needs AVX2 + FMA; the ones over an `fp16` header, `fp16`
/// weights or an `fp16` codebook need F16C on top (see
/// [`avx2_f16c_available`]).
///
/// The match is deliberately exhaustive — the lists below name every
/// [`CpuQuant`] — so a new variant does not compile until its kernel's feature
/// requirement is stated here, next to the kernel arm in [`row_dot_avx2`].
fn features_for(q: CpuQuant) -> bool {
    match q {
        // Kernels that widen fp16 metadata with `vcvtph2ps`.
        CpuQuant::Mq4G256V2
        | CpuQuant::Mq6G256V2
        | CpuQuant::Mq5G256V2
        | CpuQuant::Mq2G256V2
        | CpuQuant::Mq3G256V2
        | CpuQuant::Mq4CG256
        | CpuQuant::Tq2G128
        | CpuQuant::Bq1G128
        | CpuQuant::Mq2G256Lloyd
        | CpuQuant::Mq2G256LloydU
        | CpuQuant::Mq3G256Lloyd
        | CpuQuant::Mq4G256Lloyd
        | CpuQuant::Q8F16
        | CpuQuant::F16 => avx2_f16c_available(),
        // Kernels over a plain f32 header, or over weights that need no fp16
        // step at all.
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
        | CpuQuant::Hfq2G128
        | CpuQuant::F32
        | CpuQuant::Bf16 => avx2_available(),
    }
}

/// Whether the vector row dot can run for format `q` on this CPU, honouring a
/// forced `requested` (see [`use_avx2`]).
///
/// Resolved once per GEMV call rather than per row (`m` is thousands on every
/// real shape).
pub fn row_dot_enabled(q: CpuQuant, requested: Option<bool>) -> bool {
    use_avx2(features_for(q), requested)
}

/// Pure dispatch predicate. `requested` forces the decision (`Some(true)` /
/// `Some(false)`, for tests and for the SIMD-vs-scalar comparison), `None`
/// follows `available`.
///
/// A forced `Some(true)` on hardware without AVX2 would execute undefined
/// instructions, so the force is honoured only when `available` says the feature
/// is present: a forced request on unsupported hardware falls back to scalar
/// rather than faulting, which is also what lets the predicate be unit-tested on
/// any runner.
pub fn use_avx2(available: bool, requested: Option<bool>) -> bool {
    match requested {
        Some(true) => available,
        Some(false) => false,
        None => available,
    }
}

/// One weight row dotted with a pre-rotated activation on the AVX2 path.
///
/// This is the single format→kernel map, and like [`features_for`] it is
/// exhaustive: a new [`CpuQuant`] variant does not compile until its kernel and
/// its feature requirement are both stated. The caller has already resolved the
/// feature decision ([`row_dot_enabled`]) — which is why this takes no flags —
/// so `row` must be at least `(k / q.group_elems()) * q.group_bytes()` bytes and
/// `x.len() >= k`. Rotation is the caller's business, exactly as in
/// [`crate::quant::decode_group_codes`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_avx2(q: CpuQuant, row: &[u8], k: usize, x: &[f32]) -> f32 {
    debug_assert!(features_for(q), "row_dot_avx2({q:?}): features absent");
    debug_assert!(
        row.len() >= (k / q.group_elems()) * q.group_bytes() && x.len() >= k,
        "row_dot_avx2({q:?}): row or activation shorter than the shape"
    );
    let (p, xp) = (row.as_ptr(), x.as_ptr());
    // SAFETY: the caller established this format's CPU features through
    // `row_dot_enabled`, and the bounds are the row/activation lengths checked
    // above — the kernels' own contract.
    match q {
        // Flat f32 header, 256-element group.
        CpuQuant::Mq4G256 => unsafe { x86::mq4g256_row_dot(p, k, xp) },
        CpuQuant::Hfq4G256 => unsafe { x86::hfq4g256_row_dot(p, k, xp) },
        CpuQuant::Hfq6G256 => unsafe { x86::hfq6g256_row_dot(p, k, xp) },
        CpuQuant::Mq6G256 => unsafe { x86::mq6g256_row_dot(p, k, xp) },
        CpuQuant::Mq5G256 => unsafe { x86::mq5g256_row_dot(p, k, xp) },
        CpuQuant::Hfq3G256 => unsafe { x86::hfq3g256_row_dot(p, k, xp) },
        CpuQuant::Mq3G256 => unsafe { x86::mq3g256_row_dot(p, k, xp) },
        CpuQuant::Hfq2G256 => unsafe { x86::hfq2g256_row_dot(p, k, xp) },
        CpuQuant::Mq2G256 => unsafe { x86::mq2g256_row_dot(p, k, xp) },
        // Flat f32 header, 128-element block.
        CpuQuant::Hfq4G128 => unsafe { x86::hfq4g128_row_dot(p, k, xp) },
        CpuQuant::Hfq3G128 => unsafe { x86::hfq3g128_row_dot(p, k, xp) },
        CpuQuant::Hfq2G128 => unsafe { x86::hfq2g128_row_dot(p, k, xp) },
        // fp16 header.
        CpuQuant::Mq4CG256 => unsafe { x86::mq4cg256_row_dot(p, k, xp) },
        CpuQuant::Tq2G128 => unsafe { x86::tq2g128_row_dot(p, k, xp) },
        CpuQuant::Bq1G128 => unsafe { x86::bq1g128_row_dot(p, k, xp) },
        // V2 family: per-128 fp16 header.
        CpuQuant::Mq4G256V2 => unsafe { x86::mq4g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq6G256V2 => unsafe { x86::mq6g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq5G256V2 => unsafe { x86::mq5g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq3G256V2 => unsafe { x86::mq3g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq2G256V2 => unsafe { x86::mq2g256v2_row_dot(p, k, xp) },
        // Lloyd-Max codebooks.
        CpuQuant::Mq2G256Lloyd => unsafe { x86::mq2g256lloyd_row_dot(p, k, xp) },
        CpuQuant::Mq2G256LloydU => unsafe { x86::mq2g256lloydu_row_dot(p, k, xp) },
        CpuQuant::Mq3G256Lloyd => unsafe { x86::mq3g256lloyd_row_dot(p, k, xp) },
        CpuQuant::Mq4G256Lloyd => unsafe { x86::mq4g256lloyd_row_dot(p, k, xp) },
        // Element formats.
        CpuQuant::F16 => unsafe { x86::f16_row_dot(p, k, xp) },
        CpuQuant::Bf16 => unsafe { x86::bf16_row_dot(p, k, xp) },
        CpuQuant::F32 => unsafe { x86::f32_row_dot(p, k, xp) },
        CpuQuant::Q8F16 => unsafe { x86::q8f16_row_dot(p, k, xp) },
    }
}

/// On a non-x86_64 target there are no vector kernels at all
/// ([`avx2_available`] is `false`), so [`row_dot_enabled`] never lets a caller
/// reach this.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn row_dot_avx2(_q: CpuQuant, _row: &[u8], _k: usize, _x: &[f32]) -> f32 {
    unreachable!("row_dot_avx2 is reachable only where AVX2 was detected")
}

#[cfg(test)]
mod tests;
