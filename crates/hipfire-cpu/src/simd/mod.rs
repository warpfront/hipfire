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
//!
//! **Exactness tiers.** `Mq4G256V2` and `Mq4G128V2` (the D2 host-slot expert
//! path's gate/up and down formats) additionally have an AVX-512F tier
//! ([`x86_512`]) that is *bit-identical* to their AVX2 kernels — each lane runs
//! the same FMA sequence its AVX2 chain does — so which tier ran never changes
//! an output bit. [`resolve_tier`] picks the tier once, beside [`features_for`];
//! without AVX-512F the AVX2 tier runs (never the scalar path), and there are
//! no environment switches: A/B is by comparing binaries.

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
mod x86_512;

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

/// Whether `q`'s vector kernel widens `fp16` metadata, weights or codebook
/// values with `vcvtph2ps`, i.e. needs F16C on top of AVX2 + FMA (see
/// [`avx2_f16c_available`]).
///
/// The match is deliberately exhaustive — the lists below name every
/// [`CpuQuant`] — so a new variant does not compile until its kernel's feature
/// requirement is stated here, next to the kernel arm in [`row_dot_avx2`].
fn needs_f16c(q: CpuQuant) -> bool {
    match q {
        // Kernels that widen fp16 metadata with `vcvtph2ps`.
        CpuQuant::Mq4G256V2
        | CpuQuant::Mq4G128V2
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
        | CpuQuant::F16
        | CpuQuant::Mfp4G32E8
        // Planar slab formats widen their f16 scale plane with `vcvtph2ps`.
        | CpuQuant::Mq3I4G128R256
        | CpuQuant::Mq3I4G128R128
        | CpuQuant::E8pI4G128R256
        | CpuQuant::E8pI4G128R128 => true,
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
        | CpuQuant::Bf16 => false,
    }
}

/// Whether `q` has a vector kernel on this CPU: AVX2 + FMA, plus F16C for the
/// kernels [`needs_f16c`] names.
fn features_for(q: CpuQuant) -> bool {
    if needs_f16c(q) {
        avx2_f16c_available()
    } else {
        avx2_available()
    }
}

/// The row-dot kernel an unforced GEMV runs for `q` on this CPU, named by the
/// instruction sets it uses: `"avx2-fma-f16c"` or `"avx2-fma"` for the vector
/// kernel, `"scalar"` for the fallback. Load logs print it so every run states
/// which CPU arithmetic path its host-executed experts took.
pub fn row_dot_kernel(q: CpuQuant) -> &'static str {
    match (row_dot_enabled(q, None), needs_f16c(q)) {
        (false, _) => "scalar",
        (true, true) => "avx2-fma-f16c",
        (true, false) => "avx2-fma",
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

/// Which row-dot implementation runs a format on this CPU.
///
/// `Avx512` and `Avx2` are bit-identical for every format that has both (see
/// [`x86_512`]); `Scalar` is the reference decode, a different summation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) enum SimdTier {
    Avx512,
    Avx2,
    Scalar,
}

/// Whether the CPU reports AVX-512F and VBMI (`vpmultishiftqb`, the nibble
/// unpack) on top of everything the V2 AVX2 kernels need (AVX2, FMA, F16C) —
/// the requirement of the [`x86_512`] kernels. A CPU with F but without VBMI
/// (Skylake-X) runs the bit-identical AVX2 kernels instead.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512_available() -> bool {
    avx2_f16c_available()
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn avx512_available() -> bool {
    false
}

/// Whether `q` has an AVX-512 kernel ([`x86_512`]).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn has_avx512_kernel(q: CpuQuant) -> bool {
    matches!(q, CpuQuant::Mq4G256V2 | CpuQuant::Mq4G128V2)
}

/// The tier for `q` given what the CPU offers — the single place tier
/// precedence lives, pure so that "AVX-512 absent" is testable on any runner.
///
/// `avx2_ok` is the (possibly forced, see [`use_avx2`]) answer to "may `q`'s
/// AVX2 kernel run" and `avx512f_ok` is [`avx512_available`]. AVX-512 is used
/// only for formats with an AVX-512 kernel; absent it, a vector-capable format
/// falls back to AVX2, never to scalar.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn resolve_tier(q: CpuQuant, avx2_ok: bool, avx512f_ok: bool) -> SimdTier {
    if !avx2_ok {
        SimdTier::Scalar
    } else if avx512f_ok && has_avx512_kernel(q) {
        SimdTier::Avx512
    } else {
        SimdTier::Avx2
    }
}

/// [`resolve_tier`] against the running CPU, honouring a forced `requested`
/// exactly as [`row_dot_enabled`] does.
#[allow(dead_code)]
pub(crate) fn tier_for(q: CpuQuant, requested: Option<bool>) -> SimdTier {
    resolve_tier(q, use_avx2(features_for(q), requested), avx512_available())
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

/// Which cached-`Σx` reduction style a format's row dot can reuse.
///
/// The styles are numerically distinct reductions of the same sums, so a
/// prepared `Σx` is only valid for the format that produced it — never reused
/// across styles. See [`prepare_activation_sums`] and
/// [`row_dot_with_activation_sums`].
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ActivationSumKind {
    /// qt 13 `Mq4G256`: [`x86`]'s hand-unrolled nibble path, `(x0+x1)+(x2+x3)`
    /// per 32-wide block.
    Mq4Quartets,
    /// qt 15 `Mq6G256`: even/odd 8-lane chunk chains.
    Mq6Pairs,
    /// qt 135 / 136 E8P-I4: one sum per 8-element word (`k / 8` per row), each
    /// in `chunk_sums8` order — the fixed `hadd`/permute reduction of eight
    /// words at a time, shared with the unprepared E8P row dot.
    E8pWords,
    /// qt 44 `Mq4G256V2` (one sum per 128-element half) and qt 53 `Mq4G128V2`
    /// (one per 128-element group): the same two-chain 16-chunk reduction, so
    /// the two formats share one prepared value per 128 activations.
    V2Halves,
}

#[cfg(target_arch = "x86_64")]
impl ActivationSumKind {
    /// How many `f32` sums one activation row of length `k` prepares: one per
    /// 256-element group for the MQ kinds, one per 128 elements for
    /// [`Self::V2Halves`], one per 8-element word for [`Self::E8pWords`].
    pub(crate) fn len(self, k: usize) -> usize {
        match self {
            ActivationSumKind::Mq4Quartets | ActivationSumKind::Mq6Pairs => k / 256,
            ActivationSumKind::V2Halves => k / 128,
            ActivationSumKind::E8pWords => k / 8,
        }
    }
}

/// The cached-`Σx` style `q` uses, or `None` when `q` has no prepared row dot.
///
/// MQ4, MQ6, the MQ4 V2 formats and both E8P formats reuse prepared sums. The
/// other codebook formats have no affine zero term and do not need `Σx`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn activation_sum_kind(q: CpuQuant) -> Option<ActivationSumKind> {
    match q {
        CpuQuant::Mq4G256 => Some(ActivationSumKind::Mq4Quartets),
        CpuQuant::Mq6G256 => Some(ActivationSumKind::Mq6Pairs),
        CpuQuant::Mq4G256V2 | CpuQuant::Mq4G128V2 => Some(ActivationSumKind::V2Halves),
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => Some(ActivationSumKind::E8pWords),
        _ => None,
    }
}

/// Precompute the `Σx` sums for `kind` into `out` — `out.len() ==
/// kind.len(k)` ([`ActivationSumKind::len`]): one `f32` per 256 activation
/// elements for the MQ kinds, one per 128 for the MQ4 V2 kinds, one per
/// 8-element word for E8P.
///
/// Each style reproduces its row dot's original reduction order exactly, so a
/// [`row_dot_with_activation_sums`] row is bit-identical to the unprepared
/// [`row_dot_avx2`] row. The caller resolves the feature decision once per job
/// ([`row_dot_enabled`]) before calling either prepared entry point.
#[cfg(target_arch = "x86_64")]
pub(crate) fn prepare_activation_sums(kind: ActivationSumKind, x: &[f32], out: &mut [f32]) {
    debug_assert!(
        x.len()
            >= out.len()
                * match kind {
                    ActivationSumKind::Mq4Quartets | ActivationSumKind::Mq6Pairs => 256,
                    ActivationSumKind::V2Halves => 128,
                    ActivationSumKind::E8pWords => 8,
                },
        "prepare_activation_sums: activation shorter than the sums it covers"
    );
    unsafe {
        match kind {
            ActivationSumKind::Mq4Quartets => {
                x86::activation_sums_mq4(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
            ActivationSumKind::Mq6Pairs => {
                x86::activation_sums_mq6(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
            ActivationSumKind::V2Halves => {
                x86::activation_sums_v2(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
            ActivationSumKind::E8pWords => {
                x86::activation_sums_e8p(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
        }
    }
}

/// One weight row dotted with `x`, reusing the prepared `Σx` in `sums`
/// ([`prepare_activation_sums`]) instead of recomputing it.
///
/// Only the formats [`activation_sum_kind`] names have cached sums, and the
/// caller has already resolved the feature decision ([`row_dot_enabled`]), so
/// `q` is AVX2-enabled and `sums.len() >= kind.len(k)`. The dot FMAs, group
/// accumulation and final scaling are the same as [`row_dot_avx2`], on the
/// same tier.
///
/// `scales_row` is the row's f16 scale bytes for the planar formats (as in
/// [`row_dot_avx2`]) and empty for the row formats.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_with_activation_sums(
    q: CpuQuant,
    row: &[u8],
    scales_row: &[u8],
    k: usize,
    x: &[f32],
    sums: &[f32],
) -> f32 {
    row_dot_with_activation_sums_tier(
        resolve_tier(q, true, avx512_available()),
        q,
        row,
        scales_row,
        k,
        x,
        sums,
    )
}

/// [`row_dot_with_activation_sums`] on an explicit `tier`. `Avx512` requires
/// [`avx512_available`] (the caller's burden, as with [`row_dot_tier`]).
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_with_activation_sums_tier(
    tier: SimdTier,
    q: CpuQuant,
    row: &[u8],
    scales_row: &[u8],
    k: usize,
    x: &[f32],
    sums: &[f32],
) -> f32 {
    debug_assert!(tier != SimdTier::Scalar, "no scalar prepared row dot");
    debug_assert!(
        tier != SimdTier::Avx512 || avx512_available(),
        "row_dot_with_activation_sums_tier({q:?}): AVX-512 absent"
    );
    debug_assert!(
        activation_sum_kind(q).is_some_and(|kind| sums.len() >= kind.len(k)),
        "row_dot_with_activation_sums_tier({q:?}): sums shorter than the shape"
    );
    debug_assert!(
        !q.is_planar()
            || (row.len() >= crate::gemv::row_bytes(q, k)
                && scales_row.len() >= 2 * (k / crate::quant::PLANAR_GROUP)),
        "row_dot_with_activation_sums_tier({q:?}): planar row or scales shorter than the shape"
    );
    debug_assert!(
        x.len() >= k,
        "row_dot_with_activation_sums_tier({q:?}): short x"
    );
    let (p, xp, sp) = (row.as_ptr(), x.as_ptr(), sums.as_ptr());
    // SAFETY: as `row_dot_tier` — features resolved by the caller, bounds
    // checked above and by the callers' slices.
    unsafe {
        match (tier, q) {
            (SimdTier::Avx512, CpuQuant::Mq4G256V2) => {
                x86_512::mq4g256v2_row_dot_cached(p, k, xp, sp)
            }
            (SimdTier::Avx512, CpuQuant::Mq4G128V2) => {
                x86_512::mq4g128v2_row_dot_cached(p, k, xp, sp)
            }
            (_, CpuQuant::Mq4G256) => x86::mq4g256_row_dot_cached(p, k, xp, sp),
            (_, CpuQuant::Mq6G256) => x86::mq6g256_row_dot_cached(p, k, xp, sp),
            (_, CpuQuant::Mq4G256V2) => x86::mq4g256v2_row_dot_cached(p, k, xp, sp),
            (_, CpuQuant::Mq4G128V2) => x86::mq4g128v2_row_dot_cached(p, k, xp, sp),
            (_, CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128) => {
                x86::e8p_row_dot_cached(p, scales_row.as_ptr(), k, xp, sp)
            }
            _ => unreachable!("row_dot_with_activation_sums: {q:?} has no cached Σx"),
        }
    }
}

/// Whether `q` has a two-row prepared kernel
/// ([`row_pair_dot_with_activation_sums`]): the E8P formats, whose row dot is
/// bound by the table decode and shares each activation load across two rows.
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_row_pair_kernel(q: CpuQuant) -> bool {
    match q {
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => true,
        CpuQuant::Mq4G256
        | CpuQuant::Mq6G256
        | CpuQuant::Mq3I4G128R256
        | CpuQuant::Mq3I4G128R128
        | CpuQuant::Mq4G256V2
        | CpuQuant::Mq4G128V2
        | CpuQuant::Hfq4G256
        | CpuQuant::Hfq6G256
        | CpuQuant::Mq5G256
        | CpuQuant::Hfq3G256
        | CpuQuant::Mq3G256
        | CpuQuant::Hfq2G256
        | CpuQuant::Mq2G256
        | CpuQuant::Hfq4G128
        | CpuQuant::Hfq3G128
        | CpuQuant::Hfq2G128
        | CpuQuant::Mq4CG256
        | CpuQuant::Tq2G128
        | CpuQuant::Bq1G128
        | CpuQuant::Mfp4G32E8
        | CpuQuant::Mq6G256V2
        | CpuQuant::Mq5G256V2
        | CpuQuant::Mq3G256V2
        | CpuQuant::Mq2G256V2
        | CpuQuant::Mq2G256Lloyd
        | CpuQuant::Mq2G256LloydU
        | CpuQuant::Mq3G256Lloyd
        | CpuQuant::Mq4G256Lloyd
        | CpuQuant::F16
        | CpuQuant::Bf16
        | CpuQuant::F32
        | CpuQuant::Q8F16 => false,
    }
}

/// Two weight rows dotted with the same `x` in one pass, reusing the prepared
/// `Σx` in `sums`. Each row's result is bit-identical to
/// [`row_dot_with_activation_sums`] on that row.
///
/// Only formats with [`has_row_pair_kernel`] are legal; the caller has resolved
/// the feature decision ([`row_dot_enabled`]) and `sums.len() >= kind.len(k)`.
/// `srow0` / `srow1` are the rows' scale bytes ([`row_dot_avx2`]'s
/// `scales_row`).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn row_pair_dot_with_activation_sums(
    q: CpuQuant,
    row0: &[u8],
    srow0: &[u8],
    row1: &[u8],
    srow1: &[u8],
    k: usize,
    x: &[f32],
    sums: &[f32],
) -> [f32; 2] {
    debug_assert!(
        has_row_pair_kernel(q),
        "row_pair_dot_with_activation_sums({q:?}): no pair kernel"
    );
    debug_assert!(
        activation_sum_kind(q).is_some_and(|kind| sums.len() >= kind.len(k)),
        "row_pair_dot_with_activation_sums({q:?}): sums shorter than the shape"
    );
    debug_assert!(
        row0.len() >= crate::gemv::row_bytes(q, k)
            && row1.len() >= crate::gemv::row_bytes(q, k)
            && srow0.len() >= 2 * (k / crate::quant::PLANAR_GROUP)
            && srow1.len() >= 2 * (k / crate::quant::PLANAR_GROUP),
        "row_pair_dot_with_activation_sums({q:?}): row or scales shorter than the shape"
    );
    debug_assert!(
        x.len() >= k,
        "row_pair_dot_with_activation_sums({q:?}): short x"
    );
    unsafe {
        match q {
            CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => x86::e8p_row_pair_dot_cached(
                row0.as_ptr(),
                srow0.as_ptr(),
                row1.as_ptr(),
                srow1.as_ptr(),
                k,
                x.as_ptr(),
                sums.as_ptr(),
            ),
            _ => unreachable!("row_pair_dot_with_activation_sums: {q:?} has no pair kernel"),
        }
    }
}

/// Most activation rows one [`row_dot_multi_with_activation_sums`] call is
/// sized for (the widest tier's group; narrower tiers loop inside the call).
#[cfg(target_arch = "x86_64")]
pub(crate) const MULTI_TOKEN_MAX: usize = x86_512::MULTI_TOKENS_AVX512;

/// Whether `q` has a decode-once multi-token prepared row dot: the two MQ4 V2
/// formats, whose single-token prepared kernels it reproduces bit for bit.
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_multi_token_kernel(q: CpuQuant) -> bool {
    matches!(q, CpuQuant::Mq4G256V2 | CpuQuant::Mq4G128V2)
}

/// One weight row dotted with several activation rows (`xs[t]`, prepared sums
/// `sums[t]` from [`prepare_activation_sums`]), decoding each weight chunk once
/// for all of them. `out[t]` is bit-identical to
/// [`row_dot_with_activation_sums`]`(q, row, &[], k, xs[t], sums[t])`: every
/// token runs exactly the single-token kernel's FMA sequence, chain split, lane
/// join and affine association on its own accumulators.
///
/// Same preconditions as [`row_dot_with_activation_sums`], and
/// [`has_multi_token_kernel`]`(q)`. Any number of tokens is accepted; they are
/// processed in tier-sized groups.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_multi_with_activation_sums(
    q: CpuQuant,
    row: &[u8],
    k: usize,
    xs: &[&[f32]],
    sums: &[&[f32]],
    out: &mut [f32],
) {
    row_dot_multi_with_activation_sums_tier(
        resolve_tier(q, true, avx512_available()),
        q,
        row,
        k,
        xs,
        sums,
        out,
    )
}

/// [`row_dot_multi_with_activation_sums`] on an explicit `tier` (`Avx512`
/// requires [`avx512_available`], the caller's burden as with
/// [`row_dot_with_activation_sums_tier`]).
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_multi_with_activation_sums_tier(
    tier: SimdTier,
    q: CpuQuant,
    row: &[u8],
    k: usize,
    xs: &[&[f32]],
    sums: &[&[f32]],
    out: &mut [f32],
) {
    assert!(
        has_multi_token_kernel(q),
        "row_dot_multi_with_activation_sums_tier: {q:?} has no multi-token kernel"
    );
    debug_assert!(tier != SimdTier::Scalar, "no scalar prepared row dot");
    debug_assert!(
        tier != SimdTier::Avx512 || avx512_available(),
        "row_dot_multi_with_activation_sums_tier({q:?}): AVX-512 absent"
    );
    let n = xs.len();
    assert!(sums.len() == n && out.len() == n, "token count mismatch");
    assert!(
        k.is_multiple_of(q.group_elems()) && row.len() >= (k / q.group_elems()) * q.group_bytes(),
        "row_dot_multi_with_activation_sums_tier({q:?}): row shorter than the shape"
    );
    assert!(
        xs.iter().all(|x| x.len() >= k) && sums.iter().all(|s| s.len() >= k / 128),
        "row_dot_multi_with_activation_sums_tier({q:?}): activation or sums shorter than the shape"
    );
    let group = match tier {
        SimdTier::Avx512 => x86_512::MULTI_TOKENS_AVX512,
        _ => x86::MULTI_TOKENS_AVX2,
    };
    let mut t0 = 0;
    while t0 < n {
        let g = (n - t0).min(group);
        let (xg, sg, og) = (&xs[t0..t0 + g], &sums[t0..t0 + g], &mut out[t0..t0 + g]);
        // SAFETY: features resolved by the caller, bounds asserted above.
        unsafe {
            if tier == SimdTier::Avx512 {
                multi_group_avx512(q, row, k, xg, sg, og);
            } else {
                multi_group_avx2(q, row, k, xg, sg, og);
            }
        }
        t0 += g;
    }
}

/// Monomorphizes a multi-token group over its token count (`xs.len()`, one of
/// the listed values) for one tier's kernels.
#[cfg(target_arch = "x86_64")]
macro_rules! multi_group {
    ($name:ident, $kernels:ident, $($t:literal),+) => {
        /// # Safety
        ///
        /// The tier's CPU features, and the bounds
        /// [`row_dot_multi_with_activation_sums_tier`] asserts, for
        /// `1..=max` tokens.
        unsafe fn $name(
            q: CpuQuant,
            row: &[u8],
            k: usize,
            xs: &[&[f32]],
            sums: &[&[f32]],
            out: &mut [f32],
        ) {
            match xs.len() {
                $($t => {
                    let xp: [*const f32; $t] = core::array::from_fn(|i| xs[i].as_ptr());
                    let sp: [*const f32; $t] = core::array::from_fn(|i| sums[i].as_ptr());
                    let r = match q {
                        CpuQuant::Mq4G256V2 => {
                            $kernels::mq4g256v2_row_dot_cached_multi::<$t>(row.as_ptr(), k, &xp, &sp)
                        }
                        CpuQuant::Mq4G128V2 => {
                            $kernels::mq4g128v2_row_dot_cached_multi::<$t>(row.as_ptr(), k, &xp, &sp)
                        }
                        _ => unreachable!("{q:?} has no multi-token kernel"),
                    };
                    out.copy_from_slice(&r);
                })+
                n => unreachable!("multi-token group of {n}"),
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
multi_group!(multi_group_avx512, x86_512, 1, 2, 3, 4, 5, 6, 7, 8);
#[cfg(target_arch = "x86_64")]
multi_group!(multi_group_avx2, x86, 1, 2, 3, 4);
/// One weight row dotted with a pre-rotated activation on the best vector tier
/// for `q` ([`resolve_tier`]): AVX-512 for the formats that have a kernel, AVX2
/// otherwise.
///
/// The caller has already resolved the feature decision ([`row_dot_enabled`])
/// — which is why this takes no flags — so `row` must be at least
/// `(k / q.group_elems()) * q.group_bytes()` bytes and `x.len() >= k`. Rotation
/// is the caller's business, exactly as in [`crate::quant::decode_group_codes`].
///
/// `scales_row` is the row's f16 scale bytes for the planar formats
/// ([`CpuQuant::is_planar`]; `2 * (k / 128)` of them, `row` is then exactly the
/// row's codes) and empty for every other format. The planar kernels read only
/// within the two slices.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_avx2(q: CpuQuant, row: &[u8], scales_row: &[u8], k: usize, x: &[f32]) -> f32 {
    row_dot_tier(
        resolve_tier(q, true, avx512_available()),
        q,
        row,
        scales_row,
        k,
        x,
    )
}

/// [`row_dot_avx2`] on an explicit `tier` (`Avx2` keeps the AVX2 kernel
/// reachable for the bit-equality comparison). `Avx512` on a format without an
/// AVX-512 kernel runs its AVX2 kernel; it otherwise requires
/// [`avx512_available`], which the caller establishes.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_tier(
    tier: SimdTier,
    q: CpuQuant,
    row: &[u8],
    scales_row: &[u8],
    k: usize,
    x: &[f32],
) -> f32 {
    debug_assert!(tier != SimdTier::Scalar, "row_dot_tier({q:?}): scalar tier");
    debug_assert!(features_for(q), "row_dot_avx2({q:?}): features absent");
    debug_assert!(
        row.len() >= crate::gemv::row_bytes(q, k) && x.len() >= k,
        "row_dot_avx2({q:?}): row or activation shorter than the shape"
    );
    debug_assert!(
        !q.is_planar() || scales_row.len() >= 2 * (k / crate::quant::PLANAR_GROUP),
        "row_dot_avx2({q:?}): scales row shorter than the shape"
    );
    if tier == SimdTier::Avx512 && has_avx512_kernel(q) {
        debug_assert!(avx512_available(), "row_dot_tier({q:?}): AVX-512 absent");
        let (p, xp) = (row.as_ptr(), x.as_ptr());
        // SAFETY: AVX-512F (+ AVX2/FMA/F16C) is the caller's tier decision and
        // the bounds are the checks above — the kernels' own contract.
        return unsafe {
            match q {
                CpuQuant::Mq4G256V2 => x86_512::mq4g256v2_row_dot(p, k, xp),
                CpuQuant::Mq4G128V2 => x86_512::mq4g128v2_row_dot(p, k, xp),
                _ => unreachable!("has_avx512_kernel admitted {q:?}"),
            }
        };
    }
    row_dot_avx2_kernel(q, row, scales_row, k, x)
}

/// The AVX2 format→kernel map.
///
/// This is the single AVX2 format→kernel map, and like [`features_for`] it is
/// exhaustive: a new [`CpuQuant`] variant does not compile until its kernel and
/// its feature requirement are both stated.
#[cfg(target_arch = "x86_64")]
fn row_dot_avx2_kernel(q: CpuQuant, row: &[u8], scales_row: &[u8], k: usize, x: &[f32]) -> f32 {
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
        CpuQuant::Mfp4G32E8 => unsafe { x86::mfp4g32_e8_row_dot(p, k, xp) },
        // V2 family: per-128 fp16 header.
        CpuQuant::Mq4G256V2 => unsafe { x86::mq4g256v2_row_dot(p, k, xp) },
        // Per-128 fp16 header over a 128-element row-local group.
        CpuQuant::Mq4G128V2 => unsafe { x86::mq4g128v2_row_dot(p, k, xp) },
        // Planar slab formats: codes plane row + f16 scale plane row.
        CpuQuant::Mq3I4G128R256 | CpuQuant::Mq3I4G128R128 => unsafe {
            x86::w3_row_dot(p, scales_row.as_ptr(), k, xp)
        },
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => unsafe {
            x86::e8p_row_dot(p, scales_row.as_ptr(), k, xp)
        },
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
pub(crate) fn row_dot_avx2(
    _q: CpuQuant,
    _row: &[u8],
    _scales_row: &[u8],
    _k: usize,
    _x: &[f32],
) -> f32 {
    unreachable!("row_dot_avx2 is reachable only where AVX2 was detected")
}

#[cfg(test)]
mod tests;
