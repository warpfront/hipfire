// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Unit tests for the hipfire-dispatch layer.
//!
//! Tests cover:
//! - `ShapePredicate::eval` — all three variants, boundary values
//! - `ArchPredicate::eval_arch` — key arch identities (RDNA1/2/3)
//! - `KernelRegistry` — register, resolve, arch gating, shape gating, fallback
//! - `KernelKey::for_gemv*` — dtype/variant → key mapping
//! - `dtype_needs_rotation` — MQ family true, HFQ/F32 false
//! - `GemvFamily::resolve` — arch predicate filtering via a real registry
//! - `Pipeline::can_satisfy` — prefix-match semantics

use crate::context::DispatchCtx;
use crate::families::fused_qkv::FusedQkvFamily;
use crate::families::gemv::GemvFamily;
use crate::pipeline::Pipeline;
use crate::tables::KernelRegistry;
use crate::types::*;
use rdna_compute::DType;

// ── helpers ───────────────────────────────────────────────────────────────────

/// gfx1010 = RDNA1: no dp4a, no WMMA, no MMQ.
fn ctx_rdna1() -> DispatchCtx {
    DispatchCtx::for_test("gfx1010")
}

/// gfx1030 = RDNA2: has dp4a, no WMMA w32, no MMQ.
fn ctx_rdna2() -> DispatchCtx {
    DispatchCtx::for_test("gfx1030")
}

/// gfx1100 = RDNA3: has dp4a, WMMA, MMQ.
fn ctx_rdna3() -> DispatchCtx {
    DispatchCtx::for_test("gfx1100")
}

/// gfx1200 = RDNA4: has dp4a, WMMA, no MMQ via gfx11 path.
fn ctx_rdna4() -> DispatchCtx {
    DispatchCtx::for_test("gfx1200")
}

/// gfx906 = Vega 20: wave64, sdot4/dp4a, gemv_dp4a_enabled by default.
fn ctx_gfx906() -> DispatchCtx {
    DispatchCtx::for_test("gfx906")
}

fn always_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    }
}

fn has_wmma_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::HasWmma,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    }
}

fn dp4a_variant(key: KernelKey) -> KernelVariant {
    KernelVariant {
        key,
        arch_required: ArchPredicate::HasDot2F32F16,
        shape_gate: None,
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    }
}

// ── ShapePredicate::eval ──────────────────────────────────────────────────────

#[test]
fn shape_batch_gt_passes_when_strictly_greater() {
    let s = ShapeInfo {
        batch_size: 2,
        ..Default::default()
    };
    assert!(ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_batch_gt_fails_when_equal() {
    let s = ShapeInfo {
        batch_size: 1,
        ..Default::default()
    };
    assert!(!ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_batch_gt_fails_when_less() {
    let s = ShapeInfo {
        batch_size: 0,
        ..Default::default()
    };
    assert!(!ShapePredicate::BatchGt(1).eval(&s));
}

#[test]
fn shape_head_dim_eq_passes_on_match() {
    let s = ShapeInfo {
        head_dim: 128,
        ..Default::default()
    };
    assert!(ShapePredicate::HeadDimEq(128).eval(&s));
}

#[test]
fn shape_head_dim_eq_fails_on_mismatch() {
    let s = ShapeInfo {
        head_dim: 64,
        ..Default::default()
    };
    assert!(!ShapePredicate::HeadDimEq(128).eval(&s));
}

#[test]
fn shape_m_lt_passes_when_strictly_less() {
    let s = ShapeInfo {
        m: 7,
        ..Default::default()
    };
    assert!(ShapePredicate::MLt(8).eval(&s));
}

#[test]
fn shape_m_lt_fails_when_equal() {
    let s = ShapeInfo {
        m: 8,
        ..Default::default()
    };
    assert!(!ShapePredicate::MLt(8).eval(&s));
}

#[test]
fn shape_m_lt_fails_when_greater() {
    let s = ShapeInfo {
        m: 9,
        ..Default::default()
    };
    assert!(!ShapePredicate::MLt(8).eval(&s));
}

// ── ArchPredicate::eval_arch ──────────────────────────────────────────────────

#[test]
fn arch_always_passes_on_all_archs() {
    assert!(ArchPredicate::Always.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::Always.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::Always.eval_arch(&ctx_rdna3()));
}

#[test]
fn arch_has_wmma_requires_rdna3_or_rdna4() {
    assert!(!ArchPredicate::HasWmma.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasWmma.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasWmma.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasWmma.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_wave32_admits_all_rdna_excludes_cdna() {
    // W4: HasWave32 = is_wave32() — every RDNA gen, NOT CDNA wave64. Strict
    // superset of HasWmma on RDNA3/4 (so a HasWmma→HasWave32 flip is byte-
    // identical there) and newly admits RDNA1/2.
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasWave32.eval_arch(&ctx_rdna4()));
    assert!(!ArchPredicate::HasWave32.eval_arch(&ctx_gfx906()));
    assert!(!ArchPredicate::HasWave32.eval_arch(&DispatchCtx::for_test("gfx942")));
}

#[test]
fn arch_has_dp4a_requires_rdna1p1_or_newer() {
    assert!(!ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasDot2F32F16.eval_arch(&ctx_rdna4()));
}

#[test]
fn arch_has_mmq_on_rdna3_or_rdna4() {
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna1()));
    assert!(!ArchPredicate::HasMmq.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasMmq.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasMmq.eval_arch(&ctx_rdna4())); // RDNA4 MQ6/HFQ6
}

#[test]
fn arch_gemv_dp4a_gfx906_only() {
    // HasDp4a (=v_dot4_i32_i8, gfx906-only)
    assert!(ArchPredicate::HasDp4a.eval_arch(&ctx_gfx906()));
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna2()));
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna3()));
    assert!(!ArchPredicate::HasDp4a.eval_arch(&ctx_rdna4()));
}

#[test]
fn fused_qkv_hfq6_resolves_cross_arch() {
    // Regression: the HFQ6 fused keys (qkv / gate_up / qkvza) used to be gated
    // `HasDp4a` (gfx906-only), which dead-gated the AWQ A3B trunk's HFQ6-promoted
    // layers on RDNA3/RDNA4 → batched-prefill panic "no implementation for
    // FusedQkvzaHfq6G256". Their batched run-arms call gemm_{qkv,gate_up,qkvza}_
    // hfq6g256, which carry the full cross-arch ladder, so they are now `Always`.
    let fam = FusedQkvFamily::new();
    for ctx in [&ctx_gfx906(), &ctx_rdna3()] {
        assert!(fam
            .resolve(KernelKey::FusedQkvzaHfq6G256, ctx, None)
            .is_ok());
        assert!(fam.resolve(KernelKey::FusedQkvHfq6G256, ctx, None).is_ok());
        assert!(fam
            .resolve(KernelKey::FusedGateUpHfq6G256, ctx, None)
            .is_ok());
    }
}

#[test]
fn fused_qkv_variant_for_key_classifies_by_family() {
    use KernelKey::*;
    // 3-way QKV family (incl. Q4K + Paro QKV synthesis).
    for k in [
        FusedQkvHfq4G256,
        FusedQkvMq3G256Lloyd,
        FusedQkvMq4G256Lloyd,
        FusedQkvHfq6G256,
        FusedQkvQ4K,
        FusedQkvParo4G128T,
    ] {
        assert_eq!(
            fused_qkv_variant_for_key(k),
            Some(FusedQkvVariant::Qkv),
            "{k:?}"
        );
    }
    // 4-way QKVZA family (incl. Paro).
    for k in [
        FusedQkvzaHfq4G256,
        FusedQkvzaMq3G256Lloyd,
        FusedQkvzaMq4G256Lloyd,
        FusedQkvzaHfq6G256,
        FusedQkvzaParo4G128T,
    ] {
        assert_eq!(
            fused_qkv_variant_for_key(k),
            Some(FusedQkvVariant::Qkvza),
            "{k:?}"
        );
    }
    // 2-way Gate+Up family (incl. Q8_0 + Q4K + Paro).
    for k in [
        FusedGateUpHfq4G256,
        FusedGateUpMq3G256Lloyd,
        FusedGateUpMq4G256Lloyd,
        FusedGateUpHfq6G256,
        FusedGateUpQ4K,
        FusedGateUpQ8_0,
        FusedGateUpParo4G128T,
    ] {
        assert_eq!(
            fused_qkv_variant_for_key(k),
            Some(FusedQkvVariant::GateUp),
            "{k:?}"
        );
    }
    // Non-fused keys → None.
    assert_eq!(fused_qkv_variant_for_key(KernelKey::GemvF32), None);
    assert_eq!(fused_qkv_variant_for_key(KernelKey::GemmHfq4G256), None);
}

// ── KernelRegistry ────────────────────────────────────────────────────────────

#[test]
fn registry_resolve_happy_path() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap().key,
        KernelKey::GemvF32
    );
}

#[test]
fn registry_resolve_unregistered_key_returns_not_found() {
    let mut reg = KernelRegistry::new();
    let ctx = ctx_rdna1();
    let err = reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap_err();
    assert!(matches!(err, DispatchError::NotFound { .. }));
}

#[test]
fn registry_resolve_arch_gate_fails_returns_missing_impl() {
    let mut reg = KernelRegistry::new();
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1(); // no WMMA
    let err = reg
        .resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None)
        .unwrap_err();
    assert!(matches!(err, DispatchError::MissingImpl { .. }));
}

#[test]
fn registry_resolve_arch_gate_passes_on_capable_arch() {
    let mut reg = KernelRegistry::new();
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna3(); // has WMMA w32
    assert_eq!(
        reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None)
            .unwrap()
            .key,
        KernelKey::GemmHfq4G256Wmma,
    );
}

#[test]
fn registry_resolve_falls_through_to_second_variant() {
    // Register WMMA variant first, then fallback Always variant for same key.
    // On RDNA1 (no WMMA), the WMMA entry is skipped and fallback is selected.
    let mut reg = KernelRegistry::new();
    reg.register(has_wmma_variant(KernelKey::GemmHfq4G256Wmma));
    reg.register(always_variant(KernelKey::GemmHfq4G256Wmma));
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::GemmHfq4G256Wmma, &ctx, None)
            .unwrap()
            .key,
        KernelKey::GemmHfq4G256Wmma,
    );
}

#[test]
fn gemm_q8_0_batched_wide_exact_resolves_on_wmma_only() {
    use crate::families::gemm::GemmFamily;
    let fam = GemmFamily::new();
    let key = KernelKey::GemmQ8_0BatchedWideExact;
    assert!(
        fam.registry().all_keys().contains(&key),
        "GemmQ8_0BatchedWideExact must be registered in gemm_table"
    );
    assert_eq!(
        fam.registry().resolve(key, &ctx_rdna3(), None).unwrap().key,
        key,
        "HasWmma admits gfx1100"
    );
    assert_eq!(
        fam.registry().resolve(key, &ctx_rdna4(), None).unwrap().key,
        key,
        "HasWmma admits gfx1200"
    );
    let err = fam.registry().resolve(key, &ctx_rdna1(), None).unwrap_err();
    assert!(
        matches!(err, DispatchError::MissingImpl { .. }),
        "non-WMMA arch must reject GemmQ8_0BatchedWideExact, got {err:?}"
    );
}

#[test]
fn gemm_mq4v2_mq4c_keys_resolve_gfx12_only() {
    use crate::families::gemm::GemmFamily;
    let fam = GemmFamily::new();
    // All five V2 widths (MQ4V2 qt44 + MQ6/5/3/2V2 qt47-50) are HasWmma: batched prefill/lm_head WMMA sources exist on both gfx11 and gfx12.
    // MQ4C (qt45) remains HasWmmaGfx12: gfx12-only.
    let v2_keys = [
        KernelKey::GemmMq4G256V2,
        KernelKey::GemmMq4G256V2Residual,
        KernelKey::GemmMq4G256V2BatchedLmhead,
        KernelKey::GemmMq6G256V2,
        KernelKey::GemmMq6G256V2Residual,
        KernelKey::GemmMq6G256V2BatchedLmhead,
        KernelKey::GemmMq5G256V2,
        KernelKey::GemmMq5G256V2Residual,
        KernelKey::GemmMq5G256V2BatchedLmhead,
        KernelKey::GemmMq3G256V2,
        KernelKey::GemmMq3G256V2Residual,
        KernelKey::GemmMq3G256V2BatchedLmhead,
        KernelKey::GemmMq2G256V2,
        KernelKey::GemmMq2G256V2Residual,
        KernelKey::GemmMq2G256V2BatchedLmhead,
    ];
    let mq4c_keys = [
        KernelKey::GemmMq4CG256,
        KernelKey::GemmMq4CG256Residual,
        KernelKey::GemmMq4CG256BatchedLmhead,
    ];
    let rdna4 = ctx_rdna4();
    let rdna1 = ctx_rdna1();
    let rdna3 = ctx_rdna3();
    for &key in &v2_keys {
        assert!(
            fam.registry().all_keys().contains(&key),
            "{key:?} must be registered in gemm_table"
        );
        assert_eq!(
            fam.registry().resolve(key, &rdna4, None).unwrap().key,
            key,
            "{key:?} must resolve on gfx1200-class (HasWmma)"
        );
        assert_eq!(
            fam.registry().resolve(key, &rdna3, None).unwrap().key,
            key,
            "{key:?} must resolve on gfx1100-class (HasWmma)"
        );
        let err_rdna1 = fam.registry().resolve(key, &rdna1, None).unwrap_err();
        assert!(
            matches!(err_rdna1, DispatchError::MissingImpl { .. }),
            "{key:?} must MissingImpl on gfx1010, got {err_rdna1:?}"
        );
    }
    for &key in &mq4c_keys {
        assert!(
            fam.registry().all_keys().contains(&key),
            "{key:?} must be registered in gemm_table"
        );
        assert_eq!(
            fam.registry().resolve(key, &rdna4, None).unwrap().key,
            key,
            "{key:?} must resolve on gfx1200-class (HasWmmaGfx12)"
        );
        let err_rdna1 = fam.registry().resolve(key, &rdna1, None).unwrap_err();
        assert!(
            matches!(err_rdna1, DispatchError::MissingImpl { .. }),
            "{key:?} must MissingImpl on gfx1010, got {err_rdna1:?}"
        );
        let err_rdna3 = fam.registry().resolve(key, &rdna3, None).unwrap_err();
        assert!(
            matches!(err_rdna3, DispatchError::MissingImpl { .. }),
            "{key:?} must MissingImpl on gfx1100 (MQ4C gfx12-only), got {err_rdna3:?}"
        );
    }
}

#[test]
fn registry_resolve_shape_gate_passes_when_shape_matches() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let shape = ShapeInfo {
        head_dim: 128,
        ..Default::default()
    };
    assert_eq!(
        reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape))
            .unwrap()
            .key,
        KernelKey::AttnF32
    );
}

#[test]
fn registry_resolve_shape_gate_skips_when_shape_mismatches() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let shape = ShapeInfo {
        head_dim: 64,
        ..Default::default()
    };
    let err = reg
        .resolve(KernelKey::AttnF32, &ctx, Some(&shape))
        .unwrap_err();
    assert!(matches!(err, DispatchError::MissingImpl { .. }));
}

#[test]
fn registry_resolve_shape_none_bypasses_shape_gate() {
    // With shape=None, even a shape-gated variant should be selected.
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    assert_eq!(
        reg.resolve(KernelKey::AttnF32, &ctx, None).unwrap().key,
        KernelKey::AttnF32
    );
}

#[test]
fn registry_resolve_shape_gate_fallback_to_ungated_variant() {
    // Shape-gated fast path (head_dim=128) followed by ungated fallback.
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::AttnF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::HeadDimEq(128)),
        steps: &[],
        has_awq: false,
        tile: TileImpl::None,
    });
    reg.register(always_variant(KernelKey::AttnF32)); // ungated fallback
    let ctx = ctx_rdna1();
    let shape = ShapeInfo {
        head_dim: 64,
        ..Default::default()
    }; // doesn't match gated
    assert_eq!(
        reg.resolve(KernelKey::AttnF32, &ctx, Some(&shape))
            .unwrap()
            .key,
        KernelKey::AttnF32
    );
}

#[test]
fn resolve_honors_shape_gate() {
    let mut reg = KernelRegistry::new();
    reg.register(KernelVariant {
        key: KernelKey::GemvF32,
        arch_required: ArchPredicate::Always,
        shape_gate: Some(ShapePredicate::BatchGt(1)),
        steps: &[PipelineOp::Gemv],
        has_awq: true,
        tile: TileImpl::None,
    });
    reg.register(KernelVariant {
        key: KernelKey::GemvF32,
        arch_required: ArchPredicate::Always,
        shape_gate: None,
        steps: &[PipelineOp::Gemv],
        has_awq: false,
        tile: TileImpl::None,
    });
    let ctx = ctx_rdna1();
    let batched = ShapeInfo {
        batch_size: 8,
        head_dim: 0,
        m: 4096,
        is_tree: false,
    };
    let scalar = ShapeInfo {
        batch_size: 1,
        head_dim: 0,
        m: 4096,
        is_tree: false,
    };
    assert!(
        reg.resolve(KernelKey::GemvF32, &ctx, Some(&batched))
            .unwrap()
            .has_awq
    );
    assert!(
        !reg.resolve(KernelKey::GemvF32, &ctx, Some(&scalar))
            .unwrap()
            .has_awq
    );
    assert!(reg.resolve(KernelKey::GemvF32, &ctx, None).unwrap().has_awq);
}

#[test]
fn registry_validate_succeeds_on_populated_registry() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    assert!(reg.validate().is_ok());
}

#[test]
fn registry_all_keys_returns_registered_keys() {
    let mut reg = KernelRegistry::new();
    reg.register(always_variant(KernelKey::GemvF32));
    reg.register(always_variant(KernelKey::GemvF16));
    let keys = reg.all_keys();
    assert!(keys.contains(&KernelKey::GemvF32));
    assert!(keys.contains(&KernelKey::GemvF16));
    assert_eq!(keys.len(), 2);
}

// ── KernelKey::for_gemv* ──────────────────────────────────────────────────────

#[test]
fn for_gemv_plain_maps_all_scalar_dtypes() {
    let cases = [
        (DType::F32, KernelKey::GemvF32),
        (DType::F16, KernelKey::GemvF16),
        (DType::Q8_0, KernelKey::GemvQ8_0),
        (DType::HFQ4G256, KernelKey::GemvHfq4G256),
        (DType::MQ4G256, KernelKey::GemvMq4G256),
        (DType::MQ3G256, KernelKey::GemvMq3G256),
        (DType::MFP4G32, KernelKey::GemvMfp4G32),
    ];
    for (dtype, expected) in cases {
        assert_eq!(
            KernelKey::for_gemv(dtype, GemvVariant::Plain, false).unwrap(),
            expected,
            "dtype {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_prerotated_maps_mq_family() {
    let cases = [
        (DType::MQ4G256, KernelKey::GemvMq4G256Prerotated),
        (DType::MQ3G256, KernelKey::GemvMq3G256Prerotated),
        (DType::MQ2G256, KernelKey::GemvMq2G256Prerotated),
        (DType::MQ5G256, KernelKey::GemvMq5G256Prerotated),
        (DType::MQ6G256, KernelKey::GemvMq6G256Prerotated),
        (DType::MQ8G256, KernelKey::GemvMq8G256Prerotated),
        (DType::MFP4G32, KernelKey::GemvMfp4G32Prerotated),
    ];
    for (dtype, expected) in cases {
        assert_eq!(
            KernelKey::for_gemv_prerotated(dtype).unwrap(),
            expected,
            "dtype {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_prerotated_falls_back_to_plain_for_non_rotated() {
    // Rotation-free dtypes (RotationPlan::None) have no separate prerotated kernel —
    // their prerotated input is their plain input, so for_gemv_prerotated falls through
    // to for_gemv(Plain). This was changed to support the interpreter's unified
    // Prerotated input path (Ship 2.1) — previously these returned Err, which
    // caused a MissingImpl panic at runtime when the interpreter tried to dispatch
    // a non-rotated dtype through the Prerotated variant.
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::F32).unwrap(),
        KernelKey::GemvF32
    );
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::Q8_0).unwrap(),
        KernelKey::GemvQ8_0
    );
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::HFQ4G256).unwrap(),
        KernelKey::GemvHfq4G256
    );
    // Rotation-needing dtypes still resolve to dedicated prerotated keys.
    assert_eq!(
        KernelKey::for_gemv_prerotated(DType::MQ4G256).unwrap(),
        KernelKey::GemvMq4G256Prerotated
    );
}

/// GGUF block formats have no kernel yet: no gemv key, and the generic
/// execution gate refuses them by name instead of reinterpreting the bytes.
#[test]
fn ggml_dtypes_have_no_kernel_and_are_refused_by_name() {
    for dtype in [
        DType::GgmlQ4K,
        DType::GgmlQ5K,
        DType::GgmlQ6K,
        DType::GgmlQ8_0,
        DType::GgmlIQ4XS,
    ] {
        assert!(
            KernelKey::for_gemv(dtype, GemvVariant::Plain, false).is_err(),
            "{dtype:?} must have no plain gemv key"
        );
        assert!(KernelKey::for_gemv_prerotated(dtype).is_err(), "{dtype:?}");
        let err = crate::pipeline::batched::reject_typed_only_format(dtype)
            .expect_err("the execution gate must refuse a ggml dtype");
        let text = err.to_string();
        assert!(
            text.contains(dtype.ggml_name().unwrap()) && text.contains("no ggml kernel"),
            "{text}"
        );
    }
    // the legacy llama dtypes are untouched by the new gate.
    assert!(crate::pipeline::batched::reject_typed_only_format(DType::Q4K).is_ok());
    assert!(crate::pipeline::batched::reject_typed_only_format(DType::Q8_0).is_ok());
}

#[test]
fn for_gemv_residual_maps_hfq_and_mq() {
    let cases = [
        (DType::HFQ4G256, KernelKey::GemvHfq4G256Residual),
        (DType::HFQ3G256, KernelKey::GemvHfq3G256Residual),
        (DType::HFQ6G256, KernelKey::GemvHfq6G256Residual),
        (DType::MQ4G256, KernelKey::GemvMq4G256Residual),
        (DType::MQ3G256Lloyd, KernelKey::GemvMq3G256LloydResidual),
    ];
    for (dtype, expected) in cases {
        assert_eq!(
            KernelKey::for_gemv_residual(dtype).unwrap(),
            expected,
            "dtype {dtype:?}",
        );
    }
}

#[test]
fn for_gemv_swiglu_residual_maps_hfq_and_mq() {
    assert_eq!(
        KernelKey::for_gemv_swiglu_residual(DType::HFQ4G256).unwrap(),
        KernelKey::GemvHfq4G256SwiGLUResidual,
    );
    assert_eq!(
        KernelKey::for_gemv_swiglu_residual(DType::MQ4G256Lloyd).unwrap(),
        KernelKey::GemvMq4G256LloydSwiGLUResidual,
    );
}

// ── Admission → dispatch-arm completeness ─────────────────────────────────────
//
// A `KernelKey::for_*` constructor (or a key table / key selector) that ADMITS
// a dtype or key while the dispatch fn that finally launches it has no arm for
// it resolves fine and then fails the call at RUNTIME (`unsupported
// gemv.<variant>` / `no implementation for <key>`). Two instances so far:
// MQ6G256 in the residual/swiglu dispatch (fixed in d6c79bb17) and MQ3G256 in
// `dispatch_swiglu_residual` (the qwen3.5/3.6 27B MQ3 rungs' FFN down failed
// request 1 on gfx1201). The test below checks the whole class instead of one
// instance: every (admission, dispatch) pair in this crate is enumerated and
// every admitted dtype/key must have an explicit arm on the dispatch side.
//
// Admission side: the real `KernelKey::for_*` / `*_key_for` fns and the real
// registries are EVALUATED over every `DType` variant (`ALL_DTYPES`, checked
// against the enum's source by `all_dtypes_list_matches_the_dtype_enum`).
// Dispatch side: only the arm list is parsed out of the dispatch fn's source
// (rustfmt layout: arms one indent below the `match`), up to the fn's own
// closing brace. Shared-key cases (qt=52 MQ4G256V2Lloyd reusing the qt=44 key,
// MQ2G256LloydU mapping to GemvMq2G256Lloyd) are checked on the KEY where the
// dispatch is key-based and on the DTYPE where it is dtype-based.

/// Every `rdna_compute::DType` variant. `all_dtypes_list_matches_the_dtype_enum`
/// fails when the enum gains or loses a variant without this list following.
const ALL_DTYPES: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Q4K,
    DType::Q6K,
    DType::Q8_0,
    DType::Q4F16G64,
    DType::Q4F16G32,
    DType::Q8HFQ,
    DType::HFQ4G256,
    DType::HFQ4G128,
    DType::HFQ3G256,
    DType::HFQ3G128,
    DType::MQ4G256,
    DType::MQ4G256V2,
    DType::MQ4G256V2Lloyd,
    DType::MQ4CG256,
    DType::MQ6G256V2,
    DType::MQ5G256V2,
    DType::MQ4G128V2,
    DType::MQ4G128,
    DType::MQ3G256V2,
    DType::MQ2G256V2,
    DType::MQ8G256,
    DType::MQ6G256,
    DType::MQ5G256,
    DType::MQ3G256,
    DType::MQ2G256,
    DType::MQ2G256Lloyd,
    DType::MQ2G256LloydU,
    DType::MQ3G256Lloyd,
    DType::MQ4G256Lloyd,
    DType::MQ2G256GL,
    DType::MQ3G256GL,
    DType::HFP4G32,
    DType::MFP4G32,
    DType::MFP4G32Lloyd,
    DType::MFP4G32P,
    DType::MFP4G32E8,
    DType::MFP4G32E8SOA,
    DType::MFP3G32E8,
    DType::MFP2G32E8,
    DType::MFP4G32E8G128,
    DType::HFQ2G256,
    DType::HFQ2G128,
    DType::TQ2G128,
    DType::BQ1G128,
    DType::HFQ6G256,
    DType::ParoQ4G128,
    DType::ParoQ4G64,
    DType::FP8E4M3B128,
    DType::FP8E4M3PT,
    DType::HC8E4M3Row,
    DType::Raw,
    DType::MQ3I4G128R256,
    DType::MQ3I4G128R128,
    DType::MQ2I4G128R256,
    DType::MQ2I4G128R128,
    DType::E8PI4G128R256,
    DType::E8PI4G128R128,
    DType::MQ4G256V2P,
    DType::MQ4G128V2P,
    DType::GgmlQ4K,
    DType::GgmlQ5K,
    DType::GgmlQ6K,
    DType::GgmlQ8_0,
    DType::GgmlIQ4XS,
];

/// Registered keys that are KNOWN to have no dispatch arm today. Each entry is
/// re-checked: once the arm lands the entry is stale and the test fails until
/// it is removed, so this list can only shrink.
const KNOWN_UNDISPATCHED_KEYS: &[(&str, &str)] = &[(
    "GemmQ8_0Wmma4W",
    "registered in tables/gemm_table.rs (HasWmma) but GemmFamily::run_key has no arm \
     (run_key(GemmQ8_0Wmma4W) -> MissingImpl); its only consumer, pipeline/sandwich.rs, \
     calls gpu.gemm_q8_0_wmma_4w directly, so nothing resolves it through run_key today",
)];

/// Variant names of `pub enum DType` parsed from rdna-compute's source.
fn dtype_enum_source_variants() -> Vec<String> {
    let src = include_str!("../../rdna-compute/src/dispatch.rs");
    let mut out = Vec::new();
    let mut inside = false;
    for line in src.lines() {
        if line == "pub enum DType {" {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        if line == "}" {
            break;
        }
        let Some(rest) = line.strip_prefix("    ") else {
            continue;
        };
        let Some((ident, _)) = rest.split_once(',') else {
            continue;
        };
        if ident.starts_with(|c: char| c.is_ascii_uppercase())
            && ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            out.push(ident.to_string());
        }
    }
    out
}

#[test]
fn all_dtypes_list_matches_the_dtype_enum() {
    let listed: Vec<String> = ALL_DTYPES.iter().map(|d| format!("{d:?}")).collect();
    let source = dtype_enum_source_variants();
    assert!(
        source.len() >= 58,
        "parsed only {} DType variants from rdna-compute/src/dispatch.rs",
        source.len()
    );
    for v in &source {
        assert!(
            listed.contains(v),
            "DType::{v} exists in the enum but not in tests.rs ALL_DTYPES: add it, then check \
             every admission->dispatch pair in every_admitted_dtype_or_key_has_a_dispatch_arm"
        );
    }
    for v in &listed {
        assert!(
            source.contains(v),
            "ALL_DTYPES lists {v}, which is not a DType variant in the enum source"
        );
    }
    assert_eq!(listed.len(), source.len(), "duplicate entry in ALL_DTYPES");
}

/// Lines of the item whose first line contains `sig`: from that line up to (not
/// including) the first later line that is exactly `close` at the item's own
/// indentation. `sig` must match exactly one non-comment line.
fn item_lines(src: &'static str, sig: &str, close: &str) -> Vec<&'static str> {
    let lines: Vec<&'static str> = src.lines().collect();
    let hits: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].contains(sig) && !lines[i].trim_start().starts_with("//"))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "`{sig}` must match exactly one non-comment source line, matched {}",
        hits.len()
    );
    let start = hits[0];
    let indent = " ".repeat(lines[start].len() - lines[start].trim_start().len());
    let close_line = format!("{indent}{close}");
    let end = (start + 1..lines.len())
        .find(|&i| lines[i] == close_line)
        .unwrap_or_else(|| panic!("no closing `{close}` at the indent of `{sig}`"));
    lines[start..end].to_vec()
}

/// Every `KernelKey::<Ident>` named in the non-comment code of `lines`.
fn kernel_keys_in(lines: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        let mut rest = line.split("//").next().unwrap_or("");
        while let Some(at) = rest.find("KernelKey::") {
            rest = &rest[at + "KernelKey::".len()..];
            let ident: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                out.push(ident);
            }
        }
    }
    out
}

/// Names in one match-arm pattern: `|` alternations split, path prefixes
/// (`K::`, `KernelKey::`, `DType::`) stripped, guards dropped.
fn arm_pattern_names(pat: &str) -> Vec<String> {
    let pat = pat.split(" if ").next().unwrap_or(pat);
    pat.split('|')
        .filter_map(|alt| {
            let token = alt.split_whitespace().last()?;
            Some(token.rsplit("::").next()?.to_string())
        })
        .collect()
}

/// The arms of one dispatch fn's top-level `match`: (pattern names, arm body).
struct ArmSet {
    /// Human-readable dispatch site for failure messages.
    label: &'static str,
    arms: Vec<(Vec<String>, String)>,
}

impl ArmSet {
    /// Parse the arms of `match <scrutinee> {` directly inside the fn matched by
    /// `sig`, bounded by the fn's own closing brace. Relies on rustfmt layout
    /// (match at fn-body indent, arms one indent deeper). `min_arms` guards
    /// against a layout change silently emptying the set.
    fn parse(
        label: &'static str,
        src: &'static str,
        sig: &str,
        scrutinee: &str,
        min_arms: usize,
    ) -> Self {
        let body = item_lines(src, sig, "}");
        let fn_indent = body[0].len() - body[0].trim_start().len();
        let head = format!("match {scrutinee} {{");
        let m = body
            .iter()
            .position(|l| {
                let trimmed = l.trim();
                (trimmed == head || trimmed.strip_prefix("Ok(") == Some(head.as_str()))
                    && l.len() - l.trim_start().len() == fn_indent + 4
            })
            .unwrap_or_else(|| panic!("{label}: no `{head}` at the fn-body indent"));
        let arm_indent = fn_indent + 8;
        let match_close = format!(
            "{}{}",
            " ".repeat(fn_indent + 4),
            if body[m].trim_start().starts_with("Ok(") {
                "})"
            } else {
                "}"
            }
        );
        let end = (m + 1..body.len())
            .find(|&i| body[i] == match_close)
            .unwrap_or(body.len());

        let mut arms: Vec<(Vec<String>, String)> = Vec::new();
        let mut pending = String::new();
        for &line in &body[m + 1..end] {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            let continues_body = indent > arm_indent || trimmed.starts_with(['}', ')', ']']);
            if continues_body {
                if let Some((_, text)) = arms.last_mut() {
                    text.push_str(trimmed);
                    text.push('\n');
                }
                continue;
            }
            if trimmed.starts_with("//") {
                continue;
            }
            pending.push(' ');
            pending.push_str(trimmed);
            if let Some((pat, rest)) = pending.split_once("=>") {
                arms.push((arm_pattern_names(pat), rest.to_string()));
                pending.clear();
            }
        }
        assert!(
            arms.len() >= min_arms,
            "{label}: parsed only {} arms (expected >= {min_arms}); did the fn layout change?",
            arms.len()
        );
        ArmSet { label, arms }
    }

    /// Body of the first arm that explicitly names `name` (a wildcard / `other`
    /// arm never matches a concrete name).
    fn arm(&self, name: &str) -> Option<&str> {
        self.arms
            .iter()
            .find(|(names, _)| names.iter().any(|n| n == name))
            .map(|(_, body)| body.as_str())
    }
}

fn arm_failure(admitted_by: &str, name: &str, dispatch: &ArmSet) -> String {
    format!(
        "  {name}: admitted by {admitted_by}, but {} has no explicit arm for it \
         (runtime: `unsupported ...` / MissingImpl after resolution said Ok)",
        dispatch.label
    )
}

/// Every dtype/key an admission side returns Ok for / registers has an explicit
/// arm in the dispatch fn that launches it (see the section comment above;
/// d6c79bb17 fixed MQ6G256, and MQ3G256 in `dispatch_swiglu_residual` broke
/// qwen3.5/3.6 27B MQ3 decode on gfx1201). Pairs covered:
///
/// 1. `for_gemv(dtype, Plain)`            -> `launch` key arms (families/gemv.rs)
/// 2. `for_gemv_prerotated(dtype)`        -> `launch` key arms
/// 3. `for_gemv_residual(dtype)`          -> shared `residual_kernel` dtype arms
/// 4. `for_gemv_swiglu_residual(dtype)`   -> shared `residual_kernel` dtype arms
/// 5. GemvFamily registry (tables/gemv_table.rs `register_*`) -> `launch` key arms
///    (residual/swiglu keys are covered by 3/4; `GemvMfp4G32Fused` by `run`)
/// 6. FusedQkvFamily registry (tables/fused_qkv_table.rs) -> `dispatch_fused_qkv`
///    arms, or `launch_fused` (pipeline/steps.rs) arms that launch directly
/// 7. `FUSED_TABLE` pattern keys (pipeline/steps.rs) -> `launch_fused` arm that
///    exists and, when it forwards to `FusedQkvFamily::run`, `dispatch_fused_qkv`
/// 8. `fused_gate_up_key_for(dtype)`      -> `dispatch_fused_qkv` key arms
/// 9. `qkv_bias_fold_supported` key list (pipeline/steps.rs) ->
///    `dispatch_fused_qkv_with_qwen2_bias` key arms
/// 10. `residual_gemm_key_for(dtype)`     -> `GemmFamily::run_key` key arms
/// 11. GemmFamily registry (tables/gemm_table.rs) -> `run_key` arms (packed keys
///     via `packed_gemm_contract`, which `run_key` consumes before its match)
/// 12. `prefill_shared_gemm_key(dtype)` (pipeline/mod.rs) -> `run_key` key arms
///
/// Plain keys of dtypes whose `dtype_post_rotation_variant` is `Prerotated` (MQ*,
/// MFP4G32) are admitted by `for_gemv` but never launched as `Plain`
/// (`run_input` switches them to `Prerotated`), so they are exempt from pair 1
/// and from the registry check of pair 5.
#[test]
fn every_admitted_dtype_or_key_has_a_dispatch_arm() {
    use crate::families::fused_qkv::fused_gate_up_key_for;
    use crate::families::gemm::{residual_gemm_key_for, GemmFamily};
    use std::collections::BTreeSet;

    let gemv_src = include_str!("families/gemv.rs");
    let fused_src = include_str!("families/fused_qkv.rs");
    let gemm_src = include_str!("families/gemm.rs");
    let steps_src = include_str!("pipeline/steps.rs");
    let pipeline_src = include_str!("pipeline/mod.rs");

    let launch = ArmSet::parse(
        "families/gemv.rs::launch",
        gemv_src,
        "fn launch(",
        "key",
        35,
    );
    // Beta uses one dtype table for both residual variants. Pin the forwarding
    // edges as well as the table, so a wrapper cannot bypass the checked arms.
    for sig in ["fn dispatch_residual(", "fn dispatch_swiglu_residual("] {
        assert!(
            item_lines(gemv_src, sig, "}")
                .iter()
                .any(|line| line.trim_start().starts_with("launch_residual(")),
            "{sig} no longer forwards to the shared residual launcher"
        );
    }
    assert!(
        item_lines(gemv_src, "fn launch_residual(", "}")
            .iter()
            .any(|line| line.trim() == "match residual_kernel(w.dtype, variant)? {"),
        "the shared residual launcher no longer consumes the checked dtype table"
    );
    let residual = ArmSet::parse(
        "families/gemv.rs::residual_kernel (residual)",
        gemv_src,
        "fn residual_kernel(",
        "dtype",
        15,
    );
    let swiglu = ArmSet::parse(
        "families/gemv.rs::residual_kernel (swiglu_residual)",
        gemv_src,
        "fn residual_kernel(",
        "dtype",
        15,
    );
    let fused = ArmSet::parse(
        "families/fused_qkv.rs::dispatch_fused_qkv",
        fused_src,
        "fn dispatch_fused_qkv(",
        "params.kind",
        40,
    );
    let bias = ArmSet::parse(
        "families/fused_qkv.rs::dispatch_fused_qkv_with_qwen2_bias",
        fused_src,
        "fn dispatch_fused_qkv_with_qwen2_bias(",
        "params.kind",
        8,
    );
    let launch_fused = ArmSet::parse(
        "pipeline/steps.rs::launch_fused",
        steps_src,
        "fn launch_fused<'a>(",
        "key",
        8,
    );
    let run_key = ArmSet::parse(
        "families/gemm.rs::GemmFamily::run_key",
        gemm_src,
        "pub fn run_key(",
        "key",
        40,
    );
    let packed = ArmSet::parse(
        "families/gemm.rs::packed_gemm_contract (consumed by run_key before its match)",
        gemm_src,
        "fn packed_gemm_contract(",
        "key",
        4,
    );

    let mut failures: Vec<String> = Vec::new();
    let (mut n_plain, mut n_pre, mut n_res, mut n_swi) = (0usize, 0usize, 0usize, 0usize);
    let mut plain_unlaunched: BTreeSet<String> = BTreeSet::new();
    let mut residual_keys: BTreeSet<String> = BTreeSet::new();
    let mut swiglu_keys: BTreeSet<String> = BTreeSet::new();

    for &dtype in ALL_DTYPES {
        // 1. Plain
        if let Ok(key) = KernelKey::for_gemv(dtype, GemvVariant::Plain, false) {
            let name = format!("{key:?}");
            if matches!(dtype_post_rotation_variant(dtype), GemvVariant::Plain) {
                n_plain += 1;
                if launch.arm(&name).is_none() {
                    failures.push(arm_failure(
                        &format!("KernelKey::for_gemv({dtype:?}, Plain)"),
                        &name,
                        &launch,
                    ));
                }
            } else {
                plain_unlaunched.insert(name);
            }
        }
        // 2. Prerotated
        if let Ok(key) = KernelKey::for_gemv_prerotated(dtype) {
            n_pre += 1;
            let name = format!("{key:?}");
            if launch.arm(&name).is_none() {
                failures.push(arm_failure(
                    &format!("KernelKey::for_gemv_prerotated({dtype:?})"),
                    &name,
                    &launch,
                ));
            }
        }
        // 3. Residual (dtype-dispatched; the key is shared for qt=52, so the DTYPE is checked)
        if let Ok(key) = KernelKey::for_gemv_residual(dtype) {
            n_res += 1;
            residual_keys.insert(format!("{key:?}"));
            let name = format!("{dtype:?}");
            if residual.arm(&name).is_none() {
                failures.push(arm_failure(
                    &format!("KernelKey::for_gemv_residual({dtype:?}) = {key:?}"),
                    &name,
                    &residual,
                ));
            }
        }
        // 4. SwiGLU residual (dtype-dispatched)
        if let Ok(key) = KernelKey::for_gemv_swiglu_residual(dtype) {
            n_swi += 1;
            swiglu_keys.insert(format!("{key:?}"));
            let name = format!("{dtype:?}");
            if swiglu.arm(&name).is_none() {
                failures.push(arm_failure(
                    &format!("KernelKey::for_gemv_swiglu_residual({dtype:?}) = {key:?}"),
                    &name,
                    &swiglu,
                ));
            }
        }
    }
    assert!(n_plain >= 15, "plain: only {n_plain} admitted dtypes");
    assert!(n_pre >= 20, "prerotated: only {n_pre} admitted dtypes");
    assert!(n_res >= 15, "residual: only {n_res} admitted dtypes");
    assert!(n_swi >= 15, "swiglu_residual: only {n_swi} admitted dtypes");

    // 5. GemvFamily registry: every registered key is launchable.
    let run_body = item_lines(gemv_src, "pub fn run(", "}").join("\n");
    for key in GemvFamily::new().registry().all_keys() {
        let name = format!("{key:?}");
        if residual_keys.contains(&name) || swiglu_keys.contains(&name) {
            continue; // dtype-dispatched, covered by pairs 3/4
        }
        if name == "GemvMfp4G32Fused" {
            assert!(
                run_body.contains("KernelKey::GemvMfp4G32Fused")
                    && run_body.contains("dispatch_fused("),
                "GemvMfp4G32Fused is registered in tables/gemv_table.rs but GemvFamily::run \
                 no longer dispatches it through dispatch_fused"
            );
            continue;
        }
        if launch.arm(&name).is_none() && !plain_unlaunched.contains(&name) {
            failures.push(arm_failure(
                "the GemvFamily registry (tables/gemv_table.rs)",
                &name,
                &launch,
            ));
        }
    }

    // 6. FusedQkvFamily registry. A key is dispatchable when dispatch_fused_qkv
    //    has its arm, or launch_fused launches it directly (qt=52 Lloyd keys).
    let fused_dispatchable = |name: &str| {
        fused.arm(name).is_some()
            || launch_fused
                .arm(name)
                .is_some_and(|body| !body.contains("fused_qkv.run("))
    };
    for key in FusedQkvFamily::new().registry().all_keys() {
        let name = format!("{key:?}");
        if !fused_dispatchable(&name) {
            failures.push(arm_failure(
                "the FusedQkvFamily registry (tables/fused_qkv_table.rs)",
                &name,
                &fused,
            ));
        }
    }

    // 7. FUSED_TABLE pattern keys (what the decode pipeline actually selects).
    let table = kernel_keys_in(&item_lines(
        steps_src,
        "const FUSED_TABLE: &[FusedPattern]",
        "];",
    ));
    assert!(table.len() >= 20, "FUSED_TABLE: only {} keys", table.len());
    for name in &table {
        match launch_fused.arm(name) {
            None => failures.push(arm_failure(
                "a FUSED_TABLE pattern (pipeline/steps.rs)",
                name,
                &launch_fused,
            )),
            Some(_) if !fused_dispatchable(name) => failures.push(arm_failure(
                "a FUSED_TABLE pattern (pipeline/steps.rs; launch_fused forwards it to \
                 FusedQkvFamily::run)",
                name,
                &fused,
            )),
            Some(_) => {}
        }
    }

    // 8. fused_gate_up_key_for (panics for qt=52 by design: a refusal, not an admission).
    let mut n_gate_up = 0usize;
    for &dtype in ALL_DTYPES {
        let Ok(key) = std::panic::catch_unwind(|| fused_gate_up_key_for(dtype)) else {
            continue;
        };
        n_gate_up += 1;
        let name = format!("{key:?}");
        if fused.arm(&name).is_none() {
            failures.push(arm_failure(
                &format!("fused_gate_up_key_for({dtype:?})"),
                &name,
                &fused,
            ));
        }
    }
    assert!(n_gate_up >= 50, "fused_gate_up_key_for: {n_gate_up} dtypes");

    // 9. qkv_bias_fold_supported -> the Qwen2 bias dispatch.
    let bias_keys = kernel_keys_in(&item_lines(steps_src, "fn qkv_bias_fold_supported(", "}"));
    assert!(
        bias_keys.len() >= 5,
        "qkv_bias_fold_supported: {bias_keys:?}"
    );
    for name in &bias_keys {
        if bias.arm(name).is_none() {
            failures.push(arm_failure(
                "qkv_bias_fold_supported (pipeline/steps.rs)",
                name,
                &bias,
            ));
        }
    }

    // 10. residual_gemm_key_for (panics for qt=52 by design).
    let gemm_dispatchable = |name: &str| run_key.arm(name).is_some() || packed.arm(name).is_some();
    let mut n_gemm_res = 0usize;
    for &dtype in ALL_DTYPES {
        let Ok(key) = std::panic::catch_unwind(|| residual_gemm_key_for(dtype)) else {
            continue;
        };
        n_gemm_res += 1;
        let name = format!("{key:?}");
        if !gemm_dispatchable(&name) {
            failures.push(arm_failure(
                &format!("residual_gemm_key_for({dtype:?})"),
                &name,
                &run_key,
            ));
        }
    }
    assert!(
        n_gemm_res >= 50,
        "residual_gemm_key_for: {n_gemm_res} dtypes"
    );

    // 11. GemmFamily registry, with the explicit known-gap list.
    let mut seen_gaps: BTreeSet<&str> = BTreeSet::new();
    for key in GemmFamily::new().registry().all_keys() {
        let name = format!("{key:?}");
        if gemm_dispatchable(&name) {
            continue;
        }
        match KNOWN_UNDISPATCHED_KEYS.iter().find(|entry| entry.0 == name) {
            Some(entry) => {
                seen_gaps.insert(entry.0);
            }
            None => failures.push(arm_failure(
                "the GemmFamily registry (tables/gemm_table.rs)",
                &name,
                &run_key,
            )),
        }
    }
    for (k, reason) in KNOWN_UNDISPATCHED_KEYS {
        if !seen_gaps.contains(k) {
            failures.push(format!(
                "  {k}: listed in KNOWN_UNDISPATCHED_KEYS ({reason}) but it now has an arm or is \
                 no longer registered; remove the stale entry"
            ));
        }
    }

    // 12. prefill_shared_gemm_key (private to pipeline/): its returned keys.
    let shared = kernel_keys_in(&item_lines(
        pipeline_src,
        "fn prefill_shared_gemm_key(",
        "}",
    ));
    assert!(shared.len() >= 5, "prefill_shared_gemm_key: {shared:?}");
    for name in &shared {
        if !gemm_dispatchable(name) {
            failures.push(arm_failure(
                "prefill_shared_gemm_key (pipeline/mod.rs)",
                name,
                &run_key,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{} admitted dtype/key cases with no dispatch arm:\n{}\n",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn for_gemv_rejects_unsupported_variant_combo() {
    // Residual for F32 has no kernel.
    assert!(KernelKey::for_gemv_residual(DType::F32).is_err());
    // Prerotated for F32 now falls back to the plain key (rotation-free dtype).
    // See for_gemv_prerotated_falls_back_to_plain_for_non_rotated.
    assert!(KernelKey::for_gemv_prerotated(DType::F32).is_ok());
}

// ── dtype_needs_rotation ──────────────────────────────────────────────────────────

#[test]
fn dtype_needs_rotation_true_for_mq_family() {
    for dtype in [
        DType::MQ4G256,
        DType::MQ3G256,
        DType::MQ2G256,
        DType::MQ5G256,
        DType::MQ6G256,
        DType::MQ8G256,
        DType::MQ4G256Lloyd,
        DType::MFP4G32,
    ] {
        assert!(dtype_needs_rotation(dtype), "{dtype:?} should need FWHT");
    }
}

#[test]
fn dtype_needs_rotation_false_for_hfq_and_scalar() {
    for dtype in [
        DType::F32,
        DType::F16,
        DType::HFQ4G256,
        DType::Q8_0,
        DType::HFP4G32,
    ] {
        assert!(
            !dtype_needs_rotation(dtype),
            "{dtype:?} should NOT need FWHT"
        );
    }
}

#[test]
fn gemv_steps_rotation_matches_plan() {
    for dtype in [
        DType::MQ4G256,
        DType::MFP4G32,
        DType::ParoQ4G128,
        DType::HFQ4G256,
    ] {
        let steps = KernelKey::gemv_steps(dtype, GemvVariant::Plain);
        let plan = dtype_rotation_plan(dtype);
        let has_fwht = steps.contains(&PipelineOp::RotateFwht);
        let has_givens = steps.contains(&PipelineOp::GivensRotate);
        match plan {
            RotationPlan::Givens => {
                assert!(
                    has_givens && !has_fwht,
                    "{dtype:?}: Givens plan must emit GivensRotate, not FWHT"
                );
            }
            RotationPlan::FwhtG256 | RotationPlan::FwhtG128 => {
                assert!(
                    has_fwht && !has_givens,
                    "{dtype:?}: FWHT plan must emit RotateFwht"
                );
            }
            RotationPlan::None => {
                assert!(!has_fwht && !has_givens, "{dtype:?}: no rotation");
            }
            RotationPlan::Mq8Internal => {}
        }
    }
}

// ── GemvFamily::resolve via populated table ───────────────────────────────────

#[test]
fn mq4g128v2_is_explicitly_rejected_by_generic_dispatch_families() {
    let gemv = GemvFamily::new();
    for variant in [
        GemvVariant::Plain,
        GemvVariant::Prerotated,
        GemvVariant::WithResidual,
        GemvVariant::WithSwiGLUResidual,
    ] {
        let result = gemv.resolve(DType::MQ4G128V2, variant, false, &ctx_rdna3(), None);
        match result {
            Err(DispatchError::UnsupportedVariant { quant, .. }) => {
                assert_eq!(quant, "MQ4G128V2");
            }
            other => panic!("qt53 GEMV unexpectedly resolved through a generic family: {other:?}"),
        }
    }

    let gemm = crate::families::gemm::GemmFamily::new();
    match gemm.resolve(DType::MQ4G128V2, &ctx_rdna3(), None) {
        Err(DispatchError::UnsupportedVariant { quant, .. }) => {
            assert_eq!(quant, "MQ4G128V2");
        }
        other => panic!("qt53 GEMM unexpectedly resolved through a generic family: {other:?}"),
    }
}

#[test]
fn gemv_family_resolves_f32_on_all_archs() {
    let fam = GemvFamily::new();
    assert!(fam
        .resolve(DType::F32, GemvVariant::Plain, false, &ctx_rdna1(), None)
        .is_ok());
    assert!(fam
        .resolve(DType::F32, GemvVariant::Plain, false, &ctx_rdna3(), None)
        .is_ok());
}

#[test]
fn gemv_family_resolves_hfq4_on_all_archs() {
    let fam = GemvFamily::new();
    // HFQ4G256 uses generic wave32/wave64 kernels with a fallback for every arch
    // (gfx906 via dp4a/sdot4, gfx1010 via generic). Previously gated on HasDp4a
    // (has_dot2_f32_f16 = RDNA1.1+) which excluded gfx906/gfx1010.
    assert!(fam
        .resolve(
            DType::HFQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna1(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::HFQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::HFQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna3(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna1(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ4G256,
            GemvVariant::Plain,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
}

#[test]
fn gemv_family_resolves_mq3_prerotated_on_all_wave32_archs_not_cdna() {
    // W4: MQ3G256's GEMV is a WMMA-free [32,1,1] wave32 scalar kernel. Its arch
    // gate is now HasWave32 (was HasWmma), so it resolves on every RDNA gen
    // (RDNA1/2/3/4) but still NOT on CDNA wave64 (a [32,1,1] kernel needs wave32).
    let fam = GemvFamily::new();
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna1(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna3(),
            None
        )
        .is_ok());
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna4(),
            None
        )
        .is_ok());
    // CDNA wave64 (gfx906) still excluded by HasWave32.
    assert!(fam
        .resolve(
            DType::MQ3G256,
            GemvVariant::Prerotated,
            false,
            &ctx_gfx906(),
            None
        )
        .is_err());
    assert!(fam
        .resolve(
            DType::MQ4G256,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna2(),
            None
        )
        .is_ok());
    // F32 Prerotated now falls back to GemvF32 (rotation-free dtype → plain key).
    // It resolves on any arch because GemvF32 has no arch gate.
    assert!(fam
        .resolve(
            DType::F32,
            GemvVariant::Prerotated,
            false,
            &ctx_rdna3(),
            None
        )
        .is_ok());
}

// ── Pipeline::can_satisfy ─────────────────────────────────────────────────────

#[test]
fn pipeline_exact_match_satisfies() {
    let p = Pipeline::new(&[PipelineOp::RotateFwht, PipelineOp::Gemv]);
    assert!(p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
}

#[test]
fn pipeline_prefix_satisfies_longer_request() {
    let p = Pipeline::new(&[PipelineOp::RotateFwht]);
    assert!(p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
}

#[test]
fn pipeline_empty_satisfies_any_request() {
    let p = Pipeline::new(&[]);
    assert!(p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
    assert!(p.can_satisfy(&[]));
}

#[test]
fn pipeline_longer_than_request_fails() {
    let p = Pipeline::new(&[PipelineOp::RotateFwht, PipelineOp::Gemv]);
    assert!(!p.can_satisfy(&[PipelineOp::RotateFwht]));
}

#[test]
fn pipeline_prefix_mismatch_fails() {
    let p = Pipeline::new(&[PipelineOp::Gemv]);
    assert!(!p.can_satisfy(&[PipelineOp::RotateFwht, PipelineOp::Gemv]));
}

#[test]
fn pipeline_single_op_self_satisfies() {
    let p = Pipeline::new(&[PipelineOp::Gemv]);
    assert!(p.can_satisfy(&[PipelineOp::Gemv]));
    assert!(!p.can_satisfy(&[PipelineOp::RotateFwht]));
}

// ── MoeResolution eligibility lattice (mirrors qwen35.rs:4598-4671) ──
use crate::families::moe::{MoeDtypes, MoeResolution, MoeSharedDtypes};

fn dtypes_all_mq4() -> MoeDtypes<'static> {
    MoeDtypes {
        router: DType::MQ4G256,
        shared: Some(MoeSharedDtypes {
            selector: DType::MQ4G256,
            gate: DType::MQ4G256,
            up: DType::MQ4G256,
            down: DType::MQ4G256,
        }),
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256,
        routed_down: DType::MQ4G256,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    }
}

#[test]
fn moe_res_mq2_lloyd_u_is_indexable_but_never_rotates() {
    // MQ2G256LloydU carries UNROTATED weights. It must reach the same indexed
    // decode arms as its rotated sibling (same kernels, same byte layout) but
    // must NOT request the x rotation — feeding a FWHT-rotated x to unrotated
    // weights is silent garbage output, not an error.
    let mut d = dtypes_all_mq4();
    d.router = DType::F32;
    d.experts_all_gate_up_mq4 = false;
    d.routed_gate_up = DType::MQ2G256LloydU;
    d.routed_down = DType::MQ2G256LloydU;
    let r = MoeResolution::resolve(&d, 8);
    assert!(
        r.routed_indexable_mq2lloyd_u,
        "must reach the indexed MoE decode arms"
    );
    assert!(r.use_gpu_topk, "k=8 + indexable implies device-side top-K");
    assert!(
        !r.needs_x_rot_local,
        "UNROTATED dtype must never request the FWHT rotation"
    );
}

#[test]
fn unrotated_dtype_skips_both_rotations() {
    // THE invariant tying the resolver to the executor: a dtype is either
    // rotated in BOTH places (x before gate_up, and the intermediate before
    // down) or in NEITHER. A dtype that resolved `needs_x_rot_local == false`
    // but still took the rotating gate→down step would feed a rotated
    // activation to unrotated down weights — silent garbage, no error.
    //
    // Built as a differential over dtypes so it cannot pass vacuously: the
    // rotated and unrotated arms must DISAGREE on both flags.
    for (dt, expect_rotation) in [
        (DType::MQ2G256LloydU, false),
        (DType::MQ2G256Lloyd, true),
        (DType::MQ4G256, true),
        (DType::MQ6G256, true),
    ] {
        let mut d = dtypes_all_mq4();
        d.router = DType::F32;
        d.experts_all_gate_up_mq4 = false;
        d.routed_gate_up = dt;
        d.routed_down = dt;
        let r = MoeResolution::resolve(&d, 8);
        assert_eq!(
            r.needs_x_rot_local, expect_rotation,
            "{dt:?}: needs_x_rot_local"
        );
        assert_eq!(
            crate::pipeline::gate_down_skips_rotation(dt),
            !expect_rotation,
            "{dt:?}: gate→down rotation must agree with needs_x_rot_local"
        );
    }
}

#[test]
fn moe_res_mq2_lloyd_rotated_sibling_still_rotates() {
    // Guard: adding the unrotated arm must not disable rotation for qt19.
    let mut d = dtypes_all_mq4();
    d.router = DType::F32;
    d.experts_all_gate_up_mq4 = false;
    d.routed_gate_up = DType::MQ2G256Lloyd;
    d.routed_down = DType::MQ2G256Lloyd;
    let r = MoeResolution::resolve(&d, 8);
    assert!(
        r.needs_x_rot_local,
        "qt19 is FWHT-rotated and MUST still rotate"
    );
}

#[test]
fn moe_res_mixed_rotated_and_unrotated_experts_is_not_indexable() {
    // A layer whose gate_up is rotated and whose down is not (or vice versa)
    // has no coherent single rotation decision. It must fall out of the
    // indexed path rather than silently picking one and corrupting the other.
    let mut d = dtypes_all_mq4();
    d.router = DType::F32;
    d.experts_all_gate_up_mq4 = false;
    d.routed_gate_up = DType::MQ2G256LloydU;
    d.routed_down = DType::MQ2G256Lloyd;
    let r = MoeResolution::resolve(&d, 8);
    assert!(
        !r.routed_indexable_mq2lloyd_u,
        "rotated/unrotated mix must not resolve to the unrotated indexed arm"
    );
    assert!(
        !r.use_gpu_topk,
        "no coherent rotation decision, so no indexed decode at all"
    );
}

#[test]
fn moe_res_all_mq4_k8_uses_gpu_topk_and_xrot() {
    let r = MoeResolution::resolve(&dtypes_all_mq4(), 8);
    assert!(r.gate_side_mq4);
    assert!(r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local);
}

#[test]
fn moe_res_q8_router_still_gpu_topk() {
    // The non-obvious coupling: a Q8 router disqualifies the 4-way fused
    // gate-side GEMV (gate_side_mq4=false) but the routed experts are still
    // MQ4, so the device-side top-K + indexed path stays on (use_gpu_topk=true).
    let mut d = dtypes_all_mq4();
    d.router = DType::Q8_0;
    d.experts_all_gate_up_mq4 = true; // experts unchanged
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.gate_side_mq4);
    assert!(r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local); // routed_gate_up_mq4 alone fires x_rot
}

#[test]
fn qwen35_and_cohere_k8_preserve_gpu_topk_routed_decode() {
    // Qwen3.5 uses an MQ4 router; Cohere's route uses Q8.  The router dtype
    // only changes the fused gate-side decision, never the existing k=8
    // routed-expert top-k path.
    for (family, router) in [("Qwen3.5", DType::MQ4G256), ("Cohere", DType::Q8_0)] {
        let mut d = dtypes_all_mq4();
        d.router = router;
        let resolved = MoeResolution::resolve(&d, 8);
        assert!(
            resolved.routed_indexable_mq4,
            "{family}: routed MQ4 experts must remain indexable"
        );
        assert!(
            resolved.use_gpu_topk,
            "{family}: k=8 must retain device-side top-k routing"
        );
        assert!(
            resolved.needs_x_rot_local,
            "{family}: routed MQ4 experts require FWHT activation rotation"
        );
    }
}

#[test]
fn moe_res_k6_disables_gpu_topk_even_when_indexable() {
    // deepseek-shaped: indexable routed dtype but k != 8 => no GPU fast path
    let r = MoeResolution::resolve(&dtypes_all_mq4(), 6);
    assert!(r.routed_indexable_mq4);
    assert!(!r.use_gpu_topk);
}

#[test]
fn moe_res_mq4v2_routed_indexable() {
    // qt44 uniform: both projections MQ4G256V2 => indexable, GPU top-K on.
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ4G256V2;
    d.routed_down = DType::MQ4G256V2;
    d.experts_all_gate_up_mq4 = false;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq4v2);
    assert!(!r.routed_indexable_mq4, "must not claim the qt13 arm");
    assert!(r.use_gpu_topk);
    // qt44 is a FWHT-G256 format: its kernels read ROTATED activations.
    assert!(r.needs_x_rot_local);
}

#[test]
fn moe_res_all_mq4v2_gate_quartet_is_fusable_mq4v2() {
    // Exact-uniform V2 gate-side quartet admits the V2 fused route (one launch),
    // never the V1 fused route. Still needs the rotated activation.
    let mut d = dtypes_all_mq4();
    d.router = DType::MQ4G256V2;
    d.shared = Some(MoeSharedDtypes {
        selector: DType::MQ4G256V2,
        gate: DType::MQ4G256V2,
        up: DType::MQ4G256V2,
        down: DType::MQ4G256V2,
    });
    d.routed_gate_up = DType::MQ4G256V2;
    d.routed_down = DType::MQ4G256V2;
    d.experts_all_gate_up_mq4 = true;
    let r = MoeResolution::resolve(&d, 8);
    assert!(
        r.gate_fusable_mq4v2,
        "uniform all-gate-side V2 must admit gate_fusable_mq4v2"
    );
    assert!(
        !r.gate_fusable,
        "V2 gate quartet must not claim the V1 fused route"
    );
    assert!(!r.gate_side_mq4);
    assert!(r.needs_x_rot_local, "V2 fused gate requires rotated x");
    assert!(r.routed_indexable_mq4v2);
    assert!(r.use_gpu_topk);
}

#[test]
fn moe_res_mixed_v1_v2_gate_quartet_is_not_fusable() {
    // Mixed V1/V2 gate-side is never fusable on either predicate: V1 f32 header
    // vs V2 dual-f16 header share stride; wrong launcher is silent garbage.
    // Routed V2 indexability is independent and stays on.
    let mut d = dtypes_all_mq4();
    d.router = DType::MQ4G256V2; // V2 router, rest V1
    d.routed_gate_up = DType::MQ4G256V2;
    d.routed_down = DType::MQ4G256V2;
    d.experts_all_gate_up_mq4 = false;
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.gate_fusable, "mixed V1/V2 gate quartet must not fuse V1");
    assert!(
        !r.gate_fusable_mq4v2,
        "mixed V1/V2 gate quartet must not fuse V2"
    );
    assert!(!r.gate_side_mq4);
    assert!(
        r.routed_indexable_mq4v2,
        "routed V2 indexability unchanged by gate mix"
    );
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local);

    // V1 router + one V2 shared half
    let mut d = dtypes_all_mq4();
    d.shared.as_mut().unwrap().up = DType::MQ4G256V2;
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.gate_fusable, "single V2 shared-up disqualifies V1 fuse");
    assert!(
        !r.gate_fusable_mq4v2,
        "single V2 shared-up disqualifies V2 fuse"
    );
}

#[test]
fn moe_res_shipped_ornith15_takes_the_indexed_path() {
    // The dtype combination of the PUBLISHED artifact
    // hipfire-models/ornith1.5-35b-a3b (read from its HFQ index: 20,651 of
    // 21,093 tensors are qt44, including every routed expert).
    //
    // Note the router and shared_expert_gate are Q8, not MQ4, so `gate_fusable`
    // is FALSE here — the fused gate-side GEMV does not apply. That does not
    // disqualify the routed path: the routed experts are uniform qt44, which is
    // what drives `use_gpu_topk`. Same coupling `moe_res_q8_router_still_gpu_topk`
    // pins for qt13.
    //
    // Before qt44 had indexed MoE GEMVs this resolved to use_gpu_topk=false and
    // the shipped model decoded through the resident CPU-fallback path.
    let mut d = dtypes_all_mq4();
    d.router = DType::Q8_0;
    d.shared = Some(MoeSharedDtypes {
        selector: DType::Q8_0,
        gate: DType::MQ6G256,
        up: DType::MQ6G256,
        down: DType::MQ6G256,
    });
    d.routed_gate_up = DType::MQ4G256V2;
    d.routed_down = DType::MQ4G256V2;
    d.experts_all_gate_up_mq4 = false;
    let r = MoeResolution::resolve(&d, 8);
    assert!(
        !r.gate_fusable,
        "Q8 router disqualifies the fused gate side"
    );
    assert!(r.routed_indexable_mq4v2, "routed experts are uniform qt44");
    assert!(
        r.use_gpu_topk,
        "shipped Ornith must take the indexed decode path"
    );
    assert!(r.needs_x_rot_local, "qt44 kernels read ROTATED activations");
}

#[test]
fn moe_res_mq4v2_mixed_with_qt13_is_not_indexable() {
    // The hazard this pairing guards: qt13 and qt44 share a 136 B group stride
    // and identical nibble packing, differing ONLY in the 8-byte header (one
    // f32 scale+zero vs two f16 scale/zero pairs). A kernel handed the wrong
    // one reads plausible garbage and emits fluent, wrong text rather than
    // faulting. So a split pairing must NOT be indexable on either arm.
    for (gu, dn) in [
        (DType::MQ4G256V2, DType::MQ4G256),
        (DType::MQ4G256, DType::MQ4G256V2),
    ] {
        let mut d = dtypes_all_mq4();
        d.routed_gate_up = gu;
        d.routed_down = dn;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.routed_indexable_mq4v2, "{gu:?}/{dn:?}");
        assert!(!r.routed_indexable_mq4, "{gu:?}/{dn:?}");
        assert!(!r.use_gpu_topk, "{gu:?}/{dn:?} must fall back, not guess");
    }
}

#[test]
fn moe_res_mq5_routed_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ5G256;
    d.routed_down = DType::MQ5G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq5);
    assert!(!r.routed_indexable_mq4);
    assert!(!r.routed_indexable_mq6);
    assert!(r.use_gpu_topk);
}

/// Per-projection Lloyd mix (gate_up MQ2-Lloyd, down MQ3-Lloyd) must reach the
/// GPU-top-K path. Before this arm existed the pair fell through every
/// `routed_indexable_*` test, so `use_gpu_topk` was false and the layer silently
/// took the CPU-top-K fallback — correct output, so the only symptom was lost
/// throughput. Guard both directions.
#[test]
fn moe_res_mixed_lloyd_routed_indexable() {
    for (gu, dn) in [
        (DType::MQ2G256Lloyd, DType::MQ3G256Lloyd),
        (DType::MQ3G256Lloyd, DType::MQ2G256Lloyd),
    ] {
        let mut d = dtypes_all_mq4();
        d.routed_gate_up = gu;
        d.routed_down = dn;
        let r = MoeResolution::resolve(&d, 8);
        assert!(r.routed_indexable_mixed_lloyd, "{gu:?}/{dn:?}");
        assert!(r.use_gpu_topk, "{gu:?}/{dn:?}");
        assert!(r.routed_indexable(), "{gu:?}/{dn:?}");
        // The uniform arms must NOT claim a mixed pair.
        assert!(!r.routed_indexable_mq2lloyd, "{gu:?}/{dn:?}");
        assert!(!r.routed_indexable_mq3lloyd, "{gu:?}/{dn:?}");
        // gate_up is Lloyd either way -> x must be rotated into the local buffer.
        assert!(r.needs_x_rot_local, "{gu:?}/{dn:?}");
    }
}

/// A Lloyd gate_up paired with a NON-Lloyd down must stay unindexable — there is
/// no MQ4/MQ6 down kernel that self-combines, and `routed_down_self_combines`
/// keys on `routed_down`, so admitting this would skip the shared down-combine
/// and silently drop every expert's contribution.
#[test]
fn moe_res_lloyd_gate_up_with_nonlloyd_down_not_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ2G256Lloyd;
    d.routed_down = DType::MQ4G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(!r.routed_indexable_mixed_lloyd);
    assert!(!r.routed_indexable());
    assert!(!r.use_gpu_topk);
}

/// MQ2/MQ3-G256-GL routed experts, in every per-projection combination with each
/// other AND with the Lloyd pair (all four are FwhtG256 codebook formats whose
/// down kernels self-combine, so any cross is executable).
///
/// The target SKU is gate_up = MQ2-GL + down = MQ3-GL. If the GL dtypes were
/// missing from `routed_indexable_mixed_lloyd`, `use_gpu_topk` would be false
/// and the layer would silently take the CPU-top-K fallback — CORRECT output,
/// so the only symptom is lost throughput. That is exactly the failure this
/// pins.
#[test]
fn moe_res_gl_routed_indexable() {
    use DType::{MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL};
    let codebook = [MQ2G256Lloyd, MQ3G256Lloyd, MQ2G256GL, MQ3G256GL];
    for gu in codebook {
        for dn in codebook {
            let mut d = dtypes_all_mq4();
            d.routed_gate_up = gu;
            d.routed_down = dn;
            let r = MoeResolution::resolve(&d, 8);
            assert!(r.routed_indexable_mixed_lloyd, "{gu:?}/{dn:?}");
            assert!(r.routed_indexable(), "{gu:?}/{dn:?}");
            assert!(r.use_gpu_topk, "{gu:?}/{dn:?}");
            // use_gpu_topk implies x_rot_local must exist — run_moe_decode
            // `.expect()`s it, and every codebook gate_up kernel reads x_rot.
            assert!(r.needs_x_rot_local, "{gu:?}/{dn:?}");
        }
    }
}

/// The GL dtypes must not leak into the MQ4/MQ5/MQ6 uniform arms, and a GL
/// gate_up with a non-self-combining down must stay unindexable (same
/// double-count / drop-out hazard as the Lloyd case above).
#[test]
fn moe_res_gl_gate_up_with_nonlloyd_down_not_indexable() {
    for dn in [DType::MQ4G256, DType::MQ6G256, DType::Q8_0] {
        let mut d = dtypes_all_mq4();
        d.routed_gate_up = DType::MQ2G256GL;
        d.routed_down = dn;
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.routed_indexable_mixed_lloyd, "{dn:?}");
        assert!(!r.routed_indexable_mq4, "{dn:?}");
        assert!(!r.routed_indexable_mq6, "{dn:?}");
        assert!(!r.routed_indexable(), "{dn:?}");
        assert!(!r.use_gpu_topk, "{dn:?}");
    }
}

/// The GL formats are FWHT-256 rotated (the encoder runs `cpu_fwht_256` per
/// 256-block before quantizing), so dispatch must classify them as FwhtG256 —
/// an un-rotated activation into a rotated weight is silent garbage.
#[test]
fn gl_dtypes_are_fwht_g256() {
    use crate::types::{dtype_needs_rotation, dtype_rotation_plan, RotationPlan};
    for dt in [DType::MQ2G256GL, DType::MQ3G256GL] {
        assert_eq!(dtype_rotation_plan(dt), RotationPlan::FwhtG256, "{dt:?}");
        assert!(dtype_needs_rotation(dt), "{dt:?}");
    }
}

/// GL has MoE-indexed kernels ONLY — no dense plain/prerotated GEMV exists. Both
/// key lookups must return a clean `UnsupportedVariant` rather than resolving to
/// some other dtype's kernel. (Note `for_gemv_prerotated`'s rotation-free
/// fallthrough must NOT catch these: they are FwhtG256, not None.)
#[test]
fn gl_dtypes_have_no_dense_gemv_key() {
    use crate::types::{GemvVariant, KernelKey};
    for dt in [DType::MQ2G256GL, DType::MQ3G256GL] {
        assert!(
            KernelKey::for_gemv(dt, GemvVariant::Plain, false).is_err(),
            "{dt:?} must not resolve a plain GEMV key"
        );
        assert!(
            KernelKey::for_gemv_prerotated(dt).is_err(),
            "{dt:?} must not resolve a prerotated GEMV key"
        );
    }
}

#[test]
fn moe_res_mq6_routed_indexable() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ6G256;
    d.routed_down = DType::MQ6G256;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq6);
    assert!(!r.routed_indexable_mq4);
    assert!(r.use_gpu_topk);
}

#[test]
fn moe_decode_oplist_prefix_matches_gate_side() {
    // The 4-way fused gate-side projection is capturable as a length-1 prefix.
    let oplist = [
        PipelineOp::MoeGateSideProj,
        PipelineOp::Softmax,
        PipelineOp::TopKRenorm,
        PipelineOp::SharedExpertDown,
        PipelineOp::IndexedGateUp,
        PipelineOp::SiluMulRotate,
        PipelineOp::IndexedDownExpanded,
        PipelineOp::MoeCombine,
    ];
    let fused = Pipeline::new(&[PipelineOp::MoeGateSideProj]);
    assert!(fused.can_satisfy(&oplist));
    let too_long = Pipeline::new(&[PipelineOp::MoeGateSideProj, PipelineOp::TopKRenorm]);
    assert!(!too_long.can_satisfy(&oplist)); // second op mismatches Softmax
}

#[test]
fn moe_res_paro_needs_sidecar() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::ParoQ4G128;
    d.routed_down = DType::ParoQ4G128;
    d.has_paro_shared = false;
    assert!(!MoeResolution::resolve(&d, 8).routed_indexable_paro);
    d.has_paro_shared = true;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_paro);
    assert!(r.use_gpu_topk);
}

// ── MQV2 resolver contracts (mixed gate precedence / MQ6V2 / D3) ─────────────
//
// Pure helpers exposed by `pipeline` + `MoeResolution` lattice. These pin the
// three silent-corruption hazards from the MQV2 fix wave: mixed gate before
// representative V1/V2 arms, uniform MQ6V2 indexability without V1 collapse,
// and V2 never calling the HFQ4 ninepath D3 gate.

use crate::pipeline::{
    decode_gate_uses_mixed, gate_up_varies, ninepath_d3_family, ninepath_d4_family,
    prefill_path1_down_kind_tag_aware, prefill_path1_gate_up_kind_tag_aware,
};

#[test]
fn moe_res_mq6v2_routed_indexable() {
    // qt47 uniform: BOTH projections MQ6G256V2 ⇒ indexable, GPU top-K on.
    // Dual-half f16 header is wire-incompatible with V1 MQ6G256's f32 header;
    // the arm must not claim the V1 mq6 or mq4/mq4v2 flags.
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ6G256V2;
    d.routed_down = DType::MQ6G256V2;
    d.experts_all_gate_up_mq4 = false;
    let r = MoeResolution::resolve(&d, 8);
    assert!(r.routed_indexable_mq6v2);
    assert!(!r.routed_indexable_mq6, "must not claim the V1 MQ6 arm");
    assert!(!r.routed_indexable_mq4);
    assert!(!r.routed_indexable_mq4v2);
    assert!(r.use_gpu_topk);
    assert!(r.needs_x_rot_local, "qt47 kernels read ROTATED activations");
    assert!(r.routed_indexable());
}

#[test]
fn moe_res_mq6v2_mixed_with_v1_is_not_indexable() {
    // Same dual-half hazard as mq4v2/qt13: V1 and V2 share the 200 B group
    // stride and 6-bit packing, differing ONLY in the 8-byte header. A split
    // pairing must NOT be indexable on either arm — wrong header is silent
    // fluent garbage, not a fault.
    for (gu, dn) in [
        (DType::MQ6G256V2, DType::MQ6G256),
        (DType::MQ6G256, DType::MQ6G256V2),
        (DType::MQ6G256V2, DType::MQ4G256),
        (DType::MQ4G256V2, DType::MQ6G256V2),
    ] {
        let mut d = dtypes_all_mq4();
        d.routed_gate_up = gu;
        d.routed_down = dn;
        d.experts_all_gate_up_mq4 = false;
        let r = MoeResolution::resolve(&d, 8);
        assert!(!r.routed_indexable_mq6v2, "{gu:?}/{dn:?}");
        assert!(!r.routed_indexable_mq6, "{gu:?}/{dn:?}");
        assert!(
            !r.use_gpu_topk,
            "{gu:?}/{dn:?} must fall back, not guess a layout"
        );
    }
}

#[test]
fn moe_res_mq6v2_k_ne_8_disables_gpu_topk() {
    let mut d = dtypes_all_mq4();
    d.routed_gate_up = DType::MQ6G256V2;
    d.routed_down = DType::MQ6G256V2;
    d.experts_all_gate_up_mq4 = false;
    let r = MoeResolution::resolve(&d, 6);
    assert!(r.routed_indexable_mq6v2);
    assert!(!r.use_gpu_topk);
}

/// Exact mixed V1/V2 gate precedence: whenever gate_up exact dtype varies,
/// the mixed gate kernel runs before any representative MQ4V2/MQ6V2/V1 arm.
/// Uniform shortcut is allowed only when every gate_up DType is equal.
#[test]
fn mixed_v1_v2_gate_precedence_requires_exact_variation() {
    // No table / all-equal table ⇒ no variation ⇒ uniform shortcut.
    assert!(!gate_up_varies(None));
    assert!(!gate_up_varies(Some(&[])));
    assert!(!gate_up_varies(Some(&[DType::MQ4G256V2, DType::MQ4G256V2])));
    assert!(!gate_up_varies(Some(&[DType::MQ6G256V2; 4])));

    // Exact DType inequality (V1 vs V2, or V2 vs V2 sibling) ⇒ varies.
    // Family-level sameness (both "MQ4") is NOT enough — headers differ.
    assert!(gate_up_varies(Some(&[DType::MQ4G256, DType::MQ4G256V2])));
    assert!(gate_up_varies(Some(&[DType::MQ4G256V2, DType::MQ6G256V2])));
    assert!(gate_up_varies(Some(&[
        DType::MQ4G256V2,
        DType::MQ4G256V2,
        DType::MQ4G256,
    ])));

    // Mixed gate fires only when tags exist AND gate_up varies.
    assert!(decode_gate_uses_mixed(true, true));
    assert!(
        !decode_gate_uses_mixed(true, false),
        "tags alone must not force mixed when every gate_up DType is equal"
    );
    assert!(
        !decode_gate_uses_mixed(false, true),
        "variation without a tag table has no mixed kernel to dispatch"
    );
    assert!(!decode_gate_uses_mixed(false, false));

    // Path1 tag-aware kinds: mixed precedes representative V1/V2 arms.
    assert_eq!(
        prefill_path1_gate_up_kind_tag_aware(DType::MQ4G256V2, true, true),
        Some("mixed"),
        "varying gate_up + tags ⇒ mixed, not mq4v2 representative"
    );
    assert_eq!(
        prefill_path1_gate_up_kind_tag_aware(DType::MQ6G256V2, true, true),
        Some("mixed"),
        "varying gate_up + tags ⇒ mixed, not mq6v2 representative"
    );
    assert_eq!(
        prefill_path1_gate_up_kind_tag_aware(DType::MQ4G256, true, true),
        Some("mixed"),
        "varying gate_up + tags ⇒ mixed, not hfq4 representative"
    );
    // Uniform shortcut only with exact equality (no variation).
    assert_eq!(
        prefill_path1_gate_up_kind_tag_aware(DType::MQ4G256V2, true, false),
        Some("mq4v2"),
        "tags + uniform gate_up may take the V2 uniform arm"
    );
    assert_eq!(
        prefill_path1_gate_up_kind_tag_aware(DType::MQ6G256V2, false, false),
        Some("mq6v2")
    );
    // Path1 down: any tag table forces the mixed down launcher — never a
    // representative V1/V2 dispatch of a tagged layer.
    assert_eq!(
        prefill_path1_down_kind_tag_aware(DType::MQ4G256V2, true),
        Some("mixed")
    );
    assert_eq!(
        prefill_path1_down_kind_tag_aware(DType::MQ6G256V2, true),
        Some("mixed")
    );
    assert_eq!(
        prefill_path1_down_kind_tag_aware(DType::MQ6G256V2, false),
        Some("mq6v2")
    );
}

/// V2 D3 restriction: only HFQ4/MQ4V1 may call `gemv_hfq4g256_moe_ninepath_d3`.
/// V2 uniform pairs use exact native indexed gate + V2 D4 — never the HFQ4 D3
/// gate (dual-half header is silent fluent corruption on the same stride).
#[test]
fn ninepath_d3_restricts_v2_to_native_gate_and_d4() {
    // Sole admitted D3 pair.
    assert_eq!(
        ninepath_d3_family(DType::MQ4G256, DType::MQ4G256),
        Some("hfq4")
    );

    // V2 uniforms have D4 families but NEVER a D3 family.
    assert_eq!(ninepath_d3_family(DType::MQ4G256V2, DType::MQ4G256V2), None);
    assert_eq!(ninepath_d3_family(DType::MQ6G256V2, DType::MQ6G256V2), None);
    assert_eq!(
        ninepath_d4_family(DType::MQ4G256V2, DType::MQ4G256V2),
        Some("mq4v2"),
        "V2 still has its own D4 path"
    );
    assert_eq!(
        ninepath_d4_family(DType::MQ6G256V2, DType::MQ6G256V2),
        Some("mq6v2"),
        "V2 still has its own D4 path"
    );

    // Lloyd D4 pair is not D3 either (D3 is HFQ4-only).
    assert_eq!(
        ninepath_d3_family(DType::MQ2G256Lloyd, DType::MQ3G256Lloyd),
        None
    );

    // Split V1/V2 pairings share neither D3 nor D4.
    for (g, dn) in [
        (DType::MQ4G256V2, DType::MQ4G256),
        (DType::MQ4G256, DType::MQ4G256V2),
        (DType::MQ6G256V2, DType::MQ6G256),
        (DType::MQ4G256V2, DType::MQ6G256V2),
    ] {
        assert_eq!(ninepath_d3_family(g, dn), None, "{g:?}/{dn:?}");
        assert_eq!(ninepath_d4_family(g, dn), None, "{g:?}/{dn:?}");
    }
}

// ── op-list interpreter: match_prefix (pure logic) ──────────────────────────

use crate::families::gemv::WeightRef;
use crate::pipeline::steps::{match_prefix, GemvInput};
use crate::pipeline::{FusedPattern, Step};

fn dummy_wr<'a>(t: &'a rdna_compute::GpuTensor) -> WeightRef<'a> {
    WeightRef {
        buf: t,
        dtype: rdna_compute::DType::F32,
        m: 1,
        k: 1,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn gemv_step<'a>(t: &'a rdna_compute::GpuTensor, wr: &'a WeightRef<'a>) -> Step<'a> {
    Step::Gemv {
        w: wr,
        input: GemvInput::Raw(t),
        out: t,
    }
}

#[test]
fn match_prefix_empty_table_never_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    assert_eq!(match_prefix(&[], &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_picks_longest() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    let table = [
        FusedPattern {
            ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
            key: KernelKey::GemvF32,
            guard: |_, _| true,
        },
        FusedPattern {
            ops: &[PipelineOp::Gemv, PipelineOp::Gemv, PipelineOp::Gemv],
            key: KernelKey::GemvF16,
            guard: |_, _| true,
        },
    ];
    assert_eq!(
        match_prefix(&table, &steps, &ctx_rdna3()),
        Some((KernelKey::GemvF16, 3))
    );
}

#[test]
fn match_prefix_no_pattern_longer_than_steps() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| true,
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_single_op_consumes_one() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| true,
    }];
    // a len-1 pattern matches the first step, consuming exactly 1
    assert_eq!(
        match_prefix(&table, &steps, &ctx_rdna3()),
        Some((KernelKey::GemvF32, 1))
    );
}

#[test]
fn match_prefix_guard_false_blocks_match() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [gemv_step(&dummy, &wr), gemv_step(&dummy, &wr)];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |_, _| false, // always reject
    }];
    assert_eq!(match_prefix(&table, &steps, &ctx_rdna3()), None);
}

#[test]
fn match_prefix_guard_receives_correct_window() {
    // Guard inspects window length — verifies it gets exactly ops.len() steps.
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = dummy_wr(&dummy);
    let steps = [
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
        gemv_step(&dummy, &wr),
    ];
    let table = [FusedPattern {
        ops: &[PipelineOp::Gemv, PipelineOp::Gemv],
        key: KernelKey::GemvF32,
        guard: |window, _| window.len() == 2, // must see exactly 2 steps
    }];
    assert_eq!(
        match_prefix(&table, &steps, &ctx_rdna3()),
        Some((KernelKey::GemvF32, 2))
    );
}

// ── FUSED_TABLE guard tests ──────────────────────────────────────────────────

use crate::pipeline::steps::{
    guard_gate_up_hfq4g256, guard_gate_up_hfq6g256, guard_gate_up_mq3g256lloyd,
    guard_gate_up_mq4g256lloyd, guard_gate_up_mq4g256v2, guard_qkv_hfq4g256, guard_qkv_hfq6g256,
    guard_qkv_mq3g256lloyd, guard_qkv_mq4g256lloyd, guard_qkv_mq4g256v2, guard_qkvza_mq4g256v2,
    match_fused_prefix,
};

fn make_qkv3_steps<'a>(
    dummy: &'a rdna_compute::GpuTensor,
    wr: &'a WeightRef<'a>,
    rotation: RotationPlan,
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x: dummy,
            norm_weight: dummy,
            x_plain: dummy,
            out: dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation,
            rows: 1,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
    ]
}

fn make_gate_up2_steps<'a>(
    dummy: &'a rdna_compute::GpuTensor,
    wr: &'a WeightRef<'a>,
    rotation: RotationPlan,
) -> Vec<Step<'a>> {
    vec![
        Step::RmsnormAutomatic {
            x: dummy,
            norm_weight: dummy,
            x_plain: dummy,
            out: dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation,
            rows: 1,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
        Step::Gemv {
            w: wr,
            input: GemvInput::Prerotated(dummy),
            out: dummy,
        },
    ]
}

#[test]
fn guard_qkv_mq4g256lloyd_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_wrong_dtype() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::HFQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::None);
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_awq_scale() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: Some(&dummy),
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }; // AWQ present → reject
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_mq4g256lloyd_rejects_force_unfused() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    let mut ctx = ctx_rdna3();
    std::sync::Arc::make_mut(&mut ctx.flags).force_unfused = true;
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx));
}

#[test]
fn guard_qkv_hfq4g256_covers_mq4g256() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_hfq4g256(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_hfq4g256_covers_hfq4g256() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::HFQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::None);
    assert!(guard_qkv_hfq4g256(&steps, &ctx_rdna3()));
}

#[test]
fn guard_qkv_hfq6g256_dp4a_decoupled() {
    // Post-merge: 458's commit f478d9b6 ("MQ6 dp4a-decouple") removed the
    // `dp4a_eligible` gate from `guard_qkv_hfq6g256` — HFQ6/MQ6 QKV fusion is
    // safe + beneficial on RDNA3+ even without dp4a (the None arm falls back to
    // gemm n=1). So RDNA3 (gfx1100) now FIRES the guard where master had it
    // blocked. Only `force_unfused` or a non-uniform / wrong-length step window
    // suppresses it now.
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr_hfq6 = WeightRef {
        buf: &dummy,
        dtype: DType::HFQ6G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let wr_mq6 = WeightRef {
        buf: &dummy,
        dtype: DType::MQ6G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps_hfq6 = make_qkv3_steps(&dummy, &wr_hfq6, RotationPlan::FwhtG256);
    let steps_mq6 = make_qkv3_steps(&dummy, &wr_mq6, RotationPlan::FwhtG256);

    // gfx906 (dp4a) still fires.
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_gfx906()));
    assert!(guard_qkv_hfq6g256(&steps_mq6, &ctx_gfx906()));
    // RDNA3 (gfx1100) now ALSO fires (dp4a decoupled).
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_rdna3()));
    assert!(guard_qkv_hfq6g256(&steps_mq6, &ctx_rdna3()));
    // RDNA1 (gfx1010) fires too — the guard no longer gates on dp4a; the
    // fused_qkv None arm safely lowers to gemm n=1 on any arch.
    assert!(guard_qkv_hfq6g256(&steps_hfq6, &ctx_rdna1()));
}

#[test]
fn guard_qkv_rejects_mixed_gemv_input() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = vec![
        Step::RmsnormAutomatic {
            x: &dummy,
            norm_weight: &dummy,
            x_plain: &dummy,
            out: &dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation: RotationPlan::FwhtG256,
            rows: 1,
        },
        Step::Gemv {
            w: &wr,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
        Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&dummy),
            out: &dummy,
        }, // mixed!
        Step::Gemv {
            w: &wr,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
    ];
    assert!(!guard_qkv_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn guard_gate_up_mq4g256lloyd_fires() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256Lloyd,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_gate_up2_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_gate_up_mq4g256lloyd(&steps, &ctx_rdna3()));
}

#[test]
fn match_fused_prefix_admits_exact_mq4g256v2_qkv() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256V2,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_qkv_mq4g256v2(&steps, &ctx_rdna3()));
    assert_eq!(
        match_fused_prefix(&steps, &ctx_rdna3()),
        Some((KernelKey::FusedQkvMq4G256V2, 4))
    );
}

#[test]
fn match_fused_prefix_admits_exact_mq4g256v2_qkvza() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256V2,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    // QKVZA = QKV3 window + one extra Gemv (reuse builder, no new abstraction).
    let mut steps = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    steps.push(Step::Gemv {
        w: &wr,
        input: GemvInput::Prerotated(&dummy),
        out: &dummy,
    });
    assert!(guard_qkvza_mq4g256v2(&steps, &ctx_rdna3()));
    assert_eq!(
        match_fused_prefix(&steps, &ctx_rdna3()),
        Some((KernelKey::FusedQkvzaMq4G256V2, 5))
    );
}

#[test]
fn match_fused_prefix_admits_exact_mq4g256v2_gate_up() {
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256V2,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = make_gate_up2_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(guard_gate_up_mq4g256v2(&steps, &ctx_rdna3()));
    assert_eq!(
        match_fused_prefix(&steps, &ctx_rdna3()),
        Some((KernelKey::FusedGateUpMq4G256V2, 3))
    );
}

#[test]
fn match_fused_prefix_rejects_mixed_v1_v2_mq4_window() {
    // Mixed V1/V2 must not admit any exact V2 fused key (clean exact-dtype only).
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr_v2 = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256V2,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let wr_v1 = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let steps = vec![
        Step::RmsnormAutomatic {
            x: &dummy,
            norm_weight: &dummy,
            x_plain: &dummy,
            out: &dummy,
            awq_scale: None,
            k: 4096,
            eps: 1e-6,
            rotation: RotationPlan::FwhtG256,
            rows: 1,
        },
        Step::Gemv {
            w: &wr_v2,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
        Step::Gemv {
            w: &wr_v1,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
        Step::Gemv {
            w: &wr_v2,
            input: GemvInput::Prerotated(&dummy),
            out: &dummy,
        },
    ];
    assert!(!guard_qkv_mq4g256v2(&steps, &ctx_rdna3()));
    let got = match_fused_prefix(&steps, &ctx_rdna3());
    assert!(
        !matches!(
            got,
            Some((
                KernelKey::FusedQkvMq4G256V2
                    | KernelKey::FusedQkvzaMq4G256V2
                    | KernelKey::FusedGateUpMq4G256V2,
                _
            ))
        ),
        "mixed V1/V2 must not admit V2 fused key, got {got:?}"
    );
}

#[test]
fn match_fused_prefix_rejects_mq4g256v2_on_unsupported_arch() {
    // Exact V2 windows must not admit scalar V2 fusion off gfx1100/gfx1201.
    // gfx1200 is a near-miss RDNA4 sibling — still fail-closed.
    let dummy = rdna_compute::GpuTensor::null_for_test();
    let wr = WeightRef {
        buf: &dummy,
        dtype: DType::MQ4G256V2,
        m: 4096,
        k: 4096,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let ctx = ctx_rdna4(); // gfx1200 — not gfx1201
    let qkv = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(!guard_qkv_mq4g256v2(&qkv, &ctx));
    assert!(
        !matches!(
            match_fused_prefix(&qkv, &ctx),
            Some((KernelKey::FusedQkvMq4G256V2, _))
        ),
        "unsupported arch must not admit FusedQkvMq4G256V2"
    );
    let mut qkvza = make_qkv3_steps(&dummy, &wr, RotationPlan::FwhtG256);
    qkvza.push(Step::Gemv {
        w: &wr,
        input: GemvInput::Prerotated(&dummy),
        out: &dummy,
    });
    assert!(!guard_qkvza_mq4g256v2(&qkvza, &ctx));
    assert!(
        !matches!(
            match_fused_prefix(&qkvza, &ctx),
            Some((KernelKey::FusedQkvzaMq4G256V2, _))
        ),
        "unsupported arch must not admit FusedQkvzaMq4G256V2"
    );
    let gate_up = make_gate_up2_steps(&dummy, &wr, RotationPlan::FwhtG256);
    assert!(!guard_gate_up_mq4g256v2(&gate_up, &ctx));
    assert!(
        !matches!(
            match_fused_prefix(&gate_up, &ctx),
            Some((KernelKey::FusedGateUpMq4G256V2, _))
        ),
        "unsupported arch must not admit FusedGateUpMq4G256V2"
    );
}

// ── MoePrefillResolution cells (Ship 4.2) ─────────────────────────

use crate::families::moe::MoePrefillResolution;

/// Helper: default MoeDtypes for MQ4 routed experts (the common A3B case).
fn moe_dtypes_mq4() -> MoeDtypes<'static> {
    MoeDtypes {
        router: DType::Q8_0,
        shared: Some(MoeSharedDtypes {
            selector: DType::Q8_0,
            gate: DType::MQ4G256,
            up: DType::MQ4G256,
            down: DType::MQ4G256,
        }),
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256,
        routed_down: DType::MQ4G256,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    }
}

fn moe_dtypes_mq6() -> MoeDtypes<'static> {
    let mut d = moe_dtypes_mq4();
    d.routed_gate_up = DType::MQ6G256;
    d.routed_down = DType::MQ6G256;
    d.experts_all_gate_up_mq4 = false;
    d
}

fn moe_dtypes_paro() -> MoeDtypes<'static> {
    let mut d = moe_dtypes_mq4();
    d.routed_gate_up = DType::ParoQ4G128;
    d.routed_down = DType::ParoQ4G128;
    d.experts_all_gate_up_mq4 = false;
    d.has_paro_shared = true;
    d
}

#[test]
fn mq3v25_route_requires_marker_flag_multirow_and_exact_arch() {
    use crate::families::gemm::mq3v25_route;
    for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx942"] {
        let mut flags = rdna_compute::FeatureFlags::for_test(arch);
        flags.iu4_prefill = Some(true);
        let default_on = arch == "gfx1201";
        assert_eq!(mq3v25_route(&flags, true, 8192, 4096), default_on, "{arch}");
        assert!(!mq3v25_route(&flags, false, 8192, 4096), "{arch}");
        assert!(!mq3v25_route(&flags, true, 1, 4096), "{arch}");

        flags.mq3v25_a4 = Some(true);
        let forced_on = matches!(arch, "gfx1100" | "gfx1201");
        assert_eq!(mq3v25_route(&flags, true, 8192, 4096), forced_on, "{arch}");
        assert!(!mq3v25_route(&flags, false, 8192, 4096), "{arch}");
        assert!(!mq3v25_route(&flags, true, 1, 4096), "{arch}");
        if arch == "gfx1100" {
            assert!(!mq3v25_route(&flags, true, 8192, 4100));
            assert!(!mq3v25_route(&flags, true, 64, 4096));
        }
        if arch == "gfx1201" {
            // Speculative-verify widths stay on the fp16 WMMA route; A4 from 17 rows.
            for n in [2, 3, 4, 8, 16] {
                assert!(!mq3v25_route(&flags, true, n, 4096), "n={n}");
            }
            for n in [17, 24, 64, 8192] {
                assert!(mq3v25_route(&flags, true, n, 4096), "n={n}");
            }
        }

        flags.mq3v25_a4 = Some(false);
        assert!(!mq3v25_route(&flags, true, 8192, 4096), "{arch}");

        flags.mq3v25_a4 = Some(true);
        flags.iu4_prefill = Some(false);
        assert!(!mq3v25_route(&flags, true, 8192, 4096), "{arch}");
    }
}

#[test]
fn mq3v25_a8_route_requires_flag_symmetric_multirow_and_exact_arch() {
    use crate::families::gemm::mq3v25_a8_route;
    for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx942"] {
        let mut flags = rdna_compute::FeatureFlags::for_test(arch);
        assert!(!mq3v25_a8_route(&flags, true, 8192), "{arch}");
        flags.mq3v25_a8 = true;
        assert_eq!(
            mq3v25_a8_route(&flags, true, 8192),
            arch == "gfx1201",
            "{arch}"
        );
        assert!(!mq3v25_a8_route(&flags, false, 8192), "{arch}");
        assert!(!mq3v25_a8_route(&flags, true, 1), "{arch}");
    }
}

fn flags_default() -> rdna_compute::feature_flags::FeatureFlags {
    rdna_compute::feature_flags::FeatureFlags::for_test("gfx1100")
}

#[test]
fn moe_prefill_resolution_path2_gfx11_mq4() {
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx11 should have Path 2 (WMMA)");
    assert!(!r.down_path0, "gfx11 should not be Path 0");
    assert!(!r.paro_mode);
    assert!(!r.use_paro_i8);
    assert!(!r.use_paro_i8_k8);
}

#[test]
fn moe_prefill_resolution_path2_gfx12_mq4() {
    let arch = crate::context::DispatchCtx::for_test("gfx1200");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx12 should have Path 2 (WMMA)");
    assert!(!r.down_path0);
}

#[test]
fn moe_prefill_resolution_path2_gfx12_mq6() {
    let arch = crate::context::DispatchCtx::for_test("gfx1200");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx12 should have Path 2 for MQ6");
    assert!(!r.paro_mode);
}

#[test]
fn moe_prefill_resolution_gfx1151_mixed_mq6_fences_mq4_i8() {
    let arch = crate::context::DispatchCtx::for_test("gfx1151");

    let pure_mq4 = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(pure_mq4.use_path2);
    assert!(
        !pure_mq4.force_mq4_grouped_fp16,
        "pure MQ4 layers should keep gfx1151's existing grouped-i8 default"
    );

    let mixed_mq6 = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(mixed_mq6.use_path2);
    assert!(
        mixed_mq6.force_mq4_grouped_fp16,
        "MQ6-promoted/mixed A3B layers must not run remaining MQ4 projections through grouped-i8 by default"
    );
}

#[test]
fn moe_prefill_resolution_path2_gfx11_paro() {
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_paro(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "gfx11 should have Path 2 for Paro");
    assert!(r.paro_mode);
    assert!(!r.use_paro_i8, "gfx1100 is not gfx1151 — no i8");
}

#[test]
fn moe_prefill_resolution_path2_gfx1151_paro_i8() {
    let arch = crate::context::DispatchCtx::for_test("gfx1151");
    let r = MoePrefillResolution::resolve(&moe_dtypes_paro(), &arch.arch, &arch.flags);
    assert!(r.use_path2);
    assert!(r.paro_mode);
    assert!(r.use_paro_i8, "gfx1151 should default to i8 for Paro");
    assert!(r.use_paro_i8_k8, "gfx1151 should default to i8 k8 for Paro");
}

#[test]
fn moe_prefill_resolution_path1_gfx1030() {
    let arch = crate::context::DispatchCtx::for_test("gfx1030");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(!r.use_path2, "gfx1030 has no WMMA — no Path 2");
    assert!(!r.down_path0, "gfx1030 is not gfx9 — no Path 0");
}

#[test]
fn moe_prefill_resolution_path0_gfx906() {
    let arch = crate::context::DispatchCtx::for_test("gfx906");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(!r.use_path2, "gfx906 has no WMMA — no Path 2");
    assert!(r.down_path0, "gfx906 should be Path 0 (atomic GEMV)");
}

#[test]
fn moe_prefill_resolution_path0_gfx942() {
    let arch = crate::context::DispatchCtx::for_test("gfx942");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(!r.use_path2, "gfx942 has no WMMA — no Path 2");
    assert!(r.down_path0, "gfx942 (CDNA3) should be Path 0");
}

#[test]
fn moe_prefill_resolution_grouped_gemm_opt_out() {
    let mut flags = flags_default();
    flags.moe_grouped_gemm = false;
    let flags = std::sync::Arc::new(flags);
    let caps = rdna_compute::arch_caps::ArchCaps::new("gfx1100", flags.clone());
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &caps, &flags);
    assert!(!r.use_path2, "moe_grouped_gemm=0 should disable Path 2");
}

#[test]
fn moe_prefill_resolution_paro_i8_opt_out() {
    let mut flags = flags_default();
    flags.moe_paro_i8 = Some(false);
    let flags = std::sync::Arc::new(flags);
    let caps = rdna_compute::arch_caps::ArchCaps::new("gfx1151", flags.clone());
    let r = MoePrefillResolution::resolve(&moe_dtypes_paro(), &caps, &flags);
    assert!(r.use_path2);
    assert!(r.paro_mode);
    assert!(!r.use_paro_i8, "moe_paro_i8=0 should disable i8");
    assert!(!r.use_paro_i8_k8);
}

#[test]
fn moe_prefill_resolution_mq6_gfx11_uses_path2() {
    // Post-merge: the gfx11 MQ6 grouped-WMMA `_k2` kernel now exists, so the
    // 458 widen (8d555fc6) admits Path 2 on any wmma_w32 gfx11 arch (gfx1100
    // included), superseding master's earlier gfx12-only fallback assertion.
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(
        r.use_path2,
        "MQ6 on gfx11 should use Path 2 (gfx11 `_k2` grouped WMMA exists)"
    );
    assert!(!r.down_path0, "gfx11 is not Path 0");
}

#[test]
fn moe_prefill_resolution_mq6_gfx1151_uses_path2() {
    let arch = crate::context::DispatchCtx::for_test("gfx1151");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(
        r.use_path2,
        "MQ6 on gfx1151 should use Path 2 (grouped WMMA available)"
    );
    assert!(!r.down_path0);
}

#[test]
fn moe_prefill_resolution_mq6_gfx12_uses_path2() {
    let arch = crate::context::DispatchCtx::for_test("gfx1200");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq6(), &arch.arch, &arch.flags);
    assert!(
        r.use_path2,
        "MQ6 on gfx12 should use Path 2 (grouped WMMA available)"
    );
    assert!(!r.down_path0);
}

#[test]
fn moe_prefill_resolution_mq4_gfx11_still_path2() {
    let arch = crate::context::DispatchCtx::for_test("gfx1100");
    let r = MoePrefillResolution::resolve(&moe_dtypes_mq4(), &arch.arch, &arch.flags);
    assert!(r.use_path2, "MQ4 on gfx11 should still use Path 2");
}

// ── Uniform codebook routed pair (MQ2-Lloyd gate_up / MQ3-Lloyd down) ────────
//
// The shape every current low-bit a3b SKU ships (`--format mq4-mqlloyd-antirez`,
// no k-map ⇒ `expert_dtype_tags == None`). Batched prefill for it is Path 2 only:
// `run_moe_prefill`'s Path 0/1 indexed-GEMV matches have no MQ2/MQ3-Lloyd arm and
// return `UnsupportedVariant`, so these tests pin BOTH that Path 2 is selected on
// every WMMA arch AND that each grouped-GEMM leg names a real entry point.

/// The antirez asymmetric routed pair: gate_up MQ2-Lloyd, down MQ3-Lloyd,
/// MQ4 shared expert, no tag table.
fn moe_dtypes_codebook_pair() -> MoeDtypes<'static> {
    let mut d = moe_dtypes_mq4();
    d.routed_gate_up = DType::MQ2G256Lloyd;
    d.routed_down = DType::MQ3G256Lloyd;
    d.experts_all_gate_up_mq4 = false;
    d
}

#[test]
fn moe_prefill_resolution_codebook_pair_uses_path2_on_wmma() {
    for arch_name in [
        "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
    ] {
        let arch = crate::context::DispatchCtx::for_test(arch_name);
        let r = MoePrefillResolution::resolve(&moe_dtypes_codebook_pair(), &arch.arch, &arch.flags);
        assert!(
            r.use_path2,
            "{arch_name}: MQ2L/MQ3L routed pair must take Path 2 (grouped WMMA) — \
             Path 0/1 have no Lloyd indexed-GEMV arm"
        );
        assert!(!r.down_path0, "{arch_name} is not a Path 0 arch");
        assert!(!r.paro_mode);
    }
}

/// Non-WMMA archs fall out of Path 2. There is no Lloyd Path 0/1 arm, which is
/// exactly why `codebook_batched_admit_enabled_from_env` (hipfire-arch-qwen35)
/// refuses to admit these layers on such archs — they must never get here.
#[test]
fn moe_prefill_resolution_codebook_pair_no_path2_without_wmma() {
    for arch_name in ["gfx1010", "gfx1030", "gfx906", "gfx942"] {
        let arch = crate::context::DispatchCtx::for_test(arch_name);
        let r = MoePrefillResolution::resolve(&moe_dtypes_codebook_pair(), &arch.arch, &arch.flags);
        assert!(
            !r.use_path2,
            "{arch_name} has no grouped-WMMA kernel — must not resolve to Path 2"
        );
    }
}

/// Both grouped-GEMM legs of the pair must name a REAL `extern "C" __global__`
/// entry point that exists in the source string the launcher will hand to the
/// JIT — on both arch legs. This is the no-GPU proof that the new dispatch arms
/// resolve to actual kernels rather than to a fallback, a stale name, or a
/// gfx11 kernel wrongly reused on RDNA4.
#[test]
fn codebook_grouped_gemm_legs_name_real_kernels() {
    let legs = [
        (
            "MQ2-Lloyd",
            false,
            rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(false),
        ),
        (
            "MQ2-Lloyd",
            true,
            rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(true),
        ),
        (
            "MQ3-Lloyd",
            false,
            rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(false),
        ),
        (
            "MQ3-Lloyd",
            true,
            rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(true),
        ),
    ];
    for (label, is_gfx12, (name, src)) in legs {
        let entry = format!("__global__ void {name}(");
        assert!(
            src.contains(&entry),
            "{label} is_gfx12={is_gfx12}: source does not define `{name}`"
        );
        // The gfx11 leg must use the gfx11 WMMA builtin and the gfx12 leg the
        // `_gfx12` one — swapping them fails the JIT at first launch, which on
        // a prefill path means a serving outage, not a fallback.
        if is_gfx12 {
            assert!(
                src.contains("wmma_f32_16x16x16_f16_w32_gfx12"),
                "{label} gfx12 leg must use the _gfx12 WMMA builtin"
            );
        } else {
            assert!(
                src.contains("wmma_f32_16x16x16_f16_w32")
                    && !src.contains("wmma_f32_16x16x16_f16_w32_gfx12"),
                "{label} gfx11 leg must use the plain _w32 WMMA builtin"
            );
        }
    }
    // Distinct module names per arch — the JIT cache is keyed by module name
    // only, so a collision makes one arch's source dead code.
    assert_ne!(
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(false).0,
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(true).0
    );
    assert_ne!(
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(false).0,
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(true).0
    );
}

/// The MQ2-Lloyd and MQ3-Lloyd grouped kernels must agree with the uniform
/// grouped kernarg contract the scatter pipeline emits: 5 pointers then
/// `M, K, x_row_div, m_total`. A tenth arg (as in the merged dtype-tag kernel,
/// which inserts `dtype_tags` second) would shift every kernarg by 8 bytes.
#[test]
fn codebook_grouped_gemm_kernarg_contract_is_the_uniform_nine() {
    for (name, src) in [
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(false),
        rdna_compute::mq2g256_lloyd_moe_grouped_wmma_source(true),
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(false),
        rdna_compute::mq3g256_lloyd_moe_grouped_wmma_source(true),
    ] {
        let start = src
            .find(&format!("__global__ void {name}("))
            .unwrap_or_else(|| panic!("{name}: entry point not found"));
        let sig_end = src[start..].find(')').expect("unterminated signature") + start;
        let sig = &src[start..sig_end];
        assert!(
            !sig.contains("dtype_tags"),
            "{name}: uniform grouped kernels must NOT take a dtype_tags table"
        );
        for arg in [
            "expert_weight_ptrs",
            "expert_tile_ids",
            "sorted_slot_index",
            "X_src",
            "Y_grouped",
            "int M",
            "int K",
            "int x_row_div",
            "int m_total",
        ] {
            assert!(sig.contains(arg), "{name}: signature is missing `{arg}`");
        }
    }
}

// ── CPU-executed offload (memory.offload_exec=cpu) ────────────────────────────

/// The seam's coverage table and the launcher's rotation table must agree: the
/// CPU path applies `rotate_x` to a `Raw` activation exactly when the launcher
/// would, so a dtype that maps to a `CpuQuant` while `dtype_rotation_plan` says
/// something else would silently feed unrotated activations to rotated weights
/// (or double-rotate them) — the silent-garbage failure mode, not an error.
#[test]
fn cpu_quant_rotation_agrees_with_plan() {
    for &dtype in crate::CPU_EXECUTED_DTYPES {
        let q = crate::cpu_quant_for(dtype)
            .unwrap_or_else(|| panic!("{dtype:?} has a CPU decoder and must be in cpu_quant_for"));
        let want = match dtype_rotation_plan(dtype) {
            RotationPlan::FwhtG256 => hipfire_cpu::quant::InputRotation::FwhtG256,
            RotationPlan::FwhtG128 => hipfire_cpu::quant::InputRotation::FwhtG128V2,
            RotationPlan::None => hipfire_cpu::quant::InputRotation::None,
            other => panic!("{dtype:?}: plan {other:?} has no CPU rotation"),
        };
        assert_eq!(
            q.input_rotation(),
            want,
            "{dtype:?}: CPU rotation disagrees with dtype_rotation_plan"
        );
        assert_eq!(
            q.is_fwht_g256(),
            dtype_rotation_plan(dtype) == RotationPlan::FwhtG256,
            "{dtype:?}: is_fwht_g256 disagrees with dtype_rotation_plan"
        );
        assert_eq!(
            crate::cpu_exec::raw_input_is_rotated(dtype),
            want != hipfire_cpu::quant::InputRotation::None,
            "{dtype:?}: seam rotates a Raw input differently from the launcher"
        );
    }
}

/// Formats the CPU must refuse, including ones that look adjacent to a covered
/// format. Each of these has its own layout (codebook size, bit width, or a
/// non-FWHT rotation) that `hipfire-cpu` does not implement, so a step over one
/// stays on the GPU over PCIe.
#[test]
fn cpu_quant_refuses_unimplemented_formats() {
    // The constant declares exactly what `cpu_quant_for` executes: every entry
    // must map to `Some`, so the declaration cannot drift from the table.
    for &dtype in crate::CPU_EXECUTED_DTYPES {
        assert!(
            crate::cpu_quant_for(dtype).is_some(),
            "{dtype:?} is declared CPU-executed but has no decoder in cpu_quant_for"
        );
    }
    for dtype in [
        DType::MQ8G256, // RotationPlan::Mq8Internal: int8 activation, not FWHT-f32
        DType::MQ4G128, // RotationPlan::FwhtG128: a different transform
        DType::HFP4G32, // per-row 16 B header + per-32 block scales
        DType::MFP4G32,
        DType::MFP4G32Lloyd,
        DType::MFP4G32P,
        DType::MFP4G32E8SOA,
        DType::ParoQ4G128, // Givens rotation on the activation
        DType::Q4K,        // no dense HFQ artifact uses it
        DType::Q8HFQ,      // padded rows (row_stride), not expressible in the group model
        DType::Raw,
    ] {
        assert!(
            !crate::CPU_EXECUTED_DTYPES.contains(&dtype),
            "{dtype:?} is refused and must not be declared CPU-executed"
        );
        assert!(
            crate::cpu_quant_for(dtype).is_none(),
            "{dtype:?} must not be CPU-executed until its decode is transcribed"
        );
    }
}

/// The Redline refusal is a conjunction: it must fire for the configured
/// conflict and for nothing else, or a plain CPU-exec run would fail to load.
#[test]
fn cpu_exec_redline_conflict_is_exactly_the_conjunction() {
    use crate::cpu_exec_redline_conflict as conflict;
    assert!(
        conflict(true, 12, true),
        "cpu + spill + replay is the conflict"
    );
    assert!(!conflict(false, 12, true), "pcie must never be refused");
    assert!(
        !conflict(true, 0, true),
        "nothing spilled, nothing on the CPU"
    );
    assert!(
        !conflict(true, 12, false),
        "no replay controller: no conflict"
    );
}

/// #478: gfx1030 + MQ6G256/HFQ6G256 crashed with "no implementation for
/// GemvMq6G256Prerotated" because the GEMV rows were gated `HasMmq`
/// (gfx906/RDNA3/RDNA4). The kernels are generic wave32 scalar, so every GEMV
/// variant must resolve on RDNA1/RDNA2, and the previously admitted archs must
/// keep resolving. CDNA wave64 (no MMQ, not wave32) stays excluded.
#[test]
fn mq6_hfq6_gemv_resolves_on_rdna1_rdna2_and_keeps_prior_archs() {
    use crate::families::gemv::GemvFamily;
    let fam = GemvFamily::new();
    let variants = [
        GemvVariant::Plain,
        GemvVariant::Prerotated,
        GemvVariant::WithResidual,
        GemvVariant::WithSwiGLUResidual,
    ];
    for arch in [
        "gfx1010", "gfx1030", "gfx1031", "gfx1032", "gfx906", "gfx1100", "gfx1151", "gfx1200",
        "gfx1201",
    ] {
        let ctx = DispatchCtx::for_test(arch);
        for dtype in [DType::MQ6G256, DType::HFQ6G256] {
            for variant in variants {
                // MQ-family weights are only reached pre-rotated; HFQ6 has no
                // Prerotated key (plain GEMV), so skip an unmapped pairing.
                if variant == GemvVariant::Prerotated && dtype == DType::HFQ6G256 {
                    continue;
                }
                let r = fam.resolve(dtype, variant, false, &ctx, None);
                assert!(
                    r.is_ok(),
                    "{dtype:?}/{variant:?} must resolve on {arch}: {r:?}"
                );
            }
        }
    }
    let cdna = DispatchCtx::for_test("gfx942");
    assert!(fam
        .resolve(DType::MQ6G256, GemvVariant::Prerotated, false, &cdna, None)
        .is_err());
}

/// HIPF-222: `GemvHfq3G128` was gated on sdot4, so gfx1151/RDNA3 resolved
/// MissingImpl and LLaMA hfq3 files failed at the first generate. The kernel is
/// wave32 scalar: every wave32 arch resolves, CDNA wave64 stays excluded.
#[test]
fn hfq3g128_gemv_resolves_on_wave32_only() {
    use crate::families::gemv::GemvFamily;
    let fam = GemvFamily::new();
    for arch in ["gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1201"] {
        let r = fam.resolve(
            DType::HFQ3G128,
            GemvVariant::Plain,
            false,
            &DispatchCtx::for_test(arch),
            None,
        );
        assert!(r.is_ok(), "HFQ3G128 must resolve on {arch}: {r:?}");
    }
    for arch in ["gfx906", "gfx942"] {
        let r = fam.resolve(
            DType::HFQ3G128,
            GemvVariant::Plain,
            false,
            &DispatchCtx::for_test(arch),
            None,
        );
        assert!(r.is_err(), "HFQ3G128 must stay rejected on {arch}");
    }
}

#[test]
fn arch_has_hfq6_gemv_is_mmq_union_wave32() {
    assert!(ArchPredicate::HasHfq6Gemv.eval_arch(&ctx_rdna1()));
    assert!(ArchPredicate::HasHfq6Gemv.eval_arch(&ctx_rdna2()));
    assert!(ArchPredicate::HasHfq6Gemv.eval_arch(&ctx_rdna3()));
    assert!(ArchPredicate::HasHfq6Gemv.eval_arch(&ctx_rdna4()));
    assert!(ArchPredicate::HasHfq6Gemv.eval_arch(&ctx_gfx906()));
    assert!(!ArchPredicate::HasHfq6Gemv.eval_arch(&DispatchCtx::for_test("gfx942")));
}
