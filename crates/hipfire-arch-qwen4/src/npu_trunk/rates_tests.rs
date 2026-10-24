// SPDX-License-Identifier: Apache-2.0
//! The live rate estimator against Flash-Next's built-in calibration.
use super::{load_calibration, Family, FamilyMask, FLASH_NEXT};
use hipfire_runtime::coproc::*;

const ROUTE: &str = "overlay:gfx1151-zab-v1";
fn builtin() -> Calibration {
    let (cal, source) = load_calibration("/nonexistent/x.json", ROUTE).unwrap();
    assert_eq!(source, "builtin");
    cal
}
fn flash_next_kinds() -> Vec<usize> {
    let kinds: Vec<_> = (0..48).map(|l| if l % 4 == 3 { 1 } else { 0 }).collect();
    assert_eq!(kinds.iter().filter(|k| **k == 0).count(), 36);
    assert_eq!(kinds.iter().filter(|k| **k == 1).count(), 12);
    kinds
}
fn json(cal: &Calibration) -> serde_json::Value {
    serde_json::to_value(cal).unwrap()
}
fn sample(window_ratio: f64, npu_ratio: f64) -> RateSample {
    RateSample {
        npu_measured_ms: 100.0 * npu_ratio,
        npu_model_ms: 100.0,
        window_measured_ms: 100.0 * window_ratio,
        window_model_ms: 100.0,
    }
}

#[test]
fn new_snapshot_is_the_prior() {
    let prior = builtin();
    let e = RateEstimator::new(prior.clone());
    let s = e.snapshot();
    assert_eq!(json(&s.calibration), json(&prior));
    assert_eq!(json(e.prior()), json(&prior));
    assert_eq!(
        (s.generation, s.gpu_scale, s.npu_scale, s.samples, s.frozen),
        (0, 1.0, 1.0, 0, false)
    );
}

#[test]
fn unit_scales_reproduce_the_prior_and_scaled_output_validates() {
    for route in [ROUTE, "stock"] {
        let (prior, _) = load_calibration("/nonexistent/x.json", route).unwrap();
        assert_eq!(json(&scaled_calibration(&prior, 1.0, 1.0)), json(&prior));
        for (g, n) in [
            (0.5, 2.0),
            (2.0, 0.5),
            (1.3, 0.8),
            (SCALE_MIN, SCALE_MIN),
            (SCALE_MAX, SCALE_MAX),
        ] {
            let scaled = scaled_calibration(&prior, g, n);
            scaled.validate(&FLASH_NEXT).unwrap();
            assert_eq!(scaled.key, prior.key);
            assert_eq!(scaled.margin, prior.margin);
        }
    }
    let prior = builtin();
    let scaled = scaled_calibration(&prior, 2.0, 0.5);
    assert_eq!(
        scaled.gpu.uniform_per_layer_ms,
        prior.gpu.uniform_per_layer_ms * 2.0
    );
    assert_eq!(scaled.gpu.min_bank_rows, prior.gpu.min_bank_rows);
    assert_eq!(scaled.glue.sync_ms, prior.glue.sync_ms * 0.5);
    assert_eq!(scaled.npu.tops_corun, prior.npu.tops_corun * 2.0);
    assert_eq!(scaled.npu.bytes_per_kop, prior.npu.bytes_per_kop);
    assert_eq!(
        scaled.contention.alpha_overlap,
        prior.contention.alpha_overlap
    );
}

#[test]
fn pair_model_matches_plan_and_segments() {
    let cal = builtin();
    let kinds = flash_next_kinds();
    let model = pair_model(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::ALL.into()),
        [8192, 8192],
        [0, 8192],
        &kinds,
    );
    let p = plan(&FLASH_NEXT, &cal, FamilyMask::ALL.into(), 16384);
    assert!(
        (model.gpu_only_ms - p.t_gpu_ms).abs() < 1e-6,
        "{} vs {}",
        model.gpu_only_ms,
        p.t_gpu_ms
    );
    let a = bank_segments_moved(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::ALL.into()),
        8192,
        0,
    );
    let b = bank_segments_moved(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::ALL.into()),
        8192,
        8192,
    );
    assert!(a.npu[0] > 0.0 && a.npu[1] > 0.0);
    let expected_npu = 36.0 * (a.npu[0] + b.npu[0]) + 12.0 * (a.npu[1] + b.npu[1]);
    assert!(
        (model.npu_ms - expected_npu).abs() < 1e-6,
        "{} vs {expected_npu}",
        model.npu_ms
    );
    // Window: every chain but (0, A); (l, A) l >= 1 is hidden by bank B of layer l-1, (l, B) by bank A of layer l.
    let mut expected_window = 0.0;
    for (l, kind) in kinds.iter().enumerate() {
        let k = *kind;
        if l >= 1 {
            let w = b.gpu[kinds[l - 1]];
            expected_window += w + contention_extra(&cal, w, a.npu[k], a.bytes[k]);
        }
        let w = a.gpu[k];
        expected_window += w + contention_extra(&cal, w, b.npu[k], b.bytes[k]);
    }
    assert!(
        (model.window_ms - expected_window).abs() < 1e-6,
        "{} vs {expected_window}",
        model.window_ms
    );
    assert!(model.window_ms > 0.0);
    // Nothing moved: no chains, no window.
    let none = pair_model(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::NONE.into()),
        [8192, 8192],
        [0, 8192],
        &kinds,
    );
    assert_eq!((none.npu_ms, none.window_ms), (0.0, 0.0));
    assert!((none.gpu_only_ms - p.t_gpu_ms).abs() < 1e-6);
    assert!(none.savings_ms.abs() < 1e-9);
    // savings is exactly what the planner's pair time credits the pipe: t = gpu_only - savings + stall + fill.
    let pr = pair_time_moved(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::ALL.into()),
        8192,
        0,
        8192,
        8192,
    );
    assert!(
        (pr.t - (model.gpu_only_ms - model.savings_ms + pr.stall + pr.fill)).abs() < 1e-6,
        "{pr:?} {model:?}"
    );
    // At the built-in rates the modelled NPU waits (stall + fill) stay far below the saving: no guard trip.
    assert!(
        pr.stall + pr.fill < 0.5 * model.savings_ms,
        "{pr:?} {model:?}"
    );
    // Only GDN in-proj moved: QSA layers contribute no chain.
    let qkv = pair_model(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::from(Family::GdnQkv).into()),
        [8192, 8192],
        [0, 8192],
        &kinds,
    );
    let qa = bank_segments_moved(
        &FLASH_NEXT,
        &cal,
        Moved::of(&FLASH_NEXT, FamilyMask::from(Family::GdnQkv).into()),
        8192,
        0,
    );
    assert_eq!(qa.npu[1], 0.0);
    assert!(qkv.npu_ms > 0.0 && qkv.npu_ms < model.npu_ms);
}

#[test]
fn constant_ratios_converge_and_freeze() {
    let mut e = RateEstimator::new(builtin());
    assert!(e.observe(&sample(1.3, 0.8)));
    assert_eq!(e.snapshot().generation, 1);
    for _ in 0..12 {
        e.observe(&sample(1.3, 0.8));
    }
    let s = e.snapshot();
    assert!(
        (s.gpu_scale / 1.3 - 1.0).abs() < 0.01 && (s.npu_scale / 0.8 - 1.0).abs() < 0.01,
        "{s:?}"
    );
    assert!(s.frozen);
    assert_eq!(s.samples, 13);
    assert_eq!(
        json(&s.calibration),
        json(&scaled_calibration(e.prior(), s.gpu_scale, s.npu_scale))
    );
    let generation = s.generation;
    // A frozen snapshot does not move on samples inside the reopen band.
    assert!(!e.observe(&sample(1.33, 0.8)));
    assert_eq!(e.snapshot().generation, generation);
}

#[test]
fn npu_thirty_percent_slower_than_the_prior_converges_and_drops_zab() {
    // A stock-kernel host whose NPU leg runs x1.3 the built-in model at depth (fivetide's 57K), after short
    // requests that matched the prior: from fresh, and from a snapshot already frozen at the prior.
    let zab = |m: FamilyMask| m.contains(Family::GdnZ);
    for frozen_first in [false, true] {
        let mut e = RateEstimator::new(builtin());
        for _ in 0..if frozen_first { FREEZE_AFTER } else { 2 } {
            e.observe(&sample(1.0, 1.0));
        }
        assert_eq!(e.snapshot().frozen, frozen_first);
        for _ in 0..20 {
            e.observe(&sample(1.0, 1.3));
        }
        let s = e.snapshot();
        assert!(
            (s.npu_scale / 1.3 - 1.0).abs() < 0.02 && (s.gpu_scale - 1.0).abs() < 0.01,
            "{s:?}"
        );
        let chosen = FamilyMask::from(choose_families(
            &FLASH_NEXT,
            &s.calibration,
            FamilyMask::ALL.into(),
            57_552,
        ));
        assert!(
            !zab(chosen) && chosen.contains(Family::GdnQkv) && chosen.contains(Family::QsaQ),
            "{chosen:?}"
        );
        let p = plan_executable(&FLASH_NEXT, &s.calibration, chosen.into(), 57_552);
        assert!(p.t_pipe_ms < p.t_gpu_ms);
    }
}

#[test]
fn unit_ratios_freeze_without_changing_the_snapshot() {
    let mut e = RateEstimator::new(builtin());
    for i in 1..=FREEZE_AFTER {
        assert!(!e.observe(&sample(1.0, 1.0)));
        assert_eq!(e.snapshot().frozen, i >= FREEZE_AFTER);
    }
    assert_eq!(e.snapshot().generation, 0);
}

#[test]
fn ratios_are_clamped() {
    let mut e = RateEstimator::new(builtin());
    assert!(e.observe(&sample(10.0, 0.01)));
    assert_eq!(
        (e.snapshot().gpu_scale, e.snapshot().npu_scale),
        (SCALE_MAX, SCALE_MIN)
    );
    e.observe(&sample(0.01, 10.0));
    assert!(e.snapshot().gpu_scale <= SCALE_MAX && e.snapshot().gpu_scale >= SCALE_MIN);
    assert!(e.snapshot().npu_scale <= SCALE_MAX && e.snapshot().npu_scale >= SCALE_MIN);
    e.snapshot().calibration.validate(&FLASH_NEXT).unwrap();
}

#[test]
fn invalid_samples_are_ignored() {
    let mut e = RateEstimator::new(builtin());
    let good = sample(1.3, 0.8);
    let bad = [
        RateSample {
            npu_measured_ms: f64::NAN,
            ..good
        },
        RateSample {
            npu_model_ms: f64::INFINITY,
            ..good
        },
        RateSample {
            window_measured_ms: f64::NEG_INFINITY,
            ..good
        },
        RateSample {
            window_model_ms: f64::NAN,
            ..good
        },
        RateSample {
            npu_model_ms: MIN_MODEL_MS - 0.1,
            ..good
        },
        RateSample {
            window_model_ms: 0.0,
            ..good
        },
        RateSample {
            npu_measured_ms: 0.0,
            ..good
        },
        RateSample {
            window_measured_ms: -5.0,
            ..good
        },
    ];
    for s in bad {
        assert!(!e.observe(&s), "{s:?}");
    }
    let snap = e.snapshot();
    assert_eq!(
        (
            snap.samples,
            snap.generation,
            snap.frozen,
            snap.gpu_scale,
            snap.npu_scale
        ),
        (0, 0, false, 1.0, 1.0)
    );
    assert!(e.observe(&RateSample {
        npu_model_ms: MIN_MODEL_MS,
        window_model_ms: MIN_MODEL_MS,
        npu_measured_ms: 40.0,
        window_measured_ms: 60.0
    }));
    assert_eq!(e.snapshot().samples, 1);
}

#[test]
fn frozen_snapshot_reopens_after_consistent_deviations() {
    let mut e = RateEstimator::new(builtin());
    for _ in 0..10 {
        e.observe(&sample(1.3, 0.8));
    }
    assert!(e.snapshot().frozen);
    let generation = e.snapshot().generation;
    // Alternating directions never confirm.
    for i in 0..6 {
        assert!(!e.observe(&sample(if i % 2 == 0 { 1.7 } else { 0.9 }, 0.8)));
    }
    assert!(e.snapshot().frozen);
    assert!(!e.observe(&sample(1.3, 0.8)));
    for _ in 0..REOPEN_CONFIRM - 1 {
        assert!(!e.observe(&sample(1.7, 0.8)));
    }
    assert!(e.snapshot().frozen && e.snapshot().generation == generation);
    assert!(e.observe(&sample(1.7, 0.8)));
    let s = e.snapshot();
    assert!(!s.frozen);
    assert_eq!(s.generation, generation + 1);
    assert!(s.gpu_scale > 1.3);
    s.calibration.validate(&FLASH_NEXT).unwrap();
    // Unfrozen again, it follows the new rate and settles.
    for _ in 0..40 {
        e.observe(&sample(1.7, 0.8));
    }
    let s = e.snapshot();
    // Freezing is judged on per-sample EWMA movement, so the settled scale sits within a few percent of the rate.
    assert!((s.gpu_scale / 1.7 - 1.0).abs() < 0.05 && s.frozen, "{s:?}");
}

#[test]
fn slow_npu_drops_zab_and_nominal_rates_keep_all() {
    let prior = builtin();
    let all = FamilyMask::ALL;
    assert_eq!(
        FamilyMask::from(choose_families(&FLASH_NEXT, &prior, all.into(), 16384)),
        all
    );
    let slow = scaled_calibration(&prior, 1.0, 1.25);
    let chosen = FamilyMask::from(choose_families(&FLASH_NEXT, &slow, all.into(), 16384));
    for f in [Family::GdnZ, Family::GdnA, Family::GdnB] {
        assert!(!chosen.contains(f), "{chosen:?}");
    }
    for f in [
        Family::GdnQkv,
        Family::QsaQ,
        Family::QsaK,
        Family::QsaV,
        Family::QsaIdx,
    ] {
        assert!(chosen.contains(f), "{chosen:?}");
    }
    // Restricted to a subset, the choice stays inside it; nothing allowed -> NONE.
    let qsa_only = FamilyMask::from(choose_families(
        &FLASH_NEXT,
        &prior,
        FamilyMask::QSA.into(),
        16384,
    ));
    assert_eq!(qsa_only.bits() & !FamilyMask::QSA.bits(), 0);
    assert_eq!(
        FamilyMask::from(choose_families(
            &FLASH_NEXT,
            &prior,
            FamilyMask::NONE.into(),
            16384
        )),
        FamilyMask::NONE
    );
}

#[test]
fn scaled_requests_plan_executable_pairs_where_the_rebanked_model_wins() {
    // Halo gate, cand-auto r1: after one 16K pair the scales were gpu x0.981 / npu x0.892, and the full model
    // preferred the re-banked plan for 57,552 rows, which the executor projects to GPU chunks: the request ran
    // GPU-only. Request planning uses `plan_executable`, which keeps the natural and rebalanced pairs.
    let cal = scaled_calibration(&builtin(), 0.981, 0.892);
    let rows = 57_552;
    let pipes = |p: &Plan| {
        p.steps
            .iter()
            .filter(|s| matches!(s, Step::Pipe { .. }))
            .count()
    };
    let full = executable_plan_for(
        &FLASH_NEXT,
        &cal,
        FamilyMask::ALL.into(),
        &plan(&FLASH_NEXT, &cal, FamilyMask::ALL.into(), rows),
        rows,
        Some(0),
    );
    assert_eq!(
        pipes(&full),
        0,
        "the re-banked model plan has no executable pair"
    );
    let natural = plan_executable(&FLASH_NEXT, &cal, FamilyMask::ALL.into(), rows);
    let executable = executable_plan_for(
        &FLASH_NEXT,
        &cal,
        FamilyMask::ALL.into(),
        &natural,
        rows,
        Some(0),
    );
    assert_eq!(pipes(&executable), 4);
    assert!(executable.t_pipe_ms < executable.t_gpu_ms);
    // The unscaled prior plans the same pairs either way.
    let prior = builtin();
    assert_eq!(
        plan(&FLASH_NEXT, &prior, FamilyMask::ALL.into(), rows),
        plan_executable(&FLASH_NEXT, &prior, FamilyMask::ALL.into(), rows)
    );
}
