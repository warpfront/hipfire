// CPU-only adapter regressions. Every calibration and price below is synthetic;
// these tests neither qualify a GPU route nor constrain future measured tables.
use super::*;

fn certain_counts(
    _: DerivedCard,
    _: DerivedCostMode,
    source: DerivedSource,
    _: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    if source == DerivedSource::Mtp {
        assert_eq!(feature, DerivedFeature::Native);
    }
    Some(CalibrationCounts {
        depth_s: 10_000,
        depth_n: 10_000,
        cell_s: 0,
        cell_n: 0,
    })
}

fn eighty_counts(
    _: DerivedCard,
    _: DerivedCostMode,
    source: DerivedSource,
    _: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    assert_eq!(source, DerivedSource::Mtp);
    assert_eq!(feature, DerivedFeature::Native);
    Some(CalibrationCounts {
        depth_s: 8_000,
        depth_n: 10_000,
        cell_s: 0,
        cell_n: 0,
    })
}

fn cheap_price(
    _: DerivedCard,
    _: DerivedCostMode,
    _: DerivedSource,
    _: ContextBucket,
    k: u8,
    _: bool,
) -> Option<(f64, bool)> {
    Some((100.0 + f64::from(k), false))
}

fn no_ar(_: DerivedCard, _: DerivedCostMode, _: ContextBucket) -> Option<(f64, bool)> {
    None
}

fn expensive_ar(_: DerivedCard, _: DerivedCostMode, _: ContextBucket) -> Option<(f64, bool)> {
    Some((1_000.0, false))
}

const WIDE: DerivedTables = DerivedTables {
    counts: certain_counts,
    window_price: cheap_price,
    ar_price: expensive_ar,
};

fn greedy() -> SpecRequestConfig {
    SpecRequestConfig {
        temp: 0.0,
        ..SpecRequestConfig::default()
    }
}

fn calibrated(default_k: usize) -> Qwen4MtpDrafter {
    let mut drafter = Qwen4MtpDrafter::new(default_k, 4096, None);
    drafter.floor.seed_calibrated(30_000.0);
    drafter.floor.seed_probes_done();
    drafter
}

fn bind(drafter: &mut Qwen4MtpDrafter, rows: usize, tables: DerivedTables) {
    drafter.derived_configure(&greedy(), true);
    drafter.derived_bind(Ok(DerivedCard::R9700), tables);
    drafter.derived_refresh_rows(rows);
    drafter.derived_position = Some(1500);
}

fn bound(default_k: usize, rows: usize, tables: DerivedTables) -> Qwen4MtpDrafter {
    let mut drafter = calibrated(default_k);
    bind(&mut drafter, rows, tables);
    drafter
}

fn state(drafter: &Qwen4MtpDrafter) -> &DerivedState {
    match &drafter.derived {
        Some(DerivedPhase::Bound(state)) => state,
        _ => panic!("expected an eligible bound adapter"),
    }
}

fn pending(drafter: &Qwen4MtpDrafter) -> DerivedPending {
    state(drafter)
        .pending
        .expect("expected a priced consultation")
}

fn plan(drafter: &mut Qwen4MtpDrafter, k: usize) -> WindowPlan {
    drafter
        .plan_window_with(&[0, 1], k, None, false)
        .expect("CPU-only planning")
}

fn native_plan(k: usize) -> WindowPlan {
    WindowPlan {
        route: WindowRoute::Native { depth: k.min(1) },
        k,
        trace: false,
        floor_active: true,
        ngram_hit: false,
        ngram_confirm: true,
        ngram_candidates: Vec::new(),
    }
}

// Isolate the adapter's objective from the incumbent floor's source/routing
// decision. The script supplies a real prefix-shaped offer, not GPU outcomes.
fn consult(drafter: &mut Qwen4MtpDrafter, plan: &mut WindowPlan, caller_k: usize) {
    let Some(DerivedPhase::Bound(mut state)) = drafter.derived.take() else {
        panic!("expected a bound adapter");
    };
    state.pending = None;
    drafter.derived_consult(&mut state, plan, caller_k, 1500);
    drafter.derived = Some(DerivedPhase::Bound(state));
}

fn arm_pool(drafter: &mut Qwen4MtpDrafter, copy: bool) {
    let config = NgramModConfig {
        capacity: 4096,
        n_match: 2,
        n_min: 1,
        n_max: 63,
    };
    // Production sizes the agreement table to the pool's proposal domain
    // (`with_ngram`); the scripted arm mirrors that bound, not a new cap.
    let depths = config.n_max.min(NGRAM_DEPTHS);
    let prompt: Vec<u32> = (0..96).collect();
    drafter.ngram_config = Some(config);
    drafter.copy_lane = copy;
    if copy {
        let mut pool = MtpCopyContext::new(config).expect("scripted copy pool");
        pool.begin_request(&prompt);
        drafter.ngram_copy = Some(pool);
    } else {
        let mut pool = MtpNgramContext::new(config).expect("scripted ngram pool");
        pool.begin_request(&prompt);
        drafter.ngram = Some(pool);
    }
    drafter.ngram_active = true;
    drafter.row_exact_rows = 64;
    drafter.ngram_row_cost = Some((30.5, 6.9));
    drafter.ngram_agreement = vec![(19.8, 20.0); depths];
    drafter.ngram_last_first_accepted = true;
    drafter.ngram_yield = (20.0, 5.0);
    drafter.stats.ngram_mod_windows = 5;
    drafter.stats.ngram_mod_drafts = 15;
    drafter.stats.ngram_mod_accepted = 15;
}

#[test]
fn load_defaults_k1_k3_k10_do_not_cap_qualified_derived_widths() {
    for default_k in [1, 3, 10] {
        for rows in [8, 64] {
            let mut drafter = bound(default_k, rows, WIDE).with_explicit_mtp_k_override(None);
            drafter.spec_rows_cap = 64;
            let cap = rows - 1;
            // The incumbent's tables are admitted-capacity sized: the load
            // default K bounded by the executor's qualified rows (never by
            // the derived widths, which exceed it, as asserted below).
            assert_eq!(drafter.incumbent_max_k(), default_k.min(MAX_DEPTH).min(cap));
            assert_eq!(drafter.explicit_mtp_k, None);
            assert_eq!((drafter.k(), drafter.proposal_capacity()), (cap, cap));
            let result = plan(&mut drafter, cap);
            assert_eq!(result.route, WindowRoute::Native { depth: cap });
            assert_eq!(result.k, cap);
            assert_eq!(pending(&drafter).chosen, Some(cap as u8));
            assert_eq!(pending(&drafter).legal, cap);
        }
    }
}

#[test]
fn explicit_k_and_caller_budget_bound_native_and_both_pool_modes() {
    for rows in [8, 64] {
        for explicit in [1, 3, 10, 63, usize::MAX] {
            for source in [DerivedSource::Mtp, DerivedSource::Ngram] {
                for copy in [false, true] {
                    let mut drafter =
                        bound(1, rows, WIDE).with_explicit_mtp_k_override(Some(explicit));
                    drafter.spec_rows_cap = 64;
                    if source == DerivedSource::Ngram {
                        arm_pool(&mut drafter, copy);
                    }
                    let legal = explicit.min(rows - 1).min(63);
                    assert_eq!(drafter.derived_legal_cap(source), legal);
                    assert_eq!(drafter.proposal_capacity(), legal);
                    for caller_k in [1, 2, 7, 63] {
                        let width = legal.min(caller_k);
                        let mut window = native_plan(1);
                        if source == DerivedSource::Ngram {
                            drafter.ngram_proposal = (2..65).collect();
                            drafter.ngram_candidates = vec![2];
                            window.ngram_hit = true;
                        }
                        consult(&mut drafter, &mut window, caller_k);
                        let decision = pending(&drafter);
                        assert_eq!(decision.source, source);
                        assert_eq!(decision.chosen, Some(width as u8));
                        assert_eq!(usize::from(decision.hi), width);
                        assert_eq!((decision.legal, decision.caller_k), (legal, caller_k));
                        assert!(state(&drafter)
                            .prices
                            .iter()
                            .all(|p| usize::from(p.k) <= width));
                        if source == DerivedSource::Mtp {
                            assert_eq!(window.route, WindowRoute::Native { depth: width });
                        } else {
                            assert_eq!(drafter.ngram_candidates, drafter.ngram_proposal[..width]);
                            assert!(window.ngram_confirm);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ngram_offer_and_zero_remaining_budget_are_independent_bounds() {
    let mut drafter = bound(1, 64, WIDE);
    drafter.ngram_proposal = vec![2, 3, 4, 5];
    drafter.ngram_candidates = vec![2];
    let mut window = native_plan(1);
    window.ngram_hit = true;
    consult(&mut drafter, &mut window, 63);
    assert_eq!(pending(&drafter).chosen, Some(4));
    assert_eq!(drafter.ngram_candidates, vec![2, 3, 4, 5]);

    let mut drafter = bound(10, 64, WIDE);
    let result = plan(&mut drafter, 0);
    assert_eq!(result.k, 0);
    assert!(!matches!(result.route, WindowRoute::Native { depth } if depth > 0));
    assert!(!result.ngram_hit);
    assert!(state(&drafter).pending.is_none());
}

#[test]
fn qualified_rows_zero_or_one_and_explicit_zero_never_offer_drafts() {
    for rows in [0, 1, 64] {
        for pool in [None, Some(false), Some(true)] {
            // The copy lane is a cost-mode input stamped at bind: arming it
            // after bind is a mode flip, which invalidates the adapter.
            let mut drafter = calibrated(10);
            if let Some(copy) = pool {
                arm_pool(&mut drafter, copy);
            }
            bind(&mut drafter, rows, WIDE);
            if rows == 64 {
                drafter = drafter.with_explicit_mtp_k_override(Some(0));
            }
            assert_eq!(drafter.derived_legal_cap(DerivedSource::Mtp), 0);
            assert_eq!(drafter.derived_legal_cap(DerivedSource::Ngram), 0);
            assert_eq!((drafter.k(), drafter.proposal_capacity()), (0, 0));
            let result = plan(&mut drafter, 0);
            assert_eq!(result.k, 0);
            assert!(!matches!(result.route, WindowRoute::Native { depth } if depth > 0));
            assert!(!result.ngram_hit);
            assert!(drafter.ngram_candidates.is_empty());
            assert!(state(&drafter).pending.is_none());
            assert_eq!(state(&drafter).consults, 0);
        }
    }
}

#[test]
fn cpu_four_row_splice_cap_survives_wide_qualification_for_every_source() {
    for pool in [None, Some(false), Some(true)] {
        let mut drafter = bound(1, 64, WIDE);
        if let Some(copy) = pool {
            arm_pool(&mut drafter, copy);
        }
        drafter.spec_rows_cap = 4;
        assert_eq!((drafter.k(), drafter.proposal_capacity()), (3, 3));
        assert_eq!(drafter.derived_legal_cap(DerivedSource::Mtp), 3);
        assert_eq!(drafter.derived_legal_cap(DerivedSource::Ngram), 3);
        let mut window = native_plan(1);
        if pool.is_some() {
            drafter.ngram_proposal = (2..65).collect();
            drafter.ngram_candidates = vec![2];
            window.ngram_hit = true;
        }
        consult(&mut drafter, &mut window, 63);
        assert_eq!(pending(&drafter).chosen, Some(3));
        assert_eq!(pending(&drafter).legal, 3);
        if pool.is_some() {
            assert_eq!(drafter.ngram_candidates, vec![2, 3, 4]);
        } else {
            assert_eq!(window.route, WindowRoute::Native { depth: 3 });
        }
    }
}

fn profile_price(
    _: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    _: ContextBucket,
    k: u8,
    confirmation: bool,
) -> Option<(f64, bool)> {
    if source != DerivedSource::Ngram {
        return None;
    }
    let best = match (mode, confirmation) {
        (DerivedCostMode::OffPool, true) => 1,
        (DerivedCostMode::OffPool, false) => 3,
        (DerivedCostMode::OnCopy, true) => 5,
        (DerivedCostMode::OnCopy, false) => 7,
    };
    Some((
        if k == best {
            100.0 + f64::from(best)
        } else {
            1_000_000.0
        },
        false,
    ))
}

fn one_profile_price(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    bucket: ContextBucket,
    k: u8,
    confirmation: bool,
) -> Option<(f64, bool)> {
    if mode == DerivedCostMode::OffPool && confirmation {
        profile_price(card, mode, source, bucket, k, confirmation)
    } else {
        None
    }
}

#[test]
fn price_lookup_distinguishes_cost_mode_and_pre_window_confirmation() {
    let tables = DerivedTables {
        counts: certain_counts,
        window_price: profile_price,
        ar_price: no_ar,
    };
    for (copy, confirmation, chosen) in [
        (false, true, 1),
        (false, false, 3),
        (true, true, 5),
        (true, false, 7),
    ] {
        let mut drafter = bound(1, 8, tables).with_copy_lane(copy);
        drafter.ngram_proposal = (2..9).collect();
        drafter.ngram_candidates = vec![2];
        let mut window = native_plan(1);
        window.ngram_hit = true;
        window.ngram_confirm = confirmation;
        consult(&mut drafter, &mut window, 7);
        let decision = pending(&drafter);
        assert_eq!(
            decision.mode,
            if copy {
                DerivedCostMode::OnCopy
            } else {
                DerivedCostMode::OffPool
            }
        );
        assert_eq!(decision.confirmation, confirmation);
        assert_eq!(decision.chosen, Some(chosen));
        assert_eq!(decision.chosen_cost_us, 100.0 + f64::from(chosen));
        assert_eq!(window.ngram_confirm, confirmation);
        assert!(window.ngram_hit);
        assert_eq!(
            drafter.ngram_candidates,
            drafter.ngram_proposal[..usize::from(chosen)]
        );
    }
}

#[test]
fn missing_mode_confirmation_or_source_profile_never_borrows_another_price() {
    let tables = DerivedTables {
        counts: certain_counts,
        window_price: one_profile_price,
        // An AR price alone must not turn an unsupported window into a consult.
        ar_price: expensive_ar,
    };
    for (copy, confirmation, ngram) in [
        (true, true, true),
        (false, false, true),
        (true, false, true),
        (false, true, false),
    ] {
        let mut drafter = bound(3, 8, tables).with_copy_lane(copy);
        drafter.ngram_proposal = (2..9).collect();
        drafter.ngram_candidates = vec![2, 3];
        let mut window = native_plan(3);
        window.ngram_hit = ngram;
        window.ngram_confirm = confirmation;
        let incumbent = window.clone();
        let candidates = drafter.ngram_candidates.clone();
        consult(&mut drafter, &mut window, 7);
        assert_eq!(window, incumbent);
        assert_eq!(drafter.ngram_candidates, candidates);
        assert_eq!(state(&drafter).consults, 0);
        assert!(state(&drafter).pending.is_none());
        assert!(state(&drafter).prices.is_empty());
    }
}

fn nonmonotone_price(
    _: DerivedCard,
    _: DerivedCostMode,
    _: DerivedSource,
    _: ContextBucket,
    k: u8,
    _: bool,
) -> Option<(f64, bool)> {
    match k {
        1 => Some((30_000.0, false)),
        2 => Some((45_000.0, false)),
        3 => Some((40_000.0, false)),
        _ => None,
    }
}

const NONMONOTONE: DerivedTables = DerivedTables {
    counts: eighty_counts,
    window_price: nonmonotone_price,
    ar_price: no_ar,
};

#[test]
fn global_nonmonotone_resolve_crosses_a_bad_adjacent_step_without_recharging_heads() {
    let mut drafter = bound(1, 8, NONMONOTONE);
    let mut window = native_plan(1);
    consult(&mut drafter, &mut window, 7);
    assert_eq!(window.route, WindowRoute::Native { depth: 3 });
    assert_eq!(pending(&drafter).chosen, Some(3));
    let p = state(&drafter)
        .request
        .probability(&state(&drafter).depths[0]);
    let expected1 = 1.0 + p;
    let expected2 = expected1 + p * p;
    let expected3 = expected2 + p * p * p;
    assert!(expected2 / 45_000.0 < expected1 / 30_000.0);
    assert!(expected3 / 40_000.0 > expected1 / 30_000.0);
    for drafted in 1..=3 {
        assert_eq!(drafter.derived_resolve(drafted), drafted < 3);
        let decision = pending(&drafter);
        assert_eq!(decision.chosen, Some(3));
        assert_eq!(decision.chosen_cost_us, 40_000.0);
        assert!((decision.chosen_expected - expected3).abs() < 1e-12);
        assert!(state(&drafter)
            .resolve
            .iter()
            .all(|p| usize::from(p.k) >= drafted));
        for price in &state(&drafter).resolve {
            assert_eq!(
                price.cost_us,
                nonmonotone_price(
                    DerivedCard::R9700,
                    DerivedCostMode::OffPool,
                    DerivedSource::Mtp,
                    ContextBucket::Le2K,
                    price.k,
                    true,
                )
                .unwrap()
                .0
            );
        }
    }
    assert!(state(&drafter)
        .keys
        .iter()
        .all(|key| key.feature == DerivedFeature::Native));
}

#[test]
fn observed_residuals_change_the_global_solution_but_resolve_cannot_discard_drafted_rows() {
    let mut drafter = bound(1, 8, NONMONOTONE);
    let mut initial = native_plan(1);
    consult(&mut drafter, &mut initial, 7);
    let prior = state(&drafter)
        .request
        .probability(&state(&drafter).depths[1]);
    let mut learned = false;
    for _ in 0..16 {
        let mut window = native_plan(1);
        consult(&mut drafter, &mut window, 7);
        if pending(&drafter).chosen == Some(1) {
            learned = true;
            break;
        }
        assert_eq!(pending(&drafter).chosen, Some(3));
        // A real shaped outcome: first draft accepted, second rejected,
        // third censored. The first two tokens include the verifier bonus.
        drafter.derived_observe(
            &window,
            99,
            &Ok(MtpWindow {
                committed: vec![2, 8],
                accepted: 1,
                drafts_generated: 3,
            }),
        );
    }
    assert!(
        learned,
        "repeated observed second-row failures must change the optimum"
    );
    let current = state(&drafter);
    assert!(current.observes > 0);
    assert_eq!(current.request.last_first(DerivedSource::Mtp), Some(true));
    assert!(current.request.probability(&current.depths[1]) < prior);
    assert_eq!(current.request.probability(&current.depths[2]), prior);
    assert!(!drafter.derived_resolve(1));
    assert_eq!(pending(&drafter).chosen, Some(1));
    assert_eq!(pending(&drafter).chosen_cost_us, 30_000.0);
    // If two rows are already sunk, the better width-1 action is illegal.
    // Width 3 still beats width 2 at their original TOTAL window prices.
    assert!(drafter.derived_resolve(2));
    assert_eq!(pending(&drafter).chosen, Some(3));
    assert_eq!(pending(&drafter).chosen_cost_us, 40_000.0);
    assert_eq!(
        state(&drafter)
            .resolve
            .iter()
            .map(|p| p.k)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert!(!drafter.derived_resolve(3));
    assert_eq!(pending(&drafter).chosen, Some(3));
}

fn sparse_price(
    _: DerivedCard,
    _: DerivedCostMode,
    _: DerivedSource,
    _: ContextBucket,
    k: u8,
    _: bool,
) -> Option<(f64, bool)> {
    matches!(k, 1 | 7 | 31 | 63).then(|| (100.0 + f64::from(k), k == 63))
}

fn capacities(drafter: &Qwen4MtpDrafter) -> [usize; 6] {
    let s = state(drafter);
    [
        s.depths.capacity(),
        s.prices.capacity(),
        s.resolve.capacity(),
        s.keys.capacity(),
        s.margined.capacity(),
        s.margin_bins.capacity(),
    ]
}

#[test]
fn sparse_measured_widths_reuse_preallocated_policy_vectors_across_windows() {
    let tables = DerivedTables {
        window_price: sparse_price,
        ..WIDE
    };
    let mut drafter = bound(1, 8, tables);
    drafter.spec_rows_cap = 64;
    let reserved = capacities(&drafter);
    assert!(
        reserved[0] >= 63
            && reserved[1] >= 64
            && reserved[2] >= 64
            && reserved[3] >= 63
            && reserved[4] >= 63
            && reserved[5] >= 63
    );
    let storage = {
        let s = state(&drafter);
        (
            s.depths.as_ptr(),
            s.prices.as_ptr(),
            s.resolve.as_ptr(),
            s.keys.as_ptr(),
            s.margined.as_ptr(),
            s.margin_bins.as_ptr(),
        )
    };
    for rows in [8, 64, 16, 32, 64, 8] {
        drafter.derived_refresh_rows(rows);
        let cap = rows - 1;
        let result = plan(&mut drafter, cap);
        let chosen = [1u8, 7, 31, 63]
            .into_iter()
            .filter(|k| usize::from(*k) <= cap)
            .max()
            .unwrap();
        assert_eq!(pending(&drafter).chosen, Some(chosen));
        assert_eq!(pending(&drafter).hi, chosen);
        assert_eq!(pending(&drafter).pooled, chosen == 63);
        let expected: Vec<u8> = [0u8, 1, 7, 31, 63]
            .into_iter()
            .filter(|k| usize::from(*k) <= cap)
            .collect();
        assert_eq!(
            state(&drafter)
                .prices
                .iter()
                .map(|p| p.k)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(state(&drafter).depths.len(), usize::from(chosen));
        for drafted in 1..=usize::from(chosen) {
            assert_eq!(
                drafter.derived_resolve(drafted),
                drafted < usize::from(chosen)
            );
        }
        drafter.derived_observe(
            &result,
            99,
            &Ok(MtpWindow {
                committed: vec![2; usize::from(chosen) + 1],
                accepted: usize::from(chosen),
                drafts_generated: usize::from(chosen),
            }),
        );
        assert_eq!(capacities(&drafter), reserved);
        let s = state(&drafter);
        assert_eq!(
            (
                s.depths.as_ptr(),
                s.prices.as_ptr(),
                s.resolve.as_ptr(),
                s.keys.as_ptr(),
                s.margined.as_ptr(),
                s.margin_bins.as_ptr()
            ),
            storage
        );
    }
}

fn forbidden_counts(
    _: DerivedCard,
    _: DerivedCostMode,
    _: DerivedSource,
    _: u8,
    _: DerivedFeature,
) -> Option<CalibrationCounts> {
    panic!("disabled or ineligible adapter queried calibration")
}

fn forbidden_price(
    _: DerivedCard,
    _: DerivedCostMode,
    _: DerivedSource,
    _: ContextBucket,
    _: u8,
    _: bool,
) -> Option<(f64, bool)> {
    panic!("disabled or ineligible adapter queried a price")
}

fn forbidden_ar(_: DerivedCard, _: DerivedCostMode, _: ContextBucket) -> Option<(f64, bool)> {
    panic!("disabled or ineligible adapter queried AR")
}

#[test]
fn default_off_and_ineligible_keep_incumbent_plans_without_querying_tables() {
    let forbidden = DerivedTables {
        counts: forbidden_counts,
        window_price: forbidden_price,
        ar_price: forbidden_ar,
    };
    for (default_k, pool) in [1, 3, 10].into_iter().flat_map(|k| {
        [None, Some(false), Some(true)]
            .into_iter()
            .map(move |pool| (k, pool))
    }) {
        // The incumbent's tables follow the executor's admitted rows (the
        // same refresh every drafter receives before planning); the baseline
        // sees the same rows so only the adapter's phase differs.
        let mut baseline = calibrated(default_k);
        baseline.derived_refresh_rows(64);
        let mut off = calibrated(default_k);
        assert!(off.derived.is_none());
        off.derived_configure(&greedy(), false);
        off.derived_bind(Ok(DerivedCard::R9700), forbidden);
        off.derived_refresh_rows(64);
        assert!(off.derived.is_none());
        let mut ineligible = calibrated(default_k);
        ineligible.derived_configure(&greedy(), true);
        ineligible.derived_bind(Err("unsupported_arch"), forbidden);
        ineligible.derived_position = Some(1500);
        ineligible.derived_refresh_rows(64);
        assert!(matches!(ineligible.derived, Some(DerivedPhase::Ineligible)));
        if let Some(copy) = pool {
            arm_pool(&mut baseline, copy);
            arm_pool(&mut off, copy);
            arm_pool(&mut ineligible, copy);
        }
        assert_eq!(off.k(), baseline.k());
        assert_eq!(ineligible.proposal_capacity(), baseline.proposal_capacity());
        for k in [default_k.min(MAX_DEPTH), 1, 0] {
            let expected = plan(&mut baseline, k);
            assert_eq!(plan(&mut off, k), expected);
            assert_eq!(plan(&mut ineligible, k), expected);
            assert_eq!(off.ngram_candidates, baseline.ngram_candidates);
            assert_eq!(ineligible.ngram_candidates, baseline.ngram_candidates);
            assert_eq!(off.floor.retired(), baseline.floor.retired());
            assert_eq!(ineligible.floor.retired(), baseline.floor.retired());
        }
    }
}

#[cfg(feature = "reference-parity")]
#[test]
fn lab_width_controls_accept_zero_and_63_reject_64_and_clear_to_incumbent() {
    let mut drafter = calibrated(3);
    assert_eq!(drafter.lab_force_drafts, None);
    drafter
        .lab_force_window_drafts(Some(0))
        .expect("width zero");
    assert_eq!((drafter.k(), drafter.proposal_capacity()), (0, 0));
    let zero = plan(&mut drafter, 0);
    assert_eq!(zero.route, WindowRoute::Native { depth: 0 });
    assert_eq!(zero.k, 0);
    assert!(!zero.ngram_hit);
    drafter
        .lab_force_window_drafts(Some(63))
        .expect("physical maximum");
    assert_eq!((drafter.k(), drafter.proposal_capacity()), (63, 63));
    let wide = plan(&mut drafter, 63);
    assert_eq!(wide.route, WindowRoute::Native { depth: 63 });
    assert_eq!(wide.k, 63);
    assert!(drafter.lab_force_window_drafts(Some(64)).is_err());
    assert_eq!(drafter.lab_force_drafts, Some(63));
    let budgeted = plan(&mut drafter, 2);
    assert_eq!(budgeted.route, WindowRoute::Native { depth: 2 });
    drafter
        .lab_force_window_drafts(None)
        .expect("clear override");
    assert_eq!(drafter.lab_force_drafts, None);
    let mut baseline = calibrated(3);
    assert_eq!(drafter.k(), baseline.k());
    assert_eq!(plan(&mut drafter, 3), plan(&mut baseline, 3));
}

#[cfg(feature = "reference-parity")]
#[test]
fn lab_copy_toggle_and_retirement_preserve_the_real_pool_without_native_resurrection() {
    let mut drafter = calibrated(3);
    arm_pool(&mut drafter, true);
    let pool = drafter.ngram_copy.as_ref().unwrap() as *const MtpCopyContext;
    let agreement = drafter.ngram_agreement.clone();
    let yield_before = drafter.ngram_yield;
    drafter
        .lab_force_window_drafts(Some(7))
        .expect("force real copy width");
    drafter.lab_force_copy_enabled(Some(false));
    assert_eq!(drafter.ngram_agreement, agreement);
    assert_eq!(drafter.ngram_yield, yield_before);
    assert!(drafter.ngram_last_first_accepted);
    assert!(!plan(&mut drafter, 7).ngram_hit);
    drafter.lab_force_copy_enabled(Some(true));
    let offered = plan(&mut drafter, 7);
    assert!(offered.ngram_hit);
    assert_eq!(drafter.ngram_candidates, (2..9).collect::<Vec<_>>());
    assert!(
        !offered.ngram_confirm,
        "established source keeps its confirmation status"
    );
    assert_eq!(
        drafter.ngram_copy.as_ref().unwrap() as *const MtpCopyContext,
        pool
    );

    drafter.lab_retire_native().expect("singleton retirement");
    assert!(drafter.floor.retired() && drafter.stats.mtp_retired);
    assert_eq!(
        drafter.ngram_copy.as_ref().unwrap() as *const MtpCopyContext,
        pool
    );
    let retired_copy = plan(&mut drafter, 7);
    assert_eq!(retired_copy.route, WindowRoute::ArRetired);
    assert!(retired_copy.ngram_hit);
    assert_eq!(drafter.ngram_candidates, (2..9).collect::<Vec<_>>());
    drafter.lab_force_copy_enabled(Some(false));
    let retired_native = plan(&mut drafter, 7);
    assert_eq!(retired_native.route, WindowRoute::ArRetired);
    assert!(!retired_native.ngram_hit);
    drafter
        .lab_force_window_drafts(None)
        .expect("clear forced width");
    drafter.lab_force_copy_enabled(None);
    assert_eq!(drafter.lab_copy_enabled, None);
    assert!(drafter.floor.retired());
    let restored = plan(&mut drafter, 3);
    assert_eq!(restored.route, WindowRoute::ArRetired);
    assert!(
        restored.ngram_hit,
        "clearing controls restores the configured live copy source"
    );
    assert_eq!(
        drafter.ngram_copy.as_ref().unwrap() as *const MtpCopyContext,
        pool
    );
    assert_eq!(drafter.ngram_agreement, agreement);
    assert_eq!(drafter.ngram_yield, yield_before);
    assert!(drafter.floor.retired() && drafter.stats.mtp_retired);
}

#[cfg(feature = "reference-parity")]
#[test]
fn lab_native_retirement_rejects_a_follower_without_changing_its_floor() {
    let mut drafter = calibrated(3);
    drafter.follower = true;
    assert!(drafter.lab_retire_native().is_err());
    assert!(!drafter.floor.retired());
    assert!(!drafter.stats.mtp_retired);
}

fn off_pool_counts_only(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    match mode {
        DerivedCostMode::OffPool => certain_counts(card, mode, source, depth, feature),
        DerivedCostMode::OnCopy => None,
    }
}

fn counts_with_copy_depth_hole(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    if mode == DerivedCostMode::OnCopy {
        assert!(
            depth <= 2,
            "calibration must stop at the first unsupported depth"
        );
        if depth == 2 {
            return None;
        }
    }
    certain_counts(card, mode, source, depth, feature)
}

#[test]
fn measured_prices_without_mode_local_counts_do_not_invent_an_acceptance_prior() {
    let tables = DerivedTables {
        counts: off_pool_counts_only,
        ..WIDE
    };
    for source in [DerivedSource::Mtp, DerivedSource::Ngram] {
        let mut drafter = bound(3, 64, tables).with_copy_lane(true);
        let mut window = native_plan(3);
        if source == DerivedSource::Ngram {
            drafter.ngram_proposal = (2..65).collect();
            drafter.ngram_candidates = vec![2, 3];
            window.ngram_hit = true;
        }
        let incumbent = window.clone();
        let candidates = drafter.ngram_candidates.clone();
        consult(&mut drafter, &mut window, 63);
        assert_eq!(window, incumbent);
        assert_eq!(drafter.ngram_candidates, candidates);
        assert_eq!(state(&drafter).consults, 0);
        assert!(state(&drafter).pending.is_none());
        assert!(state(&drafter).depths.is_empty());
        assert!(state(&drafter).prices.iter().all(|price| price.k == 0));

        // The same synthetic calibration does support OffPool; it must not
        // accidentally become unavailable along with the missing OnCopy arm.
        drafter.copy_lane = false;
        consult(&mut drafter, &mut window, 7);
        assert_eq!(pending(&drafter).mode, DerivedCostMode::OffPool);
        assert_eq!(pending(&drafter).chosen, Some(7));
    }
}

#[test]
fn a_missing_intermediate_depth_excludes_every_deeper_priced_prefix() {
    let tables = DerivedTables {
        counts: counts_with_copy_depth_hole,
        ..WIDE
    };
    for source in [DerivedSource::Mtp, DerivedSource::Ngram] {
        let mut drafter = bound(3, 64, tables).with_copy_lane(true);
        let mut window = native_plan(3);
        if source == DerivedSource::Ngram {
            drafter.ngram_proposal = (2..65).collect();
            drafter.ngram_candidates = vec![2, 3, 4];
            window.ngram_hit = true;
        }
        consult(&mut drafter, &mut window, 63);
        assert_eq!(pending(&drafter).chosen, Some(1));
        assert_eq!(pending(&drafter).hi, 1);
        assert_eq!(state(&drafter).depths.len(), 1);
        assert_eq!(state(&drafter).keys.len(), 1);
        assert_eq!(
            state(&drafter)
                .prices
                .iter()
                .map(|p| p.k)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        if source == DerivedSource::Mtp {
            assert_eq!(window.route, WindowRoute::Native { depth: 1 });
            assert!(!drafter.derived_resolve(1));
        } else {
            assert_eq!(drafter.ngram_candidates, vec![2]);
        }
    }
}

// ---- Native causal-margin evidence (frozen contract) ----
//
// Every test below is CPU-only with synthetic tables; nothing here qualifies
// a GPU route or constrains future measured tables.

use hipfire_runtime::spec_economics::{native_margin_bin, NativeMarginBin};

/// Price every width identically so the global optimum is chosen by evidence
/// (margins), never by the cost shape. AR is unpriced so it never wins.
fn flat_price(
    _: DerivedCard,
    _: DerivedCostMode,
    _: DerivedSource,
    _: ContextBucket,
    k: u8,
    _: bool,
) -> Option<(f64, bool)> {
    Some((10_000.0 + f64::from(k), false))
}

/// Synthetic counts: the unconditional cell keeps `certain_counts` support
/// (the pre-window consult prices the real window), and only the armed
/// margin cells carry support. `supported` is `(observed_depth, bin,
/// target_depth)` and `cell` is the evidence each supported cell carries;
/// every other margin cell is unsupported and falls back to the
/// unconditional depth.
fn margin_lookup(
    supported: &[(u8, NativeMarginBin, u8)],
    cell: CalibrationCounts,
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    match feature {
        DerivedFeature::NativeMargin {
            observed_depth,
            margin,
        } => (source == DerivedSource::Mtp && supported.contains(&(observed_depth, margin, depth)))
            .then_some(cell),
        DerivedFeature::Native => certain_counts(card, mode, source, depth, feature),
        _ => None,
    }
}

/// A supported margin cell as certain as the unconditional evidence.
const MARGIN_CERTAIN: CalibrationCounts = CalibrationCounts {
    depth_s: 10_000,
    depth_n: 10_000,
    cell_s: 10_000,
    cell_n: 10_000,
};
/// A supported margin cell observing that the row is (almost) never accepted:
/// its survival probability is ~3e-8, far below the 1e-4 relative price step
/// of one more row.
const MARGIN_NEVER: CalibrationCounts = CalibrationCounts {
    depth_s: 0,
    depth_n: 10_000,
    cell_s: 0,
    cell_n: 10_000,
};

fn margin_weak(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    margin_lookup(
        &[(1u8, NativeMarginBin::LtQuarter, 1u8)],
        MARGIN_NEVER,
        card,
        mode,
        source,
        depth,
        feature,
    )
}
fn margin_none(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    margin_lookup(&[], MARGIN_CERTAIN, card, mode, source, depth, feature)
}
fn margin_strong(
    card: DerivedCard,
    mode: DerivedCostMode,
    source: DerivedSource,
    depth: u8,
    feature: DerivedFeature,
) -> Option<CalibrationCounts> {
    margin_lookup(
        &[
            (1u8, NativeMarginBin::GeSix, 1u8),
            (1u8, NativeMarginBin::GeSix, 2u8),
        ],
        MARGIN_CERTAIN,
        card,
        mode,
        source,
        depth,
        feature,
    )
}

/// Arm a real consulted native window (`certain` unconditional evidence,
/// flat total prices), inject one actual head-step margin, and re-solve.
/// Returns the post-row chosen width.
fn consult_then_resolve(
    counts: fn(
        DerivedCard,
        DerivedCostMode,
        DerivedSource,
        u8,
        DerivedFeature,
    ) -> Option<CalibrationCounts>,
    margin: f32,
) -> (Option<u8>, Qwen4MtpDrafter) {
    let tables = DerivedTables {
        counts,
        window_price: flat_price,
        ar_price: no_ar,
    };
    let mut drafter = bound(1, 8, tables);
    let mut window = native_plan(1);
    consult(&mut drafter, &mut window, 7);
    assert_eq!(pending(&drafter).chosen, Some(7));
    assert_eq!(state(&drafter).depths.len(), 7);
    drafter.derived_margins_reset();
    drafter.derived_margins_push(margin);
    if !margin.is_finite() || margin < 0.0 {
        assert_eq!(drafter.derived_margin_bin(1), None);
    } else {
        assert_eq!(drafter.derived_margin_bin(1), native_margin_bin(margin));
    }
    drafter.derived_resolve(1);
    let chosen = pending(&drafter).chosen;
    (chosen, drafter)
}

#[test]
fn opposite_margins_choose_different_global_optima() {
    // Every width is priced flat (a one-in-ten-thousand step per row) and the
    // unconditional evidence is certain, so the no-margin optimum is the
    // widest legal prefix (7); only the margin cells differ. A row-1
    // LtQuarter cell that observes the draft is (almost) never accepted
    // makes every deeper row worth ~nothing for its price: the optimum drops
    // to width 1 (depths 2..7 fall back to their unconditional cell, but the
    // survival probability already carries the row-1 failure). A row-1 GeSix
    // cell as certain as the unconditional evidence keeps the optimum wide.
    // Same unconditional evidence, same prices: only the margin differs.
    let (weak, _) = consult_then_resolve(margin_weak, 0.1);
    let (strong, _) = consult_then_resolve(margin_strong, 10.0);
    assert_eq!((weak, strong), (Some(1), Some(7)));
}

#[test]
fn unsupported_margin_cells_never_fabricate_support() {
    // No margin cell supported anywhere: the re-solve falls back to the
    // unconditional depths and must keep the no-margin optimum exactly.
    let (chosen, mut drafter) = consult_then_resolve(margin_none, 10.0);
    assert_eq!(chosen, Some(7));
    // The scripted re-solve still drafts rows past the first (the no-margin
    // optimum 7 exceeds the one row drafted): the margin only keys evidence.
    assert!(drafter.derived_resolve(1));
    assert_eq!(pending(&drafter).chosen, chosen);
}

#[test]
fn nonfinite_or_negative_margins_fall_back_to_unconditional_evidence() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5, -1.0] {
        let tables = DerivedTables {
            counts: margin_strong,
            window_price: flat_price,
            ar_price: no_ar,
        };
        let mut drafter = bound(1, 8, tables);
        let mut window = native_plan(1);
        consult(&mut drafter, &mut window, 7);
        drafter.derived_margins_reset();
        drafter.derived_margins_push(bad);
        // No bin, no diagnostic panic: falls back to the unconditional
        // re-solve, whose optimum (7) is unchanged.
        assert_eq!(drafter.derived_margin_bin(1), None);
        assert!(drafter.derived_resolve(1));
        assert_eq!(pending(&drafter).chosen, Some(7));
    }
}

#[test]
fn margin_evidence_informs_every_depth_not_only_remaining_prefixes() {
    // Cells exist for observed row 1 for targets 1 and 2 only; target 7 has
    // no margin cell anywhere and falls back to its unconditional depth.
    // The re-solve domain is every legal depth (acceptance still unknown),
    // the remaining-prefix filter `k >= drafted` applies only to the action
    // set. Row 1's margin re-keys depths 1..=7: measured where supported.
    let tables = DerivedTables {
        counts: margin_strong,
        window_price: flat_price,
        ar_price: no_ar,
    };
    let mut drafter = bound(1, 8, tables);
    let mut window = native_plan(1);
    consult(&mut drafter, &mut window, 7);
    drafter.derived_margins_reset();
    drafter.derived_margins_push(10.0);
    assert_eq!(drafter.derived_margin_bin(1), Some(NativeMarginBin::GeSix));
    assert!(drafter.derived_resolve(1));
    assert_eq!(pending(&drafter).chosen, Some(7));
    assert_eq!(state(&drafter).last_measured, 2);
    assert_eq!(
        state(&drafter).last_margin,
        Some((1, NativeMarginBin::GeSix)),
    );
    // Re-solving at row 2 with only a row-2 margin looks up row-2 cells
    // (unsupported) and falls back everywhere: `last_measured` drops to 0.
    drafter.derived_margins_push(0.05);
    assert_eq!(
        drafter.derived_margin_bin(2),
        Some(NativeMarginBin::LtQuarter)
    );
    assert!(drafter.derived_resolve(2));
    assert_eq!(pending(&drafter).chosen, Some(7));
    assert_eq!(state(&drafter).last_margin, None);
    assert_eq!(state(&drafter).last_measured, 0);
}

#[test]
fn native_margin_commit_observation_ages_once_and_censors_terminal() {
    // Two executed rows, accepted one, terminal (committed contains the eos
    // token): base + row-1 GeSix cells for t=1 (success); t=2 censored
    // (terminal rejection unobserved). Single aging pass, `last_first`
    // written once by the shared runtime path.
    let tables = DerivedTables {
        counts: certain_counts,
        window_price: cheap_price,
        ar_price: no_ar,
    };
    let mut drafter = bound(1, 8, tables);
    let mut window = native_plan(1);
    consult(&mut drafter, &mut window, 7);
    assert_eq!(pending(&drafter).chosen, Some(7));
    assert_eq!(state(&drafter).observes, 0);
    drafter.derived_margins_reset();
    drafter.derived_margins_push(10.0);
    drafter.derived_margins_push(10.0);
    drafter.derived_observe(
        &window,
        2,
        &Ok(MtpWindow {
            committed: vec![5, 2],
            accepted: 1,
            drafts_generated: 2,
        }),
    );
    assert_eq!(state(&drafter).observes, 1);
    assert!(state(&drafter).pending.is_none());
    let row1_key = ResidualKey {
        source: DerivedSource::Mtp,
        depth: 1,
        feature: DerivedFeature::NativeMargin {
            observed_depth: 1,
            margin: NativeMarginBin::GeSix,
        },
    };
    assert!(
        state(&drafter).request.probability(&DerivedDepth {
            key: row1_key,
            counts: CalibrationCounts {
                depth_s: 8_000,
                depth_n: 10_000,
                cell_s: 0,
                cell_n: 0,
            },
        }) > 0.0,
        "row-1 margin cell must be observed at commit"
    );
    let row2_key = ResidualKey {
        source: DerivedSource::Mtp,
        depth: 2,
        feature: DerivedFeature::NativeMargin {
            observed_depth: 2,
            margin: NativeMarginBin::GeSix,
        },
    };
    assert!(
        state(&drafter).request.probability(&DerivedDepth {
            key: row2_key,
            counts: CalibrationCounts {
                depth_s: 8_000,
                depth_n: 10_000,
                cell_s: 0,
                cell_n: 0,
            },
        }) > 0.0,
        "row-2 margin cell must be observed at commit"
    );
}

#[test]
fn native_margin_bin_rejects_negative_and_nonfinite() {
    assert_eq!(native_margin_bin(f32::NAN), None);
    assert_eq!(native_margin_bin(f32::INFINITY), None);
    assert_eq!(native_margin_bin(f32::NEG_INFINITY), None);
    assert_eq!(native_margin_bin(-0.1), None);
    assert_eq!(native_margin_bin(0.0), Some(NativeMarginBin::LtQuarter));
    assert_eq!(native_margin_bin(0.249), Some(NativeMarginBin::LtQuarter));
    assert_eq!(
        native_margin_bin(0.25),
        Some(NativeMarginBin::LtOnePointFive)
    );
    assert_eq!(
        native_margin_bin(1.499),
        Some(NativeMarginBin::LtOnePointFive)
    );
    assert_eq!(native_margin_bin(1.5), Some(NativeMarginBin::LtThree));
    assert_eq!(native_margin_bin(2.999), Some(NativeMarginBin::LtThree));
    assert_eq!(native_margin_bin(3.0), Some(NativeMarginBin::LtSix));
    assert_eq!(native_margin_bin(5.999), Some(NativeMarginBin::LtSix));
    assert_eq!(native_margin_bin(6.0), Some(NativeMarginBin::GeSix));
    assert_eq!(native_margin_bin(1e30), Some(NativeMarginBin::GeSix));
}

#[test]
fn derived_margins_scratch_is_off_path_free_and_resets_per_window() {
    // Off path: no derived state -> no allocation, no storage.
    let mut off = Qwen4MtpDrafter::new(10, 4096, None);
    off.derived_margins_reset();
    assert!(off.derived_margins.is_none());
    off.derived_margins_push(10.0);
    assert!(off.derived_margins.is_none());
    // Bound: one reservation, reused across windows, cleared per window.
    let mut drafter = bound(1, 8, WIDE);
    drafter.derived_margins_reset();
    assert_eq!(
        drafter
            .derived_margins
            .as_ref()
            .map(|scratch| scratch.capacity()),
        Some(DERIVED_MAX_DRAFTS)
    );
    drafter.derived_margins_push(1.0);
    drafter.derived_margins_push(f32::NAN);
    assert_eq!(
        drafter
            .derived_margins
            .as_ref()
            .map(|scratch| scratch.len()),
        Some(2)
    );
    let ptr = drafter
        .derived_margins
        .as_ref()
        .map(|scratch| scratch.as_ptr());
    drafter.derived_margins_reset();
    assert_eq!(
        drafter
            .derived_margins
            .as_ref()
            .map(|scratch| scratch.len()),
        Some(0)
    );
    assert_eq!(
        drafter
            .derived_margins
            .as_ref()
            .map(|scratch| scratch.as_ptr()),
        ptr,
        "scratch storage must be reused, never reallocated per window"
    );
}

#[test]
fn strict_requested_max_63_sizes_64_rows_derived_off() {
    // Sizing only: requested max 63 with derived off and the copy lane off
    // (hence no take cap) must reserve 64 rows at admission, even though the
    // incumbent execution tables stay capped at `MAX_DEPTH`.
    assert_eq!(
        crate::mtp_spec::native_mtp_proposal_drafts(63, "gfx1201"),
        63
    );
    assert_eq!(
        crate::mtp_spec::native_mtp_proposal_drafts(63, "gfx1151"),
        63
    );
    assert_eq!(
        crate::mtp_spec::native_mtp_proposal_drafts(10, "gfx1201"),
        10
    );
}

#[test]
fn measured_load_identity_builder_defaults_to_none() {
    // A directly constructed drafter carries no descriptor: synthetic tests
    // exercise the inner binder without claiming production eligibility.
    // The typed `Qwen4MeasuredLoadIdentity` module lands with LoadIdentityMuse;
    // until then only the `None` default is constructible here.
    let drafter = Qwen4MtpDrafter::new(10, 4096, None);
    assert!(drafter.measured_load_identity.is_none());
}

#[test]
fn bind_without_descriptor_stays_ineligible_incumbent() {
    // Inner binder without the production descriptor: bind path keeps the
    // real incumbent (`Ineligible`), never invented prices. Synthetic tables
    // exercise the same outcome without a fake load claim.
    let mut drafter = bound(1, 8, WIDE);
    match &drafter.derived {
        Some(DerivedPhase::Bound(_)) => {}
        other => panic!(
            "expected an eligible bound adapter, got bound={}",
            other.is_some()
        ),
    }
    drafter.derived = Some(DerivedPhase::Unbound);
    drafter.derived_bind(Err("measured_load_missing"), WIDE);
    assert!(matches!(drafter.derived, Some(DerivedPhase::Ineligible)));
}

#[test]
fn mode_flip_after_bind_invalidates_the_window() {
    // The bind-time mode hint is stamped OffPool; flipping to the copy lane
    // (OnCopy) before a plan invalidates to `Ineligible` instead of pricing
    // the window under the wrong profile. The live AR-reference check
    // additionally gates on `fn_measured_reference_supported`; with the
    // bootstrap-empty measured table that check is `false`, so this test
    // only exercises the mode-flip half.
    let mut drafter = bound(1, 8, WIDE).with_copy_lane(false);
    assert_eq!(state(&drafter).mode_hint, Some(DerivedCostMode::OffPool));
    drafter.copy_lane = true;
    let probe_plan = native_plan(1);
    assert!(drafter.derived_route_changed(&probe_plan));
    let mut drafter = bound(1, 8, WIDE);
    drafter.derived = Some(DerivedPhase::Bound(Box::new({
        let mut fresh = DerivedState::new(DerivedCard::R9700, WIDE, state(&drafter).capacity);
        fresh.mode_hint = Some(DerivedCostMode::OffPool);
        fresh
    })));
    drafter.copy_lane = true;
    drafter.derived_position = Some(1500);
    let mut plan = native_plan(1);
    plan.route = WindowRoute::Native { depth: 3 };
    drafter.derived_plan(&mut plan, 7);
    assert!(matches!(drafter.derived, Some(DerivedPhase::Ineligible)));
}
