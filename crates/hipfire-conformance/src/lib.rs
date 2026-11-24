//! hipfire conformance runner, CPU-only first half (V2-PLAN 6.4, slices B0/B1a).
//!
//! * [`digest`]: per-component digests of defined live bytes.
//! * [`receipt`]: executed-recipe receipts from the module-load audit.
//! * [`verdict`]: class table, row interface, verdicts.
//! * [`selftest`]: a host fake state with planted faults.

pub mod digest;
pub mod receipt;
pub mod verdict;

pub mod selftest {
    //! Host fake: `LANES` lanes, each with an append-only KV row stream owned by
    //! the lane, plus a fixed `recurrent` part and a fixed `sampler` part
    //! derived from the seed. Faults are planted into the candidate arm.

    use crate::digest::{h64, HostPart, HostRows, HostState};
    use crate::verdict::*;

    pub const LANES: usize = 2;
    const ROW: usize = 8;
    pub const GOOD_KERNELS: [&str; 2] = ["gemv_q8_wmma", "attn_flash"];
    pub const FALLBACK_KERNELS: [&str; 2] = ["gemv_q8_scalar", "attn_flash"];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fault {
        None,
        /// After step 2, lane 0's and lane 1's KV row 2 trade places.
        SwappedKvRow,
        /// A restore at step 3 leaves lane 1's row 1 stale (poisoned).
        StaleRestore,
        /// The candidate samples with a different seed.
        WrongSeed,
    }

    pub struct FakeArm {
        pub fault: Fault,
        /// Kernels the fake reports resolving.
        pub kernels: Vec<&'static str>,
    }

    impl FakeArm {
        pub fn good() -> Self {
            FakeArm { fault: Fault::None, kernels: GOOD_KERNELS.to_vec() }
        }
        pub fn faulty(fault: Fault) -> Self {
            FakeArm { fault, kernels: GOOD_KERNELS.to_vec() }
        }
        pub fn fallback() -> Self {
            FakeArm { fault: Fault::None, kernels: FALLBACK_KERNELS.to_vec() }
        }
    }

    fn row_bytes(seed: u64, lane: usize, step: usize) -> [u8; ROW] {
        let mut x = seed ^ ((lane as u64 + 1) << 32) ^ (step as u64 + 1).wrapping_mul(0x9e37_79b9);
        let mut out = [0u8; ROW];
        for b in &mut out {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 24) as u8;
        }
        out
    }

    impl Arm for FakeArm {
        fn run(&mut self, _recipe: &Recipe, s: &InputSchedule) -> Result<ArmRun, String> {
            let seed = if self.fault == Fault::WrongSeed { s.seed ^ 0xdead } else { s.seed };
            let mut kv: Vec<Vec<u8>> = vec![Vec::new(); LANES];
            let mut recurrent = [0u8; 8];
            let mut steps = Vec::new();
            for step in 0..s.steps {
                for (lane, rows) in kv.iter_mut().enumerate() {
                    rows.extend_from_slice(&row_bytes(seed, lane, step));
                }
                let kvh = kv.iter().fold(0u64, |h, k| h.rotate_left(3) ^ h64(k));
                recurrent = (u64::from_le_bytes(recurrent) ^ kvh).wrapping_mul(31).to_le_bytes();
                if self.fault == Fault::SwappedKvRow && step >= 2 {
                    let (a, b) = kv.split_at_mut(1);
                    let (r0, r1) = (&mut a[0], &mut b[0]);
                    for i in 0..ROW {
                        std::mem::swap(&mut r0[2 * ROW + i], &mut r1[2 * ROW + i]);
                    }
                    // swapped once, at step 2 only
                    if step > 2 {
                        for i in 0..ROW {
                            std::mem::swap(&mut r0[2 * ROW + i], &mut r1[2 * ROW + i]);
                        }
                    }
                }
                if self.fault == Fault::StaleRestore && step >= 3 {
                    kv[1][ROW..2 * ROW].fill(0xA5);
                }
                steps.push(HostState {
                    meta: (step as u64 + 1).to_le_bytes().to_vec(),
                    fixed: vec![
                        HostPart { name: "sampler".into(), bytes: seed.to_le_bytes().to_vec() },
                        HostPart { name: "recurrent".into(), bytes: recurrent.to_vec() },
                    ],
                    rows: kv
                        .iter()
                        .enumerate()
                        .map(|(l, k)| HostRows {
                            name: format!("kv.lane{l}"),
                            owner: format!("lane{l}"),
                            row_bytes: ROW,
                            bytes: k.clone(),
                        })
                        .collect(),
                });
            }
            let audit_log = self
                .kernels
                .iter()
                .map(|k| format!("HIPFIRE_MODULE_LOAD_AUDIT\tget_function\t0x1\t{k}\n"))
                .fold(
                    "HIPFIRE_MODULE_LOAD_AUDIT\tload_data\t0x1\t64\t00000000000000ff\n".to_string(),
                    |a, l| a + &l,
                );
            Ok(ArmRun { audit_log, steps })
        }
    }

    pub fn recipe_pair() -> RecipePair {
        let r = |n: &str| Recipe {
            name: n.into(),
            required_kernels: GOOD_KERNELS.iter().map(|s| s.to_string()).collect(),
        };
        RecipePair { reference: r("fake-ref"), candidate: r("fake-cand") }
    }

    pub fn row(invariant: &str, class: ComparisonClass) -> RowSpec {
        RowSpec {
            fixture: "host-fake".into(),
            invariant: invariant.into(),
            recipe_pair: recipe_pair(),
            input_schedule: InputSchedule { seed: 0x5eed, steps: 5 },
            comparison_class: class,
            bounded_difference: None,
            budgets: Budgets::new(),
        }
    }

    /// One self-test case and whether it behaved as required.
    pub struct Case {
        pub name: &'static str,
        pub ok: bool,
        pub detail: String,
    }

    fn run(table: &ClassTable, row: &RowSpec, cand: FakeArm, refr: FakeArm) -> Result<RowReport, Refusal> {
        let (mut a, mut b) = (refr, cand);
        run_row(table, row, &mut a, &mut b)
    }

    pub fn run_all() -> Vec<Case> {
        let table = ClassTable::embedded();
        let mut cases = Vec::new();
        let mut add = |name, ok: bool, detail: String| cases.push(Case { name, ok, detail });
        let base = row("I2", ComparisonClass::Exact);

        let r = run(&table, &base, FakeArm::good(), FakeArm::good()).unwrap();
        add("unmodified run is pass", r.verdict == Verdict::Pass, r.verdict.name().into());

        for (name, fault, comp, rowi, owner) in [
            ("kv row swapped between lanes", Fault::SwappedKvRow, "kv.lane0", Some(2), "lane0"),
            ("stale row after restore", Fault::StaleRestore, "kv.lane1", Some(1), "lane1"),
            ("wrong seed", Fault::WrongSeed, "sampler", None, "sampler"),
        ] {
            let r = run(&table, &base, FakeArm::faulty(fault), FakeArm::good()).unwrap();
            let (ok, detail) = match &r.verdict {
                Verdict::Fail { first_difference: d } => (
                    d.component == comp && d.row == rowi && d.owner == owner,
                    format!("fail at step {} {} row {:?} owner {}", d.step, d.component, d.row, d.owner),
                ),
                v => (false, v.name().to_string()),
            };
            add(name, ok, detail);
        }

        let r = run(&table, &base, FakeArm::fallback(), FakeArm::fallback()).unwrap();
        add(
            "fallback recipe on both arms is not covered",
            matches!(r.verdict, Verdict::NotCovered { .. }),
            r.verdict.name().into(),
        );

        let mut b = row("I5w", ComparisonClass::Bounded);
        b.bounded_difference = ClassTable::embedded().get("I5w").unwrap().bounded_difference.clone();
        let r = run(&table, &b, FakeArm::good(), FakeArm::good()).unwrap();
        add(
            "bounded row without budget is unqualified",
            matches!(r.verdict, Verdict::Unqualified { .. }),
            r.verdict.name().into(),
        );

        let e = run(&table, &row("I99", ComparisonClass::Exact), FakeArm::good(), FakeArm::good());
        add("row with no class is refused", matches!(e, Err(Refusal::NoClass(_))), "refused".into());
        cases
    }
}

#[cfg(test)]
mod tests {
    use super::selftest::*;
    use super::verdict::*;

    #[test]
    fn self_test_cases_all_behave() {
        for c in run_all() {
            assert!(c.ok, "{}: {}", c.name, c.detail);
        }
    }

    #[test]
    fn bounded_on_always_exact_invariant_is_refused() {
        let t = ClassTable::embedded();
        let r = row("I2", ComparisonClass::Bounded);
        let e = run_row(&t, &r, &mut FakeArm::good(), &mut FakeArm::good()).unwrap_err();
        assert!(matches!(e, Refusal::BoundedNotAllowed { .. }));
    }

    #[test]
    fn bounded_with_wrong_difference_name_is_refused() {
        let t = ClassTable::embedded();
        let mut r = row("I5w", ComparisonClass::Bounded);
        r.bounded_difference = Some("something else".into());
        let e = run_row(&t, &r, &mut FakeArm::good(), &mut FakeArm::good()).unwrap_err();
        assert!(matches!(e, Refusal::DifferenceMismatch { .. }));
    }

    #[test]
    fn every_invariant_of_the_plan_is_in_the_table() {
        let t = ClassTable::embedded();
        for id in ["I1", "I2", "I3", "I4k", "I4a", "I5i", "I5w", "I6", "I7", "I8", "I9", "I10"] {
            assert!(t.get(id).is_some(), "{id}");
        }
        for id in ["I2", "I3", "I4k", "I5i", "I6", "I7", "I8"] {
            assert!(t.get(id).unwrap().bounded_difference.is_none(), "{id} must never be bounded");
        }
    }
}
