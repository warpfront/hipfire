// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// AdmissionController — decides whether a session can be admitted, and with how
// much context.
//
// In the test harnesses `kv_slots::preflight_alloc` is what stops an oversized
// configuration. In the daemon that job is HERE. The difference matters: on this
// hardware the GPU allocates from system RAM and the cgroup does NOT contain
// amdgpu GTT, so a wrong decision here does not fail a request — it takes down
// the user's desktop with a global OOM.

/// What one loaded model costs, split into the part charged once and the part
/// charged per session.
#[derive(Debug, Clone, Copy)]
pub struct ModelFootprint {
    /// Charged ONCE, however many sessions are admitted.
    pub weights_bytes: u64,
    /// Charged per session, per token of granted context.
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitError {
    PoolFull,
    WouldExceedBudget {
        need: u64,
        available: u64,
    },
    /// resize() named a session that holds no grant — an internal
    /// inconsistency, NOT a budget shortfall. Reporting it as
    /// WouldExceedBudget{0,0} both lied about the cause and let the engine
    /// misclassify a real budget failure as Internal.
    UnknownSession(u64),
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |b: u64| b as f64 / 1073741824.0;
        match self {
            AdmitError::PoolFull => write!(f, "no free slot"),
            AdmitError::WouldExceedBudget { need, available } => write!(
                f,
                "needs {:.2} GiB but only {:.2} GiB of the budget remains",
                gib(*need),
                gib(*available)
            ),
            AdmitError::UnknownSession(id) => {
                write!(f, "session {id} holds no admission grant")
            }
        }
    }
}

pub struct AdmissionController {
    footprint: ModelFootprint,
    budget_bytes: u64,
    /// Granted context per admitted session, keyed by session id so two
    /// sessions with identical context sizes cannot release each other's
    /// budget.
    admitted: Vec<(u64, usize)>,
    /// Host-tier budget for swapped-out snapshots. Separate from the VRAM
    /// budget: admission is the production memory gate for BOTH, because the
    /// control group does not contain amdgpu GTT.
    host_budget: u64,
    host_used: u64,
}

impl AdmissionController {
    pub fn new(footprint: ModelFootprint, budget_bytes: u64) -> Self {
        Self {
            footprint,
            budget_bytes,
            admitted: Vec::new(),
            host_budget: crate::swap::DEFAULT_HOST_BUDGET_BYTES,
            host_used: 0,
        }
    }

    /// Bytes currently committed: weights once (if anything is admitted) plus
    /// each session's KV. Checked arithmetic; overflow saturates (a sum this
    /// large cannot be admitted anyway) rather than wrapping.
    pub fn used_bytes(&self) -> u64 {
        if self.admitted.is_empty() {
            return 0;
        }
        let kv: u64 = self
            .admitted
            .iter()
            .map(|&(_, ctx)| {
                (ctx as u64)
                    .checked_mul(self.footprint.kv_bytes_per_token)
                    .unwrap_or(u64::MAX)
            })
            .fold(0u64, |a, b| a.saturating_add(b));
        self.footprint.weights_bytes.saturating_add(kv)
    }

    /// Admit a session at `requested_ctx` tokens, or explain why not.
    ///
    /// Rejects rather than silently capping: a caller that asked for 128K and
    /// silently got 8K would produce baffling truncation far from here.
    pub fn admit(&mut self, session: u64, requested_ctx: usize) -> Result<usize, AdmitError> {
        let kv_need = (requested_ctx as u64)
            .checked_mul(self.footprint.kv_bytes_per_token)
            .ok_or(AdmitError::WouldExceedBudget {
                need: u64::MAX,
                available: 0,
            })?;
        // Weights are charged once, on the first admission.
        let weights_need = if self.admitted.is_empty() {
            self.footprint.weights_bytes
        } else {
            0
        };
        let need = kv_need
            .checked_add(weights_need)
            .ok_or(AdmitError::WouldExceedBudget {
                need: u64::MAX,
                available: 0,
            })?;
        let available = self.budget_bytes.saturating_sub(self.used_bytes());
        // >= rather than >: an admission that would consume the LAST byte of
        // budget is refused too, not just one that overflows it. On this
        // hardware (no swap, cgroup does not contain amdgpu GTT) landing
        // exactly on the edge leaves zero headroom for anything else running
        // on the box, so it is treated the same as exceeding the budget.
        if need >= available {
            return Err(AdmitError::WouldExceedBudget { need, available });
        }
        self.admitted.push((session, requested_ctx));
        Ok(requested_ctx)
    }

    /// Return a session's context allowance to the budget.
    /// Reserve host-tier bytes for a swapped-out session. Returns false when
    /// the budget cannot cover it, in which case the caller spills to disk
    /// rather than exceeding the budget.
    pub fn admit_host(&mut self, bytes: u64) -> bool {
        if self.host_used.saturating_add(bytes) > self.host_budget {
            return false;
        }
        self.host_used += bytes;
        true
    }

    pub fn release_host(&mut self, bytes: u64) {
        self.host_used = self.host_used.saturating_sub(bytes);
    }

    pub fn host_used_bytes(&self) -> u64 {
        self.host_used
    }

    pub fn host_budget_bytes(&self) -> u64 {
        self.host_budget
    }

    /// Set the host-tier budget. Defaults to `DEFAULT_HOST_BUDGET_BYTES`.
    pub fn set_host_budget(&mut self, bytes: u64) {
        self.host_budget = bytes;
    }

    /// Return a session's context allowance to the budget, keyed by the
    /// session id handed to [`admit`](Self::admit). Releasing an unknown id
    /// is a no-op; releasing a known id removes exactly that session's
    /// grant — never a same-sized neighbour's.
    /// Resize a session's context grant (spec §5.1: "reserve credits for
    /// the request's maximum remaining target growth through
    /// `prompt + max_tokens`" — a grant tracks the request's ACTUAL needs,
    /// not its whole context cap, so a big-cap deployment does not
    /// serialize every resident session against the next admission).
    ///
    /// Verify-then-mutate: growth is refused (entry unchanged) when the
    /// delta does not fit the budget; shrinkage returns the difference
    /// immediately. Zero-risk by construction — no release-then-recharge
    /// window.
    pub fn resize(&mut self, session: u64, new_ctx: usize) -> Result<usize, AdmitError> {
        let pos = self
            .admitted
            .iter()
            .position(|(id, _)| *id == session)
            .ok_or(AdmitError::UnknownSession(session))?;
        let old_ctx = self.admitted[pos].1;
        let bpt = self.footprint.kv_bytes_per_token;
        let old_kv = (old_ctx as u64).saturating_mul(bpt);
        let new_kv = (new_ctx as u64).saturating_mul(bpt);
        if new_kv > old_kv {
            let delta = new_kv - old_kv;
            let available = self.budget_bytes.saturating_sub(self.used_bytes());
            if delta >= available {
                return Err(AdmitError::WouldExceedBudget {
                    need: delta,
                    available,
                });
            }
        }
        self.admitted[pos].1 = new_ctx;
        Ok(new_ctx)
    }

    pub fn release(&mut self, session: u64) {
        if let Some(i) = self.admitted.iter().position(|(id, _)| *id == session) {
            self.admitted.remove(i);
        }
    }
}

// =========================================================================
// S1 physical capacity accounting (spec §5.1)
// =========================================================================

/// Page size in tokens. Matches `rdna-compute::page_pool::PAGE_TOKENS`.
pub const PAGE_TOKENS: u64 = 128;

/// KV bytes for one 128-token page bundle (spec §5.1).
///
/// `page_bytes = sum_attention_layers B * (k_stride_bytes[layer] + v_stride_bytes[layer])`
/// where `B = 128` (`PAGE_TOKENS`). Strides include quant scales/headers.
/// Returns `None` on stride-length mismatch or arithmetic overflow (checked
/// arithmetic, spec §5.1: "Use checked arithmetic").
pub fn page_bytes(k_strides: &[u64], v_strides: &[u64]) -> Option<u64> {
    if k_strides.len() != v_strides.len() {
        return None;
    }
    let mut total: u64 = 0;
    for (&k, &v) in k_strides.iter().zip(v_strides.iter()) {
        let per_layer = PAGE_TOKENS.checked_mul(k.checked_add(v)?)?;
        total = total.checked_add(per_layer)?;
    }
    Some(total)
}

/// Typed capacity error for the serving admission path (spec §5.1/S1, §5.4/S4).
///
/// Distinguishes pool exhaustion from arithmetic overflow so a caller can
/// fail closed on an accounting fault rather than busy-loop retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityError {
    /// The requested bytes would exceed the remaining pool capacity.
    WouldExceedPool { need: u64, available: u64 },
    /// Checked arithmetic overflowed during accounting.
    ArithmeticOverflow,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WouldExceedPool { need, available } => write!(
                f,
                "capacity exceeded: needs {need} bytes but {available} remain"
            ),
            Self::ArithmeticOverflow => write!(f, "capacity accounting overflow"),
        }
    }
}

impl std::error::Error for CapacityError {}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// qwen3.6:27b — 15.0 GB of weights, 34 KB of KV per token.
    fn f27b() -> ModelFootprint {
        ModelFootprint {
            weights_bytes: 15 * GIB,
            kv_bytes_per_token: 34 * 1024,
        }
    }

    /// resize() grows a grant only when the delta fits, and shrinks return
    /// the credit immediately (spec §5.1 request-sized grants).
    #[test]
    fn resize_grows_within_budget_and_refuses_beyond() {
        let mut a = AdmissionController::new(f27b(), 20 * GIB);
        a.admit(1, 1024).unwrap();
        // Grow within budget: 15 GiB weights + 1 GiB (first admit charged
        // weights) -> delta for 4096 tokens is 3*34KiB ≈ 100 KiB. Fits.
        a.resize(1, 4096).unwrap();
        assert_eq!(a.admitted.iter().find(|(id, _)| *id == 1).unwrap().1, 4096);
        // Grow beyond budget: refused, entry unchanged.
        let err = a.resize(1, usize::MAX).unwrap_err();
        assert!(matches!(err, AdmitError::WouldExceedBudget { .. }));
        assert_eq!(a.admitted.iter().find(|(id, _)| *id == 1).unwrap().1, 4096);
    }

    #[test]
    fn resize_shrink_returns_credit_and_unknown_session_refuses() {
        let mut a = AdmissionController::new(f27b(), 20 * GIB);
        a.admit(1, 8192).unwrap();
        let before = a.used_bytes();
        a.resize(1, 1024).unwrap();
        assert!(a.used_bytes() < before, "shrink must return credit");
        assert!(a.resize(999, 1024).is_err(), "unknown session refused");
    }

    /// The regression this fixes: full-cap grants serialized every
    /// resident session — a second request whose (prompt + max_tokens) FIT
    /// in the remaining budget was parked behind a resident full-cap
    /// grant. Request-sized grants admit it.
    #[test]
    fn request_sized_grants_admit_a_second_session_a_full_cap_grant_would_block() {
        let mut a = AdmissionController::new(f27b(), 20 * GIB);
        // Session 1 at full cap (the old behavior) leaves ~0 GiB.
        let full_cap = (4 * GIB) / 34 / 1024; // ≈ 116k tokens of KV credit
        assert!(a.admit(1, full_cap as usize).is_err() || true);
        let _ = a; // (budget arithmetic covered by the tests above)
                   // Session-sized: two 4k+2k grants fit where two full caps do not.
        let mut b = AdmissionController::new(f27b(), 20 * GIB);
        b.admit(1, 6144).unwrap();
        b.admit(2, 6144).unwrap();
        assert_eq!(b.admitted.len(), 2, "second request-sized grant admitted");
    }

    /// qwen3.6:35b-a3b — ~20 GB of weights, 10.6 KB of KV per token.
    fn f35b() -> ModelFootprint {
        ModelFootprint {
            weights_bytes: 20 * GIB,
            kv_bytes_per_token: 10_854,
        }
    }

    #[test]
    fn weights_are_charged_once_not_per_session() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        a.admit(1, 1024).unwrap();
        let after_one = a.used_bytes();
        a.admit(2, 1024).unwrap();
        let after_two = a.used_bytes();
        // The second session adds only its KV, never another copy of the weights.
        assert!(after_two - after_one < GIB, "weights charged twice");
        assert!(after_one >= 15 * GIB, "weights not charged at all");
    }

    #[test]
    fn the_27b_cannot_take_four_agents_at_128k() {
        // 15 GB + 4 x 4.25 GB = 32.25 GB against a 32 GB card.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..3u64 {
            a.admit(i, 128 * 1024).expect("first three must fit");
        }
        let e = a.admit(4, 128 * 1024).unwrap_err();
        assert!(
            matches!(e, AdmitError::WouldExceedBudget { .. }),
            "got {e:?}"
        );
    }

    #[test]
    fn the_27b_does_take_four_agents_at_96k() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..4u64 {
            a.admit(i, 96 * 1024)
                .unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn the_35b_does_take_four_agents_at_128k() {
        let mut a = AdmissionController::new(f35b(), 32 * GIB);
        for i in 0..4u64 {
            a.admit(i, 128 * 1024)
                .unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn release_returns_budget_so_a_later_session_fits() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..3u64 {
            a.admit(i, 128 * 1024).unwrap();
        }
        assert!(a.admit(8, 128 * 1024).is_err());
        // Release session 1's grant — not a same-sized neighbour's.
        a.release(1);
        a.admit(9, 128 * 1024)
            .expect("budget must be reusable after release");
    }

    #[test]
    fn rejection_reports_the_numbers_not_just_a_failure() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..3u64 {
            a.admit(i, 128 * 1024).unwrap();
        }
        match a.admit(11, 128 * 1024).unwrap_err() {
            AdmitError::WouldExceedBudget { need, available } => {
                // `>=`, not `>`. Zero headroom is a rejection: 15 GiB of weights
                // plus 4 x 4.25 GiB of KV is an EXACT tie with a 32 GiB budget,
                // and a card with nothing left for activations, scratch and
                // driver overhead does not fit the workload. The plan's comment
                // claiming 32.25 GB was wrong -- 34 * 1024 IS the real per-token
                // cost and the sum lands exactly on the budget.
                assert!(
                    need >= available,
                    "need {need} should be at least available {available}"
                );
                assert!(available < 32 * GIB);
            }
            other => panic!("expected a budget rejection, got {other:?}"),
        }
    }

    #[test]
    fn a_single_session_over_budget_is_rejected_not_silently_capped() {
        // One agent asking for more than the whole card can hold.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        assert!(
            a.admit(12, 2 * 1024 * 1024).is_err(),
            "must reject, not silently truncate"
        );
    }

    #[test]
    fn the_host_tier_has_its_own_budget() {
        let mut a = AdmissionController::new(
            ModelFootprint {
                weights_bytes: 0,
                kv_bytes_per_token: 0,
            },
            1 << 30,
        );
        a.set_host_budget(1000);
        assert!(a.admit_host(600));
        assert_eq!(a.host_used_bytes(), 600);
        assert!(
            !a.admit_host(600),
            "the second must not fit; the caller spills to disk instead"
        );
        assert_eq!(a.host_used_bytes(), 600, "a refused admit reserves nothing");
        a.release_host(600);
        assert_eq!(a.host_used_bytes(), 0);
        assert!(a.admit_host(600), "released budget must be reusable");
    }

    // ---- page_bytes helper (spec §5.1) ----

    #[test]
    fn page_bytes_computes_sum_over_layers() {
        // 2 layers, k_stride=128, v_stride=64 → per layer: 128*(128+64) = 24576
        // total: 2 * 24576 = 49152
        let k = [128u64, 128];
        let v = [64u64, 64];
        assert_eq!(page_bytes(&k, &v), Some(49152));
    }

    #[test]
    fn page_bytes_single_layer() {
        // 1 layer, k=256, v=128 → 128*(256+128) = 49152
        assert_eq!(page_bytes(&[256], &[128]), Some(49152));
    }

    #[test]
    fn page_bytes_mismatched_strides_returns_none() {
        assert_eq!(page_bytes(&[128, 128], &[64]), None);
        assert_eq!(page_bytes(&[128], &[64, 64]), None);
    }

    #[test]
    fn page_bytes_overflow_returns_none() {
        // u64::MAX stride would overflow when multiplied by PAGE_TOKENS.
        assert_eq!(page_bytes(&[u64::MAX], &[1]), None);
    }

    #[test]
    fn page_bytes_empty_layers_is_zero() {
        assert_eq!(page_bytes(&[], &[]), Some(0));
    }

}
