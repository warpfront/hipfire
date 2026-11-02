// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `moe-oracle` observer channel: which router route actually ran, and what it
//! left in the router score buffer.
//!
//! The sealed MoE route stages record one [`RouterRouteObservation`] after the
//! router branch has been chosen (admission can refuse a fused route, so the
//! flag alone never says which kernel ran). The oracle hook on the same thread
//! reads it with [`take`] right after the sealed program returns. The channel
//! is a thread-local `Cell`; it carries no GPU state, adds no launch, and the
//! whole module is compiled only under `feature = "moe-oracle"`.

use std::cell::Cell;

/// What the router score buffer holds once the route stage has finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterScoresAfterRoute {
    /// Row softmax probabilities, written in place by the incumbent softmax.
    Probabilities,
    /// The logits the route left in the buffer, untouched: a fused router
    /// softmaxes inside its own kernel and never writes the buffer. Whether
    /// the recipe's BF16 rounding is already applied to them is the
    /// observation's `round_before_softmax` fact.
    Logits,
    /// Sigmoid scores written in place by the sigmoid router.
    Sigmoid,
}

/// One executed router route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterRouteObservation {
    /// Stable name of the branch that ran.
    pub route: &'static str,
    /// What that branch left in the score buffer.
    pub scores: RouterScoresAfterRoute,
    /// The recipe rounds router logits to BF16 before softmax, but this route
    /// skipped the in-place round because its kernel rounds the logits it
    /// reads internally (the grouped top-10 router, masked or not), so the
    /// buffer still holds unrounded logits. A consumer that softmaxes a copy
    /// must BF16-round the copy first to reproduce what the kernel consumed.
    /// `false` when the buffer is already rounded in place (or the recipe does
    /// not round) and for routes that leave probabilities or sigmoid scores.
    pub round_before_softmax: bool,
}

thread_local! {
    static LAST: Cell<Option<RouterRouteObservation>> = const { Cell::new(None) };
}

/// Record the executed route, what it left in the score buffer, and whether
/// that logits buffer still needs the recipe's BF16 rounding before an
/// observer softmax (`round_before_softmax`, see [`RouterRouteObservation`]).
pub(crate) fn record(
    route: &'static str,
    scores: RouterScoresAfterRoute,
    round_before_softmax: bool,
) {
    LAST.with(|last| {
        last.set(Some(RouterRouteObservation {
            route,
            scores,
            round_before_softmax,
        }))
    });
}

/// Take (and clear) the last route executed on this thread.
pub fn take() -> Option<RouterRouteObservation> {
    LAST.with(Cell::take)
}

/// Drop any stale observation before a sealed MoE call is issued.
pub fn clear() {
    LAST.with(|last| last.set(None));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_returns_the_last_route_once_and_clear_drops_it() {
        clear();
        assert_eq!(take(), None);
        record("a", RouterScoresAfterRoute::Probabilities, false);
        record("b", RouterScoresAfterRoute::Logits, false);
        assert_eq!(
            take(),
            Some(RouterRouteObservation {
                route: "b",
                scores: RouterScoresAfterRoute::Logits,
                round_before_softmax: false,
            })
        );
        assert_eq!(take(), None);
        record("c", RouterScoresAfterRoute::Sigmoid, false);
        clear();
        assert_eq!(take(), None);
    }

    #[test]
    fn rounding_fact_is_carried_as_recorded() {
        clear();
        record("g", RouterScoresAfterRoute::Logits, true);
        assert_eq!(take().map(|o| o.round_before_softmax), Some(true));
        record("g", RouterScoresAfterRoute::Logits, false);
        assert_eq!(take().map(|o| o.round_before_softmax), Some(false));
        record("p", RouterScoresAfterRoute::Probabilities, false);
        assert_eq!(take().map(|o| o.round_before_softmax), Some(false));
    }
}
