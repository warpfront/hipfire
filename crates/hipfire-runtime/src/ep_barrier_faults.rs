//! CPU fault injection for the barrier families a [`RankPool`] aborts: the row
//! gather, the tile exchange and a third family (the trunk split is a row
//! gather; a bare [`SpinBarrier`] stands in here). No device: only the barrier,
//! crossing-tag and abort protocol runs. Assertions are on outcomes (which
//! error each waiter got, how many timeouts fired); none is a wall-clock bound,
//! and every cross-thread ordering is an explicit sync, so host load cannot
//! change a verdict.

use crate::ep_row_gather::tests::{
    arrived, bare as bare_gather, cross_once, cross_tag, enter_tag, gather_arrived,
};
use crate::ep_row_gather::{GatherKind, SpinBarrier};
use crate::ep_tile_exchange::tests::{bare as bare_tile, exchange_barriers, tile_arrived};
use crate::rank_pool::RankPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

struct Fake;

fn pool() -> RankPool<Fake> {
    let inits =
        vec![Box::new(|_rank: usize| Ok(Fake))
            as Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>];
    RankPool::spawn("barrier-faults", inits, |_: Fake| Ok(())).expect("spawn")
}

/// Block a thread on a barrier no peer will ever reach; it returns the error
/// the wait ended with (`None` if it unexpectedly succeeded).
fn blocked_on<F: FnOnce() -> Option<String> + Send + 'static>(
    wait: F,
) -> std::thread::JoinHandle<Option<String>> {
    std::thread::spawn(wait)
}

/// Spin until `cond` holds. The deadline is only a hang guard for a broken
/// test, far beyond any load-induced delay; it is never part of a verdict.
fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let guard = Instant::now();
    while !cond() {
        assert!(
            guard.elapsed() < Duration::from_secs(120),
            "never reached: {what}"
        );
        std::thread::yield_now();
    }
}

fn is_expiry(message: &str) -> bool {
    message.contains("timed out")
}

#[test]
fn one_abort_wakes_every_registered_barrier_family() {
    let pool = pool();
    let gather = bare_gather(2, Duration::from_secs(30));
    let tile = bare_tile(2, Duration::from_secs(30));
    let third = Arc::new(SpinBarrier::new(2, Duration::from_secs(30)));
    // One callback per family, registered one after another (an earlier
    // OnceLock kept only the first).
    let g = gather.clone();
    pool.set_abort(move || g.abort());
    let t = tile.clone();
    pool.set_abort(move || t.abort());
    let b = third.clone();
    pool.set_abort(move || b.abort());

    let (g, t, b) = (gather.clone(), tile.clone(), third.clone());
    let waiters = [
        blocked_on(move || {
            cross_once(&g, 0, GatherKind::ExpertModulo, 0, 4)
                .err()
                .map(|e| e.to_string())
        }),
        blocked_on(move || exchange_barriers(&t, 0, 0, 0).err().map(|e| e.to_string())),
        blocked_on(move || b.wait().err().map(str::to_string)),
    ];
    // Every waiter is inside its barrier before the failure is injected.
    wait_until("gather waiter blocked", || gather_arrived(&gather) == 1);
    wait_until("tile waiter blocked", || tile_arrived(&tile) == 1);
    wait_until("third waiter blocked", || arrived(&third) == 1);
    assert!(!gather.is_aborted() && !tile.is_aborted() && !third.is_aborted());

    pool.run_all(|| Ok(()), &|_, _: &mut Fake| {
        Err::<(), _>("injected".to_string())
    })
    .expect_err("peer fails");
    for waiter in waiters {
        let message = waiter
            .join()
            .expect("waiter")
            .expect("a blocked waiter must fail on the abort");
        assert!(
            message.contains("aborted") && !is_expiry(&message),
            "woken by the abort, not by a timeout: {message}"
        );
    }
    assert!(gather.is_aborted() && tile.is_aborted() && third.is_aborted());
}

#[test]
fn a_crossing_tag_mismatch_aborts_every_family() {
    let pool = Arc::new(pool());
    let gather = bare_gather(2, Duration::from_secs(30));
    let tile = bare_tile(2, Duration::from_secs(30));
    let (g, t) = (gather.clone(), tile.clone());
    pool.set_abort(move || g.abort());
    pool.set_abort(move || t.abort());

    // A bystander blocked on the tile exchange's barrier.
    let t = tile.clone();
    let bystander = blocked_on(move || exchange_barriers(&t, 0, 0, 0).err().map(|e| e.to_string()));
    wait_until("bystander blocked", || tile_arrived(&tile) == 1);

    // Layers 3 and 5 share a buffer (layer % 2), so the same barrier and the
    // same slot are compared and the diagnostic names both tags. Both ranks
    // publish their tag and only then cross: where the ranks meet is fixed,
    // not a scheduling accident.
    let both_entered = Arc::new(Barrier::new(2));
    let (g0, g1) = (gather.clone(), gather.clone());
    let (s0, s1) = (both_entered.clone(), both_entered.clone());
    let error = pool
        .run_all(
            move || {
                let tag = enter_tag(&g0, 0, GatherKind::ExpertModulo, 3, 8);
                s0.wait();
                cross_tag(&g0, 0, tag).map_err(|e| e.to_string())
            },
            &move |_, _: &mut Fake| {
                let tag = enter_tag(&g1, 1, GatherKind::ExpertModulo, 5, 8);
                s1.wait();
                cross_tag(&g1, 1, tag).map_err(|e| e.to_string())
            },
        )
        .expect_err("mismatch");
    assert!(error.contains("different gathers"), "{error}");
    // "rank R entered [<expected>] but rank P's slot for buffer B holds [<actual>]":
    // whichever rank reports first, the two tags are the two crossings.
    let between = |after: &str| -> String {
        let start = error
            .find(after)
            .unwrap_or_else(|| panic!("{after:?} in {error}"))
            + after.len();
        let len = error[start..].find(']').expect("closing bracket");
        error[start..start + len].to_string()
    };
    let (entered, holds) = (between("entered ["), between("holds ["));
    let pair = (entered.contains("layer 3"), entered.contains("layer 5"));
    let other = (holds.contains("layer 3"), holds.contains("layer 5"));
    assert!(
        (pair == (true, false) && other == (false, true))
            || (pair == (false, true) && other == (true, false)),
        "expected and actual tags name layers 3 and 5: {error}"
    );
    let message = bystander
        .join()
        .expect("bystander")
        .expect("bystander fails");
    assert!(
        message.contains("aborted") && !is_expiry(&message),
        "{message}"
    );
    assert!(gather.is_aborted() && tile.is_aborted());
}

#[test]
fn a_missing_participant_times_out_once_and_aborts_the_rest() {
    let pool = pool();
    // Only the gather has a short budget; the other families' own timeouts are
    // far beyond any test duration, so if the first expiry did not abort them
    // they would never be released and the joins below would hang. That the
    // joins return, with the abort (not an expiry) as their reason, is the
    // proof that exactly one timeout fired for the whole mesh.
    let gather = bare_gather(2, Duration::from_millis(400));
    let tile = bare_tile(2, Duration::from_secs(3600));
    let third = Arc::new(SpinBarrier::new(2, Duration::from_secs(3600)));
    let g = gather.clone();
    pool.set_abort(move || g.abort());
    let t = tile.clone();
    pool.set_abort(move || t.abort());
    let b = third.clone();
    pool.set_abort(move || b.abort());

    let (t, b) = (tile.clone(), third.clone());
    let waiters = [
        blocked_on(move || exchange_barriers(&t, 1, 0, 0).err().map(|e| e.to_string())),
        blocked_on(move || b.wait().err().map(str::to_string)),
    ];
    wait_until("tile waiter blocked", || tile_arrived(&tile) == 1);
    wait_until("third waiter blocked", || arrived(&third) == 1);

    let g = gather.clone();
    // Rank 1 never joins the gather.
    let error = pool
        .run_all(
            move || cross_once(&g, 0, GatherKind::ExpertModulo, 0, 4).map_err(|e| e.to_string()),
            &|_, _: &mut Fake| Ok(()),
        )
        .expect_err("rank 1 never arrives");
    assert!(is_expiry(&error), "{error}");
    assert!(
        error.contains("rank 1 has entered no crossing"),
        "names the missing rank: {error}"
    );
    let mut expirations = 1; // the gather's, reported by `run_all`
    for waiter in waiters {
        let message = waiter.join().expect("waiter").expect("waiter fails");
        assert!(
            message.contains("aborted"),
            "released by the abort: {message}"
        );
        if is_expiry(&message) {
            expirations += 1;
        }
    }
    assert_eq!(
        expirations, 1,
        "one timeout for the mesh, not one per family"
    );
    assert!(gather.is_aborted() && tile.is_aborted() && third.is_aborted());
}

#[test]
fn abort_is_idempotent() {
    let pool = pool();
    let gather = bare_gather(2, Duration::from_secs(30));
    let tile = bare_tile(2, Duration::from_secs(30));
    let calls = Arc::new(AtomicUsize::new(0));
    let (g, c) = (gather.clone(), calls.clone());
    pool.set_abort(move || {
        g.abort();
        c.fetch_add(1, Ordering::SeqCst);
    });
    let t = tile.clone();
    pool.set_abort(move || t.abort());

    pool.poison_mesh("first");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(gather.is_aborted() && tile.is_aborted());
    // Aborting again, through the pool and directly, changes nothing.
    pool.poison_mesh("second");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "callbacks run per poisoning"
    );
    pool.run_all(|| Ok(()), &|_, _: &mut Fake| Ok(()))
        .expect_err("poisoned");
    gather.abort();
    tile.abort();
    assert_eq!(
        pool.poisoned().as_deref(),
        Some("first"),
        "the first failure stays the recorded one"
    );
    assert!(gather.is_aborted() && tile.is_aborted());
    let g = cross_once(&gather, 0, GatherKind::ExpertModulo, 0, 1)
        .expect_err("aborted gather")
        .to_string();
    let t = exchange_barriers(&tile, 0, 0, 0)
        .expect_err("aborted tile")
        .to_string();
    assert!(g.contains("aborted") && t.contains("aborted"), "{g} / {t}");
}
