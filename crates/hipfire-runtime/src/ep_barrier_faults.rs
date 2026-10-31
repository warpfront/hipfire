//! CPU fault injection for the barrier families a [`RankPool`] aborts: the row
//! gather, the tile exchange and a third family (the trunk split is a row
//! gather; a bare [`SpinBarrier`] stands in here). No device: only the barrier,
//! crossing-tag and abort protocol runs.

use crate::ep_row_gather::tests::{bare as bare_gather, cross_once};
use crate::ep_row_gather::{GatherKind, SpinBarrier};
use crate::ep_tile_exchange::tests::{bare as bare_tile, exchange_barriers};
use crate::rank_pool::RankPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Fake;

fn pool() -> RankPool<Fake> {
    let inits =
        vec![Box::new(|_rank: usize| Ok(Fake))
            as Box<dyn FnOnce(usize) -> Result<Fake, String> + Send>];
    RankPool::spawn("barrier-faults", inits, |_: Fake| Ok(())).expect("spawn")
}

/// Block a thread on a barrier no peer will ever reach; it reports how long it
/// waited and whether it failed.
fn blocked_on<F: FnOnce() -> bool + Send + 'static>(
    wait: F,
) -> std::thread::JoinHandle<(bool, Duration)> {
    std::thread::spawn(move || {
        let start = Instant::now();
        let failed = wait();
        (failed, start.elapsed())
    })
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
        blocked_on(move || cross_once(&g, 0, GatherKind::ExpertModulo, 0, 4).is_err()),
        blocked_on(move || exchange_barriers(&t, 0, 0, 0).is_err()),
        blocked_on(move || b.wait().is_err()),
    ];
    std::thread::sleep(Duration::from_millis(100));
    let started = Instant::now();
    pool.run_all(|| Ok(()), &|_, _: &mut Fake| {
        Err::<(), _>("injected".to_string())
    })
    .expect_err("peer fails");
    for waiter in waiters {
        let (failed, waited) = waiter.join().expect("waiter");
        assert!(failed, "a blocked waiter must fail on the abort");
        assert!(
            waited < Duration::from_secs(5),
            "woken promptly, not by its timeout: {waited:?}"
        );
    }
    assert!(started.elapsed() < Duration::from_secs(5));
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
    let bystander = blocked_on(move || exchange_barriers(&t, 0, 0, 0).is_err());
    std::thread::sleep(Duration::from_millis(100));

    let (g0, g1) = (gather.clone(), gather.clone());
    let started = Instant::now();
    // Rank 0 gathers layer 3, rank 1 gathers layer 4: the same barrier, different
    // crossings.
    let error = pool
        .run_all(
            move || cross_once(&g0, 0, GatherKind::ExpertModulo, 3, 8).map_err(|e| e.to_string()),
            &move |_, _: &mut Fake| {
                cross_once(&g1, 1, GatherKind::ExpertModulo, 4, 8).map_err(|e| e.to_string())
            },
        )
        .expect_err("mismatch");
    assert!(error.contains("different gathers"), "{error}");
    assert!(
        error.contains("layer 3") && error.contains("layer 4"),
        "{error}"
    );
    let (failed, waited) = bystander.join().expect("bystander");
    assert!(failed && waited < Duration::from_secs(5), "{waited:?}");
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(gather.is_aborted() && tile.is_aborted());
}

#[test]
fn a_missing_participant_fails_within_one_timeout_budget() {
    let pool = pool();
    let budget = Duration::from_millis(400);
    // The first family times out after `budget`; the others would wait a long
    // time on their own timeouts.
    let gather = bare_gather(2, budget);
    let tile = bare_tile(2, Duration::from_secs(60));
    let third = Arc::new(SpinBarrier::new(2, Duration::from_secs(60)));
    let g = gather.clone();
    pool.set_abort(move || g.abort());
    let t = tile.clone();
    pool.set_abort(move || t.abort());
    let b = third.clone();
    pool.set_abort(move || b.abort());

    let (t, b) = (tile.clone(), third.clone());
    let waiters = [
        blocked_on(move || exchange_barriers(&t, 1, 0, 0).is_err()),
        blocked_on(move || b.wait().is_err()),
    ];
    std::thread::sleep(Duration::from_millis(50));
    let g = gather.clone();
    let started = Instant::now();
    // Rank 1 never joins the gather.
    let error = pool
        .run_all(
            move || cross_once(&g, 0, GatherKind::ExpertModulo, 0, 4).map_err(|e| e.to_string()),
            &|_, _: &mut Fake| Ok(()),
        )
        .expect_err("rank 1 never arrives");
    assert!(error.contains("timed out"), "{error}");
    assert!(
        error.contains("rank 1 has entered no crossing"),
        "names the missing rank: {error}"
    );
    for waiter in waiters {
        let (failed, _) = waiter.join().expect("waiter");
        assert!(failed);
    }
    let total = started.elapsed();
    assert!(total >= budget, "{total:?}");
    assert!(
        total < budget * 4,
        "one timeout budget in total, not one per family: {total:?}"
    );
}

#[test]
fn abort_is_idempotent() {
    let pool = pool();
    let gather = bare_gather(2, Duration::from_secs(30));
    let tile = bare_tile(2, Duration::from_secs(30));
    let calls = Arc::new(AtomicUsize::new(0));
    let (g, t, c) = (gather.clone(), tile.clone(), calls.clone());
    pool.set_abort(move || {
        g.abort();
        c.fetch_add(1, Ordering::SeqCst);
    });
    let t2 = tile.clone();
    pool.set_abort(move || t2.abort());
    let _ = t;

    pool.poison_mesh("first");
    let after_first = calls.load(Ordering::SeqCst);
    assert_eq!(after_first, 1);
    pool.poison_mesh("second");
    pool.run_all(|| Ok(()), &|_, _: &mut Fake| Ok(()))
        .expect_err("poisoned");
    gather.abort();
    tile.abort();
    assert!(calls.load(Ordering::SeqCst) >= after_first);
    assert_eq!(
        pool.poisoned().as_deref(),
        Some("first"),
        "the first failure stays the recorded one"
    );
    assert!(gather.is_aborted() && tile.is_aborted());
    assert!(cross_once(&gather, 0, GatherKind::ExpertModulo, 0, 1).is_err());
    assert!(exchange_barriers(&tile, 0, 0, 0).is_err());
}
