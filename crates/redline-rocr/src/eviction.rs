//! Eviction-aware completion-wait deadline.
//!
//! KFD evicts every queue of a process while it migrates or compacts the host
//! pages the process registered with the GPU (userptr BOs). A replay that is
//! in flight stalls for the whole eviction (episodes of up to 2.2 s were
//! measured, `inkling/redline/receipts/s5/slowload/RESULT.md`), and a fixed
//! 5 s [`crate::DEFAULT_WAIT_TIMEOUT`] can be hit by an unlucky tail while the
//! work is intact. HIP simply waits in that situation, so the retained-PM4
//! wait must not convert an environmental eviction into a failed turn.
//!
//! The kernel exports the accumulated eviction time per process and GPU at
//! `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms`. This module reads
//! that evidence only after a wait has already outlived the 250 ms grace
//! sample (27x a 9 ms step; ordinary waits never reach it), so the normal path
//! costs nothing beyond one extra comparison per poll.
//!
//! Policy ([`EvictionDeadline`]): the base deadline is unchanged. When it
//! expires, if the eviction counter rose since the previous sample, the
//! deadline is extended by one base window, up to [`EVICTION_WAIT_CEILING`]
//! in total. A flat counter, an unreadable counter or a platform without the
//! sysfs files expires exactly at the base deadline, as before.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

/// Hard ceiling for a wait that keeps being extended, measured from wait
/// start. The longest observed eviction episode is 2.2 s, so 60 s leaves a
/// 27x margin over the evidence, while a pathological, endlessly re-evicting
/// process still fails in one minute instead of wedging the caller (a genuine
/// GPU hang with no eviction still fails at the 5 s base deadline).
pub const EVICTION_WAIT_CEILING: Duration = Duration::from_secs(60);

/// First eviction sample of a wait, taken only once the wait has already run
/// this long. It is the baseline the expiry-time sample is compared against.
const EVICTION_GRACE_SAMPLE: Duration = Duration::from_millis(250);

/// Source of the cumulative process eviction counter, in milliseconds.
pub trait EvictionCounter {
    /// Cumulative eviction time of this process across all GPUs, or `None`
    /// when no evidence is available (non-Linux, KFD sysfs missing or
    /// unreadable).
    fn evicted_ms(&mut self) -> Option<u64>;
}

/// `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`, summed over GPUs.
#[derive(Debug, Clone)]
pub struct KfdEvictionCounter {
    proc_dir: PathBuf,
}

impl KfdEvictionCounter {
    /// Counter for the calling process.
    #[must_use]
    pub fn for_current_process() -> Self {
        Self::at(PathBuf::from(format!(
            "/sys/class/kfd/kfd/proc/{}",
            std::process::id()
        )))
    }

    /// Counter rooted at an explicit `proc/<pid>` directory.
    #[must_use]
    pub fn at(proc_dir: PathBuf) -> Self {
        Self { proc_dir }
    }
}

impl EvictionCounter for KfdEvictionCounter {
    fn evicted_ms(&mut self) -> Option<u64> {
        let mut total = 0_u64;
        let mut seen = false;
        for entry in fs::read_dir(&self.proc_dir).ok()? {
            let entry = entry.ok()?;
            if !entry.file_name().to_string_lossy().starts_with("stats_") {
                continue;
            }
            let text = fs::read_to_string(entry.path().join("evicted_ms")).ok()?;
            total = total.saturating_add(text.trim().parse::<u64>().ok()?);
            seen = true;
        }
        seen.then_some(total)
    }
}

/// Deadline state machine for one wait. Pure apart from the injected counter
/// and the `eprintln!` extension log; time is the caller's elapsed duration.
pub struct EvictionDeadline<'c, C: EvictionCounter + ?Sized> {
    counter: &'c mut C,
    base: Duration,
    ceiling: Duration,
    grace: Duration,
    deadline: Duration,
    /// Next elapsed instant at which `expired` has anything to do.
    next_event: Duration,
    baseline: Option<u64>,
    sampled_grace: bool,
}

impl<'c, C: EvictionCounter + ?Sized> EvictionDeadline<'c, C> {
    /// `base` is the unextended deadline; `ceiling` bounds the total (never
    /// below `base`).
    pub fn new(counter: &'c mut C, base: Duration, ceiling: Duration) -> Self {
        let grace = EVICTION_GRACE_SAMPLE.min(base / 2);
        Self {
            counter,
            base,
            // A zero base never extends (it would otherwise re-sample forever).
            ceiling: if base.is_zero() {
                base
            } else {
                ceiling.max(base)
            },
            grace,
            deadline: base,
            next_event: grace,
            baseline: None,
            sampled_grace: false,
        }
    }

    /// Current deadline (equals `base` unless eviction extended it).
    #[must_use]
    pub fn deadline(&self) -> Duration {
        self.deadline
    }

    /// `true` when the wait must fail now. Cheap (one comparison) until the
    /// grace sample point; touches the counter only at the grace point and at
    /// each deadline expiry.
    pub fn expired(&mut self, elapsed: Duration) -> bool {
        if elapsed < self.next_event {
            return false;
        }
        if !self.sampled_grace {
            self.sampled_grace = true;
            self.baseline = self.counter.evicted_ms();
            self.next_event = self.deadline;
            if elapsed < self.deadline {
                return false;
            }
        }
        if self.deadline >= self.ceiling {
            return true;
        }
        let (Some(previous), Some(current)) = (self.baseline, self.counter.evicted_ms()) else {
            return true;
        };
        if current <= previous {
            return true;
        }
        self.baseline = Some(current);
        self.deadline = (self.deadline + self.base).min(self.ceiling);
        self.next_event = self.deadline;
        eprintln!(
            "HIPFIRE_REDLINE_EVICT_WAIT_EXTENDED evicted_ms_delta={} extension_ms={} deadline_ms={} ceiling_ms={}",
            current - previous,
            (self.deadline - self.base).as_millis(),
            self.deadline.as_millis(),
            self.ceiling.as_millis(),
        );
        false
    }

    /// Grace sample point (exposed for diagnostics and tests).
    #[must_use]
    pub fn grace(&self) -> Duration {
        self.grace
    }
}

/// How a [`wait_extending`] loop ended.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WaitEnd {
    /// The completion predicate became true.
    Complete,
    /// The (possibly extended) deadline expired; carries it.
    Expired(Duration),
}

/// Generic wait loop shared by the queue wait and its tests: `ready` is
/// polled (and may fail, e.g. on a latched queue fault) before every
/// deadline check, `now` yields the elapsed time, `pause` spins between
/// polls.
pub fn wait_extending<E, C: EvictionCounter + ?Sized>(
    deadline: &mut EvictionDeadline<'_, C>,
    mut now: impl FnMut() -> Duration,
    mut ready: impl FnMut() -> Result<bool, E>,
    mut pause: impl FnMut(),
) -> Result<WaitEnd, E> {
    loop {
        if ready()? {
            return Ok(WaitEnd::Complete);
        }
        if deadline.expired(now()) {
            return Ok(WaitEnd::Expired(deadline.deadline()));
        }
        pause();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const BASE: Duration = Duration::from_secs(5);

    /// Fake counter: rises by `step_ms` per window of simulated time.
    struct Sim<'a> {
        clock: &'a Cell<Duration>,
        step_ms_per_s: u64,
        reads: u32,
        available: bool,
    }

    impl EvictionCounter for Sim<'_> {
        fn evicted_ms(&mut self) -> Option<u64> {
            self.reads += 1;
            self.available
                .then(|| self.clock.get().as_secs() * self.step_ms_per_s)
        }
    }

    fn run(
        step_ms_per_s: u64,
        available: bool,
        complete_at: Option<Duration>,
    ) -> (WaitEnd, Duration, u32) {
        let clock = Cell::new(Duration::ZERO);
        let mut sim = Sim {
            clock: &clock,
            step_ms_per_s,
            reads: 0,
            available,
        };
        let mut deadline = EvictionDeadline::new(&mut sim, BASE, EVICTION_WAIT_CEILING);
        let end = wait_extending::<(), _>(
            &mut deadline,
            || clock.get(),
            || Ok(complete_at.is_some_and(|at| clock.get() >= at)),
            // 100 ms of simulated time per poll
            || clock.set(clock.get() + Duration::from_millis(100)),
        )
        .unwrap();
        let reads = sim.reads;
        (end, clock.get(), reads)
    }

    #[test]
    fn rising_counter_survives_base_deadline_and_completes() {
        let (end, at, _) = run(400, true, Some(Duration::from_secs(12)));
        assert_eq!(end, WaitEnd::Complete);
        assert!(at >= Duration::from_secs(12));
    }

    #[test]
    fn flat_counter_times_out_at_base_deadline_exactly() {
        let (end, at, reads) = run(0, true, None);
        assert_eq!(end, WaitEnd::Expired(BASE));
        assert_eq!(at, BASE);
        // grace baseline + one expiry sample, nothing else
        assert_eq!(reads, 2);
    }

    #[test]
    fn unavailable_counter_is_exactly_todays_behaviour() {
        let (end, at, _) = run(400, false, None);
        assert_eq!(end, WaitEnd::Expired(BASE));
        assert_eq!(at, BASE);
    }

    #[test]
    fn counter_rising_forever_stops_at_hard_ceiling() {
        let (end, at, _) = run(400, true, None);
        assert_eq!(end, WaitEnd::Expired(EVICTION_WAIT_CEILING));
        assert_eq!(at, EVICTION_WAIT_CEILING);
    }

    #[test]
    fn extension_needs_a_rise_in_every_window() {
        // Rises only until 7 s, then freezes: the 5 s expiry sees a rise
        // (extend to 10 s), the 10 s expiry still sees the 5-7 s rise
        // (extend to 15 s), the 15 s expiry sees a flat window and fails.
        struct Frozen<'a>(&'a Cell<Duration>);
        impl EvictionCounter for Frozen<'_> {
            fn evicted_ms(&mut self) -> Option<u64> {
                Some(self.0.get().min(Duration::from_secs(7)).as_millis() as u64)
            }
        }
        let clock = Cell::new(Duration::ZERO);
        let mut sim = Frozen(&clock);
        let mut deadline = EvictionDeadline::new(&mut sim, BASE, EVICTION_WAIT_CEILING);
        let end = wait_extending::<(), _>(
            &mut deadline,
            || clock.get(),
            || Ok(false),
            || clock.set(clock.get() + Duration::from_millis(100)),
        )
        .unwrap();
        assert_eq!(end, WaitEnd::Expired(Duration::from_secs(15)));
        assert_eq!(clock.get(), Duration::from_secs(15));
    }

    #[test]
    fn normal_path_never_reads_the_counter() {
        let (end, _, reads) = run(400, true, Some(Duration::from_millis(200)));
        assert_eq!(end, WaitEnd::Complete);
        assert_eq!(reads, 0);
    }

    #[test]
    fn faults_propagate_without_extension() {
        let clock = Cell::new(Duration::ZERO);
        let mut sim = Sim {
            clock: &clock,
            step_ms_per_s: 400,
            reads: 0,
            available: true,
        };
        let mut deadline = EvictionDeadline::new(&mut sim, BASE, EVICTION_WAIT_CEILING);
        let result = wait_extending(
            &mut deadline,
            || clock.get(),
            || {
                if clock.get() >= Duration::from_secs(1) {
                    Err("fault")
                } else {
                    Ok(false)
                }
            },
            || clock.set(clock.get() + Duration::from_millis(100)),
        );
        assert_eq!(result, Err("fault"));
    }

    #[test]
    fn kfd_counter_sums_stats_dirs_and_reports_missing_sysfs() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!(".evict-wait-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (gpu, ms) in [("1", "10\n"), ("2", "32\n")] {
            let dir = root.join(format!("stats_{gpu}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("evicted_ms"), ms).unwrap();
        }
        fs::create_dir_all(root.join("queues")).unwrap();
        assert_eq!(KfdEvictionCounter::at(root.clone()).evicted_ms(), Some(42));
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(KfdEvictionCounter::at(root).evicted_ms(), None);
    }
}
