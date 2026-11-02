//! The process-wide CPU routed-expert splice pool.
//!
//! The splice's GEMV regions are memory-bandwidth bound row dots; running them
//! on every logical CPU (SMT siblings share a core's load ports and L1/L2)
//! measured slower than one worker per physical core (Strix Halo: 10 experts ×
//! 1 row, 365 µs at 16 threads vs 438 µs at 32). This module builds one
//! lazily-initialised rayon pool sized to the physical cores of the process's
//! allowed CPU set, with each worker pinned to one logical CPU of a distinct
//! core. The global rayon pool and every other caller are untouched.
//!
//! Arithmetic is independent of the pool: the shared GEMV computes each output
//! element as one row dot, so the same bytes come out for any worker count
//! (see the `gemv` determinism test).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Splice pool configuration, resolved by the caller from its knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplicePoolConfig {
    /// Worker count; `0` = one per physical core in the allowed CPU set.
    pub threads: usize,
    /// Pin each worker to a logical CPU of a distinct physical core.
    pub pin: bool,
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// The splice pool. Built on first use from `cfg`; later calls return the same
/// pool and ignore their `cfg` (the knobs are process-lifetime).
pub fn splice_pool(cfg: SplicePoolConfig) -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| build_pool(cfg))
}

/// Run `f` on the splice pool (built from `cfg` on first use). Every rayon
/// region `f` starts uses the pool's workers, not the global pool's.
pub fn install<R: Send>(cfg: SplicePoolConfig, f: impl FnOnce() -> R + Send) -> R {
    splice_pool(cfg).install(f)
}

/// Run `f(i, chunk_i)` for every `chunk`-long piece of `data` in the current
/// rayon pool (the splice pool when called inside [`install`]). The pieces are
/// disjoint, so `f` may write its piece freely; this is how the arm's per-slot
/// middle (SwiGLU + hidden rotation) joins the GEMV regions on the same
/// workers instead of running serially on the calling worker.
pub fn par_for_each_chunk_mut<T: Send>(
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    use rayon::prelude::*;
    data.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i, c)| f(i, c));
}

fn build_pool(cfg: SplicePoolConfig) -> rayon::ThreadPool {
    let fallback_threads = || {
        let n = if cfg.threads > 0 {
            cfg.threads
        } else {
            (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1)
        };
        n.max(1)
    };
    let unpinned = |n: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("hipfire-splice-{i}"))
            .build()
            .expect("splice pool: thread pool build failed")
    };
    if !cfg.pin {
        let n = if cfg.threads > 0 {
            cfg.threads
        } else {
            physical_core_cpus().map_or_else(fallback_threads, |c| c.primaries.len().max(1))
        };
        eprintln!("cpu exec: splice pool {n} threads, unpinned (HIPFIRE_CPU_EXPERT_PIN=0)");
        return unpinned(n);
    }
    let Some(cpus) = physical_core_cpus() else {
        let n = fallback_threads();
        eprintln!(
            "cpu exec: splice pool falling back to {n} unpinned threads (CPU topology unreadable)"
        );
        return unpinned(n);
    };
    let cores = cpus.primaries.len();
    let n = if cfg.threads > 0 { cfg.threads } else { cores };
    // Workers 0..cores take one CPU per physical core; beyond that (explicit
    // override) they take the remaining SMT siblings, and past the allowed set
    // they run unpinned.
    let order: Vec<usize> = cpus
        .primaries
        .iter()
        .chain(cpus.secondaries.iter())
        .copied()
        .collect();
    let failed = std::sync::Arc::new(AtomicBool::new(false));
    let handler_failed = failed.clone();
    let handler_order = order.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("hipfire-splice-{i}"))
        .start_handler(move |i| {
            if let Some(&cpu) = handler_order.get(i) {
                if !pin_current_thread(cpu) {
                    handler_failed.store(true, Ordering::Release);
                }
            }
        })
        .build()
        .expect("splice pool: thread pool build failed");
    // `start_handler` runs on each worker before it takes work; a broadcast
    // completes only after every worker has started, so `failed` is final here.
    pool.broadcast(|_| {});
    if failed.load(Ordering::Acquire) {
        let n = fallback_threads();
        eprintln!("cpu exec: splice pool pinning failed; using {n} unpinned threads");
        drop(pool);
        return unpinned(n);
    }
    eprintln!(
        "cpu exec: splice pool {n} threads pinned ({cores} physical cores in the allowed CPU set)"
    );
    pool
}

/// Allowed logical CPUs grouped by physical core.
struct PhysicalCpus {
    /// One allowed CPU (the lowest id) per distinct physical core, by core.
    primaries: Vec<usize>,
    /// The remaining allowed SMT siblings, in ascending id order.
    secondaries: Vec<usize>,
}

fn physical_core_cpus() -> Option<PhysicalCpus> {
    let allowed = allowed_cpus()?;
    let mut cores: Vec<(usize, usize)> = Vec::new(); // (core key, chosen cpu)
    let mut secondaries = Vec::new();
    for &cpu in &allowed {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let text = std::fs::read_to_string(path).ok()?;
        let key = parse_cpu_list(text.trim())?.into_iter().min()?;
        if cores.iter().any(|&(k, _)| k == key) {
            secondaries.push(cpu);
        } else {
            cores.push((key, cpu));
        }
    }
    if cores.is_empty() {
        return None;
    }
    Some(PhysicalCpus {
        primaries: cores.into_iter().map(|(_, c)| c).collect(),
        secondaries,
    })
}

/// Parse a kernel CPU list such as `0,16` or `0-3,8`.
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                if a > b {
                    return None;
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse().ok()?),
        }
    }
    Some(out)
}

#[cfg(target_os = "linux")]
fn allowed_cpus() -> Option<Vec<usize>> {
    // SAFETY: `set` is a plain bitmask the kernel fills; size matches the call.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        let v: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect();
        (!v.is_empty()).then_some(v)
    }
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpus() -> Option<Vec<usize>> {
    None
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) -> bool {
    // SAFETY: `set` is initialised zeroed and only the target bit is set.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) -> bool {
    false
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cpu_list_parses_ranges_and_commas() {
        assert_eq!(parse_cpu_list("0,16"), Some(vec![0, 16]));
        assert_eq!(parse_cpu_list("0-3,8"), Some(vec![0, 1, 2, 3, 8]));
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a"), None);
    }

    #[test]
    fn splice_pool_runs_work() {
        let pool = splice_pool(SplicePoolConfig {
            threads: 0,
            pin: true,
        });
        let sum: usize = pool.install(|| {
            use rayon::prelude::*;
            (0..1000usize).into_par_iter().sum()
        });
        assert_eq!(sum, 499_500);
        assert!(pool.current_num_threads() >= 1);
    }
}
