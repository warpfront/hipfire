// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared CPU-affinity and memory admission for CPU-only kernel compilation.

pub const BYTES_PER_JOB: u64 = 1536 * 1024 * 1024;

/// Admit workers after subtracting the caller's load/staging/safety reserve.
/// Pass the tighter of host MemAvailable and remaining cgroup memory.
/// Unknown memory permits one worker; known insufficient memory is an error.
pub fn compile_job_budget(
    cores: usize,
    available_bytes: Option<u64>,
    reserved_bytes: u64,
    override_jobs: Option<usize>,
) -> Result<usize, String> {
    let cores = cores.max(1);
    let requested = match override_jobs {
        Some(0) => return Err("compile jobs must be a positive integer".into()),
        Some(jobs) => jobs,
        None => cores.saturating_sub((cores / 8).max(1)).max(1),
    };
    let memory_jobs = match available_bytes {
        None => 1,
        Some(bytes) => {
            let jobs = bytes.saturating_sub(reserved_bytes) / BYTES_PER_JOB;
            if jobs == 0 {
                return Err(
                    "insufficient available memory for one 1536-MiB compile job after reserve"
                        .into(),
                );
            }
            usize::try_from(jobs).unwrap_or(usize::MAX)
        }
    };
    Ok(requested.min(cores).min(memory_jobs))
}

/// Parse a positive environment override without permitting an unbounded value.
pub fn parse_job_override(name: &str, value: Option<&str>) -> Result<Option<usize>, String> {
    value
        .map(|value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|&jobs| jobs > 0)
                .ok_or_else(|| format!("{name} must be a positive integer"))
        })
        .transpose()
}

/// `MemAvailable - CMA` from `/proc/meminfo` (and `/proc/zoneinfo`) text; the
/// shared rule in [`crate::uma_memory::meminfo_available_bytes`].
pub fn available_memory(text: &str, zoneinfo: Option<&str>) -> Option<u64> {
    crate::uma_memory::meminfo_available_bytes(text, zoneinfo)
}

fn memory_remaining(limit: &str, current: &str) -> Option<u64> {
    Some(
        limit
            .trim()
            .parse::<u64>()
            .ok()?
            .saturating_sub(current.trim().parse::<u64>().ok()?),
    )
}

/// Probe CPU affinity and available RAM, retaining a 1-GiB safety reserve.
/// Model loaders pass their additional staging reserve to the pure function.
pub fn host_compile_job_budget() -> Result<usize, String> {
    let cores = std::thread::available_parallelism()
        .map(usize::from)
        .map_err(|error| format!("cannot determine compile CPU affinity: {error}"))?;
    let value = hipfire_config::developer_var("HIPFIRE_JIT_JOBS").ok();
    let jobs = parse_job_override("HIPFIRE_JIT_JOBS", value.as_deref())?;
    compile_job_budget(cores, host_available_memory(), 1024 * 1024 * 1024, jobs)
}

fn tighter_memory(host: Option<u64>, cgroup: Option<u64>) -> Option<u64> {
    match (host, cgroup) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Host availability clamped to known cgroup-v2 limits, including ancestors.
/// Each limit is reduced by that cgroup's current usage, not just ours.
pub fn host_available_memory() -> Option<u64> {
    let host = crate::uma_memory::mem_available_bytes();
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok();
    let relative = groups
        .as_deref()
        .and_then(|text| text.lines().find_map(|line| line.strip_prefix("0::")));
    let mut remaining = None;
    if let Some(relative) = relative {
        let root = std::path::Path::new("/sys/fs/cgroup");
        // Reject namespace-relative parent components rather than escaping root.
        let relative = std::path::Path::new(relative.trim_start_matches('/'));
        if !relative
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            let mut path = root.join(relative);
            loop {
                let limit = std::fs::read_to_string(path.join("memory.max")).ok();
                let used = std::fs::read_to_string(path.join("memory.current")).ok();
                if let (Some(limit), Some(used)) = (limit, used) {
                    remaining = tighter_memory(remaining, memory_remaining(&limit, &used));
                }
                if path == root || !path.pop() {
                    break;
                }
            }
        }
    }
    if let Some(groups) = groups.as_deref() {
        for line in groups.lines() {
            let mut parts = line.splitn(3, ':');
            let _ = parts.next();
            let controllers = parts.next().unwrap_or("");
            let relative = parts.next().unwrap_or("").trim_start_matches('/');
            if !controllers.split(',').any(|name| name == "memory") {
                continue;
            }
            let root = std::path::Path::new("/sys/fs/cgroup/memory");
            let relative = std::path::Path::new(relative);
            if relative
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                continue;
            }
            let mut path = root.join(relative);
            loop {
                let limit = std::fs::read_to_string(path.join("memory.limit_in_bytes")).ok();
                let used = std::fs::read_to_string(path.join("memory.usage_in_bytes")).ok();
                if let (Some(limit), Some(used)) = (limit, used) {
                    remaining = tighter_memory(remaining, memory_remaining(&limit, &used));
                }
                if path == root || !path.pop() {
                    break;
                }
            }
        }
    }
    tighter_memory(host, remaining)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinity_defaults() {
        for (cores, jobs) in [(1, 1), (2, 1), (8, 7), (64, 56)] {
            assert_eq!(
                compile_job_budget(cores, Some(u64::MAX), 0, None).unwrap(),
                jobs
            );
        }
    }

    #[test]
    fn memory_boundaries_and_reserves() {
        assert_eq!(compile_job_budget(64, None, u64::MAX, None).unwrap(), 1);
        assert!(compile_job_budget(64, Some(10), 11, None).is_err());
        let reserve = 1024 * 1024 * 1024;
        for (bytes, expected) in [
            (BYTES_PER_JOB - 1, None),
            (BYTES_PER_JOB, Some(1)),
            (2 * BYTES_PER_JOB - 1, Some(1)),
            (2 * BYTES_PER_JOB, Some(2)),
        ] {
            let result = compile_job_budget(64, Some(reserve + bytes), reserve, None);
            match expected {
                Some(jobs) => assert_eq!(result.unwrap(), jobs),
                None => assert!(result.is_err()),
            }
        }
    }

    #[test]
    fn jit_overrides_remain_bounded() {
        for value in ["0", "-1", "no", "", "184467440737095516160"] {
            assert!(parse_job_override("HIPFIRE_JIT_JOBS", Some(value)).is_err());
        }
        assert_eq!(parse_job_override("HIPFIRE_JIT_JOBS", None).unwrap(), None);
        for (value, expected) in [("1", 1), ("100", 8)] {
            let jobs = parse_job_override("HIPFIRE_JIT_JOBS", Some(value)).unwrap();
            assert_eq!(
                compile_job_budget(8, Some(u64::MAX), 0, jobs).unwrap(),
                expected
            );
        }
        assert!(compile_job_budget(8, None, 0, Some(0)).is_err());
        assert_eq!(
            compile_job_budget(64, Some(3 * BYTES_PER_JOB), 0, Some(20)).unwrap(),
            3
        );
    }

    #[test]
    fn cgroup_is_tighter_than_host() {
        let available = tighter_memory(Some(64 * BYTES_PER_JOB), Some(3 * BYTES_PER_JOB));
        assert_eq!(
            compile_job_budget(64, available, BYTES_PER_JOB, None).unwrap(),
            2
        );
        assert_eq!(
            tighter_memory(None, Some(BYTES_PER_JOB)),
            Some(BYTES_PER_JOB)
        );
        assert_eq!(
            tighter_memory(Some(BYTES_PER_JOB), None),
            Some(BYTES_PER_JOB)
        );
    }

    #[test]
    fn memory_limit_fixture_parsing() {
        assert_eq!(memory_remaining("max\n", "123"), None);
        assert_eq!(memory_remaining("4096\n", "1024\n"), Some(3072));
        assert_eq!(memory_remaining("1024", "4096"), Some(0));
        assert_eq!(memory_remaining("invalid", "0"), None);
        assert_eq!(memory_remaining("4096", "invalid"), None);
        assert_eq!(
            available_memory("MemFree: 1 kB\nMemAvailable: 3145728 kB\n", None),
            Some(2 * BYTES_PER_JOB)
        );
        assert_eq!(available_memory("MemFree: 100 kB\n", None), None);
        let cma = "MemAvailable: 3145728 kB\nCmaFree: 1048576 kB\n";
        assert_eq!(available_memory(cma, None), Some(2 << 30));
        // CMA pageblocks outside any registered area (KHO scratch): zoneinfo
        // `cma` pages (262144 x 4 KiB = 1 GiB) beat a smaller CmaFree.
        let zoneinfo = "Node 0, zone   Normal\n        cma      262144\n      nr_free_cma  1\n";
        assert_eq!(available_memory(cma, Some(zoneinfo)), Some(2 << 30));
        let cma = "MemAvailable: 3145728 kB\nCmaFree: 4096 kB\n";
        assert_eq!(available_memory(cma, Some(zoneinfo)), Some(2 << 30));
    }
}
