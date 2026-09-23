// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
//! Print the hottest kernels from a rocprofv3 `_kernel_stats.csv`.

use rdna_compute::profile_rocprof::parse_rocprof_stats_csv;
use std::path::Path;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: rocprof_top <kernel_stats.csv> [limit]")?;
    let limit = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid limit: {error}"))?
        .unwrap_or(30);
    let mut kernels = parse_rocprof_stats_csv(Path::new(&path))?;
    kernels.sort_by(|left, right| {
        right
            .duration_us
            .partial_cmp(&left.duration_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "{:>10} {:>12} {:>9}  kernel",
        "calls", "total_ms", "percent"
    );
    for kernel in kernels.into_iter().take(limit) {
        println!(
            "{:>10} {:>12.3} {:>8.2}%  {}",
            kernel.calls,
            kernel.duration_us / 1_000.0,
            kernel.percent,
            kernel.name
        );
    }
    Ok(())
}
