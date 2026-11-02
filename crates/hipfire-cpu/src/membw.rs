// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host DRAM streaming-read bandwidth: the CPU side of every host-offload
//! decision.
//!
//! A spilled layer's CPU GEMV is bounded by how fast the worker pool can read
//! system RAM, so `hipfire calibrate` records that rate next to the link
//! figures. [`read_bandwidth`] is a port of the accepted `bw` cell of the
//! cpugemm study (a parallel fill of a `u64` buffer, then repeated parallel
//! wrapping-sum passes), not a new methodology: the numbers it reports are
//! comparable with the ones the offload notes quote.

use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Elements per rayon work item of the sum pass (1 MiB of `u64`).
const CHUNK_ELEMS: usize = 1 << 17;

/// One streaming-read measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadBandwidth {
    /// Median-pass rate in GB/s (10^9 bytes per second) over `bytes`.
    pub gbps: f64,
    /// Bytes the buffer actually held (`bytes` rounded down to whole `u64`s).
    pub bytes: u64,
    /// Timed passes (the untimed first pass is not counted).
    pub reps: u32,
    /// Workers of the rayon pool the measurement ran on.
    pub threads: u32,
    /// Wrapping sum of the buffer; keeps the reads observable and lets a
    /// caller check the pass really visited every element.
    pub checksum: u64,
}

/// Streaming-read bandwidth of a `bytes`-byte buffer on the *current* rayon
/// pool (the global pool unless called inside `ThreadPool::install`).
///
/// The buffer is filled in parallel with its own index, so every page is
/// resident and first-touched by the pool that reads it. `reps + 1` passes
/// follow; the first is untimed (page-table and prefetcher warm-up) and the
/// median of the rest sets the figure. `reps` is raised to 1 and the buffer to
/// one element so the figure is always defined.
pub fn read_bandwidth(bytes: usize, reps: usize) -> ReadBandwidth {
    let reps = reps.max(1);
    let mut data = vec![0u64; (bytes / 8).max(1)];
    data.par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as u64);
    let mut secs = Vec::with_capacity(reps);
    let mut checksum = 0;
    for rep in 0..reps + 1 {
        let t = Instant::now();
        checksum = black_box(
            data.par_chunks(CHUNK_ELEMS)
                .map(|x| x.iter().copied().fold(0u64, u64::wrapping_add))
                .reduce(|| 0, u64::wrapping_add),
        );
        if rep > 0 {
            secs.push(t.elapsed().as_secs_f64());
        }
    }
    secs.sort_by(f64::total_cmp);
    let median = secs[secs.len() / 2];
    let bytes = data.len() * 8;
    ReadBandwidth {
        gbps: bytes as f64 / median / 1e9,
        bytes: bytes as u64,
        reps: reps as u32,
        threads: rayon::current_num_threads() as u32,
        checksum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bandwidth_is_finite_and_sums_the_whole_buffer() {
        let r = read_bandwidth(8 << 20, 2);
        assert!(r.gbps.is_finite() && r.gbps > 0.0, "{r:?}");
        assert_eq!(r.bytes, 8 << 20);
        assert_eq!(r.reps, 2);
        assert!(r.threads >= 1);
        // Closed form of the wrapping sum of 0..n.
        let n = (r.bytes / 8) as u128;
        assert_eq!(r.checksum, (n * (n - 1) / 2) as u64);
    }
}
