// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab-only: the MTP head's routed-expert counter. `HIPFIRE_LAB_MTP_EXPERT_TRACE=<dir>`
//! appends one JSON line per head Predict MoE call (serial and batched, gathered
//! and lane-local) to `<dir>/rank<r>.jsonl`:
//!
//! `{"mode":"local"|"gather","rows":R,"slots":S,"distinct_all":A,"owned_slots":O,"distinct_owned":D}`
//!
//! `slots = rows * top_k`; `owned_slots` are the slots whose expert `e` has
//! `e % ranks == rank` (the experts this rank's compact tensors hold);
//! `distinct_*` count distinct expert ids. This is for counting distinct
//! against slot host reads per rank. Each call downloads the call's top-k ids
//! (ordered on the launch stream), so a traced run is timing-distorted.

use super::*;
use std::io::Write;

/// The trace directory; unset or empty = off.
const EXPERT_TRACE_ENV: &str = "HIPFIRE_LAB_MTP_EXPERT_TRACE";

/// `(distinct_all, owned_slots, distinct_owned)` of one call's expert ids on
/// `rank` of `ranks`. Ids below zero are not experts and are skipped.
pub(super) fn expert_counts(ids: &[i32], rank: usize, ranks: usize) -> (usize, usize, usize) {
    let mut all = std::collections::BTreeSet::new();
    let mut owned = std::collections::BTreeSet::new();
    let mut owned_slots = 0usize;
    for &id in ids.iter().filter(|&&id| id >= 0) {
        let expert = id as usize;
        all.insert(expert);
        if ranks != 0 && expert % ranks == rank {
            owned_slots += 1;
            owned.insert(expert);
        }
    }
    (all.len(), owned_slots, owned.len())
}

/// The trace line of one call.
pub(super) fn trace_line(
    local: bool,
    rows: usize,
    slots: usize,
    counts: (usize, usize, usize),
) -> String {
    format!(
        "{{\"mode\":\"{}\",\"rows\":{rows},\"slots\":{slots},\"distinct_all\":{},\"owned_slots\":{},\"distinct_owned\":{}}}",
        if local { "local" } else { "gather" },
        counts.0,
        counts.1,
        counts.2
    )
}

/// Record one head MoE call: `topk` holds its `rows * top_k` expert ids
/// (`i32`), `placement` the head's EP rank (no placement: nothing to count).
pub(super) fn record(
    gpu: &Gpu,
    topk: &GpuTensor,
    rows: usize,
    top_k: usize,
    placement: Option<Qwen4EpPlacement>,
    local: bool,
) -> Result<(), MtpGpuError> {
    let Ok(dir) = hipfire_config::developer_var(EXPERT_TRACE_ENV) else {
        return Ok(());
    };
    let (true, Some(placement)) = (!dir.is_empty(), placement) else {
        return Ok(());
    };
    let slots = rows * top_k;
    let mut raw = vec![0u8; slots * std::mem::size_of::<i32>()];
    match gpu.active_stream.as_ref() {
        Some(stream) => {
            gpu.hip
                .memcpy_dtoh_async(&mut raw, &topk.buf, Some(stream))?;
            gpu.hip.stream_synchronize(stream)?;
        }
        None => gpu.hip.memcpy_dtoh(&mut raw, &topk.buf)?,
    }
    let ids: Vec<i32> = raw
        .chunks_exact(4)
        .map(|word| i32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let line = trace_line(
        local,
        rows,
        slots,
        expert_counts(&ids, placement.rank, placement.ranks),
    );
    let io = |error: std::io::Error| invalid(format!("expert trace {dir}: {error}"));
    std::fs::create_dir_all(&dir).map_err(io)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::path::Path::new(&dir).join(format!("rank{}.jsonl", placement.rank)))
        .map_err(io)?;
    writeln!(file, "{line}").map_err(io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_split_distinct_and_owned_slots() {
        // Two rows of top-4: experts 2, 3, 3, 8 and 2, 5, 8, 9 on rank 0 of 2.
        let ids = [2, 3, 3, 8, 2, 5, 8, 9];
        // distinct {2,3,5,8,9}; owned (even) slots 2, 8, 2, 8; distinct owned {2,8}.
        assert_eq!(expert_counts(&ids, 0, 2), (5, 4, 2));
        // rank 1: odd slots 3, 3, 5, 9; distinct {3,5,9}.
        assert_eq!(expert_counts(&ids, 1, 2), (5, 4, 3));
        // Single rank owns everything.
        assert_eq!(expert_counts(&ids, 0, 1), (5, 8, 5));
        assert_eq!(expert_counts(&[-1, 4], 0, 2), (1, 1, 1));
    }

    #[test]
    fn the_line_names_the_mode_and_counts() {
        assert_eq!(
            trace_line(true, 4, 40, (31, 20, 18)),
            "{\"mode\":\"local\",\"rows\":4,\"slots\":40,\"distinct_all\":31,\"owned_slots\":20,\"distinct_owned\":18}"
        );
        assert!(trace_line(false, 1, 10, (9, 5, 5)).starts_with("{\"mode\":\"gather\""));
    }
}
