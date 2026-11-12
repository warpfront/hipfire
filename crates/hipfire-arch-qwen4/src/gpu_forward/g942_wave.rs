// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 single-device lane prefill wave (P2).
//!
//! On one MI300X (exact `gfx942`, `topology = single`, stock-FP8 Flash-Next) a serve step's
//! lane prefill tiles used to run one singleton chunk forward each, one lane after another:
//! every tile re-read the layer weights. With the wave, every run of two or more consecutive
//! tiles of a step, native-MTP lanes' included, is ONE [`Qwen4Bundle::prefill_wave`] forward: the row-local stages
//! of the lane policy (embed, PLE rows, HC read / write, the GDN and QSA projections, the FP8
//! grouped MoE, the final hyper) run once over all the wave's rows, and the stateful cores
//! (GDN, QSA, PLE conv) per lane on the lane's own rows and state. An MTP lane's wave rows are
//! captured into the resident spec hidden buffer (`prefill_wave_wide`) and its head fill then
//! runs from its own rows of that capture (`fill_head_after_wave`), the serial tile's head append.
//!
//! Why each lane's bytes equal its serial tile (checked by the lane oracle, not assumed):
//! the singleton chunk forward on gfx942 pins `Gpu::prefill_invariant_scope`
//! (`trunk_prefill_invariant_scope`), so every row-count routing decision (`route_rows`) is the
//! canonical chunk's whatever the tile's rows; the wave enters the same scope
//! ([`wave_applies`]), the BF16 trunk is the row-invariant MFMA chain (each output element is
//! one ascending-K walk of its own activation row, `gemm_bf16_mfma_gfx942_rows16[_chain]`), and
//! the FP8 grouped expert route runs the same kernels at every row count.
//!
//! The wave is not switchable: it is the exact gfx942 single-device route (resolved by the
//! arch capability and the lane store's receipt), and a lone first arrival of a burst waits
//! [`BURST_GAP`] for a peer.

use crate::bundle::Qwen4Bundle;
use crate::lane_exec::PrefillPart;
use rdna_compute::Gpu;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// How long, at most, a batch-eligible request that finds the daemon idle waits for a second
/// batched request before it takes the AR singleton route (the burst's first request would
/// otherwise prefill alone and be promoted afterwards). The wait ends the moment a peer is
/// queued. A fixed constant: G12's single admission-coalescing rule replaces it.
pub const BURST_GAP: Duration = Duration::from_millis(4);

/// The oracle run that evidences the wave (`fn_lane_oracle --wave-prefill`, k = 8 / 32 / 64,
/// contexts 57 / 512 / 2048): cited by the wave's log line.
pub const WAVE_RECEIPT: &str =
    "g0 mi300x/beat/P/p2 2026-11-10 gfx942 stock-fp8 single, lane prefill wave vs singleton";

/// The lane prefill wave applies to this forward: the device is exactly `gfx942` and the bundle
/// is the single-device root (no PP mesh, no EP).
pub(crate) fn wave_applies(gpu: &Gpu, bundle: &Qwen4Bundle) -> bool {
    gpu.arch_caps.is_gfx942() && bundle.lane_topology() == "single"
}

/// Serve admission (no device at hand): the staged store is the gfx942 single-device stock-FP8
/// lane store (its stage policy resolved under that proof).
pub fn serve_armed(bundle: &Qwen4Bundle) -> bool {
    bundle.lane_topology() == "single"
        && bundle.lanes().is_some_and(|store| {
            store
                .evidence()
                .contains(crate::gpu_forward_lanes::GFX942_STOCK_FP8_RECEIPT)
        })
}

/// Most rows one wave carries: the forward scratch's chunk rows (the prefill chunk bound). A
/// wave holding an MTP lane (`has_mtp`) is also capped by the rows of the resident spec hidden
/// buffer, the wide-hidden capture its head fills read.
pub(crate) fn wave_cap(bundle: &Qwen4Bundle, has_mtp: bool) -> usize {
    let chunk = bundle.spec_chunk_rows().unwrap_or(0);
    if has_mtp {
        chunk.min(bundle.spec_hidden_rows())
    } else {
        chunk
    }
}

/// Partition a step's prefill tiles (`tile_rows[i]` rows of the i-th Prefill request, plan
/// order; `eligible[i]`: the lane may be forwarded by a wave, an MTP lane only with a spec hidden
/// buffer to capture into) into serial tiles and joint waves: runs of consecutive eligible tiles
/// pack greedily into waves of at most `max_rows` rows; a single-tile run, a tile wider than
/// `max_rows` and every ineligible lane's tile stay serial. Parts come back in plan order and
/// cover every ordinal exactly once.
pub(crate) fn plan_parts(
    tile_rows: &[usize],
    eligible: &[bool],
    max_rows: usize,
) -> Vec<PrefillPart> {
    debug_assert_eq!(tile_rows.len(), eligible.len());
    let n = tile_rows.len();
    let mut parts = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let joins = |k: usize| eligible[k] && tile_rows[k] > 0 && tile_rows[k] <= max_rows;
        if !joins(i) {
            parts.push(PrefillPart::Serial(i));
            i += 1;
            continue;
        }
        let start = i;
        let mut rows = 0usize;
        while i < n && joins(i) && rows + tile_rows[i] <= max_rows {
            rows += tile_rows[i];
            i += 1;
        }
        if i - start >= 2 {
            parts.push(PrefillPart::Wave(start..i));
        } else {
            parts.push(PrefillPart::Serial(start));
        }
    }
    parts
}

static WAVES: AtomicU64 = AtomicU64::new(0);
static WAVE_LANES: AtomicU64 = AtomicU64::new(0);
static WAVE_ROWS: AtomicU64 = AtomicU64::new(0);

/// Waves run by this process: `(waves, lanes, rows)`. The lane oracle requires it to be
/// non-zero in its wave runs (a run that silently took the serial tiles is not evidence).
pub fn wave_counters() -> (u64, u64, u64) {
    (
        WAVES.load(Ordering::Relaxed),
        WAVE_LANES.load(Ordering::Relaxed),
        WAVE_ROWS.load(Ordering::Relaxed),
    )
}

/// Record one wave of `lanes` lanes and `rows` total rows (launched, not yet synchronized) and
/// log it: the serve log line that proves the wave ran.
pub(crate) fn note_wave(device: i32, lanes: usize, rows: usize) {
    let nth = WAVES.fetch_add(1, Ordering::Relaxed) + 1;
    WAVE_LANES.fetch_add(lanes as u64, Ordering::Relaxed);
    WAVE_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    if nth == 1 {
        eprintln!(
            "[g942-wave] lane prefill waves on (burst wait {} ms): proofs {WAVE_RECEIPT}",
            BURST_GAP.as_millis()
        );
    }
    eprintln!("[g942-wave] dev={device} wave={nth} lanes={lanes} rows={rows}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serial(range: std::ops::Range<usize>) -> Vec<PrefillPart> {
        range.map(PrefillPart::Serial).collect()
    }

    #[test]
    fn one_tile_is_its_own_serial_part() {
        assert_eq!(plan_parts(&[57], &[true], 1536), serial(0..1));
        assert!(plan_parts(&[], &[], 1536).is_empty());
    }

    #[test]
    fn a_burst_of_short_tiles_is_one_wave() {
        let rows = [57, 44, 51, 59, 48, 53, 52, 44];
        assert_eq!(
            plan_parts(&rows, &[true; 8], 1536),
            vec![PrefillPart::Wave(0..8)]
        );
    }

    #[test]
    fn a_burst_of_eight_mtp_prompts_is_one_wave() {
        // Eight ~150-row prompts (an MTP burst) are 1200 rows: one wave under the chunk cap.
        let rows = [150, 148, 152, 149, 151, 147, 153, 150];
        assert_eq!(
            plan_parts(&rows, &[true; 8], 1536),
            vec![PrefillPart::Wave(0..8)]
        );
    }

    #[test]
    fn waves_pack_to_the_row_cap_and_cover_every_ordinal_once() {
        // 26 x 57 = 1482 rows fit a 1536-row wave, the 27th starts the next.
        let rows = vec![57usize; 64];
        let parts = plan_parts(&rows, &vec![true; 64], 1536);
        assert_eq!(
            parts,
            vec![
                PrefillPart::Wave(0..26),
                PrefillPart::Wave(26..52),
                PrefillPart::Wave(52..64)
            ]
        );
    }

    #[test]
    fn a_tile_wider_than_the_cap_and_an_ineligible_lane_stay_serial() {
        let parts = plan_parts(
            &[57, 2000, 512, 512, 57],
            &[true, true, true, true, false],
            1536,
        );
        assert_eq!(
            parts,
            vec![
                PrefillPart::Serial(0),
                PrefillPart::Serial(1),
                PrefillPart::Wave(2..4),
                PrefillPart::Serial(4)
            ]
        );
    }

    #[test]
    fn a_trailing_single_tile_after_a_full_wave_is_serial() {
        let parts = plan_parts(&[1000, 500, 1200], &[true; 3], 1536);
        assert_eq!(parts, vec![PrefillPart::Wave(0..2), PrefillPart::Serial(2)]);
    }

    #[test]
    fn mixed_lengths_in_one_wave_keep_plan_order() {
        let parts = plan_parts(&[57, 512, 1024, 57], &[true; 4], 2048);
        assert_eq!(parts, vec![PrefillPart::Wave(0..4)]);
    }
}
