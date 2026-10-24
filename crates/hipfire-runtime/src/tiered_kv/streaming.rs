// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only planning for a lab streaming owner. Planning never advances state.
//! Complete the chunk, archive and stage before adopting the next frontier.
//! On failure the one-shot owner must stop, not replay DeltaNet.

use super::{invalid, product};
use hip_bridge::HipResult;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingStep {
    pub absolute_start: usize,
    pub absolute_end: usize,
    pub packed_start: usize,
    pub archive_source_rows: Vec<usize>,
    pub carry_source_row: usize,
    pub carry_rows: usize,
    pub next_selected_starts: Vec<usize>,
}

/// Deterministic sink + recent policy used only by the diagnostic owner.
pub fn sink_recent(
    block: usize,
    window_blocks: usize,
    completed_rows: usize,
) -> HipResult<Vec<usize>> {
    if block == 0 || window_blocks < 2 || !completed_rows.is_multiple_of(block) {
        return Err(invalid("tiered prefill: invalid selection geometry"));
    }
    let complete = completed_rows / block;
    if complete <= window_blocks {
        return Ok((0..complete).map(|i| i * block).collect());
    }
    Ok(std::iter::once(0)
        .chain((complete - (window_blocks - 1)..complete).map(|i| i * block))
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn plan_step(
    block: usize,
    window_blocks: usize,
    chunk_limit: usize,
    physical_cap: usize,
    logical_cap: usize,
    absolute_start: usize,
    tokens: usize,
) -> HipResult<StreamingStep> {
    if chunk_limit == 0 || tokens == 0 || tokens > chunk_limit || block == 0 {
        return Err(invalid("tiered prefill: empty or oversized chunk"));
    }
    let absolute_end = absolute_start
        .checked_add(tokens)
        .ok_or_else(|| invalid("tiered prefill: logical extent overflow"))?;
    if absolute_end > logical_cap || absolute_end > i32::MAX as usize {
        return Err(invalid("tiered prefill: logical capacity exceeded"));
    }
    let complete_before = absolute_start / block * block;
    let before = sink_recent(block, window_blocks, complete_before)?;
    let selected_rows = product(before.len(), block)?;
    let packed_start = selected_rows
        .checked_add(absolute_start % block)
        .ok_or_else(|| invalid("tiered prefill: packed extent overflow"))?;
    if packed_start
        .checked_add(tokens)
        .is_none_or(|end| end > physical_cap)
    {
        return Err(invalid("tiered prefill: fixed device capacity exceeded"));
    }
    let complete_after = absolute_end / block * block;
    let new_blocks = (complete_after - complete_before) / block;
    let carry_source_row = selected_rows + new_blocks * block;
    Ok(StreamingStep {
        absolute_start,
        absolute_end,
        packed_start,
        archive_source_rows: (0..new_blocks).map(|i| selected_rows + i * block).collect(),
        carry_source_row,
        carry_rows: absolute_end % block,
        next_selected_starts: sink_recent(block, window_blocks, complete_after)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_are_forwarded_once_and_published_only_when_complete() {
        let mut logical = 0;
        let mut archived = 0;
        for n in [127, 1, 129, 255, 7, 511, 128, 3] {
            let p = plan_step(128, 2, 512, 895, 4096, logical, n).unwrap();
            assert_eq!(p.absolute_start, logical);
            assert_eq!(
                p.packed_start,
                (archived / 128).min(2) * 128 + logical % 128
            );
            archived += p.archive_source_rows.len() * 128;
            logical += n;
            assert_eq!(archived, logical / 128 * 128);
            assert_eq!(p.carry_rows, logical % 128);
            assert_eq!(p.carry_source_row + p.carry_rows, p.packed_start + n);
            assert!(p
                .next_selected_starts
                .windows(2)
                .all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn every_remainder_and_chunk_has_no_hole_or_duplicate_block() {
        for remainder in 0..128 {
            for n in 1..=512 {
                let start = 2048 + remainder;
                let p = plan_step(128, 4, 512, 1151, 4096, start, n).unwrap();
                assert_eq!(p.packed_start, 512 + remainder);
                let count = (remainder + n) / 128;
                assert_eq!(
                    p.archive_source_rows,
                    (0..count).map(|i| 512 + i * 128).collect::<Vec<_>>()
                );
                assert_eq!(p.carry_source_row, 512 + count * 128);
                assert_eq!(p.carry_rows, (remainder + n) % 128);
                assert_eq!(p.next_selected_starts[0], 0);
                assert_eq!(
                    *p.next_selected_starts.last().unwrap(),
                    (p.absolute_end / 128 - 1) * 128
                );
            }
        }
    }

    #[test]
    fn selection_is_identity_before_window_fills() {
        assert_eq!(sink_recent(128, 2, 0).unwrap(), Vec::<usize>::new());
        assert_eq!(sink_recent(128, 2, 128).unwrap(), vec![0]);
        assert_eq!(sink_recent(128, 2, 256).unwrap(), vec![0, 128]);
        assert_eq!(sink_recent(128, 2, 384).unwrap(), vec![0, 256]);
    }

    #[test]
    fn invalid_or_overflowing_plan_refuses_before_state_change() {
        for (block, window, chunk, cap, logical, start, n) in [
            (0, 2, 512, 895, 4096, 0, 1),
            (128, 1, 512, 895, 4096, 0, 1),
            (128, 2, 512, 895, 4096, 0, 0),
            (128, 2, 512, 895, 4096, 0, 513),
            (128, 2, 512, 767, 4096, 639, 512),
            (128, 2, 512, 895, 4096, 4096, 1),
            (128, 2, 512, 895, usize::MAX, usize::MAX, 1),
        ] {
            assert!(plan_step(block, window, chunk, cap, logical, start, n).is_err());
        }
        assert!(sink_recent(128, 2, 1).is_err());
    }

    #[test]
    fn independent_absolute_row_oracle_survives_archive_stage_and_carry() {
        // The expected values are absolute row identities, not re-computed
        // planner offsets. Emulate the actual owner order with a tiny block
        // to cover repeated window saturation and incomplete-block relocation.
        for block in [2, 7, 128] {
            for chunk in [1, 3, 127, 256, 511] {
                let physical = 2 * block + chunk + block - 1;
                let mut device = vec![usize::MAX; physical];
                let mut archive = Vec::new();
                let mut start = 0;
                let end = 7 * block + 2 * chunk + 1;
                let mut dn_rows = Vec::new();
                while start < end {
                    let n = chunk.min(end - start);
                    let p = plan_step(block, 2, chunk, physical, end, start, n).unwrap();
                    for i in 0..n {
                        device[p.packed_start + i] = start + i;
                        dn_rows.push(start + i);
                    }
                    let carry =
                        device[p.carry_source_row..p.carry_source_row + p.carry_rows].to_vec();
                    for &source in &p.archive_source_rows {
                        archive.extend_from_slice(&device[source..source + block]);
                    }
                    assert_eq!(archive, (0..archive.len()).collect::<Vec<_>>());
                    let mut packed = Vec::new();
                    for &selected in &p.next_selected_starts {
                        packed.extend_from_slice(&archive[selected..selected + block]);
                    }
                    device[..packed.len()].copy_from_slice(&packed);
                    device[packed.len()..packed.len() + carry.len()].copy_from_slice(&carry);
                    let complete = p.absolute_end / block * block;
                    assert_eq!(carry, (complete..p.absolute_end).collect::<Vec<_>>());
                    assert_eq!(&device[..packed.len()], packed.as_slice());
                    assert_eq!(
                        &device[packed.len()..packed.len() + carry.len()],
                        carry.as_slice()
                    );
                    start += n;
                }
                assert_eq!(dn_rows, (0..end).collect::<Vec<_>>());
            }
        }
    }
}
