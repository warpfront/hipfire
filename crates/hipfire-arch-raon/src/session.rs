// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini on the engine's prefill snapshot cache
//! (`hipfire_runtime::session_cache`): [`SessionState`] for [`RaonBundle`].
//!
//! The request state is the persistent slab (`RaonState::persist_part`: RDA
//! recurrent matrices, q/k/v conv histories and the device `seq_len`, one
//! fixed size, overwritten in place by every forward) and one append-only
//! latent row stream per MLA layer (`RaonState::mla_cache`). mHC carries no
//! state between tokens. The expanded K/V mirror is derived from the latent
//! rows, so it is never stored: a restore only lowers its valid-row watermark
//! (`RaonState::clamp_mirror_valid_rows`) and the forward's `ensure_context`
//! re-expands the rows below the resume point in the same 512-row tiles the
//! prefill mixers expand, so the rebuilt mirror equals the cold one.
//!
//! Every reuse point is a position on the cold prefill chunk grid (a multiple
//! of the prefill width `W`, `RaonBundle::prefill_step_rows`), so a restored
//! prefill runs exactly the chunks a cold prefill runs and its output is
//! byte-identical to a cold one. A conversation's next turn resumes at the
//! previous prompt rounded down to `W` (the resident snapshot); other requests
//! reuse stride snapshots at multiples of `session_stride`. Decoded rows are
//! never reused: decode and batched prefill write different bits, so the
//! driver runs without live continuation (`begin(.., false)` and `commit`,
//! never `commit_live`) and the reply is recomputed with the next prompt.

use hipfire_runtime::session_cache::{
    RowStream, SessionRoute, SessionState, SnapshotParts, StateLayout, StatePart,
};
use rdna_compute::Gpu;

use crate::bundle::RaonBundle;

/// Snapshot boundary stride in prompt tokens, before rounding up to a
/// multiple of the prefill width.
pub const SESSION_STRIDE_TOKENS: usize = 2048;

/// Magic and version of the snapshot metadata.
const META_MAGIC: &[u8; 8] = b"RAONSES1";
/// Metadata bytes: magic, position, persist bytes, latent row bytes, MLA layers.
const META_BYTES: usize = 40;

/// Smallest multiple of the prefill `width` that is at least
/// [`SESSION_STRIDE_TOKENS`] (so stride boundaries stay on the chunk grid).
pub(crate) fn session_stride(width: usize) -> usize {
    let width = width.max(1);
    SESSION_STRIDE_TOKENS.div_ceil(width) * width
}

/// Snapshot boundaries of a `prompt_len`-token prompt in `(after, up_to]`:
/// every multiple of `stride`, plus the prompt length rounded down to the
/// prefill `width`, the last position a cold prefill of a longer prompt reaches
/// with whole chunks (a follow-up turn resumes there). Ascending and
/// deduplicated; empty for `width == 0`.
fn grid_boundaries(
    width: usize,
    stride: usize,
    prompt_len: usize,
    after: usize,
    up_to: usize,
) -> Vec<usize> {
    if width == 0 || stride == 0 {
        return Vec::new();
    }
    let up_to = up_to.min(prompt_len);
    let mut boundaries: Vec<usize> = (after / stride + 1..=up_to / stride)
        .map(|k| k * stride)
        .collect();
    let grid_end = prompt_len / width * width;
    if grid_end > after && grid_end <= up_to {
        boundaries.push(grid_end);
        // A width that does not divide the stride can put the grid end below
        // the last stride multiple.
        boundaries.sort_unstable();
        boundaries.dedup();
    }
    boundaries
}

/// The decoded fields of a snapshot's metadata.
struct Meta {
    position: usize,
    persist_bytes: usize,
    latent_row_bytes: usize,
    n_mla: usize,
}

fn encode_meta(
    position: usize,
    persist_bytes: usize,
    latent_row_bytes: usize,
    n_mla: usize,
) -> Vec<u8> {
    let mut meta = Vec::with_capacity(META_BYTES);
    meta.extend_from_slice(META_MAGIC);
    for field in [position, persist_bytes, latent_row_bytes, n_mla] {
        meta.extend_from_slice(&(field as u64).to_le_bytes());
    }
    meta
}

fn parse_meta(meta: &[u8]) -> Result<Meta, String> {
    if meta.len() != META_BYTES || &meta[..8] != META_MAGIC {
        return Err("raon1_mini session snapshot: unrecognised metadata".to_string());
    }
    let field = |i: usize| {
        let at = 8 + 8 * i;
        let bytes: [u8; 8] = meta[at..at + 8]
            .try_into()
            .expect("metadata length checked");
        u64::from_le_bytes(bytes) as usize
    };
    Ok(Meta {
        position: field(0),
        persist_bytes: field(1),
        latent_row_bytes: field(2),
        n_mla: field(3),
    })
}

impl RaonBundle {
    /// Prefill width the snapshot grid is built on.
    fn session_width(&self) -> usize {
        self.prefill_step_rows()
    }

    /// Whether `meta` was captured from a state shaped like this one.
    fn meta_matches(&self, meta: &Meta) -> bool {
        let (_, persist_bytes) = self.state.persist_part();
        let dims = self.state.dims();
        meta.persist_bytes == persist_bytes
            && meta.latent_row_bytes == dims.latent_row_bytes()
            && meta.n_mla == self.state.mla_cache.len()
            && meta.position <= self.state.max_seq()
    }

    /// The live state as session-cache ranges with `rows` valid latent rows.
    fn session_layout(&self, rows: usize) -> StateLayout<'_> {
        let (persist, total) = self.state.persist_part();
        let row_bytes = self.state.dims().latent_row_bytes();
        StateLayout {
            fixed: vec![StatePart {
                buf: persist,
                offset: 0,
                bytes: total,
            }],
            rows: self
                .state
                .mla_cache
                .iter()
                .map(|t| RowStream {
                    buf: &t.buf,
                    row_bytes,
                    rows,
                })
                .collect(),
        }
    }

    /// Scope string (see [`SessionState::snapshot_scope`]) of an Ar snapshot.
    fn session_scope(&self, route: SessionRoute) -> String {
        let width = self.session_width();
        let dims = self.state.dims();
        format!(
            "raon1_mini/{route:?}/{}/{:?}/rows{width}/stride{}/mirror-{}/budget{}/mla-{:?}/{:?}/v1",
            self.gpu_arch,
            self.prefill_mode(),
            session_stride(width),
            if dims.mirror_q8 { "q8" } else { "bf16" },
            self.state.mirror_budget_rows(),
            self.work.mla.policy(),
            self.precision_plan().effective,
        )
    }
}

impl SessionState for RaonBundle {
    /// Ar only: the MTP route owns a draft head this state does not capture.
    /// The scope carries everything that changes the bits a prefill writes:
    /// device, prefill route and width (batched and serial prefill write
    /// different bits), the mirror format and budget (which route attends),
    /// the MLA route policy and the resolved precision tiers.
    fn snapshot_scope(&self, route: SessionRoute) -> Option<String> {
        match route {
            SessionRoute::Mtp => None,
            SessionRoute::Ar => Some(self.session_scope(route)),
        }
    }

    fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
        let width = self.session_width();
        grid_boundaries(width, session_stride(width), prompt.len(), after, up_to)
    }

    fn snapshot_parts(
        &mut self,
        _route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String> {
        if self.state.n_tokens() != position {
            return Err(format!(
                "raon1_mini session capture: state at {}, not {position}",
                self.state.n_tokens()
            ));
        }
        let (_, persist_bytes) = self.state.persist_part();
        let meta = encode_meta(
            position,
            persist_bytes,
            self.state.dims().latent_row_bytes(),
            self.state.mla_cache.len(),
        );
        Ok(SnapshotParts {
            meta,
            layout: self.session_layout(position),
        })
    }

    fn snapshot_restorable(&self, _route: SessionRoute, meta: &[u8]) -> bool {
        parse_meta(meta).is_ok_and(|meta| self.meta_matches(&meta))
    }

    fn restore_parts(
        &mut self,
        gpu: &mut Gpu,
        _route: SessionRoute,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, String> {
        let meta = parse_meta(meta)?;
        if !self.meta_matches(&meta) {
            return Err(
                "raon1_mini session restore: snapshot was captured from a differently shaped state"
                    .to_string(),
            );
        }
        // The copies write latent rows [0, position): map them first.
        self.state.ensure_capacity(gpu, meta.position)?;
        Ok(self.session_layout(meta.position))
    }

    /// The mirror is derived from the latent rows: rows from `first_row` up
    /// now hold the snapshot's latent data, so the expansion of them is stale.
    fn rows_overwritten(&mut self, _route: SessionRoute, first_row: usize) {
        self.state.clamp_mirror_valid_rows(first_row);
    }

    fn finish_restore(
        &mut self,
        _gpu: &mut Gpu,
        _route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String> {
        let meta = parse_meta(meta)?;
        // The device `seq_len` came with the restored slab.
        self.state.set_host_tokens(meta.position)?;
        self.state.clamp_mirror_valid_rows(meta.position);
        Ok(())
    }

    fn live_position(&self, _route: SessionRoute) -> Option<usize> {
        Some(self.state.n_tokens())
    }

    fn growth_reserve_bytes(&self, gpu: &Gpu, through: usize) -> u64 {
        self.context_growth_bytes(gpu, through)
    }

    /// Reset the state only: the driver is mid-`begin` (or taken out of the
    /// bundle), so it must not be forgotten from here.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::{grid_boundaries, session_stride, SESSION_STRIDE_TOKENS};

    #[test]
    fn stride_is_the_smallest_width_multiple_covering_the_base_stride() {
        assert_eq!(SESSION_STRIDE_TOKENS, 2048);
        assert_eq!(session_stride(1), 2048);
        assert_eq!(session_stride(512), 2048);
        assert_eq!(session_stride(1024), 2048);
        assert_eq!(session_stride(2048), 2048);
        assert_eq!(session_stride(4096), 4096);
        assert_eq!(session_stride(700), 2100);
    }

    #[test]
    fn boundaries_are_stride_multiples_plus_the_grid_end() {
        assert_eq!(
            grid_boundaries(512, 2048, 5000, 0, 4999),
            vec![2048, 4096, 4608]
        );
        assert_eq!(grid_boundaries(512, 2048, 5000, 4096, 4999), vec![4608]);
        // Nothing at or below `after`.
        assert_eq!(
            grid_boundaries(512, 2048, 5000, 4608, 4999),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn grid_end_beyond_up_to_is_excluded() {
        assert_eq!(grid_boundaries(512, 2048, 5000, 0, 4500), vec![2048, 4096]);
        assert_eq!(
            grid_boundaries(512, 2048, 5000, 0, 2047),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn short_prompt_has_only_stride_multiples() {
        // Shorter than the width: the grid end is 0 and no stride multiple fits.
        assert_eq!(grid_boundaries(512, 2048, 300, 0, 300), Vec::<usize>::new());
        // Shorter than the stride: only the grid end.
        assert_eq!(grid_boundaries(512, 2048, 1300, 0, 1300), vec![1024]);
    }

    #[test]
    fn serial_width_resumes_at_the_prompt_end() {
        assert_eq!(grid_boundaries(1, 2048, 5000, 4096, 5000), vec![5000]);
        assert_eq!(
            grid_boundaries(1, 2048, 5000, 0, 5000),
            vec![2048, 4096, 5000]
        );
    }

    #[test]
    fn grid_end_on_a_stride_multiple_is_not_duplicated() {
        assert_eq!(grid_boundaries(512, 2048, 4096, 0, 4096), vec![2048, 4096]);
        assert_eq!(grid_boundaries(512, 2048, 4200, 0, 4200), vec![2048, 4096]);
    }

    #[test]
    fn zero_width_has_no_boundaries() {
        assert_eq!(grid_boundaries(0, 2048, 5000, 0, 5000), Vec::<usize>::new());
    }
}
