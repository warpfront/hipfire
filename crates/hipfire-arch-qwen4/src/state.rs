// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 GPU request state and CPU parity state.
//!
//! `Qwen4State` owns the GPU buffers and releases them through `free_gpu`.
//! `Reference*` structs are used only by CPU equation tests and parity probes.
//!
//! Q8Ef owns one separate F16 residual per live GDN layer. The recurrent
//! tensors, verify ring, transaction mirror and session/prefix/disk payloads
//! remain plain Q8. Restores discard EF history through a recorded zero
//! dispatch; uninterrupted forwards and all-accept commits retain it. The
//! live plane is charged once per owner, never once per snapshot or ring row.

use crate::config::{LayerType, Qwen4Config};
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::ple::PleHistory;
use crate::weights::Qwen4BandScope;
use hipfire_dispatch::pipeline::GdnRowCapture;
use hipfire_runtime::kv_backend::{
    KvChunkPlan, KvChunkPlanError, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
};
use hipfire_runtime::session_cache::{RowStream, StateLayout, StatePart};
use rdna_compute::qsa_pooled_mirror::{
    qsa_pooled_bf16_mirror_enabled, QsaPooledBf16MirrorRef, QsaPooledBf16ValidBlocks,
};
use rdna_compute::tensor_ops::{
    copy_regions, gated_delta_rollback_layers, gdn_q8ef_residual_zero, qsa_f16_arena_rows,
    CopyRegion, GatedDeltaRollbackLayers, GdnStateFormat, QsaF16ArenaRef, QsaF16ValidLen,
    QsaKvFormat, QsaStateLayout,
};
use rdna_compute::{DType, Gpu, GpuTensor, MemsetBatch};
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

/// GDN key and value head width of the few-row capture route.
const GDN_HEAD_DIM: usize = 128;

static NEXT_QWEN4_MODEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_qwen4_model_id() -> u64 {
    NEXT_QWEN4_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1)
}

/// The first `bytes` of `src` into the start of `dst`.
fn whole<'a>(src: &'a GpuTensor, dst: &'a GpuTensor, bytes: usize) -> CopyRegion<'a> {
    CopyRegion {
        dst: &dst.buf,
        dst_offset: 0,
        src: &src.buf,
        src_offset: 0,
        bytes,
    }
}

/// Device copies between the live state and `arena`: live into the arena
/// when `capture`, else back out. The GDN recurrent state is skipped when
/// the arena records an armed verify's live ring slot (that slot survives the
/// verify). The QSA raw-index tail spans the last `partial_capacity` rows
/// before the live `raw_len` (capture) or the arena mark's (restore).
fn arena_regions<'a>(
    state: &'a Qwen4State,
    arena: &'a Qwen4StateSnapshotArena,
    capture: bool,
) -> Result<Vec<CopyRegion<'a>>, StateError> {
    let pair = |live: &'a GpuTensor, saved: &'a GpuTensor, bytes: usize| {
        if capture {
            whole(live, saved, bytes)
        } else {
            whole(saved, live, bytes)
        }
    };
    let mut copies = Vec::with_capacity(2 * state.gdn.len() + 4 * state.qsa.len() + 2);
    if arena.gdn_live.is_none() {
        for (layer, saved) in state.gdn.iter().zip(&arena.recurrent) {
            copies.push(pair(&layer.recurrent, saved, layer.recurrent.byte_size()));
        }
    }
    for (layer, saved) in state.gdn.iter().zip(&arena.conv) {
        copies.push(pair(&layer.conv, saved, layer.conv.byte_size()));
    }
    for (index, layer) in state.qsa.iter().enumerate() {
        copies.push(pair(
            &layer.partial_keys,
            &arena.qsa_partial_keys[index],
            layer.partial_keys.byte_size(),
        ));
        copies.push(pair(
            &layer.partial_values,
            &arena.qsa_partial_values[index],
            layer.partial_values.byte_size(),
        ));
        copies.push(pair(
            &layer.selected_indices,
            &arena.qsa_selected[index],
            layer.selected_indices.byte_size(),
        ));
        let raw_width = layer
            .raw_index_keys
            .numel()
            .checked_div(layer.raw_capacity)
            .ok_or(StateError::SnapshotShape)?;
        let raw_len = if capture {
            layer.raw_len
        } else {
            arena.qsa_marks[index].raw_len
        };
        let rows = layer.partial_capacity.min(raw_len);
        if rows > 0 {
            let row_bytes = raw_width
                .checked_mul(layer.raw_index_keys.dtype.size())
                .ok_or(StateError::DimensionOverflow)?;
            let bytes = rows
                .checked_mul(row_bytes)
                .ok_or(StateError::DimensionOverflow)?;
            let live_offset = (raw_len - rows)
                .checked_mul(row_bytes)
                .ok_or(StateError::DimensionOverflow)?;
            let (live, saved) = (
                &layer.raw_index_keys.buf,
                &arena.qsa_raw_circular[index].buf,
            );
            copies.push(if capture {
                CopyRegion {
                    dst: saved,
                    dst_offset: 0,
                    src: live,
                    src_offset: live_offset,
                    bytes,
                }
            } else {
                CopyRegion {
                    dst: live,
                    dst_offset: live_offset,
                    src: saved,
                    src_offset: 0,
                    bytes,
                }
            });
        }
    }
    // A profile without PLE has no PLE conv state, live or saved; the two
    // sides are validated to agree (`validate_layout`) before any copy.
    if let (Some(live), Some(saved)) = (&state.ple_conv, &arena.ple_conv) {
        copies.push(pair(live, saved, live.byte_size()));
    }
    if let (Some(live), Some(saved)) = (&state.hyper_feedback, &arena.hyper_feedback) {
        copies.push(pair(live, saved, live.byte_size()));
    }
    Ok(copies)
}

/// Reference-only GDN state used by CPU equation tests.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceGdnLayerState {
    pub recurrent: Vec<f32>,
    pub conv_history: Vec<f32>,
    pub conv_cursor: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

impl ReferenceGdnLayerState {
    pub fn new(config: &Qwen4Config) -> Result<Self, StateError> {
        let key_heads = config.linear_num_key_heads;
        let value_heads = config.linear_num_value_heads;
        let key_dim = config.linear_key_head_dim;
        let value_dim = config.linear_value_head_dim;
        let conv_channels = key_heads
            .checked_mul(key_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(value_heads.checked_mul(value_dim)?))
            .ok_or(StateError::DimensionOverflow)?;
        let recurrent_len = value_heads
            .checked_mul(value_dim)
            .and_then(|v| v.checked_mul(key_dim))
            .ok_or(StateError::DimensionOverflow)?;
        let conv_len = conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))
            .ok_or(StateError::DimensionOverflow)?;
        Ok(Self {
            recurrent: vec![0.0; recurrent_len],
            conv_history: vec![0.0; conv_len],
            conv_cursor: 0,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_channels,
            conv_kernel: config.linear_conv_kernel_dim,
        })
    }

    pub fn reset(&mut self) {
        self.recurrent.fill(0.0);
        self.conv_history.fill(0.0);
        self.conv_cursor = 0;
    }

    pub fn conv_history_rows(&self) -> usize {
        self.conv_kernel.saturating_sub(1)
    }

    pub fn push_conv_row(&mut self, row: &[f32]) -> Result<(), StateError> {
        if row.len() != self.conv_channels {
            return Err(StateError::Length {
                what: "reference GDN convolution row",
                expected: self.conv_channels,
                actual: row.len(),
            });
        }
        let rows = self.conv_history_rows();
        if rows == 0 {
            return Ok(());
        }
        let start = self.conv_cursor * self.conv_channels;
        self.conv_history[start..start + self.conv_channels].copy_from_slice(row);
        self.conv_cursor = (self.conv_cursor + 1) % rows;
        Ok(())
    }
}

/// Reference-only PLE/QSA metadata used by CPU probes.  Full KV values are
/// ordinary vectors here; production Qwen4State keeps them on the device.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceQsaLayerState {
    pub full_keys: Vec<f32>,
    pub full_values: Vec<f32>,
    pub raw_index_keys: Vec<f32>,
    pub pooled_keys: Vec<f32>,
    pub pooled_positions: Vec<usize>,
    pub partial_keys: Vec<f32>,
    pub partial_values: Vec<f32>,
    pub partial_positions: Vec<usize>,
    pub selected_indices: Vec<usize>,
    pub position: usize,
}

impl ReferenceQsaLayerState {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            full_keys: Vec::with_capacity(max_seq_len),
            full_values: Vec::with_capacity(max_seq_len),
            raw_index_keys: Vec::with_capacity(max_seq_len),
            pooled_keys: Vec::new(),
            pooled_positions: Vec::new(),
            partial_keys: Vec::new(),
            partial_values: Vec::new(),
            partial_positions: Vec::new(),
            selected_indices: Vec::new(),
            position: 0,
        }
    }
}

/// Reference-only HC feedback.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceHyperConnectionState {
    pub feedback: Vec<f32>,
}

/// A production GDN state block: the recurrent state in the state's
/// `GdnStateFormat` (F32, or a Raw Q8 slot), optional live-only F16 error
/// feedback, and the F32 convolution history. Snapshots never own feedback.
/// The tensors must be freed by `Qwen4State::free_gpu`; no `Drop` attempts
/// to call HIP.
pub struct GdnGpuState {
    pub recurrent: GpuTensor,
    pub ef_residual: Option<GpuTensor>,
    pub conv: GpuTensor,
}

/// Whether a transition discards the uninterrupted live EF history. Session,
/// prefix and disk restores share `Restore`; lane reinitialization uses `Reset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GdnResidualTransition {
    Reset,
    Restore,
    PartialRollback,
    #[cfg(test)]
    Keep,
}

/// The lab event kind of a residual transition that dispatched a zero
/// sequence.
#[cfg(feature = "lab")]
fn lab_residual_kind(event: GdnResidualTransition) -> Option<crate::lab_window::LabResidualKind> {
    use crate::lab_window::LabResidualKind as Kind;
    match event {
        GdnResidualTransition::Reset => Some(Kind::Reset),
        GdnResidualTransition::Restore => Some(Kind::Restore),
        GdnResidualTransition::PartialRollback => Some(Kind::PartialRollback),
        #[cfg(test)]
        GdnResidualTransition::Keep => None,
    }
}

fn gdn_residual_reset_required(format: GdnStateFormat, event: GdnResidualTransition) -> bool {
    format.has_residual()
        && match event {
            GdnResidualTransition::Reset
            | GdnResidualTransition::Restore
            | GdnResidualTransition::PartialRollback => true,
            #[cfg(test)]
            GdnResidualTransition::Keep => false,
        }
}

/// Keep the legacy Q8/F32 metadata byte-for-byte. EF has Q8 payload extents,
/// but a distinct restore ABI even when a caller bypasses the cache domain.
fn gdn_session_tag(scope: u64, format: GdnStateFormat) -> u64 {
    match format {
        GdnStateFormat::F32 | GdnStateFormat::Q8 => scope,
        GdnStateFormat::Q8Ef => scope ^ 0x7138_5f65_662d_7273,
    }
}

fn validate_gdn_session_tag(
    scope: u64,
    format: GdnStateFormat,
    tag: u64,
) -> Result<(), StateError> {
    if tag == gdn_session_tag(scope, format) {
        Ok(())
    } else {
        Err(StateError::SnapshotShape)
    }
}

fn validate_gdn_profile(config: &Qwen4Config, format: GdnStateFormat) -> Result<(), StateError> {
    if config.profile.is_ar_only() && format.has_residual() {
        return Err(StateError::Config(
            "qwen4: stock-FP8 pins F32 GDN state; q8_ef is refused".into(),
        ));
    }
    Ok(())
}

/// A QSA layer's persistent F16 K/V arena, derived from the layer's F32 K/V
/// (`HIPFIRE_QWEN4_QSA_F16_ARENA`). `keys` is F16 `[token][width]`, `values`
/// the block-transposed F16 `[token / 4][width][token % 4]` (the layout
/// `indexed_attention_kv_f16vb` writes; `width = kv_heads * head_dim`), both
/// `qsa_f16_arena_rows(full_capacity)` rows. Never snapshot-copied: the next
/// gathered prefill attend lazily rebuilds invalid rows using the unchanged
/// converter. `valid_len` counts leading rows equal to `f16(F32 K/V)`.
pub struct QsaF16Arena {
    pub keys: GpuTensor,
    pub values: GpuTensor,
    pub valid_len: QsaF16ValidLen,
}

/// A QSA layer's persistent BF16 pooled-index mirror
/// (`HIPFIRE_QWEN4_QSA_POOLED_BF16`), derived from the F32 `pooled_keys`
/// (whose values are already BF16-rounded, so the high half of each F32 is
/// exact). `mirror` is BF16 `[pooled_capacity * index_dim]`. Never
/// snapshot-copied: `valid_blocks` counts leading pooled blocks whose mirror
/// equals the canonical pooled keys, and the selector falls back to the F32
/// keys for any block past it.
pub struct QsaPooledBf16Arena {
    pub mirror: GpuTensor,
    pub valid_blocks: QsaPooledBf16ValidBlocks,
}

/// A production QSA block.  Full K/V, raw and pooled index buffers have the
/// max sequence capacity shape; `*_len` fields are active append lengths. The
/// K/V arenas hold `format`'s rows (`full_row_units` tensor units per token)
/// and the raw/pooled index keys its `index_dtype`. Under the VMM QSA
/// backend the context arenas (the optional F16 arena included) are VMM
/// owners whose accessible prefix (`buf.size()`) is only what
/// [`Qwen4State::ensure_mapped_capacity`] has mapped; partial/selected
/// buffers are always fixed allocations.
pub struct QsaGpuState {
    pub format: QsaKvFormat,
    pub full_row_units: usize,
    pub full_keys: GpuTensor,
    pub full_values: GpuTensor,
    pub raw_index_keys: GpuTensor,
    pub pooled_keys: GpuTensor,
    pub partial_keys: GpuTensor,
    pub partial_values: GpuTensor,
    pub selected_indices: GpuTensor,
    /// Derived F16 K/V (`None` = the gathered attention converts per launch).
    pub f16_arena: Option<QsaF16Arena>,
    /// Derived BF16 pooled-index mirror (`None` = score from the F32 keys).
    pub pooled_bf16: Option<QsaPooledBf16Arena>,
    pub full_capacity: usize,
    pub raw_capacity: usize,
    pub pooled_capacity: usize,
    pub partial_capacity: usize,
    pub selected_capacity: usize,
    pub position_capacity: usize,
    pub full_len: usize,
    pub raw_len: usize,
    pub pooled_len: usize,
    pub partial_len: usize,
    pub selected_len: usize,
    pub position: usize,
}

impl QsaGpuState {
    fn mark(&self) -> QsaMark {
        QsaMark {
            full_len: self.full_len,
            raw_len: self.raw_len,
            pooled_len: self.pooled_len,
            partial_len: self.partial_len,
            selected_len: self.selected_len,
            position: self.position,
        }
    }

    fn mark_capacity(&self) -> QsaMarkCapacity {
        QsaMarkCapacity {
            full: self.full_capacity,
            raw: self.raw_capacity,
            pooled: self.pooled_capacity,
            partial: self.partial_capacity,
            selected: self.selected_capacity,
            position: self.position_capacity,
        }
    }

    /// The F16 arena as the attention reads it, `None` without one.
    pub fn f16_arena_ref(&self) -> Option<QsaF16ArenaRef<'_>> {
        self.f16_arena.as_ref().map(|arena| QsaF16ArenaRef {
            keys: &arena.keys,
            values: &arena.values,
            valid_len: &arena.valid_len,
        })
    }

    /// Leading tokens whose F16 rows are valid against the F32 cache (0
    /// without an arena).
    pub fn f16_valid_len(&self) -> usize {
        self.f16_arena_ref()
            .map_or(0, |arena| arena.valid(self.full_len))
    }

    /// The pooled BF16 mirror as the selector reads it, `None` without one.
    pub fn pooled_bf16_ref(&self) -> Option<QsaPooledBf16MirrorRef<'_>> {
        self.pooled_bf16
            .as_ref()
            .map(|arena| QsaPooledBf16MirrorRef {
                mirror: &arena.mirror,
                valid_blocks: &arena.valid_blocks,
            })
    }

    /// Leading pooled blocks whose BF16 mirror is valid against the F32
    /// pooled keys (0 without a mirror).
    pub fn pooled_bf16_valid_blocks(&self) -> usize {
        self.pooled_bf16_ref()
            .map_or(0, |mirror| mirror.valid(self.pooled_len))
    }

    /// Clamp the derived arenas' valid prefixes to the committed rows: the F16
    /// arena to `full_len` tokens and the pooled BF16 mirror to `pooled_len`
    /// blocks. Every path that rewinds `full_len`/`pooled_len` calls it: rows
    /// past the new marks are rewritten later without the arenas seeing them.
    pub(crate) fn clamp_f16_valid(&self) {
        if let Some(arena) = &self.f16_arena {
            arena.valid_len.clamp(self.full_len);
        }
        if let Some(arena) = &self.pooled_bf16 {
            arena.valid_blocks.clamp(self.pooled_len);
        }
    }

    /// Forget every converted row and mirrored block (reset, or the F32 rows
    /// and pooled keys were replaced).
    pub(crate) fn invalidate_f16_arena(&self) {
        if let Some(arena) = &self.f16_arena {
            arena.valid_len.invalidate();
        }
        if let Some(arena) = &self.pooled_bf16 {
            arena.valid_blocks.invalidate();
        }
    }

    /// The context arenas, in allocation order (full K, full V, raw and
    /// pooled index keys, then the F16 arena's K and V and the pooled BF16
    /// mirror when present).
    fn context_arenas(&self) -> impl Iterator<Item = &GpuTensor> {
        [
            &self.full_keys,
            &self.full_values,
            &self.raw_index_keys,
            &self.pooled_keys,
        ]
        .into_iter()
        .chain(
            self.f16_arena
                .iter()
                .flat_map(|arena| [&arena.keys, &arena.values]),
        )
        .chain(self.pooled_bf16.iter().map(|arena| &arena.mirror))
    }

    /// Bytes of this layer's context arenas still unmapped between `mapped`
    /// and `through` tokens, row-exact per arena (the pooled BF16 mirror
    /// grows with the pooled keys, one row per `compress` tokens).
    fn context_growth_bytes(&self, compress: usize, mapped: usize, through: usize) -> u64 {
        let plain = [
            (&self.full_keys, self.full_capacity, 1, false),
            (&self.full_values, self.full_capacity, 1, false),
            (&self.raw_index_keys, self.raw_capacity, 1, false),
            (&self.pooled_keys, self.pooled_capacity, compress, false),
        ];
        let f16_rows = qsa_f16_arena_rows(self.full_capacity);
        let f16 = self
            .f16_arena
            .iter()
            .flat_map(|arena| [&arena.keys, &arena.values])
            .map(|tensor| (tensor, f16_rows, 1, true));
        let mirror = self
            .pooled_bf16
            .iter()
            .map(|arena| (&arena.mirror, self.pooled_capacity, compress, false));
        plain
            .into_iter()
            .chain(f16)
            .chain(mirror)
            .map(|(tensor, capacity, tokens_per_row, blocks)| {
                context_arena_bytes_through(tensor, capacity, tokens_per_row, blocks, through)
                    .saturating_sub(context_arena_bytes_through(
                        tensor,
                        capacity,
                        tokens_per_row,
                        blocks,
                        mapped,
                    ))
            })
            .sum()
    }

    fn context_arenas_mut(&mut self) -> smallvec::SmallVec<[ContextArena<'_>; 7]> {
        let mut arenas = smallvec::smallvec![
            ContextArena {
                tensor: &mut self.full_keys,
                rows: self.full_capacity,
                pooled: false,
                blocks: false,
            },
            ContextArena {
                tensor: &mut self.full_values,
                rows: self.full_capacity,
                pooled: false,
                blocks: false,
            },
            ContextArena {
                tensor: &mut self.raw_index_keys,
                rows: self.raw_capacity,
                pooled: false,
                blocks: false,
            },
            ContextArena {
                tensor: &mut self.pooled_keys,
                rows: self.pooled_capacity,
                pooled: true,
                blocks: false,
            },
        ];
        if let Some(arena) = &mut self.f16_arena {
            let rows = qsa_f16_arena_rows(self.full_capacity);
            for tensor in [&mut arena.keys, &mut arena.values] {
                arenas.push(ContextArena {
                    tensor,
                    rows,
                    pooled: false,
                    blocks: true,
                });
            }
        }
        if let Some(arena) = &mut self.pooled_bf16 {
            arenas.push(ContextArena {
                tensor: &mut arena.mirror,
                rows: self.pooled_capacity,
                pooled: true,
                blocks: false,
            });
        }
        arenas
    }
}

/// One QSA context arena owner with its row capacity.
struct ContextArena<'a> {
    tensor: &'a mut GpuTensor,
    rows: usize,
    /// One row per `compress` tokens instead of one per token.
    pooled: bool,
    /// Rows are written in whole four-token blocks (the F16 arena's
    /// block-transposed V): the mapped prefix covers the covered tokens
    /// rounded up to four.
    blocks: bool,
}

impl ContextArena<'_> {
    fn tokens_per_row(&self, compress: usize) -> usize {
        if self.pooled {
            compress
        } else {
            1
        }
    }
}

/// Tokens an arena whose accessible prefix holds `mapped_rows` rows covers.
fn qsa_covered_tokens(mapped_rows: usize, tokens_per_row: usize, max_tokens: usize) -> usize {
    mapped_rows.saturating_mul(tokens_per_row).min(max_tokens)
}

/// The shared KV chunk plan for a `rows`-row arena of `row_bytes` rows in
/// driver pages of `granularity` bytes.
fn context_arena_plan(
    row_bytes: usize,
    rows: usize,
    granularity: usize,
) -> Result<KvChunkPlan, StateError> {
    KvChunkPlan::new(
        row_bytes,
        rows,
        DEFAULT_KV_CHUNK_TOKENS,
        granularity,
        DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
    )
    .map_err(StateError::MapPlan)
}

/// Bytes per row of a context arena holding `rows` capacity rows.
fn arena_row_bytes(tensor: &GpuTensor, rows: usize) -> Result<usize, StateError> {
    let bytes = tensor.byte_size();
    if rows == 0 || !bytes.is_multiple_of(rows) {
        return Err(StateError::DimensionOverflow);
    }
    Ok(bytes / rows)
}

/// Logical bytes of the first rows of a `capacity`-row context arena that
/// cover `tokens` tokens (`tokens_per_row` per row; `blocks` rounds the rows
/// up to whole four-token blocks), the rows `ensure_mapped_capacity` maps.
pub(crate) fn context_arena_bytes_through(
    tensor: &GpuTensor,
    capacity: usize,
    tokens_per_row: usize,
    blocks: bool,
    tokens: usize,
) -> u64 {
    if capacity == 0 {
        return 0;
    }
    let rows = tokens.div_ceil(tokens_per_row);
    let rows = if blocks {
        qsa_f16_arena_rows(rows)
    } else {
        rows
    };
    (tensor.byte_size() / capacity) as u64 * rows.min(capacity) as u64
}

/// Draft rows a decode request must map ahead for: the widest verify window
/// the scheduler can actually issue, not the configured MTP depth alone. A
/// window of `rows` rows (`1 + drafts`: wide verify, copy and n-gram windows
/// may be wider than `configured_draft_rows`) started at the last budgeted
/// position ends `rows - 1` positions past the budget. Mapping is capacity
/// only; no length, position or watermark advances with it.
pub(crate) fn map_ahead_draft_rows(
    configured_draft_rows: usize,
    widest_window_rows: usize,
) -> usize {
    configured_draft_rows.max(widest_window_rows.saturating_sub(1))
}

/// Tokens a decode request maps ahead of its position so per-token growth
/// checks never remap: the remaining new-token budget plus `max_draft_rows`
/// ([`map_ahead_draft_rows`]) speculative rows, clamped to `capacity`. A
/// missing or zero budget (unbounded or finished request) maps nothing ahead.
pub(crate) fn decode_map_ahead_tokens(
    position: usize,
    remaining_new_tokens: Option<usize>,
    max_draft_rows: usize,
    capacity: usize,
) -> Option<usize> {
    let budget = remaining_new_tokens.filter(|&tokens| tokens != 0)?;
    Some(
        position
            .saturating_add(budget)
            .saturating_add(max_draft_rows)
            .min(capacity),
    )
}

/// All of `tensor` as a fixed session-snapshot part.
pub(crate) fn whole_part(tensor: &GpuTensor) -> StatePart<'_> {
    StatePart {
        buf: &tensor.buf,
        offset: 0,
        bytes: tensor.byte_size(),
    }
}

/// The first `rows` of an append-only arena holding `capacity` rows.
pub(crate) fn row_stream(
    tensor: &GpuTensor,
    capacity: usize,
    rows: usize,
) -> Result<RowStream<'_>, StateError> {
    Ok(RowStream {
        buf: &tensor.buf,
        row_bytes: arena_row_bytes(tensor, capacity)?,
        rows,
    })
}

/// Session-snapshot metadata: little-endian `u64` words.
pub(crate) fn meta_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// Inverse of [`meta_bytes`]; `None` unless whole words.
pub(crate) fn meta_words(meta: &[u8]) -> Option<Vec<u64>> {
    meta.len().is_multiple_of(8).then(|| {
        meta.chunks_exact(8)
            .map(|word| u64::from_le_bytes(word.try_into().expect("8-byte chunk")))
            .collect()
    })
}

/// Map a VMM context arena of `rows` capacity rows to cover `required_rows`
/// rows and zero exactly the newly mapped bytes. Returns the rows its
/// accessible prefix then holds.
fn grow_context_arena(
    gpu: &mut Gpu,
    tensor: &mut GpuTensor,
    rows: usize,
    required_rows: usize,
    device: i32,
) -> Result<usize, StateError> {
    let row_bytes = arena_row_bytes(tensor, rows)?;
    let mapped = gpu.vmm_mapped_bytes(tensor).ok_or(StateError::VmmOwner)?;
    let granularity = gpu.vmm_granularity(tensor).ok_or(StateError::VmmOwner)?;
    let plan = context_arena_plan(row_bytes, rows, granularity)?;
    let Some(growth) = plan
        .growth(mapped, required_rows)
        .map_err(StateError::MapPlan)?
    else {
        return Ok(plan.token_capacity(mapped));
    };
    let mapped = gpu
        .grow_vmm_tensor(tensor, growth.size_bytes, &[device])
        .map_err(|error| StateError::MapGrowth {
            offset: growth.offset_bytes,
            bytes: growth.size_bytes,
            error,
        })?;
    // The owner view is capped at the logical size; a final page may pass it.
    let element = tensor.dtype.size();
    let fresh = tensor.buf.size() - growth.offset_bytes;
    let view = tensor.sub_offset(growth.offset_bytes / element, fresh / element);
    // Developer extent check: the zero-fill covers exactly the newly mapped
    // bytes `[growth.offset_bytes, growth.offset_bytes + fresh)` of the owner.
    let depth = || rdna_compute::QsaExtentDepth {
        rows,
        end_position: required_rows,
        block_count: 0,
    };
    if gpu.flags.qwen4_qsa_extent_check {
        gpu.qsa_extent_check(
            "qsa_context_grow_zero",
            tensor,
            growth.offset_bytes,
            fresh,
            depth(),
            "grow_context_arena",
        );
    }
    gpu.memset_stream_ordered(&view.buf, 0, view.buf.size())
        .map_err(StateError::Hip)?;
    if gpu.flags.qwen4_qsa_extent_check {
        gpu.qsa_extent_post_launch(
            "qsa_context_grow_zero",
            tensor,
            growth.offset_bytes,
            fresh,
            depth(),
            "grow_context_arena",
        );
    }
    Ok(plan.token_capacity(mapped))
}

/// The QSA state format for a `memory.kv_cache` request on `gpu`
/// ([`hipfire_runtime::kv_mode::resolve_qwen4`]): `bf16` is the exact
/// reference state ([`QsaKvFormat::F32`]); `fp8` needs gfx1201 or gfx942 and
/// a head geometry its kernels implement (head_dim 256, an even KV-head
/// count) and is [`QsaKvFormat::Fp8`] (OCP E4M3, BF16 index keys) on gfx1201
/// and [`QsaKvFormat::Fp8Fnuz`] (E4M3 FNUZ hardware codes, F32 index and
/// pooled keys) on gfx942; `q8` ([`QsaKvFormat::Q8`]) needs gfx11/gfx12 and
/// head_dim 256. `auto` is fp8 on gfx1201 and q8 on gfx1151 and gfx1100 where
/// the geometry holds (the reference F32 state where it does not) and the
/// reference F32 state elsewhere, gfx942 included: gfx942 serves fp8 K/V only
/// when explicitly requested, and `bf16` is the exact F32 state everywhere.
/// An explicit request the device or model cannot serve is refused, never
/// rewritten.
pub fn resolve_qsa_format(
    request: &str,
    gpu: &Gpu,
    config: &Qwen4Config,
) -> Result<QsaKvFormat, String> {
    resolve_qsa_format_for_arch(request, &gpu.arch, config)
}

/// [`resolve_qsa_format`] on the architecture name alone.
fn resolve_qsa_format_for_arch(
    request: &str,
    arch: &str,
    config: &Qwen4Config,
) -> Result<QsaKvFormat, String> {
    use hipfire_runtime::kv_mode::{resolve_qwen4, KvMode};
    let (kv_heads, head_dim) = (config.num_key_value_heads, config.head_dim);
    let fp8 = if arch == "gfx942" {
        QsaKvFormat::Fp8Fnuz
    } else {
        QsaKvFormat::Fp8
    };
    let fp8_geometry = fp8.supports(kv_heads, head_dim);
    let auto = matches!(request.trim(), "" | "auto");
    match resolve_qwen4(request, arch)?.mode {
        KvMode::Fp8 if fp8_geometry => Ok(fp8),
        KvMode::Fp8 if auto => Ok(QsaKvFormat::F32),
        KvMode::Fp8 => Err(format!(
            "qwen4: fp8 QSA K/V needs head_dim 256 and an even KV-head count \
             (have {kv_heads} x {head_dim}); use bf16"
        )),
        KvMode::Q8 if QsaKvFormat::Q8.supports(kv_heads, head_dim) => Ok(QsaKvFormat::Q8),
        KvMode::Q8 if auto => Ok(QsaKvFormat::F32),
        KvMode::Q8 => Err(format!(
            "qwen4: q8 QSA K/V needs head_dim 256 (have {kv_heads} x {head_dim}); use bf16"
        )),
        _ => Ok(QsaKvFormat::F32),
    }
}

/// The GDN recurrent state format for a `state_quant` request (Qwen3.5's
/// DeltaNet knob), independent of the device: `auto`/`q8` is Qwen3.5's Q8
/// state where its kernels apply (128x128 heads), `fp32` the exact reference
/// state. An explicit request the model cannot serve is refused, never
/// rewritten.
///
/// This is the arch-agnostic meaning; the product path resolves through
/// [`resolve_state_format`], whose `auto` is per-architecture (gfx1151 F32,
/// see there).
///
/// The AR-only Flash-Next profile is the stock-FP8 recipe's
/// (`Qwen4TierRecipe::StockFp8` is its only recipe), whose numerical contract
/// pins the F32 recurrent state: `auto` resolves to F32 there and an explicit
/// `q8` is refused.
pub fn resolve_gdn_format(request: &str, config: &Qwen4Config) -> Result<GdnStateFormat, String> {
    let request = request.trim().to_ascii_lowercase();
    if config.profile.is_ar_only() {
        return match request.as_str() {
            "" | "auto" | "fp32" | "f32" => Ok(GdnStateFormat::F32),
            "q8" | "int8" | "q8_ef" => Err(format!(
                "qwen4: the stock-FP8 artifact pins the F32 GDN recurrent state (its numerical \
                 contract); state_quant '{request}' is refused, use fp32 or leave it unset"
            )),
            other => Err(format!(
                "qwen4: unsupported GDN state_quant '{other}' (expected auto|fp32)"
            )),
        };
    }
    let (key_dim, value_dim) = (config.linear_key_head_dim, config.linear_value_head_dim);
    let q8 = GdnStateFormat::Q8.supports(key_dim, value_dim);
    match request.as_str() {
        "" | "auto" if q8 => Ok(GdnStateFormat::Q8),
        "" | "auto" | "fp32" | "f32" => Ok(GdnStateFormat::F32),
        "q8" | "int8" if q8 => Ok(GdnStateFormat::Q8),
        // Explicit-only Q8 with a live f16 error-feedback residual (Track B,
        // option (a)); `auto` never selects it.
        "q8_ef" if q8 => Ok(GdnStateFormat::Q8Ef),
        "q8" | "int8" | "q8_ef" => Err(format!(
            "qwen4: {request} GDN state needs 128x128 heads (have {key_dim} x {value_dim}); use fp32"
        )),
        other => Err(format!(
            "qwen4: unsupported GDN state_quant '{other}' (expected auto|q8|q8_ef|fp32)"
        )),
    }
}

/// [`resolve_gdn_format`] with the per-architecture `auto` policy: on exactly
/// `gfx1151`, a non-AR-only profile with Q8-capable (128x128) GDN heads and an
/// empty/`auto` request resolves to the exact F32 recurrent state (chosen by
/// KLD; this differs numerically from the former Q8 default). Every other
/// request or architecture is `resolve_gdn_format` unchanged: gfx1100,
/// gfx1201 and the rest keep Q8 for `auto`, the stock-FP8 profile stays pinned
/// to F32, and explicit `q8|int8|fp32|f32` values are honored or refused,
/// never rewritten.
///
/// `q8_inline` is the raw `HIPFIRE_QWEN4_GDN_Q8_INLINE` developer flag. A
/// resolved `q8_ef` with it set is refused on every architecture: `q8_ef` has
/// no inline WY kernel (none has run on a GPU), and the flag is never silently
/// ignored. No other result depends on it.
fn resolve_gdn_format_for_arch(
    request: &str,
    config: &Qwen4Config,
    arch: &str,
    q8_inline: bool,
) -> Result<GdnStateFormat, String> {
    let auto = matches!(request.trim().to_ascii_lowercase().as_str(), "" | "auto");
    if auto
        && arch == "gfx1151"
        && !config.profile.is_ar_only()
        && GdnStateFormat::Q8.supports(config.linear_key_head_dim, config.linear_value_head_dim)
    {
        return Ok(GdnStateFormat::F32);
    }
    let format = resolve_gdn_format(request, config)?;
    if q8_inline && format == GdnStateFormat::Q8Ef {
        return Err(
            "qwen4: state_quant q8_ef does not support the inline WY developer route \
                    (HIPFIRE_QWEN4_GDN_Q8_INLINE=1) yet; unset one of them"
                .into(),
        );
    }
    Ok(format)
}

/// Storage formats of a Qwen4 request state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Qwen4StateFormat {
    /// QSA K/V and index keys ([`resolve_qsa_format`]).
    pub qsa: QsaKvFormat,
    /// GDN recurrent states ([`resolve_gdn_format`]).
    pub gdn: GdnStateFormat,
    /// Persistent per-layer F16 K/V arenas for the gathered prefill attention
    /// (`HIPFIRE_QWEN4_QSA_F16_ARENA`, gfx1151; [`resolve_state_format`]).
    pub qsa_f16_arena: bool,
    /// Persistent per-layer BF16 pooled-index mirrors for the PM scorer
    /// (`HIPFIRE_QWEN4_QSA_POOLED_BF16`, gfx1151 F32 index_dim 128;
    /// [`resolve_state_format`]).
    pub qsa_pooled_bf16: bool,
}

impl Qwen4StateFormat {
    /// The exact F32 reference state.
    pub const F32: Self = Self {
        qsa: QsaKvFormat::F32,
        gdn: GdnStateFormat::F32,
        qsa_f16_arena: false,
        qsa_pooled_bf16: false,
    };

    /// The QSA layers' context storage: what admission accounts and the state
    /// allocates.
    pub const fn layout(&self) -> QsaStateLayout {
        QsaStateLayout {
            kv: self.qsa,
            f16_arena: self.qsa_f16_arena,
            pooled_bf16: self.qsa_pooled_bf16,
        }
    }
}

/// Both state formats: `kv_request` is `memory.kv_cache`, `state_request`
/// the `state_quant` load option. The exact F16 prefill arena is on by
/// default only for gfx1151 F32 QSA K/V with head_dim 256.
///
/// The GDN `auto` is per-architecture: gfx1151 (non-AR-only profile, 128x128
/// heads) resolves to the exact F32 reference state, chosen by KLD, which is
/// numerically different from the former Q8 default; gfx1100, gfx1201 and the
/// other architectures keep Q8. The stock-FP8 profile pins F32 everywhere.
/// The kill switch is `state_quant = "q8"` / `HIPFIRE_STATE_QUANT=q8`
/// (explicit values are honored or refused, never rewritten). `q8_ef` together
/// with the raw `HIPFIRE_QWEN4_GDN_Q8_INLINE=1` developer flag is refused on
/// every architecture (`q8_ef` has no inline kernel yet).
pub fn resolve_state_format(
    kv_request: &str,
    state_request: &str,
    gpu: &Gpu,
    config: &Qwen4Config,
) -> Result<Qwen4StateFormat, String> {
    let qsa = resolve_qsa_format(kv_request, gpu, config)?;
    Ok(Qwen4StateFormat {
        qsa,
        gdn: resolve_gdn_format_for_arch(
            state_request,
            config,
            &gpu.arch,
            gpu.flags.qwen4_gdn_q8_inline,
        )?,
        qsa_f16_arena: rdna_compute::tensor_ops::qsa_f16_arena_enabled(gpu, qsa, config.head_dim),
        qsa_pooled_bf16: qsa_pooled_bf16_mirror_enabled(
            gpu,
            qsa,
            config
                .indexer_kv_heads
                .saturating_mul(config.indexer_head_dim),
        ),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotDimensions {
    gdn_count: usize,
    qsa_count: usize,
    max_seq_len: usize,
    selected_capacity: usize,
    shape_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen4StateSnapshot {
    model_id: u64,
    reset_epoch: u64,
    transaction_generation: u64,
    arena_generation: u64,
    dimensions: SnapshotDimensions,
}

/// Whether a [`Qwen4State`] owns the snapshot arena's device mirror of the
/// state a forward overwrites in place (GDN recurrent and conv, the QSA fixed
/// arenas, the PLE convolution history and the hyper feedback): the copy
/// [`Qwen4State::snapshot`] captures into and [`Qwen4State::restore`] copies
/// back from. One choice per state, made at construction; the byte
/// accounting ([`Qwen4State::device_bytes_with_mirror`],
/// [`Qwen4State::device_allocation_sizes_with_mirror`]) takes the same value,
/// so planned bytes follow what is allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateMirror {
    /// The mirror exists: the state can be snapshotted, restored and rolled
    /// back (prompt windows, verify windows, turn transactions). The resident
    /// request state, every MTP lane state and the MTP head's own state.
    Snapshot,
    /// No mirror: the state is only stepped and reset (an AR lane, whose
    /// window is one committed row). Every mirror byte is absent from its
    /// accounting, and [`Qwen4State::snapshot`] refuses with
    /// [`StateError::MirrorAbsent`] instead of allocating one.
    Absent,
}

/// Reusable model-owned GPU snapshot storage.  The append-only QSA full K/V
/// arenas are deliberately absent: a ticket records their active marks and
/// restore only rewinds those marks.  Only state that a forward may overwrite
/// in place is copied into this bounded arena.
pub(crate) struct Qwen4StateSnapshotArena {
    /// [`StateMirror::Absent`]: every tensor below is empty and no ticket can
    /// ever be active.
    mirror: StateMirror,
    recurrent: Vec<GpuTensor>,
    conv: Vec<GpuTensor>,
    qsa_partial_keys: Vec<GpuTensor>,
    qsa_partial_values: Vec<GpuTensor>,
    qsa_raw_circular: Vec<GpuTensor>,
    qsa_selected: Vec<GpuTensor>,
    ple_conv: Option<GpuTensor>,
    hyper_feedback: Option<GpuTensor>,
    qsa_marks: Vec<QsaMark>,
    ple_history: PleHistory,
    position: usize,
    /// Live GDN ring slot when the snapshot brackets an armed verify: the
    /// recurrent state was not copied (the verify leaves that slot intact).
    gdn_live: Option<usize>,
    model_id: u64,
    active: bool,
    /// The entry stays active after a committed window so the window can
    /// still be rolled back ([`Qwen4State::retain_validated`]); the next
    /// snapshot, reset or session restore drops it.
    retained: bool,
    generation: u64,
}

impl Qwen4StateSnapshotArena {
    fn new(
        gpu: &mut Gpu,
        gdn: &[GdnGpuState],
        qsa: &[QsaGpuState],
        ple_conv: Option<&GpuTensor>,
        hyper_feedback: Option<&GpuTensor>,
        model_id: u64,
        ple_history: PleHistory,
        position: usize,
    ) -> Result<Self, StateError> {
        let mut allocated = Vec::new();
        let result = (|| -> Result<(), StateError> {
            for layer in gdn {
                allocated.push(
                    gpu.zeros(&layer.recurrent.shape, layer.recurrent.dtype)
                        .map_err(StateError::Hip)?,
                );
                allocated.push(
                    gpu.zeros(&layer.conv.shape, DType::F32)
                        .map_err(StateError::Hip)?,
                );
            }
            for layer in qsa {
                allocated.push(
                    gpu.zeros(&layer.partial_keys.shape, DType::F32)
                        .map_err(StateError::Hip)?,
                );
                allocated.push(
                    gpu.zeros(&layer.partial_values.shape, DType::F32)
                        .map_err(StateError::Hip)?,
                );
                let raw_width = layer
                    .raw_index_keys
                    .numel()
                    .checked_div(layer.raw_capacity)
                    .ok_or(StateError::DimensionOverflow)?;
                let circular_elements = layer
                    .partial_capacity
                    .checked_mul(raw_width)
                    .ok_or(StateError::DimensionOverflow)?;
                allocated.push(
                    gpu.zeros(&[circular_elements], layer.raw_index_keys.dtype)
                        .map_err(StateError::Hip)?,
                );
                allocated.push(
                    gpu.zeros(&layer.selected_indices.shape, DType::Raw)
                        .map_err(StateError::Hip)?,
                );
            }
            if let Some(ple_conv) = ple_conv {
                allocated.push(
                    gpu.zeros(&ple_conv.shape, DType::F32)
                        .map_err(StateError::Hip)?,
                );
            }
            if let Some(hyper_feedback) = hyper_feedback {
                allocated.push(
                    gpu.zeros(&hyper_feedback.shape, DType::F32)
                        .map_err(StateError::Hip)?,
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            for tensor in allocated.into_iter().rev() {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        let mut next = || allocated.remove(0);
        let mut recurrent = Vec::with_capacity(gdn.len());
        let mut conv = Vec::with_capacity(gdn.len());
        for _ in gdn {
            recurrent.push(next());
            conv.push(next());
        }
        let mut qsa_partial_keys = Vec::with_capacity(qsa.len());
        let mut qsa_partial_values = Vec::with_capacity(qsa.len());
        let mut qsa_raw_circular = Vec::with_capacity(qsa.len());
        let mut qsa_selected = Vec::with_capacity(qsa.len());
        for _ in qsa {
            qsa_partial_keys.push(next());
            qsa_partial_values.push(next());
            qsa_raw_circular.push(next());
            qsa_selected.push(next());
        }
        let ple_conv = ple_conv.map(|_| next());
        let hyper_feedback = hyper_feedback.map(|_| next());
        if !allocated.is_empty() {
            for tensor in allocated.into_iter().rev() {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(StateError::AllocationBookkeeping);
        }
        Ok(Self {
            mirror: StateMirror::Snapshot,
            recurrent,
            conv,
            qsa_partial_keys,
            qsa_partial_values,
            qsa_raw_circular,
            qsa_selected,
            ple_conv,
            hyper_feedback,
            qsa_marks: qsa.iter().map(QsaGpuState::mark).collect(),
            ple_history,
            position,
            gdn_live: None,
            model_id,
            active: false,
            retained: false,
            generation: 0,
        })
    }

    /// The arena of a [`StateMirror::Absent`] state: nothing is allocated and
    /// no ticket can ever be issued against it.
    fn absent(model_id: u64, ple_history: PleHistory) -> Self {
        Self {
            mirror: StateMirror::Absent,
            recurrent: Vec::new(),
            conv: Vec::new(),
            qsa_partial_keys: Vec::new(),
            qsa_partial_values: Vec::new(),
            qsa_raw_circular: Vec::new(),
            qsa_selected: Vec::new(),
            ple_conv: None,
            hyper_feedback: None,
            qsa_marks: Vec::new(),
            ple_history,
            position: 0,
            gdn_live: None,
            model_id,
            active: false,
            retained: false,
            generation: 0,
        }
    }

    /// The typed refusal of every snapshot, restore and rollback on a state
    /// without a mirror.
    fn require_mirror(&self) -> Result<(), StateError> {
        match self.mirror {
            StateMirror::Snapshot => Ok(()),
            StateMirror::Absent => Err(StateError::MirrorAbsent),
        }
    }

    fn dimensions(state: &Qwen4State) -> SnapshotDimensions {
        let mut hash = 0xcbf29ce484222325u64;
        let mut mix = |value: usize| {
            hash ^= value as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        };
        mix(state.gdn.len());
        mix(state.qsa.len());
        mix(state.max_seq_len);
        mix(state.qsa_selected_capacity);
        for layer in &state.gdn {
            mix(layer.recurrent.numel());
            mix(layer.conv.numel());
            mix(layer.recurrent.buf.size());
            mix(layer.conv.buf.size());
        }
        for layer in &state.qsa {
            mix(layer.full_capacity);
            mix(layer.raw_capacity);
            mix(layer.pooled_capacity);
            mix(layer.partial_capacity);
            mix(layer.selected_capacity);
            mix(layer.full_keys.numel());
            mix(layer.full_values.numel());
            mix(layer.raw_index_keys.numel());
            mix(layer.pooled_keys.numel());
            mix(layer.partial_keys.numel());
            mix(layer.partial_values.numel());
            mix(layer.selected_indices.buf.size());
            // The derived arena travels with this owner, but is not copied.
            mix(layer
                .f16_arena
                .as_ref()
                .map_or(0, |arena| arena.keys.numel()));
            mix(layer
                .pooled_bf16
                .as_ref()
                .map_or(0, |arena| arena.mirror.numel()));
        }
        mix(state.ple_conv.as_ref().map_or(0, GpuTensor::numel));
        mix(state.hyper_feedback.as_ref().map_or(0, GpuTensor::numel));
        SnapshotDimensions {
            gdn_count: state.gdn.len(),
            qsa_count: state.qsa.len(),
            max_seq_len: state.max_seq_len,
            selected_capacity: state.qsa_selected_capacity,
            shape_hash: hash,
        }
    }

    fn validate_layout(&self, state: &Qwen4State) -> Result<(), StateError> {
        let dimensions = Self::dimensions(state);
        if dimensions.gdn_count != self.recurrent.len()
            || dimensions.gdn_count != self.conv.len()
            || dimensions.qsa_count != self.qsa_marks.len()
            || dimensions.qsa_count != self.qsa_partial_keys.len()
            || dimensions.qsa_count != self.qsa_partial_values.len()
            || dimensions.qsa_count != self.qsa_raw_circular.len()
            || dimensions.qsa_count != self.qsa_selected.len()
            || dimensions.max_seq_len != state.max_seq_len
            || dimensions.selected_capacity != state.qsa_selected_capacity
        {
            return Err(StateError::SnapshotShape);
        }
        // Compare the tensors' own extents (numel, dtype, byte_size), never
        // `buf.size()`: a pooled allocation may hand back a larger buffer of
        // the same power-of-2 bucket, and the ring views the GDN states move
        // to are exact, so two equal-shaped tensors can differ in buffer size.
        // Every region copy is `byte_size()` long.
        for (layer, backup) in state.gdn.iter().zip(&self.recurrent) {
            if backup.numel() != layer.recurrent.numel() || backup.dtype != layer.recurrent.dtype {
                return Err(StateError::SnapshotShape);
            }
        }
        for (layer, backup) in state.gdn.iter().zip(&self.conv) {
            if backup.numel() != layer.conv.numel() || backup.dtype != layer.conv.dtype {
                return Err(StateError::SnapshotShape);
            }
        }
        for (index, layer) in state.qsa.iter().enumerate() {
            if self.qsa_partial_keys[index].numel() != layer.partial_keys.numel()
                || self.qsa_partial_values[index].numel() != layer.partial_values.numel()
                || self.qsa_selected[index].byte_size() != layer.selected_indices.byte_size()
            {
                return Err(StateError::SnapshotShape);
            }
            let raw_width = layer
                .raw_index_keys
                .numel()
                .checked_div(layer.raw_capacity)
                .ok_or(StateError::SnapshotShape)?;
            let expected_raw = layer
                .partial_capacity
                .checked_mul(raw_width)
                .ok_or(StateError::DimensionOverflow)?;
            if self.qsa_raw_circular[index].numel() != expected_raw
                || self.qsa_raw_circular[index].dtype != layer.raw_index_keys.dtype
            {
                return Err(StateError::SnapshotShape);
            }
        }
        if self.ple_conv.as_ref().map(GpuTensor::numel)
            != state.ple_conv.as_ref().map(GpuTensor::numel)
            || self.hyper_feedback.as_ref().map(GpuTensor::numel)
                != state.hyper_feedback.as_ref().map(GpuTensor::numel)
        {
            return Err(StateError::SnapshotShape);
        }
        Ok(())
    }

    fn validate_ticket(
        &self,
        state: &Qwen4State,
        ticket: &Qwen4StateSnapshot,
    ) -> Result<(), StateError> {
        self.require_mirror()?;
        if !self.active {
            return Err(StateError::SnapshotInactive);
        }
        if ticket.model_id != self.model_id
            || ticket.model_id != state.model_id
            || ticket.reset_epoch != state.reset_epoch
            || ticket.transaction_generation != state.transaction_generation
            || ticket.arena_generation != self.generation
            || ticket.dimensions != Self::dimensions(state)
        {
            return Err(StateError::SnapshotTicket);
        }
        if self.qsa_marks.len() != state.qsa.len() {
            return Err(StateError::SnapshotShape);
        }
        for (mark, layer) in self.qsa_marks.iter().zip(&state.qsa) {
            validate_qsa_mark(mark, &layer.mark_capacity())?;
        }
        self.validate_layout(state)
    }

    fn invalidate(&mut self) {
        if self.mirror == StateMirror::Absent {
            return;
        }
        self.active = false;
        self.retained = false;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Hand this arena to the state `model_id` now owns it for: tickets of the
    /// previous owner stop validating and the next snapshot is the new
    /// owner's.
    fn rebind(&mut self, model_id: u64) {
        self.invalidate();
        self.model_id = model_id;
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let tensors = self
            .recurrent
            .into_iter()
            .chain(self.conv)
            .chain(self.qsa_partial_keys)
            .chain(self.qsa_partial_values)
            .chain(self.qsa_raw_circular)
            .chain(self.qsa_selected)
            .chain(self.ple_conv)
            .chain(self.hyper_feedback);
        let mut first = None;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// Absolute-to-local ownership of one rank's request state (S5). A rank owns
/// the GDN/QSA state of exactly the trunk layers of its band; the PLE
/// convolution history (when the profile has PLE) and the legacy
/// `hyper_feedback` tensor belong to the input scope only. Allocation, reset,
/// digest, snapshot, session and repair loops all walk the compact local
/// vectors, so the only place an absolute layer id appears is this map.
/// [`Qwen4BandScope::single`] gives today's whole-model layout, bit for bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4StateMap {
    band: Range<usize>,
    input: bool,
    output: bool,
    /// Owns the PLE convolution history: the input scope of a PLE profile.
    ple: bool,
    layers: usize,
    /// Absolute layer of each local GDN / QSA slot, ascending.
    gdn_layers: Vec<usize>,
    qsa_layers: Vec<usize>,
    /// Local slot of each absolute layer (`None` = not owned here).
    gdn_slot: Vec<Option<usize>>,
    qsa_slot: Vec<Option<usize>>,
    /// GDN / QSA layers of the whole model before the band's first layer.
    gdn_base: usize,
    qsa_base: usize,
}

impl Qwen4StateMap {
    /// The map of `scope` over `config`'s trunk. Refuses a band that is empty
    /// or outside the trunk, input/output flags that disagree with the band
    /// ends, and a PLE profile whose zero-based PLE layer is not in the input
    /// band (the PLE history is input-owned, frozen review §3.1).
    pub fn new(config: &Qwen4Config, scope: &Qwen4BandScope) -> Result<Self, String> {
        scope.validate(config)?;
        let layers = config.layer_types.len();
        let band = scope.band.clone();
        if band.start >= band.end || band.end > layers {
            return Err(format!(
                "qwen4 state: band {}..{} is empty or outside the {layers} trunk layers",
                band.start, band.end
            ));
        }
        if scope.input != (band.start == 0) || scope.output != (band.end == layers) {
            return Err(format!(
                "qwen4 state: band {}..{} disagrees with input={} output={}",
                band.start, band.end, scope.input, scope.output
            ));
        }
        if let Some(ple) = &config.ple {
            if band.contains(&ple.layer) != scope.input {
                return Err(format!(
                    "qwen4 state: PLE layer {} and its convolution history belong to the input \
                     band; band {}..{} (input={}) does not own them consistently",
                    ple.layer, band.start, band.end, scope.input
                ));
            }
        }
        let mut gdn_layers = Vec::new();
        let mut qsa_layers = Vec::new();
        let mut gdn_slot = vec![None; layers];
        let mut qsa_slot = vec![None; layers];
        for abs in band.clone() {
            match config.layer_types[abs] {
                LayerType::LinearAttention => {
                    gdn_slot[abs] = Some(gdn_layers.len());
                    gdn_layers.push(abs);
                }
                LayerType::FullAttention => {
                    qsa_slot[abs] = Some(qsa_layers.len());
                    qsa_layers.push(abs);
                }
            }
        }
        let before = |wanted: LayerType| {
            config.layer_types[..band.start]
                .iter()
                .filter(|kind| **kind == wanted)
                .count()
        };
        let (gdn_base, qsa_base) = (
            before(LayerType::LinearAttention),
            before(LayerType::FullAttention),
        );
        Ok(Self {
            input: scope.input,
            output: scope.output,
            ple: scope.input && config.ple.is_some(),
            band,
            layers,
            gdn_layers,
            qsa_layers,
            gdn_slot,
            qsa_slot,
            gdn_base,
            qsa_base,
        })
    }

    /// The absolute trunk layers this state owns.
    pub fn band(&self) -> Range<usize> {
        self.band.clone()
    }

    /// Whole-model ownership (one band covering the trunk): today's layout.
    pub fn is_single(&self) -> bool {
        self.band == (0..self.layers)
    }

    pub fn owns_input(&self) -> bool {
        self.input
    }

    pub fn owns_output(&self) -> bool {
        self.output
    }

    /// PLE convolution history lives here (input scope of a PLE profile).
    pub fn owns_ple(&self) -> bool {
        self.ple
    }

    pub fn gdn_count(&self) -> usize {
        self.gdn_layers.len()
    }

    pub fn qsa_count(&self) -> usize {
        self.qsa_layers.len()
    }

    /// Absolute layer of each local GDN slot, ascending.
    pub fn gdn_layers(&self) -> &[usize] {
        &self.gdn_layers
    }

    /// Absolute layer of each local QSA slot, ascending.
    pub fn qsa_layers(&self) -> &[usize] {
        &self.qsa_layers
    }

    /// Local GDN slot of absolute layer `abs` (`None`: not an owned GDN layer).
    pub fn gdn_local(&self, abs: usize) -> Option<usize> {
        self.gdn_slot.get(abs).copied().flatten()
    }

    /// Local QSA slot of absolute layer `abs` (`None`: not an owned QSA layer).
    pub fn qsa_local(&self, abs: usize) -> Option<usize> {
        self.qsa_slot.get(abs).copied().flatten()
    }

    /// Absolute layer of local GDN slot `slot`.
    pub fn gdn_layer(&self, slot: usize) -> Option<usize> {
        self.gdn_layers.get(slot).copied()
    }

    /// Absolute layer of local QSA slot `slot`.
    pub fn qsa_layer(&self, slot: usize) -> Option<usize> {
        self.qsa_layers.get(slot).copied()
    }

    /// GDN layers of the whole model before this band: whole-model GDN ordinal
    /// of local slot `s` is `gdn_base() + s`.
    pub fn gdn_base(&self) -> usize {
        self.gdn_base
    }

    /// QSA layers of the whole model before this band.
    pub fn qsa_base(&self) -> usize {
        self.qsa_base
    }

    /// Identity of the ownership in a session snapshot's reserved meta word:
    /// 0 for the whole model (today's bytes), nonzero and band-specific
    /// otherwise, so a snapshot of one band never restores into another.
    pub(crate) fn session_tag(&self) -> u64 {
        if self.is_single() {
            return 0;
        }
        let mut hash = 0xcbf29ce484222325u64;
        for word in [
            self.band.start as u64,
            self.band.end as u64,
            u64::from(self.input),
            u64::from(self.output),
        ] {
            hash ^= word;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash.max(1)
    }
}

/// (GDN, QSA) layers of `config`'s trunk inside `scope.band`.
pub(crate) fn scope_layer_counts(config: &Qwen4Config, scope: &Qwen4BandScope) -> (usize, usize) {
    let mut counts = (0, 0);
    for (abs, kind) in config.layer_types.iter().enumerate() {
        if scope.band.contains(&abs) {
            match kind {
                LayerType::LinearAttention => counts.0 += 1,
                LayerType::FullAttention => counts.1 += 1,
            }
        }
    }
    counts
}

/// Shape constants every state allocation derives from `config`/`format`.
struct StateGeometry {
    gdn_recurrent: usize,
    gdn_residual: usize,
    conv_len: usize,
    full_width: usize,
    full_row_units: usize,
    raw_width: usize,
    pooled_capacity: usize,
    selected_capacity: usize,
}

impl StateGeometry {
    fn new(
        config: &Qwen4Config,
        format: &Qwen4StateFormat,
        max_seq_len: usize,
    ) -> Result<Self, StateError> {
        let state_elements = config
            .linear_num_value_heads
            .checked_mul(config.linear_key_head_dim)
            .and_then(|n| n.checked_mul(config.linear_value_head_dim))
            .ok_or(StateError::DimensionOverflow)?;
        let storage = format.gdn.storage();
        let gdn_recurrent = match storage {
            GdnStateFormat::F32 => state_elements,
            GdnStateFormat::Q8 => state_elements
                .checked_add(
                    config
                        .linear_num_value_heads
                        .checked_mul(config.linear_value_head_dim)
                        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                        .ok_or(StateError::DimensionOverflow)?,
                )
                .ok_or(StateError::DimensionOverflow)?,
            GdnStateFormat::Q8Ef => unreachable!("storage excludes live EF mode"),
        };
        let gdn_residual = format.gdn.residual_units(
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        );
        gdn_residual
            .checked_mul(DType::F16.size())
            .ok_or(StateError::DimensionOverflow)?;
        let conv_channels = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| {
                v.checked_add(
                    config
                        .linear_num_value_heads
                        .checked_mul(config.linear_value_head_dim)?,
                )
            })
            .ok_or(StateError::DimensionOverflow)?;
        let conv_len = conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))
            .ok_or(StateError::DimensionOverflow)?;
        let full_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or(StateError::DimensionOverflow)?;
        let full_row_units = format
            .qsa
            .kv_row_units(config.num_key_value_heads, config.head_dim);
        let raw_width = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)
            .ok_or(StateError::DimensionOverflow)?;
        let pooled_capacity = max_seq_len
            .checked_add(config.indexer_compress_ratio - 1)
            .ok_or(StateError::DimensionOverflow)?
            / config.indexer_compress_ratio;
        Ok(Self {
            gdn_recurrent,
            gdn_residual,
            conv_len,
            full_width,
            full_row_units,
            raw_width,
            pooled_capacity,
            selected_capacity: config.qsa_selected_capacity(),
        })
    }
}

/// One device tensor `Qwen4State` allocates: its absolute-layer key, element
/// count, dtype and whether it is a QSA context arena (VMM-owned under the
/// VMM backend).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StateTensorSpec {
    pub(crate) key: String,
    pub(crate) elements: usize,
    pub(crate) dtype: DType,
    pub(crate) context_arena: bool,
}

impl StateTensorSpec {
    fn new(key: String, elements: usize, dtype: DType, context_arena: bool) -> Self {
        Self {
            key,
            elements,
            dtype,
            context_arena,
        }
    }
}

/// Allocation order of a state over `map`: owned layers in absolute order (a
/// GDN layer's recurrent, optional EF residual, then conv; a QSA layer's seven fixed/context tensors,
/// then the F16 arena K/V, then the pooled BF16 mirror), then, on the input
/// scope only, the PLE convolution history (PLE profiles) and
/// `hyper_feedback`. Under the whole-model map this is exactly the order the
/// constructor always allocated.
fn state_tensor_plan(
    config: &Qwen4Config,
    format: &Qwen4StateFormat,
    max_seq_len: usize,
    geometry: &StateGeometry,
    map: &Qwen4StateMap,
) -> Result<Vec<StateTensorSpec>, StateError> {
    let layout = format.layout();
    let qsa_format = format.qsa;
    let ratio = config.indexer_compress_ratio;
    let mut plan = Vec::new();
    for abs in map.band() {
        match config.layer_types[abs] {
            LayerType::LinearAttention => {
                plan.push(StateTensorSpec::new(
                    format!("gdn[L{abs}].recurrent"),
                    geometry.gdn_recurrent,
                    format.gdn.storage().dtype(),
                    false,
                ));
                if format.gdn.has_residual() {
                    plan.push(StateTensorSpec::new(
                        format!("gdn[L{abs}].ef_residual"),
                        geometry.gdn_residual,
                        DType::F16,
                        false,
                    ));
                }
                plan.push(StateTensorSpec::new(
                    format!("gdn[L{abs}].conv"),
                    geometry.conv_len,
                    DType::F32,
                    false,
                ));
            }
            LayerType::FullAttention => {
                let full_elements = max_seq_len
                    .checked_mul(geometry.full_row_units)
                    .ok_or(StateError::DimensionOverflow)?;
                let raw_elements = max_seq_len
                    .checked_mul(geometry.raw_width)
                    .ok_or(StateError::DimensionOverflow)?;
                let pooled_elements = geometry
                    .pooled_capacity
                    .checked_mul(geometry.raw_width)
                    .ok_or(StateError::DimensionOverflow)?;
                let partial_key_elements = ratio
                    .checked_mul(geometry.raw_width)
                    .ok_or(StateError::DimensionOverflow)?;
                let partial_value_elements = ratio
                    .checked_mul(geometry.full_width)
                    .ok_or(StateError::DimensionOverflow)?;
                let selected_bytes = geometry
                    .selected_capacity
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(StateError::DimensionOverflow)?;
                // The derived F16 K and block-transposed V arenas: whole
                // four-token blocks, `full_width` F16 channels per token.
                let arena_elements = if layout.f16_arena {
                    qsa_f16_arena_rows(max_seq_len)
                        .checked_mul(geometry.full_width)
                        .ok_or(StateError::DimensionOverflow)?
                } else {
                    0
                };
                let mut specs = vec![
                    ("full_keys", full_elements, qsa_format.kv_dtype(), true),
                    ("full_values", full_elements, qsa_format.kv_dtype(), true),
                    ("raw_index_keys", raw_elements, layout.index_dtype(), true),
                    ("pooled_keys", pooled_elements, layout.index_dtype(), true),
                    ("partial_keys", partial_key_elements, DType::F32, false),
                    ("partial_values", partial_value_elements, DType::F32, false),
                    ("selected_indices", selected_bytes, DType::Raw, false),
                ];
                if layout.f16_arena {
                    specs.push(("f16_keys", arena_elements, DType::F16, true));
                    specs.push(("f16_values", arena_elements, DType::F16, true));
                }
                // The derived BF16 pooled-key mirror (same row count and
                // width as `pooled_keys`).
                if layout.pooled_bf16 {
                    specs.push(("pooled_bf16", pooled_elements, DType::BF16, true));
                }
                for (name, elements, dtype, context_arena) in specs {
                    plan.push(StateTensorSpec::new(
                        format!("qsa[L{abs}].{name}"),
                        elements,
                        dtype,
                        context_arena,
                    ));
                }
            }
        }
    }
    // PLE channels are the HC-wide rows of the key projection:
    // `hidden_size * hc_count` (`ple_embed_dim` is its input width). A
    // profile without PLE has no convolution history to hold.
    if map.owns_ple() {
        let ple_channels = config
            .hidden_size
            .checked_mul(config.hc_count)
            .ok_or(StateError::DimensionOverflow)?;
        let ple_elements = config
            .ple_conv_history_rows()
            .checked_mul(ple_channels)
            .ok_or(StateError::DimensionOverflow)?;
        plan.push(StateTensorSpec::new(
            "ple_conv".into(),
            ple_elements,
            DType::F32,
            false,
        ));
    }
    if map.owns_input() {
        let feedback_elements = config
            .hc_count
            .checked_mul(config.hidden_size)
            .ok_or(StateError::DimensionOverflow)?;
        plan.push(StateTensorSpec::new(
            "hyper_feedback".into(),
            feedback_elements,
            DType::F32,
            false,
        ));
    }
    Ok(plan)
}

/// Bytes one QSA context arena of `spec` commits under `context`: its whole
/// allocation under legacy storage, else the growth pages the shared KV chunk
/// plan maps for the committed tokens (pooled arenas by
/// `ceil(tokens / compress)` rows, F16 arenas by whole four-token blocks), as
/// `ensure_mapped_capacity` maps them.
fn committed_context_arena_bytes(
    spec: &StateTensorSpec,
    context: &Qwen4ContextCommit,
    geometry: &StateGeometry,
    ratio: usize,
) -> Option<usize> {
    let bytes = spec.elements.checked_mul(spec.dtype.size())?;
    if context.backend == Qwen4KvBackend::Legacy {
        return Some(bytes);
    }
    let (rows, tokens) = if spec.key.ends_with(".pooled_keys") || spec.key.ends_with(".pooled_bf16")
    {
        (geometry.pooled_capacity, context.tokens.div_ceil(ratio))
    } else if spec.key.ends_with(".f16_keys") || spec.key.ends_with(".f16_values") {
        (
            qsa_f16_arena_rows(context.max_seq),
            qsa_f16_arena_rows(context.tokens),
        )
    } else {
        (context.max_seq, context.tokens)
    };
    if rows == 0 || !bytes.is_multiple_of(rows) {
        return None;
    }
    vmm_context_arena_mapped_bytes(bytes / rows, rows, tokens, context.granularity)
}

/// Bytes the shared KV chunk plan maps for the first `tokens` rows of a
/// VMM context arena of `rows` rows of `row_bytes` bytes, in driver pages of
/// `granularity` (what `ensure_mapped_capacity` commits).
pub(crate) fn vmm_context_arena_mapped_bytes(
    row_bytes: usize,
    rows: usize,
    tokens: usize,
    granularity: usize,
) -> Option<usize> {
    context_arena_plan(row_bytes, rows, granularity)
        .ok()?
        .mapped_bytes_for_tokens(tokens)
        .ok()
}

/// `HIPFIRE_QWEN4_CAPTURE_RING_COMPACT` (developer, default ON, exact `0` is
/// the kill switch back to the double ring): lay the GDN
/// verify capture ring out as `rows + 1` state slots per layer instead of
/// `2 * rows`. The ring is only ever written at ONE slot per half (the
/// capture kernel and the rollback kernel both leave the last row's state in
/// slot `rows - 1` of the view they are handed), so the other `rows - 1`
/// slots of each half were never touched. The logical slot numbering
/// (`gdn_live`, halves, digests) is unchanged; [`capture_ring_slot`] maps it
/// onto the compact storage.
pub(crate) const CAPTURE_RING_COMPACT_ENV: &str = "HIPFIRE_QWEN4_CAPTURE_RING_COMPACT";

/// Whether new capture rings use the compact layout (unset or `1`; `0` off).
pub(crate) fn compact_capture_ring() -> bool {
    hipfire_config::developer_bool(CAPTURE_RING_COMPACT_ENV, true)
}

/// State slots one layer's capture ring of `rows` rows holds.
pub(crate) fn capture_ring_slots(compact: bool, rows: usize) -> Option<usize> {
    if compact {
        rows.checked_add(1)
    } else {
        rows.checked_mul(2)
    }
}

/// Physical ring slot of logical slot `logical` (`0..2 * rows`). The lower
/// logical half (`< rows`) lands in physical slot `rows - 1`, the upper half
/// in `rows`: exactly the slots a capture or rollback of the half writes
/// when its view starts at [`capture_view_start`].
pub(crate) fn capture_ring_slot(compact: bool, rows: usize, logical: usize) -> usize {
    if !compact {
        logical
    } else if logical < rows {
        rows - 1
    } else {
        rows
    }
}

/// First physical slot of the `n`-row view a capture or rollback into the
/// half starting at logical slot `base` hands the kernel: the kernel leaves
/// the result in view slot `n - 1`, which must be the half's physical slot.
pub(crate) fn capture_view_start(compact: bool, rows: usize, base: usize, n: usize) -> usize {
    if !compact {
        base
    } else {
        capture_ring_slot(true, rows, base) + 1 - n
    }
}

/// `HIPFIRE_QWEN4_LANE_RESERVE_TRIM` (developer, default ON, exact `0` is the
/// kill switch back to one ring allocation per GDN layer): the compact
/// capture ring of every GDN layer is a window of ONE shared pool allocation
/// ([`capture_ring_pool_slots`]) instead of `rows + 1` slots per layer.
///
/// Why this is the same bytes, kernels and row math. A layer's ring is only
/// ever touched at its two write targets, physical slots `rows - 1` and
/// `rows` of its window ([`capture_ring_slot`]): the capture kernels
/// (`gated_delta_step_persistent256_body<kCapture>` and its `_lanes`
/// variant) read only the live state slot and write only view slot `n - 1`
/// (`(rows - 1) * slot`, `kernels/src/tensor_ops.hip`), the rollback kernels
/// (`gated_delta_rollback_layers_*`) read slot `from` (a target) and write
/// slot `to + keep - 1` (a target), and `gdn_rows_lane_ranges` already treats
/// that one slot as the only ring range a lane chain writes. The other
/// `rows - 1` slots of the window exist only so the `n`-row views the kernels
/// are handed (`capture_view_start`) stay inside the allocation; no kernel
/// reads or writes them. Layer `L`'s window therefore starts
/// [`CAPTURE_POOL_STRIDE`] `* L` slots into the pool: its two targets
/// (`2 L + rows - 1`, `2 L + rows`) are distinct from every other layer's
/// (`2 L' + rows - 1`, `2 L' + rows`), and the windows overlap only in slots
/// nobody touches. The kernels, the pointer table, the logical slot numbering
/// (`gdn_live`, halves) and every view are unchanged; only the allocation
/// behind the window pointers is shared. Requires the compact layout (the
/// double ring writes every slot of its half).
pub(crate) fn lane_reserve_trim() -> bool {
    hipfire_config::qwen4_lane_reserve_trim()
}

/// Slots between the windows of two consecutive GDN layers in the shared
/// capture-ring pool: the layer's two write targets, nothing else.
pub(crate) const CAPTURE_POOL_STRIDE: usize = 2;

/// State slots of the shared capture-ring pool of `gdn_layers` layers holding
/// `rows`-row blocks: the last layer's window ends the pool, so
/// `CAPTURE_POOL_STRIDE * (gdn_layers - 1) + rows + 1`. `None` for no layer
/// or on overflow.
pub(crate) fn capture_ring_pool_slots(rows: usize, gdn_layers: usize) -> Option<usize> {
    gdn_layers
        .checked_sub(1)?
        .checked_mul(CAPTURE_POOL_STRIDE)?
        .checked_add(rows.checked_add(1)?)
}

/// Ring state slots a whole state allocates across its `gdn_layers` GDN
/// layers for `rows`-row blocks: the shared pool when `compact` and `trim`,
/// else `capture_ring_slots` per layer. `None` on overflow.
pub(crate) fn capture_ring_total_slots(
    compact: bool,
    trim: bool,
    rows: usize,
    gdn_layers: usize,
) -> Option<usize> {
    if gdn_layers == 0 {
        Some(0)
    } else if compact && trim {
        capture_ring_pool_slots(rows, gdn_layers)
    } else {
        gdn_layers.checked_mul(capture_ring_slots(compact, rows)?)
    }
}

/// [`capture_ring_total_slots`] under the process configuration.
pub(crate) fn current_capture_ring_total_slots(rows: usize, gdn_layers: usize) -> Option<usize> {
    capture_ring_total_slots(
        compact_capture_ring(),
        lane_reserve_trim(),
        rows,
        gdn_layers,
    )
}

/// The snapshot-arena mirror allocation (elements, dtype) of one state
/// tensor, `None` for state a snapshot does not copy (QSA full/pooled/F16
/// arenas, the live EF residual). Mirrors [`Qwen4StateSnapshotArena::new`]:
/// GDN recurrent and conv, QSA partial K/V, the raw-index circular block of
/// `compress_ratio` rows, selected indices, PLE conv and hyper feedback.
fn snapshot_mirror(
    spec: &StateTensorSpec,
    geometry: &StateGeometry,
    ratio: usize,
) -> Option<(usize, DType)> {
    let key = spec.key.as_str();
    if key.ends_with(".recurrent")
        || key.ends_with(".conv")
        || key.ends_with(".partial_keys")
        || key.ends_with(".partial_values")
        || key.ends_with(".selected_indices")
        || key == "ple_conv"
        || key == "hyper_feedback"
    {
        Some((spec.elements, spec.dtype))
    } else if key.ends_with(".raw_index_keys") {
        Some((ratio.checked_mul(geometry.raw_width)?, spec.dtype))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QsaDigestField {
    FullKeys,
    FullValues,
    RawIndexKeys,
    PooledKeys,
    PartialKeys,
    PartialValues,
    SelectedIndices,
}

impl QsaDigestField {
    const ALL: [Self; 7] = [
        Self::FullKeys,
        Self::FullValues,
        Self::RawIndexKeys,
        Self::PooledKeys,
        Self::PartialKeys,
        Self::PartialValues,
        Self::SelectedIndices,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::FullKeys => "full_keys",
            Self::FullValues => "full_values",
            Self::RawIndexKeys => "raw_index_keys",
            Self::PooledKeys => "pooled_keys",
            Self::PartialKeys => "partial_keys",
            Self::PartialValues => "partial_values",
            Self::SelectedIndices => "selected_indices",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureDigestField {
    Ring,
    Inputs,
    Recurrence,
}

impl CaptureDigestField {
    const ALL: [Self; 3] = [Self::Ring, Self::Inputs, Self::Recurrence];

    fn name(self) -> &'static str {
        match self {
            Self::Ring => "ring",
            Self::Inputs => "inputs",
            Self::Recurrence => "recurrence",
        }
    }
}

/// One row of a state digest; slots are local indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DigestRow {
    Position,
    /// Canonical only: the host position, once per owned absolute layer.
    LayerPosition(usize),
    PleHistoryPrevious,
    PleHistoryEos,
    GdnLive,
    /// Canonical only: the GDN ring slot, once per owned GDN slot.
    GdnLiveLayer(usize),
    RowCaptureRows,
    QsaMappedTokens,
    QsaMarks(usize),
    GdnRecurrent(usize),
    GdnResidual(usize),
    GdnConv(usize),
    Qsa(usize, QsaDigestField),
    PleConv,
    HyperFeedback,
    RowCapture(usize, CaptureDigestField),
    RowCaptureOutput,
}

/// The ordered rows of a digest over `map`. The non-canonical plan is
/// [`Qwen4State::debug_digest`]: under the whole-model map it is exactly the
/// historical row order. The canonical plan ([`Qwen4State::canonical_digest_rows`])
/// drops the capacity rows (`row_capture_rows`,
/// `qsa_mapped_tokens`) and the rollback re-run's discarded `row_capture_output`
/// scratch (written fully by the re-run launch, never read by any forward or
/// repair), and replaces the two global host rows by per-layer rows so every
/// canonical key has exactly one owner: `layer[L<abs>].position` for each
/// owned trunk layer and `gdn[L<abs>].live` for each owned GDN layer (PLE
/// history stays on the input scope, which owns it once).
fn digest_plan(
    map: &Qwen4StateMap,
    canonical: bool,
    row_capture: bool,
    row_capture_output: bool,
) -> Vec<DigestRow> {
    let mut plan = if canonical {
        map.band().map(DigestRow::LayerPosition).collect()
    } else {
        vec![DigestRow::Position]
    };
    if map.owns_input() {
        plan.push(DigestRow::PleHistoryPrevious);
        plan.push(DigestRow::PleHistoryEos);
    }
    if canonical {
        plan.extend((0..map.gdn_count()).map(DigestRow::GdnLiveLayer));
    } else {
        plan.push(DigestRow::GdnLive);
    }
    if !canonical {
        plan.push(DigestRow::RowCaptureRows);
        plan.push(DigestRow::QsaMappedTokens);
    }
    plan.extend((0..map.qsa_count()).map(DigestRow::QsaMarks));
    for slot in 0..map.gdn_count() {
        plan.push(DigestRow::GdnRecurrent(slot));
        plan.push(DigestRow::GdnConv(slot));
    }
    for slot in 0..map.qsa_count() {
        plan.extend(
            QsaDigestField::ALL
                .into_iter()
                .map(|field| DigestRow::Qsa(slot, field)),
        );
    }
    if map.owns_ple() {
        plan.push(DigestRow::PleConv);
    }
    if map.owns_input() {
        plan.push(DigestRow::HyperFeedback);
    }
    if row_capture {
        for slot in 0..map.gdn_count() {
            plan.extend(
                CaptureDigestField::ALL
                    .into_iter()
                    .map(|field| DigestRow::RowCapture(slot, field)),
            );
        }
    }
    if !canonical && row_capture_output {
        plan.push(DigestRow::RowCaptureOutput);
    }
    plan
}

/// The key of `row`: local-slot labels (`qsa[2]`) for the debug digest,
/// absolute-layer labels (`qsa[L11]`) for the canonical projection.
fn digest_key(map: &Qwen4StateMap, row: DigestRow, canonical: bool) -> String {
    let gdn = |slot: usize| {
        if canonical {
            format!("gdn[L{}]", map.gdn_layers[slot])
        } else {
            format!("gdn[{slot}]")
        }
    };
    let qsa = |slot: usize| {
        if canonical {
            format!("qsa[L{}]", map.qsa_layers[slot])
        } else {
            format!("qsa[{slot}]")
        }
    };
    match row {
        DigestRow::Position => "position".into(),
        DigestRow::LayerPosition(abs) => format!("layer[L{abs}].position"),
        DigestRow::PleHistoryPrevious => "ple_history.previous".into(),
        DigestRow::PleHistoryEos => "ple_history.eos".into(),
        DigestRow::GdnLive => "gdn_live".into(),
        DigestRow::GdnLiveLayer(slot) => format!("gdn[L{}].live", map.gdn_layers[slot]),
        DigestRow::RowCaptureRows => "row_capture_rows".into(),
        DigestRow::QsaMappedTokens => "qsa_mapped_tokens".into(),
        DigestRow::QsaMarks(slot) => format!("{}.marks", qsa(slot)),
        DigestRow::GdnRecurrent(slot) => format!("{}.recurrent", gdn(slot)),
        DigestRow::GdnResidual(slot) => format!("{}.ef_residual", gdn(slot)),
        DigestRow::GdnConv(slot) => format!("{}.conv", gdn(slot)),
        DigestRow::Qsa(slot, field) => format!("{}.{}", qsa(slot), field.name()),
        DigestRow::PleConv => "ple_conv".into(),
        DigestRow::HyperFeedback => "hyper_feedback".into(),
        DigestRow::RowCapture(slot, field) => {
            let layer = if canonical {
                format!("row_capture[L{}]", map.gdn_layers[slot])
            } else {
                format!("row_capture[{slot}]")
            };
            format!("{layer}.{}", field.name())
        }
        DigestRow::RowCaptureOutput => "row_capture_output".into(),
    }
}

/// The sole production request-state owner for Qwen4.
pub struct Qwen4State {
    pub gdn: Vec<GdnGpuState>,
    pub qsa: Vec<QsaGpuState>,
    /// The PLE short-convolution history; `None` exactly when the profile
    /// declares no PLE (`config.ple`) or the scope is not the input band.
    /// `hyper_feedback` below is not PLE state (its width is
    /// `hc_count * hidden_size` whatever the profile, and no PLE step reads or
    /// writes it), so it is allocated for every profile on the input scope
    /// and the Flash-Next layout does not depend on this option.
    pub ple_conv: Option<GpuTensor>,
    /// Legacy HC feedback tensor: owned by the input scope only (`None` on
    /// every other band); always present under the whole-model scope.
    pub hyper_feedback: Option<GpuTensor>,
    pub ple_history: PleHistory,
    pub position: usize,
    pub max_seq_len: usize,
    pub qsa_selected_capacity: usize,
    snapshot_arena: Qwen4StateSnapshotArena,
    /// Few-row verify rollback points per GDN layer: (recurrent-state ring,
    /// convolution input rows, recurrence inputs); see `GdnRowCapture`. With
    /// a ring, each GDN layer's `recurrent` is a view of its live slot: a
    /// verify writes its last row's state into the half of the ring the live
    /// slot is not in, so the pre-verify state is never overwritten;
    /// committing only moves the live slot, and rolling back re-runs the kept
    /// rows' recurrence from the pre-verify state into that half.
    row_capture: Vec<(GpuTensor, GpuTensor, GpuTensor)>,
    /// Discarded recurrence output of a rollback re-run (`rows` x the widest
    /// GDN value width) and the per-layer (recurrence, ring) device pointer
    /// table the one-launch re-run reads.
    row_capture_output: Option<(GpuTensor, GpuTensor)>,
    /// The one allocation the per-layer capture-ring windows of `row_capture`
    /// are views of ([`lane_reserve_trim`], [`capture_ring_pool_slots`]);
    /// `None`: every layer's ring in `row_capture` is its own allocation.
    row_capture_pool: Option<GpuTensor>,
    /// Rows a verify may capture (0 = none allocated); the ring holds twice
    /// as many slots.
    row_capture_rows: usize,
    /// The ring was allocated in the compact `rows + 1`-slot layout
    /// ([`capture_ring_slots`]); fixed per allocation.
    row_capture_compact: bool,
    /// Live ring slot of every GDN layer's recurrent state.
    gdn_live: usize,
    /// Storage of every GDN layer's recurrent state (and its ring slots).
    gdn_format: GdnStateFormat,
    gdn_value_heads: usize,
    /// Successful full-owner EF reset dispatch sequences, never serialized.
    gdn_residual_resets: u64,
    /// Set only around a speculative verify forward.
    pub(crate) row_capture_armed: bool,
    /// Lab observation of GDN residual resets and capture-ring events;
    /// unarmed it records nothing (`crate::lab_window`).
    #[cfg(feature = "lab")]
    lab: crate::lab_window::LabStateTap,
    /// Storage of the QSA context arenas (full K/V, raw and pooled keys).
    qsa_backend: Qwen4KvBackend,
    /// Tokens every QSA context arena's accessible prefix covers: the
    /// minimum over all owners, so the no-growth gate never scans layers.
    /// Legacy arenas cover `max_seq_len` from construction.
    qsa_mapped_tokens: usize,
    /// Raw tokens per pooled index row (`indexer_compress_ratio`).
    qsa_compress: usize,
    model_id: u64,
    reset_epoch: u64,
    /// Absolute-to-local ownership of `gdn` / `qsa` (compact local vectors).
    map: Qwen4StateMap,
    transaction_generation: u64,
}

impl Qwen4State {
    /// Allocate all mutable device buffers once, the QSA context arenas as
    /// legacy full-capacity allocations; see [`Self::new_with_backend`].
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
    ) -> Result<Self, StateError> {
        Self::new_with_backend(gpu, config, max_seq_len, format, Qwen4KvBackend::Legacy)
    }

    /// [`Self::new`] for one band of a pipeline-parallel mesh; see
    /// [`Self::new_with_backend_scoped`].
    pub fn new_scoped(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
        scope: &Qwen4BandScope,
    ) -> Result<Self, StateError> {
        Self::new_with_backend_scoped(
            gpu,
            config,
            max_seq_len,
            format,
            Qwen4KvBackend::Legacy,
            scope,
        )
    }

    /// Allocate all mutable device buffers once.  QSA full K/V, raw and
    /// pooled key arenas have the `max_seq_len` capacity shape in
    /// `format.qsa`, the GDN recurrent states `format.gdn`; only active
    /// lengths are mutable. Under [`Qwen4KvBackend::Vmm`] the four context
    /// arenas are VMM owners with nothing mapped: callers must
    /// [`Self::ensure_mapped_capacity`] before any row access. `backend` is
    /// load admission's choice; a VMM failure is an error, never a legacy
    /// fallback. The whole model: [`Qwen4BandScope::single`].
    pub(crate) fn new_with_backend(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
    ) -> Result<Self, StateError> {
        Self::new_with_backend_scoped(
            gpu,
            config,
            max_seq_len,
            format,
            backend,
            &Qwen4BandScope::single(config),
        )
    }

    /// [`Self::new_with_backend`] over `scope`: GDN/QSA state is allocated
    /// for the band's layers only (compact local vectors, absolute layer ids
    /// in [`Self::scope_map`]); the PLE convolution history and
    /// `hyper_feedback` only on the input scope. Under the single scope the
    /// allocations, their order and their bytes are the historical ones. A
    /// snapshot-capable state ([`StateMirror::Snapshot`]): the resident
    /// request state and every state that rolls a window back.
    pub(crate) fn new_with_backend_scoped(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
        scope: &Qwen4BandScope,
    ) -> Result<Self, StateError> {
        Self::new_with_mirror(
            gpu,
            config,
            max_seq_len,
            format,
            backend,
            scope,
            StateMirror::Snapshot,
        )
    }

    /// [`Self::new_with_backend_scoped`] with an explicit snapshot `mirror`.
    /// [`StateMirror::Absent`] allocates the live state alone: no mirror of
    /// the in-place parts is made, now or lazily, and a snapshot of the
    /// state is the typed [`StateError::MirrorAbsent`]. For states that are
    /// only stepped and reset (AR lanes).
    pub(crate) fn new_with_mirror(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
        scope: &Qwen4BandScope,
        mirror: StateMirror,
    ) -> Result<Self, StateError> {
        config.validate().map_err(StateError::Config)?;
        let map = Qwen4StateMap::new(config, scope).map_err(StateError::Config)?;
        let qsa_format = format.qsa;
        if !qsa_format.supports(config.num_key_value_heads, config.head_dim) {
            return Err(StateError::Config(format!(
                "QSA {} K/V does not support {} KV heads x {}",
                qsa_format.name(),
                config.num_key_value_heads,
                config.head_dim
            )));
        }
        let layout = format.layout();
        if format.qsa_f16_arena && qsa_format != QsaKvFormat::F32 {
            return Err(StateError::Config(format!(
                "QSA F16 K/V arena needs exact F32 K/V, not {}",
                qsa_format.name()
            )));
        }
        if format.qsa_f16_arena && (config.head_dim != 256 || !gpu.arch_caps.is_gfx1151()) {
            return Err(StateError::Config(format!(
                "QSA F16 K/V arena needs gfx1151 and head_dim 256, got {} and {}",
                gpu.arch, config.head_dim
            )));
        }
        if max_seq_len == 0 || max_seq_len > config.context_limit() {
            return Err(StateError::InvalidCapacity);
        }
        let gdn_format = format.gdn;
        validate_gdn_profile(config, gdn_format)?;
        if !gdn_format.supports(config.linear_key_head_dim, config.linear_value_head_dim) {
            return Err(StateError::Config(format!(
                "GDN {} state needs 128x128 heads, not {} x {}",
                gdn_format.name(),
                config.linear_key_head_dim,
                config.linear_value_head_dim
            )));
        }
        let geometry = StateGeometry::new(config, &format, max_seq_len)?;
        if format.qsa_pooled_bf16
            && (qsa_format != QsaKvFormat::F32
                || geometry.raw_width != 128
                || !gpu.arch_caps.is_gfx1151())
        {
            return Err(StateError::Config(format!(
                "QSA pooled BF16 mirror needs exact F32 K/V on gfx1151 with index_dim 128, got {} on {} with {}",
                qsa_format.name(),
                gpu.arch,
                geometry.raw_width
            )));
        }
        let plan = state_tensor_plan(config, &format, max_seq_len, &geometry, &map)?;
        let selected_capacity = geometry.selected_capacity;
        let pooled_capacity = geometry.pooled_capacity;
        let full_row_units = geometry.full_row_units;
        let device = gpu.device_id;
        let mut allocated = Vec::with_capacity(plan.len());
        let allocation = (|| -> Result<(), StateError> {
            for spec in &plan {
                let tensor = if spec.context_arena && backend == Qwen4KvBackend::Vmm {
                    // SAFETY: nothing is mapped; every access is
                    // bounded by `ensure_mapped_capacity`'s prefix
                    // (`buf.size()` reports only mapped bytes).
                    unsafe { gpu.alloc_vmm_tensor(&[spec.elements], spec.dtype, 0, &[device]) }
                } else {
                    gpu.zeros(&[spec.elements], spec.dtype)
                };
                allocated.push(tensor.map_err(StateError::Hip)?);
            }
            Ok(())
        })();
        if let Err(error) = allocation {
            for tensor in allocated.into_iter().rev() {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }

        let expected_tensors = map
            .gdn_count()
            .checked_mul(2 + usize::from(format.gdn.has_residual()))
            .and_then(|gdn| {
                map.qsa_count()
                    .checked_mul(
                        7 + 2 * usize::from(layout.f16_arena) + usize::from(layout.pooled_bf16),
                    )
                    .and_then(|qsa| gdn.checked_add(qsa))
            })
            .and_then(|layers| {
                layers.checked_add(usize::from(map.owns_ple()) + usize::from(map.owns_input()))
            })
            .ok_or(StateError::DimensionOverflow)?;
        if allocated.len() != expected_tensors {
            for tensor in allocated.into_iter().rev() {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(StateError::AllocationBookkeeping);
        }

        let mut tensors = allocated.into_iter();
        let mut next_tensor = || {
            tensors
                .next()
                .expect("Qwen4 state tensor count checked before reconstruction")
        };
        let mut gdn = Vec::with_capacity(map.gdn_count());
        let mut qsa = Vec::with_capacity(map.qsa_count());
        for abs in map.band() {
            match config.layer_types[abs] {
                LayerType::LinearAttention => {
                    let recurrent = next_tensor();
                    let ef_residual = format.gdn.has_residual().then(|| next_tensor());
                    let conv = next_tensor();
                    gdn.push(GdnGpuState {
                        recurrent,
                        ef_residual,
                        conv,
                    });
                }
                LayerType::FullAttention => {
                    let full_keys = next_tensor();
                    let full_values = next_tensor();
                    let raw_index_keys = next_tensor();
                    let pooled_keys = next_tensor();
                    let partial_keys = next_tensor();
                    let partial_values = next_tensor();
                    let selected_indices = next_tensor();
                    let f16_arena = layout.f16_arena.then(|| QsaF16Arena {
                        keys: next_tensor(),
                        values: next_tensor(),
                        valid_len: QsaF16ValidLen::default(),
                    });
                    let pooled_bf16 = layout.pooled_bf16.then(|| QsaPooledBf16Arena {
                        mirror: next_tensor(),
                        valid_blocks: QsaPooledBf16ValidBlocks::default(),
                    });
                    qsa.push(QsaGpuState {
                        format: qsa_format,
                        full_row_units,
                        full_keys,
                        full_values,
                        raw_index_keys,
                        pooled_keys,
                        partial_keys,
                        partial_values,
                        selected_indices,
                        f16_arena,
                        pooled_bf16,
                        full_capacity: max_seq_len,
                        raw_capacity: max_seq_len,
                        pooled_capacity,
                        partial_capacity: config.indexer_compress_ratio,
                        selected_capacity,
                        position_capacity: max_seq_len,
                        full_len: 0,
                        raw_len: 0,
                        pooled_len: 0,
                        partial_len: 0,
                        selected_len: 0,
                        position: 0,
                    });
                }
            }
        }
        let ple_conv = map.owns_ple().then(|| next_tensor());
        let hyper_feedback = map.owns_input().then(|| next_tensor());
        let model_id = next_qwen4_model_id();
        let ple_history = PleHistory::new(config.eos_token_id);
        let snapshot_arena = match (|| {
            for layer in &gdn {
                if let Some(residual) = &layer.ef_residual {
                    gdn_q8ef_residual_zero(gpu, residual).map_err(StateError::Hip)?;
                }
            }
            match mirror {
                StateMirror::Snapshot => Qwen4StateSnapshotArena::new(
                    gpu,
                    &gdn,
                    &qsa,
                    ple_conv.as_ref(),
                    hyper_feedback.as_ref(),
                    model_id,
                    ple_history,
                    0,
                ),
                StateMirror::Absent => Ok(Qwen4StateSnapshotArena::absent(model_id, ple_history)),
            }
        })() {
            Ok(arena) => arena,
            Err(error) => {
                // Reverse of allocation: feedback, PLE, then layers last to first.
                if let Some(hyper_feedback) = hyper_feedback {
                    let _ = gpu.free_tensor(hyper_feedback);
                }
                if let Some(ple_conv) = ple_conv {
                    let _ = gpu.free_tensor(ple_conv);
                }
                let mut owned = Vec::new();
                for abs in map.band() {
                    match config.layer_types[abs] {
                        LayerType::LinearAttention => {
                            let layer = gdn.remove(0);
                            owned.push(layer.recurrent);
                            owned.extend(layer.ef_residual);
                            owned.push(layer.conv);
                        }
                        LayerType::FullAttention => {
                            let layer = qsa.remove(0);
                            owned.extend([
                                layer.full_keys,
                                layer.full_values,
                                layer.raw_index_keys,
                                layer.pooled_keys,
                                layer.partial_keys,
                                layer.partial_values,
                                layer.selected_indices,
                            ]);
                            owned.extend(
                                layer.f16_arena.into_iter().flat_map(|a| [a.keys, a.values]),
                            );
                            owned.extend(layer.pooled_bf16.into_iter().map(|a| a.mirror));
                        }
                    }
                }
                for tensor in owned.into_iter().rev() {
                    let _ = gpu.free_tensor(tensor);
                }
                return Err(error);
            }
        };
        Ok(Self {
            gdn,
            qsa,
            ple_conv,
            hyper_feedback,
            ple_history,
            position: 0,
            max_seq_len,
            qsa_selected_capacity: selected_capacity,
            snapshot_arena,
            row_capture: Vec::new(),
            row_capture_output: None,
            row_capture_pool: None,
            row_capture_rows: 0,
            row_capture_compact: false,
            gdn_live: 0,
            gdn_format: format.gdn,
            gdn_value_heads: config.linear_num_value_heads,
            gdn_residual_resets: u64::from(format.gdn.has_residual() && map.gdn_count() != 0),
            row_capture_armed: false,
            #[cfg(feature = "lab")]
            lab: crate::lab_window::LabStateTap::default(),
            qsa_backend: backend,
            qsa_mapped_tokens: match backend {
                Qwen4KvBackend::Legacy => max_seq_len,
                Qwen4KvBackend::Vmm => 0,
            },
            qsa_compress: config.indexer_compress_ratio,
            model_id,
            reset_epoch: 0,
            transaction_generation: 0,
            map,
        })
    }

    /// Device bytes one snapshot-capable request state commits
    /// ([`StateMirror::Snapshot`]): every allocation
    /// [`Self::new_with_backend`] makes (GDN recurrent, live-only EF and conv, the QSA
    /// fixed arenas and, as `context` commits them, the context arenas, PLE
    /// conv, hyper feedback) plus the snapshot arena's mirror of the state
    /// the forwards overwrite in place. The verify ring
    /// ([`Self::ensure_row_capture`]) is not allocated here and not counted.
    pub(crate) fn device_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
    ) -> Option<usize> {
        Self::device_bytes_scoped(config, format, context, &Qwen4BandScope::single(config))
    }

    /// [`Self::device_bytes`] of one band: only the band's GDN/QSA layers, and
    /// the PLE conv history and hyper feedback (with their snapshot mirrors)
    /// only when `scope.input`. Summed over every band of a placement this is
    /// exactly the whole-model figure (no state is duplicated across bands).
    pub(crate) fn device_bytes_scoped(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        scope: &Qwen4BandScope,
    ) -> Option<usize> {
        Self::device_bytes_with_mirror(config, format, context, scope, StateMirror::Snapshot)
    }

    /// [`Self::device_bytes_scoped`] of a state built with `mirror`
    /// ([`Self::new_with_mirror`]): under [`StateMirror::Absent`] the GDN
    /// recurrent and conv mirrors, the QSA fixed-arena and raw-index circular
    /// mirrors and the PLE/feedback mirrors are all zero; nothing else
    /// (live state, EF residual, context arenas) changes.
    pub(crate) fn device_bytes_with_mirror(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        scope: &Qwen4BandScope,
        mirror: StateMirror,
    ) -> Option<usize> {
        let f32_bytes = std::mem::size_of::<f32>();
        let recurrent = format
            .gdn
            .storage()
            .state_units(
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )
            .checked_mul(format.gdn.storage().dtype().size())?;
        let residual = if format.gdn.has_residual() {
            config
                .linear_num_value_heads
                .checked_mul(config.linear_key_head_dim)?
                .checked_mul(config.linear_value_head_dim)?
                .checked_mul(DType::F16.size())?
        } else {
            0
        };
        let conv_channels = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)?
            .checked_mul(2)?
            .checked_add(
                config
                    .linear_num_value_heads
                    .checked_mul(config.linear_value_head_dim)?,
            )?;
        let conv = conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))?
            .checked_mul(f32_bytes)?;
        let raw_width = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)?;
        let full_width = config.num_key_value_heads.checked_mul(config.head_dim)?;
        let compress = config.indexer_compress_ratio;
        let partial_keys = compress.checked_mul(raw_width)?.checked_mul(f32_bytes)?;
        let partial_values = compress.checked_mul(full_width)?.checked_mul(f32_bytes)?;
        let selected = config
            .qsa_selected_capacity()
            .checked_mul(std::mem::size_of::<i32>())?;
        let circular = compress
            .checked_mul(raw_width)?
            .checked_mul(format.layout().index_dtype().size())?;
        let ple = config
            .ple_conv_history_rows()
            .checked_mul(config.hidden_size)?
            .checked_mul(config.hc_count)?
            .checked_mul(f32_bytes)?;
        let feedback = config
            .hc_count
            .checked_mul(config.hidden_size)?
            .checked_mul(f32_bytes)?;
        let context_arenas = context.committed_layer_bytes(config, format.layout())?;
        // The snapshot arena copies exactly the parts a forward overwrites in
        // place; an absent mirror holds none of them.
        let mirrored = |in_place: usize| match mirror {
            StateMirror::Snapshot => in_place,
            StateMirror::Absent => 0,
        };
        let gdn_in_place = recurrent.checked_add(conv)?;
        let gdn_layer = gdn_in_place
            .checked_add(mirrored(gdn_in_place))?
            .checked_add(residual)?;
        let qsa_in_place = partial_keys
            .checked_add(partial_values)?
            .checked_add(selected)?;
        // The mirror also holds the raw-index circular block (`compress` rows
        // of the context arena's live `raw_index_keys`, counted in
        // `context_arenas`).
        let qsa_layer = context_arenas
            .checked_add(qsa_in_place)?
            .checked_add(mirrored(qsa_in_place))?
            .checked_add(mirrored(circular))?;
        let per_request = if scope.input {
            let in_place = ple.checked_add(feedback)?;
            in_place.checked_add(mirrored(in_place))?
        } else {
            0
        };
        let (gdn_layers, qsa_layers) = scope_layer_counts(config, scope);
        gdn_layer
            .checked_mul(gdn_layers)?
            .checked_add(qsa_layer.checked_mul(qsa_layers)?)?
            .checked_add(per_request)
    }

    /// Device bytes the whole-model QSA context arenas of one state commit under
    /// `context` (every QSA layer's [`Qwen4ContextCommit::committed_layer_bytes`]): the
    /// pages `ensure_mapped_capacity` has mapped once the state holds `context.tokens`
    /// tokens, at the VMM granule. It is the very term [`Self::device_bytes_with_mirror`]
    /// adds per QSA layer, so a lane's planned page growth between two token counts is
    /// the difference of two calls and cannot differ from what the census charges. The
    /// lane page budget ([`crate::lane::pages`]) prices admissions with it.
    pub(crate) fn context_commit_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
    ) -> Option<u64> {
        let (_, qsa_layers) = scope_layer_counts(config, &Qwen4BandScope::single(config));
        let per_layer = context.committed_layer_bytes(config, format.layout())?;
        u64::try_from(per_layer.checked_mul(qsa_layers)?).ok()
    }

    /// Every device allocation [`Self::device_bytes`] counts, one entry per
    /// allocation (aliases never appear: nothing in the state is a borrowed
    /// view of a counted tensor): each tensor of [`state_tensor_plan`] over
    /// the whole-model scope, with QSA context arenas sized as `context`
    /// commits them (the full allocation under legacy storage, each VMM
    /// arena's mapped growth pages for the committed tokens), then one entry
    /// per snapshot-arena mirror allocation
    /// ([`Qwen4StateSnapshotArena::new`]). Zero-byte allocations are
    /// omitted. The entries sum to [`Self::device_bytes`]; `None` on any
    /// overflow, geometry or mapping-plan refusal. CPU-only.
    pub(crate) fn device_allocation_sizes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
    ) -> Option<Vec<u64>> {
        Self::device_allocation_sizes_with_mirror(config, format, context, StateMirror::Snapshot)
    }

    /// [`Self::device_allocation_sizes`] of a state built with `mirror`:
    /// under [`StateMirror::Absent`] no mirror allocation is listed, and the
    /// entries sum to [`Self::device_bytes_with_mirror`] over the whole-model
    /// scope.
    pub(crate) fn device_allocation_sizes_with_mirror(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        mirror: StateMirror,
    ) -> Option<Vec<u64>> {
        let ratio = config.indexer_compress_ratio;
        if ratio == 0 || context.max_seq == 0 {
            return None;
        }
        let map = Qwen4StateMap::new(config, &Qwen4BandScope::single(config)).ok()?;
        let geometry = StateGeometry::new(config, &format, context.max_seq).ok()?;
        let plan = state_tensor_plan(config, &format, context.max_seq, &geometry, &map).ok()?;
        let mut sizes = Vec::with_capacity(2 * plan.len());
        let mut push = |bytes: usize| -> Option<()> {
            if bytes > 0 {
                sizes.push(u64::try_from(bytes).ok()?);
            }
            Some(())
        };
        for spec in &plan {
            let live = if spec.context_arena {
                committed_context_arena_bytes(spec, context, &geometry, ratio)?
            } else {
                spec.elements.checked_mul(spec.dtype.size())?
            };
            push(live)?;
            if mirror == StateMirror::Snapshot {
                if let Some((elements, dtype)) = snapshot_mirror(spec, &geometry, ratio) {
                    push(elements.checked_mul(dtype.size())?)?;
                }
            }
        }
        Some(sizes)
    }

    /// Explicit numerical mode: Raw dtype alone cannot distinguish Q8 and EF.
    pub fn gdn_format(&self) -> GdnStateFormat {
        self.gdn_format
    }

    /// Full-owner reset sequences submitted on this state's stream. Never
    /// snapshot-restored: a restore itself contributes one reset event.
    pub fn gdn_residual_resets(&self) -> u64 {
        self.gdn_residual_resets
    }

    /// Arm (or disarm and clear) the lab residual/ring event tap.
    #[cfg(feature = "lab")]
    pub fn lab_observe_state(&mut self, armed: bool) {
        self.lab.set_armed(armed);
    }

    /// The lab state tap is armed.
    #[cfg(feature = "lab")]
    pub fn lab_state_observed(&self) -> bool {
        self.lab.armed()
    }

    /// Drain the residual-reset and ring events recorded since the last take.
    #[cfg(feature = "lab")]
    pub fn lab_take_state_events(&mut self) -> crate::lab_window::LabStateEvents {
        self.lab.take()
    }

    /// Live capture-ring slot of every GDN layer's recurrent state.
    #[cfg(feature = "lab")]
    pub fn lab_gdn_live(&self) -> usize {
        self.gdn_live
    }

    /// The ring half (0 or 1) the next armed multi-row verify would write
    /// (`capture_base` of the live slot); `None` without a capture ring.
    #[cfg(feature = "lab")]
    pub fn lab_next_verify_ring_half(&self) -> Option<u8> {
        (self.row_capture_rows != 0).then(|| u8::from(self.capture_base(self.gdn_live) != 0))
    }

    fn reset_gdn_residual(
        &mut self,
        gpu: &mut Gpu,
        event: GdnResidualTransition,
    ) -> Result<(), StateError> {
        if !gdn_residual_reset_required(self.gdn_format, event) || self.gdn.is_empty() {
            return Ok(());
        }
        for layer in &self.gdn {
            let residual = layer
                .ef_residual
                .as_ref()
                .ok_or(StateError::AllocationBookkeeping)?;
            gdn_q8ef_residual_zero(gpu, residual).map_err(StateError::Hip)?;
        }
        self.gdn_residual_resets = self.gdn_residual_resets.saturating_add(1);
        #[cfg(feature = "lab")]
        if let Some(kind) = lab_residual_kind(event) {
            self.lab.residual(kind, self.gdn_residual_resets);
        }
        Ok(())
    }

    /// Exact device bytes of ONE slow-link boundary stage: the sum of
    /// [`Self::session_fixed_part_sizes`], equal to what
    /// `Qwen4Bundle::prompt_txn_stage_bytes` allocates.
    pub(crate) fn session_fixed_stage_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
    ) -> Option<u64> {
        Self::session_fixed_part_sizes(config, format)?
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
    }

    /// Per-part bytes of ONE slow-link boundary stage, one entry per
    /// `gpu.zeros` of `Qwen4Bundle::allocate_prompt_stages` (one per fixed
    /// part of the whole-model session layout, unpadded), in
    /// [`Self::session_layout`] order: per GDN layer the recurrent in the
    /// state's own storage and the F32 conv, per QSA layer one `i32`
    /// selection, then the F32 PLE convolution history (PLE profiles) and the
    /// F32 HC feedback. Derived from the same element counts as
    /// [`Self::device_bytes_scoped`]. Zero-byte parts are omitted; `None` on
    /// overflow.
    pub(crate) fn session_fixed_part_sizes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
    ) -> Option<Vec<u64>> {
        let f32_bytes = std::mem::size_of::<f32>();
        let recurrent = format
            .gdn
            .state_units(
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )
            .checked_mul(format.gdn.dtype().size())?;
        let conv_channels = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)?
            .checked_mul(2)?
            .checked_add(
                config
                    .linear_num_value_heads
                    .checked_mul(config.linear_value_head_dim)?,
            )?;
        let conv = conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))?
            .checked_mul(f32_bytes)?;
        let selected = config
            .qsa_selected_capacity()
            .checked_mul(std::mem::size_of::<i32>())?;
        let ple = config
            .ple_conv_history_rows()
            .checked_mul(config.hidden_size)?
            .checked_mul(config.hc_count)?
            .checked_mul(f32_bytes)?;
        let feedback = config
            .hc_count
            .checked_mul(config.hidden_size)?
            .checked_mul(f32_bytes)?;
        let (gdn_layers, qsa_layers) = scope_layer_counts(config, &Qwen4BandScope::single(config));
        let mut sizes = Vec::with_capacity(2 * gdn_layers + qsa_layers + 2);
        for _ in 0..gdn_layers {
            sizes.extend([recurrent, conv]);
        }
        sizes.extend(std::iter::repeat(selected).take(qsa_layers));
        sizes.extend([ple, feedback]);
        sizes
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| u64::try_from(bytes).ok())
            .collect()
    }

    /// A snapshot ticket of this state's arena is outstanding.
    pub(crate) fn snapshot_active(&self) -> bool {
        self.snapshot_arena.active && !self.snapshot_arena.retained
    }

    /// Exchange the requests two states hold: a promotion moves the resident
    /// request into its lane and takes the lane's fresh state back. The
    /// states swap whole, except that a snapshot mirror belongs to the slot
    /// it was provisioned for, not to the request: when exactly one side has
    /// none, the arenas are swapped back and rebound, so the resident state
    /// keeps the mirror it snapshots prompt and verify windows with, and an
    /// AR lane stays within the bytes its census charged (none for the
    /// mirror). Two snapshot-capable states swap whole, as before.
    pub(crate) fn swap_requests(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
        if self.snapshot_arena.mirror != other.snapshot_arena.mirror {
            std::mem::swap(&mut self.snapshot_arena, &mut other.snapshot_arena);
            self.snapshot_arena.rebind(self.model_id);
            other.snapshot_arena.rebind(other.model_id);
        }
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), StateError> {
        self.reset_synced(gpu, true)
    }

    /// [`Self::reset`] that only enqueues its zero-fills into `batch`: the caller
    /// owns the one wait ([`Gpu::memset_batch_sync`], also after a failed reset) and
    /// keeps the state alive until then. A lane admission resets the trunk state, the
    /// MTP head and the pending hidden under one wait.
    pub(crate) fn reset_into(
        &mut self,
        gpu: &mut Gpu,
        batch: &mut MemsetBatch,
    ) -> Result<(), StateError> {
        self.reset_inner(gpu, true, batch)
    }

    /// One reset with its own wait: every fill enqueued, then a single stream wait, and
    /// a wait owed for fills already in flight is paid even when the reset fails.
    fn reset_synced(&mut self, gpu: &mut Gpu, clear_rows: bool) -> Result<(), StateError> {
        let mut batch = MemsetBatch::default();
        let reset = self.reset_inner(gpu, clear_rows, &mut batch);
        let synced = gpu.memset_batch_sync(&mut batch).map_err(StateError::Hip);
        reset.and(synced)
    }

    /// Cold reset of a side request: every fixed byte and every metadata
    /// word resets exactly as [`Self::reset`], but the append-only row
    /// streams of [`Self::session_layout`] (full K/V, raw index keys, pooled
    /// keys) are left untouched, so rows a main conversation wrote stay
    /// byte-exact beneath the side request's writes.
    pub(crate) fn reset_for_side(&mut self, gpu: &mut Gpu) -> Result<(), StateError> {
        self.reset_synced(gpu, false)
    }

    /// Per-stream row ends of a side request writing tokens `[0, through)`,
    /// in [`Self::session_layout`] stream order per QSA layer: full keys,
    /// full values, raw index keys, pooled keys.
    pub(crate) fn side_request_rows(&self, through: usize) -> Vec<usize> {
        let pooled = through.div_ceil(self.qsa_compress);
        let mut rows = Vec::with_capacity(4 * self.qsa.len());
        for _ in &self.qsa {
            rows.extend([through, through, through, pooled]);
        }
        rows
    }

    fn reset_inner(
        &mut self,
        gpu: &mut Gpu,
        clear_rows: bool,
        batch: &mut MemsetBatch,
    ) -> Result<(), StateError> {
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
        self.snapshot_arena.invalidate();
        self.reset_gdn_residual(gpu, GdnResidualTransition::Reset)?;
        for layer in &self.gdn {
            gpu.memset_batched(batch, &layer.recurrent.buf, 0, layer.recurrent.buf.size())
                .map_err(StateError::Hip)?;
            gpu.memset_batched(batch, &layer.conv.buf, 0, layer.conv.buf.size())
                .map_err(StateError::Hip)?;
        }
        // VMM context arenas expose only their mapped prefix: nothing past
        // it exists to clear.
        for layer in &mut self.qsa {
            // Invalidate before the first write, including a failed reset.
            layer.invalidate_f16_arena();
            for (index, tensor) in [
                &layer.full_keys,
                &layer.full_values,
                &layer.raw_index_keys,
                &layer.pooled_keys,
                &layer.partial_keys,
                &layer.partial_values,
                &layer.selected_indices,
            ]
            .into_iter()
            .enumerate()
            {
                // The first four are the append-only row streams.
                if !clear_rows && index < 4 {
                    continue;
                }
                if tensor.buf.size() == 0 {
                    continue;
                }
                // Developer extent check: the memset clears the whole exposed
                // buffer, which for a VMM owner must lie inside its mapped bytes.
                let depth = || rdna_compute::QsaExtentDepth {
                    rows: layer.full_len,
                    end_position: layer.position,
                    block_count: layer.pooled_len,
                };
                if gpu.flags.qwen4_qsa_extent_check {
                    gpu.qsa_extent_check(
                        "qsa_reset_memset",
                        tensor,
                        0,
                        tensor.buf.size(),
                        depth(),
                        "Qwen4State::reset",
                    );
                }
                gpu.memset_batched(batch, &tensor.buf, 0, tensor.buf.size())
                    .map_err(StateError::Hip)?;
                if gpu.flags.qwen4_qsa_extent_check {
                    // The post-launch check reads the fill's result: wait for it here.
                    gpu.memset_batch_sync(batch).map_err(StateError::Hip)?;
                    gpu.qsa_extent_post_launch(
                        "qsa_reset_memset",
                        tensor,
                        0,
                        tensor.buf.size(),
                        depth(),
                        "Qwen4State::reset",
                    );
                }
            }
            layer.full_len = 0;
            layer.raw_len = 0;
            layer.pooled_len = 0;
            layer.partial_len = 0;
            layer.selected_len = 0;
            layer.position = 0;
        }
        if let Some(ple_conv) = &self.ple_conv {
            gpu.memset_batched(batch, &ple_conv.buf, 0, ple_conv.buf.size())
                .map_err(StateError::Hip)?;
        }
        if let Some(hyper_feedback) = &self.hyper_feedback {
            gpu.memset_batched(batch, &hyper_feedback.buf, 0, hyper_feedback.buf.size())
                .map_err(StateError::Hip)?;
        }
        self.ple_history.reset();
        self.position = 0;
        Ok(())
    }

    /// Map every QSA context arena far enough for `required_tokens` tokens
    /// (pooled keys: `ceil(required / compress)` rows) before any write or
    /// view reaches them. A request above the admitted `max_seq_len` is
    /// refused before mapping anything; no length mark or position moves
    /// here. Growth uses the shared KV chunk plan per arena byte stride and
    /// zeros only the newly mapped bytes. Legacy arenas are fully allocated,
    /// so this is then only the bounds check. Growth is monotonic: after a
    /// failed map, completed arenas stay mapped and owned, and the next call
    /// fills the rest. Refused while a graph is capturing.
    pub fn ensure_mapped_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_tokens: usize,
    ) -> Result<(), StateError> {
        if required_tokens > self.max_seq_len {
            return Err(StateError::ContextCapacity {
                required: required_tokens,
                admitted: self.max_seq_len,
            });
        }
        if required_tokens <= self.qsa_mapped_tokens {
            return Ok(());
        }
        if gpu.graphs.capture_mode
            || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some()
        {
            return Err(StateError::GrowthDuringCapture);
        }
        let device = gpu.device_id;
        let compress = self.qsa_compress;
        let max_tokens = self.max_seq_len;
        let mut covered = max_tokens;
        for layer in &mut self.qsa {
            // Growth replaces only the owner's exposed prefix view, not
            // existing mappings. Pool drain/retry reclaims free buffers,
            // never live VMM owners; keep only the committed valid prefix.
            layer.clamp_f16_valid();
            for arena in layer.context_arenas_mut() {
                let rows = required_tokens.div_ceil(arena.tokens_per_row(compress));
                // The F16 V arena writes whole four-token blocks.
                let rows = if arena.blocks {
                    qsa_f16_arena_rows(rows).min(arena.rows)
                } else {
                    rows
                };
                let mapped_rows = grow_context_arena(gpu, arena.tensor, arena.rows, rows, device)?;
                covered = covered.min(qsa_covered_tokens(
                    mapped_rows,
                    arena.tokens_per_row(compress),
                    max_tokens,
                ));
            }
        }
        if covered < required_tokens {
            return Err(StateError::ContextCapacity {
                required: required_tokens,
                admitted: covered,
            });
        }
        self.qsa_mapped_tokens = covered;
        Ok(())
    }

    /// Device bytes the QSA context arenas commit: VMM owners' mapped pages
    /// (whole driver pages, so the last may pass the logical size), legacy
    /// arenas' full allocations.
    pub fn mapped_context_bytes(&self, gpu: &Gpu) -> Result<usize, StateError> {
        let mut bytes = 0usize;
        for layer in &self.qsa {
            for tensor in layer.context_arenas() {
                let committed = match self.qsa_backend {
                    Qwen4KvBackend::Legacy => tensor.buf.size(),
                    Qwen4KvBackend::Vmm => {
                        gpu.vmm_mapped_bytes(tensor).ok_or(StateError::VmmOwner)?
                    }
                };
                bytes = bytes
                    .checked_add(committed)
                    .ok_or(StateError::DimensionOverflow)?;
            }
        }
        Ok(bytes)
    }

    /// Tokens every QSA context arena currently covers (the no-growth gate).
    pub fn mapped_context_tokens(&self) -> usize {
        self.qsa_mapped_tokens
    }

    /// Read-only exactness digest of the request state, for the EP oracle
    /// (`fn_ep_oracle`): `(name, value)` rows in a fixed order, identical
    /// across ranks/arms that ran the same host calls.
    ///
    /// Device rows are `"<sha256 hex> bytes=<n>"` (fixed-size state) or
    /// `"<sha256 hex> bytes=<n> rows=<r>"` (QSA arenas):
    /// * fixed-size live state, hashed whole (`byte_size()`): every GDN
    ///   recurrent (the live ring-slot view), live EF residual when present,
    ///   conv, PLE conv and hyper feedback. EF is reset by recorded dispatch;
    ///   serialized layouts never include it.
    /// * allocated row-capture rings/inputs/recurrence/output, hashed whole
    ///   as transaction evidence. A verify writes only its final ring state,
    ///   not every intermediate slot; these are not logical-state digests.
    /// * QSA arenas, hashed over exactly the ACTIVE prefix the layer's marks
    ///   name (rows beyond it, and freshly mapped VMM pages, are not
    ///   guaranteed written): `full_keys` / `full_values` the first
    ///   `full_len` rows, `raw_index_keys` `raw_len`, `pooled_keys`
    ///   `pooled_len`, `partial_keys` / `partial_values` `partial_len`,
    ///   `selected_indices` `selected_len` `i32`s; bytes per row is the
    ///   arena's allocation geometry (`byte_size / capacity`).
    ///
    /// Host rows are decimal: position, PLE history, GDN live slot,
    /// row-capture rows, mapped tokens, and every QSA length/position mark.
    ///
    /// Deliberately excluded: the snapshot arena contents (scratch copies
    /// written only by `snapshot`, never by a forward); the row-capture
    /// pointer table (device addresses differ per process); the process-local
    /// `model_id` / `reset_epoch` / `transaction_generation` counters.
    pub fn debug_digest(&self, gpu: &mut Gpu) -> Result<Vec<(String, String)>, StateError> {
        self.digest_rows(gpu, false)
    }

    /// Canonical layer-ownership projection of this rank's debug receipt:
    /// rows of [`Self::debug_digest`] keyed by ABSOLUTE layer
    /// (`qsa[L11].full_keys`, `gdn[L0].recurrent`, `row_capture[L0].ring`)
    /// instead of the local slot, in ascending absolute order. Values are the
    /// same digests. The union over the ranks of a placement equals the
    /// whole-model state's canonical rows (the global rows `position`,
    /// `gdn_live` appear on every rank that owns state they describe, with
    /// equal values; PLE history on the input rank only). Capacity and
    /// scratch rows (`row_capture_rows`, `qsa_mapped_tokens`,
    /// `row_capture_output`, the snapshot arena, device pointers, process
    /// counters) are not part of it: the rollback re-run's discarded output
    /// is rewritten in full by every re-run launch before anything could
    /// read it, so no live cache hides behind that exclusion.
    ///
    /// Capture rings/inputs/recurrence and physical live indices intentionally
    /// remain for same-call EP receipts. AR-vs-verify logical comparisons must
    /// compare live semantic tensors/marks, with ring indices reported
    /// separately, rather than treating this complete receipt as logical state.
    pub fn canonical_digest_rows(
        &self,
        gpu: &mut Gpu,
    ) -> Result<Vec<(String, String)>, StateError> {
        self.digest_rows(gpu, true)
    }

    fn digest_rows(
        &self,
        gpu: &mut Gpu,
        canonical: bool,
    ) -> Result<Vec<(String, String)>, StateError> {
        use sha2::{Digest, Sha256};
        gpu.hip.device_synchronize().map_err(StateError::Hip)?;
        // SHA-256 of the first `bytes` bytes of `tensor`, as a row value.
        let hash_prefix = |tensor: &GpuTensor, bytes: usize| -> Result<String, StateError> {
            if bytes > tensor.buf.size() {
                return Err(StateError::SnapshotLength);
            }
            let mut host = vec![0u8; bytes];
            if bytes > 0 {
                gpu.hip
                    .memcpy_dtoh(&mut host, &tensor.buf)
                    .map_err(StateError::Hip)?;
            }
            Ok(Sha256::digest(&host)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect())
        };
        // A whole tensor: its logical extent (`numel * dtype`), not the
        // allocation, which a pooled allocator may round up with stale tail
        // bytes no kernel reads.
        let device = |tensor: &GpuTensor| -> Result<String, StateError> {
            let bytes = tensor.byte_size().min(tensor.buf.size());
            let hex = hash_prefix(tensor, bytes)?;
            Ok(format!("{hex} bytes={bytes}"))
        };
        // The first `live` of `capacity` rows of an arena.
        let active =
            |tensor: &GpuTensor, capacity: usize, live: usize| -> Result<String, StateError> {
                if live > capacity {
                    return Err(StateError::SnapshotLength);
                }
                let bytes = arena_row_bytes(tensor, capacity)?
                    .checked_mul(live)
                    .ok_or(StateError::DimensionOverflow)?;
                let hex = hash_prefix(tensor, bytes)?;
                Ok(format!("{hex} bytes={bytes} rows={live}"))
            };
        let missing = || StateError::AllocationBookkeeping;
        let mut plan = digest_plan(
            &self.map,
            canonical,
            !self.row_capture.is_empty(),
            self.row_capture_output.is_some(),
        );
        if self.gdn_format.has_residual() {
            plan.extend((0..self.gdn.len()).map(DigestRow::GdnResidual));
        }
        let mut rows = Vec::with_capacity(plan.len());
        for row in plan {
            let value = match row {
                DigestRow::Position => self.position.to_string(),
                DigestRow::LayerPosition(_) => self.position.to_string(),
                DigestRow::PleHistoryPrevious => format!("{:?}", self.ple_history.previous()),
                DigestRow::PleHistoryEos => self.ple_history.eos_token_id().to_string(),
                DigestRow::GdnLive => self.gdn_live.to_string(),
                DigestRow::GdnLiveLayer(_) => self.gdn_live.to_string(),
                DigestRow::RowCaptureRows => self.row_capture_rows.to_string(),
                DigestRow::QsaMappedTokens => self.qsa_mapped_tokens.to_string(),
                DigestRow::QsaMarks(slot) => {
                    let mark = self.qsa.get(slot).ok_or_else(missing)?.mark();
                    format!(
                        "full={} raw={} pooled={} partial={} selected={} position={}",
                        mark.full_len,
                        mark.raw_len,
                        mark.pooled_len,
                        mark.partial_len,
                        mark.selected_len,
                        mark.position
                    )
                }
                DigestRow::GdnRecurrent(slot) => {
                    device(&self.gdn.get(slot).ok_or_else(missing)?.recurrent)?
                }
                DigestRow::GdnResidual(slot) => device(
                    self.gdn
                        .get(slot)
                        .ok_or_else(missing)?
                        .ef_residual
                        .as_ref()
                        .ok_or_else(missing)?,
                )?,
                DigestRow::GdnConv(slot) => device(&self.gdn.get(slot).ok_or_else(missing)?.conv)?,
                DigestRow::Qsa(slot, field) => {
                    let layer = self.qsa.get(slot).ok_or_else(missing)?;
                    let (tensor, capacity, live) = match field {
                        QsaDigestField::FullKeys => {
                            (&layer.full_keys, layer.full_capacity, layer.full_len)
                        }
                        QsaDigestField::FullValues => {
                            (&layer.full_values, layer.full_capacity, layer.full_len)
                        }
                        QsaDigestField::RawIndexKeys => {
                            (&layer.raw_index_keys, layer.raw_capacity, layer.raw_len)
                        }
                        QsaDigestField::PooledKeys => {
                            (&layer.pooled_keys, layer.pooled_capacity, layer.pooled_len)
                        }
                        QsaDigestField::PartialKeys => (
                            &layer.partial_keys,
                            layer.partial_capacity,
                            layer.partial_len,
                        ),
                        QsaDigestField::PartialValues => (
                            &layer.partial_values,
                            layer.partial_capacity,
                            layer.partial_len,
                        ),
                        QsaDigestField::SelectedIndices => (
                            &layer.selected_indices,
                            layer.selected_capacity,
                            layer.selected_len,
                        ),
                    };
                    active(tensor, capacity, live)?
                }
                DigestRow::PleConv => device(self.ple_conv.as_ref().ok_or_else(missing)?)?,
                DigestRow::HyperFeedback => {
                    device(self.hyper_feedback.as_ref().ok_or_else(missing)?)?
                }
                DigestRow::RowCapture(slot, field) => {
                    let (ring, inputs, recurrence) =
                        self.row_capture.get(slot).ok_or_else(missing)?;
                    device(match field {
                        CaptureDigestField::Ring => ring,
                        CaptureDigestField::Inputs => inputs,
                        CaptureDigestField::Recurrence => recurrence,
                    })?
                }
                DigestRow::RowCaptureOutput => {
                    let (output, _pointer_table) =
                        self.row_capture_output.as_ref().ok_or_else(missing)?;
                    device(output)?
                }
            };
            rows.push((digest_key(&self.map, row, canonical), value));
        }
        Ok(rows)
    }

    /// Storage of the QSA context arenas, fixed at construction.
    pub fn qsa_backend(&self) -> Qwen4KvBackend {
        self.qsa_backend
    }
    /// Bind rollback tickets to the owning runtime transaction generation.
    pub(crate) fn bind_transaction_generation(&mut self, generation: u64) {
        self.transaction_generation = generation;
        self.snapshot_arena.invalidate();
    }

    /// Device ranges of the live state for `marks`: GDN recurrent (live slot)
    /// and conv, per QSA layer the full K/V, raw and pooled rows below the
    /// mark plus the selection, then PLE conv and hyper feedback.
    fn session_layout(&self, marks: &[QsaMark]) -> Result<StateLayout<'_>, StateError> {
        let mut fixed = Vec::with_capacity(2 * self.gdn.len() + self.qsa.len() + 2);
        let mut rows = Vec::with_capacity(4 * self.qsa.len());
        for layer in &self.gdn {
            fixed.push(whole_part(&layer.recurrent));
            fixed.push(whole_part(&layer.conv));
        }
        for (layer, mark) in self.qsa.iter().zip(marks) {
            for (tensor, capacity, len) in [
                (&layer.full_keys, layer.full_capacity, mark.full_len),
                (&layer.full_values, layer.full_capacity, mark.full_len),
                (&layer.raw_index_keys, layer.raw_capacity, mark.raw_len),
                (&layer.pooled_keys, layer.pooled_capacity, mark.pooled_len),
            ] {
                rows.push(row_stream(tensor, capacity, len)?);
            }
            fixed.push(whole_part(&layer.selected_indices));
        }
        if let Some(ple_conv) = &self.ple_conv {
            fixed.push(whole_part(ple_conv));
        }
        if let Some(hyper_feedback) = &self.hyper_feedback {
            fixed.push(whole_part(hyper_feedback));
        }
        Ok(StateLayout { fixed, rows })
    }

    /// Session-cache capture: metadata words and the device ranges that
    /// make up the live state. Never inside an armed verify.
    pub(crate) fn session_parts(&self) -> Result<(Vec<u8>, StateLayout<'_>), StateError> {
        if self.row_capture_armed {
            return Err(StateError::SnapshotBusy);
        }
        self.session_parts_at(self.position, self.ple_history, None)
    }

    /// Describe a bank-A boundary without rewinding the live bank-B owners.
    pub(crate) fn session_parts_at(
        &self,
        position: usize,
        history: PleHistory,
        commits: Option<&[(usize, usize, usize, usize, usize)]>,
    ) -> Result<(Vec<u8>, StateLayout<'_>), StateError> {
        let mut words = vec![
            gdn_session_tag(self.map.session_tag(), self.gdn_format),
            position as u64,
        ];
        if self.ple_conv.is_some() {
            let [ple0, ple1] = history.previous();
            words.extend([u64::from(ple0), u64::from(ple1)]);
        }
        let mut marks = Vec::with_capacity(self.qsa.len());
        for (slot, layer) in self.qsa.iter().enumerate() {
            let mut mark = layer.mark();
            if let Some(commits) = commits {
                let &(full, raw, pooled, selected, position) =
                    commits.get(slot).ok_or(StateError::SnapshotShape)?;
                mark.full_len = full;
                mark.raw_len = raw;
                mark.pooled_len = pooled;
                mark.selected_len = selected;
                mark.position = position;
            }
            validate_qsa_mark(&mark, &layer.mark_capacity())?;
            words.extend(
                [
                    mark.full_len,
                    mark.raw_len,
                    mark.pooled_len,
                    mark.partial_len,
                    mark.selected_len,
                    mark.position,
                ]
                .map(|value| value as u64),
            );
            marks.push(mark);
        }
        Ok((meta_bytes(&words), self.session_layout(&marks)?))
    }

    /// Byte length of a [`Self::session_parts`] meta: a reserved zero word,
    /// position, the PLE context (two words, only with PLE), then six words
    /// per QSA layer.
    pub(crate) fn session_meta_bytes(&self) -> usize {
        8 * (2 + self.session_ple_words() + 6 * self.qsa.len())
    }

    /// Meta words of the PLE token context: two with PLE, none without.
    fn session_ple_words(&self) -> usize {
        if self.ple_conv.is_some() {
            2
        } else {
            0
        }
    }

    /// Position, PLE context and QSA marks of a [`Self::session_parts`] meta.
    fn parse_session_meta(
        &self,
        meta: &[u8],
    ) -> Result<(usize, PleHistory, Vec<QsaMark>), StateError> {
        let words = meta_words(meta).ok_or(StateError::SnapshotShape)?;
        let to_u32 = |word: u64| u32::try_from(word).map_err(|_| StateError::SnapshotShape);
        if words.len() * 8 != self.session_meta_bytes() {
            return Err(StateError::SnapshotShape);
        }
        // The reserved word carries band identity and the live-only EF ABI.
        validate_gdn_session_tag(self.map.session_tag(), self.gdn_format, words[0])?;
        let ple_words = self.session_ple_words();
        let ple = if ple_words == 0 {
            self.ple_history
        } else {
            PleHistory::from_previous(
                self.ple_history.eos_token_id(),
                [to_u32(words[2])?, to_u32(words[3])?],
            )
        };
        let marks = words[2 + ple_words..]
            .chunks_exact(6)
            .map(|w| QsaMark {
                full_len: w[0] as usize,
                raw_len: w[1] as usize,
                pooled_len: w[2] as usize,
                partial_len: w[3] as usize,
                selected_len: w[4] as usize,
                position: w[5] as usize,
            })
            .collect();
        Ok((words[1] as usize, ple, marks))
    }

    pub(crate) fn session_layout_for_meta(
        &self,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, StateError> {
        let (_, _, marks) = self.parse_session_meta(meta)?;
        self.session_layout(&marks)
    }

    /// Ready the live state for a session snapshot: map the context arenas
    /// to cover every row the snapshot holds and retire every speculative
    /// ticket (like a reset). Live buffers stay in place: the caller copies
    /// into the returned destination layout, in [`Self::session_parts`]
    /// order.
    pub(crate) fn prepare_session_restore(
        &mut self,
        gpu: &mut Gpu,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, StateError> {
        if self.row_capture_armed {
            return Err(StateError::SnapshotBusy);
        }
        let (position, _, marks) = self.parse_session_meta(meta)?;
        let compress = self.qsa_compress;
        let mut required = position;
        for (mark, layer) in marks.iter().zip(&self.qsa) {
            validate_qsa_mark(mark, &layer.mark_capacity())?;
            required = required.max(mark.full_len).max(mark.raw_len).max(
                mark.pooled_len
                    .saturating_mul(compress)
                    .min(self.max_seq_len),
            );
        }
        self.ensure_mapped_capacity(gpu, required)?;
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
        self.snapshot_arena.invalidate();
        self.invalidate_f16_arenas();
        self.reset_gdn_residual(gpu, GdnResidualTransition::Restore)?;
        self.session_layout(&marks)
    }

    /// Apply a session snapshot's marks, PLE context and position after its
    /// device bytes were copied in.
    pub(crate) fn finish_session_restore(&mut self, meta: &[u8]) -> Result<(), StateError> {
        let (position, ple, marks) = self.parse_session_meta(meta)?;
        for (layer, mark) in self.qsa.iter_mut().zip(marks) {
            layer.full_len = mark.full_len;
            layer.raw_len = mark.raw_len;
            layer.pooled_len = mark.pooled_len;
            layer.partial_len = mark.partial_len;
            layer.selected_len = mark.selected_len;
            layer.position = mark.position;
            // The restored bytes replaced all F32 rows; rebuild lazily.
            layer.invalidate_f16_arena();
        }
        self.ple_history = ple;
        self.position = position;
        Ok(())
    }

    /// Context-arena bytes still unmapped below `through` tokens (at most
    /// `max_seq_len`; 0 for legacy full-capacity arenas): each arena's rows
    /// needed for `through` tokens (as [`Self::ensure_mapped_capacity`] maps
    /// them: pooled keys and the pooled BF16 mirror by `ceil(through /
    /// compress)`, the F16 arena rounded to four tokens) minus the rows
    /// already mapped.
    pub(crate) fn context_growth_bytes(&self, through: usize) -> u64 {
        if self.qsa_backend == Qwen4KvBackend::Legacy || self.max_seq_len == 0 {
            return 0;
        }
        let through = through.min(self.max_seq_len);
        self.qsa
            .iter()
            .map(|layer| {
                layer.context_growth_bytes(self.qsa_compress, self.qsa_mapped_tokens, through)
            })
            .sum()
    }

    /// Capture the fixed-size rollback state into the model-owned arena.
    /// Full-capacity append-only QSA K/V and raw arenas are never copied.
    /// A state built [`StateMirror::Absent`] has no arena to capture into:
    /// [`StateError::MirrorAbsent`].
    pub fn snapshot(&mut self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, StateError> {
        self.snapshot_arena.require_mirror()?;
        // A retained entry belongs to a window already committed: the next
        // snapshot supersedes it (its tickets stop validating).
        self.release_retained();
        let dimensions = Qwen4StateSnapshotArena::dimensions(self);
        {
            let arena = &self.snapshot_arena;
            if arena.active {
                return Err(StateError::SnapshotBusy);
            }
            if arena.model_id != self.model_id {
                return Err(StateError::SnapshotTicket);
            }
            arena.validate_layout(self)?;
        }
        for layer in &self.qsa {
            let mark = layer.mark();
            validate_qsa_mark(&mark, &layer.mark_capacity())?;
        }
        let generation = self.snapshot_arena.generation;
        let result = (|| -> Result<(), StateError> {
            self.snapshot_arena.gdn_live =
                (self.row_capture_armed && self.row_capture_ringed()).then_some(self.gdn_live);
            let copies = arena_regions(self, &self.snapshot_arena, true)?;
            copy_regions(gpu, &copies).map_err(StateError::Hip)?;
            let arena = &mut self.snapshot_arena;
            for (index, layer) in self.qsa.iter().enumerate() {
                arena.qsa_marks[index] = layer.mark();
            }
            arena.ple_history = self.ple_history;
            arena.position = self.position;
            arena.active = true;
            Ok(())
        })();
        if let Err(error) = result {
            self.snapshot_arena.invalidate();
            return Err(error);
        }
        Ok(Qwen4StateSnapshot {
            model_id: self.model_id,
            reset_epoch: self.reset_epoch,
            transaction_generation: self.transaction_generation,
            arena_generation: generation,
            dimensions,
        })
    }

    fn restore_impl(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
        consume: bool,
    ) -> Result<(), StateError> {
        {
            let arena = &self.snapshot_arena;
            arena.validate_ticket(self, &snapshot)?;
        }
        let result = (|| -> Result<(), StateError> {
            if let Some(live) = self.snapshot_arena.gdn_live {
                self.set_gdn_live(live);
                #[cfg(feature = "lab")]
                self.lab
                    .ring(crate::lab_window::LabRingEvent::Restore { live });
            }
            let copies = arena_regions(self, &self.snapshot_arena, false)?;
            copy_regions(gpu, &copies).map_err(StateError::Hip)?;
            self.reset_gdn_residual(gpu, GdnResidualTransition::Restore)?;
            let arena = &mut self.snapshot_arena;
            for (index, layer) in self.qsa.iter_mut().enumerate() {
                let mark = arena.qsa_marks[index];
                layer.full_len = mark.full_len;
                layer.raw_len = mark.raw_len;
                layer.pooled_len = mark.pooled_len;
                layer.partial_len = mark.partial_len;
                layer.selected_len = mark.selected_len;
                layer.position = mark.position;
                layer.clamp_f16_valid();
            }
            self.ple_history = arena.ple_history;
            self.position = arena.position;
            if consume {
                arena.invalidate();
            }
            Ok(())
        })();
        if result.is_err() && consume {
            self.snapshot_arena.invalidate();
        }
        result
    }

    /// Restore and consume a model-owned arena ticket. All ticket, dimension,
    /// and mark checks happen before the first device copy.
    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), StateError> {
        self.restore_impl(gpu, snapshot, true)
    }

    /// Restore a ticket while retaining it for a surrounding multi-owner
    /// transaction. The caller must eventually call [`Self::restore`] on error
    /// or [`Self::commit`] on success to invalidate the retained arena entry.
    pub(crate) fn restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), StateError> {
        self.restore_impl(gpu, snapshot, false)
    }

    pub(crate) fn validate_commit(&self, snapshot: Qwen4StateSnapshot) -> Result<(), StateError> {
        self.snapshot_arena.validate_ticket(self, &snapshot)
    }

    pub(crate) fn commit_validated(&mut self, _snapshot: Qwen4StateSnapshot) {
        debug_assert!(
            self.snapshot_arena.active,
            "Qwen4 commit_validated requires an active snapshot"
        );
        self.snapshot_arena.invalidate();
    }

    /// Keep a validated ticket's arena entry active after a committed window
    /// so the caller can still roll the state back to the ticket (a terminal
    /// stop inside the window). The next [`Self::snapshot`], any reset or a
    /// session restore releases it; [`Self::release_retained`] does so
    /// explicitly. The ticket stays valid for [`Self::restore_retain`].
    pub(crate) fn retain_validated(&mut self, _snapshot: Qwen4StateSnapshot) {
        debug_assert!(
            self.snapshot_arena.active,
            "Qwen4 retain_validated requires an active snapshot"
        );
        self.snapshot_arena.retained = true;
    }

    /// Drop the retained arena entry, if any (a no-op otherwise).
    pub(crate) fn release_retained(&mut self) {
        if self.snapshot_arena.retained {
            self.snapshot_arena.invalidate();
        }
    }

    /// Commit and consume a model-owned arena ticket without copying.
    pub fn commit(
        &mut self,
        snapshot: Qwen4StateSnapshot,
        _gpu: &mut Gpu,
    ) -> Result<(), StateError> {
        self.validate_commit(snapshot)?;
        self.commit_validated(snapshot);
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), StateError> {
        let mut first = self.snapshot_arena.free_gpu(gpu);
        let mut free = |tensor: GpuTensor| {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        };
        let ringed = !self.row_capture.is_empty();
        for layer in self.gdn {
            // With a ring, `recurrent` is a view of it (freed below).
            if !ringed {
                free(layer.recurrent);
            }
            if let Some(residual) = layer.ef_residual {
                free(residual);
            }
            free(layer.conv);
        }
        for layer in self.qsa {
            layer.invalidate_f16_arena();
            free(layer.full_keys);
            free(layer.full_values);
            free(layer.raw_index_keys);
            free(layer.pooled_keys);
            free(layer.partial_keys);
            free(layer.partial_values);
            free(layer.selected_indices);
            if let Some(arena) = layer.f16_arena {
                free(arena.keys);
                free(arena.values);
            }
            if let Some(arena) = layer.pooled_bf16 {
                free(arena.mirror);
            }
        }
        if let Some(ple_conv) = self.ple_conv {
            free(ple_conv);
        }
        if let Some(hyper_feedback) = self.hyper_feedback {
            free(hyper_feedback);
        }
        // With a shared pool the per-layer rings are views of it.
        let pooled = self.row_capture_pool.is_some();
        if let Some(pool) = self.row_capture_pool {
            free(pool);
        }
        for (ring, inputs, recurrence) in self.row_capture {
            if !pooled {
                free(ring);
            }
            free(inputs);
            free(recurrence);
        }
        if let Some((output, table)) = self.row_capture_output {
            free(output);
            free(table);
        }
        first.map_or(Ok(()), |error| Err(StateError::Hip(error)))
    }

    /// Device bytes [`Self::ensure_row_capture`] allocates for `rows`-row
    /// blocks with `gdn`-format recurrent states: per GDN layer a two-half
    /// recurrent-state ring in that format's storage, the F32 input rows and
    /// recurrence inputs, then one F32 output block and the pointer table.
    /// The single-slot recurrent state each ring replaces is freed into the
    /// GPU pool, which keeps it (it stays charged as fixed state).
    pub(crate) fn row_capture_bytes(
        config: &Qwen4Config,
        gdn: GdnStateFormat,
        rows: usize,
    ) -> Option<usize> {
        Self::row_capture_bytes_scoped(config, gdn, rows, config.n_linear_layers())
    }

    /// [`Self::row_capture_bytes`] for a band owning `gdn_layers` GDN layers.
    pub(crate) fn row_capture_bytes_scoped(
        config: &Qwen4Config,
        gdn: GdnStateFormat,
        rows: usize,
        gdn_layers: usize,
    ) -> Option<usize> {
        if rows > hipfire_config::QWEN4_VERIFY_MAX_ROWS {
            return None;
        }
        let sizes = Self::row_capture_allocation_sizes_scoped(config, gdn, rows, gdn_layers)?;
        let total = sizes
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))?;
        usize::try_from(total).ok()
    }

    /// [`Self::row_capture_bytes`] per allocation of
    /// [`Self::ensure_row_capture`], in its allocation order: with the shared
    /// pool ([`lane_reserve_trim`], default) the pool first, then per GDN layer
    /// the F32 input rows and the F32 recurrence inputs; with it off per GDN
    /// layer the recurrent-state ring (storage format), the inputs and the
    /// recurrence inputs; then the one F32 output block and the pointer table.
    /// Empty below two rows or for a band with no GDN layer (nothing is
    /// allocated); `None` on overflow.
    pub(crate) fn row_capture_allocation_sizes_scoped(
        config: &Qwen4Config,
        gdn: GdnStateFormat,
        rows: usize,
        gdn_layers: usize,
    ) -> Option<Vec<u64>> {
        Self::row_capture_allocation_sizes_layout(
            config,
            gdn,
            rows,
            gdn_layers,
            compact_capture_ring(),
            lane_reserve_trim(),
        )
    }

    /// [`Self::row_capture_allocation_sizes_scoped`] for an explicit ring
    /// layout: `compact` ([`compact_capture_ring`]) and `trim`
    /// ([`lane_reserve_trim`], effective only with `compact`).
    pub(crate) fn row_capture_allocation_sizes_layout(
        config: &Qwen4Config,
        gdn: GdnStateFormat,
        rows: usize,
        gdn_layers: usize,
        compact: bool,
        trim: bool,
    ) -> Option<Vec<u64>> {
        if rows > hipfire_config::QWEN4_VERIFY_MAX_ROWS {
            return None;
        }
        if rows < 2 || gdn_layers == 0 {
            // A band with no GDN layer allocates nothing: `ensure_row_capture`
            // returns before the output block and pointer table.
            return Some(Vec::new());
        }
        let value_heads = config.linear_num_value_heads;
        let f32_bytes = std::mem::size_of::<f32>();
        let key_units = match gdn.storage() {
            GdnStateFormat::F32 => config.linear_key_head_dim,
            GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => {
                config.linear_key_head_dim.checked_add(4)?
            }
        };
        let slot_bytes = value_heads
            .checked_mul(config.linear_value_head_dim)?
            .checked_mul(key_units)?
            .checked_mul(gdn.storage().dtype().size())?;
        let ring = slot_bytes.checked_mul(capture_ring_slots(compact, rows)?)?;
        // One shared pool for every layer's window (`lane_reserve_trim`).
        let pool = if compact && trim {
            Some(slot_bytes.checked_mul(capture_ring_pool_slots(rows, gdn_layers)?)?)
        } else {
            None
        };
        let qkv = config
            .linear_num_key_heads
            .checked_mul(2)?
            .checked_mul(config.linear_key_head_dim)?
            .checked_add(value_heads.checked_mul(config.linear_value_head_dim)?)?;
        let inputs = rows
            .checked_mul(qkv)?
            .checked_mul(std::mem::size_of::<f32>())?;
        let recurrence = rows
            .checked_mul(qkv.checked_add(value_heads.checked_mul(2)?)?)?
            .checked_mul(std::mem::size_of::<f32>())?;
        let output = rows
            .checked_mul(value_heads.checked_mul(GDN_HEAD_DIM)?)?
            .checked_mul(f32_bytes)?;
        let table = gdn_layers
            .checked_mul(2 * std::mem::size_of::<u64>())?
            .max(1);
        // Each layer's allocations are checked once; the layer count is not
        // multiplied out so an overflowing total still refuses.
        let layer = ring.checked_add(inputs)?.checked_add(recurrence)?;
        gdn_layers
            .checked_mul(layer)?
            .checked_add(output)?
            .checked_add(table)?;
        let mut sizes = Vec::with_capacity(3 * gdn_layers + 3);
        match pool {
            Some(pool) => {
                sizes.push(pool);
                for _ in 0..gdn_layers {
                    sizes.extend([inputs, recurrence]);
                }
            }
            None => {
                for _ in 0..gdn_layers {
                    sizes.extend([ring, inputs, recurrence]);
                }
            }
        }
        sizes.extend([output, table]);
        sizes
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| u64::try_from(bytes).ok())
            .collect()
    }

    /// Allocate two capture halves and retained recurrence inputs for blocks
    /// of up to `rows` rows (`conv_history` = convolution kernel - 1).
    /// Only the last state is materialized; prefix repair re-runs recurrence.
    /// Admission/growth must precede arming and taking a snapshot.
    pub(crate) fn ensure_row_capture(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
        conv_history: usize,
    ) -> Result<(), StateError> {
        if rows > hipfire_config::QWEN4_VERIFY_MAX_ROWS {
            return Err(StateError::Length {
                what: "verify capture rows",
                expected: hipfire_config::QWEN4_VERIFY_MAX_ROWS,
                actual: rows,
            });
        }
        if rows <= self.row_capture_rows || rows < 2 {
            return Ok(());
        }
        // A verify ring exists to roll a window back from a snapshot ticket.
        self.snapshot_arena.require_mirror()?;
        if self.snapshot_arena.active || self.row_capture_armed {
            return Err(StateError::SnapshotBusy);
        }
        if gpu.replay.is_recording() || gpu.graphs.capture_mode {
            return Err(StateError::Config(
                "GDN capture growth must precede replay recording or graph capture".into(),
            ));
        }
        if self.gdn.is_empty() {
            // A band without GDN layers has no recurrence to capture; the
            // rows and live slot still track the other owners so a rollback
            // agrees with them.
            self.snapshot_arena.invalidate();
            self.row_capture_rows = rows;
            self.gdn_live = 0;
            #[cfg(feature = "lab")]
            self.lab
                .ring(crate::lab_window::LabRingEvent::Realloc { ring_rows: rows });
            return Ok(());
        }
        // Validate every extent before allocating or publishing anything.
        let overflow = || StateError::DimensionOverflow;
        let value_width = self
            .gdn_value_heads
            .checked_mul(GDN_HEAD_DIM)
            .ok_or_else(overflow)?;
        let output_elements = rows.checked_mul(value_width).ok_or_else(overflow)?;
        let table_bytes = self
            .gdn
            .len()
            .checked_mul(2 * std::mem::size_of::<u64>())
            .ok_or_else(overflow)?;
        output_elements
            .checked_mul(DType::F32.size())
            .ok_or_else(overflow)?;
        for layer in &self.gdn {
            if conv_history == 0 || !layer.conv.numel().is_multiple_of(conv_history) {
                return Err(StateError::Config(
                    "invalid GDN capture convolution history".into(),
                ));
            }
            let qkv = layer.conv.numel() / conv_history;
            let ring = rows
                .checked_mul(2)
                .and_then(|n| n.checked_mul(layer.recurrent.numel()))
                .ok_or_else(overflow)?;
            ring.checked_mul(layer.recurrent.dtype.size())
                .ok_or_else(overflow)?;
            let recurrence = qkv
                .checked_add(self.gdn_value_heads.checked_mul(2).ok_or_else(overflow)?)
                .and_then(|n| rows.checked_mul(n))
                .ok_or_else(overflow)?;
            rows.checked_mul(qkv)
                .and_then(|n| n.checked_mul(DType::F32.size()))
                .ok_or_else(overflow)?;
            recurrence
                .checked_mul(DType::F32.size())
                .ok_or_else(overflow)?;
        }
        let compact = compact_capture_ring();
        let ring_slots = capture_ring_slots(compact, rows).ok_or_else(overflow)?;
        let initial_slot = capture_ring_slot(compact, rows, 0);
        // Every GDN layer's slot has one geometry (the rollback launch relies
        // on it too); a pool needs it to carve equal windows.
        let (pool_state, pool_dtype) = (self.gdn[0].recurrent.numel(), self.gdn[0].recurrent.dtype);
        let pooled = compact
            && lane_reserve_trim()
            && self.gdn.iter().all(|layer| {
                layer.recurrent.numel() == pool_state && layer.recurrent.dtype == pool_dtype
            });
        let pool_elements = if pooled {
            let slots = capture_ring_pool_slots(rows, self.gdn.len()).ok_or_else(overflow)?;
            let elements = slots.checked_mul(pool_state).ok_or_else(overflow)?;
            elements
                .checked_mul(pool_dtype.size())
                .ok_or_else(overflow)?;
            elements
        } else {
            0
        };
        // Stage all allocations, following the snapshot arena's cleanup
        // convention. Allocation/copy/upload failure leaves live pointers,
        // ticket generation, and capacities untouched. `rings` holds owned
        // per-layer rings, or (pooled) non-owning windows of `pool`.
        let mut pool: Option<GpuTensor> = None;
        let mut rings: Vec<GpuTensor> = Vec::with_capacity(self.gdn.len());
        let mut inputs_all: Vec<GpuTensor> = Vec::with_capacity(self.gdn.len());
        let mut recurrence_all: Vec<GpuTensor> = Vec::with_capacity(self.gdn.len());
        let mut tail: Vec<GpuTensor> = Vec::with_capacity(2);
        let staged = (|| -> Result<(), StateError> {
            if pooled {
                pool = Some(
                    gpu.zeros(&[pool_elements], pool_dtype)
                        .map_err(StateError::Hip)?,
                );
            }
            for (index, layer) in self.gdn.iter().enumerate() {
                let state = layer.recurrent.numel();
                match pool.as_ref() {
                    Some(pool) => rings.push(
                        pool.sub_offset(CAPTURE_POOL_STRIDE * index * state, ring_slots * state),
                    ),
                    None => rings.push(
                        gpu.zeros(&[ring_slots * state], layer.recurrent.dtype)
                            .map_err(StateError::Hip)?,
                    ),
                }
                gpu.copy_d2d(
                    &layer.recurrent,
                    &rings
                        .last()
                        .expect("ring")
                        .sub_offset(initial_slot * state, state),
                    layer.recurrent.byte_size(),
                )
                .map_err(StateError::Hip)?;
                let qkv = layer.conv.numel() / conv_history;
                inputs_all.push(
                    gpu.zeros(&[rows * qkv], DType::F32)
                        .map_err(StateError::Hip)?,
                );
                recurrence_all.push(
                    gpu.zeros(&[rows * (qkv + 2 * self.gdn_value_heads)], DType::F32)
                        .map_err(StateError::Hip)?,
                );
            }
            let pointers: Vec<u8> = recurrence_all
                .iter()
                .zip(&rings)
                .flat_map(|(recurrence, ring)| [recurrence.buf.as_ptr(), ring.buf.as_ptr()])
                .flat_map(|pointer| (pointer as u64).to_ne_bytes())
                .collect();
            tail.push(
                gpu.zeros(&[output_elements], DType::F32)
                    .map_err(StateError::Hip)?,
            );
            tail.push(
                gpu.zeros(&[table_bytes], DType::Raw)
                    .map_err(StateError::Hip)?,
            );
            gpu.upload_stream_ordered(&tail.last().expect("pointer table").buf, &pointers)
                .map_err(StateError::Hip)?;
            Ok(())
        })();
        if let Err(error) = staged {
            let owned_rings = if pool.is_some() { Vec::new() } else { rings };
            for tensor in pool
                .into_iter()
                .chain(owned_rings)
                .chain(inputs_all)
                .chain(recurrence_all)
                .chain(tail)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        let mut tail = tail.into_iter();
        let output = tail.next().expect("discard output");
        let table = tail.next().expect("pointer table");
        let next: Vec<_> = rings
            .into_iter()
            .zip(inputs_all)
            .zip(recurrence_all)
            .map(|((ring, inputs), recurrence)| (ring, inputs, recurrence))
            .collect();
        // Captured graphs may bake the recurrent pointers replaced below: drop
        // captured execution state before publishing the new ring (real growth
        // only; no-ops returned above, staged failure never reaches here).
        gpu.invalidate_for_scratch_growth();
        self.snapshot_arena.invalidate();
        let previous_output = self.row_capture_output.replace((output, table));
        let previous_pool = std::mem::replace(&mut self.row_capture_pool, pool);
        let previous = std::mem::replace(&mut self.row_capture, next);
        let ringed = !previous.is_empty();
        let mut old_states = Vec::new();
        for (index, layer) in self.gdn.iter_mut().enumerate() {
            let state = layer.recurrent.numel();
            let old = std::mem::replace(
                &mut layer.recurrent,
                self.row_capture[index]
                    .0
                    .sub_offset(initial_slot * state, state),
            );
            if !ringed {
                old_states.push(old);
            }
        }
        self.row_capture_rows = rows;
        self.row_capture_compact = compact;
        self.gdn_live = 0;
        // Publish a complete coherent ring before releasing previous storage.
        // A previous pool owns its windows; otherwise each layer's ring is owned.
        let previous_rings_owned = previous_pool.is_none();
        let mut stale = old_states;
        stale.extend(previous_pool);
        for (ring, inputs, recurrence) in previous {
            if previous_rings_owned {
                stale.push(ring);
            }
            stale.push(inputs);
            stale.push(recurrence);
        }
        stale.extend(
            previous_output
                .into_iter()
                .flat_map(|(output, table)| [output, table]),
        );
        for tensor in stale {
            gpu.free_tensor(tensor).map_err(StateError::Hip)?;
        }
        #[cfg(feature = "lab")]
        self.lab
            .ring(crate::lab_window::LabRingEvent::Realloc { ring_rows: rows });
        Ok(())
    }

    /// Rows an armed verify captures (0 = this GPU's GDN route cannot).
    pub(crate) fn row_capture_rows(&self) -> usize {
        self.row_capture_rows
    }

    /// First ring slot an armed verify writes: the half without the live slot.
    fn capture_base(&self, live: usize) -> usize {
        if live < self.row_capture_rows {
            self.row_capture_rows
        } else {
            0
        }
    }

    /// Point every GDN layer's `recurrent` at ring slot `slot`.
    fn set_gdn_live(&mut self, slot: usize) {
        self.gdn_live = slot;
        let physical = capture_ring_slot(self.row_capture_compact, self.row_capture_rows, slot);
        for (layer, (ring, _, _)) in self.gdn.iter_mut().zip(&self.row_capture) {
            let state = layer.recurrent.numel();
            layer.recurrent = ring.sub_offset(physical * state, state);
        }
    }

    /// Retained inputs and a destination half for an armed verify. Only
    /// `base + rows - 1` receives a recurrent state; earlier slots are not a
    /// state tape. Prefix repair must re-run the captured recurrence inputs.
    pub(crate) fn gdn_row_capture(&self, slot: usize, rows: usize) -> Option<GdnRowCapture<'_>> {
        if !self.row_capture_armed || rows < 2 || rows > self.row_capture_rows {
            return None;
        }
        let state = self.gdn.get(slot)?.recurrent.numel();
        let base = self.capture_base(self.gdn_live);
        self.row_capture
            .get(slot)
            .map(|(ring, inputs, recurrence)| GdnRowCapture {
                states: ring.sub_offset(
                    capture_view_start(self.row_capture_compact, self.row_capture_rows, base, rows)
                        * state,
                    rows * state,
                ),
                inputs,
                recurrence,
            })
    }

    /// An armed `rows`-row verify forward succeeded: its last row's state is
    /// live.
    pub(crate) fn commit_row_capture(&mut self, rows: usize) {
        if self.row_capture_armed && rows >= 2 && rows <= self.row_capture_rows {
            let base = self.capture_base(self.gdn_live);
            #[cfg(feature = "lab")]
            let live_before = self.gdn_live;
            self.set_gdn_live(base + rows - 1);
            #[cfg(feature = "lab")]
            self.lab.ring(crate::lab_window::LabRingEvent::Verify {
                rows,
                ring_rows: self.row_capture_rows,
                live_before,
                base,
                half: u8::from(base != 0),
                live_after: self.gdn_live,
            });
        }
    }

    /// The `(full, raw, pooled, selected, position)` lengths QSA slot `slot`
    /// currently holds: the private per-layer cursor of a layer-major prompt
    /// window (the layer's own tile wrote them), read for a cut stage.
    pub(crate) fn qsa_cursor(&self, slot: usize) -> Option<(usize, usize, usize, usize, usize)> {
        self.qsa.get(slot).map(|q| {
            (
                q.full_len,
                q.raw_len,
                q.pooled_len,
                q.selected_len,
                q.position,
            )
        })
    }

    /// Whether every QSA layer's private cursor reached the end of a
    /// `rows`-row prompt window that began at the live global position, the
    /// state the single global commit of [`Self::commit_prompt_window`]
    /// publishes. Host only; nothing is mutated.
    pub(crate) fn validate_prompt_window(&self, rows: usize) -> Result<(), StateError> {
        let end = self
            .position
            .checked_add(rows)
            .filter(|&end| end <= self.max_seq_len)
            .ok_or(StateError::DimensionOverflow)?;
        for layer in &self.qsa {
            for (what, actual) in [
                ("QSA full_len at prompt commit", layer.full_len),
                ("QSA raw_len at prompt commit", layer.raw_len),
                ("QSA position at prompt commit", layer.position),
            ] {
                if actual != end {
                    return Err(StateError::Length {
                        what,
                        expected: end,
                        actual,
                    });
                }
            }
            if layer.pooled_len != end / self.qsa_compress {
                return Err(StateError::Length {
                    what: "QSA pooled_len at prompt commit",
                    expected: end / self.qsa_compress,
                    actual: layer.pooled_len,
                });
            }
        }
        Ok(())
    }

    /// The one global commit of a layer-major prompt window: the layers'
    /// private QSA cursors already hold the window end, so only the PLE token
    /// history, the global position and the row-capture slot move. The derived
    /// F16 / pooled mirrors are clamped to the committed marks exactly as the
    /// chunk-major commit does.
    pub(crate) fn commit_prompt_window(&mut self, tokens: &[u32]) -> Result<(), StateError> {
        self.validate_prompt_window(tokens.len())?;
        for layer in &self.qsa {
            layer.clamp_f16_valid();
        }
        for &token in tokens {
            self.ple_history.push(token);
        }
        self.commit_row_capture(tokens.len());
        self.position += tokens.len();
        Ok(())
    }

    /// Roll an armed `rows`-row verify forward back to its first `keep` rows,
    /// keeping the snapshot ticket active (the caller commits or restores
    /// it). The GDN recurrent state re-runs the kept rows' recurrence from
    /// the pre-verify state (bitwise the verify's own prefix) into the free
    /// half of the ring; convolution history comes from the verify's input
    /// rows, the PLE convolution history from `ple_normed` (the verify's PLE
    /// convolution input rows), QSA marks and PLE token history from the
    /// ticket plus `keep`.  Append arenas (QSA K/V, raw and pooled index keys)
    /// need no rollback: rows past the kept end are invisible until
    /// overwritten, and a pooled block is re-pooled by the row that completes
    /// it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rollback_rows_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
        keep: usize,
        rows: usize,
        tokens: &[u32],
        ple_normed: &GpuTensor,
        ple_history_rows: usize,
        conv_history: usize,
        compress: usize,
        budget: usize,
    ) -> Result<(), StateError> {
        self.snapshot_arena.validate_ticket(self, &snapshot)?;
        if keep == 0 || keep >= rows || rows > self.row_capture_rows || tokens.len() < keep {
            return Err(StateError::Length {
                what: "verify rollback rows",
                expected: rows,
                actual: keep,
            });
        }
        let start = self.snapshot_arena.position;
        let before = self
            .snapshot_arena
            .gdn_live
            .ok_or(StateError::SnapshotTicket)?;
        let base = self.capture_base(before);
        if let Some(first) = self.gdn.first() {
            let (output, table) = self
                .row_capture_output
                .as_ref()
                .ok_or(StateError::SnapshotTicket)?;
            // One launch re-runs every layer: they share one geometry.
            let (state, conv) = (first.recurrent.numel(), first.conv.numel());
            if self
                .gdn
                .iter()
                .any(|layer| layer.recurrent.numel() != state || layer.conv.numel() != conv)
            {
                return Err(StateError::Length {
                    what: "uniform GDN rollback geometry",
                    expected: state,
                    actual: 0,
                });
            }
            let qkv = conv / conv_history;
            let value_heads = self.gdn_value_heads;
            gated_delta_rollback_layers(
                gpu,
                &GatedDeltaRollbackLayers {
                    table,
                    discard: output,
                    format: self.gdn_format.storage(),
                    layers: self.gdn.len(),
                    rows,
                    keep,
                    from: capture_ring_slot(
                        self.row_capture_compact,
                        self.row_capture_rows,
                        before,
                    ),
                    to: capture_view_start(
                        self.row_capture_compact,
                        self.row_capture_rows,
                        base,
                        keep,
                    ),
                    qkv_width: qkv,
                    key_heads: (qkv - value_heads * GDN_HEAD_DIM) / (2 * GDN_HEAD_DIM),
                    value_heads,
                    position: start + keep - 1,
                },
            )
            .map_err(StateError::Hip)?;
        }
        self.reset_gdn_residual(gpu, GdnResidualTransition::PartialRollback)?;
        self.set_gdn_live(base + keep - 1);
        #[cfg(feature = "lab")]
        self.lab.ring(crate::lab_window::LabRingEvent::Rollback {
            rows,
            keep,
            ring_rows: self.row_capture_rows,
            live_before: before,
            base,
            half: u8::from(base != 0),
            live_after: self.gdn_live,
        });
        let arena = &self.snapshot_arena;
        let ple_rollback = match (&self.ple_conv, &arena.ple_conv) {
            (Some(live), Some(saved)) => {
                if ple_history_rows == 0 {
                    return Err(StateError::Length {
                        what: "PLE rollback history rows",
                        expected: 1,
                        actual: 0,
                    });
                }
                Some((live, saved))
            }
            (None, None) => None,
            _ => return Err(StateError::SnapshotShape),
        };
        let row_bytes = ple_rollback.map_or(0, |(live, _)| live.byte_size() / ple_history_rows);
        let mut copies =
            Vec::with_capacity(self.gdn.len() * conv_history + self.qsa.len() + ple_history_rows);
        // Convolution history slot `position % conv_history`: the kept verify
        // rows' inputs where they reach, the snapshot's history elsewhere.
        let rows_from = (start + keep).saturating_sub(conv_history).max(start);
        for (index, layer) in self.gdn.iter().enumerate() {
            let (_, inputs, _) = &self.row_capture[index];
            let channel_bytes = layer.conv.byte_size() / conv_history;
            for slot in 0..conv_history {
                let kept =
                    (rows_from..start + keep).find(|position| position % conv_history == slot);
                let (src, src_offset) = match kept {
                    Some(position) => (&inputs.buf, (position - start) * channel_bytes),
                    None => (&arena.conv[index].buf, slot * channel_bytes),
                };
                copies.push(CopyRegion {
                    dst: &layer.conv.buf,
                    dst_offset: slot * channel_bytes,
                    src,
                    src_offset,
                    bytes: channel_bytes,
                });
            }
        }
        for (index, layer) in self.qsa.iter().enumerate() {
            copies.push(whole(
                &arena.qsa_selected[index],
                &layer.selected_indices,
                layer.selected_indices.byte_size(),
            ));
        }
        if let Some((live, saved)) = ple_rollback {
            for row in 0..ple_history_rows {
                let source = keep + row;
                let (src, src_offset) = if source < ple_history_rows {
                    (&saved.buf, source * row_bytes)
                } else {
                    (&ple_normed.buf, (source - ple_history_rows) * row_bytes)
                };
                copies.push(CopyRegion {
                    dst: &live.buf,
                    dst_offset: row * row_bytes,
                    src,
                    src_offset,
                    bytes: row_bytes,
                });
            }
        }
        copy_regions(gpu, &copies).map_err(StateError::Hip)?;
        for (index, layer) in self.qsa.iter_mut().enumerate() {
            let mark = arena.qsa_marks[index];
            let position = mark.position + keep;
            let complete = position / compress;
            layer.full_len = position;
            layer.raw_len = position;
            layer.pooled_len = complete;
            layer.selected_len = ((budget / compress).min(complete) * compress + position
                - complete * compress)
                .min(layer.selected_capacity);
            layer.position = position;
            layer.clamp_f16_valid();
        }
        let mut history = arena.ple_history;
        for &token in &tokens[..keep] {
            history.push(token);
        }
        self.ple_history = history;
        self.position = start + keep;
        Ok(())
    }

    /// Invalidate derived rows before replacing F32 storage, or after a
    /// failed forward whose append may not have committed.
    pub(crate) fn invalidate_f16_arenas(&self) {
        for layer in &self.qsa {
            layer.invalidate_f16_arena();
        }
    }

    /// Clamp every F16 arena's valid prefix to its layer's committed rows (a
    /// forward failed after its attention converted rows the marks never
    /// committed).
    pub fn clamp_f16_valid(&self) {
        for layer in &self.qsa {
            layer.clamp_f16_valid();
        }
    }

    pub fn qsa_mut(&mut self, full_layer_index: usize) -> Option<&mut QsaGpuState> {
        self.qsa.get_mut(full_layer_index)
    }

    pub fn gdn_mut(&mut self, linear_layer_index: usize) -> Option<&mut GdnGpuState> {
        self.gdn.get_mut(linear_layer_index)
    }

    /// The ownership map: absolute layer <-> local slot of `gdn` / `qsa`.
    pub fn scope_map(&self) -> &Qwen4StateMap {
        &self.map
    }

    /// Local `gdn` slot of absolute layer `abs` (`None`: not owned here).
    pub fn gdn_local(&self, abs: usize) -> Option<usize> {
        self.map.gdn_local(abs)
    }

    /// Local `qsa` slot of absolute layer `abs` (`None`: not owned here).
    pub fn qsa_local(&self, abs: usize) -> Option<usize> {
        self.map.qsa_local(abs)
    }

    /// GDN layers of the whole model before this band's first layer.
    pub fn gdn_base(&self) -> usize {
        self.map.gdn_base()
    }

    /// QSA layers of the whole model before this band's first layer.
    pub fn qsa_base(&self) -> usize {
        self.map.qsa_base()
    }

    /// The GDN state of absolute layer `abs`.
    pub fn gdn_by_layer(&self, abs: usize) -> Option<&GdnGpuState> {
        self.gdn.get(self.map.gdn_local(abs)?)
    }

    pub fn gdn_by_layer_mut(&mut self, abs: usize) -> Option<&mut GdnGpuState> {
        let slot = self.map.gdn_local(abs)?;
        self.gdn.get_mut(slot)
    }

    /// The QSA state of absolute layer `abs`.
    pub fn qsa_by_layer(&self, abs: usize) -> Option<&QsaGpuState> {
        self.qsa.get(self.map.qsa_local(abs)?)
    }

    pub fn qsa_by_layer_mut(&mut self, abs: usize) -> Option<&mut QsaGpuState> {
        let slot = self.map.qsa_local(abs)?;
        self.qsa.get_mut(slot)
    }

    /// A verify ring is in effect: allocated per owned GDN layer, or, for a
    /// band without GDN layers (nothing to allocate), requested rows only.
    fn row_capture_ringed(&self) -> bool {
        !self.row_capture.is_empty() || (self.gdn.is_empty() && self.row_capture_rows != 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QsaMark {
    full_len: usize,
    raw_len: usize,
    pooled_len: usize,
    partial_len: usize,
    selected_len: usize,
    position: usize,
}

#[derive(Debug, Clone, Copy)]
struct QsaMarkCapacity {
    full: usize,
    raw: usize,
    pooled: usize,
    partial: usize,
    selected: usize,
    position: usize,
}

fn validate_qsa_mark(mark: &QsaMark, capacity: &QsaMarkCapacity) -> Result<(), StateError> {
    if mark.full_len > capacity.full
        || mark.raw_len > capacity.raw
        || mark.pooled_len > capacity.pooled
        || mark.partial_len > capacity.partial
        || mark.selected_len > capacity.selected
        || mark.position > capacity.position
    {
        return Err(StateError::SnapshotLength);
    }
    Ok(())
}

#[derive(Debug)]
pub enum StateError {
    Config(String),
    Hip(hip_bridge::HipError),
    DimensionOverflow,
    InvalidCapacity,
    Length {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    AllocationBookkeeping,
    SnapshotShape,
    SnapshotLength,
    SnapshotBusy,
    SnapshotInactive,
    SnapshotTicket,
    /// The state was built without a snapshot mirror ([`StateMirror::Absent`],
    /// an AR lane): it can be stepped and reset, never snapshotted, restored
    /// or armed for a verify rollback.
    MirrorAbsent,
    /// A QSA context request above the tokens the state can cover.
    ContextCapacity {
        required: usize,
        admitted: usize,
    },
    /// QSA context growth requested while a graph is capturing.
    GrowthDuringCapture,
    /// A VMM QSA context arena is not registered with its GPU.
    VmmOwner,
    MapPlan(KvChunkPlanError),
    /// Mapping `bytes` more QSA context bytes at `offset` failed.
    MapGrowth {
        offset: usize,
        bytes: usize,
        error: hip_bridge::HipError,
    },
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 state config: {message}"),
            Self::Hip(error) => write!(f, "Qwen4 state HIP: {error}"),
            Self::DimensionOverflow => write!(f, "Qwen4 state dimension overflow"),
            Self::InvalidCapacity => write!(f, "Qwen4 state capacity is invalid"),
            Self::Length {
                what,
                expected,
                actual,
            } => write!(f, "{what}: expected {expected}, got {actual}"),
            Self::AllocationBookkeeping => write!(f, "Qwen4 state allocation bookkeeping failure"),
            Self::SnapshotShape => write!(f, "Qwen4 snapshot shape mismatch"),
            Self::SnapshotLength => write!(f, "Qwen4 snapshot mark exceeds state capacity"),
            Self::SnapshotBusy => write!(f, "Qwen4 snapshot arena already has an active ticket"),
            Self::SnapshotInactive => write!(f, "Qwen4 snapshot ticket is inactive or consumed"),
            Self::SnapshotTicket => {
                write!(f, "Qwen4 snapshot ticket does not belong to this state")
            }
            Self::MirrorAbsent => write!(f, "Qwen4 state has no snapshot mirror"),
            Self::ContextCapacity { required, admitted } => write!(
                f,
                "Qwen4 QSA context needs {required} tokens but the state covers {admitted}"
            ),
            Self::GrowthDuringCapture => {
                write!(f, "Qwen4 QSA context growth requested during graph capture")
            }
            Self::VmmOwner => write!(f, "Qwen4 QSA context arena is not a registered VMM owner"),
            Self::MapPlan(error) => write!(f, "Qwen4 QSA context map plan: {error}"),
            Self::MapGrowth {
                offset,
                bytes,
                error,
            } => write!(
                f,
                "Qwen4 QSA context map of {bytes} bytes at offset {offset} failed: {error}"
            ),
        }
    }
}

impl std::error::Error for StateError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compact capture ring (`rows + 1` slots) is observationally the
    /// `2 * rows`-slot ring: replaying random verify commits and rollbacks
    /// on both layouts, with the kernels modeled as "write only view slot
    /// n - 1", the live value always matches, no write touches the live or
    /// the rollback-source slot, and every view stays inside the ring.
    #[test]
    fn compact_capture_ring_matches_the_double_ring() {
        for rows in 2..=hipfire_config::QWEN4_VERIFY_MAX_ROWS.min(16) {
            assert_eq!(capture_ring_slots(false, rows), Some(2 * rows));
            assert_eq!(capture_ring_slots(true, rows), Some(rows + 1));
            let mut rings = [vec![0u64; 2 * rows], vec![0u64; rows + 1]];
            let compact = [false, true];
            // Initial live slot 0 holds value 1 in both layouts.
            let mut live = [0usize; 2];
            for k in 0..2 {
                let slot = capture_ring_slot(compact[k], rows, 0);
                rings[k][slot] = 1;
            }
            let capture_base = |live: usize| if live < rows { rows } else { 0 };
            let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ rows as u64;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            for step in 0..400u64 {
                let value = 1000 + step;
                let verify_rows = 2 + (next() as usize) % (rows - 1);
                let rollback = next() % 2 == 0 && verify_rows > 2;
                let keep = 1 + (next() as usize) % (verify_rows - 1);
                let before = live[0];
                assert_eq!(live[0], live[1]);
                let base = capture_base(before);
                for k in 0..2 {
                    let n = verify_rows;
                    let start = capture_view_start(compact[k], rows, base, n);
                    assert!(start + n <= rings[k].len(), "view out of ring");
                    let written = start + n - 1;
                    assert_ne!(written, capture_ring_slot(compact[k], rows, before));
                    rings[k][written] = value;
                }
                let after_verify = base + verify_rows - 1;
                for k in 0..2 {
                    live[k] = after_verify;
                }
                if rollback {
                    // Rollback re-runs `keep` rows from the pre-verify slot
                    // `before` into the half `base`.
                    for k in 0..2 {
                        let from = capture_ring_slot(compact[k], rows, before);
                        let to = capture_view_start(compact[k], rows, base, keep);
                        assert!(to + keep <= rings[k].len());
                        let written = to + keep - 1;
                        assert_ne!(written, from);
                        rings[k][written] = rings[k][from] + 7;
                        live[k] = base + keep - 1;
                    }
                }
                for k in 0..2 {
                    let slot = capture_ring_slot(compact[k], rows, live[k]);
                    assert_eq!(
                        rings[k][slot],
                        rings[1 - k][capture_ring_slot(compact[1 - k], rows, live[1 - k])]
                    );
                }
            }
        }
    }

    /// The shared capture-ring pool (`HIPFIRE_QWEN4_LANE_RESERVE_TRIM`) is observationally the
    /// per-layer compact ring: replaying random verify commits and rollbacks over every layer of
    /// a state with the kernels modeled as "read the live / source slot, write only view slot
    /// n - 1", each layer's live value matches its own private ring, every layer's views stay
    /// inside the pool, and no pool slot is ever written by two layers (so no layer's live or
    /// rollback-source slot can be clobbered by another layer's write).
    #[test]
    fn shared_capture_pool_matches_per_layer_rings() {
        for layers in [1usize, 2, 7, 36] {
            for rows in 2..=hipfire_config::QWEN4_VERIFY_MAX_ROWS.min(16) {
                let window = capture_ring_slots(true, rows).unwrap();
                let pool_slots = capture_ring_pool_slots(rows, layers).unwrap();
                assert_eq!(pool_slots, CAPTURE_POOL_STRIDE * (layers - 1) + rows + 1);
                assert_eq!(
                    capture_ring_total_slots(true, true, rows, layers),
                    Some(pool_slots)
                );
                assert_eq!(
                    capture_ring_total_slots(true, false, rows, layers),
                    Some(layers * window)
                );
                assert_eq!(
                    capture_ring_total_slots(false, true, rows, layers),
                    Some(layers * 2 * rows),
                    "the double ring is never pooled"
                );
                assert!(pool_slots <= layers * window);
                let mut pool = vec![0u64; pool_slots];
                let mut writer: Vec<Option<usize>> = vec![None; pool_slots];
                let mut private: Vec<Vec<u64>> = (0..layers).map(|_| vec![0u64; window]).collect();
                let at = |layer: usize, slot: usize| {
                    assert!(slot < window, "window slot in the layer window");
                    CAPTURE_POOL_STRIDE * layer + slot
                };
                let tag = |layer: usize, value: u64| value * 100 + layer as u64;
                let mut live = 0usize;
                for layer in 0..layers {
                    let slot = capture_ring_slot(true, rows, 0);
                    private[layer][slot] = tag(layer, 1);
                    pool[at(layer, slot)] = tag(layer, 1);
                    writer[at(layer, slot)] = Some(layer);
                }
                let capture_base = |live: usize| if live < rows { rows } else { 0 };
                let mut rng = 0xD1B5_4A32_D192_ED03u64 ^ (rows * 131 + layers) as u64;
                let mut next = move || {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    rng
                };
                for step in 0..300u64 {
                    let value = 1000 + step;
                    let verify_rows = 2 + (next() as usize) % (rows - 1);
                    let rollback = next() % 2 == 0 && verify_rows > 2;
                    let keep = 1 + (next() as usize) % (verify_rows - 1);
                    let before = live;
                    let base = capture_base(before);
                    for layer in 0..layers {
                        // The n-row view starts at `start`, ends inside the window, so inside the
                        // pool; the kernel writes only its last slot.
                        let start = capture_view_start(true, rows, base, verify_rows);
                        assert!(start + verify_rows <= window, "view out of window");
                        assert!(
                            CAPTURE_POOL_STRIDE * layer + start + verify_rows <= pool_slots,
                            "view out of pool"
                        );
                        let written = start + verify_rows - 1;
                        let src = capture_ring_slot(true, rows, before);
                        assert_ne!(written, src);
                        let abs = at(layer, written);
                        assert!(writer[abs].is_none_or(|owner| owner == layer));
                        writer[abs] = Some(layer);
                        // The live source slot still holds this layer's own value.
                        assert_eq!(pool[at(layer, src)], private[layer][src]);
                        private[layer][written] = tag(layer, value);
                        pool[abs] = tag(layer, value);
                    }
                    live = base + verify_rows - 1;
                    if rollback {
                        for layer in 0..layers {
                            let from = capture_ring_slot(true, rows, before);
                            let to = capture_view_start(true, rows, base, keep);
                            assert!(to + keep <= window);
                            assert!(CAPTURE_POOL_STRIDE * layer + to + keep <= pool_slots);
                            let written = to + keep - 1;
                            assert_ne!(written, from);
                            assert_eq!(pool[at(layer, from)], private[layer][from]);
                            let abs = at(layer, written);
                            assert!(writer[abs].is_none_or(|owner| owner == layer));
                            writer[abs] = Some(layer);
                            let result = private[layer][from] + 7;
                            private[layer][written] = result;
                            pool[abs] = result;
                        }
                        live = base + keep - 1;
                    }
                    for layer in 0..layers {
                        let slot = capture_ring_slot(true, rows, live);
                        assert_eq!(pool[at(layer, slot)], private[layer][slot]);
                    }
                }
            }
        }
    }

    #[test]
    fn gdn_q8ef_reset_lifecycle() {
        let config = crate::config::compact_test_config();
        let scope = Qwen4BandScope::single(&config);
        let map = Qwen4StateMap::new(&config, &scope).unwrap();
        let format = |gdn| Qwen4StateFormat {
            gdn,
            ..Qwen4StateFormat::F32
        };
        let plan = |gdn| {
            let format = format(gdn);
            let geometry = StateGeometry::new(&config, &format, 1024).unwrap();
            state_tensor_plan(&config, &format, 1024, &geometry, &map).unwrap()
        };
        let q8 = plan(GdnStateFormat::Q8);
        let ef = plan(GdnStateFormat::Q8Ef);
        let residuals: Vec<_> = ef
            .iter()
            .filter(|part| part.key.ends_with(".ef_residual"))
            .collect();
        assert_eq!(residuals.len(), 36);
        assert!(residuals.iter().all(|part| part.dtype == DType::F16
            && part.elements == 48 * 128 * 128
            && !part.context_arena));
        for plain in [GdnStateFormat::F32, GdnStateFormat::Q8] {
            assert!(!plain.has_residual());
            assert!(plan(plain)
                .iter()
                .all(|part| !part.key.ends_with(".ef_residual")));
        }
        // Removing only the live companions leaves exactly the old allocation
        // plan, so mirror and serialized recurrent extents cannot grow with EF.
        let stored: Vec<_> = ef
            .iter()
            .filter(|part| !part.key.ends_with(".ef_residual"))
            .cloned()
            .collect();
        assert_eq!(stored, q8);
        assert_eq!(
            residuals
                .iter()
                .map(|part| part.elements * part.dtype.size())
                .sum::<usize>(),
            54 * 1024 * 1024
        );
        let context = Qwen4ContextCommit::legacy(1024);
        let live = |gdn| Qwen4State::device_bytes(&config, format(gdn), &context).unwrap();
        assert_eq!(
            live(GdnStateFormat::Q8Ef) - live(GdnStateFormat::Q8),
            54 * 1024 * 1024
        );
        for rows in [4, 8, 32, hipfire_config::QWEN4_VERIFY_MAX_ROWS] {
            let ring = |gdn| Qwen4State::row_capture_bytes(&config, gdn, rows).unwrap();
            assert_eq!(ring(GdnStateFormat::Q8Ef), ring(GdnStateFormat::Q8));
            assert_eq!(
                live(GdnStateFormat::Q8Ef) + ring(GdnStateFormat::Q8Ef)
                    - live(GdnStateFormat::Q8)
                    - ring(GdnStateFormat::Q8),
                54 * 1024 * 1024,
            );
            for native_mtp in [false, true] {
                assert_eq!(
                    crate::bundle::session_snapshot_bytes(
                        &config,
                        format(GdnStateFormat::Q8Ef),
                        native_mtp,
                        rows
                    ),
                    crate::bundle::session_snapshot_bytes(
                        &config,
                        format(GdnStateFormat::Q8),
                        native_mtp,
                        rows
                    ),
                );
            }
        }
        // Every PP/lane owner charges only its own live residual, not another
        // owner's history; together PP bands equal the whole-model owner.
        let pp_residual: usize = pp_scopes(&[24, 24])
            .iter()
            .map(|band| {
                Qwen4State::device_bytes_scoped(
                    &config,
                    format(GdnStateFormat::Q8Ef),
                    &context,
                    band,
                )
                .unwrap()
                    - Qwen4State::device_bytes_scoped(
                        &config,
                        format(GdnStateFormat::Q8),
                        &context,
                        band,
                    )
                    .unwrap()
            })
            .sum();
        assert_eq!(pp_residual, 54 * 1024 * 1024);
    }

    #[test]
    fn gdn_q8ef_reset_lifecycle_events() {
        use GdnResidualTransition::{Keep, PartialRollback, Reset, Restore};
        for (label, event, clears) in [
            ("initialize", Reset, true),
            ("reset", Reset, true),
            ("keep0", Restore, true),
            ("partial rollback", PartialRollback, true),
            ("transaction restore", Restore, true),
            ("session restore", Restore, true),
            ("prefix restore", Restore, true),
            ("disk restore", Restore, true),
            ("lane reinit", Reset, true),
            ("all accept", Keep, false),
            ("uninterrupted AR", Keep, false),
            ("uninterrupted prefill", Keep, false),
        ] {
            // Two owners start with distinct residuals. A reset of one owner
            // never writes the other; retained histories keep their exact bits.
            let mut residual = [0x3800u16, 0xb800];
            let other = [0x3400u16, 0xb400];
            if gdn_residual_reset_required(GdnStateFormat::Q8Ef, event) {
                residual.fill(0);
            }
            assert_eq!(
                residual,
                if clears { [0, 0] } else { [0x3800, 0xb800] },
                "{label}"
            );
            assert_eq!(other, [0x3400, 0xb400], "{label}");
            for plain in [GdnStateFormat::F32, GdnStateFormat::Q8] {
                assert!(!gdn_residual_reset_required(plain, event), "{label}");
            }
        }
    }

    #[test]
    fn gdn_q8ef_reset_lifecycle_cache_abi() {
        use hipfire_runtime::session_cache::SessionRoute;
        let config = crate::config::compact_test_config();
        let map = Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).unwrap();
        let format = |gdn| Qwen4StateFormat {
            gdn,
            ..Qwen4StateFormat::F32
        };
        let scope =
            |gdn| crate::bundle::session_scope(SessionRoute::Ar, 256, format(gdn), false, None);
        let ef_scope = scope(GdnStateFormat::Q8Ef);
        assert!(ef_scope.contains("q8_ef-reset"), "{ef_scope}");
        let ef_tag = gdn_session_tag(map.session_tag(), GdnStateFormat::Q8Ef);
        assert!(validate_gdn_session_tag(map.session_tag(), GdnStateFormat::Q8Ef, ef_tag).is_ok());
        for plain in [GdnStateFormat::F32, GdnStateFormat::Q8] {
            assert_ne!(ef_scope, scope(plain));
            let tag = gdn_session_tag(map.session_tag(), plain);
            assert!(matches!(
                validate_gdn_session_tag(map.session_tag(), plain, ef_tag),
                Err(StateError::SnapshotShape),
            ));
            assert!(matches!(
                validate_gdn_session_tag(map.session_tag(), GdnStateFormat::Q8Ef, tag),
                Err(StateError::SnapshotShape),
            ));
        }
        assert_ne!(
            digest_key(&map, DigestRow::GdnResidual(0), true),
            digest_key(&map, DigestRow::GdnRecurrent(0), true),
        );
    }

    #[test]
    fn gdn_q8ef_reset_lifecycle_overflow() {
        let mut config = crate::config::compact_test_config();
        config.linear_num_value_heads = usize::MAX;
        let format = Qwen4StateFormat {
            gdn: GdnStateFormat::Q8Ef,
            ..Qwen4StateFormat::F32
        };
        assert!(matches!(
            StateGeometry::new(&config, &format, 1024),
            Err(StateError::DimensionOverflow)
        ));
        // The plain Q8 slot fits, but its separate F16 companion does not.
        config.linear_num_value_heads = usize::MAX / (128 * 128 * 2) + 1;
        assert!(matches!(
            StateGeometry::new(&config, &format, 1024),
            Err(StateError::DimensionOverflow)
        ));
    }

    #[test]
    fn gdn_q8ef_reset_lifecycle_stock_fp8_refusal() {
        let flash = crate::config::compact_test_config();
        assert!(validate_gdn_profile(&flash, GdnStateFormat::Q8Ef).is_ok());
        let stock = flash.into_ar_only().unwrap();
        let error = validate_gdn_profile(&stock, GdnStateFormat::Q8Ef).unwrap_err();
        assert!(error.to_string().contains("stock-FP8") && error.to_string().contains("q8_ef"));
        assert!(validate_gdn_profile(&stock, GdnStateFormat::F32).is_ok());
    }

    /// The AR-only (stock-FP8) profile pins the F32 GDN recurrent state:
    /// `auto`/unset/fp32 resolve to F32 and an explicit q8 is refused; the
    /// checkpoint profile keeps its Q8 default.
    #[test]
    fn stock_fp8_profile_pins_f32_gdn_state() {
        let flash = crate::config::compact_test_config();
        assert_eq!(resolve_gdn_format("", &flash).unwrap(), GdnStateFormat::Q8);
        assert_eq!(
            resolve_gdn_format("auto", &flash).unwrap(),
            GdnStateFormat::Q8
        );
        let stock = flash.into_ar_only().unwrap();
        for request in ["", "auto", " AUTO ", "fp32", "f32"] {
            assert_eq!(
                resolve_gdn_format(request, &stock).unwrap(),
                GdnStateFormat::F32,
                "{request:?}"
            );
        }
        for request in ["q8", "Q8", "int8"] {
            let error = resolve_gdn_format(request, &stock).unwrap_err();
            assert!(
                error.contains("stock-FP8") && error.contains("F32"),
                "{error}"
            );
        }
        assert!(resolve_gdn_format("bf16", &stock).is_err());
    }

    /// The per-architecture GDN `auto`: exact gfx1151 with the ordinary
    /// profile and 128x128 heads resolves F32 (KLD-chosen); every other
    /// architecture keeps Q8; non-128 geometry and the stock-FP8 profile are
    /// F32 everywhere. Explicit values are unchanged on every architecture.
    #[test]
    fn gdn_auto_precision_policy() {
        let ordinary = crate::config::compact_test_config();
        let stock = ordinary.clone().into_ar_only().unwrap();
        let mut narrow = ordinary.clone();
        narrow.linear_key_head_dim = 64;
        assert!(
            !GdnStateFormat::Q8.supports(narrow.linear_key_head_dim, narrow.linear_value_head_dim)
        );
        let profiles = [
            ("ordinary", &ordinary),
            ("stock-fp8", &stock),
            ("non-128", &narrow),
        ];
        let auto_requests = ["", "auto", " AUTO "];
        for arch in ["gfx1151", "gfx1100", "gfx1201", "gfx942", "gfx1030"] {
            for (name, config) in profiles {
                let auto_expected = if name == "ordinary" && arch != "gfx1151" {
                    GdnStateFormat::Q8
                } else {
                    GdnStateFormat::F32
                };
                for request in auto_requests {
                    assert_eq!(
                        resolve_gdn_format_for_arch(request, config, arch, false),
                        Ok(auto_expected),
                        "{request:?} on {arch} / {name}"
                    );
                }
                for request in ["fp32", "f32"] {
                    assert_eq!(
                        resolve_gdn_format_for_arch(request, config, arch, false),
                        Ok(GdnStateFormat::F32),
                        "{request:?} on {arch} / {name}"
                    );
                }
                for request in ["q8", "int8"] {
                    let result = resolve_gdn_format_for_arch(request, config, arch, false);
                    if name == "ordinary" {
                        assert_eq!(
                            result,
                            Ok(GdnStateFormat::Q8),
                            "{request:?} on {arch} / {name}"
                        );
                    } else {
                        assert!(result.is_err(), "{request:?} on {arch} / {name}");
                    }
                }
                assert!(
                    resolve_gdn_format_for_arch("bf16", config, arch, false).is_err(),
                    "bf16 on {arch} / {name}"
                );
            }
        }
        // The arch-agnostic export keeps the old meaning.
        assert_eq!(resolve_gdn_format("", &ordinary), Ok(GdnStateFormat::Q8));
    }

    /// `q8_ef` has no inline WY kernel (none has run on a GPU), so a load that
    /// asks for both it and `HIPFIRE_QWEN4_GDN_Q8_INLINE=1` is refused on every
    /// architecture; the inline flag changes no other request's result.
    #[test]
    fn q8_ef_refuses_inline_wy_developer_route() {
        const REFUSAL: &str = "qwen4: state_quant q8_ef does not support the inline WY developer \
                               route (HIPFIRE_QWEN4_GDN_Q8_INLINE=1) yet; unset one of them";
        let ordinary = crate::config::compact_test_config();
        for arch in ["gfx1151", "gfx1100", "gfx1201", "gfx942", "gfx1030"] {
            assert_eq!(
                resolve_gdn_format_for_arch("q8_ef", &ordinary, arch, true),
                Err(REFUSAL.to_string()),
                "q8_ef + inline on {arch}"
            );
            assert_eq!(
                resolve_gdn_format_for_arch("q8_ef", &ordinary, arch, false),
                Ok(GdnStateFormat::Q8Ef),
                "q8_ef on {arch}"
            );
            assert_eq!(
                resolve_gdn_format_for_arch("q8", &ordinary, arch, true),
                Ok(GdnStateFormat::Q8),
                "q8 + inline on {arch}"
            );
            assert_eq!(
                resolve_gdn_format_for_arch("fp32", &ordinary, arch, true),
                Ok(GdnStateFormat::F32),
                "fp32 + inline on {arch}"
            );
            for request in ["", "auto"] {
                assert_eq!(
                    resolve_gdn_format_for_arch(request, &ordinary, arch, true),
                    resolve_gdn_format_for_arch(request, &ordinary, arch, false),
                    "{request:?} + inline on {arch}"
                );
            }
        }
    }

    #[test]
    fn f16_arena_bytes_are_counted_for_legacy_and_vmm_states() {
        let config = crate::config::compact_test_config();
        let arena = Qwen4StateFormat {
            qsa_f16_arena: true,
            ..Qwen4StateFormat::F32
        };
        for backend in [Qwen4KvBackend::Legacy, Qwen4KvBackend::Vmm] {
            let context = Qwen4ContextCommit::new(backend, 8193, 1025, 2 << 20);
            let plain = context
                .committed_layer_bytes(&config, QsaKvFormat::F32)
                .unwrap();
            let with = context
                .committed_layer_bytes(&config, arena.layout())
                .unwrap();
            assert!(with > plain);
            assert_eq!(
                Qwen4State::device_bytes(&config, arena, &context).unwrap()
                    - Qwen4State::device_bytes(&config, Qwen4StateFormat::F32, &context).unwrap(),
                (with - plain) * config.n_full_layers(),
            );
            let logical = config
                .qsa_context_arena_bytes(8193, arena.layout())
                .unwrap()
                - config
                    .qsa_context_arena_bytes(8193, QsaKvFormat::F32)
                    .unwrap();
            assert_eq!(
                logical,
                8196 * arena.layout().f16_arena_bytes_per_token(2, 256)
            );
            if backend == Qwen4KvBackend::Legacy {
                assert_eq!(with - plain, logical);
            }
        }
    }

    #[test]
    fn pooled_bf16_mirror_bytes_are_counted_for_legacy_and_vmm_states() {
        let config = crate::config::compact_test_config();
        let mirror = Qwen4StateFormat {
            qsa_pooled_bf16: true,
            ..Qwen4StateFormat::F32
        };
        let layout = mirror.layout();
        assert_eq!(QsaStateLayout::F32.pooled_bf16_bytes_per_block(128), 0);
        assert_eq!(layout.pooled_bf16_bytes_per_block(128), 256);
        for backend in [Qwen4KvBackend::Legacy, Qwen4KvBackend::Vmm] {
            let context = Qwen4ContextCommit::new(backend, 8193, 1025, 2 << 20);
            let plain = context
                .committed_layer_bytes(&config, QsaKvFormat::F32)
                .unwrap();
            let with = context.committed_layer_bytes(&config, layout).unwrap();
            assert!(with > plain);
            assert_eq!(
                Qwen4State::device_bytes(&config, mirror, &context).unwrap()
                    - Qwen4State::device_bytes(&config, Qwen4StateFormat::F32, &context).unwrap(),
                (with - plain) * config.n_full_layers(),
            );
            // One 256-byte BF16 block per `ceil(tokens / 4)` pooled block.
            let logical = config.qsa_context_arena_bytes(8193, layout).unwrap()
                - config
                    .qsa_context_arena_bytes(8193, QsaKvFormat::F32)
                    .unwrap();
            assert_eq!(logical, 2049 * 256);
            if backend == Qwen4KvBackend::Legacy {
                assert_eq!(with - plain, logical);
            }
        }
        // Per-layer logical deltas at 32K and 256K (x12 QSA layers: 24 MiB and
        // 192 MiB); the derived mirror is not snapshot payload.
        for (tokens, per_layer) in [(32_768usize, 2usize << 20), (262_144, 16 << 20)] {
            assert_eq!(
                config.qsa_context_arena_bytes(tokens, layout).unwrap()
                    - config
                        .qsa_context_arena_bytes(tokens, QsaKvFormat::F32)
                        .unwrap(),
                per_layer
            );
            assert_eq!(per_layer * config.n_full_layers(), tokens / 4 * 256 * 12);
        }
    }

    fn allocation_formats() -> Vec<(&'static str, Qwen4StateFormat)> {
        let base = Qwen4StateFormat::F32;
        vec![
            ("f32", base),
            (
                "q8",
                Qwen4StateFormat {
                    gdn: GdnStateFormat::Q8,
                    ..base
                },
            ),
            (
                "q8_ef",
                Qwen4StateFormat {
                    gdn: GdnStateFormat::Q8Ef,
                    ..base
                },
            ),
            (
                "fp8",
                Qwen4StateFormat {
                    qsa: QsaKvFormat::Fp8,
                    gdn: GdnStateFormat::Q8,
                    ..base
                },
            ),
            (
                "f16_arena",
                Qwen4StateFormat {
                    qsa_f16_arena: true,
                    ..base
                },
            ),
            (
                "pooled_bf16",
                Qwen4StateFormat {
                    qsa_pooled_bf16: true,
                    ..base
                },
            ),
            (
                "arena_and_mirror_q8_ef",
                Qwen4StateFormat {
                    gdn: GdnStateFormat::Q8Ef,
                    qsa_f16_arena: true,
                    qsa_pooled_bf16: true,
                    ..base
                },
            ),
        ]
    }

    fn allocation_contexts() -> Vec<Qwen4ContextCommit> {
        let granularity = 2 << 20;
        vec![
            Qwen4ContextCommit::legacy(1024),
            Qwen4ContextCommit::legacy(8193),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 8193, 1025, granularity),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 1536, granularity),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 8192, granularity),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 100, 8192, granularity),
            Qwen4ContextCommit {
                backend: Qwen4KvBackend::Vmm,
                max_seq: 262_144,
                tokens: 32_769,
                granularity,
            },
        ]
    }

    /// The per-allocation inventory sums to the closed-form byte query for
    /// every state format and legacy/VMM context commit (including the F16
    /// and pooled BF16 arenas and the live-only EF residual), under both
    /// snapshot-mirror choices; the unsuffixed queries are the
    /// snapshot-capable state.
    #[test]
    fn allocation_sizes_sum_to_device_bytes() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        for (name, format) in allocation_formats() {
            for context in allocation_contexts() {
                for mirror in [StateMirror::Snapshot, StateMirror::Absent] {
                    let sizes = Qwen4State::device_allocation_sizes_with_mirror(
                        &config, format, &context, mirror,
                    )
                    .unwrap_or_else(|| panic!("{name} {context:?} {mirror:?} admits"));
                    let total = sizes
                        .iter()
                        .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
                        .expect("inventory sum fits");
                    assert_eq!(
                        Some(total),
                        Qwen4State::device_bytes_with_mirror(
                            &config, format, &context, &single, mirror
                        )
                        .map(|bytes| bytes as u64),
                        "{name} {context:?} {mirror:?}"
                    );
                    assert!(
                        sizes.iter().all(|&bytes| bytes > 0),
                        "{name} {context:?} {mirror:?}"
                    );
                }
                assert_eq!(
                    Qwen4State::device_allocation_sizes(&config, format, &context),
                    Qwen4State::device_allocation_sizes_with_mirror(
                        &config,
                        format,
                        &context,
                        StateMirror::Snapshot
                    ),
                    "{name} {context:?}"
                );
                assert_eq!(
                    Qwen4State::device_bytes(&config, format, &context),
                    Qwen4State::device_bytes_with_mirror(
                        &config,
                        format,
                        &context,
                        &single,
                        StateMirror::Snapshot
                    ),
                    "{name} {context:?}"
                );
            }
        }
    }

    /// Compared with the owner's own tensor plan: under legacy storage the
    /// inventory is each planned tensor once plus exactly one snapshot-arena
    /// mirror per in-place tensor (GDN recurrent/conv, QSA partial K/V,
    /// selected indices, the raw-index circular block, PLE conv, hyper
    /// feedback); the EF residual and the append-only/derived context arenas
    /// have no mirror, and no alias is charged twice.
    #[test]
    fn allocation_sizes_are_the_state_plan_plus_one_mirror_per_in_place_tensor() {
        let config = crate::config::compact_test_config();
        let map = Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).unwrap();
        let ratio = config.indexer_compress_ratio;
        let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
        for (name, format) in allocation_formats() {
            for max_seq in [1024usize, 8193] {
                let geometry = StateGeometry::new(&config, &format, max_seq).unwrap();
                let plan = state_tensor_plan(&config, &format, max_seq, &geometry, &map).unwrap();
                let mut expected = Vec::new();
                let mut live_only = Vec::new();
                for spec in &plan {
                    let bytes = (spec.elements * spec.dtype.size()) as u64;
                    expected.push(bytes);
                    live_only.push(bytes);
                    match spec.key.rsplit('.').next().unwrap() {
                        "recurrent" | "conv" | "partial_keys" | "partial_values"
                        | "selected_indices" | "ple_conv" | "hyper_feedback" => {
                            expected.push(bytes)
                        }
                        "raw_index_keys" => expected.push(
                            (ratio * raw_width * format.layout().index_dtype().size()) as u64,
                        ),
                        _ => {}
                    }
                }
                expected.retain(|&bytes| bytes > 0);
                expected.sort_unstable();
                live_only.retain(|&bytes| bytes > 0);
                live_only.sort_unstable();
                let context = Qwen4ContextCommit::legacy(max_seq);
                let mut sizes =
                    Qwen4State::device_allocation_sizes(&config, format, &context).unwrap();
                sizes.sort_unstable();
                assert_eq!(sizes, expected, "{name} {max_seq}");
                // Without the mirror: each planned tensor once, nothing else.
                let mut absent = Qwen4State::device_allocation_sizes_with_mirror(
                    &config,
                    format,
                    &context,
                    StateMirror::Absent,
                )
                .unwrap();
                absent.sort_unstable();
                assert_eq!(absent, live_only, "{name} {max_seq} absent");
            }
        }
    }

    /// Under VMM each context arena is its own mapped-pages entry: they sum
    /// to the owner's committed layer bytes per QSA layer, and the fixed
    /// state is unchanged by the backend.
    #[test]
    fn vmm_allocation_sizes_charge_each_context_arena_its_mapped_pages() {
        let config = crate::config::compact_test_config();
        let layers = config.n_full_layers() as u64;
        for (name, format) in allocation_formats() {
            let legacy = Qwen4ContextCommit::legacy(262_144);
            let vmm = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 1536, 2 << 20);
            let total = |context: &Qwen4ContextCommit| {
                Qwen4State::device_allocation_sizes(&config, format, context)
                    .unwrap()
                    .iter()
                    .sum::<u64>()
            };
            let committed = |context: &Qwen4ContextCommit| {
                context
                    .committed_layer_bytes(&config, format.layout())
                    .unwrap() as u64
            };
            assert!(committed(&vmm) < committed(&legacy));
            assert_eq!(
                total(&legacy) - total(&vmm),
                (committed(&legacy) - committed(&vmm)) * layers,
                "{name}"
            );
        }
    }

    /// One boundary stage is the fixed parts of the owner's tensor plan in
    /// session-layout order (per GDN layer recurrent then conv, per QSA
    /// layer the selection, then PLE conv and hyper feedback), one entry per
    /// part, summing to the stage-byte query; with and without PLE.
    #[test]
    fn session_fixed_part_sizes_follow_the_state_plan() {
        let base = crate::config::compact_test_config();
        let mut no_ple = base.clone();
        no_ple.ple = None;
        for (label, config) in [("ple", base), ("no ple", no_ple)] {
            let map = Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).unwrap();
            for (name, format) in allocation_formats() {
                let geometry = StateGeometry::new(&config, &format, 1024).unwrap();
                let plan = state_tensor_plan(&config, &format, 1024, &geometry, &map).unwrap();
                let part = |suffixes: &[&str]| -> Vec<u64> {
                    plan.iter()
                        .filter(|spec| {
                            suffixes
                                .iter()
                                .any(|s| spec.key.ends_with(s) || spec.key == *s)
                        })
                        .map(|spec| (spec.elements * spec.dtype.size()) as u64)
                        .collect()
                };
                let mut expected = part(&[".recurrent", ".conv"]);
                expected.extend(part(&[".selected_indices"]));
                expected.extend(part(&["ple_conv"]));
                expected.extend(part(&["hyper_feedback"]));
                expected.retain(|&bytes| bytes > 0);
                let sizes = Qwen4State::session_fixed_part_sizes(&config, format).unwrap();
                assert_eq!(sizes, expected, "{label} {name}");
                assert_eq!(
                    Qwen4State::session_fixed_stage_bytes(&config, format),
                    Some(sizes.iter().sum::<u64>()),
                    "{label} {name}"
                );
            }
        }
        let mut wide = crate::config::compact_test_config();
        wide.hidden_size = usize::MAX;
        assert_eq!(
            Qwen4State::session_fixed_part_sizes(&wide, Qwen4StateFormat::F32),
            None
        );
        assert_eq!(
            Qwen4State::session_fixed_stage_bytes(&wide, Qwen4StateFormat::F32),
            None
        );
    }

    /// The verify capture ring's per-allocation inventory against the owner's
    /// geometry and the byte query: three allocations per GDN layer, the F32
    /// output block and the pointer table; whole-model and band layer counts.
    #[test]
    fn row_capture_allocation_sizes_sum_to_the_byte_query() {
        let config = crate::config::compact_test_config();
        let layers = config.n_linear_layers();
        let value_heads = config.linear_num_value_heads;
        for gdn in [
            GdnStateFormat::F32,
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
        ] {
            let format = Qwen4StateFormat {
                gdn,
                ..Qwen4StateFormat::F32
            };
            let geometry = StateGeometry::new(&config, &format, 1024).unwrap();
            let history = config.linear_conv_kernel_dim - 1;
            let qkv = geometry.conv_len / history;
            for rows in [
                0usize,
                1,
                2,
                4,
                8,
                32,
                hipfire_config::QWEN4_VERIFY_MAX_ROWS,
            ] {
                let sizes =
                    Qwen4State::row_capture_allocation_sizes_scoped(&config, gdn, rows, layers)
                        .unwrap();
                assert_eq!(
                    Some(sizes.iter().sum::<u64>()),
                    Qwen4State::row_capture_bytes(&config, gdn, rows).map(|bytes| bytes as u64),
                    "{gdn:?} {rows}"
                );
                if rows < 2 {
                    assert!(sizes.is_empty());
                    continue;
                }
                let slot_bytes = gdn.storage().state_bytes(
                    value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ) as u64;
                let (inputs, recurrence) = (
                    (rows * qkv * 4) as u64,
                    (rows * (qkv + 2 * value_heads) * 4) as u64,
                );
                let pooled = compact_capture_ring() && lane_reserve_trim();
                let per_layer = if pooled {
                    // One shared pool, then (inputs, recurrence) per layer.
                    assert_eq!(sizes.len(), 2 * layers + 3);
                    assert_eq!(
                        sizes[0],
                        capture_ring_pool_slots(rows, layers).unwrap() as u64 * slot_bytes
                    );
                    for layer in sizes[1..1 + 2 * layers].chunks(2) {
                        assert_eq!(layer, [inputs, recurrence], "{gdn:?} {rows}");
                    }
                    1 + 2 * layers
                } else {
                    assert_eq!(sizes.len(), 3 * layers + 2);
                    let ring = capture_ring_slots(compact_capture_ring(), rows).unwrap() as u64
                        * slot_bytes;
                    for layer in sizes[..3 * layers].chunks(3) {
                        assert_eq!(layer, [ring, inputs, recurrence], "{gdn:?} {rows}");
                    }
                    3 * layers
                };
                assert_eq!(
                    sizes[per_layer..],
                    [
                        (rows * value_heads * GDN_HEAD_DIM * 4) as u64,
                        (layers * 16) as u64
                    ]
                );
            }
            for band in [0usize, 7] {
                assert_eq!(
                    Qwen4State::row_capture_allocation_sizes_scoped(&config, gdn, 4, band)
                        .map(|sizes| sizes.iter().sum::<u64>()),
                    Qwen4State::row_capture_bytes_scoped(&config, gdn, 4, band)
                        .map(|bytes| bytes as u64),
                    "{gdn:?} band {band}"
                );
            }
            assert_eq!(
                Qwen4State::row_capture_allocation_sizes_scoped(
                    &config,
                    gdn,
                    usize::MAX / 2,
                    layers
                ),
                None
            );
            assert_eq!(
                Qwen4State::row_capture_bytes(&config, gdn, usize::MAX / 2),
                None
            );
        }
    }

    #[test]
    fn allocation_sizes_refuse_zero_capacity_and_overflow_like_the_byte_query() {
        let config = crate::config::compact_test_config();
        let format = Qwen4StateFormat::F32;
        assert_eq!(
            Qwen4State::device_allocation_sizes(&config, format, &Qwen4ContextCommit::legacy(0)),
            None
        );
        let mut wide = config.clone();
        wide.indexer_head_dim = usize::MAX;
        let mut ratio_zero = config.clone();
        ratio_zero.indexer_compress_ratio = 0;
        let context = Qwen4ContextCommit::legacy(1024);
        for (label, config, context) in [
            ("indexer width", &wide, context),
            ("ratio zero", &ratio_zero, context),
            ("context", &config, Qwen4ContextCommit::legacy(usize::MAX)),
        ] {
            assert_eq!(
                Qwen4State::device_allocation_sizes(config, format, &context),
                None,
                "{label}"
            );
        }
        // The owner's closed form refuses the genuinely overflowing ones too.
        assert_eq!(Qwen4State::device_bytes(&wide, format, &context), None);
        assert_eq!(
            Qwen4State::device_bytes(&config, format, &Qwen4ContextCommit::legacy(usize::MAX)),
            None
        );
    }

    /// Scopes of consecutive bands `counts` over the trunk: first is input,
    /// last is output.
    fn pp_scopes(counts: &[usize]) -> Vec<Qwen4BandScope> {
        let mut start = 0;
        counts
            .iter()
            .enumerate()
            .map(|(stage, &count)| {
                let band = start..start + count;
                start += count;
                Qwen4BandScope {
                    band,
                    input: stage == 0,
                    output: stage + 1 == counts.len(),
                }
            })
            .collect()
    }

    const PP_LAYOUTS: [&[usize]; 2] = [&[9, 8, 8, 8, 8, 7], &[24, 24]];
    const QSA_FIELD_NAMES: [&str; 7] = [
        "full_keys",
        "full_values",
        "raw_index_keys",
        "pooled_keys",
        "partial_keys",
        "partial_values",
        "selected_indices",
    ];

    fn tensor_plan_for(config: &Qwen4Config, scope: &Qwen4BandScope) -> Vec<StateTensorSpec> {
        let format = Qwen4StateFormat::F32;
        let map = Qwen4StateMap::new(config, scope).expect("state map");
        let geometry = StateGeometry::new(config, &format, 1024).expect("geometry");
        state_tensor_plan(config, &format, 1024, &geometry, &map).expect("tensor plan")
    }

    fn layer_of_key(key: &str) -> Option<usize> {
        let digits = key.split_once("[L")?.1.split_once(']')?.0;
        digits.parse().ok()
    }

    #[test]
    fn single_scope_state_plan_is_todays_layout() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let map = Qwen4StateMap::new(&config, &single).expect("single map");
        assert!(map.is_single() && map.owns_input() && map.owns_output() && map.owns_ple());
        assert_eq!(
            (
                map.gdn_count(),
                map.qsa_count(),
                map.gdn_base(),
                map.qsa_base()
            ),
            (36, 12, 0, 0)
        );
        assert_eq!(map.session_tag(), 0);
        assert_eq!(scope_layer_counts(&config, &single), (36, 12));

        let plan = tensor_plan_for(&config, &single);
        assert_eq!(plan.len(), 36 * 2 + 12 * 7 + 2);
        let mut expected = Vec::new();
        for (abs, kind) in config.layer_types.iter().enumerate() {
            match kind {
                LayerType::LinearAttention => {
                    expected.push(format!("gdn[L{abs}].recurrent"));
                    expected.push(format!("gdn[L{abs}].conv"));
                }
                LayerType::FullAttention => {
                    expected.extend(QSA_FIELD_NAMES.map(|name| format!("qsa[L{abs}].{name}")));
                }
            }
        }
        expected.push("ple_conv".into());
        expected.push("hyper_feedback".into());
        let keys: Vec<_> = plan.iter().map(|spec| spec.key.clone()).collect();
        assert_eq!(keys, expected);

        // Hand geometry (F32 state at 1024 tokens; see device_bytes test).
        let by_key = |key: &str| plan.iter().find(|spec| spec.key == key).expect(key);
        assert_eq!(by_key("gdn[L0].recurrent").elements, 48 * 128 * 128);
        assert_eq!(by_key("gdn[L0].conv").elements, 10_240 * 3);
        assert_eq!(by_key("qsa[L3].full_keys").elements, 1024 * 512);
        assert!(by_key("qsa[L3].full_keys").context_arena);
        assert_eq!(by_key("qsa[L3].raw_index_keys").elements, 1024 * 128);
        assert_eq!(by_key("qsa[L3].pooled_keys").elements, 256 * 128);
        assert_eq!(by_key("qsa[L3].partial_keys").elements, 4 * 128);
        assert_eq!(by_key("qsa[L3].partial_values").elements, 4 * 512);
        assert_eq!(by_key("qsa[L3].selected_indices").elements, (2048 + 3) * 4);
        assert_eq!(by_key("qsa[L3].selected_indices").dtype, DType::Raw);
        assert!(!by_key("qsa[L3].partial_keys").context_arena);
        assert_eq!(by_key("ple_conv").elements, 9 * 2560 * 4);
        assert_eq!(by_key("hyper_feedback").elements, 4 * 2560);
    }

    #[test]
    fn single_scope_debug_digest_rows_keep_their_historical_labels() {
        let config = crate::config::compact_test_config();
        let map = Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).expect("map");
        let keys = |row_capture: bool, output: bool| -> Vec<String> {
            digest_plan(&map, false, row_capture, output)
                .into_iter()
                .map(|row| digest_key(&map, row, false))
                .collect()
        };
        let mut expected: Vec<String> = [
            "position",
            "ple_history.previous",
            "ple_history.eos",
            "gdn_live",
            "row_capture_rows",
            "qsa_mapped_tokens",
        ]
        .map(String::from)
        .into();
        expected.extend((0..12).map(|i| format!("qsa[{i}].marks")));
        for i in 0..36 {
            expected.push(format!("gdn[{i}].recurrent"));
            expected.push(format!("gdn[{i}].conv"));
        }
        for i in 0..12 {
            expected.extend(QSA_FIELD_NAMES.map(|name| format!("qsa[{i}].{name}")));
        }
        expected.push("ple_conv".into());
        expected.push("hyper_feedback".into());
        assert_eq!(keys(false, false), expected);
        for i in 0..36 {
            for name in ["ring", "inputs", "recurrence"] {
                expected.push(format!("row_capture[{i}].{name}"));
            }
        }
        assert_eq!(keys(true, false), expected);
        expected.push("row_capture_output".into());
        assert_eq!(keys(true, true), expected);
    }

    #[test]
    fn pp_state_maps_partition_the_single_layout_and_round_trip() {
        let config = crate::config::compact_test_config();
        let single = Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).expect("map");
        for counts in PP_LAYOUTS {
            let scopes = pp_scopes(counts);
            let (mut gdn_union, mut qsa_union) = (Vec::new(), Vec::new());
            let (mut gdn_before, mut qsa_before) = (0, 0);
            let mut start = 0;
            for (stage, scope) in scopes.iter().enumerate() {
                let map = Qwen4StateMap::new(&config, scope).expect("band map");
                assert_eq!(map.band(), start..start + counts[stage]);
                start += counts[stage];
                assert_eq!(map.is_single(), counts.len() == 1);
                assert_eq!(map.owns_input(), stage == 0);
                assert_eq!(map.owns_ple(), stage == 0);
                assert_eq!(map.owns_output(), stage + 1 == counts.len());
                assert_eq!((map.gdn_base(), map.qsa_base()), (gdn_before, qsa_before));
                assert_eq!(
                    (map.gdn_count(), map.qsa_count()),
                    scope_layer_counts(&config, scope)
                );
                for (slot, &abs) in map.gdn_layers().iter().enumerate() {
                    assert_eq!(config.layer_types[abs], LayerType::LinearAttention);
                    assert_eq!(map.gdn_local(abs), Some(slot));
                    assert_eq!(map.gdn_layer(slot), Some(abs));
                    assert_eq!(map.qsa_local(abs), None);
                    // Whole-model GDN ordinal = base + local slot.
                    assert_eq!(single.gdn_local(abs), Some(map.gdn_base() + slot));
                }
                for (slot, &abs) in map.qsa_layers().iter().enumerate() {
                    assert_eq!(config.layer_types[abs], LayerType::FullAttention);
                    assert_eq!(map.qsa_local(abs), Some(slot));
                    assert_eq!(map.qsa_layer(slot), Some(abs));
                    assert_eq!(map.gdn_local(abs), None);
                    assert_eq!(single.qsa_local(abs), Some(map.qsa_base() + slot));
                }
                for abs in (0..48).filter(|abs| !map.band().contains(abs)) {
                    assert_eq!((map.gdn_local(abs), map.qsa_local(abs)), (None, None));
                }
                assert_eq!(map.gdn_layer(map.gdn_count()), None);
                assert_eq!(map.qsa_layer(map.qsa_count()), None);
                gdn_before += map.gdn_count();
                qsa_before += map.qsa_count();
                gdn_union.extend_from_slice(map.gdn_layers());
                qsa_union.extend_from_slice(map.qsa_layers());
            }
            assert_eq!(start, 48);
            assert_eq!((gdn_before, qsa_before), (36, 12));
            assert_eq!(gdn_union.as_slice(), single.gdn_layers());
            assert_eq!(qsa_union.as_slice(), single.qsa_layers());
        }
    }

    #[test]
    fn pp_tensor_plans_union_to_the_single_tensors_exactly_once() {
        let config = crate::config::compact_test_config();
        let mut single = tensor_plan_for(&config, &Qwen4BandScope::single(&config));
        single.sort_by(|a, b| a.key.cmp(&b.key));
        for counts in PP_LAYOUTS {
            let scopes = pp_scopes(counts);
            let plans: Vec<_> = scopes
                .iter()
                .map(|scope| tensor_plan_for(&config, scope))
                .collect();
            let mut union: Vec<_> = plans.iter().flatten().cloned().collect();
            union.sort_by(|a, b| a.key.cmp(&b.key));
            assert_eq!(union, single, "{counts:?}");
            for (stage, (plan, scope)) in plans.iter().zip(&scopes).enumerate() {
                let (gdn, qsa) = scope_layer_counts(&config, scope);
                assert_eq!(
                    plan.len(),
                    2 * gdn + 7 * qsa + if stage == 0 { 2 } else { 0 }
                );
                let has = |key: &str| plan.iter().any(|spec| spec.key == key);
                assert_eq!(has("ple_conv"), stage == 0);
                assert_eq!(has("hyper_feedback"), stage == 0);
                // Owned layers only, ascending, then the input-only tail.
                let layers: Vec<usize> = plan
                    .iter()
                    .filter_map(|spec| layer_of_key(&spec.key))
                    .collect();
                assert!(layers.windows(2).all(|pair| pair[0] <= pair[1]));
                assert!(layers.iter().all(|layer| scope.band.contains(layer)));
                let tail = plan.len() - if stage == 0 { 2 } else { 0 };
                assert!(plan[..tail]
                    .iter()
                    .all(|spec| layer_of_key(&spec.key).is_some()));
            }
        }
    }

    #[test]
    fn pp_canonical_digest_rows_union_to_the_single_rows() {
        let config = crate::config::compact_test_config();
        let canonical_keys = |scope: &Qwen4BandScope| -> Vec<String> {
            let map = Qwen4StateMap::new(&config, scope).expect("map");
            digest_plan(&map, true, true, true)
                .into_iter()
                .map(|row| digest_key(&map, row, true))
                .collect()
        };
        let single = canonical_keys(&Qwen4BandScope::single(&config));
        assert!(single.contains(&"qsa[L11].full_keys".to_string()));
        assert!(single.contains(&"gdn[L0].recurrent".to_string()));
        assert!(single.contains(&"row_capture[L2].ring".to_string()));
        for banned in [
            "row_capture_rows",
            "qsa_mapped_tokens",
            "row_capture_output",
        ] {
            assert!(!single.iter().any(|key| key == banned), "{banned}");
        }
        // Every owned layer reports its own position, every GDN layer its
        // own ring slot; the debug digest keeps the two global rows.
        let count = |prefix: &str, suffix: &str| {
            single
                .iter()
                .filter(|key| key.starts_with(prefix) && key.ends_with(suffix))
                .count()
        };
        assert_eq!(count("layer[L", ".position"), 48);
        assert_eq!(count("gdn[L", ".live"), 36);
        assert!(!single
            .iter()
            .any(|key| key == "position" || key == "gdn_live"));
        let single_set: std::collections::BTreeSet<_> = single.iter().cloned().collect();
        assert_eq!(single_set.len(), single.len(), "canonical keys are unique");
        for counts in PP_LAYOUTS {
            let mut seen = std::collections::BTreeMap::<String, usize>::new();
            for scope in pp_scopes(counts) {
                let keys = canonical_keys(&scope);
                let unique: std::collections::BTreeSet<_> = keys.iter().collect();
                assert_eq!(unique.len(), keys.len(), "unique within a rank");
                for key in keys {
                    *seen.entry(key).or_default() += 1;
                }
            }
            let union: std::collections::BTreeSet<_> = seen.keys().cloned().collect();
            assert_eq!(union, single_set, "{counts:?}");
            for (key, count) in &seen {
                assert_eq!(*count, 1, "{key} is owned exactly once");
            }
        }
    }

    #[test]
    fn pp_state_census_sums_to_the_single_census_without_duplicates() {
        let config = crate::config::compact_test_config();
        let format = Qwen4StateFormat::F32;
        let context = Qwen4ContextCommit::legacy(1024);
        let single = Qwen4State::device_bytes(&config, format, &context).unwrap();
        let single_scope = Qwen4BandScope::single(&config);
        assert_eq!(
            Qwen4State::device_bytes_scoped(&config, format, &context, &single_scope),
            Some(single)
        );
        for counts in PP_LAYOUTS {
            let bytes: Vec<usize> = pp_scopes(counts)
                .iter()
                .map(|scope| {
                    Qwen4State::device_bytes_scoped(&config, format, &context, scope).unwrap()
                })
                .collect();
            assert_eq!(bytes.iter().sum::<usize>(), single, "{counts:?}");
        }
        // PP2 has 6 QSA + 18 GDN layers per band: the input band's surplus
        // is exactly the per-request PLE history and feedback (live+mirror).
        let pp2: Vec<usize> = pp_scopes(&[24, 24])
            .iter()
            .map(|scope| Qwen4State::device_bytes_scoped(&config, format, &context, scope).unwrap())
            .collect();
        let per_request = 2 * (9 * 2560 * 4 * 4 + 4 * 2560 * 4);
        assert_eq!(pp2[0] - pp2[1], per_request);
        // The ring census of the whole model equals the all-GDN scoped one.
        assert_eq!(
            Qwen4State::row_capture_bytes_scoped(&config, GdnStateFormat::Q8, 4, 36),
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, 4)
        );
        assert!(
            Qwen4State::row_capture_bytes_scoped(&config, GdnStateFormat::Q8, 4, 18)
                < Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, 4)
        );
    }

    #[test]
    fn state_map_refuses_inconsistent_scopes() {
        let config = crate::config::compact_test_config();
        let scope = |band: Range<usize>, input, output| Qwen4BandScope {
            band,
            input,
            output,
        };
        // PLE layer 2 is not inside an input band that starts past it.
        assert!(Qwen4StateMap::new(&config, &scope(3..48, true, true)).is_err());
        // The PLE layer inside a non-input band, wrong end flags, empty and
        // out-of-trunk bands are refused.
        assert!(Qwen4StateMap::new(&config, &scope(1..48, false, true)).is_err());
        assert!(Qwen4StateMap::new(&config, &scope(0..24, true, true)).is_err());
        assert!(Qwen4StateMap::new(&config, &scope(24..24, false, false)).is_err());
        assert!(Qwen4StateMap::new(&config, &scope(24..49, false, true)).is_err());
        let tail = Qwen4StateMap::new(&config, &scope(24..48, false, true)).expect("tail band");
        assert!(!tail.owns_input() && !tail.owns_ple() && tail.owns_output());
        assert_ne!(tail.session_tag(), 0);
    }

    #[test]
    fn band_session_tags_identify_the_ownership() {
        let config = crate::config::compact_test_config();
        let mut tags = std::collections::BTreeSet::new();
        for counts in PP_LAYOUTS {
            for scope in pp_scopes(counts) {
                let tag = Qwen4StateMap::new(&config, &scope).unwrap().session_tag();
                assert_ne!(tag, 0);
                assert!(tags.insert(tag), "{:?}", scope.band);
            }
        }
    }

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok().or_else(|| {
            eprintln!("skip: Qwen4 state arena tests require a GPU");
            None
        })
    }

    fn new_compact_state(gpu: &mut Gpu) -> Qwen4State {
        let config = crate::config::compact_test_config();
        Qwen4State::new(gpu, &config, 8, Qwen4StateFormat::F32).expect("compact Qwen4 state")
    }

    #[test]
    fn device_bytes_matches_the_constructors_allocations() {
        let config = crate::config::compact_test_config();
        let context = Qwen4ContextCommit::legacy(1024);
        // Hand computation, F32 QSA and GDN state, 36 GDN + 12 QSA layers.
        let gdn = 48 * 128 * 128 * 4 + 10_240 * 3 * 4;
        let arena = 1024 * (2 * 2048 + 512) + 256 * 512;
        assert_eq!(
            config.qsa_context_arena_bytes(1024, QsaKvFormat::F32),
            Some(arena)
        );
        let fixed = 4 * 128 * 4 + 4 * 512 * 4 + (2048 + 3) * 4;
        let circular = 4 * 128 * 4;
        let ple = 9 * 2560 * 4 * 4;
        let feedback = 4 * 2560 * 4;
        let expected = 36 * 2 * gdn + 12 * (arena + 2 * fixed + circular) + 2 * (ple + feedback);
        assert_eq!(expected, 294_822_176);
        assert_eq!(
            Qwen4State::device_bytes(&config, Qwen4StateFormat::F32, &context),
            Some(expected)
        );
    }

    /// An AR lane's state has no snapshot mirror: the hand computation of
    /// [`device_bytes_matches_the_constructors_allocations`] without the
    /// mirror terms (GDN recurrent + conv, QSA partial K/V + selection + the
    /// raw-index circular block, PLE conv + feedback), exactly. Flash-Next
    /// geometry, F32 state, 36 GDN + 12 QSA layers; the mirror is context
    /// independent, so it is the per-lane saving at every `max_seq`.
    #[test]
    fn device_bytes_without_the_mirror_is_the_live_state_alone() {
        let config = crate::config::compact_test_config();
        let context = Qwen4ContextCommit::legacy(1024);
        let single = Qwen4BandScope::single(&config);
        let gdn = 48 * 128 * 128 * 4 + 10_240 * 3 * 4;
        let arena = 1024 * (2 * 2048 + 512) + 256 * 512;
        let fixed = 4 * 128 * 4 + 4 * 512 * 4 + (2048 + 3) * 4;
        let circular = 4 * 128 * 4;
        let ple = 9 * 2560 * 4 * 4;
        let feedback = 4 * 2560 * 4;
        let live = 36 * gdn + 12 * (arena + fixed) + ple + feedback;
        let mirror = 36 * gdn + 12 * (fixed + circular) + ple + feedback;
        assert_eq!((live, mirror), (176_496_784, 118_325_392));
        assert_eq!(8 * mirror, 946_603_136);
        let bytes = |mirror| {
            Qwen4State::device_bytes_with_mirror(
                &config,
                Qwen4StateFormat::F32,
                &context,
                &single,
                mirror,
            )
        };
        assert_eq!(bytes(StateMirror::Absent), Some(live));
        assert_eq!(bytes(StateMirror::Snapshot), Some(live + mirror));
        assert_eq!(
            Qwen4State::device_bytes(&config, Qwen4StateFormat::F32, &context),
            bytes(StateMirror::Snapshot)
        );
        // The mirror does not depend on the committed context.
        for context in allocation_contexts() {
            let with = |mirror| {
                Qwen4State::device_bytes_with_mirror(
                    &config,
                    Qwen4StateFormat::F32,
                    &context,
                    &single,
                    mirror,
                )
                .unwrap()
            };
            assert_eq!(
                with(StateMirror::Snapshot) - with(StateMirror::Absent),
                118_325_392,
                "{context:?}"
            );
        }
    }

    /// Bands charge their own layers' mirrors and nothing twice: the band
    /// mirror bytes sum to the whole-model mirror, only the input band holds
    /// the PLE/feedback mirror, and the live EF residual is the same under
    /// both choices.
    #[test]
    fn band_mirror_bytes_sum_to_the_single_mirror() {
        let config = crate::config::compact_test_config();
        let context = Qwen4ContextCommit::legacy(1024);
        let single = Qwen4BandScope::single(&config);
        let format = |gdn| Qwen4StateFormat {
            qsa: QsaKvFormat::Fp8,
            gdn,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        };
        let bytes = |gdn, scope: &Qwen4BandScope, mirror| {
            Qwen4State::device_bytes_with_mirror(&config, format(gdn), &context, scope, mirror)
                .unwrap()
        };
        for gdn in [
            GdnStateFormat::F32,
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
        ] {
            let mirror_of = |scope: &Qwen4BandScope| {
                bytes(gdn, scope, StateMirror::Snapshot) - bytes(gdn, scope, StateMirror::Absent)
            };
            let whole = mirror_of(&single);
            assert!(whole > 0, "{gdn:?}");
            for counts in PP_LAYOUTS {
                let scopes = pp_scopes(counts);
                assert_eq!(
                    scopes.iter().map(mirror_of).sum::<usize>(),
                    whole,
                    "{gdn:?} {counts:?}"
                );
                // The input band's surplus over a same-shaped tail band is
                // the PLE conv + feedback mirror alone.
                let per_request = 9 * 2560 * 4 * 4 + 4 * 2560 * 4;
                let (head, tail) = (&scopes[0], scopes.last().unwrap());
                let (head_gdn, head_qsa) = scope_layer_counts(&config, head);
                let (tail_gdn, tail_qsa) = scope_layer_counts(&config, tail);
                if (head_gdn, head_qsa) == (tail_gdn, tail_qsa) {
                    assert_eq!(mirror_of(head) - mirror_of(tail), per_request);
                }
            }
        }
        // The Q8Ef residual is live-only: 54 MiB over Q8 under either choice.
        for mirror in [StateMirror::Snapshot, StateMirror::Absent] {
            assert_eq!(
                bytes(GdnStateFormat::Q8Ef, &single, mirror)
                    - bytes(GdnStateFormat::Q8, &single, mirror),
                54 * 1024 * 1024,
                "{mirror:?}"
            );
        }
    }

    #[test]
    fn device_bytes_without_ple_budgets_no_conv_history() {
        use crate::config::{synthetic_test_config, Qwen4FfnConfig};
        let build = |with_ple| {
            synthetic_test_config(
                Qwen4FfnConfig::Dense {
                    intermediate_size: 8,
                },
                16,
                4,
                with_ple,
                false,
            )
        };
        let (with, without) = (build(true), build(false));
        let context = Qwen4ContextCommit::legacy(64);
        let bytes = |config: &Qwen4Config| {
            Qwen4State::device_bytes(config, Qwen4StateFormat::F32, &context).expect("state bytes")
        };
        // Live conv history plus its snapshot-arena mirror, nothing else
        // differs: `hyper_feedback` is allocated for every profile.
        let ple = with.ple_conv_history_rows() * with.hidden_size * with.hc_count * 4;
        assert!(ple > 0 && without.ple_conv_history_rows() == 0);
        assert_eq!(bytes(&with) - bytes(&without), 2 * ple);
    }

    /// The host-side bookkeeping of a state with no device memory (no GDN,
    /// QSA, PLE or feedback tensor): what the snapshot-mirror ownership rules
    /// act on, built without a device. A `Snapshot` arena here stands for a
    /// mirror whose tensor lists are empty.
    fn host_only_state(mirror: StateMirror, model_id: u64, position: usize) -> Qwen4State {
        let config = crate::config::compact_test_config();
        let ple_history = PleHistory::new(config.eos_token_id);
        let mut arena = Qwen4StateSnapshotArena::absent(model_id, ple_history);
        arena.mirror = mirror;
        Qwen4State {
            gdn: Vec::new(),
            qsa: Vec::new(),
            ple_conv: None,
            hyper_feedback: None,
            ple_history,
            position,
            max_seq_len: 8,
            qsa_selected_capacity: 0,
            snapshot_arena: arena,
            row_capture: Vec::new(),
            row_capture_output: None,
            row_capture_pool: None,
            row_capture_rows: 0,
            row_capture_compact: false,
            gdn_live: 0,
            gdn_format: GdnStateFormat::F32,
            gdn_value_heads: config.linear_num_value_heads,
            gdn_residual_resets: 0,
            row_capture_armed: false,
            #[cfg(feature = "lab")]
            lab: crate::lab_window::LabStateTap::default(),
            qsa_backend: Qwen4KvBackend::Legacy,
            qsa_mapped_tokens: 8,
            qsa_compress: config.indexer_compress_ratio,
            model_id,
            reset_epoch: 0,
            map: Qwen4StateMap::new(&config, &Qwen4BandScope::single(&config)).unwrap(),
            transaction_generation: 0,
        }
    }

    /// A ticket shaped as `state`'s own snapshot would be.
    fn host_ticket(state: &Qwen4State) -> Qwen4StateSnapshot {
        Qwen4StateSnapshot {
            model_id: state.model_id,
            reset_epoch: state.reset_epoch,
            transaction_generation: state.transaction_generation,
            arena_generation: state.snapshot_arena.generation,
            dimensions: Qwen4StateSnapshotArena::dimensions(state),
        }
    }

    /// A state without a mirror can never hold a ticket: every ticket
    /// operation is the typed `MirrorAbsent` (not a lazy allocation, not
    /// "inactive"), it reports no outstanding snapshot, and invalidation is a
    /// no-op. A snapshot-capable state with no ticket answers differently.
    #[test]
    fn an_absent_mirror_has_no_tickets_and_ignores_invalidation() {
        let mut state = host_only_state(StateMirror::Absent, 7, 3);
        assert!(matches!(
            state.snapshot_arena.require_mirror(),
            Err(StateError::MirrorAbsent)
        ));
        assert!(!state.snapshot_active());
        let generation = state.snapshot_arena.generation;
        state.snapshot_arena.invalidate();
        state.release_retained();
        assert_eq!(state.snapshot_arena.generation, generation);
        assert!(!state.snapshot_arena.active && !state.snapshot_arena.retained);
        assert!(matches!(
            state.validate_commit(host_ticket(&state)),
            Err(StateError::MirrorAbsent)
        ));
        let capable = host_only_state(StateMirror::Snapshot, 8, 3);
        assert!(capable.snapshot_arena.require_mirror().is_ok());
        assert!(matches!(
            capable.validate_commit(host_ticket(&capable)),
            Err(StateError::SnapshotInactive)
        ));
        assert_eq!(
            StateError::MirrorAbsent.to_string(),
            "Qwen4 state has no snapshot mirror"
        );
    }

    /// A promotion moves the resident request into its lane: the request
    /// (position, model id) changes slot while each slot keeps its mirror
    /// kind, so the resident state still snapshots and an AR lane stays
    /// mirrorless. The mirror that changes hands is rebound to the state it
    /// now sits in and its previous owner's tickets stop validating; two
    /// snapshot-capable states swap whole.
    #[test]
    fn swap_requests_keeps_each_slots_mirror() {
        let mut resident = host_only_state(StateMirror::Snapshot, 11, 5);
        let mut lane = host_only_state(StateMirror::Absent, 12, 0);
        resident.snapshot_arena.generation = 3;
        resident.snapshot_arena.active = true;
        resident.snapshot_arena.retained = true;
        resident.swap_requests(&mut lane);
        assert_eq!((resident.position, lane.position), (0, 5));
        assert_eq!((resident.model_id, lane.model_id), (12, 11));
        assert_eq!(resident.snapshot_arena.mirror, StateMirror::Snapshot);
        assert_eq!(lane.snapshot_arena.mirror, StateMirror::Absent);
        assert_eq!(resident.snapshot_arena.model_id, resident.model_id);
        assert_eq!(lane.snapshot_arena.model_id, lane.model_id);
        assert!(resident.snapshot_arena.generation > 3);
        assert!(!resident.snapshot_arena.active && !resident.snapshot_arena.retained);
        assert!(!lane.snapshot_active());
        assert!(matches!(
            lane.snapshot_arena.require_mirror(),
            Err(StateError::MirrorAbsent)
        ));
        assert!(resident.snapshot_arena.require_mirror().is_ok());
        // The same swap, from the other side.
        let mut lane = host_only_state(StateMirror::Absent, 21, 9);
        let mut resident = host_only_state(StateMirror::Snapshot, 22, 0);
        lane.swap_requests(&mut resident);
        assert_eq!(lane.snapshot_arena.mirror, StateMirror::Absent);
        assert_eq!(resident.snapshot_arena.mirror, StateMirror::Snapshot);
        assert_eq!(resident.snapshot_arena.model_id, resident.model_id);
        assert_eq!((lane.position, resident.position), (0, 9));

        // Two snapshot-capable states (an MTP lane): the whole state swaps,
        // arena and generation included.
        let mut a = host_only_state(StateMirror::Snapshot, 31, 1);
        let mut b = host_only_state(StateMirror::Snapshot, 32, 2);
        a.snapshot_arena.generation = 4;
        b.snapshot_arena.generation = 6;
        a.swap_requests(&mut b);
        assert_eq!((a.model_id, b.model_id), (32, 31));
        assert_eq!(
            (a.snapshot_arena.model_id, b.snapshot_arena.model_id),
            (32, 31)
        );
        assert_eq!(
            (a.snapshot_arena.generation, b.snapshot_arena.generation),
            (6, 4)
        );
        // Two mirrorless states likewise.
        let mut c = host_only_state(StateMirror::Absent, 41, 1);
        let mut d = host_only_state(StateMirror::Absent, 42, 2);
        c.swap_requests(&mut d);
        assert_eq!((c.position, d.position), (2, 1));
        assert!(
            c.snapshot_arena.mirror == StateMirror::Absent
                && d.snapshot_arena.mirror == StateMirror::Absent
        );
    }

    /// A real mirrorless state (needs a device; skipped without one):
    /// allocates no mirror, steps and resets like any state, and refuses
    /// every snapshot, restore and verify-ring request with the typed error.
    #[test]
    fn a_mirrorless_state_refuses_snapshot_restore_and_the_verify_ring() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let config = crate::config::compact_test_config();
        let scope = Qwen4BandScope::single(&config);
        let mut state = Qwen4State::new_with_mirror(
            &mut gpu,
            &config,
            8,
            Qwen4StateFormat::F32,
            Qwen4KvBackend::Legacy,
            &scope,
            StateMirror::Absent,
        )
        .expect("mirrorless state");
        assert!(state.snapshot_arena.recurrent.is_empty());
        assert!(state.snapshot_arena.qsa_partial_keys.is_empty());
        assert!(state.snapshot_arena.ple_conv.is_none());
        assert!(state.snapshot_arena.hyper_feedback.is_none());
        assert!(matches!(
            state.snapshot(&mut gpu),
            Err(StateError::MirrorAbsent)
        ));
        let ticket = host_ticket(&state);
        assert!(matches!(
            state.restore(&mut gpu, ticket),
            Err(StateError::MirrorAbsent)
        ));
        assert!(matches!(
            state.ensure_row_capture(&mut gpu, 4, config.linear_conv_kernel_dim - 1),
            Err(StateError::MirrorAbsent)
        ));
        state.position = 2;
        state.reset(&mut gpu).expect("reset in place");
        assert_eq!(state.position, 0);
        assert!(!state.snapshot_active());
        state.free_gpu(&mut gpu).expect("free state");
    }

    #[test]
    fn snapshot_arena_enforces_ticket_lifecycle_before_copy() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let mut state = new_compact_state(&mut gpu);
        state.position = 2;

        let ticket = state.snapshot(&mut gpu).expect("first snapshot");
        assert!(matches!(
            state.snapshot(&mut gpu),
            Err(StateError::SnapshotBusy)
        ));

        let mut stale_model = ticket;
        stale_model.model_id = stale_model.model_id.wrapping_add(1);
        assert!(matches!(
            state.restore(&mut gpu, stale_model),
            Err(StateError::SnapshotTicket)
        ));
        assert_eq!(state.position, 2);
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("request snapshot");
        state.reset_epoch = state.reset_epoch.wrapping_add(1);
        assert!(matches!(
            state.restore(&mut gpu, ticket),
            Err(StateError::SnapshotTicket)
        ));
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("transaction snapshot");
        state.transaction_generation = state.transaction_generation.wrapping_add(1);
        assert!(matches!(
            state.restore(&mut gpu, ticket),
            Err(StateError::SnapshotTicket)
        ));
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("arena generation snapshot");
        state.snapshot_arena.generation = state.snapshot_arena.generation.wrapping_add(1);
        assert!(matches!(
            state.restore(&mut gpu, ticket),
            Err(StateError::SnapshotTicket)
        ));
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("shape snapshot");
        let mut stale_shape = ticket;
        stale_shape.dimensions.shape_hash ^= 1;
        assert!(matches!(
            state.restore(&mut gpu, stale_shape),
            Err(StateError::SnapshotTicket)
        ));
        assert_eq!(state.position, 2);
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("restore-consume snapshot");
        state.restore(&mut gpu, ticket).expect("restore");
        assert!(matches!(
            state.restore(&mut gpu, ticket),
            Err(StateError::SnapshotInactive)
        ));

        let ticket = state.snapshot(&mut gpu).expect("commit snapshot");
        state.commit(ticket, &mut gpu).expect("commit");
        assert!(matches!(
            state.commit(ticket, &mut gpu),
            Err(StateError::SnapshotInactive)
        ));

        let ticket = state.snapshot(&mut gpu).expect("reset snapshot");
        state.reset(&mut gpu).expect("reset");
        assert!(matches!(
            state.restore(&mut gpu, ticket),
            Err(StateError::SnapshotInactive)
        ));
        state.free_gpu(&mut gpu).expect("free state");
    }

    #[test]
    fn snapshot_arena_restores_device_state_and_marks() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let mut state = new_compact_state(&mut gpu);
        let recurrent_size = state.gdn[0].recurrent.byte_size();
        let expected = 1.25f32.to_ne_bytes();
        let mut original = vec![0u8; recurrent_size];
        original[..expected.len()].copy_from_slice(&expected);
        gpu.hip
            .memcpy_htod(&state.gdn[0].recurrent.buf, &original)
            .expect("upload recurrent fixture");
        state.position = 3;
        let qsa = &mut state.qsa[0];
        qsa.full_len = 3;
        qsa.raw_len = 3;
        qsa.pooled_len = 1;
        qsa.partial_len = 2;
        qsa.selected_len = 3;
        qsa.position = 3;

        let ticket = state.snapshot(&mut gpu).expect("snapshot");
        gpu.hip
            .memset(
                &state.gdn[0].recurrent.buf,
                0,
                state.gdn[0].recurrent.buf.size(),
            )
            .expect("mutate recurrent fixture");
        state.position = 7;
        state.qsa[0].full_len = 7;
        state.qsa[0].raw_len = 7;
        state.qsa[0].pooled_len = 2;
        state.qsa[0].partial_len = 4;
        state.qsa[0].selected_len = 7;
        state.qsa[0].position = 7;

        state.restore(&mut gpu, ticket).expect("restore");
        assert_eq!(state.position, 3);
        assert_eq!(state.qsa[0].full_len, 3);
        assert_eq!(state.qsa[0].raw_len, 3);
        let mut restored = [0u8; 4];
        gpu.hip
            .memcpy_dtoh(&mut restored, &state.gdn[0].recurrent.buf)
            .expect("download restored recurrent fixture");
        assert_eq!(restored, expected);
        state.free_gpu(&mut gpu).expect("free state");
    }

    const MIB: usize = 1 << 20;

    #[test]
    fn qsa_context_plan_maps_each_arena_by_its_own_stride() {
        // Flash-Next F32 rows at a 2 MiB driver page: K/V 2 KV heads x 256,
        // raw and pooled index keys one 128-wide head.
        let s = 262_144;
        let full = context_arena_plan(2048, s, 2 * MIB).unwrap();
        let first = full.growth(0, 1).unwrap().unwrap();
        assert_eq!((first.offset_bytes, first.size_bytes), (0, 2 * MIB));
        assert_eq!(full.token_capacity(2 * MIB), 1024);
        assert_eq!(full.growth(2 * MIB, 1024).unwrap(), None);
        let next = full.growth(2 * MIB, 1025).unwrap().unwrap();
        assert_eq!((next.offset_bytes, next.size_bytes), (2 * MIB, 2 * MIB));
        let whole = full.growth(0, s).unwrap().unwrap();
        assert_eq!(whole.size_bytes, 512 * MIB);
        assert!(full.growth(512 * MIB, s + 1).is_err());

        let raw = context_arena_plan(512, s, 2 * MIB).unwrap();
        assert_eq!(
            raw.token_capacity(raw.growth(0, 1).unwrap().unwrap().size_bytes),
            4096
        );
        // Pooled keys map ceil(required / 4) rows: 4097 tokens are 1025 rows,
        // inside the first 4096-row page, which covers 16384 tokens.
        let pooled = context_arena_plan(512, s.div_ceil(4), 2 * MIB).unwrap();
        let rows = pooled.token_capacity(
            pooled
                .growth(0, 4097usize.div_ceil(4))
                .unwrap()
                .unwrap()
                .size_bytes,
        );
        assert_eq!(rows, 4096);
        assert_eq!(qsa_covered_tokens(rows, 4, s), 16_384);
        // A short state's last pooled row covers its ragged tail only.
        assert_eq!(qsa_covered_tokens(10usize.div_ceil(4), 4, 10), 10);

        // FP8 K/V rows are not a power of two: whole rows per page only.
        assert_eq!(QsaKvFormat::Fp8.kv_row_bytes(2, 256), 516);
        let fp8 = context_arena_plan(516, s, 2 * MIB).unwrap();
        assert_eq!(fp8.token_capacity(2 * MIB), 4064);
    }

    /// gfx1100 map-only discipline for the QSA context arenas
    /// (`ensure_mapped_capacity` / `map_ahead_for_decode` ->
    /// `grow_context_arena`): across an 8K prefill, a decode map-ahead, a
    /// reset, a shorter request and a 32K request, every arena's plan only
    /// ever asks for a map at its current mapped frontier. The plan has no
    /// shrink or remap request, a covered request is a no-op, and a request
    /// after `reset` (which never unmaps) maps nothing it already holds.
    #[test]
    fn gfx1100_vmm_qsa_context_growth_only_maps_forward_at_the_frontier() {
        let s: usize = 262_144;
        let compress: usize = 4;
        // (row bytes, tokens per row, F16 four-token blocks) per arena kind
        // at the q8/F32/fp8 strides of the neighbouring plan test.
        let arenas: [(usize, usize, bool); 4] = [
            (2048, 1, false),
            (512, 1, false),
            (512, compress, false),
            (516, 1, true),
        ];
        for granularity in [4096, 2 * MIB] {
            for (row_bytes, tokens_per_row, blocks) in arenas {
                let capacity_rows = s.div_ceil(tokens_per_row);
                let plan = context_arena_plan(row_bytes, capacity_rows, granularity).unwrap();
                let mut mapped = 0usize;
                let mut maps = 0usize;
                let map_ahead = decode_map_ahead_tokens(8204, Some(64), 3, s).unwrap();
                // Prefill tiles, decode map-ahead, a reset-then-short request,
                // a longer one, and a capacity-sized request.
                for required in [
                    512, 1024, 4096, 8204, map_ahead, 1, 700, 8204, 20_000, 32_780, 700, s,
                ] {
                    let rows = required.div_ceil(tokens_per_row);
                    let rows = if blocks {
                        qsa_f16_arena_rows(rows).min(capacity_rows)
                    } else {
                        rows
                    };
                    match plan.growth(mapped, rows).unwrap() {
                        Some(growth) => {
                            assert_eq!(
                                growth.offset_bytes, mapped,
                                "growth must map at the frontier, never inside the mapped prefix"
                            );
                            assert!(growth.size_bytes > 0);
                            assert_eq!(growth.size_bytes % granularity, 0);
                            mapped += growth.size_bytes;
                            maps += 1;
                            assert_eq!(growth.token_capacity, plan.token_capacity(mapped));
                        }
                        None => assert!(
                            plan.token_capacity(mapped) >= rows,
                            "no growth is only valid when the prefix covers the request"
                        ),
                    }
                    assert!(plan.token_capacity(mapped) >= rows);
                }
                assert!(maps >= 1 && mapped <= plan.reserve_bytes());
            }
        }
    }

    /// Source audit (CPU): the Qwen4 QSA arena paths never unmap, release or
    /// reallocate a live VMM owner. `reset_inner` clears in place with
    /// `memset_batched` (a VMM owner's buffer is its mapped prefix); growth is one
    /// `grow_vmm_tensor` forward map plus a memset of exactly the new bytes;
    /// hipMemUnmap is not called anywhere in this crate.
    #[test]
    fn gfx1100_vmm_qsa_reset_and_growth_never_unmap_or_reallocate_arenas() {
        fn code(source: &str, signature: &str, end: &str) -> String {
            let start = source
                .find(signature)
                .unwrap_or_else(|| panic!("{signature} not found"));
            let rest = &source[start..];
            let body = &rest[..rest.find(end).expect("end of function")];
            body.lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        }
        let src = include_str!("state.rs");
        let unmap = format!("{}{}", "mem_", "unmap");
        let forbidden = [
            unmap.as_str(),
            "mem_release",
            "free_tensor",
            "release_tensor",
            "alloc_vmm_tensor",
            "reserve(",
            "zeros(",
            "std::mem::replace",
            "std::mem::take",
        ];

        let reset = code(src, "    fn reset_inner(", "\n    }\n");
        for token in forbidden.into_iter().chain(["grow_vmm_tensor"]) {
            assert!(!reset.contains(token), "reset_inner contains `{token}`");
        }
        assert!(
            reset.contains(".memset_batched("),
            "reset clears arenas in place"
        );
        // One lane reset enqueues every fill and waits once: no per-buffer wait between
        // the fills (the extent check's post-launch wait is the one developer exception).
        assert!(
            !reset.contains(".memset_stream_ordered("),
            "reset_inner must not wait per buffer"
        );
        assert_eq!(
            reset.matches("memset_batch_sync(").count(),
            1,
            "reset_inner waits only for the extent post-launch check"
        );
        let synced = code(src, "    fn reset_synced(", "\n    }\n");
        assert_eq!(synced.matches("memset_batch_sync(").count(), 1);
        // The kill switch (`HIPFIRE_LANE_RESET_BATCH=0`) is today's per-buffer wait.
        let dispatch = include_str!("../../rdna-compute/src/dispatch.rs");
        let batched = code(dispatch, "    pub fn memset_batched(", "\n    }\n");
        assert!(
            batched.contains("!self.flags.lane_reset_batch")
                && batched.contains("return self.memset_stream_ordered("),
            "memset_batched falls back to memset_stream_ordered under the kill switch"
        );
        // The whole lane reset (trunk, head, pending hidden) shares one batch.
        let lane = include_str!("lane.rs");
        let cell = code(lane, "fn reset_cell_into(", "\n}\n");
        assert!(cell.contains(".reset_into(gpu, batch)") && !cell.contains("memset"));
        let spec = code(
            lane,
            "    fn reset_into(&mut self, gpu: &mut Gpu, batch",
            "\n    }\n",
        );
        assert!(spec.contains(".memset_batched(") && !spec.contains("memset_stream_ordered"));
        let admit = code(
            lane,
            "                    Some(cell) => {\n                        // Trunk state",
            "                        Ok(None)",
        );
        assert_eq!(admit.matches("memset_batch_sync(").count(), 1);

        let ensure = code(src, "    pub fn ensure_mapped_capacity(", "\n    }\n");
        for token in forbidden.into_iter().chain(["memset", "grow_vmm_tensor"]) {
            assert!(
                !ensure.contains(token),
                "ensure_mapped_capacity contains `{token}`"
            );
        }
        assert!(ensure.contains("grow_context_arena("));

        let grow = code(src, "fn grow_context_arena(", "\n}\n");
        for token in forbidden {
            assert!(
                !grow.contains(token),
                "grow_context_arena contains `{token}`"
            );
        }
        assert_eq!(grow.matches(".grow_vmm_tensor(").count(), 1);
        // The zero-fill covers the view of the bytes just mapped, never the
        // already-live prefix.
        assert!(grow.contains("tensor.sub_offset(growth.offset_bytes / element"));

        fn scan(dir: &std::path::Path, needle: &str, hits: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("read_dir") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    scan(&path, needle, hits);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let text = std::fs::read_to_string(&path).expect("read source");
                    if text.contains(needle) {
                        hits.push(path.display().to_string());
                    }
                }
            }
        }
        let mut hits = Vec::new();
        scan(
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
            &format!(".{unmap}("),
            &mut hits,
        );
        assert!(
            hits.is_empty(),
            "direct hipMemUnmap in arch-qwen4: {hits:?}"
        );
    }

    /// gfx942 serves fp8 K/V only on an explicit request, as the FNUZ format
    /// (F32 index keys); `auto`/unset/`bf16` stay on the F32 reference state.
    /// gfx1201 keeps its OCP `Fp8` (BF16 index) default; the RDNA3 q8 default
    /// is covered by `rdna3_auto_is_q8_gfx1201_fp8_other_archs_f32`.
    #[test]
    fn gfx942_explicit_fp8_is_fnuz_and_auto_stays_f32() {
        let config = crate::config::compact_test_config();
        for request in ["", "auto", " auto ", "bf16"] {
            assert_eq!(
                resolve_qsa_format_for_arch(request, "gfx942", &config),
                Ok(QsaKvFormat::F32),
                "{request:?} on gfx942"
            );
        }
        for request in ["fp8", " fp8 "] {
            assert_eq!(
                resolve_qsa_format_for_arch(request, "gfx942", &config),
                Ok(QsaKvFormat::Fp8Fnuz),
                "{request:?} on gfx942"
            );
        }
        assert!(resolve_qsa_format_for_arch("q8", "gfx942", &config).is_err());
        // The RDNA formats are unchanged.
        for request in ["", "auto", "fp8"] {
            assert_eq!(
                resolve_qsa_format_for_arch(request, "gfx1201", &config),
                Ok(QsaKvFormat::Fp8),
                "{request:?} on gfx1201"
            );
        }
        // An odd KV-head count has no FNUZ kernel: explicit fp8 is refused,
        // never rewritten, and auto stays F32.
        let mut odd = config.clone();
        odd.num_key_value_heads = 3;
        let error = resolve_qsa_format_for_arch("fp8", "gfx942", &odd).unwrap_err();
        assert!(error.contains("even KV-head count"), "{error}");
        assert_eq!(
            resolve_qsa_format_for_arch("auto", "gfx942", &odd),
            Ok(QsaKvFormat::F32)
        );
    }

    /// `auto`/unset is q8 on exact gfx1151 and exact gfx1100 (where the
    /// geometry holds), fp8 on exact gfx1201 and the F32 reference state on
    /// every other arch; explicit `bf16` is the exact F32 state everywhere and
    /// explicit `q8` is served on every gfx11/gfx12 arch.
    #[test]
    fn rdna3_auto_is_q8_gfx1201_fp8_other_archs_f32() {
        let config = crate::config::compact_test_config();
        for arch in ["gfx1151", "gfx1100"] {
            for request in ["", "auto", " auto ", "q8"] {
                assert_eq!(
                    resolve_qsa_format_for_arch(request, arch, &config),
                    Ok(QsaKvFormat::Q8),
                    "{request:?} on {arch}"
                );
            }
            for request in ["bf16", " bf16 "] {
                assert_eq!(
                    resolve_qsa_format_for_arch(request, arch, &config),
                    Ok(QsaKvFormat::F32),
                    "{request:?} on {arch}"
                );
            }
            assert!(resolve_qsa_format_for_arch("fp8", arch, &config).is_err());
        }
        for request in ["", "auto"] {
            assert_eq!(
                resolve_qsa_format_for_arch(request, "gfx1201", &config),
                Ok(QsaKvFormat::Fp8),
                "{request:?} on gfx1201"
            );
        }
        for arch in ["gfx1101", "gfx1102", "gfx1150", "gfx1152", "gfx1200"] {
            for request in ["", "auto"] {
                assert_eq!(
                    resolve_qsa_format_for_arch(request, arch, &config),
                    Ok(QsaKvFormat::F32),
                    "{request:?} on {arch}"
                );
            }
            assert_eq!(
                resolve_qsa_format_for_arch("q8", arch, &config),
                Ok(QsaKvFormat::Q8),
                "q8 on {arch}"
            );
        }
        for arch in ["gfx942", "gfx1030"] {
            for request in ["", "auto"] {
                assert_eq!(
                    resolve_qsa_format_for_arch(request, arch, &config),
                    Ok(QsaKvFormat::F32),
                    "{request:?} on {arch}"
                );
            }
        }
        // A geometry q8 does not serve (head_dim != 256): auto falls back to
        // the F32 state, an explicit q8 is refused, never rewritten, and the
        // explicit bf16 state is unaffected.
        let mut narrow = config.clone();
        narrow.head_dim = 128;
        for arch in ["gfx1100", "gfx1151"] {
            for request in ["", "auto", "bf16"] {
                assert_eq!(
                    resolve_qsa_format_for_arch(request, arch, &narrow),
                    Ok(QsaKvFormat::F32),
                    "{request:?} on {arch} with head_dim 128"
                );
            }
            let error = resolve_qsa_format_for_arch("q8", arch, &narrow).unwrap_err();
            assert!(error.contains("q8 QSA K/V needs head_dim 256"), "{error}");
        }
    }

    /// Flash-Next (H24 Hkv2 D256, one 128-wide index head, compress 4) charge
    /// of the gfx942 FP8 state: K and V rows are 2 x (2 x (256 + 2)) = 1032 B,
    /// the F32 raw index key 512 B and the pooled key 128 B per token, so
    /// 1672 B per layer-token and 20,064 B over the 12 QSA layers (the F32
    /// reference state is 4736 B and 56,832 B).
    #[test]
    fn gfx942_fp8_state_charge_is_1672_bytes_per_layer_token() {
        let config = crate::config::compact_test_config();
        let fnuz = QsaKvFormat::Fp8Fnuz;
        assert_eq!(fnuz.kv_row_bytes(2, 256), 516);
        assert_eq!(fnuz.index_dtype(), DType::F32);
        assert_eq!(config.n_full_layers(), 12);
        let s = 4096; // a multiple of the compress ratio: pooled rows = s / 4.
        let layer = config.qsa_context_arena_bytes(s, fnuz).unwrap();
        assert_eq!(layer, s * 1672);
        assert_eq!(layer * config.n_full_layers(), s * 20_064);
        assert_eq!(
            config.qsa_context_arena_bytes(s, QsaKvFormat::F32).unwrap(),
            s * 4736
        );
        // The RDNA fp8 row keeps its BF16 index keys.
        assert_eq!(
            config.qsa_context_arena_bytes(s, QsaKvFormat::Fp8).unwrap(),
            s * (1032 + 256 + 64)
        );
        // Legacy storage (gfx942 is not a VMM arch) commits the whole layer.
        let context = Qwen4ContextCommit::legacy(s);
        assert_eq!(context.committed_layer_bytes(&config, fnuz), Some(layer));
        // The state charge differs from the F32 state by exactly the K/V and
        // index-row savings of the QSA layers (GDN state untouched).
        let state = |qsa| Qwen4StateFormat {
            qsa,
            ..Qwen4StateFormat::F32
        };
        assert_eq!(
            Qwen4State::device_bytes(&config, state(QsaKvFormat::F32), &context).unwrap()
                - Qwen4State::device_bytes(&config, state(fnuz), &context).unwrap(),
            s * (4736 - 1672) * config.n_full_layers()
        );
    }

    /// The gfx942 FNUZ encoder feeds `(v / s) * 0.5` to a round-to-nearest-even
    /// FNUZ convert, which cannot produce `0x80` (its NaN). Modeled on the CPU
    /// FNUZ table: every finite OCP code except `0x80` is stored as itself, the
    /// quantized negative zero `0x80` is stored as `0x00`, and `0x80` is never
    /// stored, including for an input of -0 and for negative nonzero inputs
    /// that round to zero.
    #[test]
    fn gfx942_fp8_kv_codes_canonicalize_negative_zero_and_never_store_0x80() {
        use rdna_compute::fp8_e4m3::{e4m3fn_to_f32, e4m3fnuz_to_f32};
        let encode = |x: f32| -> u8 {
            let mut best: Option<(f64, u8)> = None;
            for code in 0..=255u8 {
                let Some(value) = e4m3fnuz_to_f32(code) else {
                    continue; // 0x80: FNUZ NaN, never produced
                };
                let distance = (f64::from(x) - f64::from(value)).abs();
                best = match best {
                    Some((d, c)) if d < distance || (d == distance && c & 1 == 0) => Some((d, c)),
                    _ => Some((distance, code)),
                };
            }
            best.unwrap().1
        };
        let mut finite = 0;
        for code in 0..=255u8 {
            let Some(value) = e4m3fn_to_f32(code) else {
                continue; // 0x7F / 0xFF: OCP NaN
            };
            finite += 1;
            let stored = encode(value * 0.5);
            assert_ne!(stored, 0x80, "code {code:#04x} stored 0x80");
            assert_eq!(
                stored,
                if code == 0x80 { 0x00 } else { code },
                "code {code:#04x}"
            );
        }
        assert_eq!(finite, 254);
        // -0, a negative value below the smallest FNUZ subnormal (2^-10), and
        // the exact tie at half of it (to even: zero) all store +0.
        for x in [-0.0f32, -1e-9, -2f32.powi(-12), -2f32.powi(-11)] {
            assert_eq!(encode(x), 0x00, "{x:e}");
        }
        // Just past the tie rounds to the smallest negative subnormal 0x81.
        assert_eq!(encode(-2f32.powi(-11) * 1.0001), 0x81);
    }

    #[test]
    fn row_capture_bytes_charge_the_gdn_storage_dtype() {
        let config = crate::config::compact_test_config();
        // 36 GDN layers x (F32 inputs 4R x 10240 and recurrence 4R x 10336) + the ring slots
        // (the shared pool 2 * 35 + R + 1 = 75 slots by default; 36 x (R + 1) with
        // HIPFIRE_QWEN4_LANE_RESERVE_TRIM=0, 36 x 2R with the compact kill switch)
        // + 4R x 6144 F32 output + 36 x 16 table.
        assert_eq!(capture_ring_slots(false, 4), Some(8));
        assert_eq!(capture_ring_slots(true, 4), Some(5));
        assert_eq!(capture_ring_slots(compact_capture_ring(), 4), Some(5));
        assert_eq!(capture_ring_pool_slots(4, 36), Some(75));
        let (f32_bytes, q8_bytes) = if lane_reserve_trim() {
            (247_880_256, 72_776_256)
        } else {
            (578_181_696, 157_932_096)
        };
        assert_eq!(
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::F32, 4),
            Some(f32_bytes)
        );
        // Q8 slots are 48 x 128 x 132 bytes, not 3 MiB of F32.
        assert_eq!(
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, 4),
            Some(q8_bytes)
        );
        assert_eq!(
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, 1),
            Some(0)
        );
    }

    /// The rank-local TP2 trunk-shard config (24 value / 8 key GDN heads) sizes the GDN state
    /// from its own head counts: the recurrent slot of every format, its live F16 residual, the
    /// 3-row conv ring and the verify ring all halve against the global config while the QSA
    /// arenas (KV replicated) and everything else stay byte-for-byte, and the global config's
    /// figures are unchanged.
    #[test]
    fn trunk_shard_local_config_halves_the_gdn_state_and_the_verify_ring() {
        let global = crate::config::compact_test_config()
            .into_ar_only()
            .expect("ar-only");
        let locals = [0, 1].map(|rank| global.trunk_shard_local(rank, 2).expect("local config"));
        // The global figures of `row_capture_bytes_charge_the_gdn_storage_dtype`.
        let trim = lane_reserve_trim();
        assert_eq!(
            Qwen4State::row_capture_bytes(&global, GdnStateFormat::F32, 4),
            Some(if trim { 247_880_256 } else { 578_181_696 })
        );
        assert_eq!(
            Qwen4State::row_capture_bytes(&global, GdnStateFormat::Q8, 4),
            Some(if trim { 72_776_256 } else { 157_932_096 })
        );
        for local in &locals {
            // 36 x (F32 inputs 4R x 5120 and recurrence 4R x 5168) + the ring slots (the shared
            // 75-slot pool; 36 x (R+1) with HIPFIRE_QWEN4_LANE_RESERVE_TRIM=0) + 4R x 3072 F32
            // output + 36 x 16 table; Q8 slots are 24 x 128 x 132 bytes.
            assert_eq!(
                Qwen4State::row_capture_bytes(local, GdnStateFormat::F32, 4),
                Some(if trim { 123_940_416 } else { 289_091_136 })
            );
            assert_eq!(
                Qwen4State::row_capture_bytes(local, GdnStateFormat::Q8, 4),
                Some(if trim { 36_388_416 } else { 78_966_336 })
            );
            assert_eq!(
                Qwen4State::row_capture_bytes(local, GdnStateFormat::Q8, 1),
                Some(0)
            );
        }
        let (heads, key_heads) = (
            locals[0].linear_num_value_heads,
            locals[0].linear_num_key_heads,
        );
        assert_eq!((heads, key_heads), (24, 8));
        let conv = (2 * key_heads * GDN_HEAD_DIM + heads * GDN_HEAD_DIM)
            * locals[0].linear_conv_kernel_dim.saturating_sub(1)
            * std::mem::size_of::<f32>();
        for (name, format) in allocation_formats() {
            let slot = format
                .gdn
                .storage()
                .state_bytes(heads, GDN_HEAD_DIM, GDN_HEAD_DIM);
            let residual = format.gdn.residual_bytes(heads, GDN_HEAD_DIM, GDN_HEAD_DIM);
            // Recurrent + conv, doubled by the snapshot mirror, plus the live-only residual.
            let gdn_layer = 2 * (slot + conv) + residual;
            for context in allocation_contexts() {
                let whole = Qwen4State::device_bytes(&global, format, &context).expect("global");
                for local in &locals {
                    let part = Qwen4State::device_bytes(local, format, &context).expect("local");
                    assert_eq!(
                        whole - part,
                        local.n_linear_layers() * gdn_layer,
                        "{name} {context:?}"
                    );
                }
                assert_eq!(
                    Qwen4State::device_bytes(&locals[0], format, &context),
                    Qwen4State::device_bytes(&locals[1], format, &context),
                    "{name} {context:?}: the ranks hold equal state"
                );
            }
        }
    }

    #[test]
    fn row_capture_64_byte_plan_covers_both_halves_and_all_inputs() {
        let config = crate::config::compact_test_config();
        let rows = hipfire_config::QWEN4_VERIFY_MAX_ROWS;
        assert_eq!(rows, 64);
        let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim;
        for format in [
            GdnStateFormat::F32,
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
        ] {
            for layers in [0, 1, config.n_linear_layers()] {
                for n in 0..=rows {
                    let expected = if n < 2 || layers == 0 {
                        0
                    } else {
                        let ring = current_capture_ring_total_slots(n, layers).unwrap()
                            * format.state_bytes(
                                config.linear_num_value_heads,
                                config.linear_key_head_dim,
                                config.linear_value_head_dim,
                            );
                        let inputs = 4 * n * (2 * qkv + 2 * config.linear_num_value_heads);
                        let discard = 4 * n * config.linear_num_value_heads * GDN_HEAD_DIM;
                        ring + layers * inputs + discard + 16 * layers
                    };
                    assert_eq!(
                        Qwen4State::row_capture_bytes_scoped(&config, format, n, layers),
                        Some(expected),
                        "{format:?} rows={n} layers={layers}",
                    );
                }
            }
            for n in [rows + 1, usize::MAX / 2, usize::MAX] {
                assert_eq!(Qwen4State::row_capture_bytes(&config, format, n), None);
            }
            for field in 0..5 {
                let mut huge = config.clone();
                match field {
                    0 => huge.linear_num_value_heads = usize::MAX,
                    1 => huge.linear_num_key_heads = usize::MAX,
                    2 => huge.linear_key_head_dim = usize::MAX,
                    3 => huge.linear_value_head_dim = usize::MAX,
                    _ => {}
                }
                let layers = if field == 4 { usize::MAX } else { 1 };
                assert_eq!(
                    Qwen4State::row_capture_bytes_scoped(&huge, format, rows, layers),
                    None,
                    "{format:?} overflowing field {field}",
                );
            }
        }
    }

    fn capture_test_config() -> Qwen4Config {
        let mut config = crate::config::synthetic_test_config(
            crate::config::Qwen4FfnConfig::Dense {
                intermediate_size: 16,
            },
            16,
            1,
            false,
            false,
        );
        config.linear_num_key_heads = 1;
        config.linear_num_value_heads = 2;
        config.linear_key_head_dim = GDN_HEAD_DIM;
        config.linear_value_head_dim = GDN_HEAD_DIM;
        config
    }

    // EP's canonical digest deliberately retains capture storage and physical
    // live indices for same-call receipts. AR-vs-verify logical equality must
    // omit those transaction details, not compare untouched ring slots.
    fn capture_semantic_digest(state: &Qwen4State, gpu: &mut Gpu) -> Vec<(String, String)> {
        state
            .canonical_digest_rows(gpu)
            .expect("digest")
            .into_iter()
            .filter(|(key, _)| !key.starts_with("row_capture[") && !key.ends_with("].live"))
            .collect()
    }

    #[test]
    fn row_capture_64_growth_refuses_before_mutation_and_preserves_live_state() {
        let Some(mut gpu) = try_gpu() else { return };
        let config = capture_test_config();
        let max = hipfire_config::QWEN4_VERIFY_MAX_ROWS;
        let history = config.linear_conv_kernel_dim - 1;
        let mut state =
            Qwen4State::new(&mut gpu, &config, 256, Qwen4StateFormat::F32).expect("state");
        state
            .ensure_row_capture(&mut gpu, 4, history)
            .expect("initial ring");
        state.set_gdn_live(7);
        gpu.hip
            .memset(
                &state.gdn[0].recurrent.buf,
                0x35,
                state.gdn[0].recurrent.byte_size(),
            )
            .expect("seed old live slot");
        let before = capture_semantic_digest(&state, &mut gpu);
        let pointer = state.gdn[0].recurrent.buf.as_ptr();
        let generation = state.snapshot_arena.generation;
        for rows in [max + 1, usize::MAX] {
            assert!(matches!(
                state.ensure_row_capture(&mut gpu, rows, history),
                Err(StateError::Length { .. })
            ));
        }
        assert!(matches!(
            state.ensure_row_capture(&mut gpu, max, 0),
            Err(StateError::Config(_))
        ));
        let heads = state.gdn_value_heads;
        state.gdn_value_heads = usize::MAX;
        assert!(matches!(
            state.ensure_row_capture(&mut gpu, max, history),
            Err(StateError::DimensionOverflow)
        ));
        state.gdn_value_heads = heads;
        gpu.graphs.capture_mode = true;
        let capturing = state.ensure_row_capture(&mut gpu, max, history);
        gpu.graphs.capture_mode = false;
        assert!(matches!(capturing, Err(StateError::Config(_))));
        assert_eq!(state.snapshot_arena.generation, generation);
        assert_eq!(state.gdn[0].recurrent.buf.as_ptr(), pointer);
        assert_eq!(state.row_capture_rows, 4);
        assert_eq!(state.gdn_live, 7);
        assert_eq!(capture_semantic_digest(&state, &mut gpu), before);
        // The real upload seam fails publication's staged pointer table;
        // it is not a simulated memory-budget or malloc refusal.
        hip_bridge::arm_hip_fault("upload", 1);
        let staged_failure = state.ensure_row_capture(&mut gpu, max, history);
        hip_bridge::arm_hip_fault("upload", 0);
        assert!(matches!(&staged_failure, Err(StateError::Hip(error))
            if error.to_string().contains("injected upload failure")));
        assert_eq!(state.snapshot_arena.generation, generation);
        assert_eq!(state.gdn[0].recurrent.buf.as_ptr(), pointer);
        assert_eq!((state.row_capture_rows, state.gdn_live), (4, 7));
        assert_eq!(capture_semantic_digest(&state, &mut gpu), before);

        state.row_capture_armed = true;
        let ticket = state.snapshot(&mut gpu).expect("armed snapshot");
        assert!(matches!(
            state.ensure_row_capture(&mut gpu, max, history),
            Err(StateError::SnapshotBusy)
        ));
        state.row_capture_armed = false;
        state.retain_validated(ticket);
        assert!(matches!(
            state.ensure_row_capture(&mut gpu, max, history),
            Err(StateError::SnapshotBusy)
        ));
        state.release_retained();
        state
            .ensure_row_capture(&mut gpu, max, history)
            .expect("grow to 64");
        assert_eq!(state.row_capture_rows, max);
        assert_eq!(state.gdn_live, 0);
        assert_eq!(capture_semantic_digest(&state, &mut gpu), before);
        assert!(state.snapshot_arena.generation > generation);
        let current = state.snapshot(&mut gpu).expect("new generation");
        assert!(matches!(
            state.validate_commit(ticket),
            Err(StateError::SnapshotTicket)
        ));
        state.restore(&mut gpu, current).expect("restore current");
        assert!(matches!(
            state.validate_commit(current),
            Err(StateError::SnapshotInactive)
        ));
        // After the failed staging and successful growth, a real singleton
        // recurrence can use the published live slot and scratch.
        let qk = config.linear_num_key_heads * GDN_HEAD_DIM;
        let value = config.linear_num_value_heads * GDN_HEAD_DIM;
        rdna_compute::tensor_ops::gated_delta_step(
            &mut gpu,
            &rdna_compute::tensor_ops::GatedDeltaStep {
                q: &state.gdn[0].conv.sub_offset(0, qk),
                k: &state.gdn[0].conv.sub_offset(qk, qk),
                v: &state.gdn[0].conv.sub_offset(2 * qk, value),
                gate: &state.gdn[0]
                    .conv
                    .sub_offset(0, config.linear_num_value_heads),
                beta: &state.gdn[0]
                    .conv
                    .sub_offset(0, config.linear_num_value_heads),
                state: &state.gdn[0].recurrent,
                format: GdnStateFormat::F32,
                residual: None,
                output: &state
                    .row_capture_output
                    .as_ref()
                    .expect("discard output")
                    .0
                    .sub_offset(0, value),
                key_heads: config.linear_num_key_heads,
                value_heads: config.linear_num_value_heads,
                key_dim: GDN_HEAD_DIM,
                value_dim: GDN_HEAD_DIM,
                position: state.position,
            },
        )
        .expect("singleton after failed staging and ring growth");
        state.free_gpu(&mut gpu).expect("free");
    }

    /// Exercises the real state repair entrypoint, not a replay transcription:
    /// 64-row capture, every kept prefix (zero means abort), both halves,
    /// subsequent singleton and another captured window. GDN-only fixture;
    /// full-model QSA/PLE/transaction gates remain the strict wide oracle.
    #[test]
    fn row_capture_64_prefix_abort_and_continuation_match_singletons() {
        use rdna_compute::tensor_ops::{
            gated_delta_step, gated_delta_step_batched, GatedDeltaStep, GatedDeltaStepBatched,
        };
        let Some(mut gpu) = try_gpu() else { return };
        let config = capture_test_config();
        let rows = hipfire_config::QWEN4_VERIFY_MAX_ROWS;
        let start = 37;
        let history = config.linear_conv_kernel_dim - 1;
        let value = config.linear_num_value_heads * GDN_HEAD_DIM;
        let qk = config.linear_num_key_heads * GDN_HEAD_DIM;
        let qkv = 2 * qk + value;
        let wave = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 37 % 1009) as f32 - 504.0) * scale)
                .collect()
        };
        let projection = wave((rows + 3) * qkv, 0.002);
        let gate = vec![-0.6; (rows + 3) * config.linear_num_value_heads];
        let beta = vec![0.4; gate.len()];
        let proj = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gates = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let betas = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let initial = gpu
            .upload_f32(&wave(value * GDN_HEAD_DIM, 0.0003), &[value * GDN_HEAD_DIM])
            .expect("seed");
        let output = gpu
            .zeros(&[(rows + 3) * value], DType::F32)
            .expect("output");
        // No PLE owner in this fixture, so the repair never reads this view.
        let ple = gpu.zeros(&[1], DType::F32).expect("unused PLE input");
        let tokens: Vec<_> = (0..rows + 3).map(|i| 10 + i as u32).collect();
        let singleton = |gpu: &mut Gpu, state: &Qwen4State, row: usize| {
            gated_delta_step(
                gpu,
                &GatedDeltaStep {
                    q: &proj.sub_offset(row * qkv, qk),
                    k: &proj.sub_offset(row * qkv + qk, qk),
                    v: &proj.sub_offset(row * qkv + 2 * qk, value),
                    gate: &gates.sub_offset(
                        row * config.linear_num_value_heads,
                        config.linear_num_value_heads,
                    ),
                    beta: &betas.sub_offset(
                        row * config.linear_num_value_heads,
                        config.linear_num_value_heads,
                    ),
                    state: &state.gdn[0].recurrent,
                    format: state.gdn_format,
                    residual: None,
                    output: &output.sub_offset(row * value, value),
                    key_heads: config.linear_num_key_heads,
                    value_heads: config.linear_num_value_heads,
                    key_dim: GDN_HEAD_DIM,
                    value_dim: GDN_HEAD_DIM,
                    position: start + row,
                },
            )
            .expect("singleton recurrence");
        };
        let advance_history = |gpu: &mut Gpu, state: &mut Qwen4State, row: usize| {
            gpu.hip
                .memcpy_htod(
                    &state.gdn[0]
                        .conv
                        .sub_offset((start + row) % history * qkv, qkv)
                        .buf,
                    &projection[row * qkv..][..qkv]
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .expect("conv input");
            state.ple_history.push(tokens[row]);
            state.position = start + row + 1;
        };
        let capture = |gpu: &mut Gpu, state: &mut Qwen4State, first: usize, n: usize| {
            let captured = state.gdn_row_capture(0, n).expect("armed capture");
            let recurrence: Vec<u8> = projection[first * qkv..][..n * qkv]
                .iter()
                .chain(
                    &gate[first * config.linear_num_value_heads..]
                        [..n * config.linear_num_value_heads],
                )
                .chain(
                    &beta[first * config.linear_num_value_heads..]
                        [..n * config.linear_num_value_heads],
                )
                .flat_map(|x| x.to_ne_bytes())
                .collect();
            gpu.hip
                .memcpy_htod(&captured.recurrence.buf, &recurrence)
                .expect("recurrence inputs");
            let inputs: Vec<u8> = projection[first * qkv..][..n * qkv]
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect();
            gpu.hip
                .memcpy_htod(&captured.inputs.buf, &inputs)
                .expect("conv inputs");
            gated_delta_step_batched(
                gpu,
                &GatedDeltaStepBatched {
                    projection: &proj.sub_offset(first * qkv, n * qkv),
                    gate: &gates.sub_offset(
                        first * config.linear_num_value_heads,
                        n * config.linear_num_value_heads,
                    ),
                    beta: &betas.sub_offset(
                        first * config.linear_num_value_heads,
                        n * config.linear_num_value_heads,
                    ),
                    state: &state.gdn[0].recurrent,
                    format: state.gdn_format,
                    residual: None,
                    output: &output.sub_offset(first * value, n * value),
                    row_states: Some(&captured.states),
                    rows: n,
                    qkv_width: qkv,
                    key_heads: config.linear_num_key_heads,
                    value_heads: config.linear_num_value_heads,
                    key_dim: GDN_HEAD_DIM,
                    value_dim: GDN_HEAD_DIM,
                    position: start + first,
                },
            )
            .expect("captured recurrence");
            state.commit_row_capture(n);
            for row in first..first + n {
                advance_history(gpu, state, row);
            }
        };
        for format in [GdnStateFormat::F32, GdnStateFormat::Q8] {
            let state_format = Qwen4StateFormat {
                gdn: format,
                ..Qwen4StateFormat::F32
            };
            let mut reference =
                Qwen4State::new(&mut gpu, &config, 256, state_format).expect("reference");
            let mut candidate =
                Qwen4State::new(&mut gpu, &config, 256, state_format).expect("candidate");
            candidate
                .ensure_row_capture(&mut gpu, rows, history)
                .expect("64 ring");
            let reference_history = reference.ple_history;
            let seed = |gpu: &mut Gpu, state: &mut Qwen4State| {
                if format == GdnStateFormat::F32 {
                    gpu.copy_d2d(&initial, &state.gdn[0].recurrent, initial.byte_size())
                        .expect("seed F32 recurrence");
                } else {
                    gpu.hip
                        .memset(
                            &state.gdn[0].recurrent.buf,
                            0,
                            state.gdn[0].recurrent.byte_size(),
                        )
                        .expect("clear Q8 seed");
                    singleton(gpu, state, 0);
                }
                gpu.hip
                    .memset(&state.gdn[0].conv.buf, 0x32, state.gdn[0].conv.byte_size())
                    .expect("seed conv");
                state.position = start;
                state.ple_history = reference_history;
            };
            seed(&mut gpu, &mut reference);
            let mut expected = vec![capture_semantic_digest(&reference, &mut gpu)];
            for row in 0..rows + 3 {
                singleton(&mut gpu, &reference, row);
                advance_history(&mut gpu, &mut reference, row);
                expected.push(capture_semantic_digest(&reference, &mut gpu));
            }
            // The ring layout this allocation took (`HIPFIRE_QWEN4_CAPTURE_RING_COMPACT`, default
            // on: `rows + 1` slots, logical slots mapped by `capture_ring_slot`; `=0` is the
            // double ring where logical == physical). Run the test under both settings.
            let compact = candidate.row_capture_compact;
            let ring_slot = |state: &Qwen4State, logical: usize| {
                capture_ring_slot(state.row_capture_compact, rows, logical)
            };
            for base in [0, rows] {
                for keep in 0..=rows {
                    candidate.set_gdn_live(if base == 0 { 2 * rows - 1 } else { rows - 1 });
                    seed(&mut gpu, &mut candidate);
                    let old_live = candidate.gdn_live;
                    let state_units = candidate.gdn[0].recurrent.numel();
                    let old_bytes = gpu
                        .download_raw_bytes(&candidate.gdn[0].recurrent)
                        .expect("old live bytes");
                    if compact {
                        // Poison every physical slot but the old live one (the old state must
                        // survive the capture), the compact ring's equivalent of the free half.
                        for physical in 0..=rows {
                            if physical == ring_slot(&candidate, old_live) {
                                continue;
                            }
                            let slot = candidate.row_capture[0]
                                .0
                                .sub_offset(physical * state_units, state_units);
                            gpu.hip
                                .memset(&slot.buf, 0xa5, slot.byte_size())
                                .expect("poison free slot");
                        }
                    } else {
                        let destination = candidate.row_capture[0]
                            .0
                            .sub_offset(base * state_units, rows * state_units);
                        gpu.hip
                            .memset(&destination.buf, 0xa5, destination.byte_size())
                            .expect("poison free half");
                    }
                    candidate.row_capture_armed = true;
                    let ticket = candidate.snapshot(&mut gpu).expect("armed snapshot");
                    capture(&mut gpu, &mut candidate, 0, rows);
                    assert_eq!(candidate.gdn_live, base + rows - 1);
                    let old_slot = candidate.row_capture[0]
                        .0
                        .sub_offset(ring_slot(&candidate, old_live) * state_units, state_units);
                    assert_eq!(
                        gpu.download_raw_bytes(&old_slot).expect("old slot"),
                        old_bytes
                    );
                    if compact {
                        // Compact ring: only the captured slot and the old live slot hold
                        // state; every other physical slot is still the poison byte.
                        let written = [
                            ring_slot(&candidate, candidate.gdn_live),
                            ring_slot(&candidate, old_live),
                        ];
                        for physical in 0..=rows {
                            if written.contains(&physical) {
                                continue;
                            }
                            let slot = candidate.row_capture[0]
                                .0
                                .sub_offset(physical * state_units, state_units);
                            assert!(
                                gpu.download_raw_bytes(&slot)
                                    .expect("uncaptured slot")
                                    .iter()
                                    .all(|byte| *byte == 0xa5),
                                "intermediate state was materialized (physical slot {physical})"
                            );
                        }
                    } else {
                        for slot in [base, base + rows - 2] {
                            let intermediate = candidate.row_capture[0]
                                .0
                                .sub_offset(slot * state_units, state_units);
                            assert!(
                                gpu.download_raw_bytes(&intermediate)
                                    .expect("uncaptured slot")
                                    .iter()
                                    .all(|byte| *byte == 0xa5),
                                "intermediate state was materialized"
                            );
                        }
                    }
                    candidate.row_capture_armed = false;
                    if keep == 0 {
                        candidate.restore(&mut gpu, ticket).expect("abort");
                        assert_eq!(candidate.gdn_live, old_live);
                    } else {
                        if keep < rows {
                            candidate
                                .rollback_rows_retain(
                                    &mut gpu,
                                    ticket,
                                    keep,
                                    rows,
                                    &tokens,
                                    &ple,
                                    0,
                                    history,
                                    config.indexer_compress_ratio,
                                    config.indexer_budget,
                                )
                                .expect("native prefix repair");
                        }
                        assert_eq!(candidate.gdn_live, base + keep - 1);
                        candidate.commit(ticket, &mut gpu).expect("commit prefix");
                    }
                    assert_eq!(
                        capture_semantic_digest(&candidate, &mut gpu),
                        expected[keep],
                        "{format:?} base={base} keep={keep}"
                    );
                    singleton(&mut gpu, &candidate, keep);
                    advance_history(&mut gpu, &mut candidate, keep);
                    assert_eq!(
                        capture_semantic_digest(&candidate, &mut gpu),
                        expected[keep + 1]
                    );
                    let next_base = candidate.capture_base(candidate.gdn_live);
                    candidate.row_capture_armed = true;
                    let next = candidate.snapshot(&mut gpu).expect("continuation snapshot");
                    capture(&mut gpu, &mut candidate, keep + 1, 2);
                    candidate.row_capture_armed = false;
                    candidate
                        .commit(next, &mut gpu)
                        .expect("continuation commit");
                    assert_eq!(candidate.gdn_live, next_base + 1);
                    assert_eq!(
                        capture_semantic_digest(&candidate, &mut gpu),
                        expected[keep + 3]
                    );
                }
            }
            reference.free_gpu(&mut gpu).expect("free reference");
            candidate.free_gpu(&mut gpu).expect("free candidate");
        }
        for tensor in [proj, gates, betas, initial, output, ple] {
            gpu.free_tensor(tensor).expect("free fixture");
        }
    }

    #[test]
    fn map_ahead_draft_rows_cover_the_widest_scheduled_window() {
        // A window wider than the configured depth sets the bound.
        assert_eq!(map_ahead_draft_rows(3, 32), 31);
        assert_eq!(map_ahead_draft_rows(10, 9), 10);
        let max = hipfire_config::QWEN4_VERIFY_MAX_ROWS;
        assert_eq!(map_ahead_draft_rows(3, max), max - 1);
        // No wider window: the configured depth (an AR lane has none).
        assert_eq!(map_ahead_draft_rows(0, 0), 0);
        assert_eq!(map_ahead_draft_rows(0, 1), 0);
        assert_eq!(map_ahead_draft_rows(7, 8), 7);
        // The widest window's end stays inside the mapped target.
        let drafts = map_ahead_draft_rows(3, max);
        let target = decode_map_ahead_tokens(100, Some(64), drafts, 4096).unwrap();
        assert!(target >= 100 + 64 - 1 + max);
    }

    #[test]
    fn decode_map_ahead_is_budget_plus_draft_clamped_to_capacity() {
        // Budget plus draft rows past the position.
        assert_eq!(decode_map_ahead_tokens(100, Some(64), 0, 4096), Some(164));
        assert_eq!(decode_map_ahead_tokens(100, Some(64), 3, 4096), Some(167));
        // Clamped to the admitted capacity, at and past it.
        assert_eq!(decode_map_ahead_tokens(100, Some(64), 3, 167), Some(167));
        assert_eq!(decode_map_ahead_tokens(100, Some(64), 3, 150), Some(150));
        assert_eq!(decode_map_ahead_tokens(5000, Some(1), 0, 4096), Some(4096));
        // No budget (unbounded or finished) maps nothing ahead, draft or not.
        assert_eq!(decode_map_ahead_tokens(100, None, 3, 4096), None);
        assert_eq!(decode_map_ahead_tokens(100, Some(0), 3, 4096), None);
        assert_eq!(decode_map_ahead_tokens(100, Some(0), 0, 0), None);
        // Saturating sums never overflow; the capacity still bounds them.
        assert_eq!(
            decode_map_ahead_tokens(usize::MAX, Some(usize::MAX), usize::MAX, 4096),
            Some(4096)
        );
        assert_eq!(
            decode_map_ahead_tokens(1, Some(usize::MAX), 2, usize::MAX),
            Some(usize::MAX)
        );
        assert_eq!(decode_map_ahead_tokens(10, Some(1), 0, 0), Some(0));
    }

    #[test]
    fn legacy_context_ensure_is_a_bounds_check_over_full_allocations() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let config = crate::config::compact_test_config();
        let s = 1024;
        let mut state =
            Qwen4State::new(&mut gpu, &config, s, Qwen4StateFormat::F32).expect("legacy state");
        let full =
            config.qsa_context_arena_bytes(s, QsaKvFormat::F32).unwrap() * config.n_full_layers();
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), full);
        assert_eq!(state.mapped_context_tokens(), s);
        state
            .ensure_mapped_capacity(&mut gpu, s)
            .expect("admitted tokens");
        assert!(matches!(
            state.ensure_mapped_capacity(&mut gpu, s + 1),
            Err(StateError::ContextCapacity { required, admitted }) if required == s + 1 && admitted == s
        ));
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), full);
        state.free_gpu(&mut gpu).expect("free state");
    }

    /// Committed bytes and covered tokens the demand plan gives after
    /// monotonic requests up to `required` (F32 compact geometry).
    fn expected_vmm(granularity: usize, s: usize, required: usize) -> (usize, usize) {
        let mut bytes = 0;
        let mut tokens = s;
        for (stride, rows, per_row) in [
            (2048, s, 1),
            (2048, s, 1),
            (512, s, 1),
            (512, s.div_ceil(4), 4),
        ] {
            let chunk = (DEFAULT_KV_CHUNK_TOKENS * stride)
                .max(DEFAULT_VMM_PHYSICAL_CHUNK_BYTES)
                .next_multiple_of(granularity);
            let reserve = (stride * rows).next_multiple_of(granularity);
            let mapped = (required.div_ceil(per_row) * stride)
                .next_multiple_of(chunk)
                .min(reserve);
            bytes += mapped;
            tokens = tokens.min(((mapped / stride).min(rows) * per_row).min(s));
        }
        (bytes * 12, tokens)
    }

    #[test]
    fn vmm_context_maps_on_demand_and_refuses_above_admitted() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        if gpu.vmm_recommended_granularity().is_err() {
            eprintln!("skip: HIP VMM unavailable");
            return;
        }
        let owners = gpu.vmm_allocation_count();
        let config = crate::config::compact_test_config();
        assert_eq!(config.n_full_layers(), 12);
        let s = 16_384;
        let mut state = Qwen4State::new_with_backend(
            &mut gpu,
            &config,
            s,
            Qwen4StateFormat::F32,
            Qwen4KvBackend::Vmm,
        )
        .expect("VMM state");
        assert_eq!(gpu.vmm_allocation_count(), owners + 48);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), 0);
        assert_eq!(state.mapped_context_tokens(), 0);
        state.reset(&mut gpu).expect("reset with nothing mapped");

        // Above the admitted capacity: refused before any map or mark move.
        state.position = 5;
        state.qsa[0].full_len = 5;
        assert!(matches!(
            state.ensure_mapped_capacity(&mut gpu, s + 1),
            Err(StateError::ContextCapacity { required, admitted }) if required == s + 1 && admitted == s
        ));
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), 0);
        assert_eq!((state.position, state.qsa[0].full_len), (5, 5));

        let granularity = gpu.vmm_granularity(&state.qsa[0].full_keys).unwrap();
        state
            .ensure_mapped_capacity(&mut gpu, 1)
            .expect("first page");
        let (bytes, first) = expected_vmm(granularity, s, 1);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(state.mapped_context_tokens(), first);
        assert!(first < s);
        let pattern = vec![0xA5u8; 2048];
        gpu.hip
            .memcpy_htod(&state.qsa[0].full_keys.buf, &pattern)
            .expect("write row 0");
        state
            .ensure_mapped_capacity(&mut gpu, first)
            .expect("covered");
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);

        let old_size = state.qsa[0].full_keys.buf.size();
        state
            .ensure_mapped_capacity(&mut gpu, first + 1)
            .expect("grow");
        let (bytes, tokens) = expected_vmm(granularity, s, first + 1);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(state.mapped_context_tokens(), tokens);
        assert!(tokens > first);
        let keys = &state.qsa[0].full_keys;
        assert!(keys.buf.size() > old_size);
        let mut row = vec![0u8; 2048];
        gpu.hip
            .memcpy_dtoh(&mut row, &keys.buf)
            .expect("read row 0");
        assert_eq!(row, pattern, "growth preserves mapped rows");
        let fresh = keys.sub_offset(old_size / 4, (keys.buf.size() - old_size) / 4);
        let mut grown = vec![0xFFu8; fresh.buf.size()];
        gpu.hip
            .memcpy_dtoh(&mut grown, &fresh.buf)
            .expect("read new page");
        assert!(grown.iter().all(|&b| b == 0), "new pages are zeroed");

        state
            .ensure_mapped_capacity(&mut gpu, s)
            .expect("whole context");
        let (bytes, tokens) = expected_vmm(granularity, s, s);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(tokens, s);
        assert_eq!(state.mapped_context_tokens(), s);
        state.reset(&mut gpu).expect("reset mapped prefix");
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(state.position, 0);
        state.free_gpu(&mut gpu).expect("free state");
        assert_eq!(gpu.vmm_allocation_count(), owners);
    }

    /// After mapping the decode map-ahead target once, per-token ensure calls
    /// for the budgeted decode steps plus the draft rows never grow the
    /// mapping and never move the position.
    #[test]
    fn vmm_map_ahead_target_makes_decode_and_draft_ensures_no_ops() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        if gpu.vmm_recommended_granularity().is_err() {
            eprintln!("skip: HIP VMM unavailable");
            return;
        }
        let owners = gpu.vmm_allocation_count();
        let config = crate::config::compact_test_config();
        let s = 16_384;
        let mut state = Qwen4State::new_with_backend(
            &mut gpu,
            &config,
            s,
            Qwen4StateFormat::F32,
            Qwen4KvBackend::Vmm,
        )
        .expect("VMM state");
        let (position, budget, draft) = (100usize, 64usize, 3usize);
        state.position = position;
        let target = decode_map_ahead_tokens(position, Some(budget), draft, s)
            .expect("budgeted request maps ahead");
        assert_eq!(target, position + budget + draft);
        assert!(decode_map_ahead_tokens(position, Some(0), draft, s).is_none());
        assert!(decode_map_ahead_tokens(position, None, draft, s).is_none());

        state
            .ensure_mapped_capacity(&mut gpu, target)
            .expect("map ahead");
        let bytes = state.mapped_context_bytes(&gpu).unwrap();
        let tokens = state.mapped_context_tokens();
        assert!(bytes > 0);
        assert!(tokens >= target && tokens < s);

        // Each decode step needs its row plus the draft rows beyond it.
        for step in 1..=budget {
            state
                .ensure_mapped_capacity(&mut gpu, position + step + draft)
                .expect("budgeted decode + draft rows are already mapped");
            assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
            assert_eq!(state.mapped_context_tokens(), tokens);
            assert_eq!(state.position, position);
        }
        state
            .ensure_mapped_capacity(&mut gpu, target)
            .expect("target stays mapped");
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(state.mapped_context_tokens(), tokens);
        assert_eq!(state.position, position);

        state.free_gpu(&mut gpu).expect("free state");
        assert_eq!(gpu.vmm_allocation_count(), owners);
    }

    /// Every QSA state format the device serves (`bf16` / `auto` / `q8` /
    /// `fp8` requests that resolve here: the gfx1201 default is fp8) is
    /// demand-mapped like the F32 case above: nothing committed at
    /// construction, growth that covers each request, keeps mapped rows and
    /// zeroes only new pages, and a whole-context map that commits the legacy
    /// state's bytes up to whole driver pages per arena.
    #[test]
    fn vmm_context_maps_every_served_qsa_format() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        if !crate::kv_backend::qwen4_vmm_supported(&gpu) {
            eprintln!("skip: Qwen4 VMM unsupported on {}", gpu.arch);
            return;
        }
        let config = crate::config::compact_test_config();
        let mut formats = Vec::new();
        for request in ["bf16", "auto", "q8", "fp8"] {
            if let Ok(format) = resolve_qsa_format(request, &gpu, &config) {
                if !formats.contains(&format) {
                    formats.push(format);
                }
            }
        }
        if gpu.arch_caps.is_gfx1201() {
            assert!(
                formats.contains(&QsaKvFormat::Fp8),
                "gfx1201 serves fp8 QSA state"
            );
        }
        let s = 16_384;
        for qsa in formats {
            let format = Qwen4StateFormat {
                qsa,
                gdn: GdnStateFormat::F32,
                ..Qwen4StateFormat::F32
            };
            let owners = gpu.vmm_allocation_count();
            let legacy =
                Qwen4State::new_with_backend(&mut gpu, &config, s, format, Qwen4KvBackend::Legacy)
                    .expect("legacy state");
            let full = legacy.mapped_context_bytes(&gpu).unwrap();
            legacy.free_gpu(&mut gpu).expect("free legacy state");
            let mut state =
                Qwen4State::new_with_backend(&mut gpu, &config, s, format, Qwen4KvBackend::Vmm)
                    .expect("VMM state");
            let arenas: usize = state.qsa.iter().map(|q| q.context_arenas().count()).sum();
            assert_eq!(
                gpu.vmm_allocation_count(),
                owners + arenas,
                "{}",
                qsa.name()
            );
            assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), 0);
            let granularity = gpu.vmm_granularity(&state.qsa[0].full_keys).unwrap();
            eprintln!("{} VMM QSA state: granularity {granularity} B", qsa.name());

            state
                .ensure_mapped_capacity(&mut gpu, 1)
                .expect("first page");
            let first = state.mapped_context_tokens();
            assert!(
                first >= 1 && first < s,
                "{}: first coverage {first}",
                qsa.name()
            );
            let keys = &state.qsa[0].full_keys;
            let row_bytes = keys.byte_size() / s;
            let pattern: Vec<u8> = (0..row_bytes).map(|i| (i % 251) as u8 | 1).collect();
            gpu.hip
                .memcpy_htod(&keys.buf, &pattern)
                .expect("write row 0");

            let old_size = state.qsa[0].full_keys.buf.size();
            state
                .ensure_mapped_capacity(&mut gpu, first + 1)
                .expect("grow");
            assert!(state.mapped_context_tokens() > first, "{}", qsa.name());
            let keys = &state.qsa[0].full_keys;
            assert!(keys.buf.size() > old_size);
            let mut row = vec![0u8; row_bytes];
            gpu.hip
                .memcpy_dtoh(&mut row, &keys.buf)
                .expect("read row 0");
            assert_eq!(row, pattern, "{}: growth preserves mapped rows", qsa.name());
            let element = keys.dtype.size();
            let fresh = keys.sub_offset(old_size / element, (keys.buf.size() - old_size) / element);
            let mut grown = vec![0xFFu8; fresh.buf.size()];
            gpu.hip
                .memcpy_dtoh(&mut grown, &fresh.buf)
                .expect("read new page");
            assert!(
                grown.iter().all(|&b| b == 0),
                "{}: new pages are zeroed",
                qsa.name()
            );

            state
                .ensure_mapped_capacity(&mut gpu, s)
                .expect("whole context");
            assert_eq!(state.mapped_context_tokens(), s);
            let committed = state.mapped_context_bytes(&gpu).unwrap();
            assert!(
                committed >= full && committed < full + arenas * granularity,
                "{}: whole-context commit {committed} vs legacy {full}",
                qsa.name()
            );
            state.free_gpu(&mut gpu).expect("free VMM state");
            assert_eq!(gpu.vmm_allocation_count(), owners);
        }
    }

    /// Hardware smoke at the canonical geometry (Halo): grow a VMM state
    /// across chunk boundaries to native context and compare with a legacy
    /// state. Reads the config from `$HIPFIRE_MODELS_DIR/qwen3.8-flash-next-gptq3.mq4`
    /// when present (compact test config otherwise) and censuses device memory
    /// via `hipMemGetInfo`.
    #[test]
    #[ignore = "hardware smoke: commits up to ~14 GiB of device memory"]
    fn flash_next_vmm_state_smoke() {
        let mut gpu = Gpu::init().expect("GPU");
        // production HIPFIRE_* reads must stay config-owned (check-env-docs).
        let models_dir = std::env::var("HIPFIRE_MODELS_DIR")
            .unwrap_or_else(|_| "/home/kaden/.hipfire/models".to_string());
        let artifact = std::path::Path::new(&models_dir).join("qwen3.8-flash-next-gptq3.mq4");
        let config = if artifact.exists() {
            let hfq = hipfire_runtime::hfq::HfqFile::open(&artifact).expect("open artifact");
            Qwen4Config::from_metadata_json(&hfq.metadata_json).expect("artifact config")
        } else {
            crate::config::compact_test_config()
        };
        let vram = |gpu: &Gpu| {
            gpu.hip
                .get_vram_info()
                .map(|(free, total)| total.saturating_sub(free))
                .unwrap_or(0)
        };
        let format = Qwen4StateFormat {
            qsa: QsaKvFormat::F32,
            gdn: GdnStateFormat::Q8,
            ..Qwen4StateFormat::F32
        };
        let s = config.max_position_embeddings;
        let layers = config.n_full_layers();
        eprintln!(
            "smoke arch={} S={s} full_layers={layers} vmm_supported={}",
            gpu.arch,
            crate::kv_backend::qwen4_vmm_supported(&gpu)
        );
        for backend in [Qwen4KvBackend::Vmm, Qwen4KvBackend::Legacy] {
            let before = vram(&gpu);
            let start = std::time::Instant::now();
            let mut state =
                Qwen4State::new_with_backend(&mut gpu, &config, s, format, backend).expect("state");
            let built = start.elapsed();
            eprintln!(
                "{backend:?} new: {:.1} ms vram_delta={} MiB context_committed={} MiB covered={}",
                built.as_secs_f64() * 1e3,
                (vram(&gpu).saturating_sub(before)) / MIB,
                state.mapped_context_bytes(&gpu).unwrap() / MIB,
                state.mapped_context_tokens()
            );
            if backend == Qwen4KvBackend::Legacy {
                state.free_gpu(&mut gpu).expect("free legacy");
                continue;
            }
            let pattern = vec![0x5Au8; 2048];
            for tokens in [
                1, 1024, 1025, 4096, 4097, 8192, 8193, 16_384, 16_385, 65_536, 131_072, s,
            ] {
                let covered = state.mapped_context_tokens();
                let size = state.qsa[0].full_keys.buf.size();
                let start = std::time::Instant::now();
                state
                    .ensure_mapped_capacity(&mut gpu, tokens)
                    .expect("grow");
                let took = start.elapsed();
                let keys = &state.qsa[0].full_keys;
                if size > 0 {
                    let mut row = vec![0u8; 2048];
                    gpu.hip.memcpy_dtoh(&mut row, &keys.buf).unwrap();
                    assert_eq!(row, pattern, "row 0 preserved across growth");
                }
                if keys.buf.size() > size {
                    let fresh = keys.sub_offset(size / 4, (keys.buf.size() - size) / 4);
                    let mut grown = vec![0xFFu8; fresh.buf.size()];
                    gpu.hip.memcpy_dtoh(&mut grown, &fresh.buf).unwrap();
                    assert!(grown.iter().all(|&b| b == 0), "new pages zeroed");
                }
                if size == 0 {
                    gpu.hip.memcpy_htod(&keys.buf, &pattern).unwrap();
                }
                let touched = config.qsa_context_arena_bytes(tokens, format.qsa).unwrap() * layers;
                eprintln!(
                    "ensure {tokens:>6}: {covered:>6}->{:>6} covered, committed={:>6} MiB touched_logical={:>9.2} MiB vram={} MiB {:.2} ms",
                    state.mapped_context_tokens(),
                    state.mapped_context_bytes(&gpu).unwrap() / MIB,
                    touched as f64 / MIB as f64,
                    vram(&gpu) / MIB,
                    took.as_secs_f64() * 1e3
                );
            }
            let committed = state.mapped_context_bytes(&gpu).unwrap();
            assert!(matches!(
                state.ensure_mapped_capacity(&mut gpu, s + 1),
                Err(StateError::ContextCapacity { .. })
            ));
            assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), committed);
            eprintln!(
                "refused {} tokens; committed unchanged {} MiB",
                s + 1,
                committed / MIB
            );
            state.free_gpu(&mut gpu).expect("free VMM");
            eprintln!("vmm owners after free: {}", gpu.vmm_allocation_count());
        }
    }
}

#[cfg(test)]
#[path = "state_precision_accounting_tests.rs"]
mod precision_accounting_tests;
