// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU/reference Qwen4 / Qwen3.8-Flash-Next MTP equations and fixtures.
//!
//! Production MTP execution and request state live in [`crate::mtp_gpu`].
//! This module holds deterministic CPU transaction state and equation-level
//! fixtures. The input embedding uses the tied trunk table, and
//! [`crate::weights::Qwen4Weights`] owns the distinct lm_head.
//!
//! [`ReferenceMtpWeights`] and [`mtp_forward_step`] are deliberately small,
//! deterministic equation-level helpers.  They are useful for layerwise
//! parity fixtures and define the ordering the HIP lowering must preserve:
//! separate zero-centred norms, shared `fc_hidden` over all four branches,
//! one full-attention QSA layer, normalized top-10 MoE, and the final HC mix.
//! No acceptance policy or generation loop lives here.

use crate::ops;
use std::fmt;

/// The CPU/reference Qwen4 MTP module has one full-attention layer.
pub const MTP_LAYER_COUNT: usize = 1;
/// The checkpoint's Hyper-Connection branch count.
pub const MTP_BRANCHES: usize = 4;
/// Reference selection capacity for the native QSA geometry.
pub const MTP_SELECTED_CAPACITY: usize = MTP_INDEX_BUDGET + MTP_COMPRESS_RATIO - 1;
pub const MTP_TOP_K: usize = 10;
/// The reference model uses the config's RMS epsilon.  The pinned checkpoint
/// resolves this to 1e-6; keeping it local avoids inventing a new config field.
pub const MTP_RMS_EPS: f32 = 1.0e-6;

/// Native MTP QSA geometry shared with the equation-level parity fixtures.
pub const MTP_Q_HEADS: usize = 24;
pub const MTP_KV_HEADS: usize = 2;
pub const MTP_HEAD_DIM: usize = 256;
pub const MTP_INDEX_HEADS: usize = 4;
pub const MTP_INDEX_DIM: usize = 128;
pub const MTP_INDEX_BUDGET: usize = 2048;
pub const MTP_COMPRESS_RATIO: usize = 4;
pub const MTP_ROTARY_DIM: usize = 64;
pub const MTP_ROPE_THETA: f32 = 10_000_000.0;

/// QSA geometry is explicit so test fixtures can use small dimensions while
/// the native constructor retains the checkpoint's 24/2/256 and 4/128 shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpQsaGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub compress_ratio: usize,
    pub budget: usize,
    pub max_seq_len: usize,
    pub rotary_dim: usize,
    pub rope_theta: u32,
}

impl MtpQsaGeometry {
    pub const fn native(max_seq_len: usize) -> Self {
        Self {
            q_heads: MTP_Q_HEADS,
            kv_heads: MTP_KV_HEADS,
            head_dim: MTP_HEAD_DIM,
            index_heads: MTP_INDEX_HEADS,
            index_dim: MTP_INDEX_DIM,
            compress_ratio: MTP_COMPRESS_RATIO,
            budget: MTP_INDEX_BUDGET,
            max_seq_len,
            rotary_dim: MTP_ROTARY_DIM,
            rope_theta: MTP_ROPE_THETA as u32,
        }
    }

    pub fn selected_capacity(self) -> usize {
        self.budget + self.compress_ratio.saturating_sub(1)
    }

    pub fn full_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Separate main-KV, compressed-index, and raw-circular state for the MTP QSA
/// layer.  Main K/V grows to the active sequence length; the raw circular ring
/// remains bounded and is never confused with the persistent main K/V.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpQsaState {
    pub geometry: MtpQsaGeometry,
    pub main_keys: Vec<f32>,
    pub main_values: Vec<f32>,
    pub compressed_keys: Vec<f32>,
    pub compressed_positions: Vec<usize>,
    pub incomplete_keys: Vec<f32>,
    pub incomplete_values: Vec<f32>,
    pub incomplete_positions: Vec<usize>,
    pub raw_circular: Vec<f32>,
    pub raw_positions: Vec<usize>,
    pub raw_cursor: usize,
    pub selected_indices: Vec<usize>,
    pub position: usize,
}

impl MtpQsaState {
    pub fn new(geometry: MtpQsaGeometry) -> Result<Self, MtpError> {
        if geometry.max_seq_len == 0
            || geometry.q_heads == 0
            || geometry.kv_heads == 0
            || geometry.head_dim == 0
            || geometry.index_heads == 0
            || geometry.index_dim == 0
            || geometry.compress_ratio == 0
            || geometry.q_heads % geometry.kv_heads != 0
        {
            return Err(MtpError::Shape("invalid MTP QSA geometry"));
        }
        Ok(Self {
            geometry,
            main_keys: Vec::with_capacity(geometry.max_seq_len * geometry.full_width()),
            main_values: Vec::with_capacity(geometry.max_seq_len * geometry.full_width()),
            compressed_keys: Vec::with_capacity(
                ((geometry.max_seq_len + geometry.compress_ratio - 1) / geometry.compress_ratio)
                    * geometry.index_dim,
            ),
            compressed_positions: Vec::with_capacity(
                (geometry.max_seq_len + geometry.compress_ratio - 1) / geometry.compress_ratio,
            ),
            incomplete_keys: Vec::with_capacity(geometry.compress_ratio * geometry.index_dim),
            incomplete_values: Vec::with_capacity(geometry.compress_ratio * geometry.full_width()),
            incomplete_positions: Vec::with_capacity(geometry.compress_ratio),
            // The incomplete block and raw ring are side state.  Keep two
            // blocks so step reuse survives a one-token target-aligned replay.
            raw_circular: Vec::with_capacity(geometry.compress_ratio * 2 * geometry.index_dim),
            raw_positions: Vec::with_capacity(geometry.compress_ratio * 2),
            raw_cursor: 0,
            selected_indices: Vec::with_capacity(geometry.selected_capacity()),
            position: 0,
        })
    }

    pub fn reset(&mut self) {
        self.main_keys.clear();
        self.main_values.clear();
        self.compressed_keys.clear();
        self.compressed_positions.clear();
        self.incomplete_keys.clear();
        self.incomplete_values.clear();
        self.incomplete_positions.clear();
        self.raw_circular.clear();
        self.raw_positions.clear();
        self.raw_cursor = 0;
        self.selected_indices.clear();
        self.position = 0;
    }

    pub fn main_token_count(&self) -> usize {
        self.main_keys.len() / self.geometry.full_width()
    }

    pub fn append_token(
        &mut self,
        position: usize,
        raw_index_key: &[f32],
        full_key: &[f32],
        full_value: &[f32],
    ) -> Result<(), MtpError> {
        if position != self.position {
            return Err(MtpError::Position {
                expected: self.position,
                actual: position,
            });
        }
        if self.position >= self.geometry.max_seq_len {
            return Err(MtpError::Capacity(self.geometry.max_seq_len));
        }
        if raw_index_key.len() != self.geometry.index_dim
            || full_key.len() != self.geometry.full_width()
            || full_value.len() != self.geometry.full_width()
        {
            return Err(MtpError::Shape("MTP QSA append row"));
        }
        self.main_keys.extend_from_slice(full_key);
        self.main_values.extend_from_slice(full_value);
        self.incomplete_keys.extend_from_slice(raw_index_key);
        self.incomplete_values.extend_from_slice(full_value);
        self.incomplete_positions.push(position);
        self.push_raw(raw_index_key, position);
        if self.incomplete_positions.len() == self.geometry.compress_ratio {
            let start = self.incomplete_positions[0];
            let pooled = ops::qsa_pool_block(
                &self.incomplete_keys,
                self.geometry.compress_ratio,
                start,
                self.geometry.index_dim,
                self.geometry.rope_theta as f32,
            )
            .map_err(|_| MtpError::Shape("MTP pooled index key"))?;
            self.compressed_keys.extend_from_slice(&pooled);
            self.compressed_positions.push(start);
            self.incomplete_keys.clear();
            self.incomplete_values.clear();
            self.incomplete_positions.clear();
        }
        self.position += 1;
        Ok(())
    }

    fn push_raw(&mut self, row: &[f32], position: usize) {
        let capacity = self.geometry.compress_ratio * 2;
        if self.raw_positions.len() < capacity {
            self.raw_circular.extend_from_slice(row);
            self.raw_positions.push(position);
            return;
        }
        let offset = self.raw_cursor * self.geometry.index_dim;
        self.raw_circular[offset..offset + self.geometry.index_dim].copy_from_slice(row);
        self.raw_positions[self.raw_cursor] = position;
        self.raw_cursor = (self.raw_cursor + 1) % capacity;
    }

    /// Select complete blocks at step 0 and append every incomplete-tail token.
    /// The returned indices are token positions, not pooled block positions.
    pub fn select_step0(&mut self, query_heads: &[f32]) -> Result<Vec<usize>, MtpError> {
        if query_heads.len() != self.geometry.index_heads * self.geometry.index_dim {
            return Err(MtpError::Shape("MTP indexer query heads"));
        }
        let block_starts = ops::qsa_select_blocks(
            query_heads,
            &self.compressed_keys,
            &self.compressed_positions,
            self.geometry.budget / self.geometry.compress_ratio,
            self.geometry.index_dim,
        )
        .map_err(|_| MtpError::Shape("MTP QSA block selection"))?;
        let mut selected = Vec::with_capacity(self.geometry.selected_capacity());
        for start in block_starts {
            for offset in 0..self.geometry.compress_ratio {
                let token = start + offset;
                if token < self.position {
                    selected.push(token);
                }
            }
        }
        selected.extend(self.incomplete_positions.iter().copied());
        if selected.len() > self.geometry.selected_capacity() {
            return Err(MtpError::SelectionCapacity {
                capacity: self.geometry.selected_capacity(),
                actual: selected.len(),
            });
        }
        self.selected_indices = selected;
        Ok(self.selected_indices.clone())
    }

    /// Reuse the previous step's sparse index row.  The current token is
    /// target-visible and is appended when capacity permits; no re-selection
    /// is performed on later MTP steps.
    pub fn reuse_selection(&mut self, position: usize) -> Result<Vec<usize>, MtpError> {
        if self.selected_indices.is_empty() {
            return Err(MtpError::SelectionUnavailable);
        }
        if position >= self.geometry.max_seq_len {
            return Err(MtpError::Capacity(self.geometry.max_seq_len));
        }
        self.selected_indices.retain(|&index| index <= position);
        if !self.selected_indices.contains(&position)
            && self.selected_indices.len() < self.geometry.selected_capacity()
        {
            self.selected_indices.push(position);
        }
        Ok(self.selected_indices.clone())
    }

    /// Compact a target-aligned selection row after partial acceptance.  This
    /// is pure index compaction: rejected rows never remain visible to the next
    /// MTP step and no seed re-echo is inserted.
    pub fn compact_selection(&mut self, retained_rows: &[usize]) -> Result<(), MtpError> {
        if retained_rows.len() > self.geometry.selected_capacity()
            || retained_rows.iter().any(|&row| row >= self.position)
        {
            return Err(MtpError::SelectionCapacity {
                capacity: self.geometry.selected_capacity(),
                actual: retained_rows.len(),
            });
        }
        self.selected_indices.clear();
        self.selected_indices.extend_from_slice(retained_rows);
        Ok(())
    }

    /// Truncate active append state to a target-committed prefix.  Full KV
    /// capacity is never cloned; only active lengths are shortened.
    pub fn truncate_to(&mut self, position: usize) -> Result<(), MtpError> {
        if position > self.main_token_count() {
            return Err(MtpError::Position {
                expected: self.main_token_count(),
                actual: position,
            });
        }
        let full = position * self.geometry.full_width();
        self.main_keys.truncate(full);
        self.main_values.truncate(full);
        self.compressed_keys
            .truncate((position / self.geometry.compress_ratio) * self.geometry.index_dim);
        self.compressed_positions.retain(|&start| start < position);
        self.incomplete_keys.clear();
        self.incomplete_values.clear();
        self.incomplete_positions.clear();
        self.selected_indices.retain(|&index| index < position);
        self.raw_positions.retain(|&row| row < position);
        self.position = position;
        Ok(())
    }

    /// Return a bounded snapshot.  Main K/V are represented by lengths;
    /// compressed/selection marks and raw-circular side state are copied.
    /// The incomplete block is copied in full because appending its final row
    /// clears it into the compressed index; lengths alone cannot restore that
    /// transition after a rejected speculative window.
    pub fn snapshot(&self) -> MtpQsaSnapshot {
        MtpQsaSnapshot {
            main_key_len: self.main_keys.len(),
            main_value_len: self.main_values.len(),
            compressed_key_len: self.compressed_keys.len(),
            compressed_position_len: self.compressed_positions.len(),
            incomplete_keys: self.incomplete_keys.clone(),
            incomplete_values: self.incomplete_values.clone(),
            incomplete_positions: self.incomplete_positions.clone(),
            selected_indices: self.selected_indices.clone(),
            raw_circular: self.raw_circular.clone(),
            raw_positions: self.raw_positions.clone(),
            raw_cursor: self.raw_cursor,
            position: self.position,
        }
    }

    pub fn restore(&mut self, snapshot: &MtpQsaSnapshot) -> Result<(), MtpError> {
        for (len, actual, what) in [
            (snapshot.main_key_len, self.main_keys.len(), "MTP main keys"),
            (
                snapshot.main_value_len,
                self.main_values.len(),
                "MTP main values",
            ),
            (
                snapshot.compressed_key_len,
                self.compressed_keys.len(),
                "MTP compressed keys",
            ),
            (
                snapshot.compressed_position_len,
                self.compressed_positions.len(),
                "MTP compressed positions",
            ),
        ] {
            if len > actual {
                return Err(MtpError::SnapshotLength {
                    what,
                    requested: len,
                    actual,
                });
            }
        }

        let index_dim = self.geometry.index_dim;
        let full_width = self.geometry.full_width();
        let partial_key_len = snapshot
            .incomplete_positions
            .len()
            .checked_mul(index_dim)
            .ok_or(MtpError::Shape("MTP partial snapshot dimensions"))?;
        let partial_value_len = snapshot
            .incomplete_positions
            .len()
            .checked_mul(full_width)
            .ok_or(MtpError::Shape("MTP partial snapshot dimensions"))?;
        if snapshot.incomplete_keys.len() != partial_key_len
            || snapshot.incomplete_values.len() != partial_value_len
            || snapshot.incomplete_positions.len() >= self.geometry.compress_ratio
            || snapshot
                .incomplete_positions
                .iter()
                .any(|&row| row >= snapshot.position)
        {
            return Err(MtpError::Shape("MTP partial snapshot dimensions"));
        }
        let expected_position = snapshot
            .main_key_len
            .checked_div(full_width)
            .ok_or(MtpError::Shape("MTP main snapshot dimensions"))?;
        if snapshot.main_key_len != expected_position * full_width
            || snapshot.main_value_len != expected_position * full_width
            || snapshot.position != expected_position
            || snapshot.selected_indices.len() > self.geometry.selected_capacity()
            || snapshot
                .selected_indices
                .iter()
                .any(|&row| row >= snapshot.position)
        {
            return Err(MtpError::Shape("MTP main snapshot dimensions"));
        }
        let expected_compressed = snapshot
            .compressed_position_len
            .checked_mul(index_dim)
            .ok_or(MtpError::Shape("MTP compressed snapshot dimensions"))?;
        if snapshot.compressed_key_len != expected_compressed
            || self.compressed_positions[..snapshot.compressed_position_len]
                .iter()
                .any(|&row| row >= snapshot.position)
        {
            return Err(MtpError::Shape("MTP compressed snapshot dimensions"));
        }
        let raw_capacity = self.geometry.compress_ratio * 2;
        let expected_raw = snapshot
            .raw_positions
            .len()
            .checked_mul(index_dim)
            .ok_or(MtpError::Shape("MTP raw snapshot dimensions"))?;
        if snapshot.raw_circular.len() != expected_raw
            || snapshot.raw_positions.len() > raw_capacity
            || (snapshot.raw_positions.is_empty() && snapshot.raw_cursor != 0)
            || (!snapshot.raw_positions.is_empty() && snapshot.raw_cursor >= raw_capacity)
            || snapshot
                .raw_positions
                .iter()
                .any(|&row| row >= snapshot.position)
        {
            return Err(MtpError::Shape("MTP raw snapshot dimensions"));
        }

        self.main_keys.truncate(snapshot.main_key_len);
        self.main_values.truncate(snapshot.main_value_len);
        self.compressed_keys.truncate(snapshot.compressed_key_len);
        self.compressed_positions
            .truncate(snapshot.compressed_position_len);
        self.incomplete_keys.clone_from(&snapshot.incomplete_keys);
        self.incomplete_values
            .clone_from(&snapshot.incomplete_values);
        self.incomplete_positions
            .clone_from(&snapshot.incomplete_positions);
        self.selected_indices.clone_from(&snapshot.selected_indices);
        self.raw_circular.clone_from(&snapshot.raw_circular);
        self.raw_positions.clone_from(&snapshot.raw_positions);
        self.raw_cursor = snapshot.raw_cursor;
        self.position = snapshot.position;
        Ok(())
    }
}

/// Snapshot marks for one MTP QSA layer.  The persistent main KV is not cloned.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpQsaSnapshot {
    pub main_key_len: usize,
    pub main_value_len: usize,
    pub compressed_key_len: usize,
    pub compressed_position_len: usize,
    /// The partial block is bounded by `compress_ratio` and must be copied:
    /// appending its final row can clear it before a transaction is restored.
    pub incomplete_keys: Vec<f32>,
    pub incomplete_values: Vec<f32>,
    pub incomplete_positions: Vec<usize>,
    pub selected_indices: Vec<usize>,
    pub raw_circular: Vec<f32>,
    pub raw_positions: Vec<usize>,
    pub raw_cursor: usize,
    pub position: usize,
}

/// CPU/reference MTP request state. QSA and wide-hidden snapshots let parity
/// fixtures exercise partial-acceptance rollback without GPU allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4MtpState {
    pub qsa: MtpQsaState,
    pub wide_hidden: Vec<f32>,
    pub position: usize,
    pub step_index: usize,
    pub request_epoch: u64,
    pub cancelled: bool,
    pub(crate) in_flight: bool,
    pub(crate) quiesced: bool,
}

impl Qwen4MtpState {
    pub fn new(geometry: MtpQsaGeometry) -> Result<Self, MtpError> {
        Ok(Self {
            qsa: MtpQsaState::new(geometry)?,
            wide_hidden: Vec::new(),
            position: 0,
            step_index: 0,
            request_epoch: 0,
            cancelled: false,
            in_flight: false,
            quiesced: true,
        })
    }

    pub fn native(max_seq_len: usize) -> Result<Self, MtpError> {
        Self::new(MtpQsaGeometry::native(max_seq_len))
    }

    pub fn reset(&mut self) {
        self.qsa.reset();
        self.wide_hidden.clear();
        self.position = 0;
        self.step_index = 0;
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.cancelled = false;
        self.in_flight = false;
        self.quiesced = true;
    }

    pub fn begin_transaction(&mut self) -> Qwen4MtpSnapshot {
        self.in_flight = true;
        self.quiesced = false;
        Qwen4MtpSnapshot {
            qsa: self.qsa.snapshot(),
            wide_hidden: self.wide_hidden.clone(),
            position: self.position,
            step_index: self.step_index,
            request_epoch: self.request_epoch,
        }
    }

    pub fn restore(&mut self, snapshot: Qwen4MtpSnapshot) -> Result<(), MtpError> {
        self.qsa.restore(&snapshot.qsa)?;
        self.wide_hidden = snapshot.wide_hidden;
        self.position = snapshot.position;
        self.step_index = snapshot.step_index;
        self.request_epoch = snapshot.request_epoch;
        self.in_flight = false;
        self.quiesced = true;
        Ok(())
    }

    pub fn commit(&mut self, snapshot: Qwen4MtpSnapshot) -> Result<(), MtpError> {
        if snapshot.request_epoch != self.request_epoch {
            return Err(MtpError::SnapshotEpoch {
                expected: self.request_epoch,
                actual: snapshot.request_epoch,
            });
        }
        self.in_flight = false;
        self.quiesced = true;
        Ok(())
    }

    /// Cancellation is transactional: restore the supplied pre-window state,
    /// then mark the request cancelled.  No rejected HC feedback survives.
    pub fn cancel(&mut self, snapshot: Option<Qwen4MtpSnapshot>) -> Result<(), MtpError> {
        if let Some(snapshot) = snapshot {
            self.restore(snapshot)?;
        }
        self.cancelled = true;
        self.in_flight = false;
        self.quiesced = true;
        Ok(())
    }

    /// Forced continuation owns only the MTP-side position/cursor.  The caller
    /// supplies the already-computed rows; generation scheduling remains in the
    /// runtime engine.
    pub fn forced_advance(&mut self, token_count: usize) -> Result<(), MtpError> {
        if self.cancelled {
            return Err(MtpError::Cancelled);
        }
        self.position = self
            .position
            .checked_add(token_count)
            .ok_or(MtpError::PositionOverflow)?;
        self.step_index = self.step_index.wrapping_add(token_count);
        self.qsa.position = self.position;
        Ok(())
    }

    /// Quiesce before teardown.  A live transaction is refused rather than
    /// freeing resources whose retained path may still reference them.
    pub fn quiesce(&mut self) -> Result<(), MtpError> {
        if self.in_flight {
            return Err(MtpError::InFlight);
        }
        self.quiesced = true;
        Ok(())
    }

    pub fn is_quiesced(&self) -> bool {
        self.quiesced && !self.in_flight
    }
}

/// Bounded CPU/reference MTP snapshot. It clones the wide hidden row and
/// indexer side state; main KV remains append-marked in `qsa`.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4MtpSnapshot {
    pub qsa: MtpQsaSnapshot,
    pub wide_hidden: Vec<f32>,
    pub position: usize,
    pub step_index: usize,
    pub request_epoch: u64,
}

/// Error type for CPU/reference MTP state and equations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MtpError {
    Length {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    Shape(&'static str),
    Position {
        expected: usize,
        actual: usize,
    },
    PositionOverflow,
    Capacity(usize),
    SelectionCapacity {
        capacity: usize,
        actual: usize,
    },
    SelectionUnavailable,
    SnapshotLength {
        what: &'static str,
        requested: usize,
        actual: usize,
    },
    SnapshotEpoch {
        expected: u64,
        actual: u64,
    },
    Cancelled,
    InFlight,
}

impl fmt::Display for MtpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length {
                what,
                expected,
                actual,
            } => {
                write!(f, "{what}: expected {expected}, got {actual}")
            }
            Self::Shape(what) => write!(f, "invalid {what}"),
            Self::Position { expected, actual } => {
                write!(f, "MTP position expected {expected}, got {actual}")
            }
            Self::PositionOverflow => write!(f, "MTP position overflow"),
            Self::Capacity(capacity) => write!(f, "MTP capacity {capacity} exceeded"),
            Self::SelectionCapacity { capacity, actual } => {
                write!(f, "MTP selection has {actual} rows, capacity {capacity}")
            }
            Self::SelectionUnavailable => write!(f, "MTP QSA selection is not primed"),
            Self::SnapshotLength {
                what,
                requested,
                actual,
            } => {
                write!(
                    f,
                    "{what}: snapshot mark {requested} exceeds active {actual}"
                )
            }
            Self::SnapshotEpoch { expected, actual } => {
                write!(f, "MTP snapshot epoch expected {expected}, got {actual}")
            }
            Self::Cancelled => write!(f, "MTP request is cancelled"),
            Self::InFlight => write!(f, "MTP transaction is still in flight"),
        }
    }
}

impl std::error::Error for MtpError {}

/// Matrix used by the equation-level MTP implementation.  Production GPU
/// lowering resolves the same logical shape from `TensorRef`/`WeightHandle`;
/// this owned form is only for deterministic CPU fixtures.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl MtpMatrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, MtpError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(MtpError::Shape("matrix dimensions overflow"))?;
        if rows == 0 || cols == 0 || values.len() != expected {
            return Err(MtpError::Length {
                what: "MTP matrix",
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { rows, cols, values })
    }

    pub fn mul_vec(&self, input: &[f32]) -> Result<Vec<f32>, MtpError> {
        if input.len() != self.cols {
            return Err(MtpError::Length {
                what: "matrix input",
                expected: self.cols,
                actual: input.len(),
            });
        }
        Ok(self
            .values
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(input).map(|(&w, &x)| w * x).sum())
            .collect())
    }
}

/// Attention projection weights for one CPU/reference MTP QSA block.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpAttentionWeights {
    pub q_proj: MtpMatrix,
    pub k_proj: MtpMatrix,
    pub v_proj: MtpMatrix,
    pub o_proj: MtpMatrix,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub index_qk_proj: MtpMatrix,
    pub index_q_norm: Vec<f32>,
    pub index_k_norm: Vec<f32>,
}

/// Hyper-Connection projection set.  `block_inject` has shape
/// `[branches, branches * hidden]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpHyperWeights {
    pub norm: Vec<f32>,
    pub down: MtpMatrix,
    pub up: MtpMatrix,
    pub block_inject: MtpMatrix,
}

/// Final learned branch mixer.  There is no final write projection or extra
/// RMSNorm after the mix.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpFinalHyperWeights {
    pub norm: Vec<f32>,
    pub down: MtpMatrix,
    pub up: MtpMatrix,
}

/// Routed and shared MTP expert weights.  Routed outputs are weighted exactly
/// once in `mtp_forward_step`; shared output is gated once and then added.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpMoeWeights {
    pub router: MtpMatrix,
    pub routed_gate_up: Vec<MtpMatrix>,
    pub routed_down: Vec<MtpMatrix>,
    pub shared_gate_scalar: Vec<f32>,
    pub shared_gate: MtpMatrix,
    pub shared_up: MtpMatrix,
    pub shared_down: MtpMatrix,
}

/// CPU-only MTP tensor fixture.  Production tensor references remain in
/// [`crate::weights::Qwen4MtpWeights`]; this owned form is for deterministic
/// equation probes and never aliases or allocates a GPU vocabulary table.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceMtpWeights {
    pub pre_fc_norm_embedding: Vec<f32>,
    pub pre_fc_norm_hidden: Vec<f32>,
    pub fc_embedding: MtpMatrix,
    pub fc_hidden: MtpMatrix,
    pub attention: MtpAttentionWeights,
    pub attention_hyper: MtpHyperWeights,
    pub moe: MtpMoeWeights,
    pub mlp_hyper: MtpHyperWeights,
    pub final_hyper: MtpFinalHyperWeights,
    pub qsa_geometry: MtpQsaGeometry,
}

impl ReferenceMtpWeights {
    /// Validate the fixed CPU/reference dimensions and all expert tensor counts.
    pub fn validate(&self) -> Result<(), MtpError> {
        let hidden = self.fc_embedding.cols;
        if self.fc_embedding.rows != hidden
            || self.pre_fc_norm_embedding.len() != hidden
            || self.pre_fc_norm_hidden.len() % MTP_BRANCHES != 0
            || self.fc_hidden.rows != hidden
            || self.fc_hidden.cols != hidden
        {
            return Err(MtpError::Shape("MTP fc/norm dimensions"));
        }
        let branches = self.pre_fc_norm_hidden.len() / hidden;
        if branches == 0 || self.pre_fc_norm_hidden.len() != branches * hidden {
            return Err(MtpError::Shape("MTP hidden branch dimensions"));
        }
        if self.attention_hyper.norm.len() != branches * hidden
            || self.mlp_hyper.norm.len() != branches * hidden
            || self.final_hyper.norm.len() != branches * hidden
        {
            return Err(MtpError::Shape("MTP HC norm dimensions"));
        }
        let wide = branches * hidden;
        if self.final_hyper.down.rows == 0
            || self.final_hyper.down.cols != wide
            || self.final_hyper.up.rows != wide
            || self.final_hyper.up.cols != self.final_hyper.down.rows
        {
            return Err(MtpError::Shape("MTP final HC mix dimensions"));
        }
        if self.attention.o_proj.cols != self.qsa_geometry.q_heads * self.qsa_geometry.head_dim
            || self.attention.o_proj.rows != hidden
            || self.attention.q_proj.cols != hidden
            || self.attention.q_proj.rows
                != self.qsa_geometry.q_heads * self.qsa_geometry.head_dim * 2
            || self.attention.k_proj.cols != hidden
            || self.attention.k_proj.rows != self.qsa_geometry.kv_heads * self.qsa_geometry.head_dim
            || self.attention.v_proj.cols != hidden
            || self.attention.v_proj.rows != self.qsa_geometry.kv_heads * self.qsa_geometry.head_dim
            || self.attention.q_norm.len() != self.qsa_geometry.head_dim
            || self.attention.k_norm.len() != self.qsa_geometry.head_dim
        {
            return Err(MtpError::Shape("MTP QSA projection dimensions"));
        }
        if self.attention.index_qk_proj.cols != hidden
            || self.attention.index_qk_proj.rows
                != (self.qsa_geometry.index_heads + 1) * self.qsa_geometry.index_dim
            || self.attention.index_q_norm.len() != self.qsa_geometry.index_dim
            || self.attention.index_k_norm.len() != self.qsa_geometry.index_dim
        {
            return Err(MtpError::Shape("MTP indexer dimensions"));
        }
        if self.moe.router.cols != hidden
            || self.moe.routed_gate_up.len() != self.moe.router.rows
            || self.moe.routed_down.len() != self.moe.router.rows
            || self.moe.router.rows < MTP_TOP_K
            || self.moe.shared_gate_scalar.len() != hidden
            || self.moe.shared_gate.cols != hidden
            || self.moe.shared_up.cols != hidden
            || self.moe.shared_down.rows != hidden
        {
            return Err(MtpError::Shape("MTP MoE dimensions"));
        }
        for (gate_up, down) in self.moe.routed_gate_up.iter().zip(&self.moe.routed_down) {
            if gate_up.cols != hidden
                || gate_up.rows % 2 != 0
                || down.rows != hidden
                || down.cols != gate_up.rows / 2
            {
                return Err(MtpError::Shape("MTP routed expert dimensions"));
            }
        }
        if self.qsa_geometry.q_heads % self.qsa_geometry.kv_heads != 0
            || self.qsa_geometry.compress_ratio == 0
            || self.qsa_geometry.index_heads == 0
            || self.qsa_geometry.index_dim == 0
        {
            return Err(MtpError::Shape("MTP QSA geometry"));
        }
        Ok(())
    }
}

/// Complete MTP forward input.  The token embedding row is looked up through
/// the aliased trunk embedding by the carrier; no token table is owned here.
pub struct MtpForwardInput<'a> {
    pub token_embedding: &'a [f32],
    pub backbone_hidden: &'a [f32],
    pub position: usize,
    pub step_index: usize,
}

/// One CPU/reference MTP output: collapsed H for the distinct lm_head and wide
/// 4H for the following draft step.  `qsa_indices` records the step-0 selection
/// or later-step reuse for parity diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpForwardOutput {
    pub collapsed_hidden: Vec<f32>,
    pub wide_hidden: Vec<f32>,
    pub qsa_indices: Vec<usize>,
    pub selected_experts: [usize; MTP_TOP_K],
    pub routing_weights: [f32; MTP_TOP_K],
}

/// Execute one full-attention QSA/MoE/HC MTP step on CPU reference tensors.
/// The same operation order is used by the future typed HIP lowering.
pub fn mtp_forward_step(
    weights: &ReferenceMtpWeights,
    state: &mut Qwen4MtpState,
    input: MtpForwardInput<'_>,
) -> Result<MtpForwardOutput, MtpError> {
    weights.validate()?;
    if state.cancelled {
        return Err(MtpError::Cancelled);
    }
    if input.position != state.position {
        return Err(MtpError::Position {
            expected: state.position,
            actual: input.position,
        });
    }
    let hidden = weights.fc_embedding.cols;
    let branches = weights.pre_fc_norm_hidden.len() / hidden;
    if input.token_embedding.len() != hidden || input.backbone_hidden.len() != branches * hidden {
        return Err(MtpError::Shape("MTP input embedding/backbone hidden"));
    }
    if state.qsa.geometry.max_seq_len <= input.position {
        return Err(MtpError::Capacity(state.qsa.geometry.max_seq_len));
    }

    // 1. Normalize embedding H and wide hidden 4H separately, then apply the
    // two residual-linear projections.  The embedding projection is shared by
    // all four branches; fc_hidden is one shared matrix applied per branch.
    let mut embedding_norm = vec![0.0f32; hidden];
    ops::zero_centered_rms_norm(
        input.token_embedding,
        &weights.pre_fc_norm_embedding,
        MTP_RMS_EPS,
        &mut embedding_norm,
    )
    .map_err(|_| MtpError::Shape("MTP embedding norm"))?;
    let projected_embedding = weights.fc_embedding.mul_vec(&embedding_norm)?;
    let mut hidden_norm = vec![0.0f32; branches * hidden];
    ops::zero_centered_rms_norm(
        input.backbone_hidden,
        &weights.pre_fc_norm_hidden,
        MTP_RMS_EPS,
        &mut hidden_norm,
    )
    .map_err(|_| MtpError::Shape("MTP wide hidden norm"))?;
    let mut wide = vec![0.0f32; branches * hidden];
    for branch in 0..branches {
        let projected = weights
            .fc_hidden
            .mul_vec(&hidden_norm[branch * hidden..(branch + 1) * hidden])?;
        for j in 0..hidden {
            wide[branch * hidden + j] = projected_embedding[j] + projected[j];
        }
    }

    // 2. Read the HC branches for attention.  This is the vLLM native order:
    // per-branch norm → low-rank SiLU → sigmoid mix → branch mean.
    let attn_input = hc_read(
        &weights.attention_hyper.norm,
        &weights.attention_hyper.down,
        &weights.attention_hyper.up,
        &wide,
        branches,
        hidden,
    )?;

    // 3. Full-attention QSA.  q_proj contains Q followed by its gate half;
    // k/v use two KV heads.  The indexer has four query heads plus one key.
    let q_and_gate = weights.attention.q_proj.mul_vec(&attn_input)?;
    let key = weights.attention.k_proj.mul_vec(&attn_input)?;
    let value = weights.attention.v_proj.mul_vec(&attn_input)?;
    let mut q_rot = vec![0.0f32; weights.attention.q_norm.len() * weights.qsa_geometry.q_heads];
    let q_width = weights.qsa_geometry.q_heads * weights.qsa_geometry.head_dim;
    if q_and_gate.len() != 2 * q_width {
        return Err(MtpError::Shape("MTP Q/gate projection"));
    }
    for head in 0..weights.qsa_geometry.q_heads {
        let q = &q_and_gate
            [head * weights.qsa_geometry.head_dim..(head + 1) * weights.qsa_geometry.head_dim];
        let mut norm = vec![0.0f32; weights.qsa_geometry.head_dim];
        ops::rms_norm(q, &weights.attention.q_norm, MTP_RMS_EPS, &mut norm)
            .map_err(|_| MtpError::Shape("MTP Q norm"))?;
        ops::rope_prefix_halfsplit(
            &mut norm,
            input.position,
            weights.qsa_geometry.rotary_dim,
            weights.qsa_geometry.rope_theta as f32,
        )
        .map_err(|_| MtpError::Shape("MTP Q RoPE"))?;
        q_rot[head * weights.qsa_geometry.head_dim..(head + 1) * weights.qsa_geometry.head_dim]
            .copy_from_slice(&norm);
    }
    let mut key_rows = key;
    for head in 0..weights.qsa_geometry.kv_heads {
        let row = &mut key_rows
            [head * weights.qsa_geometry.head_dim..(head + 1) * weights.qsa_geometry.head_dim];
        let mut norm = vec![0.0f32; weights.qsa_geometry.head_dim];
        ops::rms_norm(row, &weights.attention.k_norm, MTP_RMS_EPS, &mut norm)
            .map_err(|_| MtpError::Shape("MTP K norm"))?;
        ops::rope_prefix_halfsplit(
            &mut norm,
            input.position,
            weights.qsa_geometry.rotary_dim,
            weights.qsa_geometry.rope_theta as f32,
        )
        .map_err(|_| MtpError::Shape("MTP K RoPE"))?;
        row.copy_from_slice(&norm);
    }
    let index = weights.attention.index_qk_proj.mul_vec(&attn_input)?;
    let index_query_len = weights.qsa_geometry.index_heads * weights.qsa_geometry.index_dim;
    let mut index_query = vec![0.0f32; index_query_len];
    for head in 0..weights.qsa_geometry.index_heads {
        let src = &index
            [head * weights.qsa_geometry.index_dim..(head + 1) * weights.qsa_geometry.index_dim];
        let mut dst = vec![0.0f32; weights.qsa_geometry.index_dim];
        ops::rms_norm(src, &weights.attention.index_q_norm, MTP_RMS_EPS, &mut dst)
            .map_err(|_| MtpError::Shape("MTP index query norm"))?;
        index_query
            [head * weights.qsa_geometry.index_dim..(head + 1) * weights.qsa_geometry.index_dim]
            .copy_from_slice(&dst);
    }
    let raw_key = &index[index_query_len..index_query_len + weights.qsa_geometry.index_dim];
    let mut raw_key_norm = vec![0.0f32; weights.qsa_geometry.index_dim];
    ops::rms_norm(
        raw_key,
        &weights.attention.index_k_norm,
        MTP_RMS_EPS,
        &mut raw_key_norm,
    )
    .map_err(|_| MtpError::Shape("MTP index key norm"))?;
    state
        .qsa
        .append_token(input.position, &raw_key_norm, &key_rows, &value)?;
    let selected = if input.step_index == 0 || state.qsa.selected_indices.is_empty() {
        state.qsa.select_step0(&index_query)?
    } else {
        state.qsa.reuse_selection(input.position)?
    };
    let mut q_with_gate = vec![0.0f32; 2 * q_width];
    for head in 0..weights.qsa_geometry.q_heads {
        let q_off = head * weights.qsa_geometry.head_dim;
        let out = head * 2 * weights.qsa_geometry.head_dim;
        q_with_gate[out..out + weights.qsa_geometry.head_dim]
            .copy_from_slice(&q_rot[q_off..q_off + weights.qsa_geometry.head_dim]);
        q_with_gate[out + weights.qsa_geometry.head_dim..out + 2 * weights.qsa_geometry.head_dim]
            .copy_from_slice(
                &q_and_gate[q_width + q_off..q_width + q_off + weights.qsa_geometry.head_dim],
            );
    }
    let mut attn_heads = vec![0.0f32; q_width];
    ops::qsa_attention(
        &q_with_gate,
        &state.qsa.main_keys,
        &state.qsa.main_values,
        &selected,
        weights.qsa_geometry.q_heads,
        weights.qsa_geometry.kv_heads,
        weights.qsa_geometry.head_dim,
        &mut attn_heads,
    )
    .map_err(|_| MtpError::Shape("MTP QSA attention"))?;
    let attn_output = weights.attention.o_proj.mul_vec(&attn_heads)?;

    // 4. Inject attention output into each HC branch, then route the collapsed
    // branch view through normalized top-10 MoE.  The MLP HC stage repeats the
    // same ordered inject operation; the final mixer performs a learned read.
    hc_inject_output(
        &mut wide,
        &weights.attention_hyper,
        &attn_output,
        branches,
        hidden,
    )?;
    let mut moe_input = vec![0.0f32; hidden];
    for branch in 0..branches {
        for j in 0..hidden {
            moe_input[j] += wide[branch * hidden + j] / branches as f32;
        }
    }
    let router_logits = weights.moe.router.mul_vec(&moe_input)?;
    let (selected_experts, routing_weights) =
        ops::route_top10(&router_logits).map_err(|_| MtpError::Shape("MTP top-10 router"))?;
    let mut expert_outputs: Vec<Vec<f32>> = Vec::with_capacity(MTP_TOP_K);
    for &expert in &selected_experts {
        let gate_up = weights
            .moe
            .routed_gate_up
            .get(expert)
            .ok_or(MtpError::Shape("MTP routed expert index"))?;
        let down = weights
            .moe
            .routed_down
            .get(expert)
            .ok_or(MtpError::Shape("MTP routed expert index"))?;
        let gate = gate_up.mul_vec(&moe_input)?;
        let intermediate = gate.len() / 2;
        let activated = (0..intermediate)
            .map(|i| ops::silu(gate[i]) * gate[intermediate + i])
            .collect::<Vec<_>>();
        expert_outputs.push(down.mul_vec(&activated)?);
    }
    let shared_gate_vec = weights.moe.shared_gate.mul_vec(&moe_input)?;
    let shared_up = weights.moe.shared_up.mul_vec(&moe_input)?;
    let shared_gate_dim = shared_up.len();
    if shared_gate_vec.len() != shared_gate_dim {
        return Err(MtpError::Shape("MTP shared gate/up dimensions"));
    }
    let shared_activation = (0..shared_gate_dim)
        .map(|i| ops::silu(shared_gate_vec[i]) * shared_up[i])
        .collect::<Vec<_>>();
    let shared_output = weights.moe.shared_down.mul_vec(&shared_activation)?;
    let shared_scalar = weights
        .moe
        .shared_gate_scalar
        .iter()
        .zip(&moe_input)
        .map(|(&w, &x)| w * x)
        .sum::<f32>();
    let mut moe_output = vec![0.0f32; hidden];
    for (slot, expert_output) in expert_outputs.iter().enumerate() {
        for (dst, &value) in moe_output.iter_mut().zip(expert_output) {
            *dst += routing_weights[slot] * value;
        }
    }
    let shared_scale = ops::sigmoid(shared_scalar);
    for (dst, &value) in moe_output.iter_mut().zip(&shared_output) {
        *dst += shared_scale * value;
    }
    hc_inject_output(&mut wide, &weights.mlp_hyper, &moe_output, branches, hidden)?;
    let collapsed_hidden = hc_read(
        &weights.final_hyper.norm,
        &weights.final_hyper.down,
        &weights.final_hyper.up,
        &wide,
        branches,
        hidden,
    )?;
    state.wide_hidden = wide.clone();
    state.position = input.position + 1;
    state.step_index = input.step_index.wrapping_add(1);
    state.quiesced = true;
    state.in_flight = false;
    Ok(MtpForwardOutput {
        collapsed_hidden,
        wide_hidden: wide,
        qsa_indices: selected,
        selected_experts,
        routing_weights,
    })
}

fn hc_read(
    norm: &[f32],
    down: &MtpMatrix,
    up: &MtpMatrix,
    input: &[f32],
    branches: usize,
    hidden: usize,
) -> Result<Vec<f32>, MtpError> {
    let wide = branches
        .checked_mul(hidden)
        .ok_or(MtpError::Shape("MTP HC read dimensions"))?;
    if branches == 0
        || hidden == 0
        || input.len() != wide
        || norm.len() != wide
        || down.rows == 0
        || down.cols != wide
        || up.rows != wide
        || up.cols != down.rows
    {
        return Err(MtpError::Shape("MTP HC read dimensions"));
    }
    let mut normalized = vec![0.0f32; wide];
    let mut mixed = vec![0.0f32; hidden];
    ops::hc_read(
        input,
        &down.values,
        &up.values,
        norm,
        branches,
        hidden,
        down.rows,
        MTP_RMS_EPS,
        &mut normalized,
        &mut mixed,
    )
    .map_err(|_| MtpError::Shape("MTP HC read"))?;
    Ok(mixed)
}

fn hc_inject_output(
    wide: &mut [f32],
    weights: &MtpHyperWeights,
    output: &[f32],
    branches: usize,
    hidden: usize,
) -> Result<(), MtpError> {
    if wide.len() != branches * hidden || output.len() != hidden {
        return Err(MtpError::Shape("MTP HC inject dimensions"));
    }
    let mut normed = vec![0.0f32; wide.len()];
    ops::zero_centered_rms_norm(wide, &weights.norm, MTP_RMS_EPS, &mut normed)
        .map_err(|_| MtpError::Shape("MTP HC inject norm"))?;
    if weights.block_inject.rows != branches || weights.block_inject.cols != wide.len() {
        return Err(MtpError::Shape("MTP HC block injection dimensions"));
    }
    for branch in 0..branches {
        let row = &weights.block_inject.values[branch * wide.len()..(branch + 1) * wide.len()];
        let dot = row.iter().zip(&normed).map(|(&w, &x)| w * x).sum::<f32>() / branches as f32;
        let scale = 2.0 * ops::sigmoid(dot);
        for j in 0..hidden {
            wide[branch * hidden + j] += scale * output[j];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_hc_read_uses_learned_branch_mix() {
        let weights = MtpFinalHyperWeights {
            norm: vec![0.0; 4],
            down: MtpMatrix::new(1, 4, vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
            up: MtpMatrix::new(4, 1, vec![4.0, 0.0, 0.0, -4.0]).unwrap(),
        };
        let input = [3.0, 0.0, 0.0, 1.0];
        let mixed = hc_read(&weights.norm, &weights.down, &weights.up, &input, 2, 2).unwrap();
        let first = 3.0f32 / (4.5 + MTP_RMS_EPS).sqrt();
        let second = 1.0f32 / (0.5 + MTP_RMS_EPS).sqrt();
        let gate = 4.0 * ops::silu(first / 2.0);
        assert!((mixed[0] - first * ops::sigmoid(gate) / 2.0).abs() < 1.0e-6);
        assert!((mixed[1] - second * ops::sigmoid(-gate) / 2.0).abs() < 1.0e-6);
        assert!((mixed[1] - second / 2.0).abs() > 0.1);

        let mut changed = weights.clone();
        changed.down.values.fill(0.0);
        let down_mix = hc_read(&changed.norm, &changed.down, &changed.up, &input, 2, 2).unwrap();
        assert!((down_mix[0] - mixed[0]).abs() > 0.1);
        changed = weights.clone();
        changed.up.values.fill(0.0);
        let up_mix = hc_read(&changed.norm, &changed.down, &changed.up, &input, 2, 2).unwrap();
        assert!((up_mix[1] - mixed[1]).abs() > 0.1);

        changed.down = MtpMatrix::new(1, 2, vec![0.0; 2]).unwrap();
        assert!(matches!(
            hc_read(&changed.norm, &changed.down, &changed.up, &input, 2, 2),
            Err(MtpError::Shape("MTP HC read dimensions"))
        ));
        changed = weights;
        changed.up = MtpMatrix::new(2, 1, vec![0.0; 2]).unwrap();
        assert!(matches!(
            hc_read(&changed.norm, &changed.down, &changed.up, &input, 2, 2),
            Err(MtpError::Shape("MTP HC read dimensions"))
        ));
    }

    #[test]
    fn qsa_step_reuses_and_compacts_indices() {
        let geometry = MtpQsaGeometry {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 4,
            index_dim: 2,
            compress_ratio: 2,
            budget: 4,
            max_seq_len: 8,
            rotary_dim: 2,
            rope_theta: 10_000,
        };
        let mut qsa = MtpQsaState::new(geometry).unwrap();
        for pos in 0..3 {
            qsa.append_token(pos, &[1.0, 0.0], &[1.0, 0.0], &[pos as f32, 1.0])
                .unwrap();
        }
        let selected = qsa
            .select_step0(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .unwrap();
        assert!(selected.contains(&2));
        let reused = qsa.reuse_selection(3).unwrap();
        assert!(reused.contains(&3));
        qsa.compact_selection(&[0, 2]).unwrap();
        assert_eq!(qsa.selected_indices, vec![0, 2]);
    }

    #[test]
    fn qsa_snapshot_restores_side_state_without_full_capacity_copy() {
        let geometry = MtpQsaGeometry {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 1,
            index_dim: 2,
            compress_ratio: 2,
            budget: 4,
            max_seq_len: 16,
            rotary_dim: 2,
            rope_theta: 10_000,
        };
        let mut qsa = MtpQsaState::new(geometry).unwrap();
        qsa.append_token(0, &[1.0, 0.0], &[1.0, 0.0], &[0.0, 1.0])
            .unwrap();
        let snap = qsa.snapshot();
        qsa.append_token(1, &[0.0, 1.0], &[0.0, 1.0], &[1.0, 0.0])
            .unwrap();
        qsa.restore(&snap).unwrap();
        assert_eq!(qsa.position, 1);
        assert_eq!(qsa.main_token_count(), 1);
        assert_eq!(qsa.raw_positions, vec![0]);
        assert_eq!(qsa.incomplete_positions, vec![0]);
    }

    #[test]
    fn reset_and_cancel_clear_request_state() {
        let mut state = Qwen4MtpState::new(MtpQsaGeometry {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 1,
            index_dim: 2,
            compress_ratio: 2,
            budget: 4,
            max_seq_len: 8,
            rotary_dim: 2,
            rope_theta: 10_000,
        })
        .unwrap();
        state.wide_hidden = vec![1.0, 2.0];
        let snap = state.begin_transaction();
        state.wide_hidden.push(3.0);
        state.cancel(Some(snap)).unwrap();
        assert!(state.cancelled);
        assert_eq!(state.wide_hidden, vec![1.0, 2.0]);
        state.reset();
        assert!(!state.cancelled);
        assert!(state.wide_hidden.is_empty());
        assert!(state.is_quiesced());
    }
}
