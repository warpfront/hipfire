// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 GPU request state and CPU parity state.
//!
//! `Qwen4State` owns the GPU buffers and releases them through `free_gpu`.
//! `Reference*` structs are used only by CPU equation tests and parity probes.

use crate::config::{LayerType, Qwen4Config};
use crate::ple::PleHistory;
use hipfire_dispatch::pipeline::GdnRowCapture;
use rdna_compute::tensor_ops::{
    copy_regions, gated_delta_rollback_layers, CopyRegion, GatedDeltaRollbackLayers,
    GdnStateFormat, QsaKvFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt;
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
/// `GdnStateFormat` (F32, or a Raw Q8 slot) and the F32 convolution history.
/// The tensors must be freed by `Qwen4State::free_gpu`; no `Drop` attempts
/// to call HIP.
pub struct GdnGpuState {
    pub recurrent: GpuTensor,
    pub conv: GpuTensor,
}

/// A production QSA block.  Full K/V and raw/indexer buffers are allocated at
/// max sequence capacity once; `*_len` fields are active append lengths. The
/// K/V arenas hold `format`'s rows (`full_row_units` tensor units per token)
/// and the raw/pooled index keys its `index_dtype`.
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
}

/// The QSA state format for a `memory.kv_cache` request on `gpu`
/// ([`hipfire_runtime::kv_mode::resolve_qwen4`]): `bf16` is the exact
/// reference state ([`QsaKvFormat::F32`]); `fp8` needs gfx1201 and a head
/// geometry its kernels implement (head_dim 256, an even KV-head count).
/// `auto` is fp8 where both hold and the reference state elsewhere; an
/// explicit request the device or model cannot serve is refused, never
/// rewritten.
pub fn resolve_qsa_format(
    request: &str,
    gpu: &Gpu,
    config: &Qwen4Config,
) -> Result<QsaKvFormat, String> {
    use hipfire_runtime::kv_mode::{resolve_qwen4, KvMode};
    let (kv_heads, head_dim) = (config.num_key_value_heads, config.head_dim);
    let fp8_geometry = QsaKvFormat::Fp8.supports(kv_heads, head_dim);
    let auto = matches!(request.trim(), "" | "auto");
    match resolve_qwen4(request, &gpu.arch)?.mode {
        KvMode::Fp8 if fp8_geometry => Ok(QsaKvFormat::Fp8),
        KvMode::Fp8 if auto => Ok(QsaKvFormat::F32),
        KvMode::Fp8 => Err(format!(
            "qwen4: fp8 QSA K/V needs head_dim 256 and an even KV-head count \
             (have {kv_heads} x {head_dim}); use bf16"
        )),
        _ => Ok(QsaKvFormat::F32),
    }
}

/// The GDN recurrent state format for a `state_quant` request (Qwen3.5's
/// DeltaNet knob): `auto`/`q8` is Qwen3.5's Q8 state where its kernels apply
/// (128x128 heads), `fp32` the exact reference state. An explicit request the
/// model cannot serve is refused, never rewritten.
pub fn resolve_gdn_format(request: &str, config: &Qwen4Config) -> Result<GdnStateFormat, String> {
    let (key_dim, value_dim) = (config.linear_key_head_dim, config.linear_value_head_dim);
    let q8 = GdnStateFormat::Q8.supports(key_dim, value_dim);
    match request.trim().to_ascii_lowercase().as_str() {
        "" | "auto" if q8 => Ok(GdnStateFormat::Q8),
        "" | "auto" | "fp32" | "f32" => Ok(GdnStateFormat::F32),
        "q8" | "int8" if q8 => Ok(GdnStateFormat::Q8),
        "q8" | "int8" => Err(format!(
            "qwen4: q8 GDN state needs 128x128 heads (have {key_dim} x {value_dim}); use fp32"
        )),
        other => Err(format!(
            "qwen4: unsupported GDN state_quant '{other}' (expected auto|q8|fp32)"
        )),
    }
}

/// Storage formats of a Qwen4 request state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Qwen4StateFormat {
    /// QSA K/V and index keys ([`resolve_qsa_format`]).
    pub qsa: QsaKvFormat,
    /// GDN recurrent states ([`resolve_gdn_format`]).
    pub gdn: GdnStateFormat,
}

impl Qwen4StateFormat {
    /// The exact F32 reference state.
    pub const F32: Self = Self {
        qsa: QsaKvFormat::F32,
        gdn: GdnStateFormat::F32,
    };
}

/// Both state formats: `kv_request` is `memory.kv_cache`, `state_request`
/// the `state_quant` load option.
pub fn resolve_state_format(
    kv_request: &str,
    state_request: &str,
    gpu: &Gpu,
    config: &Qwen4Config,
) -> Result<Qwen4StateFormat, String> {
    Ok(Qwen4StateFormat {
        qsa: resolve_qsa_format(kv_request, gpu, config)?,
        gdn: resolve_gdn_format(state_request, config)?,
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

/// Reusable model-owned GPU snapshot storage.  The append-only QSA full K/V
/// arenas are deliberately absent: a ticket records their active marks and
/// restore only rewinds those marks.  Only state that a forward may overwrite
/// in place is copied into this bounded arena.
pub(crate) struct Qwen4StateSnapshotArena {
    recurrent: Vec<GpuTensor>,
    conv: Vec<GpuTensor>,
    qsa_partial_keys: Vec<GpuTensor>,
    qsa_partial_values: Vec<GpuTensor>,
    qsa_raw_circular: Vec<GpuTensor>,
    qsa_selected: Vec<GpuTensor>,
    ple_conv: GpuTensor,
    hyper_feedback: GpuTensor,
    qsa_marks: Vec<QsaMark>,
    ple_history: PleHistory,
    position: usize,
    /// Live GDN ring slot when the snapshot brackets an armed verify: the
    /// recurrent state was not copied (the verify leaves that slot intact).
    gdn_live: Option<usize>,
    model_id: u64,
    active: bool,
    generation: u64,
}

impl Qwen4StateSnapshotArena {
    fn new(
        gpu: &mut Gpu,
        gdn: &[GdnGpuState],
        qsa: &[QsaGpuState],
        ple_conv: &GpuTensor,
        hyper_feedback: &GpuTensor,
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
            allocated.push(
                gpu.zeros(&ple_conv.shape, DType::F32)
                    .map_err(StateError::Hip)?,
            );
            allocated.push(
                gpu.zeros(&hyper_feedback.shape, DType::F32)
                    .map_err(StateError::Hip)?,
            );
            Ok(())
        })();
        if let Err(error) = result {
            for tensor in allocated {
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
        let ple_conv = next();
        let hyper_feedback = next();
        if !allocated.is_empty() {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(StateError::AllocationBookkeeping);
        }
        Ok(Self {
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
            generation: 0,
        })
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
        }
        mix(state.ple_conv.numel());
        mix(state.hyper_feedback.numel());
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
        for (layer, backup) in state.gdn.iter().zip(&self.recurrent) {
            if backup.numel() != layer.recurrent.numel()
                || backup.buf.size() != layer.recurrent.buf.size()
            {
                return Err(StateError::SnapshotShape);
            }
        }
        for (layer, backup) in state.gdn.iter().zip(&self.conv) {
            if backup.numel() != layer.conv.numel() || backup.buf.size() != layer.conv.buf.size() {
                return Err(StateError::SnapshotShape);
            }
        }
        for (index, layer) in state.qsa.iter().enumerate() {
            if self.qsa_partial_keys[index].numel() != layer.partial_keys.numel()
                || self.qsa_partial_values[index].numel() != layer.partial_values.numel()
                || self.qsa_selected[index].buf.size() != layer.selected_indices.buf.size()
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
        if self.ple_conv.numel() != state.ple_conv.numel()
            || self.hyper_feedback.numel() != state.hyper_feedback.numel()
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
        self.active = false;
        self.generation = self.generation.wrapping_add(1);
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
            .chain([self.ple_conv, self.hyper_feedback]);
        let mut first = None;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The sole production request-state owner for Qwen4.
pub struct Qwen4State {
    pub gdn: Vec<GdnGpuState>,
    pub qsa: Vec<QsaGpuState>,
    pub ple_conv: GpuTensor,
    pub hyper_feedback: GpuTensor,
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
    /// Rows a verify may capture (0 = none allocated); the ring holds twice
    /// as many slots.
    row_capture_rows: usize,
    /// Live ring slot of every GDN layer's recurrent state.
    gdn_live: usize,
    /// Storage of every GDN layer's recurrent state (and its ring slots).
    gdn_format: GdnStateFormat,
    gdn_value_heads: usize,
    /// Set only around a speculative verify forward.
    pub(crate) row_capture_armed: bool,
    model_id: u64,
    reset_epoch: u64,
    transaction_generation: u64,
}

impl Qwen4State {
    /// Allocate all mutable device buffers once.  QSA full K/V and raw key
    /// buffers are fixed-capacity arenas in `format.qsa`, the GDN recurrent
    /// states in `format.gdn`; only active lengths are mutable.
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
    ) -> Result<Self, StateError> {
        config.validate().map_err(StateError::Config)?;
        let qsa_format = format.qsa;
        if !qsa_format.supports(config.num_key_value_heads, config.head_dim) {
            return Err(StateError::Config(format!(
                "QSA {} K/V does not support {} KV heads x {}",
                qsa_format.name(),
                config.num_key_value_heads,
                config.head_dim
            )));
        }
        if max_seq_len == 0 || max_seq_len > config.max_position_embeddings {
            return Err(StateError::InvalidCapacity);
        }
        let gdn_format = format.gdn;
        if !gdn_format.supports(config.linear_key_head_dim, config.linear_value_head_dim) {
            return Err(StateError::Config(format!(
                "GDN {} state needs 128x128 heads, not {} x {}",
                gdn_format.name(),
                config.linear_key_head_dim,
                config.linear_value_head_dim
            )));
        }
        let gdn_recurrent = gdn_format.state_units(
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        );
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
        let full_row_units = qsa_format.kv_row_units(config.num_key_value_heads, config.head_dim);
        let raw_width = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)
            .ok_or(StateError::DimensionOverflow)?;
        let pooled_capacity = max_seq_len
            .checked_add(config.indexer_compress_ratio - 1)
            .ok_or(StateError::DimensionOverflow)?
            / config.indexer_compress_ratio;
        let selected_capacity = config.qsa_selected_capacity();
        let mut allocated = Vec::new();
        let allocation = (|| -> Result<(), StateError> {
            for kind in &config.layer_types {
                match kind {
                    LayerType::LinearAttention => {
                        allocated.push(
                            gpu.zeros(&[gdn_recurrent], format.gdn.dtype())
                                .map_err(StateError::Hip)?,
                        );
                        allocated.push(
                            gpu.zeros(&[conv_len], DType::F32)
                                .map_err(StateError::Hip)?,
                        );
                    }
                    LayerType::FullAttention => {
                        let full_elements = max_seq_len
                            .checked_mul(full_row_units)
                            .ok_or(StateError::DimensionOverflow)?;
                        let raw_elements = max_seq_len
                            .checked_mul(raw_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let pooled_elements = pooled_capacity
                            .checked_mul(raw_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let partial_key_elements = config
                            .indexer_compress_ratio
                            .checked_mul(raw_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let partial_value_elements = config
                            .indexer_compress_ratio
                            .checked_mul(full_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let selected_bytes = selected_capacity
                            .checked_mul(std::mem::size_of::<i32>())
                            .ok_or(StateError::DimensionOverflow)?;
                        for (elements, dtype) in [
                            (full_elements, qsa_format.kv_dtype()),
                            (full_elements, qsa_format.kv_dtype()),
                            (raw_elements, qsa_format.index_dtype()),
                            (pooled_elements, qsa_format.index_dtype()),
                            (partial_key_elements, DType::F32),
                            (partial_value_elements, DType::F32),
                            (selected_bytes, DType::Raw),
                        ] {
                            allocated.push(gpu.zeros(&[elements], dtype).map_err(StateError::Hip)?);
                        }
                    }
                }
            }
            let ple_channels = config
                .ple_embed_dim
                .checked_mul(config.hc_count)
                .ok_or(StateError::DimensionOverflow)?;
            let ple_elements = config
                .ple_conv_history_rows()
                .checked_mul(ple_channels)
                .ok_or(StateError::DimensionOverflow)?;
            allocated.push(
                gpu.zeros(&[ple_elements], DType::F32)
                    .map_err(StateError::Hip)?,
            );
            let feedback_elements = config
                .hc_count
                .checked_mul(config.hidden_size)
                .ok_or(StateError::DimensionOverflow)?;
            allocated.push(
                gpu.zeros(&[feedback_elements], DType::F32)
                    .map_err(StateError::Hip)?,
            );
            Ok(())
        })();
        if let Err(error) = allocation {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }

        let expected_tensors = config
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::LinearAttention => 2,
                LayerType::FullAttention => 7,
            })
            .sum::<usize>()
            .checked_add(2)
            .ok_or(StateError::DimensionOverflow)?;
        if allocated.len() != expected_tensors {
            for tensor in allocated {
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
        let mut gdn = Vec::with_capacity(config.n_linear_layers());
        let mut qsa = Vec::with_capacity(config.n_full_layers());
        for kind in &config.layer_types {
            match kind {
                LayerType::LinearAttention => {
                    let recurrent = next_tensor();
                    let conv = next_tensor();
                    gdn.push(GdnGpuState { recurrent, conv });
                }
                LayerType::FullAttention => {
                    let full_keys = next_tensor();
                    let full_values = next_tensor();
                    let raw_index_keys = next_tensor();
                    let pooled_keys = next_tensor();
                    let partial_keys = next_tensor();
                    let partial_values = next_tensor();
                    let selected_indices = next_tensor();
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
        let ple_conv = next_tensor();
        let hyper_feedback = next_tensor();
        let model_id = next_qwen4_model_id();
        let ple_history = PleHistory::new(config.eos_token_id);
        let snapshot_arena = match Qwen4StateSnapshotArena::new(
            gpu,
            &gdn,
            &qsa,
            &ple_conv,
            &hyper_feedback,
            model_id,
            ple_history,
            0,
        ) {
            Ok(arena) => arena,
            Err(error) => {
                for layer in gdn {
                    let _ = gpu.free_tensor(layer.recurrent);
                    let _ = gpu.free_tensor(layer.conv);
                }
                for layer in qsa {
                    for tensor in [
                        layer.full_keys,
                        layer.full_values,
                        layer.raw_index_keys,
                        layer.pooled_keys,
                        layer.partial_keys,
                        layer.partial_values,
                        layer.selected_indices,
                    ] {
                        let _ = gpu.free_tensor(tensor);
                    }
                }
                let _ = gpu.free_tensor(ple_conv);
                let _ = gpu.free_tensor(hyper_feedback);
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
            row_capture_rows: 0,
            gdn_live: 0,
            gdn_format: format.gdn,
            gdn_value_heads: config.linear_num_value_heads,
            row_capture_armed: false,
            model_id,
            reset_epoch: 0,
            transaction_generation: 0,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), StateError> {
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
        self.snapshot_arena.invalidate();
        for layer in &self.gdn {
            gpu.hip
                .memset(&layer.recurrent.buf, 0, layer.recurrent.buf.size())
                .map_err(StateError::Hip)?;
            gpu.hip
                .memset(&layer.conv.buf, 0, layer.conv.buf.size())
                .map_err(StateError::Hip)?;
        }
        for layer in &mut self.qsa {
            for tensor in [
                &layer.full_keys,
                &layer.full_values,
                &layer.raw_index_keys,
                &layer.pooled_keys,
                &layer.partial_keys,
                &layer.partial_values,
                &layer.selected_indices,
            ] {
                gpu.hip
                    .memset(&tensor.buf, 0, tensor.buf.size())
                    .map_err(StateError::Hip)?;
            }
            layer.full_len = 0;
            layer.raw_len = 0;
            layer.pooled_len = 0;
            layer.partial_len = 0;
            layer.selected_len = 0;
            layer.position = 0;
        }
        gpu.hip
            .memset(&self.ple_conv.buf, 0, self.ple_conv.buf.size())
            .map_err(StateError::Hip)?;
        gpu.hip
            .memset(&self.hyper_feedback.buf, 0, self.hyper_feedback.buf.size())
            .map_err(StateError::Hip)?;
        self.ple_history.reset();
        self.position = 0;
        Ok(())
    }
    /// Bind rollback tickets to the owning runtime transaction generation.
    pub(crate) fn bind_transaction_generation(&mut self, generation: u64) {
        self.transaction_generation = generation;
        self.snapshot_arena.invalidate();
    }
    /// Capture the fixed-size rollback state into the model-owned arena.
    /// Full-capacity append-only QSA K/V and raw arenas are never copied.
    pub fn snapshot(&mut self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, StateError> {
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
                (self.row_capture_armed && !self.row_capture.is_empty()).then_some(self.gdn_live);
            let arena = &self.snapshot_arena;
            let mut copies = Vec::with_capacity(2 * self.gdn.len() + 4 * self.qsa.len() + 2);
            if arena.gdn_live.is_none() {
                for (layer, destination) in self.gdn.iter().zip(&arena.recurrent) {
                    copies.push(whole(
                        &layer.recurrent,
                        destination,
                        layer.recurrent.byte_size(),
                    ));
                }
            }
            for (layer, destination) in self.gdn.iter().zip(&arena.conv) {
                copies.push(whole(&layer.conv, destination, layer.conv.byte_size()));
            }
            for (index, layer) in self.qsa.iter().enumerate() {
                copies.push(whole(
                    &layer.partial_keys,
                    &arena.qsa_partial_keys[index],
                    layer.partial_keys.byte_size(),
                ));
                copies.push(whole(
                    &layer.partial_values,
                    &arena.qsa_partial_values[index],
                    layer.partial_values.byte_size(),
                ));
                copies.push(whole(
                    &layer.selected_indices,
                    &arena.qsa_selected[index],
                    layer.selected_indices.byte_size(),
                ));
                let raw_width = layer
                    .raw_index_keys
                    .numel()
                    .checked_div(layer.raw_capacity)
                    .ok_or(StateError::SnapshotShape)?;
                let rows = layer.partial_capacity.min(layer.raw_len);
                if rows > 0 {
                    let bytes = rows
                        .checked_mul(raw_width)
                        .and_then(|v| v.checked_mul(layer.raw_index_keys.dtype.size()))
                        .ok_or(StateError::DimensionOverflow)?;
                    let source_offset = (layer.raw_len - rows)
                        .checked_mul(raw_width)
                        .and_then(|v| v.checked_mul(layer.raw_index_keys.dtype.size()))
                        .ok_or(StateError::DimensionOverflow)?;
                    copies.push(CopyRegion {
                        dst: &arena.qsa_raw_circular[index].buf,
                        dst_offset: 0,
                        src: &layer.raw_index_keys.buf,
                        src_offset: source_offset,
                        bytes,
                    });
                }
            }
            copies.push(whole(
                &self.ple_conv,
                &arena.ple_conv,
                self.ple_conv.byte_size(),
            ));
            copies.push(whole(
                &self.hyper_feedback,
                &arena.hyper_feedback,
                self.hyper_feedback.byte_size(),
            ));
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
            }
            let arena = &self.snapshot_arena;
            let mut copies = Vec::with_capacity(2 * self.gdn.len() + 4 * self.qsa.len() + 2);
            if arena.gdn_live.is_none() {
                for (layer, source) in self.gdn.iter().zip(&arena.recurrent) {
                    copies.push(whole(source, &layer.recurrent, layer.recurrent.byte_size()));
                }
            }
            for (layer, source) in self.gdn.iter().zip(&arena.conv) {
                copies.push(whole(source, &layer.conv, layer.conv.byte_size()));
            }
            for (index, layer) in self.qsa.iter().enumerate() {
                copies.push(whole(
                    &arena.qsa_partial_keys[index],
                    &layer.partial_keys,
                    layer.partial_keys.byte_size(),
                ));
                copies.push(whole(
                    &arena.qsa_partial_values[index],
                    &layer.partial_values,
                    layer.partial_values.byte_size(),
                ));
                copies.push(whole(
                    &arena.qsa_selected[index],
                    &layer.selected_indices,
                    layer.selected_indices.byte_size(),
                ));
                let mark = arena.qsa_marks[index];
                let raw_width = layer
                    .raw_index_keys
                    .numel()
                    .checked_div(layer.raw_capacity)
                    .ok_or(StateError::SnapshotShape)?;
                let rows = layer.partial_capacity.min(mark.raw_len);
                if rows > 0 {
                    let bytes = rows
                        .checked_mul(raw_width)
                        .and_then(|v| v.checked_mul(layer.raw_index_keys.dtype.size()))
                        .ok_or(StateError::DimensionOverflow)?;
                    let destination_offset = (mark.raw_len - rows)
                        .checked_mul(raw_width)
                        .and_then(|v| v.checked_mul(layer.raw_index_keys.dtype.size()))
                        .ok_or(StateError::DimensionOverflow)?;
                    copies.push(CopyRegion {
                        dst: &layer.raw_index_keys.buf,
                        dst_offset: destination_offset,
                        src: &arena.qsa_raw_circular[index].buf,
                        src_offset: 0,
                        bytes,
                    });
                }
            }
            copies.push(whole(
                &arena.ple_conv,
                &self.ple_conv,
                self.ple_conv.byte_size(),
            ));
            copies.push(whole(
                &arena.hyper_feedback,
                &self.hyper_feedback,
                self.hyper_feedback.byte_size(),
            ));
            copy_regions(gpu, &copies).map_err(StateError::Hip)?;
            let arena = &mut self.snapshot_arena;
            for (index, layer) in self.qsa.iter_mut().enumerate() {
                let mark = arena.qsa_marks[index];
                layer.full_len = mark.full_len;
                layer.raw_len = mark.raw_len;
                layer.pooled_len = mark.pooled_len;
                layer.partial_len = mark.partial_len;
                layer.selected_len = mark.selected_len;
                layer.position = mark.position;
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
            free(layer.conv);
        }
        for layer in self.qsa {
            free(layer.full_keys);
            free(layer.full_values);
            free(layer.raw_index_keys);
            free(layer.pooled_keys);
            free(layer.partial_keys);
            free(layer.partial_values);
            free(layer.selected_indices);
        }
        free(self.ple_conv);
        free(self.hyper_feedback);
        for (ring, inputs, recurrence) in self.row_capture {
            free(ring);
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
    /// blocks: per GDN layer a two-half recurrent-state ring, the input rows
    /// and the recurrence inputs, then one output block and the pointer
    /// table. The single-slot recurrent state each ring replaces is freed
    /// into the GPU pool, which keeps it.
    pub(crate) fn row_capture_bytes(config: &Qwen4Config, rows: usize) -> Option<usize> {
        if rows < 2 {
            return Some(0);
        }
        let value_heads = config.linear_num_value_heads;
        let state = value_heads
            .checked_mul(config.linear_value_head_dim)?
            .checked_mul(config.linear_key_head_dim)?;
        let qkv = (2 * config.linear_num_key_heads)
            .checked_mul(config.linear_key_head_dim)?
            .checked_add(value_heads.checked_mul(config.linear_value_head_dim)?)?;
        let layer = (2 * rows)
            .checked_mul(state)?
            .checked_add(rows.checked_mul(qkv)?)?
            .checked_add(rows.checked_mul(qkv.checked_add(2 * value_heads)?)?)?;
        let gdn_layers = config.n_linear_layers();
        gdn_layers
            .checked_mul(layer)?
            .checked_add(rows.checked_mul(value_heads * GDN_HEAD_DIM)?)?
            .checked_mul(std::mem::size_of::<f32>())?
            .checked_add(gdn_layers.checked_mul(2 * std::mem::size_of::<u64>())?.max(1))
    }

    /// Allocate the few-row verify rollback points for blocks of up to
    /// `rows` rows (`conv_history` = convolution kernel - 1). Only called
    /// where the GDN route captures rows (see `GatedDeltaNetOp::row_capture`).
    pub(crate) fn ensure_row_capture(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
        conv_history: usize,
    ) -> Result<(), StateError> {
        if rows <= self.row_capture_rows || rows < 2 {
            return Ok(());
        }
        let mut next = Vec::with_capacity(self.gdn.len());
        let mut widest_value = 0;
        for layer in &self.gdn {
            let state = layer.recurrent.numel();
            let ring = gpu
                .zeros(&[2 * rows * state], layer.recurrent.dtype)
                .map_err(StateError::Hip)?;
            gpu.copy_d2d(
                &layer.recurrent,
                &ring.sub_offset(0, state),
                layer.recurrent.byte_size(),
            )
            .map_err(StateError::Hip)?;
            let qkv = layer.conv.numel() / conv_history;
            let inputs = gpu
                .zeros(&[rows * qkv], DType::F32)
                .map_err(StateError::Hip)?;
            let value_heads = self.gdn_value_heads;
            let recurrence = gpu
                .zeros(&[rows * (qkv + 2 * value_heads)], DType::F32)
                .map_err(StateError::Hip)?;
            widest_value = widest_value.max(value_heads * GDN_HEAD_DIM);
            next.push((ring, inputs, recurrence));
        }
        let output = gpu
            .zeros(&[rows * widest_value], DType::F32)
            .map_err(StateError::Hip)?;
        let pointers: Vec<u8> = next
            .iter()
            .flat_map(|(ring, _, recurrence)| [recurrence.buf.as_ptr(), ring.buf.as_ptr()])
            .flat_map(|pointer| (pointer as u64).to_ne_bytes())
            .collect();
        let table = gpu
            .zeros(&[pointers.len().max(1)], DType::Raw)
            .map_err(StateError::Hip)?;
        gpu.memcpy_htod_auto(&table.buf, &pointers)
            .map_err(StateError::Hip)?;
        if let Some((output, table)) = self.row_capture_output.replace((output, table)) {
            gpu.free_tensor(output).map_err(StateError::Hip)?;
            gpu.free_tensor(table).map_err(StateError::Hip)?;
        }
        let previous = std::mem::replace(&mut self.row_capture, next);
        let ringed = !previous.is_empty();
        for (index, layer) in self.gdn.iter_mut().enumerate() {
            let state = layer.recurrent.numel();
            let old = std::mem::replace(
                &mut layer.recurrent,
                self.row_capture[index].0.sub_offset(0, state),
            );
            if !ringed {
                gpu.free_tensor(old).map_err(StateError::Hip)?;
            }
        }
        for (ring, inputs, recurrence) in previous {
            gpu.free_tensor(ring).map_err(StateError::Hip)?;
            gpu.free_tensor(inputs).map_err(StateError::Hip)?;
            gpu.free_tensor(recurrence).map_err(StateError::Hip)?;
        }
        self.row_capture_rows = rows;
        self.gdn_live = 0;
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
        for (layer, (ring, _, _)) in self.gdn.iter_mut().zip(&self.row_capture) {
            let state = layer.recurrent.numel();
            layer.recurrent = ring.sub_offset(slot * state, state);
        }
    }

    /// The rollback points GDN layer `slot` writes in an armed `rows`-row
    /// verify forward: its last row's state into the free half of the ring,
    /// plus the convolution and recurrence inputs.
    pub(crate) fn gdn_row_capture(&self, slot: usize, rows: usize) -> Option<GdnRowCapture<'_>> {
        if !self.row_capture_armed || rows < 2 || rows > self.row_capture_rows {
            return None;
        }
        let state = self.gdn.get(slot)?.recurrent.numel();
        let base = self.capture_base(self.gdn_live);
        self.row_capture
            .get(slot)
            .map(|(ring, inputs, recurrence)| GdnRowCapture {
                states: ring.sub_offset(base * state, rows * state),
                inputs,
                recurrence,
            })
    }

    /// An armed `rows`-row verify forward succeeded: its last row's state is
    /// live.
    pub(crate) fn commit_row_capture(&mut self, rows: usize) {
        if self.row_capture_armed && rows >= 2 && rows <= self.row_capture_rows {
            let base = self.capture_base(self.gdn_live);
            self.set_gdn_live(base + rows - 1);
        }
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
        let (output, table) = self
            .row_capture_output
            .as_ref()
            .ok_or(StateError::SnapshotTicket)?;
        if let Some(first) = self.gdn.first() {
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
                    format: self.gdn_format,
                    layers: self.gdn.len(),
                    rows,
                    keep,
                    from: before,
                    to: base,
                    qkv_width: qkv,
                    key_heads: (qkv - value_heads * GDN_HEAD_DIM) / (2 * GDN_HEAD_DIM),
                    value_heads,
                    position: start + keep - 1,
                },
            )
            .map_err(StateError::Hip)?;
        }
        self.set_gdn_live(base + keep - 1);
        let arena = &self.snapshot_arena;
        let row_bytes = self.ple_conv.byte_size() / ple_history_rows;
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
        for row in 0..ple_history_rows {
            let source = keep + row;
            let (src, src_offset) = if source < ple_history_rows {
                (&arena.ple_conv.buf, source * row_bytes)
            } else {
                (&ple_normed.buf, (source - ple_history_rows) * row_bytes)
            };
            copies.push(CopyRegion {
                dst: &self.ple_conv.buf,
                dst_offset: row * row_bytes,
                src,
                src_offset,
                bytes: row_bytes,
            });
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
        }
        let mut history = arena.ple_history;
        for &token in &tokens[..keep] {
            history.push(token);
        }
        self.ple_history = history;
        self.position = start + keep;
        Ok(())
    }
    pub fn qsa_mut(&mut self, full_layer_index: usize) -> Option<&mut QsaGpuState> {
        self.qsa.get_mut(full_layer_index)
    }

    pub fn gdn_mut(&mut self, linear_layer_index: usize) -> Option<&mut GdnGpuState> {
        self.gdn.get_mut(linear_layer_index)
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
        }
    }
}

impl std::error::Error for StateError {}

#[cfg(test)]
mod tests {
    use super::*;

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
}
