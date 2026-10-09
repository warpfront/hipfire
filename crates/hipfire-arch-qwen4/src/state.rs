// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 GPU request state and CPU parity state.
//!
//! `Qwen4State` owns the GPU buffers and releases them through `free_gpu`.
//! `Reference*` structs are used only by CPU equation tests and parity probes.

use crate::config::{LayerType, Qwen4Config};
use crate::kv_backend::Qwen4KvBackend;
use crate::ple::PleHistory;
use hipfire_dispatch::pipeline::GdnRowCapture;
use hipfire_runtime::kv_backend::{
    KvChunkPlan, KvChunkPlanError, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
};
use hipfire_runtime::session_cache::{RowStream, StateLayout, StatePart};
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
            let (live, saved) = (&layer.raw_index_keys.buf, &arena.qsa_raw_circular[index].buf);
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
    copies.push(pair(
        &state.ple_conv,
        &arena.ple_conv,
        state.ple_conv.byte_size(),
    ));
    copies.push(pair(
        &state.hyper_feedback,
        &arena.hyper_feedback,
        state.hyper_feedback.byte_size(),
    ));
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
/// `GdnStateFormat` (F32, or a Raw Q8 slot) and the F32 convolution history.
/// The tensors must be freed by `Qwen4State::free_gpu`; no `Drop` attempts
/// to call HIP.
pub struct GdnGpuState {
    pub recurrent: GpuTensor,
    pub conv: GpuTensor,
}

/// A production QSA block.  Full K/V, raw and pooled index buffers have the
/// max sequence capacity shape; `*_len` fields are active append lengths. The
/// K/V arenas hold `format`'s rows (`full_row_units` tensor units per token)
/// and the raw/pooled index keys its `index_dtype`. Under the VMM QSA
/// backend those four context arenas are VMM owners whose
/// accessible prefix (`buf.size()`) is only what
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

    /// The four context arenas, in allocation order.
    fn context_arenas(&self) -> [&GpuTensor; QSA_CONTEXT_ARENAS] {
        [
            &self.full_keys,
            &self.full_values,
            &self.raw_index_keys,
            &self.pooled_keys,
        ]
    }

    fn context_arenas_mut(&mut self) -> [ContextArena<'_>; QSA_CONTEXT_ARENAS] {
        [
            ContextArena {
                tensor: &mut self.full_keys,
                rows: self.full_capacity,
                pooled: false,
            },
            ContextArena {
                tensor: &mut self.full_values,
                rows: self.full_capacity,
                pooled: false,
            },
            ContextArena {
                tensor: &mut self.raw_index_keys,
                rows: self.raw_capacity,
                pooled: false,
            },
            ContextArena {
                tensor: &mut self.pooled_keys,
                rows: self.pooled_capacity,
                pooled: true,
            },
        ]
    }
}

/// QSA context arenas per layer: full K, full V, raw and pooled index keys
/// (the first entries `Qwen4State::new_with_backend` allocates per layer).
const QSA_CONTEXT_ARENAS: usize = 4;

/// One QSA context arena owner with its row capacity.
struct ContextArena<'a> {
    tensor: &'a mut GpuTensor,
    rows: usize,
    /// One row per `compress` tokens instead of one per token.
    pooled: bool,
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
    gpu.hip
        .memset(&view.buf, 0, view.buf.size())
        .map_err(StateError::Hip)?;
    Ok(plan.token_capacity(mapped))
}

/// The QSA state format for a `memory.kv_cache` request on `gpu`
/// ([`hipfire_runtime::kv_mode::resolve_qwen4`]): `bf16` is the exact
/// reference state ([`QsaKvFormat::F32`]); `fp8` needs gfx1201 and a head
/// geometry its kernels implement (head_dim 256, an even KV-head count);
/// `q8` ([`QsaKvFormat::Q8`], opt-in) needs gfx11/gfx12 and head_dim 256.
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
        KvMode::Q8 if QsaKvFormat::Q8.supports(kv_heads, head_dim) => Ok(QsaKvFormat::Q8),
        KvMode::Q8 => Err(format!(
            "qwen4: q8 QSA K/V needs head_dim 256 (have {kv_heads} x {head_dim}); use bf16"
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

    /// Allocate all mutable device buffers once.  QSA full K/V, raw and
    /// pooled key arenas have the `max_seq_len` capacity shape in
    /// `format.qsa`, the GDN recurrent states `format.gdn`; only active
    /// lengths are mutable. Under [`Qwen4KvBackend::Vmm`] the four context
    /// arenas are VMM owners with nothing mapped: callers must
    /// [`Self::ensure_mapped_capacity`] before any row access. `backend` is
    /// load admission's choice; a VMM failure is an error, never a legacy
    /// fallback.
    pub(crate) fn new_with_backend(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
        format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
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
                        let device = gpu.device_id;
                        for (index, (elements, dtype)) in [
                            (full_elements, qsa_format.kv_dtype()),
                            (full_elements, qsa_format.kv_dtype()),
                            (raw_elements, qsa_format.index_dtype()),
                            (pooled_elements, qsa_format.index_dtype()),
                            (partial_key_elements, DType::F32),
                            (partial_value_elements, DType::F32),
                            (selected_bytes, DType::Raw),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let tensor = if index < QSA_CONTEXT_ARENAS
                                && backend == Qwen4KvBackend::Vmm
                            {
                                // SAFETY: nothing is mapped; every access is
                                // bounded by `ensure_mapped_capacity`'s prefix
                                // (`buf.size()` reports only mapped bytes).
                                unsafe { gpu.alloc_vmm_tensor(&[elements], dtype, 0, &[device]) }
                            } else {
                                gpu.zeros(&[elements], dtype)
                            };
                            allocated.push(tensor.map_err(StateError::Hip)?);
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
            qsa_backend: backend,
            qsa_mapped_tokens: match backend {
                Qwen4KvBackend::Legacy => max_seq_len,
                Qwen4KvBackend::Vmm => 0,
            },
            qsa_compress: config.indexer_compress_ratio,
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
        // VMM context arenas expose only their mapped prefix: nothing past
        // it exists to clear.
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
                if tensor.buf.size() == 0 {
                    continue;
                }
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
            for arena in layer.context_arenas_mut() {
                let rows = required_tokens.div_ceil(arena.tokens_per_row(compress));
                let mapped_rows =
                    grow_context_arena(gpu, arena.tensor, arena.rows, rows, device)?;
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
                    Qwen4KvBackend::Vmm => gpu
                        .vmm_mapped_bytes(tensor)
                        .ok_or(StateError::VmmOwner)?,
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
        fixed.push(whole_part(&self.ple_conv));
        fixed.push(whole_part(&self.hyper_feedback));
        Ok(StateLayout { fixed, rows })
    }

    /// Session-cache capture: metadata words and the device ranges that
    /// make up the live state. Never inside an armed verify.
    pub(crate) fn session_parts(&self) -> Result<(Vec<u8>, StateLayout<'_>), StateError> {
        if self.row_capture_armed {
            return Err(StateError::SnapshotBusy);
        }
        let [ple0, ple1] = self.ple_history.previous();
        let mut words = vec![0, self.position as u64, ple0.into(), ple1.into()];
        let mut marks = Vec::with_capacity(self.qsa.len());
        for layer in &self.qsa {
            let mark = layer.mark();
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
    /// position and PLE context, then six words per QSA layer.
    pub(crate) fn session_meta_bytes(&self) -> usize {
        8 * (4 + 6 * self.qsa.len())
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
        let ple = PleHistory::from_previous(
            self.ple_history.eos_token_id(),
            [to_u32(words[2])?, to_u32(words[3])?],
        );
        let marks = words[4..]
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
            required = required
                .max(mark.full_len)
                .max(mark.raw_len)
                .max(mark.pooled_len.saturating_mul(compress).min(self.max_seq_len));
        }
        self.ensure_mapped_capacity(gpu, required)?;
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
        self.snapshot_arena.invalidate();
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
        }
        self.ple_history = ple;
        self.position = position;
        Ok(())
    }

    /// Context-arena bytes still unmapped below `max_seq_len` (0 for legacy
    /// full-capacity arenas).
    pub(crate) fn context_growth_bytes(&self) -> u64 {
        if self.qsa_backend == Qwen4KvBackend::Legacy || self.max_seq_len == 0 {
            return 0;
        }
        let capacity: u64 = self
            .qsa
            .iter()
            .flat_map(|layer| layer.context_arenas())
            .map(|tensor| tensor.byte_size() as u64)
            .sum();
        capacity * self.max_seq_len.saturating_sub(self.qsa_mapped_tokens) as u64
            / self.max_seq_len as u64
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
            }
            let copies = arena_regions(self, &self.snapshot_arena, false)?;
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
        if rows < 2 {
            return Some(0);
        }
        let value_heads = config.linear_num_value_heads;
        let ring = gdn
            .state_units(
                value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            )
            .checked_mul(gdn.dtype().size())?
            .checked_mul(rows.checked_mul(2)?)?;
        let qkv = (2 * config.linear_num_key_heads)
            .checked_mul(config.linear_key_head_dim)?
            .checked_add(value_heads.checked_mul(config.linear_value_head_dim)?)?;
        let inputs = rows
            .checked_mul(qkv)?
            .checked_add(rows.checked_mul(qkv.checked_add(2 * value_heads)?)?)?
            .checked_mul(std::mem::size_of::<f32>())?;
        let gdn_layers = config.n_linear_layers();
        let output = rows
            .checked_mul(value_heads * GDN_HEAD_DIM)?
            .checked_mul(std::mem::size_of::<f32>())?;
        gdn_layers
            .checked_mul(ring.checked_add(inputs)?)?
            .checked_add(output)?
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
        assert_eq!(raw.token_capacity(raw.growth(0, 1).unwrap().unwrap().size_bytes), 4096);
        // Pooled keys map ceil(required / 4) rows: 4097 tokens are 1025 rows,
        // inside the first 4096-row page, which covers 16384 tokens.
        let pooled = context_arena_plan(512, s.div_ceil(4), 2 * MIB).unwrap();
        let rows = pooled.token_capacity(pooled.growth(0, 4097usize.div_ceil(4)).unwrap().unwrap().size_bytes);
        assert_eq!(rows, 4096);
        assert_eq!(qsa_covered_tokens(rows, 4, s), 16_384);
        // A short state's last pooled row covers its ragged tail only.
        assert_eq!(qsa_covered_tokens(10usize.div_ceil(4), 4, 10), 10);

        // FP8 K/V rows are not a power of two: whole rows per page only.
        assert_eq!(QsaKvFormat::Fp8.kv_row_bytes(2, 256), 516);
        let fp8 = context_arena_plan(516, s, 2 * MIB).unwrap();
        assert_eq!(fp8.token_capacity(2 * MIB), 4064);
    }

    #[test]
    fn row_capture_bytes_charge_the_gdn_storage_dtype() {
        let config = crate::config::compact_test_config();
        // 36 GDN layers x (2R ring slots + F32 inputs 4R x 10240 and
        // recurrence 4R x 10336) + 4R x 6144 F32 output + 36 x 16 table.
        assert_eq!(
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::F32, 4),
            Some(917_920_320)
        );
        // Q8 slots are 48 x 128 x 132 bytes, not 3 MiB of F32.
        assert_eq!(
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, 4),
            Some(245_520_960)
        );
        assert_eq!(
            Qwen4State::row_capture_bytes(&config, GdnStateFormat::Q8, 1),
            Some(0)
        );
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
        let full = config.qsa_context_arena_bytes(s, QsaKvFormat::F32).unwrap()
            * config.n_full_layers();
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), full);
        assert_eq!(state.mapped_context_tokens(), s);
        state.ensure_mapped_capacity(&mut gpu, s).expect("admitted tokens");
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
        state.ensure_mapped_capacity(&mut gpu, 1).expect("first page");
        let (bytes, first) = expected_vmm(granularity, s, 1);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(state.mapped_context_tokens(), first);
        assert!(first < s);
        let pattern = vec![0xA5u8; 2048];
        gpu.hip
            .memcpy_htod(&state.qsa[0].full_keys.buf, &pattern)
            .expect("write row 0");
        state.ensure_mapped_capacity(&mut gpu, first).expect("covered");
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);

        let old_size = state.qsa[0].full_keys.buf.size();
        state.ensure_mapped_capacity(&mut gpu, first + 1).expect("grow");
        let (bytes, tokens) = expected_vmm(granularity, s, first + 1);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), bytes);
        assert_eq!(state.mapped_context_tokens(), tokens);
        assert!(tokens > first);
        let keys = &state.qsa[0].full_keys;
        assert!(keys.buf.size() > old_size);
        let mut row = vec![0u8; 2048];
        gpu.hip.memcpy_dtoh(&mut row, &keys.buf).expect("read row 0");
        assert_eq!(row, pattern, "growth preserves mapped rows");
        let fresh = keys.sub_offset(old_size / 4, (keys.buf.size() - old_size) / 4);
        let mut grown = vec![0xFFu8; fresh.buf.size()];
        gpu.hip.memcpy_dtoh(&mut grown, &fresh.buf).expect("read new page");
        assert!(grown.iter().all(|&b| b == 0), "new pages are zeroed");

        state.ensure_mapped_capacity(&mut gpu, s).expect("whole context");
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
            assert!(formats.contains(&QsaKvFormat::Fp8), "gfx1201 serves fp8 QSA state");
        }
        let s = 16_384;
        let arenas = config.n_full_layers() * QSA_CONTEXT_ARENAS;
        for qsa in formats {
            let format = Qwen4StateFormat {
                qsa,
                gdn: GdnStateFormat::F32,
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
            assert_eq!(gpu.vmm_allocation_count(), owners + arenas, "{}", qsa.name());
            assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), 0);
            let granularity = gpu.vmm_granularity(&state.qsa[0].full_keys).unwrap();
            eprintln!("{} VMM QSA state: granularity {granularity} B", qsa.name());

            state.ensure_mapped_capacity(&mut gpu, 1).expect("first page");
            let first = state.mapped_context_tokens();
            assert!(first >= 1 && first < s, "{}: first coverage {first}", qsa.name());
            let keys = &state.qsa[0].full_keys;
            let row_bytes = keys.byte_size() / s;
            let pattern: Vec<u8> = (0..row_bytes).map(|i| (i % 251) as u8 | 1).collect();
            gpu.hip.memcpy_htod(&keys.buf, &pattern).expect("write row 0");

            let old_size = state.qsa[0].full_keys.buf.size();
            state.ensure_mapped_capacity(&mut gpu, first + 1).expect("grow");
            assert!(state.mapped_context_tokens() > first, "{}", qsa.name());
            let keys = &state.qsa[0].full_keys;
            assert!(keys.buf.size() > old_size);
            let mut row = vec![0u8; row_bytes];
            gpu.hip.memcpy_dtoh(&mut row, &keys.buf).expect("read row 0");
            assert_eq!(row, pattern, "{}: growth preserves mapped rows", qsa.name());
            let element = keys.dtype.size();
            let fresh = keys.sub_offset(old_size / element, (keys.buf.size() - old_size) / element);
            let mut grown = vec![0xFFu8; fresh.buf.size()];
            gpu.hip.memcpy_dtoh(&mut grown, &fresh.buf).expect("read new page");
            assert!(grown.iter().all(|&b| b == 0), "{}: new pages are zeroed", qsa.name());

            state.ensure_mapped_capacity(&mut gpu, s).expect("whole context");
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
            let mut state = Qwen4State::new_with_backend(&mut gpu, &config, s, format, backend)
                .expect("state");
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
                state.ensure_mapped_capacity(&mut gpu, tokens).expect("grow");
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
            eprintln!("refused {} tokens; committed unchanged {} MiB", s + 1, committed / MIB);
            state.free_gpu(&mut gpu).expect("free VMM");
            eprintln!("vmm owners after free: {}", gpu.vmm_allocation_count());
        }
    }
}
