// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Persistent GDN stream carry of a Qwen4 request state
//! (CHUNK-CANONICAL-DESIGN §3, §4.3).
//!
//! On gfx11 the Flash-Next ordinary-prefill scan is the C16 WY chunk kernel.
//! Its numerics depend on the absolute 16-row tile grid and (for a Q8-family
//! state) on absolute 512-row quantization seams, not on how a prompt is cut
//! into outer calls, provided each layer keeps (a) the F32 state at the entry
//! of the incomplete tile and (b) that tile's pending scan inputs. This module
//! is that per-layer carry: [`GdnCarry`] beside the recurrent state, the
//! convolution history and (Q8Ef) the residual of [`super::GdnGpuState`], owned
//! by `Qwen4State`, snapshotted/restored/serialized/digested with it, never a
//! thread-local or per-device cache.
//!
//! The scan inputs are stored post-convolution (BF16 rows exactly as the
//! kernel read them), so the live convolution history is never replayed: the
//! carry holds the F32 tile-entry state, the BF16 conv rows and the gate/beta
//! rows of the incomplete tile only.

use super::*;
use rdna_compute::gdn_stream::{
    GdnPrefillPlan, GdnProjectionRecipe, GdnScanAlgorithm, GdnStateQuant, GdnStreamMeta,
    GdnStreamPhase, GdnTensorFormat, GDN_COMMIT_ROWS, GDN_STREAM_SCHEMA_VERSION,
};
use rdna_compute::gdn_stream::GdnAppendSchedule;
use rdna_compute::tensor_ops::gdn_wy_stream::{
    wy_advance_meta, wy_carry_bytes, wy_stream_admitted, wy_stream_admitted_arch, WY_TILE_ROWS,
};

/// Rows of the dense C64 scan's tile (the Scan64 executor's microtile).
pub(crate) const SCAN64_TILE_ROWS: usize = 64;
/// Width-4 convolution history rows the Scan64 carry saves at its tile entry.
pub(crate) const SCAN64_CONV_HISTORY_ROWS: usize = 3;

/// Which scan serves a state's ordinary prefill stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StreamKind {
    /// gfx11 C16 WY chunk kernel (post-convolution BF16 scan inputs in the carry).
    Wy16,
    /// gfx1201 dense C64 scan through the Scan64 stream executor (raw pre-convolution F32
    /// inputs and the convolution history in the carry, `fn_gdn_dense` edge workspace).
    Scan64,
}

impl StreamKind {
    pub(crate) fn of_plan(plan: &GdnPrefillPlan) -> Self {
        match plan.algorithm {
            GdnScanAlgorithm::Scan64 => Self::Scan64,
            _ => Self::Wy16,
        }
    }

    pub(crate) const fn tile_rows(self) -> usize {
        match self {
            Self::Wy16 => WY_TILE_ROWS,
            Self::Scan64 => SCAN64_TILE_ROWS,
        }
    }
}

/// Session metadata words one carrying GDN layer adds
/// (`[schema|phase, recipe, position, phase_start, frame_base, tile_base,
/// pending_rows, owner_generation]`).
pub(crate) const STREAM_WORDS: usize = 8;

/// Persistent stream carry of one GDN layer.
pub struct GdnCarry {
    pub plan: GdnPrefillPlan,
    /// F32 recurrent state `[value_heads, 128, 128]` at `meta.tile_base`.
    pub tile_entry: GpuTensor,
    /// BF16 `[WY_TILE_ROWS, qkv]` convolution rows of the incomplete tile.
    pub pending_conv: GpuTensor,
    /// F32 `[2, WY_TILE_ROWS, value_heads]`: gate rows then beta rows.
    pub pending_params: GpuTensor,
    /// Host stream cursor; advanced only by a successful append/seal.
    pub meta: Cell<GdnStreamMeta>,
}

impl GdnCarry {
    /// The carry tensors in session / snapshot / digest order.
    pub(crate) fn tensors(&self) -> [&GpuTensor; 3] {
        [&self.tile_entry, &self.pending_conv, &self.pending_params]
    }

    pub fn meta(&self) -> GdnStreamMeta {
        self.meta.get()
    }

    /// The stream is open: the tile-entry state and pending rows are the
    /// authoritative state, the live slot is stale between seams.
    pub fn is_open(&self) -> bool {
        self.meta.get().phase == GdnStreamPhase::Open
    }
}

/// Shared per-state staging for the output/Z rows of a replayed tile
/// (sequential layers reuse one); charged once per state, never copied by a
/// snapshot.
pub struct GdnEdge {
    pub z: GpuTensor,
    pub out: GpuTensor,
    /// BF16 `[128]` gated-norm weight stand-in: the seal's replayed rows are
    /// discarded, so only a valid address is read.
    pub norm: GpuTensor,
}

/// The numerical recipe of a Qwen4 WY stream over `format`: algorithm, tile,
/// seam, operand/output storage and state quantization. The projection
/// recipe is the producer policy's (not resolved by the state): its fields
/// stay zero until a producer plan fills them, and are part of the identity.
pub(crate) fn gdn_stream_plan(format: GdnStateFormat, kind: StreamKind) -> GdnPrefillPlan {
    let mut plan = GdnPrefillPlan {
        recipe_id: 0,
        algorithm: match kind {
            StreamKind::Wy16 => GdnScanAlgorithm::Wy16,
            StreamKind::Scan64 => GdnScanAlgorithm::Scan64,
        },
        micro_rows: kind.tile_rows(),
        commit_rows: match format {
            GdnStateFormat::F32 => 0,
            GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => GDN_COMMIT_ROWS,
        },
        projection_recipe: GdnProjectionRecipe {
            qkv: 0,
            z: 0,
            alpha: 0,
            beta: 0,
            out: 0,
        },
        scan_operand_format: match kind {
            StreamKind::Wy16 => GdnTensorFormat::Bf16,
            StreamKind::Scan64 => GdnTensorFormat::F16,
        },
        scan_output_format: GdnTensorFormat::Bf16,
        state_quant: match format {
            GdnStateFormat::F32 => GdnStateQuant::F32,
            GdnStateFormat::Q8 => GdnStateQuant::Q8,
            GdnStateFormat::Q8Ef => GdnStateQuant::Q8Ef,
        },
        producer_tile_rows: 1,
    };
    plan.recipe_id = plan.compute_recipe_id();
    plan
}

/// Resolve the stream plan from static facts only (architecture, flags,
/// geometry, configured state quantization): `Some` exactly when the C16 WY
/// scan serves ordinary prefill on this device. Never reads a row count or free
/// memory. The dense C64 route (gfx1201, `fn_gdn_dense`) and the sequential
/// persistent recurrence carry nothing here: C64 carry arrives with the C64
/// edge executor (CHUNK-CANONICAL-DESIGN §3.3), the sequential route
/// continues its own F32 accumulator.
pub(crate) fn resolve_gdn_stream(
    gpu: &Gpu,
    config: &Qwen4Config,
    format: GdnStateFormat,
) -> Option<GdnPrefillPlan> {
    let kind = if wy_stream_admitted(
        gpu,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    ) {
        StreamKind::Wy16
    } else if rdna_compute::fn_gdn_dense::stream_admitted(
        gpu,
        format,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    ) {
        StreamKind::Scan64
    } else {
        return None;
    };
    Some(gdn_stream_plan(format, kind))
}

/// [`resolve_gdn_stream`]'s kind for an architecture name (no device handle, the
/// device-default flags): the byte accounting of an admission census makes the same decision
/// the state does.
pub(crate) fn stream_kind_for_arch(
    arch: &str,
    config: &Qwen4Config,
    format: GdnStateFormat,
) -> Option<StreamKind> {
    if wy_stream_admitted_arch(
        arch,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    ) {
        Some(StreamKind::Wy16)
    } else if rdna_compute::fn_gdn_dense::stream_geometry_admitted(
        arch,
        format,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        true,
    ) {
        Some(StreamKind::Scan64)
    } else {
        None
    }
}

/// Allocation specs of one carrying GDN layer, in allocation order:
/// `(name, elements, dtype)`. Same three tensors for both kinds: the F32 tile-entry state, the
/// raw/pending scan rows (WY: BF16 post-convolution rows; Scan64: the F32 convolution history of
/// the tile entry followed by the raw pre-convolution rows) and the F32 pending parameter rows
/// (gate/alpha then beta).
pub(crate) fn carry_layer_specs(
    config: &Qwen4Config,
    kind: StreamKind,
) -> [(&'static str, usize, DType); 3] {
    let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let state = config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim;
    match kind {
        StreamKind::Wy16 => [
            ("carry.tile_entry", state, DType::F32),
            ("carry.pending_conv", WY_TILE_ROWS * qkv, DType::BF16),
            (
                "carry.pending_params",
                2 * WY_TILE_ROWS * config.linear_num_value_heads,
                DType::F32,
            ),
        ],
        StreamKind::Scan64 => [
            ("carry.tile_entry", state, DType::F32),
            (
                "carry.pending_conv",
                (SCAN64_CONV_HISTORY_ROWS + SCAN64_TILE_ROWS) * qkv,
                DType::F32,
            ),
            (
                "carry.pending_params",
                2 * SCAN64_TILE_ROWS * config.linear_num_value_heads,
                DType::F32,
            ),
        ],
    }
}

/// Allocation specs of the shared edge staging: `(name, elements, dtype)`. WY: the replayed
/// tile's Z/output staging and a gated-norm stand-in. Scan64: the nine byte planes of
/// `Gpu::gdn_stream_workspace_alloc` (one 64-row tile of raw/prepared rows, its A blocks, the
/// scan output and a convolution ring scratch), sized exactly as that allocator does.
pub(crate) fn edge_specs(
    config: &Qwen4Config,
    kind: StreamKind,
) -> Vec<(&'static str, usize, DType)> {
    match kind {
        StreamKind::Wy16 => {
            let stage =
                WY_TILE_ROWS * config.linear_num_value_heads * config.linear_value_head_dim;
            vec![
                ("gdn_edge.z", stage, DType::F32),
                ("gdn_edge.out", stage, DType::F32),
                ("gdn_edge.norm", config.linear_value_head_dim, DType::BF16),
            ]
        }
        StreamKind::Scan64 => {
            let (hk, hv) = (config.linear_num_key_heads, config.linear_num_value_heads);
            let xdim = (2 * hk + hv) * 128;
            // Scan output BF16 (the plan's `scan_output_format`).
            vec![
                ("ws.x", 64 * xdim * 4, DType::Raw),
                ("ws.alpha", 64 * hv * 4, DType::Raw),
                ("ws.beta", 64 * hv * 4, DType::Raw),
                ("ws.q", 64 * hk * 128 * 2, DType::Raw),
                ("ws.k", 64 * hk * 128 * 2, DType::Raw),
                ("ws.v", 64 * hv * 128 * 2, DType::Raw),
                ("ws.a", 64 * hv * 64 * 2, DType::Raw),
                ("ws.out", 64 * hv * 128 * 2, DType::Raw),
                ("ws.conv", xdim * 3 * 4, DType::Raw),
            ]
        }
    }
}

/// A freshly reset / never-opened cursor.
pub(crate) fn sealed_cursor(plan: &GdnPrefillPlan, position: u64, generation: u64) -> GdnStreamMeta {
    GdnStreamMeta {
        recipe_id: plan.recipe_id,
        ..GdnStreamMeta::sealed_at(position, generation)
    }
}

fn pack_phase(meta: &GdnStreamMeta) -> u64 {
    u64::from(meta.schema_version)
        | (u64::from(meta.phase == GdnStreamPhase::Open) << 32)
}

/// Session words of one layer's cursor.
pub(crate) fn encode_meta(meta: &GdnStreamMeta) -> [u64; STREAM_WORDS] {
    [
        pack_phase(meta),
        meta.recipe_id,
        meta.position,
        meta.phase_start,
        meta.frame_base,
        meta.tile_base,
        u64::from(meta.pending_rows),
        // The generation is process-local: payload bytes stay deterministic.
        0,
    ]
}

/// Parse and validate one layer's cursor words against the state's plan and
/// the session position, before any live byte is written. A cursor of another
/// schema or recipe, or one whose fields cannot describe a tile boundary of
/// this plan, is `SnapshotShape`; missing words (a snapshot that predates the
/// carry) fail the caller's length check first.
pub(crate) fn decode_meta(
    words: &[u64],
    plan: &GdnPrefillPlan,
    session_position: u64,
    generation: u64,
) -> Result<GdnStreamMeta, StateError> {
    let [head, recipe, position, phase_start, frame_base, tile_base, pending, _owner] =
        <[u64; STREAM_WORDS]>::try_from(words).map_err(|_| StateError::SnapshotShape)?;
    let open = match head >> 32 {
        0 => false,
        1 => true,
        _ => return Err(StateError::SnapshotShape),
    };
    let schema = u32::try_from(head & 0xffff_ffff).map_err(|_| StateError::SnapshotShape)?;
    let pending_rows = u32::try_from(pending).map_err(|_| StateError::SnapshotShape)?;
    let coherent = schema == GDN_STREAM_SCHEMA_VERSION
        && recipe == plan.recipe_id
        && position == session_position
        && tile_base.checked_add(pending) == Some(position)
        && (pending as usize) < plan.micro_rows
        && phase_start <= frame_base
        && frame_base <= tile_base
        && (open || (pending == 0 && phase_start == position && frame_base == position));
    if !coherent {
        return Err(StateError::SnapshotShape);
    }
    Ok(GdnStreamMeta {
        schema_version: schema,
        recipe_id: recipe,
        phase: if open {
            GdnStreamPhase::Open
        } else {
            GdnStreamPhase::Sealed
        },
        position,
        phase_start,
        frame_base,
        tile_base,
        pending_rows,
        // A restore is a new owner generation of the restored prefix.
        owner_generation: generation,
    })
}

/// Cursor of a layer that `rows` rows past `entry` at `position - rows`
/// (used to describe a session cut inside a window without running the scan).
pub(crate) fn meta_after(
    entry: GdnStreamMeta,
    plan: &GdnPrefillPlan,
    entry_position: u64,
    position: u64,
) -> GdnStreamMeta {
    if position == entry_position {
        return match entry.phase {
            GdnStreamPhase::Open => entry,
            GdnStreamPhase::Sealed => sealed_cursor(plan, position, entry.owner_generation),
        };
    }
    match StreamKind::of_plan(plan) {
        StreamKind::Wy16 => wy_advance_meta(
            entry,
            plan.recipe_id,
            entry_position,
            position - entry_position,
            plan.state_quant != GdnStateQuant::F32,
        ),
        StreamKind::Scan64 => {
            let mut open = entry;
            if entry.phase == GdnStreamPhase::Sealed {
                open.position = entry_position;
            }
            let mut meta = GdnAppendSchedule::new(
                &open,
                entry_position,
                (position - entry_position) as usize,
                SCAN64_TILE_ROWS,
                GDN_COMMIT_ROWS,
            )
            .expect("a coherent cursor advances");
            meta.recipe_id = plan.recipe_id;
            meta
        }
    }
}
