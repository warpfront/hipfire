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
    GdnAppendSchedule, GdnPrefillPlan, GdnProjectionRecipe, GdnScanAlgorithm, GdnStateQuant,
    GdnStreamMeta, GdnStreamPhase, GdnStreamWorkspace, GdnTensorFormat, GDN_COMMIT_ROWS,
    GDN_STREAM_SCHEMA_VERSION,
};
use rdna_compute::tensor_ops::gdn_wy_stream::{
    wy_advance_meta, wy_stream_admitted, WY_TILE_ROWS,
};

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

/// Which canonical scan stream serves a state's ordinary prefill (resolved once from static
/// facts: architecture, flags, geometry, configured state quantization; never a row count).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnStreamKind {
    /// The persistent / sequential recurrence (no carry: the FP32 accumulator continues).
    None,
    /// gfx11 C16 WY chunk scan.
    Wy16,
    /// gfx1201 dense C64 chunk scan (`fn_gdn_dense`).
    Scan64,
}

/// The model weights the Scan64 seal replays an incomplete tile with (BF16 as stored).
#[derive(Clone, Copy)]
pub struct GdnSealWeights<'a> {
    pub conv: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
}

impl GdnEdge {
    /// Release the staging (best effort, like every teardown path).
    pub(crate) fn free(self, gpu: &mut Gpu) {
        match self {
            Self::Wy { z, out, norm } => {
                for tensor in [norm, out, z] {
                    let _ = gpu.free_tensor(tensor);
                }
            }
            Self::Scan64 { consts, ws } => {
                let _ = gpu.gdn_stream_workspace_free(ws);
                let _ = gpu.free_tensor(consts);
            }
        }
    }
}

impl GdnStreamKind {
    pub const fn algorithm(self) -> Option<GdnScanAlgorithm> {
        match self {
            Self::None => None,
            Self::Wy16 => Some(GdnScanAlgorithm::Wy16),
            Self::Scan64 => Some(GdnScanAlgorithm::Scan64),
        }
    }
}

/// Shared per-state staging of a carrying state (charged once per state, never copied by a
/// snapshot): the WY replay's output/Z rows, or the Scan64 executor's edge workspace and the
/// widened model constants staging.
pub enum GdnEdge {
    Wy {
        z: GpuTensor,
        out: GpuTensor,
        /// BF16 `[128]` gated-norm weight stand-in: the seal's replayed rows are
        /// discarded, so only a valid address is read.
        norm: GpuTensor,
    },
    Scan64 {
        /// F32 widened conv weight / a_log / dt_bias of the layer being run.
        consts: GpuTensor,
        ws: GdnStreamWorkspace,
    },
}

/// The numerical recipe of a Qwen4 stream of `kind` over `format`: algorithm, tile, seam,
/// operand/output storage and state quantization. The projection recipe is the producer
/// policy's (not resolved by the state): its fields stay zero until a producer plan fills
/// them, and are part of the identity.
pub(crate) fn gdn_stream_plan(kind: GdnStreamKind, format: GdnStateFormat) -> GdnPrefillPlan {
    let algorithm = kind.algorithm().expect("a carrying stream kind");
    let mut plan = GdnPrefillPlan {
        recipe_id: 0,
        algorithm,
        micro_rows: algorithm.micro_rows(),
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
            GdnStreamKind::Scan64 => GdnTensorFormat::F32,
            _ => GdnTensorFormat::Bf16,
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

/// Resolve the stream kind from static facts only (architecture, flags, geometry,
/// configured state quantization). Never reads a row count or free memory.
pub(crate) fn resolve_gdn_stream_kind(
    gpu: &Gpu,
    config: &Qwen4Config,
    format: GdnStateFormat,
) -> GdnStreamKind {
    let (hk, hv) = (config.linear_num_key_heads, config.linear_num_value_heads);
    let (kd, vd) = (config.linear_key_head_dim, config.linear_value_head_dim);
    if wy_stream_admitted(gpu, hk, hv, kd, vd) {
        GdnStreamKind::Wy16
    } else if rdna_compute::fn_gdn_dense::stream_admitted(gpu, format, hk, hv, kd, vd) {
        GdnStreamKind::Scan64
    } else {
        GdnStreamKind::None
    }
}

/// The stream recipe `kind` resolves for `format`, `None` without a carrying kind.
pub(crate) fn resolve_gdn_stream(kind: GdnStreamKind, format: GdnStateFormat) -> Option<GdnPrefillPlan> {
    kind.algorithm().map(|_| gdn_stream_plan(kind, format))
}

/// Allocation specs of one carrying GDN layer, in allocation order:
/// `(name, elements, dtype)`.
pub(crate) fn carry_layer_specs(
    config: &Qwen4Config,
    kind: GdnStreamKind,
) -> [(&'static str, usize, DType); 3] {
    let state = config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim;
    match kind {
        GdnStreamKind::Scan64 => {
            let (conv, params) = rdna_compute::fn_gdn_dense::stream_carry_elems(
                config.linear_num_key_heads,
                config.linear_num_value_heads,
            );
            [
                ("carry.tile_entry", state, DType::F32),
                ("carry.conv", conv, DType::F32),
                ("carry.params", params, DType::F32),
            ]
        }
        _ => {
            let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
                + config.linear_num_value_heads * config.linear_value_head_dim;
            [
                ("carry.tile_entry", state, DType::F32),
                ("carry.pending_conv", WY_TILE_ROWS * qkv, DType::BF16),
                (
                    "carry.pending_params",
                    2 * WY_TILE_ROWS * config.linear_num_value_heads,
                    DType::F32,
                ),
            ]
        }
    }
}

/// Allocation specs of the shared edge staging of `kind` (the Scan64 workspace is allocated
/// by [`Gpu::gdn_stream_workspace_alloc`], its bytes counted by [`scan64_workspace_bytes`]):
/// `(name, elements, dtype)`.
pub(crate) fn edge_specs(config: &Qwen4Config, kind: GdnStreamKind) -> Vec<(&'static str, usize, DType)> {
    match kind {
        GdnStreamKind::Scan64 => vec![(
            "gdn_edge.consts",
            rdna_compute::fn_gdn_dense::stream_consts_elems(
                config.linear_num_key_heads,
                config.linear_num_value_heads,
            ),
            DType::F32,
        )],
        _ => {
            let stage =
                WY_TILE_ROWS * config.linear_num_value_heads * config.linear_value_head_dim;
            vec![
                ("gdn_edge.z", stage, DType::F32),
                ("gdn_edge.out", stage, DType::F32),
                ("gdn_edge.norm", config.linear_value_head_dim, DType::BF16),
            ]
        }
    }
}

/// Device bytes of each plane of the Scan64 executor's edge workspace of `config`'s geometry
/// (BF16 scan output), in [`Gpu::gdn_stream_workspace_alloc`]'s allocation order.
pub(crate) fn scan64_workspace_plane_bytes(config: &Qwen4Config) -> [usize; 9] {
    rdna_compute::gdn_stream::stream_workspace_plane_bytes(
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        2,
    )
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
    match plan.algorithm {
        GdnScanAlgorithm::Scan64 => {
            // The Scan64 executor's schedule is the shared pure arithmetic over the cursor;
            // a Sealed entry opens at `entry_position`.
            let from = match entry.phase {
                GdnStreamPhase::Open => entry,
                GdnStreamPhase::Sealed => GdnStreamMeta {
                    recipe_id: plan.recipe_id,
                    ..GdnStreamMeta::sealed_at(entry_position, entry.owner_generation)
                },
            };
            GdnAppendSchedule::new(
                &from,
                entry_position,
                (position - entry_position) as usize,
                plan.micro_rows,
                plan.commit_rows.max(1),
            )
            .map_or(from, |schedule| GdnStreamMeta {
                recipe_id: plan.recipe_id,
                ..schedule.meta
            })
        }
        _ => wy_advance_meta(
            entry,
            plan.recipe_id,
            entry_position,
            position - entry_position,
            plan.state_quant != GdnStateQuant::F32,
        ),
    }
}
